from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import importlib
import json
from math import ceil
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import Any, Callable

import torch
import torch.nn as nn
from torch.nn.parameter import UninitializedParameter
from rich.table import Table


REPO_ROOT = Path(__file__).resolve().parents[2]
LEGACY_ROOT = REPO_ROOT / "legacy"
DEFAULT_FULL_VOLUME_SHAPE = (240, 240, 155)


@dataclass(frozen=True)
class FlopsMeasurement:
    method: str
    source_entrypoint: str
    model_target: str
    measurement: str
    input_shape: tuple[int, ...] | None
    macs: float | None
    flops: float | None
    params: int | None
    status: str
    note: str = ""


@dataclass(frozen=True)
class FlopsReport:
    method: str
    source_entrypoint: str
    model_target: str
    measurements: tuple[FlopsMeasurement, ...]


@dataclass(frozen=True)
class FullVolumeStrategy:
    kind: str
    patch_shape: tuple[int, int, int] | None = None
    overlap: float = 0.5
    stride: tuple[int, int, int] | None = None


@dataclass(frozen=True)
class MethodSpec:
    name: str
    import_root: Path
    source_entrypoint: str
    model_target: str
    builder: Callable[[], nn.Module]
    adapter_kind: str
    forward_shape: tuple[int, int, int]
    full_volume_shape: tuple[int, int, int]
    full_volume_strategy: FullVolumeStrategy


@contextmanager
def _import_root(path: Path):
    sys.path.insert(0, str(path))
    try:
        yield
    finally:
        try:
            sys.path.remove(str(path))
        except ValueError:
            pass


@contextmanager
def _import_roots(paths: list[Path]):
    inserted = [str(path) for path in paths]
    for path in reversed(inserted):
        sys.path.insert(0, path)
    try:
        yield
    finally:
        for path in inserted:
            try:
                sys.path.remove(path)
            except ValueError:
                pass


def _extract_primary_tensor(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (list, tuple)):
        for item in output:
            try:
                return _extract_primary_tensor(item)
            except TypeError:
                continue
    if isinstance(output, dict):
        for item in output.values():
            try:
                return _extract_primary_tensor(item)
            except TypeError:
                continue
    raise TypeError(f"could not extract tensor from output of type {type(output)!r}")


def _mask_from_bool(batch_size: int, values: tuple[bool, bool, bool, bool], device: torch.device) -> torch.Tensor:
    return torch.tensor([values] * batch_size, dtype=torch.bool, device=device)


def _modality_bitmask(values: torch.Tensor) -> int:
    encoded = 0
    for index, value in enumerate(values.tolist()):
        if bool(value):
            encoded |= 1 << index
    return encoded


class _LegacyWrapper(nn.Module):
    def __init__(self, model: nn.Module, adapter_kind: str) -> None:
        super().__init__()
        self.model = model
        self.adapter_kind = adapter_kind

    def forward(self, images: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        flair, t1ce, t1, t2 = torch.chunk(images, 4, dim=1)

        if self.adapter_kind == "images_mask":
            result = self.model(images, mask)
        elif self.adapter_kind == "images_only":
            result = self.model(images)
        elif self.adapter_kind == "images_mode":
            result = self.model(images, val=True, mode="0,1,2,3")
        elif self.adapter_kind == "split_mask":
            result = self.model(flair, t1ce, t1, t2, mask[0])
        elif self.adapter_kind == "list_mask":
            # InOutFusion expects a scalar bitmask in the same format as its dataloader.
            modality_descriptor = _modality_bitmask(mask[0])
            result = self.model([flair, t1ce, t1, t2], modality_descriptor)
        elif self.adapter_kind == "list_mask_tensor":
            # SFusion expects a boolean mask tensor shaped like [batch, 4].
            result = self.model(images, mask)
        elif self.adapter_kind == "uhved":
            images_dict = {"Flair": flair, "T1c": t1ce, "T1": t1, "T2": t2}
            result = self.model(images_dict, mask, is_inference=True)
        elif self.adapter_kind == "mam":
            result = self.model(images, [True, True, True, True])
        elif self.adapter_kind == "m3ae":
            result = self.model(images)
        elif self.adapter_kind == "d2net":
            result = self.model(images, complete_x=None, is_test=True)
        else:
            raise ValueError(f"unsupported adapter kind: {self.adapter_kind}")

        return _extract_primary_tensor(result)


def _resolve_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("legacy/flops is CUDA-only and requires an available CUDA device")
    return torch.device("cuda")


def _first_tensor(value: Any) -> torch.Tensor | None:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
        return None
    if isinstance(value, dict):
        for item in value.values():
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
        return None
    return None


def _extract_tensor_shape(value: Any) -> tuple[int, ...] | None:
    tensor = _first_tensor(value)
    if tensor is None:
        return None
    return tuple(tensor.shape)


def _multihead_attention_macs(module: nn.MultiheadAttention, query: torch.Tensor) -> float:
    if query.ndim != 3:
        return 0.0
    if module.batch_first:
        batch_size, target_len, embed_dim = query.shape
    else:
        target_len, batch_size, embed_dim = query.shape
    source_len = target_len
    in_proj_macs = 3.0 * batch_size * target_len * embed_dim * embed_dim
    attn_macs = 2.0 * batch_size * target_len * source_len * embed_dim
    return in_proj_macs + attn_macs


def _local_attention_macs(module: nn.Module, q: torch.Tensor) -> float:
    if q.ndim != 4:
        return 0.0
    batch_size, heads, seq_len, head_dim = q.shape
    window_size = int(getattr(module, "window_size", seq_len))
    if window_size <= 0 or seq_len <= 0:
        return 0.0
    windows = max(seq_len // window_size, 1)
    look_backward = int(getattr(module, "look_backward", 1))
    look_forward = int(getattr(module, "look_forward", 0))
    context_len = (look_backward + look_forward + 1) * window_size
    return 2.0 * batch_size * heads * windows * window_size * context_len * head_dim


def _afno1d_macs(module: nn.Module, x: torch.Tensor) -> float:
    if x.ndim != 3:
        return 0.0
    batch_size, seq_len, hidden_size = x.shape
    num_blocks = int(getattr(module, "num_blocks", 1))
    hidden_size_factor = int(getattr(module, "hidden_size_factor", 1))
    if num_blocks <= 0 or hidden_size <= 0 or seq_len <= 0:
        return 0.0
    block_size = hidden_size // num_blocks
    freq_len = (seq_len // 2) + 1
    fft_macs = 10.0 * batch_size * hidden_size * seq_len * max(seq_len.bit_length() - 1, 1)
    einsum1 = 4.0 * batch_size * num_blocks * freq_len * block_size * (block_size * hidden_size_factor)
    einsum2 = 4.0 * batch_size * num_blocks * freq_len * (block_size * hidden_size_factor) * block_size
    complex_mul = 3.0 * batch_size * hidden_size * freq_len
    return fft_macs + einsum1 + einsum2 + complex_mul


def _attention_base_macs(module: nn.Module, x: torch.Tensor) -> float:
    if x.ndim != 5:
        return 0.0
    batch_size, channels, depth, height, width = x.shape
    num_heads = int(getattr(module, "num_heads", 1))
    if num_heads <= 0 or channels <= 0:
        return 0.0
    tokens = depth * height * width
    channels_per_head = channels // num_heads
    return 2.0 * batch_size * num_heads * channels_per_head * channels_per_head * tokens


def _estimate_macs_from_executed_layers(
    wrapper: nn.Module,
    images: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[float, str | None]:
    total_macs = 0.0
    hook_handles = []

    def conv_flops_counter_hook(module: nn.Module, input: tuple[Any, ...], output: Any) -> None:
        nonlocal total_macs
        output_tensor = _first_tensor(output)
        if output_tensor is None:
            return

        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            kernel_macs_per_output = module.weight.numel() / max(module.out_channels, 1)
            total_macs += float(output_tensor.numel()) * kernel_macs_per_output
            return

        if isinstance(module, nn.Linear):
            total_macs += float(output_tensor.numel()) * float(module.in_features)
            return

        if isinstance(module, nn.MultiheadAttention):
            query = _first_tensor(input[0]) if input else None
            if query is not None:
                total_macs += _multihead_attention_macs(module, query)
            return

        name = module.__class__.__name__
        if name == "LocalAttention":
            query = _first_tensor(input[0]) if input else None
            if query is not None:
                total_macs += _local_attention_macs(module, query)
            return

        if name == "AFNO1D_channelfirst":
            x = _first_tensor(input[0]) if input else None
            if x is not None:
                total_macs += _afno1d_macs(module, x)
            return

        if name == "AttentionBase":
            x = _first_tensor(input[0]) if input else None
            if x is not None:
                total_macs += _attention_base_macs(module, x)

    try:
        for m in wrapper.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear, nn.MultiheadAttention)):
                hook_handles.append(m.register_forward_hook(conv_flops_counter_hook))
            elif m.__class__.__name__ in {"LocalAttention", "AFNO1D_channelfirst", "AttentionBase"}:
                hook_handles.append(m.register_forward_hook(conv_flops_counter_hook))

        with torch.no_grad():
            wrapper(images, mask)
        return float(total_macs), None
    except Exception as exc:
        if total_macs > 0:
            return float(total_macs), f"analytical fallback was partial: {type(exc).__name__}"
        return 0.0, f"analytical fallback failed: {type(exc).__name__}"
    finally:
        for handle in hook_handles:
            try:
                handle.remove()
            except Exception:
                pass


def _estimate_macs_with_flopcount(wrapper: nn.Module, images: torch.Tensor, mask: torch.Tensor) -> float:
    try:
        from flopcount import get_flops
        with torch.no_grad():
            flops = get_flops(wrapper, (images, mask))
        if flops is not None and flops > 0:
            return float(flops) / 2.0
    except Exception:
        pass
    return 0.0


def _measure_macs(
    model: nn.Module,
    adapter_kind: str,
    images: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[float, str]:
    wrapper = _LegacyWrapper(model, adapter_kind=adapter_kind).eval()

    failures: list[str] = []

    try:
        from ptflops import get_model_complexity_info
    except ImportError as exc:
        failures.append(f"ptflops import failed: {exc}")
    else:
        for backend in ("aten", "pytorch"):
            try:
                with torch.no_grad():
                    # Create a wrapper that stores mask in closure and accepts just images
                    class SingleArgWrapper(nn.Module):
                        def __init__(self, wrapped_model: nn.Module, mask_tensor: torch.Tensor) -> None:
                            super().__init__()
                            self.wrapped_model = wrapped_model
                            self.mask_tensor = mask_tensor
                        
                        def forward(self, img: torch.Tensor) -> torch.Tensor:
                            # Ensure mask has same batch size as images
                            mask_batch = self.mask_tensor
                            if mask_batch.shape[0] != img.shape[0]:
                                mask_batch = self.mask_tensor.repeat(img.shape[0], 1)
                            return self.wrapped_model(img, mask_batch)
                    
                    single_arg_wrapper = SingleArgWrapper(wrapper, mask)
                    macs, _ = get_model_complexity_info(
                        single_arg_wrapper,
                        tuple(images.shape[1:]),
                        input_constructor=lambda shape: images,
                        as_strings=False,
                        print_per_layer_stat=False,
                        verbose=False,
                        backend=backend,
                    )
                if macs is None or macs == 0:
                    failures.append(f"ptflops/{backend}: no MAC count returned")
                    continue
                return float(macs), ""
            except Exception as exc:
                failures.append(f"ptflops/{backend}: {type(exc).__name__}")

    try:
        from thop import profile
    except ImportError as exc:
        failures.append(f"thop import failed: {exc}")
    else:
        try:
            with torch.no_grad():
                macs, _ = profile(wrapper, inputs=(images, mask), verbose=False)
            if macs is None or macs == 0:
                failures.append("thop: no MAC count returned")
            else:
                return float(macs), "estimated with thop hooks"
        except Exception as exc:
            failures.append(f"thop: {type(exc).__name__}")

    try:
        from torch.profiler import ProfilerActivity, profile as torch_profile
    except ImportError as exc:
        failures.append(f"torch.profiler import failed: {exc}")
    else:
        try:
            activities = [ProfilerActivity.CPU]
            if images.is_cuda:
                activities.append(ProfilerActivity.CUDA)

            with torch_profile(
                activities=activities,
                record_shapes=False,
                profile_memory=False,
                with_flops=True,
            ) as prof:
                with torch.no_grad():
                    wrapper(images, mask)

            total_flops = sum(getattr(event, "flops", 0) or 0 for event in prof.key_averages())
            if total_flops > 0:
                return float(total_flops) / 2.0, "estimated with torch.profiler operator FLOPs"
            failures.append("torch.profiler: no FLOPs recorded")
        except Exception as exc:
            failures.append(f"torch.profiler: {type(exc).__name__}")

    flopcount_macs = _estimate_macs_with_flopcount(wrapper, images, mask)
    if flopcount_macs > 0:
        return flopcount_macs, "estimated with flopcount"

    estimated_macs, analytical_note = _estimate_macs_from_executed_layers(wrapper, images, mask)
    if estimated_macs > 0:
        return estimated_macs, _join_notes(
            "estimated from executed layers and analytical attention formulas",
            analytical_note or "",
        )
    if analytical_note is not None:
        failures.append(analytical_note)

    raise RuntimeError(
        "no supported profiler could produce a MAC count for this model. "
        + "; ".join(failures)
    )



def _count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _has_uninitialized_parameters(model: nn.Module) -> bool:
    return any(isinstance(parameter, UninitializedParameter) for parameter in model.parameters())


def _initialize_lazy_parameters(
    model: nn.Module,
    adapter_kind: str,
    batch_size: int,
    spatial_shape: tuple[int, int, int],
    device: torch.device,
) -> None:
    if not _has_uninitialized_parameters(model):
        return
    wrapper = _LegacyWrapper(model, adapter_kind=adapter_kind).eval()
    images = _build_images(batch_size, spatial_shape, device)
    mask = _mask_from_bool(batch_size, (True, True, True, True), device)
    with torch.no_grad():
        wrapper(images, mask)


def _build_images(
    batch_size: int,
    spatial_shape: tuple[int, int, int],
    device: torch.device,
) -> torch.Tensor:
    return torch.randn(batch_size, 4, *spatial_shape, device=device, dtype=torch.float32)


def _sliding_window_count(
    full_shape: tuple[int, int, int],
    patch_shape: tuple[int, int, int],
    overlap: float,
) -> int:
    count = 1
    for full_dim, patch_dim in zip(full_shape, patch_shape, strict=True):
        if full_dim <= patch_dim:
            axis_count = 1
        else:
            step = patch_dim * (1.0 - overlap)
            axis_count = int(ceil((full_dim - patch_dim) / step) + 1)
        count *= axis_count
    return count


def _strided_window_count(
    full_shape: tuple[int, int, int],
    patch_shape: tuple[int, int, int],
    stride: tuple[int, int, int],
) -> int:
    count = 1
    for full_dim, patch_dim, step in zip(full_shape, patch_shape, stride, strict=True):
        axis_count = len(range(0, max(full_dim - patch_dim, 0) + 1, step))
        count *= max(axis_count, 1)
    return count


def _format_shape(shape: tuple[int, ...] | None) -> str:
    if shape is None:
        return "-"
    return "[" + ", ".join(str(v) for v in shape) + "]"


def _format_count(value: float | None) -> str:
    if value is None:
        return "-"
    for factor, suffix in ((1e12, "T"), (1e9, "G"), (1e6, "M"), (1e3, "K")):
        if abs(value) >= factor:
            return f"{value / factor:.3f}{suffix}"
    return f"{value:.0f}"


def build_flops_table(report: FlopsReport) -> Table:
    table = Table(
        title=f"Legacy FLOPs for {report.method} ({report.model_target})"
    )
    table.add_column("Measurement", style="bold cyan")
    table.add_column("Input Shape")
    table.add_column("MACs", justify="right")
    table.add_column("FLOPs", justify="right")
    table.add_column("Params", justify="right")
    table.add_column("Status")
    table.add_column("Note")
    for measurement in report.measurements:
        table.add_row(
            measurement.measurement,
            _format_shape(measurement.input_shape),
            _format_count(measurement.macs),
            _format_count(measurement.flops),
            _format_count(float(measurement.params) if measurement.params is not None else None),
            measurement.status,
            measurement.note,
        )
    return table


def report_to_dict(report: FlopsReport) -> dict[str, Any]:
    return {
        "method": report.method,
        "source_entrypoint": report.source_entrypoint,
        "model_target": report.model_target,
        "measurements": [asdict(m) for m in report.measurements],
    }


def _join_notes(*notes: str) -> str:
    return "; ".join(note for note in notes if note)


def _module_attr(module_name: str, attr_name: str, root: Path) -> Any:
    with _import_root(root):
        module = importlib.import_module(module_name)
        return getattr(module, attr_name)


def _construct_attr(module_name: str, attr_name: str, root: Path, **kwargs: Any) -> Any:
    with _import_root(root):
        module = importlib.import_module(module_name)
        model_class = getattr(module, attr_name)
        return model_class(**kwargs)


def _build_d2net_model() -> nn.Module:
    root = LEGACY_ROOT / "D2Net"
    with _import_root(root):
        import models

        args = SimpleNamespace(
            style_dim=16,
            train_transforms="Compose([ RandCrop3D((128,128,128)), RandomFlip(0), NumpyType((np.float32, np.int64)), ])",
            AuxDec_dim=2,
            use_style_map=True,
            use_kd=True,
            fea_dim=8,
            miss_modal=True,
            dataset="BraTSDataset",
            use_freq_map=False,
            use_freq_channel=False,
            use_freq_contrast=True,
            use_distill=True,
            use_contrast=True,
            affinity_kd=True,
            self_distill=False,
            kd_channel_attn=False,
            kd_dense_fea_attn=False,
        )
        return models.DisenNet(
            inChans_list=[4],
            base_outChans=8,
            num_class_list=[4],
            args=args,
        )


def _build_lckd_model() -> nn.Module:
    return _construct_attr(
        "DualNet",
        "DualNet",
        LEGACY_ROOT / "LCKD",
        norm_cfg="IN",
        activation_cfg="LeakyReLU",
        weight_std=True,
        num_classes=3,
        self_att=False,
        cross_att=False,
    )


def _build_shaspec_model() -> nn.Module:
    args = SimpleNamespace(
        num_classes=3,
        weight_std=True,
        input_size="80,160,160",
        mode="0,1,2,3",
    )
    return _construct_attr(
        "DualNet_SS",
        "DualNet_SS",
        LEGACY_ROOT / "ShaSpec",
        args=args,
        norm_cfg="IN",
        activation_cfg="LeakyReLU",
        num_classes=3,
        weight_std=True,
        self_att=True,
        cross_att=False,
    )


def _build_mstkd_model() -> nn.Module:
    build = _module_attr("models", "build_MSTKDNet", LEGACY_ROOT / "MST-KDNet")
    _, model_missing = build(inp_dim1=4, inp_dim2=4)
    return model_missing


def _build_mam_model() -> nn.Module:
    root = LEGACY_ROOT / "MaM" / "nnunetv2" / "utilities" / "my_dynamic_network_architectures" / "architectures"
    with _import_root(root):
        import torch.nn as nn
        from unet import MultimodalRecon

        return MultimodalRecon(
            4,
            6,
            (32, 64, 124, 256, 512, 512),
            nn.Conv3d,
            3,
            (1, 2, 2, 2, 2, 2),
            (2, 2, 2, 2, 2, 2),
            4,
            (2, 2, 2, 2, 2),
            conv_bias=False,
            norm_op=nn.InstanceNorm3d,
            norm_op_kwargs={"eps": 1e-5, "affine": True},
            dropout_op=None,
            dropout_op_kwargs=None,
            nonlin=nn.LeakyReLU,
            nonlin_kwargs={"inplace": True},
            deep_supervision=False,
            nonlin_first=False,
        )


def _build_rehydil_model() -> nn.Module:
    model_class = _module_attr("test_utils", "CPH_3d", LEGACY_ROOT / "ReHyDIL")
    return model_class(batch_size=31)


def _build_srmnet_model() -> nn.Module:
    root = LEGACY_ROOT / "SRMNet"
    dcn_root = root / "dcn"
    d3d_build = root / "dcn" / "build"
    compiled_d3d = sorted(d3d_build.rglob("D3D*.so"))
    try:
        with _import_roots([root, dcn_root]):
            module = importlib.import_module("model.net")
            model_class = getattr(module, "Model")
            return model_class(num_cls=4)
    except ModuleNotFoundError as exc:
        if exc.name == "D3D" and compiled_d3d:
            available = ", ".join(path.name for path in compiled_d3d)
            raise RuntimeError(
                "SRMNet requires the local D3D extension, but Python could not import it from "
                f"{dcn_root}. Found builds: {available}. Ensure the Python 3.12 build exists in "
                "legacy/SRMNet/dcn and that the extension dependencies are loadable."
            ) from exc
        raise


def _simple_builder(root_rel: str, module_name: str, attr_name: str, **kwargs: Any) -> Callable[[], nn.Module]:
    root = LEGACY_ROOT / root_rel

    def build() -> nn.Module:
        return _construct_attr(module_name, attr_name, root, **kwargs)

    return build


def _registry() -> dict[str, MethodSpec]:
    return {
        "D2Net": MethodSpec(
            name="D2Net",
            import_root=LEGACY_ROOT / "D2Net",
            source_entrypoint="legacy/D2Net/test.py",
            model_target="models.DisenNet",
            builder=_build_d2net_model,
            adapter_kind="d2net",
            forward_shape=(128, 128, 128),
            full_volume_shape=DEFAULT_FULL_VOLUME_SHAPE,
            full_volume_strategy=FullVolumeStrategy("patched", patch_shape=(128, 128, 128), overlap=0.5),
        ),
        "DC-Seg": MethodSpec("DC-Seg", LEGACY_ROOT / "DC-Seg", "legacy/DC-Seg/test.py", "models.DC_Seg", _simple_builder("DC-Seg", "models", "DC_Seg", num_cls=4, fusion_type="RFM"), "images_mask", (112, 112, 112), DEFAULT_FULL_VOLUME_SHAPE, FullVolumeStrategy("patched", (112, 112, 112), 0.5)),
        "IMFuse": MethodSpec("IMFuse", LEGACY_ROOT / "IMFuse", "legacy/IMFuse/test.py", "IMFuse.IMFuse", _simple_builder("IMFuse", "IMFuse", "IMFuse", num_cls=4, interleaved_tokenization=False, mamba_skip=False), "images_mask", (128, 128, 128), DEFAULT_FULL_VOLUME_SHAPE, FullVolumeStrategy("patched", (128, 128, 128), 0.5)),
        "IMS2Trans": MethodSpec("IMS2Trans", LEGACY_ROOT / "IMS2Trans", "legacy/IMS2Trans/test.py", "ims2trans.Model", _simple_builder("IMS2Trans", "ims2trans", "Model", num_cls=4, use_checkpoint=False), "images_mask", (128, 128, 128), DEFAULT_FULL_VOLUME_SHAPE, FullVolumeStrategy("patched", (128, 128, 128), 0.5)),
        "InOutFusion": MethodSpec("InOutFusion", LEGACY_ROOT / "InOutFusion", "legacy/InOutFusion/test.py", "net.Network_InOut.RsInOut_U_Hemis3D", _simple_builder("InOutFusion", "net.Network_InOut", "RsInOut_U_Hemis3D", in_channels=1, out_channels=4, levels=4, feature_maps=8, method="TF", phase="test"), "list_mask", (128, 128, 128), DEFAULT_FULL_VOLUME_SHAPE, FullVolumeStrategy("patched", (128, 128, 128), 0.5)),
        "LCKD": MethodSpec("LCKD", LEGACY_ROOT / "LCKD", "legacy/LCKD/test.py", "DualNet.DualNet", _build_lckd_model, "images_mode", (80, 160, 160), DEFAULT_FULL_VOLUME_SHAPE, FullVolumeStrategy("patched", (80, 160, 160), overlap=1.0 / 3.0)),
        "M2FTrans": MethodSpec("M2FTrans", LEGACY_ROOT / "M2FTrans" / "M2FTrans_v1", "legacy/M2FTrans/M2FTrans_v1/test.py", "models.fusiontrans.Model", _simple_builder("M2FTrans/M2FTrans_v1", "models.fusiontrans", "Model", num_cls=4), "images_mask", (80, 80, 80), DEFAULT_FULL_VOLUME_SHAPE, FullVolumeStrategy("patched", (80, 80, 80), 0.5)),
        "MIFPN": MethodSpec("MIFPN", LEGACY_ROOT / "MIFPN", "legacy/MIFPN/test.py", "models.PNT.Model", _simple_builder("MIFPN", "models.PNT", "Model", num_cls=4), "images_mask", (80, 80, 80), DEFAULT_FULL_VOLUME_SHAPE, FullVolumeStrategy("patched", (80, 80, 80), 0.5)),
        "MMMViT": MethodSpec("MMMViT", LEGACY_ROOT / "MMMViT", "legacy/MMMViT/test.py", "mmmvit.Model", _simple_builder("MMMViT", "mmmvit", "Model", num_cls=4), "images_mask", (128, 128, 128), DEFAULT_FULL_VOLUME_SHAPE, FullVolumeStrategy("patched", (128, 128, 128), 0.5)),
        "MST-KDNet": MethodSpec("MST-KDNet", LEGACY_ROOT / "MST-KDNet", "legacy/MST-KDNet/eval.py", "models.build_MSTKDNet()[1]", _build_mstkd_model, "images_only", (160, 192, 128), (160, 192, 128), FullVolumeStrategy("direct")),
        "MaM": MethodSpec("MaM", LEGACY_ROOT / "MaM", "legacy/MaM/test.py", "unet.MultimodalRecon", _build_mam_model, "mam", (128, 128, 128), DEFAULT_FULL_VOLUME_SHAPE, FullVolumeStrategy("patched", (128, 128, 128), 0.5)),
        "RFNet": MethodSpec("RFNet", LEGACY_ROOT / "RFNet", "legacy/RFNet/test.py", "models.Model", _simple_builder("RFNet", "models", "Model", num_cls=4), "images_mask", (80, 80, 80), DEFAULT_FULL_VOLUME_SHAPE, FullVolumeStrategy("patched", (80, 80, 80), 0.5)),
        "ReHyDIL": MethodSpec("ReHyDIL", LEGACY_ROOT / "ReHyDIL", "legacy/ReHyDIL/test.py", "test_utils.CPH_3d", _build_rehydil_model, "images_only", (224, 224, 155), (224, 224, 155), FullVolumeStrategy("direct")),
        "RobustSeg": MethodSpec("RobustSeg", LEGACY_ROOT / "RobustSeg", "legacy/RobustSeg/test_robustseg.py", "RobustSeg.RobustSeg", _simple_builder("RobustSeg", "RobustSeg", "RobustSeg", num_cls=4), "images_mask", (80, 80, 80), DEFAULT_FULL_VOLUME_SHAPE, FullVolumeStrategy("patched", (80, 80, 80), 0.5)),
        "SFusion": MethodSpec("SFusion", LEGACY_ROOT / "SFusion", "legacy/SFusion/test_sfusion.py", "SFusion.TF_RMBTS", _simple_builder("SFusion", "SFusion", "TF_RMBTS", in_channels=1, out_channels=4, levels=4, feature_maps=16), "list_mask_tensor", (128, 128, 128), DEFAULT_FULL_VOLUME_SHAPE, FullVolumeStrategy("patched", (128, 128, 128), 0.5)),
        "SRMNet": MethodSpec("SRMNet", LEGACY_ROOT / "SRMNet", "legacy/SRMNet/test.py", "model.net.Model", _build_srmnet_model, "images_mask", (128, 128, 128), DEFAULT_FULL_VOLUME_SHAPE, FullVolumeStrategy("patched", (128, 128, 128), 0.5)),
        "ShaSpec": MethodSpec("ShaSpec", LEGACY_ROOT / "ShaSpec", "legacy/ShaSpec/eval.py", "DualNet_SS.DualNet_SS", _build_shaspec_model, "images_mode", (80, 160, 160), DEFAULT_FULL_VOLUME_SHAPE, FullVolumeStrategy("patched", (80, 160, 160), overlap=1.0 / 3.0)),
        "UHVED": MethodSpec("UHVED", LEGACY_ROOT / "UHVED", "legacy/UHVED/test_uhved.py", "UHVED.U_HVED", _simple_builder("UHVED", "UHVED", "U_HVED", num_classes=4), "uhved", (112, 112, 112), DEFAULT_FULL_VOLUME_SHAPE, FullVolumeStrategy("patched", (112, 112, 112), 0.5)),
        "UNET-MFI": MethodSpec("UNET-MFI", LEGACY_ROOT / "UNET-MFI", "legacy/UNET-MFI/test.py", "Model.no_share_unet", _simple_builder("UNET-MFI", "Model", "no_share_unet", in_channel=1, out_channel=3, diff=True, deepSupvision=True), "split_mask", (120, 120, 120), (240, 240, 160), FullVolumeStrategy("patched", (120, 120, 120), stride=(40, 40, 40))),
        "m3ae": MethodSpec("m3ae", LEGACY_ROOT / "m3ae", "legacy/m3ae/test.py", "model.Unet.Unet_missing", _simple_builder("m3ae", "model.Unet", "Unet_missing", input_shape=[128, 128, 128], out_channels=3, mdp=3, init_channels=16, pre_train=False, mask_modal=[], patch_shape=128), "m3ae", (128, 128, 128), DEFAULT_FULL_VOLUME_SHAPE, FullVolumeStrategy("patched", (128, 128, 128), 0.5)),
        "mmFormer": MethodSpec("mmFormer", LEGACY_ROOT / "mmFormer" / "mmformer", "legacy/mmFormer/mmformer/test.py", "mmformer.Model", _simple_builder("mmFormer/mmformer", "mmformer", "Model", num_cls=4), "images_mask", (128, 128, 128), DEFAULT_FULL_VOLUME_SHAPE, FullVolumeStrategy("patched", (128, 128, 128), 0.5)),
        "reverse": MethodSpec("reverse", LEGACY_ROOT / "reverse", "legacy/reverse/test.py", "reverse.Model", _simple_builder("reverse", "reverse", "Model", num_cls=4), "images_mask", (128, 128, 128), DEFAULT_FULL_VOLUME_SHAPE, FullVolumeStrategy("patched", (128, 128, 128), 0.5)),
    }


def method_names() -> tuple[str, ...]:
    return tuple(sorted(_registry()))


def get_method_spec(name: str) -> MethodSpec:
    try:
        return _registry()[name]
    except KeyError as exc:
        raise KeyError(f"unknown legacy FLOPs method: {name}") from exc


def run_flops_for_method(
    method: str,
    *,
    batch_size: int = 1,
    measure: str = "all",
    shape: tuple[int, int, int] | None = None,
    full_volume_shape: tuple[int, int, int] | None = None,
) -> FlopsReport:
    device = _resolve_device()
    spec = get_method_spec(method)
    model_target = spec.model_target
    source_entrypoint = spec.source_entrypoint
    patch_shape = shape or spec.forward_shape
    volume_shape = full_volume_shape or spec.full_volume_shape

    try:
        model = spec.builder().to(device).eval()
        _initialize_lazy_parameters(
            model,
            spec.adapter_kind,
            batch_size,
            patch_shape,
            device,
        )
        params = _count_parameters(model)
    except Exception as exc:
        failed = FlopsMeasurement(
            method=spec.name,
            source_entrypoint=source_entrypoint,
            model_target=model_target,
            measurement="initialization",
            input_shape=None,
            macs=None,
            flops=None,
            params=None,
            status="failed",
            note=f"initialization failed: {exc}",
        )
        return FlopsReport(spec.name, source_entrypoint, model_target, (failed,))

    mask = _mask_from_bool(batch_size, (True, True, True, True), device)
    measurements: list[FlopsMeasurement] = []
    patch_macs: float | None = None

    if measure in {"forward", "all"}:
        patch_images = _build_images(batch_size, patch_shape, device)
        try:
            patch_macs, patch_note = _measure_macs(model, spec.adapter_kind, patch_images, mask)
            measurements.append(
                FlopsMeasurement(
                    method=spec.name,
                    source_entrypoint=source_entrypoint,
                    model_target=model_target,
                    measurement="forward",
                    input_shape=tuple(patch_images.shape),
                    macs=patch_macs,
                    flops=patch_macs * 2,
                    params=params,
                    status="ok",
                    note=patch_note,
                )
            )
        except Exception as exc:
            measurements.append(
                FlopsMeasurement(
                    method=spec.name,
                    source_entrypoint=source_entrypoint,
                    model_target=model_target,
                    measurement="forward",
                    input_shape=(batch_size, 4, *patch_shape),
                    macs=None,
                    flops=None,
                    params=params,
                    status="failed",
                    note=str(exc),
                )
            )

    if measure in {"full_volume_forward", "all"}:
        strategy = spec.full_volume_strategy
        if strategy.kind == "patched":
            if patch_macs is None:
                patch_images = _build_images(batch_size, strategy.patch_shape or patch_shape, device)
                try:
                    patch_macs, patch_note = _measure_macs(model, spec.adapter_kind, patch_images, mask)
                except Exception as exc:
                    measurements.append(
                        FlopsMeasurement(
                            method=spec.name,
                            source_entrypoint=source_entrypoint,
                            model_target=model_target,
                            measurement="full_volume_forward",
                            input_shape=(batch_size, 4, *volume_shape),
                            macs=None,
                            flops=None,
                            params=params,
                            status="failed",
                            note=f"could not derive patched cost: {exc}",
                        )
                    )
                    return FlopsReport(spec.name, source_entrypoint, model_target, tuple(measurements))

            if strategy.stride is not None:
                windows = _strided_window_count(volume_shape, strategy.patch_shape or patch_shape, strategy.stride)
            else:
                windows = _sliding_window_count(volume_shape, strategy.patch_shape or patch_shape, strategy.overlap)
            total_macs = patch_macs * windows
            measurements.append(
                FlopsMeasurement(
                    method=spec.name,
                    source_entrypoint=source_entrypoint,
                    model_target=model_target,
                    measurement="full_volume_forward",
                    input_shape=(batch_size, 4, *volume_shape),
                    macs=total_macs,
                    flops=total_macs * 2,
                    params=params,
                    status="ok",
                    note=_join_notes(
                        f"patched full-volume cost across {windows} windows",
                        patch_note,
                    ),
                )
            )
        else:
            volume_images = _build_images(batch_size, volume_shape, device)
            try:
                volume_macs, volume_note = _measure_macs(model, spec.adapter_kind, volume_images, mask)
                measurements.append(
                    FlopsMeasurement(
                        method=spec.name,
                        source_entrypoint=source_entrypoint,
                        model_target=model_target,
                        measurement="full_volume_forward",
                        input_shape=tuple(volume_images.shape),
                        macs=volume_macs,
                        flops=volume_macs * 2,
                        params=params,
                        status="ok",
                        note=volume_note,
                    )
                )
            except Exception as exc:
                measurements.append(
                    FlopsMeasurement(
                        method=spec.name,
                        source_entrypoint=source_entrypoint,
                        model_target=model_target,
                        measurement="full_volume_forward",
                        input_shape=(batch_size, 4, *volume_shape),
                        macs=None,
                        flops=None,
                        params=params,
                        status="failed",
                        note=str(exc),
                    )
                )

    return FlopsReport(spec.name, source_entrypoint, model_target, tuple(measurements))


def run_flops_subprocess(
    method: str,
    *,
    batch_size: int,
    measure: str,
    shape: tuple[int, int, int] | None,
    full_volume_shape: tuple[int, int, int] | None,
) -> dict[str, Any]:
    cmd = [
        sys.executable,
        "-m",
        "legacy.flops",
        "run",
        "--method",
        method,
        "--batch-size",
        str(batch_size),
        "--measure",
        measure,
        "--json",
        "-",
    ]
    if shape is not None:
        cmd.extend(["--shape", *[str(v) for v in shape]])
    if full_volume_shape is not None:
        cmd.extend(["--full-volume-shape", *[str(v) for v in full_volume_shape]])
    completed = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        return {
            "method": method,
            "source_entrypoint": "",
            "model_target": "",
            "measurements": [{
                "method": method,
                "source_entrypoint": "",
                "model_target": "",
                "measurement": "subprocess",
                "input_shape": None,
                "macs": None,
                "flops": None,
                "params": None,
                "status": "failed",
                "note": completed.stderr.strip() or completed.stdout.strip() or f"subprocess failed with code {completed.returncode}",
            }],
        }
    return json.loads(completed.stdout)
