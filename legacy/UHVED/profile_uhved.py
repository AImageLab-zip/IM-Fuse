import time
import traceback
from argparse import ArgumentParser
from pathlib import Path

import nibabel as nib
import torch
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity

from UHVED import U_HVED


# ---------------------------------------------------------------------
# 1. Arguments
# ---------------------------------------------------------------------
parser = ArgumentParser()
parser.add_argument("--subject-dir", type=Path, required=True)
parser.add_argument(
    "--mask-config",
    type=str,
    default="all_modalities",
    choices=[
        "all_modalities",
        "no_T1",
        "no_T1c",
        "no_T2",
        "no_Flair",
        "flair_only",
        "t1c_flair_only",
        "all",
    ],
    help="Modality mask config to profile. Use 'all' for a summary across all configs.",
)
parser.add_argument("--crop-size", type=int, nargs=3, default=(112, 112, 112))
parser.add_argument("--warmup", type=int, default=5)
parser.add_argument("--iters", type=int, default=20)
args = parser.parse_args()


# ---------------------------------------------------------------------
# 2. Wrapper
# ---------------------------------------------------------------------
class WrappedModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, flair, t1c, t1, t2, mask):
        images = {"Flair": flair, "T1c": t1c, "T1": t1, "T2": t2}
        return self.model(images, mask, is_inference=True)


# ---------------------------------------------------------------------
# 3. FLOP counter based on executed modules
# ---------------------------------------------------------------------
class FlopCounter:
    def __init__(self, model: nn.Module):
        self.model = model
        self.handles = []
        self.by_module = {}

    def clear(self):
        self.by_module = {}

    def total(self):
        return sum(self.by_module.values())

    def _add(self, module, flops: int):
        name = module.__class__.__name__
        self.by_module[name] = self.by_module.get(name, 0) + int(flops)

    def _conv3d_hook(self, module, inputs, output):
        if not isinstance(output, torch.Tensor):
            return

        batch = output.shape[0]
        cout = output.shape[1]
        dout, hout, wout = output.shape[2:]

        cin = module.in_channels
        kd, kh, kw = module.kernel_size
        groups = module.groups

        flops = (
            2
            * batch
            * cout
            * dout
            * hout
            * wout
            * (cin // groups)
            * kd
            * kh
            * kw
        )
        if module.bias is not None:
            flops += batch * cout * dout * hout * wout

        self._add(module, flops)

    def _convtranspose3d_hook(self, module, inputs, output):
        if not isinstance(output, torch.Tensor):
            return

        batch = output.shape[0]
        cout = output.shape[1]
        dout, hout, wout = output.shape[2:]

        cin = module.in_channels
        kd, kh, kw = module.kernel_size
        groups = module.groups

        flops = (
            2
            * batch
            * cout
            * dout
            * hout
            * wout
            * (cin // groups)
            * kd
            * kh
            * kw
        )
        if module.bias is not None:
            flops += batch * cout * dout * hout * wout

        self._add(module, flops)

    def _linear_hook(self, module, inputs, output):
        if not isinstance(output, torch.Tensor):
            return

        batch_elems = output.numel() // output.shape[-1]
        flops = 2 * batch_elems * module.in_features * module.out_features
        if module.bias is not None:
            flops += output.numel()

        self._add(module, flops)

    def _norm_hook(self, module, inputs, output):
        if not isinstance(output, torch.Tensor):
            return

        # rough normalization arithmetic estimate
        flops = 4 * output.numel()
        self._add(module, flops)

    def install(self):
        for m in self.model.modules():
            if isinstance(m, nn.Conv3d):
                self.handles.append(m.register_forward_hook(self._conv3d_hook))
            elif isinstance(m, nn.ConvTranspose3d):
                self.handles.append(m.register_forward_hook(self._convtranspose3d_hook))
            elif isinstance(m, nn.Linear):
                self.handles.append(m.register_forward_hook(self._linear_hook))
            elif isinstance(m, (nn.BatchNorm3d, nn.InstanceNorm3d)):
                self.handles.append(m.register_forward_hook(self._norm_hook))

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []


# ---------------------------------------------------------------------
# 4. Model
# ---------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"PyTorch version: {torch.__version__}")

model = U_HVED(num_classes=4).to(device).eval()
wrapped = WrappedModel(model).to(device).eval()


# ---------------------------------------------------------------------
# 5. Data
# ---------------------------------------------------------------------
sub = args.subject_dir
crop_size = tuple(args.crop_size)


def load_and_crop(path: Path, target_size):
    data = nib.load(str(path)).get_fdata()
    tensor = torch.from_numpy(data).float()

    if tensor.ndim != 3:
        raise ValueError(f"Expected 3D image, got shape {tuple(tensor.shape)} for {path}")

    d, h, w = tensor.shape
    td, th, tw = target_size

    if d < td or h < th or w < tw:
        raise ValueError(
            f"Crop size {target_size} larger than image shape {tuple(tensor.shape)} for {path}"
        )

    sd = (d - td) // 2
    sh = (h - th) // 2
    sw = (w - tw) // 2

    tensor = tensor[sd : sd + td, sh : sh + th, sw : sw + tw]
    return tensor.unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, D, H, W]


images = {}
try:
    images["Flair"] = load_and_crop(sub / f"{sub.name}-t2f.nii.gz", crop_size)
    images["T1c"] = load_and_crop(sub / f"{sub.name}-t1c.nii.gz", crop_size)
    images["T1"] = load_and_crop(sub / f"{sub.name}-t1n.nii.gz", crop_size)
    images["T2"] = load_and_crop(sub / f"{sub.name}-t2w.nii.gz", crop_size)
except Exception as e:
    print(f"Error loading files: {e}")
    traceback.print_exc()
    raise SystemExit(1)

print(f"Input shape: {tuple(images['Flair'].shape)}")


# ---------------------------------------------------------------------
# 6. Mask configs
# Order matches: ['T1', 'T1c', 'T2', 'Flair']
# ---------------------------------------------------------------------
mask_configs = {
    "all_modalities": [True, True, True, True],
    "no_T1": [False, True, True, True],
    "no_T1c": [True, False, True, True],
    "no_T2": [True, True, False, True],
    "no_Flair": [True, True, True, False],
    "flair_only": [False, False, False, True],
    "t1c_flair_only": [False, True, False, True],
}


def make_mask(mask_values):
    return torch.tensor(mask_values, dtype=torch.bool, device=device).unsqueeze(0)  # [1, 4]


# ---------------------------------------------------------------------
# 7. Helpers
# ---------------------------------------------------------------------
def get_cuda_time(evt):
    for attr in ("self_cuda_time_total", "cuda_time_total", "device_time_total"):
        if hasattr(evt, attr):
            return getattr(evt, attr)
    return 0


def benchmark_latency(model, inputs, warmup=5, iters=20):
    model.eval()

    with torch.no_grad():
        for _ in range(warmup):
            _ = model(*inputs)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        times_ms = []

        with torch.no_grad():
            for _ in range(iters):
                starter.record()
                _ = model(*inputs)
                ender.record()
                torch.cuda.synchronize()
                times_ms.append(starter.elapsed_time(ender))

        peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9
    else:
        times_ms = []
        peak_mem_gb = None

        with torch.no_grad():
            for _ in range(iters):
                t0 = time.perf_counter()
                _ = model(*inputs)
                t1 = time.perf_counter()
                times_ms.append((t1 - t0) * 1000.0)

    avg_ms = sum(times_ms) / len(times_ms)
    std_ms = (sum((x - avg_ms) ** 2 for x in times_ms) / len(times_ms)) ** 0.5
    return avg_ms, std_ms, peak_mem_gb


def run_profiler(model, inputs):
    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    with torch.no_grad():
        for _ in range(3):
            _ = model(*inputs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_flops=False,  # IMPORTANT: profiler FLOPs are not reliable here
    ) as prof:
        with record_function("model_inference"):
            with torch.no_grad():
                _ = model(*inputs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    return prof


def estimate_flops(model, inputs):
    counter = FlopCounter(model)
    counter.install()
    counter.clear()

    with torch.no_grad():
        _ = model(*inputs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    total_flops = counter.total()
    by_module = dict(sorted(counter.by_module.items(), key=lambda kv: kv[1], reverse=True))
    counter.remove()
    return total_flops, by_module


def summarize(config_name, flops, flops_by_module, avg_ms, std_ms, peak_mem_gb, prof):
    print(f"\n{'=' * 70}")
    print(f"  Config: {config_name}")
    print(f"{'=' * 70}")

    key_avgs = prof.key_averages()
    total_cpu = sum(evt.cpu_time_total for evt in key_avgs if hasattr(evt, "cpu_time_total"))
    total_cuda = sum(get_cuda_time(evt) for evt in key_avgs)

    print(f"  Estimated FLOPs      : {flops / 1e9:.3f} GFLOPs")
    print(f"  Avg latency          : {avg_ms:.2f} +/- {std_ms:.2f} ms")
    print(f"  Total CPU prof time  : {total_cpu / 1e3:.2f} ms")
    if torch.cuda.is_available():
        print(f"  Total CUDA prof time : {total_cuda / 1e3:.2f} ms")
        print(f"  Peak GPU memory      : {peak_mem_gb:.3f} GB")

    print("\n--- FLOPs by module type ---")
    for name, value in flops_by_module.items():
        print(f"{name:<20} {value / 1e9:>10.3f} GFLOPs")

    print("\n--- Top 30 ops by CPU time ---\n")
    print(key_avgs.table(sort_by="cpu_time_total", row_limit=30))

    if torch.cuda.is_available():
        print("\n--- Top 30 ops by CUDA time ---\n")
        for sort_key in ("self_cuda_time_total", "cuda_time_total", "device_time_total"):
            try:
                print(key_avgs.table(sort_by=sort_key, row_limit=30))
                break
            except Exception:
                continue


def run_one_config(config_name, mask_values, detailed=True):
    inputs = (
        images["Flair"],
        images["T1c"],
        images["T1"],
        images["T2"],
        make_mask(mask_values),
    )

    flops, flops_by_module = estimate_flops(wrapped, inputs)
    avg_ms, std_ms, peak_mem_gb = benchmark_latency(
        wrapped, inputs, warmup=args.warmup, iters=args.iters
    )

    prof = None
    if detailed:
        prof = run_profiler(wrapped, inputs)
        summarize(config_name, flops, flops_by_module, avg_ms, std_ms, peak_mem_gb, prof)

    return {
        "config": config_name,
        "flops": flops,
        "avg_ms": avg_ms,
        "std_ms": std_ms,
        "peak_mem_gb": peak_mem_gb,
    }


# ---------------------------------------------------------------------
# 8. Run
# ---------------------------------------------------------------------
if args.mask_config == "all":
    header = f"\n{'Mask Config':<25} | {'GFLOPs':<12} | {'Latency ms':<12}"
    if torch.cuda.is_available():
        header += f" | {'Peak GB':<10}"
    print(header)
    print("-" * len(header.strip()))

    results = []
    for config_name, mask_values in mask_configs.items():
        try:
            result = run_one_config(config_name, mask_values, detailed=False)
            results.append(result)

            row = (
                f"{result['config']:<25} | "
                f"{result['flops'] / 1e9:<12.3f} | "
                f"{result['avg_ms']:<12.2f}"
            )
            if torch.cuda.is_available():
                row += f" | {result['peak_mem_gb']:<10.3f}"
            print(row)

        except Exception as e:
            print(f"\nError on '{config_name}': {e}")
            traceback.print_exc()

    print("\nDetailed breakdown for 'all_modalities':")
    try:
        run_one_config("all_modalities", mask_configs["all_modalities"], detailed=True)
    except Exception as e:
        print(f"Error on detailed run: {e}")
        traceback.print_exc()

else:
    try:
        run_one_config(args.mask_config, mask_configs[args.mask_config], detailed=True)
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()

print("\nAnalysis complete.")