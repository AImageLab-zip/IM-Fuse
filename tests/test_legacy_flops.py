from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch

from legacy.flops import __main__ as flops_main
from legacy.flops import core as flops_core
from legacy.flops.core import (
    DEFAULT_FULL_VOLUME_SHAPE,
    DEFAULT_LEGACY_NONEMPTY_CROP_SHAPE,
    FlopsMeasurement,
    FlopsReport,
    FullVolumeStrategy,
    MethodSpec,
    _LegacyWrapper,
    _estimate_legacy_nonempty_crop_shape,
    _sliding_window_count,
    _strided_window_count,
    build_flops_table,
    get_method_spec,
    method_names,
    report_to_dict,
)


def test_registry_covers_expected_legacy_methods() -> None:
    expected = {
        "D2Net",
        "DC-Seg",
        "IMFuse",
        "IMS2Trans",
        "InOutFusion",
        "LCKD",
        "M2FTrans",
        "MIFPN",
        "MMMViT",
        "MST-KDNet",
        "M3FeCon",
        "RFNet",
        "ReHyDIL",
        "RobustSeg",
        "SFusion",
        "SRMNet",
        "ShaSpec",
        "UHVED",
        "UNET-MFI",
        "m3ae",
        "mmFormer",
        "rfl",
    }
    assert expected.issubset(set(method_names()))


def test_specs_use_legacy_entrypoints() -> None:
    for method in method_names():
        spec = get_method_spec(method)
        assert spec.source_entrypoint.startswith("legacy/")
        assert spec.import_root.exists()
    shaspec = get_method_spec("ShaSpec")
    assert shaspec.forward_shape == (80, 160, 160)
    imfuse = get_method_spec("IMFuse")
    assert imfuse.full_volume_shape == DEFAULT_FULL_VOLUME_SHAPE
    assert imfuse.full_volume_strategy.kind == "legacy_non_empty_patched"


def test_estimate_legacy_nonempty_crop_shape_matches_reference_default() -> None:
    assert _estimate_legacy_nonempty_crop_shape(DEFAULT_FULL_VOLUME_SHAPE) == (
        DEFAULT_LEGACY_NONEMPTY_CROP_SHAPE
    )


def test_estimate_legacy_nonempty_crop_shape_respects_minimum_extent() -> None:
    cropped = _estimate_legacy_nonempty_crop_shape((160, 160, 130))
    assert cropped == (128, 128, 128)


def test_sliding_window_count_matches_half_overlap_pattern() -> None:
    assert _sliding_window_count((240, 240, 155), (128, 128, 128), 0.5) == 18
    assert _sliding_window_count(DEFAULT_FULL_VOLUME_SHAPE, (80, 80, 80), 0.5) > 8


def test_strided_window_count_matches_unet_mfi_grid() -> None:
    assert _strided_window_count((240, 240, 160), (120, 120, 120), (40, 40, 40)) == 32


class _ImagesMaskModel(torch.nn.Module):
    def forward(self, images: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return images * mask[:, :, None, None, None]


class _SplitMaskModel(torch.nn.Module):
    def forward(
        self,
        flair: torch.Tensor,
        t1ce: torch.Tensor,
        t1: torch.Tensor,
        t2: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        return flair + t1ce + t1 + t2 + mask.sum().float()


class _ListMaskModel(torch.nn.Module):
    def forward(self, images: list[torch.Tensor], mask: int) -> torch.Tensor:
        return sum(images) + float(mask)


@pytest.mark.parametrize(
    ("adapter_kind", "model"),
    (
        ("images_mask", _ImagesMaskModel()),
        ("split_mask", _SplitMaskModel()),
        ("list_mask", _ListMaskModel()),
    ),
)
def test_wrappers_produce_tensor_outputs(adapter_kind: str, model: torch.nn.Module) -> None:
    wrapper = _LegacyWrapper(model, adapter_kind)
    images = torch.randn(1, 4, 8, 8, 8)
    mask = torch.tensor([[True, True, True, True]])
    output = wrapper(images, mask)
    assert isinstance(output, torch.Tensor)


def test_list_mask_wrapper_encodes_modality_descriptor_as_bitmask() -> None:
    class _CaptureMaskModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.seen_mask: int | None = None

        def forward(self, images: list[torch.Tensor], mask: int) -> torch.Tensor:
            self.seen_mask = mask
            return images[0]

    model = _CaptureMaskModel()
    wrapper = _LegacyWrapper(model, "list_mask")
    images = torch.randn(1, 4, 8, 8, 8)
    mask = torch.tensor([[True, True, True, True]])
    _ = wrapper(images, mask)
    assert model.seen_mask == 15


def test_list_mask_tensor_wrapper_passes_boolean_mask_tensor() -> None:
    class _CaptureTensorMaskModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.seen_mask: torch.Tensor | None = None

        def forward(self, images: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            self.seen_mask = mask
            return images

    model = _CaptureTensorMaskModel()
    wrapper = _LegacyWrapper(model, "list_mask_tensor")
    images = torch.randn(1, 4, 8, 8, 8)
    mask = torch.tensor([[True, False, True, False]])
    output = wrapper(images, mask)
    assert torch.equal(output, images)
    assert model.seen_mask is not None
    assert torch.equal(model.seen_mask, mask)


def test_report_serialization_and_table() -> None:
    report = FlopsReport(
        method="SRMNet",
        source_entrypoint="legacy/SRMNet/test.py",
        model_target="model.net.Model",
        measurements=(
            FlopsMeasurement(
                method="SRMNet",
                source_entrypoint="legacy/SRMNet/test.py",
                model_target="model.net.Model",
                measurement="forward",
                input_shape=(1, 4, 128, 128, 128),
                macs=1_000_000.0,
                flops=2_000_000.0,
                params=123,
                status="ok",
            ),
        ),
    )
    payload = report_to_dict(report)
    assert payload["method"] == "SRMNet"
    table = build_flops_table(report)
    assert "Legacy FLOPs for SRMNet" in str(table.title)


def test_run_all_methods_reports_progress_to_stderr(capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(flops_main, "method_names", lambda: ("Alpha", "Beta"))

    def fake_run_flops_subprocess(
        method: str,
        *,
        batch_size: int,
        measure: str,
        shape: tuple[int, int, int] | None,
        full_volume_shape: tuple[int, int, int] | None,
    ) -> dict[str, object]:
        assert batch_size == 2
        assert measure == "all"
        assert shape is None
        assert full_volume_shape is None
        return {
            "method": method,
            "source_entrypoint": f"legacy/{method}/test.py",
            "model_target": f"{method}.Model",
            "measurements": [{
                "method": method,
                "source_entrypoint": f"legacy/{method}/test.py",
                "model_target": f"{method}.Model",
                "measurement": "forward",
                "input_shape": None,
                "macs": None,
                "flops": None,
                "params": None,
                "status": "ok" if method == "Alpha" else "failed",
                "note": "" if method == "Alpha" else "CUDA out of memory",
            }],
        }

    monkeypatch.setattr(flops_main, "run_flops_subprocess", fake_run_flops_subprocess)

    reports = flops_main._run_all_methods(
        batch_size=2,
        measure="all",
        shape=None,
        full_volume_shape=None,
        verbose=False,
    )

    captured = capsys.readouterr()
    assert [report["method"] for report in reports] == ["Alpha", "Beta"]
    assert "[1/2] Running Alpha..." in captured.err
    assert "[1/2] Finished Alpha: ok" in captured.err
    assert "[2/2] Finished Beta: forward failed: CUDA out of memory" in captured.err


def test_run_all_methods_verbose_includes_measurement_summary(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(flops_main, "method_names", lambda: ("Alpha",))
    monkeypatch.setattr(
        flops_main,
        "run_flops_subprocess",
        lambda *args, **kwargs: {
            "method": "Alpha",
            "source_entrypoint": "legacy/Alpha/test.py",
            "model_target": "Alpha.Model",
            "measurements": [
                {"measurement": "forward", "status": "ok"},
                {"measurement": "full_volume_forward", "status": "failed"},
            ],
        },
    )

    flops_main._run_all_methods(
        batch_size=1,
        measure="all",
        shape=None,
        full_volume_shape=None,
        verbose=True,
    )

    captured = capsys.readouterr()
    assert "Finished Alpha: forward=ok, full_volume_forward=failed" in captured.err


def test_run_flops_subprocess_returns_failed_measurement_on_empty_stdout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import subprocess

    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="noisy stdout that should be ignored\n",
            stderr="",
        ),
    )

    report = flops_core.run_flops_subprocess(
        "ShaSpec",
        batch_size=1,
        measure="all",
        shape=None,
        full_volume_shape=None,
    )

    measurement = report["measurements"][0]
    assert measurement["status"] == "failed"
    assert "produced no JSON output" in measurement["note"]


def test_run_flops_subprocess_returns_failed_measurement_on_invalid_json_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import subprocess

    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        cmd = args[0]
        json_index = cmd.index("--json") + 1
        Path(cmd[json_index]).write_text("not json\n")
        return subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="unexpected child noise\n",
            stderr="",
        )

    monkeypatch.setattr(
        subprocess,
        "run",
        fake_run,
    )

    report = flops_core.run_flops_subprocess(
        "ShaSpec",
        batch_size=1,
        measure="all",
        shape=None,
        full_volume_shape=None,
    )

    measurement = report["measurements"][0]
    assert measurement["status"] == "failed"
    assert "produced invalid JSON output" in measurement["note"]


def test_run_flops_subprocess_reads_json_file_even_if_stdout_is_noisy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import subprocess

    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        cmd = args[0]
        json_index = cmd.index("--json") + 1
        Path(cmd[json_index]).write_text('{"method":"ShaSpec","measurements":[]}\n')
        return subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="spurious log line\n",
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    report = flops_core.run_flops_subprocess(
        "ShaSpec",
        batch_size=1,
        measure="all",
        shape=None,
        full_volume_shape=None,
    )

    assert report["method"] == "ShaSpec"
    assert report["measurements"] == []


def test_measure_macs_falls_back_from_aten_to_pytorch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _DummyModel(torch.nn.Module):
        def forward(self, images: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            return images

    calls: list[str] = []

    def fake_get_model_complexity_info(*args: object, **kwargs: object) -> tuple[float | None, None]:
        backend = f"ptflops/{kwargs['backend']}"
        calls.append(backend)
        if backend == "ptflops/aten":
            return None, None
        return 123.0, None

    import types

    monkeypatch.setitem(
        sys.modules,
        "ptflops",
        types.SimpleNamespace(get_model_complexity_info=fake_get_model_complexity_info),
    )

    images = torch.randn(1, 4, 8, 8, 8)
    mask = torch.tensor([[True, True, True, True]])

    macs, note = flops_core._measure_macs(_DummyModel(), "images_mask", images, mask)
    assert macs == 123.0
    assert note == ""
    assert calls == ["ptflops/aten", "ptflops/pytorch"]


def test_measure_macs_falls_back_to_thop_when_ptflops_backends_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _DummyModel(torch.nn.Module):
        def forward(self, images: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            return images

    calls: list[str] = []

    def fake_get_model_complexity_info(*args: object, **kwargs: object) -> tuple[None, None]:
        calls.append(f"ptflops/{kwargs['backend']}")
        return None, None

    import types

    monkeypatch.setitem(
        sys.modules,
        "ptflops",
        types.SimpleNamespace(get_model_complexity_info=fake_get_model_complexity_info),
    )
    monkeypatch.setitem(
        sys.modules,
        "thop",
        types.SimpleNamespace(
            profile=lambda *args, **kwargs: (456.0, 0),
        ),
    )

    images = torch.randn(1, 4, 8, 8, 8)
    mask = torch.tensor([[True, True, True, True]])

    macs, note = flops_core._measure_macs(_DummyModel(), "images_mask", images, mask)
    assert macs == 456.0
    assert note == "estimated with thop hooks"
    assert calls == ["ptflops/aten", "ptflops/pytorch"]


def test_measure_macs_falls_back_to_torch_profiler_when_thop_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _DummyModel(torch.nn.Module):
        def forward(self, images: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            return images

    calls: list[str] = []

    def fake_get_model_complexity_info(*args: object, **kwargs: object) -> tuple[None, None]:
        calls.append(f"ptflops/{kwargs['backend']}")
        return None, None

    def fake_thop_profile(*args: object, **kwargs: object) -> tuple[None, int]:
        calls.append("thop")
        raise AttributeError("'NoneType' object has no attribute 'size'")

    class _FakeEvent:
        def __init__(self, flops: int) -> None:
            self.flops = flops

    class _FakeProfile:
        def __enter__(self) -> "_FakeProfile":
            calls.append("torch.profiler")
            return self

        def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
            return False

        def key_averages(self) -> list[_FakeEvent]:
            return [_FakeEvent(200), _FakeEvent(100)]

    import types

    monkeypatch.setitem(
        sys.modules,
        "ptflops",
        types.SimpleNamespace(get_model_complexity_info=fake_get_model_complexity_info),
    )
    monkeypatch.setitem(
        sys.modules,
        "thop",
        types.SimpleNamespace(profile=fake_thop_profile),
    )
    monkeypatch.setattr(
        torch.profiler,
        "profile",
        lambda *args, **kwargs: _FakeProfile(),
    )

    images = torch.randn(1, 4, 8, 8, 8)
    mask = torch.tensor([[True, True, True, True]])

    macs, note = flops_core._measure_macs(_DummyModel(), "images_mask", images, mask)
    assert macs == 150.0
    assert note == "estimated with torch.profiler operator FLOPs"
    assert calls == ["ptflops/aten", "ptflops/pytorch", "thop", "torch.profiler"]


def test_measure_macs_raises_clear_error_when_all_profilers_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _DummyModel(torch.nn.Module):
        def forward(self, images: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            return images

    def fake_get_model_complexity_info(*args: object, **kwargs: object) -> tuple[None, None]:
        return None, None

    import types

    monkeypatch.setitem(
        sys.modules,
        "ptflops",
        types.SimpleNamespace(get_model_complexity_info=fake_get_model_complexity_info),
    )
    monkeypatch.setitem(
        sys.modules,
        "thop",
        types.SimpleNamespace(
            profile=lambda *args, **kwargs: (None, 0),
        ),
    )
    class _FakeProfile:
        def __enter__(self) -> "_FakeProfile":
            return self

        def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
            return False

        def key_averages(self) -> list[object]:
            return []

    monkeypatch.setattr(
        torch.profiler,
        "profile",
        lambda *args, **kwargs: _FakeProfile(),
    )

    images = torch.randn(1, 4, 8, 8, 8)
    mask = torch.tensor([[True, True, True, True]])

    with pytest.raises(RuntimeError, match="no supported profiler could produce a MAC count"):
        flops_core._measure_macs(_DummyModel(), "images_mask", images, mask)


def test_measure_macs_uses_executed_conv_shapes_not_parameter_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _ConvModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = torch.nn.Conv3d(4, 8, kernel_size=3, bias=False)

        def forward(self, images: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            return self.conv(images)

    import types

    monkeypatch.setitem(
        sys.modules,
        "ptflops",
        types.SimpleNamespace(get_model_complexity_info=lambda *args, **kwargs: (None, None)),
    )
    monkeypatch.setitem(
        sys.modules,
        "thop",
        types.SimpleNamespace(profile=lambda *args, **kwargs: (None, 0)),
    )

    class _FakeProfile:
        def __enter__(self) -> "_FakeProfile":
            return self

        def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
            return False

        def key_averages(self) -> list[object]:
            return []

    monkeypatch.setattr(
        torch.profiler,
        "profile",
        lambda *args, **kwargs: _FakeProfile(),
    )

    images = torch.randn(2, 4, 5, 5, 5)
    mask = torch.tensor([[True, True, True, True], [True, True, True, True]])

    macs, note = flops_core._measure_macs(_ConvModel(), "images_mask", images, mask)
    expected_outputs = 2 * 8 * 3 * 3 * 3
    expected_kernel_macs = 4 * 3 * 3 * 3
    assert macs == float(expected_outputs * expected_kernel_macs)
    assert note == "estimated from executed layers and analytical attention formulas"


def test_measure_macs_counts_multihead_attention_analytically(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _AttentionModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.attn = torch.nn.MultiheadAttention(4, 2, batch_first=True)

        def forward(self, images: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            x = images.view(images.shape[0], 2, 4)
            out, _ = self.attn(x, x, x, need_weights=False)
            return out.view_as(images)

    import types

    monkeypatch.setitem(
        sys.modules,
        "ptflops",
        types.SimpleNamespace(get_model_complexity_info=lambda *args, **kwargs: (None, None)),
    )
    monkeypatch.setitem(
        sys.modules,
        "thop",
        types.SimpleNamespace(profile=lambda *args, **kwargs: (None, 0)),
    )

    class _FakeProfile:
        def __enter__(self) -> "_FakeProfile":
            return self

        def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
            return False

        def key_averages(self) -> list[object]:
            return []

    monkeypatch.setattr(
        torch.profiler,
        "profile",
        lambda *args, **kwargs: _FakeProfile(),
    )

    images = torch.randn(1, 4, 1, 1, 2)
    mask = torch.tensor([[True, True, True, True]])

    macs, note = flops_core._measure_macs(_AttentionModel(), "images_mask", images, mask)
    assert macs == 160.0
    assert note == "estimated from executed layers and analytical attention formulas"


def test_executed_layer_estimator_returns_partial_count_on_late_failure() -> None:
    class _ExplodingModule(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = torch.nn.Conv3d(4, 4, kernel_size=1, bias=False)

        def forward(self, images: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            x = self.conv(images)
            raise AttributeError("boom")

    wrapper = flops_core._LegacyWrapper(_ExplodingModule(), "images_mask")
    images = torch.randn(1, 4, 2, 2, 2)
    mask = torch.tensor([[True, True, True, True]])

    macs, note = flops_core._estimate_macs_from_executed_layers(wrapper, images, mask)
    assert macs == 128.0
    assert note == "analytical fallback was partial: AttributeError"


def test_initialize_lazy_parameters_materializes_lazy_modules() -> None:
    class _LazyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.pool = torch.nn.AdaptiveAvgPool3d((1, 1, 1))
            self.linear = torch.nn.LazyLinear(3)

        def forward(self, images: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            pooled = self.pool(images).flatten(1)
            logits = self.linear(pooled)
            return logits[:, :, None, None, None]

    model = _LazyModel()
    assert flops_core._has_uninitialized_parameters(model)

    flops_core._initialize_lazy_parameters(
        model,
        "images_mask",
        batch_size=2,
        spatial_shape=(4, 4, 4),
        device=torch.device("cpu"),
    )

    assert not flops_core._has_uninitialized_parameters(model)
    assert flops_core._count_parameters(model) == 15


def test_run_flops_for_method_uses_legacy_nonempty_crop_before_window_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _DummyModel(torch.nn.Module):
        def forward(self, images: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            return images

    spec = MethodSpec(
        name="Dummy",
        import_root=Path("."),
        source_entrypoint="legacy/Dummy/test.py",
        model_target="dummy.Model",
        builder=lambda: _DummyModel(),
        adapter_kind="images_mask",
        forward_shape=(128, 128, 128),
        full_volume_shape=DEFAULT_FULL_VOLUME_SHAPE,
        full_volume_strategy=FullVolumeStrategy(
            "legacy_non_empty_patched",
            patch_shape=(128, 128, 128),
            overlap=0.5,
        ),
    )

    monkeypatch.setattr(flops_core, "_resolve_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(flops_core, "get_method_spec", lambda method: spec)
    monkeypatch.setattr(flops_core, "_measure_macs", lambda *args, **kwargs: (10.0, ""))

    report = flops_core.run_flops_for_method("Dummy", measure="all")
    full_volume = next(
        measurement
        for measurement in report.measurements
        if measurement.measurement == "full_volume_forward"
    )

    assert full_volume.input_shape == (1, 4, *DEFAULT_LEGACY_NONEMPTY_CROP_SHAPE)
    assert full_volume.macs == 80.0
    assert "legacy non-empty crop" in full_volume.note
