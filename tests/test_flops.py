from __future__ import annotations

import torch
import typer

from mimose.flops import (
    MODALITY_COMBINATIONS,
    center_crop_or_pad_volume,
    merge_model_kwargs,
    modality_label,
    modality_mask,
    prepare_measurement_inputs,
    resolve_2023_config_paths,
    resolve_model_name,
    validate_flops_selection,
)


def test_merge_model_kwargs_prefers_cli_overrides() -> None:
    yaml_config = {
        "custom_model_kwargs": {
            "num_cls": 4,
            "fusion_type": "RFM",
        }
    }

    merged = merge_model_kwargs(
        yaml_config,
        {
            "fusion_type": "gated",
            "extra": True,
        },
    )

    assert merged == {
        "num_cls": 4,
        "fusion_type": "gated",
        "extra": True,
    }


def test_resolve_model_name_allows_config_or_cli_model() -> None:
    assert resolve_model_name({"model": "imfuse"}, None) == "imfuse"
    assert resolve_model_name({"model": "imfuse"}, "dcseg") == "dcseg"
    assert resolve_model_name({}, "mmformer") == "mmformer"


def test_resolve_2023_config_paths_returns_sorted_2023_yaml_files(tmp_path) -> None:
    (tmp_path / "mmformer_23.yaml").touch()
    (tmp_path / "imfuse_18.yaml").touch()
    (tmp_path / "dcseg_23.yaml").touch()

    config_paths = resolve_2023_config_paths(tmp_path)

    assert [path.name for path in config_paths] == [
        "dcseg_23.yaml",
        "mmformer_23.yaml",
    ]


def test_validate_flops_selection_rejects_all_with_single_target() -> None:
    validate_flops_selection(
        all_configs=True,
        config_path=None,
        model_name=None,
    )

    try:
        validate_flops_selection(
            all_configs=True,
            config_path=None,
            model_name="imfuse",
        )
    except typer.BadParameter as exc:
        assert "--all cannot be combined" in str(exc)
    else:
        raise AssertionError("expected BadParameter")


def test_modality_combinations_cover_all_15_non_empty_subsets() -> None:
    assert len(MODALITY_COMBINATIONS) == 15
    assert len(set(MODALITY_COMBINATIONS)) == 15
    assert all(any(combo) for combo in MODALITY_COMBINATIONS)


def test_modality_mask_matches_combination_and_batch_size() -> None:
    combo = (True, False, True, False)

    mask = modality_mask(combo, batch_size=3, device=torch.device("cpu"))

    assert mask.dtype is torch.bool
    assert mask.shape == (3, 4)
    assert torch.equal(mask, torch.tensor(combo).unsqueeze(0).expand(3, -1))


def test_modality_label_joins_present_modality_names() -> None:
    assert modality_label((True, False, True, False)) == "t1c+t2f"
    assert modality_label((True, True, True, True)) == "t1c+t1n+t2f+t2w"


def test_center_crop_or_pad_volume_crops_and_pads_spatial_dims() -> None:
    images = torch.arange(1 * 1 * 5 * 7 * 3, dtype=torch.float32).reshape(1, 1, 5, 7, 3)

    cropped = center_crop_or_pad_volume(images, (3, 3, 5))

    assert cropped.shape == (1, 1, 3, 3, 5)
    expected_crop = images[:, :, 1:4, 2:5, :]
    assert torch.equal(cropped[:, :, :, :, 1:4], expected_crop)
    assert torch.equal(cropped[:, :, :, :, 0], torch.zeros(1, 1, 3, 3))
    assert torch.equal(cropped[:, :, :, :, 4], torch.zeros(1, 1, 3, 3))


def test_prepare_measurement_inputs_reports_forward_and_predict_when_patched() -> None:
    images = torch.zeros(1, 4, 10, 12, 14)

    prepared = prepare_measurement_inputs(
        images,
        patch_size=8,
        device=torch.device("cpu"),
    )

    assert [item.name for item in prepared] == ["forward", "predict"]
    assert prepared[0].input_shape == (1, 4, 8, 8, 8)
    assert prepared[1].input_shape == (1, 4, 10, 12, 14)
    assert prepared[0].images is not None
    assert prepared[1].images is not None


def test_prepare_measurement_inputs_skips_predict_without_patch_size() -> None:
    images = torch.zeros(1, 4, 10, 12, 14)

    prepared = prepare_measurement_inputs(
        images,
        patch_size=None,
        device=torch.device("cpu"),
    )

    assert [item.name for item in prepared] == ["forward", "predict"]
    assert prepared[0].input_shape == (1, 4, 10, 12, 14)
    assert prepared[1].images is None
    assert "skipped" in prepared[1].note
