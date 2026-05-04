from pathlib import Path

from mimose.cli_setup import update_setup_config


def _write_config(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "input_dir: /path/to/brats23",
                "output_dir: /path/to/preprocessed",
                "dataset_type: brats23",
                "data_dir: /path/to/preprocessed",
                "art_dir: /path/to/artifacts",
                "wandb_run_name: imfuse23_training",
                "push_to_hf: false",
                "hf_repo: /path/to/huggingface/owner-repo",
                "checkpoint_path: /path/to/checkpoint",
                "output_path: /path/to/results",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_update_setup_config_sets_placeholder_hf_repo_to_null_when_empty(tmp_path: Path) -> None:
    config_path = tmp_path / "imfuse_23.yaml"
    _write_config(config_path)

    update_setup_config(
        config_path=config_path,
        brats18_dir=None,
        brats23_dir=tmp_path / "brats23",
        preprocessed_root_dir=tmp_path / "preprocessed",
        artifacts_root_dir=tmp_path / "artifacts",
        hf_repo=None,
    )

    content = config_path.read_text(encoding="utf-8")
    assert "push_to_hf: false" in content
    assert "hf_repo: null" in content


def test_update_setup_config_sets_hf_keys_when_placeholder_and_repo_provided(tmp_path: Path) -> None:
    config_path = tmp_path / "imfuse_23.yaml"
    _write_config(config_path)

    update_setup_config(
        config_path=config_path,
        brats18_dir=None,
        brats23_dir=tmp_path / "brats23",
        preprocessed_root_dir=tmp_path / "preprocessed",
        artifacts_root_dir=tmp_path / "artifacts",
        hf_repo="owner/repo",
    )

    content = config_path.read_text(encoding="utf-8")
    assert "push_to_hf: true" in content
    assert "hf_repo: owner/repo" in content


def test_update_setup_config_preserves_non_placeholder_hf_repo(tmp_path: Path) -> None:
    config_path = tmp_path / "imfuse_23.yaml"
    _write_config(config_path)
    config_path.write_text(
        config_path.read_text(encoding="utf-8").replace(
            "hf_repo: /path/to/huggingface/owner-repo",
            "hf_repo: lab/model-zoo",
        ),
        encoding="utf-8",
    )

    update_setup_config(
        config_path=config_path,
        brats18_dir=None,
        brats23_dir=tmp_path / "brats23",
        preprocessed_root_dir=tmp_path / "preprocessed",
        artifacts_root_dir=tmp_path / "artifacts",
        hf_repo="owner/repo",
    )

    content = config_path.read_text(encoding="utf-8")
    assert "push_to_hf: false" in content
    assert "hf_repo: lab/model-zoo" in content
