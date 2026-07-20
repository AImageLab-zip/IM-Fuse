from pathlib import Path

from typer.testing import CliRunner

from mimose import cli
from mimose import cli_display
from mimose.cli_setup import update_setup_config


def _write_config(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "input_dir: /path/to/unpacked",
                "output_dir: /path/to/preprocessed",
                "dataset_type: brats23",
                "data_dir: /path/to/preprocessed",
                "art_dir: /path/to/artifacts",
                "results_dir: /path/to/results",
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
        brats_data_dir=tmp_path / "brats_data",
        preprocessed_root_dir=tmp_path / "preprocessed",
        artifacts_root_dir=tmp_path / "artifacts",
        results_root_dir=tmp_path / "results",
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
        brats_data_dir=tmp_path / "brats_data",
        preprocessed_root_dir=tmp_path / "preprocessed",
        artifacts_root_dir=tmp_path / "artifacts",
        results_root_dir=tmp_path / "results",
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
        brats_data_dir=tmp_path / "brats_data",
        preprocessed_root_dir=tmp_path / "preprocessed",
        artifacts_root_dir=tmp_path / "artifacts",
        results_root_dir=tmp_path / "results",
        hf_repo="owner/repo",
    )

    content = config_path.read_text(encoding="utf-8")
    assert "push_to_hf: false" in content
    assert "hf_repo: lab/model-zoo" in content


def test_prompt_required_directory_uses_default_dir(monkeypatch, tmp_path: Path) -> None:
    expected_default = f"{tmp_path}/"
    captured: dict[str, str] = {}

    def fake_prompt_path(prompt: str, *, default: str = "") -> str:
        captured["prompt"] = prompt
        captured["default"] = default
        return default

    monkeypatch.setattr(cli_display, "prompt_path", fake_prompt_path)

    result = cli_display.prompt_required_directory(
        label="Artifacts Root",
        prompt="Root directory for training artifacts",
        default_dir=tmp_path,
    )

    assert captured == {
        "prompt": "Root directory for training artifacts",
        "default": expected_default,
    }
    assert result == tmp_path.resolve()


def test_setup_defaults_artifacts_root_to_data_root_runs(monkeypatch, tmp_path: Path) -> None:
    runner = CliRunner()
    data_root = tmp_path / "data"
    data_root.mkdir()
    expected_preprocessed_default = data_root
    expected_artifacts_default = data_root / "runs"
    expected_results_default = data_root / "results"
    captured_default_dirs: dict[str, Path | None] = {}

    class FakeConsole:
        def print(self, *args, **kwargs) -> None:
            return None

    class FakeDisplay:
        CONSOLE = FakeConsole()

        @staticmethod
        def prompt_path(prompt: str, *, default: str = "") -> str:
            return ""

        @staticmethod
        def prompt_zip_file(*, label: str, prompt: str, default_dir: Path | None = None) -> Path:
            raise AssertionError("ZIP prompt should not be used when unpack is disabled")

        @staticmethod
        def prompt_required_existing_directory(
            *, label: str, prompt: str, default_dir: Path | None = None
        ) -> Path:
            if label == "Unpacked Data":
                return data_root / "unpacked"
            raise AssertionError(f"Unexpected label: {label}")

        @staticmethod
        def prompt_required_directory(
            *,
            label: str,
            prompt: str,
            default_dir: Path | None = None,
        ) -> Path:
            captured_default_dirs[label] = default_dir
            if label == "Preprocessed Data Root":
                return expected_preprocessed_default
            if label == "Artifacts Root":
                return expected_artifacts_default
            if label == "Results Root":
                return expected_results_default
            raise AssertionError(f"Unexpected label: {label}")

    class FakeSetup:
        CONFIG_TEMPLATES_DIR = tmp_path / "templates"

        @staticmethod
        def copy_config_templates() -> list[Path]:
            return []

        @staticmethod
        def update_setup_config(**kwargs) -> None:
            return None

        @staticmethod
        def templates_require_hf_repo_prompt() -> bool:
            return False

    monkeypatch.setattr(cli, "_get_cli_display", lambda: FakeDisplay)
    monkeypatch.setattr(cli, "_get_cli_setup", lambda: FakeSetup)

    import rich.prompt

    answers = iter([False, True])
    monkeypatch.setattr(rich.prompt.Confirm, "ask", lambda *args, **kwargs: next(answers))
    monkeypatch.setattr(rich.prompt.Prompt, "ask", lambda *args, **kwargs: "online")

    result = runner.invoke(cli.app, ["setup"])

    assert result.exit_code == 0
    assert captured_default_dirs == {
        "Preprocessed Data Root": expected_preprocessed_default,
        "Artifacts Root": expected_artifacts_default,
        "Results Root": expected_results_default,
    }
