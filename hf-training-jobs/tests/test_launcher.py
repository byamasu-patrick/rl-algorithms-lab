import sys
from pathlib import Path

import pytest

import hf_jobs.launcher as launcher
from hf_jobs import launch, parse_launcher_args
from hf_jobs.config import JobConfig, resolve_config


def test_launch_without_flag_strips_hf_flags_and_returns(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train.py", "--lr", "3", "--hf-flavor", "l4x1", "--hf-entity", "me"])
    launch()
    assert sys.argv == ["train.py", "--lr", "3", "--hf-entity", "me"]


def test_hf_job_false_runs_locally(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train.py", "--hf-job", "false", "--epochs", "1"])
    launch()
    assert sys.argv == ["train.py", "--epochs", "1"]


def test_parse_leaves_unset_options_as_none():
    opts, rest = parse_launcher_args(["--hf-job", "--seed", "1"])
    assert opts.hf_job is True
    assert opts.hf_flavor is None and opts.hf_namespace is None
    assert rest == ["--seed", "1"]


def test_config_precedence(tmp_path):
    (tmp_path / "pyproject.toml").write_text(
        '[tool.hf-jobs]\nnamespace = "from-pyproject"\nflavor = "t4-small"\ntimeout = "2h"\n', encoding="utf-8")
    config = resolve_config(tmp_path, {"flavor": "l4x1", "timeout": None},
                            environ={"HF_JOBS_TIMEOUT": "5h"})
    assert config.namespace == "from-pyproject"
    assert config.flavor == "l4x1"   # flag beats pyproject
    assert config.timeout == "5h"    # environment beats pyproject


def test_defaults_without_pyproject(tmp_path):
    assert resolve_config(tmp_path, environ={}) == JobConfig()


def test_unknown_pyproject_key_is_rejected(tmp_path):
    (tmp_path / "pyproject.toml").write_text('[tool.hf-jobs]\nflavour = "t4-small"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="flavour"):
        resolve_config(tmp_path, environ={})


def test_collect_files_skips_excluded(tmp_path):
    for relative in ["train.py", "src/model.py", ".venv/lib.py", "runs/x/events", "data/big.bin", "a.pyc"]:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x", encoding="utf-8")
    files = [relative for _, relative in launcher.collect_files(tmp_path, exclude=["data"])]
    assert files == ["src/model.py", "train.py"]


def test_remote_command_installs_runs_and_copies_outputs(tmp_path):
    project = tmp_path / "my project"
    project.mkdir()
    (project / "requirements.txt").write_text("numpy\n", encoding="utf-8")
    script = launcher.remote_command(JobConfig(outputs=["runs", "checkpoints"]), project, "train.py", ["--name", "a b"])
    assert "cd /workspace/'my project'" in script
    assert "pip install --no-cache-dir -r requirements.txt || exit 1" in script
    assert "python train.py --name 'a b'" in script
    assert "for d in runs checkpoints;" in script


def test_install_command_falls_back_to_pyproject(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'x'\n", encoding="utf-8")
    assert launcher.install_command(JobConfig(), tmp_path) == "pip install --no-cache-dir ."
    assert launcher.install_command(JobConfig(install="uv sync"), tmp_path) == "uv sync"


def test_launcher_source_only_uploaded_when_sibling_checkout(tmp_path):
    source = Path(launcher.__file__).resolve().parent.parent
    assert launcher.launcher_source_dir(tmp_path / "elsewhere") is None
    expected = source if (source / "pyproject.toml").exists() else None
    assert launcher.launcher_source_dir(source.parent / "some-project") == expected


def test_module_style_import(monkeypatch):
    from hf_jobs import launcher as module

    monkeypatch.setattr(sys, "argv", ["train.py", "--hf-dry-run", "--x", "1"])
    module.launch()
    assert sys.argv == ["train.py", "--x", "1"]


def test_hydra_overrides_after_hf_job_are_left_for_the_script():
    opts, rest = parse_launcher_args(["--hf-job", "+algorithm=ia2c", "env.time_limit=25", "-m", "seed=0,1"])
    assert opts.hf_job is True
    assert rest == ["+algorithm=ia2c", "env.time_limit=25", "-m", "seed=0,1"]


def test_only_word_values_are_taken_as_a_separate_switch_value():
    opts, rest = parse_launcher_args(["--hf-job", "false", "--hf-dry-run", "1"])
    assert opts.hf_job is False
    assert opts.hf_dry_run is True
    assert rest == ["1"]


def test_inline_values():
    opts, rest = parse_launcher_args(["--hf-job=0", "--hf-flavor=l4x1", "x=1"])
    assert opts.hf_job is False
    assert opts.hf_flavor == "l4x1"
    assert rest == ["x=1"]


def test_arguments_after_double_dash_are_not_read():
    opts, rest = parse_launcher_args(["--seed", "1", "--", "--hf-job"])
    assert opts.hf_job is False
    assert rest == ["--seed", "1", "--", "--hf-job"]


def test_value_option_without_value_is_an_error():
    with pytest.raises(SystemExit, match="--hf-flavor expects a value"):
        parse_launcher_args(["--hf-flavor", "--hf-job"])


def test_invalid_truth_value_is_an_error():
    with pytest.raises(SystemExit, match="invalid truth value"):
        parse_launcher_args(["--hf-job=maybe"])


def test_default_outputs_cover_cleanrl_and_hydra():
    assert JobConfig().outputs == ["runs", "videos", "outputs", "multirun"]


def test_output_folders_are_not_uploaded(tmp_path, monkeypatch, capsys):
    project = tmp_path / "proj"
    for relative in ["run.py", "configs/default.yaml", "outputs/a/results.csv", "multirun/b/results.csv",
                     "checkpoints/model.pt"]:
        path = project / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x", encoding="utf-8")
    (project / "pyproject.toml").write_text('[tool.hf-jobs]\noutputs = ["outputs", "checkpoints"]\n',
                                            encoding="utf-8")

    class FakeApi:
        def __init__(self, token):
            pass

        def whoami(self):
            return {"name": "me"}

    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    monkeypatch.setattr(huggingface_hub, "get_token", lambda: "hf_x")
    monkeypatch.setattr(sys, "argv", [str(project / "run.py"), "--hf-job", "--hf-dry-run", "+algorithm=ia2c"])
    with pytest.raises(SystemExit) as exit_info:
        launch()
    assert exit_info.value.code == 0
    out = capsys.readouterr().out
    assert "/proj/run.py" in out and "/proj/configs/default.yaml" in out
    assert "results.csv" not in out and "model.pt" not in out   # configured and default outputs skipped
    assert "python run.py +algorithm=ia2c" in out
    assert "for d in outputs checkpoints;" in out
