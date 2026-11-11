from typer.testing import CliRunner

from edgewise import __version__
from edgewise.cli import app
from edgewise.hardware import detect

runner = CliRunner()


def test_version():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_info_shows_device():
    result = runner.invoke(app, ["info"])
    assert result.exit_code == 0
    assert "RAM" in result.stdout


def test_detect_reports_memory():
    hw = detect()
    assert hw.ram_gb > 0
    assert 0 <= hw.model_budget_gb < hw.ram_gb


def test_models_ls_lists_catalog():
    result = runner.invoke(app, ["models", "ls"])
    assert result.exit_code == 0
    assert "qwen2.5-1.5b-instruct@mlx-4bit" in result.stdout


def test_bench_rejects_unknown_model():
    result = runner.invoke(app, ["bench", "--model", "no-such-model"])
    assert result.exit_code != 0
