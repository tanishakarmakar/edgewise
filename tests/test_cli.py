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


def test_unfinished_commands_exit_nonzero():
    for command in ("bench", "advise", "route"):
        assert runner.invoke(app, [command]).exit_code == 1
