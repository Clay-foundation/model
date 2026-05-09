"""Test `clay` CLI commands."""

from click.testing import CliRunner

from claymodel.cli import cli


def test_cli_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "Clay Foundation Model CLI" in result.output


def test_info_command():
    runner = CliRunner()
    result = runner.invoke(cli, ["info"])
    assert result.exit_code == 0
    assert "sentinel-2-l2a" in result.output
    assert "sentinel-1-rtc" in result.output


def test_info_sensor_detail():
    runner = CliRunner()
    result = runner.invoke(cli, ["info", "--sensor", "sentinel-2-l2a"])
    assert result.exit_code == 0
    assert "GSD: 10" in result.output
    assert "blue" in result.output
    assert "0.493" in result.output


def test_info_unknown_sensor():
    runner = CliRunner()
    result = runner.invoke(cli, ["info", "--sensor", "fake-sensor"])
    assert result.exit_code == 1
    assert "Unknown sensor" in result.output


def test_embed_missing_file():
    runner = CliRunner()
    result = runner.invoke(cli, ["embed", "nonexistent.tif", "--sensor", "s2", "--ckpt", "x.ckpt"])
    # click should catch the missing file before our code runs
    assert result.exit_code != 0


def test_benchmark_command():
    runner = CliRunner()
    result = runner.invoke(cli, ["benchmark", "--size", "32"])
    assert result.exit_code == 0
    assert "PASS" in result.output
    assert "Avg time" in result.output


def test_info_sar_sensor_no_rgb():
    runner = CliRunner()
    result = runner.invoke(cli, ["info", "--sensor", "sentinel-1-rtc"])
    assert result.exit_code == 0
    assert "No RGB indices" in result.output
