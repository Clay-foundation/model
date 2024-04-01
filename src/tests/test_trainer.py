"""
Tests for command-line interface to execute ChaBuDNet.

Based on advanced usage of running LightningCLI from Python at
https://lightning.ai/docs/pytorch/2.1.0/cli/lightning_cli_advanced_3.html#run-from-python
"""

import pytest

from trainer import cli_main


# %%
@pytest.mark.parametrize("subcommand", ["fit", "validate", "test"])
def test_cli_main(subcommand, capsys):
    """
    Ensure that running `python trainer.py` works with the subcommands `fit`
    and `validate`.
    """
    with pytest.raises(expected_exception=SystemExit, match="0"):
        cli_main(args=[subcommand, "--print_config=skip_null"])

    captured = capsys.readouterr()
    assert "seed_everything:" in captured.out
    assert "trainer:" in captured.out
    assert "model:" in captured.out
    assert "data:" in captured.out
