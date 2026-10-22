"""Windows PowerShell scripts must stay pure ASCII.

Windows PowerShell 5.1 reads a BOM-less ``.ps1`` file as the system ANSI code
page, not UTF-8. A UTF-8 em dash decodes to ``â€”`` there, and its last byte
becomes a curly quote that PowerShell treats as a string delimiter, so the
whole script fails to parse ("Missing closing '}'").
"""

from __future__ import annotations

from pathlib import Path

import pytest

_SCRIPTS = sorted((Path(__file__).parents[2] / "deploy" / "windows").glob("*.ps1"))


@pytest.mark.parametrize("script", _SCRIPTS, ids=lambda p: p.name)
def test_windows_powershell_script_is_ascii(script: Path) -> None:
    for lineno, line in enumerate(script.read_bytes().splitlines(), start=1):
        assert line.isascii(), f"{script.name}:{lineno} has non-ASCII characters"
