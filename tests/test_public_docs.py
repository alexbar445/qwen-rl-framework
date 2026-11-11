"""Published guides must resolve locally and their complete adapter example runs."""
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]


def test_public_markdown_relative_links_resolve():
    pages = [ROOT / name for name in ["README.md", "README.zh-CN.md", "CONTRIBUTING.md",
                                      "SECURITY.md", "THIRD_PARTY_NOTICES.md"]]
    pages += list((ROOT / "docs").glob("*.md"))
    pages += list((ROOT / "examples").rglob("README.md"))
    for page in pages:
        for target in re.findall(r"\[[^\]]*\]\(([^)]+)\)", page.read_text()):
            if re.match(r"[a-zA-Z][a-zA-Z0-9+.-]*:", target) or target.startswith("#"):
                continue
            path = unquote(target.split("#", 1)[0])
            assert (page.parent / path).exists(), f"{page.relative_to(ROOT)} -> {target}"


def test_complete_environment_guide_example_executes_without_model_dependencies():
    text = (ROOT / "docs/EXTENDING.md").read_text()
    snippets = re.findall(r"```python\n(.*?)\n```", text, flags=re.S)
    example = next(code for code in snippets if "class ChoiceEnvironment" in code)
    # Execute through the source root so this also works before installation.
    result = subprocess.run([sys.executable, "-c", example], cwd=ROOT, check=True,
                            capture_output=True, text=True, timeout=20)
    assert result.stdout.strip() == "[0.0, 1.0]"
