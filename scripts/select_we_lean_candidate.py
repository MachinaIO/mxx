#!/usr/bin/env python3
"""Select generated candidate sources for the WE Lean editor project (Python 3.11+)."""

import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile
import tomllib


def select_candidate(artifact: Path, project: Path) -> Path:
    """Copy sources, never compiled artifacts or handwritten proofs, then switch the link."""
    artifact = artifact.resolve(strict=True)
    config = tomllib.loads((project / "lakefile.toml").read_text())
    library = next(lib for lib in config["lean_lib"] if lib["name"] == "DiamondCandidate")
    sources = [artifact / f"{name}.lean" for name in library["roots"]]
    for source in [artifact / "Certificate.lean", *sources]:
        if not source.is_file():
            raise ValueError(f"incomplete passing-candidate artifact: missing {source}")
    selected = project / library["srcDir"]
    if selected.exists() and not selected.is_symlink():
        raise ValueError(f"refusing to replace a non-symlink directory: {selected}")
    snapshots = project / ".lake" / "editor-candidates"
    snapshots.mkdir(parents=True, exist_ok=True)
    snapshot = Path(tempfile.mkdtemp(prefix="candidate-", dir=snapshots))
    try:
        for source in sources:
            shutil.copyfile(source, snapshot / source.name)
        (snapshot / "selection.json").write_text(
            json.dumps({"artifact": str(artifact)}, indent=2) + "\n"
        )
        # Keep the previous complete snapshot on disk. A failed copy cannot partially switch
        # the editor to another candidate; Lake recompiles sources rather than reusing oleans.
        with tempfile.TemporaryDirectory(prefix="selection-", dir=snapshots) as staging:
            link = Path(staging) / "generated"
            link.symlink_to(os.path.relpath(snapshot, project), target_is_directory=True)
            os.replace(link, selected)
    except BaseException:
        shutil.rmtree(snapshot)
        raise
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path, help="candidate directory retained by parameter search")
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1] / "crates" / "we" / "lean"
    selected = select_candidate(args.artifact, project)
    print(f"Selected candidate sources: {selected}")
    print("Run `lake build` in crates/we/lean, then restart the Lean server in VS Code.")
    print("This selects an audit snapshot; it does not certify the current Rust/DSL sources.")


if __name__ == "__main__":
    main()
