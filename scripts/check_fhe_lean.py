#!/usr/bin/env python3
"""Compiles mxx's Lean packages built on VCVio against a built mathcal-A checkout.

mxx's Lean packages (`crates/ir-core/lean`, its `security` package, and the FHE packages under
`crates/fhe/lean`) pin VCVio and mathlib at mathcal-A's revisions, so one package set imports
both (ARCHITECTURE-SPEC.md, section 10, decision 4). This script checks that: it compiles the
modules reachable from the given roots, in dependency order and in parallel, with `lean` over
the build outputs of the checkout at `MATHCAL_A_DIR` (only read), writing the `.olean` files to
`target/lean-mxx`. Modules whose `.olean` is newer than their source and imports are reused.

Usage: MATHCAL_A_DIR=... scripts/check_fhe_lean.py [-j JOBS] [--scheme bgv|tfhe] [ROOT_MODULE...]
(default roots: MxxSecurity MxxFhe, or with `--scheme` every library root of that scheme's
package). The two schemes' generated modules share names, so each scheme's modules are written to
their own directory under `target/lean-mxx`.
"""
import argparse
import concurrent.futures
import glob
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCES = [
    "crates/ir-core/lean",
    "crates/ir-core/lean/security",
    "crates/fhe/lean/spec",
]
SCHEMES = ["bgv", "tfhe"]
OUT = os.path.join(ROOT, "target", "lean-mxx")
IMPORT = re.compile(r"^\s*(?:public\s+)?(?:meta\s+)?import\s+(?:all\s+)?([\w.«»']+)", re.M)


def scheme_roots(scheme):
    """The roots of every `lean_lib` of a scheme's lakefile."""
    text = open(os.path.join(ROOT, "crates/fhe/lean", scheme, "lakefile.toml")).read()
    roots = []
    for block in re.findall(r"roots\s*=\s*\[([^\]]*)\]", text):
        roots += re.findall(r'"([^"]+)"', block)
    return roots


def source_of(module):
    rel = module.replace(".", "/") + ".lean"
    for directory in SOURCES:
        path = os.path.join(ROOT, directory, rel)
        if os.path.exists(path):
            return path
    return None


def imports(path):
    text = open(path, encoding="utf-8").read()
    # Imports come first; stop at the first non-import, non-comment line.
    header = text.split("\nnamespace", 1)[0]
    return IMPORT.findall(header)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-j", "--jobs", type=int, default=6)
    parser.add_argument("--scheme", choices=SCHEMES)
    parser.add_argument("roots", nargs="*")
    args = parser.parse_args()
    global OUT
    if args.scheme:
        SOURCES.append(f"crates/fhe/lean/{args.scheme}")
        SOURCES.append(f"crates/fhe/lean/{args.scheme}/generated")
    roots = args.roots or (scheme_roots(args.scheme) if args.scheme else ["MxxSecurity", "MxxFhe"])
    shared = OUT
    mathcal = os.environ.get("MATHCAL_A_DIR")
    if not mathcal:
        sys.exit("set MATHCAL_A_DIR to a built mathcal-A checkout")
    lean = os.environ.get("LEAN") or subprocess.run(
        ["elan", "which", "lean"], capture_output=True, text=True, check=True
    ).stdout.strip()
    paths = sorted(glob.glob(os.path.join(mathcal, ".lake/packages/*/.lake/build/lib/lean")))
    paths.append(os.path.join(mathcal, ".lake/build/lib/lean"))
    paths.append(shared)
    if args.scheme:
        paths.append(os.path.join(shared, args.scheme))
    env = dict(os.environ, LEAN_PATH=":".join(paths))

    # The local modules reachable from the roots, and their local imports.
    deps, todo = {}, list(roots)
    while todo:
        module = todo.pop()
        if module in deps:
            continue
        path = source_of(module)
        if path is None:
            sys.exit(f"no source for {module}")
        local = [m for m in imports(path) if source_of(m) is not None]
        deps[module] = local
        todo.extend(local)

    def olean(m):
        base = shared
        if args.scheme and "/crates/fhe/lean/" + args.scheme in source_of(m):
            base = os.path.join(shared, args.scheme)
        return os.path.join(base, m.replace(".", "/") + ".olean")
    done, failed = set(), {}
    pending = dict(deps)

    def fresh(module):
        target = olean(module)
        if not os.path.exists(target):
            return False
        stamp = os.path.getmtime(target)
        if os.path.getmtime(source_of(module)) > stamp:
            return False
        return all(os.path.getmtime(olean(d)) <= stamp for d in deps[module])

    def compile_one(module):
        if fresh(module):
            return module, None
        target = olean(module)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        result = subprocess.run(
            [lean, "-o", target, "-i", target[:-6] + ".ilean", source_of(module)],
            env=env,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            if os.path.exists(target):
                os.remove(target)
            return module, (result.stdout + result.stderr)[-4000:]
        return module, None

    with concurrent.futures.ThreadPoolExecutor(args.jobs) as pool:
        running = {}
        while pending or running:
            ready = [
                m for m, ds in pending.items()
                if all(d in done for d in ds) and not any(d in failed for d in ds)
            ]
            blocked = [m for m, ds in pending.items() if any(d in failed for d in ds)]
            for m in blocked:
                failed[m] = "a dependency failed"
                del pending[m]
            for m in ready:
                del pending[m]
                running[pool.submit(compile_one, m)] = m
            if not running:
                break
            finished, _ = concurrent.futures.wait(
                running, return_when=concurrent.futures.FIRST_COMPLETED
            )
            for future in finished:
                del running[future]
                module, error = future.result()
                if error is None:
                    done.add(module)
                    print(f"ok   {module}", flush=True)
                else:
                    failed[module] = error
                    print(f"FAIL {module}\n{error}", flush=True)
    real = {m: e for m, e in failed.items() if e != "a dependency failed"}
    print(f"{len(done)} compiled, {len(real)} failed, {len(failed) - len(real)} blocked")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
