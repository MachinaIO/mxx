#!/usr/bin/env python3
"""Build every Lean module of a pipeline directory in dependency order.

Usage: build.py <out dir> [--jobs 2] [--mem-gb 20] [--threads 4] [--only M,M] [--timeout 7200] [--keep]

Modules are the `.lean` files of `<out>/generated` (the export) and `<out>` (generic, adapters,
bridges, generated proof modules). Imports not found there must come from the lake package
(LEAN_PATH, given in the environment); an import found nowhere is reported. Each module is
compiled with `lean -o <out>/olean/<M>.olean`, at most `--jobs` at a time, each under an address
space limit of `--mem-gb` GB; a module whose import failed is skipped. The table (module, status,
seconds, max RSS, error count) goes to `<out>/build_table.txt`; logs to `<out>/logs/<M>.log`.
"""
import argparse
import os
import re
import resource
import subprocess
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument('out')
ap.add_argument('--jobs', type=int, default=2)
ap.add_argument('--mem-gb', type=int, default=20)
ap.add_argument('--only')
ap.add_argument('--timeout', type=int, default=7200)
ap.add_argument('--threads', type=int, default=4)
ap.add_argument('--keep', action='store_true', help='reuse an olean newer than its source and imports')
ap.add_argument('--lean', default=os.environ.get('LEAN', 'lean'))
a = ap.parse_args()
out = os.path.abspath(a.out)
os.makedirs(f'{out}/olean', exist_ok=True)
os.makedirs(f'{out}/logs', exist_ok=True)

src = {}
for d in (f'{out}/generated', out):
    for f in sorted(os.listdir(d)):
        if f.endswith('.lean'):
            src[f[:-5]] = os.path.join(d, f)

lean_path = os.environ.get('LEAN_PATH', '').split(':')


def external(m):
    rel = m.replace('.', '/') + '.olean'
    return any(os.path.exists(os.path.join(p, rel)) for p in lean_path if p)


deps, missing = {}, {}
for m, p in src.items():
    imps = re.findall(r'^import\s+(\S+)', open(p).read(), re.M)
    deps[m] = [i for i in imps if i in src]
    miss = [i for i in imps if i not in src and not external(i)]
    if miss:
        missing[m] = miss

want = set(src)
if a.only:
    want, stack = set(), a.only.split(',')
    while stack:
        m = stack.pop()
        if m not in want:
            want.add(m)
            stack += deps[m]

status, rows = {}, []
running = {}
# largest sources first among the ready modules (the big export modules start early)
pending = sorted(want, key=lambda m: -os.path.getsize(src[m]))
timed_out = set()
peak = {}  # sampled peak RSS (KB) of a module's process group (kept when the module is killed)
last_sample = 0.0


def sample():
    groups = {r[0].pid: m for m, r in running.items()}
    for d in os.listdir('/proc'):
        if not d.isdigit():
            continue
        try:
            st = open(f'/proc/{d}/stat').read().rsplit(')', 1)[1].split()
            g = int(st[2])
            if g in groups:
                rss = int(st[21]) * resource.getpagesize() // 1024
                m = groups[g]
                peak[m] = max(peak.get(m, 0), rss)
        except (OSError, IndexError, ValueError):
            pass


def limit():
    b = a.mem_gb * 1024 ** 3
    resource.setrlimit(resource.RLIMIT_AS, (b, b))


def start(m):
    log = open(f'{out}/logs/{m}.log', 'w')
    cmd = ['/usr/bin/time', '-f', 'maxrss=%MKB', a.lean, '--tstack=1000000', f'--threads={a.threads}',
           f'--root={os.path.dirname(src[m])}', src[m], '-o', f'{out}/olean/{m}.olean']
    p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, preexec_fn=limit, start_new_session=True)
    print(f'start {m} pid {p.pid}', flush=True)
    running[m] = (p, time.time(), log)


def finish(m, rc, t):
    log = open(f'{out}/logs/{m}.log').read()
    rss = re.findall(r'maxrss=(\d+)KB', log)
    kb = int(rss[-1]) if rss else peak.get(m)
    gb = f'{kb / 1048576:.1f}' if kb else '?'
    errs = len(re.findall(r'error:', log))
    sorries = len(re.findall(r"declaration uses .sorry.", log))
    ok = rc == 0 and errs == 0
    status[m] = ok
    rows.append((m, 'ok' if ok else ('TIMEOUT' if m in timed_out else f'FAIL({rc})'), f'{t:.0f}', gb, errs, sorries))
    print(f'{m:45s} {rows[-1][1]:9s} {t:7.0f}s {gb:>5s}GB errors={errs} sorry={sorries}', flush=True)


for m, miss in missing.items():
    if m in want:
        print(f'{m}: imports found nowhere: {miss}', flush=True)

while pending or running:
    if time.time() - last_sample > 5:
        sample()
        last_sample = time.time()
    for m in list(running):
        p, t0, log = running[m]
        if p.poll() is not None:
            log.close()
            del running[m]
            finish(m, p.returncode, time.time() - t0)
        elif time.time() - t0 > a.timeout and m not in timed_out:
            # the whole session (time + lean), by its own process group id
            timed_out.add(m)
            print(f'{m}: past {a.timeout}s, killing process group {p.pid}', flush=True)
            os.killpg(p.pid, 9)
    progressed = False
    for m in list(pending):
        if len(running) >= a.jobs:
            break
        if m in missing or any(status.get(d) is False for d in deps[m]):
            pending.remove(m)
            status[m] = False
            bad = missing.get(m) or [d for d in deps[m] if status.get(d) is False]
            rows.append((m, 'skipped', '-', '-', '-', '-'))
            print(f'{m:45s} skipped (needs {",".join(bad)})', flush=True)
            progressed = True
        elif all(status.get(d) for d in deps[m]):
            pending.remove(m)
            ol = f'{out}/olean/{m}.olean'
            if a.keep and os.path.exists(ol) and os.path.getmtime(ol) > os.path.getmtime(src[m]) and \
                    all(os.path.getmtime(f'{out}/olean/{d}.olean') < os.path.getmtime(ol) for d in deps[m]):
                status[m] = True
                rows.append((m, 'cached', '-', '-', '-', '-'))
            else:
                start(m)
            progressed = True
    if not progressed:
        time.sleep(1)

with open(f'{out}/build_table.txt', 'w') as f:
    f.write(f'{"module":45s} {"status":9s} {"sec":>6s} {"maxGB":>6s} errors sorry\n')
    for r in rows:
        f.write(f'{r[0]:45s} {r[1]:9s} {r[2]:>6s} {r[3]:>6s} {r[4]!s:>6s} {r[5]!s:>5s}\n')
sys.exit(0 if all(status.get(m) for m in want) else 1)
