"""Write the bridge from an average-case export to the worst-case relation of the same graph.

The average-case export of a graph is its worst-case export with every relation taking a sampling
tape and a path, and every sampler reading the tape (`gaussianSampleAt`, …). Each tape sampler
implies its worst-case sampler, so every average-case run is a worst-case run: a worst-case
correctness proof gives the average-case claim with failure probability zero.

This script writes, in the average-case parameter directory:
- `worst/`: the worst-case export, renamed (`Stage_x` → `W_Stage_x`, `Ideal` → `W_Ideal`,
  `Requirement_0` → `W_Requirement_0`), with `W_Claim.lean` holding its `Runs` over the
  average-case claim's `ExternalInputs` and `Execution`;
- `bridge/`: per stage, `Stage_x.D.bridge : Stage_x.D … tape path … → W_Stage_x.D …` for every
  relation `D`, and `BridgeRuns.lean` with `GeneratedClaim.runs_bridge`.

Usage: `python3 gen_bridge.py <average directory> <worst directory>`.
"""
import os
import re
import sys

avg, worst = sys.argv[1], sys.argv[2]
ROOTS = re.compile(r'\b(Stage_\w+|Ideal|Requirement_0)\b')
os.makedirs(f'{avg}/worst', exist_ok=True)
os.makedirs(f'{avg}/bridge', exist_ok=True)

modules = sorted(f[:-5] for f in os.listdir(f'{avg}/generated')
                 if f.endswith('.lean') and ROOTS.fullmatch(f[:-5]))


def renamed(text):
    return ROOTS.sub(lambda m: 'W_' + m.group(1), text)


for m in modules:
    open(f'{avg}/worst/W_{m}.lean', 'w').write(renamed(open(f'{worst}/generated/{m}.lean').read()))

claim = open(f'{worst}/generated/Claim.lean').read()
imports = ''.join(f'import W_{m}\n' for m in modules)
body = claim[claim.index('def stage_0_params'):claim.index('/-- Every execution')]
open(f'{avg}/worst/W_Claim.lean', 'w').write(f'''import Claim
{imports}
/-! The worst-case relation of the graph (written by `generator/gen_bridge.py`): the worst-case
export's `Runs` over the average-case claim's inputs and executions. -/

set_option maxRecDepth 16384

namespace W_GeneratedClaim

open GeneratedClaim

{renamed(body).rstrip()}

end W_GeneratedClaim
''')


def top_args(s):
    """The top-level parenthesized groups of `s`."""
    out, depth, start = [], 0, None
    for i, ch in enumerate(s):
        if ch == '(':
            if depth == 0:
                start = i
            depth += 1
        elif ch == ')':
            depth -= 1
            if depth == 0:
                out.append(s[start + 1:i])
    return out


def blocks(text):
    """Map each top-level declaration name to `(kind, header, body lines)`."""
    out = {}
    lines = text.split('\n')
    i = 0
    while i < len(lines):
        m = re.match(r'(def|abbrev|structure) (\S+)', lines[i])
        if not m:
            i += 1
            continue
        j = i + 1
        while j < len(lines) and lines[j] != '' and not re.match(r'(def|abbrev|structure|end) ', lines[j]):
            j += 1
        out[m.group(2)] = (m.group(1), lines[i], lines[i + 1:j])
        i = j
    return out


def sub_name(atom):
    m = re.search(r'(?:^|[(\s])((?:parallel|sequential)_\w+) (?:backend )?(?:hashModel )?tape ', atom)
    assert m, atom
    return m.group(1)


for m in modules:
    text = open(f'{avg}/generated/{m}.lean').read()
    B = blocks(text)
    out = [f'''import {m}
import W_{m}
import ExponentSampling

/-! Every run of the average-case relations of `{m}` is a run of its worst-case relations
(written by `generator/gen_bridge.py`). -/

set_option maxRecDepth 65536
set_option maxHeartbeats 64000000

theorem {m}.iterRuns_mono {{State : Type}} {{f g : Nat → State → State → Prop}} {{count : Nat}}
    {{a b : State}} (h : MxxIR.IterRuns f count a b) (hfg : ∀ i c n, f i c n → g i c n) :
    MxxIR.IterRuns g count a b := by
  induction h with
  | zero s => exact .zero s
  | step _ hs ih => exact .step ih (hfg _ _ _ hs)
''']
    counter = [0]

    def fresh(p):
        counter[0] += 1
        return f'{p}{counter[0]}'

    def atom_term(atom, h):
        if 'tape' not in atom:
            return h
        if 'MxxRuntime.gaussianSampleAt tape' in atom:
            return f'(MxxExponentLut.gaussianSampleAt_sample (by decide) {h})'
        if 'MxxRuntime.uniformIntervalSampleAt tape' in atom:
            return f'(MxxExponentLut.uniformIntervalSampleAt_sample (by decide) {h})'
        if atom.startswith('(∀ i : Fin'):
            return f'(fun i ↦ {m}.{sub_name(atom)}.bridge ({h} i))'
        if atom.startswith('MxxIR.IterRuns'):
            return f'({m}.iterRuns_mono {h} (fun _ _ _ hs ↦ {m}.{sub_name(atom)}.bridge hs))'
        raise ValueError(atom)

    def signature(name, header, kind):
        """The arguments of `name` on the average-case and worst-case sides, and the binders."""
        args = top_args(header[len(f'{kind} {name} '):-len(': Prop :=')])
        avg_args, w_args, binders = [], [], []
        for k, a in enumerate(args):
            n, ty = a.split(' : ', 1)
            n = f'x{k}' if n == '_' else n
            if ty == 'MxxRuntime.SampleTape':
                avg_args.append('tape')
                binders.append('tape')
            elif ty == 'List Nat':
                avg_args.append('path')
                binders.append('path')
            elif ty == 'Params':
                avg_args.append('params')
                w_args.append("params'")
                binders += ['params', "params'"]
            else:
                avg_args.append(n)
                w_args.append(n)
                binders.append(n)
        return avg_args, w_args, binders

    def theorem(name, kind, header, script):
        avg_args, w_args, binders = signature(name, header, kind)
        return f'''
theorem {m}.{name}.bridge {{{' '.join(binders)}}}
    (h : {m}.{name} {' '.join(avg_args)}) : W_{m}.{name} {' '.join(w_args)} := by
''' + '\n'.join(script) + '\n'

    def constraint(name, h):
        """The bridge of constraint abbrev call `h` of `name`."""
        return f'({m}.{name}.bridge {h})' if 'tape' in B[name][1] else h

    for name, (kind, header, lines) in B.items():
        if not header.endswith(': Prop :='):
            continue
        counter[0] = 0
        if kind == 'abbrev' and re.search(r'\.constraints_\d+$', name):
            if 'tape' not in header:
                continue
            atoms = [l.strip() for l in lines]
            for a in atoms[:-1]:
                assert a.endswith(' ∧'), a
            atoms = [a[:-2] if a.endswith(' ∧') else a for a in atoms]
            hs = [fresh('a') for _ in atoms]
            terms = []
            for a, ha in zip(atoms, hs):
                c = re.match(r'(\w+)\.(constraints_\d+) ', a)
                terms.append(constraint(f'{c.group(1)}.{c.group(2)}', ha) if c else atom_term(a, ha))
            out.append(theorem(name, kind, header, [f'  obtain ⟨{", ".join(hs)}⟩ := h',
                                                     f'  exact ⟨{", ".join(terms)}⟩']))
            continue
        if kind != 'def':
            continue
        script = [f'  delta W_{m}.{name}']
        body = '\n'.join(lines)
        last = lines[-1].strip()
        cc = re.match(r'(\w+)\.(constraints_\d+) ', last)
        if lines[0].strip().startswith('∃ witness'):
            fields = [l.split(' : ')[0].strip() for l in B[f'{name}.Witness'][2]]
            bl = B[f'{name}.body'][2][-1].strip()
            cc = re.match(r'(\w+)\.(constraints_\d+) ', bl)
            witness = '⟨' + ', '.join(f'wit.{f}' for f in fields) + '⟩'
            script += ['  obtain ⟨wit, hb⟩ := h',
                       f'  exact ⟨{witness}, {constraint(f"{cc.group(1)}.{cc.group(2)}", "hb")}⟩']
        else:
            assert cc, (name, last)
            xs = [fresh('x') for _ in range(len(re.findall(r'^\s*∃ \(', body, re.M)))]
            if xs:
                script += [f'  obtain ⟨{", ".join(xs + ["hb"])}⟩ := h',
                           f'  exact ⟨{", ".join(xs + [constraint(f"{cc.group(1)}.{cc.group(2)}", "hb")])}⟩']
            else:
                script.append(f'  exact {constraint(f"{cc.group(1)}.{cc.group(2)}", "h")}')
        out.append(theorem(name, kind, header, script))
    open(f'{avg}/bridge/Bridge_{m}.lean', 'w').write('\n'.join(out))

# The bridge of `Runs`.
cl = open(f'{avg}/generated/Claim.lean').read()
runs = cl[cl.index('def Runs'):cl.index('/-- Executions whose')]
conj = [l.strip() for l in runs.split('\n')[2:] if l.strip()]
hs, terms = [], []
for k, c in enumerate(conj):
    h = f'r{k}'
    hs.append(h)
    r = re.match(r'(\w+)\.generatedRoot ', c)
    terms.append(f'{r.group(1)}.generatedRoot.bridge {h}' if r else h)
imports = ''.join(f'import Bridge_{m}\n' for m in modules)
open(f'{avg}/bridge/BridgeRuns.lean', 'w').write(f'''{imports}import W_Claim

/-! Every average-case run is a worst-case run (written by `generator/gen_bridge.py`). -/

set_option maxRecDepth 65536
set_option maxHeartbeats 64000000

theorem GeneratedClaim.runs_bridge {{hashModel : MxxRuntime.HashModel}}
    {{external : GeneratedClaim.ExternalInputs}} {{tape : MxxRuntime.SampleTape}}
    {{execution : GeneratedClaim.Execution}}
    (h : GeneratedClaim.Runs hashModel external tape execution) :
    W_GeneratedClaim.Runs hashModel external execution := by
  delta W_GeneratedClaim.Runs
  obtain ⟨{", ".join(hs)}⟩ := h
  exact ⟨{", ".join(terms)}⟩
''')
