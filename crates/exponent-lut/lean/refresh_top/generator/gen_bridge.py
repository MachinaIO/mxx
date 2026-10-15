"""Write the bridge from an average-case export to the worst-case relation of the same graph.

The average-case export of a graph is its worst-case export with every relation taking a sampling
tape and a path, and every sampler reading the tape (`gaussianSampleAt`, …). Each tape sampler
implies its worst-case sampler, so every average-case run is a worst-case run: a worst-case
correctness proof gives the average-case claim with failure probability zero.

This script writes, in the average-case parameter directory:
- `worst/`: the worst-case export, renamed (`Stage_x` → `W_Stage_x`, `Ideal` → `W_Ideal`,
  `Requirement_0` → `W_Requirement_0`, and the facts modules `<module>_Facts` alike), with
  `W_Claim.lean` holding its `Runs` over the average-case claim's `ExternalInputs` and
  `Execution`;
- `bridge/`: per relation module (not per facts module), `Stage_x.D.bridge : Stage_x.D … tape
  path … → W_Stage_x.D …` for every relation `D` whose signature takes a tape, and
  `BridgeRuns.lean` with `GeneratedClaim.runs_bridge`;
and deletes the `.lean` files of `worst/` and `bridge/` it no longer writes, and sets the roots
of the `<prefix>Generated`, `<prefix>Worst` and `<prefix>Bridge` libraries in `lakefile.toml`
(`<prefix>` the optional third argument, `RefreshTop` by default).

The bridge reads each relation as a proposition: `let` lines are kept by definitional unfolding;
`∃` binders (record witnesses `∃ witness : D.Witness`, scalar chains `∃ (x : T),`, gathered
values `∃ x_gathered : T,`) are opened and closed again; conjunctions are split at the top level;
a parallel or gather loop `∀ i : Fin n, …` maps its body; a sequential loop
`MxxIR.IterRuns (fun i current next => child …) …` maps its step by `iterRuns_mono`; a call of
a relation (or constraint fragment) taking a tape uses its bridge; a tape sampler uses its
`ExponentSampling` lemma; every other atom is the same proposition on both sides.

Partition modules retain the parent namespace. Each bridge imports its dependencies' bridges,
reads their declarations for resolving calls and witnesses, and proves only its own declarations.
Relation and facts partitions are both renamed on the worst-case side.

Usage: `python3 gen_bridge.py <average directory> <worst directory> [<library prefix>]`.
"""
import os
import re
import sys

from xmodules import module_sources, partition_base

avg, worst = sys.argv[1], sys.argv[2]
prefix = sys.argv[3] if len(sys.argv) > 3 else 'RefreshTop'
ROOTS = re.compile(r'\b(Stage_\w+|Ideal|Requirement_0)\b')
os.makedirs(f'{avg}/worst', exist_ok=True)
os.makedirs(f'{avg}/bridge', exist_ok=True)


def is_relation(name):
    base = partition_base(name)
    return bool(ROOTS.fullmatch(base)) and not base.endswith('_Facts')


modules = sorted(f[:-5] for f in os.listdir(f'{avg}/generated')
                 if f.endswith('.lean') and is_relation(f[:-5]))
written = {'worst': set(), 'bridge': set()}


def write(sub, name, text):
    open(f'{avg}/{sub}/{name}.lean', 'w').write(text)
    written[sub].add(name)


def renamed(text):
    text = ROOTS.sub(lambda m: 'W_' + m.group(1), text)
    return re.sub(r'\b((?:Ideal|Requirement_0)(?:_Facts)?_Part_\d+)\b',
                  lambda m: 'W_' + m.group(1), text)


for m in sorted(f[:-5] for f in os.listdir(f'{worst}/generated')
                if f.endswith('.lean') and is_relation(f[:-5])):
    write('worst', f'W_{m}', renamed(open(f'{worst}/generated/{m}.lean').read()))
# The worst-case export's facts modules (`<module>_Facts`, exported with
# `MXX_EXPONENT_LUT_REFRESH_TOP_LEAN_FACTS=1`), renamed alike: the modular proof reads them.
for f in sorted(os.listdir(f'{worst}/generated')):
    if f.endswith('.lean') and partition_base(f[:-5]).endswith('_Facts'):
        write('worst', f'W_{f[:-5]}', renamed(open(f'{worst}/generated/{f}').read()))

claim = open(f'{worst}/generated/Claim.lean').read()
imports = ''.join(f'import W_{m}\n' for m in modules)
body = claim[claim.index('def stage_0_params'):claim.index('/-- Every execution')]
write('worst', 'W_Claim', f'''import Claim
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


OPEN, CLOSE = '([{⟨', ')]}⟩'


def depths(s):
    """The bracket depth before each character of `s`."""
    out, d = [], 0
    for ch in s:
        if ch in CLOSE:
            d -= 1
        out.append(d)
        if ch in OPEN:
            d += 1
    return out


def strip_parens(e):
    """`e` without enclosing parentheses that span all of it."""
    e = e.strip()
    while e.startswith('(') and e.endswith(')'):
        d = depths(e)
        if any(d[k] == 0 for k in range(1, len(e) - 1)):
            break
        e = e[1:-1].strip()
    return e


def split_conj(e):
    """The top-level conjuncts of `e`; a conjunct opening an unparenthesized binder ends the list."""
    out, start, d = [], 0, depths(e)
    k = 0
    while k < len(e):
        if e[k] == '∧' and d[k] == 0:
            out.append(e[start:k].strip())
            start = k + 1
            if re.match(r'\s*[∃∀]', e[start:]):
                break
        k += 1
    out.append(e[start:].strip())
    return out


def binder(e):
    """Split `∃ x : T, rest` / `∃ (x : T), rest` / `∀ i : T, rest` into `(x, T, rest)`."""
    s = e[1:].lstrip()
    d = depths(s)
    comma = next(k for k, ch in enumerate(s) if ch == ',' and d[k] == 0)
    b = strip_parens(s[:comma])
    name, ty = b.split(' : ', 1)
    assert re.fullmatch(r'\w+', name.strip()), e[:200]
    return name.strip(), ty.strip(), s[comma + 1:].strip()


def expression(lines):
    """The proposition of a relation body without its `let` lines."""
    kept = [l.strip() for l in lines if not l.strip().startswith('let ')]
    return ' '.join(kept)


TAPE = re.compile(r'\btape\b')


def gen_module(m):
    text = open(f'{avg}/generated/{m}.lean').read()
    own = blocks(text)
    B = {}
    for _, source in module_sources(f'{avg}/generated', m):
        B.update(blocks(source))
    namespace = partition_base(m)
    dependencies = re.findall(r'^import (\w+)\s*$', text, re.M)
    imports = ''.join(f'import Bridge_{dep}\n' for dep in dependencies if dep in modules)
    out = [f'''{imports}import {m}
import W_{namespace}
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

    def bridged(name):
        """Whether relation `name` takes a tape (and so has a bridge)."""
        return name in B and 'MxxRuntime.SampleTape' in B[name][1]

    def term(e, h):
        """A proof of the worst-case form of proposition `e` from `h : e`."""
        e = strip_parens(e)
        if not TAPE.search(e):
            return h
        if e.startswith('∀'):
            v, _, rest = binder(e)
            return f'(fun {v} ↦ {term(rest, f"({h} {v})")})'
        if e.startswith('∃'):
            xs, wits, rest = [], [], e
            while rest.startswith('∃'):
                x, ty, rest = binder(rest)
                if ty.endswith('.Witness') and ty in B:
                    x = fresh('wit')
                    fields = [l.split(' : ')[0].strip() for l in B[ty][2]]
                    wits.append('⟨' + ', '.join(f'{x}.{f}' for f in fields) + '⟩')
                else:
                    wits.append(x)
                xs.append(x)
                rest = strip_parens(rest)
            parts = split_conj(rest)
            hs = [fresh('h') for _ in parts]
            terms = [term(p, hp) for p, hp in zip(parts, hs)]
            return f'(by obtain ⟨{", ".join(xs + hs)}⟩ := {h}; exact ⟨{", ".join(wits + terms)}⟩)'
        parts = split_conj(e)
        if len(parts) > 1:
            hs = [fresh('h') for _ in parts]
            terms = [term(p, hp) for p, hp in zip(parts, hs)]
            return f'(by obtain ⟨{", ".join(hs)}⟩ := {h}; exact ⟨{", ".join(terms)}⟩)'
        if e.startswith('MxxRuntime.gaussianSampleAt tape'):
            return f'(MxxExponentLut.gaussianSampleAt_sample (by decide) {h})'
        if e.startswith('MxxRuntime.uniformIntervalSampleAt tape'):
            return f'(MxxExponentLut.uniformIntervalSampleAt_sample (by decide) {h})'
        if e.startswith('MxxRuntime.trapdoorSampleAt '):
            return f'(MxxRuntime.trapdoorSampleAt_sample {h})'
        if e.startswith('MxxIR.IterRuns (fun'):
            s = e[len('MxxIR.IterRuns '):]
            d = depths(s)
            end = next(k for k in range(1, len(s)) if d[k] == 0)
            step = s[1:end]
            step = step[step.index('=>') + 2:]
            hs = fresh('hs')
            return f'({m}.iterRuns_mono {h} (fun _ _ _ {hs} ↦ {term(step, hs)}))'
        head = e.split(' ', 1)[0]
        if head.startswith(f'{namespace}.'):
            head = head[len(namespace) + 1:]
        if head in B and B[head][0] == 'abbrev' and head.endswith('.body'):
            return term(expression(B[head][2]), h)
        if bridged(head):
            return f'({namespace}.{head}.bridge {h})'
        raise ValueError(f'{m}: no bridge for atom {e[:300]}')

    def signature(name, header, kind):
        """The arguments of `name` on the average-case and worst-case sides, and the binders."""
        assert header.endswith(': Prop :='), header[:200]
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

    for name, (kind, header, lines) in own.items():
        if not header.endswith(': Prop :=') or not bridged(name):
            continue
        if kind == 'abbrev' and name.endswith('.body'):
            continue  # unfolded at its witness
        assert kind == 'def' or re.search(r'\.constraints_\d+$', name), (m, name)
        counter[0] = 0
        avg_args, w_args, binders = signature(name, header, kind)
        script = [] if kind == 'abbrev' else [f'  delta W_{namespace}.{name}']
        script.append(f'  exact {term(expression(lines), "h")}')
        out.append(f'''
theorem {namespace}.{name}.bridge {{{' '.join(binders)}}}
    (h : {namespace}.{name} {' '.join(avg_args)}) : W_{namespace}.{name} {' '.join(w_args)} := by
''' + '\n'.join(script) + '\n')
    write('bridge', f'Bridge_{m}', '\n'.join(out))


for m in modules:
    gen_module(m)

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
write('bridge', 'BridgeRuns', f'''{imports}import W_Claim

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

# Stale outputs of earlier exports.
for sub in ('worst', 'bridge'):
    for f in sorted(os.listdir(f'{avg}/{sub}')):
        if f.endswith('.lean') and f[:-5] not in written[sub]:
            os.remove(f'{avg}/{sub}/{f}')

# The library roots.
generated = sorted(f[:-5] for f in os.listdir(f'{avg}/generated') if f.endswith('.lean'))
roots = {f'{prefix}Generated': generated, f'{prefix}Worst': sorted(written['worst']),
         f'{prefix}Bridge': sorted(written['bridge'])}
lake = open(f'{avg}/lakefile.toml').read()
for lib, names in roots.items():
    lake, n = re.subn(rf'(name = "{lib}"\n(?:(?!\[\[).*\n)*?roots = )\[[^\]]*\]',
                      lambda mm: mm.group(1) + '[' + ', '.join(f'"{x}"' for x in names) + ']', lake)
    assert n == 1, lib
open(f'{avg}/lakefile.toml', 'w').write(lake)
