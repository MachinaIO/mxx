"""Write `Modular_Part_I_Assembly.lean`: `Part_I_Assembly.lean`'s `partAssembly` (same statement,
in namespace `RefreshTopProof.Part_I.Modular`) over the facts modules of the export
(`generated/Stage_<phase>_Facts.lean`, exported with `MXX_EXPONENT_LUT_REFRESH_TOP_LEAN_FACTS=1`).

The source proof destructures the three stage bodies (`extract_lets`, `obtain` of every
fragment), so every step runs in a context of thousands of `let` variables. Here each body
hypothesis instead becomes `F : S.Facts ctx y` (`S.facts`), where `ctx` is a generated definition
`ctx_<hyp>` of the stage's inputs and witness over the theorem's variables. A producer's output
equation `hout : y = S.outs ctx` is rewritten into its consumers' hypotheses, so a consumer's input
wire unfolds to the producer's wire and cross-stage equations stay `rfl`. The proof steps are
the source's, with every destructured name replaced by its wire constant or fact field
(`modular.destructuring`, by position).

With `--nodes`, every top-level `have` step is a `node_have` (`ModularGlue.lean`): its own
auxiliary theorem over the variables it uses, and the module is `Modular_Part_I_AssemblyNodes`.

Usage: `python3 gen_modular_assembly.py <parameter directory> <output directory> [--nodes]`.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
from modular import FactsScope, balanced_args, destructuring, rename

d, out_dir = sys.argv[1], sys.argv[2]
NODES = '--nodes' in sys.argv[3:]
MODULE = 'Modular_Part_I_AssemblyNodes' if NODES else 'Modular_Part_I_Assembly'
SOURCE = open(f'{d}/Part_I_Assembly.lean').read()
NS = 'RefreshTopProof.Part_I.Modular'
HYPS = ['hpre', 'henc', 'hon']

imports = re.findall(r'^import \S+$', SOURCE, re.M)
head_end = SOURCE.index('set_option maxHeartbeats 0 in\nset_option maxRecDepth 65536 in\ntheorem partAssembly')
opens = re.search(r'^open .*$', SOURCE, re.M).group(0)
prelude = SOURCE[SOURCE.index(opens) + len(opens):head_end].strip() + '\n\n'
proof_start = SOURCE.index(':= by\n', head_end) + len(':= by\n')
header = SOURCE[head_end:proof_start]
proof = SOURCE[proof_start:SOURCE.index('\nend RefreshTopProof.Part_I')]


def binder_groups(text):
    """The binder groups `(…)`, `{…}`, `[…]` of a theorem header, in order."""
    text = text[text.index('theorem partAssembly') + len('theorem partAssembly'):]
    groups, depth, cur = [], 0, ''
    for ch in text:
        if depth == 0 and ch == ':':
            break
        if ch in '({[':
            depth += 1
        if depth > 0:
            cur += ch
        if ch in ')}]':
            depth -= 1
            if depth == 0:
                groups.append(' '.join(cur.split()))
                cur = ''
    return groups


groups = binder_groups(header)
binder = {}
for g in groups:
    if ' : ' not in g:
        continue
    names, ty = g[1:-1].split(' : ', 1)
    for n in names.split():
        binder[n] = (g[0], ty)

# The stage hypotheses: relation, its arguments before the inputs, inputs, output variable.
stages = {}
for h in HYPS:
    rel, *args = balanced_args(binder[h][1])
    stages[h] = dict(rel=rel, env=args[:-2], inputs=args[-2], out=args[-1])
scopes = {}
for h, s in stages.items():
    module, scope = s['rel'].split('.', 1)
    scopes[h] = FactsScope(f'{d}/generated', module, scope)
    s['scope'] = scopes[h]
producer = {s['out']: h for h, s in stages.items()}

# `subst`ed equations `x = value` of the source prologue
first = proof.split('\n', 1)[0]
assert first.startswith('  subst '), first
substs = {}
for h in first.split()[1:]:
    lhs, rhs = binder[h][1].split(' = ', 1)
    substs[lhs] = rhs

params = ['i0', 'model']


def free(text):
    return [n for n in binder if n not in HYPS and n not in params and re.search(rf'(?<![\w.]){re.escape(n)}(?![\w])', text)]


ctx_params = {}
defs = []
for h, s in stages.items():
    inputs = s['inputs']
    # the prologue's `subst`s come first: the context reads their values
    for x, value in substs.items():
        inputs = re.sub(rf'(?<![\w.]){x}(?![\w])', lambda _m, value=value: f'({value})', inputs)
    ps = []
    for y, ph in producer.items():
        if re.search(rf'(?<![\w.]){y}(?![\w])', inputs):
            ps += [p for p in ctx_params[ph] if p not in ps]
            # the producer's outputs at its context (over the same parameter names)
            outs = f'({stages[ph]["scope"].qual}.outs (ctx_{ph} {" ".join(params + ctx_params[ph])}))'
            inputs = re.sub(rf'(?<![\w.]){y}(?![\w])', lambda _m, outs=outs: outs, inputs)
    ps += [n for n in free(inputs) if n not in ps and n not in producer]
    ps.append(f'w_{h}')
    ctx_params[h] = ps
    ctx_type = f'{s["scope"].qual}.Ctx {" ".join(s["env"])}'
    bs = []
    for n in ps:
        if n.startswith('w_'):
            bs.append(f'({n} : {stages[n[2:]]["scope"].qual}.Witness)')
        else:
            bs.append(f'({n} : {binder[n][1]})')
    defs.append(f'/-- The inputs and witness of `{h}`. -/\n'
                f'noncomputable def ctx_{h} (i0 : Nat) [Fact (i0 < 8)] (model : HashModel) {" ".join(bs)} :\n'
                f'    {ctx_type} :=\n  ⟨{inputs}, w_{h}⟩\n')


def ctx_call(h):
    return f'(ctx_{h} {" ".join(params + ctx_params[h])})'


# Names bound by the source destructuring.
names = {}
for h, s in stages.items():
    fname = 'F' + h[1:]
    names.update(destructuring(proof, h, s['scope'], ctx_call(h), fname))

# Split the source proof: the destructuring prologue (up to the online `obtain` chain) and steps.
lines = proof.split('\n')
last = max(i for i, line in enumerate(lines) if re.match(rf'  obtain ⟨.*⟩ := {HYPS[-1]}$', line))
steps = '\n'.join(lines[last + 1:])
out = {h: f'F{h[1:]}.c{len(s["scope"].fragments) - 1}.hout' for h, s in stages.items()}
prologue = [first]
for i, h in enumerate(HYPS):
    s = stages[h]
    prologue.append(f'  obtain ⟨w_{h}, {h}⟩ := {h}')
    prologue.append(f'  have F{h[1:]} := {s["scope"].qual}.facts {ctx_call(h)} {h}')
    consumers = [c for c in HYPS[i + 1:] if re.search(rf'(?<![\w.]){s["out"]}(?![\w])', stages[c]['inputs'])]
    if consumers:
        prologue.append(f'  rw [{out[h]}] at {" ".join(consumers)}')
# The online outputs: the source `subst`s their equation before closing the goal.
houts = re.findall(r'(hout\w*)⟩ := ' + HYPS[-1], proof)
steps = steps.replace(f'  subst {houts[0]}', f'  rw [{out[HYPS[-1]]}]')
steps = rename(steps, names)
if NODES:
    # a step leaving named goals (`?h`) for later `case`s stays a `have`
    steps = re.sub(r'^  have (?![^\n]*\?[A-Za-z_])', '  node_have ', steps, flags=re.M)
body = '\n'.join(prologue) + '\n' + steps

facts_imports = sorted({f'import {s["scope"].module}_Facts' for s in stages.values()})
text = ('\n'.join(imports + facts_imports + (['import ModularGlue'] if NODES else [])) +
        f'\n\n/-! `partAssembly` over the facts modules (written by `generator/gen_modular_assembly.py`). -/\n\n'
        f'namespace {NS}\n\n{opens}\n\n{prelude}' + '\n'.join(defs) + '\n' + header +
        body + f'\n\n#print axioms partAssembly\n\nend {NS}\n')
open(f'{out_dir}/{MODULE}.lean', 'w').write(text)
print('wrote', f'{out_dir}/{MODULE}.lean', len(text))
