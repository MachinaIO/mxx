"""Write `Gen_Requirement.lean`: the facts the requirement module checks, for any parameter set.

The requirement module (`generated/Requirement_0.lean`) checks its conditions with sequential
loops. A Boolean loop step conjoins conditions to its state, `next = decide ((cur:1/0) + (b:1/0)
= 2)` with `b = decide p`, or runs an inner Boolean loop from its state; an integer loop step adds
a term to its state (`next = cur + t`, e.g. a row of a one-hot selector). The root chains the
loops and conjoins its own conditions; its output is the last result.

This script reads the module's scope tree (`xmodel.parse_module`) and writes, without unrolling
any loop:

* for every integer step scope `S`: `sum_S : S … (cur, inputs) next → next = cur + t`;
* for every Boolean step scope `S`: `step_S : S … (cur, inputs) next → ReqStep.Spec P n cur next`,
  `P n` the conjunction of the step's conditions (inner Boolean loops as `∀ k < count, Q k`,
  inner integer loops as `init + ∑ k ∈ Finset.range count, t k`);
* `structure Facts` (one field per condition and per loop of the root) and
  `requirement_facts : generatedRoot params inputs out → out = true → Facts inputs…`.

Every value is the export's own term: family reads become `famAt f i` (`GenReq.famGet_ofNat` /
`famGet_famAt`), table selections `MxxIR.selectInt i table`, literal sums are folded. Loop facts
come from `GenIface.ReqAll.sem` / `GenIface.seq_sum` with the child's step theorem as the
`∀`-step hypothesis.

Usage: `python3 gen_requirement2.py <parameter directory> [--out FILE] [--module Requirement_0]
[--gdir DIR] [--namespace NS]` (default output `<d>/Gen_Requirement.lean`, module read from
`<d>/generated/<module>.lean`).
"""
import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import xmodel  # noqa: E402
from xmodel import app_split, strip_parens, tuple_items  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('d')
ap.add_argument('--out')
ap.add_argument('--module', default='Requirement_0')
ap.add_argument('--gdir')
ap.add_argument('--namespace', default='GenRequirement')
a = ap.parse_args()
d = a.d
gdir = a.gdir or f'{d}/generated'
MOD = a.module
out_path = a.out or f'{d}/Gen_Requirement.lean'
scopes = xmodel.parse_module(gdir, MOD)


def fail(msg):
    sys.exit(f'gen_requirement2: {msg}')


def check(cond, msg):
    if not cond:
        fail(msg)


WIRE = re.compile(r'\bw_\d+_\d+\b')
BINDER = re.compile(r'\bi_\d+\b')
LIT = re.compile(r'^-?\d+$')
IDENT = re.compile(r'^[A-Za-z_][A-Za-z_0-9.]*$')


def paren(s):
    s = s.strip()
    if (LIT.match(s) and not s.startswith('-')) or IDENT.match(s):
        return s
    if s[0] in '([' and xmodel.matching(s, 0) == len(s) - 1:
        return s
    return f'({s})'


def fold(s):
    """Fold literal integer arithmetic `(a + b)`, `(a - b)`, `(a * b)`."""
    pat = re.compile(r'\((-?\d+) ([+\-*]) (-?\d+)\)')
    while True:
        m = pat.search(s)
        if not m:
            return s
        x, op, y = int(m.group(1)), m.group(2), int(m.group(3))
        v = x + y if op == '+' else x - y if op == '-' else x * y
        s = s[:m.start()] + (str(v) if v >= 0 else f'({v})') + s[m.end():]


def subst(text, mapping):
    """Simultaneous replacement of identifiers by (parenthesized) terms."""
    if not mapping:
        return text
    pat = re.compile(r'(?<![\w.])(' + '|'.join(re.escape(k) for k in sorted(mapping, key=len, reverse=True))
                     + r')(?![\w])')
    return pat.sub(lambda m: paren(mapping[m.group(1)]), text)


def short(name):
    return re.sub(r'^(?:sequential_|parallel_)+generatedRoot_?', '', name) or 'root'


# ---------------------------------------------------------------------------------------------
# Scope analysis

class Info:
    pass


def input_index(term):
    """`inputs.2.2.1` -> 2; `inputs.1` -> 0; `inputs` -> 0."""
    t = term.strip()
    if t == 'inputs':
        return 0
    m = re.match(r'^inputs((?:\.2)*)(\.1)?$', t)
    check(m, f'input projection {term}')
    return m.group(1).count('.2')


INFO = {}


def analyse(sc, is_root):
    """Variables, wires and defining atoms of scope `sc`."""
    I = Info()
    I.sc = sc
    I.root = is_root
    I.id = len(INFO)
    binders = [n for n, t in sc.sig if t == 'Nat']
    I.nbind = len(binders)
    I.own = None if is_root else f'n{I.id}'
    I.bvars = [f'j{p}' for p in range(I.nbind - (0 if is_root else 1))] + ([] if is_root else [I.own])
    I.bmap = {b: v for b, v in zip(binders, I.bvars) if b != '_'}
    ncomp = len(sc.inputs)
    if is_root:
        I.cvars = [f'x{c}' for c in range(ncomp)]
        I.state = None
    else:
        I.state = sc.inputs[0]
        check(I.state in ('Bool', 'Int'), f'{sc.name}: state type {I.state}')
        I.cvars = ['cur'] + [f'x{c}' for c in range(1, ncomp)]
    I.wire = {}
    for nm, w in sc.wires.items():
        if w.src == 'input':
            I.wire[nm] = ('input', I.cvars[input_index(w.term)])
        elif w.src == 'let':
            I.wire[nm] = ('let', w.term)
        else:
            I.wire[nm] = ('var', nm)
    I.exists = [nm for nm, w in sc.wires.items() if w.src == 'exists']
    I.witness = [f for f, _ in sc.witness]
    I.defn = {}       # wire -> (kind, atom index): reads, selections, integer loops
    I.loops = {}      # final wire -> atom index (Boolean loops)
    for j, at in enumerate(sc.atoms):
        h = at.head
        if h in ('MxxRuntime.familyGetDynamic', 'MxxRuntime.familyGetStatic'):
            I.defn[strip_parens(at.args[2])] = ('read', j)
        elif h == 'MxxRuntime.select':
            I.defn[strip_parens(at.args[2])] = ('select', j)
        elif at.loop is not None:
            check(at.loop.kind == 'seq', f'{sc.name}: loop kind {at.loop.kind}')
            fin = strip_parens(at.loop.final)
            st = scopes[at.loop.child].inputs[0]
            if st == 'Int':
                I.defn[fin] = ('sum', j)
            elif st == 'Bool':
                I.loops[fin] = j
            else:
                fail(f'{sc.name}: loop state {st}')
        elif h in ('cmp', 'eq', 'out', 'true'):
            pass
        else:
            fail(f'{sc.name}: unexpected atom {h}')
    return I


def value(I, w, deep, seen=()):
    """The term of wire `w`: lets inlined; `var` wires by name (shallow) or by their defining
    atom's value (deep)."""
    check(w not in seen, f'cyclic wire {w}')
    kind, t = I.wire[w]
    if kind == 'input':
        return t
    if kind == 'let':
        return expr(I, t, deep, seen + (w,))
    if deep and w in I.defn:
        return defn_value(I, w, True, seen + (w,))
    return t


def expr(I, t, deep, seen=()):
    t = WIRE.sub(lambda m: paren(value(I, m.group(0), deep, seen)), t)
    t = BINDER.sub(lambda m: I.bmap.get(m.group(0), m.group(0)), t)
    return fold(t)


def index_form(I, idx, deep, seen=()):
    """('lit', n) for a literal, ('nat', v) for `Int.ofNat v`, else ('int', term)."""
    s = strip_parens(idx)
    if LIT.match(s) and not s.startswith('-'):
        return 'lit', s
    v = strip_parens(expr(I, s, deep, seen))
    m = re.match(r'^Int\.ofNat ([A-Za-z_][A-Za-z_0-9]*)$', v)
    if m:
        return 'nat', m.group(1)
    return 'int', v


def defn_value(I, w, deep, seen=()):
    kind, j = I.defn[w]
    at = I.sc.atoms[j]
    if kind == 'read':
        fam = expr(I, strip_parens(at.args[0]), deep, seen)
        f, iv = index_form(I, at.args[1], deep, seen)
        if f in ('nat', 'lit'):
            return f'famAt {paren(fam)} {iv}'
        return f'famAt {paren(fam)} {paren(iv)}.toNat'
    if kind == 'select':
        idx = expr(I, strip_parens(at.args[0]), deep, seen)
        tbl = expr(I, at.args[1], deep, seen)
        return f'MxxIR.selectInt {paren(idx)} {tbl}'
    lp = at.loop
    ch = INFO[lp.child]
    init = expr(I, strip_parens(lp.init), deep, seen)
    t = subst(ch.T, child_mapping(I, at, ch, deep, seen))
    return f'{paren(init)} + ∑ {ch.own} ∈ Finset.range {lp.count}, {paren(t)}'


def child_mapping(I, at, ch, deep, seen=()):
    """The child's free variables -> the parent's terms (index arguments and data inputs)."""
    m = re.search(r'=> (.*) next\) \(Int\.toNat', at.text, re.S)
    check(m, f'loop call in {at.text[:80]}')
    head, args = app_split(m.group(1) + ' next')
    idx_args = args[1:-2]
    check(len(idx_args) == ch.nbind, f'{head}: index arity')
    mapping = {}
    for v, x in zip(ch.bvars, idx_args):
        x = strip_parens(x)
        if x == 'i':
            check(v == ch.own, f'{head}: loop index position')
            continue
        mapping[v] = I.bmap.get(x, x) if BINDER.fullmatch(x) else expr(I, x, deep, seen)
    items = tuple_items(args[-2])
    check(strip_parens(items[0]) == 'current', f'{head}: state first')
    for v, x in zip(ch.cvars[1:], items[1:]):
        mapping[v] = expr(I, strip_parens(x), deep, seen)
    return mapping


# ---------------------------------------------------------------------------------------------
# The Boolean chain of a step / the root

def chain(I, w):
    """Links from wire `w` down to the base: [('and', fact) | ('loop', atom j)], base."""
    links = []
    while True:
        kind, t = I.wire[w]
        if kind == 'input':
            check(t == 'cur', f'{I.sc.name}: Boolean input {w} is not the state')
            return links, 'cur'
        if kind == 'let':
            s = strip_parens(t)
            if strip_parens(expr(I, s, False)) in ('decide (0 = 0)', 'true'):
                return links, 'true'
            m = re.match(r'^decide \((w_\d+_\d+) = (w_\d+_\d+|2)\)$', s)
            check(m, f'{I.sc.name}: Boolean wire {w} := {t}')
            check(m.group(2) == '2' or strip_parens(value(I, m.group(2), False)) == '2',
                  f'{I.sc.name}: {w} is not a conjunction')
            ks, ts = I.wire[m.group(1)]
            mm = re.match(r'^\((w_\d+_\d+) \+ (w_\d+_\d+)\)$', ts.strip()) if ks == 'let' else None
            check(mm, f'{I.sc.name}: conjunction sum {m.group(1)}')

            def bool_of(x):
                kx, tx = I.wire[x]
                mx = re.match(r'^if (w_\d+_\d+) then 1 else 0$', strip_parens(tx)) if kx == 'let' else None
                check(mx, f'{I.sc.name}: 1/0 of {x}')
                return mx.group(1)
            b, c = bool_of(mm.group(1)), bool_of(mm.group(2))
            kc, tc = I.wire[c]
            mc = re.match(r'^decide \((.*)\)$', strip_parens(tc)) if kc == 'let' else None
            check(mc, f'{I.sc.name}: condition {c} := {tc}')
            links.append(('and', strip_parens(expr(I, mc.group(1), True))))
            w = b
            continue
        if w in I.loops:
            j = I.loops[w]
            links.append(('loop', j))
            w = strip_parens(I.sc.atoms[j].loop.init)
            continue
        fail(f'{I.sc.name}: Boolean wire {w} has no definition')


def loop_fact(I, j):
    at = I.sc.atoms[j]
    ch = INFO[at.loop.child]
    return f'∀ {ch.own} < {at.loop.count}, {subst(ch.P, child_mapping(I, at, ch, True))}'


# ---------------------------------------------------------------------------------------------
# Proof script

def destruct(I):
    """`obtain` lines naming the wires and atoms; the atom hypothesis names."""
    sc = I.sc
    names = {}
    for j, at in enumerate(sc.atoms):
        names[j] = '-' if at.head in ('cmp', 'eq', 'true') else 'hout' if at.head == 'out' else f'a{j}'
    frags = [[names[j] for j, at in enumerate(sc.atoms) if at.frag == k]
             for k in sorted({at.frag for at in sc.atoms})]
    if sc.kind == 'record':
        L = [f'  obtain ⟨⟨{", ".join(I.witness)}⟩, hc⟩ := h']
    elif I.exists:
        L = [f'  obtain ⟨{", ".join(I.exists)}, hc⟩ := h']
    else:
        L = ['  have hc := h']
    # Peel one conjunct at a time with a two-field `obtain`, which also drops the consumed tail:
    # an `obtain` pattern over hundreds of conjuncts makes `rcases` slow, and tails left in the
    # context would all be rewritten by the later `subst`.
    def split(ns, tail, prefix):
        out = []
        for i, n in enumerate(ns):
            last = i == len(ns) - 1
            head = f'{prefix}_{i}' if isinstance(n, list) else ('-' if n == '-' else n)
            rest = f'{prefix}t{i + 1}'
            if last:
                if head != '-':
                    out.append(f'  have {head} := {tail}')
                if head != tail:
                    out.append(f'  clear {tail}')
            else:
                out.append(f'  obtain ⟨{head}, {rest}⟩ := {tail}')
            if isinstance(n, list):
                out += split(n, head, head)
            tail = rest
        return out
    L += split(frags[0] if len(frags) == 1 else frags, 'hc', 'hc')
    return L, names


def equations(I, names, rewrite=False):
    """`have` lines for every defined wire (shallow terms) and the Boolean loop specs. With
    `rewrite` the wires stay, and the equation names are returned for rewriting each fact."""
    L, subs = [], []
    for w, (kind, j) in I.defn.items():
        at = I.sc.atoms[j]
        rhs = defn_value(I, w, False)
        if kind == 'read':
            f, _ = index_form(I, at.args[1], False)
            pf = f'(GenReq.famGet_ofNat {names[j]}).2' if f == 'nat' else f'(GenReq.famGet_famAt {names[j]}).2.2'
        elif kind == 'select':
            pf = f'select_int {names[j]}'
        else:
            ch = INFO[at.loop.child]
            pf = (f'seq_sum (fun {ch.own} ↦ {subst(ch.T, child_mapping(I, at, ch, False))}) '
                  f'(fun _ _ _ hb ↦ sum_{ch.sc.name} hb) {names[j]}')
        L.append(f'  have e_{w} : {w} = {rhs} := {pf}')
        subs.append(f'e_{w}')
    # `subst` rewrites every hypothesis mentioning a wire: drop the atoms the equations consumed
    # first, and state the loop facts after it.
    used = sorted({names[j] for _, j in I.defn.values()} - {'-', 'hout'})
    if rewrite:
        for w, j in I.loops.items():
            ch = INFO[I.sc.atoms[j].loop.child]
            L.append(f'  have L_{w} := ReqAll.sem {names[j]} (fun _ _ _ hb ↦ step_{ch.sc.name} hb)')
        return L, subs
    if used:
        L.append(f'  clear {" ".join(used)}')
    if subs:
        L.append(f'  subst {" ".join(subs)}')
    for w, j in I.loops.items():
        ch = INFO[I.sc.atoms[j].loop.child]
        L.append(f'  have L_{w} := ReqAll.sem {names[j]} (fun _ _ _ hb ↦ step_{ch.sc.name} hb)')
    return L


def peel(I, links, hn):
    """`obtain` lines peeling the chain from `hn : final = true`; the fact names (base to
    final) and the last hypothesis (the base)."""
    L, cur = [], hn
    for k, (kind, x) in enumerate(links):
        nxt, f = f'hn{k + 1}', f'f{len(links) - 1 - k}'
        if kind == 'and':
            L.append(f'  obtain ⟨{nxt}, {f}⟩ := link_and {cur}')
        else:
            L.append(f'  obtain ⟨{nxt}, {f}⟩ := link_loop L_{strip_parens(I.sc.atoms[x].loop.final)} {cur}')
        cur = nxt
    return L, [f'f{k}' for k in range(len(links))], cur


def call(I, out):
    sc = I.sc
    if not I.cvars:
        tup = '()'
    elif len(I.cvars) == 1 and ' × ' not in (sc.input_type or ''):
        tup = I.cvars[0]
    else:
        tup = '(' + ', ('.join(I.cvars) + ', ()' + ')' * len(I.cvars)
    idx = ' '.join(I.bvars)
    return f'{MOD}.{sc.name} params {idx + " " if idx else ""}{tup} {out}'


def binders_text(I):
    bs = ['{params : ' + MOD + '.Params}']
    if I.bvars:
        bs.append('{' + ' '.join(I.bvars) + ' : Nat}')
    if not I.root:
        bs.append('{cur next : ' + I.state + '}')
    for v, t in zip(I.cvars, I.sc.inputs):
        if v != 'cur':
            bs.append('{' + v + ' : ' + t + '}')
    return ' '.join(bs)


# ---------------------------------------------------------------------------------------------
# Walk the tree (children first)

check('generatedRoot' in scopes, 'no generatedRoot')
order = []


def visit(n):
    if n in order:
        return
    for c in scopes[n].children:
        visit(c)
    order.append(n)


visit('generatedRoot')
blocks = []
for n in order:
    sc = scopes[n]
    I = analyse(sc, n == 'generatedRoot')
    INFO[n] = I
    if I.root:
        continue
    outs = [strip_parens(at.args[0]) for at in sc.atoms if at.head == 'out']
    check(len(outs) == 1, f'{n}: output equation')
    dl, names = destruct(I)
    if I.state == 'Int':
        kind, t = I.wire[outs[0]]
        m = re.match(r'^\((w_\d+_\d+) \+ (.*)\)$', t.strip()) if kind == 'let' else None
        check(m and I.wire[m.group(1)] == ('input', 'cur'), f'{n}: integer step is not `cur + t`')
        I.T = strip_parens(expr(I, m.group(2), True))
        body = dl + equations(I, names) + ['  exact hout']
        blocks.append(f'''/-- One step of `{n}`: it adds a term to the state. -/
theorem sum_{n} {binders_text(I)}
    (h : {call(I, 'next')}) :
    next = cur + {paren(I.T)} := by
''' + '\n'.join(body))
    else:
        links, base = chain(I, outs[0])
        check(base == 'cur' and links, f'{n}: chain base {base}, {len(links)} links')
        parts = [x if k == 'and' else loop_fact(I, x) for k, x in reversed(links)]
        I.P = ' ∧ '.join(paren(p) for p in parts) if len(parts) > 1 else parts[0]
        pl, facts, last = peel(I, links, 'hn')
        body = dl + equations(I, names) + ['  intro hn', '  rw [hout] at hn'] + pl + \
            [f'  exact ⟨{last}, {", ".join(facts)}⟩']
        blocks.append(f'''/-- One step of `{n}`: a true state stays true only if its conditions hold. -/
theorem step_{n} {binders_text(I)}
    (h : {call(I, 'next')}) :
    ReqStep.Spec (fun {I.own} ↦ {I.P}) {I.own} cur next := by
''' + '\n'.join(body))

# the root
I = INFO['generatedRoot']
sc = I.sc
outs = [strip_parens(at.args[0]) for at in sc.atoms if at.head == 'out']
check(len(outs) == 1, 'root output equation')
links, base = chain(I, outs[0])
check(base == 'true', f'root chain base {base}')
fields = []
for k, (kind, x) in enumerate(reversed(links)):
    if kind == 'and':
        fields.append((f'cond_{k}', x, 'a condition of the root'))
    else:
        ch = sc.atoms[x].loop.child
        fields.append((f'loop_{short(ch)}', loop_fact(I, x), f'every iteration of `{ch}`'))
params = ' '.join(f'({v} : {t})' for v, t in zip(I.cvars, sc.inputs))
fl = '\n'.join(f'  /-- {doc}. -/\n  {nm} : {t}' for nm, t, doc in fields)
dl, names = destruct(I)
pl, facts, last = peel(I, links, 'hn')
# The root has hundreds of loop atoms: `subst` would rewrite every one of them per equation, so
# the wires stay until the facts are peeled, and each fact is rewritten on its own.
eqs, subs = equations(I, names, rewrite=True)
rewrites = [f'  try simp only [{", ".join(subs)}] at {f}' for f in facts] if subs else []
root_body = dl + eqs + ['  have hn := ho', '  rw [hout] at hn'] + pl + rewrites + \
    [f'  exact ⟨{", ".join(facts)}⟩']

out = [f'''import GenReq
import {MOD}

/-! The facts the requirement module checks (written by `generator/gen_requirement2.py` from
`{MOD}.lean`): one theorem per loop scope (`sum_…`: the step adds a term; `step_…`: a true state
stays true only if the step's conditions hold) and `requirement_facts`: a true result gives every
condition of the root and every loop's conditions at every index (`Facts`). No loop is unrolled:
each loop fact comes from `GenIface.ReqAll.sem` / `GenIface.seq_sum` with the child's theorem as
the step hypothesis. -/

set_option maxRecDepth 65536
set_option linter.unusedVariables false
set_option linter.unusedSimpArgs false
-- The root theorem of a large requirement exceeds the default budget (one-off cost).
set_option maxHeartbeats 0

namespace {a.namespace}

open Mxx.Primitives MxxRuntime MxxExponentLut RefreshTopProof.GenIface
open scoped BigOperators

/-- The export's conjunction: a true `decide ((a:1/0) + (decide p:1/0) = 2)` gives `a` and `p`. -/
theorem link_and {{a : Bool}} {{p : Prop}} {{inst : Decidable p}}
    (h : decide (((if a = true then 1 else 0 : Int) + (if @decide p inst = true then 1 else 0)) = 2) = true) :
    a = true ∧ p := by
  cases a <;> by_cases hp : p <;> simp_all

/-- A requirement loop's true result gives a true initial state and the conditions. -/
theorem link_loop {{P : Nat → Prop}} {{c : Nat}} {{a b : Bool}} (h : ReqAll.Spec P c a b)
    (hb : b = true) : a = true ∧ ∀ i < c, P i := h hb

/-- A runtime table selection is `selectInt`. -/
theorem select_int {{table : List Int}} {{idx out : Int}} (h : MxxRuntime.select idx table out) :
    out = MxxIR.selectInt idx table := by
  obtain ⟨p, hp, rfl⟩ := h
  unfold MxxIR.selectInt
  have : idx.toNat = p.val := by omega
  rw [this]
  simp''']
out += blocks
out.append(f'''/-- The facts a true requirement result gives. -/
structure Facts {params} : Prop where
{fl}

/-- A true requirement result gives all its facts. -/
theorem requirement_facts {binders_text(I)} {{out : Bool}}
    (h : {call(I, 'out')}) (ho : out = true) : Facts {' '.join(I.cvars)} := by
''' + '\n'.join(root_body))
out.append(f'end {a.namespace}\n')
open(out_path, 'w').write('\n\n'.join(out))
print(f'wrote {out_path}: {len(order) - 1} loop scopes, {len(fields)} root facts', file=sys.stderr)
