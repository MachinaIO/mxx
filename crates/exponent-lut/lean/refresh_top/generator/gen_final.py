#!/usr/bin/env python3
"""Generate the online final combination of one instance (WP G): `Gen_Final_Certs.lean` and
`Gen_Final_Inst.lean`.

Recognizes, by atom kind and data flow only (never by wire ids or `sel_`/`mod_` names), the end of
the online instance body: starting from the final `modulusSwitchRuns` of a `matrixSub`, its
centered rebase and the CRT term `· matrixPolynomial [K]`, the half row (a `concatColumns` chain of
one `allCoefficients h`), `hint × preimage`, the smudge and fresh loop outputs, the combination
`[g | c₁] D - cₓ Y`, the decomposition of `Q_q · A'` (with `A'` the reduction of the instance's
column block of the caller's `A'`, a `sliceMatrix` of an instance input),
and the action chain `gadget -> rebase -> reduce -> ·s₁ -> ·s₂ -> rebase -> ·c' -> digits -> rebase`.
The instance constants (`Q_q`, `c'`, `s₁`, `s₂`, `K`, `h`) and moduli are copied as terms from the
export (abbreviations are qualified, `i_0` becomes `i0`), so inline `selectInt` tables work alike.

Writes:
* `Gen_Final_Certs.lean`: certificates of the instance constants (`interval_cases i0 <;> decide`),
  e.g. `R = R / q * q`, `Q_q = R / q`, `(R / M_x) c' = Q_q k`, `k s₂ ≡ 1`, `s₁ ≡ Δ (mod q)`, layouts.
* `Gen_Final_Inst.lean`: theorem `GenFinalInst.final_inst`: from the export's atoms of the segment
  (named by their record fields) and the role facts of PART_IFACE.md "Final combination (G)", the
  CRT term `TermOf q K term (DecodeLoops.decodeValue inputs G E j)`.

Usage: gen_final.py <parameter dir> [--out DIR] [--decode FILE] [--text-only]
(`<parameter dir>/generated` is the export; `--decode` the `TopDecodeLoops.lean` for `Δ` and the
decode types, default `<parameter dir>/TopDecodeLoops.lean`).
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import xmodel
from xmodel import app_split, strip_parens

class Fail(Exception):
    pass

def load_inst(gdir):
    sc = xmodel.load(gdir, 'Stage_online_eval')
    root = sc[xmodel.roots(sc)[0]]
    lp = [a for a in root.atoms if a.loop and a.loop.kind == 'par']
    inst = sc[lp[0].loop.child]
    return sc, root, lp[0], inst

def term_of(sc, w):
    x = sc.wires.get(w)
    if x is None or x.src != 'let':
        return None
    t = strip_parens(x.term)
    # strip a type ascription `(e : T)`
    parts = xmodel.split_top(t, ' : ')
    if len(parts) == 2:
        t = strip_parens(parts[0])
    return t

def let_app(sc, w, head):
    t = term_of(sc, w)
    if t is None:
        raise Fail(f'{w} is not a let (wanted {head})')
    h, a = app_split(t)
    if h.split('.')[-1] != head:
        raise Fail(f'{w} := {t[:80]} is not {head}')
    return [strip_parens(x) for x in a]

def producer(sc, w, kind):
    """The atom of `kind` whose last argument is `w` (the output wire)."""
    for a in sc.atoms:
        if a.kind == kind and a.args and strip_parens(a.args[-1]) == w:
            return [strip_parens(x) for x in a.args], a.field
    raise Fail(f'no {kind} producing {w}')

def atoms_of(sc, kind):
    return [([strip_parens(x) for x in a.args], a.field) for a in sc.atoms if a.kind == kind]

def poly_const(sc, w):
    """`w := (matrixPolynomial [c] : T)` -> c."""
    t = term_of(sc, w)
    h, a = app_split(t)
    if h.split('.')[-1] != 'matrixPolynomial':
        raise Fail(f'{w} not matrixPolynomial')
    inner = strip_parens(a[0])
    assert inner.startswith('[') and inner.endswith(']')
    items = xmodel.split_top(inner[1:-1], ', ')
    if len(items) != 1:
        raise Fail(f'{w} has {len(items)} coefficients')
    return strip_parens(items[0])

def loop_final_of(sc, w):
    for a in sc.atoms:
        if a.loop and a.loop.kind == 'seq':
            fin = xmodel.WIRE.findall(a.loop.final)
            if w in fin:
                return a.field, a.loop.child, fin.index(w)
    raise Fail(f'{w} is no loop output')

def mat_type(sc, w):
    t = sc.wires[w].type
    m = re.match(r'^Mxx\.Primitives\.ExactMatrix (.*) (\d+) (\d+) (\d+)$', t.strip())
    if not m:
        raise Fail(f'type of {w}: {t}')
    return dict(mod=strip_parens(m.group(1)), n=int(m.group(2)), rows=int(m.group(3)), cols=int(m.group(4)))

def recognize(gdir):
    sc, root, rootloop, inst = load_inst(gdir)
    S = inst
    R = {}
    sws = atoms_of(S, 'modulusSwitchRuns')
    cands = []
    for args, f in sws:
        zeta = args[-2]
        t = term_of(S, zeta)
        if t and app_split(t)[0].endswith('matrixSub'):
            cands.append((args, f))
    if len(cands) != 1:
        raise Fail(f'{len(cands)} final switches')
    (args, R['h_switch']) = cands[0]
    R['zeta'], R['r'] = args[-2], args[-1]
    rbs = [(a, f) for a, f in atoms_of(S, 'centeredRebaseRuns') if a[0] == R['r']]
    (a, R['h_lift']) = rbs[0]
    R['l'] = a[1]
    terms = [w for w in S.wires if term_of(S, w) and app_split(term_of(S, w))[0].endswith('matrixMulScalarRight')
             and strip_parens(app_split(term_of(S, w))[1][0]) == R['l']]
    R['term'] = terms[0]
    R['Kw'] = let_app(S, R['term'], 'matrixMulScalarRight')[1]
    R['K'] = poly_const(S, R['Kw'])
    Z, H = let_app(S, R['zeta'], 'matrixSub')
    R['Z'], R['H'] = Z, H
    # the half: H is the last of a concatColumns chain of one 1×1 allCoefficients wire
    cat, R['h_half'] = producer(S, H, 'concatColumns')
    base = cat[1]
    t = term_of(S, base)
    h, a = app_split(t)
    if not h.endswith('allCoefficients'):
        raise Fail('half is not allCoefficients')
    R['half_w'], R['half'] = base, strip_parens(a[0])
    HP, Ssum = let_app(S, Z, 'matrixSub')
    R['b'], R['P'] = let_app(S, HP, 'matrixMul')
    R['HP'] = HP
    CF, R['smudge'] = let_app(S, Ssum, 'matrixAdd')
    R['comb'], R['fresh'] = let_app(S, CF, 'matrixAdd')
    R['fresh_loop'] = loop_final_of(S, R['fresh'])
    R['smudge_loop'] = loop_final_of(S, R['smudge'])
    M1, M2 = let_app(S, R['comb'], 'matrixSub')
    R['gc'], R['D'] = let_app(S, M1, 'matrixMul')
    R['cx'], R['Y'] = let_app(S, M2, 'matrixMul')
    R['M1'], R['M2'] = M1, M2
    a, R['h_gc'] = producer(S, R['gc'], 'concatColumns')
    R['g'], R['c1'] = a[0], a[1]
    a, R['h_D'] = producer(S, R['D'], 'compactCenteredRebaseRuns')
    R['D_bound'], R['Dz'] = a[0], a[1]
    a, R['h_Dz'] = producer(S, R['Dz'], 'gadgetDecomposeRuns')
    R['backend'], R['base'], R['digits'], R['z'] = a[0], a[1], a[2], a[3]
    QqW, R['A1'] = let_app(S, R['z'], 'matrixMulScalarLeft')
    R['QqW'], R['Qq'] = QqW, poly_const(S, QqW)
    a, R['h_A1'] = producer(S, R['A1'], 'modulusReduceRuns')
    R['A0'] = a[0]
    a, R['h_slice'] = producer(S, R['A0'], 'sliceMatrix')
    R['slice_args'] = a
    a, R['h_Y'] = producer(S, R['Y'], 'compactCenteredRebaseRuns')
    R['Y_bound'], R['Yz'] = a[0], a[1]
    a, R['h_Yz'] = producer(S, R['Yz'], 'gadgetDecomposeRuns')
    R['base4'], R['digits4'], R['X'] = a[1], a[2], a[3]
    R['wr'], cW = let_app(S, R['X'], 'matrixMulScalarRight')
    R['cW'], R['c'] = cW, poly_const(S, cW)
    a, R['h_wr'] = producer(S, R['wr'], 'centeredRebaseRuns')
    w = a[0]
    # w = ((w0 · s_1) · s_2) … : the constant factors over the CRT prime
    facs = []
    while term_of(S, w) is not None:
        x, f = let_app(S, w, 'matrixMulScalarRight')
        facs.append((w, f, poly_const(S, f)))
        w = x
    facs.reverse()
    R['factors'] = facs
    R['w0'] = w
    a, R['h_w0'] = producer(S, w, 'modulusReduceRuns')
    R['Gp'] = a[0]
    a, R['h_Gp'] = producer(S, R['Gp'], 'centeredRebaseRuns')
    R['Gg'] = a[0]
    a, R['h_Gg'] = producer(S, R['Gg'], 'gadgetMatrixRuns')
    R['gbase'], R['gdigits'] = a[1], a[2]
    # moduli and dimensions from the wire types
    R['Rm'] = mat_type(S, R['zeta'])['mod']
    R['qm'] = mat_type(S, R['r'])['mod']
    R['Qm'] = mat_type(S, R['l'])['mod']
    R['Mxm'] = mat_type(S, R['X'])['mod']
    R['Qgm'] = mat_type(S, R['Gg'])['mod']
    R['n'] = mat_type(S, R['zeta'])['n']
    R['m'] = mat_type(S, R['zeta'])['cols']
    R['h'] = mat_type(S, R['b'])['cols']
    R['mx'] = mat_type(S, R['cx'])['cols']
    R['mg'] = mat_type(S, R['g'])['cols']
    R['S'] = rootloop.loop.count
    R['inst'] = S.name
    R['types'] = {w: S.wires[w].type for w in S.wires}
    return R


def recognize_full(gdir):
    R = recognize(gdir)
    sc, root, rootloop, inst = load_inst(gdir)
    return R, inst



# --------------------------------------------------------------------------- the preprocess decompositions

def comb_chain(S, D):
    """`D` = compact digits of `matrixMulScalarLeft (poly Qq) A1`, `A1` the reduction of a column
    block of the caller's `A'`."""
    a, hD = producer(S, D, 'compactCenteredRebaseRuns')
    Dz = a[1]
    a, hDz = producer(S, Dz, 'gadgetDecomposeRuns')
    base, digits, z = a[1], a[2], a[3]
    QqW, A1 = let_app(S, z, 'matrixMulScalarLeft')
    Qq = poly_const(S, QqW)
    a, hA1 = producer(S, A1, 'modulusReduceRuns')
    A0 = a[0]
    a, hslice = producer(S, A0, 'sliceMatrix')
    return dict(D=D, h_D=hD, Dz=Dz, h_Dz=hDz, base=base, digits=digits, z=z, Qq=Qq, A1=A1, h_A1=hA1,
                A0=A0, h_slice=hslice, slice_args=a, D_bound=producer(S, D, 'compactCenteredRebaseRuns')[0][0])


def action_chain(S, Y):
    a, hY = producer(S, Y, 'compactCenteredRebaseRuns')
    Yz = a[1]
    a, hYz = producer(S, Yz, 'gadgetDecomposeRuns')
    base4, digits4, X = a[1], a[2], a[3]
    wr, cW = let_app(S, X, 'matrixMulScalarRight')
    c = poly_const(S, cW)
    a, hwr = producer(S, wr, 'centeredRebaseRuns')
    w = a[0]
    facs = []
    while term_of(S, w) is not None:
        x, f = let_app(S, w, 'matrixMulScalarRight')
        facs.append((w, f, poly_const(S, f)))
        w = x
    facs.reverse()
    a, hw0 = producer(S, w, 'modulusReduceRuns')
    Gp = a[0]
    a, hGp = producer(S, Gp, 'centeredRebaseRuns')
    Gg = a[0]
    a, hGg = producer(S, Gg, 'gadgetMatrixRuns')
    return dict(Y=Y, h_Y=hY, Yz=Yz, h_Yz=hYz, base4=base4, digits4=digits4, X=X, wr=wr, h_wr=hwr, c=c,
                factors=facs, w0=w, h_w0=hw0, Gp=Gp, h_Gp=hGp, Gg=Gg, h_Gg=hGg, gbase=a[1], gdigits=a[2],
                Y_bound=producer(S, Y, 'compactCenteredRebaseRuns')[0][0])


def preprocess_links(gdir, R):
    """The preprocess instance's decompositions matching the online combination digits and action digits."""
    sc = xmodel.load(gdir, 'Stage_preprocess')
    root = sc[xmodel.roots(sc)[0]]
    lp = [a for a in root.atoms if a.loop and a.loop.kind == 'par'][0]
    P = sc[lp.loop.child]
    combs, acts = [], []
    for args, f in atoms_of(P, 'compactCenteredRebaseRuns'):
        out = args[-1]
        try:
            c = comb_chain(P, out)
            if (c['base'], c['digits']) == (R['base'], R['digits']):
                combs.append(c)
            continue
        except Fail:
            pass
        try:
            a = action_chain(P, out)
            if len(a['factors']) == 2 and (a['base4'], a['digits4']) == (R['base4'], R['digits4']):
                acts.append(a)
        except Fail:
            pass
    if len(combs) != 1 or len(acts) != 1:
        raise Fail(f'preprocess: {len(combs)} combination digits, {len(acts)} action digits')
    return P, combs[0], acts[0]


# --------------------------------------------------------------------------- the fresh error (errE)

WIRE = xmodel.WIRE

def bare(t):
    t = strip_parens(t)
    m = re.match(r'^(w_\d+_\d+)(?: \w+)?$', t)
    return m.group(1) if m else None

def tree(t):
    """A Lean tuple text as nested pairs (`(a, b, c)` = `(a, (b, c))`); leaves are strings."""
    t = strip_parens(t) if t.strip().startswith('(') and xmodel.matching(t.strip(), 0) == len(t.strip()) - 1 else t.strip()
    items = [x.strip() for x in xmodel.split_top(t, ', ')]
    if len(items) == 1:
        return items[0]
    def pair(it):
        return tree(it[0]) if len(it) == 1 else (tree(it[0]), pair(it[1:]))
    return pair(items)

def nav(tr, path):
    for p in path:
        if not isinstance(tr, tuple):
            raise Fail(f'projection {path} past a leaf')
        tr = tr[0] if p == '1' else tr[1]
    return tr

def find_path(tr, pred, path=()):
    if isinstance(tr, tuple):
        return find_path(tr[0], pred, path + ('1',)) or find_path(tr[1], pred, path + ('2',))
    return path if pred(tr) else None

def proj_path(term, prefix):
    rest = term[len(prefix):]
    return tuple(x for x in rest.split('.') if x)

def claim_tuple(gdir, module):
    text = open(f'{gdir}/Claim.lean').read()
    body = text.split('def Runs', 1)[1].split(':=\n', 1)[1].split('\n\n')[0]
    for c in xmodel.split_top(body.strip(), ' ∧\n'):
        c = c.strip()
        if c.startswith(module + '.generatedRoot '):
            head, args = app_split(c)
            return tree(args[-2]), args[-1]
    raise Fail('no ' + module)

class Stage:
    def __init__(self, gdir, module):
        self.sc = xmodel.load(gdir, module)
        self.root = self.sc['generatedRoot']
        self.loop = [a for a in self.root.atoms if a.loop and a.loop.kind == 'par'][0]
        self.inst = self.sc[self.loop.loop.child]
        self.claim, self.claim_out = claim_tuple(gdir, module)
        body = re.sub(r'^∀ \w+ : Fin \d+, ', '', strip_parens(self.loop.text))
        _, cargs = app_split(body)
        self.args, self.outs = tree(cargs[-2]), tree(cargs[-1])
        self.module = module

    def inst_input_path(self, w):
        x = self.inst.wires.get(w)
        if x is None or x.src != 'input':
            return None
        return proj_path(x.term, 'inputs')

    def claim_term_of_inst_input(self, w):
        p = self.inst_input_path(w)
        if p is None:
            raise Fail(f'{self.module}: {w} is not an instance input')
        rw = bare(nav(self.args, p))
        x = self.root.wires[rw]
        return nav(self.claim, proj_path(x.term, 'inputs'))

    def root_outputs(self):
        for k, (b, body) in sorted(self.root.frag_src.items()):
            m = re.search(r'outputs = (\(.*\))\s*$', body, re.S)
            if m:
                return tree(m.group(1))
        raise Fail('no root outputs')

    def inst_outputs(self):
        for k, (b, body) in sorted(self.inst.frag_src.items()):
            m = re.search(r'outputs = (\(.*\))\s*$', body, re.S)
            if m:
                return tree(m.group(1))
        raise Fail('no instance outputs')

def load_all(gdir):
    return None, Stage(gdir, 'Stage_online_eval'), Stage(gdir, 'Stage_preprocess')

def to_external(pre, term):
    """Resolve a Claim argument term of the online stage to an external input."""
    if term.startswith('external.'):
        return term
    m = re.match(r'^execution\.«stage_\d+»(\S*)$', term)
    if not m:
        raise Fail(f'unresolved {term}')
    rw = bare(nav(pre.root_outputs(), proj_path(m.group(1), '')))
    p = find_path(pre.outs, lambda leaf: bare(leaf) == rw)
    if p is None:
        raise Fail(f'{rw} is not an instance output')
    w = bare(nav(pre.inst_outputs(), p))
    if w is None or pre.inst_input_path(w) is None:
        raise Fail(f'preprocess output {rw} is not a pass-through')
    return pre.claim_term_of_inst_input(w)

def outputs_of(a):
    if a.loop:
        ws = WIRE.findall(' '.join(a.loop.outs))
        if a.loop.kind == 'seq':
            ws += WIRE.findall(a.loop.final or '')
        return set(w for w in ws if w)
    if a.args:
        w = bare(a.args[-1])
        return {w} if w else set()
    return set()

def cone(S, start):
    """Atoms of scope S in the backward data-flow cone of wire `start` (lets inlined)."""
    need, seen, atoms = {start}, set(), []
    changed = True
    while changed:
        changed = False
        for w in list(need):
            x = S.wires.get(w)
            if x is not None and x.src == 'let' and w not in seen:
                seen.add(w)
                need |= set(WIRE.findall(x.term))
                changed = True
        for a in S.atoms:
            if id(a) in seen or a.head in ('cmp', 'eq', 'out', 'true'):
                continue
            if outputs_of(a) & need:
                seen.add(id(a))
                atoms.append(a)
                ins = set(WIRE.findall(a.text)) - outputs_of(a)
                need |= ins
                changed = True
    return atoms

def req_facts(path):
    t = open(path).read()
    hdr = re.search(r'^structure Facts (.*) : Prop where', t, re.M).group(1)
    xs = re.findall(r'\((x\d+) : ', hdr)
    return t, xs

def fresh_params(gdir, roles_path, req_path, R):
    _, on, pre = load_all(gdir)
    roles = json.load(open(roles_path))
    n, m = R['n'], R['m']
    M = m * n
    S = on.inst
    cn = cone(S, R['fresh'])
    gathers = [a for a in cn if a.loop and a.loop.kind == 'gather']
    if not gathers:
        raise Fail('no gathers in the fresh cone')
    tables = []
    for a in gathers:
        idx = {bare(i) for _, i, _ in a.loop.gathered}
        if len(idx) != 1:
            raise Fail('gather reads several tables')
        tables.append(on_ext := to_external(pre, on.claim_term_of_inst_input(idx.pop())))
    count = gathers[0].loop.count
    if count % M:
        raise Fail(f'fresh gather count {count} not a multiple of {M}')
    W = count // M
    # requirement facts: the Req argument externals and the one-hot selectors (M members, W cells)
    rt, xs = req_facts(req_path)
    rq, _ = claim_tuple(gdir, 'Requirement_0')
    req_args = []
    while isinstance(rq, tuple):
        req_args.append(rq[0]); rq = rq[1]
    ext_of_x = dict(zip(xs, req_args))
    onehot = set()
    for mm in re.finditer(r'∀ \w+ < \d+, ∀ \w+ < (\d+), \(0 \+ ∑ \w+ ∈ Finset\.range (\d+), \(famAt \(famAt (x\d+) \w+\)', rt):
        if int(mm.group(1)) == M and int(mm.group(2)) == W:
            onehot.add(ext_of_x[mm.group(3)])
    # preprocess key cells: per-instance arguments = (table, selector)
    pairs = []
    for a in pre.inst.atoms:
        if not (a.loop and a.loop.kind == 'par' and a.loop.count == count):
            continue
        exts = []
        for x in a.loop.args:
            w = bare(x)
            if w and pre.inst_input_path(w) is not None:
                exts.append(pre.claim_term_of_inst_input(w))
        tab = [e for e in exts if e in tables]
        sel = [e for e in exts if e in onehot]
        if len(tab) == 1 and len(sel) == 1:
            pairs.append((sel[0], tab[0]))
    if sorted(t for _, t in pairs) != sorted(tables):
        raise Fail(f'selector/table pairing incomplete: {pairs} vs {tables}')
    # label exponents: the vector the mask's requirement reads at table entries
    lab = re.search(r'famAt (x\d+) \(famAt \(famAt x\d+ 0\)', rt)
    a_ext = ext_of_x[lab.group(1)]
    per = re.search(r'\) % (\d+)\) / (\d+)\)', rt)
    nprime = int(per.group(2))
    if int(per.group(1)) != 2 * nprime:
        raise Fail('mask period')
    # the sampler's LUT width P and table: online sampler member in the fresh cone; key table in preprocess
    rol_on = roles['Stage_online_eval']
    samp = [a for a in cn if a.loop and rol_on.get(a.loop.child, {}).get('role') == 'lut.member.sampler']
    if len(samp) != 1:
        raise Fail('fresh sampler member')
    P = branch_width(on.sc, samp[0].loop.child)
    rd = [a for a in cn if a.loop and rol_on.get(a.loop.child, {}).get('role') == 'lut.member.readout']
    nprime2 = branch_width(on.sc, rd[0].loop.child)
    if nprime2 != nprime:
        raise Fail(f'readout width {nprime2} vs mask period {nprime}')
    vals = key_table_values(pre, roles['Stage_preprocess'], M, P)
    g = [vals[2 * b + 1] for b in range(P // 2)]
    return dict(n=n, m=m, W=W, a=a_ext, pairs=pairs, P=P, nprime=nprime, g=g, B=max(abs(x) for x in g))

def scope_text(sc, name, seen=None):
    seen = seen if seen is not None else set()
    s = sc[name]
    txt = (s.def_src or '') + (s.body_src or '') + ''.join(b for _, b in s.frag_src.values())
    for c in s.children:
        if c in sc and c not in seen:
            seen.add(c)
            txt += scope_text(sc, c, seen)
    return txt

def branch_width(sc, name):
    L = {len(xmodel.split_top(m.group(1), ', ')) for m in
         re.finditer(r'ringAutomorphismRuns \(\(MxxIR\.selectInt \([^)]*\)\) \[([^\]]*)\]\)', scope_text(sc, name))}
    if len(L) != 1:
        raise Fail(f'LUT widths {L} in {name}')
    return L.pop()

def key_table_values(pre, roles, M, P):
    carrier = [a for a in pre.inst.atoms if a.loop and a.loop.count == M and
               roles.get(a.loop.child, {}).get('role') == 'lut.key-member.carrier']
    if len(carrier) != 1:
        raise Fail('fresh key carrier')
    need = {bare(x) for x in carrier[0].loop.args} - {None}
    tabs = [a for a in pre.inst.atoms if a.loop and a.loop.kind == 'par' and outputs_of(a) & need
            and roles.get(a.loop.child, {}).get('role') != 'lut.key-member.readout']
    if len(tabs) != 1:
        raise Fail('fresh key table')
    txt = scope_text(pre.sc, tabs[0].loop.child)
    for mm in re.finditer(r'let w_\d+_\d+ : Int := \(MxxIR\.selectInt \(\(Int\.ofNat i_\d+\)\) \[([^\]]*)\]\)', txt):
        it = [strip_parens(x) for x in xmodel.split_top(mm.group(1), ', ')]
        if len(it) == P and all(re.fullmatch(r'-?\d+', x) for x in it):
            return [int(x) for x in it]
    # inline form: `matrixPolynomial [(v)]` immediately followed by a `rotationPolynomial`
    lets = re.findall(r'let w_\d+_\d+ : [^\n]*?:= ([^\n]*)', txt)
    vals = []
    for a, b in zip(lets, lets[1:]):
        mm = re.match(r'^\(MxxRuntime\.matrixPolynomial \[\((-?\d+)\)\] : ', a)
        if mm and 'rotationPolynomial' in b:
            vals.append(int(mm.group(1)))
    if len(vals) != P:
        raise Fail(f'key table values {vals}')
    return vals


def errE_lean(d, ety, qd):
    """The body of `GenPart.errE` and the proof of `errE_bound` for the fresh parameters `d`."""
    n, m = d['n'], d['m']
    g = ', '.join(str(x) for x in d['g'])
    sel = ', '.join(f'GenIface.famAt (GenIface.famAt {s} j)' for s, _ in d['pairs'])
    tab = ', '.join(f'GenIface.famAt (GenIface.famAt {t} j)' for _, t in d['pairs'])
    body = (f'noncomputable def errE (external : GeneratedClaim.ExternalInputs) : {ety} := fun j ↦\n'
            f'  GenFinal.freshErr {n} {m} (GenFinal.samplerVal {d["P"]} {d["nprime"]} (GenFinal.listTable [{g}]))\n'
            f'    (GenFinal.prfZB {d["W"]} (GenIface.famAt {d["a"]})\n'
            f'      ![{sel}]\n      ![{tab}])')
    proof = ('  intro j c i\n  unfold errE\n'
             f'  exact_mod_cast GenFinal.freshErr_bound (n := {n}) (m := {m}) (Q := {qd}) (B := {d["B"]}) (by norm_num)\n'
             f'    (GenFinal.samplerVal_natAbs_le (P := {d["P"]}) (W\' := {d["nprime"]}) (g := GenFinal.listTable [{g}])\n'
             f'      (GenFinal.listTable_natAbs_le (by decide))) (by norm_num) _ c i')
    return body, proof

# --------------------------------------------------------------------------- emission

ABBR = re.compile(r'(?<![\w.])(sel_\d+|mod_\d+|modulus_\d+)\b')


def q(t, module='Stage_online_eval', prefix=''):
    """Qualify the export's abbreviations and rename the instance binder (and, for a second stage,
    prefix its wire names)."""
    t = ABBR.sub(module + r'.\1', t)
    if prefix:
        t = xmodel.WIRE.sub(lambda m: prefix + m.group(0), t)
    t = re.sub(r'\bi_0\b', 'i0', t)
    return t.replace('Mxx.Primitives.', '').replace('MxxRuntime.', '').replace('backend', 'Backend.backend')


def expand(S, t, depth=0):
    """Inline let wires (with their ascriptions) into a term."""
    def rep(m):
        w = m.group(0)
        x = S.wires.get(w)
        if x is not None and x.src == 'let' and depth < 40:
            return '(' + expand(S, x.term, depth + 1) + ')'
        return w
    return xmodel.WIRE.sub(rep, t)


def mt(R, w):
    return q(R['types'][w])


def decode_info(path):
    t = open(path).read()
    m = re.search(r'polynomialOfCoefficients inputs\.2\.1\) (-?\d+) \(decodeR', t)
    g = re.search(r'\(G : ExactMatrix (\d+) (\d+) 1 (\d+)\)', t)
    qd = re.search(r'\) : ExactMatrix (\d+) (\d+) 1 (\d+) :=\n  decodeExpected', t)
    rc = re.search(r'hterm : ∀ i : Fin (\d+), TermOf \(qN \(i / (\d+)\)\)', t)
    if not (m and g and qd and rc):
        raise Fail('decodeValue not found in ' + path)
    return dict(delta=m.group(1), Qg=g.group(1), Q=qd.group(1), Rc=int(rc.group(2)))


def slice_parts(args):
    """(source wire, first row, last row, first column, last column) of a `sliceMatrix` atom."""
    return tuple(strip_parens(x) for x in args[:5])


#: The stages of the caller's encoding of `σ_0 G` (the mask), one instance per chain.
SIGMA_ON, SIGMA_PRE = 'Stage_sigma_online_eval', 'Stage_sigma_preprocess'


def mask_params(gdir, roles_path, req_path, n):
    """The mask sampler of the sigma stages: P (online sampler-offset LUT width), n' (requirement
    period), the binary table value h (the sigma preprocessing's key table, odd entries; a term at
    the sigma instance `i_0`), checked against the requirement."""
    roles = json.load(open(roles_path))
    on, pre = Stage(gdir, SIGMA_ON), Stage(gdir, SIGMA_PRE)
    ms = [a for a in on.inst.atoms if a.loop and a.loop.count == n and
          roles[SIGMA_ON].get(a.loop.child, {}).get('role') == 'lut.member.sampler-offset']
    if len(ms) != 1:
        raise Fail('mask sampler')
    P = branch_width(on.sc, ms[0].loop.child)
    rt = open(req_path).read()
    per = re.search(r'\) % (\d+)\) / (\d+)\)', rt)
    nprime = int(per.group(2))
    rp = roles[SIGMA_PRE]
    car = [a for a in pre.inst.atoms if a.loop and a.loop.count == n and
           rp.get(a.loop.child, {}).get('role') == 'lut.key-member.sampler']
    if len(car) != 1:
        raise Fail('mask key member')
    need = {bare(x) for x in car[0].loop.args} - {None}
    tabs = [a for a in pre.inst.atoms if a.loop and a.loop.kind == 'par' and outputs_of(a) & need
            and rp.get(a.loop.child, {}).get('role') != 'lut.key-member.readout']
    txt = ''.join(scope_text(pre.sc, t.loop.child) for t in tabs)
    hs = None
    for mm in re.finditer(r'let w_\d+_\d+ : Int := \(MxxIR\.selectInt \(\(Int\.ofNat i_\d+\)\) \[(.*)\]\)$', txt, re.M):
        it = [strip_parens(x) for x in xmodel.split_top(mm.group(1), ', ')]
        if len(it) == P and all(x == '0' for x in it[0::2]) and len(set(it[1::2])) == 1:
            hs = it[1]
    if hs is None:
        lets = re.findall(r'let w_\d+_\d+ : [^\n]*?:= ([^\n]*)', txt)
        vals = []
        for x, y in zip(lets, lets[1:]):
            mm = re.match(r'^\(MxxRuntime\.matrixPolynomial \[(.*)\] : ', x)
            if mm and 'rotationPolynomial' in y:
                vals.append(strip_parens(mm.group(1)))
        if len(vals) == P and all(x == '0' for x in vals[0::2]) and len(set(vals[1::2])) == 1:
            hs = vals[1]
    if hs is None:
        raise Fail('mask table')
    return dict(P=P, nprime=nprime, h=hs)


def emit(R, S, dec, P=None, pc=None, pa=None, mk=None):
    n, m, h, mx, mg, SS = R['n'], R['m'], R['h'], R['mx'], R['mg'], R['S']
    Rt, qt, Mt, Qt = (f'({q(R[k])})' for k in ('Rm', 'qm', 'Mxm', 'Qm'))
    Qq, c, K, half = (f'({q(R[k])})' for k in ('Qq', 'c', 'K', 'half'))
    if len(R['factors']) != 2:
        raise Fail(f"action chain has {len(R['factors'])} constant factors (expected 2)")
    s9, s10 = (f'({q(f[2])})' for f in R['factors'])
    delta = dec['delta']
    kt = f'((((({Rt} / {Mt} : Nat) : Int) * {c}) / {Qq}))'
    I = f'(i0 : Nat) [h : Fact (i0 < {SS})]'
    pre = '  have := h.out\n  interval_cases i0 <;> '
    certs = []
    def cert(name, stmt, tac='decide'):
        certs.append(f'theorem {name} {I} :\n    {stmt} := by\n{pre}{tac}\n')
    cert('R_gt', f'1 < {Rt}')
    cert('q_gt', f'1 < {qt}')
    cert('Mx_gt', f'1 < {Mt}')
    cert('R_div_q', f'{Rt} = {Rt} / {qt} * {qt}')
    cert('R_quot_q', f'1 < {Rt} / {qt}')
    cert('R_div_Mx', f'{Rt} = {Rt} / {Mt} * {Mt}')
    cert('Q_mod_q', f'{Qt} % {qt} = 0')
    cert('Qq_eq', f'{Qq} = (({Rt} / {qt} : Nat) : Int)')
    cert('action', f'(({Rt} / {Mt} : Nat) : Int) * {c} = {Qq} * {kt}')
    cert('k_inv', f'{kt} * {s10} ≡ 1 [ZMOD ({qt} : Int)]')
    cert('s_delta', f'{s9} ≡ {delta} [ZMOD ({qt} : Int)]')
    cert('half_two', f'2 * {half} = ({Rt} : Int) + 1')
    Rc = dec['Rc']
    sp_on = slice_parts(R['slice_args'])
    cert('slice_index', f'({q(sp_on[3])}).toNat = {m} * (i0 % {Rc})')
    if mk is not None:
        cert('mask_half', f'2 * {mask_h(mk, Rc)} ≡ 1 [ZMOD ({qt} : Int)]')
    if pc is not None:
        sp_pre = slice_parts(pc['slice_args'])
        cert('slice_index_p', f'({q(sp_pre[3], "Stage_preprocess")}).toNat = {m} * (i0 % {Rc})')
        qp = lambda t: f'({q(t, "Stage_preprocess")})'
        cert('Qq_link', f'{Qq} = {qp(pc["Qq"])}')
        cert('s1_link', f'{s9} = {qp(pa["factors"][0][2])}')
        cert('s2_link', f'{s10} = {qp(pa["factors"][1][2])}')
        cert('c_link', f'{c} = {qp(pa["c"])}')
    lay = 'exact ⟨_, by simp [Backend.backend]; rfl, rfl⟩'
    cert('R_layout', f'∃ L : MxxRuntime.RegularLayout {Rt},\n      Backend.backend.regularLayout {Rt} {n} = some L ∧ L.droppedModuli = 0', lay)
    cert('Mx_layout', f'∃ L : MxxRuntime.RegularLayout {Mt},\n      Backend.backend.regularLayout {Mt} {n} = some L ∧ L.droppedModuli = 0', lay)
    certs_txt = f"""import Stage_online_eval
import Stage_preprocess
import {SIGMA_PRE}
import Backend
import Mathlib.Tactic.IntervalCases

/-! Generated by `generator/gen_final.py`: the instance constants of the online final combination
(instance body `{R['inst']}`, `{SS}` instances), `interval_cases i0 <;> decide`. -/

set_option maxRecDepth 65536
set_option maxHeartbeats 4000000

namespace RefreshTopProof.GenFinalInst.Certs

/-- The action's quotient `k = (R / M_x) c' / Q_q`. -/
abbrev kq (i0 : Nat) : Int := {kt}

""" + '\n'.join(certs) + '\nend RefreshTopProof.GenFinalInst.Certs\n'
    certs_txt = certs_txt.replace(kt + ' *', 'kq i0 *').replace('* ' + kt, '* kq i0')

    E = lambda w: q(expand(S, w))
    T = lambda w: mt(R, w)
    Qd, Qg = dec['Q'], dec['Qg']
    fz, fz_i = R['fresh_loop'][0], R['smudge_loop'][0]
    ns = 'RefreshTopProof.GenFinalInst.Certs'
    thm = f"""import GenFinalLink
import Gen_Final_Certs
import TopDecodeLoops

/-! Generated by `generator/gen_final.py`: the online final combination of instance body
`{R['inst']}` at a symbolic instance `i0 < {SS}`, from the export's atoms of the segment (named
by their record fields) and the role facts of PART_IFACE.md "Final combination (G)".
Recognized: switch `{R['h_switch']}`, lift `{R['h_lift']}`, half `{R['h_half']}`, combination
`{R['h_gc']}`/`{R['h_Dz']}`/`{R['h_D']}`, `A'` `{R['h_A1']}` of the column block `{R['h_slice']}`,
action `{R['h_Gg']}`/`{R['h_Gp']}`/`{R['h_w0']}`/`{R['h_wr']}`/`{R['h_Yz']}`/`{R['h_Y']}`, fresh loop
`{fz}`, smudge loop `{fz_i}`. -/

namespace RefreshTopProof.GenFinalInst

open Mxx.Primitives MxxRuntime MxxExponentLut Mxx.ExponentLut.Refresh RefreshTopProof GenDecode GenFinal

set_option maxHeartbeats 4000000 in
/-- **The online final combination of one instance**: its CRT term is the term of the decode
stage's value of component `j` modulo the instance's CRT prime. -/
theorem final_inst {I} {{inputs : DecodeLoops.Inputs}}
    {{E : Nat → ErrorMatrix {n} 1 {m}}} (j : Nat)
    -- gadgets (role facts)
    {{G : ExactMatrix {Qg} {n} 1 {m}}} (hG : gadgetMatrixRuns Backend.backend {R['gbase']} {R['gdigits']} G)
    {{Gr : ExactMatrix {Rt} {n} 1 {mg}}} (hGr : gadgetMatrixRuns Backend.backend {R['base']} {R['digits']} Gr)
    {{gx' : ExactMatrix {Mt} {n} 1 {mx}}} (hgx' : gadgetMatrixRuns Backend.backend {R['base4']} {R['digits4']} gx')
    {{gx : ExactMatrix {Rt} {n} 1 {mx}}} (hgx : centeredRebaseRuns gx' gx)
    -- atoms of the action chain
    {{{R['Gg']} : {T(R['Gg'])}}} ({R['h_Gg']} : gadgetMatrixRuns Backend.backend {R['gbase']} {R['gdigits']} {R['Gg']})
    {{{R['Gp']} : {T(R['Gp'])}}} ({R['h_Gp']} : centeredRebaseRuns {R['Gg']} {R['Gp']})
    {{{R['w0']} : {T(R['w0'])}}} ({R['h_w0']} : modulusReduceRuns {R['Gp']} {R['w0']})
    {{{R['wr']} : {T(R['wr'])}}} ({R['h_wr']} : centeredRebaseRuns {E(R['factors'][-1][0])} {R['wr']})
    {{{R['Yz']} : {T(R['Yz'])}}}
    ({R['h_Yz']} : gadgetDecomposeRuns Backend.backend {R['base4']} {R['digits4']} {E(R['X'])} {R['Yz']})
    {{{R['Y']} : {T(R['Y'])}}} ({R['h_Y']} : compactCenteredRebaseRuns {R['Y_bound']} {R['Yz']} {R['Y']})
    {{YI : ErrorMatrix {n} {mx} {m}}} (hYI : {R['Y']} = reduceMatrix _ {n} {mx} {m} YI) {{BY : Nat}} (hYb : CoeffBound YI BY)
    -- atoms of the combination
    {{{R['A0']} : {T(R['A0'])}}}
    (hhash : {R['A0']} = colBlock {m} inputs.2.2.1 ({m} * j))
    {{{R['A1']} : {T(R['A1'])}}} ({R['h_A1']} : modulusReduceRuns {R['A0']} {R['A1']})
    {{{R['Dz']} {R['D']} : {T(R['D'])}}}
    ({R['h_Dz']} : gadgetDecomposeRuns Backend.backend {R['base']} {R['digits']} {E(R['z'])} {R['Dz']})
    ({R['h_D']} : compactCenteredRebaseRuns {R['D_bound']} {R['Dz']} {R['D']})
    {{DI : ErrorMatrix {n} ({mg} + {mg}) {m}}} (hDI : {R['D']} = reduceMatrix _ {n} ({mg} + {mg}) {m} DI) {{BD : Nat}}
    (hDb : CoeffBound DI BD)
    {{{R['g']} {R['c1']} : {T(R['g'])}}} {{{R['gc']} : {T(R['gc'])}}}
    ({R['h_gc']} : concatColumns {R['g']} {R['c1']} {R['gc']})
    {{{R['cx']} : {T(R['cx'])}}}
    -- role facts: rows of the combination
    {{s β : ExactPoly {Rt} {n}}} {{x : ErrorPoly {n}}}
    {{Kg K₁ : Fin 2 → ExactMatrix {Rt} {n} 1 {mg}}} {{Kx : Fin 2 → ExactMatrix {Rt} {n} 1 {mx}}}
    {{eg e₁ : ErrorMatrix {n} 1 {mg}}} {{ex : ErrorMatrix {n} 1 {mx}}} {{Beg Be1 Bex : Nat}}
    (heg : CoeffBound eg Beg) (he1 : CoeffBound e₁ Be1) (hex : CoeffBound ex Bex)
    (hg : {R['g']} = maskPart s Kg - β • Gr + reduceMatrix _ {n} 1 {mg} eg)
    (hc₁ : {R['c1']} = maskPart s K₁ - Gr + reduceMatrix _ {n} 1 {mg} e₁)
    (hcx : {R['cx']} = maskPart s Kx -
      reducePoly {Rt} {n} (((({Rt} / {Mt} : Nat) : Int) : ErrorPoly {n}) * x) • gx + reduceMatrix _ {n} 1 {mx} ex)
    -- role facts: residues of the semantic values
    (hβ : rho {qt} β = rho {qt} (polynomialOfCoefficients inputs.2.1 : ExactPoly {Qd} {n}))
    (hx : reducePoly {qt} {n} x = rho {qt} (DecodeLoops.decodeR inputs j 0 0 *
      multiplyMonomial (liftInteger inputs.2.2.2.2.1) inputs.2.2.2.2.2.1 0 0))
    -- role facts: hint, preimage, fresh, smudge
    {{{R['b']} : {T(R['b'])}}} {{Bk : Fin 2 → ExactMatrix {Rt} {n} 1 {h}}}
    {{eb : ErrorMatrix {n} 1 {h}}} {{Beb : Nat}}
    (hb : {R['b']} = maskPart s Bk + reduceMatrix _ {n} 1 {h} eb) (heb : CoeffBound eb Beb)
    {{{R['P']} : {T(R['P'])}}} {{Pd : ErrorMatrix {n} {h} {m}}} {{BP : Nat}}
    (hP : {R['P']} = reduceMatrix _ {n} {h} {m} Pd) (hPd : CoeffBound Pd BP)
    {{Kf Ks : Fin 2 → ExactMatrix {Rt} {n} 1 {m}}}
    (hpre : ∀ k', Bk k' * {R['P']} =
      (Kg k' * topRows {R['D']} + K₁ k' * bottomRows {R['D']} - Kx k' * {R['Y']}) + Kf k' + Ks k')
    {{{R['fresh']} : {T(R['fresh'])}}} {{ef : ErrorMatrix {n} 1 {m}}} {{Bf : Nat}}
    (hf : {R['fresh']} = maskPart s Kf - ((({Rt} / {qt} : Nat) : Int) : ExactPoly {Rt} {n}) •
      reduceMatrix _ {n} 1 {m} (E j) + reduceMatrix _ {n} 1 {m} ef) (hef : CoeffBound ef Bf)
    {{{R['smudge']} : {T(R['smudge'])}}} {{Tsm : ExactMatrix {Rt} {n} 1 {m}}} {{{R['H']} : {T(R['H'])}}}
    {{es : ErrorMatrix {n} 1 {m}}} {{Bs : Nat}}
    (hs : {R['smudge']} = maskPart s Ks - Tsm + reduceMatrix _ {n} 1 {m} es) (hes : CoeffBound es Bs)
    {{M : ErrorMatrix {n} 1 {m}}} {{BM : Nat}} (hTH : Tsm - {R['H']} = reduceMatrix _ {n} 1 {m} M)
    (hM : CoeffBound M BM)
    (hnoise : 2 * ({h} * {n} * Beb * BP + ({mg} * {n} * Beg * BD + {mg} * {n} * Be1 * BD + {mx} * {n} * Bex * BY)
      + Bf + Bs + BM) < {Rt} / {qt})
    -- atoms of the switch and the lift
    {{{R['r']} : {T(R['r'])}}} ({R['h_switch']} : modulusSwitchRuns {E(R['zeta'])} {R['r']})
    {{{R['l']} : {T(R['l'])}}} ({R['h_lift']} : centeredRebaseRuns {R['r']} {R['l']}) :
    TermOf {qt} {K} {E(R['term'])} (DecodeLoops.decodeValue inputs G E j) := by
  have hn : (0 : Nat) < {n} := by norm_num
  have hR : 1 < {Rt} := {ns}.R_gt i0
  have hq : 1 < {qt} := {ns}.q_gt i0
  have hMx : 1 < {Mt} := {ns}.Mx_gt i0
  have hQ : 1 < {Qd} := by norm_num
  have hRq : {Rt} = {Rt} / {qt} * {qt} := {ns}.R_div_q i0
  have hqR : {qt} ∣ {Rt} := ⟨{Rt} / {qt}, by rw [Nat.mul_comm]; exact hRq⟩
  have hqQ : {qt} ∣ {Qd} := Nat.dvd_of_mod_eq_zero ({ns}.Q_mod_q i0)
  have hRp : {Rt} = {Rt} / {Mt} * {Mt} := {ns}.R_div_Mx i0
  obtain ⟨LX, hLX, hxX⟩ := {ns}.Mx_layout i0
  obtain ⟨L, hL, hLx⟩ := {ns}.R_layout i0
  have hact := action_chain_gen hn hRp hMx LX hLX hxX ({ns}.action i0) hgx' hgx {R['h_wr']} rfl {R['h_Yz']} {R['h_Y']} x
  obtain ⟨h0, h1⟩ := decompose_rows_gen hn L hL hLx hR {R['h_Dz']} {R['h_D']} hGr
  rw [scaled_rows_gen rfl 0] at h0
  rw [scaled_rows_gen rfl 1] at h1
  obtain ⟨ec, hec, hcomb⟩ := comb_bound_gen hn {R['h_gc']} hDI hDb h0 h1 hYI hYb hact heg he1 hex hg hc₁ hcx
  rw [{ns}.Qq_eq i0] at hcomb
  have hG' : {R['Gg']} = G := gadgetMatrixRuns_unique {R['h_Gg']} hG
  subst hG'
  have hyv := link_gen hn hq hR hqR hQ hqQ (β := β) (A' := {R['A1']})
    (A := colBlock {m} inputs.2.2.1 ({m} * j))
    (σ := fun _ _ ↦ polynomialOfCoefficients inputs.2.1) (Δ := {delta}) (r := DecodeLoops.decodeR inputs j)
    (c := inputs.2.2.2.2.1) (e := inputs.2.2.2.2.2.1) (G := {R['Gg']})
    (fun i col ↦ by rw [modulusReduce_rho hn {R['h_A1']}, rho_rho hn hq hR hqR, hhash])
    hβ (fun col ↦ by
      have ht := tx_rho_gen hn hq hR hqR hQ hqQ (k := {ns}.kq i0) (x := x) {R['h_w0']} rfl rfl
        (scalar_gen hn hq hQ hqQ ({ns}.k_inv i0) ({ns}.s_delta i0) hx) col
      rw [ht, decodeTerm, {R['h_Gp']}.2]; simp only [mul_assoc])
    (E j)
  exact final_gen hn hRq (lt_trans Nat.zero_lt_one ({ns}.R_quot_q i0)) hq
    hb heb hP hPd hpre hcomb hec hf hef hs hes hTH hM hnoise rfl {R['h_switch']} {R['h_lift']} rfl hyv

"""
    thm += slice_text(R, S, R['slice_args'], 'Stage_online_eval', 'slice_online', 'slice_index', dec)
    if pc is not None:
        thm += slice_text(R, P, pc['slice_args'], 'Stage_preprocess', 'slice_pre', 'slice_index_p', dec)
        thm += pre_link_text(R, S, P, pc, pa, dec)
    if mk is not None:
        thm += mask_text(R, mk, dec)
    thm += 'end RefreshTopProof.GenFinalInst\n'
    return certs_txt, thm


def slice_text(R, S, args, module, name, cert, dec):
    """The `A'` link: the stage's column block of its `A'` input is the decode stage's `A'` of component
    `i0 mod R` when the input is the decode stage's `A'`."""
    n, m = R['n'], R['m']
    src, r0, r1, c0, c1 = slice_parts(args)
    out = strip_parens(args[-1])
    ns = 'RefreshTopProof.GenFinalInst.Certs'
    return f"""
/-- **The `A'` link** ({module}): the column block `{out}` (atom of `{S.name}`, of the input `{src}`) is
the decode stage's `A'` of component `i0 % {dec['Rc']}` when the input is the decode stage's `A'`. -/
theorem {name} (i0 : Nat) [h : Fact (i0 < {R['S']})] {{inputs : DecodeLoops.Inputs}}
    {{A : {q(S.wires[src].type, module)}}} {{w : {q(S.wires[out].type, module)}}}
    (hw : sliceMatrix A ({q(r0, module)}) ({q(r1, module)}) ({q(c0, module)}) ({q(c1, module)}) w)
    (hA : A = inputs.2.2.1) :
    w = colBlock {m} inputs.2.2.1 ({m} * (i0 % {dec['Rc']})) := by
  rw [slice_colBlock hw, {ns}.{cert} i0, hA]

"""


def mask_h(mk, Rc):
    """The mask's table value at the chain `i0 / Rc` of part instance `i0`."""
    return '(' + q(mk['h'], SIGMA_PRE).replace('i0', f'(i0 / {Rc})') + ')'


def mask_text(R, mk, dec):
    n, Rt, qt, Qd = R['n'], f"({q(R['Rm'])})", f"({q(R['qm'])})", dec['Q']
    P, W = mk['P'], mk['nprime']
    h = mask_h(mk, dec['Rc'])
    ns = 'RefreshTopProof.GenFinalInst.Certs'
    return f"""
/-- **The mask residue** (`hβ` of `final_inst`): the mask payload `β`, a binary sampler of table
`{h}` (the sigma preprocessing's key table of the instance's chain, `2 h ≡ 1`, `Certs.mask_half`)
on the PRF outputs `zN`, has modulo the
CRT prime the residue of the decode stage's `σ` (`inputs.2.1`), by the requirement's mask formula
`hσ` (period `{2 * W}`) and `zN = zR` (the PRF outputs are the requirement's sums). -/
theorem mask_inst (i0 : Nat) [h : Fact (i0 < {R['S']})] {{inputs : DecodeLoops.Inputs}}
    {{β : ExactPoly {Rt} {n}}} {{zN : Fin {n} → Nat}} {{zR : Nat → Int}}
    (hσ : ∀ o < {n}, GenIface.famAt inputs.2.1 o = 1 - zR o % {2 * W} / {W})
    (hz : ∀ o : Fin {n}, (zN o : Int) = zR o)
    (hβ : β = reducePoly {Rt} {n} (∑ o : Fin {n}, Xe (n := {n}) o *
      (Gen.sampledG {P} (Gen.binTable {P} fun _ ↦ {h}) (Gen.readoutExpG {P} {W} (zN o)) +
        Gen.sampledG {P} (Gen.binTable {P} fun _ ↦ {h}) 1))) :
    rho {qt} β = rho {qt} (polynomialOfCoefficients inputs.2.1 : ExactPoly {Qd} {n}) := by
  have hn : (0 : Nat) < {n} := by norm_num
  have hR : 1 < {Rt} := {ns}.R_gt i0
  have hq : 1 < {qt} := {ns}.q_gt i0
  have hRq : {Rt} = {Rt} / {qt} * {qt} := {ns}.R_div_q i0
  have hqR : {qt} ∣ {Rt} := ⟨{Rt} / {qt}, by rw [Nat.mul_comm]; exact hRq⟩
  have hqQ : {qt} ∣ {Qd} := Nat.dvd_of_mod_eq_zero ({ns}.Q_mod_q i0)
  have h2 : ((2 * {h} : Int) : ZMod {qt}) = 1 := by
    have := (ZMod.intCast_eq_intCast_iff _ _ _).2 ({ns}.mask_half i0)
    simpa using this
  refine mask_rho_gen hn hq hR hqR (by norm_num) hqQ (P := {P}) (W' := {W}) (by norm_num) (by norm_num)
    (by norm_num) h2 (fun o ↦ ?_) hβ
  have e := hσ o o.2
  rw [GenIface.famAt_of_lt _ o.2] at e
  rw [e, ← hz o]
  norm_num

"""


def pre_link_text(R, S, P, pc, pa, dec):
    """`pre_link`: the online combination and action digits equal the preprocess ones."""
    n, m = R['n'], R['m']
    Qd = dec['Q']
    ns = 'RefreshTopProof.GenFinalInst.Certs'
    def side(S_, r, mod, pre):
        Eq = lambda w: q(expand(S_, w), mod, pre)
        T = lambda w: q(S_.wires[w].type, mod, pre)
        W = lambda w: pre + w
        H = lambda h: pre + h
        hh = 'hhash_p' if pre else 'hhash'
        c, a = r
        return f"""    {{{W(c['A0'])} : {T(c['A0'])}}}
    ({hh} : {W(c['A0'])} = colBlock {m} inputs.2.2.1 ({m} * j))
    {{{W(c['A1'])} : {T(c['A1'])}}} ({H(c['h_A1'])} : modulusReduceRuns {W(c['A0'])} {W(c['A1'])})
    {{{W(c['Dz'])} {W(c['D'])} : {T(c['D'])}}}
    ({H(c['h_Dz'])} : gadgetDecomposeRuns Backend.backend {c['base']} {c['digits']} {Eq(c['z'])} {W(c['Dz'])})
    ({H(c['h_D'])} : compactCenteredRebaseRuns {c['D_bound']} {W(c['Dz'])} {W(c['D'])})
    {{{W(a['Gg'])} : {T(a['Gg'])}}} ({H(a['h_Gg'])} : gadgetMatrixRuns Backend.backend {a['gbase']} {a['gdigits']} {W(a['Gg'])})
    {{{W(a['Gp'])} : {T(a['Gp'])}}} ({H(a['h_Gp'])} : centeredRebaseRuns {W(a['Gg'])} {W(a['Gp'])})
    {{{W(a['w0'])} : {T(a['w0'])}}} ({H(a['h_w0'])} : modulusReduceRuns {W(a['Gp'])} {W(a['w0'])})
    {{{W(a['wr'])} : {T(a['wr'])}}} ({H(a['h_wr'])} : centeredRebaseRuns {Eq(a['factors'][-1][0])} {W(a['wr'])})
    {{{W(a['Yz'])} : {T(a['Yz'])}}}
    ({H(a['h_Yz'])} : gadgetDecomposeRuns Backend.backend {a['base4']} {a['digits4']} {Eq(a['X'])} {W(a['Yz'])})
    {{{W(a['Y'])} : {T(a['Y'])}}} ({H(a['h_Y'])} : compactCenteredRebaseRuns {a['Y_bound']} {W(a['Yz'])} {W(a['Y'])})
"""
    ron = (dict(A0=R['A0'], h_A1=R['h_A1'], A1=R['A1'], Dz=R['Dz'], D=R['D'], h_Dz=R['h_Dz'], base=R['base'],
                digits=R['digits'], z=R['z'], h_D=R['h_D'], D_bound=R['D_bound']), R)
    p = 'p_'
    return f"""
/-- **The online decompositions are the preprocess ones** (combination digits `{R['D']}` =
`{pc['D']}`, action digits `{R['Y']}` = `{pa['Y']}` of `{P.name}`): the same column block of `A'`, gadget
and chain with equal instance constants (`Certs.Qq_link`, `s1_link`, `s2_link`, `c_link`). -/
theorem pre_link (i0 : Nat) [h : Fact (i0 < {R['S']})] {{inputs : DecodeLoops.Inputs}} (j : Nat)
{side(S, ron, 'Stage_online_eval', '')}{side(P, (pc, pa), 'Stage_preprocess', p)}    :
    {R['D']} = {p}{pc['D']} ∧ {R['Y']} = {p}{pa['Y']} :=
  ⟨comb_digits_link (hhash.trans hhash_p.symm) ({ns}.Qq_link i0) {R['h_A1']} {p}{pc['h_A1']}
      {R['h_Dz']} {p}{pc['h_Dz']} {R['h_D']} {p}{pc['h_D']},
    action_digits_link {R['h_Gg']} {p}{pa['h_Gg']} {R['h_Gp']} {p}{pa['h_Gp']} {R['h_w0']} {p}{pa['h_w0']}
      ({ns}.s1_link i0) ({ns}.s2_link i0) ({ns}.c_link i0) {R['h_wr']} {p}{pa['h_wr']}
      {R['h_Yz']} {p}{pa['h_Yz']} {R['h_Y']} {p}{pa['h_Y']}⟩

"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('d')
    ap.add_argument('--out')
    ap.add_argument('--decode')
    ap.add_argument('--roles', help='roles.json of the export (for the fresh error parameters)')
    ap.add_argument('--requirement', help='Gen_Requirement.lean (default <d>/Gen_Requirement.lean)')
    a = ap.parse_args()
    gdir = os.path.join(a.d, 'generated')
    out = a.out or a.d
    R, S = recognize_full(gdir)
    dec = decode_info(a.decode or os.path.join(a.d, 'TopDecodeLoops.lean'))
    P, pc, pa = preprocess_links(gdir, R)
    mk = None
    if a.roles:
        mk = mask_params(gdir, a.roles, a.requirement or os.path.join(a.d, 'Gen_Requirement.lean'), R['n'])
        print('mask:', mk)
    certs, thm = emit(R, S, dec, P, pc, pa, mk)
    open(os.path.join(out, 'Gen_Final_Certs.lean'), 'w').write(certs)
    open(os.path.join(out, 'Gen_Final_Inst.lean'), 'w').write(thm)
    if a.roles:
        fp = fresh_params(gdir, a.roles, a.requirement or os.path.join(a.d, 'Gen_Requirement.lean'), R)
        print('fresh error:', {k: v for k, v in fp.items() if k != 'pairs'}, len(fp['pairs']), 'buckets', fp['pairs'][:3])
    print(f"preprocess link: {P.name} {pc['D']} {pa['Y']}")
    print(f"instance {R['inst']} (S = {R['S']}, n = {R['n']}, m = {R['m']}, hint {R['h']}); "
          f"switch {R['h_switch']}, combination {R['comb']}, fresh {R['fresh']}, smudge {R['smudge']}, "
          f"action factors {[f[2] for f in R['factors']]}; wrote {out}/Gen_Final_{{Certs,Inst}}.lean")


if __name__ == '__main__':
    try:
        main()
    except Fail as e:
        sys.exit(f'gen_final.py: recognition failed: {e}')
