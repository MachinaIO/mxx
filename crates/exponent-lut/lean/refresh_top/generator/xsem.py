"""Semantic step emission for the instance theorems (WP D, used by gen_instance.py).

Works on one instance body (xmodel scope JSON) whose facts record is `F`; wires are referred to as
`<inst>.w_x ctx` (`W(x)`). Recognition is by atom kind and data flow only.

* `MaskFacts`: facts `w = fun _ _ ↦ reducePoly q n s` (secret `s`) / `w = 1` for 1×1 wires, from
  the instance's secret inputs, `(1 : …)` lets, `centeredRebaseRuns` and `modulusReduceRuns`.
* `enc_steps`: the encrypt roles: bundle children (label entries, LUT branch keys) via the bridges
  of GenBridge_Enc, root encryptions (`c = (sM A0 + 1 A1 - T) + e` with a Gaussian `e`) by the shape
  of `T`, the hint `[s, 1] A + e`.
"""
import re

ARG = re.compile(r'^\(?(w_\d+(?:_\d+)?(?:_concat_\d+)?)\)?$')


# The gadget base of the stage (`gadgetDecomposeRuns backend base digits`) and the compact rebase
# bound (`compactCenteredRebaseRuns bound`, base / 2); read from the scopes by `set_gadget`.
GBASE, GHALF = '32768', '16384'


def set_gadget(scopes):
    """Read the stage's gadget base and rebase bound (one value each across the scopes)."""
    global GBASE, GHALF
    bases, halves = set(), set()
    for s in scopes.values():
        for a in s.to_json()['atoms']:
            h = a.get('head', '')
            if h.endswith('gadgetDecomposeRuns') or h.endswith('gadgetMatrixRuns'):
                bases.add(a['args'][1].strip('()'))
            elif h.endswith('compactCenteredRebaseRuns'):
                halves.add(a['args'][0].strip('()'))
    assert len(bases) <= 1 and len(halves) <= 1, (bases, halves)
    if bases:
        GBASE = bases.pop()
    if halves:
        GHALF = halves.pop()


def wire_of(t):
    m = ARG.match(t.strip())
    return m.group(1) if m else None


class Inst:
    def __init__(self, j, fields, inst):
        self.j = j
        self.inst = inst
        self.wires = j['wires']
        self.fields = fields  # field -> (frag, type)
        self.prod = {}  # wire -> atom producing it (last argument)
        for a in j['atoms']:
            if a.get('loop'):
                continue
            if a['args']:
                w = wire_of(a['args'][-1])
                if w:
                    self.prod[w] = a
        self.lets = {w: v['term'] for w, v in self.wires.items() if v['src'] == 'let'}

    def W(self, w):
        return f'({self.inst}.{w} ctx)'

    def F(self, a):
        return f'F.c{self.fields[a["field"]][0]}.{a["field"]}'

    def head(self, w):
        a = self.prod.get(w)
        return a['head'].split('.')[-1] if a else None


class MaskFacts:
    """Proof terms of `w = fun _ _ ↦ reducePoly q n s` or `w = 1`."""

    def __init__(self, I, secrets):
        self.I = I
        self.secrets = secrets  # input wire -> (hyp name, small hyp name)
        self.cache = {}
        self.haves = []

    def kind(self, w):
        if w in self.secrets:
            return 'sec'
        t = self.I.lets.get(w, '')
        if re.match(r'^\(1 : ', t):
            return 'one'
        a = self.I.prod.get(w)
        if a and a['head'].endswith('centeredRebaseRuns'):
            return self.kind(wire_of(a['args'][0]))
        return None

    def fact(self, w):
        """Name of a hypothesis `W w = …` (emits `have`s), or `rfl` for one-lets."""
        if w in self.cache:
            return self.cache[w]
        I = self.I
        if w in self.secrets:
            r = self.secrets[w][0]
        elif re.match(r'^\(1 : ', I.lets.get(w, '')):
            r = 'rfl'
        else:
            a = I.prod.get(w)
            if not a or not a['head'].endswith('centeredRebaseRuns'):
                return None
            src = wire_of(a['args'][0])
            k = self.kind(src)
            if k is None:
                return None
            name = f'm_{w}'
            if k == 'one':
                hs = self.fact(src)
                if hs == 'rfl':
                    self.haves.append(f'  have {name} : {I.W(w)} = 1 := rebase_one_gen (by decide) (by decide) {I.F(a)}')
                else:
                    self.haves.append(f'  have {name} : {I.W(w)} = 1 := rebase_one_gen (by decide) (by decide) (by have h := {I.F(a)}; rw [{hs}] at h; exact h)')
            else:
                hs = self.fact(src)
                small = self.secret_small(src)
                self.haves.append(f'  have {name}_a := {I.F(a)}\n  rw [{hs}] at {name}_a\n  have {name} := rebase_small_gen (by decide) (by decide) {small} {name}_a')
            r = name
        self.cache[w] = r
        return r

    def secret_small(self, w):
        if w in self.secrets:
            return self.secrets[w][1]
        a = self.I.prod.get(w)
        return self.secret_small(wire_of(a['args'][0]))


def app(term):
    """`f a b` -> (f, [a, b]) for a let term (top-level split)."""
    import xmodel
    h, a = xmodel.app_split(xmodel.strip_parens(term))
    return h.split('.')[-1], [x.strip() for x in a]


def pre_of(a):
    """The tag components of a hash sample before its trailing `component, k`."""
    import xmodel
    comps = xmodel.strip_parens(a['args'][2])
    items = xmodel.split_top(comps[1:-1], ', ')
    return '[' + ', '.join(items[:-2]) + ']'


def enc_steps(I, M, roles, adapters, gt):
    """Lean steps (list of strings) and notes for the encrypt roles of instance `I`.
    `adapters`: child scope -> bundle name (e.g. `EncLutBranchKey.Comp_2`); `gt(q)`: proof of `1 < q`."""
    steps, notes = [], []
    # --- bundle children
    for a in I.j['atoms']:
        L = a.get('loop')
        if not L or L['kind'] != 'par':
            continue
        child = L['child']
        role = roles.get(child, {}).get('role')
        comp = adapters.get(child)
        args = [wire_of(x.replace(' i', '')) for x in L['args']]
        call = f'{child}.sem ({I.F(a)} i)'
        name = f's_{a["field"]}'
        if role == 'enc.label-entry':
            hs, ho = M.fact(args[4]), M.fact(args[5])
            if None in (hs, ho):
                notes.append(f'{a["field"]}: label entry masks unknown')
                continue
            steps.append(f'  node_have {name} := fun i ↦ EncLabelEntry.rel ({call}) {hs} {ho} rfl rfl')
        elif role == 'enc.lut-branch-key' and comp:
            v = comp.split('.Comp')[1]
            hs, ho = M.fact(args[0]), M.fact(args[2])
            if None in (hs, ho):
                notes.append(f'{a["field"]}: branch key masks unknown')
                continue
            mods = adapters[child + '#dims']
            if v in ('_1', '_2'):
                gts = f'({gt(mods[0])}) ({gt(mods[3])}) ({gt(mods[4])}) (by decide)'
            else:
                gts = f'({gt(mods[0])}) (by decide)'
            rel = 'rel' + v
            odd = 'odd' + v
            steps.append(f'  node_have {name} := fun i ↦ EncLutBranchKey.{rel} ({call}) {gts} {hs} {ho} '
                         + ' '.join(['rfl'] * 16) + f' (EncLutBranchKey.{odd} ({call}) rfl)')
    # --- root encryptions
    encs = []
    for a in I.j['atoms']:
        if a.get('loop') or not a['head'].endswith('gaussianSample'):
            continue
        we = wire_of(a['args'][-1])
        for c, t in I.lets.items():
            h, xs = app(t)
            if h == 'matrixAdd' and len(xs) == 2 and wire_of(xs[1]) == we:
                encs.append((c, wire_of(xs[0]), a))
    def parts(x):
        h, xs = app(I.lets.get(x, ''))
        if h != 'matrixSub':
            return None
        y, T = wire_of(xs[0]), wire_of(xs[1])
        h2, ys = app(I.lets.get(y, ''))
        if h2 != 'matrixAdd':
            return None
        p1, p2 = wire_of(ys[0]), wire_of(ys[1])
        h3, s1 = app(I.lets.get(p1, ''))
        h4, s2 = app(I.lets.get(p2, ''))
        if h3 != 'matrixMulScalarLeft' or h4 != 'matrixMulScalarLeft':
            return None
        return dict(sM=wire_of(s1[0]), A0=wire_of(s1[1]), oneM=wire_of(s2[0]), A1=wire_of(s2[1]), T=T)
    k = 0
    while k + 1 < len(encs):
        (c0, x0, e0), (c1, x1, e1) = encs[k], encs[k + 1]
        p, q = parts(x0), parts(x1)
        if not p or not q:
            k += 1
            continue
        k += 2
        name = f's_{c0}'
        hT, ts = app(I.lets.get(p['T'], ''))
        if hT == 'matrixMulScalarRight':
            hU, us = app(I.lets.get(wire_of(ts[0]), ''))
            rM, G = wire_of(us[0]), wire_of(us[1])
            hT1, ts1 = app(I.lets.get(q['T'], ''))
            hU1, us1 = app(I.lets.get(wire_of(ts1[0]), ''))
            oneR = wire_of(us1[0])
            A = I.prod.get(p['A0'])
            if A and A['head'].endswith('hashSample'):
                hs = [M.fact(w) for w in (p['sM'], p['oneM'], rM, oneR)]
                if None in hs:
                    notes.append(f'{c0}: input encryption masks unknown')
                    continue
                samp = [I.F(I.prod[w]) for w in (p['A0'], p['A1'], q['A0'], q['A1'])]
                steps.append(f'  node_have {name} := EncRoot.input {" ".join(hs)} {" ".join(samp)} {I.W(G)} '
                             f'{I.F(e0)} {I.F(e1)} (c0 := {I.W(c0)}) (c1 := {I.W(c1)}) rfl rfl')
            else:
                reb = [I.prod.get(w) for w in (p['sM'], p['oneM'], rM, oneR)]
                if any(r is None or not r['head'].endswith('centeredRebaseRuns') for r in reb):
                    notes.append(f'{c0}: source encryption masks not rebased')
                    continue
                srcs = [wire_of(r['args'][0]) for r in reb]
                if srcs[0] not in M.secrets or srcs[2] not in M.secrets:
                    notes.append(f'{c0}: source masks not rebased from secret inputs')
                    continue
                hsm, hsp = M.secrets[srcs[0]], M.secrets[srcs[2]]
                steps.append(f'  node_have {name} := EncRoot.source_rebase (by decide) (by decide) {hsm[1]} {hsp[1]} '
                             f'{hsm[0]} {hsp[0]} ' + ' '.join(I.F(r) for r in reb) +
                             f' {I.W(p["A0"])} {I.W(q["A0"])} {I.W(p["A1"])} {I.W(q["A1"])} {I.W(G)} '
                             f'{I.F(e0)} {I.F(e1)} (c0 := {I.W(c0)}) (c1 := {I.W(c1)}) rfl rfl')
        elif hT == 'matrixMulScalarLeft':
            h5, vs = app(I.lets.get(wire_of(ts[1]), ''))
            G = wire_of(vs[1])
            red = [I.prod.get(w) for w in (p['sM'], p['oneM'])]
            if any(r is None or not r['head'].endswith('modulusReduceRuns') for r in red):
                notes.append(f'{c0}: gadget encryption masks not reduced')
                continue
            src = wire_of(red[0]['args'][0])
            if src not in M.secrets:
                notes.append(f'{c0}: gadget encryption mask source unknown')
                continue
            q0 = I.wires[c0]['type'].split('ExactMatrix ', 1)[1].rsplit(' ', 3)[0]
            samp = [I.F(I.prod[w]) for w in (p['A0'], p['A1'], q['A0'], q['A1'])]
            steps.append(f'  node_have {name} := EncRoot.gadget_reduced (by decide) ({gt(q0)}) (by decide) '
                         f'{M.secrets[src][0]} {I.F(red[0])} {I.F(red[1])} {pre_of(I.prod[p["A0"]])} {" ".join(samp)} {I.W(G)} '
                         f'{I.F(e0)} {I.F(e1)} (c0 := {I.W(c0)}) (c1 := {I.W(c1)}) rfl rfl')
        else:
            notes.append(f'{c0}: unknown encryption target')
    # --- hint `[s, 1] A + e`
    for c, t in I.lets.items():
        h, xs = app(t)
        if h != 'matrixAdd' or len(xs) != 2:
            continue
        hm, ms = app(I.lets.get(wire_of(xs[0]) or '', ''))
        ea = I.prod.get(wire_of(xs[1]) or '')
        if hm != 'matrixMul' or not ea or not ea['head'].endswith('gaussianSample'):
            continue
        cc = I.prod.get(wire_of(ms[0]))
        if not cc or not cc['head'].endswith('concatColumns'):
            continue
        ra, rb = [I.prod.get(wire_of(x)) for x in cc['args'][:2]]
        if not ra or not rb or not ra['head'].endswith('modulusReduceRuns'):
            continue
        src = wire_of(ra['args'][0])
        q0 = I.wires[c]['type'].split('ExactMatrix ', 1)[1].rsplit(' ', 3)[0]
        steps.append(f'  node_have s_{c} := Gen.hint_gen (by decide) ({gt(q0)}) (by decide) {M.secrets[src][0]} rfl '
                     f'{I.F(ra)} {I.F(rb)} {I.F(cc)} {I.F(ea)} (A := {I.W(wire_of(ms[1]))})')
    return steps, notes


# --------------------------------------------------------------------------- root LUT rounds

class Flow:
    """Data-flow queries over an instance body: atoms by head and argument wires."""

    def __init__(self, I):
        self.I = I
        self.atoms = [a for a in I.j['atoms'] if not a.get('loop')]

    def find(self, head, *args, pos=None):
        """The atom with predicate `head` whose argument wires start with `args` (None = any)."""
        for a in self.atoms:
            if not a['head'].endswith(head):
                continue
            ws = [wire_of(x) for x in a['args']]
            if all(x is None or (i < len(ws) and ws[i] == x) for i, x in enumerate(args)):
                return a
        raise KeyError((head, args))

    def out(self, a):
        return wire_of(a['args'][-1])

    def let(self, head, *args):
        """The let wire whose value is `head args…` (argument wires)."""
        for w, t in self.I.lets.items():
            h, xs = app(t)
            if h == head and [wire_of(x) for x in xs] == list(args):
                return w
        raise KeyError((head, args))

    def slice(self, src, r0, r1):
        for a in self.atoms:
            if a['head'].endswith('sliceMatrix') and wire_of(a['args'][0]) == src and \
                    a['args'][1].strip('()') == str(r0) and a['args'][2].strip('()') == str(r1):
                return a
        raise KeyError(('slice', src, r0, r1))


def tuple_wires(t):
    return re.findall(r'\bw_\d+_(?:\d+|concat_\d+)\b', t)


def root_lut_match(I, loop_atom):
    """The atoms of a root LUT round around the seq loop `loop_atom` (lut_round_gen / lut_branches_gen)."""
    Fl = Flow(I)
    L = loop_atom['loop']
    args = [wire_of(x) for x in L['args'][10:]]
    hf0, key, hf1, x0, x1, x2, x3, x4, x5, f00, f01, f10, f11 = args
    init = tuple_wires(L['init'])
    fin = tuple_wires(L['final'])
    r = dict(hf0=hf0, key=key, hf1=hf1, x=[x0, x1, x2, x3, x4, x5], f=[f00, f01, f10, f11],
             init=init, final=fin, count=L['count'])
    r['init_gets'] = [Fl.find('familyGetStatic', f, None, w) for f, w in zip(r['f'], init[6:10])]
    hr = [Fl.find('ringAutomorphismRuns', None, x) for x in r['x']]
    r['hr'] = hr
    o = Fl.out
    hl01 = Fl.find('concatRows', o(hr[0]), o(hr[1]))
    hl = Fl.find('concatRows', o(hl01), o(hr[2]))
    hr01 = Fl.find('concatRows', o(hr[3]), o(hr[4]))
    hrr = Fl.find('concatRows', o(hr01), o(hr[5]))
    hboth = Fl.find('concatColumns', o(hl), o(hrr))
    r['enc'] = [hl01, hl, hr01, hrr, hboth]
    both = o(hboth)
    Lc = r['count']
    h3 = [Fl.find('familyGetStatic', f, None) for f in (hf0, hf1)]
    h3 = [a for a in Fl.atoms if a['head'].endswith('familyGetStatic') and wire_of(a['args'][0]) in (hf0, hf1)
          and a['args'][1].strip('()') == str(Lc)]
    h3 = sorted(h3, key=lambda a: 0 if wire_of(a['args'][0]) == hf0 else 1)
    r['h3'] = h3
    hhl01 = Fl.find('concatRows', o(h3[0]))
    hhl = Fl.find('concatRows', o(hhl01))
    hhr01 = Fl.find('concatRows', o(h3[1]))
    hhr = Fl.find('concatRows', o(hhr01))
    hH = Fl.find('concatColumns', o(hhl), o(hhr))
    Ad = [wire_of(hhl01['args'][1]), wire_of(hhr01['args'][1]), wire_of(hhl['args'][1]), wire_of(hhr['args'][1])]
    r['samples'] = [I.prod[w] for w in Ad]  # 00, 01, 10, 11
    lv = re.search(r'\[\.u64Le \((\d+)\), \.integer', r['samples'][0]['args'][2])
    r['level'] = int(lv.group(1))
    r['tag'] = r['samples'][0]['args'][1].strip('()')
    tab = re.search(r'selectInt \(\d+\) (\[[^\]]*\])', hr[0]['args'][0])
    r['tab'] = '[' + ', '.join(x.strip().strip('()') for x in tab.group(1)[1:-1].split(',')) + ']'
    r['help'] = [hhl01, hhl, hhr01, hhr, hH]
    H = o(hH)
    sw = fin[6:10]  # switch [[s6, s7], [s8, s9]]
    Xb = [Fl.find('concatRows', sw[0], sw[2]), Fl.find('concatRows', sw[1], sw[3])]
    D1 = [Fl.find('gadgetDecomposeRuns', o(x), pos=None) if False else
          next(a for a in Fl.atoms if a['head'].endswith('gadgetDecomposeRuns') and wire_of(a['args'][-2]) == o(x))
          for x in Xb]
    D = [Fl.find('compactCenteredRebaseRuns', None, o(x)) for x in D1]
    bd = [Fl.let('matrixMul', both, o(x)) for x in D]
    r.update(Xb=Xb, D1=D1, D=D)
    r['M0'] = [Fl.slice(x, 1, 2) for x in bd]
    r['M1'] = [Fl.slice(x, 2, 3) for x in bd]
    r['X'] = [Fl.slice(x, 0, 1) for x in bd]
    bms = lambda w: next(a for a in Fl.atoms if a['head'].endswith('blockModSwitchRuns') and wire_of(a['args'][-2]) == w)
    r['Asw'] = [[bms(o(x)) for x in r['M0']], [bms(o(x)) for x in r['M1']]]
    r['Z'] = [Fl.find('concatRows', o(r['Asw'][0][j]), o(r['Asw'][1][j])) for j in range(2)]
    r['E1'] = [next(a for a in Fl.atoms if a['head'].endswith('gadgetDecomposeRuns') and wire_of(a['args'][-2]) == o(z))
               for z in r['Z']]
    r['E'] = [Fl.find('compactCenteredRebaseRuns', None, o(x)) for x in r['E1']]
    he = [Fl.let('matrixMul', H, o(x)) for x in r['E']]
    r['Y'] = [Fl.slice(x, 0, 1) for x in he]
    r['K0'] = [Fl.slice(x, 1, 2) for x in he]
    r['K1'] = [Fl.slice(x, 2, 3) for x in he]
    # the move down (round) or not (branches)
    key_ = [[fin[2], fin[3]], [fin[4], fin[5]]]
    try:
        Kp = [[bms(Fl.let('matrixAdd', key_[k][j], o(r['K0' if k == 0 else 'K1'][j]))) for j in range(2)]
              for k in range(2)]
        Rp = [bms(Fl.let('matrixAdd', fin[j], Fl.let('matrixAdd', o(r['Y'][j]), o(r['X'][j])))) for j in range(2)]
        r['kind'] = 'round'
        r['Kp'], r['Rp'] = Kp, Rp
    except (KeyError, StopIteration):
        r['kind'] = 'branches'
    return r


# --------------------------------------------------------------------------- moduli and certificates

def ring_n(wires):
    """The ring degree `n` of a scope (the `n` of its `ExactMatrix q n r c` wires)."""
    ns = {mat_type(v['type'])[2] for v in wires.values() if 'ExactMatrix' in v['type']}
    assert len(ns) == 1, ns
    return ns.pop()


def mat_type(t):
    """`… ExactMatrix q n r c` (possibly under `Fin W →`) -> (W or None, q, n, r, c)."""
    import xmodel
    t = xmodel.strip_parens(t)
    W = None
    m = re.match(r'^\(?(?:\(m\d+ : )?Fin (\d+)\)? → (.*)$', t)
    if m:
        W, t = int(m.group(1)), m.group(2)
    t = xmodel.strip_parens(t).split('ExactMatrix ', 1)[1]
    parts = xmodel.split_top(t.strip(), ' ')
    return W, xmodel.strip_parens(parts[0]), parts[1], parts[2], parts[3]


def qterm(q):
    """Lean term of a modulus as printed in a wire type."""
    return q if re.match(r'^\w+$', q) else f'({q})'


def qname(q):
    m = re.match(r'^\(?(mod_\d+) \(\(?Int\.ofNat i_0\)?\)\)?$', q)
    return (m.group(1), True) if m else (q, False)


class Certs:
    def __init__(self, module):
        self.M = module

    def gt(self, q):
        n, inst = qname(q)
        return f'(RefreshTopProof.GenInst.Certs.{self.M}.{n}_gt i_0)' if inst else '(by decide)'

    def div(self, q, q2):
        a, ia = qname(q)
        b, ib = qname(q2)
        if ia or ib:
            return (f'(RefreshTopProof.GenInst.Certs.{self.M}.{a}_div_{b} i_0)',
                    f'(RefreshTopProof.GenInst.Certs.{self.M}.{a}_quot_{b} i_0)')
        Q, Q2 = qterm(q), qterm(q2)
        return (f'(by decide : {Q} = {Q} / {Q2} * {Q2})', f'(by decide : 1 < {Q} / {Q2})')

    def layout(self, q, nn):
        n, inst = qname(q)
        if inst:
            return f'RefreshTopProof.GenInst.Certs.{self.M}.{n}_layout i_0'
        return f'(⟨_, by simp [Backend.backend]; rfl, rfl⟩ : ∃ L : MxxRuntime.RegularLayout {qterm(q)}, Backend.backend.regularLayout {qterm(q)} {nn} = some L ∧ L.droppedModuli = 0)'


def two(xs):
    return f'(Fin.forall_fin_two.mpr ⟨{xs[0]}, {xs[1]}⟩)'


def emit_root_lut(I, a, r, C, name):
    """A segment theorem for the root LUT round `r` (from `root_lut_match`) at loop atom `a`."""
    W_ = lambda w: I.W(w)
    F = I.F
    o = lambda at: wire_of(at['args'][-1])
    _, q, nn, _, m = mat_type(I.wires[r['x'][0]]['type'])
    Wf, q2, _, _, m2 = mat_type(I.wires[r['f'][0]]['type'])
    Wh = mat_type(I.wires[r['hf0']]['type'])[0]
    log = int(nn).bit_length() - 1
    p = f'({qterm(q)} / {qterm(q2)})'
    hq, hp = C.div(q, q2)
    tab, tag, lev = r['tab'], r['tag'], r['level']
    child = a['loop']['child']
    x = ' '.join(W_(w) for w in r['x'])
    fams = ' '.join(W_(w) for w in r['f'])
    Q = qterm(q)
    Q2 = qterm(q2)
    sig = f'(sig {tab} {Wh})'
    hyp = f'''theorem {name} {{backend : BackendContext}} {{hashModel : HashModel}} {{params : Params}} {{i_0 : Nat}}
    [Fact (i_0 < S)] {{ctx : {I.inst}.Ctx backend hashModel params i_0}} {{outputs}}
    (F : {I.inst}.Facts ctx outputs) (hb : backend = Backend.backend)
    {{sm sp μ : ErrorPoly {nn}}} (hsm : polyNorm sm ≤ 1)
    {{g' : ExactMatrix {Q2} {nn} 1 {m}}} {{g : ExactMatrix {Q} {nn} 1 {m}}}
    (hg' : gadgetMatrixRuns backend {GBASE} {m} g') (hg : centeredRebaseRuns g' g) {{B : Nat}}
    (hinput : MxxExponentLut.Rel sm (inA {x})
      (fun j ↦ reducePoly {Q} {nn} ((({p} : Nat) : Int) * μ * unitPair sp j) • g) (inC {x}) B)
    {{Bh : Nat}} (hodd : ∀ k, Odd ({sig} k).toNat)
    (hhelp : ∀ k : Fin {Wh}, MxxExponentLut.Rel sm (lutAG hashModel {W_(r['key'])} {Q} {m} {tag} {lev} ({sig} k))
      (fun j ↦ reducePoly {Q} {nn} ((({p} : Nat) : Int) *
        unitPair (substitution ({sig} k).toNat (hodd k) sm) j) • g) ![{W_(r['hf0'])} k, {W_(r['hf1'])} k] Bh)'''
    L = r['count']
    pre = f'''  subst hb
{layout_obtain(C, q2, ['L', 'hL', 'hLx', '-'])}  have hq := {hq}
  have hchild := LutRoundStep.hstep (child := fun i s nxt ↦ {child} Backend.backend hashModel params i_0 i s nxt)
    (f00 := {W_(r['f'][0])}) (f01 := {W_(r['f'][1])}) (f10 := {W_(r['f'][2])}) (f11 := {W_(r['f'][3])})
    (fun i cur nxt hc ↦ {child}.sem hc) rfl (by decide) (log := {log}) rfl hq {C.gt(q)} {C.gt(q2)} L hL hLx
    hodd hsm hg' hg hinput hhelp
  have hloop := loop_of hchild (L := {L}) (by decide) {' '.join(F(g) for g in r['init_gets'])} {F(a)}
  obtain ⟨_, h3a⟩ := familyGet_eq (k := {L}) {F(r['h3'][0])}
  obtain ⟨_, h3b⟩ := familyGet_eq (k := {L}) {F(r['h3'][1])}
  obtain ⟨a00, a01, a10, a11⟩ := lutA_of_samples {' '.join(F(s) for s in r['samples'])}
  have hhelp1 := hhelp ⟨{L}, by decide⟩
  rw [show (![{W_(r['hf0'])} ⟨{L}, by decide⟩, {W_(r['hf1'])} ⟨{L}, by decide⟩] : Fin 2 → _) = ![{W_(o(r['h3'][0]))}, {W_(o(r['h3'][1]))}] by rw [h3a, h3b]] at hhelp1
  rw [show lutAG hashModel {W_(r['key'])} {Q} {m} {tag} {lev} ({sig} ⟨{L}, by decide⟩) = ![![{W_(o(r['samples'][0]))}, {W_(o(r['samples'][1]))}], ![{W_(o(r['samples'][2]))}, {W_(o(r['samples'][3]))}]] by
    rw [a00, a01, a10, a11]; funext a b; fin_cases a <;> fin_cases b <;> rfl] at hhelp1
'''
    w2 = lambda key: f"![{W_(o(r[key][0]))}, {W_(o(r[key][1]))}]"
    common = f'''    hloop.1 hloop.2 hhelp1 {' '.join(F(h) for h in r['hr'])}
    {' '.join(F(h) for h in r['enc'])}
    (Xb := {w2('Xb')}) {two([F(h) for h in r['Xb']])}
    (D' := {w2('D1')}) {two([F(h) for h in r['D1']])}
    (D := {w2('D')}) {two([F(h) for h in r['D']])}
    (M := ![{w2('M0')}, {w2('M1')}]) {two([F(h) for h in r['M0']])} {two([F(h) for h in r['M1']])}
    (Asw := ![![{W_(o(r['Asw'][0][0]))}, {W_(o(r['Asw'][0][1]))}], ![{W_(o(r['Asw'][1][0]))}, {W_(o(r['Asw'][1][1]))}]])
    (Fin.forall_fin_two.mpr ⟨{two([F(h) for h in r['Asw'][0]])}, {two([F(h) for h in r['Asw'][1]])}⟩)
    (Z := {w2('Z')}) {two([F(h) for h in r['Z']])}
    (E' := {w2('E1')}) {two([F(h) for h in r['E1']])}
    (E := {w2('E')}) {two([F(h) for h in r['E']])}
    {' '.join(F(h) for h in r['help'])}
    (Y := {w2('Y')}) (X := {w2('X')}) (K := ![{w2('K0')}, {w2('K1')}])
    {two([F(h) for h in r['Y']])} {two([F(h) for h in r['X']])} {two([F(h) for h in r['K0']])} {two([F(h) for h in r['K1']])}'''
    lbb = f'lutBranchBoundG {nn} {m} {GHALF} {p} B Bh'
    if r['kind'] == 'round':
        hyp += f'''
    {{Δ' ν : ErrorPoly {nn}}} {{gq : ExactMatrix {Q2} {nn} 1 {m2}}} {{C : Fin {Wh} → ExactPoly {Q2} {nn}}} {{BB : Nat}}
    (hswitch : ∀ k : Fin {Wh}, ∃ e : Fin 2 → ErrorMatrix {nn} 1 {m2}, (∀ j, CoeffBound (e j) BB) ∧ ∀ j,
      reducePoly {Q2} {nn} (substitution ({sig} k).toNat (hodd k) sp) • fam4 {fams} 0 j k + fam4 {fams} 1 j k =
        (C k * reducePoly {Q2} {nn} (Δ' * unitPair sp j)) • gq + reduceMatrix {Q2} {nn} 1 {m2} (e j))
    (hμ : ∀ k, polyNorm (substitution ({sig} k).toNat (hodd k) μ) ≤ 1)
    (hwalsh : ∑ k, reducePoly {Q2} {nn} (substitution ({sig} k).toNat (hodd k) μ) * C k = reducePoly {Q2} {nn} ν) :
    MxxExponentLut.Rel sm ![![{W_(o(r['Kp'][0][0]))}, {W_(o(r['Kp'][0][1]))}], ![{W_(o(r['Kp'][1][0]))}, {W_(o(r['Kp'][1][1]))}]]
      (fun j ↦ reducePoly {Q2} {nn} (Δ' * ν * unitPair sp j) • gq) ![{W_(o(r['Rp'][0]))}, {W_(o(r['Rp'][1]))}]
      (({L} * {lbb} + {lbb} + ({nn} + 2) * ({p} / 2)) / {p} + {Wh} * ({nn} * 1 * BB)) := by
'''
        body = pre + f'''  exact lut_round_gen (by decide) (log := {log}) rfl (L := {L}) (by decide) hq {C.gt(q)} {C.gt(q2)}
    (by have := {hp}; omega) L hL hLx {sig} hodd hsm hg' hg hinput
{common}
    hswitch hμ hwalsh
    (Fin.forall_fin_two.mpr ⟨{two([F(h) for h in r['Kp'][0]])}, {two([F(h) for h in r['Kp'][1]])}⟩)
    {two([F(h) for h in r['Rp']])}
'''
    else:
        fin = r['final']
        hyp += f''' :
    MxxExponentLut.Rel sm (fun k j ↦ ![![{W_(fin[2])}, {W_(fin[3])}], ![{W_(fin[4])}, {W_(fin[5])}]] k j + ![{w2('K0')}, {w2('K1')}] k j)
      (fun j ↦ ∑ i : Fin {Wh}, branchTargetG (q := {Q}) (p := {p}) {sig} hodd μ sp (fam4 {fams}) i j)
      (fun j ↦ ![{W_(fin[0])}, {W_(fin[1])}] j + ({w2('Y')} j + {w2('X')} j)) ({L} * {lbb} + {lbb}) := by
'''
        body = pre + f'''  exact lut_branches_gen (by decide) (log := {log}) rfl (L := {L}) (by decide) hq {C.gt(q)} {C.gt(q2)}
    L hL hLx {sig} hodd hsm hg' hg hinput
{common}
'''
    return hyp + '\n' + body


def emit_root_lut_keys(I, a, r, C, name):
    """The keys of the root LUT `r` (`GenKeyChain`): for a round (move down), the moved-down summed
    branch keys are the preprocessing's `keyBmsG (∑ lutKeysG …)` (`lut_root_keys`); for branches (no
    move down), the looped plus unrolled branch keys are `∑ lutKeysG …` (`branch_keys_core`)."""
    W_ = lambda w: I.W(w)
    F = I.F
    o = lambda at: wire_of(at['args'][-1])
    _, q, nn, _, m = mat_type(I.wires[r['x'][0]]['type'])
    Wf, q2, _, _, m2 = mat_type(I.wires[r['f'][0]]['type'])
    tab, tag, lev = r['tab'], r['tag'], r['level']
    child = a['loop']['child']
    L = r['count']
    x = ' '.join(W_(w) for w in r['x'])
    fams = ' '.join(W_(w) for w in r['f'])
    gd = r['D1'][0]['args']
    base, digits = gd[1].strip('()'), gd[2].strip('()')
    basis = r['Asw'][0][0]['args'][0]
    w2 = lambda key: f"![{W_(o(r[key][0]))}, {W_(o(r[key][1]))}]"
    xs = ' '.join(f'(x{k} := {W_(w)})' for k, w in enumerate(r['x']))
    fs = ' '.join(f'({n} := {W_(w)})' for n, w in zip(('f00', 'f01', 'f10', 'f11'), r['f']))
    lsum = (f'∑ k ∈ Finset.range ({L} + 1), lutKeysG Backend.backend ({base} : Int) ({digits} : Int)\n'
            f'      {basis} hashModel {W_(r["key"])} {tag} {lev} (sigN {tab}) (inA {x}) (fam4 {fams}) k')
    fin = r['final']
    Kw = f"![{w2('K0')}, {w2('K1')}]"
    if r['kind'] == 'round':
        basis2 = r['Kp'][0][0]['args'][0]
        Kp = f"![![{W_(o(r['Kp'][0][0]))}, {W_(o(r['Kp'][0][1]))}], ![{W_(o(r['Kp'][1][0]))}, {W_(o(r['Kp'][1][1]))}]]"
        concl = f'keyTupleG {Kp} = keyBmsG {basis2} ({lsum})'
        doc = f"/-- **The keys of the root LUT** `{a['field']}` (no semantic hypothesis; `lut_root_keys`). -/"
    else:
        Fk = f"![![{W_(fin[2])}, {W_(fin[3])}], ![{W_(fin[4])}, {W_(fin[5])}]]"
        concl = f'keyTupleG (fun k j ↦ {Fk} k j + {Kw} k j) = {lsum}'
        doc = f"/-- **The keys of the root LUT branches** `{a['field']}` (no move down; `branch_keys_core`). -/"
    core = f'''branch_keys_core hchild (L := {L}) rfl {' '.join(F(g) for g in r['init_gets'])} {F(a)}
    (by decide) {C.gt(q2)} L hL hLx (by rw [hd])
    {' '.join(F(h) for h in r['hr'])}
    {' '.join(F(h) for h in r['enc'])}
    (Xb := {w2('Xb')}) {two([F(h) for h in r['Xb']])}
    (D' := {w2('D1')}) {two([F(h) for h in r['D1']])}
    (D := {w2('D')}) {two([F(h) for h in r['D']])}
    (M := ![{w2('M0')}, {w2('M1')}]) {two([F(h) for h in r['M0']])} {two([F(h) for h in r['M1']])}
    (Asw := ![![{W_(o(r['Asw'][0][0]))}, {W_(o(r['Asw'][0][1]))}], ![{W_(o(r['Asw'][1][0]))}, {W_(o(r['Asw'][1][1]))}]])
    (Fin.forall_fin_two.mpr ⟨{two([F(h) for h in r['Asw'][0]])}, {two([F(h) for h in r['Asw'][1]])}⟩)
    (Z := {w2('Z')}) {two([F(h) for h in r['Z']])}
    (E' := {w2('E1')}) {two([F(h) for h in r['E1']])}
    (E := {w2('E')}) {two([F(h) for h in r['E']])}
    (h := ![{W_(o(r['h3'][0]))}, {W_(o(r['h3'][1]))}])
    (Ad := ![![{W_(o(r['samples'][0]))}, {W_(o(r['samples'][1]))}], ![{W_(o(r['samples'][2]))}, {W_(o(r['samples'][3]))}]])
    (by rw [a00, a01, a10, a11]; funext a b; fin_cases a <;> fin_cases b <;> rfl)
    {' '.join(F(h) for h in r['help'])}
    (K := {Kw}) {two([F(h) for h in r['K0']])} {two([F(h) for h in r['K1']])}'''
    if r['kind'] == 'round':
        fin_line = (f"  exact lut_root_keys ({core})\n"
                    f"    (Fin.forall_fin_two.mpr ⟨{two([F(h) for h in r['Kp'][0]])}, {two([F(h) for h in r['Kp'][1]])}⟩)\n")
    else:
        fin_line = f"  exact {core}\n"
    return f'''{doc}
theorem {name} {{backend : BackendContext}} {{hashModel : HashModel}} {{params : Params}} {{i_0 : Nat}}
    [Fact (i_0 < S)] {{ctx : {I.inst}.Ctx backend hashModel params i_0}} {{outputs}}
    (F : {I.inst}.Facts ctx outputs) (hb : backend = Backend.backend) :
    {concl} := by
  subst hb
{layout_obtain(C, q2, ['L', 'hL', 'hLx', 'hd'])}  have hchild := LutRoundStep.keystep (child := fun i s nxt ↦ {child} Backend.backend hashModel params i_0 i s nxt)
    {fs} {xs} (hf0 := {W_(r['hf0'])}) (key := {W_(r['key'])}) (hf1 := {W_(r['hf1'])})
    (fun _ _ _ hc ↦ {child}.sem hc) rfl (by decide) {C.gt(q2)} L hL hLx (by rw [hd])
  obtain ⟨a00, a01, a10, a11⟩ := lutA_of_samples {' '.join(F(s) for s in r['samples'])}
{fin_line}'''


def emit_root_private_keys(I, r, C, name):
    """The keys of the root private multiplication `r` (`root_private_match`): the moved-down product
    keys are `privKeyG` of the input keys, the hash-sampled tops (`topPairG`) and the bottoms
    (`privKeyG_of_stacks`, no semantic hypothesis)."""
    W_ = I.W
    F = I.F
    o = lambda a: wire_of(a['args'][-1])
    J = r['j']
    _, q, nn, _, m = mat_type(I.wires[r['c'][0]]['type'])
    _, q2, _, _, m2 = mat_type(I.wires[o(J[0]['cX'])]['type'])
    top = J[0]['top']
    key = wire_of(top['args'][3])
    prefix = top['args'][1].strip('()')
    pre = pre_of(top)
    gd = J[0]['D1']['args']
    base, digits = gd[1].strip('()'), gd[2].strip('()')
    basis = J[0]['cM0']['args'][0]
    A = f"![![{W_(r['A'][0][0])}, {W_(r['A'][0][1])}], ![{W_(r['A'][1][0])}, {W_(r['A'][1][1])}]]"
    c = f"![{W_(r['c'][0])}, {W_(r['c'][1])}]"
    w2 = lambda k: f"![{W_(o(J[0][k]))}, {W_(o(J[1][k]))}]"
    hd = lambda k: two([F(J[0][k]), F(J[1][k])])
    tp = f"topPairG hashModel ({W_(key)}) {qterm(q2)} {m2} {prefix} {pre}"
    return f'''/-- **The keys of the root private multiplication** `{r['both']['field']}` (`privKeyG_of_stacks`). -/
theorem {name} {{backend : BackendContext}} {{hashModel : HashModel}} {{params : Params}} {{i_0 : Nat}}
    [Fact (i_0 < S)] {{ctx : {I.inst}.Ctx backend hashModel params i_0}} {{outputs}}
    (F : {I.inst}.Facts ctx outputs) (hb : backend = Backend.backend) :
    ![{w2('cM0')}, {w2('cM1')}] = privKeyG Backend.backend ({base} : Int) ({digits} : Int) {basis} {A}
      ({tp}) ![{W_(J[0]['bot'])}, {W_(J[1]['bot'])}] := by
  subst hb
{layout_obtain(C, q2, ['L', 'hL', 'hLx', 'hd'])}  have htop : (![{W_(o(J[0]['top']))}, {W_(o(J[1]['top']))}] : Fin 2 → _) = {tp} := by
    funext j; fin_cases j
    · exact hashSample_eq {F(J[0]['top'])}
    · exact hashSample_eq {F(J[1]['top'])}
  rw [← htop]
  exact privKeyG_of_stacks (by decide) {C.gt(q2)} L hL hLx (by rw [hd]) (c := {c}) (A := {A})
    {' '.join(F(x) for x in r['enc'])}
    (top := ![{W_(o(J[0]['top']))}, {W_(o(J[1]['top']))}]) (bot := ![{W_(J[0]['bot'])}, {W_(J[1]['bot'])}])
    (Xb := {w2('Xb')}) {hd('Xb')} (D' := {w2('D1')}) {hd('D1')} (D := {w2('D')}) {hd('D')}
    (M := ![{w2('M0')}, {w2('M1')}]) {hd('M0')} {hd('M1')}
    (Fin.forall_fin_two.mpr ⟨{hd('cM0')}, {hd('cM1')}⟩)
'''


def layout_obtain(C, q, names):
    n, inst = qname(q)
    arg = f' i_0' if inst else ''
    return f'  obtain ⟨{", ".join(names)}⟩ := RefreshTopProof.GenInst.Certs.{C.M}.{n}_layout{arg}\n'


def emit_cell0(I, a, C, P, child_atoms, name):
    """Segment theorem of a bucket-0 PRF cell gather (`prf.cell0`): every cell's output encodes
    `p' (α_k μ s_i)` at the cell's table entry `k` (`PrfCell0.rel`), no unrolling."""
    L = a['loop']
    args = [x.strip() for x in L['args']]
    t = wire_of(args[0].replace(' i', ''))
    g = [x for (_, _, x) in L['gathered']]
    fams = [f for (f, _, _) in L['gathered']]
    rest = [wire_of(x.replace(' i', '')) for x in args[7:]]
    As00, As10, cs0, pay, As01, As11, cs1, seed, bot0, bot1 = rest
    W = lambda w: f'{I.inst}.{w} ctx'
    K, q, nn, _, m = mat_type(I.wires[fams[0]]['type'])
    _, _, _, _, m1 = mat_type(I.wires[As00]['type'])
    F_, q2, _, _, m2 = mat_type(I.wires[bot0]['type'])
    hs = [x for x in child_atoms if x['head'].endswith('hashSample')][0]
    pre = hs['args'][1].strip('()')
    comps = hs['args'][2]
    import xmodel
    items = xmodel.split_top(xmodel.strip_parens(comps)[1:-1], ', ')
    dec = [xmodel.strip_parens(x.strip()[len('.decimal '):]).replace('((Int.ofNat i_0))', '(Int.ofNat i_0)')
           for x in items if x.strip().startswith('.decimal')]
    Q, Q2 = qterm(q), qterm(q2)
    p = f'({Q} / {Q2})'
    hq, hp = C.div(q, q2)
    outs = ', '.join(f'{I.inst}.{wire_of(o.replace(" i", ""))} ctx i' for o in L['outs'])
    return f'''theorem {name} {{backend : BackendContext}} {{hashModel : HashModel}} {{params : Params}} {{i_0 : Nat}}
    [Fact (i_0 < S)] {{ctx : {I.inst}.Ctx backend hashModel params i_0}} {{outputs}}
    (F : {I.inst}.Facts ctx outputs) (hb : backend = Backend.backend)
    {{sm sp μ : ErrorPoly {nn}}} (hsm : polyNorm sm ≤ 1)
    {{α : Fin {K} → ErrorPoly {nn}}} (hα : ∀ k, SmulGain (α k) 1) (hαμ : ∀ k, polyNorm (α k * μ) ≤ 1)
    {{G : ExactMatrix {Q} {nn} 1 {m}}} (hG : gadgetMatrixRuns backend {GBASE} {m} G) {{Ba : Nat}}
    (ha : ∀ k : Fin {K}, MxxExponentLut.Rel sm ![![{W(fams[1])} k, {W(fams[4])} k], ![{W(fams[2])} k, {W(fams[5])} k]]
      (fun j ↦ reducePoly {Q} {nn} (α k * unitPair sm j) • G) ![{W(fams[0])} k, {W(fams[3])} k] Ba)
    (hpay : ∀ k : Fin {K}, {W(pay)} k = fun _ _ ↦ reducePoly {Q} {nn} (α k))
    {{g' : ExactMatrix {Q2} {nn} 1 {m1}}} (hg' : gadgetMatrixRuns backend {GBASE} {m1} g')
    {{g : ExactMatrix {Q} {nn} 1 {m1}}} (hg : centeredRebaseRuns g' g) {{Bs : Nat}}
    (hs : MxxExponentLut.Rel sm ![![{W(As00)}, {W(As01)}], ![{W(As10)}, {W(As11)}]]
      (fun j ↦ reducePoly {Q} {nn} ((({p} : Nat) : Int) * μ * unitPair sp j) • g) ![{W(cs0)}, {W(cs1)}] Bs)
    {{p' : Nat}} {{ν : Fin {F_} → ErrorPoly {nn}}} {{hT : ExactMatrix {Q2} {nn} 1 {m2}}}
    {{eB : Fin {F_} → Fin 2 → ErrorMatrix {nn} 1 {m2}}} {{BB : Nat}}
    (hrhs : ∀ (i : Fin {F_}) j, reducePoly {Q2} {nn} sp • topPairG hashModel ({W(seed)}) {Q2} {m2} {pre}
        (prfTopTags ({dec[0]}) ({dec[1]}) i) j + ![{W(bot0)} i, {W(bot1)} i] j =
      reducePoly {Q2} {nn} ((p' : Int) * ν i * unitPair sp j) • hT +
        reduceMatrix {Q2} {nn} 1 {m2} (eB i j))
    (heB : ∀ i j, CoeffBound (eB i j) BB) :
    ∀ i : Fin {F_}, ∃ k : Fin {K}, ((k : Nat) : Int) = {W(t)} i ∧
      EncRel sm ({outs}, ())
        (fun j ↦ reducePoly {Q2} {nn} ((p' : Int) * (α k * μ * ν i) * unitPair sp j) • hT)
        (cellBoundK {nn} {m} {m1} {GHALF} {p} Ba Bs BB 1) := by
  intro i
  obtain ⟨k, hk, hc⟩ := Gen.gather_loop6 {P} {I.F(a)} i
  subst hb
{layout_obtain(C, q, ['L', 'hL', 'hLx', 'hd'])}{layout_obtain(C, q2, ["L'", "hL'", "hLx'", "hd'"])}  refine ⟨k, hk, (PrfCell0.rel (by decide) {hq} {C.gt(q)} {C.gt(q2)} (by have := {hp}; omega)
    L hL hLx (by rw [hd]) L' hL' hLx' (by rw [hd'])
    (h := {L['child']}.sem hc) hsm hG (hα k) (hαμ k) (ha k) k hk.symm (hpay k) hg' hg hs
    (hrhs i) (heB i)).1⟩
'''


def emit_cell1(I, a, C, P, child_atoms, name):
    """Segment theorem of a bucket-1 PRF cell gather (`prf.cell1`, aligned from the labels' ring
    `Q` to the state ring `q`): every cell encodes `p' (α_k μ_o s_i)` with `o = ⌊i / P⌋`."""
    L = a['loop']
    args = [x.strip() for x in L['args']]
    t = wire_of(args[0].replace(' i', ''))
    fams = [f for (f, _, _) in L['gathered']]
    rest = [wire_of(x.replace(' i', '')) for x in args[7:]]
    As00, As10, cs0, pay, As01, As11, cs1, seed, bot0, bot1 = rest
    W = lambda w: f'{I.inst}.{w} ctx'
    K, Qa, nn, _, mA = mat_type(I.wires[fams[0]]['type'])
    Wm, q, _, _, m1 = mat_type(I.wires[As00]['type'])
    F_, q2, _, _, m2 = mat_type(I.wires[bot0]['type'])
    gm = [x for x in child_atoms if x['head'].endswith('gadgetMatrixRuns')][0]
    m = gm['args'][-2].strip('()')
    fd = [x for x in child_atoms if 'Int.fdiv' in str(x)]
    import xmodel
    hs = [x for x in child_atoms if x['head'].endswith('hashSample')][0]
    pre = hs['args'][1].strip('()')
    items = xmodel.split_top(xmodel.strip_parens(hs['args'][2])[1:-1], ', ')
    dec = [xmodel.strip_parens(x.strip()[len('.decimal '):]).replace('((Int.ofNat i_0))', '(Int.ofNat i_0)')
           for x in items if x.strip().startswith('.decimal')]
    QA, Q, Q2 = qterm(Qa), qterm(q), qterm(q2)
    pA = f'({QA} / {Q})'
    p = f'({Q} / {Q2})'
    hQ, hpA = C.div(Qa, q)
    hq, hp = C.div(q, q2)
    Pdiv = F_ // Wm
    outs = ', '.join(f'{I.inst}.{wire_of(o.replace(" i", ""))} ctx i' for o in L['outs'])
    return f'''theorem {name} {{backend : BackendContext}} {{hashModel : HashModel}} {{params : Params}} {{i_0 : Nat}}
    [Fact (i_0 < S)] {{ctx : {I.inst}.Ctx backend hashModel params i_0}} {{outputs}}
    (F : {I.inst}.Facts ctx outputs) (hb : backend = Backend.backend)
    {{sm sp : ErrorPoly {nn}}} {{μ : Fin {Wm} → ErrorPoly {nn}}} (hsm : polyNorm sm ≤ 1)
    {{α : Fin {K} → ErrorPoly {nn}}} (hα : ∀ k, SmulGain (α k) 1) (hαμ : ∀ k o, polyNorm (α k * μ o) ≤ 1)
    {{GA : ExactMatrix {QA} {nn} 1 {mA}}} (hGA : gadgetMatrixRuns backend {GBASE} {mA} GA) {{Ba : Nat}}
    (ha : ∀ k : Fin {K}, MxxExponentLut.Rel sm ![![{W(fams[1])} k, {W(fams[4])} k], ![{W(fams[2])} k, {W(fams[5])} k]]
      (fun j ↦ reducePoly {QA} {nn} (α k * unitPair sm j) • GA) ![{W(fams[0])} k, {W(fams[3])} k] Ba)
    (hpay : ∀ k : Fin {K}, {W(pay)} k = fun _ _ ↦ reducePoly {QA} {nn} (α k))
    {{G : ExactMatrix {Q} {nn} 1 {m}}} (hG : gadgetMatrixRuns backend {GBASE} {m} G)
    {{GR : ExactMatrix {QA} {nn} 1 {m}}} (hGR : centeredRebaseRuns G GR)
    {{g' : ExactMatrix {Q2} {nn} 1 {m1}}} (hg' : gadgetMatrixRuns backend {GBASE} {m1} g')
    {{g : ExactMatrix {Q} {nn} 1 {m1}}} (hg : centeredRebaseRuns g' g) {{Bs : Nat}}
    (hs : ∀ o : Fin {Wm}, MxxExponentLut.Rel sm ![![{W(As00)} o, {W(As01)} o], ![{W(As10)} o, {W(As11)} o]]
      (fun j ↦ reducePoly {Q} {nn} ((({p} : Nat) : Int) * μ o * unitPair sp j) • g) ![{W(cs0)} o, {W(cs1)} o] Bs)
    {{p' : Nat}} {{ν : Fin {F_} → ErrorPoly {nn}}} {{hT : ExactMatrix {Q2} {nn} 1 {m2}}}
    {{eB : Fin {F_} → Fin 2 → ErrorMatrix {nn} 1 {m2}}} {{BB : Nat}}
    (hrhs : ∀ (i : Fin {F_}) j, reducePoly {Q2} {nn} sp • topPairG hashModel ({W(seed)}) {Q2} {m2} {pre}
        (prfTopTags ({dec[0]}) ({dec[1]}) i) j + ![{W(bot0)} i, {W(bot1)} i] j =
      reducePoly {Q2} {nn} ((p' : Int) * ν i * unitPair sp j) • hT +
        reduceMatrix {Q2} {nn} 1 {m2} (eB i j))
    (heB : ∀ i j, CoeffBound (eB i j) BB) :
    ∀ i : Fin {F_}, ∃ k : Fin {K}, ((k : Nat) : Int) = {W(t)} i ∧ ∃ o : Fin {Wm}, ((o : Nat) : Int) = Int.fdiv (Int.ofNat i) {Pdiv} ∧
      EncRel sm ({outs}, ())
        (fun j ↦ reducePoly {Q2} {nn} ((p' : Int) * (α k * μ o * ν i) * unitPair sp j) • hT)
        (cellBoundK {nn} {m} {m1} {GHALF} {p} (({mA} * {nn} * Ba * (({GBASE} : Int).toNat / 2) + ({nn} + 2) * ({pA} / 2)) / {pA}) Bs BB 1) := by
  intro i
  obtain ⟨k, hk, hc⟩ := Gen.gather_loop6 {P} {I.F(a)} i
  subst hb
{layout_obtain(C, Qa, ['LA', 'hLA', 'hLxA', '-'])}{layout_obtain(C, q, ['L', 'hL', 'hLx', 'hd'])}{layout_obtain(C, q2, ["L'", "hL'", "hLx'", "hd'"])}  obtain ⟨o, ho, hr, -⟩ := PrfCell1.rel (by decide) {hQ} {C.gt(Qa)} (by have := {hpA}; omega) {hq} {C.gt(q)} {C.gt(q2)}
    (by have := {hp}; omega) LA hLA hLxA L hL hLx (by rw [hd]) L' hL' hLx' (by rw [hd']) (by decide)
    (h := {L['child']}.sem hc) hsm hGA (hα k) (hαμ k) (ha k) k hk.symm (hpay k) hG hGR hg' hg hs
    (hrhs i) (heB i)
  exact ⟨k, hk, o, ho, hr⟩
'''


MEMBER_BRIDGE = {'lut.member.readout': 'LutMemberReadout', 'lut.member.sampler': 'LutMemberSampler',
                 'lut.member.sampler-offset': 'LutMemberSamplerOffset', 'smudge.member': 'SmudgeMember'}


def lut_params(child_json):
    """(tag, level, tab) of a LUT member/step scope from its helper samples and automorphisms."""
    tag = level = tab = None
    for a in child_json['atoms']:
        if a['head'].endswith('hashSample') and '.integer' in a['args'][2] and tag is None:
            tag = a['args'][1].strip('()')
            level = re.search(r'\[\.u64Le \(\(?(\d+)\)?\), \.integer', a['args'][2]).group(1)
        if a['head'].endswith('ringAutomorphismRuns') and tab is None:
            m = re.search(r'selectInt \(?[^\[]*(\[[^\]]*\])', a['args'][0])
            if m:
                tab = '[' + ', '.join(x.strip().strip('()') for x in m.group(1)[1:-1].split(',')) + ']'
    return tag, level, tab


def emit_lut_member(I, a, C, role, scopes, name):
    """Segment theorem of a parallel LUT member loop (readout / sampler / sampler with offset):
    member `i` with input encoding of `p μ_i` encodes the LUT of `μ_i` (bridges of GenBridge_Lut)."""
    L = a['loop']
    br = MEMBER_BRIDGE[role]
    args = [wire_of(x.replace(' i', '')) for x in L['args']]
    per = [x.strip().endswith(' i') for x in L['args']]
    f00, f01, f10, f11, hf0, key, hf1 = args[:7]
    xs = args[7:13]
    offs = args[13:19] if role == 'lut.member.sampler-offset' else []
    cj = scopes[L['child']].to_json()
    step = [x for x in cj['atoms'] if x.get('loop')][0]['loop']['child']
    tag, level, tab = lut_params(scopes[step].to_json())
    W = lambda w: f'{I.inst}.{w} ctx'
    M = L['count']
    _, q, nn, _, m = mat_type(I.wires[xs[0]]['type'])
    Wh, q2, _, _, m2 = mat_type(I.wires[f00]['type'])
    Wh = mat_type(I.wires[hf0]['type'])[0]
    log = int(nn).bit_length() - 1
    Q, Q2 = qterm(q), qterm(q2)
    p = f'({Q} / {Q2})'
    hq, hp = C.div(q, q2)
    xi = ' '.join(f'({W(x)} i)' for x in xs)
    fams = ' '.join(f'({W(f)})' for f in (f00, f01, f10, f11))
    sig = f'(sig {tab} {Wh})'
    outs = ', '.join(f'{W(wire_of(o.replace(" i", "")))} i' for o in L['outs'])
    oq = mat_type(I.wires[wire_of(L['outs'][0].replace(' i', ''))]['type'])
    hyp = f'''theorem {name} {{backend : BackendContext}} {{hashModel : HashModel}} {{params : Params}} {{i_0 : Nat}}
    [Fact (i_0 < S)] {{ctx : {I.inst}.Ctx backend hashModel params i_0}} {{outputs}}
    (F : {I.inst}.Facts ctx outputs) (hb : backend = Backend.backend)
    {{sm sp : ErrorPoly {nn}}} {{μ : Fin {M} → ErrorPoly {nn}}} (hsm : polyNorm sm ≤ 1)
    {{g' : ExactMatrix {Q2} {nn} 1 {m}}} {{g : ExactMatrix {Q} {nn} 1 {m}}}
    (hg' : gadgetMatrixRuns backend {GBASE} {m} g') (hg : centeredRebaseRuns g' g) {{B : Nat}}
    (hinput : ∀ i : Fin {M}, MxxExponentLut.Rel sm (inA {xi})
      (fun j ↦ reducePoly {Q} {nn} ((({p} : Nat) : Int) * μ i * unitPair sp j) • g) (inC {xi}) B)
    {{Bh : Nat}} (hodd : ∀ k, Odd ({sig} k).toNat)
    (hhelp : ∀ k : Fin {Wh}, MxxExponentLut.Rel sm (lutAG hashModel ({W(key)}) {Q} {m} {tag} {level} ({sig} k))
      (fun j ↦ reducePoly {Q} {nn} ((({p} : Nat) : Int) *
        unitPair (substitution ({sig} k).toNat (hodd k) sm) j) • g) ![{W(hf0)} k, {W(hf1)} k] Bh)'''
    lbb = f'lutBranchBoundG {nn} {m} {GHALF} {p} B Bh'
    tgt = f'(fun j ↦ ∑ k : Fin {Wh}, branchTargetG (q := {Q}) (p := {p}) {sig} hodd (μ i) sp (fam4 {fams}) k j)'
    extra = ''
    if role == 'lut.member.readout':
        hyp += f'''
    {{Δ' : ErrorPoly {nn}}} {{ν : Fin {M} → ErrorPoly {nn}}} {{gq : ExactMatrix {Q2} {nn} 1 {m2}}} {{C : Fin {Wh} → ExactPoly {Q2} {nn}}} {{BB : Nat}}
    (hswitch : ∀ k : Fin {Wh}, ∃ e : Fin 2 → ErrorMatrix {nn} 1 {m2}, (∀ j, CoeffBound (e j) BB) ∧ ∀ j,
      reducePoly {Q2} {nn} (substitution ({sig} k).toNat (hodd k) sp) • fam4 {fams} 0 j k + fam4 {fams} 1 j k =
        (C k * reducePoly {Q2} {nn} (Δ' * unitPair sp j)) • gq + reduceMatrix {Q2} {nn} 1 {m2} (e j))
    (hμ : ∀ i k, polyNorm (substitution ({sig} k).toNat (hodd k) (μ i)) ≤ 1)
    (hwalsh : ∀ i, ∑ k, reducePoly {Q2} {nn} (substitution ({sig} k).toNat (hodd k) (μ i)) * C k = reducePoly {Q2} {nn} (ν i)) :
    ∀ i : Fin {M}, EncRel sm ({outs}, ()) (fun j ↦ reducePoly {Q2} {nn} (Δ' * ν i * unitPair sp j) • gq)
      (({Wh - 1} * {lbb} + {lbb} + ({nn} + 2) * ({p} / 2)) / {p} + {Wh} * ({nn} * 1 * BB)) := by
'''
        extra = '(hp := by have := ' + hp + '; omega) (hswitch := hswitch) (hμ := hμ i) (hwalsh := hwalsh i)'
    elif role == 'lut.member.sampler':
        hyp += f''' :
    ∀ i : Fin {M}, EncRel sm ({outs}, ()) {tgt} ({Wh - 1} * {lbb} + {lbb}) := by
'''
    elif role == 'lut.member.sampler-offset':
        ot = ', '.join(W(o) for o in offs)
        hyp += f'''
    {{toff : Fin 2 → ExactMatrix {Q} {nn} 1 {m2}}} {{Boff : Nat}} (hoff : EncRel sm ({ot}, ()) toff Boff) :
    ∀ i : Fin {M}, EncRel sm ({outs}, ()) (fun j ↦ {tgt} j + toff j) ({Wh - 1} * {lbb} + {lbb} + Boff) := by
'''
        extra = '(hoff := hoff)'
    elif role == 'smudge.member':
        lets = [v.get('term', '') for v in cj['wires'].values()]
        fm = [re.search(r'Int\.fmod \(\(Int\.fdiv \(\(Int\.ofNat i_1\)\) \((\d+)\)\)\) \((\d+)\)', x) for x in lets]
        fm = [x for x in fm if x][0]
        dd, NN = int(fm.group(1)), int(fm.group(2))
        # the weight `c · lift(P^(i mod d))`: one `matrixPolynomial [c]` and one power lift
        cw = [re.search(r'matrixPolynomial \[(.*)\] :', x).group(1) for x in lets if x.startswith('(MxxRuntime.matrixPolynomial')]
        pw = [re.search(r'liftInteger \(\((.*?)\) \^ \(\(Int\.fmod', x) for x in lets if x.startswith('(MxxRuntime.liftInteger')]
        pw = [x.group(1) for x in pw if x]
        assert len(cw) == 1 and len(pw) == 1, (cw, pw)
        cw = cw[0].replace('((Int.ofNat i_0))', '(Int.ofNat i_0)')
        _, qo, _, _, mo = oq
        hyp += f'''
    {{Δ' : ErrorPoly {nn}}} {{ν : Fin {M} → ErrorPoly {nn}}} {{gq : ExactMatrix {Q2} {nn} 1 {m2}}} {{C : Fin {Wh} → ExactPoly {Q2} {nn}}} {{BB : Nat}}
    (hswitch : ∀ k : Fin {Wh}, ∃ e : Fin 2 → ErrorMatrix {nn} 1 {m2}, (∀ j, CoeffBound (e j) BB) ∧ ∀ j,
      reducePoly {Q2} {nn} (substitution ({sig} k).toNat (hodd k) sp) • fam4 {fams} 0 j k + fam4 {fams} 1 j k =
        (C k * reducePoly {Q2} {nn} (Δ' * unitPair sp j)) • gq + reduceMatrix {Q2} {nn} 1 {m2} (e j))
    (hμ : ∀ i k, polyNorm (substitution ({sig} k).toNat (hodd k) (μ i)) ≤ 1)
    (hwalsh : ∀ i, ∑ k, reducePoly {Q2} {nn} (substitution ({sig} k).toNat (hodd k) (μ i)) * C k = reducePoly {Q2} {nn} (ν i))
    (hG : gadgetMatrixRuns backend {GBASE} {m2} gq) :
    ∀ i : Fin {M}, EncRel sm ({outs}, ()) (fun j ↦ reducePoly {Q2} {nn} (Xe (n := {nn}) (i / {dd} % {NN})) •
        reducePoly {Q2} {nn} (((({cw}) * ({pw[0]}) ^ ((i : Nat) % {dd}) : Int) : ErrorPoly {nn}) * (Δ' * ν i) * unitPair sp j) •
          (1 : ExactMatrix {Q2} {nn} 1 1))
      ({m2} * (({Wh - 1} * {lbb} + {lbb} + ({nn} + 2) * ({p} / 2)) / {p} + {Wh} * ({nn} * 1 * BB)) * (({GBASE} : Int).toNat / 2)) := by
'''
        extra = (f'(hp := by have := {hp}; omega) (hswitch := hswitch) (hμ := hμ i) (hwalsh := hwalsh i) (d := {dd}) (hD := rfl) (N := {NN}) '
                 f'(hNN := rfl) (hN := by decide) (hN2 := by decide) (layoutG := L) (hlayoutG := hL) (hexactG := hLx) (hG := hG)')
    else:
        return None
    body = f'''  intro i
  subst hb
{layout_obtain(C, q2, ['L', 'hL', 'hLx', '-'])}  have hq := {hq}
  exact {br}.rel (h := {L['child']}.sem ({I.F(a)} i)) (hmw := rfl) (L := {Wh - 1}) (hℓ := rfl) (hL := rfl) (hn := by decide) (log := {log})
    (hpow := rfl) (hq := hq) (hq1 := {C.gt(q)}) (hq' := {C.gt(q2)}) (layout := L) (hlayout := hL) (hexact := hLx)
    (hodd := hodd) (hsm := hsm) (hg' := hg') (hg := hg) (hinput := hinput i) (hhelp := hhelp)
    (hchild := LutRoundStep.hstep (child := fun k s nxt ↦ {step} Backend.backend hashModel params i_0 i k s nxt)
      (f00 := {W(f00)}) (f01 := {W(f01)}) (f10 := {W(f10)}) (f11 := {W(f11)})
      (fun _ _ _ hc ↦ {step}.sem hc) rfl (by decide) (log := {log}) rfl hq {C.gt(q)} {C.gt(q2)} L hL hLx hodd hsm
      hg' hg (hinput i) hhelp) {extra}
'''
    return hyp + body


# --------------------------------------------------------------------------- composition of segments

SHARED = {'backend', 'hashModel', 'params', 'i_0', 'ctx', 'outputs', 'F', 'hb', 'sm', 'sp', 'hsm'}


def parse_theorem(text):
    """(name, binders, conclusion) of a generated `theorem name binders : concl := by` text;
    binders: list of (bracket, [names], type)."""
    m = re.match(r'theorem (\S+) ', text)
    name = m.group(1)
    rest = text[m.end():]
    binders, i = [], 0
    while True:
        while rest[i].isspace():
            i += 1
        if rest[i] not in '({[':
            break
        j = matching_close(rest, i)
        inner = rest[i + 1:j]
        if rest[i] == '[':
            binders.append(('[', [], inner))
        elif ' : ' in inner:
            names, ty = inner.split(' : ', 1)
            binders.append((rest[i], names.split(), ty))
        else:
            binders.append((rest[i], inner.split(), None))
        i = j + 1
    body = rest[i:]
    assert body.startswith(':'), body[:50]
    concl = body[1:body.rindex(':= by')].strip()
    return name, binders, concl


def matching_close(s, i):
    pairs = {'(': ')', '{': '}', '[': ']', '⟨': '⟩'}
    depth = 0
    for k in range(i, len(s)):
        if s[k] in pairs:
            depth += 1
        elif s[k] in pairs.values():
            depth -= 1
            if depth == 0:
                return k
    raise ValueError('unbalanced')


def rename_text(t, ren):
    if not ren:
        return t
    pat = re.compile(r"(?<![\w.'])(" + '|'.join(re.escape(k) for k in sorted(ren, key=len, reverse=True)) + r")(?![\w'])")
    return pat.sub(lambda m: ren[m.group(1)], t)


class Composer:
    """Accumulates the binders of several segment theorems (non-shared names renamed with a
    per-step suffix) and the `have` steps applying them with named arguments."""

    def __init__(self):
        self.binders = []
        self.seen = set()
        self.steps = []

    def add(self, seg_text, tag, links=None, res=None, prefix=''):
        """Apply segment `seg_text`; `links`: binder -> value expression (no chain binder) or
        None (left to unification); other non-shared binders become chain binders `<name>_<tag>`."""
        links = links or {}
        name, binders, concl = parse_theorem(seg_text)
        ren = {}
        for br, names, ty in binders:
            for x in names:
                if x not in SHARED and x not in links:
                    ren[x] = f'{x}_{tag}'
        args = []
        for br, names, ty in binders:
            if br == '[':
                continue
            keep = [x for x in names if x not in links]
            for x in names:
                if x in links:
                    if links[x] is not None:
                        args.append(f'({x} := {links[x]})')
                else:
                    args.append(f'({x} := {ren.get(x, x)})')
            if not keep:
                continue
            new = [ren.get(x, x) for x in keep]
            fresh = [x for x in new if x not in self.seen]
            if not fresh:
                continue
            self.seen.update(fresh)
            tyr = rename_text(ty, ren) if ty else None
            if tyr is not None and links:
                tyr = rename_text(tyr, {k: f'({v})' for k, v in links.items() if v is not None})
            self.binders.append(f'{br}{" ".join(fresh)} : {tyr}{")" if br == "(" else "}"}' if tyr else
                                f'{br}{" ".join(fresh)}{")" if br == "(" else "}"}')
        res = res or f'r_{tag}'
        self.steps.append(f'  have {res} := {prefix}{name} ' + ' '.join(args))
        return res

    def theorem(self, name, header, extra_steps=()):
        steps = [re.sub(r'^  have ', '  node_have ', s) for s in list(self.steps) + list(extra_steps)]
        return (f'set_option maxHeartbeats 64000000 in\ntheorem {name} {header}\n    ' + '\n    '.join(self.binders) +
                ' : True := by\n' + '\n'.join(steps) + '\n  trivial\n')
        return (f'theorem {name} {header}\n    ' + '\n    '.join(self.binders) + ' : True := by\n' +
                '\n'.join(list(self.steps) + list(extra_steps)) + '\n  trivial\n')


def root_private_match(I):
    """Root private multiplications (`mul_private_down`): an input encoding's `both` times the
    digits of `[top_j; bot_j]` (top hash-sampled), sliced and moved down. Returns a list of dicts."""
    Fl = Flow(I)
    o = Fl.out
    res = []
    for both in [a for a in Fl.atoms if a['head'].endswith('concatColumns')]:
        bw = o(both)
        muls = [(w, app(t)[1]) for w, t in I.lets.items() if app(t)[0] == 'matrixMul' and wire_of(app(t)[1][0]) == bw]
        if len(muls) != 2:
            continue
        r = dict(both=both, j=[])
        ok = True
        for w, xs in muls:
            Dw = wire_of(xs[1])
            D = I.prod.get(Dw)
            if not D or not D['head'].endswith('compactCenteredRebaseRuns'):
                ok = False
                break
            D1 = I.prod.get(wire_of(D['args'][-2]))
            if not D1 or not D1['head'].endswith('gadgetDecomposeRuns'):
                ok = False
                break
            Xb = I.prod.get(wire_of(D1['args'][-2]))
            if not Xb or not Xb['head'].endswith('concatRows'):
                ok = False
                break
            top = I.prod.get(wire_of(Xb['args'][0]))
            if not top or not top['head'].endswith('hashSample'):
                ok = False
                break
            try:
                X, M0, M1 = Fl.slice(w, 0, 1), Fl.slice(w, 1, 2), Fl.slice(w, 2, 3)
                bms = lambda s: next(a for a in Fl.atoms if a['head'].endswith('blockModSwitchRuns') and wire_of(a['args'][-2]) == o(s))
                r['j'].append(dict(mul=w, D=D, D1=D1, Xb=Xb, top=top, bot=wire_of(Xb['args'][1]), X=X, M0=M0, M1=M1,
                                   cX=bms(X), cM0=bms(M0), cM1=bms(M1)))
            except (KeyError, StopIteration):
                ok = False
                break
        if not ok:
            continue
        comp = lambda t: int(re.search(r'\.u64Le \((\d+)\)\]\)?$', t['top']['args'][2].strip()).group(1))
        r['j'].sort(key=comp)
        l = I.prod[wire_of(both['args'][0])]
        rr = I.prod[wire_of(both['args'][1])]
        l01 = I.prod[wire_of(l['args'][0])]
        r01 = I.prod[wire_of(rr['args'][0])]
        r['enc'] = [l01, l, r01, rr, both]
        r['c'] = [wire_of(l01['args'][0]), wire_of(r01['args'][0])]
        r['A'] = [[wire_of(l01['args'][1]), wire_of(r01['args'][1])], [wire_of(l['args'][1]), wire_of(rr['args'][1])]]
        res.append(r)
    return res


def emit_root_private(I, r, C, name):
    W_ = I.W
    F = I.F
    o = lambda a: wire_of(a['args'][-1])
    J = r['j']
    _, q, nn, _, m = mat_type(I.wires[r['c'][0]]['type'])
    _, q2, _, _, m2 = mat_type(I.wires[o(J[0]['cX'])]['type'])
    Q, Q2 = qterm(q), qterm(q2)
    p = f'({Q} / {Q2})'
    hq, hp = C.div(q, q2)
    top = J[0]['top']
    key = wire_of(top['args'][3])
    prefix = top['args'][1].strip('()')
    pre = pre_of(top)
    A = f"![![{W_(r['A'][0][0])}, {W_(r['A'][0][1])}], ![{W_(r['A'][1][0])}, {W_(r['A'][1][1])}]]"
    c = f"![{W_(r['c'][0])}, {W_(r['c'][1])}]"
    w2 = lambda k: f"![{W_(o(J[0][k]))}, {W_(o(J[1][k]))}]"
    hd = lambda k: two([F(J[0][k]), F(J[1][k])])
    return f'''theorem {name} {{backend : BackendContext}} {{hashModel : HashModel}} {{params : Params}} {{i_0 : Nat}}
    [Fact (i_0 < S)] {{ctx : {I.inst}.Ctx backend hashModel params i_0}} {{outputs}}
    (F : {I.inst}.Facts ctx outputs) (hb : backend = Backend.backend)
    {{sm sp μ : ErrorPoly {nn}}} (hsm : polyNorm sm ≤ 1)
    {{g' : ExactMatrix {Q2} {nn} 1 {m}}} {{g : ExactMatrix {Q} {nn} 1 {m}}}
    (hg' : gadgetMatrixRuns backend {GBASE} {m} g') (hg : centeredRebaseRuns g' g) {{B : Nat}}
    (henc : MxxExponentLut.Rel sm {A} (fun j ↦ reducePoly {Q} {nn} ((({p} : Nat) : Int) * μ * unitPair sp j) • g) {c} B)
    {{T : Fin 2 → ExactMatrix {Q2} {nn} 1 {m2}}} {{eB : Fin 2 → ErrorMatrix {nn} 1 {m2}}} {{BB : Nat}}
    (hrhs : ∀ j, reducePoly {Q2} {nn} sp • topPairG hashModel ({W_(key)}) {Q2} {m2} {prefix} {pre} j +
      ![{W_(J[0]['bot'])}, {W_(J[1]['bot'])}] j = T j + reduceMatrix {Q2} {nn} 1 {m2} (eB j))
    (heB : ∀ j, CoeffBound (eB j) BB) (hμ : polyNorm μ ≤ 1) :
    MxxExponentLut.Rel sm ![{w2('cM0')}, {w2('cM1')}] (fun j ↦ reducePoly {Q2} {nn} μ • T j) {w2('cX')}
      ((({m} + {m}) * {nn} * B * {GHALF} + ({nn} + 2) * ({p} / 2)) / {p} + {nn} * 1 * BB) := by
  subst hb
{layout_obtain(C, q2, ['L', 'hL', 'hLx', '-'])}  have ht0 := hashSample_eq {F(J[0]['top'])}
  have ht1 := hashSample_eq {F(J[1]['top'])}
  have htop : (![{W_(o(J[0]['top']))}, {W_(o(J[1]['top']))}] : Fin 2 → _) = topPairG hashModel ({W_(key)}) {Q2} {m2} {prefix} {pre} := by
    funext j; fin_cases j
    · exact ht0
    · exact ht1
  rw [← htop] at hrhs
  exact mul_private_down (by decide) {hq} {C.gt(q)} {C.gt(q2)} (by have := {hp}; omega) L hL hLx hg' hg hsm henc
    {' '.join(F(x) for x in r['enc'])}
    (top := ![{W_(o(J[0]['top']))}, {W_(o(J[1]['top']))}]) (bot := ![{W_(J[0]['bot'])}, {W_(J[1]['bot'])}])
    (Xb := {w2('Xb')}) {hd('Xb')} (D' := {w2('D1')}) {hd('D1')} (D := {w2('D')}) {hd('D')}
    hrhs heB hμ (X := {w2('X')}) (M := ![{w2('M0')}, {w2('M1')}]) {hd('X')} {hd('M0')} {hd('M1')}
    (c' := {w2('cX')}) (A' := ![{w2('cM0')}, {w2('cM1')}]) {hd('cX')}
    (Fin.forall_fin_two.mpr ⟨{hd('cM0')}, {hd('cM1')}⟩)
'''
