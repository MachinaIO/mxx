#!/usr/bin/env python3
"""Generate `Gen_Inst_KeyEq.lean` (WP D): the keys of the online final combination's mask carrier
(`mask_keys`) and fresh row (`fresh_keys`) are the preprocessing's combination keys (`hpre` of
`GenFinalInst.final_inst`).

The keys of an online encoding carry no semantic hypothesis: each online piece's keys are a closed
form of its input keys (`GenKeyChain`: `branch_keys_core`, `lut_root_keys`, the member `.keys`,
`privKeyG_of_stacks`; `GenInstKeys`: the PRF cells), and the preprocessing evaluates the same closed
forms (F's `Gen_Inst_Stage_preprocess_Keys`: `member_*`, `keyloop_*`, `keys_*`, `memsum_*`, `colsum_*`,
`slicesum_*`, `bms_*`, `spec_*`; here: `privKeyG_of_pre` of the root private products). The generator
follows the online data flow backwards from the summand's named key witness (`mask_summand`'s
`K = …`, `fresh_summand`'s `Kf = …`) and pairs every online piece with the preprocessing piece
producing the corresponding key (by role and argument position), down to the stage inputs (stage
links, equal hash samples). Every equality is a `have` of the emitted theorem; nothing is recognized
by wire ids.

Usage: gen_key_eq.py <generated dir> <param dir> <roles.json>
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
import xmodel  # noqa: E402
import xsem  # noqa: E402
from gen_inst_keys import Gen, St, base, qual_on, ON, PRE, ENC, SIGMA  # noqa: E402


def stage_ns(M):
    return M.replace('Stage_', '').title().replace('_', '')


OK_NS = f'RefreshTopProof.GenInst.{stage_ns(ON)}Keys'
PK_NS = f'RefreshTopProof.GenInst.{stage_ns(PRE)}Keys'
OE_NS = f'RefreshTopProof.GenInst.{stage_ns(ON)}'


def proj(c, k):
    """Projection of component `c` of a `k`-fold conjunction / tuple (without the trailing unit)."""
    return '.1' if c == 0 and k > 1 else ('.2' * c + ('.1' if c < k - 1 else ''))


def tproj(c):
    """Projection of component `c` of a tuple ending in `()`."""
    return '.2' * c + '.1'


def stmt(text, name):
    m = re.search(r'^theorem %s .*?:=(?: by)?$' % re.escape(name), text, re.S | re.M)
    return m.group(0) if m else None


def loop_any(S, w):
    """The loop producing `w` (par/gather outputs or a seq loop's final tuple) and the index."""
    a, c = S.loop_of(w)
    if a is not None:
        return a, c
    for a in S.loops:
        fin = xsem.tuple_wires(a['loop'].get('final') or '')
        if base(w) in fin:
            return a, fin.index(base(w))
    return None, None


class KE(Gen):
    def __init__(self, gdir, pdir, roles):
        super().__init__(gdir, pdir, roles)
        self.gdir, self.pdir = gdir, pdir
        self.ok = open(f'{pdir}/Gen_Inst_{ON}_Keys.lean').read()
        self.kc = open(f'{pdir}/Gen_Inst_{ON}_KeyChain.lean').read()
        self.cnt = 0
        self.done = {}

    def fresh(self, p):
        self.cnt += 1
        return f'{p}{self.cnt}'

    # ------------------------------------------------------------------ inputs
    def in_eq(self, w):
        """Rewrite names turning the online input `w` into its preprocessing wire, and that wire."""
        try:
            ch, p, _ = self.to_pre(w)
        except AssertionError:
            return None
        return [c for c in ch if not isinstance(c, tuple)], p

    def sample_fact(self, X, w):
        a = X.prod.get(w)
        if not a or a[1] is not None or not a[0]['head'].endswith('hashSample'):
            return None
        return a[0]

    def sample_eq(self, w, pw):
        """`O.w = P.pw` for an online input `w` linked to a hash sample of the encrypt or online stage
        equal (same tag, linked key) to the preprocessing sample `pw`."""
        O, P, E = self.O, self.P, self.E
        key = ('s', w, pw)
        if key in self.done:
            return self.done[key]
        try:
            ch = self.root_chain(w, pw)
            ch = [c for c in ch if not isinstance(c, tuple)]
            r = ch[0]
            for c in ch[1:]:
                r = f'({r}.trans {c})'
            self.done[key] = r
            return r
        except AssertionError:
            pass
        chain = []
        X, xw = O, w
        if not self.sample_fact(O, w):
            d = self.link(O, w, P)
            if d and d[1] == pw:
                self.done[key] = d[0]
                return d[0]
            if d:
                u = self.hash_eq(d[1], pw)
                r = f'({d[0]}.trans {u})' if u else d[0]
                self.done[key] = r
                return r
            d = self.link(O, w, E)
            assert d, f'online input {w}'
            chain.append(d[0])
            X, xw = E, d[1]
        ax, ap = self.sample_fact(X, xw), self.sample_fact(P, pw)
        assert ax and ap and ax['args'][1:3] == ap['args'][1:3], f'samples {w} {pw}'
        kx, kp = base(ax['args'][3].strip('()').split(' ')[0]), base(ap['args'][3].strip('()').split(' ')[0])
        kx, kp = xsem.wire_of(ax['args'][3]), xsem.wire_of(ap['args'][3])
        ke = self.key_eq(X, kx, kp)
        name = self.fresh('hs')
        rhs = f'hashSample_unique {X.F(ax)} (by rw [{ke}]; exact {P.F(ap)})'
        self.haves.append(f'  have {name} : {X.W(xw)} = {P.W(pw)} := {rhs}')
        r = f'({chain[0]}.trans {name})' if chain else name
        self.done[key] = r
        return r

    def key_eq(self, X, kx, kp):
        """Name of `X.kx = P.kp` for two hash keys (stage inputs)."""
        P = self.P
        nm = f'k_{X.M[6]}_{kx}_{kp}'
        if nm in self.done:
            return self.done[nm]
        d = self.link(X, kx, P) if X is self.O or X is self.E else None
        if d and d[1] == kp:
            self.done[nm] = d[0]
            return d[0]
        hyp = f'lK{X.M[6]}_{kx}'
        self.hyps[hyp] = f'{X.W(kx)} = {P.W(kp)}'
        self.done[nm] = hyp
        return hyp

    # ------------------------------------------------------------------ PRF chain into a member
    def prf_chain(self, on_x, pre_x):
        """The PRF bucket chain feeding the online member input `on_x` and the preprocessing one
        feeding `pre_x` (member_keys' cells and row sums); returns the last row-sum fact name."""
        O, P = self.O, self.P

        def walk(S, x, cell1, cell0, as_pos):
            pairs = []
            while True:
                r = S.loop_of(x)[0]
                assert S.role(r) == 'prf.rowsum', S.role(r)
                c = S.loop_of(r['loop']['args'][0])[0]
                pairs.append((c, r))
                if S.role(c) == cell0:
                    return pairs[::-1]
                assert S.role(c) == cell1, S.role(c)
                # a Program in its own stages (13 arguments) passes the state keys from position 4
                x = c['loop']['args'][4 if as_pos == 7 and len(c['loop']['args']) == 13 else as_pos]
        on = walk(O, on_x, 'prf.cell1', 'prf.cell0', 7)
        pre = walk(P, pre_x, 'prf.key-cell1', 'prf.key-cell0', 2)
        assert len(on) == len(pre), (len(on), len(pre))
        key = ('prf', on[-1][1]['field'])
        if key in self.done:
            return self.done[key]
        sec, secn = self.pre_binders(pre[0][0]['field'])
        self.sec, self.secn = sec, secn
        labs = {}
        for (c, _), (pc, _) in zip(on, pre):
            k = tuple(x[0] for x in c['loop']['gathered'])
            # a Program in its own stages hash-samples the label keys in the cell (`labelKeysG`)
            if k not in labs and len(c['loop']['args']) != 13:
                labs[k] = self.fresh('hlab')
                self.body.append(self.hlab(c, pc, labs[k]))
        prev = None
        for t, ((c, r), (pc, pr)) in enumerate(zip(on, pre)):
            lab = labs.get(tuple(x[0] for x in c['loop']['gathered']))
            kc, er = self.fresh('kc'), self.fresh('er')
            self.body.append(self.cell_eq(kc, c, pc, 0 if t == 0 else 1, lab, prev))
            self.body.append(self.rowsum_eq(er, r, pr, kc))
            prev = er
        self.done[key] = prev
        return prev

    # ------------------------------------------------------------------ members
    def member_rw(self, a, pa, offs=None):
        """Rewrites of an online member's key arguments into the preprocessing member's."""
        O = self.O
        args = [x for x in a['loop']['args']]
        rw = []
        for w in args[0:4] + [args[5]]:
            r = self.in_eq(base(w))
            assert r, f'member input {w}'
            rw += r[0]
        pargs = pa['loop']['args']
        for k, pk in ((8, 1), (9, 2), (11, 4), (12, 5)):
            rw.append(self.out_rw(args[k], pargs[pk]))
        if offs:
            rw += offs
        return rw

    def out_rw(self, on_arg, pre_arg):
        """Rewrite term for an online member argument `w_k_c i` produced by a member/row-sum loop."""
        O, P = self.O, self.P
        a, c = O.loop_of(on_arg)
        pa, pc = P.loop_of(pre_arg)
        f = self.loop_eq(a, pa)
        return f'({f} i){proj(c, 4)}'

    def loop_eq(self, a, pa):
        """Name of `∀ i, O.out_c i = P.out_c i` (c < 4) for a paired online/preprocessing loop."""
        key = ('loop', a['field'])
        if key in self.done:
            return self.done[key]
        role = self.O.role(a)
        if role == 'prf.rowsum':
            r = self.prf_chain(a['loop']['outs'][0], pa['loop']['outs'][0])
            self.done[key] = r
            return r
        name = self.member_eq(a, pa)
        self.done[key] = name
        return name

    def member_eq(self, a, pa):
        O, P = self.O, self.P
        role = O.role(a)
        lemma = {'lut.member.readout': 'LutMemberReadout', 'lut.member.sampler': 'LutMemberSampler',
                 'lut.member.sampler-offset': 'LutMemberSamplerOffset'}[role]
        t = self.seg_of(a['field'])
        lay = re.search(r'^  obtain ⟨L, hL, hLx, -⟩ := (\S+(?: i_0)?)$', t, re.M).group(1)
        hq = re.search(r'\(hq\' := (\(.*?\))\) \(layout', t).group(1)
        Lm = re.search(r'\(L := (\d+)\)', t).group(1)
        child = re.search(r'\(child := (fun k s nxt ↦ .*?)\)\n', t).group(1)
        fs = re.search(r'(\(f00 := .*?\(f11 := [^)]*\))', t).group(1)
        ad = re.search(r'(\(fun _ _ _ hc ↦ \S+\.sem hc\))', t).group(1)
        offs = None
        if role == 'lut.member.sampler-offset':
            offs = self.offsets(a, pa)
        rw = self.member_rw(a, pa, offs)
        oo = [base(x) for x in a['loop']['outs']]
        po = [base(x) for x in pa['loop']['outs']]
        n = a['loop']['count']
        name = self.fresh('mk')
        concl = ' ∧ '.join(f'{O.W(oo[c])} i = {P.W(po[c])} i' for c in range(4))
        s = [f'  have {name} : ∀ i : Fin {n}, {concl} := by', '    intro i',
             f'    obtain ⟨L, hL, hLx, hd⟩ := {qual_on(lay)}',
             f'    have hO := {lemma}.keys (h := {ON}.{a["loop"]["child"]}.sem ({O.F(a)} i))',
             f'      (hmw := rfl) (L := {Lm}) (hℓ := rfl) (hL := rfl) (hn := by decide)',
             f'      (hq\' := {qual_on(hq)}) L hL hLx (by rw [hd])',
             f'      (LutRoundStep.keystep (child := {qual_on(child)})',
             f'        {qual_on(fs)}',
             f'        {qual_on(ad)} rfl (by decide)',
             f'        {qual_on(hq)} L hL hLx (by rw [hd]))',
             f'    rw [{", ".join(rw)}] at hO',
             f'    have e := hO.trans ({PK_NS}.member_{pa["field"]} FP i).symm',
             '    exact ⟨congrArg (·.1) e, congrArg (·.2.1) e, congrArg (·.2.2.1) e, congrArg (·.2.2.2.1) e⟩']
        self.body.append('\n'.join(s))
        return name

    def seg_of(self, field):
        m = re.search(r'^theorem seg_%s\b.*?(?=^theorem |\Z)' % re.escape(field), self.seg, re.S | re.M)
        return m.group(0)

    # ------------------------------------------------------------------ offsets, rounds, root products
    def offsets(self, a, pa):
        """Rewrites `O.o_c = P.o_c` (c < 4) of a sampler-with-offset member's offset keys: the online
        offsets are the keys of a root LUT's branches (`keys_<field>`), the preprocessing ones a key loop."""
        O, P = self.O, self.P
        on = [base(x) for x in a['loop']['args'][13:17]]
        pn = [base(x) for x in pa['loop']['args'][9:13]]
        h, xs = xsem.app(O.j['wires'][on[0]]['term'])
        assert h == 'matrixAdd', on[0]
        bl, _ = loop_any(O, xsem.wire_of(xs[0]))
        pl, _ = loop_any(P, pn[0])
        name = self.branches_eq(bl, pl)
        res = []
        for c in range(4):
            e = self.fresh('ob')
            self.body.append(f'  have {e} : {O.W(on[c])} = {P.W(pn[c])} := congrArg (·{tproj(c)}) {name}')
            res.append(e)
        return res

    def keys_stmt(self, field):
        return stmt(self.kc, f'keys_{field}')

    def root_x_rw(self, bl, kst, pst):
        """Rewrites of a root LUT's input keys (x1, x2, x4, x5), fams and key (online) into the
        preprocessing key loop's (`keyA a00 a10 a01 a11`, `fam4 …`, key)."""
        O, P = self.O, self.P
        args = [base(x) for x in bl['loop']['args'][10:]]
        key, x, f = args[1], args[3:9], args[9:13]
        ka = re.search(r'keyA \(\S+\.(w_\d+_\d+) ctx\) \(\S+\.(w_\d+_\d+) ctx\) \(\S+\.(w_\d+_\d+) ctx\) \(\S+\.(w_\d+_\d+) ctx\)', pst)
        a00, a10, a01, a11 = ka.groups()
        rw = []
        r = self.in_eq(key)
        assert r, key
        rw += r[0]
        for w in f:
            r = self.in_eq(w)
            assert r, w
            rw += r[0]
        rw += [self.x_eq(x[1], a00), self.x_eq(x[2], a10), self.x_eq(x[4], a01), self.x_eq(x[5], a11)]
        return rw

    def x_eq(self, w, pw):
        """`O.w = P.pw` for a root LUT input key: an output of a root LUT round (moved down) or of a
        root private product, or a stage input."""
        key = ('x', w, pw)
        if key in self.done:
            return self.done[key]
        O, P = self.O, self.P
        r = None
        pa = P.prod.get(pw)
        # a root LUT round's moved-down output, or a root private product's output?
        for f in re.findall(r'^theorem keys_(h\d+_\d+) ', self.kc, re.M):
            st = self.keys_stmt(f)
            mr = re.search(r':\n    keyTupleG (!\[!\[.*?\]\]) = keyBmsG', st, re.S)
            mp = re.search(r':\n    (!\[!\[.*?\]\]) = privKeyG', st, re.S)
            if mr and w in re.findall(r'(w_\d+_\d+) ctx', mr.group(1)):
                r = self.round_entry(f, re.findall(r'(w_\d+_\d+) ctx', mr.group(1)).index(w), pw)
                break
            if mp and w in re.findall(r'(w_\d+_\d+) ctx', mp.group(1)):
                r = self.private_entry(f, re.findall(r'(w_\d+_\d+) ctx', mp.group(1)).index(w), pw)
                break
        if r is None:
            r = self.sample_eq(w, pw)
        self.done[key] = r
        return r

    def round_entry(self, field, idx, pw):
        """`O.Kp_idx = P.pw` with `P.pw = bms (P.loop_idx)` (F's `bms_*`) for a root LUT round."""
        O, P = self.O, self.P
        key = ('round', field)
        st = self.keys_stmt(field)
        if key not in self.done:
            bl = [a for a in O.loops if a['field'] == field][0]
            b = stmt(self.P_keys, f'bms_h{pw[2:]}')
            pin = re.search(r'\(1\) \(\S+\.(w_\d+_\d+) ctx\) :=$', b).group(1)
            pl, _ = loop_any(P, pin)
            pst = stmt(self.P_keys, f'keyloop_{pl["field"]}')
            rw = self.root_x_rw(bl, st, pst)
            basis2 = re.search(r'= keyBmsG (\[[\d, ]*\])', st).group(1)
            name = self.fresh('rr')
            self.body.append(f'  have {name} := {OE_NS}.keys_{field} FO rfl\n'
                             f'  rw [{", ".join(rw)}] at {name}\n'
                             f'  replace {name} := {name}.trans (congrArg (keyBmsG {basis2}) ({PK_NS}.keyloop_{pl["field"]} FP).symm)')
            self.done[key] = (name, pl)
        name, pl = self.done[key]
        ws = re.findall(r'(w_\d+_\d+) ctx', re.search(r'keyTupleG (!\[!\[.*?\]\]) =', st).group(1))
        e = self.fresh('re')
        self.haves_after.append(None)
        self.body.append(f'  have {e} : {O.W(ws[idx])} = {P.W(pw)} := by\n'
                         f'    rw [{PK_NS}.bms_h{pw[2:]} FP]\n'
                         f'    exact congrArg (·{tproj(idx)}) {name}')
        return e

    def branches_eq(self, bl, pl):
        """`keyTupleG (looped + unrolled keys) = (P.keyloop outputs)` of a root LUT's branches."""
        key = ('branches', bl['field'])
        if key in self.done:
            return self.done[key]
        st = self.keys_stmt(bl['field'])
        pst = stmt(self.P_keys, f'keyloop_{pl["field"]}')
        rw = self.root_x_rw(bl, st, pst)
        name = self.fresh('rb')
        self.body.append(f'  have {name} := {OE_NS}.keys_{bl["field"]} FO rfl\n'
                         f'  rw [{", ".join(rw)}] at {name}\n'
                         f'  replace {name} := {name}.trans ({PK_NS}.keyloop_{pl["field"]} FP).symm')
        self.done[key] = name
        return name

    # root private products (both stages)
    def private_entry(self, field, idx, pw):
        key = ('priv', field, pw)
        name = self.private_eq(field, pw)
        e = self.fresh('pe')
        kk, jj = [(0, 0), (0, 1), (1, 0), (1, 1)][idx]
        ws = re.findall(r'(w_\d+_\d+) ctx', re.search(r':\n    (!\[!\[.*?\]\]) = privKeyG', self.keys_stmt(field), re.S).group(1))
        self.body.append(f'  have {e} : {self.O.W(ws[idx])} = {self.P.W(pw)} := congrFun (congrFun {name} {kk}) {jj}')
        return e

    def private_eq(self, field, pw):
        """`![![O A'…]] = ![![P A'…]]` of a root private product pair (online `keys_<field>`, the
        preprocessing `privKeyG_of_pre` emitted here), the preprocessing product found from `pw`."""
        O, P = self.O, self.P
        key = ('privm', field)
        if key in self.done:
            return self.done[key]
        st = self.keys_stmt(field)
        pr = self.pre_private(pw)
        # online form: A' = privKeyG backend base digits basis A (topPairG …) ![b0, b1]
        m = re.search(r':\n    (!\[!\[.*?\]\]) = privKeyG Backend\.backend (\(\S+ : Int\)) (\(\S+ : Int\)) (\[[\d, ]*\]) (!\[!\[.*?\]\])\n'
                      r'\s*\((topPairG .*?)\) !\[(.*?)\] :=', st, re.S)
        onA = re.findall(r'(w_\d+_\d+) ctx', m.group(5))  # A00 A01 A10 A11
        onB = re.findall(r'(w_\d+_\d+) ctx', m.group(7))
        rw = []
        # A: online input keys -> preprocessing A
        for w, p in zip(onA, pr['A']):
            rw.append(self.x_eq(w, p))
        for w, p in zip(onB, pr['bot']):
            r = self.in_eq(w)
            assert r and r[1] == p, f'bottom {w} {p}'
            rw += r[0]
        # tops: key and instance tables
        tk = re.search(r'topPairG hashModel \(\(\S+\.(w_\d+_\d+) ctx\)\)', m.group(6)).group(1)
        rw.append(self.key_eq(O, tk, pr['key']))
        sels_on = re.findall(r'\.decimal \(\((sel_\d+) \(\(Int\.ofNat i_0\)\)\)\)', m.group(6))
        for so, sp_ in zip(sels_on, pr['sels']):
            rw.append(self.sel_eq(so, sp_))
        name = self.fresh('rp')
        on_t = re.findall(r'(w_\d+_\d+) ctx', m.group(1))
        self.body.append(pr['text'])
        self.body.append(f'  have {name} : ![![{O.W(on_t[0])}, {O.W(on_t[1])}], ![{O.W(on_t[2])}, {O.W(on_t[3])}]] = '
                         f'![![{P.W(pr["t"][0])}, {P.W(pr["t"][1])}], ![{P.W(pr["t"][2])}, {P.W(pr["t"][3])}]] := by\n'
                         f'    have hO := {OE_NS}.keys_{field} FO rfl\n'
                         f'    rw [{", ".join(rw)}] at hO\n'
                         f'    exact hO.trans {pr["name"]}.symm')
        self.done[key] = name
        return name

    def sel_eq(self, so, sp_):
        nm = f'hsel_{so}_{sp_}'
        if nm not in self.done:
            self.body.append(f'  have {nm} : {ON}.{so} (Int.ofNat i_0) = {PRE}.{sp_} (Int.ofNat i_0) := by\n'
                             f'    first | rfl | (have h := hS.out; interval_cases i_0 <;> rfl)')
            self.done[nm] = nm
        return nm

    def pre_private(self, pw):
        """The preprocessing root private product whose moved-down output contains `pw`:
        `bms (slice k (matrixAdd (matrixMul S0 rT_j) (matrixMul S1 rB_j)))`."""
        P = self.P
        I = xsem.Inst(P.j, P.fields, P.inst)
        lets = I.lets

        def prod(w):
            return I.prod.get(w)
        b = prod(pw)
        assert b['head'].endswith('blockModSwitchRuns'), pw
        sl = prod(xsem.wire_of(b['args'][-2]))
        Lw = xsem.wire_of(sl['args'][0])
        # all (k, j) outputs sharing S0/S1
        h, xs = xsem.app(lets[Lw])
        m0, m1 = xsem.wire_of(xs[0]), xsem.wire_of(xs[1])
        S0 = xsem.wire_of(xsem.app(lets[m0])[1][0])
        S1 = xsem.wire_of(xsem.app(lets[m1])[1][0])
        Ls = []
        for w, t in lets.items():
            hh, ys = xsem.app(t)
            if hh != 'matrixAdd':
                continue
            try:
                a0, a1 = xsem.wire_of(ys[0]), xsem.wire_of(ys[1])
                if xsem.wire_of(xsem.app(lets[a0])[1][0]) == S0 and xsem.wire_of(xsem.app(lets[a1])[1][0]) == S1:
                    Ls.append(w)
            except (KeyError, IndexError, TypeError):
                pass
        J = []
        for Lj in Ls:
            hh, ys = xsem.app(lets[Lj])
            mT, mB = xsem.wire_of(ys[0]), xsem.wire_of(ys[1])
            rT = xsem.wire_of(xsem.app(lets[mT])[1][1])
            rB = xsem.wire_of(xsem.app(lets[mB])[1][1])
            hrT, hrB = prod(rT), prod(rB)
            dT, dB = xsem.wire_of(hrT['args'][-2]), xsem.wire_of(hrB['args'][-2])
            hdT, hdB = prod(dT), prod(dB)
            top, bot = xsem.wire_of(hdT['args'][-2]), xsem.wire_of(hdB['args'][-2])
            sls = [a for a in I.j['atoms'] if a['head'].endswith('sliceMatrix') and xsem.wire_of(a['args'][0]) == Lj]
            sl0 = [a for a in sls if a['args'][1].strip('()') == '0'][0]
            sl1 = [a for a in sls if a['args'][1].strip('()') == '1'][0]
            bm = lambda s: next(a for a in I.j['atoms'] if a['head'].endswith('blockModSwitchRuns') and xsem.wire_of(a['args'][-2]) == xsem.wire_of(s['args'][-1]))
            J.append(dict(L=Lj, top=top, bot=bot, hdT=hdT, hrT=hrT, hdB=hdB, hrB=hrB, sl0=sl0, sl1=sl1, b0=bm(sl0), b1=bm(sl1)))
        tops = [prod(j['top']) for j in J]
        comp = lambda a: int(re.search(r'\.u64Le \((\d+)\)\]\)?$', a['args'][2].strip()).group(1))
        order = sorted(range(len(J)), key=lambda k: comp(tops[k]))
        J = [J[k] for k in order]
        tops = [tops[k] for k in order]
        hS0, hS1 = prod(S0), prod(S1)
        A00, A10 = xsem.wire_of(hS0['args'][0]), xsem.wire_of(hS0['args'][1])
        A01, A11 = xsem.wire_of(hS1['args'][0]), xsem.wire_of(hS1['args'][1])
        o = lambda a: xsem.wire_of(a['args'][-1])
        t = [o(J[0]['b0']), o(J[1]['b0']), o(J[0]['b1']), o(J[1]['b1'])]
        F = lambda a: P.F(a).replace('FP.', 'FP.')
        two = lambda xs: f'(Fin.forall_fin_two.mpr ⟨{xs[0]}, {xs[1]}⟩)'
        top0 = tops[0]
        key = xsem.wire_of(top0['args'][3])
        prefix = top0['args'][1].strip('()')
        pre = xsem.pre_of(top0)
        sels = re.findall(r'\.decimal \(\((sel_\d+) \(\(Int\.ofNat i_0\)\)\)\)', pre)
        _, q2, nn, _, m2 = xsem.mat_type(P.j['wires'][o(J[0]['b0'])]['type'])
        tp = f'topPairG hashModel {P.W(key)} {xsem.qterm(q2)} {m2} {prefix} {pre}'
        tp = re.sub(r'(?<![\w.])(sel_\d+|mod_\d+|modulus_\d+)\b', PRE + r'.\1', tp)
        gd = J[0]['hdT']['args']
        base_, digits = gd[1].strip('()'), gd[2].strip('()')
        basis = J[0]['b0']['args'][0]
        name = self.fresh('pp')
        Aw = f'![![{P.W(A00)}, {P.W(A01)}], ![{P.W(A10)}, {P.W(A11)}]]'
        text = (f'  have {name} : ![![{P.W(t[0])}, {P.W(t[1])}], ![{P.W(t[2])}, {P.W(t[3])}]] = '
                f'privKeyG Backend.backend ({base_} : Int) ({digits} : Int) {basis} {Aw}\n'
                f'      ({tp}) ![{P.W(J[0]["bot"])}, {P.W(J[1]["bot"])}] := by\n'
                f'    have htop : (![{P.W(J[0]["top"])}, {P.W(J[1]["top"])}] : Fin 2 → _) = {tp} := by\n'
                f'      funext j; fin_cases j\n'
                f'      · exact topPairG_of_hash (pre := {re.sub(r"(?<![\w.])(sel_\d+)", PRE + r".\1", pre)}) 0 rfl {F(tops[0])}\n'
                f'      · exact topPairG_of_hash (pre := {re.sub(r"(?<![\w.])(sel_\d+)", PRE + r".\1", pre)}) 1 rfl {F(tops[1])}\n'
                f'    rw [← htop]\n'
                f'    exact privKeyG_of_pre (A := {Aw}) {F(hS0)} {F(hS1)}\n'
                f'      (top := ![{P.W(J[0]["top"])}, {P.W(J[1]["top"])}]) (bot := ![{P.W(J[0]["bot"])}, {P.W(J[1]["bot"])}])\n'
                f'      (dT := ![{P.W(o(J[0]["hdT"]))}, {P.W(o(J[1]["hdT"]))}]) (rT := ![{P.W(o(J[0]["hrT"]))}, {P.W(o(J[1]["hrT"]))}])\n'
                f'      (dB := ![{P.W(o(J[0]["hdB"]))}, {P.W(o(J[1]["hdB"]))}]) (rB := ![{P.W(o(J[0]["hrB"]))}, {P.W(o(J[1]["hrB"]))}])\n'
                f'      {two([F(J[0]["hdT"]), F(J[1]["hdT"])])} {two([F(J[0]["hrT"]), F(J[1]["hrT"])])}\n'
                f'      {two([F(J[0]["hdB"]), F(J[1]["hdB"])])} {two([F(J[0]["hrB"]), F(J[1]["hrB"])])}\n'
                f'      (M := ![![{P.W(o(J[0]["sl0"]))}, {P.W(o(J[1]["sl0"]))}], ![{P.W(o(J[0]["sl1"]))}, {P.W(o(J[1]["sl1"]))}]])\n'
                f'      {two([F(J[0]["sl0"]), F(J[1]["sl0"])])} {two([F(J[0]["sl1"]), F(J[1]["sl1"])])}\n'
                f'      (Fin.forall_fin_two.mpr ⟨{two([F(J[0]["b0"]), F(J[1]["b0"])])}, {two([F(J[0]["b1"]), F(J[1]["b1"])])}⟩)')
        return dict(name=name, text=text, t=t, A=[A00, A01, A10, A11], bot=[J[0]['bot'], J[1]['bot']], key=key, sels=sels)

    # ------------------------------------------------------------------ the mask carrier
    def mask(self):
        """`mask_keys` (the stages of `σ_0 G`): the mask row's keys (`mask_summand`'s named witness, the
        online member sum's components 1 and 3 moved down) are the preprocessing's moved-down member
        sum components 1 and 3, the keys the parts' preprocessing reads."""
        O, P = self.O, self.P
        ms = open(f'{self.pdir}/Gen_Summands_Sigma_MaskSummand.lean').read()
        kt = re.search(r'∧\n      K = (fun k ↦ blockModSwitchG (\[[\d, ]*\]) 1 \(Enc\.A \((.*?)\) k 1\)) :=', ms, re.S)
        X = re.findall(r'(w_\d+_\d+) ctx', kt.group(3))
        # the member sums
        lo, co = O.loop_of(X[1])
        assert lo is None
        lo = [a for a in O.loops if X[1] in a['wires'] and O.role(a) == 'sum.members.step'][0]
        # the preprocessing's moved-down member sum components 1 and 3 (`bms_*`)
        tgt = {}
        for name in re.findall(r'^theorem (bms_h\d+_\d+) ', self.P_keys, re.M):
            b = stmt(self.P_keys, name)
            src = re.search(r'\(1\) \(\S+\.(w_\d+_\d+) ctx\) :=$', b).group(1)
            for a in P.loops:
                fin = xsem.tuple_wires(a['loop'].get('final') or '')
                if P.role(a) == 'sum.members.step' and src in fin and fin.index(src) in (1, 3):
                    tgt[fin.index(src)] = ('w_' + name[len('bms_h'):], src, a)
        assert sorted(tgt) == [1, 3], f'preprocessing member sum keys {sorted(tgt)}'
        (ta, ia, lp), (tb, ib, _) = tgt[1], tgt[3]
        ins = [ia, ib]
        mseq = self.memsum_eq(lo, lp, X, ins)
        Kterm = qual_on(kt.group(1).replace('(Stage_online_eval.', '(Stage_online_eval.'))
        body_end = [f'  funext k', '  fin_cases k']
        for k, (t, c) in enumerate(((ta, 1), (tb, 3))):
            body_end.append(f'  · show blockModSwitchG {kt.group(2)} 1 {O.W(X[c])} = {P.W(t)}\n'
                            f'    rw [{PK_NS}.bms_h{t[2:]} FP, {mseq[c]}]')
        return (f'(fun k ↦ blockModSwitchG {kt.group(2)} 1 (Enc.A ({", ".join(O.W(w) for w in X)}, ()) k 1)) = '
                f'![{P.W(ta)}, {P.W(tb)}]'), body_end

    def memsum_eq(self, lo, lp, X, ins):
        """`O.X_c = P.Y_c` (c < 4) of paired member sums from the paired member families."""
        O, P = self.O, self.P
        ost = stmt(self.ok, f'memsum_{lo["field"]}')
        pst = stmt(self.P_keys, f'memsum_{lp["field"]}')
        ofam = re.findall(r'famAt \(\S+\.(w_\d+_\d+) ctx\) i', ost)
        pfam = re.findall(r'famAt \(\S+\.(w_\d+_\d+) ctx\) i', pst)
        ko, kp = len(ofam), len(pfam)
        pl = self.place_eq(ofam, pfam)
        names = {}
        for c in range(4):
            e = self.fresh('ms')
            self.body.append(f'  have {e} : {O.W(re.findall(r"(w_\d+_\d+) ctx\) = ∑", ost)[c])} = '
                             f'{P.W(re.findall(r"(w_\d+_\d+) ctx\) = ∑", pst)[c])} := by\n'
                             f'    rw [({OK_NS}.memsum_{lo["field"]} FO){proj(c, ko)}, ({PK_NS}.memsum_{lp["field"]} FP){proj(c, kp)}, {pl[c]}]')
            names[c] = e
        return names

    def place_eq(self, ofam, pfam):
        """`O.fam_c = P.fam_c` (c < 4) of paired placements (`spec_*`, component-wise) of paired members."""
        O, P = self.O, self.P
        ao, _ = O.loop_of(ofam[0])
        ap, _ = P.loop_of(pfam[0])
        assert O.role(ao) == 'sampler.placement' and P.role(ap) == 'sampler.placement'
        mo, _ = O.loop_of(ao['loop']['args'][0])
        mp, _ = P.loop_of(ap['loop']['args'][0])
        mk = self.loop_eq(mo, mp)
        k = len(ao['loop']['outs'])
        res = {}
        N = re.search(r'Placement\.Spec \(k := \d+\) (\S+) ', stmt(self.ok, f'spec_{ao["field"]}')).group(1)
        ins_o = [base(x) for x in ao['loop']['args']]
        ins_p = [base(x) for x in ap['loop']['args']]
        for c in range(4):
            e = self.fresh('pl')
            ex = f'(Int.fmod (((i : Nat)) : Int) {N})'
            self.body.append(f'  have {e} : {O.W(ofam[c])} = {P.W(pfam[c])} := funext fun i ↦ by\n'
                             f'    have h1 : {O.W(ofam[c])} i = multiplyMonomial ({O.W(ins_o[c])} i) {ex} := ({OK_NS}.spec_{ao["field"]} FO i) {c}\n'
                             f'    have h2 : {P.W(pfam[c])} i = multiplyMonomial ({P.W(ins_p[c])} i) {ex} := ({PK_NS}.spec_{ap["field"]} FP i) {c}\n'
                             f'    rw [h1, h2, ({mk} i){proj(c, 4)}]')
            res[c] = e
        return res

    # ------------------------------------------------------------------ the fresh row
    def fresh_row(self):
        """`fresh_keys`: the fresh summand's named witness `Kf = fun k _ l ↦ (bms (Enc.A (blkT X c) k 1) * D) 0 0`
        is the preprocessing's slice rows (`slice_column_total` on the preprocessing's slice loops)."""
        O, P = self.O, self.P
        fs = open(f'{self.pdir}/Gen_Summands_FreshSummand.lean').read()
        kt = re.search(r'∧\n      Kf = (fun k _ l ↦ \(\(blockModSwitchG (\[[\d, ]*\]) 1 \(Enc\.A \(blkT \(C := (\d+)\) \(m := (\d+)\) \((.*?)\) \(colOf l\)\) k 1\) : (ExactMatrix .*?)\) \* \(\S+\.(w_\d+_\d+) ctx\)\) 0 0) :=', fs, re.S)
        B, C, m, Xt, Ty, Dw = kt.group(2), kt.group(3), kt.group(4), kt.group(5), kt.group(6), kt.group(7)
        X = re.findall(r'(w_\d+_\d+) ctx', Xt)
        Ty = qual_on(Ty.replace('(iS : Nat)', 'i_0'))
        ps = open(f'{self.pdir}/Gen_Summands_PreSummand.lean').read()
        tg = re.search(r'\) \+ !\[\(\S+\.(w_\d+_\d+) cP\), \(\S+\.(w_\d+_\d+) cP\)\] k \+ !\[', ps)
        targets = list(tg.groups())
        # the preprocessing slice loops, their moved-down rows and the column sums
        lo = [a for a in O.loops if X[1] in xsem.tuple_wires(a['loop'].get('final') or '') and O.role(a) == 'sum.columns.step'][0]
        rows = []
        for k, t in enumerate(targets):
            sl = [a for a in P.loops if t in xsem.tuple_wires(a['loop'].get('final') or '') and P.role(a) == 'sum.columns.slice-step'][0]
            xw, vw = base(sl['loop']['args'][1]), base(sl['loop']['args'][2])
            bm = P.prod[xw][0]
            assert bm['head'].endswith('blockModSwitchRuns')
            src = xsem.wire_of(bm['args'][-2])
            init = xsem.tuple_wires(sl['loop']['init'])[0]
            ta = P.prod[init][0]
            assert ta['head'].endswith('tensorRuns')
            h, ys = xsem.app(P.j['wires'][xsem.wire_of(ta['args'][1])]['term'])
            sa = P.prod[xsem.wire_of(ys[0])][0]
            rows.append(dict(t=t, sl=sl, x=xw, v=vw, bm=bm, src=src, ta=ta, sa=sa))
        lp = [a for a in P.loops if rows[0]['src'] in xsem.tuple_wires(a['loop'].get('final') or '') and P.role(a) == 'sum.columns.step'][0]
        ost = stmt(self.ok, f'colsum_{lo["field"]}')
        pst = stmt(self.P_keys, f'colsum_{lp["field"]}')
        ofam = re.findall(r'famAt \(\S+\.(w_\d+_\d+) ctx\)', ost)
        pfam = re.findall(r'famAt \(\S+\.(w_\d+_\d+) ctx\)', pst)
        ofin = re.findall(r'(w_\d+_\d+) ctx\) = ∑', ost)
        pfin = re.findall(r'(w_\d+_\d+) ctx\) = ∑', pst)
        pl = self.place_eq(ofam, pfam)
        hc = {}
        for k, r in enumerate(rows):
            c = 2 * k + 1
            assert pfin.index(r['src']) == c and ofin.index(X[c]) == c, 'fresh components'
            e = self.fresh('cs')
            self.body.append(f'  have {e} : {O.W(X[c])} = {P.W(r["src"])} := by\n'
                             f'    rw [({OK_NS}.colsum_{lo["field"]} FO){proj(c, len(ofin))}, ({PK_NS}.colsum_{lp["field"]} FP){proj(c, len(pfin))}, {pl[c]}]\n'
                             f'    try rfl')
            hc[k] = e
        # the digits of the slice vector (same target at both stages up to the instance tables)
        hD = self.digits_eq(Dw, rows[0]['v'])
        for r in rows[1:]:
            assert r['v'] == rows[0]['v']
        concl = (f'(fun k _ l ↦ ((blockModSwitchG {B} 1 (Enc.A (blkT (C := {C}) (m := {m}) '
                 f'({", ".join(O.W(w) for w in X)}, ()) (colOf l)) k 1) : {Ty}) * {O.W(Dw)}) 0 0) = '
                 f'![{P.W(targets[0])}, {P.W(targets[1])}]')
        end = ['  funext k r l', '  fin_cases k']
        for k, r in enumerate(rows):
            hs, hb = self.fresh('sc'), self.fresh('sb')
            self.body.append(f'  have {hs} := slice_column_total (C := {C}) (m := {m}) (by norm_num) {P.F(r["sa"])} rfl {P.F(r["ta"])} {P.F(r["sl"])}\n'
                             f'    (fun i cur next hb ↦ {PRE}.{r["sl"]["loop"]["child"]}.sem hb)\n'
                             f'  have {hb} := fun c : Fin {C} ↦ blockModSwitchRuns_eqG (blockModSwitchRuns_blk (C := {C}) (m := {m}) {P.F(r["bm"])} c)')
            end.append(f'  · have hr : r = 0 := Subsingleton.elim _ _\n'
                       f'    subst hr\n'
                       f'    show ((blockModSwitchG {B} 1 (blk {O.W(X[2 * k + 1])} (colOf l)) : {Ty}) * {O.W(Dw)}) 0 0 = {P.W(r["t"])} 0 l\n'
                       f'    rw [entry_blk {P.W(r["t"])} l, {hs} (colOf l), {hb} (colOf l), {hc[k]}, {hD}]\n'
                       f'    rfl')
        return concl, end

    def digits_eq(self, ow, pw):
        """`O.ow = P.pw` for two rebased decompositions of constant polynomials (`rdecG_eq`), equal
        up to the instance tables."""
        O, P = self.O, self.P
        def chain(S, w):
            r = S.prod[w][0]
            d = S.prod[xsem.wire_of(r['args'][-2])][0]
            c = xsem.wire_of(d['args'][-2])
            return r, d, c
        ro, do, co = chain(O, ow)
        rp, dp, cp = chain(P, pw)
        so = re.findall(r'(sel_\d+)', O.j['wires'][co]['term'])
        sp_ = re.findall(r'(sel_\d+)', P.j['wires'][cp]['term'])
        hs = [self.sel_eq(a, b) for a, b in zip(so, sp_)]
        e = self.fresh('dg')
        self.body.append(f'  have {e} : {O.W(ow)} = {P.W(pw)} := by\n'
                         f'    rw [rdecG_eq {O.F(do)} {O.F(ro)}, rdecG_eq {P.F(dp)} {P.F(rp)}]\n'
                         f'    simp only [{ON}.{O.inst}.{co}, {PRE}.{P.inst}.{cp}]\n'
                         + (f'    try rw [{", ".join(hs)}]' if hs else '    try rfl'))
        return e

    # ------------------------------------------------------------------ the module
    def emit(self):
        O, P = self.O, self.P
        self.P_keys = self.pre
        thms = []
        todo = ((('mask_keys', self.mask, "The mask row's keys are the preprocessing's keys** (the parts' `hpre`'s `Kg`): the\nonline keys (`mask_summand`'s named witness) follow the data flow to the preprocessing key forms."),)
                if SIGMA else
                (('fresh_keys', self.fresh_row, "The fresh row's keys are the preprocessing's fresh keys** (`hpre`'s `Kf`): the online\nkeys (`fresh_summand`'s named witness) follow the data flow to the preprocessing slice rows."),))
        for nm, fn, doc in todo:
            self.body, self.haves, self.hyps, self.haves_after = [], [], {}, []
            self.sec, self.secn = '', []
            self.done = {}
            concl, end = fn()
            thms.append((nm, self.theorem(nm, doc, concl, end)))
        return thms

    def theorem(self, nm, doc, concl, end):
        O, P = self.O, self.P
        text = '\n'.join(self.body).replace('SECARGS', ' '.join(self.secn))
        hyps = '\n'.join(f'    ({k} : {v})' for k, v in self.hyps.items())
        thm = f'''/-- **{doc} -/
theorem {nm} {{hashModel : HashModel}}
    {{pparams : {PRE}.Params}} {{eparams : {ENC}.Params}} {{oparams : {ON}.Params}}
    {{i_0 : Nat}} [hS : Fact (i_0 < {O.S})]
    {{pctx : {PRE}.{P.inst}.Ctx Backend.backend hashModel pparams i_0}} {{pout}}
    (FP : {PRE}.{P.inst}.Facts pctx pout)
    {{ectx : {ENC}.{self.E.inst}.Ctx Backend.backend hashModel eparams i_0}} {{eout}}
    (FE : {ENC}.{self.E.inst}.Facts ectx eout)
    {{octx : {ON}.{O.inst}.Ctx Backend.backend hashModel oparams i_0}} {{oout}}
    (FO : {ON}.{O.inst}.Facts octx oout)
    {self.sec}
{hyps} :
    {concl} := by
{chr(10).join(dict.fromkeys(self.haves))}
{text}
{chr(10).join(end)}
'''
        return thm

    def module(self, thms):
        return f'''import GenKeyChain
import {'Gen_Inst_' + PRE + '_Keys' if SIGMA else 'Gen_Inst_Keys'}
import Gen_Inst_{ENC}
import Gen_Inst_{ON}_Keys
import Gen_Inst_{ON}_KeyChain
import GenFinalFresh
import GenInstLemmas

/-! Generated by `generator/gen_key_eq.py` (WP D): the keys of the online final combination's
mask carrier and fresh row are the preprocessing's combination keys (see the generator's docstring). -/

set_option maxHeartbeats 64000000
set_option maxRecDepth 262144
set_option linter.unusedVariables false

namespace RefreshTopProof.GenInst.{NS}

open RefreshTopProof.GenIface RefreshTopProof.GenIface.BrLut RefreshTopProof.GenIface.BrLutKey
open RefreshTopProof.Gen RefreshTopProof.GenInst {PK_NS} RefreshTopProof.GenFinal
open Mxx.Primitives MxxRuntime MxxExponentLut
open scoped BigOperators

{thms}
end RefreshTopProof.GenInst.{NS}
'''


#: The namespace and module stem: the parts' `KeyEq`, or the sigma stages' `KeyEqSigma`.
NS = 'KeyEqSigma' if SIGMA else 'KeyEq'


if __name__ == '__main__':
    gdir, pdir, rolesj = [a for a in sys.argv[1:] if not a.startswith('--')][:3]
    g = KE(gdir, pdir, json.load(open(rolesj)))
    # one module per theorem (one module of both exceeds the stack at n = 2^17), and an umbrella
    mods = []
    for nm, th in g.emit():
        mod = f'Gen_Inst_{NS}_' + ''.join(x.title() for x in nm.split('_'))
        open(f'{pdir}/{mod}.lean', 'w').write(g.module(th))
        mods.append(mod)
    open(f'{pdir}/Gen_Inst_{NS}.lean', 'w').write(''.join(f'import {m}\n' for m in mods) +
        '\n/-! Generated by `generator/gen_key_eq.py` (WP D): the key equalities of the final combination. -/\n')
    print(f'{pdir}/Gen_Inst_{NS}.lean: {", ".join(mods)}')
