#!/usr/bin/env python3
"""Generate `Gen_Inst_Keys.lean` (WP D): the online smudge members' keys equal the preprocessing's
smudge key members (`hkeys` of the final combination's preimage target).

For every online `smudge.member` loop the generator follows the data flow backwards by role:
member <- `prf.rowsum` <- `prf.cell1` (gather) <- `prf.rowsum` <- `prf.cell0` (gather), and pairs it
with the preprocessing chain `smudge.key-member` <- `prf.rowsum` <- `prf.key-cell1` <- `prf.rowsum`
<- `prf.key-cell0` (the member whose switching family is the online member's, through the stage
links). Inputs of the online cells are resolved to preprocessing wires through the stage links
(`Gen_Stage_Links.lean` names), through the encrypt stage's `enc.label-entry` pass-through and
through equal hash samples (`hashSample_unique`). Every link is a hypothesis `lO_w… / lE_w…`
(function equality) of the emitted theorem, discharged by `Gen_Stage_Links` in the part theorem.

The proof terms of the online cells / members (layout certificates, divisibility facts, adapters)
are taken from the segment theorems `seg_<field>` of `Gen_Inst_Stage_online_eval.lean`
(gen_instance.py); the preprocessing forms are WP F's `keys_<field>`, `spec_<field>`,
`member_<field>` (`Gen_Inst_Stage_preprocess_Keys.lean`). Nothing is recognized by wire ids or
scope numbers.

Usage: gen_inst_keys.py <generated dir> <param dir> <roles.json>
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
import xmodel  # noqa: E402
from gen_instance import facts_fields  # noqa: E402

ON, PRE, ENC = 'Stage_online_eval', 'Stage_preprocess', 'Stage_encrypt'
CTX = {ON: 'octx', PRE: 'pctx', ENC: 'ectx'}
FAC = {ON: 'FO', PRE: 'FP', ENC: 'FE'}


def base(x):
    return x.split(' ')[0].replace('_gathered', '') if x else x


class St:
    def __init__(self, gdir, M, roles):
        self.M = M
        sc = xmodel.load(gdir, M)
        root = sc['generatedRoot'].to_json()
        rl = [a for a in root['atoms'] if a.get('loop') and a['loop']['kind'] == 'par'][0]['loop']
        self.inst, self.S = rl['child'], rl['count']
        self.j = sc[self.inst].to_json()
        self.fields = facts_fields(open(f'{gdir}/{M}_Facts.lean').read(), self.inst)
        self.roles = roles.get(M, {})
        self.loops = [a for a in self.j['atoms'] if a.get('loop')]
        self.prod = {}
        for a in self.j['atoms']:
            if a.get('loop'):
                for k, o in enumerate(a['loop']['outs']):
                    self.prod[base(o)] = (a, k)
            elif a['args']:
                self.prod[base(a['args'][-1])] = (a, None)

    def role(self, a):
        return self.roles.get(a['loop']['child'], {}).get('role')

    def by_role(self, r):
        return [a for a in self.loops if self.role(a) == r]

    def W(self, w):
        return f'({self.M}.{self.inst}.{w} {CTX[self.M]})'

    def F(self, a):
        return f'{FAC[self.M]}.c{self.fields[a["field"]][0]}.{a["field"]}'

    def loop_of(self, w):
        p = self.prod.get(base(w))
        return p if p and p[0].get('loop') else (None, None)


def parse_links(path):
    L = {}
    for m in re.finditer(r'^theorem link_(Stage_[a-z_]+?)_(w_\d+_\d+)_(Stage_[a-z_]+?)_(w_\d+_\d+) ',
                         open(path).read(), re.M):
        a, b = (m.group(1), m.group(2)), (m.group(3), m.group(4))
        L.setdefault(a, set()).add(b)
        L.setdefault(b, set()).add(a)
    return L


def seg_text(text, field):
    m = re.search(r'^theorem seg_%s\b.*?(?=^theorem |\Z)' % re.escape(field), text, re.S | re.M)
    return m.group(0)


def qual_on(t):
    t = re.sub(r'(?<![\w.])((?:sequential|parallel)_parallel_\w*generatedRoot_\d+_\w+)', ON + r'.\1', t)
    t = re.sub(r'\(parallel_generatedRoot_(\d+)\.(w_\d+_\d+) ctx\)', '(' + ON + r'.parallel_generatedRoot_\1.\2 octx)', t)
    t = re.sub(r'(?<![\w.(])parallel_generatedRoot_(\d+)\.(w_\d+_\d+) ctx\b', '(' + ON + r'.parallel_generatedRoot_\1.\2 octx)', t)
    t = re.sub(r'(?<![\w.])(modulus_\d+|mod_\d+|sel_\d+)\b', ON + r'.\1', t)
    t = re.sub(r'\b(Backend\.backend|backend) hashModel params i_0', 'Backend.backend hashModel oparams i_0', t)
    t = t.replace('F.c', 'FO.c')
    return t


def qual_pre(t, inst):
    t = re.sub(r'\(%s\.(w_\d+_\d+) ctx\)' % inst, '(' + PRE + '.' + inst + r'.\1 pctx)', t)
    t = re.sub(r'(?<![\w.(])%s\.(w_\d+_\d+) ctx\b' % inst, '(' + PRE + '.' + inst + r'.\1 pctx)', t)
    t = re.sub(r'(?<![\w.])(modulus_\d+|mod_\d+|sel_\d+)\b', PRE + r'.\1', t)
    return t


class Gen:
    def __init__(self, gdir, pdir, roles):
        self.O, self.P, self.E = (St(gdir, M, roles) for M in (ON, PRE, ENC))
        self.links = parse_links(f'{pdir}/Gen_Stage_Links.lean')
        self.seg = open(f'{pdir}/Gen_Inst_Stage_online_eval.lean').read()
        self.pre = open(f'{pdir}/Gen_Inst_Stage_preprocess_Keys.lean').read()
        self.hyps = {}  # name -> statement
        self.haves = []

    # ------------------------------------------------------------------ links
    def link(self, X, w, Y):
        """Name of the hypothesis `X.w = Y.w'` and `w'`, or None."""
        for (M, w2) in sorted(self.links.get((X.M, w), ())):
            if M == Y.M:
                tag = 'lO' if X is self.O else 'lE'
                name = f'{tag}_{w}'
                self.hyps[name] = f'{X.W(w)} = {Y.W(w2)}'
                return name, w2
        return None

    def to_pre(self, w):
        """Rewrite chain (names) from the online wire `w` to a preprocessing wire, and that wire.
        `pass` entries are pass-through equalities of encrypt label entries at member `k`."""
        d = self.link(self.O, w, self.P)
        if d:
            return [d[0]], d[1], None
        d = self.link(self.O, w, self.E)
        assert d, f'no link for online {w}'
        e = d[1]
        d2 = self.link(self.E, e, self.P)
        if d2:
            return [d[0], d2[0]], d2[1], None
        a, c = self.E.loop_of(e)
        assert a is not None and self.E.role(a) == 'enc.label-entry', f'encrypt {e}'
        src = base(a['loop']['args'][c])
        d3 = self.link(self.E, src, self.P)
        assert d3, f'encrypt {src}'
        return [d[0], ('pass', a, c, e, src), d3[0]], d3[1], a

    def hash_eq(self, u, t):
        """`P.u = P.t` for equal hash samples (name of a `have`)."""
        if u == t:
            return None
        P = self.P
        au, at = P.prod[u][0], P.prod[t][0]
        assert au['head'].endswith('hashSample') and at['head'].endswith('hashSample')
        assert au['args'][:-1] == at['args'][:-1], f'{u} {t}'
        name = f'u_{u}_{t}'
        self.haves.append(f'  have {name} : {P.W(u)} = {P.W(t)} := hashSample_unique {P.F(au)} {P.F(at)}')
        return name

    def root_chain(self, w, target):
        ch, p, _ = self.to_pre(w)
        u = self.hash_eq(p, target)
        return ch + ([u] if u else [])

    # ------------------------------------------------------------------ preprocessing statements
    def pre_binders(self, field):
        """The secret binders of F's `keys_<field>` after `hb` (qualified) and their explicit names."""
        m = re.search(r'theorem keys_%s .*?\(hb : backend = Backend\.backend\)(.*?) :\n    ∀ i' % field, self.pre, re.S)
        sec = ' '.join(m.group(1).split())
        names = re.findall(r'\((\w+) : ', sec)
        return qual_pre(sec, self.P.inst), names

    def label_form(self, field):
        m = re.search(r'theorem keys_%s .*?labelKeysG hashModel (\(.*?\) \(.*?\) \(.*?\) \(\(\[[\d, ]*\]\) : Blob\))'
                      % field, self.pre, re.S)
        return qual_pre(m.group(1), self.P.inst)

    # ------------------------------------------------------------------ pieces
    def hlab(self, cell, pcell, name):
        """`![![x00 k, x01 k], ![x10 k, x11 k]] = labelKeysG … k` for the gathered label families."""
        O, P = self.O, self.P
        g = [base(x[0]) for x in cell['loop']['gathered']]
        pos = {(0, 0): g[1], (1, 0): g[2], (0, 1): g[4], (1, 1): g[5]}
        lab = self.label_form(pcell['field'])
        lines = [f'  have {name} : ∀ k : Fin {O.wires_count(g[1])}, ![![{O.W(pos[0, 0])} k, {O.W(pos[0, 1])} k], '
                 f'![{O.W(pos[1, 0])} k, {O.W(pos[1, 1])} k]] =\n      labelKeysG hashModel {lab} ((k : Nat) : Int) := by',
                 '    intro k']
        cases, entries = [], {}
        for (a_, b_), w in sorted(pos.items()):
            ch, p, ea = self.to_pre(w)
            pa, _ = P.loop_of(p)
            assert P.role(pa) == 'label.public'
            rw = []
            for c in ch:
                if isinstance(c, tuple):
                    _, la, ci, e, src = c
                    hn = f'q_{e}'
                    if la['field'] not in entries:
                        entries[la['field']] = la
                        lines.append(f'    obtain ⟨_, _, _, _, _, _, hout_{la["field"]}⟩ := '
                                     f'{ENC}.{la["loop"]["child"]}.sem ({self.E.F(la)} k)')
                    proj = '.1' if ci == 0 else '.2' * ci + '.1'
                    lines.append(f'    have {hn} : {self.E.W(e)} k = {self.E.W(src)} k := '
                                 f'congrArg (·{proj}) hout_{la["field"]}')
                    rw.append(hn)
                else:
                    rw.append(c)
            cases.append(f'    · show {O.W(w)} k = _\n      rw [{", ".join(rw)}]\n'
                         f'      exact keyPairG_of_hash (pre := [.bytes memberTagG, .u64Le (Int.ofNat k)]) {a_} {b_} '
                         f'(by norm_num) (spec_{pa["field"]} FP k)')
        lines += ['    funext a b', '    fin_cases a <;> fin_cases b'] + cases
        return '\n'.join(lines)

    def keys_call(self, cell, kind):
        """The `PrfCell*.keys` application from the segment theorem of `cell`."""
        t = seg_text(self.seg, cell['field'])
        obt = re.findall(r"^  (obtain ⟨L[\w']*, .*?⟩ := RefreshTopProof\.GenInst\.Certs\.\S+)$", t, re.M)
        m = re.search(r'(PrfCell[01])\.rel (.*?\(h := [^\n]*?\.sem hc\))', t, re.S)
        call = f'{m.group(1)}.keys (p\' := 1) {m.group(2)} k hk.symm'
        if kind == 1:
            call += (' (gadgetOf_runs (gadget_exists ⟨_, by simp [Backend.backend]; rfl, rfl, rfl, rfl⟩))'
                     ' (rebasedGadget_runs (by decide))')
        gat = re.search(r'^  (obtain ⟨k, hk, hc⟩ := Gen\.gather_loop6 .*?)$', t, re.M).group(1)
        call = re.sub(r'\n\s*', '\n      ', call)
        return qual_on(gat), [qual_on(o) for o in obt], qual_on(call)

    def cell_eq(self, name, cell, pcell, kind, lab, prev):
        O, P = self.O, self.P
        n = cell['loop']['count']
        oo = [base(x) for x in cell['loop']['outs']]
        po = [base(x) for x in pcell['loop']['outs']]
        concl = ' ∧ '.join(f'{O.W(oo[c])} i = {P.W(po[c])} i' for c in range(4))
        gat, obt, call = self.keys_call(cell, kind)
        args = [base(x) for x in cell['loop']['args']]
        pargs = [base(x) for x in pcell['loop']['args']]
        rw = [lab + ' k', 'hk'] + self.to_pre(args[0])[0]
        if kind == 0:
            # online As (positions 7, 8, 11, 12) -> preprocessing As (2, 3, 4, 5: A00, A10, A01, A11)
            for oi, pi in ((7, 2), (11, 4), (8, 3), (12, 5)):
                rw += self.root_chain(args[oi], pargs[pi])
        else:
            rw += [f'({prev} o).1', f'({prev} o).2.1', f'({prev} o).2.2.1', f'({prev} o).2.2.2']
        rw += self.to_pre(args[14])[0] + self.to_pre(args[15])[0] + self.to_pre(args[16])[0]
        pk = f'keys_{pcell["field"]} FP rfl SECARGS i'
        s = [f'  have {name} : ∀ i : Fin {n}, {concl} := by', '    intro i', '    ' + gat]
        s += ['    ' + o for o in obt]
        if kind == 0:
            s += [f'    have hO := {call}', f'    rw [{", ".join(rw)}] at hO',
                  f'    have e := hO.trans ({pk}).1.symm']
        else:
            s += [f'    obtain ⟨o, ho, hO⟩ := {call}', f'    obtain ⟨o\', ho\', hP⟩ := ({pk})',
                  '    have hoo : o = o\' := Fin.ext (by have := ho.trans ho\'.symm; exact_mod_cast this)',
                  '    subst hoo', f'    rw [{", ".join(rw)}] at hO', '    have e := hO.trans hP.1.symm']
        s.append('    exact ⟨congrFun (congrFun e 0) 0, congrFun (congrFun e 0) 1, '
                 'congrFun (congrFun e 1) 0, congrFun (congrFun e 1) 1⟩')
        return '\n'.join(s)

    def rowsum_eq(self, name, rs, prs, prev):
        O, P = self.O, self.P
        W = self.O.roles[rs['loop']['child']]['params']['W']
        n = rs['loop']['count']
        oo = [base(x) for x in rs['loop']['outs']]
        oa = [base(x) for x in rs['loop']['args']]
        po = [base(x) for x in prs['loop']['outs']]
        pa = [base(x) for x in prs['loop']['args']]
        concl = ' ∧ '.join(f'{O.W(oo[c])} o = {P.W(po[c])} o' for c in range(4))
        s = [f'  have {name} : ∀ o : Fin {n}, {concl} := by', '    intro o',
             f'    have hO := ({ON}.{rs["loop"]["child"]}.sem ({O.F(rs)} o)).2',
             f'    have hP := (spec_{prs["field"]} FP o).2']
        acc = ['.1', '.2.1', '.2.2.1', '.2.2.2']
        for c in range(4):
            s += [f'    have f{c} : {O.W(oa[c])} = {P.W(pa[c])} := funext fun i ↦ ({prev} i){acc[c]}',
                  f'    have o{c} : {O.W(oo[c])} o = ∑ j ∈ Finset.range {W}, famAt {O.W(oa[c])} ({W} * (o : Nat) + j) := hO {c}',
                  f'    have p{c} : {P.W(po[c])} o = ∑ j ∈ Finset.range {W}, famAt {P.W(pa[c])} ({W} * (o : Nat) + j) := hP {c}']
        s.append('    exact ⟨' + ', '.join(f'by rw [o{c}, p{c}, f{c}]' for c in range(4)) + '⟩')
        return '\n'.join(s)

    def member(self, sm, psm, prev):
        O = self.O
        t = seg_text(self.seg, sm['field'])
        lay = re.search(r'^  obtain ⟨L, hL, hLx, -⟩ := (\S+ i_0)$', t, re.M).group(1)
        hq = re.search(r'\(hq\' := (\(.*?\))\) \(layout', t).group(1)
        child = re.search(r'\(child := (fun k s nxt ↦ .*?)\)\n', t).group(1)
        fs = re.search(r'(\(f00 := .*?\(f11 := [^)]*\))', t).group(1)
        ad = re.search(r'(\(fun _ _ _ hc ↦ \S+\.sem hc\))', t).group(1)
        N = re.search(r'\(N := (\d+)\)', t).group(1)
        dm = re.search(r'\(d := (\d+)\)', t)
        if dm:  # migrated bridges: weight `c P^(i mod d)`, `L` looped branches
            Lm = re.search(r'\(L := (\d+)\)', t).group(1)
            shape = f'(d := {dm.group(1)}) (hD := rfl) (N := {N}) (hNN := rfl) (L := {Lm}) (hℓ := rfl) (hL := rfl)'
        else:
            shape = f'(hdd := rfl) (N := {N}) (hNN := rfl) (hW := by decide)'
        args = [base(x) for x in sm['loop']['args']]
        rw = []
        for w in (args[5], args[0], args[1], args[2], args[3]):
            rw += self.to_pre(w)[0]
        rw += [f'({prev} i).1', f'({prev} i).2.1', f'({prev} i).2.2.1', f'({prev} i).2.2.2']
        oo = [base(x) for x in sm['loop']['outs']]
        po = [base(x) for x in psm['loop']['outs']]
        s = [f'  obtain ⟨L, hL, hLx, hd⟩ := {qual_on(lay)}',
             f'  have hO := SmudgeMember.keys (h := {ON}.{sm["loop"]["child"]}.sem ({O.F(sm)} i))',
             f'    (hmw := rfl) {shape} (hn := by decide)',
             f'    (hq\' := {qual_on(hq)}) L hL hLx (by rw [hd])',
             f'    (LutRoundStep.keystep (child := {qual_on(child)})',
             f'      {qual_on(fs)}',
             f'      {qual_on(ad)} rfl (by decide)',
             f'      {qual_on(hq)} L hL hLx (by rw [hd]))',
             f'  rw [{", ".join(rw)}] at hO',
             f'  have e := hO.trans (member_{psm["field"]} FP i).symm']
        return s, oo, po

    # ------------------------------------------------------------------ the theorem
    def chain(self, sm):
        """The online bucket chain `[(cell0, rowsum), (cell1, rowsum), …]` of the member and the
        preprocessing one, paired bucket by bucket (by data flow from the members back)."""
        O, P = self.O, self.P
        f00 = base(sm['loop']['args'][0])
        pf = self.to_pre(f00)[1]
        psm = [a for a in P.by_role('smudge.key-member') if pf in [base(x) for x in a['loop']['args']]][0]

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
                x = c['loop']['args'][as_pos]
        on = walk(O, sm['loop']['args'][8], 'prf.cell1', 'prf.cell0', 7)
        pre = walk(P, psm['loop']['args'][1], 'prf.key-cell1', 'prf.key-cell0', 2)
        assert len(on) == len(pre), (len(on), len(pre))
        return psm, on, pre

    def emit(self):
        O, P = self.O, self.P
        sms = O.by_role('smudge.member')
        out = []
        for sm in sms:
            self.hyps, self.haves = {}, []
            psm, on, pre = self.chain(sm)
            sec, secn = self.pre_binders(pre[0][0]['field'])
            body, labs = [], {}
            for (c, _), (pc, _) in zip(on, pre):
                key = tuple(x[0] for x in c['loop']['gathered'])
                if key not in labs:
                    labs[key] = f'hlab{len(labs)}'
                    body.append(self.hlab(c, pc, labs[key]))
            body.append('__HAVES__')
            prev = None
            for t, ((c, r), (pc, pr)) in enumerate(zip(on, pre)):
                lab = labs[tuple(x[0] for x in c['loop']['gathered'])]
                body.append(self.cell_eq(f'kc{t}', c, pc, 0 if t == 0 else 1, lab, prev))
                body.append(self.rowsum_eq(f'er{t}', r, pr, f'kc{t}'))
                prev = f'er{t}'
            mem, oo, po = self.member(sm, psm, prev)
            n = sm['loop']['count']
            body.append('  intro i')
            body += mem
            body.append('  exact ⟨congrArg (·.2.1) e, congrArg (·.2.2.2.1) e⟩')
            text = '\n'.join(body).replace('__HAVES__', '\n'.join(self.haves)).replace('SECARGS', ' '.join(secn))
            hyps = '\n'.join(f'    ({k} : {v})' for k, v in self.hyps.items())
            out.append(f'''/-- **The keys of the online smudge members `{sm['field']}`** are the preprocessing's smudge key
members `{psm['field']}` (components 1 and 3: the keys the preimage target reads), given the stage
links (`Gen_Stage_Links`). -/
theorem member_keys_{sm['field']} {{hashModel : HashModel}}
    {{pparams : {PRE}.Params}} {{eparams : {ENC}.Params}} {{oparams : {ON}.Params}}
    {{i_0 : Nat}} [hS : Fact (i_0 < {O.S})]
    {{pctx : {PRE}.{P.inst}.Ctx Backend.backend hashModel pparams i_0}} {{pout}}
    (FP : {PRE}.{P.inst}.Facts pctx pout)
    {{ectx : {ENC}.{self.E.inst}.Ctx Backend.backend hashModel eparams i_0}} {{eout}}
    (FE : {ENC}.{self.E.inst}.Facts ectx eout)
    {{octx : {ON}.{O.inst}.Ctx Backend.backend hashModel oparams i_0}} {{oout}}
    (FO : {ON}.{O.inst}.Facts octx oout)
    {sec}
{hyps} :
    ∀ i : Fin {n}, {O.W(oo[1])} i = {P.W(po[1])} i ∧ {O.W(oo[3])} i = {P.W(po[3])} i := by
{text}
''')
        return f'''import GenInstKeys
import Gen_Inst_Stage_preprocess_Keys
import Gen_Inst_Stage_online_eval
import {ENC}_Facts
import Gen_{ENC}_Scopes

/-! Generated by `generator/gen_inst_keys.py` (WP D): the online smudge members' keys are the
preprocessing's smudge key members (see the generator's docstring). -/

set_option maxHeartbeats 64000000
set_option maxRecDepth 16384
set_option linter.unusedVariables false

namespace RefreshTopProof.GenInst.Keys

open RefreshTopProof.GenIface RefreshTopProof.GenIface.BrLut RefreshTopProof.GenIface.BrLutKey
open RefreshTopProof.Gen RefreshTopProof.GenInst RefreshTopProof.GenInst.PreprocessKeys
open Mxx.Primitives MxxRuntime MxxExponentLut
open scoped BigOperators

{chr(10).join(out)}
end RefreshTopProof.GenInst.Keys
'''


def wires_count(self, w):
    return int(re.search(r'Fin (\d+)', self.j['wires'][w]['type']).group(1))


St.wires_count = wires_count

if __name__ == '__main__':
    gdir, pdir, rolesj = sys.argv[1], sys.argv[2], sys.argv[3]
    g = Gen(gdir, pdir, json.load(open(rolesj)))
    txt = g.emit()
    open(f'{pdir}/Gen_Inst_Keys.lean', 'w').write(txt)
    print(f'{pdir}/Gen_Inst_Keys.lean: {len(re.findall(r"^theorem ", txt, re.M))} theorems')
