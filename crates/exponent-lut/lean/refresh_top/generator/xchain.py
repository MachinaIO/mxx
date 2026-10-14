"""Composition of the online segment theorems (WP D).

The PRF chains (bucket cells -> row sums -> … -> LUT members -> placement -> member sums / column
rows) and the offset chain (root private multiplications -> root LUT rounds) are found by data flow
(consumers of loop outputs) and linked by role rules. Emits the theorem `chains` (every segment
hypothesis not produced inside the instance, conclusion `True`), which checks the composition, and
the list of end facts (named `have`s) for the part theorem.
"""
import re

import xsem
from xsem import wire_of, mat_type, qterm, Composer


def base(x):
    return wire_of(x.replace(' i', '')) if x else None


class Online:
    def __init__(self, I, roles, segs, scopes):
        self.I, self.roles, self.segs, self.scopes = I, roles, segs, scopes
        self.loops = [a for a in I.j['atoms'] if a.get('loop')]
        self.C = Composer()
        self.ends = []
        self.nn = xsem.ring_n(I.wires)

    # ------------------------------------------------------------------ data flow
    def role(self, a):
        return self.roles.get(a['loop']['child'], {}).get('role')

    def consumers(self, a):
        ws = {base(o) for o in a['loop']['outs']}
        out = []
        for b in self.loops:
            L = b['loop']
            if any(base(x) in ws for x in L['args']):
                out.append(b)
        return out

    def chain_nodes(self, a0):
        nodes = [a0]
        a = a0
        while True:
            nxt = [c for c in self.consumers(a) if c['loop']['kind'] in ('par', 'gather', 'seq')]
            if not nxt:
                return nodes
            # prefer the sole par/gather consumer; a seq consumer ends the member stages
            par = [c for c in nxt if c['loop']['kind'] in ('par', 'gather')]
            if par:
                a = par[0]
                nodes.append(a)
                continue
            nodes += nxt
            return nodes

    def W(self, w):
        return self.I.W(w)

    def tup(self, ws):
        return '(' + ', '.join(self.W(w) for w in ws) + ', ())'

    def scale(self, a):
        """`(q / q')` printed as the segment emitter of `a` prints its scale."""
        r = self.role(a)
        args = [base(x) for x in a['loop']['args']]
        if r in ('prf.cell0', 'prf.cell1'):
            # the state's ring (`As00`) and the output's (`bot0`); a Program in its own stages (13
            # arguments) passes its key instead of six label families
            st, bt = (4, 11) if len(args) == 13 else (7, 15)
            q, q2 = mat_type(self.I.wires[args[st]]['type'])[1], mat_type(self.I.wires[args[bt]]['type'])[1]
        else:
            q, q2 = mat_type(self.I.wires[args[7]]['type'])[1], mat_type(self.I.wires[args[0]]['type'])[1]
        return f'({qterm(q)} / {qterm(q2)})'

    def tag(self, a):
        return a['field'].split('_')[0]

    # ------------------------------------------------------------------ the PRF chains
    def prf_chain(self, a0, offset=None):
        C, I = self.C, self.I
        nodes = self.chain_nodes(a0)
        t0 = self.tag(a0)
        mu = None
        last_cell, last_bucket = None, None
        i = 0
        while i < len(nodes):
            a = nodes[i]
            r = self.role(a)
            t = self.tag(a)
            nxt = nodes[i + 1] if i + 1 < len(nodes) else None
            if r in ('prf.cell0', 'prf.cell1'):
                # the bucket's consumer after the row sum fixes this cell's output scale
                after = nodes[i + 2] if i + 2 < len(nodes) else None
                links = {'hb': 'hb', 'F': 'F', "p'": self.scale(after)}
                if r == 'prf.cell1':
                    links.update({'hs': last_bucket, 'μ': mu, 'g': f'hT_{last_cell}', 'Bs': None,
                                  'α': f'α_{t0}', 'hα': f'hα_{t0}', 'GA': f'G_{t0}', 'hGA': f'hG_{t0}',
                                  'Ba': f'Ba_{t0}', 'ha': f'ha_{t0}', 'hpay': f'hpay_{t0}'})
                C.add(self.segs[a['field']], t, links)
                if r == 'prf.cell0':
                    mu = f'(fun (_ : Fin {nxt["loop"]["count"]}) ↦ μ_{t0})'
                last_cell = t
            elif r == 'prf.rowsum':
                M = a['loop']['count']
                cell = nodes[i - 1]
                K = mat_type(I.wires[cell['loop']['gathered'][0][0]]['type'])[0]
                Wd = cell['loop']['count'] // M
                tab = base(cell['loop']['args'][0])
                ct = self.tag(cell)
                C.binders.append(f'{{sel_{ct} : Fin {M} → Fin {Wd}}} (hsel_{ct} : ∀ o j, ν_{ct} (Gen.cellAt o j) = '
                                 f'if j = sel_{ct} o then (1 : ErrorPoly {self.nn}) else 0)')
                C.binders.append(f'{{key_{ct} : Fin {M} → Fin {K}}} (hkey_{ct} : ∀ o, ((key_{ct} o : Nat) : Int) = '
                                 f'{self.W(tab)} (Gen.cellAt o (sel_{ct} o)))')
                lemma, cells = (('bucket_of_cells', f'(fun o j ↦ r_{ct} (Gen.cellAt o j))')
                                if self.role(cell) == 'prf.cell0' else ('bucket_of_cells1', f'r_{ct}'))
                b = f'b_{t}'
                C.steps.append(f'  have {b} := fun o ↦ {lemma} α_{t0} {mu} ({self.W(tab)}) ν_{ct} {cells} o\n'
                               f'    ({a["loop"]["child"]}.sem ({I.F(a)} o)) (sel_{ct} o) (hsel_{ct} o) (key_{ct} o) (hkey_{ct} o)')
                mu = f'(fun o ↦ α_{t0} (key_{ct} o) * {mu} o)'
                last_bucket = b
            elif r in ('lut.member.readout', 'smudge.member', 'lut.member.sampler', 'lut.member.sampler-offset'):
                links = {'hb': 'hb', 'F': 'F', 'B': None}
                if last_bucket is not None and (i == 0 or self.role(nodes[i - 1]) == 'prf.rowsum'):
                    links.update({'hinput': f'fun i ↦ {last_bucket} i', 'μ': mu, 'g': f'hT_{last_cell}'})
                else:
                    prev = self.tag(nodes[i - 1])
                    links.update({'hinput': f'fun i ↦ r_{prev} i', 'μ': f'ν_{prev}', 'g': f'gq_{prev}'})
                if r in ('lut.member.readout', 'smudge.member') and nxt is not None and \
                        self.role(nxt) in ('lut.member.sampler', 'lut.member.sampler-offset'):
                    links["Δ'"] = self.pcast(self.scale(nxt))
                if r == 'lut.member.sampler-offset':
                    if offset is None and getattr(self, '_pending_offset', None):
                        privs, luts, ct, offs = self._pending_offset
                        lt2 = self.offset_chain(privs, luts, ct)
                        C.steps.append(f'  have hoff_{t} := encRel_of_rel r_{lt2} (x := {self.tup(offs)}) '
                                       f'(by intro a j; fin_cases a <;> fin_cases j <;> rfl) (by intro j; fin_cases j <;> rfl)')
                        offset = f'hoff_{t}'
                    links.update({'hoff': offset, 'toff': None, 'Boff': None})
                C.add(self.segs[a['field']], t, links)
            elif r == 'sampler.placement':
                prev = self.tag(nodes[i - 1])
                N = re.search(r'Placement\.Spec \(k := 6\) (\d+)', self.scopes_text).group(1) if False else self.place_N(a)
                C.steps.append(f'  have r_{t} := fun i : Fin {a["loop"]["count"]} ↦ place_rel (by decide) '
                               f'({a["loop"]["child"]}.sem ({I.F(a)} i)) (r_{prev} i) (e := i % {N}) (fmod_nat i {N}) (by omega)')
            elif r == 'sum.members.step':
                self.member_sums(nodes[i:], nodes[i - 1])
                return
            i += 1

    def pcast(self, p):
        return f'((({p} : Nat) : Int) : ErrorPoly {self.nn})'

    def place_N(self, a):
        cj = self.scopes[a['loop']['child']].to_json()
        for w, v in cj['wires'].items():
            m = re.search(r'Int\.fmod \(\(?\(?Int\.ofNat i_1\)?\)?\) \((\d+)\)', v.get('term', ''))
            if m:
                return m.group(1)
        for x in cj['atoms']:
            m = re.search(r'Int\.fmod \(\(?\(?Int\.ofNat i_1\)?\)?\) \((\d+)\)', ' '.join(x['args']))
            if m:
                return m.group(1)
        raise KeyError('placement period')

    # ------------------------------------------------------------------ member sums, column rows
    def static_reads(self, fams, k):
        I = self.I
        res = []
        for f in fams:
            a = [x for x in I.j['atoms'] if not x.get('loop') and x['head'].endswith('familyGetStatic')
                 and wire_of(x['args'][0]) == f and x['args'][1].strip('()') == str(k)][0]
            res.append(a)
        return res

    def member_sums(self, seqs, prev):
        C, I = self.C, self.I
        pt = self.tag(prev)
        fams = [base(o) for o in prev['loop']['outs']]
        F_ = prev['loop']['count']
        by6 = lambda xs: '(by intro c; fin_cases c; exacts [' + ', '.join(xs) + '])'
        r0, r1 = self.static_reads(fams, 0), self.static_reads(fams, 1)
        o = lambda a: wire_of(a['args'][-1])
        s0 = seqs[0]
        t = self.tag(s0)
        cnt = s0['loop']['count'] + 2
        common = (f'(fams := {self.tup(fams)}) (a := {self.tup([o(x) for x in r0])}) (b := {self.tup([o(x) for x in r1])}) '
                  f'{by6([I.F(x) for x in r0])} {by6([I.F(x) for x in r1])} (by intro c; fin_cases c <;> rfl) {I.F(s0)} '
                  f'(fun _ _ _ hb ↦ {s0["loop"]["child"]}.sem hb)')
        if cnt == F_ and len(seqs) == 1:
            C.steps.append(f'  have r_{t} := member_total_rel (M := {F_}) (by decide) {common} (fun i ↦ r_{pt} i)')
            self.ends.append((f'r_{t}', f'member total of `{prev["field"]}`'))
            return
        # column rows (Step::Sum) are the fresh / smudge forms (WP G / F): end at the members
        self.ends.append((f'r_{pt}', f'members of `{prev["field"]}` (consumed by the column row `{s0["field"]}`, G/F)'))
        return
        C.steps.append(f'  have x_{t} := member_prefix (F := {F_}) (k := 6) (Wm := {cnt}) (by decide) (by decide) {common}')
        if len(seqs) < 2:
            return
        s1 = seqs[1]
        t1 = self.tag(s1)
        init = xsem.tuple_wires(s1['loop']['init'])
        tens = [I.prod[w] for w in init]
        Cc = s1['loop']['count'] + 1
        mm = mat_type(I.wires[fams[0]]['type'])[4]
        C.steps.append(f'  have r_{t1} := fun cc ↦ column_total_rel (C := {Cc}) (W := {cnt}) (F := {F_}) (m := {mm}) (by decide) (by decide) '
                       f'(fams := {self.tup(fams)}) (x0 := {self.tup(xsem.tuple_wires(s0["loop"]["final"]))}) x_{t} '
                       f'{by6([I.F(x) for x in tens])} {I.F(s1)} (fun _ _ _ hb ↦ {s1["loop"]["child"]}.sem hb) '
                       f'(fun i ↦ r_{pt} i) cc')
        self.ends.append((f'r_{t1}', f'column blocks of `{s1["field"]}` (members of `{prev["field"]}`)'))

    # ------------------------------------------------------------------ the offset chain
    def offset_chain(self, privs, luts, cell0_tag):
        """Root private multiplications (from the one encoding of the mask chain) -> root LUT round
        -> root LUT branches; returns the term of the offset relation (sampler-offset `hoff`)."""
        C = self.C
        prevT = None
        for k, (name, r) in enumerate(privs):
            t = f'pm{k}'
            nxt_scale = self.priv_scale(privs[k + 1][1]) if k + 1 < len(privs) else self.lut_scale(luts[0][1])
            gnext = f'g_pm{k + 1}' if k + 1 < len(privs) else f'g_{luts[0][0]}'
            if k + 1 < len(privs):
                q, m = self.priv_gring(privs[k + 1][1])
            else:
                q, m = self.lut_gring(luts[0][1])
            C.binders.append(f'{{ν_{t} : ErrorPoly {self.nn}}} {{{gnext} : ExactMatrix {q} {self.nn} 1 {m}}}')
            links = {'hb': 'hb', 'F': 'F',
                     'T': f'fun j ↦ reducePoly {self.priv_out_ring(r)} {self.nn} ((({nxt_scale} : Nat) : Int) * ν_{t} * unitPair sp j) • {gnext}'}
            if k == 0:
                links.update({'henc': f'hs_{cell0_tag}', 'μ': f'μ_{cell0_tag}', 'g': f'g_{cell0_tag}', 'B': f'Bs_{cell0_tag}'})
                mu = f'μ_{cell0_tag}'
            else:
                links.update({'henc': f'rel_scale_target r_pm{k - 1}', 'μ': mu, 'g': f'g_pm{k}', 'B': None})
            C.add(self.segs[name], t, links)
            mu = f'({mu} * ν_{t})'
        lt, a = luts[0]
        C.binders.append(f'{{g_{luts[1][0]} : ExactMatrix {self.lut_gring(luts[1][1])[0]} {self.nn} 1 {self.lut_gring(luts[1][1])[1]}}}')
        C.add(self.segs[a['field']], lt, {'hb': 'hb', 'F': 'F', 'hinput': f'rel_scale_target r_pm{len(privs) - 1}',
                                         'μ': mu, 'g': f'g_{lt}', 'B': None,
                                         "Δ'": self.pcast(self.lut_scale(luts[1][1])), 'gq': f'g_{luts[1][0]}'})
        lt2, a2 = luts[1]
        C.add(self.segs[a2['field']], lt2, {'hb': 'hb', 'F': 'F', 'hinput': f'r_{lt}', 'μ': f'ν_{lt}',
                                           'g': f'g_{lt2}', 'B': None})
        return lt2

    def priv_scale(self, r):
        q = mat_type(self.I.wires[r['c'][0]]['type'])[1]
        q2 = mat_type(self.I.wires[wire_of(r['j'][0]['cX']['args'][-1])]['type'])[1]
        return f'({qterm(q)} / {qterm(q2)})'

    def priv_gring(self, r):
        _, q, _, _, m = mat_type(self.I.wires[r['c'][0]]['type'])
        return qterm(q), m

    def priv_out_ring(self, r):
        return qterm(mat_type(self.I.wires[wire_of(r['j'][0]['cX']['args'][-1])]['type'])[1])

    def lut_scale(self, a):
        args = [base(x) for x in a['loop']['args'][10:]]
        q = mat_type(self.I.wires[args[3]]['type'])[1]
        q2 = mat_type(self.I.wires[args[9]]['type'])[1]
        return f'({qterm(q)} / {qterm(q2)})'

    def lut_gring(self, a):
        args = [base(x) for x in a['loop']['args'][10:]]
        _, q, _, _, m = mat_type(self.I.wires[args[3]]['type'])
        return qterm(q), m


def compose(I, roles, segs, scopes, privs_raw, header):
    """One theorem `chain_<cell0>` per PRF chain (the mask chain with the offset chain); returns
    [(name, text, ends)]."""
    O0 = Online(I, roles, segs, scopes)
    loops = O0.loops
    cell0 = [a for a in loops if O0.role(a) == 'prf.cell0']
    privs, rest = [], list(privs_raw)
    outs = lambda r: {wire_of(x['args'][-1]) for jj in r['j'] for x in (jj['cX'], jj['cM0'], jj['cM1'])}
    while rest:
        prev = privs[-1][1] if privs else None
        nxt = [r for r in rest if prev is None and not any(r['c'][0] in outs(s) for s in rest)
               or prev is not None and r['c'][0] in outs(prev)]
        if not nxt:
            break
        privs.append((f'seg_{nxt[0]["both"]["field"]}', nxt[0]))
        rest.remove(nxt[0])
    luts = []
    if privs:
        last = outs(privs[-1][1])
        r0 = [a for a in loops if a['loop']['kind'] == 'seq' and O0.role(a) == 'lut.round.step'
              and any(base(x) in last for x in a['loop']['args'][10:])]
        if r0:
            km = xsem.root_lut_match(I, r0[0])
            kp = {wire_of(x['args'][-1]) for row in km.get('Kp', []) for x in row} | \
                 {wire_of(x['args'][-1]) for x in km.get('Rp', [])}
            r1 = [a for a in loops if a['loop']['kind'] == 'seq' and O0.role(a) == 'lut.round.step'
                  and any(base(x) in kp for x in a['loop']['args'][10:])]
            if r1:
                luts = [(O0.tag(r0[0]), r0[0]), (O0.tag(r1[0]), r1[0])]
    res = []
    for a0 in cell0:
        O = Online(I, roles, segs, scopes)
        O.C.binders.append(header)
        O.C.seen.update(xsem.SHARED)
        nodes = O.chain_nodes(a0)
        offs = [n for n in nodes if O.role(n) == 'lut.member.sampler-offset']
        if offs and privs and luts:
            O._pending_offset = (privs, luts, O.tag(a0), [base(x) for x in offs[0]['loop']['args'][13:19]])
        O.prf_chain(a0)
        name = f'chain_{O.tag(a0)}'
        res.append((name, O.C.theorem(name, ''), O.ends))
    return res
