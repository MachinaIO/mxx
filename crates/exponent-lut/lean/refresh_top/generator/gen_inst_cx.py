#!/usr/bin/env python3
"""Generate `Gen_Inst_Cx.lean` (WP D): the action input row of the final combination.

`GenFinalInst.final_inst` needs `hcx : w_X = maskPart s Kx - reducePoly R ((R / Mx) * x) • gx + e` and
`hx : reducePoly q x = rho q (decodeR j * c X^e)`. The row `w_X` is a `select` (at the instance
index) of the rows of the online source LUT (the root `lut.round.step` whose `seg_<field>` concludes
on those rows). The generator follows the data flow:

* the LUT input: the encrypt stage's input encoding (`EncRoot.input`, its call in
  `Gen_Inst_Stage_encrypt.lean`) of `c X^e` (the plaintext wires found by chasing the row's
  terms), under keys hash-sampled from the linked key wire (`hashPair_of_samples`);
* the helpers: the encrypt `enc.lut-branch-key` loop linked to the helper families;
* the switching ciphertexts: F's `switch_<field>` (`Gen_Wiring.lean`) of the preprocess key table
  linked to the switching families; its coefficients are `src_coef`'s and its Walsh sum on `X^e`
  is `src_walsh`'s `2 a b X^(1 - e)` (`GenInstSource.lean`, automorphisms `1, n + 1`);
* `rel_row_maskPart` / `select_period2` (`GenInstKeys.lean`) give the row.

Certificates (per instance, `interval_cases`): the encrypt plaintext scale is `q / R`, the
table's `a` is `R / Mx`, `2 b ≡ 1 (mod q)`. No wire id or scope number is named.

Usage: gen_inst_cx.py <generated dir> <param dir>
(param dir: Gen_Inst_Stage_online_eval.lean, Gen_Inst_Stage_encrypt.lean, Gen_Final_Inst.lean,
Gen_Stage_Links.lean, Gen_Wiring.lean)
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
import xmodel  # noqa: E402
from gen_inst_keys import St, parse_links, seg_text, qual_on, ON, PRE, ENC  # noqa: E402

WIRE = r'\(?\S*?\.?(w_\d+_\d+) ctx\)?'


def wires_in(t):
    return re.findall(r'\.(w_\d+_\d+) ctx', t)


def qual_enc(t, inst):
    t = re.sub(r'(?<![\w.])(parallel_parallel_generatedRoot_\w+)', ENC + r'.\1', t)
    t = re.sub(r'\(%s\.(w_\d+_\d+) ctx\)' % inst, '(' + ENC + '.' + inst + r'.\1 ectx)', t)
    t = re.sub(r'(?<![\w.(])%s\.(w_\d+_\d+) ctx\b' % inst, '(' + ENC + '.' + inst + r'.\1 ectx)', t)
    t = re.sub(r'(?<![\w.])(modulus_\d+|mod_\d+|sel_\d+)\b', ENC + r'.\1', t)
    return t.replace('F.c', 'FE.c')


def qual_p(t, inst):
    t = re.sub(r'\(%s\.(w_\d+_\d+) ctx\)' % inst, '(' + PRE + '.' + inst + r'.\1 pctx)', t)
    t = re.sub(r'(?<![\w.(])%s\.(w_\d+_\d+) ctx\b' % inst, '(' + PRE + '.' + inst + r'.\1 pctx)', t)
    return re.sub(r'(?<![\w.])(modulus_\d+|mod_\d+|sel_\d+)\b', PRE + r'.\1', t)


def facts_defs(path, inst):
    """wire -> defining term of the facts module's `def inst.w … :=` (one line)."""
    text = open(path).read()
    return {m.group(1): m.group(2).strip() for m in
            re.finditer(r'^def %s\.(w_\d+_\d+) .*?:=\n  (.*)$' % re.escape(inst), text, re.M)}


class Cx:
    def __init__(self, gdir, pdir, roles):
        self.gdir, self.pdir = gdir, pdir
        self.O, self.P, self.E = (St(gdir, M, roles) for M in (ON, PRE, ENC))
        self.links = parse_links(f'{pdir}/Gen_Stage_Links.lean')
        self.hyps = {}

    def link(self, X, w, Y):
        for (M, w2) in sorted(self.links.get((X.M, w), ())):
            if M == Y.M:
                name = f'{"lO" if X is self.O else "lE"}_{w}'
                self.hyps[name] = f'{X.W(w)} = {Y.W(w2)}'
                return name, w2
        raise KeyError(f'no link {X.M}.{w} -> {Y.M}')

    def sample_fact(self, X, fac, w):
        """(fact, blob, components, key wire) of the hash sample producing `w` in stage `X`."""
        for f, (fr, t) in X.fields.items():
            m = re.match(r'^MxxRuntime\.hashSample \(hashModel\) (\(\[[\d, ]*\]\)) (\(\[.*\]\)) \(\(\S+\.(w_\d+_\d+) ctx\)\) \(\S+\.%s ctx\)$' % w, t)
            if m:
                return f'{fac}.c{fr}.{f}', m.group(1), m.group(2), m.group(3)
        return None

    def cx_keys(self, d):
        """The action row's keys are the preprocessing's combination keys `Kx` (`final_inst`'s `hpre`):
        the online source LUT's keys (`keys_<seg>`, `Gen_Inst_Stage_online_eval_KeyChain`) on the hash
        samples of the input keys, against the preprocessing's key loop (`keyloop_*`), selection
        (`sel_*`) and block switch (`bms_*`) of F's `Gen_Inst_Stage_preprocess_Keys`."""
        O, P = self.O, self.P
        kc = f'{self.pdir}/Gen_Inst_Stage_online_eval_KeyChain.lean'
        ps = f'{self.pdir}/Gen_Summands_PreSummand.lean'
        pk = f'{self.pdir}/Gen_Inst_Stage_preprocess_Keys.lean'
        if not all(os.path.exists(x) for x in (kc, ps, pk)):
            return None
        kname = 'keys_' + d['segname'][4:]
        kct = open(kc).read()
        if not re.search(r'^theorem %s ' % kname, kct, re.M):
            return None
        fin = open(f'{self.pdir}/Gen_Final_Inst.lean').read()
        yw = re.search(r'- Kx k\' \* (w_\d+_\d+)\)', fin).group(1)
        m = re.search(r'!\[\(\S+\.(w_\d+_\d+) cP\), \(\S+\.(w_\d+_\d+) cP\)\] k \* \(\S+\.%s ctx\)' % yw, open(ps).read())
        if not m:
            return None
        ta, tb = m.group(1), m.group(2)
        kt = open(pk).read()
        def stmt(name):
            mm = re.search(r'^theorem %s .*?:=' % re.escape(name), kt, re.S | re.M)
            return mm.group(0) if mm else None
        out = dict(kname=kname, ta=ta, tb=tb, sides=[])
        Y = None
        for t in (ta, tb):
            b = stmt('bms_h' + t[2:])
            xm = re.search(r'= blockModSwitchG (\[[\d, ]*\]) \(1\) \(\S+\.(w_\d+_\d+) ctx\) :=$', b)
            xw = xm.group(2)
            sl = stmt('sel_h' + xw[2:])
            ents = re.findall(r'\(\S+\.(w_\d+_\d+) ctx\)', sl.split(' = [', 1)[1])
            e0, e1 = ents[0], ents[1]
            assert all(e == (e0 if k % 2 == 0 else e1) for k, e in enumerate(ents))
            Y = re.match(r'w_(\d+)_\d+', e0).group(1)
            out['sides'].append(dict(t=t, x=xw, e0=e0, e1=e1, n=len(ents)))
        kl = stmt(f'keyloop_h{Y}_3')
        if not kl:
            return None
        ka = re.search(r'keyA \(\S+\.(w_\d+_\d+) ctx\) \(\S+\.(w_\d+_\d+) ctx\) \(\S+\.(w_\d+_\d+) ctx\) \(\S+\.(w_\d+_\d+) ctx\)', kl)
        a00, a10, a01, a11 = ka.groups()
        pkey = re.search(r'hashModel \(\S+\.(w_\d+_\d+) ctx\)', kl).group(1)
        basis2 = re.search(r'= keyBmsG (\[[\d, ]*\])', re.search(r'^theorem %s .*?:= by' % kname, kct, re.S | re.M).group(0)).group(1)
        x = d['ins']
        pairs = [(x[1], a00), (x[4], a01), (x[2], a10), (x[5], a11)]  # (0,0), (0,1), (1,0), (1,1)
        facts = []
        for xo, xp in pairs:
            so, sp_ = self.sample_fact(O, 'FO', xo), self.sample_fact(P, 'FP', xp)
            if not so or not sp_ or so[1:3] != sp_[1:3]:
                return None
            facts.append((so[0], sp_[0]))
        lk = f'lKx_{d["keyw"]}'
        out.update(Y=Y, a=(a00, a10, a01, a11), pkey=pkey, basis2=basis2, facts=facts, lk=lk,
                   hyp=f'{O.W(d["keyw"])} = {P.W(pkey)}')
        return out

    def c1_keys(self, g1):
        """The preprocessing samples equal to column 1 of the gadget encoding's hash pair: for `.u64Le 1`
        and `.u64Le 3`, the encrypt sample (`EncRoot.gadget_reduced`'s `h01`/`h11`) and the preprocessing
        sample with the same blob and components up to the instance table (`.decimal (sel_k i)`)."""
        P, E = self.P, self.E
        pat = (r'^MxxRuntime\.hashSample \(hashModel\) (\(\[[\d, ]*\]\)) \(\[\.decimal \(\((sel_\d+) \(\(Int\.ofNat i_0\)\)\)\), '
               r'(\.bytes \[[\d, ]*\]), \.u64Le \((\d)\)\]\) \(\(\S+\.(w_\d+_\d+) ctx\)\) \(\S+\.(w_\d+_\d+) ctx\)$')
        enc = {}
        for f in re.findall(r'F\.c\d+\.(h\d+_0)', g1):
            m = re.match(pat, E.fields[f][1])
            if m:
                enc[int(m.group(4))] = m
        out = {}
        for f, (fr, t) in P.fields.items():
            m = re.match(pat, t)
            if m and int(m.group(4)) in (1, 3) and int(m.group(4)) in enc:
                e = enc[int(m.group(4))]
                if (m.group(1), m.group(3)) == (e.group(1), e.group(3)):
                    out[int(m.group(4))] = (f'FP.c{fr}.{f}', m.group(6), m.group(5), m.group(2))
        if set(out) != {1, 3}:
            return None
        ekey, esel = enc[1].group(5), enc[1].group(2)
        pkey, psel = out[1][2], out[1][3]
        lk = f'lK_{ekey}'
        self.hyps_c1 = {lk: f'{E.W(ekey)} = {P.W(pkey)}'}
        return dict(lk=lk, esel=esel, psel=psel, fa=out[1][0], fb=out[3][0], wa=out[1][1], wb=out[3][1])

    def gen(self):
        O, P, E = self.O, self.P, self.E
        fin = open(f'{self.pdir}/Gen_Final_Inst.lean').read()
        wx = re.search(r'\(hcx : (w_\d+_\d+) = maskPart', fin).group(1)
        m = re.search(r"\{gx' : ExactMatrix \((.*?)\) (\d+) 1 (\d+)\}", fin)
        R2, N, MX = m.group(1), m.group(2), m.group(3)
        R2 = re.sub(r'\(\(Int\.ofNat i0\)\)', '(Int.ofNat i_0)', R2)
        Q1 = re.search(r'\(hx : reducePoly \((.*?)\) \d+ x', fin).group(1).replace('((Int.ofNat i0))', '(Int.ofNat i_0)')
        R0 = f'({ON}.mod_0 (Int.ofNat i_0))'
        R2, Q1 = f'({R2})', f'({Q1})'
        # the selected rows
        frag, sty = O.fields['h' + wx[2:]]
        rows = wires_in(sty)
        r0, r1 = rows[1], rows[2]
        sel_len = len(rows) - 2
        # the source LUT segment
        segs = open(f'{self.pdir}/Gen_Inst_Stage_online_eval.lean').read()
        seg = None
        for mm in re.finditer(r'^theorem (seg_\S+) ', segs, re.M):
            t = seg_text(segs, mm.group(1)[4:])
            concl = t[:t.index(':= by')]
            if re.search(r'\) !\[\(\S+\.%s ctx\), \(\S+\.%s ctx\)\]' % (r0, r1), concl):
                seg, segname = t, mm.group(1)
                break
        assert seg, 'no source LUT segment'
        hdr = seg[:seg.index(':= by')]
        ins = wires_in(re.search(r'\(hinput : MxxExponentLut\.Rel sm \(inA (.*?)\)\n', hdr).group(1))
        sig = re.search(r'sig \[([^\]]*)\] 2', hdr).group(1)
        lm = re.search(r'lutAG hashModel \(\S+\.(w_\d+_\d+) ctx\) (\S+) (\d+) (\[[\d, ]*\]) (\d+)', hdr)
        keyw, lutm, luttag, lutlev = lm.group(1), lm.group(3), lm.group(4), lm.group(5)
        hm = re.search(r'!\[\(\S+\.(w_\d+_\d+) ctx\) k, \(\S+\.(w_\d+_\d+) ctx\) k\] Bh', hdr)
        hf0, hf1 = hm.group(1), hm.group(2)
        fam = wires_in(re.search(r'fam4 ((?:\(\S+ ctx\) ){4})', hdr).group(1))
        tail = hdr[hdr.rindex(') :'):]
        bnd = tail[tail.rindex(' ctx)]') + len(' ctx)]'):].strip()
        bnd = re.sub(r'\bBB\b', '@BB@', bnd)
        bnd = re.sub(r'\bBh\b', '@Bh@', bnd)
        bnd = re.sub(r'\bB\b', '@B@', bnd)
        outk = wires_in(re.search(r'MxxExponentLut\.Rel sm !\[!\[(.*?)\]\]\n', hdr[hdr.rindex(') :'):]).group(1))
        gm = re.search(r"\{g' : ExactMatrix \(mod_0 \(\(Int\.ofNat i_0\)\)\) \d+ 1 (\d+)\}", hdr)
        # the encrypt input encoding
        lc0, ec0 = self.link(O, ins[0], E)
        lc1, ec1 = self.link(O, ins[3], E)
        lk, ekey = self.link(O, keyw, E)
        enc = open(f'{self.pdir}/Gen_Inst_Stage_encrypt.lean').read()
        call = re.search(r'node_have \S+ := (EncRoot\.input .*?\(c0 := \(%s\.%s ctx\)\).*)$' % (E.inst, ec0), enc, re.M).group(1)
        gm2 = re.search(r'\(%s\.(w_\d+_\d+) ctx\) F\.c(\d+)\.(h\d+_0) F\.c\d+\.h\d+_0 \(c0' % E.inst, call)
        gw = gm2.group(1)
        cut_in = re.search(r'gaussianSample \(.*?\) \((.*?)\) ', E.fields[gm2.group(3)][1]).group(1)
        rh = re.search(r'^theorem roles .*?:= by', enc, re.S | re.M).group(0)
        esec = re.findall(r'(\(h_w_\d+_\d+ : .*?sm\)|\(h_w_\d+_\d+ : .*?sp\))', rh)
        esec_names = [re.match(r'\((h_w_\d+_\d+)', x).group(1) for x in esec]
        edefs = facts_defs(f'{self.gdir}/{ENC}_Facts.lean', E.inst)
        # chase c X^e and the plaintext scale from the row
        todo, seen, cw, ew, pcw = [ec0], set(), None, None, None
        while todo:
            w = todo.pop()
            if w in seen or w not in edefs:
                continue
            seen.add(w)
            t = edefs[w]
            mm = re.match(r'^MxxRuntime\.multiplyMonomial \(\S+\.(w_\d+_\d+) ctx\) \(\S+\.(w_\d+_\d+) ctx\)$', t)
            if mm and edefs.get(mm.group(1), '').startswith('MxxRuntime.liftInteger'):
                cw = wires_in(edefs[mm.group(1)])[0]
                ew = mm.group(2)
                mono = w
            todo += wires_in(t)
        for w, t in edefs.items():
            mm = re.match(r'^MxxRuntime\.matrixMulScalarLeft \(\S+\.(w_\d+_\d+) ctx\) \(\S+\.%s ctx\)$' % mono, t)
            if mm:
                pcw = re.search(r'matrixPolynomial \[\((.*)\)\] :', edefs[mm.group(1)]).group(1)
        pc = qual_enc(pcw, E.inst).replace('((Int.ofNat i_0))', '(Int.ofNat i_0)')
        # encrypt gadget of the input encoding and its rebase
        greb = gfac = None
        for f, (fr, t) in E.fields.items():
            if re.match(r'^MxxRuntime\.centeredRebaseRuns \(\S+ ctx\) \(\S+\.%s ctx\)$' % gw, t):
                greb = (f'FE.c{fr}.{f}', wires_in(t)[0])
        for f, (fr, t) in E.fields.items():
            if re.match(r'^MxxRuntime\.gadgetMatrixRuns backend \S+ \S+ \(\S+\.%s ctx\)$' % greb[1], t):
                gfac = f'FE.c{fr}.{f}'
        # online key samples
        samples = {}
        for f, (fr, t) in O.fields.items():
            mm = re.match(r'^MxxRuntime\.hashSample \(hashModel\) \((\[[\d, ]*\])\) \(\[\.bytes \[[\d, ]*\], \.u64Le \((\d)\)\]\) \(\(\S+\.%s ctx\)\) \(\S+\.(w_\d+_\d+) ctx\)$' % keyw, t)
            if mm and mm.group(3) in ins:
                samples[int(mm.group(2))] = (f'FO.c{fr}.{f}', mm.group(1))
        tag = samples[0][1]
        # helpers
        lh0, eh0 = self.link(O, hf0, E)
        lh1, eh1 = self.link(O, hf1, E)
        ha = E.loop_of(eh0)[0]
        hcall = re.search(r'node_have s_%s := fun i ↦ (.*)$' % ha['field'], enc, re.M).group(1)
        scp = open(f'{self.pdir}/Gen_{ENC}_Scopes.lean').read()
        comp = re.search(r'theorem %s\.sem .*?EncLutBranchKey\.Comp (.*?) backend ' % re.escape(ha['loop']['child']), scp, re.S).group(1)
        cut_h = xmodel.split_top(comp.strip(), ' ')[14]
        hcall = qual_enc(hcall.replace(f'.{ha["field"]} i)', f'.{ha["field"]} k)'), E.inst)
        # switching ciphertexts (F)
        lf = [self.link(O, w, P) for w in fam]
        ta = P.loop_of(lf[0][1])[0]
        wir = open(f'{self.pdir}/Gen_Wiring.lean').read()
        sw = re.search(r'^theorem switch_%s .*?:= by' % ta['field'], wir, re.S | re.M).group(0)
        psec = re.search(r'\(F : \S+ ctx outputs\)\s*(.*?)\s*\{g0 :', sw, re.S).group(1)
        psec_names = re.findall(r'\((\w+) : ', psec)
        cf = re.search(r'\(\(fun \(i : Fin 2\) ↦ (\(\(keyTableG .*?\) 0 0\))\) k \*', sw).group(1)
        ab = re.search(r'keyTableG \(\(\[\((.*?)\)\]\) : List Int\) \(\(\[\((.*?)\)\]\) : List Int\)', cf)
        a = qual_p(ab.group(1), P.inst).replace('((Int.ofNat i_0))', '(Int.ofNat i_0)')
        b = qual_p(ab.group(2), P.inst).replace('((Int.ofNat i_0))', '(Int.ofNat i_0)')
        cf = qual_p(cf, P.inst)
        g0m = re.search(r'\{g0 : ExactMatrix \((.*?)\) \(?\d+\)? 1 \(?(\d+)\)?\}', sw)
        cut_sw = re.search(r'CoeffBound \(eB k j\) (\(.*?\)\.toNat|\d+)\)', sw).group(1)
        gbase = re.search(r'\(hg0 : gadgetMatrixRuns backend \(\((\d+)\) : Int\)', sw).group(1)
        # the gadget encoding row (hc₁): the second row of the final combination's concatenation
        w1 = re.search(r'\(hc₁ : (w_\d+_\d+) = maskPart', fin).group(1)
        l1, e1w = self.link(O, w1, E)
        g1 = None
        for mm2 in re.finditer(r'node_have \S+ := (EncRoot\.gadget_reduced .*?\(c1 := \(%s\.%s ctx\)\).*)$' % (E.inst, e1w), enc, re.M):
            g1 = mm2.group(1)
        gm3 = re.search(r'\(%s\.(w_\d+_\d+) ctx\) F\.c(\d+)\.(h\d+_0) F\.c\d+\.h\d+_0 \(c0' % E.inst, g1)
        grw1 = gm3.group(1)
        grf = [f'FE.c{fr}.{f}' for f, (fr, t) in E.fields.items()
               if re.match(r'^MxxRuntime\.gadgetMatrixRuns backend \S+ \S+ \(\S+\.%s ctx\)$' % grw1, t)][0]
        c1cut = re.search(r'gaussianSample \(.*?\) \((.*?)\) ', E.fields[re.findall(r'F\.c\d+\.(h\d+_0) \(c0', g1)[0]][1]).group(1)
        # the keys of row 1: column 1 of the hash-sampled public matrix (`.u64Le 1`, `.u64Le 3`); the
        # preprocessing samples the same blocks under the same blob/components (its own instance table)
        c1keys = self.c1_keys(g1)
        # the decode modulus
        dl = open(f'{os.path.dirname(self.gdir.rstrip("/"))}/TopDecodeLoops.lean').read() \
            if os.path.exists(f'{os.path.dirname(self.gdir.rstrip("/"))}/TopDecodeLoops.lean') \
            else open(f'{self.pdir}/TopDecodeLoops.lean').read()
        Qd = re.search(r'abbrev Inputs := .*? × ExactMatrix (\d+) \d+ 1 1 ×', dl.replace('Mxx.Primitives.', '')).group(1)
        return dict(g1m=O.j['wires'][w1]['type'].split()[-1].strip(')'), w1=w1, l1=l1, e1w=e1w, g1=qual_enc(g1, E.inst), grw1=grw1, grf=grf, c1cut=c1cut, c1keys=c1keys, bnd=bnd, cut_in=cut_in, cut_h=cut_h, cut_sw=cut_sw, ec0=ec0, ec1=ec1, ekey=ekey, wx=wx, R0=R0, R2=R2, Q1=Q1, N=N, MX=MX, frag=frag, sel_len=sel_len, r0=r0, r1=r1,
                    segname=segname, ins=ins, sig=sig, keyw=keyw, lutm=lutm, luttag=luttag, lutlev=lutlev,
                    hf0=hf0, hf1=hf1, fam=fam, outk=outk, lc=(lc0, lc1, lk), call=qual_enc(call, E.inst),
                    gw=gw, greb=greb, gfac=gfac, esec=[qual_enc(x, E.inst) for x in esec],
                    esec_names=esec_names, cw=cw, ew=ew, pc=pc, samples=samples, tag=tag,
                    lh=(lh0, lh1), hcall=hcall, lf=[x[0] for x in lf], swname=ta['field'],
                    psec=qual_p(psec, P.inst), psec_names=psec_names, cf=cf, a=a, b=b,
                    g0q=qual_p(g0m.group(1), P.inst), gbase=gbase, Qd=Qd, S=O.S)

    def emit(self):
        d = self.gen()
        O, P, E = self.O, self.P, self.E
        Ow = O.W
        N, R0, R2, Q1, MX = d['N'], d['R0'], d['R2'], d['Q1'], d['MX']
        ins = [Ow(w) for w in d['ins']]
        inA = ' '.join(ins)
        c, e = E.W(d['cw']), E.W(d['ew'])
        x = (f"((((2 * {d['b']} : Int)) : ErrorPoly {N}) * Xe (n := {N}) (1 - {e}).toNat * "
             f"unitPair sp ⟨i_0 % 2, Nat.mod_lt _ (by norm_num)⟩)")
        gx = f'(rebasedGadget Backend.backend {d["gbase"]} {MX} {R2} {R0} {N} {MX})'
        j0 = '⟨i_0 % 2, Nat.mod_lt _ (by norm_num)⟩'
        s0 = d['samples']
        sig = f'(sig [{d["sig"]}] 2)'
        hyps = '\n'.join(f'    ({k} : {v})' for k, v in self.hyps.items())
        k00, k01, k10, k11 = (Ow(w) for w in d['outk'])
        lc0, lc1, lk = d['lc']
        lh0, lh1 = d['lh']
        hcall = d['hcall']
        xk = self.cx_keys(d)
        if xk:
            S_ = d['S']
            cxkb = f'\n    ({xk["lk"]} : {xk["hyp"]})'
            cxkc = f' ∧\n      Kx = ![{P.W(xk["ta"])}, {P.W(xk["tb"])}]'
            Pw = P.W
            lines = [f'  case keys =>',
                     f'    have hk : {Ow(d["keyw"])} = {Pw(xk["pkey"])} := {xk["lk"]}',
                     f'    have hA : inA {inA} = keyA {" ".join(Pw(w) for w in xk["a"])} := by',
                     f'      funext a b',
                     f'      fin_cases a <;> fin_cases b']
            for fo, fp in xk['facts']:
                lines.append(f'      · exact hashSample_unique {fo} (by rw [hk]; exact {fp})')
            lines += [f'    have hon := RefreshTopProof.GenInst.OnlineEval.{xk["kname"]} FO rfl',
                      f'    rw [hA, hk, {", ".join(d["lf"])}] at hon',
                      f'    have ht := hon.trans (congrArg (keyBmsG {xk["basis2"]}) (RefreshTopProof.GenInst.PreprocessKeys.keyloop_h{xk["Y"]}_3 FP).symm)']
            kk = [[k00, k01], [k10, k11]]
            proj = [['.1', '.2.1'], ['.2.2.1', '.2.2.2.1']]
            for r_, sd in enumerate(xk['sides']):
                e0, e1 = Pw(sd['e0']), Pw(sd['e1'])
                lines += [f'    have hs{r_} : ∀ j : Nat, j < {sd["n"]} → [{", ".join(Pw(sd["e0"] if k % 2 == 0 else sd["e1"]) for k in range(sd["n"]))}].getD j 0 = ![{e0}, {e1}] ⟨j % 2, Nat.mod_lt _ (by norm_num)⟩ := by',
                          f'      intro j hj; interval_cases j <;> rfl',
                          f'    have h{r_}0 : {kk[r_][0]} = blockModSwitchG {xk["basis2"]} 1 {e0} := congrArg (·{proj[r_][0]}) ht',
                          f'    have h{r_}1 : {kk[r_][1]} = blockModSwitchG {xk["basis2"]} 1 {e1} := congrArg (·{proj[r_][1]}) ht']
            lines += ['    funext k', '    fin_cases k']
            for r_, sd in enumerate(xk['sides']):
                lines += [f'    · show ![{kk[r_][0]}, {kk[r_][1]}] {j0} = {Pw(sd["t"])}',
                          f'      rw [RefreshTopProof.GenInst.PreprocessKeys.bms_h{sd["t"][2:]} FP, RefreshTopProof.GenInst.PreprocessKeys.sel_h{sd["x"][2:]} FP, hs{r_} i_0 (by have := hS.out; omega), h{r_}0, h{r_}1]',
                          f'      exact fin2_map _ _ _ _']
            cxkp = '\n'.join(lines) + '\n'
            cxref = ', ?keys'
            kcimp = 'import Gen_Inst_Stage_online_eval_KeyChain\n'
        else:
            cxkb = cxkc = cxkp = cxref = kcimp = ''
        ck = d['c1keys']
        if ck:
            c1kb = (f'\n    {{pparams : {PRE}.Params}} {{pctx : {PRE}.{P.inst}.Ctx Backend.backend hashModel pparams i_0}} {{pout}}'
                    f'\n    (FP : {PRE}.{P.inst}.Facts pctx pout) ({ck["lk"]} : {self.hyps_c1[ck["lk"]]})')
            c1kc = f' ∧\n      K₁ = ![{P.W(ck["wa"])}, {P.W(ck["wb"])}]'
            c1kp = (f'  refine ⟨_, e, he, hrow.trans (by rw [hx1]), ?_⟩\n'
                    f'  have hsel : {ENC}.{ck["esel"]} (Int.ofNat i_0) = {PRE}.{ck["psel"]} (Int.ofNat i_0) := by\n'
                    f'    first | rfl | (have h := hS.out; interval_cases i_0 <;> rfl)\n'
                    f'  funext k\n  fin_cases k\n'
                    f'  · exact hashPair_entry 0 1 (by rw [{ck["lk"]}, hsel]; exact {ck["fa"]})\n'
                    f'  · exact hashPair_entry 1 1 (by rw [{ck["lk"]}, hsel]; exact {ck["fb"]})\n')
        else:
            c1kb = c1kc = ''
            c1kp = '  exact ⟨_, e, he, hrow.trans (by rw [hx1])⟩\n'
        grw, gw0 = d['greb']
        sec = ' '.join(d['esec'])
        bound = qual_on(d['bnd'].replace('@B@', f"(({d['cut_in']} : Int).toNat)").replace('@Bh@', f"(({d['cut_h']} : Int).toNat)").replace('@BB@', d['cut_sw']))
        g1 = d['g1']
        psec = re.sub(r'\{sp : ErrorPoly \(?\d+\)?\}', '', d['psec'])
        psec = re.sub(r'\(hspsmall : [^)]*\)', '', psec).strip()
        return f'''import GenInstKeys
import GenInstSource
import Gen_Wiring
import Gen_Inst_Stage_online_eval
import Gen_Inst_Stage_encrypt
import Gen_Final_Inst
{kcimp}
/-! Generated by `generator/gen_inst_cx.py` (WP D): the action input row `{d['wx']}` of the final
combination (`hcx`) and the residue of its message (`hx`), from the source LUT `{d['segname']}` on the
encrypted input `c X^e` (see the generator's docstring). -/

set_option maxHeartbeats 64000000
set_option maxRecDepth 262144
set_option linter.unusedVariables false

namespace RefreshTopProof.GenInst.Cx

open RefreshTopProof.GenIface RefreshTopProof.GenIface.BrLut RefreshTopProof.GenIface.BrLutKey
open RefreshTopProof.Gen RefreshTopProof.GenInst RefreshTopProof.GenInst.PreprocessKeys
open Mxx.Primitives MxxRuntime MxxExponentLut
open scoped BigOperators

/-- The encrypt input's plaintext scale is the online LUT input's scale `Q / R`. -/
theorem pc_cert (i_0 : Nat) [h : Fact (i_0 < {d['S']})] :
    {d['pc']} = ((({ON}.modulus_0 / {R0} : Nat)) : Int) := by
  have := h.out; interval_cases i_0 <;> decide

/-- The source table's `a` is the action modulus quotient `R / Mx`. -/
theorem a_cert (i_0 : Nat) [h : Fact (i_0 < {d['S']})] :
    {d['a']} = ((({R0} / {R2} : Nat)) : Int) := by
  have := h.out; interval_cases i_0 <;> decide

/-- `2 b ≡ 1` modulo the CRT prime `q`. -/
theorem b_cert (i_0 : Nat) [h : Fact (i_0 < {d['S']})] :
    (2 * {d['b']}) % (({Q1} : Nat) : Int) = 1 := by
  have := h.out; interval_cases i_0 <;> decide

/-- The action gadget at `Mx` exists (the backend's layout). -/
theorem gx_cert (i_0 : Nat) [hS : Fact (i_0 < {d['S']})] :
    ∃ G : ExactMatrix {R2} {N} 1 {MX}, gadgetMatrixRuns Backend.backend {d['gbase']} {MX} G := by
  have := hS.out
  interval_cases i_0 <;> exact gadget_exists ⟨_, by simp [Backend.backend]; rfl, rfl, rfl, rfl⟩

/-- **The action input row** (`hcx` of `final_inst`): row `i_0 mod 2` of the source LUT on the
encrypted `c X^e` is the mask part of its keys minus `(R / Mx) x` times the rebased gadget. -/
theorem cx_{d['wx']} {{hashModel : HashModel}}
    {{pparams : {PRE}.Params}} {{eparams : {ENC}.Params}} {{oparams : {ON}.Params}}
    {{i_0 : Nat}} [hS : Fact (i_0 < {d['S']})]
    {{pctx : {PRE}.{P.inst}.Ctx Backend.backend hashModel pparams i_0}} {{pout}}
    (FP : {PRE}.{P.inst}.Facts pctx pout)
    {{ectx : {ENC}.{E.inst}.Ctx Backend.backend hashModel eparams i_0}} {{eout}}
    (FE : {ENC}.{E.inst}.Facts ectx eout)
    {{octx : {ON}.{O.inst}.Ctx Backend.backend hashModel oparams i_0}} {{oout}}
    (FO : {ON}.{O.inst}.Facts octx oout)
    {{sm sp : ErrorPoly {N}}} (hsmsmall : polyNorm sm ≤ 1) (hspsmall : polyNorm sp ≤ 1)
    {sec}
    {psec}
    (hc : {c} = 1) (he : {e} = 0 ∨ {e} = 1)
{hyps}{cxkb} :
    ∃ (Kx : Fin 2 → ExactMatrix {R0} {N} 1 {MX}) (ex : ErrorMatrix {N} 1 {MX}), CoeffBound ex {bound} ∧
      {Ow(d['wx'])} = Mxx.ExponentLut.Refresh.maskPart (reducePoly {R0} {N} sm) Kx -
        reducePoly {R0} {N} ((((({R0} / {R2} : Nat)) : Int) : ErrorPoly {N}) * {x}) • {gx} +
        reduceMatrix {R0} {N} 1 {MX} ex{cxkc} := by
  have hn : (0 : Nat) < {N} := by norm_num
  have hμ : message (n := {N}) {c} {e} = Xe (n := {N}) ({e}).toNat := by
    rw [hc]; unfold message
    rcases he with h | h <;> rw [h] <;> simp [Xe]
  have hE := {d['call']}
  have hkeys : inA {inA} = hashPair hashModel {E.W(d['ekey'])} {ON}.modulus_0 {N} {d['lutm']} {d['tag']} [] := by
    rw [← {lk}, hashPair_of_samples (pre := []) {s0[0][0]} {s0[1][0]} {s0[2][0]} {s0[3][0]}]
    funext a b; fin_cases a <;> fin_cases b <;> rfl
  have hrows : inC {inA} = ![{E.W(d['ec0'])}, {E.W(d['ec1'])}] := by
    rw [← {lc0}, ← {lc1}]; rfl
  have hinput : Rel sm (inA {inA})
      (fun j ↦ reducePoly {ON}.modulus_0 {N} ((((({ON}.modulus_0 / {R0} : Nat)) : Int) : ErrorPoly {N}) *
        Xe (n := {N}) ({e}).toNat * unitPair sp j) • {E.W(d['gw'])})
      (inC {inA}) ({d['cut_in']} : Int).toNat := by
    rw [hkeys, hrows, ← hμ, ← pc_cert i_0]
    exact hE
  have hodd : ∀ k, Odd ({sig} k).toNat := fun k ↦ by fin_cases k <;> decide
  have hR0 : 1 < {R0} := RefreshTopProof.GenInst.Certs.Stage_online_eval.mod_0_gt i_0
  have hhelp : ∀ k : Fin 2, Rel sm (lutAG hashModel {Ow(d['keyw'])} {ON}.modulus_0 {d['lutm']} {d['luttag']} {d['lutlev']} ({sig} k))
      (fun j ↦ reducePoly {ON}.modulus_0 {N} ((((({ON}.modulus_0 / {R0} : Nat)) : Int) : ErrorPoly {N}) *
        unitPair (substitution ({sig} k).toNat (hodd k) sm) j) • {E.W(d['gw'])})
      ![{Ow(d['hf0'])} k, {Ow(d['hf1'])} k] ({d['cut_h']} : Int).toNat := by
    intro k
    obtain ⟨Gq, G, hGq, hG, hr⟩ := {hcall}
    obtain rfl := gadgetMatrixRuns_unique hGq {d['gfac']}
    obtain rfl : G = {E.W(d['gw'])} := hG.2.trans {grw}.2.symm
    rw [{lk}, {lh0}, {lh1}, ← pc_cert i_0]
    exact hr
  obtain ⟨g0, hg0⟩ := gx_cert i_0
  obtain ⟨eB, heB, hsw⟩ := RefreshTopProof.Wiring.switch_{d['swname']} FP {' '.join(d['psec_names'])}
    (gadgetOf_runs ⟨g0, hg0⟩) (rebasedGadget_runs (q := {R0}) hR0)
  let Cf : Fin 2 → ExactPoly {R0} {N} := fun i ↦ {d['cf']}
  have hswitch : ∀ k : Fin 2, ∃ e : Fin 2 → ErrorMatrix {N} 1 {MX}, (∀ j, CoeffBound (e j) {d['cut_sw']}) ∧ ∀ j,
      reducePoly {R0} {N} (substitution ({sig} k).toNat (hodd k) sp) •
          fam4 {' '.join(Ow(w) for w in d['fam'])} 0 j k +
          fam4 {' '.join(Ow(w) for w in d['fam'])} 1 j k =
        (Cf k * reducePoly {R0} {N} (1 * unitPair sp j)) • {gx} + reduceMatrix {R0} {N} 1 {MX} (e j) := by
    intro k
    refine ⟨eB k, heB k, fun j ↦ ?_⟩
    rw [{', '.join(d['lf'])}]
    exact hsw k j
  have hmu : ∀ k, polyNorm (substitution ({sig} k).toNat (hodd k) (Xe (n := {N}) ({e}).toNat)) ≤ 1 :=
    fun k ↦ polyNorm_substitution_Xe_gen (by norm_num) _ _
  have hwalsh : ∑ k, reducePoly {R0} {N} (substitution ({sig} k).toNat (hodd k) (Xe (n := {N}) ({e}).toNat)) * Cf k =
      reducePoly {R0} {N} (((2 * {d['a']} * {d['b']} : Int) : ErrorPoly {N}) * Xe (n := {N}) (1 - {e}).toNat) := by
    have hm1 : Xe (n := {N}) ({e}).toNat = message (n := {N}) 1 {e} := by rw [← hc]; exact hμ.symm
    rw [hm1]
    exact src_walsh hR0 hn {sig} hodd rfl rfl _ _ he Cf (fun k ↦ src_coef hR0 hn (hodd k) _ _)
  have hseg := RefreshTopProof.GenInst.OnlineEval.{d['segname']} FO rfl hsmsmall {d['gfac']} {grw} hinput hodd hhelp
    (Δ' := 1) (gq := {gx}) (C := Cf) hswitch hmu hwalsh
  obtain ⟨ex, hex, hrow⟩ := rel_row_maskPart hseg {j0}
  have hsel := select_period2 (hl := fun k hk ↦ by
      simp only [List.length_cons, List.length_nil] at hk
      interval_cases k <;> rfl) (i := i_0) (by simp only [List.length_cons, List.length_nil]; exact hS.out) FO.c{d['frag']}.h{d['wx'][2:]}
  refine ⟨fun k ↦ (![![{k00}, {k01}], ![{k10}, {k11}]] : Fin 2 → Fin 2 → ExactMatrix {R0} {N} 1 {MX}) k {j0}, ex, hex, ?_{cxref}⟩
  · rw [hsel]
    refine hrow.trans ?_
    congr 2
    rw [a_cert i_0]
    push_cast
    ring
{cxkp}
/-- **The gadget-encoding row** (`hc₁` / `hGr` of `final_inst`): row 1 of the encrypt gadget encoding
of `1` is the mask part of its keys minus the gadget. -/
theorem c1_{d['w1']} {{hashModel : HashModel}} {{eparams : {ENC}.Params}} {{oparams : {ON}.Params}}
    {{i_0 : Nat}} [hS : Fact (i_0 < {d['S']})]
    {{ectx : {ENC}.{E.inst}.Ctx Backend.backend hashModel eparams i_0}} {{eout}}
    (FE : {ENC}.{E.inst}.Facts ectx eout)
    {{octx : {ON}.{O.inst}.Ctx Backend.backend hashModel oparams i_0}}
    {{sm : ErrorPoly {N}}} {sec}
    ({d['l1']} : {Ow(d['w1'])} = {E.W(d['e1w'])}){c1kb} :
    ∃ (K₁ : Fin 2 → ExactMatrix {R0} {N} 1 {d['g1m']}) (e₁ : ErrorMatrix {N} 1 {d['g1m']}), CoeffBound e₁ (({d['c1cut']} : Int).toNat) ∧
      {Ow(d['w1'])} = Mxx.ExponentLut.Refresh.maskPart (reducePoly {R0} {N} sm) K₁ - {E.W(d['grw1'])} +
        reduceMatrix {R0} {N} 1 {d['g1m']} e₁{c1kc} := by
  have hE := {g1}
  obtain ⟨e, he, hrow⟩ := rel_row_maskPart hE 1
  have hx1 : ∀ (G : ExactMatrix {R0} {N} 1 {d['g1m']}), reducePoly {R0} {N} (((1 : Int) : ErrorPoly {N}) * unitPair sm 1) • G = G := by
    intro G; simp [unitPair]
  rw [{d['l1']}]
{c1kp}
/-- **The residue of the action input** (`hx` of `final_inst`): `x ≡ r_j · c X^e` modulo `q`. -/
theorem hx_{d['wx']} {{i_0 : Nat}} [hS : Fact (i_0 < {d['S']})] {{inputs : DecodeLoops.Inputs}} {{sp : ErrorPoly {N}}} {{e : Int}}
    (he : e = 0 ∨ e = 1)
    (hS0 : inputs.2.2.2.1 = fun _ _ ↦ reducePoly {d['Qd']} {N} sp)
    (hc2 : inputs.2.2.2.2.1 = 1) (he3 : inputs.2.2.2.2.2.1 = 1 - e) :
    reducePoly {Q1} {N} ((((2 * {d['b']} : Int)) : ErrorPoly {N}) * Xe (n := {N}) (1 - e).toNat *
        unitPair sp {j0}) =
      RefreshTopProof.GenFinal.rho {Q1} (DecodeLoops.decodeR inputs (i_0 % 2) 0 0 *
        multiplyMonomial (liftInteger inputs.2.2.2.2.1) inputs.2.2.2.2.2.1 0 0) := by
  have hn : (0 : Nat) < {N} := by norm_num
  have hq : 1 < {Q1} := RefreshTopProof.GenFinalInst.Certs.q_gt i_0
  have hQ : 1 < {d['Qd']} := by norm_num
  have hqQ : {Q1} ∣ {d['Qd']} := Nat.dvd_of_mod_eq_zero (RefreshTopProof.GenFinalInst.Certs.Q_mod_q i_0)
  have h2b : ((2 * {d['b']} : Int) : ExactPoly {Q1} {N}) = 1 := by
    haveI : Fact (1 < {Q1}) := ⟨hq⟩
    have hz : ((2 * {d['b']} : Int) : ZMod {Q1}) = 1 := by
      rw [← ZMod.intCast_mod, b_cert i_0]; simp
    rw [← map_intCast (algebraMap (ZMod {Q1}) (ExactPoly {Q1} {N})), hz, map_one]
  have hm : message (n := {N}) 1 (1 - e) = Xe (n := {N}) (1 - e).toNat := by
    unfold message
    rw [Int.emod_eq_of_lt (by rcases he with rfl | rfl <;> decide) (by rcases he with rfl | rfl <;> decide)]
    simp [Xe]
  have hmm : multiplyMonomial (liftInteger inputs.2.2.2.2.1 : ExactMatrix {d['Qd']} {N} 1 1) inputs.2.2.2.2.2.1 0 0 =
      reducePoly {d['Qd']} {N} (Xe (n := {N}) (1 - e).toNat) := by
    rw [hc2, he3, multiplyMonomial_liftInteger, hm]
  have hr : DecodeLoops.decodeR inputs (i_0 % 2) 0 0 = reducePoly {d['Qd']} {N} (unitPair sp {j0}) := by
    rcases Nat.mod_two_eq_zero_or_one i_0 with hj | hj
    · have hf : ({j0} : Fin 2) = 0 := Fin.ext hj
      rw [hf]; simp only [hj, DecodeLoops.decodeR, hS0]; rfl
    · have hf : ({j0} : Fin 2) = 1 := Fin.ext hj
      rw [hf]; simp only [hj, DecodeLoops.decodeR]; simp [unitPair]
  rw [hmm, hr, ← map_mul, RefreshTopProof.GenFinal.rho_reduce hn hq hQ hqQ, map_mul, map_mul, map_mul, map_intCast, h2b]
  ring

end RefreshTopProof.GenInst.Cx
'''


if __name__ == '__main__':
    gdir, pdir = sys.argv[1], sys.argv[2]
    roles = json.load(open(sys.argv[3])) if len(sys.argv) > 3 else {}
    txt = Cx(gdir, pdir, roles).emit()
    open(f'{pdir}/Gen_Inst_Cx.lean', 'w').write(txt)
    print(f'{pdir}/Gen_Inst_Cx.lean: {len(re.findall(r"^theorem ", txt, re.M))} theorems')
