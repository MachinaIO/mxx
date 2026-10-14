#!/usr/bin/env python3
"""Generate `Gen_Part.lean` (WP D, L4): the part theorem's chaining of the merged stages.

From `GeneratedClaim.Runs` and an instance `i : Fin S` it obtains the facts records of the
instance bodies of every merged stage (`generatedRoot.facts`, `generatedRoot.h<K>_at`) and the
links between them read from `Claim.lean`'s `Runs` arguments: every instance input is a member of
an external family or a member of a producer stage's output family, i.e. the producer instance's
output (`hout` of the producer root). No wire id is named: the links follow the `Runs` argument
terms and the roots' pack / unpack.

Also writes the fixed names of PART_IFACE.md (`S`, `R`, `Q`, `qOf`, `KOf`, `term`).

Usage: gen_part2.py <generated dir> <out file> <roles.json>
(the roles feed `gen_final.fresh_params`, the fresh error `errE`; needs `<param dir>/Gen_Requirement.lean`)
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
import xmodel  # noqa: E402
import gen_final  # noqa: E402
import xsem  # noqa: E402
from gen_instance import facts_fields  # noqa: E402

#: The part stages, and the stages of the caller's encoding of `σ_0 G` (one instance per chain).
PART = ('Stage_preprocess', 'Stage_encrypt', 'Stage_online_eval')
SIGMA = ('Stage_sigma_preprocess', 'Stage_sigma_encrypt', 'Stage_sigma_online_eval')
SHORT = {'Stage_preprocess': 'p', 'Stage_encrypt': 'e', 'Stage_online_eval': 'o'}


def stage_keys(gdir):
    """The Claim's execution field (`stage_k`) of every part stage."""
    return {c['output'].split('«')[1].rstrip('»'): c['module'] for c in xmodel.parse_claim(gdir)
            if c.get('module') in PART}


def proj(k):
    return '.2' * k + '.1'


def root_loop(scopes):
    r = scopes['generatedRoot'].to_json()
    a = [a for a in r['atoms'] if a.get('loop') and a['loop']['kind'] == 'par'][0]
    return r, a


# Owners of the final combination's remaining inputs (comments of the skeleton's gaps).
OWNER = {
    'hcx': 'D: cx (seg_h18_9 + GenIface.rel_row_maskPart + select_period2)',
    'hx': 'D: the residue of x (source LUT payload hμ/hwalsh)',
    'hg': 'F: Gen_Summands mask_summand', 'hc₁': 'D', 'hβ': 'F: Gen_Summands mask_summand',
    'hb': 'F: Gen_Summands (hint)', 'heb': 'F: Gen_Summands', 'hP': 'F: Gen_Summands (preimage)',
    'hPd': 'F: Gen_Summands', 'hpre': 'F: Gen_Summands (preimage target, member_keys)',
    'hs': 'F: Gen_Summands (smudge)', 'hes': 'F: Gen_Summands', 'hTH': 'F: Gen_Summands',
    'hM': 'F: Gen_Summands', 'hf': 'F: Gen_Summands (fresh)', 'hef': 'F: Gen_Summands',
    'hnoise': 'D: noise certificate (needs closed bounds of the F summands)', 'hYI': 'G: action digits', 'hYb': 'G: action digits bound',
    'hDI': 'G: combination digits', 'hDb': 'G: combination digits bound',
    'hGr': 'G: gadget certificate', 'hgx\'': 'G: gadget certificate', 'hgx': 'G: rebased gadget',
    'heg': 'F: Gen_Summands mask_summand', 'he1': 'D', 'hex': 'D: cx error bound',
}
# The data each gap's proof chooses (witnesses), by the gap that determines it.
DATA = {'Gr': 'hGr', 'gx\'': 'hgx\'', 'gx': 'hgx', 'YI': 'hYI', 'BY': 'hYb', 'DI': 'hDI', 'BD': 'hDb',
        's': 'hg', 'β': 'hβ', 'Kg': 'hg', 'K₁': 'hc₁', 'eg': 'hg', 'e₁': 'hc₁', 'Beg': 'heg', 'Be1': 'he1',
        'x': 'hcx', 'Kx': 'hcx', 'ex': 'hcx', 'Bex': 'hex', 'Bk': 'hb', 'eb': 'hb', 'Beb': 'heb',
        'Pd': 'hP', 'BP': 'hPd', 'Kf': 'hpre', 'Ks': 'hpre', 'ef': 'hf', 'Bf': 'hef', 'Tsm': 'hs',
        'es': 'hs', 'Bs': 'hes', 'M': 'hTH', 'BM': 'hM'}


SH = {'Stage_preprocess': 'P', 'Stage_encrypt': 'E', 'Stage_online_eval': 'O',
      'Stage_sigma_preprocess': 'σP', 'Stage_sigma_encrypt': 'σE', 'Stage_sigma_online_eval': 'σO'}


class LinkTerms:
    """Proof terms of wire identities between the instance bodies from `Gen_Stage_Links`
    (hypotheses `houtX hinstX hinX hrootX` of `GenPartCtx.inst_<Stage>`)."""

    def __init__(self, pdir):
        text = open(f'{pdir}/Gen_Stage_Links.lean').read()
        self.links = {}
        for m in re.finditer(r'^theorem link_(Stage_[a-z_]+?)_(w_\d+_\d+)_(Stage_[a-z_]+?)_(w_\d+_\d+) (.*?):=', text, re.M | re.S):
            A, wa, B, wb, sig = m.groups()
            out = '(houtA :' in sig
            self.links[(A, wa, B, wb)] = out
        self.srcs = set(re.findall(r'^theorem src_(Stage_[a-z_]+?)_(w_\d+_\d+) ', text, re.M))
        # the external term each source wire reads (`src_<Stage>_<w> … : <Stage>.<inst>.<w> cM = <term> :=`)
        self.ext = {(m.group(1), m.group(2)): m.group(3).strip() for m in re.finditer(
            r'^theorem src_(Stage_[a-z_]+?)_(w_\d+_\d+) .*?\n    \S+ cM = (.*?) :=$', text, re.M | re.S)}

    def eq(self, X, wx, Y, wy):
        """A proof of `X.wx cX = Y.wy cY`."""
        for (A, wa, B, wb), out in self.links.items():
            if {(A, wa), (B, wb)} != {(X, wx), (Y, wy)}:
                continue
            a, b = SH[A], SH[B]
            if out:  # `B.wb = A.wa`
                t = f'(StageLinks.link_{A}_{wa}_{B}_{wb} hout{a} hinst{a} hin{b} hroot{b})'
                return t if (B, wb) == (X, wx) else t + '.symm'
            t = f'(StageLinks.link_{A}_{wa}_{B}_{wb} hin{a} hroot{a} hin{b} hroot{b})'
            return t if (A, wa) == (X, wx) else t + '.symm'
        # two wires that read the same external term
        tx, ty = self.ext.get((X, wx)), self.ext.get((Y, wy))
        if tx and tx == ty:
            return f'({self.src(X, wx)}.trans {self.src(Y, wy)}.symm)'
        return None

    def src(self, X, w):
        return f'(StageLinks.src_{X}_{w} hin{SH[X]} hroot{SH[X]})' if (X, w) in self.srcs else None


def gap_haves(gaps, proofs=None):
    """Named facts of the D instance theorems: a proof from `run_facts`, else a gap
    (`have <name> : <type> := by sorry`, owner D)."""
    proofs = proofs or {}
    return [f'  have {g} : {t} := {proofs[g]}' if g in proofs and t else
            f'  have {g} : {t} := by sorry -- D: run fact (ValidExternals / requirement)' if t
            else f'  have {g} := by sorry -- D: missing stage link' for g, t in gaps]


FCTX = {'Stage_preprocess': ('FP', 'cP'), 'Stage_encrypt': ('FE', 'cE'), 'Stage_online_eval': ('FO', 'cO')}


def rebase_secret(gdir, X, w, LT, secrets, sec):
    """`X.w = reduce s` for a centered rebase of a secret input (the stage's facts field
    `centeredRebaseRuns src w` and the stage-link source of `src`)."""
    sc = xmodel.load(gdir, X)
    r = sc['generatedRoot'].to_json()
    inst = [a for a in r['atoms'] if a.get('loop') and a['loop']['kind'] == 'par'][0]['loop']['child']
    fields = facts_fields(open(f'{gdir}/{X}_Facts.lean').read(), inst)
    for f, (fr, t) in fields.items():
        mm = re.match(r'^MxxRuntime\.centeredRebaseRuns \(%s\.(w_\d+_\d+) ctx\) \(%s\.%s ctx\)$' % (inst, inst, w), t)
        if mm and LT.src(X, mm.group(1)):
            F, c = FCTX[X]
            return (f'(by have h := {F}.c{fr}.{f}; rw [{LT.src(X, mm.group(1))}.trans {secrets[sec]}] at h; '
                    f'exact RefreshTopProof.Gen.rebase_small_gen (by norm_num) (by decide) h{sec}small h)')
    return None


def inst_args(text, name, LT, secrets, gdir=None):
    """Named arguments of the instance theorem `name` (parsed from `text`): the facts records,
    the smallness of the secrets, the secret equalities (stage-link sources + ternary samples) and
    the wire links; other explicit hypotheses become named gaps."""
    m = re.search(r'^theorem %s .*?:= by' % re.escape(name), text, re.S | re.M)
    if not m:
        return None, []
    _, binders, _ = xsem.parse_theorem(m.group(0))
    args, gaps = [], []
    for br, names, ty in binders:
        if br != '(':
            continue
        for nm in names:
            if nm in ('FP', 'FE', 'FO'):
                args.append(f'({nm} := {nm})')
            elif nm in ('hsmsmall', 'hspsmall'):
                args.append(f'({nm} := {nm})')
            elif ty and re.search(r'= fun _ _ ↦ reducePoly .* (sm|sp)$', ty.strip()):
                mm = re.match(r'^\((Stage_\w+?)\.\w+\.(w_\d+_\d+) \w+\) =', ty.strip())
                sec = re.search(r'(sm|sp)$', ty.strip()).group(1)
                src = LT.src(mm.group(1), mm.group(2))
                if src:
                    args.append(f'({nm} := {src}.trans {secrets[sec]})')
                else:
                    rb = rebase_secret(gdir, mm.group(1), mm.group(2), LT, secrets, sec) if gdir else None
                    args.append(f'({nm} := {rb})' if rb else f'({nm} := {nm})')
                    if not rb:
                        gaps.append((nm, None))
            elif ty and re.match(r'^\((Stage_\w+?)\.\w+\.(w_\d+_\d+) \w+\) = \((Stage_\w+?)\.\w+\.(w_\d+_\d+) \w+\)$', ty.strip()):
                mm = re.match(r'^\((Stage_\w+?)\.\w+\.(w_\d+_\d+) \w+\) = \((Stage_\w+?)\.\w+\.(w_\d+_\d+) \w+\)$', ty.strip())
                t = LT.eq(*mm.groups())
                args.append(f'({nm} := {t})' if t else f'({nm} := ?{nm})')
                if not t:
                    gaps.append((nm, None))
            else:
                g = f'{name}_{nm}'
                t = ty.replace(' ectx)', ' cE)').replace(' pctx)', ' cP)').replace(' octx)', ' cO)')
                args.append(f'({nm} := {g})')
                gaps.append((g, t))
    return args, gaps


def valid_bounds(gdir):
    """`external.input_k` -> (projection of `hR.1`, per-member?, lo, hi) from `ValidExternals`."""
    text = open(f'{gdir}/Claim.lean').read()
    body = text.split('def ValidExternals', 1)[1].split(':=\n', 1)[1].split('\n\n')[0]
    conj = [c.strip() for c in body.strip().split(' ∧\n')]
    out = {}
    for k, c in enumerate(conj):
        pr = 'hR.1' + '.2' * k + ('.1' if k < len(conj) - 1 else '')
        m = re.match(r'^\(\((-?\d+) : Int\) ≤ external\.(input_\d+) ∧ external\.input_\d+ ≤ \((-?\d+) : Int\)\)$', c)
        if m:
            out[m.group(2)] = (pr, False, int(m.group(1)), int(m.group(3)))
        m = re.match(r'^\(∀ contract_i_0 : Fin \d+, \(\((-?\d+) : Int\) ≤ \(external\.(input_\d+) contract_i_0\)', c)
        if m:
            hi = re.search(r'≤ \((-?\d+) : Int\)\)\)\)$', c)
            out[m.group(2)] = (pr, True, int(m.group(1)), int(hi.group(1)) if hi else None)
    return out


def run_facts(gdir, pdir, LT, ct, decode, req):
    """Proofs of the run facts of the action row: the encrypt plaintext `c = 1`, `e ∈ {0, 1}` (the
    encrypt inputs' `ValidExternals` bounds), the decode coefficient `= 1` and the decode exponent
    `= 1 - e` (the requirement's constant-family pattern `x_d = 1 - x_a`, `x_e 0 = x_a`,
    `x_e n = x_e 0`). Returns {gap name: proof}; a gap whose pattern is not found stays `sorry`."""
    vb = valid_bounds(gdir)
    hc = re.search(r'\(hc : \((Stage_encrypt\.\S+)\.(w_\d+_\d+) ectx\) = 1\)', ct)
    he = re.search(r'\(he : \((Stage_encrypt\.\S+)\.(w_\d+_\d+) ectx\) = 0 ∨', ct)
    links = open(f'{pdir}/Gen_Stage_Links.lean').read()

    def src_ext(w):
        m = re.search(r'^theorem src_Stage_encrypt_%s .*?\n    \S+ cM = \(\(?external\.(input_\d+)\)? ?(i)?\)? :=' % w, links, re.S | re.M)
        return m.group(1) if m else None
    out = {}
    cx = re.search(r'^theorem (cx_\S+) ', ct, re.M).group(1)
    hxn = 'hx_' + cx[3:]
    for gap, mm, goal in ((f'{cx}_hc', hc, 'c'), (f'{cx}_he', he, 'e')):
        x = src_ext(mm.group(2)) if mm else None
        if x and x in vb and vb[x][1]:
            out[gap] = (f'by rw [{LT.src("Stage_encrypt", mm.group(2))}]; have hv := ({vb[x][0]}) i; omega')
    dc = re.match(r'^external\.(input_\d+)$', decode[4])
    if dc and dc.group(1) in vb and not vb[dc.group(1)][1]:
        out[f'{hxn}_hc2'] = f'by have hv := {vb[dc.group(1)][0]}; show external.{dc.group(1)} = 1; omega'
    # the decode exponent: requirement pattern
    de = re.match(r'^external\.(input_\d+)$', decode[5])
    ee = src_ext(he.group(2)) if he else None
    rq = open(f'{pdir}/Gen_Requirement.lean').read()
    fields = dict(re.findall(r'^  (\w+) : (.*)$', rq.split('structure Facts', 1)[1].split('\n\n')[0], re.M))
    xi = {a.replace('(', '').replace(')', '').strip(): f'x{k}' for k, a in enumerate(req)}
    if de and ee and f'external.{de.group(1)}' in xi and f'external.{ee}' in xi:
        xd, xe = xi[f'external.{de.group(1)}'], xi[f'external.{ee}']
        f1 = f3 = f18 = xa = None
        for f, t in fields.items():
            m = re.match(r'^%s = \(1 - (x\d+)\)$' % xd, t)
            if m:
                f1, xa = f, m.group(1)
        for f, t in fields.items():
            if xa and t == f'(famAt {xe} 0) = {xa}':
                f3 = f
            if re.match(r'^∀ (n\d+) < \d+, \(famAt %s \1\) = \(famAt %s \(MxxIR\.selectInt \(Int\.ofNat \1\) \[[0, ]*\]\)\.toNat\)$' % (xe, xe), t):
                f18 = f
        if f1 and f3 and f18:
            out[f'{hxn}_he3'] = (
                f'by\n    rw [{LT.src("Stage_encrypt", he.group(2))}]\n'
                f'    show external.{de.group(1)} = 1 - external.{ee} i\n'
                f'    have h1 := hreq.{f1}\n    have h3 := hreq.{f3}\n    have h18 := hreq.{f18} (i : Nat) i.isLt\n'
                f'    rw [GenInst.selectInt_zeros (by decide), GenIface.famAt_of_lt _ i.isLt] at h18\n'
                f'    simp only [Int.toNat_zero, Fin.eta] at h18\n    omega')
    return out


NOISE_MONO = [
    '/-- **The noise certificate from closed bounds**: the final combination\'s inequality follows from',
    'the same inequality at upper bounds of the summand bounds (the closed tables of `Gen_Summands`). -/',
    'theorem noise_mono {h n m mx Beb BP Beg BD Be1 Bex BY Bf Bs BM Beg\' Bf\' Bs\' BM\' Q : Nat}',
    '    (h1 : Beg ≤ Beg\') (h2 : Bf ≤ Bf\') (h3 : Bs ≤ Bs\') (h4 : BM ≤ BM\')',
    '    (hc : 2 * (h * n * Beb * BP + (m * n * Beg\' * BD + m * n * Be1 * BD + mx * n * Bex * BY) + Bf\' + Bs\' + BM\') < Q) :',
    '    2 * (h * n * Beb * BP + (m * n * Beg * BD + m * n * Be1 * BD + mx * n * Bex * BY) + Bf + Bs + BM) < Q :=',
    '  lt_of_le_of_lt (by gcongr) hc']


def sigma_mask(pdir, gdir, binders, over, ctxa):
    """The mask row `hg` of the final combination: the caller's encoding of `σ_0 G` of the part's
    chain (the σ stages at instance `i / d`, `Gen_Summands_Sigma` mask_summand), its keys
    (`GenInst.KeyEqSigma.mask_keys`) and the stage links that cast it, its keys and its gadget to
    the part's ring (`GenFinal.castModulus_maskRow`); the residue `hβ` of its scalar
    (`GenFinalInst.mask_inst`). Fills `over` (`β`, `Kg`, `eg`, `Beg`, `heg`, `hg`, `hβ`)."""
    sm = open(f'{pdir}/Gen_Summands_Sigma_MaskSummand.lean').read()
    ns = re.search(r'^namespace (\S+)', sm, re.M).group(1)
    st = re.search(r'^theorem mask_summand .*?:= by', sm, re.S | re.M).group(0)
    links = open(f'{pdir}/Gen_Stage_Links.lean').read()
    ctx = open(f'{pdir}/Gen_Part_Ctx.lean').read()
    Sσ = int(re.search(r'^theorem inst_Stage_sigma_preprocess .*?\(i : Fin (\d+)\)', ctx, re.S | re.M).group(1))
    # the part's mask row (final_inst's `hg`) and the σ online output it casts
    hg = [ty for br, names, ty in binders if 'hg' in names][0]
    wg = re.match(r'^(w_\d+_\d+) = maskPart', hg.strip()).group(1)
    wσ = re.search(r'^\s+Stage_sigma_online_eval\.\w+\.(w_\d+_\d+) ctx = maskPart', st, re.M).group(1)
    lk = re.search(r'^theorem (link_Stage_sigma_online_eval_%s_Stage_online_eval_%s) .*?⟨i\.val / (\d+), by omega⟩'
                   % (wσ, wg), links, re.S | re.M)
    if not lk:
        raise SystemExit(f'no stage link of the mask row {wσ} -> {wg}')
    d = lk.group(2)
    # the summand's gadget (`gadgetOf … (<σ modulus> iS) n m`), as a statement about every modulus
    gq = re.search(r'• \((gadgetOf Backend\.backend \(\(\d+\) : Int\) \(\(\d+\) : Int\)) \((Stage_sigma_\w+\.mod_\d+) \(\(Int\.ofNat \(iS : Nat\)\)\)\) (\d+) (\d+)\)', st)
    gnm = re.search(r'open .*?(RefreshTopProof\.GenInst\.Sigma\w+Keys)', sm).group(1)
    motive = f'(fun Q x ↦ x = {gnm}.{gq.group(1)} Q {gq.group(3)} {gq.group(4)})'
    # the keys: the σ preprocessing's rows, linked to the part preprocessing's
    kt = ''.join(open(f'{pdir}/{m}.lean').read() for m in
                 re.findall(r'^import (\S+)', open(f'{pdir}/Gen_Inst_KeyEqSigma.lean').read(), re.M)
                 if os.path.exists(f'{pdir}/{m}.lean'))
    kst = re.search(r'^theorem mask_keys .*?:= by', kt, re.S | re.M).group(0)
    kσ = re.findall(r'\(Stage_sigma_preprocess\.\w+\.(w_\d+_\d+) pctx\)', kst[kst.rindex(') :'):])
    kp, kl = [], []
    for w in kσ:
        m = re.search(r'^theorem (link_Stage_sigma_preprocess_%s_Stage_preprocess_(w_\d+_\d+)) ' % w, links, re.M)
        if not m:
            raise SystemExit(f'no stage link of the mask key {w}')
        kl.append(f'(StageLinks.{m.group(1)} houtσP hinstσP hinP hrootP).symm')
        kp.append(m.group(2))
    pinst = re.search(r'Stage_preprocess\.(parallel_generatedRoot_\d+)\.Ctx', ctx).group(1)
    Kg = '![' + ', '.join(f'(Stage_preprocess.{pinst}.{w} cP)' for w in kp) + ']'
    LT = LinkTerms(pdir)
    ka, kg = inst_args(kt, 'mask_keys', LT, {'sp': 'hspE', 'sm': 'hsmE'}, gdir)
    ka = [a.replace('(FP := FP)', '(FP := FσP)').replace('(FE := FE)', '(FE := FσE)').replace('(FO := FO)', '(FO := FσO)')
          for a in ka]
    kg = [(g, t and t.replace(' cP)', ' cσP)').replace(' cE)', ' cσE)').replace(' cO)', ' cσO)')) for g, t in kg]
    Sp = int(re.search(r'^theorem inst_Stage_preprocess .*?\(i : Fin (\d+)\)', ctx, re.S | re.M).group(1))
    ix = f'⟨(i : Nat) / {d}, hiσ⟩'
    σctx = (ctxa.replace('(FP := FP) (hinP := hinP) (hrootP := hrootP) (houtP := houtP) (hinstP := hinstP)',
                         '(FP := FσP) (hinP := hinσP) (hrootP := hrootσP) (houtP := houtσP) (hinstP := hinstσP)')
            .replace('(FE := FE) (hinE := hinE) (hrootE := hrootE) (houtE := houtE) (hinstE := hinstE)',
                     '(FE := FσE) (hinE := hinσE) (hrootE := hrootσE) (houtE := houtσE) (hinstE := hinstσE)')
            .replace('(F := FO) (hinO := hinO) (hrootO := hrootO) (houtO := houtO) (hinstO := hinstO)',
                     '(F := FσO) (hinO := hinσO) (hrootO := hrootσO) (houtO := houtσO) (hinstO := hinstσO)'))
    hbound = ', hBeg' if re.search(r'CoeffBound eg Beg ∧ Beg ≤', st) else ''
    L = [f'  -- the mask row: the chain\'s encoding of σ_0 G (the σ stages at instance i / {d}), cast to the part\'s ring',
         f'  have hiσ : (i : Nat) / {d} < {Sσ} := by have hi : (i : Nat) < {Sp} := i.isLt; omega',
         f'  haveI : Fact ((i : Nat) / {d} < {Sσ}) := ⟨hiσ⟩',
         f'  obtain ⟨rσP, cσP, FσP, houtσP, hinstσP, hinσP, hrootσP⟩ := GenPartCtx.inst_Stage_sigma_preprocess hR {ix}',
         f'  obtain ⟨rσE, cσE, FσE, houtσE, hinstσE, hinσE, hrootσE⟩ := GenPartCtx.inst_Stage_sigma_encrypt hR {ix}',
         f'  obtain ⟨rσO, cσO, FσO, houtσO, hinstσO, hinσO, hrootσO⟩ := GenPartCtx.inst_Stage_sigma_online_eval hR {ix}',
         f'  obtain ⟨zN, Kσ, eg, Beg, heg{hbound}, hgσ, hz, hσ0, hKσ⟩ := {ns}.mask_summand (iS := {ix}) {σctx}']
    if hbound:  # the closed bound at the chain index (not at the `Fin` term, which reads `hiσ`)
        L.append(f'  replace hBeg : Beg ≤ {ns}.bound_mask_Beg ((i : Nat) / {d}) := hBeg')
    L += gap_haves(kg)
    L += [f'  have hKσK := GenInst.KeyEqSigma.mask_keys ' + ' '.join(ka),
          '  rw [hKσK] at hKσ',
          '  subst hKσ',
          f'  have hg := (StageLinks.{lk.group(1)} houtσO hinstσO hinO hrootO).trans',
          f'    (RefreshTopProof.GenFinal.castModulus_maskRow _ (K\' := {Kg})',
          '      (fun k ↦ by fin_cases k <;> [exact ' + '; exact '.join(kl) + '])',
          f'      ((Mxx.Primitives.castModulus_transport _ {motive} rfl).trans ({gnm}.gadgetOf_eq {over["hGr"]})) hgσ)',
          '  have hβ := GenFinalInst.mask_inst (i : Nat) (inputs := decodeInputs external execution)',
          '    (hσ := fun o ho ↦ hσ0 o ho) hz rfl']
    over.update({'β': None, 'Kg': Kg, 'eg': 'eg', 'Beg': None, 'heg': 'heg', 'hg': 'hg', 'hβ': 'hβ'})
    over['__keq__'] = over.get('__keq__', set()) | {'Kg'}
    over['__novar__'] = over.get('__novar__', set()) | {'Kg'}
    return L


def summand_wiring(pdir, gdir, binders, fields, inst, over):
    """Lines that obtain F's summand facts (Gen_Summands_*: hint, preimage, keys, smudge, mask, fresh,
    pre) with the shared context arguments, the digit link `pre_link` (hD/hY of `pre_summand`) and
    the noise certificate; fills `over` with final_inst's named arguments."""
    M = 'Stage_online_eval'
    ctxa = ('(hv := hR.1) (hreq := hreq) (hsm := hsmsmall) (hsp := hspsmall) (hsmE := hsmE) (hspE := hspE) '
            '(FP := FP) (hinP := hinP) (hrootP := hrootP) (houtP := houtP) (hinstP := hinstP) '
            '(FE := FE) (hinE := hinE) (hrootE := hrootE) (houtE := houtE) (hinstE := hinstE) '
            '(F := FO) (hinO := hinO) (hrootO := hrootO) (houtO := houtO) (hinstO := hinstO)')
    NS = 'RefreshTopProof.GenInst.OnlineEval'
    text = {k: open(f'{pdir}/Gen_Summands_{k}Summand.lean').read() for k in
            ('Hint', 'Preimage', 'Keys', 'Smudge', 'Mask', 'Fresh', 'Pre')
            if os.path.exists(f'{pdir}/Gen_Summands_{k}Summand.lean')}
    L = ['  -- the summands (F, Gen_Summands)']
    if 'Hint' in text:
        L.append(f'  obtain ⟨eb, heb, hb⟩ := {NS}.hint_summand {ctxa}')
        bk = re.search(r'\(fun k ↦ rowOf \((Stage_encrypt\.\S+) cE\) k\)', text['Hint'])
        over.update({'Bk': f'(fun k ↦ rowOf ({bk.group(1)} cE) k)', 'eb': 'eb', 'Beb': None, 'heb': 'heb', 'hb': 'hb'})
    if 'Preimage' in text:
        L.append(f'  obtain ⟨Pd, hP, hPd⟩ := {NS}.preimage_summand {ctxa}')
        over.update({'Pd': 'Pd', 'BP': None, 'hP': 'hP', 'hPd': 'hPd'})
    if 'Keys' in text:
        L.append(f'  obtain ⟨hks1, hks3⟩ := {NS}.keys_summand {ctxa}')
    if 'Smudge' in text:
        L.append(f'  obtain ⟨Bs, BM, Tsm, es, M, hes, hBs, hs, hTH, hM, hBM⟩ := {NS}.smudge_summand {ctxa}')
        L.append('  rw [hks1, hks3] at hs')
        ks = re.search(r'!\[\(parallel_generatedRoot_\d+\.(w_\d+_\d+) ctx\), \(parallel_generatedRoot_\d+\.(w_\d+_\d+) ctx\)\] - Tsm', text['Smudge'])
        kc = text['Keys'][text['Keys'].index('(hinstO'):]
        kp = re.findall(r'= (Stage_preprocess\.\S+\.w_\d+_\d+) cP', kc)
        over.update({'Tsm': 'Tsm', 'es': 'es', 'Bs': None, 'hs': 'hs', 'hes': 'hes', 'M': 'M', 'BM': None,
                     'hTH': 'hTH', 'hM': 'hM', 'Ks': f'![{kp[0]} cP, {kp[1]} cP]'})
    if os.path.exists(f'{pdir}/Gen_Summands_Sigma_MaskSummand.lean'):
        L += sigma_mask(pdir, gdir, binders, over, ctxa)
    if 'Fresh' in text:
        kfE = ', hKfE' if re.search(r'∧\n      Kf = fun k _ l ↦', text['Fresh']) else ''
        L.append(f'  obtain ⟨Kf, ef, Bf, hef, hBf, hf{kfE}⟩ := {NS}.fresh_summand {ctxa}')
        over.update({'Kf': 'Kf', 'ef': 'ef', 'Bf': None, 'hef': 'hef', 'hf': 'hf'})
    if 'Pre' in text:
        # D, Y of the online combination are the preprocessing ones (GenFinalInst.pre_link)
        fin = open(f'{pdir}/Gen_Final_Inst.lean').read()
        m = re.search(r'^theorem pre_link .*?:\n    w_\S+ = p_w_\S+ ∧ w_\S+ = p_w_\S+ :=', fin, re.S | re.M)
        hdr = m.group(0)[:m.group(0).rindex(' :')] + ' : True := by'
        _, pb, _ = xsem.parse_theorem(hdr)
        sc = xmodel.load(gdir, 'Stage_preprocess')
        rp = sc['generatedRoot'].to_json()
        pinst = [a for a in rp['atoms'] if a.get('loop') and a['loop']['kind'] == 'par'][0]['loop']['child']
        pf = facts_fields(open(f'{gdir}/Stage_preprocess_Facts.lean').read(), pinst)
        args, hp_fact = [], None
        for br, names, ty in pb:
            for nm in names:
                if nm in ('i0', 'model', 'inputs'):
                    continue
                if nm == 'j':
                    continue
                if nm.startswith('p_w_'):
                    args.append(f'({nm} := (Stage_preprocess.{pinst}.{nm[2:]} cP))')
                elif nm.startswith('p_h'):
                    args.append(f'({nm} := FP.c{pf[nm[2:]][0]}.{nm[2:]})')
                elif nm.startswith('w_'):
                    args.append(f'({nm} := ({M}.{inst}.{nm} cO))')
                elif nm == 'hhash':
                    args.append('(hhash := hhash)')
                elif nm == 'hhash_p':
                    hw = re.match(r'^p_(w_\d+_\d+) = ', ty).group(1)
                    for f, (fr, t) in pf.items():
                        mm = re.search(r'sliceMatrix \(%s\.(w_\d+_\d+) ctx\) .* \(%s\.%s ctx\)$' % (pinst, pinst, hw), t)
                        if mm:
                            hp_fact = (f'FP.c{fr}.{f}', mm.group(1))
                    args.append('(hhash_p := hhash_p)')
                elif re.match(r'^h\d+_\d+$', nm) and nm in fields:
                    args.append(f'({nm} := FO.c{fields[nm][0]}.{nm})')
                else:
                    args.append(f'({nm} := ?pre_link_{nm})')
        L.append(f'  have hhash_p := GenFinalInst.slice_pre (i : Nat) (inputs := decodeInputs external execution) '
                 f'{hp_fact[0]} (StageLinks.src_Stage_preprocess_{hp_fact[1]} hinP hrootP)')
        L.append(f'  obtain ⟨hD, hY⟩ := GenFinalInst.pre_link (i : Nat) (inputs := decodeInputs external execution) ((i : Nat) % R) ' + ' '.join(args))
        L.append(f'  have hpreP := {NS}.pre_summand {ctxa} hD hY')
        # the online keys are the preprocessing's (the combination / fresh key equalities)
        kp = re.findall(r'!\[\((Stage_preprocess\.\S+) cP\), \((Stage_preprocess\.\S+) cP\)\] k', text['Pre'])
        names = ['Kg', 'K₁', 'Kx', 'Kf']
        done = over.get('__keq__', set())
        hn = {'Kg': 'hKg', 'K₁': 'hK₁', 'Kx': 'hKx', 'Kf': 'hKf'}
        ke = over.get('__keyeq__', {})
        kE = {'Kg': 'hKgE' if 'hKgE' in ''.join(L) else None, 'Kf': 'hKfE' if 'hKfE' in ''.join(L) else None}
        for nm, (a, b) in zip(names, kp[:4]):
            if nm in done or nm in over.get('__novar__', set()):
                continue
            if nm in ke and kE.get(nm):
                L.append(f'  have {hn[nm]} : {nm} = ![({a} cP), ({b} cP)] := {kE[nm]}.trans {ke[nm]}')
            else:
                L.append(f'  have {hn[nm]} : {nm} = ![({a} cP), ({b} cP)] := by sorry -- D: online keys = preprocessing keys')
        L.append('  subst ' + ' '.join(hn[nm] for nm in names[:len(kp[:4])] if nm not in over.get('__novar__', set())))
        for nm, (a, b) in zip(names, kp[:4]):
            over[nm] = f'![({a} cP), ({b} cP)]'
        over.update({'hpre': 'hpreP'})
    if 'hgσ' in ''.join(L) and all(k in text for k in ('Fresh', 'Smudge')):
        # the noise certificate: per instance, at the closed bound tables
        over['hnoise'] = '(noise_mono hBeg hBf hBs hBM ?hnoise)'
        over['__cases__'] = ['  case hnoise =>', '    have hlt := i.isLt',
                             '    generalize (i : Nat) = I at hlt ⊢', '    interval_cases I <;> decide']
    return L


def part_skeleton(pdir, gdir, jterm, decode=None, req=None):
    """The body of `part`: the instance contexts (Gen_Part_Ctx), the secrets (ternary samples of
    the payload / mask stages), the decode term as the online output, the D facts (member keys,
    action row `hcx`, its residue `hx`) instantiated with stage-link terms, and
    `GenFinalInst.final_inst` with every input either a facts field, a D fact or a named gap
    (`case … => sorry`, owner in the comment)."""
    path = f'{pdir}/Gen_Final_Inst.lean'
    if not os.path.exists(path):
        return ['  sorry']
    text = open(path).read()
    m = re.search(r'^theorem final_inst .*?:= by', text, re.S | re.M)
    hdr = '\n'.join(l for l in m.group(0).split('\n') if not l.strip().startswith('--'))
    _, binders, _ = xsem.parse_theorem(hdr)
    M = 'Stage_online_eval'
    sc = xmodel.load(gdir, M)
    r, a = root_loop(sc)
    inst = a['loop']['child']
    fields = facts_fields(open(f'{gdir}/{M}_Facts.lean').read(), inst)
    wires = sc[inst].to_json()['wires']
    W = lambda w: f'({M}.{inst}.{w} cO)'
    LT = LinkTerms(pdir) if os.path.exists(f'{pdir}/Gen_Stage_Links.lean') else None
    secrets = {'sp': 'hspE', 'sm': 'hsmE'}
    # D facts
    pre, gaps, over = [], [], {}
    calls = xmodel.parse_claim(gdir)
    for k, c in enumerate(calls):
        if c.get('module') in ('Stage_payload', 'Stage_mask'):
            v = 'sp' if c['module'] == 'Stage_payload' else 'sm'
            pr = 'hR' + '.2' * k + ('.1' if k < len(calls) - 1 else '')
            pre += [f'  obtain ⟨w{v}, hw{v}, hout{v}⟩ := {pr}',
                    f'  obtain ⟨{v}, h{v}E0, h{v}small⟩ := ternary_poly hw{v}',
                    f'  have h{v}E := hout{v}.trans h{v}E0']
    kpath = f'{pdir}/Gen_Inst_Keys.lean'
    if LT and os.path.exists(kpath):
        kt = open(kpath).read()
        for nm in re.findall(r'^theorem (member_keys_\S+) ', kt, re.M):
            ka, kg = inst_args(kt, nm, LT, secrets, gdir)
            pre.append(f'  -- the online smudge members\' keys are the preprocessing\'s (F\'s `hkeys`)')
            pre += gap_haves(kg)
            pre.append(f'  have hkeys_{nm[12:]} := GenInst.Keys.{nm} ' + ' '.join(ka))
    # the online combination keys are the preprocessing's (Gen_Inst_KeyEq: mask carrier, fresh row)
    epath = f'{pdir}/Gen_Inst_KeyEq.lean'
    if LT and os.path.exists(epath):
        et = ''.join(open(f'{pdir}/{m}.lean').read() for m in re.findall(r'^import (\S+)', open(epath).read(), re.M))
        for nm, kn in (('mask_keys', 'Kg'), ('fresh_keys', 'Kf')):
            if re.search(r'^theorem %s ' % nm, et, re.M):
                ka, kg = inst_args(et, nm, LT, secrets, gdir)
                pre += gap_haves(kg)
                pre.append(f'  have h{kn}K := GenInst.KeyEq.{nm} ' + ' '.join(ka))
                over.setdefault('__keyeq__', {})[kn] = f'h{kn}K'
    def GRFACT(grw):
        w = grw.split('.')[-1]
        sc2 = xmodel.load(gdir, 'Stage_encrypt')
        r2 = sc2['generatedRoot'].to_json()
        inst2 = [a for a in r2['atoms'] if a.get('loop') and a['loop']['kind'] == 'par'][0]['loop']['child']
        f2 = facts_fields(open(f'{gdir}/Stage_encrypt_Facts.lean').read(), inst2)
        for f, (fr, t) in f2.items():
            if re.match(r'^MxxRuntime\.gadgetMatrixRuns backend \S+ \S+ \(\S+\.%s ctx\)$' % w, t):
                return f'FE.c{fr}.{f}'
        return None
    # the compact digit facts of final_inst (`w = reduceMatrix … V`, `CoeffBound V B`)
    hb = {}
    for br, names, ty in binders:
        if br == '(' and ty:
            mm = re.match(r'^(w_\d+_\d+) = reduceMatrix _ .* (\w+)$', ty.strip())
            if mm:
                hb[mm.group(2)] = (names[0], mm.group(1))
    for br, names, ty in binders:
        if br == '(' and ty:
            mm = re.match(r'^CoeffBound (\w+) (\w+)$', ty.strip())
            if mm and mm.group(1) in hb:
                hn_, w = hb[mm.group(1)]
                cf = [f'FO.c{fr}.{f}' for f, (fr, t) in fields.items()
                      if re.match(r'^MxxRuntime\.compactCenteredRebaseRuns \S+ \(\S+ ctx\) \(\S+\.%s ctx\)$' % w, t)]
                if cf:
                    V = mm.group(1)
                    pre.append(f'  obtain ⟨{V}, {hn_}, {names[0]}⟩ := {cf[0]}.2.2.2')
                    over.update({V: V, hn_: hn_, names[0]: names[0], mm.group(2): None})
    cpath = f'{pdir}/Gen_Inst_Cx.lean'
    if LT and os.path.exists(cpath):
        ct = open(cpath).read()
        cx = re.search(r'^theorem (cx_\S+) ', ct, re.M).group(1)
        ca, cg = inst_args(ct, cx, LT, secrets, gdir)
        pre.append('  -- the action input row and its residue (hcx, hx)')
        rf = run_facts(gdir, pdir, LT, ct, decode, req) if decode else {}
        pre += gap_haves(cg, rf)
        cxk = re.search(r'^theorem %s .*?∧\n      Kx = !\[' % re.escape(cx), ct, re.S | re.M)
        cxk = cxk and ':= by' not in cxk.group(0)
        pre.append(f'  obtain ⟨Kx, ex, hex, hcx{", hKx" if cxk else ""}⟩ := GenInst.Cx.{cx} ' + ' '.join(ca))
        if cxk:
            over['__keq__'] = over.get('__keq__', set()) | {'Kx'}
        c1 = re.search(r'^theorem (c1_\S+) ', ct, re.M)
        if c1:
            c1a, c1g = inst_args(ct, c1.group(1), LT, secrets, gdir)
            pre += gap_haves(c1g, rf)
            c1k = re.search(r'^theorem %s .*?∧\n      K₁ = !\[' % re.escape(c1.group(1)), ct, re.S | re.M)
            c1k = c1k and ':= by' not in c1k.group(0)
            pre.append(f'  obtain ⟨K₁, e₁, he1, hc₁{", hK₁" if c1k else ""}⟩ := GenInst.Cx.{c1.group(1)} (octx := cO) ' + ' '.join(c1a))
            if c1k:
                over['__keq__'] = over.get('__keq__', set()) | {'K₁'}
            grw = re.search(r'K₁ - \((Stage_encrypt\.\S+) ectx\)', ct).group(1)
            grf = re.search(r'gadgetMatrixRuns backend \S+ \S+ \(\S+\.(w_\d+_\d+) ctx\)', '')
            over.update({'Gr': f'({grw} cE)', 'K₁': 'K₁', 'e₁': 'e₁', 'Be1': None, 'he1': 'he1', 'hc₁': 'hc₁',
                         'hGr': GRFACT(grw)})
        hxn = 'hx_' + cx[3:]
        ew = re.search(r'\(he : (\(Stage_encrypt\.\S+ )ectx\) = 0 ∨', ct).group(1) + 'cE)'
        pre += gap_haves([(f'{hxn}_hc2', '(decodeInputs external execution).2.2.2.2.1 = 1'),
                          (f'{hxn}_he3', f'(decodeInputs external execution).2.2.2.2.2.1 = 1 - {ew}')], rf)
        hen = [g for g, _ in cg if g.endswith('_he')]
        pre.append(f'  have hx := GenInst.Cx.{hxn} (i_0 := (i : Nat)) (inputs := decodeInputs external execution) '
                   f'(sp := sp) (e := {ew}) {hen[0] if hen else "?he"} hspE {hxn}_hc2 {hxn}_he3')
        over.update({'Kx': 'Kx', 'ex': 'ex', 'Bex': None, 'hex': 'hex', 'hcx': 'hcx', 'hx': 'hx',
                     'x': None, 'gx': None,
                     's': f'(reducePoly ({M}.mod_0 (Int.ofNat (i : Nat))) _ sm)',
                     'hgx': f'(RefreshTopProof.GenInst.PreprocessKeys.rebasedGadget_runs (RefreshTopProof.GenInst.Certs.{M}.mod_0_gt (i : Nat)))',
                     "gx'": None,
                     "hgx'": f'(RefreshTopProof.GenInst.PreprocessKeys.gadgetOf_runs (GenInst.Cx.gx_cert (i : Nat)))'})
    post = []
    spath = f'{pdir}/Gen_Summands.lean'
    if os.path.exists(spath):
        post += summand_wiring(pdir, gdir, binders, fields, inst, over)
    args, holes = [], []
    hash_fact = None
    for br, names, ty in binders:
        for nm in names:
            if nm in ('i0', 'j', 'model', 'inputs', 'E', 'G', 'hG'):
                continue
            if nm.startswith('__'):
                continue
            if nm in over:
                if over[nm] is not None:
                    args.append(f'({nm} := {over[nm]})')
                continue
            if br == '{' and nm in wires:
                args.append(f'({nm} := {W(nm)})')
            elif br == '(' and nm in fields and re.match(r'^h\d+_\d+$', nm):
                args.append(f'({nm} := FO.c{fields[nm][0]}.{nm})')
            elif nm == 'hhash':
                hw = re.match(r'^(w_\d+_\d+) = ', ty).group(1)
                for f, (fr, t) in fields.items():
                    mm = re.search(r'sliceMatrix \(%s\.(w_\d+_\d+) ctx\) .* \(%s\.%s ctx\)$' % (inst, inst, hw), t)
                    if mm:
                        hash_fact = (f'FO.c{fr}.{f}', mm.group(1))
                args.append(f'(hhash := hhash)')
            else:
                args.append(f'({nm} := ?{nm})')
                holes.append(nm)
    L = ['  haveI : Fact ((i : Nat) < S) := ⟨i.isLt⟩',
         '  obtain ⟨rP, cP, FP, houtP, hinstP, hinP, hrootP⟩ := GenPartCtx.inst_Stage_preprocess hR i',
         '  obtain ⟨rE, cE, FE, houtE, hinstE, hinE, hrootE⟩ := GenPartCtx.inst_Stage_encrypt hR i',
         '  obtain ⟨rO, cO, FO, houtO, hinstO, hinO, hrootO⟩ := GenPartCtx.inst_Stage_online_eval hR i',
         '  -- the secrets (ternary samples)'] + pre + [
         '  -- the decode stage reads the online instance output',
         f'  have hterm : term execution i = ({M}.{inst}.outs cO){proj(jterm)} := by',
         f'    rw [term, houtO]; exact congrArg (fun x ↦ x{proj(jterm)}) hinstO',
         '  rw [hterm]']
    if hash_fact:
        L += ['  -- the combination\'s A\': the instance\'s column block of the decode stage\'s A\' (stage link source)',
              f'  have hhash := GenFinalInst.slice_online (i : Nat) (inputs := decodeInputs external execution) '
              f'{hash_fact[0]} (StageLinks.src_{M}_{hash_fact[1]} hinO hrootO)']
    L += post
    L.append('  refine GenFinalInst.final_inst (i : Nat) (inputs := decodeInputs external execution)')
    L.append('    (E := errE external) ((i : Nat) % R) hG')
    for k in range(0, len(args), 4):
        L.append('    ' + ' '.join(args[k:k + 4]))
    L += over.get('__cases__', [])
    for h in holes:
        own = OWNER.get(h) or (f'witness of {DATA[h]} ({OWNER.get(DATA[h], "?")})' if h in DATA else '?')
        L.append(f'  case {h} => sorry -- {own}')
    return L


def main():
    gdir, out, roles = sys.argv[1], sys.argv[2], sys.argv[3]
    claim = {c['module']: c for c in xmodel.parse_claim(gdir) if c.get('module')}
    STAGES = stage_keys(gdir)
    onkey = next(k for k, M in STAGES.items() if M == 'Stage_online_eval')
    info = {}
    for st, M in STAGES.items():
        sc = xmodel.load(gdir, M)
        r, a = root_loop(sc)
        L = a['loop']
        inst = L['child']
        ij = sc[inst].to_json()
        ins = [w for w, v in ij['wires'].items() if v['src'] == 'input']
        info[M] = dict(root=r, field=a['field'], frag=a['frag'], inst=inst, S=L['count'], args=L['args'],
                       ins=ins, iw=ij['wires'], params=claim[M]['params'])
    S = info['Stage_online_eval']['S']
    links = []
    for M in STAGES.values():
        I = info[M]
        r = I['root']
        c = SHORT[M]
        for k, (w, arg) in enumerate(zip(I['ins'], I['args'])):
            mm = re.match(r'^\(?(w_\d+_\d+)( i)?\)?$', arg.strip())
            if not mm:
                continue
            rw, per = mm.group(1), bool(mm.group(2))
            src = r['wires'][rw]
            if src['src'] != 'input':
                continue
            idx = src['term'].split('.')[1:].count('2')
            T = claim[M]['args'][idx]
            lhs = f'c{c}.' + I['iw'][w]['term']
            m2 = re.match(r'^execution\.«(stage_\d+)»((?:\.2)*)\.1$', T)
            if m2 and m2.group(1) in STAGES:
                P = STAGES[m2.group(1)]
                j = m2.group(2).count('.2')
                if per:
                    links.append((f'{lhs} = o{SHORT[P]}{proj(j)}', P))
            elif T.startswith('external.') and per:
                links.append((f'{lhs} = {T} i', None))
            elif T.startswith('external.'):
                links.append((f'{lhs} = {T}', None))
    on = info['Stage_online_eval']
    decode = claim['Stage_decode']['args']
    m3 = re.match(r'^execution\.«%s»((?:\.2)*)\.1$' % onkey, decode[0])
    jterm = m3.group(1).count('.2')
    req = claim['Requirement_0']['args']
    dl = open(f'{os.path.dirname(gdir.rstrip("/"))}/TopDecodeLoops.lean').read()
    hterm = re.search(r'hterm : ∀ i : Fin (\d+), TermOf \(qN \(i / (\d+)\)\)', dl)
    Rn = int(hterm.group(2))
    gty = re.search(r'\{G : (ExactMatrix \S+ \S+ 1 \S+)\} \(hG : (gadgetMatrixRuns backend \S+ \S+) G\)', dl)
    qd = re.search(r'\(hE : ∀ j c i, 4 \* \|\(E j 0 c\)\.coeff i\| \+ 4 < \((\d+) : Int\)\)', dl).group(1)
    ety = re.search(r'\(E : (Nat → ErrorMatrix \S+ 1 \S+)\)', dl).group(1)
    tabs = re.search(r'online `(sel_\d+) i0 = decode_KN', dl)
    # the online CRT-term role: the ring of the rebased residue and the basis element (gen_decode_loops' tables)
    qO = re.search(r'CRT primes, read from the online stage\'s instance table `(mod_\d+)`', dl)
    kO = re.search(r'CRT basis, read from the online stage\'s instance table `(sel_\d+)`', dl)
    pdir = os.path.dirname(gdir.rstrip('/'))
    fp = gen_final.fresh_params(gdir, roles, f'{pdir}/Gen_Requirement.lean', gen_final.recognize(gdir))
    err_body, err_proof = gen_final.errE_lean(fp, ety, qd)
    L = ['import Claim', 'import Gen_Requirement', 'import TopDecodeLoops', 'import GenFinal']
    if os.path.exists(f'{pdir}/Gen_Final_Inst.lean'):
        L += ['import Gen_Final_Inst', 'import Gen_Part_Ctx']
        for mod in ('Gen_Inst_Keys', 'Gen_Inst_Cx', 'Gen_Summands', 'Gen_Inst_KeyEq', 'Gen_Summands_Sigma',
                    'Gen_Inst_KeyEqSigma'):
            if os.path.exists(f'{pdir}/{mod}.lean'):
                L.append(f'import {mod}')
    for M in STAGES.values():
        L.append(f'import {M}_Facts')
    L += ['', '/-! Generated by `generator/gen_part2.py`: the part theorem of the instance pipeline',
          'at a symbolic instance: every CRT term the decode stage reads is the term of the',
          'value of its component. -/', '', 'set_option maxHeartbeats 64000000',
          'set_option maxRecDepth 65536', '', 'namespace RefreshTopProof.GenPart', '',
          'open Mxx.Primitives MxxRuntime MxxExponentLut RefreshTopProof GenDecode', '',
          f'/-- The instances of the merged stages. -/\nabbrev S : Nat := {S}', '',
          f'/-- Components per CRT prime: instance `i = t R + j`. -/\nabbrev R : Nat := {Rn}', '',
          f'/-- The decode modulus. -/\nabbrev Q : Nat := {qd}', '',
          f'/-- The CRT prime of instance `i` (the online instance table). -/',
          f'abbrev qOf (i : Nat) : Nat := Stage_online_eval.{qO.group(1)} (Int.ofNat i)', '',
          f'/-- The CRT basis element of instance `i` (the online instance table). -/',
          f'abbrev KOf (i : Nat) : Int := Stage_online_eval.{kO.group(1)} (Int.ofNat i)', '',
          f'/-- The online output family the decode stage reads. -/',
          f'abbrev term (execution : GeneratedClaim.Execution) := execution.«{onkey}»{proj(jterm)}', '',
          '/-- The decode stage\'s inputs in a run (the `Runs` argument of `Stage_decode`). -/',
          '@[reducible] def decodeInputs (external : GeneratedClaim.ExternalInputs) (execution : GeneratedClaim.Execution) :',
          '    DecodeLoops.Inputs :=',
          '  (' + ', '.join(decode) + ', ())', '',
          '/-- The requirement-derived facts the part theorem assumes: exactly the requirement facts',
          '(`GenRequirement.requirement_facts`). -/',
          '@[reducible] def Req (external : GeneratedClaim.ExternalInputs) : Prop :=',
          '  GenRequirement.Facts ' + ' '.join(f'({x})' for x in req), '',
          '/-- The error row of component `j`: the fresh error of its requirement-tied selectors, gather',
          'tables and label exponents (`GenFinal.freshErr`, written by `gen_final.errE_lean`). -/',
          err_body, '',
          'theorem errE_bound {external : GeneratedClaim.ExternalInputs} (hreq : Req external) :',
          f'    ∀ j c i, 4 * |(errE external j 0 c).coeff i| + 4 < ({qd} : Int) := by',
          err_proof, '',
          '/-- The value whose CRT terms component `j` reads (`DecodeLoops.decodeValue`). -/',
          '@[reducible] noncomputable def valueOf (external : GeneratedClaim.ExternalInputs)',
          f'    (execution : GeneratedClaim.Execution) (G : {gty.group(1)}) (j : Nat) :=',
          '  DecodeLoops.decodeValue (decodeInputs external execution) G (errE external) j', '',
          *NOISE_MONO, '',
          '/-- **The part theorem.** -/',
          'theorem part {model : HashModel} {external : GeneratedClaim.ExternalInputs}',
          '    {execution : GeneratedClaim.Execution} (hR : GeneratedClaim.Runs model external execution)',
          f'    (hreq : Req external) (i : Fin S) {{G : {gty.group(1)}}}',
          f'    (hG : {gty.group(2).replace("backend", "Backend.backend")} G) :',
          '    TermOf (qOf i) (KOf i) (term execution i) (valueOf external execution G ((i : Nat) % R)) := by',
          *part_skeleton(pdir, gdir, jterm, decode, req), '', 'end RefreshTopProof.GenPart', '']
    links = []
    open(out, 'w').write('\n'.join(L))
    print('wrote', out, len(links), 'links')


main()
