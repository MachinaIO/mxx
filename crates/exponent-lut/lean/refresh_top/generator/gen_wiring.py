#!/usr/bin/env python3
"""Generate `Gen_Wiring.lean` (WP F): instance wiring facts for the value level (`GenFinalValues`).

1. **One-hot selector rows** `onehot_<keys>` — for every PRF of the preprocessing (its key-cell loops
   `keys_h*` of `Gen_Inst_Stage_preprocess_Keys`, grouped by cell count and label-key seed; buckets in
   loop order), the selector externals read by the cells' message coefficients are one-hot per member:
   `∃ sel : Fin B → Fin W, ∀ b (k : Fin W), ![famAt (famAt external.input_s₀ j), …] b (o * W + k) =
   if k = sel b then 1 else 0` (the `hs` of `GenFinal.prf_z_gen` / `prfZB_onehot`), from
   `ValidExternals` (entries in `{0, 1}`) and the requirement's member sums (`hsum_b`, the
   `GenRequirement.Facts` field named in the docstring, verbatim shape). `cellsel_<keys>`: the cells'
   coefficient wire is `famAt (famAt external.input_s j)` (the Claim source, `Gen_Stage_Links.src_*`).
2. **Online table wires** `fam_<M>_<w>` — every online input wire that is a pass-through of a
   preprocess input fed by a per-member external family (`Gen_Stage_Links.ext_*`) is
   `famAt (famAt external.input_k i)` as a natural-index family.
3. **Key tables in Walsh form** — for every LUT key table `table_h*` (`keyTableG [a] [b] (Σ_k keyTermC/E …)`):
   `odd_h*` (automorphisms odd), `coef_h*` (`keyTableG … 0 0 = a * lutFourierCoefficient W (X^s) b T σ`,
   `T k = t_k` for coefficient tables, `X^(o t_k)` for exponent tables; `GenFinal.keyTableC_walsh` /
   `keyTableE_walsh`), and `switch_h*` (all branches in the `hswitch`/`heB` shape of
   `fresh_inst`/`mask_inst_form`: `σ_k(sp) • fam 0 j k + fam 1 j k = (Cw k * (1 * unitPair sp j)) • g' + e`,
   `fam = fam4` of the table wires, `Cw k` the `keyTableG` coefficient, for any runtime gadget `g'`
   (rebased tables: `g'` a centered rebase of a runtime gadget `g0`)).
4. **Smudge digits** — for the smudge LUT table (`smudge_table_*` of `Gen_Online_Smudge`):
   `smudge_hinverse_*`, `smudge_hC_*` (the coefficient in `smudge_digits`' `hC` shape: half-integer
   table, `a = 1`) and `smudge_digits_keys_*` (= `smudge_digits_*` with `C`, `hinverse`, `hodd`, `hC`
   discharged; remaining `hμ` (PRF output), `hwalsh` (members), `hΔ`).

Recognition: theorem statements of the generated `Gen_Inst_Stage_preprocess_Keys` /
`Gen_Online_Smudge` files (same generator family), the Claim (`ValidExternals`, `Runs`), the
requirement facts (`Gen_Requirement.lean`), and the stage wiring (`gen_stage_links`). No wire ids.

With `--sigma`, the same one-hot rows and key tables for the stages of the caller's encoding of
`σ_0 G` (`Stage_sigma_preprocess`, one instance per chain): `Gen_Wiring_Sigma.lean`, namespace
`RefreshTopProof.WiringSigma`.

Usage: gen_wiring.py <generated dir> <param dir> [--out FILE] [--sigma]
  <param dir> holds Gen_Inst_Stage_preprocess_Keys.lean, Gen_Online_Smudge.lean, Gen_Requirement.lean.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
import xmodel  # noqa: E402
import gen_stage_links as SL  # noqa: E402

SIGMA = '--sigma' in sys.argv
M_PRE = 'Stage_sigma_preprocess' if SIGMA else 'Stage_preprocess'
NS = 'RefreshTopProof.WiringSigma' if SIGMA else 'RefreshTopProof.Wiring'


class Fail(Exception):
    pass


def theorems(text):
    """name -> full text of every `theorem` (up to the next blank line)."""
    out = {}
    for m in re.finditer(r'^theorem (\S+)', text, re.M):
        end = text.find('\n\n', m.start())
        out[m.group(1)] = text[m.start(): end if end >= 0 else len(text)]
    return out


def balanced_at(s, i):
    j = xmodel.matching(s, i)
    if j < 0:
        raise Fail('unbalanced')
    return s[i:j + 1]


def tokens(s):
    return [p for p in xmodel.split_top(xmodel.strip_parens(s), ' ') if p]


def list1(tok):
    """`(([(v)]) : List Int)` -> `v` (one-element list)."""
    m = re.match(r'^\(\(\[(.*)\]\) : List Int\)$', tok.strip())
    if not m:
        raise Fail(f'list {tok}')
    items = xmodel.split_top(m.group(1), ',')
    if len(items) != 1:
        raise Fail(f'list of {len(items)} items')
    return xmodel.strip_parens(items[0])


def int_lit(tok):
    m = re.fullmatch(r'-?\d+', xmodel.strip_parens(re.sub(r'\s*:\s*Int\)$', ')', tok.strip())))
    if not m:
        t = xmodel.strip_parens(tok)
        m2 = re.fullmatch(r'\((-?\d+)\) : Int', t) or re.fullmatch(r'(-?\d+) : Int', t)
        if not m2:
            raise Fail(f'literal {tok}')
        return int(m2.group(1))
    return int(m.group(0))


# --------------------------------------------------------------------------- key tables

class Table:
    """A `table_h*` theorem of the preprocess keys file."""

    def __init__(self, name, text, S):
        self.name = name
        self.tag = name[len('table_'):]
        head, rest = text.split('\n', 1)
        stmt, proof = text.split(' := by\n', 1)
        m = re.search(r':\n    ∀ i : Fin (\d+), ', stmt)
        if not m:
            raise Fail(f'{name}: shape')
        self.W = int(m.group(1))
        self.binders = stmt[len(f'theorem {name} '): m.start()].replace('(i_0 < S)', f'(i_0 < {S})')
        body = stmt[m.end():]
        i = body.index('((keyTableG ')
        inner = balanced_at(body, i + 1)
        self.KT = f'({inner} 0 0)'
        toks = tokens(inner)
        # keyTableG A B S : ExactMatrix (q) (n) 1 1
        if toks[0] != 'keyTableG' or toks[4] != ':' or toks[5] != 'ExactMatrix':
            raise Fail(f'{name}: keyTableG form {toks[:6]}')
        self.a, self.b = list1(toks[1]), list1(toks[2])
        self.q, self.n = toks[6], xmodel.strip_parens(toks[7])
        ssum = xmodel.strip_parens(toks[3])
        mm = re.match(r'^∑ k ∈ Finset\.range \(\((\d+)\) : Int\)\.toNat, (.*)$', ssum, re.S)
        if not mm or int(mm.group(1)) != self.W:
            raise Fail(f'{name}: sum')
        tt = tokens(mm.group(2))
        self.kind = tt[0]
        if self.kind == 'keyTermC':
            self.sig, self.tv, one, s, k = tt[1:6]
            self.o = None
        elif self.kind == 'keyTermE':
            self.sig, o, self.tv, one, s, k = tt[1:7]
            self.o = int_lit(o)
        else:
            raise Fail(f'{name}: term {self.kind}')
        if xmodel.strip_parens(one) != 'matrixPolynomial (([(1)]) : List Int)':
            raise Fail(f'{name}: one {one}')
        ms = re.fullmatch(r'-\(\((\d+)\) : Int\)', xmodel.strip_parens(s))
        if not ms:
            raise Fail(f'{name}: stride {s}')
        self.s = int(ms.group(1))
        # gadget part
        g1 = re.search(r'∃ G : ExactMatrix (.+?) \((\d+)\) 1 \((\d+)\), gadgetMatrixRuns backend (\(\(\d+\) : Int\)) (\(\(\d+\) : Int\)) G ∧', body)
        g2 = re.search(r'∃ \(G0 : ExactMatrix (.+?) \((\d+)\) 1 \((\d+)\)\) \(G : ExactMatrix (.+?) \((\d+)\) 1 \((\d+)\)\), gadgetMatrixRuns backend (\(\(\d+\) : Int\)) (\(\(\d+\) : Int\)) G0 ∧ centeredRebaseRuns G0 G ∧', body)
        if g2:
            self.rebased = True
            self.q0, self.m = g2.group(1), int(g2.group(6))
            self.base, self.digits = g2.group(7), g2.group(8)
        elif g1:
            self.rebased = False
            self.m = int(g1.group(3))
            self.base, self.digits = g1.group(4), g1.group(5)
        else:
            raise Fail(f'{name}: gadget')
        me = re.search(r'∃ eB : Fin 2 → ErrorMatrix \((\d+)\) 1 \((\d+)\), \(∀ j, CoeffBound \(eB j\) (.*?)\) ∧', body)
        if not me:
            raise Fail(f'{name}: error bound')
        self.cut = me.group(3)
        mw = re.search(r'• !\[\((\S+\.w_\d+_)0 ctx i\), \(\S+\.w_\d+_1 ctx i\)\] j \+ !\[\(\S+\.w_\d+_2 ctx i\), \(\S+\.w_\d+_3 ctx i\)\] j =', body)
        if not mw:
            raise Fail(f'{name}: wires')
        self.wp = mw.group(1)
        mq = re.search(r'\(hq := ', proof)
        self.hq = balanced_at(proof, mq.start())[len('(hq := '):-1] if mq else '(by decide)'
        self.hasFact = '[Fact (i_0 <' in self.binders
        # explicit binders of the table theorem, passed through in order
        ex = []
        i = 0
        while i < len(self.binders):
            if self.binders[i] == '(':
                j = xmodel.matching(self.binders, i)
                ex.append(self.binders[i + 1:j].split(' : ')[0].strip())
                i = j + 1
            elif self.binders[i] in '{[':
                i = xmodel.matching(self.binders, i) + 1
            else:
                i += 1
        self.args = ' '.join(ex)

    def kt_at(self, var):
        return re.sub(r'\(\(i : Nat\)\)', f'(({var} : Nat))', self.KT)

    def sig_at(self, var):
        return re.sub(r'\(\(i : Nat\)\)', f'(({var} : Nat))', self.sig)

    def table_fun(self):
        n, q = self.n, self.q
        if self.kind == 'keyTermC':
            return f'(fun k ↦ ((({self.tv}) k : Int) : ExactPoly {q} {n}))'
        return f'(fun k ↦ reducePoly {q} {n} (Xe (n := {n}) ({self.o} * (({self.tv}) k).toNat)))'

    def ibind(self):
        """Binders of the instance index for the pure statements."""
        return f'(i_0 : Nat) [Fact (i_0 < {self.S})] ' if 'i_0' in self.KT or 'i_0' in self.q else ''


def emit_tables(tabs, S):
    out = []
    for t in tabs:
        t.S = S
        W, n, q = t.W, t.n, t.q
        ib = t.ibind()
        iarg = ' i_0' if ib else ''
        sigf = f'(fun (i : Fin {W}) ↦ {t.sig})'
        out.append(f'''/-- `{t.name}`: its automorphisms are odd. -/
theorem odd_{t.tag} : ∀ k : Fin {W}, Odd ({sigf} k).toNat := by
  decide
''')
        rhs = (f'(({t.a} : Int) : ExactPoly {q} {n}) * lutFourierCoefficient {W} (X {q} ^ {t.s}) '
               f'(({t.b} : Int) : ExactPoly {q} {n})\n        {t.table_fun()} ({t.sig}).toNat')
        walsh = int(n) == t.s * W
        t.walsh = walsh
        if t.kind == 'keyTermC':
            prf = (f'keyTableC_walsh\' (s := {t.s}) (W := {W}) {t.hq} (by norm_num) (odd_{t.tag} i) '
                   f'(by norm_num) _ _ _')
        else:
            prf = (f'keyTableE_walsh_int (s := {t.s}) (W := {W}) (o := {t.o}) {t.hq} (by norm_num) (odd_{t.tag} i) '
                   f'(by norm_num) _ _ _ (by decide)')
        if not walsh:
            out.append(f'-- note: {t.name}: n = {n} is not stride {t.s} * width {W}; no Walsh form\n')
        else:
          out.append(f'''/-- `{t.name}`: the key coefficient of branch `i` as its Walsh coefficient (`GenFinal.{'keyTableC_walsh' if t.kind == 'keyTermC' else 'keyTableE_walsh'}`). -/
theorem coef_{t.tag} {ib}(i : Fin {W}) :
    {t.KT} =
      {rhs} :=
  {prf}
''')
        fam = ' '.join(f'({t.wp}{c} ctx)' for c in range(4))
        lhs = (f'reducePoly {q} {n} (substitution ({sigf} k).toNat (odd_{t.tag} k) sp) •\n'
               f'          fam4 {fam} 0 j k +\n          fam4 {fam} 1 j k')
        cw = f'(fun (i : Fin {W}) ↦ {t.KT}) k'
        rhs2 = f'({cw} *\n          reducePoly {q} {n} (1 * unitPair sp j)) • g\' + reduceMatrix {q} {n} 1 {t.m} (eB'
        if t.rebased:
            gb = (f'{{g0 : ExactMatrix {t.q0} {n} 1 {t.m}}} {{g\' : ExactMatrix {q} {n} 1 {t.m}}}\n'
                  f'    (hg0 : gadgetMatrixRuns backend {t.base} {t.digits} g0) (hg\' : centeredRebaseRuns g0 g\')')
            get = ('    obtain ⟨-, -, G0, G, hG0, hG, hodd, eB, hb, he⟩ := ' + f'{t.name} {t.args} k\n'
                   '    obtain rfl := gadgetMatrixRuns_unique hG0 hg0\n'
                   '    obtain rfl : G = g\' := hG.2.trans hg\'.2.symm\n')
        else:
            gb = (f'{{g\' : ExactMatrix {q} {n} 1 {t.m}}} (hg\' : gadgetMatrixRuns backend {t.base} {t.digits} g\')')
            get = ('    obtain ⟨-, -, G, hG, hodd, eB, hb, he⟩ := ' + f'{t.name} {t.args} k\n'
                   '    obtain rfl := gadgetMatrixRuns_unique hG hg\'\n')
        one = (f'∃ eB : Fin 2 → ErrorMatrix {n} 1 {t.m}, (∀ j, CoeffBound (eB j) {t.cut}) ∧\n'
               f'      ∀ (j : Fin 2), {lhs} =\n        {rhs2} j)')
        out.append(f'''/-- `{t.name}` for all branches in the `hswitch` / `heB` shape (`fam = fam4` of the table, `Cw k` its
`keyTableG` coefficient, `coef_{t.tag}`). -/
theorem switch_{t.tag} {t.binders}
    {gb} :
    ∃ eB : Fin {W} → Fin 2 → ErrorMatrix {n} 1 {t.m}, (∀ k j, CoeffBound (eB k j) {t.cut}) ∧
      ∀ (k : Fin {W}) (j : Fin 2), {lhs} =
        {rhs2} k j) := by
  have ex : ∀ k : Fin {W}, {one} := fun k ↦ by
{get}    refine ⟨eB, hb, fun j ↦ ?_⟩
    rw [one_mul]
    fin_cases j
    · exact he 0
    · exact he 1
  choose eB hb he using ex
  exact ⟨eB, hb, he⟩
''')
    return out


# --------------------------------------------------------------------------- smudge

def emit_smudge(tabs, smudge_text, S):
    out = []
    ths = theorems(smudge_text)
    for name, txt in ths.items():
        m = re.match(r'^smudge_table_(h\d+_\d+)$', name)
        if not m:
            continue
        tag = m.group(1)
        t = next((x for x in tabs if x.tag == tag), None)
        if t is None or t.kind != 'keyTermC':
            raise Fail(f'smudge table {tag}')
        mp = re.search(r'= halfTable (\d+) (\d+) k', txt)
        p_, W = int(mp.group(1)), int(mp.group(2))
        log = W.bit_length() - 1
        if 2 ** log != W or W != t.W or t.a != '1':
            raise Fail(f'smudge table {tag}: W={W} a={t.a}')
        q, n, b = t.q, t.n, t.b
        ib, iarg = t.ibind(), (' i_0' if t.ibind() else '')
        cases = '  have := (Fact.out : i_0 < {S})\n  interval_cases i_0 <;> decide\n'.format(S=S) if ib else '  decide\n'
        out.append(f'''/-- The smudge table's inverse width is nonnegative. -/
theorem nonneg_{tag} {ib}: 0 ≤ (({b} : Int)) := by
{cases}
set_option maxRecDepth 65536 in
/-- The smudge table's inverse width: `b 2^log ≡ 1`. -/
theorem inv_{tag} {ib}: ((({b} : Int)).toNat * 2 ^ {log}) % {q} = 1 := by
{cases}
/-- `smudge_digits`' `hinverse`. -/
theorem smudge_hinverse_{tag} {ib}:
    (((({b} : Int)).toNat : Nat) : ExactPoly {q} {n}) * ((2 ^ {log} : Nat) : ExactPoly {q} {n}) = 1 :=
  cast_inverse {t.hq} (inv_{tag}{iarg})

/-- `smudge_digits`' `hC` with `C i` the smudge table's `keyTableG` coefficient. -/
theorem smudge_hC_{tag} {ib}(i : Fin {W}) :
    {t.KT} =
      lutFourierCoefficient (2 ^ {log}) (X {q} ^ {t.s}) (((({b} : Int)).toNat : Nat) : ExactPoly {q} {n})
        (fun k : Fin (2 ^ {log}) ↦ reducePoly {q} {n} ((halfTable {p_} (2 ^ {log}) k : Int) : ErrorPoly {n})) ({t.sig}).toNat := by
  have hb : (((({b} : Int)).toNat : Nat) : ExactPoly {q} {n}) = (({b} : Int) : ExactPoly {q} {n}) := by
    rw [← Int.cast_natCast, Int.toNat_of_nonneg (nonneg_{tag}{iarg})]
  rw [coef_{tag}{iarg} i, Int.cast_one, one_mul, hb]
  exact walsh_halfTable (p' := {p_}) _ _ _ smudge_table_{tag} _
''')
        # smudge_digits with C/hinverse/hodd/hC discharged
        for dname, dtxt in ths.items():
            if not dname.startswith('smudge_digits_h'):
                continue
            stmt = dtxt.split(' :=\n', 1)[0]
            lines = stmt.split('\n')
            keep = []
            for ln in lines[1:]:
                s = ln.strip()
                if s.startswith('{C : ') or s.startswith('(hinverse') or s.startswith('(hodd') or s.startswith('(hC'):
                    if s.startswith('{C : '):
                        rest = re.sub(r'^\{C : [^}]*\}\s*', '', s)
                        rest = re.sub(r'\{inverse : Nat\}', '', rest).strip()
                        if rest:
                            keep.append('    ' + rest)
                    continue
                if s.startswith('(fun k : Fin (2 ^') and keep and False:
                    continue
                keep.append(ln)
            # drop the continuation line of hC (starts with `(fun k : Fin (2 ^`)
            keep = [ln for ln in keep if not ln.strip().startswith('(fun k : Fin (2 ^')]
            body = '\n'.join(keep)
            pos = smudge_text.index(f'theorem {dname} ')
            opened = re.findall(r'^open (Stage_\w+)$', smudge_text[:pos], re.M)
            if opened:
                body = re.sub(r'(?<![\w.])(sel_\d+|mod_\d+|modulus_\d+)\b', opened[-1] + r'.\1', body)
            body = re.sub(r'\* C k\b', f'* (fun (i : Fin {W}) ↦ {t.KT}) k', body)
            body = re.sub(r'\(hodd k\)', f'(odd_{tag} k)', body)
            head = lines[0].replace(f'theorem {dname} {{i_0 : Nat}}', f'theorem {dname.replace("smudge_digits_", "smudge_digits_keys_")} (i_0 : Nat) [Fact (i_0 < {S})]')
            if 'Fact' not in head:
                raise Fail('smudge digits header')
            out.append(f'''/-- `{dname}` with the smudge table's coefficients (`C i` = `{t.name}`'s `keyTableG` coefficient):
`hinverse`, `hodd`, `hC` discharged (`smudge_hinverse_{tag}`, `odd_{tag}`, `smudge_hC_{tag}`). -/
{head}
{body} :=
  {dname} (smudge_hinverse_{tag}{iarg}) odd_{tag} (smudge_hC_{tag}{iarg}) hμ hwalsh hΔ
''')
    return out


# --------------------------------------------------------------------------- one-hot selectors

def claim_info(gdir):
    text = open(f'{gdir}/Claim.lean').read()
    types = dict(re.findall(r'^  (input_\d+) : (.*)$', text.split('structure ExternalInputs where', 1)[1].split('\n\n')[0], re.M))
    ve = text.split('def ValidExternals', 1)[1].split(':=\n', 1)[1].split('\n\n')[0]
    conj = [c.strip() for c in xmodel.split_top(ve.strip(), ' ∧\n')]
    proj = {}
    for k, c in enumerate(conj):
        ins = set(re.findall(r'external\.(input_\d+)', c))
        if len(ins) == 1:
            path = 'hv' + '.2' * k + ('.1' if k < len(conj) - 1 else '')
            proj[ins.pop()] = (path, c)
    return types, proj


def onehot_sums(param_dir, gdir):
    """selector external -> (field, J, M, W, verbatim hypothesis with the external)."""
    path = f'{param_dir}/Gen_Requirement.lean'
    if not os.path.exists(path):
        raise Fail('no Gen_Requirement.lean')
    rt = open(path).read()
    hdr = re.search(r'^structure Facts (.*) : Prop where', rt, re.M).group(1)
    xs = re.findall(r'\((x\d+) : ', hdr)
    calls = xmodel.parse_claim(gdir)
    req = next(c for c in calls if c.get('module') == 'Requirement_0')
    ext_of = dict(zip(xs, req['args']))
    out = {}
    for m in re.finditer(r'^  (\w+) : (∀ \w+ < (\d+), ∀ \w+ < (\d+), \(0 \+ ∑ \w+ ∈ Finset\.range (\d+), \(famAt \(famAt (x\d+) \w+\) .*\) = 1)$', rt, re.M):
        x = m.group(6)
        e = ext_of.get(x, '').strip()
        if not e.startswith('external.'):
            continue
        hyp = re.sub(r'\b' + x + r'\b', e, m.group(2))
        out[e] = (m.group(1), int(m.group(3)), int(m.group(4)), int(m.group(5)), hyp)
    return out


def emit_onehot(keys_text, gdir, param_dir, S):
    types, proj = claim_info(gdir)
    sums = onehot_sums(param_dir, gdir)
    calls = [c for c in xmodel.parse_claim(gdir) if c.get('module') == M_PRE]
    st = SL.Stage(gdir, calls[0])
    srcs = st.sources()
    inst = st.inst
    groups = {}
    order = []
    for name, txt in theorems(keys_text).items():
        if not re.match(r'^keys_h\d+_\d+$', name):
            continue
        m = re.search(r'∀ i : Fin (\d+), ', txt)
        mm = re.search(r'message \(\S+\.(w_\d+_\d+) ctx i\) \(\S+\.(w_\d+_\d+) ctx i\)', txt)
        ms = re.search(r'labelKeysG hashModel \(\S+\.(w_\d+_\d+) ctx\)', txt)
        if not (m and mm and ms):
            continue
        cw = mm.group(1)
        if cw not in srcs:
            continue
        if not srcs[cw][3]:
            continue
        term = srcs[cw][2].strip()
        key = (int(m.group(1)), ms.group(1))
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append((name, cw, term))
    out = []
    for key in order:
        g = groups[key]
        N = key[0]
        exts = [t for _, _, t in g]
        if not all(e in sums for e in exts):
            out.append(f'-- note: PRF {g[0][0]}: selectors {exts} without a requirement member sum\n')
            continue
        Js = {sums[e][1] for e in exts}
        Ms = {sums[e][2] for e in exts}
        Ws = {sums[e][3] for e in exts}
        if len(Js) != 1 or len(Ms) != 1 or len(Ws) != 1:
            raise Fail(f'{g[0][0]}: bucket shapes')
        J, M, W = Js.pop(), Ms.pop(), Ws.pop()
        if M * W != N:
            raise Fail(f'{g[0][0]}: {M}*{W} != {N}')
        B = len(exts)
        tag = g[0][0][len('keys_'):]
        hyps = '\n'.join(f'    (hsum{b} : {sums[e][4]})' for b, e in enumerate(exts))
        fields = ', '.join(f'`hsum{b}` = `hreq.{sums[e][0]}`' for b, e in enumerate(exts))
        vec = ', '.join(f'famAt (famAt {e} j)' for e in exts)
        cons = 'finZeroElim'
        for b in reversed(range(B)):
            cons = f'(Fin.cons (onehot_bucket (W := {W}) _ {proj[exts[b].split(".")[1]][0]} hsum{b} hj ho)\n      {cons})'
        out.append(f'''/-- The one-hot selector rows of the PRF with key cells {', '.join(f'`{n}`' for n, _, _ in g)} (buckets in that
order; `{W}` cells per member, `{M}` members): {fields} (`GenRequirement.Facts`). -/
theorem onehot_{tag} {{external : ExternalInputs}} (hv : ValidExternals external)
{hyps}
    {{j : Nat}} (hj : j < {J}) {{o : Nat}} (ho : o < {M}) :
    ∃ sel : Fin {B} → Fin {W}, ∀ b (k : Fin {W}),
      (![{vec}] : Fin {B} → Nat → Int) b (o * {W} + k) = if k = sel b then 1 else 0 :=
  onehot_rows _ o {cons}
''')
        for (name, cw, term) in g:
            if not re.fullmatch(r'external\.input_\d+', term):
                continue
            e = term
            out.append(f'''/-- The message coefficients of `{name}` are the selector family `{e}` of the member. -/
theorem cellsel_{name[len('keys_'):]} {{backend : MxxRuntime.BackendContext}} {{hashModel : MxxRuntime.HashModel}} {{external : ExternalInputs}} {{execution : Execution}} {{i : Fin {st.S}}}
    {{rM : {M_PRE}.generatedRoot.Ctx backend hashModel {st.params}}} {{cM : {M_PRE}.{inst}.Ctx backend hashModel {st.params} (i : Nat)}}
    (hin : cM.inputs = StageLinks.{M_PRE}.rootArgs rM i) (hroot : rM.inputs = StageLinks.{M_PRE}.claimArgs external execution) :
    famAt ({M_PRE}.{inst}.{cw} cM) = famAt (famAt {e} (i : Nat)) :=
  (congrArg famAt (StageLinks.src_{M_PRE}_{cw} hin hroot)).trans (famAt_member _ i)
''')
    return out


def emit_fams(gdir, links_text):
    types, _ = claim_info(gdir)
    out = []
    for name, txt in theorems(links_text).items():
        m = re.match(r'^ext_(\w+?)_(w_\d+_\d+)$', name)
        if not m:
            continue
        stmt = txt.split(' :=\n', 1)[0]
        mt = re.search(r'\n    (\S+)\.(\S+)\.(w_\d+_\d+) cB = \((external\.(input_\d+))\) i$', stmt)
        if not mt:
            continue
        ty = types.get(mt.group(5), '')
        if not re.fullmatch(r'Fin \d+ → Fin \d+ → Int', ty.strip()):
            continue
        binders = stmt[len(f'theorem {name} '): stmt.rindex(' :\n')]
        binders = re.sub(r'\b(Stage_\w+)\.(rootArgs|rootOuts|claimArgs)\b', r'StageLinks.\1.\2', binders)
        binders = binders.replace('(i : Fin S)', '(i : Fin StageLinks.S)').replace('{i : Fin S}', '{i : Fin StageLinks.S}')
        e = mt.group(4)
        out.append(f'''/-- The online table wire `{mt.group(3)}` as a natural-index family: `famAt (famAt {e} i)` (`StageLinks.{name}`). -/
theorem fam_{m.group(1)}_{m.group(2)} {binders} :
    famAt ({mt.group(1)}.{mt.group(2)}.{mt.group(3)} cB) = famAt (famAt {e} (i : Nat)) :=
  (congrArg famAt (StageLinks.{name} hinA hrootA houtA hinstA hinB hrootB)).trans (famAt_member _ i)
''')
    return out


LEMMAS = r'''/-! ## Generic lemmas -/

/-- `W` values in `{0, 1}` summing to `1` are a one-hot row. -/
theorem onehot_one {W : Nat} (x : Fin W → Int) (h01 : ∀ k, 0 ≤ x k ∧ x k ≤ 1) (hsum : ∑ k, x k = 1) :
    ∃ sel : Fin W, ∀ k, x k = if k = sel then 1 else 0 := by
  have hex : ∃ s, x s = 1 := by
    by_contra hne
    push_neg at hne
    have h0 : ∀ k, x k = 0 := fun k ↦ by have := h01 k; have := hne k; omega
    simp [h0] at hsum
  obtain ⟨s, hs⟩ := hex
  refine ⟨s, fun k ↦ ?_⟩
  split_ifs with hk
  · rw [hk, hs]
  · have hle : x s + x k ≤ ∑ j, x j := by
      have := Finset.sum_le_sum_of_subset_of_nonneg (f := x) (Finset.subset_univ {s, k})
        (fun j _ _ ↦ (h01 j).1)
      rwa [Finset.sum_pair (Ne.symm hk)] at this
    have := (h01 k).1
    omega

/-- One-hot rows in every bucket: a selected cell per bucket (`GenFinal.prfZB_onehot`'s `hs`). -/
theorem onehot_rows {B W : Nat} (s : Fin B → Nat → Int) (o : Nat)
    (h : ∀ b, (∀ k : Fin W, 0 ≤ s b (o * W + k) ∧ s b (o * W + k) ≤ 1) ∧
      ∑ k ∈ Finset.range W, s b (o * W + k) = 1) :
    ∃ sel : Fin B → Fin W, ∀ b (k : Fin W), s b (o * W + k) = if k = sel b then 1 else 0 := by
  have := fun b ↦ onehot_one (fun k : Fin W ↦ s b (o * W + k)) (h b).1
    (by rw [← (h b).2, Finset.sum_range (fun k ↦ s b (o * W + k))])
  choose sel hsel using this
  exact ⟨sel, hsel⟩

/-- A bucket of a selector family: entries in `{0, 1}` (`ValidExternals`) and member sums `1`
(the requirement's loop, verbatim shape). -/
theorem onehot_bucket {J N M W : Nat} (x : Fin J → Fin N → Int)
    (h01 : ∀ a : Fin J, ∀ c : Fin N, 0 ≤ x a c ∧ x a c ≤ 1)
    (hsum : ∀ j < J, ∀ o < M, (0 + ∑ k ∈ Finset.range W,
      famAt (famAt x j) ((Int.ofNat o * (W : Int)) + Int.ofNat k).toNat) = 1)
    {j : Nat} (hj : j < J) {o : Nat} (ho : o < M) :
    (∀ k : Fin W, 0 ≤ famAt (famAt x j) (o * W + k) ∧ famAt (famAt x j) (o * W + k) ≤ 1) ∧
      ∑ k ∈ Finset.range W, famAt (famAt x j) (o * W + k) = 1 := by
  refine ⟨fun k ↦ ?_, ?_⟩
  · unfold famAt
    split_ifs with h1 h2 <;> first | exact h01 _ _ | omega | simp
  · have := hsum j hj o ho
    rw [zero_add] at this
    rw [← this]
    refine Finset.sum_congr rfl fun k _ ↦ ?_
    congr 1 <;> omega

/-- A member of a per-instance family as a natural-index family. -/
theorem famAt_member {S N : Nat} (f : Fin S → Fin N → Int) (i : Fin S) :
    famAt (f i) = famAt (famAt f (i : Nat)) := by
  rw [famAt_of_lt _ i.isLt]

section
variable {n : Nat}

/-- A coefficient key table as its Walsh coefficient (`keyTableC_walsh`, folded form). -/
theorem keyTableC_walsh' {q : Nat} (hq : 1 < q) (hn0 : 0 < n) {σ : Int} (hodd : Odd σ.toNat) {s W : Nat}
    (hn : n = s * W) (a b : Int) (tv : Fin W → Int) :
    (keyTableG [a] [b] (∑ k ∈ Finset.range W, keyTermC σ tv (matrixPolynomial [1] : ExactMatrix q n 1 1) (-(s : Int)) k)) 0 0 =
      (a : ExactPoly q n) * lutFourierCoefficient W (X q ^ s) (b : ExactPoly q n)
        (fun k ↦ ((tv k : Int) : ExactPoly q n)) σ.toNat := by
  unfold keyTableG keyTermC
  exact keyTableC_walsh hq hn0 hodd hn a b tv

/-- An exponent key table with nonnegative values as its Walsh coefficient (`keyTableE_walsh`). -/
theorem keyTableE_walsh_int {q : Nat} (hq : 1 < q) (hn0 : 0 < n) {σ : Int} (hodd : Odd σ.toNat) {s W : Nat}
    (hn : n = s * W) (a b : Int) (o : Nat) (tv : Fin W → Int) (htv : ∀ k, 0 ≤ tv k) :
    (keyTableG [a] [b] (∑ k ∈ Finset.range W, keyTermE σ (o : Int) tv (matrixPolynomial [1] : ExactMatrix q n 1 1) (-(s : Int)) k)) 0 0 =
      (a : ExactPoly q n) * lutFourierCoefficient W (X q ^ s) (b : ExactPoly q n)
        (fun k ↦ reducePoly q n (Xe (n := n) (o * (tv k).toNat))) σ.toNat := by
  have e : tv = fun k ↦ (((tv k).toNat : Nat) : Int) := funext fun k ↦ (Int.toNat_of_nonneg (htv k)).symm
  have := keyTableE_walsh (n := n) hq hn0 hodd hn a b o (fun k ↦ (tv k).toNat)
  unfold keyTableG keyTermE
  conv_lhs => rw [e]
  exact this
end
'''


def gen(gdir, param_dir):
    keys_text = open(f'{param_dir}/Gen_Inst_{M_PRE}_Keys.lean').read()
    smudge_path = f'{param_dir}/Gen_Online_Smudge.lean'
    smudge_text = open(smudge_path).read() if os.path.exists(smudge_path) and not SIGMA else ''
    links_text, _ = SL.gen(gdir)
    # the instance count of the preprocessing stage
    S = SL.Stage(gdir, next(c for c in xmodel.parse_claim(gdir) if c.get('module') == M_PRE)).S
    tabs = []
    notes = []
    for name, txt in theorems(keys_text).items():
        if re.match(r'^table_h\d+_\d+$', name):
            try:
                tabs.append(Table(name, txt, S))
            except Fail as e:
                notes.append(f'-- note: {e}')
    for t in tabs:
        t.S = S
    imp = [f'import Gen_Inst_{M_PRE}_Keys'] if SIGMA else ['import Gen_Online_Smudge']
    keys_ns = 'RefreshTopProof.GenInst.' + M_PRE.replace('Stage_', '').title().replace('_', '') + 'Keys'
    out = [*imp, 'import GenFinalValues', 'import Gen_Stage_Links',
           'import Mathlib.Tactic.IntervalCases', '',
           '/-! Generated by `generator/gen_wiring.py` (WP F): instance wiring for the value level — one-hot',
           'selector rows, online table wires as external families, key tables in Walsh form (`hswitch` / `heB`',
           'shapes) and the smudge digits with the smudge table discharged (see the generator\'s docstring). -/', '',
           'set_option maxHeartbeats 4000000', 'set_option maxRecDepth 16384', '',
           f'namespace {NS}', '',
           'open RefreshTopProof.GenIface RefreshTopProof.GenIface.BrLut RefreshTopProof.GenIface.BrLutKey RefreshTopProof.Gen MxxExponentLut',
           f'open {keys_ns}' + ('' if SIGMA else ' RefreshTopProof.GenOnlineSmudge') + ' RefreshTopProof.GenFinal',
           'open RefreshTopProof GeneratedClaim',
           'open Mxx.Primitives MxxRuntime Mxx.ExponentLut.Refresh', 'open scoped BigOperators', '', LEMMAS, '',
           '/-! ## One-hot selector rows -/', '']
    out += emit_onehot(keys_text, gdir, param_dir, S)
    if not SIGMA:
        out += ['/-! ## Online table wires -/', '']
        out += emit_fams(gdir, links_text)
    out += ['/-! ## Key tables in Walsh form -/', '', 'section Tables', f'open {M_PRE}', '']
    out += emit_tables(tabs, S)
    out += emit_smudge(tabs, smudge_text, S) if smudge_text else []
    out += ['end Tables', '', *notes, '', f'end {NS}', '']
    return '\n'.join(out)


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    outp = sys.argv[sys.argv.index('--out') + 1] if '--out' in sys.argv else os.path.join(
        args[1], 'Gen_Wiring_Sigma.lean' if SIGMA else 'Gen_Wiring.lean')
    args = [a for a in args if a != outp]
    txt = gen(args[0], args[1])
    open(outp, 'w').write(txt)
    n = len(re.findall(r'^theorem ', txt, re.M))
    print(f'{outp}: {n} theorems, {txt.count("-- note:")} notes')
