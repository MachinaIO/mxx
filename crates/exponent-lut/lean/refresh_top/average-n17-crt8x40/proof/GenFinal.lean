import GenDecode
import GenKey
import ExponentRefresh
import ExponentPublic
import ExponentStacking

/-!
# The online final combination of one instance, for any parameter set

Instance `i` of the online evaluation ends with
`ζ = b P - ((comb + fresh) + smudge) - H` over its ring `R_i`, where `b` is the hint row
(`1 × h`), `P` the decoder preimage (`h × m`), `comb = [g | c₁] D - cₓ Y` the refresh
combination, `fresh` the fresh-noise row, `smudge` the smudge row and `H` the row of halves.
The stage then switches `ζ` to its CRT prime `q_i` (`R_i = Q_q q_i`), lifts the result centered to
the decoding modulus and multiplies it by the CRT basis element `K_i`: the CRT term that the
decode stage reads (`GenDecode.TermOf`).

Nothing here names the ring dimension `n`, the column count `m`, the hint width `h`, a modulus,
the gadget base or digit count, or the number of slots:

* `decompose_rows_gen`, `scaled_rows_gen`, `comb_gen`, `comb_bound_gen`: the combination
  `[g | c₁] D - cₓ Y` (`TopChain.refresh_columnR` without the toy constants).
* `unmask_gen`: `ζ = Q_q (Y + E') + ν` with `ν` small (`ExponentRefresh.decoded_value` on whole
  rows).
* `switch_gen`: the modulus switch returns the residue of `Y + E'`.
* `ResidueEq`, `residueEq_add_reduce`, `residueEq_reduce`: residues modulo `q` of rows over
  different multiples of `q`.
* `term_gen`, `final_gen`: the CRT term of the instance (the conclusion `GenPart.part` needs).
* `prfZ`, `samplerVal`, `freshErr`, `freshErr_coeff_le`, `freshErr_bound`: the fresh error `E'` of
  a component as a function of its requirement-tied selectors, gather tables and label
  exponents, and its bound (the body of `GenPart.errE`).
-/

namespace RefreshTopProof.GenFinal

open Mxx.Primitives MxxRuntime MxxExponentLut Mxx.ExponentLut.Refresh RefreshTopProof GenDecode
open scoped BigOperators

variable {n : Nat}

/-! ## The refresh combination -/

/-- A compact centered rebase within one modulus is the identity (any ring dimension). -/
theorem compactRebase_self_gen (hn : 0 < n) {q rows columns : Nat} (hq : 1 < q) {bound : Nat}
    {x y : ExactMatrix q n rows columns} (h : compactCenteredRebaseRuns bound x y) : y = x := by
  rw [centeredRebaseRuns_reduce h.2.2.1]
  exact reduceMatrix_centeredIntLift hq hn x

/-- The gadget times the halves of the re-encoded digits of a two-row `z` are its rows, for any
gadget base, digit count, modulus and ring dimension (base and digit count are read off the
decomposition's layout). -/
theorem decompose_rows_gen (hn : 0 < n) {R c m : Nat} {backend : BackendContext} {base digits : Int}
    (L : RegularLayout R) (hL : backend.regularLayout R n = some L) (hx : L.droppedModuli = 0)
    (hR : 1 < R) {z : ExactMatrix R n (1 + 1) c} {Dz' Dz : ExactMatrix R n (m + m) c}
    (hDz : gadgetDecomposeRuns backend base digits z Dz')
    {bound : Nat} (hc : compactCenteredRebaseRuns bound Dz' Dz)
    {G : ExactMatrix R n 1 m} (hG : gadgetMatrixRuns backend base digits G) :
    G * topRows Dz = (fun _ k ↦ z 0 k) ∧ G * bottomRows Dz = (fun _ k ↦ z 1 k) := by
  rw [compactRebase_self_gen hn hR hc]
  have hlay := hDz
  obtain ⟨L', hL', hb, hd, hw, -, -⟩ := hlay
  rw [hL] at hL'
  cases hL'
  have hm : m = 1 * L.digitCount := by omega
  have hz : concatRows (fun _ k ↦ z 0 k : ExactMatrix R n 1 c) (fun _ k ↦ z 1 k) z := by
    intro row column
    fin_cases row <;> rfl
  obtain ⟨Dx, hDx⟩ := Gen.gadgetDecomposeRuns_existsG (inner := m) hn L hL hx hR hb hd hm
    (fun _ k ↦ z 0 k : ExactMatrix R n 1 c)
  obtain ⟨Dy, hDy⟩ := Gen.gadgetDecomposeRuns_existsG (inner := m) hn L hL hx hR hb hd hm
    (fun _ k ↦ z 1 k : ExactMatrix R n 1 c)
  obtain ⟨ht, hbt⟩ := Gen.decompose_stackG L hL hx hz hDz hDx hDy
  rw [ht, hbt]
  exact ⟨gadget_mul_decompose hR hn L hL hx hG hDx, gadget_mul_decompose hR hn L hL hx hG hDy⟩

/-- The rows of `matrixMulScalarLeft (Q_q) A'` are `Q_q` times the rows of `A'`. -/
theorem scaled_rows_gen {R c : Nat} {Qq : Int} {A' z : ExactMatrix R n (1 + 1) c}
    (hz : z = matrixMulScalarLeft (matrixPolynomial [Qq]) A') (r : Fin (1 + 1)) :
    (fun _ k ↦ z r k : ExactMatrix R n 1 c) = ((Qq : Int) : ExactPoly R n) • rowOf A' r := by
  funext i k
  rw [hz, matrixMulScalarLeft_eq]
  simp only [Matrix.smul_apply, smul_eq_mul, matrixPolynomial_single, rowOf]

/-- **The refresh combination.** With the mask carrier `g` (target `β G`), the one encoding's
second block `c₁` (target `G`), the target component `cₓ` (target `Pₓ`), the actions
`G X₀ = Q_q A₀`, `G X₁ = Q_q A₁`, `Pₓ Y = Q_q Tₓ`, the combination `[g | c₁] D - cₓ Y` has key
`K_g X₀ + K₁ X₁ - Kₓ Y`, target `Q_q (β A₀ + A₁ - Tₓ)` and error `e_g X₀ + e₁ X₁ - eₓ Y`. -/
theorem comb_gen {R m mx c : Nat} {s β Qq : ExactPoly R n}
    {G : ExactMatrix R n 1 m} {Px : ExactMatrix R n 1 mx}
    {g c₁ : ExactMatrix R n 1 m} {gc : ExactMatrix R n 1 (m + m)} (hgc : concatColumns g c₁ gc)
    {D : ExactMatrix R n (m + m) c} {A₀ A₁ : ExactMatrix R n 1 c}
    (hX₀ : G * topRows D = Qq • A₀) (hX₁ : G * bottomRows D = Qq • A₁)
    {cx : ExactMatrix R n 1 mx} {Y : ExactMatrix R n mx c} {Tx : ExactMatrix R n 1 c}
    (hY : Px * Y = Qq • Tx)
    {Kg K₁ : Fin 2 → ExactMatrix R n 1 m} {Kx : Fin 2 → ExactMatrix R n 1 mx}
    {eg e₁ : ExactMatrix R n 1 m} {ex : ExactMatrix R n 1 mx}
    (hg : g = maskPart s Kg - β • G + eg) (hc₁ : c₁ = maskPart s K₁ - G + e₁)
    (hcx : cx = maskPart s Kx - Px + ex) :
    matrixSub (matrixMul gc D) (matrixMul cx Y) =
      maskPart s (fun k ↦ Kg k * topRows D + K₁ k * bottomRows D - Kx k * Y) -
        Qq • (β • A₀ + A₁ - Tx) + (eg * topRows D + e₁ * bottomRows D - ex * Y) := by
  simp only [matrixSub, matrixMul]
  rw [concatColumns_mul hgc, hg, hc₁, hcx]
  simp only [Matrix.add_mul, Matrix.sub_mul, Matrix.smul_mul, maskPart_mul, hX₀, hX₁, hY]
  simp only [maskPart, smul_add, smul_sub, smul_smul]
  module

/-- The combination with integer errors and decompositions: its error is
`e_g X₀ + e₁ X₁ - eₓ Y` with the coefficient bound of the products. -/
theorem comb_bound_gen (hn : 0 < n) {R m mx c : Nat} {s β Qq : ExactPoly R n}
    {G : ExactMatrix R n 1 m} {Px : ExactMatrix R n 1 mx}
    {g c₁ : ExactMatrix R n 1 m} {gc : ExactMatrix R n 1 (m + m)} (hgc : concatColumns g c₁ gc)
    {D : ExactMatrix R n (m + m) c} {DI : ErrorMatrix n (m + m) c} {BD : Nat}
    (hD : D = reduceMatrix R n (m + m) c DI) (hDI : CoeffBound DI BD)
    {A₀ A₁ : ExactMatrix R n 1 c}
    (hX₀ : G * topRows D = Qq • A₀) (hX₁ : G * bottomRows D = Qq • A₁)
    {cx : ExactMatrix R n 1 mx} {Y : ExactMatrix R n mx c} {YI : ErrorMatrix n mx c} {BY : Nat}
    (hYI : Y = reduceMatrix R n mx c YI) (hYb : CoeffBound YI BY) {Tx : ExactMatrix R n 1 c}
    (hY : Px * Y = Qq • Tx)
    {Kg K₁ : Fin 2 → ExactMatrix R n 1 m} {Kx : Fin 2 → ExactMatrix R n 1 mx}
    {eg e₁ : ErrorMatrix n 1 m} {ex : ErrorMatrix n 1 mx} {Beg Be1 Bex : Nat}
    (heg : CoeffBound eg Beg) (he1 : CoeffBound e₁ Be1) (hex : CoeffBound ex Bex)
    (hg : g = maskPart s Kg - β • G + reduceMatrix R n 1 m eg)
    (hc₁ : c₁ = maskPart s K₁ - G + reduceMatrix R n 1 m e₁)
    (hcx : cx = maskPart s Kx - Px + reduceMatrix R n 1 mx ex) :
    ∃ ec : ErrorMatrix n 1 c,
      CoeffBound ec (m * n * Beg * BD + m * n * Be1 * BD + mx * n * Bex * BY) ∧
      matrixSub (matrixMul gc D) (matrixMul cx Y) =
        maskPart s (fun k ↦ Kg k * topRows D + K₁ k * bottomRows D - Kx k * Y) -
          Qq • (β • A₀ + A₁ - Tx) + reduceMatrix R n 1 c ec := by
  have htop : CoeffBound (topRows DI) BD := fun r k i ↦ hDI _ k i
  have hbot : CoeffBound (bottomRows DI) BD := fun r k i ↦ hDI _ k i
  refine ⟨eg * topRows DI + e₁ * bottomRows DI - ex * YI,
    coeffBound_sub (coeffBound_add (coeffBound_mul hn heg htop) (coeffBound_mul hn he1 hbot))
      (coeffBound_mul hn hex hYb), ?_⟩
  rw [comb_gen hgc hX₀ hX₁ hY hg hc₁ hcx]
  congr 1
  have ht : topRows D = reduceMatrix R n m c (topRows DI) := by
    rw [hD]; rfl
  have hb : bottomRows D = reduceMatrix R n m c (bottomRows DI) := by
    rw [hD]; rfl
  rw [ht, hb, hYI, reduceMatrix_sub, reduceMatrix_add, reduceMatrix_mul, reduceMatrix_mul,
    reduceMatrix_mul]

/-! ## The unmasked value -/

/-- **The unmasking.** The hint `b = s B + e_b`, the preimage `P` (integer `P_d`) with
`B_k P = K_c k + K_f k + K_s k`, the combination (target `Q_q Y`), the fresh row (target `Q_q E'`)
and the smudge (target `T`, `T - H = M` small): `b P - ((comb + fresh) + smudge) - H` is
`Q_q (Y + E') + ν` with `ν = e_b P_d - e_c - e_f - e_s + M`. -/
theorem unmask_gen (hn : 0 < n) {R h m : Nat} {s : ExactPoly R n} {Qq : Int}
    {b : ExactMatrix R n 1 h} {Bk : Fin 2 → ExactMatrix R n 1 h} {eb : ErrorMatrix n 1 h} {Beb : Nat}
    (hb : b = maskPart s Bk + reduceMatrix R n 1 h eb) (heb : CoeffBound eb Beb)
    {P : ExactMatrix R n h m} {Pd : ErrorMatrix n h m} {BP : Nat}
    (hP : P = reduceMatrix R n h m Pd) (hPd : CoeffBound Pd BP)
    {Kc Kf Ks : Fin 2 → ExactMatrix R n 1 m} (hpre : ∀ k, Bk k * P = Kc k + Kf k + Ks k)
    {comb Y : ExactMatrix R n 1 m} {ec : ErrorMatrix n 1 m} {Bc : Nat}
    (hc : comb = maskPart s Kc - ((Qq : Int) : ExactPoly R n) • Y + reduceMatrix R n 1 m ec)
    (hec : CoeffBound ec Bc)
    {fresh : ExactMatrix R n 1 m} {E ef : ErrorMatrix n 1 m} {Bf : Nat}
    (hf : fresh = maskPart s Kf - ((Qq : Int) : ExactPoly R n) • reduceMatrix R n 1 m E +
      reduceMatrix R n 1 m ef) (hef : CoeffBound ef Bf)
    {smudge T H : ExactMatrix R n 1 m} {es : ErrorMatrix n 1 m} {Bs : Nat}
    (hs : smudge = maskPart s Ks - T + reduceMatrix R n 1 m es) (hes : CoeffBound es Bs)
    {M : ErrorMatrix n 1 m} {BM : Nat} (hTH : T - H = reduceMatrix R n 1 m M) (hM : CoeffBound M BM) :
    ∃ ν : ErrorMatrix n 1 m, CoeffBound ν (h * n * Beb * BP + Bc + Bf + Bs + BM) ∧
      matrixSub (matrixSub (matrixMul b P) (matrixAdd (matrixAdd comb fresh) smudge)) H =
        ((Qq : Int) : ExactPoly R n) • (Y + reduceMatrix R n 1 m E) + reduceMatrix R n 1 m ν := by
  refine ⟨eb * Pd - ec - ef - es + M,
    coeffBound_add (coeffBound_sub (coeffBound_sub (coeffBound_sub
      (coeffBound_mul hn heb hPd) hec) hef) hes) hM, ?_⟩
  have hT : T = H + reduceMatrix R n 1 m M := by rw [← hTH]; abel
  simp only [matrixSub, matrixAdd, matrixMul]
  rw [reduceMatrix_add, reduceMatrix_sub, reduceMatrix_sub, reduceMatrix_sub, reduceMatrix_mul,
    ← hP, hb, hc, hf, hs, hT, Matrix.add_mul, maskPart_mul]
  simp only [hpre, maskPart, smul_add]
  module

/-! ## The modulus switch -/

/-- **The modulus switch** from `R = Q_q q` to `q` of a row `Q_q y + ν` with `2 |ν| < Q_q` returns
the residue of `y`, coefficient by coefficient (any ring dimension and column count). -/
theorem switch_gen (hn : 0 < n) {R q Qq m : Nat} (hR : R = Qq * q) (hQq : 0 < Qq) (hq : 1 < q)
    {x : ExactMatrix R n 1 m} {out : ExactMatrix q n 1 m} (h : modulusSwitchRuns x out)
    {y : ExactMatrix R n 1 m} {ν : ErrorMatrix n 1 m}
    (hx : x = ((Qq : Int) : ExactPoly R n) • y + reduceMatrix R n 1 m ν)
    (hν : ∀ c i, 2 * |(ν 0 c).coeff i| < Qq) (c : Fin m) (i : Fin n) :
    (out 0 c).coeff i = ((((y 0 c).coeff i).val : Nat) : ZMod q) := by
  have hR1 : 1 < R := by rw [hR]; nlinarith
  haveI : NeZero R := ⟨by omega⟩
  haveI : Fact (1 < R) := ⟨hR1⟩
  have hr := modulusSwitch_residue hn hQq hR x out h 0 c i
    (((y 0 c).coeff i).val : Int) ((ν 0 c).coeff i) ?_ (hν c i)
  · rw [hr]
    push_cast
    rfl
  · apply (ZMod.intCast_eq_intCast_iff' _ _ _).mp
    rw [hx]
    simp only [Int.ofNat_eq_natCast]
    have hc : ((((Qq : Int) : ExactPoly R n) • y + reduceMatrix R n 1 m ν) 0 c).coeff i =
        (Qq : ZMod R) * (y 0 c).coeff i + ((ν 0 c).coeff i : ZMod R) := by
      rw [Matrix.add_apply, Matrix.smul_apply, smul_eq_mul, Negacyclic.coeff_add, reduceMatrix,
        reducePoly_coeff_eq hR1 hn]
      congr 1
      rw [intCast_eq_algebraMap, Negacyclic.coeff_smul]
      push_cast
      rfl
    rw [ZMod.natCast_val, ZMod.intCast_cast, ZMod.cast_id', id, hc]
    push_cast
    rw [ZMod.natCast_zmod_val]

/-! ## Residues modulo the CRT prime -/

/-- `y` (over `R`) and `v` (over `Q`) have the same residues modulo `q`. -/
def ResidueEq {R Q m : Nat} (q : Nat) (y : ExactMatrix R n 1 m) (v : ExactMatrix Q n 1 m) : Prop :=
  ∀ c i, ((((y 0 c).coeff i).val : Nat) : ZMod q) = ((((v 0 c).coeff i).val : Nat) : ZMod q)

theorem val_cast_eq_cast {R q : Nat} [NeZero R] (x : ZMod R) :
    ((x.val : Nat) : ZMod q) = ZMod.cast x := (ZMod.cast_eq_val x).symm

/-- The residue modulo `q ∣ R` of a coefficient of `y + reduce E`. -/
theorem residue_add_reduce {R q m : Nat} (hn : 0 < n) (hR : 1 < R) (hqR : q ∣ R)
    (y : ExactMatrix R n 1 m) (E : ErrorMatrix n 1 m) (c : Fin m) (i : Fin n) :
    (((((y + reduceMatrix R n 1 m E) 0 c).coeff i).val : Nat) : ZMod q) =
      ((((y 0 c).coeff i).val : Nat) : ZMod q) + (((E 0 c).coeff i : Int) : ZMod q) := by
  haveI : NeZero R := ⟨by omega⟩
  rw [Matrix.add_apply, Negacyclic.coeff_add, reduceMatrix, reducePoly_coeff_eq hR hn,
    val_cast_eq_cast, val_cast_eq_cast, ZMod.cast_add hqR, ZMod.cast_intCast hqR]

/-- Adding the same integer error to both sides keeps equal residues. -/
theorem residueEq_add_reduce {R Q q m : Nat} (hn : 0 < n) (hR : 1 < R) (hQ : 1 < Q) (hqR : q ∣ R)
    (hqQ : q ∣ Q) {y : ExactMatrix R n 1 m} {v : ExactMatrix Q n 1 m} (h : ResidueEq q y v)
    (E : ErrorMatrix n 1 m) :
    ResidueEq q (y + reduceMatrix R n 1 m E) (v + reduceMatrix Q n 1 m E) := by
  intro c i
  rw [residue_add_reduce hn hR hqR, residue_add_reduce hn hQ hqQ, h c i]

/-- Reductions of coefficientwise congruent integer rows have equal residues. -/
theorem residueEq_reduce {R Q q m : Nat} (hn : 0 < n) (hR : 1 < R) (hQ : 1 < Q) (hqR : q ∣ R)
    (hqQ : q ∣ Q) {Y V : ErrorMatrix n 1 m}
    (h : ∀ c i, (Y 0 c).coeff i ≡ (V 0 c).coeff i [ZMOD q]) :
    ResidueEq q (reduceMatrix R n 1 m Y) (reduceMatrix Q n 1 m V) := by
  haveI : NeZero R := ⟨by omega⟩
  haveI : NeZero Q := ⟨by omega⟩
  intro c i
  rw [val_cast_eq_cast, val_cast_eq_cast, reduceMatrix, reduceMatrix, reducePoly_coeff_eq hR hn,
    reducePoly_coeff_eq hQ hn, ZMod.cast_intCast hqR, ZMod.cast_intCast hqQ]
  exact (ZMod.intCast_eq_intCast_iff _ _ _).2 (h c i)

/-! ## The CRT term -/

/-- **The CRT term.** The switched row `r` holds the residues of `y`, which agree with those of
`v` modulo `q`; its centered lift `l` times `K` is a CRT term of `v`. -/
theorem term_gen {R q Q m : Nat} {y : ExactMatrix R n 1 m} {v : ExactMatrix Q n 1 m}
    {r : ExactMatrix q n 1 m} {l term : ExactMatrix Q n 1 m} {K : Int}
    (hr : ∀ c i, (r 0 c).coeff i = ((((y 0 c).coeff i).val : Nat) : ZMod q))
    (hyv : ResidueEq q y v) (hl : centeredRebaseRuns r l)
    (ht : term = matrixMulScalarRight l (matrixPolynomial [K])) : TermOf q K term v :=
  termOf_of_rebase (fun c i ↦ (hr c i).trans (hyv c i)) hl ht

/-- **The online final combination of one instance**, from the forms of its summands to the CRT
term of the value `v` (`TermOf q K term v`): `ζ = b P - ((comb + fresh) + smudge) - H`, switched
from `R = Q_q q` to `q`, lifted centered to `Q` and multiplied by `K`. The value's residues are
those of `Y + E'` (`hyv`, e.g. from `residueEq_add_reduce`), and the noise
`h n B_b B_P + B_c + B_f + B_s + B_M` stays below `Q_q / 2`. -/
theorem final_gen (hn : 0 < n) {R q Qq Q h m : Nat} (hR : R = Qq * q) (hQq : 0 < Qq) (hq : 1 < q)
    {s : ExactPoly R n}
    {b : ExactMatrix R n 1 h} {Bk : Fin 2 → ExactMatrix R n 1 h} {eb : ErrorMatrix n 1 h} {Beb : Nat}
    (hb : b = maskPart s Bk + reduceMatrix R n 1 h eb) (heb : CoeffBound eb Beb)
    {P : ExactMatrix R n h m} {Pd : ErrorMatrix n h m} {BP : Nat}
    (hP : P = reduceMatrix R n h m Pd) (hPd : CoeffBound Pd BP)
    {Kc Kf Ks : Fin 2 → ExactMatrix R n 1 m} (hpre : ∀ k, Bk k * P = Kc k + Kf k + Ks k)
    {comb Y : ExactMatrix R n 1 m} {ec : ErrorMatrix n 1 m} {Bc : Nat}
    (hc : comb = maskPart s Kc - ((Qq : Int) : ExactPoly R n) • Y + reduceMatrix R n 1 m ec)
    (hec : CoeffBound ec Bc)
    {fresh : ExactMatrix R n 1 m} {E ef : ErrorMatrix n 1 m} {Bf : Nat}
    (hf : fresh = maskPart s Kf - ((Qq : Int) : ExactPoly R n) • reduceMatrix R n 1 m E +
      reduceMatrix R n 1 m ef) (hef : CoeffBound ef Bf)
    {smudge T H : ExactMatrix R n 1 m} {es : ErrorMatrix n 1 m} {Bs : Nat}
    (hs : smudge = maskPart s Ks - T + reduceMatrix R n 1 m es) (hes : CoeffBound es Bs)
    {M : ErrorMatrix n 1 m} {BM : Nat} (hTH : T - H = reduceMatrix R n 1 m M) (hM : CoeffBound M BM)
    (hnoise : 2 * (h * n * Beb * BP + Bc + Bf + Bs + BM) < Qq)
    {ζ : ExactMatrix R n 1 m}
    (hζ : ζ = matrixSub (matrixSub (matrixMul b P) (matrixAdd (matrixAdd comb fresh) smudge)) H)
    {r : ExactMatrix q n 1 m} (hsw : modulusSwitchRuns ζ r)
    {v l term : ExactMatrix Q n 1 m} {K : Int} (hl : centeredRebaseRuns r l)
    (ht : term = matrixMulScalarRight l (matrixPolynomial [K]))
    (hyv : ResidueEq q (Y + reduceMatrix R n 1 m E) v) : TermOf q K term v := by
  obtain ⟨ν, hν, hval⟩ := unmask_gen hn hb heb hP hPd hpre hc hec hf hef hs hes hTH hM
  refine term_gen (switch_gen hn hR hQq hq hsw (hζ.trans hval) (fun c i ↦ ?_)) hyv hl ht
  have := hν 0 c i
  rw [Int.abs_eq_natAbs]
  omega

/-! ## The fresh error of a component -/

/-- The PRF output of member `o` with `W` cells per member: bucket `0` reads the label exponents
`a` at its gather table `t` with selectors `s`, bucket `1` at `t'` with `s'`:
`z_o = Σ_k s_(oW+k) a(t_(oW+k)) + Σ_k s'_(oW+k) a(t'_(oW+k))`. -/
def prfZ (W : Nat) (a : Nat → Int) (s t s' t' : Nat → Int) (o : Nat) : Int :=
  ∑ k ∈ Finset.range W, s (o * W + k) * a (t (o * W + k)).toNat +
    ∑ k ∈ Finset.range W, s' (o * W + k) * a (t' (o * W + k)).toNat

/-- The PRF output of member `o` over `B` buckets (bucket `b`: selectors `s b`, gather table
`t b`, `W` cells per member): `z_o = Σ_b Σ_k s_b(oW+k) a(t_b(oW+k))`. -/
def prfZB (W : Nat) (a : Nat → Int) {B : Nat} (s t : Fin B → Nat → Int) (o : Nat) : Int :=
  ∑ b : Fin B, ∑ k ∈ Finset.range W, s b (o * W + k) * a (t b (o * W + k)).toNat

/-- A table given by its list of values (zero past the end). -/
def listTable (l : List Int) (k : Nat) : Int := l.getD k 0

theorem listTable_natAbs_le {l : List Int} {B : Nat} (h : ∀ x ∈ l, x.natAbs ≤ B) (k : Nat) :
    (listTable l k).natAbs ≤ B := by
  unfold listTable
  rw [List.getD_eq_getElem?_getD]
  cases hk : l[k]? with
  | none => simp
  | some x => exact h x (List.mem_of_getElem? hk)

/-- The sampled value of a binary-table PRF sampler on the PRF output `z`
(`GenLut.sampled_bin`): `(-1)^⌊z / W'⌋ g(⌊(P/2) (z mod W') / W'⌋)`. -/
def samplerVal (P W' : Nat) (g : Nat → Int) (z : Int) : Int :=
  (-1) ^ (z.toNat / W') * g ((P / 2) * (z.toNat % W') / W')

/-- The fresh error of a component: column `c` is `Σ_(i<n) X^i val(z_(c n + i))` (member
`c n + i` placed at `X^i`). -/
noncomputable def freshErr (n m : Nat) (val : Int → Int) (z : Nat → Int) : ErrorMatrix n 1 m :=
  fun _ c ↦ ∑ i : Fin n, Xe (n := n) i * ((val (z (c.val * n + i.val)) : Int) : ErrorPoly n)

theorem samplerVal_natAbs_le {P W' : Nat} {g : Nat → Int} {B : Nat}
    (hg : ∀ k, (g k).natAbs ≤ B) (z : Int) : (samplerVal P W' g z).natAbs ≤ B := by
  unfold samplerVal
  rw [Int.natAbs_mul, Int.natAbs_pow]
  simpa using hg _

theorem polyNorm_Xe_intCast (hn : 1 < n) (k : Nat) (C : Int) :
    polyNorm (Xe (n := n) k * (C : ErrorPoly n)) ≤ C.natAbs := by
  rw [mul_comm]
  refine (polyNorm_intCast_mul_le C _).trans ?_
  calc C.natAbs * polyNorm (Xe (n := n) k) ≤ C.natAbs * 1 :=
        Nat.mul_le_mul_left _ (polyNorm_Xe hn k)
    _ = C.natAbs := mul_one _

/-- Every coefficient of the fresh error is at most `n B` when every member value is at most
`B`. -/
theorem freshErr_coeff_le (hn : 1 < n) {m : Nat} {val : Int → Int} {B : Nat}
    (hval : ∀ z, (val z).natAbs ≤ B) (z : Nat → Int) (c : Fin m) (i : Fin n) :
    ((freshErr n m val z 0 c).coeff i).natAbs ≤ n * B := by
  refine (coeff_natAbs_le_polyNorm _ i).trans ?_
  unfold freshErr
  refine (polyNorm_sum_le _ _).trans ?_
  calc ∑ j : Fin n, polyNorm (Xe (n := n) j * ((val (z (c.val * n + j.val)) : Int) : ErrorPoly n))
      ≤ ∑ _j : Fin n, B := Finset.sum_le_sum fun j _ ↦ (polyNorm_Xe_intCast hn _ _).trans (hval _)
    _ = n * B := by simp

/-- The decode bound of the fresh error (`GenPart.errE_bound`): `4 |e| + 4 < Q` once
`4 n B + 4 < Q`. -/
theorem freshErr_bound (hn : 1 < n) {m Q : Nat} {val : Int → Int} {B : Nat}
    (hval : ∀ z, (val z).natAbs ≤ B) (hQ : 4 * (n * B) + 4 < Q) (z : Nat → Int) :
    ∀ c i, 4 * |(freshErr n m val z 0 c).coeff i| + 4 < (Q : Int) := by
  intro c i
  have h := freshErr_coeff_le hn hval z c i
  rw [Int.abs_eq_natAbs]
  omega

end RefreshTopProof.GenFinal
