import ExponentErrorForm
import RuntimeModulusSwitch
import ExponentRuntime

/-!
# Compact encodings

A compact encoding is published in two columns over `R_{pq}`: `c̃_j = s̄ Ã_{0j} + Ã_{1j} - p x_j + e_j`
for the mask `(s̄, 1)`, a `2 × 2` key `Ã` of `1 × 1` entries, and a payload `x_j` (the payload of
the full-gadget encoding times the unit pair). A reader expands it to the full gadget width at `q`:
`ModSwitch_{pq → q}(c̃_j G̃)` for the integer lift `G̃` of the gadget `g`, whose entries are
constants of absolute value at most `γ`.

`Rel.expandCompact` shows that the expansion is an encoding of `x_j g` under the expanded key
`ModSwitch_{pq → q}(Ã G̃)`, with the error bound `(B γ + (n + 2) ⌊p / 2⌋) / p`: the error `e G̃`
is at most `B γ` (a constant scales every coefficient), and dividing by `p` adds the rounding of
the row, of the key, and of the ternary mask times the key (`Rel.modDown`).
-/

namespace MxxPe

open Mxx.Primitives MxxRuntime MxxExponentLut

variable {n : Nat}

/-- The rounding residue of `modulusSwitch` from `p q` to `q`: every coefficient's canonical
residue minus `p` times its rounded quotient. -/
noncomputable def switchResidue {rows columns : Nat} (p q : Nat)
    (input : ExactMatrix (p * q) n rows columns) : ErrorMatrix n rows columns :=
  fun row column ↦ intPoly fun i ↦
    Int.ofNat ((input row column).coeff i).val -
      (p : Int) * ((Int.ofNat ((input row column).coeff i).val * (q : Int) + Int.ofNat (p * q / 2)) /
        Int.ofNat (p * q))

/-- A switch from `p q` to `q` (both odd): `scaleUp` of the output is the input minus the rounding
residue, which is at most `⌊p / 2⌋` per coefficient. -/
theorem modulusSwitchRuns_residue (hn : 0 < n) {p q rows columns : Nat} (hp : 1 < p)
    (hpodd : p % 2 = 1) {input : ExactMatrix (p * q) n rows columns}
    {output : ExactMatrix q n rows columns} (h : modulusSwitchRuns input output) :
    CoeffBound (switchResidue p q input) (p / 2) ∧
      scaleUpMatrix (p * q) p output =
        input - reduceMatrix (p * q) n rows columns (switchResidue p q input) := by
  obtain ⟨hq, _, _, hqodd, hout⟩ := h
  have hpq1 : 1 < p * q := by nlinarith
  have hround := fun (a : Int) ↦ roundedCanonical_scaled_error a (p : Int) (q : Int)
    (by omega) (by omega) (by exact_mod_cast hpodd) (by exact_mod_cast hqodd)
  refine ⟨?_, ?_⟩
  · intro row column i
    rw [switchResidue, intPoly_coeff hn]
    obtain ⟨h1, h2⟩ := hround (Int.ofNat ((input row column).coeff i).val)
    have e1 : Int.ofNat (p * q) = (p : Int) * q := by simp
    have e2 : Int.ofNat (p * q / 2) = (p : Int) * q / 2 := by
      rw [Int.ofNat_eq_natCast, Int.natCast_div]
      push_cast
      rfl
    rw [e1, e2]
    have hdiv : ((p : Int) / 2) = ((p / 2 : Nat) : Int) := by omega
    omega
  · funext row column
    simp only [scaleUpMatrix, Matrix.sub_apply, reduceMatrix, switchResidue]
    rw [hout]
    simp only [modulusSwitch]
    rw [polynomialOfCoefficients_eq_reduce, scaleUp_reducePoly rfl hq hn]
    rw [show ∀ r : ErrorPoly n, input row column - reducePoly (p * q) n r =
        reducePoly (p * q) n (canonicalLift (input row column) - r) from fun r ↦ by
      rw [map_sub, reducePoly_canonicalLift hpq1 hn]]
    rw [reducePoly_eq_iff hpq1 hn]
    intro i
    rw [intCast_mul_coeff, intPoly_coeff hn, Negacyclic.coeff_sub, canonicalLift_coeff hn,
      intPoly_coeff hn]
    set a := Int.ofNat ((input row column).coeff i).val
    set r := (a * (q : Int) + Int.ofNat (p * q / 2)) / Int.ofNat (p * q)
    have hmod : r % (q : Int) ≡ r [ZMOD (q : Int)] := Int.emod_emod_of_dvd r dvd_rfl
    have hscaled : (p : Int) * (r % (q : Int)) ≡ (p : Int) * r [ZMOD (p * q : Nat)] := by
      push_cast
      exact Int.ModEq.mul_left' hmod
    change (p : Int) * (r % (q : Int)) ≡ a - (a - (p : Int) * r) [ZMOD ((p * q : Nat) : Int)]
    have : a - (a - (p : Int) * r) = (p : Int) * r := by ring
    rw [this]
    exact hscaled

/-- A row of integer constants `z`, reduced modulo `q`: the gadget `g` at `q` and its integer lift
`G̃` at `p q`. -/
noncomputable def constRow (q : Nat) {m : Nat} (z : Fin m → Int) : ExactMatrix q n 1 m :=
  fun _ c ↦ reducePoly q n ((z c : Int) : ErrorPoly n)

/-- The expansion `ModSwitch_{pq → q}(c̃_j G̃)` of a compact encoding of the payload `p x_j` under
the key `Ã` is an encoding of `x_j g` under the expanded key `ModSwitch_{pq → q}(Ã G̃)`, for a
ternary mask, with the error bound `(B γ + (n + 2) ⌊p / 2⌋) / p`. -/
theorem Rel.expandCompact (hn : 0 < n) {p q m : Nat} (hp : 1 < p) (hpodd : p % 2 = 1)
    (hq : 1 < q) {sm : ErrorPoly n} (hsm : polyNorm sm ≤ 1)
    {At : Fin 2 → Fin 2 → ExactMatrix (p * q) n 1 1} {x : Fin 2 → ErrorPoly n}
    {ct : Fin 2 → ExactMatrix (p * q) n 1 1} {e : Fin 2 → ErrorMatrix n 1 1} {B : Nat}
    (he : ∀ j, CoeffBound (e j) B)
    (hc : RelE sm At (fun j _ _ ↦ reducePoly (p * q) n (((p : Int) : ErrorPoly n) * x j)) ct e)
    {z : Fin m → Int} {γ : Nat} (hz : ∀ c, (z c).natAbs ≤ γ)
    {Ah : Fin 2 → Fin 2 → ExactMatrix q n 1 m}
    (hA : ∀ k j, modulusSwitchRuns (matrixMulScalarLeft (At k j) (constRow (p * q) z)) (Ah k j))
    {ch : Fin 2 → ExactMatrix q n 1 m}
    (hch : ∀ j, modulusSwitchRuns (matrixMulScalarLeft (ct j) (constRow (p * q) z)) (ch j)) :
    Rel sm Ah (fun j ↦ reducePoly q n (x j) • constRow q z) ch
      ((B * γ + (n + 2) * (p / 2)) / p) := by
  have hpq1 : 1 < p * q := by nlinarith
  let zL : ErrorMatrix n 1 m := fun _ c ↦ ((z c : Int) : ErrorPoly n)
  have hG : constRow (p * q) z = reduceMatrix (p * q) n 1 m zL := rfl
  -- The relation at `p q`, after the constant action by `G̃`.
  have hbig : Rel sm (fun k j ↦ matrixMulScalarLeft (At k j) (constRow (p * q) z))
      (fun j ↦ matrixMulScalarLeft
        (fun _ _ ↦ reducePoly (p * q) n (((p : Int) : ErrorPoly n) * x j) :
          ExactMatrix (p * q) n 1 1) (constRow (p * q) z))
      (fun j ↦ matrixMulScalarLeft (ct j) (constRow (p * q) z)) (B * γ) := by
    refine ⟨fun j ↦ (e j 0 0) • zL, fun j ↦ ?_, fun j ↦ ?_⟩
    · intro row column i
      change ((e j 0 0 * ((z column : Int) : ErrorPoly n)).coeff i).natAbs ≤ B * γ
      rw [mul_comm, intCast_mul_coeff, Int.natAbs_mul, Nat.mul_comm]
      exact Nat.mul_le_mul (he j 0 0 i) (hz column)
    · beta_reduce
      simp only [matrixMulScalarLeft_eq, hc j, hG]
      funext row column
      simp only [Matrix.add_apply, Matrix.sub_apply, Matrix.smul_apply, smul_eq_mul, reduceMatrix,
        map_mul, zL]
      ring
  refine Rel.modDown hn (q := p * q) (p := p) (q' := q) rfl hpq1 hq (by omega) hbig hsm
    (fun j ↦ ?_) (fun k j ↦ ?_) (fun j ↦ ?_)
  · funext row column
    simp only [matrixMulScalarLeft_eq, scaleUpMatrix, Matrix.smul_apply, smul_eq_mul, constRow]
    rw [show reducePoly q n (x j) * reducePoly q n ((z column : Int) : ErrorPoly n) =
        reducePoly q n (x j * ((z column : Int) : ErrorPoly n)) from (map_mul _ _ _).symm,
      scaleUp_reducePoly rfl hq hn, ← map_mul]
    congr 1
    ring
  · obtain ⟨hr, heq⟩ := modulusSwitchRuns_residue hn hp hpodd (hA k j)
    exact ⟨_, hr, heq⟩
  · obtain ⟨hr, heq⟩ := modulusSwitchRuns_residue hn hp hpodd (hch j)
    exact ⟨_, hr, heq⟩

/-- The integer lift `G̃` of the constant row `z`, as an error matrix. -/
noncomputable def constLift {m : Nat} (z : Fin m → Int) : ErrorMatrix n 1 m :=
  fun _ c ↦ ((z c : Int) : ErrorPoly n)

/-- The error of the expansion of a compact encoding with the error `e` (see
`RelE.expandCompact`): `(e G̃ - r_c + s̄ r_0 + r_1) / p` for the switching residues of the row and
of the key. -/
noncomputable def expandErr (p q : Nat) (sm : ErrorPoly n) {m : Nat} (z : Fin m → Int)
    (At : Fin 2 → Fin 2 → ExactMatrix (p * q) n 1 1) (ct : Fin 2 → ExactMatrix (p * q) n 1 1)
    (e : Fin 2 → ErrorMatrix n 1 1) : Fin 2 → ErrorMatrix n 1 m :=
  fun j ↦ divInt (p : Int)
    ((e j 0 0) • constLift z - switchResidue p q (matrixMulScalarLeft (ct j) (constRow (p * q) z)) +
      sm • switchResidue p q (matrixMulScalarLeft (At 0 j) (constRow (p * q) z)) +
      switchResidue p q (matrixMulScalarLeft (At 1 j) (constRow (p * q) z)))

/-- `Rel.expandCompact` with the error named: the expansion is the encoding of `x_j g` under the
expanded key with the error `expandErr`, which is within `(B γ + (n + 2) ⌊p / 2⌋) / p`. -/
theorem RelE.expandCompact (hn : 0 < n) {p q m : Nat} (hp : 1 < p) (hpodd : p % 2 = 1)
    (hq : 1 < q) {sm : ErrorPoly n} (hsm : polyNorm sm ≤ 1)
    {At : Fin 2 → Fin 2 → ExactMatrix (p * q) n 1 1} {x : Fin 2 → ErrorPoly n}
    {ct : Fin 2 → ExactMatrix (p * q) n 1 1} {e : Fin 2 → ErrorMatrix n 1 1} {B : Nat}
    (he : ∀ j, CoeffBound (e j) B)
    (hc : RelE sm At (fun j _ _ ↦ reducePoly (p * q) n (((p : Int) : ErrorPoly n) * x j)) ct e)
    {z : Fin m → Int} {γ : Nat} (hz : ∀ c, (z c).natAbs ≤ γ)
    {Ah : Fin 2 → Fin 2 → ExactMatrix q n 1 m}
    (hA : ∀ k j, modulusSwitchRuns (matrixMulScalarLeft (At k j) (constRow (p * q) z)) (Ah k j))
    {ch : Fin 2 → ExactMatrix q n 1 m}
    (hch : ∀ j, modulusSwitchRuns (matrixMulScalarLeft (ct j) (constRow (p * q) z)) (ch j)) :
    RelE sm Ah (fun j ↦ reducePoly q n (x j) • constRow q z) ch (expandErr p q sm z At ct e) ∧
      ∀ j, CoeffBound (expandErr p q sm z At ct e j) ((B * γ + (n + 2) * (p / 2)) / p) := by
  have hpq1 : 1 < p * q := by nlinarith
  have hG : constRow (p * q) z = reduceMatrix (p * q) n 1 m (constLift z) := rfl
  let eG : Fin 2 → ErrorMatrix n 1 m := fun j ↦ (e j 0 0) • constLift z
  have hbig : RelE sm (fun k j ↦ matrixMulScalarLeft (At k j) (constRow (p * q) z))
      (fun j ↦ matrixMulScalarLeft
        (fun _ _ ↦ reducePoly (p * q) n (((p : Int) : ErrorPoly n) * x j) :
          ExactMatrix (p * q) n 1 1) (constRow (p * q) z))
      (fun j ↦ matrixMulScalarLeft (ct j) (constRow (p * q) z)) eG := by
    intro j
    beta_reduce
    simp only [matrixMulScalarLeft_eq, hc j, hG]
    funext row column
    simp only [Matrix.add_apply, Matrix.sub_apply, Matrix.smul_apply, smul_eq_mul, reduceMatrix,
      map_mul, eG, constLift]
    ring
  have heG : ∀ j, CoeffBound (eG j) (B * γ) := by
    intro j row column i
    change ((e j 0 0 * ((z column : Int) : ErrorPoly n)).coeff i).natAbs ≤ B * γ
    rw [mul_comm, intCast_mul_coeff, Int.natAbs_mul, Nat.mul_comm]
    exact Nat.mul_le_mul (he j 0 0 i) (hz column)
  have hres := fun k j ↦ modulusSwitchRuns_residue hn hp hpodd (hA k j)
  have hresc := fun j ↦ modulusSwitchRuns_residue hn hp hpodd (hch j)
  obtain ⟨v, hv, hpv⟩ := RelE.modDown hn (q := p * q) (p := p) (q' := q) rfl hpq1 hq hbig
    (T := fun j ↦ reducePoly q n (x j) • constRow q z)
    (fun j ↦ by
      funext row column
      simp only [matrixMulScalarLeft_eq, scaleUpMatrix, Matrix.smul_apply, smul_eq_mul, constRow]
      rw [show reducePoly q n (x j) * reducePoly q n ((z column : Int) : ErrorPoly n) =
          reducePoly q n (x j * ((z column : Int) : ErrorPoly n)) from (map_mul _ _ _).symm,
        scaleUp_reducePoly rfl hq hn, ← map_mul]
      congr 1
      ring)
    (fun k j ↦ (hres k j).2) (fun j ↦ (hresc j).2)
  have hvdef : expandErr p q sm z At ct e = v := by
    funext j
    exact divInt_eq hn (by exact_mod_cast (show p ≠ 0 by omega)) fun row column ↦ by
      rw [← hpv j row column]
      simp [eG]
  rw [hvdef]
  refine ⟨hv, fun j row column i ↦ ?_⟩
  -- `p v = w` with `|w| ≤ B γ + (n + 2) ⌊p / 2⌋`.
  have hw : CoeffBound (eG j - switchResidue p q (matrixMulScalarLeft (ct j) (constRow (p * q) z)) +
      sm • switchResidue p q (matrixMulScalarLeft (At 0 j) (constRow (p * q) z)) +
      switchResidue p q (matrixMulScalarLeft (At 1 j) (constRow (p * q) z)))
      (B * γ + (n + 2) * (p / 2)) := by
    have h1 := coeffBound_add (coeffBound_add (coeffBound_sub (heG j) (hresc j).1)
      (coeffBound_smul hn hsm (hres 0 j).1)) (hres 1 j).1
    refine coeffBound_mono h1 (le_of_eq ?_)
    ring
  have hwv := congrArg (fun z : ErrorPoly n ↦ (z.coeff i).natAbs) (hpv j row column)
  simp only [intCast_mul_coeff, Int.natAbs_mul, Int.natAbs_natCast] at hwv
  rw [Nat.le_div_iff_mul_le (by omega), Nat.mul_comm, ← hwv]
  exact hw row column i

/-! ## The gadget as a constant row -/

theorem coeff_algebraMap' {R : Type} [CommRing R] (hn : 0 < n) (x : R) (i : Fin n) :
    (algebraMap R (Negacyclic n R) x).coeff i = if i.val = 0 then x else 0 := by
  by_cases hR : Nontrivial R
  · have h := Negacyclic.coeff_smul x (1 : Negacyclic n R) i
    rw [mul_one] at h
    rw [h, show (1 : Negacyclic n R) =
      AdjoinRoot.root (negacyclicModulus n R) ^ ((⟨0, hn⟩ : Fin n) : Nat) by simp,
      Negacyclic.coeff_root_pow hn]
    split <;> split <;> simp_all [Fin.ext_iff, eq_comm]
  · have : Subsingleton R := not_nontrivial_iff_subsingleton.mp hR
    exact Subsingleton.elim _ _

theorem castExactMatrix_const {a b rows columns : Nat} (h : a = b) {x : ExactMatrix a n rows columns}
    (hx : ∀ i j, ∃ w : ZMod a, x i j = algebraMap (ZMod a) (ExactPoly a n) w) :
    ∀ i j, ∃ w : ZMod b, castExactMatrix h x i j = algebraMap (ZMod b) (ExactPoly b n) w := by
  subst h
  exact hx

/-- Every entry of a runtime gadget row is a constant. -/
theorem gadgetMatrixRuns_const {q m : Nat} {backend : BackendContext} {base digits : Int}
    {g : ExactMatrix q n 1 m} (h : gadgetMatrixRuns backend base digits g) :
    ∃ w : Fin m → ZMod q, ∀ c, g 0 c = algebraMap (ZMod q) (ExactPoly q n) (w c) := by
  obtain ⟨layout, -, -, -, hwidth, rfl⟩ := h
  have hfull : ∀ i j, ∃ w : ZMod q, fullGadgetMatrix (n := n) (rows := 1) layout i j =
      algebraMap (ZMod q) (ExactPoly q n) w := by
    unfold fullGadgetMatrix
    refine castExactMatrix_const _ fun i j ↦ ?_
    unfold regularGadgetUnflattened
    split
    · exact ⟨_, rfl⟩
    · exact ⟨0, by simp⟩
  choose w hw using fun c : Fin m ↦ hfull 0 (regularColumnIndex layout (Fin.cast hwidth c))
  refine ⟨w, fun c ↦ ?_⟩
  rw [castMatrixColumns_apply]
  exact hw c

/-- A row of constants at `q` and its centered rebase at `p` are the same integer constants `z`,
each at most `⌊q / 2⌋`. -/
theorem centeredRebase_constRow (hn : 0 < n) {q p m : Nat} (hq : 1 < q) {g : ExactMatrix q n 1 m}
    {w : Fin m → ZMod q} (hw : ∀ c, g 0 c = algebraMap (ZMod q) (ExactPoly q n) (w c)) :
    ∃ z : Fin m → Int, g = constRow q z ∧ centeredRebase (p := p) g = constRow p z ∧
      ∀ c, (z c).natAbs ≤ q / 2 := by
  let z : Fin m → Int := fun c ↦ if (w c).val ≤ q / 2 then ((w c).val : Int) else ((w c).val : Int) - q
  have hlift : ∀ c, centeredIntLift (g 0 c) = ((z c : Int) : ErrorPoly n) := by
    intro c
    apply negacyclic_ext hn
    intro i
    rw [hw c, centeredIntLift, intPoly_coeff hn, coeff_algebraMap' hn, intCast_eq_algebraMap,
      coeff_algebraMap' hn]
    split
    · rfl
    · simp
  refine ⟨z, ?_, ?_, fun c ↦ ?_⟩
  · funext r c
    obtain rfl : r = 0 := Subsingleton.elim _ _
    rw [← reducePoly_centeredIntLift hq hn (g 0 c), hlift c]
    rfl
  · funext r c
    obtain rfl : r = 0 := Subsingleton.elim _ _
    rw [centeredRebase_eq, hlift c]
    rfl
  · haveI : NeZero q := ⟨by omega⟩
    have hv := (w c).val_lt
    simp only [z]
    split <;> omega

end MxxPe
