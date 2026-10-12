import TopSetup
import TopTables
import TopGadgets
import ExponentLut
import Stage_online_eval
import Stage_encrypt

/-!
# The instances of the merged stages

Instance `i0 < 8` of a merged stage is part `(i0 / 2, i0 % 2)`: the refresh chain `c = i0 / 2`
(keeping the CRT prime `q_c`) and the output component `j = i0 % 2`. Its constants are the exported
tables (`sel_k`) at `i0`; each abbreviation below is the exported expression, so types of the
stages and of the lemmas agree definitionally. The facts about them are proven once for every
instance (`interval_cases`), never inside the per-instance proof.
-/

namespace RefreshTopProof

open Mxx.Primitives MxxRuntime MxxExponentLut

theorem hi0 {i0 : Nat} [h : Fact (i0 < 8)] : i0 < 8 := h.out

theorem R_cases (i0 : Nat) (hi : i0 < 8) : (R i0 = Q1 ∧ qd i0 = q3) ∨ (R i0 = D3 ∧ qd i0 = q2) := by
  interval_cases i0 <;> decide

theorem q0_eqR (i0 : Nat) (hi : i0 < 8) : Q0 = qd i0 * R i0 := by
  rcases R_cases i0 hi with ⟨h1, h2⟩ | ⟨h1, h2⟩ <;> rw [h1, h2] <;> norm_num

theorem R_gt_one (i0 : Nat) (hi : i0 < 8) : 1 < R i0 := by
  interval_cases i0 <;> decide

theorem Mx_gt_one (i0 : Nat) (hi : i0 < 8) : 1 < Mx i0 := by
  interval_cases i0 <;> decide

theorem qc_gt_one (i0 : Nat) (hi : i0 < 8) : 1 < qc i0 := by
  interval_cases i0 <;> decide

theorem qd_pos (i0 : Nat) (hi : i0 < 8) : 0 < qd i0 := by
  interval_cases i0 <;> decide

theorem qd_gt_one (i0 : Nat) (hi : i0 < 8) : 1 < qd i0 := by
  interval_cases i0 <;> decide


theorem layoutR (i0 : Nat) (hi : i0 < 8) : ∃ L : RegularLayout (R i0),
    Backend.backend.regularLayout (R i0) N = some L ∧ L.droppedModuli = 0 ∧ L.base = 32768 ∧
      L.digitCount = 6 := by
  interval_cases i0 <;> exact ⟨_, by simp [Backend.backend]; rfl, rfl, rfl, rfl⟩

theorem half_two (i0 : Nat) (hi : i0 < 8) : 2 * halfZ (i0 : Int) = (R i0 : Int) + 1 := by
  interval_cases i0 <;> decide

theorem R_eq_pMx (i0 : Nat) (hi : i0 < 8) : R i0 = pp i0 * Mx i0 := by
  interval_cases i0 <;> decide

theorem R_eq_Qqqc (i0 : Nat) (hi : i0 < 8) : R i0 = Qq i0 * qc i0 := by
  interval_cases i0 <;> decide

theorem Qq_pos (i0 : Nat) (hi : i0 < 8) : 0 < Qq i0 := by
  interval_cases i0 <;> decide

/-- `p c' = Q_q k` with `k = c' / kdiv` (the action's scale). -/
theorem action_cert (i0 : Nat) (hi : i0 < 8) :
    ((pp i0 : Nat) : Int) * cpZ (i0 : Int) = QqZ (i0 : Int) * (cpZ (i0 : Int) / (kdiv i0 : Int)) := by
  interval_cases i0 <;> decide

theorem layoutMx (i0 : Nat) (hi : i0 < 8) : ∃ L : RegularLayout (Mx i0),
    Backend.backend.regularLayout (Mx i0) N = some L ∧ L.droppedModuli = 0 ∧ L.base = 32768 ∧
      L.digitCount = 4 := by
  interval_cases i0 <;> exact ⟨_, by simp [Backend.backend]; rfl, rfl, rfl, rfl⟩

theorem gadgetR (i0 : Nat) (hi : i0 < 8) :
    ∃ g : ExactMatrix (R i0) N 1 6, gadgetMatrixRuns Backend.backend 32768 6 g := by
  obtain ⟨L, hL, -, hb, hd⟩ := layoutR i0 hi
  exact gadget_exists hL (by simp [hb]) (by simp [hd]) (by simp [hd])

theorem gadgetMx (i0 : Nat) (hi : i0 < 8) :
    ∃ g : ExactMatrix (Mx i0) N 1 4, gadgetMatrixRuns Backend.backend 32768 4 g := by
  obtain ⟨L, hL, -, hb, hd⟩ := layoutMx i0 hi
  exact gadget_exists hL (by simp [hb]) (by simp [hd]) (by simp [hd])

theorem qc_dvd_R (i0 : Nat) (hi : i0 < 8) : qc i0 ∣ R i0 := by
  rw [R_eq_Qqqc i0 hi]; exact Dvd.intro_left _ rfl

theorem qc_dvd_Q0 (i0 : Nat) (hi : i0 < 8) : qc i0 ∣ Q0 := by
  interval_cases i0 <;> decide

theorem Qq_ge (i0 : Nat) (hi : i0 < 8) : 1152919668508929361 ≤ Qq i0 := by
  interval_cases i0 <;> decide

theorem R_Q1 (i0 : Nat) (hi : i0 < 6) : R i0 = Q1 := by
  interval_cases i0 <;> decide

/-- Gadgets of one modulus have one centered lift. -/
theorem gadget_lift {R' : Nat} (h : R' = Q1) {GR : ExactMatrix R' N 1 6} {G : ExactMatrix Q1 N 1 6}
    (hGR : gadgetMatrixRuns Backend.backend 32768 6 GR) (hG : gadgetMatrixRuns Backend.backend 32768 6 G) :
    ∀ c, centeredIntLift (GR 0 c) = centeredIntLift (G 0 c) := by
  subst h
  rw [gadgetMatrixRuns_unique hGR hG]
  intro c; rfl

/-- The link of instance `i0`'s residues: chains `0`–`2` decode at `Q₁` (the gadgets agree and the
action scales to `q₃`), chain `3`'s action and `q₃` vanish modulo its prime. -/
theorem link_case (i0 : Nat) (hi : i0 < 8) {GR : ExactMatrix (R i0) N 1 6} {G : ExactMatrix Q1 N 1 6}
    (hGR : gadgetMatrixRuns Backend.backend 32768 6 GR) (hG : gadgetMatrixRuns Backend.backend 32768 6 G) :
    ((∀ c, centeredIntLift (GR 0 c) = centeredIntLift (G 0 c)) ∧
      ((((cpZ (i0 : Int)) / (kdiv i0 : Int)) * k3Z (i0 : Int) * k1Z (i0 : Int) : Int) : ZMod (qc i0)) =
        ((q3 : Int) : ZMod (qc i0))) ∨
    (((((cpZ (i0 : Int)) / (kdiv i0 : Int)) * k3Z (i0 : Int) * k1Z (i0 : Int) : Int) : ZMod (qc i0)) = 0 ∧
      ((q3 : Int) : ZMod (qc i0)) = 0) := by
  by_cases h6 : i0 < 6
  · left
    refine ⟨gadget_lift (R_Q1 i0 h6) hGR hG, ?_⟩
    rw [ZMod.intCast_eq_intCast_iff']
    interval_cases i0 <;> decide
  · right
    rw [ZMod.intCast_zmod_eq_zero_iff_dvd, ZMod.intCast_zmod_eq_zero_iff_dvd]
    interval_cases i0 <;> first | omega | decide

/-- The instance's inverses of `2` and `4` modulo its decoding ring. -/
theorem half_mul (i0 : Nat) (hi : i0 < 8) :
    ((half i0 : Nat) : ExactPoly (R i0) N) * ((2 : Nat) : ExactPoly (R i0) N) = 1 :=
  inverse_mul (k := 1) (by interval_cases i0 <;> rfl)

theorem inv4_mul (i0 : Nat) (hi : i0 < 8) :
    ((inv4 i0 : Nat) : ExactPoly (R i0) N) * ((4 : Nat) : ExactPoly (R i0) N) = 1 :=
  inverse_mul (k := 3) (by interval_cases i0 <;> rfl)

/-- The output component (row) `j = i0 % 2` of instance `i0`. -/
def jr (i0 : Nat) : Fin 2 := ⟨i0 % 2, Nat.mod_lt _ (by norm_num)⟩

/-- A merged stage's selection of the instance's row among the two parts' wires. -/
theorem select_row {α : Type} {i0 : Nat} {idx : Int} {a b out : α}
    (hidx : idx = Int.ofNat i0) (h : MxxRuntime.select idx [a, b, a, b, a, b, a, b] out) :
    out = ![a, b] (jr i0) := by
  obtain ⟨p, hp, ho⟩ := h
  have hp' : p.val = i0 := Int.ofNat.inj (by rw [← hidx]; exact hp)
  subst ho
  have : jr i0 = ⟨p.val % 2, Nat.mod_lt _ (by norm_num)⟩ := Fin.ext (by simp [jr, hp'])
  rw [this]
  fin_cases p <;> simp

/-- A merged stage's selection of the other row. -/
theorem select_row' {α : Type} {i0 : Nat} {idx : Int} {a b out : α}
    (hidx : idx = Int.ofNat i0) (h : MxxRuntime.select idx [b, a, b, a, b, a, b, a] out) :
    out = ![a, b] (jr (i0 + 1)) := by
  obtain ⟨p, hp, ho⟩ := h
  have hp' : p.val = i0 := Int.ofNat.inj (by rw [← hidx]; exact hp)
  subst ho
  have : jr (i0 + 1) = ⟨(p.val + 1) % 2, Nat.mod_lt _ (by norm_num)⟩ := Fin.ext (by simp [jr, hp'])
  rw [this]
  fin_cases p <;> simp

/-- Numeric side conditions about the constants of an instance (`i0 < 8` in context). -/
macro "inst_num" : tactic => `(tactic| first
  | exact R_gt_one _ hi0
  | exact qd_pos _ hi0
  | exact qd_gt_one _ hi0
  | exact Mx_gt_one _ hi0
  | exact qc_gt_one _ hi0
  | exact R_eq_pMx _ hi0
  | exact R_eq_Qqqc _ hi0
  | exact Qq_pos _ hi0
  | (norm_num [-R, -qd]; done))

end RefreshTopProof
