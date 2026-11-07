import PePhase
import PeCompanions

/-!
# The anchor

The trapdoor setup publishes `B̄` with its trapdoor; the public setup lifts it to
`B_Q = P B̄ + r` over `R_{PQ}` with the rounding residue `r` of a uniform sample. The anchor
encoding `b = s B_Q + B^(2) + e_b` then rounds to `ModSwitch(b) - ModSwitch(B^(2)) = s B̄ + e`
(`Rel.modDown`), and its first entry expands to `s B'' + e''` (`expand_rel`).
-/

namespace PeProof

open Mxx.Primitives MxxRuntime MxxExponentLut

theorem setup_trap {tape : SampleTape}
    {out : ExactMatrix Q N 1 501 × TrapdoorValue (ExactMatrix Q N 1 501) Unit × Unit}
    (h : Stage_setup_trapdoor.generatedRoot Backend.backend tape [0] GeneratedClaim.stage_0_params () out) :
    out.1 = trapdoorPublic out.2.1 ∧ out.2.1.kind = .sampledSecret := by
  obtain ⟨w, h0, hout⟩ := h
  subst hout
  have hs := statTrapdoorSampleAt_sample h0
  exact ⟨(trapdoorSample_public hs).symm, trapdoorSample_sampled hs⟩

/-- `P` times the centered lift of `x` over `R_{PQ}` is `scaleUp x`. -/
theorem rebase_scale {x : ExactMatrix Q N 1 501} {y : ExactMatrix PQ N 1 501} (h : centeredRebaseRuns x y)
    {c : ExactMatrix PQ N 1 1} (hc : c 0 0 = (((P : Nat) : Int) : ExactPoly PQ N)) :
    matrixMulScalarLeft c y = scaleUpMatrix PQ P x := by
  let X : ErrorMatrix N 1 501 := fun row column ↦ centeredIntLift (x row column)
  have hy : y = reduceMatrix PQ N 1 501 X := centeredRebaseRuns_reduce h
  have hx : x = reduceMatrix Q N 1 501 X := (reduceMatrix_centeredIntLift hQ hN x).symm
  rw [hy, hx]
  funext row column
  simp only [matrixMulScalarLeft, hc, scaleUpMatrix, reduceMatrix]
  rw [scaleUp_reducePoly hPQ hQ hN, map_mul, map_intCast]

theorem matrixPolynomial_P :
    (matrixPolynomial [(67598865718973452911259282735194974463966008168342023559927392187609292957345997883634735633837025843140326117569327049993577673305918340947654355600136203031252562778990577685529163020701990351591297285624954881 : Int)] :
      ExactMatrix PQ N 1 1) 0 0 = (((P : Nat) : Int) : ExactPoly PQ N) := by
  simp [matrixPolynomial]

set_option maxRecDepth 100000 in
theorem setup_public {tape : SampleTape} {Bbar : ExactMatrix Q N 1 501} {BQ : ExactMatrix PQ N 1 501}
    (h : Stage_setup_public.generatedRoot tape [1] GeneratedClaim.stage_1_params Bbar BQ) :
    ∃ r : ErrorMatrix N 1 501, CoeffBound r (P / 2) ∧
      scaleUpMatrix PQ P Bbar = BQ - reduceMatrix PQ N 1 501 r := by
  obtain ⟨w, h2, -, h5, h6, hout⟩ := h
  have hres := MxxPe.modulusSwitchRuns_residue hN hP hPodd (p := P) (q := Q) h5
  refine ⟨MxxPe.switchResidue P Q w.w_4_0, hres.1, ?_⟩
  have e2 := rebase_scale h2 matrixPolynomial_P
  have e6 := rebase_scale h6 matrixPolynomial_P
  have e5 : scaleUpMatrix PQ P w.w_5_0 = w.w_4_0 - reduceMatrix PQ N 1 501 (MxxPe.switchResidue P Q w.w_4_0) :=
    hres.2
  rw [hout, e2, e6, e5]
  change _ = (scaleUpMatrix PQ P Bbar + (w.w_4_0 - (w.w_4_0 -
    reduceMatrix PQ N 1 501 (MxxPe.switchResidue P Q w.w_4_0)))) -
      reduceMatrix PQ N 1 501 (MxxPe.switchResidue P Q w.w_4_0)
  rw [sub_sub_cancel, add_sub_cancel_right]

/-- The rounding bound of the anchor. -/
def roundBound : Nat := (31 + (N + 2) * (P / 2)) / P

theorem switch_runs {rows columns : Nat} (x : ExactMatrix PQ N rows columns) :
    modulusSwitchRuns x (modulusSwitch (p := Q) x) :=
  ⟨hQ, ⟨P, by rw [hPQ, Nat.mul_comm]⟩, by norm_num, hQodd, rfl⟩

set_option maxRecDepth 100000 in
/-- The rounded anchor. -/
theorem anchor_round {s : ErrorPoly N} (hs : polyNorm s ≤ 1)
    {Bbar : ExactMatrix Q N 1 501} {BQ B2 bQ : ExactMatrix PQ N 1 501} {eb : ErrorMatrix N 1 501}
    (hb : bQ = reducePoly PQ N s • BQ + B2 + reduceMatrix PQ N 1 501 eb) (heb : CoeffBound eb 31)
    {r : ErrorMatrix N 1 501} (hr : CoeffBound r (P / 2))
    (hBQ : scaleUpMatrix PQ P Bbar = BQ - reduceMatrix PQ N 1 501 r) :
    ∃ Eb : ErrorMatrix N 1 501, CoeffBound Eb roundBound ∧
      modulusSwitch (p := Q) bQ - modulusSwitch (p := Q) B2 = reducePoly Q N s • Bbar + reduceMatrix Q N 1 501 Eb := by
  have hrel : Rel s (fun k _ ↦ ![BQ, B2] k) (fun _ ↦ 0) (fun _ ↦ bQ) 31 :=
    ⟨fun _ ↦ eb, fun _ ↦ heb, fun j ↦ by rw [hb]; simp⟩
  have hd := Rel.modDown hN hPQ (by norm_num) hQ (by norm_num) hrel hs
    (A' := fun k _ ↦ ![Bbar, modulusSwitch (p := Q) B2] k) (T := fun _ ↦ 0)
    (c' := fun _ ↦ modulusSwitch (p := Q) bQ)
    (fun _ ↦ by funext row column; simp [scaleUpMatrix, scaleUp_zero hPQ hQ hN])
    (fun k _ ↦ by
      fin_cases k
      · exact ⟨r, hr, hBQ⟩
      · have := MxxPe.modulusSwitchRuns_residue hN hP hPodd (p := P) (q := Q) (switch_runs B2)
        exact ⟨_, this.1, this.2⟩)
    (fun _ ↦ by
      have := MxxPe.modulusSwitchRuns_residue hN hP hPodd (p := P) (q := Q) (switch_runs bQ)
      exact ⟨_, this.1, this.2⟩)
  obtain ⟨e, he, hc⟩ := hd
  refine ⟨e 0, he 0, ?_⟩
  have h0 := hc 0
  simp only [Matrix.cons_val_zero, Matrix.cons_val_one, sub_zero] at h0
  rw [h0]
  abel

/-- The compressed anchor. -/
theorem anchor_compact {s : ErrorPoly N} (hs : polyNorm s ≤ 1)
    {BQ B2 bQ : ExactMatrix PQ N 1 501} {eb : ErrorMatrix N 1 501}
    (hb : bQ = reducePoly PQ N s • BQ + B2 + reduceMatrix PQ N 1 501 eb) (heb : CoeffBound eb 31) :
    ∃ Ec : ErrorMatrix N 1 44, CoeffBound Ec companionBound ∧
      expand (fun _ _ ↦ bQ 0 0) - expand (fun _ _ ↦ B2 0 0) =
        reducePoly Q N s • expand (fun _ _ ↦ BQ 0 0) + reduceMatrix Q N 1 44 Ec := by
  have hc : RelE (q := PQ) (m := 1) s
      (fun k (_ : Fin 2) ↦ ![(fun _ _ ↦ BQ 0 0 : ExactMatrix PQ N 1 1), fun _ _ ↦ B2 0 0] k)
      (fun _ _ _ ↦ reducePoly PQ N (((P : Nat) : Int) * (0 : ErrorPoly N)))
      (fun _ ↦ (fun _ _ ↦ bQ 0 0 : ExactMatrix PQ N 1 1))
      (fun _ ↦ (fun _ _ ↦ eb 0 0 : ErrorMatrix N 1 1)) := by
    intro j
    funext a c
    rw [Subsingleton.elim a 0, Subsingleton.elim c 0, hb]
    simp [reduceMatrix]
  have heb' : ∀ j : Fin 2, CoeffBound ((fun _ _ _ ↦ eb 0 0 : Fin 2 → ErrorMatrix N 1 1) j) 31 :=
    fun _ _ _ ↦ heb 0 0
  obtain ⟨h1, h2⟩ := expand_rel hs heb' hc
  refine ⟨_, h2 0, ?_⟩
  have := h1 0
  simp only [expandKey] at this
  rw [this]
  simp
  abel

end PeProof
