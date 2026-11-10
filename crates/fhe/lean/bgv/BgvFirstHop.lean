import Scheme
import BgvHops

set_option backward.isDefEq.respectTransparency false

/-! The first game hop of the BGV security proof: the real IND-CPA world equals the real world of
the circular key-switching assumption.

`keygen_eq` is the closed form of the generated key generation: the public key `(a, s a + t e)`
and the relinearization key `(a', s a' + t e' + c_j L)`, where `L` is the graph's CRT lift of
`s²` from `Q` to `QP` and `c_j = P Q / Q_j`. Because `L ≡ s² (mod Q)` and `Q c_j ≡ 0 (mod QP)`,
`c_j L = c_j s²` over `QP` (`factor_sq`), which is the target of `RlweKeySwitchCircular`. Both
worlds are then rewritten into the common program `hop1`. -/

namespace BgvSecurity
open MxxSecurity MxxFhe GeneratedScheme Mxx.Primitives MxxRuntime


/-- The CRT basis of `Q`, one key-switching digit. -/
abbrev basisQ : List Nat := [18014398507892737, 18014398508138497, 18014398508400641]

/-- The gadget factor `P Q / Q_j` of each key column. -/
def factor : Fin 3 → Int :=
  ![23384026194045816390345886928489845842762721017857,
    23384026193726801671378034657273812936317868343297,
    23384026193386519304488586269947798086183317045249]

/-- The secret over `QP`: its reduction to the first prime, lifted back centered. -/
noncomputable def secQP (S : ExactMatrix Q n 1 1) : ExactMatrix QP n 1 1 :=
  centeredRebase (modulusReduce (p := 18014398507892737) S)

/-- The CRT lift of `s²` to `QP` that the key's targets scale. -/
noncomputable def sqLift (S : ExactMatrix Q n 1 1) : ExactMatrix QP n 1 1 := fun _ _ ↦
  polynomialOfCoefficients fun i ↦ rnsDigit basisQ 3 0 false
    (Int.ofNat ((matrixMulScalarLeft (modulusReduce (p := Q) S) (modulusReduce (p := Q) S) 0 0).coeff i).val)

/-- The second row of the relinearization key, as the graph computes it. -/
noncomputable def keyRaw (S : ExactMatrix Q n 1 1) (a e : ExactMatrix QP n 1 3) :
    ExactMatrix QP n 1 3 :=
  fun _ j ↦ secQP S 0 0 * a 0 j + tPoly * e 0 j + (factor j : ExactPoly QP n) * sqLift S 0 0

/-- The second row of the public key. -/
noncomputable def pkRaw (S a e : ExactMatrix Q n 1 1) : ExactMatrix Q n 1 1 :=
  fun _ _ ↦ S 0 0 * a 0 0 + tPoly * e 0 0
theorem keygen_eq :
    Keygen.generatedRoot.comp Keygen_params () =
      (do let ka ← (sampleResidue : Comp (ExactMatrix QP n 1 3))
          let vS ← sampleFamily ternLaw (1 * 1 * n)
          let ve ← sampleFamily errLaw (1 * 3 * n)
          let pa ← (sampleResidue : Comp (ExactMatrix Q n 1 1))
          let vE ← sampleFamily errLaw (1 * 1 * n)
          pure (rowsOf ka (keyRaw (mat11 ternLaw vS) ka (matrixOf fun i ↦ errLaw.clamp (ve i))),
            rowsOf pa (pkRaw (mat11 ternLaw vS) pa (mat11 errLaw vE)), mat11 ternLaw vS, ())) := by
  unfold Keygen.generatedRoot.comp
  simp only [sampleInterval, sampleGaussian, sampleMatrix_clamp, bind_assoc, pure_bind]
  refine bind_congr fun ka ↦ bind_congr fun vS ↦ bind_congr fun ve ↦ bind_congr fun pa ↦
    bind_congr fun vE ↦ ?_
  congr 1
  refine determined_eq ?_ ?_
  · refine ⟨⟨ka, mat11 ternLaw vS, modulusReduce (mat11 (q := Q) ternLaw vS),
    secQP (mat11 ternLaw vS), matrixOf (fun i ↦ errLaw.clamp (ve i)),
    modulusReduce (p := Q) (mat11 (q := Q) ternLaw vS), sqLift (mat11 ternLaw vS),
    fun _ j ↦ (factor j : ExactPoly QP n) * sqLift (mat11 ternLaw vS) 0 0,
    fun _ j ↦ (factor ⟨j.val, by omega⟩ : ExactPoly QP n) * sqLift (mat11 ternLaw vS) 0 0,
    rowsOf ka (keyRaw (mat11 ternLaw vS) ka (matrixOf fun i ↦ errLaw.clamp (ve i))), pa,
    mat11 errLaw vE, rowsOf pa (pkRaw (mat11 ternLaw vS) pa (mat11 errLaw vE))⟩,
    rfl, by simp only [mat11, ternLaw], by simp only [errLaw], rfl, by simp only [mat11, errLaw], ?_⟩
    dsimp only [Keygen.generatedRoot.body]
    refine ⟨rfl, interval_holds vS, ⟨by decide, by decide, rfl⟩, ⟨by decide, rfl⟩,
      gaussian_holds (by norm_num) (by norm_num) ve, ⟨by decide, dvd_refl _, rfl⟩, ?_, ?_, ?_, ?_,
      rfl, gaussian_holds (by norm_num) (by norm_num) vE, ?_, rfl⟩
    · refine ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, ?_⟩
      intro digit row outRow column h
      fin_cases digit; fin_cases row; fin_cases outRow; fin_cases column
      rfl
    · intro row column
      fin_cases row; fin_cases column <;> simp [factor, matrixScale]
    · intro row column
      fin_cases row; fin_cases column <;> simp [factor, matrixScale]
    · intro row column
      fin_cases row <;> fin_cases column <;> simp [rowsOf, keyRaw, matrixAdd, matrixScale, tPoly, Matrix.mul_apply, Int.cast_ofNat] <;> rfl
    · intro row column
      fin_cases row <;> fin_cases column <;> simp [rowsOf, pkRaw, matrixAdd, matrixScale, tPoly, Matrix.mul_apply, Int.cast_ofNat]
  · rintro outputs ⟨w, rfl, h1, h5, rfl, h19, h⟩
    dsimp only [Keygen.generatedRoot.body] at h
    obtain ⟨-, -, hred1, hreb, -, hred2, hup, hc1, hc2, hr1, -, -, hr2, rfl⟩ := h
    rw [concatRows_eq_rowsOf hr1, concatRows_eq_rowsOf hr2]
    have hS : w.w_1_0 = mat11 ternLaw vS := h1
    obtain ⟨-, -, -, -, -, -, -, -, hT⟩ := hup
    have hT0 : w.w_10_0 0 0 = sqLift (mat11 ternLaw vS) 0 0 := by
      rw [hT ⟨0, by decide⟩ 0 0 0 rfl, hred2.2.2, hS]
      rfl
    obtain ⟨g0, g1⟩ := concatColumns_one_one hc1
    obtain ⟨g0', g1', g2⟩ := concatColumns_two_one hc2
    have h3 : w.w_3_0 = secQP (mat11 ternLaw vS) := by rw [hreb.2, hred1.2.2, hS]; rfl
    have hE : w.w_5_0 = matrixOf fun i ↦ errLaw.clamp (ve i) := by simp only [h5, errLaw]
    have hE2 : w.w_19_0 = mat11 errLaw vE := by simp only [h19, mat11, errLaw]
    have hK : matrixAdd (matrixAdd (matrixMulScalarLeft w.w_3_0 w.w_0_0) (matrixScale 1032193 w.w_5_0))
        w.w_14_0 = keyRaw (mat11 ternLaw vS) w.w_0_0 (matrixOf fun i ↦ errLaw.clamp (ve i)) := by
      rw [h3, hE]
      funext r j
      fin_cases r; fin_cases j <;>
        simp [keyRaw, matrixAdd, matrixScale, tPoly, Matrix.mul_apply, Int.cast_ofNat, g0', g1', g2,
          g0, g1, hT0, factor] <;> rfl
    have hP : matrixAdd (matrixMulScalarLeft w.w_1_0 w.w_17_0) (matrixScale 1032193 w.w_19_0) =
        pkRaw (mat11 ternLaw vS) w.w_17_0 (mat11 errLaw vE) := by
      rw [hS, hE2]
      funext r j
      fin_cases r; fin_cases j
      simp [pkRaw, matrixAdd, matrixScale, tPoly, Matrix.mul_apply, Int.cast_ofNat]
    rw [hK, hP, hS]
/-- The clamped secret coefficients. -/
noncomputable def secOf (vS : Fin (1 * 1 * n) → Int) : Fin n → Int := fun k ↦ ternLaw.clamp (vS k)

theorem secOf_bound (vS : Fin (1 * 1 * n) → Int) (i : Fin n) : -1 ≤ secOf vS i ∧ secOf vS i ≤ 1 := by
  have := SamplerLaw.clamp_supports ternLaw (vS i)
  simpa [SamplerLaw.Supports, secOf] using this

theorem secOf_small (vS : Fin (1 * 1 * n) → Int) (i : Fin n) :
    2 * ((intPoly (secOf vS)).coeff i).natAbs < 18014398507892737 := by
  rw [intPoly_coeff (by decide)]
  have := secOf_bound vS i
  omega

theorem mat11_reduce {q : Nat} (vS : Fin (1 * 1 * n) → Int) :
    (mat11 (q := q) ternLaw vS) 0 0 = reducePoly q n (intPoly (secOf vS)) := by
  rw [mat11_coeffs]
  exact polynomialOfCoefficients_eq_reduce _

theorem secQP_eq (vS : Fin (1 * 1 * n) → Int) :
    secQP (mat11 ternLaw vS) 0 0 = reducePoly QP n (intPoly (secOf vS)) := by
  have h1 : modulusReduce (p := 18014398507892737) (mat11 (q := Q) ternLaw vS) 0 0 =
      reducePoly 18014398507892737 n (intPoly (secOf vS)) := by
    simp only [modulusReduce, mat11_reduce]
    exact modulusReduce_reducePoly (by decide) (by decide) (by decide) (by decide) _
  simp only [secQP, centeredRebase, h1]
  exact centeredRebase_reducePoly (by decide) (by decide) _ (secOf_small vS)

theorem modQ_eq (vS : Fin (1 * 1 * n) → Int) :
    modulusReduce (p := Q) (mat11 (q := Q) ternLaw vS) 0 0 = reducePoly Q n (intPoly (secOf vS)) := by
  simp only [modulusReduce, mat11_reduce]
  exact modulusReduce_reducePoly (by decide) (by decide) (dvd_refl _) (by decide) _

theorem sqLift_modEq (vS : Fin (1 * 1 * n) → Int) (i : Fin n) :
    (intPoly fun i ↦ rnsDigit basisQ 3 0 false (Int.ofNat ((matrixMulScalarLeft
      (modulusReduce (p := Q) (mat11 (q := Q) ternLaw vS))
      (modulusReduce (p := Q) (mat11 (q := Q) ternLaw vS)) 0 0).coeff i).val)).coeff i ≡
      (intPoly (secOf vS) * intPoly (secOf vS)).coeff i [ZMOD Q] := by
  rw [intPoly_coeff (by decide), rnsDigit_three_whole (by decide) (by decide) (by decide)]
  refine (crt_three_centered (by decide) (by decide) (by decide) (by decide) (by decide)
    (by decide) _).trans ?_
  simp only [matrixMulScalarLeft, modQ_eq, ← map_mul, Int.ofNat_eq_natCast]
  rw [reducePoly_coeff (by decide) (by decide), val_intCast_emod (by decide)]
  exact Int.mod_modEq _ _

theorem factor_dvd (j : Fin 3) : (QP : Int) ∣ factor j * Q := by
  fin_cases j <;> decide

theorem factor_sq (vS : Fin (1 * 1 * n) → Int) (j : Fin 3) :
    (factor j : ExactPoly QP n) * sqLift (mat11 ternLaw vS) 0 0 =
      (factor j : ExactPoly QP n) * (reducePoly QP n (intPoly (secOf vS)) *
        reducePoly QP n (intPoly (secOf vS))) := by
  rw [← map_mul, sqLift, polynomialOfCoefficients_eq_reduce, ← map_intCast (reducePoly QP n) (factor j),
    ← map_mul, ← map_mul, reducePoly_eq_iff (by decide) (by decide)]
  intro i
  rw [intCast_mul_coeff, intCast_mul_coeff, Int.modEq_iff_dvd]
  obtain ⟨k, hk⟩ := (Int.modEq_iff_dvd.mp (sqLift_modEq vS i))
  obtain ⟨m, hm⟩ := factor_dvd j
  refine ⟨m * k, ?_⟩
  rw [← mul_sub, hk, ← mul_assoc, hm, mul_assoc]
theorem pk_bridge (vS : Fin (1 * 1 * n) → Int) (pa e : ExactMatrix Q n 1 1) :
    pkRaw (mat11 ternLaw vS) pa e =
      (polyOfInts (secOf vS) : ExactPoly Q n) • pa + ((1032193 : Nat) : ExactPoly Q n) • e := by
  funext r j
  fin_cases r; fin_cases j
  simp only [pkRaw, Matrix.add_apply, Matrix.smul_apply, smul_eq_mul, mat11_coeffs, tPoly]
  push_cast
  rfl

/-- The targets `c_j s²` of the circular assumption's key. -/
noncomputable def keyTarget (s : Fin n → Int) : ExactMatrix QP n 1 3 :=
  fun _ c ↦ ((params_0.factors.get ⟨c.val, c.isLt⟩ : Nat) : ExactPoly QP n) * (polyOfInts s * polyOfInts s)

theorem key_bridge (vS : Fin (1 * 1 * n) → Int) (ka e : ExactMatrix QP n 1 3) :
    keyRaw (mat11 ternLaw vS) ka e =
      (polyOfInts (secOf vS) : ExactPoly QP n) • ka + ((1032193 : Nat) : ExactPoly QP n) • e +
        keyTarget (secOf vS) := by
  funext r j
  fin_cases r
  simp only [keyRaw, Matrix.add_apply, Matrix.smul_apply, smul_eq_mul, secQP_eq, factor_sq, tPoly,
    polyOfInts, polynomialOfCoefficients_eq_reduce]
  fin_cases j <;> simp [factor, params_0, keyTarget, polyOfInts, polynomialOfCoefficients_eq_reduce] <;> push_cast <;> ring
/-- The common form of the first hop: the key-generation samples in the circular game's order. -/
noncomputable def hop1 (A : Adversary scheme) (right : Bool) : Comp Bool :=
  sampleFamily ternLaw n >>= fun vS ↦ (sampleResidue : Comp (ExactMatrix Q n 1 1)) >>= fun pa ↦
  sampleFamily errLaw (1 * 1 * n) >>= fun vE ↦
  (sampleResidue : Comp (ExactMatrix QP n 1 3)) >>= fun ka ↦ sampleFamily errLaw (1 * 3 * n) >>= fun ve ↦
  A.choose (rowsOf pa (pkRaw (mat11 ternLaw vS) pa (mat11 errLaw vE)))
    (rowsOf ka (keyRaw (mat11 ternLaw vS) ka (matrixOf fun i ↦ errLaw.clamp (ve i)))) >>= fun σ ↦
  sampleFamily ternLaw (1 * 1 * n) >>= fun vU ↦ sampleFamily errLaw (1 * 1 * n) >>= fun v1 ↦
  sampleFamily errLaw (1 * 1 * n) >>= fun v2 ↦
  A.guess σ.2.2 (Ciphertext.ct0 (encOf (rowsOf pa (pkRaw (mat11 ternLaw vS) pa (mat11 errLaw vE)))
    (mat11 ternLaw vU) (mat11 errLaw v1) (mat11 errLaw v2)
    (encodeSlots (if right = true then σ.2.1 else σ.1))))

theorem real0_eq (A : Adversary scheme) (right : Bool) :
    realHop (uses.get ⟨0, by decide⟩) A right = hop1 A right := by
  simp only [realHop, uses, List.get, AssumptionId.real, RlweKeySwitchCircular.real,
    RlweKeySwitchCircular.publicKey]
  unfold MxxFhe.reduce
  simp only [simulator_0, bind_assoc, pure_bind, sim0_setup_eq, sim0_encrypt_eq]
  rw [show sampleFamily params_0.secretLaw params_0.ringDimension = sampleFamily ternLaw n from rfl,
    sampleFamily_clamp ternLaw n,
    show uniformRow params_0.modulus params_0.ringDimension 1 =
      (sampleResidue : Comp (ExactMatrix Q n 1 1)) from rfl,
    show sampleMatrix params_0.errorLaw params_0.modulus params_0.ringDimension 1 1 =
      sampleMatrix errLaw Q n 1 1 from rfl, sampleMatrix_clamp errLaw Q n 1 1,
    show uniformRow params_0.extendedModulus params_0.ringDimension params_0.factors.length =
      (sampleResidue : Comp (ExactMatrix QP n 1 3)) from rfl,
    show sampleMatrix params_0.errorLaw params_0.extendedModulus params_0.ringDimension 1
      params_0.factors.length = sampleMatrix errLaw QP n 1 3 from rfl, sampleMatrix_clamp errLaw QP n 1 3]
  simp only [bind_assoc, pure_bind, map_eq_bind_pure_comp, Function.comp_def]
  unfold hop1
  refine bind_congr fun vS ↦ bind_congr fun pa ↦ bind_congr fun vE ↦ bind_congr fun ka ↦
    bind_congr fun ve ↦ ?_
  rw [pk_bridge, key_bridge]
  rfl

theorem world_eq (A : Adversary scheme) (right : Bool) : world scheme A right =ᵈ hop1 A right := by
  unfold world
  simp only [scheme, keyGen, enc, keygen_eq, encrypt_eq, bind_assoc, pure_bind]
  unfold hop1
  refine (swap _ _ _).trans (EvalDistEq.bind_right _ fun vS ↦ ?_)
  refine (swap2 _ _ (sampleResidue : Comp (ExactMatrix Q n 1 1)) _).trans
    (EvalDistEq.bind_right _ fun pa ↦ ?_)
  exact swap2 _ _ (sampleFamily errLaw (1 * 1 * n)) _

theorem first_hop (A : Adversary scheme) (right : Bool) :
    world scheme A right =ᵈ realHop (uses.get ⟨0, by decide⟩) A right :=
  (world_eq A right).trans (EvalDistEq.of_evalDist_eq (congrArg evalDist (real0_eq A right).symm))
end BgvSecurity
