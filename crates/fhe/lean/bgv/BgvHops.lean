import Scheme
import BgvSetupForms

set_option backward.isDefEq.respectTransparency false

/-! The middle game hops of the BGV security proof. Each hop rewrites the random world of one
reduction and the real world of the next into one common program:

* `hop2`: the circular assumption's random world (a uniform key next to an honest public key)
  and the one-sample `DecisionRlwe` real world, whose simulator scales the sample by `t`;
* `hop3`: the one-sample random world (a uniform public key with honest ciphertexts) and the
  two-sample real world. -/

namespace BgvSecurity
open MxxSecurity MxxFhe GeneratedScheme Mxx.Primitives MxxRuntime

theorem idx_val_rows {n c : Nat} (r : Fin 2) (j : Fin c) (k : Fin n) :
    (finProdFinEquiv (finProdFinEquiv (r, j), k) : Fin (2 * c * n)).val =
      k.val + n * (j.val + c * r.val) := by
  simp [finProdFinEquiv]

theorem idx_val_row {n c : Nat} (r : Fin 1) (j : Fin c) (k : Fin n) :
    (finProdFinEquiv (finProdFinEquiv (r, j), k) : Fin (1 * c * n)).val = k.val + n * j.val := by
  simp [finProdFinEquiv]

/-- A `2 × c` sample is two independent `1 × c` rows. -/
theorem sampleMatrix_split_rows (law : SamplerLaw) (q c : Nat) :
    sampleMatrix law q n 2 c = (do
      let x ← sampleMatrix law q n 1 c
      let y ← sampleMatrix law q n 1 c
      pure (rowsOf x y)) := by
  have h : 2 * c * n = 1 * c * n + 1 * c * n := by ring
  rw [sampleMatrix_eq, sampleMatrix_eq, sampleFamily_cast law h, sampleFamily_add]
  simp only [map_eq_bind_pure_comp, bind_assoc, pure_bind, Function.comp_def]
  refine bind_congr fun x ↦ bind_congr fun y ↦ congrArg pure ?_
  funext r j
  unfold matrixOf rowsOf
  by_cases hr : r.val = 0
  · rw [if_pos hr]
    refine congrArg polynomialOfCoefficients (funext fun k ↦ ?_)
    have hidx : Fin.cast h (finProdFinEquiv (finProdFinEquiv (r, j), k)) =
        Fin.castAdd _ (finProdFinEquiv (finProdFinEquiv ((0 : Fin 1), j), k)) :=
      Fin.ext (by simp only [Fin.val_cast, Fin.val_castAdd, idx_val_rows, idx_val_row, hr]; simp)
    show Fin.append x y (Fin.cast h _) = _
    rw [hidx, Fin.append_left]
  · rw [if_neg hr]
    have hr1 : r.val = 1 := by omega
    refine congrArg polynomialOfCoefficients (funext fun k ↦ ?_)
    have hidx : Fin.cast h (finProdFinEquiv (finProdFinEquiv (r, j), k)) =
        Fin.natAdd _ (finProdFinEquiv (finProdFinEquiv ((0 : Fin 1), j), k)) :=
      Fin.ext (by simp only [Fin.val_cast, Fin.val_natAdd, idx_val_rows, idx_val_row, hr1]; ring)
    show Fin.append x y (Fin.cast h _) = _
    rw [hidx, Fin.append_right]

theorem swap2 {α β γ δ : Type} (mx : Comp α) (my : α → Comp β) (mz : Comp γ)
    (f : α → β → γ → Comp δ) :
    (mx >>= fun x ↦ my x >>= fun y ↦ mz >>= fun z ↦ f x y z) =ᵈ
      (mz >>= fun z ↦ mx >>= fun x ↦ my x >>= fun y ↦ f x y z) := by
  refine (EvalDistEq.bind_right mx fun x ↦ swap (my x) mz (f x)).trans ?_
  exact swap mx mz (fun x z ↦ my x >>= fun y ↦ f x y z)

theorem mat11_coeffs {q : Nat} (law : SamplerLaw) (v : Fin (1 * 1 * n) → Int) :
    (mat11 (q := q) law v) 0 0 = polyOfInts (fun k : Fin n ↦ law.clamp (v k)) := by
  unfold mat11 matrixOf polyOfInts
  refine congrArg polynomialOfCoefficients (funext fun k ↦ ?_)
  congr 2

theorem pk_join (a u : ExactMatrix Q n 1 1) :
    pkOfRow (joinCols a u) = rowsOf (matrixScale 1032193 a) (matrixScale 1032193 u) := by
  refine ext21 ?_ ?_ <;> rfl

theorem ctOfRow_zero (b : ExactMatrix Q n 1 2) (E : ExactMatrix Q n 1 1) :
    ctOfRow b E 0 0 = tPoly * b 0 0 := if_pos rfl
theorem ctOfRow_one (b : ExactMatrix Q n 1 2) (E : ExactMatrix Q n 1 1) :
    ctOfRow b E 1 0 = tPoly * b 0 1 + E 0 0 := if_neg (by decide)
theorem rowsOf_zero {q c : Nat} (x y : ExactMatrix q n 1 c) (j : Fin c) :
    rowsOf x y 0 j = x 0 j := if_pos rfl
theorem rowsOf_one {q c : Nat} (x y : ExactMatrix q n 1 c) (j : Fin c) :
    rowsOf x y 1 j = y 0 j := if_neg (by decide)
theorem joinCols_zero {q : Nat} (x y : ExactMatrix q n 1 1) : joinCols x y 0 0 = x 0 0 := if_pos rfl
theorem joinCols_one {q : Nat} (x y : ExactMatrix q n 1 1) : joinCols x y 0 1 = y 0 0 :=
  if_neg (by decide)

theorem ct_join (a u e1 e2 : ExactMatrix Q n 1 1) (v : Fin n → Int) (E : ExactMatrix Q n 1 1) :
    ctOfRow ((polyOfInts (fun k ↦ ternLaw.clamp (v k)) : ExactPoly Q n) • joinCols a u + joinCols e1 e2) E =
      encOf (rowsOf (matrixScale 1032193 a) (matrixScale 1032193 u)) (mat11 ternLaw v) e1 e2 E := by
  refine ext21 ?_ ?_
  · rw [encOf_zero, mat11_coeffs, ctOfRow_zero, rowsOf_zero, Matrix.add_apply, Matrix.smul_apply,
      smul_eq_mul, joinCols_zero, joinCols_zero, scale_apply]
    ring
  · rw [encOf_one, mat11_coeffs, ctOfRow_one, rowsOf_one, Matrix.add_apply, Matrix.smul_apply,
      smul_eq_mul, joinCols_one, joinCols_one, scale_apply]
    ring

/-- The common form of the third hop: the encryption randomness drawn before the adversary. -/
noncomputable def hop3 (A : Adversary scheme) (right : Bool) : Comp Bool :=
  sampleMatrix (.residue Q) Q n 1 1 >>= fun a ↦ sampleMatrix (.residue Q) Q n 1 1 >>= fun u ↦
  sampleFamily ternLaw (1 * 1 * n) >>= fun vU ↦ sampleFamily errLaw (1 * 1 * n) >>= fun v1 ↦
  sampleFamily errLaw (1 * 1 * n) >>= fun v2 ↦
  (sampleResidue : Comp (ExactMatrix QP n 2 3)) >>= fun ek ↦
  A.choose (rowsOf (matrixScale 1032193 a) (matrixScale 1032193 u)) ek >>= fun σ ↦
  A.guess σ.2.2 (Ciphertext.ct0 (encOf (rowsOf (matrixScale 1032193 a) (matrixScale 1032193 u))
    (mat11 ternLaw vU) (mat11 errLaw v1) (mat11 errLaw v2)
    (encodeSlots (if right = true then σ.2.1 else σ.1))))

theorem real2_eq (A : Adversary scheme) (right : Bool) :
    realHop (uses.get ⟨2, by decide⟩) A right = hop3 A right := by
  simp only [realHop, uses, List.get, AssumptionId.real, DecisionRlwe.real]
  unfold MxxFhe.reduce
  simp only [simulator_2, bind_assoc, pure_bind, sim2_setup_eq, sim2_encrypt_eq]
  rw [show uniformRow params_2.modulus params_2.ringDimension params_2.samples =
      sampleMatrix (.residue Q) Q n 1 2 from rfl, sampleMatrix_split_columns,
    show sampleMatrix params_2.errorLaw params_2.modulus params_2.ringDimension 1 params_2.samples =
      sampleMatrix errLaw Q n 1 2 from rfl, sampleMatrix_split_columns,
    show sampleFamily params_2.secretLaw params_2.ringDimension = sampleFamily ternLaw n from rfl,
    sampleFamily_clamp ternLaw n, sampleMatrix_clamp errLaw Q n 1 1]
  simp only [bind_assoc, pure_bind, map_eq_bind_pure_comp, Function.comp_def]
  unfold hop3
  refine bind_congr fun a ↦ bind_congr fun u ↦ bind_congr fun vU ↦ bind_congr fun v1 ↦
    bind_congr fun v2 ↦ bind_congr fun ek ↦ ?_
  rw [pk_join]
  refine bind_congr fun σ ↦ ?_
  rw [show (matrixOf fun i ↦ errLaw.clamp (v1 i) : ExactMatrix Q n 1 1) = mat11 errLaw v1 from rfl,
    show (matrixOf fun i ↦ errLaw.clamp (v2 i) : ExactMatrix Q n 1 1) = mat11 errLaw v2 from rfl,
    ct_join]
theorem random1_eq (A : Adversary scheme) (right : Bool) :
    randomHop (uses.get ⟨1, by decide⟩) A right =ᵈ hop3 A right := by
  simp only [randomHop, uses, List.get, AssumptionId.random, DecisionRlwe.random]
  unfold MxxFhe.reduce
  simp only [simulator_1, bind_assoc, pure_bind, sim1_setup_eq, sim1_encrypt_eq]
  unfold hop3
  refine EvalDistEq.bind_right _ fun a ↦ EvalDistEq.bind_right _ fun u ↦ ?_
  refine (swap2 _ _ (sampleFamily ternLaw (1 * 1 * n)) _).trans
    (EvalDistEq.bind_right _ fun vU ↦ ?_)
  refine (swap2 _ _ (sampleFamily errLaw (1 * 1 * n)) _).trans
    (EvalDistEq.bind_right _ fun v1 ↦ ?_)
  exact swap2 _ _ (sampleFamily errLaw (1 * 1 * n)) _

theorem public_key_to_ciphertext (A : Adversary scheme) (right : Bool) :
    randomHop (uses.get ⟨1, by decide⟩) A right =ᵈ realHop (uses.get ⟨2, by decide⟩) A right := by
  rw [real2_eq]
  exact random1_eq A right

theorem t_coprime : IsCoprime (1032193 : Int) (Q : Int) := by
  rw [Int.isCoprime_iff_gcd_eq_one]
  rfl

/-- The public key of the circular challenge. -/
noncomputable def pkC {q : Nat} (s : Fin n → Int) (a e : ExactMatrix q n 1 1) :
    ExactMatrix q n (1 + 1) 1 :=
  rowsOf a ((polyOfInts s : ExactPoly q n) • a + ((1032193 : Nat) : ExactPoly q n) • e)

theorem pk_circ {q : Nat} (s : Fin n → Int) (a e : ExactMatrix q n 1 1) :
    rowsOf (matrixScale 1032193 a) (matrixScale 1032193 ((polyOfInts s : ExactPoly q n) • a + e)) =
      pkC s (matrixScale 1032193 a) e := by
  unfold pkC
  refine congrArg (rowsOf _) ?_
  funext r j
  rw [scale_apply, Matrix.add_apply, Matrix.smul_apply, smul_eq_mul, Matrix.add_apply,
    Matrix.smul_apply, smul_eq_mul, Matrix.smul_apply, smul_eq_mul, scale_apply]
  push_cast
  ring

/-- The second hop after the secret and the public row `a`. -/
noncomputable def hop2Body (A : Adversary scheme) (right : Bool) (s : Fin n → Int)
    (a : ExactMatrix Q n 1 1) : Comp Bool :=
  sampleMatrix errLaw Q n 1 1 >>= fun e ↦
  sampleMatrix (.residue QP) QP n 1 3 >>= fun x ↦ sampleMatrix (.residue QP) QP n 1 3 >>= fun y ↦
  A.choose (pkC s a e) (rowsOf x y) >>= fun σ ↦
  sampleFamily ternLaw (1 * 1 * n) >>= fun vU ↦ sampleFamily errLaw (1 * 1 * n) >>= fun v1 ↦
  sampleFamily errLaw (1 * 1 * n) >>= fun v2 ↦
  A.guess σ.2.2 (Ciphertext.ct0 (encOf (pkC s a e) (mat11 ternLaw vU) (mat11 errLaw v1)
    (mat11 errLaw v2) (encodeSlots (if right = true then σ.2.1 else σ.1))))

/-- The common form of the second hop. -/
noncomputable def hop2 (A : Adversary scheme) (right : Bool) : Comp Bool :=
  sampleFamily ternLaw n >>= fun s ↦ (sampleResidue : Comp (ExactMatrix Q n 1 1)) >>= hop2Body A right s
theorem real1_eq (A : Adversary scheme) (right : Bool) :
    realHop (uses.get ⟨1, by decide⟩) A right =ᵈ hop2 A right := by
  simp only [realHop, uses, List.get, AssumptionId.real, DecisionRlwe.real]
  unfold MxxFhe.reduce
  simp only [simulator_1, bind_assoc, pure_bind, sim1_setup_eq, sim1_encrypt_eq]
  rw [show (sampleResidue : Comp (ExactMatrix QP n 2 3)) = sampleMatrix (.residue QP) QP n 2 3 from rfl,
    sampleMatrix_split_rows,
    show uniformRow params_1.modulus params_1.ringDimension params_1.samples =
      sampleMatrix (.residue Q) Q n 1 1 from rfl,
    show sampleFamily params_1.secretLaw params_1.ringDimension = sampleFamily ternLaw n from rfl,
    show sampleMatrix params_1.errorLaw params_1.modulus params_1.ringDimension 1 params_1.samples =
      sampleMatrix errLaw Q n 1 1 from rfl]
  simp only [bind_assoc, pure_bind]
  unfold hop2
  refine (swap _ _ _).trans (EvalDistEq.bind_right _ fun s ↦ ?_)
  refine EvalDistEq.trans ?_ (sampleResidue_scale (by decide) t_coprime (hop2Body A right s))
  refine EvalDistEq.of_evalDist_eq (congrArg evalDist (bind_congr fun a ↦ ?_))
  unfold hop2Body
  refine bind_congr fun e ↦ bind_congr fun x ↦ bind_congr fun y ↦ ?_
  rw [pk_circ (q := Q) s a e]

theorem random0_eq (A : Adversary scheme) (right : Bool) :
    randomHop (uses.get ⟨0, by decide⟩) A right =ᵈ hop2 A right := by
  simp only [randomHop, uses, List.get, AssumptionId.random, RlweKeySwitchCircular.random,
    RlweKeySwitchCircular.publicKey]
  unfold MxxFhe.reduce
  simp only [simulator_0, bind_assoc, pure_bind, sim0_setup_eq, sim0_encrypt_eq]
  unfold hop2 hop2Body
  exact EvalDistEq.of_evalDist_eq rfl

theorem circular_to_public_key (A : Adversary scheme) (right : Bool) :
    randomHop (uses.get ⟨0, by decide⟩) A right =ᵈ realHop (uses.get ⟨1, by decide⟩) A right :=
  (random0_eq A right).trans (real1_eq A right).symm

end BgvSecurity
