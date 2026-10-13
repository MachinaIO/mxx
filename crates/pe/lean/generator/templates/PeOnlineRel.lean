import PeOnline

/-!
# The encoding relation of the online output

For an attribute `x₀ ∈ {0, 1}`, the state after the product is an encoding of `X^{x₀}` under the
companion keys `M`, and the LUT of the table `(t₀, t₁)` maps it to an encoding of `t_{x₀}`:
`X^{x₀} D_1 + σ_K(X^{x₀}) D_K = t_{x₀}`. The requirement `F(x) = 0` gives `t_{x₀} = 0`.
-/

namespace PeProof

open Mxx.Primitives MxxRuntime MxxExponentLut MxxPe

/-- `X^8191` is negated by `σ_K`: `8191 · 8193 = 16383 (mod 2N)` and `X^8192 = -1`. -/
theorem substitution_rotation_8191 :
    substitution (n := N) (R := ZMod Q) 8193 hK (AdjoinRoot.root (negacyclicModulus N (ZMod Q)) ^ 8191) =
      -AdjoinRoot.root (negacyclicModulus N (ZMod Q)) ^ 8191 := by
  rw [substitution_root_pow, root_pow_mod_two_n]
  norm_num
  rw [show 16383 = 8192 + 8191 from rfl, pow_add, Negacyclic.root_pow_n]
  ring

/-- The two branches of the table `(t₀, t₁)` cancel on `X^{x₀}` when `t_{x₀} = 0`:
`X^{x₀} D_1 + σ_K(X^{x₀}) D_K = 0`. -/
theorem lutScalar_sum (xa : Fin 2) {t0 t1 : Int} (ht : ![t0, t1] xa = 0) :
    reducePoly Q N (Xe xa) * lutScalarOne t0 t1 0 0 +
      reducePoly Q N (substitution K hK (Xe xa)) * lutScalar t0 t1 (substituteMatrix K hK) 0 0 = 0 := by
  simp only [lutScalarOne, lutScalar, matrixMulScalarLeft, matrixAdd, matrixNeg, matrixPolynomial_single,
    rotationPolynomial, substituteMatrix, id, Matrix.add_apply, Matrix.neg_apply, liftInteger]
  rw [Xe, substitution_root_pow, reducePoly_Xe, reducePoly_Xe]
  fin_cases xa
  · subst ht
    simp only [Int.toNat_zero, pow_zero, map_one, map_neg, zero_mul, Int.cast_zero, zero_add, one_mul]
    erw [substitution_rotation_8191]
    push_cast
    ring
  · subst ht
    have hX : X (n := N) Q ^ (1 * K) = -X Q := by
      rw [one_mul, show K = N + 1 from rfl, pow_add, Negacyclic.root_pow_n, pow_one]
      ring
    simp only [hX, pow_one, Int.toNat_zero, pow_zero, map_one, Int.cast_zero, zero_mul, add_zero]
    ring

/-- The public scalar action `V = Dec(D g)` on a gadget target: `(x g) V = (x D) g`. -/
theorem gadget_scalar_action {D : ExactMatrix Q N 1 1} {g : ExactMatrix Q N 1 {{D}}}
    (hg : gadgetMatrixRuns Backend.backend {{BASE}} {{D}} g) {V : ExactMatrix Q N {{D}} {{D}}}
    (hV : gadgetDecomposeRuns Backend.backend {{BASE}} {{D}} (matrixMulScalarLeft D g) V) (x : ExactPoly Q N) :
    (x • g) * V = (x * D 0 0) • g := by
  rw [Matrix.smul_mul, (decompose_row hQ hN Backend.layout0 layoutQ rfl hg hV).1, matrixMulScalarLeft_eq,
    smul_smul]

/-- The digits of a public scalar action are within `base / 2`. -/
theorem scalar_digits {D : ExactMatrix Q N 1 1} {g : ExactMatrix Q N 1 {{D}}}
    (hg : gadgetMatrixRuns Backend.backend {{BASE}} {{D}} g) {V : ExactMatrix Q N {{D}} {{D}}}
    (hV : gadgetDecomposeRuns Backend.backend {{BASE}} {{D}} (matrixMulScalarLeft D g) V) :
    ∃ v : ErrorMatrix N {{D}} {{D}}, V = reduceMatrix Q N {{D}} {{D}} v ∧ CoeffBound v {{HALFBASE}} :=
  (decompose_row hQ hN Backend.layout0 layoutQ rfl hg hV).2

/-- The digits of key column `a` reconstruct it, within `base / 2`. -/
theorem key_digits {g : ExactMatrix Q N 1 {{D}}} (hg : gadgetMatrixRuns Backend.backend {{BASE}} {{D}} g)
    (A : Fin 2 → Fin 2 → ExactMatrix Q N 1 {{D}}) (a : Fin ({{D}} + {{D}})) (r : Fin 2) :
    g * keyDigits A a r = (fun _ _ ↦ joinedEntry (A r) a) ∧
      PreimageWithin (keyDigits A a r) {{HALFBASE}} :=
  decompose_entry hQ hN Backend.layout0 layoutQ rfl hg
    (entryDigits_runs hQ hN hg Backend.layout0 layoutQ rfl _)

/-- The error bound of the online output, from the bounds of the input encoding `B0`, the attribute
ciphertext `BB`, the digit companions `Bz`, and the helper `Bh`. -/
def onlineBound (B0 BB Bz Bh : Nat) : Nat :=
  {{D}} * N * ({{M}} * N * B0 * {{HALFBASE}} + {{M}} * ({{D}} * N * Bz * {{HALFBASE}} + {{D}} * N * Bz * {{HALFBASE}}) + 1 * BB) * {{HALFBASE}} +
    {{D}} * N * ({{M}} * N * Bh * {{HALFBASE}} + ({{M}} * N * ({{M}} * N * B0 * {{HALFBASE}} +
      {{M}} * ({{D}} * N * Bz * {{HALFBASE}} + {{D}} * N * Bz * {{HALFBASE}}) + 1 * BB) * {{HALFBASE}} +
      {{M}} * ({{D}} * N * Bz * {{HALFBASE}} + {{D}} * N * Bz * {{HALFBASE}}) + 1 * Bh)) * {{HALFBASE}}

set_option maxHeartbeats 16000000 in
/-- The online output is an encoding of `0` under the output key
`A_F = M V_1 + (H Dec(T)) V_K`, for an input encoding of `X⁰`, an attribute ciphertext of `X⁰`
(`x₀ = 0`), the digit companions of the product and of the automorphism, and the helper of `K`. -/
theorem online_rel {model : HashModel} {key : ByteArray} {s : ErrorPoly N}
    {g : ExactMatrix Q N 1 {{D}}} (hg : gadgetMatrixRuns Backend.backend {{BASE}} {{D}} g)
    {xa : Fin 2} {t0 t1 : Int} (ht : ![t0, t1] xa = 0) {V1 VK : ExactMatrix Q N {{D}} {{D}}}
    (hV1 : gadgetDecomposeRuns Backend.backend {{BASE}} {{D}} (matrixMulScalarLeft (lutScalarOne t0 t1) g) V1)
    (hVK : gadgetDecomposeRuns Backend.backend {{BASE}} {{D}}
      (matrixMulScalarLeft (lutScalar t0 t1 (substituteMatrix K hK)) g) VK)
    {cIn : Fin 2 → ExactMatrix Q N 1 {{D}}} {B0 : Nat}
    (hin : MxxExponentLut.Rel s (inputKey model key)
      (fun j ↦ reducePoly Q N (((1 : Nat) : Int) * 1 * unitPair s j) • g) cIn B0)
    {Ctop Cbot : Fin 2 → ExactMatrix Q N 1 {{D}}} {eB : Fin 2 → ErrorMatrix N 1 {{D}}} {BB : Nat}
    (hrhs : ∀ j, reducePoly Q N s • Ctop j + Cbot j =
      reducePoly Q N (Xe xa * unitPair s j) • g + reduceMatrix Q N 1 {{D}} (eB j))
    (heB : ∀ j, CoeffBound (eB j) BB)
    (hD : ∀ (j : Fin 2) (l : Fin {{D}}), ∃ D' : ExactMatrix Q N ({{D}} + {{D}}) 1,
      gadgetDecomposeRuns Backend.backend {{BASE}} {{D}} (pairColumn (Ctop j 0 l) (Cbot j 0 l)) D' ∧
        compactCenteredRebaseRuns {{HALFBASE}} D' (fun a _ ↦ actionDigits (joinRow Ctop) (joinRow Cbot) j a l))
    {Bm : Fin 2 → Fin {{D}} → Fin ({{D}} + {{D}}) → Fin 2 → Fin 2 → ExactMatrix Q N 1 {{D}}}
    {z0 z1 : Fin {{L}} → ExactMatrix Q N 1 {{D}}} {Bz : Nat}
    (hz : ∀ (j : Fin 2) (l : Fin {{D}}) (a : Fin ({{D}} + {{D}})), MxxExponentLut.Rel s (Bm j l a)
      (fun r ↦ (actionDigits (joinRow Ctop) (joinRow Cbot) j a l * reducePoly Q N (unitPair s r)) • g)
      ![z0 ⟨{{D}} * (j : Nat) + l + {{M}} * a, by omega⟩, z1 ⟨{{D}} * (j : Nat) + l + {{M}} * a, by omega⟩] Bz)
    {Mmul : Fin 2 → Fin 2 → ExactMatrix Q N 1 {{D}}}
    (hM : ∀ (k j : Fin 2) (l : Fin {{D}}), Mmul k j 0 l = (∑ a, (Bm j l a k 0 *
      keyDigits (inputKey model key) a 0 + Bm j l a k 1 * keyDigits (inputKey model key) a 1)) 0 0)
    {hrow : Fin 2 → ExactMatrix Q N 1 {{D}}} {Bh : Nat}
    (hh : MxxExponentLut.Rel s (helperKey model key) (fun j ↦ reducePoly Q N ((((1 : Nat) : Int) : ErrorPoly N) *
      unitPair (substitution K hK s) j) • g) hrow Bh)
    (hDK : ∀ (j : Fin 2) (l : Fin {{D}}), ∃ D' : ExactMatrix Q N ({{D}} + {{D}}) 1,
      gadgetDecomposeRuns Backend.backend {{BASE}} {{D}}
        (pairColumn (substituteMatrix K hK (helperKey model key 0 j) 0 l)
          ((substituteMatrix K hK (helperKey model key 1 j) - substituteMatrix K hK (hrow j)) 0 l)) D' ∧
        compactCenteredRebaseRuns {{HALFBASE}} D' (fun a _ ↦ actionDigits
          (joinRow fun j ↦ substituteMatrix K hK (helperKey model key 0 j))
          (joinRow fun j ↦ substituteMatrix K hK (helperKey model key 1 j) -
            substituteMatrix K hK (hrow j)) j a l))
    {Ba : Fin 2 → Fin {{D}} → Fin ({{D}} + {{D}}) → Fin 2 → Fin 2 → ExactMatrix Q N 1 {{D}}}
    {za0 za1 : Fin {{L}} → ExactMatrix Q N 1 {{D}}}
    (hza : ∀ (j : Fin 2) (l : Fin {{D}}) (a : Fin ({{D}} + {{D}})), MxxExponentLut.Rel (substitution K hK s) (Ba j l a)
      (fun r ↦ (actionDigits
          (joinRow fun j ↦ substituteMatrix K hK (helperKey model key 0 j))
          (joinRow fun j ↦ substituteMatrix K hK (helperKey model key 1 j) -
            substituteMatrix K hK (hrow j)) j a l *
        reducePoly Q N (unitPair (substitution K hK s) r)) • g)
      ![za0 ⟨{{D}} * (j : Nat) + l + {{M}} * a, by omega⟩, za1 ⟨{{D}} * (j : Nat) + l + {{M}} * a, by omega⟩] Bz)
    {T0 T1 : ExactMatrix Q N 1 {{M}}}
    (hT : ∀ (k j : Fin 2) (l : Fin {{D}}), halves (![T0, T1] k) j 0 l = (∑ a, (Ba j l a k 0 *
      keyDigits (routedKey Mmul) a 0 + Ba j l a k 1 * keyDigits (routedKey Mmul) a 1)) 0 0)
    (hDt : ∀ (j : Fin 2) (l : Fin {{D}}), ∃ D' : ExactMatrix Q N ({{D}} + {{D}}) 1,
      gadgetDecomposeRuns Backend.backend {{BASE}} {{D}} (pairColumn (halves T0 j 0 l) (halves T1 j 0 l)) D' ∧
        compactCenteredRebaseRuns {{HALFBASE}} D' (fun a _ ↦ actionDigits T0 T1 j a l)) :
    let A := inputKey model key
    let Emul : ExactMatrix Q N 1 {{M}} := fun _ jj ↦ ∑ a, mulCompanionSum A z0 z1 jj a
    let mulRow : Fin 2 → ExactMatrix Q N 1 {{D}} := fun j ↦
      pairMul cIn (actionDigits (joinRow Ctop) (joinRow Cbot) j) + halves Emul j
    let Elut : ExactMatrix Q N 1 {{M}} := fun _ jj ↦ ∑ a, lutCompanionSum Mmul za0 za1 jj a
    let top : Fin 2 → ExactMatrix Q N 1 {{D}} := fun j ↦ substituteMatrix K hK (helperKey model key 0 j)
    let bot : Fin 2 → ExactMatrix Q N 1 {{D}} := fun j ↦
      substituteMatrix K hK (helperKey model key 1 j) - substituteMatrix K hK (hrow j)
    let σc : Fin 2 → ExactMatrix Q N 1 {{D}} := fun j ↦ substituteMatrix K hK (mulRow j)
    MxxExponentLut.Rel s (fun k j ↦ Mmul k j * V1 + pairMul (helperKey model key k) (actionDigits T0 T1 j) * VK)
      (fun _ ↦ 0)
      (fun j ↦ mulRow j * V1 + (pairMul hrow (actionDigits T0 T1 j) +
        (pairMul σc (actionDigits (joinRow top) (joinRow bot) j) + halves Elut j)) * VK)
      (onlineBound B0 BB Bz Bh) := by
  intro A Emul mulRow Elut top bot σc
  have hg0 := centeredRebaseRuns_self hQ hN g
  -- The product.
  obtain ⟨f, hf, hE⟩ := companion_rows (q := Q) (m := {{D}}) (m' := {{D}}) (Dg := {{D}}) hN (sm := s)
    (g := g) (A := A) (D := actionDigits (joinRow Ctop) (joinRow Cbot)) (B := Bm)
    (z := fun j l a ↦ ![z0 ⟨{{D}} * (j : Nat) + l + {{M}} * a, by omega⟩,
      z1 ⟨{{D}} * (j : Nat) + l + {{M}} * a, by omega⟩])
    (bound := Bz) hz (u := fun a r ↦ keyDigits A a r) (fun a r ↦ (key_digits hg A a r).1)
    (u' := fun a r ↦ (key_digits hg A a r).2.choose)
    (fun a r ↦ (key_digits hg A a r).2.choose_spec.1)
    (fun a r ↦ (key_digits hg A a r).2.choose_spec.2) (E := fun j ↦ halves Emul j) (M := Mmul)
    (fun j l ↦ by
      simp only [halves, rowWindow, Emul, mulCompanionSum, Matrix.sum_apply]
      rfl)
    hM
  have hone : SmulGain (1 : ErrorPoly N) 1 := by
    have h : SmulGain (AdjoinRoot.root (negacyclicModulus N Int) ^ 0) 1 := smulGain_root_pow hN1 0
    rwa [pow_zero] at h
  have hmul := mul_private_dynamic hN hQ Backend.layout0 layoutQ rfl hg hg0 hin hD
    (T := fun j ↦ reducePoly Q N (Xe xa * unitPair s j) • g) hrhs heB hone hE hf
  -- The identity branch.
  obtain ⟨v1, hv1, hb1⟩ := scalar_digits hg hV1
  have hid := MxxPe.Rel.blockAction hN hmul v1 hb1
  rw [← hv1] at hid
  -- The branch of `K`: the switch, the mask switch, and the scalar action.
  have henc : MxxExponentLut.Rel s Mmul
      (fun j ↦ reducePoly Q N ((((1 : Nat) : Int) : ErrorPoly N) * Xe xa * unitPair s j) • g)
      (fun j ↦ pairMul cIn (actionDigits (joinRow Ctop) (joinRow Cbot) j) + halves Emul j)
      ({{M}} * N * B0 * {{HALFBASE}} + {{M}} * ({{D}} * N * Bz * {{HALFBASE}} + {{D}} * N * Bz * {{HALFBASE}}) + 1 * BB) := by
    refine MxxExponentLut.Rel.target_eq hmul ?_
    funext j
    simp
  obtain ⟨fa, hfa, hEa⟩ := companion_rows (q := Q) (m := {{D}}) (m' := {{D}}) (Dg := {{D}}) hN
    (sm := substitution K hK s) (g := g) (A := routedKey Mmul)
    (D := actionDigits (joinRow top) (joinRow bot)) (B := Ba)
    (z := fun j l a ↦ ![za0 ⟨{{D}} * (j : Nat) + l + {{M}} * a, by omega⟩,
      za1 ⟨{{D}} * (j : Nat) + l + {{M}} * a, by omega⟩])
    (bound := Bz) hza (u := fun a r ↦ keyDigits (routedKey Mmul) a r)
    (fun a r ↦ (key_digits hg (routedKey Mmul) a r).1)
    (u' := fun a r ↦ (key_digits hg (routedKey Mmul) a r).2.choose)
    (fun a r ↦ (key_digits hg (routedKey Mmul) a r).2.choose_spec.1)
    (fun a r ↦ (key_digits hg (routedKey Mmul) a r).2.choose_spec.2) (E := fun j ↦ halves Elut j)
    (M := fun k j ↦ halves (![T0, T1] k) j)
    (fun j l ↦ by
      simp only [halves, rowWindow, Elut, lutCompanionSum, Matrix.sum_apply]
      rfl)
    hT
  have hσone : SmulGain (substitution K hK (Xe (n := N) xa)) 1 := by
    rw [Xe, substitution_root_pow]
    exact smulGain_root_pow hN1 _
  have hsw := aut_switch hN hQ Backend.layout0 layoutQ rfl hg hg0 henc hK hK hKinv hKcoprime hh
    (top := top) (bot := bot) (fun _ ↦ rfl) (fun _ ↦ rfl) hDK hσone
    (E := fun j ↦ halves Elut j) (M := fun k j ↦ halves (![T0, T1] k) j) (fun j ↦ hEa j) hfa
  have hmask := mask_switch hN hQ Backend.layout0 layoutQ rfl hg hg0 hh hsw
    (Dt := actionDigits T0 T1) (fun j l ↦ by simpa using hDt j l)
  obtain ⟨vK, hvK, hbK⟩ := scalar_digits hg hVK
  have hbranch := MxxPe.Rel.blockAction hN hmask vK hbK
  rw [← hvK] at hbranch
  -- The sum of the two branches.
  have hsum := MxxExponentLut.Rel.add hid hbranch
  refine (MxxExponentLut.Rel.mono (MxxExponentLut.Rel.target_eq hsum ?_)) ?_
  rotate_left
  refine le_of_eq ?_
  unfold onlineBound
  ring
  funext j
  simp only [Pi.add_apply, map_one, one_smul]
  rw [smul_smul, gadget_scalar_action hg hV1, gadget_scalar_action hg hVK, ← add_smul]
  have hsum := congrArg (fun y ↦ reducePoly Q N (unitPair s j) * y) (lutScalar_sum xa ht)
  simp only [mul_zero] at hsum
  rw [show reducePoly Q N (Xe xa * unitPair s j) * lutScalarOne t0 t1 0 0 +
      reducePoly Q N (substitution K hK (Xe xa)) * reducePoly Q N (unitPair s j) *
        lutScalar t0 t1 (substituteMatrix K hK) 0 0 =
      reducePoly Q N (unitPair s j) * (reducePoly Q N (Xe xa) * lutScalarOne t0 t1 0 0 +
        reducePoly Q N (substitution K hK (Xe xa)) * lutScalar t0 t1 (substituteMatrix K hK) 0 0) by
    rw [map_mul]; ring, hsum]
  simp
