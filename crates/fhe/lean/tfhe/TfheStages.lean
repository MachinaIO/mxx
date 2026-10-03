import Stage_keygen
import Stage_encrypt_left
import Stage_encrypt_right
import Stage_nand
import Stage_decrypt
import Ideal
import RuntimeLemmas
import Backend

/-!
Each lemma restates one generated TFHE stage relation through integer witnesses: LWE phases are
integer equations modulo `q`, ring values are reductions of bounded integer polynomials.
-/

namespace MxxFheTfhe

open Mxx.Primitives MxxRuntime

abbrev q : Nat := 4294967296
abbrev Δ : Nat := 536870912
abbrev N : Nat := 2048
abbrev lweN : Nat := 630
abbrev Q : Nat := 4611255024196841473

/-- What an LWE encryption stage establishes: a mask in `[0, q)` and the body
`<a, s> + (2 m - 1) Δ + e mod q` with a bounded error. -/
def EncryptionFacts (secret : Fin lweN → Int) (message : Int)
    (outputs : (Fin lweN → Int) × Int × Unit) : Prop :=
  (∀ j, 0 ≤ outputs.1 j ∧ outputs.1 j < q) ∧
  ∃ error : Int, error.natAbs ≤ 2048 ∧
    outputs.2.1 = ((∑ j : Fin lweN, outputs.1 j * secret j) + (message * 2 - 1) * Δ + error) % q

/-- The centered error of one coefficient of a bounded ring Gaussian. -/
theorem gaussian_coefficient {sample : ExactMatrix Q N 1 1} {residue : Int}
    (hsample : gaussianSample (128 / 1) 2048 sample) (hresidue : extractCoefficient 0 sample residue) :
    ∃ error : Int, error.natAbs ≤ 2048 ∧
      (if 2 * residue ≤ Q then residue else residue - Q) = error := by
  obtain ⟨witness, hwitness, hbound⟩ := gaussianSample_bounded hsample
  obtain ⟨index, hindex, hres⟩ := hresidue
  have hval : index.val = 0 := by exact_mod_cast hindex
  have hindex0 : index = 0 := Fin.ext (by simpa using hval)
  subst hindex0
  set e := (witness 0 0).coeff 0
  have he : e.natAbs ≤ 2048 := (coeff_natAbs_le_polyNorm _ _).trans (by simpa using hbound 0 0)
  refine ⟨e, he, ?_⟩
  have : residue = e % Q := by
    rw [hres, hwitness, reduceMatrix_apply, reducePoly_coeff (by decide) (by decide),
      val_intCast_emod (by decide)]
  rw [this]
  exact centered_select_of_small (by decide) (by unfold Q; omega)

/-- `select` over two candidates indexed by a decided condition. -/
theorem select_two {a b output : Int} {c : Prop} [Decidable c]
    (h : select (if decide c then 1 else 0) [a, b] output) : output = if c then b else a := by
  obtain ⟨position, hposition, houtput⟩ := h
  rw [houtput]
  by_cases hc : c
  · simp only [hc, decide_true, if_true] at hposition ⊢
    have : position = 1 := Fin.ext (by simpa using hposition)
    subst this; rfl
  · simp only [hc, decide_false, Bool.false_eq_true, if_false] at hposition ⊢
    have : position = 0 := Fin.ext (by simpa using hposition)
    subst this; rfl

theorem dot_entry {mask secret : Fin lweN → Int} {value : Int}
    (h : familyGetStatic (intMatrixVectorProduct (outer := 1) false mask secret) 0 value) :
    value = ∑ j : Fin lweN, mask j * secret j := by
  obtain ⟨position, hposition, hvalue⟩ := h
  have : position = 0 := Fin.ext (by have := hposition; omega)
  subst this
  rw [hvalue, intMatrixVectorProduct_rows (by rfl)]
  simp

theorem encrypt_left_spec {hashModel : HashModel} {key : ByteArray} {secret : Fin lweN → Int} {message : Int} {outputs}
    (h : Stage_encrypt_left.generatedRoot hashModel { unit := () } (key, secret, message, ()) outputs) :
    EncryptionFacts secret message outputs := by
  obtain ⟨witness, hmask, hdot, hE, hres, _, _, _, hsel, _, hout⟩ := h
  subst hout
  refine ⟨hashIntFamily_range hmask, ?_⟩
  obtain ⟨error, herror, hcentered⟩ := gaussian_coefficient hE hres
  refine ⟨error, herror, ?_⟩
  have h22 := select_two hsel
  rw [mul_comm] at h22
  simp only [Q, Nat.cast_ofNat] at hcentered
  dsimp only
  simp only [Δ, q, Nat.cast_ofNat]
  rw [h22, hcentered, dot_entry hdot]

theorem encrypt_right_spec {hashModel : HashModel} {key : ByteArray} {secret : Fin lweN → Int} {message : Int} {outputs}
    (h : Stage_encrypt_right.generatedRoot hashModel { unit := () } (key, secret, message, ()) outputs) :
    EncryptionFacts secret message outputs := by
  obtain ⟨witness, hmask, hdot, hE, hres, _, _, _, hsel, _, hout⟩ := h
  subst hout
  refine ⟨hashIntFamily_range hmask, ?_⟩
  obtain ⟨error, herror, hcentered⟩ := gaussian_coefficient hE hres
  refine ⟨error, herror, ?_⟩
  have h22 := select_two hsel
  rw [mul_comm] at h22
  simp only [Q, Nat.cast_ofNat] at hcentered
  dsimp only
  simp only [Δ, q, Nat.cast_ofNat]
  rw [h22, hcentered, dot_entry hdot]

/-- Decryption: the canonical phase and its sign decoding. -/
theorem decrypt_spec {b : Int} {mask secret : Fin lweN → Int} {outputs}
    (h : Stage_decrypt.generatedRoot { unit := () } (b, mask, secret, ()) outputs) :
    outputs.2.1 = (b - ∑ j : Fin lweN, mask j * secret j) % q ∧
    outputs.1 = 1 - (if (if 2 * outputs.2.1 ≤ q then outputs.2.1 else outputs.2.1 - q) < 0
      then 1 else 0) := by
  obtain ⟨witness, hdot, _, _, _, _, hsel, hout⟩ := h
  subst hout
  have h16 := select_two hsel
  rw [mul_comm] at h16
  simp only [q, Nat.cast_ofNat]
  rw [dot_entry hdot] at h16 ⊢
  refine ⟨rfl, ?_⟩
  rw [h16]
  simp only [decide_eq_true_eq]

theorem ideal_spec {left right out : Int}
    (h : Ideal.generatedRoot { unit := () } (left, right, ()) out) : out = 1 - left * right := by
  obtain ⟨_, hout⟩ := h
  exact hout

/-- Exported coefficients of a binary sample are its integer witness's 0/1 coefficients. -/
theorem binary_coefficients {sample : ExactMatrix Q N 1 1} {values : Fin N → Int}
    (hsample : uniformIntervalSample 0 1 sample) (hvalues : polynomialValues false sample values) :
    ∃ Z : ErrorPoly N, sample 0 0 = reducePoly Q N Z ∧ (∀ c, Z.coeff c = 0 ∨ Z.coeff c = 1) ∧
      ∀ c, values c = Z.coeff c := by
  obtain ⟨_, witness, hwitness, hbounds⟩ := hsample
  refine ⟨witness 0 0, by rw [hwitness]; rfl, fun c ↦ ?_, fun c ↦ ?_⟩
  · have := hbounds 0 0 c; omega
  · simp only [polynomialValues, Bool.false_eq_true, if_false] at hvalues
    rw [hvalues c, hwitness, reduceMatrix_apply, reducePoly_coeff (by decide) (by decide),
      val_intCast_emod (by decide)]
    have := hbounds 0 0 c
    exact Int.emod_eq_of_lt (by omega) (by unfold Q; omega)

/-- The binary LWE secret drawn as the coefficients of a ring sample, reduced modulo two. -/
def LweSecretFacts (secret : Fin lweN → Int) : Prop := ∀ j, secret j = 0 ∨ secret j = 1

theorem backend_layout : FheBackend.backend.regularLayout Q N = some FheBackend.layout2 := by
  simp [FheBackend.backend]

theorem layout2_exact : FheBackend.layout2.droppedModuli = 0 := rfl

theorem layout2_digitCount : FheBackend.layout2.digitCount = 8 := by decide

/-- The ring-GSW external product reconstructs each row of the decomposed column: the first
eight digits rebuild row zero and the last eight rebuild row one, each digit within `128`. -/
theorem gadget_product {g : ExactMatrix Q N 1 8} {diff : ExactMatrix Q N 2 1}
    {D : ExactMatrix Q N 16 1} (hg : gadgetMatrixRuns FheBackend.backend 256 8 g)
    (hd : gadgetDecomposeRuns FheBackend.backend 256 8 diff D) :
    PreimageWithin D 128 ∧
    (∑ c : Fin 8, g 0 c * D (Fin.castAdd 8 c) 0) = diff 0 0 ∧
    (∑ c : Fin 8, g 0 c * D (Fin.natAdd 8 c) 0) = diff 1 0 := by
  obtain ⟨layout, hl, _, _, hw, hgeq⟩ := hg
  obtain ⟨layout', hl', _, _, hw', hdeq, _⟩ := hd
  rw [backend_layout] at hl hl'
  cases hl
  cases hl'
  have hrec := regularGadgetMatrix_reconstruct FheBackend.layout2 diff (by decide) (by decide)
    layout2_exact
  have hbound := regularDecomposeMatrix_bounded FheBackend.layout2 diff (by decide) (by decide)
  refine ⟨hdeq ▸ preimageWithin_castMatrixRows _ (by simpa using hbound), ?_, ?_⟩ <;>
  · rw [← congrFun (congrFun hrec _) 0, Matrix.mul_apply, hgeq, hdeq]
    simp only [castMatrixColumns_apply, castMatrixRows_apply]
    rw [← (finCongr (by rw [layout2_digitCount] : 8 + 8 = 2 * FheBackend.layout2.digitCount)).sum_comp]
    rw [Fin.sum_univ_add]
    simp only [regularGadgetMatrix_two_rows FheBackend.layout2 layout2_exact, finCongr_apply,
      Fin.val_cast, Fin.val_castAdd, Fin.val_natAdd]
    have hlow (x : Fin 8) : (x : Nat) / FheBackend.layout2.digitCount = 0 := by
      rw [layout2_digitCount]; exact Nat.div_eq_of_lt x.isLt
    have hhigh (x : Fin 8) : (8 + (x : Nat)) / FheBackend.layout2.digitCount = 1 := by
      rw [layout2_digitCount]; omega
    simp only [hlow, hhigh, Fin.val_zero, Fin.val_one, if_true, if_false, zero_ne_one,
      one_ne_zero, zero_mul, Finset.sum_const_zero, add_zero, zero_add]
    apply Finset.sum_congr rfl
    intro x _
    congr 2
    apply Fin.ext
    simp [layout2_digitCount, Nat.mod_eq_of_lt x.isLt]

/-- One ring-GSW bootstrapping key entry encrypting `secret` under the ring secret `z`. -/
def GswFacts (secret : Int) (z : ExactPoly Q N) (A B : ExactMatrix Q N 1 16) : Prop :=
  ∃ (g : ExactMatrix Q N 1 8) (a : ExactMatrix Q N 1 16) (E : ErrorMatrix N 1 16),
    gadgetMatrixRuns FheBackend.backend 256 8 g ∧ (∀ c, polyNorm (E 0 c) ≤ 2496) ∧
    ∀ c : Fin 8,
      A 0 (Fin.castAdd 8 c) = a 0 (Fin.castAdd 8 c) + (secret : ExactPoly Q N) * g 0 c ∧
      A 0 (Fin.natAdd 8 c) = a 0 (Fin.natAdd 8 c) ∧
      B 0 (Fin.castAdd 8 c) = z * a 0 (Fin.castAdd 8 c) + reducePoly Q N (E 0 (Fin.castAdd 8 c)) ∧
      B 0 (Fin.natAdd 8 c) = z * a 0 (Fin.natAdd 8 c) + reducePoly Q N (E 0 (Fin.natAdd 8 c)) +
        (secret : ExactPoly Q N) * g 0 c

/-- The flat key-switching key: entry `e` encrypts ring-secret coefficient `e / 8` times the
gadget value `4^(e % 8) * 2^16` under the LWE secret. -/
def KeySwitchFacts (secret : Fin lweN → Int) (Z : ErrorPoly N) (kska : Fin 10321920 → Int)
    (kskb : Fin 16384 → Int) : Prop :=
  (∀ i, 0 ≤ kska i ∧ kska i < q) ∧
  ∀ e : Fin 16384, ∃ error : Int, error.natAbs ≤ 2048 ∧
    kskb e = ((∑ j : Fin lweN, kska ⟨e.val * 630 + j.val, by
      have := e.isLt; have := j.isLt; simp only [lweN] at *; omega⟩ * secret j) +
        Z.coeff ⟨e.val / 8, by have := e.isLt; simp only [N]; omega⟩ * (4 ^ (e.val % 8) * 65536) +
        error) % q

/-- Exported coefficients of a bounded Gaussian are canonical residues of bounded integers. -/
theorem gaussian_values {sample : ExactMatrix Q N 1 1} {values : Fin N → Int}
    (hsample : gaussianSample (128 / 1) 2048 sample) (hvalues : polynomialValues false sample values)
    (c : Fin N) : ∃ error : Int, error.natAbs ≤ 2048 ∧ values c = error % Q := by
  obtain ⟨witness, hwitness, hbound⟩ := gaussianSample_bounded hsample
  simp only [polynomialValues, Bool.false_eq_true, if_false] at hvalues
  refine ⟨(witness 0 0).coeff c, (coeff_natAbs_le_polyNorm _ _).trans (by simpa using hbound 0 0), ?_⟩
  rw [hvalues c, hwitness, reduceMatrix_apply, reducePoly_coeff (by decide) (by decide),
    val_intCast_emod (by decide)]

/-- One entry of a family read at a dynamic index in range. -/
theorem familyGetDynamic_val {count : Nat} {family : Fin count → Int} {index output : Int}
    (h : familyGetDynamic family index output) :
    ∃ position : Fin count, (position.val : Int) = index ∧ output = family position := h

theorem keygen_spec {hashModel : HashModel} {key : ByteArray} {outputs}
    (h : Stage_keygen.generatedRoot FheBackend.backend hashModel { unit := () } key outputs) :
    ∃ Z : ErrorPoly N, (∀ c, Z.coeff c = 0 ∨ Z.coeff c = 1) ∧
      LweSecretFacts outputs.2.2.2.2.1 ∧
      (∀ i, GswFacts (outputs.2.2.2.2.1 i) (reducePoly Q N Z) (outputs.1 i) (outputs.2.1 i)) ∧
      KeySwitchFacts outputs.2.2.2.2.1 Z outputs.2.2.1 outputs.2.2.2.1 := by
  obtain ⟨witness, hS0, hS0v, _, hsecret, hZ, _, hgsw, hmask, hZv, hE0, hE0v, hE1, hE1v, hE2, hE2v,
    hE3, hE3v, hE4, hE4v, hE5, hE5v, hE6, hE6v, hE7, hE7v, _, hksk, hout⟩ := h
  subst hout
  obtain ⟨S0, _, hS0b, hS0c⟩ := binary_coefficients hS0 hS0v
  obtain ⟨Z, hZeq, hZb, hZc⟩ := binary_coefficients hZ hZv
  have hs (j : Fin lweN) : witness.w_2_0 j = 0 ∨ witness.w_2_0 j = 1 := by
    obtain ⟨_, hj⟩ := hsecret j
    rw [hj, hS0c]
    rcases hS0b ⟨j.val, by have := j.isLt; simp only [lweN, N] at *; omega⟩ with h0 | h1
    · left; rw [h0]; rfl
    · right; rw [h1]; rfl
  refine ⟨Z, hZb, hs, fun i ↦ ?_, ?_⟩
  · obtain ⟨a, g, w6, E, w12, _, hg, hcat1, hE, hcat2, hpair⟩ := hgsw i
    obtain ⟨Emat, hEmat, hEb⟩ := gaussianSample_bounded hE
    simp only [Prod.mk.injEq, and_true] at hpair
    obtain ⟨hA, hB⟩ := hpair
    refine ⟨g, a, Emat, hg, fun c ↦ by simpa using hEb 0 c, fun c ↦ ⟨?_, ?_, ?_, ?_⟩⟩
    · dsimp only
      rw [hA]
      simp only [matrixAdd, Matrix.add_apply, concatColumns_castAdd hcat1, matrixMulScalarLeft,
        liftInteger]
    · dsimp only
      rw [hA]
      simp only [matrixAdd, Matrix.add_apply, concatColumns_natAdd hcat1, Matrix.zero_apply,
        add_zero]
    · dsimp only
      rw [hB]
      simp only [matrixAdd, Matrix.add_apply, concatColumns_castAdd hcat2, matrixMulScalarLeft,
        hZeq, hEmat, reduceMatrix_apply, Matrix.zero_apply, add_zero]
    · dsimp only
      rw [hB]
      simp only [matrixAdd, Matrix.add_apply, concatColumns_natAdd hcat2, matrixMulScalarLeft,
        hZeq, hEmat, reduceMatrix_apply, liftInteger]
  · refine ⟨hashIntFamily_range hmask, fun e ↦ ?_⟩
    obtain ⟨w5, w32, w40, w42, w44, w46, w48, w50, w52, w54, w55, w63, _, _, _, hz, _, _, _, _, hγ,
      _, _, _, _, hf0, _, _, hf1, _, _, hf2, _, _, hf3, _, _, hf4, _, _, hf5, _, _, hf6, _, _, hf7,
      _, _, _, hres, _, _, _, hcen, _, hout⟩ := hksk e
    dsimp only at hz hγ hf0 hf1 hf2 hf3 hf4 hf5 hf6 hf7 hout
    -- Every error family entry is the residue of a bounded integer.
    have hfam {sample : ExactMatrix Q N 1 1} {values : Fin N → Int} {w : Int}
        (hs : gaussianSample (128 / 1) 2048 sample) (hv : polynomialValues false sample values)
        (hw : familyGetDynamic values (Int.ofNat e.val % 2048) w) :
        ∃ error : Int, error.natAbs ≤ 2048 ∧ w = error % Q := by
      obtain ⟨c, _, hc⟩ := familyGetDynamic_val hw
      rw [hc]
      exact gaussian_values hs hv c
    have hw55 : ∃ error : Int, error.natAbs ≤ 2048 ∧ w55 = error % Q := by
      obtain ⟨position, _, hpos⟩ := hres
      rw [hpos]
      fin_cases position
      · exact hfam hE0 hE0v hf0
      · exact hfam hE1 hE1v hf1
      · exact hfam hE2 hE2v hf2
      · exact hfam hE3 hE3v hf3
      · exact hfam hE4 hE4v hf4
      · exact hfam hE5 hE5v hf5
      · exact hfam hE6 hE6v hf6
      · exact hfam hE7 hE7v hf7
    obtain ⟨error, herror, h55⟩ := hw55
    have h63 := select_two hcen
    have hc := centered_select_of_small (q := Q) (value := error) (by decide)
      (by simp only [Q]; omega)
    simp only [Q, Nat.cast_ofNat] at hc h55 h63
    rw [mul_comm, h55, hc] at h63
    refine ⟨error, herror, ?_⟩
    have hgadget : w32 = 4 ^ (e.val % 8) * 65536 := by
      obtain ⟨position, hposition, hw32⟩ := hγ
      have he : e.val % 8 = position.val := by
        have := hposition
        simp only [Int.ofNat_eq_natCast] at this
        omega
      rw [hw32, he]
      fin_cases position <;> rfl
    have hz' : w5 = Z.coeff ⟨e.val / 8, by have := e.isLt; simp only [N]; omega⟩ := by
      obtain ⟨position, hposition, hw5⟩ := familyGetDynamic_val hz
      rw [hw5, hZc]
      congr 1
      apply Fin.ext
      show position.val = e.val / 8
      have := hposition
      simp only [Int.ofNat_eq_natCast] at this
      omega
    dsimp only
    rw [hout, intMatrixVectorProduct_rows (by rfl), hz', hgadget, h63]
    rfl

/-- The norm one blind-rotation step adds: sixteen digit products of a key error and a digit. -/
abbrev stepBound : Nat := 16 * (N * 2496 * 128)

/-- The accumulator phase after `k` rotations: the lookup table rotated by the accumulated
exponent plus a bounded integer error. -/
def RotationInvariant (z V : ExactPoly Q N) (exponent : Nat → Nat) (k : Nat)
    (acc : ExactMatrix Q N 2 1) : Prop :=
  ∃ E : ErrorPoly N, polyNorm E ≤ k * stepBound ∧
    acc 1 0 - z * acc 0 0 =
      AdjoinRoot.root (negacyclicModulus N (ZMod Q)) ^ exponent k * V + reducePoly Q N E

theorem reducePoly_root_pow (k : Nat) :
    reducePoly Q N (AdjoinRoot.root (negacyclicModulus N Int) ^ k) =
      AdjoinRoot.root (negacyclicModulus N (ZMod Q)) ^ k := by
  rw [map_pow]
  simp [reducePoly]

theorem multiplyMonomial_apply {rows columns : Nat} (input : ExactMatrix Q N rows columns)
    (exponent : Int) (row : Fin rows) (column : Fin columns) :
    multiplyMonomial input exponent row column = input row column *
      AdjoinRoot.root (negacyclicModulus N (ZMod Q)) ^ (exponent % (2 * (N : Int))).toNat := rfl

/-- One blind-rotation step: the external product with an encryption of a bit `s` multiplies
the phase by `X^(s e)` and adds the digit-weighted key error. -/
theorem rotation_step {z V : ExactPoly Q N} {secret : Int} (hsecret : secret = 0 ∨ secret = 1)
    {A B : ExactMatrix Q N 1 16} (hgsw : GswFacts secret z A B) {e : Int} (he : 0 ≤ e ∧ e < 4096)
    {exponent : Nat → Nat} {k : Nat} (hnext : exponent (k + 1) = exponent k + secret.toNat * e.toNat)
    {current : ExactMatrix Q N 2 1} (hcurrent : RotationInvariant z V exponent k current)
    {C : ExactMatrix Q N 2 16} (hC : concatRows A B C) {D : ExactMatrix Q N 16 1}
    (hD : gadgetDecomposeRuns FheBackend.backend 256 8
      (matrixSub (multiplyMonomial current e) current) D) :
    RotationInvariant z V exponent (k + 1) (matrixAdd current (matrixMul C D)) := by
  obtain ⟨g, a, E, hg, hE, hkey⟩ := hgsw
  obtain ⟨⟨Dw, hDw, hDb⟩, hrow0, hrow1⟩ := gadget_product hg hD
  obtain ⟨Eacc, hEacc, hphase⟩ := hcurrent
  have he' : (e % (2 * (N : Int))).toNat = e.toNat := by
    rw [Int.emod_eq_of_lt he.1 (by simp only [N]; omega)]
  have hC' := concatRows_one_one hC
  have hdiff (r : Fin 2) : (matrixSub (multiplyMonomial current e) current) r 0 =
      current r 0 * AdjoinRoot.root (negacyclicModulus N (ZMod Q)) ^ e.toNat - current r 0 := by
    simp only [matrixSub, Matrix.sub_apply, multiplyMonomial_apply, he']
  have hD' (c : Fin 16) : D c 0 = reducePoly Q N (Dw c 0) := by rw [hDw]; rfl
  -- The external product's phase: key errors times digits plus `secret (X^e - 1)` times the
  -- accumulator phase.
  have hext : (matrixMul C D) 1 0 - z * (matrixMul C D) 0 0 =
      (∑ c : Fin 16, reducePoly Q N (E 0 c * Dw c 0)) + (secret : ExactPoly Q N) *
        ((AdjoinRoot.root (negacyclicModulus N (ZMod Q)) ^ e.toNat - 1) *
          (current 1 0 - z * current 0 0)) := by
    have hsum (r : Fin 2) : (matrixMul C D) r 0 = ∑ c : Fin 16, C r c * D c 0 := Matrix.mul_apply
    have hcast (c : Fin 8) : C 1 (Fin.castAdd 8 c) * D (Fin.castAdd 8 c) 0 -
        z * (C 0 (Fin.castAdd 8 c) * D (Fin.castAdd 8 c) 0) =
        reducePoly Q N (E 0 (Fin.castAdd 8 c) * Dw (Fin.castAdd 8 c) 0) -
          z * (secret : ExactPoly Q N) * (g 0 c * D (Fin.castAdd 8 c) 0) := by
      rw [(hC' _).1, (hC' _).2, (hkey c).1, (hkey c).2.2.1, map_mul, ← hD']
      ring
    have hnat (c : Fin 8) : C 1 (Fin.natAdd 8 c) * D (Fin.natAdd 8 c) 0 -
        z * (C 0 (Fin.natAdd 8 c) * D (Fin.natAdd 8 c) 0) =
        reducePoly Q N (E 0 (Fin.natAdd 8 c) * Dw (Fin.natAdd 8 c) 0) +
          (secret : ExactPoly Q N) * (g 0 c * D (Fin.natAdd 8 c) 0) := by
      rw [(hC' _).1, (hC' _).2, (hkey c).2.1, (hkey c).2.2.2, map_mul, ← hD']
      ring
    have hsplit (f : Fin 16 → ExactPoly Q N) : ∑ c, f c =
        ∑ c : Fin 8, f (Fin.castAdd 8 c) + ∑ c : Fin 8, f (Fin.natAdd 8 c) := Fin.sum_univ_add (a := 8) (b := 8) f
    rw [hdiff] at hrow0 hrow1
    rw [hsum, hsum, Finset.mul_sum, ← Finset.sum_sub_distrib, hsplit,
      hsplit (fun c ↦ reducePoly Q N (E 0 c * Dw c 0))]
    simp only [hcast, hnat, Finset.sum_sub_distrib, Finset.sum_add_distrib, ← Finset.mul_sum,
      hrow0, hrow1]
    ring
  refine ⟨(if secret = 1 then AdjoinRoot.root (negacyclicModulus N Int) ^ e.toNat * Eacc
    else Eacc) + ∑ c : Fin 16, E 0 c * Dw c 0, ?_, ?_⟩
  · have hterm (c : Fin 16) : polyNorm (E 0 c * Dw c 0) ≤ N * 2496 * 128 :=
      (polyNorm_mul_le (by decide) _ _).trans
        (Nat.mul_le_mul (Nat.mul_le_mul_left _ (hE c)) (polyNorm_le_of_coeff (hDb c 0)))
    have hrot : polyNorm (if secret = 1 then AdjoinRoot.root (negacyclicModulus N Int) ^ e.toNat *
        Eacc else Eacc) ≤ k * stepBound := by
      split_ifs
      · exact (polyNorm_root_pow_mul_le (by decide) _ _).trans hEacc
      · exact hEacc
    have hsum := (polyNorm_sum_le Finset.univ fun c : Fin 16 ↦ E 0 c * Dw c 0).trans
      (Finset.sum_le_sum fun c _ ↦ hterm c)
    simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, smul_eq_mul] at hsum
    refine (polyNorm_add_le _ _).trans ?_
    rw [Nat.succ_mul]
    exact Nat.add_le_add hrot hsum
  · have hsplit : (matrixAdd current (matrixMul C D)) 1 0 -
        z * (matrixAdd current (matrixMul C D)) 0 0 = (current 1 0 - z * current 0 0) +
          ((matrixMul C D) 1 0 - z * (matrixMul C D) 0 0) := by
      simp only [matrixAdd, Matrix.add_apply]
      ring
    rw [hsplit, hext, hphase, hnext, map_add, map_sum]
    rcases hsecret with rfl | rfl
    · simp only [Int.cast_zero, zero_mul, add_zero, Int.toNat_zero, if_neg (by decide : (0 : Int) ≠ 1)]
      ring
    · simp only [Int.cast_one, one_mul, Int.toNat_one, if_true, map_mul, reducePoly_root_pow]
      ring

/-- Blind rotation: after all `630` steps the accumulator phase is the lookup table rotated by
the initial exponent plus the secret-weighted rounded masks, with the accumulated key error. -/
theorem blind_rotation_spec {z : ExactPoly Q N} {S : Fin lweN → Int} (hS : LweSecretFacts S)
    {A B : Fin lweN → ExactMatrix Q N 1 16} (hgsw : ∀ i, GswFacts (S i) z (A i) (B i))
    {table : ExactMatrix Q N 1 1} {e0 : Int} (he0 : 0 ≤ e0 ∧ e0 < 4096) {masks : Fin lweN → Int}
    {outputs} (h : Stage_nand.scope_tfhe_blind_rotation FheBackend.backend { unit := () }
      (0, multiplyMonomial table e0, masks, A, B, ()) outputs) :
    ∃ E : ErrorPoly N, polyNorm E ≤ lweN * stepBound ∧
      outputs.2.1 0 0 - z * outputs.1 0 0 =
        AdjoinRoot.root (negacyclicModulus N (ZMod Q)) ^ (e0.toNat + ∑ i : Fin lweN,
          (S i).toNat * (((masks i * 4096 + 2147483648) / 4294967296) % 4096).toNat) * table 0 0 +
        reducePoly Q N E := by
  obtain ⟨witness, hcat, _, hround, _, hiter, _, _, _, _, _, _, hs8, _, _, _, _, _, _, hs9, hout⟩ := h
  subst hout
  have hr (i : Fin lweN) : witness.w_6_0 i = ((masks i * 4096 + 2147483648) / 4294967296) % 4096 := by
    obtain ⟨_, _, hi⟩ := hround i
    exact hi
  let exponent : Nat → Nat := fun k ↦ e0.toNat + ∑ j ∈ Finset.range k,
    if hj : j < lweN then (S ⟨j, hj⟩).toNat * (witness.w_6_0 ⟨j, hj⟩).toNat else 0
  have hfinal := MxxIR.IterRuns.invariant (Invariant := RotationInvariant z (table 0 0) exponent) ?_ ?_ hiter
  · obtain ⟨E, hE, hphase⟩ := hfinal
    refine ⟨E, hE, ?_⟩
    dsimp only
    rw [sliceMatrix_entry hs8 (by decide) (by decide), sliceMatrix_entry hs9 (by decide) (by decide)]
    have hexp : exponent (Int.toNat 630) = e0.toNat + ∑ i : Fin lweN,
        (S i).toNat * (((masks i * 4096 + 2147483648) / 4294967296) % 4096).toNat := by
      show e0.toNat + _ = _
      rw [Finset.sum_range (fun j ↦ if hj : j < lweN then
        (S ⟨j, hj⟩).toNat * (witness.w_6_0 ⟨j, hj⟩).toNat else 0)]
      simp only [Fin.eta, hr]
      exact congrArg (e0.toNat + ·) (Finset.sum_congr rfl fun i _ ↦ dif_pos i.isLt)
    rw [← hexp]
    exact hphase
  · refine ⟨0, by simp, ?_⟩
    obtain ⟨h0, h1⟩ := concatRows_one_one hcat 0
    have he0' : (e0 % (2 * (N : Int))).toNat = e0.toNat := by
      rw [Int.emod_eq_of_lt he0.1 (by simp only [N]; omega)]
    rw [h0, h1]
    dsimp only
    rw [multiplyMonomial_apply, he0', mul_comm]
    simp [exponent]
  · intro i current next hcurrent hstep
    obtain ⟨w3, w5, w6, w8, w11, _, _, hA, _, _, hB, hC, _, _, he, hD, hnext⟩ := hstep
    obtain ⟨pA, hpA, rfl⟩ := hA
    obtain ⟨pB, hpB, rfl⟩ := hB
    obtain ⟨pe, hpe, rfl⟩ := he
    simp only [Int.ofNat_eq_natCast, Nat.cast_inj] at hpA hpB hpe
    have hi : i < lweN := hpA ▸ pA.isLt
    have hpA' : pA = ⟨i, hi⟩ := Fin.ext hpA
    have hpB' : pB = ⟨i, hi⟩ := Fin.ext hpB
    have hpe' : pe = ⟨i, hi⟩ := Fin.ext hpe
    subst hpA' hpB' hpe' hnext
    have hrange : 0 ≤ witness.w_6_0 ⟨i, hi⟩ ∧ witness.w_6_0 ⟨i, hi⟩ < 4096 := by
      rw [hr]; omega
    exact rotation_step (hS _) (hgsw _) hrange
      (by simp only [exponent, Finset.sum_range_succ, dif_pos hi]; ring) hcurrent hC hD

end MxxFheTfhe
