import Stage_keygen
import Stage_encrypt_left
import Stage_encrypt_right
import Stage_nand
import Stage_decrypt
import Ideal
import RuntimeLemmas
import Backend

/-!
Each lemma restates one generated TFHE stage relation in terms of the sampling tape: sampled
values are tape entries at the stage's sites, LWE phases are integer equations modulo `q`, and
ring values are reductions of integer polynomials.
-/

namespace MxxFheTfhe

open Mxx.Primitives MxxRuntime

abbrev q : Nat := 4294967296
abbrev Δ : Nat := 536870912
abbrev N : Nat := 1024
abbrev lweN : Nat := 630
abbrev Q : Nat := 5234636801

/-! ## Tape keys of the sampled values -/

abbrev bitLaw : SamplerLaw := .interval 0 1
abbrev ringLaw : SamplerLaw := .gaussian (156 / 1) 2496
abbrev lweLaw : SamplerLaw := .gaussian (131072 / 1) 2097152

/-- Coefficient `j` of the binary ring sample whose first `630` coefficients are the LWE secret. -/
def lweKey (j : Nat) : SampleKey := ⟨[0, 0], 0, 0, j, bitLaw⟩
/-- Coefficient `k` of the binary ring secret. -/
def ringKey (k : Nat) : SampleKey := ⟨[0, 3], 0, 0, k, bitLaw⟩
/-- Coefficient `k` of column `c` of the mask of bootstrapping key `i`. -/
def bskMaskKey (i c k : Nat) : SampleKey := ⟨[0, 4, i, 0], 0, c, k, .residue Q⟩
/-- Coefficient `k` of column `c` of the error of bootstrapping key `i`. -/
def bskErrorKey (i c k : Nat) : SampleKey := ⟨[0, 4, i, 10], 0, c, k, ringLaw⟩
/-- The error of key-switching entry `e`: coefficient `e % 1024` of error sample `e / 1024`. -/
def kskErrorKey (e : Nat) : SampleKey := ⟨[0, 17 + 2 * (e / 1024)], 0, 0, e % 1024, lweLaw⟩
/-- The error of the encryption at stage `stage`. -/
def encErrorKey (stage : Nat) : SampleKey := ⟨[stage, 13], 0, 0, 0, lweLaw⟩

def lweSecret (t : SampleTape) (j : Fin lweN) : Int := t (lweKey j)
noncomputable def ringSecret (t : SampleTape) : ErrorPoly N := intPoly fun k ↦ t (ringKey k)
noncomputable def bskMask (t : SampleTape) (i : Nat) : ExactMatrix Q N 1 12 :=
  tapeMatrix t [0, 4, i, 0] (.residue Q)
noncomputable def bskError (t : SampleTape) (i : Nat) : ErrorMatrix N 1 12 :=
  fun _ c ↦ intPoly fun k ↦ t (bskErrorKey i c k)
def kskError (t : SampleTape) (e : Nat) : Int := t (kskErrorKey e)
def encError (t : SampleTape) (stage : Nat) : Int := t (encErrorKey stage)

/-! ## Small relation helpers -/

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

/-- One entry of a family read at a dynamic index in range. -/
theorem familyGetDynamic_val {count : Nat} {family : Fin count → Int} {index output : Int}
    (h : familyGetDynamic family index output) :
    ∃ position : Fin count, (position.val : Int) = index ∧ output = family position := h

/-- The centered residue of the constant coefficient of a tape-read Gaussian is the tape entry. -/
theorem gaussian_tape_coefficient {tape : SampleTape} {site : List Nat}
    {sample : ExactMatrix Q N 1 1} {residue : Int}
    (hsample : gaussianSampleAt tape site (131072 / 1) 2097152 sample)
    (hresidue : extractCoefficient 0 sample residue) :
    |tape ⟨site, 0, 0, 0, lweLaw⟩| ≤ 2097152 ∧
      (if 2 * residue ≤ Q then residue else residue - Q) = tape ⟨site, 0, 0, 0, lweLaw⟩ := by
  obtain ⟨_, _, hbound, _, hout⟩ := hsample
  obtain ⟨index, hindex, hres⟩ := hresidue
  have hval : index.val = 0 := by exact_mod_cast hindex
  have hindex0 : index = 0 := Fin.ext (by rw [hval]; rfl)
  subst hindex0
  have hb : |tape ⟨site, 0, 0, 0, lweLaw⟩| ≤ 2097152 := hbound 0 0 0
  refine ⟨hb, ?_⟩
  have : residue = tape ⟨site, 0, 0, 0, lweLaw⟩ % Q := by
    rw [hres, hout, tapeMatrix_coeff (by decide) (by decide), val_intCast_emod (by decide)]
    rfl
  rw [this]
  exact centered_select_of_small (by decide) (by rw [Int.abs_eq_natAbs] at hb; unfold Q; omega)

/-- What an LWE encryption stage at `stage` establishes: a mask in `[0, q)` and the body
`<a, s> + (2 m - 1) Δ + e mod q`, with `e` the tape's encryption error. -/
def EncryptionFacts (tape : SampleTape) (stage : Nat) (secret : Fin lweN → Int) (message : Int)
    (outputs : (Fin lweN → Int) × Int × Unit) : Prop :=
  (∀ j, 0 ≤ outputs.1 j ∧ outputs.1 j < q) ∧ |encError tape stage| ≤ 2097152 ∧
    outputs.2.1 = ((∑ j : Fin lweN, outputs.1 j * secret j) + (message * 2 - 1) * Δ +
      encError tape stage) % q

theorem encrypt_left_spec {hashModel : HashModel} {tape : SampleTape} {key : ByteArray}
    {secret : Fin lweN → Int} {message : Int} {outputs}
    (h : Stage_encrypt_left.generatedRoot hashModel tape [1] { unit := () }
      (key, secret, message, ()) outputs) :
    EncryptionFacts tape 1 secret message outputs := by
  obtain ⟨witness, hmask, hdot, hE, hres, _, _, _, hsel, _, hout⟩ := h
  subst hout
  obtain ⟨hbound, hcentered⟩ := gaussian_tape_coefficient hE hres
  refine ⟨hashIntFamily_range hmask, hbound, ?_⟩
  have h22 := select_two hsel
  rw [mul_comm] at h22
  simp only [Q, Nat.cast_ofNat] at hcentered
  dsimp only
  simp only [Δ, q, Nat.cast_ofNat]
  rw [h22, hcentered, dot_entry hdot]
  rfl

theorem encrypt_right_spec {hashModel : HashModel} {tape : SampleTape} {key : ByteArray}
    {secret : Fin lweN → Int} {message : Int} {outputs}
    (h : Stage_encrypt_right.generatedRoot hashModel tape [2] { unit := () }
      (key, secret, message, ()) outputs) :
    EncryptionFacts tape 2 secret message outputs := by
  obtain ⟨witness, hmask, hdot, hE, hres, _, _, _, hsel, _, hout⟩ := h
  subst hout
  obtain ⟨hbound, hcentered⟩ := gaussian_tape_coefficient hE hres
  refine ⟨hashIntFamily_range hmask, hbound, ?_⟩
  have h22 := select_two hsel
  rw [mul_comm] at h22
  simp only [Q, Nat.cast_ofNat] at hcentered
  dsimp only
  simp only [Δ, q, Nat.cast_ofNat]
  rw [h22, hcentered, dot_entry hdot]
  rfl

/-- Decryption: the canonical phase and its sign decoding. -/
theorem decrypt_spec {tape : SampleTape} {b : Int} {mask secret : Fin lweN → Int} {outputs}
    (h : Stage_decrypt.generatedRoot tape [4] { unit := () } (b, mask, secret, ()) outputs) :
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

theorem ideal_spec {tape : SampleTape} {left right out : Int}
    (h : Ideal.generatedRoot tape [5] { unit := () } (left, right, ()) out) :
    out = 1 - left * right := by
  obtain ⟨_, hout⟩ := h
  exact hout

/-! ## Key generation -/

theorem backend_layout : FheBackend.backend.regularLayout Q N = some FheBackend.layout2 := by
  simp [FheBackend.backend]

theorem layout2_exact : FheBackend.layout2.droppedModuli = 0 := rfl

theorem layout2_digitCount : FheBackend.layout2.digitCount = 6 := by decide

/-- The gadget row of the bootstrapping keys. -/
noncomputable def gadgetRow : ExactMatrix Q N 1 6 :=
  castMatrixColumns (by rw [layout2_digitCount]) (regularGadgetMatrix (rows := 1) FheBackend.layout2)

theorem gadgetMatrixRuns_eq {g : ExactMatrix Q N 1 6}
    (hg : gadgetMatrixRuns FheBackend.backend 64 6 g) : g = gadgetRow := by
  obtain ⟨layout, hl, _, _, _, hgeq⟩ := hg
  rw [backend_layout] at hl
  cases hl
  rw [hgeq]
  rfl

/-- The mask row of bootstrapping key `i`: `a + [s g | 0]`. -/
noncomputable def gswA (t : SampleTape) (i : Fin lweN) : ExactMatrix Q N 1 12 := fun r c ↦
  bskMask t i r c + if h : c.val < 6 then (lweSecret t i : ExactPoly Q N) * gadgetRow 0 ⟨c.val, h⟩
    else 0

/-- The body row of bootstrapping key `i`: `z a + E + [0 | s g]`. -/
noncomputable def gswB (t : SampleTape) (i : Fin lweN) : ExactMatrix Q N 1 12 := fun r c ↦
  reducePoly Q N (ringSecret t) * bskMask t i r c + reducePoly Q N (bskError t i r c) +
    if h : 6 ≤ c.val then (lweSecret t i : ExactPoly Q N) * gadgetRow 0 ⟨c.val - 6, by omega⟩
    else 0

theorem bit_values {tape : SampleTape} {site : List Nat} {w : ExactMatrix Q N 1 1}
    {values : Fin N → Int} (hw : uniformIntervalSampleAt tape site 0 1 w)
    (hv : polynomialValues false w values) (k : Fin N) :
    values k = tape ⟨site, 0, 0, k, bitLaw⟩ ∧
      (tape ⟨site, 0, 0, k, bitLaw⟩ = 0 ∨ tape ⟨site, 0, 0, k, bitLaw⟩ = 1) := by
  obtain ⟨_, hb, hout⟩ := hw
  have hk : 0 ≤ tape ⟨site, 0, 0, k, bitLaw⟩ ∧ tape ⟨site, 0, 0, k, bitLaw⟩ ≤ 1 := hb 0 0 k
  simp only [polynomialValues, Bool.false_eq_true, if_false] at hv
  refine ⟨?_, by omega⟩
  rw [hv k, hout]
  exact tapeMatrix_coeff_val (by decide) (by decide) tape site _ 0 0 k hk.1
    (hk.2.trans_lt (by norm_num))

theorem ringSecret_coeff (t : SampleTape) (k : Fin N) : (ringSecret t).coeff k = t (ringKey k) :=
  intPoly_coeff (by decide) _ k

/-- The centered key-switching error read from one Gaussian error sample at index `index`. -/
theorem ksk_error_value {tape : SampleTape} {site : List Nat} {sample : ExactMatrix Q N 1 1}
    {values : Fin N → Int} {index : Fin N}
    (hs : gaussianSampleAt tape site (131072 / 1) 2097152 sample)
    (hv : polynomialValues false sample values) :
    |tape ⟨site, 0, 0, index, lweLaw⟩| ≤ 2097152 ∧
      values index = tape ⟨site, 0, 0, index, lweLaw⟩ % Q := by
  obtain ⟨_, _, hb, _, hout⟩ := hs
  simp only [polynomialValues, Bool.false_eq_true, if_false] at hv
  refine ⟨hb 0 0 index, ?_⟩
  rw [hv index, hout, tapeMatrix_coeff (by decide) (by decide), val_intCast_emod (by decide)]
  rfl

theorem split_twelve (c : Fin 12) :
    (∃ c' : Fin 6, c = Fin.castAdd 6 c') ∨ ∃ c' : Fin 6, c = Fin.natAdd 6 c' :=
  Fin.addCases (m := 6) (n := 6) (motive := fun c ↦
      (∃ c' : Fin 6, c = Fin.castAdd 6 c') ∨ ∃ c' : Fin 6, c = Fin.natAdd 6 c')
    (fun c' ↦ Or.inl ⟨c', rfl⟩) (fun c' ↦ Or.inr ⟨c', rfl⟩) c

/-- What key generation establishes, in terms of the sampling tape. -/
theorem keygen_spec {hashModel : HashModel} {tape : SampleTape} {key : ByteArray} {outputs}
    (h : Stage_keygen.generatedRoot FheBackend.backend hashModel tape [0] { unit := () } key
      outputs) :
    outputs.2.2.2.2.1 = lweSecret tape ∧
    (∀ j, lweSecret tape j = 0 ∨ lweSecret tape j = 1) ∧
    (∀ k : Fin N, tape (ringKey k) = 0 ∨ tape (ringKey k) = 1) ∧
    (∀ i, outputs.1 i = gswA tape i) ∧ (∀ i, outputs.2.1 i = gswB tape i) ∧
    (∀ (i : Fin lweN) (c : Fin 12) (k : Fin N), |tape (bskErrorKey i c k)| ≤ 2496) ∧
    (∀ i, 0 ≤ outputs.2.2.1 i ∧ outputs.2.2.1 i < q) ∧
    ∀ e : Fin 8192, |kskError tape e| ≤ 2097152 ∧
      outputs.2.2.2.1 e = ((∑ j : Fin lweN, outputs.2.2.1 ⟨e.val * 630 + j.val, by
        have := e.isLt; have := j.isLt; simp only [lweN] at *; omega⟩ * lweSecret tape j) +
        tape (ringKey (e.val / 8)) * (4 ^ (e.val % 8) * 65536) + kskError tape e) % q := by
  obtain ⟨witness, hS0, hS0v, _, hsecret, hZ, _, hgsw, hmask, hZv, hE0, hE0v, hE1, hE1v, hE2, hE2v,
    hE3, hE3v, hE4, hE4v, hE5, hE5v, hE6, hE6v, hE7, hE7v, _, hksk, hout⟩ := h
  subst hout
  -- The LWE secret.
  have hs (j : Fin lweN) : witness.w_2_0 j = lweSecret tape j := by
    obtain ⟨_, hj⟩ := hsecret j
    obtain ⟨hv, hb⟩ := bit_values hS0 hS0v ⟨j.val, by have := j.isLt; simp only [lweN, N] at *; omega⟩
    rw [hj]
    show witness.w_1_0 _ % 2 = _
    rw [hv]
    rcases hb with h0 | h1
    · rw [h0]; exact h0.symm
    · rw [h1]; exact h1.symm
  have hsfun : witness.w_2_0 = lweSecret tape := funext hs
  -- The ring secret.
  have hz : witness.w_3_0 0 0 = reducePoly Q N (ringSecret tape) := by
    obtain ⟨_, _, hout⟩ := hZ
    rw [hout]
    exact polynomialOfCoefficients_eq_reduce _
  refine ⟨hsfun, fun j ↦ ?_, fun k ↦ (bit_values hZ hZv k).2, fun i ↦ ?_, fun i ↦ ?_,
    fun i c k ↦ ?_, hashIntFamily_range hmask, fun e ↦ ?_⟩
  · obtain ⟨_, hb⟩ := bit_values hS0 hS0v ⟨j.val, by have := j.isLt; simp only [lweN, N] at *; omega⟩
    exact hb
  · obtain ⟨a, g, w6, E, w12, ha, hg, hcat1, _, _, hpair⟩ := hgsw i
    simp only [Prod.mk.injEq, and_true] at hpair
    obtain ⟨hA, _⟩ := hpair
    show witness.w_4_0 i = _
    rw [hA]
    funext r c
    have hr : r = 0 := Subsingleton.elim _ _
    subst hr
    obtain ⟨c, rfl⟩ | ⟨c, rfl⟩ := split_twelve c
    · simp only [matrixAdd, Matrix.add_apply, concatColumns_castAdd hcat1, matrixMulScalarLeft,
        liftInteger, gswA, Fin.val_castAdd, c.isLt, dif_pos, ha.2, gadgetMatrixRuns_eq hg, hsfun]
      rfl
    · simp only [matrixAdd, Matrix.add_apply, concatColumns_natAdd hcat1, Matrix.zero_apply,
        add_zero, gswA, Fin.val_natAdd, ha.2]
      rw [dif_neg (by omega)]
      simp only [add_zero]
      rfl
  · obtain ⟨a, g, w6, E, w12, ha, hg, _, hE, hcat2, hpair⟩ := hgsw i
    simp only [Prod.mk.injEq, and_true] at hpair
    obtain ⟨_, hB⟩ := hpair
    show witness.w_4_1 i = _
    rw [hB]
    funext r c
    have hr : r = 0 := Subsingleton.elim _ _
    subst hr
    have hEe : E 0 c = reducePoly Q N (bskError tape i 0 c) := by
      rw [hE.2.2.2.2]
      exact polynomialOfCoefficients_eq_reduce _
    obtain ⟨c, rfl⟩ | ⟨c, rfl⟩ := split_twelve c
    · simp only [matrixAdd, Matrix.add_apply, concatColumns_castAdd hcat2, matrixMulScalarLeft,
        Matrix.zero_apply, add_zero, gswB, Fin.val_castAdd, hz, ha.2]
      rw [dif_neg (by omega), add_zero, hEe]
      rfl
    · simp only [matrixAdd, Matrix.add_apply, concatColumns_natAdd hcat2, matrixMulScalarLeft,
        liftInteger, gswB, Fin.val_natAdd, hz, ha.2, gadgetMatrixRuns_eq hg, hsfun]
      rw [dif_pos (by omega), hEe]
      have hc : (⟨6 + c.val - 6, by omega⟩ : Fin 6) = c := Fin.ext (by simp)
      rw [hc]
      rfl
  · obtain ⟨a, g, w6, E, w12, ha, hg, hcat1, hE, hcat2, hpair⟩ := hgsw i
    exact hE.2.2.1 0 c k
  · obtain ⟨w5, w32, w40, w42, w44, w46, w48, w50, w52, w54, w55, w63, _, _, _, hzv, _, _, _, _,
      hγ, _, _, _, _, hf0, _, _, hf1, _, _, hf2, _, _, hf3, _, _, hf4, _, _, hf5, _, _, hf6, _, _,
      hf7, _, _, _, hres, _, _, _, hcen, _, hout⟩ := hksk e
    have he := e.isLt
    -- The error read from sample `e / 1024` at index `e % 1024`.
    have hfam : ∀ {site : List Nat} {sample : ExactMatrix Q N 1 1} {values : Fin N → Int} {w : Int},
        gaussianSampleAt tape site (131072 / 1) 2097152 sample →
        polynomialValues false sample values →
        familyGetDynamic values (Int.ofNat e.val % 1024) w → site = [0, 17 + 2 * (e.val / 1024)] →
        |kskError tape e| ≤ 2097152 ∧ w = kskError tape e % Q := by
      intro site sample values w hs hv hw hsite
      obtain ⟨c, hc, hwc⟩ := familyGetDynamic_val hw
      have hcv : c.val = e.val % 1024 := by
        simp only [Int.ofNat_eq_natCast] at hc; omega
      obtain ⟨hb, hval⟩ := ksk_error_value (index := c) hs hv
      subst hsite
      rw [hwc, hval]
      constructor
      · simpa only [kskError, kskErrorKey, ← hcv] using hb
      · simp only [kskError, kskErrorKey, ← hcv]
    have hw55 : |kskError tape e| ≤ 2097152 ∧ w55 = kskError tape e % Q := by
      obtain ⟨position, hposition, hpos⟩ := hres
      have hp : position.val = e.val / 1024 := by
        simp only [Int.ofNat_eq_natCast] at hposition; omega
      rw [hpos]
      fin_cases position <;> simp only [Fin.zero_eta] at hp ⊢
      · exact hfam hE0 hE0v hf0 (by simp only [← hp]; rfl)
      · exact hfam hE1 hE1v hf1 (by simp only [← hp]; rfl)
      · exact hfam hE2 hE2v hf2 (by simp only [← hp]; rfl)
      · exact hfam hE3 hE3v hf3 (by simp only [← hp]; rfl)
      · exact hfam hE4 hE4v hf4 (by simp only [← hp]; rfl)
      · exact hfam hE5 hE5v hf5 (by simp only [← hp]; rfl)
      · exact hfam hE6 hE6v hf6 (by simp only [← hp]; rfl)
      · exact hfam hE7 hE7v hf7 (by simp only [← hp]; rfl)
    obtain ⟨herror, h55⟩ := hw55
    refine ⟨herror, ?_⟩
    have h63 := select_two hcen
    have hc := centered_select_of_small (q := Q) (value := kskError tape e) (by decide)
      (by rw [Int.abs_eq_natAbs] at herror; simp only [Q]; omega)
    simp only [Q, Nat.cast_ofNat] at hc h55 h63
    rw [mul_comm, h55, hc] at h63
    have hgadget : w32 = 4 ^ (e.val % 8) * 65536 := by
      obtain ⟨position, hposition, hw32⟩ := hγ
      have hpe : e.val % 8 = position.val := by
        simp only [Int.ofNat_eq_natCast] at hposition; omega
      simp only at hw32
      rw [hw32, hpe]
      fin_cases position <;> norm_num
    have hz' : w5 = tape (ringKey (e.val / 8)) := by
      obtain ⟨position, hposition, hw5⟩ := familyGetDynamic_val hzv
      simp only at hw5
      have hpe : position.val = e.val / 8 := by
        simp only [Int.ofNat_eq_natCast] at hposition; omega
      rw [hw5, (bit_values hZ hZv position).1, hpe]
      rfl
    simp only at hout
    show witness.w_33_0 e = _
    rw [hout, intMatrixVectorProduct_rows (by rfl), hz', hgadget, h63, hsfun]
    rfl

end MxxFheTfhe
