import ExponentRuntime
import ExponentEncoding
import ExponentGadget
import ExponentRefresh
import GenLabel

/-!
# The encryption stage for any parameter set

The encrypt stage of a refresh instance computes, under the mask `(s, 1)` (`s` reduced or
re-encoded to the working modulus `q`), fresh encodings

  `c_j = s A_{0j} + A_{1j} - t_j + e_j`, `e_j` a Gaussian sample,

of four kinds, which differ only in the public key `A` and the target `t`:

* label entries (`encLabel_gen`, one member of the parallel over the `ν + 1` label entries of a
  PRF): `A` is an input, `t_j = (s, 1)_j μ G` for the payload `μ = c X^e`; with a `LabelEntry`
  table the payload is `X^(s a_i)` (`encLabel_table_gen`);
* LUT branch keys (`encBranchKey_gen`, one member of the parallel over the automorphisms of a
  LUT): `A` is the hash-sampled `hashPair` under `rns-bgg/lut-a` with the level and the
  automorphism index, `t_j = p σ((s, 1))_j G` for the automorphism `σ`, the dropped prime `p`,
  and the next level's gadget re-encoded at `q`; `encBranchKey_reduced_gen` first reduces the
  mask from a larger modulus;
* source encodings (`encSource_gen`): `A` is an input, `t_j = (r, 1)_j G · p μ` for the payload
  secret `r`;
* gadget encodings (`encGadget_gen`, e.g. the refresh's one encoding with payload `1`):
  `t_j = (s, 1)_j x G`.

The unmask hint `[s, 1] A + e` is `hint_gen`. `encrypt_core_gen` is the common step; nothing here
fixes the ring dimension, a modulus, a gadget width, the number of label entries or of LUT
branches, the Gaussian parameters or the hash tags. The hash-sampled keys are `hashPair`, which
the preprocessing samples under the same tags: `hashSample_unique` and `hashPair_of_samples`
link both stages.
-/

namespace RefreshTopProof.Gen

open Mxx.Primitives MxxRuntime MxxExponentLut Mxx.ExponentLut.Refresh

variable {n : Nat}

/-! ## Small runtime facts -/

/-- `component`, the tag component naming the block of a hash-sampled public matrix. -/
abbrev encComponentBytes : Blob := [99, 111, 109, 112, 111, 110, 101, 110, 116]

/-- `rns-bgg/lut-a`, the tag of the LUT branch keys. -/
abbrev lutAPrefixGen : Blob := [114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]

theorem one_matrix_eq {q : Nat} : (1 : ExactMatrix q n 1 1) = fun _ _ ↦ reducePoly q n 1 := by
  funext r c
  obtain rfl : r = 0 := Subsingleton.elim _ _
  obtain rfl : c = 0 := Subsingleton.elim _ _
  simp

theorem matrixPolynomial_single_gen {q : Nat} (k : Int) :
    (matrixPolynomial [k] : ExactMatrix q n 1 1) = fun _ _ ↦ reducePoly q n (k : ErrorPoly n) := by
  funext r c
  simp [matrixPolynomial]

/-- The centered rebase of a small integer polynomial is its reduction. -/
theorem rebase_small_gen (hn : 0 < n) {q p : Nat} (hq : 2 < q) {s : ErrorPoly n}
    (hs : polyNorm s ≤ 1) {out : ExactMatrix p n 1 1}
    (h : centeredRebaseRuns (fun _ _ ↦ reducePoly q n s : ExactMatrix q n 1 1) out) :
    out = fun _ _ ↦ reducePoly p n s := by
  rw [centeredRebaseRuns_reduce h]
  funext r c
  rw [centeredIntLift_reducePoly (by omega) hn _ fun i ↦ by
    have := coeff_natAbs_le_polyNorm s i
    omega]

/-- The centered rebase of one is one. -/
theorem rebase_one_gen (hn : 1 < n) {q p : Nat} (hq : 2 < q) {out : ExactMatrix p n 1 1}
    (h : centeredRebaseRuns (1 : ExactMatrix q n 1 1) out) : out = 1 := by
  rw [one_matrix_eq] at h
  rw [rebase_small_gen (by omega) hq (by simpa using polyNorm_Xe (n := n) hn 0) h, one_matrix_eq]

/-- An automorphism the runtime accepts has an odd index. -/
theorem ringAutomorphismRuns_odd {q rows columns : Nat} {index : Int}
    {input output : ExactMatrix q n rows columns} (h : ringAutomorphismRuns index input output) :
    Odd index.toNat := by
  obtain ⟨hpos, _, hparity, _⟩ := h
  exact Nat.odd_iff.mpr (by omega)

/-- An automorphism of a reduced polynomial is the reduced substitution. -/
theorem ringAutomorphism_poly {q : Nat} (hq : 1 < q) (hn : 0 < n) {index : Int} (hodd : Odd index.toNat)
    {s : ErrorPoly n} {out : ExactMatrix q n 1 1}
    (h : ringAutomorphismRuns index (fun _ _ ↦ reducePoly q n s : ExactMatrix q n 1 1) out) :
    out = fun _ _ ↦ reducePoly q n (substitution index.toNat hodd s) := by
  obtain ⟨_, hsub⟩ := ringAutomorphismRuns_eq hq hn h
  rw [hsub]
  funext r c
  exact (reducePoly_substitution _ _ _).symm

/-- An automorphism fixes one. -/
theorem ringAutomorphism_one {q : Nat} (hq : 1 < q) (hn : 0 < n) {index : Int}
    {out : ExactMatrix q n 1 1} (h : ringAutomorphismRuns index (1 : ExactMatrix q n 1 1) out) :
    out = 1 := by
  obtain ⟨_, hsub⟩ := ringAutomorphismRuns_eq hq hn h
  rw [hsub]
  funext r c
  obtain rfl : r = 0 := Subsingleton.elim _ _
  obtain rfl : c = 0 := Subsingleton.elim _ _
  simp

/-! ## Hash-sampled public matrices -/

/-- The `2 × 2` public matrix of `1 × m` blocks sampled under `tagPrefix` and the leading
components `pre`; block `(k, j)` has the trailing components `component, 2 k + j`. -/
noncomputable def hashPair (model : HashModel) (key : ByteArray) (q n m : Nat) (tagPrefix : Blob)
    (pre : List HashTagComponent) : Fin 2 → Fin 2 → ExactMatrix q n 1 m :=
  fun k j ↦ model.sample q n 1 m key (completeHashTag tagPrefix
    (pre ++ [.bytes encComponentBytes, .u64Le (2 * (k : Int) + (j : Int))]))

/-- Two samples under the same tag and key agree (the link between stages). -/
theorem hashSample_unique {q rows columns : Nat} {model : HashModel} {tagPrefix : Blob}
    {components : List HashTagComponent} {key : ByteArray} {x y : ExactMatrix q n rows columns}
    (hx : hashSample model tagPrefix components key x)
    (hy : hashSample model tagPrefix components key y) : x = y :=
  (hashSample_eq hx).trans (hashSample_eq hy).symm

/-- The four samples of a public matrix are its blocks. -/
theorem hashPair_of_samples {q m : Nat} {model : HashModel} {key : ByteArray} {tagPrefix : Blob}
    {pre : List HashTagComponent} {A00 A01 A10 A11 : ExactMatrix q n 1 m}
    (h00 : hashSample model tagPrefix (pre ++ [.bytes encComponentBytes, .u64Le 0]) key A00)
    (h01 : hashSample model tagPrefix (pre ++ [.bytes encComponentBytes, .u64Le 1]) key A01)
    (h10 : hashSample model tagPrefix (pre ++ [.bytes encComponentBytes, .u64Le 2]) key A10)
    (h11 : hashSample model tagPrefix (pre ++ [.bytes encComponentBytes, .u64Le 3]) key A11) :
    hashPair model key q n m tagPrefix pre = ![![A00, A01], ![A10, A11]] := by
  funext k j
  fin_cases k <;> fin_cases j
  · exact (hashSample_eq h00).symm
  · exact (hashSample_eq h01).symm
  · exact (hashSample_eq h10).symm
  · exact (hashSample_eq h11).symm

/-- Block `(k, j)` of a public matrix from its sample. -/
theorem hashPair_entry {q m : Nat} {model : HashModel} {key : ByteArray} {tagPrefix : Blob}
    {pre : List HashTagComponent} (k j : Fin 2) {A : ExactMatrix q n 1 m}
    (h : hashSample model tagPrefix
      (pre ++ [.bytes encComponentBytes, .u64Le (2 * (k : Int) + (j : Int))]) key A) :
    hashPair model key q n m tagPrefix pre k j = A :=
  (hashSample_eq h).symm

/-! ## The encryption step -/

/-- **A fresh encryption.** The runtime computes `c_j = (s A_{0j} + 1 A_{1j} - t_j) + e_j` with
Gaussian `e_j`: an encoding of the target `t` under the key `A` within the cutoff. -/
theorem encrypt_core_gen {q m : Nat} {sm : ErrorPoly n} {sM oneM : ExactMatrix q n 1 1}
    (hsm : sM = fun _ _ ↦ reducePoly q n sm) (hone : oneM = 1)
    (A : Fin 2 → Fin 2 → ExactMatrix q n 1 m) (t : Fin 2 → ExactMatrix q n 1 m)
    {sigma : Rat} {cutoff : Int} {e0 e1 : ExactMatrix q n 1 m}
    (he0 : gaussianSample sigma cutoff e0) (he1 : gaussianSample sigma cutoff e1)
    {c0 c1 : ExactMatrix q n 1 m}
    (hc0 : c0 = matrixAdd (matrixSub (matrixAdd (matrixMulScalarLeft sM (A 0 0))
      (matrixMulScalarLeft oneM (A 1 0))) (t 0)) e0)
    (hc1 : c1 = matrixAdd (matrixSub (matrixAdd (matrixMulScalarLeft sM (A 0 1))
      (matrixMulScalarLeft oneM (A 1 1))) (t 1)) e1) :
    Rel sm A t ![c0, c1] cutoff.toNat := by
  obtain ⟨E0, hE0, hb0⟩ := gaussian_bound he0
  obtain ⟨E1, hE1, hb1⟩ := gaussian_bound he1
  subst hsm hone hc0 hc1 hE0 hE1
  refine ⟨![E0, E1], fun j ↦ by fin_cases j <;> assumption, fun j ↦ ?_⟩
  fin_cases j <;>
    simp only [matrixAdd, matrixSub, matrixMulScalarLeft_eq] <;> simp

/-- A target `t_j = (x_j G)` whose scalar is reduced. -/
theorem smul_gadget_eq {q m : Nat} {xM : ExactMatrix q n 1 1} {x : ErrorPoly n}
    (hx : xM = fun _ _ ↦ reducePoly q n x) (G : ExactMatrix q n 1 m) :
    matrixMulScalarLeft xM G = reducePoly q n x • G := by
  rw [matrixMulScalarLeft_eq, hx]

/-! ## The roles -/

/-- **A gadget encoding** `t_j = (s, 1)_j x G` (the runtime forms `x G` once and scales it by the
mask components). The label entries (`x = c X^e`) and the refresh's one encoding (`x = 1`) are
gadget encodings. -/
theorem encGadget_gen {q m : Nat} {sm : ErrorPoly n} {sM oneM : ExactMatrix q n 1 1}
    (hsm : sM = fun _ _ ↦ reducePoly q n sm) (hone : oneM = 1)
    (A00 A01 A10 A11 G : ExactMatrix q n 1 m) {xM : ExactMatrix q n 1 1} {x : ErrorPoly n}
    (hx : xM = fun _ _ ↦ reducePoly q n x)
    {sigma : Rat} {cutoff : Int} {e0 e1 : ExactMatrix q n 1 m}
    (he0 : gaussianSample sigma cutoff e0) (he1 : gaussianSample sigma cutoff e1)
    {c0 c1 : ExactMatrix q n 1 m}
    (hc0 : c0 = matrixAdd (matrixSub (matrixAdd (matrixMulScalarLeft sM A00)
      (matrixMulScalarLeft oneM A10)) (matrixMulScalarLeft sM (matrixMulScalarLeft xM G))) e0)
    (hc1 : c1 = matrixAdd (matrixSub (matrixAdd (matrixMulScalarLeft sM A01)
      (matrixMulScalarLeft oneM A11)) (matrixMulScalarLeft oneM (matrixMulScalarLeft xM G))) e1) :
    Rel sm ![![A00, A01], ![A10, A11]] (fun j ↦ reducePoly q n (x * unitPair sm j) • G)
      ![c0, c1] cutoff.toNat := by
  have h := encrypt_core_gen hsm hone ![![A00, A01], ![A10, A11]]
    ![matrixMulScalarLeft sM (matrixMulScalarLeft xM G),
      matrixMulScalarLeft oneM (matrixMulScalarLeft xM G)] he0 he1 hc0 hc1
  refine h.target_eq ?_
  funext j
  subst hsm hone hx
  fin_cases j <;>
    simp only [matrixMulScalarLeft_eq] <;> simp [unitPair, smul_smul, mul_comm]

/-- **A label entry** (one member of the parallel over the `ν + 1` entries of a PRF's label
family): the payload `μ = c X^e` and the encoding of `(s, 1)_j μ G`. -/
theorem encLabel_gen {q m : Nat} {sm : ErrorPoly n} {sM oneM : ExactMatrix q n 1 1}
    (hsm : sM = fun _ _ ↦ reducePoly q n sm) (hone : oneM = 1)
    (A00 A01 A10 A11 G : ExactMatrix q n 1 m) (c e : Int) {μ : ExactMatrix q n 1 1}
    (hμ : μ = multiplyMonomial (liftInteger c : ExactMatrix q n 1 1) e)
    {sigma : Rat} {cutoff : Int} {e0 e1 : ExactMatrix q n 1 m}
    (he0 : gaussianSample sigma cutoff e0) (he1 : gaussianSample sigma cutoff e1)
    {c0 c1 : ExactMatrix q n 1 m}
    (hc0 : c0 = matrixAdd (matrixSub (matrixAdd (matrixMulScalarLeft sM A00)
      (matrixMulScalarLeft oneM A10)) (matrixMulScalarLeft sM (matrixMulScalarLeft μ G))) e0)
    (hc1 : c1 = matrixAdd (matrixSub (matrixAdd (matrixMulScalarLeft sM A01)
      (matrixMulScalarLeft oneM A11)) (matrixMulScalarLeft oneM (matrixMulScalarLeft μ G))) e1) :
    Rel sm ![![A00, A01], ![A10, A11]]
        (fun j ↦ reducePoly q n (message (n := n) c e * unitPair sm j) • G) ![c0, c1] cutoff.toNat ∧
      μ = fun _ _ ↦ reducePoly q n (message (n := n) c e) := by
  have hμ' : μ = fun _ _ ↦ reducePoly q n (message (n := n) c e) := by
    rw [hμ, multiplyMonomial_liftInteger]
  exact ⟨encGadget_gen hsm hone A00 A01 A10 A11 G hμ' he0 he1 hc0 hc1, hμ'⟩

/-- **The label table.** The parallel over the label entries: if every member is the encryption
of its payload and the table satisfies `LabelEntry` against the label exponents `a`, member `i`
encrypts `X^(s a_i)`. -/
theorem encLabel_table_gen {q m s ν B : Nat} {sm : ErrorPoly n} {G : ExactMatrix q n 1 m}
    (A : Fin (ν + 1) → Fin 2 → Fin 2 → ExactMatrix q n 1 m) (c e : Fin (ν + 1) → Int)
    (a : Fin (ν + 1) → Nat) (hlab : ∀ i, LabelEntry n s (c i) (e i) (a i))
    (out : Fin (ν + 1) → Fin 2 → ExactMatrix q n 1 m)
    (hstep : ∀ i, Rel sm (A i) (fun j ↦ reducePoly q n (message (n := n) (c i) (e i) *
      unitPair sm j) • G) (out i) B) (i : Fin (ν + 1)) :
    Rel sm (A i) (fun j ↦ reducePoly q n (Xe (n := n) (s * a i) * unitPair sm j) • G) (out i) B := by
  have h := hstep i
  rwa [label_message_gen (hlab i)] at h

/-- **A LUT branch key** (one member of the parallel over the automorphisms `σ` of a LUT at
`level`): the key is the hash-sampled `hashPair` under `rns-bgg/lut-a`, and the target is
`p σ((s, 1))_j G` for the scalar `p` and the gadget `G` (the next level's gadget re-encoded at `q`). -/
theorem encBranchKey_gen {q : Nat} (hq : 1 < q) (hn : 0 < n) {m : Nat} {sm : ErrorPoly n}
    {sM oneM : ExactMatrix q n 1 1} (hsm : sM = fun _ _ ↦ reducePoly q n sm) (hone : oneM = 1)
    {model : HashModel} {key : ByteArray} {level σ : Int} (hodd : Odd σ.toNat) {pc : Int}
    {A00 A01 A10 A11 : ExactMatrix q n 1 m}
    (h00 : hashSample model lutAPrefixGen
      [.u64Le level, .integer σ, .bytes encComponentBytes, .u64Le 0] key A00)
    (h01 : hashSample model lutAPrefixGen
      [.u64Le level, .integer σ, .bytes encComponentBytes, .u64Le 1] key A01)
    (h10 : hashSample model lutAPrefixGen
      [.u64Le level, .integer σ, .bytes encComponentBytes, .u64Le 2] key A10)
    (h11 : hashSample model lutAPrefixGen
      [.u64Le level, .integer σ, .bytes encComponentBytes, .u64Le 3] key A11)
    {sσ oneσ : ExactMatrix q n 1 1} (hsσ : ringAutomorphismRuns σ sM sσ)
    (honeσ : ringAutomorphismRuns σ oneM oneσ) (G : ExactMatrix q n 1 m)
    {sigma : Rat} {cutoff : Int} {e0 e1 : ExactMatrix q n 1 m}
    (he0 : gaussianSample sigma cutoff e0) (he1 : gaussianSample sigma cutoff e1)
    {c0 c1 : ExactMatrix q n 1 m}
    (hc0 : c0 = matrixAdd (matrixSub (matrixAdd (matrixMulScalarLeft sM A00)
      (matrixMulScalarLeft oneM A10))
      (matrixMulScalarLeft (matrixMulScalarLeft (matrixPolynomial [pc]) sσ) G)) e0)
    (hc1 : c1 = matrixAdd (matrixSub (matrixAdd (matrixMulScalarLeft sM A01)
      (matrixMulScalarLeft oneM A11))
      (matrixMulScalarLeft (matrixMulScalarLeft (matrixPolynomial [pc]) oneσ) G)) e1) :
    Rel sm (hashPair model key q n m lutAPrefixGen [.u64Le level, .integer σ])
      (fun j ↦ reducePoly q n ((pc : ErrorPoly n) *
        unitPair (substitution σ.toNat hodd sm) j) • G) ![c0, c1] cutoff.toNat := by
  have hA := hashPair_of_samples (pre := [.u64Le level, .integer σ]) h00 h01 h10 h11
  rw [hA]
  have hsσ' : sσ = fun _ _ ↦ reducePoly q n (substitution σ.toNat hodd sm) := by
    rw [hsm] at hsσ
    exact ringAutomorphism_poly hq hn hodd hsσ
  have honeσ' : oneσ = 1 := by
    rw [hone] at honeσ
    exact ringAutomorphism_one hq hn honeσ
  have h := encrypt_core_gen hsm hone ![![A00, A01], ![A10, A11]]
    ![matrixMulScalarLeft (matrixMulScalarLeft (matrixPolynomial [pc]) sσ) G,
      matrixMulScalarLeft (matrixMulScalarLeft (matrixPolynomial [pc]) oneσ) G] he0 he1 hc0 hc1
  refine h.target_eq ?_
  funext j
  subst hsσ' honeσ'
  fin_cases j <;>
    simp only [matrixMulScalarLeft_eq, matrixPolynomial_single_gen] <;> simp [unitPair]

/-- A LUT branch key whose mask is first reduced from the larger modulus `Q` (`q ∣ Q`). -/
theorem encBranchKey_reduced_gen {Q q : Nat} (hQ : 1 < Q) (hq : 1 < q) (hn : 0 < n) {m : Nat}
    {sm : ErrorPoly n} {sMQ oneMQ : ExactMatrix Q n 1 1}
    (hsm : sMQ = fun _ _ ↦ reducePoly Q n sm) (hone : oneMQ = 1)
    {sM oneM : ExactMatrix q n 1 1} (hsr : modulusReduceRuns sMQ sM)
    (honer : modulusReduceRuns oneMQ oneM)
    {model : HashModel} {key : ByteArray} {level σ : Int} (hodd : Odd σ.toNat) {pc : Int}
    {A00 A01 A10 A11 : ExactMatrix q n 1 m}
    (h00 : hashSample model lutAPrefixGen
      [.u64Le level, .integer σ, .bytes encComponentBytes, .u64Le 0] key A00)
    (h01 : hashSample model lutAPrefixGen
      [.u64Le level, .integer σ, .bytes encComponentBytes, .u64Le 1] key A01)
    (h10 : hashSample model lutAPrefixGen
      [.u64Le level, .integer σ, .bytes encComponentBytes, .u64Le 2] key A10)
    (h11 : hashSample model lutAPrefixGen
      [.u64Le level, .integer σ, .bytes encComponentBytes, .u64Le 3] key A11)
    {sσ oneσ : ExactMatrix q n 1 1} (hsσ : ringAutomorphismRuns σ sM sσ)
    (honeσ : ringAutomorphismRuns σ oneM oneσ) (G : ExactMatrix q n 1 m)
    {sigma : Rat} {cutoff : Int} {e0 e1 : ExactMatrix q n 1 m}
    (he0 : gaussianSample sigma cutoff e0) (he1 : gaussianSample sigma cutoff e1)
    {c0 c1 : ExactMatrix q n 1 m}
    (hc0 : c0 = matrixAdd (matrixSub (matrixAdd (matrixMulScalarLeft sM A00)
      (matrixMulScalarLeft oneM A10))
      (matrixMulScalarLeft (matrixMulScalarLeft (matrixPolynomial [pc]) sσ) G)) e0)
    (hc1 : c1 = matrixAdd (matrixSub (matrixAdd (matrixMulScalarLeft sM A01)
      (matrixMulScalarLeft oneM A11))
      (matrixMulScalarLeft (matrixMulScalarLeft (matrixPolynomial [pc]) oneσ) G)) e1) :
    Rel sm (hashPair model key q n m lutAPrefixGen [.u64Le level, .integer σ])
      (fun j ↦ reducePoly q n ((pc : ErrorPoly n) *
        unitPair (substitution σ.toNat hodd sm) j) • G) ![c0, c1] cutoff.toNat := by
  have hsm' : sM = fun _ _ ↦ reducePoly q n sm := by
    rw [hsm] at hsr
    exact modulusReduce_poly hQ hq hn hsr
  have hone' : oneM = 1 := by
    rw [hone] at honer
    exact modulusReduce_one hQ hq hn honer
  exact encBranchKey_gen hq hn hsm' hone' hodd h00 h01 h10 h11 hsσ honeσ G he0 he1 hc0 hc1

/-- **A source encoding** `t_j = (r, 1)_j G · p μ` under the input key `A`, for the payload
secret `r` (`rM` reduced at `q`), the scalar `p` and the payload `μ`. -/
theorem encSource_gen {q m : Nat} {sm sp : ErrorPoly n} {sM oneM rM oneR : ExactMatrix q n 1 1}
    (hsm : sM = fun _ _ ↦ reducePoly q n sm) (hone : oneM = 1)
    (hsp : rM = fun _ _ ↦ reducePoly q n sp) (honeR : oneR = 1)
    (A00 A01 A10 A11 G : ExactMatrix q n 1 m) {pc : Int} {μ : ExactMatrix q n 1 1} {x : ErrorPoly n}
    (hμ : μ = fun _ _ ↦ reducePoly q n x)
    {sigma : Rat} {cutoff : Int} {e0 e1 : ExactMatrix q n 1 m}
    (he0 : gaussianSample sigma cutoff e0) (he1 : gaussianSample sigma cutoff e1)
    {c0 c1 : ExactMatrix q n 1 m}
    (hc0 : c0 = matrixAdd (matrixSub (matrixAdd (matrixMulScalarLeft sM A00)
      (matrixMulScalarLeft oneM A10)) (matrixMulScalarRight (matrixMulScalarLeft rM G)
        (matrixMulScalarLeft (matrixPolynomial [pc]) μ))) e0)
    (hc1 : c1 = matrixAdd (matrixSub (matrixAdd (matrixMulScalarLeft sM A01)
      (matrixMulScalarLeft oneM A11)) (matrixMulScalarRight (matrixMulScalarLeft oneR G)
        (matrixMulScalarLeft (matrixPolynomial [pc]) μ))) e1) :
    Rel sm ![![A00, A01], ![A10, A11]]
      (fun j ↦ reducePoly q n ((pc : ErrorPoly n) * x * unitPair sp j) • G) ![c0, c1]
      cutoff.toNat := by
  have h := encrypt_core_gen hsm hone ![![A00, A01], ![A10, A11]]
    ![matrixMulScalarRight (matrixMulScalarLeft rM G) (matrixMulScalarLeft (matrixPolynomial [pc]) μ),
      matrixMulScalarRight (matrixMulScalarLeft oneR G)
        (matrixMulScalarLeft (matrixPolynomial [pc]) μ)] he0 he1 hc0 hc1
  refine h.target_eq ?_
  funext j
  subst hsp honeR hμ
  fin_cases j <;>
    simp only [matrixMulScalarLeft_eq, matrixMulScalarRight_eq, matrixPolynomial_single_gen] <;>
    simp [unitPair, smul_smul, mul_comm, mul_left_comm]

/-- **The unmask hint** `[s, 1] A + e` at `q`, from the mask at `Q` (`q ∣ Q`). -/
theorem hint_gen {Q q m : Nat} (hQ : 1 < Q) (hq : 1 < q) (hn : 0 < n) {sm : ErrorPoly n}
    {sMQ oneMQ : ExactMatrix Q n 1 1} (hsm : sMQ = fun _ _ ↦ reducePoly Q n sm) (hone : oneMQ = 1)
    {a b : ExactMatrix q n 1 1} (ha : modulusReduceRuns sMQ a) (hb : modulusReduceRuns oneMQ b)
    {r : ExactMatrix q n 1 2} (hr : concatColumns a b r) {A : ExactMatrix q n 2 m}
    {sigma : Rat} {cutoff : Int} {e : ExactMatrix q n 1 m} (he : gaussianSample sigma cutoff e) :
    ∃ eI : ErrorMatrix n 1 m, CoeffBound eI cutoff.toNat ∧
      matrixAdd (matrixMul r A) e = maskPart (reducePoly q n sm) (fun i ↦ rowOf A i) +
        reduceMatrix q n 1 m eI := by
  obtain ⟨eI, heI, hb'⟩ := gaussian_bound he
  refine ⟨eI, hb', ?_⟩
  rw [hsm] at ha
  rw [hone] at hb
  rw [modulusReduce_poly hQ hq hn ha] at hr
  rw [modulusReduce_one hQ hq hn hb] at hr
  rw [heI]
  congr 1
  funext i c
  obtain rfl : i = 0 := Subsingleton.elim _ _
  simp only [matrixMul, Matrix.mul_apply, hr 0, maskPart, rowOf,
    Matrix.add_apply, Matrix.smul_apply, smul_eq_mul]
  simp [Fin.sum_univ_two, Matrix.one_apply]

end RefreshTopProof.Gen
