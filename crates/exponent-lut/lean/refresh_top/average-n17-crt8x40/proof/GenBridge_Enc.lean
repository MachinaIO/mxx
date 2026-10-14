import GenIface
import GenEnc

/-!
# Bundle bridges of the encryption stage

`GenIface` states every encoding-level scope of the encrypt stage as a computation bundle
`X.Comp` (the scope relation with dimensions, moduli and constants as parameters). This file
derives the `Rel`-level conclusions of `GenEnc` from those bundles:

* `EncLabelEntry.rel`: a label-entry encryption (one member of the parallel over the label
  entries) encrypts `μ = c X^e` times the gadget, passes the public key through, and outputs `μ`
  as a polynomial (`encLabel_gen`); `EncLabelEntry.rel_1` the same under the member's
  hash-sampled public key (`labelKeyG`, a Program in its own stages);
* `EncLutBranchKey.rel`: a LUT branch key under the
  mask `(s, 1)` at the working modulus (`encBranchKey_gen`);
* `EncLutBranchKey.rel_1`: the same after reducing the mask from larger moduli
  (as `encBranchKey_reduced_gen`, with independent moduli for `s` and `1`).

The branch-key conclusions name the gadget as the centered rebase of a `gadgetMatrixRuns` output
at the bundle's gadget modulus; the automorphism oddness is an argument (any proof; `X.odd`
derives one from the bundle). Constants that the bundle keeps as separate parameters but the
lemma needs to agree (the hash tags, levels, automorphism tables, Gaussian parameters, the dropped
prime `[pc]`) are equations, discharged by `rfl` at an instance.

The encrypt root (source, input, one encodings, unmask hint) is not a bundle; the lemmas of the
last section take the root's atoms in the form of the generated `Facts` fields.

Nothing is unrolled: the parallels over label entries and branches apply these lemmas per member.
-/

namespace RefreshTopProof.GenIface

open Mxx.Primitives MxxRuntime MxxExponentLut Mxx.ExponentLut.Refresh RefreshTopProof.Gen

/-! ## Label entries -/

/-- **Label entry** (`enc.label-entry`): outputs `(A, c, μ)` with `c` encrypting
`μ · G = c X^e · G` under `A` and the mask `(s, 1)`. -/
theorem EncLabelEntry.rel {q0 n d0 : Nat} {p0 p1 : Int} {p2 : Rat} {p3 : Int} {p4 : Rat}
    {p5 : Int} {backend : BackendContext} {hashModel : HashModel}
    {inputs : ExactMatrix q0 n 1 d0 × ExactMatrix q0 n 1 d0 × ExactMatrix q0 n 1 d0 ×
      ExactMatrix q0 n 1 d0 × ExactMatrix q0 n 1 1 × ExactMatrix q0 n 1 1 × Int × Int × Unit}
    {outputs : ExactMatrix q0 n 1 d0 × ExactMatrix q0 n 1 d0 × ExactMatrix q0 n 1 d0 ×
      ExactMatrix q0 n 1 d0 × ExactMatrix q0 n 1 d0 × ExactMatrix q0 n 1 d0 ×
      ExactMatrix q0 n 1 1 × Unit}
    (h : EncLabelEntry.Comp q0 n d0 p0 p1 p2 p3 p4 p5 backend hashModel inputs outputs)
    {sm : ErrorPoly n} (hsm : inputs.2.2.2.2.1 = fun _ _ ↦ reducePoly q0 n sm)
    (hone : inputs.2.2.2.2.2.1 = 1) (hp4 : p4 = p2) (hp5 : p5 = p3) :
    ∃ G : ExactMatrix q0 n 1 d0, gadgetMatrixRuns backend p0 p1 G ∧
      outputs.1 = inputs.1 ∧ outputs.2.1 = inputs.2.1 ∧ outputs.2.2.1 = inputs.2.2.1 ∧
      outputs.2.2.2.1 = inputs.2.2.2.1 ∧
      Rel sm ![![outputs.1, outputs.2.1], ![outputs.2.2.1, outputs.2.2.2.1]]
        (fun j ↦ reducePoly q0 n
          (message (n := n) inputs.2.2.2.2.2.2.1 inputs.2.2.2.2.2.2.2.1 * unitPair sm j) • G)
        ![outputs.2.2.2.2.1, outputs.2.2.2.2.2.1] p3.toNat ∧
      outputs.2.2.2.2.2.2.1 =
        fun _ _ ↦ reducePoly q0 n (message (n := n) inputs.2.2.2.2.2.2.1 inputs.2.2.2.2.2.2.2.1) := by
  subst hp4 hp5
  obtain ⟨v0, v1, v2, a0, a1, a2, rfl⟩ := h
  obtain ⟨hr, hμ⟩ := encLabel_gen hsm hone _ _ _ _ v0 _ _ rfl a1 a2 rfl rfl
  exact ⟨v0, a0, rfl, rfl, rfl, rfl, hr, hμ⟩

/-- **Label entry of a Program in its own stages** (`enc.label-entry`, variant 1): the public key
of member `i` is hash-sampled from the key (`labelKeyG`, components `0, 2` the first row and `1, 3`
the second), and the outputs `(c, μ)` encrypt `μ · G = c X^e · G` under it and `(s, 1)`. -/
theorem EncLabelEntry.rel_1 {q0 n d0 : Nat} {p0 p1 : Blob} {p2 p3 : Int} {p4 : Rat} {p5 : Int}
    {p6 p7 : Blob} {p8 : Rat} {p9 : Int} {backend : BackendContext} {hashModel : HashModel} {i : Nat}
    {inputs : ExactMatrix q0 n 1 1 × ByteArray × ExactMatrix q0 n 1 1 × Int × Int × Unit}
    {outputs : ExactMatrix q0 n 1 d0 × ExactMatrix q0 n 1 d0 × ExactMatrix q0 n 1 1 × Unit}
    (h : EncLabelEntry.Comp_1 q0 n d0 p0 p1 p2 p3 p4 p5 p6 p7 p8 p9 backend hashModel i inputs outputs)
    {sm : ErrorPoly n} (hsm : inputs.1 = fun _ _ ↦ reducePoly q0 n sm)
    (hone : inputs.2.2.1 = 1) (hp8 : p8 = p4) (hp9 : p9 = p5) :
    ∃ G : ExactMatrix q0 n 1 d0, gadgetMatrixRuns backend p2 p3 G ∧
      Rel sm ![![labelKeyG hashModel q0 n d0 inputs.2.1 p0 (Int.ofNat i) 0,
          labelKeyG hashModel q0 n d0 inputs.2.1 p6 (Int.ofNat i) 1],
        ![labelKeyG hashModel q0 n d0 inputs.2.1 p1 (Int.ofNat i) 2,
          labelKeyG hashModel q0 n d0 inputs.2.1 p7 (Int.ofNat i) 3]]
        (fun j ↦ reducePoly q0 n
          (message (n := n) inputs.2.2.2.1 inputs.2.2.2.2.1 * unitPair sm j) • G)
        ![outputs.1, outputs.2.1] p5.toNat ∧
      outputs.2.2.1 = fun _ _ ↦ reducePoly q0 n (message (n := n) inputs.2.2.2.1 inputs.2.2.2.2.1) := by
  subst hp8 hp9
  obtain ⟨v0, v1, v2, v3, v4, v5, v6, a0, a1, a2, a3, a4, a5, a6, rfl⟩ := h
  rw [← labelKeyG_of a0, ← labelKeyG_of a1, ← labelKeyG_of a4, ← labelKeyG_of a5]
  obtain ⟨hr, hμ⟩ := encLabel_gen hsm hone _ _ _ _ v2 _ _ rfl a3 a6 rfl rfl
  exact ⟨v2, a2, hr, hμ⟩

/-! ## LUT branch keys -/

section BranchKey

variable {n d0 : Nat} {p0 : Int} {p1 : Blob} {p2 : List Int} {p3 : Int} {p4 : Blob}
  {p5 p6 : List Int} {p7 p8 : Int} {p9 : Rat} {p10 p11 : Int} {p12 : Blob} {p13 : List Int}
  {p14 : Int} {p15 : Blob} {p16 p17 : List Int} {p18 : Rat} {p19 : Int} {p20 p21 : List Int}
  {backend : BackendContext} {hashModel : HashModel} {i : Nat}

/-- **LUT branch key** (`enc.lut-branch-key`, variant 0: the mask at the working modulus `q0`):
the outputs encrypt `pc · σ((s, 1)) · G` under the hash-sampled `hashPair` of the tag
`rns-bgg/lut-a`, the level `p0` and the automorphism `σ = p2[i]`, where `G` is the centered
rebase of the gadget matrix at `q1`. -/
theorem EncLutBranchKey.rel {q0 q1 : Nat}
    {inputs : ExactMatrix q0 n 1 1 × ByteArray × ExactMatrix q0 n 1 1 × Unit}
    {outputs : ExactMatrix q0 n 1 d0 × ExactMatrix q0 n 1 d0 × Unit}
    (h : EncLutBranchKey.Comp q0 n d0 q1 p0 p1 p2 p3 p4 p5 p6 p7 p8 p9 p10 p11 p12 p13 p14 p15
      p16 p17 p18 p19 p20 p21 backend hashModel i inputs outputs)
    (hq : 1 < q0) (hn : 0 < n) {sm : ErrorPoly n}
    (hsm : inputs.1 = fun _ _ ↦ reducePoly q0 n sm) (hone : inputs.2.2.1 = 1)
    (hp1 : p1 = lutAPrefixGen) (hp3 : p3 = p0) (hp4 : p4 = lutAPrefixGen) (hp5 : p5 = p2)
    (hp6 : p6 = p2) (hp11 : p11 = p0) (hp12 : p12 = lutAPrefixGen) (hp13 : p13 = p2)
    (hp14 : p14 = p0) (hp15 : p15 = lutAPrefixGen) (hp16 : p16 = p2) (hp17 : p17 = p2)
    (hp18 : p18 = p9) (hp19 : p19 = p10) {pc : Int} (hp20 : p20 = [pc]) (hp21 : p21 = [pc])
    (hodd : Odd (MxxIR.selectInt (Int.ofNat i) p2).toNat) :
    ∃ (Gq : ExactMatrix q1 n 1 d0) (G : ExactMatrix q0 n 1 d0),
      gadgetMatrixRuns backend p7 p8 Gq ∧ centeredRebaseRuns Gq G ∧
      Rel sm (hashPair hashModel inputs.2.1 q0 n d0 lutAPrefixGen
          [.u64Le p0, .integer (MxxIR.selectInt (Int.ofNat i) p2)])
        (fun j ↦ reducePoly q0 n ((pc : ErrorPoly n) * unitPair (substitution _ hodd sm) j) • G)
        ![outputs.1, outputs.2.1] p10.toNat := by
  subst hp1 hp3 hp4 hp5 hp6 hp11 hp12 hp13 hp14 hp15 hp16 hp17 hp18 hp19 hp20 hp21
  obtain ⟨v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, rfl⟩ := h
  exact ⟨v3, v4, a3, a4, encBranchKey_gen hq hn hsm hone hodd a0 a6 a1 a7 a2 a8 v4 a5 a9 rfl rfl⟩

/-- The automorphism index of a variant-0 branch key is odd. -/
theorem EncLutBranchKey.odd {q0 q1 : Nat}
    {inputs : ExactMatrix q0 n 1 1 × ByteArray × ExactMatrix q0 n 1 1 × Unit}
    {outputs : ExactMatrix q0 n 1 d0 × ExactMatrix q0 n 1 d0 × Unit}
    (h : EncLutBranchKey.Comp q0 n d0 q1 p0 p1 p2 p3 p4 p5 p6 p7 p8 p9 p10 p11 p12 p13 p14 p15
      p16 p17 p18 p19 p20 p21 backend hashModel i inputs outputs)
    (hp6 : p6 = p2) : Odd (MxxIR.selectInt (Int.ofNat i) p2).toNat := by
  subst hp6
  obtain ⟨_, _, _, _, _, _, _, _, _, _, _, _, a2, -⟩ := h
  exact ringAutomorphismRuns_odd a2

/-- **LUT branch key**, variant 1: the mask `s` (at `q1`) and `1` (at `q2`) are first reduced
to the working modulus `q0`; otherwise as variant 0 (gadget at `q3`). -/
theorem EncLutBranchKey.rel_1 {q0 q1 q2 q3 : Nat}
    {inputs : ExactMatrix q1 n 1 1 × ByteArray × ExactMatrix q2 n 1 1 × Unit}
    {outputs : ExactMatrix q0 n 1 d0 × ExactMatrix q0 n 1 d0 × Unit}
    (h : EncLutBranchKey.Comp_1 q0 n d0 q1 q2 q3 p0 p1 p2 p3 p4 p5 p6 p7 p8 p9 p10 p11 p12 p13
      p14 p15 p16 p17 p18 p19 p20 p21 backend hashModel i inputs outputs)
    (hq : 1 < q0) (hq1 : 1 < q1) (hq2 : 1 < q2) (hn : 0 < n) {sm : ErrorPoly n}
    (hsm : inputs.1 = fun _ _ ↦ reducePoly q1 n sm) (hone : inputs.2.2.1 = 1)
    (hp1 : p1 = lutAPrefixGen) (hp3 : p3 = p0) (hp4 : p4 = lutAPrefixGen) (hp5 : p5 = p2)
    (hp6 : p6 = p2) (hp11 : p11 = p0) (hp12 : p12 = lutAPrefixGen) (hp13 : p13 = p2)
    (hp14 : p14 = p0) (hp15 : p15 = lutAPrefixGen) (hp16 : p16 = p2) (hp17 : p17 = p2)
    (hp18 : p18 = p9) (hp19 : p19 = p10) {pc : Int} (hp20 : p20 = [pc]) (hp21 : p21 = [pc])
    (hodd : Odd (MxxIR.selectInt (Int.ofNat i) p2).toNat) :
    ∃ (Gq : ExactMatrix q3 n 1 d0) (G : ExactMatrix q0 n 1 d0),
      gadgetMatrixRuns backend p7 p8 Gq ∧ centeredRebaseRuns Gq G ∧
      Rel sm (hashPair hashModel inputs.2.1 q0 n d0 lutAPrefixGen
          [.u64Le p0, .integer (MxxIR.selectInt (Int.ofNat i) p2)])
        (fun j ↦ reducePoly q0 n ((pc : ErrorPoly n) * unitPair (substitution _ hodd sm) j) • G)
        ![outputs.1, outputs.2.1] p10.toNat := by
  subst hp1 hp3 hp4 hp5 hp6 hp11 hp12 hp13 hp14 hp15 hp16 hp17 hp18 hp19 hp20 hp21
  obtain ⟨v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11,
    a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, rfl⟩ := h
  have hs : v0 = fun _ _ ↦ reducePoly q0 n sm := by
    rw [hsm] at a0
    exact modulusReduce_poly hq1 hq hn a0
  have h1 : v2 = 1 := by
    rw [hone] at a2
    exact modulusReduce_one hq2 hq hn a2
  exact ⟨v5, v6, a5, a6, encBranchKey_gen hq hn hs h1 hodd a1 a8 a3 a9 a4 a10 v6 a7 a11 rfl rfl⟩

/-- The automorphism index of a variant-1 branch key is odd. -/
theorem EncLutBranchKey.odd_1 {q0 q1 q2 q3 : Nat}
    {inputs : ExactMatrix q1 n 1 1 × ByteArray × ExactMatrix q2 n 1 1 × Unit}
    {outputs : ExactMatrix q0 n 1 d0 × ExactMatrix q0 n 1 d0 × Unit}
    (h : EncLutBranchKey.Comp_1 q0 n d0 q1 q2 q3 p0 p1 p2 p3 p4 p5 p6 p7 p8 p9 p10 p11 p12 p13
      p14 p15 p16 p17 p18 p19 p20 p21 backend hashModel i inputs outputs)
    (hp6 : p6 = p2) : Odd (MxxIR.selectInt (Int.ofNat i) p2).toNat := by
  subst hp6
  obtain ⟨_, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, a4, -⟩ := h
  exact ringAutomorphismRuns_odd a4

end BranchKey

/-! ## The encrypt root (not a bundle)

The instance root of the encrypt stage carries, besides the parallels above, the source
encodings of the payload (one per level; the masks re-encoded by `centeredRebaseRuns` from the
input modulus), the input encoding under the hash-sampled key `rns-bgg/input/…` (masks at the
working modulus), the one encoding under `refresh/v1/one` (masks reduced by `modulusReduceRuns`)
and the unmask hint (`RefreshTopProof.Gen.hint_gen`, whose hypotheses are already the root's
`modulusReduceRuns`/`concatColumns`/`gaussianSample` facts with `hone := rfl`). The lemmas take the
root's atoms exactly as the generated `Facts` fields state them; let-bound wires are closed by
`rfl` (`hc0`, `hc1`, the literal `1` inputs). -/

/-- **Source encoding with re-encoded masks**: `c` encrypts `(r, 1)_j G · pc · c X^e` under the
input key `A`, where the masks `s`, `1`, `r`, `1` at `q` are centered rebases of small
polynomials at `Q`. -/
theorem EncRoot.source_rebase {n Q q m : Nat} (hn : 1 < n) (hQ : 2 < Q) {sm sp : ErrorPoly n}
    (hsmall : polyNorm sm ≤ 1) (hspsmall : polyNorm sp ≤ 1)
    {sQ rQ : ExactMatrix Q n 1 1} (hsQ : sQ = fun _ _ ↦ reducePoly Q n sm)
    (hrQ : rQ = fun _ _ ↦ reducePoly Q n sp)
    {sM oneM rM oneR : ExactMatrix q n 1 1}
    (hs : centeredRebaseRuns sQ sM) (hone : centeredRebaseRuns (1 : ExactMatrix Q n 1 1) oneM)
    (hr : centeredRebaseRuns rQ rM) (honeR : centeredRebaseRuns (1 : ExactMatrix Q n 1 1) oneR)
    (A00 A01 A10 A11 G : ExactMatrix q n 1 m) {pc c e : Int}
    {sigma : Rat} {cutoff : Int} {e0 e1 : ExactMatrix q n 1 m}
    (he0 : gaussianSample sigma cutoff e0) (he1 : gaussianSample sigma cutoff e1)
    {c0 c1 : ExactMatrix q n 1 m}
    (hc0 : c0 = matrixAdd (matrixSub (matrixAdd (matrixMulScalarLeft sM A00)
      (matrixMulScalarLeft oneM A10)) (matrixMulScalarRight (matrixMulScalarLeft rM G)
        (matrixMulScalarLeft (matrixPolynomial [pc])
          (multiplyMonomial (liftInteger c : ExactMatrix q n 1 1) e)))) e0)
    (hc1 : c1 = matrixAdd (matrixSub (matrixAdd (matrixMulScalarLeft sM A01)
      (matrixMulScalarLeft oneM A11)) (matrixMulScalarRight (matrixMulScalarLeft oneR G)
        (matrixMulScalarLeft (matrixPolynomial [pc])
          (multiplyMonomial (liftInteger c : ExactMatrix q n 1 1) e)))) e1) :
    Rel sm ![![A00, A01], ![A10, A11]]
      (fun j ↦ reducePoly q n ((pc : ErrorPoly n) * message (n := n) c e * unitPair sp j) • G)
      ![c0, c1] cutoff.toNat := by
  subst hsQ hrQ
  exact encSource_gen (rebase_small_gen (by omega) hQ hsmall hs) (rebase_one_gen hn hQ hone)
    (rebase_small_gen (by omega) hQ hspsmall hr) (rebase_one_gen hn hQ honeR) A00 A01 A10 A11 G
    (multiplyMonomial_liftInteger c e) he0 he1 hc0 hc1

/-- **Input encoding**: as a source encoding, with the masks at the working modulus and the key
the hash-sampled `hashPair` of the tag prefix `P` (no extra tag components). -/
theorem EncRoot.input {n q m : Nat} {model : HashModel} {key : ByteArray} {P : Blob}
    {sm sp : ErrorPoly n} {sM oneM rM oneR : ExactMatrix q n 1 1}
    (hsm : sM = fun _ _ ↦ reducePoly q n sm) (hone : oneM = 1)
    (hsp : rM = fun _ _ ↦ reducePoly q n sp) (honeR : oneR = 1)
    {A00 A01 A10 A11 : ExactMatrix q n 1 m}
    (h00 : hashSample model P [.bytes encComponentBytes, .u64Le 0] key A00)
    (h10 : hashSample model P [.bytes encComponentBytes, .u64Le 2] key A10)
    (h01 : hashSample model P [.bytes encComponentBytes, .u64Le 1] key A01)
    (h11 : hashSample model P [.bytes encComponentBytes, .u64Le 3] key A11)
    (G : ExactMatrix q n 1 m) {pc c e : Int}
    {sigma : Rat} {cutoff : Int} {e0 e1 : ExactMatrix q n 1 m}
    (he0 : gaussianSample sigma cutoff e0) (he1 : gaussianSample sigma cutoff e1)
    {c0 c1 : ExactMatrix q n 1 m}
    (hc0 : c0 = matrixAdd (matrixSub (matrixAdd (matrixMulScalarLeft sM A00)
      (matrixMulScalarLeft oneM A10)) (matrixMulScalarRight (matrixMulScalarLeft rM G)
        (matrixMulScalarLeft (matrixPolynomial [pc])
          (multiplyMonomial (liftInteger c : ExactMatrix q n 1 1) e)))) e0)
    (hc1 : c1 = matrixAdd (matrixSub (matrixAdd (matrixMulScalarLeft sM A01)
      (matrixMulScalarLeft oneM A11)) (matrixMulScalarRight (matrixMulScalarLeft oneR G)
        (matrixMulScalarLeft (matrixPolynomial [pc])
          (multiplyMonomial (liftInteger c : ExactMatrix q n 1 1) e)))) e1) :
    Rel sm (hashPair model key q n m P [])
      (fun j ↦ reducePoly q n ((pc : ErrorPoly n) * message (n := n) c e * unitPair sp j) • G)
      ![c0, c1] cutoff.toNat := by
  rw [hashPair_of_samples (pre := []) h00 h01 h10 h11]
  exact encSource_gen hsm hone hsp honeR A00 A01 A10 A11 G (multiplyMonomial_liftInteger c e)
    he0 he1 hc0 hc1

/-- **Gadget encoding with reduced masks** (the refresh's one encoding, `x = 1`): `c` encrypts
`(s, 1)_j x G` under the hash-sampled `hashPair` of `P` and the tag components `pre`, the masks
reduced from `Q` to `q`. -/
theorem EncRoot.gadget_reduced {n Q q m : Nat} (hQ : 1 < Q) (hq : 1 < q) (hn : 0 < n)
    {sm : ErrorPoly n} {sQ : ExactMatrix Q n 1 1} (hsQ : sQ = fun _ _ ↦ reducePoly Q n sm)
    {sM oneM : ExactMatrix q n 1 1} (hs : modulusReduceRuns sQ sM)
    (hone : modulusReduceRuns (1 : ExactMatrix Q n 1 1) oneM)
    {model : HashModel} {key : ByteArray} {P : Blob} (pre : List HashTagComponent)
    {A00 A01 A10 A11 : ExactMatrix q n 1 m}
    (h00 : hashSample model P (pre ++ [.bytes encComponentBytes, .u64Le 0]) key A00)
    (h10 : hashSample model P (pre ++ [.bytes encComponentBytes, .u64Le 2]) key A10)
    (h01 : hashSample model P (pre ++ [.bytes encComponentBytes, .u64Le 1]) key A01)
    (h11 : hashSample model P (pre ++ [.bytes encComponentBytes, .u64Le 3]) key A11)
    (G : ExactMatrix q n 1 m) {x : Int}
    {sigma : Rat} {cutoff : Int} {e0 e1 : ExactMatrix q n 1 m}
    (he0 : gaussianSample sigma cutoff e0) (he1 : gaussianSample sigma cutoff e1)
    {c0 c1 : ExactMatrix q n 1 m}
    (hc0 : c0 = matrixAdd (matrixSub (matrixAdd (matrixMulScalarLeft sM A00)
      (matrixMulScalarLeft oneM A10))
      (matrixMulScalarLeft sM (matrixMulScalarLeft (matrixPolynomial [x]) G))) e0)
    (hc1 : c1 = matrixAdd (matrixSub (matrixAdd (matrixMulScalarLeft sM A01)
      (matrixMulScalarLeft oneM A11))
      (matrixMulScalarLeft oneM (matrixMulScalarLeft (matrixPolynomial [x]) G))) e1) :
    Rel sm (hashPair model key q n m P pre)
      (fun j ↦ reducePoly q n ((x : ErrorPoly n) * unitPair sm j) • G) ![c0, c1] cutoff.toNat := by
  subst hsQ
  rw [hashPair_of_samples h00 h01 h10 h11]
  exact encGadget_gen (modulusReduce_poly hQ hq hn hs) (modulusReduce_one hQ hq hn hone)
    A00 A01 A10 A11 G (matrixPolynomial_single_gen x) he0 he1 hc0 hc1

end RefreshTopProof.GenIface
