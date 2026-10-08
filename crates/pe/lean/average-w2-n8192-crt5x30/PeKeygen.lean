import PePrelude

/-!
# Key generation

Key generation samples the extension `v = (v₀, v₁, v_F)` of the key and, per anchor block `i`, a
short preimage `k̃ᵢ` of the target `σᵢ(tᵢ)` under the anchor `B`, where
`t₀ = d₁ - A_F⁰ v_F - J v₀ + [g | 0] v₁` and `t₁ = -σ_K(J) v₁ + [g | 0] v₀` for the joined
helper-key row `J = [H₀₀ | H₀₁]` of `K`.
-/

namespace PeProof

open Mxx.Primitives MxxRuntime MxxExponentLut

/-- The law of the key extension. -/
abbrev keyLaw : SamplerLaw := .gaussian (36955943884422039849686363886844743345501063507786308997710330263389608971746115995555182166571772460446609040073468199866370528568323731719438794004731564382523252973706112397296780774527489663820602174997088646797830840886459663010452528843218822143/71362384635297994052914298472474756819137331200000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000) 3366109981

/-- The extension of helper block `i`. -/
noncomputable def vAt (tape : SampleTape) (path : List Nat) (i : Nat) : ErrorMatrix N 20 1 :=
  tapeWitness tape (path ++ [10, i, 0]) keyLaw

/-- The extension of the output block. -/
noncomputable def vFAt (tape : SampleTape) (path : List Nat) : ErrorMatrix N 20 1 :=
  tapeWitness tape (path ++ [7]) keyLaw

/-- The preimage cutoff. -/
abbrev keyCutoff : Nat := 3366109981

abbrev KeygenOutputs := ExactMatrix Q N 12 1 × ExactMatrix Q N 12 1 × (Fin 2 → ExactMatrix Q N 20 1) ×
  ExactMatrix Q N 20 1 × Unit

/-- The joined row `[g | 0]`. -/
noncomputable def gZero (g : ExactMatrix Q N 1 10) : ExactMatrix Q N 1 20 := joinRow ![g, 0]

theorem keyBound {tape : SampleTape} {site : List Nat} {x : ExactMatrix Q N 20 1}
    (h : gaussianSampleAt tape site (36955943884422039849686363886844743345501063507786308997710330263389608971746115995555182166571772460446609040073468199866370528568323731719438794004731564382523252973706112397296780774527489663820602174997088646797830840886459663010452528843218822143/71362384635297994052914298472474756819137331200000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000) 3366109981 x) :
    x = reduceMatrix Q N 20 1 (tapeWitness tape site keyLaw) ∧
      CoeffBound (tapeWitness (n := N) (rows := 20) (columns := 1) tape site keyLaw) keyCutoff := by
  obtain ⟨hx, hb⟩ := gaussianSampleAt_explicit hN h
  exact ⟨hx, fun r c k ↦ by have := hb r c k; simpa using this⟩

/-- A dispatched preimage under a sampled trapdoor solves its target and is short. -/
theorem preimage_short {B : ExactMatrix Q N 1 12} {td : TrapdoorValue (ExactMatrix Q N 1 12) Unit}
    (hkind : td.kind = .sampledSecret) {t : ExactMatrix Q N 1 1} {k : ExactMatrix Q N 12 1}
    (h : preimageRunsDispatched Backend.backend B td t (Int.toNat 3366109981) k) :
    B * k = t ∧ ∃ k' : ErrorMatrix N 12 1, k = reduceMatrix Q N 12 1 k' ∧ CoeffBound k' keyCutoff := by
  refine ⟨preimageRunsDispatched_equation hQ hN h, ?_⟩
  rcases h.2 with h | h
  · exact preimageRuns_bounded h
  · exact absurd (hkind.symm.trans h.1) (by decide)


theorem selectAut1 : MxxIR.selectInt (Int.ofNat 1) [1, 8193] = (K : Int) := by decide
theorem selectAut0 : MxxIR.selectInt (Int.ofNat 0) [1, 8193] = 1 := by decide

/-- A routing by `σ_K` written with its literal index. -/
theorem routing_K' {rows columns : Nat} {x y : ExactMatrix Q N rows columns}
    (h : ringAutomorphismRuns (K : Int) x y) : y = substituteMatrix K hK x := by
  obtain ⟨hodd, hy⟩ := ringAutomorphismRuns_eq hQ hN h
  rw [hy]
  funext r c
  exact congrFun (congrArg DFunLike.coe (substitution_congr (by decide) hodd hK)) _

set_option maxHeartbeats 64000000 in
/-- The target of anchor block `i`. -/
theorem keygen_target {model : HashModel} {tape : SampleTape} {path : List Nat} {key : ByteArray}
    {d : ExactMatrix Q N 1 1} {v : Fin 2 → ExactMatrix Q N 20 1} {gz : ExactMatrix Q N 1 20}
    (i : Fin 2) {out : ExactMatrix Q N 1 1}
    (h : Stage_keygen.parallel_generatedRoot_14 Backend.backend model tape path
      GeneratedClaim.stage_6_params i (d, key, v, gz, ()) out) :
    let J := joinRow (helperKey model key 0)
    (i = 0 → out = d - J * v 0 + gz * v 1) ∧
      (i = 1 → out = substituteMatrix K hK (-(substituteMatrix K hK J * v 1) + gz * v 0)) := by
  intro J
  obtain ⟨w, -, -, -, h8, -, -, h10, hout⟩ := h
  have hcount : Int.toNat 1 = 0 + 1 := rfl
  rw [hcount] at h8
  obtain ⟨cur, h0, hb⟩ := MxxIR.IterRuns.step_of_succ h8
  cases h0
  obtain ⟨w2, w3, w4, w5, w8, w13, -, -, x2, -, -, x3, x4, -, -, x5, -, -, x8, -, -, -, -, x13,
    xout⟩ := hb
  have hs : MxxIR.selectInt (Int.ofNat 0) [8193] = (K : Int) := by decide
  rw [hs] at x2 x3
  have hJ : w4 = J := by
    have hkey : ∀ j, helperKey model key 0 j = ![w2, w3] j := by
      intro j
      fin_cases j
      · exact (hashSample_eq x2).symm
      · exact (hashSample_eq x3).symm
    have := concat_joinRow (r := ![w2, w3]) x4
    rw [this]
    congr 1
    funext j
    exact (hkey j).symm
  subst hJ hout
  rw [xout] at h10
  refine ⟨fun hi ↦ ?_, fun hi ↦ ?_⟩
  · subst hi
    rw [show MxxIR.selectInt (Int.ofNat ((0 : Fin 2) : Nat)) [1, 8193] = 1 from by decide] at x5 h10
    have hw5 := routing_one x5
    have hw10 := routing_one h10
    have hi8 : Int.ofNat ((0 : Fin 2) : Nat) + Int.ofNat 0 = Int.ofNat 0 := by decide
    have hi13 : Int.ofNat 0 + MxxIR.selectInt (Int.ofNat ((0 : Fin 2) : Nat) + Int.ofNat 0) [1, 0] =
        Int.ofNat 1 := by decide
    rw [hi8] at x8
    rw [hi13] at x13
    obtain ⟨_, hw8⟩ := familyGet_eq x8
    obtain ⟨_, hw13⟩ := familyGet_eq x13
    rw [hw10]
    simp only [matrixSub, matrixAdd, matrixMul, matrixMulScalarLeft_eq, hw5, hw8, hw13]
    have hsel : MxxIR.selectInt (Int.ofNat ((0 : Fin 2) : Nat)) [1, 0] = 1 := by decide
    simp only [hsel, liftInteger]
    funext r c
    simp
    abel
  · subst hi
    rw [show MxxIR.selectInt (Int.ofNat ((1 : Fin 2) : Nat)) [1, 8193] = (K : Int) from by decide] at x5 h10
    have hw5 := routing_K' x5
    have hw10 := routing_K' h10
    have hi8 : Int.ofNat ((1 : Fin 2) : Nat) + Int.ofNat 0 = Int.ofNat 1 := by decide
    have hi13 : Int.ofNat 0 + MxxIR.selectInt (Int.ofNat ((1 : Fin 2) : Nat) + Int.ofNat 0) [1, 0] =
        Int.ofNat 0 := by decide
    rw [hi8] at x8
    rw [hi13] at x13
    obtain ⟨_, hw8⟩ := familyGet_eq x8
    obtain ⟨_, hw13⟩ := familyGet_eq x13
    rw [hw10]
    simp only [matrixSub, matrixAdd, matrixMul, matrixMulScalarLeft_eq, hw5, hw8, hw13]
    have hsel : MxxIR.selectInt (Int.ofNat ((1 : Fin 2) : Nat)) [1, 0] = 0 := by decide
    simp only [hsel, liftInteger]
    congr 1
    funext r c
    simp
    abel

set_option maxHeartbeats 64000000 in
theorem keygen_explicit {model : HashModel} {tape : SampleTape} {path : List Nat}
    {td : TrapdoorValue (ExactMatrix Q N 1 12) Unit} {key : ByteArray} {a0 a1 : ExactMatrix Q N 1 10}
    {outputs : KeygenOutputs}
    (h : Stage_keygen.generatedRoot Backend.backend model tape path GeneratedClaim.stage_6_params
      (td, key, a0, a1, ()) outputs)
    (hkind : td.kind = .sampledSecret) {g : ExactMatrix Q N 1 10}
    (hg : gadgetMatrixRuns Backend.backend 32768 10 g) :
    let vF := reduceMatrix Q N 20 1 (vFAt tape path)
    let v : Fin 2 → ExactMatrix Q N 20 1 := fun i ↦ reduceMatrix Q N 20 1 (vAt tape path i)
    let J := joinRow (helperKey model key 0)
    outputs.2.2.2.1 = vF ∧ outputs.2.2.1 = v ∧ CoeffBound (vFAt tape path) keyCutoff ∧
      (∀ i : Fin 2, CoeffBound (vAt tape path i) keyCutoff) ∧
      td.publicMatrix * outputs.1 =
        messageRow model key 0 - joinRow ![a0, a1] * vF - J * v 0 + gZero g * v 1 ∧
      td.publicMatrix * outputs.2.1 =
        substituteMatrix K hK (-(substituteMatrix K hK J * v 1) + gZero g * v 0) ∧
      (∃ k' : ErrorMatrix N 12 1, outputs.1 = reduceMatrix Q N 12 1 k' ∧ CoeffBound k' keyCutoff) ∧
      (∃ k' : ErrorMatrix N 12 1, outputs.2.1 = reduceMatrix Q N 12 1 k' ∧ CoeffBound k' keyCutoff) := by
  intro vF v J
  obtain ⟨w, h3, h6, h7, -, h10, h11, h13, -, h14, h15, -, h16, h17, -, h18, hout⟩ := h
  subst hout
  obtain ⟨hvF, hbF⟩ := keyBound h7
  have hv : ∀ i : Fin 2, w.w_10_0 i = v i ∧ CoeffBound (vAt tape path i) keyCutoff := by
    intro i
    obtain ⟨w0, x0, xout⟩ := h10 i
    rw [List.append_assoc] at x0
    obtain ⟨hx, hb⟩ := keyBound x0
    exact ⟨xout.trans hx, hb⟩
  have hw10 : w.w_10_0 = v := funext fun i ↦ (hv i).1
  have hg11 : w.w_11_0 = g := gadgetMatrixRuns_unique h11 hg
  have hw13 : w.w_13_0 = gZero g := by
    rw [concat_joinRow (r := ![w.w_11_0, 0]) h13, hg11]
    rfl
  have hw3 : w.w_3_0 = messageRow model key 0 := hashSample_eq h3
  have hw6 : w.w_6_0 = joinRow ![a0, a1] := concat_joinRow (r := ![a0, a1]) h6
  have ht0 := (keygen_target 0 (h14 0)).1 rfl
  have ht1 := (keygen_target 1 (h14 1)).2 rfl
  obtain ⟨_, hw15⟩ := familyGet_eq (k := 0) h15
  obtain ⟨_, hw17⟩ := familyGet_eq (k := 1) h17
  obtain ⟨he16, hk16⟩ := preimage_short hkind h16
  obtain ⟨he18, hk18⟩ := preimage_short hkind h18
  rw [hw15] at he16
  rw [hw17] at he18
  refine ⟨hvF, hw10, hbF, fun i ↦ (hv i).2, ?_, ?_, hk16, hk18⟩
  · show td.publicMatrix * w.w_16_0 = _
    rw [show td.publicMatrix = trapdoorPublic td from rfl, he16]
    refine ht0.trans ?_
    rw [hw10, hw13, hw3, hw6, hvF]
    simp [matrixSub, matrixMul]
    rfl
  · show td.publicMatrix * w.w_18_0 = _
    rw [show td.publicMatrix = trapdoorPublic td from rfl, he18]
    refine ht1.trans ?_
    rw [hw10, hw13]

end PeProof
