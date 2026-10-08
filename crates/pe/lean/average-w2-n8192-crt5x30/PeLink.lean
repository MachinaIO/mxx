import PeModel
import PeOnlineRel

/-!
# Runs follow the model

Every run of the protocol computes the values of `PeModel` from its tape: the secret, the anchor,
the attribute ciphertext, the Dynamic packages, the public keys, and the online output.
-/

namespace PeProof

open Mxx.Primitives MxxRuntime MxxExponentLut

theorem payload_explicit {tape : SampleTape} {out : ExactMatrix Q N 1 1}
    (h : Stage_payload.generatedRoot tape [2] GeneratedClaim.stage_2_params () out) :
    out = (fun _ _ ↦ reducePoly Q N (secOf tape)) ∧ polyNorm (secOf tape) ≤ 1 := by
  obtain ⟨w, h0, hout⟩ := h
  obtain ⟨hx, hb⟩ := ternarySampleAt_explicit hN h0
  subst hout
  refine ⟨hx, polyNorm_le_of_coeff fun k ↦ ?_⟩
  have h1 := hb k
  simp only [List.singleton_append] at h1
  rw [abs_le] at h1
  simp only [secOf]
  omega

theorem setup_explicit {tape : SampleTape} {out : ExactMatrix Q N 1 12 × TrapdoorValue (ExactMatrix Q N 1 12) Unit × Unit}
    (h : Stage_setup_trapdoor.generatedRoot Backend.backend tape [0] GeneratedClaim.stage_0_params () out) :
    out.1 = BOf tape ∧ out.2.1.publicMatrix = BOf tape ∧ out.2.1.kind = .sampledSecret := by
  obtain ⟨w, h0, hout⟩ := h
  subst hout
  obtain ⟨hs, layout, hl, hw, hpub⟩ := h0
  rw [layoutQ] at hl
  cases hl
  have hB : w.w_0_0 = BOf tape := by
    rw [hpub]
    rfl
  exact ⟨hB, (trapdoorSample_public hs).trans hB, trapdoorSample_sampled hs⟩

set_option maxHeartbeats 16000000 in
theorem prelude_link {model : HashModel} {tape : SampleTape} {x : Fin 1 → Int} {key : ByteArray}
    {m : Fin N → Int} {outputs : PreludeOutputs}
    (h : Stage_prelude.generatedRoot Backend.backend model tape [3] GeneratedClaim.stage_3_params
      (BOf tape, x, key, (fun _ _ ↦ reducePoly Q N (secOf tape)), m, ()) outputs)
    (hx : x 0 = 0) :
    ![outputs.1 0, outputs.2.1 0] = CtopOf tape ∧
      ![outputs.2.2.1 0, outputs.2.2.2.1 0] = CbotOf model key tape ∧
      outputs.2.2.2.2.1 = bOf model key tape ∧
      outputs.2.2.2.2.2.1 = reducePoly Q N (secOf tape) • messageRow model key 0 + messageRow model key 1 +
        reduceMatrix Q N 1 1 (emAt tape [3]) +
        ((halfQ : ExactPoly Q N) • (fun _ _ ↦ polynomialOfCoefficients m) : ExactMatrix Q N 1 1) ∧
      CoeffBound (ebAt tape [3]) 20 ∧ CoeffBound (zAt tape [3]) 20 ∧ CoeffBound (emAt tape [3]) 20 := by
  obtain ⟨hb, hcm, h00, h01, h10, h11, heb, hz, hem⟩ := prelude_explicit h hx rfl g0_runs
  have hb' : outputs.2.2.2.2.1 = bOf model key tape := hb.trans (bOf_eq model key tape).symm
  refine ⟨?_, ?_, hb', hcm, heb, hz, hem⟩
  · rw [CtopOf_eq]
    funext j
    fin_cases j
    · exact h00
    · exact h01
  · rw [CbotOf_eq, ← hb']
    funext j
    fin_cases j
    · exact h10
    · exact h11

theorem lane_eq (i : Fin 400) :
    (⟨(laneCol i : Nat) + 20 * (laneRow i : Nat), by have := (laneCol i).isLt; have := (laneRow i).isLt; omega⟩ :
      Fin 400) = i :=
  Fin.ext (by simp only [laneCol, laneRow]; omega)

set_option maxHeartbeats 64000000 in
theorem encrypt_link {model : HashModel} {tape : SampleTape} {key : ByteArray}
    {C00 C01 C10 C11 : Fin 1 → ExactMatrix Q N 1 10} {outputs : EncryptOutputs}
    (h : Stage_encrypt.generatedRoot Backend.backend model tape [4] GeneratedClaim.stage_4_params
      ((fun _ _ ↦ reducePoly Q N (secOf tape)), key, 1, 0, C00, C01, C10, C11, ()) outputs)
    (hCt : ![C00 0, C01 0] = CtopOf tape) (hCb : ![C10 0, C11 0] = CbotOf model key tape) :
    ![outputs.2.2.2.2.1, outputs.2.2.2.2.2.1] = c0Of model key tape ∧
      ![outputs.2.2.1 0, outputs.2.2.2.1 0] = hrowOf model key tape ∧
      (∀ i, ![outputs.2.2.2.2.2.2.1 i, outputs.2.2.2.2.2.2.2.1 i] =
        zOf model key tape (laneCol i) (laneRow i)) ∧
      (∀ i, ![outputs.1 i, outputs.2.1 i] = zaOf model key tape (laneCol i) (laneRow i)) := by
  have hhelp : ![outputs.2.2.1 0, outputs.2.2.2.1 0] = hrowOf model key tape := by
    rw [hrowOf_eq]
    exact (encrypt_helper h rfl g0_runs).eq_encRow
  refine ⟨?_, hhelp, fun i ↦ ?_, fun i ↦ ?_⟩
  · rw [c0Of_eq]
    exact (encrypt_input h rfl g0_runs).eq_encRow
  · have hz := (encrypt_rhs_digits h rfl g0_runs (laneCol i) (laneRow i)).eq_encRow
    rw [lane_eq i, hCt, hCb] at hz
    rw [zOf_eq]
    exact hz
  · have hz := (encrypt_aut_digits h rfl g0_runs (laneCol i) (laneRow i)).eq_encRow
    rw [lane_eq i, hhelp] at hz
    rw [zaOf_eq]
    exact hz

set_option maxHeartbeats 64000000 in
theorem preprocess_link {model : HashModel} {tape : SampleTape} {key : ByteArray}
    {outputs : PreprocessOutputs}
    (h : Stage_preprocess.generatedRoot Backend.backend model tape [5] GeneratedClaim.stage_5_params
      key outputs) :
    ![![outputs.1, outputs.2.1], ![outputs.2.2.1, outputs.2.2.2.1]] = MM model key ∧
      outputs.2.2.2.2.1 = TM model key 0 ∧ outputs.2.2.2.2.2.1 = TM model key 1 ∧
      ![![outputs.2.2.2.2.2.2.1, outputs.2.2.2.2.2.2.2.1],
        ![outputs.2.2.2.2.2.2.2.2.1, outputs.2.2.2.2.2.2.2.2.2.1]] = AFM model key ∧
      ![![outputs.2.2.2.2.2.2.2.2.2.2.1, outputs.2.2.2.2.2.2.2.2.2.2.2.1],
        ![outputs.2.2.2.2.2.2.2.2.2.2.2.2.1, outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.1]] = AFM model key := by
  obtain ⟨hM, hT, g, V1, VK, hg, hV1, hVK, hA, hA'⟩ := preprocess_keys h
  have hMM : ![![outputs.1, outputs.2.1], ![outputs.2.2.1, outputs.2.2.2.1]] = MM model key := by
    funext k j r l
    rw [Subsingleton.elim r 0, hM k j l]
    rfl
  have hTk : ∀ k : Fin 2, ![outputs.2.2.2.2.1, outputs.2.2.2.2.2.1] k = TM model key k := by
    intro k
    funext r p
    rw [Subsingleton.elim r 0]
    have hp := hT k ⟨p / 10, by omega⟩ ⟨p % 10, by omega⟩
    have hpe : (⟨10 * (p / 10 : Nat) + p % 10, by omega⟩ : Fin 20) = p := Fin.ext (by simp; omega)
    simp only [halves, rowWindow, hMM] at hp
    rw [hpe] at hp
    convert hp using 1
  have hV1' := V1c_eq hg hV1
  have hVK' := VKc_eq hg hVK
  have hAF : ∀ k j : Fin 2, outputKey model key
      ![![outputs.1, outputs.2.1], ![outputs.2.2.1, outputs.2.2.2.1]] outputs.2.2.2.2.1
      outputs.2.2.2.2.2.1 V1 VK k j = AFM model key k j := by
    intro k j
    rw [hMM, hV1', hVK', show outputs.2.2.2.2.1 = TM model key 0 from hTk 0,
      show outputs.2.2.2.2.2.1 = TM model key 1 from hTk 1]
    rfl
  refine ⟨hMM, hTk 0, hTk 1, ?_, ?_⟩
  · funext k j
    rw [hA k j, hAF]
  · funext k j
    rw [hA' k j, hAF]

set_option maxHeartbeats 64000000 in
theorem online_link {model : HashModel} {tape : SampleTape} {key : ByteArray}
    {v0 v1 v2 v3 c0a c0b : ExactMatrix Q N 1 10} {C00 C01 C10 C11 : Fin 1 → ExactMatrix Q N 1 10}
    {z0 z1 : Fin 400 → ExactMatrix Q N 1 10} {hf0 hf1 : Fin 1 → ExactMatrix Q N 1 10}
    {T0 T1 : ExactMatrix Q N 1 20} {za0 za1 : Fin 400 → ExactMatrix Q N 1 10}
    {M00 M01 M10 M11 : ExactMatrix Q N 1 10}
    {outputs : ExactMatrix Q N 1 10 × ExactMatrix Q N 1 10 × ExactMatrix Q N 1 10 ×
      ExactMatrix Q N 1 10 × ExactMatrix Q N 1 10 × ExactMatrix Q N 1 10 × Unit}
    (h : Stage_online_eval.generatedRoot Backend.backend model tape [7]
      GeneratedClaim.stage_7_params (v0, v1, v2, v3, c0a, c0b, C00, C01, C10, C11, z0, z1, key, hf0,
        hf1, T0, T1, za0, za1, M00, M01, M10, M11, ()) outputs)
    (hc0 : ![c0a, c0b] = c0Of model key tape) (hCt : ![C00 0, C01 0] = CtopOf tape)
    (hCb : ![C10 0, C11 0] = CbotOf model key tape)
    (hz : ∀ i, ![z0 i, z1 i] = zOf model key tape (laneCol i) (laneRow i))
    (hhf : ![hf0 0, hf1 0] = hrowOf model key tape)
    (hza : ∀ i, ![za0 i, za1 i] = zaOf model key tape (laneCol i) (laneRow i))
    (hM : ![![M00, M01], ![M10, M11]] = MM model key) (hT0 : T0 = TM model key 0)
    (hT1 : T1 = TM model key 1) :
    ![outputs.2.2.2.2.1, outputs.2.2.2.2.2.1] = cFOf model key tape ∧
      (∀ (j : Fin 2) (l : Fin 10), ∃ D' : ExactMatrix Q N (10 + 10) 1,
        gadgetDecomposeRuns Backend.backend 32768 10
          (pairColumn (CtopOf tape j 0 l) (CbotOf model key tape j 0 l)) D' ∧
          compactCenteredRebaseRuns 16384 D'
            (fun a _ ↦ actionDigits (joinRow (CtopOf tape)) (joinRow (CbotOf model key tape)) j a l)) ∧
      (∀ (j : Fin 2) (l : Fin 10), ∃ D' : ExactMatrix Q N (10 + 10) 1,
        gadgetDecomposeRuns Backend.backend 32768 10
          (pairColumn (halves (TM model key 0) j 0 l) (halves (TM model key 1) j 0 l)) D' ∧
          compactCenteredRebaseRuns 16384 D'
            (fun a _ ↦ actionDigits (TM model key 0) (TM model key 1) j a l)) ∧
      (∀ (j : Fin 2) (l : Fin 10), ∃ D' : ExactMatrix Q N (10 + 10) 1,
        gadgetDecomposeRuns Backend.backend 32768 10
          (pairColumn (switchTop model key j 0 l) (switchBot model key (hrowOf model key tape) j 0 l)) D' ∧
          compactCenteredRebaseRuns 16384 D' (fun a _ ↦ actionDigits (joinRow (switchTop model key))
            (joinRow (switchBot model key (hrowOf model key tape))) j a l)) ∧
      gadgetDecomposeRuns Backend.backend 32768 10 (matrixMulScalarLeft lutScalarOne g0) V1c ∧
      gadgetDecomposeRuns Backend.backend 32768 10
        (matrixMulScalarLeft (lutScalar (substituteMatrix K hK)) g0) VKc := by
  obtain ⟨g, V1, VK, hg, hV1, hVK, hD, hDt, hDK, -, -, -, -, hout⟩ := online_rows h
  have hz0 : z0 = fun i ↦ zOf model key tape (laneCol i) (laneRow i) 0 := funext fun i ↦ congrFun (hz i) 0
  have hz1 : z1 = fun i ↦ zOf model key tape (laneCol i) (laneRow i) 1 := funext fun i ↦ congrFun (hz i) 1
  have hza0 : za0 = fun i ↦ zaOf model key tape (laneCol i) (laneRow i) 0 :=
    funext fun i ↦ congrFun (hza i) 0
  have hza1 : za1 = fun i ↦ zaOf model key tape (laneCol i) (laneRow i) 1 :=
    funext fun i ↦ congrFun (hza i) 1
  subst hz0 hz1 hza0 hza1 hT0 hT1
  have hV1' := V1c_eq hg hV1
  have hVK' := VKc_eq hg hVK
  have hgg := gadgetMatrixRuns_unique hg g0_runs
  subst hV1' hVK' hgg
  rw [hc0, hCt, hCb, hhf, hM] at hout
  rw [hCt, hCb] at hD
  rw [hhf] at hDK
  refine ⟨?_, hD, hDt, hDK, hV1, hVK⟩
  rw [cFOf_eq]
  exact funext fun j ↦ hout j

/-- An explicit encoding row is an encoding with its named error. -/
theorem encRow_rel {m : Nat} {sm : ErrorPoly N} {A : Fin 2 → Fin 2 → ExactMatrix Q N 1 m}
    {t : Fin 2 → ExactMatrix Q N 1 m} {e : Fin 2 → ErrorMatrix N 1 m} {B : Nat}
    (he : ∀ j, CoeffBound (e j) B) : MxxExponentLut.Rel sm A t (encRow sm A t e) B :=
  ⟨e, he, fun _ ↦ rfl⟩

theorem lane_split (j : Fin 2) (l : Fin 10) (a : Fin 20) :
    laneCol ⟨10 * (j : Nat) + l + 20 * a, by omega⟩ = ⟨10 * (j : Nat) + l, by omega⟩ ∧
      laneRow ⟨10 * (j : Nat) + l + 20 * a, by omega⟩ = a := by
  constructor <;> apply Fin.ext <;> simp only [laneCol, laneRow] <;> omega

set_option maxHeartbeats 64000000 in
/-- The online output of the model is an encoding of `0` under `A_F`, with error at most
`onlineBound 20 (12 N 20²) 20 20`, given the cutoffs of the samples and the digit runs. -/
theorem online_error {model : HashModel} {key : ByteArray} {ω : SampleTape}
    (hb0 : ∀ j, CoeffBound (![errAt ω [4, 25] encLaw, errAt ω [4, 35] encLaw] j) 20)
    (hbh : ∀ j, CoeffBound (ehOf ω j) 20)
    (hbz : ∀ (i : Fin 400) j, CoeffBound (![errAt ω [4, 43, i, 26] encLaw, errAt ω [4, 43, i, 35] encLaw] j) 20)
    (hbza : ∀ (i : Fin 400) j, CoeffBound (![errAt ω [4, 8, i, 32] encLaw, errAt ω [4, 8, i, 41] encLaw] j) 20)
    (heb : CoeffBound (ebAt ω [3]) 20) (hzb : CoeffBound (zAt ω [3]) 20)
    (hD : ∀ (j : Fin 2) (l : Fin 10), ∃ D' : ExactMatrix Q N (10 + 10) 1,
        gadgetDecomposeRuns Backend.backend 32768 10
          (pairColumn (CtopOf ω j 0 l) (CbotOf model key ω j 0 l)) D' ∧
          compactCenteredRebaseRuns 16384 D'
            (fun a _ ↦ actionDigits (joinRow (CtopOf ω)) (joinRow (CbotOf model key ω)) j a l))
    (hDt : ∀ (j : Fin 2) (l : Fin 10), ∃ D' : ExactMatrix Q N (10 + 10) 1,
        gadgetDecomposeRuns Backend.backend 32768 10
          (pairColumn (halves (TM model key 0) j 0 l) (halves (TM model key 1) j 0 l)) D' ∧
          compactCenteredRebaseRuns 16384 D'
            (fun a _ ↦ actionDigits (TM model key 0) (TM model key 1) j a l))
    (hDK : ∀ (j : Fin 2) (l : Fin 10), ∃ D' : ExactMatrix Q N (10 + 10) 1,
        gadgetDecomposeRuns Backend.backend 32768 10
          (pairColumn (switchTop model key j 0 l) (switchBot model key (hrowOf model key ω) j 0 l)) D' ∧
          compactCenteredRebaseRuns 16384 D' (fun a _ ↦ actionDigits (joinRow (switchTop model key))
            (joinRow (switchBot model key (hrowOf model key ω))) j a l))
    (hV1 : gadgetDecomposeRuns Backend.backend 32768 10 (matrixMulScalarLeft lutScalarOne g0) V1c)
    (hVK : gadgetDecomposeRuns Backend.backend 32768 10
      (matrixMulScalarLeft (lutScalar (substituteMatrix K hK)) g0) VKc) :
    MxxExponentLut.Rel (secOf ω) (AFM model key) (fun _ ↦ 0) (cFOf model key ω)
      (onlineBound 20 (12 * N * 20 * 20) 20 20) := by
  have hrhs := prelude_rhs (model := model) (key := key) (tape := ω) (path := [3]) (B := BOf ω)
    (sm := secOf ω) (g := g0) (b := bOf model key ω) (C00 := CtopOf ω 0) (C01 := CtopOf ω 1)
    (C10 := CbotOf model key ω 0) (C11 := CbotOf model key ω 1) (bOf_eq model key ω)
    (by rw [CtopOf_eq]; rfl) (by rw [CtopOf_eq]; rfl) (by rw [CbotOf_eq]; rfl) (by rw [CbotOf_eq]; rfl)
  have hCt : ![CtopOf ω 0, CtopOf ω 1] = CtopOf ω := by
    funext j
    fin_cases j <;> rfl
  have hCb : ![CbotOf model key ω 0, CbotOf model key ω 1] = CbotOf model key ω := by
    funext j
    fin_cases j <;> rfl
  rw [hCt, hCb] at hrhs
  have hz : ∀ (j : Fin 2) (l : Fin 10) (a : Fin (10 + 10)), MxxExponentLut.Rel (secOf ω)
      (fun k r ↦ rhsKey model key (10 * (j : Nat) + l) a (2 * k + r))
      (fun r ↦ (actionDigits (joinRow (CtopOf ω)) (joinRow (CbotOf model key ω)) j a l *
        reducePoly Q N (unitPair (secOf ω) r)) • g0)
      ![zOf model key ω (laneCol ⟨10 * (j : Nat) + l + 20 * a, by omega⟩)
          (laneRow ⟨10 * (j : Nat) + l + 20 * a, by omega⟩) 0,
        zOf model key ω (laneCol ⟨10 * (j : Nat) + l + 20 * a, by omega⟩)
          (laneRow ⟨10 * (j : Nat) + l + 20 * a, by omega⟩) 1] 20 := by
    intro j l a
    obtain ⟨hc, hr⟩ := lane_split j l a
    rw [hc, hr]
    have hzz : ![zOf model key ω ⟨10 * (j : Nat) + l, by omega⟩ a 0,
        zOf model key ω ⟨10 * (j : Nat) + l, by omega⟩ a 1] = zOf model key ω ⟨10 * (j : Nat) + l, by omega⟩ a := by
      funext r
      fin_cases r <;> rfl
    rw [hzz, zOf_eq]
    exact encRow_rel (hbz ⟨10 * (j : Nat) + l + 20 * a, by omega⟩)
  have hza : ∀ (j : Fin 2) (l : Fin 10) (a : Fin (10 + 10)), MxxExponentLut.Rel
      (substitution K hK (secOf ω))
      (fun k r ↦ autKey model key (10 * (j : Nat) + l) a (2 * k + r))
      (fun r ↦ (actionDigits (joinRow (switchTop model key))
          (joinRow (switchBot model key (hrowOf model key ω))) j a l *
        reducePoly Q N (unitPair (substitution K hK (secOf ω)) r)) • g0)
      ![zaOf model key ω (laneCol ⟨10 * (j : Nat) + l + 20 * a, by omega⟩)
          (laneRow ⟨10 * (j : Nat) + l + 20 * a, by omega⟩) 0,
        zaOf model key ω (laneCol ⟨10 * (j : Nat) + l + 20 * a, by omega⟩)
          (laneRow ⟨10 * (j : Nat) + l + 20 * a, by omega⟩) 1] 20 := by
    intro j l a
    obtain ⟨hc, hr⟩ := lane_split j l a
    rw [hc, hr]
    have hzz : ![zaOf model key ω ⟨10 * (j : Nat) + l, by omega⟩ a 0,
        zaOf model key ω ⟨10 * (j : Nat) + l, by omega⟩ a 1] = zaOf model key ω ⟨10 * (j : Nat) + l, by omega⟩ a := by
      funext r
      fin_cases r <;> rfl
    rw [hzz, zaOf_eq]
    exact encRow_rel (hbza ⟨10 * (j : Nat) + l + 20 * a, by omega⟩)
  have hM : ∀ (k j : Fin 2) (l : Fin 10), MM model key k j 0 l =
      (∑ a : Fin (10 + 10), (rhsKey model key (10 * (j : Nat) + l) a (2 * k + 0) * keyDigits (inputKey model key) a 0 +
        rhsKey model key (10 * (j : Nat) + l) a (2 * k + 1) * keyDigits (inputKey model key) a 1)) 0 0 := by
    intro k j l
    simp only [MM, keySum, Matrix.sum_apply, add_zero]
  have hT : ∀ (k j : Fin 2) (l : Fin 10), halves (![TM model key 0, TM model key 1] k) j 0 l =
      (∑ a : Fin (10 + 10), (autKey model key (10 * (j : Nat) + l) a (2 * k + 0) * keyDigits (routedKey (MM model key)) a 0 +
        autKey model key (10 * (j : Nat) + l) a (2 * k + 1) *
          keyDigits (routedKey (MM model key)) a 1)) 0 0 := by
    intro k j l
    fin_cases k <;> simp [halves, rowWindow, TM, keySum, Matrix.sum_apply]
  have hin : MxxExponentLut.Rel (secOf ω) (inputKey model key)
      (fun j ↦ reducePoly Q N (((1 : Nat) : Int) * 1 * unitPair (secOf ω) j) • g0) (c0Of model key ω) 20 := by
    rw [c0Of_eq]
    exact encRow_rel hb0
  have hh : MxxExponentLut.Rel (secOf ω) (helperKey model key)
      (fun j ↦ reducePoly Q N ((((1 : Nat) : Int) : ErrorPoly N) *
        unitPair (substitution K hK (secOf ω)) j) • g0) (hrowOf model key ω) 20 := by
    rw [hrowOf_eq]
    exact encRow_rel hbh
  rw [cFOf_eq]
  exact online_rel (model := model) (key := key) (s := secOf ω) (g := g0) (V1 := V1c) (VK := VKc)
    (cIn := c0Of model key ω) (Ctop := CtopOf ω) (Cbot := CbotOf model key ω)
    (Bm := fun j l a k r ↦ rhsKey model key (10 * (j : Nat) + l) a (2 * k + r))
    (z0 := fun i ↦ zOf model key ω (laneCol i) (laneRow i) 0)
    (z1 := fun i ↦ zOf model key ω (laneCol i) (laneRow i) 1) (Mmul := MM model key)
    (hrow := hrowOf model key ω)
    (Ba := fun j l a k r ↦ autKey model key (10 * (j : Nat) + l) a (2 * k + r))
    (za0 := fun i ↦ zaOf model key ω (laneCol i) (laneRow i) 0)
    (za1 := fun i ↦ zaOf model key ω (laneCol i) (laneRow i) 1)
    (T0 := TM model key 0) (T1 := TM model key 1)
    g0_runs hV1 hVK hin hrhs (rhsErr_bound heb hzb) hD hz hM hh hDK hza hT hDt

end PeProof
