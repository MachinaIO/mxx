import DiamondProofParameters
import DiamondCircuitRequirementProof

open Mxx.Primitives MxxRuntime
open DiamondProofParameters

namespace DiamondGeneratedProof

set_option maxRecDepth 8192
set_option maxHeartbeats 1600000

/-- The public projection of an actual decrypt layer is an execution of the actual
    encrypt public layer. All candidates, gathers, decomposition and masks are retained. -/
theorem generated_decrypt_public_layer_runs (backend : BackendContext) (hashModel : HashModel)
    (params : Stage_decrypt.Params) (encryptParams : Stage_encrypt.Params) (layer : Nat)
    (current next : CircuitState) (activeCounts : Fin circuitDepth → Int)
    (kinds leftSources rightSources : Fin metadataCount → Int)
    (oneCipher onePublic : ExactMatrix q n 1 ell) (oneMessage : ExactMatrix q n 1 1)
    (hwidth : params.max_layer_width = encryptParams.max_layer_width)
    (hbase : params.diamond_gadget_base = encryptParams.diamond_gadget_base)
    (hdigits : params.diamond_digit_count = encryptParams.diamond_digit_count)
    (hrun : Stage_decrypt.sequential_generatedRoot_67 backend params layer
      (current.1, current.2.1, current.2.2.1, activeCounts, kinds, leftSources,
        rightSources, oneCipher, onePublic, oneMessage, ()) next) :
    Stage_encrypt.sequential_generatedRoot_32 backend hashModel encryptParams layer
      (current.2.1, activeCounts, kinds, leftSources, rightSources, onePublic, ()) next.2.1 := by
  dsimp only [Stage_decrypt.sequential_generatedRoot_67] at hrun
  rcases hrun with ⟨active, flags, w5, w6, addresses, gateKinds, leftIndices, w13, w14,
    rightIndices, w18, digits, w20, w21, w23, w24, w25, w26, w28, w29, w30, w31,
    w33, w34, w35, w36, w37, w38, w39, w40, w41, w42, w44, w45, w46, w47,
    w48, w49, w50, w51, w52, w53, h⟩
  rcases h with ⟨han, halt, ha, hw, h3, _, _, _, _, _, h7, _, h9, _, h11, _, _,
    _, _, _, h16, _, h18, _, h19, _, _, _, _, _, _, _, _, _, _,
    _, _, _, _, _, _, _, _, _, _, _, h33, _, h34, _, h35, _, h36,
    _, h37, _, h38, _, h39, _, h40, _, h41, _, h42, _, _, _, _, _, _,
    _, _, _, _, _, _, _, _, _, _, _, _, _, _, hout⟩
  rw [hout]
  have hw' : encryptParams.max_layer_width = (circuitWidth : Int) := hwidth.symm.trans hw
  refine ⟨addresses, gateKinds, leftIndices, w35, rightIndices, w18, active, w42,
    hw', ?_, hw', h9, hw', h11, hw', h35, hw', h16, hw', h18,
    han, halt, ha, hw', ?_, rfl⟩
  · intro lane
    simpa only [Stage_encrypt.parallel_sequential_generatedRoot_32_0,
      Stage_decrypt.parallel_sequential_generatedRoot_67_7, hwidth] using h7 lane
  · intro lane
    obtain ⟨selected, hkn, hklt, hlen, hsel, hselout⟩ := h41 lane
    obtain ⟨masked, hfn, hflt, hflen, hmask, hmaskout⟩ := h42 lane
    have hg : gadgetDecomposeRuns backend encryptParams.diamond_gadget_base
        encryptParams.diamond_digit_count (w18 lane) (digits lane) := by
      obtain ⟨value, hg, hout⟩ := h19 lane
      simpa only [hbase, hdigits, ← hout] using hg
    have hone : w33 lane = onePublic := h33 lane
    have hz : w34 lane = matrixSub onePublic onePublic := by rw [h34 lane, hone]
    have hn : w36 lane = matrixSub onePublic (w35 lane) := by rw [h36 lane, hone]
    have hm : w37 lane = matrixMul (w35 lane) (digits lane) := h37 lane
    have hx : w40 lane = matrixSub (matrixAdd (w35 lane) (w18 lane))
        (matrixMulScalarRight (matrixMul (w35 lane) (digits lane)) (matrixPolynomial [2])) := by
      rw [h40 lane, h38 lane, h39 lane, hm]
    have hf : flags lane = if decide ((lane.val : Int) ≤ active - 1) then 1 else 0 := h3 lane
    refine ⟨digits lane, selected, masked, hg, hkn, hklt, hlen, ?_, ?_, ?_, hflen, ?_, hmaskout⟩
    · simpa only [hz, hone, hn, hm, hx] using hsel
    · simpa only [hf] using hfn
    · simpa only [hf] using hflt
    · simpa only [hf, hz, hselout] using hmask

theorem public_select_unique {α : Type} {values : List α} {index : Int} {left right : α}
    (hl : MxxRuntime.select index values left) (hr : MxxRuntime.select index values right) :
    left = right := by
  obtain ⟨il, hil, hl⟩ := hl
  obtain ⟨ir, hir, hr⟩ := hr
  have heq : il = ir := Fin.ext (by omega)
  exact hl.trans (heq ▸ hr.symm)

theorem public_gather_scope_unique {α : Type} {N : Nat} {values : Fin N → α}
    {index : Int} {left right : α}
    (hl : ∃ value, 0 ≤ index ∧ index < N ∧ familyGetDynamic values index value ∧ left = value)
    (hr : ∃ value, 0 ≤ index ∧ index < N ∧ familyGetDynamic values index value ∧ right = value) :
    left = right := by
  obtain ⟨l, _, _, hl, hlo⟩ := hl
  obtain ⟨r, _, _, hr, hro⟩ := hr
  exact hlo.trans ((circuit_lookup_unique hl hr).trans hro.symm)

theorem generated_public_gate_deterministic (backend : BackendContext) (hashModel : HashModel)
    (params : Stage_encrypt.Params) (layer lane : Nat) (kind active : Int)
    (left right one first second : ExactMatrix q n 1 ell)
    (hf : Stage_encrypt.parallel_sequential_generatedRoot_32_14 backend hashModel params
      layer lane (kind, left, right, active, one, ()) first)
    (hs : Stage_encrypt.parallel_sequential_generatedRoot_32_14 backend hashModel params
      layer lane (kind, left, right, active, one, ()) second) : first = second := by
  obtain ⟨df, sf, mf, hdf, _, _, _, hsf, _, _, _, hmf, hof⟩ := hf
  obtain ⟨ds, ss, ms, hds, _, _, _, hss, _, _, _, hms, hos⟩ := hs
  have hd : df = ds := gadgetDecomposeRuns_deterministic hdf hds
  subst ds
  have hselected : sf = ss := public_select_unique hsf hss
  subst ss
  exact hof.trans ((public_select_unique hmf hms).trans hos.symm)

/-- Determinism of the actual public layer, with metadata and public inputs shared. -/
theorem generated_public_layer_deterministic (backend : BackendContext) (hashModel : HashModel)
    (params : Stage_encrypt.Params) (layer : Nat)
    (current first second : Fin circuitWidth → ExactMatrix q n 1 ell)
    (activeCounts : Fin circuitDepth → Int) (kinds leftSources rightSources : Fin metadataCount → Int)
    (one : ExactMatrix q n 1 ell)
    (hf : Stage_encrypt.sequential_generatedRoot_32 backend hashModel params layer
      (current, activeCounts, kinds, leftSources, rightSources, one, ()) first)
    (hs : Stage_encrypt.sequential_generatedRoot_32 backend hashModel params layer
      (current, activeCounts, kinds, leftSources, rightSources, one, ()) second) : first = second := by
  obtain ⟨af, kf, lif, lf, rif, rf, nf, of, _, haf, _, hkf, _, hlif, _, hlf,
    _, hrif, _, hrf, _, _, hnf, _, hof, houtf⟩ := hf
  obtain ⟨as', ks, lis, ls, ris, rs, ns, os, _, has, _, hks, _, hlis, _, hls,
    _, hris, _, hrs, _, _, hns, _, hos, houts⟩ := hs
  have ha : af = as' := funext fun lane ↦ (haf lane).trans (has lane).symm
  subst as'
  have hk : kf = ks := funext fun lane ↦ public_gather_scope_unique (hkf lane) (hks lane)
  subst ks
  have hli : lif = lis := funext fun lane ↦ public_gather_scope_unique (hlif lane) (hlis lane)
  subst lis
  have hl : lf = ls := funext fun lane ↦ public_gather_scope_unique (hlf lane) (hls lane)
  subst ls
  have hri : rif = ris := funext fun lane ↦ public_gather_scope_unique (hrif lane) (hris lane)
  subst ris
  have hr : rf = rs := funext fun lane ↦ public_gather_scope_unique (hrf lane) (hrs lane)
  subst rs
  have hn : nf = ns := circuit_lookup_unique hnf hns
  subst ns
  rw [houtf, houts]
  exact funext fun lane ↦ generated_public_gate_deterministic backend hashModel params layer lane
    (kf lane) nf (lf lane) (rf lane) one (of lane) (os lane) (hof lane) (hos lane)

/-- Paired induction over actual encrypt/decrypt circuit loops, including zero layers. -/
theorem generated_circuit_public_iteration_agrees (backend : BackendContext) (hashModel : HashModel)
    (params : Stage_decrypt.Params) (encryptParams : Stage_encrypt.Params) (count : Nat)
    (initial output : CircuitState)
    (publicInitial publicOutput : Fin circuitWidth → ExactMatrix q n 1 ell)
    (activeCounts : Fin circuitDepth → Int) (kinds leftSources rightSources : Fin metadataCount → Int)
    (oneCipher onePublic : ExactMatrix q n 1 ell) (oneMessage : ExactMatrix q n 1 1)
    (hwidth : params.max_layer_width = encryptParams.max_layer_width)
    (hbase : params.diamond_gadget_base = encryptParams.diamond_gadget_base)
    (hdigits : params.diamond_digit_count = encryptParams.diamond_digit_count)
    (hinitial : initial.2.1 = publicInitial)
    (hdecrypt : MxxIR.IterRuns
      (fun layer current next ↦ Stage_decrypt.sequential_generatedRoot_67 backend params layer
        (current.1, current.2.1, current.2.2.1, activeCounts, kinds, leftSources,
          rightSources, oneCipher, onePublic, oneMessage, ()) next) count initial output)
    (hencrypt : MxxIR.IterRuns
      (fun layer current next ↦ Stage_encrypt.sequential_generatedRoot_32 backend hashModel
        encryptParams layer (current, activeCounts, kinds, leftSources, rightSources,
          onePublic, ()) next) count publicInitial publicOutput) :
    output.2.1 = publicOutput := by
  induction hdecrypt generalizing publicOutput with
  | zero =>
      cases hencrypt
      exact hinitial
  | @step count initial current next hprevious hstep ih =>
      obtain ⟨publicCurrent, hpublicPrevious, hpublicStep⟩ :=
        MxxIR.IterRuns.step_of_succ hencrypt
      have hcurrent := ih publicCurrent hinitial hpublicPrevious
      have hprojection := generated_decrypt_public_layer_runs backend hashModel params
        encryptParams count current next activeCounts kinds leftSources rightSources
        oneCipher onePublic oneMessage hwidth hbase hdigits hstep
      rw [hcurrent] at hprojection
      exact generated_public_layer_deterministic backend hashModel encryptParams count
        publicCurrent next.2.1 publicOutput activeCounts kinds leftSources rightSources
        onePublic hprojection hpublicStep

/-- The actual encrypt output lookup identifies the decrypt public key at the same
    output address. In root assembly this is the hidden publicCircuit used by preimages. -/
theorem generated_public_output_lookup_agrees
    (output : CircuitState) (publicOutput : Fin circuitWidth → ExactMatrix q n 1 ell)
    (publicCircuit : ExactMatrix q n 1 ell) (index : Int) (position : Fin circuitWidth)
    (hagrees : output.2.1 = publicOutput)
    (hposition : (position.val : Int) = index)
    (hlookup : familyGetDynamic publicOutput index publicCircuit) :
    output.2.1 position = publicCircuit := by
  rw [hagrees]
  exact circuit_lookup_unique ⟨position, hposition, rfl⟩ hlookup

#print axioms generated_decrypt_public_layer_runs
#print axioms generated_public_gate_deterministic
#print axioms generated_public_layer_deterministic
#print axioms generated_circuit_public_iteration_agrees
#print axioms generated_public_output_lookup_agrees

end DiamondGeneratedProof
