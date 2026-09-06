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
    (hrun : Stage_decrypt.sequential_generatedRoot_33 backend params layer
      (current.1, current.2.1, current.2.2.1, activeCounts, oneCipher, kinds, leftSources,
        rightSources, onePublic, oneMessage, ()) next) :
    Stage_encrypt.sequential_generatedRoot_27 backend hashModel encryptParams layer
      (current.2.1, activeCounts, onePublic, kinds, leftSources, rightSources, ()) next.2.1 := by
  obtain ⟨⟨active, outCipher, outPublic, outMessage⟩, han, halt, ha, hw, hlanes, hout⟩ := hrun
  rw [hout]
  refine ⟨⟨active, outPublic⟩, han, halt, ha, hwidth.symm.trans hw, ?_, rfl⟩
  intro lane
  have h := hlanes lane
  dsimp only [Stage_decrypt.parallel_sequential_generatedRoot_33_12] at h
  rcases h with ⟨ki, li, lc, ri, rp, digits, rc, lm, sc, oc, lp, sp, op, rm, sm, om, h⟩
  dsimp only [Stage_decrypt.parallel_sequential_generatedRoot_33_12.constraints_0,
    Stage_decrypt.parallel_sequential_generatedRoot_33_12.constraints_1] at h
  rcases h with ⟨han', halt', hk, _, _, hl, _, _, hlc, _, _, hr, hrn, hrlt, hrp, hd,
    _, _, hrc, _, _, hlm, hkn, hklt, _, hsc, hfn, hflt, _, hoc, hln, hllt, hlp,
    _, _, _, hsp, _, _, _, hop, _, _, hrm, _, _, _, hsm, _, _, _, hom, hout'⟩
  dsimp only [Stage_encrypt.parallel_sequential_generatedRoot_27_8]
  refine ⟨ki, li, lp, ri, rp, digits, sp, op, ?_⟩
  dsimp only [Stage_encrypt.parallel_sequential_generatedRoot_27_8.constraints_0]
  rw [← hwidth, ← hbase, ← hdigits]
  refine ⟨han', halt', hk, han', halt', hl, hln, hllt, hlp,
    han', halt', hr, hrn, hrlt, hrp, hd, hkn, hklt, rfl, hsp, hfn, hflt, rfl, hop, ?_⟩
  exact congrArg (fun value => value.2.1) hout'

theorem public_select_unique {α : Type} {values : List α} {index : Int} {left right : α}
    (hl : MxxRuntime.select index values left) (hr : MxxRuntime.select index values right) :
    left = right := by
  obtain ⟨il, hil, hl⟩ := hl
  obtain ⟨ir, hir, hr⟩ := hr
  have heq : il = ir := Fin.ext (by omega)
  exact hl.trans (heq ▸ hr.symm)

theorem generated_public_gate_deterministic (backend : BackendContext) (hashModel : HashModel)
    (params : Stage_encrypt.Params) (layer lane : Nat) (active : Int)
    (current : Fin circuitWidth → ExactMatrix q n 1 ell)
    (kinds leftSources rightSources : Fin metadataCount → Int)
    (one first second : ExactMatrix q n 1 ell)
    (hf : Stage_encrypt.parallel_sequential_generatedRoot_27_8 backend hashModel params
      layer lane (active, one, kinds, (layer : Int), current, leftSources, rightSources, ()) first)
    (hs : Stage_encrypt.parallel_sequential_generatedRoot_27_8 backend hashModel params
      layer lane (active, one, kinds, (layer : Int), current, leftSources, rightSources, ()) second) :
    first = second := by
  dsimp only [Stage_encrypt.parallel_sequential_generatedRoot_27_8,
    Stage_encrypt.parallel_sequential_generatedRoot_27_8.constraints_0] at hf hs
  obtain ⟨kf, lif, lf, rif, rf, df, sf, mf, _, _, hkf, _, _, hlif, _, _, hlf,
    _, _, hrif, _, _, hrf, hdf, _, _, _, hsf, _, _, _, hmf, hof⟩ := hf
  obtain ⟨ks, lis, ls, ris, rs, ds, ss, ms, _, _, hks, _, _, hlis, _, _, hls,
    _, _, hris, _, _, hrs, hds, _, _, _, hss, _, _, _, hms, hos⟩ := hs
  have hk : kf = ks := circuit_lookup_unique hkf hks
  subst ks
  have hli : lif = lis := circuit_lookup_unique hlif hlis
  subst lis
  have hl : lf = ls := circuit_lookup_unique hlf hls
  subst ls
  have hri : rif = ris := circuit_lookup_unique hrif hris
  subst ris
  have hr : rf = rs := circuit_lookup_unique hrf hrs
  subst rs
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
    (hf : Stage_encrypt.sequential_generatedRoot_27 backend hashModel params layer
      (current, activeCounts, one, kinds, leftSources, rightSources, ()) first)
    (hs : Stage_encrypt.sequential_generatedRoot_27 backend hashModel params layer
      (current, activeCounts, one, kinds, leftSources, rightSources, ()) second) : first = second := by
  obtain ⟨⟨nf, of⟩, _, _, hnf, _, hof, houtf⟩ := hf
  obtain ⟨⟨ns, os⟩, _, _, hns, _, hos, houts⟩ := hs
  have hn : nf = ns := circuit_lookup_unique hnf hns
  subst ns
  rw [houtf, houts]
  exact funext fun lane ↦ generated_public_gate_deterministic backend hashModel params layer lane
    nf current kinds leftSources rightSources one (of lane) (os lane) (hof lane) (hos lane)

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
      (fun layer current next ↦ Stage_decrypt.sequential_generatedRoot_33 backend params layer
        (current.1, current.2.1, current.2.2.1, activeCounts, oneCipher, kinds, leftSources,
          rightSources, onePublic, oneMessage, ()) next) count initial output)
    (hencrypt : MxxIR.IterRuns
      (fun layer current next ↦ Stage_encrypt.sequential_generatedRoot_27 backend hashModel
        encryptParams layer (current, activeCounts, onePublic, kinds, leftSources, rightSources, ()) next) count publicInitial publicOutput) :
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
