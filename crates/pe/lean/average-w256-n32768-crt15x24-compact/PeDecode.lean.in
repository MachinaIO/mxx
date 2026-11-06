import PeKeygenRoot
import Stage_decode
import Stage_decode_Facts

/-!
# Decryption

The phase is `(c_m - d_2)` minus the contraction of the key: the anchor blocks
`Σ_i σ_i((b - B^(2)) k̃_i)`, the compact blocks `Σ_i σ_i(b'' - B''^(2)) v''_i`, the helper blocks
`Σ_(i,k) (σ_i(h_k - H_k^(2)) + [0 | g]) v_(i,k)`, and the output block `(c_F - A_F^(2)) v_F`. Its
coefficients are rounded to bits.
-/

set_option maxRecDepth 200000

namespace PeProof

open Mxx.Primitives MxxRuntime MxxExponentLut

theorem anchorStep {model : HashModel} {tape : SampleTape} {path : List Nat}
    {params : Stage_decode.Params} {x : ExactMatrix Q N 1 295}
    {anchors : Fin 256 → ExactMatrix Q N 295 1} {index : Nat} (bound : index < 256)
    {current next : ExactMatrix Q N 1 1}
    (h : Stage_decode.sequential_generatedRoot_11 Backend.backend model tape path params index
      (current, x, anchors, ()) next) :
    next = current + substituteMatrix (slotAut index) (slotAut_odd ⟨index, bound⟩)
      (x * anchors ⟨index, bound⟩) := by
  obtain ⟨sample, routed, -, -, get, -, -, route, output⟩ := h
  obtain ⟨_, entry⟩ := familyGet_eq get
  have routed_eq := routing_slot bound route
  subst output entry routed_eq
  rfl

set_option maxHeartbeats 4000000 in
/-- One iteration of the compact block contraction. -/
theorem compactStep {model : HashModel} {tape : SampleTape} {path : List Nat}
    {params : Stage_decode.Params} {x : ExactMatrix Q N 1 30} {v : Fin 256 → ExactMatrix Q N 30 1}
    {i : Nat} (hi : i < 256) {s t : ExactMatrix Q N 1 1}
    (h : Stage_decode.sequential_generatedRoot_28 Backend.backend model tape path params i (s, (x, (v, ()))) t) :
    t = s + substituteMatrix (slotAut i) (slotAut_odd ⟨i, hi⟩) x * v ⟨i, hi⟩ := by
  obtain ⟨w2, w5, -, -, h2, -, -, h5, hout⟩ := h
  have e2 := routing_slot hi h2
  obtain ⟨_, e5⟩ := familyGet_eq h5
  subst hout e2 e5
  rfl

set_option maxHeartbeats 8000000 in
/-- The compact block contraction. -/
theorem compactLoop {model : HashModel} {tape : SampleTape} {path : List Nat}
    {params : Stage_decode.Params} {x : ExactMatrix Q N 1 30} {v : Fin 256 → ExactMatrix Q N 30 1}
    {final : ExactMatrix Q N 1 1}
    (h : MxxIR.IterRuns (fun (i : Nat) (current next : ExactMatrix Q N 1 1) ↦
      Stage_decode.sequential_generatedRoot_28 Backend.backend model tape (path ++ [28, i]) params i
        (current, (x, (v, ()))) next) (Int.toNat 256) 0 final) :
    final = ∑ i : Fin 256, substituteMatrix (slotAut i) (slotAut_odd i) x * v i := by
  have hcount : Int.toNat 256 = 256 := rfl
  rw [hcount] at h
  have hacc := IterRuns.accumulate (fun s : ExactMatrix Q N 1 1 ↦ s)
    (fun i ↦ if hi : i < 256 then substituteMatrix (slotAut i) (slotAut_odd ⟨i, hi⟩) x * v ⟨i, hi⟩ else 0)
    (fun _ _ ↦ True) trivial (fun i s t hi _ hb ↦ ⟨by simp only [dif_pos hi]; exact compactStep hi hb, trivial⟩) h
  obtain ⟨hs, -⟩ := hacc
  dsimp only at hs
  rw [hs, zero_add, ← Fin.sum_univ_eq_sum_range (fun i ↦ if hi : i < 256 then
    substituteMatrix (slotAut i) (slotAut_odd ⟨i, hi⟩) x * v ⟨i, hi⟩ else 0)]
  exact Finset.sum_congr rfl fun i _ ↦ by simp [i.isLt]

/-- The joined row `[0 | g]`. -/
noncomputable def zeroG : ExactMatrix Q N 1 60 := joinRow ![0, g0]

/-- The helper family value of pair `p = 45 i + k`: `σ_i(h_k - H_k^(2)) + [0 | g]`. -/
noncomputable def helperValue (model : HashModel) (key : ByteArray) (hf0 hf1 : Fin 45 → ExactMatrix Q N 1 30)
    (p : Fin 11520) : ExactMatrix Q N 1 60 :=
  substituteMatrix (slotAut (p / 45)) (slotAut_odd ⟨p / 45, by omega⟩)
      (joinRow ![hf0 ⟨p % 45, by omega⟩, hf1 ⟨p % 45, by omega⟩] -
        joinRow (helperKey model key (kHelp (p % 45)) 1)) + zeroG

set_option maxHeartbeats 8000000 in
theorem helperContractStep {model : HashModel} {tape : SampleTape} {path : List Nat}
    {params : Stage_decode.Params} {key : ByteArray} {hf0 hf1 : Fin 45 → ExactMatrix Q N 1 30}
    {packed : ExactMatrix Q N 60 45} {row lane : Nat} (row_bound : row < 256)
    (lane_bound : lane < 45) {current next : ExactMatrix Q N 1 1}
    (h : Stage_decode.sequential_sequential_generatedRoot_37_8 Backend.backend model tape path
      params row lane (current, hf0, hf1, key, zeroG, packed, ()) next) :
    next = current + matrixMul (helperValue model key hf0 hf1 ⟨45 * row + lane, by omega⟩)
      ((fun entry _ ↦ packed entry ⟨lane, lane_bound⟩) : ExactMatrix Q N 60 1) := by
  obtain ⟨w3, w5, w6, w8, w9, w10, w12, w17, -, -, h3, -, -, h5, h6, -, -, h8, -, -, h9, h10, -, -, h12, -, -, -, -, -, -, h17, hout⟩ := h
  obtain ⟨_, e3⟩ := familyGet_eq h3
  obtain ⟨_, e5⟩ := familyGet_eq h5
  have cast := kHelp_cast lane lane_bound
  rw [← cast] at h8 h9
  have e8 : w8 = helperKey model key (kHelp lane) 1 0 :=
    helperKey_read (c := 2) (by norm_num) h8
  have e9 : w9 = helperKey model key (kHelp lane) 1 1 :=
    helperKey_read (c := 3) (by norm_num) h9
  rw [e3, e5] at h6
  rw [e8, e9] at h10
  have e6 := concat_joinRow (r := ![hf0 ⟨lane, lane_bound⟩, hf1 ⟨lane, lane_bound⟩]) h6
  have e10 := concat_joinRow (r := helperKey model key (kHelp lane) 1) h10
  have e12 := routing_slot row_bound h12
  obtain ⟨_, e17⟩ := sliceMatrix_column h17
  rw [hout]
  funext entry column
  rw [Subsingleton.elim entry 0, Subsingleton.elim column 0]
  change current 0 0 + w17 0 0 = _
  rw [e17]
  change current 0 0 + ((w12 + zeroG) * packed) 0 ⟨lane, lane_bound⟩ = _
  rw [e12]
  dsimp only [matrixSub] at h12
  simp only [matrixSub, e6, e10, helperValue]
  have div : (45 * row + lane) / 45 = row := by omega
  have mod : (45 * row + lane) % 45 = lane := by omega
  simp only [div, mod, matrixMul, Matrix.mul_apply, Matrix.add_apply]

theorem helperContractRow {model : HashModel} {tape : SampleTape} {path : List Nat}
    {params : Stage_decode.Params} {key : ByteArray} {hf0 hf1 : Fin 45 → ExactMatrix Q N 1 30}
    {packed : Fin 256 → ExactMatrix Q N 60 45} {row : Nat} (bound : row < 256)
    {current next : ExactMatrix Q N 1 1}
    (h : Stage_decode.sequential_generatedRoot_37 Backend.backend model tape path params row
      (current, hf0, hf1, key, zeroG, packed, ()) next) :
    next = current + ∑ lane : Fin 45,
      matrixMul (helperValue model key hf0 hf1 ⟨45 * row + lane, by omega⟩)
        ((fun entry _ ↦ packed ⟨row, bound⟩ entry lane) : ExactMatrix Q N 60 1) := by
  obtain ⟨witness, body⟩ := h
  let ctx : Stage_decode.sequential_generatedRoot_37.Ctx Backend.backend model tape path params row :=
    ⟨(current, hf0, hf1, key, zeroG, packed, ()), witness⟩
  have facts := Stage_decode.sequential_generatedRoot_37.facts ctx body
  obtain ⟨_, get⟩ := familyGet_eq facts.c0.h7_0
  have loop := facts.c0.h8_0
  rw [get] at loop
  rw [facts.c0.hout]
  exact stream_sum _ (fun lane initial final lane_bound body ↦
    helperContractStep bound lane_bound body) loop

theorem packed_sum {M : Type} [AddCommMonoid M] (term : Fin 11520 → M) :
    ∑ pair, term pair = ∑ row : Fin 256, ∑ lane : Fin 45,
      term ⟨45 * row + lane, by omega⟩ := by
  calc ∑ pair, term pair = ∑ pair : Fin 256 × Fin 45, term (finProdFinEquiv pair) :=
        (Equiv.sum_comp (finProdFinEquiv (m := 256) (n := 45)) term).symm
    _ = ∑ row : Fin 256, ∑ lane : Fin 45, term (finProdFinEquiv (row, lane)) :=
      Fintype.sum_prod_type _
    _ = _ := by
      refine Finset.sum_congr rfl fun row _ ↦ Finset.sum_congr rfl fun lane _ ↦ ?_
      congr 1
      ext
      simp only [finProdFinEquiv_apply_val]
      ring

end PeProof
