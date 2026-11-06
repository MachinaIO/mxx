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

namespace PeProof

open Mxx.Primitives MxxRuntime MxxExponentLut

set_option maxHeartbeats 4000000 in
/-- One iteration of the compact block contraction. -/
theorem compactStep {model : HashModel} {tape : SampleTape} {path : List Nat}
    {params : Stage_decode.Params} {x : ExactMatrix Q N 1 44} {v : Fin 128 → ExactMatrix Q N 44 1}
    {i : Nat} (hi : i < 128) {s t : ExactMatrix Q N 1 1}
    (h : Stage_decode.sequential_generatedRoot_538 Backend.backend model tape path params i (s, (x, (v, ()))) t) :
    t = s + substituteMatrix (slotAut i) (slotAut_odd ⟨i, hi⟩) x * v ⟨i, hi⟩ := by
  obtain ⟨w2, w5, -, -, h2, -, -, h5, hout⟩ := h
  have e2 := routing_slot hi h2
  obtain ⟨_, e5⟩ := familyGet_eq h5
  subst hout e2 e5
  rfl

set_option maxHeartbeats 8000000 in
/-- The compact block contraction. -/
theorem compactLoop {model : HashModel} {tape : SampleTape} {path : List Nat}
    {params : Stage_decode.Params} {x : ExactMatrix Q N 1 44} {v : Fin 128 → ExactMatrix Q N 44 1}
    {final : ExactMatrix Q N 1 1}
    (h : MxxIR.IterRuns (fun (i : Nat) (current next : ExactMatrix Q N 1 1) ↦
      Stage_decode.sequential_generatedRoot_538 Backend.backend model tape (path ++ [538, i]) params i
        (current, (x, (v, ()))) next) (Int.toNat 128) 0 final) :
    final = ∑ i : Fin 128, substituteMatrix (slotAut i) (slotAut_odd i) x * v i := by
  have hcount : Int.toNat 128 = 128 := rfl
  rw [hcount] at h
  have hacc := IterRuns.accumulate (fun s : ExactMatrix Q N 1 1 ↦ s)
    (fun i ↦ if hi : i < 128 then substituteMatrix (slotAut i) (slotAut_odd ⟨i, hi⟩) x * v ⟨i, hi⟩ else 0)
    (fun _ _ ↦ True) trivial (fun i s t hi _ hb ↦ ⟨by simp only [dif_pos hi]; exact compactStep hi hb, trivial⟩) h
  obtain ⟨hs, -⟩ := hacc
  dsimp only at hs
  rw [hs, zero_add, ← Fin.sum_univ_eq_sum_range (fun i ↦ if hi : i < 128 then
    substituteMatrix (slotAut i) (slotAut_odd ⟨i, hi⟩) x * v ⟨i, hi⟩ else 0)]
  exact Finset.sum_congr rfl fun i _ ↦ by simp [i.isLt]

/-- The joined row `[0 | g]`. -/
noncomputable def zeroG : ExactMatrix Q N 1 88 := joinRow ![0, g0]

/-- The helper family value of pair `p = 37 i + k`: `σ_i(h_k - H_k^(2)) + [0 | g]`. -/
noncomputable def helperValue (model : HashModel) (key : ByteArray) (hf0 hf1 : Fin 37 → ExactMatrix Q N 1 44)
    (p : Fin 4736) : ExactMatrix Q N 1 88 :=
  substituteMatrix (slotAut (p / 37)) (slotAut_odd ⟨p / 37, by omega⟩)
      (joinRow ![hf0 ⟨p % 37, by omega⟩, hf1 ⟨p % 37, by omega⟩] -
        joinRow (helperKey model key (kHelp (p % 37)) 1)) + zeroG

set_option maxHeartbeats 8000000 in
/-- One iteration of the helper block contraction. -/
theorem helperContractStep {model : HashModel} {tape : SampleTape} {path : List Nat}
    {params : Stage_decode.Params} {key : ByteArray} {hf0 hf1 : Fin 37 → ExactMatrix Q N 1 44}
    {v : Fin 4736 → ExactMatrix Q N 88 1} {p : Nat} (hp : p < 4736) {s t : ExactMatrix Q N 1 1}
    (h : Stage_decode.sequential_generatedRoot_547 Backend.backend model tape path params p
      (s, (hf0, (hf1, (key, (zeroG, (v, ())))))) t) :
    t = s + helperValue model key hf0 hf1 ⟨p, hp⟩ * v ⟨p, hp⟩ := by
  obtain ⟨w3, w5, w6, w8, w9, w10, w12, w17, -, -, -, h3, -, -, h5, h6, -, -, -, h8, -, -, -, h9, h10, -, -,
    -, h12, -, -, h17, hout⟩ := h
  have hk : Int.fmod (Int.ofNat p) 37 = Int.ofNat (p % 37) := by
    rw [Int.fmod_eq_emod_of_nonneg _ (by norm_num)]; simp only [Int.ofNat_eq_natCast]; omega
  have hi : Int.fdiv (Int.ofNat p) 37 = Int.ofNat (p / 37) := by
    rw [Int.fdiv_eq_ediv_of_nonneg _ (by norm_num)]; simp only [Int.ofNat_eq_natCast]; omega
  rw [hk] at h3 h5 h8 h9
  rw [hi] at h12
  obtain ⟨_, e3⟩ := familyGet_eq h3
  obtain ⟨_, e5⟩ := familyGet_eq h5
  have hc := kHelp_cast (p % 37) (by omega)
  rw [← hc] at h8 h9
  have e8 : w8 = helperKey model key (kHelp (p % 37)) 1 0 := helperKey_read (c := 2) (by norm_num) h8
  have e9 : w9 = helperKey model key (kHelp (p % 37)) 1 1 := helperKey_read (c := 3) (by norm_num) h9
  rw [e3, e5] at h6
  rw [e8, e9] at h10
  have e6 := concat_joinRow (r := ![hf0 ⟨p % 37, by omega⟩, hf1 ⟨p % 37, by omega⟩]) h6
  have e10 := concat_joinRow (r := helperKey model key (kHelp (p % 37)) 1) h10
  have e12 := routing_slot (by omega) h12
  obtain ⟨_, e17⟩ := familyGet_eq h17
  subst hout e6 e10 e12 e17
  rfl

set_option maxRecDepth 100000 in
set_option maxHeartbeats 16000000 in
/-- The helper block contraction. -/
theorem helperContract {model : HashModel} {tape : SampleTape} {path : List Nat}
    {params : Stage_decode.Params} {key : ByteArray} {hf0 hf1 : Fin 37 → ExactMatrix Q N 1 44}
    {v : Fin 4736 → ExactMatrix Q N 88 1} {final : ExactMatrix Q N 1 1}
    (h : MxxIR.IterRuns (fun (i : Nat) (current next : ExactMatrix Q N 1 1) ↦
      Stage_decode.sequential_generatedRoot_547 Backend.backend model tape (path ++ [547, i]) params i
        (current, (hf0, (hf1, (key, (zeroG, (v, ())))))) next) (Int.toNat 4736) 0 final) :
    final = ∑ p : Fin 4736, helperValue model key hf0 hf1 p * v p := by
  have hcount : Int.toNat 4736 = 4736 := rfl
  rw [hcount] at h
  have hacc := IterRuns.accumulate (fun s : ExactMatrix Q N 1 1 ↦ s)
    (fun p ↦ if hp : p < 4736 then helperValue model key hf0 hf1 ⟨p, hp⟩ * v ⟨p, hp⟩ else 0)
    (fun _ _ ↦ True) trivial
    (fun p s t hp _ hb ↦ ⟨by simp only [dif_pos hp]; exact helperContractStep hp hb, trivial⟩) h
  obtain ⟨hs, -⟩ := hacc
  dsimp only at hs
  rw [hs, zero_add, ← Fin.sum_univ_eq_sum_range (fun p ↦ if hp : p < 4736 then
    helperValue model key hf0 hf1 ⟨p, hp⟩ * v ⟨p, hp⟩ else 0)]
  exact Finset.sum_congr rfl fun p _ ↦ by simp [p.isLt]

end PeProof
