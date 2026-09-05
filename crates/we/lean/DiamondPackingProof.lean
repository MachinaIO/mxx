import DiamondProofParameters
import Stage_decrypt

open Mxx.Primitives MxxRuntime

open DiamondProofParameters

namespace DiamondGeneratedProof

theorem packing_step (acc bit : Int) (width : Nat)
    (hacc : 0 ≤ acc ∧ acc < 2 ^ width) (hbit : 0 ≤ bit ∧ bit ≤ 1) :
    (0 ≤ acc + bit * 2 ^ width ∧ acc + bit * 2 ^ width < 2 ^ (width + 1)) ∧
      (acc + bit * 2 ^ width) / 2 ^ width % 2 = bit ∧
      ∀ index, index < width →
        (acc + bit * 2 ^ width) / 2 ^ index % 2 = acc / 2 ^ index % 2 := by
  have hpos : (0 : Int) < 2 ^ width := pow_pos (by decide) _
  refine ⟨?_, ?_, ?_⟩
  · rw [pow_succ]
    constructor <;> nlinarith
  · rw [Int.add_mul_ediv_right _ _ (ne_of_gt hpos),
      Int.ediv_eq_zero_of_lt hacc.1 hacc.2, zero_add]
    exact Int.emod_eq_of_lt hbit.1 (by omega)
  · intro index hindex
    have hexponent : width = index + (width - index - 1) + 1 := by omega
    have hpower : (2 : Int) ^ width =
        (2 ^ (width - index - 1) * 2) * 2 ^ index := by
      conv_lhs => rw [hexponent, pow_add, pow_add]
      ring
    have hterm : bit * (2 : Int) ^ width =
        (bit * 2 ^ (width - index - 1) * 2) * 2 ^ index := by rw [hpower]; ring
    rw [hterm, Int.add_mul_ediv_right _ _ (ne_of_gt (pow_pos (by decide) _))]
    simp

theorem generated_packed_bits
    (backend : BackendContext) (params : Stage_decrypt.Params) (layer : Nat)
    (raw : Fin witnessSlots → Int) (output : Int)
    (hraw : ∀ position, 0 ≤ raw position ∧ raw position ≤ 1)
    (hrun : Stage_decrypt.parallel_generatedRoot_6 backend params layer raw output) :
    (0 ≤ output ∧ output < (2 : Int) ^ params.diamond_batch_bits.toNat) ∧
      ∀ bit, bit < params.diamond_batch_bits.toNat →
        ∃ position : Fin witnessSlots,
          (position.val : Int) = (layer : Int) * params.diamond_batch_bits + (bit : Int) ∧
          output / (2 ^ bit) % 2 = raw position := by
  rcases hrun with ⟨finalAcc, finalPower, _, hscan, hout⟩
  let Invariant := fun (width : Nat) (state : Int × Int × Unit) ↦
    state.2.1 = 2 ^ width ∧ (0 ≤ state.1 ∧ state.1 < 2 ^ width) ∧
      ∀ bit, bit < width → ∃ position : Fin witnessSlots,
        (position.val : Int) = (layer : Int) * params.diamond_batch_bits + (bit : Int) ∧
        state.1 / 2 ^ bit % 2 = raw position
  have hstart : Invariant 0 (0, 1, ()) := by
    refine ⟨rfl, by norm_num, ?_⟩
    intro bit hbit
    omega
  have hfinal := MxxIR.IterRuns.invariant (Invariant := Invariant) hstart
    (by
      intro width current next ih hstep
      rcases ih with ⟨hpower, hacc, hbits⟩
      rcases hstep with ⟨value, _, _, ⟨position, hposition, hvalue⟩, hnext⟩
      change (position.val : Int) = (layer : Int) * params.diamond_batch_bits +
        (width : Int) at hposition
      have hstepBounds := packing_step current.1 (raw position) width hacc (hraw position)
      rw [hnext]
      change current.2.1 * 2 = 2 ^ (width + 1) ∧
        (0 ≤ current.1 + value * current.2.1 ∧
          current.1 + value * current.2.1 < 2 ^ (width + 1)) ∧ _
      rw [hpower, hvalue]
      refine ⟨by rw [pow_succ], hstepBounds.1, ?_⟩
      intro bit hbit
      by_cases htop : bit = width
      · subst bit
        exact ⟨position, hposition, hstepBounds.2.1⟩
      · have hlt : bit < width := by omega
        obtain ⟨oldPosition, hindex, hbitValue⟩ := hbits bit hlt
        exact ⟨oldPosition, hindex, (hstepBounds.2.2 bit hlt).trans hbitValue⟩)
    hscan
  rw [hout]
  exact hfinal.2

def rawWitnessBits (raw : Fin circuitWidth → Int) (index : Nat) : Bool :=
  if h : index < circuitWidth then decide (raw ⟨index, h⟩ = 1) else false

theorem rawWitnessBits_at (raw : Fin circuitWidth → Int) (position : Fin circuitWidth) :
    rawWitnessBits raw position.val = decide (raw position = 1) := by
  unfold rawWitnessBits
  rw [dif_pos position.isLt]

theorem generated_witness_prefix
    (backend : BackendContext) (params : Stage_decrypt.Params) (i : Fin witnessSlots)
    (raw : Fin circuitWidth → Int) (index value : Int)
    (hindex : Stage_decrypt.parallel_generatedRoot_3 backend params i.val () index)
    (hget : Stage_decrypt.parallel_generatedRoot_5 backend params i.val
      (index, raw, ()) value) :
    ∃ position : Fin circuitWidth, position.val = i.val ∧ value = raw position := by
  rcases hget with ⟨selected, _, _, ⟨position, hposition, hvalue⟩, hout⟩
  refine ⟨position, ?_, hout.trans hvalue⟩
  change index = Int.ofNat i.val + 0 at hindex
  have h := hposition.trans hindex
  change (position.val : Int) = (i.val : Int) + 0 at h
  omega

theorem generated_packed_raw_witness
    (backend : BackendContext) (params : Stage_decrypt.Params)
    (raw : Fin circuitWidth → Int) (prefixIndices rawPrefix : Fin witnessSlots → Int)
    (packed : Fin inputCount → Int)
    (hraw : ∀ position, 0 ≤ raw position ∧ raw position ≤ 1)
    (hindices : ∀ i : Fin witnessSlots, Stage_decrypt.parallel_generatedRoot_3 backend params i.val ()
      (prefixIndices i))
    (hprefix : ∀ i : Fin witnessSlots, Stage_decrypt.parallel_generatedRoot_5 backend params i.val
      (prefixIndices i, raw, ()) (rawPrefix i))
    (hpacking : ∀ i : Fin inputCount, Stage_decrypt.parallel_generatedRoot_6 backend params i.val
      rawPrefix (packed i)) :
    (∀ i, 0 ≤ packed i ∧ packed i < (2 : Int) ^ params.diamond_batch_bits.toNat) ∧
      ∀ i : Fin inputCount, ∀ bit, bit < params.diamond_batch_bits.toNat →
        rawWitnessBits raw (i.val * params.diamond_batch_bits.toNat + bit) =
          decide ((packed i / 2 ^ bit) % 2 = 1) := by
  have hprefixRange : ∀ i, 0 ≤ rawPrefix i ∧ rawPrefix i ≤ 1 := by
    intro i
    obtain ⟨position, _, hvalue⟩ := generated_witness_prefix backend params i raw _ _
      (hindices i) (hprefix i)
    simpa only [hvalue] using hraw position
  have hpacked := fun i ↦ generated_packed_bits backend params i.val rawPrefix (packed i)
    hprefixRange (hpacking i)
  refine ⟨fun i ↦ (hpacked i).1, ?_⟩
  intro i bit hbit
  obtain ⟨prefixPosition, hprefixPosition, hbitValue⟩ := (hpacked i).2 bit hbit
  obtain ⟨position, hposition, hvalue⟩ := generated_witness_prefix backend params prefixPosition
    raw _ _ (hindices prefixPosition) (hprefix prefixPosition)
  have hbatch : 0 ≤ params.diamond_batch_bits := by
    rcases hpacking i with ⟨_, _, hnonneg, _, _⟩
    exact hnonneg
  have haddress : position.val = i.val * params.diamond_batch_bits.toNat + bit := by
    rw [← hposition] at hprefixPosition
    rw [← Int.toNat_of_nonneg hbatch] at hprefixPosition
    exact_mod_cast hprefixPosition
  rw [← haddress, rawWitnessBits_at, hbitValue, hvalue]

#print axioms generated_packed_raw_witness
#print axioms packing_step
#print axioms generated_packed_bits

end DiamondGeneratedProof
