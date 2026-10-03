import TfheNand
import Claim

/-!
The generated TFHE correctness claim: every run of keygen, two encryptions, the bootstrapped
NAND gate, and decryption decodes `1 - m1 m2`, with the decryption phase within `Δ` of the
encoded bit.
-/

namespace MxxFheTfhe

open Mxx.Primitives MxxRuntime GeneratedClaim

theorem correctness : CorrectnessClaim := by
  intro hashModel external execution hruns
  obtain ⟨hvalid, hk, hl, hr, hn, hd, hi⟩ := hruns
  obtain ⟨Z, hZ, hS, hgsw, hks⟩ := keygen_spec hk
  obtain ⟨_, ⟨hm1a, hm1b⟩, _, ⟨hm2a, hm2b⟩, _⟩ := hvalid
  have hm1 : external.input_1 = 0 ∨ external.input_1 = 1 := by omega
  have hm2 : external.input_3 = 0 ∨ external.input_3 = 1 := by omega
  obtain ⟨η, hη, hphase⟩ :=
    nand_spec hS hZ hgsw hks hm1 hm2 (encrypt_left_spec hl) (encrypt_right_spec hr) hn
  obtain ⟨hdec, hbit⟩ := decrypt_spec hd
  have hideal := ideal_spec hi
  -- The decryption phase is the NAND output phase.
  have hv : ((execution.«stage_4».2.1 : Int) : ZMod q) =
      (((1 - external.input_1 * external.input_3) * 2 - 1) * Δ + η : Int) := by
    rw [hdec, ZMod.intCast_mod]
    exact hphase
  have hvq := (ZMod.intCast_eq_intCast_iff' _ _ _).mp hv
  have hrange : 0 ≤ execution.«stage_4».2.1 ∧ execution.«stage_4».2.1 < 4294967296 := by
    rw [hdec]
    simp only [q, Nat.cast_ofNat]
    omega
  simp only [q, Δ, Nat.cast_ofNat] at hvq hbit
  rw [abs_le] at hη
  rw [Int.emod_eq_of_lt hrange.1 hrange.2] at hvq
  constructor
  · intro index
    unfold observedResidual TfheSemantics.messageCenter TfheSemantics.decoderRadius
    have hres : ((execution.«stage_4».2.1 : Int) : ZMod 4294967296) -
        (((if execution.«ideal» = 1 then 536870912 else (4294967296 : Nat) - 536870912 : Int) :
          Int) : ZMod 4294967296) = (η : ZMod 4294967296) := by
      have hq0 : ((4294967296 : Int) : ZMod 4294967296) = 0 := by
        exact_mod_cast ZMod.natCast_self 4294967296
      have hq0' : (4294967296 : ZMod 4294967296) = 0 := by simpa using hq0
      rw [show (execution.«stage_4».2.1 : ZMod 4294967296) = ((execution.«stage_4».2.1 : Int) :
        ZMod q) from rfl, hv, hideal]
      simp only [Δ]
      rcases hm1 with h1 | h1 <;> rcases hm2 with h2 | h2 <;>
        · simp only [h1, h2]
          norm_num
          try (ring_nf; simp [hq0'])
    rw [hres, centeredLift_intCast (by decide) (by omega)]
    omega
  · rw [hbit, hideal]
    rcases hm1 with h1 | h1 <;> rcases hm2 with h2 | h2 <;> simp only [h1, h2] at hvq ⊢ <;>
      split_ifs <;> omega

end MxxFheTfhe
