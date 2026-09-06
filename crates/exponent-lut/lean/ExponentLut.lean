import ExponentReadoutSetup
import ExponentMonomial

namespace MxxExponentLut
open Mxx.Primitives
open scoped BigOperators

private theorem root_power_signedMonomial {q n : Nat} (hn : 0 < n) (exponent : Nat) :
    SignedMonomial (AdjoinRoot.root (negacyclicModulus n (ZMod q)) ^ exponent) := by
  refine ⟨⟨exponent % (2 * n), Nat.mod_lt _ (by omega)⟩, ?_⟩
  have hperiod : AdjoinRoot.root (negacyclicModulus n (ZMod q)) ^ (2 * n) = 1 := by
    rw [Nat.mul_comm, pow_mul, Negacyclic.root_pow_n]
    ring
  conv_lhs => rw [← Nat.div_add_mod exponent (2 * n)]
  rw [pow_add, pow_mul, hperiod]
  simp

/-- The rotated native message remains a signed monomial by the actual
quotient-ring substitution, so its action on sampled errors has unit L1. -/
theorem signedMonomial_automorphism {q n : Nat} (hn : 0 < n) (index : Nat) (hodd : Odd index)
    {message : ExactPoly q n} (hmessage : SignedMonomial message) :
    SignedMonomial (readoutAutomorphismHom index hodd message) := by
  obtain ⟨exponent, he⟩ := hmessage
  rw [he, map_pow]
  have hroot : readoutAutomorphismHom index hodd
      (AdjoinRoot.root (negacyclicModulus n (ZMod q))) =
      AdjoinRoot.root (negacyclicModulus n (ZMod q)) ^ index := by
    simp [readoutAutomorphismHom]
  rw [hroot, ← pow_mul]
  exact root_power_signedMonomial hn _

/-- LUT cache equations differ from normalized readout by the actual Fourier
coefficient in the fixed RHS. The sampled error still appears negatively and
is multiplied only by the rotated native message, not the Fourier coefficient. -/
theorem lut_setup_of_cache_equations {q n secret columns : Nat}
    (hq : 1 < q) (hn : 0 < n) (index : Nat) (hodd : Odd index)
    (mask payload : ExactMatrix q n 1 secret)
    (publicKey maskKey gadget : ExactMatrix q n secret columns)
    (action helperAction : ErrorMatrix n columns columns)
    (switchError : ErrorMatrix n 1 columns)
    (message coefficient : ExactPoly q n) (messageLift : ErrorPoly n)
    (hgadget : MxxRuntime.ringAutomorphism index gadget = gadget)
    (hswitch : MxxRuntime.ringAutomorphism index payload * gadget *
        reduceMatrix q n columns columns action =
      coefficient • (payload * gadget) + reduceMatrix q n 1 columns switchError)
    (halign : gadget * reduceMatrix q n columns columns helperAction =
      MxxRuntime.ringAutomorphism index publicKey * reduceMatrix q n columns columns action)
    (hmessage : readoutAutomorphismHom index hodd message = reducePoly q n messageLift) :
    MxxRuntime.ringAutomorphism index (mask * publicKey - message • (payload * gadget)) *
        reduceMatrix q n columns columns action +
      (mask * maskKey - MxxRuntime.ringAutomorphism index mask * gadget) *
        reduceMatrix q n columns columns helperAction =
      (mask * (maskKey * reduceMatrix q n columns columns helperAction) -
        (readoutAutomorphismHom index hodd message * coefficient) • (payload * gadget)) +
      reduceMatrix q n 1 columns ((-messageLift) • switchError) := by
  rw [ringAutomorphism_sub hq hn index hodd,
    ringAutomorphism_mul hq hn index hodd,
    ringAutomorphism_polynomial_action hq hn index hodd,
    ringAutomorphism_mul hq hn index hodd, hgadget]
  rw [Matrix.sub_mul, Matrix.sub_mul, Matrix.smul_mul, hswitch, smul_add, smul_smul]
  rw [Matrix.mul_assoc mask maskKey, Matrix.mul_assoc _ gadget, halign,
    ← Matrix.mul_assoc]
  rw [reduce_polynomial_action, map_neg, neg_smul, ← hmessage]
  simp only [Matrix.mul_assoc]
  abel

/-- One actual LUT branch (`mask * branch + raw * switch`) from
`encoding.rs::single_input_lut_table_parts`. All helper premises are primitive
encoding/cache equations. No already-bounded branch output is assumed. -/
theorem lut_encoding_branch_transfer {q log secret columns bound cutoff actionGain helperGain : Nat}
    (hq : 1 < q) (index : Nat) (hodd : Odd index)
    {input raw maskCipher : ExactMatrix q (2 ^ log) 1 columns}
    (mask payload : ExactMatrix q (2 ^ log) 1 secret)
    (publicKey maskKey gadget : ExactMatrix q (2 ^ log) secret columns)
    (action helperAction : ErrorMatrix (2 ^ log) columns columns)
    (switchError maskError : ErrorMatrix (2 ^ log) 1 columns)
    (message coefficient : ExactPoly q (2 ^ log))
    (hinput : Approx input (mask * publicKey - message • (payload * gadget)) bound)
    (hrun : MxxRuntime.ringAutomorphismRuns (index : Int) input raw)
    (hmessage : SignedMonomial message)
    (hgadget : MxxRuntime.ringAutomorphism index gadget = gadget)
    (hswitch : MxxRuntime.ringAutomorphism index payload * gadget *
        reduceMatrix q (2 ^ log) columns columns action =
      coefficient • (payload * gadget) + reduceMatrix q (2 ^ log) 1 columns switchError)
    (halign : gadget * reduceMatrix q (2 ^ log) columns columns helperAction =
      MxxRuntime.ringAutomorphism index publicKey * reduceMatrix q (2 ^ log) columns columns action)
    (hmask : maskCipher = mask * maskKey - MxxRuntime.ringAutomorphism index mask * gadget +
      reduceMatrix q (2 ^ log) 1 columns maskError)
    (hswitchBound : CoeffBound switchError cutoff) (hmaskBound : CoeffBound maskError cutoff)
    (haction : ActionGain action actionGain) (hhelper : ActionGain helperAction helperGain) :
    Approx (maskCipher * reduceMatrix q (2 ^ log) columns columns helperAction +
      raw * reduceMatrix q (2 ^ log) columns columns action)
      (mask * (maskKey * reduceMatrix q (2 ^ log) columns columns helperAction) -
        (readoutAutomorphismHom index hodd message * coefficient) • (payload * gadget))
      (actionGain * bound + cutoff * (1 + helperGain)) := by
  obtain ⟨lift, hlift, hl1⟩ := signedMonomial_integer_lift (by positivity)
    (signedMonomial_automorphism (by positivity) index hodd hmessage)
  have hsetup := lut_setup_of_cache_equations hq (by positivity) index hodd mask payload
    publicKey maskKey gadget action helperAction switchError message coefficient lift
    hgadget hswitch halign hlift.symm
  have hneg : polynomialL1 (-lift) ≤ 1 := by simpa using hl1
  have h := readout_branch_transfer (targetPlainGain := 1) hq hinput hrun hmask
    (by simpa only [Int.toNat_natCast] using hsetup) hswitchBound hmaskBound hneg haction hhelper
  convert h using 1
  ring

/-- Worst-case LUT aggregation is an ordinary sum of branch bounds, exactly
as `aggregate_lut_branches`; no independence premise is used. -/
def lutNoiseBound {count : Nat} (inputBound cutoff : Nat)
    (actionGain helperGain : Fin count → Nat) : Nat :=
  ∑ i, (actionGain i * inputBound + cutoff * (1 + helperGain i))

theorem lut_encoding_transfer {q n log secret columns count bound cutoff : Nat}
    (hq : 1 < q) (hdimension : n = 2 ^ log)
    (index : Fin count → Nat) (hodd : ∀ i, Odd (index i))
    {input : ExactMatrix q n 1 columns}
    (raw maskCipher : Fin count → ExactMatrix q n 1 columns)
    (mask payload : ExactMatrix q n 1 secret)
    (publicKey gadget : ExactMatrix q n secret columns)
    (maskKey : Fin count → ExactMatrix q n secret columns)
    (action helperAction : Fin count → ErrorMatrix n columns columns)
    (switchError maskError : Fin count → ErrorMatrix n 1 columns)
    (message : ExactPoly q n) (coefficient : Fin count → ExactPoly q n)
    (actionGain helperGain : Fin count → Nat)
    (hinput : Approx input (mask * publicKey - message • (payload * gadget)) bound)
    (hrun : ∀ i, MxxRuntime.ringAutomorphismRuns (index i : Int) input (raw i))
    (hmessage : SignedMonomial message)
    (hgadget : ∀ i, MxxRuntime.ringAutomorphism (index i) gadget = gadget)
    (hswitch : ∀ i, MxxRuntime.ringAutomorphism (index i) payload * gadget *
        reduceMatrix q n columns columns (action i) =
      coefficient i • (payload * gadget) + reduceMatrix q n 1 columns (switchError i))
    (halign : ∀ i, gadget * reduceMatrix q n columns columns (helperAction i) =
      MxxRuntime.ringAutomorphism (index i) publicKey * reduceMatrix q n columns columns (action i))
    (hmask : ∀ i, maskCipher i = mask * maskKey i -
      MxxRuntime.ringAutomorphism (index i) mask * gadget + reduceMatrix q n 1 columns (maskError i))
    (hswitchBound : ∀ i, CoeffBound (switchError i) cutoff)
    (hmaskBound : ∀ i, CoeffBound (maskError i) cutoff)
    (haction : ∀ i, ActionGain (action i) (actionGain i))
    (hhelper : ∀ i, ActionGain (helperAction i) (helperGain i)) :
    Approx (∑ i, (maskCipher i * reduceMatrix q n columns columns (helperAction i) +
      raw i * reduceMatrix q n columns columns (action i)))
      (mask * (∑ i, maskKey i * reduceMatrix q n columns columns (helperAction i)) -
        (∑ i, readoutAutomorphismHom (index i) (hodd i) message * coefficient i) • (payload * gadget))
      (lutNoiseBound bound cutoff actionGain helperGain) := by
  subst n
  have hb (i : Fin count) := lut_encoding_branch_transfer hq (index i) (hodd i)
    mask payload publicKey (maskKey i) gadget (action i) (helperAction i)
    (switchError i) (maskError i) message (coefficient i) hinput (hrun i) hmessage
    (hgadget i) (hswitch i) (halign i) (hmask i) (hswitchBound i) (hmaskBound i)
    (haction i) (hhelper i)
  have hs := approximate_sum Finset.univ _ _ _ (fun i _ => hb i)
  simpa only [lutNoiseBound, Finset.sum_sub_distrib, Matrix.mul_sum, Finset.sum_smul] using hs

end MxxExponentLut
