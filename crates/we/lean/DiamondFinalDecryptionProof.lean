import Stage_decrypt
import DiamondProofParameters
import Decoder

open Mxx.Primitives MxxRuntime

namespace DiamondGeneratedProof

open DiamondProofParameters

set_option maxRecDepth 8192
set_option maxHeartbeats 800000

/-- Extract the final cancellation expression and the exact coefficient decoder from the
actual generated root. All public matrices and the final decomposition are input projections. -/
theorem generated_final_root
    (backend : BackendContext) (params : Stage_decrypt.Params) {inputs outputs}
    (hrun : Stage_decrypt.generatedRoot backend params inputs outputs) :
    ∃ (state : ExactMatrix q n 1 inner) (circuit : ExactMatrix q n 1 ell)
      (coefficient : Int),
      outputs.2.2.2.2.1 = state * inputs.2.2.2.1 -
        (state * inputs.2.2.2.2.1 + (state * inputs.2.2.2.2.2.1 - circuit) *
          inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1) ∧
      extractCoefficient 0 outputs.2.2.2.2.1 coefficient ∧
      outputs.2.2.2.2.2.1 =
        decide (MxxIR.roundDiv (params.diamond_modulus - 2) 4 ≤ coefficient ∧
          coefficient ≤ 3 * MxxIR.roundDiv (params.diamond_modulus - 2) 4) ∧
      familyGetStatic outputs.2.2.2.2.2.2.1 0 state := by
  dsimp only [Stage_decrypt.generatedRoot] at hrun
  rcases hrun with ⟨w2, w3, w5, w6, w8, state, w19, w24, w26, w27, w28, w29,
    w31, w32, w33, w35, w36, w37, w38, w40, w42, w43, w44, w45, w46, w47,
    w48, w49, w50, w53, w54, w55, w56, w57, w58, w59, w60, w61, w62,
    w67a, w67b, w67c, w70, circuit, coefficient, h⟩
  have hextract : extractCoefficient 0
      (state * inputs.2.2.2.1 - (state * inputs.2.2.2.2.1 +
        (state * inputs.2.2.2.2.2.1 - circuit) *
          inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1)) coefficient := by
    tauto
  have hstate : familyGetStatic w8 0 state := by tauto
  repeat' obtain ⟨_, h⟩ := h
  refine ⟨state, circuit, coefficient, rfl, hextract, ?_, hstate⟩
  dsimp only
  split_ifs <;> simp_all <;> omega

/-- Local matching encoding premises expose exactly which secret and public-key
correlations must be instantiated by the injector and accepting-circuit induction. -/
theorem final_exact_cancellation {q n width : Nat}
    (decoder key center eDecoder eKey : ExactMatrix q n 1 1)
    (one circuit secret publicOne publicCircuit eOne eCircuit : ExactMatrix q n 1 width)
    (digits : ExactMatrix q n width 1)
    (hdecoder : decoder = key + secret * digits + center + eDecoder - eKey)
    (hone : one = publicOne + eOne)
    (hcircuit : circuit = publicCircuit + eCircuit)
    (haccept : publicOne - publicCircuit = secret) :
    decoder - (key + (one - circuit) * digits) =
      center + eDecoder - eKey - (eOne - eCircuit) * digits := by
  rw [hdecoder, hone, hcircuit]
  have heq : publicOne + eOne - (publicCircuit + eCircuit) =
      secret + (eOne - eCircuit) := by rw [← haccept]; abel
  rw [heq, Matrix.add_mul]
  abel

theorem final_coeffBound_sub {n rows columns leftBound rightBound : Nat}
    {left right : ErrorMatrix n rows columns}
    (hl : CoeffBound left leftBound) (hr : CoeffBound right rightBound) :
    CoeffBound (left - right) (leftBound + rightBound) := by
  have hn : CoeffBound (-right) rightBound := by
    intro i j k
    simpa using hr i j k
  simpa [sub_eq_add_neg] using coeffBound_add hl hn

/-- The integer residual retains the same right factor. No independence premise is used. -/
theorem final_integer_residual_bound {n width B0 BH D : Nat} (hn : 0 < n)
    (eDecoder eKey : ErrorMatrix n 1 1) (eOne eCircuit : ErrorMatrix n 1 width)
    (digits : ErrorMatrix n width 1)
    (hd : CoeffBound eDecoder B0) (hk : CoeffBound eKey B0)
    (ho : CoeffBound eOne B0) (hc : CoeffBound eCircuit BH)
    (hD : CoeffBound digits D) :
    CoeffBound (eDecoder - eKey - (eOne - eCircuit) * digits)
      (2 * B0 + (width * n * D) * (B0 + BH)) := by
  have h := final_coeffBound_sub (final_coeffBound_sub hd hk)
    (coeffBound_mul hn (final_coeffBound_sub ho hc) hD)
  convert h using 1; ring

theorem final_reduce_sub {q n rows columns : Nat} (left right : ErrorMatrix n rows columns) :
    reduceMatrix q n rows columns (left - right) =
      reduceMatrix q n rows columns left - reduceMatrix q n rows columns right := by
  funext i j
  exact (reducePoly q n).map_sub _ _

/-- Local encoding equations and the matching public relation yield a bounded integer
residual with the actual shared decomposition. Instantiating these four encoding equations
and the accepting public relation is the remaining injector/circuit proof obligation. -/
theorem final_encoding_approx {q n width B0 BH D : Nat} (hn : 0 < n)
    (decoder key center publicDecoder publicKey : ExactMatrix q n 1 1)
    (one circuit publicOne publicCircuit : ExactMatrix q n 1 width)
    (eDecoder eKey : ErrorMatrix n 1 1) (eOne eCircuit : ErrorMatrix n 1 width)
    (digits : ErrorMatrix n width 1)
    (hdecoder : decoder = publicDecoder + center + reduceMatrix q n 1 1 eDecoder)
    (hkey : key = publicKey + reduceMatrix q n 1 1 eKey)
    (hone : one = publicOne + reduceMatrix q n 1 width eOne)
    (hcircuit : circuit = publicCircuit + reduceMatrix q n 1 width eCircuit)
    (hpublic : publicDecoder = publicKey +
      (publicOne - publicCircuit) * reduceMatrix q n width 1 digits)
    (hd : CoeffBound eDecoder B0) (hk : CoeffBound eKey B0)
    (ho : CoeffBound eOne B0) (hc : CoeffBound eCircuit BH)
    (hD : CoeffBound digits D) :
    Approx (decoder - (key + (one - circuit) * reduceMatrix q n width 1 digits)) center
      (2 * B0 + (width * n * D) * (B0 + BH)) := by
  refine ⟨eDecoder - eKey - (eOne - eCircuit) * digits, ?_,
    final_integer_residual_bound hn eDecoder eKey eOne eCircuit digits hd hk ho hc hD⟩
  rw [final_reduce_sub, final_reduce_sub, reduceMatrix_mul, final_reduce_sub]
  have hdec : decoder = key + (publicOne - publicCircuit) *
      reduceMatrix q n width 1 digits + center + reduceMatrix q n 1 1 eDecoder -
        reduceMatrix q n 1 1 eKey := by rw [hdecoder, hkey, hpublic]; abel
  rw [final_exact_cancellation decoder key center _ _ one circuit
    (publicOne - publicCircuit) publicOne publicCircuit _ _ _ hdec hone hcircuit rfl]
  abel

/-- Centering cannot enlarge a small integer representative, including odd moduli. -/
theorem final_centered_small {q B : Nat} (hq : 4 ≤ q)
    (hB : B < MxxWe.decoderRadius q) (e : Int) (he : e.natAbs ≤ B) :
    centeredLift q (e : ZMod q) = e := by
  letI : NeZero q := ⟨by omega⟩
  have hb := (MxxWe.decoderRadius_conditions hB).1
  have heb := MxxWe.natAbs_bounds he
  have hv := ZMod.val_intCast (n := q) e
  have hq4 : MxxWe.quarter q = q / 4 := rfl
  by_cases hp : 0 ≤ e
  · have heq : e % (q : Int) = e := Int.emod_eq_of_lt hp (by omega)
    rw [heq] at hv
    unfold centeredLift
    split <;> omega
  · have hm : e % (q : Int) = e + q := by
      have hn : 0 ≤ e + (q : Int) := by omega
      have hl : e + (q : Int) < q := by omega
      have ht := Int.emod_eq_of_lt hn hl
      simpa using ht
    rw [hm] at hv
    unfold centeredLift
    split <;> omega

/-- A coefficient equation and its integer witness imply both observed noise and the
inclusive Boolean decoder result. The witness is supplied by a whole-polynomial bound. -/
theorem final_coefficient_decoder {q B : Nat} (hq : 4 ≤ q)
    (hB : B < MxxWe.decoderRadius q) (message : Bool) (value : ZMod q)
    (e : Int) (he : e.natAbs ≤ B)
    (hvalue : value = (MxxWe.messageCenter q message : ZMod q) + (e : ZMod q)) :
    (centeredLift q (value - (MxxWe.messageCenter q message : ZMod q))).natAbs <
      MxxWe.decoderRadius q ∧ MxxWe.decoded q value.val = message := by
  letI : NeZero q := ⟨by omega⟩
  constructor
  · rw [hvalue, add_sub_cancel_left, final_centered_small hq hB e he]
    exact he.trans_lt hB
  · have hv (z : Int) : (z : ZMod q).val = MxxWe.canonicalCoeff q z := by
      have h := ZMod.val_intCast (n := q) z
      have hn := Int.emod_nonneg z (by omega : (q : Int) ≠ 0)
      exact Int.ofNat_inj.mp (by simpa [MxxWe.canonicalCoeff, Int.toNat_of_nonneg hn] using h)
    cases message
    · simp only [MxxWe.messageCenter, Bool.false_eq_true, ↓reduceIte,
        Nat.cast_zero, zero_add] at hvalue
      rw [hvalue, hv]
      exact MxxWe.decode_zero_of_small_error hq hB he
    · have hz : value = ((MxxWe.half q : Int) + e : Int) := by
        simpa [MxxWe.messageCenter] using hvalue
      rw [hz, hv]
      exact MxxWe.decode_one_of_small_error hq hB he

/-- Whole-polynomial approximation supplies the coefficient witness rather than assuming
the desired observed residual inequality. -/
theorem final_approx_decoder {q n B : Nat} (hq : 4 ≤ q) (hn : 0 < n)
    (hB : B < MxxWe.decoderRadius q) (message : Bool)
    (actual ideal : ExactMatrix q n 1 1) (happrox : Approx actual ideal B)
    (hcenter : (ideal 0 0).coeff ⟨0, hn⟩ = (MxxWe.messageCenter q message : ZMod q)) :
    (centeredLift q ((actual 0 0).coeff ⟨0, hn⟩ -
      (MxxWe.messageCenter q message : ZMod q))).natAbs < MxxWe.decoderRadius q ∧
    MxxWe.decoded q ((actual 0 0).coeff ⟨0, hn⟩).val = message := by
  obtain ⟨error, heq, hbound⟩ := happrox
  apply final_coefficient_decoder hq hB message _ ((error 0 0).coeff ⟨0, hn⟩)
    (hbound 0 0 ⟨0, hn⟩)
  rw [heq]
  change ((ideal 0 0) + reducePoly q n (error 0 0)).coeff _ = _
  rw [Negacyclic.coeff_add, hcenter, reducePoly_coeff (by omega) hn]

/-- The DSL rounding expression is the decoder quarter for any modulus. -/
theorem final_rounded_quarter (modulus : Nat) :
    MxxIR.roundDiv ((modulus : Int) - 2) 4 = ((modulus / 4 : Nat) : Int) := by
  unfold MxxIR.roundDiv
  rw [Int.fdiv_eq_ediv_of_nonneg _ (by norm_num)]
  omega

/-- The generated comparisons use the same inclusive decoder, at the exact endpoint
projection selected by the emitted Claim. -/
theorem generated_final_decoder
    (backend : BackendContext) (params : Stage_decrypt.Params) {inputs outputs}
    (hq : params.diamond_modulus = (q : Int))
    (hrun : Stage_decrypt.generatedRoot backend params inputs outputs) :
    outputs.2.2.2.2.2.1 =
      MxxWe.decoded q ((outputs.2.2.2.2.1 0 0).coeff ⟨0, by decide⟩).val := by
  obtain ⟨state, circuit, coefficient, _, hextract, hdecode, _⟩ :=
    generated_final_root backend params hrun
  obtain ⟨index, hindex, hcoeff⟩ := hextract
  have hi : index = ⟨0, by decide⟩ := Fin.ext (by change index.val = 0; omega)
  subst index
  rw [hdecode, hcoeff, hq]
  rw [final_rounded_quarter]
  unfold MxxWe.decoded MxxWe.quarter
  exact decide_eq_decide.mpr (by omega)

#print axioms generated_final_root
#print axioms final_exact_cancellation
#print axioms final_integer_residual_bound
#print axioms final_encoding_approx
#print axioms final_centered_small
#print axioms final_coefficient_decoder
#print axioms final_approx_decoder
#print axioms generated_final_decoder

end DiamondGeneratedProof
