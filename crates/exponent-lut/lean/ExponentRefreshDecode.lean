import ExponentPairedSwitch
import ExponentDecoder

namespace MxxExponentLut
open Mxx.Primitives

theorem refresh_lookup_eq {α : Type} {count : Nat} {family : Fin count → α}
    {index : Int} {output : α} (position : Fin count)
    (hindex : index = (position.val : Int))
    (h : MxxRuntime.familyGetStatic family index output) : output = family position := by
  obtain ⟨actual, ha, hv⟩ := h
  have same : actual = position := by apply Fin.ext; omega
  simpa only [same] using hv

/-- The sampled preimage reconstructs the switched key minus its public
correction. Consequently decoder cancellation retains the secret times that
correction in the ideal. This is the sign used by `noise_refresh/setup.rs`
and the actual subtraction in `noise_refresh/refresh.rs`. -/
theorem refresh_switched_decoder
    {q p n columns inner factor pre gain cutoff : Nat}
    (hn : 0 < n) (hf : 0 < factor) (hq : q = factor * p)
    (secret : ErrorMatrix n 1 2) (ideal : ErrorMatrix n 1 columns)
    (ciphertext : ExactMatrix q n 1 columns) (publicKey : ExactMatrix q n 2 columns)
    (switchedCiphertext : ExactMatrix p n 1 columns)
    (switchedKey correction : ExactMatrix p n 2 columns)
    (basis : ExactMatrix p n 2 inner) (anchor : ExactMatrix p n 1 inner)
    (decomposition : ExactMatrix p n inner columns)
    (anchorError : ErrorMatrix n 1 inner) (action : ErrorMatrix n inner columns)
    (hsmall : ∀ i, ((secret 0 0).coeff i).natAbs ≤ 1) (hunit : secret 0 1 = 1)
    (hsource : Approx ciphertext (reduceMatrix q n 1 2 secret * publicKey +
      reduceMatrix q n 1 columns ((factor : ErrorPoly n) • ideal)) pre)
    (hc : MxxRuntime.modulusSwitchRuns ciphertext switchedCiphertext)
    (ha : MxxRuntime.modulusSwitchRuns publicKey switchedKey)
    (hanchor : anchor = reduceMatrix p n 1 2 secret * basis +
      reduceMatrix p n 1 inner anchorError)
    (hreconstruct : basis * decomposition = switchedKey - correction)
    (haction : decomposition = reduceMatrix p n inner columns action)
    (hgain : ActionGain action gain) (herror : CoeffBound anchorError cutoff) :
    Approx (switchedCiphertext - anchor * decomposition)
      (reduceMatrix p n 1 columns ideal + reduceMatrix p n 1 2 secret * correction)
      (switchedErrorBound pre factor n (gain * cutoff)) := by
  obtain ⟨preError, sourceEquation, preBound⟩ := hsource
  obtain ⟨pair, pairEquation, pairBound⟩ := paired_switch_error hn hf hq secret ideal preError
    ciphertext publicKey switchedCiphertext switchedKey hsmall hunit preBound
    (by rw [reduceMatrix_add]; simpa only [add_assoc] using sourceEquation) hc ha
  have adjusted : (switchedCiphertext - reduceMatrix p n 1 2 secret * correction) -
      reduceMatrix p n 1 2 secret * (switchedKey - correction) -
      reduceMatrix p n 1 columns ideal = reduceMatrix p n 1 columns pair := by
    rw [Matrix.mul_sub]
    rw [← pairEquation]
    abel
  obtain ⟨error, equation, bounded⟩ := decoder_cancellation_bound hn hf
    (reduceMatrix p n 1 2 secret) basis (switchedKey - correction) decomposition
    (switchedCiphertext - reduceMatrix p n 1 2 secret * correction) anchor
    ideal pair anchorError action adjusted hanchor hreconstruct haction hgain herror pairBound
  refine ⟨error, ?_, bounded⟩
  calc
    _ = ((switchedCiphertext - reduceMatrix p n 1 2 secret * correction) -
        anchor * decomposition - reduceMatrix p n 1 columns ideal) +
        (reduceMatrix p n 1 columns ideal + reduceMatrix p n 1 2 secret * correction) := by abel
    _ = _ := by rw [equation]; abel

/-- Each actual level coefficient rounds to its intended plaintext residue
once the ideal has been split into a spacing multiple and the independent
mask. The error witness comes from decoder cancellation, not a decoded-output
assumption. -/
theorem refresh_rounded_residues
    {q plaintext n rows columns pre factor gain cutoff mask spacing : Nat}
    (hn : 0 < n) (hqgt : 1 < q)
    (hp : 0 < plaintext) (hs : 0 < spacing) (hq : q = plaintext * spacing)
    (level ideal : ExactMatrix q n rows columns)
    (message maskError : ErrorMatrix n rows columns)
    (hlevel : Approx level ideal (switchedErrorBound pre factor n (gain * cutoff)))
    (hideal : ∀ row column coefficient, (ideal row column).coeff coefficient =
      ((spacing : Int) * (message row column).coeff coefficient +
        (maskError row column).coeff coefficient : Int))
    (hm : CoeffBound maskError mask)
    (haccepted : 2 * (mask + switchedErrorBound pre factor n (gain * cutoff)) < spacing) :
    ∀ row column coefficient,
      (((plaintext : Int) * Int.ofNat ((level row column).coeff coefficient).val +
        Int.ofNat (q / 2)) / Int.ofNat q) % (plaintext : Int) =
          (message row column).coeff coefficient % (plaintext : Int) := by
  obtain ⟨error, equation, bounded⟩ := hlevel
  intro row column coefficient
  apply decoder_rounded_digit hp hs hq _ _ _ _ ?_
    (hm row column coefficient) (bounded row column coefficient) haccepted
  rw [equation, Matrix.add_apply, Negacyclic.coeff_add, hideal]
  simp only [reduceMatrix, reducePoly_coeff hqgt hn, Int.cast_add]
  ring

end MxxExponentLut
