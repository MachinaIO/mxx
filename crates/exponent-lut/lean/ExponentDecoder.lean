import ExponentRefreshBounds
import ExponentRounding

namespace MxxExponentLut

open Mxx.Primitives

/-- Cancel the reconstructed public key against the sampled decoder anchor.
The remaining error is the paired switch error minus the anchor error acted on
by the actual integer decomposition. This matches the decoder term in
`noise_refresh/simulation.rs`; no output error estimate is a premise. -/
theorem decoder_cancellation_bound
    {q n secret inner columns pre lift gain cutoff : Nat}
    (hn : 0 < n) (hlift : 0 < lift)
    (s : ExactMatrix q n 1 secret) (b : ExactMatrix q n secret inner)
    (a : ExactMatrix q n secret columns) (d : ExactMatrix q n inner columns)
    (ciphertext : ExactMatrix q n 1 columns) (anchor : ExactMatrix q n 1 inner)
    (ideal pair : ErrorMatrix n 1 columns) (anchorError : ErrorMatrix n 1 inner)
    (action : ErrorMatrix n inner columns)
    (hpair : ciphertext - s * a - reduceMatrix q n 1 columns ideal =
      reduceMatrix q n 1 columns pair)
    (hanchor : anchor = s * b + reduceMatrix q n 1 inner anchorError)
    (hreconstruct : b * d = a)
    (haction : d = reduceMatrix q n inner columns action)
    (hgain : ActionGain action gain) (herror : CoeffBound anchorError cutoff)
    (hscaled : ∀ row column coefficient,
      2 * lift * ((pair row column).coeff coefficient).natAbs ≤
        2 * pre + (lift - 1) * (n + 2)) :
    ∃ error : ErrorMatrix n 1 columns,
      ciphertext - anchor * d - reduceMatrix q n 1 columns ideal =
        reduceMatrix q n 1 columns error ∧
      CoeffBound error (switchedErrorBound pre lift n (gain * cutoff)) := by
  refine ⟨pair - anchorError * action, ?_, ?_⟩
  · have hreduce : reduceMatrix q n 1 columns (pair - anchorError * action) =
        reduceMatrix q n 1 columns pair -
          reduceMatrix q n 1 columns (anchorError * action) := by
      ext row column
      exact map_sub (reducePoly q n) _ _
    rw [hreduce, reduceMatrix_mul, ← haction, ← hpair, hanchor,
      Matrix.add_mul, Matrix.mul_assoc, hreconstruct]
    abel
  · have hb := right_action_bound hn herror hgain
    intro row column coefficient
    simpa only [Matrix.sub_apply, Negacyclic.coeff_sub, sub_eq_add_neg, Negacyclic.coeff_add, Negacyclic.coeff_neg] using
      (switched_error_bound hlift (hscaled row column coefficient)
        (by simpa using hb row column coefficient) :
        (((pair row column).coeff coefficient) +
          -((anchorError * action) row column).coeff coefficient).natAbs ≤
            switchedErrorBound pre lift n (gain * cutoff))

/-- The exact acceptance inequality supplies the strict rounding margin for
all coefficients after decoder cancellation, including the independent mask. -/
theorem decoder_mask_strict_rounding
    {n rows columns pre lift gain cutoff mask spacing : Nat}
    {error maskError : ErrorMatrix n rows columns}
    (he : CoeffBound error (switchedErrorBound pre lift n (gain * cutoff)))
    (hm : CoeffBound maskError mask)
    (haccepted : 2 * (mask + switchedErrorBound pre lift n (gain * cutoff)) < spacing) :
    ∀ row column coefficient,
      2 * (((maskError + error) row column).coeff coefficient).natAbs < spacing := by
  intro row column coefficient
  simpa only [Matrix.add_apply, Negacyclic.coeff_add] using
    strict_rounding_bound (hm row column coefficient) (he row column coefficient) haccepted

/-- Apply the runtime canonical nearest-digit formula to a decoded coefficient.
The coefficient equation is supplied by cancellation and the ideal/mask split;
the rounding conclusion follows from the simulator acceptance check. -/
theorem decoder_rounded_digit
    {q p n pre lift gain cutoff mask spacing : Nat}
    (hp : 0 < p) (hs : 0 < spacing) (hq : q = p * spacing)
    (x : ZMod q) (message maskError routeError : Int)
    (hx : x = ((spacing : Int) * message + (maskError + routeError) : Int))
    (hm : maskError.natAbs ≤ mask)
    (hr : routeError.natAbs ≤ switchedErrorBound pre lift n (gain * cutoff))
    (haccepted : 2 * (mask + switchedErrorBound pre lift n (gain * cutoff)) < spacing) :
    (((p : Int) * Int.ofNat x.val + Int.ofNat (q / 2)) / Int.ofNat q) % (p : Int) =
      message % (p : Int) := by
  exact crt_rounded_digit_of_strict_error hp hs hq x message (maskError + routeError)
    hx (strict_rounding_bound hm hr haccepted)

end MxxExponentLut
