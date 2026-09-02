import MxxGadgets.GadgetMatrix
import MxxPrimitives.Preimage

namespace Mxx.Gadgets

open Mxx.Primitives

/- The matrix-level algebra is delegated to the primitive package.  This wrapper gives gadget and
   application layers a stable name while retaining the rectangular matrix dimensions. -/
theorem consume_right_preimage_matrix
    {R : Type u} [Semiring R]
    {s k c r : Type v} [Fintype s] [Fintype k]
    {B : Matrix s k R} {K : Matrix k c R} {P E : Matrix s c R}
    {X eX : Matrix r k R} {L : Matrix r s R}
    (hvalue : X = L * B + eX)
    (htarget : B * K = P + E) :
    X * K = L * P + (L * E + eX * K) :=
  consume_rectangular_semiring B K (P + E) P L X eX E hvalue htarget (by rfl)

/- This is the equation used by input injection when the target carries an error matrix.  The
   target ideal remains arbitrary, so public sources occurring in it are preserved. -/
theorem input_injector_equation
    {R : Type u} [Semiring R]
    {s k c r : Type v} [Fintype s] [Fintype k]
    {B : Matrix s k R} {K : Matrix k c R} {P E : Matrix s c R}
    {X eX : Matrix r k R} {L : Matrix r s R}
    (hvalue : X = L * B + eX)
    (htarget : B * K = P + E) :
    X * K = L * P + (L * E + eX * K) :=
  consume_right_preimage_matrix hvalue htarget

/- Concrete proof-side interface used by applications. `Approx.error` is the internally derived
   expression `leftMagnitude.lift * targetApprox.error + valueApprox.error * preimageLift.witness`.
   The proof itself is delegated to the primitive theorem. -/
noncomputable def input_injector_consumption
    {q n sourceRows inner targetColumns resultRows : Nat}
    (source : ExactMatrix q n sourceRows inner)
    (actualPreimage : ExactMatrix q n inner targetColumns)
    (actualTarget : ExactMatrix q n sourceRows targetColumns)
    (left : ExactMatrix q n resultRows sourceRows)
    (value : ExactMatrix q n resultRows inner)
    (idealTarget : ExactMatrix q n sourceRows targetColumns)
    (relation : RightPreimage source actualPreimage actualTarget)
    (leftMagnitude : MagnitudeFact left)
    {preimageBound xNoiseBound targetNoiseBound : Nat}
    (preimageLift : BoundedLift actualPreimage preimageBound)
    (valueApprox : ApproxWithin value (left * source) xNoiseBound)
    (targetApprox : ApproxWithin actualTarget idealTarget targetNoiseBound) :
    Approx (value * actualPreimage) (left * idealTarget) :=
  consume_right_preimage_rectangular source actualPreimage actualTarget left value idealTarget
    relation leftMagnitude preimageLift valueApprox targetApprox

/- The bounded companion derives both product estimates in the primitive package. -/
noncomputable def input_injector_within
    {q n sourceRows inner targetColumns resultRows : Nat}
    (hn : 0 < n)
    (source : ExactMatrix q n sourceRows inner)
    (actualPreimage : ExactMatrix q n inner targetColumns)
    (actualTarget : ExactMatrix q n sourceRows targetColumns)
    (left : ExactMatrix q n resultRows sourceRows)
    (value : ExactMatrix q n resultRows inner)
    (idealTarget : ExactMatrix q n sourceRows targetColumns)
    (relation : RightPreimage source actualPreimage actualTarget)
    (leftMagnitude : MagnitudeFact left)
    {preimageBound xNoiseBound targetNoiseBound : Nat}
    (preimageLift : BoundedLift actualPreimage preimageBound)
    (valueApprox : ApproxWithin value (left * source) xNoiseBound)
    (targetApprox : ApproxWithin actualTarget idealTarget targetNoiseBound) :
      ApproxWithin (value * actualPreimage) (left * idealTarget)
      (sourceRows * n * leftMagnitude.bound * targetNoiseBound +
        inner * n * xNoiseBound * preimageBound) :=
  consume_right_preimage_rectangular_with_bound hn source actualPreimage actualTarget left value
    idealTarget relation leftMagnitude preimageLift valueApprox targetApprox

noncomputable def input_injector_within_constant_left
    {q n sourceRows inner targetColumns resultRows : Nat}
    (hn : 0 < n)
    (source : ExactMatrix q n sourceRows inner)
    (actualPreimage : ExactMatrix q n inner targetColumns)
    (actualTarget : ExactMatrix q n sourceRows targetColumns)
    (left : ExactMatrix q n resultRows sourceRows)
    (value : ExactMatrix q n resultRows inner)
    (idealTarget : ExactMatrix q n sourceRows targetColumns)
    (relation : RightPreimage source actualPreimage actualTarget)
    (leftMagnitude : MagnitudeFact left)
    (hconstant : leftMagnitude.support = SupportClass.constant)
    {preimageBound xNoiseBound targetNoiseBound : Nat}
    (preimageLift : BoundedLift actualPreimage preimageBound)
    (valueApprox : ApproxWithin value (left * source) xNoiseBound)
    (targetApprox : ApproxWithin actualTarget idealTarget targetNoiseBound) :
    ApproxWithin (value * actualPreimage) (left * idealTarget)
      (sourceRows * leftMagnitude.bound * targetNoiseBound +
        inner * n * xNoiseBound * preimageBound) :=
  consume_right_preimage_rectangular_with_constant_left_bound hn source actualPreimage
    actualTarget left value idealTarget relation leftMagnitude hconstant preimageLift valueApprox
    targetApprox

end Mxx.Gadgets
