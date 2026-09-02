import MxxGadgets.GadgetMatrix
import MxxPrimitives.Radix

namespace Mxx.Gadgets

open Mxx.Primitives
variable {q n rows columns : Nat}

/- The useful non-vacuous bridge for a concrete matrix.  The reconstruction fact is the only
   premise: no carrier metadata or operation-specific digit rule is inferred here. -/
theorem column_digits_reconstruct
    {matrix : ExactMatrix q n rows columns}
    {Limb : Type u} [Fintype Limb]
    (digits : ColumnDigits matrix Limb)
    (column : Fin columns) :
    (fun row => matrix row column) =
      fun row => ∑ limb : Limb,
        reducePoly q n (digits.digit column limb) * digits.route limb row ⟨⟩ :=
  digits.reconstruct column

theorem gadget_target_preserved
    {gadget : ExactMatrix q n rows columns}
    {decomposition : ExactMatrix q n columns columns}
    {target : ExactMatrix q n rows columns}
    (fact : GadgetDecomposition gadget decomposition target) :
    gadget * decomposition = target :=
  fact.equation

end Mxx.Gadgets
