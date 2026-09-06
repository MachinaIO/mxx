import PrimitivesPreimage

namespace Mxx.Gadgets

open Mxx.Primitives

variable {q n sourceRows inner targetColumns : Nat}

/- The residual belongs to this exact source/decomposition pair. It is an integer witness,
   not a fresh independent noise sample. Exact right preimages remain a separate relation. -/
abbrev GadgetDecomposition
    (gadget : ExactMatrix q n sourceRows inner)
    (decomposition : ExactMatrix q n inner targetColumns)
    (target : ExactMatrix q n sourceRows targetColumns) (bound : Nat) : Prop :=
  Approx target (gadget * decomposition) bound

theorem gadget_decomposition_equation
    {gadget : ExactMatrix q n sourceRows inner}
    {decomposition : ExactMatrix q n inner targetColumns}
    {target : ExactMatrix q n sourceRows targetColumns}
    {bound : Nat}
    (fact : GadgetDecomposition gadget decomposition target bound) :
    ∃ residual : ErrorMatrix n sourceRows targetColumns,
      target = gadget * decomposition + reduceMatrix q n sourceRows targetColumns residual ∧
        CoeffBound residual bound := fact

end Mxx.Gadgets
