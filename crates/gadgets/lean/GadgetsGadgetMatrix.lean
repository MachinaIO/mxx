import PrimitivesPreimage

namespace Mxx.Gadgets

open Mxx.Primitives

variable {q n sourceRows inner targetColumns : Nat}

/- A gadget decomposition is deliberately the same relation as an ordinary right preimage.
   The gadget-specific construction and its sampler live in Rust; Lean only consumes its
   checked equation. -/
abbrev GadgetDecomposition
    (gadget : ExactMatrix q n sourceRows inner)
    (decomposition : ExactMatrix q n inner targetColumns)
    (target : ExactMatrix q n sourceRows targetColumns) : Prop :=
  RightPreimage gadget decomposition target

theorem gadget_decomposition_equation
    {gadget : ExactMatrix q n sourceRows inner}
    {decomposition : ExactMatrix q n inner targetColumns}
    {target : ExactMatrix q n sourceRows targetColumns}
    (fact : GadgetDecomposition gadget decomposition target) :
    gadget * decomposition = target :=
  fact.equation

end Mxx.Gadgets
