import MxxBgg.Encoding
import MxxGadgets.GadgetMatrix

namespace Mxx.Bgg

open Mxx.Primitives
open Mxx.Gadgets

variable {q n gadgetColumns secretColumns : Nat}

/- The public transition used by multiplication.  Its target is unrestricted:
   a public term such as `B` or `G` is retained literally in `targetIdeal`. -/
structure RightPublicTransition
    (gadget : ExactMatrix q n secretColumns gadgetColumns)
    (decomposition : ExactMatrix q n gadgetColumns gadgetColumns)
    (actualTarget idealTarget : ExactMatrix q n secretColumns gadgetColumns) where
  relation : GadgetDecomposition gadget decomposition actualTarget
  approximation : Approx actualTarget idealTarget

theorem RightPublicTransition.relation_equation
    {gadget : ExactMatrix q n secretColumns gadgetColumns}
    {decomposition : ExactMatrix q n gadgetColumns gadgetColumns}
    {actualTarget idealTarget : ExactMatrix q n secretColumns gadgetColumns}
    (transition : RightPublicTransition gadget decomposition actualTarget idealTarget) :
    gadget * decomposition = actualTarget :=
  transition.relation.equation

theorem RightPublicTransition.target_equation
    {gadget : ExactMatrix q n secretColumns gadgetColumns}
    {decomposition : ExactMatrix q n gadgetColumns gadgetColumns}
    {actualTarget idealTarget : ExactMatrix q n secretColumns gadgetColumns}
    (transition : RightPublicTransition gadget decomposition actualTarget idealTarget) :
    actualTarget = idealTarget +
      reduceMatrix q n secretColumns gadgetColumns transition.approximation.error :=
  transition.approximation.equation

end Mxx.Bgg
