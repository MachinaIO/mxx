import MxxPrimitives.Preimage

namespace Mxx.Primitives

inductive SampleKind where
  | gaussian
  | uniformInterval
  | uniformResidue
  | preimage
  | gadgetDecomposition

structure CutoffSampleFact {q n rows columns : Nat}
    (value : ExactMatrix q n rows columns) (bound : Nat) where
  witness : ErrorMatrix n rows columns
  equation : value = reduceMatrix q n rows columns witness
  bounded : CoeffBound witness bound

structure TrapdoorSampleFact {q n rows columns : Nat}
    (publicMatrix : ExactMatrix q n rows columns) where
  token : Type

structure PreimageSampleFact {q n sourceRows inner targetColumns : Nat}
    (source : ExactMatrix q n sourceRows inner)
    (preimage : ExactMatrix q n inner targetColumns)
    (target : ExactMatrix q n sourceRows targetColumns) (bound : Nat) : Prop where
  relation : RightPreimage source preimage target
  bounded : PreimageWithin preimage bound

def hasMagnitude {q n rows columns : Nat}
    (value : ExactMatrix q n rows columns) (bound : Nat) : Prop :=
  Approx value (0 : ExactMatrix q n rows columns) bound

end Mxx.Primitives
