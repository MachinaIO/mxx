import MxxPrimitives.Preimage

namespace Mxx.Primitives

variable {q n rows columns sourceRows inner targetColumns : Nat}

inductive SampleKind
  | binary
  | interval
  | gaussian
  | uniform
  | hash
  | trapdoor

structure CutoffSampleFact
    (actual : ExactMatrix q n rows columns) (bound : Nat) where
  lift : BoundedLift actual bound
  kind : SampleKind
  kind_is_cutoff : kind = .binary ∨ kind = .interval ∨ kind = .gaussian

structure CanonicalLiftFact
    (actual : ExactMatrix q n rows columns) where
  lift : ErrorMatrix n rows columns
  reduce_eq : reduceMatrix q n rows columns lift = actual

structure TrapdoorSampleFact
    (actual : ExactMatrix q n rows columns) where
  present : True

structure PreimageSampleFact
    (source : ExactMatrix q n sourceRows inner)
    (preimage : ExactMatrix q n inner targetColumns)
    (target : ExactMatrix q n sourceRows targetColumns)
    (preimageBound : Nat) where
  relation : RightPreimage source preimage target
  bounded : PreimageWithin preimage preimageBound

def hasMagnitude {q n rows columns : Nat}
    (actual : ExactMatrix q n rows columns) (bound : Nat) : Prop :=
  Nonempty (BoundedLift actual bound)

end Mxx.Primitives
