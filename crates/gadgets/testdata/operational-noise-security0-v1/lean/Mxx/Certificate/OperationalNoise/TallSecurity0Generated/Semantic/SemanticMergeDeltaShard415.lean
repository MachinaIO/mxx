import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge67743
def owner : Owner := ⟨.program ⟨214⟩, ⟨12368⟩⟩
def mergeEvent : Nat := 67743
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9815⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6785⟩⟩] } }
def leftRaw : List Term := Proof.Events264.exact67736RawTerms
def rightRaw : List Term := Proof.Events012.exact3204RawTerms
def group : MergeGroup := .operator 67736 3204
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 67736) (leftOrdinal := 0)
    (rightResult := 3204) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6785⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨9815⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9815⟩⟩], [⟨.program ⟨214⟩, ⟨6785⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge67743

namespace LeftMerge67748
def owner : Owner := ⟨.program ⟨214⟩, ⟨9816⟩⟩
def mergeEvent : Nat := 67748
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9815⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events012.exact3204RawTerms
def rightRaw : List Term := Proof.Events255.exact65295RawTerms
def group : MergeGroup := .operator 3204 65295
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 3204) (leftOrdinal := 0)
    (rightResult := 65295) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨9815⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9815⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge67748

namespace LeftMerge67753
def owner : Owner := ⟨.program ⟨214⟩, ⟨7183⟩⟩
def mergeEvent : Nat := 67753
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6765⟩⟩] } }
def leftRaw : List Term := Proof.Events254.exact65165RawTerms
def rightRaw : List Term := Proof.Events035.exact9018RawTerms
def group : MergeGroup := .operator 65165 9018
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 65165) (leftOrdinal := 0)
    (rightResult := 9018) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6765⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6765⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge67753

namespace LeftMerge67770
def owner : Owner := ⟨.program ⟨214⟩, ⟨9819⟩⟩
def mergeEvent : Nat := 67770
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9815⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7867⟩⟩] } }
def leftRaw : List Term := Proof.Events264.exact67764RawTerms
def rightRaw : List Term := Proof.Events035.exact9007RawTerms
def group : MergeGroup := .operator 67764 9007
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 67764) (leftOrdinal := 1)
    (rightResult := 9007) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9815⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨7867⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9815⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7867⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge67770

namespace LeftMerge67772
def owner : Owner := ⟨.program ⟨214⟩, ⟨9819⟩⟩
def mergeEvent : Nat := 67772
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9815⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6785⟩⟩] } }
def rhsRaw : List Term := Proof.Events035.exact8977RawTerms
def group : MergeGroup := .relation 67771
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 67771) (rhsResult := 8977)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9815⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7867⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨7867⟩⟩) ⟨6785⟩ 8977) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6785⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9815⟩⟩], [⟨.program ⟨214⟩, ⟨6785⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge67772

namespace LeftMerge67773
def owner : Owner := ⟨.program ⟨214⟩, ⟨9819⟩⟩
def mergeEvent : Nat := 67773
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6765⟩⟩, ⟨.program ⟨214⟩, ⟨7867⟩⟩] } }
def leftRaw : List Term := Proof.Events264.exact67764RawTerms
def rightRaw : List Term := Proof.Events035.exact9007RawTerms
def group : MergeGroup := .operator 67764 9007
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 67764) (leftOrdinal := 0)
    (rightResult := 9007) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6765⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨7867⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6765⟩⟩, ⟨.program ⟨214⟩, ⟨7867⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge67773

namespace LeftMerge67778
def owner : Owner := ⟨.program ⟨214⟩, ⟨12369⟩⟩
def mergeEvent : Nat := 67778
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9815⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6785⟩⟩] } }
def leftRaw : List Term := Proof.Events264.exact67774RawTerms
def rightRaw : List Term := Proof.Events264.exact67744RawTerms
def group : MergeGroup := .operator 67774 67744
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 67774) (leftOrdinal := 1)
    (rightResult := 67744) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9815⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6785⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9815⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6785⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9815⟩⟩], [⟨.program ⟨214⟩, ⟨6785⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge67778

namespace LeftMerge67786
def owner : Owner := ⟨.program ⟨214⟩, ⟨25369⟩⟩
def mergeEvent : Nat := 67786
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9815⟩⟩, ⟨.program ⟨214⟩, ⟨12362⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25368⟩⟩] } }
def leftRaw : List Term := Proof.Events264.exact67780RawTerms
def rightRaw : List Term := Proof.Events264.exact67716RawTerms
def group : MergeGroup := .operator 67780 67716
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 67780) (leftOrdinal := 1)
    (rightResult := 67716) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9815⟩⟩, ⟨.program ⟨214⟩, ⟨12362⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25368⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9815⟩⟩, ⟨.program ⟨214⟩, ⟨12362⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25368⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge67786

namespace LeftMerge67788
def owner : Owner := ⟨.program ⟨214⟩, ⟨25369⟩⟩
def mergeEvent : Nat := 67788
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9815⟩⟩, ⟨.program ⟨214⟩, ⟨12362⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23204⟩⟩] } }
def rhsRaw : List Term := Proof.Events264.exact67713RawTerms
def group : MergeGroup := .relation 67787
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 67787) (rhsResult := 67713)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9815⟩⟩, ⟨.program ⟨214⟩, ⟨12362⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25368⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨25368⟩⟩) ⟨23204⟩ 67713) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23204⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9815⟩⟩, ⟨.program ⟨214⟩, ⟨12362⟩⟩], [⟨.program ⟨214⟩, ⟨23204⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge67788

namespace LeftMerge67789
def owner : Owner := ⟨.program ⟨214⟩, ⟨25369⟩⟩
def mergeEvent : Nat := 67789
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6765⟩⟩, ⟨.program ⟨214⟩, ⟨7867⟩⟩, ⟨.program ⟨214⟩, ⟨25368⟩⟩] } }
def leftRaw : List Term := Proof.Events264.exact67780RawTerms
def rightRaw : List Term := Proof.Events264.exact67716RawTerms
def group : MergeGroup := .operator 67780 67716
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 67780) (leftOrdinal := 0)
    (rightResult := 67716) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6765⟩⟩, ⟨.program ⟨214⟩, ⟨7867⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25368⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6765⟩⟩, ⟨.program ⟨214⟩, ⟨7867⟩⟩, ⟨.program ⟨214⟩, ⟨25368⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge67789

namespace LeftMerge67803
def owner : Owner := ⟨.program ⟨214⟩, ⟨19887⟩⟩
def mergeEvent : Nat := 67803
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19884⟩⟩] } }
def leftRaw : List Term := Proof.Events255.exact65387RawTerms
def rightRaw : List Term := Proof.Events264.exact67797RawTerms
def group : MergeGroup := .operator 65387 67797
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 65387) (leftOrdinal := 0)
    (rightResult := 67797) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨19884⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19884⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge67803

namespace LeftMerge67882
def owner : Owner := ⟨.program ⟨214⟩, ⟨12363⟩⟩
def mergeEvent : Nat := 67882
def frameStart : Nat := 67852
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨9815⟩⟩, ⟨.program ⟨214⟩, ⟨12362⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events265.exact67878RawTerms
def rightRaw : List Term := Proof.Events265.exact67875RawTerms
def group : MergeGroup := .operator 67878 67875
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 67878) (leftOrdinal := 0)
    (rightResult := 67875) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨9815⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨12362⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨9815⟩⟩, ⟨.program ⟨214⟩, ⟨12362⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge67882

namespace LeftMerge67912
def owner : Owner := ⟨.program ⟨214⟩, ⟨12464⟩⟩
def mergeEvent : Nat := 67912
def frameStart : Nat := 67852
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨9815⟩⟩, ⟨.program ⟨214⟩, ⟨12362⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events265.exact67908RawTerms
def rightRaw : List Term := Proof.Events265.exact67906RawTerms
def group : MergeGroup := .operator 67908 67906
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 67908) (leftOrdinal := 0)
    (rightResult := 67906) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨9815⟩⟩, ⟨.program ⟨214⟩, ⟨12362⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨9815⟩⟩, ⟨.program ⟨214⟩, ⟨12362⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge67912

namespace LeftMerge67935
def owner : Owner := ⟨.program ⟨214⟩, ⟨7869⟩⟩
def mergeEvent : Nat := 67935
def frameStart : Nat := 67852
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6765⟩⟩, ⟨.program ⟨214⟩, ⟨7867⟩⟩] } }
def leftRaw : List Term := Proof.Events265.exact67931RawTerms
def rightRaw : List Term := Proof.Events265.exact67928RawTerms
def group : MergeGroup := .operator 67931 67928
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 67931) (leftOrdinal := 0)
    (rightResult := 67928) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6765⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨7867⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6765⟩⟩, ⟨.program ⟨214⟩, ⟨7867⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge67935

namespace LeftMerge67944
def owner : Owner := ⟨.program ⟨214⟩, ⟨25371⟩⟩
def mergeEvent : Nat := 67944
def frameStart : Nat := 67852
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6765⟩⟩, ⟨.program ⟨214⟩, ⟨7867⟩⟩, ⟨.program ⟨214⟩, ⟨25368⟩⟩] } }
def leftRaw : List Term := Proof.Events265.exact67940RawTerms
def rightRaw : List Term := Proof.Events265.exact67897RawTerms
def group : MergeGroup := .operator 67940 67897
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 67940) (leftOrdinal := 0)
    (rightResult := 67897) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6765⟩⟩, ⟨.program ⟨214⟩, ⟨7867⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25368⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6765⟩⟩, ⟨.program ⟨214⟩, ⟨7867⟩⟩, ⟨.program ⟨214⟩, ⟨25368⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge67944

namespace LeftMerge67945
def owner : Owner := ⟨.program ⟨214⟩, ⟨25371⟩⟩
def mergeEvent : Nat := 67945
def frameStart : Nat := 67852
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨9815⟩⟩, ⟨.program ⟨214⟩, ⟨12362⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25368⟩⟩] } }
def leftRaw : List Term := Proof.Events265.exact67940RawTerms
def rightRaw : List Term := Proof.Events265.exact67897RawTerms
def group : MergeGroup := .operator 67940 67897
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 67940) (leftOrdinal := 1)
    (rightResult := 67897) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨9815⟩⟩, ⟨.program ⟨214⟩, ⟨12362⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25368⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨9815⟩⟩, ⟨.program ⟨214⟩, ⟨12362⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25368⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge67945

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
