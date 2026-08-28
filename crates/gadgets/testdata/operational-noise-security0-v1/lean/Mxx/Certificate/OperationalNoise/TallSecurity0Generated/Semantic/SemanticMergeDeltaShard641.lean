import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge103742
def owner : Owner := ⟨.program ⟨214⟩, ⟨30067⟩⟩
def mergeEvent : Nat := 103742
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6733⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩] } }
def leftRaw : List Term := Proof.Events405.exact103728RawTerms
def rightRaw : List Term := Proof.Events399.exact102336RawTerms
def group : MergeGroup := .operator 103728 102336
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 103728) (leftOrdinal := 12)
    (rightResult := 102336) (rightOrdinal := 12) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6733⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6733⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6733⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge103742

namespace LeftMerge103743
def owner : Owner := ⟨.program ⟨214⟩, ⟨30067⟩⟩
def mergeEvent : Nat := 103743
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨17897⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18612⟩⟩] } }
def leftRaw : List Term := Proof.Events405.exact103728RawTerms
def rightRaw : List Term := Proof.Events399.exact102336RawTerms
def group : MergeGroup := .operator 103728 102336
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 103728) (leftOrdinal := 33)
    (rightResult := 102336) (rightOrdinal := 32) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨17897⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18612⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨17897⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18612⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨17897⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge103743

namespace LeftMerge103744
def owner : Owner := ⟨.program ⟨214⟩, ⟨30067⟩⟩
def mergeEvent : Nat := 103744
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6731⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩] } }
def leftRaw : List Term := Proof.Events405.exact103728RawTerms
def rightRaw : List Term := Proof.Events399.exact102336RawTerms
def group : MergeGroup := .operator 103728 102336
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 103728) (leftOrdinal := 11)
    (rightResult := 102336) (rightOrdinal := 11) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6731⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6731⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6731⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge103744

namespace LeftMerge103745
def owner : Owner := ⟨.program ⟨214⟩, ⟨30067⟩⟩
def mergeEvent : Nat := 103745
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨17113⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18612⟩⟩] } }
def leftRaw : List Term := Proof.Events405.exact103728RawTerms
def rightRaw : List Term := Proof.Events399.exact102336RawTerms
def group : MergeGroup := .operator 103728 102336
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 103728) (leftOrdinal := 31)
    (rightResult := 102336) (rightOrdinal := 30) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨17113⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18612⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨17113⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18612⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨17113⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge103745

namespace LeftMerge103746
def owner : Owner := ⟨.program ⟨214⟩, ⟨30067⟩⟩
def mergeEvent : Nat := 103746
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6729⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩] } }
def leftRaw : List Term := Proof.Events405.exact103728RawTerms
def rightRaw : List Term := Proof.Events399.exact102336RawTerms
def group : MergeGroup := .operator 103728 102336
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 103728) (leftOrdinal := 10)
    (rightResult := 102336) (rightOrdinal := 10) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6729⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6729⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6729⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge103746

namespace LeftMerge103747
def owner : Owner := ⟨.program ⟨214⟩, ⟨30067⟩⟩
def mergeEvent : Nat := 103747
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16301⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18612⟩⟩] } }
def leftRaw : List Term := Proof.Events405.exact103728RawTerms
def rightRaw : List Term := Proof.Events399.exact102336RawTerms
def group : MergeGroup := .operator 103728 102336
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 103728) (leftOrdinal := 27)
    (rightResult := 102336) (rightOrdinal := 26) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16301⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18612⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16301⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18612⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16301⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge103747

namespace LeftMerge103748
def owner : Owner := ⟨.program ⟨214⟩, ⟨30067⟩⟩
def mergeEvent : Nat := 103748
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6727⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩] } }
def leftRaw : List Term := Proof.Events405.exact103728RawTerms
def rightRaw : List Term := Proof.Events399.exact102336RawTerms
def group : MergeGroup := .operator 103728 102336
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 103728) (leftOrdinal := 9)
    (rightResult := 102336) (rightOrdinal := 9) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6727⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6727⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6727⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge103748

namespace LeftMerge103749
def owner : Owner := ⟨.program ⟨214⟩, ⟨30067⟩⟩
def mergeEvent : Nat := 103749
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨18303⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18612⟩⟩] } }
def leftRaw : List Term := Proof.Events405.exact103728RawTerms
def rightRaw : List Term := Proof.Events399.exact102336RawTerms
def group : MergeGroup := .operator 103728 102336
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 103728) (leftOrdinal := 36)
    (rightResult := 102336) (rightOrdinal := 35) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨18303⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18612⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨18303⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18612⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨18303⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge103749

namespace LeftMerge103750
def owner : Owner := ⟨.program ⟨214⟩, ⟨30067⟩⟩
def mergeEvent : Nat := 103750
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6725⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩] } }
def leftRaw : List Term := Proof.Events405.exact103728RawTerms
def rightRaw : List Term := Proof.Events399.exact102336RawTerms
def group : MergeGroup := .operator 103728 102336
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 103728) (leftOrdinal := 8)
    (rightResult := 102336) (rightOrdinal := 8) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6725⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6725⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6725⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge103750

namespace LeftMerge103751
def owner : Owner := ⟨.program ⟨214⟩, ⟨30067⟩⟩
def mergeEvent : Nat := 103751
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16098⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18612⟩⟩] } }
def leftRaw : List Term := Proof.Events405.exact103728RawTerms
def rightRaw : List Term := Proof.Events399.exact102336RawTerms
def group : MergeGroup := .operator 103728 102336
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 103728) (leftOrdinal := 26)
    (rightResult := 102336) (rightOrdinal := 25) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16098⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18612⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16098⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18612⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16098⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge103751

namespace LeftMerge103752
def owner : Owner := ⟨.program ⟨214⟩, ⟨30067⟩⟩
def mergeEvent : Nat := 103752
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6723⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩] } }
def leftRaw : List Term := Proof.Events405.exact103728RawTerms
def rightRaw : List Term := Proof.Events399.exact102336RawTerms
def group : MergeGroup := .operator 103728 102336
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 103728) (leftOrdinal := 7)
    (rightResult := 102336) (rightOrdinal := 7) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6723⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6723⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6723⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge103752

namespace LeftMerge103753
def owner : Owner := ⟨.program ⟨214⟩, ⟨30067⟩⟩
def mergeEvent : Nat := 103753
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15979⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18612⟩⟩] } }
def leftRaw : List Term := Proof.Events405.exact103728RawTerms
def rightRaw : List Term := Proof.Events399.exact102336RawTerms
def group : MergeGroup := .operator 103728 102336
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 103728) (leftOrdinal := 25)
    (rightResult := 102336) (rightOrdinal := 24) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15979⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18612⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15979⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18612⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15979⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge103753

namespace LeftMerge103754
def owner : Owner := ⟨.program ⟨214⟩, ⟨30067⟩⟩
def mergeEvent : Nat := 103754
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6721⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩] } }
def leftRaw : List Term := Proof.Events405.exact103728RawTerms
def rightRaw : List Term := Proof.Events399.exact102336RawTerms
def group : MergeGroup := .operator 103728 102336
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 103728) (leftOrdinal := 6)
    (rightResult := 102336) (rightOrdinal := 6) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6721⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6721⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6721⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge103754

namespace LeftMerge103755
def owner : Owner := ⟨.program ⟨214⟩, ⟨30067⟩⟩
def mergeEvent : Nat := 103755
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15860⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18612⟩⟩] } }
def leftRaw : List Term := Proof.Events405.exact103728RawTerms
def rightRaw : List Term := Proof.Events399.exact102336RawTerms
def group : MergeGroup := .operator 103728 102336
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 103728) (leftOrdinal := 24)
    (rightResult := 102336) (rightOrdinal := 23) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15860⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18612⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15860⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18612⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15860⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge103755

namespace LeftMerge103756
def owner : Owner := ⟨.program ⟨214⟩, ⟨30067⟩⟩
def mergeEvent : Nat := 103756
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6719⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩] } }
def leftRaw : List Term := Proof.Events405.exact103728RawTerms
def rightRaw : List Term := Proof.Events399.exact102336RawTerms
def group : MergeGroup := .operator 103728 102336
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 103728) (leftOrdinal := 5)
    (rightResult := 102336) (rightOrdinal := 5) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6719⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6719⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6719⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge103756

namespace LeftMerge103757
def owner : Owner := ⟨.program ⟨214⟩, ⟨30067⟩⟩
def mergeEvent : Nat := 103757
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15741⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18612⟩⟩] } }
def leftRaw : List Term := Proof.Events405.exact103728RawTerms
def rightRaw : List Term := Proof.Events399.exact102336RawTerms
def group : MergeGroup := .operator 103728 102336
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 103728) (leftOrdinal := 23)
    (rightResult := 102336) (rightOrdinal := 22) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15741⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18612⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15741⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18612⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15741⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge103757

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
