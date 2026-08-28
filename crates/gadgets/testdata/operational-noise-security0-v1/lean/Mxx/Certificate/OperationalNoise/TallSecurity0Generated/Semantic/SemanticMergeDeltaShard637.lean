import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge103644
def owner : Owner := ⟨.program ⟨214⟩, ⟨18675⟩⟩
def mergeEvent : Nat := 103644
def frameStart : Nat := 102927
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15979⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18612⟩⟩] } }
def rhsRaw : List Term := Proof.Events404.exact103428RawTerms
def group : MergeGroup := .relation 103643
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 103643) (rhsResult := 103428)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨15979⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18674⟩⟩) ⟨18612⟩ 103428) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18612⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨15979⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge103644

namespace LeftMerge103645
def owner : Owner := ⟨.program ⟨214⟩, ⟨18675⟩⟩
def mergeEvent : Nat := 103645
def frameStart : Nat := 102927
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15860⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩] } }
def leftRaw : List Term := Proof.Events404.exact103590RawTerms
def rightRaw : List Term := Proof.Events404.exact103431RawTerms
def group : MergeGroup := .operator 103590 103431
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 103590) (leftOrdinal := 23)
    (rightResult := 103431) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15860⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18674⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨15860⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge103645

namespace LeftMerge103647
def owner : Owner := ⟨.program ⟨214⟩, ⟨18675⟩⟩
def mergeEvent : Nat := 103647
def frameStart : Nat := 102927
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15860⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18612⟩⟩] } }
def rhsRaw : List Term := Proof.Events404.exact103428RawTerms
def group : MergeGroup := .relation 103646
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 103646) (rhsResult := 103428)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨15860⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18674⟩⟩) ⟨18612⟩ 103428) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18612⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨15860⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge103647

namespace LeftMerge103648
def owner : Owner := ⟨.program ⟨214⟩, ⟨18675⟩⟩
def mergeEvent : Nat := 103648
def frameStart : Nat := 102927
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15741⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩] } }
def leftRaw : List Term := Proof.Events404.exact103590RawTerms
def rightRaw : List Term := Proof.Events404.exact103431RawTerms
def group : MergeGroup := .operator 103590 103431
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 103590) (leftOrdinal := 22)
    (rightResult := 103431) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15741⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18674⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨15741⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge103648

namespace LeftMerge103650
def owner : Owner := ⟨.program ⟨214⟩, ⟨18675⟩⟩
def mergeEvent : Nat := 103650
def frameStart : Nat := 102927
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15741⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18612⟩⟩] } }
def rhsRaw : List Term := Proof.Events404.exact103428RawTerms
def group : MergeGroup := .relation 103649
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 103649) (rhsResult := 103428)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨15741⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18674⟩⟩) ⟨18612⟩ 103428) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18612⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨15741⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge103650

namespace LeftMerge103651
def owner : Owner := ⟨.program ⟨214⟩, ⟨18675⟩⟩
def mergeEvent : Nat := 103651
def frameStart : Nat := 102927
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15622⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩] } }
def leftRaw : List Term := Proof.Events404.exact103590RawTerms
def rightRaw : List Term := Proof.Events404.exact103431RawTerms
def group : MergeGroup := .operator 103590 103431
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 103590) (leftOrdinal := 21)
    (rightResult := 103431) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15622⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18674⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨15622⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge103651

namespace LeftMerge103653
def owner : Owner := ⟨.program ⟨214⟩, ⟨18675⟩⟩
def mergeEvent : Nat := 103653
def frameStart : Nat := 102927
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15622⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18612⟩⟩] } }
def rhsRaw : List Term := Proof.Events404.exact103428RawTerms
def group : MergeGroup := .relation 103652
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 103652) (rhsResult := 103428)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨15622⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18674⟩⟩) ⟨18612⟩ 103428) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18612⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨15622⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge103653

namespace LeftMerge103654
def owner : Owner := ⟨.program ⟨214⟩, ⟨18675⟩⟩
def mergeEvent : Nat := 103654
def frameStart : Nat := 102927
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨17302⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩] } }
def leftRaw : List Term := Proof.Events404.exact103590RawTerms
def rightRaw : List Term := Proof.Events404.exact103431RawTerms
def group : MergeGroup := .operator 103590 103431
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 103590) (leftOrdinal := 31)
    (rightResult := 103431) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17302⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18674⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨17302⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge103654

namespace LeftMerge103656
def owner : Owner := ⟨.program ⟨214⟩, ⟨18675⟩⟩
def mergeEvent : Nat := 103656
def frameStart : Nat := 102927
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨17302⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18612⟩⟩] } }
def rhsRaw : List Term := Proof.Events404.exact103428RawTerms
def group : MergeGroup := .relation 103655
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 103655) (rhsResult := 103428)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨17302⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18674⟩⟩) ⟨18612⟩ 103428) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18612⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨17302⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge103656

namespace LeftMerge103657
def owner : Owner := ⟨.program ⟨214⟩, ⟨18675⟩⟩
def mergeEvent : Nat := 103657
def frameStart : Nat := 102927
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15356⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩] } }
def leftRaw : List Term := Proof.Events404.exact103590RawTerms
def rightRaw : List Term := Proof.Events404.exact103431RawTerms
def group : MergeGroup := .operator 103590 103431
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 103590) (leftOrdinal := 20)
    (rightResult := 103431) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15356⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18674⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨15356⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge103657

namespace LeftMerge103659
def owner : Owner := ⟨.program ⟨214⟩, ⟨18675⟩⟩
def mergeEvent : Nat := 103659
def frameStart : Nat := 102927
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15356⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18612⟩⟩] } }
def rhsRaw : List Term := Proof.Events404.exact103428RawTerms
def group : MergeGroup := .relation 103658
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 103658) (rhsResult := 103428)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨15356⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18674⟩⟩) ⟨18612⟩ 103428) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18612⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨15356⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge103659

namespace LeftMerge103660
def owner : Owner := ⟨.program ⟨214⟩, ⟨18675⟩⟩
def mergeEvent : Nat := 103660
def frameStart : Nat := 102927
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15300⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩] } }
def leftRaw : List Term := Proof.Events404.exact103590RawTerms
def rightRaw : List Term := Proof.Events404.exact103431RawTerms
def group : MergeGroup := .operator 103590 103431
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 103590) (leftOrdinal := 19)
    (rightResult := 103431) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15300⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18674⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨15300⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge103660

namespace LeftMerge103662
def owner : Owner := ⟨.program ⟨214⟩, ⟨18675⟩⟩
def mergeEvent : Nat := 103662
def frameStart : Nat := 102927
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15300⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18612⟩⟩] } }
def rhsRaw : List Term := Proof.Events404.exact103428RawTerms
def group : MergeGroup := .relation 103661
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 103661) (rhsResult := 103428)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨15300⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18674⟩⟩) ⟨18612⟩ 103428) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18612⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨15300⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge103662

namespace LeftMerge103663
def owner : Owner := ⟨.program ⟨214⟩, ⟨18675⟩⟩
def mergeEvent : Nat := 103663
def frameStart : Nat := 102927
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15258⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩] } }
def leftRaw : List Term := Proof.Events404.exact103590RawTerms
def rightRaw : List Term := Proof.Events404.exact103431RawTerms
def group : MergeGroup := .operator 103590 103431
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 103590) (leftOrdinal := 18)
    (rightResult := 103431) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15258⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18674⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨15258⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge103663

namespace LeftMerge103665
def owner : Owner := ⟨.program ⟨214⟩, ⟨18675⟩⟩
def mergeEvent : Nat := 103665
def frameStart : Nat := 102927
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15258⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18612⟩⟩] } }
def rhsRaw : List Term := Proof.Events404.exact103428RawTerms
def group : MergeGroup := .relation 103664
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 103664) (rhsResult := 103428)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨15258⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18674⟩⟩) ⟨18612⟩ 103428) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18612⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨15258⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge103665

namespace LeftMerge103673
def owner : Owner := ⟨.program ⟨214⟩, ⟨18487⟩⟩
def mergeEvent : Nat := 103673
def frameStart : Nat := 102927
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨18485⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events404.exact103442RawTerms
def rightRaw : List Term := Proof.Events404.exact103669RawTerms
def group : MergeGroup := .operator 103442 103669
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 103442) (leftOrdinal := 0)
    (rightResult := 103669) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18485⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨18485⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge103673

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
