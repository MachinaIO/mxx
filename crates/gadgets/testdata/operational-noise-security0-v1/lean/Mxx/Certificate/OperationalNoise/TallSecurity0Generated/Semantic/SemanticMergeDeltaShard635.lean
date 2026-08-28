import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge103601
def owner : Owner := ⟨.program ⟨214⟩, ⟨18675⟩⟩
def mergeEvent : Nat := 103601
def frameStart : Nat := 102927
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6729⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩] } }
def leftRaw : List Term := Proof.Events404.exact103590RawTerms
def rightRaw : List Term := Proof.Events404.exact103431RawTerms
def group : MergeGroup := .operator 103590 103431
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 103590) (leftOrdinal := 10)
    (rightResult := 103431) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6729⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18674⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6729⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge103601

namespace LeftMerge103602
def owner : Owner := ⟨.program ⟨214⟩, ⟨18675⟩⟩
def mergeEvent : Nat := 103602
def frameStart : Nat := 102927
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6727⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩] } }
def leftRaw : List Term := Proof.Events404.exact103590RawTerms
def rightRaw : List Term := Proof.Events404.exact103431RawTerms
def group : MergeGroup := .operator 103590 103431
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 103590) (leftOrdinal := 9)
    (rightResult := 103431) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6727⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18674⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6727⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge103602

namespace LeftMerge103603
def owner : Owner := ⟨.program ⟨214⟩, ⟨18675⟩⟩
def mergeEvent : Nat := 103603
def frameStart : Nat := 102927
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6725⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩] } }
def leftRaw : List Term := Proof.Events404.exact103590RawTerms
def rightRaw : List Term := Proof.Events404.exact103431RawTerms
def group : MergeGroup := .operator 103590 103431
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 103590) (leftOrdinal := 8)
    (rightResult := 103431) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6725⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18674⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6725⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge103603

namespace LeftMerge103604
def owner : Owner := ⟨.program ⟨214⟩, ⟨18675⟩⟩
def mergeEvent : Nat := 103604
def frameStart : Nat := 102927
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6723⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩] } }
def leftRaw : List Term := Proof.Events404.exact103590RawTerms
def rightRaw : List Term := Proof.Events404.exact103431RawTerms
def group : MergeGroup := .operator 103590 103431
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 103590) (leftOrdinal := 7)
    (rightResult := 103431) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6723⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18674⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6723⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge103604

namespace LeftMerge103605
def owner : Owner := ⟨.program ⟨214⟩, ⟨18675⟩⟩
def mergeEvent : Nat := 103605
def frameStart : Nat := 102927
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6721⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩] } }
def leftRaw : List Term := Proof.Events404.exact103590RawTerms
def rightRaw : List Term := Proof.Events404.exact103431RawTerms
def group : MergeGroup := .operator 103590 103431
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 103590) (leftOrdinal := 6)
    (rightResult := 103431) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6721⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18674⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6721⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge103605

namespace LeftMerge103606
def owner : Owner := ⟨.program ⟨214⟩, ⟨18675⟩⟩
def mergeEvent : Nat := 103606
def frameStart : Nat := 102927
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6719⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩] } }
def leftRaw : List Term := Proof.Events404.exact103590RawTerms
def rightRaw : List Term := Proof.Events404.exact103431RawTerms
def group : MergeGroup := .operator 103590 103431
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 103590) (leftOrdinal := 5)
    (rightResult := 103431) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6719⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18674⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6719⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge103606

namespace LeftMerge103607
def owner : Owner := ⟨.program ⟨214⟩, ⟨18675⟩⟩
def mergeEvent : Nat := 103607
def frameStart : Nat := 102927
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6717⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩] } }
def leftRaw : List Term := Proof.Events404.exact103590RawTerms
def rightRaw : List Term := Proof.Events404.exact103431RawTerms
def group : MergeGroup := .operator 103590 103431
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 103590) (leftOrdinal := 4)
    (rightResult := 103431) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6717⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18674⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6717⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge103607

namespace LeftMerge103608
def owner : Owner := ⟨.program ⟨214⟩, ⟨18675⟩⟩
def mergeEvent : Nat := 103608
def frameStart : Nat := 102927
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6715⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩] } }
def leftRaw : List Term := Proof.Events404.exact103590RawTerms
def rightRaw : List Term := Proof.Events404.exact103431RawTerms
def group : MergeGroup := .operator 103590 103431
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 103590) (leftOrdinal := 3)
    (rightResult := 103431) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6715⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18674⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6715⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge103608

namespace LeftMerge103609
def owner : Owner := ⟨.program ⟨214⟩, ⟨18675⟩⟩
def mergeEvent : Nat := 103609
def frameStart : Nat := 102927
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6713⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩] } }
def leftRaw : List Term := Proof.Events404.exact103590RawTerms
def rightRaw : List Term := Proof.Events404.exact103431RawTerms
def group : MergeGroup := .operator 103590 103431
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 103590) (leftOrdinal := 2)
    (rightResult := 103431) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6713⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18674⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6713⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge103609

namespace LeftMerge103610
def owner : Owner := ⟨.program ⟨214⟩, ⟨18675⟩⟩
def mergeEvent : Nat := 103610
def frameStart : Nat := 102927
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6711⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩] } }
def leftRaw : List Term := Proof.Events404.exact103590RawTerms
def rightRaw : List Term := Proof.Events404.exact103431RawTerms
def group : MergeGroup := .operator 103590 103431
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 103590) (leftOrdinal := 1)
    (rightResult := 103431) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6711⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18674⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6711⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge103610

namespace LeftMerge103611
def owner : Owner := ⟨.program ⟨214⟩, ⟨18675⟩⟩
def mergeEvent : Nat := 103611
def frameStart : Nat := 102927
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6709⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩] } }
def leftRaw : List Term := Proof.Events404.exact103590RawTerms
def rightRaw : List Term := Proof.Events404.exact103431RawTerms
def group : MergeGroup := .operator 103590 103431
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 103590) (leftOrdinal := 0)
    (rightResult := 103431) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6709⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18674⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6709⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge103611

namespace LeftMerge103612
def owner : Owner := ⟨.program ⟨214⟩, ⟨18675⟩⟩
def mergeEvent : Nat := 103612
def frameStart : Nat := 102927
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨18163⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩] } }
def leftRaw : List Term := Proof.Events404.exact103590RawTerms
def rightRaw : List Term := Proof.Events404.exact103431RawTerms
def group : MergeGroup := .operator 103590 103431
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 103590) (leftOrdinal := 33)
    (rightResult := 103431) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18163⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18674⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨18163⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge103612

namespace LeftMerge103614
def owner : Owner := ⟨.program ⟨214⟩, ⟨18675⟩⟩
def mergeEvent : Nat := 103614
def frameStart : Nat := 102927
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨18163⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18612⟩⟩] } }
def rhsRaw : List Term := Proof.Events404.exact103428RawTerms
def group : MergeGroup := .relation 103613
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 103613) (rhsResult := 103428)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨18163⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18674⟩⟩) ⟨18612⟩ 103428) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18612⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨18163⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge103614

namespace LeftMerge103615
def owner : Owner := ⟨.program ⟨214⟩, ⟨18675⟩⟩
def mergeEvent : Nat := 103615
def frameStart : Nat := 102927
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨17078⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩] } }
def leftRaw : List Term := Proof.Events404.exact103590RawTerms
def rightRaw : List Term := Proof.Events404.exact103431RawTerms
def group : MergeGroup := .operator 103590 103431
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 103590) (leftOrdinal := 29)
    (rightResult := 103431) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17078⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18674⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨17078⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge103615

namespace LeftMerge103617
def owner : Owner := ⟨.program ⟨214⟩, ⟨18675⟩⟩
def mergeEvent : Nat := 103617
def frameStart : Nat := 102927
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨17078⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18612⟩⟩] } }
def rhsRaw : List Term := Proof.Events404.exact103428RawTerms
def group : MergeGroup := .relation 103616
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 103616) (rhsResult := 103428)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨17078⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18674⟩⟩) ⟨18612⟩ 103428) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18612⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨17078⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge103617

namespace LeftMerge103618
def owner : Owner := ⟨.program ⟨214⟩, ⟨18675⟩⟩
def mergeEvent : Nat := 103618
def frameStart : Nat := 102927
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16791⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩] } }
def leftRaw : List Term := Proof.Events404.exact103590RawTerms
def rightRaw : List Term := Proof.Events404.exact103431RawTerms
def group : MergeGroup := .operator 103590 103431
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 103590) (leftOrdinal := 28)
    (rightResult := 103431) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16791⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18674⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨16791⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge103618

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
