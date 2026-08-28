import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge103455
def owner : Owner := ⟨.program ⟨214⟩, ⟨18641⟩⟩
def mergeEvent : Nat := 103455
def frameStart : Nat := 102927
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16098⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events404.exact103442RawTerms
def rightRaw : List Term := Proof.Events404.exact103440RawTerms
def group : MergeGroup := .operator 103442 103440
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 103442) (leftOrdinal := 0)
    (rightResult := 103440) (rightOrdinal := 7) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16098⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨16098⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge103455

namespace LeftMerge103456
def owner : Owner := ⟨.program ⟨214⟩, ⟨18641⟩⟩
def mergeEvent : Nat := 103456
def frameStart : Nat := 102927
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15979⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events404.exact103442RawTerms
def rightRaw : List Term := Proof.Events404.exact103440RawTerms
def group : MergeGroup := .operator 103442 103440
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 103442) (leftOrdinal := 0)
    (rightResult := 103440) (rightOrdinal := 6) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15979⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨15979⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge103456

namespace LeftMerge103457
def owner : Owner := ⟨.program ⟨214⟩, ⟨18641⟩⟩
def mergeEvent : Nat := 103457
def frameStart : Nat := 102927
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15860⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events404.exact103442RawTerms
def rightRaw : List Term := Proof.Events404.exact103440RawTerms
def group : MergeGroup := .operator 103442 103440
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 103442) (leftOrdinal := 0)
    (rightResult := 103440) (rightOrdinal := 5) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15860⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨15860⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge103457

namespace LeftMerge103458
def owner : Owner := ⟨.program ⟨214⟩, ⟨18641⟩⟩
def mergeEvent : Nat := 103458
def frameStart : Nat := 102927
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15741⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events404.exact103442RawTerms
def rightRaw : List Term := Proof.Events404.exact103440RawTerms
def group : MergeGroup := .operator 103442 103440
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 103442) (leftOrdinal := 0)
    (rightResult := 103440) (rightOrdinal := 4) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15741⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨15741⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge103458

namespace LeftMerge103459
def owner : Owner := ⟨.program ⟨214⟩, ⟨18641⟩⟩
def mergeEvent : Nat := 103459
def frameStart : Nat := 102927
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15622⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events404.exact103442RawTerms
def rightRaw : List Term := Proof.Events404.exact103440RawTerms
def group : MergeGroup := .operator 103442 103440
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 103442) (leftOrdinal := 0)
    (rightResult := 103440) (rightOrdinal := 3) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15622⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨15622⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge103459

namespace LeftMerge103460
def owner : Owner := ⟨.program ⟨214⟩, ⟨18641⟩⟩
def mergeEvent : Nat := 103460
def frameStart : Nat := 102927
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨17302⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events404.exact103442RawTerms
def rightRaw : List Term := Proof.Events404.exact103440RawTerms
def group : MergeGroup := .operator 103442 103440
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 103442) (leftOrdinal := 0)
    (rightResult := 103440) (rightOrdinal := 13) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17302⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨17302⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge103460

namespace LeftMerge103461
def owner : Owner := ⟨.program ⟨214⟩, ⟨18641⟩⟩
def mergeEvent : Nat := 103461
def frameStart : Nat := 102927
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15356⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events404.exact103442RawTerms
def rightRaw : List Term := Proof.Events404.exact103440RawTerms
def group : MergeGroup := .operator 103442 103440
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 103442) (leftOrdinal := 0)
    (rightResult := 103440) (rightOrdinal := 2) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15356⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨15356⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge103461

namespace LeftMerge103462
def owner : Owner := ⟨.program ⟨214⟩, ⟨18641⟩⟩
def mergeEvent : Nat := 103462
def frameStart : Nat := 102927
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15300⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events404.exact103442RawTerms
def rightRaw : List Term := Proof.Events404.exact103440RawTerms
def group : MergeGroup := .operator 103442 103440
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 103442) (leftOrdinal := 0)
    (rightResult := 103440) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15300⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨15300⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge103462

namespace LeftMerge103463
def owner : Owner := ⟨.program ⟨214⟩, ⟨18641⟩⟩
def mergeEvent : Nat := 103463
def frameStart : Nat := 102927
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15258⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events404.exact103442RawTerms
def rightRaw : List Term := Proof.Events404.exact103440RawTerms
def group : MergeGroup := .operator 103442 103440
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 103442) (leftOrdinal := 0)
    (rightResult := 103440) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15258⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨15258⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge103463

namespace LeftMerge103594
def owner : Owner := ⟨.program ⟨214⟩, ⟨18675⟩⟩
def mergeEvent : Nat := 103594
def frameStart : Nat := 102927
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6743⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩] } }
def leftRaw : List Term := Proof.Events404.exact103590RawTerms
def rightRaw : List Term := Proof.Events404.exact103431RawTerms
def group : MergeGroup := .operator 103590 103431
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 103590) (leftOrdinal := 17)
    (rightResult := 103431) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6743⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18674⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6743⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge103594

namespace LeftMerge103595
def owner : Owner := ⟨.program ⟨214⟩, ⟨18675⟩⟩
def mergeEvent : Nat := 103595
def frameStart : Nat := 102927
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6741⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩] } }
def leftRaw : List Term := Proof.Events404.exact103590RawTerms
def rightRaw : List Term := Proof.Events404.exact103431RawTerms
def group : MergeGroup := .operator 103590 103431
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 103590) (leftOrdinal := 16)
    (rightResult := 103431) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6741⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18674⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6741⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge103595

namespace LeftMerge103596
def owner : Owner := ⟨.program ⟨214⟩, ⟨18675⟩⟩
def mergeEvent : Nat := 103596
def frameStart : Nat := 102927
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6739⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩] } }
def leftRaw : List Term := Proof.Events404.exact103590RawTerms
def rightRaw : List Term := Proof.Events404.exact103431RawTerms
def group : MergeGroup := .operator 103590 103431
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 103590) (leftOrdinal := 15)
    (rightResult := 103431) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6739⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18674⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6739⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge103596

namespace LeftMerge103597
def owner : Owner := ⟨.program ⟨214⟩, ⟨18675⟩⟩
def mergeEvent : Nat := 103597
def frameStart : Nat := 102927
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6737⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩] } }
def leftRaw : List Term := Proof.Events404.exact103590RawTerms
def rightRaw : List Term := Proof.Events404.exact103431RawTerms
def group : MergeGroup := .operator 103590 103431
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 103590) (leftOrdinal := 14)
    (rightResult := 103431) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6737⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18674⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6737⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge103597

namespace LeftMerge103598
def owner : Owner := ⟨.program ⟨214⟩, ⟨18675⟩⟩
def mergeEvent : Nat := 103598
def frameStart : Nat := 102927
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6735⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩] } }
def leftRaw : List Term := Proof.Events404.exact103590RawTerms
def rightRaw : List Term := Proof.Events404.exact103431RawTerms
def group : MergeGroup := .operator 103590 103431
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 103590) (leftOrdinal := 13)
    (rightResult := 103431) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6735⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18674⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6735⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge103598

namespace LeftMerge103599
def owner : Owner := ⟨.program ⟨214⟩, ⟨18675⟩⟩
def mergeEvent : Nat := 103599
def frameStart : Nat := 102927
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6733⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩] } }
def leftRaw : List Term := Proof.Events404.exact103590RawTerms
def rightRaw : List Term := Proof.Events404.exact103431RawTerms
def group : MergeGroup := .operator 103590 103431
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 103590) (leftOrdinal := 12)
    (rightResult := 103431) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6733⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18674⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6733⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge103599

namespace LeftMerge103600
def owner : Owner := ⟨.program ⟨214⟩, ⟨18675⟩⟩
def mergeEvent : Nat := 103600
def frameStart : Nat := 102927
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6731⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩] } }
def leftRaw : List Term := Proof.Events404.exact103590RawTerms
def rightRaw : List Term := Proof.Events404.exact103431RawTerms
def group : MergeGroup := .operator 103590 103431
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 103590) (leftOrdinal := 11)
    (rightResult := 103431) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6731⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18674⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6731⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge103600

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
