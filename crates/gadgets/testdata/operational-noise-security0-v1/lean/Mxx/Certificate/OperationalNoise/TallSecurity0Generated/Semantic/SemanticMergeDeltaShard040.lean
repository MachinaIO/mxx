import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge7519
def owner : Owner := ⟨.program ⟨214⟩, ⟨7376⟩⟩
def mergeEvent : Nat := 7519
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6768⟩⟩] } }
def leftRaw : List Term := Proof.Events024.exact6314RawTerms
def rightRaw : List Term := Proof.Events029.exact7515RawTerms
def group : MergeGroup := .operator 6314 7515
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 6314) (leftOrdinal := 0)
    (rightResult := 7515) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6768⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6768⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge7519

namespace LeftMerge7536
def owner : Owner := ⟨.program ⟨214⟩, ⟨10159⟩⟩
def mergeEvent : Nat := 7536
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨10155⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7876⟩⟩] } }
def leftRaw : List Term := Proof.Events029.exact7530RawTerms
def rightRaw : List Term := Proof.Events029.exact7504RawTerms
def group : MergeGroup := .operator 7530 7504
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 7530) (leftOrdinal := 1)
    (rightResult := 7504) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨10155⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨7876⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨10155⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7876⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge7536

namespace LeftMerge7538
def owner : Owner := ⟨.program ⟨214⟩, ⟨10159⟩⟩
def mergeEvent : Nat := 7538
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨10155⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6788⟩⟩] } }
def rhsRaw : List Term := Proof.Events029.exact7474RawTerms
def group : MergeGroup := .relation 7537
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 7537) (rhsResult := 7474)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨10155⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7876⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨7876⟩⟩) ⟨6788⟩ 7474) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6788⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨10155⟩⟩], [⟨.program ⟨214⟩, ⟨6788⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge7538

namespace LeftMerge7539
def owner : Owner := ⟨.program ⟨214⟩, ⟨10159⟩⟩
def mergeEvent : Nat := 7539
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6768⟩⟩, ⟨.program ⟨214⟩, ⟨7876⟩⟩] } }
def leftRaw : List Term := Proof.Events029.exact7530RawTerms
def rightRaw : List Term := Proof.Events029.exact7504RawTerms
def group : MergeGroup := .operator 7530 7504
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 7530) (leftOrdinal := 0)
    (rightResult := 7504) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6768⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨7876⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6768⟩⟩, ⟨.program ⟨214⟩, ⟨7876⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge7539

namespace LeftMerge7544
def owner : Owner := ⟨.program ⟨214⟩, ⟨12997⟩⟩
def mergeEvent : Nat := 7544
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨10155⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6788⟩⟩] } }
def leftRaw : List Term := Proof.Events029.exact7540RawTerms
def rightRaw : List Term := Proof.Events029.exact7497RawTerms
def group : MergeGroup := .operator 7540 7497
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 7540) (leftOrdinal := 1)
    (rightResult := 7497) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨10155⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6788⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨10155⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6788⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨10155⟩⟩], [⟨.program ⟨214⟩, ⟨6788⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge7544

namespace LeftMerge7552
def owner : Owner := ⟨.program ⟨214⟩, ⟨25625⟩⟩
def mergeEvent : Nat := 7552
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨10155⟩⟩, ⟨.program ⟨214⟩, ⟨12990⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25624⟩⟩] } }
def leftRaw : List Term := Proof.Events029.exact7546RawTerms
def rightRaw : List Term := Proof.Events029.exact7463RawTerms
def group : MergeGroup := .operator 7546 7463
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 7546) (leftOrdinal := 1)
    (rightResult := 7463) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨10155⟩⟩, ⟨.program ⟨214⟩, ⟨12990⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25624⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨10155⟩⟩, ⟨.program ⟨214⟩, ⟨12990⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25624⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge7552

namespace LeftMerge7554
def owner : Owner := ⟨.program ⟨214⟩, ⟨25625⟩⟩
def mergeEvent : Nat := 7554
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨10155⟩⟩, ⟨.program ⟨214⟩, ⟨12990⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23340⟩⟩] } }
def rhsRaw : List Term := Proof.Events029.exact7460RawTerms
def group : MergeGroup := .relation 7553
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 7553) (rhsResult := 7460)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨10155⟩⟩, ⟨.program ⟨214⟩, ⟨12990⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25624⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨25624⟩⟩) ⟨23340⟩ 7460) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23340⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨10155⟩⟩, ⟨.program ⟨214⟩, ⟨12990⟩⟩], [⟨.program ⟨214⟩, ⟨23340⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge7554

namespace LeftMerge7555
def owner : Owner := ⟨.program ⟨214⟩, ⟨25625⟩⟩
def mergeEvent : Nat := 7555
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6768⟩⟩, ⟨.program ⟨214⟩, ⟨7876⟩⟩, ⟨.program ⟨214⟩, ⟨25624⟩⟩] } }
def leftRaw : List Term := Proof.Events029.exact7546RawTerms
def rightRaw : List Term := Proof.Events029.exact7463RawTerms
def group : MergeGroup := .operator 7546 7463
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 7546) (leftOrdinal := 0)
    (rightResult := 7463) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6768⟩⟩, ⟨.program ⟨214⟩, ⟨7876⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25624⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6768⟩⟩, ⟨.program ⟨214⟩, ⟨7876⟩⟩, ⟨.program ⟨214⟩, ⟨25624⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge7555

namespace LeftMerge7569
def owner : Owner := ⟨.program ⟨214⟩, ⟨20123⟩⟩
def mergeEvent : Nat := 7569
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20120⟩⟩] } }
def leftRaw : List Term := Proof.Events025.exact6561RawTerms
def rightRaw : List Term := Proof.Events029.exact7563RawTerms
def group : MergeGroup := .operator 6561 7563
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 6561) (leftOrdinal := 0)
    (rightResult := 7563) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨20120⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20120⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge7569

namespace LeftMerge7648
def owner : Owner := ⟨.program ⟨214⟩, ⟨12991⟩⟩
def mergeEvent : Nat := 7648
def frameStart : Nat := 7618
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨10155⟩⟩, ⟨.program ⟨214⟩, ⟨12990⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events029.exact7644RawTerms
def rightRaw : List Term := Proof.Events029.exact7641RawTerms
def group : MergeGroup := .operator 7644 7641
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 7644) (leftOrdinal := 0)
    (rightResult := 7641) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10155⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨12990⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨10155⟩⟩, ⟨.program ⟨214⟩, ⟨12990⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge7648

namespace LeftMerge7678
def owner : Owner := ⟨.program ⟨214⟩, ⟨13072⟩⟩
def mergeEvent : Nat := 7678
def frameStart : Nat := 7618
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨10155⟩⟩, ⟨.program ⟨214⟩, ⟨12990⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events029.exact7674RawTerms
def rightRaw : List Term := Proof.Events029.exact7672RawTerms
def group : MergeGroup := .operator 7674 7672
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 7674) (leftOrdinal := 0)
    (rightResult := 7672) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10155⟩⟩, ⟨.program ⟨214⟩, ⟨12990⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨10155⟩⟩, ⟨.program ⟨214⟩, ⟨12990⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge7678

namespace LeftMerge7701
def owner : Owner := ⟨.program ⟨214⟩, ⟨7878⟩⟩
def mergeEvent : Nat := 7701
def frameStart : Nat := 7618
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6768⟩⟩, ⟨.program ⟨214⟩, ⟨7876⟩⟩] } }
def leftRaw : List Term := Proof.Events030.exact7697RawTerms
def rightRaw : List Term := Proof.Events030.exact7694RawTerms
def group : MergeGroup := .operator 7697 7694
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 7697) (leftOrdinal := 0)
    (rightResult := 7694) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6768⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨7876⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6768⟩⟩, ⟨.program ⟨214⟩, ⟨7876⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge7701

namespace LeftMerge7710
def owner : Owner := ⟨.program ⟨214⟩, ⟨25627⟩⟩
def mergeEvent : Nat := 7710
def frameStart : Nat := 7618
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨10155⟩⟩, ⟨.program ⟨214⟩, ⟨12990⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25624⟩⟩] } }
def leftRaw : List Term := Proof.Events030.exact7706RawTerms
def rightRaw : List Term := Proof.Events029.exact7663RawTerms
def group : MergeGroup := .operator 7706 7663
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 7706) (leftOrdinal := 1)
    (rightResult := 7663) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10155⟩⟩, ⟨.program ⟨214⟩, ⟨12990⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25624⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨10155⟩⟩, ⟨.program ⟨214⟩, ⟨12990⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25624⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge7710

namespace LeftMerge7712
def owner : Owner := ⟨.program ⟨214⟩, ⟨25627⟩⟩
def mergeEvent : Nat := 7712
def frameStart : Nat := 7618
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨10155⟩⟩, ⟨.program ⟨214⟩, ⟨12990⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23340⟩⟩] } }
def rhsRaw : List Term := Proof.Events029.exact7660RawTerms
def group : MergeGroup := .relation 7711
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 7711) (rhsResult := 7660)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨10155⟩⟩, ⟨.program ⟨214⟩, ⟨12990⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25624⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨25624⟩⟩) ⟨23340⟩ 7660) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23340⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨10155⟩⟩, ⟨.program ⟨214⟩, ⟨12990⟩⟩], [⟨.program ⟨214⟩, ⟨23340⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge7712

namespace LeftMerge7713
def owner : Owner := ⟨.program ⟨214⟩, ⟨25627⟩⟩
def mergeEvent : Nat := 7713
def frameStart : Nat := 7618
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6768⟩⟩, ⟨.program ⟨214⟩, ⟨7876⟩⟩, ⟨.program ⟨214⟩, ⟨25624⟩⟩] } }
def leftRaw : List Term := Proof.Events030.exact7706RawTerms
def rightRaw : List Term := Proof.Events029.exact7663RawTerms
def group : MergeGroup := .operator 7706 7663
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 7706) (leftOrdinal := 0)
    (rightResult := 7663) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6768⟩⟩, ⟨.program ⟨214⟩, ⟨7876⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25624⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6768⟩⟩, ⟨.program ⟨214⟩, ⟨7876⟩⟩, ⟨.program ⟨214⟩, ⟨25624⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge7713

namespace LeftMerge7721
def owner : Owner := ⟨.program ⟨214⟩, ⟨16770⟩⟩
def mergeEvent : Nat := 7721
def frameStart : Nat := 7618
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16768⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events029.exact7674RawTerms
def rightRaw : List Term := Proof.Events030.exact7717RawTerms
def group : MergeGroup := .operator 7674 7717
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 7674) (leftOrdinal := 0)
    (rightResult := 7717) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16768⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨16768⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge7721

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
