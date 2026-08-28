import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge141681
def owner : Owner := ⟨.program ⟨257⟩, ⟨7794⟩⟩
def mergeEvent : Nat := 141681
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7286⟩⟩] } }
def leftRaw : List Term := Proof.Events524.exact134273RawTerms
def rightRaw : List Term := Proof.Events096.exact24636RawTerms
def group : MergeGroup := .operator 134273 24636
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 134273) (leftOrdinal := 0)
    (rightResult := 24636) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7286⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7286⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge141681

namespace LeftMerge141698
def owner : Owner := ⟨.program ⟨257⟩, ⟨21000⟩⟩
def mergeEvent : Nat := 141698
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨20996⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩] } }
def leftRaw : List Term := Proof.Events553.exact141692RawTerms
def rightRaw : List Term := Proof.Events096.exact24625RawTerms
def group : MergeGroup := .operator 141692 24625
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 141692) (leftOrdinal := 1)
    (rightResult := 24625) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨20996⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9574⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨20996⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge141698

namespace LeftMerge141700
def owner : Owner := ⟨.program ⟨257⟩, ⟨21000⟩⟩
def mergeEvent : Nat := 141700
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨20996⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7306⟩⟩] } }
def rhsRaw : List Term := Proof.Events096.exact24595RawTerms
def group : MergeGroup := .relation 141699
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 141699) (rhsResult := 24595)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨20996⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨9574⟩⟩) ⟨7306⟩ 24595) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7306⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨20996⟩⟩], [⟨.program ⟨257⟩, ⟨7306⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge141700

namespace LeftMerge141701
def owner : Owner := ⟨.program ⟨257⟩, ⟨21000⟩⟩
def mergeEvent : Nat := 141701
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩] } }
def leftRaw : List Term := Proof.Events553.exact141692RawTerms
def rightRaw : List Term := Proof.Events096.exact24625RawTerms
def group : MergeGroup := .operator 141692 24625
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 141692) (leftOrdinal := 0)
    (rightResult := 24625) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7286⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9574⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge141701

namespace LeftMerge141706
def owner : Owner := ⟨.program ⟨257⟩, ⟨21333⟩⟩
def mergeEvent : Nat := 141706
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨20996⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7306⟩⟩] } }
def leftRaw : List Term := Proof.Events553.exact141702RawTerms
def rightRaw : List Term := Proof.Events553.exact141672RawTerms
def group : MergeGroup := .operator 141702 141672
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 141702) (leftOrdinal := 1)
    (rightResult := 141672) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨20996⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7306⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨20996⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7306⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨20996⟩⟩], [⟨.program ⟨257⟩, ⟨7306⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge141706

namespace LeftMerge141714
def owner : Owner := ⟨.program ⟨257⟩, ⟨23363⟩⟩
def mergeEvent : Nat := 141714
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨20996⟩⟩, ⟨.program ⟨257⟩, ⟨21326⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨23362⟩⟩] } }
def leftRaw : List Term := Proof.Events553.exact141708RawTerms
def rightRaw : List Term := Proof.Events553.exact141644RawTerms
def group : MergeGroup := .operator 141708 141644
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 141708) (leftOrdinal := 1)
    (rightResult := 141644) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨20996⟩⟩, ⟨.program ⟨257⟩, ⟨21326⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨23362⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨20996⟩⟩, ⟨.program ⟨257⟩, ⟨21326⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨23362⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge141714

namespace LeftMerge141716
def owner : Owner := ⟨.program ⟨257⟩, ⟨23363⟩⟩
def mergeEvent : Nat := 141716
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨20996⟩⟩, ⟨.program ⟨257⟩, ⟨21326⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨22887⟩⟩] } }
def rhsRaw : List Term := Proof.Events553.exact141641RawTerms
def group : MergeGroup := .relation 141715
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 141715) (rhsResult := 141641)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨20996⟩⟩, ⟨.program ⟨257⟩, ⟨21326⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨23362⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨23362⟩⟩) ⟨22887⟩ 141641) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨22887⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨20996⟩⟩, ⟨.program ⟨257⟩, ⟨21326⟩⟩], [⟨.program ⟨257⟩, ⟨22887⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge141716

namespace LeftMerge141717
def owner : Owner := ⟨.program ⟨257⟩, ⟨23363⟩⟩
def mergeEvent : Nat := 141717
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23362⟩⟩] } }
def leftRaw : List Term := Proof.Events553.exact141708RawTerms
def rightRaw : List Term := Proof.Events553.exact141644RawTerms
def group : MergeGroup := .operator 141708 141644
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 141708) (leftOrdinal := 0)
    (rightResult := 141644) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨23362⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23362⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge141717

namespace LeftMerge141731
def owner : Owner := ⟨.program ⟨257⟩, ⟨22302⟩⟩
def mergeEvent : Nat := 141731
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22299⟩⟩] } }
def leftRaw : List Term := Proof.Events525.exact134495RawTerms
def rightRaw : List Term := Proof.Events553.exact141725RawTerms
def group : MergeGroup := .operator 134495 141725
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 134495) (leftOrdinal := 0)
    (rightResult := 141725) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨22299⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22299⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge141731

namespace LeftMerge141810
def owner : Owner := ⟨.program ⟨257⟩, ⟨21327⟩⟩
def mergeEvent : Nat := 141810
def frameStart : Nat := 141780
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨20996⟩⟩, ⟨.program ⟨257⟩, ⟨21326⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events553.exact141806RawTerms
def rightRaw : List Term := Proof.Events553.exact141803RawTerms
def group : MergeGroup := .operator 141806 141803
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 141806) (leftOrdinal := 0)
    (rightResult := 141803) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨20996⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨21326⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨20996⟩⟩, ⟨.program ⟨257⟩, ⟨21326⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge141810

namespace LeftMerge141840
def owner : Owner := ⟨.program ⟨257⟩, ⟨23180⟩⟩
def mergeEvent : Nat := 141840
def frameStart : Nat := 141780
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨20996⟩⟩, ⟨.program ⟨257⟩, ⟨21326⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events554.exact141836RawTerms
def rightRaw : List Term := Proof.Events554.exact141834RawTerms
def group : MergeGroup := .operator 141836 141834
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 141836) (leftOrdinal := 0)
    (rightResult := 141834) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨20996⟩⟩, ⟨.program ⟨257⟩, ⟨21326⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨20996⟩⟩, ⟨.program ⟨257⟩, ⟨21326⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge141840

namespace LeftMerge141863
def owner : Owner := ⟨.program ⟨257⟩, ⟨9576⟩⟩
def mergeEvent : Nat := 141863
def frameStart : Nat := 141780
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩] } }
def leftRaw : List Term := Proof.Events554.exact141859RawTerms
def rightRaw : List Term := Proof.Events554.exact141856RawTerms
def group : MergeGroup := .operator 141859 141856
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 141859) (leftOrdinal := 0)
    (rightResult := 141856) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7286⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9574⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge141863

namespace LeftMerge141872
def owner : Owner := ⟨.program ⟨257⟩, ⟨23365⟩⟩
def mergeEvent : Nat := 141872
def frameStart : Nat := 141780
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23362⟩⟩] } }
def leftRaw : List Term := Proof.Events554.exact141868RawTerms
def rightRaw : List Term := Proof.Events554.exact141825RawTerms
def group : MergeGroup := .operator 141868 141825
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 141868) (leftOrdinal := 0)
    (rightResult := 141825) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨23362⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23362⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge141872

namespace LeftMerge141873
def owner : Owner := ⟨.program ⟨257⟩, ⟨23365⟩⟩
def mergeEvent : Nat := 141873
def frameStart : Nat := 141780
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨20996⟩⟩, ⟨.program ⟨257⟩, ⟨21326⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨23362⟩⟩] } }
def leftRaw : List Term := Proof.Events554.exact141868RawTerms
def rightRaw : List Term := Proof.Events554.exact141825RawTerms
def group : MergeGroup := .operator 141868 141825
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 141868) (leftOrdinal := 1)
    (rightResult := 141825) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨20996⟩⟩, ⟨.program ⟨257⟩, ⟨21326⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨23362⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨20996⟩⟩, ⟨.program ⟨257⟩, ⟨21326⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨23362⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge141873

namespace LeftMerge141875
def owner : Owner := ⟨.program ⟨257⟩, ⟨23365⟩⟩
def mergeEvent : Nat := 141875
def frameStart : Nat := 141780
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨20996⟩⟩, ⟨.program ⟨257⟩, ⟨21326⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨22887⟩⟩] } }
def rhsRaw : List Term := Proof.Events553.exact141822RawTerms
def group : MergeGroup := .relation 141874
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 141874) (rhsResult := 141822)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨20996⟩⟩, ⟨.program ⟨257⟩, ⟨21326⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨23362⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨23362⟩⟩) ⟨22887⟩ 141822) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨22887⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨20996⟩⟩, ⟨.program ⟨257⟩, ⟨21326⟩⟩], [⟨.program ⟨257⟩, ⟨22887⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge141875

namespace LeftMerge141883
def owner : Owner := ⟨.program ⟨257⟩, ⟨21754⟩⟩
def mergeEvent : Nat := 141883
def frameStart : Nat := 141780
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨21752⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events554.exact141836RawTerms
def rightRaw : List Term := Proof.Events554.exact141879RawTerms
def group : MergeGroup := .operator 141836 141879
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 141836) (leftOrdinal := 0)
    (rightResult := 141879) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨21752⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨21752⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge141883

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
