import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge141900
def owner : Owner := ⟨.program ⟨257⟩, ⟨22302⟩⟩
def mergeEvent : Nat := 141900
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7181⟩⟩] } }
def rhsRaw : List Term := Proof.Events554.exact141897RawTerms
def group : MergeGroup := .relation 141899
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 141899) (rhsResult := 141897)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22299⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 141898 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22299⟩⟩]⟩) (none) 141897) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7181⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7181⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge141900

namespace LeftMerge141901
def owner : Owner := ⟨.program ⟨257⟩, ⟨22302⟩⟩
def mergeEvent : Nat := 141901
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23362⟩⟩] } }
def rhsRaw : List Term := Proof.Events554.exact141897RawTerms
def group : MergeGroup := .relation 141899
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 141899) (rhsResult := 141897)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22299⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 141898 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22299⟩⟩]⟩) (none) 141897) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23362⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23362⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge141901

namespace LeftMerge141902
def owner : Owner := ⟨.program ⟨257⟩, ⟨22302⟩⟩
def mergeEvent : Nat := 141902
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨20996⟩⟩, ⟨.program ⟨257⟩, ⟨21326⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨22887⟩⟩] } }
def rhsRaw : List Term := Proof.Events554.exact141897RawTerms
def group : MergeGroup := .relation 141899
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 141899) (rhsResult := 141897)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22299⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 141898 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22299⟩⟩]⟩) (none) 141897) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨20996⟩⟩, ⟨.program ⟨257⟩, ⟨21326⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨22887⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨20996⟩⟩, ⟨.program ⟨257⟩, ⟨21326⟩⟩], [⟨.program ⟨257⟩, ⟨22887⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge141902

namespace LeftMerge141903
def owner : Owner := ⟨.program ⟨257⟩, ⟨22302⟩⟩
def mergeEvent : Nat := 141903
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨21752⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events554.exact141897RawTerms
def group : MergeGroup := .relation 141899
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 141899) (rhsResult := 141897)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22299⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 141898 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22299⟩⟩]⟩) (none) 141897) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨21752⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨21752⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge141903

namespace LeftMerge141908
def owner : Owner := ⟨.program ⟨257⟩, ⟨23364⟩⟩
def mergeEvent : Nat := 141908
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨20996⟩⟩, ⟨.program ⟨257⟩, ⟨21326⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨22887⟩⟩] } }
def leftRaw : List Term := Proof.Events554.exact141904RawTerms
def rightRaw : List Term := Proof.Events553.exact141718RawTerms
def group : MergeGroup := .operator 141904 141718
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 141904) (leftOrdinal := 2)
    (rightResult := 141718) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨20996⟩⟩, ⟨.program ⟨257⟩, ⟨21326⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨22887⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨20996⟩⟩, ⟨.program ⟨257⟩, ⟨21326⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨22887⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨20996⟩⟩, ⟨.program ⟨257⟩, ⟨21326⟩⟩], [⟨.program ⟨257⟩, ⟨22887⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge141908

namespace LeftMerge141909
def owner : Owner := ⟨.program ⟨257⟩, ⟨23364⟩⟩
def mergeEvent : Nat := 141909
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23362⟩⟩] } }
def leftRaw : List Term := Proof.Events554.exact141904RawTerms
def rightRaw : List Term := Proof.Events553.exact141718RawTerms
def group : MergeGroup := .operator 141904 141718
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 141904) (leftOrdinal := 1)
    (rightResult := 141718) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23362⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23362⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23362⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge141909

namespace LeftMerge141917
def owner : Owner := ⟨.program ⟨257⟩, ⟨23657⟩⟩
def mergeEvent : Nat := 141917
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7181⟩⟩, ⟨.program ⟨257⟩, ⟨23655⟩⟩] } }
def leftRaw : List Term := Proof.Events554.exact141911RawTerms
def rightRaw : List Term := Proof.Events553.exact141634RawTerms
def group : MergeGroup := .operator 141911 141634
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 141911) (leftOrdinal := 0)
    (rightResult := 141634) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7181⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨23655⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7181⟩⟩, ⟨.program ⟨257⟩, ⟨23655⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge141917

namespace LeftMerge141918
def owner : Owner := ⟨.program ⟨257⟩, ⟨23657⟩⟩
def mergeEvent : Nat := 141918
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨21752⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨23655⟩⟩] } }
def leftRaw : List Term := Proof.Events554.exact141911RawTerms
def rightRaw : List Term := Proof.Events553.exact141634RawTerms
def group : MergeGroup := .operator 141911 141634
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 141911) (leftOrdinal := 1)
    (rightResult := 141634) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨21752⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨23655⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨21752⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨23655⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge141918

namespace LeftMerge141920
def owner : Owner := ⟨.program ⟨257⟩, ⟨23657⟩⟩
def mergeEvent : Nat := 141920
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨21752⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨23018⟩⟩] } }
def rhsRaw : List Term := Proof.Events553.exact141631RawTerms
def group : MergeGroup := .relation 141919
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 141919) (rhsResult := 141631)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨21752⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨23655⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨23655⟩⟩) ⟨23018⟩ 141631) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨23018⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨21752⟩⟩], [⟨.program ⟨257⟩, ⟨23018⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge141920

namespace LeftMerge141934
def owner : Owner := ⟨.program ⟨257⟩, ⟨22539⟩⟩
def mergeEvent : Nat := 141934
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22536⟩⟩] } }
def leftRaw : List Term := Proof.Events525.exact134495RawTerms
def rightRaw : List Term := Proof.Events554.exact141928RawTerms
def group : MergeGroup := .operator 134495 141928
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 134495) (leftOrdinal := 0)
    (rightResult := 141928) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨22536⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22536⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge141934

namespace LeftMerge142055
def owner : Owner := ⟨.program ⟨257⟩, ⟨23260⟩⟩
def mergeEvent : Nat := 142055
def frameStart : Nat := 141989
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨21752⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events554.exact142051RawTerms
def rightRaw : List Term := Proof.Events554.exact142049RawTerms
def group : MergeGroup := .operator 142051 142049
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 142051) (leftOrdinal := 0)
    (rightResult := 142049) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨21752⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨21752⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge142055

namespace LeftMerge142067
def owner : Owner := ⟨.program ⟨257⟩, ⟨23656⟩⟩
def mergeEvent : Nat := 142067
def frameStart : Nat := 141989
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7181⟩⟩, ⟨.program ⟨257⟩, ⟨23655⟩⟩] } }
def leftRaw : List Term := Proof.Events554.exact142063RawTerms
def rightRaw : List Term := Proof.Events554.exact142040RawTerms
def group : MergeGroup := .operator 142063 142040
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 142063) (leftOrdinal := 0)
    (rightResult := 142040) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7181⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨23655⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7181⟩⟩, ⟨.program ⟨257⟩, ⟨23655⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge142067

namespace LeftMerge142068
def owner : Owner := ⟨.program ⟨257⟩, ⟨23656⟩⟩
def mergeEvent : Nat := 142068
def frameStart : Nat := 141989
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨21752⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨23655⟩⟩] } }
def leftRaw : List Term := Proof.Events554.exact142063RawTerms
def rightRaw : List Term := Proof.Events554.exact142040RawTerms
def group : MergeGroup := .operator 142063 142040
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 142063) (leftOrdinal := 1)
    (rightResult := 142040) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨21752⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨23655⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨21752⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨23655⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge142068

namespace LeftMerge142070
def owner : Owner := ⟨.program ⟨257⟩, ⟨23656⟩⟩
def mergeEvent : Nat := 142070
def frameStart : Nat := 141989
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨21752⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨23018⟩⟩] } }
def rhsRaw : List Term := Proof.Events554.exact142037RawTerms
def group : MergeGroup := .relation 142069
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 142069) (rhsResult := 142037)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨21752⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨23655⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨23655⟩⟩) ⟨23018⟩ 142037) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨23018⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨21752⟩⟩], [⟨.program ⟨257⟩, ⟨23018⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge142070

namespace LeftMerge142078
def owner : Owner := ⟨.program ⟨257⟩, ⟨21955⟩⟩
def mergeEvent : Nat := 142078
def frameStart : Nat := 141989
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨21953⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events554.exact142051RawTerms
def rightRaw : List Term := Proof.Events554.exact142074RawTerms
def group : MergeGroup := .operator 142051 142074
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 142051) (leftOrdinal := 0)
    (rightResult := 142074) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨21953⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨21953⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge142078

namespace LeftMerge142095
def owner : Owner := ⟨.program ⟨257⟩, ⟨22539⟩⟩
def mergeEvent : Nat := 142095
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7202⟩⟩] } }
def rhsRaw : List Term := Proof.Events555.exact142092RawTerms
def group : MergeGroup := .relation 142094
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 142094) (rhsResult := 142092)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22536⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 142093 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22536⟩⟩]⟩) (none) 142092) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7202⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge142095

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
