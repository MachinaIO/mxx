import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge119840
def owner : Owner := ⟨.program ⟨257⟩, ⟨47745⟩⟩
def mergeEvent : Nat := 119840
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨15021⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7285⟩⟩] } }
def leftRaw : List Term := Proof.Events468.exact119836RawTerms
def rightRaw : List Term := Proof.Events467.exact119806RawTerms
def group : MergeGroup := .operator 119836 119806
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 119836) (leftOrdinal := 1)
    (rightResult := 119806) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨15021⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7285⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨15021⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7285⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨15021⟩⟩], [⟨.program ⟨257⟩, ⟨7285⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge119840

namespace LeftMerge119848
def owner : Owner := ⟨.program ⟨257⟩, ⟨49616⟩⟩
def mergeEvent : Nat := 119848
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨15021⟩⟩, ⟨.program ⟨257⟩, ⟨47738⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨49615⟩⟩] } }
def leftRaw : List Term := Proof.Events468.exact119842RawTerms
def rightRaw : List Term := Proof.Events467.exact119773RawTerms
def group : MergeGroup := .operator 119842 119773
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 119842) (leftOrdinal := 1)
    (rightResult := 119773) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨15021⟩⟩, ⟨.program ⟨257⟩, ⟨47738⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨49615⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨15021⟩⟩, ⟨.program ⟨257⟩, ⟨47738⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨49615⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge119848

namespace LeftMerge119850
def owner : Owner := ⟨.program ⟨257⟩, ⟨49616⟩⟩
def mergeEvent : Nat := 119850
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨15021⟩⟩, ⟨.program ⟨257⟩, ⟨47738⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨49125⟩⟩] } }
def rhsRaw : List Term := Proof.Events467.exact119770RawTerms
def group : MergeGroup := .relation 119849
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 119849) (rhsResult := 119770)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨15021⟩⟩, ⟨.program ⟨257⟩, ⟨47738⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨49615⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨49615⟩⟩) ⟨49125⟩ 119770) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨49125⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨15021⟩⟩, ⟨.program ⟨257⟩, ⟨47738⟩⟩], [⟨.program ⟨257⟩, ⟨49125⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge119850

namespace LeftMerge119851
def owner : Owner := ⟨.program ⟨257⟩, ⟨49616⟩⟩
def mergeEvent : Nat := 119851
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49615⟩⟩] } }
def leftRaw : List Term := Proof.Events468.exact119842RawTerms
def rightRaw : List Term := Proof.Events467.exact119773RawTerms
def group : MergeGroup := .operator 119842 119773
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 119842) (leftOrdinal := 0)
    (rightResult := 119773) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨49615⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49615⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge119851

namespace LeftMerge119863
def owner : Owner := ⟨.program ⟨257⟩, ⟨5526⟩⟩
def mergeEvent : Nat := 119863
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }
def leftRaw : List Term := Proof.Events467.exact119648RawTerms
def rightRaw : List Term := Proof.Events067.exact17158RawTerms
def group : MergeGroup := .operator 119648 17158
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 119648) (leftOrdinal := 0)
    (rightResult := 17158) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge119863

namespace LeftMerge119876
def owner : Owner := ⟨.program ⟨257⟩, ⟨48552⟩⟩
def mergeEvent : Nat := 119876
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48549⟩⟩] } }
def leftRaw : List Term := Proof.Events468.exact119870RawTerms
def rightRaw : List Term := Proof.Events468.exact119859RawTerms
def group : MergeGroup := .operator 119870 119859
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 119870) (leftOrdinal := 0)
    (rightResult := 119859) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨48549⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48549⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge119876

namespace LeftMerge119955
def owner : Owner := ⟨.program ⟨257⟩, ⟨47739⟩⟩
def mergeEvent : Nat := 119955
def frameStart : Nat := 119925
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨15021⟩⟩, ⟨.program ⟨257⟩, ⟨47738⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events468.exact119951RawTerms
def rightRaw : List Term := Proof.Events468.exact119948RawTerms
def group : MergeGroup := .operator 119951 119948
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 119951) (leftOrdinal := 0)
    (rightResult := 119948) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨15021⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨47738⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨15021⟩⟩, ⟨.program ⟨257⟩, ⟨47738⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge119955

namespace LeftMerge119985
def owner : Owner := ⟨.program ⟨257⟩, ⟨49412⟩⟩
def mergeEvent : Nat := 119985
def frameStart : Nat := 119925
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨15021⟩⟩, ⟨.program ⟨257⟩, ⟨47738⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events468.exact119981RawTerms
def rightRaw : List Term := Proof.Events468.exact119979RawTerms
def group : MergeGroup := .operator 119981 119979
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 119981) (leftOrdinal := 0)
    (rightResult := 119979) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨15021⟩⟩, ⟨.program ⟨257⟩, ⟨47738⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨15021⟩⟩, ⟨.program ⟨257⟩, ⟨47738⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge119985

namespace LeftMerge120008
def owner : Owner := ⟨.program ⟨257⟩, ⟨9567⟩⟩
def mergeEvent : Nat := 120008
def frameStart : Nat := 119925
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩] } }
def leftRaw : List Term := Proof.Events468.exact120004RawTerms
def rightRaw : List Term := Proof.Events468.exact120001RawTerms
def group : MergeGroup := .operator 120004 120001
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 120004) (leftOrdinal := 0)
    (rightResult := 120001) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9565⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge120008

namespace LeftMerge120017
def owner : Owner := ⟨.program ⟨257⟩, ⟨49618⟩⟩
def mergeEvent : Nat := 120017
def frameStart : Nat := 119925
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49615⟩⟩] } }
def leftRaw : List Term := Proof.Events468.exact120013RawTerms
def rightRaw : List Term := Proof.Events468.exact119970RawTerms
def group : MergeGroup := .operator 120013 119970
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 120013) (leftOrdinal := 0)
    (rightResult := 119970) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨49615⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49615⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge120017

namespace LeftMerge120018
def owner : Owner := ⟨.program ⟨257⟩, ⟨49618⟩⟩
def mergeEvent : Nat := 120018
def frameStart : Nat := 119925
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨15021⟩⟩, ⟨.program ⟨257⟩, ⟨47738⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨49615⟩⟩] } }
def leftRaw : List Term := Proof.Events468.exact120013RawTerms
def rightRaw : List Term := Proof.Events468.exact119970RawTerms
def group : MergeGroup := .operator 120013 119970
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 120013) (leftOrdinal := 1)
    (rightResult := 119970) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨15021⟩⟩, ⟨.program ⟨257⟩, ⟨47738⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨49615⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨15021⟩⟩, ⟨.program ⟨257⟩, ⟨47738⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨49615⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge120018

namespace LeftMerge120020
def owner : Owner := ⟨.program ⟨257⟩, ⟨49618⟩⟩
def mergeEvent : Nat := 120020
def frameStart : Nat := 119925
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨15021⟩⟩, ⟨.program ⟨257⟩, ⟨47738⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨49125⟩⟩] } }
def rhsRaw : List Term := Proof.Events468.exact119967RawTerms
def group : MergeGroup := .relation 120019
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 120019) (rhsResult := 119967)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨15021⟩⟩, ⟨.program ⟨257⟩, ⟨47738⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨49615⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨49615⟩⟩) ⟨49125⟩ 119967) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨49125⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨15021⟩⟩, ⟨.program ⟨257⟩, ⟨47738⟩⟩], [⟨.program ⟨257⟩, ⟨49125⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge120020

namespace LeftMerge120028
def owner : Owner := ⟨.program ⟨257⟩, ⟨48118⟩⟩
def mergeEvent : Nat := 120028
def frameStart : Nat := 119925
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨48116⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events468.exact119981RawTerms
def rightRaw : List Term := Proof.Events468.exact120024RawTerms
def group : MergeGroup := .operator 119981 120024
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 119981) (leftOrdinal := 0)
    (rightResult := 120024) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨48116⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨48116⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge120028

namespace LeftMerge120045
def owner : Owner := ⟨.program ⟨257⟩, ⟨48552⟩⟩
def mergeEvent : Nat := 120045
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7196⟩⟩] } }
def rhsRaw : List Term := Proof.Events468.exact120042RawTerms
def group : MergeGroup := .relation 120044
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 120044) (rhsResult := 120042)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48549⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 120043 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48549⟩⟩]⟩) (none) 120042) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7196⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7196⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge120045

namespace LeftMerge120046
def owner : Owner := ⟨.program ⟨257⟩, ⟨48552⟩⟩
def mergeEvent : Nat := 120046
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49615⟩⟩] } }
def rhsRaw : List Term := Proof.Events468.exact120042RawTerms
def group : MergeGroup := .relation 120044
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 120044) (rhsResult := 120042)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48549⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 120043 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48549⟩⟩]⟩) (none) 120042) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49615⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49615⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge120046

namespace LeftMerge120047
def owner : Owner := ⟨.program ⟨257⟩, ⟨48552⟩⟩
def mergeEvent : Nat := 120047
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨15021⟩⟩, ⟨.program ⟨257⟩, ⟨47738⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨49125⟩⟩] } }
def rhsRaw : List Term := Proof.Events468.exact120042RawTerms
def group : MergeGroup := .relation 120044
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 120044) (rhsResult := 120042)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48549⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 120043 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48549⟩⟩]⟩) (none) 120042) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨15021⟩⟩, ⟨.program ⟨257⟩, ⟨47738⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨49125⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨15021⟩⟩, ⟨.program ⟨257⟩, ⟨47738⟩⟩], [⟨.program ⟨257⟩, ⟨49125⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge120047

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
