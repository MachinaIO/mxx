import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge301796
def owner : Owner := ⟨.program ⟨257⟩, ⟨23168⟩⟩
def mergeEvent : Nat := 301796
def frameStart : Nat := 301748
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨20951⟩⟩, ⟨.program ⟨257⟩, ⟨21254⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1178.exact301792RawTerms
def rightRaw : List Term := Proof.Events1178.exact301790RawTerms
def group : MergeGroup := .operator 301792 301790
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 301792) (leftOrdinal := 0)
    (rightResult := 301790) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨20951⟩⟩, ⟨.program ⟨257⟩, ⟨21254⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨20951⟩⟩, ⟨.program ⟨257⟩, ⟨21254⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge301796

namespace LeftMerge301819
def owner : Owner := ⟨.program ⟨257⟩, ⟨9576⟩⟩
def mergeEvent : Nat := 301819
def frameStart : Nat := 301748
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩] } }
def leftRaw : List Term := Proof.Events1178.exact301815RawTerms
def rightRaw : List Term := Proof.Events1178.exact301812RawTerms
def group : MergeGroup := .operator 301815 301812
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 301815) (leftOrdinal := 0)
    (rightResult := 301812) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7286⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9574⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge301819

namespace LeftMerge301828
def owner : Owner := ⟨.program ⟨257⟩, ⟨23332⟩⟩
def mergeEvent : Nat := 301828
def frameStart : Nat := 301748
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23329⟩⟩] } }
def leftRaw : List Term := Proof.Events1179.exact301824RawTerms
def rightRaw : List Term := Proof.Events1178.exact301781RawTerms
def group : MergeGroup := .operator 301824 301781
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 301824) (leftOrdinal := 0)
    (rightResult := 301781) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨23329⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23329⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge301828

namespace LeftMerge301829
def owner : Owner := ⟨.program ⟨257⟩, ⟨23332⟩⟩
def mergeEvent : Nat := 301829
def frameStart : Nat := 301748
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨20951⟩⟩, ⟨.program ⟨257⟩, ⟨21254⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨23329⟩⟩] } }
def leftRaw : List Term := Proof.Events1179.exact301824RawTerms
def rightRaw : List Term := Proof.Events1178.exact301781RawTerms
def group : MergeGroup := .operator 301824 301781
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 301824) (leftOrdinal := 1)
    (rightResult := 301781) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨20951⟩⟩, ⟨.program ⟨257⟩, ⟨21254⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨23329⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨20951⟩⟩, ⟨.program ⟨257⟩, ⟨21254⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨23329⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge301829

namespace LeftMerge301831
def owner : Owner := ⟨.program ⟨257⟩, ⟨23332⟩⟩
def mergeEvent : Nat := 301831
def frameStart : Nat := 301748
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨20951⟩⟩, ⟨.program ⟨257⟩, ⟨21254⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨22869⟩⟩] } }
def rhsRaw : List Term := Proof.Events1178.exact301778RawTerms
def group : MergeGroup := .relation 301830
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 301830) (rhsResult := 301778)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨20951⟩⟩, ⟨.program ⟨257⟩, ⟨21254⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨23329⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨23329⟩⟩) ⟨22869⟩ 301778) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨22869⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨20951⟩⟩, ⟨.program ⟨257⟩, ⟨21254⟩⟩], [⟨.program ⟨257⟩, ⟨22869⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge301831

namespace LeftMerge301839
def owner : Owner := ⟨.program ⟨257⟩, ⟨21730⟩⟩
def mergeEvent : Nat := 301839
def frameStart : Nat := 301748
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨21728⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1178.exact301792RawTerms
def rightRaw : List Term := Proof.Events1179.exact301835RawTerms
def group : MergeGroup := .operator 301792 301835
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 301792) (leftOrdinal := 0)
    (rightResult := 301835) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨21728⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨21728⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge301839

namespace LeftMerge301856
def owner : Owner := ⟨.program ⟨257⟩, ⟨22272⟩⟩
def mergeEvent : Nat := 301856
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7181⟩⟩] } }
def rhsRaw : List Term := Proof.Events1179.exact301853RawTerms
def group : MergeGroup := .relation 301855
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 301855) (rhsResult := 301853)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22269⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 301854 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22269⟩⟩]⟩) (none) 301853) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7181⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7181⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge301856

namespace LeftMerge301857
def owner : Owner := ⟨.program ⟨257⟩, ⟨22272⟩⟩
def mergeEvent : Nat := 301857
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23329⟩⟩] } }
def rhsRaw : List Term := Proof.Events1179.exact301853RawTerms
def group : MergeGroup := .relation 301855
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 301855) (rhsResult := 301853)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22269⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 301854 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22269⟩⟩]⟩) (none) 301853) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23329⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23329⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge301857

namespace LeftMerge301858
def owner : Owner := ⟨.program ⟨257⟩, ⟨22272⟩⟩
def mergeEvent : Nat := 301858
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨20951⟩⟩, ⟨.program ⟨257⟩, ⟨21254⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨22869⟩⟩] } }
def rhsRaw : List Term := Proof.Events1179.exact301853RawTerms
def group : MergeGroup := .relation 301855
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 301855) (rhsResult := 301853)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22269⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 301854 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22269⟩⟩]⟩) (none) 301853) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨20951⟩⟩, ⟨.program ⟨257⟩, ⟨21254⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨22869⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨20951⟩⟩, ⟨.program ⟨257⟩, ⟨21254⟩⟩], [⟨.program ⟨257⟩, ⟨22869⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge301858

namespace LeftMerge301859
def owner : Owner := ⟨.program ⟨257⟩, ⟨22272⟩⟩
def mergeEvent : Nat := 301859
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨21728⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events1179.exact301853RawTerms
def group : MergeGroup := .relation 301855
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 301855) (rhsResult := 301853)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22269⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 301854 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22269⟩⟩]⟩) (none) 301853) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨21728⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨21728⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge301859

namespace LeftMerge301864
def owner : Owner := ⟨.program ⟨257⟩, ⟨23331⟩⟩
def mergeEvent : Nat := 301864
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨20951⟩⟩, ⟨.program ⟨257⟩, ⟨21254⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨22869⟩⟩] } }
def leftRaw : List Term := Proof.Events1179.exact301860RawTerms
def rightRaw : List Term := Proof.Events1178.exact301698RawTerms
def group : MergeGroup := .operator 301860 301698
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 301860) (leftOrdinal := 2)
    (rightResult := 301698) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨20951⟩⟩, ⟨.program ⟨257⟩, ⟨21254⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨22869⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨20951⟩⟩, ⟨.program ⟨257⟩, ⟨21254⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨22869⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨20951⟩⟩, ⟨.program ⟨257⟩, ⟨21254⟩⟩], [⟨.program ⟨257⟩, ⟨22869⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge301864

namespace LeftMerge301865
def owner : Owner := ⟨.program ⟨257⟩, ⟨23331⟩⟩
def mergeEvent : Nat := 301865
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23329⟩⟩] } }
def leftRaw : List Term := Proof.Events1179.exact301860RawTerms
def rightRaw : List Term := Proof.Events1178.exact301698RawTerms
def group : MergeGroup := .operator 301860 301698
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 301860) (leftOrdinal := 1)
    (rightResult := 301698) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23329⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23329⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23329⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge301865

namespace LeftMerge301873
def owner : Owner := ⟨.program ⟨257⟩, ⟨23564⟩⟩
def mergeEvent : Nat := 301873
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7181⟩⟩, ⟨.program ⟨257⟩, ⟨23562⟩⟩] } }
def leftRaw : List Term := Proof.Events1179.exact301867RawTerms
def rightRaw : List Term := Proof.Events1178.exact301614RawTerms
def group : MergeGroup := .operator 301867 301614
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 301867) (leftOrdinal := 0)
    (rightResult := 301614) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7181⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨23562⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7181⟩⟩, ⟨.program ⟨257⟩, ⟨23562⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge301873

namespace LeftMerge301874
def owner : Owner := ⟨.program ⟨257⟩, ⟨23564⟩⟩
def mergeEvent : Nat := 301874
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨21728⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨23562⟩⟩] } }
def leftRaw : List Term := Proof.Events1179.exact301867RawTerms
def rightRaw : List Term := Proof.Events1178.exact301614RawTerms
def group : MergeGroup := .operator 301867 301614
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 301867) (leftOrdinal := 1)
    (rightResult := 301614) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨21728⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨23562⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨21728⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨23562⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge301874

namespace LeftMerge301876
def owner : Owner := ⟨.program ⟨257⟩, ⟨23564⟩⟩
def mergeEvent : Nat := 301876
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨21728⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨22991⟩⟩] } }
def rhsRaw : List Term := Proof.Events1178.exact301611RawTerms
def group : MergeGroup := .relation 301875
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 301875) (rhsResult := 301611)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨21728⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨23562⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨23562⟩⟩) ⟨22991⟩ 301611) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨22991⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨21728⟩⟩], [⟨.program ⟨257⟩, ⟨22991⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge301876

namespace LeftMerge301890
def owner : Owner := ⟨.program ⟨257⟩, ⟨22479⟩⟩
def mergeEvent : Nat := 301890
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22476⟩⟩] } }
def leftRaw : List Term := Proof.Events1153.exact295195RawTerms
def rightRaw : List Term := Proof.Events1179.exact301884RawTerms
def group : MergeGroup := .operator 295195 301884
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 295195) (leftOrdinal := 0)
    (rightResult := 301884) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨22476⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22476⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge301890

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
