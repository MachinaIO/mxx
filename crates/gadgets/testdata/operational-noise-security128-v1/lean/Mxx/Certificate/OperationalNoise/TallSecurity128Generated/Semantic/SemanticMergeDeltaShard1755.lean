import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge283818
def owner : Owner := ⟨.program ⟨257⟩, ⟨30821⟩⟩
def mergeEvent : Nat := 283818
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨29040⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨30187⟩⟩] } }
def rhsRaw : List Term := Proof.Events1107.exact283531RawTerms
def group : MergeGroup := .relation 283817
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 283817) (rhsResult := 283531)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨29040⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨30819⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨30819⟩⟩) ⟨30187⟩ 283531) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨30187⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨29040⟩⟩], [⟨.program ⟨257⟩, ⟨30187⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge283818

namespace LeftMerge283832
def owner : Owner := ⟨.program ⟨257⟩, ⟨29719⟩⟩
def mergeEvent : Nat := 283832
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29716⟩⟩] } }
def leftRaw : List Term := Proof.Events1096.exact280745RawTerms
def rightRaw : List Term := Proof.Events1108.exact283826RawTerms
def group : MergeGroup := .operator 280745 283826
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 280745) (leftOrdinal := 0)
    (rightResult := 283826) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨29716⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29716⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge283832

namespace LeftMerge283953
def owner : Owner := ⟨.program ⟨257⟩, ⟨30424⟩⟩
def mergeEvent : Nat := 283953
def frameStart : Nat := 283887
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨29040⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1109.exact283949RawTerms
def rightRaw : List Term := Proof.Events1109.exact283947RawTerms
def group : MergeGroup := .operator 283949 283947
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 283949) (leftOrdinal := 0)
    (rightResult := 283947) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨29040⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨29040⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge283953

namespace LeftMerge283965
def owner : Owner := ⟨.program ⟨257⟩, ⟨30820⟩⟩
def mergeEvent : Nat := 283965
def frameStart : Nat := 283887
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨30819⟩⟩] } }
def leftRaw : List Term := Proof.Events1109.exact283961RawTerms
def rightRaw : List Term := Proof.Events1109.exact283938RawTerms
def group : MergeGroup := .operator 283961 283938
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 283961) (leftOrdinal := 0)
    (rightResult := 283938) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7190⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨30819⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨30819⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge283965

namespace LeftMerge283966
def owner : Owner := ⟨.program ⟨257⟩, ⟨30820⟩⟩
def mergeEvent : Nat := 283966
def frameStart : Nat := 283887
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨29040⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨30819⟩⟩] } }
def leftRaw : List Term := Proof.Events1109.exact283961RawTerms
def rightRaw : List Term := Proof.Events1109.exact283938RawTerms
def group : MergeGroup := .operator 283961 283938
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 283961) (leftOrdinal := 1)
    (rightResult := 283938) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨29040⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨30819⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨29040⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨30819⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge283966

namespace LeftMerge283968
def owner : Owner := ⟨.program ⟨257⟩, ⟨30820⟩⟩
def mergeEvent : Nat := 283968
def frameStart : Nat := 283887
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨29040⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨30187⟩⟩] } }
def rhsRaw : List Term := Proof.Events1109.exact283935RawTerms
def group : MergeGroup := .relation 283967
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 283967) (rhsResult := 283935)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨29040⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨30819⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨30819⟩⟩) ⟨30187⟩ 283935) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨30187⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨29040⟩⟩], [⟨.program ⟨257⟩, ⟨30187⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge283968

namespace LeftMerge283976
def owner : Owner := ⟨.program ⟨257⟩, ⟨29222⟩⟩
def mergeEvent : Nat := 283976
def frameStart : Nat := 283887
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨29221⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1109.exact283949RawTerms
def rightRaw : List Term := Proof.Events1109.exact283972RawTerms
def group : MergeGroup := .operator 283949 283972
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 283949) (leftOrdinal := 0)
    (rightResult := 283972) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨29221⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨29221⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge283976

namespace LeftMerge283993
def owner : Owner := ⟨.program ⟨257⟩, ⟨29719⟩⟩
def mergeEvent : Nat := 283993
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7220⟩⟩] } }
def rhsRaw : List Term := Proof.Events1109.exact283990RawTerms
def group : MergeGroup := .relation 283992
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 283992) (rhsResult := 283990)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29716⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 283991 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29716⟩⟩]⟩) (none) 283990) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7220⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge283993

namespace LeftMerge283994
def owner : Owner := ⟨.program ⟨257⟩, ⟨29719⟩⟩
def mergeEvent : Nat := 283994
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨30819⟩⟩] } }
def rhsRaw : List Term := Proof.Events1109.exact283990RawTerms
def group : MergeGroup := .relation 283992
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 283992) (rhsResult := 283990)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29716⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 283991 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29716⟩⟩]⟩) (none) 283990) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨30819⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨30819⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge283994

namespace LeftMerge283995
def owner : Owner := ⟨.program ⟨257⟩, ⟨29719⟩⟩
def mergeEvent : Nat := 283995
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨29040⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨30187⟩⟩] } }
def rhsRaw : List Term := Proof.Events1109.exact283990RawTerms
def group : MergeGroup := .relation 283992
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 283992) (rhsResult := 283990)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29716⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 283991 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29716⟩⟩]⟩) (none) 283990) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨29040⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨30187⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨29040⟩⟩], [⟨.program ⟨257⟩, ⟨30187⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge283995

namespace LeftMerge283996
def owner : Owner := ⟨.program ⟨257⟩, ⟨29719⟩⟩
def mergeEvent : Nat := 283996
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨29221⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events1109.exact283990RawTerms
def group : MergeGroup := .relation 283992
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 283992) (rhsResult := 283990)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29716⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 283991 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29716⟩⟩]⟩) (none) 283990) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨29221⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨29221⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge283996

namespace LeftMerge284001
def owner : Owner := ⟨.program ⟨257⟩, ⟨30822⟩⟩
def mergeEvent : Nat := 284001
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨30819⟩⟩] } }
def leftRaw : List Term := Proof.Events1109.exact283997RawTerms
def rightRaw : List Term := Proof.Events1108.exact283819RawTerms
def group : MergeGroup := .operator 283997 283819
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 283997) (leftOrdinal := 0)
    (rightResult := 283819) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨30819⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨30819⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨30819⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge284001

namespace LeftMerge284002
def owner : Owner := ⟨.program ⟨257⟩, ⟨30822⟩⟩
def mergeEvent : Nat := 284002
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨29040⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨30187⟩⟩] } }
def leftRaw : List Term := Proof.Events1109.exact283997RawTerms
def rightRaw : List Term := Proof.Events1108.exact283819RawTerms
def group : MergeGroup := .operator 283997 283819
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 283997) (leftOrdinal := 2)
    (rightResult := 283819) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨29040⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨30187⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨29040⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨30187⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨29040⟩⟩], [⟨.program ⟨257⟩, ⟨30187⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge284002

namespace LeftMerge284028
def owner : Owner := ⟨.program ⟨257⟩, ⟨25953⟩⟩
def mergeEvent : Nat := 284028
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨25950⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events053.exact13713RawTerms
def rightRaw : List Term := Proof.Events1096.exact280653RawTerms
def group : MergeGroup := .operator 13713 280653
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 13713) (leftOrdinal := 0)
    (rightResult := 280653) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25950⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨25950⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge284028

namespace LeftMerge284033
def owner : Owner := ⟨.program ⟨257⟩, ⟨7900⟩⟩
def mergeEvent : Nat := 284033
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7278⟩⟩] } }
def leftRaw : List Term := Proof.Events1095.exact280523RawTerms
def rightRaw : List Term := Proof.Events080.exact20587RawTerms
def group : MergeGroup := .operator 280523 20587
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 280523) (leftOrdinal := 0)
    (rightResult := 20587) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7278⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7278⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge284033

namespace LeftMerge284050
def owner : Owner := ⟨.program ⟨257⟩, ⟨25956⟩⟩
def mergeEvent : Nat := 284050
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨12891⟩⟩, ⟨.program ⟨257⟩, ⟨25950⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1109.exact284044RawTerms
def rightRaw : List Term := Proof.Events053.exact13716RawTerms
def group : MergeGroup := .operator 284044 13716
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 284044) (leftOrdinal := 1)
    (rightResult := 13716) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨25950⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨12891⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨12891⟩⟩, ⟨.program ⟨257⟩, ⟨25950⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge284050

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
