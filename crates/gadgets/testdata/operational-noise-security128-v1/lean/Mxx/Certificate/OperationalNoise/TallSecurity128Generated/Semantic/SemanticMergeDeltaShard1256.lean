import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge205060
def owner : Owner := ⟨.program ⟨257⟩, ⟨70321⟩⟩
def mergeEvent : Nat := 205060
def frameStart : Nat := 204981
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨65804⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70320⟩⟩] } }
def leftRaw : List Term := Proof.Events800.exact205055RawTerms
def rightRaw : List Term := Proof.Events800.exact205032RawTerms
def group : MergeGroup := .operator 205055 205032
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 205055) (leftOrdinal := 1)
    (rightResult := 205032) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨65804⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨70320⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨65804⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70320⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge205060

namespace LeftMerge205062
def owner : Owner := ⟨.program ⟨257⟩, ⟨70321⟩⟩
def mergeEvent : Nat := 205062
def frameStart : Nat := 204981
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨65804⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68699⟩⟩] } }
def rhsRaw : List Term := Proof.Events800.exact205029RawTerms
def group : MergeGroup := .relation 205061
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 205061) (rhsResult := 205029)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨65804⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70320⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨70320⟩⟩) ⟨68699⟩ 205029) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68699⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨65804⟩⟩], [⟨.program ⟨257⟩, ⟨68699⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge205062

namespace LeftMerge205070
def owner : Owner := ⟨.program ⟨257⟩, ⟨66739⟩⟩
def mergeEvent : Nat := 205070
def frameStart : Nat := 204981
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨66728⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events800.exact205043RawTerms
def rightRaw : List Term := Proof.Events801.exact205066RawTerms
def group : MergeGroup := .operator 205043 205066
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 205043) (leftOrdinal := 0)
    (rightResult := 205066) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨66728⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨66728⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge205070

namespace LeftMerge205087
def owner : Owner := ⟨.program ⟨257⟩, ⟨68116⟩⟩
def mergeEvent : Nat := 205087
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7215⟩⟩] } }
def rhsRaw : List Term := Proof.Events801.exact205084RawTerms
def group : MergeGroup := .relation 205086
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 205086) (rhsResult := 205084)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68113⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 205085 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68113⟩⟩]⟩) (none) 205084) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7215⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7215⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge205087

namespace LeftMerge205088
def owner : Owner := ⟨.program ⟨257⟩, ⟨68116⟩⟩
def mergeEvent : Nat := 205088
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7188⟩⟩, ⟨.program ⟨257⟩, ⟨70320⟩⟩] } }
def rhsRaw : List Term := Proof.Events801.exact205084RawTerms
def group : MergeGroup := .relation 205086
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 205086) (rhsResult := 205084)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68113⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 205085 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68113⟩⟩]⟩) (none) 205084) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7188⟩⟩, ⟨.program ⟨257⟩, ⟨70320⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7188⟩⟩, ⟨.program ⟨257⟩, ⟨70320⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge205088

namespace LeftMerge205089
def owner : Owner := ⟨.program ⟨257⟩, ⟨68116⟩⟩
def mergeEvent : Nat := 205089
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨65804⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68699⟩⟩] } }
def rhsRaw : List Term := Proof.Events801.exact205084RawTerms
def group : MergeGroup := .relation 205086
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 205086) (rhsResult := 205084)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68113⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 205085 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68113⟩⟩]⟩) (none) 205084) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨65804⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68699⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨65804⟩⟩], [⟨.program ⟨257⟩, ⟨68699⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge205089

namespace LeftMerge205090
def owner : Owner := ⟨.program ⟨257⟩, ⟨68116⟩⟩
def mergeEvent : Nat := 205090
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨66728⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events801.exact205084RawTerms
def group : MergeGroup := .relation 205086
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 205086) (rhsResult := 205084)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68113⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 205085 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68113⟩⟩]⟩) (none) 205084) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨66728⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨66728⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge205090

namespace LeftMerge205095
def owner : Owner := ⟨.program ⟨257⟩, ⟨70323⟩⟩
def mergeEvent : Nat := 205095
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7188⟩⟩, ⟨.program ⟨257⟩, ⟨70320⟩⟩] } }
def leftRaw : List Term := Proof.Events801.exact205091RawTerms
def rightRaw : List Term := Proof.Events800.exact204913RawTerms
def group : MergeGroup := .operator 205091 204913
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 205091) (leftOrdinal := 0)
    (rightResult := 204913) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7188⟩⟩, ⟨.program ⟨257⟩, ⟨70320⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7188⟩⟩, ⟨.program ⟨257⟩, ⟨70320⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7188⟩⟩, ⟨.program ⟨257⟩, ⟨70320⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge205095

namespace LeftMerge205096
def owner : Owner := ⟨.program ⟨257⟩, ⟨70323⟩⟩
def mergeEvent : Nat := 205096
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨65804⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68699⟩⟩] } }
def leftRaw : List Term := Proof.Events801.exact205091RawTerms
def rightRaw : List Term := Proof.Events800.exact204913RawTerms
def group : MergeGroup := .operator 205091 204913
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 205091) (leftOrdinal := 2)
    (rightResult := 204913) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨65804⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68699⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨65804⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68699⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨65804⟩⟩], [⟨.program ⟨257⟩, ⟨68699⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge205096

namespace LeftMerge205104
def owner : Owner := ⟨.program ⟨257⟩, ⟨70324⟩⟩
def mergeEvent : Nat := 205104
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7215⟩⟩, ⟨.program ⟨257⟩, ⟨7173⟩⟩] } }
def leftRaw : List Term := Proof.Events801.exact205098RawTerms
def rightRaw : List Term := Proof.Events061.exact15702RawTerms
def group : MergeGroup := .operator 205098 15702
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 205098) (leftOrdinal := 0)
    (rightResult := 15702) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7215⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7173⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7215⟩⟩, ⟨.program ⟨257⟩, ⟨7173⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge205104

namespace LeftMerge205105
def owner : Owner := ⟨.program ⟨257⟩, ⟨70324⟩⟩
def mergeEvent : Nat := 205105
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨66728⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7173⟩⟩] } }
def leftRaw : List Term := Proof.Events801.exact205098RawTerms
def rightRaw : List Term := Proof.Events061.exact15702RawTerms
def group : MergeGroup := .operator 205098 15702
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 205098) (leftOrdinal := 1)
    (rightResult := 15702) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨66728⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7173⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨66728⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7173⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge205105

namespace LeftMerge205107
def owner : Owner := ⟨.program ⟨257⟩, ⟨70324⟩⟩
def mergeEvent : Nat := 205107
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66728⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events061.exact15695RawTerms
def group : MergeGroup := .relation 205106
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 205106) (rhsResult := 15695)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨66728⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7173⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨7173⟩⟩) ⟨7052⟩ 15695) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6870⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66728⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge205107

namespace LeftMerge205121
def owner : Owner := ⟨.program ⟨257⟩, ⟨64929⟩⟩
def mergeEvent : Nat := 205121
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7187⟩⟩, ⟨.program ⟨257⟩, ⟨64927⟩⟩] } }
def leftRaw : List Term := Proof.Events771.exact197519RawTerms
def rightRaw : List Term := Proof.Events801.exact205115RawTerms
def group : MergeGroup := .operator 197519 205115
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 197519) (leftOrdinal := 0)
    (rightResult := 205115) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7187⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨64927⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7187⟩⟩, ⟨.program ⟨257⟩, ⟨64927⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge205121

namespace LeftMerge205122
def owner : Owner := ⟨.program ⟨257⟩, ⟨64929⟩⟩
def mergeEvent : Nat := 205122
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨62824⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨64927⟩⟩] } }
def leftRaw : List Term := Proof.Events771.exact197519RawTerms
def rightRaw : List Term := Proof.Events801.exact205115RawTerms
def group : MergeGroup := .operator 197519 205115
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 197519) (leftOrdinal := 1)
    (rightResult := 205115) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨62824⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨64927⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨62824⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨64927⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge205122

namespace LeftMerge205124
def owner : Owner := ⟨.program ⟨257⟩, ⟨64929⟩⟩
def mergeEvent : Nat := 205124
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨62824⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨64098⟩⟩] } }
def rhsRaw : List Term := Proof.Events801.exact205112RawTerms
def group : MergeGroup := .relation 205123
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 205123) (rhsResult := 205112)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨62824⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨64927⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨64927⟩⟩) ⟨64098⟩ 205112) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨64098⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨62824⟩⟩], [⟨.program ⟨257⟩, ⟨64098⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge205124

namespace LeftMerge205138
def owner : Owner := ⟨.program ⟨257⟩, ⟨63715⟩⟩
def mergeEvent : Nat := 205138
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63712⟩⟩] } }
def leftRaw : List Term := Proof.Events753.exact192995RawTerms
def rightRaw : List Term := Proof.Events801.exact205132RawTerms
def group : MergeGroup := .operator 192995 205132
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 192995) (leftOrdinal := 0)
    (rightResult := 205132) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨63712⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63712⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge205138

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
