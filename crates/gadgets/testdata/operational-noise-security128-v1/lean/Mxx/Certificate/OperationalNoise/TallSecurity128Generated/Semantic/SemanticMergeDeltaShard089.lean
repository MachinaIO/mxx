import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge17853
def owner : Owner := ⟨.program ⟨257⟩, ⟨46885⟩⟩
def mergeEvent : Nat := 17853
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨14651⟩⟩, ⟨.program ⟨257⟩, ⟨44946⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46417⟩⟩] } }
def leftRaw : List Term := Proof.Events069.exact17849RawTerms
def rightRaw : List Term := Proof.Events068.exact17663RawTerms
def group : MergeGroup := .operator 17849 17663
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 17849) (leftOrdinal := 2)
    (rightResult := 17663) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨14651⟩⟩, ⟨.program ⟨257⟩, ⟨44946⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46417⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨14651⟩⟩, ⟨.program ⟨257⟩, ⟨44946⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46417⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨14651⟩⟩, ⟨.program ⟨257⟩, ⟨44946⟩⟩], [⟨.program ⟨257⟩, ⟨46417⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge17853

namespace LeftMerge17854
def owner : Owner := ⟨.program ⟨257⟩, ⟨46885⟩⟩
def mergeEvent : Nat := 17854
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨46883⟩⟩] } }
def leftRaw : List Term := Proof.Events069.exact17849RawTerms
def rightRaw : List Term := Proof.Events068.exact17663RawTerms
def group : MergeGroup := .operator 17849 17663
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 17849) (leftOrdinal := 1)
    (rightResult := 17663) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨46883⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨46883⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨46883⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge17854

namespace LeftMerge17862
def owner : Owner := ⟨.program ⟨257⟩, ⟨47133⟩⟩
def mergeEvent : Nat := 17862
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨45398⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨47131⟩⟩] } }
def leftRaw : List Term := Proof.Events069.exact17856RawTerms
def rightRaw : List Term := Proof.Events068.exact17560RawTerms
def group : MergeGroup := .operator 17856 17560
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 17856) (leftOrdinal := 1)
    (rightResult := 17560) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨45398⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨47131⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨45398⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨47131⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge17862

namespace LeftMerge17864
def owner : Owner := ⟨.program ⟨257⟩, ⟨47133⟩⟩
def mergeEvent : Nat := 17864
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨45398⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46543⟩⟩] } }
def rhsRaw : List Term := Proof.Events068.exact17557RawTerms
def group : MergeGroup := .relation 17863
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 17863) (rhsResult := 17557)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨45398⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨47131⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨47131⟩⟩) ⟨46543⟩ 17557) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨46543⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨45398⟩⟩], [⟨.program ⟨257⟩, ⟨46543⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge17864

namespace LeftMerge17865
def owner : Owner := ⟨.program ⟨257⟩, ⟨47133⟩⟩
def mergeEvent : Nat := 17865
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47131⟩⟩] } }
def leftRaw : List Term := Proof.Events069.exact17856RawTerms
def rightRaw : List Term := Proof.Events068.exact17560RawTerms
def group : MergeGroup := .operator 17856 17560
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 17856) (leftOrdinal := 0)
    (rightResult := 17560) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7195⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨47131⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47131⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge17865

namespace LeftMerge17879
def owner : Owner := ⟨.program ⟨257⟩, ⟨46045⟩⟩
def mergeEvent : Nat := 17879
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46042⟩⟩] } }
def leftRaw : List Term := Proof.Events067.exact17169RawTerms
def rightRaw : List Term := Proof.Events069.exact17873RawTerms
def group : MergeGroup := .operator 17169 17873
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 17169) (leftOrdinal := 0)
    (rightResult := 17873) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨46042⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46042⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge17879

namespace LeftMerge18000
def owner : Owner := ⟨.program ⟨257⟩, ⟨46792⟩⟩
def mergeEvent : Nat := 18000
def frameStart : Nat := 17934
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨45398⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events070.exact17996RawTerms
def rightRaw : List Term := Proof.Events070.exact17994RawTerms
def group : MergeGroup := .operator 17996 17994
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 17996) (leftOrdinal := 0)
    (rightResult := 17994) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨45398⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨45398⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge18000

namespace LeftMerge18012
def owner : Owner := ⟨.program ⟨257⟩, ⟨47132⟩⟩
def mergeEvent : Nat := 18012
def frameStart : Nat := 17934
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨45398⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨47131⟩⟩] } }
def leftRaw : List Term := Proof.Events070.exact18008RawTerms
def rightRaw : List Term := Proof.Events070.exact17985RawTerms
def group : MergeGroup := .operator 18008 17985
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 18008) (leftOrdinal := 1)
    (rightResult := 17985) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨45398⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨47131⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨45398⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨47131⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge18012

namespace LeftMerge18014
def owner : Owner := ⟨.program ⟨257⟩, ⟨47132⟩⟩
def mergeEvent : Nat := 18014
def frameStart : Nat := 17934
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨45398⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46543⟩⟩] } }
def rhsRaw : List Term := Proof.Events070.exact17982RawTerms
def group : MergeGroup := .relation 18013
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 18013) (rhsResult := 17982)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨45398⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨47131⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨47131⟩⟩) ⟨46543⟩ 17982) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨46543⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨45398⟩⟩], [⟨.program ⟨257⟩, ⟨46543⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge18014

namespace LeftMerge18015
def owner : Owner := ⟨.program ⟨257⟩, ⟨47132⟩⟩
def mergeEvent : Nat := 18015
def frameStart : Nat := 17934
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47131⟩⟩] } }
def leftRaw : List Term := Proof.Events070.exact18008RawTerms
def rightRaw : List Term := Proof.Events070.exact17985RawTerms
def group : MergeGroup := .operator 18008 17985
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 18008) (leftOrdinal := 0)
    (rightResult := 17985) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7195⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨47131⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47131⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge18015

namespace LeftMerge18023
def owner : Owner := ⟨.program ⟨257⟩, ⟨45570⟩⟩
def mergeEvent : Nat := 18023
def frameStart : Nat := 17934
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨45569⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events070.exact17996RawTerms
def rightRaw : List Term := Proof.Events070.exact18019RawTerms
def group : MergeGroup := .operator 17996 18019
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 17996) (leftOrdinal := 0)
    (rightResult := 18019) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨45569⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨45569⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge18023

namespace LeftMerge18040
def owner : Owner := ⟨.program ⟨257⟩, ⟨46045⟩⟩
def mergeEvent : Nat := 18040
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨45398⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46543⟩⟩] } }
def rhsRaw : List Term := Proof.Events070.exact18037RawTerms
def group : MergeGroup := .relation 18039
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 18039) (rhsResult := 18037)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46042⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 18038 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46042⟩⟩]⟩) (none) 18037) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨45398⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46543⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨45398⟩⟩], [⟨.program ⟨257⟩, ⟨46543⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge18040

namespace LeftMerge18041
def owner : Owner := ⟨.program ⟨257⟩, ⟨46045⟩⟩
def mergeEvent : Nat := 18041
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47131⟩⟩] } }
def rhsRaw : List Term := Proof.Events070.exact18037RawTerms
def group : MergeGroup := .relation 18039
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 18039) (rhsResult := 18037)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46042⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 18038 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46042⟩⟩]⟩) (none) 18037) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47131⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47131⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge18041

namespace LeftMerge18042
def owner : Owner := ⟨.program ⟨257⟩, ⟨46045⟩⟩
def mergeEvent : Nat := 18042
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨45569⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events070.exact18037RawTerms
def group : MergeGroup := .relation 18039
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 18039) (rhsResult := 18037)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46042⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 18038 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46042⟩⟩]⟩) (none) 18037) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨45569⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨45569⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge18042

namespace LeftMerge18043
def owner : Owner := ⟨.program ⟨257⟩, ⟨46045⟩⟩
def mergeEvent : Nat := 18043
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7230⟩⟩] } }
def rhsRaw : List Term := Proof.Events070.exact18037RawTerms
def group : MergeGroup := .relation 18039
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 18039) (rhsResult := 18037)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46042⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 18038 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46042⟩⟩]⟩) (none) 18037) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7230⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7230⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge18043

namespace LeftMerge18048
def owner : Owner := ⟨.program ⟨257⟩, ⟨47134⟩⟩
def mergeEvent : Nat := 18048
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨45398⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46543⟩⟩] } }
def leftRaw : List Term := Proof.Events070.exact18044RawTerms
def rightRaw : List Term := Proof.Events069.exact17866RawTerms
def group : MergeGroup := .operator 18044 17866
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 18044) (leftOrdinal := 2)
    (rightResult := 17866) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨45398⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46543⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨45398⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46543⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨45398⟩⟩], [⟨.program ⟨257⟩, ⟨46543⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge18048

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
