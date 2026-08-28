import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge224646
def owner : Owner := ⟨.program ⟨257⟩, ⟨36249⟩⟩
def mergeEvent : Nat := 224646
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨13566⟩⟩, ⟨.program ⟨257⟩, ⟨34410⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35743⟩⟩] } }
def rhsRaw : List Term := Proof.Events877.exact224571RawTerms
def group : MergeGroup := .relation 224645
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 224645) (rhsResult := 224571)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨13566⟩⟩, ⟨.program ⟨257⟩, ⟨34410⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36248⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨36248⟩⟩) ⟨35743⟩ 224571) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨35743⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨13566⟩⟩, ⟨.program ⟨257⟩, ⟨34410⟩⟩], [⟨.program ⟨257⟩, ⟨35743⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge224646

namespace LeftMerge224647
def owner : Owner := ⟨.program ⟨257⟩, ⟨36249⟩⟩
def mergeEvent : Nat := 224647
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩, ⟨.program ⟨257⟩, ⟨36248⟩⟩] } }
def leftRaw : List Term := Proof.Events877.exact224638RawTerms
def rightRaw : List Term := Proof.Events877.exact224574RawTerms
def group : MergeGroup := .operator 224638 224574
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 224638) (leftOrdinal := 0)
    (rightResult := 224574) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨36248⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩, ⟨.program ⟨257⟩, ⟨36248⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge224647

namespace LeftMerge224661
def owner : Owner := ⟨.program ⟨257⟩, ⟨35182⟩⟩
def mergeEvent : Nat := 224661
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35179⟩⟩] } }
def leftRaw : List Term := Proof.Events868.exact222245RawTerms
def rightRaw : List Term := Proof.Events877.exact224655RawTerms
def group : MergeGroup := .operator 222245 224655
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 222245) (leftOrdinal := 0)
    (rightResult := 224655) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨35179⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35179⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge224661

namespace LeftMerge224740
def owner : Owner := ⟨.program ⟨257⟩, ⟨34411⟩⟩
def mergeEvent : Nat := 224740
def frameStart : Nat := 224710
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨13566⟩⟩, ⟨.program ⟨257⟩, ⟨34410⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events877.exact224736RawTerms
def rightRaw : List Term := Proof.Events877.exact224733RawTerms
def group : MergeGroup := .operator 224736 224733
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 224736) (leftOrdinal := 0)
    (rightResult := 224733) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13566⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨34410⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨13566⟩⟩, ⟨.program ⟨257⟩, ⟨34410⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge224740

namespace LeftMerge224770
def owner : Owner := ⟨.program ⟨257⟩, ⟨36024⟩⟩
def mergeEvent : Nat := 224770
def frameStart : Nat := 224710
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨13566⟩⟩, ⟨.program ⟨257⟩, ⟨34410⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events877.exact224766RawTerms
def rightRaw : List Term := Proof.Events877.exact224764RawTerms
def group : MergeGroup := .operator 224766 224764
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 224766) (leftOrdinal := 0)
    (rightResult := 224764) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13566⟩⟩, ⟨.program ⟨257⟩, ⟨34410⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨13566⟩⟩, ⟨.program ⟨257⟩, ⟨34410⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge224770

namespace LeftMerge224793
def owner : Owner := ⟨.program ⟨257⟩, ⟨9552⟩⟩
def mergeEvent : Nat := 224793
def frameStart : Nat := 224710
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩] } }
def leftRaw : List Term := Proof.Events878.exact224789RawTerms
def rightRaw : List Term := Proof.Events878.exact224786RawTerms
def group : MergeGroup := .operator 224789 224786
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 224789) (leftOrdinal := 0)
    (rightResult := 224786) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7297⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9550⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge224793

namespace LeftMerge224802
def owner : Owner := ⟨.program ⟨257⟩, ⟨36251⟩⟩
def mergeEvent : Nat := 224802
def frameStart : Nat := 224710
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩, ⟨.program ⟨257⟩, ⟨36248⟩⟩] } }
def leftRaw : List Term := Proof.Events878.exact224798RawTerms
def rightRaw : List Term := Proof.Events877.exact224755RawTerms
def group : MergeGroup := .operator 224798 224755
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 224798) (leftOrdinal := 0)
    (rightResult := 224755) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨36248⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩, ⟨.program ⟨257⟩, ⟨36248⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge224802

namespace LeftMerge224803
def owner : Owner := ⟨.program ⟨257⟩, ⟨36251⟩⟩
def mergeEvent : Nat := 224803
def frameStart : Nat := 224710
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨13566⟩⟩, ⟨.program ⟨257⟩, ⟨34410⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36248⟩⟩] } }
def leftRaw : List Term := Proof.Events878.exact224798RawTerms
def rightRaw : List Term := Proof.Events877.exact224755RawTerms
def group : MergeGroup := .operator 224798 224755
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 224798) (leftOrdinal := 1)
    (rightResult := 224755) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13566⟩⟩, ⟨.program ⟨257⟩, ⟨34410⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨36248⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨13566⟩⟩, ⟨.program ⟨257⟩, ⟨34410⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36248⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge224803

namespace LeftMerge224805
def owner : Owner := ⟨.program ⟨257⟩, ⟨36251⟩⟩
def mergeEvent : Nat := 224805
def frameStart : Nat := 224710
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨13566⟩⟩, ⟨.program ⟨257⟩, ⟨34410⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35743⟩⟩] } }
def rhsRaw : List Term := Proof.Events877.exact224752RawTerms
def group : MergeGroup := .relation 224804
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 224804) (rhsResult := 224752)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨13566⟩⟩, ⟨.program ⟨257⟩, ⟨34410⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36248⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨36248⟩⟩) ⟨35743⟩ 224752) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨35743⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨13566⟩⟩, ⟨.program ⟨257⟩, ⟨34410⟩⟩], [⟨.program ⟨257⟩, ⟨35743⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge224805

namespace LeftMerge224813
def owner : Owner := ⟨.program ⟨257⟩, ⟨34742⟩⟩
def mergeEvent : Nat := 224813
def frameStart : Nat := 224710
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨34740⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events877.exact224766RawTerms
def rightRaw : List Term := Proof.Events878.exact224809RawTerms
def group : MergeGroup := .operator 224766 224809
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 224766) (leftOrdinal := 0)
    (rightResult := 224809) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨34740⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨34740⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge224813

namespace LeftMerge224830
def owner : Owner := ⟨.program ⟨257⟩, ⟨35182⟩⟩
def mergeEvent : Nat := 224830
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7191⟩⟩] } }
def rhsRaw : List Term := Proof.Events878.exact224827RawTerms
def group : MergeGroup := .relation 224829
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 224829) (rhsResult := 224827)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35179⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 224828 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35179⟩⟩]⟩) (none) 224827) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7191⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7191⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge224830

namespace LeftMerge224831
def owner : Owner := ⟨.program ⟨257⟩, ⟨35182⟩⟩
def mergeEvent : Nat := 224831
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩, ⟨.program ⟨257⟩, ⟨36248⟩⟩] } }
def rhsRaw : List Term := Proof.Events878.exact224827RawTerms
def group : MergeGroup := .relation 224829
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 224829) (rhsResult := 224827)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35179⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 224828 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35179⟩⟩]⟩) (none) 224827) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩, ⟨.program ⟨257⟩, ⟨36248⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩, ⟨.program ⟨257⟩, ⟨36248⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge224831

namespace LeftMerge224832
def owner : Owner := ⟨.program ⟨257⟩, ⟨35182⟩⟩
def mergeEvent : Nat := 224832
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨13566⟩⟩, ⟨.program ⟨257⟩, ⟨34410⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35743⟩⟩] } }
def rhsRaw : List Term := Proof.Events878.exact224827RawTerms
def group : MergeGroup := .relation 224829
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 224829) (rhsResult := 224827)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35179⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 224828 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35179⟩⟩]⟩) (none) 224827) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13566⟩⟩, ⟨.program ⟨257⟩, ⟨34410⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35743⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨13566⟩⟩, ⟨.program ⟨257⟩, ⟨34410⟩⟩], [⟨.program ⟨257⟩, ⟨35743⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge224832

namespace LeftMerge224833
def owner : Owner := ⟨.program ⟨257⟩, ⟨35182⟩⟩
def mergeEvent : Nat := 224833
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨34740⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events878.exact224827RawTerms
def group : MergeGroup := .relation 224829
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 224829) (rhsResult := 224827)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35179⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 224828 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35179⟩⟩]⟩) (none) 224827) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨34740⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨34740⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge224833

namespace LeftMerge224838
def owner : Owner := ⟨.program ⟨257⟩, ⟨36250⟩⟩
def mergeEvent : Nat := 224838
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨13566⟩⟩, ⟨.program ⟨257⟩, ⟨34410⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35743⟩⟩] } }
def leftRaw : List Term := Proof.Events878.exact224834RawTerms
def rightRaw : List Term := Proof.Events877.exact224648RawTerms
def group : MergeGroup := .operator 224834 224648
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 224834) (leftOrdinal := 2)
    (rightResult := 224648) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨13566⟩⟩, ⟨.program ⟨257⟩, ⟨34410⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35743⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨13566⟩⟩, ⟨.program ⟨257⟩, ⟨34410⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35743⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨13566⟩⟩, ⟨.program ⟨257⟩, ⟨34410⟩⟩], [⟨.program ⟨257⟩, ⟨35743⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge224838

namespace LeftMerge224839
def owner : Owner := ⟨.program ⟨257⟩, ⟨36250⟩⟩
def mergeEvent : Nat := 224839
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩, ⟨.program ⟨257⟩, ⟨36248⟩⟩] } }
def leftRaw : List Term := Proof.Events878.exact224834RawTerms
def rightRaw : List Term := Proof.Events877.exact224648RawTerms
def group : MergeGroup := .operator 224834 224648
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 224834) (leftOrdinal := 1)
    (rightResult := 224648) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩, ⟨.program ⟨257⟩, ⟨36248⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩, ⟨.program ⟨257⟩, ⟨36248⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩, ⟨.program ⟨257⟩, ⟨36248⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge224839

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
