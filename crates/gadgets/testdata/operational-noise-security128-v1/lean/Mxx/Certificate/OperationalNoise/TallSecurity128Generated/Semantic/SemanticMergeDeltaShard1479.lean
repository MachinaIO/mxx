import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge239920
def owner : Owner := ⟨.program ⟨257⟩, ⟨29074⟩⟩
def mergeEvent : Nat := 239920
def frameStart : Nat := 239817
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨29072⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events937.exact239873RawTerms
def rightRaw : List Term := Proof.Events937.exact239916RawTerms
def group : MergeGroup := .operator 239873 239916
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 239873) (leftOrdinal := 0)
    (rightResult := 239916) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨29072⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨29072⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge239920

namespace LeftMerge239937
def owner : Owner := ⟨.program ⟨257⟩, ⟨29512⟩⟩
def mergeEvent : Nat := 239937
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7190⟩⟩] } }
def rhsRaw : List Term := Proof.Events937.exact239934RawTerms
def group : MergeGroup := .relation 239936
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 239936) (rhsResult := 239934)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29509⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 239935 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29509⟩⟩]⟩) (none) 239934) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7190⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7190⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge239937

namespace LeftMerge239938
def owner : Owner := ⟨.program ⟨257⟩, ⟨29512⟩⟩
def mergeEvent : Nat := 239938
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩, ⟨.program ⟨257⟩, ⟨30577⟩⟩] } }
def rhsRaw : List Term := Proof.Events937.exact239934RawTerms
def group : MergeGroup := .relation 239936
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 239936) (rhsResult := 239934)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29509⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 239935 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29509⟩⟩]⟩) (none) 239934) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩, ⟨.program ⟨257⟩, ⟨30577⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩, ⟨.program ⟨257⟩, ⟨30577⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge239938

namespace LeftMerge239939
def owner : Owner := ⟨.program ⟨257⟩, ⟨29512⟩⟩
def mergeEvent : Nat := 239939
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨13251⟩⟩, ⟨.program ⟨257⟩, ⟨28726⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨30077⟩⟩] } }
def rhsRaw : List Term := Proof.Events937.exact239934RawTerms
def group : MergeGroup := .relation 239936
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 239936) (rhsResult := 239934)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29509⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 239935 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29509⟩⟩]⟩) (none) 239934) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13251⟩⟩, ⟨.program ⟨257⟩, ⟨28726⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨30077⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨13251⟩⟩, ⟨.program ⟨257⟩, ⟨28726⟩⟩], [⟨.program ⟨257⟩, ⟨30077⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge239939

namespace LeftMerge239940
def owner : Owner := ⟨.program ⟨257⟩, ⟨29512⟩⟩
def mergeEvent : Nat := 239940
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨29072⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events937.exact239934RawTerms
def group : MergeGroup := .relation 239936
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 239936) (rhsResult := 239934)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29509⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 239935 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29509⟩⟩]⟩) (none) 239934) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨29072⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨29072⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge239940

namespace LeftMerge239945
def owner : Owner := ⟨.program ⟨257⟩, ⟨30579⟩⟩
def mergeEvent : Nat := 239945
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨13251⟩⟩, ⟨.program ⟨257⟩, ⟨28726⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨30077⟩⟩] } }
def leftRaw : List Term := Proof.Events937.exact239941RawTerms
def rightRaw : List Term := Proof.Events936.exact239755RawTerms
def group : MergeGroup := .operator 239941 239755
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 239941) (leftOrdinal := 2)
    (rightResult := 239755) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨13251⟩⟩, ⟨.program ⟨257⟩, ⟨28726⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨30077⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨13251⟩⟩, ⟨.program ⟨257⟩, ⟨28726⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨30077⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨13251⟩⟩, ⟨.program ⟨257⟩, ⟨28726⟩⟩], [⟨.program ⟨257⟩, ⟨30077⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge239945

namespace LeftMerge239946
def owner : Owner := ⟨.program ⟨257⟩, ⟨30579⟩⟩
def mergeEvent : Nat := 239946
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩, ⟨.program ⟨257⟩, ⟨30577⟩⟩] } }
def leftRaw : List Term := Proof.Events937.exact239941RawTerms
def rightRaw : List Term := Proof.Events936.exact239755RawTerms
def group : MergeGroup := .operator 239941 239755
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 239941) (leftOrdinal := 1)
    (rightResult := 239755) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩, ⟨.program ⟨257⟩, ⟨30577⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩, ⟨.program ⟨257⟩, ⟨30577⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩, ⟨.program ⟨257⟩, ⟨30577⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge239946

namespace LeftMerge239954
def owner : Owner := ⟨.program ⟨257⟩, ⟨30921⟩⟩
def mergeEvent : Nat := 239954
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨30919⟩⟩] } }
def leftRaw : List Term := Proof.Events937.exact239948RawTerms
def rightRaw : List Term := Proof.Events936.exact239671RawTerms
def group : MergeGroup := .operator 239948 239671
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 239948) (leftOrdinal := 0)
    (rightResult := 239671) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7190⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨30919⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨30919⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge239954

namespace LeftMerge239955
def owner : Owner := ⟨.program ⟨257⟩, ⟨30921⟩⟩
def mergeEvent : Nat := 239955
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨29072⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨30919⟩⟩] } }
def leftRaw : List Term := Proof.Events937.exact239948RawTerms
def rightRaw : List Term := Proof.Events936.exact239671RawTerms
def group : MergeGroup := .operator 239948 239671
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 239948) (leftOrdinal := 1)
    (rightResult := 239671) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨29072⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨30919⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨29072⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨30919⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge239955

namespace LeftMerge239957
def owner : Owner := ⟨.program ⟨257⟩, ⟨30921⟩⟩
def mergeEvent : Nat := 239957
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨29072⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨30223⟩⟩] } }
def rhsRaw : List Term := Proof.Events936.exact239668RawTerms
def group : MergeGroup := .relation 239956
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 239956) (rhsResult := 239668)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨29072⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨30919⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨30919⟩⟩) ⟨30223⟩ 239668) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨30223⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨29072⟩⟩], [⟨.program ⟨257⟩, ⟨30223⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge239957

namespace LeftMerge239971
def owner : Owner := ⟨.program ⟨257⟩, ⟨29799⟩⟩
def mergeEvent : Nat := 239971
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29796⟩⟩] } }
def leftRaw : List Term := Proof.Events925.exact236870RawTerms
def rightRaw : List Term := Proof.Events937.exact239965RawTerms
def group : MergeGroup := .operator 236870 239965
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 236870) (leftOrdinal := 0)
    (rightResult := 239965) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨29796⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29796⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge239971

namespace LeftMerge240092
def owner : Owner := ⟨.program ⟨257⟩, ⟨30440⟩⟩
def mergeEvent : Nat := 240092
def frameStart : Nat := 240026
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨29072⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events937.exact240088RawTerms
def rightRaw : List Term := Proof.Events937.exact240086RawTerms
def group : MergeGroup := .operator 240088 240086
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 240088) (leftOrdinal := 0)
    (rightResult := 240086) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨29072⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨29072⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge240092

namespace LeftMerge240104
def owner : Owner := ⟨.program ⟨257⟩, ⟨30920⟩⟩
def mergeEvent : Nat := 240104
def frameStart : Nat := 240026
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨30919⟩⟩] } }
def leftRaw : List Term := Proof.Events937.exact240100RawTerms
def rightRaw : List Term := Proof.Events937.exact240077RawTerms
def group : MergeGroup := .operator 240100 240077
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 240100) (leftOrdinal := 0)
    (rightResult := 240077) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7190⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨30919⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨30919⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge240104

namespace LeftMerge240105
def owner : Owner := ⟨.program ⟨257⟩, ⟨30920⟩⟩
def mergeEvent : Nat := 240105
def frameStart : Nat := 240026
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨29072⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨30919⟩⟩] } }
def leftRaw : List Term := Proof.Events937.exact240100RawTerms
def rightRaw : List Term := Proof.Events937.exact240077RawTerms
def group : MergeGroup := .operator 240100 240077
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 240100) (leftOrdinal := 1)
    (rightResult := 240077) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨29072⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨30919⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨29072⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨30919⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge240105

namespace LeftMerge240107
def owner : Owner := ⟨.program ⟨257⟩, ⟨30920⟩⟩
def mergeEvent : Nat := 240107
def frameStart : Nat := 240026
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨29072⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨30223⟩⟩] } }
def rhsRaw : List Term := Proof.Events937.exact240074RawTerms
def group : MergeGroup := .relation 240106
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 240106) (rhsResult := 240074)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨29072⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨30919⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨30919⟩⟩) ⟨30223⟩ 240074) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨30223⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨29072⟩⟩], [⟨.program ⟨257⟩, ⟨30223⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge240107

namespace LeftMerge240115
def owner : Owner := ⟨.program ⟨257⟩, ⟨29274⟩⟩
def mergeEvent : Nat := 240115
def frameStart : Nat := 240026
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨29273⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events937.exact240088RawTerms
def rightRaw : List Term := Proof.Events937.exact240111RawTerms
def group : MergeGroup := .operator 240088 240111
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 240088) (leftOrdinal := 0)
    (rightResult := 240111) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨29273⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨29273⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge240115

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
