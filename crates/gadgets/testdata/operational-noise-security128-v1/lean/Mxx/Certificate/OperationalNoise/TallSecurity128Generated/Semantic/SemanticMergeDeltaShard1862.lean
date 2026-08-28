import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge300963
def owner : Owner := ⟨.program ⟨257⟩, ⟨52412⟩⟩
def mergeEvent : Nat := 300963
def frameStart : Nat := 300880
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24410⟩⟩, ⟨.program ⟨257⟩, ⟨50275⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨51949⟩⟩] } }
def rhsRaw : List Term := Proof.Events1175.exact300910RawTerms
def group : MergeGroup := .relation 300962
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 300962) (rhsResult := 300910)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨24410⟩⟩, ⟨.program ⟨257⟩, ⟨50275⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨52409⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨52409⟩⟩) ⟨51949⟩ 300910) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨51949⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨24410⟩⟩, ⟨.program ⟨257⟩, ⟨50275⟩⟩], [⟨.program ⟨257⟩, ⟨51949⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge300963

namespace LeftMerge300971
def owner : Owner := ⟨.program ⟨257⟩, ⟨50810⟩⟩
def mergeEvent : Nat := 300971
def frameStart : Nat := 300880
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨50808⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1175.exact300924RawTerms
def rightRaw : List Term := Proof.Events1175.exact300967RawTerms
def group : MergeGroup := .operator 300924 300967
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 300924) (leftOrdinal := 0)
    (rightResult := 300967) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨50808⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨50808⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge300971

namespace LeftMerge300988
def owner : Owner := ⟨.program ⟨257⟩, ⟨51352⟩⟩
def mergeEvent : Nat := 300988
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7183⟩⟩] } }
def rhsRaw : List Term := Proof.Events1175.exact300985RawTerms
def group : MergeGroup := .relation 300987
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 300987) (rhsResult := 300985)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51349⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 300986 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51349⟩⟩]⟩) (none) 300985) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7183⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7183⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge300988

namespace LeftMerge300989
def owner : Owner := ⟨.program ⟨257⟩, ⟨51352⟩⟩
def mergeEvent : Nat := 300989
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52409⟩⟩] } }
def rhsRaw : List Term := Proof.Events1175.exact300985RawTerms
def group : MergeGroup := .relation 300987
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 300987) (rhsResult := 300985)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51349⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 300986 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51349⟩⟩]⟩) (none) 300985) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52409⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52409⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge300989

namespace LeftMerge300990
def owner : Owner := ⟨.program ⟨257⟩, ⟨51352⟩⟩
def mergeEvent : Nat := 300990
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨24410⟩⟩, ⟨.program ⟨257⟩, ⟨50275⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨51949⟩⟩] } }
def rhsRaw : List Term := Proof.Events1175.exact300985RawTerms
def group : MergeGroup := .relation 300987
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 300987) (rhsResult := 300985)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51349⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 300986 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51349⟩⟩]⟩) (none) 300985) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24410⟩⟩, ⟨.program ⟨257⟩, ⟨50275⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨51949⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨24410⟩⟩, ⟨.program ⟨257⟩, ⟨50275⟩⟩], [⟨.program ⟨257⟩, ⟨51949⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge300990

namespace LeftMerge300991
def owner : Owner := ⟨.program ⟨257⟩, ⟨51352⟩⟩
def mergeEvent : Nat := 300991
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨50808⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events1175.exact300985RawTerms
def group : MergeGroup := .relation 300987
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 300987) (rhsResult := 300985)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51349⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 300986 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51349⟩⟩]⟩) (none) 300985) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨50808⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨50808⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge300991

namespace LeftMerge300996
def owner : Owner := ⟨.program ⟨257⟩, ⟨52411⟩⟩
def mergeEvent : Nat := 300996
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨24410⟩⟩, ⟨.program ⟨257⟩, ⟨50275⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨51949⟩⟩] } }
def leftRaw : List Term := Proof.Events1175.exact300992RawTerms
def rightRaw : List Term := Proof.Events1175.exact300830RawTerms
def group : MergeGroup := .operator 300992 300830
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 300992) (leftOrdinal := 2)
    (rightResult := 300830) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨24410⟩⟩, ⟨.program ⟨257⟩, ⟨50275⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨51949⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨24410⟩⟩, ⟨.program ⟨257⟩, ⟨50275⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨51949⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨24410⟩⟩, ⟨.program ⟨257⟩, ⟨50275⟩⟩], [⟨.program ⟨257⟩, ⟨51949⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge300996

namespace LeftMerge300997
def owner : Owner := ⟨.program ⟨257⟩, ⟨52411⟩⟩
def mergeEvent : Nat := 300997
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52409⟩⟩] } }
def leftRaw : List Term := Proof.Events1175.exact300992RawTerms
def rightRaw : List Term := Proof.Events1175.exact300830RawTerms
def group : MergeGroup := .operator 300992 300830
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 300992) (leftOrdinal := 1)
    (rightResult := 300830) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52409⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52409⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52409⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge300997

namespace LeftMerge301005
def owner : Owner := ⟨.program ⟨257⟩, ⟨52644⟩⟩
def mergeEvent : Nat := 301005
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨52642⟩⟩] } }
def leftRaw : List Term := Proof.Events1175.exact300999RawTerms
def rightRaw : List Term := Proof.Events1174.exact300746RawTerms
def group : MergeGroup := .operator 300999 300746
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 300999) (leftOrdinal := 0)
    (rightResult := 300746) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7183⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨52642⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨52642⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge301005

namespace LeftMerge301006
def owner : Owner := ⟨.program ⟨257⟩, ⟨52644⟩⟩
def mergeEvent : Nat := 301006
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨50808⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨52642⟩⟩] } }
def leftRaw : List Term := Proof.Events1175.exact300999RawTerms
def rightRaw : List Term := Proof.Events1174.exact300746RawTerms
def group : MergeGroup := .operator 300999 300746
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 300999) (leftOrdinal := 1)
    (rightResult := 300746) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨50808⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨52642⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨50808⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨52642⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge301006

namespace LeftMerge301008
def owner : Owner := ⟨.program ⟨257⟩, ⟨52644⟩⟩
def mergeEvent : Nat := 301008
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨50808⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨52071⟩⟩] } }
def rhsRaw : List Term := Proof.Events1174.exact300743RawTerms
def group : MergeGroup := .relation 301007
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 301007) (rhsResult := 300743)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨50808⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨52642⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨52642⟩⟩) ⟨52071⟩ 300743) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨52071⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨50808⟩⟩], [⟨.program ⟨257⟩, ⟨52071⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge301008

namespace LeftMerge301022
def owner : Owner := ⟨.program ⟨257⟩, ⟨51559⟩⟩
def mergeEvent : Nat := 301022
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51556⟩⟩] } }
def leftRaw : List Term := Proof.Events1153.exact295195RawTerms
def rightRaw : List Term := Proof.Events1175.exact301016RawTerms
def group : MergeGroup := .operator 295195 301016
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 295195) (leftOrdinal := 0)
    (rightResult := 301016) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨51556⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51556⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge301022

namespace LeftMerge301119
def owner : Owner := ⟨.program ⟨257⟩, ⟨52328⟩⟩
def mergeEvent : Nat := 301119
def frameStart : Nat := 301065
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨50808⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1176.exact301115RawTerms
def rightRaw : List Term := Proof.Events1176.exact301113RawTerms
def group : MergeGroup := .operator 301115 301113
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 301115) (leftOrdinal := 0)
    (rightResult := 301113) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨50808⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨50808⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge301119

namespace LeftMerge301131
def owner : Owner := ⟨.program ⟨257⟩, ⟨52643⟩⟩
def mergeEvent : Nat := 301131
def frameStart : Nat := 301065
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨52642⟩⟩] } }
def leftRaw : List Term := Proof.Events1176.exact301127RawTerms
def rightRaw : List Term := Proof.Events1176.exact301104RawTerms
def group : MergeGroup := .operator 301127 301104
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 301127) (leftOrdinal := 0)
    (rightResult := 301104) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7183⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨52642⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨52642⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge301131

namespace LeftMerge301132
def owner : Owner := ⟨.program ⟨257⟩, ⟨52643⟩⟩
def mergeEvent : Nat := 301132
def frameStart : Nat := 301065
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨50808⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨52642⟩⟩] } }
def leftRaw : List Term := Proof.Events1176.exact301127RawTerms
def rightRaw : List Term := Proof.Events1176.exact301104RawTerms
def group : MergeGroup := .operator 301127 301104
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 301127) (leftOrdinal := 1)
    (rightResult := 301104) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨50808⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨52642⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨50808⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨52642⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge301132

namespace LeftMerge301134
def owner : Owner := ⟨.program ⟨257⟩, ⟨52643⟩⟩
def mergeEvent : Nat := 301134
def frameStart : Nat := 301065
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨50808⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨52071⟩⟩] } }
def rhsRaw : List Term := Proof.Events1176.exact301101RawTerms
def group : MergeGroup := .relation 301133
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 301133) (rhsResult := 301101)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨50808⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨52642⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨52642⟩⟩) ⟨52071⟩ 301101) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨52071⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨50808⟩⟩], [⟨.program ⟨257⟩, ⟨52071⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge301134

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
