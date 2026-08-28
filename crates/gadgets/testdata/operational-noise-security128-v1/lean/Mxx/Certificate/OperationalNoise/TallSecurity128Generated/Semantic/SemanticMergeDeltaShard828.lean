import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge135947
def owner : Owner := ⟨.program ⟨257⟩, ⟨40482⟩⟩
def mergeEvent : Nat := 135947
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40479⟩⟩] } }
def leftRaw : List Term := Proof.Events525.exact134495RawTerms
def rightRaw : List Term := Proof.Events531.exact135941RawTerms
def group : MergeGroup := .operator 134495 135941
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 134495) (leftOrdinal := 0)
    (rightResult := 135941) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨40479⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40479⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge135947

namespace LeftMerge136026
def owner : Owner := ⟨.program ⟨257⟩, ⟨39627⟩⟩
def mergeEvent : Nat := 136026
def frameStart : Nat := 135996
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14076⟩⟩, ⟨.program ⟨257⟩, ⟨39626⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events531.exact136022RawTerms
def rightRaw : List Term := Proof.Events531.exact136019RawTerms
def group : MergeGroup := .operator 136022 136019
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 136022) (leftOrdinal := 0)
    (rightResult := 136019) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14076⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨39626⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨14076⟩⟩, ⟨.program ⟨257⟩, ⟨39626⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge136026

namespace LeftMerge136056
def owner : Owner := ⟨.program ⟨257⟩, ⟨41360⟩⟩
def mergeEvent : Nat := 136056
def frameStart : Nat := 135996
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14076⟩⟩, ⟨.program ⟨257⟩, ⟨39626⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events531.exact136052RawTerms
def rightRaw : List Term := Proof.Events531.exact136050RawTerms
def group : MergeGroup := .operator 136052 136050
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 136052) (leftOrdinal := 0)
    (rightResult := 136050) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14076⟩⟩, ⟨.program ⟨257⟩, ⟨39626⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨14076⟩⟩, ⟨.program ⟨257⟩, ⟨39626⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge136056

namespace LeftMerge136079
def owner : Owner := ⟨.program ⟨257⟩, ⟨9558⟩⟩
def mergeEvent : Nat := 136079
def frameStart : Nat := 135996
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩] } }
def leftRaw : List Term := Proof.Events531.exact136075RawTerms
def rightRaw : List Term := Proof.Events531.exact136072RawTerms
def group : MergeGroup := .operator 136075 136072
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 136075) (leftOrdinal := 0)
    (rightResult := 136072) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9556⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge136079

namespace LeftMerge136088
def owner : Owner := ⟨.program ⟨257⟩, ⟨41545⟩⟩
def mergeEvent : Nat := 136088
def frameStart : Nat := 135996
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41542⟩⟩] } }
def leftRaw : List Term := Proof.Events531.exact136084RawTerms
def rightRaw : List Term := Proof.Events531.exact136041RawTerms
def group : MergeGroup := .operator 136084 136041
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 136084) (leftOrdinal := 0)
    (rightResult := 136041) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨41542⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41542⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge136088

namespace LeftMerge136089
def owner : Owner := ⟨.program ⟨257⟩, ⟨41545⟩⟩
def mergeEvent : Nat := 136089
def frameStart : Nat := 135996
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14076⟩⟩, ⟨.program ⟨257⟩, ⟨39626⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨41542⟩⟩] } }
def leftRaw : List Term := Proof.Events531.exact136084RawTerms
def rightRaw : List Term := Proof.Events531.exact136041RawTerms
def group : MergeGroup := .operator 136084 136041
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 136084) (leftOrdinal := 1)
    (rightResult := 136041) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14076⟩⟩, ⟨.program ⟨257⟩, ⟨39626⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨41542⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨14076⟩⟩, ⟨.program ⟨257⟩, ⟨39626⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨41542⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge136089

namespace LeftMerge136091
def owner : Owner := ⟨.program ⟨257⟩, ⟨41545⟩⟩
def mergeEvent : Nat := 136091
def frameStart : Nat := 135996
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14076⟩⟩, ⟨.program ⟨257⟩, ⟨39626⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41067⟩⟩] } }
def rhsRaw : List Term := Proof.Events531.exact136038RawTerms
def group : MergeGroup := .relation 136090
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 136090) (rhsResult := 136038)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨14076⟩⟩, ⟨.program ⟨257⟩, ⟨39626⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨41542⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨41542⟩⟩) ⟨41067⟩ 136038) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨41067⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨14076⟩⟩, ⟨.program ⟨257⟩, ⟨39626⟩⟩], [⟨.program ⟨257⟩, ⟨41067⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge136091

namespace LeftMerge136099
def owner : Owner := ⟨.program ⟨257⟩, ⟨40054⟩⟩
def mergeEvent : Nat := 136099
def frameStart : Nat := 135996
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨40052⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events531.exact136052RawTerms
def rightRaw : List Term := Proof.Events531.exact136095RawTerms
def group : MergeGroup := .operator 136052 136095
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 136052) (leftOrdinal := 0)
    (rightResult := 136095) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨40052⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨40052⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge136099

namespace LeftMerge136116
def owner : Owner := ⟨.program ⟨257⟩, ⟨40482⟩⟩
def mergeEvent : Nat := 136116
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7193⟩⟩] } }
def rhsRaw : List Term := Proof.Events531.exact136113RawTerms
def group : MergeGroup := .relation 136115
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 136115) (rhsResult := 136113)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40479⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 136114 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40479⟩⟩]⟩) (none) 136113) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7193⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7193⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge136116

namespace LeftMerge136117
def owner : Owner := ⟨.program ⟨257⟩, ⟨40482⟩⟩
def mergeEvent : Nat := 136117
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41542⟩⟩] } }
def rhsRaw : List Term := Proof.Events531.exact136113RawTerms
def group : MergeGroup := .relation 136115
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 136115) (rhsResult := 136113)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40479⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 136114 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40479⟩⟩]⟩) (none) 136113) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41542⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41542⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge136117

namespace LeftMerge136118
def owner : Owner := ⟨.program ⟨257⟩, ⟨40482⟩⟩
def mergeEvent : Nat := 136118
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨14076⟩⟩, ⟨.program ⟨257⟩, ⟨39626⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41067⟩⟩] } }
def rhsRaw : List Term := Proof.Events531.exact136113RawTerms
def group : MergeGroup := .relation 136115
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 136115) (rhsResult := 136113)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40479⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 136114 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40479⟩⟩]⟩) (none) 136113) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14076⟩⟩, ⟨.program ⟨257⟩, ⟨39626⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41067⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨14076⟩⟩, ⟨.program ⟨257⟩, ⟨39626⟩⟩], [⟨.program ⟨257⟩, ⟨41067⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge136118

namespace LeftMerge136119
def owner : Owner := ⟨.program ⟨257⟩, ⟨40482⟩⟩
def mergeEvent : Nat := 136119
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨40052⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events531.exact136113RawTerms
def group : MergeGroup := .relation 136115
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 136115) (rhsResult := 136113)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40479⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 136114 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40479⟩⟩]⟩) (none) 136113) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨40052⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨40052⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge136119

namespace LeftMerge136124
def owner : Owner := ⟨.program ⟨257⟩, ⟨41544⟩⟩
def mergeEvent : Nat := 136124
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨14076⟩⟩, ⟨.program ⟨257⟩, ⟨39626⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41067⟩⟩] } }
def leftRaw : List Term := Proof.Events531.exact136120RawTerms
def rightRaw : List Term := Proof.Events530.exact135934RawTerms
def group : MergeGroup := .operator 136120 135934
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 136120) (leftOrdinal := 2)
    (rightResult := 135934) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨14076⟩⟩, ⟨.program ⟨257⟩, ⟨39626⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41067⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨14076⟩⟩, ⟨.program ⟨257⟩, ⟨39626⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41067⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨14076⟩⟩, ⟨.program ⟨257⟩, ⟨39626⟩⟩], [⟨.program ⟨257⟩, ⟨41067⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge136124

namespace LeftMerge136125
def owner : Owner := ⟨.program ⟨257⟩, ⟨41544⟩⟩
def mergeEvent : Nat := 136125
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41542⟩⟩] } }
def leftRaw : List Term := Proof.Events531.exact136120RawTerms
def rightRaw : List Term := Proof.Events530.exact135934RawTerms
def group : MergeGroup := .operator 136120 135934
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 136120) (leftOrdinal := 1)
    (rightResult := 135934) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41542⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41542⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41542⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge136125

namespace LeftMerge136133
def owner : Owner := ⟨.program ⟨257⟩, ⟨41816⟩⟩
def mergeEvent : Nat := 136133
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨41814⟩⟩] } }
def leftRaw : List Term := Proof.Events531.exact136127RawTerms
def rightRaw : List Term := Proof.Events530.exact135850RawTerms
def group : MergeGroup := .operator 136127 135850
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 136127) (leftOrdinal := 0)
    (rightResult := 135850) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7193⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨41814⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨41814⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge136133

namespace LeftMerge136134
def owner : Owner := ⟨.program ⟨257⟩, ⟨41816⟩⟩
def mergeEvent : Nat := 136134
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨40052⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨41814⟩⟩] } }
def leftRaw : List Term := Proof.Events531.exact136127RawTerms
def rightRaw : List Term := Proof.Events530.exact135850RawTerms
def group : MergeGroup := .operator 136127 135850
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 136127) (leftOrdinal := 1)
    (rightResult := 135850) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨40052⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨41814⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨40052⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨41814⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge136134

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
