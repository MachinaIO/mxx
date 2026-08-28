import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge269001
def owner : Owner := ⟨.program ⟨257⟩, ⟨30509⟩⟩
def mergeEvent : Nat := 269001
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨13156⟩⟩, ⟨.program ⟨257⟩, ⟨28574⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨30508⟩⟩] } }
def leftRaw : List Term := Proof.Events1050.exact268995RawTerms
def rightRaw : List Term := Proof.Events1050.exact268931RawTerms
def group : MergeGroup := .operator 268995 268931
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 268995) (leftOrdinal := 1)
    (rightResult := 268931) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨13156⟩⟩, ⟨.program ⟨257⟩, ⟨28574⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨30508⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨13156⟩⟩, ⟨.program ⟨257⟩, ⟨28574⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨30508⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge269001

namespace LeftMerge269003
def owner : Owner := ⟨.program ⟨257⟩, ⟨30509⟩⟩
def mergeEvent : Nat := 269003
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨13156⟩⟩, ⟨.program ⟨257⟩, ⟨28574⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨30039⟩⟩] } }
def rhsRaw : List Term := Proof.Events1050.exact268928RawTerms
def group : MergeGroup := .relation 269002
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 269002) (rhsResult := 268928)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨13156⟩⟩, ⟨.program ⟨257⟩, ⟨28574⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨30508⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨30508⟩⟩) ⟨30039⟩ 268928) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨30039⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨13156⟩⟩, ⟨.program ⟨257⟩, ⟨28574⟩⟩], [⟨.program ⟨257⟩, ⟨30039⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge269003

namespace LeftMerge269004
def owner : Owner := ⟨.program ⟨257⟩, ⟨30509⟩⟩
def mergeEvent : Nat := 269004
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩, ⟨.program ⟨257⟩, ⟨30508⟩⟩] } }
def leftRaw : List Term := Proof.Events1050.exact268995RawTerms
def rightRaw : List Term := Proof.Events1050.exact268931RawTerms
def group : MergeGroup := .operator 268995 268931
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 268995) (leftOrdinal := 0)
    (rightResult := 268931) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨30508⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩, ⟨.program ⟨257⟩, ⟨30508⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge269004

namespace LeftMerge269018
def owner : Owner := ⟨.program ⟨257⟩, ⟨29449⟩⟩
def mergeEvent : Nat := 269018
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29446⟩⟩] } }
def leftRaw : List Term := Proof.Events1039.exact266120RawTerms
def rightRaw : List Term := Proof.Events1050.exact269012RawTerms
def group : MergeGroup := .operator 266120 269012
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 266120) (leftOrdinal := 0)
    (rightResult := 269012) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨29446⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29446⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge269018

namespace LeftMerge269097
def owner : Owner := ⟨.program ⟨257⟩, ⟨28575⟩⟩
def mergeEvent : Nat := 269097
def frameStart : Nat := 269067
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨13156⟩⟩, ⟨.program ⟨257⟩, ⟨28574⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events1051.exact269093RawTerms
def rightRaw : List Term := Proof.Events1051.exact269090RawTerms
def group : MergeGroup := .operator 269093 269090
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 269093) (leftOrdinal := 0)
    (rightResult := 269090) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13156⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨28574⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨13156⟩⟩, ⟨.program ⟨257⟩, ⟨28574⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge269097

namespace LeftMerge269127
def owner : Owner := ⟨.program ⟨257⟩, ⟨30336⟩⟩
def mergeEvent : Nat := 269127
def frameStart : Nat := 269067
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨13156⟩⟩, ⟨.program ⟨257⟩, ⟨28574⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1051.exact269123RawTerms
def rightRaw : List Term := Proof.Events1051.exact269121RawTerms
def group : MergeGroup := .operator 269123 269121
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 269123) (leftOrdinal := 0)
    (rightResult := 269121) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13156⟩⟩, ⟨.program ⟨257⟩, ⟨28574⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨13156⟩⟩, ⟨.program ⟨257⟩, ⟨28574⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge269127

namespace LeftMerge269150
def owner : Owner := ⟨.program ⟨257⟩, ⟨9549⟩⟩
def mergeEvent : Nat := 269150
def frameStart : Nat := 269067
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩] } }
def leftRaw : List Term := Proof.Events1051.exact269146RawTerms
def rightRaw : List Term := Proof.Events1051.exact269143RawTerms
def group : MergeGroup := .operator 269146 269143
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 269146) (leftOrdinal := 0)
    (rightResult := 269143) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7296⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9547⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge269150

namespace LeftMerge269159
def owner : Owner := ⟨.program ⟨257⟩, ⟨30511⟩⟩
def mergeEvent : Nat := 269159
def frameStart : Nat := 269067
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩, ⟨.program ⟨257⟩, ⟨30508⟩⟩] } }
def leftRaw : List Term := Proof.Events1051.exact269155RawTerms
def rightRaw : List Term := Proof.Events1051.exact269112RawTerms
def group : MergeGroup := .operator 269155 269112
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 269155) (leftOrdinal := 0)
    (rightResult := 269112) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨30508⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩, ⟨.program ⟨257⟩, ⟨30508⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge269159

namespace LeftMerge269160
def owner : Owner := ⟨.program ⟨257⟩, ⟨30511⟩⟩
def mergeEvent : Nat := 269160
def frameStart : Nat := 269067
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨13156⟩⟩, ⟨.program ⟨257⟩, ⟨28574⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨30508⟩⟩] } }
def leftRaw : List Term := Proof.Events1051.exact269155RawTerms
def rightRaw : List Term := Proof.Events1051.exact269112RawTerms
def group : MergeGroup := .operator 269155 269112
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 269155) (leftOrdinal := 1)
    (rightResult := 269112) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13156⟩⟩, ⟨.program ⟨257⟩, ⟨28574⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨30508⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨13156⟩⟩, ⟨.program ⟨257⟩, ⟨28574⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨30508⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge269160

namespace LeftMerge269162
def owner : Owner := ⟨.program ⟨257⟩, ⟨30511⟩⟩
def mergeEvent : Nat := 269162
def frameStart : Nat := 269067
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨13156⟩⟩, ⟨.program ⟨257⟩, ⟨28574⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨30039⟩⟩] } }
def rhsRaw : List Term := Proof.Events1051.exact269109RawTerms
def group : MergeGroup := .relation 269161
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 269161) (rhsResult := 269109)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨13156⟩⟩, ⟨.program ⟨257⟩, ⟨28574⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨30508⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨30508⟩⟩) ⟨30039⟩ 269109) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨30039⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨13156⟩⟩, ⟨.program ⟨257⟩, ⟨28574⟩⟩], [⟨.program ⟨257⟩, ⟨30039⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge269162

namespace LeftMerge269170
def owner : Owner := ⟨.program ⟨257⟩, ⟨29024⟩⟩
def mergeEvent : Nat := 269170
def frameStart : Nat := 269067
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨29022⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1051.exact269123RawTerms
def rightRaw : List Term := Proof.Events1051.exact269166RawTerms
def group : MergeGroup := .operator 269123 269166
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 269123) (leftOrdinal := 0)
    (rightResult := 269166) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨29022⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨29022⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge269170

namespace LeftMerge269187
def owner : Owner := ⟨.program ⟨257⟩, ⟨29449⟩⟩
def mergeEvent : Nat := 269187
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7190⟩⟩] } }
def rhsRaw : List Term := Proof.Events1051.exact269184RawTerms
def group : MergeGroup := .relation 269186
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 269186) (rhsResult := 269184)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29446⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 269185 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29446⟩⟩]⟩) (none) 269184) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7190⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7190⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge269187

namespace LeftMerge269188
def owner : Owner := ⟨.program ⟨257⟩, ⟨29449⟩⟩
def mergeEvent : Nat := 269188
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩, ⟨.program ⟨257⟩, ⟨30508⟩⟩] } }
def rhsRaw : List Term := Proof.Events1051.exact269184RawTerms
def group : MergeGroup := .relation 269186
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 269186) (rhsResult := 269184)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29446⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 269185 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29446⟩⟩]⟩) (none) 269184) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩, ⟨.program ⟨257⟩, ⟨30508⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩, ⟨.program ⟨257⟩, ⟨30508⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge269188

namespace LeftMerge269189
def owner : Owner := ⟨.program ⟨257⟩, ⟨29449⟩⟩
def mergeEvent : Nat := 269189
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨13156⟩⟩, ⟨.program ⟨257⟩, ⟨28574⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨30039⟩⟩] } }
def rhsRaw : List Term := Proof.Events1051.exact269184RawTerms
def group : MergeGroup := .relation 269186
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 269186) (rhsResult := 269184)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29446⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 269185 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29446⟩⟩]⟩) (none) 269184) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13156⟩⟩, ⟨.program ⟨257⟩, ⟨28574⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨30039⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨13156⟩⟩, ⟨.program ⟨257⟩, ⟨28574⟩⟩], [⟨.program ⟨257⟩, ⟨30039⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge269189

namespace LeftMerge269190
def owner : Owner := ⟨.program ⟨257⟩, ⟨29449⟩⟩
def mergeEvent : Nat := 269190
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨29022⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events1051.exact269184RawTerms
def group : MergeGroup := .relation 269186
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 269186) (rhsResult := 269184)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29446⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 269185 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29446⟩⟩]⟩) (none) 269184) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨29022⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨29022⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge269190

namespace LeftMerge269195
def owner : Owner := ⟨.program ⟨257⟩, ⟨30510⟩⟩
def mergeEvent : Nat := 269195
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨13156⟩⟩, ⟨.program ⟨257⟩, ⟨28574⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨30039⟩⟩] } }
def leftRaw : List Term := Proof.Events1051.exact269191RawTerms
def rightRaw : List Term := Proof.Events1050.exact269005RawTerms
def group : MergeGroup := .operator 269191 269005
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 269191) (leftOrdinal := 2)
    (rightResult := 269005) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨13156⟩⟩, ⟨.program ⟨257⟩, ⟨28574⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨30039⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨13156⟩⟩, ⟨.program ⟨257⟩, ⟨28574⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨30039⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨13156⟩⟩, ⟨.program ⟨257⟩, ⟨28574⟩⟩], [⟨.program ⟨257⟩, ⟨30039⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge269195

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
