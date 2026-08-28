import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge286016
def owner : Owner := ⟨.program ⟨257⟩, ⟨58414⟩⟩
def mergeEvent : Nat := 286016
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨24938⟩⟩, ⟨.program ⟨257⟩, ⟨56343⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨57933⟩⟩] } }
def rhsRaw : List Term := Proof.Events1116.exact285941RawTerms
def group : MergeGroup := .relation 286015
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 286015) (rhsResult := 285941)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨24938⟩⟩, ⟨.program ⟨257⟩, ⟨56343⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨58413⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨58413⟩⟩) ⟨57933⟩ 285941) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨57933⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨24938⟩⟩, ⟨.program ⟨257⟩, ⟨56343⟩⟩], [⟨.program ⟨257⟩, ⟨57933⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge286016

namespace LeftMerge286017
def owner : Owner := ⟨.program ⟨257⟩, ⟨58414⟩⟩
def mergeEvent : Nat := 286017
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩, ⟨.program ⟨257⟩, ⟨58413⟩⟩] } }
def leftRaw : List Term := Proof.Events1117.exact286008RawTerms
def rightRaw : List Term := Proof.Events1116.exact285944RawTerms
def group : MergeGroup := .operator 286008 285944
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 286008) (leftOrdinal := 0)
    (rightResult := 285944) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨58413⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩, ⟨.program ⟨257⟩, ⟨58413⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge286017

namespace LeftMerge286031
def owner : Owner := ⟨.program ⟨257⟩, ⟨57352⟩⟩
def mergeEvent : Nat := 286031
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57349⟩⟩] } }
def leftRaw : List Term := Proof.Events1096.exact280745RawTerms
def rightRaw : List Term := Proof.Events1117.exact286025RawTerms
def group : MergeGroup := .operator 280745 286025
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 280745) (leftOrdinal := 0)
    (rightResult := 286025) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨57349⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57349⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge286031

namespace LeftMerge286110
def owner : Owner := ⟨.program ⟨257⟩, ⟨56344⟩⟩
def mergeEvent : Nat := 286110
def frameStart : Nat := 286080
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24938⟩⟩, ⟨.program ⟨257⟩, ⟨56343⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events1117.exact286106RawTerms
def rightRaw : List Term := Proof.Events1117.exact286103RawTerms
def group : MergeGroup := .operator 286106 286103
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 286106) (leftOrdinal := 0)
    (rightResult := 286103) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨56343⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24938⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨24938⟩⟩, ⟨.program ⟨257⟩, ⟨56343⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge286110

namespace LeftMerge286140
def owner : Owner := ⟨.program ⟨257⟩, ⟨58224⟩⟩
def mergeEvent : Nat := 286140
def frameStart : Nat := 286080
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24938⟩⟩, ⟨.program ⟨257⟩, ⟨56343⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1117.exact286136RawTerms
def rightRaw : List Term := Proof.Events1117.exact286134RawTerms
def group : MergeGroup := .operator 286136 286134
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 286136) (leftOrdinal := 0)
    (rightResult := 286134) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24938⟩⟩, ⟨.program ⟨257⟩, ⟨56343⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨24938⟩⟩, ⟨.program ⟨257⟩, ⟨56343⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge286140

namespace LeftMerge286161
def owner : Owner := ⟨.program ⟨257⟩, ⟨9534⟩⟩
def mergeEvent : Nat := 286161
def frameStart : Nat := 286080
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩] } }
def leftRaw : List Term := Proof.Events1117.exact286157RawTerms
def rightRaw : List Term := Proof.Events1117.exact286154RawTerms
def group : MergeGroup := .operator 286157 286154
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 286157) (leftOrdinal := 0)
    (rightResult := 286154) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7290⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9532⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge286161

namespace LeftMerge286170
def owner : Owner := ⟨.program ⟨257⟩, ⟨58416⟩⟩
def mergeEvent : Nat := 286170
def frameStart : Nat := 286080
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩, ⟨.program ⟨257⟩, ⟨58413⟩⟩] } }
def leftRaw : List Term := Proof.Events1117.exact286166RawTerms
def rightRaw : List Term := Proof.Events1117.exact286125RawTerms
def group : MergeGroup := .operator 286166 286125
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 286166) (leftOrdinal := 0)
    (rightResult := 286125) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨58413⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩, ⟨.program ⟨257⟩, ⟨58413⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge286170

namespace LeftMerge286171
def owner : Owner := ⟨.program ⟨257⟩, ⟨58416⟩⟩
def mergeEvent : Nat := 286171
def frameStart : Nat := 286080
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24938⟩⟩, ⟨.program ⟨257⟩, ⟨56343⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨58413⟩⟩] } }
def leftRaw : List Term := Proof.Events1117.exact286166RawTerms
def rightRaw : List Term := Proof.Events1117.exact286125RawTerms
def group : MergeGroup := .operator 286166 286125
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 286166) (leftOrdinal := 1)
    (rightResult := 286125) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24938⟩⟩, ⟨.program ⟨257⟩, ⟨56343⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨58413⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨24938⟩⟩, ⟨.program ⟨257⟩, ⟨56343⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨58413⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge286171

namespace LeftMerge286173
def owner : Owner := ⟨.program ⟨257⟩, ⟨58416⟩⟩
def mergeEvent : Nat := 286173
def frameStart : Nat := 286080
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24938⟩⟩, ⟨.program ⟨257⟩, ⟨56343⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨57933⟩⟩] } }
def rhsRaw : List Term := Proof.Events1117.exact286122RawTerms
def group : MergeGroup := .relation 286172
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 286172) (rhsResult := 286122)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨24938⟩⟩, ⟨.program ⟨257⟩, ⟨56343⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨58413⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨58413⟩⟩) ⟨57933⟩ 286122) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨57933⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨24938⟩⟩, ⟨.program ⟨257⟩, ⟨56343⟩⟩], [⟨.program ⟨257⟩, ⟨57933⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge286173

namespace LeftMerge286181
def owner : Owner := ⟨.program ⟨257⟩, ⟨56802⟩⟩
def mergeEvent : Nat := 286181
def frameStart : Nat := 286080
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨56800⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1117.exact286136RawTerms
def rightRaw : List Term := Proof.Events1117.exact286177RawTerms
def group : MergeGroup := .operator 286136 286177
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 286136) (leftOrdinal := 0)
    (rightResult := 286177) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨56800⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨56800⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge286181

namespace LeftMerge286198
def owner : Owner := ⟨.program ⟨257⟩, ⟨57352⟩⟩
def mergeEvent : Nat := 286198
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7185⟩⟩] } }
def rhsRaw : List Term := Proof.Events1117.exact286195RawTerms
def group : MergeGroup := .relation 286197
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 286197) (rhsResult := 286195)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57349⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 286196 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57349⟩⟩]⟩) (none) 286195) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7185⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7185⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge286198

namespace LeftMerge286199
def owner : Owner := ⟨.program ⟨257⟩, ⟨57352⟩⟩
def mergeEvent : Nat := 286199
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩, ⟨.program ⟨257⟩, ⟨58413⟩⟩] } }
def rhsRaw : List Term := Proof.Events1117.exact286195RawTerms
def group : MergeGroup := .relation 286197
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 286197) (rhsResult := 286195)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57349⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 286196 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57349⟩⟩]⟩) (none) 286195) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩, ⟨.program ⟨257⟩, ⟨58413⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩, ⟨.program ⟨257⟩, ⟨58413⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge286199

namespace LeftMerge286200
def owner : Owner := ⟨.program ⟨257⟩, ⟨57352⟩⟩
def mergeEvent : Nat := 286200
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨24938⟩⟩, ⟨.program ⟨257⟩, ⟨56343⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨57933⟩⟩] } }
def rhsRaw : List Term := Proof.Events1117.exact286195RawTerms
def group : MergeGroup := .relation 286197
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 286197) (rhsResult := 286195)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57349⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 286196 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57349⟩⟩]⟩) (none) 286195) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24938⟩⟩, ⟨.program ⟨257⟩, ⟨56343⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨57933⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨24938⟩⟩, ⟨.program ⟨257⟩, ⟨56343⟩⟩], [⟨.program ⟨257⟩, ⟨57933⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge286200

namespace LeftMerge286201
def owner : Owner := ⟨.program ⟨257⟩, ⟨57352⟩⟩
def mergeEvent : Nat := 286201
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨56800⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events1117.exact286195RawTerms
def group : MergeGroup := .relation 286197
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 286197) (rhsResult := 286195)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57349⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 286196 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57349⟩⟩]⟩) (none) 286195) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨56800⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨56800⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge286201

namespace LeftMerge286206
def owner : Owner := ⟨.program ⟨257⟩, ⟨58415⟩⟩
def mergeEvent : Nat := 286206
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨24938⟩⟩, ⟨.program ⟨257⟩, ⟨56343⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨57933⟩⟩] } }
def leftRaw : List Term := Proof.Events1117.exact286202RawTerms
def rightRaw : List Term := Proof.Events1117.exact286018RawTerms
def group : MergeGroup := .operator 286202 286018
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 286202) (leftOrdinal := 2)
    (rightResult := 286018) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨24938⟩⟩, ⟨.program ⟨257⟩, ⟨56343⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨57933⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨24938⟩⟩, ⟨.program ⟨257⟩, ⟨56343⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨57933⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨24938⟩⟩, ⟨.program ⟨257⟩, ⟨56343⟩⟩], [⟨.program ⟨257⟩, ⟨57933⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge286206

namespace LeftMerge286207
def owner : Owner := ⟨.program ⟨257⟩, ⟨58415⟩⟩
def mergeEvent : Nat := 286207
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩, ⟨.program ⟨257⟩, ⟨58413⟩⟩] } }
def leftRaw : List Term := Proof.Events1117.exact286202RawTerms
def rightRaw : List Term := Proof.Events1117.exact286018RawTerms
def group : MergeGroup := .operator 286202 286018
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 286202) (leftOrdinal := 1)
    (rightResult := 286018) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩, ⟨.program ⟨257⟩, ⟨58413⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩, ⟨.program ⟨257⟩, ⟨58413⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩, ⟨.program ⟨257⟩, ⟨58413⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge286207

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
