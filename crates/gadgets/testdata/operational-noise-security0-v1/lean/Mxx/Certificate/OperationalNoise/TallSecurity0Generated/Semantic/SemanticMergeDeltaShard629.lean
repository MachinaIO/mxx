import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge102000
def owner : Owner := ⟨.program ⟨214⟩, ⟨24900⟩⟩
def mergeEvent : Nat := 102000
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6771⟩⟩, ⟨.program ⟨214⟩, ⟨7831⟩⟩, ⟨.program ⟨214⟩, ⟨24898⟩⟩] } }
def leftRaw : List Term := Proof.Events398.exact101995RawTerms
def rightRaw : List Term := Proof.Events397.exact101833RawTerms
def group : MergeGroup := .operator 101995 101833
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 101995) (leftOrdinal := 1)
    (rightResult := 101833) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6771⟩⟩, ⟨.program ⟨214⟩, ⟨7831⟩⟩, ⟨.program ⟨214⟩, ⟨24898⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6771⟩⟩, ⟨.program ⟨214⟩, ⟨7831⟩⟩, ⟨.program ⟨214⟩, ⟨24898⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6771⟩⟩, ⟨.program ⟨214⟩, ⟨7831⟩⟩, ⟨.program ⟨214⟩, ⟨24898⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge102000

namespace LeftMerge102008
def owner : Owner := ⟨.program ⟨214⟩, ⟨26328⟩⟩
def mergeEvent : Nat := 102008
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6690⟩⟩, ⟨.program ⟨214⟩, ⟨26326⟩⟩] } }
def leftRaw : List Term := Proof.Events398.exact102002RawTerms
def rightRaw : List Term := Proof.Events397.exact101749RawTerms
def group : MergeGroup := .operator 102002 101749
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 102002) (leftOrdinal := 0)
    (rightResult := 101749) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6690⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨26326⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6690⟩⟩, ⟨.program ⟨214⟩, ⟨26326⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge102008

namespace LeftMerge102009
def owner : Owner := ⟨.program ⟨214⟩, ⟨26328⟩⟩
def mergeEvent : Nat := 102009
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨14782⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26326⟩⟩] } }
def leftRaw : List Term := Proof.Events398.exact102002RawTerms
def rightRaw : List Term := Proof.Events397.exact101749RawTerms
def group : MergeGroup := .operator 102002 101749
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 102002) (leftOrdinal := 1)
    (rightResult := 101749) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨14782⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨26326⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨14782⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26326⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge102009

namespace LeftMerge102011
def owner : Owner := ⟨.program ⟨214⟩, ⟨26328⟩⟩
def mergeEvent : Nat := 102011
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨14782⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23712⟩⟩] } }
def rhsRaw : List Term := Proof.Events397.exact101746RawTerms
def group : MergeGroup := .relation 102010
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 102010) (rhsResult := 101746)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨14782⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26326⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨26326⟩⟩) ⟨23712⟩ 101746) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23712⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨14782⟩⟩], [⟨.program ⟨214⟩, ⟨23712⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge102011

namespace LeftMerge102025
def owner : Owner := ⟨.program ⟨214⟩, ⟨20384⟩⟩
def mergeEvent : Nat := 102025
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20381⟩⟩] } }
def leftRaw : List Term := Proof.Events368.exact94462RawTerms
def rightRaw : List Term := Proof.Events398.exact102019RawTerms
def group : MergeGroup := .operator 94462 102019
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 94462) (leftOrdinal := 0)
    (rightResult := 102019) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨20381⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20381⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge102025

namespace LeftMerge102122
def owner : Owner := ⟨.program ⟨214⟩, ⟨14826⟩⟩
def mergeEvent : Nat := 102122
def frameStart : Nat := 102068
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨14782⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events398.exact102118RawTerms
def rightRaw : List Term := Proof.Events398.exact102116RawTerms
def group : MergeGroup := .operator 102118 102116
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 102118) (leftOrdinal := 0)
    (rightResult := 102116) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨14782⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨14782⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge102122

namespace LeftMerge102134
def owner : Owner := ⟨.program ⟨214⟩, ⟨26327⟩⟩
def mergeEvent : Nat := 102134
def frameStart : Nat := 102068
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6690⟩⟩, ⟨.program ⟨214⟩, ⟨26326⟩⟩] } }
def leftRaw : List Term := Proof.Events398.exact102130RawTerms
def rightRaw : List Term := Proof.Events398.exact102107RawTerms
def group : MergeGroup := .operator 102130 102107
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 102130) (leftOrdinal := 0)
    (rightResult := 102107) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6690⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨26326⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6690⟩⟩, ⟨.program ⟨214⟩, ⟨26326⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge102134

namespace LeftMerge102135
def owner : Owner := ⟨.program ⟨214⟩, ⟨26327⟩⟩
def mergeEvent : Nat := 102135
def frameStart : Nat := 102068
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨14782⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26326⟩⟩] } }
def leftRaw : List Term := Proof.Events398.exact102130RawTerms
def rightRaw : List Term := Proof.Events398.exact102107RawTerms
def group : MergeGroup := .operator 102130 102107
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 102130) (leftOrdinal := 1)
    (rightResult := 102107) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨14782⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨26326⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨14782⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26326⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge102135

namespace LeftMerge102137
def owner : Owner := ⟨.program ⟨214⟩, ⟨26327⟩⟩
def mergeEvent : Nat := 102137
def frameStart : Nat := 102068
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨14782⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23712⟩⟩] } }
def rhsRaw : List Term := Proof.Events398.exact102104RawTerms
def group : MergeGroup := .relation 102136
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 102136) (rhsResult := 102104)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨14782⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26326⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨26326⟩⟩) ⟨23712⟩ 102104) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23712⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨14782⟩⟩], [⟨.program ⟨214⟩, ⟨23712⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge102137

namespace LeftMerge102145
def owner : Owner := ⟨.program ⟨214⟩, ⟨15259⟩⟩
def mergeEvent : Nat := 102145
def frameStart : Nat := 102068
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15258⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events398.exact102118RawTerms
def rightRaw : List Term := Proof.Events398.exact102141RawTerms
def group : MergeGroup := .operator 102118 102141
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 102118) (leftOrdinal := 0)
    (rightResult := 102141) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15258⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨15258⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge102145

namespace LeftMerge102162
def owner : Owner := ⟨.program ⟨214⟩, ⟨20384⟩⟩
def mergeEvent : Nat := 102162
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6709⟩⟩] } }
def rhsRaw : List Term := Proof.Events399.exact102159RawTerms
def group : MergeGroup := .relation 102161
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 102161) (rhsResult := 102159)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20381⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 102160 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20381⟩⟩]⟩) (none) 102159) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6709⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge102162

namespace LeftMerge102163
def owner : Owner := ⟨.program ⟨214⟩, ⟨20384⟩⟩
def mergeEvent : Nat := 102163
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6690⟩⟩, ⟨.program ⟨214⟩, ⟨26326⟩⟩] } }
def rhsRaw : List Term := Proof.Events399.exact102159RawTerms
def group : MergeGroup := .relation 102161
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 102161) (rhsResult := 102159)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20381⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 102160 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20381⟩⟩]⟩) (none) 102159) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6690⟩⟩, ⟨.program ⟨214⟩, ⟨26326⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6690⟩⟩, ⟨.program ⟨214⟩, ⟨26326⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge102163

namespace LeftMerge102164
def owner : Owner := ⟨.program ⟨214⟩, ⟨20384⟩⟩
def mergeEvent : Nat := 102164
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨14782⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23712⟩⟩] } }
def rhsRaw : List Term := Proof.Events399.exact102159RawTerms
def group : MergeGroup := .relation 102161
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 102161) (rhsResult := 102159)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20381⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 102160 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20381⟩⟩]⟩) (none) 102159) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨14782⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23712⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨14782⟩⟩], [⟨.program ⟨214⟩, ⟨23712⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge102164

namespace LeftMerge102165
def owner : Owner := ⟨.program ⟨214⟩, ⟨20384⟩⟩
def mergeEvent : Nat := 102165
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15258⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def rhsRaw : List Term := Proof.Events399.exact102159RawTerms
def group : MergeGroup := .relation 102161
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 102161) (rhsResult := 102159)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20381⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 102160 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20381⟩⟩]⟩) (none) 102159) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15258⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15258⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge102165

namespace LeftMerge102170
def owner : Owner := ⟨.program ⟨214⟩, ⟨26329⟩⟩
def mergeEvent : Nat := 102170
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6690⟩⟩, ⟨.program ⟨214⟩, ⟨26326⟩⟩] } }
def leftRaw : List Term := Proof.Events399.exact102166RawTerms
def rightRaw : List Term := Proof.Events398.exact102012RawTerms
def group : MergeGroup := .operator 102166 102012
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 102166) (leftOrdinal := 0)
    (rightResult := 102012) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6690⟩⟩, ⟨.program ⟨214⟩, ⟨26326⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6690⟩⟩, ⟨.program ⟨214⟩, ⟨26326⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6690⟩⟩, ⟨.program ⟨214⟩, ⟨26326⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge102170

namespace LeftMerge102171
def owner : Owner := ⟨.program ⟨214⟩, ⟨26329⟩⟩
def mergeEvent : Nat := 102171
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨14782⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23712⟩⟩] } }
def leftRaw : List Term := Proof.Events399.exact102166RawTerms
def rightRaw : List Term := Proof.Events398.exact102012RawTerms
def group : MergeGroup := .operator 102166 102012
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 102166) (leftOrdinal := 2)
    (rightResult := 102012) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨14782⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23712⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨14782⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23712⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨14782⟩⟩], [⟨.program ⟨214⟩, ⟨23712⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge102171

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
