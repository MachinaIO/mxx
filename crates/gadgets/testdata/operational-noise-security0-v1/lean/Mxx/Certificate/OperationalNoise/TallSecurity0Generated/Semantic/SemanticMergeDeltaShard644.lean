import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge103981
def owner : Owner := ⟨.program ⟨214⟩, ⟨29779⟩⟩
def mergeEvent : Nat := 103981
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16861⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29777⟩⟩] } }
def leftRaw : List Term := Proof.Events371.exact95058RawTerms
def rightRaw : List Term := Proof.Events406.exact103974RawTerms
def group : MergeGroup := .operator 95058 103974
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 95058) (leftOrdinal := 1)
    (rightResult := 103974) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16861⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨29777⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16861⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29777⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge103981

namespace LeftMerge103983
def owner : Owner := ⟨.program ⟨214⟩, ⟨29779⟩⟩
def mergeEvent : Nat := 103983
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16861⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24719⟩⟩] } }
def rhsRaw : List Term := Proof.Events406.exact103971RawTerms
def group : MergeGroup := .relation 103982
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 103982) (rhsResult := 103971)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16861⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29777⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨29777⟩⟩) ⟨24719⟩ 103971) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨24719⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16861⟩⟩], [⟨.program ⟨214⟩, ⟨24719⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge103983

namespace LeftMerge103997
def owner : Owner := ⟨.program ⟨214⟩, ⟨22616⟩⟩
def mergeEvent : Nat := 103997
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22613⟩⟩] } }
def leftRaw : List Term := Proof.Events368.exact94462RawTerms
def rightRaw : List Term := Proof.Events406.exact103991RawTerms
def group : MergeGroup := .operator 94462 103991
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 94462) (leftOrdinal := 0)
    (rightResult := 103991) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨22613⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22613⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge103997

namespace LeftMerge104094
def owner : Owner := ⟨.program ⟨214⟩, ⟨16961⟩⟩
def mergeEvent : Nat := 104094
def frameStart : Nat := 104040
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16861⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events406.exact104090RawTerms
def rightRaw : List Term := Proof.Events406.exact104088RawTerms
def group : MergeGroup := .operator 104090 104088
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 104090) (leftOrdinal := 0)
    (rightResult := 104088) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16861⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨16861⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge104094

namespace LeftMerge104106
def owner : Owner := ⟨.program ⟨214⟩, ⟨29778⟩⟩
def mergeEvent : Nat := 104106
def frameStart : Nat := 104040
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6706⟩⟩, ⟨.program ⟨214⟩, ⟨29777⟩⟩] } }
def leftRaw : List Term := Proof.Events406.exact104102RawTerms
def rightRaw : List Term := Proof.Events406.exact104079RawTerms
def group : MergeGroup := .operator 104102 104079
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 104102) (leftOrdinal := 0)
    (rightResult := 104079) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6706⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨29777⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6706⟩⟩, ⟨.program ⟨214⟩, ⟨29777⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge104106

namespace LeftMerge104107
def owner : Owner := ⟨.program ⟨214⟩, ⟨29778⟩⟩
def mergeEvent : Nat := 104107
def frameStart : Nat := 104040
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16861⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29777⟩⟩] } }
def leftRaw : List Term := Proof.Events406.exact104102RawTerms
def rightRaw : List Term := Proof.Events406.exact104079RawTerms
def group : MergeGroup := .operator 104102 104079
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 104102) (leftOrdinal := 1)
    (rightResult := 104079) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16861⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨29777⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨16861⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29777⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge104107

namespace LeftMerge104109
def owner : Owner := ⟨.program ⟨214⟩, ⟨29778⟩⟩
def mergeEvent : Nat := 104109
def frameStart : Nat := 104040
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16861⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24719⟩⟩] } }
def rhsRaw : List Term := Proof.Events406.exact104076RawTerms
def group : MergeGroup := .relation 104108
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 104108) (rhsResult := 104076)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨16861⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29777⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨29777⟩⟩) ⟨24719⟩ 104076) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨24719⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨16861⟩⟩], [⟨.program ⟨214⟩, ⟨24719⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge104109

namespace LeftMerge104117
def owner : Owner := ⟨.program ⟨214⟩, ⟨16919⟩⟩
def mergeEvent : Nat := 104117
def frameStart : Nat := 104040
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16917⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events406.exact104090RawTerms
def rightRaw : List Term := Proof.Events406.exact104113RawTerms
def group : MergeGroup := .operator 104090 104113
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 104090) (leftOrdinal := 0)
    (rightResult := 104113) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16917⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨16917⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge104117

namespace LeftMerge104134
def owner : Owner := ⟨.program ⟨214⟩, ⟨22616⟩⟩
def mergeEvent : Nat := 104134
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6740⟩⟩] } }
def rhsRaw : List Term := Proof.Events406.exact104131RawTerms
def group : MergeGroup := .relation 104133
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 104133) (rhsResult := 104131)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22613⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 104132 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22613⟩⟩]⟩) (none) 104131) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6740⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6740⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge104134

namespace LeftMerge104135
def owner : Owner := ⟨.program ⟨214⟩, ⟨22616⟩⟩
def mergeEvent : Nat := 104135
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6706⟩⟩, ⟨.program ⟨214⟩, ⟨29777⟩⟩] } }
def rhsRaw : List Term := Proof.Events406.exact104131RawTerms
def group : MergeGroup := .relation 104133
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 104133) (rhsResult := 104131)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22613⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 104132 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22613⟩⟩]⟩) (none) 104131) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6706⟩⟩, ⟨.program ⟨214⟩, ⟨29777⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6706⟩⟩, ⟨.program ⟨214⟩, ⟨29777⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge104135

namespace LeftMerge104136
def owner : Owner := ⟨.program ⟨214⟩, ⟨22616⟩⟩
def mergeEvent : Nat := 104136
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16861⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24719⟩⟩] } }
def rhsRaw : List Term := Proof.Events406.exact104131RawTerms
def group : MergeGroup := .relation 104133
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 104133) (rhsResult := 104131)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22613⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 104132 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22613⟩⟩]⟩) (none) 104131) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16861⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24719⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16861⟩⟩], [⟨.program ⟨214⟩, ⟨24719⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge104136

namespace LeftMerge104137
def owner : Owner := ⟨.program ⟨214⟩, ⟨22616⟩⟩
def mergeEvent : Nat := 104137
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16917⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def rhsRaw : List Term := Proof.Events406.exact104131RawTerms
def group : MergeGroup := .relation 104133
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 104133) (rhsResult := 104131)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22613⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 104132 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22613⟩⟩]⟩) (none) 104131) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16917⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16917⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge104137

namespace LeftMerge104142
def owner : Owner := ⟨.program ⟨214⟩, ⟨29780⟩⟩
def mergeEvent : Nat := 104142
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6706⟩⟩, ⟨.program ⟨214⟩, ⟨29777⟩⟩] } }
def leftRaw : List Term := Proof.Events406.exact104138RawTerms
def rightRaw : List Term := Proof.Events406.exact103984RawTerms
def group : MergeGroup := .operator 104138 103984
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 104138) (leftOrdinal := 0)
    (rightResult := 103984) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6706⟩⟩, ⟨.program ⟨214⟩, ⟨29777⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6706⟩⟩, ⟨.program ⟨214⟩, ⟨29777⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6706⟩⟩, ⟨.program ⟨214⟩, ⟨29777⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge104142

namespace LeftMerge104143
def owner : Owner := ⟨.program ⟨214⟩, ⟨29780⟩⟩
def mergeEvent : Nat := 104143
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16861⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24719⟩⟩] } }
def leftRaw : List Term := Proof.Events406.exact104138RawTerms
def rightRaw : List Term := Proof.Events406.exact103984RawTerms
def group : MergeGroup := .operator 104138 103984
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 104138) (leftOrdinal := 2)
    (rightResult := 103984) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16861⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24719⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16861⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24719⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16861⟩⟩], [⟨.program ⟨214⟩, ⟨24719⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge104143

namespace LeftMerge104151
def owner : Owner := ⟨.program ⟨214⟩, ⟨29781⟩⟩
def mergeEvent : Nat := 104151
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6740⟩⟩, ⟨.program ⟨214⟩, ⟨6659⟩⟩] } }
def leftRaw : List Term := Proof.Events406.exact104145RawTerms
def rightRaw : List Term := Proof.Events021.exact5539RawTerms
def group : MergeGroup := .operator 104145 5539
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 104145) (leftOrdinal := 0)
    (rightResult := 5539) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6740⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6659⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6740⟩⟩, ⟨.program ⟨214⟩, ⟨6659⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge104151

namespace LeftMerge104152
def owner : Owner := ⟨.program ⟨214⟩, ⟨29781⟩⟩
def mergeEvent : Nat := 104152
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16917⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨6659⟩⟩] } }
def leftRaw : List Term := Proof.Events406.exact104145RawTerms
def rightRaw : List Term := Proof.Events021.exact5539RawTerms
def group : MergeGroup := .operator 104145 5539
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 104145) (leftOrdinal := 1)
    (rightResult := 5539) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16917⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6659⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16917⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨6659⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge104152

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
