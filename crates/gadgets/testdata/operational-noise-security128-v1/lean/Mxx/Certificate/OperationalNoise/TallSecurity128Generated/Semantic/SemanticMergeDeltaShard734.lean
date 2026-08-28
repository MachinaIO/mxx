import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge120992
def owner : Owner := ⟨.program ⟨257⟩, ⟨42758⟩⟩
def mergeEvent : Nat := 120992
def frameStart : Nat := 120889
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨42756⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events472.exact120945RawTerms
def rightRaw : List Term := Proof.Events472.exact120988RawTerms
def group : MergeGroup := .operator 120945 120988
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 120945) (leftOrdinal := 0)
    (rightResult := 120988) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨42756⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨42756⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge120992

namespace LeftMerge121009
def owner : Owner := ⟨.program ⟨257⟩, ⟨43192⟩⟩
def mergeEvent : Nat := 121009
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7194⟩⟩] } }
def rhsRaw : List Term := Proof.Events472.exact121006RawTerms
def group : MergeGroup := .relation 121008
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 121008) (rhsResult := 121006)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43189⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 121007 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43189⟩⟩]⟩) (none) 121006) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7194⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7194⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge121009

namespace LeftMerge121010
def owner : Owner := ⟨.program ⟨257⟩, ⟨43192⟩⟩
def mergeEvent : Nat := 121010
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩, ⟨.program ⟨257⟩, ⟨44255⟩⟩] } }
def rhsRaw : List Term := Proof.Events472.exact121006RawTerms
def group : MergeGroup := .relation 121008
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 121008) (rhsResult := 121006)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43189⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 121007 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43189⟩⟩]⟩) (none) 121006) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩, ⟨.program ⟨257⟩, ⟨44255⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩, ⟨.program ⟨257⟩, ⟨44255⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge121010

namespace LeftMerge121011
def owner : Owner := ⟨.program ⟨257⟩, ⟨43192⟩⟩
def mergeEvent : Nat := 121011
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨14421⟩⟩, ⟨.program ⟨257⟩, ⟨42378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨43765⟩⟩] } }
def rhsRaw : List Term := Proof.Events472.exact121006RawTerms
def group : MergeGroup := .relation 121008
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 121008) (rhsResult := 121006)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43189⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 121007 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43189⟩⟩]⟩) (none) 121006) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14421⟩⟩, ⟨.program ⟨257⟩, ⟨42378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨43765⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨14421⟩⟩, ⟨.program ⟨257⟩, ⟨42378⟩⟩], [⟨.program ⟨257⟩, ⟨43765⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge121011

namespace LeftMerge121012
def owner : Owner := ⟨.program ⟨257⟩, ⟨43192⟩⟩
def mergeEvent : Nat := 121012
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨42756⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events472.exact121006RawTerms
def group : MergeGroup := .relation 121008
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 121008) (rhsResult := 121006)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43189⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 121007 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43189⟩⟩]⟩) (none) 121006) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨42756⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨42756⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge121012

namespace LeftMerge121017
def owner : Owner := ⟨.program ⟨257⟩, ⟨44257⟩⟩
def mergeEvent : Nat := 121017
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨14421⟩⟩, ⟨.program ⟨257⟩, ⟨42378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨43765⟩⟩] } }
def leftRaw : List Term := Proof.Events472.exact121013RawTerms
def rightRaw : List Term := Proof.Events471.exact120827RawTerms
def group : MergeGroup := .operator 121013 120827
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 121013) (leftOrdinal := 2)
    (rightResult := 120827) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨14421⟩⟩, ⟨.program ⟨257⟩, ⟨42378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨43765⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨14421⟩⟩, ⟨.program ⟨257⟩, ⟨42378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨43765⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨14421⟩⟩, ⟨.program ⟨257⟩, ⟨42378⟩⟩], [⟨.program ⟨257⟩, ⟨43765⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge121017

namespace LeftMerge121018
def owner : Owner := ⟨.program ⟨257⟩, ⟨44257⟩⟩
def mergeEvent : Nat := 121018
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩, ⟨.program ⟨257⟩, ⟨44255⟩⟩] } }
def leftRaw : List Term := Proof.Events472.exact121013RawTerms
def rightRaw : List Term := Proof.Events471.exact120827RawTerms
def group : MergeGroup := .operator 121013 120827
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 121013) (leftOrdinal := 1)
    (rightResult := 120827) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩, ⟨.program ⟨257⟩, ⟨44255⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩, ⟨.program ⟨257⟩, ⟨44255⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩, ⟨.program ⟨257⟩, ⟨44255⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge121018

namespace LeftMerge121026
def owner : Owner := ⟨.program ⟨257⟩, ⟨44571⟩⟩
def mergeEvent : Nat := 121026
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44569⟩⟩] } }
def leftRaw : List Term := Proof.Events472.exact121020RawTerms
def rightRaw : List Term := Proof.Events471.exact120743RawTerms
def group : MergeGroup := .operator 121020 120743
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 121020) (leftOrdinal := 0)
    (rightResult := 120743) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7194⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨44569⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44569⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge121026

namespace LeftMerge121027
def owner : Owner := ⟨.program ⟨257⟩, ⟨44571⟩⟩
def mergeEvent : Nat := 121027
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨42756⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨44569⟩⟩] } }
def leftRaw : List Term := Proof.Events472.exact121020RawTerms
def rightRaw : List Term := Proof.Events471.exact120743RawTerms
def group : MergeGroup := .operator 121020 120743
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 121020) (leftOrdinal := 1)
    (rightResult := 120743) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨42756⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨44569⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨42756⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨44569⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge121027

namespace LeftMerge121029
def owner : Owner := ⟨.program ⟨257⟩, ⟨44571⟩⟩
def mergeEvent : Nat := 121029
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨42756⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨43905⟩⟩] } }
def rhsRaw : List Term := Proof.Events471.exact120740RawTerms
def group : MergeGroup := .relation 121028
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 121028) (rhsResult := 120740)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨42756⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨44569⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨44569⟩⟩) ⟨43905⟩ 120740) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨43905⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨42756⟩⟩], [⟨.program ⟨257⟩, ⟨43905⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge121029

namespace LeftMerge121043
def owner : Owner := ⟨.program ⟨257⟩, ⟨43459⟩⟩
def mergeEvent : Nat := 121043
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43456⟩⟩] } }
def leftRaw : List Term := Proof.Events468.exact119870RawTerms
def rightRaw : List Term := Proof.Events472.exact121037RawTerms
def group : MergeGroup := .operator 119870 121037
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 119870) (leftOrdinal := 0)
    (rightResult := 121037) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨43456⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43456⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge121043

namespace LeftMerge121164
def owner : Owner := ⟨.program ⟨257⟩, ⟨44132⟩⟩
def mergeEvent : Nat := 121164
def frameStart : Nat := 121098
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨42756⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events473.exact121160RawTerms
def rightRaw : List Term := Proof.Events473.exact121158RawTerms
def group : MergeGroup := .operator 121160 121158
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 121160) (leftOrdinal := 0)
    (rightResult := 121158) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨42756⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨42756⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge121164

namespace LeftMerge121176
def owner : Owner := ⟨.program ⟨257⟩, ⟨44570⟩⟩
def mergeEvent : Nat := 121176
def frameStart : Nat := 121098
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44569⟩⟩] } }
def leftRaw : List Term := Proof.Events473.exact121172RawTerms
def rightRaw : List Term := Proof.Events473.exact121149RawTerms
def group : MergeGroup := .operator 121172 121149
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 121172) (leftOrdinal := 0)
    (rightResult := 121149) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7194⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨44569⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44569⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge121176

namespace LeftMerge121177
def owner : Owner := ⟨.program ⟨257⟩, ⟨44570⟩⟩
def mergeEvent : Nat := 121177
def frameStart : Nat := 121098
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨42756⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨44569⟩⟩] } }
def leftRaw : List Term := Proof.Events473.exact121172RawTerms
def rightRaw : List Term := Proof.Events473.exact121149RawTerms
def group : MergeGroup := .operator 121172 121149
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 121172) (leftOrdinal := 1)
    (rightResult := 121149) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨42756⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨44569⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨42756⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨44569⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge121177

namespace LeftMerge121179
def owner : Owner := ⟨.program ⟨257⟩, ⟨44570⟩⟩
def mergeEvent : Nat := 121179
def frameStart : Nat := 121098
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨42756⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨43905⟩⟩] } }
def rhsRaw : List Term := Proof.Events473.exact121146RawTerms
def group : MergeGroup := .relation 121178
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 121178) (rhsResult := 121146)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨42756⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨44569⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨44569⟩⟩) ⟨43905⟩ 121146) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨43905⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨42756⟩⟩], [⟨.program ⟨257⟩, ⟨43905⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge121179

namespace LeftMerge121187
def owner : Owner := ⟨.program ⟨257⟩, ⟨42948⟩⟩
def mergeEvent : Nat := 121187
def frameStart : Nat := 121098
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨42947⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events473.exact121160RawTerms
def rightRaw : List Term := Proof.Events473.exact121183RawTerms
def group : MergeGroup := .operator 121160 121183
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 121160) (leftOrdinal := 0)
    (rightResult := 121183) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨42947⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨42947⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge121187

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
