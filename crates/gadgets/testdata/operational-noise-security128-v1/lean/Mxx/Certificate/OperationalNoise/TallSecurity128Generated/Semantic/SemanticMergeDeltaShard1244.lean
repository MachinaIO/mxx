import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge203146
def owner : Owner := ⟨.program ⟨257⟩, ⟨68393⟩⟩
def mergeEvent : Nat := 203146
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨18904⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68842⟩⟩] } }
def rhsRaw : List Term := Proof.Events793.exact203108RawTerms
def group : MergeGroup := .relation 203110
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 203110) (rhsResult := 203108)
    (sourceTermOrdinal := 20) (source := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68390⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 203109 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68390⟩⟩]⟩) (none) 203108) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨18904⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68842⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨18904⟩⟩], [⟨.program ⟨257⟩, ⟨68842⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge203146

namespace LeftMerge203147
def owner : Owner := ⟨.program ⟨257⟩, ⟨68393⟩⟩
def mergeEvent : Nat := 203147
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨16067⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68842⟩⟩] } }
def rhsRaw : List Term := Proof.Events793.exact203108RawTerms
def group : MergeGroup := .relation 203110
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 203110) (rhsResult := 203108)
    (sourceTermOrdinal := 19) (source := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68390⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 203109 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68390⟩⟩]⟩) (none) 203108) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨16067⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68842⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨16067⟩⟩], [⟨.program ⟨257⟩, ⟨68842⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge203147

namespace LeftMerge203148
def owner : Owner := ⟨.program ⟨257⟩, ⟨68393⟩⟩
def mergeEvent : Nat := 203148
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨67494⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events793.exact203108RawTerms
def group : MergeGroup := .relation 203110
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 203110) (rhsResult := 203108)
    (sourceTermOrdinal := 37) (source := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68390⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 203109 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68390⟩⟩]⟩) (none) 203108) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨67494⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨67494⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge203148

namespace LeftMerge203153
def owner : Owner := ⟨.program ⟨257⟩, ⟨71300⟩⟩
def mergeEvent : Nat := 203153
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7232⟩⟩, ⟨.program ⟨257⟩, ⟨71297⟩⟩] } }
def leftRaw : List Term := Proof.Events793.exact203149RawTerms
def rightRaw : List Term := Proof.Events788.exact201733RawTerms
def group : MergeGroup := .operator 203149 201733
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 203149) (leftOrdinal := 17)
    (rightResult := 201733) (rightOrdinal := 17) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7232⟩⟩, ⟨.program ⟨257⟩, ⟨71297⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7232⟩⟩, ⟨.program ⟨257⟩, ⟨71297⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7232⟩⟩, ⟨.program ⟨257⟩, ⟨71297⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge203153

namespace LeftMerge203154
def owner : Owner := ⟨.program ⟨257⟩, ⟨71300⟩⟩
def mergeEvent : Nat := 203154
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨48389⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68842⟩⟩] } }
def leftRaw : List Term := Proof.Events793.exact203149RawTerms
def rightRaw : List Term := Proof.Events788.exact201733RawTerms
def group : MergeGroup := .operator 203149 201733
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 203149) (leftOrdinal := 30)
    (rightResult := 201733) (rightOrdinal := 29) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨48389⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68842⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨48389⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68842⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨48389⟩⟩], [⟨.program ⟨257⟩, ⟨68842⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge203154

namespace LeftMerge203155
def owner : Owner := ⟨.program ⟨257⟩, ⟨71300⟩⟩
def mergeEvent : Nat := 203155
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨71297⟩⟩] } }
def leftRaw : List Term := Proof.Events793.exact203149RawTerms
def rightRaw : List Term := Proof.Events788.exact201733RawTerms
def group : MergeGroup := .operator 203149 201733
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 203149) (leftOrdinal := 16)
    (rightResult := 201733) (rightOrdinal := 16) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨71297⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨71297⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨71297⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge203155

namespace LeftMerge203156
def owner : Owner := ⟨.program ⟨257⟩, ⟨71300⟩⟩
def mergeEvent : Nat := 203156
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨45709⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68842⟩⟩] } }
def leftRaw : List Term := Proof.Events793.exact203149RawTerms
def rightRaw : List Term := Proof.Events788.exact201733RawTerms
def group : MergeGroup := .operator 203149 201733
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 203149) (leftOrdinal := 29)
    (rightResult := 201733) (rightOrdinal := 28) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨45709⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68842⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨45709⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68842⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨45709⟩⟩], [⟨.program ⟨257⟩, ⟨68842⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge203156

namespace LeftMerge203157
def owner : Owner := ⟨.program ⟨257⟩, ⟨71300⟩⟩
def mergeEvent : Nat := 203157
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨71297⟩⟩] } }
def leftRaw : List Term := Proof.Events793.exact203149RawTerms
def rightRaw : List Term := Proof.Events788.exact201733RawTerms
def group : MergeGroup := .operator 203149 201733
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 203149) (leftOrdinal := 15)
    (rightResult := 201733) (rightOrdinal := 15) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨71297⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨71297⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨71297⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge203157

namespace LeftMerge203158
def owner : Owner := ⟨.program ⟨257⟩, ⟨71300⟩⟩
def mergeEvent : Nat := 203158
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨43025⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68842⟩⟩] } }
def leftRaw : List Term := Proof.Events793.exact203149RawTerms
def rightRaw : List Term := Proof.Events788.exact201733RawTerms
def group : MergeGroup := .operator 203149 201733
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 203149) (leftOrdinal := 28)
    (rightResult := 201733) (rightOrdinal := 27) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨43025⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68842⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨43025⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68842⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨43025⟩⟩], [⟨.program ⟨257⟩, ⟨68842⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge203158

namespace LeftMerge203159
def owner : Owner := ⟨.program ⟨257⟩, ⟨71300⟩⟩
def mergeEvent : Nat := 203159
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨71297⟩⟩] } }
def leftRaw : List Term := Proof.Events793.exact203149RawTerms
def rightRaw : List Term := Proof.Events788.exact201733RawTerms
def group : MergeGroup := .operator 203149 201733
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 203149) (leftOrdinal := 14)
    (rightResult := 201733) (rightOrdinal := 14) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨71297⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨71297⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨71297⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge203159

namespace LeftMerge203160
def owner : Owner := ⟨.program ⟨257⟩, ⟨71300⟩⟩
def mergeEvent : Nat := 203160
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨40345⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68842⟩⟩] } }
def leftRaw : List Term := Proof.Events793.exact203149RawTerms
def rightRaw : List Term := Proof.Events788.exact201733RawTerms
def group : MergeGroup := .operator 203149 201733
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 203149) (leftOrdinal := 27)
    (rightResult := 201733) (rightOrdinal := 26) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨40345⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68842⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨40345⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68842⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨40345⟩⟩], [⟨.program ⟨257⟩, ⟨68842⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge203160

namespace LeftMerge203161
def owner : Owner := ⟨.program ⟨257⟩, ⟨71300⟩⟩
def mergeEvent : Nat := 203161
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨71297⟩⟩] } }
def leftRaw : List Term := Proof.Events793.exact203149RawTerms
def rightRaw : List Term := Proof.Events788.exact201733RawTerms
def group : MergeGroup := .operator 203149 201733
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 203149) (leftOrdinal := 13)
    (rightResult := 201733) (rightOrdinal := 13) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨71297⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨71297⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨71297⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge203161

namespace LeftMerge203162
def owner : Owner := ⟨.program ⟨257⟩, ⟨71300⟩⟩
def mergeEvent : Nat := 203162
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨37669⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68842⟩⟩] } }
def leftRaw : List Term := Proof.Events793.exact203149RawTerms
def rightRaw : List Term := Proof.Events788.exact201733RawTerms
def group : MergeGroup := .operator 203149 201733
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 203149) (leftOrdinal := 26)
    (rightResult := 201733) (rightOrdinal := 25) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨37669⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68842⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨37669⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68842⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨37669⟩⟩], [⟨.program ⟨257⟩, ⟨68842⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge203162

namespace LeftMerge203163
def owner : Owner := ⟨.program ⟨257⟩, ⟨71300⟩⟩
def mergeEvent : Nat := 203163
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7222⟩⟩, ⟨.program ⟨257⟩, ⟨71297⟩⟩] } }
def leftRaw : List Term := Proof.Events793.exact203149RawTerms
def rightRaw : List Term := Proof.Events788.exact201733RawTerms
def group : MergeGroup := .operator 203149 201733
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 203149) (leftOrdinal := 12)
    (rightResult := 201733) (rightOrdinal := 12) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7222⟩⟩, ⟨.program ⟨257⟩, ⟨71297⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7222⟩⟩, ⟨.program ⟨257⟩, ⟨71297⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7222⟩⟩, ⟨.program ⟨257⟩, ⟨71297⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge203163

namespace LeftMerge203164
def owner : Owner := ⟨.program ⟨257⟩, ⟨71300⟩⟩
def mergeEvent : Nat := 203164
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨34989⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68842⟩⟩] } }
def leftRaw : List Term := Proof.Events793.exact203149RawTerms
def rightRaw : List Term := Proof.Events788.exact201733RawTerms
def group : MergeGroup := .operator 203149 201733
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 203149) (leftOrdinal := 25)
    (rightResult := 201733) (rightOrdinal := 24) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨34989⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68842⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨34989⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68842⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨34989⟩⟩], [⟨.program ⟨257⟩, ⟨68842⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge203164

namespace LeftMerge203165
def owner : Owner := ⟨.program ⟨257⟩, ⟨71300⟩⟩
def mergeEvent : Nat := 203165
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7220⟩⟩, ⟨.program ⟨257⟩, ⟨71297⟩⟩] } }
def leftRaw : List Term := Proof.Events793.exact203149RawTerms
def rightRaw : List Term := Proof.Events788.exact201733RawTerms
def group : MergeGroup := .operator 203149 201733
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 203149) (leftOrdinal := 11)
    (rightResult := 201733) (rightOrdinal := 11) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7220⟩⟩, ⟨.program ⟨257⟩, ⟨71297⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7220⟩⟩, ⟨.program ⟨257⟩, ⟨71297⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩, ⟨.program ⟨257⟩, ⟨71297⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge203165

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
