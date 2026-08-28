import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge194109
def owner : Owner := ⟨.program ⟨257⟩, ⟨44324⟩⟩
def mergeEvent : Nat := 194109
def frameStart : Nat := 194014
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14511⟩⟩, ⟨.program ⟨257⟩, ⟨42522⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨43801⟩⟩] } }
def rhsRaw : List Term := Proof.Events758.exact194056RawTerms
def group : MergeGroup := .relation 194108
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 194108) (rhsResult := 194056)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨14511⟩⟩, ⟨.program ⟨257⟩, ⟨42522⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨44321⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨44321⟩⟩) ⟨43801⟩ 194056) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨43801⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨14511⟩⟩, ⟨.program ⟨257⟩, ⟨42522⟩⟩], [⟨.program ⟨257⟩, ⟨43801⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge194109

namespace LeftMerge194117
def owner : Owner := ⟨.program ⟨257⟩, ⟨42806⟩⟩
def mergeEvent : Nat := 194117
def frameStart : Nat := 194014
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨42804⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events758.exact194070RawTerms
def rightRaw : List Term := Proof.Events758.exact194113RawTerms
def group : MergeGroup := .operator 194070 194113
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 194070) (leftOrdinal := 0)
    (rightResult := 194113) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨42804⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨42804⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge194117

namespace LeftMerge194134
def owner : Owner := ⟨.program ⟨257⟩, ⟨43252⟩⟩
def mergeEvent : Nat := 194134
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7194⟩⟩] } }
def rhsRaw : List Term := Proof.Events758.exact194131RawTerms
def group : MergeGroup := .relation 194133
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 194133) (rhsResult := 194131)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43249⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 194132 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43249⟩⟩]⟩) (none) 194131) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7194⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7194⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge194134

namespace LeftMerge194135
def owner : Owner := ⟨.program ⟨257⟩, ⟨43252⟩⟩
def mergeEvent : Nat := 194135
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩, ⟨.program ⟨257⟩, ⟨44321⟩⟩] } }
def rhsRaw : List Term := Proof.Events758.exact194131RawTerms
def group : MergeGroup := .relation 194133
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 194133) (rhsResult := 194131)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43249⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 194132 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43249⟩⟩]⟩) (none) 194131) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩, ⟨.program ⟨257⟩, ⟨44321⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩, ⟨.program ⟨257⟩, ⟨44321⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge194135

namespace LeftMerge194136
def owner : Owner := ⟨.program ⟨257⟩, ⟨43252⟩⟩
def mergeEvent : Nat := 194136
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨14511⟩⟩, ⟨.program ⟨257⟩, ⟨42522⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨43801⟩⟩] } }
def rhsRaw : List Term := Proof.Events758.exact194131RawTerms
def group : MergeGroup := .relation 194133
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 194133) (rhsResult := 194131)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43249⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 194132 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43249⟩⟩]⟩) (none) 194131) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14511⟩⟩, ⟨.program ⟨257⟩, ⟨42522⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨43801⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨14511⟩⟩, ⟨.program ⟨257⟩, ⟨42522⟩⟩], [⟨.program ⟨257⟩, ⟨43801⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge194136

namespace LeftMerge194137
def owner : Owner := ⟨.program ⟨257⟩, ⟨43252⟩⟩
def mergeEvent : Nat := 194137
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨42804⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events758.exact194131RawTerms
def group : MergeGroup := .relation 194133
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 194133) (rhsResult := 194131)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43249⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 194132 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43249⟩⟩]⟩) (none) 194131) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨42804⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨42804⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge194137

namespace LeftMerge194142
def owner : Owner := ⟨.program ⟨257⟩, ⟨44323⟩⟩
def mergeEvent : Nat := 194142
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨14511⟩⟩, ⟨.program ⟨257⟩, ⟨42522⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨43801⟩⟩] } }
def leftRaw : List Term := Proof.Events758.exact194138RawTerms
def rightRaw : List Term := Proof.Events757.exact193952RawTerms
def group : MergeGroup := .operator 194138 193952
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 194138) (leftOrdinal := 2)
    (rightResult := 193952) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨14511⟩⟩, ⟨.program ⟨257⟩, ⟨42522⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨43801⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨14511⟩⟩, ⟨.program ⟨257⟩, ⟨42522⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨43801⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨14511⟩⟩, ⟨.program ⟨257⟩, ⟨42522⟩⟩], [⟨.program ⟨257⟩, ⟨43801⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge194142

namespace LeftMerge194143
def owner : Owner := ⟨.program ⟨257⟩, ⟨44323⟩⟩
def mergeEvent : Nat := 194143
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩, ⟨.program ⟨257⟩, ⟨44321⟩⟩] } }
def leftRaw : List Term := Proof.Events758.exact194138RawTerms
def rightRaw : List Term := Proof.Events757.exact193952RawTerms
def group : MergeGroup := .operator 194138 193952
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 194138) (leftOrdinal := 1)
    (rightResult := 193952) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩, ⟨.program ⟨257⟩, ⟨44321⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩, ⟨.program ⟨257⟩, ⟨44321⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩, ⟨.program ⟨257⟩, ⟨44321⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge194143

namespace LeftMerge194151
def owner : Owner := ⟨.program ⟨257⟩, ⟨44721⟩⟩
def mergeEvent : Nat := 194151
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44719⟩⟩] } }
def leftRaw : List Term := Proof.Events758.exact194145RawTerms
def rightRaw : List Term := Proof.Events757.exact193868RawTerms
def group : MergeGroup := .operator 194145 193868
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 194145) (leftOrdinal := 0)
    (rightResult := 193868) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7194⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨44719⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44719⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge194151

namespace LeftMerge194152
def owner : Owner := ⟨.program ⟨257⟩, ⟨44721⟩⟩
def mergeEvent : Nat := 194152
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨42804⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨44719⟩⟩] } }
def leftRaw : List Term := Proof.Events758.exact194145RawTerms
def rightRaw : List Term := Proof.Events757.exact193868RawTerms
def group : MergeGroup := .operator 194145 193868
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 194145) (leftOrdinal := 1)
    (rightResult := 193868) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨42804⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨44719⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨42804⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨44719⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge194152

namespace LeftMerge194154
def owner : Owner := ⟨.program ⟨257⟩, ⟨44721⟩⟩
def mergeEvent : Nat := 194154
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨42804⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨43959⟩⟩] } }
def rhsRaw : List Term := Proof.Events757.exact193865RawTerms
def group : MergeGroup := .relation 194153
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 194153) (rhsResult := 193865)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨42804⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨44719⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨44719⟩⟩) ⟨43959⟩ 193865) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨43959⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨42804⟩⟩], [⟨.program ⟨257⟩, ⟨43959⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge194154

namespace LeftMerge194168
def owner : Owner := ⟨.program ⟨257⟩, ⟨43579⟩⟩
def mergeEvent : Nat := 194168
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43576⟩⟩] } }
def leftRaw : List Term := Proof.Events753.exact192995RawTerms
def rightRaw : List Term := Proof.Events758.exact194162RawTerms
def group : MergeGroup := .operator 192995 194162
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 192995) (leftOrdinal := 0)
    (rightResult := 194162) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨43576⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43576⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge194168

namespace LeftMerge194289
def owner : Owner := ⟨.program ⟨257⟩, ⟨44156⟩⟩
def mergeEvent : Nat := 194289
def frameStart : Nat := 194223
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨42804⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events758.exact194285RawTerms
def rightRaw : List Term := Proof.Events758.exact194283RawTerms
def group : MergeGroup := .operator 194285 194283
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 194285) (leftOrdinal := 0)
    (rightResult := 194283) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨42804⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨42804⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge194289

namespace LeftMerge194301
def owner : Owner := ⟨.program ⟨257⟩, ⟨44720⟩⟩
def mergeEvent : Nat := 194301
def frameStart : Nat := 194223
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44719⟩⟩] } }
def leftRaw : List Term := Proof.Events758.exact194297RawTerms
def rightRaw : List Term := Proof.Events758.exact194274RawTerms
def group : MergeGroup := .operator 194297 194274
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 194297) (leftOrdinal := 0)
    (rightResult := 194274) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7194⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨44719⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44719⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge194301

namespace LeftMerge194302
def owner : Owner := ⟨.program ⟨257⟩, ⟨44720⟩⟩
def mergeEvent : Nat := 194302
def frameStart : Nat := 194223
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨42804⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨44719⟩⟩] } }
def leftRaw : List Term := Proof.Events758.exact194297RawTerms
def rightRaw : List Term := Proof.Events758.exact194274RawTerms
def group : MergeGroup := .operator 194297 194274
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 194297) (leftOrdinal := 1)
    (rightResult := 194274) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨42804⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨44719⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨42804⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨44719⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge194302

namespace LeftMerge194304
def owner : Owner := ⟨.program ⟨257⟩, ⟨44720⟩⟩
def mergeEvent : Nat := 194304
def frameStart : Nat := 194223
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨42804⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨43959⟩⟩] } }
def rhsRaw : List Term := Proof.Events758.exact194271RawTerms
def group : MergeGroup := .relation 194303
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 194303) (rhsResult := 194271)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨42804⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨44719⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨44719⟩⟩) ⟨43959⟩ 194271) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨43959⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨42804⟩⟩], [⟨.program ⟨257⟩, ⟨43959⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge194304

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
