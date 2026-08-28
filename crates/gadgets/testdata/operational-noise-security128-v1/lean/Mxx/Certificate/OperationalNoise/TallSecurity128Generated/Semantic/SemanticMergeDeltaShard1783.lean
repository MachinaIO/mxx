import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge289020
def owner : Owner := ⟨.program ⟨257⟩, ⟨17104⟩⟩
def mergeEvent : Nat := 289020
def frameStart : Nat := 288960
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨12291⟩⟩, ⟨.program ⟨257⟩, ⟨15330⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1128.exact289016RawTerms
def rightRaw : List Term := Proof.Events1128.exact289014RawTerms
def group : MergeGroup := .operator 289016 289014
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 289016) (leftOrdinal := 0)
    (rightResult := 289014) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨12291⟩⟩, ⟨.program ⟨257⟩, ⟨15330⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨12291⟩⟩, ⟨.program ⟨257⟩, ⟨15330⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge289020

namespace LeftMerge289041
def owner : Owner := ⟨.program ⟨257⟩, ⟨9570⟩⟩
def mergeEvent : Nat := 289041
def frameStart : Nat := 288960
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩] } }
def leftRaw : List Term := Proof.Events1129.exact289037RawTerms
def rightRaw : List Term := Proof.Events1129.exact289034RawTerms
def group : MergeGroup := .operator 289037 289034
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 289037) (leftOrdinal := 0)
    (rightResult := 289034) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7303⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9568⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge289041

namespace LeftMerge289050
def owner : Owner := ⟨.program ⟨257⟩, ⟨17296⟩⟩
def mergeEvent : Nat := 289050
def frameStart : Nat := 288960
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17293⟩⟩] } }
def leftRaw : List Term := Proof.Events1129.exact289046RawTerms
def rightRaw : List Term := Proof.Events1128.exact289005RawTerms
def group : MergeGroup := .operator 289046 289005
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 289046) (leftOrdinal := 0)
    (rightResult := 289005) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨17293⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17293⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge289050

namespace LeftMerge289051
def owner : Owner := ⟨.program ⟨257⟩, ⟨17296⟩⟩
def mergeEvent : Nat := 289051
def frameStart : Nat := 288960
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨12291⟩⟩, ⟨.program ⟨257⟩, ⟨15330⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17293⟩⟩] } }
def leftRaw : List Term := Proof.Events1129.exact289046RawTerms
def rightRaw : List Term := Proof.Events1128.exact289005RawTerms
def group : MergeGroup := .operator 289046 289005
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 289046) (leftOrdinal := 1)
    (rightResult := 289005) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨12291⟩⟩, ⟨.program ⟨257⟩, ⟨15330⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨17293⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨12291⟩⟩, ⟨.program ⟨257⟩, ⟨15330⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17293⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge289051

namespace LeftMerge289053
def owner : Owner := ⟨.program ⟨257⟩, ⟨17296⟩⟩
def mergeEvent : Nat := 289053
def frameStart : Nat := 288960
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨12291⟩⟩, ⟨.program ⟨257⟩, ⟨15330⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨16813⟩⟩] } }
def rhsRaw : List Term := Proof.Events1128.exact289002RawTerms
def group : MergeGroup := .relation 289052
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 289052) (rhsResult := 289002)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨12291⟩⟩, ⟨.program ⟨257⟩, ⟨15330⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17293⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨17293⟩⟩) ⟨16813⟩ 289002) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨16813⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨12291⟩⟩, ⟨.program ⟨257⟩, ⟨15330⟩⟩], [⟨.program ⟨257⟩, ⟨16813⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge289053

namespace LeftMerge289061
def owner : Owner := ⟨.program ⟨257⟩, ⟨15742⟩⟩
def mergeEvent : Nat := 289061
def frameStart : Nat := 288960
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨15740⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1128.exact289016RawTerms
def rightRaw : List Term := Proof.Events1129.exact289057RawTerms
def group : MergeGroup := .operator 289016 289057
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 289016) (leftOrdinal := 0)
    (rightResult := 289057) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨15740⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨15740⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge289061

namespace LeftMerge289078
def owner : Owner := ⟨.program ⟨257⟩, ⟨16232⟩⟩
def mergeEvent : Nat := 289078
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7179⟩⟩] } }
def rhsRaw : List Term := Proof.Events1129.exact289075RawTerms
def group : MergeGroup := .relation 289077
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 289077) (rhsResult := 289075)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16229⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 289076 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16229⟩⟩]⟩) (none) 289075) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7179⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7179⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge289078

namespace LeftMerge289079
def owner : Owner := ⟨.program ⟨257⟩, ⟨16232⟩⟩
def mergeEvent : Nat := 289079
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17293⟩⟩] } }
def rhsRaw : List Term := Proof.Events1129.exact289075RawTerms
def group : MergeGroup := .relation 289077
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 289077) (rhsResult := 289075)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16229⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 289076 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16229⟩⟩]⟩) (none) 289075) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17293⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17293⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge289079

namespace LeftMerge289080
def owner : Owner := ⟨.program ⟨257⟩, ⟨16232⟩⟩
def mergeEvent : Nat := 289080
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨12291⟩⟩, ⟨.program ⟨257⟩, ⟨15330⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨16813⟩⟩] } }
def rhsRaw : List Term := Proof.Events1129.exact289075RawTerms
def group : MergeGroup := .relation 289077
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 289077) (rhsResult := 289075)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16229⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 289076 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16229⟩⟩]⟩) (none) 289075) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨12291⟩⟩, ⟨.program ⟨257⟩, ⟨15330⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨16813⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨12291⟩⟩, ⟨.program ⟨257⟩, ⟨15330⟩⟩], [⟨.program ⟨257⟩, ⟨16813⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge289080

namespace LeftMerge289081
def owner : Owner := ⟨.program ⟨257⟩, ⟨16232⟩⟩
def mergeEvent : Nat := 289081
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨15740⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events1129.exact289075RawTerms
def group : MergeGroup := .relation 289077
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 289077) (rhsResult := 289075)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16229⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 289076 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16229⟩⟩]⟩) (none) 289075) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨15740⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨15740⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge289081

namespace LeftMerge289086
def owner : Owner := ⟨.program ⟨257⟩, ⟨17295⟩⟩
def mergeEvent : Nat := 289086
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨12291⟩⟩, ⟨.program ⟨257⟩, ⟨15330⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨16813⟩⟩] } }
def leftRaw : List Term := Proof.Events1129.exact289082RawTerms
def rightRaw : List Term := Proof.Events1128.exact288898RawTerms
def group : MergeGroup := .operator 289082 288898
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 289082) (leftOrdinal := 2)
    (rightResult := 288898) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨12291⟩⟩, ⟨.program ⟨257⟩, ⟨15330⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨16813⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨12291⟩⟩, ⟨.program ⟨257⟩, ⟨15330⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨16813⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨12291⟩⟩, ⟨.program ⟨257⟩, ⟨15330⟩⟩], [⟨.program ⟨257⟩, ⟨16813⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge289086

namespace LeftMerge289087
def owner : Owner := ⟨.program ⟨257⟩, ⟨17295⟩⟩
def mergeEvent : Nat := 289087
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17293⟩⟩] } }
def leftRaw : List Term := Proof.Events1129.exact289082RawTerms
def rightRaw : List Term := Proof.Events1128.exact288898RawTerms
def group : MergeGroup := .operator 289082 288898
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 289082) (leftOrdinal := 1)
    (rightResult := 288898) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17293⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17293⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17293⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge289087

namespace LeftMerge289095
def owner : Owner := ⟨.program ⟨257⟩, ⟨17595⟩⟩
def mergeEvent : Nat := 289095
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17593⟩⟩] } }
def leftRaw : List Term := Proof.Events1129.exact289089RawTerms
def rightRaw : List Term := Proof.Events1128.exact288814RawTerms
def group : MergeGroup := .operator 289089 288814
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 289089) (leftOrdinal := 0)
    (rightResult := 288814) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7179⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨17593⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17593⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge289095

namespace LeftMerge289096
def owner : Owner := ⟨.program ⟨257⟩, ⟨17595⟩⟩
def mergeEvent : Nat := 289096
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨15740⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17593⟩⟩] } }
def leftRaw : List Term := Proof.Events1129.exact289089RawTerms
def rightRaw : List Term := Proof.Events1128.exact288814RawTerms
def group : MergeGroup := .operator 289089 288814
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 289089) (leftOrdinal := 1)
    (rightResult := 288814) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨15740⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨17593⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨15740⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17593⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge289096

namespace LeftMerge289098
def owner : Owner := ⟨.program ⟨257⟩, ⟨17595⟩⟩
def mergeEvent : Nat := 289098
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨15740⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨16947⟩⟩] } }
def rhsRaw : List Term := Proof.Events1128.exact288811RawTerms
def group : MergeGroup := .relation 289097
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 289097) (rhsResult := 288811)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨15740⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17593⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨17593⟩⟩) ⟨16947⟩ 288811) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨16947⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨15740⟩⟩], [⟨.program ⟨257⟩, ⟨16947⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge289098

namespace LeftMerge289112
def owner : Owner := ⟨.program ⟨257⟩, ⟨16479⟩⟩
def mergeEvent : Nat := 289112
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16476⟩⟩] } }
def leftRaw : List Term := Proof.Events1096.exact280745RawTerms
def rightRaw : List Term := Proof.Events1129.exact289106RawTerms
def group : MergeGroup := .operator 280745 289106
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 280745) (leftOrdinal := 0)
    (rightResult := 289106) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨16476⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16476⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge289112

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
