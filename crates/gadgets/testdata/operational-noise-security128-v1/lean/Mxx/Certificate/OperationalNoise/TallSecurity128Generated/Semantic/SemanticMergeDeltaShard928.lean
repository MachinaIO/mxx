import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge152160
def owner : Owner := ⟨.program ⟨257⟩, ⟨30569⟩⟩
def mergeEvent : Nat := 152160
def frameStart : Nat := 152067
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨13236⟩⟩, ⟨.program ⟨257⟩, ⟨28702⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨30566⟩⟩] } }
def leftRaw : List Term := Proof.Events594.exact152155RawTerms
def rightRaw : List Term := Proof.Events594.exact152112RawTerms
def group : MergeGroup := .operator 152155 152112
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 152155) (leftOrdinal := 1)
    (rightResult := 152112) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13236⟩⟩, ⟨.program ⟨257⟩, ⟨28702⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨30566⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨13236⟩⟩, ⟨.program ⟨257⟩, ⟨28702⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨30566⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge152160

namespace LeftMerge152162
def owner : Owner := ⟨.program ⟨257⟩, ⟨30569⟩⟩
def mergeEvent : Nat := 152162
def frameStart : Nat := 152067
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨13236⟩⟩, ⟨.program ⟨257⟩, ⟨28702⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨30071⟩⟩] } }
def rhsRaw : List Term := Proof.Events594.exact152109RawTerms
def group : MergeGroup := .relation 152161
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 152161) (rhsResult := 152109)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨13236⟩⟩, ⟨.program ⟨257⟩, ⟨28702⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨30566⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨30566⟩⟩) ⟨30071⟩ 152109) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨30071⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨13236⟩⟩, ⟨.program ⟨257⟩, ⟨28702⟩⟩], [⟨.program ⟨257⟩, ⟨30071⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge152162

namespace LeftMerge152170
def owner : Owner := ⟨.program ⟨257⟩, ⟨29066⟩⟩
def mergeEvent : Nat := 152170
def frameStart : Nat := 152067
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨29064⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events594.exact152123RawTerms
def rightRaw : List Term := Proof.Events594.exact152166RawTerms
def group : MergeGroup := .operator 152123 152166
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 152123) (leftOrdinal := 0)
    (rightResult := 152166) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨29064⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨29064⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge152170

namespace LeftMerge152187
def owner : Owner := ⟨.program ⟨257⟩, ⟨29502⟩⟩
def mergeEvent : Nat := 152187
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7190⟩⟩] } }
def rhsRaw : List Term := Proof.Events594.exact152184RawTerms
def group : MergeGroup := .relation 152186
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 152186) (rhsResult := 152184)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29499⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 152185 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29499⟩⟩]⟩) (none) 152184) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7190⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7190⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge152187

namespace LeftMerge152188
def owner : Owner := ⟨.program ⟨257⟩, ⟨29502⟩⟩
def mergeEvent : Nat := 152188
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩, ⟨.program ⟨257⟩, ⟨30566⟩⟩] } }
def rhsRaw : List Term := Proof.Events594.exact152184RawTerms
def group : MergeGroup := .relation 152186
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 152186) (rhsResult := 152184)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29499⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 152185 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29499⟩⟩]⟩) (none) 152184) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩, ⟨.program ⟨257⟩, ⟨30566⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩, ⟨.program ⟨257⟩, ⟨30566⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge152188

namespace LeftMerge152189
def owner : Owner := ⟨.program ⟨257⟩, ⟨29502⟩⟩
def mergeEvent : Nat := 152189
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨13236⟩⟩, ⟨.program ⟨257⟩, ⟨28702⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨30071⟩⟩] } }
def rhsRaw : List Term := Proof.Events594.exact152184RawTerms
def group : MergeGroup := .relation 152186
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 152186) (rhsResult := 152184)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29499⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 152185 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29499⟩⟩]⟩) (none) 152184) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13236⟩⟩, ⟨.program ⟨257⟩, ⟨28702⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨30071⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨13236⟩⟩, ⟨.program ⟨257⟩, ⟨28702⟩⟩], [⟨.program ⟨257⟩, ⟨30071⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge152189

namespace LeftMerge152190
def owner : Owner := ⟨.program ⟨257⟩, ⟨29502⟩⟩
def mergeEvent : Nat := 152190
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨29064⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events594.exact152184RawTerms
def group : MergeGroup := .relation 152186
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 152186) (rhsResult := 152184)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29499⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 152185 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29499⟩⟩]⟩) (none) 152184) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨29064⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨29064⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge152190

namespace LeftMerge152195
def owner : Owner := ⟨.program ⟨257⟩, ⟨30568⟩⟩
def mergeEvent : Nat := 152195
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨13236⟩⟩, ⟨.program ⟨257⟩, ⟨28702⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨30071⟩⟩] } }
def leftRaw : List Term := Proof.Events594.exact152191RawTerms
def rightRaw : List Term := Proof.Events593.exact152005RawTerms
def group : MergeGroup := .operator 152191 152005
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 152191) (leftOrdinal := 2)
    (rightResult := 152005) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨13236⟩⟩, ⟨.program ⟨257⟩, ⟨28702⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨30071⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨13236⟩⟩, ⟨.program ⟨257⟩, ⟨28702⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨30071⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨13236⟩⟩, ⟨.program ⟨257⟩, ⟨28702⟩⟩], [⟨.program ⟨257⟩, ⟨30071⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge152195

namespace LeftMerge152196
def owner : Owner := ⟨.program ⟨257⟩, ⟨30568⟩⟩
def mergeEvent : Nat := 152196
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩, ⟨.program ⟨257⟩, ⟨30566⟩⟩] } }
def leftRaw : List Term := Proof.Events594.exact152191RawTerms
def rightRaw : List Term := Proof.Events593.exact152005RawTerms
def group : MergeGroup := .operator 152191 152005
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 152191) (leftOrdinal := 1)
    (rightResult := 152005) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩, ⟨.program ⟨257⟩, ⟨30566⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩, ⟨.program ⟨257⟩, ⟨30566⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩, ⟨.program ⟨257⟩, ⟨30566⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge152196

namespace LeftMerge152204
def owner : Owner := ⟨.program ⟨257⟩, ⟨30896⟩⟩
def mergeEvent : Nat := 152204
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨30894⟩⟩] } }
def leftRaw : List Term := Proof.Events594.exact152198RawTerms
def rightRaw : List Term := Proof.Events593.exact151921RawTerms
def group : MergeGroup := .operator 152198 151921
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 152198) (leftOrdinal := 0)
    (rightResult := 151921) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7190⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨30894⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨30894⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge152204

namespace LeftMerge152205
def owner : Owner := ⟨.program ⟨257⟩, ⟨30896⟩⟩
def mergeEvent : Nat := 152205
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨29064⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨30894⟩⟩] } }
def leftRaw : List Term := Proof.Events594.exact152198RawTerms
def rightRaw : List Term := Proof.Events593.exact151921RawTerms
def group : MergeGroup := .operator 152198 151921
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 152198) (leftOrdinal := 1)
    (rightResult := 151921) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨29064⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨30894⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨29064⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨30894⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge152205

namespace LeftMerge152207
def owner : Owner := ⟨.program ⟨257⟩, ⟨30896⟩⟩
def mergeEvent : Nat := 152207
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨29064⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨30214⟩⟩] } }
def rhsRaw : List Term := Proof.Events593.exact151918RawTerms
def group : MergeGroup := .relation 152206
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 152206) (rhsResult := 151918)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨29064⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨30894⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨30894⟩⟩) ⟨30214⟩ 151918) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨30214⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨29064⟩⟩], [⟨.program ⟨257⟩, ⟨30214⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge152207

namespace LeftMerge152221
def owner : Owner := ⟨.program ⟨257⟩, ⟨29779⟩⟩
def mergeEvent : Nat := 152221
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29776⟩⟩] } }
def leftRaw : List Term := Proof.Events582.exact149120RawTerms
def rightRaw : List Term := Proof.Events594.exact152215RawTerms
def group : MergeGroup := .operator 149120 152215
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 149120) (leftOrdinal := 0)
    (rightResult := 152215) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨29776⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29776⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge152221

namespace LeftMerge152342
def owner : Owner := ⟨.program ⟨257⟩, ⟨30436⟩⟩
def mergeEvent : Nat := 152342
def frameStart : Nat := 152276
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨29064⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events595.exact152338RawTerms
def rightRaw : List Term := Proof.Events595.exact152336RawTerms
def group : MergeGroup := .operator 152338 152336
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 152338) (leftOrdinal := 0)
    (rightResult := 152336) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨29064⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨29064⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge152342

namespace LeftMerge152354
def owner : Owner := ⟨.program ⟨257⟩, ⟨30895⟩⟩
def mergeEvent : Nat := 152354
def frameStart : Nat := 152276
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨30894⟩⟩] } }
def leftRaw : List Term := Proof.Events595.exact152350RawTerms
def rightRaw : List Term := Proof.Events595.exact152327RawTerms
def group : MergeGroup := .operator 152350 152327
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 152350) (leftOrdinal := 0)
    (rightResult := 152327) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7190⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨30894⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨30894⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge152354

namespace LeftMerge152355
def owner : Owner := ⟨.program ⟨257⟩, ⟨30895⟩⟩
def mergeEvent : Nat := 152355
def frameStart : Nat := 152276
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨29064⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨30894⟩⟩] } }
def leftRaw : List Term := Proof.Events595.exact152350RawTerms
def rightRaw : List Term := Proof.Events595.exact152327RawTerms
def group : MergeGroup := .operator 152350 152327
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 152350) (leftOrdinal := 1)
    (rightResult := 152327) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨29064⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨30894⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨29064⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨30894⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge152355

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
