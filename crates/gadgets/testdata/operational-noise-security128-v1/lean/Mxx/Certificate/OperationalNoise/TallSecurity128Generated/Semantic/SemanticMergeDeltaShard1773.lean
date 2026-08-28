import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge287160
def owner : Owner := ⟨.program ⟨257⟩, ⟨51392⟩⟩
def mergeEvent : Nat := 287160
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨24458⟩⟩, ⟨.program ⟨257⟩, ⟨50383⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨51973⟩⟩] } }
def rhsRaw : List Term := Proof.Events1121.exact287155RawTerms
def group : MergeGroup := .relation 287157
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 287157) (rhsResult := 287155)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51389⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 287156 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51389⟩⟩]⟩) (none) 287155) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24458⟩⟩, ⟨.program ⟨257⟩, ⟨50383⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨51973⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨24458⟩⟩, ⟨.program ⟨257⟩, ⟨50383⟩⟩], [⟨.program ⟨257⟩, ⟨51973⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge287160

namespace LeftMerge287161
def owner : Owner := ⟨.program ⟨257⟩, ⟨51392⟩⟩
def mergeEvent : Nat := 287161
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨50840⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events1121.exact287155RawTerms
def group : MergeGroup := .relation 287157
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 287157) (rhsResult := 287155)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51389⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 287156 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51389⟩⟩]⟩) (none) 287155) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨50840⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨50840⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge287161

namespace LeftMerge287166
def owner : Owner := ⟨.program ⟨257⟩, ⟨52455⟩⟩
def mergeEvent : Nat := 287166
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨24458⟩⟩, ⟨.program ⟨257⟩, ⟨50383⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨51973⟩⟩] } }
def leftRaw : List Term := Proof.Events1121.exact287162RawTerms
def rightRaw : List Term := Proof.Events1121.exact286978RawTerms
def group : MergeGroup := .operator 287162 286978
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 287162) (leftOrdinal := 2)
    (rightResult := 286978) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨24458⟩⟩, ⟨.program ⟨257⟩, ⟨50383⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨51973⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨24458⟩⟩, ⟨.program ⟨257⟩, ⟨50383⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨51973⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨24458⟩⟩, ⟨.program ⟨257⟩, ⟨50383⟩⟩], [⟨.program ⟨257⟩, ⟨51973⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge287166

namespace LeftMerge287167
def owner : Owner := ⟨.program ⟨257⟩, ⟨52455⟩⟩
def mergeEvent : Nat := 287167
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52453⟩⟩] } }
def leftRaw : List Term := Proof.Events1121.exact287162RawTerms
def rightRaw : List Term := Proof.Events1121.exact286978RawTerms
def group : MergeGroup := .operator 287162 286978
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 287162) (leftOrdinal := 1)
    (rightResult := 286978) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52453⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52453⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52453⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge287167

namespace LeftMerge287175
def owner : Owner := ⟨.program ⟨257⟩, ⟨52768⟩⟩
def mergeEvent : Nat := 287175
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨52766⟩⟩] } }
def leftRaw : List Term := Proof.Events1121.exact287169RawTerms
def rightRaw : List Term := Proof.Events1120.exact286894RawTerms
def group : MergeGroup := .operator 287169 286894
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 287169) (leftOrdinal := 0)
    (rightResult := 286894) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7183⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨52766⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨52766⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge287175

namespace LeftMerge287176
def owner : Owner := ⟨.program ⟨257⟩, ⟨52768⟩⟩
def mergeEvent : Nat := 287176
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨50840⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨52766⟩⟩] } }
def leftRaw : List Term := Proof.Events1121.exact287169RawTerms
def rightRaw : List Term := Proof.Events1120.exact286894RawTerms
def group : MergeGroup := .operator 287169 286894
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 287169) (leftOrdinal := 1)
    (rightResult := 286894) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨50840⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨52766⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨50840⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨52766⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge287176

namespace LeftMerge287178
def owner : Owner := ⟨.program ⟨257⟩, ⟨52768⟩⟩
def mergeEvent : Nat := 287178
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨50840⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨52107⟩⟩] } }
def rhsRaw : List Term := Proof.Events1120.exact286891RawTerms
def group : MergeGroup := .relation 287177
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 287177) (rhsResult := 286891)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨50840⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨52766⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨52766⟩⟩) ⟨52107⟩ 286891) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨52107⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨50840⟩⟩], [⟨.program ⟨257⟩, ⟨52107⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge287178

namespace LeftMerge287192
def owner : Owner := ⟨.program ⟨257⟩, ⟨51639⟩⟩
def mergeEvent : Nat := 287192
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51636⟩⟩] } }
def leftRaw : List Term := Proof.Events1096.exact280745RawTerms
def rightRaw : List Term := Proof.Events1121.exact287186RawTerms
def group : MergeGroup := .operator 280745 287186
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 280745) (leftOrdinal := 0)
    (rightResult := 287186) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨51636⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51636⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge287192

namespace LeftMerge287313
def owner : Owner := ⟨.program ⟨257⟩, ⟨52344⟩⟩
def mergeEvent : Nat := 287313
def frameStart : Nat := 287247
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨50840⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1122.exact287309RawTerms
def rightRaw : List Term := Proof.Events1122.exact287307RawTerms
def group : MergeGroup := .operator 287309 287307
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 287309) (leftOrdinal := 0)
    (rightResult := 287307) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨50840⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨50840⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge287313

namespace LeftMerge287325
def owner : Owner := ⟨.program ⟨257⟩, ⟨52767⟩⟩
def mergeEvent : Nat := 287325
def frameStart : Nat := 287247
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨52766⟩⟩] } }
def leftRaw : List Term := Proof.Events1122.exact287321RawTerms
def rightRaw : List Term := Proof.Events1122.exact287298RawTerms
def group : MergeGroup := .operator 287321 287298
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 287321) (leftOrdinal := 0)
    (rightResult := 287298) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7183⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨52766⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨52766⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge287325

namespace LeftMerge287326
def owner : Owner := ⟨.program ⟨257⟩, ⟨52767⟩⟩
def mergeEvent : Nat := 287326
def frameStart : Nat := 287247
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨50840⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨52766⟩⟩] } }
def leftRaw : List Term := Proof.Events1122.exact287321RawTerms
def rightRaw : List Term := Proof.Events1122.exact287298RawTerms
def group : MergeGroup := .operator 287321 287298
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 287321) (leftOrdinal := 1)
    (rightResult := 287298) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨50840⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨52766⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨50840⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨52766⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge287326

namespace LeftMerge287328
def owner : Owner := ⟨.program ⟨257⟩, ⟨52767⟩⟩
def mergeEvent : Nat := 287328
def frameStart : Nat := 287247
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨50840⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨52107⟩⟩] } }
def rhsRaw : List Term := Proof.Events1122.exact287295RawTerms
def group : MergeGroup := .relation 287327
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 287327) (rhsResult := 287295)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨50840⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨52766⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨52766⟩⟩) ⟨52107⟩ 287295) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨52107⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨50840⟩⟩], [⟨.program ⟨257⟩, ⟨52107⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge287328

namespace LeftMerge287336
def owner : Owner := ⟨.program ⟨257⟩, ⟨51049⟩⟩
def mergeEvent : Nat := 287336
def frameStart : Nat := 287247
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨51047⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1122.exact287309RawTerms
def rightRaw : List Term := Proof.Events1122.exact287332RawTerms
def group : MergeGroup := .operator 287309 287332
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 287309) (leftOrdinal := 0)
    (rightResult := 287332) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨51047⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨51047⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge287336

namespace LeftMerge287353
def owner : Owner := ⟨.program ⟨257⟩, ⟨51639⟩⟩
def mergeEvent : Nat := 287353
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7206⟩⟩] } }
def rhsRaw : List Term := Proof.Events1122.exact287350RawTerms
def group : MergeGroup := .relation 287352
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 287352) (rhsResult := 287350)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51636⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 287351 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51636⟩⟩]⟩) (none) 287350) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7206⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge287353

namespace LeftMerge287354
def owner : Owner := ⟨.program ⟨257⟩, ⟨51639⟩⟩
def mergeEvent : Nat := 287354
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨52766⟩⟩] } }
def rhsRaw : List Term := Proof.Events1122.exact287350RawTerms
def group : MergeGroup := .relation 287352
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 287352) (rhsResult := 287350)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51636⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 287351 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51636⟩⟩]⟩) (none) 287350) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨52766⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨52766⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge287354

namespace LeftMerge287355
def owner : Owner := ⟨.program ⟨257⟩, ⟨51639⟩⟩
def mergeEvent : Nat := 287355
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨50840⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨52107⟩⟩] } }
def rhsRaw : List Term := Proof.Events1122.exact287350RawTerms
def group : MergeGroup := .relation 287352
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 287352) (rhsResult := 287350)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51636⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 287351 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51636⟩⟩]⟩) (none) 287350) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨50840⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨52107⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨50840⟩⟩], [⟨.program ⟨257⟩, ⟨52107⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge287355

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
