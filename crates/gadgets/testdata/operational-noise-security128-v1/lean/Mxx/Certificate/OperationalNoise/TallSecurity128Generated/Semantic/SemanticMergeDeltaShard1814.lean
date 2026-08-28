import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge294076
def owner : Owner := ⟨.program ⟨257⟩, ⟨32575⟩⟩
def mergeEvent : Nat := 294076
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨31987⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events1148.exact294070RawTerms
def group : MergeGroup := .relation 294072
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 294072) (rhsResult := 294070)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32572⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 294071 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32572⟩⟩]⟩) (none) 294070) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨31987⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨31987⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge294076

namespace LeftMerge294081
def owner : Owner := ⟨.program ⟨257⟩, ⟨33702⟩⟩
def mergeEvent : Nat := 294081
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7182⟩⟩, ⟨.program ⟨257⟩, ⟨33699⟩⟩] } }
def leftRaw : List Term := Proof.Events1148.exact294077RawTerms
def rightRaw : List Term := Proof.Events1148.exact293899RawTerms
def group : MergeGroup := .operator 294077 293899
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 294077) (leftOrdinal := 0)
    (rightResult := 293899) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7182⟩⟩, ⟨.program ⟨257⟩, ⟨33699⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7182⟩⟩, ⟨.program ⟨257⟩, ⟨33699⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7182⟩⟩, ⟨.program ⟨257⟩, ⟨33699⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge294081

namespace LeftMerge294082
def owner : Owner := ⟨.program ⟨257⟩, ⟨33702⟩⟩
def mergeEvent : Nat := 294082
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨31780⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨33046⟩⟩] } }
def leftRaw : List Term := Proof.Events1148.exact294077RawTerms
def rightRaw : List Term := Proof.Events1148.exact293899RawTerms
def group : MergeGroup := .operator 294077 293899
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 294077) (leftOrdinal := 2)
    (rightResult := 293899) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨31780⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨33046⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨31780⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨33046⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨31780⟩⟩], [⟨.program ⟨257⟩, ⟨33046⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge294082

namespace LeftMerge294090
def owner : Owner := ⟨.program ⟨257⟩, ⟨33703⟩⟩
def mergeEvent : Nat := 294090
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩] } }
def leftRaw : List Term := Proof.Events1148.exact294084RawTerms
def rightRaw : List Term := Proof.Events061.exact15822RawTerms
def group : MergeGroup := .operator 294084 15822
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 294084) (leftOrdinal := 0)
    (rightResult := 15822) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7203⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7145⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge294090

namespace LeftMerge294091
def owner : Owner := ⟨.program ⟨257⟩, ⟨33703⟩⟩
def mergeEvent : Nat := 294091
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨31987⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩] } }
def leftRaw : List Term := Proof.Events1148.exact294084RawTerms
def rightRaw : List Term := Proof.Events061.exact15822RawTerms
def group : MergeGroup := .operator 294084 15822
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 294084) (leftOrdinal := 1)
    (rightResult := 15822) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨31987⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7145⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨31987⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge294091

namespace LeftMerge294093
def owner : Owner := ⟨.program ⟨257⟩, ⟨33703⟩⟩
def mergeEvent : Nat := 294093
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨31987⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events061.exact15815RawTerms
def group : MergeGroup := .relation 294092
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 294092) (rhsResult := 15815)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨31987⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨7145⟩⟩) ⟨7038⟩ 15815) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6794⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨31987⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge294093

namespace LeftMerge294107
def owner : Owner := ⟨.program ⟨257⟩, ⟨23681⟩⟩
def mergeEvent : Nat := 294107
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7181⟩⟩, ⟨.program ⟨257⟩, ⟨23679⟩⟩] } }
def leftRaw : List Term := Proof.Events1125.exact288129RawTerms
def rightRaw : List Term := Proof.Events1148.exact294101RawTerms
def group : MergeGroup := .operator 288129 294101
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 288129) (leftOrdinal := 0)
    (rightResult := 294101) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7181⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨23679⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7181⟩⟩, ⟨.program ⟨257⟩, ⟨23679⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge294107

namespace LeftMerge294108
def owner : Owner := ⟨.program ⟨257⟩, ⟨23681⟩⟩
def mergeEvent : Nat := 294108
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨21760⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨23679⟩⟩] } }
def leftRaw : List Term := Proof.Events1125.exact288129RawTerms
def rightRaw : List Term := Proof.Events1148.exact294101RawTerms
def group : MergeGroup := .operator 288129 294101
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 288129) (leftOrdinal := 1)
    (rightResult := 294101) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨21760⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨23679⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨21760⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨23679⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge294108

namespace LeftMerge294110
def owner : Owner := ⟨.program ⟨257⟩, ⟨23681⟩⟩
def mergeEvent : Nat := 294110
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨21760⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨23026⟩⟩] } }
def rhsRaw : List Term := Proof.Events1148.exact294098RawTerms
def group : MergeGroup := .relation 294109
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 294109) (rhsResult := 294098)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨21760⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨23679⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨23679⟩⟩) ⟨23026⟩ 294098) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨23026⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨21760⟩⟩], [⟨.program ⟨257⟩, ⟨23026⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge294110

namespace LeftMerge294124
def owner : Owner := ⟨.program ⟨257⟩, ⟨22555⟩⟩
def mergeEvent : Nat := 294124
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22552⟩⟩] } }
def leftRaw : List Term := Proof.Events1096.exact280745RawTerms
def rightRaw : List Term := Proof.Events1148.exact294118RawTerms
def group : MergeGroup := .operator 280745 294118
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 280745) (leftOrdinal := 0)
    (rightResult := 294118) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨22552⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22552⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge294124

namespace LeftMerge294245
def owner : Owner := ⟨.program ⟨257⟩, ⟨23264⟩⟩
def mergeEvent : Nat := 294245
def frameStart : Nat := 294179
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨21760⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1149.exact294241RawTerms
def rightRaw : List Term := Proof.Events1149.exact294239RawTerms
def group : MergeGroup := .operator 294241 294239
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 294241) (leftOrdinal := 0)
    (rightResult := 294239) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨21760⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨21760⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge294245

namespace LeftMerge294257
def owner : Owner := ⟨.program ⟨257⟩, ⟨23680⟩⟩
def mergeEvent : Nat := 294257
def frameStart : Nat := 294179
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7181⟩⟩, ⟨.program ⟨257⟩, ⟨23679⟩⟩] } }
def leftRaw : List Term := Proof.Events1149.exact294253RawTerms
def rightRaw : List Term := Proof.Events1149.exact294230RawTerms
def group : MergeGroup := .operator 294253 294230
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 294253) (leftOrdinal := 0)
    (rightResult := 294230) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7181⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨23679⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7181⟩⟩, ⟨.program ⟨257⟩, ⟨23679⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge294257

namespace LeftMerge294258
def owner : Owner := ⟨.program ⟨257⟩, ⟨23680⟩⟩
def mergeEvent : Nat := 294258
def frameStart : Nat := 294179
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨21760⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨23679⟩⟩] } }
def leftRaw : List Term := Proof.Events1149.exact294253RawTerms
def rightRaw : List Term := Proof.Events1149.exact294230RawTerms
def group : MergeGroup := .operator 294253 294230
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 294253) (leftOrdinal := 1)
    (rightResult := 294230) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨21760⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨23679⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨21760⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨23679⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge294258

namespace LeftMerge294260
def owner : Owner := ⟨.program ⟨257⟩, ⟨23680⟩⟩
def mergeEvent : Nat := 294260
def frameStart : Nat := 294179
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨21760⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨23026⟩⟩] } }
def rhsRaw : List Term := Proof.Events1149.exact294227RawTerms
def group : MergeGroup := .relation 294259
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 294259) (rhsResult := 294227)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨21760⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨23679⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨23679⟩⟩) ⟨23026⟩ 294227) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨23026⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨21760⟩⟩], [⟨.program ⟨257⟩, ⟨23026⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge294260

namespace LeftMerge294268
def owner : Owner := ⟨.program ⟨257⟩, ⟨21970⟩⟩
def mergeEvent : Nat := 294268
def frameStart : Nat := 294179
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨21967⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1149.exact294241RawTerms
def rightRaw : List Term := Proof.Events1149.exact294264RawTerms
def group : MergeGroup := .operator 294241 294264
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 294241) (leftOrdinal := 0)
    (rightResult := 294264) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨21967⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨21967⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge294268

namespace LeftMerge294285
def owner : Owner := ⟨.program ⟨257⟩, ⟨22555⟩⟩
def mergeEvent : Nat := 294285
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7201⟩⟩] } }
def rhsRaw : List Term := Proof.Events1149.exact294282RawTerms
def group : MergeGroup := .relation 294284
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 294284) (rhsResult := 294282)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22552⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 294283 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22552⟩⟩]⟩) (none) 294282) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7201⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge294285

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
