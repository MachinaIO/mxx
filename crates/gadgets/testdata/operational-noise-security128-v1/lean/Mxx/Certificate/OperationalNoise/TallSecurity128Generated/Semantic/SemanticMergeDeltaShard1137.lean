import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge186033
def owner : Owner := ⟨.program ⟨257⟩, ⟨12727⟩⟩
def mergeEvent : Nat := 186033
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨12726⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events033.exact8693RawTerms
def rightRaw : List Term := Proof.Events696.exact178278RawTerms
def group : MergeGroup := .operator 8693 178278
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 8693) (leftOrdinal := 0)
    (rightResult := 178278) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨12726⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨12726⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge186033

namespace LeftMerge186038
def owner : Owner := ⟨.program ⟨257⟩, ⟨8925⟩⟩
def mergeEvent : Nat := 186038
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7277⟩⟩] } }
def leftRaw : List Term := Proof.Events695.exact178148RawTerms
def rightRaw : List Term := Proof.Events098.exact25137RawTerms
def group : MergeGroup := .operator 178148 25137
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 178148) (leftOrdinal := 0)
    (rightResult := 25137) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7277⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7277⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge186038

namespace LeftMerge186055
def owner : Owner := ⟨.program ⟨257⟩, ⟨12730⟩⟩
def mergeEvent : Nat := 186055
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨12726⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩] } }
def leftRaw : List Term := Proof.Events726.exact186049RawTerms
def rightRaw : List Term := Proof.Events098.exact25126RawTerms
def group : MergeGroup := .operator 186049 25126
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 186049) (leftOrdinal := 1)
    (rightResult := 25126) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨12726⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9571⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨12726⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge186055

namespace LeftMerge186057
def owner : Owner := ⟨.program ⟨257⟩, ⟨12730⟩⟩
def mergeEvent : Nat := 186057
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨12726⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7305⟩⟩] } }
def rhsRaw : List Term := Proof.Events098.exact25096RawTerms
def group : MergeGroup := .relation 186056
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 186056) (rhsResult := 25096)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨12726⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨9571⟩⟩) ⟨7305⟩ 25096) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7305⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨12726⟩⟩], [⟨.program ⟨257⟩, ⟨7305⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge186057

namespace LeftMerge186058
def owner : Owner := ⟨.program ⟨257⟩, ⟨12730⟩⟩
def mergeEvent : Nat := 186058
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩] } }
def leftRaw : List Term := Proof.Events726.exact186049RawTerms
def rightRaw : List Term := Proof.Events098.exact25126RawTerms
def group : MergeGroup := .operator 186049 25126
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 186049) (leftOrdinal := 0)
    (rightResult := 25126) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7277⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9571⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge186058

namespace LeftMerge186063
def owner : Owner := ⟨.program ⟨257⟩, ⟨18353⟩⟩
def mergeEvent : Nat := 186063
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨12726⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7305⟩⟩] } }
def leftRaw : List Term := Proof.Events726.exact186059RawTerms
def rightRaw : List Term := Proof.Events726.exact186029RawTerms
def group : MergeGroup := .operator 186059 186029
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 186059) (leftOrdinal := 1)
    (rightResult := 186029) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨12726⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7305⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨12726⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7305⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨12726⟩⟩], [⟨.program ⟨257⟩, ⟨7305⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge186063

namespace LeftMerge186071
def owner : Owner := ⟨.program ⟨257⟩, ⟨20253⟩⟩
def mergeEvent : Nat := 186071
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨12726⟩⟩, ⟨.program ⟨257⟩, ⟨18346⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨20252⟩⟩] } }
def leftRaw : List Term := Proof.Events726.exact186065RawTerms
def rightRaw : List Term := Proof.Events726.exact186001RawTerms
def group : MergeGroup := .operator 186065 186001
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 186065) (leftOrdinal := 1)
    (rightResult := 186001) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨12726⟩⟩, ⟨.program ⟨257⟩, ⟨18346⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨20252⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨12726⟩⟩, ⟨.program ⟨257⟩, ⟨18346⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨20252⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge186071

namespace LeftMerge186073
def owner : Owner := ⟨.program ⟨257⟩, ⟨20253⟩⟩
def mergeEvent : Nat := 186073
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨12726⟩⟩, ⟨.program ⟨257⟩, ⟨18346⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨19727⟩⟩] } }
def rhsRaw : List Term := Proof.Events726.exact185998RawTerms
def group : MergeGroup := .relation 186072
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 186072) (rhsResult := 185998)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨12726⟩⟩, ⟨.program ⟨257⟩, ⟨18346⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨20252⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨20252⟩⟩) ⟨19727⟩ 185998) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨19727⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨12726⟩⟩, ⟨.program ⟨257⟩, ⟨18346⟩⟩], [⟨.program ⟨257⟩, ⟨19727⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge186073

namespace LeftMerge186074
def owner : Owner := ⟨.program ⟨257⟩, ⟨20253⟩⟩
def mergeEvent : Nat := 186074
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩, ⟨.program ⟨257⟩, ⟨20252⟩⟩] } }
def leftRaw : List Term := Proof.Events726.exact186065RawTerms
def rightRaw : List Term := Proof.Events726.exact186001RawTerms
def group : MergeGroup := .operator 186065 186001
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 186065) (leftOrdinal := 0)
    (rightResult := 186001) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨20252⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩, ⟨.program ⟨257⟩, ⟨20252⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge186074

namespace LeftMerge186088
def owner : Owner := ⟨.program ⟨257⟩, ⟨19182⟩⟩
def mergeEvent : Nat := 186088
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19179⟩⟩] } }
def leftRaw : List Term := Proof.Events696.exact178370RawTerms
def rightRaw : List Term := Proof.Events726.exact186082RawTerms
def group : MergeGroup := .operator 178370 186082
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 178370) (leftOrdinal := 0)
    (rightResult := 186082) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨19179⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19179⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge186088

namespace LeftMerge186167
def owner : Owner := ⟨.program ⟨257⟩, ⟨18347⟩⟩
def mergeEvent : Nat := 186167
def frameStart : Nat := 186137
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨12726⟩⟩, ⟨.program ⟨257⟩, ⟨18346⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events727.exact186163RawTerms
def rightRaw : List Term := Proof.Events727.exact186160RawTerms
def group : MergeGroup := .operator 186163 186160
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 186163) (leftOrdinal := 0)
    (rightResult := 186160) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨12726⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨18346⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨12726⟩⟩, ⟨.program ⟨257⟩, ⟨18346⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge186167

namespace LeftMerge186197
def owner : Owner := ⟨.program ⟨257⟩, ⟨20000⟩⟩
def mergeEvent : Nat := 186197
def frameStart : Nat := 186137
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨12726⟩⟩, ⟨.program ⟨257⟩, ⟨18346⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events727.exact186193RawTerms
def rightRaw : List Term := Proof.Events727.exact186191RawTerms
def group : MergeGroup := .operator 186193 186191
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 186193) (leftOrdinal := 0)
    (rightResult := 186191) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨12726⟩⟩, ⟨.program ⟨257⟩, ⟨18346⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨12726⟩⟩, ⟨.program ⟨257⟩, ⟨18346⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge186197

namespace LeftMerge186220
def owner : Owner := ⟨.program ⟨257⟩, ⟨9573⟩⟩
def mergeEvent : Nat := 186220
def frameStart : Nat := 186137
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩] } }
def leftRaw : List Term := Proof.Events727.exact186216RawTerms
def rightRaw : List Term := Proof.Events727.exact186213RawTerms
def group : MergeGroup := .operator 186216 186213
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 186216) (leftOrdinal := 0)
    (rightResult := 186213) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7277⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9571⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge186220

namespace LeftMerge186229
def owner : Owner := ⟨.program ⟨257⟩, ⟨20255⟩⟩
def mergeEvent : Nat := 186229
def frameStart : Nat := 186137
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩, ⟨.program ⟨257⟩, ⟨20252⟩⟩] } }
def leftRaw : List Term := Proof.Events727.exact186225RawTerms
def rightRaw : List Term := Proof.Events727.exact186182RawTerms
def group : MergeGroup := .operator 186225 186182
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 186225) (leftOrdinal := 0)
    (rightResult := 186182) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨20252⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩, ⟨.program ⟨257⟩, ⟨20252⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge186229

namespace LeftMerge186230
def owner : Owner := ⟨.program ⟨257⟩, ⟨20255⟩⟩
def mergeEvent : Nat := 186230
def frameStart : Nat := 186137
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨12726⟩⟩, ⟨.program ⟨257⟩, ⟨18346⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨20252⟩⟩] } }
def leftRaw : List Term := Proof.Events727.exact186225RawTerms
def rightRaw : List Term := Proof.Events727.exact186182RawTerms
def group : MergeGroup := .operator 186225 186182
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 186225) (leftOrdinal := 1)
    (rightResult := 186182) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨12726⟩⟩, ⟨.program ⟨257⟩, ⟨18346⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨20252⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨12726⟩⟩, ⟨.program ⟨257⟩, ⟨18346⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨20252⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge186230

namespace LeftMerge186232
def owner : Owner := ⟨.program ⟨257⟩, ⟨20255⟩⟩
def mergeEvent : Nat := 186232
def frameStart : Nat := 186137
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨12726⟩⟩, ⟨.program ⟨257⟩, ⟨18346⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨19727⟩⟩] } }
def rhsRaw : List Term := Proof.Events727.exact186179RawTerms
def group : MergeGroup := .relation 186231
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 186231) (rhsResult := 186179)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨12726⟩⟩, ⟨.program ⟨257⟩, ⟨18346⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨20252⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨20252⟩⟩) ⟨19727⟩ 186179) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨19727⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨12726⟩⟩, ⟨.program ⟨257⟩, ⟨18346⟩⟩], [⟨.program ⟨257⟩, ⟨19727⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge186232

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
