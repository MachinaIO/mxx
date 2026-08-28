import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge56015
def owner : Owner := ⟨.program ⟨214⟩, ⟨14003⟩⟩
def mergeEvent : Nat := 56015
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨13999⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events010.exact2594RawTerms
def rightRaw : List Term := Proof.Events197.exact50670RawTerms
def group : MergeGroup := .operator 2594 50670
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 2594) (leftOrdinal := 0)
    (rightResult := 50670) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨13999⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨13999⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge56015

namespace LeftMerge56020
def owner : Owner := ⟨.program ⟨214⟩, ⟨7252⟩⟩
def mergeEvent : Nat := 56020
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6758⟩⟩] } }
def leftRaw : List Term := Proof.Events197.exact50540RawTerms
def rightRaw : List Term := Proof.Events046.exact12024RawTerms
def group : MergeGroup := .operator 50540 12024
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 50540) (leftOrdinal := 0)
    (rightResult := 12024) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6758⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6758⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge56020

namespace LeftMerge56037
def owner : Owner := ⟨.program ⟨214⟩, ⟨14006⟩⟩
def mergeEvent : Nat := 56037
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨13999⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7849⟩⟩] } }
def leftRaw : List Term := Proof.Events218.exact56031RawTerms
def rightRaw : List Term := Proof.Events046.exact12013RawTerms
def group : MergeGroup := .operator 56031 12013
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 56031) (leftOrdinal := 1)
    (rightResult := 12013) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨13999⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨7849⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨13999⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7849⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge56037

namespace LeftMerge56039
def owner : Owner := ⟨.program ⟨214⟩, ⟨14006⟩⟩
def mergeEvent : Nat := 56039
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨13999⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6778⟩⟩] } }
def rhsRaw : List Term := Proof.Events046.exact11983RawTerms
def group : MergeGroup := .relation 56038
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 56038) (rhsResult := 11983)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨13999⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7849⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨7849⟩⟩) ⟨6778⟩ 11983) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6778⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨13999⟩⟩], [⟨.program ⟨214⟩, ⟨6778⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge56039

namespace LeftMerge56040
def owner : Owner := ⟨.program ⟨214⟩, ⟨14006⟩⟩
def mergeEvent : Nat := 56040
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6758⟩⟩, ⟨.program ⟨214⟩, ⟨7849⟩⟩] } }
def leftRaw : List Term := Proof.Events218.exact56031RawTerms
def rightRaw : List Term := Proof.Events046.exact12013RawTerms
def group : MergeGroup := .operator 56031 12013
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 56031) (leftOrdinal := 0)
    (rightResult := 12013) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6758⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨7849⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6758⟩⟩, ⟨.program ⟨214⟩, ⟨7849⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge56040

namespace LeftMerge56045
def owner : Owner := ⟨.program ⟨214⟩, ⟨14007⟩⟩
def mergeEvent : Nat := 56045
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨13999⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6778⟩⟩] } }
def leftRaw : List Term := Proof.Events218.exact56041RawTerms
def rightRaw : List Term := Proof.Events218.exact56011RawTerms
def group : MergeGroup := .operator 56041 56011
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 56041) (leftOrdinal := 1)
    (rightResult := 56011) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨13999⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6778⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨13999⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6778⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨13999⟩⟩], [⟨.program ⟨214⟩, ⟨6778⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge56045

namespace LeftMerge56053
def owner : Owner := ⟨.program ⟨214⟩, ⟨25995⟩⟩
def mergeEvent : Nat := 56053
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨11389⟩⟩, ⟨.program ⟨214⟩, ⟨13999⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25994⟩⟩] } }
def leftRaw : List Term := Proof.Events218.exact56047RawTerms
def rightRaw : List Term := Proof.Events218.exact55983RawTerms
def group : MergeGroup := .operator 56047 55983
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 56047) (leftOrdinal := 1)
    (rightResult := 55983) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨11389⟩⟩, ⟨.program ⟨214⟩, ⟨13999⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25994⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨11389⟩⟩, ⟨.program ⟨214⟩, ⟨13999⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25994⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge56053

namespace LeftMerge56055
def owner : Owner := ⟨.program ⟨214⟩, ⟨25995⟩⟩
def mergeEvent : Nat := 56055
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨11389⟩⟩, ⟨.program ⟨214⟩, ⟨13999⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23544⟩⟩] } }
def rhsRaw : List Term := Proof.Events218.exact55980RawTerms
def group : MergeGroup := .relation 56054
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 56054) (rhsResult := 55980)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨11389⟩⟩, ⟨.program ⟨214⟩, ⟨13999⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25994⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨25994⟩⟩) ⟨23544⟩ 55980) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23544⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨11389⟩⟩, ⟨.program ⟨214⟩, ⟨13999⟩⟩], [⟨.program ⟨214⟩, ⟨23544⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge56055

namespace LeftMerge56056
def owner : Owner := ⟨.program ⟨214⟩, ⟨25995⟩⟩
def mergeEvent : Nat := 56056
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6758⟩⟩, ⟨.program ⟨214⟩, ⟨7849⟩⟩, ⟨.program ⟨214⟩, ⟨25994⟩⟩] } }
def leftRaw : List Term := Proof.Events218.exact56047RawTerms
def rightRaw : List Term := Proof.Events218.exact55983RawTerms
def group : MergeGroup := .operator 56047 55983
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 56047) (leftOrdinal := 0)
    (rightResult := 55983) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6758⟩⟩, ⟨.program ⟨214⟩, ⟨7849⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25994⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6758⟩⟩, ⟨.program ⟨214⟩, ⟨7849⟩⟩, ⟨.program ⟨214⟩, ⟨25994⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge56056

namespace LeftMerge56070
def owner : Owner := ⟨.program ⟨214⟩, ⟨19463⟩⟩
def mergeEvent : Nat := 56070
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19460⟩⟩] } }
def leftRaw : List Term := Proof.Events198.exact50762RawTerms
def rightRaw : List Term := Proof.Events219.exact56064RawTerms
def group : MergeGroup := .operator 50762 56064
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 50762) (leftOrdinal := 0)
    (rightResult := 56064) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨19460⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19460⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge56070

namespace LeftMerge56149
def owner : Owner := ⟨.program ⟨214⟩, ⟨14000⟩⟩
def mergeEvent : Nat := 56149
def frameStart : Nat := 56119
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨11389⟩⟩, ⟨.program ⟨214⟩, ⟨13999⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events219.exact56145RawTerms
def rightRaw : List Term := Proof.Events219.exact56142RawTerms
def group : MergeGroup := .operator 56145 56142
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 56145) (leftOrdinal := 0)
    (rightResult := 56142) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨13999⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11389⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨11389⟩⟩, ⟨.program ⟨214⟩, ⟨13999⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge56149

namespace LeftMerge56179
def owner : Owner := ⟨.program ⟨214⟩, ⟨14103⟩⟩
def mergeEvent : Nat := 56179
def frameStart : Nat := 56119
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨11389⟩⟩, ⟨.program ⟨214⟩, ⟨13999⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events219.exact56175RawTerms
def rightRaw : List Term := Proof.Events219.exact56173RawTerms
def group : MergeGroup := .operator 56175 56173
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 56175) (leftOrdinal := 0)
    (rightResult := 56173) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11389⟩⟩, ⟨.program ⟨214⟩, ⟨13999⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨11389⟩⟩, ⟨.program ⟨214⟩, ⟨13999⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge56179

namespace LeftMerge56202
def owner : Owner := ⟨.program ⟨214⟩, ⟨7851⟩⟩
def mergeEvent : Nat := 56202
def frameStart : Nat := 56119
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6758⟩⟩, ⟨.program ⟨214⟩, ⟨7849⟩⟩] } }
def leftRaw : List Term := Proof.Events219.exact56198RawTerms
def rightRaw : List Term := Proof.Events219.exact56195RawTerms
def group : MergeGroup := .operator 56198 56195
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 56198) (leftOrdinal := 0)
    (rightResult := 56195) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6758⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨7849⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6758⟩⟩, ⟨.program ⟨214⟩, ⟨7849⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge56202

namespace LeftMerge56211
def owner : Owner := ⟨.program ⟨214⟩, ⟨25997⟩⟩
def mergeEvent : Nat := 56211
def frameStart : Nat := 56119
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6758⟩⟩, ⟨.program ⟨214⟩, ⟨7849⟩⟩, ⟨.program ⟨214⟩, ⟨25994⟩⟩] } }
def leftRaw : List Term := Proof.Events219.exact56207RawTerms
def rightRaw : List Term := Proof.Events219.exact56164RawTerms
def group : MergeGroup := .operator 56207 56164
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 56207) (leftOrdinal := 0)
    (rightResult := 56164) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6758⟩⟩, ⟨.program ⟨214⟩, ⟨7849⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25994⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6758⟩⟩, ⟨.program ⟨214⟩, ⟨7849⟩⟩, ⟨.program ⟨214⟩, ⟨25994⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge56211

namespace LeftMerge56212
def owner : Owner := ⟨.program ⟨214⟩, ⟨25997⟩⟩
def mergeEvent : Nat := 56212
def frameStart : Nat := 56119
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨11389⟩⟩, ⟨.program ⟨214⟩, ⟨13999⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25994⟩⟩] } }
def leftRaw : List Term := Proof.Events219.exact56207RawTerms
def rightRaw : List Term := Proof.Events219.exact56164RawTerms
def group : MergeGroup := .operator 56207 56164
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 56207) (leftOrdinal := 1)
    (rightResult := 56164) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11389⟩⟩, ⟨.program ⟨214⟩, ⟨13999⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25994⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨11389⟩⟩, ⟨.program ⟨214⟩, ⟨13999⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25994⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge56212

namespace LeftMerge56214
def owner : Owner := ⟨.program ⟨214⟩, ⟨25997⟩⟩
def mergeEvent : Nat := 56214
def frameStart : Nat := 56119
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨11389⟩⟩, ⟨.program ⟨214⟩, ⟨13999⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23544⟩⟩] } }
def rhsRaw : List Term := Proof.Events219.exact56161RawTerms
def group : MergeGroup := .relation 56213
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 56213) (rhsResult := 56161)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨11389⟩⟩, ⟨.program ⟨214⟩, ⟨13999⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25994⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨25994⟩⟩) ⟨23544⟩ 56161) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23544⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨11389⟩⟩, ⟨.program ⟨214⟩, ⟨13999⟩⟩], [⟨.program ⟨214⟩, ⟨23544⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge56214

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
