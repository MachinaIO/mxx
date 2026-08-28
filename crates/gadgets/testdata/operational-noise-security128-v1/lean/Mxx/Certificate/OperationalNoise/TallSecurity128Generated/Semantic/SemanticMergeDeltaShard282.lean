import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge49130
def owner : Owner := ⟨.program ⟨257⟩, ⟨13705⟩⟩
def mergeEvent : Nat := 49130
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨13701⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7280⟩⟩] } }
def rhsRaw : List Term := Proof.Events076.exact19585RawTerms
def group : MergeGroup := .relation 49129
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 49129) (rhsResult := 19585)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨13701⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨9550⟩⟩) ⟨7280⟩ 19585) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7280⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨13701⟩⟩], [⟨.program ⟨257⟩, ⟨7280⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge49130

namespace LeftMerge49131
def owner : Owner := ⟨.program ⟨257⟩, ⟨13705⟩⟩
def mergeEvent : Nat := 49131
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩] } }
def leftRaw : List Term := Proof.Events191.exact49122RawTerms
def rightRaw : List Term := Proof.Events076.exact19615RawTerms
def group : MergeGroup := .operator 49122 19615
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 49122) (leftOrdinal := 0)
    (rightResult := 19615) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7297⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9550⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge49131

namespace LeftMerge49136
def owner : Owner := ⟨.program ⟨257⟩, ⟨34633⟩⟩
def mergeEvent : Nat := 49136
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨13701⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7280⟩⟩] } }
def leftRaw : List Term := Proof.Events191.exact49132RawTerms
def rightRaw : List Term := Proof.Events191.exact49102RawTerms
def group : MergeGroup := .operator 49132 49102
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 49132) (leftOrdinal := 1)
    (rightResult := 49102) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨13701⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7280⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨13701⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7280⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨13701⟩⟩], [⟨.program ⟨257⟩, ⟨7280⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge49136

namespace LeftMerge49144
def owner : Owner := ⟨.program ⟨257⟩, ⟨36348⟩⟩
def mergeEvent : Nat := 49144
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨13701⟩⟩, ⟨.program ⟨257⟩, ⟨34626⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36347⟩⟩] } }
def leftRaw : List Term := Proof.Events191.exact49138RawTerms
def rightRaw : List Term := Proof.Events191.exact49074RawTerms
def group : MergeGroup := .operator 49138 49074
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 49138) (leftOrdinal := 1)
    (rightResult := 49074) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨13701⟩⟩, ⟨.program ⟨257⟩, ⟨34626⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨36347⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨13701⟩⟩, ⟨.program ⟨257⟩, ⟨34626⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36347⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge49144

namespace LeftMerge49146
def owner : Owner := ⟨.program ⟨257⟩, ⟨36348⟩⟩
def mergeEvent : Nat := 49146
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨13701⟩⟩, ⟨.program ⟨257⟩, ⟨34626⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35797⟩⟩] } }
def rhsRaw : List Term := Proof.Events191.exact49071RawTerms
def group : MergeGroup := .relation 49145
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 49145) (rhsResult := 49071)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨13701⟩⟩, ⟨.program ⟨257⟩, ⟨34626⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36347⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨36347⟩⟩) ⟨35797⟩ 49071) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨35797⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨13701⟩⟩, ⟨.program ⟨257⟩, ⟨34626⟩⟩], [⟨.program ⟨257⟩, ⟨35797⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge49146

namespace LeftMerge49147
def owner : Owner := ⟨.program ⟨257⟩, ⟨36348⟩⟩
def mergeEvent : Nat := 49147
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩, ⟨.program ⟨257⟩, ⟨36347⟩⟩] } }
def leftRaw : List Term := Proof.Events191.exact49138RawTerms
def rightRaw : List Term := Proof.Events191.exact49074RawTerms
def group : MergeGroup := .operator 49138 49074
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 49138) (leftOrdinal := 0)
    (rightResult := 49074) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨36347⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩, ⟨.program ⟨257⟩, ⟨36347⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge49147

namespace LeftMerge49161
def owner : Owner := ⟨.program ⟨257⟩, ⟨35272⟩⟩
def mergeEvent : Nat := 49161
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35269⟩⟩] } }
def leftRaw : List Term := Proof.Events182.exact46745RawTerms
def rightRaw : List Term := Proof.Events192.exact49155RawTerms
def group : MergeGroup := .operator 46745 49155
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 46745) (leftOrdinal := 0)
    (rightResult := 49155) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨35269⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35269⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge49161

namespace LeftMerge49240
def owner : Owner := ⟨.program ⟨257⟩, ⟨34627⟩⟩
def mergeEvent : Nat := 49240
def frameStart : Nat := 49210
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨13701⟩⟩, ⟨.program ⟨257⟩, ⟨34626⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events192.exact49236RawTerms
def rightRaw : List Term := Proof.Events192.exact49233RawTerms
def group : MergeGroup := .operator 49236 49233
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 49236) (leftOrdinal := 0)
    (rightResult := 49233) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13701⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨34626⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨13701⟩⟩, ⟨.program ⟨257⟩, ⟨34626⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge49240

namespace LeftMerge49270
def owner : Owner := ⟨.program ⟨257⟩, ⟨36060⟩⟩
def mergeEvent : Nat := 49270
def frameStart : Nat := 49210
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨13701⟩⟩, ⟨.program ⟨257⟩, ⟨34626⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events192.exact49266RawTerms
def rightRaw : List Term := Proof.Events192.exact49264RawTerms
def group : MergeGroup := .operator 49266 49264
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 49266) (leftOrdinal := 0)
    (rightResult := 49264) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13701⟩⟩, ⟨.program ⟨257⟩, ⟨34626⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨13701⟩⟩, ⟨.program ⟨257⟩, ⟨34626⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge49270

namespace LeftMerge49293
def owner : Owner := ⟨.program ⟨257⟩, ⟨9552⟩⟩
def mergeEvent : Nat := 49293
def frameStart : Nat := 49210
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩] } }
def leftRaw : List Term := Proof.Events192.exact49289RawTerms
def rightRaw : List Term := Proof.Events192.exact49286RawTerms
def group : MergeGroup := .operator 49289 49286
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 49289) (leftOrdinal := 0)
    (rightResult := 49286) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7297⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9550⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge49293

namespace LeftMerge49302
def owner : Owner := ⟨.program ⟨257⟩, ⟨36350⟩⟩
def mergeEvent : Nat := 49302
def frameStart : Nat := 49210
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩, ⟨.program ⟨257⟩, ⟨36347⟩⟩] } }
def leftRaw : List Term := Proof.Events192.exact49298RawTerms
def rightRaw : List Term := Proof.Events192.exact49255RawTerms
def group : MergeGroup := .operator 49298 49255
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 49298) (leftOrdinal := 0)
    (rightResult := 49255) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨36347⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩, ⟨.program ⟨257⟩, ⟨36347⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge49302

namespace LeftMerge49303
def owner : Owner := ⟨.program ⟨257⟩, ⟨36350⟩⟩
def mergeEvent : Nat := 49303
def frameStart : Nat := 49210
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨13701⟩⟩, ⟨.program ⟨257⟩, ⟨34626⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36347⟩⟩] } }
def leftRaw : List Term := Proof.Events192.exact49298RawTerms
def rightRaw : List Term := Proof.Events192.exact49255RawTerms
def group : MergeGroup := .operator 49298 49255
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 49298) (leftOrdinal := 1)
    (rightResult := 49255) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13701⟩⟩, ⟨.program ⟨257⟩, ⟨34626⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨36347⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨13701⟩⟩, ⟨.program ⟨257⟩, ⟨34626⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36347⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge49303

namespace LeftMerge49305
def owner : Owner := ⟨.program ⟨257⟩, ⟨36350⟩⟩
def mergeEvent : Nat := 49305
def frameStart : Nat := 49210
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨13701⟩⟩, ⟨.program ⟨257⟩, ⟨34626⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35797⟩⟩] } }
def rhsRaw : List Term := Proof.Events192.exact49252RawTerms
def group : MergeGroup := .relation 49304
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 49304) (rhsResult := 49252)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨13701⟩⟩, ⟨.program ⟨257⟩, ⟨34626⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36347⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨36347⟩⟩) ⟨35797⟩ 49252) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨35797⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨13701⟩⟩, ⟨.program ⟨257⟩, ⟨34626⟩⟩], [⟨.program ⟨257⟩, ⟨35797⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge49305

namespace LeftMerge49313
def owner : Owner := ⟨.program ⟨257⟩, ⟨34814⟩⟩
def mergeEvent : Nat := 49313
def frameStart : Nat := 49210
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨34812⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events192.exact49266RawTerms
def rightRaw : List Term := Proof.Events192.exact49309RawTerms
def group : MergeGroup := .operator 49266 49309
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 49266) (leftOrdinal := 0)
    (rightResult := 49309) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨34812⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨34812⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge49313

namespace LeftMerge49330
def owner : Owner := ⟨.program ⟨257⟩, ⟨35272⟩⟩
def mergeEvent : Nat := 49330
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7191⟩⟩] } }
def rhsRaw : List Term := Proof.Events192.exact49327RawTerms
def group : MergeGroup := .relation 49329
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 49329) (rhsResult := 49327)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35269⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 49328 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35269⟩⟩]⟩) (none) 49327) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7191⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7191⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge49330

namespace LeftMerge49331
def owner : Owner := ⟨.program ⟨257⟩, ⟨35272⟩⟩
def mergeEvent : Nat := 49331
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩, ⟨.program ⟨257⟩, ⟨36347⟩⟩] } }
def rhsRaw : List Term := Proof.Events192.exact49327RawTerms
def group : MergeGroup := .relation 49329
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 49329) (rhsResult := 49327)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35269⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 49328 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35269⟩⟩]⟩) (none) 49327) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩, ⟨.program ⟨257⟩, ⟨36347⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩, ⟨.program ⟨257⟩, ⟨36347⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge49331

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
