import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge10001
def owner : Owner := ⟨.program ⟨214⟩, ⟨11799⟩⟩
def mergeEvent : Nat := 10001
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨9630⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6783⟩⟩] } }
def leftRaw : List Term := Proof.Events039.exact9994RawTerms
def rightRaw : List Term := Proof.Events000.exact215RawTerms
def group : MergeGroup := .operator 9994 215
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 9994) (leftOrdinal := 0)
    (rightResult := 215) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6783⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨9630⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨9630⟩⟩], [⟨.program ⟨214⟩, ⟨6783⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge10001

namespace LeftMerge10016
def owner : Owner := ⟨.program ⟨214⟩, ⟨9631⟩⟩
def mergeEvent : Nat := 10016
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨9630⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events000.exact215RawTerms
def rightRaw : List Term := Proof.Events025.exact6449RawTerms
def group : MergeGroup := .operator 215 6449
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 215) (leftOrdinal := 0)
    (rightResult := 6449) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨9630⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨9630⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge10016

namespace LeftMerge10024
def owner : Owner := ⟨.program ⟨214⟩, ⟨7371⟩⟩
def mergeEvent : Nat := 10024
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6763⟩⟩] } }
def leftRaw : List Term := Proof.Events024.exact6314RawTerms
def rightRaw : List Term := Proof.Events039.exact10020RawTerms
def group : MergeGroup := .operator 6314 10020
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 6314) (leftOrdinal := 0)
    (rightResult := 10020) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6763⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6763⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge10024

namespace LeftMerge10041
def owner : Owner := ⟨.program ⟨214⟩, ⟨9634⟩⟩
def mergeEvent : Nat := 10041
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨9630⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7861⟩⟩] } }
def leftRaw : List Term := Proof.Events039.exact10035RawTerms
def rightRaw : List Term := Proof.Events039.exact10009RawTerms
def group : MergeGroup := .operator 10035 10009
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 10035) (leftOrdinal := 1)
    (rightResult := 10009) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨9630⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨7861⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨9630⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7861⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge10041

namespace LeftMerge10043
def owner : Owner := ⟨.program ⟨214⟩, ⟨9634⟩⟩
def mergeEvent : Nat := 10043
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨9630⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6783⟩⟩] } }
def rhsRaw : List Term := Proof.Events038.exact9979RawTerms
def group : MergeGroup := .relation 10042
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 10042) (rhsResult := 9979)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨9630⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7861⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨7861⟩⟩) ⟨6783⟩ 9979) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6783⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨9630⟩⟩], [⟨.program ⟨214⟩, ⟨6783⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge10043

namespace LeftMerge10044
def owner : Owner := ⟨.program ⟨214⟩, ⟨9634⟩⟩
def mergeEvent : Nat := 10044
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6763⟩⟩, ⟨.program ⟨214⟩, ⟨7861⟩⟩] } }
def leftRaw : List Term := Proof.Events039.exact10035RawTerms
def rightRaw : List Term := Proof.Events039.exact10009RawTerms
def group : MergeGroup := .operator 10035 10009
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 10035) (leftOrdinal := 0)
    (rightResult := 10009) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6763⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨7861⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6763⟩⟩, ⟨.program ⟨214⟩, ⟨7861⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge10044

namespace LeftMerge10049
def owner : Owner := ⟨.program ⟨214⟩, ⟨11800⟩⟩
def mergeEvent : Nat := 10049
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨9630⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6783⟩⟩] } }
def leftRaw : List Term := Proof.Events039.exact10045RawTerms
def rightRaw : List Term := Proof.Events039.exact10002RawTerms
def group : MergeGroup := .operator 10045 10002
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 10045) (leftOrdinal := 1)
    (rightResult := 10002) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨9630⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6783⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨9630⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6783⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨9630⟩⟩], [⟨.program ⟨214⟩, ⟨6783⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge10049

namespace LeftMerge10057
def owner : Owner := ⟨.program ⟨214⟩, ⟨25163⟩⟩
def mergeEvent : Nat := 10057
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨9630⟩⟩, ⟨.program ⟨214⟩, ⟨11793⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25162⟩⟩] } }
def leftRaw : List Term := Proof.Events039.exact10051RawTerms
def rightRaw : List Term := Proof.Events038.exact9968RawTerms
def group : MergeGroup := .operator 10051 9968
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 10051) (leftOrdinal := 1)
    (rightResult := 9968) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨9630⟩⟩, ⟨.program ⟨214⟩, ⟨11793⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25162⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨9630⟩⟩, ⟨.program ⟨214⟩, ⟨11793⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25162⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge10057

namespace LeftMerge10059
def owner : Owner := ⟨.program ⟨214⟩, ⟨25163⟩⟩
def mergeEvent : Nat := 10059
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨9630⟩⟩, ⟨.program ⟨214⟩, ⟨11793⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23088⟩⟩] } }
def rhsRaw : List Term := Proof.Events038.exact9965RawTerms
def group : MergeGroup := .relation 10058
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 10058) (rhsResult := 9965)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨9630⟩⟩, ⟨.program ⟨214⟩, ⟨11793⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25162⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨25162⟩⟩) ⟨23088⟩ 9965) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23088⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨9630⟩⟩, ⟨.program ⟨214⟩, ⟨11793⟩⟩], [⟨.program ⟨214⟩, ⟨23088⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge10059

namespace LeftMerge10060
def owner : Owner := ⟨.program ⟨214⟩, ⟨25163⟩⟩
def mergeEvent : Nat := 10060
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6763⟩⟩, ⟨.program ⟨214⟩, ⟨7861⟩⟩, ⟨.program ⟨214⟩, ⟨25162⟩⟩] } }
def leftRaw : List Term := Proof.Events039.exact10051RawTerms
def rightRaw : List Term := Proof.Events038.exact9968RawTerms
def group : MergeGroup := .operator 10051 9968
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 10051) (leftOrdinal := 0)
    (rightResult := 9968) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6763⟩⟩, ⟨.program ⟨214⟩, ⟨7861⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25162⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6763⟩⟩, ⟨.program ⟨214⟩, ⟨7861⟩⟩, ⟨.program ⟨214⟩, ⟨25162⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge10060

namespace LeftMerge10074
def owner : Owner := ⟨.program ⟨214⟩, ⟨19763⟩⟩
def mergeEvent : Nat := 10074
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19760⟩⟩] } }
def leftRaw : List Term := Proof.Events025.exact6561RawTerms
def rightRaw : List Term := Proof.Events039.exact10068RawTerms
def group : MergeGroup := .operator 6561 10068
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 6561) (leftOrdinal := 0)
    (rightResult := 10068) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨19760⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19760⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge10074

namespace LeftMerge10153
def owner : Owner := ⟨.program ⟨214⟩, ⟨11794⟩⟩
def mergeEvent : Nat := 10153
def frameStart : Nat := 10123
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨9630⟩⟩, ⟨.program ⟨214⟩, ⟨11793⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events039.exact10149RawTerms
def rightRaw : List Term := Proof.Events039.exact10146RawTerms
def group : MergeGroup := .operator 10149 10146
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 10149) (leftOrdinal := 0)
    (rightResult := 10146) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨9630⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11793⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨9630⟩⟩, ⟨.program ⟨214⟩, ⟨11793⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge10153

namespace LeftMerge10183
def owner : Owner := ⟨.program ⟨214⟩, ⟨11875⟩⟩
def mergeEvent : Nat := 10183
def frameStart : Nat := 10123
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨9630⟩⟩, ⟨.program ⟨214⟩, ⟨11793⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events039.exact10179RawTerms
def rightRaw : List Term := Proof.Events039.exact10177RawTerms
def group : MergeGroup := .operator 10179 10177
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 10179) (leftOrdinal := 0)
    (rightResult := 10177) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨9630⟩⟩, ⟨.program ⟨214⟩, ⟨11793⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨9630⟩⟩, ⟨.program ⟨214⟩, ⟨11793⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge10183

namespace LeftMerge10206
def owner : Owner := ⟨.program ⟨214⟩, ⟨7863⟩⟩
def mergeEvent : Nat := 10206
def frameStart : Nat := 10123
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6763⟩⟩, ⟨.program ⟨214⟩, ⟨7861⟩⟩] } }
def leftRaw : List Term := Proof.Events039.exact10202RawTerms
def rightRaw : List Term := Proof.Events039.exact10199RawTerms
def group : MergeGroup := .operator 10202 10199
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 10202) (leftOrdinal := 0)
    (rightResult := 10199) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6763⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨7861⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6763⟩⟩, ⟨.program ⟨214⟩, ⟨7861⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge10206

namespace LeftMerge10215
def owner : Owner := ⟨.program ⟨214⟩, ⟨25165⟩⟩
def mergeEvent : Nat := 10215
def frameStart : Nat := 10123
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨9630⟩⟩, ⟨.program ⟨214⟩, ⟨11793⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25162⟩⟩] } }
def leftRaw : List Term := Proof.Events039.exact10211RawTerms
def rightRaw : List Term := Proof.Events039.exact10168RawTerms
def group : MergeGroup := .operator 10211 10168
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 10211) (leftOrdinal := 1)
    (rightResult := 10168) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨9630⟩⟩, ⟨.program ⟨214⟩, ⟨11793⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25162⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨9630⟩⟩, ⟨.program ⟨214⟩, ⟨11793⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25162⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge10215

namespace LeftMerge10217
def owner : Owner := ⟨.program ⟨214⟩, ⟨25165⟩⟩
def mergeEvent : Nat := 10217
def frameStart : Nat := 10123
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨9630⟩⟩, ⟨.program ⟨214⟩, ⟨11793⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23088⟩⟩] } }
def rhsRaw : List Term := Proof.Events039.exact10165RawTerms
def group : MergeGroup := .relation 10216
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 10216) (rhsResult := 10165)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨9630⟩⟩, ⟨.program ⟨214⟩, ⟨11793⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25162⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨25162⟩⟩) ⟨23088⟩ 10165) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23088⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨9630⟩⟩, ⟨.program ⟨214⟩, ⟨11793⟩⟩], [⟨.program ⟨214⟩, ⟨23088⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge10217

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
