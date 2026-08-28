import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge97042
def owner : Owner := ⟨.program ⟨214⟩, ⟨9704⟩⟩
def mergeEvent : Nat := 97042
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6764⟩⟩, ⟨.program ⟨214⟩, ⟨7864⟩⟩] } }
def leftRaw : List Term := Proof.Events379.exact97033RawTerms
def rightRaw : List Term := Proof.Events037.exact9508RawTerms
def group : MergeGroup := .operator 97033 9508
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 97033) (leftOrdinal := 0)
    (rightResult := 9508) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6764⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨7864⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6764⟩⟩, ⟨.program ⟨214⟩, ⟨7864⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge97042

namespace LeftMerge97047
def owner : Owner := ⟨.program ⟨214⟩, ⟨11940⟩⟩
def mergeEvent : Nat := 97047
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨9700⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6784⟩⟩] } }
def leftRaw : List Term := Proof.Events379.exact97043RawTerms
def rightRaw : List Term := Proof.Events378.exact97013RawTerms
def group : MergeGroup := .operator 97043 97013
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 97043) (leftOrdinal := 1)
    (rightResult := 97013) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨9700⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6784⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨9700⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6784⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨9700⟩⟩], [⟨.program ⟨214⟩, ⟨6784⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge97047

namespace LeftMerge97055
def owner : Owner := ⟨.program ⟨214⟩, ⟨25207⟩⟩
def mergeEvent : Nat := 97055
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨9700⟩⟩, ⟨.program ⟨214⟩, ⟨11933⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25206⟩⟩] } }
def leftRaw : List Term := Proof.Events379.exact97049RawTerms
def rightRaw : List Term := Proof.Events378.exact96985RawTerms
def group : MergeGroup := .operator 97049 96985
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 97049) (leftOrdinal := 1)
    (rightResult := 96985) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨9700⟩⟩, ⟨.program ⟨214⟩, ⟨11933⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25206⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨9700⟩⟩, ⟨.program ⟨214⟩, ⟨11933⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25206⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge97055

namespace LeftMerge97057
def owner : Owner := ⟨.program ⟨214⟩, ⟨25207⟩⟩
def mergeEvent : Nat := 97057
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨9700⟩⟩, ⟨.program ⟨214⟩, ⟨11933⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23116⟩⟩] } }
def rhsRaw : List Term := Proof.Events378.exact96982RawTerms
def group : MergeGroup := .relation 97056
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 97056) (rhsResult := 96982)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨9700⟩⟩, ⟨.program ⟨214⟩, ⟨11933⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25206⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨25206⟩⟩) ⟨23116⟩ 96982) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23116⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨9700⟩⟩, ⟨.program ⟨214⟩, ⟨11933⟩⟩], [⟨.program ⟨214⟩, ⟨23116⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge97057

namespace LeftMerge97058
def owner : Owner := ⟨.program ⟨214⟩, ⟨25207⟩⟩
def mergeEvent : Nat := 97058
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6764⟩⟩, ⟨.program ⟨214⟩, ⟨7864⟩⟩, ⟨.program ⟨214⟩, ⟨25206⟩⟩] } }
def leftRaw : List Term := Proof.Events379.exact97049RawTerms
def rightRaw : List Term := Proof.Events378.exact96985RawTerms
def group : MergeGroup := .operator 97049 96985
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 97049) (leftOrdinal := 0)
    (rightResult := 96985) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6764⟩⟩, ⟨.program ⟨214⟩, ⟨7864⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25206⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6764⟩⟩, ⟨.program ⟨214⟩, ⟨7864⟩⟩, ⟨.program ⟨214⟩, ⟨25206⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge97058

namespace LeftMerge97072
def owner : Owner := ⟨.program ⟨214⟩, ⟨19808⟩⟩
def mergeEvent : Nat := 97072
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19805⟩⟩] } }
def leftRaw : List Term := Proof.Events368.exact94462RawTerms
def rightRaw : List Term := Proof.Events379.exact97066RawTerms
def group : MergeGroup := .operator 94462 97066
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 94462) (leftOrdinal := 0)
    (rightResult := 97066) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨19805⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19805⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge97072

namespace LeftMerge97127
def owner : Owner := ⟨.program ⟨214⟩, ⟨11934⟩⟩
def mergeEvent : Nat := 97127
def frameStart : Nat := 97109
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨9700⟩⟩, ⟨.program ⟨214⟩, ⟨11933⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events379.exact97123RawTerms
def rightRaw : List Term := Proof.Events379.exact97120RawTerms
def group : MergeGroup := .operator 97123 97120
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 97123) (leftOrdinal := 0)
    (rightResult := 97120) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨9700⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11933⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨9700⟩⟩, ⟨.program ⟨214⟩, ⟨11933⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge97127

namespace LeftMerge97157
def owner : Owner := ⟨.program ⟨214⟩, ⟨12047⟩⟩
def mergeEvent : Nat := 97157
def frameStart : Nat := 97109
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨9700⟩⟩, ⟨.program ⟨214⟩, ⟨11933⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events379.exact97153RawTerms
def rightRaw : List Term := Proof.Events379.exact97151RawTerms
def group : MergeGroup := .operator 97153 97151
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 97153) (leftOrdinal := 0)
    (rightResult := 97151) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨9700⟩⟩, ⟨.program ⟨214⟩, ⟨11933⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨9700⟩⟩, ⟨.program ⟨214⟩, ⟨11933⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge97157

namespace LeftMerge97180
def owner : Owner := ⟨.program ⟨214⟩, ⟨7866⟩⟩
def mergeEvent : Nat := 97180
def frameStart : Nat := 97109
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6764⟩⟩, ⟨.program ⟨214⟩, ⟨7864⟩⟩] } }
def leftRaw : List Term := Proof.Events379.exact97176RawTerms
def rightRaw : List Term := Proof.Events379.exact97173RawTerms
def group : MergeGroup := .operator 97176 97173
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 97176) (leftOrdinal := 0)
    (rightResult := 97173) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6764⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨7864⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6764⟩⟩, ⟨.program ⟨214⟩, ⟨7864⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge97180

namespace LeftMerge97189
def owner : Owner := ⟨.program ⟨214⟩, ⟨25209⟩⟩
def mergeEvent : Nat := 97189
def frameStart : Nat := 97109
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6764⟩⟩, ⟨.program ⟨214⟩, ⟨7864⟩⟩, ⟨.program ⟨214⟩, ⟨25206⟩⟩] } }
def leftRaw : List Term := Proof.Events379.exact97185RawTerms
def rightRaw : List Term := Proof.Events379.exact97142RawTerms
def group : MergeGroup := .operator 97185 97142
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 97185) (leftOrdinal := 0)
    (rightResult := 97142) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6764⟩⟩, ⟨.program ⟨214⟩, ⟨7864⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25206⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6764⟩⟩, ⟨.program ⟨214⟩, ⟨7864⟩⟩, ⟨.program ⟨214⟩, ⟨25206⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge97189

namespace LeftMerge97190
def owner : Owner := ⟨.program ⟨214⟩, ⟨25209⟩⟩
def mergeEvent : Nat := 97190
def frameStart : Nat := 97109
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨9700⟩⟩, ⟨.program ⟨214⟩, ⟨11933⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25206⟩⟩] } }
def leftRaw : List Term := Proof.Events379.exact97185RawTerms
def rightRaw : List Term := Proof.Events379.exact97142RawTerms
def group : MergeGroup := .operator 97185 97142
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 97185) (leftOrdinal := 1)
    (rightResult := 97142) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨9700⟩⟩, ⟨.program ⟨214⟩, ⟨11933⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25206⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨9700⟩⟩, ⟨.program ⟨214⟩, ⟨11933⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25206⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge97190

namespace LeftMerge97192
def owner : Owner := ⟨.program ⟨214⟩, ⟨25209⟩⟩
def mergeEvent : Nat := 97192
def frameStart : Nat := 97109
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨9700⟩⟩, ⟨.program ⟨214⟩, ⟨11933⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23116⟩⟩] } }
def rhsRaw : List Term := Proof.Events379.exact97139RawTerms
def group : MergeGroup := .relation 97191
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 97191) (rhsResult := 97139)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨9700⟩⟩, ⟨.program ⟨214⟩, ⟨11933⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25206⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨25206⟩⟩) ⟨23116⟩ 97139) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23116⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨9700⟩⟩, ⟨.program ⟨214⟩, ⟨11933⟩⟩], [⟨.program ⟨214⟩, ⟨23116⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge97192

namespace LeftMerge97200
def owner : Owner := ⟨.program ⟨214⟩, ⟨16373⟩⟩
def mergeEvent : Nat := 97200
def frameStart : Nat := 97109
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16371⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events379.exact97153RawTerms
def rightRaw : List Term := Proof.Events379.exact97196RawTerms
def group : MergeGroup := .operator 97153 97196
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 97153) (leftOrdinal := 0)
    (rightResult := 97196) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16371⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨16371⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge97200

namespace LeftMerge97217
def owner : Owner := ⟨.program ⟨214⟩, ⟨19808⟩⟩
def mergeEvent : Nat := 97217
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6701⟩⟩] } }
def rhsRaw : List Term := Proof.Events379.exact97214RawTerms
def group : MergeGroup := .relation 97216
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 97216) (rhsResult := 97214)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19805⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 97215 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19805⟩⟩]⟩) (none) 97214) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6701⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6701⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge97217

namespace LeftMerge97218
def owner : Owner := ⟨.program ⟨214⟩, ⟨19808⟩⟩
def mergeEvent : Nat := 97218
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6764⟩⟩, ⟨.program ⟨214⟩, ⟨7864⟩⟩, ⟨.program ⟨214⟩, ⟨25206⟩⟩] } }
def rhsRaw : List Term := Proof.Events379.exact97214RawTerms
def group : MergeGroup := .relation 97216
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 97216) (rhsResult := 97214)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19805⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 97215 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19805⟩⟩]⟩) (none) 97214) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6764⟩⟩, ⟨.program ⟨214⟩, ⟨7864⟩⟩, ⟨.program ⟨214⟩, ⟨25206⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6764⟩⟩, ⟨.program ⟨214⟩, ⟨7864⟩⟩, ⟨.program ⟨214⟩, ⟨25206⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge97218

namespace LeftMerge97219
def owner : Owner := ⟨.program ⟨214⟩, ⟨19808⟩⟩
def mergeEvent : Nat := 97219
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨9700⟩⟩, ⟨.program ⟨214⟩, ⟨11933⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23116⟩⟩] } }
def rhsRaw : List Term := Proof.Events379.exact97214RawTerms
def group : MergeGroup := .relation 97216
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 97216) (rhsResult := 97214)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19805⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 97215 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19805⟩⟩]⟩) (none) 97214) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨9700⟩⟩, ⟨.program ⟨214⟩, ⟨11933⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23116⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨9700⟩⟩, ⟨.program ⟨214⟩, ⟨11933⟩⟩], [⟨.program ⟨214⟩, ⟨23116⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge97219

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
