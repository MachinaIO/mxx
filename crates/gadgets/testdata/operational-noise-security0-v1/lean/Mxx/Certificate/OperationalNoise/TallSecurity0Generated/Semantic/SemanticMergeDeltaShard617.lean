import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge100049
def owner : Owner := ⟨.program ⟨214⟩, ⟨13532⟩⟩
def mergeEvent : Nat := 100049
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨11205⟩⟩, ⟨.program ⟨214⟩, ⟨13529⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events390.exact100043RawTerms
def rightRaw : List Term := Proof.Events019.exact4868RawTerms
def group : MergeGroup := .operator 100043 4868
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 100043) (leftOrdinal := 1)
    (rightResult := 4868) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨11205⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨13529⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨11205⟩⟩, ⟨.program ⟨214⟩, ⟨13529⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge100049

namespace LeftMerge100050
def owner : Owner := ⟨.program ⟨214⟩, ⟨13532⟩⟩
def mergeEvent : Nat := 100050
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨13529⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6776⟩⟩] } }
def leftRaw : List Term := Proof.Events390.exact100043RawTerms
def rightRaw : List Term := Proof.Events019.exact4868RawTerms
def group : MergeGroup := .operator 100043 4868
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 100043) (leftOrdinal := 0)
    (rightResult := 4868) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6776⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨13529⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨13529⟩⟩], [⟨.program ⟨214⟩, ⟨6776⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge100050

namespace LeftMerge100055
def owner : Owner := ⟨.program ⟨214⟩, ⟨13533⟩⟩
def mergeEvent : Nat := 100055
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨13529⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events019.exact4868RawTerms
def rightRaw : List Term := Proof.Events000.exact32RawTerms
def group : MergeGroup := .operator 4868 32
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 4868) (leftOrdinal := 0)
    (rightResult := 32) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨13529⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨13529⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge100055

namespace LeftMerge100060
def owner : Owner := ⟨.program ⟨214⟩, ⟨7130⟩⟩
def mergeEvent : Nat := 100060
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6793⟩⟩] } }
def leftRaw : List Term := Proof.Events000.exact27RawTerms
def rightRaw : List Term := Proof.Events050.exact13026RawTerms
def group : MergeGroup := .operator 27 13026
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 27) (leftOrdinal := 0)
    (rightResult := 13026) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6793⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6793⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge100060

namespace LeftMerge100077
def owner : Owner := ⟨.program ⟨214⟩, ⟨13536⟩⟩
def mergeEvent : Nat := 100077
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨13529⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7843⟩⟩] } }
def leftRaw : List Term := Proof.Events390.exact100071RawTerms
def rightRaw : List Term := Proof.Events050.exact13015RawTerms
def group : MergeGroup := .operator 100071 13015
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 100071) (leftOrdinal := 1)
    (rightResult := 13015) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨13529⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨7843⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨13529⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7843⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge100077

namespace LeftMerge100079
def owner : Owner := ⟨.program ⟨214⟩, ⟨13536⟩⟩
def mergeEvent : Nat := 100079
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨13529⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6776⟩⟩] } }
def rhsRaw : List Term := Proof.Events050.exact12985RawTerms
def group : MergeGroup := .relation 100078
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 100078) (rhsResult := 12985)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨13529⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7843⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨7843⟩⟩) ⟨6776⟩ 12985) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6776⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨13529⟩⟩], [⟨.program ⟨214⟩, ⟨6776⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge100079

namespace LeftMerge100080
def owner : Owner := ⟨.program ⟨214⟩, ⟨13536⟩⟩
def mergeEvent : Nat := 100080
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6793⟩⟩, ⟨.program ⟨214⟩, ⟨7843⟩⟩] } }
def leftRaw : List Term := Proof.Events390.exact100071RawTerms
def rightRaw : List Term := Proof.Events050.exact13015RawTerms
def group : MergeGroup := .operator 100071 13015
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 100071) (leftOrdinal := 0)
    (rightResult := 13015) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6793⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨7843⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6793⟩⟩, ⟨.program ⟨214⟩, ⟨7843⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge100080

namespace LeftMerge100085
def owner : Owner := ⟨.program ⟨214⟩, ⟨13537⟩⟩
def mergeEvent : Nat := 100085
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨13529⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6776⟩⟩] } }
def leftRaw : List Term := Proof.Events390.exact100081RawTerms
def rightRaw : List Term := Proof.Events390.exact100051RawTerms
def group : MergeGroup := .operator 100081 100051
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 100081) (leftOrdinal := 1)
    (rightResult := 100051) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨13529⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6776⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨13529⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6776⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨13529⟩⟩], [⟨.program ⟨214⟩, ⟨6776⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge100085

namespace LeftMerge100093
def owner : Owner := ⟨.program ⟨214⟩, ⟨25823⟩⟩
def mergeEvent : Nat := 100093
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨11205⟩⟩, ⟨.program ⟨214⟩, ⟨13529⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25822⟩⟩] } }
def leftRaw : List Term := Proof.Events390.exact100087RawTerms
def rightRaw : List Term := Proof.Events390.exact100023RawTerms
def group : MergeGroup := .operator 100087 100023
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 100087) (leftOrdinal := 1)
    (rightResult := 100023) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨11205⟩⟩, ⟨.program ⟨214⟩, ⟨13529⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25822⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨11205⟩⟩, ⟨.program ⟨214⟩, ⟨13529⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25822⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge100093

namespace LeftMerge100095
def owner : Owner := ⟨.program ⟨214⟩, ⟨25823⟩⟩
def mergeEvent : Nat := 100095
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨11205⟩⟩, ⟨.program ⟨214⟩, ⟨13529⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23452⟩⟩] } }
def rhsRaw : List Term := Proof.Events390.exact100020RawTerms
def group : MergeGroup := .relation 100094
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 100094) (rhsResult := 100020)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨11205⟩⟩, ⟨.program ⟨214⟩, ⟨13529⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25822⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨25822⟩⟩) ⟨23452⟩ 100020) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23452⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨11205⟩⟩, ⟨.program ⟨214⟩, ⟨13529⟩⟩], [⟨.program ⟨214⟩, ⟨23452⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge100095

namespace LeftMerge100096
def owner : Owner := ⟨.program ⟨214⟩, ⟨25823⟩⟩
def mergeEvent : Nat := 100096
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6793⟩⟩, ⟨.program ⟨214⟩, ⟨7843⟩⟩, ⟨.program ⟨214⟩, ⟨25822⟩⟩] } }
def leftRaw : List Term := Proof.Events390.exact100087RawTerms
def rightRaw : List Term := Proof.Events390.exact100023RawTerms
def group : MergeGroup := .operator 100087 100023
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 100087) (leftOrdinal := 0)
    (rightResult := 100023) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6793⟩⟩, ⟨.program ⟨214⟩, ⟨7843⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25822⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6793⟩⟩, ⟨.program ⟨214⟩, ⟨7843⟩⟩, ⟨.program ⟨214⟩, ⟨25822⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge100096

namespace LeftMerge100110
def owner : Owner := ⟨.program ⟨214⟩, ⟨19304⟩⟩
def mergeEvent : Nat := 100110
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19301⟩⟩] } }
def leftRaw : List Term := Proof.Events368.exact94462RawTerms
def rightRaw : List Term := Proof.Events391.exact100104RawTerms
def group : MergeGroup := .operator 94462 100104
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 94462) (leftOrdinal := 0)
    (rightResult := 100104) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨19301⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19301⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge100110

namespace LeftMerge100165
def owner : Owner := ⟨.program ⟨214⟩, ⟨13530⟩⟩
def mergeEvent : Nat := 100165
def frameStart : Nat := 100147
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨11205⟩⟩, ⟨.program ⟨214⟩, ⟨13529⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events391.exact100161RawTerms
def rightRaw : List Term := Proof.Events391.exact100158RawTerms
def group : MergeGroup := .operator 100161 100158
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 100161) (leftOrdinal := 0)
    (rightResult := 100158) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨13529⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11205⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨11205⟩⟩, ⟨.program ⟨214⟩, ⟨13529⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge100165

namespace LeftMerge100195
def owner : Owner := ⟨.program ⟨214⟩, ⟨13657⟩⟩
def mergeEvent : Nat := 100195
def frameStart : Nat := 100147
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨11205⟩⟩, ⟨.program ⟨214⟩, ⟨13529⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events391.exact100191RawTerms
def rightRaw : List Term := Proof.Events391.exact100189RawTerms
def group : MergeGroup := .operator 100191 100189
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 100191) (leftOrdinal := 0)
    (rightResult := 100189) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11205⟩⟩, ⟨.program ⟨214⟩, ⟨13529⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨11205⟩⟩, ⟨.program ⟨214⟩, ⟨13529⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge100195

namespace LeftMerge100218
def owner : Owner := ⟨.program ⟨214⟩, ⟨7845⟩⟩
def mergeEvent : Nat := 100218
def frameStart : Nat := 100147
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6793⟩⟩, ⟨.program ⟨214⟩, ⟨7843⟩⟩] } }
def leftRaw : List Term := Proof.Events391.exact100214RawTerms
def rightRaw : List Term := Proof.Events391.exact100211RawTerms
def group : MergeGroup := .operator 100214 100211
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 100214) (leftOrdinal := 0)
    (rightResult := 100211) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6793⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨7843⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6793⟩⟩, ⟨.program ⟨214⟩, ⟨7843⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge100218

namespace LeftMerge100227
def owner : Owner := ⟨.program ⟨214⟩, ⟨25825⟩⟩
def mergeEvent : Nat := 100227
def frameStart : Nat := 100147
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6793⟩⟩, ⟨.program ⟨214⟩, ⟨7843⟩⟩, ⟨.program ⟨214⟩, ⟨25822⟩⟩] } }
def leftRaw : List Term := Proof.Events391.exact100223RawTerms
def rightRaw : List Term := Proof.Events391.exact100180RawTerms
def group : MergeGroup := .operator 100223 100180
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 100223) (leftOrdinal := 0)
    (rightResult := 100180) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6793⟩⟩, ⟨.program ⟨214⟩, ⟨7843⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25822⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6793⟩⟩, ⟨.program ⟨214⟩, ⟨7843⟩⟩, ⟨.program ⟨214⟩, ⟨25822⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge100227

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
