import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge52986
def owner : Owner := ⟨.program ⟨257⟩, ⟨50768⟩⟩
def mergeEvent : Nat := 52986
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨50761⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7308⟩⟩] } }
def rhsRaw : List Term := Proof.Events092.exact23593RawTerms
def group : MergeGroup := .relation 52985
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 52985) (rhsResult := 23593)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨50761⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨9580⟩⟩) ⟨7308⟩ 23593) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7308⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨50761⟩⟩], [⟨.program ⟨257⟩, ⟨7308⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge52986

namespace LeftMerge52987
def owner : Owner := ⟨.program ⟨257⟩, ⟨50768⟩⟩
def mergeEvent : Nat := 52987
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩] } }
def leftRaw : List Term := Proof.Events206.exact52978RawTerms
def rightRaw : List Term := Proof.Events092.exact23623RawTerms
def group : MergeGroup := .operator 52978 23623
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 52978) (leftOrdinal := 0)
    (rightResult := 23623) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7288⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9580⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge52987

namespace LeftMerge52992
def owner : Owner := ⟨.program ⟨257⟩, ⟨50769⟩⟩
def mergeEvent : Nat := 52992
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨50761⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7308⟩⟩] } }
def leftRaw : List Term := Proof.Events206.exact52988RawTerms
def rightRaw : List Term := Proof.Events206.exact52958RawTerms
def group : MergeGroup := .operator 52988 52958
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 52988) (leftOrdinal := 1)
    (rightResult := 52958) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨50761⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7308⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨50761⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7308⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨50761⟩⟩], [⟨.program ⟨257⟩, ⟨7308⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge52992

namespace LeftMerge53000
def owner : Owner := ⟨.program ⟨257⟩, ⟨52608⟩⟩
def mergeEvent : Nat := 53000
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨24626⟩⟩, ⟨.program ⟨257⟩, ⟨50761⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨52607⟩⟩] } }
def leftRaw : List Term := Proof.Events207.exact52994RawTerms
def rightRaw : List Term := Proof.Events206.exact52930RawTerms
def group : MergeGroup := .operator 52994 52930
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 52994) (leftOrdinal := 1)
    (rightResult := 52930) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨24626⟩⟩, ⟨.program ⟨257⟩, ⟨50761⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨52607⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨24626⟩⟩, ⟨.program ⟨257⟩, ⟨50761⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨52607⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge53000

namespace LeftMerge53002
def owner : Owner := ⟨.program ⟨257⟩, ⟨52608⟩⟩
def mergeEvent : Nat := 53002
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨24626⟩⟩, ⟨.program ⟨257⟩, ⟨50761⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨52057⟩⟩] } }
def rhsRaw : List Term := Proof.Events206.exact52927RawTerms
def group : MergeGroup := .relation 53001
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 53001) (rhsResult := 52927)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨24626⟩⟩, ⟨.program ⟨257⟩, ⟨50761⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨52607⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨52607⟩⟩) ⟨52057⟩ 52927) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨52057⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨24626⟩⟩, ⟨.program ⟨257⟩, ⟨50761⟩⟩], [⟨.program ⟨257⟩, ⟨52057⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge53002

namespace LeftMerge53003
def owner : Owner := ⟨.program ⟨257⟩, ⟨52608⟩⟩
def mergeEvent : Nat := 53003
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52607⟩⟩] } }
def leftRaw : List Term := Proof.Events207.exact52994RawTerms
def rightRaw : List Term := Proof.Events206.exact52930RawTerms
def group : MergeGroup := .operator 52994 52930
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 52994) (leftOrdinal := 0)
    (rightResult := 52930) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨52607⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52607⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge53003

namespace LeftMerge53017
def owner : Owner := ⟨.program ⟨257⟩, ⟨51532⟩⟩
def mergeEvent : Nat := 53017
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51529⟩⟩] } }
def leftRaw : List Term := Proof.Events182.exact46745RawTerms
def rightRaw : List Term := Proof.Events207.exact53011RawTerms
def group : MergeGroup := .operator 46745 53011
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 46745) (leftOrdinal := 0)
    (rightResult := 53011) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨51529⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51529⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge53017

namespace LeftMerge53096
def owner : Owner := ⟨.program ⟨257⟩, ⟨50762⟩⟩
def mergeEvent : Nat := 53096
def frameStart : Nat := 53066
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24626⟩⟩, ⟨.program ⟨257⟩, ⟨50761⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events207.exact53092RawTerms
def rightRaw : List Term := Proof.Events207.exact53089RawTerms
def group : MergeGroup := .operator 53092 53089
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 53092) (leftOrdinal := 0)
    (rightResult := 53089) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨50761⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24626⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨24626⟩⟩, ⟨.program ⟨257⟩, ⟨50761⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge53096

namespace LeftMerge53126
def owner : Owner := ⟨.program ⟨257⟩, ⟨52320⟩⟩
def mergeEvent : Nat := 53126
def frameStart : Nat := 53066
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24626⟩⟩, ⟨.program ⟨257⟩, ⟨50761⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events207.exact53122RawTerms
def rightRaw : List Term := Proof.Events207.exact53120RawTerms
def group : MergeGroup := .operator 53122 53120
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 53122) (leftOrdinal := 0)
    (rightResult := 53120) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24626⟩⟩, ⟨.program ⟨257⟩, ⟨50761⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨24626⟩⟩, ⟨.program ⟨257⟩, ⟨50761⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge53126

namespace LeftMerge53149
def owner : Owner := ⟨.program ⟨257⟩, ⟨9582⟩⟩
def mergeEvent : Nat := 53149
def frameStart : Nat := 53066
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩] } }
def leftRaw : List Term := Proof.Events207.exact53145RawTerms
def rightRaw : List Term := Proof.Events207.exact53142RawTerms
def group : MergeGroup := .operator 53145 53142
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 53145) (leftOrdinal := 0)
    (rightResult := 53142) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7288⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9580⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge53149

namespace LeftMerge53158
def owner : Owner := ⟨.program ⟨257⟩, ⟨52610⟩⟩
def mergeEvent : Nat := 53158
def frameStart : Nat := 53066
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52607⟩⟩] } }
def leftRaw : List Term := Proof.Events207.exact53154RawTerms
def rightRaw : List Term := Proof.Events207.exact53111RawTerms
def group : MergeGroup := .operator 53154 53111
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 53154) (leftOrdinal := 0)
    (rightResult := 53111) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨52607⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52607⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge53158

namespace LeftMerge53159
def owner : Owner := ⟨.program ⟨257⟩, ⟨52610⟩⟩
def mergeEvent : Nat := 53159
def frameStart : Nat := 53066
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24626⟩⟩, ⟨.program ⟨257⟩, ⟨50761⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨52607⟩⟩] } }
def leftRaw : List Term := Proof.Events207.exact53154RawTerms
def rightRaw : List Term := Proof.Events207.exact53111RawTerms
def group : MergeGroup := .operator 53154 53111
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 53154) (leftOrdinal := 1)
    (rightResult := 53111) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24626⟩⟩, ⟨.program ⟨257⟩, ⟨50761⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨52607⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨24626⟩⟩, ⟨.program ⟨257⟩, ⟨50761⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨52607⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge53159

namespace LeftMerge53161
def owner : Owner := ⟨.program ⟨257⟩, ⟨52610⟩⟩
def mergeEvent : Nat := 53161
def frameStart : Nat := 53066
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24626⟩⟩, ⟨.program ⟨257⟩, ⟨50761⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨52057⟩⟩] } }
def rhsRaw : List Term := Proof.Events207.exact53108RawTerms
def group : MergeGroup := .relation 53160
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 53160) (rhsResult := 53108)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨24626⟩⟩, ⟨.program ⟨257⟩, ⟨50761⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨52607⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨52607⟩⟩) ⟨52057⟩ 53108) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨52057⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨24626⟩⟩, ⟨.program ⟨257⟩, ⟨50761⟩⟩], [⟨.program ⟨257⟩, ⟨52057⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge53161

namespace LeftMerge53169
def owner : Owner := ⟨.program ⟨257⟩, ⟨50954⟩⟩
def mergeEvent : Nat := 53169
def frameStart : Nat := 53066
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨50952⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events207.exact53122RawTerms
def rightRaw : List Term := Proof.Events207.exact53165RawTerms
def group : MergeGroup := .operator 53122 53165
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 53122) (leftOrdinal := 0)
    (rightResult := 53165) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨50952⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨50952⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge53169

namespace LeftMerge53186
def owner : Owner := ⟨.program ⟨257⟩, ⟨51532⟩⟩
def mergeEvent : Nat := 53186
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7183⟩⟩] } }
def rhsRaw : List Term := Proof.Events207.exact53183RawTerms
def group : MergeGroup := .relation 53185
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 53185) (rhsResult := 53183)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51529⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 53184 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51529⟩⟩]⟩) (none) 53183) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7183⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7183⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge53186

namespace LeftMerge53187
def owner : Owner := ⟨.program ⟨257⟩, ⟨51532⟩⟩
def mergeEvent : Nat := 53187
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52607⟩⟩] } }
def rhsRaw : List Term := Proof.Events207.exact53183RawTerms
def group : MergeGroup := .relation 53185
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 53185) (rhsResult := 53183)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51529⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 53184 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51529⟩⟩]⟩) (none) 53183) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52607⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52607⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge53187

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
