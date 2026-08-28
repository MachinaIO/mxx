import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge37989
def owner : Owner := ⟨.program ⟨257⟩, ⟨53769⟩⟩
def mergeEvent : Nat := 37989
def frameStart : Nat := 37959
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24878⟩⟩, ⟨.program ⟨257⟩, ⟨53768⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events148.exact37985RawTerms
def rightRaw : List Term := Proof.Events148.exact37982RawTerms
def group : MergeGroup := .operator 37985 37982
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 37985) (leftOrdinal := 0)
    (rightResult := 37982) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨53768⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24878⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨24878⟩⟩, ⟨.program ⟨257⟩, ⟨53768⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge37989

namespace LeftMerge38019
def owner : Owner := ⟨.program ⟨257⟩, ⟨55304⟩⟩
def mergeEvent : Nat := 38019
def frameStart : Nat := 37959
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24878⟩⟩, ⟨.program ⟨257⟩, ⟨53768⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events148.exact38015RawTerms
def rightRaw : List Term := Proof.Events148.exact38013RawTerms
def group : MergeGroup := .operator 38015 38013
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 38015) (leftOrdinal := 0)
    (rightResult := 38013) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24878⟩⟩, ⟨.program ⟨257⟩, ⟨53768⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨24878⟩⟩, ⟨.program ⟨257⟩, ⟨53768⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge38019

namespace LeftMerge38042
def owner : Owner := ⟨.program ⟨257⟩, ⟨9531⟩⟩
def mergeEvent : Nat := 38042
def frameStart : Nat := 37959
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩] } }
def leftRaw : List Term := Proof.Events148.exact38038RawTerms
def rightRaw : List Term := Proof.Events148.exact38035RawTerms
def group : MergeGroup := .operator 38038 38035
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 38038) (leftOrdinal := 0)
    (rightResult := 38035) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9529⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge38042

namespace LeftMerge38051
def owner : Owner := ⟨.program ⟨257⟩, ⟨55601⟩⟩
def mergeEvent : Nat := 38051
def frameStart : Nat := 37959
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55598⟩⟩] } }
def leftRaw : List Term := Proof.Events148.exact38047RawTerms
def rightRaw : List Term := Proof.Events148.exact38004RawTerms
def group : MergeGroup := .operator 38047 38004
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 38047) (leftOrdinal := 0)
    (rightResult := 38004) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨55598⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55598⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge38051

namespace LeftMerge38052
def owner : Owner := ⟨.program ⟨257⟩, ⟨55601⟩⟩
def mergeEvent : Nat := 38052
def frameStart : Nat := 37959
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24878⟩⟩, ⟨.program ⟨257⟩, ⟨53768⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨55598⟩⟩] } }
def leftRaw : List Term := Proof.Events148.exact38047RawTerms
def rightRaw : List Term := Proof.Events148.exact38004RawTerms
def group : MergeGroup := .operator 38047 38004
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 38047) (leftOrdinal := 1)
    (rightResult := 38004) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24878⟩⟩, ⟨.program ⟨257⟩, ⟨53768⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨55598⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨24878⟩⟩, ⟨.program ⟨257⟩, ⟨53768⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨55598⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge38052

namespace LeftMerge38054
def owner : Owner := ⟨.program ⟨257⟩, ⟨55601⟩⟩
def mergeEvent : Nat := 38054
def frameStart : Nat := 37959
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24878⟩⟩, ⟨.program ⟨257⟩, ⟨53768⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨55043⟩⟩] } }
def rhsRaw : List Term := Proof.Events148.exact38001RawTerms
def group : MergeGroup := .relation 38053
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 38053) (rhsResult := 38001)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨24878⟩⟩, ⟨.program ⟨257⟩, ⟨53768⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨55598⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨55598⟩⟩) ⟨55043⟩ 38001) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨55043⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨24878⟩⟩, ⟨.program ⟨257⟩, ⟨53768⟩⟩], [⟨.program ⟨257⟩, ⟨55043⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge38054

namespace LeftMerge38062
def owner : Owner := ⟨.program ⟨257⟩, ⟨53942⟩⟩
def mergeEvent : Nat := 38062
def frameStart : Nat := 37959
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨53940⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events148.exact38015RawTerms
def rightRaw : List Term := Proof.Events148.exact38058RawTerms
def group : MergeGroup := .operator 38015 38058
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 38015) (leftOrdinal := 0)
    (rightResult := 38058) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨53940⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨53940⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge38062

namespace LeftMerge38079
def owner : Owner := ⟨.program ⟨257⟩, ⟨54522⟩⟩
def mergeEvent : Nat := 38079
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7184⟩⟩] } }
def rhsRaw : List Term := Proof.Events148.exact38076RawTerms
def group : MergeGroup := .relation 38078
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 38078) (rhsResult := 38076)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54519⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 38077 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54519⟩⟩]⟩) (none) 38076) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7184⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7184⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge38079

namespace LeftMerge38080
def owner : Owner := ⟨.program ⟨257⟩, ⟨54522⟩⟩
def mergeEvent : Nat := 38080
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55598⟩⟩] } }
def rhsRaw : List Term := Proof.Events148.exact38076RawTerms
def group : MergeGroup := .relation 38078
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 38078) (rhsResult := 38076)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54519⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 38077 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54519⟩⟩]⟩) (none) 38076) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55598⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55598⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge38080

namespace LeftMerge38081
def owner : Owner := ⟨.program ⟨257⟩, ⟨54522⟩⟩
def mergeEvent : Nat := 38081
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨24878⟩⟩, ⟨.program ⟨257⟩, ⟨53768⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨55043⟩⟩] } }
def rhsRaw : List Term := Proof.Events148.exact38076RawTerms
def group : MergeGroup := .relation 38078
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 38078) (rhsResult := 38076)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54519⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 38077 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54519⟩⟩]⟩) (none) 38076) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24878⟩⟩, ⟨.program ⟨257⟩, ⟨53768⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨55043⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨24878⟩⟩, ⟨.program ⟨257⟩, ⟨53768⟩⟩], [⟨.program ⟨257⟩, ⟨55043⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge38081

namespace LeftMerge38082
def owner : Owner := ⟨.program ⟨257⟩, ⟨54522⟩⟩
def mergeEvent : Nat := 38082
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨53940⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events148.exact38076RawTerms
def group : MergeGroup := .relation 38078
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 38078) (rhsResult := 38076)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54519⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 38077 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54519⟩⟩]⟩) (none) 38076) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨53940⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨53940⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge38082

namespace LeftMerge38087
def owner : Owner := ⟨.program ⟨257⟩, ⟨55600⟩⟩
def mergeEvent : Nat := 38087
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨24878⟩⟩, ⟨.program ⟨257⟩, ⟨53768⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨55043⟩⟩] } }
def leftRaw : List Term := Proof.Events148.exact38083RawTerms
def rightRaw : List Term := Proof.Events148.exact37897RawTerms
def group : MergeGroup := .operator 38083 37897
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 38083) (leftOrdinal := 2)
    (rightResult := 37897) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨24878⟩⟩, ⟨.program ⟨257⟩, ⟨53768⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨55043⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨24878⟩⟩, ⟨.program ⟨257⟩, ⟨53768⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨55043⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨24878⟩⟩, ⟨.program ⟨257⟩, ⟨53768⟩⟩], [⟨.program ⟨257⟩, ⟨55043⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge38087

namespace LeftMerge38088
def owner : Owner := ⟨.program ⟨257⟩, ⟨55600⟩⟩
def mergeEvent : Nat := 38088
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55598⟩⟩] } }
def leftRaw : List Term := Proof.Events148.exact38083RawTerms
def rightRaw : List Term := Proof.Events148.exact37897RawTerms
def group : MergeGroup := .operator 38083 37897
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 38083) (leftOrdinal := 1)
    (rightResult := 37897) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55598⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55598⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55598⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge38088

namespace LeftMerge38096
def owner : Owner := ⟨.program ⟨257⟩, ⟨56213⟩⟩
def mergeEvent : Nat := 38096
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨56211⟩⟩] } }
def leftRaw : List Term := Proof.Events148.exact38090RawTerms
def rightRaw : List Term := Proof.Events147.exact37813RawTerms
def group : MergeGroup := .operator 38090 37813
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 38090) (leftOrdinal := 0)
    (rightResult := 37813) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7184⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨56211⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨56211⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge38096

namespace LeftMerge38097
def owner : Owner := ⟨.program ⟨257⟩, ⟨56213⟩⟩
def mergeEvent : Nat := 38097
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨53940⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨56211⟩⟩] } }
def leftRaw : List Term := Proof.Events148.exact38090RawTerms
def rightRaw : List Term := Proof.Events147.exact37813RawTerms
def group : MergeGroup := .operator 38090 37813
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 38090) (leftOrdinal := 1)
    (rightResult := 37813) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨53940⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨56211⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨53940⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨56211⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge38097

namespace LeftMerge38099
def owner : Owner := ⟨.program ⟨257⟩, ⟨56213⟩⟩
def mergeEvent : Nat := 38099
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨53940⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨55222⟩⟩] } }
def rhsRaw : List Term := Proof.Events147.exact37810RawTerms
def group : MergeGroup := .relation 38098
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 38098) (rhsResult := 37810)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨53940⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨56211⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨56211⟩⟩) ⟨55222⟩ 37810) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨55222⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨53940⟩⟩], [⟨.program ⟨257⟩, ⟨55222⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge38099

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
