import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge140742
def owner : Owner := ⟨.program ⟨257⟩, ⟨50364⟩⟩
def mergeEvent : Nat := 140742
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨50356⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7308⟩⟩] } }
def leftRaw : List Term := Proof.Events549.exact140738RawTerms
def rightRaw : List Term := Proof.Events549.exact140708RawTerms
def group : MergeGroup := .operator 140738 140708
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 140738) (leftOrdinal := 1)
    (rightResult := 140708) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨50356⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7308⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨50356⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7308⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨50356⟩⟩], [⟨.program ⟨257⟩, ⟨7308⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge140742

namespace LeftMerge140750
def owner : Owner := ⟨.program ⟨257⟩, ⟨52443⟩⟩
def mergeEvent : Nat := 140750
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨24446⟩⟩, ⟨.program ⟨257⟩, ⟨50356⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨52442⟩⟩] } }
def leftRaw : List Term := Proof.Events549.exact140744RawTerms
def rightRaw : List Term := Proof.Events549.exact140680RawTerms
def group : MergeGroup := .operator 140744 140680
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 140744) (leftOrdinal := 1)
    (rightResult := 140680) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨24446⟩⟩, ⟨.program ⟨257⟩, ⟨50356⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨52442⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨24446⟩⟩, ⟨.program ⟨257⟩, ⟨50356⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨52442⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge140750

namespace LeftMerge140752
def owner : Owner := ⟨.program ⟨257⟩, ⟨52443⟩⟩
def mergeEvent : Nat := 140752
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨24446⟩⟩, ⟨.program ⟨257⟩, ⟨50356⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨51967⟩⟩] } }
def rhsRaw : List Term := Proof.Events549.exact140677RawTerms
def group : MergeGroup := .relation 140751
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 140751) (rhsResult := 140677)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨24446⟩⟩, ⟨.program ⟨257⟩, ⟨50356⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨52442⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨52442⟩⟩) ⟨51967⟩ 140677) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨51967⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨24446⟩⟩, ⟨.program ⟨257⟩, ⟨50356⟩⟩], [⟨.program ⟨257⟩, ⟨51967⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge140752

namespace LeftMerge140753
def owner : Owner := ⟨.program ⟨257⟩, ⟨52443⟩⟩
def mergeEvent : Nat := 140753
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52442⟩⟩] } }
def leftRaw : List Term := Proof.Events549.exact140744RawTerms
def rightRaw : List Term := Proof.Events549.exact140680RawTerms
def group : MergeGroup := .operator 140744 140680
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 140744) (leftOrdinal := 0)
    (rightResult := 140680) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨52442⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52442⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge140753

namespace LeftMerge140767
def owner : Owner := ⟨.program ⟨257⟩, ⟨51382⟩⟩
def mergeEvent : Nat := 140767
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51379⟩⟩] } }
def leftRaw : List Term := Proof.Events525.exact134495RawTerms
def rightRaw : List Term := Proof.Events549.exact140761RawTerms
def group : MergeGroup := .operator 134495 140761
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 134495) (leftOrdinal := 0)
    (rightResult := 140761) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨51379⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51379⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge140767

namespace LeftMerge140846
def owner : Owner := ⟨.program ⟨257⟩, ⟨50357⟩⟩
def mergeEvent : Nat := 140846
def frameStart : Nat := 140816
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24446⟩⟩, ⟨.program ⟨257⟩, ⟨50356⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events550.exact140842RawTerms
def rightRaw : List Term := Proof.Events550.exact140839RawTerms
def group : MergeGroup := .operator 140842 140839
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 140842) (leftOrdinal := 0)
    (rightResult := 140839) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨50356⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24446⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨24446⟩⟩, ⟨.program ⟨257⟩, ⟨50356⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge140846

namespace LeftMerge140876
def owner : Owner := ⟨.program ⟨257⟩, ⟨52260⟩⟩
def mergeEvent : Nat := 140876
def frameStart : Nat := 140816
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24446⟩⟩, ⟨.program ⟨257⟩, ⟨50356⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events550.exact140872RawTerms
def rightRaw : List Term := Proof.Events550.exact140870RawTerms
def group : MergeGroup := .operator 140872 140870
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 140872) (leftOrdinal := 0)
    (rightResult := 140870) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24446⟩⟩, ⟨.program ⟨257⟩, ⟨50356⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨24446⟩⟩, ⟨.program ⟨257⟩, ⟨50356⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge140876

namespace LeftMerge140899
def owner : Owner := ⟨.program ⟨257⟩, ⟨9582⟩⟩
def mergeEvent : Nat := 140899
def frameStart : Nat := 140816
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩] } }
def leftRaw : List Term := Proof.Events550.exact140895RawTerms
def rightRaw : List Term := Proof.Events550.exact140892RawTerms
def group : MergeGroup := .operator 140895 140892
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 140895) (leftOrdinal := 0)
    (rightResult := 140892) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7288⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9580⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge140899

namespace LeftMerge140908
def owner : Owner := ⟨.program ⟨257⟩, ⟨52445⟩⟩
def mergeEvent : Nat := 140908
def frameStart : Nat := 140816
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52442⟩⟩] } }
def leftRaw : List Term := Proof.Events550.exact140904RawTerms
def rightRaw : List Term := Proof.Events550.exact140861RawTerms
def group : MergeGroup := .operator 140904 140861
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 140904) (leftOrdinal := 0)
    (rightResult := 140861) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨52442⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52442⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge140908

namespace LeftMerge140909
def owner : Owner := ⟨.program ⟨257⟩, ⟨52445⟩⟩
def mergeEvent : Nat := 140909
def frameStart : Nat := 140816
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24446⟩⟩, ⟨.program ⟨257⟩, ⟨50356⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨52442⟩⟩] } }
def leftRaw : List Term := Proof.Events550.exact140904RawTerms
def rightRaw : List Term := Proof.Events550.exact140861RawTerms
def group : MergeGroup := .operator 140904 140861
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 140904) (leftOrdinal := 1)
    (rightResult := 140861) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24446⟩⟩, ⟨.program ⟨257⟩, ⟨50356⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨52442⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨24446⟩⟩, ⟨.program ⟨257⟩, ⟨50356⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨52442⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge140909

namespace LeftMerge140911
def owner : Owner := ⟨.program ⟨257⟩, ⟨52445⟩⟩
def mergeEvent : Nat := 140911
def frameStart : Nat := 140816
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24446⟩⟩, ⟨.program ⟨257⟩, ⟨50356⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨51967⟩⟩] } }
def rhsRaw : List Term := Proof.Events550.exact140858RawTerms
def group : MergeGroup := .relation 140910
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 140910) (rhsResult := 140858)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨24446⟩⟩, ⟨.program ⟨257⟩, ⟨50356⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨52442⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨52442⟩⟩) ⟨51967⟩ 140858) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨51967⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨24446⟩⟩, ⟨.program ⟨257⟩, ⟨50356⟩⟩], [⟨.program ⟨257⟩, ⟨51967⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge140911

namespace LeftMerge140919
def owner : Owner := ⟨.program ⟨257⟩, ⟨50834⟩⟩
def mergeEvent : Nat := 140919
def frameStart : Nat := 140816
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨50832⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events550.exact140872RawTerms
def rightRaw : List Term := Proof.Events550.exact140915RawTerms
def group : MergeGroup := .operator 140872 140915
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 140872) (leftOrdinal := 0)
    (rightResult := 140915) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨50832⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨50832⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge140919

namespace LeftMerge140936
def owner : Owner := ⟨.program ⟨257⟩, ⟨51382⟩⟩
def mergeEvent : Nat := 140936
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7183⟩⟩] } }
def rhsRaw : List Term := Proof.Events550.exact140933RawTerms
def group : MergeGroup := .relation 140935
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 140935) (rhsResult := 140933)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51379⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 140934 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51379⟩⟩]⟩) (none) 140933) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7183⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7183⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge140936

namespace LeftMerge140937
def owner : Owner := ⟨.program ⟨257⟩, ⟨51382⟩⟩
def mergeEvent : Nat := 140937
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52442⟩⟩] } }
def rhsRaw : List Term := Proof.Events550.exact140933RawTerms
def group : MergeGroup := .relation 140935
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 140935) (rhsResult := 140933)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51379⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 140934 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51379⟩⟩]⟩) (none) 140933) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52442⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52442⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge140937

namespace LeftMerge140938
def owner : Owner := ⟨.program ⟨257⟩, ⟨51382⟩⟩
def mergeEvent : Nat := 140938
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨24446⟩⟩, ⟨.program ⟨257⟩, ⟨50356⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨51967⟩⟩] } }
def rhsRaw : List Term := Proof.Events550.exact140933RawTerms
def group : MergeGroup := .relation 140935
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 140935) (rhsResult := 140933)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51379⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 140934 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51379⟩⟩]⟩) (none) 140933) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24446⟩⟩, ⟨.program ⟨257⟩, ⟨50356⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨51967⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨24446⟩⟩, ⟨.program ⟨257⟩, ⟨50356⟩⟩], [⟨.program ⟨257⟩, ⟨51967⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge140938

namespace LeftMerge140939
def owner : Owner := ⟨.program ⟨257⟩, ⟨51382⟩⟩
def mergeEvent : Nat := 140939
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨50832⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events550.exact140933RawTerms
def group : MergeGroup := .relation 140935
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 140935) (rhsResult := 140933)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51379⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 140934 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51379⟩⟩]⟩) (none) 140933) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨50832⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨50832⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge140939

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
