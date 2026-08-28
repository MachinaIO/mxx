import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge280723
def owner : Owner := ⟨.program ⟨257⟩, ⟨49594⟩⟩
def mergeEvent : Nat := 280723
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨14991⟩⟩, ⟨.program ⟨257⟩, ⟨47690⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨49593⟩⟩] } }
def leftRaw : List Term := Proof.Events1096.exact280717RawTerms
def rightRaw : List Term := Proof.Events1096.exact280648RawTerms
def group : MergeGroup := .operator 280717 280648
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 280717) (leftOrdinal := 1)
    (rightResult := 280648) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨14991⟩⟩, ⟨.program ⟨257⟩, ⟨47690⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨49593⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨14991⟩⟩, ⟨.program ⟨257⟩, ⟨47690⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨49593⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge280723

namespace LeftMerge280725
def owner : Owner := ⟨.program ⟨257⟩, ⟨49594⟩⟩
def mergeEvent : Nat := 280725
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨14991⟩⟩, ⟨.program ⟨257⟩, ⟨47690⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨49113⟩⟩] } }
def rhsRaw : List Term := Proof.Events1096.exact280645RawTerms
def group : MergeGroup := .relation 280724
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 280724) (rhsResult := 280645)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨14991⟩⟩, ⟨.program ⟨257⟩, ⟨47690⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨49593⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨49593⟩⟩) ⟨49113⟩ 280645) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨49113⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨14991⟩⟩, ⟨.program ⟨257⟩, ⟨47690⟩⟩], [⟨.program ⟨257⟩, ⟨49113⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge280725

namespace LeftMerge280726
def owner : Owner := ⟨.program ⟨257⟩, ⟨49594⟩⟩
def mergeEvent : Nat := 280726
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49593⟩⟩] } }
def leftRaw : List Term := Proof.Events1096.exact280717RawTerms
def rightRaw : List Term := Proof.Events1096.exact280648RawTerms
def group : MergeGroup := .operator 280717 280648
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 280717) (leftOrdinal := 0)
    (rightResult := 280648) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨49593⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49593⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge280726

namespace LeftMerge280738
def owner : Owner := ⟨.program ⟨257⟩, ⟨5490⟩⟩
def mergeEvent : Nat := 280738
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }
def leftRaw : List Term := Proof.Events1095.exact280523RawTerms
def rightRaw : List Term := Proof.Events067.exact17158RawTerms
def group : MergeGroup := .operator 280523 17158
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 280523) (leftOrdinal := 0)
    (rightResult := 17158) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge280738

namespace LeftMerge280751
def owner : Owner := ⟨.program ⟨257⟩, ⟨48532⟩⟩
def mergeEvent : Nat := 280751
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48529⟩⟩] } }
def leftRaw : List Term := Proof.Events1096.exact280745RawTerms
def rightRaw : List Term := Proof.Events1096.exact280734RawTerms
def group : MergeGroup := .operator 280745 280734
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 280745) (leftOrdinal := 0)
    (rightResult := 280734) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨48529⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48529⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge280751

namespace LeftMerge280830
def owner : Owner := ⟨.program ⟨257⟩, ⟨47691⟩⟩
def mergeEvent : Nat := 280830
def frameStart : Nat := 280800
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14991⟩⟩, ⟨.program ⟨257⟩, ⟨47690⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events1096.exact280826RawTerms
def rightRaw : List Term := Proof.Events1096.exact280823RawTerms
def group : MergeGroup := .operator 280826 280823
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 280826) (leftOrdinal := 0)
    (rightResult := 280823) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14991⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨47690⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨14991⟩⟩, ⟨.program ⟨257⟩, ⟨47690⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge280830

namespace LeftMerge280860
def owner : Owner := ⟨.program ⟨257⟩, ⟨49404⟩⟩
def mergeEvent : Nat := 280860
def frameStart : Nat := 280800
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14991⟩⟩, ⟨.program ⟨257⟩, ⟨47690⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1097.exact280856RawTerms
def rightRaw : List Term := Proof.Events1097.exact280854RawTerms
def group : MergeGroup := .operator 280856 280854
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 280856) (leftOrdinal := 0)
    (rightResult := 280854) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14991⟩⟩, ⟨.program ⟨257⟩, ⟨47690⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨14991⟩⟩, ⟨.program ⟨257⟩, ⟨47690⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge280860

namespace LeftMerge280881
def owner : Owner := ⟨.program ⟨257⟩, ⟨9567⟩⟩
def mergeEvent : Nat := 280881
def frameStart : Nat := 280800
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩] } }
def leftRaw : List Term := Proof.Events1097.exact280877RawTerms
def rightRaw : List Term := Proof.Events1097.exact280874RawTerms
def group : MergeGroup := .operator 280877 280874
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 280877) (leftOrdinal := 0)
    (rightResult := 280874) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9565⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge280881

namespace LeftMerge280890
def owner : Owner := ⟨.program ⟨257⟩, ⟨49596⟩⟩
def mergeEvent : Nat := 280890
def frameStart : Nat := 280800
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49593⟩⟩] } }
def leftRaw : List Term := Proof.Events1097.exact280886RawTerms
def rightRaw : List Term := Proof.Events1097.exact280845RawTerms
def group : MergeGroup := .operator 280886 280845
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 280886) (leftOrdinal := 0)
    (rightResult := 280845) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨49593⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49593⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge280890

namespace LeftMerge280891
def owner : Owner := ⟨.program ⟨257⟩, ⟨49596⟩⟩
def mergeEvent : Nat := 280891
def frameStart : Nat := 280800
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14991⟩⟩, ⟨.program ⟨257⟩, ⟨47690⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨49593⟩⟩] } }
def leftRaw : List Term := Proof.Events1097.exact280886RawTerms
def rightRaw : List Term := Proof.Events1097.exact280845RawTerms
def group : MergeGroup := .operator 280886 280845
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 280886) (leftOrdinal := 1)
    (rightResult := 280845) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14991⟩⟩, ⟨.program ⟨257⟩, ⟨47690⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨49593⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨14991⟩⟩, ⟨.program ⟨257⟩, ⟨47690⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨49593⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge280891

namespace LeftMerge280893
def owner : Owner := ⟨.program ⟨257⟩, ⟨49596⟩⟩
def mergeEvent : Nat := 280893
def frameStart : Nat := 280800
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14991⟩⟩, ⟨.program ⟨257⟩, ⟨47690⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨49113⟩⟩] } }
def rhsRaw : List Term := Proof.Events1097.exact280842RawTerms
def group : MergeGroup := .relation 280892
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 280892) (rhsResult := 280842)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨14991⟩⟩, ⟨.program ⟨257⟩, ⟨47690⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨49593⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨49593⟩⟩) ⟨49113⟩ 280842) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨49113⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨14991⟩⟩, ⟨.program ⟨257⟩, ⟨47690⟩⟩], [⟨.program ⟨257⟩, ⟨49113⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge280893

namespace LeftMerge280901
def owner : Owner := ⟨.program ⟨257⟩, ⟨48102⟩⟩
def mergeEvent : Nat := 280901
def frameStart : Nat := 280800
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨48100⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1097.exact280856RawTerms
def rightRaw : List Term := Proof.Events1097.exact280897RawTerms
def group : MergeGroup := .operator 280856 280897
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 280856) (leftOrdinal := 0)
    (rightResult := 280897) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨48100⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨48100⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge280901

namespace LeftMerge280918
def owner : Owner := ⟨.program ⟨257⟩, ⟨48532⟩⟩
def mergeEvent : Nat := 280918
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7196⟩⟩] } }
def rhsRaw : List Term := Proof.Events1097.exact280915RawTerms
def group : MergeGroup := .relation 280917
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 280917) (rhsResult := 280915)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48529⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 280916 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48529⟩⟩]⟩) (none) 280915) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7196⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7196⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge280918

namespace LeftMerge280919
def owner : Owner := ⟨.program ⟨257⟩, ⟨48532⟩⟩
def mergeEvent : Nat := 280919
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49593⟩⟩] } }
def rhsRaw : List Term := Proof.Events1097.exact280915RawTerms
def group : MergeGroup := .relation 280917
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 280917) (rhsResult := 280915)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48529⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 280916 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48529⟩⟩]⟩) (none) 280915) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49593⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49593⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge280919

namespace LeftMerge280920
def owner : Owner := ⟨.program ⟨257⟩, ⟨48532⟩⟩
def mergeEvent : Nat := 280920
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨14991⟩⟩, ⟨.program ⟨257⟩, ⟨47690⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨49113⟩⟩] } }
def rhsRaw : List Term := Proof.Events1097.exact280915RawTerms
def group : MergeGroup := .relation 280917
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 280917) (rhsResult := 280915)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48529⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 280916 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48529⟩⟩]⟩) (none) 280915) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14991⟩⟩, ⟨.program ⟨257⟩, ⟨47690⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨49113⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨14991⟩⟩, ⟨.program ⟨257⟩, ⟨47690⟩⟩], [⟨.program ⟨257⟩, ⟨49113⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge280920

namespace LeftMerge280921
def owner : Owner := ⟨.program ⟨257⟩, ⟨48532⟩⟩
def mergeEvent : Nat := 280921
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨48100⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events1097.exact280915RawTerms
def group : MergeGroup := .relation 280917
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 280917) (rhsResult := 280915)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48529⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 280916 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48529⟩⟩]⟩) (none) 280915) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨48100⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨48100⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge280921

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
