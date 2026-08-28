import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge46723
def owner : Owner := ⟨.program ⟨257⟩, ⟨49748⟩⟩
def mergeEvent : Nat := 46723
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨15201⟩⟩, ⟨.program ⟨257⟩, ⟨48026⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨49747⟩⟩] } }
def leftRaw : List Term := Proof.Events182.exact46717RawTerms
def rightRaw : List Term := Proof.Events182.exact46648RawTerms
def group : MergeGroup := .operator 46717 46648
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 46717) (leftOrdinal := 1)
    (rightResult := 46648) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨15201⟩⟩, ⟨.program ⟨257⟩, ⟨48026⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨49747⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨15201⟩⟩, ⟨.program ⟨257⟩, ⟨48026⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨49747⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge46723

namespace LeftMerge46725
def owner : Owner := ⟨.program ⟨257⟩, ⟨49748⟩⟩
def mergeEvent : Nat := 46725
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨15201⟩⟩, ⟨.program ⟨257⟩, ⟨48026⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨49197⟩⟩] } }
def rhsRaw : List Term := Proof.Events182.exact46645RawTerms
def group : MergeGroup := .relation 46724
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 46724) (rhsResult := 46645)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨15201⟩⟩, ⟨.program ⟨257⟩, ⟨48026⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨49747⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨49747⟩⟩) ⟨49197⟩ 46645) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨49197⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨15201⟩⟩, ⟨.program ⟨257⟩, ⟨48026⟩⟩], [⟨.program ⟨257⟩, ⟨49197⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge46725

namespace LeftMerge46726
def owner : Owner := ⟨.program ⟨257⟩, ⟨49748⟩⟩
def mergeEvent : Nat := 46726
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49747⟩⟩] } }
def leftRaw : List Term := Proof.Events182.exact46717RawTerms
def rightRaw : List Term := Proof.Events182.exact46648RawTerms
def group : MergeGroup := .operator 46717 46648
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 46717) (leftOrdinal := 0)
    (rightResult := 46648) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨49747⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49747⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge46726

namespace LeftMerge46738
def owner : Owner := ⟨.program ⟨257⟩, ⟨11215⟩⟩
def mergeEvent : Nat := 46738
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }
def leftRaw : List Term := Proof.Events181.exact46523RawTerms
def rightRaw : List Term := Proof.Events067.exact17158RawTerms
def group : MergeGroup := .operator 46523 17158
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 46523) (leftOrdinal := 0)
    (rightResult := 17158) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge46738

namespace LeftMerge46751
def owner : Owner := ⟨.program ⟨257⟩, ⟨48672⟩⟩
def mergeEvent : Nat := 46751
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48669⟩⟩] } }
def leftRaw : List Term := Proof.Events182.exact46745RawTerms
def rightRaw : List Term := Proof.Events182.exact46734RawTerms
def group : MergeGroup := .operator 46745 46734
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 46745) (leftOrdinal := 0)
    (rightResult := 46734) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨48669⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48669⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge46751

namespace LeftMerge46830
def owner : Owner := ⟨.program ⟨257⟩, ⟨48027⟩⟩
def mergeEvent : Nat := 46830
def frameStart : Nat := 46800
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨15201⟩⟩, ⟨.program ⟨257⟩, ⟨48026⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events182.exact46826RawTerms
def rightRaw : List Term := Proof.Events182.exact46823RawTerms
def group : MergeGroup := .operator 46826 46823
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 46826) (leftOrdinal := 0)
    (rightResult := 46823) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨15201⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨48026⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨15201⟩⟩, ⟨.program ⟨257⟩, ⟨48026⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge46830

namespace LeftMerge46860
def owner : Owner := ⟨.program ⟨257⟩, ⟨49460⟩⟩
def mergeEvent : Nat := 46860
def frameStart : Nat := 46800
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨15201⟩⟩, ⟨.program ⟨257⟩, ⟨48026⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events183.exact46856RawTerms
def rightRaw : List Term := Proof.Events183.exact46854RawTerms
def group : MergeGroup := .operator 46856 46854
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 46856) (leftOrdinal := 0)
    (rightResult := 46854) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨15201⟩⟩, ⟨.program ⟨257⟩, ⟨48026⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨15201⟩⟩, ⟨.program ⟨257⟩, ⟨48026⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge46860

namespace LeftMerge46883
def owner : Owner := ⟨.program ⟨257⟩, ⟨9567⟩⟩
def mergeEvent : Nat := 46883
def frameStart : Nat := 46800
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩] } }
def leftRaw : List Term := Proof.Events183.exact46879RawTerms
def rightRaw : List Term := Proof.Events183.exact46876RawTerms
def group : MergeGroup := .operator 46879 46876
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 46879) (leftOrdinal := 0)
    (rightResult := 46876) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9565⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge46883

namespace LeftMerge46892
def owner : Owner := ⟨.program ⟨257⟩, ⟨49750⟩⟩
def mergeEvent : Nat := 46892
def frameStart : Nat := 46800
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49747⟩⟩] } }
def leftRaw : List Term := Proof.Events183.exact46888RawTerms
def rightRaw : List Term := Proof.Events182.exact46845RawTerms
def group : MergeGroup := .operator 46888 46845
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 46888) (leftOrdinal := 0)
    (rightResult := 46845) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨49747⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49747⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge46892

namespace LeftMerge46893
def owner : Owner := ⟨.program ⟨257⟩, ⟨49750⟩⟩
def mergeEvent : Nat := 46893
def frameStart : Nat := 46800
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨15201⟩⟩, ⟨.program ⟨257⟩, ⟨48026⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨49747⟩⟩] } }
def leftRaw : List Term := Proof.Events183.exact46888RawTerms
def rightRaw : List Term := Proof.Events182.exact46845RawTerms
def group : MergeGroup := .operator 46888 46845
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 46888) (leftOrdinal := 1)
    (rightResult := 46845) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨15201⟩⟩, ⟨.program ⟨257⟩, ⟨48026⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨49747⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨15201⟩⟩, ⟨.program ⟨257⟩, ⟨48026⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨49747⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge46893

namespace LeftMerge46895
def owner : Owner := ⟨.program ⟨257⟩, ⟨49750⟩⟩
def mergeEvent : Nat := 46895
def frameStart : Nat := 46800
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨15201⟩⟩, ⟨.program ⟨257⟩, ⟨48026⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨49197⟩⟩] } }
def rhsRaw : List Term := Proof.Events182.exact46842RawTerms
def group : MergeGroup := .relation 46894
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 46894) (rhsResult := 46842)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨15201⟩⟩, ⟨.program ⟨257⟩, ⟨48026⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨49747⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨49747⟩⟩) ⟨49197⟩ 46842) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨49197⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨15201⟩⟩, ⟨.program ⟨257⟩, ⟨48026⟩⟩], [⟨.program ⟨257⟩, ⟨49197⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge46895

namespace LeftMerge46903
def owner : Owner := ⟨.program ⟨257⟩, ⟨48214⟩⟩
def mergeEvent : Nat := 46903
def frameStart : Nat := 46800
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨48212⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events183.exact46856RawTerms
def rightRaw : List Term := Proof.Events183.exact46899RawTerms
def group : MergeGroup := .operator 46856 46899
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 46856) (leftOrdinal := 0)
    (rightResult := 46899) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨48212⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨48212⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge46903

namespace LeftMerge46920
def owner : Owner := ⟨.program ⟨257⟩, ⟨48672⟩⟩
def mergeEvent : Nat := 46920
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7196⟩⟩] } }
def rhsRaw : List Term := Proof.Events183.exact46917RawTerms
def group : MergeGroup := .relation 46919
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 46919) (rhsResult := 46917)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48669⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 46918 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48669⟩⟩]⟩) (none) 46917) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7196⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7196⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge46920

namespace LeftMerge46921
def owner : Owner := ⟨.program ⟨257⟩, ⟨48672⟩⟩
def mergeEvent : Nat := 46921
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49747⟩⟩] } }
def rhsRaw : List Term := Proof.Events183.exact46917RawTerms
def group : MergeGroup := .relation 46919
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 46919) (rhsResult := 46917)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48669⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 46918 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48669⟩⟩]⟩) (none) 46917) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49747⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49747⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge46921

namespace LeftMerge46922
def owner : Owner := ⟨.program ⟨257⟩, ⟨48672⟩⟩
def mergeEvent : Nat := 46922
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨15201⟩⟩, ⟨.program ⟨257⟩, ⟨48026⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨49197⟩⟩] } }
def rhsRaw : List Term := Proof.Events183.exact46917RawTerms
def group : MergeGroup := .relation 46919
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 46919) (rhsResult := 46917)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48669⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 46918 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48669⟩⟩]⟩) (none) 46917) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨15201⟩⟩, ⟨.program ⟨257⟩, ⟨48026⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨49197⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨15201⟩⟩, ⟨.program ⟨257⟩, ⟨48026⟩⟩], [⟨.program ⟨257⟩, ⟨49197⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge46922

namespace LeftMerge46923
def owner : Owner := ⟨.program ⟨257⟩, ⟨48672⟩⟩
def mergeEvent : Nat := 46923
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨48212⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events183.exact46917RawTerms
def group : MergeGroup := .relation 46919
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 46919) (rhsResult := 46917)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48669⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 46918 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48669⟩⟩]⟩) (none) 46917) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨48212⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨48212⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge46923

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
