import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge202879
def owner : Owner := ⟨.program ⟨257⟩, ⟨69097⟩⟩
def mergeEvent : Nat := 202879
def frameStart : Nat := 202336
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨54179⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events792.exact202863RawTerms
def rightRaw : List Term := Proof.Events792.exact202861RawTerms
def group : MergeGroup := .operator 202863 202861
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 202863) (leftOrdinal := 0)
    (rightResult := 202861) (rightOrdinal := 13) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨54179⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨54179⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge202879

namespace LeftMerge202880
def owner : Owner := ⟨.program ⟨257⟩, ⟨69097⟩⟩
def mergeEvent : Nat := 202880
def frameStart : Nat := 202336
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨51199⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events792.exact202863RawTerms
def rightRaw : List Term := Proof.Events792.exact202861RawTerms
def group : MergeGroup := .operator 202863 202861
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 202863) (leftOrdinal := 0)
    (rightResult := 202861) (rightOrdinal := 12) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨51199⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨51199⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge202880

namespace LeftMerge202881
def owner : Owner := ⟨.program ⟨257⟩, ⟨69097⟩⟩
def mergeEvent : Nat := 202881
def frameStart : Nat := 202336
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨32144⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events792.exact202863RawTerms
def rightRaw : List Term := Proof.Events792.exact202861RawTerms
def group : MergeGroup := .operator 202863 202861
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 202863) (leftOrdinal := 0)
    (rightResult := 202861) (rightOrdinal := 5) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨32144⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨32144⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge202881

namespace LeftMerge202882
def owner : Owner := ⟨.program ⟨257⟩, ⟨69097⟩⟩
def mergeEvent : Nat := 202882
def frameStart : Nat := 202336
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨22124⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events792.exact202863RawTerms
def rightRaw : List Term := Proof.Events792.exact202861RawTerms
def group : MergeGroup := .operator 202863 202861
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 202863) (leftOrdinal := 0)
    (rightResult := 202861) (rightOrdinal := 2) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨22124⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨22124⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge202882

namespace LeftMerge202883
def owner : Owner := ⟨.program ⟨257⟩, ⟨69097⟩⟩
def mergeEvent : Nat := 202883
def frameStart : Nat := 202336
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨18904⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events792.exact202863RawTerms
def rightRaw : List Term := Proof.Events792.exact202861RawTerms
def group : MergeGroup := .operator 202863 202861
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 202863) (leftOrdinal := 0)
    (rightResult := 202861) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨18904⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨18904⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge202883

namespace LeftMerge202884
def owner : Owner := ⟨.program ⟨257⟩, ⟨69097⟩⟩
def mergeEvent : Nat := 202884
def frameStart : Nat := 202336
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨16067⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events792.exact202863RawTerms
def rightRaw : List Term := Proof.Events792.exact202861RawTerms
def group : MergeGroup := .operator 202863 202861
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 202863) (leftOrdinal := 0)
    (rightResult := 202861) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨16067⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨16067⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge202884

namespace LeftMerge203015
def owner : Owner := ⟨.program ⟨257⟩, ⟨71298⟩⟩
def mergeEvent : Nat := 203015
def frameStart : Nat := 202336
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7232⟩⟩, ⟨.program ⟨257⟩, ⟨71297⟩⟩] } }
def leftRaw : List Term := Proof.Events793.exact203011RawTerms
def rightRaw : List Term := Proof.Events792.exact202852RawTerms
def group : MergeGroup := .operator 203011 202852
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 203011) (leftOrdinal := 17)
    (rightResult := 202852) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7232⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71297⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7232⟩⟩, ⟨.program ⟨257⟩, ⟨71297⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge203015

namespace LeftMerge203016
def owner : Owner := ⟨.program ⟨257⟩, ⟨71298⟩⟩
def mergeEvent : Nat := 203016
def frameStart : Nat := 202336
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨71297⟩⟩] } }
def leftRaw : List Term := Proof.Events793.exact203011RawTerms
def rightRaw : List Term := Proof.Events792.exact202852RawTerms
def group : MergeGroup := .operator 203011 202852
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 203011) (leftOrdinal := 16)
    (rightResult := 202852) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7230⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71297⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨71297⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge203016

namespace LeftMerge203017
def owner : Owner := ⟨.program ⟨257⟩, ⟨71298⟩⟩
def mergeEvent : Nat := 203017
def frameStart : Nat := 202336
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨71297⟩⟩] } }
def leftRaw : List Term := Proof.Events793.exact203011RawTerms
def rightRaw : List Term := Proof.Events792.exact202852RawTerms
def group : MergeGroup := .operator 203011 202852
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 203011) (leftOrdinal := 15)
    (rightResult := 202852) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7228⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71297⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨71297⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge203017

namespace LeftMerge203018
def owner : Owner := ⟨.program ⟨257⟩, ⟨71298⟩⟩
def mergeEvent : Nat := 203018
def frameStart : Nat := 202336
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨71297⟩⟩] } }
def leftRaw : List Term := Proof.Events793.exact203011RawTerms
def rightRaw : List Term := Proof.Events792.exact202852RawTerms
def group : MergeGroup := .operator 203011 202852
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 203011) (leftOrdinal := 14)
    (rightResult := 202852) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7226⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71297⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨71297⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge203018

namespace LeftMerge203019
def owner : Owner := ⟨.program ⟨257⟩, ⟨71298⟩⟩
def mergeEvent : Nat := 203019
def frameStart : Nat := 202336
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨71297⟩⟩] } }
def leftRaw : List Term := Proof.Events793.exact203011RawTerms
def rightRaw : List Term := Proof.Events792.exact202852RawTerms
def group : MergeGroup := .operator 203011 202852
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 203011) (leftOrdinal := 13)
    (rightResult := 202852) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7224⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71297⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨71297⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge203019

namespace LeftMerge203020
def owner : Owner := ⟨.program ⟨257⟩, ⟨71298⟩⟩
def mergeEvent : Nat := 203020
def frameStart : Nat := 202336
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7222⟩⟩, ⟨.program ⟨257⟩, ⟨71297⟩⟩] } }
def leftRaw : List Term := Proof.Events793.exact203011RawTerms
def rightRaw : List Term := Proof.Events792.exact202852RawTerms
def group : MergeGroup := .operator 203011 202852
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 203011) (leftOrdinal := 12)
    (rightResult := 202852) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7222⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71297⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7222⟩⟩, ⟨.program ⟨257⟩, ⟨71297⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge203020

namespace LeftMerge203021
def owner : Owner := ⟨.program ⟨257⟩, ⟨71298⟩⟩
def mergeEvent : Nat := 203021
def frameStart : Nat := 202336
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7220⟩⟩, ⟨.program ⟨257⟩, ⟨71297⟩⟩] } }
def leftRaw : List Term := Proof.Events793.exact203011RawTerms
def rightRaw : List Term := Proof.Events792.exact202852RawTerms
def group : MergeGroup := .operator 203011 202852
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 203011) (leftOrdinal := 11)
    (rightResult := 202852) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7220⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71297⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7220⟩⟩, ⟨.program ⟨257⟩, ⟨71297⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge203021

namespace LeftMerge203022
def owner : Owner := ⟨.program ⟨257⟩, ⟨71298⟩⟩
def mergeEvent : Nat := 203022
def frameStart : Nat := 202336
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7218⟩⟩, ⟨.program ⟨257⟩, ⟨71297⟩⟩] } }
def leftRaw : List Term := Proof.Events793.exact203011RawTerms
def rightRaw : List Term := Proof.Events792.exact202852RawTerms
def group : MergeGroup := .operator 203011 202852
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 203011) (leftOrdinal := 10)
    (rightResult := 202852) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7218⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71297⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7218⟩⟩, ⟨.program ⟨257⟩, ⟨71297⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge203022

namespace LeftMerge203023
def owner : Owner := ⟨.program ⟨257⟩, ⟨71298⟩⟩
def mergeEvent : Nat := 203023
def frameStart : Nat := 202336
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7216⟩⟩, ⟨.program ⟨257⟩, ⟨71297⟩⟩] } }
def leftRaw : List Term := Proof.Events793.exact203011RawTerms
def rightRaw : List Term := Proof.Events792.exact202852RawTerms
def group : MergeGroup := .operator 203011 202852
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 203011) (leftOrdinal := 9)
    (rightResult := 202852) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7216⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71297⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7216⟩⟩, ⟨.program ⟨257⟩, ⟨71297⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge203023

namespace LeftMerge203024
def owner : Owner := ⟨.program ⟨257⟩, ⟨71298⟩⟩
def mergeEvent : Nat := 203024
def frameStart : Nat := 202336
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7214⟩⟩, ⟨.program ⟨257⟩, ⟨71297⟩⟩] } }
def leftRaw : List Term := Proof.Events793.exact203011RawTerms
def rightRaw : List Term := Proof.Events792.exact202852RawTerms
def group : MergeGroup := .operator 203011 202852
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 203011) (leftOrdinal := 8)
    (rightResult := 202852) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7214⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71297⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7214⟩⟩, ⟨.program ⟨257⟩, ⟨71297⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge203024

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
