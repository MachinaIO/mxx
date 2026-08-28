import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge86017
def owner : Owner := ⟨.program ⟨257⟩, ⟨71438⟩⟩
def mergeEvent : Nat := 86017
def frameStart : Nat := 85336
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩] } }
def leftRaw : List Term := Proof.Events335.exact86011RawTerms
def rightRaw : List Term := Proof.Events335.exact85852RawTerms
def group : MergeGroup := .operator 86011 85852
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 86011) (leftOrdinal := 15)
    (rightResult := 85852) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7228⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71437⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge86017

namespace LeftMerge86018
def owner : Owner := ⟨.program ⟨257⟩, ⟨71438⟩⟩
def mergeEvent : Nat := 86018
def frameStart : Nat := 85336
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩] } }
def leftRaw : List Term := Proof.Events335.exact86011RawTerms
def rightRaw : List Term := Proof.Events335.exact85852RawTerms
def group : MergeGroup := .operator 86011 85852
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 86011) (leftOrdinal := 14)
    (rightResult := 85852) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7226⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71437⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge86018

namespace LeftMerge86019
def owner : Owner := ⟨.program ⟨257⟩, ⟨71438⟩⟩
def mergeEvent : Nat := 86019
def frameStart : Nat := 85336
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩] } }
def leftRaw : List Term := Proof.Events335.exact86011RawTerms
def rightRaw : List Term := Proof.Events335.exact85852RawTerms
def group : MergeGroup := .operator 86011 85852
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 86011) (leftOrdinal := 13)
    (rightResult := 85852) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7224⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71437⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge86019

namespace LeftMerge86020
def owner : Owner := ⟨.program ⟨257⟩, ⟨71438⟩⟩
def mergeEvent : Nat := 86020
def frameStart : Nat := 85336
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7222⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩] } }
def leftRaw : List Term := Proof.Events335.exact86011RawTerms
def rightRaw : List Term := Proof.Events335.exact85852RawTerms
def group : MergeGroup := .operator 86011 85852
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 86011) (leftOrdinal := 12)
    (rightResult := 85852) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7222⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71437⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7222⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge86020

namespace LeftMerge86021
def owner : Owner := ⟨.program ⟨257⟩, ⟨71438⟩⟩
def mergeEvent : Nat := 86021
def frameStart : Nat := 85336
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7220⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩] } }
def leftRaw : List Term := Proof.Events335.exact86011RawTerms
def rightRaw : List Term := Proof.Events335.exact85852RawTerms
def group : MergeGroup := .operator 86011 85852
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 86011) (leftOrdinal := 11)
    (rightResult := 85852) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7220⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71437⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7220⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge86021

namespace LeftMerge86022
def owner : Owner := ⟨.program ⟨257⟩, ⟨71438⟩⟩
def mergeEvent : Nat := 86022
def frameStart : Nat := 85336
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7218⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩] } }
def leftRaw : List Term := Proof.Events335.exact86011RawTerms
def rightRaw : List Term := Proof.Events335.exact85852RawTerms
def group : MergeGroup := .operator 86011 85852
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 86011) (leftOrdinal := 10)
    (rightResult := 85852) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7218⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71437⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7218⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge86022

namespace LeftMerge86023
def owner : Owner := ⟨.program ⟨257⟩, ⟨71438⟩⟩
def mergeEvent : Nat := 86023
def frameStart : Nat := 85336
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7216⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩] } }
def leftRaw : List Term := Proof.Events335.exact86011RawTerms
def rightRaw : List Term := Proof.Events335.exact85852RawTerms
def group : MergeGroup := .operator 86011 85852
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 86011) (leftOrdinal := 9)
    (rightResult := 85852) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7216⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71437⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7216⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge86023

namespace LeftMerge86024
def owner : Owner := ⟨.program ⟨257⟩, ⟨71438⟩⟩
def mergeEvent : Nat := 86024
def frameStart : Nat := 85336
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7214⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩] } }
def leftRaw : List Term := Proof.Events335.exact86011RawTerms
def rightRaw : List Term := Proof.Events335.exact85852RawTerms
def group : MergeGroup := .operator 86011 85852
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 86011) (leftOrdinal := 8)
    (rightResult := 85852) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7214⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71437⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7214⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge86024

namespace LeftMerge86025
def owner : Owner := ⟨.program ⟨257⟩, ⟨71438⟩⟩
def mergeEvent : Nat := 86025
def frameStart : Nat := 85336
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩] } }
def leftRaw : List Term := Proof.Events335.exact86011RawTerms
def rightRaw : List Term := Proof.Events335.exact85852RawTerms
def group : MergeGroup := .operator 86011 85852
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 86011) (leftOrdinal := 7)
    (rightResult := 85852) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7212⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71437⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge86025

namespace LeftMerge86026
def owner : Owner := ⟨.program ⟨257⟩, ⟨71438⟩⟩
def mergeEvent : Nat := 86026
def frameStart : Nat := 85336
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩] } }
def leftRaw : List Term := Proof.Events335.exact86011RawTerms
def rightRaw : List Term := Proof.Events335.exact85852RawTerms
def group : MergeGroup := .operator 86011 85852
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 86011) (leftOrdinal := 6)
    (rightResult := 85852) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7210⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71437⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge86026

namespace LeftMerge86027
def owner : Owner := ⟨.program ⟨257⟩, ⟨71438⟩⟩
def mergeEvent : Nat := 86027
def frameStart : Nat := 85336
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩] } }
def leftRaw : List Term := Proof.Events335.exact86011RawTerms
def rightRaw : List Term := Proof.Events335.exact85852RawTerms
def group : MergeGroup := .operator 86011 85852
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 86011) (leftOrdinal := 5)
    (rightResult := 85852) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7208⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71437⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge86027

namespace LeftMerge86028
def owner : Owner := ⟨.program ⟨257⟩, ⟨71438⟩⟩
def mergeEvent : Nat := 86028
def frameStart : Nat := 85336
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩] } }
def leftRaw : List Term := Proof.Events335.exact86011RawTerms
def rightRaw : List Term := Proof.Events335.exact85852RawTerms
def group : MergeGroup := .operator 86011 85852
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 86011) (leftOrdinal := 4)
    (rightResult := 85852) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7206⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71437⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge86028

namespace LeftMerge86029
def owner : Owner := ⟨.program ⟨257⟩, ⟨71438⟩⟩
def mergeEvent : Nat := 86029
def frameStart : Nat := 85336
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩] } }
def leftRaw : List Term := Proof.Events335.exact86011RawTerms
def rightRaw : List Term := Proof.Events335.exact85852RawTerms
def group : MergeGroup := .operator 86011 85852
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 86011) (leftOrdinal := 3)
    (rightResult := 85852) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7204⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71437⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge86029

namespace LeftMerge86030
def owner : Owner := ⟨.program ⟨257⟩, ⟨71438⟩⟩
def mergeEvent : Nat := 86030
def frameStart : Nat := 85336
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7202⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩] } }
def leftRaw : List Term := Proof.Events335.exact86011RawTerms
def rightRaw : List Term := Proof.Events335.exact85852RawTerms
def group : MergeGroup := .operator 86011 85852
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 86011) (leftOrdinal := 2)
    (rightResult := 85852) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7202⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71437⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge86030

namespace LeftMerge86031
def owner : Owner := ⟨.program ⟨257⟩, ⟨71438⟩⟩
def mergeEvent : Nat := 86031
def frameStart : Nat := 85336
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩] } }
def leftRaw : List Term := Proof.Events335.exact86011RawTerms
def rightRaw : List Term := Proof.Events335.exact85852RawTerms
def group : MergeGroup := .operator 86011 85852
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 86011) (leftOrdinal := 1)
    (rightResult := 85852) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7200⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71437⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge86031

namespace LeftMerge86032
def owner : Owner := ⟨.program ⟨257⟩, ⟨71438⟩⟩
def mergeEvent : Nat := 86032
def frameStart : Nat := 85336
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩] } }
def leftRaw : List Term := Proof.Events335.exact86011RawTerms
def rightRaw : List Term := Proof.Events335.exact85852RawTerms
def group : MergeGroup := .operator 86011 85852
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 86011) (leftOrdinal := 0)
    (rightResult := 85852) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7198⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71437⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge86032

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
