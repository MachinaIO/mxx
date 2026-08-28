import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge150704
def owner : Owner := ⟨.program ⟨257⟩, ⟨9558⟩⟩
def mergeEvent : Nat := 150704
def frameStart : Nat := 150621
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩] } }
def leftRaw : List Term := Proof.Events588.exact150700RawTerms
def rightRaw : List Term := Proof.Events588.exact150697RawTerms
def group : MergeGroup := .operator 150700 150697
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 150700) (leftOrdinal := 0)
    (rightResult := 150697) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9556⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge150704

namespace LeftMerge150713
def owner : Owner := ⟨.program ⟨257⟩, ⟨41589⟩⟩
def mergeEvent : Nat := 150713
def frameStart : Nat := 150621
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41586⟩⟩] } }
def leftRaw : List Term := Proof.Events588.exact150709RawTerms
def rightRaw : List Term := Proof.Events588.exact150666RawTerms
def group : MergeGroup := .operator 150709 150666
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 150709) (leftOrdinal := 0)
    (rightResult := 150666) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨41586⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41586⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge150713

namespace LeftMerge150714
def owner : Owner := ⟨.program ⟨257⟩, ⟨41589⟩⟩
def mergeEvent : Nat := 150714
def frameStart : Nat := 150621
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14136⟩⟩, ⟨.program ⟨257⟩, ⟨39722⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨41586⟩⟩] } }
def leftRaw : List Term := Proof.Events588.exact150709RawTerms
def rightRaw : List Term := Proof.Events588.exact150666RawTerms
def group : MergeGroup := .operator 150709 150666
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 150709) (leftOrdinal := 1)
    (rightResult := 150666) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14136⟩⟩, ⟨.program ⟨257⟩, ⟨39722⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨41586⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨14136⟩⟩, ⟨.program ⟨257⟩, ⟨39722⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨41586⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge150714

namespace LeftMerge150716
def owner : Owner := ⟨.program ⟨257⟩, ⟨41589⟩⟩
def mergeEvent : Nat := 150716
def frameStart : Nat := 150621
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14136⟩⟩, ⟨.program ⟨257⟩, ⟨39722⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41091⟩⟩] } }
def rhsRaw : List Term := Proof.Events588.exact150663RawTerms
def group : MergeGroup := .relation 150715
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 150715) (rhsResult := 150663)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨14136⟩⟩, ⟨.program ⟨257⟩, ⟨39722⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨41586⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨41586⟩⟩) ⟨41091⟩ 150663) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨41091⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨14136⟩⟩, ⟨.program ⟨257⟩, ⟨39722⟩⟩], [⟨.program ⟨257⟩, ⟨41091⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge150716

namespace LeftMerge150724
def owner : Owner := ⟨.program ⟨257⟩, ⟨40086⟩⟩
def mergeEvent : Nat := 150724
def frameStart : Nat := 150621
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨40084⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events588.exact150677RawTerms
def rightRaw : List Term := Proof.Events588.exact150720RawTerms
def group : MergeGroup := .operator 150677 150720
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 150677) (leftOrdinal := 0)
    (rightResult := 150720) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨40084⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨40084⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge150724

namespace LeftMerge150741
def owner : Owner := ⟨.program ⟨257⟩, ⟨40522⟩⟩
def mergeEvent : Nat := 150741
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7193⟩⟩] } }
def rhsRaw : List Term := Proof.Events588.exact150738RawTerms
def group : MergeGroup := .relation 150740
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 150740) (rhsResult := 150738)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40519⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 150739 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40519⟩⟩]⟩) (none) 150738) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7193⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7193⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge150741

namespace LeftMerge150742
def owner : Owner := ⟨.program ⟨257⟩, ⟨40522⟩⟩
def mergeEvent : Nat := 150742
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41586⟩⟩] } }
def rhsRaw : List Term := Proof.Events588.exact150738RawTerms
def group : MergeGroup := .relation 150740
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 150740) (rhsResult := 150738)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40519⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 150739 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40519⟩⟩]⟩) (none) 150738) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41586⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41586⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge150742

namespace LeftMerge150743
def owner : Owner := ⟨.program ⟨257⟩, ⟨40522⟩⟩
def mergeEvent : Nat := 150743
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨14136⟩⟩, ⟨.program ⟨257⟩, ⟨39722⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41091⟩⟩] } }
def rhsRaw : List Term := Proof.Events588.exact150738RawTerms
def group : MergeGroup := .relation 150740
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 150740) (rhsResult := 150738)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40519⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 150739 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40519⟩⟩]⟩) (none) 150738) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14136⟩⟩, ⟨.program ⟨257⟩, ⟨39722⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41091⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨14136⟩⟩, ⟨.program ⟨257⟩, ⟨39722⟩⟩], [⟨.program ⟨257⟩, ⟨41091⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge150743

namespace LeftMerge150744
def owner : Owner := ⟨.program ⟨257⟩, ⟨40522⟩⟩
def mergeEvent : Nat := 150744
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨40084⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events588.exact150738RawTerms
def group : MergeGroup := .relation 150740
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 150740) (rhsResult := 150738)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40519⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 150739 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40519⟩⟩]⟩) (none) 150738) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨40084⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨40084⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge150744

namespace LeftMerge150749
def owner : Owner := ⟨.program ⟨257⟩, ⟨41588⟩⟩
def mergeEvent : Nat := 150749
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨14136⟩⟩, ⟨.program ⟨257⟩, ⟨39722⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41091⟩⟩] } }
def leftRaw : List Term := Proof.Events588.exact150745RawTerms
def rightRaw : List Term := Proof.Events588.exact150559RawTerms
def group : MergeGroup := .operator 150745 150559
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 150745) (leftOrdinal := 2)
    (rightResult := 150559) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨14136⟩⟩, ⟨.program ⟨257⟩, ⟨39722⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41091⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨14136⟩⟩, ⟨.program ⟨257⟩, ⟨39722⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41091⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨14136⟩⟩, ⟨.program ⟨257⟩, ⟨39722⟩⟩], [⟨.program ⟨257⟩, ⟨41091⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge150749

namespace LeftMerge150750
def owner : Owner := ⟨.program ⟨257⟩, ⟨41588⟩⟩
def mergeEvent : Nat := 150750
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41586⟩⟩] } }
def leftRaw : List Term := Proof.Events588.exact150745RawTerms
def rightRaw : List Term := Proof.Events588.exact150559RawTerms
def group : MergeGroup := .operator 150745 150559
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 150745) (leftOrdinal := 1)
    (rightResult := 150559) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41586⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41586⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41586⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge150750

namespace LeftMerge150758
def owner : Owner := ⟨.program ⟨257⟩, ⟨41916⟩⟩
def mergeEvent : Nat := 150758
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨41914⟩⟩] } }
def leftRaw : List Term := Proof.Events588.exact150752RawTerms
def rightRaw : List Term := Proof.Events587.exact150475RawTerms
def group : MergeGroup := .operator 150752 150475
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 150752) (leftOrdinal := 0)
    (rightResult := 150475) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7193⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨41914⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨41914⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge150758

namespace LeftMerge150759
def owner : Owner := ⟨.program ⟨257⟩, ⟨41916⟩⟩
def mergeEvent : Nat := 150759
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨40084⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨41914⟩⟩] } }
def leftRaw : List Term := Proof.Events588.exact150752RawTerms
def rightRaw : List Term := Proof.Events587.exact150475RawTerms
def group : MergeGroup := .operator 150752 150475
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 150752) (leftOrdinal := 1)
    (rightResult := 150475) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨40084⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨41914⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨40084⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨41914⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge150759

namespace LeftMerge150761
def owner : Owner := ⟨.program ⟨257⟩, ⟨41916⟩⟩
def mergeEvent : Nat := 150761
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨40084⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41234⟩⟩] } }
def rhsRaw : List Term := Proof.Events587.exact150472RawTerms
def group : MergeGroup := .relation 150760
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 150760) (rhsResult := 150472)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨40084⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨41914⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨41914⟩⟩) ⟨41234⟩ 150472) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨41234⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨40084⟩⟩], [⟨.program ⟨257⟩, ⟨41234⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge150761

namespace LeftMerge150775
def owner : Owner := ⟨.program ⟨257⟩, ⟨40799⟩⟩
def mergeEvent : Nat := 150775
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40796⟩⟩] } }
def leftRaw : List Term := Proof.Events582.exact149120RawTerms
def rightRaw : List Term := Proof.Events588.exact150769RawTerms
def group : MergeGroup := .operator 149120 150769
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 149120) (leftOrdinal := 0)
    (rightResult := 150769) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨40796⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40796⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge150775

namespace LeftMerge150896
def owner : Owner := ⟨.program ⟨257⟩, ⟨41456⟩⟩
def mergeEvent : Nat := 150896
def frameStart : Nat := 150830
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨40084⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events589.exact150892RawTerms
def rightRaw : List Term := Proof.Events589.exact150890RawTerms
def group : MergeGroup := .operator 150892 150890
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 150892) (leftOrdinal := 0)
    (rightResult := 150890) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨40084⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨40084⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge150896

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
