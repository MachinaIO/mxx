import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge101540
def owner : Owner := ⟨.program ⟨214⟩, ⟨14945⟩⟩
def mergeEvent : Nat := 101540
def frameStart : Nat := 101449
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨14943⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events396.exact101493RawTerms
def rightRaw : List Term := Proof.Events396.exact101536RawTerms
def group : MergeGroup := .operator 101493 101536
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 101493) (leftOrdinal := 0)
    (rightResult := 101536) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨14943⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨14943⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge101540

namespace LeftMerge101557
def owner : Owner := ⟨.program ⟨214⟩, ⟨19088⟩⟩
def mergeEvent : Nat := 101557
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6691⟩⟩] } }
def rhsRaw : List Term := Proof.Events396.exact101554RawTerms
def group : MergeGroup := .relation 101556
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 101556) (rhsResult := 101554)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19085⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 101555 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19085⟩⟩]⟩) (none) 101554) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6691⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6691⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge101557

namespace LeftMerge101558
def owner : Owner := ⟨.program ⟨214⟩, ⟨19088⟩⟩
def mergeEvent : Nat := 101558
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6782⟩⟩, ⟨.program ⟨214⟩, ⟨7834⟩⟩, ⟨.program ⟨214⟩, ⟨24975⟩⟩] } }
def rhsRaw : List Term := Proof.Events396.exact101554RawTerms
def group : MergeGroup := .relation 101556
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 101556) (rhsResult := 101554)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19085⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 101555 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19085⟩⟩]⟩) (none) 101554) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6782⟩⟩, ⟨.program ⟨214⟩, ⟨7834⟩⟩, ⟨.program ⟨214⟩, ⟨24975⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6782⟩⟩, ⟨.program ⟨214⟩, ⟨7834⟩⟩, ⟨.program ⟨214⟩, ⟨24975⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge101558

namespace LeftMerge101559
def owner : Owner := ⟨.program ⟨214⟩, ⟨19088⟩⟩
def mergeEvent : Nat := 101559
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨9490⟩⟩, ⟨.program ⟨214⟩, ⟨10652⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨22990⟩⟩] } }
def rhsRaw : List Term := Proof.Events396.exact101554RawTerms
def group : MergeGroup := .relation 101556
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 101556) (rhsResult := 101554)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19085⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 101555 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19085⟩⟩]⟩) (none) 101554) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨9490⟩⟩, ⟨.program ⟨214⟩, ⟨10652⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨22990⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨9490⟩⟩, ⟨.program ⟨214⟩, ⟨10652⟩⟩], [⟨.program ⟨214⟩, ⟨22990⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge101559

namespace LeftMerge101560
def owner : Owner := ⟨.program ⟨214⟩, ⟨19088⟩⟩
def mergeEvent : Nat := 101560
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨14943⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def rhsRaw : List Term := Proof.Events396.exact101554RawTerms
def group : MergeGroup := .relation 101556
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 101556) (rhsResult := 101554)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19085⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 101555 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19085⟩⟩]⟩) (none) 101554) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨14943⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨14943⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge101560

namespace LeftMerge101565
def owner : Owner := ⟨.program ⟨214⟩, ⟨24977⟩⟩
def mergeEvent : Nat := 101565
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨9490⟩⟩, ⟨.program ⟨214⟩, ⟨10652⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨22990⟩⟩] } }
def leftRaw : List Term := Proof.Events396.exact101561RawTerms
def rightRaw : List Term := Proof.Events396.exact101399RawTerms
def group : MergeGroup := .operator 101561 101399
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 101561) (leftOrdinal := 2)
    (rightResult := 101399) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨9490⟩⟩, ⟨.program ⟨214⟩, ⟨10652⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨22990⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨9490⟩⟩, ⟨.program ⟨214⟩, ⟨10652⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨22990⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨9490⟩⟩, ⟨.program ⟨214⟩, ⟨10652⟩⟩], [⟨.program ⟨214⟩, ⟨22990⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge101565

namespace LeftMerge101566
def owner : Owner := ⟨.program ⟨214⟩, ⟨24977⟩⟩
def mergeEvent : Nat := 101566
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6782⟩⟩, ⟨.program ⟨214⟩, ⟨7834⟩⟩, ⟨.program ⟨214⟩, ⟨24975⟩⟩] } }
def leftRaw : List Term := Proof.Events396.exact101561RawTerms
def rightRaw : List Term := Proof.Events396.exact101399RawTerms
def group : MergeGroup := .operator 101561 101399
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 101561) (leftOrdinal := 1)
    (rightResult := 101399) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6782⟩⟩, ⟨.program ⟨214⟩, ⟨7834⟩⟩, ⟨.program ⟨214⟩, ⟨24975⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6782⟩⟩, ⟨.program ⟨214⟩, ⟨7834⟩⟩, ⟨.program ⟨214⟩, ⟨24975⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6782⟩⟩, ⟨.program ⟨214⟩, ⟨7834⟩⟩, ⟨.program ⟨214⟩, ⟨24975⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge101566

namespace LeftMerge101574
def owner : Owner := ⟨.program ⟨214⟩, ⟨26531⟩⟩
def mergeEvent : Nat := 101574
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6691⟩⟩, ⟨.program ⟨214⟩, ⟨26529⟩⟩] } }
def leftRaw : List Term := Proof.Events396.exact101568RawTerms
def rightRaw : List Term := Proof.Events395.exact101315RawTerms
def group : MergeGroup := .operator 101568 101315
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 101568) (leftOrdinal := 0)
    (rightResult := 101315) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6691⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨26529⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6691⟩⟩, ⟨.program ⟨214⟩, ⟨26529⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge101574

namespace LeftMerge101575
def owner : Owner := ⟨.program ⟨214⟩, ⟨26531⟩⟩
def mergeEvent : Nat := 101575
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨14943⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26529⟩⟩] } }
def leftRaw : List Term := Proof.Events396.exact101568RawTerms
def rightRaw : List Term := Proof.Events395.exact101315RawTerms
def group : MergeGroup := .operator 101568 101315
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 101568) (leftOrdinal := 1)
    (rightResult := 101315) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨14943⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨26529⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨14943⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26529⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge101575

namespace LeftMerge101577
def owner : Owner := ⟨.program ⟨214⟩, ⟨26531⟩⟩
def mergeEvent : Nat := 101577
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨14943⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23775⟩⟩] } }
def rhsRaw : List Term := Proof.Events395.exact101312RawTerms
def group : MergeGroup := .relation 101576
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 101576) (rhsResult := 101312)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨14943⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26529⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨26529⟩⟩) ⟨23775⟩ 101312) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23775⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨14943⟩⟩], [⟨.program ⟨214⟩, ⟨23775⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge101577

namespace LeftMerge101591
def owner : Owner := ⟨.program ⟨214⟩, ⟨20528⟩⟩
def mergeEvent : Nat := 101591
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20525⟩⟩] } }
def leftRaw : List Term := Proof.Events368.exact94462RawTerms
def rightRaw : List Term := Proof.Events396.exact101585RawTerms
def group : MergeGroup := .operator 94462 101585
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 94462) (leftOrdinal := 0)
    (rightResult := 101585) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨20525⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20525⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge101591

namespace LeftMerge101688
def owner : Owner := ⟨.program ⟨214⟩, ⟨14987⟩⟩
def mergeEvent : Nat := 101688
def frameStart : Nat := 101634
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨14943⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events397.exact101684RawTerms
def rightRaw : List Term := Proof.Events397.exact101682RawTerms
def group : MergeGroup := .operator 101684 101682
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 101684) (leftOrdinal := 0)
    (rightResult := 101682) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨14943⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨14943⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge101688

namespace LeftMerge101700
def owner : Owner := ⟨.program ⟨214⟩, ⟨26530⟩⟩
def mergeEvent : Nat := 101700
def frameStart : Nat := 101634
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6691⟩⟩, ⟨.program ⟨214⟩, ⟨26529⟩⟩] } }
def leftRaw : List Term := Proof.Events397.exact101696RawTerms
def rightRaw : List Term := Proof.Events397.exact101673RawTerms
def group : MergeGroup := .operator 101696 101673
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 101696) (leftOrdinal := 0)
    (rightResult := 101673) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6691⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨26529⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6691⟩⟩, ⟨.program ⟨214⟩, ⟨26529⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge101700

namespace LeftMerge101701
def owner : Owner := ⟨.program ⟨214⟩, ⟨26530⟩⟩
def mergeEvent : Nat := 101701
def frameStart : Nat := 101634
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨14943⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26529⟩⟩] } }
def leftRaw : List Term := Proof.Events397.exact101696RawTerms
def rightRaw : List Term := Proof.Events397.exact101673RawTerms
def group : MergeGroup := .operator 101696 101673
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 101696) (leftOrdinal := 1)
    (rightResult := 101673) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨14943⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨26529⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨14943⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26529⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge101701

namespace LeftMerge101703
def owner : Owner := ⟨.program ⟨214⟩, ⟨26530⟩⟩
def mergeEvent : Nat := 101703
def frameStart : Nat := 101634
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨14943⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23775⟩⟩] } }
def rhsRaw : List Term := Proof.Events397.exact101670RawTerms
def group : MergeGroup := .relation 101702
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 101702) (rhsResult := 101670)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨14943⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26529⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨26529⟩⟩) ⟨23775⟩ 101670) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23775⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨14943⟩⟩], [⟨.program ⟨214⟩, ⟨23775⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge101703

namespace LeftMerge101711
def owner : Owner := ⟨.program ⟨214⟩, ⟨15302⟩⟩
def mergeEvent : Nat := 101711
def frameStart : Nat := 101634
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15300⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events397.exact101684RawTerms
def rightRaw : List Term := Proof.Events397.exact101707RawTerms
def group : MergeGroup := .operator 101684 101707
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 101684) (leftOrdinal := 0)
    (rightResult := 101707) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15300⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨15300⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge101711

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
