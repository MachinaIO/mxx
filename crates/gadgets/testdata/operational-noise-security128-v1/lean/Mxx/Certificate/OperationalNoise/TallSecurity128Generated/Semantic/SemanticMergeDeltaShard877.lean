import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge144654
def owner : Owner := ⟨.program ⟨257⟩, ⟨71020⟩⟩
def mergeEvent : Nat := 144654
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨48272⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68788⟩⟩] } }
def leftRaw : List Term := Proof.Events565.exact144649RawTerms
def rightRaw : List Term := Proof.Events559.exact143233RawTerms
def group : MergeGroup := .operator 144649 143233
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 144649) (leftOrdinal := 30)
    (rightResult := 143233) (rightOrdinal := 29) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨48272⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68788⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨48272⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68788⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨48272⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge144654

namespace LeftMerge144655
def owner : Owner := ⟨.program ⟨257⟩, ⟨71020⟩⟩
def mergeEvent : Nat := 144655
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩] } }
def leftRaw : List Term := Proof.Events565.exact144649RawTerms
def rightRaw : List Term := Proof.Events559.exact143233RawTerms
def group : MergeGroup := .operator 144649 143233
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 144649) (leftOrdinal := 16)
    (rightResult := 143233) (rightOrdinal := 16) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge144655

namespace LeftMerge144656
def owner : Owner := ⟨.program ⟨257⟩, ⟨71020⟩⟩
def mergeEvent : Nat := 144656
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨45592⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68788⟩⟩] } }
def leftRaw : List Term := Proof.Events565.exact144649RawTerms
def rightRaw : List Term := Proof.Events559.exact143233RawTerms
def group : MergeGroup := .operator 144649 143233
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 144649) (leftOrdinal := 29)
    (rightResult := 143233) (rightOrdinal := 28) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨45592⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68788⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨45592⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68788⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨45592⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge144656

namespace LeftMerge144657
def owner : Owner := ⟨.program ⟨257⟩, ⟨71020⟩⟩
def mergeEvent : Nat := 144657
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩] } }
def leftRaw : List Term := Proof.Events565.exact144649RawTerms
def rightRaw : List Term := Proof.Events559.exact143233RawTerms
def group : MergeGroup := .operator 144649 143233
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 144649) (leftOrdinal := 15)
    (rightResult := 143233) (rightOrdinal := 15) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge144657

namespace LeftMerge144658
def owner : Owner := ⟨.program ⟨257⟩, ⟨71020⟩⟩
def mergeEvent : Nat := 144658
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨42908⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68788⟩⟩] } }
def leftRaw : List Term := Proof.Events565.exact144649RawTerms
def rightRaw : List Term := Proof.Events559.exact143233RawTerms
def group : MergeGroup := .operator 144649 143233
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 144649) (leftOrdinal := 28)
    (rightResult := 143233) (rightOrdinal := 27) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨42908⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68788⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨42908⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68788⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨42908⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge144658

namespace LeftMerge144659
def owner : Owner := ⟨.program ⟨257⟩, ⟨71020⟩⟩
def mergeEvent : Nat := 144659
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩] } }
def leftRaw : List Term := Proof.Events565.exact144649RawTerms
def rightRaw : List Term := Proof.Events559.exact143233RawTerms
def group : MergeGroup := .operator 144649 143233
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 144649) (leftOrdinal := 14)
    (rightResult := 143233) (rightOrdinal := 14) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge144659

namespace LeftMerge144660
def owner : Owner := ⟨.program ⟨257⟩, ⟨71020⟩⟩
def mergeEvent : Nat := 144660
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨40228⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68788⟩⟩] } }
def leftRaw : List Term := Proof.Events565.exact144649RawTerms
def rightRaw : List Term := Proof.Events559.exact143233RawTerms
def group : MergeGroup := .operator 144649 143233
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 144649) (leftOrdinal := 27)
    (rightResult := 143233) (rightOrdinal := 26) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨40228⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68788⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨40228⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68788⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨40228⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge144660

namespace LeftMerge144661
def owner : Owner := ⟨.program ⟨257⟩, ⟨71020⟩⟩
def mergeEvent : Nat := 144661
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩] } }
def leftRaw : List Term := Proof.Events565.exact144649RawTerms
def rightRaw : List Term := Proof.Events559.exact143233RawTerms
def group : MergeGroup := .operator 144649 143233
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 144649) (leftOrdinal := 13)
    (rightResult := 143233) (rightOrdinal := 13) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge144661

namespace LeftMerge144662
def owner : Owner := ⟨.program ⟨257⟩, ⟨71020⟩⟩
def mergeEvent : Nat := 144662
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨37552⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68788⟩⟩] } }
def leftRaw : List Term := Proof.Events565.exact144649RawTerms
def rightRaw : List Term := Proof.Events559.exact143233RawTerms
def group : MergeGroup := .operator 144649 143233
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 144649) (leftOrdinal := 26)
    (rightResult := 143233) (rightOrdinal := 25) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨37552⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68788⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨37552⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68788⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨37552⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge144662

namespace LeftMerge144663
def owner : Owner := ⟨.program ⟨257⟩, ⟨71020⟩⟩
def mergeEvent : Nat := 144663
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7222⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩] } }
def leftRaw : List Term := Proof.Events565.exact144649RawTerms
def rightRaw : List Term := Proof.Events559.exact143233RawTerms
def group : MergeGroup := .operator 144649 143233
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 144649) (leftOrdinal := 12)
    (rightResult := 143233) (rightOrdinal := 12) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7222⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7222⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7222⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge144663

namespace LeftMerge144664
def owner : Owner := ⟨.program ⟨257⟩, ⟨71020⟩⟩
def mergeEvent : Nat := 144664
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨34872⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68788⟩⟩] } }
def leftRaw : List Term := Proof.Events565.exact144649RawTerms
def rightRaw : List Term := Proof.Events559.exact143233RawTerms
def group : MergeGroup := .operator 144649 143233
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 144649) (leftOrdinal := 25)
    (rightResult := 143233) (rightOrdinal := 24) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨34872⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68788⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨34872⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68788⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨34872⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge144664

namespace LeftMerge144665
def owner : Owner := ⟨.program ⟨257⟩, ⟨71020⟩⟩
def mergeEvent : Nat := 144665
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7220⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩] } }
def leftRaw : List Term := Proof.Events565.exact144649RawTerms
def rightRaw : List Term := Proof.Events559.exact143233RawTerms
def group : MergeGroup := .operator 144649 143233
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 144649) (leftOrdinal := 11)
    (rightResult := 143233) (rightOrdinal := 11) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7220⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7220⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge144665

namespace LeftMerge144666
def owner : Owner := ⟨.program ⟨257⟩, ⟨71020⟩⟩
def mergeEvent : Nat := 144666
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨29208⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68788⟩⟩] } }
def leftRaw : List Term := Proof.Events565.exact144649RawTerms
def rightRaw : List Term := Proof.Events559.exact143233RawTerms
def group : MergeGroup := .operator 144649 143233
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 144649) (leftOrdinal := 23)
    (rightResult := 143233) (rightOrdinal := 22) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨29208⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68788⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨29208⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68788⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨29208⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge144666

namespace LeftMerge144667
def owner : Owner := ⟨.program ⟨257⟩, ⟨71020⟩⟩
def mergeEvent : Nat := 144667
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7218⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩] } }
def leftRaw : List Term := Proof.Events565.exact144649RawTerms
def rightRaw : List Term := Proof.Events559.exact143233RawTerms
def group : MergeGroup := .operator 144649 143233
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 144649) (leftOrdinal := 10)
    (rightResult := 143233) (rightOrdinal := 10) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7218⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7218⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge144667

namespace LeftMerge144668
def owner : Owner := ⟨.program ⟨257⟩, ⟨71020⟩⟩
def mergeEvent : Nat := 144668
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨26528⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68788⟩⟩] } }
def leftRaw : List Term := Proof.Events565.exact144649RawTerms
def rightRaw : List Term := Proof.Events559.exact143233RawTerms
def group : MergeGroup := .operator 144649 143233
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 144649) (leftOrdinal := 22)
    (rightResult := 143233) (rightOrdinal := 21) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨26528⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68788⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨26528⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68788⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨26528⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge144668

namespace LeftMerge144669
def owner : Owner := ⟨.program ⟨257⟩, ⟨71020⟩⟩
def mergeEvent : Nat := 144669
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7216⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩] } }
def leftRaw : List Term := Proof.Events565.exact144649RawTerms
def rightRaw : List Term := Proof.Events559.exact143233RawTerms
def group : MergeGroup := .operator 144649 143233
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 144649) (leftOrdinal := 9)
    (rightResult := 143233) (rightOrdinal := 9) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7216⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7216⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge144669

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
