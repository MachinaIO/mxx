import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge71530
def owner : Owner := ⟨.program ⟨257⟩, ⟨71472⟩⟩
def mergeEvent : Nat := 71530
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩] } }
def leftRaw : List Term := Proof.Events279.exact71524RawTerms
def rightRaw : List Term := Proof.Events273.exact70108RawTerms
def group : MergeGroup := .operator 71524 70108
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 71524) (leftOrdinal := 16)
    (rightResult := 70108) (rightOrdinal := 16) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge71530

namespace LeftMerge71531
def owner : Owner := ⟨.program ⟨257⟩, ⟨71472⟩⟩
def mergeEvent : Nat := 71531
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨45774⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68872⟩⟩] } }
def leftRaw : List Term := Proof.Events279.exact71524RawTerms
def rightRaw : List Term := Proof.Events273.exact70108RawTerms
def group : MergeGroup := .operator 71524 70108
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 71524) (leftOrdinal := 29)
    (rightResult := 70108) (rightOrdinal := 28) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨45774⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68872⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨45774⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68872⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨45774⟩⟩], [⟨.program ⟨257⟩, ⟨68872⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge71531

namespace LeftMerge71532
def owner : Owner := ⟨.program ⟨257⟩, ⟨71472⟩⟩
def mergeEvent : Nat := 71532
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩] } }
def leftRaw : List Term := Proof.Events279.exact71524RawTerms
def rightRaw : List Term := Proof.Events273.exact70108RawTerms
def group : MergeGroup := .operator 71524 70108
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 71524) (leftOrdinal := 15)
    (rightResult := 70108) (rightOrdinal := 15) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge71532

namespace LeftMerge71533
def owner : Owner := ⟨.program ⟨257⟩, ⟨71472⟩⟩
def mergeEvent : Nat := 71533
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨43090⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68872⟩⟩] } }
def leftRaw : List Term := Proof.Events279.exact71524RawTerms
def rightRaw : List Term := Proof.Events273.exact70108RawTerms
def group : MergeGroup := .operator 71524 70108
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 71524) (leftOrdinal := 28)
    (rightResult := 70108) (rightOrdinal := 27) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨43090⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68872⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨43090⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68872⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨43090⟩⟩], [⟨.program ⟨257⟩, ⟨68872⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge71533

namespace LeftMerge71534
def owner : Owner := ⟨.program ⟨257⟩, ⟨71472⟩⟩
def mergeEvent : Nat := 71534
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩] } }
def leftRaw : List Term := Proof.Events279.exact71524RawTerms
def rightRaw : List Term := Proof.Events273.exact70108RawTerms
def group : MergeGroup := .operator 71524 70108
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 71524) (leftOrdinal := 14)
    (rightResult := 70108) (rightOrdinal := 14) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge71534

namespace LeftMerge71535
def owner : Owner := ⟨.program ⟨257⟩, ⟨71472⟩⟩
def mergeEvent : Nat := 71535
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨40410⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68872⟩⟩] } }
def leftRaw : List Term := Proof.Events279.exact71524RawTerms
def rightRaw : List Term := Proof.Events273.exact70108RawTerms
def group : MergeGroup := .operator 71524 70108
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 71524) (leftOrdinal := 27)
    (rightResult := 70108) (rightOrdinal := 26) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨40410⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68872⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨40410⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68872⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨40410⟩⟩], [⟨.program ⟨257⟩, ⟨68872⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge71535

namespace LeftMerge71536
def owner : Owner := ⟨.program ⟨257⟩, ⟨71472⟩⟩
def mergeEvent : Nat := 71536
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩] } }
def leftRaw : List Term := Proof.Events279.exact71524RawTerms
def rightRaw : List Term := Proof.Events273.exact70108RawTerms
def group : MergeGroup := .operator 71524 70108
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 71524) (leftOrdinal := 13)
    (rightResult := 70108) (rightOrdinal := 13) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge71536

namespace LeftMerge71537
def owner : Owner := ⟨.program ⟨257⟩, ⟨71472⟩⟩
def mergeEvent : Nat := 71537
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨37734⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68872⟩⟩] } }
def leftRaw : List Term := Proof.Events279.exact71524RawTerms
def rightRaw : List Term := Proof.Events273.exact70108RawTerms
def group : MergeGroup := .operator 71524 70108
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 71524) (leftOrdinal := 26)
    (rightResult := 70108) (rightOrdinal := 25) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨37734⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68872⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨37734⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68872⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨37734⟩⟩], [⟨.program ⟨257⟩, ⟨68872⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge71537

namespace LeftMerge71538
def owner : Owner := ⟨.program ⟨257⟩, ⟨71472⟩⟩
def mergeEvent : Nat := 71538
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7222⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩] } }
def leftRaw : List Term := Proof.Events279.exact71524RawTerms
def rightRaw : List Term := Proof.Events273.exact70108RawTerms
def group : MergeGroup := .operator 71524 70108
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 71524) (leftOrdinal := 12)
    (rightResult := 70108) (rightOrdinal := 12) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7222⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7222⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7222⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge71538

namespace LeftMerge71539
def owner : Owner := ⟨.program ⟨257⟩, ⟨71472⟩⟩
def mergeEvent : Nat := 71539
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨35054⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68872⟩⟩] } }
def leftRaw : List Term := Proof.Events279.exact71524RawTerms
def rightRaw : List Term := Proof.Events273.exact70108RawTerms
def group : MergeGroup := .operator 71524 70108
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 71524) (leftOrdinal := 25)
    (rightResult := 70108) (rightOrdinal := 24) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨35054⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68872⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨35054⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68872⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨35054⟩⟩], [⟨.program ⟨257⟩, ⟨68872⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge71539

namespace LeftMerge71540
def owner : Owner := ⟨.program ⟨257⟩, ⟨71472⟩⟩
def mergeEvent : Nat := 71540
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7220⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩] } }
def leftRaw : List Term := Proof.Events279.exact71524RawTerms
def rightRaw : List Term := Proof.Events273.exact70108RawTerms
def group : MergeGroup := .operator 71524 70108
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 71524) (leftOrdinal := 11)
    (rightResult := 70108) (rightOrdinal := 11) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7220⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7220⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge71540

namespace LeftMerge71541
def owner : Owner := ⟨.program ⟨257⟩, ⟨71472⟩⟩
def mergeEvent : Nat := 71541
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨29390⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68872⟩⟩] } }
def leftRaw : List Term := Proof.Events279.exact71524RawTerms
def rightRaw : List Term := Proof.Events273.exact70108RawTerms
def group : MergeGroup := .operator 71524 70108
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 71524) (leftOrdinal := 23)
    (rightResult := 70108) (rightOrdinal := 22) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨29390⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68872⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨29390⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68872⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨29390⟩⟩], [⟨.program ⟨257⟩, ⟨68872⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge71541

namespace LeftMerge71542
def owner : Owner := ⟨.program ⟨257⟩, ⟨71472⟩⟩
def mergeEvent : Nat := 71542
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7218⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩] } }
def leftRaw : List Term := Proof.Events279.exact71524RawTerms
def rightRaw : List Term := Proof.Events273.exact70108RawTerms
def group : MergeGroup := .operator 71524 70108
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 71524) (leftOrdinal := 10)
    (rightResult := 70108) (rightOrdinal := 10) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7218⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7218⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge71542

namespace LeftMerge71543
def owner : Owner := ⟨.program ⟨257⟩, ⟨71472⟩⟩
def mergeEvent : Nat := 71543
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨26710⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68872⟩⟩] } }
def leftRaw : List Term := Proof.Events279.exact71524RawTerms
def rightRaw : List Term := Proof.Events273.exact70108RawTerms
def group : MergeGroup := .operator 71524 70108
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 71524) (leftOrdinal := 22)
    (rightResult := 70108) (rightOrdinal := 21) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨26710⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68872⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨26710⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68872⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨26710⟩⟩], [⟨.program ⟨257⟩, ⟨68872⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge71543

namespace LeftMerge71544
def owner : Owner := ⟨.program ⟨257⟩, ⟨71472⟩⟩
def mergeEvent : Nat := 71544
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7216⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩] } }
def leftRaw : List Term := Proof.Events279.exact71524RawTerms
def rightRaw : List Term := Proof.Events273.exact70108RawTerms
def group : MergeGroup := .operator 71524 70108
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 71524) (leftOrdinal := 9)
    (rightResult := 70108) (rightOrdinal := 9) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7216⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7216⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge71544

namespace LeftMerge71545
def owner : Owner := ⟨.program ⟨257⟩, ⟨71472⟩⟩
def mergeEvent : Nat := 71545
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨67091⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68872⟩⟩] } }
def leftRaw : List Term := Proof.Events279.exact71524RawTerms
def rightRaw : List Term := Proof.Events273.exact70108RawTerms
def group : MergeGroup := .operator 71524 70108
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 71524) (leftOrdinal := 36)
    (rightResult := 70108) (rightOrdinal := 35) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨67091⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68872⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨67091⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68872⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨67091⟩⟩], [⟨.program ⟨257⟩, ⟨68872⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge71545

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
