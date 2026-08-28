import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge89861
def owner : Owner := ⟨.program ⟨214⟩, ⟨18649⟩⟩
def mergeEvent : Nat := 89861
def frameStart : Nat := 89317
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15629⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events350.exact89844RawTerms
def rightRaw : List Term := Proof.Events350.exact89842RawTerms
def group : MergeGroup := .operator 89844 89842
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 89844) (leftOrdinal := 0)
    (rightResult := 89842) (rightOrdinal := 3) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15629⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨15629⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge89861

namespace LeftMerge89862
def owner : Owner := ⟨.program ⟨214⟩, ⟨18649⟩⟩
def mergeEvent : Nat := 89862
def frameStart : Nat := 89317
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨17327⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events350.exact89844RawTerms
def rightRaw : List Term := Proof.Events350.exact89842RawTerms
def group : MergeGroup := .operator 89844 89842
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 89844) (leftOrdinal := 0)
    (rightResult := 89842) (rightOrdinal := 13) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17327⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨17327⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge89862

namespace LeftMerge89863
def owner : Owner := ⟨.program ⟨214⟩, ⟨18649⟩⟩
def mergeEvent : Nat := 89863
def frameStart : Nat := 89317
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15366⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events350.exact89844RawTerms
def rightRaw : List Term := Proof.Events350.exact89842RawTerms
def group : MergeGroup := .operator 89844 89842
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 89844) (leftOrdinal := 0)
    (rightResult := 89842) (rightOrdinal := 2) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15366⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨15366⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge89863

namespace LeftMerge89864
def owner : Owner := ⟨.program ⟨214⟩, ⟨18649⟩⟩
def mergeEvent : Nat := 89864
def frameStart : Nat := 89317
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15310⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events350.exact89844RawTerms
def rightRaw : List Term := Proof.Events350.exact89842RawTerms
def group : MergeGroup := .operator 89844 89842
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 89844) (leftOrdinal := 0)
    (rightResult := 89842) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15310⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨15310⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge89864

namespace LeftMerge89865
def owner : Owner := ⟨.program ⟨214⟩, ⟨18649⟩⟩
def mergeEvent : Nat := 89865
def frameStart : Nat := 89317
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15265⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events350.exact89844RawTerms
def rightRaw : List Term := Proof.Events350.exact89842RawTerms
def group : MergeGroup := .operator 89844 89842
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 89844) (leftOrdinal := 0)
    (rightResult := 89842) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15265⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨15265⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge89865

namespace LeftMerge89996
def owner : Owner := ⟨.program ⟨214⟩, ⟨18682⟩⟩
def mergeEvent : Nat := 89996
def frameStart : Nat := 89317
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6743⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩] } }
def leftRaw : List Term := Proof.Events351.exact89992RawTerms
def rightRaw : List Term := Proof.Events350.exact89833RawTerms
def group : MergeGroup := .operator 89992 89833
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 89992) (leftOrdinal := 17)
    (rightResult := 89833) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6743⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18681⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6743⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge89996

namespace LeftMerge89997
def owner : Owner := ⟨.program ⟨214⟩, ⟨18682⟩⟩
def mergeEvent : Nat := 89997
def frameStart : Nat := 89317
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6741⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩] } }
def leftRaw : List Term := Proof.Events351.exact89992RawTerms
def rightRaw : List Term := Proof.Events350.exact89833RawTerms
def group : MergeGroup := .operator 89992 89833
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 89992) (leftOrdinal := 16)
    (rightResult := 89833) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6741⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18681⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6741⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge89997

namespace LeftMerge89998
def owner : Owner := ⟨.program ⟨214⟩, ⟨18682⟩⟩
def mergeEvent : Nat := 89998
def frameStart : Nat := 89317
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6739⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩] } }
def leftRaw : List Term := Proof.Events351.exact89992RawTerms
def rightRaw : List Term := Proof.Events350.exact89833RawTerms
def group : MergeGroup := .operator 89992 89833
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 89992) (leftOrdinal := 15)
    (rightResult := 89833) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6739⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18681⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6739⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge89998

namespace LeftMerge89999
def owner : Owner := ⟨.program ⟨214⟩, ⟨18682⟩⟩
def mergeEvent : Nat := 89999
def frameStart : Nat := 89317
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6737⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩] } }
def leftRaw : List Term := Proof.Events351.exact89992RawTerms
def rightRaw : List Term := Proof.Events350.exact89833RawTerms
def group : MergeGroup := .operator 89992 89833
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 89992) (leftOrdinal := 14)
    (rightResult := 89833) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6737⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18681⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6737⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge89999

namespace LeftMerge90000
def owner : Owner := ⟨.program ⟨214⟩, ⟨18682⟩⟩
def mergeEvent : Nat := 90000
def frameStart : Nat := 89317
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6735⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩] } }
def leftRaw : List Term := Proof.Events351.exact89992RawTerms
def rightRaw : List Term := Proof.Events350.exact89833RawTerms
def group : MergeGroup := .operator 89992 89833
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 89992) (leftOrdinal := 13)
    (rightResult := 89833) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6735⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18681⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6735⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge90000

namespace LeftMerge90001
def owner : Owner := ⟨.program ⟨214⟩, ⟨18682⟩⟩
def mergeEvent : Nat := 90001
def frameStart : Nat := 89317
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6733⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩] } }
def leftRaw : List Term := Proof.Events351.exact89992RawTerms
def rightRaw : List Term := Proof.Events350.exact89833RawTerms
def group : MergeGroup := .operator 89992 89833
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 89992) (leftOrdinal := 12)
    (rightResult := 89833) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6733⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18681⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6733⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge90001

namespace LeftMerge90002
def owner : Owner := ⟨.program ⟨214⟩, ⟨18682⟩⟩
def mergeEvent : Nat := 90002
def frameStart : Nat := 89317
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6731⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩] } }
def leftRaw : List Term := Proof.Events351.exact89992RawTerms
def rightRaw : List Term := Proof.Events350.exact89833RawTerms
def group : MergeGroup := .operator 89992 89833
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 89992) (leftOrdinal := 11)
    (rightResult := 89833) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6731⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18681⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6731⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge90002

namespace LeftMerge90003
def owner : Owner := ⟨.program ⟨214⟩, ⟨18682⟩⟩
def mergeEvent : Nat := 90003
def frameStart : Nat := 89317
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6729⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩] } }
def leftRaw : List Term := Proof.Events351.exact89992RawTerms
def rightRaw : List Term := Proof.Events350.exact89833RawTerms
def group : MergeGroup := .operator 89992 89833
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 89992) (leftOrdinal := 10)
    (rightResult := 89833) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6729⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18681⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6729⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge90003

namespace LeftMerge90004
def owner : Owner := ⟨.program ⟨214⟩, ⟨18682⟩⟩
def mergeEvent : Nat := 90004
def frameStart : Nat := 89317
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6727⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩] } }
def leftRaw : List Term := Proof.Events351.exact89992RawTerms
def rightRaw : List Term := Proof.Events350.exact89833RawTerms
def group : MergeGroup := .operator 89992 89833
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 89992) (leftOrdinal := 9)
    (rightResult := 89833) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6727⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18681⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6727⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge90004

namespace LeftMerge90005
def owner : Owner := ⟨.program ⟨214⟩, ⟨18682⟩⟩
def mergeEvent : Nat := 90005
def frameStart : Nat := 89317
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6725⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩] } }
def leftRaw : List Term := Proof.Events351.exact89992RawTerms
def rightRaw : List Term := Proof.Events350.exact89833RawTerms
def group : MergeGroup := .operator 89992 89833
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 89992) (leftOrdinal := 8)
    (rightResult := 89833) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6725⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18681⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6725⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge90005

namespace LeftMerge90006
def owner : Owner := ⟨.program ⟨214⟩, ⟨18682⟩⟩
def mergeEvent : Nat := 90006
def frameStart : Nat := 89317
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6723⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩] } }
def leftRaw : List Term := Proof.Events351.exact89992RawTerms
def rightRaw : List Term := Proof.Events350.exact89833RawTerms
def group : MergeGroup := .operator 89992 89833
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 89992) (leftOrdinal := 7)
    (rightResult := 89833) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6723⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18681⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6723⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge90006

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
