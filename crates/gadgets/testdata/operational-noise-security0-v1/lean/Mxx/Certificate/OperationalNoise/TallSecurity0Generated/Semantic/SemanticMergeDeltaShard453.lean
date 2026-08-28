import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge75269
def owner : Owner := ⟨.program ⟨214⟩, ⟨18645⟩⟩
def mergeEvent : Nat := 75269
def frameStart : Nat := 74728
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15983⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events293.exact75255RawTerms
def rightRaw : List Term := Proof.Events293.exact75253RawTerms
def group : MergeGroup := .operator 75255 75253
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 75255) (leftOrdinal := 0)
    (rightResult := 75253) (rightOrdinal := 6) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15983⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨15983⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge75269

namespace LeftMerge75270
def owner : Owner := ⟨.program ⟨214⟩, ⟨18645⟩⟩
def mergeEvent : Nat := 75270
def frameStart : Nat := 74728
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15864⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events293.exact75255RawTerms
def rightRaw : List Term := Proof.Events293.exact75253RawTerms
def group : MergeGroup := .operator 75255 75253
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 75255) (leftOrdinal := 0)
    (rightResult := 75253) (rightOrdinal := 5) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15864⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨15864⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge75270

namespace LeftMerge75271
def owner : Owner := ⟨.program ⟨214⟩, ⟨18645⟩⟩
def mergeEvent : Nat := 75271
def frameStart : Nat := 74728
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15745⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events293.exact75255RawTerms
def rightRaw : List Term := Proof.Events293.exact75253RawTerms
def group : MergeGroup := .operator 75255 75253
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 75255) (leftOrdinal := 0)
    (rightResult := 75253) (rightOrdinal := 4) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15745⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨15745⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge75271

namespace LeftMerge75272
def owner : Owner := ⟨.program ⟨214⟩, ⟨18645⟩⟩
def mergeEvent : Nat := 75272
def frameStart : Nat := 74728
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15626⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events293.exact75255RawTerms
def rightRaw : List Term := Proof.Events293.exact75253RawTerms
def group : MergeGroup := .operator 75255 75253
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 75255) (leftOrdinal := 0)
    (rightResult := 75253) (rightOrdinal := 3) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15626⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨15626⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge75272

namespace LeftMerge75273
def owner : Owner := ⟨.program ⟨214⟩, ⟨18645⟩⟩
def mergeEvent : Nat := 75273
def frameStart : Nat := 74728
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨17318⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events293.exact75255RawTerms
def rightRaw : List Term := Proof.Events293.exact75253RawTerms
def group : MergeGroup := .operator 75255 75253
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 75255) (leftOrdinal := 0)
    (rightResult := 75253) (rightOrdinal := 13) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17318⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨17318⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge75273

namespace LeftMerge75274
def owner : Owner := ⟨.program ⟨214⟩, ⟨18645⟩⟩
def mergeEvent : Nat := 75274
def frameStart : Nat := 74728
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15362⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events293.exact75255RawTerms
def rightRaw : List Term := Proof.Events293.exact75253RawTerms
def group : MergeGroup := .operator 75255 75253
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 75255) (leftOrdinal := 0)
    (rightResult := 75253) (rightOrdinal := 2) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15362⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨15362⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge75274

namespace LeftMerge75275
def owner : Owner := ⟨.program ⟨214⟩, ⟨18645⟩⟩
def mergeEvent : Nat := 75275
def frameStart : Nat := 74728
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15306⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events293.exact75255RawTerms
def rightRaw : List Term := Proof.Events293.exact75253RawTerms
def group : MergeGroup := .operator 75255 75253
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 75255) (leftOrdinal := 0)
    (rightResult := 75253) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15306⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨15306⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge75275

namespace LeftMerge75276
def owner : Owner := ⟨.program ⟨214⟩, ⟨18645⟩⟩
def mergeEvent : Nat := 75276
def frameStart : Nat := 74728
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15262⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events293.exact75255RawTerms
def rightRaw : List Term := Proof.Events293.exact75253RawTerms
def group : MergeGroup := .operator 75255 75253
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 75255) (leftOrdinal := 0)
    (rightResult := 75253) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15262⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨15262⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge75276

namespace LeftMerge75407
def owner : Owner := ⟨.program ⟨214⟩, ⟨18679⟩⟩
def mergeEvent : Nat := 75407
def frameStart : Nat := 74728
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6743⟩⟩, ⟨.program ⟨214⟩, ⟨18678⟩⟩] } }
def leftRaw : List Term := Proof.Events294.exact75403RawTerms
def rightRaw : List Term := Proof.Events293.exact75244RawTerms
def group : MergeGroup := .operator 75403 75244
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 75403) (leftOrdinal := 17)
    (rightResult := 75244) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6743⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18678⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6743⟩⟩, ⟨.program ⟨214⟩, ⟨18678⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge75407

namespace LeftMerge75408
def owner : Owner := ⟨.program ⟨214⟩, ⟨18679⟩⟩
def mergeEvent : Nat := 75408
def frameStart : Nat := 74728
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6741⟩⟩, ⟨.program ⟨214⟩, ⟨18678⟩⟩] } }
def leftRaw : List Term := Proof.Events294.exact75403RawTerms
def rightRaw : List Term := Proof.Events293.exact75244RawTerms
def group : MergeGroup := .operator 75403 75244
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 75403) (leftOrdinal := 16)
    (rightResult := 75244) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6741⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18678⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6741⟩⟩, ⟨.program ⟨214⟩, ⟨18678⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge75408

namespace LeftMerge75409
def owner : Owner := ⟨.program ⟨214⟩, ⟨18679⟩⟩
def mergeEvent : Nat := 75409
def frameStart : Nat := 74728
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6739⟩⟩, ⟨.program ⟨214⟩, ⟨18678⟩⟩] } }
def leftRaw : List Term := Proof.Events294.exact75403RawTerms
def rightRaw : List Term := Proof.Events293.exact75244RawTerms
def group : MergeGroup := .operator 75403 75244
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 75403) (leftOrdinal := 15)
    (rightResult := 75244) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6739⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18678⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6739⟩⟩, ⟨.program ⟨214⟩, ⟨18678⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge75409

namespace LeftMerge75410
def owner : Owner := ⟨.program ⟨214⟩, ⟨18679⟩⟩
def mergeEvent : Nat := 75410
def frameStart : Nat := 74728
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6737⟩⟩, ⟨.program ⟨214⟩, ⟨18678⟩⟩] } }
def leftRaw : List Term := Proof.Events294.exact75403RawTerms
def rightRaw : List Term := Proof.Events293.exact75244RawTerms
def group : MergeGroup := .operator 75403 75244
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 75403) (leftOrdinal := 14)
    (rightResult := 75244) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6737⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18678⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6737⟩⟩, ⟨.program ⟨214⟩, ⟨18678⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge75410

namespace LeftMerge75411
def owner : Owner := ⟨.program ⟨214⟩, ⟨18679⟩⟩
def mergeEvent : Nat := 75411
def frameStart : Nat := 74728
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6735⟩⟩, ⟨.program ⟨214⟩, ⟨18678⟩⟩] } }
def leftRaw : List Term := Proof.Events294.exact75403RawTerms
def rightRaw : List Term := Proof.Events293.exact75244RawTerms
def group : MergeGroup := .operator 75403 75244
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 75403) (leftOrdinal := 13)
    (rightResult := 75244) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6735⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18678⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6735⟩⟩, ⟨.program ⟨214⟩, ⟨18678⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge75411

namespace LeftMerge75412
def owner : Owner := ⟨.program ⟨214⟩, ⟨18679⟩⟩
def mergeEvent : Nat := 75412
def frameStart : Nat := 74728
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6733⟩⟩, ⟨.program ⟨214⟩, ⟨18678⟩⟩] } }
def leftRaw : List Term := Proof.Events294.exact75403RawTerms
def rightRaw : List Term := Proof.Events293.exact75244RawTerms
def group : MergeGroup := .operator 75403 75244
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 75403) (leftOrdinal := 12)
    (rightResult := 75244) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6733⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18678⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6733⟩⟩, ⟨.program ⟨214⟩, ⟨18678⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge75412

namespace LeftMerge75413
def owner : Owner := ⟨.program ⟨214⟩, ⟨18679⟩⟩
def mergeEvent : Nat := 75413
def frameStart : Nat := 74728
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6731⟩⟩, ⟨.program ⟨214⟩, ⟨18678⟩⟩] } }
def leftRaw : List Term := Proof.Events294.exact75403RawTerms
def rightRaw : List Term := Proof.Events293.exact75244RawTerms
def group : MergeGroup := .operator 75403 75244
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 75403) (leftOrdinal := 11)
    (rightResult := 75244) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6731⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18678⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6731⟩⟩, ⟨.program ⟨214⟩, ⟨18678⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge75413

namespace LeftMerge75414
def owner : Owner := ⟨.program ⟨214⟩, ⟨18679⟩⟩
def mergeEvent : Nat := 75414
def frameStart : Nat := 74728
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6729⟩⟩, ⟨.program ⟨214⟩, ⟨18678⟩⟩] } }
def leftRaw : List Term := Proof.Events294.exact75403RawTerms
def rightRaw : List Term := Proof.Events293.exact75244RawTerms
def group : MergeGroup := .operator 75403 75244
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 75403) (leftOrdinal := 10)
    (rightResult := 75244) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6729⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18678⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6729⟩⟩, ⟨.program ⟨214⟩, ⟨18678⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge75414

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
