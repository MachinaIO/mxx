import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge304189
def owner : Owner := ⟨.program ⟨257⟩, ⟨69049⟩⟩
def mergeEvent : Nat := 304189
def frameStart : Nat := 303660
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨59911⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1188.exact304175RawTerms
def rightRaw : List Term := Proof.Events1188.exact304173RawTerms
def group : MergeGroup := .operator 304175 304173
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 304175) (leftOrdinal := 0)
    (rightResult := 304173) (rightOrdinal := 15) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨59911⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨59911⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge304189

namespace LeftMerge304190
def owner : Owner := ⟨.program ⟨257⟩, ⟨69049⟩⟩
def mergeEvent : Nat := 304190
def frameStart : Nat := 303660
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨56931⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1188.exact304175RawTerms
def rightRaw : List Term := Proof.Events1188.exact304173RawTerms
def group : MergeGroup := .operator 304175 304173
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 304175) (leftOrdinal := 0)
    (rightResult := 304173) (rightOrdinal := 14) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨56931⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨56931⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge304190

namespace LeftMerge304191
def owner : Owner := ⟨.program ⟨257⟩, ⟨69049⟩⟩
def mergeEvent : Nat := 304191
def frameStart : Nat := 303660
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨53951⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1188.exact304175RawTerms
def rightRaw : List Term := Proof.Events1188.exact304173RawTerms
def group : MergeGroup := .operator 304175 304173
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 304175) (leftOrdinal := 0)
    (rightResult := 304173) (rightOrdinal := 13) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨53951⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨53951⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge304191

namespace LeftMerge304192
def owner : Owner := ⟨.program ⟨257⟩, ⟨69049⟩⟩
def mergeEvent : Nat := 304192
def frameStart : Nat := 303660
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨50971⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1188.exact304175RawTerms
def rightRaw : List Term := Proof.Events1188.exact304173RawTerms
def group : MergeGroup := .operator 304175 304173
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 304175) (leftOrdinal := 0)
    (rightResult := 304173) (rightOrdinal := 12) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨50971⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨50971⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge304192

namespace LeftMerge304193
def owner : Owner := ⟨.program ⟨257⟩, ⟨69049⟩⟩
def mergeEvent : Nat := 304193
def frameStart : Nat := 303660
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨31916⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1188.exact304175RawTerms
def rightRaw : List Term := Proof.Events1188.exact304173RawTerms
def group : MergeGroup := .operator 304175 304173
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 304175) (leftOrdinal := 0)
    (rightResult := 304173) (rightOrdinal := 5) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨31916⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨31916⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge304193

namespace LeftMerge304194
def owner : Owner := ⟨.program ⟨257⟩, ⟨69049⟩⟩
def mergeEvent : Nat := 304194
def frameStart : Nat := 303660
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨21896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1188.exact304175RawTerms
def rightRaw : List Term := Proof.Events1188.exact304173RawTerms
def group : MergeGroup := .operator 304175 304173
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 304175) (leftOrdinal := 0)
    (rightResult := 304173) (rightOrdinal := 2) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨21896⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨21896⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge304194

namespace LeftMerge304195
def owner : Owner := ⟨.program ⟨257⟩, ⟨69049⟩⟩
def mergeEvent : Nat := 304195
def frameStart : Nat := 303660
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨18676⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1188.exact304175RawTerms
def rightRaw : List Term := Proof.Events1188.exact304173RawTerms
def group : MergeGroup := .operator 304175 304173
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 304175) (leftOrdinal := 0)
    (rightResult := 304173) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨18676⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨18676⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge304195

namespace LeftMerge304196
def owner : Owner := ⟨.program ⟨257⟩, ⟨69049⟩⟩
def mergeEvent : Nat := 304196
def frameStart : Nat := 303660
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨15875⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1188.exact304175RawTerms
def rightRaw : List Term := Proof.Events1188.exact304173RawTerms
def group : MergeGroup := .operator 304175 304173
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 304175) (leftOrdinal := 0)
    (rightResult := 304173) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨15875⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨15875⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge304196

namespace LeftMerge304327
def owner : Owner := ⟨.program ⟨257⟩, ⟨70935⟩⟩
def mergeEvent : Nat := 304327
def frameStart : Nat := 303660
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7232⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩] } }
def leftRaw : List Term := Proof.Events1188.exact304323RawTerms
def rightRaw : List Term := Proof.Events1188.exact304164RawTerms
def group : MergeGroup := .operator 304323 304164
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 304323) (leftOrdinal := 17)
    (rightResult := 304164) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7232⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨70934⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7232⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge304327

namespace LeftMerge304328
def owner : Owner := ⟨.program ⟨257⟩, ⟨70935⟩⟩
def mergeEvent : Nat := 304328
def frameStart : Nat := 303660
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩] } }
def leftRaw : List Term := Proof.Events1188.exact304323RawTerms
def rightRaw : List Term := Proof.Events1188.exact304164RawTerms
def group : MergeGroup := .operator 304323 304164
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 304323) (leftOrdinal := 16)
    (rightResult := 304164) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7230⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨70934⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge304328

namespace LeftMerge304329
def owner : Owner := ⟨.program ⟨257⟩, ⟨70935⟩⟩
def mergeEvent : Nat := 304329
def frameStart : Nat := 303660
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩] } }
def leftRaw : List Term := Proof.Events1188.exact304323RawTerms
def rightRaw : List Term := Proof.Events1188.exact304164RawTerms
def group : MergeGroup := .operator 304323 304164
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 304323) (leftOrdinal := 15)
    (rightResult := 304164) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7228⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨70934⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge304329

namespace LeftMerge304330
def owner : Owner := ⟨.program ⟨257⟩, ⟨70935⟩⟩
def mergeEvent : Nat := 304330
def frameStart : Nat := 303660
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩] } }
def leftRaw : List Term := Proof.Events1188.exact304323RawTerms
def rightRaw : List Term := Proof.Events1188.exact304164RawTerms
def group : MergeGroup := .operator 304323 304164
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 304323) (leftOrdinal := 14)
    (rightResult := 304164) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7226⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨70934⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge304330

namespace LeftMerge304331
def owner : Owner := ⟨.program ⟨257⟩, ⟨70935⟩⟩
def mergeEvent : Nat := 304331
def frameStart : Nat := 303660
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩] } }
def leftRaw : List Term := Proof.Events1188.exact304323RawTerms
def rightRaw : List Term := Proof.Events1188.exact304164RawTerms
def group : MergeGroup := .operator 304323 304164
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 304323) (leftOrdinal := 13)
    (rightResult := 304164) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7224⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨70934⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge304331

namespace LeftMerge304332
def owner : Owner := ⟨.program ⟨257⟩, ⟨70935⟩⟩
def mergeEvent : Nat := 304332
def frameStart : Nat := 303660
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7222⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩] } }
def leftRaw : List Term := Proof.Events1188.exact304323RawTerms
def rightRaw : List Term := Proof.Events1188.exact304164RawTerms
def group : MergeGroup := .operator 304323 304164
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 304323) (leftOrdinal := 12)
    (rightResult := 304164) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7222⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨70934⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7222⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge304332

namespace LeftMerge304333
def owner : Owner := ⟨.program ⟨257⟩, ⟨70935⟩⟩
def mergeEvent : Nat := 304333
def frameStart : Nat := 303660
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7220⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩] } }
def leftRaw : List Term := Proof.Events1188.exact304323RawTerms
def rightRaw : List Term := Proof.Events1188.exact304164RawTerms
def group : MergeGroup := .operator 304323 304164
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 304323) (leftOrdinal := 11)
    (rightResult := 304164) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7220⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨70934⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7220⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge304333

namespace LeftMerge304334
def owner : Owner := ⟨.program ⟨257⟩, ⟨70935⟩⟩
def mergeEvent : Nat := 304334
def frameStart : Nat := 303660
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7218⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩] } }
def leftRaw : List Term := Proof.Events1188.exact304323RawTerms
def rightRaw : List Term := Proof.Events1188.exact304164RawTerms
def group : MergeGroup := .operator 304323 304164
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 304323) (leftOrdinal := 10)
    (rightResult := 304164) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7218⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨70934⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7218⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge304334

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
