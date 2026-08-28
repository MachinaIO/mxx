import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge46159
def owner : Owner := ⟨.program ⟨214⟩, ⟨18688⟩⟩
def mergeEvent : Nat := 46159
def frameStart : Nat := 45478
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6739⟩⟩, ⟨.program ⟨214⟩, ⟨18687⟩⟩] } }
def leftRaw : List Term := Proof.Events180.exact46153RawTerms
def rightRaw : List Term := Proof.Events179.exact45994RawTerms
def group : MergeGroup := .operator 46153 45994
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 46153) (leftOrdinal := 15)
    (rightResult := 45994) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6739⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18687⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6739⟩⟩, ⟨.program ⟨214⟩, ⟨18687⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge46159

namespace LeftMerge46160
def owner : Owner := ⟨.program ⟨214⟩, ⟨18688⟩⟩
def mergeEvent : Nat := 46160
def frameStart : Nat := 45478
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6737⟩⟩, ⟨.program ⟨214⟩, ⟨18687⟩⟩] } }
def leftRaw : List Term := Proof.Events180.exact46153RawTerms
def rightRaw : List Term := Proof.Events179.exact45994RawTerms
def group : MergeGroup := .operator 46153 45994
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 46153) (leftOrdinal := 14)
    (rightResult := 45994) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6737⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18687⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6737⟩⟩, ⟨.program ⟨214⟩, ⟨18687⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge46160

namespace LeftMerge46161
def owner : Owner := ⟨.program ⟨214⟩, ⟨18688⟩⟩
def mergeEvent : Nat := 46161
def frameStart : Nat := 45478
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6735⟩⟩, ⟨.program ⟨214⟩, ⟨18687⟩⟩] } }
def leftRaw : List Term := Proof.Events180.exact46153RawTerms
def rightRaw : List Term := Proof.Events179.exact45994RawTerms
def group : MergeGroup := .operator 46153 45994
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 46153) (leftOrdinal := 13)
    (rightResult := 45994) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6735⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18687⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6735⟩⟩, ⟨.program ⟨214⟩, ⟨18687⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge46161

namespace LeftMerge46162
def owner : Owner := ⟨.program ⟨214⟩, ⟨18688⟩⟩
def mergeEvent : Nat := 46162
def frameStart : Nat := 45478
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6733⟩⟩, ⟨.program ⟨214⟩, ⟨18687⟩⟩] } }
def leftRaw : List Term := Proof.Events180.exact46153RawTerms
def rightRaw : List Term := Proof.Events179.exact45994RawTerms
def group : MergeGroup := .operator 46153 45994
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 46153) (leftOrdinal := 12)
    (rightResult := 45994) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6733⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18687⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6733⟩⟩, ⟨.program ⟨214⟩, ⟨18687⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge46162

namespace LeftMerge46163
def owner : Owner := ⟨.program ⟨214⟩, ⟨18688⟩⟩
def mergeEvent : Nat := 46163
def frameStart : Nat := 45478
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6731⟩⟩, ⟨.program ⟨214⟩, ⟨18687⟩⟩] } }
def leftRaw : List Term := Proof.Events180.exact46153RawTerms
def rightRaw : List Term := Proof.Events179.exact45994RawTerms
def group : MergeGroup := .operator 46153 45994
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 46153) (leftOrdinal := 11)
    (rightResult := 45994) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6731⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18687⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6731⟩⟩, ⟨.program ⟨214⟩, ⟨18687⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge46163

namespace LeftMerge46164
def owner : Owner := ⟨.program ⟨214⟩, ⟨18688⟩⟩
def mergeEvent : Nat := 46164
def frameStart : Nat := 45478
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6729⟩⟩, ⟨.program ⟨214⟩, ⟨18687⟩⟩] } }
def leftRaw : List Term := Proof.Events180.exact46153RawTerms
def rightRaw : List Term := Proof.Events179.exact45994RawTerms
def group : MergeGroup := .operator 46153 45994
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 46153) (leftOrdinal := 10)
    (rightResult := 45994) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6729⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18687⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6729⟩⟩, ⟨.program ⟨214⟩, ⟨18687⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge46164

namespace LeftMerge46165
def owner : Owner := ⟨.program ⟨214⟩, ⟨18688⟩⟩
def mergeEvent : Nat := 46165
def frameStart : Nat := 45478
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6727⟩⟩, ⟨.program ⟨214⟩, ⟨18687⟩⟩] } }
def leftRaw : List Term := Proof.Events180.exact46153RawTerms
def rightRaw : List Term := Proof.Events179.exact45994RawTerms
def group : MergeGroup := .operator 46153 45994
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 46153) (leftOrdinal := 9)
    (rightResult := 45994) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6727⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18687⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6727⟩⟩, ⟨.program ⟨214⟩, ⟨18687⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge46165

namespace LeftMerge46166
def owner : Owner := ⟨.program ⟨214⟩, ⟨18688⟩⟩
def mergeEvent : Nat := 46166
def frameStart : Nat := 45478
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6725⟩⟩, ⟨.program ⟨214⟩, ⟨18687⟩⟩] } }
def leftRaw : List Term := Proof.Events180.exact46153RawTerms
def rightRaw : List Term := Proof.Events179.exact45994RawTerms
def group : MergeGroup := .operator 46153 45994
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 46153) (leftOrdinal := 8)
    (rightResult := 45994) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6725⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18687⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6725⟩⟩, ⟨.program ⟨214⟩, ⟨18687⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge46166

namespace LeftMerge46167
def owner : Owner := ⟨.program ⟨214⟩, ⟨18688⟩⟩
def mergeEvent : Nat := 46167
def frameStart : Nat := 45478
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6723⟩⟩, ⟨.program ⟨214⟩, ⟨18687⟩⟩] } }
def leftRaw : List Term := Proof.Events180.exact46153RawTerms
def rightRaw : List Term := Proof.Events179.exact45994RawTerms
def group : MergeGroup := .operator 46153 45994
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 46153) (leftOrdinal := 7)
    (rightResult := 45994) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6723⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18687⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6723⟩⟩, ⟨.program ⟨214⟩, ⟨18687⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge46167

namespace LeftMerge46168
def owner : Owner := ⟨.program ⟨214⟩, ⟨18688⟩⟩
def mergeEvent : Nat := 46168
def frameStart : Nat := 45478
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6721⟩⟩, ⟨.program ⟨214⟩, ⟨18687⟩⟩] } }
def leftRaw : List Term := Proof.Events180.exact46153RawTerms
def rightRaw : List Term := Proof.Events179.exact45994RawTerms
def group : MergeGroup := .operator 46153 45994
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 46153) (leftOrdinal := 6)
    (rightResult := 45994) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6721⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18687⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6721⟩⟩, ⟨.program ⟨214⟩, ⟨18687⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge46168

namespace LeftMerge46169
def owner : Owner := ⟨.program ⟨214⟩, ⟨18688⟩⟩
def mergeEvent : Nat := 46169
def frameStart : Nat := 45478
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6719⟩⟩, ⟨.program ⟨214⟩, ⟨18687⟩⟩] } }
def leftRaw : List Term := Proof.Events180.exact46153RawTerms
def rightRaw : List Term := Proof.Events179.exact45994RawTerms
def group : MergeGroup := .operator 46153 45994
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 46153) (leftOrdinal := 5)
    (rightResult := 45994) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6719⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18687⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6719⟩⟩, ⟨.program ⟨214⟩, ⟨18687⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge46169

namespace LeftMerge46170
def owner : Owner := ⟨.program ⟨214⟩, ⟨18688⟩⟩
def mergeEvent : Nat := 46170
def frameStart : Nat := 45478
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6717⟩⟩, ⟨.program ⟨214⟩, ⟨18687⟩⟩] } }
def leftRaw : List Term := Proof.Events180.exact46153RawTerms
def rightRaw : List Term := Proof.Events179.exact45994RawTerms
def group : MergeGroup := .operator 46153 45994
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 46153) (leftOrdinal := 4)
    (rightResult := 45994) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6717⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18687⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6717⟩⟩, ⟨.program ⟨214⟩, ⟨18687⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge46170

namespace LeftMerge46171
def owner : Owner := ⟨.program ⟨214⟩, ⟨18688⟩⟩
def mergeEvent : Nat := 46171
def frameStart : Nat := 45478
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6715⟩⟩, ⟨.program ⟨214⟩, ⟨18687⟩⟩] } }
def leftRaw : List Term := Proof.Events180.exact46153RawTerms
def rightRaw : List Term := Proof.Events179.exact45994RawTerms
def group : MergeGroup := .operator 46153 45994
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 46153) (leftOrdinal := 3)
    (rightResult := 45994) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6715⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18687⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6715⟩⟩, ⟨.program ⟨214⟩, ⟨18687⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge46171

namespace LeftMerge46172
def owner : Owner := ⟨.program ⟨214⟩, ⟨18688⟩⟩
def mergeEvent : Nat := 46172
def frameStart : Nat := 45478
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6713⟩⟩, ⟨.program ⟨214⟩, ⟨18687⟩⟩] } }
def leftRaw : List Term := Proof.Events180.exact46153RawTerms
def rightRaw : List Term := Proof.Events179.exact45994RawTerms
def group : MergeGroup := .operator 46153 45994
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 46153) (leftOrdinal := 2)
    (rightResult := 45994) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6713⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18687⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6713⟩⟩, ⟨.program ⟨214⟩, ⟨18687⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge46172

namespace LeftMerge46173
def owner : Owner := ⟨.program ⟨214⟩, ⟨18688⟩⟩
def mergeEvent : Nat := 46173
def frameStart : Nat := 45478
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6711⟩⟩, ⟨.program ⟨214⟩, ⟨18687⟩⟩] } }
def leftRaw : List Term := Proof.Events180.exact46153RawTerms
def rightRaw : List Term := Proof.Events179.exact45994RawTerms
def group : MergeGroup := .operator 46153 45994
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 46153) (leftOrdinal := 1)
    (rightResult := 45994) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6711⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18687⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6711⟩⟩, ⟨.program ⟨214⟩, ⟨18687⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge46173

namespace LeftMerge46174
def owner : Owner := ⟨.program ⟨214⟩, ⟨18688⟩⟩
def mergeEvent : Nat := 46174
def frameStart : Nat := 45478
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6709⟩⟩, ⟨.program ⟨214⟩, ⟨18687⟩⟩] } }
def leftRaw : List Term := Proof.Events180.exact46153RawTerms
def rightRaw : List Term := Proof.Events179.exact45994RawTerms
def group : MergeGroup := .operator 46153 45994
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 46153) (leftOrdinal := 0)
    (rightResult := 45994) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6709⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18687⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6709⟩⟩, ⟨.program ⟨214⟩, ⟨18687⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge46174

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
