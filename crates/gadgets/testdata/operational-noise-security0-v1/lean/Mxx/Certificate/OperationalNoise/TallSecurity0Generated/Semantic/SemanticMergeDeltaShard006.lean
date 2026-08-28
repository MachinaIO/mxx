import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge2223
def owner : Owner := ⟨.program ⟨214⟩, ⟨14897⟩⟩
def mergeEvent : Nat := 2223
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨14896⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events008.exact2219RawTerms
def rightRaw : List Term := Proof.Events002.exact713RawTerms
def group : MergeGroup := .operator 2219 713
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 2219) (leftOrdinal := 0)
    (rightResult := 713) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨14896⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6495⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨14896⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge2223

namespace LeftMerge2304
def owner : Owner := ⟨.program ⟨214⟩, ⟨18874⟩⟩
def mergeEvent : Nat := 2304
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6410⟩⟩, ⟨.program ⟨214⟩, ⟨6425⟩⟩, ⟨.program ⟨214⟩, ⟨18503⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events008.exact2300RawTerms
def rightRaw : List Term := Proof.Events006.exact1577RawTerms
def group : MergeGroup := .operator 2300 1577
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 2300) (leftOrdinal := 5)
    (rightResult := 1577) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6410⟩⟩, ⟨.program ⟨214⟩, ⟨18503⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6425⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨6410⟩⟩, ⟨.program ⟨214⟩, ⟨6425⟩⟩, ⟨.program ⟨214⟩, ⟨18503⟩⟩], []⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge2304

namespace LeftMerge2305
def owner : Owner := ⟨.program ⟨214⟩, ⟨18874⟩⟩
def mergeEvent : Nat := 2305
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6425⟩⟩, ⟨.program ⟨214⟩, ⟨6435⟩⟩, ⟨.program ⟨214⟩, ⟨18132⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events008.exact2300RawTerms
def rightRaw : List Term := Proof.Events006.exact1577RawTerms
def group : MergeGroup := .operator 2300 1577
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 2300) (leftOrdinal := 7)
    (rightResult := 1577) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6435⟩⟩, ⟨.program ⟨214⟩, ⟨18132⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6425⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨6425⟩⟩, ⟨.program ⟨214⟩, ⟨6435⟩⟩, ⟨.program ⟨214⟩, ⟨18132⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge2305

namespace LeftMerge2306
def owner : Owner := ⟨.program ⟨214⟩, ⟨18874⟩⟩
def mergeEvent : Nat := 2306
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6425⟩⟩, ⟨.program ⟨214⟩, ⟨6437⟩⟩, ⟨.program ⟨214⟩, ⟨16935⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events008.exact2300RawTerms
def rightRaw : List Term := Proof.Events006.exact1577RawTerms
def group : MergeGroup := .operator 2300 1577
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 2300) (leftOrdinal := 8)
    (rightResult := 1577) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6437⟩⟩, ⟨.program ⟨214⟩, ⟨16935⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6425⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨6425⟩⟩, ⟨.program ⟨214⟩, ⟨6437⟩⟩, ⟨.program ⟨214⟩, ⟨16935⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge2306

namespace LeftMerge2307
def owner : Owner := ⟨.program ⟨214⟩, ⟨18874⟩⟩
def mergeEvent : Nat := 2307
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6425⟩⟩, ⟨.program ⟨214⟩, ⟨6449⟩⟩, ⟨.program ⟨214⟩, ⟨17502⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events008.exact2300RawTerms
def rightRaw : List Term := Proof.Events006.exact1577RawTerms
def group : MergeGroup := .operator 2300 1577
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 2300) (leftOrdinal := 9)
    (rightResult := 1577) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6449⟩⟩, ⟨.program ⟨214⟩, ⟨17502⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6425⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨6425⟩⟩, ⟨.program ⟨214⟩, ⟨6449⟩⟩, ⟨.program ⟨214⟩, ⟨17502⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge2307

namespace LeftMerge2308
def owner : Owner := ⟨.program ⟨214⟩, ⟨18874⟩⟩
def mergeEvent : Nat := 2308
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6425⟩⟩, ⟨.program ⟨214⟩, ⟨6459⟩⟩, ⟨.program ⟨214⟩, ⟨17726⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events008.exact2300RawTerms
def rightRaw : List Term := Proof.Events006.exact1577RawTerms
def group : MergeGroup := .operator 2300 1577
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 2300) (leftOrdinal := 11)
    (rightResult := 1577) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6459⟩⟩, ⟨.program ⟨214⟩, ⟨17726⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6425⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨6425⟩⟩, ⟨.program ⟨214⟩, ⟨6459⟩⟩, ⟨.program ⟨214⟩, ⟨17726⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge2308

namespace LeftMerge2309
def owner : Owner := ⟨.program ⟨214⟩, ⟨18874⟩⟩
def mergeEvent : Nat := 2309
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6425⟩⟩, ⟨.program ⟨214⟩, ⟨6467⟩⟩, ⟨.program ⟨214⟩, ⟨17957⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events008.exact2300RawTerms
def rightRaw : List Term := Proof.Events006.exact1577RawTerms
def group : MergeGroup := .operator 2300 1577
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 2300) (leftOrdinal := 12)
    (rightResult := 1577) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6467⟩⟩, ⟨.program ⟨214⟩, ⟨17957⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6425⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨6425⟩⟩, ⟨.program ⟨214⟩, ⟨6467⟩⟩, ⟨.program ⟨214⟩, ⟨17957⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge2309

namespace LeftMerge2310
def owner : Owner := ⟨.program ⟨214⟩, ⟨18874⟩⟩
def mergeEvent : Nat := 2310
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6425⟩⟩, ⟨.program ⟨214⟩, ⟨6473⟩⟩, ⟨.program ⟨214⟩, ⟨17558⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events008.exact2300RawTerms
def rightRaw : List Term := Proof.Events006.exact1577RawTerms
def group : MergeGroup := .operator 2300 1577
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 2300) (leftOrdinal := 13)
    (rightResult := 1577) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6473⟩⟩, ⟨.program ⟨214⟩, ⟨17558⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6425⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨6425⟩⟩, ⟨.program ⟨214⟩, ⟨6473⟩⟩, ⟨.program ⟨214⟩, ⟨17558⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge2310

namespace LeftMerge2311
def owner : Owner := ⟨.program ⟨214⟩, ⟨18874⟩⟩
def mergeEvent : Nat := 2311
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6425⟩⟩, ⟨.program ⟨214⟩, ⟨6490⟩⟩, ⟨.program ⟨214⟩, ⟨18863⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events008.exact2300RawTerms
def rightRaw : List Term := Proof.Events006.exact1577RawTerms
def group : MergeGroup := .operator 2300 1577
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 2300) (leftOrdinal := 15)
    (rightResult := 1577) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6490⟩⟩, ⟨.program ⟨214⟩, ⟨18863⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6425⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨6425⟩⟩, ⟨.program ⟨214⟩, ⟨6490⟩⟩, ⟨.program ⟨214⟩, ⟨18863⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge2311

namespace LeftMerge2312
def owner : Owner := ⟨.program ⟨214⟩, ⟨18874⟩⟩
def mergeEvent : Nat := 2312
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6425⟩⟩, ⟨.program ⟨214⟩, ⟨6494⟩⟩, ⟨.program ⟨214⟩, ⟨17614⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events008.exact2300RawTerms
def rightRaw : List Term := Proof.Events006.exact1577RawTerms
def group : MergeGroup := .operator 2300 1577
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 2300) (leftOrdinal := 16)
    (rightResult := 1577) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6494⟩⟩, ⟨.program ⟨214⟩, ⟨17614⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6425⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨6425⟩⟩, ⟨.program ⟨214⟩, ⟨6494⟩⟩, ⟨.program ⟨214⟩, ⟨17614⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge2312

namespace LeftMerge2313
def owner : Owner := ⟨.program ⟨214⟩, ⟨18874⟩⟩
def mergeEvent : Nat := 2313
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6425⟩⟩, ⟨.program ⟨214⟩, ⟨6502⟩⟩, ⟨.program ⟨214⟩, ⟨17670⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events008.exact2300RawTerms
def rightRaw : List Term := Proof.Events006.exact1577RawTerms
def group : MergeGroup := .operator 2300 1577
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 2300) (leftOrdinal := 18)
    (rightResult := 1577) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6502⟩⟩, ⟨.program ⟨214⟩, ⟨17670⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6425⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨6425⟩⟩, ⟨.program ⟨214⟩, ⟨6502⟩⟩, ⟨.program ⟨214⟩, ⟨17670⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge2313

namespace LeftMerge2314
def owner : Owner := ⟨.program ⟨214⟩, ⟨18874⟩⟩
def mergeEvent : Nat := 2314
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6383⟩⟩, ⟨.program ⟨214⟩, ⟨6425⟩⟩, ⟨.program ⟨214⟩, ⟨18049⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events008.exact2300RawTerms
def rightRaw : List Term := Proof.Events006.exact1577RawTerms
def group : MergeGroup := .operator 2300 1577
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 2300) (leftOrdinal := 0)
    (rightResult := 1577) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6383⟩⟩, ⟨.program ⟨214⟩, ⟨18049⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6425⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨6383⟩⟩, ⟨.program ⟨214⟩, ⟨6425⟩⟩, ⟨.program ⟨214⟩, ⟨18049⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge2314

namespace LeftMerge2315
def owner : Owner := ⟨.program ⟨214⟩, ⟨18874⟩⟩
def mergeEvent : Nat := 2315
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6387⟩⟩, ⟨.program ⟨214⟩, ⟨6425⟩⟩, ⟨.program ⟨214⟩, ⟨17173⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events008.exact2300RawTerms
def rightRaw : List Term := Proof.Events006.exact1577RawTerms
def group : MergeGroup := .operator 2300 1577
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 2300) (leftOrdinal := 1)
    (rightResult := 1577) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6387⟩⟩, ⟨.program ⟨214⟩, ⟨17173⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6425⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨6387⟩⟩, ⟨.program ⟨214⟩, ⟨6425⟩⟩, ⟨.program ⟨214⟩, ⟨17173⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge2315

namespace LeftMerge2316
def owner : Owner := ⟨.program ⟨214⟩, ⟨18874⟩⟩
def mergeEvent : Nat := 2316
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6391⟩⟩, ⟨.program ⟨214⟩, ⟨6425⟩⟩, ⟨.program ⟨214⟩, ⟨17229⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events008.exact2300RawTerms
def rightRaw : List Term := Proof.Events006.exact1577RawTerms
def group : MergeGroup := .operator 2300 1577
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 2300) (leftOrdinal := 2)
    (rightResult := 1577) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6391⟩⟩, ⟨.program ⟨214⟩, ⟨17229⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6425⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨6391⟩⟩, ⟨.program ⟨214⟩, ⟨6425⟩⟩, ⟨.program ⟨214⟩, ⟨17229⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge2316

namespace LeftMerge2317
def owner : Owner := ⟨.program ⟨214⟩, ⟨18874⟩⟩
def mergeEvent : Nat := 2317
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6398⟩⟩, ⟨.program ⟨214⟩, ⟨6425⟩⟩, ⟨.program ⟨214⟩, ⟨17446⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events008.exact2300RawTerms
def rightRaw : List Term := Proof.Events006.exact1577RawTerms
def group : MergeGroup := .operator 2300 1577
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 2300) (leftOrdinal := 3)
    (rightResult := 1577) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6398⟩⟩, ⟨.program ⟨214⟩, ⟨17446⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6425⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨6398⟩⟩, ⟨.program ⟨214⟩, ⟨6425⟩⟩, ⟨.program ⟨214⟩, ⟨17446⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge2317

namespace LeftMerge2318
def owner : Owner := ⟨.program ⟨214⟩, ⟨18874⟩⟩
def mergeEvent : Nat := 2318
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6407⟩⟩, ⟨.program ⟨214⟩, ⟨6425⟩⟩, ⟨.program ⟨214⟩, ⟨17830⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events008.exact2300RawTerms
def rightRaw : List Term := Proof.Events006.exact1577RawTerms
def group : MergeGroup := .operator 2300 1577
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 2300) (leftOrdinal := 4)
    (rightResult := 1577) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6407⟩⟩, ⟨.program ⟨214⟩, ⟨17830⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6425⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨6407⟩⟩, ⟨.program ⟨214⟩, ⟨6425⟩⟩, ⟨.program ⟨214⟩, ⟨17830⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge2318

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
