import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge94251
def owner : Owner := ⟨.program ⟨214⟩, ⟨30127⟩⟩
def mergeEvent : Nat := 94251
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6755⟩⟩, ⟨.program ⟨214⟩, ⟨7885⟩⟩, ⟨.program ⟨214⟩, ⟨7829⟩⟩] } }
def leftRaw : List Term := Proof.Events367.exact94188RawTerms
def rightRaw : List Term := Proof.Events024.exact6161RawTerms
def group : MergeGroup := .operator 94188 6161
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 94188) (leftOrdinal := 0)
    (rightResult := 6161) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6755⟩⟩, ⟨.program ⟨214⟩, ⟨7885⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨7829⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6755⟩⟩, ⟨.program ⟨214⟩, ⟨7885⟩⟩, ⟨.program ⟨214⟩, ⟨7829⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge94251

namespace LeftMerge94256
def owner : Owner := ⟨.program ⟨214⟩, ⟨30128⟩⟩
def mergeEvent : Nat := 94256
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨6410⟩⟩, ⟨.program ⟨214⟩, ⟨18495⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6756⟩⟩] } }
def leftRaw : List Term := Proof.Events368.exact94252RawTerms
def rightRaw : List Term := Proof.Events312.exact79885RawTerms
def group : MergeGroup := .operator 94252 79885
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 94252) (leftOrdinal := 6)
    (rightResult := 79885) (rightOrdinal := 24) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨6410⟩⟩, ⟨.program ⟨214⟩, ⟨18495⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6756⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨6410⟩⟩, ⟨.program ⟨214⟩, ⟨18495⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6756⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨6410⟩⟩, ⟨.program ⟨214⟩, ⟨18495⟩⟩], [⟨.program ⟨214⟩, ⟨6756⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge94256

namespace LeftMerge94257
def owner : Owner := ⟨.program ⟨214⟩, ⟨30128⟩⟩
def mergeEvent : Nat := 94257
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨6435⟩⟩, ⟨.program ⟨214⟩, ⟨18124⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6756⟩⟩] } }
def leftRaw : List Term := Proof.Events368.exact94252RawTerms
def rightRaw : List Term := Proof.Events312.exact79885RawTerms
def group : MergeGroup := .operator 94252 79885
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 94252) (leftOrdinal := 8)
    (rightResult := 79885) (rightOrdinal := 26) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨6435⟩⟩, ⟨.program ⟨214⟩, ⟨18124⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6756⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨6435⟩⟩, ⟨.program ⟨214⟩, ⟨18124⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6756⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨6435⟩⟩, ⟨.program ⟨214⟩, ⟨18124⟩⟩], [⟨.program ⟨214⟩, ⟨6756⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge94257

namespace LeftMerge94258
def owner : Owner := ⟨.program ⟨214⟩, ⟨30128⟩⟩
def mergeEvent : Nat := 94258
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨6437⟩⟩, ⟨.program ⟨214⟩, ⟨16927⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6756⟩⟩] } }
def leftRaw : List Term := Proof.Events368.exact94252RawTerms
def rightRaw : List Term := Proof.Events312.exact79885RawTerms
def group : MergeGroup := .operator 94252 79885
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 94252) (leftOrdinal := 9)
    (rightResult := 79885) (rightOrdinal := 27) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨6437⟩⟩, ⟨.program ⟨214⟩, ⟨16927⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6756⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨6437⟩⟩, ⟨.program ⟨214⟩, ⟨16927⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6756⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨6437⟩⟩, ⟨.program ⟨214⟩, ⟨16927⟩⟩], [⟨.program ⟨214⟩, ⟨6756⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge94258

namespace LeftMerge94259
def owner : Owner := ⟨.program ⟨214⟩, ⟨30128⟩⟩
def mergeEvent : Nat := 94259
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨6449⟩⟩, ⟨.program ⟨214⟩, ⟨17494⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6756⟩⟩] } }
def leftRaw : List Term := Proof.Events368.exact94252RawTerms
def rightRaw : List Term := Proof.Events312.exact79885RawTerms
def group : MergeGroup := .operator 94252 79885
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 94252) (leftOrdinal := 10)
    (rightResult := 79885) (rightOrdinal := 28) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨6449⟩⟩, ⟨.program ⟨214⟩, ⟨17494⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6756⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨6449⟩⟩, ⟨.program ⟨214⟩, ⟨17494⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6756⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨6449⟩⟩, ⟨.program ⟨214⟩, ⟨17494⟩⟩], [⟨.program ⟨214⟩, ⟨6756⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge94259

namespace LeftMerge94260
def owner : Owner := ⟨.program ⟨214⟩, ⟨30128⟩⟩
def mergeEvent : Nat := 94260
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨6459⟩⟩, ⟨.program ⟨214⟩, ⟨17718⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6756⟩⟩] } }
def leftRaw : List Term := Proof.Events368.exact94252RawTerms
def rightRaw : List Term := Proof.Events312.exact79885RawTerms
def group : MergeGroup := .operator 94252 79885
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 94252) (leftOrdinal := 12)
    (rightResult := 79885) (rightOrdinal := 30) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨6459⟩⟩, ⟨.program ⟨214⟩, ⟨17718⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6756⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨6459⟩⟩, ⟨.program ⟨214⟩, ⟨17718⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6756⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨6459⟩⟩, ⟨.program ⟨214⟩, ⟨17718⟩⟩], [⟨.program ⟨214⟩, ⟨6756⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge94260

namespace LeftMerge94261
def owner : Owner := ⟨.program ⟨214⟩, ⟨30128⟩⟩
def mergeEvent : Nat := 94261
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨6467⟩⟩, ⟨.program ⟨214⟩, ⟨17949⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6756⟩⟩] } }
def leftRaw : List Term := Proof.Events368.exact94252RawTerms
def rightRaw : List Term := Proof.Events312.exact79885RawTerms
def group : MergeGroup := .operator 94252 79885
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 94252) (leftOrdinal := 13)
    (rightResult := 79885) (rightOrdinal := 31) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨6467⟩⟩, ⟨.program ⟨214⟩, ⟨17949⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6756⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨6467⟩⟩, ⟨.program ⟨214⟩, ⟨17949⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6756⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨6467⟩⟩, ⟨.program ⟨214⟩, ⟨17949⟩⟩], [⟨.program ⟨214⟩, ⟨6756⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge94261

namespace LeftMerge94262
def owner : Owner := ⟨.program ⟨214⟩, ⟨30128⟩⟩
def mergeEvent : Nat := 94262
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨6473⟩⟩, ⟨.program ⟨214⟩, ⟨17550⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6756⟩⟩] } }
def leftRaw : List Term := Proof.Events368.exact94252RawTerms
def rightRaw : List Term := Proof.Events312.exact79885RawTerms
def group : MergeGroup := .operator 94252 79885
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 94252) (leftOrdinal := 14)
    (rightResult := 79885) (rightOrdinal := 32) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨6473⟩⟩, ⟨.program ⟨214⟩, ⟨17550⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6756⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨6473⟩⟩, ⟨.program ⟨214⟩, ⟨17550⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6756⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨6473⟩⟩, ⟨.program ⟨214⟩, ⟨17550⟩⟩], [⟨.program ⟨214⟩, ⟨6756⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge94262

namespace LeftMerge94263
def owner : Owner := ⟨.program ⟨214⟩, ⟨30128⟩⟩
def mergeEvent : Nat := 94263
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨6490⟩⟩, ⟨.program ⟨214⟩, ⟨18832⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6756⟩⟩] } }
def leftRaw : List Term := Proof.Events368.exact94252RawTerms
def rightRaw : List Term := Proof.Events312.exact79885RawTerms
def group : MergeGroup := .operator 94252 79885
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 94252) (leftOrdinal := 16)
    (rightResult := 79885) (rightOrdinal := 34) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨6490⟩⟩, ⟨.program ⟨214⟩, ⟨18832⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6756⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨6490⟩⟩, ⟨.program ⟨214⟩, ⟨18832⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6756⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨6490⟩⟩, ⟨.program ⟨214⟩, ⟨18832⟩⟩], [⟨.program ⟨214⟩, ⟨6756⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge94263

namespace LeftMerge94264
def owner : Owner := ⟨.program ⟨214⟩, ⟨30128⟩⟩
def mergeEvent : Nat := 94264
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨6494⟩⟩, ⟨.program ⟨214⟩, ⟨17606⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6756⟩⟩] } }
def leftRaw : List Term := Proof.Events368.exact94252RawTerms
def rightRaw : List Term := Proof.Events312.exact79885RawTerms
def group : MergeGroup := .operator 94252 79885
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 94252) (leftOrdinal := 17)
    (rightResult := 79885) (rightOrdinal := 35) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨6494⟩⟩, ⟨.program ⟨214⟩, ⟨17606⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6756⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨6494⟩⟩, ⟨.program ⟨214⟩, ⟨17606⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6756⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨6494⟩⟩, ⟨.program ⟨214⟩, ⟨17606⟩⟩], [⟨.program ⟨214⟩, ⟨6756⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge94264

namespace LeftMerge94265
def owner : Owner := ⟨.program ⟨214⟩, ⟨30128⟩⟩
def mergeEvent : Nat := 94265
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨6502⟩⟩, ⟨.program ⟨214⟩, ⟨17662⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6756⟩⟩] } }
def leftRaw : List Term := Proof.Events368.exact94252RawTerms
def rightRaw : List Term := Proof.Events312.exact79885RawTerms
def group : MergeGroup := .operator 94252 79885
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 94252) (leftOrdinal := 19)
    (rightResult := 79885) (rightOrdinal := 37) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨6502⟩⟩, ⟨.program ⟨214⟩, ⟨17662⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6756⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨6502⟩⟩, ⟨.program ⟨214⟩, ⟨17662⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6756⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨6502⟩⟩, ⟨.program ⟨214⟩, ⟨17662⟩⟩], [⟨.program ⟨214⟩, ⟨6756⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge94265

namespace LeftMerge94266
def owner : Owner := ⟨.program ⟨214⟩, ⟨30128⟩⟩
def mergeEvent : Nat := 94266
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨6383⟩⟩, ⟨.program ⟨214⟩, ⟨18035⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6756⟩⟩] } }
def leftRaw : List Term := Proof.Events368.exact94252RawTerms
def rightRaw : List Term := Proof.Events312.exact79885RawTerms
def group : MergeGroup := .operator 94252 79885
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 94252) (leftOrdinal := 1)
    (rightResult := 79885) (rightOrdinal := 19) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨6383⟩⟩, ⟨.program ⟨214⟩, ⟨18035⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6756⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨6383⟩⟩, ⟨.program ⟨214⟩, ⟨18035⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6756⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨6383⟩⟩, ⟨.program ⟨214⟩, ⟨18035⟩⟩], [⟨.program ⟨214⟩, ⟨6756⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge94266

namespace LeftMerge94267
def owner : Owner := ⟨.program ⟨214⟩, ⟨30128⟩⟩
def mergeEvent : Nat := 94267
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨6387⟩⟩, ⟨.program ⟨214⟩, ⟨17165⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6756⟩⟩] } }
def leftRaw : List Term := Proof.Events368.exact94252RawTerms
def rightRaw : List Term := Proof.Events312.exact79885RawTerms
def group : MergeGroup := .operator 94252 79885
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 94252) (leftOrdinal := 2)
    (rightResult := 79885) (rightOrdinal := 20) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨6387⟩⟩, ⟨.program ⟨214⟩, ⟨17165⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6756⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨6387⟩⟩, ⟨.program ⟨214⟩, ⟨17165⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6756⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨6387⟩⟩, ⟨.program ⟨214⟩, ⟨17165⟩⟩], [⟨.program ⟨214⟩, ⟨6756⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge94267

namespace LeftMerge94268
def owner : Owner := ⟨.program ⟨214⟩, ⟨30128⟩⟩
def mergeEvent : Nat := 94268
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨6391⟩⟩, ⟨.program ⟨214⟩, ⟨17221⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6756⟩⟩] } }
def leftRaw : List Term := Proof.Events368.exact94252RawTerms
def rightRaw : List Term := Proof.Events312.exact79885RawTerms
def group : MergeGroup := .operator 94252 79885
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 94252) (leftOrdinal := 3)
    (rightResult := 79885) (rightOrdinal := 21) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨6391⟩⟩, ⟨.program ⟨214⟩, ⟨17221⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6756⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨6391⟩⟩, ⟨.program ⟨214⟩, ⟨17221⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6756⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨6391⟩⟩, ⟨.program ⟨214⟩, ⟨17221⟩⟩], [⟨.program ⟨214⟩, ⟨6756⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge94268

namespace LeftMerge94269
def owner : Owner := ⟨.program ⟨214⟩, ⟨30128⟩⟩
def mergeEvent : Nat := 94269
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨6398⟩⟩, ⟨.program ⟨214⟩, ⟨17438⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6756⟩⟩] } }
def leftRaw : List Term := Proof.Events368.exact94252RawTerms
def rightRaw : List Term := Proof.Events312.exact79885RawTerms
def group : MergeGroup := .operator 94252 79885
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 94252) (leftOrdinal := 4)
    (rightResult := 79885) (rightOrdinal := 22) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨6398⟩⟩, ⟨.program ⟨214⟩, ⟨17438⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6756⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨6398⟩⟩, ⟨.program ⟨214⟩, ⟨17438⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6756⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨6398⟩⟩, ⟨.program ⟨214⟩, ⟨17438⟩⟩], [⟨.program ⟨214⟩, ⟨6756⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge94269

namespace LeftMerge94270
def owner : Owner := ⟨.program ⟨214⟩, ⟨30128⟩⟩
def mergeEvent : Nat := 94270
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨6407⟩⟩, ⟨.program ⟨214⟩, ⟨17814⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6756⟩⟩] } }
def leftRaw : List Term := Proof.Events368.exact94252RawTerms
def rightRaw : List Term := Proof.Events312.exact79885RawTerms
def group : MergeGroup := .operator 94252 79885
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 94252) (leftOrdinal := 5)
    (rightResult := 79885) (rightOrdinal := 23) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨6407⟩⟩, ⟨.program ⟨214⟩, ⟨17814⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6756⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨6407⟩⟩, ⟨.program ⟨214⟩, ⟨17814⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6756⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨6407⟩⟩, ⟨.program ⟨214⟩, ⟨17814⟩⟩], [⟨.program ⟨214⟩, ⟨6756⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge94270

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
