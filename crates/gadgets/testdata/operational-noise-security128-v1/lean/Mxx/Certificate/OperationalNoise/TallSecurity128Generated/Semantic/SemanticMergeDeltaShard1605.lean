import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge261375
def owner : Owner := ⟨.program ⟨257⟩, ⟨69069⟩⟩
def mergeEvent : Nat := 261375
def frameStart : Nat := 260836
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨66251⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1020.exact261363RawTerms
def rightRaw : List Term := Proof.Events1020.exact261361RawTerms
def group : MergeGroup := .operator 261363 261361
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 261363) (leftOrdinal := 0)
    (rightResult := 261361) (rightOrdinal := 17) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨66251⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨66251⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge261375

namespace LeftMerge261376
def owner : Owner := ⟨.program ⟨257⟩, ⟨69069⟩⟩
def mergeEvent : Nat := 261376
def frameStart : Nat := 260836
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨62986⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1020.exact261363RawTerms
def rightRaw : List Term := Proof.Events1020.exact261361RawTerms
def group : MergeGroup := .operator 261363 261361
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 261363) (leftOrdinal := 0)
    (rightResult := 261361) (rightOrdinal := 16) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨62986⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨62986⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge261376

namespace LeftMerge261377
def owner : Owner := ⟨.program ⟨257⟩, ⟨69069⟩⟩
def mergeEvent : Nat := 261377
def frameStart : Nat := 260836
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨60006⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1020.exact261363RawTerms
def rightRaw : List Term := Proof.Events1020.exact261361RawTerms
def group : MergeGroup := .operator 261363 261361
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 261363) (leftOrdinal := 0)
    (rightResult := 261361) (rightOrdinal := 15) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨60006⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨60006⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge261377

namespace LeftMerge261378
def owner : Owner := ⟨.program ⟨257⟩, ⟨69069⟩⟩
def mergeEvent : Nat := 261378
def frameStart : Nat := 260836
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨57026⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1020.exact261363RawTerms
def rightRaw : List Term := Proof.Events1020.exact261361RawTerms
def group : MergeGroup := .operator 261363 261361
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 261363) (leftOrdinal := 0)
    (rightResult := 261361) (rightOrdinal := 14) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨57026⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨57026⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge261378

namespace LeftMerge261379
def owner : Owner := ⟨.program ⟨257⟩, ⟨69069⟩⟩
def mergeEvent : Nat := 261379
def frameStart : Nat := 260836
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨54046⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1020.exact261363RawTerms
def rightRaw : List Term := Proof.Events1020.exact261361RawTerms
def group : MergeGroup := .operator 261363 261361
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 261363) (leftOrdinal := 0)
    (rightResult := 261361) (rightOrdinal := 13) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨54046⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨54046⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge261379

namespace LeftMerge261380
def owner : Owner := ⟨.program ⟨257⟩, ⟨69069⟩⟩
def mergeEvent : Nat := 261380
def frameStart : Nat := 260836
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨51066⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1020.exact261363RawTerms
def rightRaw : List Term := Proof.Events1020.exact261361RawTerms
def group : MergeGroup := .operator 261363 261361
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 261363) (leftOrdinal := 0)
    (rightResult := 261361) (rightOrdinal := 12) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨51066⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨51066⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge261380

namespace LeftMerge261381
def owner : Owner := ⟨.program ⟨257⟩, ⟨69069⟩⟩
def mergeEvent : Nat := 261381
def frameStart : Nat := 260836
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨32011⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1020.exact261363RawTerms
def rightRaw : List Term := Proof.Events1020.exact261361RawTerms
def group : MergeGroup := .operator 261363 261361
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 261363) (leftOrdinal := 0)
    (rightResult := 261361) (rightOrdinal := 5) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨32011⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨32011⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge261381

namespace LeftMerge261382
def owner : Owner := ⟨.program ⟨257⟩, ⟨69069⟩⟩
def mergeEvent : Nat := 261382
def frameStart : Nat := 260836
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨21991⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1020.exact261363RawTerms
def rightRaw : List Term := Proof.Events1020.exact261361RawTerms
def group : MergeGroup := .operator 261363 261361
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 261363) (leftOrdinal := 0)
    (rightResult := 261361) (rightOrdinal := 2) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨21991⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨21991⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge261382

namespace LeftMerge261383
def owner : Owner := ⟨.program ⟨257⟩, ⟨69069⟩⟩
def mergeEvent : Nat := 261383
def frameStart : Nat := 260836
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨18771⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1020.exact261363RawTerms
def rightRaw : List Term := Proof.Events1020.exact261361RawTerms
def group : MergeGroup := .operator 261363 261361
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 261363) (leftOrdinal := 0)
    (rightResult := 261361) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨18771⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨18771⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge261383

namespace LeftMerge261384
def owner : Owner := ⟨.program ⟨257⟩, ⟨69069⟩⟩
def mergeEvent : Nat := 261384
def frameStart : Nat := 260836
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨15955⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1020.exact261363RawTerms
def rightRaw : List Term := Proof.Events1020.exact261361RawTerms
def group : MergeGroup := .operator 261363 261361
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 261363) (leftOrdinal := 0)
    (rightResult := 261361) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨15955⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨15955⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge261384

namespace LeftMerge261515
def owner : Owner := ⟨.program ⟨257⟩, ⟨71083⟩⟩
def mergeEvent : Nat := 261515
def frameStart : Nat := 260836
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7232⟩⟩, ⟨.program ⟨257⟩, ⟨71082⟩⟩] } }
def leftRaw : List Term := Proof.Events1021.exact261511RawTerms
def rightRaw : List Term := Proof.Events1020.exact261352RawTerms
def group : MergeGroup := .operator 261511 261352
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 261511) (leftOrdinal := 17)
    (rightResult := 261352) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7232⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71082⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7232⟩⟩, ⟨.program ⟨257⟩, ⟨71082⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge261515

namespace LeftMerge261516
def owner : Owner := ⟨.program ⟨257⟩, ⟨71083⟩⟩
def mergeEvent : Nat := 261516
def frameStart : Nat := 260836
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨71082⟩⟩] } }
def leftRaw : List Term := Proof.Events1021.exact261511RawTerms
def rightRaw : List Term := Proof.Events1020.exact261352RawTerms
def group : MergeGroup := .operator 261511 261352
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 261511) (leftOrdinal := 16)
    (rightResult := 261352) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7230⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71082⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨71082⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge261516

namespace LeftMerge261517
def owner : Owner := ⟨.program ⟨257⟩, ⟨71083⟩⟩
def mergeEvent : Nat := 261517
def frameStart : Nat := 260836
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨71082⟩⟩] } }
def leftRaw : List Term := Proof.Events1021.exact261511RawTerms
def rightRaw : List Term := Proof.Events1020.exact261352RawTerms
def group : MergeGroup := .operator 261511 261352
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 261511) (leftOrdinal := 15)
    (rightResult := 261352) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7228⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71082⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨71082⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge261517

namespace LeftMerge261518
def owner : Owner := ⟨.program ⟨257⟩, ⟨71083⟩⟩
def mergeEvent : Nat := 261518
def frameStart : Nat := 260836
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨71082⟩⟩] } }
def leftRaw : List Term := Proof.Events1021.exact261511RawTerms
def rightRaw : List Term := Proof.Events1020.exact261352RawTerms
def group : MergeGroup := .operator 261511 261352
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 261511) (leftOrdinal := 14)
    (rightResult := 261352) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7226⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71082⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨71082⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge261518

namespace LeftMerge261519
def owner : Owner := ⟨.program ⟨257⟩, ⟨71083⟩⟩
def mergeEvent : Nat := 261519
def frameStart : Nat := 260836
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨71082⟩⟩] } }
def leftRaw : List Term := Proof.Events1021.exact261511RawTerms
def rightRaw : List Term := Proof.Events1020.exact261352RawTerms
def group : MergeGroup := .operator 261511 261352
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 261511) (leftOrdinal := 13)
    (rightResult := 261352) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7224⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71082⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨71082⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge261519

namespace LeftMerge261520
def owner : Owner := ⟨.program ⟨257⟩, ⟨71083⟩⟩
def mergeEvent : Nat := 261520
def frameStart : Nat := 260836
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7222⟩⟩, ⟨.program ⟨257⟩, ⟨71082⟩⟩] } }
def leftRaw : List Term := Proof.Events1021.exact261511RawTerms
def rightRaw : List Term := Proof.Events1020.exact261352RawTerms
def group : MergeGroup := .operator 261511 261352
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 261511) (leftOrdinal := 12)
    (rightResult := 261352) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7222⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71082⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7222⟩⟩, ⟨.program ⟨257⟩, ⟨71082⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge261520

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
