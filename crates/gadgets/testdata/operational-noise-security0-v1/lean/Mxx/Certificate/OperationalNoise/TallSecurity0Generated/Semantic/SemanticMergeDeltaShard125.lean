import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge21367
def owner : Owner := ⟨.program ⟨214⟩, ⟨18891⟩⟩
def mergeEvent : Nat := 21367
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨6435⟩⟩, ⟨.program ⟨214⟩, ⟨18136⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6748⟩⟩] } }
def leftRaw : List Term := Proof.Events083.exact21305RawTerms
def rightRaw : List Term := Proof.Events006.exact1552RawTerms
def group : MergeGroup := .operator 21305 1552
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 21305) (leftOrdinal := 1)
    (rightResult := 1552) (rightOrdinal := 7) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6748⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6435⟩⟩, ⟨.program ⟨214⟩, ⟨18136⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨6435⟩⟩, ⟨.program ⟨214⟩, ⟨18136⟩⟩], [⟨.program ⟨214⟩, ⟨6748⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge21367

namespace LeftMerge21368
def owner : Owner := ⟨.program ⟨214⟩, ⟨18891⟩⟩
def mergeEvent : Nat := 21368
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨6437⟩⟩, ⟨.program ⟨214⟩, ⟨16939⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6748⟩⟩] } }
def leftRaw : List Term := Proof.Events083.exact21305RawTerms
def rightRaw : List Term := Proof.Events006.exact1552RawTerms
def group : MergeGroup := .operator 21305 1552
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 21305) (leftOrdinal := 1)
    (rightResult := 1552) (rightOrdinal := 8) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6748⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6437⟩⟩, ⟨.program ⟨214⟩, ⟨16939⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨6437⟩⟩, ⟨.program ⟨214⟩, ⟨16939⟩⟩], [⟨.program ⟨214⟩, ⟨6748⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge21368

namespace LeftMerge21369
def owner : Owner := ⟨.program ⟨214⟩, ⟨18891⟩⟩
def mergeEvent : Nat := 21369
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨6449⟩⟩, ⟨.program ⟨214⟩, ⟨17506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6748⟩⟩] } }
def leftRaw : List Term := Proof.Events083.exact21305RawTerms
def rightRaw : List Term := Proof.Events006.exact1552RawTerms
def group : MergeGroup := .operator 21305 1552
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 21305) (leftOrdinal := 1)
    (rightResult := 1552) (rightOrdinal := 9) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6748⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6449⟩⟩, ⟨.program ⟨214⟩, ⟨17506⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨6449⟩⟩, ⟨.program ⟨214⟩, ⟨17506⟩⟩], [⟨.program ⟨214⟩, ⟨6748⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge21369

namespace LeftMerge21370
def owner : Owner := ⟨.program ⟨214⟩, ⟨18891⟩⟩
def mergeEvent : Nat := 21370
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨6459⟩⟩, ⟨.program ⟨214⟩, ⟨17730⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6748⟩⟩] } }
def leftRaw : List Term := Proof.Events083.exact21305RawTerms
def rightRaw : List Term := Proof.Events006.exact1552RawTerms
def group : MergeGroup := .operator 21305 1552
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 21305) (leftOrdinal := 1)
    (rightResult := 1552) (rightOrdinal := 11) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6748⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6459⟩⟩, ⟨.program ⟨214⟩, ⟨17730⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨6459⟩⟩, ⟨.program ⟨214⟩, ⟨17730⟩⟩], [⟨.program ⟨214⟩, ⟨6748⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge21370

namespace LeftMerge21371
def owner : Owner := ⟨.program ⟨214⟩, ⟨18891⟩⟩
def mergeEvent : Nat := 21371
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨6467⟩⟩, ⟨.program ⟨214⟩, ⟨17961⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6748⟩⟩] } }
def leftRaw : List Term := Proof.Events083.exact21305RawTerms
def rightRaw : List Term := Proof.Events006.exact1552RawTerms
def group : MergeGroup := .operator 21305 1552
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 21305) (leftOrdinal := 1)
    (rightResult := 1552) (rightOrdinal := 12) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6748⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6467⟩⟩, ⟨.program ⟨214⟩, ⟨17961⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨6467⟩⟩, ⟨.program ⟨214⟩, ⟨17961⟩⟩], [⟨.program ⟨214⟩, ⟨6748⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge21371

namespace LeftMerge21372
def owner : Owner := ⟨.program ⟨214⟩, ⟨18891⟩⟩
def mergeEvent : Nat := 21372
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨6473⟩⟩, ⟨.program ⟨214⟩, ⟨17562⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6748⟩⟩] } }
def leftRaw : List Term := Proof.Events083.exact21305RawTerms
def rightRaw : List Term := Proof.Events006.exact1552RawTerms
def group : MergeGroup := .operator 21305 1552
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 21305) (leftOrdinal := 1)
    (rightResult := 1552) (rightOrdinal := 13) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6748⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6473⟩⟩, ⟨.program ⟨214⟩, ⟨17562⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨6473⟩⟩, ⟨.program ⟨214⟩, ⟨17562⟩⟩], [⟨.program ⟨214⟩, ⟨6748⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge21372

namespace LeftMerge21373
def owner : Owner := ⟨.program ⟨214⟩, ⟨18891⟩⟩
def mergeEvent : Nat := 21373
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨6490⟩⟩, ⟨.program ⟨214⟩, ⟨18878⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6748⟩⟩] } }
def leftRaw : List Term := Proof.Events083.exact21305RawTerms
def rightRaw : List Term := Proof.Events006.exact1552RawTerms
def group : MergeGroup := .operator 21305 1552
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 21305) (leftOrdinal := 1)
    (rightResult := 1552) (rightOrdinal := 15) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6748⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6490⟩⟩, ⟨.program ⟨214⟩, ⟨18878⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨6490⟩⟩, ⟨.program ⟨214⟩, ⟨18878⟩⟩], [⟨.program ⟨214⟩, ⟨6748⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge21373

namespace LeftMerge21374
def owner : Owner := ⟨.program ⟨214⟩, ⟨18891⟩⟩
def mergeEvent : Nat := 21374
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨6494⟩⟩, ⟨.program ⟨214⟩, ⟨17618⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6748⟩⟩] } }
def leftRaw : List Term := Proof.Events083.exact21305RawTerms
def rightRaw : List Term := Proof.Events006.exact1552RawTerms
def group : MergeGroup := .operator 21305 1552
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 21305) (leftOrdinal := 1)
    (rightResult := 1552) (rightOrdinal := 16) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6748⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6494⟩⟩, ⟨.program ⟨214⟩, ⟨17618⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨6494⟩⟩, ⟨.program ⟨214⟩, ⟨17618⟩⟩], [⟨.program ⟨214⟩, ⟨6748⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge21374

namespace LeftMerge21375
def owner : Owner := ⟨.program ⟨214⟩, ⟨18891⟩⟩
def mergeEvent : Nat := 21375
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨6502⟩⟩, ⟨.program ⟨214⟩, ⟨17674⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6748⟩⟩] } }
def leftRaw : List Term := Proof.Events083.exact21305RawTerms
def rightRaw : List Term := Proof.Events006.exact1552RawTerms
def group : MergeGroup := .operator 21305 1552
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 21305) (leftOrdinal := 1)
    (rightResult := 1552) (rightOrdinal := 18) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6748⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6502⟩⟩, ⟨.program ⟨214⟩, ⟨17674⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨6502⟩⟩, ⟨.program ⟨214⟩, ⟨17674⟩⟩], [⟨.program ⟨214⟩, ⟨6748⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge21375

namespace LeftMerge21376
def owner : Owner := ⟨.program ⟨214⟩, ⟨18891⟩⟩
def mergeEvent : Nat := 21376
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨6383⟩⟩, ⟨.program ⟨214⟩, ⟨18056⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6748⟩⟩] } }
def leftRaw : List Term := Proof.Events083.exact21305RawTerms
def rightRaw : List Term := Proof.Events006.exact1552RawTerms
def group : MergeGroup := .operator 21305 1552
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 21305) (leftOrdinal := 1)
    (rightResult := 1552) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6748⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6383⟩⟩, ⟨.program ⟨214⟩, ⟨18056⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨6383⟩⟩, ⟨.program ⟨214⟩, ⟨18056⟩⟩], [⟨.program ⟨214⟩, ⟨6748⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge21376

namespace LeftMerge21377
def owner : Owner := ⟨.program ⟨214⟩, ⟨18891⟩⟩
def mergeEvent : Nat := 21377
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨6387⟩⟩, ⟨.program ⟨214⟩, ⟨17177⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6748⟩⟩] } }
def leftRaw : List Term := Proof.Events083.exact21305RawTerms
def rightRaw : List Term := Proof.Events006.exact1552RawTerms
def group : MergeGroup := .operator 21305 1552
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 21305) (leftOrdinal := 1)
    (rightResult := 1552) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6748⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6387⟩⟩, ⟨.program ⟨214⟩, ⟨17177⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨6387⟩⟩, ⟨.program ⟨214⟩, ⟨17177⟩⟩], [⟨.program ⟨214⟩, ⟨6748⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge21377

namespace LeftMerge21378
def owner : Owner := ⟨.program ⟨214⟩, ⟨18891⟩⟩
def mergeEvent : Nat := 21378
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨6391⟩⟩, ⟨.program ⟨214⟩, ⟨17233⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6748⟩⟩] } }
def leftRaw : List Term := Proof.Events083.exact21305RawTerms
def rightRaw : List Term := Proof.Events006.exact1552RawTerms
def group : MergeGroup := .operator 21305 1552
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 21305) (leftOrdinal := 1)
    (rightResult := 1552) (rightOrdinal := 2) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6748⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6391⟩⟩, ⟨.program ⟨214⟩, ⟨17233⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨6391⟩⟩, ⟨.program ⟨214⟩, ⟨17233⟩⟩], [⟨.program ⟨214⟩, ⟨6748⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge21378

namespace LeftMerge21379
def owner : Owner := ⟨.program ⟨214⟩, ⟨18891⟩⟩
def mergeEvent : Nat := 21379
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨6398⟩⟩, ⟨.program ⟨214⟩, ⟨17450⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6748⟩⟩] } }
def leftRaw : List Term := Proof.Events083.exact21305RawTerms
def rightRaw : List Term := Proof.Events006.exact1552RawTerms
def group : MergeGroup := .operator 21305 1552
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 21305) (leftOrdinal := 1)
    (rightResult := 1552) (rightOrdinal := 3) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6748⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6398⟩⟩, ⟨.program ⟨214⟩, ⟨17450⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨6398⟩⟩, ⟨.program ⟨214⟩, ⟨17450⟩⟩], [⟨.program ⟨214⟩, ⟨6748⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge21379

namespace LeftMerge21380
def owner : Owner := ⟨.program ⟨214⟩, ⟨18891⟩⟩
def mergeEvent : Nat := 21380
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨6407⟩⟩, ⟨.program ⟨214⟩, ⟨17838⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6748⟩⟩] } }
def leftRaw : List Term := Proof.Events083.exact21305RawTerms
def rightRaw : List Term := Proof.Events006.exact1552RawTerms
def group : MergeGroup := .operator 21305 1552
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 21305) (leftOrdinal := 1)
    (rightResult := 1552) (rightOrdinal := 4) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6748⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6407⟩⟩, ⟨.program ⟨214⟩, ⟨17838⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨6407⟩⟩, ⟨.program ⟨214⟩, ⟨17838⟩⟩], [⟨.program ⟨214⟩, ⟨6748⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge21380

namespace LeftMerge21381
def owner : Owner := ⟨.program ⟨214⟩, ⟨18891⟩⟩
def mergeEvent : Nat := 21381
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨6427⟩⟩, ⟨.program ⟨214⟩, ⟨15531⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6748⟩⟩] } }
def leftRaw : List Term := Proof.Events083.exact21305RawTerms
def rightRaw : List Term := Proof.Events006.exact1552RawTerms
def group : MergeGroup := .operator 21305 1552
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 21305) (leftOrdinal := 1)
    (rightResult := 1552) (rightOrdinal := 6) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6748⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6427⟩⟩, ⟨.program ⟨214⟩, ⟨15531⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨6427⟩⟩, ⟨.program ⟨214⟩, ⟨15531⟩⟩], [⟨.program ⟨214⟩, ⟨6748⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge21381

namespace LeftMerge21382
def owner : Owner := ⟨.program ⟨214⟩, ⟨18891⟩⟩
def mergeEvent : Nat := 21382
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨6452⟩⟩, ⟨.program ⟨214⟩, ⟨15223⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6748⟩⟩] } }
def leftRaw : List Term := Proof.Events083.exact21305RawTerms
def rightRaw : List Term := Proof.Events006.exact1552RawTerms
def group : MergeGroup := .operator 21305 1552
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 21305) (leftOrdinal := 1)
    (rightResult := 1552) (rightOrdinal := 10) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6748⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6452⟩⟩, ⟨.program ⟨214⟩, ⟨15223⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨6452⟩⟩, ⟨.program ⟨214⟩, ⟨15223⟩⟩], [⟨.program ⟨214⟩, ⟨6748⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge21382

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
