import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge1561
def owner : Owner := ⟨.program ⟨214⟩, ⟨18889⟩⟩
def mergeEvent : Nat := 1561
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6467⟩⟩, ⟨.program ⟨214⟩, ⟨6543⟩⟩, ⟨.program ⟨214⟩, ⟨17961⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events006.exact1552RawTerms
def rightRaw : List Term := Proof.Events003.exact829RawTerms
def group : MergeGroup := .operator 1552 829
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 1552) (leftOrdinal := 12)
    (rightResult := 829) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6467⟩⟩, ⟨.program ⟨214⟩, ⟨17961⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6543⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨6467⟩⟩, ⟨.program ⟨214⟩, ⟨6543⟩⟩, ⟨.program ⟨214⟩, ⟨17961⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge1561

namespace LeftMerge1562
def owner : Owner := ⟨.program ⟨214⟩, ⟨18889⟩⟩
def mergeEvent : Nat := 1562
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6473⟩⟩, ⟨.program ⟨214⟩, ⟨6543⟩⟩, ⟨.program ⟨214⟩, ⟨17562⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events006.exact1552RawTerms
def rightRaw : List Term := Proof.Events003.exact829RawTerms
def group : MergeGroup := .operator 1552 829
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 1552) (leftOrdinal := 13)
    (rightResult := 829) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6473⟩⟩, ⟨.program ⟨214⟩, ⟨17562⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6543⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨6473⟩⟩, ⟨.program ⟨214⟩, ⟨6543⟩⟩, ⟨.program ⟨214⟩, ⟨17562⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge1562

namespace LeftMerge1563
def owner : Owner := ⟨.program ⟨214⟩, ⟨18889⟩⟩
def mergeEvent : Nat := 1563
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6490⟩⟩, ⟨.program ⟨214⟩, ⟨6543⟩⟩, ⟨.program ⟨214⟩, ⟨18878⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events006.exact1552RawTerms
def rightRaw : List Term := Proof.Events003.exact829RawTerms
def group : MergeGroup := .operator 1552 829
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 1552) (leftOrdinal := 15)
    (rightResult := 829) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6490⟩⟩, ⟨.program ⟨214⟩, ⟨18878⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6543⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨6490⟩⟩, ⟨.program ⟨214⟩, ⟨6543⟩⟩, ⟨.program ⟨214⟩, ⟨18878⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge1563

namespace LeftMerge1564
def owner : Owner := ⟨.program ⟨214⟩, ⟨18889⟩⟩
def mergeEvent : Nat := 1564
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6494⟩⟩, ⟨.program ⟨214⟩, ⟨6543⟩⟩, ⟨.program ⟨214⟩, ⟨17618⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events006.exact1552RawTerms
def rightRaw : List Term := Proof.Events003.exact829RawTerms
def group : MergeGroup := .operator 1552 829
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 1552) (leftOrdinal := 16)
    (rightResult := 829) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6494⟩⟩, ⟨.program ⟨214⟩, ⟨17618⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6543⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨6494⟩⟩, ⟨.program ⟨214⟩, ⟨6543⟩⟩, ⟨.program ⟨214⟩, ⟨17618⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge1564

namespace LeftMerge1565
def owner : Owner := ⟨.program ⟨214⟩, ⟨18889⟩⟩
def mergeEvent : Nat := 1565
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6502⟩⟩, ⟨.program ⟨214⟩, ⟨6543⟩⟩, ⟨.program ⟨214⟩, ⟨17674⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events006.exact1552RawTerms
def rightRaw : List Term := Proof.Events003.exact829RawTerms
def group : MergeGroup := .operator 1552 829
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 1552) (leftOrdinal := 18)
    (rightResult := 829) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6502⟩⟩, ⟨.program ⟨214⟩, ⟨17674⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6543⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨6502⟩⟩, ⟨.program ⟨214⟩, ⟨6543⟩⟩, ⟨.program ⟨214⟩, ⟨17674⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge1565

namespace LeftMerge1566
def owner : Owner := ⟨.program ⟨214⟩, ⟨18889⟩⟩
def mergeEvent : Nat := 1566
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6383⟩⟩, ⟨.program ⟨214⟩, ⟨6543⟩⟩, ⟨.program ⟨214⟩, ⟨18056⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events006.exact1552RawTerms
def rightRaw : List Term := Proof.Events003.exact829RawTerms
def group : MergeGroup := .operator 1552 829
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 1552) (leftOrdinal := 0)
    (rightResult := 829) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6383⟩⟩, ⟨.program ⟨214⟩, ⟨18056⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6543⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨6383⟩⟩, ⟨.program ⟨214⟩, ⟨6543⟩⟩, ⟨.program ⟨214⟩, ⟨18056⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge1566

namespace LeftMerge1567
def owner : Owner := ⟨.program ⟨214⟩, ⟨18889⟩⟩
def mergeEvent : Nat := 1567
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6387⟩⟩, ⟨.program ⟨214⟩, ⟨6543⟩⟩, ⟨.program ⟨214⟩, ⟨17177⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events006.exact1552RawTerms
def rightRaw : List Term := Proof.Events003.exact829RawTerms
def group : MergeGroup := .operator 1552 829
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 1552) (leftOrdinal := 1)
    (rightResult := 829) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6387⟩⟩, ⟨.program ⟨214⟩, ⟨17177⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6543⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨6387⟩⟩, ⟨.program ⟨214⟩, ⟨6543⟩⟩, ⟨.program ⟨214⟩, ⟨17177⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge1567

namespace LeftMerge1568
def owner : Owner := ⟨.program ⟨214⟩, ⟨18889⟩⟩
def mergeEvent : Nat := 1568
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6391⟩⟩, ⟨.program ⟨214⟩, ⟨6543⟩⟩, ⟨.program ⟨214⟩, ⟨17233⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events006.exact1552RawTerms
def rightRaw : List Term := Proof.Events003.exact829RawTerms
def group : MergeGroup := .operator 1552 829
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 1552) (leftOrdinal := 2)
    (rightResult := 829) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6391⟩⟩, ⟨.program ⟨214⟩, ⟨17233⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6543⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨6391⟩⟩, ⟨.program ⟨214⟩, ⟨6543⟩⟩, ⟨.program ⟨214⟩, ⟨17233⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge1568

namespace LeftMerge1569
def owner : Owner := ⟨.program ⟨214⟩, ⟨18889⟩⟩
def mergeEvent : Nat := 1569
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6398⟩⟩, ⟨.program ⟨214⟩, ⟨6543⟩⟩, ⟨.program ⟨214⟩, ⟨17450⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events006.exact1552RawTerms
def rightRaw : List Term := Proof.Events003.exact829RawTerms
def group : MergeGroup := .operator 1552 829
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 1552) (leftOrdinal := 3)
    (rightResult := 829) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6398⟩⟩, ⟨.program ⟨214⟩, ⟨17450⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6543⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨6398⟩⟩, ⟨.program ⟨214⟩, ⟨6543⟩⟩, ⟨.program ⟨214⟩, ⟨17450⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge1569

namespace LeftMerge1570
def owner : Owner := ⟨.program ⟨214⟩, ⟨18889⟩⟩
def mergeEvent : Nat := 1570
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6407⟩⟩, ⟨.program ⟨214⟩, ⟨6543⟩⟩, ⟨.program ⟨214⟩, ⟨17838⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events006.exact1552RawTerms
def rightRaw : List Term := Proof.Events003.exact829RawTerms
def group : MergeGroup := .operator 1552 829
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 1552) (leftOrdinal := 4)
    (rightResult := 829) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6407⟩⟩, ⟨.program ⟨214⟩, ⟨17838⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6543⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨6407⟩⟩, ⟨.program ⟨214⟩, ⟨6543⟩⟩, ⟨.program ⟨214⟩, ⟨17838⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge1570

namespace LeftMerge1571
def owner : Owner := ⟨.program ⟨214⟩, ⟨18889⟩⟩
def mergeEvent : Nat := 1571
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6427⟩⟩, ⟨.program ⟨214⟩, ⟨6543⟩⟩, ⟨.program ⟨214⟩, ⟨15531⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events006.exact1552RawTerms
def rightRaw : List Term := Proof.Events003.exact829RawTerms
def group : MergeGroup := .operator 1552 829
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 1552) (leftOrdinal := 6)
    (rightResult := 829) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6427⟩⟩, ⟨.program ⟨214⟩, ⟨15531⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6543⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨6427⟩⟩, ⟨.program ⟨214⟩, ⟨6543⟩⟩, ⟨.program ⟨214⟩, ⟨15531⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge1571

namespace LeftMerge1572
def owner : Owner := ⟨.program ⟨214⟩, ⟨18889⟩⟩
def mergeEvent : Nat := 1572
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6452⟩⟩, ⟨.program ⟨214⟩, ⟨6543⟩⟩, ⟨.program ⟨214⟩, ⟨15223⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events006.exact1552RawTerms
def rightRaw : List Term := Proof.Events003.exact829RawTerms
def group : MergeGroup := .operator 1552 829
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 1552) (leftOrdinal := 10)
    (rightResult := 829) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6452⟩⟩, ⟨.program ⟨214⟩, ⟨15223⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6543⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨6452⟩⟩, ⟨.program ⟨214⟩, ⟨6543⟩⟩, ⟨.program ⟨214⟩, ⟨15223⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge1572

namespace LeftMerge1573
def owner : Owner := ⟨.program ⟨214⟩, ⟨18889⟩⟩
def mergeEvent : Nat := 1573
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨6543⟩⟩, ⟨.program ⟨214⟩, ⟨15062⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events006.exact1552RawTerms
def rightRaw : List Term := Proof.Events003.exact829RawTerms
def group : MergeGroup := .operator 1552 829
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 1552) (leftOrdinal := 14)
    (rightResult := 829) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨15062⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6543⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨6543⟩⟩, ⟨.program ⟨214⟩, ⟨15062⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge1573

namespace LeftMerge1574
def owner : Owner := ⟨.program ⟨214⟩, ⟨18889⟩⟩
def mergeEvent : Nat := 1574
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨6543⟩⟩, ⟨.program ⟨214⟩, ⟨14901⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events006.exact1552RawTerms
def rightRaw : List Term := Proof.Events003.exact829RawTerms
def group : MergeGroup := .operator 1552 829
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 1552) (leftOrdinal := 17)
    (rightResult := 829) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨14901⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6543⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨6543⟩⟩, ⟨.program ⟨214⟩, ⟨14901⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge1574

namespace LeftMerge2079
def owner : Owner := ⟨.program ⟨214⟩, ⟨18504⟩⟩
def mergeEvent : Nat := 2079
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6410⟩⟩, ⟨.program ⟨214⟩, ⟨18503⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events008.exact2075RawTerms
def rightRaw : List Term := Proof.Events000.exact36RawTerms
def group : MergeGroup := .operator 2075 36
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 2075) (leftOrdinal := 0)
    (rightResult := 36) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18503⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6410⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨6410⟩⟩, ⟨.program ⟨214⟩, ⟨18503⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge2079

namespace LeftMerge2087
def owner : Owner := ⟨.program ⟨214⟩, ⟨18133⟩⟩
def mergeEvent : Nat := 2087
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6435⟩⟩, ⟨.program ⟨214⟩, ⟨18132⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events008.exact2083RawTerms
def rightRaw : List Term := Proof.Events002.exact543RawTerms
def group : MergeGroup := .operator 2083 543
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 2083) (leftOrdinal := 0)
    (rightResult := 543) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18132⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6435⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨6435⟩⟩, ⟨.program ⟨214⟩, ⟨18132⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge2087

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
