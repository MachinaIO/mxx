import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge16481
def owner : Owner := ⟨.program ⟨257⟩, ⟨9677⟩⟩
def mergeEvent : Nat := 16481
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7258⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩, ⟨.program ⟨257⟩, ⟨9515⟩⟩, ⟨.program ⟨257⟩, ⟨7175⟩⟩] } }
def leftRaw : List Term := Proof.Events064.exact16477RawTerms
def rightRaw : List Term := Proof.Events064.exact16454RawTerms
def group : MergeGroup := .operator 16477 16454
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 16477) (leftOrdinal := 0)
    (rightResult := 16454) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7258⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩, ⟨.program ⟨257⟩, ⟨9515⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7175⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7258⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩, ⟨.program ⟨257⟩, ⟨9515⟩⟩, ⟨.program ⟨257⟩, ⟨7175⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge16481

namespace LeftMerge16486
def owner : Owner := ⟨.program ⟨257⟩, ⟨7032⟩⟩
def mergeEvent : Nat := 16486
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6770⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events000.exact2RawTerms
def rightRaw : List Term := Proof.Events038.exact9805RawTerms
def group : MergeGroup := .operator 2 9805
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 2) (leftOrdinal := 0)
    (rightResult := 9805) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6770⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6770⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge16486

namespace LeftMerge16511
def owner : Owner := ⟨.program ⟨257⟩, ⟨9598⟩⟩
def mergeEvent : Nat := 16511
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7260⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩] } }
def leftRaw : List Term := Proof.Events064.exact16507RawTerms
def rightRaw : List Term := Proof.Events062.exact15984RawTerms
def group : MergeGroup := .operator 16507 15984
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 16507) (leftOrdinal := 0)
    (rightResult := 15984) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7260⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9583⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7260⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge16511

namespace LeftMerge16516
def owner : Owner := ⟨.program ⟨257⟩, ⟨9659⟩⟩
def mergeEvent : Nat := 16516
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7260⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩, ⟨.program ⟨257⟩, ⟨9517⟩⟩] } }
def leftRaw : List Term := Proof.Events064.exact16512RawTerms
def rightRaw : List Term := Proof.Events064.exact16504RawTerms
def group : MergeGroup := .operator 16512 16504
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 16512) (leftOrdinal := 0)
    (rightResult := 16504) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7260⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9517⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7260⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩, ⟨.program ⟨257⟩, ⟨9517⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge16516

namespace LeftMerge16521
def owner : Owner := ⟨.program ⟨257⟩, ⟨9678⟩⟩
def mergeEvent : Nat := 16521
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7260⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩, ⟨.program ⟨257⟩, ⟨9517⟩⟩, ⟨.program ⟨257⟩, ⟨7133⟩⟩] } }
def leftRaw : List Term := Proof.Events064.exact16517RawTerms
def rightRaw : List Term := Proof.Events064.exact16494RawTerms
def group : MergeGroup := .operator 16517 16494
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 16517) (leftOrdinal := 0)
    (rightResult := 16494) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7260⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩, ⟨.program ⟨257⟩, ⟨9517⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7133⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7260⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩, ⟨.program ⟨257⟩, ⟨9517⟩⟩, ⟨.program ⟨257⟩, ⟨7133⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge16521

namespace LeftMerge16526
def owner : Owner := ⟨.program ⟨257⟩, ⟨7023⟩⟩
def mergeEvent : Nat := 16526
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6748⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events000.exact2RawTerms
def rightRaw : List Term := Proof.Events041.exact10553RawTerms
def group : MergeGroup := .operator 2 10553
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 2) (leftOrdinal := 0)
    (rightResult := 10553) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6748⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6748⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge16526

namespace LeftMerge16551
def owner : Owner := ⟨.program ⟨257⟩, ⟨9599⟩⟩
def mergeEvent : Nat := 16551
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7262⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩] } }
def leftRaw : List Term := Proof.Events064.exact16547RawTerms
def rightRaw : List Term := Proof.Events062.exact15984RawTerms
def group : MergeGroup := .operator 16547 15984
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 16547) (leftOrdinal := 0)
    (rightResult := 15984) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7262⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9583⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7262⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge16551

namespace LeftMerge16556
def owner : Owner := ⟨.program ⟨257⟩, ⟨9660⟩⟩
def mergeEvent : Nat := 16556
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7262⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩, ⟨.program ⟨257⟩, ⟨9519⟩⟩] } }
def leftRaw : List Term := Proof.Events064.exact16552RawTerms
def rightRaw : List Term := Proof.Events064.exact16544RawTerms
def group : MergeGroup := .operator 16552 16544
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 16552) (leftOrdinal := 0)
    (rightResult := 16544) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7262⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9519⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7262⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩, ⟨.program ⟨257⟩, ⟨9519⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge16556

namespace LeftMerge16561
def owner : Owner := ⟨.program ⟨257⟩, ⟨9679⟩⟩
def mergeEvent : Nat := 16561
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7262⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩, ⟨.program ⟨257⟩, ⟨9519⟩⟩, ⟨.program ⟨257⟩, ⟨7115⟩⟩] } }
def leftRaw : List Term := Proof.Events064.exact16557RawTerms
def rightRaw : List Term := Proof.Events064.exact16534RawTerms
def group : MergeGroup := .operator 16557 16534
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 16557) (leftOrdinal := 0)
    (rightResult := 16534) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7262⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩, ⟨.program ⟨257⟩, ⟨9519⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7115⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7262⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩, ⟨.program ⟨257⟩, ⟨9519⟩⟩, ⟨.program ⟨257⟩, ⟨7115⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge16561

namespace LeftMerge16566
def owner : Owner := ⟨.program ⟨257⟩, ⟨7034⟩⟩
def mergeEvent : Nat := 16566
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6773⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events000.exact2RawTerms
def rightRaw : List Term := Proof.Events044.exact11301RawTerms
def group : MergeGroup := .operator 2 11301
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 2) (leftOrdinal := 0)
    (rightResult := 11301) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6773⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6773⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge16566

namespace LeftMerge16591
def owner : Owner := ⟨.program ⟨257⟩, ⟨9600⟩⟩
def mergeEvent : Nat := 16591
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7264⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩] } }
def leftRaw : List Term := Proof.Events064.exact16587RawTerms
def rightRaw : List Term := Proof.Events062.exact15984RawTerms
def group : MergeGroup := .operator 16587 15984
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 16587) (leftOrdinal := 0)
    (rightResult := 15984) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7264⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9583⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7264⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge16591

namespace LeftMerge16596
def owner : Owner := ⟨.program ⟨257⟩, ⟨9661⟩⟩
def mergeEvent : Nat := 16596
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7264⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩, ⟨.program ⟨257⟩, ⟨9521⟩⟩] } }
def leftRaw : List Term := Proof.Events064.exact16592RawTerms
def rightRaw : List Term := Proof.Events064.exact16584RawTerms
def group : MergeGroup := .operator 16592 16584
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 16592) (leftOrdinal := 0)
    (rightResult := 16584) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7264⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9521⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7264⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩, ⟨.program ⟨257⟩, ⟨9521⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge16596

namespace LeftMerge16601
def owner : Owner := ⟨.program ⟨257⟩, ⟨9680⟩⟩
def mergeEvent : Nat := 16601
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7264⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩, ⟨.program ⟨257⟩, ⟨9521⟩⟩, ⟨.program ⟨257⟩, ⟨7137⟩⟩] } }
def leftRaw : List Term := Proof.Events064.exact16597RawTerms
def rightRaw : List Term := Proof.Events064.exact16574RawTerms
def group : MergeGroup := .operator 16597 16574
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 16597) (leftOrdinal := 0)
    (rightResult := 16574) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7264⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩, ⟨.program ⟨257⟩, ⟨9521⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7137⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7264⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩, ⟨.program ⟨257⟩, ⟨9521⟩⟩, ⟨.program ⟨257⟩, ⟨7137⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge16601

namespace LeftMerge16606
def owner : Owner := ⟨.program ⟨257⟩, ⟨7018⟩⟩
def mergeEvent : Nat := 16606
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6739⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events000.exact2RawTerms
def rightRaw : List Term := Proof.Events047.exact12049RawTerms
def group : MergeGroup := .operator 2 12049
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 2) (leftOrdinal := 0)
    (rightResult := 12049) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6739⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6739⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge16606

namespace LeftMerge16631
def owner : Owner := ⟨.program ⟨257⟩, ⟨9601⟩⟩
def mergeEvent : Nat := 16631
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7266⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩] } }
def leftRaw : List Term := Proof.Events064.exact16627RawTerms
def rightRaw : List Term := Proof.Events062.exact15984RawTerms
def group : MergeGroup := .operator 16627 15984
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 16627) (leftOrdinal := 0)
    (rightResult := 15984) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7266⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9583⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7266⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge16631

namespace LeftMerge16636
def owner : Owner := ⟨.program ⟨257⟩, ⟨9662⟩⟩
def mergeEvent : Nat := 16636
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7266⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩, ⟨.program ⟨257⟩, ⟨9523⟩⟩] } }
def leftRaw : List Term := Proof.Events064.exact16632RawTerms
def rightRaw : List Term := Proof.Events064.exact16624RawTerms
def group : MergeGroup := .operator 16632 16624
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 16632) (leftOrdinal := 0)
    (rightResult := 16624) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7266⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9523⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7266⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩, ⟨.program ⟨257⟩, ⟨9523⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge16636

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
