import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge16001
def owner : Owner := ⟨.program ⟨257⟩, ⟨9665⟩⟩
def mergeEvent : Nat := 16001
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7234⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩, ⟨.program ⟨257⟩, ⟨9491⟩⟩, ⟨.program ⟨257⟩, ⟨7129⟩⟩] } }
def leftRaw : List Term := Proof.Events062.exact15997RawTerms
def rightRaw : List Term := Proof.Events060.exact15499RawTerms
def group : MergeGroup := .operator 15997 15499
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 15997) (leftOrdinal := 0)
    (rightResult := 15499) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7234⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩, ⟨.program ⟨257⟩, ⟨9491⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7129⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7234⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩, ⟨.program ⟨257⟩, ⟨9491⟩⟩, ⟨.program ⟨257⟩, ⟨7129⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge16001

namespace LeftMerge16006
def owner : Owner := ⟨.program ⟨257⟩, ⟨7022⟩⟩
def mergeEvent : Nat := 16006
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6746⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events000.exact2RawTerms
def rightRaw : List Term := Proof.Events003.exact829RawTerms
def group : MergeGroup := .operator 2 829
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 2) (leftOrdinal := 0)
    (rightResult := 829) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6746⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6746⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge16006

namespace LeftMerge16031
def owner : Owner := ⟨.program ⟨257⟩, ⟨9586⟩⟩
def mergeEvent : Nat := 16031
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7236⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩] } }
def leftRaw : List Term := Proof.Events062.exact16027RawTerms
def rightRaw : List Term := Proof.Events062.exact15984RawTerms
def group : MergeGroup := .operator 16027 15984
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 16027) (leftOrdinal := 0)
    (rightResult := 15984) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7236⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9583⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7236⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge16031

namespace LeftMerge16036
def owner : Owner := ⟨.program ⟨257⟩, ⟨9647⟩⟩
def mergeEvent : Nat := 16036
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7236⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩, ⟨.program ⟨257⟩, ⟨9493⟩⟩] } }
def leftRaw : List Term := Proof.Events062.exact16032RawTerms
def rightRaw : List Term := Proof.Events062.exact16024RawTerms
def group : MergeGroup := .operator 16032 16024
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 16032) (leftOrdinal := 0)
    (rightResult := 16024) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7236⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9493⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7236⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩, ⟨.program ⟨257⟩, ⟨9493⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge16036

namespace LeftMerge16041
def owner : Owner := ⟨.program ⟨257⟩, ⟨9666⟩⟩
def mergeEvent : Nat := 16041
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7236⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩, ⟨.program ⟨257⟩, ⟨9493⟩⟩, ⟨.program ⟨257⟩, ⟨7113⟩⟩] } }
def leftRaw : List Term := Proof.Events062.exact16037RawTerms
def rightRaw : List Term := Proof.Events062.exact16014RawTerms
def group : MergeGroup := .operator 16037 16014
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 16037) (leftOrdinal := 0)
    (rightResult := 16014) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7236⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩, ⟨.program ⟨257⟩, ⟨9493⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7113⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7236⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩, ⟨.program ⟨257⟩, ⟨9493⟩⟩, ⟨.program ⟨257⟩, ⟨7113⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge16041

namespace LeftMerge16046
def owner : Owner := ⟨.program ⟨257⟩, ⟨7037⟩⟩
def mergeEvent : Nat := 16046
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6780⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events000.exact2RawTerms
def rightRaw : List Term := Proof.Events006.exact1577RawTerms
def group : MergeGroup := .operator 2 1577
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 2) (leftOrdinal := 0)
    (rightResult := 1577) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6780⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6780⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge16046

namespace LeftMerge16071
def owner : Owner := ⟨.program ⟨257⟩, ⟨9587⟩⟩
def mergeEvent : Nat := 16071
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7238⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩] } }
def leftRaw : List Term := Proof.Events062.exact16067RawTerms
def rightRaw : List Term := Proof.Events062.exact15984RawTerms
def group : MergeGroup := .operator 16067 15984
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 16067) (leftOrdinal := 0)
    (rightResult := 15984) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7238⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9583⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7238⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge16071

namespace LeftMerge16076
def owner : Owner := ⟨.program ⟨257⟩, ⟨9648⟩⟩
def mergeEvent : Nat := 16076
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7238⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩, ⟨.program ⟨257⟩, ⟨9495⟩⟩] } }
def leftRaw : List Term := Proof.Events062.exact16072RawTerms
def rightRaw : List Term := Proof.Events062.exact16064RawTerms
def group : MergeGroup := .operator 16072 16064
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 16072) (leftOrdinal := 0)
    (rightResult := 16064) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7238⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9495⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7238⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩, ⟨.program ⟨257⟩, ⟨9495⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge16076

namespace LeftMerge16081
def owner : Owner := ⟨.program ⟨257⟩, ⟨9667⟩⟩
def mergeEvent : Nat := 16081
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7238⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩, ⟨.program ⟨257⟩, ⟨9495⟩⟩, ⟨.program ⟨257⟩, ⟨7143⟩⟩] } }
def leftRaw : List Term := Proof.Events062.exact16077RawTerms
def rightRaw : List Term := Proof.Events062.exact16054RawTerms
def group : MergeGroup := .operator 16077 16054
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 16077) (leftOrdinal := 0)
    (rightResult := 16054) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7238⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩, ⟨.program ⟨257⟩, ⟨9495⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7143⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7238⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩, ⟨.program ⟨257⟩, ⟨9495⟩⟩, ⟨.program ⟨257⟩, ⟨7143⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge16081

namespace LeftMerge16086
def owner : Owner := ⟨.program ⟨257⟩, ⟨7036⟩⟩
def mergeEvent : Nat := 16086
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6779⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events000.exact2RawTerms
def rightRaw : List Term := Proof.Events009.exact2325RawTerms
def group : MergeGroup := .operator 2 2325
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 2) (leftOrdinal := 0)
    (rightResult := 2325) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6779⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6779⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge16086

namespace LeftMerge16111
def owner : Owner := ⟨.program ⟨257⟩, ⟨9588⟩⟩
def mergeEvent : Nat := 16111
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7240⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩] } }
def leftRaw : List Term := Proof.Events062.exact16107RawTerms
def rightRaw : List Term := Proof.Events062.exact15984RawTerms
def group : MergeGroup := .operator 16107 15984
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 16107) (leftOrdinal := 0)
    (rightResult := 15984) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7240⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9583⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7240⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge16111

namespace LeftMerge16116
def owner : Owner := ⟨.program ⟨257⟩, ⟨9649⟩⟩
def mergeEvent : Nat := 16116
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7240⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩, ⟨.program ⟨257⟩, ⟨9497⟩⟩] } }
def leftRaw : List Term := Proof.Events062.exact16112RawTerms
def rightRaw : List Term := Proof.Events062.exact16104RawTerms
def group : MergeGroup := .operator 16112 16104
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 16112) (leftOrdinal := 0)
    (rightResult := 16104) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7240⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9497⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7240⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩, ⟨.program ⟨257⟩, ⟨9497⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge16116

namespace LeftMerge16121
def owner : Owner := ⟨.program ⟨257⟩, ⟨9668⟩⟩
def mergeEvent : Nat := 16121
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7240⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩, ⟨.program ⟨257⟩, ⟨9497⟩⟩, ⟨.program ⟨257⟩, ⟨7141⟩⟩] } }
def leftRaw : List Term := Proof.Events062.exact16117RawTerms
def rightRaw : List Term := Proof.Events062.exact16094RawTerms
def group : MergeGroup := .operator 16117 16094
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 16117) (leftOrdinal := 0)
    (rightResult := 16094) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7240⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩, ⟨.program ⟨257⟩, ⟨9497⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7141⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7240⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩, ⟨.program ⟨257⟩, ⟨9497⟩⟩, ⟨.program ⟨257⟩, ⟨7141⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge16121

namespace LeftMerge16126
def owner : Owner := ⟨.program ⟨257⟩, ⟨7016⟩⟩
def mergeEvent : Nat := 16126
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6733⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events000.exact2RawTerms
def rightRaw : List Term := Proof.Events012.exact3073RawTerms
def group : MergeGroup := .operator 2 3073
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 2) (leftOrdinal := 0)
    (rightResult := 3073) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6733⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6733⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge16126

namespace LeftMerge16151
def owner : Owner := ⟨.program ⟨257⟩, ⟨9589⟩⟩
def mergeEvent : Nat := 16151
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7242⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩] } }
def leftRaw : List Term := Proof.Events063.exact16147RawTerms
def rightRaw : List Term := Proof.Events062.exact15984RawTerms
def group : MergeGroup := .operator 16147 15984
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 16147) (leftOrdinal := 0)
    (rightResult := 15984) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7242⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9583⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7242⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge16151

namespace LeftMerge16156
def owner : Owner := ⟨.program ⟨257⟩, ⟨9650⟩⟩
def mergeEvent : Nat := 16156
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7242⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩, ⟨.program ⟨257⟩, ⟨9499⟩⟩] } }
def leftRaw : List Term := Proof.Events063.exact16152RawTerms
def rightRaw : List Term := Proof.Events063.exact16144RawTerms
def group : MergeGroup := .operator 16152 16144
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 16152) (leftOrdinal := 0)
    (rightResult := 16144) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7242⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9499⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7242⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩, ⟨.program ⟨257⟩, ⟨9499⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge16156

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
