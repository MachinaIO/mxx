import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge16641
def owner : Owner := ⟨.program ⟨257⟩, ⟨9681⟩⟩
def mergeEvent : Nat := 16641
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7266⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩, ⟨.program ⟨257⟩, ⟨9523⟩⟩, ⟨.program ⟨257⟩, ⟨7105⟩⟩] } }
def leftRaw : List Term := Proof.Events064.exact16637RawTerms
def rightRaw : List Term := Proof.Events064.exact16614RawTerms
def group : MergeGroup := .operator 16637 16614
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 16637) (leftOrdinal := 0)
    (rightResult := 16614) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7266⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩, ⟨.program ⟨257⟩, ⟨9523⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7105⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7266⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩, ⟨.program ⟨257⟩, ⟨9523⟩⟩, ⟨.program ⟨257⟩, ⟨7105⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge16641

namespace LeftMerge16646
def owner : Owner := ⟨.program ⟨257⟩, ⟨7044⟩⟩
def mergeEvent : Nat := 16646
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6826⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events000.exact2RawTerms
def rightRaw : List Term := Proof.Events049.exact12797RawTerms
def group : MergeGroup := .operator 2 12797
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 2) (leftOrdinal := 0)
    (rightResult := 12797) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6826⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6826⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge16646

namespace LeftMerge16671
def owner : Owner := ⟨.program ⟨257⟩, ⟨9602⟩⟩
def mergeEvent : Nat := 16671
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7268⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩] } }
def leftRaw : List Term := Proof.Events065.exact16667RawTerms
def rightRaw : List Term := Proof.Events062.exact15984RawTerms
def group : MergeGroup := .operator 16667 15984
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 16667) (leftOrdinal := 0)
    (rightResult := 15984) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7268⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9583⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7268⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge16671

namespace LeftMerge16676
def owner : Owner := ⟨.program ⟨257⟩, ⟨9663⟩⟩
def mergeEvent : Nat := 16676
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7268⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩, ⟨.program ⟨257⟩, ⟨9525⟩⟩] } }
def leftRaw : List Term := Proof.Events065.exact16672RawTerms
def rightRaw : List Term := Proof.Events065.exact16664RawTerms
def group : MergeGroup := .operator 16672 16664
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 16672) (leftOrdinal := 0)
    (rightResult := 16664) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7268⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9525⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7268⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩, ⟨.program ⟨257⟩, ⟨9525⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge16676

namespace LeftMerge16681
def owner : Owner := ⟨.program ⟨257⟩, ⟨9682⟩⟩
def mergeEvent : Nat := 16681
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7268⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩, ⟨.program ⟨257⟩, ⟨9525⟩⟩, ⟨.program ⟨257⟩, ⟨7157⟩⟩] } }
def leftRaw : List Term := Proof.Events065.exact16677RawTerms
def rightRaw : List Term := Proof.Events065.exact16654RawTerms
def group : MergeGroup := .operator 16677 16654
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 16677) (leftOrdinal := 0)
    (rightResult := 16654) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7268⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩, ⟨.program ⟨257⟩, ⟨9525⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7157⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7268⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩, ⟨.program ⟨257⟩, ⟨9525⟩⟩, ⟨.program ⟨257⟩, ⟨7157⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge16681

namespace LeftMerge16686
def owner : Owner := ⟨.program ⟨257⟩, ⟨7026⟩⟩
def mergeEvent : Nat := 16686
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6754⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events000.exact2RawTerms
def rightRaw : List Term := Proof.Events052.exact13545RawTerms
def group : MergeGroup := .operator 2 13545
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 2) (leftOrdinal := 0)
    (rightResult := 13545) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6754⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6754⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge16686

namespace LeftMerge16711
def owner : Owner := ⟨.program ⟨257⟩, ⟨9603⟩⟩
def mergeEvent : Nat := 16711
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7270⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩] } }
def leftRaw : List Term := Proof.Events065.exact16707RawTerms
def rightRaw : List Term := Proof.Events062.exact15984RawTerms
def group : MergeGroup := .operator 16707 15984
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 16707) (leftOrdinal := 0)
    (rightResult := 15984) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7270⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9583⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7270⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge16711

namespace LeftMerge16716
def owner : Owner := ⟨.program ⟨257⟩, ⟨9664⟩⟩
def mergeEvent : Nat := 16716
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7270⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩, ⟨.program ⟨257⟩, ⟨9527⟩⟩] } }
def leftRaw : List Term := Proof.Events065.exact16712RawTerms
def rightRaw : List Term := Proof.Events065.exact16704RawTerms
def group : MergeGroup := .operator 16712 16704
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 16712) (leftOrdinal := 0)
    (rightResult := 16704) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7270⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9527⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7270⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩, ⟨.program ⟨257⟩, ⟨9527⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge16716

namespace LeftMerge16721
def owner : Owner := ⟨.program ⟨257⟩, ⟨9683⟩⟩
def mergeEvent : Nat := 16721
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7270⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩, ⟨.program ⟨257⟩, ⟨9527⟩⟩, ⟨.program ⟨257⟩, ⟨7121⟩⟩] } }
def leftRaw : List Term := Proof.Events065.exact16717RawTerms
def rightRaw : List Term := Proof.Events065.exact16694RawTerms
def group : MergeGroup := .operator 16717 16694
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 16717) (leftOrdinal := 0)
    (rightResult := 16694) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7270⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩, ⟨.program ⟨257⟩, ⟨9527⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7121⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7270⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩, ⟨.program ⟨257⟩, ⟨9527⟩⟩, ⟨.program ⟨257⟩, ⟨7121⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge16721

namespace LeftMerge16726
def owner : Owner := ⟨.program ⟨257⟩, ⟨7020⟩⟩
def mergeEvent : Nat := 16726
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6742⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events000.exact2RawTerms
def rightRaw : List Term := Proof.Events055.exact14287RawTerms
def group : MergeGroup := .operator 2 14287
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 2) (leftOrdinal := 0)
    (rightResult := 14287) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6742⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6742⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge16726

namespace LeftMerge16738
def owner : Owner := ⟨.program ⟨257⟩, ⟨9444⟩⟩
def mergeEvent : Nat := 16738
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7233⟩⟩, ⟨.program ⟨257⟩, ⟨7139⟩⟩, ⟨.program ⟨257⟩, ⟨7109⟩⟩] } }
def leftRaw : List Term := Proof.Events062.exact15977RawTerms
def rightRaw : List Term := Proof.Events065.exact16734RawTerms
def group : MergeGroup := .operator 15977 16734
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 15977) (leftOrdinal := 18)
    (rightResult := 16734) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7233⟩⟩, ⟨.program ⟨257⟩, ⟨7139⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7109⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7233⟩⟩, ⟨.program ⟨257⟩, ⟨7139⟩⟩, ⟨.program ⟨257⟩, ⟨7109⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge16738

namespace LeftMerge16739
def owner : Owner := ⟨.program ⟨257⟩, ⟨9444⟩⟩
def mergeEvent : Nat := 16739
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7231⟩⟩, ⟨.program ⟨257⟩, ⟨7147⟩⟩, ⟨.program ⟨257⟩, ⟨7109⟩⟩] } }
def leftRaw : List Term := Proof.Events062.exact15977RawTerms
def rightRaw : List Term := Proof.Events065.exact16734RawTerms
def group : MergeGroup := .operator 15977 16734
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 15977) (leftOrdinal := 17)
    (rightResult := 16734) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7231⟩⟩, ⟨.program ⟨257⟩, ⟨7147⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7109⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7231⟩⟩, ⟨.program ⟨257⟩, ⟨7147⟩⟩, ⟨.program ⟨257⟩, ⟨7109⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge16739

namespace LeftMerge16740
def owner : Owner := ⟨.program ⟨257⟩, ⟨9444⟩⟩
def mergeEvent : Nat := 16740
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7229⟩⟩, ⟨.program ⟨257⟩, ⟨7151⟩⟩, ⟨.program ⟨257⟩, ⟨7109⟩⟩] } }
def leftRaw : List Term := Proof.Events062.exact15977RawTerms
def rightRaw : List Term := Proof.Events065.exact16734RawTerms
def group : MergeGroup := .operator 15977 16734
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 15977) (leftOrdinal := 16)
    (rightResult := 16734) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7229⟩⟩, ⟨.program ⟨257⟩, ⟨7151⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7109⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7229⟩⟩, ⟨.program ⟨257⟩, ⟨7151⟩⟩, ⟨.program ⟨257⟩, ⟨7109⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge16740

namespace LeftMerge16741
def owner : Owner := ⟨.program ⟨257⟩, ⟨9444⟩⟩
def mergeEvent : Nat := 16741
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7227⟩⟩, ⟨.program ⟨257⟩, ⟨7153⟩⟩, ⟨.program ⟨257⟩, ⟨7109⟩⟩] } }
def leftRaw : List Term := Proof.Events062.exact15977RawTerms
def rightRaw : List Term := Proof.Events065.exact16734RawTerms
def group : MergeGroup := .operator 15977 16734
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 15977) (leftOrdinal := 15)
    (rightResult := 16734) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7227⟩⟩, ⟨.program ⟨257⟩, ⟨7153⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7109⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7227⟩⟩, ⟨.program ⟨257⟩, ⟨7153⟩⟩, ⟨.program ⟨257⟩, ⟨7109⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge16741

namespace LeftMerge16742
def owner : Owner := ⟨.program ⟨257⟩, ⟨9444⟩⟩
def mergeEvent : Nat := 16742
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7225⟩⟩, ⟨.program ⟨257⟩, ⟨7159⟩⟩, ⟨.program ⟨257⟩, ⟨7109⟩⟩] } }
def leftRaw : List Term := Proof.Events062.exact15977RawTerms
def rightRaw : List Term := Proof.Events065.exact16734RawTerms
def group : MergeGroup := .operator 15977 16734
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 15977) (leftOrdinal := 14)
    (rightResult := 16734) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7225⟩⟩, ⟨.program ⟨257⟩, ⟨7159⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7109⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7225⟩⟩, ⟨.program ⟨257⟩, ⟨7159⟩⟩, ⟨.program ⟨257⟩, ⟨7109⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge16742

namespace LeftMerge16743
def owner : Owner := ⟨.program ⟨257⟩, ⟨9444⟩⟩
def mergeEvent : Nat := 16743
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7223⟩⟩, ⟨.program ⟨257⟩, ⟨7161⟩⟩, ⟨.program ⟨257⟩, ⟨7109⟩⟩] } }
def leftRaw : List Term := Proof.Events062.exact15977RawTerms
def rightRaw : List Term := Proof.Events065.exact16734RawTerms
def group : MergeGroup := .operator 15977 16734
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 15977) (leftOrdinal := 13)
    (rightResult := 16734) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7223⟩⟩, ⟨.program ⟨257⟩, ⟨7161⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7109⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7223⟩⟩, ⟨.program ⟨257⟩, ⟨7161⟩⟩, ⟨.program ⟨257⟩, ⟨7109⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge16743

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
