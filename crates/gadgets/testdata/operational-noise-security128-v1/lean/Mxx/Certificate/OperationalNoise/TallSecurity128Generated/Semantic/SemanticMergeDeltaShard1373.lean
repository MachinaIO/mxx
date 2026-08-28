import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge222673
def owner : Owner := ⟨.program ⟨257⟩, ⟨45136⟩⟩
def mergeEvent : Nat := 222673
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨14766⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7284⟩⟩] } }
def leftRaw : List Term := Proof.Events869.exact222666RawTerms
def rightRaw : List Term := Proof.Events041.exact10592RawTerms
def group : MergeGroup := .operator 222666 10592
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 222666) (leftOrdinal := 0)
    (rightResult := 10592) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7284⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14766⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨14766⟩⟩], [⟨.program ⟨257⟩, ⟨7284⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge222673

namespace LeftMerge222678
def owner : Owner := ⟨.program ⟨257⟩, ⟨14767⟩⟩
def mergeEvent : Nat := 222678
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨14766⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events041.exact10592RawTerms
def rightRaw : List Term := Proof.Events867.exact222153RawTerms
def group : MergeGroup := .operator 10592 222153
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 10592) (leftOrdinal := 0)
    (rightResult := 222153) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14766⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨14766⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge222678

namespace LeftMerge222683
def owner : Owner := ⟨.program ⟨257⟩, ⟨8493⟩⟩
def mergeEvent : Nat := 222683
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩] } }
def leftRaw : List Term := Proof.Events867.exact222023RawTerms
def rightRaw : List Term := Proof.Events068.exact17622RawTerms
def group : MergeGroup := .operator 222023 17622
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 222023) (leftOrdinal := 0)
    (rightResult := 17622) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7301⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge222683

namespace LeftMerge222700
def owner : Owner := ⟨.program ⟨257⟩, ⟨14770⟩⟩
def mergeEvent : Nat := 222700
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨14766⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩] } }
def leftRaw : List Term := Proof.Events869.exact222694RawTerms
def rightRaw : List Term := Proof.Events068.exact17611RawTerms
def group : MergeGroup := .operator 222694 17611
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 222694) (leftOrdinal := 1)
    (rightResult := 17611) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨14766⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9562⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨14766⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge222700

namespace LeftMerge222702
def owner : Owner := ⟨.program ⟨257⟩, ⟨14770⟩⟩
def mergeEvent : Nat := 222702
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨14766⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7284⟩⟩] } }
def rhsRaw : List Term := Proof.Events068.exact17581RawTerms
def group : MergeGroup := .relation 222701
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 222701) (rhsResult := 17581)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨14766⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨9562⟩⟩) ⟨7284⟩ 17581) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7284⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨14766⟩⟩], [⟨.program ⟨257⟩, ⟨7284⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge222702

namespace LeftMerge222703
def owner : Owner := ⟨.program ⟨257⟩, ⟨14770⟩⟩
def mergeEvent : Nat := 222703
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩] } }
def leftRaw : List Term := Proof.Events869.exact222694RawTerms
def rightRaw : List Term := Proof.Events068.exact17611RawTerms
def group : MergeGroup := .operator 222694 17611
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 222694) (leftOrdinal := 0)
    (rightResult := 17611) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9562⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge222703

namespace LeftMerge222708
def owner : Owner := ⟨.program ⟨257⟩, ⟨45137⟩⟩
def mergeEvent : Nat := 222708
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨14766⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7284⟩⟩] } }
def leftRaw : List Term := Proof.Events869.exact222704RawTerms
def rightRaw : List Term := Proof.Events869.exact222674RawTerms
def group : MergeGroup := .operator 222704 222674
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 222704) (leftOrdinal := 1)
    (rightResult := 222674) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨14766⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7284⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨14766⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7284⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨14766⟩⟩], [⟨.program ⟨257⟩, ⟨7284⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge222708

namespace LeftMerge222716
def owner : Owner := ⟨.program ⟨257⟩, ⟨46969⟩⟩
def mergeEvent : Nat := 222716
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨14766⟩⟩, ⟨.program ⟨257⟩, ⟨45130⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨46968⟩⟩] } }
def leftRaw : List Term := Proof.Events869.exact222710RawTerms
def rightRaw : List Term := Proof.Events869.exact222646RawTerms
def group : MergeGroup := .operator 222710 222646
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 222710) (leftOrdinal := 1)
    (rightResult := 222646) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨14766⟩⟩, ⟨.program ⟨257⟩, ⟨45130⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨46968⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨14766⟩⟩, ⟨.program ⟨257⟩, ⟨45130⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨46968⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge222716

namespace LeftMerge222718
def owner : Owner := ⟨.program ⟨257⟩, ⟨46969⟩⟩
def mergeEvent : Nat := 222718
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨14766⟩⟩, ⟨.program ⟨257⟩, ⟨45130⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46463⟩⟩] } }
def rhsRaw : List Term := Proof.Events869.exact222643RawTerms
def group : MergeGroup := .relation 222717
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 222717) (rhsResult := 222643)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨14766⟩⟩, ⟨.program ⟨257⟩, ⟨45130⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨46968⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨46968⟩⟩) ⟨46463⟩ 222643) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨46463⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨14766⟩⟩, ⟨.program ⟨257⟩, ⟨45130⟩⟩], [⟨.program ⟨257⟩, ⟨46463⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge222718

namespace LeftMerge222719
def owner : Owner := ⟨.program ⟨257⟩, ⟨46969⟩⟩
def mergeEvent : Nat := 222719
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨46968⟩⟩] } }
def leftRaw : List Term := Proof.Events869.exact222710RawTerms
def rightRaw : List Term := Proof.Events869.exact222646RawTerms
def group : MergeGroup := .operator 222710 222646
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 222710) (leftOrdinal := 0)
    (rightResult := 222646) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨46968⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨46968⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge222719

namespace LeftMerge222733
def owner : Owner := ⟨.program ⟨257⟩, ⟨45902⟩⟩
def mergeEvent : Nat := 222733
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨45899⟩⟩] } }
def leftRaw : List Term := Proof.Events868.exact222245RawTerms
def rightRaw : List Term := Proof.Events870.exact222727RawTerms
def group : MergeGroup := .operator 222245 222727
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 222245) (leftOrdinal := 0)
    (rightResult := 222727) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨45899⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨45899⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge222733

namespace LeftMerge222812
def owner : Owner := ⟨.program ⟨257⟩, ⟨45131⟩⟩
def mergeEvent : Nat := 222812
def frameStart : Nat := 222782
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14766⟩⟩, ⟨.program ⟨257⟩, ⟨45130⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events870.exact222808RawTerms
def rightRaw : List Term := Proof.Events870.exact222805RawTerms
def group : MergeGroup := .operator 222808 222805
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 222808) (leftOrdinal := 0)
    (rightResult := 222805) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14766⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨45130⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨14766⟩⟩, ⟨.program ⟨257⟩, ⟨45130⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge222812

namespace LeftMerge222842
def owner : Owner := ⟨.program ⟨257⟩, ⟨46744⟩⟩
def mergeEvent : Nat := 222842
def frameStart : Nat := 222782
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14766⟩⟩, ⟨.program ⟨257⟩, ⟨45130⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events870.exact222838RawTerms
def rightRaw : List Term := Proof.Events870.exact222836RawTerms
def group : MergeGroup := .operator 222838 222836
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 222838) (leftOrdinal := 0)
    (rightResult := 222836) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14766⟩⟩, ⟨.program ⟨257⟩, ⟨45130⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨14766⟩⟩, ⟨.program ⟨257⟩, ⟨45130⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge222842

namespace LeftMerge222865
def owner : Owner := ⟨.program ⟨257⟩, ⟨9564⟩⟩
def mergeEvent : Nat := 222865
def frameStart : Nat := 222782
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩] } }
def leftRaw : List Term := Proof.Events870.exact222861RawTerms
def rightRaw : List Term := Proof.Events870.exact222858RawTerms
def group : MergeGroup := .operator 222861 222858
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 222861) (leftOrdinal := 0)
    (rightResult := 222858) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9562⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge222865

namespace LeftMerge222874
def owner : Owner := ⟨.program ⟨257⟩, ⟨46971⟩⟩
def mergeEvent : Nat := 222874
def frameStart : Nat := 222782
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨46968⟩⟩] } }
def leftRaw : List Term := Proof.Events870.exact222870RawTerms
def rightRaw : List Term := Proof.Events870.exact222827RawTerms
def group : MergeGroup := .operator 222870 222827
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 222870) (leftOrdinal := 0)
    (rightResult := 222827) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨46968⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨46968⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge222874

namespace LeftMerge222875
def owner : Owner := ⟨.program ⟨257⟩, ⟨46971⟩⟩
def mergeEvent : Nat := 222875
def frameStart : Nat := 222782
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14766⟩⟩, ⟨.program ⟨257⟩, ⟨45130⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨46968⟩⟩] } }
def leftRaw : List Term := Proof.Events870.exact222870RawTerms
def rightRaw : List Term := Proof.Events870.exact222827RawTerms
def group : MergeGroup := .operator 222870 222827
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 222870) (leftOrdinal := 1)
    (rightResult := 222827) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14766⟩⟩, ⟨.program ⟨257⟩, ⟨45130⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨46968⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨14766⟩⟩, ⟨.program ⟨257⟩, ⟨45130⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨46968⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge222875

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
