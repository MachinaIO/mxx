import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge18628
def owner : Owner := ⟨.program ⟨257⟩, ⟨7617⟩⟩
def mergeEvent : Nat := 18628
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩] } }
def leftRaw : List Term := Proof.Events066.exact16922RawTerms
def rightRaw : List Term := Proof.Events072.exact18624RawTerms
def group : MergeGroup := .operator 16922 18624
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 16922) (leftOrdinal := 0)
    (rightResult := 18624) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7299⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge18628

namespace LeftMerge18645
def owner : Owner := ⟨.program ⟨257⟩, ⟨14055⟩⟩
def mergeEvent : Nat := 18645
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨14051⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩] } }
def leftRaw : List Term := Proof.Events072.exact18639RawTerms
def rightRaw : List Term := Proof.Events072.exact18613RawTerms
def group : MergeGroup := .operator 18639 18613
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 18639) (leftOrdinal := 1)
    (rightResult := 18613) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨14051⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9556⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨14051⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge18645

namespace LeftMerge18647
def owner : Owner := ⟨.program ⟨257⟩, ⟨14055⟩⟩
def mergeEvent : Nat := 18647
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨14051⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7282⟩⟩] } }
def rhsRaw : List Term := Proof.Events072.exact18583RawTerms
def group : MergeGroup := .relation 18646
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 18646) (rhsResult := 18583)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨14051⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨9556⟩⟩) ⟨7282⟩ 18583) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7282⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨14051⟩⟩], [⟨.program ⟨257⟩, ⟨7282⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge18647

namespace LeftMerge18648
def owner : Owner := ⟨.program ⟨257⟩, ⟨14055⟩⟩
def mergeEvent : Nat := 18648
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩] } }
def leftRaw : List Term := Proof.Events072.exact18639RawTerms
def rightRaw : List Term := Proof.Events072.exact18613RawTerms
def group : MergeGroup := .operator 18639 18613
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 18639) (leftOrdinal := 0)
    (rightResult := 18613) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9556⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge18648

namespace LeftMerge18653
def owner : Owner := ⟨.program ⟨257⟩, ⟨39593⟩⟩
def mergeEvent : Nat := 18653
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨14051⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7282⟩⟩] } }
def leftRaw : List Term := Proof.Events072.exact18649RawTerms
def rightRaw : List Term := Proof.Events072.exact18606RawTerms
def group : MergeGroup := .operator 18649 18606
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 18649) (leftOrdinal := 1)
    (rightResult := 18606) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨14051⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7282⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨14051⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7282⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨14051⟩⟩], [⟨.program ⟨257⟩, ⟨7282⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge18653

namespace LeftMerge18661
def owner : Owner := ⟨.program ⟨257⟩, ⟨41524⟩⟩
def mergeEvent : Nat := 18661
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨14051⟩⟩, ⟨.program ⟨257⟩, ⟨39586⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨41523⟩⟩] } }
def leftRaw : List Term := Proof.Events072.exact18655RawTerms
def rightRaw : List Term := Proof.Events072.exact18572RawTerms
def group : MergeGroup := .operator 18655 18572
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 18655) (leftOrdinal := 1)
    (rightResult := 18572) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨14051⟩⟩, ⟨.program ⟨257⟩, ⟨39586⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨41523⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨14051⟩⟩, ⟨.program ⟨257⟩, ⟨39586⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨41523⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge18661

namespace LeftMerge18663
def owner : Owner := ⟨.program ⟨257⟩, ⟨41524⟩⟩
def mergeEvent : Nat := 18663
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨14051⟩⟩, ⟨.program ⟨257⟩, ⟨39586⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41057⟩⟩] } }
def rhsRaw : List Term := Proof.Events072.exact18569RawTerms
def group : MergeGroup := .relation 18662
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 18662) (rhsResult := 18569)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨14051⟩⟩, ⟨.program ⟨257⟩, ⟨39586⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨41523⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨41523⟩⟩) ⟨41057⟩ 18569) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨41057⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨14051⟩⟩, ⟨.program ⟨257⟩, ⟨39586⟩⟩], [⟨.program ⟨257⟩, ⟨41057⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge18663

namespace LeftMerge18664
def owner : Owner := ⟨.program ⟨257⟩, ⟨41524⟩⟩
def mergeEvent : Nat := 18664
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41523⟩⟩] } }
def leftRaw : List Term := Proof.Events072.exact18655RawTerms
def rightRaw : List Term := Proof.Events072.exact18572RawTerms
def group : MergeGroup := .operator 18655 18572
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 18655) (leftOrdinal := 0)
    (rightResult := 18572) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨41523⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41523⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge18664

namespace LeftMerge18678
def owner : Owner := ⟨.program ⟨257⟩, ⟨40465⟩⟩
def mergeEvent : Nat := 18678
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40462⟩⟩] } }
def leftRaw : List Term := Proof.Events067.exact17169RawTerms
def rightRaw : List Term := Proof.Events072.exact18672RawTerms
def group : MergeGroup := .operator 17169 18672
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 17169) (leftOrdinal := 0)
    (rightResult := 18672) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨40462⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40462⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge18678

namespace LeftMerge18757
def owner : Owner := ⟨.program ⟨257⟩, ⟨39587⟩⟩
def mergeEvent : Nat := 18757
def frameStart : Nat := 18727
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14051⟩⟩, ⟨.program ⟨257⟩, ⟨39586⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events073.exact18753RawTerms
def rightRaw : List Term := Proof.Events073.exact18750RawTerms
def group : MergeGroup := .operator 18753 18750
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 18753) (leftOrdinal := 0)
    (rightResult := 18750) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14051⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨39586⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨14051⟩⟩, ⟨.program ⟨257⟩, ⟨39586⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge18757

namespace LeftMerge18787
def owner : Owner := ⟨.program ⟨257⟩, ⟨41352⟩⟩
def mergeEvent : Nat := 18787
def frameStart : Nat := 18727
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14051⟩⟩, ⟨.program ⟨257⟩, ⟨39586⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events073.exact18783RawTerms
def rightRaw : List Term := Proof.Events073.exact18781RawTerms
def group : MergeGroup := .operator 18783 18781
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 18783) (leftOrdinal := 0)
    (rightResult := 18781) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14051⟩⟩, ⟨.program ⟨257⟩, ⟨39586⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨14051⟩⟩, ⟨.program ⟨257⟩, ⟨39586⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge18787

namespace LeftMerge18810
def owner : Owner := ⟨.program ⟨257⟩, ⟨9558⟩⟩
def mergeEvent : Nat := 18810
def frameStart : Nat := 18727
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩] } }
def leftRaw : List Term := Proof.Events073.exact18806RawTerms
def rightRaw : List Term := Proof.Events073.exact18803RawTerms
def group : MergeGroup := .operator 18806 18803
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 18806) (leftOrdinal := 0)
    (rightResult := 18803) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9556⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge18810

namespace LeftMerge18819
def owner : Owner := ⟨.program ⟨257⟩, ⟨41526⟩⟩
def mergeEvent : Nat := 18819
def frameStart : Nat := 18727
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14051⟩⟩, ⟨.program ⟨257⟩, ⟨39586⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨41523⟩⟩] } }
def leftRaw : List Term := Proof.Events073.exact18815RawTerms
def rightRaw : List Term := Proof.Events073.exact18772RawTerms
def group : MergeGroup := .operator 18815 18772
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 18815) (leftOrdinal := 1)
    (rightResult := 18772) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14051⟩⟩, ⟨.program ⟨257⟩, ⟨39586⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨41523⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨14051⟩⟩, ⟨.program ⟨257⟩, ⟨39586⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨41523⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge18819

namespace LeftMerge18821
def owner : Owner := ⟨.program ⟨257⟩, ⟨41526⟩⟩
def mergeEvent : Nat := 18821
def frameStart : Nat := 18727
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14051⟩⟩, ⟨.program ⟨257⟩, ⟨39586⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41057⟩⟩] } }
def rhsRaw : List Term := Proof.Events073.exact18769RawTerms
def group : MergeGroup := .relation 18820
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 18820) (rhsResult := 18769)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨14051⟩⟩, ⟨.program ⟨257⟩, ⟨39586⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨41523⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨41523⟩⟩) ⟨41057⟩ 18769) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨41057⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨14051⟩⟩, ⟨.program ⟨257⟩, ⟨39586⟩⟩], [⟨.program ⟨257⟩, ⟨41057⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge18821

namespace LeftMerge18822
def owner : Owner := ⟨.program ⟨257⟩, ⟨41526⟩⟩
def mergeEvent : Nat := 18822
def frameStart : Nat := 18727
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41523⟩⟩] } }
def leftRaw : List Term := Proof.Events073.exact18815RawTerms
def rightRaw : List Term := Proof.Events073.exact18772RawTerms
def group : MergeGroup := .operator 18815 18772
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 18815) (leftOrdinal := 0)
    (rightResult := 18772) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨41523⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41523⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge18822

namespace LeftMerge18830
def owner : Owner := ⟨.program ⟨257⟩, ⟨40040⟩⟩
def mergeEvent : Nat := 18830
def frameStart : Nat := 18727
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨40038⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events073.exact18783RawTerms
def rightRaw : List Term := Proof.Events073.exact18826RawTerms
def group : MergeGroup := .operator 18783 18826
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 18783) (leftOrdinal := 0)
    (rightResult := 18826) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨40038⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨40038⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge18830

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
