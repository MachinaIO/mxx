import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge27922
def owner : Owner := ⟨.program ⟨257⟩, ⟨47127⟩⟩
def mergeEvent : Nat := 27922
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨45398⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨47125⟩⟩] } }
def leftRaw : List Term := Proof.Events069.exact17856RawTerms
def rightRaw : List Term := Proof.Events109.exact27916RawTerms
def group : MergeGroup := .operator 17856 27916
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 17856) (leftOrdinal := 1)
    (rightResult := 27916) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨45398⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨47125⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨45398⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨47125⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge27922

namespace LeftMerge27924
def owner : Owner := ⟨.program ⟨257⟩, ⟨47127⟩⟩
def mergeEvent : Nat := 27924
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨45398⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46542⟩⟩] } }
def rhsRaw : List Term := Proof.Events109.exact27913RawTerms
def group : MergeGroup := .relation 27923
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 27923) (rhsResult := 27913)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨45398⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨47125⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨47125⟩⟩) ⟨46542⟩ 27913) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨46542⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨45398⟩⟩], [⟨.program ⟨257⟩, ⟨46542⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge27924

namespace LeftMerge27925
def owner : Owner := ⟨.program ⟨257⟩, ⟨47127⟩⟩
def mergeEvent : Nat := 27925
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47125⟩⟩] } }
def leftRaw : List Term := Proof.Events069.exact17856RawTerms
def rightRaw : List Term := Proof.Events109.exact27916RawTerms
def group : MergeGroup := .operator 17856 27916
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 17856) (leftOrdinal := 0)
    (rightResult := 27916) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7195⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨47125⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47125⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge27925

namespace LeftMerge27939
def owner : Owner := ⟨.program ⟨257⟩, ⟨46041⟩⟩
def mergeEvent : Nat := 27939
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46038⟩⟩] } }
def leftRaw : List Term := Proof.Events067.exact17169RawTerms
def rightRaw : List Term := Proof.Events109.exact27933RawTerms
def group : MergeGroup := .operator 17169 27933
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 17169) (leftOrdinal := 0)
    (rightResult := 27933) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨46038⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46038⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge27939

namespace LeftMerge28060
def owner : Owner := ⟨.program ⟨257⟩, ⟨46792⟩⟩
def mergeEvent : Nat := 28060
def frameStart : Nat := 27994
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨45398⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events109.exact28056RawTerms
def rightRaw : List Term := Proof.Events109.exact28054RawTerms
def group : MergeGroup := .operator 28056 28054
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 28056) (leftOrdinal := 0)
    (rightResult := 28054) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨45398⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨45398⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge28060

namespace LeftMerge28072
def owner : Owner := ⟨.program ⟨257⟩, ⟨47126⟩⟩
def mergeEvent : Nat := 28072
def frameStart : Nat := 27994
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨45398⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨47125⟩⟩] } }
def leftRaw : List Term := Proof.Events109.exact28068RawTerms
def rightRaw : List Term := Proof.Events109.exact28045RawTerms
def group : MergeGroup := .operator 28068 28045
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 28068) (leftOrdinal := 1)
    (rightResult := 28045) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨45398⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨47125⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨45398⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨47125⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge28072

namespace LeftMerge28074
def owner : Owner := ⟨.program ⟨257⟩, ⟨47126⟩⟩
def mergeEvent : Nat := 28074
def frameStart : Nat := 27994
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨45398⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46542⟩⟩] } }
def rhsRaw : List Term := Proof.Events109.exact28042RawTerms
def group : MergeGroup := .relation 28073
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 28073) (rhsResult := 28042)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨45398⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨47125⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨47125⟩⟩) ⟨46542⟩ 28042) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨46542⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨45398⟩⟩], [⟨.program ⟨257⟩, ⟨46542⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge28074

namespace LeftMerge28075
def owner : Owner := ⟨.program ⟨257⟩, ⟨47126⟩⟩
def mergeEvent : Nat := 28075
def frameStart : Nat := 27994
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47125⟩⟩] } }
def leftRaw : List Term := Proof.Events109.exact28068RawTerms
def rightRaw : List Term := Proof.Events109.exact28045RawTerms
def group : MergeGroup := .operator 28068 28045
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 28068) (leftOrdinal := 0)
    (rightResult := 28045) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7195⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨47125⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47125⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge28075

namespace LeftMerge28083
def owner : Owner := ⟨.program ⟨257⟩, ⟨45567⟩⟩
def mergeEvent : Nat := 28083
def frameStart : Nat := 27994
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨45565⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events109.exact28056RawTerms
def rightRaw : List Term := Proof.Events109.exact28079RawTerms
def group : MergeGroup := .operator 28056 28079
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 28056) (leftOrdinal := 0)
    (rightResult := 28079) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨45565⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨45565⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge28083

namespace LeftMerge28100
def owner : Owner := ⟨.program ⟨257⟩, ⟨46041⟩⟩
def mergeEvent : Nat := 28100
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7229⟩⟩] } }
def rhsRaw : List Term := Proof.Events109.exact28097RawTerms
def group : MergeGroup := .relation 28099
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 28099) (rhsResult := 28097)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46038⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 28098 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46038⟩⟩]⟩) (none) 28097) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7229⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7229⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge28100

namespace LeftMerge28101
def owner : Owner := ⟨.program ⟨257⟩, ⟨46041⟩⟩
def mergeEvent : Nat := 28101
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨45398⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46542⟩⟩] } }
def rhsRaw : List Term := Proof.Events109.exact28097RawTerms
def group : MergeGroup := .relation 28099
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 28099) (rhsResult := 28097)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46038⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 28098 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46038⟩⟩]⟩) (none) 28097) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨45398⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46542⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨45398⟩⟩], [⟨.program ⟨257⟩, ⟨46542⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge28101

namespace LeftMerge28102
def owner : Owner := ⟨.program ⟨257⟩, ⟨46041⟩⟩
def mergeEvent : Nat := 28102
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47125⟩⟩] } }
def rhsRaw : List Term := Proof.Events109.exact28097RawTerms
def group : MergeGroup := .relation 28099
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 28099) (rhsResult := 28097)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46038⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 28098 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46038⟩⟩]⟩) (none) 28097) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47125⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47125⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge28102

namespace LeftMerge28103
def owner : Owner := ⟨.program ⟨257⟩, ⟨46041⟩⟩
def mergeEvent : Nat := 28103
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨45565⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events109.exact28097RawTerms
def group : MergeGroup := .relation 28099
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 28099) (rhsResult := 28097)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46038⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 28098 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46038⟩⟩]⟩) (none) 28097) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨45565⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨45565⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge28103

namespace LeftMerge28108
def owner : Owner := ⟨.program ⟨257⟩, ⟨47128⟩⟩
def mergeEvent : Nat := 28108
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨45398⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46542⟩⟩] } }
def leftRaw : List Term := Proof.Events109.exact28104RawTerms
def rightRaw : List Term := Proof.Events109.exact27926RawTerms
def group : MergeGroup := .operator 28104 27926
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 28104) (leftOrdinal := 2)
    (rightResult := 27926) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨45398⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46542⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨45398⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46542⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨45398⟩⟩], [⟨.program ⟨257⟩, ⟨46542⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge28108

namespace LeftMerge28109
def owner : Owner := ⟨.program ⟨257⟩, ⟨47128⟩⟩
def mergeEvent : Nat := 28109
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47125⟩⟩] } }
def leftRaw : List Term := Proof.Events109.exact28104RawTerms
def rightRaw : List Term := Proof.Events109.exact27926RawTerms
def group : MergeGroup := .operator 28104 27926
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 28104) (leftOrdinal := 0)
    (rightResult := 27926) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47125⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47125⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47125⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge28109

namespace LeftMerge28117
def owner : Owner := ⟨.program ⟨257⟩, ⟨47129⟩⟩
def mergeEvent : Nat := 28117
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7229⟩⟩, ⟨.program ⟨257⟩, ⟨7151⟩⟩] } }
def leftRaw : List Term := Proof.Events109.exact28111RawTerms
def rightRaw : List Term := Proof.Events060.exact15562RawTerms
def group : MergeGroup := .operator 28111 15562
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 28111) (leftOrdinal := 0)
    (rightResult := 15562) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7229⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7151⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7229⟩⟩, ⟨.program ⟨257⟩, ⟨7151⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge28117

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
