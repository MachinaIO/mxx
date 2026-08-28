import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge122754
def owner : Owner := ⟨.program ⟨257⟩, ⟨30556⟩⟩
def mergeEvent : Nat := 122754
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩, ⟨.program ⟨257⟩, ⟨30555⟩⟩] } }
def leftRaw : List Term := Proof.Events479.exact122745RawTerms
def rightRaw : List Term := Proof.Events479.exact122681RawTerms
def group : MergeGroup := .operator 122745 122681
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 122745) (leftOrdinal := 0)
    (rightResult := 122681) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨30555⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩, ⟨.program ⟨257⟩, ⟨30555⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge122754

namespace LeftMerge122768
def owner : Owner := ⟨.program ⟨257⟩, ⟨29492⟩⟩
def mergeEvent : Nat := 122768
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29489⟩⟩] } }
def leftRaw : List Term := Proof.Events468.exact119870RawTerms
def rightRaw : List Term := Proof.Events479.exact122762RawTerms
def group : MergeGroup := .operator 119870 122762
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 119870) (leftOrdinal := 0)
    (rightResult := 122762) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨29489⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29489⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge122768

namespace LeftMerge122847
def owner : Owner := ⟨.program ⟨257⟩, ⟨28679⟩⟩
def mergeEvent : Nat := 122847
def frameStart : Nat := 122817
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨13221⟩⟩, ⟨.program ⟨257⟩, ⟨28678⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events479.exact122843RawTerms
def rightRaw : List Term := Proof.Events479.exact122840RawTerms
def group : MergeGroup := .operator 122843 122840
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 122843) (leftOrdinal := 0)
    (rightResult := 122840) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13221⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨28678⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨13221⟩⟩, ⟨.program ⟨257⟩, ⟨28678⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge122847

namespace LeftMerge122877
def owner : Owner := ⟨.program ⟨257⟩, ⟨30352⟩⟩
def mergeEvent : Nat := 122877
def frameStart : Nat := 122817
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨13221⟩⟩, ⟨.program ⟨257⟩, ⟨28678⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events479.exact122873RawTerms
def rightRaw : List Term := Proof.Events479.exact122871RawTerms
def group : MergeGroup := .operator 122873 122871
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 122873) (leftOrdinal := 0)
    (rightResult := 122871) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13221⟩⟩, ⟨.program ⟨257⟩, ⟨28678⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨13221⟩⟩, ⟨.program ⟨257⟩, ⟨28678⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge122877

namespace LeftMerge122900
def owner : Owner := ⟨.program ⟨257⟩, ⟨9549⟩⟩
def mergeEvent : Nat := 122900
def frameStart : Nat := 122817
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩] } }
def leftRaw : List Term := Proof.Events480.exact122896RawTerms
def rightRaw : List Term := Proof.Events480.exact122893RawTerms
def group : MergeGroup := .operator 122896 122893
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 122896) (leftOrdinal := 0)
    (rightResult := 122893) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7296⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9547⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge122900

namespace LeftMerge122909
def owner : Owner := ⟨.program ⟨257⟩, ⟨30558⟩⟩
def mergeEvent : Nat := 122909
def frameStart : Nat := 122817
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩, ⟨.program ⟨257⟩, ⟨30555⟩⟩] } }
def leftRaw : List Term := Proof.Events480.exact122905RawTerms
def rightRaw : List Term := Proof.Events479.exact122862RawTerms
def group : MergeGroup := .operator 122905 122862
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 122905) (leftOrdinal := 0)
    (rightResult := 122862) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨30555⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩, ⟨.program ⟨257⟩, ⟨30555⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge122909

namespace LeftMerge122910
def owner : Owner := ⟨.program ⟨257⟩, ⟨30558⟩⟩
def mergeEvent : Nat := 122910
def frameStart : Nat := 122817
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨13221⟩⟩, ⟨.program ⟨257⟩, ⟨28678⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨30555⟩⟩] } }
def leftRaw : List Term := Proof.Events480.exact122905RawTerms
def rightRaw : List Term := Proof.Events479.exact122862RawTerms
def group : MergeGroup := .operator 122905 122862
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 122905) (leftOrdinal := 1)
    (rightResult := 122862) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13221⟩⟩, ⟨.program ⟨257⟩, ⟨28678⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨30555⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨13221⟩⟩, ⟨.program ⟨257⟩, ⟨28678⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨30555⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge122910

namespace LeftMerge122912
def owner : Owner := ⟨.program ⟨257⟩, ⟨30558⟩⟩
def mergeEvent : Nat := 122912
def frameStart : Nat := 122817
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨13221⟩⟩, ⟨.program ⟨257⟩, ⟨28678⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨30065⟩⟩] } }
def rhsRaw : List Term := Proof.Events479.exact122859RawTerms
def group : MergeGroup := .relation 122911
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 122911) (rhsResult := 122859)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨13221⟩⟩, ⟨.program ⟨257⟩, ⟨28678⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨30555⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨30555⟩⟩) ⟨30065⟩ 122859) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨30065⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨13221⟩⟩, ⟨.program ⟨257⟩, ⟨28678⟩⟩], [⟨.program ⟨257⟩, ⟨30065⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge122912

namespace LeftMerge122920
def owner : Owner := ⟨.program ⟨257⟩, ⟨29058⟩⟩
def mergeEvent : Nat := 122920
def frameStart : Nat := 122817
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨29056⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events479.exact122873RawTerms
def rightRaw : List Term := Proof.Events480.exact122916RawTerms
def group : MergeGroup := .operator 122873 122916
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 122873) (leftOrdinal := 0)
    (rightResult := 122916) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨29056⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨29056⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge122920

namespace LeftMerge122937
def owner : Owner := ⟨.program ⟨257⟩, ⟨29492⟩⟩
def mergeEvent : Nat := 122937
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7190⟩⟩] } }
def rhsRaw : List Term := Proof.Events480.exact122934RawTerms
def group : MergeGroup := .relation 122936
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 122936) (rhsResult := 122934)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29489⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 122935 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29489⟩⟩]⟩) (none) 122934) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7190⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7190⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge122937

namespace LeftMerge122938
def owner : Owner := ⟨.program ⟨257⟩, ⟨29492⟩⟩
def mergeEvent : Nat := 122938
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩, ⟨.program ⟨257⟩, ⟨30555⟩⟩] } }
def rhsRaw : List Term := Proof.Events480.exact122934RawTerms
def group : MergeGroup := .relation 122936
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 122936) (rhsResult := 122934)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29489⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 122935 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29489⟩⟩]⟩) (none) 122934) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩, ⟨.program ⟨257⟩, ⟨30555⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩, ⟨.program ⟨257⟩, ⟨30555⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge122938

namespace LeftMerge122939
def owner : Owner := ⟨.program ⟨257⟩, ⟨29492⟩⟩
def mergeEvent : Nat := 122939
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨13221⟩⟩, ⟨.program ⟨257⟩, ⟨28678⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨30065⟩⟩] } }
def rhsRaw : List Term := Proof.Events480.exact122934RawTerms
def group : MergeGroup := .relation 122936
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 122936) (rhsResult := 122934)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29489⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 122935 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29489⟩⟩]⟩) (none) 122934) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13221⟩⟩, ⟨.program ⟨257⟩, ⟨28678⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨30065⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨13221⟩⟩, ⟨.program ⟨257⟩, ⟨28678⟩⟩], [⟨.program ⟨257⟩, ⟨30065⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge122939

namespace LeftMerge122940
def owner : Owner := ⟨.program ⟨257⟩, ⟨29492⟩⟩
def mergeEvent : Nat := 122940
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨29056⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events480.exact122934RawTerms
def group : MergeGroup := .relation 122936
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 122936) (rhsResult := 122934)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29489⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 122935 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29489⟩⟩]⟩) (none) 122934) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨29056⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨29056⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge122940

namespace LeftMerge122945
def owner : Owner := ⟨.program ⟨257⟩, ⟨30557⟩⟩
def mergeEvent : Nat := 122945
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨13221⟩⟩, ⟨.program ⟨257⟩, ⟨28678⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨30065⟩⟩] } }
def leftRaw : List Term := Proof.Events480.exact122941RawTerms
def rightRaw : List Term := Proof.Events479.exact122755RawTerms
def group : MergeGroup := .operator 122941 122755
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 122941) (leftOrdinal := 2)
    (rightResult := 122755) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨13221⟩⟩, ⟨.program ⟨257⟩, ⟨28678⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨30065⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨13221⟩⟩, ⟨.program ⟨257⟩, ⟨28678⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨30065⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨13221⟩⟩, ⟨.program ⟨257⟩, ⟨28678⟩⟩], [⟨.program ⟨257⟩, ⟨30065⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge122945

namespace LeftMerge122946
def owner : Owner := ⟨.program ⟨257⟩, ⟨30557⟩⟩
def mergeEvent : Nat := 122946
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩, ⟨.program ⟨257⟩, ⟨30555⟩⟩] } }
def leftRaw : List Term := Proof.Events480.exact122941RawTerms
def rightRaw : List Term := Proof.Events479.exact122755RawTerms
def group : MergeGroup := .operator 122941 122755
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 122941) (leftOrdinal := 1)
    (rightResult := 122755) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩, ⟨.program ⟨257⟩, ⟨30555⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩, ⟨.program ⟨257⟩, ⟨30555⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩, ⟨.program ⟨257⟩, ⟨30555⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge122946

namespace LeftMerge122954
def owner : Owner := ⟨.program ⟨257⟩, ⟨30871⟩⟩
def mergeEvent : Nat := 122954
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨30869⟩⟩] } }
def leftRaw : List Term := Proof.Events480.exact122948RawTerms
def rightRaw : List Term := Proof.Events479.exact122671RawTerms
def group : MergeGroup := .operator 122948 122671
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 122948) (leftOrdinal := 0)
    (rightResult := 122671) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7190⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨30869⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨30869⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge122954

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
