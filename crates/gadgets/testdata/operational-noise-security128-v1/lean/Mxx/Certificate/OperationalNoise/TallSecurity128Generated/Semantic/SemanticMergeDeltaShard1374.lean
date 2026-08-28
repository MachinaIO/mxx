import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge222877
def owner : Owner := ⟨.program ⟨257⟩, ⟨46971⟩⟩
def mergeEvent : Nat := 222877
def frameStart : Nat := 222782
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14766⟩⟩, ⟨.program ⟨257⟩, ⟨45130⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46463⟩⟩] } }
def rhsRaw : List Term := Proof.Events870.exact222824RawTerms
def group : MergeGroup := .relation 222876
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 222876) (rhsResult := 222824)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨14766⟩⟩, ⟨.program ⟨257⟩, ⟨45130⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨46968⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨46968⟩⟩) ⟨46463⟩ 222824) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨46463⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨14766⟩⟩, ⟨.program ⟨257⟩, ⟨45130⟩⟩], [⟨.program ⟨257⟩, ⟨46463⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge222877

namespace LeftMerge222885
def owner : Owner := ⟨.program ⟨257⟩, ⟨45462⟩⟩
def mergeEvent : Nat := 222885
def frameStart : Nat := 222782
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨45460⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events870.exact222838RawTerms
def rightRaw : List Term := Proof.Events870.exact222881RawTerms
def group : MergeGroup := .operator 222838 222881
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 222838) (leftOrdinal := 0)
    (rightResult := 222881) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨45460⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨45460⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge222885

namespace LeftMerge222902
def owner : Owner := ⟨.program ⟨257⟩, ⟨45902⟩⟩
def mergeEvent : Nat := 222902
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7195⟩⟩] } }
def rhsRaw : List Term := Proof.Events870.exact222899RawTerms
def group : MergeGroup := .relation 222901
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 222901) (rhsResult := 222899)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨45899⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 222900 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨45899⟩⟩]⟩) (none) 222899) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7195⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7195⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge222902

namespace LeftMerge222903
def owner : Owner := ⟨.program ⟨257⟩, ⟨45902⟩⟩
def mergeEvent : Nat := 222903
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨46968⟩⟩] } }
def rhsRaw : List Term := Proof.Events870.exact222899RawTerms
def group : MergeGroup := .relation 222901
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 222901) (rhsResult := 222899)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨45899⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 222900 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨45899⟩⟩]⟩) (none) 222899) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨46968⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨46968⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge222903

namespace LeftMerge222904
def owner : Owner := ⟨.program ⟨257⟩, ⟨45902⟩⟩
def mergeEvent : Nat := 222904
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨14766⟩⟩, ⟨.program ⟨257⟩, ⟨45130⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46463⟩⟩] } }
def rhsRaw : List Term := Proof.Events870.exact222899RawTerms
def group : MergeGroup := .relation 222901
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 222901) (rhsResult := 222899)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨45899⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 222900 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨45899⟩⟩]⟩) (none) 222899) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14766⟩⟩, ⟨.program ⟨257⟩, ⟨45130⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46463⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨14766⟩⟩, ⟨.program ⟨257⟩, ⟨45130⟩⟩], [⟨.program ⟨257⟩, ⟨46463⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge222904

namespace LeftMerge222905
def owner : Owner := ⟨.program ⟨257⟩, ⟨45902⟩⟩
def mergeEvent : Nat := 222905
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨45460⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events870.exact222899RawTerms
def group : MergeGroup := .relation 222901
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 222901) (rhsResult := 222899)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨45899⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 222900 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨45899⟩⟩]⟩) (none) 222899) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨45460⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨45460⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge222905

namespace LeftMerge222910
def owner : Owner := ⟨.program ⟨257⟩, ⟨46970⟩⟩
def mergeEvent : Nat := 222910
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨14766⟩⟩, ⟨.program ⟨257⟩, ⟨45130⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46463⟩⟩] } }
def leftRaw : List Term := Proof.Events870.exact222906RawTerms
def rightRaw : List Term := Proof.Events870.exact222720RawTerms
def group : MergeGroup := .operator 222906 222720
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 222906) (leftOrdinal := 2)
    (rightResult := 222720) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨14766⟩⟩, ⟨.program ⟨257⟩, ⟨45130⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46463⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨14766⟩⟩, ⟨.program ⟨257⟩, ⟨45130⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46463⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨14766⟩⟩, ⟨.program ⟨257⟩, ⟨45130⟩⟩], [⟨.program ⟨257⟩, ⟨46463⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge222910

namespace LeftMerge222911
def owner : Owner := ⟨.program ⟨257⟩, ⟨46970⟩⟩
def mergeEvent : Nat := 222911
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨46968⟩⟩] } }
def leftRaw : List Term := Proof.Events870.exact222906RawTerms
def rightRaw : List Term := Proof.Events870.exact222720RawTerms
def group : MergeGroup := .operator 222906 222720
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 222906) (leftOrdinal := 1)
    (rightResult := 222720) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨46968⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨46968⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨46968⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge222911

namespace LeftMerge222919
def owner : Owner := ⟨.program ⟨257⟩, ⟨47326⟩⟩
def mergeEvent : Nat := 222919
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47324⟩⟩] } }
def leftRaw : List Term := Proof.Events870.exact222913RawTerms
def rightRaw : List Term := Proof.Events869.exact222636RawTerms
def group : MergeGroup := .operator 222913 222636
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 222913) (leftOrdinal := 0)
    (rightResult := 222636) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7195⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨47324⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47324⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge222919

namespace LeftMerge222920
def owner : Owner := ⟨.program ⟨257⟩, ⟨47326⟩⟩
def mergeEvent : Nat := 222920
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨45460⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨47324⟩⟩] } }
def leftRaw : List Term := Proof.Events870.exact222913RawTerms
def rightRaw : List Term := Proof.Events869.exact222636RawTerms
def group : MergeGroup := .operator 222913 222636
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 222913) (leftOrdinal := 1)
    (rightResult := 222636) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨45460⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨47324⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨45460⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨47324⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge222920

namespace LeftMerge222922
def owner : Owner := ⟨.program ⟨257⟩, ⟨47326⟩⟩
def mergeEvent : Nat := 222922
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨45460⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46612⟩⟩] } }
def rhsRaw : List Term := Proof.Events869.exact222633RawTerms
def group : MergeGroup := .relation 222921
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 222921) (rhsResult := 222633)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨45460⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨47324⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨47324⟩⟩) ⟨46612⟩ 222633) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨46612⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨45460⟩⟩], [⟨.program ⟨257⟩, ⟨46612⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge222922

namespace LeftMerge222936
def owner : Owner := ⟨.program ⟨257⟩, ⟨46199⟩⟩
def mergeEvent : Nat := 222936
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46196⟩⟩] } }
def leftRaw : List Term := Proof.Events868.exact222245RawTerms
def rightRaw : List Term := Proof.Events870.exact222930RawTerms
def group : MergeGroup := .operator 222245 222930
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 222245) (leftOrdinal := 0)
    (rightResult := 222930) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨46196⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46196⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge222936

namespace LeftMerge223057
def owner : Owner := ⟨.program ⟨257⟩, ⟨46824⟩⟩
def mergeEvent : Nat := 223057
def frameStart : Nat := 222991
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨45460⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events871.exact223053RawTerms
def rightRaw : List Term := Proof.Events871.exact223051RawTerms
def group : MergeGroup := .operator 223053 223051
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 223053) (leftOrdinal := 0)
    (rightResult := 223051) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨45460⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨45460⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge223057

namespace LeftMerge223069
def owner : Owner := ⟨.program ⟨257⟩, ⟨47325⟩⟩
def mergeEvent : Nat := 223069
def frameStart : Nat := 222991
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47324⟩⟩] } }
def leftRaw : List Term := Proof.Events871.exact223065RawTerms
def rightRaw : List Term := Proof.Events871.exact223042RawTerms
def group : MergeGroup := .operator 223065 223042
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 223065) (leftOrdinal := 0)
    (rightResult := 223042) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7195⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨47324⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47324⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge223069

namespace LeftMerge223070
def owner : Owner := ⟨.program ⟨257⟩, ⟨47325⟩⟩
def mergeEvent : Nat := 223070
def frameStart : Nat := 222991
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨45460⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨47324⟩⟩] } }
def leftRaw : List Term := Proof.Events871.exact223065RawTerms
def rightRaw : List Term := Proof.Events871.exact223042RawTerms
def group : MergeGroup := .operator 223065 223042
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 223065) (leftOrdinal := 1)
    (rightResult := 223042) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨45460⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨47324⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨45460⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨47324⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge223070

namespace LeftMerge223072
def owner : Owner := ⟨.program ⟨257⟩, ⟨47325⟩⟩
def mergeEvent : Nat := 223072
def frameStart : Nat := 222991
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨45460⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46612⟩⟩] } }
def rhsRaw : List Term := Proof.Events871.exact223039RawTerms
def group : MergeGroup := .relation 223071
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 223071) (rhsResult := 223039)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨45460⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨47324⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨47324⟩⟩) ⟨46612⟩ 223039) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨46612⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨45460⟩⟩], [⟨.program ⟨257⟩, ⟨46612⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge223072

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
