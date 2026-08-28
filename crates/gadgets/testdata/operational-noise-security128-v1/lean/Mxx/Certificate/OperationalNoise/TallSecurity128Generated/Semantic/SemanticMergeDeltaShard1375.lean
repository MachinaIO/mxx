import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge223080
def owner : Owner := ⟨.program ⟨257⟩, ⟨45671⟩⟩
def mergeEvent : Nat := 223080
def frameStart : Nat := 222991
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨45670⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events871.exact223053RawTerms
def rightRaw : List Term := Proof.Events871.exact223076RawTerms
def group : MergeGroup := .operator 223053 223076
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 223053) (leftOrdinal := 0)
    (rightResult := 223076) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨45670⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨45670⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge223080

namespace LeftMerge223097
def owner : Owner := ⟨.program ⟨257⟩, ⟨46199⟩⟩
def mergeEvent : Nat := 223097
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7230⟩⟩] } }
def rhsRaw : List Term := Proof.Events871.exact223094RawTerms
def group : MergeGroup := .relation 223096
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 223096) (rhsResult := 223094)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46196⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 223095 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46196⟩⟩]⟩) (none) 223094) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7230⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7230⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge223097

namespace LeftMerge223098
def owner : Owner := ⟨.program ⟨257⟩, ⟨46199⟩⟩
def mergeEvent : Nat := 223098
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47324⟩⟩] } }
def rhsRaw : List Term := Proof.Events871.exact223094RawTerms
def group : MergeGroup := .relation 223096
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 223096) (rhsResult := 223094)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46196⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 223095 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46196⟩⟩]⟩) (none) 223094) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47324⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47324⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge223098

namespace LeftMerge223099
def owner : Owner := ⟨.program ⟨257⟩, ⟨46199⟩⟩
def mergeEvent : Nat := 223099
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨45460⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46612⟩⟩] } }
def rhsRaw : List Term := Proof.Events871.exact223094RawTerms
def group : MergeGroup := .relation 223096
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 223096) (rhsResult := 223094)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46196⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 223095 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46196⟩⟩]⟩) (none) 223094) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨45460⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46612⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨45460⟩⟩], [⟨.program ⟨257⟩, ⟨46612⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge223099

namespace LeftMerge223100
def owner : Owner := ⟨.program ⟨257⟩, ⟨46199⟩⟩
def mergeEvent : Nat := 223100
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨45670⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events871.exact223094RawTerms
def group : MergeGroup := .relation 223096
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 223096) (rhsResult := 223094)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46196⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 223095 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46196⟩⟩]⟩) (none) 223094) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨45670⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨45670⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge223100

namespace LeftMerge223105
def owner : Owner := ⟨.program ⟨257⟩, ⟨47327⟩⟩
def mergeEvent : Nat := 223105
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47324⟩⟩] } }
def leftRaw : List Term := Proof.Events871.exact223101RawTerms
def rightRaw : List Term := Proof.Events870.exact222923RawTerms
def group : MergeGroup := .operator 223101 222923
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 223101) (leftOrdinal := 0)
    (rightResult := 222923) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47324⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47324⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47324⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge223105

namespace LeftMerge223106
def owner : Owner := ⟨.program ⟨257⟩, ⟨47327⟩⟩
def mergeEvent : Nat := 223106
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨45460⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46612⟩⟩] } }
def leftRaw : List Term := Proof.Events871.exact223101RawTerms
def rightRaw : List Term := Proof.Events870.exact222923RawTerms
def group : MergeGroup := .operator 223101 222923
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 223101) (leftOrdinal := 2)
    (rightResult := 222923) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨45460⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46612⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨45460⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46612⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨45460⟩⟩], [⟨.program ⟨257⟩, ⟨46612⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge223106

namespace LeftMerge223132
def owner : Owner := ⟨.program ⟨257⟩, ⟨42453⟩⟩
def mergeEvent : Nat := 223132
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨42450⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events041.exact10612RawTerms
def rightRaw : List Term := Proof.Events867.exact222153RawTerms
def group : MergeGroup := .operator 10612 222153
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 10612) (leftOrdinal := 0)
    (rightResult := 222153) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨42450⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨42450⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge223132

namespace LeftMerge223137
def owner : Owner := ⟨.program ⟨257⟩, ⟨8475⟩⟩
def mergeEvent : Nat := 223137
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7283⟩⟩] } }
def leftRaw : List Term := Proof.Events867.exact222023RawTerms
def rightRaw : List Term := Proof.Events070.exact18082RawTerms
def group : MergeGroup := .operator 222023 18082
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 222023) (leftOrdinal := 0)
    (rightResult := 18082) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7283⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7283⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge223137

namespace LeftMerge223154
def owner : Owner := ⟨.program ⟨257⟩, ⟨42456⟩⟩
def mergeEvent : Nat := 223154
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨14466⟩⟩, ⟨.program ⟨257⟩, ⟨42450⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events871.exact223148RawTerms
def rightRaw : List Term := Proof.Events041.exact10615RawTerms
def group : MergeGroup := .operator 223148 10615
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 223148) (leftOrdinal := 1)
    (rightResult := 10615) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨42450⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14466⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨14466⟩⟩, ⟨.program ⟨257⟩, ⟨42450⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge223154

namespace LeftMerge223155
def owner : Owner := ⟨.program ⟨257⟩, ⟨42456⟩⟩
def mergeEvent : Nat := 223155
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨14466⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7283⟩⟩] } }
def leftRaw : List Term := Proof.Events871.exact223148RawTerms
def rightRaw : List Term := Proof.Events041.exact10615RawTerms
def group : MergeGroup := .operator 223148 10615
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 223148) (leftOrdinal := 0)
    (rightResult := 10615) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7283⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14466⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨14466⟩⟩], [⟨.program ⟨257⟩, ⟨7283⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge223155

namespace LeftMerge223160
def owner : Owner := ⟨.program ⟨257⟩, ⟨14467⟩⟩
def mergeEvent : Nat := 223160
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨14466⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events041.exact10615RawTerms
def rightRaw : List Term := Proof.Events867.exact222153RawTerms
def group : MergeGroup := .operator 10615 222153
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 10615) (leftOrdinal := 0)
    (rightResult := 222153) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14466⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨14466⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge223160

namespace LeftMerge223165
def owner : Owner := ⟨.program ⟨257⟩, ⟨8492⟩⟩
def mergeEvent : Nat := 223165
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7300⟩⟩] } }
def leftRaw : List Term := Proof.Events867.exact222023RawTerms
def rightRaw : List Term := Proof.Events070.exact18123RawTerms
def group : MergeGroup := .operator 222023 18123
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 222023) (leftOrdinal := 0)
    (rightResult := 18123) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7300⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7300⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge223165

namespace LeftMerge223182
def owner : Owner := ⟨.program ⟨257⟩, ⟨14470⟩⟩
def mergeEvent : Nat := 223182
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨14466⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩] } }
def leftRaw : List Term := Proof.Events871.exact223176RawTerms
def rightRaw : List Term := Proof.Events070.exact18112RawTerms
def group : MergeGroup := .operator 223176 18112
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 223176) (leftOrdinal := 1)
    (rightResult := 18112) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨14466⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9559⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨14466⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge223182

namespace LeftMerge223184
def owner : Owner := ⟨.program ⟨257⟩, ⟨14470⟩⟩
def mergeEvent : Nat := 223184
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨14466⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7283⟩⟩] } }
def rhsRaw : List Term := Proof.Events070.exact18082RawTerms
def group : MergeGroup := .relation 223183
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 223183) (rhsResult := 18082)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨14466⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨9559⟩⟩) ⟨7283⟩ 18082) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7283⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨14466⟩⟩], [⟨.program ⟨257⟩, ⟨7283⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge223184

namespace LeftMerge223185
def owner : Owner := ⟨.program ⟨257⟩, ⟨14470⟩⟩
def mergeEvent : Nat := 223185
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩] } }
def leftRaw : List Term := Proof.Events871.exact223176RawTerms
def rightRaw : List Term := Proof.Events070.exact18112RawTerms
def group : MergeGroup := .operator 223176 18112
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 223176) (leftOrdinal := 0)
    (rightResult := 18112) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7300⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9559⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge223185

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
