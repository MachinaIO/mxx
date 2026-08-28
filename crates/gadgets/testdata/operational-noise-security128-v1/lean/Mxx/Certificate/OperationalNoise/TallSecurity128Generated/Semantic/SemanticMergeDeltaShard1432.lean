import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge232854
def owner : Owner := ⟨.program ⟨257⟩, ⟨46195⟩⟩
def mergeEvent : Nat := 232854
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47318⟩⟩] } }
def rhsRaw : List Term := Proof.Events909.exact232850RawTerms
def group : MergeGroup := .relation 232852
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 232852) (rhsResult := 232850)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46192⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 232851 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46192⟩⟩]⟩) (none) 232850) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47318⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47318⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge232854

namespace LeftMerge232855
def owner : Owner := ⟨.program ⟨257⟩, ⟨46195⟩⟩
def mergeEvent : Nat := 232855
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨45460⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46611⟩⟩] } }
def rhsRaw : List Term := Proof.Events909.exact232850RawTerms
def group : MergeGroup := .relation 232852
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 232852) (rhsResult := 232850)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46192⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 232851 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46192⟩⟩]⟩) (none) 232850) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨45460⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46611⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨45460⟩⟩], [⟨.program ⟨257⟩, ⟨46611⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge232855

namespace LeftMerge232856
def owner : Owner := ⟨.program ⟨257⟩, ⟨46195⟩⟩
def mergeEvent : Nat := 232856
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨45666⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events909.exact232850RawTerms
def group : MergeGroup := .relation 232852
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 232852) (rhsResult := 232850)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46192⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 232851 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46192⟩⟩]⟩) (none) 232850) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨45666⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨45666⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge232856

namespace LeftMerge232861
def owner : Owner := ⟨.program ⟨257⟩, ⟨47321⟩⟩
def mergeEvent : Nat := 232861
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47318⟩⟩] } }
def leftRaw : List Term := Proof.Events909.exact232857RawTerms
def rightRaw : List Term := Proof.Events908.exact232679RawTerms
def group : MergeGroup := .operator 232857 232679
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 232857) (leftOrdinal := 0)
    (rightResult := 232679) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47318⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47318⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47318⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge232861

namespace LeftMerge232862
def owner : Owner := ⟨.program ⟨257⟩, ⟨47321⟩⟩
def mergeEvent : Nat := 232862
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨45460⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46611⟩⟩] } }
def leftRaw : List Term := Proof.Events909.exact232857RawTerms
def rightRaw : List Term := Proof.Events908.exact232679RawTerms
def group : MergeGroup := .operator 232857 232679
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 232857) (leftOrdinal := 2)
    (rightResult := 232679) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨45460⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46611⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨45460⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46611⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨45460⟩⟩], [⟨.program ⟨257⟩, ⟨46611⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge232862

namespace LeftMerge232870
def owner : Owner := ⟨.program ⟨257⟩, ⟨47322⟩⟩
def mergeEvent : Nat := 232870
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7229⟩⟩, ⟨.program ⟨257⟩, ⟨7151⟩⟩] } }
def leftRaw : List Term := Proof.Events909.exact232864RawTerms
def rightRaw : List Term := Proof.Events060.exact15562RawTerms
def group : MergeGroup := .operator 232864 15562
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 232864) (leftOrdinal := 0)
    (rightResult := 15562) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7229⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7151⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7229⟩⟩, ⟨.program ⟨257⟩, ⟨7151⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge232870

namespace LeftMerge232871
def owner : Owner := ⟨.program ⟨257⟩, ⟨47322⟩⟩
def mergeEvent : Nat := 232871
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨45666⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7151⟩⟩] } }
def leftRaw : List Term := Proof.Events909.exact232864RawTerms
def rightRaw : List Term := Proof.Events060.exact15562RawTerms
def group : MergeGroup := .operator 232864 15562
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 232864) (leftOrdinal := 1)
    (rightResult := 15562) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨45666⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7151⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨45666⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7151⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge232871

namespace LeftMerge232873
def owner : Owner := ⟨.program ⟨257⟩, ⟨47322⟩⟩
def mergeEvent : Nat := 232873
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45666⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events060.exact15555RawTerms
def group : MergeGroup := .relation 232872
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 232872) (rhsResult := 15555)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨45666⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7151⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨7151⟩⟩) ⟨7041⟩ 15555) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6807⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45666⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge232873

namespace LeftMerge232887
def owner : Owner := ⟨.program ⟨257⟩, ⟨44640⟩⟩
def mergeEvent : Nat := 232887
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44638⟩⟩] } }
def leftRaw : List Term := Proof.Events872.exact223395RawTerms
def rightRaw : List Term := Proof.Events909.exact232881RawTerms
def group : MergeGroup := .operator 223395 232881
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 223395) (leftOrdinal := 0)
    (rightResult := 232881) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7194⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨44638⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44638⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge232887

namespace LeftMerge232888
def owner : Owner := ⟨.program ⟨257⟩, ⟨44640⟩⟩
def mergeEvent : Nat := 232888
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨42780⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨44638⟩⟩] } }
def leftRaw : List Term := Proof.Events872.exact223395RawTerms
def rightRaw : List Term := Proof.Events909.exact232881RawTerms
def group : MergeGroup := .operator 223395 232881
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 223395) (leftOrdinal := 1)
    (rightResult := 232881) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨42780⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨44638⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨42780⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨44638⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge232888

namespace LeftMerge232890
def owner : Owner := ⟨.program ⟨257⟩, ⟨44640⟩⟩
def mergeEvent : Nat := 232890
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨42780⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨43931⟩⟩] } }
def rhsRaw : List Term := Proof.Events909.exact232878RawTerms
def group : MergeGroup := .relation 232889
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 232889) (rhsResult := 232878)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨42780⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨44638⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨44638⟩⟩) ⟨43931⟩ 232878) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨43931⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨42780⟩⟩], [⟨.program ⟨257⟩, ⟨43931⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge232890

namespace LeftMerge232904
def owner : Owner := ⟨.program ⟨257⟩, ⟨43515⟩⟩
def mergeEvent : Nat := 232904
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43512⟩⟩] } }
def leftRaw : List Term := Proof.Events868.exact222245RawTerms
def rightRaw : List Term := Proof.Events909.exact232898RawTerms
def group : MergeGroup := .operator 222245 232898
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 222245) (leftOrdinal := 0)
    (rightResult := 232898) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨43512⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43512⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge232904

namespace LeftMerge233025
def owner : Owner := ⟨.program ⟨257⟩, ⟨44144⟩⟩
def mergeEvent : Nat := 233025
def frameStart : Nat := 232959
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨42780⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events910.exact233021RawTerms
def rightRaw : List Term := Proof.Events910.exact233019RawTerms
def group : MergeGroup := .operator 233021 233019
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 233021) (leftOrdinal := 0)
    (rightResult := 233019) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨42780⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨42780⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge233025

namespace LeftMerge233037
def owner : Owner := ⟨.program ⟨257⟩, ⟨44639⟩⟩
def mergeEvent : Nat := 233037
def frameStart : Nat := 232959
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44638⟩⟩] } }
def leftRaw : List Term := Proof.Events910.exact233033RawTerms
def rightRaw : List Term := Proof.Events910.exact233010RawTerms
def group : MergeGroup := .operator 233033 233010
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 233033) (leftOrdinal := 0)
    (rightResult := 233010) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7194⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨44638⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44638⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge233037

namespace LeftMerge233038
def owner : Owner := ⟨.program ⟨257⟩, ⟨44639⟩⟩
def mergeEvent : Nat := 233038
def frameStart : Nat := 232959
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨42780⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨44638⟩⟩] } }
def leftRaw : List Term := Proof.Events910.exact233033RawTerms
def rightRaw : List Term := Proof.Events910.exact233010RawTerms
def group : MergeGroup := .operator 233033 233010
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 233033) (leftOrdinal := 1)
    (rightResult := 233010) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨42780⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨44638⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨42780⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨44638⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge233038

namespace LeftMerge233040
def owner : Owner := ⟨.program ⟨257⟩, ⟨44639⟩⟩
def mergeEvent : Nat := 233040
def frameStart : Nat := 232959
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨42780⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨43931⟩⟩] } }
def rhsRaw : List Term := Proof.Events910.exact233007RawTerms
def group : MergeGroup := .relation 233039
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 233039) (rhsResult := 233007)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨42780⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨44638⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨44638⟩⟩) ⟨43931⟩ 233007) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨43931⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨42780⟩⟩], [⟨.program ⟨257⟩, ⟨43931⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge233040

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
