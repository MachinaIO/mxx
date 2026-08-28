import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge66881
def owner : Owner := ⟨.program ⟨257⟩, ⟨57859⟩⟩
def mergeEvent : Nat := 66881
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57856⟩⟩] } }
def leftRaw : List Term := Proof.Events239.exact61370RawTerms
def rightRaw : List Term := Proof.Events261.exact66875RawTerms
def group : MergeGroup := .operator 61370 66875
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 61370) (leftOrdinal := 0)
    (rightResult := 66875) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨57856⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57856⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge66881

namespace LeftMerge67002
def owner : Owner := ⟨.program ⟨257⟩, ⟨58356⟩⟩
def mergeEvent : Nat := 67002
def frameStart : Nat := 66936
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨56904⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events261.exact66998RawTerms
def rightRaw : List Term := Proof.Events261.exact66996RawTerms
def group : MergeGroup := .operator 66998 66996
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 66998) (leftOrdinal := 0)
    (rightResult := 66996) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨56904⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨56904⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge67002

namespace LeftMerge67014
def owner : Owner := ⟨.program ⟨257⟩, ⟨59130⟩⟩
def mergeEvent : Nat := 67014
def frameStart : Nat := 66936
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7185⟩⟩, ⟨.program ⟨257⟩, ⟨59129⟩⟩] } }
def leftRaw : List Term := Proof.Events261.exact67010RawTerms
def rightRaw : List Term := Proof.Events261.exact66987RawTerms
def group : MergeGroup := .operator 67010 66987
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 67010) (leftOrdinal := 0)
    (rightResult := 66987) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7185⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨59129⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7185⟩⟩, ⟨.program ⟨257⟩, ⟨59129⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge67014

namespace LeftMerge67015
def owner : Owner := ⟨.program ⟨257⟩, ⟨59130⟩⟩
def mergeEvent : Nat := 67015
def frameStart : Nat := 66936
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨56904⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨59129⟩⟩] } }
def leftRaw : List Term := Proof.Events261.exact67010RawTerms
def rightRaw : List Term := Proof.Events261.exact66987RawTerms
def group : MergeGroup := .operator 67010 66987
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 67010) (leftOrdinal := 1)
    (rightResult := 66987) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨56904⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨59129⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨56904⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨59129⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge67015

namespace LeftMerge67017
def owner : Owner := ⟨.program ⟨257⟩, ⟨59130⟩⟩
def mergeEvent : Nat := 67017
def frameStart : Nat := 66936
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨56904⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨58184⟩⟩] } }
def rhsRaw : List Term := Proof.Events261.exact66984RawTerms
def group : MergeGroup := .relation 67016
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 67016) (rhsResult := 66984)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨56904⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨59129⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨59129⟩⟩) ⟨58184⟩ 66984) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨58184⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨56904⟩⟩], [⟨.program ⟨257⟩, ⟨58184⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge67017

namespace LeftMerge67025
def owner : Owner := ⟨.program ⟨257⟩, ⟨57256⟩⟩
def mergeEvent : Nat := 67025
def frameStart : Nat := 66936
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨57254⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events261.exact66998RawTerms
def rightRaw : List Term := Proof.Events261.exact67021RawTerms
def group : MergeGroup := .operator 66998 67021
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 66998) (leftOrdinal := 0)
    (rightResult := 67021) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨57254⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨57254⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge67025

namespace LeftMerge67042
def owner : Owner := ⟨.program ⟨257⟩, ⟨57859⟩⟩
def mergeEvent : Nat := 67042
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7210⟩⟩] } }
def rhsRaw : List Term := Proof.Events261.exact67039RawTerms
def group : MergeGroup := .relation 67041
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 67041) (rhsResult := 67039)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57856⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 67040 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57856⟩⟩]⟩) (none) 67039) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7210⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge67042

namespace LeftMerge67043
def owner : Owner := ⟨.program ⟨257⟩, ⟨57859⟩⟩
def mergeEvent : Nat := 67043
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7185⟩⟩, ⟨.program ⟨257⟩, ⟨59129⟩⟩] } }
def rhsRaw : List Term := Proof.Events261.exact67039RawTerms
def group : MergeGroup := .relation 67041
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 67041) (rhsResult := 67039)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57856⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 67040 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57856⟩⟩]⟩) (none) 67039) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7185⟩⟩, ⟨.program ⟨257⟩, ⟨59129⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7185⟩⟩, ⟨.program ⟨257⟩, ⟨59129⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge67043

namespace LeftMerge67044
def owner : Owner := ⟨.program ⟨257⟩, ⟨57859⟩⟩
def mergeEvent : Nat := 67044
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨56904⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨58184⟩⟩] } }
def rhsRaw : List Term := Proof.Events261.exact67039RawTerms
def group : MergeGroup := .relation 67041
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 67041) (rhsResult := 67039)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57856⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 67040 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57856⟩⟩]⟩) (none) 67039) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨56904⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨58184⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨56904⟩⟩], [⟨.program ⟨257⟩, ⟨58184⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge67044

namespace LeftMerge67045
def owner : Owner := ⟨.program ⟨257⟩, ⟨57859⟩⟩
def mergeEvent : Nat := 67045
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨57254⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events261.exact67039RawTerms
def group : MergeGroup := .relation 67041
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 67041) (rhsResult := 67039)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57856⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 67040 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57856⟩⟩]⟩) (none) 67039) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨57254⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨57254⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge67045

namespace LeftMerge67050
def owner : Owner := ⟨.program ⟨257⟩, ⟨59132⟩⟩
def mergeEvent : Nat := 67050
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7185⟩⟩, ⟨.program ⟨257⟩, ⟨59129⟩⟩] } }
def leftRaw : List Term := Proof.Events261.exact67046RawTerms
def rightRaw : List Term := Proof.Events261.exact66868RawTerms
def group : MergeGroup := .operator 67046 66868
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 67046) (leftOrdinal := 0)
    (rightResult := 66868) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7185⟩⟩, ⟨.program ⟨257⟩, ⟨59129⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7185⟩⟩, ⟨.program ⟨257⟩, ⟨59129⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7185⟩⟩, ⟨.program ⟨257⟩, ⟨59129⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge67050

namespace LeftMerge67051
def owner : Owner := ⟨.program ⟨257⟩, ⟨59132⟩⟩
def mergeEvent : Nat := 67051
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨56904⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨58184⟩⟩] } }
def leftRaw : List Term := Proof.Events261.exact67046RawTerms
def rightRaw : List Term := Proof.Events261.exact66868RawTerms
def group : MergeGroup := .operator 67046 66868
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 67046) (leftOrdinal := 2)
    (rightResult := 66868) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨56904⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨58184⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨56904⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨58184⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨56904⟩⟩], [⟨.program ⟨257⟩, ⟨58184⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge67051

namespace LeftMerge67077
def owner : Owner := ⟨.program ⟨257⟩, ⟨24855⟩⟩
def mergeEvent : Nat := 67077
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨24854⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events010.exact2614RawTerms
def rightRaw : List Term := Proof.Events239.exact61278RawTerms
def group : MergeGroup := .operator 2614 61278
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 2614) (leftOrdinal := 0)
    (rightResult := 61278) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24854⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨24854⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge67077

namespace LeftMerge67082
def owner : Owner := ⟨.program ⟨257⟩, ⟨10754⟩⟩
def mergeEvent : Nat := 67082
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7272⟩⟩] } }
def leftRaw : List Term := Proof.Events238.exact61148RawTerms
def rightRaw : List Term := Proof.Events090.exact23092RawTerms
def group : MergeGroup := .operator 61148 23092
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 61148) (leftOrdinal := 0)
    (rightResult := 23092) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7272⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7272⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge67082

namespace LeftMerge67099
def owner : Owner := ⟨.program ⟨257⟩, ⟨53717⟩⟩
def mergeEvent : Nat := 67099
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨24854⟩⟩, ⟨.program ⟨257⟩, ⟨53714⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events262.exact67093RawTerms
def rightRaw : List Term := Proof.Events010.exact2617RawTerms
def group : MergeGroup := .operator 67093 2617
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 67093) (leftOrdinal := 1)
    (rightResult := 2617) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨24854⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨53714⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨24854⟩⟩, ⟨.program ⟨257⟩, ⟨53714⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge67099

namespace LeftMerge67100
def owner : Owner := ⟨.program ⟨257⟩, ⟨53717⟩⟩
def mergeEvent : Nat := 67100
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨53714⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7272⟩⟩] } }
def leftRaw : List Term := Proof.Events262.exact67093RawTerms
def rightRaw : List Term := Proof.Events010.exact2617RawTerms
def group : MergeGroup := .operator 67093 2617
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 67093) (leftOrdinal := 0)
    (rightResult := 2617) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7272⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨53714⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨53714⟩⟩], [⟨.program ⟨257⟩, ⟨7272⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge67100

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
