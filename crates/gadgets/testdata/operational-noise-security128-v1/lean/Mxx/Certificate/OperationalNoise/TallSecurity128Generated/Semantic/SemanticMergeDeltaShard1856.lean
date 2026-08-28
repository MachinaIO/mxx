import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge299944
def owner : Owner := ⟨.program ⟨257⟩, ⟨56242⟩⟩
def mergeEvent : Nat := 299944
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨56235⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7273⟩⟩] } }
def rhsRaw : List Term := Proof.Events088.exact22591RawTerms
def group : MergeGroup := .relation 299943
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 299943) (rhsResult := 22591)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨56235⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨9532⟩⟩) ⟨7273⟩ 22591) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7273⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨56235⟩⟩], [⟨.program ⟨257⟩, ⟨7273⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge299944

namespace LeftMerge299945
def owner : Owner := ⟨.program ⟨257⟩, ⟨56242⟩⟩
def mergeEvent : Nat := 299945
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩] } }
def leftRaw : List Term := Proof.Events1171.exact299936RawTerms
def rightRaw : List Term := Proof.Events088.exact22621RawTerms
def group : MergeGroup := .operator 299936 22621
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 299936) (leftOrdinal := 0)
    (rightResult := 22621) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7290⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9532⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge299945

namespace LeftMerge299950
def owner : Owner := ⟨.program ⟨257⟩, ⟨56243⟩⟩
def mergeEvent : Nat := 299950
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨56235⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7273⟩⟩] } }
def leftRaw : List Term := Proof.Events1171.exact299946RawTerms
def rightRaw : List Term := Proof.Events1171.exact299916RawTerms
def group : MergeGroup := .operator 299946 299916
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 299946) (leftOrdinal := 1)
    (rightResult := 299916) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨56235⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7273⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨56235⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7273⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨56235⟩⟩], [⟨.program ⟨257⟩, ⟨7273⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge299950

namespace LeftMerge299958
def owner : Owner := ⟨.program ⟨257⟩, ⟨58370⟩⟩
def mergeEvent : Nat := 299958
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨24890⟩⟩, ⟨.program ⟨257⟩, ⟨56235⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨58369⟩⟩] } }
def leftRaw : List Term := Proof.Events1171.exact299952RawTerms
def rightRaw : List Term := Proof.Events1171.exact299888RawTerms
def group : MergeGroup := .operator 299952 299888
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 299952) (leftOrdinal := 1)
    (rightResult := 299888) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨24890⟩⟩, ⟨.program ⟨257⟩, ⟨56235⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨58369⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨24890⟩⟩, ⟨.program ⟨257⟩, ⟨56235⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨58369⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge299958

namespace LeftMerge299960
def owner : Owner := ⟨.program ⟨257⟩, ⟨58370⟩⟩
def mergeEvent : Nat := 299960
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨24890⟩⟩, ⟨.program ⟨257⟩, ⟨56235⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨57909⟩⟩] } }
def rhsRaw : List Term := Proof.Events1171.exact299885RawTerms
def group : MergeGroup := .relation 299959
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 299959) (rhsResult := 299885)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨24890⟩⟩, ⟨.program ⟨257⟩, ⟨56235⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨58369⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨58369⟩⟩) ⟨57909⟩ 299885) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨57909⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨24890⟩⟩, ⟨.program ⟨257⟩, ⟨56235⟩⟩], [⟨.program ⟨257⟩, ⟨57909⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge299960

namespace LeftMerge299961
def owner : Owner := ⟨.program ⟨257⟩, ⟨58370⟩⟩
def mergeEvent : Nat := 299961
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩, ⟨.program ⟨257⟩, ⟨58369⟩⟩] } }
def leftRaw : List Term := Proof.Events1171.exact299952RawTerms
def rightRaw : List Term := Proof.Events1171.exact299888RawTerms
def group : MergeGroup := .operator 299952 299888
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 299952) (leftOrdinal := 0)
    (rightResult := 299888) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨58369⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩, ⟨.program ⟨257⟩, ⟨58369⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge299961

namespace LeftMerge299975
def owner : Owner := ⟨.program ⟨257⟩, ⟨57312⟩⟩
def mergeEvent : Nat := 299975
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57309⟩⟩] } }
def leftRaw : List Term := Proof.Events1153.exact295195RawTerms
def rightRaw : List Term := Proof.Events1171.exact299969RawTerms
def group : MergeGroup := .operator 295195 299969
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 295195) (leftOrdinal := 0)
    (rightResult := 299969) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨57309⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57309⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge299975

namespace LeftMerge300030
def owner : Owner := ⟨.program ⟨257⟩, ⟨56236⟩⟩
def mergeEvent : Nat := 300030
def frameStart : Nat := 300012
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24890⟩⟩, ⟨.program ⟨257⟩, ⟨56235⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events1171.exact300026RawTerms
def rightRaw : List Term := Proof.Events1171.exact300023RawTerms
def group : MergeGroup := .operator 300026 300023
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 300026) (leftOrdinal := 0)
    (rightResult := 300023) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨56235⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24890⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨24890⟩⟩, ⟨.program ⟨257⟩, ⟨56235⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge300030

namespace LeftMerge300060
def owner : Owner := ⟨.program ⟨257⟩, ⟨58208⟩⟩
def mergeEvent : Nat := 300060
def frameStart : Nat := 300012
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24890⟩⟩, ⟨.program ⟨257⟩, ⟨56235⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1172.exact300056RawTerms
def rightRaw : List Term := Proof.Events1172.exact300054RawTerms
def group : MergeGroup := .operator 300056 300054
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 300056) (leftOrdinal := 0)
    (rightResult := 300054) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24890⟩⟩, ⟨.program ⟨257⟩, ⟨56235⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨24890⟩⟩, ⟨.program ⟨257⟩, ⟨56235⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge300060

namespace LeftMerge300083
def owner : Owner := ⟨.program ⟨257⟩, ⟨9534⟩⟩
def mergeEvent : Nat := 300083
def frameStart : Nat := 300012
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩] } }
def leftRaw : List Term := Proof.Events1172.exact300079RawTerms
def rightRaw : List Term := Proof.Events1172.exact300076RawTerms
def group : MergeGroup := .operator 300079 300076
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 300079) (leftOrdinal := 0)
    (rightResult := 300076) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7290⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9532⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge300083

namespace LeftMerge300092
def owner : Owner := ⟨.program ⟨257⟩, ⟨58372⟩⟩
def mergeEvent : Nat := 300092
def frameStart : Nat := 300012
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩, ⟨.program ⟨257⟩, ⟨58369⟩⟩] } }
def leftRaw : List Term := Proof.Events1172.exact300088RawTerms
def rightRaw : List Term := Proof.Events1172.exact300045RawTerms
def group : MergeGroup := .operator 300088 300045
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 300088) (leftOrdinal := 0)
    (rightResult := 300045) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨58369⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩, ⟨.program ⟨257⟩, ⟨58369⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge300092

namespace LeftMerge300093
def owner : Owner := ⟨.program ⟨257⟩, ⟨58372⟩⟩
def mergeEvent : Nat := 300093
def frameStart : Nat := 300012
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24890⟩⟩, ⟨.program ⟨257⟩, ⟨56235⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨58369⟩⟩] } }
def leftRaw : List Term := Proof.Events1172.exact300088RawTerms
def rightRaw : List Term := Proof.Events1172.exact300045RawTerms
def group : MergeGroup := .operator 300088 300045
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 300088) (leftOrdinal := 1)
    (rightResult := 300045) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24890⟩⟩, ⟨.program ⟨257⟩, ⟨56235⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨58369⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨24890⟩⟩, ⟨.program ⟨257⟩, ⟨56235⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨58369⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge300093

namespace LeftMerge300095
def owner : Owner := ⟨.program ⟨257⟩, ⟨58372⟩⟩
def mergeEvent : Nat := 300095
def frameStart : Nat := 300012
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24890⟩⟩, ⟨.program ⟨257⟩, ⟨56235⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨57909⟩⟩] } }
def rhsRaw : List Term := Proof.Events1172.exact300042RawTerms
def group : MergeGroup := .relation 300094
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 300094) (rhsResult := 300042)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨24890⟩⟩, ⟨.program ⟨257⟩, ⟨56235⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨58369⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨58369⟩⟩) ⟨57909⟩ 300042) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨57909⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨24890⟩⟩, ⟨.program ⟨257⟩, ⟨56235⟩⟩], [⟨.program ⟨257⟩, ⟨57909⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge300095

namespace LeftMerge300103
def owner : Owner := ⟨.program ⟨257⟩, ⟨56770⟩⟩
def mergeEvent : Nat := 300103
def frameStart : Nat := 300012
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨56768⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1172.exact300056RawTerms
def rightRaw : List Term := Proof.Events1172.exact300099RawTerms
def group : MergeGroup := .operator 300056 300099
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 300056) (leftOrdinal := 0)
    (rightResult := 300099) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨56768⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨56768⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge300103

namespace LeftMerge300120
def owner : Owner := ⟨.program ⟨257⟩, ⟨57312⟩⟩
def mergeEvent : Nat := 300120
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7185⟩⟩] } }
def rhsRaw : List Term := Proof.Events1172.exact300117RawTerms
def group : MergeGroup := .relation 300119
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 300119) (rhsResult := 300117)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57309⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 300118 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57309⟩⟩]⟩) (none) 300117) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7185⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7185⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge300120

namespace LeftMerge300121
def owner : Owner := ⟨.program ⟨257⟩, ⟨57312⟩⟩
def mergeEvent : Nat := 300121
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩, ⟨.program ⟨257⟩, ⟨58369⟩⟩] } }
def rhsRaw : List Term := Proof.Events1172.exact300117RawTerms
def group : MergeGroup := .relation 300119
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 300119) (rhsResult := 300117)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57309⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 300118 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57309⟩⟩]⟩) (none) 300117) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩, ⟨.program ⟨257⟩, ⟨58369⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩, ⟨.program ⟨257⟩, ⟨58369⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge300121

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
