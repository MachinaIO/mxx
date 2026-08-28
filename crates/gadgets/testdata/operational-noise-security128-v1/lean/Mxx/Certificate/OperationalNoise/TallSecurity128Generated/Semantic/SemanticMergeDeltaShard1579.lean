import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge256016
def owner : Owner := ⟨.program ⟨257⟩, ⟨64386⟩⟩
def mergeEvent : Nat := 256016
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨25430⟩⟩, ⟨.program ⟨257⟩, ⟨62330⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨63899⟩⟩] } }
def leftRaw : List Term := Proof.Events1000.exact256012RawTerms
def rightRaw : List Term := Proof.Events999.exact255826RawTerms
def group : MergeGroup := .operator 256012 255826
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 256012) (leftOrdinal := 2)
    (rightResult := 255826) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨25430⟩⟩, ⟨.program ⟨257⟩, ⟨62330⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨63899⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨25430⟩⟩, ⟨.program ⟨257⟩, ⟨62330⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨63899⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨25430⟩⟩, ⟨.program ⟨257⟩, ⟨62330⟩⟩], [⟨.program ⟨257⟩, ⟨63899⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge256016

namespace LeftMerge256017
def owner : Owner := ⟨.program ⟨257⟩, ⟨64386⟩⟩
def mergeEvent : Nat := 256017
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩, ⟨.program ⟨257⟩, ⟨64384⟩⟩] } }
def leftRaw : List Term := Proof.Events1000.exact256012RawTerms
def rightRaw : List Term := Proof.Events999.exact255826RawTerms
def group : MergeGroup := .operator 256012 255826
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 256012) (leftOrdinal := 1)
    (rightResult := 255826) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩, ⟨.program ⟨257⟩, ⟨64384⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩, ⟨.program ⟨257⟩, ⟨64384⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩, ⟨.program ⟨257⟩, ⟨64384⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge256017

namespace LeftMerge256025
def owner : Owner := ⟨.program ⟨257⟩, ⟨64719⟩⟩
def mergeEvent : Nat := 256025
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7187⟩⟩, ⟨.program ⟨257⟩, ⟨64717⟩⟩] } }
def leftRaw : List Term := Proof.Events1000.exact256019RawTerms
def rightRaw : List Term := Proof.Events998.exact255742RawTerms
def group : MergeGroup := .operator 256019 255742
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 256019) (leftOrdinal := 0)
    (rightResult := 255742) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7187⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨64717⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨7187⟩⟩, ⟨.program ⟨257⟩, ⟨64717⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge256025

namespace LeftMerge256026
def owner : Owner := ⟨.program ⟨257⟩, ⟨64719⟩⟩
def mergeEvent : Nat := 256026
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨62768⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨64717⟩⟩] } }
def leftRaw : List Term := Proof.Events1000.exact256019RawTerms
def rightRaw : List Term := Proof.Events998.exact255742RawTerms
def group : MergeGroup := .operator 256019 255742
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 256019) (leftOrdinal := 1)
    (rightResult := 255742) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨62768⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨64717⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨62768⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨64717⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge256026

namespace LeftMerge256028
def owner : Owner := ⟨.program ⟨257⟩, ⟨64719⟩⟩
def mergeEvent : Nat := 256028
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨62768⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨64036⟩⟩] } }
def rhsRaw : List Term := Proof.Events998.exact255739RawTerms
def group : MergeGroup := .relation 256027
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 256027) (rhsResult := 255739)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨62768⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨64717⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨64717⟩⟩) ⟨64036⟩ 255739) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨64036⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨62768⟩⟩], [⟨.program ⟨257⟩, ⟨64036⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge256028

namespace LeftMerge256042
def owner : Owner := ⟨.program ⟨257⟩, ⟨63579⟩⟩
def mergeEvent : Nat := 256042
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63576⟩⟩] } }
def leftRaw : List Term := Proof.Events982.exact251495RawTerms
def rightRaw : List Term := Proof.Events1000.exact256036RawTerms
def group : MergeGroup := .operator 251495 256036
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 251495) (leftOrdinal := 0)
    (rightResult := 256036) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨63576⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63576⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge256042

namespace LeftMerge256163
def owner : Owner := ⟨.program ⟨257⟩, ⟨64268⟩⟩
def mergeEvent : Nat := 256163
def frameStart : Nat := 256097
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨62768⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1000.exact256159RawTerms
def rightRaw : List Term := Proof.Events1000.exact256157RawTerms
def group : MergeGroup := .operator 256159 256157
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 256159) (leftOrdinal := 0)
    (rightResult := 256157) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨62768⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨62768⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge256163

namespace LeftMerge256175
def owner : Owner := ⟨.program ⟨257⟩, ⟨64718⟩⟩
def mergeEvent : Nat := 256175
def frameStart : Nat := 256097
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7187⟩⟩, ⟨.program ⟨257⟩, ⟨64717⟩⟩] } }
def leftRaw : List Term := Proof.Events1000.exact256171RawTerms
def rightRaw : List Term := Proof.Events1000.exact256148RawTerms
def group : MergeGroup := .operator 256171 256148
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 256171) (leftOrdinal := 0)
    (rightResult := 256148) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7187⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨64717⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7187⟩⟩, ⟨.program ⟨257⟩, ⟨64717⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge256175

namespace LeftMerge256176
def owner : Owner := ⟨.program ⟨257⟩, ⟨64718⟩⟩
def mergeEvent : Nat := 256176
def frameStart : Nat := 256097
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨62768⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨64717⟩⟩] } }
def leftRaw : List Term := Proof.Events1000.exact256171RawTerms
def rightRaw : List Term := Proof.Events1000.exact256148RawTerms
def group : MergeGroup := .operator 256171 256148
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 256171) (leftOrdinal := 1)
    (rightResult := 256148) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨62768⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨64717⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨62768⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨64717⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge256176

namespace LeftMerge256178
def owner : Owner := ⟨.program ⟨257⟩, ⟨64718⟩⟩
def mergeEvent : Nat := 256178
def frameStart : Nat := 256097
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨62768⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨64036⟩⟩] } }
def rhsRaw : List Term := Proof.Events1000.exact256145RawTerms
def group : MergeGroup := .relation 256177
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 256177) (rhsResult := 256145)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨62768⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨64717⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨64717⟩⟩) ⟨64036⟩ 256145) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨64036⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨62768⟩⟩], [⟨.program ⟨257⟩, ⟨64036⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge256178

namespace LeftMerge256186
def owner : Owner := ⟨.program ⟨257⟩, ⟨62988⟩⟩
def mergeEvent : Nat := 256186
def frameStart : Nat := 256097
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨62986⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1000.exact256159RawTerms
def rightRaw : List Term := Proof.Events1000.exact256182RawTerms
def group : MergeGroup := .operator 256159 256182
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 256159) (leftOrdinal := 0)
    (rightResult := 256182) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨62986⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨62986⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge256186

namespace LeftMerge256203
def owner : Owner := ⟨.program ⟨257⟩, ⟨63579⟩⟩
def mergeEvent : Nat := 256203
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7214⟩⟩] } }
def rhsRaw : List Term := Proof.Events1000.exact256200RawTerms
def group : MergeGroup := .relation 256202
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 256202) (rhsResult := 256200)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63576⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 256201 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63576⟩⟩]⟩) (none) 256200) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7214⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge256203

namespace LeftMerge256204
def owner : Owner := ⟨.program ⟨257⟩, ⟨63579⟩⟩
def mergeEvent : Nat := 256204
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7187⟩⟩, ⟨.program ⟨257⟩, ⟨64717⟩⟩] } }
def rhsRaw : List Term := Proof.Events1000.exact256200RawTerms
def group : MergeGroup := .relation 256202
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 256202) (rhsResult := 256200)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63576⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 256201 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63576⟩⟩]⟩) (none) 256200) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7187⟩⟩, ⟨.program ⟨257⟩, ⟨64717⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨7187⟩⟩, ⟨.program ⟨257⟩, ⟨64717⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge256204

namespace LeftMerge256205
def owner : Owner := ⟨.program ⟨257⟩, ⟨63579⟩⟩
def mergeEvent : Nat := 256205
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨62768⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨64036⟩⟩] } }
def rhsRaw : List Term := Proof.Events1000.exact256200RawTerms
def group : MergeGroup := .relation 256202
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 256202) (rhsResult := 256200)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63576⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 256201 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63576⟩⟩]⟩) (none) 256200) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨62768⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨64036⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨62768⟩⟩], [⟨.program ⟨257⟩, ⟨64036⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge256205

namespace LeftMerge256206
def owner : Owner := ⟨.program ⟨257⟩, ⟨63579⟩⟩
def mergeEvent : Nat := 256206
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨62986⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events1000.exact256200RawTerms
def group : MergeGroup := .relation 256202
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 256202) (rhsResult := 256200)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63576⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 256201 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63576⟩⟩]⟩) (none) 256200) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨62986⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨62986⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge256206

namespace LeftMerge256211
def owner : Owner := ⟨.program ⟨257⟩, ⟨64720⟩⟩
def mergeEvent : Nat := 256211
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7187⟩⟩, ⟨.program ⟨257⟩, ⟨64717⟩⟩] } }
def leftRaw : List Term := Proof.Events1000.exact256207RawTerms
def rightRaw : List Term := Proof.Events1000.exact256029RawTerms
def group : MergeGroup := .operator 256207 256029
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 256207) (leftOrdinal := 0)
    (rightResult := 256029) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7187⟩⟩, ⟨.program ⟨257⟩, ⟨64717⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7187⟩⟩, ⟨.program ⟨257⟩, ⟨64717⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨7187⟩⟩, ⟨.program ⟨257⟩, ⟨64717⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge256211

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
