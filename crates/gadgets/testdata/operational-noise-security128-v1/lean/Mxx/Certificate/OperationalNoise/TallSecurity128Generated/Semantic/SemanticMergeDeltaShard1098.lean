import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge178841
def owner : Owner := ⟨.program ⟨257⟩, ⟨47013⟩⟩
def mergeEvent : Nat := 178841
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨14826⟩⟩, ⟨.program ⟨257⟩, ⟨45226⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨47012⟩⟩] } }
def leftRaw : List Term := Proof.Events698.exact178835RawTerms
def rightRaw : List Term := Proof.Events698.exact178771RawTerms
def group : MergeGroup := .operator 178835 178771
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 178835) (leftOrdinal := 1)
    (rightResult := 178771) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨14826⟩⟩, ⟨.program ⟨257⟩, ⟨45226⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨47012⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨14826⟩⟩, ⟨.program ⟨257⟩, ⟨45226⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨47012⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge178841

namespace LeftMerge178843
def owner : Owner := ⟨.program ⟨257⟩, ⟨47013⟩⟩
def mergeEvent : Nat := 178843
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨14826⟩⟩, ⟨.program ⟨257⟩, ⟨45226⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46487⟩⟩] } }
def rhsRaw : List Term := Proof.Events698.exact178768RawTerms
def group : MergeGroup := .relation 178842
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 178842) (rhsResult := 178768)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨14826⟩⟩, ⟨.program ⟨257⟩, ⟨45226⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨47012⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨47012⟩⟩) ⟨46487⟩ 178768) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨46487⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨14826⟩⟩, ⟨.program ⟨257⟩, ⟨45226⟩⟩], [⟨.program ⟨257⟩, ⟨46487⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge178843

namespace LeftMerge178844
def owner : Owner := ⟨.program ⟨257⟩, ⟨47013⟩⟩
def mergeEvent : Nat := 178844
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨47012⟩⟩] } }
def leftRaw : List Term := Proof.Events698.exact178835RawTerms
def rightRaw : List Term := Proof.Events698.exact178771RawTerms
def group : MergeGroup := .operator 178835 178771
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 178835) (leftOrdinal := 0)
    (rightResult := 178771) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨47012⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨47012⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge178844

namespace LeftMerge178858
def owner : Owner := ⟨.program ⟨257⟩, ⟨45942⟩⟩
def mergeEvent : Nat := 178858
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨45939⟩⟩] } }
def leftRaw : List Term := Proof.Events696.exact178370RawTerms
def rightRaw : List Term := Proof.Events698.exact178852RawTerms
def group : MergeGroup := .operator 178370 178852
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 178370) (leftOrdinal := 0)
    (rightResult := 178852) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨45939⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨45939⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge178858

namespace LeftMerge178937
def owner : Owner := ⟨.program ⟨257⟩, ⟨45227⟩⟩
def mergeEvent : Nat := 178937
def frameStart : Nat := 178907
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14826⟩⟩, ⟨.program ⟨257⟩, ⟨45226⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events698.exact178933RawTerms
def rightRaw : List Term := Proof.Events698.exact178930RawTerms
def group : MergeGroup := .operator 178933 178930
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 178933) (leftOrdinal := 0)
    (rightResult := 178930) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14826⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨45226⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨14826⟩⟩, ⟨.program ⟨257⟩, ⟨45226⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge178937

namespace LeftMerge178967
def owner : Owner := ⟨.program ⟨257⟩, ⟨46760⟩⟩
def mergeEvent : Nat := 178967
def frameStart : Nat := 178907
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14826⟩⟩, ⟨.program ⟨257⟩, ⟨45226⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events699.exact178963RawTerms
def rightRaw : List Term := Proof.Events699.exact178961RawTerms
def group : MergeGroup := .operator 178963 178961
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 178963) (leftOrdinal := 0)
    (rightResult := 178961) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14826⟩⟩, ⟨.program ⟨257⟩, ⟨45226⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨14826⟩⟩, ⟨.program ⟨257⟩, ⟨45226⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge178967

namespace LeftMerge178990
def owner : Owner := ⟨.program ⟨257⟩, ⟨9564⟩⟩
def mergeEvent : Nat := 178990
def frameStart : Nat := 178907
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩] } }
def leftRaw : List Term := Proof.Events699.exact178986RawTerms
def rightRaw : List Term := Proof.Events699.exact178983RawTerms
def group : MergeGroup := .operator 178986 178983
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 178986) (leftOrdinal := 0)
    (rightResult := 178983) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9562⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge178990

namespace LeftMerge178999
def owner : Owner := ⟨.program ⟨257⟩, ⟨47015⟩⟩
def mergeEvent : Nat := 178999
def frameStart : Nat := 178907
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨47012⟩⟩] } }
def leftRaw : List Term := Proof.Events699.exact178995RawTerms
def rightRaw : List Term := Proof.Events699.exact178952RawTerms
def group : MergeGroup := .operator 178995 178952
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 178995) (leftOrdinal := 0)
    (rightResult := 178952) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨47012⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨47012⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge178999

namespace LeftMerge179000
def owner : Owner := ⟨.program ⟨257⟩, ⟨47015⟩⟩
def mergeEvent : Nat := 179000
def frameStart : Nat := 178907
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14826⟩⟩, ⟨.program ⟨257⟩, ⟨45226⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨47012⟩⟩] } }
def leftRaw : List Term := Proof.Events699.exact178995RawTerms
def rightRaw : List Term := Proof.Events699.exact178952RawTerms
def group : MergeGroup := .operator 178995 178952
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 178995) (leftOrdinal := 1)
    (rightResult := 178952) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14826⟩⟩, ⟨.program ⟨257⟩, ⟨45226⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨47012⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨14826⟩⟩, ⟨.program ⟨257⟩, ⟨45226⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨47012⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge179000

namespace LeftMerge179002
def owner : Owner := ⟨.program ⟨257⟩, ⟨47015⟩⟩
def mergeEvent : Nat := 179002
def frameStart : Nat := 178907
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14826⟩⟩, ⟨.program ⟨257⟩, ⟨45226⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46487⟩⟩] } }
def rhsRaw : List Term := Proof.Events699.exact178949RawTerms
def group : MergeGroup := .relation 179001
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 179001) (rhsResult := 178949)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨14826⟩⟩, ⟨.program ⟨257⟩, ⟨45226⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨47012⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨47012⟩⟩) ⟨46487⟩ 178949) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨46487⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨14826⟩⟩, ⟨.program ⟨257⟩, ⟨45226⟩⟩], [⟨.program ⟨257⟩, ⟨46487⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge179002

namespace LeftMerge179010
def owner : Owner := ⟨.program ⟨257⟩, ⟨45494⟩⟩
def mergeEvent : Nat := 179010
def frameStart : Nat := 178907
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨45492⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events699.exact178963RawTerms
def rightRaw : List Term := Proof.Events699.exact179006RawTerms
def group : MergeGroup := .operator 178963 179006
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 178963) (leftOrdinal := 0)
    (rightResult := 179006) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨45492⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨45492⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge179010

namespace LeftMerge179027
def owner : Owner := ⟨.program ⟨257⟩, ⟨45942⟩⟩
def mergeEvent : Nat := 179027
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7195⟩⟩] } }
def rhsRaw : List Term := Proof.Events699.exact179024RawTerms
def group : MergeGroup := .relation 179026
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 179026) (rhsResult := 179024)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨45939⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 179025 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨45939⟩⟩]⟩) (none) 179024) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7195⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7195⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge179027

namespace LeftMerge179028
def owner : Owner := ⟨.program ⟨257⟩, ⟨45942⟩⟩
def mergeEvent : Nat := 179028
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨47012⟩⟩] } }
def rhsRaw : List Term := Proof.Events699.exact179024RawTerms
def group : MergeGroup := .relation 179026
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 179026) (rhsResult := 179024)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨45939⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 179025 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨45939⟩⟩]⟩) (none) 179024) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨47012⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨47012⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge179028

namespace LeftMerge179029
def owner : Owner := ⟨.program ⟨257⟩, ⟨45942⟩⟩
def mergeEvent : Nat := 179029
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨14826⟩⟩, ⟨.program ⟨257⟩, ⟨45226⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46487⟩⟩] } }
def rhsRaw : List Term := Proof.Events699.exact179024RawTerms
def group : MergeGroup := .relation 179026
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 179026) (rhsResult := 179024)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨45939⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 179025 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨45939⟩⟩]⟩) (none) 179024) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14826⟩⟩, ⟨.program ⟨257⟩, ⟨45226⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46487⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨14826⟩⟩, ⟨.program ⟨257⟩, ⟨45226⟩⟩], [⟨.program ⟨257⟩, ⟨46487⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge179029

namespace LeftMerge179030
def owner : Owner := ⟨.program ⟨257⟩, ⟨45942⟩⟩
def mergeEvent : Nat := 179030
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨45492⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events699.exact179024RawTerms
def group : MergeGroup := .relation 179026
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 179026) (rhsResult := 179024)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨45939⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 179025 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨45939⟩⟩]⟩) (none) 179024) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨45492⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨45492⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge179030

namespace LeftMerge179035
def owner : Owner := ⟨.program ⟨257⟩, ⟨47014⟩⟩
def mergeEvent : Nat := 179035
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨14826⟩⟩, ⟨.program ⟨257⟩, ⟨45226⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46487⟩⟩] } }
def leftRaw : List Term := Proof.Events699.exact179031RawTerms
def rightRaw : List Term := Proof.Events698.exact178845RawTerms
def group : MergeGroup := .operator 179031 178845
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 179031) (leftOrdinal := 2)
    (rightResult := 178845) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨14826⟩⟩, ⟨.program ⟨257⟩, ⟨45226⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46487⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨14826⟩⟩, ⟨.program ⟨257⟩, ⟨45226⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46487⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨14826⟩⟩, ⟨.program ⟨257⟩, ⟨45226⟩⟩], [⟨.program ⟨257⟩, ⟨46487⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge179035

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
