import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge188440
def owner : Owner := ⟨.program ⟨257⟩, ⟨71330⟩⟩
def mergeEvent : Nat := 188440
def frameStart : Nat := 187711
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨60158⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68848⟩⟩] } }
def rhsRaw : List Term := Proof.Events735.exact188224RawTerms
def group : MergeGroup := .relation 188439
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 188439) (rhsResult := 188224)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨60158⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71329⟩⟩) ⟨68848⟩ 188224) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68848⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨60158⟩⟩], [⟨.program ⟨257⟩, ⟨68848⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge188440

namespace LeftMerge188441
def owner : Owner := ⟨.program ⟨257⟩, ⟨71330⟩⟩
def mergeEvent : Nat := 188441
def frameStart : Nat := 187711
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨57178⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩] } }
def leftRaw : List Term := Proof.Events735.exact188386RawTerms
def rightRaw : List Term := Proof.Events735.exact188227RawTerms
def group : MergeGroup := .operator 188386 188227
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 188386) (leftOrdinal := 32)
    (rightResult := 188227) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨57178⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71329⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨57178⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge188441

namespace LeftMerge188443
def owner : Owner := ⟨.program ⟨257⟩, ⟨71330⟩⟩
def mergeEvent : Nat := 188443
def frameStart : Nat := 187711
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨57178⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68848⟩⟩] } }
def rhsRaw : List Term := Proof.Events735.exact188224RawTerms
def group : MergeGroup := .relation 188442
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 188442) (rhsResult := 188224)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨57178⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71329⟩⟩) ⟨68848⟩ 188224) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68848⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨57178⟩⟩], [⟨.program ⟨257⟩, ⟨68848⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge188443

namespace LeftMerge188444
def owner : Owner := ⟨.program ⟨257⟩, ⟨71330⟩⟩
def mergeEvent : Nat := 188444
def frameStart : Nat := 187711
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨54198⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩] } }
def leftRaw : List Term := Proof.Events735.exact188386RawTerms
def rightRaw : List Term := Proof.Events735.exact188227RawTerms
def group : MergeGroup := .operator 188386 188227
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 188386) (leftOrdinal := 31)
    (rightResult := 188227) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨54198⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71329⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨54198⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge188444

namespace LeftMerge188446
def owner : Owner := ⟨.program ⟨257⟩, ⟨71330⟩⟩
def mergeEvent : Nat := 188446
def frameStart : Nat := 187711
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨54198⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68848⟩⟩] } }
def rhsRaw : List Term := Proof.Events735.exact188224RawTerms
def group : MergeGroup := .relation 188445
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 188445) (rhsResult := 188224)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨54198⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71329⟩⟩) ⟨68848⟩ 188224) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68848⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨54198⟩⟩], [⟨.program ⟨257⟩, ⟨68848⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge188446

namespace LeftMerge188447
def owner : Owner := ⟨.program ⟨257⟩, ⟨71330⟩⟩
def mergeEvent : Nat := 188447
def frameStart : Nat := 187711
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨51218⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩] } }
def leftRaw : List Term := Proof.Events735.exact188386RawTerms
def rightRaw : List Term := Proof.Events735.exact188227RawTerms
def group : MergeGroup := .operator 188386 188227
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 188386) (leftOrdinal := 30)
    (rightResult := 188227) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨51218⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71329⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨51218⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge188447

namespace LeftMerge188449
def owner : Owner := ⟨.program ⟨257⟩, ⟨71330⟩⟩
def mergeEvent : Nat := 188449
def frameStart : Nat := 187711
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨51218⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68848⟩⟩] } }
def rhsRaw : List Term := Proof.Events735.exact188224RawTerms
def group : MergeGroup := .relation 188448
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 188448) (rhsResult := 188224)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨51218⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71329⟩⟩) ⟨68848⟩ 188224) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68848⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨51218⟩⟩], [⟨.program ⟨257⟩, ⟨68848⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge188449

namespace LeftMerge188450
def owner : Owner := ⟨.program ⟨257⟩, ⟨71330⟩⟩
def mergeEvent : Nat := 188450
def frameStart : Nat := 187711
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨32163⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩] } }
def leftRaw : List Term := Proof.Events735.exact188386RawTerms
def rightRaw : List Term := Proof.Events735.exact188227RawTerms
def group : MergeGroup := .operator 188386 188227
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 188386) (leftOrdinal := 23)
    (rightResult := 188227) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨32163⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71329⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨32163⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge188450

namespace LeftMerge188452
def owner : Owner := ⟨.program ⟨257⟩, ⟨71330⟩⟩
def mergeEvent : Nat := 188452
def frameStart : Nat := 187711
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨32163⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68848⟩⟩] } }
def rhsRaw : List Term := Proof.Events735.exact188224RawTerms
def group : MergeGroup := .relation 188451
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 188451) (rhsResult := 188224)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨32163⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71329⟩⟩) ⟨68848⟩ 188224) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68848⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨32163⟩⟩], [⟨.program ⟨257⟩, ⟨68848⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge188452

namespace LeftMerge188453
def owner : Owner := ⟨.program ⟨257⟩, ⟨71330⟩⟩
def mergeEvent : Nat := 188453
def frameStart : Nat := 187711
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨22143⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩] } }
def leftRaw : List Term := Proof.Events735.exact188386RawTerms
def rightRaw : List Term := Proof.Events735.exact188227RawTerms
def group : MergeGroup := .operator 188386 188227
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 188386) (leftOrdinal := 20)
    (rightResult := 188227) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨22143⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71329⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨22143⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge188453

namespace LeftMerge188455
def owner : Owner := ⟨.program ⟨257⟩, ⟨71330⟩⟩
def mergeEvent : Nat := 188455
def frameStart : Nat := 187711
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨22143⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68848⟩⟩] } }
def rhsRaw : List Term := Proof.Events735.exact188224RawTerms
def group : MergeGroup := .relation 188454
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 188454) (rhsResult := 188224)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨22143⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71329⟩⟩) ⟨68848⟩ 188224) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68848⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨22143⟩⟩], [⟨.program ⟨257⟩, ⟨68848⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge188455

namespace LeftMerge188456
def owner : Owner := ⟨.program ⟨257⟩, ⟨71330⟩⟩
def mergeEvent : Nat := 188456
def frameStart : Nat := 187711
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨18923⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩] } }
def leftRaw : List Term := Proof.Events735.exact188386RawTerms
def rightRaw : List Term := Proof.Events735.exact188227RawTerms
def group : MergeGroup := .operator 188386 188227
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 188386) (leftOrdinal := 19)
    (rightResult := 188227) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨18923⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71329⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨18923⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge188456

namespace LeftMerge188458
def owner : Owner := ⟨.program ⟨257⟩, ⟨71330⟩⟩
def mergeEvent : Nat := 188458
def frameStart : Nat := 187711
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨18923⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68848⟩⟩] } }
def rhsRaw : List Term := Proof.Events735.exact188224RawTerms
def group : MergeGroup := .relation 188457
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 188457) (rhsResult := 188224)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨18923⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71329⟩⟩) ⟨68848⟩ 188224) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68848⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨18923⟩⟩], [⟨.program ⟨257⟩, ⟨68848⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge188458

namespace LeftMerge188459
def owner : Owner := ⟨.program ⟨257⟩, ⟨71330⟩⟩
def mergeEvent : Nat := 188459
def frameStart : Nat := 187711
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨16083⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩] } }
def leftRaw : List Term := Proof.Events735.exact188386RawTerms
def rightRaw : List Term := Proof.Events735.exact188227RawTerms
def group : MergeGroup := .operator 188386 188227
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 188386) (leftOrdinal := 18)
    (rightResult := 188227) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨16083⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71329⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨16083⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge188459

namespace LeftMerge188461
def owner : Owner := ⟨.program ⟨257⟩, ⟨71330⟩⟩
def mergeEvent : Nat := 188461
def frameStart : Nat := 187711
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨16083⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68848⟩⟩] } }
def rhsRaw : List Term := Proof.Events735.exact188224RawTerms
def group : MergeGroup := .relation 188460
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 188460) (rhsResult := 188224)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨16083⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71329⟩⟩) ⟨68848⟩ 188224) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68848⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨16083⟩⟩], [⟨.program ⟨257⟩, ⟨68848⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge188461

namespace LeftMerge188469
def owner : Owner := ⟨.program ⟨257⟩, ⟨67516⟩⟩
def mergeEvent : Nat := 188469
def frameStart : Nat := 187711
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨67514⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events735.exact188238RawTerms
def rightRaw : List Term := Proof.Events736.exact188465RawTerms
def group : MergeGroup := .operator 188238 188465
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 188238) (leftOrdinal := 0)
    (rightResult := 188465) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨67514⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨67514⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge188469

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
