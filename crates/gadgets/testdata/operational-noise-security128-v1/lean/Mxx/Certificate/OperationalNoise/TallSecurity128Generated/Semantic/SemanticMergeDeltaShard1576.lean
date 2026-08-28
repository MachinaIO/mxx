import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge255501
def owner : Owner := ⟨.program ⟨257⟩, ⟨69188⟩⟩
def mergeEvent : Nat := 255501
def frameStart : Nat := 255406
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨25670⟩⟩, ⟨.program ⟨257⟩, ⟨65310⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68500⟩⟩] } }
def rhsRaw : List Term := Proof.Events997.exact255448RawTerms
def group : MergeGroup := .relation 255500
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 255500) (rhsResult := 255448)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨25670⟩⟩, ⟨.program ⟨257⟩, ⟨65310⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨69185⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨69185⟩⟩) ⟨68500⟩ 255448) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68500⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨25670⟩⟩, ⟨.program ⟨257⟩, ⟨65310⟩⟩], [⟨.program ⟨257⟩, ⟨68500⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge255501

namespace LeftMerge255509
def owner : Owner := ⟨.program ⟨257⟩, ⟨65750⟩⟩
def mergeEvent : Nat := 255509
def frameStart : Nat := 255406
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨65748⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events997.exact255462RawTerms
def rightRaw : List Term := Proof.Events998.exact255505RawTerms
def group : MergeGroup := .operator 255462 255505
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 255462) (leftOrdinal := 0)
    (rightResult := 255505) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨65748⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨65748⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge255509

namespace LeftMerge255526
def owner : Owner := ⟨.program ⟨257⟩, ⟨67723⟩⟩
def mergeEvent : Nat := 255526
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7188⟩⟩] } }
def rhsRaw : List Term := Proof.Events998.exact255523RawTerms
def group : MergeGroup := .relation 255525
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 255525) (rhsResult := 255523)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67720⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 255524 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67720⟩⟩]⟩) (none) 255523) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7188⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨7188⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge255526

namespace LeftMerge255527
def owner : Owner := ⟨.program ⟨257⟩, ⟨67723⟩⟩
def mergeEvent : Nat := 255527
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩, ⟨.program ⟨257⟩, ⟨69185⟩⟩] } }
def rhsRaw : List Term := Proof.Events998.exact255523RawTerms
def group : MergeGroup := .relation 255525
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 255525) (rhsResult := 255523)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67720⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 255524 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67720⟩⟩]⟩) (none) 255523) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩, ⟨.program ⟨257⟩, ⟨69185⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩, ⟨.program ⟨257⟩, ⟨69185⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge255527

namespace LeftMerge255528
def owner : Owner := ⟨.program ⟨257⟩, ⟨67723⟩⟩
def mergeEvent : Nat := 255528
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨25670⟩⟩, ⟨.program ⟨257⟩, ⟨65310⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68500⟩⟩] } }
def rhsRaw : List Term := Proof.Events998.exact255523RawTerms
def group : MergeGroup := .relation 255525
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 255525) (rhsResult := 255523)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67720⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 255524 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67720⟩⟩]⟩) (none) 255523) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25670⟩⟩, ⟨.program ⟨257⟩, ⟨65310⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68500⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨25670⟩⟩, ⟨.program ⟨257⟩, ⟨65310⟩⟩], [⟨.program ⟨257⟩, ⟨68500⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge255528

namespace LeftMerge255529
def owner : Owner := ⟨.program ⟨257⟩, ⟨67723⟩⟩
def mergeEvent : Nat := 255529
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨65748⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events998.exact255523RawTerms
def group : MergeGroup := .relation 255525
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 255525) (rhsResult := 255523)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67720⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 255524 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67720⟩⟩]⟩) (none) 255523) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨65748⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨65748⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge255529

namespace LeftMerge255534
def owner : Owner := ⟨.program ⟨257⟩, ⟨69187⟩⟩
def mergeEvent : Nat := 255534
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨25670⟩⟩, ⟨.program ⟨257⟩, ⟨65310⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68500⟩⟩] } }
def leftRaw : List Term := Proof.Events998.exact255530RawTerms
def rightRaw : List Term := Proof.Events997.exact255344RawTerms
def group : MergeGroup := .operator 255530 255344
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 255530) (leftOrdinal := 2)
    (rightResult := 255344) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨25670⟩⟩, ⟨.program ⟨257⟩, ⟨65310⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68500⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨25670⟩⟩, ⟨.program ⟨257⟩, ⟨65310⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68500⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨25670⟩⟩, ⟨.program ⟨257⟩, ⟨65310⟩⟩], [⟨.program ⟨257⟩, ⟨68500⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge255534

namespace LeftMerge255535
def owner : Owner := ⟨.program ⟨257⟩, ⟨69187⟩⟩
def mergeEvent : Nat := 255535
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩, ⟨.program ⟨257⟩, ⟨69185⟩⟩] } }
def leftRaw : List Term := Proof.Events998.exact255530RawTerms
def rightRaw : List Term := Proof.Events997.exact255344RawTerms
def group : MergeGroup := .operator 255530 255344
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 255530) (leftOrdinal := 1)
    (rightResult := 255344) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩, ⟨.program ⟨257⟩, ⟨69185⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩, ⟨.program ⟨257⟩, ⟨69185⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩, ⟨.program ⟨257⟩, ⟨69185⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge255535

namespace LeftMerge255543
def owner : Owner := ⟨.program ⟨257⟩, ⟨69784⟩⟩
def mergeEvent : Nat := 255543
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7188⟩⟩, ⟨.program ⟨257⟩, ⟨69782⟩⟩] } }
def leftRaw : List Term := Proof.Events998.exact255537RawTerms
def rightRaw : List Term := Proof.Events997.exact255260RawTerms
def group : MergeGroup := .operator 255537 255260
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 255537) (leftOrdinal := 0)
    (rightResult := 255260) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7188⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨69782⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨7188⟩⟩, ⟨.program ⟨257⟩, ⟨69782⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge255543

namespace LeftMerge255544
def owner : Owner := ⟨.program ⟨257⟩, ⟨69784⟩⟩
def mergeEvent : Nat := 255544
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨65748⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨69782⟩⟩] } }
def leftRaw : List Term := Proof.Events998.exact255537RawTerms
def rightRaw : List Term := Proof.Events997.exact255260RawTerms
def group : MergeGroup := .operator 255537 255260
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 255537) (leftOrdinal := 1)
    (rightResult := 255260) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨65748⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨69782⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨65748⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨69782⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge255544

namespace LeftMerge255546
def owner : Owner := ⟨.program ⟨257⟩, ⟨69784⟩⟩
def mergeEvent : Nat := 255546
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨65748⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68637⟩⟩] } }
def rhsRaw : List Term := Proof.Events997.exact255257RawTerms
def group : MergeGroup := .relation 255545
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 255545) (rhsResult := 255257)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨65748⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨69782⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨69782⟩⟩) ⟨68637⟩ 255257) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68637⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨65748⟩⟩], [⟨.program ⟨257⟩, ⟨68637⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge255546

namespace LeftMerge255560
def owner : Owner := ⟨.program ⟨257⟩, ⟨67980⟩⟩
def mergeEvent : Nat := 255560
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67977⟩⟩] } }
def leftRaw : List Term := Proof.Events982.exact251495RawTerms
def rightRaw : List Term := Proof.Events998.exact255554RawTerms
def group : MergeGroup := .operator 251495 255554
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 251495) (leftOrdinal := 0)
    (rightResult := 255554) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨67977⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67977⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge255560

namespace LeftMerge255681
def owner : Owner := ⟨.program ⟨257⟩, ⟨68989⟩⟩
def mergeEvent : Nat := 255681
def frameStart : Nat := 255615
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨65748⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events998.exact255677RawTerms
def rightRaw : List Term := Proof.Events998.exact255675RawTerms
def group : MergeGroup := .operator 255677 255675
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 255677) (leftOrdinal := 0)
    (rightResult := 255675) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨65748⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨65748⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge255681

namespace LeftMerge255693
def owner : Owner := ⟨.program ⟨257⟩, ⟨69783⟩⟩
def mergeEvent : Nat := 255693
def frameStart : Nat := 255615
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7188⟩⟩, ⟨.program ⟨257⟩, ⟨69782⟩⟩] } }
def leftRaw : List Term := Proof.Events998.exact255689RawTerms
def rightRaw : List Term := Proof.Events998.exact255666RawTerms
def group : MergeGroup := .operator 255689 255666
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 255689) (leftOrdinal := 0)
    (rightResult := 255666) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7188⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨69782⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7188⟩⟩, ⟨.program ⟨257⟩, ⟨69782⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge255693

namespace LeftMerge255694
def owner : Owner := ⟨.program ⟨257⟩, ⟨69783⟩⟩
def mergeEvent : Nat := 255694
def frameStart : Nat := 255615
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨65748⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨69782⟩⟩] } }
def leftRaw : List Term := Proof.Events998.exact255689RawTerms
def rightRaw : List Term := Proof.Events998.exact255666RawTerms
def group : MergeGroup := .operator 255689 255666
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 255689) (leftOrdinal := 1)
    (rightResult := 255666) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨65748⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨69782⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨65748⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨69782⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge255694

namespace LeftMerge255696
def owner : Owner := ⟨.program ⟨257⟩, ⟨69783⟩⟩
def mergeEvent : Nat := 255696
def frameStart : Nat := 255615
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨65748⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68637⟩⟩] } }
def rhsRaw : List Term := Proof.Events998.exact255663RawTerms
def group : MergeGroup := .relation 255695
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 255695) (rhsResult := 255663)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨65748⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨69782⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨69782⟩⟩) ⟨68637⟩ 255663) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68637⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨65748⟩⟩], [⟨.program ⟨257⟩, ⟨68637⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge255696

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
