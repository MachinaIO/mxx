import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge124293
def owner : Owner := ⟨.program ⟨257⟩, ⟨62358⟩⟩
def mergeEvent : Nat := 124293
def frameStart : Nat := 124263
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨25442⟩⟩, ⟨.program ⟨257⟩, ⟨62357⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events485.exact124289RawTerms
def rightRaw : List Term := Proof.Events485.exact124286RawTerms
def group : MergeGroup := .operator 124289 124286
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 124289) (leftOrdinal := 0)
    (rightResult := 124286) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨62357⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25442⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨25442⟩⟩, ⟨.program ⟨257⟩, ⟨62357⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge124293

namespace LeftMerge124323
def owner : Owner := ⟨.program ⟨257⟩, ⟨64192⟩⟩
def mergeEvent : Nat := 124323
def frameStart : Nat := 124263
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨25442⟩⟩, ⟨.program ⟨257⟩, ⟨62357⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events485.exact124319RawTerms
def rightRaw : List Term := Proof.Events485.exact124317RawTerms
def group : MergeGroup := .operator 124319 124317
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 124319) (leftOrdinal := 0)
    (rightResult := 124317) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25442⟩⟩, ⟨.program ⟨257⟩, ⟨62357⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨25442⟩⟩, ⟨.program ⟨257⟩, ⟨62357⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge124323

namespace LeftMerge124346
def owner : Owner := ⟨.program ⟨257⟩, ⟨9540⟩⟩
def mergeEvent : Nat := 124346
def frameStart : Nat := 124263
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩] } }
def leftRaw : List Term := Proof.Events485.exact124342RawTerms
def rightRaw : List Term := Proof.Events485.exact124339RawTerms
def group : MergeGroup := .operator 124342 124339
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 124342) (leftOrdinal := 0)
    (rightResult := 124339) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7293⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9538⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge124346

namespace LeftMerge124355
def owner : Owner := ⟨.program ⟨257⟩, ⟨64398⟩⟩
def mergeEvent : Nat := 124355
def frameStart : Nat := 124263
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩, ⟨.program ⟨257⟩, ⟨64395⟩⟩] } }
def leftRaw : List Term := Proof.Events485.exact124351RawTerms
def rightRaw : List Term := Proof.Events485.exact124308RawTerms
def group : MergeGroup := .operator 124351 124308
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 124351) (leftOrdinal := 0)
    (rightResult := 124308) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨64395⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩, ⟨.program ⟨257⟩, ⟨64395⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge124355

namespace LeftMerge124356
def owner : Owner := ⟨.program ⟨257⟩, ⟨64398⟩⟩
def mergeEvent : Nat := 124356
def frameStart : Nat := 124263
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨25442⟩⟩, ⟨.program ⟨257⟩, ⟨62357⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨64395⟩⟩] } }
def leftRaw : List Term := Proof.Events485.exact124351RawTerms
def rightRaw : List Term := Proof.Events485.exact124308RawTerms
def group : MergeGroup := .operator 124351 124308
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 124351) (leftOrdinal := 1)
    (rightResult := 124308) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25442⟩⟩, ⟨.program ⟨257⟩, ⟨62357⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨64395⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨25442⟩⟩, ⟨.program ⟨257⟩, ⟨62357⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨64395⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge124356

namespace LeftMerge124358
def owner : Owner := ⟨.program ⟨257⟩, ⟨64398⟩⟩
def mergeEvent : Nat := 124358
def frameStart : Nat := 124263
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨25442⟩⟩, ⟨.program ⟨257⟩, ⟨62357⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨63905⟩⟩] } }
def rhsRaw : List Term := Proof.Events485.exact124305RawTerms
def group : MergeGroup := .relation 124357
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 124357) (rhsResult := 124305)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨25442⟩⟩, ⟨.program ⟨257⟩, ⟨62357⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨64395⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨64395⟩⟩) ⟨63905⟩ 124305) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨63905⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨25442⟩⟩, ⟨.program ⟨257⟩, ⟨62357⟩⟩], [⟨.program ⟨257⟩, ⟨63905⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge124358

namespace LeftMerge124366
def owner : Owner := ⟨.program ⟨257⟩, ⟨62778⟩⟩
def mergeEvent : Nat := 124366
def frameStart : Nat := 124263
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨62776⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events485.exact124319RawTerms
def rightRaw : List Term := Proof.Events485.exact124362RawTerms
def group : MergeGroup := .operator 124319 124362
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 124319) (leftOrdinal := 0)
    (rightResult := 124362) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨62776⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨62776⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge124366

namespace LeftMerge124383
def owner : Owner := ⟨.program ⟨257⟩, ⟨63332⟩⟩
def mergeEvent : Nat := 124383
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7187⟩⟩] } }
def rhsRaw : List Term := Proof.Events485.exact124380RawTerms
def group : MergeGroup := .relation 124382
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 124382) (rhsResult := 124380)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63329⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 124381 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63329⟩⟩]⟩) (none) 124380) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7187⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7187⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge124383

namespace LeftMerge124384
def owner : Owner := ⟨.program ⟨257⟩, ⟨63332⟩⟩
def mergeEvent : Nat := 124384
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩, ⟨.program ⟨257⟩, ⟨64395⟩⟩] } }
def rhsRaw : List Term := Proof.Events485.exact124380RawTerms
def group : MergeGroup := .relation 124382
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 124382) (rhsResult := 124380)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63329⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 124381 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63329⟩⟩]⟩) (none) 124380) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩, ⟨.program ⟨257⟩, ⟨64395⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩, ⟨.program ⟨257⟩, ⟨64395⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge124384

namespace LeftMerge124385
def owner : Owner := ⟨.program ⟨257⟩, ⟨63332⟩⟩
def mergeEvent : Nat := 124385
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨25442⟩⟩, ⟨.program ⟨257⟩, ⟨62357⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨63905⟩⟩] } }
def rhsRaw : List Term := Proof.Events485.exact124380RawTerms
def group : MergeGroup := .relation 124382
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 124382) (rhsResult := 124380)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63329⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 124381 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63329⟩⟩]⟩) (none) 124380) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25442⟩⟩, ⟨.program ⟨257⟩, ⟨62357⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨63905⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨25442⟩⟩, ⟨.program ⟨257⟩, ⟨62357⟩⟩], [⟨.program ⟨257⟩, ⟨63905⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge124385

namespace LeftMerge124386
def owner : Owner := ⟨.program ⟨257⟩, ⟨63332⟩⟩
def mergeEvent : Nat := 124386
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨62776⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events485.exact124380RawTerms
def group : MergeGroup := .relation 124382
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 124382) (rhsResult := 124380)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63329⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 124381 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63329⟩⟩]⟩) (none) 124380) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨62776⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨62776⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge124386

namespace LeftMerge124391
def owner : Owner := ⟨.program ⟨257⟩, ⟨64397⟩⟩
def mergeEvent : Nat := 124391
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨25442⟩⟩, ⟨.program ⟨257⟩, ⟨62357⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨63905⟩⟩] } }
def leftRaw : List Term := Proof.Events485.exact124387RawTerms
def rightRaw : List Term := Proof.Events485.exact124201RawTerms
def group : MergeGroup := .operator 124387 124201
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 124387) (leftOrdinal := 2)
    (rightResult := 124201) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨25442⟩⟩, ⟨.program ⟨257⟩, ⟨62357⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨63905⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨25442⟩⟩, ⟨.program ⟨257⟩, ⟨62357⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨63905⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨25442⟩⟩, ⟨.program ⟨257⟩, ⟨62357⟩⟩], [⟨.program ⟨257⟩, ⟨63905⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge124391

namespace LeftMerge124392
def owner : Owner := ⟨.program ⟨257⟩, ⟨64397⟩⟩
def mergeEvent : Nat := 124392
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩, ⟨.program ⟨257⟩, ⟨64395⟩⟩] } }
def leftRaw : List Term := Proof.Events485.exact124387RawTerms
def rightRaw : List Term := Proof.Events485.exact124201RawTerms
def group : MergeGroup := .operator 124387 124201
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 124387) (leftOrdinal := 1)
    (rightResult := 124201) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩, ⟨.program ⟨257⟩, ⟨64395⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩, ⟨.program ⟨257⟩, ⟨64395⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩, ⟨.program ⟨257⟩, ⟨64395⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge124392

namespace LeftMerge124400
def owner : Owner := ⟨.program ⟨257⟩, ⟨64750⟩⟩
def mergeEvent : Nat := 124400
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7187⟩⟩, ⟨.program ⟨257⟩, ⟨64748⟩⟩] } }
def leftRaw : List Term := Proof.Events485.exact124394RawTerms
def rightRaw : List Term := Proof.Events484.exact124117RawTerms
def group : MergeGroup := .operator 124394 124117
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 124394) (leftOrdinal := 0)
    (rightResult := 124117) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7187⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨64748⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7187⟩⟩, ⟨.program ⟨257⟩, ⟨64748⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge124400

namespace LeftMerge124401
def owner : Owner := ⟨.program ⟨257⟩, ⟨64750⟩⟩
def mergeEvent : Nat := 124401
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨62776⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨64748⟩⟩] } }
def leftRaw : List Term := Proof.Events485.exact124394RawTerms
def rightRaw : List Term := Proof.Events484.exact124117RawTerms
def group : MergeGroup := .operator 124394 124117
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 124394) (leftOrdinal := 1)
    (rightResult := 124117) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨62776⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨64748⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨62776⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨64748⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge124401

namespace LeftMerge124403
def owner : Owner := ⟨.program ⟨257⟩, ⟨64750⟩⟩
def mergeEvent : Nat := 124403
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨62776⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨64045⟩⟩] } }
def rhsRaw : List Term := Proof.Events484.exact124114RawTerms
def group : MergeGroup := .relation 124402
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 124402) (rhsResult := 124114)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨62776⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨64748⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨64748⟩⟩) ⟨64045⟩ 124114) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨64045⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨62776⟩⟩], [⟨.program ⟨257⟩, ⟨64045⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge124403

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
