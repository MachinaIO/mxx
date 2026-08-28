import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge286215
def owner : Owner := ⟨.program ⟨257⟩, ⟨58728⟩⟩
def mergeEvent : Nat := 286215
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7185⟩⟩, ⟨.program ⟨257⟩, ⟨58726⟩⟩] } }
def leftRaw : List Term := Proof.Events1118.exact286209RawTerms
def rightRaw : List Term := Proof.Events1116.exact285934RawTerms
def group : MergeGroup := .operator 286209 285934
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 286209) (leftOrdinal := 0)
    (rightResult := 285934) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7185⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨58726⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7185⟩⟩, ⟨.program ⟨257⟩, ⟨58726⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge286215

namespace LeftMerge286216
def owner : Owner := ⟨.program ⟨257⟩, ⟨58728⟩⟩
def mergeEvent : Nat := 286216
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨56800⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨58726⟩⟩] } }
def leftRaw : List Term := Proof.Events1118.exact286209RawTerms
def rightRaw : List Term := Proof.Events1116.exact285934RawTerms
def group : MergeGroup := .operator 286209 285934
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 286209) (leftOrdinal := 1)
    (rightResult := 285934) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨56800⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨58726⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨56800⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨58726⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge286216

namespace LeftMerge286218
def owner : Owner := ⟨.program ⟨257⟩, ⟨58728⟩⟩
def mergeEvent : Nat := 286218
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨56800⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨58067⟩⟩] } }
def rhsRaw : List Term := Proof.Events1116.exact285931RawTerms
def group : MergeGroup := .relation 286217
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 286217) (rhsResult := 285931)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨56800⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨58726⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨58726⟩⟩) ⟨58067⟩ 285931) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨58067⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨56800⟩⟩], [⟨.program ⟨257⟩, ⟨58067⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge286218

namespace LeftMerge286232
def owner : Owner := ⟨.program ⟨257⟩, ⟨57599⟩⟩
def mergeEvent : Nat := 286232
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57596⟩⟩] } }
def leftRaw : List Term := Proof.Events1096.exact280745RawTerms
def rightRaw : List Term := Proof.Events1118.exact286226RawTerms
def group : MergeGroup := .operator 280745 286226
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 280745) (leftOrdinal := 0)
    (rightResult := 286226) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨57596⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57596⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge286232

namespace LeftMerge286353
def owner : Owner := ⟨.program ⟨257⟩, ⟨58304⟩⟩
def mergeEvent : Nat := 286353
def frameStart : Nat := 286287
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨56800⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1118.exact286349RawTerms
def rightRaw : List Term := Proof.Events1118.exact286347RawTerms
def group : MergeGroup := .operator 286349 286347
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 286349) (leftOrdinal := 0)
    (rightResult := 286347) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨56800⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨56800⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge286353

namespace LeftMerge286365
def owner : Owner := ⟨.program ⟨257⟩, ⟨58727⟩⟩
def mergeEvent : Nat := 286365
def frameStart : Nat := 286287
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7185⟩⟩, ⟨.program ⟨257⟩, ⟨58726⟩⟩] } }
def leftRaw : List Term := Proof.Events1118.exact286361RawTerms
def rightRaw : List Term := Proof.Events1118.exact286338RawTerms
def group : MergeGroup := .operator 286361 286338
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 286361) (leftOrdinal := 0)
    (rightResult := 286338) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7185⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨58726⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7185⟩⟩, ⟨.program ⟨257⟩, ⟨58726⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge286365

namespace LeftMerge286366
def owner : Owner := ⟨.program ⟨257⟩, ⟨58727⟩⟩
def mergeEvent : Nat := 286366
def frameStart : Nat := 286287
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨56800⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨58726⟩⟩] } }
def leftRaw : List Term := Proof.Events1118.exact286361RawTerms
def rightRaw : List Term := Proof.Events1118.exact286338RawTerms
def group : MergeGroup := .operator 286361 286338
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 286361) (leftOrdinal := 1)
    (rightResult := 286338) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨56800⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨58726⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨56800⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨58726⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge286366

namespace LeftMerge286368
def owner : Owner := ⟨.program ⟨257⟩, ⟨58727⟩⟩
def mergeEvent : Nat := 286368
def frameStart : Nat := 286287
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨56800⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨58067⟩⟩] } }
def rhsRaw : List Term := Proof.Events1118.exact286335RawTerms
def group : MergeGroup := .relation 286367
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 286367) (rhsResult := 286335)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨56800⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨58726⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨58726⟩⟩) ⟨58067⟩ 286335) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨58067⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨56800⟩⟩], [⟨.program ⟨257⟩, ⟨58067⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge286368

namespace LeftMerge286376
def owner : Owner := ⟨.program ⟨257⟩, ⟨57009⟩⟩
def mergeEvent : Nat := 286376
def frameStart : Nat := 286287
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨57007⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1118.exact286349RawTerms
def rightRaw : List Term := Proof.Events1118.exact286372RawTerms
def group : MergeGroup := .operator 286349 286372
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 286349) (leftOrdinal := 0)
    (rightResult := 286372) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨57007⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨57007⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge286376

namespace LeftMerge286393
def owner : Owner := ⟨.program ⟨257⟩, ⟨57599⟩⟩
def mergeEvent : Nat := 286393
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7210⟩⟩] } }
def rhsRaw : List Term := Proof.Events1118.exact286390RawTerms
def group : MergeGroup := .relation 286392
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 286392) (rhsResult := 286390)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57596⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 286391 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57596⟩⟩]⟩) (none) 286390) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7210⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge286393

namespace LeftMerge286394
def owner : Owner := ⟨.program ⟨257⟩, ⟨57599⟩⟩
def mergeEvent : Nat := 286394
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7185⟩⟩, ⟨.program ⟨257⟩, ⟨58726⟩⟩] } }
def rhsRaw : List Term := Proof.Events1118.exact286390RawTerms
def group : MergeGroup := .relation 286392
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 286392) (rhsResult := 286390)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57596⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 286391 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57596⟩⟩]⟩) (none) 286390) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7185⟩⟩, ⟨.program ⟨257⟩, ⟨58726⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7185⟩⟩, ⟨.program ⟨257⟩, ⟨58726⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge286394

namespace LeftMerge286395
def owner : Owner := ⟨.program ⟨257⟩, ⟨57599⟩⟩
def mergeEvent : Nat := 286395
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨56800⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨58067⟩⟩] } }
def rhsRaw : List Term := Proof.Events1118.exact286390RawTerms
def group : MergeGroup := .relation 286392
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 286392) (rhsResult := 286390)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57596⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 286391 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57596⟩⟩]⟩) (none) 286390) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨56800⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨58067⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨56800⟩⟩], [⟨.program ⟨257⟩, ⟨58067⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge286395

namespace LeftMerge286396
def owner : Owner := ⟨.program ⟨257⟩, ⟨57599⟩⟩
def mergeEvent : Nat := 286396
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨57007⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events1118.exact286390RawTerms
def group : MergeGroup := .relation 286392
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 286392) (rhsResult := 286390)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57596⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 286391 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57596⟩⟩]⟩) (none) 286390) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨57007⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨57007⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge286396

namespace LeftMerge286401
def owner : Owner := ⟨.program ⟨257⟩, ⟨58729⟩⟩
def mergeEvent : Nat := 286401
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7185⟩⟩, ⟨.program ⟨257⟩, ⟨58726⟩⟩] } }
def leftRaw : List Term := Proof.Events1118.exact286397RawTerms
def rightRaw : List Term := Proof.Events1118.exact286219RawTerms
def group : MergeGroup := .operator 286397 286219
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 286397) (leftOrdinal := 0)
    (rightResult := 286219) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7185⟩⟩, ⟨.program ⟨257⟩, ⟨58726⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7185⟩⟩, ⟨.program ⟨257⟩, ⟨58726⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7185⟩⟩, ⟨.program ⟨257⟩, ⟨58726⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge286401

namespace LeftMerge286402
def owner : Owner := ⟨.program ⟨257⟩, ⟨58729⟩⟩
def mergeEvent : Nat := 286402
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨56800⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨58067⟩⟩] } }
def leftRaw : List Term := Proof.Events1118.exact286397RawTerms
def rightRaw : List Term := Proof.Events1118.exact286219RawTerms
def group : MergeGroup := .operator 286397 286219
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 286397) (leftOrdinal := 2)
    (rightResult := 286219) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨56800⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨58067⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨56800⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨58067⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨56800⟩⟩], [⟨.program ⟨257⟩, ⟨58067⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge286402

namespace LeftMerge286428
def owner : Owner := ⟨.program ⟨257⟩, ⟨24699⟩⟩
def mergeEvent : Nat := 286428
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨24698⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events054.exact13828RawTerms
def rightRaw : List Term := Proof.Events1096.exact280653RawTerms
def group : MergeGroup := .operator 13828 280653
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 13828) (leftOrdinal := 0)
    (rightResult := 280653) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24698⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨24698⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge286428

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
