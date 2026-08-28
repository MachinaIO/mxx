import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge296472
def owner : Owner := ⟨.program ⟨257⟩, ⟨14035⟩⟩
def mergeEvent : Nat := 296472
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨14031⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7282⟩⟩] } }
def rhsRaw : List Term := Proof.Events072.exact18583RawTerms
def group : MergeGroup := .relation 296471
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 296471) (rhsResult := 18583)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨14031⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨9556⟩⟩) ⟨7282⟩ 18583) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7282⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨14031⟩⟩], [⟨.program ⟨257⟩, ⟨7282⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge296472

namespace LeftMerge296473
def owner : Owner := ⟨.program ⟨257⟩, ⟨14035⟩⟩
def mergeEvent : Nat := 296473
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩] } }
def leftRaw : List Term := Proof.Events1158.exact296464RawTerms
def rightRaw : List Term := Proof.Events072.exact18613RawTerms
def group : MergeGroup := .operator 296464 18613
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 296464) (leftOrdinal := 0)
    (rightResult := 18613) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9556⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge296473

namespace LeftMerge296478
def owner : Owner := ⟨.program ⟨257⟩, ⟨39561⟩⟩
def mergeEvent : Nat := 296478
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨14031⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7282⟩⟩] } }
def leftRaw : List Term := Proof.Events1158.exact296474RawTerms
def rightRaw : List Term := Proof.Events1157.exact296444RawTerms
def group : MergeGroup := .operator 296474 296444
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 296474) (leftOrdinal := 1)
    (rightResult := 296444) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨14031⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7282⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨14031⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7282⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨14031⟩⟩], [⟨.program ⟨257⟩, ⟨7282⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge296478

namespace LeftMerge296486
def owner : Owner := ⟨.program ⟨257⟩, ⟨41510⟩⟩
def mergeEvent : Nat := 296486
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨14031⟩⟩, ⟨.program ⟨257⟩, ⟨39554⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨41509⟩⟩] } }
def leftRaw : List Term := Proof.Events1158.exact296480RawTerms
def rightRaw : List Term := Proof.Events1157.exact296416RawTerms
def group : MergeGroup := .operator 296480 296416
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 296480) (leftOrdinal := 1)
    (rightResult := 296416) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨14031⟩⟩, ⟨.program ⟨257⟩, ⟨39554⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨41509⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨14031⟩⟩, ⟨.program ⟨257⟩, ⟨39554⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨41509⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge296486

namespace LeftMerge296488
def owner : Owner := ⟨.program ⟨257⟩, ⟨41510⟩⟩
def mergeEvent : Nat := 296488
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨14031⟩⟩, ⟨.program ⟨257⟩, ⟨39554⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41049⟩⟩] } }
def rhsRaw : List Term := Proof.Events1157.exact296413RawTerms
def group : MergeGroup := .relation 296487
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 296487) (rhsResult := 296413)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨14031⟩⟩, ⟨.program ⟨257⟩, ⟨39554⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨41509⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨41509⟩⟩) ⟨41049⟩ 296413) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨41049⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨14031⟩⟩, ⟨.program ⟨257⟩, ⟨39554⟩⟩], [⟨.program ⟨257⟩, ⟨41049⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge296488

namespace LeftMerge296489
def owner : Owner := ⟨.program ⟨257⟩, ⟨41510⟩⟩
def mergeEvent : Nat := 296489
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41509⟩⟩] } }
def leftRaw : List Term := Proof.Events1158.exact296480RawTerms
def rightRaw : List Term := Proof.Events1157.exact296416RawTerms
def group : MergeGroup := .operator 296480 296416
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 296480) (leftOrdinal := 0)
    (rightResult := 296416) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨41509⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41509⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge296489

namespace LeftMerge296503
def owner : Owner := ⟨.program ⟨257⟩, ⟨40452⟩⟩
def mergeEvent : Nat := 296503
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40449⟩⟩] } }
def leftRaw : List Term := Proof.Events1153.exact295195RawTerms
def rightRaw : List Term := Proof.Events1158.exact296497RawTerms
def group : MergeGroup := .operator 295195 296497
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 295195) (leftOrdinal := 0)
    (rightResult := 296497) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨40449⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40449⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge296503

namespace LeftMerge296558
def owner : Owner := ⟨.program ⟨257⟩, ⟨39555⟩⟩
def mergeEvent : Nat := 296558
def frameStart : Nat := 296540
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14031⟩⟩, ⟨.program ⟨257⟩, ⟨39554⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events1158.exact296554RawTerms
def rightRaw : List Term := Proof.Events1158.exact296551RawTerms
def group : MergeGroup := .operator 296554 296551
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 296554) (leftOrdinal := 0)
    (rightResult := 296551) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14031⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨39554⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨14031⟩⟩, ⟨.program ⟨257⟩, ⟨39554⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge296558

namespace LeftMerge296588
def owner : Owner := ⟨.program ⟨257⟩, ⟨41348⟩⟩
def mergeEvent : Nat := 296588
def frameStart : Nat := 296540
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14031⟩⟩, ⟨.program ⟨257⟩, ⟨39554⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1158.exact296584RawTerms
def rightRaw : List Term := Proof.Events1158.exact296582RawTerms
def group : MergeGroup := .operator 296584 296582
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 296584) (leftOrdinal := 0)
    (rightResult := 296582) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14031⟩⟩, ⟨.program ⟨257⟩, ⟨39554⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨14031⟩⟩, ⟨.program ⟨257⟩, ⟨39554⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge296588

namespace LeftMerge296611
def owner : Owner := ⟨.program ⟨257⟩, ⟨9558⟩⟩
def mergeEvent : Nat := 296611
def frameStart : Nat := 296540
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩] } }
def leftRaw : List Term := Proof.Events1158.exact296607RawTerms
def rightRaw : List Term := Proof.Events1158.exact296604RawTerms
def group : MergeGroup := .operator 296607 296604
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 296607) (leftOrdinal := 0)
    (rightResult := 296604) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9556⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge296611

namespace LeftMerge296620
def owner : Owner := ⟨.program ⟨257⟩, ⟨41512⟩⟩
def mergeEvent : Nat := 296620
def frameStart : Nat := 296540
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41509⟩⟩] } }
def leftRaw : List Term := Proof.Events1158.exact296616RawTerms
def rightRaw : List Term := Proof.Events1158.exact296573RawTerms
def group : MergeGroup := .operator 296616 296573
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 296616) (leftOrdinal := 0)
    (rightResult := 296573) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨41509⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41509⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge296620

namespace LeftMerge296621
def owner : Owner := ⟨.program ⟨257⟩, ⟨41512⟩⟩
def mergeEvent : Nat := 296621
def frameStart : Nat := 296540
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14031⟩⟩, ⟨.program ⟨257⟩, ⟨39554⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨41509⟩⟩] } }
def leftRaw : List Term := Proof.Events1158.exact296616RawTerms
def rightRaw : List Term := Proof.Events1158.exact296573RawTerms
def group : MergeGroup := .operator 296616 296573
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 296616) (leftOrdinal := 1)
    (rightResult := 296573) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14031⟩⟩, ⟨.program ⟨257⟩, ⟨39554⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨41509⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨14031⟩⟩, ⟨.program ⟨257⟩, ⟨39554⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨41509⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge296621

namespace LeftMerge296623
def owner : Owner := ⟨.program ⟨257⟩, ⟨41512⟩⟩
def mergeEvent : Nat := 296623
def frameStart : Nat := 296540
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14031⟩⟩, ⟨.program ⟨257⟩, ⟨39554⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41049⟩⟩] } }
def rhsRaw : List Term := Proof.Events1158.exact296570RawTerms
def group : MergeGroup := .relation 296622
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 296622) (rhsResult := 296570)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨14031⟩⟩, ⟨.program ⟨257⟩, ⟨39554⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨41509⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨41509⟩⟩) ⟨41049⟩ 296570) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨41049⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨14031⟩⟩, ⟨.program ⟨257⟩, ⟨39554⟩⟩], [⟨.program ⟨257⟩, ⟨41049⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge296623

namespace LeftMerge296631
def owner : Owner := ⟨.program ⟨257⟩, ⟨40030⟩⟩
def mergeEvent : Nat := 296631
def frameStart : Nat := 296540
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨40028⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1158.exact296584RawTerms
def rightRaw : List Term := Proof.Events1158.exact296627RawTerms
def group : MergeGroup := .operator 296584 296627
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 296584) (leftOrdinal := 0)
    (rightResult := 296627) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨40028⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨40028⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge296631

namespace LeftMerge296648
def owner : Owner := ⟨.program ⟨257⟩, ⟨40452⟩⟩
def mergeEvent : Nat := 296648
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7193⟩⟩] } }
def rhsRaw : List Term := Proof.Events1158.exact296645RawTerms
def group : MergeGroup := .relation 296647
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 296647) (rhsResult := 296645)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40449⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 296646 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40449⟩⟩]⟩) (none) 296645) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7193⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7193⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge296648

namespace LeftMerge296649
def owner : Owner := ⟨.program ⟨257⟩, ⟨40452⟩⟩
def mergeEvent : Nat := 296649
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41509⟩⟩] } }
def rhsRaw : List Term := Proof.Events1158.exact296645RawTerms
def group : MergeGroup := .relation 296647
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 296647) (rhsResult := 296645)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40449⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 296646 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40449⟩⟩]⟩) (none) 296645) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41509⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41509⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge296649

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
