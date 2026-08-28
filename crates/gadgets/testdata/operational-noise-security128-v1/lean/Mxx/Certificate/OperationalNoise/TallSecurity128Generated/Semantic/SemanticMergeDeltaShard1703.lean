import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge276270
def owner : Owner := ⟨.program ⟨257⟩, ⟨68290⟩⟩
def mergeEvent : Nat := 276270
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨21929⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68780⟩⟩] } }
def rhsRaw : List Term := Proof.Events1079.exact276233RawTerms
def group : MergeGroup := .relation 276235
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 276235) (rhsResult := 276233)
    (sourceTermOrdinal := 21) (source := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68287⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 276234 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68287⟩⟩]⟩) (none) 276233) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨21929⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68780⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨21929⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge276270

namespace LeftMerge276271
def owner : Owner := ⟨.program ⟨257⟩, ⟨68290⟩⟩
def mergeEvent : Nat := 276271
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨18709⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68780⟩⟩] } }
def rhsRaw : List Term := Proof.Events1079.exact276233RawTerms
def group : MergeGroup := .relation 276235
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 276235) (rhsResult := 276233)
    (sourceTermOrdinal := 20) (source := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68287⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 276234 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68287⟩⟩]⟩) (none) 276233) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨18709⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68780⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨18709⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge276271

namespace LeftMerge276272
def owner : Owner := ⟨.program ⟨257⟩, ⟨68290⟩⟩
def mergeEvent : Nat := 276272
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨15903⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68780⟩⟩] } }
def rhsRaw : List Term := Proof.Events1079.exact276233RawTerms
def group : MergeGroup := .relation 276235
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 276235) (rhsResult := 276233)
    (sourceTermOrdinal := 19) (source := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68287⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 276234 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68287⟩⟩]⟩) (none) 276233) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨15903⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68780⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨15903⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge276272

namespace LeftMerge276273
def owner : Owner := ⟨.program ⟨257⟩, ⟨68290⟩⟩
def mergeEvent : Nat := 276273
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨67300⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events1079.exact276233RawTerms
def group : MergeGroup := .relation 276235
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 276235) (rhsResult := 276233)
    (sourceTermOrdinal := 37) (source := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68287⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 276234 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68287⟩⟩]⟩) (none) 276233) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨67300⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨67300⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge276273

namespace LeftMerge276278
def owner : Owner := ⟨.program ⟨257⟩, ⟨70982⟩⟩
def mergeEvent : Nat := 276278
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7232⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩] } }
def leftRaw : List Term := Proof.Events1079.exact276274RawTerms
def rightRaw : List Term := Proof.Events1073.exact274858RawTerms
def group : MergeGroup := .operator 276274 274858
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 276274) (leftOrdinal := 17)
    (rightResult := 274858) (rightOrdinal := 17) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7232⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7232⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7232⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge276278

namespace LeftMerge276279
def owner : Owner := ⟨.program ⟨257⟩, ⟨70982⟩⟩
def mergeEvent : Nat := 276279
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨48256⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68780⟩⟩] } }
def leftRaw : List Term := Proof.Events1079.exact276274RawTerms
def rightRaw : List Term := Proof.Events1073.exact274858RawTerms
def group : MergeGroup := .operator 276274 274858
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 276274) (leftOrdinal := 30)
    (rightResult := 274858) (rightOrdinal := 29) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨48256⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68780⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨48256⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68780⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨48256⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge276279

namespace LeftMerge276280
def owner : Owner := ⟨.program ⟨257⟩, ⟨70982⟩⟩
def mergeEvent : Nat := 276280
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩] } }
def leftRaw : List Term := Proof.Events1079.exact276274RawTerms
def rightRaw : List Term := Proof.Events1073.exact274858RawTerms
def group : MergeGroup := .operator 276274 274858
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 276274) (leftOrdinal := 16)
    (rightResult := 274858) (rightOrdinal := 16) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge276280

namespace LeftMerge276281
def owner : Owner := ⟨.program ⟨257⟩, ⟨70982⟩⟩
def mergeEvent : Nat := 276281
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨45576⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68780⟩⟩] } }
def leftRaw : List Term := Proof.Events1079.exact276274RawTerms
def rightRaw : List Term := Proof.Events1073.exact274858RawTerms
def group : MergeGroup := .operator 276274 274858
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 276274) (leftOrdinal := 29)
    (rightResult := 274858) (rightOrdinal := 28) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨45576⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68780⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨45576⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68780⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨45576⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge276281

namespace LeftMerge276282
def owner : Owner := ⟨.program ⟨257⟩, ⟨70982⟩⟩
def mergeEvent : Nat := 276282
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩] } }
def leftRaw : List Term := Proof.Events1079.exact276274RawTerms
def rightRaw : List Term := Proof.Events1073.exact274858RawTerms
def group : MergeGroup := .operator 276274 274858
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 276274) (leftOrdinal := 15)
    (rightResult := 274858) (rightOrdinal := 15) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge276282

namespace LeftMerge276283
def owner : Owner := ⟨.program ⟨257⟩, ⟨70982⟩⟩
def mergeEvent : Nat := 276283
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨42892⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68780⟩⟩] } }
def leftRaw : List Term := Proof.Events1079.exact276274RawTerms
def rightRaw : List Term := Proof.Events1073.exact274858RawTerms
def group : MergeGroup := .operator 276274 274858
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 276274) (leftOrdinal := 28)
    (rightResult := 274858) (rightOrdinal := 27) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨42892⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68780⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨42892⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68780⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨42892⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge276283

namespace LeftMerge276284
def owner : Owner := ⟨.program ⟨257⟩, ⟨70982⟩⟩
def mergeEvent : Nat := 276284
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩] } }
def leftRaw : List Term := Proof.Events1079.exact276274RawTerms
def rightRaw : List Term := Proof.Events1073.exact274858RawTerms
def group : MergeGroup := .operator 276274 274858
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 276274) (leftOrdinal := 14)
    (rightResult := 274858) (rightOrdinal := 14) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge276284

namespace LeftMerge276285
def owner : Owner := ⟨.program ⟨257⟩, ⟨70982⟩⟩
def mergeEvent : Nat := 276285
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨40212⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68780⟩⟩] } }
def leftRaw : List Term := Proof.Events1079.exact276274RawTerms
def rightRaw : List Term := Proof.Events1073.exact274858RawTerms
def group : MergeGroup := .operator 276274 274858
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 276274) (leftOrdinal := 27)
    (rightResult := 274858) (rightOrdinal := 26) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨40212⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68780⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨40212⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68780⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨40212⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge276285

namespace LeftMerge276286
def owner : Owner := ⟨.program ⟨257⟩, ⟨70982⟩⟩
def mergeEvent : Nat := 276286
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩] } }
def leftRaw : List Term := Proof.Events1079.exact276274RawTerms
def rightRaw : List Term := Proof.Events1073.exact274858RawTerms
def group : MergeGroup := .operator 276274 274858
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 276274) (leftOrdinal := 13)
    (rightResult := 274858) (rightOrdinal := 13) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge276286

namespace LeftMerge276287
def owner : Owner := ⟨.program ⟨257⟩, ⟨70982⟩⟩
def mergeEvent : Nat := 276287
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨37536⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68780⟩⟩] } }
def leftRaw : List Term := Proof.Events1079.exact276274RawTerms
def rightRaw : List Term := Proof.Events1073.exact274858RawTerms
def group : MergeGroup := .operator 276274 274858
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 276274) (leftOrdinal := 26)
    (rightResult := 274858) (rightOrdinal := 25) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨37536⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68780⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨37536⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68780⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨37536⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge276287

namespace LeftMerge276288
def owner : Owner := ⟨.program ⟨257⟩, ⟨70982⟩⟩
def mergeEvent : Nat := 276288
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7222⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩] } }
def leftRaw : List Term := Proof.Events1079.exact276274RawTerms
def rightRaw : List Term := Proof.Events1073.exact274858RawTerms
def group : MergeGroup := .operator 276274 274858
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 276274) (leftOrdinal := 12)
    (rightResult := 274858) (rightOrdinal := 12) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7222⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7222⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7222⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge276288

namespace LeftMerge276289
def owner : Owner := ⟨.program ⟨257⟩, ⟨70982⟩⟩
def mergeEvent : Nat := 276289
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨34856⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68780⟩⟩] } }
def leftRaw : List Term := Proof.Events1079.exact276274RawTerms
def rightRaw : List Term := Proof.Events1073.exact274858RawTerms
def group : MergeGroup := .operator 276274 274858
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 276274) (leftOrdinal := 25)
    (rightResult := 274858) (rightOrdinal := 24) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨34856⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68780⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨34856⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68780⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨34856⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge276289

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
