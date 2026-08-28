import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge44287
def owner : Owner := ⟨.program ⟨214⟩, ⟨7303⟩⟩
def mergeEvent : Nat := 44287
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6771⟩⟩] } }
def leftRaw : List Term := Proof.Events140.exact35915RawTerms
def rightRaw : List Term := Proof.Events058.exact15030RawTerms
def group : MergeGroup := .operator 35915 15030
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 35915) (leftOrdinal := 0)
    (rightResult := 15030) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6771⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6771⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge44287

namespace LeftMerge44304
def owner : Owner := ⟨.program ⟨214⟩, ⟨9414⟩⟩
def mergeEvent : Nat := 44304
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨9410⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7831⟩⟩] } }
def leftRaw : List Term := Proof.Events173.exact44298RawTerms
def rightRaw : List Term := Proof.Events058.exact15019RawTerms
def group : MergeGroup := .operator 44298 15019
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 44298) (leftOrdinal := 1)
    (rightResult := 15019) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨9410⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨7831⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨9410⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7831⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge44304

namespace LeftMerge44306
def owner : Owner := ⟨.program ⟨214⟩, ⟨9414⟩⟩
def mergeEvent : Nat := 44306
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨9410⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6772⟩⟩] } }
def rhsRaw : List Term := Proof.Events058.exact14989RawTerms
def group : MergeGroup := .relation 44305
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 44305) (rhsResult := 14989)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨9410⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7831⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨7831⟩⟩) ⟨6772⟩ 14989) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6772⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨9410⟩⟩], [⟨.program ⟨214⟩, ⟨6772⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge44306

namespace LeftMerge44307
def owner : Owner := ⟨.program ⟨214⟩, ⟨9414⟩⟩
def mergeEvent : Nat := 44307
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6771⟩⟩, ⟨.program ⟨214⟩, ⟨7831⟩⟩] } }
def leftRaw : List Term := Proof.Events173.exact44298RawTerms
def rightRaw : List Term := Proof.Events058.exact15019RawTerms
def group : MergeGroup := .operator 44298 15019
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 44298) (leftOrdinal := 0)
    (rightResult := 15019) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6771⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨7831⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6771⟩⟩, ⟨.program ⟨214⟩, ⟨7831⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge44307

namespace LeftMerge44312
def owner : Owner := ⟨.program ⟨214⟩, ⟨10503⟩⟩
def mergeEvent : Nat := 44312
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨9410⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6772⟩⟩] } }
def leftRaw : List Term := Proof.Events173.exact44308RawTerms
def rightRaw : List Term := Proof.Events172.exact44278RawTerms
def group : MergeGroup := .operator 44308 44278
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 44308) (leftOrdinal := 1)
    (rightResult := 44278) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨9410⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6772⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨9410⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6772⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨9410⟩⟩], [⟨.program ⟨214⟩, ⟨6772⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge44312

namespace LeftMerge44320
def owner : Owner := ⟨.program ⟨214⟩, ⟨24922⟩⟩
def mergeEvent : Nat := 44320
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨9410⟩⟩, ⟨.program ⟨214⟩, ⟨10496⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨24921⟩⟩] } }
def leftRaw : List Term := Proof.Events173.exact44314RawTerms
def rightRaw : List Term := Proof.Events172.exact44250RawTerms
def group : MergeGroup := .operator 44314 44250
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 44314) (leftOrdinal := 1)
    (rightResult := 44250) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨9410⟩⟩, ⟨.program ⟨214⟩, ⟨10496⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨24921⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨9410⟩⟩, ⟨.program ⟨214⟩, ⟨10496⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨24921⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge44320

namespace LeftMerge44322
def owner : Owner := ⟨.program ⟨214⟩, ⟨24922⟩⟩
def mergeEvent : Nat := 44322
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨9410⟩⟩, ⟨.program ⟨214⟩, ⟨10496⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨22958⟩⟩] } }
def rhsRaw : List Term := Proof.Events172.exact44247RawTerms
def group : MergeGroup := .relation 44321
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 44321) (rhsResult := 44247)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨9410⟩⟩, ⟨.program ⟨214⟩, ⟨10496⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨24921⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨24921⟩⟩) ⟨22958⟩ 44247) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨22958⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨9410⟩⟩, ⟨.program ⟨214⟩, ⟨10496⟩⟩], [⟨.program ⟨214⟩, ⟨22958⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge44322

namespace LeftMerge44323
def owner : Owner := ⟨.program ⟨214⟩, ⟨24922⟩⟩
def mergeEvent : Nat := 44323
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6771⟩⟩, ⟨.program ⟨214⟩, ⟨7831⟩⟩, ⟨.program ⟨214⟩, ⟨24921⟩⟩] } }
def leftRaw : List Term := Proof.Events173.exact44314RawTerms
def rightRaw : List Term := Proof.Events172.exact44250RawTerms
def group : MergeGroup := .operator 44314 44250
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 44314) (leftOrdinal := 0)
    (rightResult := 44250) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6771⟩⟩, ⟨.program ⟨214⟩, ⟨7831⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨24921⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6771⟩⟩, ⟨.program ⟨214⟩, ⟨7831⟩⟩, ⟨.program ⟨214⟩, ⟨24921⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge44323

namespace LeftMerge44337
def owner : Owner := ⟨.program ⟨214⟩, ⟨19035⟩⟩
def mergeEvent : Nat := 44337
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19032⟩⟩] } }
def leftRaw : List Term := Proof.Events141.exact36137RawTerms
def rightRaw : List Term := Proof.Events173.exact44331RawTerms
def group : MergeGroup := .operator 36137 44331
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 36137) (leftOrdinal := 0)
    (rightResult := 44331) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨19032⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19032⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge44337

namespace LeftMerge44416
def owner : Owner := ⟨.program ⟨214⟩, ⟨10497⟩⟩
def mergeEvent : Nat := 44416
def frameStart : Nat := 44386
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨9410⟩⟩, ⟨.program ⟨214⟩, ⟨10496⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events173.exact44412RawTerms
def rightRaw : List Term := Proof.Events173.exact44409RawTerms
def group : MergeGroup := .operator 44412 44409
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 44412) (leftOrdinal := 0)
    (rightResult := 44409) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨9410⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10496⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨9410⟩⟩, ⟨.program ⟨214⟩, ⟨10496⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge44416

namespace LeftMerge44446
def owner : Owner := ⟨.program ⟨214⟩, ⟨10586⟩⟩
def mergeEvent : Nat := 44446
def frameStart : Nat := 44386
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨9410⟩⟩, ⟨.program ⟨214⟩, ⟨10496⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events173.exact44442RawTerms
def rightRaw : List Term := Proof.Events173.exact44440RawTerms
def group : MergeGroup := .operator 44442 44440
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 44442) (leftOrdinal := 0)
    (rightResult := 44440) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨9410⟩⟩, ⟨.program ⟨214⟩, ⟨10496⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨9410⟩⟩, ⟨.program ⟨214⟩, ⟨10496⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge44446

namespace LeftMerge44469
def owner : Owner := ⟨.program ⟨214⟩, ⟨7833⟩⟩
def mergeEvent : Nat := 44469
def frameStart : Nat := 44386
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6771⟩⟩, ⟨.program ⟨214⟩, ⟨7831⟩⟩] } }
def leftRaw : List Term := Proof.Events173.exact44465RawTerms
def rightRaw : List Term := Proof.Events173.exact44462RawTerms
def group : MergeGroup := .operator 44465 44462
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 44465) (leftOrdinal := 0)
    (rightResult := 44462) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6771⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨7831⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6771⟩⟩, ⟨.program ⟨214⟩, ⟨7831⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge44469

namespace LeftMerge44478
def owner : Owner := ⟨.program ⟨214⟩, ⟨24924⟩⟩
def mergeEvent : Nat := 44478
def frameStart : Nat := 44386
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6771⟩⟩, ⟨.program ⟨214⟩, ⟨7831⟩⟩, ⟨.program ⟨214⟩, ⟨24921⟩⟩] } }
def leftRaw : List Term := Proof.Events173.exact44474RawTerms
def rightRaw : List Term := Proof.Events173.exact44431RawTerms
def group : MergeGroup := .operator 44474 44431
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 44474) (leftOrdinal := 0)
    (rightResult := 44431) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6771⟩⟩, ⟨.program ⟨214⟩, ⟨7831⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨24921⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6771⟩⟩, ⟨.program ⟨214⟩, ⟨7831⟩⟩, ⟨.program ⟨214⟩, ⟨24921⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge44478

namespace LeftMerge44479
def owner : Owner := ⟨.program ⟨214⟩, ⟨24924⟩⟩
def mergeEvent : Nat := 44479
def frameStart : Nat := 44386
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨9410⟩⟩, ⟨.program ⟨214⟩, ⟨10496⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨24921⟩⟩] } }
def leftRaw : List Term := Proof.Events173.exact44474RawTerms
def rightRaw : List Term := Proof.Events173.exact44431RawTerms
def group : MergeGroup := .operator 44474 44431
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 44474) (leftOrdinal := 1)
    (rightResult := 44431) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨9410⟩⟩, ⟨.program ⟨214⟩, ⟨10496⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨24921⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨9410⟩⟩, ⟨.program ⟨214⟩, ⟨10496⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨24921⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge44479

namespace LeftMerge44481
def owner : Owner := ⟨.program ⟨214⟩, ⟨24924⟩⟩
def mergeEvent : Nat := 44481
def frameStart : Nat := 44386
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨9410⟩⟩, ⟨.program ⟨214⟩, ⟨10496⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨22958⟩⟩] } }
def rhsRaw : List Term := Proof.Events173.exact44428RawTerms
def group : MergeGroup := .relation 44480
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 44480) (rhsResult := 44428)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨9410⟩⟩, ⟨.program ⟨214⟩, ⟨10496⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨24921⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨24921⟩⟩) ⟨22958⟩ 44428) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨22958⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨9410⟩⟩, ⟨.program ⟨214⟩, ⟨10496⟩⟩], [⟨.program ⟨214⟩, ⟨22958⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge44481

namespace LeftMerge44489
def owner : Owner := ⟨.program ⟨214⟩, ⟨14802⟩⟩
def mergeEvent : Nat := 44489
def frameStart : Nat := 44386
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨14800⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events173.exact44442RawTerms
def rightRaw : List Term := Proof.Events173.exact44485RawTerms
def group : MergeGroup := .operator 44442 44485
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 44442) (leftOrdinal := 0)
    (rightResult := 44485) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨14800⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨14800⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge44489

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
