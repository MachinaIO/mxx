import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge99560
def owner : Owner := ⟨.program ⟨214⟩, ⟨21248⟩⟩
def mergeEvent : Nat := 99560
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15811⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24090⟩⟩] } }
def rhsRaw : List Term := Proof.Events388.exact99555RawTerms
def group : MergeGroup := .relation 99557
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 99557) (rhsResult := 99555)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21245⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 99556 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21245⟩⟩]⟩) (none) 99555) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15811⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24090⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15811⟩⟩], [⟨.program ⟨214⟩, ⟨24090⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge99560

namespace LeftMerge99561
def owner : Owner := ⟨.program ⟨214⟩, ⟨21248⟩⟩
def mergeEvent : Nat := 99561
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15860⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def rhsRaw : List Term := Proof.Events388.exact99555RawTerms
def group : MergeGroup := .relation 99557
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 99557) (rhsResult := 99555)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21245⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 99556 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21245⟩⟩]⟩) (none) 99555) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15860⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15860⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge99561

namespace LeftMerge99566
def owner : Owner := ⟨.program ⟨214⟩, ⟨27617⟩⟩
def mergeEvent : Nat := 99566
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6696⟩⟩, ⟨.program ⟨214⟩, ⟨27614⟩⟩] } }
def leftRaw : List Term := Proof.Events388.exact99562RawTerms
def rightRaw : List Term := Proof.Events388.exact99408RawTerms
def group : MergeGroup := .operator 99562 99408
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 99562) (leftOrdinal := 0)
    (rightResult := 99408) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6696⟩⟩, ⟨.program ⟨214⟩, ⟨27614⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6696⟩⟩, ⟨.program ⟨214⟩, ⟨27614⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6696⟩⟩, ⟨.program ⟨214⟩, ⟨27614⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge99566

namespace LeftMerge99567
def owner : Owner := ⟨.program ⟨214⟩, ⟨27617⟩⟩
def mergeEvent : Nat := 99567
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15811⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24090⟩⟩] } }
def leftRaw : List Term := Proof.Events388.exact99562RawTerms
def rightRaw : List Term := Proof.Events388.exact99408RawTerms
def group : MergeGroup := .operator 99562 99408
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 99562) (leftOrdinal := 2)
    (rightResult := 99408) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15811⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24090⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15811⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24090⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15811⟩⟩], [⟨.program ⟨214⟩, ⟨24090⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge99567

namespace LeftMerge99593
def owner : Owner := ⟨.program ⟨214⟩, ⟨11290⟩⟩
def mergeEvent : Nat := 99593
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨11289⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events018.exact4842RawTerms
def rightRaw : List Term := Proof.Events000.exact32RawTerms
def group : MergeGroup := .operator 4842 32
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 4842) (leftOrdinal := 0)
    (rightResult := 32) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11289⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨11289⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge99593

namespace LeftMerge99598
def owner : Owner := ⟨.program ⟨214⟩, ⟨7114⟩⟩
def mergeEvent : Nat := 99598
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6777⟩⟩] } }
def leftRaw : List Term := Proof.Events000.exact27RawTerms
def rightRaw : List Term := Proof.Events048.exact12484RawTerms
def group : MergeGroup := .operator 27 12484
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 27) (leftOrdinal := 0)
    (rightResult := 12484) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6777⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6777⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge99598

namespace LeftMerge99615
def owner : Owner := ⟨.program ⟨214⟩, ⟨13749⟩⟩
def mergeEvent : Nat := 99615
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨11289⟩⟩, ⟨.program ⟨214⟩, ⟨13746⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events389.exact99609RawTerms
def rightRaw : List Term := Proof.Events018.exact4845RawTerms
def group : MergeGroup := .operator 99609 4845
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 99609) (leftOrdinal := 1)
    (rightResult := 4845) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨11289⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨13746⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨11289⟩⟩, ⟨.program ⟨214⟩, ⟨13746⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge99615

namespace LeftMerge99616
def owner : Owner := ⟨.program ⟨214⟩, ⟨13749⟩⟩
def mergeEvent : Nat := 99616
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨13746⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6777⟩⟩] } }
def leftRaw : List Term := Proof.Events389.exact99609RawTerms
def rightRaw : List Term := Proof.Events018.exact4845RawTerms
def group : MergeGroup := .operator 99609 4845
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 99609) (leftOrdinal := 0)
    (rightResult := 4845) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6777⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨13746⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨13746⟩⟩], [⟨.program ⟨214⟩, ⟨6777⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge99616

namespace LeftMerge99621
def owner : Owner := ⟨.program ⟨214⟩, ⟨13750⟩⟩
def mergeEvent : Nat := 99621
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨13746⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events018.exact4845RawTerms
def rightRaw : List Term := Proof.Events000.exact32RawTerms
def group : MergeGroup := .operator 4845 32
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 4845) (leftOrdinal := 0)
    (rightResult := 32) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨13746⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨13746⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge99621

namespace LeftMerge99626
def owner : Owner := ⟨.program ⟨214⟩, ⟨7131⟩⟩
def mergeEvent : Nat := 99626
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6794⟩⟩] } }
def leftRaw : List Term := Proof.Events000.exact27RawTerms
def rightRaw : List Term := Proof.Events048.exact12525RawTerms
def group : MergeGroup := .operator 27 12525
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 27) (leftOrdinal := 0)
    (rightResult := 12525) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6794⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6794⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge99626

namespace LeftMerge99643
def owner : Owner := ⟨.program ⟨214⟩, ⟨13753⟩⟩
def mergeEvent : Nat := 99643
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨13746⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7846⟩⟩] } }
def leftRaw : List Term := Proof.Events389.exact99637RawTerms
def rightRaw : List Term := Proof.Events048.exact12514RawTerms
def group : MergeGroup := .operator 99637 12514
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 99637) (leftOrdinal := 1)
    (rightResult := 12514) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨13746⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨7846⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨13746⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7846⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge99643

namespace LeftMerge99645
def owner : Owner := ⟨.program ⟨214⟩, ⟨13753⟩⟩
def mergeEvent : Nat := 99645
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨13746⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6777⟩⟩] } }
def rhsRaw : List Term := Proof.Events048.exact12484RawTerms
def group : MergeGroup := .relation 99644
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 99644) (rhsResult := 12484)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨13746⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7846⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨7846⟩⟩) ⟨6777⟩ 12484) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6777⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨13746⟩⟩], [⟨.program ⟨214⟩, ⟨6777⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge99645

namespace LeftMerge99646
def owner : Owner := ⟨.program ⟨214⟩, ⟨13753⟩⟩
def mergeEvent : Nat := 99646
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6794⟩⟩, ⟨.program ⟨214⟩, ⟨7846⟩⟩] } }
def leftRaw : List Term := Proof.Events389.exact99637RawTerms
def rightRaw : List Term := Proof.Events048.exact12514RawTerms
def group : MergeGroup := .operator 99637 12514
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 99637) (leftOrdinal := 0)
    (rightResult := 12514) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6794⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨7846⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6794⟩⟩, ⟨.program ⟨214⟩, ⟨7846⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge99646

namespace LeftMerge99651
def owner : Owner := ⟨.program ⟨214⟩, ⟨13754⟩⟩
def mergeEvent : Nat := 99651
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨13746⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6777⟩⟩] } }
def leftRaw : List Term := Proof.Events389.exact99647RawTerms
def rightRaw : List Term := Proof.Events389.exact99617RawTerms
def group : MergeGroup := .operator 99647 99617
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 99647) (leftOrdinal := 1)
    (rightResult := 99617) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨13746⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6777⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨13746⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6777⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨13746⟩⟩], [⟨.program ⟨214⟩, ⟨6777⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge99651

namespace LeftMerge99659
def owner : Owner := ⟨.program ⟨214⟩, ⟨25900⟩⟩
def mergeEvent : Nat := 99659
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨11289⟩⟩, ⟨.program ⟨214⟩, ⟨13746⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25899⟩⟩] } }
def leftRaw : List Term := Proof.Events389.exact99653RawTerms
def rightRaw : List Term := Proof.Events389.exact99589RawTerms
def group : MergeGroup := .operator 99653 99589
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 99653) (leftOrdinal := 1)
    (rightResult := 99589) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨11289⟩⟩, ⟨.program ⟨214⟩, ⟨13746⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25899⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨11289⟩⟩, ⟨.program ⟨214⟩, ⟨13746⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25899⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge99659

namespace LeftMerge99661
def owner : Owner := ⟨.program ⟨214⟩, ⟨25900⟩⟩
def mergeEvent : Nat := 99661
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨11289⟩⟩, ⟨.program ⟨214⟩, ⟨13746⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23494⟩⟩] } }
def rhsRaw : List Term := Proof.Events389.exact99586RawTerms
def group : MergeGroup := .relation 99660
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 99660) (rhsResult := 99586)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨11289⟩⟩, ⟨.program ⟨214⟩, ⟨13746⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25899⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨25899⟩⟩) ⟨23494⟩ 99586) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23494⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨11289⟩⟩, ⟨.program ⟨214⟩, ⟨13746⟩⟩], [⟨.program ⟨214⟩, ⟨23494⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge99661

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
