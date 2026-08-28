import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge100514
def owner : Owner := ⟨.program ⟨214⟩, ⟨12143⟩⟩
def mergeEvent : Nat := 100514
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6792⟩⟩, ⟨.program ⟨214⟩, ⟨7840⟩⟩] } }
def leftRaw : List Term := Proof.Events392.exact100505RawTerms
def rightRaw : List Term := Proof.Events052.exact13516RawTerms
def group : MergeGroup := .operator 100505 13516
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 100505) (leftOrdinal := 0)
    (rightResult := 13516) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6792⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨7840⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6792⟩⟩, ⟨.program ⟨214⟩, ⟨7840⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge100514

namespace LeftMerge100519
def owner : Owner := ⟨.program ⟨214⟩, ⟨12144⟩⟩
def mergeEvent : Nat := 100519
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨12136⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6775⟩⟩] } }
def leftRaw : List Term := Proof.Events392.exact100515RawTerms
def rightRaw : List Term := Proof.Events392.exact100485RawTerms
def group : MergeGroup := .operator 100515 100485
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 100515) (leftOrdinal := 1)
    (rightResult := 100485) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨12136⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6775⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨12136⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6775⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨12136⟩⟩], [⟨.program ⟨214⟩, ⟨6775⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge100519

namespace LeftMerge100527
def owner : Owner := ⟨.program ⟨214⟩, ⟨25284⟩⟩
def mergeEvent : Nat := 100527
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨11121⟩⟩, ⟨.program ⟨214⟩, ⟨12136⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25283⟩⟩] } }
def leftRaw : List Term := Proof.Events392.exact100521RawTerms
def rightRaw : List Term := Proof.Events392.exact100457RawTerms
def group : MergeGroup := .operator 100521 100457
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 100521) (leftOrdinal := 1)
    (rightResult := 100457) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨11121⟩⟩, ⟨.program ⟨214⟩, ⟨12136⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25283⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨11121⟩⟩, ⟨.program ⟨214⟩, ⟨12136⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25283⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge100527

namespace LeftMerge100529
def owner : Owner := ⟨.program ⟨214⟩, ⟨25284⟩⟩
def mergeEvent : Nat := 100529
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨11121⟩⟩, ⟨.program ⟨214⟩, ⟨12136⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23158⟩⟩] } }
def rhsRaw : List Term := Proof.Events392.exact100454RawTerms
def group : MergeGroup := .relation 100528
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 100528) (rhsResult := 100454)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨11121⟩⟩, ⟨.program ⟨214⟩, ⟨12136⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25283⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨25283⟩⟩) ⟨23158⟩ 100454) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23158⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨11121⟩⟩, ⟨.program ⟨214⟩, ⟨12136⟩⟩], [⟨.program ⟨214⟩, ⟨23158⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge100529

namespace LeftMerge100530
def owner : Owner := ⟨.program ⟨214⟩, ⟨25284⟩⟩
def mergeEvent : Nat := 100530
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6792⟩⟩, ⟨.program ⟨214⟩, ⟨7840⟩⟩, ⟨.program ⟨214⟩, ⟨25283⟩⟩] } }
def leftRaw : List Term := Proof.Events392.exact100521RawTerms
def rightRaw : List Term := Proof.Events392.exact100457RawTerms
def group : MergeGroup := .operator 100521 100457
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 100521) (leftOrdinal := 0)
    (rightResult := 100457) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6792⟩⟩, ⟨.program ⟨214⟩, ⟨7840⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25283⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6792⟩⟩, ⟨.program ⟨214⟩, ⟨7840⟩⟩, ⟨.program ⟨214⟩, ⟨25283⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge100530

namespace LeftMerge100544
def owner : Owner := ⟨.program ⟨214⟩, ⟨19232⟩⟩
def mergeEvent : Nat := 100544
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19229⟩⟩] } }
def leftRaw : List Term := Proof.Events368.exact94462RawTerms
def rightRaw : List Term := Proof.Events392.exact100538RawTerms
def group : MergeGroup := .operator 94462 100538
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 94462) (leftOrdinal := 0)
    (rightResult := 100538) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨19229⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19229⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge100544

namespace LeftMerge100599
def owner : Owner := ⟨.program ⟨214⟩, ⟨12137⟩⟩
def mergeEvent : Nat := 100599
def frameStart : Nat := 100581
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨11121⟩⟩, ⟨.program ⟨214⟩, ⟨12136⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events392.exact100595RawTerms
def rightRaw : List Term := Proof.Events392.exact100592RawTerms
def group : MergeGroup := .operator 100595 100592
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 100595) (leftOrdinal := 0)
    (rightResult := 100592) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨12136⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11121⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨11121⟩⟩, ⟨.program ⟨214⟩, ⟨12136⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge100599

namespace LeftMerge100629
def owner : Owner := ⟨.program ⟨214⟩, ⟨12264⟩⟩
def mergeEvent : Nat := 100629
def frameStart : Nat := 100581
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨11121⟩⟩, ⟨.program ⟨214⟩, ⟨12136⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events393.exact100625RawTerms
def rightRaw : List Term := Proof.Events393.exact100623RawTerms
def group : MergeGroup := .operator 100625 100623
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 100625) (leftOrdinal := 0)
    (rightResult := 100623) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11121⟩⟩, ⟨.program ⟨214⟩, ⟨12136⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨11121⟩⟩, ⟨.program ⟨214⟩, ⟨12136⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge100629

namespace LeftMerge100652
def owner : Owner := ⟨.program ⟨214⟩, ⟨7842⟩⟩
def mergeEvent : Nat := 100652
def frameStart : Nat := 100581
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6792⟩⟩, ⟨.program ⟨214⟩, ⟨7840⟩⟩] } }
def leftRaw : List Term := Proof.Events393.exact100648RawTerms
def rightRaw : List Term := Proof.Events393.exact100645RawTerms
def group : MergeGroup := .operator 100648 100645
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 100648) (leftOrdinal := 0)
    (rightResult := 100645) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6792⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨7840⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6792⟩⟩, ⟨.program ⟨214⟩, ⟨7840⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge100652

namespace LeftMerge100661
def owner : Owner := ⟨.program ⟨214⟩, ⟨25286⟩⟩
def mergeEvent : Nat := 100661
def frameStart : Nat := 100581
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6792⟩⟩, ⟨.program ⟨214⟩, ⟨7840⟩⟩, ⟨.program ⟨214⟩, ⟨25283⟩⟩] } }
def leftRaw : List Term := Proof.Events393.exact100657RawTerms
def rightRaw : List Term := Proof.Events393.exact100614RawTerms
def group : MergeGroup := .operator 100657 100614
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 100657) (leftOrdinal := 0)
    (rightResult := 100614) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6792⟩⟩, ⟨.program ⟨214⟩, ⟨7840⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25283⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6792⟩⟩, ⟨.program ⟨214⟩, ⟨7840⟩⟩, ⟨.program ⟨214⟩, ⟨25283⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge100661

namespace LeftMerge100662
def owner : Owner := ⟨.program ⟨214⟩, ⟨25286⟩⟩
def mergeEvent : Nat := 100662
def frameStart : Nat := 100581
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨11121⟩⟩, ⟨.program ⟨214⟩, ⟨12136⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25283⟩⟩] } }
def leftRaw : List Term := Proof.Events393.exact100657RawTerms
def rightRaw : List Term := Proof.Events393.exact100614RawTerms
def group : MergeGroup := .operator 100657 100614
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 100657) (leftOrdinal := 1)
    (rightResult := 100614) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11121⟩⟩, ⟨.program ⟨214⟩, ⟨12136⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25283⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨11121⟩⟩, ⟨.program ⟨214⟩, ⟨12136⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25283⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge100662

namespace LeftMerge100664
def owner : Owner := ⟨.program ⟨214⟩, ⟨25286⟩⟩
def mergeEvent : Nat := 100664
def frameStart : Nat := 100581
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨11121⟩⟩, ⟨.program ⟨214⟩, ⟨12136⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23158⟩⟩] } }
def rhsRaw : List Term := Proof.Events393.exact100611RawTerms
def group : MergeGroup := .relation 100663
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 100663) (rhsResult := 100611)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨11121⟩⟩, ⟨.program ⟨214⟩, ⟨12136⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25283⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨25283⟩⟩) ⟨23158⟩ 100611) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23158⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨11121⟩⟩, ⟨.program ⟨214⟩, ⟨12136⟩⟩], [⟨.program ⟨214⟩, ⟨23158⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge100664

namespace LeftMerge100672
def owner : Owner := ⟨.program ⟨214⟩, ⟨15414⟩⟩
def mergeEvent : Nat := 100672
def frameStart : Nat := 100581
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15412⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events393.exact100625RawTerms
def rightRaw : List Term := Proof.Events393.exact100668RawTerms
def group : MergeGroup := .operator 100625 100668
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 100625) (leftOrdinal := 0)
    (rightResult := 100668) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15412⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨15412⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge100672

namespace LeftMerge100689
def owner : Owner := ⟨.program ⟨214⟩, ⟨19232⟩⟩
def mergeEvent : Nat := 100689
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6693⟩⟩] } }
def rhsRaw : List Term := Proof.Events393.exact100686RawTerms
def group : MergeGroup := .relation 100688
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 100688) (rhsResult := 100686)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19229⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 100687 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19229⟩⟩]⟩) (none) 100686) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6693⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6693⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge100689

namespace LeftMerge100690
def owner : Owner := ⟨.program ⟨214⟩, ⟨19232⟩⟩
def mergeEvent : Nat := 100690
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6792⟩⟩, ⟨.program ⟨214⟩, ⟨7840⟩⟩, ⟨.program ⟨214⟩, ⟨25283⟩⟩] } }
def rhsRaw : List Term := Proof.Events393.exact100686RawTerms
def group : MergeGroup := .relation 100688
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 100688) (rhsResult := 100686)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19229⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 100687 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19229⟩⟩]⟩) (none) 100686) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6792⟩⟩, ⟨.program ⟨214⟩, ⟨7840⟩⟩, ⟨.program ⟨214⟩, ⟨25283⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6792⟩⟩, ⟨.program ⟨214⟩, ⟨7840⟩⟩, ⟨.program ⟨214⟩, ⟨25283⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge100690

namespace LeftMerge100691
def owner : Owner := ⟨.program ⟨214⟩, ⟨19232⟩⟩
def mergeEvent : Nat := 100691
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨11121⟩⟩, ⟨.program ⟨214⟩, ⟨12136⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23158⟩⟩] } }
def rhsRaw : List Term := Proof.Events393.exact100686RawTerms
def group : MergeGroup := .relation 100688
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 100688) (rhsResult := 100686)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19229⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 100687 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19229⟩⟩]⟩) (none) 100686) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11121⟩⟩, ⟨.program ⟨214⟩, ⟨12136⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23158⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨11121⟩⟩, ⟨.program ⟨214⟩, ⟨12136⟩⟩], [⟨.program ⟨214⟩, ⟨23158⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge100691

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
