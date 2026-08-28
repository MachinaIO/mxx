import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge56535
def owner : Owner := ⟨.program ⟨214⟩, ⟨25918⟩⟩
def mergeEvent : Nat := 56535
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨11305⟩⟩, ⟨.program ⟨214⟩, ⟨13782⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25917⟩⟩] } }
def leftRaw : List Term := Proof.Events220.exact56529RawTerms
def rightRaw : List Term := Proof.Events220.exact56465RawTerms
def group : MergeGroup := .operator 56529 56465
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 56529) (leftOrdinal := 1)
    (rightResult := 56465) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨11305⟩⟩, ⟨.program ⟨214⟩, ⟨13782⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25917⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨11305⟩⟩, ⟨.program ⟨214⟩, ⟨13782⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25917⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge56535

namespace LeftMerge56537
def owner : Owner := ⟨.program ⟨214⟩, ⟨25918⟩⟩
def mergeEvent : Nat := 56537
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨11305⟩⟩, ⟨.program ⟨214⟩, ⟨13782⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23502⟩⟩] } }
def rhsRaw : List Term := Proof.Events220.exact56462RawTerms
def group : MergeGroup := .relation 56536
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 56536) (rhsResult := 56462)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨11305⟩⟩, ⟨.program ⟨214⟩, ⟨13782⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25917⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨25917⟩⟩) ⟨23502⟩ 56462) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23502⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨11305⟩⟩, ⟨.program ⟨214⟩, ⟨13782⟩⟩], [⟨.program ⟨214⟩, ⟨23502⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge56537

namespace LeftMerge56538
def owner : Owner := ⟨.program ⟨214⟩, ⟨25918⟩⟩
def mergeEvent : Nat := 56538
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6794⟩⟩, ⟨.program ⟨214⟩, ⟨7846⟩⟩, ⟨.program ⟨214⟩, ⟨25917⟩⟩] } }
def leftRaw : List Term := Proof.Events220.exact56529RawTerms
def rightRaw : List Term := Proof.Events220.exact56465RawTerms
def group : MergeGroup := .operator 56529 56465
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 56529) (leftOrdinal := 0)
    (rightResult := 56465) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6794⟩⟩, ⟨.program ⟨214⟩, ⟨7846⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25917⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6794⟩⟩, ⟨.program ⟨214⟩, ⟨7846⟩⟩, ⟨.program ⟨214⟩, ⟨25917⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge56538

namespace LeftMerge56552
def owner : Owner := ⟨.program ⟨214⟩, ⟨19391⟩⟩
def mergeEvent : Nat := 56552
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19388⟩⟩] } }
def leftRaw : List Term := Proof.Events198.exact50762RawTerms
def rightRaw : List Term := Proof.Events220.exact56546RawTerms
def group : MergeGroup := .operator 50762 56546
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 50762) (leftOrdinal := 0)
    (rightResult := 56546) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨19388⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19388⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge56552

namespace LeftMerge56631
def owner : Owner := ⟨.program ⟨214⟩, ⟨13783⟩⟩
def mergeEvent : Nat := 56631
def frameStart : Nat := 56601
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨11305⟩⟩, ⟨.program ⟨214⟩, ⟨13782⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events221.exact56627RawTerms
def rightRaw : List Term := Proof.Events221.exact56624RawTerms
def group : MergeGroup := .operator 56627 56624
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 56627) (leftOrdinal := 0)
    (rightResult := 56624) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨13782⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11305⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨11305⟩⟩, ⟨.program ⟨214⟩, ⟨13782⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge56631

namespace LeftMerge56661
def owner : Owner := ⟨.program ⟨214⟩, ⟨13886⟩⟩
def mergeEvent : Nat := 56661
def frameStart : Nat := 56601
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨11305⟩⟩, ⟨.program ⟨214⟩, ⟨13782⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events221.exact56657RawTerms
def rightRaw : List Term := Proof.Events221.exact56655RawTerms
def group : MergeGroup := .operator 56657 56655
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 56657) (leftOrdinal := 0)
    (rightResult := 56655) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11305⟩⟩, ⟨.program ⟨214⟩, ⟨13782⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨11305⟩⟩, ⟨.program ⟨214⟩, ⟨13782⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge56661

namespace LeftMerge56684
def owner : Owner := ⟨.program ⟨214⟩, ⟨7848⟩⟩
def mergeEvent : Nat := 56684
def frameStart : Nat := 56601
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6794⟩⟩, ⟨.program ⟨214⟩, ⟨7846⟩⟩] } }
def leftRaw : List Term := Proof.Events221.exact56680RawTerms
def rightRaw : List Term := Proof.Events221.exact56677RawTerms
def group : MergeGroup := .operator 56680 56677
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 56680) (leftOrdinal := 0)
    (rightResult := 56677) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6794⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨7846⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6794⟩⟩, ⟨.program ⟨214⟩, ⟨7846⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge56684

namespace LeftMerge56693
def owner : Owner := ⟨.program ⟨214⟩, ⟨25920⟩⟩
def mergeEvent : Nat := 56693
def frameStart : Nat := 56601
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6794⟩⟩, ⟨.program ⟨214⟩, ⟨7846⟩⟩, ⟨.program ⟨214⟩, ⟨25917⟩⟩] } }
def leftRaw : List Term := Proof.Events221.exact56689RawTerms
def rightRaw : List Term := Proof.Events221.exact56646RawTerms
def group : MergeGroup := .operator 56689 56646
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 56689) (leftOrdinal := 0)
    (rightResult := 56646) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6794⟩⟩, ⟨.program ⟨214⟩, ⟨7846⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25917⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6794⟩⟩, ⟨.program ⟨214⟩, ⟨7846⟩⟩, ⟨.program ⟨214⟩, ⟨25917⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge56693

namespace LeftMerge56694
def owner : Owner := ⟨.program ⟨214⟩, ⟨25920⟩⟩
def mergeEvent : Nat := 56694
def frameStart : Nat := 56601
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨11305⟩⟩, ⟨.program ⟨214⟩, ⟨13782⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25917⟩⟩] } }
def leftRaw : List Term := Proof.Events221.exact56689RawTerms
def rightRaw : List Term := Proof.Events221.exact56646RawTerms
def group : MergeGroup := .operator 56689 56646
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 56689) (leftOrdinal := 1)
    (rightResult := 56646) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11305⟩⟩, ⟨.program ⟨214⟩, ⟨13782⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25917⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨11305⟩⟩, ⟨.program ⟨214⟩, ⟨13782⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25917⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge56694

namespace LeftMerge56696
def owner : Owner := ⟨.program ⟨214⟩, ⟨25920⟩⟩
def mergeEvent : Nat := 56696
def frameStart : Nat := 56601
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨11305⟩⟩, ⟨.program ⟨214⟩, ⟨13782⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23502⟩⟩] } }
def rhsRaw : List Term := Proof.Events221.exact56643RawTerms
def group : MergeGroup := .relation 56695
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 56695) (rhsResult := 56643)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨11305⟩⟩, ⟨.program ⟨214⟩, ⟨13782⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25917⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨25917⟩⟩) ⟨23502⟩ 56643) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23502⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨11305⟩⟩, ⟨.program ⟨214⟩, ⟨13782⟩⟩], [⟨.program ⟨214⟩, ⟨23502⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge56696

namespace LeftMerge56704
def owner : Owner := ⟨.program ⟨214⟩, ⟨15708⟩⟩
def mergeEvent : Nat := 56704
def frameStart : Nat := 56601
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15706⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events221.exact56657RawTerms
def rightRaw : List Term := Proof.Events221.exact56700RawTerms
def group : MergeGroup := .operator 56657 56700
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 56657) (leftOrdinal := 0)
    (rightResult := 56700) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15706⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨15706⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge56704

namespace LeftMerge56721
def owner : Owner := ⟨.program ⟨214⟩, ⟨19391⟩⟩
def mergeEvent : Nat := 56721
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6695⟩⟩] } }
def rhsRaw : List Term := Proof.Events221.exact56718RawTerms
def group : MergeGroup := .relation 56720
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 56720) (rhsResult := 56718)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19388⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 56719 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19388⟩⟩]⟩) (none) 56718) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6695⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6695⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge56721

namespace LeftMerge56722
def owner : Owner := ⟨.program ⟨214⟩, ⟨19391⟩⟩
def mergeEvent : Nat := 56722
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6794⟩⟩, ⟨.program ⟨214⟩, ⟨7846⟩⟩, ⟨.program ⟨214⟩, ⟨25917⟩⟩] } }
def rhsRaw : List Term := Proof.Events221.exact56718RawTerms
def group : MergeGroup := .relation 56720
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 56720) (rhsResult := 56718)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19388⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 56719 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19388⟩⟩]⟩) (none) 56718) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6794⟩⟩, ⟨.program ⟨214⟩, ⟨7846⟩⟩, ⟨.program ⟨214⟩, ⟨25917⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6794⟩⟩, ⟨.program ⟨214⟩, ⟨7846⟩⟩, ⟨.program ⟨214⟩, ⟨25917⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge56722

namespace LeftMerge56723
def owner : Owner := ⟨.program ⟨214⟩, ⟨19391⟩⟩
def mergeEvent : Nat := 56723
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨11305⟩⟩, ⟨.program ⟨214⟩, ⟨13782⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23502⟩⟩] } }
def rhsRaw : List Term := Proof.Events221.exact56718RawTerms
def group : MergeGroup := .relation 56720
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 56720) (rhsResult := 56718)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19388⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 56719 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19388⟩⟩]⟩) (none) 56718) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11305⟩⟩, ⟨.program ⟨214⟩, ⟨13782⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23502⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨11305⟩⟩, ⟨.program ⟨214⟩, ⟨13782⟩⟩], [⟨.program ⟨214⟩, ⟨23502⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge56723

namespace LeftMerge56724
def owner : Owner := ⟨.program ⟨214⟩, ⟨19391⟩⟩
def mergeEvent : Nat := 56724
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨15706⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def rhsRaw : List Term := Proof.Events221.exact56718RawTerms
def group : MergeGroup := .relation 56720
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 56720) (rhsResult := 56718)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19388⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 56719 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19388⟩⟩]⟩) (none) 56718) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15706⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨15706⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge56724

namespace LeftMerge56729
def owner : Owner := ⟨.program ⟨214⟩, ⟨25919⟩⟩
def mergeEvent : Nat := 56729
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨11305⟩⟩, ⟨.program ⟨214⟩, ⟨13782⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23502⟩⟩] } }
def leftRaw : List Term := Proof.Events221.exact56725RawTerms
def rightRaw : List Term := Proof.Events220.exact56539RawTerms
def group : MergeGroup := .operator 56725 56539
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 56725) (leftOrdinal := 2)
    (rightResult := 56539) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨11305⟩⟩, ⟨.program ⟨214⟩, ⟨13782⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23502⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨11305⟩⟩, ⟨.program ⟨214⟩, ⟨13782⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23502⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨11305⟩⟩, ⟨.program ⟨214⟩, ⟨13782⟩⟩], [⟨.program ⟨214⟩, ⟨23502⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge56729

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
