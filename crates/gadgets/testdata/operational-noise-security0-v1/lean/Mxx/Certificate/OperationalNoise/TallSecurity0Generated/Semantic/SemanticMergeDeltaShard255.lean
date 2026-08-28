import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge42831
def owner : Owner := ⟨.program ⟨214⟩, ⟨12184⟩⟩
def mergeEvent : Nat := 42831
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨12181⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6775⟩⟩] } }
def leftRaw : List Term := Proof.Events167.exact42824RawTerms
def rightRaw : List Term := Proof.Events007.exact1915RawTerms
def group : MergeGroup := .operator 42824 1915
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 42824) (leftOrdinal := 0)
    (rightResult := 1915) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6775⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨12181⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨12181⟩⟩], [⟨.program ⟨214⟩, ⟨6775⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge42831

namespace LeftMerge42836
def owner : Owner := ⟨.program ⟨214⟩, ⟨12185⟩⟩
def mergeEvent : Nat := 42836
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨12181⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events007.exact1915RawTerms
def rightRaw : List Term := Proof.Events140.exact36045RawTerms
def group : MergeGroup := .operator 1915 36045
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 1915) (leftOrdinal := 0)
    (rightResult := 36045) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨12181⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨12181⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge42836

namespace LeftMerge42841
def owner : Owner := ⟨.program ⟨214⟩, ⟨7324⟩⟩
def mergeEvent : Nat := 42841
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6792⟩⟩] } }
def leftRaw : List Term := Proof.Events140.exact35915RawTerms
def rightRaw : List Term := Proof.Events052.exact13527RawTerms
def group : MergeGroup := .operator 35915 13527
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 35915) (leftOrdinal := 0)
    (rightResult := 13527) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6792⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6792⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge42841

namespace LeftMerge42858
def owner : Owner := ⟨.program ⟨214⟩, ⟨12188⟩⟩
def mergeEvent : Nat := 42858
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨12181⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7840⟩⟩] } }
def leftRaw : List Term := Proof.Events167.exact42852RawTerms
def rightRaw : List Term := Proof.Events052.exact13516RawTerms
def group : MergeGroup := .operator 42852 13516
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 42852) (leftOrdinal := 1)
    (rightResult := 13516) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨12181⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨7840⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨12181⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7840⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge42858

namespace LeftMerge42860
def owner : Owner := ⟨.program ⟨214⟩, ⟨12188⟩⟩
def mergeEvent : Nat := 42860
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨12181⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6775⟩⟩] } }
def rhsRaw : List Term := Proof.Events052.exact13486RawTerms
def group : MergeGroup := .relation 42859
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 42859) (rhsResult := 13486)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨12181⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7840⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨7840⟩⟩) ⟨6775⟩ 13486) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6775⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨12181⟩⟩], [⟨.program ⟨214⟩, ⟨6775⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge42860

namespace LeftMerge42861
def owner : Owner := ⟨.program ⟨214⟩, ⟨12188⟩⟩
def mergeEvent : Nat := 42861
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6792⟩⟩, ⟨.program ⟨214⟩, ⟨7840⟩⟩] } }
def leftRaw : List Term := Proof.Events167.exact42852RawTerms
def rightRaw : List Term := Proof.Events052.exact13516RawTerms
def group : MergeGroup := .operator 42852 13516
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 42852) (leftOrdinal := 0)
    (rightResult := 13516) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6792⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨7840⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6792⟩⟩, ⟨.program ⟨214⟩, ⟨7840⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge42861

namespace LeftMerge42866
def owner : Owner := ⟨.program ⟨214⟩, ⟨12189⟩⟩
def mergeEvent : Nat := 42866
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨12181⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6775⟩⟩] } }
def leftRaw : List Term := Proof.Events167.exact42862RawTerms
def rightRaw : List Term := Proof.Events167.exact42832RawTerms
def group : MergeGroup := .operator 42862 42832
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 42862) (leftOrdinal := 1)
    (rightResult := 42832) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨12181⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6775⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨12181⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6775⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨12181⟩⟩], [⟨.program ⟨214⟩, ⟨6775⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge42866

namespace LeftMerge42874
def owner : Owner := ⟨.program ⟨214⟩, ⟨25307⟩⟩
def mergeEvent : Nat := 42874
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨11141⟩⟩, ⟨.program ⟨214⟩, ⟨12181⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25306⟩⟩] } }
def leftRaw : List Term := Proof.Events167.exact42868RawTerms
def rightRaw : List Term := Proof.Events167.exact42804RawTerms
def group : MergeGroup := .operator 42868 42804
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 42868) (leftOrdinal := 1)
    (rightResult := 42804) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨11141⟩⟩, ⟨.program ⟨214⟩, ⟨12181⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25306⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨11141⟩⟩, ⟨.program ⟨214⟩, ⟨12181⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25306⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge42874

namespace LeftMerge42876
def owner : Owner := ⟨.program ⟨214⟩, ⟨25307⟩⟩
def mergeEvent : Nat := 42876
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨11141⟩⟩, ⟨.program ⟨214⟩, ⟨12181⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23168⟩⟩] } }
def rhsRaw : List Term := Proof.Events167.exact42801RawTerms
def group : MergeGroup := .relation 42875
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 42875) (rhsResult := 42801)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨11141⟩⟩, ⟨.program ⟨214⟩, ⟨12181⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25306⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨25306⟩⟩) ⟨23168⟩ 42801) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23168⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨11141⟩⟩, ⟨.program ⟨214⟩, ⟨12181⟩⟩], [⟨.program ⟨214⟩, ⟨23168⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge42876

namespace LeftMerge42877
def owner : Owner := ⟨.program ⟨214⟩, ⟨25307⟩⟩
def mergeEvent : Nat := 42877
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6792⟩⟩, ⟨.program ⟨214⟩, ⟨7840⟩⟩, ⟨.program ⟨214⟩, ⟨25306⟩⟩] } }
def leftRaw : List Term := Proof.Events167.exact42868RawTerms
def rightRaw : List Term := Proof.Events167.exact42804RawTerms
def group : MergeGroup := .operator 42868 42804
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 42868) (leftOrdinal := 0)
    (rightResult := 42804) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6792⟩⟩, ⟨.program ⟨214⟩, ⟨7840⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25306⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6792⟩⟩, ⟨.program ⟨214⟩, ⟨7840⟩⟩, ⟨.program ⟨214⟩, ⟨25306⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge42877

namespace LeftMerge42891
def owner : Owner := ⟨.program ⟨214⟩, ⟨19251⟩⟩
def mergeEvent : Nat := 42891
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19248⟩⟩] } }
def leftRaw : List Term := Proof.Events141.exact36137RawTerms
def rightRaw : List Term := Proof.Events167.exact42885RawTerms
def group : MergeGroup := .operator 36137 42885
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 36137) (leftOrdinal := 0)
    (rightResult := 42885) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨19248⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19248⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge42891

namespace LeftMerge42970
def owner : Owner := ⟨.program ⟨214⟩, ⟨12182⟩⟩
def mergeEvent : Nat := 42970
def frameStart : Nat := 42940
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨11141⟩⟩, ⟨.program ⟨214⟩, ⟨12181⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events167.exact42966RawTerms
def rightRaw : List Term := Proof.Events167.exact42963RawTerms
def group : MergeGroup := .operator 42966 42963
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 42966) (leftOrdinal := 0)
    (rightResult := 42963) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨12181⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11141⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨11141⟩⟩, ⟨.program ⟨214⟩, ⟨12181⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge42970

namespace LeftMerge43000
def owner : Owner := ⟨.program ⟨214⟩, ⟨12280⟩⟩
def mergeEvent : Nat := 43000
def frameStart : Nat := 42940
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨11141⟩⟩, ⟨.program ⟨214⟩, ⟨12181⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events167.exact42996RawTerms
def rightRaw : List Term := Proof.Events167.exact42994RawTerms
def group : MergeGroup := .operator 42996 42994
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 42996) (leftOrdinal := 0)
    (rightResult := 42994) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11141⟩⟩, ⟨.program ⟨214⟩, ⟨12181⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨11141⟩⟩, ⟨.program ⟨214⟩, ⟨12181⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge43000

namespace LeftMerge43023
def owner : Owner := ⟨.program ⟨214⟩, ⟨7842⟩⟩
def mergeEvent : Nat := 43023
def frameStart : Nat := 42940
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6792⟩⟩, ⟨.program ⟨214⟩, ⟨7840⟩⟩] } }
def leftRaw : List Term := Proof.Events168.exact43019RawTerms
def rightRaw : List Term := Proof.Events168.exact43016RawTerms
def group : MergeGroup := .operator 43019 43016
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 43019) (leftOrdinal := 0)
    (rightResult := 43016) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6792⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨7840⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6792⟩⟩, ⟨.program ⟨214⟩, ⟨7840⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge43023

namespace LeftMerge43032
def owner : Owner := ⟨.program ⟨214⟩, ⟨25309⟩⟩
def mergeEvent : Nat := 43032
def frameStart : Nat := 42940
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6792⟩⟩, ⟨.program ⟨214⟩, ⟨7840⟩⟩, ⟨.program ⟨214⟩, ⟨25306⟩⟩] } }
def leftRaw : List Term := Proof.Events168.exact43028RawTerms
def rightRaw : List Term := Proof.Events167.exact42985RawTerms
def group : MergeGroup := .operator 43028 42985
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 43028) (leftOrdinal := 0)
    (rightResult := 42985) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6792⟩⟩, ⟨.program ⟨214⟩, ⟨7840⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25306⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6792⟩⟩, ⟨.program ⟨214⟩, ⟨7840⟩⟩, ⟨.program ⟨214⟩, ⟨25306⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge43032

namespace LeftMerge43033
def owner : Owner := ⟨.program ⟨214⟩, ⟨25309⟩⟩
def mergeEvent : Nat := 43033
def frameStart : Nat := 42940
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨11141⟩⟩, ⟨.program ⟨214⟩, ⟨12181⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25306⟩⟩] } }
def leftRaw : List Term := Proof.Events168.exact43028RawTerms
def rightRaw : List Term := Proof.Events167.exact42985RawTerms
def group : MergeGroup := .operator 43028 42985
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 43028) (leftOrdinal := 1)
    (rightResult := 42985) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11141⟩⟩, ⟨.program ⟨214⟩, ⟨12181⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25306⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨11141⟩⟩, ⟨.program ⟨214⟩, ⟨12181⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25306⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge43033

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
