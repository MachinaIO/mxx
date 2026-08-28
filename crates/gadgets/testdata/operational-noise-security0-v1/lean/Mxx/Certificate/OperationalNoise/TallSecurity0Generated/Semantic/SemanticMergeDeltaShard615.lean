import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge99662
def owner : Owner := ⟨.program ⟨214⟩, ⟨25900⟩⟩
def mergeEvent : Nat := 99662
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6794⟩⟩, ⟨.program ⟨214⟩, ⟨7846⟩⟩, ⟨.program ⟨214⟩, ⟨25899⟩⟩] } }
def leftRaw : List Term := Proof.Events389.exact99653RawTerms
def rightRaw : List Term := Proof.Events389.exact99589RawTerms
def group : MergeGroup := .operator 99653 99589
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 99653) (leftOrdinal := 0)
    (rightResult := 99589) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6794⟩⟩, ⟨.program ⟨214⟩, ⟨7846⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25899⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6794⟩⟩, ⟨.program ⟨214⟩, ⟨7846⟩⟩, ⟨.program ⟨214⟩, ⟨25899⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge99662

namespace LeftMerge99676
def owner : Owner := ⟨.program ⟨214⟩, ⟨19376⟩⟩
def mergeEvent : Nat := 99676
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19373⟩⟩] } }
def leftRaw : List Term := Proof.Events368.exact94462RawTerms
def rightRaw : List Term := Proof.Events389.exact99670RawTerms
def group : MergeGroup := .operator 94462 99670
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 94462) (leftOrdinal := 0)
    (rightResult := 99670) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨19373⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19373⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge99676

namespace LeftMerge99731
def owner : Owner := ⟨.program ⟨214⟩, ⟨13747⟩⟩
def mergeEvent : Nat := 99731
def frameStart : Nat := 99713
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨11289⟩⟩, ⟨.program ⟨214⟩, ⟨13746⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events389.exact99727RawTerms
def rightRaw : List Term := Proof.Events389.exact99724RawTerms
def group : MergeGroup := .operator 99727 99724
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 99727) (leftOrdinal := 0)
    (rightResult := 99724) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨13746⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11289⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨11289⟩⟩, ⟨.program ⟨214⟩, ⟨13746⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge99731

namespace LeftMerge99761
def owner : Owner := ⟨.program ⟨214⟩, ⟨13874⟩⟩
def mergeEvent : Nat := 99761
def frameStart : Nat := 99713
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨11289⟩⟩, ⟨.program ⟨214⟩, ⟨13746⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events389.exact99757RawTerms
def rightRaw : List Term := Proof.Events389.exact99755RawTerms
def group : MergeGroup := .operator 99757 99755
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 99757) (leftOrdinal := 0)
    (rightResult := 99755) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11289⟩⟩, ⟨.program ⟨214⟩, ⟨13746⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨11289⟩⟩, ⟨.program ⟨214⟩, ⟨13746⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge99761

namespace LeftMerge99784
def owner : Owner := ⟨.program ⟨214⟩, ⟨7848⟩⟩
def mergeEvent : Nat := 99784
def frameStart : Nat := 99713
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6794⟩⟩, ⟨.program ⟨214⟩, ⟨7846⟩⟩] } }
def leftRaw : List Term := Proof.Events389.exact99780RawTerms
def rightRaw : List Term := Proof.Events389.exact99777RawTerms
def group : MergeGroup := .operator 99780 99777
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 99780) (leftOrdinal := 0)
    (rightResult := 99777) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6794⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨7846⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6794⟩⟩, ⟨.program ⟨214⟩, ⟨7846⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge99784

namespace LeftMerge99793
def owner : Owner := ⟨.program ⟨214⟩, ⟨25902⟩⟩
def mergeEvent : Nat := 99793
def frameStart : Nat := 99713
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6794⟩⟩, ⟨.program ⟨214⟩, ⟨7846⟩⟩, ⟨.program ⟨214⟩, ⟨25899⟩⟩] } }
def leftRaw : List Term := Proof.Events389.exact99789RawTerms
def rightRaw : List Term := Proof.Events389.exact99746RawTerms
def group : MergeGroup := .operator 99789 99746
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 99789) (leftOrdinal := 0)
    (rightResult := 99746) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6794⟩⟩, ⟨.program ⟨214⟩, ⟨7846⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25899⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6794⟩⟩, ⟨.program ⟨214⟩, ⟨7846⟩⟩, ⟨.program ⟨214⟩, ⟨25899⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge99793

namespace LeftMerge99794
def owner : Owner := ⟨.program ⟨214⟩, ⟨25902⟩⟩
def mergeEvent : Nat := 99794
def frameStart : Nat := 99713
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨11289⟩⟩, ⟨.program ⟨214⟩, ⟨13746⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25899⟩⟩] } }
def leftRaw : List Term := Proof.Events389.exact99789RawTerms
def rightRaw : List Term := Proof.Events389.exact99746RawTerms
def group : MergeGroup := .operator 99789 99746
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 99789) (leftOrdinal := 1)
    (rightResult := 99746) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11289⟩⟩, ⟨.program ⟨214⟩, ⟨13746⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25899⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨11289⟩⟩, ⟨.program ⟨214⟩, ⟨13746⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25899⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge99794

namespace LeftMerge99796
def owner : Owner := ⟨.program ⟨214⟩, ⟨25902⟩⟩
def mergeEvent : Nat := 99796
def frameStart : Nat := 99713
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨11289⟩⟩, ⟨.program ⟨214⟩, ⟨13746⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23494⟩⟩] } }
def rhsRaw : List Term := Proof.Events389.exact99743RawTerms
def group : MergeGroup := .relation 99795
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 99795) (rhsResult := 99743)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨11289⟩⟩, ⟨.program ⟨214⟩, ⟨13746⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25899⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨25899⟩⟩) ⟨23494⟩ 99743) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23494⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨11289⟩⟩, ⟨.program ⟨214⟩, ⟨13746⟩⟩], [⟨.program ⟨214⟩, ⟨23494⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge99796

namespace LeftMerge99804
def owner : Owner := ⟨.program ⟨214⟩, ⟨15694⟩⟩
def mergeEvent : Nat := 99804
def frameStart : Nat := 99713
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15692⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events389.exact99757RawTerms
def rightRaw : List Term := Proof.Events389.exact99800RawTerms
def group : MergeGroup := .operator 99757 99800
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 99757) (leftOrdinal := 0)
    (rightResult := 99800) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15692⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨15692⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge99804

namespace LeftMerge99821
def owner : Owner := ⟨.program ⟨214⟩, ⟨19376⟩⟩
def mergeEvent : Nat := 99821
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6695⟩⟩] } }
def rhsRaw : List Term := Proof.Events389.exact99818RawTerms
def group : MergeGroup := .relation 99820
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 99820) (rhsResult := 99818)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19373⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 99819 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19373⟩⟩]⟩) (none) 99818) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6695⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6695⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge99821

namespace LeftMerge99822
def owner : Owner := ⟨.program ⟨214⟩, ⟨19376⟩⟩
def mergeEvent : Nat := 99822
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6794⟩⟩, ⟨.program ⟨214⟩, ⟨7846⟩⟩, ⟨.program ⟨214⟩, ⟨25899⟩⟩] } }
def rhsRaw : List Term := Proof.Events389.exact99818RawTerms
def group : MergeGroup := .relation 99820
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 99820) (rhsResult := 99818)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19373⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 99819 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19373⟩⟩]⟩) (none) 99818) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6794⟩⟩, ⟨.program ⟨214⟩, ⟨7846⟩⟩, ⟨.program ⟨214⟩, ⟨25899⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6794⟩⟩, ⟨.program ⟨214⟩, ⟨7846⟩⟩, ⟨.program ⟨214⟩, ⟨25899⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge99822

namespace LeftMerge99823
def owner : Owner := ⟨.program ⟨214⟩, ⟨19376⟩⟩
def mergeEvent : Nat := 99823
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨11289⟩⟩, ⟨.program ⟨214⟩, ⟨13746⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23494⟩⟩] } }
def rhsRaw : List Term := Proof.Events389.exact99818RawTerms
def group : MergeGroup := .relation 99820
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 99820) (rhsResult := 99818)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19373⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 99819 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19373⟩⟩]⟩) (none) 99818) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11289⟩⟩, ⟨.program ⟨214⟩, ⟨13746⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23494⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨11289⟩⟩, ⟨.program ⟨214⟩, ⟨13746⟩⟩], [⟨.program ⟨214⟩, ⟨23494⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge99823

namespace LeftMerge99824
def owner : Owner := ⟨.program ⟨214⟩, ⟨19376⟩⟩
def mergeEvent : Nat := 99824
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15692⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def rhsRaw : List Term := Proof.Events389.exact99818RawTerms
def group : MergeGroup := .relation 99820
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 99820) (rhsResult := 99818)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19373⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 99819 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19373⟩⟩]⟩) (none) 99818) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15692⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15692⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge99824

namespace LeftMerge99829
def owner : Owner := ⟨.program ⟨214⟩, ⟨25901⟩⟩
def mergeEvent : Nat := 99829
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨11289⟩⟩, ⟨.program ⟨214⟩, ⟨13746⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23494⟩⟩] } }
def leftRaw : List Term := Proof.Events389.exact99825RawTerms
def rightRaw : List Term := Proof.Events389.exact99663RawTerms
def group : MergeGroup := .operator 99825 99663
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 99825) (leftOrdinal := 2)
    (rightResult := 99663) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨11289⟩⟩, ⟨.program ⟨214⟩, ⟨13746⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23494⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨11289⟩⟩, ⟨.program ⟨214⟩, ⟨13746⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23494⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨11289⟩⟩, ⟨.program ⟨214⟩, ⟨13746⟩⟩], [⟨.program ⟨214⟩, ⟨23494⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge99829

namespace LeftMerge99830
def owner : Owner := ⟨.program ⟨214⟩, ⟨25901⟩⟩
def mergeEvent : Nat := 99830
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6794⟩⟩, ⟨.program ⟨214⟩, ⟨7846⟩⟩, ⟨.program ⟨214⟩, ⟨25899⟩⟩] } }
def leftRaw : List Term := Proof.Events389.exact99825RawTerms
def rightRaw : List Term := Proof.Events389.exact99663RawTerms
def group : MergeGroup := .operator 99825 99663
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 99825) (leftOrdinal := 1)
    (rightResult := 99663) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6794⟩⟩, ⟨.program ⟨214⟩, ⟨7846⟩⟩, ⟨.program ⟨214⟩, ⟨25899⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6794⟩⟩, ⟨.program ⟨214⟩, ⟨7846⟩⟩, ⟨.program ⟨214⟩, ⟨25899⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6794⟩⟩, ⟨.program ⟨214⟩, ⟨7846⟩⟩, ⟨.program ⟨214⟩, ⟨25899⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge99830

namespace LeftMerge99838
def owner : Owner := ⟨.program ⟨214⟩, ⟨27399⟩⟩
def mergeEvent : Nat := 99838
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6695⟩⟩, ⟨.program ⟨214⟩, ⟨27397⟩⟩] } }
def leftRaw : List Term := Proof.Events389.exact99832RawTerms
def rightRaw : List Term := Proof.Events388.exact99579RawTerms
def group : MergeGroup := .operator 99832 99579
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 99832) (leftOrdinal := 0)
    (rightResult := 99579) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6695⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨27397⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6695⟩⟩, ⟨.program ⟨214⟩, ⟨27397⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge99838

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
