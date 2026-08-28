import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge33744
def owner : Owner := ⟨.program ⟨257⟩, ⟨40642⟩⟩
def mergeEvent : Nat := 33744
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨40180⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events131.exact33738RawTerms
def group : MergeGroup := .relation 33740
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 33740) (rhsResult := 33738)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40639⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 33739 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40639⟩⟩]⟩) (none) 33738) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨40180⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨40180⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge33744

namespace LeftMerge33749
def owner : Owner := ⟨.program ⟨257⟩, ⟨41720⟩⟩
def mergeEvent : Nat := 33749
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨14316⟩⟩, ⟨.program ⟨257⟩, ⟨40010⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41163⟩⟩] } }
def leftRaw : List Term := Proof.Events131.exact33745RawTerms
def rightRaw : List Term := Proof.Events131.exact33559RawTerms
def group : MergeGroup := .operator 33745 33559
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 33745) (leftOrdinal := 2)
    (rightResult := 33559) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨14316⟩⟩, ⟨.program ⟨257⟩, ⟨40010⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41163⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨14316⟩⟩, ⟨.program ⟨257⟩, ⟨40010⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41163⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨14316⟩⟩, ⟨.program ⟨257⟩, ⟨40010⟩⟩], [⟨.program ⟨257⟩, ⟨41163⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge33749

namespace LeftMerge33750
def owner : Owner := ⟨.program ⟨257⟩, ⟨41720⟩⟩
def mergeEvent : Nat := 33750
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41718⟩⟩] } }
def leftRaw : List Term := Proof.Events131.exact33745RawTerms
def rightRaw : List Term := Proof.Events131.exact33559RawTerms
def group : MergeGroup := .operator 33745 33559
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 33745) (leftOrdinal := 1)
    (rightResult := 33559) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41718⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41718⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41718⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge33750

namespace LeftMerge33758
def owner : Owner := ⟨.program ⟨257⟩, ⟨42216⟩⟩
def mergeEvent : Nat := 33758
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨42214⟩⟩] } }
def leftRaw : List Term := Proof.Events131.exact33752RawTerms
def rightRaw : List Term := Proof.Events130.exact33475RawTerms
def group : MergeGroup := .operator 33752 33475
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 33752) (leftOrdinal := 0)
    (rightResult := 33475) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7193⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨42214⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨42214⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge33758

namespace LeftMerge33759
def owner : Owner := ⟨.program ⟨257⟩, ⟨42216⟩⟩
def mergeEvent : Nat := 33759
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨40180⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨42214⟩⟩] } }
def leftRaw : List Term := Proof.Events131.exact33752RawTerms
def rightRaw : List Term := Proof.Events130.exact33475RawTerms
def group : MergeGroup := .operator 33752 33475
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 33752) (leftOrdinal := 1)
    (rightResult := 33475) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨40180⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨42214⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨40180⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨42214⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge33759

namespace LeftMerge33761
def owner : Owner := ⟨.program ⟨257⟩, ⟨42216⟩⟩
def mergeEvent : Nat := 33761
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨40180⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41342⟩⟩] } }
def rhsRaw : List Term := Proof.Events130.exact33472RawTerms
def group : MergeGroup := .relation 33760
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 33760) (rhsResult := 33472)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨40180⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨42214⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨42214⟩⟩) ⟨41342⟩ 33472) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨41342⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨40180⟩⟩], [⟨.program ⟨257⟩, ⟨41342⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge33761

namespace LeftMerge33775
def owner : Owner := ⟨.program ⟨257⟩, ⟨41039⟩⟩
def mergeEvent : Nat := 33775
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨41036⟩⟩] } }
def leftRaw : List Term := Proof.Events125.exact32120RawTerms
def rightRaw : List Term := Proof.Events131.exact33769RawTerms
def group : MergeGroup := .operator 32120 33769
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 32120) (leftOrdinal := 0)
    (rightResult := 33769) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨41036⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨41036⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge33775

namespace LeftMerge33896
def owner : Owner := ⟨.program ⟨257⟩, ⟨41504⟩⟩
def mergeEvent : Nat := 33896
def frameStart : Nat := 33830
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨40180⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events132.exact33892RawTerms
def rightRaw : List Term := Proof.Events132.exact33890RawTerms
def group : MergeGroup := .operator 33892 33890
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 33892) (leftOrdinal := 0)
    (rightResult := 33890) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨40180⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨40180⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge33896

namespace LeftMerge33908
def owner : Owner := ⟨.program ⟨257⟩, ⟨42215⟩⟩
def mergeEvent : Nat := 33908
def frameStart : Nat := 33830
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨42214⟩⟩] } }
def leftRaw : List Term := Proof.Events132.exact33904RawTerms
def rightRaw : List Term := Proof.Events132.exact33881RawTerms
def group : MergeGroup := .operator 33904 33881
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 33904) (leftOrdinal := 0)
    (rightResult := 33881) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7193⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨42214⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨42214⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge33908

namespace LeftMerge33909
def owner : Owner := ⟨.program ⟨257⟩, ⟨42215⟩⟩
def mergeEvent : Nat := 33909
def frameStart : Nat := 33830
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨40180⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨42214⟩⟩] } }
def leftRaw : List Term := Proof.Events132.exact33904RawTerms
def rightRaw : List Term := Proof.Events132.exact33881RawTerms
def group : MergeGroup := .operator 33904 33881
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 33904) (leftOrdinal := 1)
    (rightResult := 33881) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨40180⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨42214⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨40180⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨42214⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge33909

namespace LeftMerge33911
def owner : Owner := ⟨.program ⟨257⟩, ⟨42215⟩⟩
def mergeEvent : Nat := 33911
def frameStart : Nat := 33830
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨40180⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41342⟩⟩] } }
def rhsRaw : List Term := Proof.Events132.exact33878RawTerms
def group : MergeGroup := .relation 33910
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 33910) (rhsResult := 33878)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨40180⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨42214⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨42214⟩⟩) ⟨41342⟩ 33878) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨41342⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨40180⟩⟩], [⟨.program ⟨257⟩, ⟨41342⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge33911

namespace LeftMerge33919
def owner : Owner := ⟨.program ⟨257⟩, ⟨40437⟩⟩
def mergeEvent : Nat := 33919
def frameStart : Nat := 33830
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨40436⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events132.exact33892RawTerms
def rightRaw : List Term := Proof.Events132.exact33915RawTerms
def group : MergeGroup := .operator 33892 33915
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 33892) (leftOrdinal := 0)
    (rightResult := 33915) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨40436⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨40436⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge33919

namespace LeftMerge33936
def owner : Owner := ⟨.program ⟨257⟩, ⟨41039⟩⟩
def mergeEvent : Nat := 33936
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7226⟩⟩] } }
def rhsRaw : List Term := Proof.Events132.exact33933RawTerms
def group : MergeGroup := .relation 33935
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 33935) (rhsResult := 33933)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨41036⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 33934 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨41036⟩⟩]⟩) (none) 33933) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7226⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7226⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge33936

namespace LeftMerge33937
def owner : Owner := ⟨.program ⟨257⟩, ⟨41039⟩⟩
def mergeEvent : Nat := 33937
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨42214⟩⟩] } }
def rhsRaw : List Term := Proof.Events132.exact33933RawTerms
def group : MergeGroup := .relation 33935
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 33935) (rhsResult := 33933)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨41036⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 33934 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨41036⟩⟩]⟩) (none) 33933) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨42214⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨42214⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge33937

namespace LeftMerge33938
def owner : Owner := ⟨.program ⟨257⟩, ⟨41039⟩⟩
def mergeEvent : Nat := 33938
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨40180⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41342⟩⟩] } }
def rhsRaw : List Term := Proof.Events132.exact33933RawTerms
def group : MergeGroup := .relation 33935
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 33935) (rhsResult := 33933)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨41036⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 33934 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨41036⟩⟩]⟩) (none) 33933) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨40180⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41342⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨40180⟩⟩], [⟨.program ⟨257⟩, ⟨41342⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge33938

namespace LeftMerge33939
def owner : Owner := ⟨.program ⟨257⟩, ⟨41039⟩⟩
def mergeEvent : Nat := 33939
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨40436⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events132.exact33933RawTerms
def group : MergeGroup := .relation 33935
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 33935) (rhsResult := 33933)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨41036⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 33934 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨41036⟩⟩]⟩) (none) 33933) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨40436⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨40436⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge33939

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
