import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge230589
def owner : Owner := ⟨.program ⟨257⟩, ⟨17351⟩⟩
def mergeEvent : Nat := 230589
def frameStart : Nat := 230494
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨12366⟩⟩, ⟨.program ⟨257⟩, ⟨15450⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨16843⟩⟩] } }
def rhsRaw : List Term := Proof.Events900.exact230536RawTerms
def group : MergeGroup := .relation 230588
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 230588) (rhsResult := 230536)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨12366⟩⟩, ⟨.program ⟨257⟩, ⟨15450⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17348⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨17348⟩⟩) ⟨16843⟩ 230536) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨16843⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨12366⟩⟩, ⟨.program ⟨257⟩, ⟨15450⟩⟩], [⟨.program ⟨257⟩, ⟨16843⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge230589

namespace LeftMerge230597
def owner : Owner := ⟨.program ⟨257⟩, ⟨15782⟩⟩
def mergeEvent : Nat := 230597
def frameStart : Nat := 230494
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨15780⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events900.exact230550RawTerms
def rightRaw : List Term := Proof.Events900.exact230593RawTerms
def group : MergeGroup := .operator 230550 230593
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 230550) (leftOrdinal := 0)
    (rightResult := 230593) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨15780⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨15780⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge230597

namespace LeftMerge230614
def owner : Owner := ⟨.program ⟨257⟩, ⟨16282⟩⟩
def mergeEvent : Nat := 230614
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7179⟩⟩] } }
def rhsRaw : List Term := Proof.Events900.exact230611RawTerms
def group : MergeGroup := .relation 230613
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 230613) (rhsResult := 230611)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16279⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 230612 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16279⟩⟩]⟩) (none) 230611) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7179⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7179⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge230614

namespace LeftMerge230615
def owner : Owner := ⟨.program ⟨257⟩, ⟨16282⟩⟩
def mergeEvent : Nat := 230615
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17348⟩⟩] } }
def rhsRaw : List Term := Proof.Events900.exact230611RawTerms
def group : MergeGroup := .relation 230613
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 230613) (rhsResult := 230611)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16279⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 230612 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16279⟩⟩]⟩) (none) 230611) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17348⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17348⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge230615

namespace LeftMerge230616
def owner : Owner := ⟨.program ⟨257⟩, ⟨16282⟩⟩
def mergeEvent : Nat := 230616
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨12366⟩⟩, ⟨.program ⟨257⟩, ⟨15450⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨16843⟩⟩] } }
def rhsRaw : List Term := Proof.Events900.exact230611RawTerms
def group : MergeGroup := .relation 230613
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 230613) (rhsResult := 230611)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16279⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 230612 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16279⟩⟩]⟩) (none) 230611) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨12366⟩⟩, ⟨.program ⟨257⟩, ⟨15450⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨16843⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨12366⟩⟩, ⟨.program ⟨257⟩, ⟨15450⟩⟩], [⟨.program ⟨257⟩, ⟨16843⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge230616

namespace LeftMerge230617
def owner : Owner := ⟨.program ⟨257⟩, ⟨16282⟩⟩
def mergeEvent : Nat := 230617
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨15780⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events900.exact230611RawTerms
def group : MergeGroup := .relation 230613
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 230613) (rhsResult := 230611)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16279⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 230612 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16279⟩⟩]⟩) (none) 230611) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨15780⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨15780⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge230617

namespace LeftMerge230622
def owner : Owner := ⟨.program ⟨257⟩, ⟨17350⟩⟩
def mergeEvent : Nat := 230622
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨12366⟩⟩, ⟨.program ⟨257⟩, ⟨15450⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨16843⟩⟩] } }
def leftRaw : List Term := Proof.Events900.exact230618RawTerms
def rightRaw : List Term := Proof.Events900.exact230432RawTerms
def group : MergeGroup := .operator 230618 230432
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 230618) (leftOrdinal := 2)
    (rightResult := 230432) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨12366⟩⟩, ⟨.program ⟨257⟩, ⟨15450⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨16843⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨12366⟩⟩, ⟨.program ⟨257⟩, ⟨15450⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨16843⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨12366⟩⟩, ⟨.program ⟨257⟩, ⟨15450⟩⟩], [⟨.program ⟨257⟩, ⟨16843⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge230622

namespace LeftMerge230623
def owner : Owner := ⟨.program ⟨257⟩, ⟨17350⟩⟩
def mergeEvent : Nat := 230623
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17348⟩⟩] } }
def leftRaw : List Term := Proof.Events900.exact230618RawTerms
def rightRaw : List Term := Proof.Events900.exact230432RawTerms
def group : MergeGroup := .operator 230618 230432
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 230618) (leftOrdinal := 1)
    (rightResult := 230432) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17348⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17348⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17348⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge230623

namespace LeftMerge230631
def owner : Owner := ⟨.program ⟨257⟩, ⟨17735⟩⟩
def mergeEvent : Nat := 230631
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17733⟩⟩] } }
def leftRaw : List Term := Proof.Events900.exact230625RawTerms
def rightRaw : List Term := Proof.Events899.exact230348RawTerms
def group : MergeGroup := .operator 230625 230348
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 230625) (leftOrdinal := 0)
    (rightResult := 230348) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7179⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨17733⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17733⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge230631

namespace LeftMerge230632
def owner : Owner := ⟨.program ⟨257⟩, ⟨17735⟩⟩
def mergeEvent : Nat := 230632
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨15780⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17733⟩⟩] } }
def leftRaw : List Term := Proof.Events900.exact230625RawTerms
def rightRaw : List Term := Proof.Events899.exact230348RawTerms
def group : MergeGroup := .operator 230625 230348
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 230625) (leftOrdinal := 1)
    (rightResult := 230348) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨15780⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨17733⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨15780⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17733⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge230632

namespace LeftMerge230634
def owner : Owner := ⟨.program ⟨257⟩, ⟨17735⟩⟩
def mergeEvent : Nat := 230634
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨15780⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨16992⟩⟩] } }
def rhsRaw : List Term := Proof.Events899.exact230345RawTerms
def group : MergeGroup := .relation 230633
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 230633) (rhsResult := 230345)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨15780⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17733⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨17733⟩⟩) ⟨16992⟩ 230345) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨16992⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨15780⟩⟩], [⟨.program ⟨257⟩, ⟨16992⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge230634

namespace LeftMerge230648
def owner : Owner := ⟨.program ⟨257⟩, ⟨16579⟩⟩
def mergeEvent : Nat := 230648
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16576⟩⟩] } }
def leftRaw : List Term := Proof.Events868.exact222245RawTerms
def rightRaw : List Term := Proof.Events900.exact230642RawTerms
def group : MergeGroup := .operator 222245 230642
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 222245) (leftOrdinal := 0)
    (rightResult := 230642) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨16576⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16576⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge230648

namespace LeftMerge230769
def owner : Owner := ⟨.program ⟨257⟩, ⟨17204⟩⟩
def mergeEvent : Nat := 230769
def frameStart : Nat := 230703
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨15780⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events901.exact230765RawTerms
def rightRaw : List Term := Proof.Events901.exact230763RawTerms
def group : MergeGroup := .operator 230765 230763
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 230765) (leftOrdinal := 0)
    (rightResult := 230763) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨15780⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨15780⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge230769

namespace LeftMerge230781
def owner : Owner := ⟨.program ⟨257⟩, ⟨17734⟩⟩
def mergeEvent : Nat := 230781
def frameStart : Nat := 230703
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17733⟩⟩] } }
def leftRaw : List Term := Proof.Events901.exact230777RawTerms
def rightRaw : List Term := Proof.Events901.exact230754RawTerms
def group : MergeGroup := .operator 230777 230754
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 230777) (leftOrdinal := 0)
    (rightResult := 230754) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7179⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨17733⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17733⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge230781

namespace LeftMerge230782
def owner : Owner := ⟨.program ⟨257⟩, ⟨17734⟩⟩
def mergeEvent : Nat := 230782
def frameStart : Nat := 230703
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨15780⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17733⟩⟩] } }
def leftRaw : List Term := Proof.Events901.exact230777RawTerms
def rightRaw : List Term := Proof.Events901.exact230754RawTerms
def group : MergeGroup := .operator 230777 230754
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 230777) (leftOrdinal := 1)
    (rightResult := 230754) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨15780⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨17733⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨15780⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17733⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge230782

namespace LeftMerge230784
def owner : Owner := ⟨.program ⟨257⟩, ⟨17734⟩⟩
def mergeEvent : Nat := 230784
def frameStart : Nat := 230703
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨15780⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨16992⟩⟩] } }
def rhsRaw : List Term := Proof.Events901.exact230751RawTerms
def group : MergeGroup := .relation 230783
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 230783) (rhsResult := 230751)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨15780⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17733⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨17733⟩⟩) ⟨16992⟩ 230751) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨16992⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨15780⟩⟩], [⟨.program ⟨257⟩, ⟨16992⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge230784

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
