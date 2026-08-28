import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge86816
def owner : Owner := ⟨.program ⟨257⟩, ⟨43655⟩⟩
def mergeEvent : Nat := 86816
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44813⟩⟩] } }
def rhsRaw : List Term := Proof.Events339.exact86812RawTerms
def group : MergeGroup := .relation 86814
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 86814) (rhsResult := 86812)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43652⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 86813 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43652⟩⟩]⟩) (none) 86812) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44813⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44813⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge86816

namespace LeftMerge86817
def owner : Owner := ⟨.program ⟨257⟩, ⟨43655⟩⟩
def mergeEvent : Nat := 86817
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨42836⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨43994⟩⟩] } }
def rhsRaw : List Term := Proof.Events339.exact86812RawTerms
def group : MergeGroup := .relation 86814
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 86814) (rhsResult := 86812)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43652⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 86813 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43652⟩⟩]⟩) (none) 86812) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨42836⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨43994⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨42836⟩⟩], [⟨.program ⟨257⟩, ⟨43994⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge86817

namespace LeftMerge86818
def owner : Owner := ⟨.program ⟨257⟩, ⟨43655⟩⟩
def mergeEvent : Nat := 86818
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨43080⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events339.exact86812RawTerms
def group : MergeGroup := .relation 86814
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 86814) (rhsResult := 86812)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43652⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 86813 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43652⟩⟩]⟩) (none) 86812) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨43080⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨43080⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge86818

namespace LeftMerge86823
def owner : Owner := ⟨.program ⟨257⟩, ⟨44816⟩⟩
def mergeEvent : Nat := 86823
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44813⟩⟩] } }
def leftRaw : List Term := Proof.Events339.exact86819RawTerms
def rightRaw : List Term := Proof.Events338.exact86641RawTerms
def group : MergeGroup := .operator 86819 86641
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 86819) (leftOrdinal := 0)
    (rightResult := 86641) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44813⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44813⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44813⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge86823

namespace LeftMerge86824
def owner : Owner := ⟨.program ⟨257⟩, ⟨44816⟩⟩
def mergeEvent : Nat := 86824
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨42836⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨43994⟩⟩] } }
def leftRaw : List Term := Proof.Events339.exact86819RawTerms
def rightRaw : List Term := Proof.Events338.exact86641RawTerms
def group : MergeGroup := .operator 86819 86641
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 86819) (leftOrdinal := 2)
    (rightResult := 86641) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨42836⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨43994⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨42836⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨43994⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨42836⟩⟩], [⟨.program ⟨257⟩, ⟨43994⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge86824

namespace LeftMerge86832
def owner : Owner := ⟨.program ⟨257⟩, ⟨44817⟩⟩
def mergeEvent : Nat := 86832
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7227⟩⟩, ⟨.program ⟨257⟩, ⟨7153⟩⟩] } }
def leftRaw : List Term := Proof.Events339.exact86826RawTerms
def rightRaw : List Term := Proof.Events060.exact15582RawTerms
def group : MergeGroup := .operator 86826 15582
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 86826) (leftOrdinal := 0)
    (rightResult := 15582) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7227⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7153⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7227⟩⟩, ⟨.program ⟨257⟩, ⟨7153⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge86832

namespace LeftMerge86833
def owner : Owner := ⟨.program ⟨257⟩, ⟨44817⟩⟩
def mergeEvent : Nat := 86833
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨43080⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7153⟩⟩] } }
def leftRaw : List Term := Proof.Events339.exact86826RawTerms
def rightRaw : List Term := Proof.Events060.exact15582RawTerms
def group : MergeGroup := .operator 86826 15582
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 86826) (leftOrdinal := 1)
    (rightResult := 15582) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨43080⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7153⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨43080⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7153⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge86833

namespace LeftMerge86835
def owner : Owner := ⟨.program ⟨257⟩, ⟨44817⟩⟩
def mergeEvent : Nat := 86835
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨43080⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events060.exact15575RawTerms
def group : MergeGroup := .relation 86834
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 86834) (rhsResult := 15575)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨43080⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7153⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨7153⟩⟩) ⟨7042⟩ 15575) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6817⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨43080⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge86835

namespace LeftMerge86849
def owner : Owner := ⟨.program ⟨257⟩, ⟨42135⟩⟩
def mergeEvent : Nat := 86849
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨42133⟩⟩] } }
def leftRaw : List Term := Proof.Events303.exact77627RawTerms
def rightRaw : List Term := Proof.Events339.exact86843RawTerms
def group : MergeGroup := .operator 77627 86843
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 77627) (leftOrdinal := 0)
    (rightResult := 86843) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7193⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨42133⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨42133⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge86849

namespace LeftMerge86850
def owner : Owner := ⟨.program ⟨257⟩, ⟨42135⟩⟩
def mergeEvent : Nat := 86850
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨40156⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨42133⟩⟩] } }
def leftRaw : List Term := Proof.Events303.exact77627RawTerms
def rightRaw : List Term := Proof.Events339.exact86843RawTerms
def group : MergeGroup := .operator 77627 86843
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 77627) (leftOrdinal := 1)
    (rightResult := 86843) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨40156⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨42133⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨40156⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨42133⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge86850

namespace LeftMerge86852
def owner : Owner := ⟨.program ⟨257⟩, ⟨42135⟩⟩
def mergeEvent : Nat := 86852
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨40156⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41314⟩⟩] } }
def rhsRaw : List Term := Proof.Events339.exact86840RawTerms
def group : MergeGroup := .relation 86851
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 86851) (rhsResult := 86840)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨40156⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨42133⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨42133⟩⟩) ⟨41314⟩ 86840) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨41314⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨40156⟩⟩], [⟨.program ⟨257⟩, ⟨41314⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge86852

namespace LeftMerge86866
def owner : Owner := ⟨.program ⟨257⟩, ⟨40975⟩⟩
def mergeEvent : Nat := 86866
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40972⟩⟩] } }
def leftRaw : List Term := Proof.Events296.exact75995RawTerms
def rightRaw : List Term := Proof.Events339.exact86860RawTerms
def group : MergeGroup := .operator 75995 86860
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 75995) (leftOrdinal := 0)
    (rightResult := 86860) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨40972⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40972⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge86866

namespace LeftMerge86987
def owner : Owner := ⟨.program ⟨257⟩, ⟨41492⟩⟩
def mergeEvent : Nat := 86987
def frameStart : Nat := 86921
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨40156⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events339.exact86983RawTerms
def rightRaw : List Term := Proof.Events339.exact86981RawTerms
def group : MergeGroup := .operator 86983 86981
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 86983) (leftOrdinal := 0)
    (rightResult := 86981) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨40156⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨40156⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge86987

namespace LeftMerge86999
def owner : Owner := ⟨.program ⟨257⟩, ⟨42134⟩⟩
def mergeEvent : Nat := 86999
def frameStart : Nat := 86921
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨42133⟩⟩] } }
def leftRaw : List Term := Proof.Events339.exact86995RawTerms
def rightRaw : List Term := Proof.Events339.exact86972RawTerms
def group : MergeGroup := .operator 86995 86972
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 86995) (leftOrdinal := 0)
    (rightResult := 86972) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7193⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨42133⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨42133⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge86999

namespace LeftMerge87000
def owner : Owner := ⟨.program ⟨257⟩, ⟨42134⟩⟩
def mergeEvent : Nat := 87000
def frameStart : Nat := 86921
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨40156⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨42133⟩⟩] } }
def leftRaw : List Term := Proof.Events339.exact86995RawTerms
def rightRaw : List Term := Proof.Events339.exact86972RawTerms
def group : MergeGroup := .operator 86995 86972
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 86995) (leftOrdinal := 1)
    (rightResult := 86972) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨40156⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨42133⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨40156⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨42133⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge87000

namespace LeftMerge87002
def owner : Owner := ⟨.program ⟨257⟩, ⟨42134⟩⟩
def mergeEvent : Nat := 87002
def frameStart : Nat := 86921
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨40156⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41314⟩⟩] } }
def rhsRaw : List Term := Proof.Events339.exact86969RawTerms
def group : MergeGroup := .relation 87001
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 87001) (rhsResult := 86969)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨40156⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨42133⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨42133⟩⟩) ⟨41314⟩ 86969) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨41314⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨40156⟩⟩], [⟨.program ⟨257⟩, ⟨41314⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge87002

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
