import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge217810
def owner : Owner := ⟨.program ⟨257⟩, ⟨71239⟩⟩
def mergeEvent : Nat := 217810
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }
def leftRaw : List Term := Proof.Events850.exact217774RawTerms
def rightRaw : List Term := Proof.Events845.exact216358RawTerms
def group : MergeGroup := .operator 217774 216358
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 217774) (leftOrdinal := 1)
    (rightResult := 216358) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge217810

namespace LeftMerge217811
def owner : Owner := ⟨.program ⟨257⟩, ⟨71239⟩⟩
def mergeEvent : Nat := 217811
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨18866⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }
def leftRaw : List Term := Proof.Events850.exact217774RawTerms
def rightRaw : List Term := Proof.Events845.exact216358RawTerms
def group : MergeGroup := .operator 217774 216358
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 217774) (leftOrdinal := 20)
    (rightResult := 216358) (rightOrdinal := 19) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨18866⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨18866⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨18866⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge217811

namespace LeftMerge217812
def owner : Owner := ⟨.program ⟨257⟩, ⟨71239⟩⟩
def mergeEvent : Nat := 217812
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }
def leftRaw : List Term := Proof.Events850.exact217774RawTerms
def rightRaw : List Term := Proof.Events845.exact216358RawTerms
def group : MergeGroup := .operator 217774 216358
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 217774) (leftOrdinal := 0)
    (rightResult := 216358) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge217812

namespace LeftMerge217813
def owner : Owner := ⟨.program ⟨257⟩, ⟨71239⟩⟩
def mergeEvent : Nat := 217813
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨16035⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }
def leftRaw : List Term := Proof.Events850.exact217774RawTerms
def rightRaw : List Term := Proof.Events845.exact216358RawTerms
def group : MergeGroup := .operator 217774 216358
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 217774) (leftOrdinal := 19)
    (rightResult := 216358) (rightOrdinal := 18) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨16035⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨16035⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨16035⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge217813

namespace LeftMerge217821
def owner : Owner := ⟨.program ⟨257⟩, ⟨71240⟩⟩
def mergeEvent : Nat := 217821
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7233⟩⟩, ⟨.program ⟨257⟩, ⟨7139⟩⟩] } }
def leftRaw : List Term := Proof.Events850.exact217815RawTerms
def rightRaw : List Term := Proof.Events060.exact15522RawTerms
def group : MergeGroup := .operator 217815 15522
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 217815) (leftOrdinal := 0)
    (rightResult := 15522) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7233⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7139⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7233⟩⟩, ⟨.program ⟨257⟩, ⟨7139⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge217821

namespace LeftMerge217822
def owner : Owner := ⟨.program ⟨257⟩, ⟨71240⟩⟩
def mergeEvent : Nat := 217822
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨67457⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7139⟩⟩] } }
def leftRaw : List Term := Proof.Events850.exact217815RawTerms
def rightRaw : List Term := Proof.Events060.exact15522RawTerms
def group : MergeGroup := .operator 217815 15522
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 217815) (leftOrdinal := 1)
    (rightResult := 15522) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨67457⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7139⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨67457⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7139⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge217822

namespace LeftMerge217824
def owner : Owner := ⟨.program ⟨257⟩, ⟨71240⟩⟩
def mergeEvent : Nat := 217824
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67457⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events060.exact15515RawTerms
def group : MergeGroup := .relation 217823
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 217823) (rhsResult := 15515)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨67457⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7139⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨7139⟩⟩) ⟨7035⟩ 15515) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6774⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67457⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge217824

namespace LeftMerge217838
def owner : Owner := ⟨.program ⟨257⟩, ⟨50025⟩⟩
def mergeEvent : Nat := 217838
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨50023⟩⟩] } }
def leftRaw : List Term := Proof.Events811.exact207806RawTerms
def rightRaw : List Term := Proof.Events850.exact217832RawTerms
def group : MergeGroup := .operator 207806 217832
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 207806) (leftOrdinal := 0)
    (rightResult := 217832) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7196⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨50023⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨50023⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge217838

namespace LeftMerge217839
def owner : Owner := ⟨.program ⟨257⟩, ⟨50025⟩⟩
def mergeEvent : Nat := 217839
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨48148⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨50023⟩⟩] } }
def leftRaw : List Term := Proof.Events811.exact207806RawTerms
def rightRaw : List Term := Proof.Events850.exact217832RawTerms
def group : MergeGroup := .operator 207806 217832
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 207806) (leftOrdinal := 1)
    (rightResult := 217832) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨48148⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨50023⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨48148⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨50023⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge217839

namespace LeftMerge217841
def owner : Owner := ⟨.program ⟨257⟩, ⟨50025⟩⟩
def mergeEvent : Nat := 217841
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨48148⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨49300⟩⟩] } }
def rhsRaw : List Term := Proof.Events850.exact217829RawTerms
def group : MergeGroup := .relation 217840
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 217840) (rhsResult := 217829)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨48148⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨50023⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨50023⟩⟩) ⟨49300⟩ 217829) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨49300⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨48148⟩⟩], [⟨.program ⟨257⟩, ⟨49300⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge217841

namespace LeftMerge217855
def owner : Owner := ⟨.program ⟨257⟩, ⟨48895⟩⟩
def mergeEvent : Nat := 217855
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48892⟩⟩] } }
def leftRaw : List Term := Proof.Events811.exact207620RawTerms
def rightRaw : List Term := Proof.Events850.exact217849RawTerms
def group : MergeGroup := .operator 207620 217849
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 207620) (leftOrdinal := 0)
    (rightResult := 217849) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨48892⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48892⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge217855

namespace LeftMerge217976
def owner : Owner := ⟨.program ⟨257⟩, ⟨49508⟩⟩
def mergeEvent : Nat := 217976
def frameStart : Nat := 217910
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨48148⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events851.exact217972RawTerms
def rightRaw : List Term := Proof.Events851.exact217970RawTerms
def group : MergeGroup := .operator 217972 217970
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 217972) (leftOrdinal := 0)
    (rightResult := 217970) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨48148⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨48148⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge217976

namespace LeftMerge217988
def owner : Owner := ⟨.program ⟨257⟩, ⟨50024⟩⟩
def mergeEvent : Nat := 217988
def frameStart : Nat := 217910
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨50023⟩⟩] } }
def leftRaw : List Term := Proof.Events851.exact217984RawTerms
def rightRaw : List Term := Proof.Events851.exact217961RawTerms
def group : MergeGroup := .operator 217984 217961
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 217984) (leftOrdinal := 0)
    (rightResult := 217961) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7196⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨50023⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨50023⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge217988

namespace LeftMerge217989
def owner : Owner := ⟨.program ⟨257⟩, ⟨50024⟩⟩
def mergeEvent : Nat := 217989
def frameStart : Nat := 217910
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨48148⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨50023⟩⟩] } }
def leftRaw : List Term := Proof.Events851.exact217984RawTerms
def rightRaw : List Term := Proof.Events851.exact217961RawTerms
def group : MergeGroup := .operator 217984 217961
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 217984) (leftOrdinal := 1)
    (rightResult := 217961) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨48148⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨50023⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨48148⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨50023⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge217989

namespace LeftMerge217991
def owner : Owner := ⟨.program ⟨257⟩, ⟨50024⟩⟩
def mergeEvent : Nat := 217991
def frameStart : Nat := 217910
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨48148⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨49300⟩⟩] } }
def rhsRaw : List Term := Proof.Events851.exact217958RawTerms
def group : MergeGroup := .relation 217990
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 217990) (rhsResult := 217958)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨48148⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨50023⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨50023⟩⟩) ⟨49300⟩ 217958) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨49300⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨48148⟩⟩], [⟨.program ⟨257⟩, ⟨49300⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge217991

namespace LeftMerge217999
def owner : Owner := ⟨.program ⟨257⟩, ⟨48361⟩⟩
def mergeEvent : Nat := 217999
def frameStart : Nat := 217910
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨48359⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events851.exact217972RawTerms
def rightRaw : List Term := Proof.Events851.exact217995RawTerms
def group : MergeGroup := .operator 217972 217995
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 217972) (leftOrdinal := 0)
    (rightResult := 217995) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨48359⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨48359⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge217999

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
