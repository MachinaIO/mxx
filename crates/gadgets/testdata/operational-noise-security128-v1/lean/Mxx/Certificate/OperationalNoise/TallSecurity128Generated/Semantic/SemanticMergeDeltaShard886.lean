import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge145969
def owner : Owner := ⟨.program ⟨257⟩, ⟨36452⟩⟩
def mergeEvent : Nat := 145969
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨34868⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7163⟩⟩] } }
def leftRaw : List Term := Proof.Events570.exact145962RawTerms
def rightRaw : List Term := Proof.Events061.exact15642RawTerms
def group : MergeGroup := .operator 145962 15642
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 145962) (leftOrdinal := 1)
    (rightResult := 15642) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨34868⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7163⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨34868⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7163⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge145969

namespace LeftMerge145971
def owner : Owner := ⟨.program ⟨257⟩, ⟨36452⟩⟩
def mergeEvent : Nat := 145971
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34868⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events061.exact15635RawTerms
def group : MergeGroup := .relation 145970
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 145970) (rhsResult := 15635)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨34868⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7163⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨7163⟩⟩) ⟨7047⟩ 15635) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6842⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34868⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge145971

namespace LeftMerge145985
def owner : Owner := ⟨.program ⟨257⟩, ⟨30790⟩⟩
def mergeEvent : Nat := 145985
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨30788⟩⟩] } }
def leftRaw : List Term := Proof.Events537.exact137573RawTerms
def rightRaw : List Term := Proof.Events570.exact145979RawTerms
def group : MergeGroup := .operator 137573 145979
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 137573) (leftOrdinal := 0)
    (rightResult := 145979) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7190⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨30788⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨30788⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge145985

namespace LeftMerge145986
def owner : Owner := ⟨.program ⟨257⟩, ⟨30790⟩⟩
def mergeEvent : Nat := 145986
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨29032⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨30788⟩⟩] } }
def leftRaw : List Term := Proof.Events537.exact137573RawTerms
def rightRaw : List Term := Proof.Events570.exact145979RawTerms
def group : MergeGroup := .operator 137573 145979
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 137573) (leftOrdinal := 1)
    (rightResult := 145979) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨29032⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨30788⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨29032⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨30788⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge145986

namespace LeftMerge145988
def owner : Owner := ⟨.program ⟨257⟩, ⟨30790⟩⟩
def mergeEvent : Nat := 145988
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨29032⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨30177⟩⟩] } }
def rhsRaw : List Term := Proof.Events570.exact145976RawTerms
def group : MergeGroup := .relation 145987
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 145987) (rhsResult := 145976)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨29032⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨30788⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨30788⟩⟩) ⟨30177⟩ 145976) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨30177⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨29032⟩⟩], [⟨.program ⟨257⟩, ⟨30177⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge145988

namespace LeftMerge146002
def owner : Owner := ⟨.program ⟨257⟩, ⟨29695⟩⟩
def mergeEvent : Nat := 146002
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29692⟩⟩] } }
def leftRaw : List Term := Proof.Events525.exact134495RawTerms
def rightRaw : List Term := Proof.Events570.exact145996RawTerms
def group : MergeGroup := .operator 134495 145996
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 134495) (leftOrdinal := 0)
    (rightResult := 145996) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨29692⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29692⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge146002

namespace LeftMerge146123
def owner : Owner := ⟨.program ⟨257⟩, ⟨30420⟩⟩
def mergeEvent : Nat := 146123
def frameStart : Nat := 146057
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨29032⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events570.exact146119RawTerms
def rightRaw : List Term := Proof.Events570.exact146117RawTerms
def group : MergeGroup := .operator 146119 146117
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 146119) (leftOrdinal := 0)
    (rightResult := 146117) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨29032⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨29032⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge146123

namespace LeftMerge146135
def owner : Owner := ⟨.program ⟨257⟩, ⟨30789⟩⟩
def mergeEvent : Nat := 146135
def frameStart : Nat := 146057
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨30788⟩⟩] } }
def leftRaw : List Term := Proof.Events570.exact146131RawTerms
def rightRaw : List Term := Proof.Events570.exact146108RawTerms
def group : MergeGroup := .operator 146131 146108
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 146131) (leftOrdinal := 0)
    (rightResult := 146108) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7190⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨30788⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨30788⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge146135

namespace LeftMerge146136
def owner : Owner := ⟨.program ⟨257⟩, ⟨30789⟩⟩
def mergeEvent : Nat := 146136
def frameStart : Nat := 146057
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨29032⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨30788⟩⟩] } }
def leftRaw : List Term := Proof.Events570.exact146131RawTerms
def rightRaw : List Term := Proof.Events570.exact146108RawTerms
def group : MergeGroup := .operator 146131 146108
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 146131) (leftOrdinal := 1)
    (rightResult := 146108) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨29032⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨30788⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨29032⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨30788⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge146136

namespace LeftMerge146138
def owner : Owner := ⟨.program ⟨257⟩, ⟨30789⟩⟩
def mergeEvent : Nat := 146138
def frameStart : Nat := 146057
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨29032⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨30177⟩⟩] } }
def rhsRaw : List Term := Proof.Events570.exact146105RawTerms
def group : MergeGroup := .relation 146137
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 146137) (rhsResult := 146105)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨29032⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨30788⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨30788⟩⟩) ⟨30177⟩ 146105) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨30177⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨29032⟩⟩], [⟨.program ⟨257⟩, ⟨30177⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge146138

namespace LeftMerge146146
def owner : Owner := ⟨.program ⟨257⟩, ⟨29213⟩⟩
def mergeEvent : Nat := 146146
def frameStart : Nat := 146057
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨29211⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events570.exact146119RawTerms
def rightRaw : List Term := Proof.Events570.exact146142RawTerms
def group : MergeGroup := .operator 146119 146142
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 146119) (leftOrdinal := 0)
    (rightResult := 146142) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨29211⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨29211⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge146146

namespace LeftMerge146163
def owner : Owner := ⟨.program ⟨257⟩, ⟨29695⟩⟩
def mergeEvent : Nat := 146163
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7219⟩⟩] } }
def rhsRaw : List Term := Proof.Events570.exact146160RawTerms
def group : MergeGroup := .relation 146162
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 146162) (rhsResult := 146160)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29692⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 146161 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29692⟩⟩]⟩) (none) 146160) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7219⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7219⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge146163

namespace LeftMerge146164
def owner : Owner := ⟨.program ⟨257⟩, ⟨29695⟩⟩
def mergeEvent : Nat := 146164
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨30788⟩⟩] } }
def rhsRaw : List Term := Proof.Events570.exact146160RawTerms
def group : MergeGroup := .relation 146162
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 146162) (rhsResult := 146160)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29692⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 146161 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29692⟩⟩]⟩) (none) 146160) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨30788⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨30788⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge146164

namespace LeftMerge146165
def owner : Owner := ⟨.program ⟨257⟩, ⟨29695⟩⟩
def mergeEvent : Nat := 146165
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨29032⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨30177⟩⟩] } }
def rhsRaw : List Term := Proof.Events570.exact146160RawTerms
def group : MergeGroup := .relation 146162
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 146162) (rhsResult := 146160)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29692⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 146161 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29692⟩⟩]⟩) (none) 146160) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨29032⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨30177⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨29032⟩⟩], [⟨.program ⟨257⟩, ⟨30177⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge146165

namespace LeftMerge146166
def owner : Owner := ⟨.program ⟨257⟩, ⟨29695⟩⟩
def mergeEvent : Nat := 146166
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨29211⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events570.exact146160RawTerms
def group : MergeGroup := .relation 146162
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 146162) (rhsResult := 146160)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29692⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 146161 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29692⟩⟩]⟩) (none) 146160) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨29211⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨29211⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge146166

namespace LeftMerge146171
def owner : Owner := ⟨.program ⟨257⟩, ⟨30791⟩⟩
def mergeEvent : Nat := 146171
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨30788⟩⟩] } }
def leftRaw : List Term := Proof.Events570.exact146167RawTerms
def rightRaw : List Term := Proof.Events570.exact145989RawTerms
def group : MergeGroup := .operator 146167 145989
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 146167) (leftOrdinal := 0)
    (rightResult := 145989) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨30788⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨30788⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨30788⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge146171

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
