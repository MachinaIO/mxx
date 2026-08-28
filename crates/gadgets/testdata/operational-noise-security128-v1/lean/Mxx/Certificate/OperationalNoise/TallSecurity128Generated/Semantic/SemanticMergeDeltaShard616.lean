import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge103144
def owner : Owner := ⟨.program ⟨257⟩, ⟨62043⟩⟩
def mergeEvent : Nat := 103144
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7186⟩⟩, ⟨.program ⟨257⟩, ⟨62040⟩⟩] } }
def leftRaw : List Term := Proof.Events402.exact103140RawTerms
def rightRaw : List Term := Proof.Events402.exact102962RawTerms
def group : MergeGroup := .operator 103140 102962
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 103140) (leftOrdinal := 0)
    (rightResult := 102962) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7186⟩⟩, ⟨.program ⟨257⟩, ⟨62040⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7186⟩⟩, ⟨.program ⟨257⟩, ⟨62040⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7186⟩⟩, ⟨.program ⟨257⟩, ⟨62040⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge103144

namespace LeftMerge103145
def owner : Owner := ⟨.program ⟨257⟩, ⟨62043⟩⟩
def mergeEvent : Nat := 103145
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨59868⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨61145⟩⟩] } }
def leftRaw : List Term := Proof.Events402.exact103140RawTerms
def rightRaw : List Term := Proof.Events402.exact102962RawTerms
def group : MergeGroup := .operator 103140 102962
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 103140) (leftOrdinal := 2)
    (rightResult := 102962) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨59868⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨61145⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨59868⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨61145⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨59868⟩⟩], [⟨.program ⟨257⟩, ⟨61145⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge103145

namespace LeftMerge103153
def owner : Owner := ⟨.program ⟨257⟩, ⟨62044⟩⟩
def mergeEvent : Nat := 103153
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7211⟩⟩, ⟨.program ⟨257⟩, ⟨7103⟩⟩] } }
def leftRaw : List Term := Proof.Events402.exact103147RawTerms
def rightRaw : List Term := Proof.Events061.exact15742RawTerms
def group : MergeGroup := .operator 103147 15742
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 103147) (leftOrdinal := 0)
    (rightResult := 15742) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7211⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7103⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7211⟩⟩, ⟨.program ⟨257⟩, ⟨7103⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge103153

namespace LeftMerge103154
def owner : Owner := ⟨.program ⟨257⟩, ⟨62044⟩⟩
def mergeEvent : Nat := 103154
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨60200⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7103⟩⟩] } }
def leftRaw : List Term := Proof.Events402.exact103147RawTerms
def rightRaw : List Term := Proof.Events061.exact15742RawTerms
def group : MergeGroup := .operator 103147 15742
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 103147) (leftOrdinal := 1)
    (rightResult := 15742) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨60200⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7103⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨60200⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7103⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge103154

namespace LeftMerge103156
def owner : Owner := ⟨.program ⟨257⟩, ⟨62044⟩⟩
def mergeEvent : Nat := 103156
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨60200⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events061.exact15735RawTerms
def group : MergeGroup := .relation 103155
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 103155) (rhsResult := 15735)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨60200⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7103⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨7103⟩⟩) ⟨7017⟩ 15735) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6736⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨60200⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge103156

namespace LeftMerge103170
def owner : Owner := ⟨.program ⟨257⟩, ⟨59062⟩⟩
def mergeEvent : Nat := 103170
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7185⟩⟩, ⟨.program ⟨257⟩, ⟨59060⟩⟩] } }
def leftRaw : List Term := Proof.Events375.exact96108RawTerms
def rightRaw : List Term := Proof.Events402.exact103164RawTerms
def group : MergeGroup := .operator 96108 103164
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 96108) (leftOrdinal := 0)
    (rightResult := 103164) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7185⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨59060⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7185⟩⟩, ⟨.program ⟨257⟩, ⟨59060⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge103170

namespace LeftMerge103171
def owner : Owner := ⟨.program ⟨257⟩, ⟨59062⟩⟩
def mergeEvent : Nat := 103171
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨56888⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨59060⟩⟩] } }
def leftRaw : List Term := Proof.Events375.exact96108RawTerms
def rightRaw : List Term := Proof.Events402.exact103164RawTerms
def group : MergeGroup := .operator 96108 103164
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 96108) (leftOrdinal := 1)
    (rightResult := 103164) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨56888⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨59060⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨56888⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨59060⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge103171

namespace LeftMerge103173
def owner : Owner := ⟨.program ⟨257⟩, ⟨59062⟩⟩
def mergeEvent : Nat := 103173
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨56888⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨58165⟩⟩] } }
def rhsRaw : List Term := Proof.Events402.exact103161RawTerms
def group : MergeGroup := .relation 103172
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 103172) (rhsResult := 103161)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨56888⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨59060⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨59060⟩⟩) ⟨58165⟩ 103161) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨58165⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨56888⟩⟩], [⟨.program ⟨257⟩, ⟨58165⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge103173

namespace LeftMerge103187
def owner : Owner := ⟨.program ⟨257⟩, ⟨57815⟩⟩
def mergeEvent : Nat := 103187
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57812⟩⟩] } }
def leftRaw : List Term := Proof.Events353.exact90620RawTerms
def rightRaw : List Term := Proof.Events403.exact103181RawTerms
def group : MergeGroup := .operator 90620 103181
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 90620) (leftOrdinal := 0)
    (rightResult := 103181) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨57812⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57812⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge103187

namespace LeftMerge103308
def owner : Owner := ⟨.program ⟨257⟩, ⟨58348⟩⟩
def mergeEvent : Nat := 103308
def frameStart : Nat := 103242
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨56888⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events403.exact103304RawTerms
def rightRaw : List Term := Proof.Events403.exact103302RawTerms
def group : MergeGroup := .operator 103304 103302
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 103304) (leftOrdinal := 0)
    (rightResult := 103302) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨56888⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨56888⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge103308

namespace LeftMerge103320
def owner : Owner := ⟨.program ⟨257⟩, ⟨59061⟩⟩
def mergeEvent : Nat := 103320
def frameStart : Nat := 103242
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7185⟩⟩, ⟨.program ⟨257⟩, ⟨59060⟩⟩] } }
def leftRaw : List Term := Proof.Events403.exact103316RawTerms
def rightRaw : List Term := Proof.Events403.exact103293RawTerms
def group : MergeGroup := .operator 103316 103293
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 103316) (leftOrdinal := 0)
    (rightResult := 103293) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7185⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨59060⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7185⟩⟩, ⟨.program ⟨257⟩, ⟨59060⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge103320

namespace LeftMerge103321
def owner : Owner := ⟨.program ⟨257⟩, ⟨59061⟩⟩
def mergeEvent : Nat := 103321
def frameStart : Nat := 103242
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨56888⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨59060⟩⟩] } }
def leftRaw : List Term := Proof.Events403.exact103316RawTerms
def rightRaw : List Term := Proof.Events403.exact103293RawTerms
def group : MergeGroup := .operator 103316 103293
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 103316) (leftOrdinal := 1)
    (rightResult := 103293) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨56888⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨59060⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨56888⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨59060⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge103321

namespace LeftMerge103323
def owner : Owner := ⟨.program ⟨257⟩, ⟨59061⟩⟩
def mergeEvent : Nat := 103323
def frameStart : Nat := 103242
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨56888⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨58165⟩⟩] } }
def rhsRaw : List Term := Proof.Events403.exact103290RawTerms
def group : MergeGroup := .relation 103322
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 103322) (rhsResult := 103290)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨56888⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨59060⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨59060⟩⟩) ⟨58165⟩ 103290) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨58165⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨56888⟩⟩], [⟨.program ⟨257⟩, ⟨58165⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge103323

namespace LeftMerge103331
def owner : Owner := ⟨.program ⟨257⟩, ⟨57223⟩⟩
def mergeEvent : Nat := 103331
def frameStart : Nat := 103242
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨57220⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events403.exact103304RawTerms
def rightRaw : List Term := Proof.Events403.exact103327RawTerms
def group : MergeGroup := .operator 103304 103327
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 103304) (leftOrdinal := 0)
    (rightResult := 103327) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨57220⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨57220⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge103331

namespace LeftMerge103348
def owner : Owner := ⟨.program ⟨257⟩, ⟨57815⟩⟩
def mergeEvent : Nat := 103348
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7209⟩⟩] } }
def rhsRaw : List Term := Proof.Events403.exact103345RawTerms
def group : MergeGroup := .relation 103347
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 103347) (rhsResult := 103345)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57812⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 103346 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57812⟩⟩]⟩) (none) 103345) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7209⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7209⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge103348

namespace LeftMerge103349
def owner : Owner := ⟨.program ⟨257⟩, ⟨57815⟩⟩
def mergeEvent : Nat := 103349
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7185⟩⟩, ⟨.program ⟨257⟩, ⟨59060⟩⟩] } }
def rhsRaw : List Term := Proof.Events403.exact103345RawTerms
def group : MergeGroup := .relation 103347
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 103347) (rhsResult := 103345)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57812⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 103346 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57812⟩⟩]⟩) (none) 103345) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7185⟩⟩, ⟨.program ⟨257⟩, ⟨59060⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7185⟩⟩, ⟨.program ⟨257⟩, ⟨59060⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge103349

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
