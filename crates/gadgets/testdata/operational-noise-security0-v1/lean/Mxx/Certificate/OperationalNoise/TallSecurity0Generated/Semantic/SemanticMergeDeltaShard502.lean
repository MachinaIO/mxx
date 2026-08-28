import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge81441
def owner : Owner := ⟨.program ⟨214⟩, ⟨25528⟩⟩
def mergeEvent : Nat := 81441
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨10030⟩⟩, ⟨.program ⟨214⟩, ⟨12762⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25527⟩⟩] } }
def leftRaw : List Term := Proof.Events318.exact81435RawTerms
def rightRaw : List Term := Proof.Events317.exact81371RawTerms
def group : MergeGroup := .operator 81435 81371
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 81435) (leftOrdinal := 1)
    (rightResult := 81371) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨10030⟩⟩, ⟨.program ⟨214⟩, ⟨12762⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25527⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨10030⟩⟩, ⟨.program ⟨214⟩, ⟨12762⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25527⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge81441

namespace LeftMerge81443
def owner : Owner := ⟨.program ⟨214⟩, ⟨25528⟩⟩
def mergeEvent : Nat := 81443
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨10030⟩⟩, ⟨.program ⟨214⟩, ⟨12762⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23290⟩⟩] } }
def rhsRaw : List Term := Proof.Events317.exact81368RawTerms
def group : MergeGroup := .relation 81442
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 81442) (rhsResult := 81368)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨10030⟩⟩, ⟨.program ⟨214⟩, ⟨12762⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25527⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨25527⟩⟩) ⟨23290⟩ 81368) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23290⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨10030⟩⟩, ⟨.program ⟨214⟩, ⟨12762⟩⟩], [⟨.program ⟨214⟩, ⟨23290⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge81443

namespace LeftMerge81444
def owner : Owner := ⟨.program ⟨214⟩, ⟨25528⟩⟩
def mergeEvent : Nat := 81444
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6767⟩⟩, ⟨.program ⟨214⟩, ⟨7873⟩⟩, ⟨.program ⟨214⟩, ⟨25527⟩⟩] } }
def leftRaw : List Term := Proof.Events318.exact81435RawTerms
def rightRaw : List Term := Proof.Events317.exact81371RawTerms
def group : MergeGroup := .operator 81435 81371
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 81435) (leftOrdinal := 0)
    (rightResult := 81371) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6767⟩⟩, ⟨.program ⟨214⟩, ⟨7873⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25527⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6767⟩⟩, ⟨.program ⟨214⟩, ⟨7873⟩⟩, ⟨.program ⟨214⟩, ⟨25527⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge81444

namespace LeftMerge81458
def owner : Owner := ⟨.program ⟨214⟩, ⟨20035⟩⟩
def mergeEvent : Nat := 81458
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20032⟩⟩] } }
def leftRaw : List Term := Proof.Events312.exact80012RawTerms
def rightRaw : List Term := Proof.Events318.exact81452RawTerms
def group : MergeGroup := .operator 80012 81452
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 80012) (leftOrdinal := 0)
    (rightResult := 81452) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨20032⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20032⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge81458

namespace LeftMerge81537
def owner : Owner := ⟨.program ⟨214⟩, ⟨12763⟩⟩
def mergeEvent : Nat := 81537
def frameStart : Nat := 81507
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨10030⟩⟩, ⟨.program ⟨214⟩, ⟨12762⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events318.exact81533RawTerms
def rightRaw : List Term := Proof.Events318.exact81530RawTerms
def group : MergeGroup := .operator 81533 81530
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 81533) (leftOrdinal := 0)
    (rightResult := 81530) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10030⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨12762⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨10030⟩⟩, ⟨.program ⟨214⟩, ⟨12762⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge81537

namespace LeftMerge81567
def owner : Owner := ⟨.program ⟨214⟩, ⟨12860⟩⟩
def mergeEvent : Nat := 81567
def frameStart : Nat := 81507
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨10030⟩⟩, ⟨.program ⟨214⟩, ⟨12762⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events318.exact81563RawTerms
def rightRaw : List Term := Proof.Events318.exact81561RawTerms
def group : MergeGroup := .operator 81563 81561
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 81563) (leftOrdinal := 0)
    (rightResult := 81561) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10030⟩⟩, ⟨.program ⟨214⟩, ⟨12762⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨10030⟩⟩, ⟨.program ⟨214⟩, ⟨12762⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge81567

namespace LeftMerge81588
def owner : Owner := ⟨.program ⟨214⟩, ⟨7875⟩⟩
def mergeEvent : Nat := 81588
def frameStart : Nat := 81507
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6767⟩⟩, ⟨.program ⟨214⟩, ⟨7873⟩⟩] } }
def leftRaw : List Term := Proof.Events318.exact81584RawTerms
def rightRaw : List Term := Proof.Events318.exact81581RawTerms
def group : MergeGroup := .operator 81584 81581
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 81584) (leftOrdinal := 0)
    (rightResult := 81581) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6767⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨7873⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6767⟩⟩, ⟨.program ⟨214⟩, ⟨7873⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge81588

namespace LeftMerge81597
def owner : Owner := ⟨.program ⟨214⟩, ⟨25530⟩⟩
def mergeEvent : Nat := 81597
def frameStart : Nat := 81507
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6767⟩⟩, ⟨.program ⟨214⟩, ⟨7873⟩⟩, ⟨.program ⟨214⟩, ⟨25527⟩⟩] } }
def leftRaw : List Term := Proof.Events318.exact81593RawTerms
def rightRaw : List Term := Proof.Events318.exact81552RawTerms
def group : MergeGroup := .operator 81593 81552
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 81593) (leftOrdinal := 0)
    (rightResult := 81552) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6767⟩⟩, ⟨.program ⟨214⟩, ⟨7873⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25527⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6767⟩⟩, ⟨.program ⟨214⟩, ⟨7873⟩⟩, ⟨.program ⟨214⟩, ⟨25527⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge81597

namespace LeftMerge81598
def owner : Owner := ⟨.program ⟨214⟩, ⟨25530⟩⟩
def mergeEvent : Nat := 81598
def frameStart : Nat := 81507
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨10030⟩⟩, ⟨.program ⟨214⟩, ⟨12762⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25527⟩⟩] } }
def leftRaw : List Term := Proof.Events318.exact81593RawTerms
def rightRaw : List Term := Proof.Events318.exact81552RawTerms
def group : MergeGroup := .operator 81593 81552
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 81593) (leftOrdinal := 1)
    (rightResult := 81552) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10030⟩⟩, ⟨.program ⟨214⟩, ⟨12762⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25527⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨10030⟩⟩, ⟨.program ⟨214⟩, ⟨12762⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25527⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge81598

namespace LeftMerge81600
def owner : Owner := ⟨.program ⟨214⟩, ⟨25530⟩⟩
def mergeEvent : Nat := 81600
def frameStart : Nat := 81507
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨10030⟩⟩, ⟨.program ⟨214⟩, ⟨12762⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23290⟩⟩] } }
def rhsRaw : List Term := Proof.Events318.exact81549RawTerms
def group : MergeGroup := .relation 81599
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 81599) (rhsResult := 81549)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨10030⟩⟩, ⟨.program ⟨214⟩, ⟨12762⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25527⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨25527⟩⟩) ⟨23290⟩ 81549) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23290⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨10030⟩⟩, ⟨.program ⟨214⟩, ⟨12762⟩⟩], [⟨.program ⟨214⟩, ⟨23290⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge81600

namespace LeftMerge81608
def owner : Owner := ⟨.program ⟨214⟩, ⟨16635⟩⟩
def mergeEvent : Nat := 81608
def frameStart : Nat := 81507
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16633⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events318.exact81563RawTerms
def rightRaw : List Term := Proof.Events318.exact81604RawTerms
def group : MergeGroup := .operator 81563 81604
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 81563) (leftOrdinal := 0)
    (rightResult := 81604) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16633⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨16633⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge81608

namespace LeftMerge81625
def owner : Owner := ⟨.program ⟨214⟩, ⟨20035⟩⟩
def mergeEvent : Nat := 81625
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6704⟩⟩] } }
def rhsRaw : List Term := Proof.Events318.exact81622RawTerms
def group : MergeGroup := .relation 81624
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 81624) (rhsResult := 81622)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20032⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 81623 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20032⟩⟩]⟩) (none) 81622) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6704⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6704⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge81625

namespace LeftMerge81626
def owner : Owner := ⟨.program ⟨214⟩, ⟨20035⟩⟩
def mergeEvent : Nat := 81626
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6767⟩⟩, ⟨.program ⟨214⟩, ⟨7873⟩⟩, ⟨.program ⟨214⟩, ⟨25527⟩⟩] } }
def rhsRaw : List Term := Proof.Events318.exact81622RawTerms
def group : MergeGroup := .relation 81624
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 81624) (rhsResult := 81622)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20032⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 81623 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20032⟩⟩]⟩) (none) 81622) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6767⟩⟩, ⟨.program ⟨214⟩, ⟨7873⟩⟩, ⟨.program ⟨214⟩, ⟨25527⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6767⟩⟩, ⟨.program ⟨214⟩, ⟨7873⟩⟩, ⟨.program ⟨214⟩, ⟨25527⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge81626

namespace LeftMerge81627
def owner : Owner := ⟨.program ⟨214⟩, ⟨20035⟩⟩
def mergeEvent : Nat := 81627
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨10030⟩⟩, ⟨.program ⟨214⟩, ⟨12762⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23290⟩⟩] } }
def rhsRaw : List Term := Proof.Events318.exact81622RawTerms
def group : MergeGroup := .relation 81624
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 81624) (rhsResult := 81622)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20032⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 81623 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20032⟩⟩]⟩) (none) 81622) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10030⟩⟩, ⟨.program ⟨214⟩, ⟨12762⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23290⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨10030⟩⟩, ⟨.program ⟨214⟩, ⟨12762⟩⟩], [⟨.program ⟨214⟩, ⟨23290⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge81627

namespace LeftMerge81628
def owner : Owner := ⟨.program ⟨214⟩, ⟨20035⟩⟩
def mergeEvent : Nat := 81628
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16633⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def rhsRaw : List Term := Proof.Events318.exact81622RawTerms
def group : MergeGroup := .relation 81624
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 81624) (rhsResult := 81622)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20032⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 81623 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20032⟩⟩]⟩) (none) 81622) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16633⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16633⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge81628

namespace LeftMerge81633
def owner : Owner := ⟨.program ⟨214⟩, ⟨25529⟩⟩
def mergeEvent : Nat := 81633
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨10030⟩⟩, ⟨.program ⟨214⟩, ⟨12762⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23290⟩⟩] } }
def leftRaw : List Term := Proof.Events318.exact81629RawTerms
def rightRaw : List Term := Proof.Events318.exact81445RawTerms
def group : MergeGroup := .operator 81629 81445
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 81629) (leftOrdinal := 2)
    (rightResult := 81445) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨10030⟩⟩, ⟨.program ⟨214⟩, ⟨12762⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23290⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨10030⟩⟩, ⟨.program ⟨214⟩, ⟨12762⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23290⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨10030⟩⟩, ⟨.program ⟨214⟩, ⟨12762⟩⟩], [⟨.program ⟨214⟩, ⟨23290⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge81633

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
