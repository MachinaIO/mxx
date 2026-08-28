import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge290898
def owner : Owner := ⟨.program ⟨257⟩, ⟨71051⟩⟩
def mergeEvent : Nat := 290898
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨21972⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68794⟩⟩] } }
def leftRaw : List Term := Proof.Events1136.exact290863RawTerms
def rightRaw : List Term := Proof.Events1130.exact289447RawTerms
def group : MergeGroup := .operator 290863 289447
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 290863) (leftOrdinal := 21)
    (rightResult := 289447) (rightOrdinal := 20) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨21972⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68794⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨21972⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68794⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨21972⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge290898

namespace LeftMerge290899
def owner : Owner := ⟨.program ⟨257⟩, ⟨71051⟩⟩
def mergeEvent : Nat := 290899
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩] } }
def leftRaw : List Term := Proof.Events1136.exact290863RawTerms
def rightRaw : List Term := Proof.Events1130.exact289447RawTerms
def group : MergeGroup := .operator 290863 289447
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 290863) (leftOrdinal := 1)
    (rightResult := 289447) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge290899

namespace LeftMerge290900
def owner : Owner := ⟨.program ⟨257⟩, ⟨71051⟩⟩
def mergeEvent : Nat := 290900
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨18752⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68794⟩⟩] } }
def leftRaw : List Term := Proof.Events1136.exact290863RawTerms
def rightRaw : List Term := Proof.Events1130.exact289447RawTerms
def group : MergeGroup := .operator 290863 289447
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 290863) (leftOrdinal := 20)
    (rightResult := 289447) (rightOrdinal := 19) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨18752⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68794⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨18752⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68794⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨18752⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge290900

namespace LeftMerge290901
def owner : Owner := ⟨.program ⟨257⟩, ⟨71051⟩⟩
def mergeEvent : Nat := 290901
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩] } }
def leftRaw : List Term := Proof.Events1136.exact290863RawTerms
def rightRaw : List Term := Proof.Events1130.exact289447RawTerms
def group : MergeGroup := .operator 290863 289447
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 290863) (leftOrdinal := 0)
    (rightResult := 289447) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge290901

namespace LeftMerge290902
def owner : Owner := ⟨.program ⟨257⟩, ⟨71051⟩⟩
def mergeEvent : Nat := 290902
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨15939⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68794⟩⟩] } }
def leftRaw : List Term := Proof.Events1136.exact290863RawTerms
def rightRaw : List Term := Proof.Events1130.exact289447RawTerms
def group : MergeGroup := .operator 290863 289447
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 290863) (leftOrdinal := 19)
    (rightResult := 289447) (rightOrdinal := 18) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨15939⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68794⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨15939⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68794⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨15939⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge290902

namespace LeftMerge290910
def owner : Owner := ⟨.program ⟨257⟩, ⟨71052⟩⟩
def mergeEvent : Nat := 290910
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7233⟩⟩, ⟨.program ⟨257⟩, ⟨7139⟩⟩] } }
def leftRaw : List Term := Proof.Events1136.exact290904RawTerms
def rightRaw : List Term := Proof.Events060.exact15522RawTerms
def group : MergeGroup := .operator 290904 15522
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 290904) (leftOrdinal := 0)
    (rightResult := 15522) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7233⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7139⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7233⟩⟩, ⟨.program ⟨257⟩, ⟨7139⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge290910

namespace LeftMerge290911
def owner : Owner := ⟨.program ⟨257⟩, ⟨71052⟩⟩
def mergeEvent : Nat := 290911
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨67341⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7139⟩⟩] } }
def leftRaw : List Term := Proof.Events1136.exact290904RawTerms
def rightRaw : List Term := Proof.Events060.exact15522RawTerms
def group : MergeGroup := .operator 290904 15522
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 290904) (leftOrdinal := 1)
    (rightResult := 15522) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨67341⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7139⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨67341⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7139⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge290911

namespace LeftMerge290913
def owner : Owner := ⟨.program ⟨257⟩, ⟨71052⟩⟩
def mergeEvent : Nat := 290913
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67341⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events060.exact15515RawTerms
def group : MergeGroup := .relation 290912
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 290912) (rhsResult := 15515)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨67341⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7139⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨7139⟩⟩) ⟨7035⟩ 15515) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6774⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67341⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge290913

namespace LeftMerge290927
def owner : Owner := ⟨.program ⟨257⟩, ⟨49875⟩⟩
def mergeEvent : Nat := 290927
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨49873⟩⟩] } }
def leftRaw : List Term := Proof.Events1097.exact280929RawTerms
def rightRaw : List Term := Proof.Events1136.exact290921RawTerms
def group : MergeGroup := .operator 280929 290921
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 280929) (leftOrdinal := 0)
    (rightResult := 290921) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7196⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨49873⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨49873⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge290927

namespace LeftMerge290928
def owner : Owner := ⟨.program ⟨257⟩, ⟨49875⟩⟩
def mergeEvent : Nat := 290928
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨48100⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨49873⟩⟩] } }
def leftRaw : List Term := Proof.Events1097.exact280929RawTerms
def rightRaw : List Term := Proof.Events1136.exact290921RawTerms
def group : MergeGroup := .operator 280929 290921
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 280929) (leftOrdinal := 1)
    (rightResult := 290921) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨48100⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨49873⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨48100⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨49873⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge290928

namespace LeftMerge290930
def owner : Owner := ⟨.program ⟨257⟩, ⟨49875⟩⟩
def mergeEvent : Nat := 290930
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨48100⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨49246⟩⟩] } }
def rhsRaw : List Term := Proof.Events1136.exact290918RawTerms
def group : MergeGroup := .relation 290929
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 290929) (rhsResult := 290918)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨48100⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨49873⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨49873⟩⟩) ⟨49246⟩ 290918) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨49246⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨48100⟩⟩], [⟨.program ⟨257⟩, ⟨49246⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge290930

namespace LeftMerge290944
def owner : Owner := ⟨.program ⟨257⟩, ⟨48775⟩⟩
def mergeEvent : Nat := 290944
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48772⟩⟩] } }
def leftRaw : List Term := Proof.Events1096.exact280745RawTerms
def rightRaw : List Term := Proof.Events1136.exact290938RawTerms
def group : MergeGroup := .operator 280745 290938
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 280745) (leftOrdinal := 0)
    (rightResult := 290938) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨48772⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48772⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge290944

namespace LeftMerge291065
def owner : Owner := ⟨.program ⟨257⟩, ⟨49484⟩⟩
def mergeEvent : Nat := 291065
def frameStart : Nat := 290999
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨48100⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1136.exact291061RawTerms
def rightRaw : List Term := Proof.Events1136.exact291059RawTerms
def group : MergeGroup := .operator 291061 291059
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 291061) (leftOrdinal := 0)
    (rightResult := 291059) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨48100⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨48100⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge291065

namespace LeftMerge291077
def owner : Owner := ⟨.program ⟨257⟩, ⟨49874⟩⟩
def mergeEvent : Nat := 291077
def frameStart : Nat := 290999
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨49873⟩⟩] } }
def leftRaw : List Term := Proof.Events1137.exact291073RawTerms
def rightRaw : List Term := Proof.Events1136.exact291050RawTerms
def group : MergeGroup := .operator 291073 291050
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 291073) (leftOrdinal := 0)
    (rightResult := 291050) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7196⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨49873⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨49873⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge291077

namespace LeftMerge291078
def owner : Owner := ⟨.program ⟨257⟩, ⟨49874⟩⟩
def mergeEvent : Nat := 291078
def frameStart : Nat := 290999
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨48100⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨49873⟩⟩] } }
def leftRaw : List Term := Proof.Events1137.exact291073RawTerms
def rightRaw : List Term := Proof.Events1136.exact291050RawTerms
def group : MergeGroup := .operator 291073 291050
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 291073) (leftOrdinal := 1)
    (rightResult := 291050) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨48100⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨49873⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨48100⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨49873⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge291078

namespace LeftMerge291080
def owner : Owner := ⟨.program ⟨257⟩, ⟨49874⟩⟩
def mergeEvent : Nat := 291080
def frameStart : Nat := 290999
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨48100⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨49246⟩⟩] } }
def rhsRaw : List Term := Proof.Events1136.exact291047RawTerms
def group : MergeGroup := .relation 291079
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 291079) (rhsResult := 291047)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨48100⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨49873⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨49873⟩⟩) ⟨49246⟩ 291047) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨49246⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨48100⟩⟩], [⟨.program ⟨257⟩, ⟨49246⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge291080

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
