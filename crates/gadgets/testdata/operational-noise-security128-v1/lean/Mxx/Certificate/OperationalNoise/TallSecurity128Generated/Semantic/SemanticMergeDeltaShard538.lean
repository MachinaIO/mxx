import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge90281
def owner : Owner := ⟨.program ⟨257⟩, ⟨71446⟩⟩
def mergeEvent : Nat := 90281
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨35037⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7243⟩⟩] } }
def leftRaw : List Term := Proof.Events352.exact90271RawTerms
def rightRaw : List Term := Proof.Events296.exact75868RawTerms
def group : MergeGroup := .operator 90271 75868
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 90271) (leftOrdinal := 14)
    (rightResult := 75868) (rightOrdinal := 32) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨35037⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7243⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨35037⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7243⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨35037⟩⟩], [⟨.program ⟨257⟩, ⟨7243⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge90281

namespace LeftMerge90282
def owner : Owner := ⟨.program ⟨257⟩, ⟨71446⟩⟩
def mergeEvent : Nat := 90282
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨29380⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7243⟩⟩] } }
def leftRaw : List Term := Proof.Events352.exact90271RawTerms
def rightRaw : List Term := Proof.Events296.exact75868RawTerms
def group : MergeGroup := .operator 90271 75868
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 90271) (leftOrdinal := 16)
    (rightResult := 75868) (rightOrdinal := 34) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨29380⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7243⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨29380⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7243⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨29380⟩⟩], [⟨.program ⟨257⟩, ⟨7243⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge90282

namespace LeftMerge90283
def owner : Owner := ⟨.program ⟨257⟩, ⟨71446⟩⟩
def mergeEvent : Nat := 90283
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨26700⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7243⟩⟩] } }
def leftRaw : List Term := Proof.Events352.exact90271RawTerms
def rightRaw : List Term := Proof.Events296.exact75868RawTerms
def group : MergeGroup := .operator 90271 75868
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 90271) (leftOrdinal := 17)
    (rightResult := 75868) (rightOrdinal := 35) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨26700⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7243⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨26700⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7243⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨26700⟩⟩], [⟨.program ⟨257⟩, ⟨7243⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge90283

namespace LeftMerge90284
def owner : Owner := ⟨.program ⟨257⟩, ⟨71446⟩⟩
def mergeEvent : Nat := 90284
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨67008⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7243⟩⟩] } }
def leftRaw : List Term := Proof.Events352.exact90271RawTerms
def rightRaw : List Term := Proof.Events296.exact75868RawTerms
def group : MergeGroup := .operator 90271 75868
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 90271) (leftOrdinal := 19)
    (rightResult := 75868) (rightOrdinal := 37) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨67008⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7243⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨67008⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7243⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨67008⟩⟩], [⟨.program ⟨257⟩, ⟨7243⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge90284

namespace LeftMerge90285
def owner : Owner := ⟨.program ⟨257⟩, ⟨71446⟩⟩
def mergeEvent : Nat := 90285
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨63199⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7243⟩⟩] } }
def leftRaw : List Term := Proof.Events352.exact90271RawTerms
def rightRaw : List Term := Proof.Events296.exact75868RawTerms
def group : MergeGroup := .operator 90271 75868
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 90271) (leftOrdinal := 1)
    (rightResult := 75868) (rightOrdinal := 19) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨63199⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7243⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨63199⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7243⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨63199⟩⟩], [⟨.program ⟨257⟩, ⟨7243⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge90285

namespace LeftMerge90286
def owner : Owner := ⟨.program ⟨257⟩, ⟨71446⟩⟩
def mergeEvent : Nat := 90286
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨60219⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7243⟩⟩] } }
def leftRaw : List Term := Proof.Events352.exact90271RawTerms
def rightRaw : List Term := Proof.Events296.exact75868RawTerms
def group : MergeGroup := .operator 90271 75868
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 90271) (leftOrdinal := 2)
    (rightResult := 75868) (rightOrdinal := 20) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨60219⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7243⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨60219⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7243⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨60219⟩⟩], [⟨.program ⟨257⟩, ⟨7243⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge90286

namespace LeftMerge90287
def owner : Owner := ⟨.program ⟨257⟩, ⟨71446⟩⟩
def mergeEvent : Nat := 90287
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨57239⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7243⟩⟩] } }
def leftRaw : List Term := Proof.Events352.exact90271RawTerms
def rightRaw : List Term := Proof.Events296.exact75868RawTerms
def group : MergeGroup := .operator 90271 75868
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 90271) (leftOrdinal := 3)
    (rightResult := 75868) (rightOrdinal := 21) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨57239⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7243⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨57239⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7243⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨57239⟩⟩], [⟨.program ⟨257⟩, ⟨7243⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge90287

namespace LeftMerge90288
def owner : Owner := ⟨.program ⟨257⟩, ⟨71446⟩⟩
def mergeEvent : Nat := 90288
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨54259⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7243⟩⟩] } }
def leftRaw : List Term := Proof.Events352.exact90271RawTerms
def rightRaw : List Term := Proof.Events296.exact75868RawTerms
def group : MergeGroup := .operator 90271 75868
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 90271) (leftOrdinal := 4)
    (rightResult := 75868) (rightOrdinal := 22) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨54259⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7243⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨54259⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7243⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨54259⟩⟩], [⟨.program ⟨257⟩, ⟨7243⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge90288

namespace LeftMerge90289
def owner : Owner := ⟨.program ⟨257⟩, ⟨71446⟩⟩
def mergeEvent : Nat := 90289
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨51279⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7243⟩⟩] } }
def leftRaw : List Term := Proof.Events352.exact90271RawTerms
def rightRaw : List Term := Proof.Events296.exact75868RawTerms
def group : MergeGroup := .operator 90271 75868
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 90271) (leftOrdinal := 5)
    (rightResult := 75868) (rightOrdinal := 23) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨51279⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7243⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨51279⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7243⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨51279⟩⟩], [⟨.program ⟨257⟩, ⟨7243⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge90289

namespace LeftMerge90290
def owner : Owner := ⟨.program ⟨257⟩, ⟨71446⟩⟩
def mergeEvent : Nat := 90290
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨32215⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7243⟩⟩] } }
def leftRaw : List Term := Proof.Events352.exact90271RawTerms
def rightRaw : List Term := Proof.Events296.exact75868RawTerms
def group : MergeGroup := .operator 90271 75868
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 90271) (leftOrdinal := 7)
    (rightResult := 75868) (rightOrdinal := 25) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨32215⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7243⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨32215⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7243⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨32215⟩⟩], [⟨.program ⟨257⟩, ⟨7243⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge90290

namespace LeftMerge90291
def owner : Owner := ⟨.program ⟨257⟩, ⟨71446⟩⟩
def mergeEvent : Nat := 90291
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨22195⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7243⟩⟩] } }
def leftRaw : List Term := Proof.Events352.exact90271RawTerms
def rightRaw : List Term := Proof.Events296.exact75868RawTerms
def group : MergeGroup := .operator 90271 75868
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 90271) (leftOrdinal := 11)
    (rightResult := 75868) (rightOrdinal := 29) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨22195⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7243⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨22195⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7243⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨22195⟩⟩], [⟨.program ⟨257⟩, ⟨7243⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge90291

namespace LeftMerge90292
def owner : Owner := ⟨.program ⟨257⟩, ⟨71446⟩⟩
def mergeEvent : Nat := 90292
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨18975⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7243⟩⟩] } }
def leftRaw : List Term := Proof.Events352.exact90271RawTerms
def rightRaw : List Term := Proof.Events296.exact75868RawTerms
def group : MergeGroup := .operator 90271 75868
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 90271) (leftOrdinal := 15)
    (rightResult := 75868) (rightOrdinal := 33) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨18975⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7243⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨18975⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7243⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨18975⟩⟩], [⟨.program ⟨257⟩, ⟨7243⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge90292

namespace LeftMerge90293
def owner : Owner := ⟨.program ⟨257⟩, ⟨71446⟩⟩
def mergeEvent : Nat := 90293
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨16126⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7243⟩⟩] } }
def leftRaw : List Term := Proof.Events352.exact90271RawTerms
def rightRaw : List Term := Proof.Events296.exact75868RawTerms
def group : MergeGroup := .operator 90271 75868
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 90271) (leftOrdinal := 18)
    (rightResult := 75868) (rightOrdinal := 36) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨16126⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7243⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨16126⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7243⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨16126⟩⟩], [⟨.program ⟨257⟩, ⟨7243⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge90293

namespace LeftMerge90301
def owner : Owner := ⟨.program ⟨257⟩, ⟨71447⟩⟩
def mergeEvent : Nat := 90301
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67586⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7101⟩⟩] } }
def leftRaw : List Term := Proof.Events352.exact90295RawTerms
def rightRaw : List Term := Proof.Events063.exact16134RawTerms
def group : MergeGroup := .operator 90295 16134
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 90295) (leftOrdinal := 6)
    (rightResult := 16134) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67586⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7101⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67586⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7101⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge90301

namespace LeftMerge90303
def owner : Owner := ⟨.program ⟨257⟩, ⟨71447⟩⟩
def mergeEvent : Nat := 90303
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6733⟩⟩, ⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67586⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events062.exact16127RawTerms
def group : MergeGroup := .relation 90302
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 90302) (rhsResult := 16127)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67586⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7101⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨7101⟩⟩) ⟨7016⟩ 16127) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6733⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6733⟩⟩, ⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67586⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge90303

namespace LeftMerge90304
def owner : Owner := ⟨.program ⟨257⟩, ⟨71447⟩⟩
def mergeEvent : Nat := 90304
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48437⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7101⟩⟩] } }
def leftRaw : List Term := Proof.Events352.exact90295RawTerms
def rightRaw : List Term := Proof.Events063.exact16134RawTerms
def group : MergeGroup := .operator 90295 16134
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 90295) (leftOrdinal := 8)
    (rightResult := 16134) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48437⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7101⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48437⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7101⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge90304

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
