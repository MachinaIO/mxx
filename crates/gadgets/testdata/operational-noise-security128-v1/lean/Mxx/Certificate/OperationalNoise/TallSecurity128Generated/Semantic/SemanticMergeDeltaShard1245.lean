import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge203166
def owner : Owner := ⟨.program ⟨257⟩, ⟨71300⟩⟩
def mergeEvent : Nat := 203166
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨29325⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68842⟩⟩] } }
def leftRaw : List Term := Proof.Events793.exact203149RawTerms
def rightRaw : List Term := Proof.Events788.exact201733RawTerms
def group : MergeGroup := .operator 203149 201733
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 203149) (leftOrdinal := 23)
    (rightResult := 201733) (rightOrdinal := 22) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨29325⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68842⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨29325⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68842⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨29325⟩⟩], [⟨.program ⟨257⟩, ⟨68842⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge203166

namespace LeftMerge203167
def owner : Owner := ⟨.program ⟨257⟩, ⟨71300⟩⟩
def mergeEvent : Nat := 203167
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7218⟩⟩, ⟨.program ⟨257⟩, ⟨71297⟩⟩] } }
def leftRaw : List Term := Proof.Events793.exact203149RawTerms
def rightRaw : List Term := Proof.Events788.exact201733RawTerms
def group : MergeGroup := .operator 203149 201733
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 203149) (leftOrdinal := 10)
    (rightResult := 201733) (rightOrdinal := 10) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7218⟩⟩, ⟨.program ⟨257⟩, ⟨71297⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7218⟩⟩, ⟨.program ⟨257⟩, ⟨71297⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩, ⟨.program ⟨257⟩, ⟨71297⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge203167

namespace LeftMerge203168
def owner : Owner := ⟨.program ⟨257⟩, ⟨71300⟩⟩
def mergeEvent : Nat := 203168
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨26645⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68842⟩⟩] } }
def leftRaw : List Term := Proof.Events793.exact203149RawTerms
def rightRaw : List Term := Proof.Events788.exact201733RawTerms
def group : MergeGroup := .operator 203149 201733
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 203149) (leftOrdinal := 22)
    (rightResult := 201733) (rightOrdinal := 21) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨26645⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68842⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨26645⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68842⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨26645⟩⟩], [⟨.program ⟨257⟩, ⟨68842⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge203168

namespace LeftMerge203169
def owner : Owner := ⟨.program ⟨257⟩, ⟨71300⟩⟩
def mergeEvent : Nat := 203169
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7216⟩⟩, ⟨.program ⟨257⟩, ⟨71297⟩⟩] } }
def leftRaw : List Term := Proof.Events793.exact203149RawTerms
def rightRaw : List Term := Proof.Events788.exact201733RawTerms
def group : MergeGroup := .operator 203149 201733
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 203149) (leftOrdinal := 9)
    (rightResult := 201733) (rightOrdinal := 9) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7216⟩⟩, ⟨.program ⟨257⟩, ⟨71297⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7216⟩⟩, ⟨.program ⟨257⟩, ⟨71297⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩, ⟨.program ⟨257⟩, ⟨71297⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge203169

namespace LeftMerge203170
def owner : Owner := ⟨.program ⟨257⟩, ⟨71300⟩⟩
def mergeEvent : Nat := 203170
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨66741⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68842⟩⟩] } }
def leftRaw : List Term := Proof.Events793.exact203149RawTerms
def rightRaw : List Term := Proof.Events788.exact201733RawTerms
def group : MergeGroup := .operator 203149 201733
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 203149) (leftOrdinal := 36)
    (rightResult := 201733) (rightOrdinal := 35) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨66741⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68842⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨66741⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68842⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨66741⟩⟩], [⟨.program ⟨257⟩, ⟨68842⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge203170

namespace LeftMerge203171
def owner : Owner := ⟨.program ⟨257⟩, ⟨71300⟩⟩
def mergeEvent : Nat := 203171
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7214⟩⟩, ⟨.program ⟨257⟩, ⟨71297⟩⟩] } }
def leftRaw : List Term := Proof.Events793.exact203149RawTerms
def rightRaw : List Term := Proof.Events788.exact201733RawTerms
def group : MergeGroup := .operator 203149 201733
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 203149) (leftOrdinal := 8)
    (rightResult := 201733) (rightOrdinal := 8) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7214⟩⟩, ⟨.program ⟨257⟩, ⟨71297⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7214⟩⟩, ⟨.program ⟨257⟩, ⟨71297⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩, ⟨.program ⟨257⟩, ⟨71297⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge203171

namespace LeftMerge203172
def owner : Owner := ⟨.program ⟨257⟩, ⟨71300⟩⟩
def mergeEvent : Nat := 203172
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨63119⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68842⟩⟩] } }
def leftRaw : List Term := Proof.Events793.exact203149RawTerms
def rightRaw : List Term := Proof.Events788.exact201733RawTerms
def group : MergeGroup := .operator 203149 201733
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 203149) (leftOrdinal := 35)
    (rightResult := 201733) (rightOrdinal := 34) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨63119⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68842⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨63119⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68842⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨63119⟩⟩], [⟨.program ⟨257⟩, ⟨68842⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge203172

namespace LeftMerge203173
def owner : Owner := ⟨.program ⟨257⟩, ⟨71300⟩⟩
def mergeEvent : Nat := 203173
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨71297⟩⟩] } }
def leftRaw : List Term := Proof.Events793.exact203149RawTerms
def rightRaw : List Term := Proof.Events788.exact201733RawTerms
def group : MergeGroup := .operator 203149 201733
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 203149) (leftOrdinal := 7)
    (rightResult := 201733) (rightOrdinal := 7) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨71297⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨71297⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨71297⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge203173

namespace LeftMerge203174
def owner : Owner := ⟨.program ⟨257⟩, ⟨71300⟩⟩
def mergeEvent : Nat := 203174
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨60139⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68842⟩⟩] } }
def leftRaw : List Term := Proof.Events793.exact203149RawTerms
def rightRaw : List Term := Proof.Events788.exact201733RawTerms
def group : MergeGroup := .operator 203149 201733
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 203149) (leftOrdinal := 34)
    (rightResult := 201733) (rightOrdinal := 33) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨60139⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68842⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨60139⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68842⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨60139⟩⟩], [⟨.program ⟨257⟩, ⟨68842⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge203174

namespace LeftMerge203175
def owner : Owner := ⟨.program ⟨257⟩, ⟨71300⟩⟩
def mergeEvent : Nat := 203175
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨71297⟩⟩] } }
def leftRaw : List Term := Proof.Events793.exact203149RawTerms
def rightRaw : List Term := Proof.Events788.exact201733RawTerms
def group : MergeGroup := .operator 203149 201733
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 203149) (leftOrdinal := 6)
    (rightResult := 201733) (rightOrdinal := 6) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨71297⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨71297⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨71297⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge203175

namespace LeftMerge203176
def owner : Owner := ⟨.program ⟨257⟩, ⟨71300⟩⟩
def mergeEvent : Nat := 203176
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨57159⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68842⟩⟩] } }
def leftRaw : List Term := Proof.Events793.exact203149RawTerms
def rightRaw : List Term := Proof.Events788.exact201733RawTerms
def group : MergeGroup := .operator 203149 201733
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 203149) (leftOrdinal := 33)
    (rightResult := 201733) (rightOrdinal := 32) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨57159⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68842⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨57159⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68842⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨57159⟩⟩], [⟨.program ⟨257⟩, ⟨68842⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge203176

namespace LeftMerge203177
def owner : Owner := ⟨.program ⟨257⟩, ⟨71300⟩⟩
def mergeEvent : Nat := 203177
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨71297⟩⟩] } }
def leftRaw : List Term := Proof.Events793.exact203149RawTerms
def rightRaw : List Term := Proof.Events788.exact201733RawTerms
def group : MergeGroup := .operator 203149 201733
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 203149) (leftOrdinal := 5)
    (rightResult := 201733) (rightOrdinal := 5) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨71297⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨71297⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨71297⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge203177

namespace LeftMerge203178
def owner : Owner := ⟨.program ⟨257⟩, ⟨71300⟩⟩
def mergeEvent : Nat := 203178
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨54179⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68842⟩⟩] } }
def leftRaw : List Term := Proof.Events793.exact203149RawTerms
def rightRaw : List Term := Proof.Events788.exact201733RawTerms
def group : MergeGroup := .operator 203149 201733
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 203149) (leftOrdinal := 32)
    (rightResult := 201733) (rightOrdinal := 31) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨54179⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68842⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨54179⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68842⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨54179⟩⟩], [⟨.program ⟨257⟩, ⟨68842⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge203178

namespace LeftMerge203179
def owner : Owner := ⟨.program ⟨257⟩, ⟨71300⟩⟩
def mergeEvent : Nat := 203179
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨71297⟩⟩] } }
def leftRaw : List Term := Proof.Events793.exact203149RawTerms
def rightRaw : List Term := Proof.Events788.exact201733RawTerms
def group : MergeGroup := .operator 203149 201733
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 203149) (leftOrdinal := 4)
    (rightResult := 201733) (rightOrdinal := 4) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨71297⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨71297⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨71297⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge203179

namespace LeftMerge203180
def owner : Owner := ⟨.program ⟨257⟩, ⟨71300⟩⟩
def mergeEvent : Nat := 203180
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨51199⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68842⟩⟩] } }
def leftRaw : List Term := Proof.Events793.exact203149RawTerms
def rightRaw : List Term := Proof.Events788.exact201733RawTerms
def group : MergeGroup := .operator 203149 201733
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 203149) (leftOrdinal := 31)
    (rightResult := 201733) (rightOrdinal := 30) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨51199⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68842⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨51199⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68842⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨51199⟩⟩], [⟨.program ⟨257⟩, ⟨68842⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge203180

namespace LeftMerge203181
def owner : Owner := ⟨.program ⟨257⟩, ⟨71300⟩⟩
def mergeEvent : Nat := 203181
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨71297⟩⟩] } }
def leftRaw : List Term := Proof.Events793.exact203149RawTerms
def rightRaw : List Term := Proof.Events788.exact201733RawTerms
def group : MergeGroup := .operator 203149 201733
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 203149) (leftOrdinal := 3)
    (rightResult := 201733) (rightOrdinal := 3) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨71297⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨71297⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨71297⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge203181

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
