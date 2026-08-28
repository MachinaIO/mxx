import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge159298
def owner : Owner := ⟨.program ⟨257⟩, ⟨71145⟩⟩
def mergeEvent : Nat := 159298
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩] } }
def leftRaw : List Term := Proof.Events622.exact159274RawTerms
def rightRaw : List Term := Proof.Events616.exact157858RawTerms
def group : MergeGroup := .operator 159274 157858
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 159274) (leftOrdinal := 7)
    (rightResult := 157858) (rightOrdinal := 7) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge159298

namespace LeftMerge159299
def owner : Owner := ⟨.program ⟨257⟩, ⟨71145⟩⟩
def mergeEvent : Nat := 159299
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨60044⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68812⟩⟩] } }
def leftRaw : List Term := Proof.Events622.exact159274RawTerms
def rightRaw : List Term := Proof.Events616.exact157858RawTerms
def group : MergeGroup := .operator 159274 157858
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 159274) (leftOrdinal := 34)
    (rightResult := 157858) (rightOrdinal := 33) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨60044⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68812⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨60044⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68812⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨60044⟩⟩], [⟨.program ⟨257⟩, ⟨68812⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge159299

namespace LeftMerge159300
def owner : Owner := ⟨.program ⟨257⟩, ⟨71145⟩⟩
def mergeEvent : Nat := 159300
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩] } }
def leftRaw : List Term := Proof.Events622.exact159274RawTerms
def rightRaw : List Term := Proof.Events616.exact157858RawTerms
def group : MergeGroup := .operator 159274 157858
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 159274) (leftOrdinal := 6)
    (rightResult := 157858) (rightOrdinal := 6) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge159300

namespace LeftMerge159301
def owner : Owner := ⟨.program ⟨257⟩, ⟨71145⟩⟩
def mergeEvent : Nat := 159301
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨57064⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68812⟩⟩] } }
def leftRaw : List Term := Proof.Events622.exact159274RawTerms
def rightRaw : List Term := Proof.Events616.exact157858RawTerms
def group : MergeGroup := .operator 159274 157858
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 159274) (leftOrdinal := 33)
    (rightResult := 157858) (rightOrdinal := 32) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨57064⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68812⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨57064⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68812⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨57064⟩⟩], [⟨.program ⟨257⟩, ⟨68812⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge159301

namespace LeftMerge159302
def owner : Owner := ⟨.program ⟨257⟩, ⟨71145⟩⟩
def mergeEvent : Nat := 159302
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩] } }
def leftRaw : List Term := Proof.Events622.exact159274RawTerms
def rightRaw : List Term := Proof.Events616.exact157858RawTerms
def group : MergeGroup := .operator 159274 157858
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 159274) (leftOrdinal := 5)
    (rightResult := 157858) (rightOrdinal := 5) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge159302

namespace LeftMerge159303
def owner : Owner := ⟨.program ⟨257⟩, ⟨71145⟩⟩
def mergeEvent : Nat := 159303
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨54084⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68812⟩⟩] } }
def leftRaw : List Term := Proof.Events622.exact159274RawTerms
def rightRaw : List Term := Proof.Events616.exact157858RawTerms
def group : MergeGroup := .operator 159274 157858
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 159274) (leftOrdinal := 32)
    (rightResult := 157858) (rightOrdinal := 31) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨54084⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68812⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨54084⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68812⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨54084⟩⟩], [⟨.program ⟨257⟩, ⟨68812⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge159303

namespace LeftMerge159304
def owner : Owner := ⟨.program ⟨257⟩, ⟨71145⟩⟩
def mergeEvent : Nat := 159304
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩] } }
def leftRaw : List Term := Proof.Events622.exact159274RawTerms
def rightRaw : List Term := Proof.Events616.exact157858RawTerms
def group : MergeGroup := .operator 159274 157858
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 159274) (leftOrdinal := 4)
    (rightResult := 157858) (rightOrdinal := 4) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge159304

namespace LeftMerge159305
def owner : Owner := ⟨.program ⟨257⟩, ⟨71145⟩⟩
def mergeEvent : Nat := 159305
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨51104⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68812⟩⟩] } }
def leftRaw : List Term := Proof.Events622.exact159274RawTerms
def rightRaw : List Term := Proof.Events616.exact157858RawTerms
def group : MergeGroup := .operator 159274 157858
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 159274) (leftOrdinal := 31)
    (rightResult := 157858) (rightOrdinal := 30) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨51104⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68812⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨51104⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68812⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨51104⟩⟩], [⟨.program ⟨257⟩, ⟨68812⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge159305

namespace LeftMerge159306
def owner : Owner := ⟨.program ⟨257⟩, ⟨71145⟩⟩
def mergeEvent : Nat := 159306
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩] } }
def leftRaw : List Term := Proof.Events622.exact159274RawTerms
def rightRaw : List Term := Proof.Events616.exact157858RawTerms
def group : MergeGroup := .operator 159274 157858
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 159274) (leftOrdinal := 3)
    (rightResult := 157858) (rightOrdinal := 3) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge159306

namespace LeftMerge159307
def owner : Owner := ⟨.program ⟨257⟩, ⟨71145⟩⟩
def mergeEvent : Nat := 159307
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨32049⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68812⟩⟩] } }
def leftRaw : List Term := Proof.Events622.exact159274RawTerms
def rightRaw : List Term := Proof.Events616.exact157858RawTerms
def group : MergeGroup := .operator 159274 157858
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 159274) (leftOrdinal := 24)
    (rightResult := 157858) (rightOrdinal := 23) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨32049⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68812⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨32049⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68812⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨32049⟩⟩], [⟨.program ⟨257⟩, ⟨68812⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge159307

namespace LeftMerge159308
def owner : Owner := ⟨.program ⟨257⟩, ⟨71145⟩⟩
def mergeEvent : Nat := 159308
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7202⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩] } }
def leftRaw : List Term := Proof.Events622.exact159274RawTerms
def rightRaw : List Term := Proof.Events616.exact157858RawTerms
def group : MergeGroup := .operator 159274 157858
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 159274) (leftOrdinal := 2)
    (rightResult := 157858) (rightOrdinal := 2) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7202⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7202⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge159308

namespace LeftMerge159309
def owner : Owner := ⟨.program ⟨257⟩, ⟨71145⟩⟩
def mergeEvent : Nat := 159309
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨22029⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68812⟩⟩] } }
def leftRaw : List Term := Proof.Events622.exact159274RawTerms
def rightRaw : List Term := Proof.Events616.exact157858RawTerms
def group : MergeGroup := .operator 159274 157858
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 159274) (leftOrdinal := 21)
    (rightResult := 157858) (rightOrdinal := 20) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨22029⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68812⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨22029⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68812⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨22029⟩⟩], [⟨.program ⟨257⟩, ⟨68812⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge159309

namespace LeftMerge159310
def owner : Owner := ⟨.program ⟨257⟩, ⟨71145⟩⟩
def mergeEvent : Nat := 159310
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩] } }
def leftRaw : List Term := Proof.Events622.exact159274RawTerms
def rightRaw : List Term := Proof.Events616.exact157858RawTerms
def group : MergeGroup := .operator 159274 157858
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 159274) (leftOrdinal := 1)
    (rightResult := 157858) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge159310

namespace LeftMerge159311
def owner : Owner := ⟨.program ⟨257⟩, ⟨71145⟩⟩
def mergeEvent : Nat := 159311
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨18809⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68812⟩⟩] } }
def leftRaw : List Term := Proof.Events622.exact159274RawTerms
def rightRaw : List Term := Proof.Events616.exact157858RawTerms
def group : MergeGroup := .operator 159274 157858
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 159274) (leftOrdinal := 20)
    (rightResult := 157858) (rightOrdinal := 19) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨18809⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68812⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨18809⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68812⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨18809⟩⟩], [⟨.program ⟨257⟩, ⟨68812⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge159311

namespace LeftMerge159312
def owner : Owner := ⟨.program ⟨257⟩, ⟨71145⟩⟩
def mergeEvent : Nat := 159312
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩] } }
def leftRaw : List Term := Proof.Events622.exact159274RawTerms
def rightRaw : List Term := Proof.Events616.exact157858RawTerms
def group : MergeGroup := .operator 159274 157858
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 159274) (leftOrdinal := 0)
    (rightResult := 157858) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge159312

namespace LeftMerge159313
def owner : Owner := ⟨.program ⟨257⟩, ⟨71145⟩⟩
def mergeEvent : Nat := 159313
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨15987⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68812⟩⟩] } }
def leftRaw : List Term := Proof.Events622.exact159274RawTerms
def rightRaw : List Term := Proof.Events616.exact157858RawTerms
def group : MergeGroup := .operator 159274 157858
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 159274) (leftOrdinal := 19)
    (rightResult := 157858) (rightOrdinal := 18) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨15987⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68812⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨15987⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68812⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨15987⟩⟩], [⟨.program ⟨257⟩, ⟨68812⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge159313

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
