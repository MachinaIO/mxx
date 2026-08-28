import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge221901
def owner : Owner := ⟨.program ⟨257⟩, ⟨71245⟩⟩
def mergeEvent : Nat := 221901
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48359⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7261⟩⟩] } }
def leftRaw : List Term := Proof.Events866.exact221896RawTerms
def rightRaw : List Term := Proof.Events810.exact207493RawTerms
def group : MergeGroup := .operator 221896 207493
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 221896) (leftOrdinal := 8)
    (rightResult := 207493) (rightOrdinal := 26) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48359⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7261⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48359⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7261⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48359⟩⟩], [⟨.program ⟨257⟩, ⟨7261⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge221901

namespace LeftMerge221902
def owner : Owner := ⟨.program ⟨257⟩, ⟨71245⟩⟩
def mergeEvent : Nat := 221902
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45679⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7261⟩⟩] } }
def leftRaw : List Term := Proof.Events866.exact221896RawTerms
def rightRaw : List Term := Proof.Events810.exact207493RawTerms
def group : MergeGroup := .operator 221896 207493
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 221896) (leftOrdinal := 9)
    (rightResult := 207493) (rightOrdinal := 27) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45679⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7261⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45679⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7261⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45679⟩⟩], [⟨.program ⟨257⟩, ⟨7261⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge221902

namespace LeftMerge221903
def owner : Owner := ⟨.program ⟨257⟩, ⟨71245⟩⟩
def mergeEvent : Nat := 221903
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨43002⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7261⟩⟩] } }
def leftRaw : List Term := Proof.Events866.exact221896RawTerms
def rightRaw : List Term := Proof.Events810.exact207493RawTerms
def group : MergeGroup := .operator 221896 207493
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 221896) (leftOrdinal := 10)
    (rightResult := 207493) (rightOrdinal := 28) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨43002⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7261⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨43002⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7261⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨43002⟩⟩], [⟨.program ⟨257⟩, ⟨7261⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge221903

namespace LeftMerge221904
def owner : Owner := ⟨.program ⟨257⟩, ⟨71245⟩⟩
def mergeEvent : Nat := 221904
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40322⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7261⟩⟩] } }
def leftRaw : List Term := Proof.Events866.exact221896RawTerms
def rightRaw : List Term := Proof.Events810.exact207493RawTerms
def group : MergeGroup := .operator 221896 207493
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 221896) (leftOrdinal := 12)
    (rightResult := 207493) (rightOrdinal := 30) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40322⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7261⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40322⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7261⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40322⟩⟩], [⟨.program ⟨257⟩, ⟨7261⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge221904

namespace LeftMerge221905
def owner : Owner := ⟨.program ⟨257⟩, ⟨71245⟩⟩
def mergeEvent : Nat := 221905
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37639⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7261⟩⟩] } }
def leftRaw : List Term := Proof.Events866.exact221896RawTerms
def rightRaw : List Term := Proof.Events810.exact207493RawTerms
def group : MergeGroup := .operator 221896 207493
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 221896) (leftOrdinal := 13)
    (rightResult := 207493) (rightOrdinal := 31) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37639⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7261⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37639⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7261⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37639⟩⟩], [⟨.program ⟨257⟩, ⟨7261⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge221905

namespace LeftMerge221906
def owner : Owner := ⟨.program ⟨257⟩, ⟨71245⟩⟩
def mergeEvent : Nat := 221906
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34959⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7261⟩⟩] } }
def leftRaw : List Term := Proof.Events866.exact221896RawTerms
def rightRaw : List Term := Proof.Events810.exact207493RawTerms
def group : MergeGroup := .operator 221896 207493
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 221896) (leftOrdinal := 14)
    (rightResult := 207493) (rightOrdinal := 32) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34959⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7261⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34959⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7261⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34959⟩⟩], [⟨.program ⟨257⟩, ⟨7261⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge221906

namespace LeftMerge221907
def owner : Owner := ⟨.program ⟨257⟩, ⟨71245⟩⟩
def mergeEvent : Nat := 221907
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29302⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7261⟩⟩] } }
def leftRaw : List Term := Proof.Events866.exact221896RawTerms
def rightRaw : List Term := Proof.Events810.exact207493RawTerms
def group : MergeGroup := .operator 221896 207493
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 221896) (leftOrdinal := 16)
    (rightResult := 207493) (rightOrdinal := 34) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29302⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7261⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29302⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7261⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29302⟩⟩], [⟨.program ⟨257⟩, ⟨7261⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge221907

namespace LeftMerge221908
def owner : Owner := ⟨.program ⟨257⟩, ⟨71245⟩⟩
def mergeEvent : Nat := 221908
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26622⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7261⟩⟩] } }
def leftRaw : List Term := Proof.Events866.exact221896RawTerms
def rightRaw : List Term := Proof.Events810.exact207493RawTerms
def group : MergeGroup := .operator 221896 207493
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 221896) (leftOrdinal := 17)
    (rightResult := 207493) (rightOrdinal := 35) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26622⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7261⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26622⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7261⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26622⟩⟩], [⟨.program ⟨257⟩, ⟨7261⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge221908

namespace LeftMerge221909
def owner : Owner := ⟨.program ⟨257⟩, ⟨71245⟩⟩
def mergeEvent : Nat := 221909
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66588⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7261⟩⟩] } }
def leftRaw : List Term := Proof.Events866.exact221896RawTerms
def rightRaw : List Term := Proof.Events810.exact207493RawTerms
def group : MergeGroup := .operator 221896 207493
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 221896) (leftOrdinal := 19)
    (rightResult := 207493) (rightOrdinal := 37) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66588⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7261⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66588⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7261⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66588⟩⟩], [⟨.program ⟨257⟩, ⟨7261⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge221909

namespace LeftMerge221910
def owner : Owner := ⟨.program ⟨257⟩, ⟨71245⟩⟩
def mergeEvent : Nat := 221910
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63085⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7261⟩⟩] } }
def leftRaw : List Term := Proof.Events866.exact221896RawTerms
def rightRaw : List Term := Proof.Events810.exact207493RawTerms
def group : MergeGroup := .operator 221896 207493
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 221896) (leftOrdinal := 1)
    (rightResult := 207493) (rightOrdinal := 19) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63085⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7261⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63085⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7261⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63085⟩⟩], [⟨.program ⟨257⟩, ⟨7261⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge221910

namespace LeftMerge221911
def owner : Owner := ⟨.program ⟨257⟩, ⟨71245⟩⟩
def mergeEvent : Nat := 221911
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60105⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7261⟩⟩] } }
def leftRaw : List Term := Proof.Events866.exact221896RawTerms
def rightRaw : List Term := Proof.Events810.exact207493RawTerms
def group : MergeGroup := .operator 221896 207493
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 221896) (leftOrdinal := 2)
    (rightResult := 207493) (rightOrdinal := 20) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60105⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7261⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60105⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7261⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60105⟩⟩], [⟨.program ⟨257⟩, ⟨7261⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge221911

namespace LeftMerge221912
def owner : Owner := ⟨.program ⟨257⟩, ⟨71245⟩⟩
def mergeEvent : Nat := 221912
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57125⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7261⟩⟩] } }
def leftRaw : List Term := Proof.Events866.exact221896RawTerms
def rightRaw : List Term := Proof.Events810.exact207493RawTerms
def group : MergeGroup := .operator 221896 207493
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 221896) (leftOrdinal := 3)
    (rightResult := 207493) (rightOrdinal := 21) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57125⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7261⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57125⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7261⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57125⟩⟩], [⟨.program ⟨257⟩, ⟨7261⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge221912

namespace LeftMerge221913
def owner : Owner := ⟨.program ⟨257⟩, ⟨71245⟩⟩
def mergeEvent : Nat := 221913
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54145⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7261⟩⟩] } }
def leftRaw : List Term := Proof.Events866.exact221896RawTerms
def rightRaw : List Term := Proof.Events810.exact207493RawTerms
def group : MergeGroup := .operator 221896 207493
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 221896) (leftOrdinal := 4)
    (rightResult := 207493) (rightOrdinal := 22) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54145⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7261⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54145⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7261⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54145⟩⟩], [⟨.program ⟨257⟩, ⟨7261⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge221913

namespace LeftMerge221914
def owner : Owner := ⟨.program ⟨257⟩, ⟨71245⟩⟩
def mergeEvent : Nat := 221914
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51165⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7261⟩⟩] } }
def leftRaw : List Term := Proof.Events866.exact221896RawTerms
def rightRaw : List Term := Proof.Events810.exact207493RawTerms
def group : MergeGroup := .operator 221896 207493
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 221896) (leftOrdinal := 5)
    (rightResult := 207493) (rightOrdinal := 23) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51165⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7261⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51165⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7261⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51165⟩⟩], [⟨.program ⟨257⟩, ⟨7261⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge221914

namespace LeftMerge221915
def owner : Owner := ⟨.program ⟨257⟩, ⟨71245⟩⟩
def mergeEvent : Nat := 221915
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32101⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7261⟩⟩] } }
def leftRaw : List Term := Proof.Events866.exact221896RawTerms
def rightRaw : List Term := Proof.Events810.exact207493RawTerms
def group : MergeGroup := .operator 221896 207493
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 221896) (leftOrdinal := 7)
    (rightResult := 207493) (rightOrdinal := 25) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32101⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7261⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32101⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7261⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32101⟩⟩], [⟨.program ⟨257⟩, ⟨7261⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge221915

namespace LeftMerge221916
def owner : Owner := ⟨.program ⟨257⟩, ⟨71245⟩⟩
def mergeEvent : Nat := 221916
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22081⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7261⟩⟩] } }
def leftRaw : List Term := Proof.Events866.exact221896RawTerms
def rightRaw : List Term := Proof.Events810.exact207493RawTerms
def group : MergeGroup := .operator 221896 207493
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 221896) (leftOrdinal := 11)
    (rightResult := 207493) (rightOrdinal := 29) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22081⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7261⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22081⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7261⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22081⟩⟩], [⟨.program ⟨257⟩, ⟨7261⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge221916

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
