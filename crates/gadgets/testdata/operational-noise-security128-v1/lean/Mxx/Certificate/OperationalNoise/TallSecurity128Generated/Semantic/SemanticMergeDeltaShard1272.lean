import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge207269
def owner : Owner := ⟨.program ⟨257⟩, ⟨71305⟩⟩
def mergeEvent : Nat := 207269
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16062⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7259⟩⟩] } }
def rhsRaw : List Term := Proof.Events064.exact16457RawTerms
def group : MergeGroup := .relation 207268
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 207268) (rhsResult := 16457)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16062⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9515⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨9515⟩⟩) ⟨7259⟩ 16457) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7259⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16062⟩⟩], [⟨.program ⟨257⟩, ⟨7259⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge207269

namespace LeftMerge207270
def owner : Owner := ⟨.program ⟨257⟩, ⟨71305⟩⟩
def mergeEvent : Nat := 207270
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7258⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩, ⟨.program ⟨257⟩, ⟨9515⟩⟩] } }
def leftRaw : List Term := Proof.Events809.exact207207RawTerms
def rightRaw : List Term := Proof.Events064.exact16464RawTerms
def group : MergeGroup := .operator 207207 16464
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 207207) (leftOrdinal := 0)
    (rightResult := 16464) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7258⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9515⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7258⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩, ⟨.program ⟨257⟩, ⟨9515⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge207270

namespace LeftMerge207275
def owner : Owner := ⟨.program ⟨257⟩, ⟨71306⟩⟩
def mergeEvent : Nat := 207275
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67494⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7259⟩⟩] } }
def leftRaw : List Term := Proof.Events809.exact207271RawTerms
def rightRaw : List Term := Proof.Events753.exact192868RawTerms
def group : MergeGroup := .operator 207271 192868
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 207271) (leftOrdinal := 6)
    (rightResult := 192868) (rightOrdinal := 24) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67494⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7259⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67494⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7259⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67494⟩⟩], [⟨.program ⟨257⟩, ⟨7259⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge207275

namespace LeftMerge207276
def owner : Owner := ⟨.program ⟨257⟩, ⟨71306⟩⟩
def mergeEvent : Nat := 207276
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48385⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7259⟩⟩] } }
def leftRaw : List Term := Proof.Events809.exact207271RawTerms
def rightRaw : List Term := Proof.Events753.exact192868RawTerms
def group : MergeGroup := .operator 207271 192868
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 207271) (leftOrdinal := 8)
    (rightResult := 192868) (rightOrdinal := 26) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48385⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7259⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48385⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7259⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48385⟩⟩], [⟨.program ⟨257⟩, ⟨7259⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge207276

namespace LeftMerge207277
def owner : Owner := ⟨.program ⟨257⟩, ⟨71306⟩⟩
def mergeEvent : Nat := 207277
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45705⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7259⟩⟩] } }
def leftRaw : List Term := Proof.Events809.exact207271RawTerms
def rightRaw : List Term := Proof.Events753.exact192868RawTerms
def group : MergeGroup := .operator 207271 192868
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 207271) (leftOrdinal := 9)
    (rightResult := 192868) (rightOrdinal := 27) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45705⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7259⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45705⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7259⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45705⟩⟩], [⟨.program ⟨257⟩, ⟨7259⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge207277

namespace LeftMerge207278
def owner : Owner := ⟨.program ⟨257⟩, ⟨71306⟩⟩
def mergeEvent : Nat := 207278
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨43028⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7259⟩⟩] } }
def leftRaw : List Term := Proof.Events809.exact207271RawTerms
def rightRaw : List Term := Proof.Events753.exact192868RawTerms
def group : MergeGroup := .operator 207271 192868
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 207271) (leftOrdinal := 10)
    (rightResult := 192868) (rightOrdinal := 28) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨43028⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7259⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨43028⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7259⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨43028⟩⟩], [⟨.program ⟨257⟩, ⟨7259⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge207278

namespace LeftMerge207279
def owner : Owner := ⟨.program ⟨257⟩, ⟨71306⟩⟩
def mergeEvent : Nat := 207279
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40348⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7259⟩⟩] } }
def leftRaw : List Term := Proof.Events809.exact207271RawTerms
def rightRaw : List Term := Proof.Events753.exact192868RawTerms
def group : MergeGroup := .operator 207271 192868
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 207271) (leftOrdinal := 12)
    (rightResult := 192868) (rightOrdinal := 30) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40348⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7259⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40348⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7259⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40348⟩⟩], [⟨.program ⟨257⟩, ⟨7259⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge207279

namespace LeftMerge207280
def owner : Owner := ⟨.program ⟨257⟩, ⟨71306⟩⟩
def mergeEvent : Nat := 207280
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37665⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7259⟩⟩] } }
def leftRaw : List Term := Proof.Events809.exact207271RawTerms
def rightRaw : List Term := Proof.Events753.exact192868RawTerms
def group : MergeGroup := .operator 207271 192868
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 207271) (leftOrdinal := 13)
    (rightResult := 192868) (rightOrdinal := 31) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37665⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7259⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37665⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7259⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37665⟩⟩], [⟨.program ⟨257⟩, ⟨7259⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge207280

namespace LeftMerge207281
def owner : Owner := ⟨.program ⟨257⟩, ⟨71306⟩⟩
def mergeEvent : Nat := 207281
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34985⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7259⟩⟩] } }
def leftRaw : List Term := Proof.Events809.exact207271RawTerms
def rightRaw : List Term := Proof.Events753.exact192868RawTerms
def group : MergeGroup := .operator 207271 192868
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 207271) (leftOrdinal := 14)
    (rightResult := 192868) (rightOrdinal := 32) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34985⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7259⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34985⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7259⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34985⟩⟩], [⟨.program ⟨257⟩, ⟨7259⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge207281

namespace LeftMerge207282
def owner : Owner := ⟨.program ⟨257⟩, ⟨71306⟩⟩
def mergeEvent : Nat := 207282
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29328⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7259⟩⟩] } }
def leftRaw : List Term := Proof.Events809.exact207271RawTerms
def rightRaw : List Term := Proof.Events753.exact192868RawTerms
def group : MergeGroup := .operator 207271 192868
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 207271) (leftOrdinal := 16)
    (rightResult := 192868) (rightOrdinal := 34) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29328⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7259⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29328⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7259⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29328⟩⟩], [⟨.program ⟨257⟩, ⟨7259⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge207282

namespace LeftMerge207283
def owner : Owner := ⟨.program ⟨257⟩, ⟨71306⟩⟩
def mergeEvent : Nat := 207283
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26648⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7259⟩⟩] } }
def leftRaw : List Term := Proof.Events809.exact207271RawTerms
def rightRaw : List Term := Proof.Events753.exact192868RawTerms
def group : MergeGroup := .operator 207271 192868
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 207271) (leftOrdinal := 17)
    (rightResult := 192868) (rightOrdinal := 35) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26648⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7259⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26648⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7259⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26648⟩⟩], [⟨.program ⟨257⟩, ⟨7259⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge207283

namespace LeftMerge207284
def owner : Owner := ⟨.program ⟨257⟩, ⟨71306⟩⟩
def mergeEvent : Nat := 207284
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66728⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7259⟩⟩] } }
def leftRaw : List Term := Proof.Events809.exact207271RawTerms
def rightRaw : List Term := Proof.Events753.exact192868RawTerms
def group : MergeGroup := .operator 207271 192868
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 207271) (leftOrdinal := 19)
    (rightResult := 192868) (rightOrdinal := 37) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66728⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7259⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66728⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7259⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66728⟩⟩], [⟨.program ⟨257⟩, ⟨7259⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge207284

namespace LeftMerge207285
def owner : Owner := ⟨.program ⟨257⟩, ⟨71306⟩⟩
def mergeEvent : Nat := 207285
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63123⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7259⟩⟩] } }
def leftRaw : List Term := Proof.Events809.exact207271RawTerms
def rightRaw : List Term := Proof.Events753.exact192868RawTerms
def group : MergeGroup := .operator 207271 192868
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 207271) (leftOrdinal := 1)
    (rightResult := 192868) (rightOrdinal := 19) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63123⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7259⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63123⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7259⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63123⟩⟩], [⟨.program ⟨257⟩, ⟨7259⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge207285

namespace LeftMerge207286
def owner : Owner := ⟨.program ⟨257⟩, ⟨71306⟩⟩
def mergeEvent : Nat := 207286
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60143⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7259⟩⟩] } }
def leftRaw : List Term := Proof.Events809.exact207271RawTerms
def rightRaw : List Term := Proof.Events753.exact192868RawTerms
def group : MergeGroup := .operator 207271 192868
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 207271) (leftOrdinal := 2)
    (rightResult := 192868) (rightOrdinal := 20) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60143⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7259⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60143⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7259⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60143⟩⟩], [⟨.program ⟨257⟩, ⟨7259⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge207286

namespace LeftMerge207287
def owner : Owner := ⟨.program ⟨257⟩, ⟨71306⟩⟩
def mergeEvent : Nat := 207287
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57163⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7259⟩⟩] } }
def leftRaw : List Term := Proof.Events809.exact207271RawTerms
def rightRaw : List Term := Proof.Events753.exact192868RawTerms
def group : MergeGroup := .operator 207271 192868
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 207271) (leftOrdinal := 3)
    (rightResult := 192868) (rightOrdinal := 21) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57163⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7259⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57163⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7259⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57163⟩⟩], [⟨.program ⟨257⟩, ⟨7259⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge207287

namespace LeftMerge207288
def owner : Owner := ⟨.program ⟨257⟩, ⟨71306⟩⟩
def mergeEvent : Nat := 207288
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54183⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7259⟩⟩] } }
def leftRaw : List Term := Proof.Events809.exact207271RawTerms
def rightRaw : List Term := Proof.Events753.exact192868RawTerms
def group : MergeGroup := .operator 207271 192868
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 207271) (leftOrdinal := 4)
    (rightResult := 192868) (rightOrdinal := 22) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54183⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7259⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54183⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7259⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54183⟩⟩], [⟨.program ⟨257⟩, ⟨7259⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge207288

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
