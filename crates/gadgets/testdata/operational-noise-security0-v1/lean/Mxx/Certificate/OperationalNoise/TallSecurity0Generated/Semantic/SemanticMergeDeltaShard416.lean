import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge67947
def owner : Owner := ⟨.program ⟨214⟩, ⟨25371⟩⟩
def mergeEvent : Nat := 67947
def frameStart : Nat := 67852
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨9815⟩⟩, ⟨.program ⟨214⟩, ⟨12362⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23204⟩⟩] } }
def rhsRaw : List Term := Proof.Events265.exact67894RawTerms
def group : MergeGroup := .relation 67946
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 67946) (rhsResult := 67894)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨9815⟩⟩, ⟨.program ⟨214⟩, ⟨12362⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25368⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨25368⟩⟩) ⟨23204⟩ 67894) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23204⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨9815⟩⟩, ⟨.program ⟨214⟩, ⟨12362⟩⟩], [⟨.program ⟨214⟩, ⟨23204⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge67947

namespace LeftMerge67955
def owner : Owner := ⟨.program ⟨214⟩, ⟨16463⟩⟩
def mergeEvent : Nat := 67955
def frameStart : Nat := 67852
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16461⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events265.exact67908RawTerms
def rightRaw : List Term := Proof.Events265.exact67951RawTerms
def group : MergeGroup := .operator 67908 67951
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 67908) (leftOrdinal := 0)
    (rightResult := 67951) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16461⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨16461⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge67955

namespace LeftMerge67972
def owner : Owner := ⟨.program ⟨214⟩, ⟨19887⟩⟩
def mergeEvent : Nat := 67972
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6702⟩⟩] } }
def rhsRaw : List Term := Proof.Events265.exact67969RawTerms
def group : MergeGroup := .relation 67971
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 67971) (rhsResult := 67969)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19884⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 67970 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19884⟩⟩]⟩) (none) 67969) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6702⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6702⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge67972

namespace LeftMerge67973
def owner : Owner := ⟨.program ⟨214⟩, ⟨19887⟩⟩
def mergeEvent : Nat := 67973
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6765⟩⟩, ⟨.program ⟨214⟩, ⟨7867⟩⟩, ⟨.program ⟨214⟩, ⟨25368⟩⟩] } }
def rhsRaw : List Term := Proof.Events265.exact67969RawTerms
def group : MergeGroup := .relation 67971
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 67971) (rhsResult := 67969)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19884⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 67970 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19884⟩⟩]⟩) (none) 67969) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6765⟩⟩, ⟨.program ⟨214⟩, ⟨7867⟩⟩, ⟨.program ⟨214⟩, ⟨25368⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6765⟩⟩, ⟨.program ⟨214⟩, ⟨7867⟩⟩, ⟨.program ⟨214⟩, ⟨25368⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge67973

namespace LeftMerge67974
def owner : Owner := ⟨.program ⟨214⟩, ⟨19887⟩⟩
def mergeEvent : Nat := 67974
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9815⟩⟩, ⟨.program ⟨214⟩, ⟨12362⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23204⟩⟩] } }
def rhsRaw : List Term := Proof.Events265.exact67969RawTerms
def group : MergeGroup := .relation 67971
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 67971) (rhsResult := 67969)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19884⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 67970 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19884⟩⟩]⟩) (none) 67969) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨9815⟩⟩, ⟨.program ⟨214⟩, ⟨12362⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23204⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9815⟩⟩, ⟨.program ⟨214⟩, ⟨12362⟩⟩], [⟨.program ⟨214⟩, ⟨23204⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge67974

namespace LeftMerge67975
def owner : Owner := ⟨.program ⟨214⟩, ⟨19887⟩⟩
def mergeEvent : Nat := 67975
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16461⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def rhsRaw : List Term := Proof.Events265.exact67969RawTerms
def group : MergeGroup := .relation 67971
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 67971) (rhsResult := 67969)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19884⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 67970 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19884⟩⟩]⟩) (none) 67969) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16461⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16461⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge67975

namespace LeftMerge67980
def owner : Owner := ⟨.program ⟨214⟩, ⟨25370⟩⟩
def mergeEvent : Nat := 67980
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9815⟩⟩, ⟨.program ⟨214⟩, ⟨12362⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23204⟩⟩] } }
def leftRaw : List Term := Proof.Events265.exact67976RawTerms
def rightRaw : List Term := Proof.Events264.exact67790RawTerms
def group : MergeGroup := .operator 67976 67790
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 67976) (leftOrdinal := 2)
    (rightResult := 67790) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9815⟩⟩, ⟨.program ⟨214⟩, ⟨12362⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23204⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9815⟩⟩, ⟨.program ⟨214⟩, ⟨12362⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23204⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9815⟩⟩, ⟨.program ⟨214⟩, ⟨12362⟩⟩], [⟨.program ⟨214⟩, ⟨23204⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge67980

namespace LeftMerge67981
def owner : Owner := ⟨.program ⟨214⟩, ⟨25370⟩⟩
def mergeEvent : Nat := 67981
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6765⟩⟩, ⟨.program ⟨214⟩, ⟨7867⟩⟩, ⟨.program ⟨214⟩, ⟨25368⟩⟩] } }
def leftRaw : List Term := Proof.Events265.exact67976RawTerms
def rightRaw : List Term := Proof.Events264.exact67790RawTerms
def group : MergeGroup := .operator 67976 67790
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 67976) (leftOrdinal := 1)
    (rightResult := 67790) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6765⟩⟩, ⟨.program ⟨214⟩, ⟨7867⟩⟩, ⟨.program ⟨214⟩, ⟨25368⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6765⟩⟩, ⟨.program ⟨214⟩, ⟨7867⟩⟩, ⟨.program ⟨214⟩, ⟨25368⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6765⟩⟩, ⟨.program ⟨214⟩, ⟨7867⟩⟩, ⟨.program ⟨214⟩, ⟨25368⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge67981

namespace LeftMerge67989
def owner : Owner := ⟨.program ⟨214⟩, ⟨28940⟩⟩
def mergeEvent : Nat := 67989
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6702⟩⟩, ⟨.program ⟨214⟩, ⟨28938⟩⟩] } }
def leftRaw : List Term := Proof.Events265.exact67983RawTerms
def rightRaw : List Term := Proof.Events264.exact67706RawTerms
def group : MergeGroup := .operator 67983 67706
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 67983) (leftOrdinal := 0)
    (rightResult := 67706) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6702⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨28938⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6702⟩⟩, ⟨.program ⟨214⟩, ⟨28938⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge67989

namespace LeftMerge67990
def owner : Owner := ⟨.program ⟨214⟩, ⟨28940⟩⟩
def mergeEvent : Nat := 67990
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16461⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28938⟩⟩] } }
def leftRaw : List Term := Proof.Events265.exact67983RawTerms
def rightRaw : List Term := Proof.Events264.exact67706RawTerms
def group : MergeGroup := .operator 67983 67706
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 67983) (leftOrdinal := 1)
    (rightResult := 67706) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16461⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨28938⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16461⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28938⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge67990

namespace LeftMerge67992
def owner : Owner := ⟨.program ⟨214⟩, ⟨28940⟩⟩
def mergeEvent : Nat := 67992
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16461⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24474⟩⟩] } }
def rhsRaw : List Term := Proof.Events264.exact67703RawTerms
def group : MergeGroup := .relation 67991
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 67991) (rhsResult := 67703)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16461⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28938⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨28938⟩⟩) ⟨24474⟩ 67703) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨24474⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16461⟩⟩], [⟨.program ⟨214⟩, ⟨24474⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge67992

namespace LeftMerge68006
def owner : Owner := ⟨.program ⟨214⟩, ⟨22119⟩⟩
def mergeEvent : Nat := 68006
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22116⟩⟩] } }
def leftRaw : List Term := Proof.Events255.exact65387RawTerms
def rightRaw : List Term := Proof.Events265.exact68000RawTerms
def group : MergeGroup := .operator 65387 68000
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 65387) (leftOrdinal := 0)
    (rightResult := 68000) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨22116⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22116⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge68006

namespace LeftMerge68127
def owner : Owner := ⟨.program ⟨214⟩, ⟨16503⟩⟩
def mergeEvent : Nat := 68127
def frameStart : Nat := 68061
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16461⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events266.exact68123RawTerms
def rightRaw : List Term := Proof.Events266.exact68121RawTerms
def group : MergeGroup := .operator 68123 68121
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 68123) (leftOrdinal := 0)
    (rightResult := 68121) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16461⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨16461⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge68127

namespace LeftMerge68139
def owner : Owner := ⟨.program ⟨214⟩, ⟨28939⟩⟩
def mergeEvent : Nat := 68139
def frameStart : Nat := 68061
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6702⟩⟩, ⟨.program ⟨214⟩, ⟨28938⟩⟩] } }
def leftRaw : List Term := Proof.Events266.exact68135RawTerms
def rightRaw : List Term := Proof.Events266.exact68112RawTerms
def group : MergeGroup := .operator 68135 68112
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 68135) (leftOrdinal := 0)
    (rightResult := 68112) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6702⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨28938⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6702⟩⟩, ⟨.program ⟨214⟩, ⟨28938⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge68139

namespace LeftMerge68140
def owner : Owner := ⟨.program ⟨214⟩, ⟨28939⟩⟩
def mergeEvent : Nat := 68140
def frameStart : Nat := 68061
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16461⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28938⟩⟩] } }
def leftRaw : List Term := Proof.Events266.exact68135RawTerms
def rightRaw : List Term := Proof.Events266.exact68112RawTerms
def group : MergeGroup := .operator 68135 68112
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 68135) (leftOrdinal := 1)
    (rightResult := 68112) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16461⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨28938⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨16461⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28938⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge68140

namespace LeftMerge68142
def owner : Owner := ⟨.program ⟨214⟩, ⟨28939⟩⟩
def mergeEvent : Nat := 68142
def frameStart : Nat := 68061
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16461⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24474⟩⟩] } }
def rhsRaw : List Term := Proof.Events266.exact68109RawTerms
def group : MergeGroup := .relation 68141
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 68141) (rhsResult := 68109)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨16461⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28938⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨28938⟩⟩) ⟨24474⟩ 68109) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨24474⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨16461⟩⟩], [⟨.program ⟨214⟩, ⟨24474⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge68142

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
