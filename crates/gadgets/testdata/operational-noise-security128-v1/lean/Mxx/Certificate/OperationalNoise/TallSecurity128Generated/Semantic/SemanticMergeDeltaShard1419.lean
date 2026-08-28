import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge230944
def owner : Owner := ⟨.program ⟨257⟩, ⟨71206⟩⟩
def mergeEvent : Nat := 230944
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨66531⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }
def leftRaw : List Term := Proof.Events901.exact230905RawTerms
def rightRaw : List Term := Proof.Events867.exact222128RawTerms
def group : MergeGroup := .operator 230905 222128
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 230905) (leftOrdinal := 35)
    (rightResult := 222128) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨66531⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71204⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨66531⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge230944

namespace LeftMerge230946
def owner : Owner := ⟨.program ⟨257⟩, ⟨71206⟩⟩
def mergeEvent : Nat := 230946
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨66531⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }
def rhsRaw : List Term := Proof.Events867.exact222125RawTerms
def group : MergeGroup := .relation 230945
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 230945) (rhsResult := 222125)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨66531⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71204⟩⟩) ⟨68824⟩ 222125) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨66531⟩⟩], [⟨.program ⟨257⟩, ⟨68824⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge230946

namespace LeftMerge230947
def owner : Owner := ⟨.program ⟨257⟩, ⟨71206⟩⟩
def mergeEvent : Nat := 230947
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7214⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }
def leftRaw : List Term := Proof.Events901.exact230905RawTerms
def rightRaw : List Term := Proof.Events867.exact222128RawTerms
def group : MergeGroup := .operator 230905 222128
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 230905) (leftOrdinal := 8)
    (rightResult := 222128) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7214⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71204⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge230947

namespace LeftMerge230948
def owner : Owner := ⟨.program ⟨257⟩, ⟨71206⟩⟩
def mergeEvent : Nat := 230948
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨63062⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }
def leftRaw : List Term := Proof.Events901.exact230905RawTerms
def rightRaw : List Term := Proof.Events867.exact222128RawTerms
def group : MergeGroup := .operator 230905 222128
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 230905) (leftOrdinal := 34)
    (rightResult := 222128) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨63062⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71204⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨63062⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge230948

namespace LeftMerge230950
def owner : Owner := ⟨.program ⟨257⟩, ⟨71206⟩⟩
def mergeEvent : Nat := 230950
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨63062⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }
def rhsRaw : List Term := Proof.Events867.exact222125RawTerms
def group : MergeGroup := .relation 230949
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 230949) (rhsResult := 222125)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨63062⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71204⟩⟩) ⟨68824⟩ 222125) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨63062⟩⟩], [⟨.program ⟨257⟩, ⟨68824⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge230950

namespace LeftMerge230951
def owner : Owner := ⟨.program ⟨257⟩, ⟨71206⟩⟩
def mergeEvent : Nat := 230951
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }
def leftRaw : List Term := Proof.Events901.exact230905RawTerms
def rightRaw : List Term := Proof.Events867.exact222128RawTerms
def group : MergeGroup := .operator 230905 222128
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 230905) (leftOrdinal := 7)
    (rightResult := 222128) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7212⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71204⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge230951

namespace LeftMerge230952
def owner : Owner := ⟨.program ⟨257⟩, ⟨71206⟩⟩
def mergeEvent : Nat := 230952
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨60082⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }
def leftRaw : List Term := Proof.Events901.exact230905RawTerms
def rightRaw : List Term := Proof.Events867.exact222128RawTerms
def group : MergeGroup := .operator 230905 222128
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 230905) (leftOrdinal := 33)
    (rightResult := 222128) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨60082⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71204⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨60082⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge230952

namespace LeftMerge230954
def owner : Owner := ⟨.program ⟨257⟩, ⟨71206⟩⟩
def mergeEvent : Nat := 230954
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨60082⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }
def rhsRaw : List Term := Proof.Events867.exact222125RawTerms
def group : MergeGroup := .relation 230953
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 230953) (rhsResult := 222125)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨60082⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71204⟩⟩) ⟨68824⟩ 222125) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨60082⟩⟩], [⟨.program ⟨257⟩, ⟨68824⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge230954

namespace LeftMerge230955
def owner : Owner := ⟨.program ⟨257⟩, ⟨71206⟩⟩
def mergeEvent : Nat := 230955
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }
def leftRaw : List Term := Proof.Events901.exact230905RawTerms
def rightRaw : List Term := Proof.Events867.exact222128RawTerms
def group : MergeGroup := .operator 230905 222128
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 230905) (leftOrdinal := 6)
    (rightResult := 222128) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7210⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71204⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge230955

namespace LeftMerge230956
def owner : Owner := ⟨.program ⟨257⟩, ⟨71206⟩⟩
def mergeEvent : Nat := 230956
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨57102⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }
def leftRaw : List Term := Proof.Events901.exact230905RawTerms
def rightRaw : List Term := Proof.Events867.exact222128RawTerms
def group : MergeGroup := .operator 230905 222128
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 230905) (leftOrdinal := 32)
    (rightResult := 222128) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨57102⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71204⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨57102⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge230956

namespace LeftMerge230958
def owner : Owner := ⟨.program ⟨257⟩, ⟨71206⟩⟩
def mergeEvent : Nat := 230958
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨57102⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }
def rhsRaw : List Term := Proof.Events867.exact222125RawTerms
def group : MergeGroup := .relation 230957
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 230957) (rhsResult := 222125)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨57102⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71204⟩⟩) ⟨68824⟩ 222125) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨57102⟩⟩], [⟨.program ⟨257⟩, ⟨68824⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge230958

namespace LeftMerge230959
def owner : Owner := ⟨.program ⟨257⟩, ⟨71206⟩⟩
def mergeEvent : Nat := 230959
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }
def leftRaw : List Term := Proof.Events901.exact230905RawTerms
def rightRaw : List Term := Proof.Events867.exact222128RawTerms
def group : MergeGroup := .operator 230905 222128
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 230905) (leftOrdinal := 5)
    (rightResult := 222128) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7208⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71204⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge230959

namespace LeftMerge230960
def owner : Owner := ⟨.program ⟨257⟩, ⟨71206⟩⟩
def mergeEvent : Nat := 230960
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨54122⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }
def leftRaw : List Term := Proof.Events901.exact230905RawTerms
def rightRaw : List Term := Proof.Events867.exact222128RawTerms
def group : MergeGroup := .operator 230905 222128
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 230905) (leftOrdinal := 31)
    (rightResult := 222128) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨54122⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71204⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨54122⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge230960

namespace LeftMerge230962
def owner : Owner := ⟨.program ⟨257⟩, ⟨71206⟩⟩
def mergeEvent : Nat := 230962
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨54122⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }
def rhsRaw : List Term := Proof.Events867.exact222125RawTerms
def group : MergeGroup := .relation 230961
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 230961) (rhsResult := 222125)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨54122⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71204⟩⟩) ⟨68824⟩ 222125) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨54122⟩⟩], [⟨.program ⟨257⟩, ⟨68824⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge230962

namespace LeftMerge230963
def owner : Owner := ⟨.program ⟨257⟩, ⟨71206⟩⟩
def mergeEvent : Nat := 230963
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }
def leftRaw : List Term := Proof.Events901.exact230905RawTerms
def rightRaw : List Term := Proof.Events867.exact222128RawTerms
def group : MergeGroup := .operator 230905 222128
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 230905) (leftOrdinal := 4)
    (rightResult := 222128) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7206⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71204⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge230963

namespace LeftMerge230964
def owner : Owner := ⟨.program ⟨257⟩, ⟨71206⟩⟩
def mergeEvent : Nat := 230964
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨51142⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }
def leftRaw : List Term := Proof.Events901.exact230905RawTerms
def rightRaw : List Term := Proof.Events867.exact222128RawTerms
def group : MergeGroup := .operator 230905 222128
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 230905) (leftOrdinal := 30)
    (rightResult := 222128) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨51142⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71204⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨51142⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge230964

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
