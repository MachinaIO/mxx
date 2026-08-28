import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge45880
def owner : Owner := ⟨.program ⟨257⟩, ⟨20925⟩⟩
def mergeEvent : Nat := 45880
def frameStart : Nat := 45802
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7180⟩⟩, ⟨.program ⟨257⟩, ⟨20924⟩⟩] } }
def leftRaw : List Term := Proof.Events179.exact45876RawTerms
def rightRaw : List Term := Proof.Events179.exact45853RawTerms
def group : MergeGroup := .operator 45876 45853
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 45876) (leftOrdinal := 0)
    (rightResult := 45853) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7180⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨20924⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7180⟩⟩, ⟨.program ⟨257⟩, ⟨20924⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge45880

namespace LeftMerge45881
def owner : Owner := ⟨.program ⟨257⟩, ⟨20925⟩⟩
def mergeEvent : Nat := 45881
def frameStart : Nat := 45802
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨18660⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨20924⟩⟩] } }
def leftRaw : List Term := Proof.Events179.exact45876RawTerms
def rightRaw : List Term := Proof.Events179.exact45853RawTerms
def group : MergeGroup := .operator 45876 45853
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 45876) (leftOrdinal := 1)
    (rightResult := 45853) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨18660⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨20924⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨18660⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨20924⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge45881

namespace LeftMerge45883
def owner : Owner := ⟨.program ⟨257⟩, ⟨20925⟩⟩
def mergeEvent : Nat := 45883
def frameStart : Nat := 45802
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨18660⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨19941⟩⟩] } }
def rhsRaw : List Term := Proof.Events179.exact45850RawTerms
def group : MergeGroup := .relation 45882
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 45882) (rhsResult := 45850)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨18660⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨20924⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨20924⟩⟩) ⟨19941⟩ 45850) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨19941⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨18660⟩⟩], [⟨.program ⟨257⟩, ⟨19941⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge45883

namespace LeftMerge45891
def owner : Owner := ⟨.program ⟨257⟩, ⟨19035⟩⟩
def mergeEvent : Nat := 45891
def frameStart : Nat := 45802
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨19032⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events179.exact45864RawTerms
def rightRaw : List Term := Proof.Events179.exact45887RawTerms
def group : MergeGroup := .operator 45864 45887
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 45864) (leftOrdinal := 0)
    (rightResult := 45887) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨19032⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨19032⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge45891

namespace LeftMerge45908
def owner : Owner := ⟨.program ⟨257⟩, ⟨19635⟩⟩
def mergeEvent : Nat := 45908
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7199⟩⟩] } }
def rhsRaw : List Term := Proof.Events179.exact45905RawTerms
def group : MergeGroup := .relation 45907
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 45907) (rhsResult := 45905)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19632⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 45906 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19632⟩⟩]⟩) (none) 45905) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7199⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge45908

namespace LeftMerge45909
def owner : Owner := ⟨.program ⟨257⟩, ⟨19635⟩⟩
def mergeEvent : Nat := 45909
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7180⟩⟩, ⟨.program ⟨257⟩, ⟨20924⟩⟩] } }
def rhsRaw : List Term := Proof.Events179.exact45905RawTerms
def group : MergeGroup := .relation 45907
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 45907) (rhsResult := 45905)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19632⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 45906 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19632⟩⟩]⟩) (none) 45905) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7180⟩⟩, ⟨.program ⟨257⟩, ⟨20924⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7180⟩⟩, ⟨.program ⟨257⟩, ⟨20924⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge45909

namespace LeftMerge45910
def owner : Owner := ⟨.program ⟨257⟩, ⟨19635⟩⟩
def mergeEvent : Nat := 45910
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨18660⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨19941⟩⟩] } }
def rhsRaw : List Term := Proof.Events179.exact45905RawTerms
def group : MergeGroup := .relation 45907
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 45907) (rhsResult := 45905)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19632⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 45906 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19632⟩⟩]⟩) (none) 45905) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨18660⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨19941⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨18660⟩⟩], [⟨.program ⟨257⟩, ⟨19941⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge45910

namespace LeftMerge45911
def owner : Owner := ⟨.program ⟨257⟩, ⟨19635⟩⟩
def mergeEvent : Nat := 45911
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨19032⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events179.exact45905RawTerms
def group : MergeGroup := .relation 45907
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 45907) (rhsResult := 45905)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19632⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 45906 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19632⟩⟩]⟩) (none) 45905) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨19032⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨19032⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge45911

namespace LeftMerge45916
def owner : Owner := ⟨.program ⟨257⟩, ⟨20927⟩⟩
def mergeEvent : Nat := 45916
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7180⟩⟩, ⟨.program ⟨257⟩, ⟨20924⟩⟩] } }
def leftRaw : List Term := Proof.Events179.exact45912RawTerms
def rightRaw : List Term := Proof.Events178.exact45734RawTerms
def group : MergeGroup := .operator 45912 45734
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 45912) (leftOrdinal := 0)
    (rightResult := 45734) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7180⟩⟩, ⟨.program ⟨257⟩, ⟨20924⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7180⟩⟩, ⟨.program ⟨257⟩, ⟨20924⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7180⟩⟩, ⟨.program ⟨257⟩, ⟨20924⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge45916

namespace LeftMerge45917
def owner : Owner := ⟨.program ⟨257⟩, ⟨20927⟩⟩
def mergeEvent : Nat := 45917
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨18660⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨19941⟩⟩] } }
def leftRaw : List Term := Proof.Events179.exact45912RawTerms
def rightRaw : List Term := Proof.Events178.exact45734RawTerms
def group : MergeGroup := .operator 45912 45734
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 45912) (leftOrdinal := 2)
    (rightResult := 45734) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨18660⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨19941⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨18660⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨19941⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨18660⟩⟩], [⟨.program ⟨257⟩, ⟨19941⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge45917

namespace LeftMerge45925
def owner : Owner := ⟨.program ⟨257⟩, ⟨20928⟩⟩
def mergeEvent : Nat := 45925
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩] } }
def leftRaw : List Term := Proof.Events179.exact45919RawTerms
def rightRaw : List Term := Proof.Events061.exact15862RawTerms
def group : MergeGroup := .operator 45919 15862
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 45919) (leftOrdinal := 0)
    (rightResult := 15862) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7199⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7165⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge45925

namespace LeftMerge45926
def owner : Owner := ⟨.program ⟨257⟩, ⟨20928⟩⟩
def mergeEvent : Nat := 45926
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨19032⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩] } }
def leftRaw : List Term := Proof.Events179.exact45919RawTerms
def rightRaw : List Term := Proof.Events061.exact15862RawTerms
def group : MergeGroup := .operator 45919 15862
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 45919) (leftOrdinal := 1)
    (rightResult := 15862) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨19032⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7165⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨19032⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge45926

namespace LeftMerge45928
def owner : Owner := ⟨.program ⟨257⟩, ⟨20928⟩⟩
def mergeEvent : Nat := 45928
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨19032⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events061.exact15855RawTerms
def group : MergeGroup := .relation 45927
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 45927) (rhsResult := 15855)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨19032⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨7165⟩⟩) ⟨7048⟩ 15855) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨19032⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge45928

namespace LeftMerge45942
def owner : Owner := ⟨.program ⟨257⟩, ⟨18008⟩⟩
def mergeEvent : Nat := 45942
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨18006⟩⟩] } }
def leftRaw : List Term := Proof.Events158.exact40500RawTerms
def rightRaw : List Term := Proof.Events179.exact45936RawTerms
def group : MergeGroup := .operator 40500 45936
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 40500) (leftOrdinal := 0)
    (rightResult := 45936) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7179⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨18006⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨18006⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge45942

namespace LeftMerge45943
def owner : Owner := ⟨.program ⟨257⟩, ⟨18008⟩⟩
def mergeEvent : Nat := 45943
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨15860⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨18006⟩⟩] } }
def leftRaw : List Term := Proof.Events158.exact40500RawTerms
def rightRaw : List Term := Proof.Events179.exact45936RawTerms
def group : MergeGroup := .operator 40500 45936
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 40500) (leftOrdinal := 1)
    (rightResult := 45936) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨15860⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨18006⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨15860⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨18006⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge45943

namespace LeftMerge45945
def owner : Owner := ⟨.program ⟨257⟩, ⟨18008⟩⟩
def mergeEvent : Nat := 45945
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨15860⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨17081⟩⟩] } }
def rhsRaw : List Term := Proof.Events179.exact45933RawTerms
def group : MergeGroup := .relation 45944
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 45944) (rhsResult := 45933)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨15860⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨18006⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨18006⟩⟩) ⟨17081⟩ 45933) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨17081⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨15860⟩⟩], [⟨.program ⟨257⟩, ⟨17081⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge45945

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
