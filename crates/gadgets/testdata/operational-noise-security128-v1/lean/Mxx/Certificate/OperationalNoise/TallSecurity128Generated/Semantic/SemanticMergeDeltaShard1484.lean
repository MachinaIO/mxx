import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge240864
def owner : Owner := ⟨.program ⟨257⟩, ⟨9543⟩⟩
def mergeEvent : Nat := 240864
def frameStart : Nat := 240781
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩] } }
def leftRaw : List Term := Proof.Events940.exact240860RawTerms
def rightRaw : List Term := Proof.Events940.exact240857RawTerms
def group : MergeGroup := .operator 240860 240857
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 240860) (leftOrdinal := 0)
    (rightResult := 240857) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7294⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9541⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge240864

namespace LeftMerge240873
def owner : Owner := ⟨.program ⟨257⟩, ⟨69221⟩⟩
def mergeEvent : Nat := 240873
def frameStart : Nat := 240781
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩, ⟨.program ⟨257⟩, ⟨69218⟩⟩] } }
def leftRaw : List Term := Proof.Events940.exact240869RawTerms
def rightRaw : List Term := Proof.Events940.exact240826RawTerms
def group : MergeGroup := .operator 240869 240826
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 240869) (leftOrdinal := 0)
    (rightResult := 240826) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨69218⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩, ⟨.program ⟨257⟩, ⟨69218⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge240873

namespace LeftMerge240874
def owner : Owner := ⟨.program ⟨257⟩, ⟨69221⟩⟩
def mergeEvent : Nat := 240874
def frameStart : Nat := 240781
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨25706⟩⟩, ⟨.program ⟨257⟩, ⟨65391⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨69218⟩⟩] } }
def leftRaw : List Term := Proof.Events940.exact240869RawTerms
def rightRaw : List Term := Proof.Events940.exact240826RawTerms
def group : MergeGroup := .operator 240869 240826
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 240869) (leftOrdinal := 1)
    (rightResult := 240826) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25706⟩⟩, ⟨.program ⟨257⟩, ⟨65391⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨69218⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨25706⟩⟩, ⟨.program ⟨257⟩, ⟨65391⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨69218⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge240874

namespace LeftMerge240876
def owner : Owner := ⟨.program ⟨257⟩, ⟨69221⟩⟩
def mergeEvent : Nat := 240876
def frameStart : Nat := 240781
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨25706⟩⟩, ⟨.program ⟨257⟩, ⟨65391⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68518⟩⟩] } }
def rhsRaw : List Term := Proof.Events940.exact240823RawTerms
def group : MergeGroup := .relation 240875
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 240875) (rhsResult := 240823)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨25706⟩⟩, ⟨.program ⟨257⟩, ⟨65391⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨69218⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨69218⟩⟩) ⟨68518⟩ 240823) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68518⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨25706⟩⟩, ⟨.program ⟨257⟩, ⟨65391⟩⟩], [⟨.program ⟨257⟩, ⟨68518⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge240876

namespace LeftMerge240884
def owner : Owner := ⟨.program ⟨257⟩, ⟨65774⟩⟩
def mergeEvent : Nat := 240884
def frameStart : Nat := 240781
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨65772⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events940.exact240837RawTerms
def rightRaw : List Term := Proof.Events940.exact240880RawTerms
def group : MergeGroup := .operator 240837 240880
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 240837) (leftOrdinal := 0)
    (rightResult := 240880) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨65772⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨65772⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge240884

namespace LeftMerge240901
def owner : Owner := ⟨.program ⟨257⟩, ⟨67753⟩⟩
def mergeEvent : Nat := 240901
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7188⟩⟩] } }
def rhsRaw : List Term := Proof.Events941.exact240898RawTerms
def group : MergeGroup := .relation 240900
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 240900) (rhsResult := 240898)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67750⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 240899 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67750⟩⟩]⟩) (none) 240898) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7188⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7188⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge240901

namespace LeftMerge240902
def owner : Owner := ⟨.program ⟨257⟩, ⟨67753⟩⟩
def mergeEvent : Nat := 240902
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩, ⟨.program ⟨257⟩, ⟨69218⟩⟩] } }
def rhsRaw : List Term := Proof.Events941.exact240898RawTerms
def group : MergeGroup := .relation 240900
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 240900) (rhsResult := 240898)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67750⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 240899 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67750⟩⟩]⟩) (none) 240898) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩, ⟨.program ⟨257⟩, ⟨69218⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩, ⟨.program ⟨257⟩, ⟨69218⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge240902

namespace LeftMerge240903
def owner : Owner := ⟨.program ⟨257⟩, ⟨67753⟩⟩
def mergeEvent : Nat := 240903
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨25706⟩⟩, ⟨.program ⟨257⟩, ⟨65391⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68518⟩⟩] } }
def rhsRaw : List Term := Proof.Events941.exact240898RawTerms
def group : MergeGroup := .relation 240900
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 240900) (rhsResult := 240898)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67750⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 240899 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67750⟩⟩]⟩) (none) 240898) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25706⟩⟩, ⟨.program ⟨257⟩, ⟨65391⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68518⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨25706⟩⟩, ⟨.program ⟨257⟩, ⟨65391⟩⟩], [⟨.program ⟨257⟩, ⟨68518⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge240903

namespace LeftMerge240904
def owner : Owner := ⟨.program ⟨257⟩, ⟨67753⟩⟩
def mergeEvent : Nat := 240904
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨65772⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events941.exact240898RawTerms
def group : MergeGroup := .relation 240900
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 240900) (rhsResult := 240898)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67750⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 240899 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67750⟩⟩]⟩) (none) 240898) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨65772⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨65772⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge240904

namespace LeftMerge240909
def owner : Owner := ⟨.program ⟨257⟩, ⟨69220⟩⟩
def mergeEvent : Nat := 240909
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨25706⟩⟩, ⟨.program ⟨257⟩, ⟨65391⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68518⟩⟩] } }
def leftRaw : List Term := Proof.Events941.exact240905RawTerms
def rightRaw : List Term := Proof.Events940.exact240719RawTerms
def group : MergeGroup := .operator 240905 240719
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 240905) (leftOrdinal := 2)
    (rightResult := 240719) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨25706⟩⟩, ⟨.program ⟨257⟩, ⟨65391⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68518⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨25706⟩⟩, ⟨.program ⟨257⟩, ⟨65391⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68518⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨25706⟩⟩, ⟨.program ⟨257⟩, ⟨65391⟩⟩], [⟨.program ⟨257⟩, ⟨68518⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge240909

namespace LeftMerge240910
def owner : Owner := ⟨.program ⟨257⟩, ⟨69220⟩⟩
def mergeEvent : Nat := 240910
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩, ⟨.program ⟨257⟩, ⟨69218⟩⟩] } }
def leftRaw : List Term := Proof.Events941.exact240905RawTerms
def rightRaw : List Term := Proof.Events940.exact240719RawTerms
def group : MergeGroup := .operator 240905 240719
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 240905) (leftOrdinal := 1)
    (rightResult := 240719) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩, ⟨.program ⟨257⟩, ⟨69218⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩, ⟨.program ⟨257⟩, ⟨69218⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩, ⟨.program ⟨257⟩, ⟨69218⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge240910

namespace LeftMerge240918
def owner : Owner := ⟨.program ⟨257⟩, ⟨70021⟩⟩
def mergeEvent : Nat := 240918
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7188⟩⟩, ⟨.program ⟨257⟩, ⟨70019⟩⟩] } }
def leftRaw : List Term := Proof.Events941.exact240912RawTerms
def rightRaw : List Term := Proof.Events939.exact240635RawTerms
def group : MergeGroup := .operator 240912 240635
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 240912) (leftOrdinal := 0)
    (rightResult := 240635) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7188⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨70019⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7188⟩⟩, ⟨.program ⟨257⟩, ⟨70019⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge240918

namespace LeftMerge240919
def owner : Owner := ⟨.program ⟨257⟩, ⟨70021⟩⟩
def mergeEvent : Nat := 240919
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨65772⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70019⟩⟩] } }
def leftRaw : List Term := Proof.Events941.exact240912RawTerms
def rightRaw : List Term := Proof.Events939.exact240635RawTerms
def group : MergeGroup := .operator 240912 240635
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 240912) (leftOrdinal := 1)
    (rightResult := 240635) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨65772⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨70019⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨65772⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70019⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge240919

namespace LeftMerge240921
def owner : Owner := ⟨.program ⟨257⟩, ⟨70021⟩⟩
def mergeEvent : Nat := 240921
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨65772⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68664⟩⟩] } }
def rhsRaw : List Term := Proof.Events939.exact240632RawTerms
def group : MergeGroup := .relation 240920
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 240920) (rhsResult := 240632)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨65772⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70019⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨70019⟩⟩) ⟨68664⟩ 240632) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68664⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨65772⟩⟩], [⟨.program ⟨257⟩, ⟨68664⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge240921

namespace LeftMerge240935
def owner : Owner := ⟨.program ⟨257⟩, ⟨68040⟩⟩
def mergeEvent : Nat := 240935
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68037⟩⟩] } }
def leftRaw : List Term := Proof.Events925.exact236870RawTerms
def rightRaw : List Term := Proof.Events941.exact240929RawTerms
def group : MergeGroup := .operator 236870 240929
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 236870) (leftOrdinal := 0)
    (rightResult := 240929) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68037⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68037⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge240935

namespace LeftMerge241056
def owner : Owner := ⟨.program ⟨257⟩, ⟨69001⟩⟩
def mergeEvent : Nat := 241056
def frameStart : Nat := 240990
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨65772⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events941.exact241052RawTerms
def rightRaw : List Term := Proof.Events941.exact241050RawTerms
def group : MergeGroup := .operator 241052 241050
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 241052) (leftOrdinal := 0)
    (rightResult := 241050) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨65772⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨65772⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge241056

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
