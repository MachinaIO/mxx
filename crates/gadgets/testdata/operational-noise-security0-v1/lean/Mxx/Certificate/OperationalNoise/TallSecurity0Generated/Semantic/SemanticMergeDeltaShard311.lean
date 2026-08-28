import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge50910
def owner : Owner := ⟨.program ⟨214⟩, ⟨25766⟩⟩
def mergeEvent : Nat := 50910
def frameStart : Nat := 50817
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨10350⟩⟩, ⟨.program ⟨214⟩, ⟨13358⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25763⟩⟩] } }
def leftRaw : List Term := Proof.Events198.exact50905RawTerms
def rightRaw : List Term := Proof.Events198.exact50862RawTerms
def group : MergeGroup := .operator 50905 50862
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 50905) (leftOrdinal := 1)
    (rightResult := 50862) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10350⟩⟩, ⟨.program ⟨214⟩, ⟨13358⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25763⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨10350⟩⟩, ⟨.program ⟨214⟩, ⟨13358⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25763⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge50910

namespace LeftMerge50912
def owner : Owner := ⟨.program ⟨214⟩, ⟨25766⟩⟩
def mergeEvent : Nat := 50912
def frameStart : Nat := 50817
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨10350⟩⟩, ⟨.program ⟨214⟩, ⟨13358⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23418⟩⟩] } }
def rhsRaw : List Term := Proof.Events198.exact50859RawTerms
def group : MergeGroup := .relation 50911
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 50911) (rhsResult := 50859)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨10350⟩⟩, ⟨.program ⟨214⟩, ⟨13358⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25763⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨25763⟩⟩) ⟨23418⟩ 50859) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23418⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨10350⟩⟩, ⟨.program ⟨214⟩, ⟨13358⟩⟩], [⟨.program ⟨214⟩, ⟨23418⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge50912

namespace LeftMerge50920
def owner : Owner := ⟨.program ⟨214⟩, ⟨17017⟩⟩
def mergeEvent : Nat := 50920
def frameStart : Nat := 50817
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨17015⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events198.exact50873RawTerms
def rightRaw : List Term := Proof.Events198.exact50916RawTerms
def group : MergeGroup := .operator 50873 50916
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 50873) (leftOrdinal := 0)
    (rightResult := 50916) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17015⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨17015⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge50920

namespace LeftMerge50937
def owner : Owner := ⟨.program ⟨214⟩, ⟨20255⟩⟩
def mergeEvent : Nat := 50937
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6707⟩⟩] } }
def rhsRaw : List Term := Proof.Events198.exact50934RawTerms
def group : MergeGroup := .relation 50936
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 50936) (rhsResult := 50934)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20252⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 50935 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20252⟩⟩]⟩) (none) 50934) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6707⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6707⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge50937

namespace LeftMerge50938
def owner : Owner := ⟨.program ⟨214⟩, ⟨20255⟩⟩
def mergeEvent : Nat := 50938
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6770⟩⟩, ⟨.program ⟨214⟩, ⟨7882⟩⟩, ⟨.program ⟨214⟩, ⟨25763⟩⟩] } }
def rhsRaw : List Term := Proof.Events198.exact50934RawTerms
def group : MergeGroup := .relation 50936
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 50936) (rhsResult := 50934)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20252⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 50935 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20252⟩⟩]⟩) (none) 50934) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6770⟩⟩, ⟨.program ⟨214⟩, ⟨7882⟩⟩, ⟨.program ⟨214⟩, ⟨25763⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6770⟩⟩, ⟨.program ⟨214⟩, ⟨7882⟩⟩, ⟨.program ⟨214⟩, ⟨25763⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge50938

namespace LeftMerge50939
def owner : Owner := ⟨.program ⟨214⟩, ⟨20255⟩⟩
def mergeEvent : Nat := 50939
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨10350⟩⟩, ⟨.program ⟨214⟩, ⟨13358⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23418⟩⟩] } }
def rhsRaw : List Term := Proof.Events198.exact50934RawTerms
def group : MergeGroup := .relation 50936
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 50936) (rhsResult := 50934)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20252⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 50935 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20252⟩⟩]⟩) (none) 50934) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10350⟩⟩, ⟨.program ⟨214⟩, ⟨13358⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23418⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨10350⟩⟩, ⟨.program ⟨214⟩, ⟨13358⟩⟩], [⟨.program ⟨214⟩, ⟨23418⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge50939

namespace LeftMerge50940
def owner : Owner := ⟨.program ⟨214⟩, ⟨20255⟩⟩
def mergeEvent : Nat := 50940
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨17015⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def rhsRaw : List Term := Proof.Events198.exact50934RawTerms
def group : MergeGroup := .relation 50936
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 50936) (rhsResult := 50934)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20252⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 50935 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20252⟩⟩]⟩) (none) 50934) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17015⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨17015⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge50940

namespace LeftMerge50945
def owner : Owner := ⟨.program ⟨214⟩, ⟨25765⟩⟩
def mergeEvent : Nat := 50945
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨10350⟩⟩, ⟨.program ⟨214⟩, ⟨13358⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23418⟩⟩] } }
def leftRaw : List Term := Proof.Events198.exact50941RawTerms
def rightRaw : List Term := Proof.Events198.exact50744RawTerms
def group : MergeGroup := .operator 50941 50744
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 50941) (leftOrdinal := 2)
    (rightResult := 50744) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨10350⟩⟩, ⟨.program ⟨214⟩, ⟨13358⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23418⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨10350⟩⟩, ⟨.program ⟨214⟩, ⟨13358⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23418⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨10350⟩⟩, ⟨.program ⟨214⟩, ⟨13358⟩⟩], [⟨.program ⟨214⟩, ⟨23418⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge50945

namespace LeftMerge50946
def owner : Owner := ⟨.program ⟨214⟩, ⟨25765⟩⟩
def mergeEvent : Nat := 50946
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6770⟩⟩, ⟨.program ⟨214⟩, ⟨7882⟩⟩, ⟨.program ⟨214⟩, ⟨25763⟩⟩] } }
def leftRaw : List Term := Proof.Events198.exact50941RawTerms
def rightRaw : List Term := Proof.Events198.exact50744RawTerms
def group : MergeGroup := .operator 50941 50744
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 50941) (leftOrdinal := 1)
    (rightResult := 50744) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6770⟩⟩, ⟨.program ⟨214⟩, ⟨7882⟩⟩, ⟨.program ⟨214⟩, ⟨25763⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6770⟩⟩, ⟨.program ⟨214⟩, ⟨7882⟩⟩, ⟨.program ⟨214⟩, ⟨25763⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6770⟩⟩, ⟨.program ⟨214⟩, ⟨7882⟩⟩, ⟨.program ⟨214⟩, ⟨25763⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge50946

namespace LeftMerge50954
def owner : Owner := ⟨.program ⟨214⟩, ⟨30141⟩⟩
def mergeEvent : Nat := 50954
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6707⟩⟩, ⟨.program ⟨214⟩, ⟨30139⟩⟩] } }
def leftRaw : List Term := Proof.Events199.exact50948RawTerms
def rightRaw : List Term := Proof.Events197.exact50655RawTerms
def group : MergeGroup := .operator 50948 50655
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 50948) (leftOrdinal := 0)
    (rightResult := 50655) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6707⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨30139⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6707⟩⟩, ⟨.program ⟨214⟩, ⟨30139⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge50954

namespace LeftMerge50955
def owner : Owner := ⟨.program ⟨214⟩, ⟨30141⟩⟩
def mergeEvent : Nat := 50955
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨17015⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨30139⟩⟩] } }
def leftRaw : List Term := Proof.Events199.exact50948RawTerms
def rightRaw : List Term := Proof.Events197.exact50655RawTerms
def group : MergeGroup := .operator 50948 50655
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 50948) (leftOrdinal := 1)
    (rightResult := 50655) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨17015⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨30139⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨17015⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨30139⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge50955

namespace LeftMerge50957
def owner : Owner := ⟨.program ⟨214⟩, ⟨30141⟩⟩
def mergeEvent : Nat := 50957
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨17015⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24795⟩⟩] } }
def rhsRaw : List Term := Proof.Events197.exact50652RawTerms
def group : MergeGroup := .relation 50956
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 50956) (rhsResult := 50652)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨17015⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨30139⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨30139⟩⟩) ⟨24795⟩ 50652) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨24795⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨17015⟩⟩], [⟨.program ⟨214⟩, ⟨24795⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge50957

namespace LeftMerge50971
def owner : Owner := ⟨.program ⟨214⟩, ⟨22847⟩⟩
def mergeEvent : Nat := 50971
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22844⟩⟩] } }
def leftRaw : List Term := Proof.Events198.exact50762RawTerms
def rightRaw : List Term := Proof.Events199.exact50965RawTerms
def group : MergeGroup := .operator 50762 50965
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 50762) (leftOrdinal := 0)
    (rightResult := 50965) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨22844⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22844⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge50971

namespace LeftMerge51092
def owner : Owner := ⟨.program ⟨214⟩, ⟨17057⟩⟩
def mergeEvent : Nat := 51092
def frameStart : Nat := 51026
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨17015⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events199.exact51088RawTerms
def rightRaw : List Term := Proof.Events199.exact51086RawTerms
def group : MergeGroup := .operator 51088 51086
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 51088) (leftOrdinal := 0)
    (rightResult := 51086) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17015⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨17015⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge51092

namespace LeftMerge51104
def owner : Owner := ⟨.program ⟨214⟩, ⟨30140⟩⟩
def mergeEvent : Nat := 51104
def frameStart : Nat := 51026
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6707⟩⟩, ⟨.program ⟨214⟩, ⟨30139⟩⟩] } }
def leftRaw : List Term := Proof.Events199.exact51100RawTerms
def rightRaw : List Term := Proof.Events199.exact51077RawTerms
def group : MergeGroup := .operator 51100 51077
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 51100) (leftOrdinal := 0)
    (rightResult := 51077) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6707⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨30139⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6707⟩⟩, ⟨.program ⟨214⟩, ⟨30139⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge51104

namespace LeftMerge51105
def owner : Owner := ⟨.program ⟨214⟩, ⟨30140⟩⟩
def mergeEvent : Nat := 51105
def frameStart : Nat := 51026
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨17015⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨30139⟩⟩] } }
def leftRaw : List Term := Proof.Events199.exact51100RawTerms
def rightRaw : List Term := Proof.Events199.exact51077RawTerms
def group : MergeGroup := .operator 51100 51077
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 51100) (leftOrdinal := 1)
    (rightResult := 51077) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17015⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨30139⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨17015⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨30139⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge51105

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
