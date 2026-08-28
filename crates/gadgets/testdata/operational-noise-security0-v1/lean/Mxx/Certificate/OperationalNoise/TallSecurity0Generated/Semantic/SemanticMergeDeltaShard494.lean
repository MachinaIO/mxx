import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge79982
def owner : Owner := ⟨.program ⟨214⟩, ⟨13357⟩⟩
def mergeEvent : Nat := 79982
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨10345⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6790⟩⟩] } }
def leftRaw : List Term := Proof.Events312.exact79978RawTerms
def rightRaw : List Term := Proof.Events312.exact79948RawTerms
def group : MergeGroup := .operator 79978 79948
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 79978) (leftOrdinal := 1)
    (rightResult := 79948) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨10345⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6790⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨10345⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6790⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨10345⟩⟩], [⟨.program ⟨214⟩, ⟨6790⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge79982

namespace LeftMerge79990
def owner : Owner := ⟨.program ⟨214⟩, ⟨25759⟩⟩
def mergeEvent : Nat := 79990
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨10345⟩⟩, ⟨.program ⟨214⟩, ⟨13350⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25758⟩⟩] } }
def leftRaw : List Term := Proof.Events312.exact79984RawTerms
def rightRaw : List Term := Proof.Events312.exact79915RawTerms
def group : MergeGroup := .operator 79984 79915
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 79984) (leftOrdinal := 1)
    (rightResult := 79915) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨10345⟩⟩, ⟨.program ⟨214⟩, ⟨13350⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25758⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨10345⟩⟩, ⟨.program ⟨214⟩, ⟨13350⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25758⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge79990

namespace LeftMerge79992
def owner : Owner := ⟨.program ⟨214⟩, ⟨25759⟩⟩
def mergeEvent : Nat := 79992
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨10345⟩⟩, ⟨.program ⟨214⟩, ⟨13350⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23416⟩⟩] } }
def rhsRaw : List Term := Proof.Events312.exact79912RawTerms
def group : MergeGroup := .relation 79991
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 79991) (rhsResult := 79912)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨10345⟩⟩, ⟨.program ⟨214⟩, ⟨13350⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25758⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨25758⟩⟩) ⟨23416⟩ 79912) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23416⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨10345⟩⟩, ⟨.program ⟨214⟩, ⟨13350⟩⟩], [⟨.program ⟨214⟩, ⟨23416⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge79992

namespace LeftMerge79993
def owner : Owner := ⟨.program ⟨214⟩, ⟨25759⟩⟩
def mergeEvent : Nat := 79993
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6770⟩⟩, ⟨.program ⟨214⟩, ⟨7882⟩⟩, ⟨.program ⟨214⟩, ⟨25758⟩⟩] } }
def leftRaw : List Term := Proof.Events312.exact79984RawTerms
def rightRaw : List Term := Proof.Events312.exact79915RawTerms
def group : MergeGroup := .operator 79984 79915
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 79984) (leftOrdinal := 0)
    (rightResult := 79915) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6770⟩⟩, ⟨.program ⟨214⟩, ⟨7882⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25758⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6770⟩⟩, ⟨.program ⟨214⟩, ⟨7882⟩⟩, ⟨.program ⟨214⟩, ⟨25758⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge79993

namespace LeftMerge80005
def owner : Owner := ⟨.program ⟨214⟩, ⟨5540⟩⟩
def mergeEvent : Nat := 80005
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩] } }
def leftRaw : List Term := Proof.Events311.exact79790RawTerms
def rightRaw : List Term := Proof.Events025.exact6550RawTerms
def group : MergeGroup := .operator 79790 6550
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 79790) (leftOrdinal := 0)
    (rightResult := 6550) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge80005

namespace LeftMerge80018
def owner : Owner := ⟨.program ⟨214⟩, ⟨20251⟩⟩
def mergeEvent : Nat := 80018
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20248⟩⟩] } }
def leftRaw : List Term := Proof.Events312.exact80012RawTerms
def rightRaw : List Term := Proof.Events312.exact80001RawTerms
def group : MergeGroup := .operator 80012 80001
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 80012) (leftOrdinal := 0)
    (rightResult := 80001) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨20248⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20248⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge80018

namespace LeftMerge80097
def owner : Owner := ⟨.program ⟨214⟩, ⟨13351⟩⟩
def mergeEvent : Nat := 80097
def frameStart : Nat := 80067
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨10345⟩⟩, ⟨.program ⟨214⟩, ⟨13350⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events312.exact80093RawTerms
def rightRaw : List Term := Proof.Events312.exact80090RawTerms
def group : MergeGroup := .operator 80093 80090
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 80093) (leftOrdinal := 0)
    (rightResult := 80090) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10345⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨13350⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨10345⟩⟩, ⟨.program ⟨214⟩, ⟨13350⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge80097

namespace LeftMerge80127
def owner : Owner := ⟨.program ⟨214⟩, ⟨13448⟩⟩
def mergeEvent : Nat := 80127
def frameStart : Nat := 80067
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨10345⟩⟩, ⟨.program ⟨214⟩, ⟨13350⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events312.exact80123RawTerms
def rightRaw : List Term := Proof.Events312.exact80121RawTerms
def group : MergeGroup := .operator 80123 80121
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 80123) (leftOrdinal := 0)
    (rightResult := 80121) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10345⟩⟩, ⟨.program ⟨214⟩, ⟨13350⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨10345⟩⟩, ⟨.program ⟨214⟩, ⟨13350⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge80127

namespace LeftMerge80148
def owner : Owner := ⟨.program ⟨214⟩, ⟨7884⟩⟩
def mergeEvent : Nat := 80148
def frameStart : Nat := 80067
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6770⟩⟩, ⟨.program ⟨214⟩, ⟨7882⟩⟩] } }
def leftRaw : List Term := Proof.Events313.exact80144RawTerms
def rightRaw : List Term := Proof.Events313.exact80141RawTerms
def group : MergeGroup := .operator 80144 80141
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 80144) (leftOrdinal := 0)
    (rightResult := 80141) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6770⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨7882⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6770⟩⟩, ⟨.program ⟨214⟩, ⟨7882⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge80148

namespace LeftMerge80157
def owner : Owner := ⟨.program ⟨214⟩, ⟨25761⟩⟩
def mergeEvent : Nat := 80157
def frameStart : Nat := 80067
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6770⟩⟩, ⟨.program ⟨214⟩, ⟨7882⟩⟩, ⟨.program ⟨214⟩, ⟨25758⟩⟩] } }
def leftRaw : List Term := Proof.Events313.exact80153RawTerms
def rightRaw : List Term := Proof.Events312.exact80112RawTerms
def group : MergeGroup := .operator 80153 80112
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 80153) (leftOrdinal := 0)
    (rightResult := 80112) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6770⟩⟩, ⟨.program ⟨214⟩, ⟨7882⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25758⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6770⟩⟩, ⟨.program ⟨214⟩, ⟨7882⟩⟩, ⟨.program ⟨214⟩, ⟨25758⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge80157

namespace LeftMerge80158
def owner : Owner := ⟨.program ⟨214⟩, ⟨25761⟩⟩
def mergeEvent : Nat := 80158
def frameStart : Nat := 80067
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨10345⟩⟩, ⟨.program ⟨214⟩, ⟨13350⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25758⟩⟩] } }
def leftRaw : List Term := Proof.Events313.exact80153RawTerms
def rightRaw : List Term := Proof.Events312.exact80112RawTerms
def group : MergeGroup := .operator 80153 80112
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 80153) (leftOrdinal := 1)
    (rightResult := 80112) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10345⟩⟩, ⟨.program ⟨214⟩, ⟨13350⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25758⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨10345⟩⟩, ⟨.program ⟨214⟩, ⟨13350⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25758⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge80158

namespace LeftMerge80160
def owner : Owner := ⟨.program ⟨214⟩, ⟨25761⟩⟩
def mergeEvent : Nat := 80160
def frameStart : Nat := 80067
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨10345⟩⟩, ⟨.program ⟨214⟩, ⟨13350⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23416⟩⟩] } }
def rhsRaw : List Term := Proof.Events312.exact80109RawTerms
def group : MergeGroup := .relation 80159
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 80159) (rhsResult := 80109)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨10345⟩⟩, ⟨.program ⟨214⟩, ⟨13350⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25758⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨25758⟩⟩) ⟨23416⟩ 80109) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23416⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨10345⟩⟩, ⟨.program ⟨214⟩, ⟨13350⟩⟩], [⟨.program ⟨214⟩, ⟨23416⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge80160

namespace LeftMerge80168
def owner : Owner := ⟨.program ⟨214⟩, ⟨17013⟩⟩
def mergeEvent : Nat := 80168
def frameStart : Nat := 80067
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨17011⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events312.exact80123RawTerms
def rightRaw : List Term := Proof.Events313.exact80164RawTerms
def group : MergeGroup := .operator 80123 80164
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 80123) (leftOrdinal := 0)
    (rightResult := 80164) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17011⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨17011⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge80168

namespace LeftMerge80185
def owner : Owner := ⟨.program ⟨214⟩, ⟨20251⟩⟩
def mergeEvent : Nat := 80185
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6707⟩⟩] } }
def rhsRaw : List Term := Proof.Events313.exact80182RawTerms
def group : MergeGroup := .relation 80184
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 80184) (rhsResult := 80182)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20248⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 80183 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20248⟩⟩]⟩) (none) 80182) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6707⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6707⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge80185

namespace LeftMerge80186
def owner : Owner := ⟨.program ⟨214⟩, ⟨20251⟩⟩
def mergeEvent : Nat := 80186
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6770⟩⟩, ⟨.program ⟨214⟩, ⟨7882⟩⟩, ⟨.program ⟨214⟩, ⟨25758⟩⟩] } }
def rhsRaw : List Term := Proof.Events313.exact80182RawTerms
def group : MergeGroup := .relation 80184
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 80184) (rhsResult := 80182)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20248⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 80183 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20248⟩⟩]⟩) (none) 80182) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6770⟩⟩, ⟨.program ⟨214⟩, ⟨7882⟩⟩, ⟨.program ⟨214⟩, ⟨25758⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6770⟩⟩, ⟨.program ⟨214⟩, ⟨7882⟩⟩, ⟨.program ⟨214⟩, ⟨25758⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge80186

namespace LeftMerge80187
def owner : Owner := ⟨.program ⟨214⟩, ⟨20251⟩⟩
def mergeEvent : Nat := 80187
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨10345⟩⟩, ⟨.program ⟨214⟩, ⟨13350⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23416⟩⟩] } }
def rhsRaw : List Term := Proof.Events313.exact80182RawTerms
def group : MergeGroup := .relation 80184
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 80184) (rhsResult := 80182)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20248⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 80183 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20248⟩⟩]⟩) (none) 80182) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10345⟩⟩, ⟨.program ⟨214⟩, ⟨13350⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23416⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨10345⟩⟩, ⟨.program ⟨214⟩, ⟨13350⟩⟩], [⟨.program ⟨214⟩, ⟨23416⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge80187

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
