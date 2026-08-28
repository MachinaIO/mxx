import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge91078
def owner : Owner := ⟨.program ⟨257⟩, ⟨14860⟩⟩
def mergeEvent : Nat := 91078
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩] } }
def leftRaw : List Term := Proof.Events355.exact91069RawTerms
def rightRaw : List Term := Proof.Events068.exact17611RawTerms
def group : MergeGroup := .operator 91069 17611
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 91069) (leftOrdinal := 0)
    (rightResult := 17611) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9562⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge91078

namespace LeftMerge91083
def owner : Owner := ⟨.program ⟨257⟩, ⟨45281⟩⟩
def mergeEvent : Nat := 91083
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨14856⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7284⟩⟩] } }
def leftRaw : List Term := Proof.Events355.exact91079RawTerms
def rightRaw : List Term := Proof.Events355.exact91049RawTerms
def group : MergeGroup := .operator 91079 91049
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 91079) (leftOrdinal := 1)
    (rightResult := 91049) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨14856⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7284⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨14856⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7284⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨14856⟩⟩], [⟨.program ⟨257⟩, ⟨7284⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge91083

namespace LeftMerge91091
def owner : Owner := ⟨.program ⟨257⟩, ⟨47035⟩⟩
def mergeEvent : Nat := 91091
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨14856⟩⟩, ⟨.program ⟨257⟩, ⟨45274⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨47034⟩⟩] } }
def leftRaw : List Term := Proof.Events355.exact91085RawTerms
def rightRaw : List Term := Proof.Events355.exact91021RawTerms
def group : MergeGroup := .operator 91085 91021
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 91085) (leftOrdinal := 1)
    (rightResult := 91021) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨14856⟩⟩, ⟨.program ⟨257⟩, ⟨45274⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨47034⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨14856⟩⟩, ⟨.program ⟨257⟩, ⟨45274⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨47034⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge91091

namespace LeftMerge91093
def owner : Owner := ⟨.program ⟨257⟩, ⟨47035⟩⟩
def mergeEvent : Nat := 91093
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨14856⟩⟩, ⟨.program ⟨257⟩, ⟨45274⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46499⟩⟩] } }
def rhsRaw : List Term := Proof.Events355.exact91018RawTerms
def group : MergeGroup := .relation 91092
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 91092) (rhsResult := 91018)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨14856⟩⟩, ⟨.program ⟨257⟩, ⟨45274⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨47034⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨47034⟩⟩) ⟨46499⟩ 91018) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨46499⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨14856⟩⟩, ⟨.program ⟨257⟩, ⟨45274⟩⟩], [⟨.program ⟨257⟩, ⟨46499⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge91093

namespace LeftMerge91094
def owner : Owner := ⟨.program ⟨257⟩, ⟨47035⟩⟩
def mergeEvent : Nat := 91094
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨47034⟩⟩] } }
def leftRaw : List Term := Proof.Events355.exact91085RawTerms
def rightRaw : List Term := Proof.Events355.exact91021RawTerms
def group : MergeGroup := .operator 91085 91021
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 91085) (leftOrdinal := 0)
    (rightResult := 91021) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨47034⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨47034⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge91094

namespace LeftMerge91108
def owner : Owner := ⟨.program ⟨257⟩, ⟨45962⟩⟩
def mergeEvent : Nat := 91108
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨45959⟩⟩] } }
def leftRaw : List Term := Proof.Events353.exact90620RawTerms
def rightRaw : List Term := Proof.Events355.exact91102RawTerms
def group : MergeGroup := .operator 90620 91102
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 90620) (leftOrdinal := 0)
    (rightResult := 91102) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨45959⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨45959⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge91108

namespace LeftMerge91187
def owner : Owner := ⟨.program ⟨257⟩, ⟨45275⟩⟩
def mergeEvent : Nat := 91187
def frameStart : Nat := 91157
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14856⟩⟩, ⟨.program ⟨257⟩, ⟨45274⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events356.exact91183RawTerms
def rightRaw : List Term := Proof.Events356.exact91180RawTerms
def group : MergeGroup := .operator 91183 91180
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 91183) (leftOrdinal := 0)
    (rightResult := 91180) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14856⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨45274⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨14856⟩⟩, ⟨.program ⟨257⟩, ⟨45274⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge91187

namespace LeftMerge91217
def owner : Owner := ⟨.program ⟨257⟩, ⟨46768⟩⟩
def mergeEvent : Nat := 91217
def frameStart : Nat := 91157
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14856⟩⟩, ⟨.program ⟨257⟩, ⟨45274⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events356.exact91213RawTerms
def rightRaw : List Term := Proof.Events356.exact91211RawTerms
def group : MergeGroup := .operator 91213 91211
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 91213) (leftOrdinal := 0)
    (rightResult := 91211) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14856⟩⟩, ⟨.program ⟨257⟩, ⟨45274⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨14856⟩⟩, ⟨.program ⟨257⟩, ⟨45274⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge91217

namespace LeftMerge91240
def owner : Owner := ⟨.program ⟨257⟩, ⟨9564⟩⟩
def mergeEvent : Nat := 91240
def frameStart : Nat := 91157
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩] } }
def leftRaw : List Term := Proof.Events356.exact91236RawTerms
def rightRaw : List Term := Proof.Events356.exact91233RawTerms
def group : MergeGroup := .operator 91236 91233
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 91236) (leftOrdinal := 0)
    (rightResult := 91233) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9562⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge91240

namespace LeftMerge91249
def owner : Owner := ⟨.program ⟨257⟩, ⟨47037⟩⟩
def mergeEvent : Nat := 91249
def frameStart : Nat := 91157
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨47034⟩⟩] } }
def leftRaw : List Term := Proof.Events356.exact91245RawTerms
def rightRaw : List Term := Proof.Events356.exact91202RawTerms
def group : MergeGroup := .operator 91245 91202
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 91245) (leftOrdinal := 0)
    (rightResult := 91202) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨47034⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨47034⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge91249

namespace LeftMerge91250
def owner : Owner := ⟨.program ⟨257⟩, ⟨47037⟩⟩
def mergeEvent : Nat := 91250
def frameStart : Nat := 91157
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14856⟩⟩, ⟨.program ⟨257⟩, ⟨45274⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨47034⟩⟩] } }
def leftRaw : List Term := Proof.Events356.exact91245RawTerms
def rightRaw : List Term := Proof.Events356.exact91202RawTerms
def group : MergeGroup := .operator 91245 91202
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 91245) (leftOrdinal := 1)
    (rightResult := 91202) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14856⟩⟩, ⟨.program ⟨257⟩, ⟨45274⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨47034⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨14856⟩⟩, ⟨.program ⟨257⟩, ⟨45274⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨47034⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge91250

namespace LeftMerge91252
def owner : Owner := ⟨.program ⟨257⟩, ⟨47037⟩⟩
def mergeEvent : Nat := 91252
def frameStart : Nat := 91157
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14856⟩⟩, ⟨.program ⟨257⟩, ⟨45274⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46499⟩⟩] } }
def rhsRaw : List Term := Proof.Events356.exact91199RawTerms
def group : MergeGroup := .relation 91251
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 91251) (rhsResult := 91199)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨14856⟩⟩, ⟨.program ⟨257⟩, ⟨45274⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨47034⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨47034⟩⟩) ⟨46499⟩ 91199) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨46499⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨14856⟩⟩, ⟨.program ⟨257⟩, ⟨45274⟩⟩], [⟨.program ⟨257⟩, ⟨46499⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge91252

namespace LeftMerge91260
def owner : Owner := ⟨.program ⟨257⟩, ⟨45510⟩⟩
def mergeEvent : Nat := 91260
def frameStart : Nat := 91157
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨45508⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events356.exact91213RawTerms
def rightRaw : List Term := Proof.Events356.exact91256RawTerms
def group : MergeGroup := .operator 91213 91256
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 91213) (leftOrdinal := 0)
    (rightResult := 91256) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨45508⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨45508⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge91260

namespace LeftMerge91277
def owner : Owner := ⟨.program ⟨257⟩, ⟨45962⟩⟩
def mergeEvent : Nat := 91277
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7195⟩⟩] } }
def rhsRaw : List Term := Proof.Events356.exact91274RawTerms
def group : MergeGroup := .relation 91276
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 91276) (rhsResult := 91274)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨45959⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 91275 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨45959⟩⟩]⟩) (none) 91274) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7195⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7195⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge91277

namespace LeftMerge91278
def owner : Owner := ⟨.program ⟨257⟩, ⟨45962⟩⟩
def mergeEvent : Nat := 91278
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨47034⟩⟩] } }
def rhsRaw : List Term := Proof.Events356.exact91274RawTerms
def group : MergeGroup := .relation 91276
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 91276) (rhsResult := 91274)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨45959⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 91275 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨45959⟩⟩]⟩) (none) 91274) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨47034⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨47034⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge91278

namespace LeftMerge91279
def owner : Owner := ⟨.program ⟨257⟩, ⟨45962⟩⟩
def mergeEvent : Nat := 91279
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨14856⟩⟩, ⟨.program ⟨257⟩, ⟨45274⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46499⟩⟩] } }
def rhsRaw : List Term := Proof.Events356.exact91274RawTerms
def group : MergeGroup := .relation 91276
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 91276) (rhsResult := 91274)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨45959⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 91275 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨45959⟩⟩]⟩) (none) 91274) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14856⟩⟩, ⟨.program ⟨257⟩, ⟨45274⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46499⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨14856⟩⟩, ⟨.program ⟨257⟩, ⟨45274⟩⟩], [⟨.program ⟨257⟩, ⟨46499⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge91279

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
