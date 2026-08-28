import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge61797
def owner : Owner := ⟨.program ⟨257⟩, ⟨45328⟩⟩
def mergeEvent : Nat := 61797
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨14886⟩⟩, ⟨.program ⟨257⟩, ⟨45322⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events241.exact61791RawTerms
def rightRaw : List Term := Proof.Events009.exact2364RawTerms
def group : MergeGroup := .operator 61791 2364
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 61791) (leftOrdinal := 1)
    (rightResult := 2364) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨45322⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14886⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨14886⟩⟩, ⟨.program ⟨257⟩, ⟨45322⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge61797

namespace LeftMerge61798
def owner : Owner := ⟨.program ⟨257⟩, ⟨45328⟩⟩
def mergeEvent : Nat := 61798
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨14886⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7284⟩⟩] } }
def leftRaw : List Term := Proof.Events241.exact61791RawTerms
def rightRaw : List Term := Proof.Events009.exact2364RawTerms
def group : MergeGroup := .operator 61791 2364
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 61791) (leftOrdinal := 0)
    (rightResult := 2364) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7284⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14886⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨14886⟩⟩], [⟨.program ⟨257⟩, ⟨7284⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge61798

namespace LeftMerge61803
def owner : Owner := ⟨.program ⟨257⟩, ⟨14887⟩⟩
def mergeEvent : Nat := 61803
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨14886⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events009.exact2364RawTerms
def rightRaw : List Term := Proof.Events239.exact61278RawTerms
def group : MergeGroup := .operator 2364 61278
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 2364) (leftOrdinal := 0)
    (rightResult := 61278) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14886⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨14886⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge61803

namespace LeftMerge61808
def owner : Owner := ⟨.program ⟨257⟩, ⟨10783⟩⟩
def mergeEvent : Nat := 61808
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩] } }
def leftRaw : List Term := Proof.Events238.exact61148RawTerms
def rightRaw : List Term := Proof.Events068.exact17622RawTerms
def group : MergeGroup := .operator 61148 17622
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 61148) (leftOrdinal := 0)
    (rightResult := 17622) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7301⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge61808

namespace LeftMerge61825
def owner : Owner := ⟨.program ⟨257⟩, ⟨14890⟩⟩
def mergeEvent : Nat := 61825
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨14886⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩] } }
def leftRaw : List Term := Proof.Events241.exact61819RawTerms
def rightRaw : List Term := Proof.Events068.exact17611RawTerms
def group : MergeGroup := .operator 61819 17611
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 61819) (leftOrdinal := 1)
    (rightResult := 17611) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨14886⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9562⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨14886⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge61825

namespace LeftMerge61827
def owner : Owner := ⟨.program ⟨257⟩, ⟨14890⟩⟩
def mergeEvent : Nat := 61827
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨14886⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7284⟩⟩] } }
def rhsRaw : List Term := Proof.Events068.exact17581RawTerms
def group : MergeGroup := .relation 61826
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 61826) (rhsResult := 17581)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨14886⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨9562⟩⟩) ⟨7284⟩ 17581) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7284⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨14886⟩⟩], [⟨.program ⟨257⟩, ⟨7284⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge61827

namespace LeftMerge61828
def owner : Owner := ⟨.program ⟨257⟩, ⟨14890⟩⟩
def mergeEvent : Nat := 61828
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩] } }
def leftRaw : List Term := Proof.Events241.exact61819RawTerms
def rightRaw : List Term := Proof.Events068.exact17611RawTerms
def group : MergeGroup := .operator 61819 17611
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 61819) (leftOrdinal := 0)
    (rightResult := 17611) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9562⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge61828

namespace LeftMerge61833
def owner : Owner := ⟨.program ⟨257⟩, ⟨45329⟩⟩
def mergeEvent : Nat := 61833
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨14886⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7284⟩⟩] } }
def leftRaw : List Term := Proof.Events241.exact61829RawTerms
def rightRaw : List Term := Proof.Events241.exact61799RawTerms
def group : MergeGroup := .operator 61829 61799
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 61829) (leftOrdinal := 1)
    (rightResult := 61799) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨14886⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7284⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨14886⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7284⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨14886⟩⟩], [⟨.program ⟨257⟩, ⟨7284⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge61833

namespace LeftMerge61841
def owner : Owner := ⟨.program ⟨257⟩, ⟨47057⟩⟩
def mergeEvent : Nat := 61841
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨14886⟩⟩, ⟨.program ⟨257⟩, ⟨45322⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨47056⟩⟩] } }
def leftRaw : List Term := Proof.Events241.exact61835RawTerms
def rightRaw : List Term := Proof.Events241.exact61771RawTerms
def group : MergeGroup := .operator 61835 61771
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 61835) (leftOrdinal := 1)
    (rightResult := 61771) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨14886⟩⟩, ⟨.program ⟨257⟩, ⟨45322⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨47056⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨14886⟩⟩, ⟨.program ⟨257⟩, ⟨45322⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨47056⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge61841

namespace LeftMerge61843
def owner : Owner := ⟨.program ⟨257⟩, ⟨47057⟩⟩
def mergeEvent : Nat := 61843
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨14886⟩⟩, ⟨.program ⟨257⟩, ⟨45322⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46511⟩⟩] } }
def rhsRaw : List Term := Proof.Events241.exact61768RawTerms
def group : MergeGroup := .relation 61842
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 61842) (rhsResult := 61768)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨14886⟩⟩, ⟨.program ⟨257⟩, ⟨45322⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨47056⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨47056⟩⟩) ⟨46511⟩ 61768) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨46511⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨14886⟩⟩, ⟨.program ⟨257⟩, ⟨45322⟩⟩], [⟨.program ⟨257⟩, ⟨46511⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge61843

namespace LeftMerge61844
def owner : Owner := ⟨.program ⟨257⟩, ⟨47057⟩⟩
def mergeEvent : Nat := 61844
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨47056⟩⟩] } }
def leftRaw : List Term := Proof.Events241.exact61835RawTerms
def rightRaw : List Term := Proof.Events241.exact61771RawTerms
def group : MergeGroup := .operator 61835 61771
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 61835) (leftOrdinal := 0)
    (rightResult := 61771) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨47056⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨47056⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge61844

namespace LeftMerge61858
def owner : Owner := ⟨.program ⟨257⟩, ⟨45982⟩⟩
def mergeEvent : Nat := 61858
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨45979⟩⟩] } }
def leftRaw : List Term := Proof.Events239.exact61370RawTerms
def rightRaw : List Term := Proof.Events241.exact61852RawTerms
def group : MergeGroup := .operator 61370 61852
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 61370) (leftOrdinal := 0)
    (rightResult := 61852) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨45979⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨45979⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge61858

namespace LeftMerge61937
def owner : Owner := ⟨.program ⟨257⟩, ⟨45323⟩⟩
def mergeEvent : Nat := 61937
def frameStart : Nat := 61907
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14886⟩⟩, ⟨.program ⟨257⟩, ⟨45322⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events241.exact61933RawTerms
def rightRaw : List Term := Proof.Events241.exact61930RawTerms
def group : MergeGroup := .operator 61933 61930
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 61933) (leftOrdinal := 0)
    (rightResult := 61930) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14886⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨45322⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨14886⟩⟩, ⟨.program ⟨257⟩, ⟨45322⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge61937

namespace LeftMerge61967
def owner : Owner := ⟨.program ⟨257⟩, ⟨46776⟩⟩
def mergeEvent : Nat := 61967
def frameStart : Nat := 61907
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14886⟩⟩, ⟨.program ⟨257⟩, ⟨45322⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events242.exact61963RawTerms
def rightRaw : List Term := Proof.Events242.exact61961RawTerms
def group : MergeGroup := .operator 61963 61961
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 61963) (leftOrdinal := 0)
    (rightResult := 61961) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14886⟩⟩, ⟨.program ⟨257⟩, ⟨45322⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨14886⟩⟩, ⟨.program ⟨257⟩, ⟨45322⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge61967

namespace LeftMerge61990
def owner : Owner := ⟨.program ⟨257⟩, ⟨9564⟩⟩
def mergeEvent : Nat := 61990
def frameStart : Nat := 61907
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩] } }
def leftRaw : List Term := Proof.Events242.exact61986RawTerms
def rightRaw : List Term := Proof.Events242.exact61983RawTerms
def group : MergeGroup := .operator 61986 61983
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 61986) (leftOrdinal := 0)
    (rightResult := 61983) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9562⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge61990

namespace LeftMerge61999
def owner : Owner := ⟨.program ⟨257⟩, ⟨47059⟩⟩
def mergeEvent : Nat := 61999
def frameStart : Nat := 61907
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨47056⟩⟩] } }
def leftRaw : List Term := Proof.Events242.exact61995RawTerms
def rightRaw : List Term := Proof.Events242.exact61952RawTerms
def group : MergeGroup := .operator 61995 61952
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 61995) (leftOrdinal := 0)
    (rightResult := 61952) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨47056⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨47056⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge61999

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
