import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge283006
def owner : Owner := ⟨.program ⟨257⟩, ⟨39160⟩⟩
def mergeEvent : Nat := 283006
def frameStart : Nat := 282927
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨37380⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨39159⟩⟩] } }
def leftRaw : List Term := Proof.Events1105.exact283001RawTerms
def rightRaw : List Term := Proof.Events1105.exact282978RawTerms
def group : MergeGroup := .operator 283001 282978
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 283001) (leftOrdinal := 1)
    (rightResult := 282978) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨37380⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨39159⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨37380⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨39159⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge283006

namespace LeftMerge283008
def owner : Owner := ⟨.program ⟨257⟩, ⟨39160⟩⟩
def mergeEvent : Nat := 283008
def frameStart : Nat := 282927
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨37380⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38527⟩⟩] } }
def rhsRaw : List Term := Proof.Events1105.exact282975RawTerms
def group : MergeGroup := .relation 283007
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 283007) (rhsResult := 282975)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨37380⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨39159⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨39159⟩⟩) ⟨38527⟩ 282975) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨38527⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨37380⟩⟩], [⟨.program ⟨257⟩, ⟨38527⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge283008

namespace LeftMerge283016
def owner : Owner := ⟨.program ⟨257⟩, ⟨37566⟩⟩
def mergeEvent : Nat := 283016
def frameStart : Nat := 282927
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨37565⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1105.exact282989RawTerms
def rightRaw : List Term := Proof.Events1105.exact283012RawTerms
def group : MergeGroup := .operator 282989 283012
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 282989) (leftOrdinal := 0)
    (rightResult := 283012) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨37565⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨37565⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge283016

namespace LeftMerge283033
def owner : Owner := ⟨.program ⟨257⟩, ⟨38059⟩⟩
def mergeEvent : Nat := 283033
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7224⟩⟩] } }
def rhsRaw : List Term := Proof.Events1105.exact283030RawTerms
def group : MergeGroup := .relation 283032
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 283032) (rhsResult := 283030)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨38056⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 283031 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨38056⟩⟩]⟩) (none) 283030) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7224⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7224⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge283033

namespace LeftMerge283034
def owner : Owner := ⟨.program ⟨257⟩, ⟨38059⟩⟩
def mergeEvent : Nat := 283034
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7192⟩⟩, ⟨.program ⟨257⟩, ⟨39159⟩⟩] } }
def rhsRaw : List Term := Proof.Events1105.exact283030RawTerms
def group : MergeGroup := .relation 283032
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 283032) (rhsResult := 283030)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨38056⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 283031 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨38056⟩⟩]⟩) (none) 283030) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7192⟩⟩, ⟨.program ⟨257⟩, ⟨39159⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7192⟩⟩, ⟨.program ⟨257⟩, ⟨39159⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge283034

namespace LeftMerge283035
def owner : Owner := ⟨.program ⟨257⟩, ⟨38059⟩⟩
def mergeEvent : Nat := 283035
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨37380⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38527⟩⟩] } }
def rhsRaw : List Term := Proof.Events1105.exact283030RawTerms
def group : MergeGroup := .relation 283032
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 283032) (rhsResult := 283030)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨38056⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 283031 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨38056⟩⟩]⟩) (none) 283030) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨37380⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38527⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨37380⟩⟩], [⟨.program ⟨257⟩, ⟨38527⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge283035

namespace LeftMerge283036
def owner : Owner := ⟨.program ⟨257⟩, ⟨38059⟩⟩
def mergeEvent : Nat := 283036
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨37565⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events1105.exact283030RawTerms
def group : MergeGroup := .relation 283032
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 283032) (rhsResult := 283030)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨38056⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 283031 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨38056⟩⟩]⟩) (none) 283030) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨37565⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨37565⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge283036

namespace LeftMerge283041
def owner : Owner := ⟨.program ⟨257⟩, ⟨39162⟩⟩
def mergeEvent : Nat := 283041
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7192⟩⟩, ⟨.program ⟨257⟩, ⟨39159⟩⟩] } }
def leftRaw : List Term := Proof.Events1105.exact283037RawTerms
def rightRaw : List Term := Proof.Events1104.exact282859RawTerms
def group : MergeGroup := .operator 283037 282859
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 283037) (leftOrdinal := 0)
    (rightResult := 282859) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7192⟩⟩, ⟨.program ⟨257⟩, ⟨39159⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7192⟩⟩, ⟨.program ⟨257⟩, ⟨39159⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7192⟩⟩, ⟨.program ⟨257⟩, ⟨39159⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge283041

namespace LeftMerge283042
def owner : Owner := ⟨.program ⟨257⟩, ⟨39162⟩⟩
def mergeEvent : Nat := 283042
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨37380⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38527⟩⟩] } }
def leftRaw : List Term := Proof.Events1105.exact283037RawTerms
def rightRaw : List Term := Proof.Events1104.exact282859RawTerms
def group : MergeGroup := .operator 283037 282859
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 283037) (leftOrdinal := 2)
    (rightResult := 282859) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨37380⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38527⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨37380⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38527⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨37380⟩⟩], [⟨.program ⟨257⟩, ⟨38527⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge283042

namespace LeftMerge283068
def owner : Owner := ⟨.program ⟨257⟩, ⟨34293⟩⟩
def mergeEvent : Nat := 283068
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨34290⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events053.exact13667RawTerms
def rightRaw : List Term := Proof.Events1096.exact280653RawTerms
def group : MergeGroup := .operator 13667 280653
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 13667) (leftOrdinal := 0)
    (rightResult := 280653) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨34290⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨34290⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge283068

namespace LeftMerge283073
def owner : Owner := ⟨.program ⟨257⟩, ⟨7902⟩⟩
def mergeEvent : Nat := 283073
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7280⟩⟩] } }
def leftRaw : List Term := Proof.Events1095.exact280523RawTerms
def rightRaw : List Term := Proof.Events076.exact19585RawTerms
def group : MergeGroup := .operator 280523 19585
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 280523) (leftOrdinal := 0)
    (rightResult := 19585) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7280⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7280⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge283073

namespace LeftMerge283090
def owner : Owner := ⟨.program ⟨257⟩, ⟨34296⟩⟩
def mergeEvent : Nat := 283090
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨13491⟩⟩, ⟨.program ⟨257⟩, ⟨34290⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1105.exact283084RawTerms
def rightRaw : List Term := Proof.Events053.exact13670RawTerms
def group : MergeGroup := .operator 283084 13670
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 283084) (leftOrdinal := 1)
    (rightResult := 13670) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨34290⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13491⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨13491⟩⟩, ⟨.program ⟨257⟩, ⟨34290⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge283090

namespace LeftMerge283091
def owner : Owner := ⟨.program ⟨257⟩, ⟨34296⟩⟩
def mergeEvent : Nat := 283091
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨13491⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7280⟩⟩] } }
def leftRaw : List Term := Proof.Events1105.exact283084RawTerms
def rightRaw : List Term := Proof.Events053.exact13670RawTerms
def group : MergeGroup := .operator 283084 13670
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 283084) (leftOrdinal := 0)
    (rightResult := 13670) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7280⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13491⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨13491⟩⟩], [⟨.program ⟨257⟩, ⟨7280⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge283091

namespace LeftMerge283096
def owner : Owner := ⟨.program ⟨257⟩, ⟨13492⟩⟩
def mergeEvent : Nat := 283096
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨13491⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events053.exact13670RawTerms
def rightRaw : List Term := Proof.Events1096.exact280653RawTerms
def group : MergeGroup := .operator 13670 280653
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 13670) (leftOrdinal := 0)
    (rightResult := 280653) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13491⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨13491⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge283096

namespace LeftMerge283101
def owner : Owner := ⟨.program ⟨257⟩, ⟨7919⟩⟩
def mergeEvent : Nat := 283101
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7297⟩⟩] } }
def leftRaw : List Term := Proof.Events1095.exact280523RawTerms
def rightRaw : List Term := Proof.Events076.exact19626RawTerms
def group : MergeGroup := .operator 280523 19626
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 280523) (leftOrdinal := 0)
    (rightResult := 19626) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7297⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7297⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge283101

namespace LeftMerge283118
def owner : Owner := ⟨.program ⟨257⟩, ⟨13495⟩⟩
def mergeEvent : Nat := 283118
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨13491⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩] } }
def leftRaw : List Term := Proof.Events1105.exact283112RawTerms
def rightRaw : List Term := Proof.Events076.exact19615RawTerms
def group : MergeGroup := .operator 283112 19615
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 283112) (leftOrdinal := 1)
    (rightResult := 19615) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨13491⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9550⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨13491⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge283118

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
