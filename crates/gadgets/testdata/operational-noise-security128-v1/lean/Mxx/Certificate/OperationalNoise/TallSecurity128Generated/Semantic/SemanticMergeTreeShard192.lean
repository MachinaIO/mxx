import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard001
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard015
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard017
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard020
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard022
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard029
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard031
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard034
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard036
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard039
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard041
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard043
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard046
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard213

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace LeftOperatorMerge38819
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨31732⟩⟩
def group : MergeGroup := .operator 1167 32028
def deltas0_0 : Polynomial Owner := [LeftMerge38819.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge38819.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge38819.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge38819.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨31728⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right true false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge38819

namespace LeftOperatorMerge38841
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨31735⟩⟩
def group : MergeGroup := .operator 38835 24124
def deltas0_0 : Polynomial Owner := [LeftMerge38841.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge38841.deltaAt
def deltas0_1 : Polynomial Owner := [LeftMerge38844.delta]
theorem rows0_1 : MergeDeltasAt history frameStart owner group deltas0_1 := by
  exact .leaf LeftMerge38844.deltaAt
def deltas1_0 : Polynomial Owner := deltas0_0 ++ deltas0_1
theorem rows1_0 : MergeDeltasAt history frameStart owner group deltas1_0 := by
  exact .append rows0_0 rows0_1
abbrev deltas : Polynomial Owner := deltas1_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows1_0
def left : Polynomial Owner := LeftMerge38841.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge38841.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨31728⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge38841

namespace LeftOperatorMerge38849
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨31736⟩⟩
def group : MergeGroup := .operator 38845 38815
def deltas0_0 : Polynomial Owner := [LeftMerge38849.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge38849.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge38849.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge38849.rightRaw.map Term.toExact
def base : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨24398⟩⟩, ⟨.program ⟨257⟩, ⟨31728⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨31728⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7307⟩⟩] } }]
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨24398⟩⟩, ⟨.program ⟨257⟩, ⟨31728⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (add left right) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge38849

namespace LeftOperatorMerge14899
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨31912⟩⟩
def group : MergeGroup := .operator 14895 683
def deltas0_0 : Polynomial Owner := [LeftMerge14899.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge14899.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge14899.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge14899.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨31911⟩⟩], orderedFactors := [] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge14899

namespace LeftOperatorMerge690
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨31936⟩⟩
def group : MergeGroup := .operator 686 683
def deltas0_0 : Polynomial Owner := [LeftMerge690.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge690.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge690.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge690.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨31935⟩⟩], orderedFactors := [] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge690

namespace LeftOperatorMerge13419
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨31945⟩⟩
def group : MergeGroup := .operator 13415 683
def deltas0_0 : Polynomial Owner := [LeftMerge13419.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge13419.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge13419.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge13419.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨31944⟩⟩], orderedFactors := [] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge13419

namespace LeftOperatorMerge6687
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨31969⟩⟩
def group : MergeGroup := .operator 6683 683
def deltas0_0 : Polynomial Owner := [LeftMerge6687.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge6687.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge6687.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge6687.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨31968⟩⟩], orderedFactors := [] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge6687

namespace LeftOperatorMerge14161
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨31988⟩⟩
def group : MergeGroup := .operator 14157 683
def deltas0_0 : Polynomial Owner := [LeftMerge14161.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge14161.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge14161.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge14161.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨31987⟩⟩], orderedFactors := [] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge14161

namespace LeftOperatorMerge12671
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨32007⟩⟩
def group : MergeGroup := .operator 12667 683
def deltas0_0 : Polynomial Owner := [LeftMerge12671.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge12671.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge12671.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge12671.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32006⟩⟩], orderedFactors := [] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge12671

namespace LeftOperatorMerge5939
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨32026⟩⟩
def group : MergeGroup := .operator 5935 683
def deltas0_0 : Polynomial Owner := [LeftMerge5939.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge5939.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge5939.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge5939.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32025⟩⟩], orderedFactors := [] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge5939

namespace LeftOperatorMerge7435
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨32045⟩⟩
def group : MergeGroup := .operator 7431 683
def deltas0_0 : Polynomial Owner := [LeftMerge7435.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge7435.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge7435.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge7435.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32044⟩⟩], orderedFactors := [] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge7435

namespace LeftOperatorMerge11923
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨32064⟩⟩
def group : MergeGroup := .operator 11919 683
def deltas0_0 : Polynomial Owner := [LeftMerge11923.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge11923.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge11923.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge11923.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32063⟩⟩], orderedFactors := [] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge11923

namespace LeftOperatorMerge11175
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨32083⟩⟩
def group : MergeGroup := .operator 11171 683
def deltas0_0 : Polynomial Owner := [LeftMerge11175.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge11175.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge11175.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge11175.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32082⟩⟩], orderedFactors := [] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge11175

namespace LeftOperatorMerge10427
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨32102⟩⟩
def group : MergeGroup := .operator 10423 683
def deltas0_0 : Polynomial Owner := [LeftMerge10427.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge10427.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge10427.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge10427.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32101⟩⟩], orderedFactors := [] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge10427

namespace LeftOperatorMerge5191
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨32121⟩⟩
def group : MergeGroup := .operator 5187 683
def deltas0_0 : Polynomial Owner := [LeftMerge5191.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge5191.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge5191.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge5191.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32120⟩⟩], orderedFactors := [] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge5191

namespace LeftOperatorMerge9679
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨32140⟩⟩
def group : MergeGroup := .operator 9675 683
def deltas0_0 : Polynomial Owner := [LeftMerge9679.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge9679.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge9679.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge9679.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32139⟩⟩], orderedFactors := [] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge9679

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
