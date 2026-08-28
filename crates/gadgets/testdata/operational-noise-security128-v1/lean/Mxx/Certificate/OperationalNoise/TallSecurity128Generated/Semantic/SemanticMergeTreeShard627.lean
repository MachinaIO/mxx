import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard1470
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard1471
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard1472
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard1473
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard1474

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace LeftOperatorMerge238454
def frameStart : Nat := 238371
def owner : Owner := ⟨.program ⟨257⟩, ⟨9558⟩⟩
def group : MergeGroup := .operator 238450 238447
def deltas0_0 : Polynomial Owner := [LeftMerge238454.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge238454.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge238454.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge238454.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge238454

namespace LeftOperatorMerge238401
def frameStart : Nat := 238371
def owner : Owner := ⟨.program ⟨257⟩, ⟨39747⟩⟩
def group : MergeGroup := .operator 238397 238394
def deltas0_0 : Polynomial Owner := [LeftMerge238401.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge238401.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge238401.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge238401.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14151⟩⟩, ⟨.program ⟨257⟩, ⟨39746⟩⟩], orderedFactors := [] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge238401

namespace LeftOperatorMerge238474
def frameStart : Nat := 238371
def owner : Owner := ⟨.program ⟨257⟩, ⟨40094⟩⟩
def group : MergeGroup := .operator 238427 238470
def deltas0_0 : Polynomial Owner := [LeftMerge238474.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge238474.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge238474.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge238474.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨40092⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge238474

namespace LeftOperatorMerge238431
def frameStart : Nat := 238371
def owner : Owner := ⟨.program ⟨257⟩, ⟨41380⟩⟩
def group : MergeGroup := .operator 238427 238425
def deltas0_0 : Polynomial Owner := [LeftMerge238431.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge238431.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge238431.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge238431.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14151⟩⟩, ⟨.program ⟨257⟩, ⟨39746⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge238431

namespace LeftOperatorMerge238463
def frameStart : Nat := 238371
def owner : Owner := ⟨.program ⟨257⟩, ⟨41600⟩⟩
def group : MergeGroup := .operator 238459 238416
def deltas0_0 : Polynomial Owner := [LeftMerge238463.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge238463.deltaAt
def deltas0_1 : Polynomial Owner := [LeftMerge238464.delta]
theorem rows0_1 : MergeDeltasAt history frameStart owner group deltas0_1 := by
  exact .leaf LeftMerge238464.deltaAt
def deltas1_0 : Polynomial Owner := deltas0_0 ++ deltas0_1
theorem rows1_0 : MergeDeltasAt history frameStart owner group deltas1_0 := by
  exact .append rows0_0 rows0_1
abbrev deltas : Polynomial Owner := deltas1_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows1_0
def left : Polynomial Owner := LeftMerge238463.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge238463.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41597⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14151⟩⟩, ⟨.program ⟨257⟩, ⟨39746⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨41597⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge238463

namespace LeftOperatorMerge238669
def frameStart : Nat := 238580
def owner : Owner := ⟨.program ⟨257⟩, ⟨40294⟩⟩
def group : MergeGroup := .operator 238642 238665
def deltas0_0 : Polynomial Owner := [LeftMerge238669.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge238669.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge238669.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge238669.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨40293⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge238669

namespace LeftOperatorMerge238646
def frameStart : Nat := 238580
def owner : Owner := ⟨.program ⟨257⟩, ⟨41460⟩⟩
def group : MergeGroup := .operator 238642 238640
def deltas0_0 : Polynomial Owner := [LeftMerge238646.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge238646.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge238646.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge238646.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨40092⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge238646

namespace LeftOperatorMerge238658
def frameStart : Nat := 238580
def owner : Owner := ⟨.program ⟨257⟩, ⟨41940⟩⟩
def group : MergeGroup := .operator 238654 238631
def deltas0_0 : Polynomial Owner := [LeftMerge238658.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge238658.deltaAt
def deltas0_1 : Polynomial Owner := [LeftMerge238659.delta]
theorem rows0_1 : MergeDeltasAt history frameStart owner group deltas0_1 := by
  exact .leaf LeftMerge238659.deltaAt
def deltas1_0 : Polynomial Owner := deltas0_0 ++ deltas0_1
theorem rows1_0 : MergeDeltasAt history frameStart owner group deltas1_0 := by
  exact .append rows0_0 rows0_1
abbrev deltas : Polynomial Owner := deltas1_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows1_0
def left : Polynomial Owner := LeftMerge238658.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge238658.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨41939⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨40092⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨41939⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge238658

namespace LeftOperatorMerge238936
def frameStart : Nat := 238853
def owner : Owner := ⟨.program ⟨257⟩, ⟨9555⟩⟩
def group : MergeGroup := .operator 238932 238929
def deltas0_0 : Polynomial Owner := [LeftMerge238936.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge238936.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge238936.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge238936.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge238936

namespace LeftOperatorMerge238883
def frameStart : Nat := 238853
def owner : Owner := ⟨.program ⟨257⟩, ⟨37067⟩⟩
def group : MergeGroup := .operator 238879 238876
def deltas0_0 : Polynomial Owner := [LeftMerge238883.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge238883.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge238883.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge238883.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨13851⟩⟩, ⟨.program ⟨257⟩, ⟨37066⟩⟩], orderedFactors := [] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge238883

namespace LeftOperatorMerge238956
def frameStart : Nat := 238853
def owner : Owner := ⟨.program ⟨257⟩, ⟨37414⟩⟩
def group : MergeGroup := .operator 238909 238952
def deltas0_0 : Polynomial Owner := [LeftMerge238956.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge238956.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge238956.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge238956.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨37412⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge238956

namespace LeftOperatorMerge238913
def frameStart : Nat := 238853
def owner : Owner := ⟨.program ⟨257⟩, ⟨38700⟩⟩
def group : MergeGroup := .operator 238909 238907
def deltas0_0 : Polynomial Owner := [LeftMerge238913.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge238913.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge238913.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge238913.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨13851⟩⟩, ⟨.program ⟨257⟩, ⟨37066⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge238913

namespace LeftOperatorMerge238945
def frameStart : Nat := 238853
def owner : Owner := ⟨.program ⟨257⟩, ⟨38920⟩⟩
def group : MergeGroup := .operator 238941 238898
def deltas0_0 : Polynomial Owner := [LeftMerge238945.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge238945.deltaAt
def deltas0_1 : Polynomial Owner := [LeftMerge238946.delta]
theorem rows0_1 : MergeDeltasAt history frameStart owner group deltas0_1 := by
  exact .leaf LeftMerge238946.deltaAt
def deltas1_0 : Polynomial Owner := deltas0_0 ++ deltas0_1
theorem rows1_0 : MergeDeltasAt history frameStart owner group deltas1_0 := by
  exact .append rows0_0 rows0_1
abbrev deltas : Polynomial Owner := deltas1_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows1_0
def left : Polynomial Owner := LeftMerge238945.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge238945.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨38917⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨13851⟩⟩, ⟨.program ⟨257⟩, ⟨37066⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨38917⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge238945

namespace LeftOperatorMerge239151
def frameStart : Nat := 239062
def owner : Owner := ⟨.program ⟨257⟩, ⟨37618⟩⟩
def group : MergeGroup := .operator 239124 239147
def deltas0_0 : Polynomial Owner := [LeftMerge239151.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge239151.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge239151.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge239151.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨37617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge239151

namespace LeftOperatorMerge239128
def frameStart : Nat := 239062
def owner : Owner := ⟨.program ⟨257⟩, ⟨38780⟩⟩
def group : MergeGroup := .operator 239124 239122
def deltas0_0 : Polynomial Owner := [LeftMerge239128.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge239128.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge239128.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge239128.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨37412⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge239128

namespace LeftOperatorMerge239140
def frameStart : Nat := 239062
def owner : Owner := ⟨.program ⟨257⟩, ⟨39260⟩⟩
def group : MergeGroup := .operator 239136 239113
def deltas0_0 : Polynomial Owner := [LeftMerge239140.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge239140.deltaAt
def deltas0_1 : Polynomial Owner := [LeftMerge239141.delta]
theorem rows0_1 : MergeDeltasAt history frameStart owner group deltas0_1 := by
  exact .leaf LeftMerge239141.deltaAt
def deltas1_0 : Polynomial Owner := deltas0_0 ++ deltas0_1
theorem rows1_0 : MergeDeltasAt history frameStart owner group deltas1_0 := by
  exact .append rows0_0 rows0_1
abbrev deltas : Polynomial Owner := deltas1_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows1_0
def left : Polynomial Owner := LeftMerge239140.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge239140.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7192⟩⟩, ⟨.program ⟨257⟩, ⟨39259⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨37412⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨39259⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge239140

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
