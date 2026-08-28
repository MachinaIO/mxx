import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard698
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard699
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard700
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard701
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard702
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard703
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard704

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace LeftOperatorMerge116048
def frameStart : Nat := 115959
def owner : Owner := ⟨.program ⟨257⟩, ⟨43017⟩⟩
def group : MergeGroup := .operator 116021 116044
def deltas0_0 : Polynomial Owner := [LeftMerge116048.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge116048.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge116048.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge116048.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨43015⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge116048

namespace LeftOperatorMerge116025
def frameStart : Nat := 115959
def owner : Owner := ⟨.program ⟨257⟩, ⟨44152⟩⟩
def group : MergeGroup := .operator 116021 116019
def deltas0_0 : Polynomial Owner := [LeftMerge116025.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge116025.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge116025.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge116025.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨42796⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge116025

namespace LeftOperatorMerge116037
def frameStart : Nat := 115959
def owner : Owner := ⟨.program ⟨257⟩, ⟨44689⟩⟩
def group : MergeGroup := .operator 116033 116010
def deltas0_0 : Polynomial Owner := [LeftMerge116037.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge116037.deltaAt
def deltas0_1 : Polynomial Owner := [LeftMerge116038.delta]
theorem rows0_1 : MergeDeltasAt history frameStart owner group deltas0_1 := by
  exact .leaf LeftMerge116038.deltaAt
def deltas1_0 : Polynomial Owner := deltas0_0 ++ deltas0_1
theorem rows1_0 : MergeDeltasAt history frameStart owner group deltas1_0 := by
  exact .append rows0_0 rows0_1
abbrev deltas : Polynomial Owner := deltas1_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows1_0
def left : Polynomial Owner := LeftMerge116037.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge116037.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44688⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨42796⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨44688⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge116037

namespace LeftOperatorMerge116260
def frameStart : Nat := 116171
def owner : Owner := ⟨.program ⟨257⟩, ⟨40337⟩⟩
def group : MergeGroup := .operator 116233 116256
def deltas0_0 : Polynomial Owner := [LeftMerge116260.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge116260.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge116260.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge116260.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨40335⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge116260

namespace LeftOperatorMerge116237
def frameStart : Nat := 116171
def owner : Owner := ⟨.program ⟨257⟩, ⟨41472⟩⟩
def group : MergeGroup := .operator 116233 116231
def deltas0_0 : Polynomial Owner := [LeftMerge116237.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge116237.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge116237.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge116237.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨40116⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge116237

namespace LeftOperatorMerge116249
def frameStart : Nat := 116171
def owner : Owner := ⟨.program ⟨257⟩, ⟨42009⟩⟩
def group : MergeGroup := .operator 116245 116222
def deltas0_0 : Polynomial Owner := [LeftMerge116249.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge116249.deltaAt
def deltas0_1 : Polynomial Owner := [LeftMerge116250.delta]
theorem rows0_1 : MergeDeltasAt history frameStart owner group deltas0_1 := by
  exact .leaf LeftMerge116250.deltaAt
def deltas1_0 : Polynomial Owner := deltas0_0 ++ deltas0_1
theorem rows1_0 : MergeDeltasAt history frameStart owner group deltas1_0 := by
  exact .append rows0_0 rows0_1
abbrev deltas : Polynomial Owner := deltas1_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows1_0
def left : Polynomial Owner := LeftMerge116249.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge116249.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨42008⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨40116⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨42008⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge116249

namespace LeftOperatorMerge116472
def frameStart : Nat := 116383
def owner : Owner := ⟨.program ⟨257⟩, ⟨37654⟩⟩
def group : MergeGroup := .operator 116445 116468
def deltas0_0 : Polynomial Owner := [LeftMerge116472.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge116472.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge116472.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge116472.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨37652⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge116472

namespace LeftOperatorMerge116449
def frameStart : Nat := 116383
def owner : Owner := ⟨.program ⟨257⟩, ⟨38792⟩⟩
def group : MergeGroup := .operator 116445 116443
def deltas0_0 : Polynomial Owner := [LeftMerge116449.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge116449.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge116449.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge116449.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨37436⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge116449

namespace LeftOperatorMerge116461
def frameStart : Nat := 116383
def owner : Owner := ⟨.program ⟨257⟩, ⟨39329⟩⟩
def group : MergeGroup := .operator 116457 116434
def deltas0_0 : Polynomial Owner := [LeftMerge116461.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge116461.deltaAt
def deltas0_1 : Polynomial Owner := [LeftMerge116462.delta]
theorem rows0_1 : MergeDeltasAt history frameStart owner group deltas0_1 := by
  exact .leaf LeftMerge116462.deltaAt
def deltas1_0 : Polynomial Owner := deltas0_0 ++ deltas0_1
theorem rows1_0 : MergeDeltasAt history frameStart owner group deltas1_0 := by
  exact .append rows0_0 rows0_1
abbrev deltas : Polynomial Owner := deltas1_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows1_0
def left : Polynomial Owner := LeftMerge116461.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge116461.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7192⟩⟩, ⟨.program ⟨257⟩, ⟨39328⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨37436⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨39328⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge116461

namespace LeftOperatorMerge116684
def frameStart : Nat := 116595
def owner : Owner := ⟨.program ⟨257⟩, ⟨34974⟩⟩
def group : MergeGroup := .operator 116657 116680
def deltas0_0 : Polynomial Owner := [LeftMerge116684.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge116684.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge116684.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge116684.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨34972⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge116684

namespace LeftOperatorMerge116661
def frameStart : Nat := 116595
def owner : Owner := ⟨.program ⟨257⟩, ⟨36112⟩⟩
def group : MergeGroup := .operator 116657 116655
def deltas0_0 : Polynomial Owner := [LeftMerge116661.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge116661.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge116661.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge116661.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨34756⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge116661

namespace LeftOperatorMerge116673
def frameStart : Nat := 116595
def owner : Owner := ⟨.program ⟨257⟩, ⟨36649⟩⟩
def group : MergeGroup := .operator 116669 116646
def deltas0_0 : Polynomial Owner := [LeftMerge116673.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge116673.deltaAt
def deltas0_1 : Polynomial Owner := [LeftMerge116674.delta]
theorem rows0_1 : MergeDeltasAt history frameStart owner group deltas0_1 := by
  exact .leaf LeftMerge116674.deltaAt
def deltas1_0 : Polynomial Owner := deltas0_0 ++ deltas0_1
theorem rows1_0 : MergeDeltasAt history frameStart owner group deltas1_0 := by
  exact .append rows0_0 rows0_1
abbrev deltas : Polynomial Owner := deltas1_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows1_0
def left : Polynomial Owner := LeftMerge116673.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge116673.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7191⟩⟩, ⟨.program ⟨257⟩, ⟨36648⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨34756⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36648⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge116673

namespace LeftOperatorMerge116896
def frameStart : Nat := 116807
def owner : Owner := ⟨.program ⟨257⟩, ⟨29317⟩⟩
def group : MergeGroup := .operator 116869 116892
def deltas0_0 : Polynomial Owner := [LeftMerge116896.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge116896.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge116896.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge116896.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨29315⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge116896

namespace LeftOperatorMerge116873
def frameStart : Nat := 116807
def owner : Owner := ⟨.program ⟨257⟩, ⟨30452⟩⟩
def group : MergeGroup := .operator 116869 116867
def deltas0_0 : Polynomial Owner := [LeftMerge116873.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge116873.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge116873.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge116873.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨29096⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge116873

namespace LeftOperatorMerge116885
def frameStart : Nat := 116807
def owner : Owner := ⟨.program ⟨257⟩, ⟨30989⟩⟩
def group : MergeGroup := .operator 116881 116858
def deltas0_0 : Polynomial Owner := [LeftMerge116885.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge116885.deltaAt
def deltas0_1 : Polynomial Owner := [LeftMerge116886.delta]
theorem rows0_1 : MergeDeltasAt history frameStart owner group deltas0_1 := by
  exact .leaf LeftMerge116886.deltaAt
def deltas1_0 : Polynomial Owner := deltas0_0 ++ deltas0_1
theorem rows1_0 : MergeDeltasAt history frameStart owner group deltas1_0 := by
  exact .append rows0_0 rows0_1
abbrev deltas : Polynomial Owner := deltas1_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows1_0
def left : Polynomial Owner := LeftMerge116885.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge116885.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨30988⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨29096⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨30988⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge116885

namespace LeftOperatorMerge117108
def frameStart : Nat := 117019
def owner : Owner := ⟨.program ⟨257⟩, ⟨26637⟩⟩
def group : MergeGroup := .operator 117081 117104
def deltas0_0 : Polynomial Owner := [LeftMerge117108.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge117108.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge117108.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge117108.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨26635⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge117108

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
