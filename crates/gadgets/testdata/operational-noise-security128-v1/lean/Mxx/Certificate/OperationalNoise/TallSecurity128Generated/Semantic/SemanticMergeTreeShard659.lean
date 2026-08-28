import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard1689
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard1691
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard1692
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard1696
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard1697
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard1698
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard1699
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard1700
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard1705
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard1706

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace LeftOperatorMerge274162
def frameStart : Nat := 274096
def owner : Owner := ⟨.program ⟨257⟩, ⟨20036⟩⟩
def group : MergeGroup := .operator 274158 274156
def deltas0_0 : Polynomial Owner := [LeftMerge274162.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge274162.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge274162.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge274162.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨18522⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge274162

namespace LeftOperatorMerge274174
def frameStart : Nat := 274096
def owner : Owner := ⟨.program ⟨257⟩, ⟨20396⟩⟩
def group : MergeGroup := .operator 274170 274147
def deltas0_0 : Polynomial Owner := [LeftMerge274174.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge274174.deltaAt
def deltas0_1 : Polynomial Owner := [LeftMerge274175.delta]
theorem rows0_1 : MergeDeltasAt history frameStart owner group deltas0_1 := by
  exact .leaf LeftMerge274175.deltaAt
def deltas1_0 : Polynomial Owner := deltas0_0 ++ deltas0_1
theorem rows1_0 : MergeDeltasAt history frameStart owner group deltas1_0 := by
  exact .append rows0_0 rows0_1
abbrev deltas : Polynomial Owner := deltas1_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows1_0
def left : Polynomial Owner := LeftMerge274174.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge274174.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7180⟩⟩, ⟨.program ⟨257⟩, ⟨20395⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨18522⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨20395⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge274174

namespace LeftOperatorMerge274452
def frameStart : Nat := 274369
def owner : Owner := ⟨.program ⟨257⟩, ⟨9570⟩⟩
def group : MergeGroup := .operator 274448 274445
def deltas0_0 : Polynomial Owner := [LeftMerge274452.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge274452.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge274452.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge274452.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge274452

namespace LeftOperatorMerge274399
def frameStart : Nat := 274369
def owner : Owner := ⟨.program ⟨257⟩, ⟨15275⟩⟩
def group : MergeGroup := .operator 274395 274392
def deltas0_0 : Polynomial Owner := [LeftMerge274399.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge274399.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge274399.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge274399.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨12256⟩⟩, ⟨.program ⟨257⟩, ⟨15274⟩⟩], orderedFactors := [] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge274399

namespace LeftOperatorMerge274472
def frameStart : Nat := 274369
def owner : Owner := ⟨.program ⟨257⟩, ⟨15724⟩⟩
def group : MergeGroup := .operator 274425 274468
def deltas0_0 : Polynomial Owner := [LeftMerge274472.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge274472.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge274472.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge274472.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨15722⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge274472

namespace LeftOperatorMerge274429
def frameStart : Nat := 274369
def owner : Owner := ⟨.program ⟨257⟩, ⟨17096⟩⟩
def group : MergeGroup := .operator 274425 274423
def deltas0_0 : Polynomial Owner := [LeftMerge274429.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge274429.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge274429.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge274429.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨12256⟩⟩, ⟨.program ⟨257⟩, ⟨15274⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge274429

namespace LeftOperatorMerge274461
def frameStart : Nat := 274369
def owner : Owner := ⟨.program ⟨257⟩, ⟨17271⟩⟩
def group : MergeGroup := .operator 274457 274414
def deltas0_0 : Polynomial Owner := [LeftMerge274461.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge274461.deltaAt
def deltas0_1 : Polynomial Owner := [LeftMerge274462.delta]
theorem rows0_1 : MergeDeltasAt history frameStart owner group deltas0_1 := by
  exact .leaf LeftMerge274462.deltaAt
def deltas1_0 : Polynomial Owner := deltas0_0 ++ deltas0_1
theorem rows1_0 : MergeDeltasAt history frameStart owner group deltas1_0 := by
  exact .append rows0_0 rows0_1
abbrev deltas : Polynomial Owner := deltas1_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows1_0
def left : Polynomial Owner := LeftMerge274461.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge274461.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17268⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨12256⟩⟩, ⟨.program ⟨257⟩, ⟨15274⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17268⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge274461

namespace LeftOperatorMerge274667
def frameStart : Nat := 274578
def owner : Owner := ⟨.program ⟨257⟩, ⟨15904⟩⟩
def group : MergeGroup := .operator 274640 274663
def deltas0_0 : Polynomial Owner := [LeftMerge274667.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge274667.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge274667.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge274667.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨15903⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge274667

namespace LeftOperatorMerge274644
def frameStart : Nat := 274578
def owner : Owner := ⟨.program ⟨257⟩, ⟨17176⟩⟩
def group : MergeGroup := .operator 274640 274638
def deltas0_0 : Polynomial Owner := [LeftMerge274644.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge274644.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge274644.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge274644.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨15722⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge274644

namespace LeftOperatorMerge274656
def frameStart : Nat := 274578
def owner : Owner := ⟨.program ⟨257⟩, ⟨17530⟩⟩
def group : MergeGroup := .operator 274652 274629
def deltas0_0 : Polynomial Owner := [LeftMerge274656.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge274656.deltaAt
def deltas0_1 : Polynomial Owner := [LeftMerge274657.delta]
theorem rows0_1 : MergeDeltasAt history frameStart owner group deltas0_1 := by
  exact .leaf LeftMerge274657.deltaAt
def deltas1_0 : Polynomial Owner := deltas0_0 ++ deltas0_1
theorem rows1_0 : MergeDeltasAt history frameStart owner group deltas1_0 := by
  exact .append rows0_0 rows0_1
abbrev deltas : Polynomial Owner := deltas1_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows1_0
def left : Polynomial Owner := LeftMerge274656.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge274656.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17529⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨15722⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17529⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge274656

namespace LeftOperatorMerge276219
def frameStart : Nat := 275461
def owner : Owner := ⟨.program ⟨257⟩, ⟨67302⟩⟩
def group : MergeGroup := .operator 275988 276215
def deltas0_0 : Polynomial Owner := [LeftMerge276219.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge276219.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge276219.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge276219.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨67300⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge276219

namespace LeftOperatorMerge275992
def frameStart : Nat := 275461
def owner : Owner := ⟨.program ⟨257⟩, ⟨69057⟩⟩
def group : MergeGroup := .operator 275988 275986
def deltas0_0 : Polynomial Owner := [LeftMerge275992.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge275992.deltaAt
def deltas0_1 : Polynomial Owner := [LeftMerge275993.delta]
theorem rows0_1 : MergeDeltasAt history frameStart owner group deltas0_1 := by
  exact .leaf LeftMerge275993.deltaAt
def deltas0_2 : Polynomial Owner := [LeftMerge275994.delta]
theorem rows0_2 : MergeDeltasAt history frameStart owner group deltas0_2 := by
  exact .leaf LeftMerge275994.deltaAt
def deltas0_3 : Polynomial Owner := [LeftMerge275995.delta]
theorem rows0_3 : MergeDeltasAt history frameStart owner group deltas0_3 := by
  exact .leaf LeftMerge275995.deltaAt
def deltas0_4 : Polynomial Owner := [LeftMerge275996.delta]
theorem rows0_4 : MergeDeltasAt history frameStart owner group deltas0_4 := by
  exact .leaf LeftMerge275996.deltaAt
def deltas0_5 : Polynomial Owner := [LeftMerge275997.delta]
theorem rows0_5 : MergeDeltasAt history frameStart owner group deltas0_5 := by
  exact .leaf LeftMerge275997.deltaAt
def deltas0_6 : Polynomial Owner := [LeftMerge275998.delta]
theorem rows0_6 : MergeDeltasAt history frameStart owner group deltas0_6 := by
  exact .leaf LeftMerge275998.deltaAt
def deltas0_7 : Polynomial Owner := [LeftMerge275999.delta]
theorem rows0_7 : MergeDeltasAt history frameStart owner group deltas0_7 := by
  exact .leaf LeftMerge275999.deltaAt
def deltas0_8 : Polynomial Owner := [LeftMerge276000.delta]
theorem rows0_8 : MergeDeltasAt history frameStart owner group deltas0_8 := by
  exact .leaf LeftMerge276000.deltaAt
def deltas0_9 : Polynomial Owner := [LeftMerge276001.delta]
theorem rows0_9 : MergeDeltasAt history frameStart owner group deltas0_9 := by
  exact .leaf LeftMerge276001.deltaAt
def deltas0_10 : Polynomial Owner := [LeftMerge276002.delta]
theorem rows0_10 : MergeDeltasAt history frameStart owner group deltas0_10 := by
  exact .leaf LeftMerge276002.deltaAt
def deltas0_11 : Polynomial Owner := [LeftMerge276003.delta]
theorem rows0_11 : MergeDeltasAt history frameStart owner group deltas0_11 := by
  exact .leaf LeftMerge276003.deltaAt
def deltas0_12 : Polynomial Owner := [LeftMerge276004.delta]
theorem rows0_12 : MergeDeltasAt history frameStart owner group deltas0_12 := by
  exact .leaf LeftMerge276004.deltaAt
def deltas0_13 : Polynomial Owner := [LeftMerge276005.delta]
theorem rows0_13 : MergeDeltasAt history frameStart owner group deltas0_13 := by
  exact .leaf LeftMerge276005.deltaAt
def deltas0_14 : Polynomial Owner := [LeftMerge276006.delta]
theorem rows0_14 : MergeDeltasAt history frameStart owner group deltas0_14 := by
  exact .leaf LeftMerge276006.deltaAt
def deltas0_15 : Polynomial Owner := [LeftMerge276007.delta]
theorem rows0_15 : MergeDeltasAt history frameStart owner group deltas0_15 := by
  exact .leaf LeftMerge276007.deltaAt
def deltas0_16 : Polynomial Owner := [LeftMerge276008.delta]
theorem rows0_16 : MergeDeltasAt history frameStart owner group deltas0_16 := by
  exact .leaf LeftMerge276008.deltaAt
def deltas0_17 : Polynomial Owner := [LeftMerge276009.delta]
theorem rows0_17 : MergeDeltasAt history frameStart owner group deltas0_17 := by
  exact .leaf LeftMerge276009.deltaAt
def deltas1_0 : Polynomial Owner := deltas0_0 ++ deltas0_1
theorem rows1_0 : MergeDeltasAt history frameStart owner group deltas1_0 := by
  exact .append rows0_0 rows0_1
def deltas1_1 : Polynomial Owner := deltas0_2 ++ deltas0_3
theorem rows1_1 : MergeDeltasAt history frameStart owner group deltas1_1 := by
  exact .append rows0_2 rows0_3
def deltas1_2 : Polynomial Owner := deltas0_4 ++ deltas0_5
theorem rows1_2 : MergeDeltasAt history frameStart owner group deltas1_2 := by
  exact .append rows0_4 rows0_5
def deltas1_3 : Polynomial Owner := deltas0_6 ++ deltas0_7
theorem rows1_3 : MergeDeltasAt history frameStart owner group deltas1_3 := by
  exact .append rows0_6 rows0_7
def deltas1_4 : Polynomial Owner := deltas0_8 ++ deltas0_9
theorem rows1_4 : MergeDeltasAt history frameStart owner group deltas1_4 := by
  exact .append rows0_8 rows0_9
def deltas1_5 : Polynomial Owner := deltas0_10 ++ deltas0_11
theorem rows1_5 : MergeDeltasAt history frameStart owner group deltas1_5 := by
  exact .append rows0_10 rows0_11
def deltas1_6 : Polynomial Owner := deltas0_12 ++ deltas0_13
theorem rows1_6 : MergeDeltasAt history frameStart owner group deltas1_6 := by
  exact .append rows0_12 rows0_13
def deltas1_7 : Polynomial Owner := deltas0_14 ++ deltas0_15
theorem rows1_7 : MergeDeltasAt history frameStart owner group deltas1_7 := by
  exact .append rows0_14 rows0_15
def deltas1_8 : Polynomial Owner := deltas0_16 ++ deltas0_17
theorem rows1_8 : MergeDeltasAt history frameStart owner group deltas1_8 := by
  exact .append rows0_16 rows0_17
def deltas2_0 : Polynomial Owner := deltas1_0 ++ deltas1_1
theorem rows2_0 : MergeDeltasAt history frameStart owner group deltas2_0 := by
  exact .append rows1_0 rows1_1
def deltas2_1 : Polynomial Owner := deltas1_2 ++ deltas1_3
theorem rows2_1 : MergeDeltasAt history frameStart owner group deltas2_1 := by
  exact .append rows1_2 rows1_3
def deltas2_2 : Polynomial Owner := deltas1_4 ++ deltas1_5
theorem rows2_2 : MergeDeltasAt history frameStart owner group deltas2_2 := by
  exact .append rows1_4 rows1_5
def deltas2_3 : Polynomial Owner := deltas1_6 ++ deltas1_7
theorem rows2_3 : MergeDeltasAt history frameStart owner group deltas2_3 := by
  exact .append rows1_6 rows1_7
def deltas3_0 : Polynomial Owner := deltas2_0 ++ deltas2_1
theorem rows3_0 : MergeDeltasAt history frameStart owner group deltas3_0 := by
  exact .append rows2_0 rows2_1
def deltas3_1 : Polynomial Owner := deltas2_2 ++ deltas2_3
theorem rows3_1 : MergeDeltasAt history frameStart owner group deltas3_1 := by
  exact .append rows2_2 rows2_3
def deltas4_0 : Polynomial Owner := deltas3_0 ++ deltas3_1
theorem rows4_0 : MergeDeltasAt history frameStart owner group deltas4_0 := by
  exact .append rows3_0 rows3_1
def deltas5_0 : Polynomial Owner := deltas4_0 ++ deltas1_8
theorem rows5_0 : MergeDeltasAt history frameStart owner group deltas5_0 := by
  exact .append rows4_0 rows1_8
abbrev deltas : Polynomial Owner := deltas5_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows5_0
def left : Polynomial Owner := LeftMerge275992.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge275992.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨15903⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨18709⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨21929⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨26512⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨29192⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨31949⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨34856⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨37536⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨40212⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨42892⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨45576⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨48256⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨51004⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨53984⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨56964⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨59944⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨62924⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨66019⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge275992

namespace LeftOperatorMerge276140
def frameStart : Nat := 275461
def owner : Owner := ⟨.program ⟨257⟩, ⟨70980⟩⟩
def group : MergeGroup := .operator 276136 275977
def deltas0_0 : Polynomial Owner := [LeftMerge276140.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge276140.deltaAt
def deltas0_1 : Polynomial Owner := [LeftMerge276141.delta]
theorem rows0_1 : MergeDeltasAt history frameStart owner group deltas0_1 := by
  exact .leaf LeftMerge276141.deltaAt
def deltas0_2 : Polynomial Owner := [LeftMerge276142.delta]
theorem rows0_2 : MergeDeltasAt history frameStart owner group deltas0_2 := by
  exact .leaf LeftMerge276142.deltaAt
def deltas0_3 : Polynomial Owner := [LeftMerge276143.delta]
theorem rows0_3 : MergeDeltasAt history frameStart owner group deltas0_3 := by
  exact .leaf LeftMerge276143.deltaAt
def deltas0_4 : Polynomial Owner := [LeftMerge276144.delta]
theorem rows0_4 : MergeDeltasAt history frameStart owner group deltas0_4 := by
  exact .leaf LeftMerge276144.deltaAt
def deltas0_5 : Polynomial Owner := [LeftMerge276145.delta]
theorem rows0_5 : MergeDeltasAt history frameStart owner group deltas0_5 := by
  exact .leaf LeftMerge276145.deltaAt
def deltas0_6 : Polynomial Owner := [LeftMerge276146.delta]
theorem rows0_6 : MergeDeltasAt history frameStart owner group deltas0_6 := by
  exact .leaf LeftMerge276146.deltaAt
def deltas0_7 : Polynomial Owner := [LeftMerge276147.delta]
theorem rows0_7 : MergeDeltasAt history frameStart owner group deltas0_7 := by
  exact .leaf LeftMerge276147.deltaAt
def deltas0_8 : Polynomial Owner := [LeftMerge276148.delta]
theorem rows0_8 : MergeDeltasAt history frameStart owner group deltas0_8 := by
  exact .leaf LeftMerge276148.deltaAt
def deltas0_9 : Polynomial Owner := [LeftMerge276149.delta]
theorem rows0_9 : MergeDeltasAt history frameStart owner group deltas0_9 := by
  exact .leaf LeftMerge276149.deltaAt
def deltas0_10 : Polynomial Owner := [LeftMerge276150.delta]
theorem rows0_10 : MergeDeltasAt history frameStart owner group deltas0_10 := by
  exact .leaf LeftMerge276150.deltaAt
def deltas0_11 : Polynomial Owner := [LeftMerge276151.delta]
theorem rows0_11 : MergeDeltasAt history frameStart owner group deltas0_11 := by
  exact .leaf LeftMerge276151.deltaAt
def deltas0_12 : Polynomial Owner := [LeftMerge276152.delta]
theorem rows0_12 : MergeDeltasAt history frameStart owner group deltas0_12 := by
  exact .leaf LeftMerge276152.deltaAt
def deltas0_13 : Polynomial Owner := [LeftMerge276153.delta]
theorem rows0_13 : MergeDeltasAt history frameStart owner group deltas0_13 := by
  exact .leaf LeftMerge276153.deltaAt
def deltas0_14 : Polynomial Owner := [LeftMerge276154.delta]
theorem rows0_14 : MergeDeltasAt history frameStart owner group deltas0_14 := by
  exact .leaf LeftMerge276154.deltaAt
def deltas0_15 : Polynomial Owner := [LeftMerge276155.delta]
theorem rows0_15 : MergeDeltasAt history frameStart owner group deltas0_15 := by
  exact .leaf LeftMerge276155.deltaAt
def deltas0_16 : Polynomial Owner := [LeftMerge276156.delta]
theorem rows0_16 : MergeDeltasAt history frameStart owner group deltas0_16 := by
  exact .leaf LeftMerge276156.deltaAt
def deltas0_17 : Polynomial Owner := [LeftMerge276157.delta]
theorem rows0_17 : MergeDeltasAt history frameStart owner group deltas0_17 := by
  exact .leaf LeftMerge276157.deltaAt
def deltas0_18 : Polynomial Owner := [LeftMerge276158.delta]
theorem rows0_18 : MergeDeltasAt history frameStart owner group deltas0_18 := by
  exact .leaf LeftMerge276158.deltaAt
def deltas0_19 : Polynomial Owner := [LeftMerge276161.delta]
theorem rows0_19 : MergeDeltasAt history frameStart owner group deltas0_19 := by
  exact .leaf LeftMerge276161.deltaAt
def deltas0_20 : Polynomial Owner := [LeftMerge276164.delta]
theorem rows0_20 : MergeDeltasAt history frameStart owner group deltas0_20 := by
  exact .leaf LeftMerge276164.deltaAt
def deltas0_21 : Polynomial Owner := [LeftMerge276167.delta]
theorem rows0_21 : MergeDeltasAt history frameStart owner group deltas0_21 := by
  exact .leaf LeftMerge276167.deltaAt
def deltas0_22 : Polynomial Owner := [LeftMerge276170.delta]
theorem rows0_22 : MergeDeltasAt history frameStart owner group deltas0_22 := by
  exact .leaf LeftMerge276170.deltaAt
def deltas0_23 : Polynomial Owner := [LeftMerge276173.delta]
theorem rows0_23 : MergeDeltasAt history frameStart owner group deltas0_23 := by
  exact .leaf LeftMerge276173.deltaAt
def deltas0_24 : Polynomial Owner := [LeftMerge276176.delta]
theorem rows0_24 : MergeDeltasAt history frameStart owner group deltas0_24 := by
  exact .leaf LeftMerge276176.deltaAt
def deltas0_25 : Polynomial Owner := [LeftMerge276179.delta]
theorem rows0_25 : MergeDeltasAt history frameStart owner group deltas0_25 := by
  exact .leaf LeftMerge276179.deltaAt
def deltas0_26 : Polynomial Owner := [LeftMerge276182.delta]
theorem rows0_26 : MergeDeltasAt history frameStart owner group deltas0_26 := by
  exact .leaf LeftMerge276182.deltaAt
def deltas0_27 : Polynomial Owner := [LeftMerge276185.delta]
theorem rows0_27 : MergeDeltasAt history frameStart owner group deltas0_27 := by
  exact .leaf LeftMerge276185.deltaAt
def deltas0_28 : Polynomial Owner := [LeftMerge276188.delta]
theorem rows0_28 : MergeDeltasAt history frameStart owner group deltas0_28 := by
  exact .leaf LeftMerge276188.deltaAt
def deltas0_29 : Polynomial Owner := [LeftMerge276191.delta]
theorem rows0_29 : MergeDeltasAt history frameStart owner group deltas0_29 := by
  exact .leaf LeftMerge276191.deltaAt
def deltas0_30 : Polynomial Owner := [LeftMerge276194.delta]
theorem rows0_30 : MergeDeltasAt history frameStart owner group deltas0_30 := by
  exact .leaf LeftMerge276194.deltaAt
def deltas0_31 : Polynomial Owner := [LeftMerge276197.delta]
theorem rows0_31 : MergeDeltasAt history frameStart owner group deltas0_31 := by
  exact .leaf LeftMerge276197.deltaAt
def deltas0_32 : Polynomial Owner := [LeftMerge276200.delta]
theorem rows0_32 : MergeDeltasAt history frameStart owner group deltas0_32 := by
  exact .leaf LeftMerge276200.deltaAt
def deltas0_33 : Polynomial Owner := [LeftMerge276203.delta]
theorem rows0_33 : MergeDeltasAt history frameStart owner group deltas0_33 := by
  exact .leaf LeftMerge276203.deltaAt
def deltas0_34 : Polynomial Owner := [LeftMerge276206.delta]
theorem rows0_34 : MergeDeltasAt history frameStart owner group deltas0_34 := by
  exact .leaf LeftMerge276206.deltaAt
def deltas0_35 : Polynomial Owner := [LeftMerge276209.delta]
theorem rows0_35 : MergeDeltasAt history frameStart owner group deltas0_35 := by
  exact .leaf LeftMerge276209.deltaAt
def deltas1_0 : Polynomial Owner := deltas0_0 ++ deltas0_1
theorem rows1_0 : MergeDeltasAt history frameStart owner group deltas1_0 := by
  exact .append rows0_0 rows0_1
def deltas1_1 : Polynomial Owner := deltas0_2 ++ deltas0_3
theorem rows1_1 : MergeDeltasAt history frameStart owner group deltas1_1 := by
  exact .append rows0_2 rows0_3
def deltas1_2 : Polynomial Owner := deltas0_4 ++ deltas0_5
theorem rows1_2 : MergeDeltasAt history frameStart owner group deltas1_2 := by
  exact .append rows0_4 rows0_5
def deltas1_3 : Polynomial Owner := deltas0_6 ++ deltas0_7
theorem rows1_3 : MergeDeltasAt history frameStart owner group deltas1_3 := by
  exact .append rows0_6 rows0_7
def deltas1_4 : Polynomial Owner := deltas0_8 ++ deltas0_9
theorem rows1_4 : MergeDeltasAt history frameStart owner group deltas1_4 := by
  exact .append rows0_8 rows0_9
def deltas1_5 : Polynomial Owner := deltas0_10 ++ deltas0_11
theorem rows1_5 : MergeDeltasAt history frameStart owner group deltas1_5 := by
  exact .append rows0_10 rows0_11
def deltas1_6 : Polynomial Owner := deltas0_12 ++ deltas0_13
theorem rows1_6 : MergeDeltasAt history frameStart owner group deltas1_6 := by
  exact .append rows0_12 rows0_13
def deltas1_7 : Polynomial Owner := deltas0_14 ++ deltas0_15
theorem rows1_7 : MergeDeltasAt history frameStart owner group deltas1_7 := by
  exact .append rows0_14 rows0_15
def deltas1_8 : Polynomial Owner := deltas0_16 ++ deltas0_17
theorem rows1_8 : MergeDeltasAt history frameStart owner group deltas1_8 := by
  exact .append rows0_16 rows0_17
def deltas1_9 : Polynomial Owner := deltas0_18 ++ deltas0_19
theorem rows1_9 : MergeDeltasAt history frameStart owner group deltas1_9 := by
  exact .append rows0_18 rows0_19
def deltas1_10 : Polynomial Owner := deltas0_20 ++ deltas0_21
theorem rows1_10 : MergeDeltasAt history frameStart owner group deltas1_10 := by
  exact .append rows0_20 rows0_21
def deltas1_11 : Polynomial Owner := deltas0_22 ++ deltas0_23
theorem rows1_11 : MergeDeltasAt history frameStart owner group deltas1_11 := by
  exact .append rows0_22 rows0_23
def deltas1_12 : Polynomial Owner := deltas0_24 ++ deltas0_25
theorem rows1_12 : MergeDeltasAt history frameStart owner group deltas1_12 := by
  exact .append rows0_24 rows0_25
def deltas1_13 : Polynomial Owner := deltas0_26 ++ deltas0_27
theorem rows1_13 : MergeDeltasAt history frameStart owner group deltas1_13 := by
  exact .append rows0_26 rows0_27
def deltas1_14 : Polynomial Owner := deltas0_28 ++ deltas0_29
theorem rows1_14 : MergeDeltasAt history frameStart owner group deltas1_14 := by
  exact .append rows0_28 rows0_29
def deltas1_15 : Polynomial Owner := deltas0_30 ++ deltas0_31
theorem rows1_15 : MergeDeltasAt history frameStart owner group deltas1_15 := by
  exact .append rows0_30 rows0_31
def deltas1_16 : Polynomial Owner := deltas0_32 ++ deltas0_33
theorem rows1_16 : MergeDeltasAt history frameStart owner group deltas1_16 := by
  exact .append rows0_32 rows0_33
def deltas1_17 : Polynomial Owner := deltas0_34 ++ deltas0_35
theorem rows1_17 : MergeDeltasAt history frameStart owner group deltas1_17 := by
  exact .append rows0_34 rows0_35
def deltas2_0 : Polynomial Owner := deltas1_0 ++ deltas1_1
theorem rows2_0 : MergeDeltasAt history frameStart owner group deltas2_0 := by
  exact .append rows1_0 rows1_1
def deltas2_1 : Polynomial Owner := deltas1_2 ++ deltas1_3
theorem rows2_1 : MergeDeltasAt history frameStart owner group deltas2_1 := by
  exact .append rows1_2 rows1_3
def deltas2_2 : Polynomial Owner := deltas1_4 ++ deltas1_5
theorem rows2_2 : MergeDeltasAt history frameStart owner group deltas2_2 := by
  exact .append rows1_4 rows1_5
def deltas2_3 : Polynomial Owner := deltas1_6 ++ deltas1_7
theorem rows2_3 : MergeDeltasAt history frameStart owner group deltas2_3 := by
  exact .append rows1_6 rows1_7
def deltas2_4 : Polynomial Owner := deltas1_8 ++ deltas1_9
theorem rows2_4 : MergeDeltasAt history frameStart owner group deltas2_4 := by
  exact .append rows1_8 rows1_9
def deltas2_5 : Polynomial Owner := deltas1_10 ++ deltas1_11
theorem rows2_5 : MergeDeltasAt history frameStart owner group deltas2_5 := by
  exact .append rows1_10 rows1_11
def deltas2_6 : Polynomial Owner := deltas1_12 ++ deltas1_13
theorem rows2_6 : MergeDeltasAt history frameStart owner group deltas2_6 := by
  exact .append rows1_12 rows1_13
def deltas2_7 : Polynomial Owner := deltas1_14 ++ deltas1_15
theorem rows2_7 : MergeDeltasAt history frameStart owner group deltas2_7 := by
  exact .append rows1_14 rows1_15
def deltas2_8 : Polynomial Owner := deltas1_16 ++ deltas1_17
theorem rows2_8 : MergeDeltasAt history frameStart owner group deltas2_8 := by
  exact .append rows1_16 rows1_17
def deltas3_0 : Polynomial Owner := deltas2_0 ++ deltas2_1
theorem rows3_0 : MergeDeltasAt history frameStart owner group deltas3_0 := by
  exact .append rows2_0 rows2_1
def deltas3_1 : Polynomial Owner := deltas2_2 ++ deltas2_3
theorem rows3_1 : MergeDeltasAt history frameStart owner group deltas3_1 := by
  exact .append rows2_2 rows2_3
def deltas3_2 : Polynomial Owner := deltas2_4 ++ deltas2_5
theorem rows3_2 : MergeDeltasAt history frameStart owner group deltas3_2 := by
  exact .append rows2_4 rows2_5
def deltas3_3 : Polynomial Owner := deltas2_6 ++ deltas2_7
theorem rows3_3 : MergeDeltasAt history frameStart owner group deltas3_3 := by
  exact .append rows2_6 rows2_7
def deltas4_0 : Polynomial Owner := deltas3_0 ++ deltas3_1
theorem rows4_0 : MergeDeltasAt history frameStart owner group deltas4_0 := by
  exact .append rows3_0 rows3_1
def deltas4_1 : Polynomial Owner := deltas3_2 ++ deltas3_3
theorem rows4_1 : MergeDeltasAt history frameStart owner group deltas4_1 := by
  exact .append rows3_2 rows3_3
def deltas5_0 : Polynomial Owner := deltas4_0 ++ deltas4_1
theorem rows5_0 : MergeDeltasAt history frameStart owner group deltas5_0 := by
  exact .append rows4_0 rows4_1
def deltas6_0 : Polynomial Owner := deltas5_0 ++ deltas2_8
theorem rows6_0 : MergeDeltasAt history frameStart owner group deltas6_0 := by
  exact .append rows5_0 rows2_8
abbrev deltas : Polynomial Owner := deltas6_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows6_0
def left : Polynomial Owner := LeftMerge276140.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge276140.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7202⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7214⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7216⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7218⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7220⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7222⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7232⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨15903⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨18709⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨21929⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨26512⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨29192⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨31949⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨34856⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨37536⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨40212⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨42892⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨45576⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨48256⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨51004⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨53984⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨56964⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨59944⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨62924⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨66019⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge276140

namespace LeftOperatorMerge276499
def frameStart : Nat := 276410
def owner : Owner := ⟨.program ⟨257⟩, ⟨48254⟩⟩
def group : MergeGroup := .operator 276472 276495
def deltas0_0 : Polynomial Owner := [LeftMerge276499.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge276499.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge276499.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge276499.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨48252⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge276499

namespace LeftOperatorMerge276476
def frameStart : Nat := 276410
def owner : Owner := ⟨.program ⟨257⟩, ⟨49476⟩⟩
def group : MergeGroup := .operator 276472 276470
def deltas0_0 : Polynomial Owner := [LeftMerge276476.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge276476.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge276476.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge276476.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨48082⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge276476

namespace LeftOperatorMerge276488
def frameStart : Nat := 276410
def owner : Owner := ⟨.program ⟨257⟩, ⟨49817⟩⟩
def group : MergeGroup := .operator 276484 276461
def deltas0_0 : Polynomial Owner := [LeftMerge276488.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge276488.deltaAt
def deltas0_1 : Polynomial Owner := [LeftMerge276489.delta]
theorem rows0_1 : MergeDeltasAt history frameStart owner group deltas0_1 := by
  exact .leaf LeftMerge276489.deltaAt
def deltas1_0 : Polynomial Owner := deltas0_0 ++ deltas0_1
theorem rows1_0 : MergeDeltasAt history frameStart owner group deltas1_0 := by
  exact .append rows0_0 rows0_1
abbrev deltas : Polynomial Owner := deltas1_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows1_0
def left : Polynomial Owner := LeftMerge276488.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge276488.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨49816⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨48082⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨49816⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge276488

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
