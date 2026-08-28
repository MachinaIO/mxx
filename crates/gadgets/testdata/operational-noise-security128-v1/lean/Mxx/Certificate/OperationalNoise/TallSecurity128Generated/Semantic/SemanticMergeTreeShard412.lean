import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard014
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard016
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard017
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard018
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard019
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard021
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard022
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard023
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard031
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard032
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard033
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard034
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard035
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard036
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard037
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard038
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard725
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard726
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard727
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard908
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard909
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard910
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard1275
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard1276
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard1277
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard1278
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard1367
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard1368
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard1369
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard1370
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard1459
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard1460
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard1461

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace LeftOperatorMerge5819
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨67383⟩⟩
def group : MergeGroup := .operator 5815 36
def deltas0_0 : Polynomial Owner := [LeftMerge5819.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge5819.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge5819.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge5819.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67382⟩⟩], orderedFactors := [] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge5819

namespace LeftOperatorMerge6044
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨67386⟩⟩
def group : MergeGroup := .operator 6040 5317
def deltas0_0 : Polynomial Owner := [LeftMerge6044.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge6044.deltaAt
def deltas0_1 : Polynomial Owner := [LeftMerge6045.delta]
theorem rows0_1 : MergeDeltasAt history frameStart owner group deltas0_1 := by
  exact .leaf LeftMerge6045.deltaAt
def deltas0_2 : Polynomial Owner := [LeftMerge6046.delta]
theorem rows0_2 : MergeDeltasAt history frameStart owner group deltas0_2 := by
  exact .leaf LeftMerge6046.deltaAt
def deltas0_3 : Polynomial Owner := [LeftMerge6047.delta]
theorem rows0_3 : MergeDeltasAt history frameStart owner group deltas0_3 := by
  exact .leaf LeftMerge6047.deltaAt
def deltas0_4 : Polynomial Owner := [LeftMerge6048.delta]
theorem rows0_4 : MergeDeltasAt history frameStart owner group deltas0_4 := by
  exact .leaf LeftMerge6048.deltaAt
def deltas0_5 : Polynomial Owner := [LeftMerge6049.delta]
theorem rows0_5 : MergeDeltasAt history frameStart owner group deltas0_5 := by
  exact .leaf LeftMerge6049.deltaAt
def deltas0_6 : Polynomial Owner := [LeftMerge6050.delta]
theorem rows0_6 : MergeDeltasAt history frameStart owner group deltas0_6 := by
  exact .leaf LeftMerge6050.deltaAt
def deltas0_7 : Polynomial Owner := [LeftMerge6051.delta]
theorem rows0_7 : MergeDeltasAt history frameStart owner group deltas0_7 := by
  exact .leaf LeftMerge6051.deltaAt
def deltas0_8 : Polynomial Owner := [LeftMerge6052.delta]
theorem rows0_8 : MergeDeltasAt history frameStart owner group deltas0_8 := by
  exact .leaf LeftMerge6052.deltaAt
def deltas0_9 : Polynomial Owner := [LeftMerge6053.delta]
theorem rows0_9 : MergeDeltasAt history frameStart owner group deltas0_9 := by
  exact .leaf LeftMerge6053.deltaAt
def deltas0_10 : Polynomial Owner := [LeftMerge6054.delta]
theorem rows0_10 : MergeDeltasAt history frameStart owner group deltas0_10 := by
  exact .leaf LeftMerge6054.deltaAt
def deltas0_11 : Polynomial Owner := [LeftMerge6055.delta]
theorem rows0_11 : MergeDeltasAt history frameStart owner group deltas0_11 := by
  exact .leaf LeftMerge6055.deltaAt
def deltas0_12 : Polynomial Owner := [LeftMerge6056.delta]
theorem rows0_12 : MergeDeltasAt history frameStart owner group deltas0_12 := by
  exact .leaf LeftMerge6056.deltaAt
def deltas0_13 : Polynomial Owner := [LeftMerge6057.delta]
theorem rows0_13 : MergeDeltasAt history frameStart owner group deltas0_13 := by
  exact .leaf LeftMerge6057.deltaAt
def deltas0_14 : Polynomial Owner := [LeftMerge6058.delta]
theorem rows0_14 : MergeDeltasAt history frameStart owner group deltas0_14 := by
  exact .leaf LeftMerge6058.deltaAt
def deltas0_15 : Polynomial Owner := [LeftMerge6059.delta]
theorem rows0_15 : MergeDeltasAt history frameStart owner group deltas0_15 := by
  exact .leaf LeftMerge6059.deltaAt
def deltas0_16 : Polynomial Owner := [LeftMerge6060.delta]
theorem rows0_16 : MergeDeltasAt history frameStart owner group deltas0_16 := by
  exact .leaf LeftMerge6060.deltaAt
def deltas0_17 : Polynomial Owner := [LeftMerge6061.delta]
theorem rows0_17 : MergeDeltasAt history frameStart owner group deltas0_17 := by
  exact .leaf LeftMerge6061.deltaAt
def deltas0_18 : Polynomial Owner := [LeftMerge6062.delta]
theorem rows0_18 : MergeDeltasAt history frameStart owner group deltas0_18 := by
  exact .leaf LeftMerge6062.deltaAt
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
def deltas2_4 : Polynomial Owner := deltas1_8 ++ deltas0_18
theorem rows2_4 : MergeDeltasAt history frameStart owner group deltas2_4 := by
  exact .append rows1_8 rows0_18
def deltas3_0 : Polynomial Owner := deltas2_0 ++ deltas2_1
theorem rows3_0 : MergeDeltasAt history frameStart owner group deltas3_0 := by
  exact .append rows2_0 rows2_1
def deltas3_1 : Polynomial Owner := deltas2_2 ++ deltas2_3
theorem rows3_1 : MergeDeltasAt history frameStart owner group deltas3_1 := by
  exact .append rows2_2 rows2_3
def deltas4_0 : Polynomial Owner := deltas3_0 ++ deltas3_1
theorem rows4_0 : MergeDeltasAt history frameStart owner group deltas4_0 := by
  exact .append rows3_0 rows3_1
def deltas5_0 : Polynomial Owner := deltas4_0 ++ deltas2_4
theorem rows5_0 : MergeDeltasAt history frameStart owner group deltas5_0 := by
  exact .append rows4_0 rows2_4
abbrev deltas : Polynomial Owner := deltas5_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows5_0
def left : Polynomial Owner := LeftMerge6044.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge6044.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨6745⟩⟩, ⟨.program ⟨257⟩, ⟨63009⟩⟩], orderedFactors := [] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨6745⟩⟩, ⟨.program ⟨257⟩, ⟨60029⟩⟩], orderedFactors := [] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨6745⟩⟩, ⟨.program ⟨257⟩, ⟨57049⟩⟩], orderedFactors := [] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6745⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54069⟩⟩], orderedFactors := [] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6745⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51089⟩⟩], orderedFactors := [] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6745⟩⟩, ⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67382⟩⟩], orderedFactors := [] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6745⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32025⟩⟩], orderedFactors := [] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6745⟩⟩, ⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48307⟩⟩], orderedFactors := [] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6745⟩⟩, ⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45627⟩⟩], orderedFactors := [] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6745⟩⟩, ⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨42950⟩⟩], orderedFactors := [] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6745⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22005⟩⟩], orderedFactors := [] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6745⟩⟩, ⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40270⟩⟩], orderedFactors := [] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6745⟩⟩, ⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37587⟩⟩], orderedFactors := [] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6745⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34907⟩⟩], orderedFactors := [] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6745⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18785⟩⟩], orderedFactors := [] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6745⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29250⟩⟩], orderedFactors := [] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6745⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26570⟩⟩], orderedFactors := [] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6745⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15966⟩⟩], orderedFactors := [] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6745⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66308⟩⟩], orderedFactors := [] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge6044

namespace LeftOperatorMerge119705
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨67387⟩⟩
def group : MergeGroup := .operator 119663 6040
def deltas0_0 : Polynomial Owner := [LeftMerge119705.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge119705.deltaAt
def deltas0_1 : Polynomial Owner := [LeftMerge119706.delta]
theorem rows0_1 : MergeDeltasAt history frameStart owner group deltas0_1 := by
  exact .leaf LeftMerge119706.deltaAt
def deltas0_2 : Polynomial Owner := [LeftMerge119707.delta]
theorem rows0_2 : MergeDeltasAt history frameStart owner group deltas0_2 := by
  exact .leaf LeftMerge119707.deltaAt
def deltas0_3 : Polynomial Owner := [LeftMerge119708.delta]
theorem rows0_3 : MergeDeltasAt history frameStart owner group deltas0_3 := by
  exact .leaf LeftMerge119708.deltaAt
def deltas0_4 : Polynomial Owner := [LeftMerge119709.delta]
theorem rows0_4 : MergeDeltasAt history frameStart owner group deltas0_4 := by
  exact .leaf LeftMerge119709.deltaAt
def deltas0_5 : Polynomial Owner := [LeftMerge119710.delta]
theorem rows0_5 : MergeDeltasAt history frameStart owner group deltas0_5 := by
  exact .leaf LeftMerge119710.deltaAt
def deltas0_6 : Polynomial Owner := [LeftMerge119711.delta]
theorem rows0_6 : MergeDeltasAt history frameStart owner group deltas0_6 := by
  exact .leaf LeftMerge119711.deltaAt
def deltas0_7 : Polynomial Owner := [LeftMerge119712.delta]
theorem rows0_7 : MergeDeltasAt history frameStart owner group deltas0_7 := by
  exact .leaf LeftMerge119712.deltaAt
def deltas0_8 : Polynomial Owner := [LeftMerge119713.delta]
theorem rows0_8 : MergeDeltasAt history frameStart owner group deltas0_8 := by
  exact .leaf LeftMerge119713.deltaAt
def deltas0_9 : Polynomial Owner := [LeftMerge119714.delta]
theorem rows0_9 : MergeDeltasAt history frameStart owner group deltas0_9 := by
  exact .leaf LeftMerge119714.deltaAt
def deltas0_10 : Polynomial Owner := [LeftMerge119715.delta]
theorem rows0_10 : MergeDeltasAt history frameStart owner group deltas0_10 := by
  exact .leaf LeftMerge119715.deltaAt
def deltas0_11 : Polynomial Owner := [LeftMerge119716.delta]
theorem rows0_11 : MergeDeltasAt history frameStart owner group deltas0_11 := by
  exact .leaf LeftMerge119716.deltaAt
def deltas0_12 : Polynomial Owner := [LeftMerge119717.delta]
theorem rows0_12 : MergeDeltasAt history frameStart owner group deltas0_12 := by
  exact .leaf LeftMerge119717.deltaAt
def deltas0_13 : Polynomial Owner := [LeftMerge119718.delta]
theorem rows0_13 : MergeDeltasAt history frameStart owner group deltas0_13 := by
  exact .leaf LeftMerge119718.deltaAt
def deltas0_14 : Polynomial Owner := [LeftMerge119719.delta]
theorem rows0_14 : MergeDeltasAt history frameStart owner group deltas0_14 := by
  exact .leaf LeftMerge119719.deltaAt
def deltas0_15 : Polynomial Owner := [LeftMerge119720.delta]
theorem rows0_15 : MergeDeltasAt history frameStart owner group deltas0_15 := by
  exact .leaf LeftMerge119720.deltaAt
def deltas0_16 : Polynomial Owner := [LeftMerge119721.delta]
theorem rows0_16 : MergeDeltasAt history frameStart owner group deltas0_16 := by
  exact .leaf LeftMerge119721.deltaAt
def deltas0_17 : Polynomial Owner := [LeftMerge119722.delta]
theorem rows0_17 : MergeDeltasAt history frameStart owner group deltas0_17 := by
  exact .leaf LeftMerge119722.deltaAt
def deltas0_18 : Polynomial Owner := [LeftMerge119723.delta]
theorem rows0_18 : MergeDeltasAt history frameStart owner group deltas0_18 := by
  exact .leaf LeftMerge119723.deltaAt
def deltas0_19 : Polynomial Owner := [LeftMerge119724.delta]
theorem rows0_19 : MergeDeltasAt history frameStart owner group deltas0_19 := by
  exact .leaf LeftMerge119724.deltaAt
def deltas0_20 : Polynomial Owner := [LeftMerge119725.delta]
theorem rows0_20 : MergeDeltasAt history frameStart owner group deltas0_20 := by
  exact .leaf LeftMerge119725.deltaAt
def deltas0_21 : Polynomial Owner := [LeftMerge119726.delta]
theorem rows0_21 : MergeDeltasAt history frameStart owner group deltas0_21 := by
  exact .leaf LeftMerge119726.deltaAt
def deltas0_22 : Polynomial Owner := [LeftMerge119727.delta]
theorem rows0_22 : MergeDeltasAt history frameStart owner group deltas0_22 := by
  exact .leaf LeftMerge119727.deltaAt
def deltas0_23 : Polynomial Owner := [LeftMerge119728.delta]
theorem rows0_23 : MergeDeltasAt history frameStart owner group deltas0_23 := by
  exact .leaf LeftMerge119728.deltaAt
def deltas0_24 : Polynomial Owner := [LeftMerge119729.delta]
theorem rows0_24 : MergeDeltasAt history frameStart owner group deltas0_24 := by
  exact .leaf LeftMerge119729.deltaAt
def deltas0_25 : Polynomial Owner := [LeftMerge119730.delta]
theorem rows0_25 : MergeDeltasAt history frameStart owner group deltas0_25 := by
  exact .leaf LeftMerge119730.deltaAt
def deltas0_26 : Polynomial Owner := [LeftMerge119731.delta]
theorem rows0_26 : MergeDeltasAt history frameStart owner group deltas0_26 := by
  exact .leaf LeftMerge119731.deltaAt
def deltas0_27 : Polynomial Owner := [LeftMerge119732.delta]
theorem rows0_27 : MergeDeltasAt history frameStart owner group deltas0_27 := by
  exact .leaf LeftMerge119732.deltaAt
def deltas0_28 : Polynomial Owner := [LeftMerge119733.delta]
theorem rows0_28 : MergeDeltasAt history frameStart owner group deltas0_28 := by
  exact .leaf LeftMerge119733.deltaAt
def deltas0_29 : Polynomial Owner := [LeftMerge119734.delta]
theorem rows0_29 : MergeDeltasAt history frameStart owner group deltas0_29 := by
  exact .leaf LeftMerge119734.deltaAt
def deltas0_30 : Polynomial Owner := [LeftMerge119735.delta]
theorem rows0_30 : MergeDeltasAt history frameStart owner group deltas0_30 := by
  exact .leaf LeftMerge119735.deltaAt
def deltas0_31 : Polynomial Owner := [LeftMerge119736.delta]
theorem rows0_31 : MergeDeltasAt history frameStart owner group deltas0_31 := by
  exact .leaf LeftMerge119736.deltaAt
def deltas0_32 : Polynomial Owner := [LeftMerge119737.delta]
theorem rows0_32 : MergeDeltasAt history frameStart owner group deltas0_32 := by
  exact .leaf LeftMerge119737.deltaAt
def deltas0_33 : Polynomial Owner := [LeftMerge119738.delta]
theorem rows0_33 : MergeDeltasAt history frameStart owner group deltas0_33 := by
  exact .leaf LeftMerge119738.deltaAt
def deltas0_34 : Polynomial Owner := [LeftMerge119739.delta]
theorem rows0_34 : MergeDeltasAt history frameStart owner group deltas0_34 := by
  exact .leaf LeftMerge119739.deltaAt
def deltas0_35 : Polynomial Owner := [LeftMerge119740.delta]
theorem rows0_35 : MergeDeltasAt history frameStart owner group deltas0_35 := by
  exact .leaf LeftMerge119740.deltaAt
def deltas0_36 : Polynomial Owner := [LeftMerge119741.delta]
theorem rows0_36 : MergeDeltasAt history frameStart owner group deltas0_36 := by
  exact .leaf LeftMerge119741.deltaAt
def deltas0_37 : Polynomial Owner := [LeftMerge119742.delta]
theorem rows0_37 : MergeDeltasAt history frameStart owner group deltas0_37 := by
  exact .leaf LeftMerge119742.deltaAt
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
def deltas1_18 : Polynomial Owner := deltas0_36 ++ deltas0_37
theorem rows1_18 : MergeDeltasAt history frameStart owner group deltas1_18 := by
  exact .append rows0_36 rows0_37
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
def deltas3_4 : Polynomial Owner := deltas2_8 ++ deltas1_18
theorem rows3_4 : MergeDeltasAt history frameStart owner group deltas3_4 := by
  exact .append rows2_8 rows1_18
def deltas4_0 : Polynomial Owner := deltas3_0 ++ deltas3_1
theorem rows4_0 : MergeDeltasAt history frameStart owner group deltas4_0 := by
  exact .append rows3_0 rows3_1
def deltas4_1 : Polynomial Owner := deltas3_2 ++ deltas3_3
theorem rows4_1 : MergeDeltasAt history frameStart owner group deltas4_1 := by
  exact .append rows3_2 rows3_3
def deltas5_0 : Polynomial Owner := deltas4_0 ++ deltas4_1
theorem rows5_0 : MergeDeltasAt history frameStart owner group deltas5_0 := by
  exact .append rows4_0 rows4_1
def deltas6_0 : Polynomial Owner := deltas5_0 ++ deltas3_4
theorem rows6_0 : MergeDeltasAt history frameStart owner group deltas6_0 := by
  exact .append rows5_0 rows3_4
abbrev deltas : Polynomial Owner := deltas6_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows6_0
def left : Polynomial Owner := LeftMerge119705.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge119705.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63009⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60029⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57049⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54069⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51089⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67382⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32025⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48307⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45627⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨42950⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22005⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37587⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34907⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18785⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29250⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26570⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15966⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66308⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63009⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7249⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60029⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7249⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57049⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7249⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54069⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7249⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51089⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7249⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67382⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7249⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32025⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7249⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48307⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7249⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45627⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7249⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨42950⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7249⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22005⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7249⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7249⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37587⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7249⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34907⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7249⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18785⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7249⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29250⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7249⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26570⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7249⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15966⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7249⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66308⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7249⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge119705

namespace LeftOperatorMerge7315
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨67400⟩⟩
def group : MergeGroup := .operator 7311 36
def deltas0_0 : Polynomial Owner := [LeftMerge7315.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge7315.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge7315.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge7315.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67399⟩⟩], orderedFactors := [] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge7315

namespace LeftOperatorMerge7540
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨67403⟩⟩
def group : MergeGroup := .operator 7536 6813
def deltas0_0 : Polynomial Owner := [LeftMerge7540.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge7540.deltaAt
def deltas0_1 : Polynomial Owner := [LeftMerge7541.delta]
theorem rows0_1 : MergeDeltasAt history frameStart owner group deltas0_1 := by
  exact .leaf LeftMerge7541.deltaAt
def deltas0_2 : Polynomial Owner := [LeftMerge7542.delta]
theorem rows0_2 : MergeDeltasAt history frameStart owner group deltas0_2 := by
  exact .leaf LeftMerge7542.deltaAt
def deltas0_3 : Polynomial Owner := [LeftMerge7543.delta]
theorem rows0_3 : MergeDeltasAt history frameStart owner group deltas0_3 := by
  exact .leaf LeftMerge7543.deltaAt
def deltas0_4 : Polynomial Owner := [LeftMerge7544.delta]
theorem rows0_4 : MergeDeltasAt history frameStart owner group deltas0_4 := by
  exact .leaf LeftMerge7544.deltaAt
def deltas0_5 : Polynomial Owner := [LeftMerge7545.delta]
theorem rows0_5 : MergeDeltasAt history frameStart owner group deltas0_5 := by
  exact .leaf LeftMerge7545.deltaAt
def deltas0_6 : Polynomial Owner := [LeftMerge7546.delta]
theorem rows0_6 : MergeDeltasAt history frameStart owner group deltas0_6 := by
  exact .leaf LeftMerge7546.deltaAt
def deltas0_7 : Polynomial Owner := [LeftMerge7547.delta]
theorem rows0_7 : MergeDeltasAt history frameStart owner group deltas0_7 := by
  exact .leaf LeftMerge7547.deltaAt
def deltas0_8 : Polynomial Owner := [LeftMerge7548.delta]
theorem rows0_8 : MergeDeltasAt history frameStart owner group deltas0_8 := by
  exact .leaf LeftMerge7548.deltaAt
def deltas0_9 : Polynomial Owner := [LeftMerge7549.delta]
theorem rows0_9 : MergeDeltasAt history frameStart owner group deltas0_9 := by
  exact .leaf LeftMerge7549.deltaAt
def deltas0_10 : Polynomial Owner := [LeftMerge7550.delta]
theorem rows0_10 : MergeDeltasAt history frameStart owner group deltas0_10 := by
  exact .leaf LeftMerge7550.deltaAt
def deltas0_11 : Polynomial Owner := [LeftMerge7551.delta]
theorem rows0_11 : MergeDeltasAt history frameStart owner group deltas0_11 := by
  exact .leaf LeftMerge7551.deltaAt
def deltas0_12 : Polynomial Owner := [LeftMerge7552.delta]
theorem rows0_12 : MergeDeltasAt history frameStart owner group deltas0_12 := by
  exact .leaf LeftMerge7552.deltaAt
def deltas0_13 : Polynomial Owner := [LeftMerge7553.delta]
theorem rows0_13 : MergeDeltasAt history frameStart owner group deltas0_13 := by
  exact .leaf LeftMerge7553.deltaAt
def deltas0_14 : Polynomial Owner := [LeftMerge7554.delta]
theorem rows0_14 : MergeDeltasAt history frameStart owner group deltas0_14 := by
  exact .leaf LeftMerge7554.deltaAt
def deltas0_15 : Polynomial Owner := [LeftMerge7555.delta]
theorem rows0_15 : MergeDeltasAt history frameStart owner group deltas0_15 := by
  exact .leaf LeftMerge7555.deltaAt
def deltas0_16 : Polynomial Owner := [LeftMerge7556.delta]
theorem rows0_16 : MergeDeltasAt history frameStart owner group deltas0_16 := by
  exact .leaf LeftMerge7556.deltaAt
def deltas0_17 : Polynomial Owner := [LeftMerge7557.delta]
theorem rows0_17 : MergeDeltasAt history frameStart owner group deltas0_17 := by
  exact .leaf LeftMerge7557.deltaAt
def deltas0_18 : Polynomial Owner := [LeftMerge7558.delta]
theorem rows0_18 : MergeDeltasAt history frameStart owner group deltas0_18 := by
  exact .leaf LeftMerge7558.deltaAt
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
def deltas2_4 : Polynomial Owner := deltas1_8 ++ deltas0_18
theorem rows2_4 : MergeDeltasAt history frameStart owner group deltas2_4 := by
  exact .append rows1_8 rows0_18
def deltas3_0 : Polynomial Owner := deltas2_0 ++ deltas2_1
theorem rows3_0 : MergeDeltasAt history frameStart owner group deltas3_0 := by
  exact .append rows2_0 rows2_1
def deltas3_1 : Polynomial Owner := deltas2_2 ++ deltas2_3
theorem rows3_1 : MergeDeltasAt history frameStart owner group deltas3_1 := by
  exact .append rows2_2 rows2_3
def deltas4_0 : Polynomial Owner := deltas3_0 ++ deltas3_1
theorem rows4_0 : MergeDeltasAt history frameStart owner group deltas4_0 := by
  exact .append rows3_0 rows3_1
def deltas5_0 : Polynomial Owner := deltas4_0 ++ deltas2_4
theorem rows5_0 : MergeDeltasAt history frameStart owner group deltas5_0 := by
  exact .append rows4_0 rows2_4
abbrev deltas : Polynomial Owner := deltas5_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows5_0
def left : Polynomial Owner := LeftMerge7540.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge7540.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨6771⟩⟩, ⟨.program ⟨257⟩, ⟨63028⟩⟩], orderedFactors := [] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨6771⟩⟩, ⟨.program ⟨257⟩, ⟨60048⟩⟩], orderedFactors := [] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨6771⟩⟩, ⟨.program ⟨257⟩, ⟨57068⟩⟩], orderedFactors := [] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨6771⟩⟩, ⟨.program ⟨257⟩, ⟨54088⟩⟩], orderedFactors := [] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨6771⟩⟩, ⟨.program ⟨257⟩, ⟨51108⟩⟩], orderedFactors := [] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6771⟩⟩, ⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67399⟩⟩], orderedFactors := [] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6771⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32044⟩⟩], orderedFactors := [] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6771⟩⟩, ⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48320⟩⟩], orderedFactors := [] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6771⟩⟩, ⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45640⟩⟩], orderedFactors := [] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6771⟩⟩, ⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨42963⟩⟩], orderedFactors := [] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6771⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22024⟩⟩], orderedFactors := [] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6771⟩⟩, ⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40283⟩⟩], orderedFactors := [] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6771⟩⟩, ⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37600⟩⟩], orderedFactors := [] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6771⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34920⟩⟩], orderedFactors := [] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6771⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18804⟩⟩], orderedFactors := [] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6771⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29263⟩⟩], orderedFactors := [] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6771⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26583⟩⟩], orderedFactors := [] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6771⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15982⟩⟩], orderedFactors := [] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6771⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66378⟩⟩], orderedFactors := [] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge7540

namespace LeftOperatorMerge148955
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨67404⟩⟩
def group : MergeGroup := .operator 148913 7536
def deltas0_0 : Polynomial Owner := [LeftMerge148955.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge148955.deltaAt
def deltas0_1 : Polynomial Owner := [LeftMerge148956.delta]
theorem rows0_1 : MergeDeltasAt history frameStart owner group deltas0_1 := by
  exact .leaf LeftMerge148956.deltaAt
def deltas0_2 : Polynomial Owner := [LeftMerge148957.delta]
theorem rows0_2 : MergeDeltasAt history frameStart owner group deltas0_2 := by
  exact .leaf LeftMerge148957.deltaAt
def deltas0_3 : Polynomial Owner := [LeftMerge148958.delta]
theorem rows0_3 : MergeDeltasAt history frameStart owner group deltas0_3 := by
  exact .leaf LeftMerge148958.deltaAt
def deltas0_4 : Polynomial Owner := [LeftMerge148959.delta]
theorem rows0_4 : MergeDeltasAt history frameStart owner group deltas0_4 := by
  exact .leaf LeftMerge148959.deltaAt
def deltas0_5 : Polynomial Owner := [LeftMerge148960.delta]
theorem rows0_5 : MergeDeltasAt history frameStart owner group deltas0_5 := by
  exact .leaf LeftMerge148960.deltaAt
def deltas0_6 : Polynomial Owner := [LeftMerge148961.delta]
theorem rows0_6 : MergeDeltasAt history frameStart owner group deltas0_6 := by
  exact .leaf LeftMerge148961.deltaAt
def deltas0_7 : Polynomial Owner := [LeftMerge148962.delta]
theorem rows0_7 : MergeDeltasAt history frameStart owner group deltas0_7 := by
  exact .leaf LeftMerge148962.deltaAt
def deltas0_8 : Polynomial Owner := [LeftMerge148963.delta]
theorem rows0_8 : MergeDeltasAt history frameStart owner group deltas0_8 := by
  exact .leaf LeftMerge148963.deltaAt
def deltas0_9 : Polynomial Owner := [LeftMerge148964.delta]
theorem rows0_9 : MergeDeltasAt history frameStart owner group deltas0_9 := by
  exact .leaf LeftMerge148964.deltaAt
def deltas0_10 : Polynomial Owner := [LeftMerge148965.delta]
theorem rows0_10 : MergeDeltasAt history frameStart owner group deltas0_10 := by
  exact .leaf LeftMerge148965.deltaAt
def deltas0_11 : Polynomial Owner := [LeftMerge148966.delta]
theorem rows0_11 : MergeDeltasAt history frameStart owner group deltas0_11 := by
  exact .leaf LeftMerge148966.deltaAt
def deltas0_12 : Polynomial Owner := [LeftMerge148967.delta]
theorem rows0_12 : MergeDeltasAt history frameStart owner group deltas0_12 := by
  exact .leaf LeftMerge148967.deltaAt
def deltas0_13 : Polynomial Owner := [LeftMerge148968.delta]
theorem rows0_13 : MergeDeltasAt history frameStart owner group deltas0_13 := by
  exact .leaf LeftMerge148968.deltaAt
def deltas0_14 : Polynomial Owner := [LeftMerge148969.delta]
theorem rows0_14 : MergeDeltasAt history frameStart owner group deltas0_14 := by
  exact .leaf LeftMerge148969.deltaAt
def deltas0_15 : Polynomial Owner := [LeftMerge148970.delta]
theorem rows0_15 : MergeDeltasAt history frameStart owner group deltas0_15 := by
  exact .leaf LeftMerge148970.deltaAt
def deltas0_16 : Polynomial Owner := [LeftMerge148971.delta]
theorem rows0_16 : MergeDeltasAt history frameStart owner group deltas0_16 := by
  exact .leaf LeftMerge148971.deltaAt
def deltas0_17 : Polynomial Owner := [LeftMerge148972.delta]
theorem rows0_17 : MergeDeltasAt history frameStart owner group deltas0_17 := by
  exact .leaf LeftMerge148972.deltaAt
def deltas0_18 : Polynomial Owner := [LeftMerge148973.delta]
theorem rows0_18 : MergeDeltasAt history frameStart owner group deltas0_18 := by
  exact .leaf LeftMerge148973.deltaAt
def deltas0_19 : Polynomial Owner := [LeftMerge148974.delta]
theorem rows0_19 : MergeDeltasAt history frameStart owner group deltas0_19 := by
  exact .leaf LeftMerge148974.deltaAt
def deltas0_20 : Polynomial Owner := [LeftMerge148975.delta]
theorem rows0_20 : MergeDeltasAt history frameStart owner group deltas0_20 := by
  exact .leaf LeftMerge148975.deltaAt
def deltas0_21 : Polynomial Owner := [LeftMerge148976.delta]
theorem rows0_21 : MergeDeltasAt history frameStart owner group deltas0_21 := by
  exact .leaf LeftMerge148976.deltaAt
def deltas0_22 : Polynomial Owner := [LeftMerge148977.delta]
theorem rows0_22 : MergeDeltasAt history frameStart owner group deltas0_22 := by
  exact .leaf LeftMerge148977.deltaAt
def deltas0_23 : Polynomial Owner := [LeftMerge148978.delta]
theorem rows0_23 : MergeDeltasAt history frameStart owner group deltas0_23 := by
  exact .leaf LeftMerge148978.deltaAt
def deltas0_24 : Polynomial Owner := [LeftMerge148979.delta]
theorem rows0_24 : MergeDeltasAt history frameStart owner group deltas0_24 := by
  exact .leaf LeftMerge148979.deltaAt
def deltas0_25 : Polynomial Owner := [LeftMerge148980.delta]
theorem rows0_25 : MergeDeltasAt history frameStart owner group deltas0_25 := by
  exact .leaf LeftMerge148980.deltaAt
def deltas0_26 : Polynomial Owner := [LeftMerge148981.delta]
theorem rows0_26 : MergeDeltasAt history frameStart owner group deltas0_26 := by
  exact .leaf LeftMerge148981.deltaAt
def deltas0_27 : Polynomial Owner := [LeftMerge148982.delta]
theorem rows0_27 : MergeDeltasAt history frameStart owner group deltas0_27 := by
  exact .leaf LeftMerge148982.deltaAt
def deltas0_28 : Polynomial Owner := [LeftMerge148983.delta]
theorem rows0_28 : MergeDeltasAt history frameStart owner group deltas0_28 := by
  exact .leaf LeftMerge148983.deltaAt
def deltas0_29 : Polynomial Owner := [LeftMerge148984.delta]
theorem rows0_29 : MergeDeltasAt history frameStart owner group deltas0_29 := by
  exact .leaf LeftMerge148984.deltaAt
def deltas0_30 : Polynomial Owner := [LeftMerge148985.delta]
theorem rows0_30 : MergeDeltasAt history frameStart owner group deltas0_30 := by
  exact .leaf LeftMerge148985.deltaAt
def deltas0_31 : Polynomial Owner := [LeftMerge148986.delta]
theorem rows0_31 : MergeDeltasAt history frameStart owner group deltas0_31 := by
  exact .leaf LeftMerge148986.deltaAt
def deltas0_32 : Polynomial Owner := [LeftMerge148987.delta]
theorem rows0_32 : MergeDeltasAt history frameStart owner group deltas0_32 := by
  exact .leaf LeftMerge148987.deltaAt
def deltas0_33 : Polynomial Owner := [LeftMerge148988.delta]
theorem rows0_33 : MergeDeltasAt history frameStart owner group deltas0_33 := by
  exact .leaf LeftMerge148988.deltaAt
def deltas0_34 : Polynomial Owner := [LeftMerge148989.delta]
theorem rows0_34 : MergeDeltasAt history frameStart owner group deltas0_34 := by
  exact .leaf LeftMerge148989.deltaAt
def deltas0_35 : Polynomial Owner := [LeftMerge148990.delta]
theorem rows0_35 : MergeDeltasAt history frameStart owner group deltas0_35 := by
  exact .leaf LeftMerge148990.deltaAt
def deltas0_36 : Polynomial Owner := [LeftMerge148991.delta]
theorem rows0_36 : MergeDeltasAt history frameStart owner group deltas0_36 := by
  exact .leaf LeftMerge148991.deltaAt
def deltas0_37 : Polynomial Owner := [LeftMerge148992.delta]
theorem rows0_37 : MergeDeltasAt history frameStart owner group deltas0_37 := by
  exact .leaf LeftMerge148992.deltaAt
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
def deltas1_18 : Polynomial Owner := deltas0_36 ++ deltas0_37
theorem rows1_18 : MergeDeltasAt history frameStart owner group deltas1_18 := by
  exact .append rows0_36 rows0_37
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
def deltas3_4 : Polynomial Owner := deltas2_8 ++ deltas1_18
theorem rows3_4 : MergeDeltasAt history frameStart owner group deltas3_4 := by
  exact .append rows2_8 rows1_18
def deltas4_0 : Polynomial Owner := deltas3_0 ++ deltas3_1
theorem rows4_0 : MergeDeltasAt history frameStart owner group deltas4_0 := by
  exact .append rows3_0 rows3_1
def deltas4_1 : Polynomial Owner := deltas3_2 ++ deltas3_3
theorem rows4_1 : MergeDeltasAt history frameStart owner group deltas4_1 := by
  exact .append rows3_2 rows3_3
def deltas5_0 : Polynomial Owner := deltas4_0 ++ deltas4_1
theorem rows5_0 : MergeDeltasAt history frameStart owner group deltas5_0 := by
  exact .append rows4_0 rows4_1
def deltas6_0 : Polynomial Owner := deltas5_0 ++ deltas3_4
theorem rows6_0 : MergeDeltasAt history frameStart owner group deltas6_0 := by
  exact .append rows5_0 rows3_4
abbrev deltas : Polynomial Owner := deltas6_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows6_0
def left : Polynomial Owner := LeftMerge148955.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge148955.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63028⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60048⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57068⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54088⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51108⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67399⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32044⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48320⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45640⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨42963⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22024⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40283⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37600⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34920⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18804⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29263⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26583⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15982⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63028⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7253⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60048⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7253⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57068⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7253⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54088⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7253⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51108⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7253⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67399⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7253⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32044⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7253⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48320⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7253⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45640⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7253⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨42963⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7253⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22024⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7253⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40283⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7253⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37600⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7253⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34920⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7253⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18804⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7253⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29263⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7253⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26583⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7253⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15982⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7253⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7253⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge148955

namespace LeftOperatorMerge11803
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨67418⟩⟩
def group : MergeGroup := .operator 11799 36
def deltas0_0 : Polynomial Owner := [LeftMerge11803.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge11803.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge11803.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge11803.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67417⟩⟩], orderedFactors := [] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge11803

namespace LeftOperatorMerge12028
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨67421⟩⟩
def group : MergeGroup := .operator 12024 11301
def deltas0_0 : Polynomial Owner := [LeftMerge12028.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge12028.deltaAt
def deltas0_1 : Polynomial Owner := [LeftMerge12029.delta]
theorem rows0_1 : MergeDeltasAt history frameStart owner group deltas0_1 := by
  exact .leaf LeftMerge12029.deltaAt
def deltas0_2 : Polynomial Owner := [LeftMerge12030.delta]
theorem rows0_2 : MergeDeltasAt history frameStart owner group deltas0_2 := by
  exact .leaf LeftMerge12030.deltaAt
def deltas0_3 : Polynomial Owner := [LeftMerge12031.delta]
theorem rows0_3 : MergeDeltasAt history frameStart owner group deltas0_3 := by
  exact .leaf LeftMerge12031.deltaAt
def deltas0_4 : Polynomial Owner := [LeftMerge12032.delta]
theorem rows0_4 : MergeDeltasAt history frameStart owner group deltas0_4 := by
  exact .leaf LeftMerge12032.deltaAt
def deltas0_5 : Polynomial Owner := [LeftMerge12033.delta]
theorem rows0_5 : MergeDeltasAt history frameStart owner group deltas0_5 := by
  exact .leaf LeftMerge12033.deltaAt
def deltas0_6 : Polynomial Owner := [LeftMerge12034.delta]
theorem rows0_6 : MergeDeltasAt history frameStart owner group deltas0_6 := by
  exact .leaf LeftMerge12034.deltaAt
def deltas0_7 : Polynomial Owner := [LeftMerge12035.delta]
theorem rows0_7 : MergeDeltasAt history frameStart owner group deltas0_7 := by
  exact .leaf LeftMerge12035.deltaAt
def deltas0_8 : Polynomial Owner := [LeftMerge12036.delta]
theorem rows0_8 : MergeDeltasAt history frameStart owner group deltas0_8 := by
  exact .leaf LeftMerge12036.deltaAt
def deltas0_9 : Polynomial Owner := [LeftMerge12037.delta]
theorem rows0_9 : MergeDeltasAt history frameStart owner group deltas0_9 := by
  exact .leaf LeftMerge12037.deltaAt
def deltas0_10 : Polynomial Owner := [LeftMerge12038.delta]
theorem rows0_10 : MergeDeltasAt history frameStart owner group deltas0_10 := by
  exact .leaf LeftMerge12038.deltaAt
def deltas0_11 : Polynomial Owner := [LeftMerge12039.delta]
theorem rows0_11 : MergeDeltasAt history frameStart owner group deltas0_11 := by
  exact .leaf LeftMerge12039.deltaAt
def deltas0_12 : Polynomial Owner := [LeftMerge12040.delta]
theorem rows0_12 : MergeDeltasAt history frameStart owner group deltas0_12 := by
  exact .leaf LeftMerge12040.deltaAt
def deltas0_13 : Polynomial Owner := [LeftMerge12041.delta]
theorem rows0_13 : MergeDeltasAt history frameStart owner group deltas0_13 := by
  exact .leaf LeftMerge12041.deltaAt
def deltas0_14 : Polynomial Owner := [LeftMerge12042.delta]
theorem rows0_14 : MergeDeltasAt history frameStart owner group deltas0_14 := by
  exact .leaf LeftMerge12042.deltaAt
def deltas0_15 : Polynomial Owner := [LeftMerge12043.delta]
theorem rows0_15 : MergeDeltasAt history frameStart owner group deltas0_15 := by
  exact .leaf LeftMerge12043.deltaAt
def deltas0_16 : Polynomial Owner := [LeftMerge12044.delta]
theorem rows0_16 : MergeDeltasAt history frameStart owner group deltas0_16 := by
  exact .leaf LeftMerge12044.deltaAt
def deltas0_17 : Polynomial Owner := [LeftMerge12045.delta]
theorem rows0_17 : MergeDeltasAt history frameStart owner group deltas0_17 := by
  exact .leaf LeftMerge12045.deltaAt
def deltas0_18 : Polynomial Owner := [LeftMerge12046.delta]
theorem rows0_18 : MergeDeltasAt history frameStart owner group deltas0_18 := by
  exact .leaf LeftMerge12046.deltaAt
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
def deltas2_4 : Polynomial Owner := deltas1_8 ++ deltas0_18
theorem rows2_4 : MergeDeltasAt history frameStart owner group deltas2_4 := by
  exact .append rows1_8 rows0_18
def deltas3_0 : Polynomial Owner := deltas2_0 ++ deltas2_1
theorem rows3_0 : MergeDeltasAt history frameStart owner group deltas3_0 := by
  exact .append rows2_0 rows2_1
def deltas3_1 : Polynomial Owner := deltas2_2 ++ deltas2_3
theorem rows3_1 : MergeDeltasAt history frameStart owner group deltas3_1 := by
  exact .append rows2_2 rows2_3
def deltas4_0 : Polynomial Owner := deltas3_0 ++ deltas3_1
theorem rows4_0 : MergeDeltasAt history frameStart owner group deltas4_0 := by
  exact .append rows3_0 rows3_1
def deltas5_0 : Polynomial Owner := deltas4_0 ++ deltas2_4
theorem rows5_0 : MergeDeltasAt history frameStart owner group deltas5_0 := by
  exact .append rows4_0 rows2_4
abbrev deltas : Polynomial Owner := deltas5_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows5_0
def left : Polynomial Owner := LeftMerge12028.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge12028.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨6773⟩⟩, ⟨.program ⟨257⟩, ⟨63047⟩⟩], orderedFactors := [] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨6773⟩⟩, ⟨.program ⟨257⟩, ⟨60067⟩⟩], orderedFactors := [] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨6773⟩⟩, ⟨.program ⟨257⟩, ⟨57087⟩⟩], orderedFactors := [] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨6773⟩⟩, ⟨.program ⟨257⟩, ⟨54107⟩⟩], orderedFactors := [] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨6773⟩⟩, ⟨.program ⟨257⟩, ⟨51127⟩⟩], orderedFactors := [] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6773⟩⟩, ⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67417⟩⟩], orderedFactors := [] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6773⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32063⟩⟩], orderedFactors := [] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6773⟩⟩, ⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48333⟩⟩], orderedFactors := [] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6773⟩⟩, ⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45653⟩⟩], orderedFactors := [] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6773⟩⟩, ⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨42976⟩⟩], orderedFactors := [] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6773⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22043⟩⟩], orderedFactors := [] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6773⟩⟩, ⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40296⟩⟩], orderedFactors := [] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6773⟩⟩, ⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37613⟩⟩], orderedFactors := [] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6773⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34933⟩⟩], orderedFactors := [] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6773⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18823⟩⟩], orderedFactors := [] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6773⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29276⟩⟩], orderedFactors := [] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6773⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26596⟩⟩], orderedFactors := [] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6773⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15998⟩⟩], orderedFactors := [] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6773⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66448⟩⟩], orderedFactors := [] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge12028

namespace LeftOperatorMerge236705
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨67423⟩⟩
def group : MergeGroup := .operator 236663 12024
def deltas0_0 : Polynomial Owner := [LeftMerge236705.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge236705.deltaAt
def deltas0_1 : Polynomial Owner := [LeftMerge236706.delta]
theorem rows0_1 : MergeDeltasAt history frameStart owner group deltas0_1 := by
  exact .leaf LeftMerge236706.deltaAt
def deltas0_2 : Polynomial Owner := [LeftMerge236707.delta]
theorem rows0_2 : MergeDeltasAt history frameStart owner group deltas0_2 := by
  exact .leaf LeftMerge236707.deltaAt
def deltas0_3 : Polynomial Owner := [LeftMerge236708.delta]
theorem rows0_3 : MergeDeltasAt history frameStart owner group deltas0_3 := by
  exact .leaf LeftMerge236708.deltaAt
def deltas0_4 : Polynomial Owner := [LeftMerge236709.delta]
theorem rows0_4 : MergeDeltasAt history frameStart owner group deltas0_4 := by
  exact .leaf LeftMerge236709.deltaAt
def deltas0_5 : Polynomial Owner := [LeftMerge236710.delta]
theorem rows0_5 : MergeDeltasAt history frameStart owner group deltas0_5 := by
  exact .leaf LeftMerge236710.deltaAt
def deltas0_6 : Polynomial Owner := [LeftMerge236711.delta]
theorem rows0_6 : MergeDeltasAt history frameStart owner group deltas0_6 := by
  exact .leaf LeftMerge236711.deltaAt
def deltas0_7 : Polynomial Owner := [LeftMerge236712.delta]
theorem rows0_7 : MergeDeltasAt history frameStart owner group deltas0_7 := by
  exact .leaf LeftMerge236712.deltaAt
def deltas0_8 : Polynomial Owner := [LeftMerge236713.delta]
theorem rows0_8 : MergeDeltasAt history frameStart owner group deltas0_8 := by
  exact .leaf LeftMerge236713.deltaAt
def deltas0_9 : Polynomial Owner := [LeftMerge236714.delta]
theorem rows0_9 : MergeDeltasAt history frameStart owner group deltas0_9 := by
  exact .leaf LeftMerge236714.deltaAt
def deltas0_10 : Polynomial Owner := [LeftMerge236715.delta]
theorem rows0_10 : MergeDeltasAt history frameStart owner group deltas0_10 := by
  exact .leaf LeftMerge236715.deltaAt
def deltas0_11 : Polynomial Owner := [LeftMerge236716.delta]
theorem rows0_11 : MergeDeltasAt history frameStart owner group deltas0_11 := by
  exact .leaf LeftMerge236716.deltaAt
def deltas0_12 : Polynomial Owner := [LeftMerge236717.delta]
theorem rows0_12 : MergeDeltasAt history frameStart owner group deltas0_12 := by
  exact .leaf LeftMerge236717.deltaAt
def deltas0_13 : Polynomial Owner := [LeftMerge236718.delta]
theorem rows0_13 : MergeDeltasAt history frameStart owner group deltas0_13 := by
  exact .leaf LeftMerge236718.deltaAt
def deltas0_14 : Polynomial Owner := [LeftMerge236719.delta]
theorem rows0_14 : MergeDeltasAt history frameStart owner group deltas0_14 := by
  exact .leaf LeftMerge236719.deltaAt
def deltas0_15 : Polynomial Owner := [LeftMerge236720.delta]
theorem rows0_15 : MergeDeltasAt history frameStart owner group deltas0_15 := by
  exact .leaf LeftMerge236720.deltaAt
def deltas0_16 : Polynomial Owner := [LeftMerge236721.delta]
theorem rows0_16 : MergeDeltasAt history frameStart owner group deltas0_16 := by
  exact .leaf LeftMerge236721.deltaAt
def deltas0_17 : Polynomial Owner := [LeftMerge236722.delta]
theorem rows0_17 : MergeDeltasAt history frameStart owner group deltas0_17 := by
  exact .leaf LeftMerge236722.deltaAt
def deltas0_18 : Polynomial Owner := [LeftMerge236723.delta]
theorem rows0_18 : MergeDeltasAt history frameStart owner group deltas0_18 := by
  exact .leaf LeftMerge236723.deltaAt
def deltas0_19 : Polynomial Owner := [LeftMerge236724.delta]
theorem rows0_19 : MergeDeltasAt history frameStart owner group deltas0_19 := by
  exact .leaf LeftMerge236724.deltaAt
def deltas0_20 : Polynomial Owner := [LeftMerge236725.delta]
theorem rows0_20 : MergeDeltasAt history frameStart owner group deltas0_20 := by
  exact .leaf LeftMerge236725.deltaAt
def deltas0_21 : Polynomial Owner := [LeftMerge236726.delta]
theorem rows0_21 : MergeDeltasAt history frameStart owner group deltas0_21 := by
  exact .leaf LeftMerge236726.deltaAt
def deltas0_22 : Polynomial Owner := [LeftMerge236727.delta]
theorem rows0_22 : MergeDeltasAt history frameStart owner group deltas0_22 := by
  exact .leaf LeftMerge236727.deltaAt
def deltas0_23 : Polynomial Owner := [LeftMerge236728.delta]
theorem rows0_23 : MergeDeltasAt history frameStart owner group deltas0_23 := by
  exact .leaf LeftMerge236728.deltaAt
def deltas0_24 : Polynomial Owner := [LeftMerge236729.delta]
theorem rows0_24 : MergeDeltasAt history frameStart owner group deltas0_24 := by
  exact .leaf LeftMerge236729.deltaAt
def deltas0_25 : Polynomial Owner := [LeftMerge236730.delta]
theorem rows0_25 : MergeDeltasAt history frameStart owner group deltas0_25 := by
  exact .leaf LeftMerge236730.deltaAt
def deltas0_26 : Polynomial Owner := [LeftMerge236731.delta]
theorem rows0_26 : MergeDeltasAt history frameStart owner group deltas0_26 := by
  exact .leaf LeftMerge236731.deltaAt
def deltas0_27 : Polynomial Owner := [LeftMerge236732.delta]
theorem rows0_27 : MergeDeltasAt history frameStart owner group deltas0_27 := by
  exact .leaf LeftMerge236732.deltaAt
def deltas0_28 : Polynomial Owner := [LeftMerge236733.delta]
theorem rows0_28 : MergeDeltasAt history frameStart owner group deltas0_28 := by
  exact .leaf LeftMerge236733.deltaAt
def deltas0_29 : Polynomial Owner := [LeftMerge236734.delta]
theorem rows0_29 : MergeDeltasAt history frameStart owner group deltas0_29 := by
  exact .leaf LeftMerge236734.deltaAt
def deltas0_30 : Polynomial Owner := [LeftMerge236735.delta]
theorem rows0_30 : MergeDeltasAt history frameStart owner group deltas0_30 := by
  exact .leaf LeftMerge236735.deltaAt
def deltas0_31 : Polynomial Owner := [LeftMerge236736.delta]
theorem rows0_31 : MergeDeltasAt history frameStart owner group deltas0_31 := by
  exact .leaf LeftMerge236736.deltaAt
def deltas0_32 : Polynomial Owner := [LeftMerge236737.delta]
theorem rows0_32 : MergeDeltasAt history frameStart owner group deltas0_32 := by
  exact .leaf LeftMerge236737.deltaAt
def deltas0_33 : Polynomial Owner := [LeftMerge236738.delta]
theorem rows0_33 : MergeDeltasAt history frameStart owner group deltas0_33 := by
  exact .leaf LeftMerge236738.deltaAt
def deltas0_34 : Polynomial Owner := [LeftMerge236739.delta]
theorem rows0_34 : MergeDeltasAt history frameStart owner group deltas0_34 := by
  exact .leaf LeftMerge236739.deltaAt
def deltas0_35 : Polynomial Owner := [LeftMerge236740.delta]
theorem rows0_35 : MergeDeltasAt history frameStart owner group deltas0_35 := by
  exact .leaf LeftMerge236740.deltaAt
def deltas0_36 : Polynomial Owner := [LeftMerge236741.delta]
theorem rows0_36 : MergeDeltasAt history frameStart owner group deltas0_36 := by
  exact .leaf LeftMerge236741.deltaAt
def deltas0_37 : Polynomial Owner := [LeftMerge236742.delta]
theorem rows0_37 : MergeDeltasAt history frameStart owner group deltas0_37 := by
  exact .leaf LeftMerge236742.deltaAt
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
def deltas1_18 : Polynomial Owner := deltas0_36 ++ deltas0_37
theorem rows1_18 : MergeDeltasAt history frameStart owner group deltas1_18 := by
  exact .append rows0_36 rows0_37
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
def deltas3_4 : Polynomial Owner := deltas2_8 ++ deltas1_18
theorem rows3_4 : MergeDeltasAt history frameStart owner group deltas3_4 := by
  exact .append rows2_8 rows1_18
def deltas4_0 : Polynomial Owner := deltas3_0 ++ deltas3_1
theorem rows4_0 : MergeDeltasAt history frameStart owner group deltas4_0 := by
  exact .append rows3_0 rows3_1
def deltas4_1 : Polynomial Owner := deltas3_2 ++ deltas3_3
theorem rows4_1 : MergeDeltasAt history frameStart owner group deltas4_1 := by
  exact .append rows3_2 rows3_3
def deltas5_0 : Polynomial Owner := deltas4_0 ++ deltas4_1
theorem rows5_0 : MergeDeltasAt history frameStart owner group deltas5_0 := by
  exact .append rows4_0 rows4_1
def deltas6_0 : Polynomial Owner := deltas5_0 ++ deltas3_4
theorem rows6_0 : MergeDeltasAt history frameStart owner group deltas6_0 := by
  exact .append rows5_0 rows3_4
abbrev deltas : Polynomial Owner := deltas6_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows6_0
def left : Polynomial Owner := LeftMerge236705.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge236705.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63047⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60067⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57087⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54107⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51127⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67417⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32063⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48333⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45653⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨42976⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22043⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40296⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37613⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34933⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18823⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29276⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26596⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15998⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66448⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63047⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7265⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60067⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7265⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57087⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7265⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54107⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7265⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51127⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7265⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67417⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7265⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32063⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7265⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48333⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7265⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45653⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7265⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨42976⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7265⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22043⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7265⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40296⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7265⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37613⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7265⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34933⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7265⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18823⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7265⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29276⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7265⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26596⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7265⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15998⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7265⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66448⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7265⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge236705

namespace LeftOperatorMerge11055
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨67438⟩⟩
def group : MergeGroup := .operator 11051 36
def deltas0_0 : Polynomial Owner := [LeftMerge11055.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge11055.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge11055.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge11055.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67437⟩⟩], orderedFactors := [] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge11055

namespace LeftOperatorMerge11280
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨67441⟩⟩
def group : MergeGroup := .operator 11276 10553
def deltas0_0 : Polynomial Owner := [LeftMerge11280.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge11280.deltaAt
def deltas0_1 : Polynomial Owner := [LeftMerge11281.delta]
theorem rows0_1 : MergeDeltasAt history frameStart owner group deltas0_1 := by
  exact .leaf LeftMerge11281.deltaAt
def deltas0_2 : Polynomial Owner := [LeftMerge11282.delta]
theorem rows0_2 : MergeDeltasAt history frameStart owner group deltas0_2 := by
  exact .leaf LeftMerge11282.deltaAt
def deltas0_3 : Polynomial Owner := [LeftMerge11283.delta]
theorem rows0_3 : MergeDeltasAt history frameStart owner group deltas0_3 := by
  exact .leaf LeftMerge11283.deltaAt
def deltas0_4 : Polynomial Owner := [LeftMerge11284.delta]
theorem rows0_4 : MergeDeltasAt history frameStart owner group deltas0_4 := by
  exact .leaf LeftMerge11284.deltaAt
def deltas0_5 : Polynomial Owner := [LeftMerge11285.delta]
theorem rows0_5 : MergeDeltasAt history frameStart owner group deltas0_5 := by
  exact .leaf LeftMerge11285.deltaAt
def deltas0_6 : Polynomial Owner := [LeftMerge11286.delta]
theorem rows0_6 : MergeDeltasAt history frameStart owner group deltas0_6 := by
  exact .leaf LeftMerge11286.deltaAt
def deltas0_7 : Polynomial Owner := [LeftMerge11287.delta]
theorem rows0_7 : MergeDeltasAt history frameStart owner group deltas0_7 := by
  exact .leaf LeftMerge11287.deltaAt
def deltas0_8 : Polynomial Owner := [LeftMerge11288.delta]
theorem rows0_8 : MergeDeltasAt history frameStart owner group deltas0_8 := by
  exact .leaf LeftMerge11288.deltaAt
def deltas0_9 : Polynomial Owner := [LeftMerge11289.delta]
theorem rows0_9 : MergeDeltasAt history frameStart owner group deltas0_9 := by
  exact .leaf LeftMerge11289.deltaAt
def deltas0_10 : Polynomial Owner := [LeftMerge11290.delta]
theorem rows0_10 : MergeDeltasAt history frameStart owner group deltas0_10 := by
  exact .leaf LeftMerge11290.deltaAt
def deltas0_11 : Polynomial Owner := [LeftMerge11291.delta]
theorem rows0_11 : MergeDeltasAt history frameStart owner group deltas0_11 := by
  exact .leaf LeftMerge11291.deltaAt
def deltas0_12 : Polynomial Owner := [LeftMerge11292.delta]
theorem rows0_12 : MergeDeltasAt history frameStart owner group deltas0_12 := by
  exact .leaf LeftMerge11292.deltaAt
def deltas0_13 : Polynomial Owner := [LeftMerge11293.delta]
theorem rows0_13 : MergeDeltasAt history frameStart owner group deltas0_13 := by
  exact .leaf LeftMerge11293.deltaAt
def deltas0_14 : Polynomial Owner := [LeftMerge11294.delta]
theorem rows0_14 : MergeDeltasAt history frameStart owner group deltas0_14 := by
  exact .leaf LeftMerge11294.deltaAt
def deltas0_15 : Polynomial Owner := [LeftMerge11295.delta]
theorem rows0_15 : MergeDeltasAt history frameStart owner group deltas0_15 := by
  exact .leaf LeftMerge11295.deltaAt
def deltas0_16 : Polynomial Owner := [LeftMerge11296.delta]
theorem rows0_16 : MergeDeltasAt history frameStart owner group deltas0_16 := by
  exact .leaf LeftMerge11296.deltaAt
def deltas0_17 : Polynomial Owner := [LeftMerge11297.delta]
theorem rows0_17 : MergeDeltasAt history frameStart owner group deltas0_17 := by
  exact .leaf LeftMerge11297.deltaAt
def deltas0_18 : Polynomial Owner := [LeftMerge11298.delta]
theorem rows0_18 : MergeDeltasAt history frameStart owner group deltas0_18 := by
  exact .leaf LeftMerge11298.deltaAt
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
def deltas2_4 : Polynomial Owner := deltas1_8 ++ deltas0_18
theorem rows2_4 : MergeDeltasAt history frameStart owner group deltas2_4 := by
  exact .append rows1_8 rows0_18
def deltas3_0 : Polynomial Owner := deltas2_0 ++ deltas2_1
theorem rows3_0 : MergeDeltasAt history frameStart owner group deltas3_0 := by
  exact .append rows2_0 rows2_1
def deltas3_1 : Polynomial Owner := deltas2_2 ++ deltas2_3
theorem rows3_1 : MergeDeltasAt history frameStart owner group deltas3_1 := by
  exact .append rows2_2 rows2_3
def deltas4_0 : Polynomial Owner := deltas3_0 ++ deltas3_1
theorem rows4_0 : MergeDeltasAt history frameStart owner group deltas4_0 := by
  exact .append rows3_0 rows3_1
def deltas5_0 : Polynomial Owner := deltas4_0 ++ deltas2_4
theorem rows5_0 : MergeDeltasAt history frameStart owner group deltas5_0 := by
  exact .append rows4_0 rows2_4
abbrev deltas : Polynomial Owner := deltas5_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows5_0
def left : Polynomial Owner := LeftMerge11280.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge11280.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨6748⟩⟩, ⟨.program ⟨257⟩, ⟨63066⟩⟩], orderedFactors := [] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨6748⟩⟩, ⟨.program ⟨257⟩, ⟨60086⟩⟩], orderedFactors := [] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨6748⟩⟩, ⟨.program ⟨257⟩, ⟨57106⟩⟩], orderedFactors := [] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6748⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54126⟩⟩], orderedFactors := [] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6748⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51146⟩⟩], orderedFactors := [] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6748⟩⟩, ⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67437⟩⟩], orderedFactors := [] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6748⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32082⟩⟩], orderedFactors := [] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6748⟩⟩, ⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48346⟩⟩], orderedFactors := [] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6748⟩⟩, ⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45666⟩⟩], orderedFactors := [] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6748⟩⟩, ⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨42989⟩⟩], orderedFactors := [] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6748⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22062⟩⟩], orderedFactors := [] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6748⟩⟩, ⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40309⟩⟩], orderedFactors := [] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6748⟩⟩, ⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37626⟩⟩], orderedFactors := [] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6748⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34946⟩⟩], orderedFactors := [] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6748⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18842⟩⟩], orderedFactors := [] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6748⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29289⟩⟩], orderedFactors := [] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6748⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26609⟩⟩], orderedFactors := [] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6748⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16014⟩⟩], orderedFactors := [] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6748⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66518⟩⟩], orderedFactors := [] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge11280

namespace LeftOperatorMerge222080
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨67443⟩⟩
def group : MergeGroup := .operator 222038 11276
def deltas0_0 : Polynomial Owner := [LeftMerge222080.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge222080.deltaAt
def deltas0_1 : Polynomial Owner := [LeftMerge222081.delta]
theorem rows0_1 : MergeDeltasAt history frameStart owner group deltas0_1 := by
  exact .leaf LeftMerge222081.deltaAt
def deltas0_2 : Polynomial Owner := [LeftMerge222082.delta]
theorem rows0_2 : MergeDeltasAt history frameStart owner group deltas0_2 := by
  exact .leaf LeftMerge222082.deltaAt
def deltas0_3 : Polynomial Owner := [LeftMerge222083.delta]
theorem rows0_3 : MergeDeltasAt history frameStart owner group deltas0_3 := by
  exact .leaf LeftMerge222083.deltaAt
def deltas0_4 : Polynomial Owner := [LeftMerge222084.delta]
theorem rows0_4 : MergeDeltasAt history frameStart owner group deltas0_4 := by
  exact .leaf LeftMerge222084.deltaAt
def deltas0_5 : Polynomial Owner := [LeftMerge222085.delta]
theorem rows0_5 : MergeDeltasAt history frameStart owner group deltas0_5 := by
  exact .leaf LeftMerge222085.deltaAt
def deltas0_6 : Polynomial Owner := [LeftMerge222086.delta]
theorem rows0_6 : MergeDeltasAt history frameStart owner group deltas0_6 := by
  exact .leaf LeftMerge222086.deltaAt
def deltas0_7 : Polynomial Owner := [LeftMerge222087.delta]
theorem rows0_7 : MergeDeltasAt history frameStart owner group deltas0_7 := by
  exact .leaf LeftMerge222087.deltaAt
def deltas0_8 : Polynomial Owner := [LeftMerge222088.delta]
theorem rows0_8 : MergeDeltasAt history frameStart owner group deltas0_8 := by
  exact .leaf LeftMerge222088.deltaAt
def deltas0_9 : Polynomial Owner := [LeftMerge222089.delta]
theorem rows0_9 : MergeDeltasAt history frameStart owner group deltas0_9 := by
  exact .leaf LeftMerge222089.deltaAt
def deltas0_10 : Polynomial Owner := [LeftMerge222090.delta]
theorem rows0_10 : MergeDeltasAt history frameStart owner group deltas0_10 := by
  exact .leaf LeftMerge222090.deltaAt
def deltas0_11 : Polynomial Owner := [LeftMerge222091.delta]
theorem rows0_11 : MergeDeltasAt history frameStart owner group deltas0_11 := by
  exact .leaf LeftMerge222091.deltaAt
def deltas0_12 : Polynomial Owner := [LeftMerge222092.delta]
theorem rows0_12 : MergeDeltasAt history frameStart owner group deltas0_12 := by
  exact .leaf LeftMerge222092.deltaAt
def deltas0_13 : Polynomial Owner := [LeftMerge222093.delta]
theorem rows0_13 : MergeDeltasAt history frameStart owner group deltas0_13 := by
  exact .leaf LeftMerge222093.deltaAt
def deltas0_14 : Polynomial Owner := [LeftMerge222094.delta]
theorem rows0_14 : MergeDeltasAt history frameStart owner group deltas0_14 := by
  exact .leaf LeftMerge222094.deltaAt
def deltas0_15 : Polynomial Owner := [LeftMerge222095.delta]
theorem rows0_15 : MergeDeltasAt history frameStart owner group deltas0_15 := by
  exact .leaf LeftMerge222095.deltaAt
def deltas0_16 : Polynomial Owner := [LeftMerge222096.delta]
theorem rows0_16 : MergeDeltasAt history frameStart owner group deltas0_16 := by
  exact .leaf LeftMerge222096.deltaAt
def deltas0_17 : Polynomial Owner := [LeftMerge222097.delta]
theorem rows0_17 : MergeDeltasAt history frameStart owner group deltas0_17 := by
  exact .leaf LeftMerge222097.deltaAt
def deltas0_18 : Polynomial Owner := [LeftMerge222098.delta]
theorem rows0_18 : MergeDeltasAt history frameStart owner group deltas0_18 := by
  exact .leaf LeftMerge222098.deltaAt
def deltas0_19 : Polynomial Owner := [LeftMerge222099.delta]
theorem rows0_19 : MergeDeltasAt history frameStart owner group deltas0_19 := by
  exact .leaf LeftMerge222099.deltaAt
def deltas0_20 : Polynomial Owner := [LeftMerge222100.delta]
theorem rows0_20 : MergeDeltasAt history frameStart owner group deltas0_20 := by
  exact .leaf LeftMerge222100.deltaAt
def deltas0_21 : Polynomial Owner := [LeftMerge222101.delta]
theorem rows0_21 : MergeDeltasAt history frameStart owner group deltas0_21 := by
  exact .leaf LeftMerge222101.deltaAt
def deltas0_22 : Polynomial Owner := [LeftMerge222102.delta]
theorem rows0_22 : MergeDeltasAt history frameStart owner group deltas0_22 := by
  exact .leaf LeftMerge222102.deltaAt
def deltas0_23 : Polynomial Owner := [LeftMerge222103.delta]
theorem rows0_23 : MergeDeltasAt history frameStart owner group deltas0_23 := by
  exact .leaf LeftMerge222103.deltaAt
def deltas0_24 : Polynomial Owner := [LeftMerge222104.delta]
theorem rows0_24 : MergeDeltasAt history frameStart owner group deltas0_24 := by
  exact .leaf LeftMerge222104.deltaAt
def deltas0_25 : Polynomial Owner := [LeftMerge222105.delta]
theorem rows0_25 : MergeDeltasAt history frameStart owner group deltas0_25 := by
  exact .leaf LeftMerge222105.deltaAt
def deltas0_26 : Polynomial Owner := [LeftMerge222106.delta]
theorem rows0_26 : MergeDeltasAt history frameStart owner group deltas0_26 := by
  exact .leaf LeftMerge222106.deltaAt
def deltas0_27 : Polynomial Owner := [LeftMerge222107.delta]
theorem rows0_27 : MergeDeltasAt history frameStart owner group deltas0_27 := by
  exact .leaf LeftMerge222107.deltaAt
def deltas0_28 : Polynomial Owner := [LeftMerge222108.delta]
theorem rows0_28 : MergeDeltasAt history frameStart owner group deltas0_28 := by
  exact .leaf LeftMerge222108.deltaAt
def deltas0_29 : Polynomial Owner := [LeftMerge222109.delta]
theorem rows0_29 : MergeDeltasAt history frameStart owner group deltas0_29 := by
  exact .leaf LeftMerge222109.deltaAt
def deltas0_30 : Polynomial Owner := [LeftMerge222110.delta]
theorem rows0_30 : MergeDeltasAt history frameStart owner group deltas0_30 := by
  exact .leaf LeftMerge222110.deltaAt
def deltas0_31 : Polynomial Owner := [LeftMerge222111.delta]
theorem rows0_31 : MergeDeltasAt history frameStart owner group deltas0_31 := by
  exact .leaf LeftMerge222111.deltaAt
def deltas0_32 : Polynomial Owner := [LeftMerge222112.delta]
theorem rows0_32 : MergeDeltasAt history frameStart owner group deltas0_32 := by
  exact .leaf LeftMerge222112.deltaAt
def deltas0_33 : Polynomial Owner := [LeftMerge222113.delta]
theorem rows0_33 : MergeDeltasAt history frameStart owner group deltas0_33 := by
  exact .leaf LeftMerge222113.deltaAt
def deltas0_34 : Polynomial Owner := [LeftMerge222114.delta]
theorem rows0_34 : MergeDeltasAt history frameStart owner group deltas0_34 := by
  exact .leaf LeftMerge222114.deltaAt
def deltas0_35 : Polynomial Owner := [LeftMerge222115.delta]
theorem rows0_35 : MergeDeltasAt history frameStart owner group deltas0_35 := by
  exact .leaf LeftMerge222115.deltaAt
def deltas0_36 : Polynomial Owner := [LeftMerge222116.delta]
theorem rows0_36 : MergeDeltasAt history frameStart owner group deltas0_36 := by
  exact .leaf LeftMerge222116.deltaAt
def deltas0_37 : Polynomial Owner := [LeftMerge222117.delta]
theorem rows0_37 : MergeDeltasAt history frameStart owner group deltas0_37 := by
  exact .leaf LeftMerge222117.deltaAt
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
def deltas1_18 : Polynomial Owner := deltas0_36 ++ deltas0_37
theorem rows1_18 : MergeDeltasAt history frameStart owner group deltas1_18 := by
  exact .append rows0_36 rows0_37
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
def deltas3_4 : Polynomial Owner := deltas2_8 ++ deltas1_18
theorem rows3_4 : MergeDeltasAt history frameStart owner group deltas3_4 := by
  exact .append rows2_8 rows1_18
def deltas4_0 : Polynomial Owner := deltas3_0 ++ deltas3_1
theorem rows4_0 : MergeDeltasAt history frameStart owner group deltas4_0 := by
  exact .append rows3_0 rows3_1
def deltas4_1 : Polynomial Owner := deltas3_2 ++ deltas3_3
theorem rows4_1 : MergeDeltasAt history frameStart owner group deltas4_1 := by
  exact .append rows3_2 rows3_3
def deltas5_0 : Polynomial Owner := deltas4_0 ++ deltas4_1
theorem rows5_0 : MergeDeltasAt history frameStart owner group deltas5_0 := by
  exact .append rows4_0 rows4_1
def deltas6_0 : Polynomial Owner := deltas5_0 ++ deltas3_4
theorem rows6_0 : MergeDeltasAt history frameStart owner group deltas6_0 := by
  exact .append rows5_0 rows3_4
abbrev deltas : Polynomial Owner := deltas6_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows6_0
def left : Polynomial Owner := LeftMerge222080.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge222080.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63066⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60086⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57106⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54126⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51146⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67437⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32082⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48346⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45666⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨42989⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22062⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40309⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37626⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34946⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18842⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29289⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26609⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16014⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66518⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63066⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7263⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60086⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7263⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57106⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7263⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54126⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7263⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51146⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7263⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67437⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7263⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32082⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7263⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48346⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7263⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45666⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7263⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨42989⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7263⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22062⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7263⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40309⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7263⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37626⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7263⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34946⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7263⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18842⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7263⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29289⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7263⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26609⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7263⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16014⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7263⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66518⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7263⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge222080

namespace LeftOperatorMerge10307
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨67458⟩⟩
def group : MergeGroup := .operator 10303 36
def deltas0_0 : Polynomial Owner := [LeftMerge10307.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge10307.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge10307.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge10307.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67457⟩⟩], orderedFactors := [] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge10307

namespace LeftOperatorMerge10532
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨67461⟩⟩
def group : MergeGroup := .operator 10528 9805
def deltas0_0 : Polynomial Owner := [LeftMerge10532.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge10532.deltaAt
def deltas0_1 : Polynomial Owner := [LeftMerge10533.delta]
theorem rows0_1 : MergeDeltasAt history frameStart owner group deltas0_1 := by
  exact .leaf LeftMerge10533.deltaAt
def deltas0_2 : Polynomial Owner := [LeftMerge10534.delta]
theorem rows0_2 : MergeDeltasAt history frameStart owner group deltas0_2 := by
  exact .leaf LeftMerge10534.deltaAt
def deltas0_3 : Polynomial Owner := [LeftMerge10535.delta]
theorem rows0_3 : MergeDeltasAt history frameStart owner group deltas0_3 := by
  exact .leaf LeftMerge10535.deltaAt
def deltas0_4 : Polynomial Owner := [LeftMerge10536.delta]
theorem rows0_4 : MergeDeltasAt history frameStart owner group deltas0_4 := by
  exact .leaf LeftMerge10536.deltaAt
def deltas0_5 : Polynomial Owner := [LeftMerge10537.delta]
theorem rows0_5 : MergeDeltasAt history frameStart owner group deltas0_5 := by
  exact .leaf LeftMerge10537.deltaAt
def deltas0_6 : Polynomial Owner := [LeftMerge10538.delta]
theorem rows0_6 : MergeDeltasAt history frameStart owner group deltas0_6 := by
  exact .leaf LeftMerge10538.deltaAt
def deltas0_7 : Polynomial Owner := [LeftMerge10539.delta]
theorem rows0_7 : MergeDeltasAt history frameStart owner group deltas0_7 := by
  exact .leaf LeftMerge10539.deltaAt
def deltas0_8 : Polynomial Owner := [LeftMerge10540.delta]
theorem rows0_8 : MergeDeltasAt history frameStart owner group deltas0_8 := by
  exact .leaf LeftMerge10540.deltaAt
def deltas0_9 : Polynomial Owner := [LeftMerge10541.delta]
theorem rows0_9 : MergeDeltasAt history frameStart owner group deltas0_9 := by
  exact .leaf LeftMerge10541.deltaAt
def deltas0_10 : Polynomial Owner := [LeftMerge10542.delta]
theorem rows0_10 : MergeDeltasAt history frameStart owner group deltas0_10 := by
  exact .leaf LeftMerge10542.deltaAt
def deltas0_11 : Polynomial Owner := [LeftMerge10543.delta]
theorem rows0_11 : MergeDeltasAt history frameStart owner group deltas0_11 := by
  exact .leaf LeftMerge10543.deltaAt
def deltas0_12 : Polynomial Owner := [LeftMerge10544.delta]
theorem rows0_12 : MergeDeltasAt history frameStart owner group deltas0_12 := by
  exact .leaf LeftMerge10544.deltaAt
def deltas0_13 : Polynomial Owner := [LeftMerge10545.delta]
theorem rows0_13 : MergeDeltasAt history frameStart owner group deltas0_13 := by
  exact .leaf LeftMerge10545.deltaAt
def deltas0_14 : Polynomial Owner := [LeftMerge10546.delta]
theorem rows0_14 : MergeDeltasAt history frameStart owner group deltas0_14 := by
  exact .leaf LeftMerge10546.deltaAt
def deltas0_15 : Polynomial Owner := [LeftMerge10547.delta]
theorem rows0_15 : MergeDeltasAt history frameStart owner group deltas0_15 := by
  exact .leaf LeftMerge10547.deltaAt
def deltas0_16 : Polynomial Owner := [LeftMerge10548.delta]
theorem rows0_16 : MergeDeltasAt history frameStart owner group deltas0_16 := by
  exact .leaf LeftMerge10548.deltaAt
def deltas0_17 : Polynomial Owner := [LeftMerge10549.delta]
theorem rows0_17 : MergeDeltasAt history frameStart owner group deltas0_17 := by
  exact .leaf LeftMerge10549.deltaAt
def deltas0_18 : Polynomial Owner := [LeftMerge10550.delta]
theorem rows0_18 : MergeDeltasAt history frameStart owner group deltas0_18 := by
  exact .leaf LeftMerge10550.deltaAt
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
def deltas2_4 : Polynomial Owner := deltas1_8 ++ deltas0_18
theorem rows2_4 : MergeDeltasAt history frameStart owner group deltas2_4 := by
  exact .append rows1_8 rows0_18
def deltas3_0 : Polynomial Owner := deltas2_0 ++ deltas2_1
theorem rows3_0 : MergeDeltasAt history frameStart owner group deltas3_0 := by
  exact .append rows2_0 rows2_1
def deltas3_1 : Polynomial Owner := deltas2_2 ++ deltas2_3
theorem rows3_1 : MergeDeltasAt history frameStart owner group deltas3_1 := by
  exact .append rows2_2 rows2_3
def deltas4_0 : Polynomial Owner := deltas3_0 ++ deltas3_1
theorem rows4_0 : MergeDeltasAt history frameStart owner group deltas4_0 := by
  exact .append rows3_0 rows3_1
def deltas5_0 : Polynomial Owner := deltas4_0 ++ deltas2_4
theorem rows5_0 : MergeDeltasAt history frameStart owner group deltas5_0 := by
  exact .append rows4_0 rows2_4
abbrev deltas : Polynomial Owner := deltas5_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows5_0
def left : Polynomial Owner := LeftMerge10532.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge10532.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨6770⟩⟩, ⟨.program ⟨257⟩, ⟨63085⟩⟩], orderedFactors := [] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨6770⟩⟩, ⟨.program ⟨257⟩, ⟨60105⟩⟩], orderedFactors := [] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨6770⟩⟩, ⟨.program ⟨257⟩, ⟨57125⟩⟩], orderedFactors := [] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨6770⟩⟩, ⟨.program ⟨257⟩, ⟨54145⟩⟩], orderedFactors := [] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨6770⟩⟩, ⟨.program ⟨257⟩, ⟨51165⟩⟩], orderedFactors := [] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6770⟩⟩, ⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67457⟩⟩], orderedFactors := [] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6770⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32101⟩⟩], orderedFactors := [] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6770⟩⟩, ⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48359⟩⟩], orderedFactors := [] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6770⟩⟩, ⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45679⟩⟩], orderedFactors := [] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6770⟩⟩, ⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨43002⟩⟩], orderedFactors := [] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6770⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22081⟩⟩], orderedFactors := [] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6770⟩⟩, ⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40322⟩⟩], orderedFactors := [] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6770⟩⟩, ⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37639⟩⟩], orderedFactors := [] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6770⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34959⟩⟩], orderedFactors := [] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6770⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18861⟩⟩], orderedFactors := [] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6770⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29302⟩⟩], orderedFactors := [] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6770⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26622⟩⟩], orderedFactors := [] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6770⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16030⟩⟩], orderedFactors := [] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6770⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66588⟩⟩], orderedFactors := [] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge10532

namespace LeftOperatorMerge207455
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨67463⟩⟩
def group : MergeGroup := .operator 207413 10528
def deltas0_0 : Polynomial Owner := [LeftMerge207455.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge207455.deltaAt
def deltas0_1 : Polynomial Owner := [LeftMerge207456.delta]
theorem rows0_1 : MergeDeltasAt history frameStart owner group deltas0_1 := by
  exact .leaf LeftMerge207456.deltaAt
def deltas0_2 : Polynomial Owner := [LeftMerge207457.delta]
theorem rows0_2 : MergeDeltasAt history frameStart owner group deltas0_2 := by
  exact .leaf LeftMerge207457.deltaAt
def deltas0_3 : Polynomial Owner := [LeftMerge207458.delta]
theorem rows0_3 : MergeDeltasAt history frameStart owner group deltas0_3 := by
  exact .leaf LeftMerge207458.deltaAt
def deltas0_4 : Polynomial Owner := [LeftMerge207459.delta]
theorem rows0_4 : MergeDeltasAt history frameStart owner group deltas0_4 := by
  exact .leaf LeftMerge207459.deltaAt
def deltas0_5 : Polynomial Owner := [LeftMerge207460.delta]
theorem rows0_5 : MergeDeltasAt history frameStart owner group deltas0_5 := by
  exact .leaf LeftMerge207460.deltaAt
def deltas0_6 : Polynomial Owner := [LeftMerge207461.delta]
theorem rows0_6 : MergeDeltasAt history frameStart owner group deltas0_6 := by
  exact .leaf LeftMerge207461.deltaAt
def deltas0_7 : Polynomial Owner := [LeftMerge207462.delta]
theorem rows0_7 : MergeDeltasAt history frameStart owner group deltas0_7 := by
  exact .leaf LeftMerge207462.deltaAt
def deltas0_8 : Polynomial Owner := [LeftMerge207463.delta]
theorem rows0_8 : MergeDeltasAt history frameStart owner group deltas0_8 := by
  exact .leaf LeftMerge207463.deltaAt
def deltas0_9 : Polynomial Owner := [LeftMerge207464.delta]
theorem rows0_9 : MergeDeltasAt history frameStart owner group deltas0_9 := by
  exact .leaf LeftMerge207464.deltaAt
def deltas0_10 : Polynomial Owner := [LeftMerge207465.delta]
theorem rows0_10 : MergeDeltasAt history frameStart owner group deltas0_10 := by
  exact .leaf LeftMerge207465.deltaAt
def deltas0_11 : Polynomial Owner := [LeftMerge207466.delta]
theorem rows0_11 : MergeDeltasAt history frameStart owner group deltas0_11 := by
  exact .leaf LeftMerge207466.deltaAt
def deltas0_12 : Polynomial Owner := [LeftMerge207467.delta]
theorem rows0_12 : MergeDeltasAt history frameStart owner group deltas0_12 := by
  exact .leaf LeftMerge207467.deltaAt
def deltas0_13 : Polynomial Owner := [LeftMerge207468.delta]
theorem rows0_13 : MergeDeltasAt history frameStart owner group deltas0_13 := by
  exact .leaf LeftMerge207468.deltaAt
def deltas0_14 : Polynomial Owner := [LeftMerge207469.delta]
theorem rows0_14 : MergeDeltasAt history frameStart owner group deltas0_14 := by
  exact .leaf LeftMerge207469.deltaAt
def deltas0_15 : Polynomial Owner := [LeftMerge207470.delta]
theorem rows0_15 : MergeDeltasAt history frameStart owner group deltas0_15 := by
  exact .leaf LeftMerge207470.deltaAt
def deltas0_16 : Polynomial Owner := [LeftMerge207471.delta]
theorem rows0_16 : MergeDeltasAt history frameStart owner group deltas0_16 := by
  exact .leaf LeftMerge207471.deltaAt
def deltas0_17 : Polynomial Owner := [LeftMerge207472.delta]
theorem rows0_17 : MergeDeltasAt history frameStart owner group deltas0_17 := by
  exact .leaf LeftMerge207472.deltaAt
def deltas0_18 : Polynomial Owner := [LeftMerge207473.delta]
theorem rows0_18 : MergeDeltasAt history frameStart owner group deltas0_18 := by
  exact .leaf LeftMerge207473.deltaAt
def deltas0_19 : Polynomial Owner := [LeftMerge207474.delta]
theorem rows0_19 : MergeDeltasAt history frameStart owner group deltas0_19 := by
  exact .leaf LeftMerge207474.deltaAt
def deltas0_20 : Polynomial Owner := [LeftMerge207475.delta]
theorem rows0_20 : MergeDeltasAt history frameStart owner group deltas0_20 := by
  exact .leaf LeftMerge207475.deltaAt
def deltas0_21 : Polynomial Owner := [LeftMerge207476.delta]
theorem rows0_21 : MergeDeltasAt history frameStart owner group deltas0_21 := by
  exact .leaf LeftMerge207476.deltaAt
def deltas0_22 : Polynomial Owner := [LeftMerge207477.delta]
theorem rows0_22 : MergeDeltasAt history frameStart owner group deltas0_22 := by
  exact .leaf LeftMerge207477.deltaAt
def deltas0_23 : Polynomial Owner := [LeftMerge207478.delta]
theorem rows0_23 : MergeDeltasAt history frameStart owner group deltas0_23 := by
  exact .leaf LeftMerge207478.deltaAt
def deltas0_24 : Polynomial Owner := [LeftMerge207479.delta]
theorem rows0_24 : MergeDeltasAt history frameStart owner group deltas0_24 := by
  exact .leaf LeftMerge207479.deltaAt
def deltas0_25 : Polynomial Owner := [LeftMerge207480.delta]
theorem rows0_25 : MergeDeltasAt history frameStart owner group deltas0_25 := by
  exact .leaf LeftMerge207480.deltaAt
def deltas0_26 : Polynomial Owner := [LeftMerge207481.delta]
theorem rows0_26 : MergeDeltasAt history frameStart owner group deltas0_26 := by
  exact .leaf LeftMerge207481.deltaAt
def deltas0_27 : Polynomial Owner := [LeftMerge207482.delta]
theorem rows0_27 : MergeDeltasAt history frameStart owner group deltas0_27 := by
  exact .leaf LeftMerge207482.deltaAt
def deltas0_28 : Polynomial Owner := [LeftMerge207483.delta]
theorem rows0_28 : MergeDeltasAt history frameStart owner group deltas0_28 := by
  exact .leaf LeftMerge207483.deltaAt
def deltas0_29 : Polynomial Owner := [LeftMerge207484.delta]
theorem rows0_29 : MergeDeltasAt history frameStart owner group deltas0_29 := by
  exact .leaf LeftMerge207484.deltaAt
def deltas0_30 : Polynomial Owner := [LeftMerge207485.delta]
theorem rows0_30 : MergeDeltasAt history frameStart owner group deltas0_30 := by
  exact .leaf LeftMerge207485.deltaAt
def deltas0_31 : Polynomial Owner := [LeftMerge207486.delta]
theorem rows0_31 : MergeDeltasAt history frameStart owner group deltas0_31 := by
  exact .leaf LeftMerge207486.deltaAt
def deltas0_32 : Polynomial Owner := [LeftMerge207487.delta]
theorem rows0_32 : MergeDeltasAt history frameStart owner group deltas0_32 := by
  exact .leaf LeftMerge207487.deltaAt
def deltas0_33 : Polynomial Owner := [LeftMerge207488.delta]
theorem rows0_33 : MergeDeltasAt history frameStart owner group deltas0_33 := by
  exact .leaf LeftMerge207488.deltaAt
def deltas0_34 : Polynomial Owner := [LeftMerge207489.delta]
theorem rows0_34 : MergeDeltasAt history frameStart owner group deltas0_34 := by
  exact .leaf LeftMerge207489.deltaAt
def deltas0_35 : Polynomial Owner := [LeftMerge207490.delta]
theorem rows0_35 : MergeDeltasAt history frameStart owner group deltas0_35 := by
  exact .leaf LeftMerge207490.deltaAt
def deltas0_36 : Polynomial Owner := [LeftMerge207491.delta]
theorem rows0_36 : MergeDeltasAt history frameStart owner group deltas0_36 := by
  exact .leaf LeftMerge207491.deltaAt
def deltas0_37 : Polynomial Owner := [LeftMerge207492.delta]
theorem rows0_37 : MergeDeltasAt history frameStart owner group deltas0_37 := by
  exact .leaf LeftMerge207492.deltaAt
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
def deltas1_18 : Polynomial Owner := deltas0_36 ++ deltas0_37
theorem rows1_18 : MergeDeltasAt history frameStart owner group deltas1_18 := by
  exact .append rows0_36 rows0_37
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
def deltas3_4 : Polynomial Owner := deltas2_8 ++ deltas1_18
theorem rows3_4 : MergeDeltasAt history frameStart owner group deltas3_4 := by
  exact .append rows2_8 rows1_18
def deltas4_0 : Polynomial Owner := deltas3_0 ++ deltas3_1
theorem rows4_0 : MergeDeltasAt history frameStart owner group deltas4_0 := by
  exact .append rows3_0 rows3_1
def deltas4_1 : Polynomial Owner := deltas3_2 ++ deltas3_3
theorem rows4_1 : MergeDeltasAt history frameStart owner group deltas4_1 := by
  exact .append rows3_2 rows3_3
def deltas5_0 : Polynomial Owner := deltas4_0 ++ deltas4_1
theorem rows5_0 : MergeDeltasAt history frameStart owner group deltas5_0 := by
  exact .append rows4_0 rows4_1
def deltas6_0 : Polynomial Owner := deltas5_0 ++ deltas3_4
theorem rows6_0 : MergeDeltasAt history frameStart owner group deltas6_0 := by
  exact .append rows5_0 rows3_4
abbrev deltas : Polynomial Owner := deltas6_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows6_0
def left : Polynomial Owner := LeftMerge207455.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge207455.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63085⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60105⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57125⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54145⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51165⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67457⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32101⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48359⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45679⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨43002⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22081⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40322⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37639⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34959⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18861⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29302⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26622⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16030⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66588⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63085⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7261⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60105⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7261⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57125⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7261⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54145⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7261⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51165⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7261⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67457⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7261⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32101⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7261⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48359⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7261⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45679⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7261⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨43002⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7261⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22081⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7261⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40322⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7261⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37639⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7261⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34959⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7261⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18861⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7261⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29302⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7261⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26622⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7261⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16030⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7261⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66588⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7261⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge207455

namespace LeftOperatorMerge5071
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨67477⟩⟩
def group : MergeGroup := .operator 5067 36
def deltas0_0 : Polynomial Owner := [LeftMerge5071.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge5071.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge5071.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge5071.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67476⟩⟩], orderedFactors := [] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge5071

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
