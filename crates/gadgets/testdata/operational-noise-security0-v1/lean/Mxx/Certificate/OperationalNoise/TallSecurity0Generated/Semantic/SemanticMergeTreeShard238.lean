import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeDeltaShard626
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeDeltaShard628
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeDeltaShard629
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeDeltaShard633
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeDeltaShard634
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeDeltaShard635
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeDeltaShard636
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeDeltaShard637
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeDeltaShard643

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace LeftOperatorMerge101688
def frameStart : Nat := 101634
def owner : Owner := ⟨.program ⟨214⟩, ⟨14987⟩⟩
def group : MergeGroup := .operator 101684 101682
def deltas0_0 : Polynomial Owner := [LeftMerge101688.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge101688.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge101688.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge101688.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨14943⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge101688

namespace LeftOperatorMerge101711
def frameStart : Nat := 101634
def owner : Owner := ⟨.program ⟨214⟩, ⟨15302⟩⟩
def group : MergeGroup := .operator 101684 101707
def deltas0_0 : Polynomial Owner := [LeftMerge101711.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge101711.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge101711.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge101711.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15300⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge101711

namespace LeftOperatorMerge101700
def frameStart : Nat := 101634
def owner : Owner := ⟨.program ⟨214⟩, ⟨26530⟩⟩
def group : MergeGroup := .operator 101696 101673
def deltas0_0 : Polynomial Owner := [LeftMerge101700.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge101700.deltaAt
def deltas0_1 : Polynomial Owner := [LeftMerge101701.delta]
theorem rows0_1 : MergeDeltasAt history frameStart owner group deltas0_1 := by
  exact .leaf LeftMerge101701.deltaAt
def deltas1_0 : Polynomial Owner := deltas0_0 ++ deltas0_1
theorem rows1_0 : MergeDeltasAt history frameStart owner group deltas1_0 := by
  exact .append rows0_0 rows0_1
abbrev deltas : Polynomial Owner := deltas1_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows1_0
def left : Polynomial Owner := LeftMerge101700.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge101700.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6691⟩⟩, ⟨.program ⟨214⟩, ⟨26529⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨14943⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26529⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge101700

namespace LeftOperatorMerge101954
def frameStart : Nat := 101883
def owner : Owner := ⟨.program ⟨214⟩, ⟨7833⟩⟩
def group : MergeGroup := .operator 101950 101947
def deltas0_0 : Polynomial Owner := [LeftMerge101954.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge101954.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge101954.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge101954.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6771⟩⟩, ⟨.program ⟨214⟩, ⟨7831⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge101954

namespace LeftOperatorMerge101901
def frameStart : Nat := 101883
def owner : Owner := ⟨.program ⟨214⟩, ⟨10457⟩⟩
def group : MergeGroup := .operator 101897 101894
def deltas0_0 : Polynomial Owner := [LeftMerge101901.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge101901.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge101901.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge101901.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨9385⟩⟩, ⟨.program ⟨214⟩, ⟨10456⟩⟩], orderedFactors := [] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge101901

namespace LeftOperatorMerge101931
def frameStart : Nat := 101883
def owner : Owner := ⟨.program ⟨214⟩, ⟨10570⟩⟩
def group : MergeGroup := .operator 101927 101925
def deltas0_0 : Polynomial Owner := [LeftMerge101931.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge101931.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge101931.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge101931.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨9385⟩⟩, ⟨.program ⟨214⟩, ⟨10456⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge101931

namespace LeftOperatorMerge101974
def frameStart : Nat := 101883
def owner : Owner := ⟨.program ⟨214⟩, ⟨14784⟩⟩
def group : MergeGroup := .operator 101927 101970
def deltas0_0 : Polynomial Owner := [LeftMerge101974.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge101974.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge101974.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge101974.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨14782⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge101974

namespace LeftOperatorMerge101963
def frameStart : Nat := 101883
def owner : Owner := ⟨.program ⟨214⟩, ⟨24901⟩⟩
def group : MergeGroup := .operator 101959 101916
def deltas0_0 : Polynomial Owner := [LeftMerge101963.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge101963.deltaAt
def deltas0_1 : Polynomial Owner := [LeftMerge101964.delta]
theorem rows0_1 : MergeDeltasAt history frameStart owner group deltas0_1 := by
  exact .leaf LeftMerge101964.deltaAt
def deltas1_0 : Polynomial Owner := deltas0_0 ++ deltas0_1
theorem rows1_0 : MergeDeltasAt history frameStart owner group deltas1_0 := by
  exact .append rows0_0 rows0_1
abbrev deltas : Polynomial Owner := deltas1_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows1_0
def left : Polynomial Owner := LeftMerge101963.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge101963.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6771⟩⟩, ⟨.program ⟨214⟩, ⟨7831⟩⟩, ⟨.program ⟨214⟩, ⟨24898⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨9385⟩⟩, ⟨.program ⟨214⟩, ⟨10456⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨24898⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge101963

namespace LeftOperatorMerge102122
def frameStart : Nat := 102068
def owner : Owner := ⟨.program ⟨214⟩, ⟨14826⟩⟩
def group : MergeGroup := .operator 102118 102116
def deltas0_0 : Polynomial Owner := [LeftMerge102122.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge102122.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge102122.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge102122.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨14782⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge102122

namespace LeftOperatorMerge102145
def frameStart : Nat := 102068
def owner : Owner := ⟨.program ⟨214⟩, ⟨15259⟩⟩
def group : MergeGroup := .operator 102118 102141
def deltas0_0 : Polynomial Owner := [LeftMerge102145.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge102145.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge102145.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge102145.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15258⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge102145

namespace LeftOperatorMerge102134
def frameStart : Nat := 102068
def owner : Owner := ⟨.program ⟨214⟩, ⟨26327⟩⟩
def group : MergeGroup := .operator 102130 102107
def deltas0_0 : Polynomial Owner := [LeftMerge102134.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge102134.deltaAt
def deltas0_1 : Polynomial Owner := [LeftMerge102135.delta]
theorem rows0_1 : MergeDeltasAt history frameStart owner group deltas0_1 := by
  exact .leaf LeftMerge102135.deltaAt
def deltas1_0 : Polynomial Owner := deltas0_0 ++ deltas0_1
theorem rows1_0 : MergeDeltasAt history frameStart owner group deltas1_0 := by
  exact .append rows0_0 rows0_1
abbrev deltas : Polynomial Owner := deltas1_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows1_0
def left : Polynomial Owner := LeftMerge102134.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge102134.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6690⟩⟩, ⟨.program ⟨214⟩, ⟨26326⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨14782⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26326⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge102134

namespace LeftOperatorMerge103673
def frameStart : Nat := 102927
def owner : Owner := ⟨.program ⟨214⟩, ⟨18487⟩⟩
def group : MergeGroup := .operator 103442 103669
def deltas0_0 : Polynomial Owner := [LeftMerge103673.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge103673.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge103673.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge103673.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨18485⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge103673

namespace LeftOperatorMerge103446
def frameStart : Nat := 102927
def owner : Owner := ⟨.program ⟨214⟩, ⟨18641⟩⟩
def group : MergeGroup := .operator 103442 103440
def deltas0_0 : Polynomial Owner := [LeftMerge103446.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge103446.deltaAt
def deltas0_1 : Polynomial Owner := [LeftMerge103447.delta]
theorem rows0_1 : MergeDeltasAt history frameStart owner group deltas0_1 := by
  exact .leaf LeftMerge103447.deltaAt
def deltas0_2 : Polynomial Owner := [LeftMerge103448.delta]
theorem rows0_2 : MergeDeltasAt history frameStart owner group deltas0_2 := by
  exact .leaf LeftMerge103448.deltaAt
def deltas0_3 : Polynomial Owner := [LeftMerge103449.delta]
theorem rows0_3 : MergeDeltasAt history frameStart owner group deltas0_3 := by
  exact .leaf LeftMerge103449.deltaAt
def deltas0_4 : Polynomial Owner := [LeftMerge103450.delta]
theorem rows0_4 : MergeDeltasAt history frameStart owner group deltas0_4 := by
  exact .leaf LeftMerge103450.deltaAt
def deltas0_5 : Polynomial Owner := [LeftMerge103451.delta]
theorem rows0_5 : MergeDeltasAt history frameStart owner group deltas0_5 := by
  exact .leaf LeftMerge103451.deltaAt
def deltas0_6 : Polynomial Owner := [LeftMerge103452.delta]
theorem rows0_6 : MergeDeltasAt history frameStart owner group deltas0_6 := by
  exact .leaf LeftMerge103452.deltaAt
def deltas0_7 : Polynomial Owner := [LeftMerge103453.delta]
theorem rows0_7 : MergeDeltasAt history frameStart owner group deltas0_7 := by
  exact .leaf LeftMerge103453.deltaAt
def deltas0_8 : Polynomial Owner := [LeftMerge103454.delta]
theorem rows0_8 : MergeDeltasAt history frameStart owner group deltas0_8 := by
  exact .leaf LeftMerge103454.deltaAt
def deltas0_9 : Polynomial Owner := [LeftMerge103455.delta]
theorem rows0_9 : MergeDeltasAt history frameStart owner group deltas0_9 := by
  exact .leaf LeftMerge103455.deltaAt
def deltas0_10 : Polynomial Owner := [LeftMerge103456.delta]
theorem rows0_10 : MergeDeltasAt history frameStart owner group deltas0_10 := by
  exact .leaf LeftMerge103456.deltaAt
def deltas0_11 : Polynomial Owner := [LeftMerge103457.delta]
theorem rows0_11 : MergeDeltasAt history frameStart owner group deltas0_11 := by
  exact .leaf LeftMerge103457.deltaAt
def deltas0_12 : Polynomial Owner := [LeftMerge103458.delta]
theorem rows0_12 : MergeDeltasAt history frameStart owner group deltas0_12 := by
  exact .leaf LeftMerge103458.deltaAt
def deltas0_13 : Polynomial Owner := [LeftMerge103459.delta]
theorem rows0_13 : MergeDeltasAt history frameStart owner group deltas0_13 := by
  exact .leaf LeftMerge103459.deltaAt
def deltas0_14 : Polynomial Owner := [LeftMerge103460.delta]
theorem rows0_14 : MergeDeltasAt history frameStart owner group deltas0_14 := by
  exact .leaf LeftMerge103460.deltaAt
def deltas0_15 : Polynomial Owner := [LeftMerge103461.delta]
theorem rows0_15 : MergeDeltasAt history frameStart owner group deltas0_15 := by
  exact .leaf LeftMerge103461.deltaAt
def deltas0_16 : Polynomial Owner := [LeftMerge103462.delta]
theorem rows0_16 : MergeDeltasAt history frameStart owner group deltas0_16 := by
  exact .leaf LeftMerge103462.deltaAt
def deltas0_17 : Polynomial Owner := [LeftMerge103463.delta]
theorem rows0_17 : MergeDeltasAt history frameStart owner group deltas0_17 := by
  exact .leaf LeftMerge103463.deltaAt
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
def left : Polynomial Owner := LeftMerge103446.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge103446.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15258⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15300⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15356⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15622⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15741⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15860⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15979⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16098⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16301⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16672⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16791⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨17078⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨17113⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨17302⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨17897⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨18163⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨18198⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨18303⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge103446

namespace LeftOperatorMerge103594
def frameStart : Nat := 102927
def owner : Owner := ⟨.program ⟨214⟩, ⟨18675⟩⟩
def group : MergeGroup := .operator 103590 103431
def deltas0_0 : Polynomial Owner := [LeftMerge103594.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge103594.deltaAt
def deltas0_1 : Polynomial Owner := [LeftMerge103595.delta]
theorem rows0_1 : MergeDeltasAt history frameStart owner group deltas0_1 := by
  exact .leaf LeftMerge103595.deltaAt
def deltas0_2 : Polynomial Owner := [LeftMerge103596.delta]
theorem rows0_2 : MergeDeltasAt history frameStart owner group deltas0_2 := by
  exact .leaf LeftMerge103596.deltaAt
def deltas0_3 : Polynomial Owner := [LeftMerge103597.delta]
theorem rows0_3 : MergeDeltasAt history frameStart owner group deltas0_3 := by
  exact .leaf LeftMerge103597.deltaAt
def deltas0_4 : Polynomial Owner := [LeftMerge103598.delta]
theorem rows0_4 : MergeDeltasAt history frameStart owner group deltas0_4 := by
  exact .leaf LeftMerge103598.deltaAt
def deltas0_5 : Polynomial Owner := [LeftMerge103599.delta]
theorem rows0_5 : MergeDeltasAt history frameStart owner group deltas0_5 := by
  exact .leaf LeftMerge103599.deltaAt
def deltas0_6 : Polynomial Owner := [LeftMerge103600.delta]
theorem rows0_6 : MergeDeltasAt history frameStart owner group deltas0_6 := by
  exact .leaf LeftMerge103600.deltaAt
def deltas0_7 : Polynomial Owner := [LeftMerge103601.delta]
theorem rows0_7 : MergeDeltasAt history frameStart owner group deltas0_7 := by
  exact .leaf LeftMerge103601.deltaAt
def deltas0_8 : Polynomial Owner := [LeftMerge103602.delta]
theorem rows0_8 : MergeDeltasAt history frameStart owner group deltas0_8 := by
  exact .leaf LeftMerge103602.deltaAt
def deltas0_9 : Polynomial Owner := [LeftMerge103603.delta]
theorem rows0_9 : MergeDeltasAt history frameStart owner group deltas0_9 := by
  exact .leaf LeftMerge103603.deltaAt
def deltas0_10 : Polynomial Owner := [LeftMerge103604.delta]
theorem rows0_10 : MergeDeltasAt history frameStart owner group deltas0_10 := by
  exact .leaf LeftMerge103604.deltaAt
def deltas0_11 : Polynomial Owner := [LeftMerge103605.delta]
theorem rows0_11 : MergeDeltasAt history frameStart owner group deltas0_11 := by
  exact .leaf LeftMerge103605.deltaAt
def deltas0_12 : Polynomial Owner := [LeftMerge103606.delta]
theorem rows0_12 : MergeDeltasAt history frameStart owner group deltas0_12 := by
  exact .leaf LeftMerge103606.deltaAt
def deltas0_13 : Polynomial Owner := [LeftMerge103607.delta]
theorem rows0_13 : MergeDeltasAt history frameStart owner group deltas0_13 := by
  exact .leaf LeftMerge103607.deltaAt
def deltas0_14 : Polynomial Owner := [LeftMerge103608.delta]
theorem rows0_14 : MergeDeltasAt history frameStart owner group deltas0_14 := by
  exact .leaf LeftMerge103608.deltaAt
def deltas0_15 : Polynomial Owner := [LeftMerge103609.delta]
theorem rows0_15 : MergeDeltasAt history frameStart owner group deltas0_15 := by
  exact .leaf LeftMerge103609.deltaAt
def deltas0_16 : Polynomial Owner := [LeftMerge103610.delta]
theorem rows0_16 : MergeDeltasAt history frameStart owner group deltas0_16 := by
  exact .leaf LeftMerge103610.deltaAt
def deltas0_17 : Polynomial Owner := [LeftMerge103611.delta]
theorem rows0_17 : MergeDeltasAt history frameStart owner group deltas0_17 := by
  exact .leaf LeftMerge103611.deltaAt
def deltas0_18 : Polynomial Owner := [LeftMerge103612.delta]
theorem rows0_18 : MergeDeltasAt history frameStart owner group deltas0_18 := by
  exact .leaf LeftMerge103612.deltaAt
def deltas0_19 : Polynomial Owner := [LeftMerge103615.delta]
theorem rows0_19 : MergeDeltasAt history frameStart owner group deltas0_19 := by
  exact .leaf LeftMerge103615.deltaAt
def deltas0_20 : Polynomial Owner := [LeftMerge103618.delta]
theorem rows0_20 : MergeDeltasAt history frameStart owner group deltas0_20 := by
  exact .leaf LeftMerge103618.deltaAt
def deltas0_21 : Polynomial Owner := [LeftMerge103621.delta]
theorem rows0_21 : MergeDeltasAt history frameStart owner group deltas0_21 := by
  exact .leaf LeftMerge103621.deltaAt
def deltas0_22 : Polynomial Owner := [LeftMerge103624.delta]
theorem rows0_22 : MergeDeltasAt history frameStart owner group deltas0_22 := by
  exact .leaf LeftMerge103624.deltaAt
def deltas0_23 : Polynomial Owner := [LeftMerge103627.delta]
theorem rows0_23 : MergeDeltasAt history frameStart owner group deltas0_23 := by
  exact .leaf LeftMerge103627.deltaAt
def deltas0_24 : Polynomial Owner := [LeftMerge103630.delta]
theorem rows0_24 : MergeDeltasAt history frameStart owner group deltas0_24 := by
  exact .leaf LeftMerge103630.deltaAt
def deltas0_25 : Polynomial Owner := [LeftMerge103633.delta]
theorem rows0_25 : MergeDeltasAt history frameStart owner group deltas0_25 := by
  exact .leaf LeftMerge103633.deltaAt
def deltas0_26 : Polynomial Owner := [LeftMerge103636.delta]
theorem rows0_26 : MergeDeltasAt history frameStart owner group deltas0_26 := by
  exact .leaf LeftMerge103636.deltaAt
def deltas0_27 : Polynomial Owner := [LeftMerge103639.delta]
theorem rows0_27 : MergeDeltasAt history frameStart owner group deltas0_27 := by
  exact .leaf LeftMerge103639.deltaAt
def deltas0_28 : Polynomial Owner := [LeftMerge103642.delta]
theorem rows0_28 : MergeDeltasAt history frameStart owner group deltas0_28 := by
  exact .leaf LeftMerge103642.deltaAt
def deltas0_29 : Polynomial Owner := [LeftMerge103645.delta]
theorem rows0_29 : MergeDeltasAt history frameStart owner group deltas0_29 := by
  exact .leaf LeftMerge103645.deltaAt
def deltas0_30 : Polynomial Owner := [LeftMerge103648.delta]
theorem rows0_30 : MergeDeltasAt history frameStart owner group deltas0_30 := by
  exact .leaf LeftMerge103648.deltaAt
def deltas0_31 : Polynomial Owner := [LeftMerge103651.delta]
theorem rows0_31 : MergeDeltasAt history frameStart owner group deltas0_31 := by
  exact .leaf LeftMerge103651.deltaAt
def deltas0_32 : Polynomial Owner := [LeftMerge103654.delta]
theorem rows0_32 : MergeDeltasAt history frameStart owner group deltas0_32 := by
  exact .leaf LeftMerge103654.deltaAt
def deltas0_33 : Polynomial Owner := [LeftMerge103657.delta]
theorem rows0_33 : MergeDeltasAt history frameStart owner group deltas0_33 := by
  exact .leaf LeftMerge103657.deltaAt
def deltas0_34 : Polynomial Owner := [LeftMerge103660.delta]
theorem rows0_34 : MergeDeltasAt history frameStart owner group deltas0_34 := by
  exact .leaf LeftMerge103660.deltaAt
def deltas0_35 : Polynomial Owner := [LeftMerge103663.delta]
theorem rows0_35 : MergeDeltasAt history frameStart owner group deltas0_35 := by
  exact .leaf LeftMerge103663.deltaAt
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
def left : Polynomial Owner := LeftMerge103594.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge103594.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6709⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6711⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6713⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6715⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6717⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6719⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6721⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6723⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6725⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6727⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6729⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6731⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6733⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6735⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6737⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6739⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6741⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6743⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15258⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15300⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15356⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15622⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15741⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15860⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15979⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16098⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16301⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16672⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16791⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨17078⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨17113⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨17302⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨17897⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨18163⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨18198⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨18303⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge103594

namespace LeftOperatorMerge103906
def frameStart : Nat := 103852
def owner : Owner := ⟨.program ⟨214⟩, ⟨17045⟩⟩
def group : MergeGroup := .operator 103902 103900
def deltas0_0 : Polynomial Owner := [LeftMerge103906.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge103906.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge103906.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge103906.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨17001⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge103906

namespace LeftOperatorMerge103929
def frameStart : Nat := 103852
def owner : Owner := ⟨.program ⟨214⟩, ⟨18116⟩⟩
def group : MergeGroup := .operator 103902 103925
def deltas0_0 : Polynomial Owner := [LeftMerge103929.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge103929.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge103929.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge103929.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨18114⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge103929

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
