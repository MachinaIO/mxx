import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeDeltaShard027
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeDeltaShard028
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeDeltaShard030
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeDeltaShard031
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeDeltaShard446
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeDeltaShard538
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeDeltaShard627

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace LeftOperatorMerge6093
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨214⟩, ⟨7914⟩⟩
def group : MergeGroup := .operator 6089 6081
def deltas0_0 : Polynomial Owner := [LeftMerge6093.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge6093.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge6093.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge6093.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6751⟩⟩, ⟨.program ⟨214⟩, ⟨7885⟩⟩, ⟨.program ⟨214⟩, ⟨7825⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge6093

namespace LeftOperatorMerge6133
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨214⟩, ⟨7915⟩⟩
def group : MergeGroup := .operator 6129 6121
def deltas0_0 : Polynomial Owner := [LeftMerge6133.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge6133.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge6133.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge6133.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6753⟩⟩, ⟨.program ⟨214⟩, ⟨7885⟩⟩, ⟨.program ⟨214⟩, ⟨7827⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge6133

namespace LeftOperatorMerge6173
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨214⟩, ⟨7916⟩⟩
def group : MergeGroup := .operator 6169 6161
def deltas0_0 : Polynomial Owner := [LeftMerge6173.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge6173.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge6173.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge6173.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6755⟩⟩, ⟨.program ⟨214⟩, ⟨7885⟩⟩, ⟨.program ⟨214⟩, ⟨7829⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge6173

namespace LeftOperatorMerge5978
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨214⟩, ⟨7917⟩⟩
def group : MergeGroup := .operator 5974 5476
def deltas0_0 : Polynomial Owner := [LeftMerge5978.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge5978.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge5978.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge5978.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6745⟩⟩, ⟨.program ⟨214⟩, ⟨7885⟩⟩, ⟨.program ⟨214⟩, ⟨7819⟩⟩, ⟨.program ⟨214⟩, ⟨6645⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge5978

namespace LeftOperatorMerge6018
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨214⟩, ⟨7918⟩⟩
def group : MergeGroup := .operator 6014 5991
def deltas0_0 : Polynomial Owner := [LeftMerge6018.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge6018.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge6018.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge6018.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6747⟩⟩, ⟨.program ⟨214⟩, ⟨7885⟩⟩, ⟨.program ⟨214⟩, ⟨7821⟩⟩, ⟨.program ⟨214⟩, ⟨6687⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge6018

namespace LeftOperatorMerge6058
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨214⟩, ⟨7919⟩⟩
def group : MergeGroup := .operator 6054 6031
def deltas0_0 : Polynomial Owner := [LeftMerge6058.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge6058.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge6058.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge6058.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6749⟩⟩, ⟨.program ⟨214⟩, ⟨7885⟩⟩, ⟨.program ⟨214⟩, ⟨7823⟩⟩, ⟨.program ⟨214⟩, ⟨6653⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge6058

namespace LeftOperatorMerge6098
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨214⟩, ⟨7920⟩⟩
def group : MergeGroup := .operator 6094 6071
def deltas0_0 : Polynomial Owner := [LeftMerge6098.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge6098.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge6098.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge6098.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6751⟩⟩, ⟨.program ⟨214⟩, ⟨7885⟩⟩, ⟨.program ⟨214⟩, ⟨7825⟩⟩, ⟨.program ⟨214⟩, ⟨6675⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge6098

namespace LeftOperatorMerge6138
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨214⟩, ⟨7921⟩⟩
def group : MergeGroup := .operator 6134 6111
def deltas0_0 : Polynomial Owner := [LeftMerge6138.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge6138.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge6138.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge6138.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6753⟩⟩, ⟨.program ⟨214⟩, ⟨7885⟩⟩, ⟨.program ⟨214⟩, ⟨7827⟩⟩, ⟨.program ⟨214⟩, ⟨6685⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge6138

namespace LeftOperatorMerge6178
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨214⟩, ⟨7922⟩⟩
def group : MergeGroup := .operator 6174 6151
def deltas0_0 : Polynomial Owner := [LeftMerge6178.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge6178.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge6178.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge6178.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6755⟩⟩, ⟨.program ⟨214⟩, ⟨7885⟩⟩, ⟨.program ⟨214⟩, ⟨7829⟩⟩, ⟨.program ⟨214⟩, ⟨6683⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge6178

namespace LeftOperatorMerge6246
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨214⟩, ⟨7929⟩⟩
def group : MergeGroup := .operator 27 6242
def deltas0_0 : Polynomial Owner := [LeftMerge6246.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge6246.deltaAt
def deltas0_1 : Polynomial Owner := [LeftMerge6247.delta]
theorem rows0_1 : MergeDeltasAt history frameStart owner group deltas0_1 := by
  exact .leaf LeftMerge6247.deltaAt
def deltas0_2 : Polynomial Owner := [LeftMerge6248.delta]
theorem rows0_2 : MergeDeltasAt history frameStart owner group deltas0_2 := by
  exact .leaf LeftMerge6248.deltaAt
def deltas0_3 : Polynomial Owner := [LeftMerge6249.delta]
theorem rows0_3 : MergeDeltasAt history frameStart owner group deltas0_3 := by
  exact .leaf LeftMerge6249.deltaAt
def deltas0_4 : Polynomial Owner := [LeftMerge6250.delta]
theorem rows0_4 : MergeDeltasAt history frameStart owner group deltas0_4 := by
  exact .leaf LeftMerge6250.deltaAt
def deltas0_5 : Polynomial Owner := [LeftMerge6251.delta]
theorem rows0_5 : MergeDeltasAt history frameStart owner group deltas0_5 := by
  exact .leaf LeftMerge6251.deltaAt
def deltas0_6 : Polynomial Owner := [LeftMerge6252.delta]
theorem rows0_6 : MergeDeltasAt history frameStart owner group deltas0_6 := by
  exact .leaf LeftMerge6252.deltaAt
def deltas0_7 : Polynomial Owner := [LeftMerge6253.delta]
theorem rows0_7 : MergeDeltasAt history frameStart owner group deltas0_7 := by
  exact .leaf LeftMerge6253.deltaAt
def deltas0_8 : Polynomial Owner := [LeftMerge6254.delta]
theorem rows0_8 : MergeDeltasAt history frameStart owner group deltas0_8 := by
  exact .leaf LeftMerge6254.deltaAt
def deltas0_9 : Polynomial Owner := [LeftMerge6255.delta]
theorem rows0_9 : MergeDeltasAt history frameStart owner group deltas0_9 := by
  exact .leaf LeftMerge6255.deltaAt
def deltas0_10 : Polynomial Owner := [LeftMerge6256.delta]
theorem rows0_10 : MergeDeltasAt history frameStart owner group deltas0_10 := by
  exact .leaf LeftMerge6256.deltaAt
def deltas0_11 : Polynomial Owner := [LeftMerge6257.delta]
theorem rows0_11 : MergeDeltasAt history frameStart owner group deltas0_11 := by
  exact .leaf LeftMerge6257.deltaAt
def deltas0_12 : Polynomial Owner := [LeftMerge6258.delta]
theorem rows0_12 : MergeDeltasAt history frameStart owner group deltas0_12 := by
  exact .leaf LeftMerge6258.deltaAt
def deltas0_13 : Polynomial Owner := [LeftMerge6259.delta]
theorem rows0_13 : MergeDeltasAt history frameStart owner group deltas0_13 := by
  exact .leaf LeftMerge6259.deltaAt
def deltas0_14 : Polynomial Owner := [LeftMerge6260.delta]
theorem rows0_14 : MergeDeltasAt history frameStart owner group deltas0_14 := by
  exact .leaf LeftMerge6260.deltaAt
def deltas0_15 : Polynomial Owner := [LeftMerge6261.delta]
theorem rows0_15 : MergeDeltasAt history frameStart owner group deltas0_15 := by
  exact .leaf LeftMerge6261.deltaAt
def deltas0_16 : Polynomial Owner := [LeftMerge6262.delta]
theorem rows0_16 : MergeDeltasAt history frameStart owner group deltas0_16 := by
  exact .leaf LeftMerge6262.deltaAt
def deltas0_17 : Polynomial Owner := [LeftMerge6263.delta]
theorem rows0_17 : MergeDeltasAt history frameStart owner group deltas0_17 := by
  exact .leaf LeftMerge6263.deltaAt
def deltas0_18 : Polynomial Owner := [LeftMerge6264.delta]
theorem rows0_18 : MergeDeltasAt history frameStart owner group deltas0_18 := by
  exact .leaf LeftMerge6264.deltaAt
def deltas0_19 : Polynomial Owner := [LeftMerge6265.delta]
theorem rows0_19 : MergeDeltasAt history frameStart owner group deltas0_19 := by
  exact .leaf LeftMerge6265.deltaAt
def deltas0_20 : Polynomial Owner := [LeftMerge6266.delta]
theorem rows0_20 : MergeDeltasAt history frameStart owner group deltas0_20 := by
  exact .leaf LeftMerge6266.deltaAt
def deltas0_21 : Polynomial Owner := [LeftMerge6267.delta]
theorem rows0_21 : MergeDeltasAt history frameStart owner group deltas0_21 := by
  exact .leaf LeftMerge6267.deltaAt
def deltas0_22 : Polynomial Owner := [LeftMerge6268.delta]
theorem rows0_22 : MergeDeltasAt history frameStart owner group deltas0_22 := by
  exact .leaf LeftMerge6268.deltaAt
def deltas0_23 : Polynomial Owner := [LeftMerge6269.delta]
theorem rows0_23 : MergeDeltasAt history frameStart owner group deltas0_23 := by
  exact .leaf LeftMerge6269.deltaAt
def deltas0_24 : Polynomial Owner := [LeftMerge6270.delta]
theorem rows0_24 : MergeDeltasAt history frameStart owner group deltas0_24 := by
  exact .leaf LeftMerge6270.deltaAt
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
def deltas3_0 : Polynomial Owner := deltas2_0 ++ deltas2_1
theorem rows3_0 : MergeDeltasAt history frameStart owner group deltas3_0 := by
  exact .append rows2_0 rows2_1
def deltas3_1 : Polynomial Owner := deltas2_2 ++ deltas2_3
theorem rows3_1 : MergeDeltasAt history frameStart owner group deltas3_1 := by
  exact .append rows2_2 rows2_3
def deltas3_2 : Polynomial Owner := deltas2_4 ++ deltas2_5
theorem rows3_2 : MergeDeltasAt history frameStart owner group deltas3_2 := by
  exact .append rows2_4 rows2_5
def deltas4_0 : Polynomial Owner := deltas3_0 ++ deltas3_1
theorem rows4_0 : MergeDeltasAt history frameStart owner group deltas4_0 := by
  exact .append rows3_0 rows3_1
def deltas4_1 : Polynomial Owner := deltas3_2 ++ deltas0_24
theorem rows4_1 : MergeDeltasAt history frameStart owner group deltas4_1 := by
  exact .append rows3_2 rows0_24
def deltas5_0 : Polynomial Owner := deltas4_0 ++ deltas4_1
theorem rows5_0 : MergeDeltasAt history frameStart owner group deltas5_0 := by
  exact .append rows4_0 rows4_1
abbrev deltas : Polynomial Owner := deltas5_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows5_0
def left : Polynomial Owner := LeftMerge6246.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge6246.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6708⟩⟩, ⟨.program ⟨214⟩, ⟨6679⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6710⟩⟩, ⟨.program ⟨214⟩, ⟨6671⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6712⟩⟩, ⟨.program ⟨214⟩, ⟨6663⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6714⟩⟩, ⟨.program ⟨214⟩, ⟨6655⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6716⟩⟩, ⟨.program ⟨214⟩, ⟨6649⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6718⟩⟩, ⟨.program ⟨214⟩, ⟨6647⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6720⟩⟩, ⟨.program ⟨214⟩, ⟨6643⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6722⟩⟩, ⟨.program ⟨214⟩, ⟨6641⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6724⟩⟩, ⟨.program ⟨214⟩, ⟨6637⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6726⟩⟩, ⟨.program ⟨214⟩, ⟨6681⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6728⟩⟩, ⟨.program ⟨214⟩, ⟨6677⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6730⟩⟩, ⟨.program ⟨214⟩, ⟨6673⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6732⟩⟩, ⟨.program ⟨214⟩, ⟨6669⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6734⟩⟩, ⟨.program ⟨214⟩, ⟨6667⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6736⟩⟩, ⟨.program ⟨214⟩, ⟨6665⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6738⟩⟩, ⟨.program ⟨214⟩, ⟨6661⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6740⟩⟩, ⟨.program ⟨214⟩, ⟨6659⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6742⟩⟩, ⟨.program ⟨214⟩, ⟨6657⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6744⟩⟩, ⟨.program ⟨214⟩, ⟨6651⟩⟩, ⟨.program ⟨214⟩, ⟨6639⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6745⟩⟩, ⟨.program ⟨214⟩, ⟨7885⟩⟩, ⟨.program ⟨214⟩, ⟨7819⟩⟩, ⟨.program ⟨214⟩, ⟨6645⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6747⟩⟩, ⟨.program ⟨214⟩, ⟨7885⟩⟩, ⟨.program ⟨214⟩, ⟨7821⟩⟩, ⟨.program ⟨214⟩, ⟨6687⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6749⟩⟩, ⟨.program ⟨214⟩, ⟨7885⟩⟩, ⟨.program ⟨214⟩, ⟨7823⟩⟩, ⟨.program ⟨214⟩, ⟨6653⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6751⟩⟩, ⟨.program ⟨214⟩, ⟨7885⟩⟩, ⟨.program ⟨214⟩, ⟨7825⟩⟩, ⟨.program ⟨214⟩, ⟨6675⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6753⟩⟩, ⟨.program ⟨214⟩, ⟨7885⟩⟩, ⟨.program ⟨214⟩, ⟨7827⟩⟩, ⟨.program ⟨214⟩, ⟨6685⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6755⟩⟩, ⟨.program ⟨214⟩, ⟨7885⟩⟩, ⟨.program ⟨214⟩, ⟨7829⟩⟩, ⟨.program ⟨214⟩, ⟨6683⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right true false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge6246

namespace LeftOperatorMerge101791
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨214⟩, ⟨9386⟩⟩
def group : MergeGroup := .operator 4960 32
def deltas0_0 : Polynomial Owner := [LeftMerge101791.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge101791.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge101791.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge101791.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨9385⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right true false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge101791

namespace LeftOperatorMerge101813
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨214⟩, ⟨9389⟩⟩
def group : MergeGroup := .operator 101807 15019
def deltas0_0 : Polynomial Owner := [LeftMerge101813.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge101813.deltaAt
def deltas0_1 : Polynomial Owner := [LeftMerge101816.delta]
theorem rows0_1 : MergeDeltasAt history frameStart owner group deltas0_1 := by
  exact .leaf LeftMerge101816.deltaAt
def deltas1_0 : Polynomial Owner := deltas0_0 ++ deltas0_1
theorem rows1_0 : MergeDeltasAt history frameStart owner group deltas1_0 := by
  exact .append rows0_0 rows0_1
abbrev deltas : Polynomial Owner := deltas1_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows1_0
def left : Polynomial Owner := LeftMerge101813.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge101813.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6771⟩⟩, ⟨.program ⟨214⟩, ⟨7831⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨9385⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7831⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge101813

namespace LeftOperatorMerge73532
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨214⟩, ⟨9396⟩⟩
def group : MergeGroup := .operator 3480 65295
def deltas0_0 : Polynomial Owner := [LeftMerge73532.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge73532.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge73532.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge73532.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9395⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right true false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge73532

namespace LeftOperatorMerge73554
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨214⟩, ⟨9399⟩⟩
def group : MergeGroup := .operator 73548 15019
def deltas0_0 : Polynomial Owner := [LeftMerge73554.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge73554.deltaAt
def deltas0_1 : Polynomial Owner := [LeftMerge73557.delta]
theorem rows0_1 : MergeDeltasAt history frameStart owner group deltas0_1 := by
  exact .leaf LeftMerge73557.deltaAt
def deltas1_0 : Polynomial Owner := deltas0_0 ++ deltas0_1
theorem rows1_0 : MergeDeltasAt history frameStart owner group deltas1_0 := by
  exact .append rows0_0 rows0_1
abbrev deltas : Polynomial Owner := deltas1_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows1_0
def left : Polynomial Owner := LeftMerge73554.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge73554.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6771⟩⟩, ⟨.program ⟨214⟩, ⟨7831⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9395⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7831⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge73554

namespace LeftOperatorMerge88123
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨214⟩, ⟨9401⟩⟩
def group : MergeGroup := .operator 4222 79920
def deltas0_0 : Polynomial Owner := [LeftMerge88123.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge88123.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge88123.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge88123.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨9400⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right true false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge88123

namespace LeftOperatorMerge88145
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨214⟩, ⟨9404⟩⟩
def group : MergeGroup := .operator 88139 15019
def deltas0_0 : Polynomial Owner := [LeftMerge88145.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge88145.deltaAt
def deltas0_1 : Polynomial Owner := [LeftMerge88148.delta]
theorem rows0_1 : MergeDeltasAt history frameStart owner group deltas0_1 := by
  exact .leaf LeftMerge88148.deltaAt
def deltas1_0 : Polynomial Owner := deltas0_0 ++ deltas0_1
theorem rows1_0 : MergeDeltasAt history frameStart owner group deltas1_0 := by
  exact .append rows0_0 rows0_1
abbrev deltas : Polynomial Owner := deltas1_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows1_0
def left : Polynomial Owner := LeftMerge88145.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge88145.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6771⟩⟩, ⟨.program ⟨214⟩, ⟨7831⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨9400⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7831⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge88145

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
