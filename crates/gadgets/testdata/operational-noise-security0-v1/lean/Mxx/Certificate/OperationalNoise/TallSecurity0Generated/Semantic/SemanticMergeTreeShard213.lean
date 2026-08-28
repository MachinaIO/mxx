import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeDeltaShard445
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeDeltaShard447
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeDeltaShard448
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeDeltaShard452
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeDeltaShard453
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeDeltaShard454
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeDeltaShard455
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeDeltaShard456
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeDeltaShard462
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeDeltaShard463

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace LeftOperatorMerge73441
def frameStart : Nat := 73363
def owner : Owner := ⟨.program ⟨214⟩, ⟨26552⟩⟩
def group : MergeGroup := .operator 73437 73414
def deltas0_0 : Polynomial Owner := [LeftMerge73441.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge73441.deltaAt
def deltas0_1 : Polynomial Owner := [LeftMerge73442.delta]
theorem rows0_1 : MergeDeltasAt history frameStart owner group deltas0_1 := by
  exact .leaf LeftMerge73442.deltaAt
def deltas1_0 : Polynomial Owner := deltas0_0 ++ deltas0_1
theorem rows1_0 : MergeDeltasAt history frameStart owner group deltas1_0 := by
  exact .append rows0_0 rows0_1
abbrev deltas : Polynomial Owner := deltas1_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows1_0
def left : Polynomial Owner := LeftMerge73441.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge73441.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6691⟩⟩, ⟨.program ⟨214⟩, ⟨26551⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨14949⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26551⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge73441

namespace LeftOperatorMerge73719
def frameStart : Nat := 73636
def owner : Owner := ⟨.program ⟨214⟩, ⟨7833⟩⟩
def group : MergeGroup := .operator 73715 73712
def deltas0_0 : Polynomial Owner := [LeftMerge73719.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge73719.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge73719.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge73719.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6771⟩⟩, ⟨.program ⟨214⟩, ⟨7831⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge73719

namespace LeftOperatorMerge73666
def frameStart : Nat := 73636
def owner : Owner := ⟨.program ⟨214⟩, ⟨10473⟩⟩
def group : MergeGroup := .operator 73662 73659
def deltas0_0 : Polynomial Owner := [LeftMerge73666.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge73666.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge73666.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge73666.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨9395⟩⟩, ⟨.program ⟨214⟩, ⟨10472⟩⟩], orderedFactors := [] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge73666

namespace LeftOperatorMerge73696
def frameStart : Nat := 73636
def owner : Owner := ⟨.program ⟨214⟩, ⟨10574⟩⟩
def group : MergeGroup := .operator 73692 73690
def deltas0_0 : Polynomial Owner := [LeftMerge73696.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge73696.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge73696.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge73696.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨9395⟩⟩, ⟨.program ⟨214⟩, ⟨10472⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge73696

namespace LeftOperatorMerge73739
def frameStart : Nat := 73636
def owner : Owner := ⟨.program ⟨214⟩, ⟨14790⟩⟩
def group : MergeGroup := .operator 73692 73735
def deltas0_0 : Polynomial Owner := [LeftMerge73739.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge73739.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge73739.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge73739.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨14788⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge73739

namespace LeftOperatorMerge73728
def frameStart : Nat := 73636
def owner : Owner := ⟨.program ⟨214⟩, ⟨24909⟩⟩
def group : MergeGroup := .operator 73724 73681
def deltas0_0 : Polynomial Owner := [LeftMerge73728.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge73728.deltaAt
def deltas0_1 : Polynomial Owner := [LeftMerge73729.delta]
theorem rows0_1 : MergeDeltasAt history frameStart owner group deltas0_1 := by
  exact .leaf LeftMerge73729.deltaAt
def deltas1_0 : Polynomial Owner := deltas0_0 ++ deltas0_1
theorem rows1_0 : MergeDeltasAt history frameStart owner group deltas1_0 := by
  exact .append rows0_0 rows0_1
abbrev deltas : Polynomial Owner := deltas1_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows1_0
def left : Polynomial Owner := LeftMerge73728.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge73728.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6771⟩⟩, ⟨.program ⟨214⟩, ⟨7831⟩⟩, ⟨.program ⟨214⟩, ⟨24906⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨9395⟩⟩, ⟨.program ⟨214⟩, ⟨10472⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨24906⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge73728

namespace LeftOperatorMerge73911
def frameStart : Nat := 73845
def owner : Owner := ⟨.program ⟨214⟩, ⟨14830⟩⟩
def group : MergeGroup := .operator 73907 73905
def deltas0_0 : Polynomial Owner := [LeftMerge73911.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge73911.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge73911.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge73911.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨14788⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge73911

namespace LeftOperatorMerge73934
def frameStart : Nat := 73845
def owner : Owner := ⟨.program ⟨214⟩, ⟨15263⟩⟩
def group : MergeGroup := .operator 73907 73930
def deltas0_0 : Polynomial Owner := [LeftMerge73934.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge73934.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge73934.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge73934.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15262⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge73934

namespace LeftOperatorMerge73923
def frameStart : Nat := 73845
def owner : Owner := ⟨.program ⟨214⟩, ⟨26347⟩⟩
def group : MergeGroup := .operator 73919 73896
def deltas0_0 : Polynomial Owner := [LeftMerge73923.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge73923.deltaAt
def deltas0_1 : Polynomial Owner := [LeftMerge73924.delta]
theorem rows0_1 : MergeDeltasAt history frameStart owner group deltas0_1 := by
  exact .leaf LeftMerge73924.deltaAt
def deltas1_0 : Polynomial Owner := deltas0_0 ++ deltas0_1
theorem rows1_0 : MergeDeltasAt history frameStart owner group deltas1_0 := by
  exact .append rows0_0 rows0_1
abbrev deltas : Polynomial Owner := deltas1_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows1_0
def left : Polynomial Owner := LeftMerge73923.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge73923.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6690⟩⟩, ⟨.program ⟨214⟩, ⟨26346⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨14788⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26346⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge73923

namespace LeftOperatorMerge75486
def frameStart : Nat := 74728
def owner : Owner := ⟨.program ⟨214⟩, ⟨18493⟩⟩
def group : MergeGroup := .operator 75255 75482
def deltas0_0 : Polynomial Owner := [LeftMerge75486.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge75486.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge75486.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge75486.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨18491⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge75486

namespace LeftOperatorMerge75259
def frameStart : Nat := 74728
def owner : Owner := ⟨.program ⟨214⟩, ⟨18645⟩⟩
def group : MergeGroup := .operator 75255 75253
def deltas0_0 : Polynomial Owner := [LeftMerge75259.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge75259.deltaAt
def deltas0_1 : Polynomial Owner := [LeftMerge75260.delta]
theorem rows0_1 : MergeDeltasAt history frameStart owner group deltas0_1 := by
  exact .leaf LeftMerge75260.deltaAt
def deltas0_2 : Polynomial Owner := [LeftMerge75261.delta]
theorem rows0_2 : MergeDeltasAt history frameStart owner group deltas0_2 := by
  exact .leaf LeftMerge75261.deltaAt
def deltas0_3 : Polynomial Owner := [LeftMerge75262.delta]
theorem rows0_3 : MergeDeltasAt history frameStart owner group deltas0_3 := by
  exact .leaf LeftMerge75262.deltaAt
def deltas0_4 : Polynomial Owner := [LeftMerge75263.delta]
theorem rows0_4 : MergeDeltasAt history frameStart owner group deltas0_4 := by
  exact .leaf LeftMerge75263.deltaAt
def deltas0_5 : Polynomial Owner := [LeftMerge75264.delta]
theorem rows0_5 : MergeDeltasAt history frameStart owner group deltas0_5 := by
  exact .leaf LeftMerge75264.deltaAt
def deltas0_6 : Polynomial Owner := [LeftMerge75265.delta]
theorem rows0_6 : MergeDeltasAt history frameStart owner group deltas0_6 := by
  exact .leaf LeftMerge75265.deltaAt
def deltas0_7 : Polynomial Owner := [LeftMerge75266.delta]
theorem rows0_7 : MergeDeltasAt history frameStart owner group deltas0_7 := by
  exact .leaf LeftMerge75266.deltaAt
def deltas0_8 : Polynomial Owner := [LeftMerge75267.delta]
theorem rows0_8 : MergeDeltasAt history frameStart owner group deltas0_8 := by
  exact .leaf LeftMerge75267.deltaAt
def deltas0_9 : Polynomial Owner := [LeftMerge75268.delta]
theorem rows0_9 : MergeDeltasAt history frameStart owner group deltas0_9 := by
  exact .leaf LeftMerge75268.deltaAt
def deltas0_10 : Polynomial Owner := [LeftMerge75269.delta]
theorem rows0_10 : MergeDeltasAt history frameStart owner group deltas0_10 := by
  exact .leaf LeftMerge75269.deltaAt
def deltas0_11 : Polynomial Owner := [LeftMerge75270.delta]
theorem rows0_11 : MergeDeltasAt history frameStart owner group deltas0_11 := by
  exact .leaf LeftMerge75270.deltaAt
def deltas0_12 : Polynomial Owner := [LeftMerge75271.delta]
theorem rows0_12 : MergeDeltasAt history frameStart owner group deltas0_12 := by
  exact .leaf LeftMerge75271.deltaAt
def deltas0_13 : Polynomial Owner := [LeftMerge75272.delta]
theorem rows0_13 : MergeDeltasAt history frameStart owner group deltas0_13 := by
  exact .leaf LeftMerge75272.deltaAt
def deltas0_14 : Polynomial Owner := [LeftMerge75273.delta]
theorem rows0_14 : MergeDeltasAt history frameStart owner group deltas0_14 := by
  exact .leaf LeftMerge75273.deltaAt
def deltas0_15 : Polynomial Owner := [LeftMerge75274.delta]
theorem rows0_15 : MergeDeltasAt history frameStart owner group deltas0_15 := by
  exact .leaf LeftMerge75274.deltaAt
def deltas0_16 : Polynomial Owner := [LeftMerge75275.delta]
theorem rows0_16 : MergeDeltasAt history frameStart owner group deltas0_16 := by
  exact .leaf LeftMerge75275.deltaAt
def deltas0_17 : Polynomial Owner := [LeftMerge75276.delta]
theorem rows0_17 : MergeDeltasAt history frameStart owner group deltas0_17 := by
  exact .leaf LeftMerge75276.deltaAt
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
def left : Polynomial Owner := LeftMerge75259.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge75259.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15262⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15306⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15362⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15626⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15745⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15864⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15983⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16102⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16305⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16676⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16795⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨17082⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨17117⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨17318⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨17901⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨18167⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨18202⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨18327⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge75259

namespace LeftOperatorMerge75407
def frameStart : Nat := 74728
def owner : Owner := ⟨.program ⟨214⟩, ⟨18679⟩⟩
def group : MergeGroup := .operator 75403 75244
def deltas0_0 : Polynomial Owner := [LeftMerge75407.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge75407.deltaAt
def deltas0_1 : Polynomial Owner := [LeftMerge75408.delta]
theorem rows0_1 : MergeDeltasAt history frameStart owner group deltas0_1 := by
  exact .leaf LeftMerge75408.deltaAt
def deltas0_2 : Polynomial Owner := [LeftMerge75409.delta]
theorem rows0_2 : MergeDeltasAt history frameStart owner group deltas0_2 := by
  exact .leaf LeftMerge75409.deltaAt
def deltas0_3 : Polynomial Owner := [LeftMerge75410.delta]
theorem rows0_3 : MergeDeltasAt history frameStart owner group deltas0_3 := by
  exact .leaf LeftMerge75410.deltaAt
def deltas0_4 : Polynomial Owner := [LeftMerge75411.delta]
theorem rows0_4 : MergeDeltasAt history frameStart owner group deltas0_4 := by
  exact .leaf LeftMerge75411.deltaAt
def deltas0_5 : Polynomial Owner := [LeftMerge75412.delta]
theorem rows0_5 : MergeDeltasAt history frameStart owner group deltas0_5 := by
  exact .leaf LeftMerge75412.deltaAt
def deltas0_6 : Polynomial Owner := [LeftMerge75413.delta]
theorem rows0_6 : MergeDeltasAt history frameStart owner group deltas0_6 := by
  exact .leaf LeftMerge75413.deltaAt
def deltas0_7 : Polynomial Owner := [LeftMerge75414.delta]
theorem rows0_7 : MergeDeltasAt history frameStart owner group deltas0_7 := by
  exact .leaf LeftMerge75414.deltaAt
def deltas0_8 : Polynomial Owner := [LeftMerge75415.delta]
theorem rows0_8 : MergeDeltasAt history frameStart owner group deltas0_8 := by
  exact .leaf LeftMerge75415.deltaAt
def deltas0_9 : Polynomial Owner := [LeftMerge75416.delta]
theorem rows0_9 : MergeDeltasAt history frameStart owner group deltas0_9 := by
  exact .leaf LeftMerge75416.deltaAt
def deltas0_10 : Polynomial Owner := [LeftMerge75417.delta]
theorem rows0_10 : MergeDeltasAt history frameStart owner group deltas0_10 := by
  exact .leaf LeftMerge75417.deltaAt
def deltas0_11 : Polynomial Owner := [LeftMerge75418.delta]
theorem rows0_11 : MergeDeltasAt history frameStart owner group deltas0_11 := by
  exact .leaf LeftMerge75418.deltaAt
def deltas0_12 : Polynomial Owner := [LeftMerge75419.delta]
theorem rows0_12 : MergeDeltasAt history frameStart owner group deltas0_12 := by
  exact .leaf LeftMerge75419.deltaAt
def deltas0_13 : Polynomial Owner := [LeftMerge75420.delta]
theorem rows0_13 : MergeDeltasAt history frameStart owner group deltas0_13 := by
  exact .leaf LeftMerge75420.deltaAt
def deltas0_14 : Polynomial Owner := [LeftMerge75421.delta]
theorem rows0_14 : MergeDeltasAt history frameStart owner group deltas0_14 := by
  exact .leaf LeftMerge75421.deltaAt
def deltas0_15 : Polynomial Owner := [LeftMerge75422.delta]
theorem rows0_15 : MergeDeltasAt history frameStart owner group deltas0_15 := by
  exact .leaf LeftMerge75422.deltaAt
def deltas0_16 : Polynomial Owner := [LeftMerge75423.delta]
theorem rows0_16 : MergeDeltasAt history frameStart owner group deltas0_16 := by
  exact .leaf LeftMerge75423.deltaAt
def deltas0_17 : Polynomial Owner := [LeftMerge75424.delta]
theorem rows0_17 : MergeDeltasAt history frameStart owner group deltas0_17 := by
  exact .leaf LeftMerge75424.deltaAt
def deltas0_18 : Polynomial Owner := [LeftMerge75425.delta]
theorem rows0_18 : MergeDeltasAt history frameStart owner group deltas0_18 := by
  exact .leaf LeftMerge75425.deltaAt
def deltas0_19 : Polynomial Owner := [LeftMerge75428.delta]
theorem rows0_19 : MergeDeltasAt history frameStart owner group deltas0_19 := by
  exact .leaf LeftMerge75428.deltaAt
def deltas0_20 : Polynomial Owner := [LeftMerge75431.delta]
theorem rows0_20 : MergeDeltasAt history frameStart owner group deltas0_20 := by
  exact .leaf LeftMerge75431.deltaAt
def deltas0_21 : Polynomial Owner := [LeftMerge75434.delta]
theorem rows0_21 : MergeDeltasAt history frameStart owner group deltas0_21 := by
  exact .leaf LeftMerge75434.deltaAt
def deltas0_22 : Polynomial Owner := [LeftMerge75437.delta]
theorem rows0_22 : MergeDeltasAt history frameStart owner group deltas0_22 := by
  exact .leaf LeftMerge75437.deltaAt
def deltas0_23 : Polynomial Owner := [LeftMerge75440.delta]
theorem rows0_23 : MergeDeltasAt history frameStart owner group deltas0_23 := by
  exact .leaf LeftMerge75440.deltaAt
def deltas0_24 : Polynomial Owner := [LeftMerge75443.delta]
theorem rows0_24 : MergeDeltasAt history frameStart owner group deltas0_24 := by
  exact .leaf LeftMerge75443.deltaAt
def deltas0_25 : Polynomial Owner := [LeftMerge75446.delta]
theorem rows0_25 : MergeDeltasAt history frameStart owner group deltas0_25 := by
  exact .leaf LeftMerge75446.deltaAt
def deltas0_26 : Polynomial Owner := [LeftMerge75449.delta]
theorem rows0_26 : MergeDeltasAt history frameStart owner group deltas0_26 := by
  exact .leaf LeftMerge75449.deltaAt
def deltas0_27 : Polynomial Owner := [LeftMerge75452.delta]
theorem rows0_27 : MergeDeltasAt history frameStart owner group deltas0_27 := by
  exact .leaf LeftMerge75452.deltaAt
def deltas0_28 : Polynomial Owner := [LeftMerge75455.delta]
theorem rows0_28 : MergeDeltasAt history frameStart owner group deltas0_28 := by
  exact .leaf LeftMerge75455.deltaAt
def deltas0_29 : Polynomial Owner := [LeftMerge75458.delta]
theorem rows0_29 : MergeDeltasAt history frameStart owner group deltas0_29 := by
  exact .leaf LeftMerge75458.deltaAt
def deltas0_30 : Polynomial Owner := [LeftMerge75461.delta]
theorem rows0_30 : MergeDeltasAt history frameStart owner group deltas0_30 := by
  exact .leaf LeftMerge75461.deltaAt
def deltas0_31 : Polynomial Owner := [LeftMerge75464.delta]
theorem rows0_31 : MergeDeltasAt history frameStart owner group deltas0_31 := by
  exact .leaf LeftMerge75464.deltaAt
def deltas0_32 : Polynomial Owner := [LeftMerge75467.delta]
theorem rows0_32 : MergeDeltasAt history frameStart owner group deltas0_32 := by
  exact .leaf LeftMerge75467.deltaAt
def deltas0_33 : Polynomial Owner := [LeftMerge75470.delta]
theorem rows0_33 : MergeDeltasAt history frameStart owner group deltas0_33 := by
  exact .leaf LeftMerge75470.deltaAt
def deltas0_34 : Polynomial Owner := [LeftMerge75473.delta]
theorem rows0_34 : MergeDeltasAt history frameStart owner group deltas0_34 := by
  exact .leaf LeftMerge75473.deltaAt
def deltas0_35 : Polynomial Owner := [LeftMerge75476.delta]
theorem rows0_35 : MergeDeltasAt history frameStart owner group deltas0_35 := by
  exact .leaf LeftMerge75476.deltaAt
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
def left : Polynomial Owner := LeftMerge75407.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge75407.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6709⟩⟩, ⟨.program ⟨214⟩, ⟨18678⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6711⟩⟩, ⟨.program ⟨214⟩, ⟨18678⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6713⟩⟩, ⟨.program ⟨214⟩, ⟨18678⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6715⟩⟩, ⟨.program ⟨214⟩, ⟨18678⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6717⟩⟩, ⟨.program ⟨214⟩, ⟨18678⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6719⟩⟩, ⟨.program ⟨214⟩, ⟨18678⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6721⟩⟩, ⟨.program ⟨214⟩, ⟨18678⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6723⟩⟩, ⟨.program ⟨214⟩, ⟨18678⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6725⟩⟩, ⟨.program ⟨214⟩, ⟨18678⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6727⟩⟩, ⟨.program ⟨214⟩, ⟨18678⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6729⟩⟩, ⟨.program ⟨214⟩, ⟨18678⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6731⟩⟩, ⟨.program ⟨214⟩, ⟨18678⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6733⟩⟩, ⟨.program ⟨214⟩, ⟨18678⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6735⟩⟩, ⟨.program ⟨214⟩, ⟨18678⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6737⟩⟩, ⟨.program ⟨214⟩, ⟨18678⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6739⟩⟩, ⟨.program ⟨214⟩, ⟨18678⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6741⟩⟩, ⟨.program ⟨214⟩, ⟨18678⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6743⟩⟩, ⟨.program ⟨214⟩, ⟨18678⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15262⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18678⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15306⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18678⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15362⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18678⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15626⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18678⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15745⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18678⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15864⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18678⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15983⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18678⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16102⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18678⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16305⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18678⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16676⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18678⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16795⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18678⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨17082⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18678⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨17117⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18678⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨17318⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18678⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨17901⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18678⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨18167⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18678⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨18202⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18678⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨18327⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18678⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge75407

namespace LeftOperatorMerge75743
def frameStart : Nat := 75677
def owner : Owner := ⟨.program ⟨214⟩, ⟨17049⟩⟩
def group : MergeGroup := .operator 75739 75737
def deltas0_0 : Polynomial Owner := [LeftMerge75743.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge75743.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge75743.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge75743.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨17007⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge75743

namespace LeftOperatorMerge75766
def frameStart : Nat := 75677
def owner : Owner := ⟨.program ⟨214⟩, ⟨18122⟩⟩
def group : MergeGroup := .operator 75739 75762
def deltas0_0 : Polynomial Owner := [LeftMerge75766.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge75766.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge75766.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge75766.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨18120⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge75766

namespace LeftOperatorMerge75755
def frameStart : Nat := 75677
def owner : Owner := ⟨.program ⟨214⟩, ⟨30089⟩⟩
def group : MergeGroup := .operator 75751 75728
def deltas0_0 : Polynomial Owner := [LeftMerge75755.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge75755.deltaAt
def deltas0_1 : Polynomial Owner := [LeftMerge75756.delta]
theorem rows0_1 : MergeDeltasAt history frameStart owner group deltas0_1 := by
  exact .leaf LeftMerge75756.deltaAt
def deltas1_0 : Polynomial Owner := deltas0_0 ++ deltas0_1
theorem rows1_0 : MergeDeltasAt history frameStart owner group deltas1_0 := by
  exact .append rows0_0 rows0_1
abbrev deltas : Polynomial Owner := deltas1_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows1_0
def left : Polynomial Owner := LeftMerge75755.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge75755.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6707⟩⟩, ⟨.program ⟨214⟩, ⟨30088⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨17007⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨30088⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge75755

namespace LeftOperatorMerge75978
def frameStart : Nat := 75889
def owner : Owner := ⟨.program ⟨214⟩, ⟨16925⟩⟩
def group : MergeGroup := .operator 75951 75974
def deltas0_0 : Polynomial Owner := [LeftMerge75978.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge75978.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge75978.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge75978.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16923⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge75978

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
