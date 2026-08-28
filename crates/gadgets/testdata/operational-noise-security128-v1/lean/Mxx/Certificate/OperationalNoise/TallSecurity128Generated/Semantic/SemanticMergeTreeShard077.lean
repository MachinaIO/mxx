import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard462
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard554
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard646
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard1013
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard1105
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard1197
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard1289
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard1380
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard1381
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard1472
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard1473

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace LeftOperatorMerge238771
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨13855⟩⟩
def group : MergeGroup := .operator 238765 19114
def deltas0_0 : Polynomial Owner := [LeftMerge238771.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge238771.deltaAt
def deltas0_1 : Polynomial Owner := [LeftMerge238774.delta]
theorem rows0_1 : MergeDeltasAt history frameStart owner group deltas0_1 := by
  exact .leaf LeftMerge238774.deltaAt
def deltas1_0 : Polynomial Owner := deltas0_0 ++ deltas0_1
theorem rows1_0 : MergeDeltasAt history frameStart owner group deltas1_0 := by
  exact .append rows0_0 rows0_1
abbrev deltas : Polynomial Owner := deltas1_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows1_0
def left : Polynomial Owner := LeftMerge238771.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge238771.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨13851⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge238771

namespace LeftOperatorMerge224124
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨13867⟩⟩
def group : MergeGroup := .operator 10661 222153
def deltas0_0 : Polynomial Owner := [LeftMerge224124.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge224124.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge224124.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge224124.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨13866⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right true false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge224124

namespace LeftOperatorMerge224146
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨13870⟩⟩
def group : MergeGroup := .operator 224140 19114
def deltas0_0 : Polynomial Owner := [LeftMerge224146.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge224146.deltaAt
def deltas0_1 : Polynomial Owner := [LeftMerge224149.delta]
theorem rows0_1 : MergeDeltasAt history frameStart owner group deltas0_1 := by
  exact .leaf LeftMerge224149.deltaAt
def deltas1_0 : Polynomial Owner := deltas0_0 ++ deltas0_1
theorem rows1_0 : MergeDeltasAt history frameStart owner group deltas1_0 := by
  exact .append rows0_0 rows0_1
abbrev deltas : Polynomial Owner := deltas1_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows1_0
def left : Polynomial Owner := LeftMerge224146.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge224146.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨13866⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge224146

namespace LeftOperatorMerge209499
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨13882⟩⟩
def group : MergeGroup := .operator 9913 207528
def deltas0_0 : Polynomial Owner := [LeftMerge209499.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge209499.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge209499.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge209499.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨13881⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right true false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge209499

namespace LeftOperatorMerge209521
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨13885⟩⟩
def group : MergeGroup := .operator 209515 19114
def deltas0_0 : Polynomial Owner := [LeftMerge209521.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge209521.deltaAt
def deltas0_1 : Polynomial Owner := [LeftMerge209524.delta]
theorem rows0_1 : MergeDeltasAt history frameStart owner group deltas0_1 := by
  exact .leaf LeftMerge209524.deltaAt
def deltas1_0 : Polynomial Owner := deltas0_0 ++ deltas0_1
theorem rows1_0 : MergeDeltasAt history frameStart owner group deltas1_0 := by
  exact .append rows0_0 rows0_1
abbrev deltas : Polynomial Owner := deltas1_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows1_0
def left : Polynomial Owner := LeftMerge209521.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge209521.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨13881⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge209521

namespace LeftOperatorMerge107124
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨13897⟩⟩
def group : MergeGroup := .operator 4677 105153
def deltas0_0 : Polynomial Owner := [LeftMerge107124.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge107124.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge107124.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge107124.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨13896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right true false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge107124

namespace LeftOperatorMerge107146
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨13900⟩⟩
def group : MergeGroup := .operator 107140 19114
def deltas0_0 : Polynomial Owner := [LeftMerge107146.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge107146.deltaAt
def deltas0_1 : Polynomial Owner := [LeftMerge107149.delta]
theorem rows0_1 : MergeDeltasAt history frameStart owner group deltas0_1 := by
  exact .leaf LeftMerge107149.deltaAt
def deltas1_0 : Polynomial Owner := deltas0_0 ++ deltas0_1
theorem rows1_0 : MergeDeltasAt history frameStart owner group deltas1_0 := by
  exact .append rows0_0 rows0_1
abbrev deltas : Polynomial Owner := deltas1_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows1_0
def left : Polynomial Owner := LeftMerge107146.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge107146.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨13896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge107146

namespace LeftOperatorMerge194874
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨13912⟩⟩
def group : MergeGroup := .operator 9165 192903
def deltas0_0 : Polynomial Owner := [LeftMerge194874.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge194874.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge194874.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge194874.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨13911⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right true false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge194874

namespace LeftOperatorMerge194896
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨13915⟩⟩
def group : MergeGroup := .operator 194890 19114
def deltas0_0 : Polynomial Owner := [LeftMerge194896.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge194896.deltaAt
def deltas0_1 : Polynomial Owner := [LeftMerge194899.delta]
theorem rows0_1 : MergeDeltasAt history frameStart owner group deltas0_1 := by
  exact .leaf LeftMerge194899.deltaAt
def deltas1_0 : Polynomial Owner := deltas0_0 ++ deltas0_1
theorem rows1_0 : MergeDeltasAt history frameStart owner group deltas1_0 := by
  exact .append rows0_0 rows0_1
abbrev deltas : Polynomial Owner := deltas1_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows1_0
def left : Polynomial Owner := LeftMerge194896.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge194896.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨13911⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge194896

namespace LeftOperatorMerge180249
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨13927⟩⟩
def group : MergeGroup := .operator 8417 178278
def deltas0_0 : Polynomial Owner := [LeftMerge180249.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge180249.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge180249.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge180249.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨13926⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right true false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge180249

namespace LeftOperatorMerge180271
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨13930⟩⟩
def group : MergeGroup := .operator 180265 19114
def deltas0_0 : Polynomial Owner := [LeftMerge180271.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge180271.deltaAt
def deltas0_1 : Polynomial Owner := [LeftMerge180274.delta]
theorem rows0_1 : MergeDeltasAt history frameStart owner group deltas0_1 := by
  exact .leaf LeftMerge180274.deltaAt
def deltas1_0 : Polynomial Owner := deltas0_0 ++ deltas0_1
theorem rows1_0 : MergeDeltasAt history frameStart owner group deltas1_0 := by
  exact .append rows0_0 rows0_1
abbrev deltas : Polynomial Owner := deltas1_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows1_0
def left : Polynomial Owner := LeftMerge180271.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge180271.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨13926⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge180271

namespace LeftOperatorMerge165624
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨13942⟩⟩
def group : MergeGroup := .operator 7669 163653
def deltas0_0 : Polynomial Owner := [LeftMerge165624.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge165624.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge165624.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge165624.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨13941⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right true false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge165624

namespace LeftOperatorMerge165646
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨13945⟩⟩
def group : MergeGroup := .operator 165640 19114
def deltas0_0 : Polynomial Owner := [LeftMerge165646.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge165646.deltaAt
def deltas0_1 : Polynomial Owner := [LeftMerge165649.delta]
theorem rows0_1 : MergeDeltasAt history frameStart owner group deltas0_1 := by
  exact .leaf LeftMerge165649.deltaAt
def deltas1_0 : Polynomial Owner := deltas0_0 ++ deltas0_1
theorem rows1_0 : MergeDeltasAt history frameStart owner group deltas1_0 := by
  exact .append rows0_0 rows0_1
abbrev deltas : Polynomial Owner := deltas1_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows1_0
def left : Polynomial Owner := LeftMerge165646.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge165646.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨13941⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge165646

namespace LeftOperatorMerge92499
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨13957⟩⟩
def group : MergeGroup := .operator 3929 90528
def deltas0_0 : Polynomial Owner := [LeftMerge92499.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge92499.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge92499.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge92499.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨13956⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right true false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge92499

namespace LeftOperatorMerge92521
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨13960⟩⟩
def group : MergeGroup := .operator 92515 19114
def deltas0_0 : Polynomial Owner := [LeftMerge92521.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge92521.deltaAt
def deltas0_1 : Polynomial Owner := [LeftMerge92524.delta]
theorem rows0_1 : MergeDeltasAt history frameStart owner group deltas0_1 := by
  exact .leaf LeftMerge92524.deltaAt
def deltas1_0 : Polynomial Owner := deltas0_0 ++ deltas0_1
theorem rows1_0 : MergeDeltasAt history frameStart owner group deltas1_0 := by
  exact .append rows0_0 rows0_1
abbrev deltas : Polynomial Owner := deltas1_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows1_0
def left : Polynomial Owner := LeftMerge92521.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge92521.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨13956⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge92521

namespace LeftOperatorMerge77874
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨13972⟩⟩
def group : MergeGroup := .operator 3181 75903
def deltas0_0 : Polynomial Owner := [LeftMerge77874.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge77874.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge77874.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge77874.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨13971⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right true false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge77874

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
