import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard087
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard088
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard821
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard822
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard1556
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard1648
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard1740
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard1829

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace LeftOperatorMerge295574
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨44920⟩⟩
def group : MergeGroup := .operator 295568 14316
def deltas0_0 : Polynomial Owner := [LeftMerge295574.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge295574.deltaAt
def deltas0_1 : Polynomial Owner := [LeftMerge295575.delta]
theorem rows0_1 : MergeDeltasAt history frameStart owner group deltas0_1 := by
  exact .leaf LeftMerge295575.deltaAt
def deltas1_0 : Polynomial Owner := deltas0_0 ++ deltas0_1
theorem rows1_0 : MergeDeltasAt history frameStart owner group deltas1_0 := by
  exact .append rows0_0 rows0_1
abbrev deltas : Polynomial Owner := deltas1_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows1_0
def left : Polynomial Owner := LeftMerge295574.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge295574.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨14631⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7284⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨14631⟩⟩, ⟨.program ⟨257⟩, ⟨44914⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge295574

namespace LeftOperatorMerge295610
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨44921⟩⟩
def group : MergeGroup := .operator 295606 295576
def deltas0_0 : Polynomial Owner := [LeftMerge295610.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge295610.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge295610.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge295610.rightRaw.map Term.toExact
def base : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨14631⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7284⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨14631⟩⟩, ⟨.program ⟨257⟩, ⟨44914⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }]
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨14631⟩⟩, ⟨.program ⟨257⟩, ⟨44914⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (add left right) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge295610

namespace LeftOperatorMerge17577
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨44949⟩⟩
def group : MergeGroup := .operator 74 17057
def deltas0_0 : Polynomial Owner := [LeftMerge17577.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge17577.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge17577.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge17577.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨44946⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right true false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge17577

namespace LeftOperatorMerge17602
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨44952⟩⟩
def group : MergeGroup := .operator 17596 77
def deltas0_0 : Polynomial Owner := [LeftMerge17602.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge17602.deltaAt
def deltas0_1 : Polynomial Owner := [LeftMerge17603.delta]
theorem rows0_1 : MergeDeltasAt history frameStart owner group deltas0_1 := by
  exact .leaf LeftMerge17603.deltaAt
def deltas1_0 : Polynomial Owner := deltas0_0 ++ deltas0_1
theorem rows1_0 : MergeDeltasAt history frameStart owner group deltas1_0 := by
  exact .append rows0_0 rows0_1
abbrev deltas : Polynomial Owner := deltas1_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows1_0
def left : Polynomial Owner := LeftMerge17602.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge17602.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨14651⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7284⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨14651⟩⟩, ⟨.program ⟨257⟩, ⟨44946⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge17602

namespace LeftOperatorMerge17651
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨44953⟩⟩
def group : MergeGroup := .operator 17647 17604
def deltas0_0 : Polynomial Owner := [LeftMerge17651.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge17651.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge17651.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge17651.rightRaw.map Term.toExact
def base : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨14651⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7284⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨14651⟩⟩, ⟨.program ⟨257⟩, ⟨44946⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }]
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨14651⟩⟩, ⟨.program ⟨257⟩, ⟨44946⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (add left right) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge17651

namespace LeftOperatorMerge266525
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨44957⟩⟩
def group : MergeGroup := .operator 12833 266028
def deltas0_0 : Polynomial Owner := [LeftMerge266525.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge266525.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge266525.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge266525.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨44954⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right true false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge266525

namespace LeftOperatorMerge266547
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨44960⟩⟩
def group : MergeGroup := .operator 266541 12836
def deltas0_0 : Polynomial Owner := [LeftMerge266547.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge266547.deltaAt
def deltas0_1 : Polynomial Owner := [LeftMerge266548.delta]
theorem rows0_1 : MergeDeltasAt history frameStart owner group deltas0_1 := by
  exact .leaf LeftMerge266548.deltaAt
def deltas1_0 : Polynomial Owner := deltas0_0 ++ deltas0_1
theorem rows1_0 : MergeDeltasAt history frameStart owner group deltas1_0 := by
  exact .append rows0_0 rows0_1
abbrev deltas : Polynomial Owner := deltas1_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows1_0
def left : Polynomial Owner := LeftMerge266547.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge266547.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨14656⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7284⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨14656⟩⟩, ⟨.program ⟨257⟩, ⟨44954⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge266547

namespace LeftOperatorMerge266583
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨44961⟩⟩
def group : MergeGroup := .operator 266579 266549
def deltas0_0 : Polynomial Owner := [LeftMerge266583.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge266583.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge266583.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge266583.rightRaw.map Term.toExact
def base : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨14656⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7284⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨14656⟩⟩, ⟨.program ⟨257⟩, ⟨44954⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }]
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨14656⟩⟩, ⟨.program ⟨257⟩, ⟨44954⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (add left right) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge266583

namespace LeftOperatorMerge134900
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨44989⟩⟩
def group : MergeGroup := .operator 6101 134403
def deltas0_0 : Polynomial Owner := [LeftMerge134900.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge134900.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge134900.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge134900.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨44986⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right true false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge134900

namespace LeftOperatorMerge134922
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨44992⟩⟩
def group : MergeGroup := .operator 134916 6104
def deltas0_0 : Polynomial Owner := [LeftMerge134922.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge134922.deltaAt
def deltas0_1 : Polynomial Owner := [LeftMerge134923.delta]
theorem rows0_1 : MergeDeltasAt history frameStart owner group deltas0_1 := by
  exact .leaf LeftMerge134923.deltaAt
def deltas1_0 : Polynomial Owner := deltas0_0 ++ deltas0_1
theorem rows1_0 : MergeDeltasAt history frameStart owner group deltas1_0 := by
  exact .append rows0_0 rows0_1
abbrev deltas : Polynomial Owner := deltas1_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows1_0
def left : Polynomial Owner := LeftMerge134922.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge134922.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨14676⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7284⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨14676⟩⟩, ⟨.program ⟨257⟩, ⟨44986⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge134922

namespace LeftOperatorMerge134958
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨44993⟩⟩
def group : MergeGroup := .operator 134954 134924
def deltas0_0 : Polynomial Owner := [LeftMerge134958.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge134958.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge134958.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge134958.rightRaw.map Term.toExact
def base : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨14676⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7284⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨14676⟩⟩, ⟨.program ⟨257⟩, ⟨44986⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }]
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨14676⟩⟩, ⟨.program ⟨257⟩, ⟨44986⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (add left right) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge134958

namespace LeftOperatorMerge281148
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨45013⟩⟩
def group : MergeGroup := .operator 13575 280653
def deltas0_0 : Polynomial Owner := [LeftMerge281148.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge281148.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge281148.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge281148.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨45010⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right true false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge281148

namespace LeftOperatorMerge281170
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨45016⟩⟩
def group : MergeGroup := .operator 281164 13578
def deltas0_0 : Polynomial Owner := [LeftMerge281170.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge281170.deltaAt
def deltas0_1 : Polynomial Owner := [LeftMerge281171.delta]
theorem rows0_1 : MergeDeltasAt history frameStart owner group deltas0_1 := by
  exact .leaf LeftMerge281171.deltaAt
def deltas1_0 : Polynomial Owner := deltas0_0 ++ deltas0_1
theorem rows1_0 : MergeDeltasAt history frameStart owner group deltas1_0 := by
  exact .append rows0_0 rows0_1
abbrev deltas : Polynomial Owner := deltas1_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows1_0
def left : Polynomial Owner := LeftMerge281170.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge281170.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨14691⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7284⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨14691⟩⟩, ⟨.program ⟨257⟩, ⟨45010⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge281170

namespace LeftOperatorMerge281206
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨45017⟩⟩
def group : MergeGroup := .operator 281202 281172
def deltas0_0 : Polynomial Owner := [LeftMerge281206.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge281206.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge281206.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge281206.rightRaw.map Term.toExact
def base : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨14691⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7284⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨14691⟩⟩, ⟨.program ⟨257⟩, ⟨45010⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }]
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨14691⟩⟩, ⟨.program ⟨257⟩, ⟨45010⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (add left right) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge281206

namespace LeftOperatorMerge251900
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨45037⟩⟩
def group : MergeGroup := .operator 12085 251403
def deltas0_0 : Polynomial Owner := [LeftMerge251900.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge251900.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge251900.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge251900.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨45034⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right true false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge251900

namespace LeftOperatorMerge251922
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨45040⟩⟩
def group : MergeGroup := .operator 251916 12088
def deltas0_0 : Polynomial Owner := [LeftMerge251922.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge251922.deltaAt
def deltas0_1 : Polynomial Owner := [LeftMerge251923.delta]
theorem rows0_1 : MergeDeltasAt history frameStart owner group deltas0_1 := by
  exact .leaf LeftMerge251923.deltaAt
def deltas1_0 : Polynomial Owner := deltas0_0 ++ deltas0_1
theorem rows1_0 : MergeDeltasAt history frameStart owner group deltas1_0 := by
  exact .append rows0_0 rows0_1
abbrev deltas : Polynomial Owner := deltas1_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows1_0
def left : Polynomial Owner := LeftMerge251922.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge251922.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨14706⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7284⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨14706⟩⟩, ⟨.program ⟨257⟩, ⟨45034⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge251922

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
