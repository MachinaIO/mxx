import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard727
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard730
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard733
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard735
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard738
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard740
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard743
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard746
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard748
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard761
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard764
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard766
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard769
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard772
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard908

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace LeftOperatorMerge123682
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨8144⟩⟩
def group : MergeGroup := .operator 119648 21129
def deltas0_0 : Polynomial Owner := [LeftMerge123682.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge123682.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge123682.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge123682.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7294⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right true false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge123682

namespace LeftOperatorMerge123200
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨8145⟩⟩
def group : MergeGroup := .operator 119648 20628
def deltas0_0 : Polynomial Owner := [LeftMerge123200.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge123200.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge123200.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge123200.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7295⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right true false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge123200

namespace LeftOperatorMerge122718
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨8146⟩⟩
def group : MergeGroup := .operator 119648 20127
def deltas0_0 : Polynomial Owner := [LeftMerge122718.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge122718.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge122718.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge122718.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7296⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right true false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge122718

namespace LeftOperatorMerge122236
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨8147⟩⟩
def group : MergeGroup := .operator 119648 19626
def deltas0_0 : Polynomial Owner := [LeftMerge122236.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge122236.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge122236.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge122236.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7297⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right true false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge122236

namespace LeftOperatorMerge121754
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨8148⟩⟩
def group : MergeGroup := .operator 119648 19125
def deltas0_0 : Polynomial Owner := [LeftMerge121754.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge121754.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge121754.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge121754.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right true false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge121754

namespace LeftOperatorMerge121272
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨8149⟩⟩
def group : MergeGroup := .operator 119648 18624
def deltas0_0 : Polynomial Owner := [LeftMerge121272.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge121272.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge121272.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge121272.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right true false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge121272

namespace LeftOperatorMerge120790
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨8150⟩⟩
def group : MergeGroup := .operator 119648 18123
def deltas0_0 : Polynomial Owner := [LeftMerge120790.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge120790.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge120790.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge120790.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7300⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right true false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge120790

namespace LeftOperatorMerge120308
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨8151⟩⟩
def group : MergeGroup := .operator 119648 17622
def deltas0_0 : Polynomial Owner := [LeftMerge120308.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge120308.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge120308.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge120308.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right true false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge120308

namespace LeftOperatorMerge119815
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨8152⟩⟩
def group : MergeGroup := .operator 119648 17106
def deltas0_0 : Polynomial Owner := [LeftMerge119815.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge119815.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge119815.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge119815.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right true false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge119815

namespace LeftOperatorMerge128020
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨8153⟩⟩
def group : MergeGroup := .operator 119648 25638
def deltas0_0 : Polynomial Owner := [LeftMerge128020.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge128020.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge128020.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge128020.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7303⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right true false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge128020

namespace LeftOperatorMerge127992
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨8154⟩⟩
def group : MergeGroup := .operator 119648 25597
def deltas0_0 : Polynomial Owner := [LeftMerge127992.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge127992.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge127992.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge127992.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7304⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right true false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge127992

namespace LeftOperatorMerge127510
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨8155⟩⟩
def group : MergeGroup := .operator 119648 25096
def deltas0_0 : Polynomial Owner := [LeftMerge127510.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge127510.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge127510.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge127510.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7305⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right true false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge127510

namespace LeftOperatorMerge127028
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨8156⟩⟩
def group : MergeGroup := .operator 119648 24595
def deltas0_0 : Polynomial Owner := [LeftMerge127028.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge127028.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge127028.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge127028.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7306⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right true false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge127028

namespace LeftOperatorMerge126546
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨8157⟩⟩
def group : MergeGroup := .operator 119648 24094
def deltas0_0 : Polynomial Owner := [LeftMerge126546.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge126546.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge126546.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge126546.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7307⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right true false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge126546

namespace LeftOperatorMerge126064
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨8158⟩⟩
def group : MergeGroup := .operator 119648 23593
def deltas0_0 : Polynomial Owner := [LeftMerge126064.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge126064.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge126064.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge126064.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7308⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right true false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge126064

namespace LeftOperatorMerge148902
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨8235⟩⟩
def group : MergeGroup := .operator 148898 16337
def deltas0_0 : Polynomial Owner := [LeftMerge148902.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge148902.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge148902.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge148902.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7253⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right true false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge148902

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
