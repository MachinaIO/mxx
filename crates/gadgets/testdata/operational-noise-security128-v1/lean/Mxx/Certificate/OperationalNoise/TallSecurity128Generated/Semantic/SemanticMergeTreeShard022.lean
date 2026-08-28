import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard727
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard730
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard732
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard735
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard738
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard740
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard743
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard745
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard751
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard754
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard756
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard759
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard761
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard764
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard767
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard807

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace LeftOperatorMerge123172
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨8128⟩⟩
def group : MergeGroup := .operator 119648 20587
def deltas0_0 : Polynomial Owner := [LeftMerge123172.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge123172.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge123172.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge123172.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7278⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right true false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge123172

namespace LeftOperatorMerge122690
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨8129⟩⟩
def group : MergeGroup := .operator 119648 20086
def deltas0_0 : Polynomial Owner := [LeftMerge122690.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge122690.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge122690.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge122690.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7279⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right true false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge122690

namespace LeftOperatorMerge122208
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨8130⟩⟩
def group : MergeGroup := .operator 119648 19585
def deltas0_0 : Polynomial Owner := [LeftMerge122208.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge122208.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge122208.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge122208.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7280⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right true false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge122208

namespace LeftOperatorMerge121726
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨8131⟩⟩
def group : MergeGroup := .operator 119648 19084
def deltas0_0 : Polynomial Owner := [LeftMerge121726.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge121726.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge121726.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge121726.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7281⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right true false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge121726

namespace LeftOperatorMerge121244
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨8132⟩⟩
def group : MergeGroup := .operator 119648 18583
def deltas0_0 : Polynomial Owner := [LeftMerge121244.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge121244.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge121244.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge121244.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7282⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right true false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge121244

namespace LeftOperatorMerge120762
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨8133⟩⟩
def group : MergeGroup := .operator 119648 18082
def deltas0_0 : Polynomial Owner := [LeftMerge120762.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge120762.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge120762.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge120762.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7283⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right true false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge120762

namespace LeftOperatorMerge120280
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨8134⟩⟩
def group : MergeGroup := .operator 119648 17581
def deltas0_0 : Polynomial Owner := [LeftMerge120280.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge120280.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge120280.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge120280.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7284⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right true false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge120280

namespace LeftOperatorMerge119787
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨8135⟩⟩
def group : MergeGroup := .operator 119648 17065
def deltas0_0 : Polynomial Owner := [LeftMerge119787.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge119787.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge119787.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge119787.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7285⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right true false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge119787

namespace LeftOperatorMerge127056
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨8136⟩⟩
def group : MergeGroup := .operator 119648 24636
def deltas0_0 : Polynomial Owner := [LeftMerge127056.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge127056.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge127056.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge127056.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7286⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right true false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge127056

namespace LeftOperatorMerge126574
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨8137⟩⟩
def group : MergeGroup := .operator 119648 24135
def deltas0_0 : Polynomial Owner := [LeftMerge126574.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge126574.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge126574.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge126574.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right true false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge126574

namespace LeftOperatorMerge126092
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨8138⟩⟩
def group : MergeGroup := .operator 119648 23634
def deltas0_0 : Polynomial Owner := [LeftMerge126092.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge126092.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge126092.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge126092.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7288⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right true false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge126092

namespace LeftOperatorMerge125610
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨8139⟩⟩
def group : MergeGroup := .operator 119648 23133
def deltas0_0 : Polynomial Owner := [LeftMerge125610.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge125610.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge125610.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge125610.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right true false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge125610

namespace LeftOperatorMerge125128
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨8140⟩⟩
def group : MergeGroup := .operator 119648 22632
def deltas0_0 : Polynomial Owner := [LeftMerge125128.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge125128.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge125128.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge125128.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7290⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right true false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge125128

namespace LeftOperatorMerge124646
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨8141⟩⟩
def group : MergeGroup := .operator 119648 22131
def deltas0_0 : Polynomial Owner := [LeftMerge124646.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge124646.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge124646.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge124646.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7291⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right true false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge124646

namespace LeftOperatorMerge133900
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨8142⟩⟩
def group : MergeGroup := .operator 119648 15896
def deltas0_0 : Polynomial Owner := [LeftMerge133900.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge133900.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge133900.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge133900.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7292⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right true false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge133900

namespace LeftOperatorMerge124164
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨8143⟩⟩
def group : MergeGroup := .operator 119648 21630
def deltas0_0 : Polynomial Owner := [LeftMerge124164.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge124164.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge124164.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge124164.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7293⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right true false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge124164

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
