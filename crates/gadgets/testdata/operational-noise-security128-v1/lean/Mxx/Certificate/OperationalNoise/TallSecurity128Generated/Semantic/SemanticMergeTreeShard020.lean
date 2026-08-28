import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard1553
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard1556
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard1562
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard1564
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard1567
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard1569
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard1572
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard1575
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard1577
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard1580
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard1583
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard1585
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard1588
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard1590
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard1593
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard1634

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace LeftOperatorMerge251905
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨8020⟩⟩
def group : MergeGroup := .operator 251273 17581
def deltas0_0 : Polynomial Owner := [LeftMerge251905.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge251905.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge251905.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge251905.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7284⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right true false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge251905

namespace LeftOperatorMerge251412
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨8021⟩⟩
def group : MergeGroup := .operator 251273 17065
def deltas0_0 : Polynomial Owner := [LeftMerge251412.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge251412.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge251412.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge251412.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7285⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right true false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge251412

namespace LeftOperatorMerge258681
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨8022⟩⟩
def group : MergeGroup := .operator 251273 24636
def deltas0_0 : Polynomial Owner := [LeftMerge258681.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge258681.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge258681.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge258681.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7286⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right true false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge258681

namespace LeftOperatorMerge258199
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨8023⟩⟩
def group : MergeGroup := .operator 251273 24135
def deltas0_0 : Polynomial Owner := [LeftMerge258199.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge258199.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge258199.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge258199.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right true false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge258199

namespace LeftOperatorMerge257717
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨8024⟩⟩
def group : MergeGroup := .operator 251273 23634
def deltas0_0 : Polynomial Owner := [LeftMerge257717.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge257717.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge257717.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge257717.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7288⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right true false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge257717

namespace LeftOperatorMerge257235
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨8025⟩⟩
def group : MergeGroup := .operator 251273 23133
def deltas0_0 : Polynomial Owner := [LeftMerge257235.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge257235.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge257235.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge257235.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right true false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge257235

namespace LeftOperatorMerge256753
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨8026⟩⟩
def group : MergeGroup := .operator 251273 22632
def deltas0_0 : Polynomial Owner := [LeftMerge256753.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge256753.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge256753.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge256753.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7290⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right true false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge256753

namespace LeftOperatorMerge256271
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨8027⟩⟩
def group : MergeGroup := .operator 251273 22131
def deltas0_0 : Polynomial Owner := [LeftMerge256271.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge256271.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge256271.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge256271.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7291⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right true false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge256271

namespace LeftOperatorMerge265525
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨8028⟩⟩
def group : MergeGroup := .operator 251273 15896
def deltas0_0 : Polynomial Owner := [LeftMerge265525.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge265525.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge265525.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge265525.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7292⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right true false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge265525

namespace LeftOperatorMerge255789
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨8029⟩⟩
def group : MergeGroup := .operator 251273 21630
def deltas0_0 : Polynomial Owner := [LeftMerge255789.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge255789.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge255789.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge255789.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7293⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right true false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge255789

namespace LeftOperatorMerge255307
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨8030⟩⟩
def group : MergeGroup := .operator 251273 21129
def deltas0_0 : Polynomial Owner := [LeftMerge255307.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge255307.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge255307.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge255307.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7294⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right true false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge255307

namespace LeftOperatorMerge254825
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨8031⟩⟩
def group : MergeGroup := .operator 251273 20628
def deltas0_0 : Polynomial Owner := [LeftMerge254825.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge254825.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge254825.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge254825.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7295⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right true false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge254825

namespace LeftOperatorMerge254343
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨8032⟩⟩
def group : MergeGroup := .operator 251273 20127
def deltas0_0 : Polynomial Owner := [LeftMerge254343.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge254343.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge254343.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge254343.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7296⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right true false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge254343

namespace LeftOperatorMerge253861
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨8033⟩⟩
def group : MergeGroup := .operator 251273 19626
def deltas0_0 : Polynomial Owner := [LeftMerge253861.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge253861.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge253861.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge253861.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7297⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right true false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge253861

namespace LeftOperatorMerge253379
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨8034⟩⟩
def group : MergeGroup := .operator 251273 19125
def deltas0_0 : Polynomial Owner := [LeftMerge253379.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge253379.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge253379.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge253379.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right true false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge253379

namespace LeftOperatorMerge252897
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨8035⟩⟩
def group : MergeGroup := .operator 251273 18624
def deltas0_0 : Polynomial Owner := [LeftMerge252897.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge252897.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge252897.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge252897.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right true false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge252897

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
