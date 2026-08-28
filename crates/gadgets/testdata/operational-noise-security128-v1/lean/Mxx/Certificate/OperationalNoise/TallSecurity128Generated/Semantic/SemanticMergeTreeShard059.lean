import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard268
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard271
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard274
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard276
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard279
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard281
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard284
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard287
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard289
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard292
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard295
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard297
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard300
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard302
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard305
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard348

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace LeftOperatorMerge53449
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨11193⟩⟩
def group : MergeGroup := .operator 46523 24135
def deltas0_0 : Polynomial Owner := [LeftMerge53449.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge53449.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge53449.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge53449.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right true false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge53449

namespace LeftOperatorMerge52967
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨11194⟩⟩
def group : MergeGroup := .operator 46523 23634
def deltas0_0 : Polynomial Owner := [LeftMerge52967.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge52967.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge52967.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge52967.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7288⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right true false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge52967

namespace LeftOperatorMerge52485
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨11195⟩⟩
def group : MergeGroup := .operator 46523 23133
def deltas0_0 : Polynomial Owner := [LeftMerge52485.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge52485.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge52485.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge52485.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right true false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge52485

namespace LeftOperatorMerge52003
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨11196⟩⟩
def group : MergeGroup := .operator 46523 22632
def deltas0_0 : Polynomial Owner := [LeftMerge52003.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge52003.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge52003.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge52003.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7290⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right true false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge52003

namespace LeftOperatorMerge51521
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨11197⟩⟩
def group : MergeGroup := .operator 46523 22131
def deltas0_0 : Polynomial Owner := [LeftMerge51521.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge51521.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge51521.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge51521.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7291⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right true false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge51521

namespace LeftOperatorMerge60775
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨11198⟩⟩
def group : MergeGroup := .operator 46523 15896
def deltas0_0 : Polynomial Owner := [LeftMerge60775.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge60775.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge60775.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge60775.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7292⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right true false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge60775

namespace LeftOperatorMerge51039
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨11199⟩⟩
def group : MergeGroup := .operator 46523 21630
def deltas0_0 : Polynomial Owner := [LeftMerge51039.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge51039.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge51039.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge51039.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7293⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right true false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge51039

namespace LeftOperatorMerge50557
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨11200⟩⟩
def group : MergeGroup := .operator 46523 21129
def deltas0_0 : Polynomial Owner := [LeftMerge50557.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge50557.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge50557.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge50557.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7294⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right true false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge50557

namespace LeftOperatorMerge50075
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨11201⟩⟩
def group : MergeGroup := .operator 46523 20628
def deltas0_0 : Polynomial Owner := [LeftMerge50075.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge50075.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge50075.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge50075.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7295⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right true false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge50075

namespace LeftOperatorMerge49593
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨11202⟩⟩
def group : MergeGroup := .operator 46523 20127
def deltas0_0 : Polynomial Owner := [LeftMerge49593.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge49593.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge49593.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge49593.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7296⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right true false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge49593

namespace LeftOperatorMerge49111
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨11203⟩⟩
def group : MergeGroup := .operator 46523 19626
def deltas0_0 : Polynomial Owner := [LeftMerge49111.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge49111.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge49111.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge49111.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7297⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right true false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge49111

namespace LeftOperatorMerge48629
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨11204⟩⟩
def group : MergeGroup := .operator 46523 19125
def deltas0_0 : Polynomial Owner := [LeftMerge48629.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge48629.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge48629.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge48629.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right true false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge48629

namespace LeftOperatorMerge48147
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨11205⟩⟩
def group : MergeGroup := .operator 46523 18624
def deltas0_0 : Polynomial Owner := [LeftMerge48147.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge48147.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge48147.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge48147.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right true false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge48147

namespace LeftOperatorMerge47665
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨11206⟩⟩
def group : MergeGroup := .operator 46523 18123
def deltas0_0 : Polynomial Owner := [LeftMerge47665.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge47665.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge47665.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge47665.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7300⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right true false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge47665

namespace LeftOperatorMerge47183
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨11207⟩⟩
def group : MergeGroup := .operator 46523 17622
def deltas0_0 : Polynomial Owner := [LeftMerge47183.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge47183.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge47183.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge47183.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right true false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge47183

namespace LeftOperatorMerge46690
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨11208⟩⟩
def group : MergeGroup := .operator 46523 17106
def deltas0_0 : Polynomial Owner := [LeftMerge46690.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge46690.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge46690.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge46690.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right true false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge46690

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
