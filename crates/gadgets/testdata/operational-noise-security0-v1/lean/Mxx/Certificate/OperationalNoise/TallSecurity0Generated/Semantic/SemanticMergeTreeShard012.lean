import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeDeltaShard309
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeDeltaShard312
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeDeltaShard315
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeDeltaShard317
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeDeltaShard320
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeDeltaShard323
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeDeltaShard325
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeDeltaShard328
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeDeltaShard330
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeDeltaShard333
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeDeltaShard336
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeDeltaShard338
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeDeltaShard341
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeDeltaShard344
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeDeltaShard349
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeDeltaShard352

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace LeftOperatorMerge56956
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨214⟩, ⟨7270⟩⟩
def group : MergeGroup := .operator 50540 12985
def deltas0_0 : Polynomial Owner := [LeftMerge56956.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge56956.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge56956.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge56956.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6776⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right true false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge56956

namespace LeftOperatorMerge56474
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨214⟩, ⟨7271⟩⟩
def group : MergeGroup := .operator 50540 12484
def deltas0_0 : Polynomial Owner := [LeftMerge56474.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge56474.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge56474.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge56474.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6777⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right true false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge56474

namespace LeftOperatorMerge55992
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨214⟩, ⟨7272⟩⟩
def group : MergeGroup := .operator 50540 11983
def deltas0_0 : Polynomial Owner := [LeftMerge55992.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge55992.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge55992.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge55992.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6778⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right true false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge55992

namespace LeftOperatorMerge55510
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨214⟩, ⟨7273⟩⟩
def group : MergeGroup := .operator 50540 11482
def deltas0_0 : Polynomial Owner := [LeftMerge55510.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge55510.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge55510.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge55510.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6779⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right true false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge55510

namespace LeftOperatorMerge55028
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨214⟩, ⟨7274⟩⟩
def group : MergeGroup := .operator 50540 10981
def deltas0_0 : Polynomial Owner := [LeftMerge55028.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge55028.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge55028.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge55028.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6780⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right true false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge55028

namespace LeftOperatorMerge54546
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨214⟩, ⟨7275⟩⟩
def group : MergeGroup := .operator 50540 10480
def deltas0_0 : Polynomial Owner := [LeftMerge54546.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge54546.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge54546.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge54546.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6781⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right true false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge54546

namespace LeftOperatorMerge58430
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨214⟩, ⟨7276⟩⟩
def group : MergeGroup := .operator 50540 14529
def deltas0_0 : Polynomial Owner := [LeftMerge58430.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge58430.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge58430.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge58430.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6782⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right true false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge58430

namespace LeftOperatorMerge54064
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨214⟩, ⟨7277⟩⟩
def group : MergeGroup := .operator 50540 9979
def deltas0_0 : Polynomial Owner := [LeftMerge54064.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge54064.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge54064.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge54064.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6783⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right true false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge54064

namespace LeftOperatorMerge53582
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨214⟩, ⟨7278⟩⟩
def group : MergeGroup := .operator 50540 9478
def deltas0_0 : Polynomial Owner := [LeftMerge53582.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge53582.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge53582.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge53582.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6784⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right true false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge53582

namespace LeftOperatorMerge53100
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨214⟩, ⟨7279⟩⟩
def group : MergeGroup := .operator 50540 8977
def deltas0_0 : Polynomial Owner := [LeftMerge53100.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge53100.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge53100.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge53100.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6785⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right true false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge53100

namespace LeftOperatorMerge52618
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨214⟩, ⟨7280⟩⟩
def group : MergeGroup := .operator 50540 8476
def deltas0_0 : Polynomial Owner := [LeftMerge52618.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge52618.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge52618.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge52618.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6786⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right true false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge52618

namespace LeftOperatorMerge52136
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨214⟩, ⟨7281⟩⟩
def group : MergeGroup := .operator 50540 7975
def deltas0_0 : Polynomial Owner := [LeftMerge52136.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge52136.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge52136.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge52136.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6787⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right true false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge52136

namespace LeftOperatorMerge51654
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨214⟩, ⟨7282⟩⟩
def group : MergeGroup := .operator 50540 7474
def deltas0_0 : Polynomial Owner := [LeftMerge51654.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge51654.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge51654.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge51654.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6788⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right true false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge51654

namespace LeftOperatorMerge51172
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨214⟩, ⟨7283⟩⟩
def group : MergeGroup := .operator 50540 6973
def deltas0_0 : Polynomial Owner := [LeftMerge51172.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge51172.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge51172.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge51172.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6789⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right true false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge51172

namespace LeftOperatorMerge50679
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨214⟩, ⟨7284⟩⟩
def group : MergeGroup := .operator 50540 6457
def deltas0_0 : Polynomial Owner := [LeftMerge50679.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge50679.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge50679.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge50679.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6790⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right true false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge50679

namespace LeftOperatorMerge57948
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨214⟩, ⟨7285⟩⟩
def group : MergeGroup := .operator 50540 14028
def deltas0_0 : Polynomial Owner := [LeftMerge57948.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge57948.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge57948.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge57948.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6791⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right true false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge57948

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
