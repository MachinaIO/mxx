import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeDeltaShard218
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeDeltaShard231
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeDeltaShard233
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeDeltaShard236
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeDeltaShard239
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeDeltaShard241
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeDeltaShard244
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeDeltaShard247
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeDeltaShard249
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeDeltaShard252
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeDeltaShard254
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeDeltaShard257
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeDeltaShard260
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeDeltaShard262
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeDeltaShard263

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace LeftOperatorMerge36082
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨214⟩, ⟨7302⟩⟩
def group : MergeGroup := .operator 35915 6498
def deltas0_0 : Polynomial Owner := [LeftMerge36082.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge36082.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge36082.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge36082.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6770⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right true false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge36082

namespace LeftOperatorMerge44287
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨214⟩, ⟨7303⟩⟩
def group : MergeGroup := .operator 35915 15030
def deltas0_0 : Polynomial Owner := [LeftMerge44287.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge44287.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge44287.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge44287.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6771⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right true false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge44287

namespace LeftOperatorMerge44259
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨214⟩, ⟨7304⟩⟩
def group : MergeGroup := .operator 35915 14989
def deltas0_0 : Polynomial Owner := [LeftMerge44259.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge44259.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge44259.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge44259.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6772⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right true false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge44259

namespace LeftOperatorMerge43777
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨214⟩, ⟨7305⟩⟩
def group : MergeGroup := .operator 35915 14488
def deltas0_0 : Polynomial Owner := [LeftMerge43777.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge43777.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge43777.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge43777.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6773⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right true false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge43777

namespace LeftOperatorMerge43295
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨214⟩, ⟨7306⟩⟩
def group : MergeGroup := .operator 35915 13987
def deltas0_0 : Polynomial Owner := [LeftMerge43295.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge43295.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge43295.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge43295.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6774⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right true false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge43295

namespace LeftOperatorMerge42813
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨214⟩, ⟨7307⟩⟩
def group : MergeGroup := .operator 35915 13486
def deltas0_0 : Polynomial Owner := [LeftMerge42813.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge42813.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge42813.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge42813.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6775⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right true false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge42813

namespace LeftOperatorMerge42331
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨214⟩, ⟨7308⟩⟩
def group : MergeGroup := .operator 35915 12985
def deltas0_0 : Polynomial Owner := [LeftMerge42331.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge42331.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge42331.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge42331.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6776⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right true false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge42331

namespace LeftOperatorMerge41849
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨214⟩, ⟨7309⟩⟩
def group : MergeGroup := .operator 35915 12484
def deltas0_0 : Polynomial Owner := [LeftMerge41849.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge41849.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge41849.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge41849.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6777⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right true false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge41849

namespace LeftOperatorMerge41367
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨214⟩, ⟨7310⟩⟩
def group : MergeGroup := .operator 35915 11983
def deltas0_0 : Polynomial Owner := [LeftMerge41367.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge41367.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge41367.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge41367.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6778⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right true false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge41367

namespace LeftOperatorMerge40885
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨214⟩, ⟨7311⟩⟩
def group : MergeGroup := .operator 35915 11482
def deltas0_0 : Polynomial Owner := [LeftMerge40885.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge40885.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge40885.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge40885.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6779⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right true false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge40885

namespace LeftOperatorMerge40403
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨214⟩, ⟨7312⟩⟩
def group : MergeGroup := .operator 35915 10981
def deltas0_0 : Polynomial Owner := [LeftMerge40403.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge40403.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge40403.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge40403.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6780⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right true false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge40403

namespace LeftOperatorMerge39921
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨214⟩, ⟨7313⟩⟩
def group : MergeGroup := .operator 35915 10480
def deltas0_0 : Polynomial Owner := [LeftMerge39921.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge39921.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge39921.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge39921.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6781⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right true false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge39921

namespace LeftOperatorMerge43805
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨214⟩, ⟨7314⟩⟩
def group : MergeGroup := .operator 35915 14529
def deltas0_0 : Polynomial Owner := [LeftMerge43805.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge43805.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge43805.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge43805.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6782⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right true false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge43805

namespace LeftOperatorMerge39439
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨214⟩, ⟨7315⟩⟩
def group : MergeGroup := .operator 35915 9979
def deltas0_0 : Polynomial Owner := [LeftMerge39439.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge39439.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge39439.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge39439.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6783⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right true false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge39439

namespace LeftOperatorMerge38957
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨214⟩, ⟨7316⟩⟩
def group : MergeGroup := .operator 35915 9478
def deltas0_0 : Polynomial Owner := [LeftMerge38957.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge38957.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge38957.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge38957.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6784⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right true false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge38957

namespace LeftOperatorMerge38475
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨214⟩, ⟨7317⟩⟩
def group : MergeGroup := .operator 35915 8977
def deltas0_0 : Polynomial Owner := [LeftMerge38475.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge38475.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge38475.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge38475.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6785⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right true false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge38475

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
