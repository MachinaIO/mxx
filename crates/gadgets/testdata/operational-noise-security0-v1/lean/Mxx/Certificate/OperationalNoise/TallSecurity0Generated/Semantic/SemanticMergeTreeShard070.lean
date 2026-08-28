import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeDeltaShard000
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeDeltaShard002
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeDeltaShard004
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeDeltaShard007
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeDeltaShard009
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeDeltaShard012
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeDeltaShard014
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeDeltaShard452
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeDeltaShard633

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace LeftOperatorMerge5063
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨214⟩, ⟨18115⟩⟩
def group : MergeGroup := .operator 5059 543
def deltas0_0 : Polynomial Owner := [LeftMerge5063.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge5063.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge5063.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge5063.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6435⟩⟩, ⟨.program ⟨214⟩, ⟨18114⟩⟩], orderedFactors := [] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge5063

namespace LeftOperatorMerge3583
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨214⟩, ⟨18121⟩⟩
def group : MergeGroup := .operator 3579 543
def deltas0_0 : Polynomial Owner := [LeftMerge3583.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge3583.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge3583.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge3583.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6435⟩⟩, ⟨.program ⟨214⟩, ⟨18120⟩⟩], orderedFactors := [] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge3583

namespace LeftOperatorMerge4325
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨214⟩, ⟨18125⟩⟩
def group : MergeGroup := .operator 4321 543
def deltas0_0 : Polynomial Owner := [LeftMerge4325.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge4325.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge4325.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge4325.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6435⟩⟩, ⟨.program ⟨214⟩, ⟨18124⟩⟩], orderedFactors := [] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge4325

namespace LeftOperatorMerge2835
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨214⟩, ⟨18129⟩⟩
def group : MergeGroup := .operator 2831 543
def deltas0_0 : Polynomial Owner := [LeftMerge2835.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge2835.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge2835.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge2835.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6435⟩⟩, ⟨.program ⟨214⟩, ⟨18128⟩⟩], orderedFactors := [] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge2835

namespace LeftOperatorMerge2087
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨214⟩, ⟨18133⟩⟩
def group : MergeGroup := .operator 2083 543
def deltas0_0 : Polynomial Owner := [LeftMerge2087.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge2087.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge2087.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge2087.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6435⟩⟩, ⟨.program ⟨214⟩, ⟨18132⟩⟩], orderedFactors := [] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge2087

namespace LeftOperatorMerge1339
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨214⟩, ⟨18137⟩⟩
def group : MergeGroup := .operator 1335 543
def deltas0_0 : Polynomial Owner := [LeftMerge1339.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge1339.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge1339.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge1339.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6435⟩⟩, ⟨.program ⟨214⟩, ⟨18136⟩⟩], orderedFactors := [] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge1339

namespace LeftOperatorMerge550
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨214⟩, ⟨18141⟩⟩
def group : MergeGroup := .operator 546 543
def deltas0_0 : Polynomial Owner := [LeftMerge550.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge550.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge550.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge550.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6435⟩⟩, ⟨.program ⟨214⟩, ⟨18140⟩⟩], orderedFactors := [] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge550

namespace LeftOperatorMerge5055
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨214⟩, ⟨18486⟩⟩
def group : MergeGroup := .operator 5051 36
def deltas0_0 : Polynomial Owner := [LeftMerge5055.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge5055.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge5055.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge5055.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6410⟩⟩, ⟨.program ⟨214⟩, ⟨18485⟩⟩], orderedFactors := [] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge5055

namespace LeftOperatorMerge3575
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨214⟩, ⟨18492⟩⟩
def group : MergeGroup := .operator 3571 36
def deltas0_0 : Polynomial Owner := [LeftMerge3575.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge3575.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge3575.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge3575.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6410⟩⟩, ⟨.program ⟨214⟩, ⟨18491⟩⟩], orderedFactors := [] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge3575

namespace LeftOperatorMerge4317
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨214⟩, ⟨18496⟩⟩
def group : MergeGroup := .operator 4313 36
def deltas0_0 : Polynomial Owner := [LeftMerge4317.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge4317.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge4317.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge4317.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6410⟩⟩, ⟨.program ⟨214⟩, ⟨18495⟩⟩], orderedFactors := [] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge4317

namespace LeftOperatorMerge2827
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨214⟩, ⟨18500⟩⟩
def group : MergeGroup := .operator 2823 36
def deltas0_0 : Polynomial Owner := [LeftMerge2827.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge2827.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge2827.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge2827.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6410⟩⟩, ⟨.program ⟨214⟩, ⟨18499⟩⟩], orderedFactors := [] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge2827

namespace LeftOperatorMerge2079
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨214⟩, ⟨18504⟩⟩
def group : MergeGroup := .operator 2075 36
def deltas0_0 : Polynomial Owner := [LeftMerge2079.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge2079.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge2079.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge2079.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6410⟩⟩, ⟨.program ⟨214⟩, ⟨18503⟩⟩], orderedFactors := [] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge2079

namespace LeftOperatorMerge1331
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨214⟩, ⟨18508⟩⟩
def group : MergeGroup := .operator 1327 36
def deltas0_0 : Polynomial Owner := [LeftMerge1331.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge1331.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge1331.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge1331.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6410⟩⟩, ⟨.program ⟨214⟩, ⟨18507⟩⟩], orderedFactors := [] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge1331

namespace LeftOperatorMerge540
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨214⟩, ⟨18512⟩⟩
def group : MergeGroup := .operator 536 36
def deltas0_0 : Polynomial Owner := [LeftMerge540.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge540.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge540.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge540.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6410⟩⟩, ⟨.program ⟨214⟩, ⟨18511⟩⟩], orderedFactors := [] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge540

namespace LeftOperatorMerge102349
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨214⟩, ⟨18551⟩⟩
def group : MergeGroup := .operator 94462 102343
def deltas0_0 : Polynomial Owner := [LeftMerge102349.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge102349.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge102349.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge102349.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18548⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge102349

namespace LeftOperatorMerge74138
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨214⟩, ⟨18558⟩⟩
def group : MergeGroup := .operator 65387 74132
def deltas0_0 : Polynomial Owner := [LeftMerge74138.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge74138.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge74138.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge74138.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18555⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge74138

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
