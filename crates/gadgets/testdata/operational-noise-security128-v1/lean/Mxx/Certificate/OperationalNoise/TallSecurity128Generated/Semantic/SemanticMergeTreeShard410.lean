import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard003
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard005
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard007
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard010
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard012
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard014
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard017
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard022
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard024
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard026
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard029
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard031
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard033
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard036
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard038
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard045

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace LeftOperatorMerge12623
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨66239⟩⟩
def group : MergeGroup := .operator 12619 623
def deltas0_0 : Polynomial Owner := [LeftMerge12623.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge12623.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge12623.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge12623.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66238⟩⟩], orderedFactors := [] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge12623

namespace LeftOperatorMerge5891
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨66309⟩⟩
def group : MergeGroup := .operator 5887 623
def deltas0_0 : Polynomial Owner := [LeftMerge5891.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge5891.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge5891.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge5891.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66308⟩⟩], orderedFactors := [] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge5891

namespace LeftOperatorMerge7387
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨66379⟩⟩
def group : MergeGroup := .operator 7383 623
def deltas0_0 : Polynomial Owner := [LeftMerge7387.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge7387.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge7387.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge7387.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66378⟩⟩], orderedFactors := [] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge7387

namespace LeftOperatorMerge11875
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨66449⟩⟩
def group : MergeGroup := .operator 11871 623
def deltas0_0 : Polynomial Owner := [LeftMerge11875.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge11875.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge11875.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge11875.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66448⟩⟩], orderedFactors := [] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge11875

namespace LeftOperatorMerge11127
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨66519⟩⟩
def group : MergeGroup := .operator 11123 623
def deltas0_0 : Polynomial Owner := [LeftMerge11127.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge11127.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge11127.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge11127.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66518⟩⟩], orderedFactors := [] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge11127

namespace LeftOperatorMerge10379
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨66589⟩⟩
def group : MergeGroup := .operator 10375 623
def deltas0_0 : Polynomial Owner := [LeftMerge10379.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge10379.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge10379.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge10379.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66588⟩⟩], orderedFactors := [] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge10379

namespace LeftOperatorMerge5143
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨66659⟩⟩
def group : MergeGroup := .operator 5139 623
def deltas0_0 : Polynomial Owner := [LeftMerge5143.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge5143.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge5143.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge5143.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66658⟩⟩], orderedFactors := [] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge5143

namespace LeftOperatorMerge9631
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨66729⟩⟩
def group : MergeGroup := .operator 9627 623
def deltas0_0 : Polynomial Owner := [LeftMerge9631.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge9631.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge9631.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge9631.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66728⟩⟩], orderedFactors := [] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge9631

namespace LeftOperatorMerge8883
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨66799⟩⟩
def group : MergeGroup := .operator 8879 623
def deltas0_0 : Polynomial Owner := [LeftMerge8883.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge8883.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge8883.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge8883.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66798⟩⟩], orderedFactors := [] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge8883

namespace LeftOperatorMerge8135
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨66869⟩⟩
def group : MergeGroup := .operator 8131 623
def deltas0_0 : Polynomial Owner := [LeftMerge8135.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge8135.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge8135.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge8135.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66868⟩⟩], orderedFactors := [] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge8135

namespace LeftOperatorMerge4395
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨66939⟩⟩
def group : MergeGroup := .operator 4391 623
def deltas0_0 : Polynomial Owner := [LeftMerge4395.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge4395.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge4395.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge4395.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66938⟩⟩], orderedFactors := [] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge4395

namespace LeftOperatorMerge3647
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨67009⟩⟩
def group : MergeGroup := .operator 3643 623
def deltas0_0 : Polynomial Owner := [LeftMerge3647.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge3647.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge3647.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge3647.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨67008⟩⟩], orderedFactors := [] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge3647

namespace LeftOperatorMerge2899
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨67079⟩⟩
def group : MergeGroup := .operator 2895 623
def deltas0_0 : Polynomial Owner := [LeftMerge2899.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge2899.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge2899.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge2899.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨67078⟩⟩], orderedFactors := [] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge2899

namespace LeftOperatorMerge2151
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨67149⟩⟩
def group : MergeGroup := .operator 2147 623
def deltas0_0 : Polynomial Owner := [LeftMerge2151.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge2151.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge2151.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge2151.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨67148⟩⟩], orderedFactors := [] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge2151

namespace LeftOperatorMerge1403
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨67219⟩⟩
def group : MergeGroup := .operator 1399 623
def deltas0_0 : Polynomial Owner := [LeftMerge1403.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge1403.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge1403.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge1403.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨67218⟩⟩], orderedFactors := [] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge1403

namespace LeftOperatorMerge14779
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨67272⟩⟩
def group : MergeGroup := .operator 14775 36
def deltas0_0 : Polynomial Owner := [LeftMerge14779.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge14779.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge14779.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge14779.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67271⟩⟩], orderedFactors := [] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge14779

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
