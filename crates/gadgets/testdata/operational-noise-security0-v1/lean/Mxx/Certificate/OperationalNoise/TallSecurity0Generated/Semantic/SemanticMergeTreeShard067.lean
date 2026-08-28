import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeDeltaShard000
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeDeltaShard002
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeDeltaShard003
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeDeltaShard005
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeDeltaShard007
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeDeltaShard010
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeDeltaShard012
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeDeltaShard014

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace LeftOperatorMerge3623
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨214⟩, ⟨17547⟩⟩
def group : MergeGroup := .operator 3619 593
def deltas0_0 : Polynomial Owner := [LeftMerge3623.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge3623.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge3623.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge3623.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6473⟩⟩, ⟨.program ⟨214⟩, ⟨17546⟩⟩], orderedFactors := [] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge3623

namespace LeftOperatorMerge4365
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨214⟩, ⟨17551⟩⟩
def group : MergeGroup := .operator 4361 593
def deltas0_0 : Polynomial Owner := [LeftMerge4365.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge4365.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge4365.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge4365.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6473⟩⟩, ⟨.program ⟨214⟩, ⟨17550⟩⟩], orderedFactors := [] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge4365

namespace LeftOperatorMerge2875
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨214⟩, ⟨17555⟩⟩
def group : MergeGroup := .operator 2871 593
def deltas0_0 : Polynomial Owner := [LeftMerge2875.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge2875.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge2875.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge2875.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6473⟩⟩, ⟨.program ⟨214⟩, ⟨17554⟩⟩], orderedFactors := [] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge2875

namespace LeftOperatorMerge2127
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨214⟩, ⟨17559⟩⟩
def group : MergeGroup := .operator 2123 593
def deltas0_0 : Polynomial Owner := [LeftMerge2127.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge2127.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge2127.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge2127.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6473⟩⟩, ⟨.program ⟨214⟩, ⟨17558⟩⟩], orderedFactors := [] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge2127

namespace LeftOperatorMerge1379
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨214⟩, ⟨17563⟩⟩
def group : MergeGroup := .operator 1375 593
def deltas0_0 : Polynomial Owner := [LeftMerge1379.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge1379.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge1379.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge1379.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6473⟩⟩, ⟨.program ⟨214⟩, ⟨17562⟩⟩], orderedFactors := [] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge1379

namespace LeftOperatorMerge600
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨214⟩, ⟨17567⟩⟩
def group : MergeGroup := .operator 596 593
def deltas0_0 : Polynomial Owner := [LeftMerge600.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge600.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge600.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge600.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6473⟩⟩, ⟨.program ⟨214⟩, ⟨17566⟩⟩], orderedFactors := [] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge600

namespace LeftOperatorMerge5119
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨214⟩, ⟨17597⟩⟩
def group : MergeGroup := .operator 5115 613
def deltas0_0 : Polynomial Owner := [LeftMerge5119.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge5119.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge5119.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge5119.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6494⟩⟩, ⟨.program ⟨214⟩, ⟨17596⟩⟩], orderedFactors := [] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge5119

namespace LeftOperatorMerge3639
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨214⟩, ⟨17603⟩⟩
def group : MergeGroup := .operator 3635 613
def deltas0_0 : Polynomial Owner := [LeftMerge3639.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge3639.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge3639.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge3639.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6494⟩⟩, ⟨.program ⟨214⟩, ⟨17602⟩⟩], orderedFactors := [] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge3639

namespace LeftOperatorMerge4381
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨214⟩, ⟨17607⟩⟩
def group : MergeGroup := .operator 4377 613
def deltas0_0 : Polynomial Owner := [LeftMerge4381.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge4381.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge4381.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge4381.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6494⟩⟩, ⟨.program ⟨214⟩, ⟨17606⟩⟩], orderedFactors := [] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge4381

namespace LeftOperatorMerge2891
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨214⟩, ⟨17611⟩⟩
def group : MergeGroup := .operator 2887 613
def deltas0_0 : Polynomial Owner := [LeftMerge2891.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge2891.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge2891.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge2891.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6494⟩⟩, ⟨.program ⟨214⟩, ⟨17610⟩⟩], orderedFactors := [] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge2891

namespace LeftOperatorMerge2143
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨214⟩, ⟨17615⟩⟩
def group : MergeGroup := .operator 2139 613
def deltas0_0 : Polynomial Owner := [LeftMerge2143.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge2143.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge2143.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge2143.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6494⟩⟩, ⟨.program ⟨214⟩, ⟨17614⟩⟩], orderedFactors := [] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge2143

namespace LeftOperatorMerge1395
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨214⟩, ⟨17619⟩⟩
def group : MergeGroup := .operator 1391 613
def deltas0_0 : Polynomial Owner := [LeftMerge1395.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge1395.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge1395.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge1395.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6494⟩⟩, ⟨.program ⟨214⟩, ⟨17618⟩⟩], orderedFactors := [] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge1395

namespace LeftOperatorMerge620
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨214⟩, ⟨17623⟩⟩
def group : MergeGroup := .operator 616 613
def deltas0_0 : Polynomial Owner := [LeftMerge620.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge620.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge620.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge620.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6494⟩⟩, ⟨.program ⟨214⟩, ⟨17622⟩⟩], orderedFactors := [] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge620

namespace LeftOperatorMerge5127
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨214⟩, ⟨17653⟩⟩
def group : MergeGroup := .operator 5123 623
def deltas0_0 : Polynomial Owner := [LeftMerge5127.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge5127.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge5127.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge5127.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6502⟩⟩, ⟨.program ⟨214⟩, ⟨17652⟩⟩], orderedFactors := [] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge5127

namespace LeftOperatorMerge3647
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨214⟩, ⟨17659⟩⟩
def group : MergeGroup := .operator 3643 623
def deltas0_0 : Polynomial Owner := [LeftMerge3647.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge3647.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge3647.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge3647.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6502⟩⟩, ⟨.program ⟨214⟩, ⟨17658⟩⟩], orderedFactors := [] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge3647

namespace LeftOperatorMerge4389
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨214⟩, ⟨17663⟩⟩
def group : MergeGroup := .operator 4385 623
def deltas0_0 : Polynomial Owner := [LeftMerge4389.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge4389.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge4389.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge4389.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6502⟩⟩, ⟨.program ⟨214⟩, ⟨17662⟩⟩], orderedFactors := [] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge4389

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
