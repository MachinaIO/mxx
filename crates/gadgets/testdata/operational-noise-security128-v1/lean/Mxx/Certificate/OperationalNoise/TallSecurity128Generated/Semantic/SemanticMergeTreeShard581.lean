import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard1124
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard1125
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard1127
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard1128
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard1129
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard1130

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace LeftOperatorMerge183819
def frameStart : Nat := 183727
def owner : Owner := ⟨.program ⟨257⟩, ⟨58515⟩⟩
def group : MergeGroup := .operator 183815 183772
def deltas0_0 : Polynomial Owner := [LeftMerge183819.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge183819.deltaAt
def deltas0_1 : Polynomial Owner := [LeftMerge183820.delta]
theorem rows0_1 : MergeDeltasAt history frameStart owner group deltas0_1 := by
  exact .leaf LeftMerge183820.deltaAt
def deltas1_0 : Polynomial Owner := deltas0_0 ++ deltas0_1
theorem rows1_0 : MergeDeltasAt history frameStart owner group deltas1_0 := by
  exact .append rows0_0 rows0_1
abbrev deltas : Polynomial Owner := deltas1_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows1_0
def left : Polynomial Owner := LeftMerge183819.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge183819.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩, ⟨.program ⟨257⟩, ⟨58512⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨25046⟩⟩, ⟨.program ⟨257⟩, ⟨56586⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨58512⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge183819

namespace LeftOperatorMerge184025
def frameStart : Nat := 183936
def owner : Owner := ⟨.program ⟨257⟩, ⟨57180⟩⟩
def group : MergeGroup := .operator 183998 184021
def deltas0_0 : Polynomial Owner := [LeftMerge184025.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge184025.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge184025.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge184025.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨57178⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge184025

namespace LeftOperatorMerge184002
def frameStart : Nat := 183936
def owner : Owner := ⟨.program ⟨257⟩, ⟨58340⟩⟩
def group : MergeGroup := .operator 183998 183996
def deltas0_0 : Polynomial Owner := [LeftMerge184002.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge184002.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge184002.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge184002.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨56872⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge184002

namespace LeftOperatorMerge184014
def frameStart : Nat := 183936
def owner : Owner := ⟨.program ⟨257⟩, ⟨59006⟩⟩
def group : MergeGroup := .operator 184010 183987
def deltas0_0 : Polynomial Owner := [LeftMerge184014.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge184014.deltaAt
def deltas0_1 : Polynomial Owner := [LeftMerge184015.delta]
theorem rows0_1 : MergeDeltasAt history frameStart owner group deltas0_1 := by
  exact .leaf LeftMerge184015.deltaAt
def deltas1_0 : Polynomial Owner := deltas0_0 ++ deltas0_1
theorem rows1_0 : MergeDeltasAt history frameStart owner group deltas1_0 := by
  exact .append rows0_0 rows0_1
abbrev deltas : Polynomial Owner := deltas1_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows1_0
def left : Polynomial Owner := LeftMerge184014.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge184014.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7185⟩⟩, ⟨.program ⟨257⟩, ⟨59005⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨56872⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨59005⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge184014

namespace LeftOperatorMerge184292
def frameStart : Nat := 184209
def owner : Owner := ⟨.program ⟨257⟩, ⟨9531⟩⟩
def group : MergeGroup := .operator 184288 184285
def deltas0_0 : Polynomial Owner := [LeftMerge184292.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge184292.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge184292.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge184292.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge184292

namespace LeftOperatorMerge184239
def frameStart : Nat := 184209
def owner : Owner := ⟨.program ⟨257⟩, ⟨53607⟩⟩
def group : MergeGroup := .operator 184235 184232
def deltas0_0 : Polynomial Owner := [LeftMerge184239.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge184239.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge184239.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge184239.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24806⟩⟩, ⟨.program ⟨257⟩, ⟨53606⟩⟩], orderedFactors := [] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge184239

namespace LeftOperatorMerge184312
def frameStart : Nat := 184209
def owner : Owner := ⟨.program ⟨257⟩, ⟨53894⟩⟩
def group : MergeGroup := .operator 184265 184308
def deltas0_0 : Polynomial Owner := [LeftMerge184312.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge184312.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge184312.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge184312.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨53892⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge184312

namespace LeftOperatorMerge184269
def frameStart : Nat := 184209
def owner : Owner := ⟨.program ⟨257⟩, ⟨55280⟩⟩
def group : MergeGroup := .operator 184265 184263
def deltas0_0 : Polynomial Owner := [LeftMerge184269.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge184269.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge184269.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge184269.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24806⟩⟩, ⟨.program ⟨257⟩, ⟨53606⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge184269

namespace LeftOperatorMerge184301
def frameStart : Nat := 184209
def owner : Owner := ⟨.program ⟨257⟩, ⟨55535⟩⟩
def group : MergeGroup := .operator 184297 184254
def deltas0_0 : Polynomial Owner := [LeftMerge184301.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge184301.deltaAt
def deltas0_1 : Polynomial Owner := [LeftMerge184302.delta]
theorem rows0_1 : MergeDeltasAt history frameStart owner group deltas0_1 := by
  exact .leaf LeftMerge184302.deltaAt
def deltas1_0 : Polynomial Owner := deltas0_0 ++ deltas0_1
theorem rows1_0 : MergeDeltasAt history frameStart owner group deltas1_0 := by
  exact .append rows0_0 rows0_1
abbrev deltas : Polynomial Owner := deltas1_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows1_0
def left : Polynomial Owner := LeftMerge184301.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge184301.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55532⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24806⟩⟩, ⟨.program ⟨257⟩, ⟨53606⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨55532⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge184301

namespace LeftOperatorMerge184507
def frameStart : Nat := 184418
def owner : Owner := ⟨.program ⟨257⟩, ⟨54200⟩⟩
def group : MergeGroup := .operator 184480 184503
def deltas0_0 : Polynomial Owner := [LeftMerge184507.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge184507.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge184507.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge184507.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨54198⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge184507

namespace LeftOperatorMerge184484
def frameStart : Nat := 184418
def owner : Owner := ⟨.program ⟨257⟩, ⟨55360⟩⟩
def group : MergeGroup := .operator 184480 184478
def deltas0_0 : Polynomial Owner := [LeftMerge184484.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge184484.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge184484.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge184484.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨53892⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge184484

namespace LeftOperatorMerge184496
def frameStart : Nat := 184418
def owner : Owner := ⟨.program ⟨257⟩, ⟨56026⟩⟩
def group : MergeGroup := .operator 184492 184469
def deltas0_0 : Polynomial Owner := [LeftMerge184496.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge184496.deltaAt
def deltas0_1 : Polynomial Owner := [LeftMerge184497.delta]
theorem rows0_1 : MergeDeltasAt history frameStart owner group deltas0_1 := by
  exact .leaf LeftMerge184497.deltaAt
def deltas1_0 : Polynomial Owner := deltas0_0 ++ deltas0_1
theorem rows1_0 : MergeDeltasAt history frameStart owner group deltas1_0 := by
  exact .append rows0_0 rows0_1
abbrev deltas : Polynomial Owner := deltas1_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows1_0
def left : Polynomial Owner := LeftMerge184496.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge184496.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨56025⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨53892⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨56025⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge184496

namespace LeftOperatorMerge184774
def frameStart : Nat := 184691
def owner : Owner := ⟨.program ⟨257⟩, ⟨9582⟩⟩
def group : MergeGroup := .operator 184770 184767
def deltas0_0 : Polynomial Owner := [LeftMerge184774.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge184774.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge184774.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge184774.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge184774

namespace LeftOperatorMerge184721
def frameStart : Nat := 184691
def owner : Owner := ⟨.program ⟨257⟩, ⟨50627⟩⟩
def group : MergeGroup := .operator 184717 184714
def deltas0_0 : Polynomial Owner := [LeftMerge184721.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge184721.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge184721.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge184721.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24566⟩⟩, ⟨.program ⟨257⟩, ⟨50626⟩⟩], orderedFactors := [] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge184721

namespace LeftOperatorMerge184794
def frameStart : Nat := 184691
def owner : Owner := ⟨.program ⟨257⟩, ⟨50914⟩⟩
def group : MergeGroup := .operator 184747 184790
def deltas0_0 : Polynomial Owner := [LeftMerge184794.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge184794.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge184794.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge184794.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨50912⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge184794

namespace LeftOperatorMerge184751
def frameStart : Nat := 184691
def owner : Owner := ⟨.program ⟨257⟩, ⟨52300⟩⟩
def group : MergeGroup := .operator 184747 184745
def deltas0_0 : Polynomial Owner := [LeftMerge184751.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge184751.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge184751.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge184751.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24566⟩⟩, ⟨.program ⟨257⟩, ⟨50626⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge184751

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
