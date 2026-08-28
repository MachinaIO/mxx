import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard073
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard074
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard075
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard076
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard1083
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard1175
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard1267

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace LeftOperatorMerge207040
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨9483⟩⟩
def group : MergeGroup := .operator 207036 207036
def deltas0_0 : Polynomial Owner := [LeftMerge207040.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge207040.deltaAt
def deltas0_1 : Polynomial Owner := [LeftMerge207041.delta]
theorem rows0_1 : MergeDeltasAt history frameStart owner group deltas0_1 := by
  exact .leaf LeftMerge207041.deltaAt
def deltas1_0 : Polynomial Owner := deltas0_0 ++ deltas0_1
theorem rows1_0 : MergeDeltasAt history frameStart owner group deltas1_0 := by
  exact .append rows0_0 rows0_1
abbrev deltas : Polynomial Owner := deltas1_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows1_0
def left : Polynomial Owner := LeftMerge207040.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge207040.rightRaw.map Term.toExact
def base : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7292⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6727⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }]
def working : Polynomial Owner := []
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (subtract left right) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge207040

namespace LeftOperatorMerge192415
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨9486⟩⟩
def group : MergeGroup := .operator 192411 192411
def deltas0_0 : Polynomial Owner := [LeftMerge192415.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge192415.deltaAt
def deltas0_1 : Polynomial Owner := [LeftMerge192416.delta]
theorem rows0_1 : MergeDeltasAt history frameStart owner group deltas0_1 := by
  exact .leaf LeftMerge192416.deltaAt
def deltas1_0 : Polynomial Owner := deltas0_0 ++ deltas0_1
theorem rows1_0 : MergeDeltasAt history frameStart owner group deltas1_0 := by
  exact .append rows0_0 rows0_1
abbrev deltas : Polynomial Owner := deltas1_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows1_0
def left : Polynomial Owner := LeftMerge192415.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge192415.rightRaw.map Term.toExact
def base : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7292⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6727⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }]
def working : Polynomial Owner := []
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (subtract left right) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge192415

namespace LeftOperatorMerge177790
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨9489⟩⟩
def group : MergeGroup := .operator 177786 177786
def deltas0_0 : Polynomial Owner := [LeftMerge177790.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge177790.deltaAt
def deltas0_1 : Polynomial Owner := [LeftMerge177791.delta]
theorem rows0_1 : MergeDeltasAt history frameStart owner group deltas0_1 := by
  exact .leaf LeftMerge177791.deltaAt
def deltas1_0 : Polynomial Owner := deltas0_0 ++ deltas0_1
theorem rows1_0 : MergeDeltasAt history frameStart owner group deltas1_0 := by
  exact .append rows0_0 rows0_1
abbrev deltas : Polynomial Owner := deltas1_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows1_0
def left : Polynomial Owner := LeftMerge177790.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge177790.rightRaw.map Term.toExact
def base : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7292⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6727⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }]
def working : Polynomial Owner := []
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (subtract left right) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge177790

namespace LeftOperatorMerge15991
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨9585⟩⟩
def group : MergeGroup := .operator 15987 15984
def deltas0_0 : Polynomial Owner := [LeftMerge15991.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge15991.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge15991.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge15991.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7234⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge15991

namespace LeftOperatorMerge16031
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨9586⟩⟩
def group : MergeGroup := .operator 16027 15984
def deltas0_0 : Polynomial Owner := [LeftMerge16031.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge16031.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge16031.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge16031.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7236⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge16031

namespace LeftOperatorMerge16071
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨9587⟩⟩
def group : MergeGroup := .operator 16067 15984
def deltas0_0 : Polynomial Owner := [LeftMerge16071.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge16071.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge16071.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge16071.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7238⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge16071

namespace LeftOperatorMerge16111
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨9588⟩⟩
def group : MergeGroup := .operator 16107 15984
def deltas0_0 : Polynomial Owner := [LeftMerge16111.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge16111.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge16111.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge16111.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7240⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge16111

namespace LeftOperatorMerge16151
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨9589⟩⟩
def group : MergeGroup := .operator 16147 15984
def deltas0_0 : Polynomial Owner := [LeftMerge16151.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge16151.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge16151.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge16151.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7242⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge16151

namespace LeftOperatorMerge16191
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨9590⟩⟩
def group : MergeGroup := .operator 16187 15984
def deltas0_0 : Polynomial Owner := [LeftMerge16191.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge16191.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge16191.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge16191.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7244⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge16191

namespace LeftOperatorMerge16231
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨9591⟩⟩
def group : MergeGroup := .operator 16227 15984
def deltas0_0 : Polynomial Owner := [LeftMerge16231.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge16231.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge16231.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge16231.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7246⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge16231

namespace LeftOperatorMerge16271
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨9592⟩⟩
def group : MergeGroup := .operator 16267 15984
def deltas0_0 : Polynomial Owner := [LeftMerge16271.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge16271.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge16271.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge16271.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7248⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge16271

namespace LeftOperatorMerge16311
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨9593⟩⟩
def group : MergeGroup := .operator 16307 15984
def deltas0_0 : Polynomial Owner := [LeftMerge16311.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge16311.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge16311.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge16311.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7250⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge16311

namespace LeftOperatorMerge16351
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨9594⟩⟩
def group : MergeGroup := .operator 16347 15984
def deltas0_0 : Polynomial Owner := [LeftMerge16351.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge16351.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge16351.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge16351.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7252⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge16351

namespace LeftOperatorMerge16391
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨9595⟩⟩
def group : MergeGroup := .operator 16387 15984
def deltas0_0 : Polynomial Owner := [LeftMerge16391.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge16391.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge16391.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge16391.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7254⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge16391

namespace LeftOperatorMerge16431
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨9596⟩⟩
def group : MergeGroup := .operator 16427 15984
def deltas0_0 : Polynomial Owner := [LeftMerge16431.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge16431.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge16431.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge16431.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7256⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge16431

namespace LeftOperatorMerge16471
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨9597⟩⟩
def group : MergeGroup := .operator 16467 15984
def deltas0_0 : Polynomial Owner := [LeftMerge16471.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge16471.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge16471.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge16471.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7258⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge16471

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
