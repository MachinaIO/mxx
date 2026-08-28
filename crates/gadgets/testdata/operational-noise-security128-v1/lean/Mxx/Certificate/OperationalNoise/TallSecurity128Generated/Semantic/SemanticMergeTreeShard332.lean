import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard667
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard1126
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard1218
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard1310
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard1401
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard1402

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace LeftOperatorMerge227980
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨53502⟩⟩
def group : MergeGroup := .operator 10845 222153
def deltas0_0 : Polynomial Owner := [LeftMerge227980.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge227980.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge227980.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge227980.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨53498⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right true false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge227980

namespace LeftOperatorMerge228002
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨53505⟩⟩
def group : MergeGroup := .operator 227996 23122
def deltas0_0 : Polynomial Owner := [LeftMerge228002.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge228002.deltaAt
def deltas0_1 : Polynomial Owner := [LeftMerge228005.delta]
theorem rows0_1 : MergeDeltasAt history frameStart owner group deltas0_1 := by
  exact .leaf LeftMerge228005.deltaAt
def deltas1_0 : Polynomial Owner := deltas0_0 ++ deltas0_1
theorem rows1_0 : MergeDeltasAt history frameStart owner group deltas1_0 := by
  exact .append rows0_0 rows0_1
abbrev deltas : Polynomial Owner := deltas1_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows1_0
def left : Polynomial Owner := LeftMerge228002.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge228002.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨53498⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge228002

namespace LeftOperatorMerge228010
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨53506⟩⟩
def group : MergeGroup := .operator 228006 227976
def deltas0_0 : Polynomial Owner := [LeftMerge228010.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge228010.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge228010.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge228010.rightRaw.map Term.toExact
def base : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨24758⟩⟩, ⟨.program ⟨257⟩, ⟨53498⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨53498⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7272⟩⟩] } }]
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨24758⟩⟩, ⟨.program ⟨257⟩, ⟨53498⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (add left right) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge228010

namespace LeftOperatorMerge213349
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨53528⟩⟩
def group : MergeGroup := .operator 213343 10097
def deltas0_0 : Polynomial Owner := [LeftMerge213349.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge213349.deltaAt
def deltas0_1 : Polynomial Owner := [LeftMerge213350.delta]
theorem rows0_1 : MergeDeltasAt history frameStart owner group deltas0_1 := by
  exact .leaf LeftMerge213350.deltaAt
def deltas1_0 : Polynomial Owner := deltas0_0 ++ deltas0_1
theorem rows1_0 : MergeDeltasAt history frameStart owner group deltas1_0 := by
  exact .append rows0_0 rows0_1
abbrev deltas : Polynomial Owner := deltas1_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows1_0
def left : Polynomial Owner := LeftMerge213349.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge213349.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨24770⟩⟩, ⟨.program ⟨257⟩, ⟨53525⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨53525⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7272⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge213349

namespace LeftOperatorMerge213355
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨53529⟩⟩
def group : MergeGroup := .operator 10097 207528
def deltas0_0 : Polynomial Owner := [LeftMerge213355.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge213355.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge213355.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge213355.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨53525⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right true false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge213355

namespace LeftOperatorMerge213377
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨53532⟩⟩
def group : MergeGroup := .operator 213371 23122
def deltas0_0 : Polynomial Owner := [LeftMerge213377.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge213377.deltaAt
def deltas0_1 : Polynomial Owner := [LeftMerge213380.delta]
theorem rows0_1 : MergeDeltasAt history frameStart owner group deltas0_1 := by
  exact .leaf LeftMerge213380.deltaAt
def deltas1_0 : Polynomial Owner := deltas0_0 ++ deltas0_1
theorem rows1_0 : MergeDeltasAt history frameStart owner group deltas1_0 := by
  exact .append rows0_0 rows0_1
abbrev deltas : Polynomial Owner := deltas1_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows1_0
def left : Polynomial Owner := LeftMerge213377.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge213377.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨53525⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge213377

namespace LeftOperatorMerge213385
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨53533⟩⟩
def group : MergeGroup := .operator 213381 213351
def deltas0_0 : Polynomial Owner := [LeftMerge213385.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge213385.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge213385.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge213385.rightRaw.map Term.toExact
def base : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨24770⟩⟩, ⟨.program ⟨257⟩, ⟨53525⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨53525⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7272⟩⟩] } }]
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨24770⟩⟩, ⟨.program ⟨257⟩, ⟨53525⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (add left right) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge213385

namespace LeftOperatorMerge110974
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨53555⟩⟩
def group : MergeGroup := .operator 110968 4861
def deltas0_0 : Polynomial Owner := [LeftMerge110974.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge110974.deltaAt
def deltas0_1 : Polynomial Owner := [LeftMerge110975.delta]
theorem rows0_1 : MergeDeltasAt history frameStart owner group deltas0_1 := by
  exact .leaf LeftMerge110975.deltaAt
def deltas1_0 : Polynomial Owner := deltas0_0 ++ deltas0_1
theorem rows1_0 : MergeDeltasAt history frameStart owner group deltas1_0 := by
  exact .append rows0_0 rows0_1
abbrev deltas : Polynomial Owner := deltas1_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows1_0
def left : Polynomial Owner := LeftMerge110974.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge110974.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨24782⟩⟩, ⟨.program ⟨257⟩, ⟨53552⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨53552⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7272⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge110974

namespace LeftOperatorMerge110980
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨53556⟩⟩
def group : MergeGroup := .operator 4861 105153
def deltas0_0 : Polynomial Owner := [LeftMerge110980.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge110980.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge110980.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge110980.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨53552⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right true false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge110980

namespace LeftOperatorMerge111002
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨53559⟩⟩
def group : MergeGroup := .operator 110996 23122
def deltas0_0 : Polynomial Owner := [LeftMerge111002.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge111002.deltaAt
def deltas0_1 : Polynomial Owner := [LeftMerge111005.delta]
theorem rows0_1 : MergeDeltasAt history frameStart owner group deltas0_1 := by
  exact .leaf LeftMerge111005.deltaAt
def deltas1_0 : Polynomial Owner := deltas0_0 ++ deltas0_1
theorem rows1_0 : MergeDeltasAt history frameStart owner group deltas1_0 := by
  exact .append rows0_0 rows0_1
abbrev deltas : Polynomial Owner := deltas1_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows1_0
def left : Polynomial Owner := LeftMerge111002.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge111002.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨53552⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge111002

namespace LeftOperatorMerge111010
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨53560⟩⟩
def group : MergeGroup := .operator 111006 110976
def deltas0_0 : Polynomial Owner := [LeftMerge111010.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge111010.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge111010.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge111010.rightRaw.map Term.toExact
def base : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨24782⟩⟩, ⟨.program ⟨257⟩, ⟨53552⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨53552⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7272⟩⟩] } }]
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨24782⟩⟩, ⟨.program ⟨257⟩, ⟨53552⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (add left right) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge111010

namespace LeftOperatorMerge198724
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨53582⟩⟩
def group : MergeGroup := .operator 198718 9349
def deltas0_0 : Polynomial Owner := [LeftMerge198724.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge198724.deltaAt
def deltas0_1 : Polynomial Owner := [LeftMerge198725.delta]
theorem rows0_1 : MergeDeltasAt history frameStart owner group deltas0_1 := by
  exact .leaf LeftMerge198725.deltaAt
def deltas1_0 : Polynomial Owner := deltas0_0 ++ deltas0_1
theorem rows1_0 : MergeDeltasAt history frameStart owner group deltas1_0 := by
  exact .append rows0_0 rows0_1
abbrev deltas : Polynomial Owner := deltas1_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows1_0
def left : Polynomial Owner := LeftMerge198724.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge198724.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨24794⟩⟩, ⟨.program ⟨257⟩, ⟨53579⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨53579⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7272⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge198724

namespace LeftOperatorMerge198730
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨53583⟩⟩
def group : MergeGroup := .operator 9349 192903
def deltas0_0 : Polynomial Owner := [LeftMerge198730.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge198730.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge198730.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge198730.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨53579⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right true false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge198730

namespace LeftOperatorMerge198752
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨53586⟩⟩
def group : MergeGroup := .operator 198746 23122
def deltas0_0 : Polynomial Owner := [LeftMerge198752.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge198752.deltaAt
def deltas0_1 : Polynomial Owner := [LeftMerge198755.delta]
theorem rows0_1 : MergeDeltasAt history frameStart owner group deltas0_1 := by
  exact .leaf LeftMerge198755.deltaAt
def deltas1_0 : Polynomial Owner := deltas0_0 ++ deltas0_1
theorem rows1_0 : MergeDeltasAt history frameStart owner group deltas1_0 := by
  exact .append rows0_0 rows0_1
abbrev deltas : Polynomial Owner := deltas1_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows1_0
def left : Polynomial Owner := LeftMerge198752.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge198752.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨53579⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge198752

namespace LeftOperatorMerge198760
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨53587⟩⟩
def group : MergeGroup := .operator 198756 198726
def deltas0_0 : Polynomial Owner := [LeftMerge198760.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge198760.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge198760.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge198760.rightRaw.map Term.toExact
def base : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨24794⟩⟩, ⟨.program ⟨257⟩, ⟨53579⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨53579⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7272⟩⟩] } }]
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨24794⟩⟩, ⟨.program ⟨257⟩, ⟨53579⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (add left right) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge198760

namespace LeftOperatorMerge184099
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨53609⟩⟩
def group : MergeGroup := .operator 184093 8601
def deltas0_0 : Polynomial Owner := [LeftMerge184099.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge184099.deltaAt
def deltas0_1 : Polynomial Owner := [LeftMerge184100.delta]
theorem rows0_1 : MergeDeltasAt history frameStart owner group deltas0_1 := by
  exact .leaf LeftMerge184100.deltaAt
def deltas1_0 : Polynomial Owner := deltas0_0 ++ deltas0_1
theorem rows1_0 : MergeDeltasAt history frameStart owner group deltas1_0 := by
  exact .append rows0_0 rows0_1
abbrev deltas : Polynomial Owner := deltas1_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows1_0
def left : Polynomial Owner := LeftMerge184099.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge184099.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨24806⟩⟩, ⟨.program ⟨257⟩, ⟨53606⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨53606⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7272⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge184099

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
