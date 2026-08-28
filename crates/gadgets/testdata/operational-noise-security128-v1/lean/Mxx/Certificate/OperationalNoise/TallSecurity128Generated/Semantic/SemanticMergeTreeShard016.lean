import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard819
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard822
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard824
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard827
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard830
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard832
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard835
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard853
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard856
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard858
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard861
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard864
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard1734
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard1766
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard1769

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace LeftOperatorMerge137343
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨7804⟩⟩
def group : MergeGroup := .operator 134273 20127
def deltas0_0 : Polynomial Owner := [LeftMerge137343.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge137343.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge137343.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge137343.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7296⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right true false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge137343

namespace LeftOperatorMerge136861
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨7805⟩⟩
def group : MergeGroup := .operator 134273 19626
def deltas0_0 : Polynomial Owner := [LeftMerge136861.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge136861.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge136861.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge136861.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7297⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right true false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge136861

namespace LeftOperatorMerge136379
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨7806⟩⟩
def group : MergeGroup := .operator 134273 19125
def deltas0_0 : Polynomial Owner := [LeftMerge136379.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge136379.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge136379.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge136379.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right true false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge136379

namespace LeftOperatorMerge135897
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨7807⟩⟩
def group : MergeGroup := .operator 134273 18624
def deltas0_0 : Polynomial Owner := [LeftMerge135897.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge135897.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge135897.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge135897.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right true false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge135897

namespace LeftOperatorMerge135415
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨7808⟩⟩
def group : MergeGroup := .operator 134273 18123
def deltas0_0 : Polynomial Owner := [LeftMerge135415.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge135415.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge135415.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge135415.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7300⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right true false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge135415

namespace LeftOperatorMerge134933
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨7809⟩⟩
def group : MergeGroup := .operator 134273 17622
def deltas0_0 : Polynomial Owner := [LeftMerge134933.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge134933.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge134933.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge134933.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right true false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge134933

namespace LeftOperatorMerge134440
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨7810⟩⟩
def group : MergeGroup := .operator 134273 17106
def deltas0_0 : Polynomial Owner := [LeftMerge134440.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge134440.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge134440.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge134440.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right true false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge134440

namespace LeftOperatorMerge142645
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨7811⟩⟩
def group : MergeGroup := .operator 134273 25638
def deltas0_0 : Polynomial Owner := [LeftMerge142645.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge142645.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge142645.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge142645.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7303⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right true false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge142645

namespace LeftOperatorMerge142617
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨7812⟩⟩
def group : MergeGroup := .operator 134273 25597
def deltas0_0 : Polynomial Owner := [LeftMerge142617.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge142617.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge142617.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge142617.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7304⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right true false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge142617

namespace LeftOperatorMerge142135
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨7813⟩⟩
def group : MergeGroup := .operator 134273 25096
def deltas0_0 : Polynomial Owner := [LeftMerge142135.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge142135.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge142135.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge142135.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7305⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right true false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge142135

namespace LeftOperatorMerge141653
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨7814⟩⟩
def group : MergeGroup := .operator 134273 24595
def deltas0_0 : Polynomial Owner := [LeftMerge141653.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge141653.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge141653.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge141653.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7306⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right true false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge141653

namespace LeftOperatorMerge141171
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨7815⟩⟩
def group : MergeGroup := .operator 134273 24094
def deltas0_0 : Polynomial Owner := [LeftMerge141171.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge141171.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge141171.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge141171.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7307⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right true false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge141171

namespace LeftOperatorMerge140689
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨7816⟩⟩
def group : MergeGroup := .operator 134273 23593
def deltas0_0 : Polynomial Owner := [LeftMerge140689.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge140689.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge140689.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge140689.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7308⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right true false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge140689

namespace LeftOperatorMerge280527
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨7893⟩⟩
def group : MergeGroup := .operator 280523 16697
def deltas0_0 : Polynomial Owner := [LeftMerge280527.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge280527.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge280527.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge280527.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7271⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right true false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge280527

namespace LeftOperatorMerge286433
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨7894⟩⟩
def group : MergeGroup := .operator 280523 23092
def deltas0_0 : Polynomial Owner := [LeftMerge286433.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge286433.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge286433.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge286433.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7272⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right true false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge286433

namespace LeftOperatorMerge285953
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨257⟩, ⟨7895⟩⟩
def group : MergeGroup := .operator 280523 22591
def deltas0_0 : Polynomial Owner := [LeftMerge285953.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge285953.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge285953.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge285953.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7273⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right true false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge285953

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
