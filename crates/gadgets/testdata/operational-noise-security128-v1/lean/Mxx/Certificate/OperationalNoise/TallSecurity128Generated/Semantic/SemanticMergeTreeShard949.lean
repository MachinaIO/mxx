import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard242
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard243
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard245
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard246
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard247
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard248
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard249
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard250
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard251
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard252
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard254
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard255
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard256
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard269
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard270
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard272

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace LeftRelationMerge43550
def frameStart : Nat := 43470
def owner : Owner := ⟨.program ⟨257⟩, ⟨36849⟩⟩
def group : MergeGroup := .relation 43550
def deltas0_0 : Polynomial Owner := [LeftMerge43551.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge43551.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def accumulator : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7191⟩⟩, ⟨.program ⟨257⟩, ⟨36848⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨34820⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36848⟩⟩] } }]
def source : MonomialKey Owner := ⟨[⟨.program ⟨257⟩, ⟨34820⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36848⟩⟩]⟩
def rhs : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨35981⟩⟩] } }]
def base : Polynomial Owner := subtract accumulator [{ coefficient := (-1), key := source }]
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7191⟩⟩, ⟨.program ⟨257⟩, ⟨36848⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨34820⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35981⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem relationAgreement : CanonicalAgreement (add base reconstruction.deltas)
    (relationPoly accumulator source
      (relationContext source source.centralFactors 0 2) (-1) rhs) := by
  dsimp [reconstruction]
  decide +kernel
end LeftRelationMerge43550

namespace LeftRelationMerge43762
def frameStart : Nat := 43682
def owner : Owner := ⟨.program ⟨257⟩, ⟨31189⟩⟩
def group : MergeGroup := .relation 43762
def deltas0_0 : Polynomial Owner := [LeftMerge43763.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge43763.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def accumulator : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨31188⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨29160⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨31188⟩⟩] } }]
def source : MonomialKey Owner := ⟨[⟨.program ⟨257⟩, ⟨29160⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨31188⟩⟩]⟩
def rhs : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨30321⟩⟩] } }]
def base : Polynomial Owner := subtract accumulator [{ coefficient := (-1), key := source }]
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨31188⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨29160⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨30321⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem relationAgreement : CanonicalAgreement (add base reconstruction.deltas)
    (relationPoly accumulator source
      (relationContext source source.centralFactors 0 2) (-1) rhs) := by
  dsimp [reconstruction]
  decide +kernel
end LeftRelationMerge43762

namespace LeftRelationMerge43974
def frameStart : Nat := 43894
def owner : Owner := ⟨.program ⟨257⟩, ⟨28509⟩⟩
def group : MergeGroup := .relation 43974
def deltas0_0 : Polynomial Owner := [LeftMerge43975.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge43975.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def accumulator : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7189⟩⟩, ⟨.program ⟨257⟩, ⟨28508⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨26480⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨28508⟩⟩] } }]
def source : MonomialKey Owner := ⟨[⟨.program ⟨257⟩, ⟨26480⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨28508⟩⟩]⟩
def rhs : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨27641⟩⟩] } }]
def base : Polynomial Owner := subtract accumulator [{ coefficient := (-1), key := source }]
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7189⟩⟩, ⟨.program ⟨257⟩, ⟨28508⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨26480⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨27641⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem relationAgreement : CanonicalAgreement (add base reconstruction.deltas)
    (relationPoly accumulator source
      (relationContext source source.centralFactors 0 2) (-1) rhs) := by
  dsimp [reconstruction]
  decide +kernel
end LeftRelationMerge43974

namespace LeftRelationMerge44186
def frameStart : Nat := 44106
def owner : Owner := ⟨.program ⟨257⟩, ⟨70874⟩⟩
def group : MergeGroup := .relation 44186
def deltas0_0 : Polynomial Owner := [LeftMerge44187.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge44187.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def accumulator : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7188⟩⟩, ⟨.program ⟨257⟩, ⟨70873⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨65860⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70873⟩⟩] } }]
def source : MonomialKey Owner := ⟨[⟨.program ⟨257⟩, ⟨65860⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70873⟩⟩]⟩
def rhs : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68762⟩⟩] } }]
def base : Polynomial Owner := subtract accumulator [{ coefficient := (-1), key := source }]
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7188⟩⟩, ⟨.program ⟨257⟩, ⟨70873⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨65860⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68762⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem relationAgreement : CanonicalAgreement (add base reconstruction.deltas)
    (relationPoly accumulator source
      (relationContext source source.centralFactors 0 2) (-1) rhs) := by
  dsimp [reconstruction]
  decide +kernel
end LeftRelationMerge44186

namespace LeftRelationMerge44398
def frameStart : Nat := 44318
def owner : Owner := ⟨.program ⟨257⟩, ⟨65145⟩⟩
def group : MergeGroup := .relation 44398
def deltas0_0 : Polynomial Owner := [LeftMerge44399.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge44399.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def accumulator : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7187⟩⟩, ⟨.program ⟨257⟩, ⟨65144⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨62880⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨65144⟩⟩] } }]
def source : MonomialKey Owner := ⟨[⟨.program ⟨257⟩, ⟨62880⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨65144⟩⟩]⟩
def rhs : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨64161⟩⟩] } }]
def base : Polynomial Owner := subtract accumulator [{ coefficient := (-1), key := source }]
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7187⟩⟩, ⟨.program ⟨257⟩, ⟨65144⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨62880⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨64161⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem relationAgreement : CanonicalAgreement (add base reconstruction.deltas)
    (relationPoly accumulator source
      (relationContext source source.centralFactors 0 2) (-1) rhs) := by
  dsimp [reconstruction]
  decide +kernel
end LeftRelationMerge44398

namespace LeftRelationMerge44610
def frameStart : Nat := 44530
def owner : Owner := ⟨.program ⟨257⟩, ⟨62165⟩⟩
def group : MergeGroup := .relation 44610
def deltas0_0 : Polynomial Owner := [LeftMerge44611.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge44611.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def accumulator : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7186⟩⟩, ⟨.program ⟨257⟩, ⟨62164⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨59900⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨62164⟩⟩] } }]
def source : MonomialKey Owner := ⟨[⟨.program ⟨257⟩, ⟨59900⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨62164⟩⟩]⟩
def rhs : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨61181⟩⟩] } }]
def base : Polynomial Owner := subtract accumulator [{ coefficient := (-1), key := source }]
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7186⟩⟩, ⟨.program ⟨257⟩, ⟨62164⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨59900⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨61181⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem relationAgreement : CanonicalAgreement (add base reconstruction.deltas)
    (relationPoly accumulator source
      (relationContext source source.centralFactors 0 2) (-1) rhs) := by
  dsimp [reconstruction]
  decide +kernel
end LeftRelationMerge44610

namespace LeftRelationMerge44822
def frameStart : Nat := 44742
def owner : Owner := ⟨.program ⟨257⟩, ⟨59185⟩⟩
def group : MergeGroup := .relation 44822
def deltas0_0 : Polynomial Owner := [LeftMerge44823.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge44823.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def accumulator : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7185⟩⟩, ⟨.program ⟨257⟩, ⟨59184⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨56920⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨59184⟩⟩] } }]
def source : MonomialKey Owner := ⟨[⟨.program ⟨257⟩, ⟨56920⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨59184⟩⟩]⟩
def rhs : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨58201⟩⟩] } }]
def base : Polynomial Owner := subtract accumulator [{ coefficient := (-1), key := source }]
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7185⟩⟩, ⟨.program ⟨257⟩, ⟨59184⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨56920⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨58201⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem relationAgreement : CanonicalAgreement (add base reconstruction.deltas)
    (relationPoly accumulator source
      (relationContext source source.centralFactors 0 2) (-1) rhs) := by
  dsimp [reconstruction]
  decide +kernel
end LeftRelationMerge44822

namespace LeftRelationMerge45034
def frameStart : Nat := 44954
def owner : Owner := ⟨.program ⟨257⟩, ⟨56205⟩⟩
def group : MergeGroup := .relation 45034
def deltas0_0 : Polynomial Owner := [LeftMerge45035.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge45035.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def accumulator : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨56204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨53940⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨56204⟩⟩] } }]
def source : MonomialKey Owner := ⟨[⟨.program ⟨257⟩, ⟨53940⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨56204⟩⟩]⟩
def rhs : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨55221⟩⟩] } }]
def base : Polynomial Owner := subtract accumulator [{ coefficient := (-1), key := source }]
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨56204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨53940⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨55221⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem relationAgreement : CanonicalAgreement (add base reconstruction.deltas)
    (relationPoly accumulator source
      (relationContext source source.centralFactors 0 2) (-1) rhs) := by
  dsimp [reconstruction]
  decide +kernel
end LeftRelationMerge45034

namespace LeftRelationMerge45246
def frameStart : Nat := 45166
def owner : Owner := ⟨.program ⟨257⟩, ⟨53225⟩⟩
def group : MergeGroup := .relation 45246
def deltas0_0 : Polynomial Owner := [LeftMerge45247.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge45247.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def accumulator : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨53224⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨50960⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨53224⟩⟩] } }]
def source : MonomialKey Owner := ⟨[⟨.program ⟨257⟩, ⟨50960⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨53224⟩⟩]⟩
def rhs : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨52241⟩⟩] } }]
def base : Polynomial Owner := subtract accumulator [{ coefficient := (-1), key := source }]
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨53224⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨50960⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨52241⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem relationAgreement : CanonicalAgreement (add base reconstruction.deltas)
    (relationPoly accumulator source
      (relationContext source source.centralFactors 0 2) (-1) rhs) := by
  dsimp [reconstruction]
  decide +kernel
end LeftRelationMerge45246

namespace LeftRelationMerge45458
def frameStart : Nat := 45378
def owner : Owner := ⟨.program ⟨257⟩, ⟨34165⟩⟩
def group : MergeGroup := .relation 45458
def deltas0_0 : Polynomial Owner := [LeftMerge45459.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge45459.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def accumulator : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7182⟩⟩, ⟨.program ⟨257⟩, ⟨34164⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨31900⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨34164⟩⟩] } }]
def source : MonomialKey Owner := ⟨[⟨.program ⟨257⟩, ⟨31900⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨34164⟩⟩]⟩
def rhs : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨33181⟩⟩] } }]
def base : Polynomial Owner := subtract accumulator [{ coefficient := (-1), key := source }]
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7182⟩⟩, ⟨.program ⟨257⟩, ⟨34164⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨31900⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨33181⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem relationAgreement : CanonicalAgreement (add base reconstruction.deltas)
    (relationPoly accumulator source
      (relationContext source source.centralFactors 0 2) (-1) rhs) := by
  dsimp [reconstruction]
  decide +kernel
end LeftRelationMerge45458

namespace LeftRelationMerge45670
def frameStart : Nat := 45590
def owner : Owner := ⟨.program ⟨257⟩, ⟨24145⟩⟩
def group : MergeGroup := .relation 45670
def deltas0_0 : Polynomial Owner := [LeftMerge45671.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge45671.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def accumulator : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7181⟩⟩, ⟨.program ⟨257⟩, ⟨24144⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨21880⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨24144⟩⟩] } }]
def source : MonomialKey Owner := ⟨[⟨.program ⟨257⟩, ⟨21880⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨24144⟩⟩]⟩
def rhs : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨23161⟩⟩] } }]
def base : Polynomial Owner := subtract accumulator [{ coefficient := (-1), key := source }]
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7181⟩⟩, ⟨.program ⟨257⟩, ⟨24144⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨21880⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨23161⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem relationAgreement : CanonicalAgreement (add base reconstruction.deltas)
    (relationPoly accumulator source
      (relationContext source source.centralFactors 0 2) (-1) rhs) := by
  dsimp [reconstruction]
  decide +kernel
end LeftRelationMerge45670

namespace LeftRelationMerge45882
def frameStart : Nat := 45802
def owner : Owner := ⟨.program ⟨257⟩, ⟨20925⟩⟩
def group : MergeGroup := .relation 45882
def deltas0_0 : Polynomial Owner := [LeftMerge45883.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge45883.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def accumulator : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7180⟩⟩, ⟨.program ⟨257⟩, ⟨20924⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨18660⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨20924⟩⟩] } }]
def source : MonomialKey Owner := ⟨[⟨.program ⟨257⟩, ⟨18660⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨20924⟩⟩]⟩
def rhs : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨19941⟩⟩] } }]
def base : Polynomial Owner := subtract accumulator [{ coefficient := (-1), key := source }]
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7180⟩⟩, ⟨.program ⟨257⟩, ⟨20924⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨18660⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨19941⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem relationAgreement : CanonicalAgreement (add base reconstruction.deltas)
    (relationPoly accumulator source
      (relationContext source source.centralFactors 0 2) (-1) rhs) := by
  dsimp [reconstruction]
  decide +kernel
end LeftRelationMerge45882

namespace LeftRelationMerge46094
def frameStart : Nat := 46014
def owner : Owner := ⟨.program ⟨257⟩, ⟨18007⟩⟩
def group : MergeGroup := .relation 46094
def deltas0_0 : Polynomial Owner := [LeftMerge46095.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge46095.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def accumulator : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨18006⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨15860⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨18006⟩⟩] } }]
def source : MonomialKey Owner := ⟨[⟨.program ⟨257⟩, ⟨15860⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨18006⟩⟩]⟩
def rhs : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨17081⟩⟩] } }]
def base : Polynomial Owner := subtract accumulator [{ coefficient := (-1), key := source }]
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨18006⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨15860⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨17081⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem relationAgreement : CanonicalAgreement (add base reconstruction.deltas)
    (relationPoly accumulator source
      (relationContext source source.centralFactors 0 2) (-1) rhs) := by
  dsimp [reconstruction]
  decide +kernel
end LeftRelationMerge46094

namespace LeftRelationMerge46894
def frameStart : Nat := 46800
def owner : Owner := ⟨.program ⟨257⟩, ⟨49750⟩⟩
def group : MergeGroup := .relation 46894
def deltas0_0 : Polynomial Owner := [LeftMerge46895.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge46895.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def accumulator : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49747⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨15201⟩⟩, ⟨.program ⟨257⟩, ⟨48026⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨49747⟩⟩] } }]
def source : MonomialKey Owner := ⟨[⟨.program ⟨257⟩, ⟨15201⟩⟩, ⟨.program ⟨257⟩, ⟨48026⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨49747⟩⟩]⟩
def rhs : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨49197⟩⟩] } }]
def base : Polynomial Owner := subtract accumulator [{ coefficient := (-1), key := source }]
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49747⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨15201⟩⟩, ⟨.program ⟨257⟩, ⟨48026⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨49197⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem relationAgreement : CanonicalAgreement (add base reconstruction.deltas)
    (relationPoly accumulator source
      (relationContext source source.centralFactors 0 2) (-1) rhs) := by
  dsimp [reconstruction]
  decide +kernel
end LeftRelationMerge46894

namespace LeftRelationMerge47089
def frameStart : Nat := 47009
def owner : Owner := ⟨.program ⟨257⟩, ⟨50230⟩⟩
def group : MergeGroup := .relation 47089
def deltas0_0 : Polynomial Owner := [LeftMerge47090.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge47090.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def accumulator : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨50229⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨48212⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨50229⟩⟩] } }]
def source : MonomialKey Owner := ⟨[⟨.program ⟨257⟩, ⟨48212⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨50229⟩⟩]⟩
def rhs : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨49373⟩⟩] } }]
def base : Polynomial Owner := subtract accumulator [{ coefficient := (-1), key := source }]
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨50229⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨48212⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨49373⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem relationAgreement : CanonicalAgreement (add base reconstruction.deltas)
    (relationPoly accumulator source
      (relationContext source source.centralFactors 0 2) (-1) rhs) := by
  dsimp [reconstruction]
  decide +kernel
end LeftRelationMerge47089

namespace LeftRelationMerge47376
def frameStart : Nat := 47282
def owner : Owner := ⟨.program ⟨257⟩, ⟨47070⟩⟩
def group : MergeGroup := .relation 47376
def deltas0_0 : Polynomial Owner := [LeftMerge47377.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge47377.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def accumulator : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨47067⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14901⟩⟩, ⟨.program ⟨257⟩, ⟨45346⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨47067⟩⟩] } }]
def source : MonomialKey Owner := ⟨[⟨.program ⟨257⟩, ⟨14901⟩⟩, ⟨.program ⟨257⟩, ⟨45346⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨47067⟩⟩]⟩
def rhs : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨46517⟩⟩] } }]
def base : Polynomial Owner := subtract accumulator [{ coefficient := (-1), key := source }]
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨47067⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14901⟩⟩, ⟨.program ⟨257⟩, ⟨45346⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46517⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem relationAgreement : CanonicalAgreement (add base reconstruction.deltas)
    (relationPoly accumulator source
      (relationContext source source.centralFactors 0 2) (-1) rhs) := by
  dsimp [reconstruction]
  decide +kernel
end LeftRelationMerge47376

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
