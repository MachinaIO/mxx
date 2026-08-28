import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeDeltaShard044
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeDeltaShard098
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeDeltaShard280
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeDeltaShard281
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeDeltaShard316
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeDeltaShard372
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeDeltaShard373
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeDeltaShard408
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeDeltaShard464
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeDeltaShard465
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeDeltaShard500
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeDeltaShard555
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeDeltaShard556
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeDeltaShard589
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeDeltaShard645
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeDeltaShard646

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace LeftRelationMerge17935
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨214⟩, ⟨29434⟩⟩
def group : MergeGroup := .relation 17935
def deltas0_0 : Polynomial Owner := [LeftMerge17936.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge17936.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def accumulator : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6736⟩⟩, ⟨.program ⟨214⟩, ⟨6665⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨17734⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨6665⟩⟩] } }]
def source : MonomialKey Owner := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨17734⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨6665⟩⟩]⟩
def rhs : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6459⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def base : Polynomial Owner := subtract accumulator [{ coefficient := (-1), key := source }]
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6736⟩⟩, ⟨.program ⟨214⟩, ⟨6665⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6459⟩⟩, ⟨.program ⟨214⟩, ⟨17734⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem relationAgreement : CanonicalAgreement (add base reconstruction.deltas)
    (relationPoly accumulator source
      (relationContext source source.centralFactors 0 2) (-1) rhs) := by
  dsimp [reconstruction]
  decide +kernel
end LeftRelationMerge17935

namespace LeftRelationMerge8257
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨214⟩, ⟨29439⟩⟩
def group : MergeGroup := .relation 8257
def deltas0_0 : Polynomial Owner := [LeftMerge8258.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge8258.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def accumulator : Polynomial Owner := [{ coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨16649⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29437⟩⟩] } }]
def source : MonomialKey Owner := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨16649⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29437⟩⟩]⟩
def rhs : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨24615⟩⟩] } }]
def base : Polynomial Owner := subtract accumulator [{ coefficient := (-1), key := source }]
def working : Polynomial Owner := [{ coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨16649⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24615⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem relationAgreement : CanonicalAgreement (add base reconstruction.deltas)
    (relationPoly accumulator source
      (relationContext source source.centralFactors 0 2) (-1) rhs) := by
  dsimp [reconstruction]
  decide +kernel
end LeftRelationMerge8257

namespace LeftRelationMerge104170
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨214⟩, ⟨29562⟩⟩
def group : MergeGroup := .relation 104170
def deltas0_0 : Polynomial Owner := [LeftMerge104171.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge104171.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def accumulator : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6705⟩⟩, ⟨.program ⟨214⟩, ⟨29560⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16742⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29560⟩⟩] } }]
def source : MonomialKey Owner := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16742⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29560⟩⟩]⟩
def rhs : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨24656⟩⟩] } }]
def base : Polynomial Owner := subtract accumulator [{ coefficient := (-1), key := source }]
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6705⟩⟩, ⟨.program ⟨214⟩, ⟨29560⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16742⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24656⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem relationAgreement : CanonicalAgreement (add base reconstruction.deltas)
    (relationPoly accumulator source
      (relationContext source source.centralFactors 0 2) (-1) rhs) := by
  dsimp [reconstruction]
  decide +kernel
end LeftRelationMerge104170

namespace LeftRelationMerge104341
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨214⟩, ⟨29564⟩⟩
def group : MergeGroup := .relation 104341
def deltas0_0 : Polynomial Owner := [LeftMerge104342.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge104342.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def accumulator : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6738⟩⟩, ⟨.program ⟨214⟩, ⟨6661⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨17484⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨6661⟩⟩] } }]
def source : MonomialKey Owner := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨17484⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨6661⟩⟩]⟩
def rhs : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6449⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def base : Polynomial Owner := subtract accumulator [{ coefficient := (-1), key := source }]
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6738⟩⟩, ⟨.program ⟨214⟩, ⟨6661⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6449⟩⟩, ⟨.program ⟨214⟩, ⟨17484⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem relationAgreement : CanonicalAgreement (add base reconstruction.deltas)
    (relationPoly accumulator source
      (relationContext source source.centralFactors 0 2) (-1) rhs) := by
  dsimp [reconstruction]
  decide +kernel
end LeftRelationMerge104341

namespace LeftRelationMerge95500
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨214⟩, ⟨29569⟩⟩
def group : MergeGroup := .relation 95500
def deltas0_0 : Polynomial Owner := [LeftMerge95501.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge95501.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def accumulator : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6705⟩⟩, ⟨.program ⟨214⟩, ⟨29567⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16742⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29567⟩⟩] } }]
def source : MonomialKey Owner := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16742⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29567⟩⟩]⟩
def rhs : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨24657⟩⟩] } }]
def base : Polynomial Owner := subtract accumulator [{ coefficient := (-1), key := source }]
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6705⟩⟩, ⟨.program ⟨214⟩, ⟨29567⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16742⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24657⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem relationAgreement : CanonicalAgreement (add base reconstruction.deltas)
    (relationPoly accumulator source
      (relationContext source source.centralFactors 0 2) (-1) rhs) := by
  dsimp [reconstruction]
  decide +kernel
end LeftRelationMerge95500

namespace LeftRelationMerge76031
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨214⟩, ⟨29584⟩⟩
def group : MergeGroup := .relation 76031
def deltas0_0 : Polynomial Owner := [LeftMerge76032.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge76032.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def accumulator : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6705⟩⟩, ⟨.program ⟨214⟩, ⟨29582⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16748⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29582⟩⟩] } }]
def source : MonomialKey Owner := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16748⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29582⟩⟩]⟩
def rhs : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨24662⟩⟩] } }]
def base : Polynomial Owner := subtract accumulator [{ coefficient := (-1), key := source }]
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6705⟩⟩, ⟨.program ⟨214⟩, ⟨29582⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16748⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24662⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem relationAgreement : CanonicalAgreement (add base reconstruction.deltas)
    (relationPoly accumulator source
      (relationContext source source.centralFactors 0 2) (-1) rhs) := by
  dsimp [reconstruction]
  decide +kernel
end LeftRelationMerge76031

namespace LeftRelationMerge76226
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨214⟩, ⟨29586⟩⟩
def group : MergeGroup := .relation 76226
def deltas0_0 : Polynomial Owner := [LeftMerge76227.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge76227.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def accumulator : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6738⟩⟩, ⟨.program ⟨214⟩, ⟨6661⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨17490⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨6661⟩⟩] } }]
def source : MonomialKey Owner := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨17490⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨6661⟩⟩]⟩
def rhs : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6449⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def base : Polynomial Owner := subtract accumulator [{ coefficient := (-1), key := source }]
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6738⟩⟩, ⟨.program ⟨214⟩, ⟨6661⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨6449⟩⟩, ⟨.program ⟨214⟩, ⟨17490⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem relationAgreement : CanonicalAgreement (add base reconstruction.deltas)
    (relationPoly accumulator source
      (relationContext source source.centralFactors 0 2) (-1) rhs) := by
  dsimp [reconstruction]
  decide +kernel
end LeftRelationMerge76226

namespace LeftRelationMerge66545
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨214⟩, ⟨29591⟩⟩
def group : MergeGroup := .relation 66545
def deltas0_0 : Polynomial Owner := [LeftMerge66546.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge66546.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def accumulator : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6705⟩⟩, ⟨.program ⟨214⟩, ⟨29589⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16748⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29589⟩⟩] } }]
def source : MonomialKey Owner := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16748⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29589⟩⟩]⟩
def rhs : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨24663⟩⟩] } }]
def base : Polynomial Owner := subtract accumulator [{ coefficient := (-1), key := source }]
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6705⟩⟩, ⟨.program ⟨214⟩, ⟨29589⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16748⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24663⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem relationAgreement : CanonicalAgreement (add base reconstruction.deltas)
    (relationPoly accumulator source
      (relationContext source source.centralFactors 0 2) (-1) rhs) := by
  dsimp [reconstruction]
  decide +kernel
end LeftRelationMerge66545

namespace LeftRelationMerge90620
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨214⟩, ⟨29597⟩⟩
def group : MergeGroup := .relation 90620
def deltas0_0 : Polynomial Owner := [LeftMerge90621.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge90621.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def accumulator : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6705⟩⟩, ⟨.program ⟨214⟩, ⟨29595⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16752⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29595⟩⟩] } }]
def source : MonomialKey Owner := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16752⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29595⟩⟩]⟩
def rhs : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨24665⟩⟩] } }]
def base : Polynomial Owner := subtract accumulator [{ coefficient := (-1), key := source }]
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6705⟩⟩, ⟨.program ⟨214⟩, ⟨29595⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16752⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24665⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem relationAgreement : CanonicalAgreement (add base reconstruction.deltas)
    (relationPoly accumulator source
      (relationContext source source.centralFactors 0 2) (-1) rhs) := by
  dsimp [reconstruction]
  decide +kernel
end LeftRelationMerge90620

namespace LeftRelationMerge90815
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨214⟩, ⟨29599⟩⟩
def group : MergeGroup := .relation 90815
def deltas0_0 : Polynomial Owner := [LeftMerge90816.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge90816.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def accumulator : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6738⟩⟩, ⟨.program ⟨214⟩, ⟨6661⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨17494⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨6661⟩⟩] } }]
def source : MonomialKey Owner := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨17494⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨6661⟩⟩]⟩
def rhs : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6449⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def base : Polynomial Owner := subtract accumulator [{ coefficient := (-1), key := source }]
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6738⟩⟩, ⟨.program ⟨214⟩, ⟨6661⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨6449⟩⟩, ⟨.program ⟨214⟩, ⟨17494⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem relationAgreement : CanonicalAgreement (add base reconstruction.deltas)
    (relationPoly accumulator source
      (relationContext source source.centralFactors 0 2) (-1) rhs) := by
  dsimp [reconstruction]
  decide +kernel
end LeftRelationMerge90815

namespace LeftRelationMerge81164
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨214⟩, ⟨29604⟩⟩
def group : MergeGroup := .relation 81164
def deltas0_0 : Polynomial Owner := [LeftMerge81165.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge81165.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def accumulator : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6705⟩⟩, ⟨.program ⟨214⟩, ⟨29602⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16752⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29602⟩⟩] } }]
def source : MonomialKey Owner := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16752⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29602⟩⟩]⟩
def rhs : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨24666⟩⟩] } }]
def base : Polynomial Owner := subtract accumulator [{ coefficient := (-1), key := source }]
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6705⟩⟩, ⟨.program ⟨214⟩, ⟨29602⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16752⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24666⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem relationAgreement : CanonicalAgreement (add base reconstruction.deltas)
    (relationPoly accumulator source
      (relationContext source source.centralFactors 0 2) (-1) rhs) := by
  dsimp [reconstruction]
  decide +kernel
end LeftRelationMerge81164

namespace LeftRelationMerge61406
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨214⟩, ⟨29610⟩⟩
def group : MergeGroup := .relation 61406
def deltas0_0 : Polynomial Owner := [LeftMerge61407.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge61407.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def accumulator : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6705⟩⟩, ⟨.program ⟨214⟩, ⟨29608⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨16756⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29608⟩⟩] } }]
def source : MonomialKey Owner := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨16756⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29608⟩⟩]⟩
def rhs : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨24668⟩⟩] } }]
def base : Polynomial Owner := subtract accumulator [{ coefficient := (-1), key := source }]
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6705⟩⟩, ⟨.program ⟨214⟩, ⟨29608⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨16756⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24668⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem relationAgreement : CanonicalAgreement (add base reconstruction.deltas)
    (relationPoly accumulator source
      (relationContext source source.centralFactors 0 2) (-1) rhs) := by
  dsimp [reconstruction]
  decide +kernel
end LeftRelationMerge61406

namespace LeftRelationMerge61601
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨214⟩, ⟨29612⟩⟩
def group : MergeGroup := .relation 61601
def deltas0_0 : Polynomial Owner := [LeftMerge61602.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge61602.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def accumulator : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6738⟩⟩, ⟨.program ⟨214⟩, ⟨6661⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨17498⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨6661⟩⟩] } }]
def source : MonomialKey Owner := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨17498⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨6661⟩⟩]⟩
def rhs : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6449⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def base : Polynomial Owner := subtract accumulator [{ coefficient := (-1), key := source }]
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6738⟩⟩, ⟨.program ⟨214⟩, ⟨6661⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨6449⟩⟩, ⟨.program ⟨214⟩, ⟨17498⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem relationAgreement : CanonicalAgreement (add base reconstruction.deltas)
    (relationPoly accumulator source
      (relationContext source source.centralFactors 0 2) (-1) rhs) := by
  dsimp [reconstruction]
  decide +kernel
end LeftRelationMerge61601

namespace LeftRelationMerge51920
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨214⟩, ⟨29617⟩⟩
def group : MergeGroup := .relation 51920
def deltas0_0 : Polynomial Owner := [LeftMerge51921.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge51921.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def accumulator : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6705⟩⟩, ⟨.program ⟨214⟩, ⟨29615⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨16756⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29615⟩⟩] } }]
def source : MonomialKey Owner := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨16756⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29615⟩⟩]⟩
def rhs : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨24669⟩⟩] } }]
def base : Polynomial Owner := subtract accumulator [{ coefficient := (-1), key := source }]
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6705⟩⟩, ⟨.program ⟨214⟩, ⟨29615⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨16756⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24669⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem relationAgreement : CanonicalAgreement (add base reconstruction.deltas)
    (relationPoly accumulator source
      (relationContext source source.centralFactors 0 2) (-1) rhs) := by
  dsimp [reconstruction]
  decide +kernel
end LeftRelationMerge51920

namespace LeftRelationMerge46781
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨214⟩, ⟨29623⟩⟩
def group : MergeGroup := .relation 46781
def deltas0_0 : Polynomial Owner := [LeftMerge46782.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge46782.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def accumulator : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6705⟩⟩, ⟨.program ⟨214⟩, ⟨29621⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨16760⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29621⟩⟩] } }]
def source : MonomialKey Owner := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨16760⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29621⟩⟩]⟩
def rhs : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨24671⟩⟩] } }]
def base : Polynomial Owner := subtract accumulator [{ coefficient := (-1), key := source }]
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6705⟩⟩, ⟨.program ⟨214⟩, ⟨29621⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨16760⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24671⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem relationAgreement : CanonicalAgreement (add base reconstruction.deltas)
    (relationPoly accumulator source
      (relationContext source source.centralFactors 0 2) (-1) rhs) := by
  dsimp [reconstruction]
  decide +kernel
end LeftRelationMerge46781

namespace LeftRelationMerge46976
def frameStart : Nat := 0
def owner : Owner := ⟨.program ⟨214⟩, ⟨29625⟩⟩
def group : MergeGroup := .relation 46976
def deltas0_0 : Polynomial Owner := [LeftMerge46977.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge46977.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def accumulator : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6738⟩⟩, ⟨.program ⟨214⟩, ⟨6661⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨17502⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨6661⟩⟩] } }]
def source : MonomialKey Owner := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨17502⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨6661⟩⟩]⟩
def rhs : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6449⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def base : Polynomial Owner := subtract accumulator [{ coefficient := (-1), key := source }]
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6738⟩⟩, ⟨.program ⟨214⟩, ⟨6661⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨6449⟩⟩, ⟨.program ⟨214⟩, ⟨17502⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem relationAgreement : CanonicalAgreement (add base reconstruction.deltas)
    (relationPoly accumulator source
      (relationContext source source.centralFactors 0 2) (-1) rhs) := by
  dsimp [reconstruction]
  decide +kernel
end LeftRelationMerge46976

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
