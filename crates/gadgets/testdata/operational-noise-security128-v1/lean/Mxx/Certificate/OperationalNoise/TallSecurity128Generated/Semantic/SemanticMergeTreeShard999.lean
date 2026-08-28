import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard1261
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard1262
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard1263
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard1265
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard1266
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard1279
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard1280
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard1282
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard1283
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard1284
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard1285
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard1287
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard1288
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard1290
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard1291
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeDeltaShard1292

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace LeftRelationMerge206121
def frameStart : Nat := 206041
def owner : Owner := ⟨.program ⟨257⟩, ⟨53008⟩⟩
def group : MergeGroup := .relation 206121
def deltas0_0 : Polynomial Owner := [LeftMerge206122.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge206122.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def accumulator : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨53007⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨50904⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨53007⟩⟩] } }]
def source : MonomialKey Owner := ⟨[⟨.program ⟨257⟩, ⟨50904⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨53007⟩⟩]⟩
def rhs : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨52178⟩⟩] } }]
def base : Polynomial Owner := subtract accumulator [{ coefficient := (-1), key := source }]
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨53007⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨50904⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨52178⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem relationAgreement : CanonicalAgreement (add base reconstruction.deltas)
    (relationPoly accumulator source
      (relationContext source source.centralFactors 0 2) (-1) rhs) := by
  dsimp [reconstruction]
  decide +kernel
end LeftRelationMerge206121

namespace LeftRelationMerge206333
def frameStart : Nat := 206253
def owner : Owner := ⟨.program ⟨257⟩, ⟨33948⟩⟩
def group : MergeGroup := .relation 206333
def deltas0_0 : Polynomial Owner := [LeftMerge206334.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge206334.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def accumulator : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7182⟩⟩, ⟨.program ⟨257⟩, ⟨33947⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨31844⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨33947⟩⟩] } }]
def source : MonomialKey Owner := ⟨[⟨.program ⟨257⟩, ⟨31844⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨33947⟩⟩]⟩
def rhs : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨33118⟩⟩] } }]
def base : Polynomial Owner := subtract accumulator [{ coefficient := (-1), key := source }]
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7182⟩⟩, ⟨.program ⟨257⟩, ⟨33947⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨31844⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨33118⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem relationAgreement : CanonicalAgreement (add base reconstruction.deltas)
    (relationPoly accumulator source
      (relationContext source source.centralFactors 0 2) (-1) rhs) := by
  dsimp [reconstruction]
  decide +kernel
end LeftRelationMerge206333

namespace LeftRelationMerge206545
def frameStart : Nat := 206465
def owner : Owner := ⟨.program ⟨257⟩, ⟨23928⟩⟩
def group : MergeGroup := .relation 206545
def deltas0_0 : Polynomial Owner := [LeftMerge206546.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge206546.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def accumulator : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7181⟩⟩, ⟨.program ⟨257⟩, ⟨23927⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨21824⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨23927⟩⟩] } }]
def source : MonomialKey Owner := ⟨[⟨.program ⟨257⟩, ⟨21824⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨23927⟩⟩]⟩
def rhs : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨23098⟩⟩] } }]
def base : Polynomial Owner := subtract accumulator [{ coefficient := (-1), key := source }]
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7181⟩⟩, ⟨.program ⟨257⟩, ⟨23927⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨21824⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨23098⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem relationAgreement : CanonicalAgreement (add base reconstruction.deltas)
    (relationPoly accumulator source
      (relationContext source source.centralFactors 0 2) (-1) rhs) := by
  dsimp [reconstruction]
  decide +kernel
end LeftRelationMerge206545

namespace LeftRelationMerge206757
def frameStart : Nat := 206677
def owner : Owner := ⟨.program ⟨257⟩, ⟨20708⟩⟩
def group : MergeGroup := .relation 206757
def deltas0_0 : Polynomial Owner := [LeftMerge206758.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge206758.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def accumulator : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7180⟩⟩, ⟨.program ⟨257⟩, ⟨20707⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨18604⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨20707⟩⟩] } }]
def source : MonomialKey Owner := ⟨[⟨.program ⟨257⟩, ⟨18604⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨20707⟩⟩]⟩
def rhs : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨19878⟩⟩] } }]
def base : Polynomial Owner := subtract accumulator [{ coefficient := (-1), key := source }]
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7180⟩⟩, ⟨.program ⟨257⟩, ⟨20707⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨18604⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨19878⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem relationAgreement : CanonicalAgreement (add base reconstruction.deltas)
    (relationPoly accumulator source
      (relationContext source source.centralFactors 0 2) (-1) rhs) := by
  dsimp [reconstruction]
  decide +kernel
end LeftRelationMerge206757

namespace LeftRelationMerge206969
def frameStart : Nat := 206889
def owner : Owner := ⟨.program ⟨257⟩, ⟨17811⟩⟩
def group : MergeGroup := .relation 206969
def deltas0_0 : Polynomial Owner := [LeftMerge206970.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge206970.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def accumulator : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17810⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨15804⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17810⟩⟩] } }]
def source : MonomialKey Owner := ⟨[⟨.program ⟨257⟩, ⟨15804⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17810⟩⟩]⟩
def rhs : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨17018⟩⟩] } }]
def base : Polynomial Owner := subtract accumulator [{ coefficient := (-1), key := source }]
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17810⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨15804⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨17018⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem relationAgreement : CanonicalAgreement (add base reconstruction.deltas)
    (relationPoly accumulator source
      (relationContext source source.centralFactors 0 2) (-1) rhs) := by
  dsimp [reconstruction]
  decide +kernel
end LeftRelationMerge206969

namespace LeftRelationMerge207769
def frameStart : Nat := 207675
def owner : Owner := ⟨.program ⟨257⟩, ⟨49662⟩⟩
def group : MergeGroup := .relation 207769
def deltas0_0 : Polynomial Owner := [LeftMerge207770.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge207770.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def accumulator : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49659⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨15081⟩⟩, ⟨.program ⟨257⟩, ⟨47834⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨49659⟩⟩] } }]
def source : MonomialKey Owner := ⟨[⟨.program ⟨257⟩, ⟨15081⟩⟩, ⟨.program ⟨257⟩, ⟨47834⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨49659⟩⟩]⟩
def rhs : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨49149⟩⟩] } }]
def base : Polynomial Owner := subtract accumulator [{ coefficient := (-1), key := source }]
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49659⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨15081⟩⟩, ⟨.program ⟨257⟩, ⟨47834⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨49149⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem relationAgreement : CanonicalAgreement (add base reconstruction.deltas)
    (relationPoly accumulator source
      (relationContext source source.centralFactors 0 2) (-1) rhs) := by
  dsimp [reconstruction]
  decide +kernel
end LeftRelationMerge207769

namespace LeftRelationMerge207964
def frameStart : Nat := 207884
def owner : Owner := ⟨.program ⟨257⟩, ⟨50030⟩⟩
def group : MergeGroup := .relation 207964
def deltas0_0 : Polynomial Owner := [LeftMerge207965.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge207965.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def accumulator : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨50029⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨48148⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨50029⟩⟩] } }]
def source : MonomialKey Owner := ⟨[⟨.program ⟨257⟩, ⟨48148⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨50029⟩⟩]⟩
def rhs : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨49301⟩⟩] } }]
def base : Polynomial Owner := subtract accumulator [{ coefficient := (-1), key := source }]
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨50029⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨48148⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨49301⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem relationAgreement : CanonicalAgreement (add base reconstruction.deltas)
    (relationPoly accumulator source
      (relationContext source source.centralFactors 0 2) (-1) rhs) := by
  dsimp [reconstruction]
  decide +kernel
end LeftRelationMerge207964

namespace LeftRelationMerge208251
def frameStart : Nat := 208157
def owner : Owner := ⟨.program ⟨257⟩, ⟨46982⟩⟩
def group : MergeGroup := .relation 208251
def deltas0_0 : Polynomial Owner := [LeftMerge208252.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge208252.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def accumulator : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨46979⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14781⟩⟩, ⟨.program ⟨257⟩, ⟨45154⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨46979⟩⟩] } }]
def source : MonomialKey Owner := ⟨[⟨.program ⟨257⟩, ⟨14781⟩⟩, ⟨.program ⟨257⟩, ⟨45154⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨46979⟩⟩]⟩
def rhs : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨46469⟩⟩] } }]
def base : Polynomial Owner := subtract accumulator [{ coefficient := (-1), key := source }]
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨46979⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14781⟩⟩, ⟨.program ⟨257⟩, ⟨45154⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46469⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem relationAgreement : CanonicalAgreement (add base reconstruction.deltas)
    (relationPoly accumulator source
      (relationContext source source.centralFactors 0 2) (-1) rhs) := by
  dsimp [reconstruction]
  decide +kernel
end LeftRelationMerge208251

namespace LeftRelationMerge208446
def frameStart : Nat := 208366
def owner : Owner := ⟨.program ⟨257⟩, ⟨47350⟩⟩
def group : MergeGroup := .relation 208446
def deltas0_0 : Polynomial Owner := [LeftMerge208447.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge208447.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def accumulator : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47349⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨45468⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨47349⟩⟩] } }]
def source : MonomialKey Owner := ⟨[⟨.program ⟨257⟩, ⟨45468⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨47349⟩⟩]⟩
def rhs : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨46621⟩⟩] } }]
def base : Polynomial Owner := subtract accumulator [{ coefficient := (-1), key := source }]
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47349⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨45468⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46621⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem relationAgreement : CanonicalAgreement (add base reconstruction.deltas)
    (relationPoly accumulator source
      (relationContext source source.centralFactors 0 2) (-1) rhs) := by
  dsimp [reconstruction]
  decide +kernel
end LeftRelationMerge208446

namespace LeftRelationMerge208733
def frameStart : Nat := 208639
def owner : Owner := ⟨.program ⟨257⟩, ⟨44302⟩⟩
def group : MergeGroup := .relation 208733
def deltas0_0 : Polynomial Owner := [LeftMerge208734.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge208734.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def accumulator : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩, ⟨.program ⟨257⟩, ⟨44299⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14481⟩⟩, ⟨.program ⟨257⟩, ⟨42474⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨44299⟩⟩] } }]
def source : MonomialKey Owner := ⟨[⟨.program ⟨257⟩, ⟨14481⟩⟩, ⟨.program ⟨257⟩, ⟨42474⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨44299⟩⟩]⟩
def rhs : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨43789⟩⟩] } }]
def base : Polynomial Owner := subtract accumulator [{ coefficient := (-1), key := source }]
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩, ⟨.program ⟨257⟩, ⟨44299⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14481⟩⟩, ⟨.program ⟨257⟩, ⟨42474⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨43789⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem relationAgreement : CanonicalAgreement (add base reconstruction.deltas)
    (relationPoly accumulator source
      (relationContext source source.centralFactors 0 2) (-1) rhs) := by
  dsimp [reconstruction]
  decide +kernel
end LeftRelationMerge208733

namespace LeftRelationMerge208928
def frameStart : Nat := 208848
def owner : Owner := ⟨.program ⟨257⟩, ⟨44670⟩⟩
def group : MergeGroup := .relation 208928
def deltas0_0 : Polynomial Owner := [LeftMerge208929.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge208929.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def accumulator : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44669⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨42788⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨44669⟩⟩] } }]
def source : MonomialKey Owner := ⟨[⟨.program ⟨257⟩, ⟨42788⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨44669⟩⟩]⟩
def rhs : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨43941⟩⟩] } }]
def base : Polynomial Owner := subtract accumulator [{ coefficient := (-1), key := source }]
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44669⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨42788⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨43941⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem relationAgreement : CanonicalAgreement (add base reconstruction.deltas)
    (relationPoly accumulator source
      (relationContext source source.centralFactors 0 2) (-1) rhs) := by
  dsimp [reconstruction]
  decide +kernel
end LeftRelationMerge208928

namespace LeftRelationMerge209215
def frameStart : Nat := 209121
def owner : Owner := ⟨.program ⟨257⟩, ⟨41622⟩⟩
def group : MergeGroup := .relation 209215
def deltas0_0 : Polynomial Owner := [LeftMerge209216.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge209216.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def accumulator : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41619⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14181⟩⟩, ⟨.program ⟨257⟩, ⟨39794⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨41619⟩⟩] } }]
def source : MonomialKey Owner := ⟨[⟨.program ⟨257⟩, ⟨14181⟩⟩, ⟨.program ⟨257⟩, ⟨39794⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨41619⟩⟩]⟩
def rhs : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨41109⟩⟩] } }]
def base : Polynomial Owner := subtract accumulator [{ coefficient := (-1), key := source }]
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41619⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14181⟩⟩, ⟨.program ⟨257⟩, ⟨39794⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41109⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem relationAgreement : CanonicalAgreement (add base reconstruction.deltas)
    (relationPoly accumulator source
      (relationContext source source.centralFactors 0 2) (-1) rhs) := by
  dsimp [reconstruction]
  decide +kernel
end LeftRelationMerge209215

namespace LeftRelationMerge209410
def frameStart : Nat := 209330
def owner : Owner := ⟨.program ⟨257⟩, ⟨41990⟩⟩
def group : MergeGroup := .relation 209410
def deltas0_0 : Polynomial Owner := [LeftMerge209411.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge209411.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def accumulator : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨41989⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨40108⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨41989⟩⟩] } }]
def source : MonomialKey Owner := ⟨[⟨.program ⟨257⟩, ⟨40108⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨41989⟩⟩]⟩
def rhs : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨41261⟩⟩] } }]
def base : Polynomial Owner := subtract accumulator [{ coefficient := (-1), key := source }]
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨41989⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨40108⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41261⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem relationAgreement : CanonicalAgreement (add base reconstruction.deltas)
    (relationPoly accumulator source
      (relationContext source source.centralFactors 0 2) (-1) rhs) := by
  dsimp [reconstruction]
  decide +kernel
end LeftRelationMerge209410

namespace LeftRelationMerge209697
def frameStart : Nat := 209603
def owner : Owner := ⟨.program ⟨257⟩, ⟨38942⟩⟩
def group : MergeGroup := .relation 209697
def deltas0_0 : Polynomial Owner := [LeftMerge209698.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge209698.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def accumulator : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨38939⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨13881⟩⟩, ⟨.program ⟨257⟩, ⟨37114⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨38939⟩⟩] } }]
def source : MonomialKey Owner := ⟨[⟨.program ⟨257⟩, ⟨13881⟩⟩, ⟨.program ⟨257⟩, ⟨37114⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨38939⟩⟩]⟩
def rhs : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨38429⟩⟩] } }]
def base : Polynomial Owner := subtract accumulator [{ coefficient := (-1), key := source }]
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨38939⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨13881⟩⟩, ⟨.program ⟨257⟩, ⟨37114⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38429⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem relationAgreement : CanonicalAgreement (add base reconstruction.deltas)
    (relationPoly accumulator source
      (relationContext source source.centralFactors 0 2) (-1) rhs) := by
  dsimp [reconstruction]
  decide +kernel
end LeftRelationMerge209697

namespace LeftRelationMerge209892
def frameStart : Nat := 209812
def owner : Owner := ⟨.program ⟨257⟩, ⟨39310⟩⟩
def group : MergeGroup := .relation 209892
def deltas0_0 : Polynomial Owner := [LeftMerge209893.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge209893.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def accumulator : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7192⟩⟩, ⟨.program ⟨257⟩, ⟨39309⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨37428⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨39309⟩⟩] } }]
def source : MonomialKey Owner := ⟨[⟨.program ⟨257⟩, ⟨37428⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨39309⟩⟩]⟩
def rhs : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨38581⟩⟩] } }]
def base : Polynomial Owner := subtract accumulator [{ coefficient := (-1), key := source }]
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7192⟩⟩, ⟨.program ⟨257⟩, ⟨39309⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨37428⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38581⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem relationAgreement : CanonicalAgreement (add base reconstruction.deltas)
    (relationPoly accumulator source
      (relationContext source source.centralFactors 0 2) (-1) rhs) := by
  dsimp [reconstruction]
  decide +kernel
end LeftRelationMerge209892

namespace LeftRelationMerge210179
def frameStart : Nat := 210085
def owner : Owner := ⟨.program ⟨257⟩, ⟨36262⟩⟩
def group : MergeGroup := .relation 210179
def deltas0_0 : Polynomial Owner := [LeftMerge210180.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge210180.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def accumulator : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩, ⟨.program ⟨257⟩, ⟨36259⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨13581⟩⟩, ⟨.program ⟨257⟩, ⟨34434⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36259⟩⟩] } }]
def source : MonomialKey Owner := ⟨[⟨.program ⟨257⟩, ⟨13581⟩⟩, ⟨.program ⟨257⟩, ⟨34434⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36259⟩⟩]⟩
def rhs : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨35749⟩⟩] } }]
def base : Polynomial Owner := subtract accumulator [{ coefficient := (-1), key := source }]
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩, ⟨.program ⟨257⟩, ⟨36259⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨13581⟩⟩, ⟨.program ⟨257⟩, ⟨34434⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35749⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem relationAgreement : CanonicalAgreement (add base reconstruction.deltas)
    (relationPoly accumulator source
      (relationContext source source.centralFactors 0 2) (-1) rhs) := by
  dsimp [reconstruction]
  decide +kernel
end LeftRelationMerge210179

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
