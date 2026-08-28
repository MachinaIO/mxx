import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeDeltaShard326
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeDeltaShard327
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeDeltaShard329
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeDeltaShard330

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace LeftOperatorMerge53792
def frameStart : Nat := 53709
def owner : Owner := ⟨.program ⟨214⟩, ⟨7866⟩⟩
def group : MergeGroup := .operator 53788 53785
def deltas0_0 : Polynomial Owner := [LeftMerge53792.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge53792.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge53792.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge53792.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6764⟩⟩, ⟨.program ⟨214⟩, ⟨7864⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge53792

namespace LeftOperatorMerge53739
def frameStart : Nat := 53709
def owner : Owner := ⟨.program ⟨214⟩, ⟨11966⟩⟩
def group : MergeGroup := .operator 53735 53732
def deltas0_0 : Polynomial Owner := [LeftMerge53739.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge53739.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge53739.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge53739.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨9720⟩⟩, ⟨.program ⟨214⟩, ⟨11965⟩⟩], orderedFactors := [] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge53739

namespace LeftOperatorMerge53769
def frameStart : Nat := 53709
def owner : Owner := ⟨.program ⟨214⟩, ⟨12059⟩⟩
def group : MergeGroup := .operator 53765 53763
def deltas0_0 : Polynomial Owner := [LeftMerge53769.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge53769.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge53769.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge53769.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨9720⟩⟩, ⟨.program ⟨214⟩, ⟨11965⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge53769

namespace LeftOperatorMerge53812
def frameStart : Nat := 53709
def owner : Owner := ⟨.program ⟨214⟩, ⟨16387⟩⟩
def group : MergeGroup := .operator 53765 53808
def deltas0_0 : Polynomial Owner := [LeftMerge53812.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge53812.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge53812.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge53812.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16385⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge53812

namespace LeftOperatorMerge53801
def frameStart : Nat := 53709
def owner : Owner := ⟨.program ⟨214⟩, ⟨25227⟩⟩
def group : MergeGroup := .operator 53797 53754
def deltas0_0 : Polynomial Owner := [LeftMerge53801.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge53801.deltaAt
def deltas0_1 : Polynomial Owner := [LeftMerge53802.delta]
theorem rows0_1 : MergeDeltasAt history frameStart owner group deltas0_1 := by
  exact .leaf LeftMerge53802.deltaAt
def deltas1_0 : Polynomial Owner := deltas0_0 ++ deltas0_1
theorem rows1_0 : MergeDeltasAt history frameStart owner group deltas1_0 := by
  exact .append rows0_0 rows0_1
abbrev deltas : Polynomial Owner := deltas1_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows1_0
def left : Polynomial Owner := LeftMerge53801.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge53801.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6764⟩⟩, ⟨.program ⟨214⟩, ⟨7864⟩⟩, ⟨.program ⟨214⟩, ⟨25224⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨9720⟩⟩, ⟨.program ⟨214⟩, ⟨11965⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25224⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge53801

namespace LeftOperatorMerge53984
def frameStart : Nat := 53918
def owner : Owner := ⟨.program ⟨214⟩, ⟨16427⟩⟩
def group : MergeGroup := .operator 53980 53978
def deltas0_0 : Polynomial Owner := [LeftMerge53984.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge53984.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge53984.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge53984.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16385⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge53984

namespace LeftOperatorMerge54007
def frameStart : Nat := 53918
def owner : Owner := ⟨.program ⟨214⟩, ⟨17124⟩⟩
def group : MergeGroup := .operator 53980 54003
def deltas0_0 : Polynomial Owner := [LeftMerge54007.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge54007.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge54007.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge54007.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨17123⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge54007

namespace LeftOperatorMerge53996
def frameStart : Nat := 53918
def owner : Owner := ⟨.program ⟨214⟩, ⟨28748⟩⟩
def group : MergeGroup := .operator 53992 53969
def deltas0_0 : Polynomial Owner := [LeftMerge53996.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge53996.deltaAt
def deltas0_1 : Polynomial Owner := [LeftMerge53997.delta]
theorem rows0_1 : MergeDeltasAt history frameStart owner group deltas0_1 := by
  exact .leaf LeftMerge53997.deltaAt
def deltas1_0 : Polynomial Owner := deltas0_0 ++ deltas0_1
theorem rows1_0 : MergeDeltasAt history frameStart owner group deltas1_0 := by
  exact .append rows0_0 rows0_1
abbrev deltas : Polynomial Owner := deltas1_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows1_0
def left : Polynomial Owner := LeftMerge53996.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge53996.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6701⟩⟩, ⟨.program ⟨214⟩, ⟨28747⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16385⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28747⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge53996

namespace LeftOperatorMerge54274
def frameStart : Nat := 54191
def owner : Owner := ⟨.program ⟨214⟩, ⟨7863⟩⟩
def group : MergeGroup := .operator 54270 54267
def deltas0_0 : Polynomial Owner := [LeftMerge54274.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge54274.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge54274.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge54274.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6763⟩⟩, ⟨.program ⟨214⟩, ⟨7861⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge54274

namespace LeftOperatorMerge54221
def frameStart : Nat := 54191
def owner : Owner := ⟨.program ⟨214⟩, ⟨11770⟩⟩
def group : MergeGroup := .operator 54217 54214
def deltas0_0 : Polynomial Owner := [LeftMerge54221.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge54221.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge54221.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge54221.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨9615⟩⟩, ⟨.program ⟨214⟩, ⟨11769⟩⟩], orderedFactors := [] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge54221

namespace LeftOperatorMerge54251
def frameStart : Nat := 54191
def owner : Owner := ⟨.program ⟨214⟩, ⟨11863⟩⟩
def group : MergeGroup := .operator 54247 54245
def deltas0_0 : Polynomial Owner := [LeftMerge54251.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge54251.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge54251.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge54251.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨9615⟩⟩, ⟨.program ⟨214⟩, ⟨11769⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge54251

namespace LeftOperatorMerge54294
def frameStart : Nat := 54191
def owner : Owner := ⟨.program ⟨214⟩, ⟨16268⟩⟩
def group : MergeGroup := .operator 54247 54290
def deltas0_0 : Polynomial Owner := [LeftMerge54294.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge54294.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge54294.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge54294.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16266⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge54294

namespace LeftOperatorMerge54283
def frameStart : Nat := 54191
def owner : Owner := ⟨.program ⟨214⟩, ⟨25150⟩⟩
def group : MergeGroup := .operator 54279 54236
def deltas0_0 : Polynomial Owner := [LeftMerge54283.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge54283.deltaAt
def deltas0_1 : Polynomial Owner := [LeftMerge54284.delta]
theorem rows0_1 : MergeDeltasAt history frameStart owner group deltas0_1 := by
  exact .leaf LeftMerge54284.deltaAt
def deltas1_0 : Polynomial Owner := deltas0_0 ++ deltas0_1
theorem rows1_0 : MergeDeltasAt history frameStart owner group deltas1_0 := by
  exact .append rows0_0 rows0_1
abbrev deltas : Polynomial Owner := deltas1_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows1_0
def left : Polynomial Owner := LeftMerge54283.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge54283.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6763⟩⟩, ⟨.program ⟨214⟩, ⟨7861⟩⟩, ⟨.program ⟨214⟩, ⟨25147⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨9615⟩⟩, ⟨.program ⟨214⟩, ⟨11769⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25147⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge54283

namespace LeftOperatorMerge54489
def frameStart : Nat := 54400
def owner : Owner := ⟨.program ⟨214⟩, ⟨16312⟩⟩
def group : MergeGroup := .operator 54462 54485
def deltas0_0 : Polynomial Owner := [LeftMerge54489.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge54489.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge54489.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge54489.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16311⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge54489

namespace LeftOperatorMerge54466
def frameStart : Nat := 54400
def owner : Owner := ⟨.program ⟨214⟩, ⟨16343⟩⟩
def group : MergeGroup := .operator 54462 54460
def deltas0_0 : Polynomial Owner := [LeftMerge54466.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge54466.deltaAt
abbrev deltas : Polynomial Owner := deltas0_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows0_0
def left : Polynomial Owner := LeftMerge54466.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge54466.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16266⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge54466

namespace LeftOperatorMerge54478
def frameStart : Nat := 54400
def owner : Owner := ⟨.program ⟨214⟩, ⟨28531⟩⟩
def group : MergeGroup := .operator 54474 54451
def deltas0_0 : Polynomial Owner := [LeftMerge54478.delta]
theorem rows0_0 : MergeDeltasAt history frameStart owner group deltas0_0 := by
  exact .leaf LeftMerge54478.deltaAt
def deltas0_1 : Polynomial Owner := [LeftMerge54479.delta]
theorem rows0_1 : MergeDeltasAt history frameStart owner group deltas0_1 := by
  exact .leaf LeftMerge54479.deltaAt
def deltas1_0 : Polynomial Owner := deltas0_0 ++ deltas0_1
theorem rows1_0 : MergeDeltasAt history frameStart owner group deltas1_0 := by
  exact .append rows0_0 rows0_1
abbrev deltas : Polynomial Owner := deltas1_0
theorem rows : MergeDeltasAt history frameStart owner group deltas := rows1_0
def left : Polynomial Owner := LeftMerge54478.leftRaw.map Term.toExact
def right : Polynomial Owner := LeftMerge54478.rightRaw.map Term.toExact
def base : Polynomial Owner := []
def working : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6700⟩⟩, ⟨.program ⟨214⟩, ⟨28530⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16266⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28530⟩⟩] } }]
def reconstruction : MergeReconstructionAt history frameStart owner group base working :=
  { deltas := deltas
    rows := rows
    agreement := by decide +kernel }
theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) (productPoly left right false false) := by
  dsimp [reconstruction]
  decide +kernel
end LeftOperatorMerge54478

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
