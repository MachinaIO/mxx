import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard042
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard039
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard040
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard041

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult5212
def owner : Owner := ⟨.program ⟨214⟩, ⟨15197⟩⟩
def rawTerms : List Term := Proof.Events020.exact5212RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 5212
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult5212.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 5209) (rightBinding := 5210)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨15036⟩) (rightExpression := ⟨15196⟩)
    (transferEvent := 5211)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult5208.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult5184.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult5212

namespace SemanticResult5216
def owner : Owner := ⟨.program ⟨214⟩, ⟨15505⟩⟩
def rawTerms : List Term := Proof.Events020.exact5216RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 5216
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult5216.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 5213) (rightBinding := 5214)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨15197⟩) (rightExpression := ⟨15504⟩)
    (transferEvent := 5215)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult5212.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult5176.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult5216

namespace SemanticResult5220
def owner : Owner := ⟨.program ⟨214⟩, ⟨17794⟩⟩
def rawTerms : List Term := Proof.Events020.exact5220RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 5220
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult5220.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 5217) (rightBinding := 5218)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨15505⟩) (rightExpression := ⟨17793⟩)
    (transferEvent := 5219)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult5216.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult5168.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult5220

namespace SemanticResult5224
def owner : Owner := ⟨.program ⟨214⟩, ⟨17795⟩⟩
def rawTerms : List Term := Proof.Events020.exact5224RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 5224
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult5224.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 5221) (rightBinding := 5222)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨17794⟩) (rightExpression := ⟨17429⟩)
    (transferEvent := 5223)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult5220.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult5160.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult5224

namespace SemanticResult5228
def owner : Owner := ⟨.program ⟨214⟩, ⟨17796⟩⟩
def rawTerms : List Term := Proof.Events020.exact5228RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 5228
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult5228.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 5225) (rightBinding := 5226)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨17795⟩) (rightExpression := ⟨17212⟩)
    (transferEvent := 5227)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult5224.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult5152.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult5228

namespace SemanticResult5232
def owner : Owner := ⟨.program ⟨214⟩, ⟨17797⟩⟩
def rawTerms : List Term := Proof.Events020.exact5232RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 5232
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult5232.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 5229) (rightBinding := 5230)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨17796⟩) (rightExpression := ⟨17156⟩)
    (transferEvent := 5231)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult5228.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult5144.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult5232

namespace SemanticResult5236
def owner : Owner := ⟨.program ⟨214⟩, ⟨18018⟩⟩
def rawTerms : List Term := Proof.Events020.exact5236RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 5236
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult5236.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 5233) (rightBinding := 5234)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨17797⟩) (rightExpression := ⟨18017⟩)
    (transferEvent := 5235)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult5232.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult5136.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult5236

namespace SemanticResult5240
def owner : Owner := ⟨.program ⟨214⟩, ⟨18019⟩⟩
def rawTerms : List Term := Proof.Events020.exact5240RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 5240
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult5240.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 5237) (rightBinding := 5238)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨18018⟩) (rightExpression := ⟨17653⟩)
    (transferEvent := 5239)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult5236.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult5128.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult5240

namespace SemanticResult5244
def owner : Owner := ⟨.program ⟨214⟩, ⟨18020⟩⟩
def rawTerms : List Term := Proof.Events020.exact5244RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 5244
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult5244.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 5241) (rightBinding := 5242)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨18019⟩) (rightExpression := ⟨17597⟩)
    (transferEvent := 5243)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult5240.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult5120.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult5244

namespace SemanticResult5248
def owner : Owner := ⟨.program ⟨214⟩, ⟨18794⟩⟩
def rawTerms : List Term := Proof.Events020.exact5248RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 5248
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult5248.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 5245) (rightBinding := 5246)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨18020⟩) (rightExpression := ⟨18793⟩)
    (transferEvent := 5247)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult5244.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult5112.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult5248

namespace SemanticResult5252
def owner : Owner := ⟨.program ⟨214⟩, ⟨18795⟩⟩
def rawTerms : List Term := Proof.Events020.exact5252RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 5252
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult5252.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 5249) (rightBinding := 5250)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨18794⟩) (rightExpression := ⟨17541⟩)
    (transferEvent := 5251)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult5248.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult5104.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult5252

namespace SemanticResult5256
def owner : Owner := ⟨.program ⟨214⟩, ⟨18796⟩⟩
def rawTerms : List Term := Proof.Events020.exact5256RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 5256
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult5256.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 5253) (rightBinding := 5254)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨18795⟩) (rightExpression := ⟨17940⟩)
    (transferEvent := 5255)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult5252.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult5096.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult5256

namespace SemanticResult5260
def owner : Owner := ⟨.program ⟨214⟩, ⟨18797⟩⟩
def rawTerms : List Term := Proof.Events020.exact5260RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 5260
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult5260.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 5257) (rightBinding := 5258)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨18796⟩) (rightExpression := ⟨17709⟩)
    (transferEvent := 5259)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult5256.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult5088.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult5260

namespace SemanticResult5264
def owner : Owner := ⟨.program ⟨214⟩, ⟨18798⟩⟩
def rawTerms : List Term := Proof.Events020.exact5264RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 5264
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult5264.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 5261) (rightBinding := 5262)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨18797⟩) (rightExpression := ⟨17485⟩)
    (transferEvent := 5263)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult5260.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult5080.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult5264

namespace SemanticResult5268
def owner : Owner := ⟨.program ⟨214⟩, ⟨18799⟩⟩
def rawTerms : List Term := Proof.Events020.exact5268RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 5268
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult5268.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 5265) (rightBinding := 5266)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨18798⟩) (rightExpression := ⟨16918⟩)
    (transferEvent := 5267)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult5264.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult5072.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult5268

namespace SemanticResult5272
def owner : Owner := ⟨.program ⟨214⟩, ⟨18800⟩⟩
def rawTerms : List Term := Proof.Events020.exact5272RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 5272
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult5272.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 5269) (rightBinding := 5270)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨18799⟩) (rightExpression := ⟨18115⟩)
    (transferEvent := 5271)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult5268.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult5064.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult5272

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
