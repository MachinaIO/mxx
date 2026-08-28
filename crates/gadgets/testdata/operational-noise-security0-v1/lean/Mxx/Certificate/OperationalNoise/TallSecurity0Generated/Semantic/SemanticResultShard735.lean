import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard735
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard733
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard734

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult103353
def owner : Owner := ⟨.program ⟨214⟩, ⟨15301⟩⟩
def rawTerms : List Term := Proof.Events403.exact103353RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 103353
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult103353.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 103350) (rightBinding := 103351)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨15258⟩) (rightExpression := ⟨15300⟩)
    (transferEvent := 103352)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult103349.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult103326.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult103353

namespace SemanticResult103357
def owner : Owner := ⟨.program ⟨214⟩, ⟨15357⟩⟩
def rawTerms : List Term := Proof.Events403.exact103357RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 103357
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult103357.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 103354) (rightBinding := 103355)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨15301⟩) (rightExpression := ⟨15356⟩)
    (transferEvent := 103356)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult103353.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult103303.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult103357

namespace SemanticResult103361
def owner : Owner := ⟨.program ⟨214⟩, ⟨17303⟩⟩
def rawTerms : List Term := Proof.Events403.exact103361RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 103361
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult103361.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 103358) (rightBinding := 103359)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨15357⟩) (rightExpression := ⟨17302⟩)
    (transferEvent := 103360)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult103357.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult103280.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult103361

namespace SemanticResult103365
def owner : Owner := ⟨.program ⟨214⟩, ⟨17304⟩⟩
def rawTerms : List Term := Proof.Events403.exact103365RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 103365
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult103365.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 103362) (rightBinding := 103363)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨17303⟩) (rightExpression := ⟨15622⟩)
    (transferEvent := 103364)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult103361.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult103257.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult103365

namespace SemanticResult103369
def owner : Owner := ⟨.program ⟨214⟩, ⟨17305⟩⟩
def rawTerms : List Term := Proof.Events403.exact103369RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 103369
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult103369.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 103366) (rightBinding := 103367)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨17304⟩) (rightExpression := ⟨15741⟩)
    (transferEvent := 103368)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult103365.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult103234.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult103369

namespace SemanticResult103373
def owner : Owner := ⟨.program ⟨214⟩, ⟨17306⟩⟩
def rawTerms : List Term := Proof.Events403.exact103373RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 103373
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult103373.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 103370) (rightBinding := 103371)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨17305⟩) (rightExpression := ⟨15860⟩)
    (transferEvent := 103372)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult103369.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult103211.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult103373

namespace SemanticResult103377
def owner : Owner := ⟨.program ⟨214⟩, ⟨17307⟩⟩
def rawTerms : List Term := Proof.Events403.exact103377RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 103377
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult103377.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 103374) (rightBinding := 103375)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨17306⟩) (rightExpression := ⟨15979⟩)
    (transferEvent := 103376)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult103373.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult103188.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult103377

namespace SemanticResult103381
def owner : Owner := ⟨.program ⟨214⟩, ⟨17308⟩⟩
def rawTerms : List Term := Proof.Events403.exact103381RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 103381
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult103381.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 103378) (rightBinding := 103379)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨17307⟩) (rightExpression := ⟨16098⟩)
    (transferEvent := 103380)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult103377.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult103165.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult103381

namespace SemanticResult103385
def owner : Owner := ⟨.program ⟨214⟩, ⟨18304⟩⟩
def rawTerms : List Term := Proof.Events403.exact103385RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 103385
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult103385.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 103382) (rightBinding := 103383)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨17308⟩) (rightExpression := ⟨18303⟩)
    (transferEvent := 103384)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult103381.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult103142.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult103385

namespace SemanticResult103389
def owner : Owner := ⟨.program ⟨214⟩, ⟨18305⟩⟩
def rawTerms : List Term := Proof.Events403.exact103389RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 103389
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult103389.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 103386) (rightBinding := 103387)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨18304⟩) (rightExpression := ⟨16301⟩)
    (transferEvent := 103388)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult103385.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult103119.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult103389

namespace SemanticResult103393
def owner : Owner := ⟨.program ⟨214⟩, ⟨18306⟩⟩
def rawTerms : List Term := Proof.Events403.exact103393RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 103393
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult103393.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 103390) (rightBinding := 103391)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨18305⟩) (rightExpression := ⟨17113⟩)
    (transferEvent := 103392)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult103389.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult103096.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult103393

namespace SemanticResult103397
def owner : Owner := ⟨.program ⟨214⟩, ⟨18307⟩⟩
def rawTerms : List Term := Proof.Events403.exact103397RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 103397
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult103397.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 103394) (rightBinding := 103395)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨18306⟩) (rightExpression := ⟨17897⟩)
    (transferEvent := 103396)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult103393.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult103073.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult103397

namespace SemanticResult103401
def owner : Owner := ⟨.program ⟨214⟩, ⟨18308⟩⟩
def rawTerms : List Term := Proof.Events403.exact103401RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 103401
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult103401.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 103398) (rightBinding := 103399)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨18307⟩) (rightExpression := ⟨18198⟩)
    (transferEvent := 103400)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult103397.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult103050.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult103401

namespace SemanticResult103405
def owner : Owner := ⟨.program ⟨214⟩, ⟨18309⟩⟩
def rawTerms : List Term := Proof.Events403.exact103405RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 103405
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult103405.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 103402) (rightBinding := 103403)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨18308⟩) (rightExpression := ⟨16672⟩)
    (transferEvent := 103404)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult103401.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult103027.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult103405

namespace SemanticResult103409
def owner : Owner := ⟨.program ⟨214⟩, ⟨18310⟩⟩
def rawTerms : List Term := Proof.Events403.exact103409RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 103409
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult103409.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 103406) (rightBinding := 103407)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨18309⟩) (rightExpression := ⟨16791⟩)
    (transferEvent := 103408)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult103405.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult103004.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult103409

namespace SemanticResult103413
def owner : Owner := ⟨.program ⟨214⟩, ⟨18311⟩⟩
def rawTerms : List Term := Proof.Events403.exact103413RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 103413
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult103413.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 103410) (rightBinding := 103411)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨18310⟩) (rightExpression := ⟨17078⟩)
    (transferEvent := 103412)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult103409.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult102981.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult103413

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
