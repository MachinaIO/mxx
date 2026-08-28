import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard233
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard231
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard232

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult31307
def owner : Owner := ⟨.program ⟨214⟩, ⟨17357⟩⟩
def rawTerms : List Term := Proof.Events122.exact31307RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 31307
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult31307.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 31304) (rightBinding := 31305)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨17356⟩) (rightExpression := ⟨15757⟩)
    (transferEvent := 31306)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult31303.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult31172.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult31307

namespace SemanticResult31311
def owner : Owner := ⟨.program ⟨214⟩, ⟨17358⟩⟩
def rawTerms : List Term := Proof.Events122.exact31311RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 31311
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult31311.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 31308) (rightBinding := 31309)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨17357⟩) (rightExpression := ⟨15876⟩)
    (transferEvent := 31310)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult31307.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult31149.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult31311

namespace SemanticResult31315
def owner : Owner := ⟨.program ⟨214⟩, ⟨17359⟩⟩
def rawTerms : List Term := Proof.Events122.exact31315RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 31315
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult31315.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 31312) (rightBinding := 31313)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨17358⟩) (rightExpression := ⟨15995⟩)
    (transferEvent := 31314)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult31311.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult31126.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult31315

namespace SemanticResult31319
def owner : Owner := ⟨.program ⟨214⟩, ⟨17360⟩⟩
def rawTerms : List Term := Proof.Events122.exact31319RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 31319
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult31319.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 31316) (rightBinding := 31317)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨17359⟩) (rightExpression := ⟨16114⟩)
    (transferEvent := 31318)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult31315.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult31103.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult31319

namespace SemanticResult31323
def owner : Owner := ⟨.program ⟨214⟩, ⟨18380⟩⟩
def rawTerms : List Term := Proof.Events122.exact31323RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 31323
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult31323.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 31320) (rightBinding := 31321)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨17360⟩) (rightExpression := ⟨18379⟩)
    (transferEvent := 31322)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult31319.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult31080.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult31323

namespace SemanticResult31327
def owner : Owner := ⟨.program ⟨214⟩, ⟨18381⟩⟩
def rawTerms : List Term := Proof.Events122.exact31327RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 31327
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult31327.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 31324) (rightBinding := 31325)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨18380⟩) (rightExpression := ⟨16317⟩)
    (transferEvent := 31326)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult31323.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult31057.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult31327

namespace SemanticResult31331
def owner : Owner := ⟨.program ⟨214⟩, ⟨18382⟩⟩
def rawTerms : List Term := Proof.Events122.exact31331RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 31331
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult31331.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 31328) (rightBinding := 31329)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨18381⟩) (rightExpression := ⟨17129⟩)
    (transferEvent := 31330)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult31327.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult31034.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult31331

namespace SemanticResult31335
def owner : Owner := ⟨.program ⟨214⟩, ⟨18383⟩⟩
def rawTerms : List Term := Proof.Events122.exact31335RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 31335
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult31335.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 31332) (rightBinding := 31333)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨18382⟩) (rightExpression := ⟨17913⟩)
    (transferEvent := 31334)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult31331.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult31011.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult31335

namespace SemanticResult31339
def owner : Owner := ⟨.program ⟨214⟩, ⟨18384⟩⟩
def rawTerms : List Term := Proof.Events122.exact31339RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 31339
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult31339.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 31336) (rightBinding := 31337)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨18383⟩) (rightExpression := ⟨18214⟩)
    (transferEvent := 31338)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult31335.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult30988.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult31339

namespace SemanticResult31343
def owner : Owner := ⟨.program ⟨214⟩, ⟨18385⟩⟩
def rawTerms : List Term := Proof.Events122.exact31343RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 31343
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult31343.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 31340) (rightBinding := 31341)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨18384⟩) (rightExpression := ⟨16688⟩)
    (transferEvent := 31342)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult31339.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult30965.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult31343

namespace SemanticResult31347
def owner : Owner := ⟨.program ⟨214⟩, ⟨18386⟩⟩
def rawTerms : List Term := Proof.Events122.exact31347RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 31347
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult31347.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 31344) (rightBinding := 31345)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨18385⟩) (rightExpression := ⟨16807⟩)
    (transferEvent := 31346)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult31343.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult30942.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult31347

namespace SemanticResult31351
def owner : Owner := ⟨.program ⟨214⟩, ⟨18387⟩⟩
def rawTerms : List Term := Proof.Events122.exact31351RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 31351
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult31351.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 31348) (rightBinding := 31349)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨18386⟩) (rightExpression := ⟨17094⟩)
    (transferEvent := 31350)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult31347.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult30919.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult31351

namespace SemanticResult31355
def owner : Owner := ⟨.program ⟨214⟩, ⟨18388⟩⟩
def rawTerms : List Term := Proof.Events122.exact31355RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 31355
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult31355.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 31352) (rightBinding := 31353)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨18387⟩) (rightExpression := ⟨18179⟩)
    (transferEvent := 31354)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult31351.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult30896.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult31355

namespace SemanticResult31366
def owner : Owner := ⟨.program ⟨214⟩, ⟨18624⟩⟩
def rawTerms : List Term := Proof.Events122.exact31366RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 31366
def producerEvent : Nat := 31365
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult31366.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.operator), 30853, .large, .authorityOperator, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult31366

namespace SemanticResult31369
def owner : Owner := ⟨.program ⟨214⟩, ⟨18690⟩⟩
def rawTerms : List Term := Proof.Events122.exact31369RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 31369
def producerEvent : Nat := 31368
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult31369.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.operator), 30853, .finite 8192, .authorityOperator, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult31369

namespace SemanticResult31378
def owner : Owner := ⟨.program ⟨214⟩, ⟨18660⟩⟩
def rawTerms : List Term := Proof.Events122.exact31378RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 31378
def producerEvent : Nat := 31377
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult31378.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.identity (.predecessor 0 31376 .coefficient), 30853, .finite 1059, .identity (.predecessor 0 31376 .coefficient), ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult31378

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
