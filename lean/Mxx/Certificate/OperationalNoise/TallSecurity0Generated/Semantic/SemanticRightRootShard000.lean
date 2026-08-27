import Mxx.Certificate.OperationalNoise.TallSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open TallSecurity0ABI
open TallSemantics

theorem rightRootOwnerAtSelector214 (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32) (expression : SchemaV1.ExpressionRef) :
    ownerAtSelector document (some selector) ⟨.program ⟨214⟩, expression⟩ := by
  simp [ownerAtSelector, document, selectorLower, selectorUpper]

namespace SemanticRightRootResult2

def owner : Owner := ⟨.program ⟨214⟩, ⟨6544⟩⟩
def rawTerms : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def summary : Bound := .exactZero
def producerEvent : Nat := 1
def resultEvent : Nat := 2
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual resultEvent
theorem terminalAt (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, rightRootOwnerAtSelector214 selector selectorLower selectorUpper _, ?_⟩
  refine ⟨.authority (.factStore), 0, .authorityFactStore, ?_, ?_⟩
  · rfl
  · rfl
theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness
    (terminalAt selector selectorLower selectorUpper)

end SemanticRightRootResult2

namespace SemanticRightRootResult27

def owner : Owner := ⟨.program ⟨214⟩, ⟨5506⟩⟩
def rawTerms : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩] } }]
def summary : Bound := .exactZero
def producerEvent : Nat := 26
def resultEvent : Nat := 27
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual resultEvent
theorem terminalAt (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, rightRootOwnerAtSelector214 selector selectorLower selectorUpper _, ?_⟩
  refine ⟨.identity (.predecessor 0 21 .coefficient), 0, .identity (.predecessor 0 21 .coefficient), ?_, ?_⟩
  · rfl
  · rfl
theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness
    (terminalAt selector selectorLower selectorUpper)

end SemanticRightRootResult27

namespace SemanticRightRootResult32

def leftRaw : List Term := SemanticRightRootResult27.rawTerms
def rightRaw : List Term := SemanticRightRootResult2.rawTerms
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def owner : Owner := ⟨.program ⟨214⟩, ⟨6564⟩⟩
def rawTerms : List Term := outputRaw
def summary : Bound := .exactZero
def resultEvent : Nat := 32
def leftScalar : Bool := true
def rightScalar : Bool := false
theorem resultAgreement : CanonicalAgreement output
    (productPoly left right leftScalar rightScalar) := by
  decide +kernel

theorem resultSound (env : Env Owner) :
    evalPolynomial env output = evalPolynomial env left * evalPolynomial env right := by
  exact productCanonicalResultSound env left right output leftScalar rightScalar resultAgreement

def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  SemanticRightRootResult27.actual selector witness *
    SemanticRightRootResult2.actual selector witness
theorem claimOfHistory (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841)
    (expressionAt : document.expressions.lookup 6564 =
      some Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Expression025.ExpressionRow6564)
    (leftPredecessorAt : (history.lookup 28).map AnnotatedEvent.event =
      some (.predecessor owner 0 ⟨5506⟩ 27))
    (rightPredecessorAt : (history.lookup 29).map AnnotatedEvent.event =
      some (.predecessor owner 1 ⟨6544⟩ 2))
    (ruleAt : (history.lookup 30).map AnnotatedEvent.event =
      some (.boundTransfer owner (.product (.predecessor 0 28 .coefficient) (.predecessor 1 29 .coefficient) ⟨false, false, none, none, none⟩)))
    (leftClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult27.resultEvent
      SemanticRightRootResult27.owner
      (SemanticRightRootResult27.actual selector witness)
      SemanticRightRootResult27.rawTerms
      SemanticRightRootResult27.summary)
    (rightClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult2.resultEvent
      SemanticRightRootResult2.owner
      (SemanticRightRootResult2.actual selector witness)
      SemanticRightRootResult2.rawTerms
      SemanticRightRootResult2.summary)
    (outputAt : (history.lookup resultEvent).map AnnotatedEvent.event =
      some (.resultExact owner rawTerms summary)) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  refine ⟨outputAt, ?_⟩
  exact exactValueClaim_product_of_mod_zero 100418593683253592432016548326729029359133068138294319235841 witness.env
    (SemanticRightRootResult27.actual selector witness)
    (SemanticRightRootResult2.actual selector witness) left right output
    (by simpa [left, leftRaw, SemanticRightRootResult27.summary] using leftClaim.claim)
    (by simpa [right, rightRaw, SemanticRightRootResult2.summary] using rightClaim.claim)
    (resultSound witness.env) (by decide)

theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply claimOfHistory selector witness (by rfl) (by rfl) (by rfl) (by rfl)
  · exact SemanticRightRootResult27.claimSound
      selector selectorLower selectorUpper witness
  · exact SemanticRightRootResult2.claimSound
      selector selectorLower selectorUpper witness
  · rfl

end SemanticRightRootResult32

namespace SemanticRightRootResult34

def owner : Owner := ⟨.program ⟨214⟩, ⟨6396⟩⟩
def rawTerms : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6396⟩⟩], orderedFactors := [] } }]
def summary : Bound := .exactZero
def producerEvent : Nat := 33
def resultEvent : Nat := 34
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual resultEvent
theorem terminalAt (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, rightRootOwnerAtSelector214 selector selectorLower selectorUpper _, ?_⟩
  refine ⟨.authority (.factStore), 0, .authorityFactStore, ?_, ?_⟩
  · rfl
  · rfl
theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness
    (terminalAt selector selectorLower selectorUpper)

end SemanticRightRootResult34

namespace SemanticRightRootResult36

def owner : Owner := ⟨.program ⟨214⟩, ⟨6410⟩⟩
def rawTerms : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6410⟩⟩], orderedFactors := [] } }]
def summary : Bound := .exactZero
def producerEvent : Nat := 35
def resultEvent : Nat := 36
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual resultEvent
theorem terminalAt (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, rightRootOwnerAtSelector214 selector selectorLower selectorUpper _, ?_⟩
  refine ⟨.authority (.factStore), 0, .authorityFactStore, ?_, ?_⟩
  · rfl
  · rfl
theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness
    (terminalAt selector selectorLower selectorUpper)

end SemanticRightRootResult36

namespace SemanticRightRootResult536

def owner : Owner := ⟨.program ⟨214⟩, ⟨18511⟩⟩
def rawTerms : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18511⟩⟩], orderedFactors := [] } }]
def summary : Bound := .exactZero
def producerEvent : Nat := 535
def resultEvent : Nat := 536
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual resultEvent
theorem terminalAt (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, rightRootOwnerAtSelector214 selector selectorLower selectorUpper _, ?_⟩
  refine ⟨.authority (.programFamilyFact), 0, .authorityProgramFamilyFact, ?_, ?_⟩
  · rfl
  · rfl
theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness
    (terminalAt selector selectorLower selectorUpper)

end SemanticRightRootResult536

namespace SemanticRightRootResult541

def leftRaw : List Term := SemanticRightRootResult536.rawTerms
def rightRaw : List Term := SemanticRightRootResult36.rawTerms
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6410⟩⟩, ⟨.program ⟨214⟩, ⟨18511⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def owner : Owner := ⟨.program ⟨214⟩, ⟨18512⟩⟩
def rawTerms : List Term := outputRaw
def summary : Bound := .exactZero
def resultEvent : Nat := 541
def leftScalar : Bool := false
def rightScalar : Bool := false
theorem resultAgreement : CanonicalAgreement output
    (productPoly left right leftScalar rightScalar) := by
  decide +kernel

theorem resultSound (env : Env Owner) :
    evalPolynomial env output = evalPolynomial env left * evalPolynomial env right := by
  exact productCanonicalResultSound env left right output leftScalar rightScalar resultAgreement

def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  SemanticRightRootResult536.actual selector witness *
    SemanticRightRootResult36.actual selector witness
theorem claimOfHistory (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841)
    (expressionAt : document.expressions.lookup 18512 =
      some Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Expression072.ExpressionRow18512)
    (leftPredecessorAt : (history.lookup 537).map AnnotatedEvent.event =
      some (.predecessor owner 0 ⟨18511⟩ 536))
    (rightPredecessorAt : (history.lookup 538).map AnnotatedEvent.event =
      some (.predecessor owner 1 ⟨6410⟩ 36))
    (ruleAt : (history.lookup 539).map AnnotatedEvent.event =
      some (.boundTransfer owner (.product (.predecessor 0 537 .coefficient) (.predecessor 1 538 .coefficient) ⟨true, true, none, some 1, some 1⟩)))
    (leftClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult536.resultEvent
      SemanticRightRootResult536.owner
      (SemanticRightRootResult536.actual selector witness)
      SemanticRightRootResult536.rawTerms
      SemanticRightRootResult536.summary)
    (rightClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult36.resultEvent
      SemanticRightRootResult36.owner
      (SemanticRightRootResult36.actual selector witness)
      SemanticRightRootResult36.rawTerms
      SemanticRightRootResult36.summary)
    (outputAt : (history.lookup resultEvent).map AnnotatedEvent.event =
      some (.resultExact owner rawTerms summary)) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  refine ⟨outputAt, ?_⟩
  exact exactValueClaim_product_of_mod_zero 100418593683253592432016548326729029359133068138294319235841 witness.env
    (SemanticRightRootResult536.actual selector witness)
    (SemanticRightRootResult36.actual selector witness) left right output
    (by simpa [left, leftRaw, SemanticRightRootResult536.summary] using leftClaim.claim)
    (by simpa [right, rightRaw, SemanticRightRootResult36.summary] using rightClaim.claim)
    (resultSound witness.env) (by decide)

theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply claimOfHistory selector witness (by rfl) (by rfl) (by rfl) (by rfl)
  · exact SemanticRightRootResult536.claimSound
      selector selectorLower selectorUpper witness
  · exact SemanticRightRootResult36.claimSound
      selector selectorLower selectorUpper witness
  · rfl

end SemanticRightRootResult541

namespace SemanticRightRootResult543

def owner : Owner := ⟨.program ⟨214⟩, ⟨6435⟩⟩
def rawTerms : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6435⟩⟩], orderedFactors := [] } }]
def summary : Bound := .exactZero
def producerEvent : Nat := 542
def resultEvent : Nat := 543
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual resultEvent
theorem terminalAt (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, rightRootOwnerAtSelector214 selector selectorLower selectorUpper _, ?_⟩
  refine ⟨.authority (.factStore), 0, .authorityFactStore, ?_, ?_⟩
  · rfl
  · rfl
theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness
    (terminalAt selector selectorLower selectorUpper)

end SemanticRightRootResult543

namespace SemanticRightRootResult546

def owner : Owner := ⟨.program ⟨214⟩, ⟨18140⟩⟩
def rawTerms : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18140⟩⟩], orderedFactors := [] } }]
def summary : Bound := .exactZero
def producerEvent : Nat := 545
def resultEvent : Nat := 546
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual resultEvent
theorem terminalAt (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, rightRootOwnerAtSelector214 selector selectorLower selectorUpper _, ?_⟩
  refine ⟨.authority (.programFamilyFact), 0, .authorityProgramFamilyFact, ?_, ?_⟩
  · rfl
  · rfl
theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness
    (terminalAt selector selectorLower selectorUpper)

end SemanticRightRootResult546

namespace SemanticRightRootResult551

def leftRaw : List Term := SemanticRightRootResult546.rawTerms
def rightRaw : List Term := SemanticRightRootResult543.rawTerms
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6435⟩⟩, ⟨.program ⟨214⟩, ⟨18140⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def owner : Owner := ⟨.program ⟨214⟩, ⟨18141⟩⟩
def rawTerms : List Term := outputRaw
def summary : Bound := .exactZero
def resultEvent : Nat := 551
def leftScalar : Bool := false
def rightScalar : Bool := false
theorem resultAgreement : CanonicalAgreement output
    (productPoly left right leftScalar rightScalar) := by
  decide +kernel

theorem resultSound (env : Env Owner) :
    evalPolynomial env output = evalPolynomial env left * evalPolynomial env right := by
  exact productCanonicalResultSound env left right output leftScalar rightScalar resultAgreement

def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  SemanticRightRootResult546.actual selector witness *
    SemanticRightRootResult543.actual selector witness
theorem claimOfHistory (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841)
    (expressionAt : document.expressions.lookup 18141 =
      some Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Expression070.ExpressionRow18141)
    (leftPredecessorAt : (history.lookup 547).map AnnotatedEvent.event =
      some (.predecessor owner 0 ⟨18140⟩ 546))
    (rightPredecessorAt : (history.lookup 548).map AnnotatedEvent.event =
      some (.predecessor owner 1 ⟨6435⟩ 543))
    (ruleAt : (history.lookup 549).map AnnotatedEvent.event =
      some (.boundTransfer owner (.product (.predecessor 0 547 .coefficient) (.predecessor 1 548 .coefficient) ⟨true, true, none, some 1, some 1⟩)))
    (leftClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult546.resultEvent
      SemanticRightRootResult546.owner
      (SemanticRightRootResult546.actual selector witness)
      SemanticRightRootResult546.rawTerms
      SemanticRightRootResult546.summary)
    (rightClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult543.resultEvent
      SemanticRightRootResult543.owner
      (SemanticRightRootResult543.actual selector witness)
      SemanticRightRootResult543.rawTerms
      SemanticRightRootResult543.summary)
    (outputAt : (history.lookup resultEvent).map AnnotatedEvent.event =
      some (.resultExact owner rawTerms summary)) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  refine ⟨outputAt, ?_⟩
  exact exactValueClaim_product_of_mod_zero 100418593683253592432016548326729029359133068138294319235841 witness.env
    (SemanticRightRootResult546.actual selector witness)
    (SemanticRightRootResult543.actual selector witness) left right output
    (by simpa [left, leftRaw, SemanticRightRootResult546.summary] using leftClaim.claim)
    (by simpa [right, rightRaw, SemanticRightRootResult543.summary] using rightClaim.claim)
    (resultSound witness.env) (by decide)

theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply claimOfHistory selector witness (by rfl) (by rfl) (by rfl) (by rfl)
  · exact SemanticRightRootResult546.claimSound
      selector selectorLower selectorUpper witness
  · exact SemanticRightRootResult543.claimSound
      selector selectorLower selectorUpper witness
  · rfl

end SemanticRightRootResult551

namespace SemanticRightRootResult553

def owner : Owner := ⟨.program ⟨214⟩, ⟨6437⟩⟩
def rawTerms : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6437⟩⟩], orderedFactors := [] } }]
def summary : Bound := .exactZero
def producerEvent : Nat := 552
def resultEvent : Nat := 553
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual resultEvent
theorem terminalAt (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, rightRootOwnerAtSelector214 selector selectorLower selectorUpper _, ?_⟩
  refine ⟨.authority (.factStore), 0, .authorityFactStore, ?_, ?_⟩
  · rfl
  · rfl
theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness
    (terminalAt selector selectorLower selectorUpper)

end SemanticRightRootResult553

namespace SemanticRightRootResult556

def owner : Owner := ⟨.program ⟨214⟩, ⟨16943⟩⟩
def rawTerms : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16943⟩⟩], orderedFactors := [] } }]
def summary : Bound := .exactZero
def producerEvent : Nat := 555
def resultEvent : Nat := 556
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual resultEvent
theorem terminalAt (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, rightRootOwnerAtSelector214 selector selectorLower selectorUpper _, ?_⟩
  refine ⟨.authority (.programFamilyFact), 0, .authorityProgramFamilyFact, ?_, ?_⟩
  · rfl
  · rfl
theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness
    (terminalAt selector selectorLower selectorUpper)

end SemanticRightRootResult556

namespace SemanticRightRootResult561

def leftRaw : List Term := SemanticRightRootResult556.rawTerms
def rightRaw : List Term := SemanticRightRootResult553.rawTerms
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6437⟩⟩, ⟨.program ⟨214⟩, ⟨16943⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def owner : Owner := ⟨.program ⟨214⟩, ⟨16944⟩⟩
def rawTerms : List Term := outputRaw
def summary : Bound := .exactZero
def resultEvent : Nat := 561
def leftScalar : Bool := false
def rightScalar : Bool := false
theorem resultAgreement : CanonicalAgreement output
    (productPoly left right leftScalar rightScalar) := by
  decide +kernel

theorem resultSound (env : Env Owner) :
    evalPolynomial env output = evalPolynomial env left * evalPolynomial env right := by
  exact productCanonicalResultSound env left right output leftScalar rightScalar resultAgreement

def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  SemanticRightRootResult556.actual selector witness *
    SemanticRightRootResult553.actual selector witness
theorem claimOfHistory (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841)
    (expressionAt : document.expressions.lookup 16944 =
      some Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Expression066.ExpressionRow16944)
    (leftPredecessorAt : (history.lookup 557).map AnnotatedEvent.event =
      some (.predecessor owner 0 ⟨16943⟩ 556))
    (rightPredecessorAt : (history.lookup 558).map AnnotatedEvent.event =
      some (.predecessor owner 1 ⟨6437⟩ 553))
    (ruleAt : (history.lookup 559).map AnnotatedEvent.event =
      some (.boundTransfer owner (.product (.predecessor 0 557 .coefficient) (.predecessor 1 558 .coefficient) ⟨true, true, none, some 1, some 1⟩)))
    (leftClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult556.resultEvent
      SemanticRightRootResult556.owner
      (SemanticRightRootResult556.actual selector witness)
      SemanticRightRootResult556.rawTerms
      SemanticRightRootResult556.summary)
    (rightClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult553.resultEvent
      SemanticRightRootResult553.owner
      (SemanticRightRootResult553.actual selector witness)
      SemanticRightRootResult553.rawTerms
      SemanticRightRootResult553.summary)
    (outputAt : (history.lookup resultEvent).map AnnotatedEvent.event =
      some (.resultExact owner rawTerms summary)) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  refine ⟨outputAt, ?_⟩
  exact exactValueClaim_product_of_mod_zero 100418593683253592432016548326729029359133068138294319235841 witness.env
    (SemanticRightRootResult556.actual selector witness)
    (SemanticRightRootResult553.actual selector witness) left right output
    (by simpa [left, leftRaw, SemanticRightRootResult556.summary] using leftClaim.claim)
    (by simpa [right, rightRaw, SemanticRightRootResult553.summary] using rightClaim.claim)
    (resultSound witness.env) (by decide)

theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply claimOfHistory selector witness (by rfl) (by rfl) (by rfl) (by rfl)
  · exact SemanticRightRootResult556.claimSound
      selector selectorLower selectorUpper witness
  · exact SemanticRightRootResult553.claimSound
      selector selectorLower selectorUpper witness
  · rfl

end SemanticRightRootResult561

namespace SemanticRightRootResult563

def owner : Owner := ⟨.program ⟨214⟩, ⟨6449⟩⟩
def rawTerms : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6449⟩⟩], orderedFactors := [] } }]
def summary : Bound := .exactZero
def producerEvent : Nat := 562
def resultEvent : Nat := 563
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual resultEvent
theorem terminalAt (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, rightRootOwnerAtSelector214 selector selectorLower selectorUpper _, ?_⟩
  refine ⟨.authority (.factStore), 0, .authorityFactStore, ?_, ?_⟩
  · rfl
  · rfl
theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness
    (terminalAt selector selectorLower selectorUpper)

end SemanticRightRootResult563

namespace SemanticRightRootResult566

def owner : Owner := ⟨.program ⟨214⟩, ⟨17510⟩⟩
def rawTerms : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17510⟩⟩], orderedFactors := [] } }]
def summary : Bound := .exactZero
def producerEvent : Nat := 565
def resultEvent : Nat := 566
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual resultEvent
theorem terminalAt (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, rightRootOwnerAtSelector214 selector selectorLower selectorUpper _, ?_⟩
  refine ⟨.authority (.programFamilyFact), 0, .authorityProgramFamilyFact, ?_, ?_⟩
  · rfl
  · rfl
theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness
    (terminalAt selector selectorLower selectorUpper)

end SemanticRightRootResult566

namespace SemanticRightRootResult571

def leftRaw : List Term := SemanticRightRootResult566.rawTerms
def rightRaw : List Term := SemanticRightRootResult563.rawTerms
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6449⟩⟩, ⟨.program ⟨214⟩, ⟨17510⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def owner : Owner := ⟨.program ⟨214⟩, ⟨17511⟩⟩
def rawTerms : List Term := outputRaw
def summary : Bound := .exactZero
def resultEvent : Nat := 571
def leftScalar : Bool := false
def rightScalar : Bool := false
theorem resultAgreement : CanonicalAgreement output
    (productPoly left right leftScalar rightScalar) := by
  decide +kernel

theorem resultSound (env : Env Owner) :
    evalPolynomial env output = evalPolynomial env left * evalPolynomial env right := by
  exact productCanonicalResultSound env left right output leftScalar rightScalar resultAgreement

def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  SemanticRightRootResult566.actual selector witness *
    SemanticRightRootResult563.actual selector witness
theorem claimOfHistory (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841)
    (expressionAt : document.expressions.lookup 17511 =
      some Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Expression068.ExpressionRow17511)
    (leftPredecessorAt : (history.lookup 567).map AnnotatedEvent.event =
      some (.predecessor owner 0 ⟨17510⟩ 566))
    (rightPredecessorAt : (history.lookup 568).map AnnotatedEvent.event =
      some (.predecessor owner 1 ⟨6449⟩ 563))
    (ruleAt : (history.lookup 569).map AnnotatedEvent.event =
      some (.boundTransfer owner (.product (.predecessor 0 567 .coefficient) (.predecessor 1 568 .coefficient) ⟨true, true, none, some 1, some 1⟩)))
    (leftClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult566.resultEvent
      SemanticRightRootResult566.owner
      (SemanticRightRootResult566.actual selector witness)
      SemanticRightRootResult566.rawTerms
      SemanticRightRootResult566.summary)
    (rightClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult563.resultEvent
      SemanticRightRootResult563.owner
      (SemanticRightRootResult563.actual selector witness)
      SemanticRightRootResult563.rawTerms
      SemanticRightRootResult563.summary)
    (outputAt : (history.lookup resultEvent).map AnnotatedEvent.event =
      some (.resultExact owner rawTerms summary)) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  refine ⟨outputAt, ?_⟩
  exact exactValueClaim_product_of_mod_zero 100418593683253592432016548326729029359133068138294319235841 witness.env
    (SemanticRightRootResult566.actual selector witness)
    (SemanticRightRootResult563.actual selector witness) left right output
    (by simpa [left, leftRaw, SemanticRightRootResult566.summary] using leftClaim.claim)
    (by simpa [right, rightRaw, SemanticRightRootResult563.summary] using rightClaim.claim)
    (resultSound witness.env) (by decide)

theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply claimOfHistory selector witness (by rfl) (by rfl) (by rfl) (by rfl)
  · exact SemanticRightRootResult566.claimSound
      selector selectorLower selectorUpper witness
  · exact SemanticRightRootResult563.claimSound
      selector selectorLower selectorUpper witness
  · rfl

end SemanticRightRootResult571

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
