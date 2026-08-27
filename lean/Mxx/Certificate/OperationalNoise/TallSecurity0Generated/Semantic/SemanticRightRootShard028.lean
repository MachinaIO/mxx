import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticRightRootShard027

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open TallSecurity0ABI
open TallSemantics

namespace SemanticRightRootResult5502

def owner : Owner := ⟨.program ⟨214⟩, ⟨6744⟩⟩
def rawTerms : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6744⟩⟩] } }]
def summary : Bound := .exactZero
def producerEvent : Nat := 5501
def resultEvent : Nat := 5502
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual resultEvent
theorem terminalAt (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, rightRootOwnerAtSelector214 selector selectorLower selectorUpper _, ?_⟩
  refine ⟨.authority (.operator), 0, .authorityOperator, ?_, ?_⟩
  · rfl
  · rfl
theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness
    (terminalAt selector selectorLower selectorUpper)

end SemanticRightRootResult5502

namespace SemanticRightRootResult5507

def leftRaw : List Term := SemanticRightRootResult5502.rawTerms
def rightRaw : List Term := SemanticRightRootResult5499.rawTerms
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6744⟩⟩, ⟨.program ⟨214⟩, ⟨6651⟩⟩] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def owner : Owner := ⟨.program ⟨214⟩, ⟨7649⟩⟩
def rawTerms : List Term := outputRaw
def summary : Bound := .exactZero
def resultEvent : Nat := 5507
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
  SemanticRightRootResult5502.actual selector witness *
    SemanticRightRootResult5499.actual selector witness
theorem claimOfHistory (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841)
    (expressionAt : document.expressions.lookup 7649 =
      some Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Expression029.ExpressionRow7649)
    (leftPredecessorAt : (history.lookup 5503).map AnnotatedEvent.event =
      some (.predecessor owner 0 ⟨6744⟩ 5502))
    (rightPredecessorAt : (history.lookup 5504).map AnnotatedEvent.event =
      some (.predecessor owner 1 ⟨6652⟩ 5499))
    (ruleAt : (history.lookup 5505).map AnnotatedEvent.event =
      some (.boundTransfer owner (.product (.predecessor 0 5503 .coefficient) (.predecessor 1 5504 .coefficient) ⟨false, false, none, none, none⟩)))
    (leftClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult5502.resultEvent
      SemanticRightRootResult5502.owner
      (SemanticRightRootResult5502.actual selector witness)
      SemanticRightRootResult5502.rawTerms
      SemanticRightRootResult5502.summary)
    (rightClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult5499.resultEvent
      SemanticRightRootResult5499.owner
      (SemanticRightRootResult5499.actual selector witness)
      SemanticRightRootResult5499.rawTerms
      SemanticRightRootResult5499.summary)
    (outputAt : (history.lookup resultEvent).map AnnotatedEvent.event =
      some (.resultExact owner rawTerms summary)) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  refine ⟨outputAt, ?_⟩
  exact exactValueClaim_product_of_mod_zero 100418593683253592432016548326729029359133068138294319235841 witness.env
    (SemanticRightRootResult5502.actual selector witness)
    (SemanticRightRootResult5499.actual selector witness) left right output
    (by simpa [left, leftRaw, SemanticRightRootResult5502.summary] using leftClaim.claim)
    (by simpa [right, rightRaw, SemanticRightRootResult5499.summary] using rightClaim.claim)
    (resultSound witness.env) (by decide)

theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply claimOfHistory selector witness (by rfl) (by rfl) (by rfl) (by rfl)
  · exact SemanticRightRootResult5502.claimSound
      selector selectorLower selectorUpper witness
  · exact SemanticRightRootResult5499.claimSound
      selector selectorLower selectorUpper witness
  · rfl

end SemanticRightRootResult5507

namespace SemanticRightRootResult5519

def owner : Owner := ⟨.program ⟨214⟩, ⟨6658⟩⟩
def rawTerms : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6657⟩⟩] } }]
def summary : Bound := .exactZero
def producerEvent : Nat := 5518
def resultEvent : Nat := 5519
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual resultEvent
theorem terminalAt (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, rightRootOwnerAtSelector214 selector selectorLower selectorUpper _, ?_⟩
  refine ⟨.scale (.predecessor 0 5516 .coefficient) (.value (.predecessor 1 5517 .coefficient)), 0, .scale (.predecessor 0 5516 .coefficient) (.value (.predecessor 1 5517 .coefficient)), ?_, ?_⟩
  · rfl
  · rfl
theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness
    (terminalAt selector selectorLower selectorUpper)

end SemanticRightRootResult5519

namespace SemanticRightRootResult5522

def owner : Owner := ⟨.program ⟨214⟩, ⟨6742⟩⟩
def rawTerms : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6742⟩⟩] } }]
def summary : Bound := .exactZero
def producerEvent : Nat := 5521
def resultEvent : Nat := 5522
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual resultEvent
theorem terminalAt (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, rightRootOwnerAtSelector214 selector selectorLower selectorUpper _, ?_⟩
  refine ⟨.authority (.operator), 0, .authorityOperator, ?_, ?_⟩
  · rfl
  · rfl
theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness
    (terminalAt selector selectorLower selectorUpper)

end SemanticRightRootResult5522

namespace SemanticRightRootResult5527

def leftRaw : List Term := SemanticRightRootResult5522.rawTerms
def rightRaw : List Term := SemanticRightRootResult5519.rawTerms
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6742⟩⟩, ⟨.program ⟨214⟩, ⟨6657⟩⟩] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def owner : Owner := ⟨.program ⟨214⟩, ⟨7648⟩⟩
def rawTerms : List Term := outputRaw
def summary : Bound := .exactZero
def resultEvent : Nat := 5527
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
  SemanticRightRootResult5522.actual selector witness *
    SemanticRightRootResult5519.actual selector witness
theorem claimOfHistory (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841)
    (expressionAt : document.expressions.lookup 7648 =
      some Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Expression029.ExpressionRow7648)
    (leftPredecessorAt : (history.lookup 5523).map AnnotatedEvent.event =
      some (.predecessor owner 0 ⟨6742⟩ 5522))
    (rightPredecessorAt : (history.lookup 5524).map AnnotatedEvent.event =
      some (.predecessor owner 1 ⟨6658⟩ 5519))
    (ruleAt : (history.lookup 5525).map AnnotatedEvent.event =
      some (.boundTransfer owner (.product (.predecessor 0 5523 .coefficient) (.predecessor 1 5524 .coefficient) ⟨false, false, none, none, none⟩)))
    (leftClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult5522.resultEvent
      SemanticRightRootResult5522.owner
      (SemanticRightRootResult5522.actual selector witness)
      SemanticRightRootResult5522.rawTerms
      SemanticRightRootResult5522.summary)
    (rightClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult5519.resultEvent
      SemanticRightRootResult5519.owner
      (SemanticRightRootResult5519.actual selector witness)
      SemanticRightRootResult5519.rawTerms
      SemanticRightRootResult5519.summary)
    (outputAt : (history.lookup resultEvent).map AnnotatedEvent.event =
      some (.resultExact owner rawTerms summary)) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  refine ⟨outputAt, ?_⟩
  exact exactValueClaim_product_of_mod_zero 100418593683253592432016548326729029359133068138294319235841 witness.env
    (SemanticRightRootResult5522.actual selector witness)
    (SemanticRightRootResult5519.actual selector witness) left right output
    (by simpa [left, leftRaw, SemanticRightRootResult5522.summary] using leftClaim.claim)
    (by simpa [right, rightRaw, SemanticRightRootResult5519.summary] using rightClaim.claim)
    (resultSound witness.env) (by decide)

theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply claimOfHistory selector witness (by rfl) (by rfl) (by rfl) (by rfl)
  · exact SemanticRightRootResult5522.claimSound
      selector selectorLower selectorUpper witness
  · exact SemanticRightRootResult5519.claimSound
      selector selectorLower selectorUpper witness
  · rfl

end SemanticRightRootResult5527

namespace SemanticRightRootResult5539

def owner : Owner := ⟨.program ⟨214⟩, ⟨6660⟩⟩
def rawTerms : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6659⟩⟩] } }]
def summary : Bound := .exactZero
def producerEvent : Nat := 5538
def resultEvent : Nat := 5539
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual resultEvent
theorem terminalAt (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, rightRootOwnerAtSelector214 selector selectorLower selectorUpper _, ?_⟩
  refine ⟨.scale (.predecessor 0 5536 .coefficient) (.value (.predecessor 1 5537 .coefficient)), 0, .scale (.predecessor 0 5536 .coefficient) (.value (.predecessor 1 5537 .coefficient)), ?_, ?_⟩
  · rfl
  · rfl
theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness
    (terminalAt selector selectorLower selectorUpper)

end SemanticRightRootResult5539

namespace SemanticRightRootResult5542

def owner : Owner := ⟨.program ⟨214⟩, ⟨6740⟩⟩
def rawTerms : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6740⟩⟩] } }]
def summary : Bound := .exactZero
def producerEvent : Nat := 5541
def resultEvent : Nat := 5542
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual resultEvent
theorem terminalAt (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, rightRootOwnerAtSelector214 selector selectorLower selectorUpper _, ?_⟩
  refine ⟨.authority (.operator), 0, .authorityOperator, ?_, ?_⟩
  · rfl
  · rfl
theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness
    (terminalAt selector selectorLower selectorUpper)

end SemanticRightRootResult5542

namespace SemanticRightRootResult5547

def leftRaw : List Term := SemanticRightRootResult5542.rawTerms
def rightRaw : List Term := SemanticRightRootResult5539.rawTerms
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6740⟩⟩, ⟨.program ⟨214⟩, ⟨6659⟩⟩] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def owner : Owner := ⟨.program ⟨214⟩, ⟨7647⟩⟩
def rawTerms : List Term := outputRaw
def summary : Bound := .exactZero
def resultEvent : Nat := 5547
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
  SemanticRightRootResult5542.actual selector witness *
    SemanticRightRootResult5539.actual selector witness
theorem claimOfHistory (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841)
    (expressionAt : document.expressions.lookup 7647 =
      some Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Expression029.ExpressionRow7647)
    (leftPredecessorAt : (history.lookup 5543).map AnnotatedEvent.event =
      some (.predecessor owner 0 ⟨6740⟩ 5542))
    (rightPredecessorAt : (history.lookup 5544).map AnnotatedEvent.event =
      some (.predecessor owner 1 ⟨6660⟩ 5539))
    (ruleAt : (history.lookup 5545).map AnnotatedEvent.event =
      some (.boundTransfer owner (.product (.predecessor 0 5543 .coefficient) (.predecessor 1 5544 .coefficient) ⟨false, false, none, none, none⟩)))
    (leftClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult5542.resultEvent
      SemanticRightRootResult5542.owner
      (SemanticRightRootResult5542.actual selector witness)
      SemanticRightRootResult5542.rawTerms
      SemanticRightRootResult5542.summary)
    (rightClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult5539.resultEvent
      SemanticRightRootResult5539.owner
      (SemanticRightRootResult5539.actual selector witness)
      SemanticRightRootResult5539.rawTerms
      SemanticRightRootResult5539.summary)
    (outputAt : (history.lookup resultEvent).map AnnotatedEvent.event =
      some (.resultExact owner rawTerms summary)) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  refine ⟨outputAt, ?_⟩
  exact exactValueClaim_product_of_mod_zero 100418593683253592432016548326729029359133068138294319235841 witness.env
    (SemanticRightRootResult5542.actual selector witness)
    (SemanticRightRootResult5539.actual selector witness) left right output
    (by simpa [left, leftRaw, SemanticRightRootResult5542.summary] using leftClaim.claim)
    (by simpa [right, rightRaw, SemanticRightRootResult5539.summary] using rightClaim.claim)
    (resultSound witness.env) (by decide)

theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply claimOfHistory selector witness (by rfl) (by rfl) (by rfl) (by rfl)
  · exact SemanticRightRootResult5542.claimSound
      selector selectorLower selectorUpper witness
  · exact SemanticRightRootResult5539.claimSound
      selector selectorLower selectorUpper witness
  · rfl

end SemanticRightRootResult5547

namespace SemanticRightRootResult5559

def owner : Owner := ⟨.program ⟨214⟩, ⟨6662⟩⟩
def rawTerms : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6661⟩⟩] } }]
def summary : Bound := .exactZero
def producerEvent : Nat := 5558
def resultEvent : Nat := 5559
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual resultEvent
theorem terminalAt (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, rightRootOwnerAtSelector214 selector selectorLower selectorUpper _, ?_⟩
  refine ⟨.scale (.predecessor 0 5556 .coefficient) (.value (.predecessor 1 5557 .coefficient)), 0, .scale (.predecessor 0 5556 .coefficient) (.value (.predecessor 1 5557 .coefficient)), ?_, ?_⟩
  · rfl
  · rfl
theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness
    (terminalAt selector selectorLower selectorUpper)

end SemanticRightRootResult5559

namespace SemanticRightRootResult5562

def owner : Owner := ⟨.program ⟨214⟩, ⟨6738⟩⟩
def rawTerms : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6738⟩⟩] } }]
def summary : Bound := .exactZero
def producerEvent : Nat := 5561
def resultEvent : Nat := 5562
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual resultEvent
theorem terminalAt (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, rightRootOwnerAtSelector214 selector selectorLower selectorUpper _, ?_⟩
  refine ⟨.authority (.operator), 0, .authorityOperator, ?_, ?_⟩
  · rfl
  · rfl
theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness
    (terminalAt selector selectorLower selectorUpper)

end SemanticRightRootResult5562

namespace SemanticRightRootResult5567

def leftRaw : List Term := SemanticRightRootResult5562.rawTerms
def rightRaw : List Term := SemanticRightRootResult5559.rawTerms
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6738⟩⟩, ⟨.program ⟨214⟩, ⟨6661⟩⟩] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def owner : Owner := ⟨.program ⟨214⟩, ⟨7646⟩⟩
def rawTerms : List Term := outputRaw
def summary : Bound := .exactZero
def resultEvent : Nat := 5567
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
  SemanticRightRootResult5562.actual selector witness *
    SemanticRightRootResult5559.actual selector witness
theorem claimOfHistory (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841)
    (expressionAt : document.expressions.lookup 7646 =
      some Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Expression029.ExpressionRow7646)
    (leftPredecessorAt : (history.lookup 5563).map AnnotatedEvent.event =
      some (.predecessor owner 0 ⟨6738⟩ 5562))
    (rightPredecessorAt : (history.lookup 5564).map AnnotatedEvent.event =
      some (.predecessor owner 1 ⟨6662⟩ 5559))
    (ruleAt : (history.lookup 5565).map AnnotatedEvent.event =
      some (.boundTransfer owner (.product (.predecessor 0 5563 .coefficient) (.predecessor 1 5564 .coefficient) ⟨false, false, none, none, none⟩)))
    (leftClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult5562.resultEvent
      SemanticRightRootResult5562.owner
      (SemanticRightRootResult5562.actual selector witness)
      SemanticRightRootResult5562.rawTerms
      SemanticRightRootResult5562.summary)
    (rightClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult5559.resultEvent
      SemanticRightRootResult5559.owner
      (SemanticRightRootResult5559.actual selector witness)
      SemanticRightRootResult5559.rawTerms
      SemanticRightRootResult5559.summary)
    (outputAt : (history.lookup resultEvent).map AnnotatedEvent.event =
      some (.resultExact owner rawTerms summary)) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  refine ⟨outputAt, ?_⟩
  exact exactValueClaim_product_of_mod_zero 100418593683253592432016548326729029359133068138294319235841 witness.env
    (SemanticRightRootResult5562.actual selector witness)
    (SemanticRightRootResult5559.actual selector witness) left right output
    (by simpa [left, leftRaw, SemanticRightRootResult5562.summary] using leftClaim.claim)
    (by simpa [right, rightRaw, SemanticRightRootResult5559.summary] using rightClaim.claim)
    (resultSound witness.env) (by decide)

theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply claimOfHistory selector witness (by rfl) (by rfl) (by rfl) (by rfl)
  · exact SemanticRightRootResult5562.claimSound
      selector selectorLower selectorUpper witness
  · exact SemanticRightRootResult5559.claimSound
      selector selectorLower selectorUpper witness
  · rfl

end SemanticRightRootResult5567

namespace SemanticRightRootResult5579

def owner : Owner := ⟨.program ⟨214⟩, ⟨6666⟩⟩
def rawTerms : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6665⟩⟩] } }]
def summary : Bound := .exactZero
def producerEvent : Nat := 5578
def resultEvent : Nat := 5579
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual resultEvent
theorem terminalAt (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, rightRootOwnerAtSelector214 selector selectorLower selectorUpper _, ?_⟩
  refine ⟨.scale (.predecessor 0 5576 .coefficient) (.value (.predecessor 1 5577 .coefficient)), 0, .scale (.predecessor 0 5576 .coefficient) (.value (.predecessor 1 5577 .coefficient)), ?_, ?_⟩
  · rfl
  · rfl
theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness
    (terminalAt selector selectorLower selectorUpper)

end SemanticRightRootResult5579

namespace SemanticRightRootResult5582

def owner : Owner := ⟨.program ⟨214⟩, ⟨6736⟩⟩
def rawTerms : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6736⟩⟩] } }]
def summary : Bound := .exactZero
def producerEvent : Nat := 5581
def resultEvent : Nat := 5582
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual resultEvent
theorem terminalAt (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, rightRootOwnerAtSelector214 selector selectorLower selectorUpper _, ?_⟩
  refine ⟨.authority (.operator), 0, .authorityOperator, ?_, ?_⟩
  · rfl
  · rfl
theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness
    (terminalAt selector selectorLower selectorUpper)

end SemanticRightRootResult5582

namespace SemanticRightRootResult5587

def leftRaw : List Term := SemanticRightRootResult5582.rawTerms
def rightRaw : List Term := SemanticRightRootResult5579.rawTerms
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6736⟩⟩, ⟨.program ⟨214⟩, ⟨6665⟩⟩] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def owner : Owner := ⟨.program ⟨214⟩, ⟨7645⟩⟩
def rawTerms : List Term := outputRaw
def summary : Bound := .exactZero
def resultEvent : Nat := 5587
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
  SemanticRightRootResult5582.actual selector witness *
    SemanticRightRootResult5579.actual selector witness
theorem claimOfHistory (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841)
    (expressionAt : document.expressions.lookup 7645 =
      some Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Expression029.ExpressionRow7645)
    (leftPredecessorAt : (history.lookup 5583).map AnnotatedEvent.event =
      some (.predecessor owner 0 ⟨6736⟩ 5582))
    (rightPredecessorAt : (history.lookup 5584).map AnnotatedEvent.event =
      some (.predecessor owner 1 ⟨6666⟩ 5579))
    (ruleAt : (history.lookup 5585).map AnnotatedEvent.event =
      some (.boundTransfer owner (.product (.predecessor 0 5583 .coefficient) (.predecessor 1 5584 .coefficient) ⟨false, false, none, none, none⟩)))
    (leftClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult5582.resultEvent
      SemanticRightRootResult5582.owner
      (SemanticRightRootResult5582.actual selector witness)
      SemanticRightRootResult5582.rawTerms
      SemanticRightRootResult5582.summary)
    (rightClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult5579.resultEvent
      SemanticRightRootResult5579.owner
      (SemanticRightRootResult5579.actual selector witness)
      SemanticRightRootResult5579.rawTerms
      SemanticRightRootResult5579.summary)
    (outputAt : (history.lookup resultEvent).map AnnotatedEvent.event =
      some (.resultExact owner rawTerms summary)) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  refine ⟨outputAt, ?_⟩
  exact exactValueClaim_product_of_mod_zero 100418593683253592432016548326729029359133068138294319235841 witness.env
    (SemanticRightRootResult5582.actual selector witness)
    (SemanticRightRootResult5579.actual selector witness) left right output
    (by simpa [left, leftRaw, SemanticRightRootResult5582.summary] using leftClaim.claim)
    (by simpa [right, rightRaw, SemanticRightRootResult5579.summary] using rightClaim.claim)
    (resultSound witness.env) (by decide)

theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply claimOfHistory selector witness (by rfl) (by rfl) (by rfl) (by rfl)
  · exact SemanticRightRootResult5582.claimSound
      selector selectorLower selectorUpper witness
  · exact SemanticRightRootResult5579.claimSound
      selector selectorLower selectorUpper witness
  · rfl

end SemanticRightRootResult5587

namespace SemanticRightRootResult5599

def owner : Owner := ⟨.program ⟨214⟩, ⟨6668⟩⟩
def rawTerms : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6667⟩⟩] } }]
def summary : Bound := .exactZero
def producerEvent : Nat := 5598
def resultEvent : Nat := 5599
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual resultEvent
theorem terminalAt (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, rightRootOwnerAtSelector214 selector selectorLower selectorUpper _, ?_⟩
  refine ⟨.scale (.predecessor 0 5596 .coefficient) (.value (.predecessor 1 5597 .coefficient)), 0, .scale (.predecessor 0 5596 .coefficient) (.value (.predecessor 1 5597 .coefficient)), ?_, ?_⟩
  · rfl
  · rfl
theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness
    (terminalAt selector selectorLower selectorUpper)

end SemanticRightRootResult5599

namespace SemanticRightRootResult5602

def owner : Owner := ⟨.program ⟨214⟩, ⟨6734⟩⟩
def rawTerms : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6734⟩⟩] } }]
def summary : Bound := .exactZero
def producerEvent : Nat := 5601
def resultEvent : Nat := 5602
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual resultEvent
theorem terminalAt (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, rightRootOwnerAtSelector214 selector selectorLower selectorUpper _, ?_⟩
  refine ⟨.authority (.operator), 0, .authorityOperator, ?_, ?_⟩
  · rfl
  · rfl
theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness
    (terminalAt selector selectorLower selectorUpper)

end SemanticRightRootResult5602

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
