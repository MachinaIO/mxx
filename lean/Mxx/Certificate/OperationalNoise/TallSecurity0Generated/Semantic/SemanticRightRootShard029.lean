import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticRightRootShard028

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open TallSecurity0ABI
open TallSemantics

namespace SemanticRightRootResult5607

def leftRaw : List Term := SemanticRightRootResult5602.rawTerms
def rightRaw : List Term := SemanticRightRootResult5599.rawTerms
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6734⟩⟩, ⟨.program ⟨214⟩, ⟨6667⟩⟩] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def owner : Owner := ⟨.program ⟨214⟩, ⟨7644⟩⟩
def rawTerms : List Term := outputRaw
def summary : Bound := .exactZero
def resultEvent : Nat := 5607
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
  SemanticRightRootResult5602.actual selector witness *
    SemanticRightRootResult5599.actual selector witness
theorem claimOfHistory (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841)
    (expressionAt : document.expressions.lookup 7644 =
      some Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Expression029.ExpressionRow7644)
    (leftPredecessorAt : (history.lookup 5603).map AnnotatedEvent.event =
      some (.predecessor owner 0 ⟨6734⟩ 5602))
    (rightPredecessorAt : (history.lookup 5604).map AnnotatedEvent.event =
      some (.predecessor owner 1 ⟨6668⟩ 5599))
    (ruleAt : (history.lookup 5605).map AnnotatedEvent.event =
      some (.boundTransfer owner (.product (.predecessor 0 5603 .coefficient) (.predecessor 1 5604 .coefficient) ⟨false, false, none, none, none⟩)))
    (leftClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult5602.resultEvent
      SemanticRightRootResult5602.owner
      (SemanticRightRootResult5602.actual selector witness)
      SemanticRightRootResult5602.rawTerms
      SemanticRightRootResult5602.summary)
    (rightClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult5599.resultEvent
      SemanticRightRootResult5599.owner
      (SemanticRightRootResult5599.actual selector witness)
      SemanticRightRootResult5599.rawTerms
      SemanticRightRootResult5599.summary)
    (outputAt : (history.lookup resultEvent).map AnnotatedEvent.event =
      some (.resultExact owner rawTerms summary)) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  refine ⟨outputAt, ?_⟩
  exact exactValueClaim_product_of_mod_zero 100418593683253592432016548326729029359133068138294319235841 witness.env
    (SemanticRightRootResult5602.actual selector witness)
    (SemanticRightRootResult5599.actual selector witness) left right output
    (by simpa [left, leftRaw, SemanticRightRootResult5602.summary] using leftClaim.claim)
    (by simpa [right, rightRaw, SemanticRightRootResult5599.summary] using rightClaim.claim)
    (resultSound witness.env) (by decide)

theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply claimOfHistory selector witness (by rfl) (by rfl) (by rfl) (by rfl)
  · exact SemanticRightRootResult5602.claimSound
      selector selectorLower selectorUpper witness
  · exact SemanticRightRootResult5599.claimSound
      selector selectorLower selectorUpper witness
  · rfl

end SemanticRightRootResult5607

namespace SemanticRightRootResult5619

def owner : Owner := ⟨.program ⟨214⟩, ⟨6670⟩⟩
def rawTerms : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6669⟩⟩] } }]
def summary : Bound := .exactZero
def producerEvent : Nat := 5618
def resultEvent : Nat := 5619
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual resultEvent
theorem terminalAt (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, rightRootOwnerAtSelector214 selector selectorLower selectorUpper _, ?_⟩
  refine ⟨.scale (.predecessor 0 5616 .coefficient) (.value (.predecessor 1 5617 .coefficient)), 0, .scale (.predecessor 0 5616 .coefficient) (.value (.predecessor 1 5617 .coefficient)), ?_, ?_⟩
  · rfl
  · rfl
theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness
    (terminalAt selector selectorLower selectorUpper)

end SemanticRightRootResult5619

namespace SemanticRightRootResult5622

def owner : Owner := ⟨.program ⟨214⟩, ⟨6732⟩⟩
def rawTerms : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6732⟩⟩] } }]
def summary : Bound := .exactZero
def producerEvent : Nat := 5621
def resultEvent : Nat := 5622
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

end SemanticRightRootResult5622

namespace SemanticRightRootResult5627

def leftRaw : List Term := SemanticRightRootResult5622.rawTerms
def rightRaw : List Term := SemanticRightRootResult5619.rawTerms
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6732⟩⟩, ⟨.program ⟨214⟩, ⟨6669⟩⟩] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def owner : Owner := ⟨.program ⟨214⟩, ⟨7643⟩⟩
def rawTerms : List Term := outputRaw
def summary : Bound := .exactZero
def resultEvent : Nat := 5627
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
  SemanticRightRootResult5622.actual selector witness *
    SemanticRightRootResult5619.actual selector witness
theorem claimOfHistory (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841)
    (expressionAt : document.expressions.lookup 7643 =
      some Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Expression029.ExpressionRow7643)
    (leftPredecessorAt : (history.lookup 5623).map AnnotatedEvent.event =
      some (.predecessor owner 0 ⟨6732⟩ 5622))
    (rightPredecessorAt : (history.lookup 5624).map AnnotatedEvent.event =
      some (.predecessor owner 1 ⟨6670⟩ 5619))
    (ruleAt : (history.lookup 5625).map AnnotatedEvent.event =
      some (.boundTransfer owner (.product (.predecessor 0 5623 .coefficient) (.predecessor 1 5624 .coefficient) ⟨false, false, none, none, none⟩)))
    (leftClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult5622.resultEvent
      SemanticRightRootResult5622.owner
      (SemanticRightRootResult5622.actual selector witness)
      SemanticRightRootResult5622.rawTerms
      SemanticRightRootResult5622.summary)
    (rightClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult5619.resultEvent
      SemanticRightRootResult5619.owner
      (SemanticRightRootResult5619.actual selector witness)
      SemanticRightRootResult5619.rawTerms
      SemanticRightRootResult5619.summary)
    (outputAt : (history.lookup resultEvent).map AnnotatedEvent.event =
      some (.resultExact owner rawTerms summary)) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  refine ⟨outputAt, ?_⟩
  exact exactValueClaim_product_of_mod_zero 100418593683253592432016548326729029359133068138294319235841 witness.env
    (SemanticRightRootResult5622.actual selector witness)
    (SemanticRightRootResult5619.actual selector witness) left right output
    (by simpa [left, leftRaw, SemanticRightRootResult5622.summary] using leftClaim.claim)
    (by simpa [right, rightRaw, SemanticRightRootResult5619.summary] using rightClaim.claim)
    (resultSound witness.env) (by decide)

theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply claimOfHistory selector witness (by rfl) (by rfl) (by rfl) (by rfl)
  · exact SemanticRightRootResult5622.claimSound
      selector selectorLower selectorUpper witness
  · exact SemanticRightRootResult5619.claimSound
      selector selectorLower selectorUpper witness
  · rfl

end SemanticRightRootResult5627

namespace SemanticRightRootResult5639

def owner : Owner := ⟨.program ⟨214⟩, ⟨6674⟩⟩
def rawTerms : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6673⟩⟩] } }]
def summary : Bound := .exactZero
def producerEvent : Nat := 5638
def resultEvent : Nat := 5639
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual resultEvent
theorem terminalAt (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, rightRootOwnerAtSelector214 selector selectorLower selectorUpper _, ?_⟩
  refine ⟨.scale (.predecessor 0 5636 .coefficient) (.value (.predecessor 1 5637 .coefficient)), 0, .scale (.predecessor 0 5636 .coefficient) (.value (.predecessor 1 5637 .coefficient)), ?_, ?_⟩
  · rfl
  · rfl
theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness
    (terminalAt selector selectorLower selectorUpper)

end SemanticRightRootResult5639

namespace SemanticRightRootResult5642

def owner : Owner := ⟨.program ⟨214⟩, ⟨6730⟩⟩
def rawTerms : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6730⟩⟩] } }]
def summary : Bound := .exactZero
def producerEvent : Nat := 5641
def resultEvent : Nat := 5642
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

end SemanticRightRootResult5642

namespace SemanticRightRootResult5647

def leftRaw : List Term := SemanticRightRootResult5642.rawTerms
def rightRaw : List Term := SemanticRightRootResult5639.rawTerms
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6730⟩⟩, ⟨.program ⟨214⟩, ⟨6673⟩⟩] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def owner : Owner := ⟨.program ⟨214⟩, ⟨7642⟩⟩
def rawTerms : List Term := outputRaw
def summary : Bound := .exactZero
def resultEvent : Nat := 5647
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
  SemanticRightRootResult5642.actual selector witness *
    SemanticRightRootResult5639.actual selector witness
theorem claimOfHistory (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841)
    (expressionAt : document.expressions.lookup 7642 =
      some Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Expression029.ExpressionRow7642)
    (leftPredecessorAt : (history.lookup 5643).map AnnotatedEvent.event =
      some (.predecessor owner 0 ⟨6730⟩ 5642))
    (rightPredecessorAt : (history.lookup 5644).map AnnotatedEvent.event =
      some (.predecessor owner 1 ⟨6674⟩ 5639))
    (ruleAt : (history.lookup 5645).map AnnotatedEvent.event =
      some (.boundTransfer owner (.product (.predecessor 0 5643 .coefficient) (.predecessor 1 5644 .coefficient) ⟨false, false, none, none, none⟩)))
    (leftClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult5642.resultEvent
      SemanticRightRootResult5642.owner
      (SemanticRightRootResult5642.actual selector witness)
      SemanticRightRootResult5642.rawTerms
      SemanticRightRootResult5642.summary)
    (rightClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult5639.resultEvent
      SemanticRightRootResult5639.owner
      (SemanticRightRootResult5639.actual selector witness)
      SemanticRightRootResult5639.rawTerms
      SemanticRightRootResult5639.summary)
    (outputAt : (history.lookup resultEvent).map AnnotatedEvent.event =
      some (.resultExact owner rawTerms summary)) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  refine ⟨outputAt, ?_⟩
  exact exactValueClaim_product_of_mod_zero 100418593683253592432016548326729029359133068138294319235841 witness.env
    (SemanticRightRootResult5642.actual selector witness)
    (SemanticRightRootResult5639.actual selector witness) left right output
    (by simpa [left, leftRaw, SemanticRightRootResult5642.summary] using leftClaim.claim)
    (by simpa [right, rightRaw, SemanticRightRootResult5639.summary] using rightClaim.claim)
    (resultSound witness.env) (by decide)

theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply claimOfHistory selector witness (by rfl) (by rfl) (by rfl) (by rfl)
  · exact SemanticRightRootResult5642.claimSound
      selector selectorLower selectorUpper witness
  · exact SemanticRightRootResult5639.claimSound
      selector selectorLower selectorUpper witness
  · rfl

end SemanticRightRootResult5647

namespace SemanticRightRootResult5659

def owner : Owner := ⟨.program ⟨214⟩, ⟨6678⟩⟩
def rawTerms : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6677⟩⟩] } }]
def summary : Bound := .exactZero
def producerEvent : Nat := 5658
def resultEvent : Nat := 5659
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual resultEvent
theorem terminalAt (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, rightRootOwnerAtSelector214 selector selectorLower selectorUpper _, ?_⟩
  refine ⟨.scale (.predecessor 0 5656 .coefficient) (.value (.predecessor 1 5657 .coefficient)), 0, .scale (.predecessor 0 5656 .coefficient) (.value (.predecessor 1 5657 .coefficient)), ?_, ?_⟩
  · rfl
  · rfl
theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness
    (terminalAt selector selectorLower selectorUpper)

end SemanticRightRootResult5659

namespace SemanticRightRootResult5662

def owner : Owner := ⟨.program ⟨214⟩, ⟨6728⟩⟩
def rawTerms : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6728⟩⟩] } }]
def summary : Bound := .exactZero
def producerEvent : Nat := 5661
def resultEvent : Nat := 5662
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

end SemanticRightRootResult5662

namespace SemanticRightRootResult5667

def leftRaw : List Term := SemanticRightRootResult5662.rawTerms
def rightRaw : List Term := SemanticRightRootResult5659.rawTerms
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6728⟩⟩, ⟨.program ⟨214⟩, ⟨6677⟩⟩] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def owner : Owner := ⟨.program ⟨214⟩, ⟨7641⟩⟩
def rawTerms : List Term := outputRaw
def summary : Bound := .exactZero
def resultEvent : Nat := 5667
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
  SemanticRightRootResult5662.actual selector witness *
    SemanticRightRootResult5659.actual selector witness
theorem claimOfHistory (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841)
    (expressionAt : document.expressions.lookup 7641 =
      some Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Expression029.ExpressionRow7641)
    (leftPredecessorAt : (history.lookup 5663).map AnnotatedEvent.event =
      some (.predecessor owner 0 ⟨6728⟩ 5662))
    (rightPredecessorAt : (history.lookup 5664).map AnnotatedEvent.event =
      some (.predecessor owner 1 ⟨6678⟩ 5659))
    (ruleAt : (history.lookup 5665).map AnnotatedEvent.event =
      some (.boundTransfer owner (.product (.predecessor 0 5663 .coefficient) (.predecessor 1 5664 .coefficient) ⟨false, false, none, none, none⟩)))
    (leftClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult5662.resultEvent
      SemanticRightRootResult5662.owner
      (SemanticRightRootResult5662.actual selector witness)
      SemanticRightRootResult5662.rawTerms
      SemanticRightRootResult5662.summary)
    (rightClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult5659.resultEvent
      SemanticRightRootResult5659.owner
      (SemanticRightRootResult5659.actual selector witness)
      SemanticRightRootResult5659.rawTerms
      SemanticRightRootResult5659.summary)
    (outputAt : (history.lookup resultEvent).map AnnotatedEvent.event =
      some (.resultExact owner rawTerms summary)) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  refine ⟨outputAt, ?_⟩
  exact exactValueClaim_product_of_mod_zero 100418593683253592432016548326729029359133068138294319235841 witness.env
    (SemanticRightRootResult5662.actual selector witness)
    (SemanticRightRootResult5659.actual selector witness) left right output
    (by simpa [left, leftRaw, SemanticRightRootResult5662.summary] using leftClaim.claim)
    (by simpa [right, rightRaw, SemanticRightRootResult5659.summary] using rightClaim.claim)
    (resultSound witness.env) (by decide)

theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply claimOfHistory selector witness (by rfl) (by rfl) (by rfl) (by rfl)
  · exact SemanticRightRootResult5662.claimSound
      selector selectorLower selectorUpper witness
  · exact SemanticRightRootResult5659.claimSound
      selector selectorLower selectorUpper witness
  · rfl

end SemanticRightRootResult5667

namespace SemanticRightRootResult5679

def owner : Owner := ⟨.program ⟨214⟩, ⟨6682⟩⟩
def rawTerms : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6681⟩⟩] } }]
def summary : Bound := .exactZero
def producerEvent : Nat := 5678
def resultEvent : Nat := 5679
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual resultEvent
theorem terminalAt (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, rightRootOwnerAtSelector214 selector selectorLower selectorUpper _, ?_⟩
  refine ⟨.scale (.predecessor 0 5676 .coefficient) (.value (.predecessor 1 5677 .coefficient)), 0, .scale (.predecessor 0 5676 .coefficient) (.value (.predecessor 1 5677 .coefficient)), ?_, ?_⟩
  · rfl
  · rfl
theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness
    (terminalAt selector selectorLower selectorUpper)

end SemanticRightRootResult5679

namespace SemanticRightRootResult5682

def owner : Owner := ⟨.program ⟨214⟩, ⟨6726⟩⟩
def rawTerms : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6726⟩⟩] } }]
def summary : Bound := .exactZero
def producerEvent : Nat := 5681
def resultEvent : Nat := 5682
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

end SemanticRightRootResult5682

namespace SemanticRightRootResult5687

def leftRaw : List Term := SemanticRightRootResult5682.rawTerms
def rightRaw : List Term := SemanticRightRootResult5679.rawTerms
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6726⟩⟩, ⟨.program ⟨214⟩, ⟨6681⟩⟩] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def owner : Owner := ⟨.program ⟨214⟩, ⟨7640⟩⟩
def rawTerms : List Term := outputRaw
def summary : Bound := .exactZero
def resultEvent : Nat := 5687
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
  SemanticRightRootResult5682.actual selector witness *
    SemanticRightRootResult5679.actual selector witness
theorem claimOfHistory (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841)
    (expressionAt : document.expressions.lookup 7640 =
      some Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Expression029.ExpressionRow7640)
    (leftPredecessorAt : (history.lookup 5683).map AnnotatedEvent.event =
      some (.predecessor owner 0 ⟨6726⟩ 5682))
    (rightPredecessorAt : (history.lookup 5684).map AnnotatedEvent.event =
      some (.predecessor owner 1 ⟨6682⟩ 5679))
    (ruleAt : (history.lookup 5685).map AnnotatedEvent.event =
      some (.boundTransfer owner (.product (.predecessor 0 5683 .coefficient) (.predecessor 1 5684 .coefficient) ⟨false, false, none, none, none⟩)))
    (leftClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult5682.resultEvent
      SemanticRightRootResult5682.owner
      (SemanticRightRootResult5682.actual selector witness)
      SemanticRightRootResult5682.rawTerms
      SemanticRightRootResult5682.summary)
    (rightClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult5679.resultEvent
      SemanticRightRootResult5679.owner
      (SemanticRightRootResult5679.actual selector witness)
      SemanticRightRootResult5679.rawTerms
      SemanticRightRootResult5679.summary)
    (outputAt : (history.lookup resultEvent).map AnnotatedEvent.event =
      some (.resultExact owner rawTerms summary)) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  refine ⟨outputAt, ?_⟩
  exact exactValueClaim_product_of_mod_zero 100418593683253592432016548326729029359133068138294319235841 witness.env
    (SemanticRightRootResult5682.actual selector witness)
    (SemanticRightRootResult5679.actual selector witness) left right output
    (by simpa [left, leftRaw, SemanticRightRootResult5682.summary] using leftClaim.claim)
    (by simpa [right, rightRaw, SemanticRightRootResult5679.summary] using rightClaim.claim)
    (resultSound witness.env) (by decide)

theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply claimOfHistory selector witness (by rfl) (by rfl) (by rfl) (by rfl)
  · exact SemanticRightRootResult5682.claimSound
      selector selectorLower selectorUpper witness
  · exact SemanticRightRootResult5679.claimSound
      selector selectorLower selectorUpper witness
  · rfl

end SemanticRightRootResult5687

namespace SemanticRightRootResult5699

def owner : Owner := ⟨.program ⟨214⟩, ⟨6638⟩⟩
def rawTerms : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6637⟩⟩] } }]
def summary : Bound := .exactZero
def producerEvent : Nat := 5698
def resultEvent : Nat := 5699
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual resultEvent
theorem terminalAt (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, rightRootOwnerAtSelector214 selector selectorLower selectorUpper _, ?_⟩
  refine ⟨.scale (.predecessor 0 5696 .coefficient) (.value (.predecessor 1 5697 .coefficient)), 0, .scale (.predecessor 0 5696 .coefficient) (.value (.predecessor 1 5697 .coefficient)), ?_, ?_⟩
  · rfl
  · rfl
theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness
    (terminalAt selector selectorLower selectorUpper)

end SemanticRightRootResult5699

namespace SemanticRightRootResult5702

def owner : Owner := ⟨.program ⟨214⟩, ⟨6724⟩⟩
def rawTerms : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6724⟩⟩] } }]
def summary : Bound := .exactZero
def producerEvent : Nat := 5701
def resultEvent : Nat := 5702
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

end SemanticRightRootResult5702

namespace SemanticRightRootResult5707

def leftRaw : List Term := SemanticRightRootResult5702.rawTerms
def rightRaw : List Term := SemanticRightRootResult5699.rawTerms
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6724⟩⟩, ⟨.program ⟨214⟩, ⟨6637⟩⟩] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def owner : Owner := ⟨.program ⟨214⟩, ⟨7639⟩⟩
def rawTerms : List Term := outputRaw
def summary : Bound := .exactZero
def resultEvent : Nat := 5707
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
  SemanticRightRootResult5702.actual selector witness *
    SemanticRightRootResult5699.actual selector witness
theorem claimOfHistory (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841)
    (expressionAt : document.expressions.lookup 7639 =
      some Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Expression029.ExpressionRow7639)
    (leftPredecessorAt : (history.lookup 5703).map AnnotatedEvent.event =
      some (.predecessor owner 0 ⟨6724⟩ 5702))
    (rightPredecessorAt : (history.lookup 5704).map AnnotatedEvent.event =
      some (.predecessor owner 1 ⟨6638⟩ 5699))
    (ruleAt : (history.lookup 5705).map AnnotatedEvent.event =
      some (.boundTransfer owner (.product (.predecessor 0 5703 .coefficient) (.predecessor 1 5704 .coefficient) ⟨false, false, none, none, none⟩)))
    (leftClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult5702.resultEvent
      SemanticRightRootResult5702.owner
      (SemanticRightRootResult5702.actual selector witness)
      SemanticRightRootResult5702.rawTerms
      SemanticRightRootResult5702.summary)
    (rightClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult5699.resultEvent
      SemanticRightRootResult5699.owner
      (SemanticRightRootResult5699.actual selector witness)
      SemanticRightRootResult5699.rawTerms
      SemanticRightRootResult5699.summary)
    (outputAt : (history.lookup resultEvent).map AnnotatedEvent.event =
      some (.resultExact owner rawTerms summary)) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  refine ⟨outputAt, ?_⟩
  exact exactValueClaim_product_of_mod_zero 100418593683253592432016548326729029359133068138294319235841 witness.env
    (SemanticRightRootResult5702.actual selector witness)
    (SemanticRightRootResult5699.actual selector witness) left right output
    (by simpa [left, leftRaw, SemanticRightRootResult5702.summary] using leftClaim.claim)
    (by simpa [right, rightRaw, SemanticRightRootResult5699.summary] using rightClaim.claim)
    (resultSound witness.env) (by decide)

theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply claimOfHistory selector witness (by rfl) (by rfl) (by rfl) (by rfl)
  · exact SemanticRightRootResult5702.claimSound
      selector selectorLower selectorUpper witness
  · exact SemanticRightRootResult5699.claimSound
      selector selectorLower selectorUpper witness
  · rfl

end SemanticRightRootResult5707

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
