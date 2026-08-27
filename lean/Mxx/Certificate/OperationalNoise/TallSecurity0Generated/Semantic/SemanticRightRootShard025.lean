import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticRightRootShard024

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open TallSecurity0ABI
open TallSemantics

namespace SemanticRightRootResult5136

def leftRaw : List Term := SemanticRightRootResult5131.rawTerms
def rightRaw : List Term := SemanticRightRootResult633.rawTerms
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6383⟩⟩, ⟨.program ⟨214⟩, ⟨18016⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def owner : Owner := ⟨.program ⟨214⟩, ⟨18017⟩⟩
def rawTerms : List Term := outputRaw
def summary : Bound := .exactZero
def resultEvent : Nat := 5136
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
  SemanticRightRootResult5131.actual selector witness *
    SemanticRightRootResult633.actual selector witness
theorem claimOfHistory (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841)
    (expressionAt : document.expressions.lookup 18017 =
      some Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Expression070.ExpressionRow18017)
    (leftPredecessorAt : (history.lookup 5132).map AnnotatedEvent.event =
      some (.predecessor owner 0 ⟨18016⟩ 5131))
    (rightPredecessorAt : (history.lookup 5133).map AnnotatedEvent.event =
      some (.predecessor owner 1 ⟨6383⟩ 633))
    (ruleAt : (history.lookup 5134).map AnnotatedEvent.event =
      some (.boundTransfer owner (.product (.predecessor 0 5132 .coefficient) (.predecessor 1 5133 .coefficient) ⟨true, true, none, some 1, some 1⟩)))
    (leftClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult5131.resultEvent
      SemanticRightRootResult5131.owner
      (SemanticRightRootResult5131.actual selector witness)
      SemanticRightRootResult5131.rawTerms
      SemanticRightRootResult5131.summary)
    (rightClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult633.resultEvent
      SemanticRightRootResult633.owner
      (SemanticRightRootResult633.actual selector witness)
      SemanticRightRootResult633.rawTerms
      SemanticRightRootResult633.summary)
    (outputAt : (history.lookup resultEvent).map AnnotatedEvent.event =
      some (.resultExact owner rawTerms summary)) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  refine ⟨outputAt, ?_⟩
  exact exactValueClaim_product_of_mod_zero 100418593683253592432016548326729029359133068138294319235841 witness.env
    (SemanticRightRootResult5131.actual selector witness)
    (SemanticRightRootResult633.actual selector witness) left right output
    (by simpa [left, leftRaw, SemanticRightRootResult5131.summary] using leftClaim.claim)
    (by simpa [right, rightRaw, SemanticRightRootResult633.summary] using rightClaim.claim)
    (resultSound witness.env) (by decide)

theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply claimOfHistory selector witness (by rfl) (by rfl) (by rfl) (by rfl)
  · exact SemanticRightRootResult5131.claimSound
      selector selectorLower selectorUpper witness
  · exact SemanticRightRootResult633.claimSound
      selector selectorLower selectorUpper witness
  · rfl

end SemanticRightRootResult5136

namespace SemanticRightRootResult5139

def owner : Owner := ⟨.program ⟨214⟩, ⟨17155⟩⟩
def rawTerms : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17155⟩⟩], orderedFactors := [] } }]
def summary : Bound := .exactZero
def producerEvent : Nat := 5138
def resultEvent : Nat := 5139
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

end SemanticRightRootResult5139

namespace SemanticRightRootResult5144

def leftRaw : List Term := SemanticRightRootResult5139.rawTerms
def rightRaw : List Term := SemanticRightRootResult643.rawTerms
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6387⟩⟩, ⟨.program ⟨214⟩, ⟨17155⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def owner : Owner := ⟨.program ⟨214⟩, ⟨17156⟩⟩
def rawTerms : List Term := outputRaw
def summary : Bound := .exactZero
def resultEvent : Nat := 5144
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
  SemanticRightRootResult5139.actual selector witness *
    SemanticRightRootResult643.actual selector witness
theorem claimOfHistory (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841)
    (expressionAt : document.expressions.lookup 17156 =
      some Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Expression067.ExpressionRow17156)
    (leftPredecessorAt : (history.lookup 5140).map AnnotatedEvent.event =
      some (.predecessor owner 0 ⟨17155⟩ 5139))
    (rightPredecessorAt : (history.lookup 5141).map AnnotatedEvent.event =
      some (.predecessor owner 1 ⟨6387⟩ 643))
    (ruleAt : (history.lookup 5142).map AnnotatedEvent.event =
      some (.boundTransfer owner (.product (.predecessor 0 5140 .coefficient) (.predecessor 1 5141 .coefficient) ⟨true, true, none, some 1, some 1⟩)))
    (leftClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult5139.resultEvent
      SemanticRightRootResult5139.owner
      (SemanticRightRootResult5139.actual selector witness)
      SemanticRightRootResult5139.rawTerms
      SemanticRightRootResult5139.summary)
    (rightClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult643.resultEvent
      SemanticRightRootResult643.owner
      (SemanticRightRootResult643.actual selector witness)
      SemanticRightRootResult643.rawTerms
      SemanticRightRootResult643.summary)
    (outputAt : (history.lookup resultEvent).map AnnotatedEvent.event =
      some (.resultExact owner rawTerms summary)) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  refine ⟨outputAt, ?_⟩
  exact exactValueClaim_product_of_mod_zero 100418593683253592432016548326729029359133068138294319235841 witness.env
    (SemanticRightRootResult5139.actual selector witness)
    (SemanticRightRootResult643.actual selector witness) left right output
    (by simpa [left, leftRaw, SemanticRightRootResult5139.summary] using leftClaim.claim)
    (by simpa [right, rightRaw, SemanticRightRootResult643.summary] using rightClaim.claim)
    (resultSound witness.env) (by decide)

theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply claimOfHistory selector witness (by rfl) (by rfl) (by rfl) (by rfl)
  · exact SemanticRightRootResult5139.claimSound
      selector selectorLower selectorUpper witness
  · exact SemanticRightRootResult643.claimSound
      selector selectorLower selectorUpper witness
  · rfl

end SemanticRightRootResult5144

namespace SemanticRightRootResult5147

def owner : Owner := ⟨.program ⟨214⟩, ⟨17211⟩⟩
def rawTerms : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17211⟩⟩], orderedFactors := [] } }]
def summary : Bound := .exactZero
def producerEvent : Nat := 5146
def resultEvent : Nat := 5147
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

end SemanticRightRootResult5147

namespace SemanticRightRootResult5152

def leftRaw : List Term := SemanticRightRootResult5147.rawTerms
def rightRaw : List Term := SemanticRightRootResult653.rawTerms
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6391⟩⟩, ⟨.program ⟨214⟩, ⟨17211⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def owner : Owner := ⟨.program ⟨214⟩, ⟨17212⟩⟩
def rawTerms : List Term := outputRaw
def summary : Bound := .exactZero
def resultEvent : Nat := 5152
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
  SemanticRightRootResult5147.actual selector witness *
    SemanticRightRootResult653.actual selector witness
theorem claimOfHistory (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841)
    (expressionAt : document.expressions.lookup 17212 =
      some Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Expression067.ExpressionRow17212)
    (leftPredecessorAt : (history.lookup 5148).map AnnotatedEvent.event =
      some (.predecessor owner 0 ⟨17211⟩ 5147))
    (rightPredecessorAt : (history.lookup 5149).map AnnotatedEvent.event =
      some (.predecessor owner 1 ⟨6391⟩ 653))
    (ruleAt : (history.lookup 5150).map AnnotatedEvent.event =
      some (.boundTransfer owner (.product (.predecessor 0 5148 .coefficient) (.predecessor 1 5149 .coefficient) ⟨true, true, none, some 1, some 1⟩)))
    (leftClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult5147.resultEvent
      SemanticRightRootResult5147.owner
      (SemanticRightRootResult5147.actual selector witness)
      SemanticRightRootResult5147.rawTerms
      SemanticRightRootResult5147.summary)
    (rightClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult653.resultEvent
      SemanticRightRootResult653.owner
      (SemanticRightRootResult653.actual selector witness)
      SemanticRightRootResult653.rawTerms
      SemanticRightRootResult653.summary)
    (outputAt : (history.lookup resultEvent).map AnnotatedEvent.event =
      some (.resultExact owner rawTerms summary)) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  refine ⟨outputAt, ?_⟩
  exact exactValueClaim_product_of_mod_zero 100418593683253592432016548326729029359133068138294319235841 witness.env
    (SemanticRightRootResult5147.actual selector witness)
    (SemanticRightRootResult653.actual selector witness) left right output
    (by simpa [left, leftRaw, SemanticRightRootResult5147.summary] using leftClaim.claim)
    (by simpa [right, rightRaw, SemanticRightRootResult653.summary] using rightClaim.claim)
    (resultSound witness.env) (by decide)

theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply claimOfHistory selector witness (by rfl) (by rfl) (by rfl) (by rfl)
  · exact SemanticRightRootResult5147.claimSound
      selector selectorLower selectorUpper witness
  · exact SemanticRightRootResult653.claimSound
      selector selectorLower selectorUpper witness
  · rfl

end SemanticRightRootResult5152

namespace SemanticRightRootResult5155

def owner : Owner := ⟨.program ⟨214⟩, ⟨17428⟩⟩
def rawTerms : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17428⟩⟩], orderedFactors := [] } }]
def summary : Bound := .exactZero
def producerEvent : Nat := 5154
def resultEvent : Nat := 5155
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

end SemanticRightRootResult5155

namespace SemanticRightRootResult5160

def leftRaw : List Term := SemanticRightRootResult5155.rawTerms
def rightRaw : List Term := SemanticRightRootResult663.rawTerms
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6398⟩⟩, ⟨.program ⟨214⟩, ⟨17428⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def owner : Owner := ⟨.program ⟨214⟩, ⟨17429⟩⟩
def rawTerms : List Term := outputRaw
def summary : Bound := .exactZero
def resultEvent : Nat := 5160
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
  SemanticRightRootResult5155.actual selector witness *
    SemanticRightRootResult663.actual selector witness
theorem claimOfHistory (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841)
    (expressionAt : document.expressions.lookup 17429 =
      some Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Expression068.ExpressionRow17429)
    (leftPredecessorAt : (history.lookup 5156).map AnnotatedEvent.event =
      some (.predecessor owner 0 ⟨17428⟩ 5155))
    (rightPredecessorAt : (history.lookup 5157).map AnnotatedEvent.event =
      some (.predecessor owner 1 ⟨6398⟩ 663))
    (ruleAt : (history.lookup 5158).map AnnotatedEvent.event =
      some (.boundTransfer owner (.product (.predecessor 0 5156 .coefficient) (.predecessor 1 5157 .coefficient) ⟨true, true, none, some 1, some 1⟩)))
    (leftClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult5155.resultEvent
      SemanticRightRootResult5155.owner
      (SemanticRightRootResult5155.actual selector witness)
      SemanticRightRootResult5155.rawTerms
      SemanticRightRootResult5155.summary)
    (rightClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult663.resultEvent
      SemanticRightRootResult663.owner
      (SemanticRightRootResult663.actual selector witness)
      SemanticRightRootResult663.rawTerms
      SemanticRightRootResult663.summary)
    (outputAt : (history.lookup resultEvent).map AnnotatedEvent.event =
      some (.resultExact owner rawTerms summary)) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  refine ⟨outputAt, ?_⟩
  exact exactValueClaim_product_of_mod_zero 100418593683253592432016548326729029359133068138294319235841 witness.env
    (SemanticRightRootResult5155.actual selector witness)
    (SemanticRightRootResult663.actual selector witness) left right output
    (by simpa [left, leftRaw, SemanticRightRootResult5155.summary] using leftClaim.claim)
    (by simpa [right, rightRaw, SemanticRightRootResult663.summary] using rightClaim.claim)
    (resultSound witness.env) (by decide)

theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply claimOfHistory selector witness (by rfl) (by rfl) (by rfl) (by rfl)
  · exact SemanticRightRootResult5155.claimSound
      selector selectorLower selectorUpper witness
  · exact SemanticRightRootResult663.claimSound
      selector selectorLower selectorUpper witness
  · rfl

end SemanticRightRootResult5160

namespace SemanticRightRootResult5163

def owner : Owner := ⟨.program ⟨214⟩, ⟨17792⟩⟩
def rawTerms : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17792⟩⟩], orderedFactors := [] } }]
def summary : Bound := .exactZero
def producerEvent : Nat := 5162
def resultEvent : Nat := 5163
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

end SemanticRightRootResult5163

namespace SemanticRightRootResult5168

def leftRaw : List Term := SemanticRightRootResult5163.rawTerms
def rightRaw : List Term := SemanticRightRootResult673.rawTerms
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6407⟩⟩, ⟨.program ⟨214⟩, ⟨17792⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def owner : Owner := ⟨.program ⟨214⟩, ⟨17793⟩⟩
def rawTerms : List Term := outputRaw
def summary : Bound := .exactZero
def resultEvent : Nat := 5168
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
  SemanticRightRootResult5163.actual selector witness *
    SemanticRightRootResult673.actual selector witness
theorem claimOfHistory (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841)
    (expressionAt : document.expressions.lookup 17793 =
      some Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Expression069.ExpressionRow17793)
    (leftPredecessorAt : (history.lookup 5164).map AnnotatedEvent.event =
      some (.predecessor owner 0 ⟨17792⟩ 5163))
    (rightPredecessorAt : (history.lookup 5165).map AnnotatedEvent.event =
      some (.predecessor owner 1 ⟨6407⟩ 673))
    (ruleAt : (history.lookup 5166).map AnnotatedEvent.event =
      some (.boundTransfer owner (.product (.predecessor 0 5164 .coefficient) (.predecessor 1 5165 .coefficient) ⟨true, true, none, some 1, some 1⟩)))
    (leftClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult5163.resultEvent
      SemanticRightRootResult5163.owner
      (SemanticRightRootResult5163.actual selector witness)
      SemanticRightRootResult5163.rawTerms
      SemanticRightRootResult5163.summary)
    (rightClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult673.resultEvent
      SemanticRightRootResult673.owner
      (SemanticRightRootResult673.actual selector witness)
      SemanticRightRootResult673.rawTerms
      SemanticRightRootResult673.summary)
    (outputAt : (history.lookup resultEvent).map AnnotatedEvent.event =
      some (.resultExact owner rawTerms summary)) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  refine ⟨outputAt, ?_⟩
  exact exactValueClaim_product_of_mod_zero 100418593683253592432016548326729029359133068138294319235841 witness.env
    (SemanticRightRootResult5163.actual selector witness)
    (SemanticRightRootResult673.actual selector witness) left right output
    (by simpa [left, leftRaw, SemanticRightRootResult5163.summary] using leftClaim.claim)
    (by simpa [right, rightRaw, SemanticRightRootResult673.summary] using rightClaim.claim)
    (resultSound witness.env) (by decide)

theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply claimOfHistory selector witness (by rfl) (by rfl) (by rfl) (by rfl)
  · exact SemanticRightRootResult5163.claimSound
      selector selectorLower selectorUpper witness
  · exact SemanticRightRootResult673.claimSound
      selector selectorLower selectorUpper witness
  · rfl

end SemanticRightRootResult5168

namespace SemanticRightRootResult5171

def owner : Owner := ⟨.program ⟨214⟩, ⟨15503⟩⟩
def rawTerms : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15503⟩⟩], orderedFactors := [] } }]
def summary : Bound := .exactZero
def producerEvent : Nat := 5170
def resultEvent : Nat := 5171
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

end SemanticRightRootResult5171

namespace SemanticRightRootResult5176

def leftRaw : List Term := SemanticRightRootResult5171.rawTerms
def rightRaw : List Term := SemanticRightRootResult683.rawTerms
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6427⟩⟩, ⟨.program ⟨214⟩, ⟨15503⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def owner : Owner := ⟨.program ⟨214⟩, ⟨15504⟩⟩
def rawTerms : List Term := outputRaw
def summary : Bound := .exactZero
def resultEvent : Nat := 5176
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
  SemanticRightRootResult5171.actual selector witness *
    SemanticRightRootResult683.actual selector witness
theorem claimOfHistory (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841)
    (expressionAt : document.expressions.lookup 15504 =
      some Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Expression060.ExpressionRow15504)
    (leftPredecessorAt : (history.lookup 5172).map AnnotatedEvent.event =
      some (.predecessor owner 0 ⟨15503⟩ 5171))
    (rightPredecessorAt : (history.lookup 5173).map AnnotatedEvent.event =
      some (.predecessor owner 1 ⟨6427⟩ 683))
    (ruleAt : (history.lookup 5174).map AnnotatedEvent.event =
      some (.boundTransfer owner (.product (.predecessor 0 5172 .coefficient) (.predecessor 1 5173 .coefficient) ⟨true, true, none, some 1, some 1⟩)))
    (leftClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult5171.resultEvent
      SemanticRightRootResult5171.owner
      (SemanticRightRootResult5171.actual selector witness)
      SemanticRightRootResult5171.rawTerms
      SemanticRightRootResult5171.summary)
    (rightClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult683.resultEvent
      SemanticRightRootResult683.owner
      (SemanticRightRootResult683.actual selector witness)
      SemanticRightRootResult683.rawTerms
      SemanticRightRootResult683.summary)
    (outputAt : (history.lookup resultEvent).map AnnotatedEvent.event =
      some (.resultExact owner rawTerms summary)) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  refine ⟨outputAt, ?_⟩
  exact exactValueClaim_product_of_mod_zero 100418593683253592432016548326729029359133068138294319235841 witness.env
    (SemanticRightRootResult5171.actual selector witness)
    (SemanticRightRootResult683.actual selector witness) left right output
    (by simpa [left, leftRaw, SemanticRightRootResult5171.summary] using leftClaim.claim)
    (by simpa [right, rightRaw, SemanticRightRootResult683.summary] using rightClaim.claim)
    (resultSound witness.env) (by decide)

theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply claimOfHistory selector witness (by rfl) (by rfl) (by rfl) (by rfl)
  · exact SemanticRightRootResult5171.claimSound
      selector selectorLower selectorUpper witness
  · exact SemanticRightRootResult683.claimSound
      selector selectorLower selectorUpper witness
  · rfl

end SemanticRightRootResult5176

namespace SemanticRightRootResult5179

def owner : Owner := ⟨.program ⟨214⟩, ⟨15195⟩⟩
def rawTerms : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15195⟩⟩], orderedFactors := [] } }]
def summary : Bound := .exactZero
def producerEvent : Nat := 5178
def resultEvent : Nat := 5179
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

end SemanticRightRootResult5179

namespace SemanticRightRootResult5184

def leftRaw : List Term := SemanticRightRootResult5179.rawTerms
def rightRaw : List Term := SemanticRightRootResult693.rawTerms
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6452⟩⟩, ⟨.program ⟨214⟩, ⟨15195⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def owner : Owner := ⟨.program ⟨214⟩, ⟨15196⟩⟩
def rawTerms : List Term := outputRaw
def summary : Bound := .exactZero
def resultEvent : Nat := 5184
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
  SemanticRightRootResult5179.actual selector witness *
    SemanticRightRootResult693.actual selector witness
theorem claimOfHistory (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841)
    (expressionAt : document.expressions.lookup 15196 =
      some Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Expression059.ExpressionRow15196)
    (leftPredecessorAt : (history.lookup 5180).map AnnotatedEvent.event =
      some (.predecessor owner 0 ⟨15195⟩ 5179))
    (rightPredecessorAt : (history.lookup 5181).map AnnotatedEvent.event =
      some (.predecessor owner 1 ⟨6452⟩ 693))
    (ruleAt : (history.lookup 5182).map AnnotatedEvent.event =
      some (.boundTransfer owner (.product (.predecessor 0 5180 .coefficient) (.predecessor 1 5181 .coefficient) ⟨true, true, none, some 1, some 1⟩)))
    (leftClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult5179.resultEvent
      SemanticRightRootResult5179.owner
      (SemanticRightRootResult5179.actual selector witness)
      SemanticRightRootResult5179.rawTerms
      SemanticRightRootResult5179.summary)
    (rightClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult693.resultEvent
      SemanticRightRootResult693.owner
      (SemanticRightRootResult693.actual selector witness)
      SemanticRightRootResult693.rawTerms
      SemanticRightRootResult693.summary)
    (outputAt : (history.lookup resultEvent).map AnnotatedEvent.event =
      some (.resultExact owner rawTerms summary)) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  refine ⟨outputAt, ?_⟩
  exact exactValueClaim_product_of_mod_zero 100418593683253592432016548326729029359133068138294319235841 witness.env
    (SemanticRightRootResult5179.actual selector witness)
    (SemanticRightRootResult693.actual selector witness) left right output
    (by simpa [left, leftRaw, SemanticRightRootResult5179.summary] using leftClaim.claim)
    (by simpa [right, rightRaw, SemanticRightRootResult693.summary] using rightClaim.claim)
    (resultSound witness.env) (by decide)

theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply claimOfHistory selector witness (by rfl) (by rfl) (by rfl) (by rfl)
  · exact SemanticRightRootResult5179.claimSound
      selector selectorLower selectorUpper witness
  · exact SemanticRightRootResult693.claimSound
      selector selectorLower selectorUpper witness
  · rfl

end SemanticRightRootResult5184

namespace SemanticRightRootResult5187

def owner : Owner := ⟨.program ⟨214⟩, ⟨15034⟩⟩
def rawTerms : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15034⟩⟩], orderedFactors := [] } }]
def summary : Bound := .exactZero
def producerEvent : Nat := 5186
def resultEvent : Nat := 5187
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

end SemanticRightRootResult5187

namespace SemanticRightRootResult5192

def leftRaw : List Term := SemanticRightRootResult5187.rawTerms
def rightRaw : List Term := SemanticRightRootResult703.rawTerms
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨15034⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def owner : Owner := ⟨.program ⟨214⟩, ⟨15035⟩⟩
def rawTerms : List Term := outputRaw
def summary : Bound := .exactZero
def resultEvent : Nat := 5192
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
  SemanticRightRootResult5187.actual selector witness *
    SemanticRightRootResult703.actual selector witness
theorem claimOfHistory (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841)
    (expressionAt : document.expressions.lookup 15035 =
      some Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Expression058.ExpressionRow15035)
    (leftPredecessorAt : (history.lookup 5188).map AnnotatedEvent.event =
      some (.predecessor owner 0 ⟨15034⟩ 5187))
    (rightPredecessorAt : (history.lookup 5189).map AnnotatedEvent.event =
      some (.predecessor owner 1 ⟨6475⟩ 703))
    (ruleAt : (history.lookup 5190).map AnnotatedEvent.event =
      some (.boundTransfer owner (.product (.predecessor 0 5188 .coefficient) (.predecessor 1 5189 .coefficient) ⟨true, true, none, some 1, some 1⟩)))
    (leftClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult5187.resultEvent
      SemanticRightRootResult5187.owner
      (SemanticRightRootResult5187.actual selector witness)
      SemanticRightRootResult5187.rawTerms
      SemanticRightRootResult5187.summary)
    (rightClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult703.resultEvent
      SemanticRightRootResult703.owner
      (SemanticRightRootResult703.actual selector witness)
      SemanticRightRootResult703.rawTerms
      SemanticRightRootResult703.summary)
    (outputAt : (history.lookup resultEvent).map AnnotatedEvent.event =
      some (.resultExact owner rawTerms summary)) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  refine ⟨outputAt, ?_⟩
  exact exactValueClaim_product_of_mod_zero 100418593683253592432016548326729029359133068138294319235841 witness.env
    (SemanticRightRootResult5187.actual selector witness)
    (SemanticRightRootResult703.actual selector witness) left right output
    (by simpa [left, leftRaw, SemanticRightRootResult5187.summary] using leftClaim.claim)
    (by simpa [right, rightRaw, SemanticRightRootResult703.summary] using rightClaim.claim)
    (resultSound witness.env) (by decide)

theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply claimOfHistory selector witness (by rfl) (by rfl) (by rfl) (by rfl)
  · exact SemanticRightRootResult5187.claimSound
      selector selectorLower selectorUpper witness
  · exact SemanticRightRootResult703.claimSound
      selector selectorLower selectorUpper witness
  · rfl

end SemanticRightRootResult5192

namespace SemanticRightRootResult5195

def owner : Owner := ⟨.program ⟨214⟩, ⟨14873⟩⟩
def rawTerms : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨14873⟩⟩], orderedFactors := [] } }]
def summary : Bound := .exactZero
def producerEvent : Nat := 5194
def resultEvent : Nat := 5195
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

end SemanticRightRootResult5195

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
