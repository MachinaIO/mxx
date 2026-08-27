import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticRightRootShard010

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open TallSecurity0ABI
open TallSemantics

namespace SemanticRightRootResult2208

def leftRaw : List Term := SemanticRightRootResult2203.rawTerms
def rightRaw : List Term := SemanticRightRootResult693.rawTerms
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6452⟩⟩, ⟨.program ⟨214⟩, ⟨15218⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def owner : Owner := ⟨.program ⟨214⟩, ⟨15219⟩⟩
def rawTerms : List Term := outputRaw
def summary : Bound := .exactZero
def resultEvent : Nat := 2208
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
  SemanticRightRootResult2203.actual selector witness *
    SemanticRightRootResult693.actual selector witness
theorem claimOfHistory (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841)
    (expressionAt : document.expressions.lookup 15219 =
      some Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Expression059.ExpressionRow15219)
    (leftPredecessorAt : (history.lookup 2204).map AnnotatedEvent.event =
      some (.predecessor owner 0 ⟨15218⟩ 2203))
    (rightPredecessorAt : (history.lookup 2205).map AnnotatedEvent.event =
      some (.predecessor owner 1 ⟨6452⟩ 693))
    (ruleAt : (history.lookup 2206).map AnnotatedEvent.event =
      some (.boundTransfer owner (.product (.predecessor 0 2204 .coefficient) (.predecessor 1 2205 .coefficient) ⟨true, true, none, some 1, some 1⟩)))
    (leftClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult2203.resultEvent
      SemanticRightRootResult2203.owner
      (SemanticRightRootResult2203.actual selector witness)
      SemanticRightRootResult2203.rawTerms
      SemanticRightRootResult2203.summary)
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
    (SemanticRightRootResult2203.actual selector witness)
    (SemanticRightRootResult693.actual selector witness) left right output
    (by simpa [left, leftRaw, SemanticRightRootResult2203.summary] using leftClaim.claim)
    (by simpa [right, rightRaw, SemanticRightRootResult693.summary] using rightClaim.claim)
    (resultSound witness.env) (by decide)

theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply claimOfHistory selector witness (by rfl) (by rfl) (by rfl) (by rfl)
  · exact SemanticRightRootResult2203.claimSound
      selector selectorLower selectorUpper witness
  · exact SemanticRightRootResult693.claimSound
      selector selectorLower selectorUpper witness
  · rfl

end SemanticRightRootResult2208

namespace SemanticRightRootResult2211

def owner : Owner := ⟨.program ⟨214⟩, ⟨15057⟩⟩
def rawTerms : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15057⟩⟩], orderedFactors := [] } }]
def summary : Bound := .exactZero
def producerEvent : Nat := 2210
def resultEvent : Nat := 2211
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

end SemanticRightRootResult2211

namespace SemanticRightRootResult2216

def leftRaw : List Term := SemanticRightRootResult2211.rawTerms
def rightRaw : List Term := SemanticRightRootResult703.rawTerms
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨15057⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def owner : Owner := ⟨.program ⟨214⟩, ⟨15058⟩⟩
def rawTerms : List Term := outputRaw
def summary : Bound := .exactZero
def resultEvent : Nat := 2216
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
  SemanticRightRootResult2211.actual selector witness *
    SemanticRightRootResult703.actual selector witness
theorem claimOfHistory (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841)
    (expressionAt : document.expressions.lookup 15058 =
      some Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Expression058.ExpressionRow15058)
    (leftPredecessorAt : (history.lookup 2212).map AnnotatedEvent.event =
      some (.predecessor owner 0 ⟨15057⟩ 2211))
    (rightPredecessorAt : (history.lookup 2213).map AnnotatedEvent.event =
      some (.predecessor owner 1 ⟨6475⟩ 703))
    (ruleAt : (history.lookup 2214).map AnnotatedEvent.event =
      some (.boundTransfer owner (.product (.predecessor 0 2212 .coefficient) (.predecessor 1 2213 .coefficient) ⟨true, true, none, some 1, some 1⟩)))
    (leftClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult2211.resultEvent
      SemanticRightRootResult2211.owner
      (SemanticRightRootResult2211.actual selector witness)
      SemanticRightRootResult2211.rawTerms
      SemanticRightRootResult2211.summary)
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
    (SemanticRightRootResult2211.actual selector witness)
    (SemanticRightRootResult703.actual selector witness) left right output
    (by simpa [left, leftRaw, SemanticRightRootResult2211.summary] using leftClaim.claim)
    (by simpa [right, rightRaw, SemanticRightRootResult703.summary] using rightClaim.claim)
    (resultSound witness.env) (by decide)

theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply claimOfHistory selector witness (by rfl) (by rfl) (by rfl) (by rfl)
  · exact SemanticRightRootResult2211.claimSound
      selector selectorLower selectorUpper witness
  · exact SemanticRightRootResult703.claimSound
      selector selectorLower selectorUpper witness
  · rfl

end SemanticRightRootResult2216

namespace SemanticRightRootResult2219

def owner : Owner := ⟨.program ⟨214⟩, ⟨14896⟩⟩
def rawTerms : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨14896⟩⟩], orderedFactors := [] } }]
def summary : Bound := .exactZero
def producerEvent : Nat := 2218
def resultEvent : Nat := 2219
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

end SemanticRightRootResult2219

namespace SemanticRightRootResult2224

def leftRaw : List Term := SemanticRightRootResult2219.rawTerms
def rightRaw : List Term := SemanticRightRootResult713.rawTerms
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨14896⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def owner : Owner := ⟨.program ⟨214⟩, ⟨14897⟩⟩
def rawTerms : List Term := outputRaw
def summary : Bound := .exactZero
def resultEvent : Nat := 2224
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
  SemanticRightRootResult2219.actual selector witness *
    SemanticRightRootResult713.actual selector witness
theorem claimOfHistory (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841)
    (expressionAt : document.expressions.lookup 14897 =
      some Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Expression058.ExpressionRow14897)
    (leftPredecessorAt : (history.lookup 2220).map AnnotatedEvent.event =
      some (.predecessor owner 0 ⟨14896⟩ 2219))
    (rightPredecessorAt : (history.lookup 2221).map AnnotatedEvent.event =
      some (.predecessor owner 1 ⟨6495⟩ 713))
    (ruleAt : (history.lookup 2222).map AnnotatedEvent.event =
      some (.boundTransfer owner (.product (.predecessor 0 2220 .coefficient) (.predecessor 1 2221 .coefficient) ⟨true, true, none, some 1, some 1⟩)))
    (leftClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult2219.resultEvent
      SemanticRightRootResult2219.owner
      (SemanticRightRootResult2219.actual selector witness)
      SemanticRightRootResult2219.rawTerms
      SemanticRightRootResult2219.summary)
    (rightClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult713.resultEvent
      SemanticRightRootResult713.owner
      (SemanticRightRootResult713.actual selector witness)
      SemanticRightRootResult713.rawTerms
      SemanticRightRootResult713.summary)
    (outputAt : (history.lookup resultEvent).map AnnotatedEvent.event =
      some (.resultExact owner rawTerms summary)) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  refine ⟨outputAt, ?_⟩
  exact exactValueClaim_product_of_mod_zero 100418593683253592432016548326729029359133068138294319235841 witness.env
    (SemanticRightRootResult2219.actual selector witness)
    (SemanticRightRootResult713.actual selector witness) left right output
    (by simpa [left, leftRaw, SemanticRightRootResult2219.summary] using leftClaim.claim)
    (by simpa [right, rightRaw, SemanticRightRootResult713.summary] using rightClaim.claim)
    (resultSound witness.env) (by decide)

theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply claimOfHistory selector witness (by rfl) (by rfl) (by rfl) (by rfl)
  · exact SemanticRightRootResult2219.claimSound
      selector selectorLower selectorUpper witness
  · exact SemanticRightRootResult713.claimSound
      selector selectorLower selectorUpper witness
  · rfl

end SemanticRightRootResult2224

namespace SemanticRightRootResult2228

def leftRaw : List Term := SemanticRightRootResult728.rawTerms
def rightRaw : List Term := SemanticRightRootResult2224.rawTerms
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨14896⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def owner : Owner := ⟨.program ⟨214⟩, ⟨14898⟩⟩
def rawTerms : List Term := outputRaw
def summary : Bound := .exactZero
def resultEvent : Nat := 2228
theorem resultAgreement : CanonicalAgreement output (add left right) := by
  decide +kernel

theorem resultSound (env : Env Owner) :
    evalPolynomial env output = evalPolynomial env left + evalPolynomial env right := by
  exact addCanonicalResultSound env left right output resultAgreement

def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  SemanticRightRootResult728.actual selector witness +
    SemanticRightRootResult2224.actual selector witness
theorem claimOfHistory (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841)
    (expressionAt : document.expressions.lookup 14898 =
      some Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Expression058.ExpressionRow14898)
    (leftPredecessorAt : (history.lookup 2225).map AnnotatedEvent.event =
      some (.predecessor owner 0 ⟨6379⟩ 728))
    (rightPredecessorAt : (history.lookup 2226).map AnnotatedEvent.event =
      some (.predecessor owner 1 ⟨14897⟩ 2224))
    (ruleAt : (history.lookup 2227).map AnnotatedEvent.event =
      some (.boundTransfer owner (.sum [.predecessor 0 2225 .coefficient, .predecessor 1 2226 .coefficient])))
    (leftClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult728.resultEvent
      SemanticRightRootResult728.owner
      (SemanticRightRootResult728.actual selector witness)
      SemanticRightRootResult728.rawTerms
      SemanticRightRootResult728.summary)
    (rightClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult2224.resultEvent
      SemanticRightRootResult2224.owner
      (SemanticRightRootResult2224.actual selector witness)
      SemanticRightRootResult2224.rawTerms
      SemanticRightRootResult2224.summary)
    (outputAt : (history.lookup resultEvent).map AnnotatedEvent.event =
      some (.resultExact owner rawTerms summary)) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  refine ⟨outputAt, ?_⟩
  exact exactValueClaim_add_of_mod_zero 100418593683253592432016548326729029359133068138294319235841 witness.env
    (SemanticRightRootResult728.actual selector witness)
    (SemanticRightRootResult2224.actual selector witness) left right output
    (by simpa [left, leftRaw, SemanticRightRootResult728.summary] using leftClaim.claim)
    (by simpa [right, rightRaw, SemanticRightRootResult2224.summary] using rightClaim.claim)
    (resultSound witness.env) (by decide)

theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply claimOfHistory selector witness (by rfl) (by rfl) (by rfl) (by rfl)
  · exact SemanticRightRootResult728.claimSound
      selector selectorLower selectorUpper witness
  · exact SemanticRightRootResult2224.claimSound
      selector selectorLower selectorUpper witness
  · rfl

end SemanticRightRootResult2228

namespace SemanticRightRootResult2232

def leftRaw : List Term := SemanticRightRootResult2228.rawTerms
def rightRaw : List Term := SemanticRightRootResult2216.rawTerms
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨15057⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨14896⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def owner : Owner := ⟨.program ⟨214⟩, ⟨15059⟩⟩
def rawTerms : List Term := outputRaw
def summary : Bound := .exactZero
def resultEvent : Nat := 2232
theorem resultAgreement : CanonicalAgreement output (add left right) := by
  decide +kernel

theorem resultSound (env : Env Owner) :
    evalPolynomial env output = evalPolynomial env left + evalPolynomial env right := by
  exact addCanonicalResultSound env left right output resultAgreement

def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  SemanticRightRootResult2228.actual selector witness +
    SemanticRightRootResult2216.actual selector witness
theorem claimOfHistory (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841)
    (expressionAt : document.expressions.lookup 15059 =
      some Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Expression058.ExpressionRow15059)
    (leftPredecessorAt : (history.lookup 2229).map AnnotatedEvent.event =
      some (.predecessor owner 0 ⟨14898⟩ 2228))
    (rightPredecessorAt : (history.lookup 2230).map AnnotatedEvent.event =
      some (.predecessor owner 1 ⟨15058⟩ 2216))
    (ruleAt : (history.lookup 2231).map AnnotatedEvent.event =
      some (.boundTransfer owner (.sum [.predecessor 0 2229 .coefficient, .predecessor 1 2230 .coefficient])))
    (leftClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult2228.resultEvent
      SemanticRightRootResult2228.owner
      (SemanticRightRootResult2228.actual selector witness)
      SemanticRightRootResult2228.rawTerms
      SemanticRightRootResult2228.summary)
    (rightClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult2216.resultEvent
      SemanticRightRootResult2216.owner
      (SemanticRightRootResult2216.actual selector witness)
      SemanticRightRootResult2216.rawTerms
      SemanticRightRootResult2216.summary)
    (outputAt : (history.lookup resultEvent).map AnnotatedEvent.event =
      some (.resultExact owner rawTerms summary)) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  refine ⟨outputAt, ?_⟩
  exact exactValueClaim_add_of_mod_zero 100418593683253592432016548326729029359133068138294319235841 witness.env
    (SemanticRightRootResult2228.actual selector witness)
    (SemanticRightRootResult2216.actual selector witness) left right output
    (by simpa [left, leftRaw, SemanticRightRootResult2228.summary] using leftClaim.claim)
    (by simpa [right, rightRaw, SemanticRightRootResult2216.summary] using rightClaim.claim)
    (resultSound witness.env) (by decide)

theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply claimOfHistory selector witness (by rfl) (by rfl) (by rfl) (by rfl)
  · exact SemanticRightRootResult2228.claimSound
      selector selectorLower selectorUpper witness
  · exact SemanticRightRootResult2216.claimSound
      selector selectorLower selectorUpper witness
  · rfl

end SemanticRightRootResult2232

namespace SemanticRightRootResult2236

def leftRaw : List Term := SemanticRightRootResult2232.rawTerms
def rightRaw : List Term := SemanticRightRootResult2208.rawTerms
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6452⟩⟩, ⟨.program ⟨214⟩, ⟨15218⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨15057⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨14896⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def owner : Owner := ⟨.program ⟨214⟩, ⟨15220⟩⟩
def rawTerms : List Term := outputRaw
def summary : Bound := .exactZero
def resultEvent : Nat := 2236
theorem resultAgreement : CanonicalAgreement output (add left right) := by
  decide +kernel

theorem resultSound (env : Env Owner) :
    evalPolynomial env output = evalPolynomial env left + evalPolynomial env right := by
  exact addCanonicalResultSound env left right output resultAgreement

def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  SemanticRightRootResult2232.actual selector witness +
    SemanticRightRootResult2208.actual selector witness
theorem claimOfHistory (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841)
    (expressionAt : document.expressions.lookup 15220 =
      some Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Expression059.ExpressionRow15220)
    (leftPredecessorAt : (history.lookup 2233).map AnnotatedEvent.event =
      some (.predecessor owner 0 ⟨15059⟩ 2232))
    (rightPredecessorAt : (history.lookup 2234).map AnnotatedEvent.event =
      some (.predecessor owner 1 ⟨15219⟩ 2208))
    (ruleAt : (history.lookup 2235).map AnnotatedEvent.event =
      some (.boundTransfer owner (.sum [.predecessor 0 2233 .coefficient, .predecessor 1 2234 .coefficient])))
    (leftClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult2232.resultEvent
      SemanticRightRootResult2232.owner
      (SemanticRightRootResult2232.actual selector witness)
      SemanticRightRootResult2232.rawTerms
      SemanticRightRootResult2232.summary)
    (rightClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult2208.resultEvent
      SemanticRightRootResult2208.owner
      (SemanticRightRootResult2208.actual selector witness)
      SemanticRightRootResult2208.rawTerms
      SemanticRightRootResult2208.summary)
    (outputAt : (history.lookup resultEvent).map AnnotatedEvent.event =
      some (.resultExact owner rawTerms summary)) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  refine ⟨outputAt, ?_⟩
  exact exactValueClaim_add_of_mod_zero 100418593683253592432016548326729029359133068138294319235841 witness.env
    (SemanticRightRootResult2232.actual selector witness)
    (SemanticRightRootResult2208.actual selector witness) left right output
    (by simpa [left, leftRaw, SemanticRightRootResult2232.summary] using leftClaim.claim)
    (by simpa [right, rightRaw, SemanticRightRootResult2208.summary] using rightClaim.claim)
    (resultSound witness.env) (by decide)

theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply claimOfHistory selector witness (by rfl) (by rfl) (by rfl) (by rfl)
  · exact SemanticRightRootResult2232.claimSound
      selector selectorLower selectorUpper witness
  · exact SemanticRightRootResult2208.claimSound
      selector selectorLower selectorUpper witness
  · rfl

end SemanticRightRootResult2236

namespace SemanticRightRootResult2240

def leftRaw : List Term := SemanticRightRootResult2236.rawTerms
def rightRaw : List Term := SemanticRightRootResult2200.rawTerms
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6427⟩⟩, ⟨.program ⟨214⟩, ⟨15526⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6452⟩⟩, ⟨.program ⟨214⟩, ⟨15218⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨15057⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨14896⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def owner : Owner := ⟨.program ⟨214⟩, ⟨15528⟩⟩
def rawTerms : List Term := outputRaw
def summary : Bound := .exactZero
def resultEvent : Nat := 2240
theorem resultAgreement : CanonicalAgreement output (add left right) := by
  decide +kernel

theorem resultSound (env : Env Owner) :
    evalPolynomial env output = evalPolynomial env left + evalPolynomial env right := by
  exact addCanonicalResultSound env left right output resultAgreement

def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  SemanticRightRootResult2236.actual selector witness +
    SemanticRightRootResult2200.actual selector witness
theorem claimOfHistory (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841)
    (expressionAt : document.expressions.lookup 15528 =
      some Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Expression060.ExpressionRow15528)
    (leftPredecessorAt : (history.lookup 2237).map AnnotatedEvent.event =
      some (.predecessor owner 0 ⟨15220⟩ 2236))
    (rightPredecessorAt : (history.lookup 2238).map AnnotatedEvent.event =
      some (.predecessor owner 1 ⟨15527⟩ 2200))
    (ruleAt : (history.lookup 2239).map AnnotatedEvent.event =
      some (.boundTransfer owner (.sum [.predecessor 0 2237 .coefficient, .predecessor 1 2238 .coefficient])))
    (leftClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult2236.resultEvent
      SemanticRightRootResult2236.owner
      (SemanticRightRootResult2236.actual selector witness)
      SemanticRightRootResult2236.rawTerms
      SemanticRightRootResult2236.summary)
    (rightClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult2200.resultEvent
      SemanticRightRootResult2200.owner
      (SemanticRightRootResult2200.actual selector witness)
      SemanticRightRootResult2200.rawTerms
      SemanticRightRootResult2200.summary)
    (outputAt : (history.lookup resultEvent).map AnnotatedEvent.event =
      some (.resultExact owner rawTerms summary)) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  refine ⟨outputAt, ?_⟩
  exact exactValueClaim_add_of_mod_zero 100418593683253592432016548326729029359133068138294319235841 witness.env
    (SemanticRightRootResult2236.actual selector witness)
    (SemanticRightRootResult2200.actual selector witness) left right output
    (by simpa [left, leftRaw, SemanticRightRootResult2236.summary] using leftClaim.claim)
    (by simpa [right, rightRaw, SemanticRightRootResult2200.summary] using rightClaim.claim)
    (resultSound witness.env) (by decide)

theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply claimOfHistory selector witness (by rfl) (by rfl) (by rfl) (by rfl)
  · exact SemanticRightRootResult2236.claimSound
      selector selectorLower selectorUpper witness
  · exact SemanticRightRootResult2200.claimSound
      selector selectorLower selectorUpper witness
  · rfl

end SemanticRightRootResult2240

namespace SemanticRightRootResult2244

def leftRaw : List Term := SemanticRightRootResult2240.rawTerms
def rightRaw : List Term := SemanticRightRootResult2192.rawTerms
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6407⟩⟩, ⟨.program ⟨214⟩, ⟨17830⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6427⟩⟩, ⟨.program ⟨214⟩, ⟨15526⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6452⟩⟩, ⟨.program ⟨214⟩, ⟨15218⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨15057⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨14896⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def owner : Owner := ⟨.program ⟨214⟩, ⟨17832⟩⟩
def rawTerms : List Term := outputRaw
def summary : Bound := .exactZero
def resultEvent : Nat := 2244
theorem resultAgreement : CanonicalAgreement output (add left right) := by
  decide +kernel

theorem resultSound (env : Env Owner) :
    evalPolynomial env output = evalPolynomial env left + evalPolynomial env right := by
  exact addCanonicalResultSound env left right output resultAgreement

def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  SemanticRightRootResult2240.actual selector witness +
    SemanticRightRootResult2192.actual selector witness
theorem claimOfHistory (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841)
    (expressionAt : document.expressions.lookup 17832 =
      some Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Expression069.ExpressionRow17832)
    (leftPredecessorAt : (history.lookup 2241).map AnnotatedEvent.event =
      some (.predecessor owner 0 ⟨15528⟩ 2240))
    (rightPredecessorAt : (history.lookup 2242).map AnnotatedEvent.event =
      some (.predecessor owner 1 ⟨17831⟩ 2192))
    (ruleAt : (history.lookup 2243).map AnnotatedEvent.event =
      some (.boundTransfer owner (.sum [.predecessor 0 2241 .coefficient, .predecessor 1 2242 .coefficient])))
    (leftClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult2240.resultEvent
      SemanticRightRootResult2240.owner
      (SemanticRightRootResult2240.actual selector witness)
      SemanticRightRootResult2240.rawTerms
      SemanticRightRootResult2240.summary)
    (rightClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult2192.resultEvent
      SemanticRightRootResult2192.owner
      (SemanticRightRootResult2192.actual selector witness)
      SemanticRightRootResult2192.rawTerms
      SemanticRightRootResult2192.summary)
    (outputAt : (history.lookup resultEvent).map AnnotatedEvent.event =
      some (.resultExact owner rawTerms summary)) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  refine ⟨outputAt, ?_⟩
  exact exactValueClaim_add_of_mod_zero 100418593683253592432016548326729029359133068138294319235841 witness.env
    (SemanticRightRootResult2240.actual selector witness)
    (SemanticRightRootResult2192.actual selector witness) left right output
    (by simpa [left, leftRaw, SemanticRightRootResult2240.summary] using leftClaim.claim)
    (by simpa [right, rightRaw, SemanticRightRootResult2192.summary] using rightClaim.claim)
    (resultSound witness.env) (by decide)

theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply claimOfHistory selector witness (by rfl) (by rfl) (by rfl) (by rfl)
  · exact SemanticRightRootResult2240.claimSound
      selector selectorLower selectorUpper witness
  · exact SemanticRightRootResult2192.claimSound
      selector selectorLower selectorUpper witness
  · rfl

end SemanticRightRootResult2244

namespace SemanticRightRootResult2248

def leftRaw : List Term := SemanticRightRootResult2244.rawTerms
def rightRaw : List Term := SemanticRightRootResult2184.rawTerms
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6398⟩⟩, ⟨.program ⟨214⟩, ⟨17446⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6407⟩⟩, ⟨.program ⟨214⟩, ⟨17830⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6427⟩⟩, ⟨.program ⟨214⟩, ⟨15526⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6452⟩⟩, ⟨.program ⟨214⟩, ⟨15218⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨15057⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨14896⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def owner : Owner := ⟨.program ⟨214⟩, ⟨17833⟩⟩
def rawTerms : List Term := outputRaw
def summary : Bound := .exactZero
def resultEvent : Nat := 2248
theorem resultAgreement : CanonicalAgreement output (add left right) := by
  decide +kernel

theorem resultSound (env : Env Owner) :
    evalPolynomial env output = evalPolynomial env left + evalPolynomial env right := by
  exact addCanonicalResultSound env left right output resultAgreement

def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  SemanticRightRootResult2244.actual selector witness +
    SemanticRightRootResult2184.actual selector witness
theorem claimOfHistory (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841)
    (expressionAt : document.expressions.lookup 17833 =
      some Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Expression069.ExpressionRow17833)
    (leftPredecessorAt : (history.lookup 2245).map AnnotatedEvent.event =
      some (.predecessor owner 0 ⟨17832⟩ 2244))
    (rightPredecessorAt : (history.lookup 2246).map AnnotatedEvent.event =
      some (.predecessor owner 1 ⟨17447⟩ 2184))
    (ruleAt : (history.lookup 2247).map AnnotatedEvent.event =
      some (.boundTransfer owner (.sum [.predecessor 0 2245 .coefficient, .predecessor 1 2246 .coefficient])))
    (leftClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult2244.resultEvent
      SemanticRightRootResult2244.owner
      (SemanticRightRootResult2244.actual selector witness)
      SemanticRightRootResult2244.rawTerms
      SemanticRightRootResult2244.summary)
    (rightClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult2184.resultEvent
      SemanticRightRootResult2184.owner
      (SemanticRightRootResult2184.actual selector witness)
      SemanticRightRootResult2184.rawTerms
      SemanticRightRootResult2184.summary)
    (outputAt : (history.lookup resultEvent).map AnnotatedEvent.event =
      some (.resultExact owner rawTerms summary)) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  refine ⟨outputAt, ?_⟩
  exact exactValueClaim_add_of_mod_zero 100418593683253592432016548326729029359133068138294319235841 witness.env
    (SemanticRightRootResult2244.actual selector witness)
    (SemanticRightRootResult2184.actual selector witness) left right output
    (by simpa [left, leftRaw, SemanticRightRootResult2244.summary] using leftClaim.claim)
    (by simpa [right, rightRaw, SemanticRightRootResult2184.summary] using rightClaim.claim)
    (resultSound witness.env) (by decide)

theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply claimOfHistory selector witness (by rfl) (by rfl) (by rfl) (by rfl)
  · exact SemanticRightRootResult2244.claimSound
      selector selectorLower selectorUpper witness
  · exact SemanticRightRootResult2184.claimSound
      selector selectorLower selectorUpper witness
  · rfl

end SemanticRightRootResult2248

namespace SemanticRightRootResult2252

def leftRaw : List Term := SemanticRightRootResult2248.rawTerms
def rightRaw : List Term := SemanticRightRootResult2176.rawTerms
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6391⟩⟩, ⟨.program ⟨214⟩, ⟨17229⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6398⟩⟩, ⟨.program ⟨214⟩, ⟨17446⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6407⟩⟩, ⟨.program ⟨214⟩, ⟨17830⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6427⟩⟩, ⟨.program ⟨214⟩, ⟨15526⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6452⟩⟩, ⟨.program ⟨214⟩, ⟨15218⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨15057⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨14896⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def owner : Owner := ⟨.program ⟨214⟩, ⟨17834⟩⟩
def rawTerms : List Term := outputRaw
def summary : Bound := .exactZero
def resultEvent : Nat := 2252
theorem resultAgreement : CanonicalAgreement output (add left right) := by
  decide +kernel

theorem resultSound (env : Env Owner) :
    evalPolynomial env output = evalPolynomial env left + evalPolynomial env right := by
  exact addCanonicalResultSound env left right output resultAgreement

def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  SemanticRightRootResult2248.actual selector witness +
    SemanticRightRootResult2176.actual selector witness
theorem claimOfHistory (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841)
    (expressionAt : document.expressions.lookup 17834 =
      some Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Expression069.ExpressionRow17834)
    (leftPredecessorAt : (history.lookup 2249).map AnnotatedEvent.event =
      some (.predecessor owner 0 ⟨17833⟩ 2248))
    (rightPredecessorAt : (history.lookup 2250).map AnnotatedEvent.event =
      some (.predecessor owner 1 ⟨17230⟩ 2176))
    (ruleAt : (history.lookup 2251).map AnnotatedEvent.event =
      some (.boundTransfer owner (.sum [.predecessor 0 2249 .coefficient, .predecessor 1 2250 .coefficient])))
    (leftClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult2248.resultEvent
      SemanticRightRootResult2248.owner
      (SemanticRightRootResult2248.actual selector witness)
      SemanticRightRootResult2248.rawTerms
      SemanticRightRootResult2248.summary)
    (rightClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult2176.resultEvent
      SemanticRightRootResult2176.owner
      (SemanticRightRootResult2176.actual selector witness)
      SemanticRightRootResult2176.rawTerms
      SemanticRightRootResult2176.summary)
    (outputAt : (history.lookup resultEvent).map AnnotatedEvent.event =
      some (.resultExact owner rawTerms summary)) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  refine ⟨outputAt, ?_⟩
  exact exactValueClaim_add_of_mod_zero 100418593683253592432016548326729029359133068138294319235841 witness.env
    (SemanticRightRootResult2248.actual selector witness)
    (SemanticRightRootResult2176.actual selector witness) left right output
    (by simpa [left, leftRaw, SemanticRightRootResult2248.summary] using leftClaim.claim)
    (by simpa [right, rightRaw, SemanticRightRootResult2176.summary] using rightClaim.claim)
    (resultSound witness.env) (by decide)

theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply claimOfHistory selector witness (by rfl) (by rfl) (by rfl) (by rfl)
  · exact SemanticRightRootResult2248.claimSound
      selector selectorLower selectorUpper witness
  · exact SemanticRightRootResult2176.claimSound
      selector selectorLower selectorUpper witness
  · rfl

end SemanticRightRootResult2252

namespace SemanticRightRootResult2256

def leftRaw : List Term := SemanticRightRootResult2252.rawTerms
def rightRaw : List Term := SemanticRightRootResult2168.rawTerms
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6387⟩⟩, ⟨.program ⟨214⟩, ⟨17173⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6391⟩⟩, ⟨.program ⟨214⟩, ⟨17229⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6398⟩⟩, ⟨.program ⟨214⟩, ⟨17446⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6407⟩⟩, ⟨.program ⟨214⟩, ⟨17830⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6427⟩⟩, ⟨.program ⟨214⟩, ⟨15526⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6452⟩⟩, ⟨.program ⟨214⟩, ⟨15218⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨15057⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨14896⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def owner : Owner := ⟨.program ⟨214⟩, ⟨17835⟩⟩
def rawTerms : List Term := outputRaw
def summary : Bound := .exactZero
def resultEvent : Nat := 2256
theorem resultAgreement : CanonicalAgreement output (add left right) := by
  decide +kernel

theorem resultSound (env : Env Owner) :
    evalPolynomial env output = evalPolynomial env left + evalPolynomial env right := by
  exact addCanonicalResultSound env left right output resultAgreement

def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  SemanticRightRootResult2252.actual selector witness +
    SemanticRightRootResult2168.actual selector witness
theorem claimOfHistory (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841)
    (expressionAt : document.expressions.lookup 17835 =
      some Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Expression069.ExpressionRow17835)
    (leftPredecessorAt : (history.lookup 2253).map AnnotatedEvent.event =
      some (.predecessor owner 0 ⟨17834⟩ 2252))
    (rightPredecessorAt : (history.lookup 2254).map AnnotatedEvent.event =
      some (.predecessor owner 1 ⟨17174⟩ 2168))
    (ruleAt : (history.lookup 2255).map AnnotatedEvent.event =
      some (.boundTransfer owner (.sum [.predecessor 0 2253 .coefficient, .predecessor 1 2254 .coefficient])))
    (leftClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult2252.resultEvent
      SemanticRightRootResult2252.owner
      (SemanticRightRootResult2252.actual selector witness)
      SemanticRightRootResult2252.rawTerms
      SemanticRightRootResult2252.summary)
    (rightClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult2168.resultEvent
      SemanticRightRootResult2168.owner
      (SemanticRightRootResult2168.actual selector witness)
      SemanticRightRootResult2168.rawTerms
      SemanticRightRootResult2168.summary)
    (outputAt : (history.lookup resultEvent).map AnnotatedEvent.event =
      some (.resultExact owner rawTerms summary)) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  refine ⟨outputAt, ?_⟩
  exact exactValueClaim_add_of_mod_zero 100418593683253592432016548326729029359133068138294319235841 witness.env
    (SemanticRightRootResult2252.actual selector witness)
    (SemanticRightRootResult2168.actual selector witness) left right output
    (by simpa [left, leftRaw, SemanticRightRootResult2252.summary] using leftClaim.claim)
    (by simpa [right, rightRaw, SemanticRightRootResult2168.summary] using rightClaim.claim)
    (resultSound witness.env) (by decide)

theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply claimOfHistory selector witness (by rfl) (by rfl) (by rfl) (by rfl)
  · exact SemanticRightRootResult2252.claimSound
      selector selectorLower selectorUpper witness
  · exact SemanticRightRootResult2168.claimSound
      selector selectorLower selectorUpper witness
  · rfl

end SemanticRightRootResult2256

namespace SemanticRightRootResult2260

def leftRaw : List Term := SemanticRightRootResult2256.rawTerms
def rightRaw : List Term := SemanticRightRootResult2160.rawTerms
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6383⟩⟩, ⟨.program ⟨214⟩, ⟨18049⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6387⟩⟩, ⟨.program ⟨214⟩, ⟨17173⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6391⟩⟩, ⟨.program ⟨214⟩, ⟨17229⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6398⟩⟩, ⟨.program ⟨214⟩, ⟨17446⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6407⟩⟩, ⟨.program ⟨214⟩, ⟨17830⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6427⟩⟩, ⟨.program ⟨214⟩, ⟨15526⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6452⟩⟩, ⟨.program ⟨214⟩, ⟨15218⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨15057⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨14896⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def owner : Owner := ⟨.program ⟨214⟩, ⟨18051⟩⟩
def rawTerms : List Term := outputRaw
def summary : Bound := .exactZero
def resultEvent : Nat := 2260
theorem resultAgreement : CanonicalAgreement output (add left right) := by
  decide +kernel

theorem resultSound (env : Env Owner) :
    evalPolynomial env output = evalPolynomial env left + evalPolynomial env right := by
  exact addCanonicalResultSound env left right output resultAgreement

def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  SemanticRightRootResult2256.actual selector witness +
    SemanticRightRootResult2160.actual selector witness
theorem claimOfHistory (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841)
    (expressionAt : document.expressions.lookup 18051 =
      some Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Expression070.ExpressionRow18051)
    (leftPredecessorAt : (history.lookup 2257).map AnnotatedEvent.event =
      some (.predecessor owner 0 ⟨17835⟩ 2256))
    (rightPredecessorAt : (history.lookup 2258).map AnnotatedEvent.event =
      some (.predecessor owner 1 ⟨18050⟩ 2160))
    (ruleAt : (history.lookup 2259).map AnnotatedEvent.event =
      some (.boundTransfer owner (.sum [.predecessor 0 2257 .coefficient, .predecessor 1 2258 .coefficient])))
    (leftClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult2256.resultEvent
      SemanticRightRootResult2256.owner
      (SemanticRightRootResult2256.actual selector witness)
      SemanticRightRootResult2256.rawTerms
      SemanticRightRootResult2256.summary)
    (rightClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult2160.resultEvent
      SemanticRightRootResult2160.owner
      (SemanticRightRootResult2160.actual selector witness)
      SemanticRightRootResult2160.rawTerms
      SemanticRightRootResult2160.summary)
    (outputAt : (history.lookup resultEvent).map AnnotatedEvent.event =
      some (.resultExact owner rawTerms summary)) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  refine ⟨outputAt, ?_⟩
  exact exactValueClaim_add_of_mod_zero 100418593683253592432016548326729029359133068138294319235841 witness.env
    (SemanticRightRootResult2256.actual selector witness)
    (SemanticRightRootResult2160.actual selector witness) left right output
    (by simpa [left, leftRaw, SemanticRightRootResult2256.summary] using leftClaim.claim)
    (by simpa [right, rightRaw, SemanticRightRootResult2160.summary] using rightClaim.claim)
    (resultSound witness.env) (by decide)

theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply claimOfHistory selector witness (by rfl) (by rfl) (by rfl) (by rfl)
  · exact SemanticRightRootResult2256.claimSound
      selector selectorLower selectorUpper witness
  · exact SemanticRightRootResult2160.claimSound
      selector selectorLower selectorUpper witness
  · rfl

end SemanticRightRootResult2260

namespace SemanticRightRootResult2264

def leftRaw : List Term := SemanticRightRootResult2260.rawTerms
def rightRaw : List Term := SemanticRightRootResult2152.rawTerms
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6383⟩⟩, ⟨.program ⟨214⟩, ⟨18049⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6387⟩⟩, ⟨.program ⟨214⟩, ⟨17173⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6391⟩⟩, ⟨.program ⟨214⟩, ⟨17229⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6398⟩⟩, ⟨.program ⟨214⟩, ⟨17446⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6407⟩⟩, ⟨.program ⟨214⟩, ⟨17830⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6427⟩⟩, ⟨.program ⟨214⟩, ⟨15526⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6452⟩⟩, ⟨.program ⟨214⟩, ⟨15218⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨15057⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨14896⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6502⟩⟩, ⟨.program ⟨214⟩, ⟨17670⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def owner : Owner := ⟨.program ⟨214⟩, ⟨18052⟩⟩
def rawTerms : List Term := outputRaw
def summary : Bound := .exactZero
def resultEvent : Nat := 2264
theorem resultAgreement : CanonicalAgreement output (add left right) := by
  decide +kernel

theorem resultSound (env : Env Owner) :
    evalPolynomial env output = evalPolynomial env left + evalPolynomial env right := by
  exact addCanonicalResultSound env left right output resultAgreement

def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  SemanticRightRootResult2260.actual selector witness +
    SemanticRightRootResult2152.actual selector witness
theorem claimOfHistory (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841)
    (expressionAt : document.expressions.lookup 18052 =
      some Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Expression070.ExpressionRow18052)
    (leftPredecessorAt : (history.lookup 2261).map AnnotatedEvent.event =
      some (.predecessor owner 0 ⟨18051⟩ 2260))
    (rightPredecessorAt : (history.lookup 2262).map AnnotatedEvent.event =
      some (.predecessor owner 1 ⟨17671⟩ 2152))
    (ruleAt : (history.lookup 2263).map AnnotatedEvent.event =
      some (.boundTransfer owner (.sum [.predecessor 0 2261 .coefficient, .predecessor 1 2262 .coefficient])))
    (leftClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult2260.resultEvent
      SemanticRightRootResult2260.owner
      (SemanticRightRootResult2260.actual selector witness)
      SemanticRightRootResult2260.rawTerms
      SemanticRightRootResult2260.summary)
    (rightClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult2152.resultEvent
      SemanticRightRootResult2152.owner
      (SemanticRightRootResult2152.actual selector witness)
      SemanticRightRootResult2152.rawTerms
      SemanticRightRootResult2152.summary)
    (outputAt : (history.lookup resultEvent).map AnnotatedEvent.event =
      some (.resultExact owner rawTerms summary)) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  refine ⟨outputAt, ?_⟩
  exact exactValueClaim_add_of_mod_zero 100418593683253592432016548326729029359133068138294319235841 witness.env
    (SemanticRightRootResult2260.actual selector witness)
    (SemanticRightRootResult2152.actual selector witness) left right output
    (by simpa [left, leftRaw, SemanticRightRootResult2260.summary] using leftClaim.claim)
    (by simpa [right, rightRaw, SemanticRightRootResult2152.summary] using rightClaim.claim)
    (resultSound witness.env) (by decide)

theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply claimOfHistory selector witness (by rfl) (by rfl) (by rfl) (by rfl)
  · exact SemanticRightRootResult2260.claimSound
      selector selectorLower selectorUpper witness
  · exact SemanticRightRootResult2152.claimSound
      selector selectorLower selectorUpper witness
  · rfl

end SemanticRightRootResult2264

namespace SemanticRightRootResult2268

def leftRaw : List Term := SemanticRightRootResult2264.rawTerms
def rightRaw : List Term := SemanticRightRootResult2144.rawTerms
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6383⟩⟩, ⟨.program ⟨214⟩, ⟨18049⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6387⟩⟩, ⟨.program ⟨214⟩, ⟨17173⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6391⟩⟩, ⟨.program ⟨214⟩, ⟨17229⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6398⟩⟩, ⟨.program ⟨214⟩, ⟨17446⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6407⟩⟩, ⟨.program ⟨214⟩, ⟨17830⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6427⟩⟩, ⟨.program ⟨214⟩, ⟨15526⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6452⟩⟩, ⟨.program ⟨214⟩, ⟨15218⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨15057⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6494⟩⟩, ⟨.program ⟨214⟩, ⟨17614⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨14896⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6502⟩⟩, ⟨.program ⟨214⟩, ⟨17670⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def owner : Owner := ⟨.program ⟨214⟩, ⟨18053⟩⟩
def rawTerms : List Term := outputRaw
def summary : Bound := .exactZero
def resultEvent : Nat := 2268
theorem resultAgreement : CanonicalAgreement output (add left right) := by
  decide +kernel

theorem resultSound (env : Env Owner) :
    evalPolynomial env output = evalPolynomial env left + evalPolynomial env right := by
  exact addCanonicalResultSound env left right output resultAgreement

def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  SemanticRightRootResult2264.actual selector witness +
    SemanticRightRootResult2144.actual selector witness
theorem claimOfHistory (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841)
    (expressionAt : document.expressions.lookup 18053 =
      some Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Expression070.ExpressionRow18053)
    (leftPredecessorAt : (history.lookup 2265).map AnnotatedEvent.event =
      some (.predecessor owner 0 ⟨18052⟩ 2264))
    (rightPredecessorAt : (history.lookup 2266).map AnnotatedEvent.event =
      some (.predecessor owner 1 ⟨17615⟩ 2144))
    (ruleAt : (history.lookup 2267).map AnnotatedEvent.event =
      some (.boundTransfer owner (.sum [.predecessor 0 2265 .coefficient, .predecessor 1 2266 .coefficient])))
    (leftClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult2264.resultEvent
      SemanticRightRootResult2264.owner
      (SemanticRightRootResult2264.actual selector witness)
      SemanticRightRootResult2264.rawTerms
      SemanticRightRootResult2264.summary)
    (rightClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult2144.resultEvent
      SemanticRightRootResult2144.owner
      (SemanticRightRootResult2144.actual selector witness)
      SemanticRightRootResult2144.rawTerms
      SemanticRightRootResult2144.summary)
    (outputAt : (history.lookup resultEvent).map AnnotatedEvent.event =
      some (.resultExact owner rawTerms summary)) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  refine ⟨outputAt, ?_⟩
  exact exactValueClaim_add_of_mod_zero 100418593683253592432016548326729029359133068138294319235841 witness.env
    (SemanticRightRootResult2264.actual selector witness)
    (SemanticRightRootResult2144.actual selector witness) left right output
    (by simpa [left, leftRaw, SemanticRightRootResult2264.summary] using leftClaim.claim)
    (by simpa [right, rightRaw, SemanticRightRootResult2144.summary] using rightClaim.claim)
    (resultSound witness.env) (by decide)

theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply claimOfHistory selector witness (by rfl) (by rfl) (by rfl) (by rfl)
  · exact SemanticRightRootResult2264.claimSound
      selector selectorLower selectorUpper witness
  · exact SemanticRightRootResult2144.claimSound
      selector selectorLower selectorUpper witness
  · rfl

end SemanticRightRootResult2268

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
