import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticRightRootShard033

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open TallSecurity0ABI
open TallSemantics

namespace SemanticRightRootResult6084

def owner : Owner := ⟨.program ⟨214⟩, ⟨6751⟩⟩
def rawTerms : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6751⟩⟩] } }]
def summary : Bound := .exactZero
def producerEvent : Nat := 6083
def resultEvent : Nat := 6084
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

end SemanticRightRootResult6084

namespace SemanticRightRootResult6089

def leftRaw : List Term := SemanticRightRootResult6084.rawTerms
def rightRaw : List Term := SemanticRightRootResult5961.rawTerms
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6751⟩⟩, ⟨.program ⟨214⟩, ⟨7885⟩⟩] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def owner : Owner := ⟨.program ⟨214⟩, ⟨7890⟩⟩
def rawTerms : List Term := outputRaw
def summary : Bound := .exactZero
def resultEvent : Nat := 6089
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
  SemanticRightRootResult6084.actual selector witness *
    SemanticRightRootResult5961.actual selector witness
theorem claimOfHistory (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841)
    (expressionAt : document.expressions.lookup 7890 =
      some Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Expression030.ExpressionRow7890)
    (leftPredecessorAt : (history.lookup 6085).map AnnotatedEvent.event =
      some (.predecessor owner 0 ⟨6751⟩ 6084))
    (rightPredecessorAt : (history.lookup 6086).map AnnotatedEvent.event =
      some (.predecessor owner 1 ⟨7886⟩ 5961))
    (ruleAt : (history.lookup 6087).map AnnotatedEvent.event =
      some (.boundTransfer owner (.product (.predecessor 0 6085 .coefficient) (.predecessor 1 6086 .coefficient) ⟨false, false, none, none, none⟩)))
    (leftClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult6084.resultEvent
      SemanticRightRootResult6084.owner
      (SemanticRightRootResult6084.actual selector witness)
      SemanticRightRootResult6084.rawTerms
      SemanticRightRootResult6084.summary)
    (rightClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult5961.resultEvent
      SemanticRightRootResult5961.owner
      (SemanticRightRootResult5961.actual selector witness)
      SemanticRightRootResult5961.rawTerms
      SemanticRightRootResult5961.summary)
    (outputAt : (history.lookup resultEvent).map AnnotatedEvent.event =
      some (.resultExact owner rawTerms summary)) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  refine ⟨outputAt, ?_⟩
  exact exactValueClaim_product_of_mod_zero 100418593683253592432016548326729029359133068138294319235841 witness.env
    (SemanticRightRootResult6084.actual selector witness)
    (SemanticRightRootResult5961.actual selector witness) left right output
    (by simpa [left, leftRaw, SemanticRightRootResult6084.summary] using leftClaim.claim)
    (by simpa [right, rightRaw, SemanticRightRootResult5961.summary] using rightClaim.claim)
    (resultSound witness.env) (by decide)

theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply claimOfHistory selector witness (by rfl) (by rfl) (by rfl) (by rfl)
  · exact SemanticRightRootResult6084.claimSound
      selector selectorLower selectorUpper witness
  · exact SemanticRightRootResult5961.claimSound
      selector selectorLower selectorUpper witness
  · rfl

end SemanticRightRootResult6089

namespace SemanticRightRootResult6094

def leftRaw : List Term := SemanticRightRootResult6089.rawTerms
def rightRaw : List Term := SemanticRightRootResult6081.rawTerms
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6751⟩⟩, ⟨.program ⟨214⟩, ⟨7885⟩⟩, ⟨.program ⟨214⟩, ⟨7825⟩⟩] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def owner : Owner := ⟨.program ⟨214⟩, ⟨7914⟩⟩
def rawTerms : List Term := outputRaw
def summary : Bound := .exactZero
def resultEvent : Nat := 6094
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
  SemanticRightRootResult6089.actual selector witness *
    SemanticRightRootResult6081.actual selector witness
theorem claimOfHistory (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841)
    (expressionAt : document.expressions.lookup 7914 =
      some Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Expression030.ExpressionRow7914)
    (leftPredecessorAt : (history.lookup 6090).map AnnotatedEvent.event =
      some (.predecessor owner 0 ⟨7890⟩ 6089))
    (rightPredecessorAt : (history.lookup 6091).map AnnotatedEvent.event =
      some (.predecessor owner 1 ⟨7826⟩ 6081))
    (ruleAt : (history.lookup 6092).map AnnotatedEvent.event =
      some (.boundTransfer owner (.product (.predecessor 0 6090 .coefficient) (.predecessor 1 6091 .coefficient) ⟨false, false, none, none, none⟩)))
    (leftClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult6089.resultEvent
      SemanticRightRootResult6089.owner
      (SemanticRightRootResult6089.actual selector witness)
      SemanticRightRootResult6089.rawTerms
      SemanticRightRootResult6089.summary)
    (rightClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult6081.resultEvent
      SemanticRightRootResult6081.owner
      (SemanticRightRootResult6081.actual selector witness)
      SemanticRightRootResult6081.rawTerms
      SemanticRightRootResult6081.summary)
    (outputAt : (history.lookup resultEvent).map AnnotatedEvent.event =
      some (.resultExact owner rawTerms summary)) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  refine ⟨outputAt, ?_⟩
  exact exactValueClaim_product_of_mod_zero 100418593683253592432016548326729029359133068138294319235841 witness.env
    (SemanticRightRootResult6089.actual selector witness)
    (SemanticRightRootResult6081.actual selector witness) left right output
    (by simpa [left, leftRaw, SemanticRightRootResult6089.summary] using leftClaim.claim)
    (by simpa [right, rightRaw, SemanticRightRootResult6081.summary] using rightClaim.claim)
    (resultSound witness.env) (by decide)

theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply claimOfHistory selector witness (by rfl) (by rfl) (by rfl) (by rfl)
  · exact SemanticRightRootResult6089.claimSound
      selector selectorLower selectorUpper witness
  · exact SemanticRightRootResult6081.claimSound
      selector selectorLower selectorUpper witness
  · rfl

end SemanticRightRootResult6094

namespace SemanticRightRootResult6099

def leftRaw : List Term := SemanticRightRootResult6094.rawTerms
def rightRaw : List Term := SemanticRightRootResult6071.rawTerms
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6751⟩⟩, ⟨.program ⟨214⟩, ⟨7885⟩⟩, ⟨.program ⟨214⟩, ⟨7825⟩⟩, ⟨.program ⟨214⟩, ⟨6675⟩⟩] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def owner : Owner := ⟨.program ⟨214⟩, ⟨7920⟩⟩
def rawTerms : List Term := outputRaw
def summary : Bound := .exactZero
def resultEvent : Nat := 6099
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
  SemanticRightRootResult6094.actual selector witness *
    SemanticRightRootResult6071.actual selector witness
theorem claimOfHistory (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841)
    (expressionAt : document.expressions.lookup 7920 =
      some Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Expression030.ExpressionRow7920)
    (leftPredecessorAt : (history.lookup 6095).map AnnotatedEvent.event =
      some (.predecessor owner 0 ⟨7914⟩ 6094))
    (rightPredecessorAt : (history.lookup 6096).map AnnotatedEvent.event =
      some (.predecessor owner 1 ⟨6676⟩ 6071))
    (ruleAt : (history.lookup 6097).map AnnotatedEvent.event =
      some (.boundTransfer owner (.product (.predecessor 0 6095 .coefficient) (.predecessor 1 6096 .coefficient) ⟨false, false, none, none, none⟩)))
    (leftClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult6094.resultEvent
      SemanticRightRootResult6094.owner
      (SemanticRightRootResult6094.actual selector witness)
      SemanticRightRootResult6094.rawTerms
      SemanticRightRootResult6094.summary)
    (rightClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult6071.resultEvent
      SemanticRightRootResult6071.owner
      (SemanticRightRootResult6071.actual selector witness)
      SemanticRightRootResult6071.rawTerms
      SemanticRightRootResult6071.summary)
    (outputAt : (history.lookup resultEvent).map AnnotatedEvent.event =
      some (.resultExact owner rawTerms summary)) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  refine ⟨outputAt, ?_⟩
  exact exactValueClaim_product_of_mod_zero 100418593683253592432016548326729029359133068138294319235841 witness.env
    (SemanticRightRootResult6094.actual selector witness)
    (SemanticRightRootResult6071.actual selector witness) left right output
    (by simpa [left, leftRaw, SemanticRightRootResult6094.summary] using leftClaim.claim)
    (by simpa [right, rightRaw, SemanticRightRootResult6071.summary] using rightClaim.claim)
    (resultSound witness.env) (by decide)

theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply claimOfHistory selector witness (by rfl) (by rfl) (by rfl) (by rfl)
  · exact SemanticRightRootResult6094.claimSound
      selector selectorLower selectorUpper witness
  · exact SemanticRightRootResult6071.claimSound
      selector selectorLower selectorUpper witness
  · rfl

end SemanticRightRootResult6099

namespace SemanticRightRootResult6111

def owner : Owner := ⟨.program ⟨214⟩, ⟨6686⟩⟩
def rawTerms : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6685⟩⟩] } }]
def summary : Bound := .exactZero
def producerEvent : Nat := 6110
def resultEvent : Nat := 6111
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual resultEvent
theorem terminalAt (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, rightRootOwnerAtSelector214 selector selectorLower selectorUpper _, ?_⟩
  refine ⟨.scale (.predecessor 0 6108 .coefficient) (.value (.predecessor 1 6109 .coefficient)), 0, .scale (.predecessor 0 6108 .coefficient) (.value (.predecessor 1 6109 .coefficient)), ?_, ?_⟩
  · rfl
  · rfl
theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness
    (terminalAt selector selectorLower selectorUpper)

end SemanticRightRootResult6111

namespace SemanticRightRootResult6121

def owner : Owner := ⟨.program ⟨214⟩, ⟨7828⟩⟩
def rawTerms : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨7827⟩⟩] } }]
def summary : Bound := .exactZero
def producerEvent : Nat := 6120
def resultEvent : Nat := 6121
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual resultEvent
theorem terminalAt (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, rightRootOwnerAtSelector214 selector selectorLower selectorUpper _, ?_⟩
  refine ⟨.scale (.predecessor 0 6118 .coefficient) (.value (.predecessor 1 6119 .coefficient)), 0, .scale (.predecessor 0 6118 .coefficient) (.value (.predecessor 1 6119 .coefficient)), ?_, ?_⟩
  · rfl
  · rfl
theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness
    (terminalAt selector selectorLower selectorUpper)

end SemanticRightRootResult6121

namespace SemanticRightRootResult6124

def owner : Owner := ⟨.program ⟨214⟩, ⟨6753⟩⟩
def rawTerms : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6753⟩⟩] } }]
def summary : Bound := .exactZero
def producerEvent : Nat := 6123
def resultEvent : Nat := 6124
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

end SemanticRightRootResult6124

namespace SemanticRightRootResult6129

def leftRaw : List Term := SemanticRightRootResult6124.rawTerms
def rightRaw : List Term := SemanticRightRootResult5961.rawTerms
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6753⟩⟩, ⟨.program ⟨214⟩, ⟨7885⟩⟩] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def owner : Owner := ⟨.program ⟨214⟩, ⟨7891⟩⟩
def rawTerms : List Term := outputRaw
def summary : Bound := .exactZero
def resultEvent : Nat := 6129
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
  SemanticRightRootResult6124.actual selector witness *
    SemanticRightRootResult5961.actual selector witness
theorem claimOfHistory (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841)
    (expressionAt : document.expressions.lookup 7891 =
      some Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Expression030.ExpressionRow7891)
    (leftPredecessorAt : (history.lookup 6125).map AnnotatedEvent.event =
      some (.predecessor owner 0 ⟨6753⟩ 6124))
    (rightPredecessorAt : (history.lookup 6126).map AnnotatedEvent.event =
      some (.predecessor owner 1 ⟨7886⟩ 5961))
    (ruleAt : (history.lookup 6127).map AnnotatedEvent.event =
      some (.boundTransfer owner (.product (.predecessor 0 6125 .coefficient) (.predecessor 1 6126 .coefficient) ⟨false, false, none, none, none⟩)))
    (leftClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult6124.resultEvent
      SemanticRightRootResult6124.owner
      (SemanticRightRootResult6124.actual selector witness)
      SemanticRightRootResult6124.rawTerms
      SemanticRightRootResult6124.summary)
    (rightClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult5961.resultEvent
      SemanticRightRootResult5961.owner
      (SemanticRightRootResult5961.actual selector witness)
      SemanticRightRootResult5961.rawTerms
      SemanticRightRootResult5961.summary)
    (outputAt : (history.lookup resultEvent).map AnnotatedEvent.event =
      some (.resultExact owner rawTerms summary)) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  refine ⟨outputAt, ?_⟩
  exact exactValueClaim_product_of_mod_zero 100418593683253592432016548326729029359133068138294319235841 witness.env
    (SemanticRightRootResult6124.actual selector witness)
    (SemanticRightRootResult5961.actual selector witness) left right output
    (by simpa [left, leftRaw, SemanticRightRootResult6124.summary] using leftClaim.claim)
    (by simpa [right, rightRaw, SemanticRightRootResult5961.summary] using rightClaim.claim)
    (resultSound witness.env) (by decide)

theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply claimOfHistory selector witness (by rfl) (by rfl) (by rfl) (by rfl)
  · exact SemanticRightRootResult6124.claimSound
      selector selectorLower selectorUpper witness
  · exact SemanticRightRootResult5961.claimSound
      selector selectorLower selectorUpper witness
  · rfl

end SemanticRightRootResult6129

namespace SemanticRightRootResult6134

def leftRaw : List Term := SemanticRightRootResult6129.rawTerms
def rightRaw : List Term := SemanticRightRootResult6121.rawTerms
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6753⟩⟩, ⟨.program ⟨214⟩, ⟨7885⟩⟩, ⟨.program ⟨214⟩, ⟨7827⟩⟩] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def owner : Owner := ⟨.program ⟨214⟩, ⟨7915⟩⟩
def rawTerms : List Term := outputRaw
def summary : Bound := .exactZero
def resultEvent : Nat := 6134
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
  SemanticRightRootResult6129.actual selector witness *
    SemanticRightRootResult6121.actual selector witness
theorem claimOfHistory (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841)
    (expressionAt : document.expressions.lookup 7915 =
      some Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Expression030.ExpressionRow7915)
    (leftPredecessorAt : (history.lookup 6130).map AnnotatedEvent.event =
      some (.predecessor owner 0 ⟨7891⟩ 6129))
    (rightPredecessorAt : (history.lookup 6131).map AnnotatedEvent.event =
      some (.predecessor owner 1 ⟨7828⟩ 6121))
    (ruleAt : (history.lookup 6132).map AnnotatedEvent.event =
      some (.boundTransfer owner (.product (.predecessor 0 6130 .coefficient) (.predecessor 1 6131 .coefficient) ⟨false, false, none, none, none⟩)))
    (leftClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult6129.resultEvent
      SemanticRightRootResult6129.owner
      (SemanticRightRootResult6129.actual selector witness)
      SemanticRightRootResult6129.rawTerms
      SemanticRightRootResult6129.summary)
    (rightClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult6121.resultEvent
      SemanticRightRootResult6121.owner
      (SemanticRightRootResult6121.actual selector witness)
      SemanticRightRootResult6121.rawTerms
      SemanticRightRootResult6121.summary)
    (outputAt : (history.lookup resultEvent).map AnnotatedEvent.event =
      some (.resultExact owner rawTerms summary)) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  refine ⟨outputAt, ?_⟩
  exact exactValueClaim_product_of_mod_zero 100418593683253592432016548326729029359133068138294319235841 witness.env
    (SemanticRightRootResult6129.actual selector witness)
    (SemanticRightRootResult6121.actual selector witness) left right output
    (by simpa [left, leftRaw, SemanticRightRootResult6129.summary] using leftClaim.claim)
    (by simpa [right, rightRaw, SemanticRightRootResult6121.summary] using rightClaim.claim)
    (resultSound witness.env) (by decide)

theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply claimOfHistory selector witness (by rfl) (by rfl) (by rfl) (by rfl)
  · exact SemanticRightRootResult6129.claimSound
      selector selectorLower selectorUpper witness
  · exact SemanticRightRootResult6121.claimSound
      selector selectorLower selectorUpper witness
  · rfl

end SemanticRightRootResult6134

namespace SemanticRightRootResult6139

def leftRaw : List Term := SemanticRightRootResult6134.rawTerms
def rightRaw : List Term := SemanticRightRootResult6111.rawTerms
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6753⟩⟩, ⟨.program ⟨214⟩, ⟨7885⟩⟩, ⟨.program ⟨214⟩, ⟨7827⟩⟩, ⟨.program ⟨214⟩, ⟨6685⟩⟩] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def owner : Owner := ⟨.program ⟨214⟩, ⟨7921⟩⟩
def rawTerms : List Term := outputRaw
def summary : Bound := .exactZero
def resultEvent : Nat := 6139
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
  SemanticRightRootResult6134.actual selector witness *
    SemanticRightRootResult6111.actual selector witness
theorem claimOfHistory (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841)
    (expressionAt : document.expressions.lookup 7921 =
      some Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Expression030.ExpressionRow7921)
    (leftPredecessorAt : (history.lookup 6135).map AnnotatedEvent.event =
      some (.predecessor owner 0 ⟨7915⟩ 6134))
    (rightPredecessorAt : (history.lookup 6136).map AnnotatedEvent.event =
      some (.predecessor owner 1 ⟨6686⟩ 6111))
    (ruleAt : (history.lookup 6137).map AnnotatedEvent.event =
      some (.boundTransfer owner (.product (.predecessor 0 6135 .coefficient) (.predecessor 1 6136 .coefficient) ⟨false, false, none, none, none⟩)))
    (leftClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult6134.resultEvent
      SemanticRightRootResult6134.owner
      (SemanticRightRootResult6134.actual selector witness)
      SemanticRightRootResult6134.rawTerms
      SemanticRightRootResult6134.summary)
    (rightClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult6111.resultEvent
      SemanticRightRootResult6111.owner
      (SemanticRightRootResult6111.actual selector witness)
      SemanticRightRootResult6111.rawTerms
      SemanticRightRootResult6111.summary)
    (outputAt : (history.lookup resultEvent).map AnnotatedEvent.event =
      some (.resultExact owner rawTerms summary)) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  refine ⟨outputAt, ?_⟩
  exact exactValueClaim_product_of_mod_zero 100418593683253592432016548326729029359133068138294319235841 witness.env
    (SemanticRightRootResult6134.actual selector witness)
    (SemanticRightRootResult6111.actual selector witness) left right output
    (by simpa [left, leftRaw, SemanticRightRootResult6134.summary] using leftClaim.claim)
    (by simpa [right, rightRaw, SemanticRightRootResult6111.summary] using rightClaim.claim)
    (resultSound witness.env) (by decide)

theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply claimOfHistory selector witness (by rfl) (by rfl) (by rfl) (by rfl)
  · exact SemanticRightRootResult6134.claimSound
      selector selectorLower selectorUpper witness
  · exact SemanticRightRootResult6111.claimSound
      selector selectorLower selectorUpper witness
  · rfl

end SemanticRightRootResult6139

namespace SemanticRightRootResult6151

def owner : Owner := ⟨.program ⟨214⟩, ⟨6684⟩⟩
def rawTerms : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6683⟩⟩] } }]
def summary : Bound := .exactZero
def producerEvent : Nat := 6150
def resultEvent : Nat := 6151
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual resultEvent
theorem terminalAt (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, rightRootOwnerAtSelector214 selector selectorLower selectorUpper _, ?_⟩
  refine ⟨.scale (.predecessor 0 6148 .coefficient) (.value (.predecessor 1 6149 .coefficient)), 0, .scale (.predecessor 0 6148 .coefficient) (.value (.predecessor 1 6149 .coefficient)), ?_, ?_⟩
  · rfl
  · rfl
theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness
    (terminalAt selector selectorLower selectorUpper)

end SemanticRightRootResult6151

namespace SemanticRightRootResult6161

def owner : Owner := ⟨.program ⟨214⟩, ⟨7830⟩⟩
def rawTerms : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨7829⟩⟩] } }]
def summary : Bound := .exactZero
def producerEvent : Nat := 6160
def resultEvent : Nat := 6161
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual resultEvent
theorem terminalAt (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, rightRootOwnerAtSelector214 selector selectorLower selectorUpper _, ?_⟩
  refine ⟨.scale (.predecessor 0 6158 .coefficient) (.value (.predecessor 1 6159 .coefficient)), 0, .scale (.predecessor 0 6158 .coefficient) (.value (.predecessor 1 6159 .coefficient)), ?_, ?_⟩
  · rfl
  · rfl
theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness
    (terminalAt selector selectorLower selectorUpper)

end SemanticRightRootResult6161

namespace SemanticRightRootResult6164

def owner : Owner := ⟨.program ⟨214⟩, ⟨6755⟩⟩
def rawTerms : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6755⟩⟩] } }]
def summary : Bound := .exactZero
def producerEvent : Nat := 6163
def resultEvent : Nat := 6164
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

end SemanticRightRootResult6164

namespace SemanticRightRootResult6169

def leftRaw : List Term := SemanticRightRootResult6164.rawTerms
def rightRaw : List Term := SemanticRightRootResult5961.rawTerms
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6755⟩⟩, ⟨.program ⟨214⟩, ⟨7885⟩⟩] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def owner : Owner := ⟨.program ⟨214⟩, ⟨7892⟩⟩
def rawTerms : List Term := outputRaw
def summary : Bound := .exactZero
def resultEvent : Nat := 6169
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
  SemanticRightRootResult6164.actual selector witness *
    SemanticRightRootResult5961.actual selector witness
theorem claimOfHistory (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841)
    (expressionAt : document.expressions.lookup 7892 =
      some Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Expression030.ExpressionRow7892)
    (leftPredecessorAt : (history.lookup 6165).map AnnotatedEvent.event =
      some (.predecessor owner 0 ⟨6755⟩ 6164))
    (rightPredecessorAt : (history.lookup 6166).map AnnotatedEvent.event =
      some (.predecessor owner 1 ⟨7886⟩ 5961))
    (ruleAt : (history.lookup 6167).map AnnotatedEvent.event =
      some (.boundTransfer owner (.product (.predecessor 0 6165 .coefficient) (.predecessor 1 6166 .coefficient) ⟨false, false, none, none, none⟩)))
    (leftClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult6164.resultEvent
      SemanticRightRootResult6164.owner
      (SemanticRightRootResult6164.actual selector witness)
      SemanticRightRootResult6164.rawTerms
      SemanticRightRootResult6164.summary)
    (rightClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult5961.resultEvent
      SemanticRightRootResult5961.owner
      (SemanticRightRootResult5961.actual selector witness)
      SemanticRightRootResult5961.rawTerms
      SemanticRightRootResult5961.summary)
    (outputAt : (history.lookup resultEvent).map AnnotatedEvent.event =
      some (.resultExact owner rawTerms summary)) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  refine ⟨outputAt, ?_⟩
  exact exactValueClaim_product_of_mod_zero 100418593683253592432016548326729029359133068138294319235841 witness.env
    (SemanticRightRootResult6164.actual selector witness)
    (SemanticRightRootResult5961.actual selector witness) left right output
    (by simpa [left, leftRaw, SemanticRightRootResult6164.summary] using leftClaim.claim)
    (by simpa [right, rightRaw, SemanticRightRootResult5961.summary] using rightClaim.claim)
    (resultSound witness.env) (by decide)

theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply claimOfHistory selector witness (by rfl) (by rfl) (by rfl) (by rfl)
  · exact SemanticRightRootResult6164.claimSound
      selector selectorLower selectorUpper witness
  · exact SemanticRightRootResult5961.claimSound
      selector selectorLower selectorUpper witness
  · rfl

end SemanticRightRootResult6169

namespace SemanticRightRootResult6174

def leftRaw : List Term := SemanticRightRootResult6169.rawTerms
def rightRaw : List Term := SemanticRightRootResult6161.rawTerms
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6755⟩⟩, ⟨.program ⟨214⟩, ⟨7885⟩⟩, ⟨.program ⟨214⟩, ⟨7829⟩⟩] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def owner : Owner := ⟨.program ⟨214⟩, ⟨7916⟩⟩
def rawTerms : List Term := outputRaw
def summary : Bound := .exactZero
def resultEvent : Nat := 6174
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
  SemanticRightRootResult6169.actual selector witness *
    SemanticRightRootResult6161.actual selector witness
theorem claimOfHistory (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841)
    (expressionAt : document.expressions.lookup 7916 =
      some Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Expression030.ExpressionRow7916)
    (leftPredecessorAt : (history.lookup 6170).map AnnotatedEvent.event =
      some (.predecessor owner 0 ⟨7892⟩ 6169))
    (rightPredecessorAt : (history.lookup 6171).map AnnotatedEvent.event =
      some (.predecessor owner 1 ⟨7830⟩ 6161))
    (ruleAt : (history.lookup 6172).map AnnotatedEvent.event =
      some (.boundTransfer owner (.product (.predecessor 0 6170 .coefficient) (.predecessor 1 6171 .coefficient) ⟨false, false, none, none, none⟩)))
    (leftClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult6169.resultEvent
      SemanticRightRootResult6169.owner
      (SemanticRightRootResult6169.actual selector witness)
      SemanticRightRootResult6169.rawTerms
      SemanticRightRootResult6169.summary)
    (rightClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult6161.resultEvent
      SemanticRightRootResult6161.owner
      (SemanticRightRootResult6161.actual selector witness)
      SemanticRightRootResult6161.rawTerms
      SemanticRightRootResult6161.summary)
    (outputAt : (history.lookup resultEvent).map AnnotatedEvent.event =
      some (.resultExact owner rawTerms summary)) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  refine ⟨outputAt, ?_⟩
  exact exactValueClaim_product_of_mod_zero 100418593683253592432016548326729029359133068138294319235841 witness.env
    (SemanticRightRootResult6169.actual selector witness)
    (SemanticRightRootResult6161.actual selector witness) left right output
    (by simpa [left, leftRaw, SemanticRightRootResult6169.summary] using leftClaim.claim)
    (by simpa [right, rightRaw, SemanticRightRootResult6161.summary] using rightClaim.claim)
    (resultSound witness.env) (by decide)

theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply claimOfHistory selector witness (by rfl) (by rfl) (by rfl) (by rfl)
  · exact SemanticRightRootResult6169.claimSound
      selector selectorLower selectorUpper witness
  · exact SemanticRightRootResult6161.claimSound
      selector selectorLower selectorUpper witness
  · rfl

end SemanticRightRootResult6174

namespace SemanticRightRootResult6179

def leftRaw : List Term := SemanticRightRootResult6174.rawTerms
def rightRaw : List Term := SemanticRightRootResult6151.rawTerms
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6755⟩⟩, ⟨.program ⟨214⟩, ⟨7885⟩⟩, ⟨.program ⟨214⟩, ⟨7829⟩⟩, ⟨.program ⟨214⟩, ⟨6683⟩⟩] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def owner : Owner := ⟨.program ⟨214⟩, ⟨7922⟩⟩
def rawTerms : List Term := outputRaw
def summary : Bound := .exactZero
def resultEvent : Nat := 6179
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
  SemanticRightRootResult6174.actual selector witness *
    SemanticRightRootResult6151.actual selector witness
theorem claimOfHistory (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841)
    (expressionAt : document.expressions.lookup 7922 =
      some Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Expression030.ExpressionRow7922)
    (leftPredecessorAt : (history.lookup 6175).map AnnotatedEvent.event =
      some (.predecessor owner 0 ⟨7916⟩ 6174))
    (rightPredecessorAt : (history.lookup 6176).map AnnotatedEvent.event =
      some (.predecessor owner 1 ⟨6684⟩ 6151))
    (ruleAt : (history.lookup 6177).map AnnotatedEvent.event =
      some (.boundTransfer owner (.product (.predecessor 0 6175 .coefficient) (.predecessor 1 6176 .coefficient) ⟨false, false, none, none, none⟩)))
    (leftClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult6174.resultEvent
      SemanticRightRootResult6174.owner
      (SemanticRightRootResult6174.actual selector witness)
      SemanticRightRootResult6174.rawTerms
      SemanticRightRootResult6174.summary)
    (rightClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult6151.resultEvent
      SemanticRightRootResult6151.owner
      (SemanticRightRootResult6151.actual selector witness)
      SemanticRightRootResult6151.rawTerms
      SemanticRightRootResult6151.summary)
    (outputAt : (history.lookup resultEvent).map AnnotatedEvent.event =
      some (.resultExact owner rawTerms summary)) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  refine ⟨outputAt, ?_⟩
  exact exactValueClaim_product_of_mod_zero 100418593683253592432016548326729029359133068138294319235841 witness.env
    (SemanticRightRootResult6174.actual selector witness)
    (SemanticRightRootResult6151.actual selector witness) left right output
    (by simpa [left, leftRaw, SemanticRightRootResult6174.summary] using leftClaim.claim)
    (by simpa [right, rightRaw, SemanticRightRootResult6151.summary] using rightClaim.claim)
    (resultSound witness.env) (by decide)

theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply claimOfHistory selector witness (by rfl) (by rfl) (by rfl) (by rfl)
  · exact SemanticRightRootResult6174.claimSound
      selector selectorLower selectorUpper witness
  · exact SemanticRightRootResult6151.claimSound
      selector selectorLower selectorUpper witness
  · rfl

end SemanticRightRootResult6179

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
