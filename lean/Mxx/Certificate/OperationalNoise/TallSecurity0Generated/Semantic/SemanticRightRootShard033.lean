import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticRightRootShard032

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open TallSecurity0ABI
open TallSemantics

namespace SemanticRightRootResult5974

def leftRaw : List Term := SemanticRightRootResult5969.rawTerms
def rightRaw : List Term := SemanticRightRootResult5487.rawTerms
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6745⟩⟩, ⟨.program ⟨214⟩, ⟨7885⟩⟩, ⟨.program ⟨214⟩, ⟨7819⟩⟩] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def owner : Owner := ⟨.program ⟨214⟩, ⟨7911⟩⟩
def rawTerms : List Term := outputRaw
def summary : Bound := .exactZero
def resultEvent : Nat := 5974
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
  SemanticRightRootResult5969.actual selector witness *
    SemanticRightRootResult5487.actual selector witness
theorem claimOfHistory (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841)
    (expressionAt : document.expressions.lookup 7911 =
      some Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Expression030.ExpressionRow7911)
    (leftPredecessorAt : (history.lookup 5970).map AnnotatedEvent.event =
      some (.predecessor owner 0 ⟨7887⟩ 5969))
    (rightPredecessorAt : (history.lookup 5971).map AnnotatedEvent.event =
      some (.predecessor owner 1 ⟨7820⟩ 5487))
    (ruleAt : (history.lookup 5972).map AnnotatedEvent.event =
      some (.boundTransfer owner (.product (.predecessor 0 5970 .coefficient) (.predecessor 1 5971 .coefficient) ⟨false, false, none, none, none⟩)))
    (leftClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult5969.resultEvent
      SemanticRightRootResult5969.owner
      (SemanticRightRootResult5969.actual selector witness)
      SemanticRightRootResult5969.rawTerms
      SemanticRightRootResult5969.summary)
    (rightClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult5487.resultEvent
      SemanticRightRootResult5487.owner
      (SemanticRightRootResult5487.actual selector witness)
      SemanticRightRootResult5487.rawTerms
      SemanticRightRootResult5487.summary)
    (outputAt : (history.lookup resultEvent).map AnnotatedEvent.event =
      some (.resultExact owner rawTerms summary)) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  refine ⟨outputAt, ?_⟩
  exact exactValueClaim_product_of_mod_zero 100418593683253592432016548326729029359133068138294319235841 witness.env
    (SemanticRightRootResult5969.actual selector witness)
    (SemanticRightRootResult5487.actual selector witness) left right output
    (by simpa [left, leftRaw, SemanticRightRootResult5969.summary] using leftClaim.claim)
    (by simpa [right, rightRaw, SemanticRightRootResult5487.summary] using rightClaim.claim)
    (resultSound witness.env) (by decide)

theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply claimOfHistory selector witness (by rfl) (by rfl) (by rfl) (by rfl)
  · exact SemanticRightRootResult5969.claimSound
      selector selectorLower selectorUpper witness
  · exact SemanticRightRootResult5487.claimSound
      selector selectorLower selectorUpper witness
  · rfl

end SemanticRightRootResult5974

namespace SemanticRightRootResult5979

def leftRaw : List Term := SemanticRightRootResult5974.rawTerms
def rightRaw : List Term := SemanticRightRootResult5476.rawTerms
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6745⟩⟩, ⟨.program ⟨214⟩, ⟨7885⟩⟩, ⟨.program ⟨214⟩, ⟨7819⟩⟩, ⟨.program ⟨214⟩, ⟨6645⟩⟩] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def owner : Owner := ⟨.program ⟨214⟩, ⟨7917⟩⟩
def rawTerms : List Term := outputRaw
def summary : Bound := .exactZero
def resultEvent : Nat := 5979
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
  SemanticRightRootResult5974.actual selector witness *
    SemanticRightRootResult5476.actual selector witness
theorem claimOfHistory (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841)
    (expressionAt : document.expressions.lookup 7917 =
      some Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Expression030.ExpressionRow7917)
    (leftPredecessorAt : (history.lookup 5975).map AnnotatedEvent.event =
      some (.predecessor owner 0 ⟨7911⟩ 5974))
    (rightPredecessorAt : (history.lookup 5976).map AnnotatedEvent.event =
      some (.predecessor owner 1 ⟨6646⟩ 5476))
    (ruleAt : (history.lookup 5977).map AnnotatedEvent.event =
      some (.boundTransfer owner (.product (.predecessor 0 5975 .coefficient) (.predecessor 1 5976 .coefficient) ⟨false, false, none, none, none⟩)))
    (leftClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult5974.resultEvent
      SemanticRightRootResult5974.owner
      (SemanticRightRootResult5974.actual selector witness)
      SemanticRightRootResult5974.rawTerms
      SemanticRightRootResult5974.summary)
    (rightClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult5476.resultEvent
      SemanticRightRootResult5476.owner
      (SemanticRightRootResult5476.actual selector witness)
      SemanticRightRootResult5476.rawTerms
      SemanticRightRootResult5476.summary)
    (outputAt : (history.lookup resultEvent).map AnnotatedEvent.event =
      some (.resultExact owner rawTerms summary)) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  refine ⟨outputAt, ?_⟩
  exact exactValueClaim_product_of_mod_zero 100418593683253592432016548326729029359133068138294319235841 witness.env
    (SemanticRightRootResult5974.actual selector witness)
    (SemanticRightRootResult5476.actual selector witness) left right output
    (by simpa [left, leftRaw, SemanticRightRootResult5974.summary] using leftClaim.claim)
    (by simpa [right, rightRaw, SemanticRightRootResult5476.summary] using rightClaim.claim)
    (resultSound witness.env) (by decide)

theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply claimOfHistory selector witness (by rfl) (by rfl) (by rfl) (by rfl)
  · exact SemanticRightRootResult5974.claimSound
      selector selectorLower selectorUpper witness
  · exact SemanticRightRootResult5476.claimSound
      selector selectorLower selectorUpper witness
  · rfl

end SemanticRightRootResult5979

namespace SemanticRightRootResult5991

def owner : Owner := ⟨.program ⟨214⟩, ⟨6688⟩⟩
def rawTerms : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6687⟩⟩] } }]
def summary : Bound := .exactZero
def producerEvent : Nat := 5990
def resultEvent : Nat := 5991
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual resultEvent
theorem terminalAt (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, rightRootOwnerAtSelector214 selector selectorLower selectorUpper _, ?_⟩
  refine ⟨.scale (.predecessor 0 5988 .coefficient) (.value (.predecessor 1 5989 .coefficient)), 0, .scale (.predecessor 0 5988 .coefficient) (.value (.predecessor 1 5989 .coefficient)), ?_, ?_⟩
  · rfl
  · rfl
theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness
    (terminalAt selector selectorLower selectorUpper)

end SemanticRightRootResult5991

namespace SemanticRightRootResult6001

def owner : Owner := ⟨.program ⟨214⟩, ⟨7822⟩⟩
def rawTerms : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨7821⟩⟩] } }]
def summary : Bound := .exactZero
def producerEvent : Nat := 6000
def resultEvent : Nat := 6001
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual resultEvent
theorem terminalAt (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, rightRootOwnerAtSelector214 selector selectorLower selectorUpper _, ?_⟩
  refine ⟨.scale (.predecessor 0 5998 .coefficient) (.value (.predecessor 1 5999 .coefficient)), 0, .scale (.predecessor 0 5998 .coefficient) (.value (.predecessor 1 5999 .coefficient)), ?_, ?_⟩
  · rfl
  · rfl
theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness
    (terminalAt selector selectorLower selectorUpper)

end SemanticRightRootResult6001

namespace SemanticRightRootResult6004

def owner : Owner := ⟨.program ⟨214⟩, ⟨6747⟩⟩
def rawTerms : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6747⟩⟩] } }]
def summary : Bound := .exactZero
def producerEvent : Nat := 6003
def resultEvent : Nat := 6004
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

end SemanticRightRootResult6004

namespace SemanticRightRootResult6009

def leftRaw : List Term := SemanticRightRootResult6004.rawTerms
def rightRaw : List Term := SemanticRightRootResult5961.rawTerms
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6747⟩⟩, ⟨.program ⟨214⟩, ⟨7885⟩⟩] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def owner : Owner := ⟨.program ⟨214⟩, ⟨7888⟩⟩
def rawTerms : List Term := outputRaw
def summary : Bound := .exactZero
def resultEvent : Nat := 6009
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
  SemanticRightRootResult6004.actual selector witness *
    SemanticRightRootResult5961.actual selector witness
theorem claimOfHistory (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841)
    (expressionAt : document.expressions.lookup 7888 =
      some Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Expression030.ExpressionRow7888)
    (leftPredecessorAt : (history.lookup 6005).map AnnotatedEvent.event =
      some (.predecessor owner 0 ⟨6747⟩ 6004))
    (rightPredecessorAt : (history.lookup 6006).map AnnotatedEvent.event =
      some (.predecessor owner 1 ⟨7886⟩ 5961))
    (ruleAt : (history.lookup 6007).map AnnotatedEvent.event =
      some (.boundTransfer owner (.product (.predecessor 0 6005 .coefficient) (.predecessor 1 6006 .coefficient) ⟨false, false, none, none, none⟩)))
    (leftClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult6004.resultEvent
      SemanticRightRootResult6004.owner
      (SemanticRightRootResult6004.actual selector witness)
      SemanticRightRootResult6004.rawTerms
      SemanticRightRootResult6004.summary)
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
    (SemanticRightRootResult6004.actual selector witness)
    (SemanticRightRootResult5961.actual selector witness) left right output
    (by simpa [left, leftRaw, SemanticRightRootResult6004.summary] using leftClaim.claim)
    (by simpa [right, rightRaw, SemanticRightRootResult5961.summary] using rightClaim.claim)
    (resultSound witness.env) (by decide)

theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply claimOfHistory selector witness (by rfl) (by rfl) (by rfl) (by rfl)
  · exact SemanticRightRootResult6004.claimSound
      selector selectorLower selectorUpper witness
  · exact SemanticRightRootResult5961.claimSound
      selector selectorLower selectorUpper witness
  · rfl

end SemanticRightRootResult6009

namespace SemanticRightRootResult6014

def leftRaw : List Term := SemanticRightRootResult6009.rawTerms
def rightRaw : List Term := SemanticRightRootResult6001.rawTerms
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6747⟩⟩, ⟨.program ⟨214⟩, ⟨7885⟩⟩, ⟨.program ⟨214⟩, ⟨7821⟩⟩] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def owner : Owner := ⟨.program ⟨214⟩, ⟨7912⟩⟩
def rawTerms : List Term := outputRaw
def summary : Bound := .exactZero
def resultEvent : Nat := 6014
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
  SemanticRightRootResult6009.actual selector witness *
    SemanticRightRootResult6001.actual selector witness
theorem claimOfHistory (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841)
    (expressionAt : document.expressions.lookup 7912 =
      some Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Expression030.ExpressionRow7912)
    (leftPredecessorAt : (history.lookup 6010).map AnnotatedEvent.event =
      some (.predecessor owner 0 ⟨7888⟩ 6009))
    (rightPredecessorAt : (history.lookup 6011).map AnnotatedEvent.event =
      some (.predecessor owner 1 ⟨7822⟩ 6001))
    (ruleAt : (history.lookup 6012).map AnnotatedEvent.event =
      some (.boundTransfer owner (.product (.predecessor 0 6010 .coefficient) (.predecessor 1 6011 .coefficient) ⟨false, false, none, none, none⟩)))
    (leftClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult6009.resultEvent
      SemanticRightRootResult6009.owner
      (SemanticRightRootResult6009.actual selector witness)
      SemanticRightRootResult6009.rawTerms
      SemanticRightRootResult6009.summary)
    (rightClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult6001.resultEvent
      SemanticRightRootResult6001.owner
      (SemanticRightRootResult6001.actual selector witness)
      SemanticRightRootResult6001.rawTerms
      SemanticRightRootResult6001.summary)
    (outputAt : (history.lookup resultEvent).map AnnotatedEvent.event =
      some (.resultExact owner rawTerms summary)) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  refine ⟨outputAt, ?_⟩
  exact exactValueClaim_product_of_mod_zero 100418593683253592432016548326729029359133068138294319235841 witness.env
    (SemanticRightRootResult6009.actual selector witness)
    (SemanticRightRootResult6001.actual selector witness) left right output
    (by simpa [left, leftRaw, SemanticRightRootResult6009.summary] using leftClaim.claim)
    (by simpa [right, rightRaw, SemanticRightRootResult6001.summary] using rightClaim.claim)
    (resultSound witness.env) (by decide)

theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply claimOfHistory selector witness (by rfl) (by rfl) (by rfl) (by rfl)
  · exact SemanticRightRootResult6009.claimSound
      selector selectorLower selectorUpper witness
  · exact SemanticRightRootResult6001.claimSound
      selector selectorLower selectorUpper witness
  · rfl

end SemanticRightRootResult6014

namespace SemanticRightRootResult6019

def leftRaw : List Term := SemanticRightRootResult6014.rawTerms
def rightRaw : List Term := SemanticRightRootResult5991.rawTerms
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6747⟩⟩, ⟨.program ⟨214⟩, ⟨7885⟩⟩, ⟨.program ⟨214⟩, ⟨7821⟩⟩, ⟨.program ⟨214⟩, ⟨6687⟩⟩] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def owner : Owner := ⟨.program ⟨214⟩, ⟨7918⟩⟩
def rawTerms : List Term := outputRaw
def summary : Bound := .exactZero
def resultEvent : Nat := 6019
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
  SemanticRightRootResult6014.actual selector witness *
    SemanticRightRootResult5991.actual selector witness
theorem claimOfHistory (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841)
    (expressionAt : document.expressions.lookup 7918 =
      some Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Expression030.ExpressionRow7918)
    (leftPredecessorAt : (history.lookup 6015).map AnnotatedEvent.event =
      some (.predecessor owner 0 ⟨7912⟩ 6014))
    (rightPredecessorAt : (history.lookup 6016).map AnnotatedEvent.event =
      some (.predecessor owner 1 ⟨6688⟩ 5991))
    (ruleAt : (history.lookup 6017).map AnnotatedEvent.event =
      some (.boundTransfer owner (.product (.predecessor 0 6015 .coefficient) (.predecessor 1 6016 .coefficient) ⟨false, false, none, none, none⟩)))
    (leftClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult6014.resultEvent
      SemanticRightRootResult6014.owner
      (SemanticRightRootResult6014.actual selector witness)
      SemanticRightRootResult6014.rawTerms
      SemanticRightRootResult6014.summary)
    (rightClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult5991.resultEvent
      SemanticRightRootResult5991.owner
      (SemanticRightRootResult5991.actual selector witness)
      SemanticRightRootResult5991.rawTerms
      SemanticRightRootResult5991.summary)
    (outputAt : (history.lookup resultEvent).map AnnotatedEvent.event =
      some (.resultExact owner rawTerms summary)) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  refine ⟨outputAt, ?_⟩
  exact exactValueClaim_product_of_mod_zero 100418593683253592432016548326729029359133068138294319235841 witness.env
    (SemanticRightRootResult6014.actual selector witness)
    (SemanticRightRootResult5991.actual selector witness) left right output
    (by simpa [left, leftRaw, SemanticRightRootResult6014.summary] using leftClaim.claim)
    (by simpa [right, rightRaw, SemanticRightRootResult5991.summary] using rightClaim.claim)
    (resultSound witness.env) (by decide)

theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply claimOfHistory selector witness (by rfl) (by rfl) (by rfl) (by rfl)
  · exact SemanticRightRootResult6014.claimSound
      selector selectorLower selectorUpper witness
  · exact SemanticRightRootResult5991.claimSound
      selector selectorLower selectorUpper witness
  · rfl

end SemanticRightRootResult6019

namespace SemanticRightRootResult6031

def owner : Owner := ⟨.program ⟨214⟩, ⟨6654⟩⟩
def rawTerms : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6653⟩⟩] } }]
def summary : Bound := .exactZero
def producerEvent : Nat := 6030
def resultEvent : Nat := 6031
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual resultEvent
theorem terminalAt (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, rightRootOwnerAtSelector214 selector selectorLower selectorUpper _, ?_⟩
  refine ⟨.scale (.predecessor 0 6028 .coefficient) (.value (.predecessor 1 6029 .coefficient)), 0, .scale (.predecessor 0 6028 .coefficient) (.value (.predecessor 1 6029 .coefficient)), ?_, ?_⟩
  · rfl
  · rfl
theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness
    (terminalAt selector selectorLower selectorUpper)

end SemanticRightRootResult6031

namespace SemanticRightRootResult6041

def owner : Owner := ⟨.program ⟨214⟩, ⟨7824⟩⟩
def rawTerms : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨7823⟩⟩] } }]
def summary : Bound := .exactZero
def producerEvent : Nat := 6040
def resultEvent : Nat := 6041
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual resultEvent
theorem terminalAt (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, rightRootOwnerAtSelector214 selector selectorLower selectorUpper _, ?_⟩
  refine ⟨.scale (.predecessor 0 6038 .coefficient) (.value (.predecessor 1 6039 .coefficient)), 0, .scale (.predecessor 0 6038 .coefficient) (.value (.predecessor 1 6039 .coefficient)), ?_, ?_⟩
  · rfl
  · rfl
theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness
    (terminalAt selector selectorLower selectorUpper)

end SemanticRightRootResult6041

namespace SemanticRightRootResult6044

def owner : Owner := ⟨.program ⟨214⟩, ⟨6749⟩⟩
def rawTerms : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6749⟩⟩] } }]
def summary : Bound := .exactZero
def producerEvent : Nat := 6043
def resultEvent : Nat := 6044
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

end SemanticRightRootResult6044

namespace SemanticRightRootResult6049

def leftRaw : List Term := SemanticRightRootResult6044.rawTerms
def rightRaw : List Term := SemanticRightRootResult5961.rawTerms
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6749⟩⟩, ⟨.program ⟨214⟩, ⟨7885⟩⟩] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def owner : Owner := ⟨.program ⟨214⟩, ⟨7889⟩⟩
def rawTerms : List Term := outputRaw
def summary : Bound := .exactZero
def resultEvent : Nat := 6049
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
  SemanticRightRootResult6044.actual selector witness *
    SemanticRightRootResult5961.actual selector witness
theorem claimOfHistory (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841)
    (expressionAt : document.expressions.lookup 7889 =
      some Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Expression030.ExpressionRow7889)
    (leftPredecessorAt : (history.lookup 6045).map AnnotatedEvent.event =
      some (.predecessor owner 0 ⟨6749⟩ 6044))
    (rightPredecessorAt : (history.lookup 6046).map AnnotatedEvent.event =
      some (.predecessor owner 1 ⟨7886⟩ 5961))
    (ruleAt : (history.lookup 6047).map AnnotatedEvent.event =
      some (.boundTransfer owner (.product (.predecessor 0 6045 .coefficient) (.predecessor 1 6046 .coefficient) ⟨false, false, none, none, none⟩)))
    (leftClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult6044.resultEvent
      SemanticRightRootResult6044.owner
      (SemanticRightRootResult6044.actual selector witness)
      SemanticRightRootResult6044.rawTerms
      SemanticRightRootResult6044.summary)
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
    (SemanticRightRootResult6044.actual selector witness)
    (SemanticRightRootResult5961.actual selector witness) left right output
    (by simpa [left, leftRaw, SemanticRightRootResult6044.summary] using leftClaim.claim)
    (by simpa [right, rightRaw, SemanticRightRootResult5961.summary] using rightClaim.claim)
    (resultSound witness.env) (by decide)

theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply claimOfHistory selector witness (by rfl) (by rfl) (by rfl) (by rfl)
  · exact SemanticRightRootResult6044.claimSound
      selector selectorLower selectorUpper witness
  · exact SemanticRightRootResult5961.claimSound
      selector selectorLower selectorUpper witness
  · rfl

end SemanticRightRootResult6049

namespace SemanticRightRootResult6054

def leftRaw : List Term := SemanticRightRootResult6049.rawTerms
def rightRaw : List Term := SemanticRightRootResult6041.rawTerms
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6749⟩⟩, ⟨.program ⟨214⟩, ⟨7885⟩⟩, ⟨.program ⟨214⟩, ⟨7823⟩⟩] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def owner : Owner := ⟨.program ⟨214⟩, ⟨7913⟩⟩
def rawTerms : List Term := outputRaw
def summary : Bound := .exactZero
def resultEvent : Nat := 6054
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
  SemanticRightRootResult6049.actual selector witness *
    SemanticRightRootResult6041.actual selector witness
theorem claimOfHistory (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841)
    (expressionAt : document.expressions.lookup 7913 =
      some Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Expression030.ExpressionRow7913)
    (leftPredecessorAt : (history.lookup 6050).map AnnotatedEvent.event =
      some (.predecessor owner 0 ⟨7889⟩ 6049))
    (rightPredecessorAt : (history.lookup 6051).map AnnotatedEvent.event =
      some (.predecessor owner 1 ⟨7824⟩ 6041))
    (ruleAt : (history.lookup 6052).map AnnotatedEvent.event =
      some (.boundTransfer owner (.product (.predecessor 0 6050 .coefficient) (.predecessor 1 6051 .coefficient) ⟨false, false, none, none, none⟩)))
    (leftClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult6049.resultEvent
      SemanticRightRootResult6049.owner
      (SemanticRightRootResult6049.actual selector witness)
      SemanticRightRootResult6049.rawTerms
      SemanticRightRootResult6049.summary)
    (rightClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult6041.resultEvent
      SemanticRightRootResult6041.owner
      (SemanticRightRootResult6041.actual selector witness)
      SemanticRightRootResult6041.rawTerms
      SemanticRightRootResult6041.summary)
    (outputAt : (history.lookup resultEvent).map AnnotatedEvent.event =
      some (.resultExact owner rawTerms summary)) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  refine ⟨outputAt, ?_⟩
  exact exactValueClaim_product_of_mod_zero 100418593683253592432016548326729029359133068138294319235841 witness.env
    (SemanticRightRootResult6049.actual selector witness)
    (SemanticRightRootResult6041.actual selector witness) left right output
    (by simpa [left, leftRaw, SemanticRightRootResult6049.summary] using leftClaim.claim)
    (by simpa [right, rightRaw, SemanticRightRootResult6041.summary] using rightClaim.claim)
    (resultSound witness.env) (by decide)

theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply claimOfHistory selector witness (by rfl) (by rfl) (by rfl) (by rfl)
  · exact SemanticRightRootResult6049.claimSound
      selector selectorLower selectorUpper witness
  · exact SemanticRightRootResult6041.claimSound
      selector selectorLower selectorUpper witness
  · rfl

end SemanticRightRootResult6054

namespace SemanticRightRootResult6059

def leftRaw : List Term := SemanticRightRootResult6054.rawTerms
def rightRaw : List Term := SemanticRightRootResult6031.rawTerms
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6749⟩⟩, ⟨.program ⟨214⟩, ⟨7885⟩⟩, ⟨.program ⟨214⟩, ⟨7823⟩⟩, ⟨.program ⟨214⟩, ⟨6653⟩⟩] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def owner : Owner := ⟨.program ⟨214⟩, ⟨7919⟩⟩
def rawTerms : List Term := outputRaw
def summary : Bound := .exactZero
def resultEvent : Nat := 6059
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
  SemanticRightRootResult6054.actual selector witness *
    SemanticRightRootResult6031.actual selector witness
theorem claimOfHistory (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841)
    (expressionAt : document.expressions.lookup 7919 =
      some Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Expression030.ExpressionRow7919)
    (leftPredecessorAt : (history.lookup 6055).map AnnotatedEvent.event =
      some (.predecessor owner 0 ⟨7913⟩ 6054))
    (rightPredecessorAt : (history.lookup 6056).map AnnotatedEvent.event =
      some (.predecessor owner 1 ⟨6654⟩ 6031))
    (ruleAt : (history.lookup 6057).map AnnotatedEvent.event =
      some (.boundTransfer owner (.product (.predecessor 0 6055 .coefficient) (.predecessor 1 6056 .coefficient) ⟨false, false, none, none, none⟩)))
    (leftClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult6054.resultEvent
      SemanticRightRootResult6054.owner
      (SemanticRightRootResult6054.actual selector witness)
      SemanticRightRootResult6054.rawTerms
      SemanticRightRootResult6054.summary)
    (rightClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult6031.resultEvent
      SemanticRightRootResult6031.owner
      (SemanticRightRootResult6031.actual selector witness)
      SemanticRightRootResult6031.rawTerms
      SemanticRightRootResult6031.summary)
    (outputAt : (history.lookup resultEvent).map AnnotatedEvent.event =
      some (.resultExact owner rawTerms summary)) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  refine ⟨outputAt, ?_⟩
  exact exactValueClaim_product_of_mod_zero 100418593683253592432016548326729029359133068138294319235841 witness.env
    (SemanticRightRootResult6054.actual selector witness)
    (SemanticRightRootResult6031.actual selector witness) left right output
    (by simpa [left, leftRaw, SemanticRightRootResult6054.summary] using leftClaim.claim)
    (by simpa [right, rightRaw, SemanticRightRootResult6031.summary] using rightClaim.claim)
    (resultSound witness.env) (by decide)

theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply claimOfHistory selector witness (by rfl) (by rfl) (by rfl) (by rfl)
  · exact SemanticRightRootResult6054.claimSound
      selector selectorLower selectorUpper witness
  · exact SemanticRightRootResult6031.claimSound
      selector selectorLower selectorUpper witness
  · rfl

end SemanticRightRootResult6059

namespace SemanticRightRootResult6071

def owner : Owner := ⟨.program ⟨214⟩, ⟨6676⟩⟩
def rawTerms : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6675⟩⟩] } }]
def summary : Bound := .exactZero
def producerEvent : Nat := 6070
def resultEvent : Nat := 6071
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual resultEvent
theorem terminalAt (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, rightRootOwnerAtSelector214 selector selectorLower selectorUpper _, ?_⟩
  refine ⟨.scale (.predecessor 0 6068 .coefficient) (.value (.predecessor 1 6069 .coefficient)), 0, .scale (.predecessor 0 6068 .coefficient) (.value (.predecessor 1 6069 .coefficient)), ?_, ?_⟩
  · rfl
  · rfl
theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness
    (terminalAt selector selectorLower selectorUpper)

end SemanticRightRootResult6071

namespace SemanticRightRootResult6081

def owner : Owner := ⟨.program ⟨214⟩, ⟨7826⟩⟩
def rawTerms : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨7825⟩⟩] } }]
def summary : Bound := .exactZero
def producerEvent : Nat := 6080
def resultEvent : Nat := 6081
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual resultEvent
theorem terminalAt (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, rightRootOwnerAtSelector214 selector selectorLower selectorUpper _, ?_⟩
  refine ⟨.scale (.predecessor 0 6078 .coefficient) (.value (.predecessor 1 6079 .coefficient)), 0, .scale (.predecessor 0 6078 .coefficient) (.value (.predecessor 1 6079 .coefficient)), ?_, ?_⟩
  · rfl
  · rfl
theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness
    (terminalAt selector selectorLower selectorUpper)

end SemanticRightRootResult6081

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
