import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticRightRootShard017

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open TallSecurity0ABI
open TallSemantics

namespace SemanticRightRootResult3680

def leftRaw : List Term := SemanticRightRootResult3675.rawTerms
def rightRaw : List Term := SemanticRightRootResult663.rawTerms
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6398⟩⟩, ⟨.program ⟨214⟩, ⟨17434⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def owner : Owner := ⟨.program ⟨214⟩, ⟨17435⟩⟩
def rawTerms : List Term := outputRaw
def summary : Bound := .exactZero
def resultEvent : Nat := 3680
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
  SemanticRightRootResult3675.actual selector witness *
    SemanticRightRootResult663.actual selector witness
theorem claimOfHistory (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841)
    (expressionAt : document.expressions.lookup 17435 =
      some Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Expression068.ExpressionRow17435)
    (leftPredecessorAt : (history.lookup 3676).map AnnotatedEvent.event =
      some (.predecessor owner 0 ⟨17434⟩ 3675))
    (rightPredecessorAt : (history.lookup 3677).map AnnotatedEvent.event =
      some (.predecessor owner 1 ⟨6398⟩ 663))
    (ruleAt : (history.lookup 3678).map AnnotatedEvent.event =
      some (.boundTransfer owner (.product (.predecessor 0 3676 .coefficient) (.predecessor 1 3677 .coefficient) ⟨true, true, none, some 1, some 1⟩)))
    (leftClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult3675.resultEvent
      SemanticRightRootResult3675.owner
      (SemanticRightRootResult3675.actual selector witness)
      SemanticRightRootResult3675.rawTerms
      SemanticRightRootResult3675.summary)
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
    (SemanticRightRootResult3675.actual selector witness)
    (SemanticRightRootResult663.actual selector witness) left right output
    (by simpa [left, leftRaw, SemanticRightRootResult3675.summary] using leftClaim.claim)
    (by simpa [right, rightRaw, SemanticRightRootResult663.summary] using rightClaim.claim)
    (resultSound witness.env) (by decide)

theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply claimOfHistory selector witness (by rfl) (by rfl) (by rfl) (by rfl)
  · exact SemanticRightRootResult3675.claimSound
      selector selectorLower selectorUpper witness
  · exact SemanticRightRootResult663.claimSound
      selector selectorLower selectorUpper witness
  · rfl

end SemanticRightRootResult3680

namespace SemanticRightRootResult3683

def owner : Owner := ⟨.program ⟨214⟩, ⟨17806⟩⟩
def rawTerms : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17806⟩⟩], orderedFactors := [] } }]
def summary : Bound := .exactZero
def producerEvent : Nat := 3682
def resultEvent : Nat := 3683
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

end SemanticRightRootResult3683

namespace SemanticRightRootResult3688

def leftRaw : List Term := SemanticRightRootResult3683.rawTerms
def rightRaw : List Term := SemanticRightRootResult673.rawTerms
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6407⟩⟩, ⟨.program ⟨214⟩, ⟨17806⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def owner : Owner := ⟨.program ⟨214⟩, ⟨17807⟩⟩
def rawTerms : List Term := outputRaw
def summary : Bound := .exactZero
def resultEvent : Nat := 3688
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
  SemanticRightRootResult3683.actual selector witness *
    SemanticRightRootResult673.actual selector witness
theorem claimOfHistory (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841)
    (expressionAt : document.expressions.lookup 17807 =
      some Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Expression069.ExpressionRow17807)
    (leftPredecessorAt : (history.lookup 3684).map AnnotatedEvent.event =
      some (.predecessor owner 0 ⟨17806⟩ 3683))
    (rightPredecessorAt : (history.lookup 3685).map AnnotatedEvent.event =
      some (.predecessor owner 1 ⟨6407⟩ 673))
    (ruleAt : (history.lookup 3686).map AnnotatedEvent.event =
      some (.boundTransfer owner (.product (.predecessor 0 3684 .coefficient) (.predecessor 1 3685 .coefficient) ⟨true, true, none, some 1, some 1⟩)))
    (leftClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult3683.resultEvent
      SemanticRightRootResult3683.owner
      (SemanticRightRootResult3683.actual selector witness)
      SemanticRightRootResult3683.rawTerms
      SemanticRightRootResult3683.summary)
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
    (SemanticRightRootResult3683.actual selector witness)
    (SemanticRightRootResult673.actual selector witness) left right output
    (by simpa [left, leftRaw, SemanticRightRootResult3683.summary] using leftClaim.claim)
    (by simpa [right, rightRaw, SemanticRightRootResult673.summary] using rightClaim.claim)
    (resultSound witness.env) (by decide)

theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply claimOfHistory selector witness (by rfl) (by rfl) (by rfl) (by rfl)
  · exact SemanticRightRootResult3683.claimSound
      selector selectorLower selectorUpper witness
  · exact SemanticRightRootResult673.claimSound
      selector selectorLower selectorUpper witness
  · rfl

end SemanticRightRootResult3688

namespace SemanticRightRootResult3691

def owner : Owner := ⟨.program ⟨214⟩, ⟨15511⟩⟩
def rawTerms : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15511⟩⟩], orderedFactors := [] } }]
def summary : Bound := .exactZero
def producerEvent : Nat := 3690
def resultEvent : Nat := 3691
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

end SemanticRightRootResult3691

namespace SemanticRightRootResult3696

def leftRaw : List Term := SemanticRightRootResult3691.rawTerms
def rightRaw : List Term := SemanticRightRootResult683.rawTerms
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6427⟩⟩, ⟨.program ⟨214⟩, ⟨15511⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def owner : Owner := ⟨.program ⟨214⟩, ⟨15512⟩⟩
def rawTerms : List Term := outputRaw
def summary : Bound := .exactZero
def resultEvent : Nat := 3696
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
  SemanticRightRootResult3691.actual selector witness *
    SemanticRightRootResult683.actual selector witness
theorem claimOfHistory (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841)
    (expressionAt : document.expressions.lookup 15512 =
      some Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Expression060.ExpressionRow15512)
    (leftPredecessorAt : (history.lookup 3692).map AnnotatedEvent.event =
      some (.predecessor owner 0 ⟨15511⟩ 3691))
    (rightPredecessorAt : (history.lookup 3693).map AnnotatedEvent.event =
      some (.predecessor owner 1 ⟨6427⟩ 683))
    (ruleAt : (history.lookup 3694).map AnnotatedEvent.event =
      some (.boundTransfer owner (.product (.predecessor 0 3692 .coefficient) (.predecessor 1 3693 .coefficient) ⟨true, true, none, some 1, some 1⟩)))
    (leftClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult3691.resultEvent
      SemanticRightRootResult3691.owner
      (SemanticRightRootResult3691.actual selector witness)
      SemanticRightRootResult3691.rawTerms
      SemanticRightRootResult3691.summary)
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
    (SemanticRightRootResult3691.actual selector witness)
    (SemanticRightRootResult683.actual selector witness) left right output
    (by simpa [left, leftRaw, SemanticRightRootResult3691.summary] using leftClaim.claim)
    (by simpa [right, rightRaw, SemanticRightRootResult683.summary] using rightClaim.claim)
    (resultSound witness.env) (by decide)

theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply claimOfHistory selector witness (by rfl) (by rfl) (by rfl) (by rfl)
  · exact SemanticRightRootResult3691.claimSound
      selector selectorLower selectorUpper witness
  · exact SemanticRightRootResult683.claimSound
      selector selectorLower selectorUpper witness
  · rfl

end SemanticRightRootResult3696

namespace SemanticRightRootResult3699

def owner : Owner := ⟨.program ⟨214⟩, ⟨15203⟩⟩
def rawTerms : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15203⟩⟩], orderedFactors := [] } }]
def summary : Bound := .exactZero
def producerEvent : Nat := 3698
def resultEvent : Nat := 3699
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

end SemanticRightRootResult3699

namespace SemanticRightRootResult3704

def leftRaw : List Term := SemanticRightRootResult3699.rawTerms
def rightRaw : List Term := SemanticRightRootResult693.rawTerms
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6452⟩⟩, ⟨.program ⟨214⟩, ⟨15203⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def owner : Owner := ⟨.program ⟨214⟩, ⟨15204⟩⟩
def rawTerms : List Term := outputRaw
def summary : Bound := .exactZero
def resultEvent : Nat := 3704
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
  SemanticRightRootResult3699.actual selector witness *
    SemanticRightRootResult693.actual selector witness
theorem claimOfHistory (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841)
    (expressionAt : document.expressions.lookup 15204 =
      some Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Expression059.ExpressionRow15204)
    (leftPredecessorAt : (history.lookup 3700).map AnnotatedEvent.event =
      some (.predecessor owner 0 ⟨15203⟩ 3699))
    (rightPredecessorAt : (history.lookup 3701).map AnnotatedEvent.event =
      some (.predecessor owner 1 ⟨6452⟩ 693))
    (ruleAt : (history.lookup 3702).map AnnotatedEvent.event =
      some (.boundTransfer owner (.product (.predecessor 0 3700 .coefficient) (.predecessor 1 3701 .coefficient) ⟨true, true, none, some 1, some 1⟩)))
    (leftClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult3699.resultEvent
      SemanticRightRootResult3699.owner
      (SemanticRightRootResult3699.actual selector witness)
      SemanticRightRootResult3699.rawTerms
      SemanticRightRootResult3699.summary)
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
    (SemanticRightRootResult3699.actual selector witness)
    (SemanticRightRootResult693.actual selector witness) left right output
    (by simpa [left, leftRaw, SemanticRightRootResult3699.summary] using leftClaim.claim)
    (by simpa [right, rightRaw, SemanticRightRootResult693.summary] using rightClaim.claim)
    (resultSound witness.env) (by decide)

theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply claimOfHistory selector witness (by rfl) (by rfl) (by rfl) (by rfl)
  · exact SemanticRightRootResult3699.claimSound
      selector selectorLower selectorUpper witness
  · exact SemanticRightRootResult693.claimSound
      selector selectorLower selectorUpper witness
  · rfl

end SemanticRightRootResult3704

namespace SemanticRightRootResult3707

def owner : Owner := ⟨.program ⟨214⟩, ⟨15042⟩⟩
def rawTerms : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15042⟩⟩], orderedFactors := [] } }]
def summary : Bound := .exactZero
def producerEvent : Nat := 3706
def resultEvent : Nat := 3707
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

end SemanticRightRootResult3707

namespace SemanticRightRootResult3712

def leftRaw : List Term := SemanticRightRootResult3707.rawTerms
def rightRaw : List Term := SemanticRightRootResult703.rawTerms
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨15042⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def owner : Owner := ⟨.program ⟨214⟩, ⟨15043⟩⟩
def rawTerms : List Term := outputRaw
def summary : Bound := .exactZero
def resultEvent : Nat := 3712
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
  SemanticRightRootResult3707.actual selector witness *
    SemanticRightRootResult703.actual selector witness
theorem claimOfHistory (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841)
    (expressionAt : document.expressions.lookup 15043 =
      some Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Expression058.ExpressionRow15043)
    (leftPredecessorAt : (history.lookup 3708).map AnnotatedEvent.event =
      some (.predecessor owner 0 ⟨15042⟩ 3707))
    (rightPredecessorAt : (history.lookup 3709).map AnnotatedEvent.event =
      some (.predecessor owner 1 ⟨6475⟩ 703))
    (ruleAt : (history.lookup 3710).map AnnotatedEvent.event =
      some (.boundTransfer owner (.product (.predecessor 0 3708 .coefficient) (.predecessor 1 3709 .coefficient) ⟨true, true, none, some 1, some 1⟩)))
    (leftClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult3707.resultEvent
      SemanticRightRootResult3707.owner
      (SemanticRightRootResult3707.actual selector witness)
      SemanticRightRootResult3707.rawTerms
      SemanticRightRootResult3707.summary)
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
    (SemanticRightRootResult3707.actual selector witness)
    (SemanticRightRootResult703.actual selector witness) left right output
    (by simpa [left, leftRaw, SemanticRightRootResult3707.summary] using leftClaim.claim)
    (by simpa [right, rightRaw, SemanticRightRootResult703.summary] using rightClaim.claim)
    (resultSound witness.env) (by decide)

theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply claimOfHistory selector witness (by rfl) (by rfl) (by rfl) (by rfl)
  · exact SemanticRightRootResult3707.claimSound
      selector selectorLower selectorUpper witness
  · exact SemanticRightRootResult703.claimSound
      selector selectorLower selectorUpper witness
  · rfl

end SemanticRightRootResult3712

namespace SemanticRightRootResult3715

def owner : Owner := ⟨.program ⟨214⟩, ⟨14881⟩⟩
def rawTerms : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨14881⟩⟩], orderedFactors := [] } }]
def summary : Bound := .exactZero
def producerEvent : Nat := 3714
def resultEvent : Nat := 3715
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

end SemanticRightRootResult3715

namespace SemanticRightRootResult3720

def leftRaw : List Term := SemanticRightRootResult3715.rawTerms
def rightRaw : List Term := SemanticRightRootResult713.rawTerms
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨14881⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def owner : Owner := ⟨.program ⟨214⟩, ⟨14882⟩⟩
def rawTerms : List Term := outputRaw
def summary : Bound := .exactZero
def resultEvent : Nat := 3720
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
  SemanticRightRootResult3715.actual selector witness *
    SemanticRightRootResult713.actual selector witness
theorem claimOfHistory (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841)
    (expressionAt : document.expressions.lookup 14882 =
      some Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Expression058.ExpressionRow14882)
    (leftPredecessorAt : (history.lookup 3716).map AnnotatedEvent.event =
      some (.predecessor owner 0 ⟨14881⟩ 3715))
    (rightPredecessorAt : (history.lookup 3717).map AnnotatedEvent.event =
      some (.predecessor owner 1 ⟨6495⟩ 713))
    (ruleAt : (history.lookup 3718).map AnnotatedEvent.event =
      some (.boundTransfer owner (.product (.predecessor 0 3716 .coefficient) (.predecessor 1 3717 .coefficient) ⟨true, true, none, some 1, some 1⟩)))
    (leftClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult3715.resultEvent
      SemanticRightRootResult3715.owner
      (SemanticRightRootResult3715.actual selector witness)
      SemanticRightRootResult3715.rawTerms
      SemanticRightRootResult3715.summary)
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
    (SemanticRightRootResult3715.actual selector witness)
    (SemanticRightRootResult713.actual selector witness) left right output
    (by simpa [left, leftRaw, SemanticRightRootResult3715.summary] using leftClaim.claim)
    (by simpa [right, rightRaw, SemanticRightRootResult713.summary] using rightClaim.claim)
    (resultSound witness.env) (by decide)

theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply claimOfHistory selector witness (by rfl) (by rfl) (by rfl) (by rfl)
  · exact SemanticRightRootResult3715.claimSound
      selector selectorLower selectorUpper witness
  · exact SemanticRightRootResult713.claimSound
      selector selectorLower selectorUpper witness
  · rfl

end SemanticRightRootResult3720

namespace SemanticRightRootResult3724

def leftRaw : List Term := SemanticRightRootResult728.rawTerms
def rightRaw : List Term := SemanticRightRootResult3720.rawTerms
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨14881⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def owner : Owner := ⟨.program ⟨214⟩, ⟨14883⟩⟩
def rawTerms : List Term := outputRaw
def summary : Bound := .exactZero
def resultEvent : Nat := 3724
theorem resultAgreement : CanonicalAgreement output (add left right) := by
  decide +kernel

theorem resultSound (env : Env Owner) :
    evalPolynomial env output = evalPolynomial env left + evalPolynomial env right := by
  exact addCanonicalResultSound env left right output resultAgreement

def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  SemanticRightRootResult728.actual selector witness +
    SemanticRightRootResult3720.actual selector witness
theorem claimOfHistory (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841)
    (expressionAt : document.expressions.lookup 14883 =
      some Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Expression058.ExpressionRow14883)
    (leftPredecessorAt : (history.lookup 3721).map AnnotatedEvent.event =
      some (.predecessor owner 0 ⟨6379⟩ 728))
    (rightPredecessorAt : (history.lookup 3722).map AnnotatedEvent.event =
      some (.predecessor owner 1 ⟨14882⟩ 3720))
    (ruleAt : (history.lookup 3723).map AnnotatedEvent.event =
      some (.boundTransfer owner (.sum [.predecessor 0 3721 .coefficient, .predecessor 1 3722 .coefficient])))
    (leftClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult728.resultEvent
      SemanticRightRootResult728.owner
      (SemanticRightRootResult728.actual selector witness)
      SemanticRightRootResult728.rawTerms
      SemanticRightRootResult728.summary)
    (rightClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult3720.resultEvent
      SemanticRightRootResult3720.owner
      (SemanticRightRootResult3720.actual selector witness)
      SemanticRightRootResult3720.rawTerms
      SemanticRightRootResult3720.summary)
    (outputAt : (history.lookup resultEvent).map AnnotatedEvent.event =
      some (.resultExact owner rawTerms summary)) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  refine ⟨outputAt, ?_⟩
  exact exactValueClaim_add_of_mod_zero 100418593683253592432016548326729029359133068138294319235841 witness.env
    (SemanticRightRootResult728.actual selector witness)
    (SemanticRightRootResult3720.actual selector witness) left right output
    (by simpa [left, leftRaw, SemanticRightRootResult728.summary] using leftClaim.claim)
    (by simpa [right, rightRaw, SemanticRightRootResult3720.summary] using rightClaim.claim)
    (resultSound witness.env) (by decide)

theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply claimOfHistory selector witness (by rfl) (by rfl) (by rfl) (by rfl)
  · exact SemanticRightRootResult728.claimSound
      selector selectorLower selectorUpper witness
  · exact SemanticRightRootResult3720.claimSound
      selector selectorLower selectorUpper witness
  · rfl

end SemanticRightRootResult3724

namespace SemanticRightRootResult3728

def leftRaw : List Term := SemanticRightRootResult3724.rawTerms
def rightRaw : List Term := SemanticRightRootResult3712.rawTerms
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨15042⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨14881⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def owner : Owner := ⟨.program ⟨214⟩, ⟨15044⟩⟩
def rawTerms : List Term := outputRaw
def summary : Bound := .exactZero
def resultEvent : Nat := 3728
theorem resultAgreement : CanonicalAgreement output (add left right) := by
  decide +kernel

theorem resultSound (env : Env Owner) :
    evalPolynomial env output = evalPolynomial env left + evalPolynomial env right := by
  exact addCanonicalResultSound env left right output resultAgreement

def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  SemanticRightRootResult3724.actual selector witness +
    SemanticRightRootResult3712.actual selector witness
theorem claimOfHistory (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841)
    (expressionAt : document.expressions.lookup 15044 =
      some Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Expression058.ExpressionRow15044)
    (leftPredecessorAt : (history.lookup 3725).map AnnotatedEvent.event =
      some (.predecessor owner 0 ⟨14883⟩ 3724))
    (rightPredecessorAt : (history.lookup 3726).map AnnotatedEvent.event =
      some (.predecessor owner 1 ⟨15043⟩ 3712))
    (ruleAt : (history.lookup 3727).map AnnotatedEvent.event =
      some (.boundTransfer owner (.sum [.predecessor 0 3725 .coefficient, .predecessor 1 3726 .coefficient])))
    (leftClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult3724.resultEvent
      SemanticRightRootResult3724.owner
      (SemanticRightRootResult3724.actual selector witness)
      SemanticRightRootResult3724.rawTerms
      SemanticRightRootResult3724.summary)
    (rightClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult3712.resultEvent
      SemanticRightRootResult3712.owner
      (SemanticRightRootResult3712.actual selector witness)
      SemanticRightRootResult3712.rawTerms
      SemanticRightRootResult3712.summary)
    (outputAt : (history.lookup resultEvent).map AnnotatedEvent.event =
      some (.resultExact owner rawTerms summary)) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  refine ⟨outputAt, ?_⟩
  exact exactValueClaim_add_of_mod_zero 100418593683253592432016548326729029359133068138294319235841 witness.env
    (SemanticRightRootResult3724.actual selector witness)
    (SemanticRightRootResult3712.actual selector witness) left right output
    (by simpa [left, leftRaw, SemanticRightRootResult3724.summary] using leftClaim.claim)
    (by simpa [right, rightRaw, SemanticRightRootResult3712.summary] using rightClaim.claim)
    (resultSound witness.env) (by decide)

theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply claimOfHistory selector witness (by rfl) (by rfl) (by rfl) (by rfl)
  · exact SemanticRightRootResult3724.claimSound
      selector selectorLower selectorUpper witness
  · exact SemanticRightRootResult3712.claimSound
      selector selectorLower selectorUpper witness
  · rfl

end SemanticRightRootResult3728

namespace SemanticRightRootResult3732

def leftRaw : List Term := SemanticRightRootResult3728.rawTerms
def rightRaw : List Term := SemanticRightRootResult3704.rawTerms
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6452⟩⟩, ⟨.program ⟨214⟩, ⟨15203⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨15042⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨14881⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def owner : Owner := ⟨.program ⟨214⟩, ⟨15205⟩⟩
def rawTerms : List Term := outputRaw
def summary : Bound := .exactZero
def resultEvent : Nat := 3732
theorem resultAgreement : CanonicalAgreement output (add left right) := by
  decide +kernel

theorem resultSound (env : Env Owner) :
    evalPolynomial env output = evalPolynomial env left + evalPolynomial env right := by
  exact addCanonicalResultSound env left right output resultAgreement

def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  SemanticRightRootResult3728.actual selector witness +
    SemanticRightRootResult3704.actual selector witness
theorem claimOfHistory (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841)
    (expressionAt : document.expressions.lookup 15205 =
      some Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Expression059.ExpressionRow15205)
    (leftPredecessorAt : (history.lookup 3729).map AnnotatedEvent.event =
      some (.predecessor owner 0 ⟨15044⟩ 3728))
    (rightPredecessorAt : (history.lookup 3730).map AnnotatedEvent.event =
      some (.predecessor owner 1 ⟨15204⟩ 3704))
    (ruleAt : (history.lookup 3731).map AnnotatedEvent.event =
      some (.boundTransfer owner (.sum [.predecessor 0 3729 .coefficient, .predecessor 1 3730 .coefficient])))
    (leftClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult3728.resultEvent
      SemanticRightRootResult3728.owner
      (SemanticRightRootResult3728.actual selector witness)
      SemanticRightRootResult3728.rawTerms
      SemanticRightRootResult3728.summary)
    (rightClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult3704.resultEvent
      SemanticRightRootResult3704.owner
      (SemanticRightRootResult3704.actual selector witness)
      SemanticRightRootResult3704.rawTerms
      SemanticRightRootResult3704.summary)
    (outputAt : (history.lookup resultEvent).map AnnotatedEvent.event =
      some (.resultExact owner rawTerms summary)) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  refine ⟨outputAt, ?_⟩
  exact exactValueClaim_add_of_mod_zero 100418593683253592432016548326729029359133068138294319235841 witness.env
    (SemanticRightRootResult3728.actual selector witness)
    (SemanticRightRootResult3704.actual selector witness) left right output
    (by simpa [left, leftRaw, SemanticRightRootResult3728.summary] using leftClaim.claim)
    (by simpa [right, rightRaw, SemanticRightRootResult3704.summary] using rightClaim.claim)
    (resultSound witness.env) (by decide)

theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply claimOfHistory selector witness (by rfl) (by rfl) (by rfl) (by rfl)
  · exact SemanticRightRootResult3728.claimSound
      selector selectorLower selectorUpper witness
  · exact SemanticRightRootResult3704.claimSound
      selector selectorLower selectorUpper witness
  · rfl

end SemanticRightRootResult3732

namespace SemanticRightRootResult3736

def leftRaw : List Term := SemanticRightRootResult3732.rawTerms
def rightRaw : List Term := SemanticRightRootResult3696.rawTerms
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6427⟩⟩, ⟨.program ⟨214⟩, ⟨15511⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6452⟩⟩, ⟨.program ⟨214⟩, ⟨15203⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨15042⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨14881⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def owner : Owner := ⟨.program ⟨214⟩, ⟨15513⟩⟩
def rawTerms : List Term := outputRaw
def summary : Bound := .exactZero
def resultEvent : Nat := 3736
theorem resultAgreement : CanonicalAgreement output (add left right) := by
  decide +kernel

theorem resultSound (env : Env Owner) :
    evalPolynomial env output = evalPolynomial env left + evalPolynomial env right := by
  exact addCanonicalResultSound env left right output resultAgreement

def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  SemanticRightRootResult3732.actual selector witness +
    SemanticRightRootResult3696.actual selector witness
theorem claimOfHistory (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841)
    (expressionAt : document.expressions.lookup 15513 =
      some Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Expression060.ExpressionRow15513)
    (leftPredecessorAt : (history.lookup 3733).map AnnotatedEvent.event =
      some (.predecessor owner 0 ⟨15205⟩ 3732))
    (rightPredecessorAt : (history.lookup 3734).map AnnotatedEvent.event =
      some (.predecessor owner 1 ⟨15512⟩ 3696))
    (ruleAt : (history.lookup 3735).map AnnotatedEvent.event =
      some (.boundTransfer owner (.sum [.predecessor 0 3733 .coefficient, .predecessor 1 3734 .coefficient])))
    (leftClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult3732.resultEvent
      SemanticRightRootResult3732.owner
      (SemanticRightRootResult3732.actual selector witness)
      SemanticRightRootResult3732.rawTerms
      SemanticRightRootResult3732.summary)
    (rightClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult3696.resultEvent
      SemanticRightRootResult3696.owner
      (SemanticRightRootResult3696.actual selector witness)
      SemanticRightRootResult3696.rawTerms
      SemanticRightRootResult3696.summary)
    (outputAt : (history.lookup resultEvent).map AnnotatedEvent.event =
      some (.resultExact owner rawTerms summary)) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  refine ⟨outputAt, ?_⟩
  exact exactValueClaim_add_of_mod_zero 100418593683253592432016548326729029359133068138294319235841 witness.env
    (SemanticRightRootResult3732.actual selector witness)
    (SemanticRightRootResult3696.actual selector witness) left right output
    (by simpa [left, leftRaw, SemanticRightRootResult3732.summary] using leftClaim.claim)
    (by simpa [right, rightRaw, SemanticRightRootResult3696.summary] using rightClaim.claim)
    (resultSound witness.env) (by decide)

theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply claimOfHistory selector witness (by rfl) (by rfl) (by rfl) (by rfl)
  · exact SemanticRightRootResult3732.claimSound
      selector selectorLower selectorUpper witness
  · exact SemanticRightRootResult3696.claimSound
      selector selectorLower selectorUpper witness
  · rfl

end SemanticRightRootResult3736

namespace SemanticRightRootResult3740

def leftRaw : List Term := SemanticRightRootResult3736.rawTerms
def rightRaw : List Term := SemanticRightRootResult3688.rawTerms
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6407⟩⟩, ⟨.program ⟨214⟩, ⟨17806⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6427⟩⟩, ⟨.program ⟨214⟩, ⟨15511⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6452⟩⟩, ⟨.program ⟨214⟩, ⟨15203⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨15042⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨14881⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def owner : Owner := ⟨.program ⟨214⟩, ⟨17808⟩⟩
def rawTerms : List Term := outputRaw
def summary : Bound := .exactZero
def resultEvent : Nat := 3740
theorem resultAgreement : CanonicalAgreement output (add left right) := by
  decide +kernel

theorem resultSound (env : Env Owner) :
    evalPolynomial env output = evalPolynomial env left + evalPolynomial env right := by
  exact addCanonicalResultSound env left right output resultAgreement

def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  SemanticRightRootResult3736.actual selector witness +
    SemanticRightRootResult3688.actual selector witness
theorem claimOfHistory (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841)
    (expressionAt : document.expressions.lookup 17808 =
      some Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Expression069.ExpressionRow17808)
    (leftPredecessorAt : (history.lookup 3737).map AnnotatedEvent.event =
      some (.predecessor owner 0 ⟨15513⟩ 3736))
    (rightPredecessorAt : (history.lookup 3738).map AnnotatedEvent.event =
      some (.predecessor owner 1 ⟨17807⟩ 3688))
    (ruleAt : (history.lookup 3739).map AnnotatedEvent.event =
      some (.boundTransfer owner (.sum [.predecessor 0 3737 .coefficient, .predecessor 1 3738 .coefficient])))
    (leftClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult3736.resultEvent
      SemanticRightRootResult3736.owner
      (SemanticRightRootResult3736.actual selector witness)
      SemanticRightRootResult3736.rawTerms
      SemanticRightRootResult3736.summary)
    (rightClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult3688.resultEvent
      SemanticRightRootResult3688.owner
      (SemanticRightRootResult3688.actual selector witness)
      SemanticRightRootResult3688.rawTerms
      SemanticRightRootResult3688.summary)
    (outputAt : (history.lookup resultEvent).map AnnotatedEvent.event =
      some (.resultExact owner rawTerms summary)) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  refine ⟨outputAt, ?_⟩
  exact exactValueClaim_add_of_mod_zero 100418593683253592432016548326729029359133068138294319235841 witness.env
    (SemanticRightRootResult3736.actual selector witness)
    (SemanticRightRootResult3688.actual selector witness) left right output
    (by simpa [left, leftRaw, SemanticRightRootResult3736.summary] using leftClaim.claim)
    (by simpa [right, rightRaw, SemanticRightRootResult3688.summary] using rightClaim.claim)
    (resultSound witness.env) (by decide)

theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply claimOfHistory selector witness (by rfl) (by rfl) (by rfl) (by rfl)
  · exact SemanticRightRootResult3736.claimSound
      selector selectorLower selectorUpper witness
  · exact SemanticRightRootResult3688.claimSound
      selector selectorLower selectorUpper witness
  · rfl

end SemanticRightRootResult3740

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
