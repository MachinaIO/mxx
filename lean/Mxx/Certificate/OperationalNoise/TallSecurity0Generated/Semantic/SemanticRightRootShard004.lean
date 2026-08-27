import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticRightRootShard003

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open TallSecurity0ABI
open TallSemantics

namespace SemanticRightRootResult736

def leftRaw : List Term := SemanticRightRootResult732.rawTerms
def rightRaw : List Term := SemanticRightRootResult711.rawTerms
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨15067⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨14906⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def owner : Owner := ⟨.program ⟨214⟩, ⟨15069⟩⟩
def rawTerms : List Term := outputRaw
def summary : Bound := .exactZero
def resultEvent : Nat := 736
theorem resultAgreement : CanonicalAgreement output (add left right) := by
  decide +kernel

theorem resultSound (env : Env Owner) :
    evalPolynomial env output = evalPolynomial env left + evalPolynomial env right := by
  exact addCanonicalResultSound env left right output resultAgreement

def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  SemanticRightRootResult732.actual selector witness +
    SemanticRightRootResult711.actual selector witness
theorem claimOfHistory (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841)
    (expressionAt : document.expressions.lookup 15069 =
      some Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Expression058.ExpressionRow15069)
    (leftPredecessorAt : (history.lookup 733).map AnnotatedEvent.event =
      some (.predecessor owner 0 ⟨14908⟩ 732))
    (rightPredecessorAt : (history.lookup 734).map AnnotatedEvent.event =
      some (.predecessor owner 1 ⟨15068⟩ 711))
    (ruleAt : (history.lookup 735).map AnnotatedEvent.event =
      some (.boundTransfer owner (.sum [.predecessor 0 733 .coefficient, .predecessor 1 734 .coefficient])))
    (leftClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult732.resultEvent
      SemanticRightRootResult732.owner
      (SemanticRightRootResult732.actual selector witness)
      SemanticRightRootResult732.rawTerms
      SemanticRightRootResult732.summary)
    (rightClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult711.resultEvent
      SemanticRightRootResult711.owner
      (SemanticRightRootResult711.actual selector witness)
      SemanticRightRootResult711.rawTerms
      SemanticRightRootResult711.summary)
    (outputAt : (history.lookup resultEvent).map AnnotatedEvent.event =
      some (.resultExact owner rawTerms summary)) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  refine ⟨outputAt, ?_⟩
  exact exactValueClaim_add_of_mod_zero 100418593683253592432016548326729029359133068138294319235841 witness.env
    (SemanticRightRootResult732.actual selector witness)
    (SemanticRightRootResult711.actual selector witness) left right output
    (by simpa [left, leftRaw, SemanticRightRootResult732.summary] using leftClaim.claim)
    (by simpa [right, rightRaw, SemanticRightRootResult711.summary] using rightClaim.claim)
    (resultSound witness.env) (by decide)

theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply claimOfHistory selector witness (by rfl) (by rfl) (by rfl) (by rfl)
  · exact SemanticRightRootResult732.claimSound
      selector selectorLower selectorUpper witness
  · exact SemanticRightRootResult711.claimSound
      selector selectorLower selectorUpper witness
  · rfl

end SemanticRightRootResult736

namespace SemanticRightRootResult740

def leftRaw : List Term := SemanticRightRootResult736.rawTerms
def rightRaw : List Term := SemanticRightRootResult701.rawTerms
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6452⟩⟩, ⟨.program ⟨214⟩, ⟨15228⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨15067⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨14906⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def owner : Owner := ⟨.program ⟨214⟩, ⟨15230⟩⟩
def rawTerms : List Term := outputRaw
def summary : Bound := .exactZero
def resultEvent : Nat := 740
theorem resultAgreement : CanonicalAgreement output (add left right) := by
  decide +kernel

theorem resultSound (env : Env Owner) :
    evalPolynomial env output = evalPolynomial env left + evalPolynomial env right := by
  exact addCanonicalResultSound env left right output resultAgreement

def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  SemanticRightRootResult736.actual selector witness +
    SemanticRightRootResult701.actual selector witness
theorem claimOfHistory (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841)
    (expressionAt : document.expressions.lookup 15230 =
      some Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Expression059.ExpressionRow15230)
    (leftPredecessorAt : (history.lookup 737).map AnnotatedEvent.event =
      some (.predecessor owner 0 ⟨15069⟩ 736))
    (rightPredecessorAt : (history.lookup 738).map AnnotatedEvent.event =
      some (.predecessor owner 1 ⟨15229⟩ 701))
    (ruleAt : (history.lookup 739).map AnnotatedEvent.event =
      some (.boundTransfer owner (.sum [.predecessor 0 737 .coefficient, .predecessor 1 738 .coefficient])))
    (leftClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult736.resultEvent
      SemanticRightRootResult736.owner
      (SemanticRightRootResult736.actual selector witness)
      SemanticRightRootResult736.rawTerms
      SemanticRightRootResult736.summary)
    (rightClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult701.resultEvent
      SemanticRightRootResult701.owner
      (SemanticRightRootResult701.actual selector witness)
      SemanticRightRootResult701.rawTerms
      SemanticRightRootResult701.summary)
    (outputAt : (history.lookup resultEvent).map AnnotatedEvent.event =
      some (.resultExact owner rawTerms summary)) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  refine ⟨outputAt, ?_⟩
  exact exactValueClaim_add_of_mod_zero 100418593683253592432016548326729029359133068138294319235841 witness.env
    (SemanticRightRootResult736.actual selector witness)
    (SemanticRightRootResult701.actual selector witness) left right output
    (by simpa [left, leftRaw, SemanticRightRootResult736.summary] using leftClaim.claim)
    (by simpa [right, rightRaw, SemanticRightRootResult701.summary] using rightClaim.claim)
    (resultSound witness.env) (by decide)

theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply claimOfHistory selector witness (by rfl) (by rfl) (by rfl) (by rfl)
  · exact SemanticRightRootResult736.claimSound
      selector selectorLower selectorUpper witness
  · exact SemanticRightRootResult701.claimSound
      selector selectorLower selectorUpper witness
  · rfl

end SemanticRightRootResult740

namespace SemanticRightRootResult744

def leftRaw : List Term := SemanticRightRootResult740.rawTerms
def rightRaw : List Term := SemanticRightRootResult691.rawTerms
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6427⟩⟩, ⟨.program ⟨214⟩, ⟨15536⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6452⟩⟩, ⟨.program ⟨214⟩, ⟨15228⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨15067⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨14906⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def owner : Owner := ⟨.program ⟨214⟩, ⟨15538⟩⟩
def rawTerms : List Term := outputRaw
def summary : Bound := .exactZero
def resultEvent : Nat := 744
theorem resultAgreement : CanonicalAgreement output (add left right) := by
  decide +kernel

theorem resultSound (env : Env Owner) :
    evalPolynomial env output = evalPolynomial env left + evalPolynomial env right := by
  exact addCanonicalResultSound env left right output resultAgreement

def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  SemanticRightRootResult740.actual selector witness +
    SemanticRightRootResult691.actual selector witness
theorem claimOfHistory (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841)
    (expressionAt : document.expressions.lookup 15538 =
      some Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Expression060.ExpressionRow15538)
    (leftPredecessorAt : (history.lookup 741).map AnnotatedEvent.event =
      some (.predecessor owner 0 ⟨15230⟩ 740))
    (rightPredecessorAt : (history.lookup 742).map AnnotatedEvent.event =
      some (.predecessor owner 1 ⟨15537⟩ 691))
    (ruleAt : (history.lookup 743).map AnnotatedEvent.event =
      some (.boundTransfer owner (.sum [.predecessor 0 741 .coefficient, .predecessor 1 742 .coefficient])))
    (leftClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult740.resultEvent
      SemanticRightRootResult740.owner
      (SemanticRightRootResult740.actual selector witness)
      SemanticRightRootResult740.rawTerms
      SemanticRightRootResult740.summary)
    (rightClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult691.resultEvent
      SemanticRightRootResult691.owner
      (SemanticRightRootResult691.actual selector witness)
      SemanticRightRootResult691.rawTerms
      SemanticRightRootResult691.summary)
    (outputAt : (history.lookup resultEvent).map AnnotatedEvent.event =
      some (.resultExact owner rawTerms summary)) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  refine ⟨outputAt, ?_⟩
  exact exactValueClaim_add_of_mod_zero 100418593683253592432016548326729029359133068138294319235841 witness.env
    (SemanticRightRootResult740.actual selector witness)
    (SemanticRightRootResult691.actual selector witness) left right output
    (by simpa [left, leftRaw, SemanticRightRootResult740.summary] using leftClaim.claim)
    (by simpa [right, rightRaw, SemanticRightRootResult691.summary] using rightClaim.claim)
    (resultSound witness.env) (by decide)

theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply claimOfHistory selector witness (by rfl) (by rfl) (by rfl) (by rfl)
  · exact SemanticRightRootResult740.claimSound
      selector selectorLower selectorUpper witness
  · exact SemanticRightRootResult691.claimSound
      selector selectorLower selectorUpper witness
  · rfl

end SemanticRightRootResult744

namespace SemanticRightRootResult748

def leftRaw : List Term := SemanticRightRootResult744.rawTerms
def rightRaw : List Term := SemanticRightRootResult681.rawTerms
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6407⟩⟩, ⟨.program ⟨214⟩, ⟨17846⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6427⟩⟩, ⟨.program ⟨214⟩, ⟨15536⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6452⟩⟩, ⟨.program ⟨214⟩, ⟨15228⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨15067⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨14906⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def owner : Owner := ⟨.program ⟨214⟩, ⟨17848⟩⟩
def rawTerms : List Term := outputRaw
def summary : Bound := .exactZero
def resultEvent : Nat := 748
theorem resultAgreement : CanonicalAgreement output (add left right) := by
  decide +kernel

theorem resultSound (env : Env Owner) :
    evalPolynomial env output = evalPolynomial env left + evalPolynomial env right := by
  exact addCanonicalResultSound env left right output resultAgreement

def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  SemanticRightRootResult744.actual selector witness +
    SemanticRightRootResult681.actual selector witness
theorem claimOfHistory (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841)
    (expressionAt : document.expressions.lookup 17848 =
      some Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Expression069.ExpressionRow17848)
    (leftPredecessorAt : (history.lookup 745).map AnnotatedEvent.event =
      some (.predecessor owner 0 ⟨15538⟩ 744))
    (rightPredecessorAt : (history.lookup 746).map AnnotatedEvent.event =
      some (.predecessor owner 1 ⟨17847⟩ 681))
    (ruleAt : (history.lookup 747).map AnnotatedEvent.event =
      some (.boundTransfer owner (.sum [.predecessor 0 745 .coefficient, .predecessor 1 746 .coefficient])))
    (leftClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult744.resultEvent
      SemanticRightRootResult744.owner
      (SemanticRightRootResult744.actual selector witness)
      SemanticRightRootResult744.rawTerms
      SemanticRightRootResult744.summary)
    (rightClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult681.resultEvent
      SemanticRightRootResult681.owner
      (SemanticRightRootResult681.actual selector witness)
      SemanticRightRootResult681.rawTerms
      SemanticRightRootResult681.summary)
    (outputAt : (history.lookup resultEvent).map AnnotatedEvent.event =
      some (.resultExact owner rawTerms summary)) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  refine ⟨outputAt, ?_⟩
  exact exactValueClaim_add_of_mod_zero 100418593683253592432016548326729029359133068138294319235841 witness.env
    (SemanticRightRootResult744.actual selector witness)
    (SemanticRightRootResult681.actual selector witness) left right output
    (by simpa [left, leftRaw, SemanticRightRootResult744.summary] using leftClaim.claim)
    (by simpa [right, rightRaw, SemanticRightRootResult681.summary] using rightClaim.claim)
    (resultSound witness.env) (by decide)

theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply claimOfHistory selector witness (by rfl) (by rfl) (by rfl) (by rfl)
  · exact SemanticRightRootResult744.claimSound
      selector selectorLower selectorUpper witness
  · exact SemanticRightRootResult681.claimSound
      selector selectorLower selectorUpper witness
  · rfl

end SemanticRightRootResult748

namespace SemanticRightRootResult752

def leftRaw : List Term := SemanticRightRootResult748.rawTerms
def rightRaw : List Term := SemanticRightRootResult671.rawTerms
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6398⟩⟩, ⟨.program ⟨214⟩, ⟨17454⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6407⟩⟩, ⟨.program ⟨214⟩, ⟨17846⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6427⟩⟩, ⟨.program ⟨214⟩, ⟨15536⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6452⟩⟩, ⟨.program ⟨214⟩, ⟨15228⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨15067⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨14906⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def owner : Owner := ⟨.program ⟨214⟩, ⟨17849⟩⟩
def rawTerms : List Term := outputRaw
def summary : Bound := .exactZero
def resultEvent : Nat := 752
theorem resultAgreement : CanonicalAgreement output (add left right) := by
  decide +kernel

theorem resultSound (env : Env Owner) :
    evalPolynomial env output = evalPolynomial env left + evalPolynomial env right := by
  exact addCanonicalResultSound env left right output resultAgreement

def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  SemanticRightRootResult748.actual selector witness +
    SemanticRightRootResult671.actual selector witness
theorem claimOfHistory (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841)
    (expressionAt : document.expressions.lookup 17849 =
      some Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Expression069.ExpressionRow17849)
    (leftPredecessorAt : (history.lookup 749).map AnnotatedEvent.event =
      some (.predecessor owner 0 ⟨17848⟩ 748))
    (rightPredecessorAt : (history.lookup 750).map AnnotatedEvent.event =
      some (.predecessor owner 1 ⟨17455⟩ 671))
    (ruleAt : (history.lookup 751).map AnnotatedEvent.event =
      some (.boundTransfer owner (.sum [.predecessor 0 749 .coefficient, .predecessor 1 750 .coefficient])))
    (leftClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult748.resultEvent
      SemanticRightRootResult748.owner
      (SemanticRightRootResult748.actual selector witness)
      SemanticRightRootResult748.rawTerms
      SemanticRightRootResult748.summary)
    (rightClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult671.resultEvent
      SemanticRightRootResult671.owner
      (SemanticRightRootResult671.actual selector witness)
      SemanticRightRootResult671.rawTerms
      SemanticRightRootResult671.summary)
    (outputAt : (history.lookup resultEvent).map AnnotatedEvent.event =
      some (.resultExact owner rawTerms summary)) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  refine ⟨outputAt, ?_⟩
  exact exactValueClaim_add_of_mod_zero 100418593683253592432016548326729029359133068138294319235841 witness.env
    (SemanticRightRootResult748.actual selector witness)
    (SemanticRightRootResult671.actual selector witness) left right output
    (by simpa [left, leftRaw, SemanticRightRootResult748.summary] using leftClaim.claim)
    (by simpa [right, rightRaw, SemanticRightRootResult671.summary] using rightClaim.claim)
    (resultSound witness.env) (by decide)

theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply claimOfHistory selector witness (by rfl) (by rfl) (by rfl) (by rfl)
  · exact SemanticRightRootResult748.claimSound
      selector selectorLower selectorUpper witness
  · exact SemanticRightRootResult671.claimSound
      selector selectorLower selectorUpper witness
  · rfl

end SemanticRightRootResult752

namespace SemanticRightRootResult756

def leftRaw : List Term := SemanticRightRootResult752.rawTerms
def rightRaw : List Term := SemanticRightRootResult661.rawTerms
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6391⟩⟩, ⟨.program ⟨214⟩, ⟨17237⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6398⟩⟩, ⟨.program ⟨214⟩, ⟨17454⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6407⟩⟩, ⟨.program ⟨214⟩, ⟨17846⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6427⟩⟩, ⟨.program ⟨214⟩, ⟨15536⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6452⟩⟩, ⟨.program ⟨214⟩, ⟨15228⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨15067⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨14906⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def owner : Owner := ⟨.program ⟨214⟩, ⟨17850⟩⟩
def rawTerms : List Term := outputRaw
def summary : Bound := .exactZero
def resultEvent : Nat := 756
theorem resultAgreement : CanonicalAgreement output (add left right) := by
  decide +kernel

theorem resultSound (env : Env Owner) :
    evalPolynomial env output = evalPolynomial env left + evalPolynomial env right := by
  exact addCanonicalResultSound env left right output resultAgreement

def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  SemanticRightRootResult752.actual selector witness +
    SemanticRightRootResult661.actual selector witness
theorem claimOfHistory (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841)
    (expressionAt : document.expressions.lookup 17850 =
      some Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Expression069.ExpressionRow17850)
    (leftPredecessorAt : (history.lookup 753).map AnnotatedEvent.event =
      some (.predecessor owner 0 ⟨17849⟩ 752))
    (rightPredecessorAt : (history.lookup 754).map AnnotatedEvent.event =
      some (.predecessor owner 1 ⟨17238⟩ 661))
    (ruleAt : (history.lookup 755).map AnnotatedEvent.event =
      some (.boundTransfer owner (.sum [.predecessor 0 753 .coefficient, .predecessor 1 754 .coefficient])))
    (leftClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult752.resultEvent
      SemanticRightRootResult752.owner
      (SemanticRightRootResult752.actual selector witness)
      SemanticRightRootResult752.rawTerms
      SemanticRightRootResult752.summary)
    (rightClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult661.resultEvent
      SemanticRightRootResult661.owner
      (SemanticRightRootResult661.actual selector witness)
      SemanticRightRootResult661.rawTerms
      SemanticRightRootResult661.summary)
    (outputAt : (history.lookup resultEvent).map AnnotatedEvent.event =
      some (.resultExact owner rawTerms summary)) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  refine ⟨outputAt, ?_⟩
  exact exactValueClaim_add_of_mod_zero 100418593683253592432016548326729029359133068138294319235841 witness.env
    (SemanticRightRootResult752.actual selector witness)
    (SemanticRightRootResult661.actual selector witness) left right output
    (by simpa [left, leftRaw, SemanticRightRootResult752.summary] using leftClaim.claim)
    (by simpa [right, rightRaw, SemanticRightRootResult661.summary] using rightClaim.claim)
    (resultSound witness.env) (by decide)

theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply claimOfHistory selector witness (by rfl) (by rfl) (by rfl) (by rfl)
  · exact SemanticRightRootResult752.claimSound
      selector selectorLower selectorUpper witness
  · exact SemanticRightRootResult661.claimSound
      selector selectorLower selectorUpper witness
  · rfl

end SemanticRightRootResult756

namespace SemanticRightRootResult760

def leftRaw : List Term := SemanticRightRootResult756.rawTerms
def rightRaw : List Term := SemanticRightRootResult651.rawTerms
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6387⟩⟩, ⟨.program ⟨214⟩, ⟨17181⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6391⟩⟩, ⟨.program ⟨214⟩, ⟨17237⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6398⟩⟩, ⟨.program ⟨214⟩, ⟨17454⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6407⟩⟩, ⟨.program ⟨214⟩, ⟨17846⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6427⟩⟩, ⟨.program ⟨214⟩, ⟨15536⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6452⟩⟩, ⟨.program ⟨214⟩, ⟨15228⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨15067⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨14906⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def owner : Owner := ⟨.program ⟨214⟩, ⟨17851⟩⟩
def rawTerms : List Term := outputRaw
def summary : Bound := .exactZero
def resultEvent : Nat := 760
theorem resultAgreement : CanonicalAgreement output (add left right) := by
  decide +kernel

theorem resultSound (env : Env Owner) :
    evalPolynomial env output = evalPolynomial env left + evalPolynomial env right := by
  exact addCanonicalResultSound env left right output resultAgreement

def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  SemanticRightRootResult756.actual selector witness +
    SemanticRightRootResult651.actual selector witness
theorem claimOfHistory (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841)
    (expressionAt : document.expressions.lookup 17851 =
      some Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Expression069.ExpressionRow17851)
    (leftPredecessorAt : (history.lookup 757).map AnnotatedEvent.event =
      some (.predecessor owner 0 ⟨17850⟩ 756))
    (rightPredecessorAt : (history.lookup 758).map AnnotatedEvent.event =
      some (.predecessor owner 1 ⟨17182⟩ 651))
    (ruleAt : (history.lookup 759).map AnnotatedEvent.event =
      some (.boundTransfer owner (.sum [.predecessor 0 757 .coefficient, .predecessor 1 758 .coefficient])))
    (leftClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult756.resultEvent
      SemanticRightRootResult756.owner
      (SemanticRightRootResult756.actual selector witness)
      SemanticRightRootResult756.rawTerms
      SemanticRightRootResult756.summary)
    (rightClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult651.resultEvent
      SemanticRightRootResult651.owner
      (SemanticRightRootResult651.actual selector witness)
      SemanticRightRootResult651.rawTerms
      SemanticRightRootResult651.summary)
    (outputAt : (history.lookup resultEvent).map AnnotatedEvent.event =
      some (.resultExact owner rawTerms summary)) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  refine ⟨outputAt, ?_⟩
  exact exactValueClaim_add_of_mod_zero 100418593683253592432016548326729029359133068138294319235841 witness.env
    (SemanticRightRootResult756.actual selector witness)
    (SemanticRightRootResult651.actual selector witness) left right output
    (by simpa [left, leftRaw, SemanticRightRootResult756.summary] using leftClaim.claim)
    (by simpa [right, rightRaw, SemanticRightRootResult651.summary] using rightClaim.claim)
    (resultSound witness.env) (by decide)

theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply claimOfHistory selector witness (by rfl) (by rfl) (by rfl) (by rfl)
  · exact SemanticRightRootResult756.claimSound
      selector selectorLower selectorUpper witness
  · exact SemanticRightRootResult651.claimSound
      selector selectorLower selectorUpper witness
  · rfl

end SemanticRightRootResult760

namespace SemanticRightRootResult764

def leftRaw : List Term := SemanticRightRootResult760.rawTerms
def rightRaw : List Term := SemanticRightRootResult641.rawTerms
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6383⟩⟩, ⟨.program ⟨214⟩, ⟨18063⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6387⟩⟩, ⟨.program ⟨214⟩, ⟨17181⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6391⟩⟩, ⟨.program ⟨214⟩, ⟨17237⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6398⟩⟩, ⟨.program ⟨214⟩, ⟨17454⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6407⟩⟩, ⟨.program ⟨214⟩, ⟨17846⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6427⟩⟩, ⟨.program ⟨214⟩, ⟨15536⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6452⟩⟩, ⟨.program ⟨214⟩, ⟨15228⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨15067⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨14906⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def owner : Owner := ⟨.program ⟨214⟩, ⟨18065⟩⟩
def rawTerms : List Term := outputRaw
def summary : Bound := .exactZero
def resultEvent : Nat := 764
theorem resultAgreement : CanonicalAgreement output (add left right) := by
  decide +kernel

theorem resultSound (env : Env Owner) :
    evalPolynomial env output = evalPolynomial env left + evalPolynomial env right := by
  exact addCanonicalResultSound env left right output resultAgreement

def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  SemanticRightRootResult760.actual selector witness +
    SemanticRightRootResult641.actual selector witness
theorem claimOfHistory (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841)
    (expressionAt : document.expressions.lookup 18065 =
      some Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Expression070.ExpressionRow18065)
    (leftPredecessorAt : (history.lookup 761).map AnnotatedEvent.event =
      some (.predecessor owner 0 ⟨17851⟩ 760))
    (rightPredecessorAt : (history.lookup 762).map AnnotatedEvent.event =
      some (.predecessor owner 1 ⟨18064⟩ 641))
    (ruleAt : (history.lookup 763).map AnnotatedEvent.event =
      some (.boundTransfer owner (.sum [.predecessor 0 761 .coefficient, .predecessor 1 762 .coefficient])))
    (leftClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult760.resultEvent
      SemanticRightRootResult760.owner
      (SemanticRightRootResult760.actual selector witness)
      SemanticRightRootResult760.rawTerms
      SemanticRightRootResult760.summary)
    (rightClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult641.resultEvent
      SemanticRightRootResult641.owner
      (SemanticRightRootResult641.actual selector witness)
      SemanticRightRootResult641.rawTerms
      SemanticRightRootResult641.summary)
    (outputAt : (history.lookup resultEvent).map AnnotatedEvent.event =
      some (.resultExact owner rawTerms summary)) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  refine ⟨outputAt, ?_⟩
  exact exactValueClaim_add_of_mod_zero 100418593683253592432016548326729029359133068138294319235841 witness.env
    (SemanticRightRootResult760.actual selector witness)
    (SemanticRightRootResult641.actual selector witness) left right output
    (by simpa [left, leftRaw, SemanticRightRootResult760.summary] using leftClaim.claim)
    (by simpa [right, rightRaw, SemanticRightRootResult641.summary] using rightClaim.claim)
    (resultSound witness.env) (by decide)

theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply claimOfHistory selector witness (by rfl) (by rfl) (by rfl) (by rfl)
  · exact SemanticRightRootResult760.claimSound
      selector selectorLower selectorUpper witness
  · exact SemanticRightRootResult641.claimSound
      selector selectorLower selectorUpper witness
  · rfl

end SemanticRightRootResult764

namespace SemanticRightRootResult768

def leftRaw : List Term := SemanticRightRootResult764.rawTerms
def rightRaw : List Term := SemanticRightRootResult631.rawTerms
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6383⟩⟩, ⟨.program ⟨214⟩, ⟨18063⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6387⟩⟩, ⟨.program ⟨214⟩, ⟨17181⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6391⟩⟩, ⟨.program ⟨214⟩, ⟨17237⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6398⟩⟩, ⟨.program ⟨214⟩, ⟨17454⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6407⟩⟩, ⟨.program ⟨214⟩, ⟨17846⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6427⟩⟩, ⟨.program ⟨214⟩, ⟨15536⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6452⟩⟩, ⟨.program ⟨214⟩, ⟨15228⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨15067⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨14906⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6502⟩⟩, ⟨.program ⟨214⟩, ⟨17678⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def owner : Owner := ⟨.program ⟨214⟩, ⟨18066⟩⟩
def rawTerms : List Term := outputRaw
def summary : Bound := .exactZero
def resultEvent : Nat := 768
theorem resultAgreement : CanonicalAgreement output (add left right) := by
  decide +kernel

theorem resultSound (env : Env Owner) :
    evalPolynomial env output = evalPolynomial env left + evalPolynomial env right := by
  exact addCanonicalResultSound env left right output resultAgreement

def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  SemanticRightRootResult764.actual selector witness +
    SemanticRightRootResult631.actual selector witness
theorem claimOfHistory (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841)
    (expressionAt : document.expressions.lookup 18066 =
      some Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Expression070.ExpressionRow18066)
    (leftPredecessorAt : (history.lookup 765).map AnnotatedEvent.event =
      some (.predecessor owner 0 ⟨18065⟩ 764))
    (rightPredecessorAt : (history.lookup 766).map AnnotatedEvent.event =
      some (.predecessor owner 1 ⟨17679⟩ 631))
    (ruleAt : (history.lookup 767).map AnnotatedEvent.event =
      some (.boundTransfer owner (.sum [.predecessor 0 765 .coefficient, .predecessor 1 766 .coefficient])))
    (leftClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult764.resultEvent
      SemanticRightRootResult764.owner
      (SemanticRightRootResult764.actual selector witness)
      SemanticRightRootResult764.rawTerms
      SemanticRightRootResult764.summary)
    (rightClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult631.resultEvent
      SemanticRightRootResult631.owner
      (SemanticRightRootResult631.actual selector witness)
      SemanticRightRootResult631.rawTerms
      SemanticRightRootResult631.summary)
    (outputAt : (history.lookup resultEvent).map AnnotatedEvent.event =
      some (.resultExact owner rawTerms summary)) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  refine ⟨outputAt, ?_⟩
  exact exactValueClaim_add_of_mod_zero 100418593683253592432016548326729029359133068138294319235841 witness.env
    (SemanticRightRootResult764.actual selector witness)
    (SemanticRightRootResult631.actual selector witness) left right output
    (by simpa [left, leftRaw, SemanticRightRootResult764.summary] using leftClaim.claim)
    (by simpa [right, rightRaw, SemanticRightRootResult631.summary] using rightClaim.claim)
    (resultSound witness.env) (by decide)

theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply claimOfHistory selector witness (by rfl) (by rfl) (by rfl) (by rfl)
  · exact SemanticRightRootResult764.claimSound
      selector selectorLower selectorUpper witness
  · exact SemanticRightRootResult631.claimSound
      selector selectorLower selectorUpper witness
  · rfl

end SemanticRightRootResult768

namespace SemanticRightRootResult772

def leftRaw : List Term := SemanticRightRootResult768.rawTerms
def rightRaw : List Term := SemanticRightRootResult621.rawTerms
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6383⟩⟩, ⟨.program ⟨214⟩, ⟨18063⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6387⟩⟩, ⟨.program ⟨214⟩, ⟨17181⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6391⟩⟩, ⟨.program ⟨214⟩, ⟨17237⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6398⟩⟩, ⟨.program ⟨214⟩, ⟨17454⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6407⟩⟩, ⟨.program ⟨214⟩, ⟨17846⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6427⟩⟩, ⟨.program ⟨214⟩, ⟨15536⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6452⟩⟩, ⟨.program ⟨214⟩, ⟨15228⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨15067⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6494⟩⟩, ⟨.program ⟨214⟩, ⟨17622⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨14906⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6502⟩⟩, ⟨.program ⟨214⟩, ⟨17678⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def owner : Owner := ⟨.program ⟨214⟩, ⟨18067⟩⟩
def rawTerms : List Term := outputRaw
def summary : Bound := .exactZero
def resultEvent : Nat := 772
theorem resultAgreement : CanonicalAgreement output (add left right) := by
  decide +kernel

theorem resultSound (env : Env Owner) :
    evalPolynomial env output = evalPolynomial env left + evalPolynomial env right := by
  exact addCanonicalResultSound env left right output resultAgreement

def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  SemanticRightRootResult768.actual selector witness +
    SemanticRightRootResult621.actual selector witness
theorem claimOfHistory (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841)
    (expressionAt : document.expressions.lookup 18067 =
      some Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Expression070.ExpressionRow18067)
    (leftPredecessorAt : (history.lookup 769).map AnnotatedEvent.event =
      some (.predecessor owner 0 ⟨18066⟩ 768))
    (rightPredecessorAt : (history.lookup 770).map AnnotatedEvent.event =
      some (.predecessor owner 1 ⟨17623⟩ 621))
    (ruleAt : (history.lookup 771).map AnnotatedEvent.event =
      some (.boundTransfer owner (.sum [.predecessor 0 769 .coefficient, .predecessor 1 770 .coefficient])))
    (leftClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult768.resultEvent
      SemanticRightRootResult768.owner
      (SemanticRightRootResult768.actual selector witness)
      SemanticRightRootResult768.rawTerms
      SemanticRightRootResult768.summary)
    (rightClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult621.resultEvent
      SemanticRightRootResult621.owner
      (SemanticRightRootResult621.actual selector witness)
      SemanticRightRootResult621.rawTerms
      SemanticRightRootResult621.summary)
    (outputAt : (history.lookup resultEvent).map AnnotatedEvent.event =
      some (.resultExact owner rawTerms summary)) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  refine ⟨outputAt, ?_⟩
  exact exactValueClaim_add_of_mod_zero 100418593683253592432016548326729029359133068138294319235841 witness.env
    (SemanticRightRootResult768.actual selector witness)
    (SemanticRightRootResult621.actual selector witness) left right output
    (by simpa [left, leftRaw, SemanticRightRootResult768.summary] using leftClaim.claim)
    (by simpa [right, rightRaw, SemanticRightRootResult621.summary] using rightClaim.claim)
    (resultSound witness.env) (by decide)

theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply claimOfHistory selector witness (by rfl) (by rfl) (by rfl) (by rfl)
  · exact SemanticRightRootResult768.claimSound
      selector selectorLower selectorUpper witness
  · exact SemanticRightRootResult621.claimSound
      selector selectorLower selectorUpper witness
  · rfl

end SemanticRightRootResult772

namespace SemanticRightRootResult776

def leftRaw : List Term := SemanticRightRootResult772.rawTerms
def rightRaw : List Term := SemanticRightRootResult611.rawTerms
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6383⟩⟩, ⟨.program ⟨214⟩, ⟨18063⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6387⟩⟩, ⟨.program ⟨214⟩, ⟨17181⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6391⟩⟩, ⟨.program ⟨214⟩, ⟨17237⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6398⟩⟩, ⟨.program ⟨214⟩, ⟨17454⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6407⟩⟩, ⟨.program ⟨214⟩, ⟨17846⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6427⟩⟩, ⟨.program ⟨214⟩, ⟨15536⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6452⟩⟩, ⟨.program ⟨214⟩, ⟨15228⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨15067⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6490⟩⟩, ⟨.program ⟨214⟩, ⟨18893⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6494⟩⟩, ⟨.program ⟨214⟩, ⟨17622⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨14906⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6502⟩⟩, ⟨.program ⟨214⟩, ⟨17678⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def owner : Owner := ⟨.program ⟨214⟩, ⟨18895⟩⟩
def rawTerms : List Term := outputRaw
def summary : Bound := .exactZero
def resultEvent : Nat := 776
theorem resultAgreement : CanonicalAgreement output (add left right) := by
  decide +kernel

theorem resultSound (env : Env Owner) :
    evalPolynomial env output = evalPolynomial env left + evalPolynomial env right := by
  exact addCanonicalResultSound env left right output resultAgreement

def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  SemanticRightRootResult772.actual selector witness +
    SemanticRightRootResult611.actual selector witness
theorem claimOfHistory (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841)
    (expressionAt : document.expressions.lookup 18895 =
      some Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Expression073.ExpressionRow18895)
    (leftPredecessorAt : (history.lookup 773).map AnnotatedEvent.event =
      some (.predecessor owner 0 ⟨18067⟩ 772))
    (rightPredecessorAt : (history.lookup 774).map AnnotatedEvent.event =
      some (.predecessor owner 1 ⟨18894⟩ 611))
    (ruleAt : (history.lookup 775).map AnnotatedEvent.event =
      some (.boundTransfer owner (.sum [.predecessor 0 773 .coefficient, .predecessor 1 774 .coefficient])))
    (leftClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult772.resultEvent
      SemanticRightRootResult772.owner
      (SemanticRightRootResult772.actual selector witness)
      SemanticRightRootResult772.rawTerms
      SemanticRightRootResult772.summary)
    (rightClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult611.resultEvent
      SemanticRightRootResult611.owner
      (SemanticRightRootResult611.actual selector witness)
      SemanticRightRootResult611.rawTerms
      SemanticRightRootResult611.summary)
    (outputAt : (history.lookup resultEvent).map AnnotatedEvent.event =
      some (.resultExact owner rawTerms summary)) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  refine ⟨outputAt, ?_⟩
  exact exactValueClaim_add_of_mod_zero 100418593683253592432016548326729029359133068138294319235841 witness.env
    (SemanticRightRootResult772.actual selector witness)
    (SemanticRightRootResult611.actual selector witness) left right output
    (by simpa [left, leftRaw, SemanticRightRootResult772.summary] using leftClaim.claim)
    (by simpa [right, rightRaw, SemanticRightRootResult611.summary] using rightClaim.claim)
    (resultSound witness.env) (by decide)

theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply claimOfHistory selector witness (by rfl) (by rfl) (by rfl) (by rfl)
  · exact SemanticRightRootResult772.claimSound
      selector selectorLower selectorUpper witness
  · exact SemanticRightRootResult611.claimSound
      selector selectorLower selectorUpper witness
  · rfl

end SemanticRightRootResult776

namespace SemanticRightRootResult780

def leftRaw : List Term := SemanticRightRootResult776.rawTerms
def rightRaw : List Term := SemanticRightRootResult601.rawTerms
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6383⟩⟩, ⟨.program ⟨214⟩, ⟨18063⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6387⟩⟩, ⟨.program ⟨214⟩, ⟨17181⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6391⟩⟩, ⟨.program ⟨214⟩, ⟨17237⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6398⟩⟩, ⟨.program ⟨214⟩, ⟨17454⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6407⟩⟩, ⟨.program ⟨214⟩, ⟨17846⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6427⟩⟩, ⟨.program ⟨214⟩, ⟨15536⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6452⟩⟩, ⟨.program ⟨214⟩, ⟨15228⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6473⟩⟩, ⟨.program ⟨214⟩, ⟨17566⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨15067⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6490⟩⟩, ⟨.program ⟨214⟩, ⟨18893⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6494⟩⟩, ⟨.program ⟨214⟩, ⟨17622⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨14906⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6502⟩⟩, ⟨.program ⟨214⟩, ⟨17678⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def owner : Owner := ⟨.program ⟨214⟩, ⟨18896⟩⟩
def rawTerms : List Term := outputRaw
def summary : Bound := .exactZero
def resultEvent : Nat := 780
theorem resultAgreement : CanonicalAgreement output (add left right) := by
  decide +kernel

theorem resultSound (env : Env Owner) :
    evalPolynomial env output = evalPolynomial env left + evalPolynomial env right := by
  exact addCanonicalResultSound env left right output resultAgreement

def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  SemanticRightRootResult776.actual selector witness +
    SemanticRightRootResult601.actual selector witness
theorem claimOfHistory (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841)
    (expressionAt : document.expressions.lookup 18896 =
      some Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Expression073.ExpressionRow18896)
    (leftPredecessorAt : (history.lookup 777).map AnnotatedEvent.event =
      some (.predecessor owner 0 ⟨18895⟩ 776))
    (rightPredecessorAt : (history.lookup 778).map AnnotatedEvent.event =
      some (.predecessor owner 1 ⟨17567⟩ 601))
    (ruleAt : (history.lookup 779).map AnnotatedEvent.event =
      some (.boundTransfer owner (.sum [.predecessor 0 777 .coefficient, .predecessor 1 778 .coefficient])))
    (leftClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult776.resultEvent
      SemanticRightRootResult776.owner
      (SemanticRightRootResult776.actual selector witness)
      SemanticRightRootResult776.rawTerms
      SemanticRightRootResult776.summary)
    (rightClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult601.resultEvent
      SemanticRightRootResult601.owner
      (SemanticRightRootResult601.actual selector witness)
      SemanticRightRootResult601.rawTerms
      SemanticRightRootResult601.summary)
    (outputAt : (history.lookup resultEvent).map AnnotatedEvent.event =
      some (.resultExact owner rawTerms summary)) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  refine ⟨outputAt, ?_⟩
  exact exactValueClaim_add_of_mod_zero 100418593683253592432016548326729029359133068138294319235841 witness.env
    (SemanticRightRootResult776.actual selector witness)
    (SemanticRightRootResult601.actual selector witness) left right output
    (by simpa [left, leftRaw, SemanticRightRootResult776.summary] using leftClaim.claim)
    (by simpa [right, rightRaw, SemanticRightRootResult601.summary] using rightClaim.claim)
    (resultSound witness.env) (by decide)

theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply claimOfHistory selector witness (by rfl) (by rfl) (by rfl) (by rfl)
  · exact SemanticRightRootResult776.claimSound
      selector selectorLower selectorUpper witness
  · exact SemanticRightRootResult601.claimSound
      selector selectorLower selectorUpper witness
  · rfl

end SemanticRightRootResult780

namespace SemanticRightRootResult784

def leftRaw : List Term := SemanticRightRootResult780.rawTerms
def rightRaw : List Term := SemanticRightRootResult591.rawTerms
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6383⟩⟩, ⟨.program ⟨214⟩, ⟨18063⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6387⟩⟩, ⟨.program ⟨214⟩, ⟨17181⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6391⟩⟩, ⟨.program ⟨214⟩, ⟨17237⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6398⟩⟩, ⟨.program ⟨214⟩, ⟨17454⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6407⟩⟩, ⟨.program ⟨214⟩, ⟨17846⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6427⟩⟩, ⟨.program ⟨214⟩, ⟨15536⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6452⟩⟩, ⟨.program ⟨214⟩, ⟨15228⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6467⟩⟩, ⟨.program ⟨214⟩, ⟨17965⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6473⟩⟩, ⟨.program ⟨214⟩, ⟨17566⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨15067⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6490⟩⟩, ⟨.program ⟨214⟩, ⟨18893⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6494⟩⟩, ⟨.program ⟨214⟩, ⟨17622⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨14906⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6502⟩⟩, ⟨.program ⟨214⟩, ⟨17678⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def owner : Owner := ⟨.program ⟨214⟩, ⟨18897⟩⟩
def rawTerms : List Term := outputRaw
def summary : Bound := .exactZero
def resultEvent : Nat := 784
theorem resultAgreement : CanonicalAgreement output (add left right) := by
  decide +kernel

theorem resultSound (env : Env Owner) :
    evalPolynomial env output = evalPolynomial env left + evalPolynomial env right := by
  exact addCanonicalResultSound env left right output resultAgreement

def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  SemanticRightRootResult780.actual selector witness +
    SemanticRightRootResult591.actual selector witness
theorem claimOfHistory (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841)
    (expressionAt : document.expressions.lookup 18897 =
      some Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Expression073.ExpressionRow18897)
    (leftPredecessorAt : (history.lookup 781).map AnnotatedEvent.event =
      some (.predecessor owner 0 ⟨18896⟩ 780))
    (rightPredecessorAt : (history.lookup 782).map AnnotatedEvent.event =
      some (.predecessor owner 1 ⟨17966⟩ 591))
    (ruleAt : (history.lookup 783).map AnnotatedEvent.event =
      some (.boundTransfer owner (.sum [.predecessor 0 781 .coefficient, .predecessor 1 782 .coefficient])))
    (leftClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult780.resultEvent
      SemanticRightRootResult780.owner
      (SemanticRightRootResult780.actual selector witness)
      SemanticRightRootResult780.rawTerms
      SemanticRightRootResult780.summary)
    (rightClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult591.resultEvent
      SemanticRightRootResult591.owner
      (SemanticRightRootResult591.actual selector witness)
      SemanticRightRootResult591.rawTerms
      SemanticRightRootResult591.summary)
    (outputAt : (history.lookup resultEvent).map AnnotatedEvent.event =
      some (.resultExact owner rawTerms summary)) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  refine ⟨outputAt, ?_⟩
  exact exactValueClaim_add_of_mod_zero 100418593683253592432016548326729029359133068138294319235841 witness.env
    (SemanticRightRootResult780.actual selector witness)
    (SemanticRightRootResult591.actual selector witness) left right output
    (by simpa [left, leftRaw, SemanticRightRootResult780.summary] using leftClaim.claim)
    (by simpa [right, rightRaw, SemanticRightRootResult591.summary] using rightClaim.claim)
    (resultSound witness.env) (by decide)

theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply claimOfHistory selector witness (by rfl) (by rfl) (by rfl) (by rfl)
  · exact SemanticRightRootResult780.claimSound
      selector selectorLower selectorUpper witness
  · exact SemanticRightRootResult591.claimSound
      selector selectorLower selectorUpper witness
  · rfl

end SemanticRightRootResult784

namespace SemanticRightRootResult788

def leftRaw : List Term := SemanticRightRootResult784.rawTerms
def rightRaw : List Term := SemanticRightRootResult581.rawTerms
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6383⟩⟩, ⟨.program ⟨214⟩, ⟨18063⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6387⟩⟩, ⟨.program ⟨214⟩, ⟨17181⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6391⟩⟩, ⟨.program ⟨214⟩, ⟨17237⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6398⟩⟩, ⟨.program ⟨214⟩, ⟨17454⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6407⟩⟩, ⟨.program ⟨214⟩, ⟨17846⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6427⟩⟩, ⟨.program ⟨214⟩, ⟨15536⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6452⟩⟩, ⟨.program ⟨214⟩, ⟨15228⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6459⟩⟩, ⟨.program ⟨214⟩, ⟨17734⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6467⟩⟩, ⟨.program ⟨214⟩, ⟨17965⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6473⟩⟩, ⟨.program ⟨214⟩, ⟨17566⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨15067⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6490⟩⟩, ⟨.program ⟨214⟩, ⟨18893⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6494⟩⟩, ⟨.program ⟨214⟩, ⟨17622⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨14906⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6502⟩⟩, ⟨.program ⟨214⟩, ⟨17678⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def owner : Owner := ⟨.program ⟨214⟩, ⟨18898⟩⟩
def rawTerms : List Term := outputRaw
def summary : Bound := .exactZero
def resultEvent : Nat := 788
theorem resultAgreement : CanonicalAgreement output (add left right) := by
  decide +kernel

theorem resultSound (env : Env Owner) :
    evalPolynomial env output = evalPolynomial env left + evalPolynomial env right := by
  exact addCanonicalResultSound env left right output resultAgreement

def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  SemanticRightRootResult784.actual selector witness +
    SemanticRightRootResult581.actual selector witness
theorem claimOfHistory (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841)
    (expressionAt : document.expressions.lookup 18898 =
      some Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Expression073.ExpressionRow18898)
    (leftPredecessorAt : (history.lookup 785).map AnnotatedEvent.event =
      some (.predecessor owner 0 ⟨18897⟩ 784))
    (rightPredecessorAt : (history.lookup 786).map AnnotatedEvent.event =
      some (.predecessor owner 1 ⟨17735⟩ 581))
    (ruleAt : (history.lookup 787).map AnnotatedEvent.event =
      some (.boundTransfer owner (.sum [.predecessor 0 785 .coefficient, .predecessor 1 786 .coefficient])))
    (leftClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult784.resultEvent
      SemanticRightRootResult784.owner
      (SemanticRightRootResult784.actual selector witness)
      SemanticRightRootResult784.rawTerms
      SemanticRightRootResult784.summary)
    (rightClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult581.resultEvent
      SemanticRightRootResult581.owner
      (SemanticRightRootResult581.actual selector witness)
      SemanticRightRootResult581.rawTerms
      SemanticRightRootResult581.summary)
    (outputAt : (history.lookup resultEvent).map AnnotatedEvent.event =
      some (.resultExact owner rawTerms summary)) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  refine ⟨outputAt, ?_⟩
  exact exactValueClaim_add_of_mod_zero 100418593683253592432016548326729029359133068138294319235841 witness.env
    (SemanticRightRootResult784.actual selector witness)
    (SemanticRightRootResult581.actual selector witness) left right output
    (by simpa [left, leftRaw, SemanticRightRootResult784.summary] using leftClaim.claim)
    (by simpa [right, rightRaw, SemanticRightRootResult581.summary] using rightClaim.claim)
    (resultSound witness.env) (by decide)

theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply claimOfHistory selector witness (by rfl) (by rfl) (by rfl) (by rfl)
  · exact SemanticRightRootResult784.claimSound
      selector selectorLower selectorUpper witness
  · exact SemanticRightRootResult581.claimSound
      selector selectorLower selectorUpper witness
  · rfl

end SemanticRightRootResult788

namespace SemanticRightRootResult792

def leftRaw : List Term := SemanticRightRootResult788.rawTerms
def rightRaw : List Term := SemanticRightRootResult571.rawTerms
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6383⟩⟩, ⟨.program ⟨214⟩, ⟨18063⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6387⟩⟩, ⟨.program ⟨214⟩, ⟨17181⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6391⟩⟩, ⟨.program ⟨214⟩, ⟨17237⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6398⟩⟩, ⟨.program ⟨214⟩, ⟨17454⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6407⟩⟩, ⟨.program ⟨214⟩, ⟨17846⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6427⟩⟩, ⟨.program ⟨214⟩, ⟨15536⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6449⟩⟩, ⟨.program ⟨214⟩, ⟨17510⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6452⟩⟩, ⟨.program ⟨214⟩, ⟨15228⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6459⟩⟩, ⟨.program ⟨214⟩, ⟨17734⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6467⟩⟩, ⟨.program ⟨214⟩, ⟨17965⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6473⟩⟩, ⟨.program ⟨214⟩, ⟨17566⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨15067⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6490⟩⟩, ⟨.program ⟨214⟩, ⟨18893⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6494⟩⟩, ⟨.program ⟨214⟩, ⟨17622⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨14906⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6502⟩⟩, ⟨.program ⟨214⟩, ⟨17678⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def owner : Owner := ⟨.program ⟨214⟩, ⟨18899⟩⟩
def rawTerms : List Term := outputRaw
def summary : Bound := .exactZero
def resultEvent : Nat := 792
theorem resultAgreement : CanonicalAgreement output (add left right) := by
  decide +kernel

theorem resultSound (env : Env Owner) :
    evalPolynomial env output = evalPolynomial env left + evalPolynomial env right := by
  exact addCanonicalResultSound env left right output resultAgreement

def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  SemanticRightRootResult788.actual selector witness +
    SemanticRightRootResult571.actual selector witness
theorem claimOfHistory (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841)
    (expressionAt : document.expressions.lookup 18899 =
      some Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Expression073.ExpressionRow18899)
    (leftPredecessorAt : (history.lookup 789).map AnnotatedEvent.event =
      some (.predecessor owner 0 ⟨18898⟩ 788))
    (rightPredecessorAt : (history.lookup 790).map AnnotatedEvent.event =
      some (.predecessor owner 1 ⟨17511⟩ 571))
    (ruleAt : (history.lookup 791).map AnnotatedEvent.event =
      some (.boundTransfer owner (.sum [.predecessor 0 789 .coefficient, .predecessor 1 790 .coefficient])))
    (leftClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult788.resultEvent
      SemanticRightRootResult788.owner
      (SemanticRightRootResult788.actual selector witness)
      SemanticRightRootResult788.rawTerms
      SemanticRightRootResult788.summary)
    (rightClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult571.resultEvent
      SemanticRightRootResult571.owner
      (SemanticRightRootResult571.actual selector witness)
      SemanticRightRootResult571.rawTerms
      SemanticRightRootResult571.summary)
    (outputAt : (history.lookup resultEvent).map AnnotatedEvent.event =
      some (.resultExact owner rawTerms summary)) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  refine ⟨outputAt, ?_⟩
  exact exactValueClaim_add_of_mod_zero 100418593683253592432016548326729029359133068138294319235841 witness.env
    (SemanticRightRootResult788.actual selector witness)
    (SemanticRightRootResult571.actual selector witness) left right output
    (by simpa [left, leftRaw, SemanticRightRootResult788.summary] using leftClaim.claim)
    (by simpa [right, rightRaw, SemanticRightRootResult571.summary] using rightClaim.claim)
    (resultSound witness.env) (by decide)

theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply claimOfHistory selector witness (by rfl) (by rfl) (by rfl) (by rfl)
  · exact SemanticRightRootResult788.claimSound
      selector selectorLower selectorUpper witness
  · exact SemanticRightRootResult571.claimSound
      selector selectorLower selectorUpper witness
  · rfl

end SemanticRightRootResult792

namespace SemanticRightRootResult796

def leftRaw : List Term := SemanticRightRootResult792.rawTerms
def rightRaw : List Term := SemanticRightRootResult561.rawTerms
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6383⟩⟩, ⟨.program ⟨214⟩, ⟨18063⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6387⟩⟩, ⟨.program ⟨214⟩, ⟨17181⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6391⟩⟩, ⟨.program ⟨214⟩, ⟨17237⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6398⟩⟩, ⟨.program ⟨214⟩, ⟨17454⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6407⟩⟩, ⟨.program ⟨214⟩, ⟨17846⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6427⟩⟩, ⟨.program ⟨214⟩, ⟨15536⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6437⟩⟩, ⟨.program ⟨214⟩, ⟨16943⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6449⟩⟩, ⟨.program ⟨214⟩, ⟨17510⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6452⟩⟩, ⟨.program ⟨214⟩, ⟨15228⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6459⟩⟩, ⟨.program ⟨214⟩, ⟨17734⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6467⟩⟩, ⟨.program ⟨214⟩, ⟨17965⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6473⟩⟩, ⟨.program ⟨214⟩, ⟨17566⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨15067⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6490⟩⟩, ⟨.program ⟨214⟩, ⟨18893⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6494⟩⟩, ⟨.program ⟨214⟩, ⟨17622⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨14906⟩⟩], orderedFactors := [] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6502⟩⟩, ⟨.program ⟨214⟩, ⟨17678⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def owner : Owner := ⟨.program ⟨214⟩, ⟨18900⟩⟩
def rawTerms : List Term := outputRaw
def summary : Bound := .exactZero
def resultEvent : Nat := 796
theorem resultAgreement : CanonicalAgreement output (add left right) := by
  decide +kernel

theorem resultSound (env : Env Owner) :
    evalPolynomial env output = evalPolynomial env left + evalPolynomial env right := by
  exact addCanonicalResultSound env left right output resultAgreement

def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  SemanticRightRootResult792.actual selector witness +
    SemanticRightRootResult561.actual selector witness
theorem claimOfHistory (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841)
    (expressionAt : document.expressions.lookup 18900 =
      some Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Expression073.ExpressionRow18900)
    (leftPredecessorAt : (history.lookup 793).map AnnotatedEvent.event =
      some (.predecessor owner 0 ⟨18899⟩ 792))
    (rightPredecessorAt : (history.lookup 794).map AnnotatedEvent.event =
      some (.predecessor owner 1 ⟨16944⟩ 561))
    (ruleAt : (history.lookup 795).map AnnotatedEvent.event =
      some (.boundTransfer owner (.sum [.predecessor 0 793 .coefficient, .predecessor 1 794 .coefficient])))
    (leftClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult792.resultEvent
      SemanticRightRootResult792.owner
      (SemanticRightRootResult792.actual selector witness)
      SemanticRightRootResult792.rawTerms
      SemanticRightRootResult792.summary)
    (rightClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult561.resultEvent
      SemanticRightRootResult561.owner
      (SemanticRightRootResult561.actual selector witness)
      SemanticRightRootResult561.rawTerms
      SemanticRightRootResult561.summary)
    (outputAt : (history.lookup resultEvent).map AnnotatedEvent.event =
      some (.resultExact owner rawTerms summary)) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  refine ⟨outputAt, ?_⟩
  exact exactValueClaim_add_of_mod_zero 100418593683253592432016548326729029359133068138294319235841 witness.env
    (SemanticRightRootResult792.actual selector witness)
    (SemanticRightRootResult561.actual selector witness) left right output
    (by simpa [left, leftRaw, SemanticRightRootResult792.summary] using leftClaim.claim)
    (by simpa [right, rightRaw, SemanticRightRootResult561.summary] using rightClaim.claim)
    (resultSound witness.env) (by decide)

theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply claimOfHistory selector witness (by rfl) (by rfl) (by rfl) (by rfl)
  · exact SemanticRightRootResult792.claimSound
      selector selectorLower selectorUpper witness
  · exact SemanticRightRootResult561.claimSound
      selector selectorLower selectorUpper witness
  · rfl

end SemanticRightRootResult796

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
