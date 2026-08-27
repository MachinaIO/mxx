import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticRightRootShard030

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open TallSecurity0ABI
open TallSemantics

namespace SemanticRightRootResult5822

def owner : Owner := ⟨.program ⟨214⟩, ⟨6712⟩⟩
def rawTerms : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6712⟩⟩] } }]
def summary : Bound := .exactZero
def producerEvent : Nat := 5821
def resultEvent : Nat := 5822
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

end SemanticRightRootResult5822

namespace SemanticRightRootResult5827

def leftRaw : List Term := SemanticRightRootResult5822.rawTerms
def rightRaw : List Term := SemanticRightRootResult5819.rawTerms
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6712⟩⟩, ⟨.program ⟨214⟩, ⟨6663⟩⟩] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def owner : Owner := ⟨.program ⟨214⟩, ⟨7633⟩⟩
def rawTerms : List Term := outputRaw
def summary : Bound := .exactZero
def resultEvent : Nat := 5827
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
  SemanticRightRootResult5822.actual selector witness *
    SemanticRightRootResult5819.actual selector witness
theorem claimOfHistory (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841)
    (expressionAt : document.expressions.lookup 7633 =
      some Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Expression029.ExpressionRow7633)
    (leftPredecessorAt : (history.lookup 5823).map AnnotatedEvent.event =
      some (.predecessor owner 0 ⟨6712⟩ 5822))
    (rightPredecessorAt : (history.lookup 5824).map AnnotatedEvent.event =
      some (.predecessor owner 1 ⟨6664⟩ 5819))
    (ruleAt : (history.lookup 5825).map AnnotatedEvent.event =
      some (.boundTransfer owner (.product (.predecessor 0 5823 .coefficient) (.predecessor 1 5824 .coefficient) ⟨false, false, none, none, none⟩)))
    (leftClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult5822.resultEvent
      SemanticRightRootResult5822.owner
      (SemanticRightRootResult5822.actual selector witness)
      SemanticRightRootResult5822.rawTerms
      SemanticRightRootResult5822.summary)
    (rightClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult5819.resultEvent
      SemanticRightRootResult5819.owner
      (SemanticRightRootResult5819.actual selector witness)
      SemanticRightRootResult5819.rawTerms
      SemanticRightRootResult5819.summary)
    (outputAt : (history.lookup resultEvent).map AnnotatedEvent.event =
      some (.resultExact owner rawTerms summary)) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  refine ⟨outputAt, ?_⟩
  exact exactValueClaim_product_of_mod_zero 100418593683253592432016548326729029359133068138294319235841 witness.env
    (SemanticRightRootResult5822.actual selector witness)
    (SemanticRightRootResult5819.actual selector witness) left right output
    (by simpa [left, leftRaw, SemanticRightRootResult5822.summary] using leftClaim.claim)
    (by simpa [right, rightRaw, SemanticRightRootResult5819.summary] using rightClaim.claim)
    (resultSound witness.env) (by decide)

theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply claimOfHistory selector witness (by rfl) (by rfl) (by rfl) (by rfl)
  · exact SemanticRightRootResult5822.claimSound
      selector selectorLower selectorUpper witness
  · exact SemanticRightRootResult5819.claimSound
      selector selectorLower selectorUpper witness
  · rfl

end SemanticRightRootResult5827

namespace SemanticRightRootResult5839

def owner : Owner := ⟨.program ⟨214⟩, ⟨6672⟩⟩
def rawTerms : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6671⟩⟩] } }]
def summary : Bound := .exactZero
def producerEvent : Nat := 5838
def resultEvent : Nat := 5839
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual resultEvent
theorem terminalAt (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, rightRootOwnerAtSelector214 selector selectorLower selectorUpper _, ?_⟩
  refine ⟨.scale (.predecessor 0 5836 .coefficient) (.value (.predecessor 1 5837 .coefficient)), 0, .scale (.predecessor 0 5836 .coefficient) (.value (.predecessor 1 5837 .coefficient)), ?_, ?_⟩
  · rfl
  · rfl
theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness
    (terminalAt selector selectorLower selectorUpper)

end SemanticRightRootResult5839

namespace SemanticRightRootResult5842

def owner : Owner := ⟨.program ⟨214⟩, ⟨6710⟩⟩
def rawTerms : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6710⟩⟩] } }]
def summary : Bound := .exactZero
def producerEvent : Nat := 5841
def resultEvent : Nat := 5842
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

end SemanticRightRootResult5842

namespace SemanticRightRootResult5847

def leftRaw : List Term := SemanticRightRootResult5842.rawTerms
def rightRaw : List Term := SemanticRightRootResult5839.rawTerms
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6710⟩⟩, ⟨.program ⟨214⟩, ⟨6671⟩⟩] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def owner : Owner := ⟨.program ⟨214⟩, ⟨7632⟩⟩
def rawTerms : List Term := outputRaw
def summary : Bound := .exactZero
def resultEvent : Nat := 5847
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
  SemanticRightRootResult5842.actual selector witness *
    SemanticRightRootResult5839.actual selector witness
theorem claimOfHistory (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841)
    (expressionAt : document.expressions.lookup 7632 =
      some Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Expression029.ExpressionRow7632)
    (leftPredecessorAt : (history.lookup 5843).map AnnotatedEvent.event =
      some (.predecessor owner 0 ⟨6710⟩ 5842))
    (rightPredecessorAt : (history.lookup 5844).map AnnotatedEvent.event =
      some (.predecessor owner 1 ⟨6672⟩ 5839))
    (ruleAt : (history.lookup 5845).map AnnotatedEvent.event =
      some (.boundTransfer owner (.product (.predecessor 0 5843 .coefficient) (.predecessor 1 5844 .coefficient) ⟨false, false, none, none, none⟩)))
    (leftClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult5842.resultEvent
      SemanticRightRootResult5842.owner
      (SemanticRightRootResult5842.actual selector witness)
      SemanticRightRootResult5842.rawTerms
      SemanticRightRootResult5842.summary)
    (rightClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult5839.resultEvent
      SemanticRightRootResult5839.owner
      (SemanticRightRootResult5839.actual selector witness)
      SemanticRightRootResult5839.rawTerms
      SemanticRightRootResult5839.summary)
    (outputAt : (history.lookup resultEvent).map AnnotatedEvent.event =
      some (.resultExact owner rawTerms summary)) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  refine ⟨outputAt, ?_⟩
  exact exactValueClaim_product_of_mod_zero 100418593683253592432016548326729029359133068138294319235841 witness.env
    (SemanticRightRootResult5842.actual selector witness)
    (SemanticRightRootResult5839.actual selector witness) left right output
    (by simpa [left, leftRaw, SemanticRightRootResult5842.summary] using leftClaim.claim)
    (by simpa [right, rightRaw, SemanticRightRootResult5839.summary] using rightClaim.claim)
    (resultSound witness.env) (by decide)

theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply claimOfHistory selector witness (by rfl) (by rfl) (by rfl) (by rfl)
  · exact SemanticRightRootResult5842.claimSound
      selector selectorLower selectorUpper witness
  · exact SemanticRightRootResult5839.claimSound
      selector selectorLower selectorUpper witness
  · rfl

end SemanticRightRootResult5847

namespace SemanticRightRootResult5859

def owner : Owner := ⟨.program ⟨214⟩, ⟨6680⟩⟩
def rawTerms : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6679⟩⟩] } }]
def summary : Bound := .exactZero
def producerEvent : Nat := 5858
def resultEvent : Nat := 5859
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual resultEvent
theorem terminalAt (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, rightRootOwnerAtSelector214 selector selectorLower selectorUpper _, ?_⟩
  refine ⟨.scale (.predecessor 0 5856 .coefficient) (.value (.predecessor 1 5857 .coefficient)), 0, .scale (.predecessor 0 5856 .coefficient) (.value (.predecessor 1 5857 .coefficient)), ?_, ?_⟩
  · rfl
  · rfl
theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness
    (terminalAt selector selectorLower selectorUpper)

end SemanticRightRootResult5859

namespace SemanticRightRootResult5862

def owner : Owner := ⟨.program ⟨214⟩, ⟨6708⟩⟩
def rawTerms : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6708⟩⟩] } }]
def summary : Bound := .exactZero
def producerEvent : Nat := 5861
def resultEvent : Nat := 5862
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

end SemanticRightRootResult5862

namespace SemanticRightRootResult5867

def leftRaw : List Term := SemanticRightRootResult5862.rawTerms
def rightRaw : List Term := SemanticRightRootResult5859.rawTerms
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6708⟩⟩, ⟨.program ⟨214⟩, ⟨6679⟩⟩] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def owner : Owner := ⟨.program ⟨214⟩, ⟨7631⟩⟩
def rawTerms : List Term := outputRaw
def summary : Bound := .exactZero
def resultEvent : Nat := 5867
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
  SemanticRightRootResult5862.actual selector witness *
    SemanticRightRootResult5859.actual selector witness
theorem claimOfHistory (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841)
    (expressionAt : document.expressions.lookup 7631 =
      some Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Expression029.ExpressionRow7631)
    (leftPredecessorAt : (history.lookup 5863).map AnnotatedEvent.event =
      some (.predecessor owner 0 ⟨6708⟩ 5862))
    (rightPredecessorAt : (history.lookup 5864).map AnnotatedEvent.event =
      some (.predecessor owner 1 ⟨6680⟩ 5859))
    (ruleAt : (history.lookup 5865).map AnnotatedEvent.event =
      some (.boundTransfer owner (.product (.predecessor 0 5863 .coefficient) (.predecessor 1 5864 .coefficient) ⟨false, false, none, none, none⟩)))
    (leftClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult5862.resultEvent
      SemanticRightRootResult5862.owner
      (SemanticRightRootResult5862.actual selector witness)
      SemanticRightRootResult5862.rawTerms
      SemanticRightRootResult5862.summary)
    (rightClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult5859.resultEvent
      SemanticRightRootResult5859.owner
      (SemanticRightRootResult5859.actual selector witness)
      SemanticRightRootResult5859.rawTerms
      SemanticRightRootResult5859.summary)
    (outputAt : (history.lookup resultEvent).map AnnotatedEvent.event =
      some (.resultExact owner rawTerms summary)) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  refine ⟨outputAt, ?_⟩
  exact exactValueClaim_product_of_mod_zero 100418593683253592432016548326729029359133068138294319235841 witness.env
    (SemanticRightRootResult5862.actual selector witness)
    (SemanticRightRootResult5859.actual selector witness) left right output
    (by simpa [left, leftRaw, SemanticRightRootResult5862.summary] using leftClaim.claim)
    (by simpa [right, rightRaw, SemanticRightRootResult5859.summary] using rightClaim.claim)
    (resultSound witness.env) (by decide)

theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply claimOfHistory selector witness (by rfl) (by rfl) (by rfl) (by rfl)
  · exact SemanticRightRootResult5862.claimSound
      selector selectorLower selectorUpper witness
  · exact SemanticRightRootResult5859.claimSound
      selector selectorLower selectorUpper witness
  · rfl

end SemanticRightRootResult5867

namespace SemanticRightRootResult5873

def owner : Owner := ⟨.program ⟨214⟩, ⟨6760⟩⟩
def rawTerms : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6760⟩⟩] } }]
def summary : Bound := .exactZero
def producerEvent : Nat := 5872
def resultEvent : Nat := 5873
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual resultEvent
theorem terminalAt (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, rightRootOwnerAtSelector214 selector selectorLower selectorUpper _, ?_⟩
  refine ⟨.identity (.predecessor 0 5871 .coefficient), 0, .identity (.predecessor 0 5871 .coefficient), ?_, ?_⟩
  · rfl
  · rfl
theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness
    (terminalAt selector selectorLower selectorUpper)

end SemanticRightRootResult5873

namespace SemanticRightRootResult5878

def leftRaw : List Term := SemanticRightRootResult5873.rawTerms
def rightRaw : List Term := SemanticRightRootResult5873.rawTerms
def outputRaw : List Term := []
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def owner : Owner := ⟨.program ⟨214⟩, ⟨7650⟩⟩
def rawTerms : List Term := outputRaw
def summary : Bound := .exactZero
def resultEvent : Nat := 5878
theorem resultAgreement : CanonicalAgreement output (subtract left right) := by
  decide +kernel

theorem resultSound (env : Env Owner) :
    evalPolynomial env output = evalPolynomial env left - evalPolynomial env right := by
  exact subCanonicalResultSound env left right output resultAgreement

def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  SemanticRightRootResult5873.actual selector witness -
    SemanticRightRootResult5873.actual selector witness
theorem claimOfHistory (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841)
    (expressionAt : document.expressions.lookup 7650 =
      some Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Expression029.ExpressionRow7650)
    (leftPredecessorAt : (history.lookup 5874).map AnnotatedEvent.event =
      some (.predecessor owner 0 ⟨6760⟩ 5873))
    (rightPredecessorAt : (history.lookup 5875).map AnnotatedEvent.event =
      some (.predecessor owner 1 ⟨6760⟩ 5873))
    (ruleAt : (history.lookup 5876).map AnnotatedEvent.event =
      some (.boundTransfer owner (.sum [.predecessor 0 5874 .coefficient, .predecessor 1 5875 .coefficient])))
    (leftClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult5873.resultEvent
      SemanticRightRootResult5873.owner
      (SemanticRightRootResult5873.actual selector witness)
      SemanticRightRootResult5873.rawTerms
      SemanticRightRootResult5873.summary)
    (rightClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult5873.resultEvent
      SemanticRightRootResult5873.owner
      (SemanticRightRootResult5873.actual selector witness)
      SemanticRightRootResult5873.rawTerms
      SemanticRightRootResult5873.summary)
    (outputAt : (history.lookup resultEvent).map AnnotatedEvent.event =
      some (.resultExact owner rawTerms summary)) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  refine ⟨outputAt, ?_⟩
  exact exactValueClaim_sub_exactZero_of_mod_zero 100418593683253592432016548326729029359133068138294319235841 witness.env
    (SemanticRightRootResult5873.actual selector witness)
    (SemanticRightRootResult5873.actual selector witness) left right output
    (by simpa [left, leftRaw, SemanticRightRootResult5873.summary] using leftClaim.claim)
    (by simpa [right, rightRaw, SemanticRightRootResult5873.summary] using rightClaim.claim)
    (resultSound witness.env) (by decide)

theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply claimOfHistory selector witness (by rfl) (by rfl) (by rfl) (by rfl)
  · exact SemanticRightRootResult5873.claimSound
      selector selectorLower selectorUpper witness
  · exact SemanticRightRootResult5873.claimSound
      selector selectorLower selectorUpper witness
  · rfl

end SemanticRightRootResult5878

namespace SemanticRightRootResult5882

def leftRaw : List Term := SemanticRightRootResult5878.rawTerms
def rightRaw : List Term := SemanticRightRootResult5867.rawTerms
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6708⟩⟩, ⟨.program ⟨214⟩, ⟨6679⟩⟩] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def owner : Owner := ⟨.program ⟨214⟩, ⟨7651⟩⟩
def rawTerms : List Term := outputRaw
def summary : Bound := .exactZero
def resultEvent : Nat := 5882
theorem resultAgreement : CanonicalAgreement output (add left right) := by
  decide +kernel

theorem resultSound (env : Env Owner) :
    evalPolynomial env output = evalPolynomial env left + evalPolynomial env right := by
  exact addCanonicalResultSound env left right output resultAgreement

def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  SemanticRightRootResult5878.actual selector witness +
    SemanticRightRootResult5867.actual selector witness
theorem claimOfHistory (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841)
    (expressionAt : document.expressions.lookup 7651 =
      some Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Expression029.ExpressionRow7651)
    (leftPredecessorAt : (history.lookup 5879).map AnnotatedEvent.event =
      some (.predecessor owner 0 ⟨7650⟩ 5878))
    (rightPredecessorAt : (history.lookup 5880).map AnnotatedEvent.event =
      some (.predecessor owner 1 ⟨7631⟩ 5867))
    (ruleAt : (history.lookup 5881).map AnnotatedEvent.event =
      some (.boundTransfer owner (.sum [.predecessor 0 5879 .coefficient, .predecessor 1 5880 .coefficient])))
    (leftClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult5878.resultEvent
      SemanticRightRootResult5878.owner
      (SemanticRightRootResult5878.actual selector witness)
      SemanticRightRootResult5878.rawTerms
      SemanticRightRootResult5878.summary)
    (rightClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult5867.resultEvent
      SemanticRightRootResult5867.owner
      (SemanticRightRootResult5867.actual selector witness)
      SemanticRightRootResult5867.rawTerms
      SemanticRightRootResult5867.summary)
    (outputAt : (history.lookup resultEvent).map AnnotatedEvent.event =
      some (.resultExact owner rawTerms summary)) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  refine ⟨outputAt, ?_⟩
  exact exactValueClaim_add_of_mod_zero 100418593683253592432016548326729029359133068138294319235841 witness.env
    (SemanticRightRootResult5878.actual selector witness)
    (SemanticRightRootResult5867.actual selector witness) left right output
    (by simpa [left, leftRaw, SemanticRightRootResult5878.summary] using leftClaim.claim)
    (by simpa [right, rightRaw, SemanticRightRootResult5867.summary] using rightClaim.claim)
    (resultSound witness.env) (by decide)

theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply claimOfHistory selector witness (by rfl) (by rfl) (by rfl) (by rfl)
  · exact SemanticRightRootResult5878.claimSound
      selector selectorLower selectorUpper witness
  · exact SemanticRightRootResult5867.claimSound
      selector selectorLower selectorUpper witness
  · rfl

end SemanticRightRootResult5882

namespace SemanticRightRootResult5886

def leftRaw : List Term := SemanticRightRootResult5882.rawTerms
def rightRaw : List Term := SemanticRightRootResult5847.rawTerms
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6708⟩⟩, ⟨.program ⟨214⟩, ⟨6679⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6710⟩⟩, ⟨.program ⟨214⟩, ⟨6671⟩⟩] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def owner : Owner := ⟨.program ⟨214⟩, ⟨7652⟩⟩
def rawTerms : List Term := outputRaw
def summary : Bound := .exactZero
def resultEvent : Nat := 5886
theorem resultAgreement : CanonicalAgreement output (add left right) := by
  decide +kernel

theorem resultSound (env : Env Owner) :
    evalPolynomial env output = evalPolynomial env left + evalPolynomial env right := by
  exact addCanonicalResultSound env left right output resultAgreement

def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  SemanticRightRootResult5882.actual selector witness +
    SemanticRightRootResult5847.actual selector witness
theorem claimOfHistory (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841)
    (expressionAt : document.expressions.lookup 7652 =
      some Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Expression029.ExpressionRow7652)
    (leftPredecessorAt : (history.lookup 5883).map AnnotatedEvent.event =
      some (.predecessor owner 0 ⟨7651⟩ 5882))
    (rightPredecessorAt : (history.lookup 5884).map AnnotatedEvent.event =
      some (.predecessor owner 1 ⟨7632⟩ 5847))
    (ruleAt : (history.lookup 5885).map AnnotatedEvent.event =
      some (.boundTransfer owner (.sum [.predecessor 0 5883 .coefficient, .predecessor 1 5884 .coefficient])))
    (leftClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult5882.resultEvent
      SemanticRightRootResult5882.owner
      (SemanticRightRootResult5882.actual selector witness)
      SemanticRightRootResult5882.rawTerms
      SemanticRightRootResult5882.summary)
    (rightClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult5847.resultEvent
      SemanticRightRootResult5847.owner
      (SemanticRightRootResult5847.actual selector witness)
      SemanticRightRootResult5847.rawTerms
      SemanticRightRootResult5847.summary)
    (outputAt : (history.lookup resultEvent).map AnnotatedEvent.event =
      some (.resultExact owner rawTerms summary)) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  refine ⟨outputAt, ?_⟩
  exact exactValueClaim_add_of_mod_zero 100418593683253592432016548326729029359133068138294319235841 witness.env
    (SemanticRightRootResult5882.actual selector witness)
    (SemanticRightRootResult5847.actual selector witness) left right output
    (by simpa [left, leftRaw, SemanticRightRootResult5882.summary] using leftClaim.claim)
    (by simpa [right, rightRaw, SemanticRightRootResult5847.summary] using rightClaim.claim)
    (resultSound witness.env) (by decide)

theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply claimOfHistory selector witness (by rfl) (by rfl) (by rfl) (by rfl)
  · exact SemanticRightRootResult5882.claimSound
      selector selectorLower selectorUpper witness
  · exact SemanticRightRootResult5847.claimSound
      selector selectorLower selectorUpper witness
  · rfl

end SemanticRightRootResult5886

namespace SemanticRightRootResult5890

def leftRaw : List Term := SemanticRightRootResult5886.rawTerms
def rightRaw : List Term := SemanticRightRootResult5827.rawTerms
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6708⟩⟩, ⟨.program ⟨214⟩, ⟨6679⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6710⟩⟩, ⟨.program ⟨214⟩, ⟨6671⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6712⟩⟩, ⟨.program ⟨214⟩, ⟨6663⟩⟩] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def owner : Owner := ⟨.program ⟨214⟩, ⟨7653⟩⟩
def rawTerms : List Term := outputRaw
def summary : Bound := .exactZero
def resultEvent : Nat := 5890
theorem resultAgreement : CanonicalAgreement output (add left right) := by
  decide +kernel

theorem resultSound (env : Env Owner) :
    evalPolynomial env output = evalPolynomial env left + evalPolynomial env right := by
  exact addCanonicalResultSound env left right output resultAgreement

def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  SemanticRightRootResult5886.actual selector witness +
    SemanticRightRootResult5827.actual selector witness
theorem claimOfHistory (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841)
    (expressionAt : document.expressions.lookup 7653 =
      some Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Expression029.ExpressionRow7653)
    (leftPredecessorAt : (history.lookup 5887).map AnnotatedEvent.event =
      some (.predecessor owner 0 ⟨7652⟩ 5886))
    (rightPredecessorAt : (history.lookup 5888).map AnnotatedEvent.event =
      some (.predecessor owner 1 ⟨7633⟩ 5827))
    (ruleAt : (history.lookup 5889).map AnnotatedEvent.event =
      some (.boundTransfer owner (.sum [.predecessor 0 5887 .coefficient, .predecessor 1 5888 .coefficient])))
    (leftClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult5886.resultEvent
      SemanticRightRootResult5886.owner
      (SemanticRightRootResult5886.actual selector witness)
      SemanticRightRootResult5886.rawTerms
      SemanticRightRootResult5886.summary)
    (rightClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult5827.resultEvent
      SemanticRightRootResult5827.owner
      (SemanticRightRootResult5827.actual selector witness)
      SemanticRightRootResult5827.rawTerms
      SemanticRightRootResult5827.summary)
    (outputAt : (history.lookup resultEvent).map AnnotatedEvent.event =
      some (.resultExact owner rawTerms summary)) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  refine ⟨outputAt, ?_⟩
  exact exactValueClaim_add_of_mod_zero 100418593683253592432016548326729029359133068138294319235841 witness.env
    (SemanticRightRootResult5886.actual selector witness)
    (SemanticRightRootResult5827.actual selector witness) left right output
    (by simpa [left, leftRaw, SemanticRightRootResult5886.summary] using leftClaim.claim)
    (by simpa [right, rightRaw, SemanticRightRootResult5827.summary] using rightClaim.claim)
    (resultSound witness.env) (by decide)

theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply claimOfHistory selector witness (by rfl) (by rfl) (by rfl) (by rfl)
  · exact SemanticRightRootResult5886.claimSound
      selector selectorLower selectorUpper witness
  · exact SemanticRightRootResult5827.claimSound
      selector selectorLower selectorUpper witness
  · rfl

end SemanticRightRootResult5890

namespace SemanticRightRootResult5894

def leftRaw : List Term := SemanticRightRootResult5890.rawTerms
def rightRaw : List Term := SemanticRightRootResult5807.rawTerms
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6708⟩⟩, ⟨.program ⟨214⟩, ⟨6679⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6710⟩⟩, ⟨.program ⟨214⟩, ⟨6671⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6712⟩⟩, ⟨.program ⟨214⟩, ⟨6663⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6714⟩⟩, ⟨.program ⟨214⟩, ⟨6655⟩⟩] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def owner : Owner := ⟨.program ⟨214⟩, ⟨7654⟩⟩
def rawTerms : List Term := outputRaw
def summary : Bound := .exactZero
def resultEvent : Nat := 5894
theorem resultAgreement : CanonicalAgreement output (add left right) := by
  decide +kernel

theorem resultSound (env : Env Owner) :
    evalPolynomial env output = evalPolynomial env left + evalPolynomial env right := by
  exact addCanonicalResultSound env left right output resultAgreement

def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  SemanticRightRootResult5890.actual selector witness +
    SemanticRightRootResult5807.actual selector witness
theorem claimOfHistory (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841)
    (expressionAt : document.expressions.lookup 7654 =
      some Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Expression029.ExpressionRow7654)
    (leftPredecessorAt : (history.lookup 5891).map AnnotatedEvent.event =
      some (.predecessor owner 0 ⟨7653⟩ 5890))
    (rightPredecessorAt : (history.lookup 5892).map AnnotatedEvent.event =
      some (.predecessor owner 1 ⟨7634⟩ 5807))
    (ruleAt : (history.lookup 5893).map AnnotatedEvent.event =
      some (.boundTransfer owner (.sum [.predecessor 0 5891 .coefficient, .predecessor 1 5892 .coefficient])))
    (leftClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult5890.resultEvent
      SemanticRightRootResult5890.owner
      (SemanticRightRootResult5890.actual selector witness)
      SemanticRightRootResult5890.rawTerms
      SemanticRightRootResult5890.summary)
    (rightClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult5807.resultEvent
      SemanticRightRootResult5807.owner
      (SemanticRightRootResult5807.actual selector witness)
      SemanticRightRootResult5807.rawTerms
      SemanticRightRootResult5807.summary)
    (outputAt : (history.lookup resultEvent).map AnnotatedEvent.event =
      some (.resultExact owner rawTerms summary)) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  refine ⟨outputAt, ?_⟩
  exact exactValueClaim_add_of_mod_zero 100418593683253592432016548326729029359133068138294319235841 witness.env
    (SemanticRightRootResult5890.actual selector witness)
    (SemanticRightRootResult5807.actual selector witness) left right output
    (by simpa [left, leftRaw, SemanticRightRootResult5890.summary] using leftClaim.claim)
    (by simpa [right, rightRaw, SemanticRightRootResult5807.summary] using rightClaim.claim)
    (resultSound witness.env) (by decide)

theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply claimOfHistory selector witness (by rfl) (by rfl) (by rfl) (by rfl)
  · exact SemanticRightRootResult5890.claimSound
      selector selectorLower selectorUpper witness
  · exact SemanticRightRootResult5807.claimSound
      selector selectorLower selectorUpper witness
  · rfl

end SemanticRightRootResult5894

namespace SemanticRightRootResult5898

def leftRaw : List Term := SemanticRightRootResult5894.rawTerms
def rightRaw : List Term := SemanticRightRootResult5787.rawTerms
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6708⟩⟩, ⟨.program ⟨214⟩, ⟨6679⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6710⟩⟩, ⟨.program ⟨214⟩, ⟨6671⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6712⟩⟩, ⟨.program ⟨214⟩, ⟨6663⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6714⟩⟩, ⟨.program ⟨214⟩, ⟨6655⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6716⟩⟩, ⟨.program ⟨214⟩, ⟨6649⟩⟩] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def owner : Owner := ⟨.program ⟨214⟩, ⟨7655⟩⟩
def rawTerms : List Term := outputRaw
def summary : Bound := .exactZero
def resultEvent : Nat := 5898
theorem resultAgreement : CanonicalAgreement output (add left right) := by
  decide +kernel

theorem resultSound (env : Env Owner) :
    evalPolynomial env output = evalPolynomial env left + evalPolynomial env right := by
  exact addCanonicalResultSound env left right output resultAgreement

def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  SemanticRightRootResult5894.actual selector witness +
    SemanticRightRootResult5787.actual selector witness
theorem claimOfHistory (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841)
    (expressionAt : document.expressions.lookup 7655 =
      some Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Expression029.ExpressionRow7655)
    (leftPredecessorAt : (history.lookup 5895).map AnnotatedEvent.event =
      some (.predecessor owner 0 ⟨7654⟩ 5894))
    (rightPredecessorAt : (history.lookup 5896).map AnnotatedEvent.event =
      some (.predecessor owner 1 ⟨7635⟩ 5787))
    (ruleAt : (history.lookup 5897).map AnnotatedEvent.event =
      some (.boundTransfer owner (.sum [.predecessor 0 5895 .coefficient, .predecessor 1 5896 .coefficient])))
    (leftClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult5894.resultEvent
      SemanticRightRootResult5894.owner
      (SemanticRightRootResult5894.actual selector witness)
      SemanticRightRootResult5894.rawTerms
      SemanticRightRootResult5894.summary)
    (rightClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult5787.resultEvent
      SemanticRightRootResult5787.owner
      (SemanticRightRootResult5787.actual selector witness)
      SemanticRightRootResult5787.rawTerms
      SemanticRightRootResult5787.summary)
    (outputAt : (history.lookup resultEvent).map AnnotatedEvent.event =
      some (.resultExact owner rawTerms summary)) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  refine ⟨outputAt, ?_⟩
  exact exactValueClaim_add_of_mod_zero 100418593683253592432016548326729029359133068138294319235841 witness.env
    (SemanticRightRootResult5894.actual selector witness)
    (SemanticRightRootResult5787.actual selector witness) left right output
    (by simpa [left, leftRaw, SemanticRightRootResult5894.summary] using leftClaim.claim)
    (by simpa [right, rightRaw, SemanticRightRootResult5787.summary] using rightClaim.claim)
    (resultSound witness.env) (by decide)

theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply claimOfHistory selector witness (by rfl) (by rfl) (by rfl) (by rfl)
  · exact SemanticRightRootResult5894.claimSound
      selector selectorLower selectorUpper witness
  · exact SemanticRightRootResult5787.claimSound
      selector selectorLower selectorUpper witness
  · rfl

end SemanticRightRootResult5898

namespace SemanticRightRootResult5902

def leftRaw : List Term := SemanticRightRootResult5898.rawTerms
def rightRaw : List Term := SemanticRightRootResult5767.rawTerms
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6708⟩⟩, ⟨.program ⟨214⟩, ⟨6679⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6710⟩⟩, ⟨.program ⟨214⟩, ⟨6671⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6712⟩⟩, ⟨.program ⟨214⟩, ⟨6663⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6714⟩⟩, ⟨.program ⟨214⟩, ⟨6655⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6716⟩⟩, ⟨.program ⟨214⟩, ⟨6649⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6718⟩⟩, ⟨.program ⟨214⟩, ⟨6647⟩⟩] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def owner : Owner := ⟨.program ⟨214⟩, ⟨7656⟩⟩
def rawTerms : List Term := outputRaw
def summary : Bound := .exactZero
def resultEvent : Nat := 5902
theorem resultAgreement : CanonicalAgreement output (add left right) := by
  decide +kernel

theorem resultSound (env : Env Owner) :
    evalPolynomial env output = evalPolynomial env left + evalPolynomial env right := by
  exact addCanonicalResultSound env left right output resultAgreement

def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  SemanticRightRootResult5898.actual selector witness +
    SemanticRightRootResult5767.actual selector witness
theorem claimOfHistory (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841)
    (expressionAt : document.expressions.lookup 7656 =
      some Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Expression029.ExpressionRow7656)
    (leftPredecessorAt : (history.lookup 5899).map AnnotatedEvent.event =
      some (.predecessor owner 0 ⟨7655⟩ 5898))
    (rightPredecessorAt : (history.lookup 5900).map AnnotatedEvent.event =
      some (.predecessor owner 1 ⟨7636⟩ 5767))
    (ruleAt : (history.lookup 5901).map AnnotatedEvent.event =
      some (.boundTransfer owner (.sum [.predecessor 0 5899 .coefficient, .predecessor 1 5900 .coefficient])))
    (leftClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult5898.resultEvent
      SemanticRightRootResult5898.owner
      (SemanticRightRootResult5898.actual selector witness)
      SemanticRightRootResult5898.rawTerms
      SemanticRightRootResult5898.summary)
    (rightClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult5767.resultEvent
      SemanticRightRootResult5767.owner
      (SemanticRightRootResult5767.actual selector witness)
      SemanticRightRootResult5767.rawTerms
      SemanticRightRootResult5767.summary)
    (outputAt : (history.lookup resultEvent).map AnnotatedEvent.event =
      some (.resultExact owner rawTerms summary)) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  refine ⟨outputAt, ?_⟩
  exact exactValueClaim_add_of_mod_zero 100418593683253592432016548326729029359133068138294319235841 witness.env
    (SemanticRightRootResult5898.actual selector witness)
    (SemanticRightRootResult5767.actual selector witness) left right output
    (by simpa [left, leftRaw, SemanticRightRootResult5898.summary] using leftClaim.claim)
    (by simpa [right, rightRaw, SemanticRightRootResult5767.summary] using rightClaim.claim)
    (resultSound witness.env) (by decide)

theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply claimOfHistory selector witness (by rfl) (by rfl) (by rfl) (by rfl)
  · exact SemanticRightRootResult5898.claimSound
      selector selectorLower selectorUpper witness
  · exact SemanticRightRootResult5767.claimSound
      selector selectorLower selectorUpper witness
  · rfl

end SemanticRightRootResult5902

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
