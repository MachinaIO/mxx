import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticRightRootShard031

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open TallSecurity0ABI
open TallSemantics

namespace SemanticRightRootResult5906

def leftRaw : List Term := SemanticRightRootResult5902.rawTerms
def rightRaw : List Term := SemanticRightRootResult5747.rawTerms
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6708⟩⟩, ⟨.program ⟨214⟩, ⟨6679⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6710⟩⟩, ⟨.program ⟨214⟩, ⟨6671⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6712⟩⟩, ⟨.program ⟨214⟩, ⟨6663⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6714⟩⟩, ⟨.program ⟨214⟩, ⟨6655⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6716⟩⟩, ⟨.program ⟨214⟩, ⟨6649⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6718⟩⟩, ⟨.program ⟨214⟩, ⟨6647⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6720⟩⟩, ⟨.program ⟨214⟩, ⟨6643⟩⟩] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def owner : Owner := ⟨.program ⟨214⟩, ⟨7657⟩⟩
def rawTerms : List Term := outputRaw
def summary : Bound := .exactZero
def resultEvent : Nat := 5906
theorem resultAgreement : CanonicalAgreement output (add left right) := by
  decide +kernel

theorem resultSound (env : Env Owner) :
    evalPolynomial env output = evalPolynomial env left + evalPolynomial env right := by
  exact addCanonicalResultSound env left right output resultAgreement

def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  SemanticRightRootResult5902.actual selector witness +
    SemanticRightRootResult5747.actual selector witness
theorem claimOfHistory (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841)
    (expressionAt : document.expressions.lookup 7657 =
      some Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Expression029.ExpressionRow7657)
    (leftPredecessorAt : (history.lookup 5903).map AnnotatedEvent.event =
      some (.predecessor owner 0 ⟨7656⟩ 5902))
    (rightPredecessorAt : (history.lookup 5904).map AnnotatedEvent.event =
      some (.predecessor owner 1 ⟨7637⟩ 5747))
    (ruleAt : (history.lookup 5905).map AnnotatedEvent.event =
      some (.boundTransfer owner (.sum [.predecessor 0 5903 .coefficient, .predecessor 1 5904 .coefficient])))
    (leftClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult5902.resultEvent
      SemanticRightRootResult5902.owner
      (SemanticRightRootResult5902.actual selector witness)
      SemanticRightRootResult5902.rawTerms
      SemanticRightRootResult5902.summary)
    (rightClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult5747.resultEvent
      SemanticRightRootResult5747.owner
      (SemanticRightRootResult5747.actual selector witness)
      SemanticRightRootResult5747.rawTerms
      SemanticRightRootResult5747.summary)
    (outputAt : (history.lookup resultEvent).map AnnotatedEvent.event =
      some (.resultExact owner rawTerms summary)) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  refine ⟨outputAt, ?_⟩
  exact exactValueClaim_add_of_mod_zero 100418593683253592432016548326729029359133068138294319235841 witness.env
    (SemanticRightRootResult5902.actual selector witness)
    (SemanticRightRootResult5747.actual selector witness) left right output
    (by simpa [left, leftRaw, SemanticRightRootResult5902.summary] using leftClaim.claim)
    (by simpa [right, rightRaw, SemanticRightRootResult5747.summary] using rightClaim.claim)
    (resultSound witness.env) (by decide)

theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply claimOfHistory selector witness (by rfl) (by rfl) (by rfl) (by rfl)
  · exact SemanticRightRootResult5902.claimSound
      selector selectorLower selectorUpper witness
  · exact SemanticRightRootResult5747.claimSound
      selector selectorLower selectorUpper witness
  · rfl

end SemanticRightRootResult5906

namespace SemanticRightRootResult5910

def leftRaw : List Term := SemanticRightRootResult5906.rawTerms
def rightRaw : List Term := SemanticRightRootResult5727.rawTerms
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6708⟩⟩, ⟨.program ⟨214⟩, ⟨6679⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6710⟩⟩, ⟨.program ⟨214⟩, ⟨6671⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6712⟩⟩, ⟨.program ⟨214⟩, ⟨6663⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6714⟩⟩, ⟨.program ⟨214⟩, ⟨6655⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6716⟩⟩, ⟨.program ⟨214⟩, ⟨6649⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6718⟩⟩, ⟨.program ⟨214⟩, ⟨6647⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6720⟩⟩, ⟨.program ⟨214⟩, ⟨6643⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6722⟩⟩, ⟨.program ⟨214⟩, ⟨6641⟩⟩] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def owner : Owner := ⟨.program ⟨214⟩, ⟨7658⟩⟩
def rawTerms : List Term := outputRaw
def summary : Bound := .exactZero
def resultEvent : Nat := 5910
theorem resultAgreement : CanonicalAgreement output (add left right) := by
  decide +kernel

theorem resultSound (env : Env Owner) :
    evalPolynomial env output = evalPolynomial env left + evalPolynomial env right := by
  exact addCanonicalResultSound env left right output resultAgreement

def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  SemanticRightRootResult5906.actual selector witness +
    SemanticRightRootResult5727.actual selector witness
theorem claimOfHistory (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841)
    (expressionAt : document.expressions.lookup 7658 =
      some Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Expression029.ExpressionRow7658)
    (leftPredecessorAt : (history.lookup 5907).map AnnotatedEvent.event =
      some (.predecessor owner 0 ⟨7657⟩ 5906))
    (rightPredecessorAt : (history.lookup 5908).map AnnotatedEvent.event =
      some (.predecessor owner 1 ⟨7638⟩ 5727))
    (ruleAt : (history.lookup 5909).map AnnotatedEvent.event =
      some (.boundTransfer owner (.sum [.predecessor 0 5907 .coefficient, .predecessor 1 5908 .coefficient])))
    (leftClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult5906.resultEvent
      SemanticRightRootResult5906.owner
      (SemanticRightRootResult5906.actual selector witness)
      SemanticRightRootResult5906.rawTerms
      SemanticRightRootResult5906.summary)
    (rightClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult5727.resultEvent
      SemanticRightRootResult5727.owner
      (SemanticRightRootResult5727.actual selector witness)
      SemanticRightRootResult5727.rawTerms
      SemanticRightRootResult5727.summary)
    (outputAt : (history.lookup resultEvent).map AnnotatedEvent.event =
      some (.resultExact owner rawTerms summary)) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  refine ⟨outputAt, ?_⟩
  exact exactValueClaim_add_of_mod_zero 100418593683253592432016548326729029359133068138294319235841 witness.env
    (SemanticRightRootResult5906.actual selector witness)
    (SemanticRightRootResult5727.actual selector witness) left right output
    (by simpa [left, leftRaw, SemanticRightRootResult5906.summary] using leftClaim.claim)
    (by simpa [right, rightRaw, SemanticRightRootResult5727.summary] using rightClaim.claim)
    (resultSound witness.env) (by decide)

theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply claimOfHistory selector witness (by rfl) (by rfl) (by rfl) (by rfl)
  · exact SemanticRightRootResult5906.claimSound
      selector selectorLower selectorUpper witness
  · exact SemanticRightRootResult5727.claimSound
      selector selectorLower selectorUpper witness
  · rfl

end SemanticRightRootResult5910

namespace SemanticRightRootResult5914

def leftRaw : List Term := SemanticRightRootResult5910.rawTerms
def rightRaw : List Term := SemanticRightRootResult5707.rawTerms
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6708⟩⟩, ⟨.program ⟨214⟩, ⟨6679⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6710⟩⟩, ⟨.program ⟨214⟩, ⟨6671⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6712⟩⟩, ⟨.program ⟨214⟩, ⟨6663⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6714⟩⟩, ⟨.program ⟨214⟩, ⟨6655⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6716⟩⟩, ⟨.program ⟨214⟩, ⟨6649⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6718⟩⟩, ⟨.program ⟨214⟩, ⟨6647⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6720⟩⟩, ⟨.program ⟨214⟩, ⟨6643⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6722⟩⟩, ⟨.program ⟨214⟩, ⟨6641⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6724⟩⟩, ⟨.program ⟨214⟩, ⟨6637⟩⟩] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def owner : Owner := ⟨.program ⟨214⟩, ⟨7659⟩⟩
def rawTerms : List Term := outputRaw
def summary : Bound := .exactZero
def resultEvent : Nat := 5914
theorem resultAgreement : CanonicalAgreement output (add left right) := by
  decide +kernel

theorem resultSound (env : Env Owner) :
    evalPolynomial env output = evalPolynomial env left + evalPolynomial env right := by
  exact addCanonicalResultSound env left right output resultAgreement

def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  SemanticRightRootResult5910.actual selector witness +
    SemanticRightRootResult5707.actual selector witness
theorem claimOfHistory (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841)
    (expressionAt : document.expressions.lookup 7659 =
      some Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Expression029.ExpressionRow7659)
    (leftPredecessorAt : (history.lookup 5911).map AnnotatedEvent.event =
      some (.predecessor owner 0 ⟨7658⟩ 5910))
    (rightPredecessorAt : (history.lookup 5912).map AnnotatedEvent.event =
      some (.predecessor owner 1 ⟨7639⟩ 5707))
    (ruleAt : (history.lookup 5913).map AnnotatedEvent.event =
      some (.boundTransfer owner (.sum [.predecessor 0 5911 .coefficient, .predecessor 1 5912 .coefficient])))
    (leftClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult5910.resultEvent
      SemanticRightRootResult5910.owner
      (SemanticRightRootResult5910.actual selector witness)
      SemanticRightRootResult5910.rawTerms
      SemanticRightRootResult5910.summary)
    (rightClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult5707.resultEvent
      SemanticRightRootResult5707.owner
      (SemanticRightRootResult5707.actual selector witness)
      SemanticRightRootResult5707.rawTerms
      SemanticRightRootResult5707.summary)
    (outputAt : (history.lookup resultEvent).map AnnotatedEvent.event =
      some (.resultExact owner rawTerms summary)) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  refine ⟨outputAt, ?_⟩
  exact exactValueClaim_add_of_mod_zero 100418593683253592432016548326729029359133068138294319235841 witness.env
    (SemanticRightRootResult5910.actual selector witness)
    (SemanticRightRootResult5707.actual selector witness) left right output
    (by simpa [left, leftRaw, SemanticRightRootResult5910.summary] using leftClaim.claim)
    (by simpa [right, rightRaw, SemanticRightRootResult5707.summary] using rightClaim.claim)
    (resultSound witness.env) (by decide)

theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply claimOfHistory selector witness (by rfl) (by rfl) (by rfl) (by rfl)
  · exact SemanticRightRootResult5910.claimSound
      selector selectorLower selectorUpper witness
  · exact SemanticRightRootResult5707.claimSound
      selector selectorLower selectorUpper witness
  · rfl

end SemanticRightRootResult5914

namespace SemanticRightRootResult5918

def leftRaw : List Term := SemanticRightRootResult5914.rawTerms
def rightRaw : List Term := SemanticRightRootResult5687.rawTerms
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6708⟩⟩, ⟨.program ⟨214⟩, ⟨6679⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6710⟩⟩, ⟨.program ⟨214⟩, ⟨6671⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6712⟩⟩, ⟨.program ⟨214⟩, ⟨6663⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6714⟩⟩, ⟨.program ⟨214⟩, ⟨6655⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6716⟩⟩, ⟨.program ⟨214⟩, ⟨6649⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6718⟩⟩, ⟨.program ⟨214⟩, ⟨6647⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6720⟩⟩, ⟨.program ⟨214⟩, ⟨6643⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6722⟩⟩, ⟨.program ⟨214⟩, ⟨6641⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6724⟩⟩, ⟨.program ⟨214⟩, ⟨6637⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6726⟩⟩, ⟨.program ⟨214⟩, ⟨6681⟩⟩] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def owner : Owner := ⟨.program ⟨214⟩, ⟨7660⟩⟩
def rawTerms : List Term := outputRaw
def summary : Bound := .exactZero
def resultEvent : Nat := 5918
theorem resultAgreement : CanonicalAgreement output (add left right) := by
  decide +kernel

theorem resultSound (env : Env Owner) :
    evalPolynomial env output = evalPolynomial env left + evalPolynomial env right := by
  exact addCanonicalResultSound env left right output resultAgreement

def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  SemanticRightRootResult5914.actual selector witness +
    SemanticRightRootResult5687.actual selector witness
theorem claimOfHistory (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841)
    (expressionAt : document.expressions.lookup 7660 =
      some Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Expression029.ExpressionRow7660)
    (leftPredecessorAt : (history.lookup 5915).map AnnotatedEvent.event =
      some (.predecessor owner 0 ⟨7659⟩ 5914))
    (rightPredecessorAt : (history.lookup 5916).map AnnotatedEvent.event =
      some (.predecessor owner 1 ⟨7640⟩ 5687))
    (ruleAt : (history.lookup 5917).map AnnotatedEvent.event =
      some (.boundTransfer owner (.sum [.predecessor 0 5915 .coefficient, .predecessor 1 5916 .coefficient])))
    (leftClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult5914.resultEvent
      SemanticRightRootResult5914.owner
      (SemanticRightRootResult5914.actual selector witness)
      SemanticRightRootResult5914.rawTerms
      SemanticRightRootResult5914.summary)
    (rightClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult5687.resultEvent
      SemanticRightRootResult5687.owner
      (SemanticRightRootResult5687.actual selector witness)
      SemanticRightRootResult5687.rawTerms
      SemanticRightRootResult5687.summary)
    (outputAt : (history.lookup resultEvent).map AnnotatedEvent.event =
      some (.resultExact owner rawTerms summary)) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  refine ⟨outputAt, ?_⟩
  exact exactValueClaim_add_of_mod_zero 100418593683253592432016548326729029359133068138294319235841 witness.env
    (SemanticRightRootResult5914.actual selector witness)
    (SemanticRightRootResult5687.actual selector witness) left right output
    (by simpa [left, leftRaw, SemanticRightRootResult5914.summary] using leftClaim.claim)
    (by simpa [right, rightRaw, SemanticRightRootResult5687.summary] using rightClaim.claim)
    (resultSound witness.env) (by decide)

theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply claimOfHistory selector witness (by rfl) (by rfl) (by rfl) (by rfl)
  · exact SemanticRightRootResult5914.claimSound
      selector selectorLower selectorUpper witness
  · exact SemanticRightRootResult5687.claimSound
      selector selectorLower selectorUpper witness
  · rfl

end SemanticRightRootResult5918

namespace SemanticRightRootResult5922

def leftRaw : List Term := SemanticRightRootResult5918.rawTerms
def rightRaw : List Term := SemanticRightRootResult5667.rawTerms
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6708⟩⟩, ⟨.program ⟨214⟩, ⟨6679⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6710⟩⟩, ⟨.program ⟨214⟩, ⟨6671⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6712⟩⟩, ⟨.program ⟨214⟩, ⟨6663⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6714⟩⟩, ⟨.program ⟨214⟩, ⟨6655⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6716⟩⟩, ⟨.program ⟨214⟩, ⟨6649⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6718⟩⟩, ⟨.program ⟨214⟩, ⟨6647⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6720⟩⟩, ⟨.program ⟨214⟩, ⟨6643⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6722⟩⟩, ⟨.program ⟨214⟩, ⟨6641⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6724⟩⟩, ⟨.program ⟨214⟩, ⟨6637⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6726⟩⟩, ⟨.program ⟨214⟩, ⟨6681⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6728⟩⟩, ⟨.program ⟨214⟩, ⟨6677⟩⟩] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def owner : Owner := ⟨.program ⟨214⟩, ⟨7661⟩⟩
def rawTerms : List Term := outputRaw
def summary : Bound := .exactZero
def resultEvent : Nat := 5922
theorem resultAgreement : CanonicalAgreement output (add left right) := by
  decide +kernel

theorem resultSound (env : Env Owner) :
    evalPolynomial env output = evalPolynomial env left + evalPolynomial env right := by
  exact addCanonicalResultSound env left right output resultAgreement

def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  SemanticRightRootResult5918.actual selector witness +
    SemanticRightRootResult5667.actual selector witness
theorem claimOfHistory (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841)
    (expressionAt : document.expressions.lookup 7661 =
      some Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Expression029.ExpressionRow7661)
    (leftPredecessorAt : (history.lookup 5919).map AnnotatedEvent.event =
      some (.predecessor owner 0 ⟨7660⟩ 5918))
    (rightPredecessorAt : (history.lookup 5920).map AnnotatedEvent.event =
      some (.predecessor owner 1 ⟨7641⟩ 5667))
    (ruleAt : (history.lookup 5921).map AnnotatedEvent.event =
      some (.boundTransfer owner (.sum [.predecessor 0 5919 .coefficient, .predecessor 1 5920 .coefficient])))
    (leftClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult5918.resultEvent
      SemanticRightRootResult5918.owner
      (SemanticRightRootResult5918.actual selector witness)
      SemanticRightRootResult5918.rawTerms
      SemanticRightRootResult5918.summary)
    (rightClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult5667.resultEvent
      SemanticRightRootResult5667.owner
      (SemanticRightRootResult5667.actual selector witness)
      SemanticRightRootResult5667.rawTerms
      SemanticRightRootResult5667.summary)
    (outputAt : (history.lookup resultEvent).map AnnotatedEvent.event =
      some (.resultExact owner rawTerms summary)) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  refine ⟨outputAt, ?_⟩
  exact exactValueClaim_add_of_mod_zero 100418593683253592432016548326729029359133068138294319235841 witness.env
    (SemanticRightRootResult5918.actual selector witness)
    (SemanticRightRootResult5667.actual selector witness) left right output
    (by simpa [left, leftRaw, SemanticRightRootResult5918.summary] using leftClaim.claim)
    (by simpa [right, rightRaw, SemanticRightRootResult5667.summary] using rightClaim.claim)
    (resultSound witness.env) (by decide)

theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply claimOfHistory selector witness (by rfl) (by rfl) (by rfl) (by rfl)
  · exact SemanticRightRootResult5918.claimSound
      selector selectorLower selectorUpper witness
  · exact SemanticRightRootResult5667.claimSound
      selector selectorLower selectorUpper witness
  · rfl

end SemanticRightRootResult5922

namespace SemanticRightRootResult5926

def leftRaw : List Term := SemanticRightRootResult5922.rawTerms
def rightRaw : List Term := SemanticRightRootResult5647.rawTerms
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6708⟩⟩, ⟨.program ⟨214⟩, ⟨6679⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6710⟩⟩, ⟨.program ⟨214⟩, ⟨6671⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6712⟩⟩, ⟨.program ⟨214⟩, ⟨6663⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6714⟩⟩, ⟨.program ⟨214⟩, ⟨6655⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6716⟩⟩, ⟨.program ⟨214⟩, ⟨6649⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6718⟩⟩, ⟨.program ⟨214⟩, ⟨6647⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6720⟩⟩, ⟨.program ⟨214⟩, ⟨6643⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6722⟩⟩, ⟨.program ⟨214⟩, ⟨6641⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6724⟩⟩, ⟨.program ⟨214⟩, ⟨6637⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6726⟩⟩, ⟨.program ⟨214⟩, ⟨6681⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6728⟩⟩, ⟨.program ⟨214⟩, ⟨6677⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6730⟩⟩, ⟨.program ⟨214⟩, ⟨6673⟩⟩] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def owner : Owner := ⟨.program ⟨214⟩, ⟨7662⟩⟩
def rawTerms : List Term := outputRaw
def summary : Bound := .exactZero
def resultEvent : Nat := 5926
theorem resultAgreement : CanonicalAgreement output (add left right) := by
  decide +kernel

theorem resultSound (env : Env Owner) :
    evalPolynomial env output = evalPolynomial env left + evalPolynomial env right := by
  exact addCanonicalResultSound env left right output resultAgreement

def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  SemanticRightRootResult5922.actual selector witness +
    SemanticRightRootResult5647.actual selector witness
theorem claimOfHistory (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841)
    (expressionAt : document.expressions.lookup 7662 =
      some Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Expression029.ExpressionRow7662)
    (leftPredecessorAt : (history.lookup 5923).map AnnotatedEvent.event =
      some (.predecessor owner 0 ⟨7661⟩ 5922))
    (rightPredecessorAt : (history.lookup 5924).map AnnotatedEvent.event =
      some (.predecessor owner 1 ⟨7642⟩ 5647))
    (ruleAt : (history.lookup 5925).map AnnotatedEvent.event =
      some (.boundTransfer owner (.sum [.predecessor 0 5923 .coefficient, .predecessor 1 5924 .coefficient])))
    (leftClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult5922.resultEvent
      SemanticRightRootResult5922.owner
      (SemanticRightRootResult5922.actual selector witness)
      SemanticRightRootResult5922.rawTerms
      SemanticRightRootResult5922.summary)
    (rightClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult5647.resultEvent
      SemanticRightRootResult5647.owner
      (SemanticRightRootResult5647.actual selector witness)
      SemanticRightRootResult5647.rawTerms
      SemanticRightRootResult5647.summary)
    (outputAt : (history.lookup resultEvent).map AnnotatedEvent.event =
      some (.resultExact owner rawTerms summary)) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  refine ⟨outputAt, ?_⟩
  exact exactValueClaim_add_of_mod_zero 100418593683253592432016548326729029359133068138294319235841 witness.env
    (SemanticRightRootResult5922.actual selector witness)
    (SemanticRightRootResult5647.actual selector witness) left right output
    (by simpa [left, leftRaw, SemanticRightRootResult5922.summary] using leftClaim.claim)
    (by simpa [right, rightRaw, SemanticRightRootResult5647.summary] using rightClaim.claim)
    (resultSound witness.env) (by decide)

theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply claimOfHistory selector witness (by rfl) (by rfl) (by rfl) (by rfl)
  · exact SemanticRightRootResult5922.claimSound
      selector selectorLower selectorUpper witness
  · exact SemanticRightRootResult5647.claimSound
      selector selectorLower selectorUpper witness
  · rfl

end SemanticRightRootResult5926

namespace SemanticRightRootResult5930

def leftRaw : List Term := SemanticRightRootResult5926.rawTerms
def rightRaw : List Term := SemanticRightRootResult5627.rawTerms
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6708⟩⟩, ⟨.program ⟨214⟩, ⟨6679⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6710⟩⟩, ⟨.program ⟨214⟩, ⟨6671⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6712⟩⟩, ⟨.program ⟨214⟩, ⟨6663⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6714⟩⟩, ⟨.program ⟨214⟩, ⟨6655⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6716⟩⟩, ⟨.program ⟨214⟩, ⟨6649⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6718⟩⟩, ⟨.program ⟨214⟩, ⟨6647⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6720⟩⟩, ⟨.program ⟨214⟩, ⟨6643⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6722⟩⟩, ⟨.program ⟨214⟩, ⟨6641⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6724⟩⟩, ⟨.program ⟨214⟩, ⟨6637⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6726⟩⟩, ⟨.program ⟨214⟩, ⟨6681⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6728⟩⟩, ⟨.program ⟨214⟩, ⟨6677⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6730⟩⟩, ⟨.program ⟨214⟩, ⟨6673⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6732⟩⟩, ⟨.program ⟨214⟩, ⟨6669⟩⟩] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def owner : Owner := ⟨.program ⟨214⟩, ⟨7663⟩⟩
def rawTerms : List Term := outputRaw
def summary : Bound := .exactZero
def resultEvent : Nat := 5930
theorem resultAgreement : CanonicalAgreement output (add left right) := by
  decide +kernel

theorem resultSound (env : Env Owner) :
    evalPolynomial env output = evalPolynomial env left + evalPolynomial env right := by
  exact addCanonicalResultSound env left right output resultAgreement

def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  SemanticRightRootResult5926.actual selector witness +
    SemanticRightRootResult5627.actual selector witness
theorem claimOfHistory (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841)
    (expressionAt : document.expressions.lookup 7663 =
      some Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Expression029.ExpressionRow7663)
    (leftPredecessorAt : (history.lookup 5927).map AnnotatedEvent.event =
      some (.predecessor owner 0 ⟨7662⟩ 5926))
    (rightPredecessorAt : (history.lookup 5928).map AnnotatedEvent.event =
      some (.predecessor owner 1 ⟨7643⟩ 5627))
    (ruleAt : (history.lookup 5929).map AnnotatedEvent.event =
      some (.boundTransfer owner (.sum [.predecessor 0 5927 .coefficient, .predecessor 1 5928 .coefficient])))
    (leftClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult5926.resultEvent
      SemanticRightRootResult5926.owner
      (SemanticRightRootResult5926.actual selector witness)
      SemanticRightRootResult5926.rawTerms
      SemanticRightRootResult5926.summary)
    (rightClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult5627.resultEvent
      SemanticRightRootResult5627.owner
      (SemanticRightRootResult5627.actual selector witness)
      SemanticRightRootResult5627.rawTerms
      SemanticRightRootResult5627.summary)
    (outputAt : (history.lookup resultEvent).map AnnotatedEvent.event =
      some (.resultExact owner rawTerms summary)) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  refine ⟨outputAt, ?_⟩
  exact exactValueClaim_add_of_mod_zero 100418593683253592432016548326729029359133068138294319235841 witness.env
    (SemanticRightRootResult5926.actual selector witness)
    (SemanticRightRootResult5627.actual selector witness) left right output
    (by simpa [left, leftRaw, SemanticRightRootResult5926.summary] using leftClaim.claim)
    (by simpa [right, rightRaw, SemanticRightRootResult5627.summary] using rightClaim.claim)
    (resultSound witness.env) (by decide)

theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply claimOfHistory selector witness (by rfl) (by rfl) (by rfl) (by rfl)
  · exact SemanticRightRootResult5926.claimSound
      selector selectorLower selectorUpper witness
  · exact SemanticRightRootResult5627.claimSound
      selector selectorLower selectorUpper witness
  · rfl

end SemanticRightRootResult5930

namespace SemanticRightRootResult5934

def leftRaw : List Term := SemanticRightRootResult5930.rawTerms
def rightRaw : List Term := SemanticRightRootResult5607.rawTerms
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6708⟩⟩, ⟨.program ⟨214⟩, ⟨6679⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6710⟩⟩, ⟨.program ⟨214⟩, ⟨6671⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6712⟩⟩, ⟨.program ⟨214⟩, ⟨6663⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6714⟩⟩, ⟨.program ⟨214⟩, ⟨6655⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6716⟩⟩, ⟨.program ⟨214⟩, ⟨6649⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6718⟩⟩, ⟨.program ⟨214⟩, ⟨6647⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6720⟩⟩, ⟨.program ⟨214⟩, ⟨6643⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6722⟩⟩, ⟨.program ⟨214⟩, ⟨6641⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6724⟩⟩, ⟨.program ⟨214⟩, ⟨6637⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6726⟩⟩, ⟨.program ⟨214⟩, ⟨6681⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6728⟩⟩, ⟨.program ⟨214⟩, ⟨6677⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6730⟩⟩, ⟨.program ⟨214⟩, ⟨6673⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6732⟩⟩, ⟨.program ⟨214⟩, ⟨6669⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6734⟩⟩, ⟨.program ⟨214⟩, ⟨6667⟩⟩] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def owner : Owner := ⟨.program ⟨214⟩, ⟨7664⟩⟩
def rawTerms : List Term := outputRaw
def summary : Bound := .exactZero
def resultEvent : Nat := 5934
theorem resultAgreement : CanonicalAgreement output (add left right) := by
  decide +kernel

theorem resultSound (env : Env Owner) :
    evalPolynomial env output = evalPolynomial env left + evalPolynomial env right := by
  exact addCanonicalResultSound env left right output resultAgreement

def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  SemanticRightRootResult5930.actual selector witness +
    SemanticRightRootResult5607.actual selector witness
theorem claimOfHistory (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841)
    (expressionAt : document.expressions.lookup 7664 =
      some Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Expression029.ExpressionRow7664)
    (leftPredecessorAt : (history.lookup 5931).map AnnotatedEvent.event =
      some (.predecessor owner 0 ⟨7663⟩ 5930))
    (rightPredecessorAt : (history.lookup 5932).map AnnotatedEvent.event =
      some (.predecessor owner 1 ⟨7644⟩ 5607))
    (ruleAt : (history.lookup 5933).map AnnotatedEvent.event =
      some (.boundTransfer owner (.sum [.predecessor 0 5931 .coefficient, .predecessor 1 5932 .coefficient])))
    (leftClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult5930.resultEvent
      SemanticRightRootResult5930.owner
      (SemanticRightRootResult5930.actual selector witness)
      SemanticRightRootResult5930.rawTerms
      SemanticRightRootResult5930.summary)
    (rightClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult5607.resultEvent
      SemanticRightRootResult5607.owner
      (SemanticRightRootResult5607.actual selector witness)
      SemanticRightRootResult5607.rawTerms
      SemanticRightRootResult5607.summary)
    (outputAt : (history.lookup resultEvent).map AnnotatedEvent.event =
      some (.resultExact owner rawTerms summary)) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  refine ⟨outputAt, ?_⟩
  exact exactValueClaim_add_of_mod_zero 100418593683253592432016548326729029359133068138294319235841 witness.env
    (SemanticRightRootResult5930.actual selector witness)
    (SemanticRightRootResult5607.actual selector witness) left right output
    (by simpa [left, leftRaw, SemanticRightRootResult5930.summary] using leftClaim.claim)
    (by simpa [right, rightRaw, SemanticRightRootResult5607.summary] using rightClaim.claim)
    (resultSound witness.env) (by decide)

theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply claimOfHistory selector witness (by rfl) (by rfl) (by rfl) (by rfl)
  · exact SemanticRightRootResult5930.claimSound
      selector selectorLower selectorUpper witness
  · exact SemanticRightRootResult5607.claimSound
      selector selectorLower selectorUpper witness
  · rfl

end SemanticRightRootResult5934

namespace SemanticRightRootResult5938

def leftRaw : List Term := SemanticRightRootResult5934.rawTerms
def rightRaw : List Term := SemanticRightRootResult5587.rawTerms
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6708⟩⟩, ⟨.program ⟨214⟩, ⟨6679⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6710⟩⟩, ⟨.program ⟨214⟩, ⟨6671⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6712⟩⟩, ⟨.program ⟨214⟩, ⟨6663⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6714⟩⟩, ⟨.program ⟨214⟩, ⟨6655⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6716⟩⟩, ⟨.program ⟨214⟩, ⟨6649⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6718⟩⟩, ⟨.program ⟨214⟩, ⟨6647⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6720⟩⟩, ⟨.program ⟨214⟩, ⟨6643⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6722⟩⟩, ⟨.program ⟨214⟩, ⟨6641⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6724⟩⟩, ⟨.program ⟨214⟩, ⟨6637⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6726⟩⟩, ⟨.program ⟨214⟩, ⟨6681⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6728⟩⟩, ⟨.program ⟨214⟩, ⟨6677⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6730⟩⟩, ⟨.program ⟨214⟩, ⟨6673⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6732⟩⟩, ⟨.program ⟨214⟩, ⟨6669⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6734⟩⟩, ⟨.program ⟨214⟩, ⟨6667⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6736⟩⟩, ⟨.program ⟨214⟩, ⟨6665⟩⟩] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def owner : Owner := ⟨.program ⟨214⟩, ⟨7665⟩⟩
def rawTerms : List Term := outputRaw
def summary : Bound := .exactZero
def resultEvent : Nat := 5938
theorem resultAgreement : CanonicalAgreement output (add left right) := by
  decide +kernel

theorem resultSound (env : Env Owner) :
    evalPolynomial env output = evalPolynomial env left + evalPolynomial env right := by
  exact addCanonicalResultSound env left right output resultAgreement

def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  SemanticRightRootResult5934.actual selector witness +
    SemanticRightRootResult5587.actual selector witness
theorem claimOfHistory (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841)
    (expressionAt : document.expressions.lookup 7665 =
      some Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Expression029.ExpressionRow7665)
    (leftPredecessorAt : (history.lookup 5935).map AnnotatedEvent.event =
      some (.predecessor owner 0 ⟨7664⟩ 5934))
    (rightPredecessorAt : (history.lookup 5936).map AnnotatedEvent.event =
      some (.predecessor owner 1 ⟨7645⟩ 5587))
    (ruleAt : (history.lookup 5937).map AnnotatedEvent.event =
      some (.boundTransfer owner (.sum [.predecessor 0 5935 .coefficient, .predecessor 1 5936 .coefficient])))
    (leftClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult5934.resultEvent
      SemanticRightRootResult5934.owner
      (SemanticRightRootResult5934.actual selector witness)
      SemanticRightRootResult5934.rawTerms
      SemanticRightRootResult5934.summary)
    (rightClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult5587.resultEvent
      SemanticRightRootResult5587.owner
      (SemanticRightRootResult5587.actual selector witness)
      SemanticRightRootResult5587.rawTerms
      SemanticRightRootResult5587.summary)
    (outputAt : (history.lookup resultEvent).map AnnotatedEvent.event =
      some (.resultExact owner rawTerms summary)) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  refine ⟨outputAt, ?_⟩
  exact exactValueClaim_add_of_mod_zero 100418593683253592432016548326729029359133068138294319235841 witness.env
    (SemanticRightRootResult5934.actual selector witness)
    (SemanticRightRootResult5587.actual selector witness) left right output
    (by simpa [left, leftRaw, SemanticRightRootResult5934.summary] using leftClaim.claim)
    (by simpa [right, rightRaw, SemanticRightRootResult5587.summary] using rightClaim.claim)
    (resultSound witness.env) (by decide)

theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply claimOfHistory selector witness (by rfl) (by rfl) (by rfl) (by rfl)
  · exact SemanticRightRootResult5934.claimSound
      selector selectorLower selectorUpper witness
  · exact SemanticRightRootResult5587.claimSound
      selector selectorLower selectorUpper witness
  · rfl

end SemanticRightRootResult5938

namespace SemanticRightRootResult5942

def leftRaw : List Term := SemanticRightRootResult5938.rawTerms
def rightRaw : List Term := SemanticRightRootResult5567.rawTerms
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6708⟩⟩, ⟨.program ⟨214⟩, ⟨6679⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6710⟩⟩, ⟨.program ⟨214⟩, ⟨6671⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6712⟩⟩, ⟨.program ⟨214⟩, ⟨6663⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6714⟩⟩, ⟨.program ⟨214⟩, ⟨6655⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6716⟩⟩, ⟨.program ⟨214⟩, ⟨6649⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6718⟩⟩, ⟨.program ⟨214⟩, ⟨6647⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6720⟩⟩, ⟨.program ⟨214⟩, ⟨6643⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6722⟩⟩, ⟨.program ⟨214⟩, ⟨6641⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6724⟩⟩, ⟨.program ⟨214⟩, ⟨6637⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6726⟩⟩, ⟨.program ⟨214⟩, ⟨6681⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6728⟩⟩, ⟨.program ⟨214⟩, ⟨6677⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6730⟩⟩, ⟨.program ⟨214⟩, ⟨6673⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6732⟩⟩, ⟨.program ⟨214⟩, ⟨6669⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6734⟩⟩, ⟨.program ⟨214⟩, ⟨6667⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6736⟩⟩, ⟨.program ⟨214⟩, ⟨6665⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6738⟩⟩, ⟨.program ⟨214⟩, ⟨6661⟩⟩] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def owner : Owner := ⟨.program ⟨214⟩, ⟨7666⟩⟩
def rawTerms : List Term := outputRaw
def summary : Bound := .exactZero
def resultEvent : Nat := 5942
theorem resultAgreement : CanonicalAgreement output (add left right) := by
  decide +kernel

theorem resultSound (env : Env Owner) :
    evalPolynomial env output = evalPolynomial env left + evalPolynomial env right := by
  exact addCanonicalResultSound env left right output resultAgreement

def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  SemanticRightRootResult5938.actual selector witness +
    SemanticRightRootResult5567.actual selector witness
theorem claimOfHistory (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841)
    (expressionAt : document.expressions.lookup 7666 =
      some Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Expression029.ExpressionRow7666)
    (leftPredecessorAt : (history.lookup 5939).map AnnotatedEvent.event =
      some (.predecessor owner 0 ⟨7665⟩ 5938))
    (rightPredecessorAt : (history.lookup 5940).map AnnotatedEvent.event =
      some (.predecessor owner 1 ⟨7646⟩ 5567))
    (ruleAt : (history.lookup 5941).map AnnotatedEvent.event =
      some (.boundTransfer owner (.sum [.predecessor 0 5939 .coefficient, .predecessor 1 5940 .coefficient])))
    (leftClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult5938.resultEvent
      SemanticRightRootResult5938.owner
      (SemanticRightRootResult5938.actual selector witness)
      SemanticRightRootResult5938.rawTerms
      SemanticRightRootResult5938.summary)
    (rightClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult5567.resultEvent
      SemanticRightRootResult5567.owner
      (SemanticRightRootResult5567.actual selector witness)
      SemanticRightRootResult5567.rawTerms
      SemanticRightRootResult5567.summary)
    (outputAt : (history.lookup resultEvent).map AnnotatedEvent.event =
      some (.resultExact owner rawTerms summary)) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  refine ⟨outputAt, ?_⟩
  exact exactValueClaim_add_of_mod_zero 100418593683253592432016548326729029359133068138294319235841 witness.env
    (SemanticRightRootResult5938.actual selector witness)
    (SemanticRightRootResult5567.actual selector witness) left right output
    (by simpa [left, leftRaw, SemanticRightRootResult5938.summary] using leftClaim.claim)
    (by simpa [right, rightRaw, SemanticRightRootResult5567.summary] using rightClaim.claim)
    (resultSound witness.env) (by decide)

theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply claimOfHistory selector witness (by rfl) (by rfl) (by rfl) (by rfl)
  · exact SemanticRightRootResult5938.claimSound
      selector selectorLower selectorUpper witness
  · exact SemanticRightRootResult5567.claimSound
      selector selectorLower selectorUpper witness
  · rfl

end SemanticRightRootResult5942

namespace SemanticRightRootResult5946

def leftRaw : List Term := SemanticRightRootResult5942.rawTerms
def rightRaw : List Term := SemanticRightRootResult5547.rawTerms
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6708⟩⟩, ⟨.program ⟨214⟩, ⟨6679⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6710⟩⟩, ⟨.program ⟨214⟩, ⟨6671⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6712⟩⟩, ⟨.program ⟨214⟩, ⟨6663⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6714⟩⟩, ⟨.program ⟨214⟩, ⟨6655⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6716⟩⟩, ⟨.program ⟨214⟩, ⟨6649⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6718⟩⟩, ⟨.program ⟨214⟩, ⟨6647⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6720⟩⟩, ⟨.program ⟨214⟩, ⟨6643⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6722⟩⟩, ⟨.program ⟨214⟩, ⟨6641⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6724⟩⟩, ⟨.program ⟨214⟩, ⟨6637⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6726⟩⟩, ⟨.program ⟨214⟩, ⟨6681⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6728⟩⟩, ⟨.program ⟨214⟩, ⟨6677⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6730⟩⟩, ⟨.program ⟨214⟩, ⟨6673⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6732⟩⟩, ⟨.program ⟨214⟩, ⟨6669⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6734⟩⟩, ⟨.program ⟨214⟩, ⟨6667⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6736⟩⟩, ⟨.program ⟨214⟩, ⟨6665⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6738⟩⟩, ⟨.program ⟨214⟩, ⟨6661⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6740⟩⟩, ⟨.program ⟨214⟩, ⟨6659⟩⟩] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def owner : Owner := ⟨.program ⟨214⟩, ⟨7667⟩⟩
def rawTerms : List Term := outputRaw
def summary : Bound := .exactZero
def resultEvent : Nat := 5946
theorem resultAgreement : CanonicalAgreement output (add left right) := by
  decide +kernel

theorem resultSound (env : Env Owner) :
    evalPolynomial env output = evalPolynomial env left + evalPolynomial env right := by
  exact addCanonicalResultSound env left right output resultAgreement

def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  SemanticRightRootResult5942.actual selector witness +
    SemanticRightRootResult5547.actual selector witness
theorem claimOfHistory (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841)
    (expressionAt : document.expressions.lookup 7667 =
      some Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Expression029.ExpressionRow7667)
    (leftPredecessorAt : (history.lookup 5943).map AnnotatedEvent.event =
      some (.predecessor owner 0 ⟨7666⟩ 5942))
    (rightPredecessorAt : (history.lookup 5944).map AnnotatedEvent.event =
      some (.predecessor owner 1 ⟨7647⟩ 5547))
    (ruleAt : (history.lookup 5945).map AnnotatedEvent.event =
      some (.boundTransfer owner (.sum [.predecessor 0 5943 .coefficient, .predecessor 1 5944 .coefficient])))
    (leftClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult5942.resultEvent
      SemanticRightRootResult5942.owner
      (SemanticRightRootResult5942.actual selector witness)
      SemanticRightRootResult5942.rawTerms
      SemanticRightRootResult5942.summary)
    (rightClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult5547.resultEvent
      SemanticRightRootResult5547.owner
      (SemanticRightRootResult5547.actual selector witness)
      SemanticRightRootResult5547.rawTerms
      SemanticRightRootResult5547.summary)
    (outputAt : (history.lookup resultEvent).map AnnotatedEvent.event =
      some (.resultExact owner rawTerms summary)) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  refine ⟨outputAt, ?_⟩
  exact exactValueClaim_add_of_mod_zero 100418593683253592432016548326729029359133068138294319235841 witness.env
    (SemanticRightRootResult5942.actual selector witness)
    (SemanticRightRootResult5547.actual selector witness) left right output
    (by simpa [left, leftRaw, SemanticRightRootResult5942.summary] using leftClaim.claim)
    (by simpa [right, rightRaw, SemanticRightRootResult5547.summary] using rightClaim.claim)
    (resultSound witness.env) (by decide)

theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply claimOfHistory selector witness (by rfl) (by rfl) (by rfl) (by rfl)
  · exact SemanticRightRootResult5942.claimSound
      selector selectorLower selectorUpper witness
  · exact SemanticRightRootResult5547.claimSound
      selector selectorLower selectorUpper witness
  · rfl

end SemanticRightRootResult5946

namespace SemanticRightRootResult5950

def leftRaw : List Term := SemanticRightRootResult5946.rawTerms
def rightRaw : List Term := SemanticRightRootResult5527.rawTerms
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6708⟩⟩, ⟨.program ⟨214⟩, ⟨6679⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6710⟩⟩, ⟨.program ⟨214⟩, ⟨6671⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6712⟩⟩, ⟨.program ⟨214⟩, ⟨6663⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6714⟩⟩, ⟨.program ⟨214⟩, ⟨6655⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6716⟩⟩, ⟨.program ⟨214⟩, ⟨6649⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6718⟩⟩, ⟨.program ⟨214⟩, ⟨6647⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6720⟩⟩, ⟨.program ⟨214⟩, ⟨6643⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6722⟩⟩, ⟨.program ⟨214⟩, ⟨6641⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6724⟩⟩, ⟨.program ⟨214⟩, ⟨6637⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6726⟩⟩, ⟨.program ⟨214⟩, ⟨6681⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6728⟩⟩, ⟨.program ⟨214⟩, ⟨6677⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6730⟩⟩, ⟨.program ⟨214⟩, ⟨6673⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6732⟩⟩, ⟨.program ⟨214⟩, ⟨6669⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6734⟩⟩, ⟨.program ⟨214⟩, ⟨6667⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6736⟩⟩, ⟨.program ⟨214⟩, ⟨6665⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6738⟩⟩, ⟨.program ⟨214⟩, ⟨6661⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6740⟩⟩, ⟨.program ⟨214⟩, ⟨6659⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6742⟩⟩, ⟨.program ⟨214⟩, ⟨6657⟩⟩] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def owner : Owner := ⟨.program ⟨214⟩, ⟨7668⟩⟩
def rawTerms : List Term := outputRaw
def summary : Bound := .exactZero
def resultEvent : Nat := 5950
theorem resultAgreement : CanonicalAgreement output (add left right) := by
  decide +kernel

theorem resultSound (env : Env Owner) :
    evalPolynomial env output = evalPolynomial env left + evalPolynomial env right := by
  exact addCanonicalResultSound env left right output resultAgreement

def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  SemanticRightRootResult5946.actual selector witness +
    SemanticRightRootResult5527.actual selector witness
theorem claimOfHistory (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841)
    (expressionAt : document.expressions.lookup 7668 =
      some Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Expression029.ExpressionRow7668)
    (leftPredecessorAt : (history.lookup 5947).map AnnotatedEvent.event =
      some (.predecessor owner 0 ⟨7667⟩ 5946))
    (rightPredecessorAt : (history.lookup 5948).map AnnotatedEvent.event =
      some (.predecessor owner 1 ⟨7648⟩ 5527))
    (ruleAt : (history.lookup 5949).map AnnotatedEvent.event =
      some (.boundTransfer owner (.sum [.predecessor 0 5947 .coefficient, .predecessor 1 5948 .coefficient])))
    (leftClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult5946.resultEvent
      SemanticRightRootResult5946.owner
      (SemanticRightRootResult5946.actual selector witness)
      SemanticRightRootResult5946.rawTerms
      SemanticRightRootResult5946.summary)
    (rightClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult5527.resultEvent
      SemanticRightRootResult5527.owner
      (SemanticRightRootResult5527.actual selector witness)
      SemanticRightRootResult5527.rawTerms
      SemanticRightRootResult5527.summary)
    (outputAt : (history.lookup resultEvent).map AnnotatedEvent.event =
      some (.resultExact owner rawTerms summary)) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  refine ⟨outputAt, ?_⟩
  exact exactValueClaim_add_of_mod_zero 100418593683253592432016548326729029359133068138294319235841 witness.env
    (SemanticRightRootResult5946.actual selector witness)
    (SemanticRightRootResult5527.actual selector witness) left right output
    (by simpa [left, leftRaw, SemanticRightRootResult5946.summary] using leftClaim.claim)
    (by simpa [right, rightRaw, SemanticRightRootResult5527.summary] using rightClaim.claim)
    (resultSound witness.env) (by decide)

theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply claimOfHistory selector witness (by rfl) (by rfl) (by rfl) (by rfl)
  · exact SemanticRightRootResult5946.claimSound
      selector selectorLower selectorUpper witness
  · exact SemanticRightRootResult5527.claimSound
      selector selectorLower selectorUpper witness
  · rfl

end SemanticRightRootResult5950

namespace SemanticRightRootResult5954

def leftRaw : List Term := SemanticRightRootResult5950.rawTerms
def rightRaw : List Term := SemanticRightRootResult5507.rawTerms
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6708⟩⟩, ⟨.program ⟨214⟩, ⟨6679⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6710⟩⟩, ⟨.program ⟨214⟩, ⟨6671⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6712⟩⟩, ⟨.program ⟨214⟩, ⟨6663⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6714⟩⟩, ⟨.program ⟨214⟩, ⟨6655⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6716⟩⟩, ⟨.program ⟨214⟩, ⟨6649⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6718⟩⟩, ⟨.program ⟨214⟩, ⟨6647⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6720⟩⟩, ⟨.program ⟨214⟩, ⟨6643⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6722⟩⟩, ⟨.program ⟨214⟩, ⟨6641⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6724⟩⟩, ⟨.program ⟨214⟩, ⟨6637⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6726⟩⟩, ⟨.program ⟨214⟩, ⟨6681⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6728⟩⟩, ⟨.program ⟨214⟩, ⟨6677⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6730⟩⟩, ⟨.program ⟨214⟩, ⟨6673⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6732⟩⟩, ⟨.program ⟨214⟩, ⟨6669⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6734⟩⟩, ⟨.program ⟨214⟩, ⟨6667⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6736⟩⟩, ⟨.program ⟨214⟩, ⟨6665⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6738⟩⟩, ⟨.program ⟨214⟩, ⟨6661⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6740⟩⟩, ⟨.program ⟨214⟩, ⟨6659⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6742⟩⟩, ⟨.program ⟨214⟩, ⟨6657⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6744⟩⟩, ⟨.program ⟨214⟩, ⟨6651⟩⟩] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def owner : Owner := ⟨.program ⟨214⟩, ⟨7795⟩⟩
def rawTerms : List Term := outputRaw
def summary : Bound := .exactZero
def resultEvent : Nat := 5954
theorem resultAgreement : CanonicalAgreement output (subtract left right) := by
  decide +kernel

theorem resultSound (env : Env Owner) :
    evalPolynomial env output = evalPolynomial env left - evalPolynomial env right := by
  exact subCanonicalResultSound env left right output resultAgreement

def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  SemanticRightRootResult5950.actual selector witness -
    SemanticRightRootResult5507.actual selector witness
theorem claimOfHistory (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841)
    (expressionAt : document.expressions.lookup 7795 =
      some Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Expression030.ExpressionRow7795)
    (leftPredecessorAt : (history.lookup 5951).map AnnotatedEvent.event =
      some (.predecessor owner 0 ⟨7668⟩ 5950))
    (rightPredecessorAt : (history.lookup 5952).map AnnotatedEvent.event =
      some (.predecessor owner 1 ⟨7649⟩ 5507))
    (ruleAt : (history.lookup 5953).map AnnotatedEvent.event =
      some (.boundTransfer owner (.sum [.predecessor 0 5951 .coefficient, .predecessor 1 5952 .coefficient])))
    (leftClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult5950.resultEvent
      SemanticRightRootResult5950.owner
      (SemanticRightRootResult5950.actual selector witness)
      SemanticRightRootResult5950.rawTerms
      SemanticRightRootResult5950.summary)
    (rightClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult5507.resultEvent
      SemanticRightRootResult5507.owner
      (SemanticRightRootResult5507.actual selector witness)
      SemanticRightRootResult5507.rawTerms
      SemanticRightRootResult5507.summary)
    (outputAt : (history.lookup resultEvent).map AnnotatedEvent.event =
      some (.resultExact owner rawTerms summary)) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  refine ⟨outputAt, ?_⟩
  exact exactValueClaim_sub_exactZero_of_mod_zero 100418593683253592432016548326729029359133068138294319235841 witness.env
    (SemanticRightRootResult5950.actual selector witness)
    (SemanticRightRootResult5507.actual selector witness) left right output
    (by simpa [left, leftRaw, SemanticRightRootResult5950.summary] using leftClaim.claim)
    (by simpa [right, rightRaw, SemanticRightRootResult5507.summary] using rightClaim.claim)
    (resultSound witness.env) (by decide)

theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply claimOfHistory selector witness (by rfl) (by rfl) (by rfl) (by rfl)
  · exact SemanticRightRootResult5950.claimSound
      selector selectorLower selectorUpper witness
  · exact SemanticRightRootResult5507.claimSound
      selector selectorLower selectorUpper witness
  · rfl

end SemanticRightRootResult5954

namespace SemanticRightRootResult5961

def owner : Owner := ⟨.program ⟨214⟩, ⟨7886⟩⟩
def rawTerms : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨7885⟩⟩] } }]
def summary : Bound := .exactZero
def producerEvent : Nat := 5960
def resultEvent : Nat := 5961
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual resultEvent
theorem terminalAt (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, rightRootOwnerAtSelector214 selector selectorLower selectorUpper _, ?_⟩
  refine ⟨.scale (.predecessor 0 5958 .coefficient) (.value (.predecessor 1 5959 .coefficient)), 0, .scale (.predecessor 0 5958 .coefficient) (.value (.predecessor 1 5959 .coefficient)), ?_, ?_⟩
  · rfl
  · rfl
theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness
    (terminalAt selector selectorLower selectorUpper)

end SemanticRightRootResult5961

namespace SemanticRightRootResult5964

def owner : Owner := ⟨.program ⟨214⟩, ⟨6745⟩⟩
def rawTerms : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6745⟩⟩] } }]
def summary : Bound := .exactZero
def producerEvent : Nat := 5963
def resultEvent : Nat := 5964
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

end SemanticRightRootResult5964

namespace SemanticRightRootResult5969

def leftRaw : List Term := SemanticRightRootResult5964.rawTerms
def rightRaw : List Term := SemanticRightRootResult5961.rawTerms
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6745⟩⟩, ⟨.program ⟨214⟩, ⟨7885⟩⟩] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def owner : Owner := ⟨.program ⟨214⟩, ⟨7887⟩⟩
def rawTerms : List Term := outputRaw
def summary : Bound := .exactZero
def resultEvent : Nat := 5969
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
  SemanticRightRootResult5964.actual selector witness *
    SemanticRightRootResult5961.actual selector witness
theorem claimOfHistory (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841)
    (expressionAt : document.expressions.lookup 7887 =
      some Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Expression030.ExpressionRow7887)
    (leftPredecessorAt : (history.lookup 5965).map AnnotatedEvent.event =
      some (.predecessor owner 0 ⟨6745⟩ 5964))
    (rightPredecessorAt : (history.lookup 5966).map AnnotatedEvent.event =
      some (.predecessor owner 1 ⟨7886⟩ 5961))
    (ruleAt : (history.lookup 5967).map AnnotatedEvent.event =
      some (.boundTransfer owner (.product (.predecessor 0 5965 .coefficient) (.predecessor 1 5966 .coefficient) ⟨false, false, none, none, none⟩)))
    (leftClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult5964.resultEvent
      SemanticRightRootResult5964.owner
      (SemanticRightRootResult5964.actual selector witness)
      SemanticRightRootResult5964.rawTerms
      SemanticRightRootResult5964.summary)
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
    (SemanticRightRootResult5964.actual selector witness)
    (SemanticRightRootResult5961.actual selector witness) left right output
    (by simpa [left, leftRaw, SemanticRightRootResult5964.summary] using leftClaim.claim)
    (by simpa [right, rightRaw, SemanticRightRootResult5961.summary] using rightClaim.claim)
    (resultSound witness.env) (by decide)

theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply claimOfHistory selector witness (by rfl) (by rfl) (by rfl) (by rfl)
  · exact SemanticRightRootResult5964.claimSound
      selector selectorLower selectorUpper witness
  · exact SemanticRightRootResult5961.claimSound
      selector selectorLower selectorUpper witness
  · rfl

end SemanticRightRootResult5969

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
