import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticRightRootShard029

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open TallSecurity0ABI
open TallSemantics

namespace SemanticRightRootResult5719

def owner : Owner := ⟨.program ⟨214⟩, ⟨6642⟩⟩
def rawTerms : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6641⟩⟩] } }]
def summary : Bound := .exactZero
def producerEvent : Nat := 5718
def resultEvent : Nat := 5719
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual resultEvent
theorem terminalAt (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, rightRootOwnerAtSelector214 selector selectorLower selectorUpper _, ?_⟩
  refine ⟨.scale (.predecessor 0 5716 .coefficient) (.value (.predecessor 1 5717 .coefficient)), 0, .scale (.predecessor 0 5716 .coefficient) (.value (.predecessor 1 5717 .coefficient)), ?_, ?_⟩
  · rfl
  · rfl
theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness
    (terminalAt selector selectorLower selectorUpper)

end SemanticRightRootResult5719

namespace SemanticRightRootResult5722

def owner : Owner := ⟨.program ⟨214⟩, ⟨6722⟩⟩
def rawTerms : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6722⟩⟩] } }]
def summary : Bound := .exactZero
def producerEvent : Nat := 5721
def resultEvent : Nat := 5722
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

end SemanticRightRootResult5722

namespace SemanticRightRootResult5727

def leftRaw : List Term := SemanticRightRootResult5722.rawTerms
def rightRaw : List Term := SemanticRightRootResult5719.rawTerms
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6722⟩⟩, ⟨.program ⟨214⟩, ⟨6641⟩⟩] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def owner : Owner := ⟨.program ⟨214⟩, ⟨7638⟩⟩
def rawTerms : List Term := outputRaw
def summary : Bound := .exactZero
def resultEvent : Nat := 5727
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
  SemanticRightRootResult5722.actual selector witness *
    SemanticRightRootResult5719.actual selector witness
theorem claimOfHistory (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841)
    (expressionAt : document.expressions.lookup 7638 =
      some Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Expression029.ExpressionRow7638)
    (leftPredecessorAt : (history.lookup 5723).map AnnotatedEvent.event =
      some (.predecessor owner 0 ⟨6722⟩ 5722))
    (rightPredecessorAt : (history.lookup 5724).map AnnotatedEvent.event =
      some (.predecessor owner 1 ⟨6642⟩ 5719))
    (ruleAt : (history.lookup 5725).map AnnotatedEvent.event =
      some (.boundTransfer owner (.product (.predecessor 0 5723 .coefficient) (.predecessor 1 5724 .coefficient) ⟨false, false, none, none, none⟩)))
    (leftClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult5722.resultEvent
      SemanticRightRootResult5722.owner
      (SemanticRightRootResult5722.actual selector witness)
      SemanticRightRootResult5722.rawTerms
      SemanticRightRootResult5722.summary)
    (rightClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult5719.resultEvent
      SemanticRightRootResult5719.owner
      (SemanticRightRootResult5719.actual selector witness)
      SemanticRightRootResult5719.rawTerms
      SemanticRightRootResult5719.summary)
    (outputAt : (history.lookup resultEvent).map AnnotatedEvent.event =
      some (.resultExact owner rawTerms summary)) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  refine ⟨outputAt, ?_⟩
  exact exactValueClaim_product_of_mod_zero 100418593683253592432016548326729029359133068138294319235841 witness.env
    (SemanticRightRootResult5722.actual selector witness)
    (SemanticRightRootResult5719.actual selector witness) left right output
    (by simpa [left, leftRaw, SemanticRightRootResult5722.summary] using leftClaim.claim)
    (by simpa [right, rightRaw, SemanticRightRootResult5719.summary] using rightClaim.claim)
    (resultSound witness.env) (by decide)

theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply claimOfHistory selector witness (by rfl) (by rfl) (by rfl) (by rfl)
  · exact SemanticRightRootResult5722.claimSound
      selector selectorLower selectorUpper witness
  · exact SemanticRightRootResult5719.claimSound
      selector selectorLower selectorUpper witness
  · rfl

end SemanticRightRootResult5727

namespace SemanticRightRootResult5739

def owner : Owner := ⟨.program ⟨214⟩, ⟨6644⟩⟩
def rawTerms : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6643⟩⟩] } }]
def summary : Bound := .exactZero
def producerEvent : Nat := 5738
def resultEvent : Nat := 5739
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual resultEvent
theorem terminalAt (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, rightRootOwnerAtSelector214 selector selectorLower selectorUpper _, ?_⟩
  refine ⟨.scale (.predecessor 0 5736 .coefficient) (.value (.predecessor 1 5737 .coefficient)), 0, .scale (.predecessor 0 5736 .coefficient) (.value (.predecessor 1 5737 .coefficient)), ?_, ?_⟩
  · rfl
  · rfl
theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness
    (terminalAt selector selectorLower selectorUpper)

end SemanticRightRootResult5739

namespace SemanticRightRootResult5742

def owner : Owner := ⟨.program ⟨214⟩, ⟨6720⟩⟩
def rawTerms : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6720⟩⟩] } }]
def summary : Bound := .exactZero
def producerEvent : Nat := 5741
def resultEvent : Nat := 5742
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

end SemanticRightRootResult5742

namespace SemanticRightRootResult5747

def leftRaw : List Term := SemanticRightRootResult5742.rawTerms
def rightRaw : List Term := SemanticRightRootResult5739.rawTerms
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6720⟩⟩, ⟨.program ⟨214⟩, ⟨6643⟩⟩] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def owner : Owner := ⟨.program ⟨214⟩, ⟨7637⟩⟩
def rawTerms : List Term := outputRaw
def summary : Bound := .exactZero
def resultEvent : Nat := 5747
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
  SemanticRightRootResult5742.actual selector witness *
    SemanticRightRootResult5739.actual selector witness
theorem claimOfHistory (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841)
    (expressionAt : document.expressions.lookup 7637 =
      some Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Expression029.ExpressionRow7637)
    (leftPredecessorAt : (history.lookup 5743).map AnnotatedEvent.event =
      some (.predecessor owner 0 ⟨6720⟩ 5742))
    (rightPredecessorAt : (history.lookup 5744).map AnnotatedEvent.event =
      some (.predecessor owner 1 ⟨6644⟩ 5739))
    (ruleAt : (history.lookup 5745).map AnnotatedEvent.event =
      some (.boundTransfer owner (.product (.predecessor 0 5743 .coefficient) (.predecessor 1 5744 .coefficient) ⟨false, false, none, none, none⟩)))
    (leftClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult5742.resultEvent
      SemanticRightRootResult5742.owner
      (SemanticRightRootResult5742.actual selector witness)
      SemanticRightRootResult5742.rawTerms
      SemanticRightRootResult5742.summary)
    (rightClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult5739.resultEvent
      SemanticRightRootResult5739.owner
      (SemanticRightRootResult5739.actual selector witness)
      SemanticRightRootResult5739.rawTerms
      SemanticRightRootResult5739.summary)
    (outputAt : (history.lookup resultEvent).map AnnotatedEvent.event =
      some (.resultExact owner rawTerms summary)) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  refine ⟨outputAt, ?_⟩
  exact exactValueClaim_product_of_mod_zero 100418593683253592432016548326729029359133068138294319235841 witness.env
    (SemanticRightRootResult5742.actual selector witness)
    (SemanticRightRootResult5739.actual selector witness) left right output
    (by simpa [left, leftRaw, SemanticRightRootResult5742.summary] using leftClaim.claim)
    (by simpa [right, rightRaw, SemanticRightRootResult5739.summary] using rightClaim.claim)
    (resultSound witness.env) (by decide)

theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply claimOfHistory selector witness (by rfl) (by rfl) (by rfl) (by rfl)
  · exact SemanticRightRootResult5742.claimSound
      selector selectorLower selectorUpper witness
  · exact SemanticRightRootResult5739.claimSound
      selector selectorLower selectorUpper witness
  · rfl

end SemanticRightRootResult5747

namespace SemanticRightRootResult5759

def owner : Owner := ⟨.program ⟨214⟩, ⟨6648⟩⟩
def rawTerms : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6647⟩⟩] } }]
def summary : Bound := .exactZero
def producerEvent : Nat := 5758
def resultEvent : Nat := 5759
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual resultEvent
theorem terminalAt (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, rightRootOwnerAtSelector214 selector selectorLower selectorUpper _, ?_⟩
  refine ⟨.scale (.predecessor 0 5756 .coefficient) (.value (.predecessor 1 5757 .coefficient)), 0, .scale (.predecessor 0 5756 .coefficient) (.value (.predecessor 1 5757 .coefficient)), ?_, ?_⟩
  · rfl
  · rfl
theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness
    (terminalAt selector selectorLower selectorUpper)

end SemanticRightRootResult5759

namespace SemanticRightRootResult5762

def owner : Owner := ⟨.program ⟨214⟩, ⟨6718⟩⟩
def rawTerms : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6718⟩⟩] } }]
def summary : Bound := .exactZero
def producerEvent : Nat := 5761
def resultEvent : Nat := 5762
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

end SemanticRightRootResult5762

namespace SemanticRightRootResult5767

def leftRaw : List Term := SemanticRightRootResult5762.rawTerms
def rightRaw : List Term := SemanticRightRootResult5759.rawTerms
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6718⟩⟩, ⟨.program ⟨214⟩, ⟨6647⟩⟩] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def owner : Owner := ⟨.program ⟨214⟩, ⟨7636⟩⟩
def rawTerms : List Term := outputRaw
def summary : Bound := .exactZero
def resultEvent : Nat := 5767
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
  SemanticRightRootResult5762.actual selector witness *
    SemanticRightRootResult5759.actual selector witness
theorem claimOfHistory (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841)
    (expressionAt : document.expressions.lookup 7636 =
      some Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Expression029.ExpressionRow7636)
    (leftPredecessorAt : (history.lookup 5763).map AnnotatedEvent.event =
      some (.predecessor owner 0 ⟨6718⟩ 5762))
    (rightPredecessorAt : (history.lookup 5764).map AnnotatedEvent.event =
      some (.predecessor owner 1 ⟨6648⟩ 5759))
    (ruleAt : (history.lookup 5765).map AnnotatedEvent.event =
      some (.boundTransfer owner (.product (.predecessor 0 5763 .coefficient) (.predecessor 1 5764 .coefficient) ⟨false, false, none, none, none⟩)))
    (leftClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult5762.resultEvent
      SemanticRightRootResult5762.owner
      (SemanticRightRootResult5762.actual selector witness)
      SemanticRightRootResult5762.rawTerms
      SemanticRightRootResult5762.summary)
    (rightClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult5759.resultEvent
      SemanticRightRootResult5759.owner
      (SemanticRightRootResult5759.actual selector witness)
      SemanticRightRootResult5759.rawTerms
      SemanticRightRootResult5759.summary)
    (outputAt : (history.lookup resultEvent).map AnnotatedEvent.event =
      some (.resultExact owner rawTerms summary)) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  refine ⟨outputAt, ?_⟩
  exact exactValueClaim_product_of_mod_zero 100418593683253592432016548326729029359133068138294319235841 witness.env
    (SemanticRightRootResult5762.actual selector witness)
    (SemanticRightRootResult5759.actual selector witness) left right output
    (by simpa [left, leftRaw, SemanticRightRootResult5762.summary] using leftClaim.claim)
    (by simpa [right, rightRaw, SemanticRightRootResult5759.summary] using rightClaim.claim)
    (resultSound witness.env) (by decide)

theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply claimOfHistory selector witness (by rfl) (by rfl) (by rfl) (by rfl)
  · exact SemanticRightRootResult5762.claimSound
      selector selectorLower selectorUpper witness
  · exact SemanticRightRootResult5759.claimSound
      selector selectorLower selectorUpper witness
  · rfl

end SemanticRightRootResult5767

namespace SemanticRightRootResult5779

def owner : Owner := ⟨.program ⟨214⟩, ⟨6650⟩⟩
def rawTerms : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6649⟩⟩] } }]
def summary : Bound := .exactZero
def producerEvent : Nat := 5778
def resultEvent : Nat := 5779
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual resultEvent
theorem terminalAt (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, rightRootOwnerAtSelector214 selector selectorLower selectorUpper _, ?_⟩
  refine ⟨.scale (.predecessor 0 5776 .coefficient) (.value (.predecessor 1 5777 .coefficient)), 0, .scale (.predecessor 0 5776 .coefficient) (.value (.predecessor 1 5777 .coefficient)), ?_, ?_⟩
  · rfl
  · rfl
theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness
    (terminalAt selector selectorLower selectorUpper)

end SemanticRightRootResult5779

namespace SemanticRightRootResult5782

def owner : Owner := ⟨.program ⟨214⟩, ⟨6716⟩⟩
def rawTerms : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6716⟩⟩] } }]
def summary : Bound := .exactZero
def producerEvent : Nat := 5781
def resultEvent : Nat := 5782
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

end SemanticRightRootResult5782

namespace SemanticRightRootResult5787

def leftRaw : List Term := SemanticRightRootResult5782.rawTerms
def rightRaw : List Term := SemanticRightRootResult5779.rawTerms
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6716⟩⟩, ⟨.program ⟨214⟩, ⟨6649⟩⟩] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def owner : Owner := ⟨.program ⟨214⟩, ⟨7635⟩⟩
def rawTerms : List Term := outputRaw
def summary : Bound := .exactZero
def resultEvent : Nat := 5787
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
  SemanticRightRootResult5782.actual selector witness *
    SemanticRightRootResult5779.actual selector witness
theorem claimOfHistory (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841)
    (expressionAt : document.expressions.lookup 7635 =
      some Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Expression029.ExpressionRow7635)
    (leftPredecessorAt : (history.lookup 5783).map AnnotatedEvent.event =
      some (.predecessor owner 0 ⟨6716⟩ 5782))
    (rightPredecessorAt : (history.lookup 5784).map AnnotatedEvent.event =
      some (.predecessor owner 1 ⟨6650⟩ 5779))
    (ruleAt : (history.lookup 5785).map AnnotatedEvent.event =
      some (.boundTransfer owner (.product (.predecessor 0 5783 .coefficient) (.predecessor 1 5784 .coefficient) ⟨false, false, none, none, none⟩)))
    (leftClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult5782.resultEvent
      SemanticRightRootResult5782.owner
      (SemanticRightRootResult5782.actual selector witness)
      SemanticRightRootResult5782.rawTerms
      SemanticRightRootResult5782.summary)
    (rightClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult5779.resultEvent
      SemanticRightRootResult5779.owner
      (SemanticRightRootResult5779.actual selector witness)
      SemanticRightRootResult5779.rawTerms
      SemanticRightRootResult5779.summary)
    (outputAt : (history.lookup resultEvent).map AnnotatedEvent.event =
      some (.resultExact owner rawTerms summary)) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  refine ⟨outputAt, ?_⟩
  exact exactValueClaim_product_of_mod_zero 100418593683253592432016548326729029359133068138294319235841 witness.env
    (SemanticRightRootResult5782.actual selector witness)
    (SemanticRightRootResult5779.actual selector witness) left right output
    (by simpa [left, leftRaw, SemanticRightRootResult5782.summary] using leftClaim.claim)
    (by simpa [right, rightRaw, SemanticRightRootResult5779.summary] using rightClaim.claim)
    (resultSound witness.env) (by decide)

theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply claimOfHistory selector witness (by rfl) (by rfl) (by rfl) (by rfl)
  · exact SemanticRightRootResult5782.claimSound
      selector selectorLower selectorUpper witness
  · exact SemanticRightRootResult5779.claimSound
      selector selectorLower selectorUpper witness
  · rfl

end SemanticRightRootResult5787

namespace SemanticRightRootResult5799

def owner : Owner := ⟨.program ⟨214⟩, ⟨6656⟩⟩
def rawTerms : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6655⟩⟩] } }]
def summary : Bound := .exactZero
def producerEvent : Nat := 5798
def resultEvent : Nat := 5799
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual resultEvent
theorem terminalAt (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, rightRootOwnerAtSelector214 selector selectorLower selectorUpper _, ?_⟩
  refine ⟨.scale (.predecessor 0 5796 .coefficient) (.value (.predecessor 1 5797 .coefficient)), 0, .scale (.predecessor 0 5796 .coefficient) (.value (.predecessor 1 5797 .coefficient)), ?_, ?_⟩
  · rfl
  · rfl
theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness
    (terminalAt selector selectorLower selectorUpper)

end SemanticRightRootResult5799

namespace SemanticRightRootResult5802

def owner : Owner := ⟨.program ⟨214⟩, ⟨6714⟩⟩
def rawTerms : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6714⟩⟩] } }]
def summary : Bound := .exactZero
def producerEvent : Nat := 5801
def resultEvent : Nat := 5802
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

end SemanticRightRootResult5802

namespace SemanticRightRootResult5807

def leftRaw : List Term := SemanticRightRootResult5802.rawTerms
def rightRaw : List Term := SemanticRightRootResult5799.rawTerms
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6714⟩⟩, ⟨.program ⟨214⟩, ⟨6655⟩⟩] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def owner : Owner := ⟨.program ⟨214⟩, ⟨7634⟩⟩
def rawTerms : List Term := outputRaw
def summary : Bound := .exactZero
def resultEvent : Nat := 5807
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
  SemanticRightRootResult5802.actual selector witness *
    SemanticRightRootResult5799.actual selector witness
theorem claimOfHistory (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841)
    (expressionAt : document.expressions.lookup 7634 =
      some Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Expression029.ExpressionRow7634)
    (leftPredecessorAt : (history.lookup 5803).map AnnotatedEvent.event =
      some (.predecessor owner 0 ⟨6714⟩ 5802))
    (rightPredecessorAt : (history.lookup 5804).map AnnotatedEvent.event =
      some (.predecessor owner 1 ⟨6656⟩ 5799))
    (ruleAt : (history.lookup 5805).map AnnotatedEvent.event =
      some (.boundTransfer owner (.product (.predecessor 0 5803 .coefficient) (.predecessor 1 5804 .coefficient) ⟨false, false, none, none, none⟩)))
    (leftClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult5802.resultEvent
      SemanticRightRootResult5802.owner
      (SemanticRightRootResult5802.actual selector witness)
      SemanticRightRootResult5802.rawTerms
      SemanticRightRootResult5802.summary)
    (rightClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult5799.resultEvent
      SemanticRightRootResult5799.owner
      (SemanticRightRootResult5799.actual selector witness)
      SemanticRightRootResult5799.rawTerms
      SemanticRightRootResult5799.summary)
    (outputAt : (history.lookup resultEvent).map AnnotatedEvent.event =
      some (.resultExact owner rawTerms summary)) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  refine ⟨outputAt, ?_⟩
  exact exactValueClaim_product_of_mod_zero 100418593683253592432016548326729029359133068138294319235841 witness.env
    (SemanticRightRootResult5802.actual selector witness)
    (SemanticRightRootResult5799.actual selector witness) left right output
    (by simpa [left, leftRaw, SemanticRightRootResult5802.summary] using leftClaim.claim)
    (by simpa [right, rightRaw, SemanticRightRootResult5799.summary] using rightClaim.claim)
    (resultSound witness.env) (by decide)

theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply claimOfHistory selector witness (by rfl) (by rfl) (by rfl) (by rfl)
  · exact SemanticRightRootResult5802.claimSound
      selector selectorLower selectorUpper witness
  · exact SemanticRightRootResult5799.claimSound
      selector selectorLower selectorUpper witness
  · rfl

end SemanticRightRootResult5807

namespace SemanticRightRootResult5819

def owner : Owner := ⟨.program ⟨214⟩, ⟨6664⟩⟩
def rawTerms : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6663⟩⟩] } }]
def summary : Bound := .exactZero
def producerEvent : Nat := 5818
def resultEvent : Nat := 5819
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual resultEvent
theorem terminalAt (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, rightRootOwnerAtSelector214 selector selectorLower selectorUpper _, ?_⟩
  refine ⟨.scale (.predecessor 0 5816 .coefficient) (.value (.predecessor 1 5817 .coefficient)), 0, .scale (.predecessor 0 5816 .coefficient) (.value (.predecessor 1 5817 .coefficient)), ?_, ?_⟩
  · rfl
  · rfl
theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness
    (terminalAt selector selectorLower selectorUpper)

end SemanticRightRootResult5819

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
