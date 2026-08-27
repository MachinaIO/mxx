import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticRightRootShard002

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open TallSecurity0ABI
open TallSemantics

namespace SemanticRightRootResult681

def leftRaw : List Term := SemanticRightRootResult676.rawTerms
def rightRaw : List Term := SemanticRightRootResult673.rawTerms
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6407⟩⟩, ⟨.program ⟨214⟩, ⟨17846⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def owner : Owner := ⟨.program ⟨214⟩, ⟨17847⟩⟩
def rawTerms : List Term := outputRaw
def summary : Bound := .exactZero
def resultEvent : Nat := 681
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
  SemanticRightRootResult676.actual selector witness *
    SemanticRightRootResult673.actual selector witness
theorem claimOfHistory (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841)
    (expressionAt : document.expressions.lookup 17847 =
      some Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Expression069.ExpressionRow17847)
    (leftPredecessorAt : (history.lookup 677).map AnnotatedEvent.event =
      some (.predecessor owner 0 ⟨17846⟩ 676))
    (rightPredecessorAt : (history.lookup 678).map AnnotatedEvent.event =
      some (.predecessor owner 1 ⟨6407⟩ 673))
    (ruleAt : (history.lookup 679).map AnnotatedEvent.event =
      some (.boundTransfer owner (.product (.predecessor 0 677 .coefficient) (.predecessor 1 678 .coefficient) ⟨true, true, none, some 1, some 1⟩)))
    (leftClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult676.resultEvent
      SemanticRightRootResult676.owner
      (SemanticRightRootResult676.actual selector witness)
      SemanticRightRootResult676.rawTerms
      SemanticRightRootResult676.summary)
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
    (SemanticRightRootResult676.actual selector witness)
    (SemanticRightRootResult673.actual selector witness) left right output
    (by simpa [left, leftRaw, SemanticRightRootResult676.summary] using leftClaim.claim)
    (by simpa [right, rightRaw, SemanticRightRootResult673.summary] using rightClaim.claim)
    (resultSound witness.env) (by decide)

theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply claimOfHistory selector witness (by rfl) (by rfl) (by rfl) (by rfl)
  · exact SemanticRightRootResult676.claimSound
      selector selectorLower selectorUpper witness
  · exact SemanticRightRootResult673.claimSound
      selector selectorLower selectorUpper witness
  · rfl

end SemanticRightRootResult681

namespace SemanticRightRootResult683

def owner : Owner := ⟨.program ⟨214⟩, ⟨6427⟩⟩
def rawTerms : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6427⟩⟩], orderedFactors := [] } }]
def summary : Bound := .exactZero
def producerEvent : Nat := 682
def resultEvent : Nat := 683
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual resultEvent
theorem terminalAt (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, rightRootOwnerAtSelector214 selector selectorLower selectorUpper _, ?_⟩
  refine ⟨.authority (.factStore), 0, .authorityFactStore, ?_, ?_⟩
  · rfl
  · rfl
theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness
    (terminalAt selector selectorLower selectorUpper)

end SemanticRightRootResult683

namespace SemanticRightRootResult686

def owner : Owner := ⟨.program ⟨214⟩, ⟨15536⟩⟩
def rawTerms : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15536⟩⟩], orderedFactors := [] } }]
def summary : Bound := .exactZero
def producerEvent : Nat := 685
def resultEvent : Nat := 686
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

end SemanticRightRootResult686

namespace SemanticRightRootResult691

def leftRaw : List Term := SemanticRightRootResult686.rawTerms
def rightRaw : List Term := SemanticRightRootResult683.rawTerms
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6427⟩⟩, ⟨.program ⟨214⟩, ⟨15536⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def owner : Owner := ⟨.program ⟨214⟩, ⟨15537⟩⟩
def rawTerms : List Term := outputRaw
def summary : Bound := .exactZero
def resultEvent : Nat := 691
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
  SemanticRightRootResult686.actual selector witness *
    SemanticRightRootResult683.actual selector witness
theorem claimOfHistory (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841)
    (expressionAt : document.expressions.lookup 15537 =
      some Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Expression060.ExpressionRow15537)
    (leftPredecessorAt : (history.lookup 687).map AnnotatedEvent.event =
      some (.predecessor owner 0 ⟨15536⟩ 686))
    (rightPredecessorAt : (history.lookup 688).map AnnotatedEvent.event =
      some (.predecessor owner 1 ⟨6427⟩ 683))
    (ruleAt : (history.lookup 689).map AnnotatedEvent.event =
      some (.boundTransfer owner (.product (.predecessor 0 687 .coefficient) (.predecessor 1 688 .coefficient) ⟨true, true, none, some 1, some 1⟩)))
    (leftClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult686.resultEvent
      SemanticRightRootResult686.owner
      (SemanticRightRootResult686.actual selector witness)
      SemanticRightRootResult686.rawTerms
      SemanticRightRootResult686.summary)
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
    (SemanticRightRootResult686.actual selector witness)
    (SemanticRightRootResult683.actual selector witness) left right output
    (by simpa [left, leftRaw, SemanticRightRootResult686.summary] using leftClaim.claim)
    (by simpa [right, rightRaw, SemanticRightRootResult683.summary] using rightClaim.claim)
    (resultSound witness.env) (by decide)

theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply claimOfHistory selector witness (by rfl) (by rfl) (by rfl) (by rfl)
  · exact SemanticRightRootResult686.claimSound
      selector selectorLower selectorUpper witness
  · exact SemanticRightRootResult683.claimSound
      selector selectorLower selectorUpper witness
  · rfl

end SemanticRightRootResult691

namespace SemanticRightRootResult693

def owner : Owner := ⟨.program ⟨214⟩, ⟨6452⟩⟩
def rawTerms : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6452⟩⟩], orderedFactors := [] } }]
def summary : Bound := .exactZero
def producerEvent : Nat := 692
def resultEvent : Nat := 693
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual resultEvent
theorem terminalAt (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, rightRootOwnerAtSelector214 selector selectorLower selectorUpper _, ?_⟩
  refine ⟨.authority (.factStore), 0, .authorityFactStore, ?_, ?_⟩
  · rfl
  · rfl
theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness
    (terminalAt selector selectorLower selectorUpper)

end SemanticRightRootResult693

namespace SemanticRightRootResult696

def owner : Owner := ⟨.program ⟨214⟩, ⟨15228⟩⟩
def rawTerms : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15228⟩⟩], orderedFactors := [] } }]
def summary : Bound := .exactZero
def producerEvent : Nat := 695
def resultEvent : Nat := 696
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

end SemanticRightRootResult696

namespace SemanticRightRootResult701

def leftRaw : List Term := SemanticRightRootResult696.rawTerms
def rightRaw : List Term := SemanticRightRootResult693.rawTerms
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6452⟩⟩, ⟨.program ⟨214⟩, ⟨15228⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def owner : Owner := ⟨.program ⟨214⟩, ⟨15229⟩⟩
def rawTerms : List Term := outputRaw
def summary : Bound := .exactZero
def resultEvent : Nat := 701
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
  SemanticRightRootResult696.actual selector witness *
    SemanticRightRootResult693.actual selector witness
theorem claimOfHistory (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841)
    (expressionAt : document.expressions.lookup 15229 =
      some Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Expression059.ExpressionRow15229)
    (leftPredecessorAt : (history.lookup 697).map AnnotatedEvent.event =
      some (.predecessor owner 0 ⟨15228⟩ 696))
    (rightPredecessorAt : (history.lookup 698).map AnnotatedEvent.event =
      some (.predecessor owner 1 ⟨6452⟩ 693))
    (ruleAt : (history.lookup 699).map AnnotatedEvent.event =
      some (.boundTransfer owner (.product (.predecessor 0 697 .coefficient) (.predecessor 1 698 .coefficient) ⟨true, true, none, some 1, some 1⟩)))
    (leftClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult696.resultEvent
      SemanticRightRootResult696.owner
      (SemanticRightRootResult696.actual selector witness)
      SemanticRightRootResult696.rawTerms
      SemanticRightRootResult696.summary)
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
    (SemanticRightRootResult696.actual selector witness)
    (SemanticRightRootResult693.actual selector witness) left right output
    (by simpa [left, leftRaw, SemanticRightRootResult696.summary] using leftClaim.claim)
    (by simpa [right, rightRaw, SemanticRightRootResult693.summary] using rightClaim.claim)
    (resultSound witness.env) (by decide)

theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply claimOfHistory selector witness (by rfl) (by rfl) (by rfl) (by rfl)
  · exact SemanticRightRootResult696.claimSound
      selector selectorLower selectorUpper witness
  · exact SemanticRightRootResult693.claimSound
      selector selectorLower selectorUpper witness
  · rfl

end SemanticRightRootResult701

namespace SemanticRightRootResult703

def owner : Owner := ⟨.program ⟨214⟩, ⟨6475⟩⟩
def rawTerms : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6475⟩⟩], orderedFactors := [] } }]
def summary : Bound := .exactZero
def producerEvent : Nat := 702
def resultEvent : Nat := 703
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual resultEvent
theorem terminalAt (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, rightRootOwnerAtSelector214 selector selectorLower selectorUpper _, ?_⟩
  refine ⟨.authority (.factStore), 0, .authorityFactStore, ?_, ?_⟩
  · rfl
  · rfl
theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness
    (terminalAt selector selectorLower selectorUpper)

end SemanticRightRootResult703

namespace SemanticRightRootResult706

def owner : Owner := ⟨.program ⟨214⟩, ⟨15067⟩⟩
def rawTerms : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15067⟩⟩], orderedFactors := [] } }]
def summary : Bound := .exactZero
def producerEvent : Nat := 705
def resultEvent : Nat := 706
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

end SemanticRightRootResult706

namespace SemanticRightRootResult711

def leftRaw : List Term := SemanticRightRootResult706.rawTerms
def rightRaw : List Term := SemanticRightRootResult703.rawTerms
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨15067⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def owner : Owner := ⟨.program ⟨214⟩, ⟨15068⟩⟩
def rawTerms : List Term := outputRaw
def summary : Bound := .exactZero
def resultEvent : Nat := 711
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
  SemanticRightRootResult706.actual selector witness *
    SemanticRightRootResult703.actual selector witness
theorem claimOfHistory (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841)
    (expressionAt : document.expressions.lookup 15068 =
      some Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Expression058.ExpressionRow15068)
    (leftPredecessorAt : (history.lookup 707).map AnnotatedEvent.event =
      some (.predecessor owner 0 ⟨15067⟩ 706))
    (rightPredecessorAt : (history.lookup 708).map AnnotatedEvent.event =
      some (.predecessor owner 1 ⟨6475⟩ 703))
    (ruleAt : (history.lookup 709).map AnnotatedEvent.event =
      some (.boundTransfer owner (.product (.predecessor 0 707 .coefficient) (.predecessor 1 708 .coefficient) ⟨true, true, none, some 1, some 1⟩)))
    (leftClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult706.resultEvent
      SemanticRightRootResult706.owner
      (SemanticRightRootResult706.actual selector witness)
      SemanticRightRootResult706.rawTerms
      SemanticRightRootResult706.summary)
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
    (SemanticRightRootResult706.actual selector witness)
    (SemanticRightRootResult703.actual selector witness) left right output
    (by simpa [left, leftRaw, SemanticRightRootResult706.summary] using leftClaim.claim)
    (by simpa [right, rightRaw, SemanticRightRootResult703.summary] using rightClaim.claim)
    (resultSound witness.env) (by decide)

theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply claimOfHistory selector witness (by rfl) (by rfl) (by rfl) (by rfl)
  · exact SemanticRightRootResult706.claimSound
      selector selectorLower selectorUpper witness
  · exact SemanticRightRootResult703.claimSound
      selector selectorLower selectorUpper witness
  · rfl

end SemanticRightRootResult711

namespace SemanticRightRootResult713

def owner : Owner := ⟨.program ⟨214⟩, ⟨6495⟩⟩
def rawTerms : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6495⟩⟩], orderedFactors := [] } }]
def summary : Bound := .exactZero
def producerEvent : Nat := 712
def resultEvent : Nat := 713
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual resultEvent
theorem terminalAt (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, rightRootOwnerAtSelector214 selector selectorLower selectorUpper _, ?_⟩
  refine ⟨.authority (.factStore), 0, .authorityFactStore, ?_, ?_⟩
  · rfl
  · rfl
theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness
    (terminalAt selector selectorLower selectorUpper)

end SemanticRightRootResult713

namespace SemanticRightRootResult716

def owner : Owner := ⟨.program ⟨214⟩, ⟨14906⟩⟩
def rawTerms : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨14906⟩⟩], orderedFactors := [] } }]
def summary : Bound := .exactZero
def producerEvent : Nat := 715
def resultEvent : Nat := 716
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

end SemanticRightRootResult716

namespace SemanticRightRootResult721

def leftRaw : List Term := SemanticRightRootResult716.rawTerms
def rightRaw : List Term := SemanticRightRootResult713.rawTerms
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨14906⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def owner : Owner := ⟨.program ⟨214⟩, ⟨14907⟩⟩
def rawTerms : List Term := outputRaw
def summary : Bound := .exactZero
def resultEvent : Nat := 721
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
  SemanticRightRootResult716.actual selector witness *
    SemanticRightRootResult713.actual selector witness
theorem claimOfHistory (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841)
    (expressionAt : document.expressions.lookup 14907 =
      some Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Expression058.ExpressionRow14907)
    (leftPredecessorAt : (history.lookup 717).map AnnotatedEvent.event =
      some (.predecessor owner 0 ⟨14906⟩ 716))
    (rightPredecessorAt : (history.lookup 718).map AnnotatedEvent.event =
      some (.predecessor owner 1 ⟨6495⟩ 713))
    (ruleAt : (history.lookup 719).map AnnotatedEvent.event =
      some (.boundTransfer owner (.product (.predecessor 0 717 .coefficient) (.predecessor 1 718 .coefficient) ⟨true, true, none, some 1, some 1⟩)))
    (leftClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult716.resultEvent
      SemanticRightRootResult716.owner
      (SemanticRightRootResult716.actual selector witness)
      SemanticRightRootResult716.rawTerms
      SemanticRightRootResult716.summary)
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
    (SemanticRightRootResult716.actual selector witness)
    (SemanticRightRootResult713.actual selector witness) left right output
    (by simpa [left, leftRaw, SemanticRightRootResult716.summary] using leftClaim.claim)
    (by simpa [right, rightRaw, SemanticRightRootResult713.summary] using rightClaim.claim)
    (resultSound witness.env) (by decide)

theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply claimOfHistory selector witness (by rfl) (by rfl) (by rfl) (by rfl)
  · exact SemanticRightRootResult716.claimSound
      selector selectorLower selectorUpper witness
  · exact SemanticRightRootResult713.claimSound
      selector selectorLower selectorUpper witness
  · rfl

end SemanticRightRootResult721

namespace SemanticRightRootResult723

def owner : Owner := ⟨.program ⟨214⟩, ⟨6378⟩⟩
def rawTerms : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6378⟩⟩], orderedFactors := [] } }]
def summary : Bound := .exactZero
def producerEvent : Nat := 722
def resultEvent : Nat := 723
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual resultEvent
theorem terminalAt (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, rightRootOwnerAtSelector214 selector selectorLower selectorUpper _, ?_⟩
  refine ⟨.authority (.factStore), 0, .authorityFactStore, ?_, ?_⟩
  · rfl
  · rfl
theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness
    (terminalAt selector selectorLower selectorUpper)

end SemanticRightRootResult723

namespace SemanticRightRootResult728

def leftRaw : List Term := SemanticRightRootResult723.rawTerms
def rightRaw : List Term := SemanticRightRootResult723.rawTerms
def outputRaw : List Term := []
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def owner : Owner := ⟨.program ⟨214⟩, ⟨6379⟩⟩
def rawTerms : List Term := outputRaw
def summary : Bound := .exactZero
def resultEvent : Nat := 728
theorem resultAgreement : CanonicalAgreement output (subtract left right) := by
  decide +kernel

theorem resultSound (env : Env Owner) :
    evalPolynomial env output = evalPolynomial env left - evalPolynomial env right := by
  exact subCanonicalResultSound env left right output resultAgreement

def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  SemanticRightRootResult723.actual selector witness -
    SemanticRightRootResult723.actual selector witness
theorem claimOfHistory (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841)
    (expressionAt : document.expressions.lookup 6379 =
      some Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Expression024.ExpressionRow6379)
    (leftPredecessorAt : (history.lookup 724).map AnnotatedEvent.event =
      some (.predecessor owner 0 ⟨6378⟩ 723))
    (rightPredecessorAt : (history.lookup 725).map AnnotatedEvent.event =
      some (.predecessor owner 1 ⟨6378⟩ 723))
    (ruleAt : (history.lookup 726).map AnnotatedEvent.event =
      some (.boundTransfer owner (.sum [.predecessor 0 724 .coefficient, .predecessor 1 725 .coefficient])))
    (leftClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult723.resultEvent
      SemanticRightRootResult723.owner
      (SemanticRightRootResult723.actual selector witness)
      SemanticRightRootResult723.rawTerms
      SemanticRightRootResult723.summary)
    (rightClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult723.resultEvent
      SemanticRightRootResult723.owner
      (SemanticRightRootResult723.actual selector witness)
      SemanticRightRootResult723.rawTerms
      SemanticRightRootResult723.summary)
    (outputAt : (history.lookup resultEvent).map AnnotatedEvent.event =
      some (.resultExact owner rawTerms summary)) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  refine ⟨outputAt, ?_⟩
  exact exactValueClaim_sub_exactZero_of_mod_zero 100418593683253592432016548326729029359133068138294319235841 witness.env
    (SemanticRightRootResult723.actual selector witness)
    (SemanticRightRootResult723.actual selector witness) left right output
    (by simpa [left, leftRaw, SemanticRightRootResult723.summary] using leftClaim.claim)
    (by simpa [right, rightRaw, SemanticRightRootResult723.summary] using rightClaim.claim)
    (resultSound witness.env) (by decide)

theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply claimOfHistory selector witness (by rfl) (by rfl) (by rfl) (by rfl)
  · exact SemanticRightRootResult723.claimSound
      selector selectorLower selectorUpper witness
  · exact SemanticRightRootResult723.claimSound
      selector selectorLower selectorUpper witness
  · rfl

end SemanticRightRootResult728

namespace SemanticRightRootResult732

def leftRaw : List Term := SemanticRightRootResult728.rawTerms
def rightRaw : List Term := SemanticRightRootResult721.rawTerms
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨14906⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def owner : Owner := ⟨.program ⟨214⟩, ⟨14908⟩⟩
def rawTerms : List Term := outputRaw
def summary : Bound := .exactZero
def resultEvent : Nat := 732
theorem resultAgreement : CanonicalAgreement output (add left right) := by
  decide +kernel

theorem resultSound (env : Env Owner) :
    evalPolynomial env output = evalPolynomial env left + evalPolynomial env right := by
  exact addCanonicalResultSound env left right output resultAgreement

def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  SemanticRightRootResult728.actual selector witness +
    SemanticRightRootResult721.actual selector witness
theorem claimOfHistory (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841)
    (expressionAt : document.expressions.lookup 14908 =
      some Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Expression058.ExpressionRow14908)
    (leftPredecessorAt : (history.lookup 729).map AnnotatedEvent.event =
      some (.predecessor owner 0 ⟨6379⟩ 728))
    (rightPredecessorAt : (history.lookup 730).map AnnotatedEvent.event =
      some (.predecessor owner 1 ⟨14907⟩ 721))
    (ruleAt : (history.lookup 731).map AnnotatedEvent.event =
      some (.boundTransfer owner (.sum [.predecessor 0 729 .coefficient, .predecessor 1 730 .coefficient])))
    (leftClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult728.resultEvent
      SemanticRightRootResult728.owner
      (SemanticRightRootResult728.actual selector witness)
      SemanticRightRootResult728.rawTerms
      SemanticRightRootResult728.summary)
    (rightClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult721.resultEvent
      SemanticRightRootResult721.owner
      (SemanticRightRootResult721.actual selector witness)
      SemanticRightRootResult721.rawTerms
      SemanticRightRootResult721.summary)
    (outputAt : (history.lookup resultEvent).map AnnotatedEvent.event =
      some (.resultExact owner rawTerms summary)) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  refine ⟨outputAt, ?_⟩
  exact exactValueClaim_add_of_mod_zero 100418593683253592432016548326729029359133068138294319235841 witness.env
    (SemanticRightRootResult728.actual selector witness)
    (SemanticRightRootResult721.actual selector witness) left right output
    (by simpa [left, leftRaw, SemanticRightRootResult728.summary] using leftClaim.claim)
    (by simpa [right, rightRaw, SemanticRightRootResult721.summary] using rightClaim.claim)
    (resultSound witness.env) (by decide)

theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply claimOfHistory selector witness (by rfl) (by rfl) (by rfl) (by rfl)
  · exact SemanticRightRootResult728.claimSound
      selector selectorLower selectorUpper witness
  · exact SemanticRightRootResult721.claimSound
      selector selectorLower selectorUpper witness
  · rfl

end SemanticRightRootResult732

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
