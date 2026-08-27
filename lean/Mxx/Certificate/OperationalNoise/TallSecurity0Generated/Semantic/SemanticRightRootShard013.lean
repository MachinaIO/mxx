import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticRightRootShard012

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open TallSecurity0ABI
open TallSemantics

namespace SemanticRightRootResult2847

def owner : Owner := ⟨.program ⟨214⟩, ⟨17498⟩⟩
def rawTerms : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17498⟩⟩], orderedFactors := [] } }]
def summary : Bound := .exactZero
def producerEvent : Nat := 2846
def resultEvent : Nat := 2847
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

end SemanticRightRootResult2847

namespace SemanticRightRootResult2852

def leftRaw : List Term := SemanticRightRootResult2847.rawTerms
def rightRaw : List Term := SemanticRightRootResult563.rawTerms
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6449⟩⟩, ⟨.program ⟨214⟩, ⟨17498⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def owner : Owner := ⟨.program ⟨214⟩, ⟨17499⟩⟩
def rawTerms : List Term := outputRaw
def summary : Bound := .exactZero
def resultEvent : Nat := 2852
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
  SemanticRightRootResult2847.actual selector witness *
    SemanticRightRootResult563.actual selector witness
theorem claimOfHistory (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841)
    (expressionAt : document.expressions.lookup 17499 =
      some Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Expression068.ExpressionRow17499)
    (leftPredecessorAt : (history.lookup 2848).map AnnotatedEvent.event =
      some (.predecessor owner 0 ⟨17498⟩ 2847))
    (rightPredecessorAt : (history.lookup 2849).map AnnotatedEvent.event =
      some (.predecessor owner 1 ⟨6449⟩ 563))
    (ruleAt : (history.lookup 2850).map AnnotatedEvent.event =
      some (.boundTransfer owner (.product (.predecessor 0 2848 .coefficient) (.predecessor 1 2849 .coefficient) ⟨true, true, none, some 1, some 1⟩)))
    (leftClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult2847.resultEvent
      SemanticRightRootResult2847.owner
      (SemanticRightRootResult2847.actual selector witness)
      SemanticRightRootResult2847.rawTerms
      SemanticRightRootResult2847.summary)
    (rightClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult563.resultEvent
      SemanticRightRootResult563.owner
      (SemanticRightRootResult563.actual selector witness)
      SemanticRightRootResult563.rawTerms
      SemanticRightRootResult563.summary)
    (outputAt : (history.lookup resultEvent).map AnnotatedEvent.event =
      some (.resultExact owner rawTerms summary)) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  refine ⟨outputAt, ?_⟩
  exact exactValueClaim_product_of_mod_zero 100418593683253592432016548326729029359133068138294319235841 witness.env
    (SemanticRightRootResult2847.actual selector witness)
    (SemanticRightRootResult563.actual selector witness) left right output
    (by simpa [left, leftRaw, SemanticRightRootResult2847.summary] using leftClaim.claim)
    (by simpa [right, rightRaw, SemanticRightRootResult563.summary] using rightClaim.claim)
    (resultSound witness.env) (by decide)

theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply claimOfHistory selector witness (by rfl) (by rfl) (by rfl) (by rfl)
  · exact SemanticRightRootResult2847.claimSound
      selector selectorLower selectorUpper witness
  · exact SemanticRightRootResult563.claimSound
      selector selectorLower selectorUpper witness
  · rfl

end SemanticRightRootResult2852

namespace SemanticRightRootResult2855

def owner : Owner := ⟨.program ⟨214⟩, ⟨17722⟩⟩
def rawTerms : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17722⟩⟩], orderedFactors := [] } }]
def summary : Bound := .exactZero
def producerEvent : Nat := 2854
def resultEvent : Nat := 2855
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

end SemanticRightRootResult2855

namespace SemanticRightRootResult2860

def leftRaw : List Term := SemanticRightRootResult2855.rawTerms
def rightRaw : List Term := SemanticRightRootResult573.rawTerms
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6459⟩⟩, ⟨.program ⟨214⟩, ⟨17722⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def owner : Owner := ⟨.program ⟨214⟩, ⟨17723⟩⟩
def rawTerms : List Term := outputRaw
def summary : Bound := .exactZero
def resultEvent : Nat := 2860
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
  SemanticRightRootResult2855.actual selector witness *
    SemanticRightRootResult573.actual selector witness
theorem claimOfHistory (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841)
    (expressionAt : document.expressions.lookup 17723 =
      some Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Expression069.ExpressionRow17723)
    (leftPredecessorAt : (history.lookup 2856).map AnnotatedEvent.event =
      some (.predecessor owner 0 ⟨17722⟩ 2855))
    (rightPredecessorAt : (history.lookup 2857).map AnnotatedEvent.event =
      some (.predecessor owner 1 ⟨6459⟩ 573))
    (ruleAt : (history.lookup 2858).map AnnotatedEvent.event =
      some (.boundTransfer owner (.product (.predecessor 0 2856 .coefficient) (.predecessor 1 2857 .coefficient) ⟨true, true, none, some 1, some 1⟩)))
    (leftClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult2855.resultEvent
      SemanticRightRootResult2855.owner
      (SemanticRightRootResult2855.actual selector witness)
      SemanticRightRootResult2855.rawTerms
      SemanticRightRootResult2855.summary)
    (rightClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult573.resultEvent
      SemanticRightRootResult573.owner
      (SemanticRightRootResult573.actual selector witness)
      SemanticRightRootResult573.rawTerms
      SemanticRightRootResult573.summary)
    (outputAt : (history.lookup resultEvent).map AnnotatedEvent.event =
      some (.resultExact owner rawTerms summary)) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  refine ⟨outputAt, ?_⟩
  exact exactValueClaim_product_of_mod_zero 100418593683253592432016548326729029359133068138294319235841 witness.env
    (SemanticRightRootResult2855.actual selector witness)
    (SemanticRightRootResult573.actual selector witness) left right output
    (by simpa [left, leftRaw, SemanticRightRootResult2855.summary] using leftClaim.claim)
    (by simpa [right, rightRaw, SemanticRightRootResult573.summary] using rightClaim.claim)
    (resultSound witness.env) (by decide)

theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply claimOfHistory selector witness (by rfl) (by rfl) (by rfl) (by rfl)
  · exact SemanticRightRootResult2855.claimSound
      selector selectorLower selectorUpper witness
  · exact SemanticRightRootResult573.claimSound
      selector selectorLower selectorUpper witness
  · rfl

end SemanticRightRootResult2860

namespace SemanticRightRootResult2863

def owner : Owner := ⟨.program ⟨214⟩, ⟨17953⟩⟩
def rawTerms : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17953⟩⟩], orderedFactors := [] } }]
def summary : Bound := .exactZero
def producerEvent : Nat := 2862
def resultEvent : Nat := 2863
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

end SemanticRightRootResult2863

namespace SemanticRightRootResult2868

def leftRaw : List Term := SemanticRightRootResult2863.rawTerms
def rightRaw : List Term := SemanticRightRootResult583.rawTerms
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6467⟩⟩, ⟨.program ⟨214⟩, ⟨17953⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def owner : Owner := ⟨.program ⟨214⟩, ⟨17954⟩⟩
def rawTerms : List Term := outputRaw
def summary : Bound := .exactZero
def resultEvent : Nat := 2868
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
  SemanticRightRootResult2863.actual selector witness *
    SemanticRightRootResult583.actual selector witness
theorem claimOfHistory (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841)
    (expressionAt : document.expressions.lookup 17954 =
      some Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Expression070.ExpressionRow17954)
    (leftPredecessorAt : (history.lookup 2864).map AnnotatedEvent.event =
      some (.predecessor owner 0 ⟨17953⟩ 2863))
    (rightPredecessorAt : (history.lookup 2865).map AnnotatedEvent.event =
      some (.predecessor owner 1 ⟨6467⟩ 583))
    (ruleAt : (history.lookup 2866).map AnnotatedEvent.event =
      some (.boundTransfer owner (.product (.predecessor 0 2864 .coefficient) (.predecessor 1 2865 .coefficient) ⟨true, true, none, some 1, some 1⟩)))
    (leftClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult2863.resultEvent
      SemanticRightRootResult2863.owner
      (SemanticRightRootResult2863.actual selector witness)
      SemanticRightRootResult2863.rawTerms
      SemanticRightRootResult2863.summary)
    (rightClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult583.resultEvent
      SemanticRightRootResult583.owner
      (SemanticRightRootResult583.actual selector witness)
      SemanticRightRootResult583.rawTerms
      SemanticRightRootResult583.summary)
    (outputAt : (history.lookup resultEvent).map AnnotatedEvent.event =
      some (.resultExact owner rawTerms summary)) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  refine ⟨outputAt, ?_⟩
  exact exactValueClaim_product_of_mod_zero 100418593683253592432016548326729029359133068138294319235841 witness.env
    (SemanticRightRootResult2863.actual selector witness)
    (SemanticRightRootResult583.actual selector witness) left right output
    (by simpa [left, leftRaw, SemanticRightRootResult2863.summary] using leftClaim.claim)
    (by simpa [right, rightRaw, SemanticRightRootResult583.summary] using rightClaim.claim)
    (resultSound witness.env) (by decide)

theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply claimOfHistory selector witness (by rfl) (by rfl) (by rfl) (by rfl)
  · exact SemanticRightRootResult2863.claimSound
      selector selectorLower selectorUpper witness
  · exact SemanticRightRootResult583.claimSound
      selector selectorLower selectorUpper witness
  · rfl

end SemanticRightRootResult2868

namespace SemanticRightRootResult2871

def owner : Owner := ⟨.program ⟨214⟩, ⟨17554⟩⟩
def rawTerms : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17554⟩⟩], orderedFactors := [] } }]
def summary : Bound := .exactZero
def producerEvent : Nat := 2870
def resultEvent : Nat := 2871
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

end SemanticRightRootResult2871

namespace SemanticRightRootResult2876

def leftRaw : List Term := SemanticRightRootResult2871.rawTerms
def rightRaw : List Term := SemanticRightRootResult593.rawTerms
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6473⟩⟩, ⟨.program ⟨214⟩, ⟨17554⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def owner : Owner := ⟨.program ⟨214⟩, ⟨17555⟩⟩
def rawTerms : List Term := outputRaw
def summary : Bound := .exactZero
def resultEvent : Nat := 2876
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
  SemanticRightRootResult2871.actual selector witness *
    SemanticRightRootResult593.actual selector witness
theorem claimOfHistory (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841)
    (expressionAt : document.expressions.lookup 17555 =
      some Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Expression068.ExpressionRow17555)
    (leftPredecessorAt : (history.lookup 2872).map AnnotatedEvent.event =
      some (.predecessor owner 0 ⟨17554⟩ 2871))
    (rightPredecessorAt : (history.lookup 2873).map AnnotatedEvent.event =
      some (.predecessor owner 1 ⟨6473⟩ 593))
    (ruleAt : (history.lookup 2874).map AnnotatedEvent.event =
      some (.boundTransfer owner (.product (.predecessor 0 2872 .coefficient) (.predecessor 1 2873 .coefficient) ⟨true, true, none, some 1, some 1⟩)))
    (leftClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult2871.resultEvent
      SemanticRightRootResult2871.owner
      (SemanticRightRootResult2871.actual selector witness)
      SemanticRightRootResult2871.rawTerms
      SemanticRightRootResult2871.summary)
    (rightClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult593.resultEvent
      SemanticRightRootResult593.owner
      (SemanticRightRootResult593.actual selector witness)
      SemanticRightRootResult593.rawTerms
      SemanticRightRootResult593.summary)
    (outputAt : (history.lookup resultEvent).map AnnotatedEvent.event =
      some (.resultExact owner rawTerms summary)) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  refine ⟨outputAt, ?_⟩
  exact exactValueClaim_product_of_mod_zero 100418593683253592432016548326729029359133068138294319235841 witness.env
    (SemanticRightRootResult2871.actual selector witness)
    (SemanticRightRootResult593.actual selector witness) left right output
    (by simpa [left, leftRaw, SemanticRightRootResult2871.summary] using leftClaim.claim)
    (by simpa [right, rightRaw, SemanticRightRootResult593.summary] using rightClaim.claim)
    (resultSound witness.env) (by decide)

theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply claimOfHistory selector witness (by rfl) (by rfl) (by rfl) (by rfl)
  · exact SemanticRightRootResult2871.claimSound
      selector selectorLower selectorUpper witness
  · exact SemanticRightRootResult593.claimSound
      selector selectorLower selectorUpper witness
  · rfl

end SemanticRightRootResult2876

namespace SemanticRightRootResult2879

def owner : Owner := ⟨.program ⟨214⟩, ⟨18848⟩⟩
def rawTerms : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18848⟩⟩], orderedFactors := [] } }]
def summary : Bound := .exactZero
def producerEvent : Nat := 2878
def resultEvent : Nat := 2879
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

end SemanticRightRootResult2879

namespace SemanticRightRootResult2884

def leftRaw : List Term := SemanticRightRootResult2879.rawTerms
def rightRaw : List Term := SemanticRightRootResult603.rawTerms
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6490⟩⟩, ⟨.program ⟨214⟩, ⟨18848⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def owner : Owner := ⟨.program ⟨214⟩, ⟨18849⟩⟩
def rawTerms : List Term := outputRaw
def summary : Bound := .exactZero
def resultEvent : Nat := 2884
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
  SemanticRightRootResult2879.actual selector witness *
    SemanticRightRootResult603.actual selector witness
theorem claimOfHistory (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841)
    (expressionAt : document.expressions.lookup 18849 =
      some Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Expression073.ExpressionRow18849)
    (leftPredecessorAt : (history.lookup 2880).map AnnotatedEvent.event =
      some (.predecessor owner 0 ⟨18848⟩ 2879))
    (rightPredecessorAt : (history.lookup 2881).map AnnotatedEvent.event =
      some (.predecessor owner 1 ⟨6490⟩ 603))
    (ruleAt : (history.lookup 2882).map AnnotatedEvent.event =
      some (.boundTransfer owner (.product (.predecessor 0 2880 .coefficient) (.predecessor 1 2881 .coefficient) ⟨true, true, none, some 1, some 1⟩)))
    (leftClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult2879.resultEvent
      SemanticRightRootResult2879.owner
      (SemanticRightRootResult2879.actual selector witness)
      SemanticRightRootResult2879.rawTerms
      SemanticRightRootResult2879.summary)
    (rightClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult603.resultEvent
      SemanticRightRootResult603.owner
      (SemanticRightRootResult603.actual selector witness)
      SemanticRightRootResult603.rawTerms
      SemanticRightRootResult603.summary)
    (outputAt : (history.lookup resultEvent).map AnnotatedEvent.event =
      some (.resultExact owner rawTerms summary)) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  refine ⟨outputAt, ?_⟩
  exact exactValueClaim_product_of_mod_zero 100418593683253592432016548326729029359133068138294319235841 witness.env
    (SemanticRightRootResult2879.actual selector witness)
    (SemanticRightRootResult603.actual selector witness) left right output
    (by simpa [left, leftRaw, SemanticRightRootResult2879.summary] using leftClaim.claim)
    (by simpa [right, rightRaw, SemanticRightRootResult603.summary] using rightClaim.claim)
    (resultSound witness.env) (by decide)

theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply claimOfHistory selector witness (by rfl) (by rfl) (by rfl) (by rfl)
  · exact SemanticRightRootResult2879.claimSound
      selector selectorLower selectorUpper witness
  · exact SemanticRightRootResult603.claimSound
      selector selectorLower selectorUpper witness
  · rfl

end SemanticRightRootResult2884

namespace SemanticRightRootResult2887

def owner : Owner := ⟨.program ⟨214⟩, ⟨17610⟩⟩
def rawTerms : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17610⟩⟩], orderedFactors := [] } }]
def summary : Bound := .exactZero
def producerEvent : Nat := 2886
def resultEvent : Nat := 2887
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

end SemanticRightRootResult2887

namespace SemanticRightRootResult2892

def leftRaw : List Term := SemanticRightRootResult2887.rawTerms
def rightRaw : List Term := SemanticRightRootResult613.rawTerms
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6494⟩⟩, ⟨.program ⟨214⟩, ⟨17610⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def owner : Owner := ⟨.program ⟨214⟩, ⟨17611⟩⟩
def rawTerms : List Term := outputRaw
def summary : Bound := .exactZero
def resultEvent : Nat := 2892
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
  SemanticRightRootResult2887.actual selector witness *
    SemanticRightRootResult613.actual selector witness
theorem claimOfHistory (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841)
    (expressionAt : document.expressions.lookup 17611 =
      some Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Expression068.ExpressionRow17611)
    (leftPredecessorAt : (history.lookup 2888).map AnnotatedEvent.event =
      some (.predecessor owner 0 ⟨17610⟩ 2887))
    (rightPredecessorAt : (history.lookup 2889).map AnnotatedEvent.event =
      some (.predecessor owner 1 ⟨6494⟩ 613))
    (ruleAt : (history.lookup 2890).map AnnotatedEvent.event =
      some (.boundTransfer owner (.product (.predecessor 0 2888 .coefficient) (.predecessor 1 2889 .coefficient) ⟨true, true, none, some 1, some 1⟩)))
    (leftClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult2887.resultEvent
      SemanticRightRootResult2887.owner
      (SemanticRightRootResult2887.actual selector witness)
      SemanticRightRootResult2887.rawTerms
      SemanticRightRootResult2887.summary)
    (rightClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult613.resultEvent
      SemanticRightRootResult613.owner
      (SemanticRightRootResult613.actual selector witness)
      SemanticRightRootResult613.rawTerms
      SemanticRightRootResult613.summary)
    (outputAt : (history.lookup resultEvent).map AnnotatedEvent.event =
      some (.resultExact owner rawTerms summary)) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  refine ⟨outputAt, ?_⟩
  exact exactValueClaim_product_of_mod_zero 100418593683253592432016548326729029359133068138294319235841 witness.env
    (SemanticRightRootResult2887.actual selector witness)
    (SemanticRightRootResult613.actual selector witness) left right output
    (by simpa [left, leftRaw, SemanticRightRootResult2887.summary] using leftClaim.claim)
    (by simpa [right, rightRaw, SemanticRightRootResult613.summary] using rightClaim.claim)
    (resultSound witness.env) (by decide)

theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply claimOfHistory selector witness (by rfl) (by rfl) (by rfl) (by rfl)
  · exact SemanticRightRootResult2887.claimSound
      selector selectorLower selectorUpper witness
  · exact SemanticRightRootResult613.claimSound
      selector selectorLower selectorUpper witness
  · rfl

end SemanticRightRootResult2892

namespace SemanticRightRootResult2895

def owner : Owner := ⟨.program ⟨214⟩, ⟨17666⟩⟩
def rawTerms : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17666⟩⟩], orderedFactors := [] } }]
def summary : Bound := .exactZero
def producerEvent : Nat := 2894
def resultEvent : Nat := 2895
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

end SemanticRightRootResult2895

namespace SemanticRightRootResult2900

def leftRaw : List Term := SemanticRightRootResult2895.rawTerms
def rightRaw : List Term := SemanticRightRootResult623.rawTerms
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6502⟩⟩, ⟨.program ⟨214⟩, ⟨17666⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def owner : Owner := ⟨.program ⟨214⟩, ⟨17667⟩⟩
def rawTerms : List Term := outputRaw
def summary : Bound := .exactZero
def resultEvent : Nat := 2900
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
  SemanticRightRootResult2895.actual selector witness *
    SemanticRightRootResult623.actual selector witness
theorem claimOfHistory (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841)
    (expressionAt : document.expressions.lookup 17667 =
      some Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Expression069.ExpressionRow17667)
    (leftPredecessorAt : (history.lookup 2896).map AnnotatedEvent.event =
      some (.predecessor owner 0 ⟨17666⟩ 2895))
    (rightPredecessorAt : (history.lookup 2897).map AnnotatedEvent.event =
      some (.predecessor owner 1 ⟨6502⟩ 623))
    (ruleAt : (history.lookup 2898).map AnnotatedEvent.event =
      some (.boundTransfer owner (.product (.predecessor 0 2896 .coefficient) (.predecessor 1 2897 .coefficient) ⟨true, true, none, some 1, some 1⟩)))
    (leftClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult2895.resultEvent
      SemanticRightRootResult2895.owner
      (SemanticRightRootResult2895.actual selector witness)
      SemanticRightRootResult2895.rawTerms
      SemanticRightRootResult2895.summary)
    (rightClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult623.resultEvent
      SemanticRightRootResult623.owner
      (SemanticRightRootResult623.actual selector witness)
      SemanticRightRootResult623.rawTerms
      SemanticRightRootResult623.summary)
    (outputAt : (history.lookup resultEvent).map AnnotatedEvent.event =
      some (.resultExact owner rawTerms summary)) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  refine ⟨outputAt, ?_⟩
  exact exactValueClaim_product_of_mod_zero 100418593683253592432016548326729029359133068138294319235841 witness.env
    (SemanticRightRootResult2895.actual selector witness)
    (SemanticRightRootResult623.actual selector witness) left right output
    (by simpa [left, leftRaw, SemanticRightRootResult2895.summary] using leftClaim.claim)
    (by simpa [right, rightRaw, SemanticRightRootResult623.summary] using rightClaim.claim)
    (resultSound witness.env) (by decide)

theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply claimOfHistory selector witness (by rfl) (by rfl) (by rfl) (by rfl)
  · exact SemanticRightRootResult2895.claimSound
      selector selectorLower selectorUpper witness
  · exact SemanticRightRootResult623.claimSound
      selector selectorLower selectorUpper witness
  · rfl

end SemanticRightRootResult2900

namespace SemanticRightRootResult2903

def owner : Owner := ⟨.program ⟨214⟩, ⟨18042⟩⟩
def rawTerms : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18042⟩⟩], orderedFactors := [] } }]
def summary : Bound := .exactZero
def producerEvent : Nat := 2902
def resultEvent : Nat := 2903
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

end SemanticRightRootResult2903

namespace SemanticRightRootResult2908

def leftRaw : List Term := SemanticRightRootResult2903.rawTerms
def rightRaw : List Term := SemanticRightRootResult633.rawTerms
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6383⟩⟩, ⟨.program ⟨214⟩, ⟨18042⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def owner : Owner := ⟨.program ⟨214⟩, ⟨18043⟩⟩
def rawTerms : List Term := outputRaw
def summary : Bound := .exactZero
def resultEvent : Nat := 2908
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
  SemanticRightRootResult2903.actual selector witness *
    SemanticRightRootResult633.actual selector witness
theorem claimOfHistory (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841)
    (expressionAt : document.expressions.lookup 18043 =
      some Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Expression070.ExpressionRow18043)
    (leftPredecessorAt : (history.lookup 2904).map AnnotatedEvent.event =
      some (.predecessor owner 0 ⟨18042⟩ 2903))
    (rightPredecessorAt : (history.lookup 2905).map AnnotatedEvent.event =
      some (.predecessor owner 1 ⟨6383⟩ 633))
    (ruleAt : (history.lookup 2906).map AnnotatedEvent.event =
      some (.boundTransfer owner (.product (.predecessor 0 2904 .coefficient) (.predecessor 1 2905 .coefficient) ⟨true, true, none, some 1, some 1⟩)))
    (leftClaim : ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult2903.resultEvent
      SemanticRightRootResult2903.owner
      (SemanticRightRootResult2903.actual selector witness)
      SemanticRightRootResult2903.rawTerms
      SemanticRightRootResult2903.summary)
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
    (SemanticRightRootResult2903.actual selector witness)
    (SemanticRightRootResult633.actual selector witness) left right output
    (by simpa [left, leftRaw, SemanticRightRootResult2903.summary] using leftClaim.claim)
    (by simpa [right, rightRaw, SemanticRightRootResult633.summary] using rightClaim.claim)
    (resultSound witness.env) (by decide)

theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply claimOfHistory selector witness (by rfl) (by rfl) (by rfl) (by rfl)
  · exact SemanticRightRootResult2903.claimSound
      selector selectorLower selectorUpper witness
  · exact SemanticRightRootResult633.claimSound
      selector selectorLower selectorUpper witness
  · rfl

end SemanticRightRootResult2908

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
