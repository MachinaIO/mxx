import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard248
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard006
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard128
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard142
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard143
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard235
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard236
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard238
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard239
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard241
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard242
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard243
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard245
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard246
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard247

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult31516
def owner : Owner := ⟨.program ⟨257⟩, ⟨118⟩⟩
def rawTerms : List Term := Proof.Events123.exact31516RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 31516
def producerEvent : Nat := 31515
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult31516.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.identity (.predecessor 0 31514 .coefficient), 0, .finite 26, .identity (.predecessor 0 31514 .coefficient), ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult31516

namespace SemanticResult31521
def owner : Owner := ⟨.program ⟨257⟩, ⟨7059⟩⟩
def rawTerms : List Term := Proof.Events123.exact31521RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 31521
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult31521.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge31520.working .exactZero) := by
  apply operatorTensorMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge31520.frameStart)
    (transferEvent := 31519) (owner := owner)
    (leftResult := 723) (rightResult := 17057)
    (working := LeftOperatorMerge31520.working)
    (reconstruction := LeftOperatorMerge31520.reconstruction)
    (leftReference := .predecessor 0 31517 .coefficient) (rightReference := .predecessor 1 31518 .coefficient)
  · rfl
  · rfl
  · exact SemanticResult723.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult17057.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge31520.operationAgreement
  · decide

theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply exactClaimAt_of_mergeClaim
    (mergeClaim selector selectorLower selectorUpper witness)
  · decide +kernel
  · rfl
end SemanticResult31521

namespace SemanticResult31526
def owner : Owner := ⟨.program ⟨257⟩, ⟨7610⟩⟩
def rawTerms : List Term := Proof.Events123.exact31526RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 31526
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult31526.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge31525.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge31525.frameStart)
    (transferEvent := 31524) (owner := owner)
    (leftResult := 16922) (rightResult := 15896)
    (working := LeftOperatorMerge31525.working)
    (reconstruction := LeftOperatorMerge31525.reconstruction)
    (leftReference := .predecessor 0 31522 .coefficient) (rightReference := .predecessor 1 31523 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult16922.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult15896.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge31525.operationAgreement
  · decide

theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply exactClaimAt_of_mergeClaim
    (mergeClaim selector selectorLower selectorUpper witness)
  · decide +kernel
  · rfl
end SemanticResult31526

namespace SemanticResult31530
def owner : Owner := ⟨.program ⟨257⟩, ⟨9285⟩⟩
def rawTerms : List Term := Proof.Events123.exact31530RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 31530
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult31530.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 31527) (rightBinding := 31528)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7610⟩) (rightExpression := ⟨7059⟩)
    (transferEvent := 31529)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult31526.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult31521.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult31530

namespace SemanticResult31536
def owner : Owner := ⟨.program ⟨257⟩, ⟨9286⟩⟩
def rawTerms : List Term := Proof.Events123.exact31536RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 31536
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult31536.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (coefficientTransfer := 31533) (survivorTransfer := 31534)
    (survivorEvent := 31535) (resultEvent := resultEvent)
    (rightCoefficientProducer := 31515)
    (owner := owner) (leftOwner := SemanticResult31530.owner)
    (rightOwner := SemanticResult31516.owner)
    (leftResult := 31530) (rightResult := 31516)
    (leftBinding := 31531) (rightBinding := 31532)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨9285⟩) (rightExpression := ⟨118⟩)
    (leftActual := SemanticResult31530.actual selector witness)
    (rightActual := SemanticResult31516.actual selector witness)
    (leftRaw := SemanticResult31530.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨257⟩, ⟨118⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftBound31515.actual selector witness)
    (survivorMagnitude := LeftBound31534.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult31530.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult31516.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound31515.bound, RecordedBoundRefines] <;> decide)
      (LeftBound31515.derived selector witness)
  · exact LeftBound31534.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult31536

namespace SemanticResult31543
def owner : Owner := ⟨.program ⟨257⟩, ⟨9451⟩⟩
def rawTerms : List Term := Proof.Events123.exact31543RawTerms
def summary : Bound := (.finite 52)
def resultEvent : Nat := 31543
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult31543.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubFiniteMergeClaimAt
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := LeftOperatorMerge31540.frameStart)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult31536.owner)
    (rightOwner := SemanticResult31536.owner)
    (leftResult := 31536) (rightResult := 31536)
    (leftActual := SemanticResult31536.actual selector witness)
    (rightActual := SemanticResult31536.actual selector witness)
    (leftRaw := SemanticResult31536.rawTerms)
    (rightRaw := SemanticResult31536.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 26)
    (rightMaximum := 26) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 31537) (rightBinding := 31538)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨9286⟩) (rightExpression := ⟨9286⟩)
    (coefficientTransfer := 31539) (summaryTransfer := 31542)
    (base := LeftOperatorMerge31540.base)
    (reconstruction := LeftOperatorMerge31540.reconstruction)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult31536.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult31536.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge31540.operationAgreement
  · rfl
  · decide
end SemanticResult31543

namespace SemanticResult31548
def owner : Owner := ⟨.program ⟨257⟩, ⟨17515⟩⟩
def rawTerms : List Term := Proof.Events123.exact31548RawTerms
def summary : Bound := (.finite 345624685687166110058245054666339432529972)
def resultEvent : Nat := 31548
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult31548.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult31543.owner)
    (rightOwner := SemanticResult31513.owner)
    (leftResult := 31543) (rightResult := 31513)
    (leftActual := SemanticResult31543.actual selector witness)
    (rightActual := SemanticResult31513.actual selector witness)
    (leftRaw := SemanticResult31543.rawTerms)
    (rightRaw := SemanticResult31513.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 52)
    (rightMaximum := 345624685687166110058245054666339432529920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 31544) (rightBinding := 31545)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨9451⟩) (rightExpression := ⟨17514⟩)
    (transferEvent := 31546) (summaryTransferEvent := 31547)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult31543.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult31513.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult31548

namespace SemanticResult31553
def owner : Owner := ⟨.program ⟨257⟩, ⟨20380⟩⟩
def rawTerms : List Term := Proof.Events123.exact31553RawTerms
def summary : Bound := (.finite 691250426059631610003352154589745737891892)
def resultEvent : Nat := 31553
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult31553.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult31548.owner)
    (rightOwner := SemanticResult31301.owner)
    (leftResult := 31548) (rightResult := 31301)
    (leftActual := SemanticResult31548.actual selector witness)
    (rightActual := SemanticResult31301.actual selector witness)
    (leftRaw := SemanticResult31548.rawTerms)
    (rightRaw := SemanticResult31301.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 345624685687166110058245054666339432529972)
    (rightMaximum := 345625740372465499945107099923406305361920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 31549) (rightBinding := 31550)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨17515⟩) (rightExpression := ⟨20379⟩)
    (transferEvent := 31551) (summaryTransferEvent := 31552)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult31548.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult31301.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult31553

namespace SemanticResult31558
def owner : Owner := ⟨.program ⟨257⟩, ⟨23600⟩⟩
def rawTerms : List Term := Proof.Events123.exact31558RawTerms
def summary : Bound := (.finite 1036877221117396499835321299770218916085812)
def resultEvent : Nat := 31558
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult31558.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult31553.owner)
    (rightOwner := SemanticResult31089.owner)
    (leftResult := 31553) (rightResult := 31089)
    (leftActual := SemanticResult31553.actual selector witness)
    (rightActual := SemanticResult31089.actual selector witness)
    (leftRaw := SemanticResult31553.rawTerms)
    (rightRaw := SemanticResult31089.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 691250426059631610003352154589745737891892)
    (rightMaximum := 345626795057764889831969145180473178193920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 31554) (rightBinding := 31555)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨20380⟩) (rightExpression := ⟨23599⟩)
    (transferEvent := 31556) (summaryTransferEvent := 31557)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult31553.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult31089.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult31558

namespace SemanticResult31563
def owner : Owner := ⟨.program ⟨257⟩, ⟨33620⟩⟩
def rawTerms : List Term := Proof.Events123.exact31563RawTerms
def summary : Bound := (.finite 1382506125545760169441014535464825839943732)
def resultEvent : Nat := 31563
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult31563.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult31558.owner)
    (rightOwner := SemanticResult30877.owner)
    (leftResult := 31558) (rightResult := 30877)
    (leftActual := SemanticResult31558.actual selector witness)
    (rightActual := SemanticResult30877.actual selector witness)
    (leftRaw := SemanticResult31558.rawTerms)
    (rightRaw := SemanticResult30877.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 1036877221117396499835321299770218916085812)
    (rightMaximum := 345628904428363669605693235694606923857920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 31559) (rightBinding := 31560)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨23600⟩) (rightExpression := ⟨33619⟩)
    (transferEvent := 31561) (summaryTransferEvent := 31562)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult31558.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult30877.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult31563

namespace SemanticResult31568
def owner : Owner := ⟨.program ⟨257⟩, ⟨52680⟩⟩
def rawTerms : List Term := Proof.Events123.exact31568RawTerms
def summary : Bound := (.finite 1728139248715321398594155952187700255129652)
def resultEvent : Nat := 31568
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult31568.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult31563.owner)
    (rightOwner := SemanticResult30665.owner)
    (leftResult := 31563) (rightResult := 30665)
    (leftActual := SemanticResult31563.actual selector witness)
    (rightActual := SemanticResult30665.actual selector witness)
    (leftRaw := SemanticResult31563.rawTerms)
    (rightRaw := SemanticResult30665.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 1382506125545760169441014535464825839943732)
    (rightMaximum := 345633123169561229153141416722874415185920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 31564) (rightBinding := 31565)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨33620⟩) (rightExpression := ⟨52679⟩)
    (transferEvent := 31566) (summaryTransferEvent := 31567)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult31563.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult30665.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult31568

namespace SemanticResult31573
def owner : Owner := ⟨.program ⟨257⟩, ⟨55660⟩⟩
def rawTerms : List Term := Proof.Events123.exact31573RawTerms
def summary : Bound := (.finite 2073774481255481407521021459424708415979572)
def resultEvent : Nat := 31573
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult31573.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult31568.owner)
    (rightOwner := SemanticResult30453.owner)
    (leftResult := 31568) (rightResult := 30453)
    (leftActual := SemanticResult31568.actual selector witness)
    (rightActual := SemanticResult30453.actual selector witness)
    (leftRaw := SemanticResult31568.rawTerms)
    (rightRaw := SemanticResult30453.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 1728139248715321398594155952187700255129652)
    (rightMaximum := 345635232540160008926865507237008160849920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 31569) (rightBinding := 31570)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨52680⟩) (rightExpression := ⟨55659⟩)
    (transferEvent := 31571) (summaryTransferEvent := 31572)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult31568.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult30453.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult31573

namespace SemanticResult31578
def owner : Owner := ⟨.program ⟨257⟩, ⟨58640⟩⟩
def rawTerms : List Term := Proof.Events123.exact31578RawTerms
def summary : Bound := (.finite 2419413932536838975995335147689984068157492)
def resultEvent : Nat := 31578
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult31578.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult31573.owner)
    (rightOwner := SemanticResult30241.owner)
    (leftResult := 31573) (rightResult := 30241)
    (leftActual := SemanticResult31573.actual selector witness)
    (rightActual := SemanticResult30241.actual selector witness)
    (leftRaw := SemanticResult31573.rawTerms)
    (rightRaw := SemanticResult30241.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 2073774481255481407521021459424708415979572)
    (rightMaximum := 345639451281357568474313688265275652177920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 31574) (rightBinding := 31575)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨55660⟩) (rightExpression := ⟨58639⟩)
    (transferEvent := 31576) (summaryTransferEvent := 31577)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult31573.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult30241.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult31578

namespace SemanticResult31583
def owner : Owner := ⟨.program ⟨257⟩, ⟨61620⟩⟩
def rawTerms : List Term := Proof.Events123.exact31583RawTerms
def summary : Bound := (.finite 2765055493188795324243372926469393465999412)
def resultEvent : Nat := 31583
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult31583.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult31578.owner)
    (rightOwner := SemanticResult30029.owner)
    (leftResult := 31578) (rightResult := 30029)
    (leftActual := SemanticResult31578.actual selector witness)
    (rightActual := SemanticResult30029.actual selector witness)
    (leftRaw := SemanticResult31578.rawTerms)
    (rightRaw := SemanticResult30029.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 2419413932536838975995335147689984068157492)
    (rightMaximum := 345641560651956348248037778779409397841920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 31579) (rightBinding := 31580)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨58640⟩) (rightExpression := ⟨61619⟩)
    (transferEvent := 31581) (summaryTransferEvent := 31582)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult31578.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult30029.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult31583

namespace SemanticResult31588
def owner : Owner := ⟨.program ⟨257⟩, ⟨64600⟩⟩
def rawTerms : List Term := Proof.Events123.exact31588RawTerms
def summary : Bound := (.finite 3110701272581949232038858886277070355169332)
def resultEvent : Nat := 31588
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult31588.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult31583.owner)
    (rightOwner := SemanticResult29817.owner)
    (leftResult := 31583) (rightResult := 29817)
    (leftActual := SemanticResult31583.actual selector witness)
    (rightActual := SemanticResult29817.actual selector witness)
    (leftRaw := SemanticResult31583.rawTerms)
    (rightRaw := SemanticResult29817.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 2765055493188795324243372926469393465999412)
    (rightMaximum := 345645779393153907795485959807676889169920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 31584) (rightBinding := 31585)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨61620⟩) (rightExpression := ⟨64599⟩)
    (transferEvent := 31586) (summaryTransferEvent := 31587)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult31583.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult29817.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult31588

namespace SemanticResult31593
def owner : Owner := ⟨.program ⟨257⟩, ⟨69481⟩⟩
def rawTerms : List Term := Proof.Events123.exact31593RawTerms
def summary : Bound := (.finite 3456353380086899479155517117627148481331252)
def resultEvent : Nat := 31593
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult31593.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult31588.owner)
    (rightOwner := SemanticResult29605.owner)
    (leftResult := 31588) (rightResult := 29605)
    (leftActual := SemanticResult31588.actual selector witness)
    (rightActual := SemanticResult29605.actual selector witness)
    (leftRaw := SemanticResult31588.rawTerms)
    (rightRaw := SemanticResult29605.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 3110701272581949232038858886277070355169332)
    (rightMaximum := 345652107504950247116658231350078126161920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 31589) (rightBinding := 31590)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨64600⟩) (rightExpression := ⟨69480⟩)
    (transferEvent := 31591) (summaryTransferEvent := 31592)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult31588.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult29605.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult31593

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
