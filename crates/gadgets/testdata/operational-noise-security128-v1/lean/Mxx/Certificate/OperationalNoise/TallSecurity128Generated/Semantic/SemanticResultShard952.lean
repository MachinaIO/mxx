import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard952
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard006
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard128
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard248
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard853
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard854
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard938
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard939
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard940
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard942
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard943
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard945
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard946
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard947
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard949
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard950
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard951

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult133896
def owner : Owner := ⟨.program ⟨257⟩, ⟨7073⟩⟩
def rawTerms : List Term := Proof.Events523.exact133896RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 133896
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult133896.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge133895.working .exactZero) := by
  apply operatorTensorMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge133895.frameStart)
    (transferEvent := 133894) (owner := owner)
    (leftResult := 723) (rightResult := 119778)
    (working := LeftOperatorMerge133895.working)
    (reconstruction := LeftOperatorMerge133895.reconstruction)
    (leftReference := .predecessor 0 133892 .coefficient) (rightReference := .predecessor 1 133893 .coefficient)
  · rfl
  · rfl
  · exact SemanticResult723.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult119778.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge133895.operationAgreement
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
end SemanticResult133896

namespace SemanticResult133901
def owner : Owner := ⟨.program ⟨257⟩, ⟨8142⟩⟩
def rawTerms : List Term := Proof.Events523.exact133901RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 133901
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult133901.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge133900.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge133900.frameStart)
    (transferEvent := 133899) (owner := owner)
    (leftResult := 119648) (rightResult := 15896)
    (working := LeftOperatorMerge133900.working)
    (reconstruction := LeftOperatorMerge133900.reconstruction)
    (leftReference := .predecessor 0 133897 .coefficient) (rightReference := .predecessor 1 133898 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult119648.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult15896.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge133900.operationAgreement
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
end SemanticResult133901

namespace SemanticResult133905
def owner : Owner := ⟨.program ⟨257⟩, ⟨9341⟩⟩
def rawTerms : List Term := Proof.Events523.exact133905RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 133905
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult133905.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 133902) (rightBinding := 133903)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨8142⟩) (rightExpression := ⟨7073⟩)
    (transferEvent := 133904)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult133901.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult133896.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult133905

namespace SemanticResult133911
def owner : Owner := ⟨.program ⟨257⟩, ⟨9342⟩⟩
def rawTerms : List Term := Proof.Events523.exact133911RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 133911
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult133911.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (coefficientTransfer := 133908) (survivorTransfer := 133909)
    (survivorEvent := 133910) (resultEvent := resultEvent)
    (rightCoefficientProducer := 31515)
    (owner := owner) (leftOwner := SemanticResult133905.owner)
    (rightOwner := SemanticResult31516.owner)
    (leftResult := 133905) (rightResult := 31516)
    (leftBinding := 133906) (rightBinding := 133907)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨9341⟩) (rightExpression := ⟨118⟩)
    (leftActual := SemanticResult133905.actual selector witness)
    (rightActual := SemanticResult31516.actual selector witness)
    (leftRaw := SemanticResult133905.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨257⟩, ⟨118⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftBound31515.actual selector witness)
    (survivorMagnitude := LeftBound133909.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult133905.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult31516.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound31515.bound, RecordedBoundRefines] <;> decide)
      (LeftBound31515.derived selector witness)
  · exact LeftBound133909.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult133911

namespace SemanticResult133918
def owner : Owner := ⟨.program ⟨257⟩, ⟨9465⟩⟩
def rawTerms : List Term := Proof.Events523.exact133918RawTerms
def summary : Bound := (.finite 52)
def resultEvent : Nat := 133918
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult133918.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubFiniteMergeClaimAt
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := LeftOperatorMerge133915.frameStart)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult133911.owner)
    (rightOwner := SemanticResult133911.owner)
    (leftResult := 133911) (rightResult := 133911)
    (leftActual := SemanticResult133911.actual selector witness)
    (rightActual := SemanticResult133911.actual selector witness)
    (leftRaw := SemanticResult133911.rawTerms)
    (rightRaw := SemanticResult133911.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 26)
    (rightMaximum := 26) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 133912) (rightBinding := 133913)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨9342⟩) (rightExpression := ⟨9342⟩)
    (coefficientTransfer := 133914) (summaryTransfer := 133917)
    (base := LeftOperatorMerge133915.base)
    (reconstruction := LeftOperatorMerge133915.reconstruction)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult133911.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult133911.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge133915.operationAgreement
  · rfl
  · decide
end SemanticResult133918

namespace SemanticResult133923
def owner : Owner := ⟨.program ⟨257⟩, ⟨17647⟩⟩
def rawTerms : List Term := Proof.Events523.exact133923RawTerms
def summary : Bound := (.finite 345624685687166110058245054666339432529972)
def resultEvent : Nat := 133923
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult133923.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult133918.owner)
    (rightOwner := SemanticResult133891.owner)
    (leftResult := 133918) (rightResult := 133891)
    (leftActual := SemanticResult133918.actual selector witness)
    (rightActual := SemanticResult133891.actual selector witness)
    (leftRaw := SemanticResult133918.rawTerms)
    (rightRaw := SemanticResult133891.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 52)
    (rightMaximum := 345624685687166110058245054666339432529920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 133919) (rightBinding := 133920)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨9465⟩) (rightExpression := ⟨17646⟩)
    (transferEvent := 133921) (summaryTransferEvent := 133922)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult133918.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult133891.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult133923

namespace SemanticResult133928
def owner : Owner := ⟨.program ⟨257⟩, ⟨20526⟩⟩
def rawTerms : List Term := Proof.Events523.exact133928RawTerms
def summary : Bound := (.finite 691250426059631610003352154589745737891892)
def resultEvent : Nat := 133928
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult133928.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult133923.owner)
    (rightOwner := SemanticResult133679.owner)
    (leftResult := 133923) (rightResult := 133679)
    (leftActual := SemanticResult133923.actual selector witness)
    (rightActual := SemanticResult133679.actual selector witness)
    (leftRaw := SemanticResult133923.rawTerms)
    (rightRaw := SemanticResult133679.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 345624685687166110058245054666339432529972)
    (rightMaximum := 345625740372465499945107099923406305361920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 133924) (rightBinding := 133925)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨17647⟩) (rightExpression := ⟨20525⟩)
    (transferEvent := 133926) (summaryTransferEvent := 133927)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult133923.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult133679.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult133928

namespace SemanticResult133933
def owner : Owner := ⟨.program ⟨257⟩, ⟨23746⟩⟩
def rawTerms : List Term := Proof.Events523.exact133933RawTerms
def summary : Bound := (.finite 1036877221117396499835321299770218916085812)
def resultEvent : Nat := 133933
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult133933.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult133928.owner)
    (rightOwner := SemanticResult133467.owner)
    (leftResult := 133928) (rightResult := 133467)
    (leftActual := SemanticResult133928.actual selector witness)
    (rightActual := SemanticResult133467.actual selector witness)
    (leftRaw := SemanticResult133928.rawTerms)
    (rightRaw := SemanticResult133467.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 691250426059631610003352154589745737891892)
    (rightMaximum := 345626795057764889831969145180473178193920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 133929) (rightBinding := 133930)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨20526⟩) (rightExpression := ⟨23745⟩)
    (transferEvent := 133931) (summaryTransferEvent := 133932)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult133928.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult133467.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult133933

namespace SemanticResult133938
def owner : Owner := ⟨.program ⟨257⟩, ⟨33766⟩⟩
def rawTerms : List Term := Proof.Events523.exact133938RawTerms
def summary : Bound := (.finite 1382506125545760169441014535464825839943732)
def resultEvent : Nat := 133938
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult133938.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult133933.owner)
    (rightOwner := SemanticResult133255.owner)
    (leftResult := 133933) (rightResult := 133255)
    (leftActual := SemanticResult133933.actual selector witness)
    (rightActual := SemanticResult133255.actual selector witness)
    (leftRaw := SemanticResult133933.rawTerms)
    (rightRaw := SemanticResult133255.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 1036877221117396499835321299770218916085812)
    (rightMaximum := 345628904428363669605693235694606923857920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 133934) (rightBinding := 133935)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨23746⟩) (rightExpression := ⟨33765⟩)
    (transferEvent := 133936) (summaryTransferEvent := 133937)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult133933.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult133255.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult133938

namespace SemanticResult133943
def owner : Owner := ⟨.program ⟨257⟩, ⟨52826⟩⟩
def rawTerms : List Term := Proof.Events523.exact133943RawTerms
def summary : Bound := (.finite 1728139248715321398594155952187700255129652)
def resultEvent : Nat := 133943
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult133943.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult133938.owner)
    (rightOwner := SemanticResult133043.owner)
    (leftResult := 133938) (rightResult := 133043)
    (leftActual := SemanticResult133938.actual selector witness)
    (rightActual := SemanticResult133043.actual selector witness)
    (leftRaw := SemanticResult133938.rawTerms)
    (rightRaw := SemanticResult133043.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 1382506125545760169441014535464825839943732)
    (rightMaximum := 345633123169561229153141416722874415185920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 133939) (rightBinding := 133940)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨33766⟩) (rightExpression := ⟨52825⟩)
    (transferEvent := 133941) (summaryTransferEvent := 133942)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult133938.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult133043.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult133943

namespace SemanticResult133948
def owner : Owner := ⟨.program ⟨257⟩, ⟨55806⟩⟩
def rawTerms : List Term := Proof.Events523.exact133948RawTerms
def summary : Bound := (.finite 2073774481255481407521021459424708415979572)
def resultEvent : Nat := 133948
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult133948.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult133943.owner)
    (rightOwner := SemanticResult132831.owner)
    (leftResult := 133943) (rightResult := 132831)
    (leftActual := SemanticResult133943.actual selector witness)
    (rightActual := SemanticResult132831.actual selector witness)
    (leftRaw := SemanticResult133943.rawTerms)
    (rightRaw := SemanticResult132831.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 1728139248715321398594155952187700255129652)
    (rightMaximum := 345635232540160008926865507237008160849920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 133944) (rightBinding := 133945)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨52826⟩) (rightExpression := ⟨55805⟩)
    (transferEvent := 133946) (summaryTransferEvent := 133947)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult133943.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult132831.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult133948

namespace SemanticResult133953
def owner : Owner := ⟨.program ⟨257⟩, ⟨58786⟩⟩
def rawTerms : List Term := Proof.Events523.exact133953RawTerms
def summary : Bound := (.finite 2419413932536838975995335147689984068157492)
def resultEvent : Nat := 133953
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult133953.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult133948.owner)
    (rightOwner := SemanticResult132619.owner)
    (leftResult := 133948) (rightResult := 132619)
    (leftActual := SemanticResult133948.actual selector witness)
    (rightActual := SemanticResult132619.actual selector witness)
    (leftRaw := SemanticResult133948.rawTerms)
    (rightRaw := SemanticResult132619.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 2073774481255481407521021459424708415979572)
    (rightMaximum := 345639451281357568474313688265275652177920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 133949) (rightBinding := 133950)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨55806⟩) (rightExpression := ⟨58785⟩)
    (transferEvent := 133951) (summaryTransferEvent := 133952)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult133948.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult132619.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult133953

namespace SemanticResult133958
def owner : Owner := ⟨.program ⟨257⟩, ⟨61766⟩⟩
def rawTerms : List Term := Proof.Events523.exact133958RawTerms
def summary : Bound := (.finite 2765055493188795324243372926469393465999412)
def resultEvent : Nat := 133958
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult133958.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult133953.owner)
    (rightOwner := SemanticResult132407.owner)
    (leftResult := 133953) (rightResult := 132407)
    (leftActual := SemanticResult133953.actual selector witness)
    (rightActual := SemanticResult132407.actual selector witness)
    (leftRaw := SemanticResult133953.rawTerms)
    (rightRaw := SemanticResult132407.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 2419413932536838975995335147689984068157492)
    (rightMaximum := 345641560651956348248037778779409397841920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 133954) (rightBinding := 133955)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨58786⟩) (rightExpression := ⟨61765⟩)
    (transferEvent := 133956) (summaryTransferEvent := 133957)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult133953.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult132407.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult133958

namespace SemanticResult133963
def owner : Owner := ⟨.program ⟨257⟩, ⟨64746⟩⟩
def rawTerms : List Term := Proof.Events523.exact133963RawTerms
def summary : Bound := (.finite 3110701272581949232038858886277070355169332)
def resultEvent : Nat := 133963
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult133963.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult133958.owner)
    (rightOwner := SemanticResult132195.owner)
    (leftResult := 133958) (rightResult := 132195)
    (leftActual := SemanticResult133958.actual selector witness)
    (rightActual := SemanticResult132195.actual selector witness)
    (leftRaw := SemanticResult133958.rawTerms)
    (rightRaw := SemanticResult132195.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 2765055493188795324243372926469393465999412)
    (rightMaximum := 345645779393153907795485959807676889169920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 133959) (rightBinding := 133960)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨61766⟩) (rightExpression := ⟨64745⟩)
    (transferEvent := 133961) (summaryTransferEvent := 133962)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult133958.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult132195.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult133963

namespace SemanticResult133968
def owner : Owner := ⟨.program ⟨257⟩, ⟨69851⟩⟩
def rawTerms : List Term := Proof.Events523.exact133968RawTerms
def summary : Bound := (.finite 3456353380086899479155517117627148481331252)
def resultEvent : Nat := 133968
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult133968.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult133963.owner)
    (rightOwner := SemanticResult131983.owner)
    (leftResult := 133963) (rightResult := 131983)
    (leftActual := SemanticResult133963.actual selector witness)
    (rightActual := SemanticResult131983.actual selector witness)
    (leftRaw := SemanticResult133963.rawTerms)
    (rightRaw := SemanticResult131983.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 3110701272581949232038858886277070355169332)
    (rightMaximum := 345652107504950247116658231350078126161920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 133964) (rightBinding := 133965)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨64746⟩) (rightExpression := ⟨69850⟩)
    (transferEvent := 133966) (summaryTransferEvent := 133967)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult133963.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult131983.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult133968

namespace SemanticResult133973
def owner : Owner := ⟨.program ⟨257⟩, ⟨69852⟩⟩
def rawTerms : List Term := Proof.Events523.exact133973RawTerms
def summary : Bound := (.finite 3802007596962448506045899439491360353157172)
def resultEvent : Nat := 133973
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult133973.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult133968.owner)
    (rightOwner := SemanticResult131771.owner)
    (leftResult := 133968) (rightResult := 131771)
    (leftActual := SemanticResult133968.actual selector witness)
    (rightActual := SemanticResult131771.actual selector witness)
    (leftRaw := SemanticResult133968.rawTerms)
    (rightRaw := SemanticResult131771.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 3456353380086899479155517117627148481331252)
    (rightMaximum := 345654216875549026890382321864211871825920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 133969) (rightBinding := 133970)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨69851⟩) (rightExpression := ⟨28187⟩)
    (transferEvent := 133971) (summaryTransferEvent := 133972)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult133968.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult131771.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult133973

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
