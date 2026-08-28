import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard450
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard000
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard129
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard130
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard351
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard424
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard425
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard427
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard428
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard429
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard431
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard432
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard434
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard435
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard436
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard438
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard449

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult60838
def owner : Owner := ⟨.program ⟨257⟩, ⟨65118⟩⟩
def rawTerms : List Term := Proof.Events237.exact60838RawTerms
def summary : Bound := (.finite 3110701272581949232038858886277070355169332)
def resultEvent : Nat := 60838
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult60838.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult60833.owner)
    (rightOwner := SemanticResult59070.owner)
    (leftResult := 60833) (rightResult := 59070)
    (leftActual := SemanticResult60833.actual selector witness)
    (rightActual := SemanticResult59070.actual selector witness)
    (leftRaw := SemanticResult60833.rawTerms)
    (rightRaw := SemanticResult59070.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 2765055493188795324243372926469393465999412)
    (rightMaximum := 345645779393153907795485959807676889169920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 60834) (rightBinding := 60835)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨62138⟩) (rightExpression := ⟨65117⟩)
    (transferEvent := 60836) (summaryTransferEvent := 60837)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult60833.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult59070.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult60838

namespace SemanticResult60843
def owner : Owner := ⟨.program ⟨257⟩, ⟨70799⟩⟩
def rawTerms : List Term := Proof.Events237.exact60843RawTerms
def summary : Bound := (.finite 3456353380086899479155517117627148481331252)
def resultEvent : Nat := 60843
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult60843.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult60838.owner)
    (rightOwner := SemanticResult58858.owner)
    (leftResult := 60838) (rightResult := 58858)
    (leftActual := SemanticResult60838.actual selector witness)
    (rightActual := SemanticResult58858.actual selector witness)
    (leftRaw := SemanticResult60838.rawTerms)
    (rightRaw := SemanticResult58858.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 3110701272581949232038858886277070355169332)
    (rightMaximum := 345652107504950247116658231350078126161920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 60839) (rightBinding := 60840)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨65118⟩) (rightExpression := ⟨70798⟩)
    (transferEvent := 60841) (summaryTransferEvent := 60842)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult60838.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult58858.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult60843

namespace SemanticResult60848
def owner : Owner := ⟨.program ⟨257⟩, ⟨70800⟩⟩
def rawTerms : List Term := Proof.Events237.exact60848RawTerms
def summary : Bound := (.finite 3802007596962448506045899439491360353157172)
def resultEvent : Nat := 60848
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult60848.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult60843.owner)
    (rightOwner := SemanticResult58646.owner)
    (leftResult := 60843) (rightResult := 58646)
    (leftActual := SemanticResult60843.actual selector witness)
    (rightActual := SemanticResult58646.actual selector witness)
    (leftRaw := SemanticResult60843.rawTerms)
    (rightRaw := SemanticResult58646.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 3456353380086899479155517117627148481331252)
    (rightMaximum := 345654216875549026890382321864211871825920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 60844) (rightBinding := 60845)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨70799⟩) (rightExpression := ⟨28487⟩)
    (transferEvent := 60846) (summaryTransferEvent := 60847)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult60843.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult58646.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult60848

namespace SemanticResult60853
def owner : Owner := ⟨.program ⟨257⟩, ⟨70801⟩⟩
def rawTerms : List Term := Proof.Events237.exact60853RawTerms
def summary : Bound := (.finite 4147668141949793872257454032897973461975092)
def resultEvent : Nat := 60853
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult60853.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult60848.owner)
    (rightOwner := SemanticResult58434.owner)
    (leftResult := 60848) (rightResult := 58434)
    (leftActual := SemanticResult60848.actual selector witness)
    (rightActual := SemanticResult58434.actual selector witness)
    (leftRaw := SemanticResult60848.rawTerms)
    (rightRaw := SemanticResult58434.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 3802007596962448506045899439491360353157172)
    (rightMaximum := 345660544987345366211554593406613108817920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 60849) (rightBinding := 60850)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨70800⟩) (rightExpression := ⟨31167⟩)
    (transferEvent := 60851) (summaryTransferEvent := 60852)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult60848.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult58434.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult60853

namespace SemanticResult60858
def owner : Owner := ⟨.program ⟨257⟩, ⟨70802⟩⟩
def rawTerms : List Term := Proof.Events237.exact60858RawTerms
def summary : Bound := (.finite 4493332905678336798016456807332854062121012)
def resultEvent : Nat := 60858
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult60858.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult60853.owner)
    (rightOwner := SemanticResult58222.owner)
    (leftResult := 60853) (rightResult := 58222)
    (leftActual := SemanticResult60853.actual selector witness)
    (rightActual := SemanticResult58222.actual selector witness)
    (leftRaw := SemanticResult60853.rawTerms)
    (rightRaw := SemanticResult58222.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 4147668141949793872257454032897973461975092)
    (rightMaximum := 345664763728542925759002774434880600145920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 60854) (rightBinding := 60855)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨70801⟩) (rightExpression := ⟨36827⟩)
    (transferEvent := 60856) (summaryTransferEvent := 60857)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult60853.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult58222.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult60858

namespace SemanticResult60863
def owner : Owner := ⟨.program ⟨257⟩, ⟨70803⟩⟩
def rawTerms : List Term := Proof.Events237.exact60863RawTerms
def summary : Bound := (.finite 4838999778777478503549183672281868407930932)
def resultEvent : Nat := 60863
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult60863.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult60858.owner)
    (rightOwner := SemanticResult58010.owner)
    (leftResult := 60858) (rightResult := 58010)
    (leftActual := SemanticResult60858.actual selector witness)
    (rightActual := SemanticResult58010.actual selector witness)
    (leftRaw := SemanticResult60858.rawTerms)
    (rightRaw := SemanticResult58010.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 4493332905678336798016456807332854062121012)
    (rightMaximum := 345666873099141705532726864949014345809920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 60859) (rightBinding := 60860)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨70802⟩) (rightExpression := ⟨39507⟩)
    (transferEvent := 60861) (summaryTransferEvent := 60862)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult60858.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult58010.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult60863

namespace SemanticResult60868
def owner : Owner := ⟨.program ⟨257⟩, ⟨70804⟩⟩
def rawTerms : List Term := Proof.Events237.exact60868RawTerms
def summary : Bound := (.finite 5184670870617817768629358718259150245068852)
def resultEvent : Nat := 60868
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult60868.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult60863.owner)
    (rightOwner := SemanticResult57798.owner)
    (leftResult := 60863) (rightResult := 57798)
    (leftActual := SemanticResult60863.actual selector witness)
    (rightActual := SemanticResult57798.actual selector witness)
    (leftRaw := SemanticResult60863.rawTerms)
    (rightRaw := SemanticResult57798.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 4838999778777478503549183672281868407930932)
    (rightMaximum := 345671091840339265080175045977281837137920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 60864) (rightBinding := 60865)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨70803⟩) (rightExpression := ⟨42187⟩)
    (transferEvent := 60866) (summaryTransferEvent := 60867)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult60863.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult57798.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult60868

namespace SemanticResult60873
def owner : Owner := ⟨.program ⟨257⟩, ⟨70805⟩⟩
def rawTerms : List Term := Proof.Events237.exact60873RawTerms
def summary : Bound := (.finite 5530348290569953373030706035778833319198772)
def resultEvent : Nat := 60873
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult60873.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult60868.owner)
    (rightOwner := SemanticResult57586.owner)
    (leftResult := 60868) (rightResult := 57586)
    (leftActual := SemanticResult60868.actual selector witness)
    (rightActual := SemanticResult57586.actual selector witness)
    (leftRaw := SemanticResult60868.rawTerms)
    (rightRaw := SemanticResult57586.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 5184670870617817768629358718259150245068852)
    (rightMaximum := 345677419952135604401347317519683074129920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 60869) (rightBinding := 60870)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨70804⟩) (rightExpression := ⟨44867⟩)
    (transferEvent := 60871) (summaryTransferEvent := 60872)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult60868.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult57586.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult60873

namespace SemanticResult60878
def owner : Owner := ⟨.program ⟨257⟩, ⟨70806⟩⟩
def rawTerms : List Term := Proof.Events237.exact60878RawTerms
def summary : Bound := (.finite 5876032038633885316753225624840917630320692)
def resultEvent : Nat := 60878
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult60878.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult60873.owner)
    (rightOwner := SemanticResult57374.owner)
    (leftResult := 60873) (rightResult := 57374)
    (leftActual := SemanticResult60873.actual selector witness)
    (rightActual := SemanticResult57374.actual selector witness)
    (leftRaw := SemanticResult60873.rawTerms)
    (rightRaw := SemanticResult57374.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 5530348290569953373030706035778833319198772)
    (rightMaximum := 345683748063931943722519589062084311121920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 60874) (rightBinding := 60875)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨70805⟩) (rightExpression := ⟨47547⟩)
    (transferEvent := 60876) (summaryTransferEvent := 60877)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult60873.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult57374.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult60878

namespace SemanticResult60883
def owner : Owner := ⟨.program ⟨257⟩, ⟨70807⟩⟩
def rawTerms : List Term := Proof.Events237.exact60883RawTerms
def summary : Bound := (.finite 6221717896068416040249469304417135687106612)
def resultEvent : Nat := 60883
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult60883.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult60878.owner)
    (rightOwner := SemanticResult57162.owner)
    (leftResult := 60878) (rightResult := 57162)
    (leftActual := SemanticResult60878.actual selector witness)
    (rightActual := SemanticResult57162.actual selector witness)
    (leftRaw := SemanticResult60878.rawTerms)
    (rightRaw := SemanticResult57162.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 5876032038633885316753225624840917630320692)
    (rightMaximum := 345685857434530723496243679576218056785920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 60879) (rightBinding := 60880)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨70806⟩) (rightExpression := ⟨50227⟩)
    (transferEvent := 60881) (summaryTransferEvent := 60882)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult60878.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult57162.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult60883

namespace SemanticResult60888
def owner : Owner := ⟨.program ⟨257⟩, ⟨71507⟩⟩
def rawTerms : List Term := Proof.Events237.exact60888RawTerms
def summary : Bound := (.finite 66805187227601152574551644069558752530002096506798132)
def resultEvent : Nat := 60888
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult60888.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult60883.owner)
    (rightOwner := SemanticResult56950.owner)
    (leftResult := 60883) (rightResult := 56950)
    (leftActual := SemanticResult60883.actual selector witness)
    (rightActual := SemanticResult56950.actual selector witness)
    (leftRaw := SemanticResult60883.rawTerms)
    (rightRaw := SemanticResult56950.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 6221717896068416040249469304417135687106612)
    (rightMaximum := 66805187221379434678483228029309283225584960819691520) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 60884) (rightBinding := 60885)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨70807⟩) (rightExpression := ⟨71505⟩)
    (transferEvent := 60886) (summaryTransferEvent := 60887)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult60883.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult56950.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult60888

namespace SemanticResult60890
def owner : Owner := ⟨.program ⟨257⟩, ⟨28⟩⟩
def rawTerms : List Term := Proof.Events237.exact60890RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 60890
def producerEvent : Nat := 60889
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult60890.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.operator), 0, .finite 26, .authorityOperator, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult60890

namespace SemanticResult60895
def owner : Owner := ⟨.program ⟨257⟩, ⟨7403⟩⟩
def rawTerms : List Term := Proof.Events237.exact60895RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 60895
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult60895.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge60894.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge60894.frameStart)
    (transferEvent := 60893) (owner := owner)
    (leftResult := 27) (rightResult := 16067)
    (working := LeftOperatorMerge60894.working)
    (reconstruction := LeftOperatorMerge60894.reconstruction)
    (leftReference := .predecessor 0 60891 .coefficient) (rightReference := .predecessor 1 60892 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult27.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult16067.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge60894.operationAgreement
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
end SemanticResult60895

namespace SemanticResult60899
def owner : Owner := ⟨.program ⟨257⟩, ⟨11217⟩⟩
def rawTerms : List Term := Proof.Events237.exact60899RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 60899
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult60899.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 60896) (rightBinding := 60897)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7403⟩) (rightExpression := ⟨11176⟩)
    (transferEvent := 60898)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult60895.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult46653.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult60899

namespace SemanticResult60905
def owner : Owner := ⟨.program ⟨257⟩, ⟨11218⟩⟩
def rawTerms : List Term := Proof.Events237.exact60905RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 60905
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult60905.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (coefficientTransfer := 60902) (survivorTransfer := 60903)
    (survivorEvent := 60904) (resultEvent := resultEvent)
    (rightCoefficientProducer := 60889)
    (owner := owner) (leftOwner := SemanticResult60899.owner)
    (rightOwner := SemanticResult60890.owner)
    (leftResult := 60899) (rightResult := 60890)
    (leftBinding := 60900) (rightBinding := 60901)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨11217⟩) (rightExpression := ⟨28⟩)
    (leftActual := SemanticResult60899.actual selector witness)
    (rightActual := SemanticResult60890.actual selector witness)
    (leftRaw := SemanticResult60899.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨257⟩, ⟨28⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftAuthority60889.actual selector witness)
    (survivorMagnitude := LeftBound60903.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult60899.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult60890.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority60889.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority60889.derived selector witness)
  · exact LeftBound60903.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult60905

namespace SemanticResult60933
def owner : Owner := ⟨.program ⟨257⟩, ⟨11219⟩⟩
def rawTerms : List Term := Proof.Events238.exact60933RawTerms
def summary : Bound := (.finite 279172874240)
def resultEvent : Nat := 60933
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult60933.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 1310720
      (.finite ⟨26, by decide⟩)
      (.finite ⟨8192, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge60911.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge60911.frameStart)
    (owner := owner) (leftOwner := SemanticResult60905.owner)
    (rightOwner := SemanticResult15984.owner)
    (leftResult := 60905) (rightResult := 15984)
    (leftActual := SemanticResult60905.actual selector witness)
    (rightActual := SemanticResult15984.actual selector witness)
    (leftRaw := SemanticResult60905.rawTerms)
    (rightRaw := SemanticResult15984.rawTerms)
    (working := LeftOperatorMerge60911.working)
    (leftBinding := 60906) (rightBinding := 60907)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨11218⟩) (rightExpression := ⟨9584⟩)
    (coefficientTransfer := 60908) (summaryTransfer := 60910)
    (rightCoefficientProducer := 15983)
    (rightSummaryTransfer := 60909)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨8192, by decide⟩)
    (rightRecordedMaximum := 8192)
    (rightSummaryMaximum := ⟨8192, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 1310720)
    (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40) (base := LeftOperatorMerge60911.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftBound15983.actual selector witness)
    (summaryMagnitude := LeftBound60910.actual selector witness)
    (reconstruction := LeftOperatorMerge60911.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult60905.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult15984.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15983.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftBound15983.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge60911.operationAgreement
  · exact LeftBound60910.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge60911.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 60912 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7238⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7205⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7207⟩⟩, ⟨.program ⟨257⟩, ⟨7125⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7209⟩⟩, ⟨.program ⟨257⟩, ⟨7107⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7211⟩⟩, ⟨.program ⟨257⟩, ⟨7103⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7213⟩⟩, ⟨.program ⟨257⟩, ⟨7099⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7215⟩⟩, ⟨.program ⟨257⟩, ⟨7173⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7217⟩⟩, ⟨.program ⟨257⟩, ⟨7169⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7219⟩⟩, ⟨.program ⟨257⟩, ⟨7167⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7221⟩⟩, ⟨.program ⟨257⟩, ⟨7163⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7223⟩⟩, ⟨.program ⟨257⟩, ⟨7161⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7225⟩⟩, ⟨.program ⟨257⟩, ⟨7159⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7227⟩⟩, ⟨.program ⟨257⟩, ⟨7153⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7229⟩⟩, ⟨.program ⟨257⟩, ⟨7151⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7231⟩⟩, ⟨.program ⟨257⟩, ⟨7147⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7233⟩⟩, ⟨.program ⟨257⟩, ⟨7139⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7205⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7207⟩⟩, ⟨.program ⟨257⟩, ⟨7125⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7209⟩⟩, ⟨.program ⟨257⟩, ⟨7107⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7211⟩⟩, ⟨.program ⟨257⟩, ⟨7103⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7213⟩⟩, ⟨.program ⟨257⟩, ⟨7099⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7215⟩⟩, ⟨.program ⟨257⟩, ⟨7173⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7217⟩⟩, ⟨.program ⟨257⟩, ⟨7169⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7219⟩⟩, ⟨.program ⟨257⟩, ⟨7167⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7221⟩⟩, ⟨.program ⟨257⟩, ⟨7163⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7223⟩⟩, ⟨.program ⟨257⟩, ⟨7161⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7225⟩⟩, ⟨.program ⟨257⟩, ⟨7159⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7227⟩⟩, ⟨.program ⟨257⟩, ⟨7153⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7229⟩⟩, ⟨.program ⟨257⟩, ⟨7151⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7231⟩⟩, ⟨.program ⟨257⟩, ⟨7147⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7233⟩⟩, ⟨.program ⟨257⟩, ⟨7139⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge60911.working
    [{ coefficient := (-1), key := LeftRelationMerge60912.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge60912.frameStart
      LeftRelationMerge60912.owner (.relation 60912) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge60912.deltas
    rows := LeftRelationMerge60912.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge60911.working LeftRelationMerge60912.source
        (relationContext LeftRelationMerge60912.source
          LeftRelationMerge60912.source.centralFactors 0 2) (-1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge60911.working, LeftRelationMerge60912.deltas,
    LeftRelationMerge60912.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (witness := witness) (application := 60912)
    (frameStart := 0) (owner := ⟨.program ⟨257⟩, ⟨11219⟩⟩)
    (source := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge60911.working) (working := relationWorking0)
    (reconstruction := relationReconstruction0)
    (actual := actual selector witness) (summary := summary)
  · exact relationApplicationAt0 selector selectorLower selectorUpper
  · rfl
  · rfl
  · decide +kernel
  · exact mergeClaim selector selectorLower selectorUpper witness
  · exact relationAgreement0
  · decide +kernel
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply exactClaimAt_of_mergeClaim
    (relationClaim0 selector selectorLower selectorUpper witness)
  · decide +kernel
  · rfl
end SemanticResult60933

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
