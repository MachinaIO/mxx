import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard1355
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard000
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard136
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1256
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1329
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1330
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1332
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1333
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1334
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1336
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1337
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1339
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1340
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1341
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1343
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1344
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1354

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult192458
def owner : Owner := ⟨.program ⟨257⟩, ⟨61983⟩⟩
def rawTerms : List Term := Proof.Events751.exact192458RawTerms
def summary : Bound := (.finite 2765055493188795324243372926469393465999412)
def resultEvent : Nat := 192458
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult192458.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult192453.owner)
    (rightOwner := SemanticResult190907.owner)
    (leftResult := 192453) (rightResult := 190907)
    (leftActual := SemanticResult192453.actual selector witness)
    (rightActual := SemanticResult190907.actual selector witness)
    (leftRaw := SemanticResult192453.rawTerms)
    (rightRaw := SemanticResult190907.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 2419413932536838975995335147689984068157492)
    (rightMaximum := 345641560651956348248037778779409397841920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 192454) (rightBinding := 192455)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨59003⟩) (rightExpression := ⟨61982⟩)
    (transferEvent := 192456) (summaryTransferEvent := 192457)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult192453.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult190907.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult192458

namespace SemanticResult192463
def owner : Owner := ⟨.program ⟨257⟩, ⟨64963⟩⟩
def rawTerms : List Term := Proof.Events751.exact192463RawTerms
def summary : Bound := (.finite 3110701272581949232038858886277070355169332)
def resultEvent : Nat := 192463
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult192463.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult192458.owner)
    (rightOwner := SemanticResult190695.owner)
    (leftResult := 192458) (rightResult := 190695)
    (leftActual := SemanticResult192458.actual selector witness)
    (rightActual := SemanticResult190695.actual selector witness)
    (leftRaw := SemanticResult192458.rawTerms)
    (rightRaw := SemanticResult190695.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 2765055493188795324243372926469393465999412)
    (rightMaximum := 345645779393153907795485959807676889169920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 192459) (rightBinding := 192460)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨61983⟩) (rightExpression := ⟨64962⟩)
    (transferEvent := 192461) (summaryTransferEvent := 192462)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult192458.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult190695.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult192463

namespace SemanticResult192468
def owner : Owner := ⟨.program ⟨257⟩, ⟨70404⟩⟩
def rawTerms : List Term := Proof.Events751.exact192468RawTerms
def summary : Bound := (.finite 3456353380086899479155517117627148481331252)
def resultEvent : Nat := 192468
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult192468.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult192463.owner)
    (rightOwner := SemanticResult190483.owner)
    (leftResult := 192463) (rightResult := 190483)
    (leftActual := SemanticResult192463.actual selector witness)
    (rightActual := SemanticResult190483.actual selector witness)
    (leftRaw := SemanticResult192463.rawTerms)
    (rightRaw := SemanticResult190483.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 3110701272581949232038858886277070355169332)
    (rightMaximum := 345652107504950247116658231350078126161920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 192464) (rightBinding := 192465)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨64963⟩) (rightExpression := ⟨70403⟩)
    (transferEvent := 192466) (summaryTransferEvent := 192467)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult192463.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult190483.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult192468

namespace SemanticResult192473
def owner : Owner := ⟨.program ⟨257⟩, ⟨70405⟩⟩
def rawTerms : List Term := Proof.Events751.exact192473RawTerms
def summary : Bound := (.finite 3802007596962448506045899439491360353157172)
def resultEvent : Nat := 192473
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult192473.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult192468.owner)
    (rightOwner := SemanticResult190271.owner)
    (leftResult := 192468) (rightResult := 190271)
    (leftActual := SemanticResult192468.actual selector witness)
    (rightActual := SemanticResult190271.actual selector witness)
    (leftRaw := SemanticResult192468.rawTerms)
    (rightRaw := SemanticResult190271.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 3456353380086899479155517117627148481331252)
    (rightMaximum := 345654216875549026890382321864211871825920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 192469) (rightBinding := 192470)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨70404⟩) (rightExpression := ⟨28362⟩)
    (transferEvent := 192471) (summaryTransferEvent := 192472)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult192468.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult190271.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult192473

namespace SemanticResult192478
def owner : Owner := ⟨.program ⟨257⟩, ⟨70406⟩⟩
def rawTerms : List Term := Proof.Events751.exact192478RawTerms
def summary : Bound := (.finite 4147668141949793872257454032897973461975092)
def resultEvent : Nat := 192478
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult192478.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult192473.owner)
    (rightOwner := SemanticResult190059.owner)
    (leftResult := 192473) (rightResult := 190059)
    (leftActual := SemanticResult192473.actual selector witness)
    (rightActual := SemanticResult190059.actual selector witness)
    (leftRaw := SemanticResult192473.rawTerms)
    (rightRaw := SemanticResult190059.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 3802007596962448506045899439491360353157172)
    (rightMaximum := 345660544987345366211554593406613108817920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 192474) (rightBinding := 192475)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨70405⟩) (rightExpression := ⟨31042⟩)
    (transferEvent := 192476) (summaryTransferEvent := 192477)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult192473.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult190059.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult192478

namespace SemanticResult192483
def owner : Owner := ⟨.program ⟨257⟩, ⟨70407⟩⟩
def rawTerms : List Term := Proof.Events751.exact192483RawTerms
def summary : Bound := (.finite 4493332905678336798016456807332854062121012)
def resultEvent : Nat := 192483
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult192483.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult192478.owner)
    (rightOwner := SemanticResult189847.owner)
    (leftResult := 192478) (rightResult := 189847)
    (leftActual := SemanticResult192478.actual selector witness)
    (rightActual := SemanticResult189847.actual selector witness)
    (leftRaw := SemanticResult192478.rawTerms)
    (rightRaw := SemanticResult189847.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 4147668141949793872257454032897973461975092)
    (rightMaximum := 345664763728542925759002774434880600145920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 192479) (rightBinding := 192480)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨70406⟩) (rightExpression := ⟨36702⟩)
    (transferEvent := 192481) (summaryTransferEvent := 192482)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult192478.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult189847.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult192483

namespace SemanticResult192488
def owner : Owner := ⟨.program ⟨257⟩, ⟨70408⟩⟩
def rawTerms : List Term := Proof.Events751.exact192488RawTerms
def summary : Bound := (.finite 4838999778777478503549183672281868407930932)
def resultEvent : Nat := 192488
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult192488.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult192483.owner)
    (rightOwner := SemanticResult189635.owner)
    (leftResult := 192483) (rightResult := 189635)
    (leftActual := SemanticResult192483.actual selector witness)
    (rightActual := SemanticResult189635.actual selector witness)
    (leftRaw := SemanticResult192483.rawTerms)
    (rightRaw := SemanticResult189635.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 4493332905678336798016456807332854062121012)
    (rightMaximum := 345666873099141705532726864949014345809920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 192484) (rightBinding := 192485)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨70407⟩) (rightExpression := ⟨39382⟩)
    (transferEvent := 192486) (summaryTransferEvent := 192487)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult192483.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult189635.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult192488

namespace SemanticResult192493
def owner : Owner := ⟨.program ⟨257⟩, ⟨70409⟩⟩
def rawTerms : List Term := Proof.Events751.exact192493RawTerms
def summary : Bound := (.finite 5184670870617817768629358718259150245068852)
def resultEvent : Nat := 192493
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult192493.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult192488.owner)
    (rightOwner := SemanticResult189423.owner)
    (leftResult := 192488) (rightResult := 189423)
    (leftActual := SemanticResult192488.actual selector witness)
    (rightActual := SemanticResult189423.actual selector witness)
    (leftRaw := SemanticResult192488.rawTerms)
    (rightRaw := SemanticResult189423.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 4838999778777478503549183672281868407930932)
    (rightMaximum := 345671091840339265080175045977281837137920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 192489) (rightBinding := 192490)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨70408⟩) (rightExpression := ⟨42062⟩)
    (transferEvent := 192491) (summaryTransferEvent := 192492)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult192488.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult189423.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult192493

namespace SemanticResult192498
def owner : Owner := ⟨.program ⟨257⟩, ⟨70410⟩⟩
def rawTerms : List Term := Proof.Events751.exact192498RawTerms
def summary : Bound := (.finite 5530348290569953373030706035778833319198772)
def resultEvent : Nat := 192498
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult192498.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult192493.owner)
    (rightOwner := SemanticResult189211.owner)
    (leftResult := 192493) (rightResult := 189211)
    (leftActual := SemanticResult192493.actual selector witness)
    (rightActual := SemanticResult189211.actual selector witness)
    (leftRaw := SemanticResult192493.rawTerms)
    (rightRaw := SemanticResult189211.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 5184670870617817768629358718259150245068852)
    (rightMaximum := 345677419952135604401347317519683074129920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 192494) (rightBinding := 192495)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨70409⟩) (rightExpression := ⟨44742⟩)
    (transferEvent := 192496) (summaryTransferEvent := 192497)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult192493.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult189211.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult192498

namespace SemanticResult192503
def owner : Owner := ⟨.program ⟨257⟩, ⟨70411⟩⟩
def rawTerms : List Term := Proof.Events751.exact192503RawTerms
def summary : Bound := (.finite 5876032038633885316753225624840917630320692)
def resultEvent : Nat := 192503
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult192503.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult192498.owner)
    (rightOwner := SemanticResult188999.owner)
    (leftResult := 192498) (rightResult := 188999)
    (leftActual := SemanticResult192498.actual selector witness)
    (rightActual := SemanticResult188999.actual selector witness)
    (leftRaw := SemanticResult192498.rawTerms)
    (rightRaw := SemanticResult188999.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 5530348290569953373030706035778833319198772)
    (rightMaximum := 345683748063931943722519589062084311121920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 192499) (rightBinding := 192500)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨70410⟩) (rightExpression := ⟨47422⟩)
    (transferEvent := 192501) (summaryTransferEvent := 192502)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult192498.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult188999.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult192503

namespace SemanticResult192508
def owner : Owner := ⟨.program ⟨257⟩, ⟨70412⟩⟩
def rawTerms : List Term := Proof.Events751.exact192508RawTerms
def summary : Bound := (.finite 6221717896068416040249469304417135687106612)
def resultEvent : Nat := 192508
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult192508.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult192503.owner)
    (rightOwner := SemanticResult188787.owner)
    (leftResult := 192503) (rightResult := 188787)
    (leftActual := SemanticResult192503.actual selector witness)
    (rightActual := SemanticResult188787.actual selector witness)
    (leftRaw := SemanticResult192503.rawTerms)
    (rightRaw := SemanticResult188787.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 5876032038633885316753225624840917630320692)
    (rightMaximum := 345685857434530723496243679576218056785920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 192504) (rightBinding := 192505)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨70411⟩) (rightExpression := ⟨50102⟩)
    (transferEvent := 192506) (summaryTransferEvent := 192507)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult192503.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult188787.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult192508

namespace SemanticResult192513
def owner : Owner := ⟨.program ⟨257⟩, ⟨71335⟩⟩
def rawTerms : List Term := Proof.Events752.exact192513RawTerms
def summary : Bound := (.finite 66805187227601152574551644069558752530002096506798132)
def resultEvent : Nat := 192513
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult192513.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult192508.owner)
    (rightOwner := SemanticResult188575.owner)
    (leftResult := 192508) (rightResult := 188575)
    (leftActual := SemanticResult192508.actual selector witness)
    (rightActual := SemanticResult188575.actual selector witness)
    (leftRaw := SemanticResult192508.rawTerms)
    (rightRaw := SemanticResult188575.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 6221717896068416040249469304417135687106612)
    (rightMaximum := 66805187221379434678483228029309283225584960819691520) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 192509) (rightBinding := 192510)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨70412⟩) (rightExpression := ⟨71333⟩)
    (transferEvent := 192511) (summaryTransferEvent := 192512)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult192508.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult188575.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult192513

namespace SemanticResult192515
def owner : Owner := ⟨.program ⟨257⟩, ⟨8⟩⟩
def rawTerms : List Term := Proof.Events752.exact192515RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 192515
def producerEvent : Nat := 192514
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult192515.actual selector witness
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
end SemanticResult192515

namespace SemanticResult192520
def owner : Owner := ⟨.program ⟨257⟩, ⟨7412⟩⟩
def rawTerms : List Term := Proof.Events752.exact192520RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 192520
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult192520.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge192519.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge192519.frameStart)
    (transferEvent := 192518) (owner := owner)
    (leftResult := 27) (rightResult := 16427)
    (working := LeftOperatorMerge192519.working)
    (reconstruction := LeftOperatorMerge192519.reconstruction)
    (leftReference := .predecessor 0 192516 .coefficient) (rightReference := .predecessor 1 192517 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult27.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult16427.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge192519.operationAgreement
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
end SemanticResult192520

namespace SemanticResult192524
def owner : Owner := ⟨.program ⟨257⟩, ⟨9227⟩⟩
def rawTerms : List Term := Proof.Events752.exact192524RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 192524
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult192524.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 192521) (rightBinding := 192522)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7412⟩) (rightExpression := ⟨7004⟩)
    (transferEvent := 192523)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult192520.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult178278.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult192524

namespace SemanticResult192530
def owner : Owner := ⟨.program ⟨257⟩, ⟨9228⟩⟩
def rawTerms : List Term := Proof.Events752.exact192530RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 192530
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult192530.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (coefficientTransfer := 192527) (survivorTransfer := 192528)
    (survivorEvent := 192529) (resultEvent := resultEvent)
    (rightCoefficientProducer := 192514)
    (owner := owner) (leftOwner := SemanticResult192524.owner)
    (rightOwner := SemanticResult192515.owner)
    (leftResult := 192524) (rightResult := 192515)
    (leftBinding := 192525) (rightBinding := 192526)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨9227⟩) (rightExpression := ⟨8⟩)
    (leftActual := SemanticResult192524.actual selector witness)
    (rightActual := SemanticResult192515.actual selector witness)
    (leftRaw := SemanticResult192524.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨257⟩, ⟨8⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftAuthority192514.actual selector witness)
    (survivorMagnitude := LeftBound192528.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult192524.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult192515.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority192514.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority192514.derived selector witness)
  · exact LeftBound192528.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult192530

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
