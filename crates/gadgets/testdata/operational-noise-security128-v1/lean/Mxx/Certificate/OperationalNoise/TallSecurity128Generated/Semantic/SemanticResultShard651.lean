import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard651
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard000
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard131
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard552
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard625
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard626
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard628
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard629
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard631
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard632
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard633
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard635
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard636
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard637
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard639
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard640
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard642
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard650

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult90078
def owner : Owner := ⟨.program ⟨257⟩, ⟨59096⟩⟩
def rawTerms : List Term := Proof.Events351.exact90078RawTerms
def summary : Bound := (.finite 2419413932536838975995335147689984068157492)
def resultEvent : Nat := 90078
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult90078.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult90073.owner)
    (rightOwner := SemanticResult88744.owner)
    (leftResult := 90073) (rightResult := 88744)
    (leftActual := SemanticResult90073.actual selector witness)
    (rightActual := SemanticResult88744.actual selector witness)
    (leftRaw := SemanticResult90073.rawTerms)
    (rightRaw := SemanticResult88744.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 2073774481255481407521021459424708415979572)
    (rightMaximum := 345639451281357568474313688265275652177920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 90074) (rightBinding := 90075)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨56116⟩) (rightExpression := ⟨59095⟩)
    (transferEvent := 90076) (summaryTransferEvent := 90077)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult90073.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult88744.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult90078

namespace SemanticResult90083
def owner : Owner := ⟨.program ⟨257⟩, ⟨62076⟩⟩
def rawTerms : List Term := Proof.Events351.exact90083RawTerms
def summary : Bound := (.finite 2765055493188795324243372926469393465999412)
def resultEvent : Nat := 90083
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult90083.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult90078.owner)
    (rightOwner := SemanticResult88532.owner)
    (leftResult := 90078) (rightResult := 88532)
    (leftActual := SemanticResult90078.actual selector witness)
    (rightActual := SemanticResult88532.actual selector witness)
    (leftRaw := SemanticResult90078.rawTerms)
    (rightRaw := SemanticResult88532.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 2419413932536838975995335147689984068157492)
    (rightMaximum := 345641560651956348248037778779409397841920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 90079) (rightBinding := 90080)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨59096⟩) (rightExpression := ⟨62075⟩)
    (transferEvent := 90081) (summaryTransferEvent := 90082)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult90078.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult88532.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult90083

namespace SemanticResult90088
def owner : Owner := ⟨.program ⟨257⟩, ⟨65056⟩⟩
def rawTerms : List Term := Proof.Events351.exact90088RawTerms
def summary : Bound := (.finite 3110701272581949232038858886277070355169332)
def resultEvent : Nat := 90088
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult90088.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult90083.owner)
    (rightOwner := SemanticResult88320.owner)
    (leftResult := 90083) (rightResult := 88320)
    (leftActual := SemanticResult90083.actual selector witness)
    (rightActual := SemanticResult88320.actual selector witness)
    (leftRaw := SemanticResult90083.rawTerms)
    (rightRaw := SemanticResult88320.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 2765055493188795324243372926469393465999412)
    (rightMaximum := 345645779393153907795485959807676889169920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 90084) (rightBinding := 90085)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨62076⟩) (rightExpression := ⟨65055⟩)
    (transferEvent := 90086) (summaryTransferEvent := 90087)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult90083.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult88320.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult90088

namespace SemanticResult90093
def owner : Owner := ⟨.program ⟨257⟩, ⟨70641⟩⟩
def rawTerms : List Term := Proof.Events351.exact90093RawTerms
def summary : Bound := (.finite 3456353380086899479155517117627148481331252)
def resultEvent : Nat := 90093
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult90093.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult90088.owner)
    (rightOwner := SemanticResult88108.owner)
    (leftResult := 90088) (rightResult := 88108)
    (leftActual := SemanticResult90088.actual selector witness)
    (rightActual := SemanticResult88108.actual selector witness)
    (leftRaw := SemanticResult90088.rawTerms)
    (rightRaw := SemanticResult88108.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 3110701272581949232038858886277070355169332)
    (rightMaximum := 345652107504950247116658231350078126161920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 90089) (rightBinding := 90090)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨65056⟩) (rightExpression := ⟨70640⟩)
    (transferEvent := 90091) (summaryTransferEvent := 90092)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult90088.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult88108.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult90093

namespace SemanticResult90098
def owner : Owner := ⟨.program ⟨257⟩, ⟨70642⟩⟩
def rawTerms : List Term := Proof.Events351.exact90098RawTerms
def summary : Bound := (.finite 3802007596962448506045899439491360353157172)
def resultEvent : Nat := 90098
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult90098.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult90093.owner)
    (rightOwner := SemanticResult87896.owner)
    (leftResult := 90093) (rightResult := 87896)
    (leftActual := SemanticResult90093.actual selector witness)
    (rightActual := SemanticResult87896.actual selector witness)
    (leftRaw := SemanticResult90093.rawTerms)
    (rightRaw := SemanticResult87896.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 3456353380086899479155517117627148481331252)
    (rightMaximum := 345654216875549026890382321864211871825920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 90094) (rightBinding := 90095)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨70641⟩) (rightExpression := ⟨28437⟩)
    (transferEvent := 90096) (summaryTransferEvent := 90097)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult90093.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult87896.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult90098

namespace SemanticResult90103
def owner : Owner := ⟨.program ⟨257⟩, ⟨70643⟩⟩
def rawTerms : List Term := Proof.Events351.exact90103RawTerms
def summary : Bound := (.finite 4147668141949793872257454032897973461975092)
def resultEvent : Nat := 90103
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult90103.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult90098.owner)
    (rightOwner := SemanticResult87684.owner)
    (leftResult := 90098) (rightResult := 87684)
    (leftActual := SemanticResult90098.actual selector witness)
    (rightActual := SemanticResult87684.actual selector witness)
    (leftRaw := SemanticResult90098.rawTerms)
    (rightRaw := SemanticResult87684.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 3802007596962448506045899439491360353157172)
    (rightMaximum := 345660544987345366211554593406613108817920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 90099) (rightBinding := 90100)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨70642⟩) (rightExpression := ⟨31117⟩)
    (transferEvent := 90101) (summaryTransferEvent := 90102)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult90098.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult87684.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult90103

namespace SemanticResult90108
def owner : Owner := ⟨.program ⟨257⟩, ⟨70644⟩⟩
def rawTerms : List Term := Proof.Events351.exact90108RawTerms
def summary : Bound := (.finite 4493332905678336798016456807332854062121012)
def resultEvent : Nat := 90108
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult90108.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult90103.owner)
    (rightOwner := SemanticResult87472.owner)
    (leftResult := 90103) (rightResult := 87472)
    (leftActual := SemanticResult90103.actual selector witness)
    (rightActual := SemanticResult87472.actual selector witness)
    (leftRaw := SemanticResult90103.rawTerms)
    (rightRaw := SemanticResult87472.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 4147668141949793872257454032897973461975092)
    (rightMaximum := 345664763728542925759002774434880600145920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 90104) (rightBinding := 90105)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨70643⟩) (rightExpression := ⟨36777⟩)
    (transferEvent := 90106) (summaryTransferEvent := 90107)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult90103.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult87472.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult90108

namespace SemanticResult90113
def owner : Owner := ⟨.program ⟨257⟩, ⟨70645⟩⟩
def rawTerms : List Term := Proof.Events352.exact90113RawTerms
def summary : Bound := (.finite 4838999778777478503549183672281868407930932)
def resultEvent : Nat := 90113
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult90113.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult90108.owner)
    (rightOwner := SemanticResult87260.owner)
    (leftResult := 90108) (rightResult := 87260)
    (leftActual := SemanticResult90108.actual selector witness)
    (rightActual := SemanticResult87260.actual selector witness)
    (leftRaw := SemanticResult90108.rawTerms)
    (rightRaw := SemanticResult87260.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 4493332905678336798016456807332854062121012)
    (rightMaximum := 345666873099141705532726864949014345809920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 90109) (rightBinding := 90110)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨70644⟩) (rightExpression := ⟨39457⟩)
    (transferEvent := 90111) (summaryTransferEvent := 90112)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult90108.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult87260.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult90113

namespace SemanticResult90118
def owner : Owner := ⟨.program ⟨257⟩, ⟨70646⟩⟩
def rawTerms : List Term := Proof.Events352.exact90118RawTerms
def summary : Bound := (.finite 5184670870617817768629358718259150245068852)
def resultEvent : Nat := 90118
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult90118.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult90113.owner)
    (rightOwner := SemanticResult87048.owner)
    (leftResult := 90113) (rightResult := 87048)
    (leftActual := SemanticResult90113.actual selector witness)
    (rightActual := SemanticResult87048.actual selector witness)
    (leftRaw := SemanticResult90113.rawTerms)
    (rightRaw := SemanticResult87048.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 4838999778777478503549183672281868407930932)
    (rightMaximum := 345671091840339265080175045977281837137920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 90114) (rightBinding := 90115)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨70645⟩) (rightExpression := ⟨42137⟩)
    (transferEvent := 90116) (summaryTransferEvent := 90117)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult90113.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult87048.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult90118

namespace SemanticResult90123
def owner : Owner := ⟨.program ⟨257⟩, ⟨70647⟩⟩
def rawTerms : List Term := Proof.Events352.exact90123RawTerms
def summary : Bound := (.finite 5530348290569953373030706035778833319198772)
def resultEvent : Nat := 90123
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult90123.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult90118.owner)
    (rightOwner := SemanticResult86836.owner)
    (leftResult := 90118) (rightResult := 86836)
    (leftActual := SemanticResult90118.actual selector witness)
    (rightActual := SemanticResult86836.actual selector witness)
    (leftRaw := SemanticResult90118.rawTerms)
    (rightRaw := SemanticResult86836.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 5184670870617817768629358718259150245068852)
    (rightMaximum := 345677419952135604401347317519683074129920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 90119) (rightBinding := 90120)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨70646⟩) (rightExpression := ⟨44817⟩)
    (transferEvent := 90121) (summaryTransferEvent := 90122)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult90118.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult86836.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult90123

namespace SemanticResult90128
def owner : Owner := ⟨.program ⟨257⟩, ⟨70648⟩⟩
def rawTerms : List Term := Proof.Events352.exact90128RawTerms
def summary : Bound := (.finite 5876032038633885316753225624840917630320692)
def resultEvent : Nat := 90128
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult90128.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult90123.owner)
    (rightOwner := SemanticResult86624.owner)
    (leftResult := 90123) (rightResult := 86624)
    (leftActual := SemanticResult90123.actual selector witness)
    (rightActual := SemanticResult86624.actual selector witness)
    (leftRaw := SemanticResult90123.rawTerms)
    (rightRaw := SemanticResult86624.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 5530348290569953373030706035778833319198772)
    (rightMaximum := 345683748063931943722519589062084311121920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 90124) (rightBinding := 90125)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨70647⟩) (rightExpression := ⟨47497⟩)
    (transferEvent := 90126) (summaryTransferEvent := 90127)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult90123.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult86624.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult90128

namespace SemanticResult90133
def owner : Owner := ⟨.program ⟨257⟩, ⟨70649⟩⟩
def rawTerms : List Term := Proof.Events352.exact90133RawTerms
def summary : Bound := (.finite 6221717896068416040249469304417135687106612)
def resultEvent : Nat := 90133
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult90133.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult90128.owner)
    (rightOwner := SemanticResult86412.owner)
    (leftResult := 90128) (rightResult := 86412)
    (leftActual := SemanticResult90128.actual selector witness)
    (rightActual := SemanticResult86412.actual selector witness)
    (leftRaw := SemanticResult90128.rawTerms)
    (rightRaw := SemanticResult86412.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 5876032038633885316753225624840917630320692)
    (rightMaximum := 345685857434530723496243679576218056785920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 90129) (rightBinding := 90130)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨70648⟩) (rightExpression := ⟨50177⟩)
    (transferEvent := 90131) (summaryTransferEvent := 90132)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult90128.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult86412.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult90133

namespace SemanticResult90138
def owner : Owner := ⟨.program ⟨257⟩, ⟨71443⟩⟩
def rawTerms : List Term := Proof.Events352.exact90138RawTerms
def summary : Bound := (.finite 66805187227601152574551644069558752530002096506798132)
def resultEvent : Nat := 90138
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult90138.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult90133.owner)
    (rightOwner := SemanticResult86200.owner)
    (leftResult := 90133) (rightResult := 86200)
    (leftActual := SemanticResult90133.actual selector witness)
    (rightActual := SemanticResult86200.actual selector witness)
    (leftRaw := SemanticResult90133.rawTerms)
    (rightRaw := SemanticResult86200.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 6221717896068416040249469304417135687106612)
    (rightMaximum := 66805187221379434678483228029309283225584960819691520) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 90134) (rightBinding := 90135)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨70649⟩) (rightExpression := ⟨71441⟩)
    (transferEvent := 90136) (summaryTransferEvent := 90137)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult90133.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult86200.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult90138

namespace SemanticResult90140
def owner : Owner := ⟨.program ⟨257⟩, ⟨24⟩⟩
def rawTerms : List Term := Proof.Events352.exact90140RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 90140
def producerEvent : Nat := 90139
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult90140.actual selector witness
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
end SemanticResult90140

namespace SemanticResult90145
def owner : Owner := ⟨.program ⟨257⟩, ⟨7405⟩⟩
def rawTerms : List Term := Proof.Events352.exact90145RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 90145
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult90145.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge90144.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge90144.frameStart)
    (transferEvent := 90143) (owner := owner)
    (leftResult := 27) (rightResult := 16147)
    (working := LeftOperatorMerge90144.working)
    (reconstruction := LeftOperatorMerge90144.reconstruction)
    (leftReference := .predecessor 0 90141 .coefficient) (rightReference := .predecessor 1 90142 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult27.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult16147.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge90144.operationAgreement
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
end SemanticResult90145

namespace SemanticResult90149
def owner : Owner := ⟨.program ⟨257⟩, ⟨10369⟩⟩
def rawTerms : List Term := Proof.Events352.exact90149RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 90149
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult90149.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 90146) (rightBinding := 90147)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7405⟩) (rightExpression := ⟨10328⟩)
    (transferEvent := 90148)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult90145.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult75903.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult90149

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
