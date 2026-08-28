import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard852
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard826
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard828
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard829
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard830
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard832
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard833
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard834
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard836
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard837
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard839
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard840
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard841
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard843
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard844
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard845
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard851

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult119318
def owner : Owner := ⟨.program ⟨257⟩, ⟨52981⟩⟩
def rawTerms : List Term := Proof.Events466.exact119318RawTerms
def summary : Bound := (.finite 1728139248715321398594155952187700255129652)
def resultEvent : Nat := 119318
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult119318.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult119313.owner)
    (rightOwner := SemanticResult118418.owner)
    (leftResult := 119313) (rightResult := 118418)
    (leftActual := SemanticResult119313.actual selector witness)
    (rightActual := SemanticResult118418.actual selector witness)
    (leftRaw := SemanticResult119313.rawTerms)
    (rightRaw := SemanticResult118418.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 1382506125545760169441014535464825839943732)
    (rightMaximum := 345633123169561229153141416722874415185920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 119314) (rightBinding := 119315)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨33921⟩) (rightExpression := ⟨52980⟩)
    (transferEvent := 119316) (summaryTransferEvent := 119317)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult119313.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult118418.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult119318

namespace SemanticResult119323
def owner : Owner := ⟨.program ⟨257⟩, ⟨55961⟩⟩
def rawTerms : List Term := Proof.Events466.exact119323RawTerms
def summary : Bound := (.finite 2073774481255481407521021459424708415979572)
def resultEvent : Nat := 119323
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult119323.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult119318.owner)
    (rightOwner := SemanticResult118206.owner)
    (leftResult := 119318) (rightResult := 118206)
    (leftActual := SemanticResult119318.actual selector witness)
    (rightActual := SemanticResult118206.actual selector witness)
    (leftRaw := SemanticResult119318.rawTerms)
    (rightRaw := SemanticResult118206.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 1728139248715321398594155952187700255129652)
    (rightMaximum := 345635232540160008926865507237008160849920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 119319) (rightBinding := 119320)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨52981⟩) (rightExpression := ⟨55960⟩)
    (transferEvent := 119321) (summaryTransferEvent := 119322)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult119318.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult118206.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult119323

namespace SemanticResult119328
def owner : Owner := ⟨.program ⟨257⟩, ⟨58941⟩⟩
def rawTerms : List Term := Proof.Events466.exact119328RawTerms
def summary : Bound := (.finite 2419413932536838975995335147689984068157492)
def resultEvent : Nat := 119328
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult119328.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult119323.owner)
    (rightOwner := SemanticResult117994.owner)
    (leftResult := 119323) (rightResult := 117994)
    (leftActual := SemanticResult119323.actual selector witness)
    (rightActual := SemanticResult117994.actual selector witness)
    (leftRaw := SemanticResult119323.rawTerms)
    (rightRaw := SemanticResult117994.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 2073774481255481407521021459424708415979572)
    (rightMaximum := 345639451281357568474313688265275652177920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 119324) (rightBinding := 119325)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨55961⟩) (rightExpression := ⟨58940⟩)
    (transferEvent := 119326) (summaryTransferEvent := 119327)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult119323.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult117994.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult119328

namespace SemanticResult119333
def owner : Owner := ⟨.program ⟨257⟩, ⟨61921⟩⟩
def rawTerms : List Term := Proof.Events466.exact119333RawTerms
def summary : Bound := (.finite 2765055493188795324243372926469393465999412)
def resultEvent : Nat := 119333
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult119333.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult119328.owner)
    (rightOwner := SemanticResult117782.owner)
    (leftResult := 119328) (rightResult := 117782)
    (leftActual := SemanticResult119328.actual selector witness)
    (rightActual := SemanticResult117782.actual selector witness)
    (leftRaw := SemanticResult119328.rawTerms)
    (rightRaw := SemanticResult117782.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 2419413932536838975995335147689984068157492)
    (rightMaximum := 345641560651956348248037778779409397841920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 119329) (rightBinding := 119330)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨58941⟩) (rightExpression := ⟨61920⟩)
    (transferEvent := 119331) (summaryTransferEvent := 119332)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult119328.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult117782.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult119333

namespace SemanticResult119338
def owner : Owner := ⟨.program ⟨257⟩, ⟨64901⟩⟩
def rawTerms : List Term := Proof.Events466.exact119338RawTerms
def summary : Bound := (.finite 3110701272581949232038858886277070355169332)
def resultEvent : Nat := 119338
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult119338.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult119333.owner)
    (rightOwner := SemanticResult117570.owner)
    (leftResult := 119333) (rightResult := 117570)
    (leftActual := SemanticResult119333.actual selector witness)
    (rightActual := SemanticResult117570.actual selector witness)
    (leftRaw := SemanticResult119333.rawTerms)
    (rightRaw := SemanticResult117570.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 2765055493188795324243372926469393465999412)
    (rightMaximum := 345645779393153907795485959807676889169920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 119334) (rightBinding := 119335)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨61921⟩) (rightExpression := ⟨64900⟩)
    (transferEvent := 119336) (summaryTransferEvent := 119337)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult119333.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult117570.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult119338

namespace SemanticResult119343
def owner : Owner := ⟨.program ⟨257⟩, ⟨70246⟩⟩
def rawTerms : List Term := Proof.Events466.exact119343RawTerms
def summary : Bound := (.finite 3456353380086899479155517117627148481331252)
def resultEvent : Nat := 119343
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult119343.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult119338.owner)
    (rightOwner := SemanticResult117358.owner)
    (leftResult := 119338) (rightResult := 117358)
    (leftActual := SemanticResult119338.actual selector witness)
    (rightActual := SemanticResult117358.actual selector witness)
    (leftRaw := SemanticResult119338.rawTerms)
    (rightRaw := SemanticResult117358.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 3110701272581949232038858886277070355169332)
    (rightMaximum := 345652107504950247116658231350078126161920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 119339) (rightBinding := 119340)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨64901⟩) (rightExpression := ⟨70245⟩)
    (transferEvent := 119341) (summaryTransferEvent := 119342)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult119338.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult117358.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult119343

namespace SemanticResult119348
def owner : Owner := ⟨.program ⟨257⟩, ⟨70247⟩⟩
def rawTerms : List Term := Proof.Events466.exact119348RawTerms
def summary : Bound := (.finite 3802007596962448506045899439491360353157172)
def resultEvent : Nat := 119348
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult119348.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult119343.owner)
    (rightOwner := SemanticResult117146.owner)
    (leftResult := 119343) (rightResult := 117146)
    (leftActual := SemanticResult119343.actual selector witness)
    (rightActual := SemanticResult117146.actual selector witness)
    (leftRaw := SemanticResult119343.rawTerms)
    (rightRaw := SemanticResult117146.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 3456353380086899479155517117627148481331252)
    (rightMaximum := 345654216875549026890382321864211871825920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 119344) (rightBinding := 119345)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨70246⟩) (rightExpression := ⟨28312⟩)
    (transferEvent := 119346) (summaryTransferEvent := 119347)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult119343.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult117146.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult119348

namespace SemanticResult119353
def owner : Owner := ⟨.program ⟨257⟩, ⟨70248⟩⟩
def rawTerms : List Term := Proof.Events466.exact119353RawTerms
def summary : Bound := (.finite 4147668141949793872257454032897973461975092)
def resultEvent : Nat := 119353
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult119353.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult119348.owner)
    (rightOwner := SemanticResult116934.owner)
    (leftResult := 119348) (rightResult := 116934)
    (leftActual := SemanticResult119348.actual selector witness)
    (rightActual := SemanticResult116934.actual selector witness)
    (leftRaw := SemanticResult119348.rawTerms)
    (rightRaw := SemanticResult116934.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 3802007596962448506045899439491360353157172)
    (rightMaximum := 345660544987345366211554593406613108817920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 119349) (rightBinding := 119350)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨70247⟩) (rightExpression := ⟨30992⟩)
    (transferEvent := 119351) (summaryTransferEvent := 119352)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult119348.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult116934.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult119353

namespace SemanticResult119358
def owner : Owner := ⟨.program ⟨257⟩, ⟨70249⟩⟩
def rawTerms : List Term := Proof.Events466.exact119358RawTerms
def summary : Bound := (.finite 4493332905678336798016456807332854062121012)
def resultEvent : Nat := 119358
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult119358.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult119353.owner)
    (rightOwner := SemanticResult116722.owner)
    (leftResult := 119353) (rightResult := 116722)
    (leftActual := SemanticResult119353.actual selector witness)
    (rightActual := SemanticResult116722.actual selector witness)
    (leftRaw := SemanticResult119353.rawTerms)
    (rightRaw := SemanticResult116722.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 4147668141949793872257454032897973461975092)
    (rightMaximum := 345664763728542925759002774434880600145920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 119354) (rightBinding := 119355)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨70248⟩) (rightExpression := ⟨36652⟩)
    (transferEvent := 119356) (summaryTransferEvent := 119357)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult119353.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult116722.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult119358

namespace SemanticResult119363
def owner : Owner := ⟨.program ⟨257⟩, ⟨70250⟩⟩
def rawTerms : List Term := Proof.Events466.exact119363RawTerms
def summary : Bound := (.finite 4838999778777478503549183672281868407930932)
def resultEvent : Nat := 119363
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult119363.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult119358.owner)
    (rightOwner := SemanticResult116510.owner)
    (leftResult := 119358) (rightResult := 116510)
    (leftActual := SemanticResult119358.actual selector witness)
    (rightActual := SemanticResult116510.actual selector witness)
    (leftRaw := SemanticResult119358.rawTerms)
    (rightRaw := SemanticResult116510.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 4493332905678336798016456807332854062121012)
    (rightMaximum := 345666873099141705532726864949014345809920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 119359) (rightBinding := 119360)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨70249⟩) (rightExpression := ⟨39332⟩)
    (transferEvent := 119361) (summaryTransferEvent := 119362)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult119358.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult116510.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult119363

namespace SemanticResult119368
def owner : Owner := ⟨.program ⟨257⟩, ⟨70251⟩⟩
def rawTerms : List Term := Proof.Events466.exact119368RawTerms
def summary : Bound := (.finite 5184670870617817768629358718259150245068852)
def resultEvent : Nat := 119368
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult119368.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult119363.owner)
    (rightOwner := SemanticResult116298.owner)
    (leftResult := 119363) (rightResult := 116298)
    (leftActual := SemanticResult119363.actual selector witness)
    (rightActual := SemanticResult116298.actual selector witness)
    (leftRaw := SemanticResult119363.rawTerms)
    (rightRaw := SemanticResult116298.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 4838999778777478503549183672281868407930932)
    (rightMaximum := 345671091840339265080175045977281837137920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 119364) (rightBinding := 119365)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨70250⟩) (rightExpression := ⟨42012⟩)
    (transferEvent := 119366) (summaryTransferEvent := 119367)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult119363.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult116298.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult119368

namespace SemanticResult119373
def owner : Owner := ⟨.program ⟨257⟩, ⟨70252⟩⟩
def rawTerms : List Term := Proof.Events466.exact119373RawTerms
def summary : Bound := (.finite 5530348290569953373030706035778833319198772)
def resultEvent : Nat := 119373
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult119373.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult119368.owner)
    (rightOwner := SemanticResult116086.owner)
    (leftResult := 119368) (rightResult := 116086)
    (leftActual := SemanticResult119368.actual selector witness)
    (rightActual := SemanticResult116086.actual selector witness)
    (leftRaw := SemanticResult119368.rawTerms)
    (rightRaw := SemanticResult116086.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 5184670870617817768629358718259150245068852)
    (rightMaximum := 345677419952135604401347317519683074129920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 119369) (rightBinding := 119370)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨70251⟩) (rightExpression := ⟨44692⟩)
    (transferEvent := 119371) (summaryTransferEvent := 119372)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult119368.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult116086.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult119373

namespace SemanticResult119378
def owner : Owner := ⟨.program ⟨257⟩, ⟨70253⟩⟩
def rawTerms : List Term := Proof.Events466.exact119378RawTerms
def summary : Bound := (.finite 5876032038633885316753225624840917630320692)
def resultEvent : Nat := 119378
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult119378.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult119373.owner)
    (rightOwner := SemanticResult115874.owner)
    (leftResult := 119373) (rightResult := 115874)
    (leftActual := SemanticResult119373.actual selector witness)
    (rightActual := SemanticResult115874.actual selector witness)
    (leftRaw := SemanticResult119373.rawTerms)
    (rightRaw := SemanticResult115874.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 5530348290569953373030706035778833319198772)
    (rightMaximum := 345683748063931943722519589062084311121920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 119374) (rightBinding := 119375)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨70252⟩) (rightExpression := ⟨47372⟩)
    (transferEvent := 119376) (summaryTransferEvent := 119377)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult119373.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult115874.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult119378

namespace SemanticResult119383
def owner : Owner := ⟨.program ⟨257⟩, ⟨70254⟩⟩
def rawTerms : List Term := Proof.Events466.exact119383RawTerms
def summary : Bound := (.finite 6221717896068416040249469304417135687106612)
def resultEvent : Nat := 119383
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult119383.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult119378.owner)
    (rightOwner := SemanticResult115662.owner)
    (leftResult := 119378) (rightResult := 115662)
    (leftActual := SemanticResult119378.actual selector witness)
    (rightActual := SemanticResult115662.actual selector witness)
    (leftRaw := SemanticResult119378.rawTerms)
    (rightRaw := SemanticResult115662.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 5876032038633885316753225624840917630320692)
    (rightMaximum := 345685857434530723496243679576218056785920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 119379) (rightBinding := 119380)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨70253⟩) (rightExpression := ⟨50052⟩)
    (transferEvent := 119381) (summaryTransferEvent := 119382)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult119378.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult115662.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult119383

namespace SemanticResult119388
def owner : Owner := ⟨.program ⟨257⟩, ⟨71273⟩⟩
def rawTerms : List Term := Proof.Events466.exact119388RawTerms
def summary : Bound := (.finite 66805187227601152574551644069558752530002096506798132)
def resultEvent : Nat := 119388
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult119388.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult119383.owner)
    (rightOwner := SemanticResult115450.owner)
    (leftResult := 119383) (rightResult := 115450)
    (leftActual := SemanticResult119383.actual selector witness)
    (rightActual := SemanticResult115450.actual selector witness)
    (leftRaw := SemanticResult119383.rawTerms)
    (rightRaw := SemanticResult115450.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 6221717896068416040249469304417135687106612)
    (rightMaximum := 66805187221379434678483228029309283225584960819691520) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 119384) (rightBinding := 119385)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨70254⟩) (rightExpression := ⟨71271⟩)
    (transferEvent := 119386) (summaryTransferEvent := 119387)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult119383.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult115450.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult119388

namespace SemanticResult119390
def owner : Owner := ⟨.program ⟨257⟩, ⟨19⟩⟩
def rawTerms : List Term := Proof.Events466.exact119390RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 119390
def producerEvent : Nat := 119389
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult119390.actual selector witness
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
end SemanticResult119390

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
