import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard349
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard326
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard327
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard329
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard330
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard332
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard333
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard334
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard336
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard337
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard338
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard340
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard341
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard343
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard344
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard345
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard347
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard348

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult46178
def owner : Owner := ⟨.program ⟨257⟩, ⟨20929⟩⟩
def rawTerms : List Term := Proof.Events180.exact46178RawTerms
def summary : Bound := (.finite 691250426059631610003352154589745737891892)
def resultEvent : Nat := 46178
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult46178.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult46173.owner)
    (rightOwner := SemanticResult45929.owner)
    (leftResult := 46173) (rightResult := 45929)
    (leftActual := SemanticResult46173.actual selector witness)
    (rightActual := SemanticResult45929.actual selector witness)
    (leftRaw := SemanticResult46173.rawTerms)
    (rightRaw := SemanticResult45929.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 345624685687166110058245054666339432529972)
    (rightMaximum := 345625740372465499945107099923406305361920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 46174) (rightBinding := 46175)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨18011⟩) (rightExpression := ⟨20928⟩)
    (transferEvent := 46176) (summaryTransferEvent := 46177)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult46173.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult45929.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult46178

namespace SemanticResult46183
def owner : Owner := ⟨.program ⟨257⟩, ⟨24149⟩⟩
def rawTerms : List Term := Proof.Events180.exact46183RawTerms
def summary : Bound := (.finite 1036877221117396499835321299770218916085812)
def resultEvent : Nat := 46183
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult46183.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult46178.owner)
    (rightOwner := SemanticResult45717.owner)
    (leftResult := 46178) (rightResult := 45717)
    (leftActual := SemanticResult46178.actual selector witness)
    (rightActual := SemanticResult45717.actual selector witness)
    (leftRaw := SemanticResult46178.rawTerms)
    (rightRaw := SemanticResult45717.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 691250426059631610003352154589745737891892)
    (rightMaximum := 345626795057764889831969145180473178193920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 46179) (rightBinding := 46180)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨20929⟩) (rightExpression := ⟨24148⟩)
    (transferEvent := 46181) (summaryTransferEvent := 46182)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult46178.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult45717.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult46183

namespace SemanticResult46188
def owner : Owner := ⟨.program ⟨257⟩, ⟨34169⟩⟩
def rawTerms : List Term := Proof.Events180.exact46188RawTerms
def summary : Bound := (.finite 1382506125545760169441014535464825839943732)
def resultEvent : Nat := 46188
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult46188.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult46183.owner)
    (rightOwner := SemanticResult45505.owner)
    (leftResult := 46183) (rightResult := 45505)
    (leftActual := SemanticResult46183.actual selector witness)
    (rightActual := SemanticResult45505.actual selector witness)
    (leftRaw := SemanticResult46183.rawTerms)
    (rightRaw := SemanticResult45505.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 1036877221117396499835321299770218916085812)
    (rightMaximum := 345628904428363669605693235694606923857920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 46184) (rightBinding := 46185)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨24149⟩) (rightExpression := ⟨34168⟩)
    (transferEvent := 46186) (summaryTransferEvent := 46187)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult46183.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult45505.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult46188

namespace SemanticResult46193
def owner : Owner := ⟨.program ⟨257⟩, ⟨53229⟩⟩
def rawTerms : List Term := Proof.Events180.exact46193RawTerms
def summary : Bound := (.finite 1728139248715321398594155952187700255129652)
def resultEvent : Nat := 46193
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult46193.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult46188.owner)
    (rightOwner := SemanticResult45293.owner)
    (leftResult := 46188) (rightResult := 45293)
    (leftActual := SemanticResult46188.actual selector witness)
    (rightActual := SemanticResult45293.actual selector witness)
    (leftRaw := SemanticResult46188.rawTerms)
    (rightRaw := SemanticResult45293.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 1382506125545760169441014535464825839943732)
    (rightMaximum := 345633123169561229153141416722874415185920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 46189) (rightBinding := 46190)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨34169⟩) (rightExpression := ⟨53228⟩)
    (transferEvent := 46191) (summaryTransferEvent := 46192)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult46188.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult45293.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult46193

namespace SemanticResult46198
def owner : Owner := ⟨.program ⟨257⟩, ⟨56209⟩⟩
def rawTerms : List Term := Proof.Events180.exact46198RawTerms
def summary : Bound := (.finite 2073774481255481407521021459424708415979572)
def resultEvent : Nat := 46198
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult46198.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult46193.owner)
    (rightOwner := SemanticResult45081.owner)
    (leftResult := 46193) (rightResult := 45081)
    (leftActual := SemanticResult46193.actual selector witness)
    (rightActual := SemanticResult45081.actual selector witness)
    (leftRaw := SemanticResult46193.rawTerms)
    (rightRaw := SemanticResult45081.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 1728139248715321398594155952187700255129652)
    (rightMaximum := 345635232540160008926865507237008160849920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 46194) (rightBinding := 46195)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨53229⟩) (rightExpression := ⟨56208⟩)
    (transferEvent := 46196) (summaryTransferEvent := 46197)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult46193.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult45081.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult46198

namespace SemanticResult46203
def owner : Owner := ⟨.program ⟨257⟩, ⟨59189⟩⟩
def rawTerms : List Term := Proof.Events180.exact46203RawTerms
def summary : Bound := (.finite 2419413932536838975995335147689984068157492)
def resultEvent : Nat := 46203
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult46203.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult46198.owner)
    (rightOwner := SemanticResult44869.owner)
    (leftResult := 46198) (rightResult := 44869)
    (leftActual := SemanticResult46198.actual selector witness)
    (rightActual := SemanticResult44869.actual selector witness)
    (leftRaw := SemanticResult46198.rawTerms)
    (rightRaw := SemanticResult44869.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 2073774481255481407521021459424708415979572)
    (rightMaximum := 345639451281357568474313688265275652177920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 46199) (rightBinding := 46200)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨56209⟩) (rightExpression := ⟨59188⟩)
    (transferEvent := 46201) (summaryTransferEvent := 46202)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult46198.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult44869.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult46203

namespace SemanticResult46208
def owner : Owner := ⟨.program ⟨257⟩, ⟨62169⟩⟩
def rawTerms : List Term := Proof.Events180.exact46208RawTerms
def summary : Bound := (.finite 2765055493188795324243372926469393465999412)
def resultEvent : Nat := 46208
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult46208.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult46203.owner)
    (rightOwner := SemanticResult44657.owner)
    (leftResult := 46203) (rightResult := 44657)
    (leftActual := SemanticResult46203.actual selector witness)
    (rightActual := SemanticResult44657.actual selector witness)
    (leftRaw := SemanticResult46203.rawTerms)
    (rightRaw := SemanticResult44657.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 2419413932536838975995335147689984068157492)
    (rightMaximum := 345641560651956348248037778779409397841920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 46204) (rightBinding := 46205)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨59189⟩) (rightExpression := ⟨62168⟩)
    (transferEvent := 46206) (summaryTransferEvent := 46207)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult46203.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult44657.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult46208

namespace SemanticResult46213
def owner : Owner := ⟨.program ⟨257⟩, ⟨65149⟩⟩
def rawTerms : List Term := Proof.Events180.exact46213RawTerms
def summary : Bound := (.finite 3110701272581949232038858886277070355169332)
def resultEvent : Nat := 46213
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult46213.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult46208.owner)
    (rightOwner := SemanticResult44445.owner)
    (leftResult := 46208) (rightResult := 44445)
    (leftActual := SemanticResult46208.actual selector witness)
    (rightActual := SemanticResult44445.actual selector witness)
    (leftRaw := SemanticResult46208.rawTerms)
    (rightRaw := SemanticResult44445.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 2765055493188795324243372926469393465999412)
    (rightMaximum := 345645779393153907795485959807676889169920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 46209) (rightBinding := 46210)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨62169⟩) (rightExpression := ⟨65148⟩)
    (transferEvent := 46211) (summaryTransferEvent := 46212)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult46208.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult44445.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult46213

namespace SemanticResult46218
def owner : Owner := ⟨.program ⟨257⟩, ⟨70878⟩⟩
def rawTerms : List Term := Proof.Events180.exact46218RawTerms
def summary : Bound := (.finite 3456353380086899479155517117627148481331252)
def resultEvent : Nat := 46218
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult46218.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult46213.owner)
    (rightOwner := SemanticResult44233.owner)
    (leftResult := 46213) (rightResult := 44233)
    (leftActual := SemanticResult46213.actual selector witness)
    (rightActual := SemanticResult44233.actual selector witness)
    (leftRaw := SemanticResult46213.rawTerms)
    (rightRaw := SemanticResult44233.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 3110701272581949232038858886277070355169332)
    (rightMaximum := 345652107504950247116658231350078126161920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 46214) (rightBinding := 46215)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨65149⟩) (rightExpression := ⟨70877⟩)
    (transferEvent := 46216) (summaryTransferEvent := 46217)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult46213.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult44233.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult46218

namespace SemanticResult46223
def owner : Owner := ⟨.program ⟨257⟩, ⟨70879⟩⟩
def rawTerms : List Term := Proof.Events180.exact46223RawTerms
def summary : Bound := (.finite 3802007596962448506045899439491360353157172)
def resultEvent : Nat := 46223
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult46223.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult46218.owner)
    (rightOwner := SemanticResult44021.owner)
    (leftResult := 46218) (rightResult := 44021)
    (leftActual := SemanticResult46218.actual selector witness)
    (rightActual := SemanticResult44021.actual selector witness)
    (leftRaw := SemanticResult46218.rawTerms)
    (rightRaw := SemanticResult44021.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 3456353380086899479155517117627148481331252)
    (rightMaximum := 345654216875549026890382321864211871825920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 46219) (rightBinding := 46220)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨70878⟩) (rightExpression := ⟨28512⟩)
    (transferEvent := 46221) (summaryTransferEvent := 46222)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult46218.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult44021.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult46223

namespace SemanticResult46228
def owner : Owner := ⟨.program ⟨257⟩, ⟨70880⟩⟩
def rawTerms : List Term := Proof.Events180.exact46228RawTerms
def summary : Bound := (.finite 4147668141949793872257454032897973461975092)
def resultEvent : Nat := 46228
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult46228.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult46223.owner)
    (rightOwner := SemanticResult43809.owner)
    (leftResult := 46223) (rightResult := 43809)
    (leftActual := SemanticResult46223.actual selector witness)
    (rightActual := SemanticResult43809.actual selector witness)
    (leftRaw := SemanticResult46223.rawTerms)
    (rightRaw := SemanticResult43809.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 3802007596962448506045899439491360353157172)
    (rightMaximum := 345660544987345366211554593406613108817920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 46224) (rightBinding := 46225)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨70879⟩) (rightExpression := ⟨31192⟩)
    (transferEvent := 46226) (summaryTransferEvent := 46227)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult46223.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult43809.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult46228

namespace SemanticResult46233
def owner : Owner := ⟨.program ⟨257⟩, ⟨70881⟩⟩
def rawTerms : List Term := Proof.Events180.exact46233RawTerms
def summary : Bound := (.finite 4493332905678336798016456807332854062121012)
def resultEvent : Nat := 46233
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult46233.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult46228.owner)
    (rightOwner := SemanticResult43597.owner)
    (leftResult := 46228) (rightResult := 43597)
    (leftActual := SemanticResult46228.actual selector witness)
    (rightActual := SemanticResult43597.actual selector witness)
    (leftRaw := SemanticResult46228.rawTerms)
    (rightRaw := SemanticResult43597.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 4147668141949793872257454032897973461975092)
    (rightMaximum := 345664763728542925759002774434880600145920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 46229) (rightBinding := 46230)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨70880⟩) (rightExpression := ⟨36852⟩)
    (transferEvent := 46231) (summaryTransferEvent := 46232)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult46228.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult43597.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult46233

namespace SemanticResult46238
def owner : Owner := ⟨.program ⟨257⟩, ⟨70882⟩⟩
def rawTerms : List Term := Proof.Events180.exact46238RawTerms
def summary : Bound := (.finite 4838999778777478503549183672281868407930932)
def resultEvent : Nat := 46238
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult46238.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult46233.owner)
    (rightOwner := SemanticResult43385.owner)
    (leftResult := 46233) (rightResult := 43385)
    (leftActual := SemanticResult46233.actual selector witness)
    (rightActual := SemanticResult43385.actual selector witness)
    (leftRaw := SemanticResult46233.rawTerms)
    (rightRaw := SemanticResult43385.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 4493332905678336798016456807332854062121012)
    (rightMaximum := 345666873099141705532726864949014345809920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 46234) (rightBinding := 46235)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨70881⟩) (rightExpression := ⟨39532⟩)
    (transferEvent := 46236) (summaryTransferEvent := 46237)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult46233.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult43385.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult46238

namespace SemanticResult46243
def owner : Owner := ⟨.program ⟨257⟩, ⟨70883⟩⟩
def rawTerms : List Term := Proof.Events180.exact46243RawTerms
def summary : Bound := (.finite 5184670870617817768629358718259150245068852)
def resultEvent : Nat := 46243
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult46243.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult46238.owner)
    (rightOwner := SemanticResult43173.owner)
    (leftResult := 46238) (rightResult := 43173)
    (leftActual := SemanticResult46238.actual selector witness)
    (rightActual := SemanticResult43173.actual selector witness)
    (leftRaw := SemanticResult46238.rawTerms)
    (rightRaw := SemanticResult43173.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 4838999778777478503549183672281868407930932)
    (rightMaximum := 345671091840339265080175045977281837137920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 46239) (rightBinding := 46240)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨70882⟩) (rightExpression := ⟨42212⟩)
    (transferEvent := 46241) (summaryTransferEvent := 46242)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult46238.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult43173.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult46243

namespace SemanticResult46248
def owner : Owner := ⟨.program ⟨257⟩, ⟨70884⟩⟩
def rawTerms : List Term := Proof.Events180.exact46248RawTerms
def summary : Bound := (.finite 5530348290569953373030706035778833319198772)
def resultEvent : Nat := 46248
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult46248.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult46243.owner)
    (rightOwner := SemanticResult42961.owner)
    (leftResult := 46243) (rightResult := 42961)
    (leftActual := SemanticResult46243.actual selector witness)
    (rightActual := SemanticResult42961.actual selector witness)
    (leftRaw := SemanticResult46243.rawTerms)
    (rightRaw := SemanticResult42961.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 5184670870617817768629358718259150245068852)
    (rightMaximum := 345677419952135604401347317519683074129920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 46244) (rightBinding := 46245)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨70883⟩) (rightExpression := ⟨44892⟩)
    (transferEvent := 46246) (summaryTransferEvent := 46247)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult46243.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult42961.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult46248

namespace SemanticResult46253
def owner : Owner := ⟨.program ⟨257⟩, ⟨70885⟩⟩
def rawTerms : List Term := Proof.Events180.exact46253RawTerms
def summary : Bound := (.finite 5876032038633885316753225624840917630320692)
def resultEvent : Nat := 46253
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult46253.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult46248.owner)
    (rightOwner := SemanticResult42749.owner)
    (leftResult := 46248) (rightResult := 42749)
    (leftActual := SemanticResult46248.actual selector witness)
    (rightActual := SemanticResult42749.actual selector witness)
    (leftRaw := SemanticResult46248.rawTerms)
    (rightRaw := SemanticResult42749.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 5530348290569953373030706035778833319198772)
    (rightMaximum := 345683748063931943722519589062084311121920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 46249) (rightBinding := 46250)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨70884⟩) (rightExpression := ⟨47572⟩)
    (transferEvent := 46251) (summaryTransferEvent := 46252)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult46248.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult42749.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult46253

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
