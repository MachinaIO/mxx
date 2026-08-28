import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard1556
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard000
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard137
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1530
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1531
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1533
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1534
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1536
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1537
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1538
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1540
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1541
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1542
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1544
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1545
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1547
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1548
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1555

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult221698
def owner : Owner := ⟨.program ⟨257⟩, ⟨55930⟩⟩
def rawTerms : List Term := Proof.Events866.exact221698RawTerms
def summary : Bound := (.finite 2073774481255481407521021459424708415979572)
def resultEvent : Nat := 221698
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult221698.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult221693.owner)
    (rightOwner := SemanticResult220581.owner)
    (leftResult := 221693) (rightResult := 220581)
    (leftActual := SemanticResult221693.actual selector witness)
    (rightActual := SemanticResult220581.actual selector witness)
    (leftRaw := SemanticResult221693.rawTerms)
    (rightRaw := SemanticResult220581.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 1728139248715321398594155952187700255129652)
    (rightMaximum := 345635232540160008926865507237008160849920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 221694) (rightBinding := 221695)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨52950⟩) (rightExpression := ⟨55929⟩)
    (transferEvent := 221696) (summaryTransferEvent := 221697)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult221693.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult220581.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult221698

namespace SemanticResult221703
def owner : Owner := ⟨.program ⟨257⟩, ⟨58910⟩⟩
def rawTerms : List Term := Proof.Events866.exact221703RawTerms
def summary : Bound := (.finite 2419413932536838975995335147689984068157492)
def resultEvent : Nat := 221703
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult221703.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult221698.owner)
    (rightOwner := SemanticResult220369.owner)
    (leftResult := 221698) (rightResult := 220369)
    (leftActual := SemanticResult221698.actual selector witness)
    (rightActual := SemanticResult220369.actual selector witness)
    (leftRaw := SemanticResult221698.rawTerms)
    (rightRaw := SemanticResult220369.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 2073774481255481407521021459424708415979572)
    (rightMaximum := 345639451281357568474313688265275652177920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 221699) (rightBinding := 221700)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨55930⟩) (rightExpression := ⟨58909⟩)
    (transferEvent := 221701) (summaryTransferEvent := 221702)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult221698.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult220369.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult221703

namespace SemanticResult221708
def owner : Owner := ⟨.program ⟨257⟩, ⟨61890⟩⟩
def rawTerms : List Term := Proof.Events866.exact221708RawTerms
def summary : Bound := (.finite 2765055493188795324243372926469393465999412)
def resultEvent : Nat := 221708
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult221708.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult221703.owner)
    (rightOwner := SemanticResult220157.owner)
    (leftResult := 221703) (rightResult := 220157)
    (leftActual := SemanticResult221703.actual selector witness)
    (rightActual := SemanticResult220157.actual selector witness)
    (leftRaw := SemanticResult221703.rawTerms)
    (rightRaw := SemanticResult220157.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 2419413932536838975995335147689984068157492)
    (rightMaximum := 345641560651956348248037778779409397841920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 221704) (rightBinding := 221705)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨58910⟩) (rightExpression := ⟨61889⟩)
    (transferEvent := 221706) (summaryTransferEvent := 221707)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult221703.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult220157.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult221708

namespace SemanticResult221713
def owner : Owner := ⟨.program ⟨257⟩, ⟨64870⟩⟩
def rawTerms : List Term := Proof.Events866.exact221713RawTerms
def summary : Bound := (.finite 3110701272581949232038858886277070355169332)
def resultEvent : Nat := 221713
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult221713.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult221708.owner)
    (rightOwner := SemanticResult219945.owner)
    (leftResult := 221708) (rightResult := 219945)
    (leftActual := SemanticResult221708.actual selector witness)
    (rightActual := SemanticResult219945.actual selector witness)
    (leftRaw := SemanticResult221708.rawTerms)
    (rightRaw := SemanticResult219945.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 2765055493188795324243372926469393465999412)
    (rightMaximum := 345645779393153907795485959807676889169920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 221709) (rightBinding := 221710)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨61890⟩) (rightExpression := ⟨64869⟩)
    (transferEvent := 221711) (summaryTransferEvent := 221712)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult221708.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult219945.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult221713

namespace SemanticResult221718
def owner : Owner := ⟨.program ⟨257⟩, ⟨70167⟩⟩
def rawTerms : List Term := Proof.Events866.exact221718RawTerms
def summary : Bound := (.finite 3456353380086899479155517117627148481331252)
def resultEvent : Nat := 221718
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult221718.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult221713.owner)
    (rightOwner := SemanticResult219733.owner)
    (leftResult := 221713) (rightResult := 219733)
    (leftActual := SemanticResult221713.actual selector witness)
    (rightActual := SemanticResult219733.actual selector witness)
    (leftRaw := SemanticResult221713.rawTerms)
    (rightRaw := SemanticResult219733.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 3110701272581949232038858886277070355169332)
    (rightMaximum := 345652107504950247116658231350078126161920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 221714) (rightBinding := 221715)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨64870⟩) (rightExpression := ⟨70166⟩)
    (transferEvent := 221716) (summaryTransferEvent := 221717)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult221713.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult219733.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult221718

namespace SemanticResult221723
def owner : Owner := ⟨.program ⟨257⟩, ⟨70168⟩⟩
def rawTerms : List Term := Proof.Events866.exact221723RawTerms
def summary : Bound := (.finite 3802007596962448506045899439491360353157172)
def resultEvent : Nat := 221723
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult221723.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult221718.owner)
    (rightOwner := SemanticResult219521.owner)
    (leftResult := 221718) (rightResult := 219521)
    (leftActual := SemanticResult221718.actual selector witness)
    (rightActual := SemanticResult219521.actual selector witness)
    (leftRaw := SemanticResult221718.rawTerms)
    (rightRaw := SemanticResult219521.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 3456353380086899479155517117627148481331252)
    (rightMaximum := 345654216875549026890382321864211871825920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 221719) (rightBinding := 221720)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨70167⟩) (rightExpression := ⟨28287⟩)
    (transferEvent := 221721) (summaryTransferEvent := 221722)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult221718.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult219521.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult221723

namespace SemanticResult221728
def owner : Owner := ⟨.program ⟨257⟩, ⟨70169⟩⟩
def rawTerms : List Term := Proof.Events866.exact221728RawTerms
def summary : Bound := (.finite 4147668141949793872257454032897973461975092)
def resultEvent : Nat := 221728
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult221728.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult221723.owner)
    (rightOwner := SemanticResult219309.owner)
    (leftResult := 221723) (rightResult := 219309)
    (leftActual := SemanticResult221723.actual selector witness)
    (rightActual := SemanticResult219309.actual selector witness)
    (leftRaw := SemanticResult221723.rawTerms)
    (rightRaw := SemanticResult219309.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 3802007596962448506045899439491360353157172)
    (rightMaximum := 345660544987345366211554593406613108817920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 221724) (rightBinding := 221725)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨70168⟩) (rightExpression := ⟨30967⟩)
    (transferEvent := 221726) (summaryTransferEvent := 221727)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult221723.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult219309.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult221728

namespace SemanticResult221733
def owner : Owner := ⟨.program ⟨257⟩, ⟨70170⟩⟩
def rawTerms : List Term := Proof.Events866.exact221733RawTerms
def summary : Bound := (.finite 4493332905678336798016456807332854062121012)
def resultEvent : Nat := 221733
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult221733.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult221728.owner)
    (rightOwner := SemanticResult219097.owner)
    (leftResult := 221728) (rightResult := 219097)
    (leftActual := SemanticResult221728.actual selector witness)
    (rightActual := SemanticResult219097.actual selector witness)
    (leftRaw := SemanticResult221728.rawTerms)
    (rightRaw := SemanticResult219097.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 4147668141949793872257454032897973461975092)
    (rightMaximum := 345664763728542925759002774434880600145920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 221729) (rightBinding := 221730)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨70169⟩) (rightExpression := ⟨36627⟩)
    (transferEvent := 221731) (summaryTransferEvent := 221732)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult221728.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult219097.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult221733

namespace SemanticResult221738
def owner : Owner := ⟨.program ⟨257⟩, ⟨70171⟩⟩
def rawTerms : List Term := Proof.Events866.exact221738RawTerms
def summary : Bound := (.finite 4838999778777478503549183672281868407930932)
def resultEvent : Nat := 221738
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult221738.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult221733.owner)
    (rightOwner := SemanticResult218885.owner)
    (leftResult := 221733) (rightResult := 218885)
    (leftActual := SemanticResult221733.actual selector witness)
    (rightActual := SemanticResult218885.actual selector witness)
    (leftRaw := SemanticResult221733.rawTerms)
    (rightRaw := SemanticResult218885.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 4493332905678336798016456807332854062121012)
    (rightMaximum := 345666873099141705532726864949014345809920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 221734) (rightBinding := 221735)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨70170⟩) (rightExpression := ⟨39307⟩)
    (transferEvent := 221736) (summaryTransferEvent := 221737)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult221733.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult218885.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult221738

namespace SemanticResult221743
def owner : Owner := ⟨.program ⟨257⟩, ⟨70172⟩⟩
def rawTerms : List Term := Proof.Events866.exact221743RawTerms
def summary : Bound := (.finite 5184670870617817768629358718259150245068852)
def resultEvent : Nat := 221743
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult221743.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult221738.owner)
    (rightOwner := SemanticResult218673.owner)
    (leftResult := 221738) (rightResult := 218673)
    (leftActual := SemanticResult221738.actual selector witness)
    (rightActual := SemanticResult218673.actual selector witness)
    (leftRaw := SemanticResult221738.rawTerms)
    (rightRaw := SemanticResult218673.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 4838999778777478503549183672281868407930932)
    (rightMaximum := 345671091840339265080175045977281837137920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 221739) (rightBinding := 221740)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨70171⟩) (rightExpression := ⟨41987⟩)
    (transferEvent := 221741) (summaryTransferEvent := 221742)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult221738.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult218673.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult221743

namespace SemanticResult221748
def owner : Owner := ⟨.program ⟨257⟩, ⟨70173⟩⟩
def rawTerms : List Term := Proof.Events866.exact221748RawTerms
def summary : Bound := (.finite 5530348290569953373030706035778833319198772)
def resultEvent : Nat := 221748
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult221748.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult221743.owner)
    (rightOwner := SemanticResult218461.owner)
    (leftResult := 221743) (rightResult := 218461)
    (leftActual := SemanticResult221743.actual selector witness)
    (rightActual := SemanticResult218461.actual selector witness)
    (leftRaw := SemanticResult221743.rawTerms)
    (rightRaw := SemanticResult218461.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 5184670870617817768629358718259150245068852)
    (rightMaximum := 345677419952135604401347317519683074129920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 221744) (rightBinding := 221745)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨70172⟩) (rightExpression := ⟨44667⟩)
    (transferEvent := 221746) (summaryTransferEvent := 221747)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult221743.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult218461.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult221748

namespace SemanticResult221753
def owner : Owner := ⟨.program ⟨257⟩, ⟨70174⟩⟩
def rawTerms : List Term := Proof.Events866.exact221753RawTerms
def summary : Bound := (.finite 5876032038633885316753225624840917630320692)
def resultEvent : Nat := 221753
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult221753.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult221748.owner)
    (rightOwner := SemanticResult218249.owner)
    (leftResult := 221748) (rightResult := 218249)
    (leftActual := SemanticResult221748.actual selector witness)
    (rightActual := SemanticResult218249.actual selector witness)
    (leftRaw := SemanticResult221748.rawTerms)
    (rightRaw := SemanticResult218249.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 5530348290569953373030706035778833319198772)
    (rightMaximum := 345683748063931943722519589062084311121920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 221749) (rightBinding := 221750)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨70173⟩) (rightExpression := ⟨47347⟩)
    (transferEvent := 221751) (summaryTransferEvent := 221752)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult221748.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult218249.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult221753

namespace SemanticResult221758
def owner : Owner := ⟨.program ⟨257⟩, ⟨70175⟩⟩
def rawTerms : List Term := Proof.Events866.exact221758RawTerms
def summary : Bound := (.finite 6221717896068416040249469304417135687106612)
def resultEvent : Nat := 221758
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult221758.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult221753.owner)
    (rightOwner := SemanticResult218037.owner)
    (leftResult := 221753) (rightResult := 218037)
    (leftActual := SemanticResult221753.actual selector witness)
    (rightActual := SemanticResult218037.actual selector witness)
    (leftRaw := SemanticResult221753.rawTerms)
    (rightRaw := SemanticResult218037.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 5876032038633885316753225624840917630320692)
    (rightMaximum := 345685857434530723496243679576218056785920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 221754) (rightBinding := 221755)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨70174⟩) (rightExpression := ⟨50027⟩)
    (transferEvent := 221756) (summaryTransferEvent := 221757)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult221753.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult218037.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult221758

namespace SemanticResult221763
def owner : Owner := ⟨.program ⟨257⟩, ⟨71242⟩⟩
def rawTerms : List Term := Proof.Events866.exact221763RawTerms
def summary : Bound := (.finite 66805187227601152574551644069558752530002096506798132)
def resultEvent : Nat := 221763
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult221763.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult221758.owner)
    (rightOwner := SemanticResult217825.owner)
    (leftResult := 221758) (rightResult := 217825)
    (leftActual := SemanticResult221758.actual selector witness)
    (rightActual := SemanticResult217825.actual selector witness)
    (leftRaw := SemanticResult221758.rawTerms)
    (rightRaw := SemanticResult217825.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 6221717896068416040249469304417135687106612)
    (rightMaximum := 66805187221379434678483228029309283225584960819691520) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 221759) (rightBinding := 221760)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨70175⟩) (rightExpression := ⟨71240⟩)
    (transferEvent := 221761) (summaryTransferEvent := 221762)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult221758.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult217825.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult221763

namespace SemanticResult221765
def owner : Owner := ⟨.program ⟨257⟩, ⟨4⟩⟩
def rawTerms : List Term := Proof.Events866.exact221765RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 221765
def producerEvent : Nat := 221764
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult221765.actual selector witness
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
end SemanticResult221765

namespace SemanticResult221770
def owner : Owner := ⟨.program ⟨257⟩, ⟨7414⟩⟩
def rawTerms : List Term := Proof.Events866.exact221770RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 221770
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult221770.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge221769.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge221769.frameStart)
    (transferEvent := 221768) (owner := owner)
    (leftResult := 27) (rightResult := 16507)
    (working := LeftOperatorMerge221769.working)
    (reconstruction := LeftOperatorMerge221769.reconstruction)
    (leftReference := .predecessor 0 221766 .coefficient) (rightReference := .predecessor 1 221767 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult27.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult16507.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge221769.operationAgreement
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
end SemanticResult221770

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
