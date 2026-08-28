import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard262
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard241
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard242
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard243
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard245
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard246
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard247
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard249
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard250
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard252
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard253
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard254
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard256
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard257
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard258
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard260
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard261

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult35565
def owner : Owner := ⟨.program ⟨214⟩, ⟨26392⟩⟩
def rawTerms : List Term := Proof.Events138.exact35565RawTerms
def summary : Bound := (.finite 4741253940199267499646124084)
def resultEvent : Nat := 35565
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult35565.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult35560.owner)
    (rightOwner := SemanticResult35533.owner)
    (leftResult := 35560) (rightResult := 35533)
    (leftActual := SemanticResult35560.actual selector witness)
    (rightActual := SemanticResult35533.actual selector witness)
    (leftRaw := SemanticResult35560.rawTerms)
    (rightRaw := SemanticResult35533.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 52)
    (rightMaximum := 4741253940199267499646124032) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 35561) (rightBinding := 35562)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7811⟩) (rightExpression := ⟨26391⟩)
    (transferEvent := 35563) (summaryTransferEvent := 35564)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult35560.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult35533.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult35565

namespace SemanticResult35570
def owner : Owner := ⟨.program ⟨214⟩, ⟨26601⟩⟩
def rawTerms : List Term := Proof.Events138.exact35570RawTerms
def summary : Bound := (.finite 9482549007414447334737575988)
def resultEvent : Nat := 35570
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult35570.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult35565.owner)
    (rightOwner := SemanticResult35321.owner)
    (leftResult := 35565) (rightResult := 35321)
    (leftActual := SemanticResult35565.actual selector witness)
    (rightActual := SemanticResult35321.actual selector witness)
    (leftRaw := SemanticResult35565.rawTerms)
    (rightRaw := SemanticResult35321.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 4741253940199267499646124084)
    (rightMaximum := 4741295067215179835091451904) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 35566) (rightBinding := 35567)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨26392⟩) (rightExpression := ⟨26600⟩)
    (transferEvent := 35568) (summaryTransferEvent := 35569)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult35565.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult35321.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult35570

namespace SemanticResult35575
def owner : Owner := ⟨.program ⟨214⟩, ⟨26818⟩⟩
def rawTerms : List Term := Proof.Events138.exact35575RawTerms
def summary : Bound := (.finite 14223885201645539505274355764)
def resultEvent : Nat := 35575
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult35575.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult35570.owner)
    (rightOwner := SemanticResult35109.owner)
    (leftResult := 35570) (rightResult := 35109)
    (leftActual := SemanticResult35570.actual selector witness)
    (rightActual := SemanticResult35109.actual selector witness)
    (leftRaw := SemanticResult35570.rawTerms)
    (rightRaw := SemanticResult35109.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 9482549007414447334737575988)
    (rightMaximum := 4741336194231092170536779776) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 35571) (rightBinding := 35572)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨26601⟩) (rightExpression := ⟨26817⟩)
    (transferEvent := 35573) (summaryTransferEvent := 35574)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult35570.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult35109.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult35575

namespace SemanticResult35580
def owner : Owner := ⟨.program ⟨214⟩, ⟨27035⟩⟩
def rawTerms : List Term := Proof.Events138.exact35580RawTerms
def summary : Bound := (.finite 18965303649908456346701791284)
def resultEvent : Nat := 35580
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult35580.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult35575.owner)
    (rightOwner := SemanticResult34897.owner)
    (leftResult := 35575) (rightResult := 34897)
    (leftActual := SemanticResult35575.actual selector witness)
    (rightActual := SemanticResult34897.actual selector witness)
    (leftRaw := SemanticResult35575.rawTerms)
    (rightRaw := SemanticResult34897.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 14223885201645539505274355764)
    (rightMaximum := 4741418448262916841427435520) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 35576) (rightBinding := 35577)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨26818⟩) (rightExpression := ⟨27034⟩)
    (transferEvent := 35578) (summaryTransferEvent := 35579)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult35575.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult34897.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult35580

namespace SemanticResult35585
def owner : Owner := ⟨.program ⟨214⟩, ⟨27252⟩⟩
def rawTerms : List Term := Proof.Events139.exact35585RawTerms
def summary : Bound := (.finite 23706886606235022529910538292)
def resultEvent : Nat := 35585
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult35585.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult35580.owner)
    (rightOwner := SemanticResult34685.owner)
    (leftResult := 35580) (rightResult := 34685)
    (leftActual := SemanticResult35580.actual selector witness)
    (rightActual := SemanticResult34685.actual selector witness)
    (leftRaw := SemanticResult35580.rawTerms)
    (rightRaw := SemanticResult34685.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 18965303649908456346701791284)
    (rightMaximum := 4741582956326566183208747008) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 35581) (rightBinding := 35582)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨27035⟩) (rightExpression := ⟨27251⟩)
    (transferEvent := 35583) (summaryTransferEvent := 35584)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult35580.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult34685.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult35585

namespace SemanticResult35590
def owner : Owner := ⟨.program ⟨214⟩, ⟨27469⟩⟩
def rawTerms : List Term := Proof.Events139.exact35590RawTerms
def summary : Bound := (.finite 28448551816593413384009941044)
def resultEvent : Nat := 35590
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult35590.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult35585.owner)
    (rightOwner := SemanticResult34473.owner)
    (leftResult := 35585) (rightResult := 34473)
    (leftActual := SemanticResult35585.actual selector witness)
    (rightActual := SemanticResult34473.actual selector witness)
    (leftRaw := SemanticResult35585.rawTerms)
    (rightRaw := SemanticResult34473.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 23706886606235022529910538292)
    (rightMaximum := 4741665210358390854099402752) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 35586) (rightBinding := 35587)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨27252⟩) (rightExpression := ⟨27468⟩)
    (transferEvent := 35588) (summaryTransferEvent := 35589)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult35585.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult34473.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult35590

namespace SemanticResult35595
def owner : Owner := ⟨.program ⟨214⟩, ⟨27686⟩⟩
def rawTerms : List Term := Proof.Events139.exact35595RawTerms
def summary : Bound := (.finite 33190381535015453579890655284)
def resultEvent : Nat := 35595
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult35595.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult35590.owner)
    (rightOwner := SemanticResult34261.owner)
    (leftResult := 35590) (rightResult := 34261)
    (leftActual := SemanticResult35590.actual selector witness)
    (rightActual := SemanticResult34261.actual selector witness)
    (leftRaw := SemanticResult35590.rawTerms)
    (rightRaw := SemanticResult34261.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 28448551816593413384009941044)
    (rightMaximum := 4741829718422040195880714240) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 35591) (rightBinding := 35592)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨27469⟩) (rightExpression := ⟨27685⟩)
    (transferEvent := 35593) (summaryTransferEvent := 35594)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult35590.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult34261.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult35595

namespace SemanticResult35600
def owner : Owner := ⟨.program ⟨214⟩, ⟨27903⟩⟩
def rawTerms : List Term := Proof.Events139.exact35600RawTerms
def summary : Bound := (.finite 37932293507469318446662025268)
def resultEvent : Nat := 35600
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult35600.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult35595.owner)
    (rightOwner := SemanticResult34049.owner)
    (leftResult := 35595) (rightResult := 34049)
    (leftActual := SemanticResult35595.actual selector witness)
    (rightActual := SemanticResult34049.actual selector witness)
    (leftRaw := SemanticResult35595.rawTerms)
    (rightRaw := SemanticResult34049.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 33190381535015453579890655284)
    (rightMaximum := 4741911972453864866771369984) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 35596) (rightBinding := 35597)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨27686⟩) (rightExpression := ⟨27902⟩)
    (transferEvent := 35598) (summaryTransferEvent := 35599)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult35595.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult34049.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult35600

namespace SemanticResult35605
def owner : Owner := ⟨.program ⟨214⟩, ⟨28120⟩⟩
def rawTerms : List Term := Proof.Events139.exact35605RawTerms
def summary : Bound := (.finite 42674369987986832655214706740)
def resultEvent : Nat := 35605
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult35605.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult35600.owner)
    (rightOwner := SemanticResult33837.owner)
    (leftResult := 35600) (rightResult := 33837)
    (leftActual := SemanticResult35600.actual selector witness)
    (rightActual := SemanticResult33837.actual selector witness)
    (leftRaw := SemanticResult35600.rawTerms)
    (rightRaw := SemanticResult33837.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 37932293507469318446662025268)
    (rightMaximum := 4742076480517514208552681472) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 35601) (rightBinding := 35602)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨27903⟩) (rightExpression := ⟨28119⟩)
    (transferEvent := 35603) (summaryTransferEvent := 35604)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult35600.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult33837.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult35605

namespace SemanticResult35610
def owner : Owner := ⟨.program ⟨214⟩, ⟨28337⟩⟩
def rawTerms : List Term := Proof.Events139.exact35610RawTerms
def summary : Bound := (.finite 47416693230599820876439355444)
def resultEvent : Nat := 35610
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult35610.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult35605.owner)
    (rightOwner := SemanticResult33625.owner)
    (leftResult := 35605) (rightResult := 33625)
    (leftActual := SemanticResult35605.actual selector witness)
    (rightActual := SemanticResult33625.actual selector witness)
    (leftRaw := SemanticResult35605.rawTerms)
    (rightRaw := SemanticResult33625.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 42674369987986832655214706740)
    (rightMaximum := 4742323242612988221224648704) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 35606) (rightBinding := 35607)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨28120⟩) (rightExpression := ⟨28336⟩)
    (transferEvent := 35608) (summaryTransferEvent := 35609)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult35605.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult33625.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult35610

namespace SemanticResult35615
def owner : Owner := ⟨.program ⟨214⟩, ⟨28554⟩⟩
def rawTerms : List Term := Proof.Events139.exact35615RawTerms
def summary : Bound := (.finite 52159098727244633768554659892)
def resultEvent : Nat := 35615
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult35615.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult35610.owner)
    (rightOwner := SemanticResult33413.owner)
    (leftResult := 35610) (rightResult := 33413)
    (leftActual := SemanticResult35610.actual selector witness)
    (rightActual := SemanticResult33413.actual selector witness)
    (leftRaw := SemanticResult35610.rawTerms)
    (rightRaw := SemanticResult33413.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 47416693230599820876439355444)
    (rightMaximum := 4742405496644812892115304448) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 35611) (rightBinding := 35612)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨28337⟩) (rightExpression := ⟨28553⟩)
    (transferEvent := 35613) (summaryTransferEvent := 35614)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult35610.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult33413.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult35615

namespace SemanticResult35620
def owner : Owner := ⟨.program ⟨214⟩, ⟨28771⟩⟩
def rawTerms : List Term := Proof.Events139.exact35620RawTerms
def summary : Bound := (.finite 56901750985984920673341931572)
def resultEvent : Nat := 35620
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult35620.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult35615.owner)
    (rightOwner := SemanticResult33201.owner)
    (leftResult := 35615) (rightResult := 33201)
    (leftActual := SemanticResult35615.actual selector witness)
    (rightActual := SemanticResult33201.actual selector witness)
    (leftRaw := SemanticResult35615.rawTerms)
    (rightRaw := SemanticResult33201.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 52159098727244633768554659892)
    (rightMaximum := 4742652258740286904787271680) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 35616) (rightBinding := 35617)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨28554⟩) (rightExpression := ⟨28770⟩)
    (transferEvent := 35618) (summaryTransferEvent := 35619)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult35615.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult33201.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult35620

namespace SemanticResult35625
def owner : Owner := ⟨.program ⟨214⟩, ⟨28988⟩⟩
def rawTerms : List Term := Proof.Events139.exact35625RawTerms
def summary : Bound := (.finite 61644567752788856919910514740)
def resultEvent : Nat := 35625
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult35625.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult35620.owner)
    (rightOwner := SemanticResult32989.owner)
    (leftResult := 35620) (rightResult := 32989)
    (leftActual := SemanticResult35620.actual selector witness)
    (rightActual := SemanticResult32989.actual selector witness)
    (leftRaw := SemanticResult35620.rawTerms)
    (rightRaw := SemanticResult32989.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 56901750985984920673341931572)
    (rightMaximum := 4742816766803936246568583168) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 35621) (rightBinding := 35622)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨28771⟩) (rightExpression := ⟨28987⟩)
    (transferEvent := 35623) (summaryTransferEvent := 35624)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult35620.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult32989.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult35625

namespace SemanticResult35630
def owner : Owner := ⟨.program ⟨214⟩, ⟨29205⟩⟩
def rawTerms : List Term := Proof.Events139.exact35630RawTerms
def summary : Bound := (.finite 66387466773624617837369753652)
def resultEvent : Nat := 35630
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult35630.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult35625.owner)
    (rightOwner := SemanticResult32777.owner)
    (leftResult := 35625) (rightResult := 32777)
    (leftActual := SemanticResult35625.actual selector witness)
    (rightActual := SemanticResult32777.actual selector witness)
    (leftRaw := SemanticResult35625.rawTerms)
    (rightRaw := SemanticResult32777.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 61644567752788856919910514740)
    (rightMaximum := 4742899020835760917459238912) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 35626) (rightBinding := 35627)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨28988⟩) (rightExpression := ⟨29204⟩)
    (transferEvent := 35628) (summaryTransferEvent := 35629)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult35625.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult32777.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult35630

namespace SemanticResult35635
def owner : Owner := ⟨.program ⟨214⟩, ⟨29422⟩⟩
def rawTerms : List Term := Proof.Events139.exact35635RawTerms
def summary : Bound := (.finite 71130530302524028096610304052)
def resultEvent : Nat := 35635
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult35635.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult35630.owner)
    (rightOwner := SemanticResult32565.owner)
    (leftResult := 35630) (rightResult := 32565)
    (leftActual := SemanticResult35630.actual selector witness)
    (rightActual := SemanticResult32565.actual selector witness)
    (leftRaw := SemanticResult35630.rawTerms)
    (rightRaw := SemanticResult32565.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 66387466773624617837369753652)
    (rightMaximum := 4743063528899410259240550400) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 35631) (rightBinding := 35632)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨29205⟩) (rightExpression := ⟨29421⟩)
    (transferEvent := 35633) (summaryTransferEvent := 35634)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult35630.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult32565.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult35635

namespace SemanticResult35640
def owner : Owner := ⟨.program ⟨214⟩, ⟨29639⟩⟩
def rawTerms : List Term := Proof.Events139.exact35640RawTerms
def summary : Bound := (.finite 75873840593518912368522821684)
def resultEvent : Nat := 35640
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult35640.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult35635.owner)
    (rightOwner := SemanticResult32353.owner)
    (leftResult := 35635) (rightResult := 32353)
    (leftActual := SemanticResult35635.actual selector witness)
    (rightActual := SemanticResult32353.actual selector witness)
    (leftRaw := SemanticResult35635.rawTerms)
    (rightRaw := SemanticResult32353.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 71130530302524028096610304052)
    (rightMaximum := 4743310290994884271912517632) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 35636) (rightBinding := 35637)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨29422⟩) (rightExpression := ⟨29638⟩)
    (transferEvent := 35638) (summaryTransferEvent := 35639)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult35635.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult32353.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult35640

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
