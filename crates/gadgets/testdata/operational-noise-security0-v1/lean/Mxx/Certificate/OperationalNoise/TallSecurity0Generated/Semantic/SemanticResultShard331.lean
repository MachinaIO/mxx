import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard331
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard268
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard271
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard275
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard279
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard282
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard286
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard290
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard294
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard297
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard301
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard305
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard308
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard312
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard316
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard319
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard323
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard330

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult44722
def owner : Owner := ⟨.program ⟨214⟩, ⟨26811⟩⟩
def rawTerms : List Term := Proof.Events174.exact44722RawTerms
def summary : Bound := (.finite 3875701141805795807232)
def resultEvent : Nat := 44722
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult44722.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult44717.owner)
    (rightOwner := SemanticResult43748.owner)
    (leftResult := 44717) (rightResult := 43748)
    (leftActual := SemanticResult44717.actual selector witness)
    (rightActual := SemanticResult43748.actual selector witness)
    (leftRaw := SemanticResult44717.rawTerms)
    (rightRaw := SemanticResult43748.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 2583789554981353578496)
    (rightMaximum := 1291911586824442228736) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 44718) (rightBinding := 44719)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨26594⟩) (rightExpression := ⟨26810⟩)
    (transferEvent := 44720) (summaryTransferEvent := 44721)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult44717.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult43748.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult44722

namespace SemanticResult44727
def owner : Owner := ⟨.program ⟨214⟩, ⟨27028⟩⟩
def rawTerms : List Term := Proof.Events174.exact44727RawTerms
def summary : Bound := (.finite 5167635141075258621952)
def resultEvent : Nat := 44727
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult44727.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult44722.owner)
    (rightOwner := SemanticResult43266.owner)
    (leftResult := 44722) (rightResult := 43266)
    (leftActual := SemanticResult44722.actual selector witness)
    (rightActual := SemanticResult43266.actual selector witness)
    (leftRaw := SemanticResult44722.rawTerms)
    (rightRaw := SemanticResult43266.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 3875701141805795807232)
    (rightMaximum := 1291933999269462814720) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 44723) (rightBinding := 44724)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨26811⟩) (rightExpression := ⟨27027⟩)
    (transferEvent := 44725) (summaryTransferEvent := 44726)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult44722.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult43266.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult44727

namespace SemanticResult44732
def owner : Owner := ⟨.program ⟨214⟩, ⟨27245⟩⟩
def rawTerms : List Term := Proof.Events174.exact44732RawTerms
def summary : Bound := (.finite 6459613965234762608640)
def resultEvent : Nat := 44732
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult44732.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult44727.owner)
    (rightOwner := SemanticResult42784.owner)
    (leftResult := 44727) (rightResult := 42784)
    (leftActual := SemanticResult44727.actual selector witness)
    (rightActual := SemanticResult42784.actual selector witness)
    (leftRaw := SemanticResult44727.rawTerms)
    (rightRaw := SemanticResult42784.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 5167635141075258621952)
    (rightMaximum := 1291978824159503986688) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 44728) (rightBinding := 44729)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨27028⟩) (rightExpression := ⟨27244⟩)
    (transferEvent := 44730) (summaryTransferEvent := 44731)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult44727.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult42784.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult44732

namespace SemanticResult44737
def owner : Owner := ⟨.program ⟨214⟩, ⟨27462⟩⟩
def rawTerms : List Term := Proof.Events174.exact44737RawTerms
def summary : Bound := (.finite 7751615201839287181312)
def resultEvent : Nat := 44737
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult44737.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult44732.owner)
    (rightOwner := SemanticResult42302.owner)
    (leftResult := 44732) (rightResult := 42302)
    (leftActual := SemanticResult44732.actual selector witness)
    (rightActual := SemanticResult42302.actual selector witness)
    (leftRaw := SemanticResult44732.rawTerms)
    (rightRaw := SemanticResult42302.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 6459613965234762608640)
    (rightMaximum := 1292001236604524572672) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 44733) (rightBinding := 44734)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨27245⟩) (rightExpression := ⟨27461⟩)
    (transferEvent := 44735) (summaryTransferEvent := 44736)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult44732.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult42302.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult44737

namespace SemanticResult44742
def owner : Owner := ⟨.program ⟨214⟩, ⟨27679⟩⟩
def rawTerms : List Term := Proof.Events174.exact44742RawTerms
def summary : Bound := (.finite 9043661263333852925952)
def resultEvent : Nat := 44742
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult44742.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult44737.owner)
    (rightOwner := SemanticResult41820.owner)
    (leftResult := 44737) (rightResult := 41820)
    (leftActual := SemanticResult44737.actual selector witness)
    (rightActual := SemanticResult41820.actual selector witness)
    (leftRaw := SemanticResult44737.rawTerms)
    (rightRaw := SemanticResult41820.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 7751615201839287181312)
    (rightMaximum := 1292046061494565744640) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 44738) (rightBinding := 44739)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨27462⟩) (rightExpression := ⟨27678⟩)
    (transferEvent := 44740) (summaryTransferEvent := 44741)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult44737.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult41820.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult44742

namespace SemanticResult44747
def owner : Owner := ⟨.program ⟨214⟩, ⟨27896⟩⟩
def rawTerms : List Term := Proof.Events174.exact44747RawTerms
def summary : Bound := (.finite 10335729737273439256576)
def resultEvent : Nat := 44747
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult44747.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult44742.owner)
    (rightOwner := SemanticResult41338.owner)
    (leftResult := 44742) (rightResult := 41338)
    (leftActual := SemanticResult44742.actual selector witness)
    (rightActual := SemanticResult41338.actual selector witness)
    (leftRaw := SemanticResult44742.rawTerms)
    (rightRaw := SemanticResult41338.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 9043661263333852925952)
    (rightMaximum := 1292068473939586330624) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 44743) (rightBinding := 44744)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨27679⟩) (rightExpression := ⟨27895⟩)
    (transferEvent := 44745) (summaryTransferEvent := 44746)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult44742.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult41338.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult44747

namespace SemanticResult44752
def owner : Owner := ⟨.program ⟨214⟩, ⟨28113⟩⟩
def rawTerms : List Term := Proof.Events174.exact44752RawTerms
def summary : Bound := (.finite 11627843036103066759168)
def resultEvent : Nat := 44752
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult44752.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult44747.owner)
    (rightOwner := SemanticResult40856.owner)
    (leftResult := 44747) (rightResult := 40856)
    (leftActual := SemanticResult44747.actual selector witness)
    (rightActual := SemanticResult40856.actual selector witness)
    (leftRaw := SemanticResult44747.rawTerms)
    (rightRaw := SemanticResult40856.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 10335729737273439256576)
    (rightMaximum := 1292113298829627502592) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 44748) (rightBinding := 44749)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨27896⟩) (rightExpression := ⟨28112⟩)
    (transferEvent := 44750) (summaryTransferEvent := 44751)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult44747.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult40856.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult44752

namespace SemanticResult44757
def owner : Owner := ⟨.program ⟨214⟩, ⟨28330⟩⟩
def rawTerms : List Term := Proof.Events174.exact44757RawTerms
def summary : Bound := (.finite 12920023572267756019712)
def resultEvent : Nat := 44757
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult44757.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult44752.owner)
    (rightOwner := SemanticResult40374.owner)
    (leftResult := 44752) (rightResult := 40374)
    (leftActual := SemanticResult44752.actual selector witness)
    (rightActual := SemanticResult40374.actual selector witness)
    (leftRaw := SemanticResult44752.rawTerms)
    (rightRaw := SemanticResult40374.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 11627843036103066759168)
    (rightMaximum := 1292180536164689260544) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 44753) (rightBinding := 44754)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨28113⟩) (rightExpression := ⟨28329⟩)
    (transferEvent := 44755) (summaryTransferEvent := 44756)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult44752.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult40374.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult44757

namespace SemanticResult44762
def owner : Owner := ⟨.program ⟨214⟩, ⟨28547⟩⟩
def rawTerms : List Term := Proof.Events174.exact44762RawTerms
def summary : Bound := (.finite 14212226520877465866240)
def resultEvent : Nat := 44762
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult44762.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult44757.owner)
    (rightOwner := SemanticResult39892.owner)
    (leftResult := 44757) (rightResult := 39892)
    (leftActual := SemanticResult44757.actual selector witness)
    (rightActual := SemanticResult39892.actual selector witness)
    (leftRaw := SemanticResult44757.rawTerms)
    (rightRaw := SemanticResult39892.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 12920023572267756019712)
    (rightMaximum := 1292202948609709846528) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 44758) (rightBinding := 44759)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨28330⟩) (rightExpression := ⟨28546⟩)
    (transferEvent := 44760) (summaryTransferEvent := 44761)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult44757.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult39892.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult44762

namespace SemanticResult44767
def owner : Owner := ⟨.program ⟨214⟩, ⟨28764⟩⟩
def rawTerms : List Term := Proof.Events174.exact44767RawTerms
def summary : Bound := (.finite 15504496706822237470720)
def resultEvent : Nat := 44767
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult44767.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult44762.owner)
    (rightOwner := SemanticResult39410.owner)
    (leftResult := 44762) (rightResult := 39410)
    (leftActual := SemanticResult44762.actual selector witness)
    (rightActual := SemanticResult39410.actual selector witness)
    (leftRaw := SemanticResult44762.rawTerms)
    (rightRaw := SemanticResult39410.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 14212226520877465866240)
    (rightMaximum := 1292270185944771604480) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 44763) (rightBinding := 44764)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨28547⟩) (rightExpression := ⟨28763⟩)
    (transferEvent := 44765) (summaryTransferEvent := 44766)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult44762.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult39410.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult44767

namespace SemanticResult44772
def owner : Owner := ⟨.program ⟨214⟩, ⟨28981⟩⟩
def rawTerms : List Term := Proof.Events174.exact44772RawTerms
def summary : Bound := (.finite 16796811717657050247168)
def resultEvent : Nat := 44772
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult44772.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult44767.owner)
    (rightOwner := SemanticResult38928.owner)
    (leftResult := 44767) (rightResult := 38928)
    (leftActual := SemanticResult44767.actual selector witness)
    (rightActual := SemanticResult38928.actual selector witness)
    (leftRaw := SemanticResult44767.rawTerms)
    (rightRaw := SemanticResult38928.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 15504496706822237470720)
    (rightMaximum := 1292315010834812776448) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 44768) (rightBinding := 44769)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨28764⟩) (rightExpression := ⟨28980⟩)
    (transferEvent := 44770) (summaryTransferEvent := 44771)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult44767.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult38928.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult44772

namespace SemanticResult44777
def owner : Owner := ⟨.program ⟨214⟩, ⟨29198⟩⟩
def rawTerms : List Term := Proof.Events174.exact44777RawTerms
def summary : Bound := (.finite 18089149140936883609600)
def resultEvent : Nat := 44777
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult44777.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult44772.owner)
    (rightOwner := SemanticResult38446.owner)
    (leftResult := 44772) (rightResult := 38446)
    (leftActual := SemanticResult44772.actual selector witness)
    (rightActual := SemanticResult38446.actual selector witness)
    (leftRaw := SemanticResult44772.rawTerms)
    (rightRaw := SemanticResult38446.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 16796811717657050247168)
    (rightMaximum := 1292337423279833362432) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 44773) (rightBinding := 44774)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨28981⟩) (rightExpression := ⟨29197⟩)
    (transferEvent := 44775) (summaryTransferEvent := 44776)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult44772.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult38446.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult44777

namespace SemanticResult44782
def owner : Owner := ⟨.program ⟨214⟩, ⟨29415⟩⟩
def rawTerms : List Term := Proof.Events174.exact44782RawTerms
def summary : Bound := (.finite 19381531389106758144000)
def resultEvent : Nat := 44782
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult44782.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult44777.owner)
    (rightOwner := SemanticResult37964.owner)
    (leftResult := 44777) (rightResult := 37964)
    (leftActual := SemanticResult44777.actual selector witness)
    (rightActual := SemanticResult37964.actual selector witness)
    (leftRaw := SemanticResult44777.rawTerms)
    (rightRaw := SemanticResult37964.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 18089149140936883609600)
    (rightMaximum := 1292382248169874534400) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 44778) (rightBinding := 44779)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨29198⟩) (rightExpression := ⟨29414⟩)
    (transferEvent := 44780) (summaryTransferEvent := 44781)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult44777.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult37964.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult44782

namespace SemanticResult44787
def owner : Owner := ⟨.program ⟨214⟩, ⟨29632⟩⟩
def rawTerms : List Term := Proof.Events174.exact44787RawTerms
def summary : Bound := (.finite 20673980874611694436352)
def resultEvent : Nat := 44787
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult44787.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult44782.owner)
    (rightOwner := SemanticResult37482.owner)
    (leftResult := 44782) (rightResult := 37482)
    (leftActual := SemanticResult44782.actual selector witness)
    (rightActual := SemanticResult37482.actual selector witness)
    (leftRaw := SemanticResult44782.rawTerms)
    (rightRaw := SemanticResult37482.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 19381531389106758144000)
    (rightMaximum := 1292449485504936292352) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 44783) (rightBinding := 44784)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨29415⟩) (rightExpression := ⟨29631⟩)
    (transferEvent := 44785) (summaryTransferEvent := 44786)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult44782.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult37482.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult44787

namespace SemanticResult44792
def owner : Owner := ⟨.program ⟨214⟩, ⟨29849⟩⟩
def rawTerms : List Term := Proof.Events174.exact44792RawTerms
def summary : Bound := (.finite 21966497597451692486656)
def resultEvent : Nat := 44792
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult44792.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult44787.owner)
    (rightOwner := SemanticResult37000.owner)
    (leftResult := 44787) (rightResult := 37000)
    (leftActual := SemanticResult44787.actual selector witness)
    (rightActual := SemanticResult37000.actual selector witness)
    (leftRaw := SemanticResult44787.rawTerms)
    (rightRaw := SemanticResult37000.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 20673980874611694436352)
    (rightMaximum := 1292516722839998050304) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 44788) (rightBinding := 44789)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨29632⟩) (rightExpression := ⟨29848⟩)
    (transferEvent := 44790) (summaryTransferEvent := 44791)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult44787.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult37000.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult44792

namespace SemanticResult44797
def owner : Owner := ⟨.program ⟨214⟩, ⟨30165⟩⟩
def rawTerms : List Term := Proof.Events174.exact44797RawTerms
def summary : Bound := (.finite 23259036732736711122944)
def resultEvent : Nat := 44797
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult44797.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult44792.owner)
    (rightOwner := SemanticResult36518.owner)
    (leftResult := 44792) (rightResult := 36518)
    (leftActual := SemanticResult44792.actual selector witness)
    (rightActual := SemanticResult36518.actual selector witness)
    (leftRaw := SemanticResult44792.rawTerms)
    (rightRaw := SemanticResult36518.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 21966497597451692486656)
    (rightMaximum := 1292539135285018636288) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 44793) (rightBinding := 44794)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨29849⟩) (rightExpression := ⟨30164⟩)
    (transferEvent := 44795) (summaryTransferEvent := 44796)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult44792.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult36518.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult44797

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
