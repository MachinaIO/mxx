import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard635
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard633
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard634

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult89755
def owner : Owner := ⟨.program ⟨214⟩, ⟨15311⟩⟩
def rawTerms : List Term := Proof.Events350.exact89755RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 89755
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult89755.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 89752) (rightBinding := 89753)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨15265⟩) (rightExpression := ⟨15310⟩)
    (transferEvent := 89754)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult89751.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult89728.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult89755

namespace SemanticResult89759
def owner : Owner := ⟨.program ⟨214⟩, ⟨15367⟩⟩
def rawTerms : List Term := Proof.Events350.exact89759RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 89759
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult89759.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 89756) (rightBinding := 89757)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨15311⟩) (rightExpression := ⟨15366⟩)
    (transferEvent := 89758)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult89755.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult89705.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult89759

namespace SemanticResult89763
def owner : Owner := ⟨.program ⟨214⟩, ⟨17328⟩⟩
def rawTerms : List Term := Proof.Events350.exact89763RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 89763
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult89763.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 89760) (rightBinding := 89761)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨15367⟩) (rightExpression := ⟨17327⟩)
    (transferEvent := 89762)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult89759.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult89682.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult89763

namespace SemanticResult89767
def owner : Owner := ⟨.program ⟨214⟩, ⟨17329⟩⟩
def rawTerms : List Term := Proof.Events350.exact89767RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 89767
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult89767.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 89764) (rightBinding := 89765)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨17328⟩) (rightExpression := ⟨15629⟩)
    (transferEvent := 89766)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult89763.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult89659.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult89767

namespace SemanticResult89771
def owner : Owner := ⟨.program ⟨214⟩, ⟨17330⟩⟩
def rawTerms : List Term := Proof.Events350.exact89771RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 89771
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult89771.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 89768) (rightBinding := 89769)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨17329⟩) (rightExpression := ⟨15748⟩)
    (transferEvent := 89770)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult89767.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult89636.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult89771

namespace SemanticResult89775
def owner : Owner := ⟨.program ⟨214⟩, ⟨17331⟩⟩
def rawTerms : List Term := Proof.Events350.exact89775RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 89775
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult89775.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 89772) (rightBinding := 89773)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨17330⟩) (rightExpression := ⟨15867⟩)
    (transferEvent := 89774)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult89771.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult89613.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult89775

namespace SemanticResult89779
def owner : Owner := ⟨.program ⟨214⟩, ⟨17332⟩⟩
def rawTerms : List Term := Proof.Events350.exact89779RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 89779
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult89779.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 89776) (rightBinding := 89777)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨17331⟩) (rightExpression := ⟨15986⟩)
    (transferEvent := 89778)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult89775.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult89590.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult89779

namespace SemanticResult89783
def owner : Owner := ⟨.program ⟨214⟩, ⟨17333⟩⟩
def rawTerms : List Term := Proof.Events350.exact89783RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 89783
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult89783.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 89780) (rightBinding := 89781)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨17332⟩) (rightExpression := ⟨16105⟩)
    (transferEvent := 89782)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult89779.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult89567.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult89783

namespace SemanticResult89787
def owner : Owner := ⟨.program ⟨214⟩, ⟨18341⟩⟩
def rawTerms : List Term := Proof.Events350.exact89787RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 89787
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult89787.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 89784) (rightBinding := 89785)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨17333⟩) (rightExpression := ⟨18340⟩)
    (transferEvent := 89786)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult89783.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult89544.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult89787

namespace SemanticResult89791
def owner : Owner := ⟨.program ⟨214⟩, ⟨18342⟩⟩
def rawTerms : List Term := Proof.Events350.exact89791RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 89791
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult89791.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 89788) (rightBinding := 89789)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨18341⟩) (rightExpression := ⟨16308⟩)
    (transferEvent := 89790)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult89787.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult89521.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult89791

namespace SemanticResult89795
def owner : Owner := ⟨.program ⟨214⟩, ⟨18343⟩⟩
def rawTerms : List Term := Proof.Events350.exact89795RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 89795
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult89795.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 89792) (rightBinding := 89793)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨18342⟩) (rightExpression := ⟨17120⟩)
    (transferEvent := 89794)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult89791.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult89498.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult89795

namespace SemanticResult89799
def owner : Owner := ⟨.program ⟨214⟩, ⟨18344⟩⟩
def rawTerms : List Term := Proof.Events350.exact89799RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 89799
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult89799.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 89796) (rightBinding := 89797)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨18343⟩) (rightExpression := ⟨17904⟩)
    (transferEvent := 89798)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult89795.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult89475.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult89799

namespace SemanticResult89803
def owner : Owner := ⟨.program ⟨214⟩, ⟨18345⟩⟩
def rawTerms : List Term := Proof.Events350.exact89803RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 89803
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult89803.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 89800) (rightBinding := 89801)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨18344⟩) (rightExpression := ⟨18205⟩)
    (transferEvent := 89802)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult89799.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult89452.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult89803

namespace SemanticResult89807
def owner : Owner := ⟨.program ⟨214⟩, ⟨18346⟩⟩
def rawTerms : List Term := Proof.Events350.exact89807RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 89807
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult89807.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 89804) (rightBinding := 89805)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨18345⟩) (rightExpression := ⟨16679⟩)
    (transferEvent := 89806)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult89803.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult89429.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult89807

namespace SemanticResult89811
def owner : Owner := ⟨.program ⟨214⟩, ⟨18347⟩⟩
def rawTerms : List Term := Proof.Events350.exact89811RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 89811
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult89811.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 89808) (rightBinding := 89809)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨18346⟩) (rightExpression := ⟨16798⟩)
    (transferEvent := 89810)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult89807.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult89406.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult89811

namespace SemanticResult89815
def owner : Owner := ⟨.program ⟨214⟩, ⟨18348⟩⟩
def rawTerms : List Term := Proof.Events350.exact89815RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 89815
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult89815.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 89812) (rightBinding := 89813)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨18347⟩) (rightExpression := ⟨17085⟩)
    (transferEvent := 89814)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult89811.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult89383.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult89815

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
