import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard024
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard005
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard006
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard022
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard023

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult2972
def owner : Owner := ⟨.program ⟨214⟩, ⟨14892⟩⟩
def rawTerms : List Term := Proof.Events011.exact2972RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 2972
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult2972.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge2971.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge2971.frameStart)
    (transferEvent := 2970) (owner := owner)
    (leftResult := 2967) (rightResult := 713)
    (working := LeftOperatorMerge2971.working)
    (reconstruction := LeftOperatorMerge2971.reconstruction)
    (leftReference := .predecessor 0 2968 .coefficient) (rightReference := .predecessor 1 2969 .coefficient)
    (facts := ⟨true, true, none, some 1, some 1⟩)
    (leftScalar := false) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult2967.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult713.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge2971.operationAgreement
  · decide

theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply exactClaimAt_of_mergeClaim
    (mergeClaim selector selectorLower selectorUpper witness)
  · decide +kernel
  · rfl
end SemanticResult2972

namespace SemanticResult2976
def owner : Owner := ⟨.program ⟨214⟩, ⟨14893⟩⟩
def rawTerms : List Term := Proof.Events011.exact2976RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 2976
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult2976.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 2973) (rightBinding := 2974)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨6379⟩) (rightExpression := ⟨14892⟩)
    (transferEvent := 2975)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult728.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult2972.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult2976

namespace SemanticResult2980
def owner : Owner := ⟨.program ⟨214⟩, ⟨15054⟩⟩
def rawTerms : List Term := Proof.Events011.exact2980RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 2980
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult2980.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 2977) (rightBinding := 2978)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨14893⟩) (rightExpression := ⟨15053⟩)
    (transferEvent := 2979)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult2976.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult2964.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult2980

namespace SemanticResult2984
def owner : Owner := ⟨.program ⟨214⟩, ⟨15215⟩⟩
def rawTerms : List Term := Proof.Events011.exact2984RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 2984
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult2984.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 2981) (rightBinding := 2982)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨15054⟩) (rightExpression := ⟨15214⟩)
    (transferEvent := 2983)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult2980.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult2956.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult2984

namespace SemanticResult2988
def owner : Owner := ⟨.program ⟨214⟩, ⟨15523⟩⟩
def rawTerms : List Term := Proof.Events011.exact2988RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 2988
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult2988.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 2985) (rightBinding := 2986)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨15215⟩) (rightExpression := ⟨15522⟩)
    (transferEvent := 2987)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult2984.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult2948.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult2988

namespace SemanticResult2992
def owner : Owner := ⟨.program ⟨214⟩, ⟨17824⟩⟩
def rawTerms : List Term := Proof.Events011.exact2992RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 2992
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult2992.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 2989) (rightBinding := 2990)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨15523⟩) (rightExpression := ⟨17823⟩)
    (transferEvent := 2991)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult2988.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult2940.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult2992

namespace SemanticResult2996
def owner : Owner := ⟨.program ⟨214⟩, ⟨17825⟩⟩
def rawTerms : List Term := Proof.Events011.exact2996RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 2996
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult2996.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 2993) (rightBinding := 2994)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨17824⟩) (rightExpression := ⟨17443⟩)
    (transferEvent := 2995)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult2992.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult2932.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult2996

namespace SemanticResult3000
def owner : Owner := ⟨.program ⟨214⟩, ⟨17826⟩⟩
def rawTerms : List Term := Proof.Events011.exact3000RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 3000
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult3000.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 2997) (rightBinding := 2998)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨17825⟩) (rightExpression := ⟨17226⟩)
    (transferEvent := 2999)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult2996.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult2924.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult3000

namespace SemanticResult3004
def owner : Owner := ⟨.program ⟨214⟩, ⟨17827⟩⟩
def rawTerms : List Term := Proof.Events011.exact3004RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 3004
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult3004.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 3001) (rightBinding := 3002)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨17826⟩) (rightExpression := ⟨17170⟩)
    (transferEvent := 3003)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult3000.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult2916.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult3004

namespace SemanticResult3008
def owner : Owner := ⟨.program ⟨214⟩, ⟨18044⟩⟩
def rawTerms : List Term := Proof.Events011.exact3008RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 3008
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult3008.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 3005) (rightBinding := 3006)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨17827⟩) (rightExpression := ⟨18043⟩)
    (transferEvent := 3007)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult3004.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult2908.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult3008

namespace SemanticResult3012
def owner : Owner := ⟨.program ⟨214⟩, ⟨18045⟩⟩
def rawTerms : List Term := Proof.Events011.exact3012RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 3012
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult3012.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 3009) (rightBinding := 3010)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨18044⟩) (rightExpression := ⟨17667⟩)
    (transferEvent := 3011)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult3008.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult2900.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult3012

namespace SemanticResult3016
def owner : Owner := ⟨.program ⟨214⟩, ⟨18046⟩⟩
def rawTerms : List Term := Proof.Events011.exact3016RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 3016
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult3016.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 3013) (rightBinding := 3014)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨18045⟩) (rightExpression := ⟨17611⟩)
    (transferEvent := 3015)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult3012.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult2892.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult3016

namespace SemanticResult3020
def owner : Owner := ⟨.program ⟨214⟩, ⟨18850⟩⟩
def rawTerms : List Term := Proof.Events011.exact3020RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 3020
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult3020.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 3017) (rightBinding := 3018)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨18046⟩) (rightExpression := ⟨18849⟩)
    (transferEvent := 3019)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult3016.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult2884.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult3020

namespace SemanticResult3024
def owner : Owner := ⟨.program ⟨214⟩, ⟨18851⟩⟩
def rawTerms : List Term := Proof.Events011.exact3024RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 3024
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult3024.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 3021) (rightBinding := 3022)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨18850⟩) (rightExpression := ⟨17555⟩)
    (transferEvent := 3023)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult3020.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult2876.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult3024

namespace SemanticResult3028
def owner : Owner := ⟨.program ⟨214⟩, ⟨18852⟩⟩
def rawTerms : List Term := Proof.Events011.exact3028RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 3028
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult3028.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 3025) (rightBinding := 3026)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨18851⟩) (rightExpression := ⟨17954⟩)
    (transferEvent := 3027)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult3024.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult2868.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult3028

namespace SemanticResult3032
def owner : Owner := ⟨.program ⟨214⟩, ⟨18853⟩⟩
def rawTerms : List Term := Proof.Events011.exact3032RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 3032
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult3032.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 3029) (rightBinding := 3030)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨18852⟩) (rightExpression := ⟨17723⟩)
    (transferEvent := 3031)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult3028.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult2860.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult3032

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
