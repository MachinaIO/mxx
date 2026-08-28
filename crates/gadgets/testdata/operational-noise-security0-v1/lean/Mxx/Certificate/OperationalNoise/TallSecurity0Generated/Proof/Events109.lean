import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events109

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event27904 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7843⟩⟩) (.authority (.operator))

def exact27905RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨7843⟩⟩]⟩, (1)⟩]

theorem exact27905RawTermsValid :
    exact27905RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event27905 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7843⟩⟩) exact27905RawTerms (.finite 8192) 27904 .exactZero (none)

def event27906 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7844⟩⟩) 0 ⟨7843⟩ 27905

def event27907 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7844⟩⟩) 1 ⟨2348⟩ 27896

def event27908 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7844⟩⟩) (.scale (.predecessor 0 27906 .coefficient) (.value (.predecessor 1 27907 .coefficient)))

def exact27909RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨7843⟩⟩]⟩, (1)⟩]

theorem exact27909RawTermsValid :
    exact27909RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event27909 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7844⟩⟩) exact27909RawTerms (.finite 8192) 27908 .exactZero (none)

def event27910 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6793⟩⟩) 0 ⟨6757⟩ 27899

def event27911 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6793⟩⟩) (.identity (.predecessor 0 27910 .coefficient))

def exact27912RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6793⟩⟩]⟩, (1)⟩]

theorem exact27912RawTermsValid :
    exact27912RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event27912 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6793⟩⟩) exact27912RawTerms .large 27911 .exactZero (none)

def event27913 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7845⟩⟩) 0 ⟨6793⟩ 27912

def event27914 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7845⟩⟩) 1 ⟨7844⟩ 27909

def event27915 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7845⟩⟩) (.product (.predecessor 0 27913 .coefficient) (.predecessor 1 27914 .coefficient) (⟨false, false, none, none, none⟩))

def event27916 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7845⟩⟩, .operator (⟨27912, 0⟩, ⟨27909, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6793⟩⟩, ⟨.program ⟨214⟩, ⟨7843⟩⟩]⟩, (1)⟩)

def exact27917RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6793⟩⟩, ⟨.program ⟨214⟩, ⟨7843⟩⟩]⟩, (1)⟩]

theorem exact27917RawTermsValid :
    exact27917RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event27917 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7845⟩⟩) exact27917RawTerms .large 27915 .exactZero (none)

def event27918 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13678⟩⟩) 0 ⟨7845⟩ 27917

def event27919 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13678⟩⟩) 1 ⟨13677⟩ 27894

def event27920 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13678⟩⟩) (.sum [.predecessor 0 27918 .coefficient, .predecessor 1 27919 .coefficient])

def exact27921RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6793⟩⟩, ⟨.program ⟨214⟩, ⟨7843⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨11229⟩⟩, ⟨.program ⟨214⟩, ⟨13583⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact27921RawTermsValid :
    exact27921RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event27921 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13678⟩⟩) exact27921RawTerms .large 27920 .exactZero (none)

def event27922 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25853⟩⟩) 0 ⟨13678⟩ 27921

def event27923 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25853⟩⟩) 1 ⟨25850⟩ 27878

def event27924 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25853⟩⟩) (.product (.predecessor 0 27922 .coefficient) (.predecessor 1 27923 .coefficient) (⟨false, false, none, none, none⟩))

def event27925 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25853⟩⟩, .operator (⟨27921, 0⟩, ⟨27878, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6793⟩⟩, ⟨.program ⟨214⟩, ⟨7843⟩⟩, ⟨.program ⟨214⟩, ⟨25850⟩⟩]⟩, (1)⟩)

def event27926 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25853⟩⟩, .operator (⟨27921, 1⟩, ⟨27878, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨11229⟩⟩, ⟨.program ⟨214⟩, ⟨13583⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25850⟩⟩]⟩, (-1)⟩)

def event27927 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨25853⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨11229⟩⟩, ⟨.program ⟨214⟩, ⟨13583⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25850⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨25850⟩⟩) ⟨23464⟩ 27875)

def event27928 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25853⟩⟩, .relation 27927 0, ⟨[⟨.program ⟨214⟩, ⟨11229⟩⟩, ⟨.program ⟨214⟩, ⟨13583⟩⟩], [⟨.program ⟨214⟩, ⟨23464⟩⟩]⟩, (-1)⟩)

def exact27929RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6793⟩⟩, ⟨.program ⟨214⟩, ⟨7843⟩⟩, ⟨.program ⟨214⟩, ⟨25850⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨11229⟩⟩, ⟨.program ⟨214⟩, ⟨13583⟩⟩], [⟨.program ⟨214⟩, ⟨23464⟩⟩]⟩, (-1)⟩]

theorem exact27929RawTermsValid :
    exact27929RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event27929 : Event := .resultExact (⟨.program ⟨214⟩, ⟨25853⟩⟩) exact27929RawTerms .large 27924 .exactZero (none)

def event27930 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15595⟩⟩) 0 ⟨13585⟩ 27867

def event27931 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15595⟩⟩) (.authority (.programFamilyFact))

def exact27932RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15595⟩⟩], []⟩, (1)⟩]

theorem exact27932RawTermsValid :
    exact27932RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event27932 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15595⟩⟩) exact27932RawTerms (.finite 10) 27931 .exactZero (none)

def event27933 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15597⟩⟩) 0 ⟨6544⟩ 27889

def event27934 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15597⟩⟩) 1 ⟨15595⟩ 27932

def event27935 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15597⟩⟩) (.product (.predecessor 0 27933 .coefficient) (.predecessor 1 27934 .coefficient) (⟨false, true, none, none, some 1⟩))

def event27936 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨15597⟩⟩, .operator (⟨27889, 0⟩, ⟨27932, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨15595⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact27937RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15595⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact27937RawTermsValid :
    exact27937RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event27937 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15597⟩⟩) exact27937RawTerms .large 27935 .exactZero (none)

def event27938 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6694⟩⟩) 0 ⟨6689⟩ 27871

def event27939 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6694⟩⟩) (.authority (.operator))

def exact27940RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6694⟩⟩]⟩, (1)⟩]

theorem exact27940RawTermsValid :
    exact27940RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event27940 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6694⟩⟩) exact27940RawTerms .large 27939 .exactZero (none)

def event27941 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15598⟩⟩) 0 ⟨6694⟩ 27940

def event27942 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15598⟩⟩) 1 ⟨15597⟩ 27937

def event27943 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15598⟩⟩) (.sum [.predecessor 0 27941 .coefficient, .predecessor 1 27942 .coefficient])

def exact27944RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6694⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15595⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact27944RawTermsValid :
    exact27944RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event27944 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15598⟩⟩) exact27944RawTerms .large 27943 .exactZero (none)

def event27945 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25854⟩⟩) 0 ⟨15598⟩ 27944

def event27946 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25854⟩⟩) 1 ⟨25853⟩ 27929

def event27947 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25854⟩⟩) (.sum [.predecessor 0 27945 .coefficient, .predecessor 1 27946 .coefficient])

def exact27948RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6694⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6793⟩⟩, ⟨.program ⟨214⟩, ⟨7843⟩⟩, ⟨.program ⟨214⟩, ⟨25850⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨11229⟩⟩, ⟨.program ⟨214⟩, ⟨13583⟩⟩], [⟨.program ⟨214⟩, ⟨23464⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15595⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact27948RawTermsValid :
    exact27948RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event27948 : Event := .resultExact (⟨.program ⟨214⟩, ⟨25854⟩⟩) exact27948RawTerms .large 27947 .exactZero (none)

def event27949 : Event := .preFoldPolynomial 27948 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6694⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6793⟩⟩, ⟨.program ⟨214⟩, ⟨7843⟩⟩, ⟨.program ⟨214⟩, ⟨25850⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨11229⟩⟩, ⟨.program ⟨214⟩, ⟨13583⟩⟩], [⟨.program ⟨214⟩, ⟨23464⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15595⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩] .exactZero none

def exact27950RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6694⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6793⟩⟩, ⟨.program ⟨214⟩, ⟨7843⟩⟩, ⟨.program ⟨214⟩, ⟨25850⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨11229⟩⟩, ⟨.program ⟨214⟩, ⟨13583⟩⟩], [⟨.program ⟨214⟩, ⟨23464⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15595⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

def event27950 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨25854⟩⟩) 27949 exact27950RawTerms .large 27947 .exactZero (none)

def event27951 : Event := .specializationComputed (⟨.program ⟨214⟩, ⟨13585⟩⟩) ⟨⟨107⟩, ⟨12⟩, ⟨109⟩⟩ ⟨27785, 27951⟩

def event27952 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨19327⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19324⟩⟩]⟩) (1) 0 2 (.universal 27951 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19324⟩⟩]⟩) (none) 27950)

def event27953 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19327⟩⟩, .relation 27952 0, ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6694⟩⟩]⟩, (1)⟩)

def event27954 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19327⟩⟩, .relation 27952 1, ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6793⟩⟩, ⟨.program ⟨214⟩, ⟨7843⟩⟩, ⟨.program ⟨214⟩, ⟨25850⟩⟩]⟩, (-1)⟩)

def event27955 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19327⟩⟩, .relation 27952 2, ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨11229⟩⟩, ⟨.program ⟨214⟩, ⟨13583⟩⟩], [⟨.program ⟨214⟩, ⟨23464⟩⟩]⟩, (1)⟩)

def event27956 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19327⟩⟩, .relation 27952 3, ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15595⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact27957RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6694⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6793⟩⟩, ⟨.program ⟨214⟩, ⟨7843⟩⟩, ⟨.program ⟨214⟩, ⟨25850⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨11229⟩⟩, ⟨.program ⟨214⟩, ⟨13583⟩⟩], [⟨.program ⟨214⟩, ⟨23464⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15595⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact27957RawTermsValid :
    exact27957RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event27957 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19327⟩⟩) exact27957RawTerms .large 27781 (.finite 1811303510016) (some (27783))

def event27958 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25852⟩⟩) 0 ⟨19327⟩ 27957

def event27959 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25852⟩⟩) 1 ⟨25851⟩ 27771

def event27960 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25852⟩⟩) (.sum [.predecessor 0 27958 .coefficient, .predecessor 1 27959 .coefficient])

def event27961 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25852⟩⟩, .operator (⟨27957, 2⟩, ⟨27771, 1⟩), ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨11229⟩⟩, ⟨.program ⟨214⟩, ⟨13583⟩⟩], [⟨.program ⟨214⟩, ⟨23464⟩⟩]⟩, (-1)⟩)

def event27962 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25852⟩⟩, .operator (⟨27957, 1⟩, ⟨27771, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6793⟩⟩, ⟨.program ⟨214⟩, ⟨7843⟩⟩, ⟨.program ⟨214⟩, ⟨25850⟩⟩]⟩, (1)⟩)

def event27963 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25852⟩⟩) (.sum [.result 27957 .summary, .result 27771 .summary])

def exact27964RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6694⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15595⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact27964RawTermsValid :
    exact27964RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event27964 : Event := .resultExact (⟨.program ⟨214⟩, ⟨25852⟩⟩) exact27964RawTerms .large 27960 (.finite 352036291489792) (some (27963))

def event27965 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27256⟩⟩) 0 ⟨25852⟩ 27964

def event27966 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27256⟩⟩) 1 ⟨27254⟩ 27687

def event27967 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27256⟩⟩) (.product (.predecessor 0 27965 .coefficient) (.predecessor 1 27966 .coefficient) (⟨false, false, none, none, none⟩))

def event27968 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27256⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨27254⟩⟩]⟩) [⟨.result 27687 .coefficient, false, none⟩])

def event27969 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27256⟩⟩) (.product (.result 27964 .summary) (.transfer 27968) (⟨false, false, none, none, none⟩))

def event27970 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27256⟩⟩, .operator (⟨27964, 0⟩, ⟨27687, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6694⟩⟩, ⟨.program ⟨214⟩, ⟨27254⟩⟩]⟩, (1)⟩)

def event27971 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27256⟩⟩, .operator (⟨27964, 1⟩, ⟨27687, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15595⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27254⟩⟩]⟩, (-1)⟩)

def event27972 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨27256⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15595⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27254⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨27254⟩⟩) ⟨23982⟩ 27684)

def event27973 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27256⟩⟩, .relation 27972 0, ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15595⟩⟩], [⟨.program ⟨214⟩, ⟨23982⟩⟩]⟩, (-1)⟩)

def exact27974RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6694⟩⟩, ⟨.program ⟨214⟩, ⟨27254⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15595⟩⟩], [⟨.program ⟨214⟩, ⟨23982⟩⟩]⟩, (-1)⟩]

theorem exact27974RawTermsValid :
    exact27974RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event27974 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27256⟩⟩) exact27974RawTerms .large 27967 (.finite 1291978822348200476672) (some (27969))

def event27975 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20980⟩⟩) 0 ⟨15596⟩ 1158

def event27976 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20980⟩⟩) (.authority (.relationPreimageSource ⟨37⟩))

def exact27977RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨20980⟩⟩]⟩, (1)⟩]

theorem exact27977RawTermsValid :
    exact27977RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event27977 : Event := .resultExact (⟨.program ⟨214⟩, ⟨20980⟩⟩) exact27977RawTerms (.finite 136065468) 27976 .exactZero (none)

def event27978 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20982⟩⟩) 0 ⟨20980⟩ 27977

def event27979 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20982⟩⟩) 1 ⟨2348⟩ 4

def event27980 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20982⟩⟩) (.scale (.predecessor 0 27978 .coefficient) (.value (.predecessor 1 27979 .coefficient)))

def exact27981RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨20980⟩⟩]⟩, (1)⟩]

theorem exact27981RawTermsValid :
    exact27981RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event27981 : Event := .resultExact (⟨.program ⟨214⟩, ⟨20982⟩⟩) exact27981RawTerms (.finite 136065468) 27980 .exactZero (none)

def event27982 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20983⟩⟩) 0 ⟨5559⟩ 21512

def event27983 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20983⟩⟩) 1 ⟨20982⟩ 27981

def event27984 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20983⟩⟩) (.product (.predecessor 0 27982 .coefficient) (.predecessor 1 27983 .coefficient) (⟨false, false, none, none, none⟩))

def event27985 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20983⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨20980⟩⟩]⟩) [⟨.result 27977 .coefficient, false, none⟩])

def event27986 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20983⟩⟩) (.product (.result 21512 .summary) (.transfer 27985) (⟨false, false, none, none, none⟩))

def event27987 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20983⟩⟩, .operator (⟨21512, 0⟩, ⟨27981, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20980⟩⟩]⟩, (1)⟩)

def event27988 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨20981⟩⟩)

def event27989 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event27990 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event27991 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨4989⟩⟩) (.authority (.operator))

def event27992 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨4989⟩⟩) (.finite 5)

def event27993 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event27994 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event27995 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event27996 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event27997 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 27996

def event27998 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 27994

def event27999 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 27997 .coefficient) (.value (.predecessor 1 27998 .coefficient)))

def event28000 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event28001 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5514⟩⟩) 0 ⟨5503⟩ 28000

def event28002 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5514⟩⟩) 1 ⟨4989⟩ 27992

def event28003 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5514⟩⟩) (.sum [.predecessor 0 28001 .coefficient, .predecessor 1 28002 .coefficient])

def event28004 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5514⟩⟩) (.finite 222)

def event28005 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5554⟩⟩) 0 ⟨5514⟩ 28004

def event28006 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5554⟩⟩) 1 ⟨961⟩ 27990

def event28007 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5554⟩⟩) (.identity (.predecessor 1 28006 .coefficient))

def event28008 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5554⟩⟩) (.finite 224)

def event28009 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11229⟩⟩) 0 ⟨5554⟩ 28008

def event28010 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11229⟩⟩) (.authority (.programFamilyFact))

def exact28011RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11229⟩⟩], []⟩, (1)⟩]

theorem exact28011RawTermsValid :
    exact28011RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28011 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11229⟩⟩) exact28011RawTerms (.finite 10) 28010 .exactZero (none)

def event28012 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13583⟩⟩) 0 ⟨5554⟩ 28008

def event28013 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13583⟩⟩) (.authority (.programFamilyFact))

def exact28014RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨13583⟩⟩], []⟩, (1)⟩]

theorem exact28014RawTermsValid :
    exact28014RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28014 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13583⟩⟩) exact28014RawTerms (.finite 10) 28013 .exactZero (none)

def event28015 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13584⟩⟩) 0 ⟨13583⟩ 28014

def event28016 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13584⟩⟩) 1 ⟨11229⟩ 28011

def event28017 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13584⟩⟩) (.product (.predecessor 0 28015 .coefficient) (.predecessor 1 28016 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event28018 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13584⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨11229⟩⟩, ⟨.program ⟨214⟩, ⟨13583⟩⟩], []⟩) [⟨.result 28014 .coefficient, true, some 1⟩, ⟨.result 28011 .coefficient, true, some 1⟩])

def event28019 : Event := .survivorFold (1) 28018

def exact28020RawTerms : List Term := []

theorem exact28020RawTermsValid :
    exact28020RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28020 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13584⟩⟩) exact28020RawTerms (.finite 100) 28017 (.finite 100) (some (28018))

def event28021 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13585⟩⟩) 0 ⟨13584⟩ 28020

def event28022 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13585⟩⟩) (.identity (.predecessor 0 28021 .coefficient))

def event28023 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨13585⟩⟩) (.finite 100)

def event28024 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15595⟩⟩) 0 ⟨13585⟩ 28023

def event28025 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15595⟩⟩) (.authority (.programFamilyFact))

def exact28026RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15595⟩⟩], []⟩, (1)⟩]

theorem exact28026RawTermsValid :
    exact28026RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28026 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15595⟩⟩) exact28026RawTerms (.finite 10) 28025 .exactZero (none)

def event28027 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15596⟩⟩) 0 ⟨15595⟩ 28026

def event28028 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15596⟩⟩) (.identity (.predecessor 0 28027 .coefficient))

def event28029 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨15596⟩⟩) (.finite 10)

def event28030 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20980⟩⟩) 0 ⟨15596⟩ 28029

def event28031 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20980⟩⟩) (.authority (.relationPreimageSource ⟨37⟩))

def exact28032RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨20980⟩⟩]⟩, (1)⟩]

theorem exact28032RawTermsValid :
    exact28032RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28032 : Event := .resultExact (⟨.program ⟨214⟩, ⟨20980⟩⟩) exact28032RawTerms (.finite 136065468) 28031 .exactZero (none)

def event28033 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6⟩⟩) (.authority (.operator))

def exact28034RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩]⟩, (1)⟩]

theorem exact28034RawTermsValid :
    exact28034RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28034 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6⟩⟩) exact28034RawTerms .large 28033 .exactZero (none)

def event28035 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20981⟩⟩) 0 ⟨6⟩ 28034

def event28036 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20981⟩⟩) 1 ⟨20980⟩ 28032

def event28037 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20981⟩⟩) (.product (.predecessor 0 28035 .coefficient) (.predecessor 1 28036 .coefficient) (⟨false, false, none, none, none⟩))

def event28038 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20981⟩⟩, .operator (⟨28034, 0⟩, ⟨28032, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20980⟩⟩]⟩, (1)⟩)

def exact28039RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20980⟩⟩]⟩, (1)⟩]

theorem exact28039RawTermsValid :
    exact28039RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28039 : Event := .resultExact (⟨.program ⟨214⟩, ⟨20981⟩⟩) exact28039RawTerms .large 28037 .exactZero (none)

def event28040 : Event := .preFoldPolynomial 28039 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20980⟩⟩]⟩, (1)⟩] .exactZero none

def exact28041RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20980⟩⟩]⟩, (1)⟩]

def event28041 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨20981⟩⟩) 28040 exact28041RawTerms .large 28037 .exactZero (none)

def event28042 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨27259⟩⟩)

def event28043 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event28044 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event28045 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨4989⟩⟩) (.authority (.operator))

def event28046 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨4989⟩⟩) (.finite 5)

def event28047 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event28048 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event28049 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event28050 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event28051 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 28050

def event28052 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 28048

def event28053 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 28051 .coefficient) (.value (.predecessor 1 28052 .coefficient)))

def event28054 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event28055 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5514⟩⟩) 0 ⟨5503⟩ 28054

def event28056 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5514⟩⟩) 1 ⟨4989⟩ 28046

def event28057 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5514⟩⟩) (.sum [.predecessor 0 28055 .coefficient, .predecessor 1 28056 .coefficient])

def event28058 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5514⟩⟩) (.finite 222)

def event28059 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5554⟩⟩) 0 ⟨5514⟩ 28058

def event28060 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5554⟩⟩) 1 ⟨961⟩ 28044

def event28061 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5554⟩⟩) (.identity (.predecessor 1 28060 .coefficient))

def event28062 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5554⟩⟩) (.finite 224)

def event28063 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11229⟩⟩) 0 ⟨5554⟩ 28062

def event28064 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11229⟩⟩) (.authority (.programFamilyFact))

def exact28065RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11229⟩⟩], []⟩, (1)⟩]

theorem exact28065RawTermsValid :
    exact28065RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28065 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11229⟩⟩) exact28065RawTerms (.finite 10) 28064 .exactZero (none)

def event28066 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13583⟩⟩) 0 ⟨5554⟩ 28062

def event28067 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13583⟩⟩) (.authority (.programFamilyFact))

def exact28068RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨13583⟩⟩], []⟩, (1)⟩]

theorem exact28068RawTermsValid :
    exact28068RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28068 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13583⟩⟩) exact28068RawTerms (.finite 10) 28067 .exactZero (none)

def event28069 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13584⟩⟩) 0 ⟨13583⟩ 28068

def event28070 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13584⟩⟩) 1 ⟨11229⟩ 28065

def event28071 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13584⟩⟩) (.product (.predecessor 0 28069 .coefficient) (.predecessor 1 28070 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event28072 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨13584⟩⟩, .operator (⟨28068, 0⟩, ⟨28065, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨11229⟩⟩, ⟨.program ⟨214⟩, ⟨13583⟩⟩], []⟩, (1)⟩)

def exact28073RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11229⟩⟩, ⟨.program ⟨214⟩, ⟨13583⟩⟩], []⟩, (1)⟩]

theorem exact28073RawTermsValid :
    exact28073RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28073 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13584⟩⟩) exact28073RawTerms (.finite 100) 28071 .exactZero (none)

def event28074 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13585⟩⟩) 0 ⟨13584⟩ 28073

def event28075 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13585⟩⟩) (.identity (.predecessor 0 28074 .coefficient))

def event28076 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨13585⟩⟩) (.finite 100)

def event28077 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15595⟩⟩) 0 ⟨13585⟩ 28076

def event28078 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15595⟩⟩) (.authority (.programFamilyFact))

def exact28079RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15595⟩⟩], []⟩, (1)⟩]

theorem exact28079RawTermsValid :
    exact28079RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28079 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15595⟩⟩) exact28079RawTerms (.finite 10) 28078 .exactZero (none)

def event28080 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15596⟩⟩) 0 ⟨15595⟩ 28079

def event28081 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15596⟩⟩) (.identity (.predecessor 0 28080 .coefficient))

def event28082 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨15596⟩⟩) (.finite 10)

def event28083 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23980⟩⟩) 0 ⟨15596⟩ 28082

def event28084 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23980⟩⟩) (.authority (.programFamilyFact))

def event28085 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨23980⟩⟩) (.finite 3720)

def event28086 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨6689⟩⟩) .missing

def event28087 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23982⟩⟩) 0 ⟨6689⟩ 28086

def event28088 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23982⟩⟩) 1 ⟨23980⟩ 28085

def event28089 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23982⟩⟩) (.authority (.operator))

def exact28090RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨23982⟩⟩]⟩, (1)⟩]

theorem exact28090RawTermsValid :
    exact28090RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28090 : Event := .resultExact (⟨.program ⟨214⟩, ⟨23982⟩⟩) exact28090RawTerms .large 28089 .exactZero (none)

def event28091 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27254⟩⟩) 0 ⟨23982⟩ 28090

def event28092 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27254⟩⟩) (.authority (.operator))

def exact28093RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨27254⟩⟩]⟩, (1)⟩]

theorem exact28093RawTermsValid :
    exact28093RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28093 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27254⟩⟩) exact28093RawTerms (.finite 8192) 28092 .exactZero (none)

def event28094 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨110⟩⟩) (.authority (.operator))

def event28095 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨110⟩⟩) .exactZero

def event28096 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15670⟩⟩) 0 ⟨15596⟩ 28082

def event28097 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15670⟩⟩) 1 ⟨110⟩ 28095

def event28098 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15670⟩⟩) (.sum [.predecessor 0 28096 .coefficient, .predecessor 1 28097 .coefficient])

def event28099 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨15670⟩⟩) (.finite 10)

def event28100 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15671⟩⟩) 0 ⟨15670⟩ 28099

def event28101 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15671⟩⟩) (.identity (.predecessor 0 28100 .coefficient))

def exact28102RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15595⟩⟩], []⟩, (1)⟩]

theorem exact28102RawTermsValid :
    exact28102RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28102 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15671⟩⟩) exact28102RawTerms (.finite 10) 28101 .exactZero (none)

def event28103 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6544⟩⟩) (.authority (.factStore))

def exact28104RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact28104RawTermsValid :
    exact28104RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28104 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6544⟩⟩) exact28104RawTerms .large 28103 .exactZero (none)

def event28105 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15672⟩⟩) 0 ⟨6544⟩ 28104

def event28106 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15672⟩⟩) 1 ⟨15671⟩ 28102

def event28107 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15672⟩⟩) (.product (.predecessor 0 28105 .coefficient) (.predecessor 1 28106 .coefficient) (⟨false, false, none, none, none⟩))

def event28108 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨15672⟩⟩, .operator (⟨28104, 0⟩, ⟨28102, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨15595⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact28109RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15595⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact28109RawTermsValid :
    exact28109RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28109 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15672⟩⟩) exact28109RawTerms .large 28107 .exactZero (none)

def event28110 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6694⟩⟩) 0 ⟨6689⟩ 28086

def event28111 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6694⟩⟩) (.authority (.operator))

def exact28112RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6694⟩⟩]⟩, (1)⟩]

theorem exact28112RawTermsValid :
    exact28112RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28112 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6694⟩⟩) exact28112RawTerms .large 28111 .exactZero (none)

def event28113 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15673⟩⟩) 0 ⟨6694⟩ 28112

def event28114 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15673⟩⟩) 1 ⟨15672⟩ 28109

def event28115 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15673⟩⟩) (.sum [.predecessor 0 28113 .coefficient, .predecessor 1 28114 .coefficient])

def exact28116RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6694⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15595⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact28116RawTermsValid :
    exact28116RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28116 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15673⟩⟩) exact28116RawTerms .large 28115 .exactZero (none)

def event28117 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27255⟩⟩) 0 ⟨15673⟩ 28116

def event28118 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27255⟩⟩) 1 ⟨27254⟩ 28093

def event28119 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27255⟩⟩) (.product (.predecessor 0 28117 .coefficient) (.predecessor 1 28118 .coefficient) (⟨false, false, none, none, none⟩))

def event28120 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27255⟩⟩, .operator (⟨28116, 0⟩, ⟨28093, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6694⟩⟩, ⟨.program ⟨214⟩, ⟨27254⟩⟩]⟩, (1)⟩)

def event28121 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27255⟩⟩, .operator (⟨28116, 1⟩, ⟨28093, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨15595⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27254⟩⟩]⟩, (-1)⟩)

def event28122 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨27255⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨15595⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27254⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨27254⟩⟩) ⟨23982⟩ 28090)

def event28123 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27255⟩⟩, .relation 28122 0, ⟨[⟨.program ⟨214⟩, ⟨15595⟩⟩], [⟨.program ⟨214⟩, ⟨23982⟩⟩]⟩, (-1)⟩)

def exact28124RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6694⟩⟩, ⟨.program ⟨214⟩, ⟨27254⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15595⟩⟩], [⟨.program ⟨214⟩, ⟨23982⟩⟩]⟩, (-1)⟩]

theorem exact28124RawTermsValid :
    exact28124RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28124 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27255⟩⟩) exact28124RawTerms .large 28119 .exactZero (none)

def event28125 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15638⟩⟩) 0 ⟨15596⟩ 28082

def event28126 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15638⟩⟩) (.authority (.programFamilyFact))

def exact28127RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15638⟩⟩], []⟩, (1)⟩]

theorem exact28127RawTermsValid :
    exact28127RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28127 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15638⟩⟩) exact28127RawTerms (.finite 58) 28126 .exactZero (none)

def event28128 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15639⟩⟩) 0 ⟨6544⟩ 28104

def event28129 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15639⟩⟩) 1 ⟨15638⟩ 28127

def event28130 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15639⟩⟩) (.product (.predecessor 0 28128 .coefficient) (.predecessor 1 28129 .coefficient) (⟨false, true, none, none, some 1⟩))

def event28131 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨15639⟩⟩, .operator (⟨28104, 0⟩, ⟨28127, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨15638⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact28132RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15638⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact28132RawTermsValid :
    exact28132RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28132 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15639⟩⟩) exact28132RawTerms .large 28130 .exactZero (none)

def event28133 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6717⟩⟩) 0 ⟨6689⟩ 28086

def event28134 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6717⟩⟩) (.authority (.operator))

def exact28135RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6717⟩⟩]⟩, (1)⟩]

theorem exact28135RawTermsValid :
    exact28135RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28135 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6717⟩⟩) exact28135RawTerms .large 28134 .exactZero (none)

def event28136 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15640⟩⟩) 0 ⟨6717⟩ 28135

def event28137 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15640⟩⟩) 1 ⟨15639⟩ 28132

def event28138 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15640⟩⟩) (.sum [.predecessor 0 28136 .coefficient, .predecessor 1 28137 .coefficient])

def exact28139RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6717⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15638⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact28139RawTermsValid :
    exact28139RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28139 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15640⟩⟩) exact28139RawTerms .large 28138 .exactZero (none)

def event28140 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27259⟩⟩) 0 ⟨15640⟩ 28139

def event28141 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27259⟩⟩) 1 ⟨27255⟩ 28124

def event28142 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27259⟩⟩) (.sum [.predecessor 0 28140 .coefficient, .predecessor 1 28141 .coefficient])

def exact28143RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6694⟩⟩, ⟨.program ⟨214⟩, ⟨27254⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6717⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15595⟩⟩], [⟨.program ⟨214⟩, ⟨23982⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15638⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact28143RawTermsValid :
    exact28143RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28143 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27259⟩⟩) exact28143RawTerms .large 28142 .exactZero (none)

def event28144 : Event := .preFoldPolynomial 28143 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6694⟩⟩, ⟨.program ⟨214⟩, ⟨27254⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6717⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15595⟩⟩], [⟨.program ⟨214⟩, ⟨23982⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15638⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩] .exactZero none

def exact28145RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6694⟩⟩, ⟨.program ⟨214⟩, ⟨27254⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6717⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15595⟩⟩], [⟨.program ⟨214⟩, ⟨23982⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15638⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

def event28145 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨27259⟩⟩) 28144 exact28145RawTerms .large 28142 .exactZero (none)

def event28146 : Event := .specializationComputed (⟨.program ⟨214⟩, ⟨15596⟩⟩) ⟨⟨130⟩, ⟨37⟩, ⟨109⟩⟩ ⟨27988, 28146⟩

def event28147 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨20983⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20980⟩⟩]⟩) (1) 0 2 (.universal 28146 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20980⟩⟩]⟩) (none) 28145)

def event28148 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20983⟩⟩, .relation 28147 1, ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6717⟩⟩]⟩, (1)⟩)

def event28149 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20983⟩⟩, .relation 28147 0, ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6694⟩⟩, ⟨.program ⟨214⟩, ⟨27254⟩⟩]⟩, (-1)⟩)

def event28150 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20983⟩⟩, .relation 28147 2, ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15595⟩⟩], [⟨.program ⟨214⟩, ⟨23982⟩⟩]⟩, (1)⟩)

def event28151 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20983⟩⟩, .relation 28147 3, ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15638⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact28152RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6694⟩⟩, ⟨.program ⟨214⟩, ⟨27254⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6717⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15595⟩⟩], [⟨.program ⟨214⟩, ⟨23982⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15638⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact28152RawTermsValid :
    exact28152RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28152 : Event := .resultExact (⟨.program ⟨214⟩, ⟨20983⟩⟩) exact28152RawTerms .large 27984 (.finite 1811303510016) (some (27986))

def event28153 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27257⟩⟩) 0 ⟨20983⟩ 28152

def event28154 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27257⟩⟩) 1 ⟨27256⟩ 27974

def event28155 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27257⟩⟩) (.sum [.predecessor 0 28153 .coefficient, .predecessor 1 28154 .coefficient])

def event28156 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27257⟩⟩, .operator (⟨28152, 0⟩, ⟨27974, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6694⟩⟩, ⟨.program ⟨214⟩, ⟨27254⟩⟩]⟩, (1)⟩)

def event28157 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27257⟩⟩, .operator (⟨28152, 2⟩, ⟨27974, 1⟩), ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15595⟩⟩], [⟨.program ⟨214⟩, ⟨23982⟩⟩]⟩, (-1)⟩)

def event28158 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27257⟩⟩) (.sum [.result 28152 .summary, .result 27974 .summary])

def exact28159RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6717⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15638⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact28159RawTermsValid :
    exact28159RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28159 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27257⟩⟩) exact28159RawTerms .large 28155 (.finite 1291978824159503986688) (some (28158))

def eventLeaf1744 : Array AnnotatedEvent := #[
  { event := event27904
    frameStart := 27833 },
  { event := event27905
    frameStart := 27833 },
  { event := event27906
    frameStart := 27833 },
  { event := event27907
    frameStart := 27833 },
  { event := event27908
    frameStart := 27833 },
  { event := event27909
    frameStart := 27833 },
  { event := event27910
    frameStart := 27833 },
  { event := event27911
    frameStart := 27833 },
  { event := event27912
    frameStart := 27833 },
  { event := event27913
    frameStart := 27833 },
  { event := event27914
    frameStart := 27833 },
  { event := event27915
    frameStart := 27833 },
  { event := event27916
    frameStart := 27833 },
  { event := event27917
    frameStart := 27833 },
  { event := event27918
    frameStart := 27833 },
  { event := event27919
    frameStart := 27833 }
]

def eventLeaf1745 : Array AnnotatedEvent := #[
  { event := event27920
    frameStart := 27833 },
  { event := event27921
    frameStart := 27833 },
  { event := event27922
    frameStart := 27833 },
  { event := event27923
    frameStart := 27833 },
  { event := event27924
    frameStart := 27833 },
  { event := event27925
    frameStart := 27833 },
  { event := event27926
    frameStart := 27833 },
  { event := event27927
    frameStart := 27833 },
  { event := event27928
    frameStart := 27833 },
  { event := event27929
    frameStart := 27833 },
  { event := event27930
    frameStart := 27833 },
  { event := event27931
    frameStart := 27833 },
  { event := event27932
    frameStart := 27833 },
  { event := event27933
    frameStart := 27833 },
  { event := event27934
    frameStart := 27833 },
  { event := event27935
    frameStart := 27833 }
]

def eventLeaf1746 : Array AnnotatedEvent := #[
  { event := event27936
    frameStart := 27833 },
  { event := event27937
    frameStart := 27833 },
  { event := event27938
    frameStart := 27833 },
  { event := event27939
    frameStart := 27833 },
  { event := event27940
    frameStart := 27833 },
  { event := event27941
    frameStart := 27833 },
  { event := event27942
    frameStart := 27833 },
  { event := event27943
    frameStart := 27833 },
  { event := event27944
    frameStart := 27833 },
  { event := event27945
    frameStart := 27833 },
  { event := event27946
    frameStart := 27833 },
  { event := event27947
    frameStart := 27833 },
  { event := event27948
    frameStart := 27833 },
  { event := event27949
    frameStart := 27833 },
  { event := event27950
    frameStart := 27833 },
  { event := event27951
    frameStart := 0 }
]

def eventLeaf1747 : Array AnnotatedEvent := #[
  { event := event27952
    frameStart := 0 },
  { event := event27953
    frameStart := 0 },
  { event := event27954
    frameStart := 0 },
  { event := event27955
    frameStart := 0 },
  { event := event27956
    frameStart := 0 },
  { event := event27957
    frameStart := 0 },
  { event := event27958
    frameStart := 0 },
  { event := event27959
    frameStart := 0 },
  { event := event27960
    frameStart := 0 },
  { event := event27961
    frameStart := 0 },
  { event := event27962
    frameStart := 0 },
  { event := event27963
    frameStart := 0 },
  { event := event27964
    frameStart := 0 },
  { event := event27965
    frameStart := 0 },
  { event := event27966
    frameStart := 0 },
  { event := event27967
    frameStart := 0 }
]

def eventLeaf1748 : Array AnnotatedEvent := #[
  { event := event27968
    frameStart := 0 },
  { event := event27969
    frameStart := 0 },
  { event := event27970
    frameStart := 0 },
  { event := event27971
    frameStart := 0 },
  { event := event27972
    frameStart := 0 },
  { event := event27973
    frameStart := 0 },
  { event := event27974
    frameStart := 0 },
  { event := event27975
    frameStart := 0 },
  { event := event27976
    frameStart := 0 },
  { event := event27977
    frameStart := 0 },
  { event := event27978
    frameStart := 0 },
  { event := event27979
    frameStart := 0 },
  { event := event27980
    frameStart := 0 },
  { event := event27981
    frameStart := 0 },
  { event := event27982
    frameStart := 0 },
  { event := event27983
    frameStart := 0 }
]

def eventLeaf1749 : Array AnnotatedEvent := #[
  { event := event27984
    frameStart := 0 },
  { event := event27985
    frameStart := 0 },
  { event := event27986
    frameStart := 0 },
  { event := event27987
    frameStart := 0 },
  { event := event27988
    frameStart := 27988 },
  { event := event27989
    frameStart := 27988 },
  { event := event27990
    frameStart := 27988 },
  { event := event27991
    frameStart := 27988 },
  { event := event27992
    frameStart := 27988 },
  { event := event27993
    frameStart := 27988 },
  { event := event27994
    frameStart := 27988 },
  { event := event27995
    frameStart := 27988 },
  { event := event27996
    frameStart := 27988 },
  { event := event27997
    frameStart := 27988 },
  { event := event27998
    frameStart := 27988 },
  { event := event27999
    frameStart := 27988 }
]

def eventLeaf1750 : Array AnnotatedEvent := #[
  { event := event28000
    frameStart := 27988 },
  { event := event28001
    frameStart := 27988 },
  { event := event28002
    frameStart := 27988 },
  { event := event28003
    frameStart := 27988 },
  { event := event28004
    frameStart := 27988 },
  { event := event28005
    frameStart := 27988 },
  { event := event28006
    frameStart := 27988 },
  { event := event28007
    frameStart := 27988 },
  { event := event28008
    frameStart := 27988 },
  { event := event28009
    frameStart := 27988 },
  { event := event28010
    frameStart := 27988 },
  { event := event28011
    frameStart := 27988 },
  { event := event28012
    frameStart := 27988 },
  { event := event28013
    frameStart := 27988 },
  { event := event28014
    frameStart := 27988 },
  { event := event28015
    frameStart := 27988 }
]

def eventLeaf1751 : Array AnnotatedEvent := #[
  { event := event28016
    frameStart := 27988 },
  { event := event28017
    frameStart := 27988 },
  { event := event28018
    frameStart := 27988 },
  { event := event28019
    frameStart := 27988 },
  { event := event28020
    frameStart := 27988 },
  { event := event28021
    frameStart := 27988 },
  { event := event28022
    frameStart := 27988 },
  { event := event28023
    frameStart := 27988 },
  { event := event28024
    frameStart := 27988 },
  { event := event28025
    frameStart := 27988 },
  { event := event28026
    frameStart := 27988 },
  { event := event28027
    frameStart := 27988 },
  { event := event28028
    frameStart := 27988 },
  { event := event28029
    frameStart := 27988 },
  { event := event28030
    frameStart := 27988 },
  { event := event28031
    frameStart := 27988 }
]

def eventLeaf1752 : Array AnnotatedEvent := #[
  { event := event28032
    frameStart := 27988 },
  { event := event28033
    frameStart := 27988 },
  { event := event28034
    frameStart := 27988 },
  { event := event28035
    frameStart := 27988 },
  { event := event28036
    frameStart := 27988 },
  { event := event28037
    frameStart := 27988 },
  { event := event28038
    frameStart := 27988 },
  { event := event28039
    frameStart := 27988 },
  { event := event28040
    frameStart := 27988 },
  { event := event28041
    frameStart := 27988 },
  { event := event28042
    frameStart := 28042 },
  { event := event28043
    frameStart := 28042 },
  { event := event28044
    frameStart := 28042 },
  { event := event28045
    frameStart := 28042 },
  { event := event28046
    frameStart := 28042 },
  { event := event28047
    frameStart := 28042 }
]

def eventLeaf1753 : Array AnnotatedEvent := #[
  { event := event28048
    frameStart := 28042 },
  { event := event28049
    frameStart := 28042 },
  { event := event28050
    frameStart := 28042 },
  { event := event28051
    frameStart := 28042 },
  { event := event28052
    frameStart := 28042 },
  { event := event28053
    frameStart := 28042 },
  { event := event28054
    frameStart := 28042 },
  { event := event28055
    frameStart := 28042 },
  { event := event28056
    frameStart := 28042 },
  { event := event28057
    frameStart := 28042 },
  { event := event28058
    frameStart := 28042 },
  { event := event28059
    frameStart := 28042 },
  { event := event28060
    frameStart := 28042 },
  { event := event28061
    frameStart := 28042 },
  { event := event28062
    frameStart := 28042 },
  { event := event28063
    frameStart := 28042 }
]

def eventLeaf1754 : Array AnnotatedEvent := #[
  { event := event28064
    frameStart := 28042 },
  { event := event28065
    frameStart := 28042 },
  { event := event28066
    frameStart := 28042 },
  { event := event28067
    frameStart := 28042 },
  { event := event28068
    frameStart := 28042 },
  { event := event28069
    frameStart := 28042 },
  { event := event28070
    frameStart := 28042 },
  { event := event28071
    frameStart := 28042 },
  { event := event28072
    frameStart := 28042 },
  { event := event28073
    frameStart := 28042 },
  { event := event28074
    frameStart := 28042 },
  { event := event28075
    frameStart := 28042 },
  { event := event28076
    frameStart := 28042 },
  { event := event28077
    frameStart := 28042 },
  { event := event28078
    frameStart := 28042 },
  { event := event28079
    frameStart := 28042 }
]

def eventLeaf1755 : Array AnnotatedEvent := #[
  { event := event28080
    frameStart := 28042 },
  { event := event28081
    frameStart := 28042 },
  { event := event28082
    frameStart := 28042 },
  { event := event28083
    frameStart := 28042 },
  { event := event28084
    frameStart := 28042 },
  { event := event28085
    frameStart := 28042 },
  { event := event28086
    frameStart := 28042 },
  { event := event28087
    frameStart := 28042 },
  { event := event28088
    frameStart := 28042 },
  { event := event28089
    frameStart := 28042 },
  { event := event28090
    frameStart := 28042 },
  { event := event28091
    frameStart := 28042 },
  { event := event28092
    frameStart := 28042 },
  { event := event28093
    frameStart := 28042 },
  { event := event28094
    frameStart := 28042 },
  { event := event28095
    frameStart := 28042 }
]

def eventLeaf1756 : Array AnnotatedEvent := #[
  { event := event28096
    frameStart := 28042 },
  { event := event28097
    frameStart := 28042 },
  { event := event28098
    frameStart := 28042 },
  { event := event28099
    frameStart := 28042 },
  { event := event28100
    frameStart := 28042 },
  { event := event28101
    frameStart := 28042 },
  { event := event28102
    frameStart := 28042 },
  { event := event28103
    frameStart := 28042 },
  { event := event28104
    frameStart := 28042 },
  { event := event28105
    frameStart := 28042 },
  { event := event28106
    frameStart := 28042 },
  { event := event28107
    frameStart := 28042 },
  { event := event28108
    frameStart := 28042 },
  { event := event28109
    frameStart := 28042 },
  { event := event28110
    frameStart := 28042 },
  { event := event28111
    frameStart := 28042 }
]

def eventLeaf1757 : Array AnnotatedEvent := #[
  { event := event28112
    frameStart := 28042 },
  { event := event28113
    frameStart := 28042 },
  { event := event28114
    frameStart := 28042 },
  { event := event28115
    frameStart := 28042 },
  { event := event28116
    frameStart := 28042 },
  { event := event28117
    frameStart := 28042 },
  { event := event28118
    frameStart := 28042 },
  { event := event28119
    frameStart := 28042 },
  { event := event28120
    frameStart := 28042 },
  { event := event28121
    frameStart := 28042 },
  { event := event28122
    frameStart := 28042 },
  { event := event28123
    frameStart := 28042 },
  { event := event28124
    frameStart := 28042 },
  { event := event28125
    frameStart := 28042 },
  { event := event28126
    frameStart := 28042 },
  { event := event28127
    frameStart := 28042 }
]

def eventLeaf1758 : Array AnnotatedEvent := #[
  { event := event28128
    frameStart := 28042 },
  { event := event28129
    frameStart := 28042 },
  { event := event28130
    frameStart := 28042 },
  { event := event28131
    frameStart := 28042 },
  { event := event28132
    frameStart := 28042 },
  { event := event28133
    frameStart := 28042 },
  { event := event28134
    frameStart := 28042 },
  { event := event28135
    frameStart := 28042 },
  { event := event28136
    frameStart := 28042 },
  { event := event28137
    frameStart := 28042 },
  { event := event28138
    frameStart := 28042 },
  { event := event28139
    frameStart := 28042 },
  { event := event28140
    frameStart := 28042 },
  { event := event28141
    frameStart := 28042 },
  { event := event28142
    frameStart := 28042 },
  { event := event28143
    frameStart := 28042 }
]

def eventLeaf1759 : Array AnnotatedEvent := #[
  { event := event28144
    frameStart := 28042 },
  { event := event28145
    frameStart := 28042 },
  { event := event28146
    frameStart := 0 },
  { event := event28147
    frameStart := 0 },
  { event := event28148
    frameStart := 0 },
  { event := event28149
    frameStart := 0 },
  { event := event28150
    frameStart := 0 },
  { event := event28151
    frameStart := 0 },
  { event := event28152
    frameStart := 0 },
  { event := event28153
    frameStart := 0 },
  { event := event28154
    frameStart := 0 },
  { event := event28155
    frameStart := 0 },
  { event := event28156
    frameStart := 0 },
  { event := event28157
    frameStart := 0 },
  { event := event28158
    frameStart := 0 },
  { event := event28159
    frameStart := 0 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events109
