import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events383

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event98048 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7860⟩⟩, .operator (⟨98044, 0⟩, ⟨98041, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6762⟩⟩, ⟨.program ⟨214⟩, ⟨7858⟩⟩]⟩, (1)⟩)

def exact98049RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6762⟩⟩, ⟨.program ⟨214⟩, ⟨7858⟩⟩]⟩, (1)⟩]

theorem exact98049RawTermsValid :
    exact98049RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event98049 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7860⟩⟩) exact98049RawTerms .large 98047 .exactZero (none)

def event98050 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14743⟩⟩) 0 ⟨7860⟩ 98049

def event98051 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14743⟩⟩) 1 ⟨14742⟩ 98026

def event98052 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14743⟩⟩) (.sum [.predecessor 0 98050 .coefficient, .predecessor 1 98051 .coefficient])

def exact98053RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6762⟩⟩, ⟨.program ⟨214⟩, ⟨7858⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨11625⟩⟩, ⟨.program ⟨214⟩, ⟨14614⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact98053RawTermsValid :
    exact98053RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event98053 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14743⟩⟩) exact98053RawTerms .large 98052 .exactZero (none)

def event98054 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26210⟩⟩) 0 ⟨14743⟩ 98053

def event98055 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26210⟩⟩) 1 ⟨26207⟩ 98010

def event98056 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26210⟩⟩) (.product (.predecessor 0 98054 .coefficient) (.predecessor 1 98055 .coefficient) (⟨false, false, none, none, none⟩))

def event98057 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26210⟩⟩, .operator (⟨98053, 0⟩, ⟨98010, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6762⟩⟩, ⟨.program ⟨214⟩, ⟨7858⟩⟩, ⟨.program ⟨214⟩, ⟨26207⟩⟩]⟩, (1)⟩)

def event98058 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26210⟩⟩, .operator (⟨98053, 1⟩, ⟨98010, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨11625⟩⟩, ⟨.program ⟨214⟩, ⟨14614⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26207⟩⟩]⟩, (-1)⟩)

def event98059 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨26210⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨11625⟩⟩, ⟨.program ⟨214⟩, ⟨14614⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26207⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨26207⟩⟩) ⟨23662⟩ 98007)

def event98060 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26210⟩⟩, .relation 98059 0, ⟨[⟨.program ⟨214⟩, ⟨11625⟩⟩, ⟨.program ⟨214⟩, ⟨14614⟩⟩], [⟨.program ⟨214⟩, ⟨23662⟩⟩]⟩, (-1)⟩)

def exact98061RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6762⟩⟩, ⟨.program ⟨214⟩, ⟨7858⟩⟩, ⟨.program ⟨214⟩, ⟨26207⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨11625⟩⟩, ⟨.program ⟨214⟩, ⟨14614⟩⟩], [⟨.program ⟨214⟩, ⟨23662⟩⟩]⟩, (-1)⟩]

theorem exact98061RawTermsValid :
    exact98061RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event98061 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26210⟩⟩) exact98061RawTerms .large 98056 .exactZero (none)

def event98062 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16168⟩⟩) 0 ⟨14616⟩ 97999

def event98063 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16168⟩⟩) (.authority (.programFamilyFact))

def exact98064RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16168⟩⟩], []⟩, (1)⟩]

theorem exact98064RawTermsValid :
    exact98064RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event98064 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16168⟩⟩) exact98064RawTerms (.finite 28) 98063 .exactZero (none)

def event98065 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16170⟩⟩) 0 ⟨6544⟩ 98021

def event98066 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16170⟩⟩) 1 ⟨16168⟩ 98064

def event98067 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16170⟩⟩) (.product (.predecessor 0 98065 .coefficient) (.predecessor 1 98066 .coefficient) (⟨false, true, none, none, some 1⟩))

def event98068 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨16170⟩⟩, .operator (⟨98021, 0⟩, ⟨98064, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨16168⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact98069RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16168⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact98069RawTermsValid :
    exact98069RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event98069 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16170⟩⟩) exact98069RawTerms .large 98067 .exactZero (none)

def event98070 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6699⟩⟩) 0 ⟨6689⟩ 98003

def event98071 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6699⟩⟩) (.authority (.operator))

def exact98072RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6699⟩⟩]⟩, (1)⟩]

theorem exact98072RawTermsValid :
    exact98072RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event98072 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6699⟩⟩) exact98072RawTerms .large 98071 .exactZero (none)

def event98073 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16171⟩⟩) 0 ⟨6699⟩ 98072

def event98074 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16171⟩⟩) 1 ⟨16170⟩ 98069

def event98075 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16171⟩⟩) (.sum [.predecessor 0 98073 .coefficient, .predecessor 1 98074 .coefficient])

def exact98076RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6699⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16168⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact98076RawTermsValid :
    exact98076RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event98076 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16171⟩⟩) exact98076RawTerms .large 98075 .exactZero (none)

def event98077 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26211⟩⟩) 0 ⟨16171⟩ 98076

def event98078 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26211⟩⟩) 1 ⟨26210⟩ 98061

def event98079 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26211⟩⟩) (.sum [.predecessor 0 98077 .coefficient, .predecessor 1 98078 .coefficient])

def exact98080RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6699⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6762⟩⟩, ⟨.program ⟨214⟩, ⟨7858⟩⟩, ⟨.program ⟨214⟩, ⟨26207⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨11625⟩⟩, ⟨.program ⟨214⟩, ⟨14614⟩⟩], [⟨.program ⟨214⟩, ⟨23662⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16168⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact98080RawTermsValid :
    exact98080RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event98080 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26211⟩⟩) exact98080RawTerms .large 98079 .exactZero (none)

def event98081 : Event := .preFoldPolynomial 98080 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6699⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6762⟩⟩, ⟨.program ⟨214⟩, ⟨7858⟩⟩, ⟨.program ⟨214⟩, ⟨26207⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨11625⟩⟩, ⟨.program ⟨214⟩, ⟨14614⟩⟩], [⟨.program ⟨214⟩, ⟨23662⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16168⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩] .exactZero none

def exact98082RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6699⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6762⟩⟩, ⟨.program ⟨214⟩, ⟨7858⟩⟩, ⟨.program ⟨214⟩, ⟨26207⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨11625⟩⟩, ⟨.program ⟨214⟩, ⟨14614⟩⟩], [⟨.program ⟨214⟩, ⟨23662⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16168⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

def event98082 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨26211⟩⟩) 98081 exact98082RawTerms .large 98079 .exactZero (none)

def event98083 : Event := .specializationComputed (⟨.program ⟨214⟩, ⟨14616⟩⟩) ⟨⟨112⟩, ⟨17⟩, ⟨109⟩⟩ ⟨97941, 98083⟩

def event98084 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨19664⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19661⟩⟩]⟩) (1) 0 2 (.universal 98083 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19661⟩⟩]⟩) (none) 98082)

def event98085 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19664⟩⟩, .relation 98084 0, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6699⟩⟩]⟩, (1)⟩)

def event98086 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19664⟩⟩, .relation 98084 1, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6762⟩⟩, ⟨.program ⟨214⟩, ⟨7858⟩⟩, ⟨.program ⟨214⟩, ⟨26207⟩⟩]⟩, (-1)⟩)

def event98087 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19664⟩⟩, .relation 98084 2, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨11625⟩⟩, ⟨.program ⟨214⟩, ⟨14614⟩⟩], [⟨.program ⟨214⟩, ⟨23662⟩⟩]⟩, (1)⟩)

def event98088 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19664⟩⟩, .relation 98084 3, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16168⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact98089RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6699⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6762⟩⟩, ⟨.program ⟨214⟩, ⟨7858⟩⟩, ⟨.program ⟨214⟩, ⟨26207⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨11625⟩⟩, ⟨.program ⟨214⟩, ⟨14614⟩⟩], [⟨.program ⟨214⟩, ⟨23662⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16168⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact98089RawTermsValid :
    exact98089RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event98089 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19664⟩⟩) exact98089RawTerms .large 97937 (.finite 1811303510016) (some (97939))

def event98090 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26209⟩⟩) 0 ⟨19664⟩ 98089

def event98091 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26209⟩⟩) 1 ⟨26208⟩ 97927

def event98092 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26209⟩⟩) (.sum [.predecessor 0 98090 .coefficient, .predecessor 1 98091 .coefficient])

def event98093 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26209⟩⟩, .operator (⟨98089, 2⟩, ⟨97927, 1⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨11625⟩⟩, ⟨.program ⟨214⟩, ⟨14614⟩⟩], [⟨.program ⟨214⟩, ⟨23662⟩⟩]⟩, (-1)⟩)

def event98094 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26209⟩⟩, .operator (⟨98089, 1⟩, ⟨97927, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6762⟩⟩, ⟨.program ⟨214⟩, ⟨7858⟩⟩, ⟨.program ⟨214⟩, ⟨26207⟩⟩]⟩, (1)⟩)

def event98095 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26209⟩⟩) (.sum [.result 98089 .summary, .result 97927 .summary])

def exact98096RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6699⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16168⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact98096RawTermsValid :
    exact98096RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event98096 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26209⟩⟩) exact98096RawTerms .large 98092 (.finite 352091253649408) (some (98095))

def event98097 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28267⟩⟩) 0 ⟨26209⟩ 98096

def event98098 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28267⟩⟩) 1 ⟨28265⟩ 97843

def event98099 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28267⟩⟩) (.product (.predecessor 0 98097 .coefficient) (.predecessor 1 98098 .coefficient) (⟨false, false, none, none, none⟩))

def event98100 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28267⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨28265⟩⟩]⟩) [⟨.result 97843 .coefficient, false, none⟩])

def event98101 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28267⟩⟩) (.product (.result 98096 .summary) (.transfer 98100) (⟨false, false, none, none, none⟩))

def event98102 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28267⟩⟩, .operator (⟨98096, 0⟩, ⟨97843, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28265⟩⟩]⟩, (1)⟩)

def event98103 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28267⟩⟩, .operator (⟨98096, 1⟩, ⟨97843, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16168⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28265⟩⟩]⟩, (-1)⟩)

def event98104 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨28267⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16168⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28265⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨28265⟩⟩) ⟨24279⟩ 97840)

def event98105 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28267⟩⟩, .relation 98104 0, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16168⟩⟩], [⟨.program ⟨214⟩, ⟨24279⟩⟩]⟩, (-1)⟩)

def exact98106RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28265⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16168⟩⟩], [⟨.program ⟨214⟩, ⟨24279⟩⟩]⟩, (-1)⟩]

theorem exact98106RawTermsValid :
    exact98106RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event98106 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28267⟩⟩) exact98106RawTerms .large 98099 (.finite 1292180534353385750528) (some (98101))

def event98107 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21677⟩⟩) 0 ⟨16169⟩ 4767

def event98108 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21677⟩⟩) (.authority (.relationPreimageSource ⟨48⟩))

def exact98109RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨21677⟩⟩]⟩, (1)⟩]

theorem exact98109RawTermsValid :
    exact98109RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event98109 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21677⟩⟩) exact98109RawTerms (.finite 136065468) 98108 .exactZero (none)

def event98110 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21679⟩⟩) 0 ⟨21677⟩ 98109

def event98111 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21679⟩⟩) 1 ⟨2348⟩ 4

def event98112 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21679⟩⟩) (.scale (.predecessor 0 98110 .coefficient) (.value (.predecessor 1 98111 .coefficient)))

def exact98113RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨21677⟩⟩]⟩, (1)⟩]

theorem exact98113RawTermsValid :
    exact98113RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event98113 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21679⟩⟩) exact98113RawTerms (.finite 136065468) 98112 .exactZero (none)

def event98114 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21680⟩⟩) 0 ⟨5509⟩ 94462

def event98115 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21680⟩⟩) 1 ⟨21679⟩ 98113

def event98116 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21680⟩⟩) (.product (.predecessor 0 98114 .coefficient) (.predecessor 1 98115 .coefficient) (⟨false, false, none, none, none⟩))

def event98117 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21680⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨21677⟩⟩]⟩) [⟨.result 98109 .coefficient, false, none⟩])

def event98118 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21680⟩⟩) (.product (.result 94462 .summary) (.transfer 98117) (⟨false, false, none, none, none⟩))

def event98119 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21680⟩⟩, .operator (⟨94462, 0⟩, ⟨98113, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21677⟩⟩]⟩, (1)⟩)

def event98120 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨21678⟩⟩)

def event98121 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event98122 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event98123 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event98124 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event98125 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 98124

def event98126 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 98122

def event98127 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 98125 .coefficient) (.value (.predecessor 1 98126 .coefficient)))

def event98128 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event98129 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11625⟩⟩) 0 ⟨5503⟩ 98128

def event98130 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11625⟩⟩) (.authority (.programFamilyFact))

def exact98131RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11625⟩⟩], []⟩, (1)⟩]

theorem exact98131RawTermsValid :
    exact98131RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event98131 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11625⟩⟩) exact98131RawTerms (.finite 28) 98130 .exactZero (none)

def event98132 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14614⟩⟩) 0 ⟨5503⟩ 98128

def event98133 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14614⟩⟩) (.authority (.programFamilyFact))

def exact98134RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨14614⟩⟩], []⟩, (1)⟩]

theorem exact98134RawTermsValid :
    exact98134RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event98134 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14614⟩⟩) exact98134RawTerms (.finite 28) 98133 .exactZero (none)

def event98135 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14615⟩⟩) 0 ⟨14614⟩ 98134

def event98136 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14615⟩⟩) 1 ⟨11625⟩ 98131

def event98137 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14615⟩⟩) (.product (.predecessor 0 98135 .coefficient) (.predecessor 1 98136 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event98138 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14615⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨11625⟩⟩, ⟨.program ⟨214⟩, ⟨14614⟩⟩], []⟩) [⟨.result 98134 .coefficient, true, some 1⟩, ⟨.result 98131 .coefficient, true, some 1⟩])

def event98139 : Event := .survivorFold (1) 98138

def exact98140RawTerms : List Term := []

theorem exact98140RawTermsValid :
    exact98140RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event98140 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14615⟩⟩) exact98140RawTerms (.finite 784) 98137 (.finite 784) (some (98138))

def event98141 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14616⟩⟩) 0 ⟨14615⟩ 98140

def event98142 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14616⟩⟩) (.identity (.predecessor 0 98141 .coefficient))

def event98143 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨14616⟩⟩) (.finite 784)

def event98144 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16168⟩⟩) 0 ⟨14616⟩ 98143

def event98145 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16168⟩⟩) (.authority (.programFamilyFact))

def exact98146RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16168⟩⟩], []⟩, (1)⟩]

theorem exact98146RawTermsValid :
    exact98146RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event98146 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16168⟩⟩) exact98146RawTerms (.finite 28) 98145 .exactZero (none)

def event98147 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16169⟩⟩) 0 ⟨16168⟩ 98146

def event98148 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16169⟩⟩) (.identity (.predecessor 0 98147 .coefficient))

def event98149 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨16169⟩⟩) (.finite 28)

def event98150 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21677⟩⟩) 0 ⟨16169⟩ 98149

def event98151 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21677⟩⟩) (.authority (.relationPreimageSource ⟨48⟩))

def exact98152RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨21677⟩⟩]⟩, (1)⟩]

theorem exact98152RawTermsValid :
    exact98152RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event98152 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21677⟩⟩) exact98152RawTerms (.finite 136065468) 98151 .exactZero (none)

def event98153 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6⟩⟩) (.authority (.operator))

def exact98154RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩]⟩, (1)⟩]

theorem exact98154RawTermsValid :
    exact98154RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event98154 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6⟩⟩) exact98154RawTerms .large 98153 .exactZero (none)

def event98155 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21678⟩⟩) 0 ⟨6⟩ 98154

def event98156 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21678⟩⟩) 1 ⟨21677⟩ 98152

def event98157 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21678⟩⟩) (.product (.predecessor 0 98155 .coefficient) (.predecessor 1 98156 .coefficient) (⟨false, false, none, none, none⟩))

def event98158 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21678⟩⟩, .operator (⟨98154, 0⟩, ⟨98152, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21677⟩⟩]⟩, (1)⟩)

def exact98159RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21677⟩⟩]⟩, (1)⟩]

theorem exact98159RawTermsValid :
    exact98159RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event98159 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21678⟩⟩) exact98159RawTerms .large 98157 .exactZero (none)

def event98160 : Event := .preFoldPolynomial 98159 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21677⟩⟩]⟩, (1)⟩] .exactZero none

def exact98161RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21677⟩⟩]⟩, (1)⟩]

def event98161 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨21678⟩⟩) 98160 exact98161RawTerms .large 98157 .exactZero (none)

def event98162 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨28270⟩⟩)

def event98163 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event98164 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event98165 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event98166 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event98167 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 98166

def event98168 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 98164

def event98169 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 98167 .coefficient) (.value (.predecessor 1 98168 .coefficient)))

def event98170 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event98171 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11625⟩⟩) 0 ⟨5503⟩ 98170

def event98172 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11625⟩⟩) (.authority (.programFamilyFact))

def exact98173RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11625⟩⟩], []⟩, (1)⟩]

theorem exact98173RawTermsValid :
    exact98173RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event98173 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11625⟩⟩) exact98173RawTerms (.finite 28) 98172 .exactZero (none)

def event98174 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14614⟩⟩) 0 ⟨5503⟩ 98170

def event98175 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14614⟩⟩) (.authority (.programFamilyFact))

def exact98176RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨14614⟩⟩], []⟩, (1)⟩]

theorem exact98176RawTermsValid :
    exact98176RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event98176 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14614⟩⟩) exact98176RawTerms (.finite 28) 98175 .exactZero (none)

def event98177 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14615⟩⟩) 0 ⟨14614⟩ 98176

def event98178 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14615⟩⟩) 1 ⟨11625⟩ 98173

def event98179 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14615⟩⟩) (.product (.predecessor 0 98177 .coefficient) (.predecessor 1 98178 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event98180 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨14615⟩⟩, .operator (⟨98176, 0⟩, ⟨98173, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨11625⟩⟩, ⟨.program ⟨214⟩, ⟨14614⟩⟩], []⟩, (1)⟩)

def exact98181RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11625⟩⟩, ⟨.program ⟨214⟩, ⟨14614⟩⟩], []⟩, (1)⟩]

theorem exact98181RawTermsValid :
    exact98181RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event98181 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14615⟩⟩) exact98181RawTerms (.finite 784) 98179 .exactZero (none)

def event98182 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14616⟩⟩) 0 ⟨14615⟩ 98181

def event98183 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14616⟩⟩) (.identity (.predecessor 0 98182 .coefficient))

def event98184 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨14616⟩⟩) (.finite 784)

def event98185 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16168⟩⟩) 0 ⟨14616⟩ 98184

def event98186 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16168⟩⟩) (.authority (.programFamilyFact))

def exact98187RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16168⟩⟩], []⟩, (1)⟩]

theorem exact98187RawTermsValid :
    exact98187RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event98187 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16168⟩⟩) exact98187RawTerms (.finite 28) 98186 .exactZero (none)

def event98188 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16169⟩⟩) 0 ⟨16168⟩ 98187

def event98189 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16169⟩⟩) (.identity (.predecessor 0 98188 .coefficient))

def event98190 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨16169⟩⟩) (.finite 28)

def event98191 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24277⟩⟩) 0 ⟨16169⟩ 98190

def event98192 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24277⟩⟩) (.authority (.programFamilyFact))

def event98193 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨24277⟩⟩) (.finite 3720)

def event98194 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨6689⟩⟩) .missing

def event98195 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24279⟩⟩) 0 ⟨6689⟩ 98194

def event98196 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24279⟩⟩) 1 ⟨24277⟩ 98193

def event98197 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24279⟩⟩) (.authority (.operator))

def exact98198RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨24279⟩⟩]⟩, (1)⟩]

theorem exact98198RawTermsValid :
    exact98198RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event98198 : Event := .resultExact (⟨.program ⟨214⟩, ⟨24279⟩⟩) exact98198RawTerms .large 98197 .exactZero (none)

def event98199 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28265⟩⟩) 0 ⟨24279⟩ 98198

def event98200 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28265⟩⟩) (.authority (.operator))

def exact98201RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨28265⟩⟩]⟩, (1)⟩]

theorem exact98201RawTermsValid :
    exact98201RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event98201 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28265⟩⟩) exact98201RawTerms (.finite 8192) 98200 .exactZero (none)

def event98202 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨110⟩⟩) (.authority (.operator))

def event98203 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨110⟩⟩) .exactZero

def event98204 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16210⟩⟩) 0 ⟨16169⟩ 98190

def event98205 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16210⟩⟩) 1 ⟨110⟩ 98203

def event98206 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16210⟩⟩) (.sum [.predecessor 0 98204 .coefficient, .predecessor 1 98205 .coefficient])

def event98207 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨16210⟩⟩) (.finite 28)

def event98208 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16211⟩⟩) 0 ⟨16210⟩ 98207

def event98209 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16211⟩⟩) (.identity (.predecessor 0 98208 .coefficient))

def exact98210RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16168⟩⟩], []⟩, (1)⟩]

theorem exact98210RawTermsValid :
    exact98210RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event98210 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16211⟩⟩) exact98210RawTerms (.finite 28) 98209 .exactZero (none)

def event98211 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6544⟩⟩) (.authority (.factStore))

def exact98212RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact98212RawTermsValid :
    exact98212RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event98212 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6544⟩⟩) exact98212RawTerms .large 98211 .exactZero (none)

def event98213 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16212⟩⟩) 0 ⟨6544⟩ 98212

def event98214 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16212⟩⟩) 1 ⟨16211⟩ 98210

def event98215 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16212⟩⟩) (.product (.predecessor 0 98213 .coefficient) (.predecessor 1 98214 .coefficient) (⟨false, false, none, none, none⟩))

def event98216 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨16212⟩⟩, .operator (⟨98212, 0⟩, ⟨98210, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨16168⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact98217RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16168⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact98217RawTermsValid :
    exact98217RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event98217 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16212⟩⟩) exact98217RawTerms .large 98215 .exactZero (none)

def event98218 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6699⟩⟩) 0 ⟨6689⟩ 98194

def event98219 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6699⟩⟩) (.authority (.operator))

def exact98220RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6699⟩⟩]⟩, (1)⟩]

theorem exact98220RawTermsValid :
    exact98220RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event98220 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6699⟩⟩) exact98220RawTerms .large 98219 .exactZero (none)

def event98221 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16213⟩⟩) 0 ⟨6699⟩ 98220

def event98222 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16213⟩⟩) 1 ⟨16212⟩ 98217

def event98223 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16213⟩⟩) (.sum [.predecessor 0 98221 .coefficient, .predecessor 1 98222 .coefficient])

def exact98224RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6699⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16168⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact98224RawTermsValid :
    exact98224RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event98224 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16213⟩⟩) exact98224RawTerms .large 98223 .exactZero (none)

def event98225 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28266⟩⟩) 0 ⟨16213⟩ 98224

def event98226 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28266⟩⟩) 1 ⟨28265⟩ 98201

def event98227 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28266⟩⟩) (.product (.predecessor 0 98225 .coefficient) (.predecessor 1 98226 .coefficient) (⟨false, false, none, none, none⟩))

def event98228 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28266⟩⟩, .operator (⟨98224, 0⟩, ⟨98201, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28265⟩⟩]⟩, (1)⟩)

def event98229 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28266⟩⟩, .operator (⟨98224, 1⟩, ⟨98201, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨16168⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28265⟩⟩]⟩, (-1)⟩)

def event98230 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨28266⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨16168⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28265⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨28265⟩⟩) ⟨24279⟩ 98198)

def event98231 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28266⟩⟩, .relation 98230 0, ⟨[⟨.program ⟨214⟩, ⟨16168⟩⟩], [⟨.program ⟨214⟩, ⟨24279⟩⟩]⟩, (-1)⟩)

def exact98232RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28265⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16168⟩⟩], [⟨.program ⟨214⟩, ⟨24279⟩⟩]⟩, (-1)⟩]

theorem exact98232RawTermsValid :
    exact98232RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event98232 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28266⟩⟩) exact98232RawTerms .large 98227 .exactZero (none)

def event98233 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18303⟩⟩) 0 ⟨16169⟩ 98190

def event98234 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18303⟩⟩) (.authority (.programFamilyFact))

def exact98235RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨18303⟩⟩], []⟩, (1)⟩]

theorem exact98235RawTermsValid :
    exact98235RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event98235 : Event := .resultExact (⟨.program ⟨214⟩, ⟨18303⟩⟩) exact98235RawTerms (.finite 62) 98234 .exactZero (none)

def event98236 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18314⟩⟩) 0 ⟨6544⟩ 98212

def event98237 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18314⟩⟩) 1 ⟨18303⟩ 98235

def event98238 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18314⟩⟩) (.product (.predecessor 0 98236 .coefficient) (.predecessor 1 98237 .coefficient) (⟨false, true, none, none, some 1⟩))

def event98239 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18314⟩⟩, .operator (⟨98212, 0⟩, ⟨98235, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨18303⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact98240RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨18303⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact98240RawTermsValid :
    exact98240RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event98240 : Event := .resultExact (⟨.program ⟨214⟩, ⟨18314⟩⟩) exact98240RawTerms .large 98238 .exactZero (none)

def event98241 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6727⟩⟩) 0 ⟨6689⟩ 98194

def event98242 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6727⟩⟩) (.authority (.operator))

def exact98243RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6727⟩⟩]⟩, (1)⟩]

theorem exact98243RawTermsValid :
    exact98243RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event98243 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6727⟩⟩) exact98243RawTerms .large 98242 .exactZero (none)

def event98244 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18315⟩⟩) 0 ⟨6727⟩ 98243

def event98245 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18315⟩⟩) 1 ⟨18314⟩ 98240

def event98246 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18315⟩⟩) (.sum [.predecessor 0 98244 .coefficient, .predecessor 1 98245 .coefficient])

def exact98247RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6727⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨18303⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact98247RawTermsValid :
    exact98247RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event98247 : Event := .resultExact (⟨.program ⟨214⟩, ⟨18315⟩⟩) exact98247RawTerms .large 98246 .exactZero (none)

def event98248 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28270⟩⟩) 0 ⟨18315⟩ 98247

def event98249 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28270⟩⟩) 1 ⟨28266⟩ 98232

def event98250 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28270⟩⟩) (.sum [.predecessor 0 98248 .coefficient, .predecessor 1 98249 .coefficient])

def exact98251RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28265⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6727⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16168⟩⟩], [⟨.program ⟨214⟩, ⟨24279⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨18303⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact98251RawTermsValid :
    exact98251RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event98251 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28270⟩⟩) exact98251RawTerms .large 98250 .exactZero (none)

def event98252 : Event := .preFoldPolynomial 98251 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28265⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6727⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16168⟩⟩], [⟨.program ⟨214⟩, ⟨24279⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨18303⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩] .exactZero none

def exact98253RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28265⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6727⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16168⟩⟩], [⟨.program ⟨214⟩, ⟨24279⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨18303⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

def event98253 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨28270⟩⟩) 98252 exact98253RawTerms .large 98250 .exactZero (none)

def event98254 : Event := .specializationComputed (⟨.program ⟨214⟩, ⟨16169⟩⟩) ⟨⟨140⟩, ⟨48⟩, ⟨109⟩⟩ ⟨98120, 98254⟩

def event98255 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨21680⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21677⟩⟩]⟩) (1) 0 2 (.universal 98254 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21677⟩⟩]⟩) (none) 98253)

def event98256 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21680⟩⟩, .relation 98255 1, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6727⟩⟩]⟩, (1)⟩)

def event98257 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21680⟩⟩, .relation 98255 0, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28265⟩⟩]⟩, (-1)⟩)

def event98258 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21680⟩⟩, .relation 98255 2, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16168⟩⟩], [⟨.program ⟨214⟩, ⟨24279⟩⟩]⟩, (1)⟩)

def event98259 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21680⟩⟩, .relation 98255 3, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨18303⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact98260RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28265⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6727⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16168⟩⟩], [⟨.program ⟨214⟩, ⟨24279⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨18303⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact98260RawTermsValid :
    exact98260RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event98260 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21680⟩⟩) exact98260RawTerms .large 98116 (.finite 1811303510016) (some (98118))

def event98261 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28268⟩⟩) 0 ⟨21680⟩ 98260

def event98262 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28268⟩⟩) 1 ⟨28267⟩ 98106

def event98263 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28268⟩⟩) (.sum [.predecessor 0 98261 .coefficient, .predecessor 1 98262 .coefficient])

def event98264 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28268⟩⟩, .operator (⟨98260, 0⟩, ⟨98106, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28265⟩⟩]⟩, (1)⟩)

def event98265 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28268⟩⟩, .operator (⟨98260, 2⟩, ⟨98106, 1⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16168⟩⟩], [⟨.program ⟨214⟩, ⟨24279⟩⟩]⟩, (-1)⟩)

def event98266 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28268⟩⟩) (.sum [.result 98260 .summary, .result 98106 .summary])

def exact98267RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6727⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨18303⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact98267RawTermsValid :
    exact98267RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event98267 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28268⟩⟩) exact98267RawTerms .large 98263 (.finite 1292180536164689260544) (some (98266))

def event98268 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24214⟩⟩) 0 ⟨16050⟩ 4790

def event98269 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24214⟩⟩) (.authority (.programFamilyFact))

def event98270 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨24214⟩⟩) (.finite 3720)

def event98271 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24216⟩⟩) 0 ⟨6689⟩ 5477

def event98272 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24216⟩⟩) 1 ⟨24214⟩ 98270

def event98273 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24216⟩⟩) (.authority (.operator))

def exact98274RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨24216⟩⟩]⟩, (1)⟩]

theorem exact98274RawTermsValid :
    exact98274RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event98274 : Event := .resultExact (⟨.program ⟨214⟩, ⟨24216⟩⟩) exact98274RawTerms .large 98273 .exactZero (none)

def event98275 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28048⟩⟩) 0 ⟨24216⟩ 98274

def event98276 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28048⟩⟩) (.authority (.operator))

def exact98277RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨28048⟩⟩]⟩, (1)⟩]

theorem exact98277RawTermsValid :
    exact98277RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event98277 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28048⟩⟩) exact98277RawTerms (.finite 8192) 98276 .exactZero (none)

def event98278 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23619⟩⟩) 0 ⟨14399⟩ 4784

def event98279 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23619⟩⟩) (.authority (.programFamilyFact))

def event98280 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨23619⟩⟩) (.finite 3720)

def event98281 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23620⟩⟩) 0 ⟨6689⟩ 5477

def event98282 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23620⟩⟩) 1 ⟨23619⟩ 98280

def event98283 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23620⟩⟩) (.authority (.operator))

def exact98284RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨23620⟩⟩]⟩, (1)⟩]

theorem exact98284RawTermsValid :
    exact98284RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event98284 : Event := .resultExact (⟨.program ⟨214⟩, ⟨23620⟩⟩) exact98284RawTerms .large 98283 .exactZero (none)

def event98285 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26130⟩⟩) 0 ⟨23620⟩ 98284

def event98286 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26130⟩⟩) (.authority (.operator))

def exact98287RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨26130⟩⟩]⟩, (1)⟩]

theorem exact98287RawTermsValid :
    exact98287RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event98287 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26130⟩⟩) exact98287RawTerms (.finite 8192) 98286 .exactZero (none)

def event98288 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11542⟩⟩) 0 ⟨11541⟩ 4773

def event98289 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11542⟩⟩) 1 ⟨6564⟩ 32

def event98290 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11542⟩⟩) (.tensor (.predecessor 0 98288 .coefficient) (.predecessor 1 98289 .coefficient) true false)

def event98291 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨11542⟩⟩, .operator (⟨4773, 0⟩, ⟨32, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨11541⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact98292RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨11541⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact98292RawTermsValid :
    exact98292RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event98292 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11542⟩⟩) exact98292RawTerms .large 98290 .exactZero (none)

def event98293 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7117⟩⟩) 0 ⟨5506⟩ 27

def event98294 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7117⟩⟩) 1 ⟨6780⟩ 10981

def event98295 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7117⟩⟩) (.product (.predecessor 0 98293 .coefficient) (.predecessor 1 98294 .coefficient) (⟨false, false, none, none, none⟩))

def event98296 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7117⟩⟩, .operator (⟨27, 0⟩, ⟨10981, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6780⟩⟩]⟩, (1)⟩)

def exact98297RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6780⟩⟩]⟩, (1)⟩]

theorem exact98297RawTermsValid :
    exact98297RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event98297 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7117⟩⟩) exact98297RawTerms .large 98295 .exactZero (none)

def event98298 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11543⟩⟩) 0 ⟨7117⟩ 98297

def event98299 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11543⟩⟩) 1 ⟨11542⟩ 98292

def event98300 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11543⟩⟩) (.sum [.predecessor 0 98298 .coefficient, .predecessor 1 98299 .coefficient])

def exact98301RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6780⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨11541⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact98301RawTermsValid :
    exact98301RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event98301 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11543⟩⟩) exact98301RawTerms .large 98300 .exactZero (none)

def event98302 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11544⟩⟩) 0 ⟨11543⟩ 98301

def event98303 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11544⟩⟩) 1 ⟨94⟩ 10973

def eventLeaf6128 : Array AnnotatedEvent := #[
  { event := event98048
    frameStart := 97977 },
  { event := event98049
    frameStart := 97977 },
  { event := event98050
    frameStart := 97977 },
  { event := event98051
    frameStart := 97977 },
  { event := event98052
    frameStart := 97977 },
  { event := event98053
    frameStart := 97977 },
  { event := event98054
    frameStart := 97977 },
  { event := event98055
    frameStart := 97977 },
  { event := event98056
    frameStart := 97977 },
  { event := event98057
    frameStart := 97977 },
  { event := event98058
    frameStart := 97977 },
  { event := event98059
    frameStart := 97977 },
  { event := event98060
    frameStart := 97977 },
  { event := event98061
    frameStart := 97977 },
  { event := event98062
    frameStart := 97977 },
  { event := event98063
    frameStart := 97977 }
]

def eventLeaf6129 : Array AnnotatedEvent := #[
  { event := event98064
    frameStart := 97977 },
  { event := event98065
    frameStart := 97977 },
  { event := event98066
    frameStart := 97977 },
  { event := event98067
    frameStart := 97977 },
  { event := event98068
    frameStart := 97977 },
  { event := event98069
    frameStart := 97977 },
  { event := event98070
    frameStart := 97977 },
  { event := event98071
    frameStart := 97977 },
  { event := event98072
    frameStart := 97977 },
  { event := event98073
    frameStart := 97977 },
  { event := event98074
    frameStart := 97977 },
  { event := event98075
    frameStart := 97977 },
  { event := event98076
    frameStart := 97977 },
  { event := event98077
    frameStart := 97977 },
  { event := event98078
    frameStart := 97977 },
  { event := event98079
    frameStart := 97977 }
]

def eventLeaf6130 : Array AnnotatedEvent := #[
  { event := event98080
    frameStart := 97977 },
  { event := event98081
    frameStart := 97977 },
  { event := event98082
    frameStart := 97977 },
  { event := event98083
    frameStart := 0 },
  { event := event98084
    frameStart := 0 },
  { event := event98085
    frameStart := 0 },
  { event := event98086
    frameStart := 0 },
  { event := event98087
    frameStart := 0 },
  { event := event98088
    frameStart := 0 },
  { event := event98089
    frameStart := 0 },
  { event := event98090
    frameStart := 0 },
  { event := event98091
    frameStart := 0 },
  { event := event98092
    frameStart := 0 },
  { event := event98093
    frameStart := 0 },
  { event := event98094
    frameStart := 0 },
  { event := event98095
    frameStart := 0 }
]

def eventLeaf6131 : Array AnnotatedEvent := #[
  { event := event98096
    frameStart := 0 },
  { event := event98097
    frameStart := 0 },
  { event := event98098
    frameStart := 0 },
  { event := event98099
    frameStart := 0 },
  { event := event98100
    frameStart := 0 },
  { event := event98101
    frameStart := 0 },
  { event := event98102
    frameStart := 0 },
  { event := event98103
    frameStart := 0 },
  { event := event98104
    frameStart := 0 },
  { event := event98105
    frameStart := 0 },
  { event := event98106
    frameStart := 0 },
  { event := event98107
    frameStart := 0 },
  { event := event98108
    frameStart := 0 },
  { event := event98109
    frameStart := 0 },
  { event := event98110
    frameStart := 0 },
  { event := event98111
    frameStart := 0 }
]

def eventLeaf6132 : Array AnnotatedEvent := #[
  { event := event98112
    frameStart := 0 },
  { event := event98113
    frameStart := 0 },
  { event := event98114
    frameStart := 0 },
  { event := event98115
    frameStart := 0 },
  { event := event98116
    frameStart := 0 },
  { event := event98117
    frameStart := 0 },
  { event := event98118
    frameStart := 0 },
  { event := event98119
    frameStart := 0 },
  { event := event98120
    frameStart := 98120 },
  { event := event98121
    frameStart := 98120 },
  { event := event98122
    frameStart := 98120 },
  { event := event98123
    frameStart := 98120 },
  { event := event98124
    frameStart := 98120 },
  { event := event98125
    frameStart := 98120 },
  { event := event98126
    frameStart := 98120 },
  { event := event98127
    frameStart := 98120 }
]

def eventLeaf6133 : Array AnnotatedEvent := #[
  { event := event98128
    frameStart := 98120 },
  { event := event98129
    frameStart := 98120 },
  { event := event98130
    frameStart := 98120 },
  { event := event98131
    frameStart := 98120 },
  { event := event98132
    frameStart := 98120 },
  { event := event98133
    frameStart := 98120 },
  { event := event98134
    frameStart := 98120 },
  { event := event98135
    frameStart := 98120 },
  { event := event98136
    frameStart := 98120 },
  { event := event98137
    frameStart := 98120 },
  { event := event98138
    frameStart := 98120 },
  { event := event98139
    frameStart := 98120 },
  { event := event98140
    frameStart := 98120 },
  { event := event98141
    frameStart := 98120 },
  { event := event98142
    frameStart := 98120 },
  { event := event98143
    frameStart := 98120 }
]

def eventLeaf6134 : Array AnnotatedEvent := #[
  { event := event98144
    frameStart := 98120 },
  { event := event98145
    frameStart := 98120 },
  { event := event98146
    frameStart := 98120 },
  { event := event98147
    frameStart := 98120 },
  { event := event98148
    frameStart := 98120 },
  { event := event98149
    frameStart := 98120 },
  { event := event98150
    frameStart := 98120 },
  { event := event98151
    frameStart := 98120 },
  { event := event98152
    frameStart := 98120 },
  { event := event98153
    frameStart := 98120 },
  { event := event98154
    frameStart := 98120 },
  { event := event98155
    frameStart := 98120 },
  { event := event98156
    frameStart := 98120 },
  { event := event98157
    frameStart := 98120 },
  { event := event98158
    frameStart := 98120 },
  { event := event98159
    frameStart := 98120 }
]

def eventLeaf6135 : Array AnnotatedEvent := #[
  { event := event98160
    frameStart := 98120 },
  { event := event98161
    frameStart := 98120 },
  { event := event98162
    frameStart := 98162 },
  { event := event98163
    frameStart := 98162 },
  { event := event98164
    frameStart := 98162 },
  { event := event98165
    frameStart := 98162 },
  { event := event98166
    frameStart := 98162 },
  { event := event98167
    frameStart := 98162 },
  { event := event98168
    frameStart := 98162 },
  { event := event98169
    frameStart := 98162 },
  { event := event98170
    frameStart := 98162 },
  { event := event98171
    frameStart := 98162 },
  { event := event98172
    frameStart := 98162 },
  { event := event98173
    frameStart := 98162 },
  { event := event98174
    frameStart := 98162 },
  { event := event98175
    frameStart := 98162 }
]

def eventLeaf6136 : Array AnnotatedEvent := #[
  { event := event98176
    frameStart := 98162 },
  { event := event98177
    frameStart := 98162 },
  { event := event98178
    frameStart := 98162 },
  { event := event98179
    frameStart := 98162 },
  { event := event98180
    frameStart := 98162 },
  { event := event98181
    frameStart := 98162 },
  { event := event98182
    frameStart := 98162 },
  { event := event98183
    frameStart := 98162 },
  { event := event98184
    frameStart := 98162 },
  { event := event98185
    frameStart := 98162 },
  { event := event98186
    frameStart := 98162 },
  { event := event98187
    frameStart := 98162 },
  { event := event98188
    frameStart := 98162 },
  { event := event98189
    frameStart := 98162 },
  { event := event98190
    frameStart := 98162 },
  { event := event98191
    frameStart := 98162 }
]

def eventLeaf6137 : Array AnnotatedEvent := #[
  { event := event98192
    frameStart := 98162 },
  { event := event98193
    frameStart := 98162 },
  { event := event98194
    frameStart := 98162 },
  { event := event98195
    frameStart := 98162 },
  { event := event98196
    frameStart := 98162 },
  { event := event98197
    frameStart := 98162 },
  { event := event98198
    frameStart := 98162 },
  { event := event98199
    frameStart := 98162 },
  { event := event98200
    frameStart := 98162 },
  { event := event98201
    frameStart := 98162 },
  { event := event98202
    frameStart := 98162 },
  { event := event98203
    frameStart := 98162 },
  { event := event98204
    frameStart := 98162 },
  { event := event98205
    frameStart := 98162 },
  { event := event98206
    frameStart := 98162 },
  { event := event98207
    frameStart := 98162 }
]

def eventLeaf6138 : Array AnnotatedEvent := #[
  { event := event98208
    frameStart := 98162 },
  { event := event98209
    frameStart := 98162 },
  { event := event98210
    frameStart := 98162 },
  { event := event98211
    frameStart := 98162 },
  { event := event98212
    frameStart := 98162 },
  { event := event98213
    frameStart := 98162 },
  { event := event98214
    frameStart := 98162 },
  { event := event98215
    frameStart := 98162 },
  { event := event98216
    frameStart := 98162 },
  { event := event98217
    frameStart := 98162 },
  { event := event98218
    frameStart := 98162 },
  { event := event98219
    frameStart := 98162 },
  { event := event98220
    frameStart := 98162 },
  { event := event98221
    frameStart := 98162 },
  { event := event98222
    frameStart := 98162 },
  { event := event98223
    frameStart := 98162 }
]

def eventLeaf6139 : Array AnnotatedEvent := #[
  { event := event98224
    frameStart := 98162 },
  { event := event98225
    frameStart := 98162 },
  { event := event98226
    frameStart := 98162 },
  { event := event98227
    frameStart := 98162 },
  { event := event98228
    frameStart := 98162 },
  { event := event98229
    frameStart := 98162 },
  { event := event98230
    frameStart := 98162 },
  { event := event98231
    frameStart := 98162 },
  { event := event98232
    frameStart := 98162 },
  { event := event98233
    frameStart := 98162 },
  { event := event98234
    frameStart := 98162 },
  { event := event98235
    frameStart := 98162 },
  { event := event98236
    frameStart := 98162 },
  { event := event98237
    frameStart := 98162 },
  { event := event98238
    frameStart := 98162 },
  { event := event98239
    frameStart := 98162 }
]

def eventLeaf6140 : Array AnnotatedEvent := #[
  { event := event98240
    frameStart := 98162 },
  { event := event98241
    frameStart := 98162 },
  { event := event98242
    frameStart := 98162 },
  { event := event98243
    frameStart := 98162 },
  { event := event98244
    frameStart := 98162 },
  { event := event98245
    frameStart := 98162 },
  { event := event98246
    frameStart := 98162 },
  { event := event98247
    frameStart := 98162 },
  { event := event98248
    frameStart := 98162 },
  { event := event98249
    frameStart := 98162 },
  { event := event98250
    frameStart := 98162 },
  { event := event98251
    frameStart := 98162 },
  { event := event98252
    frameStart := 98162 },
  { event := event98253
    frameStart := 98162 },
  { event := event98254
    frameStart := 0 },
  { event := event98255
    frameStart := 0 }
]

def eventLeaf6141 : Array AnnotatedEvent := #[
  { event := event98256
    frameStart := 0 },
  { event := event98257
    frameStart := 0 },
  { event := event98258
    frameStart := 0 },
  { event := event98259
    frameStart := 0 },
  { event := event98260
    frameStart := 0 },
  { event := event98261
    frameStart := 0 },
  { event := event98262
    frameStart := 0 },
  { event := event98263
    frameStart := 0 },
  { event := event98264
    frameStart := 0 },
  { event := event98265
    frameStart := 0 },
  { event := event98266
    frameStart := 0 },
  { event := event98267
    frameStart := 0 },
  { event := event98268
    frameStart := 0 },
  { event := event98269
    frameStart := 0 },
  { event := event98270
    frameStart := 0 },
  { event := event98271
    frameStart := 0 }
]

def eventLeaf6142 : Array AnnotatedEvent := #[
  { event := event98272
    frameStart := 0 },
  { event := event98273
    frameStart := 0 },
  { event := event98274
    frameStart := 0 },
  { event := event98275
    frameStart := 0 },
  { event := event98276
    frameStart := 0 },
  { event := event98277
    frameStart := 0 },
  { event := event98278
    frameStart := 0 },
  { event := event98279
    frameStart := 0 },
  { event := event98280
    frameStart := 0 },
  { event := event98281
    frameStart := 0 },
  { event := event98282
    frameStart := 0 },
  { event := event98283
    frameStart := 0 },
  { event := event98284
    frameStart := 0 },
  { event := event98285
    frameStart := 0 },
  { event := event98286
    frameStart := 0 },
  { event := event98287
    frameStart := 0 }
]

def eventLeaf6143 : Array AnnotatedEvent := #[
  { event := event98288
    frameStart := 0 },
  { event := event98289
    frameStart := 0 },
  { event := event98290
    frameStart := 0 },
  { event := event98291
    frameStart := 0 },
  { event := event98292
    frameStart := 0 },
  { event := event98293
    frameStart := 0 },
  { event := event98294
    frameStart := 0 },
  { event := event98295
    frameStart := 0 },
  { event := event98296
    frameStart := 0 },
  { event := event98297
    frameStart := 0 },
  { event := event98298
    frameStart := 0 },
  { event := event98299
    frameStart := 0 },
  { event := event98300
    frameStart := 0 },
  { event := event98301
    frameStart := 0 },
  { event := event98302
    frameStart := 0 },
  { event := event98303
    frameStart := 0 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events383
