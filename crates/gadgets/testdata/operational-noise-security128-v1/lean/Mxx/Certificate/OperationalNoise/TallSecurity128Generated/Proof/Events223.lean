import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events223

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event57088 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event57089 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49538⟩⟩) 0 ⟨48213⟩ 57075

def event57090 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49538⟩⟩) 1 ⟨136⟩ 57088

def event57091 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49538⟩⟩) (.sum [.predecessor 0 57089 .coefficient, .predecessor 1 57090 .coefficient])

def event57092 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨49538⟩⟩) (.finite 60)

def event57093 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49539⟩⟩) 0 ⟨49538⟩ 57092

def event57094 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49539⟩⟩) (.identity (.predecessor 0 57093 .coefficient))

def exact57095RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨48212⟩⟩], []⟩, (1)⟩]

theorem exact57095RawTermsValid :
    exact57095RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event57095 : Event := .resultExact (⟨.program ⟨257⟩, ⟨49539⟩⟩) exact57095RawTerms (.finite 60) 57094 .exactZero (none)

def event57096 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact57097RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact57097RawTermsValid :
    exact57097RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event57097 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact57097RawTerms .large 57096 .exactZero (none)

def event57098 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49540⟩⟩) 0 ⟨6908⟩ 57097

def event57099 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49540⟩⟩) 1 ⟨49539⟩ 57095

def event57100 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49540⟩⟩) (.product (.predecessor 0 57098 .coefficient) (.predecessor 1 57099 .coefficient) (⟨false, false, none, none, none⟩))

def event57101 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨49540⟩⟩, .operator (⟨57097, 0⟩, ⟨57095, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨48212⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact57102RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨48212⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact57102RawTermsValid :
    exact57102RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event57102 : Event := .resultExact (⟨.program ⟨257⟩, ⟨49540⟩⟩) exact57102RawTerms .large 57100 .exactZero (none)

def event57103 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7196⟩⟩) 0 ⟨7177⟩ 57079

def event57104 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7196⟩⟩) (.authority (.operator))

def exact57105RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7196⟩⟩]⟩, (1)⟩]

theorem exact57105RawTermsValid :
    exact57105RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event57105 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7196⟩⟩) exact57105RawTerms .large 57104 .exactZero (none)

def event57106 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49541⟩⟩) 0 ⟨7196⟩ 57105

def event57107 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49541⟩⟩) 1 ⟨49540⟩ 57102

def event57108 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49541⟩⟩) (.sum [.predecessor 0 57106 .coefficient, .predecessor 1 57107 .coefficient])

def exact57109RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7196⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨48212⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact57109RawTermsValid :
    exact57109RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event57109 : Event := .resultExact (⟨.program ⟨257⟩, ⟨49541⟩⟩) exact57109RawTerms .large 57108 .exactZero (none)

def event57110 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50224⟩⟩) 0 ⟨49541⟩ 57109

def event57111 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50224⟩⟩) 1 ⟨50223⟩ 57086

def event57112 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50224⟩⟩) (.product (.predecessor 0 57110 .coefficient) (.predecessor 1 57111 .coefficient) (⟨false, false, none, none, none⟩))

def event57113 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨50224⟩⟩, .operator (⟨57109, 0⟩, ⟨57086, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨50223⟩⟩]⟩, (1)⟩)

def event57114 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨50224⟩⟩, .operator (⟨57109, 1⟩, ⟨57086, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨48212⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨50223⟩⟩]⟩, (-1)⟩)

def event57115 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨50224⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨48212⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨50223⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨50223⟩⟩) ⟨49372⟩ 57083)

def event57116 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨50224⟩⟩, .relation 57115 0, ⟨[⟨.program ⟨257⟩, ⟨48212⟩⟩], [⟨.program ⟨257⟩, ⟨49372⟩⟩]⟩, (-1)⟩)

def exact57117RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨50223⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨48212⟩⟩], [⟨.program ⟨257⟩, ⟨49372⟩⟩]⟩, (-1)⟩]

theorem exact57117RawTermsValid :
    exact57117RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event57117 : Event := .resultExact (⟨.program ⟨257⟩, ⟨50224⟩⟩) exact57117RawTerms .large 57112 .exactZero (none)

def event57118 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48463⟩⟩) 0 ⟨48213⟩ 57075

def event57119 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48463⟩⟩) (.authority (.programFamilyFact))

def exact57120RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨48463⟩⟩], []⟩, (1)⟩]

theorem exact57120RawTermsValid :
    exact57120RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event57120 : Event := .resultExact (⟨.program ⟨257⟩, ⟨48463⟩⟩) exact57120RawTerms (.finite 60) 57119 .exactZero (none)

def event57121 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48465⟩⟩) 0 ⟨6908⟩ 57097

def event57122 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48465⟩⟩) 1 ⟨48463⟩ 57120

def event57123 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48465⟩⟩) (.product (.predecessor 0 57121 .coefficient) (.predecessor 1 57122 .coefficient) (⟨false, true, none, none, some 1⟩))

def event57124 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨48465⟩⟩, .operator (⟨57097, 0⟩, ⟨57120, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨48463⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact57125RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨48463⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact57125RawTermsValid :
    exact57125RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event57125 : Event := .resultExact (⟨.program ⟨257⟩, ⟨48465⟩⟩) exact57125RawTerms .large 57123 .exactZero (none)

def event57126 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7231⟩⟩) 0 ⟨7177⟩ 57079

def event57127 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7231⟩⟩) (.authority (.operator))

def exact57128RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7231⟩⟩]⟩, (1)⟩]

theorem exact57128RawTermsValid :
    exact57128RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event57128 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7231⟩⟩) exact57128RawTerms .large 57127 .exactZero (none)

def event57129 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48466⟩⟩) 0 ⟨7231⟩ 57128

def event57130 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48466⟩⟩) 1 ⟨48465⟩ 57125

def event57131 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48466⟩⟩) (.sum [.predecessor 0 57129 .coefficient, .predecessor 1 57130 .coefficient])

def exact57132RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7231⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨48463⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact57132RawTermsValid :
    exact57132RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event57132 : Event := .resultExact (⟨.program ⟨257⟩, ⟨48466⟩⟩) exact57132RawTerms .large 57131 .exactZero (none)

def event57133 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50228⟩⟩) 0 ⟨48466⟩ 57132

def event57134 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50228⟩⟩) 1 ⟨50224⟩ 57117

def event57135 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50228⟩⟩) (.sum [.predecessor 0 57133 .coefficient, .predecessor 1 57134 .coefficient])

def exact57136RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨50223⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7231⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨48212⟩⟩], [⟨.program ⟨257⟩, ⟨49372⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨48463⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact57136RawTermsValid :
    exact57136RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event57136 : Event := .resultExact (⟨.program ⟨257⟩, ⟨50228⟩⟩) exact57136RawTerms .large 57135 .exactZero (none)

def event57137 : Event := .preFoldPolynomial 57136 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨50223⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7231⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨48212⟩⟩], [⟨.program ⟨257⟩, ⟨49372⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨48463⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact57138RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨50223⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7231⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨48212⟩⟩], [⟨.program ⟨257⟩, ⟨49372⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨48463⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event57138 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨50228⟩⟩) 57137 exact57138RawTerms .large 57135 .exactZero (none)

def event57139 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨48213⟩⟩) ⟨⟨110⟩, ⟨93⟩, ⟨135⟩⟩ ⟨56981, 57139⟩

def event57140 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨49055⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨49052⟩⟩]⟩) (1) 0 2 (.universal 57139 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨49052⟩⟩]⟩) (none) 57138)

def event57141 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨49055⟩⟩, .relation 57140 1, ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7231⟩⟩]⟩, (1)⟩)

def event57142 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨49055⟩⟩, .relation 57140 0, ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨50223⟩⟩]⟩, (-1)⟩)

def event57143 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨49055⟩⟩, .relation 57140 2, ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨48212⟩⟩], [⟨.program ⟨257⟩, ⟨49372⟩⟩]⟩, (1)⟩)

def event57144 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨49055⟩⟩, .relation 57140 3, ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨48463⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact57145RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨50223⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7231⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨48212⟩⟩], [⟨.program ⟨257⟩, ⟨49372⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨48463⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact57145RawTermsValid :
    exact57145RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event57145 : Event := .resultExact (⟨.program ⟨257⟩, ⟨49055⟩⟩) exact57145RawTerms .large 56977 (.finite 202072841853861888) (some (56979))

def event57146 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50226⟩⟩) 0 ⟨49055⟩ 57145

def event57147 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50226⟩⟩) 1 ⟨50225⟩ 56967

def event57148 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50226⟩⟩) (.sum [.predecessor 0 57146 .coefficient, .predecessor 1 57147 .coefficient])

def event57149 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨50226⟩⟩, .operator (⟨57145, 0⟩, ⟨56967, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨50223⟩⟩]⟩, (1)⟩)

def event57150 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨50226⟩⟩, .operator (⟨57145, 2⟩, ⟨56967, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨48212⟩⟩], [⟨.program ⟨257⟩, ⟨49372⟩⟩]⟩, (-1)⟩)

def event57151 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50226⟩⟩) (.sum [.result 57145 .summary, .result 56967 .summary])

def exact57152RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7231⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨48463⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact57152RawTermsValid :
    exact57152RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event57152 : Event := .resultExact (⟨.program ⟨257⟩, ⟨50226⟩⟩) exact57152RawTerms .large 57148 (.finite 32194504275408640829496428331008) (some (57151))

def event57153 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50227⟩⟩) 0 ⟨50226⟩ 57152

def event57154 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50227⟩⟩) 1 ⟨7148⟩ 15542

def event57155 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50227⟩⟩) (.product (.predecessor 0 57153 .coefficient) (.predecessor 1 57154 .coefficient) (⟨false, false, none, none, none⟩))

def event57156 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50227⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨7147⟩⟩]⟩) [⟨.result 15538 .coefficient, false, none⟩])

def event57157 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50227⟩⟩) (.product (.result 57152 .summary) (.transfer 57156) (⟨false, false, none, none, none⟩))

def event57158 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨50227⟩⟩, .operator (⟨57152, 0⟩, ⟨15542, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7231⟩⟩, ⟨.program ⟨257⟩, ⟨7147⟩⟩]⟩, (1)⟩)

def event57159 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨50227⟩⟩, .operator (⟨57152, 1⟩, ⟨15542, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨48463⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7147⟩⟩]⟩, (-1)⟩)

def event57160 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨50227⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨48463⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7147⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨7147⟩⟩) ⟨7039⟩ 15535)

def event57161 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨50227⟩⟩, .relation 57160 0, ⟨[⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨48463⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact57162RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨48463⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7231⟩⟩, ⟨.program ⟨257⟩, ⟨7147⟩⟩]⟩, (1)⟩]

theorem exact57162RawTermsValid :
    exact57162RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event57162 : Event := .resultExact (⟨.program ⟨257⟩, ⟨50227⟩⟩) exact57162RawTerms .large 57155 (.finite 345685857434530723496243679576218056785920) (some (57157))

def event57163 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46692⟩⟩) 0 ⟨7177⟩ 15500

def event57164 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46692⟩⟩) 1 ⟨46691⟩ 47129

def event57165 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46692⟩⟩) (.authority (.operator))

def exact57166RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨46692⟩⟩]⟩, (1)⟩]

theorem exact57166RawTermsValid :
    exact57166RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event57166 : Event := .resultExact (⟨.program ⟨257⟩, ⟨46692⟩⟩) exact57166RawTerms .large 57165 .exactZero (none)

def event57167 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47543⟩⟩) 0 ⟨46692⟩ 57166

def event57168 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47543⟩⟩) (.authority (.operator))

def exact57169RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨47543⟩⟩]⟩, (1)⟩]

theorem exact57169RawTermsValid :
    exact57169RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event57169 : Event := .resultExact (⟨.program ⟨257⟩, ⟨47543⟩⟩) exact57169RawTerms (.finite 8192) 57168 .exactZero (none)

def event57170 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47545⟩⟩) 0 ⟨47069⟩ 47413

def event57171 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47545⟩⟩) 1 ⟨47543⟩ 57169

def event57172 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47545⟩⟩) (.product (.predecessor 0 57170 .coefficient) (.predecessor 1 57171 .coefficient) (⟨false, false, none, none, none⟩))

def event57173 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47545⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨47543⟩⟩]⟩) [⟨.result 57169 .coefficient, false, none⟩])

def event57174 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47545⟩⟩) (.product (.result 47413 .summary) (.transfer 57173) (⟨false, false, none, none, none⟩))

def event57175 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨47545⟩⟩, .operator (⟨47413, 0⟩, ⟨57169, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47543⟩⟩]⟩, (1)⟩)

def event57176 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨47545⟩⟩, .operator (⟨47413, 1⟩, ⟨57169, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨45532⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨47543⟩⟩]⟩, (-1)⟩)

def event57177 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨47545⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨45532⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨47543⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨47543⟩⟩) ⟨46692⟩ 57166)

def event57178 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨47545⟩⟩, .relation 57177 0, ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨45532⟩⟩], [⟨.program ⟨257⟩, ⟨46692⟩⟩]⟩, (-1)⟩)

def exact57179RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47543⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨45532⟩⟩], [⟨.program ⟨257⟩, ⟨46692⟩⟩]⟩, (-1)⟩]

theorem exact57179RawTermsValid :
    exact57179RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event57179 : Event := .resultExact (⟨.program ⟨257⟩, ⟨47545⟩⟩) exact57179RawTerms .large 57172 (.finite 32194307824962751379413684715520) (some (57174))

def event57180 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46372⟩⟩) 0 ⟨45533⟩ 1630

def event57181 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46372⟩⟩) (.authority (.relationPreimageSource ⟨91⟩))

def exact57182RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨46372⟩⟩]⟩, (1)⟩]

theorem exact57182RawTermsValid :
    exact57182RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event57182 : Event := .resultExact (⟨.program ⟨257⟩, ⟨46372⟩⟩) exact57182RawTerms (.finite 5647228698) 57181 .exactZero (none)

def event57183 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46374⟩⟩) 0 ⟨46372⟩ 57182

def event57184 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46374⟩⟩) 1 ⟨2370⟩ 4

def event57185 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46374⟩⟩) (.scale (.predecessor 0 57183 .coefficient) (.value (.predecessor 1 57184 .coefficient)))

def exact57186RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨46372⟩⟩]⟩, (1)⟩]

theorem exact57186RawTermsValid :
    exact57186RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event57186 : Event := .resultExact (⟨.program ⟨257⟩, ⟨46374⟩⟩) exact57186RawTerms (.finite 5647228698) 57185 .exactZero (none)

def event57187 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46375⟩⟩) 0 ⟨11216⟩ 46745

def event57188 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46375⟩⟩) 1 ⟨46374⟩ 57186

def event57189 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46375⟩⟩) (.product (.predecessor 0 57187 .coefficient) (.predecessor 1 57188 .coefficient) (⟨false, false, none, none, none⟩))

def event57190 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46375⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨46372⟩⟩]⟩) [⟨.result 57182 .coefficient, false, none⟩])

def event57191 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46375⟩⟩) (.product (.result 46745 .summary) (.transfer 57190) (⟨false, false, none, none, none⟩))

def event57192 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨46375⟩⟩, .operator (⟨46745, 0⟩, ⟨57186, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46372⟩⟩]⟩, (1)⟩)

def event57193 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨46373⟩⟩)

def event57194 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event57195 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event57196 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨11115⟩⟩) (.authority (.operator))

def event57197 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨11115⟩⟩) (.finite 17)

def event57198 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event57199 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event57200 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event57201 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event57202 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 57201

def event57203 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 57199

def event57204 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 57202 .coefficient) (.value (.predecessor 1 57203 .coefficient)))

def event57205 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event57206 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11117⟩⟩) 0 ⟨392⟩ 57205

def event57207 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11117⟩⟩) 1 ⟨11115⟩ 57197

def event57208 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨11117⟩⟩) (.sum [.predecessor 0 57206 .coefficient, .predecessor 1 57207 .coefficient])

def event57209 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨11117⟩⟩) (.finite 655357)

def event57210 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11173⟩⟩) 0 ⟨11117⟩ 57209

def event57211 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11173⟩⟩) 1 ⟨5426⟩ 57195

def event57212 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨11173⟩⟩) (.identity (.predecessor 1 57211 .coefficient))

def event57213 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨11173⟩⟩) (.finite 655360)

def event57214 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45346⟩⟩) 0 ⟨11173⟩ 57213

def event57215 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45346⟩⟩) (.authority (.programFamilyFact))

def exact57216RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨45346⟩⟩], []⟩, (1)⟩]

theorem exact57216RawTermsValid :
    exact57216RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event57216 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45346⟩⟩) exact57216RawTerms (.finite 58) 57215 .exactZero (none)

def event57217 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14901⟩⟩) 0 ⟨11173⟩ 57213

def event57218 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14901⟩⟩) (.authority (.programFamilyFact))

def exact57219RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14901⟩⟩], []⟩, (1)⟩]

theorem exact57219RawTermsValid :
    exact57219RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event57219 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14901⟩⟩) exact57219RawTerms (.finite 58) 57218 .exactZero (none)

def event57220 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45347⟩⟩) 0 ⟨14901⟩ 57219

def event57221 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45347⟩⟩) 1 ⟨45346⟩ 57216

def event57222 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45347⟩⟩) (.product (.predecessor 0 57220 .coefficient) (.predecessor 1 57221 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event57223 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45347⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨14901⟩⟩, ⟨.program ⟨257⟩, ⟨45346⟩⟩], []⟩) [⟨.result 57219 .coefficient, true, some 1⟩, ⟨.result 57216 .coefficient, true, some 1⟩])

def event57224 : Event := .survivorFold (1) 57223

def exact57225RawTerms : List Term := []

theorem exact57225RawTermsValid :
    exact57225RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event57225 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45347⟩⟩) exact57225RawTerms (.finite 3364) 57222 (.finite 3364) (some (57223))

def event57226 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45348⟩⟩) 0 ⟨45347⟩ 57225

def event57227 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45348⟩⟩) (.identity (.predecessor 0 57226 .coefficient))

def event57228 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨45348⟩⟩) (.finite 3364)

def event57229 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45532⟩⟩) 0 ⟨45348⟩ 57228

def event57230 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45532⟩⟩) (.authority (.programFamilyFact))

def exact57231RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨45532⟩⟩], []⟩, (1)⟩]

theorem exact57231RawTermsValid :
    exact57231RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event57231 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45532⟩⟩) exact57231RawTerms (.finite 58) 57230 .exactZero (none)

def event57232 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45533⟩⟩) 0 ⟨45532⟩ 57231

def event57233 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45533⟩⟩) (.identity (.predecessor 0 57232 .coefficient))

def event57234 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨45533⟩⟩) (.finite 58)

def event57235 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46372⟩⟩) 0 ⟨45533⟩ 57234

def event57236 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46372⟩⟩) (.authority (.relationPreimageSource ⟨91⟩))

def exact57237RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨46372⟩⟩]⟩, (1)⟩]

theorem exact57237RawTermsValid :
    exact57237RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event57237 : Event := .resultExact (⟨.program ⟨257⟩, ⟨46372⟩⟩) exact57237RawTerms (.finite 5647228698) 57236 .exactZero (none)

def event57238 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35⟩⟩) (.authority (.operator))

def exact57239RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact57239RawTermsValid :
    exact57239RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event57239 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35⟩⟩) exact57239RawTerms .large 57238 .exactZero (none)

def event57240 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46373⟩⟩) 0 ⟨35⟩ 57239

def event57241 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46373⟩⟩) 1 ⟨46372⟩ 57237

def event57242 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46373⟩⟩) (.product (.predecessor 0 57240 .coefficient) (.predecessor 1 57241 .coefficient) (⟨false, false, none, none, none⟩))

def event57243 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨46373⟩⟩, .operator (⟨57239, 0⟩, ⟨57237, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46372⟩⟩]⟩, (1)⟩)

def exact57244RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46372⟩⟩]⟩, (1)⟩]

theorem exact57244RawTermsValid :
    exact57244RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event57244 : Event := .resultExact (⟨.program ⟨257⟩, ⟨46373⟩⟩) exact57244RawTerms .large 57242 .exactZero (none)

def event57245 : Event := .preFoldPolynomial 57244 [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46372⟩⟩]⟩, (1)⟩] .exactZero none

def exact57246RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46372⟩⟩]⟩, (1)⟩]

def event57246 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨46373⟩⟩) 57245 exact57246RawTerms .large 57242 .exactZero (none)

def event57247 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨47548⟩⟩)

def event57248 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event57249 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event57250 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨11115⟩⟩) (.authority (.operator))

def event57251 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨11115⟩⟩) (.finite 17)

def event57252 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event57253 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event57254 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event57255 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event57256 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 57255

def event57257 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 57253

def event57258 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 57256 .coefficient) (.value (.predecessor 1 57257 .coefficient)))

def event57259 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event57260 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11117⟩⟩) 0 ⟨392⟩ 57259

def event57261 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11117⟩⟩) 1 ⟨11115⟩ 57251

def event57262 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨11117⟩⟩) (.sum [.predecessor 0 57260 .coefficient, .predecessor 1 57261 .coefficient])

def event57263 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨11117⟩⟩) (.finite 655357)

def event57264 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11173⟩⟩) 0 ⟨11117⟩ 57263

def event57265 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11173⟩⟩) 1 ⟨5426⟩ 57249

def event57266 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨11173⟩⟩) (.identity (.predecessor 1 57265 .coefficient))

def event57267 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨11173⟩⟩) (.finite 655360)

def event57268 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45346⟩⟩) 0 ⟨11173⟩ 57267

def event57269 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45346⟩⟩) (.authority (.programFamilyFact))

def exact57270RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨45346⟩⟩], []⟩, (1)⟩]

theorem exact57270RawTermsValid :
    exact57270RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event57270 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45346⟩⟩) exact57270RawTerms (.finite 58) 57269 .exactZero (none)

def event57271 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14901⟩⟩) 0 ⟨11173⟩ 57267

def event57272 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14901⟩⟩) (.authority (.programFamilyFact))

def exact57273RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14901⟩⟩], []⟩, (1)⟩]

theorem exact57273RawTermsValid :
    exact57273RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event57273 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14901⟩⟩) exact57273RawTerms (.finite 58) 57272 .exactZero (none)

def event57274 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45347⟩⟩) 0 ⟨14901⟩ 57273

def event57275 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45347⟩⟩) 1 ⟨45346⟩ 57270

def event57276 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45347⟩⟩) (.product (.predecessor 0 57274 .coefficient) (.predecessor 1 57275 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event57277 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨45347⟩⟩, .operator (⟨57273, 0⟩, ⟨57270, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨14901⟩⟩, ⟨.program ⟨257⟩, ⟨45346⟩⟩], []⟩, (1)⟩)

def exact57278RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14901⟩⟩, ⟨.program ⟨257⟩, ⟨45346⟩⟩], []⟩, (1)⟩]

theorem exact57278RawTermsValid :
    exact57278RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event57278 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45347⟩⟩) exact57278RawTerms (.finite 3364) 57276 .exactZero (none)

def event57279 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45348⟩⟩) 0 ⟨45347⟩ 57278

def event57280 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45348⟩⟩) (.identity (.predecessor 0 57279 .coefficient))

def event57281 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨45348⟩⟩) (.finite 3364)

def event57282 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45532⟩⟩) 0 ⟨45348⟩ 57281

def event57283 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45532⟩⟩) (.authority (.programFamilyFact))

def exact57284RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨45532⟩⟩], []⟩, (1)⟩]

theorem exact57284RawTermsValid :
    exact57284RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event57284 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45532⟩⟩) exact57284RawTerms (.finite 58) 57283 .exactZero (none)

def event57285 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45533⟩⟩) 0 ⟨45532⟩ 57284

def event57286 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45533⟩⟩) (.identity (.predecessor 0 57285 .coefficient))

def event57287 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨45533⟩⟩) (.finite 58)

def event57288 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46691⟩⟩) 0 ⟨45533⟩ 57287

def event57289 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46691⟩⟩) (.authority (.programFamilyFact))

def event57290 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨46691⟩⟩) (.finite 3720)

def event57291 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨7177⟩⟩) .missing

def event57292 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46692⟩⟩) 0 ⟨7177⟩ 57291

def event57293 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46692⟩⟩) 1 ⟨46691⟩ 57290

def event57294 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46692⟩⟩) (.authority (.operator))

def exact57295RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨46692⟩⟩]⟩, (1)⟩]

theorem exact57295RawTermsValid :
    exact57295RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event57295 : Event := .resultExact (⟨.program ⟨257⟩, ⟨46692⟩⟩) exact57295RawTerms .large 57294 .exactZero (none)

def event57296 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47543⟩⟩) 0 ⟨46692⟩ 57295

def event57297 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47543⟩⟩) (.authority (.operator))

def exact57298RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨47543⟩⟩]⟩, (1)⟩]

theorem exact57298RawTermsValid :
    exact57298RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event57298 : Event := .resultExact (⟨.program ⟨257⟩, ⟨47543⟩⟩) exact57298RawTerms (.finite 8192) 57297 .exactZero (none)

def event57299 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event57300 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event57301 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46858⟩⟩) 0 ⟨45533⟩ 57287

def event57302 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46858⟩⟩) 1 ⟨136⟩ 57300

def event57303 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46858⟩⟩) (.sum [.predecessor 0 57301 .coefficient, .predecessor 1 57302 .coefficient])

def event57304 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨46858⟩⟩) (.finite 58)

def event57305 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46859⟩⟩) 0 ⟨46858⟩ 57304

def event57306 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46859⟩⟩) (.identity (.predecessor 0 57305 .coefficient))

def exact57307RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨45532⟩⟩], []⟩, (1)⟩]

theorem exact57307RawTermsValid :
    exact57307RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event57307 : Event := .resultExact (⟨.program ⟨257⟩, ⟨46859⟩⟩) exact57307RawTerms (.finite 58) 57306 .exactZero (none)

def event57308 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact57309RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact57309RawTermsValid :
    exact57309RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event57309 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact57309RawTerms .large 57308 .exactZero (none)

def event57310 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46860⟩⟩) 0 ⟨6908⟩ 57309

def event57311 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46860⟩⟩) 1 ⟨46859⟩ 57307

def event57312 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46860⟩⟩) (.product (.predecessor 0 57310 .coefficient) (.predecessor 1 57311 .coefficient) (⟨false, false, none, none, none⟩))

def event57313 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨46860⟩⟩, .operator (⟨57309, 0⟩, ⟨57307, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨45532⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact57314RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨45532⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact57314RawTermsValid :
    exact57314RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event57314 : Event := .resultExact (⟨.program ⟨257⟩, ⟨46860⟩⟩) exact57314RawTerms .large 57312 .exactZero (none)

def event57315 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7195⟩⟩) 0 ⟨7177⟩ 57291

def event57316 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7195⟩⟩) (.authority (.operator))

def exact57317RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7195⟩⟩]⟩, (1)⟩]

theorem exact57317RawTermsValid :
    exact57317RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event57317 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7195⟩⟩) exact57317RawTerms .large 57316 .exactZero (none)

def event57318 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46861⟩⟩) 0 ⟨7195⟩ 57317

def event57319 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46861⟩⟩) 1 ⟨46860⟩ 57314

def event57320 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46861⟩⟩) (.sum [.predecessor 0 57318 .coefficient, .predecessor 1 57319 .coefficient])

def exact57321RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7195⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨45532⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact57321RawTermsValid :
    exact57321RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event57321 : Event := .resultExact (⟨.program ⟨257⟩, ⟨46861⟩⟩) exact57321RawTerms .large 57320 .exactZero (none)

def event57322 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47544⟩⟩) 0 ⟨46861⟩ 57321

def event57323 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47544⟩⟩) 1 ⟨47543⟩ 57298

def event57324 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47544⟩⟩) (.product (.predecessor 0 57322 .coefficient) (.predecessor 1 57323 .coefficient) (⟨false, false, none, none, none⟩))

def event57325 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨47544⟩⟩, .operator (⟨57321, 0⟩, ⟨57298, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47543⟩⟩]⟩, (1)⟩)

def event57326 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨47544⟩⟩, .operator (⟨57321, 1⟩, ⟨57298, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨45532⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨47543⟩⟩]⟩, (-1)⟩)

def event57327 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨47544⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨45532⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨47543⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨47543⟩⟩) ⟨46692⟩ 57295)

def event57328 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨47544⟩⟩, .relation 57327 0, ⟨[⟨.program ⟨257⟩, ⟨45532⟩⟩], [⟨.program ⟨257⟩, ⟨46692⟩⟩]⟩, (-1)⟩)

def exact57329RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47543⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨45532⟩⟩], [⟨.program ⟨257⟩, ⟨46692⟩⟩]⟩, (-1)⟩]

theorem exact57329RawTermsValid :
    exact57329RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event57329 : Event := .resultExact (⟨.program ⟨257⟩, ⟨47544⟩⟩) exact57329RawTerms .large 57324 .exactZero (none)

def event57330 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45783⟩⟩) 0 ⟨45533⟩ 57287

def event57331 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45783⟩⟩) (.authority (.programFamilyFact))

def exact57332RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨45783⟩⟩], []⟩, (1)⟩]

theorem exact57332RawTermsValid :
    exact57332RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event57332 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45783⟩⟩) exact57332RawTerms (.finite 58) 57331 .exactZero (none)

def event57333 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45785⟩⟩) 0 ⟨6908⟩ 57309

def event57334 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45785⟩⟩) 1 ⟨45783⟩ 57332

def event57335 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45785⟩⟩) (.product (.predecessor 0 57333 .coefficient) (.predecessor 1 57334 .coefficient) (⟨false, true, none, none, some 1⟩))

def event57336 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨45785⟩⟩, .operator (⟨57309, 0⟩, ⟨57332, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨45783⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact57337RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨45783⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact57337RawTermsValid :
    exact57337RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event57337 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45785⟩⟩) exact57337RawTerms .large 57335 .exactZero (none)

def event57338 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7229⟩⟩) 0 ⟨7177⟩ 57291

def event57339 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7229⟩⟩) (.authority (.operator))

def exact57340RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7229⟩⟩]⟩, (1)⟩]

theorem exact57340RawTermsValid :
    exact57340RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event57340 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7229⟩⟩) exact57340RawTerms .large 57339 .exactZero (none)

def event57341 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45786⟩⟩) 0 ⟨7229⟩ 57340

def event57342 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45786⟩⟩) 1 ⟨45785⟩ 57337

def event57343 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45786⟩⟩) (.sum [.predecessor 0 57341 .coefficient, .predecessor 1 57342 .coefficient])

def eventLeaf3568 : Array AnnotatedEvent := #[
  { event := event57088
    frameStart := 57035 },
  { event := event57089
    frameStart := 57035 },
  { event := event57090
    frameStart := 57035 },
  { event := event57091
    frameStart := 57035 },
  { event := event57092
    frameStart := 57035 },
  { event := event57093
    frameStart := 57035 },
  { event := event57094
    frameStart := 57035 },
  { event := event57095
    frameStart := 57035 },
  { event := event57096
    frameStart := 57035 },
  { event := event57097
    frameStart := 57035 },
  { event := event57098
    frameStart := 57035 },
  { event := event57099
    frameStart := 57035 },
  { event := event57100
    frameStart := 57035 },
  { event := event57101
    frameStart := 57035 },
  { event := event57102
    frameStart := 57035 },
  { event := event57103
    frameStart := 57035 }
]

def eventLeaf3569 : Array AnnotatedEvent := #[
  { event := event57104
    frameStart := 57035 },
  { event := event57105
    frameStart := 57035 },
  { event := event57106
    frameStart := 57035 },
  { event := event57107
    frameStart := 57035 },
  { event := event57108
    frameStart := 57035 },
  { event := event57109
    frameStart := 57035 },
  { event := event57110
    frameStart := 57035 },
  { event := event57111
    frameStart := 57035 },
  { event := event57112
    frameStart := 57035 },
  { event := event57113
    frameStart := 57035 },
  { event := event57114
    frameStart := 57035 },
  { event := event57115
    frameStart := 57035 },
  { event := event57116
    frameStart := 57035 },
  { event := event57117
    frameStart := 57035 },
  { event := event57118
    frameStart := 57035 },
  { event := event57119
    frameStart := 57035 }
]

def eventLeaf3570 : Array AnnotatedEvent := #[
  { event := event57120
    frameStart := 57035 },
  { event := event57121
    frameStart := 57035 },
  { event := event57122
    frameStart := 57035 },
  { event := event57123
    frameStart := 57035 },
  { event := event57124
    frameStart := 57035 },
  { event := event57125
    frameStart := 57035 },
  { event := event57126
    frameStart := 57035 },
  { event := event57127
    frameStart := 57035 },
  { event := event57128
    frameStart := 57035 },
  { event := event57129
    frameStart := 57035 },
  { event := event57130
    frameStart := 57035 },
  { event := event57131
    frameStart := 57035 },
  { event := event57132
    frameStart := 57035 },
  { event := event57133
    frameStart := 57035 },
  { event := event57134
    frameStart := 57035 },
  { event := event57135
    frameStart := 57035 }
]

def eventLeaf3571 : Array AnnotatedEvent := #[
  { event := event57136
    frameStart := 57035 },
  { event := event57137
    frameStart := 57035 },
  { event := event57138
    frameStart := 57035 },
  { event := event57139
    frameStart := 0 },
  { event := event57140
    frameStart := 0 },
  { event := event57141
    frameStart := 0 },
  { event := event57142
    frameStart := 0 },
  { event := event57143
    frameStart := 0 },
  { event := event57144
    frameStart := 0 },
  { event := event57145
    frameStart := 0 },
  { event := event57146
    frameStart := 0 },
  { event := event57147
    frameStart := 0 },
  { event := event57148
    frameStart := 0 },
  { event := event57149
    frameStart := 0 },
  { event := event57150
    frameStart := 0 },
  { event := event57151
    frameStart := 0 }
]

def eventLeaf3572 : Array AnnotatedEvent := #[
  { event := event57152
    frameStart := 0 },
  { event := event57153
    frameStart := 0 },
  { event := event57154
    frameStart := 0 },
  { event := event57155
    frameStart := 0 },
  { event := event57156
    frameStart := 0 },
  { event := event57157
    frameStart := 0 },
  { event := event57158
    frameStart := 0 },
  { event := event57159
    frameStart := 0 },
  { event := event57160
    frameStart := 0 },
  { event := event57161
    frameStart := 0 },
  { event := event57162
    frameStart := 0 },
  { event := event57163
    frameStart := 0 },
  { event := event57164
    frameStart := 0 },
  { event := event57165
    frameStart := 0 },
  { event := event57166
    frameStart := 0 },
  { event := event57167
    frameStart := 0 }
]

def eventLeaf3573 : Array AnnotatedEvent := #[
  { event := event57168
    frameStart := 0 },
  { event := event57169
    frameStart := 0 },
  { event := event57170
    frameStart := 0 },
  { event := event57171
    frameStart := 0 },
  { event := event57172
    frameStart := 0 },
  { event := event57173
    frameStart := 0 },
  { event := event57174
    frameStart := 0 },
  { event := event57175
    frameStart := 0 },
  { event := event57176
    frameStart := 0 },
  { event := event57177
    frameStart := 0 },
  { event := event57178
    frameStart := 0 },
  { event := event57179
    frameStart := 0 },
  { event := event57180
    frameStart := 0 },
  { event := event57181
    frameStart := 0 },
  { event := event57182
    frameStart := 0 },
  { event := event57183
    frameStart := 0 }
]

def eventLeaf3574 : Array AnnotatedEvent := #[
  { event := event57184
    frameStart := 0 },
  { event := event57185
    frameStart := 0 },
  { event := event57186
    frameStart := 0 },
  { event := event57187
    frameStart := 0 },
  { event := event57188
    frameStart := 0 },
  { event := event57189
    frameStart := 0 },
  { event := event57190
    frameStart := 0 },
  { event := event57191
    frameStart := 0 },
  { event := event57192
    frameStart := 0 },
  { event := event57193
    frameStart := 57193 },
  { event := event57194
    frameStart := 57193 },
  { event := event57195
    frameStart := 57193 },
  { event := event57196
    frameStart := 57193 },
  { event := event57197
    frameStart := 57193 },
  { event := event57198
    frameStart := 57193 },
  { event := event57199
    frameStart := 57193 }
]

def eventLeaf3575 : Array AnnotatedEvent := #[
  { event := event57200
    frameStart := 57193 },
  { event := event57201
    frameStart := 57193 },
  { event := event57202
    frameStart := 57193 },
  { event := event57203
    frameStart := 57193 },
  { event := event57204
    frameStart := 57193 },
  { event := event57205
    frameStart := 57193 },
  { event := event57206
    frameStart := 57193 },
  { event := event57207
    frameStart := 57193 },
  { event := event57208
    frameStart := 57193 },
  { event := event57209
    frameStart := 57193 },
  { event := event57210
    frameStart := 57193 },
  { event := event57211
    frameStart := 57193 },
  { event := event57212
    frameStart := 57193 },
  { event := event57213
    frameStart := 57193 },
  { event := event57214
    frameStart := 57193 },
  { event := event57215
    frameStart := 57193 }
]

def eventLeaf3576 : Array AnnotatedEvent := #[
  { event := event57216
    frameStart := 57193 },
  { event := event57217
    frameStart := 57193 },
  { event := event57218
    frameStart := 57193 },
  { event := event57219
    frameStart := 57193 },
  { event := event57220
    frameStart := 57193 },
  { event := event57221
    frameStart := 57193 },
  { event := event57222
    frameStart := 57193 },
  { event := event57223
    frameStart := 57193 },
  { event := event57224
    frameStart := 57193 },
  { event := event57225
    frameStart := 57193 },
  { event := event57226
    frameStart := 57193 },
  { event := event57227
    frameStart := 57193 },
  { event := event57228
    frameStart := 57193 },
  { event := event57229
    frameStart := 57193 },
  { event := event57230
    frameStart := 57193 },
  { event := event57231
    frameStart := 57193 }
]

def eventLeaf3577 : Array AnnotatedEvent := #[
  { event := event57232
    frameStart := 57193 },
  { event := event57233
    frameStart := 57193 },
  { event := event57234
    frameStart := 57193 },
  { event := event57235
    frameStart := 57193 },
  { event := event57236
    frameStart := 57193 },
  { event := event57237
    frameStart := 57193 },
  { event := event57238
    frameStart := 57193 },
  { event := event57239
    frameStart := 57193 },
  { event := event57240
    frameStart := 57193 },
  { event := event57241
    frameStart := 57193 },
  { event := event57242
    frameStart := 57193 },
  { event := event57243
    frameStart := 57193 },
  { event := event57244
    frameStart := 57193 },
  { event := event57245
    frameStart := 57193 },
  { event := event57246
    frameStart := 57193 },
  { event := event57247
    frameStart := 57247 }
]

def eventLeaf3578 : Array AnnotatedEvent := #[
  { event := event57248
    frameStart := 57247 },
  { event := event57249
    frameStart := 57247 },
  { event := event57250
    frameStart := 57247 },
  { event := event57251
    frameStart := 57247 },
  { event := event57252
    frameStart := 57247 },
  { event := event57253
    frameStart := 57247 },
  { event := event57254
    frameStart := 57247 },
  { event := event57255
    frameStart := 57247 },
  { event := event57256
    frameStart := 57247 },
  { event := event57257
    frameStart := 57247 },
  { event := event57258
    frameStart := 57247 },
  { event := event57259
    frameStart := 57247 },
  { event := event57260
    frameStart := 57247 },
  { event := event57261
    frameStart := 57247 },
  { event := event57262
    frameStart := 57247 },
  { event := event57263
    frameStart := 57247 }
]

def eventLeaf3579 : Array AnnotatedEvent := #[
  { event := event57264
    frameStart := 57247 },
  { event := event57265
    frameStart := 57247 },
  { event := event57266
    frameStart := 57247 },
  { event := event57267
    frameStart := 57247 },
  { event := event57268
    frameStart := 57247 },
  { event := event57269
    frameStart := 57247 },
  { event := event57270
    frameStart := 57247 },
  { event := event57271
    frameStart := 57247 },
  { event := event57272
    frameStart := 57247 },
  { event := event57273
    frameStart := 57247 },
  { event := event57274
    frameStart := 57247 },
  { event := event57275
    frameStart := 57247 },
  { event := event57276
    frameStart := 57247 },
  { event := event57277
    frameStart := 57247 },
  { event := event57278
    frameStart := 57247 },
  { event := event57279
    frameStart := 57247 }
]

def eventLeaf3580 : Array AnnotatedEvent := #[
  { event := event57280
    frameStart := 57247 },
  { event := event57281
    frameStart := 57247 },
  { event := event57282
    frameStart := 57247 },
  { event := event57283
    frameStart := 57247 },
  { event := event57284
    frameStart := 57247 },
  { event := event57285
    frameStart := 57247 },
  { event := event57286
    frameStart := 57247 },
  { event := event57287
    frameStart := 57247 },
  { event := event57288
    frameStart := 57247 },
  { event := event57289
    frameStart := 57247 },
  { event := event57290
    frameStart := 57247 },
  { event := event57291
    frameStart := 57247 },
  { event := event57292
    frameStart := 57247 },
  { event := event57293
    frameStart := 57247 },
  { event := event57294
    frameStart := 57247 },
  { event := event57295
    frameStart := 57247 }
]

def eventLeaf3581 : Array AnnotatedEvent := #[
  { event := event57296
    frameStart := 57247 },
  { event := event57297
    frameStart := 57247 },
  { event := event57298
    frameStart := 57247 },
  { event := event57299
    frameStart := 57247 },
  { event := event57300
    frameStart := 57247 },
  { event := event57301
    frameStart := 57247 },
  { event := event57302
    frameStart := 57247 },
  { event := event57303
    frameStart := 57247 },
  { event := event57304
    frameStart := 57247 },
  { event := event57305
    frameStart := 57247 },
  { event := event57306
    frameStart := 57247 },
  { event := event57307
    frameStart := 57247 },
  { event := event57308
    frameStart := 57247 },
  { event := event57309
    frameStart := 57247 },
  { event := event57310
    frameStart := 57247 },
  { event := event57311
    frameStart := 57247 }
]

def eventLeaf3582 : Array AnnotatedEvent := #[
  { event := event57312
    frameStart := 57247 },
  { event := event57313
    frameStart := 57247 },
  { event := event57314
    frameStart := 57247 },
  { event := event57315
    frameStart := 57247 },
  { event := event57316
    frameStart := 57247 },
  { event := event57317
    frameStart := 57247 },
  { event := event57318
    frameStart := 57247 },
  { event := event57319
    frameStart := 57247 },
  { event := event57320
    frameStart := 57247 },
  { event := event57321
    frameStart := 57247 },
  { event := event57322
    frameStart := 57247 },
  { event := event57323
    frameStart := 57247 },
  { event := event57324
    frameStart := 57247 },
  { event := event57325
    frameStart := 57247 },
  { event := event57326
    frameStart := 57247 },
  { event := event57327
    frameStart := 57247 }
]

def eventLeaf3583 : Array AnnotatedEvent := #[
  { event := event57328
    frameStart := 57247 },
  { event := event57329
    frameStart := 57247 },
  { event := event57330
    frameStart := 57247 },
  { event := event57331
    frameStart := 57247 },
  { event := event57332
    frameStart := 57247 },
  { event := event57333
    frameStart := 57247 },
  { event := event57334
    frameStart := 57247 },
  { event := event57335
    frameStart := 57247 },
  { event := event57336
    frameStart := 57247 },
  { event := event57337
    frameStart := 57247 },
  { event := event57338
    frameStart := 57247 },
  { event := event57339
    frameStart := 57247 },
  { event := event57340
    frameStart := 57247 },
  { event := event57341
    frameStart := 57247 },
  { event := event57342
    frameStart := 57247 },
  { event := event57343
    frameStart := 57247 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events223
