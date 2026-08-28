import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events309

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event79104 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14949⟩⟩) 0 ⟨10670⟩ 79103

def event79105 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14949⟩⟩) (.authority (.programFamilyFact))

def exact79106RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨14949⟩⟩], []⟩, (1)⟩]

theorem exact79106RawTermsValid :
    exact79106RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79106 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14949⟩⟩) exact79106RawTerms (.finite 3) 79105 .exactZero (none)

def event79107 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14950⟩⟩) 0 ⟨14949⟩ 79106

def event79108 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14950⟩⟩) (.identity (.predecessor 0 79107 .coefficient))

def event79109 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨14950⟩⟩) (.finite 3)

def event79110 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23779⟩⟩) 0 ⟨14950⟩ 79109

def event79111 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23779⟩⟩) (.authority (.programFamilyFact))

def event79112 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨23779⟩⟩) (.finite 3720)

def event79113 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨6689⟩⟩) .missing

def event79114 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23780⟩⟩) 0 ⟨6689⟩ 79113

def event79115 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23780⟩⟩) 1 ⟨23779⟩ 79112

def event79116 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23780⟩⟩) (.authority (.operator))

def exact79117RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨23780⟩⟩]⟩, (1)⟩]

theorem exact79117RawTermsValid :
    exact79117RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79117 : Event := .resultExact (⟨.program ⟨214⟩, ⟨23780⟩⟩) exact79117RawTerms .large 79116 .exactZero (none)

def event79118 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26544⟩⟩) 0 ⟨23780⟩ 79117

def event79119 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26544⟩⟩) (.authority (.operator))

def exact79120RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨26544⟩⟩]⟩, (1)⟩]

theorem exact79120RawTermsValid :
    exact79120RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79120 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26544⟩⟩) exact79120RawTerms (.finite 8192) 79119 .exactZero (none)

def event79121 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨110⟩⟩) (.authority (.operator))

def event79122 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨110⟩⟩) .exactZero

def event79123 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14989⟩⟩) 0 ⟨14950⟩ 79109

def event79124 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14989⟩⟩) 1 ⟨110⟩ 79122

def event79125 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14989⟩⟩) (.sum [.predecessor 0 79123 .coefficient, .predecessor 1 79124 .coefficient])

def event79126 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨14989⟩⟩) (.finite 3)

def event79127 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14990⟩⟩) 0 ⟨14989⟩ 79126

def event79128 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14990⟩⟩) (.identity (.predecessor 0 79127 .coefficient))

def exact79129RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨14949⟩⟩], []⟩, (1)⟩]

theorem exact79129RawTermsValid :
    exact79129RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79129 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14990⟩⟩) exact79129RawTerms (.finite 3) 79128 .exactZero (none)

def event79130 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6544⟩⟩) (.authority (.factStore))

def exact79131RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact79131RawTermsValid :
    exact79131RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79131 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6544⟩⟩) exact79131RawTerms .large 79130 .exactZero (none)

def event79132 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14991⟩⟩) 0 ⟨6544⟩ 79131

def event79133 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14991⟩⟩) 1 ⟨14990⟩ 79129

def event79134 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14991⟩⟩) (.product (.predecessor 0 79132 .coefficient) (.predecessor 1 79133 .coefficient) (⟨false, false, none, none, none⟩))

def event79135 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨14991⟩⟩, .operator (⟨79131, 0⟩, ⟨79129, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨14949⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact79136RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨14949⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact79136RawTermsValid :
    exact79136RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79136 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14991⟩⟩) exact79136RawTerms .large 79134 .exactZero (none)

def event79137 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6691⟩⟩) 0 ⟨6689⟩ 79113

def event79138 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6691⟩⟩) (.authority (.operator))

def exact79139RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6691⟩⟩]⟩, (1)⟩]

theorem exact79139RawTermsValid :
    exact79139RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79139 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6691⟩⟩) exact79139RawTerms .large 79138 .exactZero (none)

def event79140 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14992⟩⟩) 0 ⟨6691⟩ 79139

def event79141 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14992⟩⟩) 1 ⟨14991⟩ 79136

def event79142 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14992⟩⟩) (.sum [.predecessor 0 79140 .coefficient, .predecessor 1 79141 .coefficient])

def exact79143RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6691⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨14949⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact79143RawTermsValid :
    exact79143RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79143 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14992⟩⟩) exact79143RawTerms .large 79142 .exactZero (none)

def event79144 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26545⟩⟩) 0 ⟨14992⟩ 79143

def event79145 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26545⟩⟩) 1 ⟨26544⟩ 79120

def event79146 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26545⟩⟩) (.product (.predecessor 0 79144 .coefficient) (.predecessor 1 79145 .coefficient) (⟨false, false, none, none, none⟩))

def event79147 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26545⟩⟩, .operator (⟨79143, 0⟩, ⟨79120, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6691⟩⟩, ⟨.program ⟨214⟩, ⟨26544⟩⟩]⟩, (1)⟩)

def event79148 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26545⟩⟩, .operator (⟨79143, 1⟩, ⟨79120, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨14949⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26544⟩⟩]⟩, (-1)⟩)

def event79149 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨26545⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨14949⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26544⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨26544⟩⟩) ⟨23780⟩ 79117)

def event79150 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26545⟩⟩, .relation 79149 0, ⟨[⟨.program ⟨214⟩, ⟨14949⟩⟩], [⟨.program ⟨214⟩, ⟨23780⟩⟩]⟩, (-1)⟩)

def exact79151RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6691⟩⟩, ⟨.program ⟨214⟩, ⟨26544⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨14949⟩⟩], [⟨.program ⟨214⟩, ⟨23780⟩⟩]⟩, (-1)⟩]

theorem exact79151RawTermsValid :
    exact79151RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79151 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26545⟩⟩) exact79151RawTerms .large 79146 .exactZero (none)

def event79152 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15042⟩⟩) 0 ⟨14950⟩ 79109

def event79153 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15042⟩⟩) (.authority (.programFamilyFact))

def exact79154RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15042⟩⟩], []⟩, (1)⟩]

theorem exact79154RawTermsValid :
    exact79154RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79154 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15042⟩⟩) exact79154RawTerms (.finite 3) 79153 .exactZero (none)

def event79155 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15045⟩⟩) 0 ⟨6544⟩ 79131

def event79156 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15045⟩⟩) 1 ⟨15042⟩ 79154

def event79157 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15045⟩⟩) (.product (.predecessor 0 79155 .coefficient) (.predecessor 1 79156 .coefficient) (⟨false, true, none, none, some 1⟩))

def event79158 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨15045⟩⟩, .operator (⟨79131, 0⟩, ⟨79154, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨15042⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact79159RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15042⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact79159RawTermsValid :
    exact79159RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79159 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15045⟩⟩) exact79159RawTerms .large 79157 .exactZero (none)

def event79160 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6710⟩⟩) 0 ⟨6689⟩ 79113

def event79161 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6710⟩⟩) (.authority (.operator))

def exact79162RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6710⟩⟩]⟩, (1)⟩]

theorem exact79162RawTermsValid :
    exact79162RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79162 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6710⟩⟩) exact79162RawTerms .large 79161 .exactZero (none)

def event79163 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15046⟩⟩) 0 ⟨6710⟩ 79162

def event79164 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15046⟩⟩) 1 ⟨15045⟩ 79159

def event79165 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15046⟩⟩) (.sum [.predecessor 0 79163 .coefficient, .predecessor 1 79164 .coefficient])

def exact79166RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6710⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15042⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact79166RawTermsValid :
    exact79166RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79166 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15046⟩⟩) exact79166RawTerms .large 79165 .exactZero (none)

def event79167 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26550⟩⟩) 0 ⟨15046⟩ 79166

def event79168 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26550⟩⟩) 1 ⟨26545⟩ 79151

def event79169 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26550⟩⟩) (.sum [.predecessor 0 79167 .coefficient, .predecessor 1 79168 .coefficient])

def exact79170RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6691⟩⟩, ⟨.program ⟨214⟩, ⟨26544⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6710⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨14949⟩⟩], [⟨.program ⟨214⟩, ⟨23780⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15042⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact79170RawTermsValid :
    exact79170RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79170 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26550⟩⟩) exact79170RawTerms .large 79169 .exactZero (none)

def event79171 : Event := .preFoldPolynomial 79170 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6691⟩⟩, ⟨.program ⟨214⟩, ⟨26544⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6710⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨14949⟩⟩], [⟨.program ⟨214⟩, ⟨23780⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15042⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩] .exactZero none

def exact79172RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6691⟩⟩, ⟨.program ⟨214⟩, ⟨26544⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6710⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨14949⟩⟩], [⟨.program ⟨214⟩, ⟨23780⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15042⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

def event79172 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨26550⟩⟩) 79171 exact79172RawTerms .large 79169 .exactZero (none)

def event79173 : Event := .specializationComputed (⟨.program ⟨214⟩, ⟨14950⟩⟩) ⟨⟨123⟩, ⟨29⟩, ⟨109⟩⟩ ⟨79015, 79173⟩

def event79174 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨20463⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20460⟩⟩]⟩) (1) 0 2 (.universal 79173 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20460⟩⟩]⟩) (none) 79172)

def event79175 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20463⟩⟩, .relation 79174 1, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6710⟩⟩]⟩, (1)⟩)

def event79176 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20463⟩⟩, .relation 79174 0, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6691⟩⟩, ⟨.program ⟨214⟩, ⟨26544⟩⟩]⟩, (-1)⟩)

def event79177 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20463⟩⟩, .relation 79174 2, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨14949⟩⟩], [⟨.program ⟨214⟩, ⟨23780⟩⟩]⟩, (1)⟩)

def event79178 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20463⟩⟩, .relation 79174 3, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15042⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact79179RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6691⟩⟩, ⟨.program ⟨214⟩, ⟨26544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6710⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨14949⟩⟩], [⟨.program ⟨214⟩, ⟨23780⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15042⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact79179RawTermsValid :
    exact79179RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79179 : Event := .resultExact (⟨.program ⟨214⟩, ⟨20463⟩⟩) exact79179RawTerms .large 79011 (.finite 1811303510016) (some (79013))

def event79180 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26547⟩⟩) 0 ⟨20463⟩ 79179

def event79181 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26547⟩⟩) 1 ⟨26546⟩ 79001

def event79182 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26547⟩⟩) (.sum [.predecessor 0 79180 .coefficient, .predecessor 1 79181 .coefficient])

def event79183 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26547⟩⟩, .operator (⟨79179, 0⟩, ⟨79001, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6691⟩⟩, ⟨.program ⟨214⟩, ⟨26544⟩⟩]⟩, (1)⟩)

def event79184 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26547⟩⟩, .operator (⟨79179, 2⟩, ⟨79001, 1⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨14949⟩⟩], [⟨.program ⟨214⟩, ⟨23780⟩⟩]⟩, (-1)⟩)

def event79185 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26547⟩⟩) (.sum [.result 79179 .summary, .result 79001 .summary])

def exact79186RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6710⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15042⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact79186RawTermsValid :
    exact79186RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79186 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26547⟩⟩) exact79186RawTerms .large 79182 (.finite 1291900380601931935744) (some (79185))

def event79187 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26548⟩⟩) 0 ⟨26547⟩ 79186

def event79188 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26548⟩⟩) 1 ⟨6672⟩ 5839

def event79189 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26548⟩⟩) (.product (.predecessor 0 79187 .coefficient) (.predecessor 1 79188 .coefficient) (⟨false, false, none, none, none⟩))

def event79190 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26548⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨6671⟩⟩]⟩) [⟨.result 5835 .coefficient, false, none⟩])

def event79191 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26548⟩⟩) (.product (.result 79186 .summary) (.transfer 79190) (⟨false, false, none, none, none⟩))

def event79192 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26548⟩⟩, .operator (⟨79186, 0⟩, ⟨5839, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6710⟩⟩, ⟨.program ⟨214⟩, ⟨6671⟩⟩]⟩, (1)⟩)

def event79193 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26548⟩⟩, .operator (⟨79186, 1⟩, ⟨5839, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15042⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨6671⟩⟩]⟩, (-1)⟩)

def event79194 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨26548⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15042⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨6671⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨6671⟩⟩) ⟨6607⟩ 5832)

def event79195 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26548⟩⟩, .relation 79194 0, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨15042⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact79196RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6710⟩⟩, ⟨.program ⟨214⟩, ⟨6671⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨15042⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact79196RawTermsValid :
    exact79196RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79196 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26548⟩⟩) exact79196RawTerms .large 79189 (.finite 4741295067215179835091451904) (some (79191))

def event79197 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23717⟩⟩) 0 ⟨6689⟩ 5477

def event79198 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23717⟩⟩) 1 ⟨23716⟩ 73483

def event79199 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23717⟩⟩) (.authority (.operator))

def exact79200RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨23717⟩⟩]⟩, (1)⟩]

theorem exact79200RawTermsValid :
    exact79200RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79200 : Event := .resultExact (⟨.program ⟨214⟩, ⟨23717⟩⟩) exact79200RawTerms .large 79199 .exactZero (none)

def event79201 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26339⟩⟩) 0 ⟨23717⟩ 79200

def event79202 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26339⟩⟩) (.authority (.operator))

def exact79203RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨26339⟩⟩]⟩, (1)⟩]

theorem exact79203RawTermsValid :
    exact79203RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79203 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26339⟩⟩) exact79203RawTerms (.finite 8192) 79202 .exactZero (none)

def event79204 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26341⟩⟩) 0 ⟨24908⟩ 73767

def event79205 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26341⟩⟩) 1 ⟨26339⟩ 79203

def event79206 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26341⟩⟩) (.product (.predecessor 0 79204 .coefficient) (.predecessor 1 79205 .coefficient) (⟨false, false, none, none, none⟩))

def event79207 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26341⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨26339⟩⟩]⟩) [⟨.result 79203 .coefficient, false, none⟩])

def event79208 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26341⟩⟩) (.product (.result 73767 .summary) (.transfer 79207) (⟨false, false, none, none, none⟩))

def event79209 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26341⟩⟩, .operator (⟨73767, 0⟩, ⟨79203, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6690⟩⟩, ⟨.program ⟨214⟩, ⟨26339⟩⟩]⟩, (1)⟩)

def event79210 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26341⟩⟩, .operator (⟨73767, 1⟩, ⟨79203, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨14788⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26339⟩⟩]⟩, (-1)⟩)

def event79211 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨26341⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨14788⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26339⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨26339⟩⟩) ⟨23717⟩ 79200)

def event79212 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26341⟩⟩, .relation 79211 0, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨14788⟩⟩], [⟨.program ⟨214⟩, ⟨23717⟩⟩]⟩, (-1)⟩)

def exact79213RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6690⟩⟩, ⟨.program ⟨214⟩, ⟨26339⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨14788⟩⟩], [⟨.program ⟨214⟩, ⟨23717⟩⟩]⟩, (-1)⟩]

theorem exact79213RawTermsValid :
    exact79213RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79213 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26341⟩⟩) exact79213RawTerms .large 79206 (.finite 1291889172568118132736) (some (79208))

def event79214 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20316⟩⟩) 0 ⟨14789⟩ 3494

def event79215 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20316⟩⟩) (.authority (.relationPreimageSource ⟨27⟩))

def exact79216RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨20316⟩⟩]⟩, (1)⟩]

theorem exact79216RawTermsValid :
    exact79216RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79216 : Event := .resultExact (⟨.program ⟨214⟩, ⟨20316⟩⟩) exact79216RawTerms (.finite 136065468) 79215 .exactZero (none)

def event79217 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20318⟩⟩) 0 ⟨20316⟩ 79216

def event79218 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20318⟩⟩) 1 ⟨2348⟩ 4

def event79219 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20318⟩⟩) (.scale (.predecessor 0 79217 .coefficient) (.value (.predecessor 1 79218 .coefficient)))

def exact79220RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨20316⟩⟩]⟩, (1)⟩]

theorem exact79220RawTermsValid :
    exact79220RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79220 : Event := .resultExact (⟨.program ⟨214⟩, ⟨20318⟩⟩) exact79220RawTerms (.finite 136065468) 79219 .exactZero (none)

def event79221 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20319⟩⟩) 0 ⟨5535⟩ 65387

def event79222 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20319⟩⟩) 1 ⟨20318⟩ 79220

def event79223 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20319⟩⟩) (.product (.predecessor 0 79221 .coefficient) (.predecessor 1 79222 .coefficient) (⟨false, false, none, none, none⟩))

def event79224 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20319⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨20316⟩⟩]⟩) [⟨.result 79216 .coefficient, false, none⟩])

def event79225 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20319⟩⟩) (.product (.result 65387 .summary) (.transfer 79224) (⟨false, false, none, none, none⟩))

def event79226 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20319⟩⟩, .operator (⟨65387, 0⟩, ⟨79220, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20316⟩⟩]⟩, (1)⟩)

def event79227 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨20317⟩⟩)

def event79228 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event79229 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event79230 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨360⟩⟩) (.authority (.operator))

def event79231 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨360⟩⟩) (.finite 2)

def event79232 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event79233 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event79234 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event79235 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event79236 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 79235

def event79237 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 79233

def event79238 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 79236 .coefficient) (.value (.predecessor 1 79237 .coefficient)))

def event79239 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event79240 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5504⟩⟩) 0 ⟨5503⟩ 79239

def event79241 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5504⟩⟩) 1 ⟨360⟩ 79231

def event79242 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5504⟩⟩) (.sum [.predecessor 0 79240 .coefficient, .predecessor 1 79241 .coefficient])

def event79243 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5504⟩⟩) (.finite 219)

def event79244 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5530⟩⟩) 0 ⟨5504⟩ 79243

def event79245 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5530⟩⟩) 1 ⟨961⟩ 79229

def event79246 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5530⟩⟩) (.identity (.predecessor 1 79245 .coefficient))

def event79247 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5530⟩⟩) (.finite 224)

def event79248 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10472⟩⟩) 0 ⟨5530⟩ 79247

def event79249 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10472⟩⟩) (.authority (.programFamilyFact))

def exact79250RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨10472⟩⟩], []⟩, (1)⟩]

theorem exact79250RawTermsValid :
    exact79250RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79250 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10472⟩⟩) exact79250RawTerms (.finite 2) 79249 .exactZero (none)

def event79251 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9395⟩⟩) 0 ⟨5530⟩ 79247

def event79252 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨9395⟩⟩) (.authority (.programFamilyFact))

def exact79253RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨9395⟩⟩], []⟩, (1)⟩]

theorem exact79253RawTermsValid :
    exact79253RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79253 : Event := .resultExact (⟨.program ⟨214⟩, ⟨9395⟩⟩) exact79253RawTerms (.finite 2) 79252 .exactZero (none)

def event79254 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10473⟩⟩) 0 ⟨9395⟩ 79253

def event79255 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10473⟩⟩) 1 ⟨10472⟩ 79250

def event79256 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10473⟩⟩) (.product (.predecessor 0 79254 .coefficient) (.predecessor 1 79255 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event79257 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10473⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨9395⟩⟩, ⟨.program ⟨214⟩, ⟨10472⟩⟩], []⟩) [⟨.result 79253 .coefficient, true, some 1⟩, ⟨.result 79250 .coefficient, true, some 1⟩])

def event79258 : Event := .survivorFold (1) 79257

def exact79259RawTerms : List Term := []

theorem exact79259RawTermsValid :
    exact79259RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79259 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10473⟩⟩) exact79259RawTerms (.finite 4) 79256 (.finite 4) (some (79257))

def event79260 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10474⟩⟩) 0 ⟨10473⟩ 79259

def event79261 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10474⟩⟩) (.identity (.predecessor 0 79260 .coefficient))

def event79262 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨10474⟩⟩) (.finite 4)

def event79263 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14788⟩⟩) 0 ⟨10474⟩ 79262

def event79264 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14788⟩⟩) (.authority (.programFamilyFact))

def exact79265RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨14788⟩⟩], []⟩, (1)⟩]

theorem exact79265RawTermsValid :
    exact79265RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79265 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14788⟩⟩) exact79265RawTerms (.finite 2) 79264 .exactZero (none)

def event79266 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14789⟩⟩) 0 ⟨14788⟩ 79265

def event79267 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14789⟩⟩) (.identity (.predecessor 0 79266 .coefficient))

def event79268 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨14789⟩⟩) (.finite 2)

def event79269 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20316⟩⟩) 0 ⟨14789⟩ 79268

def event79270 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20316⟩⟩) (.authority (.relationPreimageSource ⟨27⟩))

def exact79271RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨20316⟩⟩]⟩, (1)⟩]

theorem exact79271RawTermsValid :
    exact79271RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79271 : Event := .resultExact (⟨.program ⟨214⟩, ⟨20316⟩⟩) exact79271RawTerms (.finite 136065468) 79270 .exactZero (none)

def event79272 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6⟩⟩) (.authority (.operator))

def exact79273RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩]⟩, (1)⟩]

theorem exact79273RawTermsValid :
    exact79273RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79273 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6⟩⟩) exact79273RawTerms .large 79272 .exactZero (none)

def event79274 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20317⟩⟩) 0 ⟨6⟩ 79273

def event79275 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20317⟩⟩) 1 ⟨20316⟩ 79271

def event79276 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20317⟩⟩) (.product (.predecessor 0 79274 .coefficient) (.predecessor 1 79275 .coefficient) (⟨false, false, none, none, none⟩))

def event79277 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20317⟩⟩, .operator (⟨79273, 0⟩, ⟨79271, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20316⟩⟩]⟩, (1)⟩)

def exact79278RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20316⟩⟩]⟩, (1)⟩]

theorem exact79278RawTermsValid :
    exact79278RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79278 : Event := .resultExact (⟨.program ⟨214⟩, ⟨20317⟩⟩) exact79278RawTerms .large 79276 .exactZero (none)

def event79279 : Event := .preFoldPolynomial 79278 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20316⟩⟩]⟩, (1)⟩] .exactZero none

def exact79280RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20316⟩⟩]⟩, (1)⟩]

def event79280 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨20317⟩⟩) 79279 exact79280RawTerms .large 79276 .exactZero (none)

def event79281 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨26345⟩⟩)

def event79282 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event79283 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event79284 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨360⟩⟩) (.authority (.operator))

def event79285 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨360⟩⟩) (.finite 2)

def event79286 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event79287 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event79288 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event79289 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event79290 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 79289

def event79291 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 79287

def event79292 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 79290 .coefficient) (.value (.predecessor 1 79291 .coefficient)))

def event79293 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event79294 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5504⟩⟩) 0 ⟨5503⟩ 79293

def event79295 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5504⟩⟩) 1 ⟨360⟩ 79285

def event79296 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5504⟩⟩) (.sum [.predecessor 0 79294 .coefficient, .predecessor 1 79295 .coefficient])

def event79297 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5504⟩⟩) (.finite 219)

def event79298 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5530⟩⟩) 0 ⟨5504⟩ 79297

def event79299 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5530⟩⟩) 1 ⟨961⟩ 79283

def event79300 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5530⟩⟩) (.identity (.predecessor 1 79299 .coefficient))

def event79301 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5530⟩⟩) (.finite 224)

def event79302 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10472⟩⟩) 0 ⟨5530⟩ 79301

def event79303 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10472⟩⟩) (.authority (.programFamilyFact))

def exact79304RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨10472⟩⟩], []⟩, (1)⟩]

theorem exact79304RawTermsValid :
    exact79304RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79304 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10472⟩⟩) exact79304RawTerms (.finite 2) 79303 .exactZero (none)

def event79305 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9395⟩⟩) 0 ⟨5530⟩ 79301

def event79306 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨9395⟩⟩) (.authority (.programFamilyFact))

def exact79307RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨9395⟩⟩], []⟩, (1)⟩]

theorem exact79307RawTermsValid :
    exact79307RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79307 : Event := .resultExact (⟨.program ⟨214⟩, ⟨9395⟩⟩) exact79307RawTerms (.finite 2) 79306 .exactZero (none)

def event79308 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10473⟩⟩) 0 ⟨9395⟩ 79307

def event79309 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10473⟩⟩) 1 ⟨10472⟩ 79304

def event79310 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10473⟩⟩) (.product (.predecessor 0 79308 .coefficient) (.predecessor 1 79309 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event79311 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨10473⟩⟩, .operator (⟨79307, 0⟩, ⟨79304, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨9395⟩⟩, ⟨.program ⟨214⟩, ⟨10472⟩⟩], []⟩, (1)⟩)

def exact79312RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨9395⟩⟩, ⟨.program ⟨214⟩, ⟨10472⟩⟩], []⟩, (1)⟩]

theorem exact79312RawTermsValid :
    exact79312RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79312 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10473⟩⟩) exact79312RawTerms (.finite 4) 79310 .exactZero (none)

def event79313 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10474⟩⟩) 0 ⟨10473⟩ 79312

def event79314 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10474⟩⟩) (.identity (.predecessor 0 79313 .coefficient))

def event79315 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨10474⟩⟩) (.finite 4)

def event79316 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14788⟩⟩) 0 ⟨10474⟩ 79315

def event79317 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14788⟩⟩) (.authority (.programFamilyFact))

def exact79318RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨14788⟩⟩], []⟩, (1)⟩]

theorem exact79318RawTermsValid :
    exact79318RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79318 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14788⟩⟩) exact79318RawTerms (.finite 2) 79317 .exactZero (none)

def event79319 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14789⟩⟩) 0 ⟨14788⟩ 79318

def event79320 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14789⟩⟩) (.identity (.predecessor 0 79319 .coefficient))

def event79321 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨14789⟩⟩) (.finite 2)

def event79322 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23716⟩⟩) 0 ⟨14789⟩ 79321

def event79323 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23716⟩⟩) (.authority (.programFamilyFact))

def event79324 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨23716⟩⟩) (.finite 3720)

def event79325 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨6689⟩⟩) .missing

def event79326 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23717⟩⟩) 0 ⟨6689⟩ 79325

def event79327 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23717⟩⟩) 1 ⟨23716⟩ 79324

def event79328 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23717⟩⟩) (.authority (.operator))

def exact79329RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨23717⟩⟩]⟩, (1)⟩]

theorem exact79329RawTermsValid :
    exact79329RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79329 : Event := .resultExact (⟨.program ⟨214⟩, ⟨23717⟩⟩) exact79329RawTerms .large 79328 .exactZero (none)

def event79330 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26339⟩⟩) 0 ⟨23717⟩ 79329

def event79331 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26339⟩⟩) (.authority (.operator))

def exact79332RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨26339⟩⟩]⟩, (1)⟩]

theorem exact79332RawTermsValid :
    exact79332RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79332 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26339⟩⟩) exact79332RawTerms (.finite 8192) 79331 .exactZero (none)

def event79333 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨110⟩⟩) (.authority (.operator))

def event79334 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨110⟩⟩) .exactZero

def event79335 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14828⟩⟩) 0 ⟨14789⟩ 79321

def event79336 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14828⟩⟩) 1 ⟨110⟩ 79334

def event79337 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14828⟩⟩) (.sum [.predecessor 0 79335 .coefficient, .predecessor 1 79336 .coefficient])

def event79338 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨14828⟩⟩) (.finite 2)

def event79339 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14829⟩⟩) 0 ⟨14828⟩ 79338

def event79340 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14829⟩⟩) (.identity (.predecessor 0 79339 .coefficient))

def exact79341RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨14788⟩⟩], []⟩, (1)⟩]

theorem exact79341RawTermsValid :
    exact79341RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79341 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14829⟩⟩) exact79341RawTerms (.finite 2) 79340 .exactZero (none)

def event79342 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6544⟩⟩) (.authority (.factStore))

def exact79343RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact79343RawTermsValid :
    exact79343RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79343 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6544⟩⟩) exact79343RawTerms .large 79342 .exactZero (none)

def event79344 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14830⟩⟩) 0 ⟨6544⟩ 79343

def event79345 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14830⟩⟩) 1 ⟨14829⟩ 79341

def event79346 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14830⟩⟩) (.product (.predecessor 0 79344 .coefficient) (.predecessor 1 79345 .coefficient) (⟨false, false, none, none, none⟩))

def event79347 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨14830⟩⟩, .operator (⟨79343, 0⟩, ⟨79341, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨14788⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact79348RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨14788⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact79348RawTermsValid :
    exact79348RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79348 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14830⟩⟩) exact79348RawTerms .large 79346 .exactZero (none)

def event79349 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6690⟩⟩) 0 ⟨6689⟩ 79325

def event79350 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6690⟩⟩) (.authority (.operator))

def exact79351RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6690⟩⟩]⟩, (1)⟩]

theorem exact79351RawTermsValid :
    exact79351RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79351 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6690⟩⟩) exact79351RawTerms .large 79350 .exactZero (none)

def event79352 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14831⟩⟩) 0 ⟨6690⟩ 79351

def event79353 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14831⟩⟩) 1 ⟨14830⟩ 79348

def event79354 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14831⟩⟩) (.sum [.predecessor 0 79352 .coefficient, .predecessor 1 79353 .coefficient])

def exact79355RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6690⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨14788⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact79355RawTermsValid :
    exact79355RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79355 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14831⟩⟩) exact79355RawTerms .large 79354 .exactZero (none)

def event79356 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26340⟩⟩) 0 ⟨14831⟩ 79355

def event79357 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26340⟩⟩) 1 ⟨26339⟩ 79332

def event79358 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26340⟩⟩) (.product (.predecessor 0 79356 .coefficient) (.predecessor 1 79357 .coefficient) (⟨false, false, none, none, none⟩))

def event79359 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26340⟩⟩, .operator (⟨79355, 0⟩, ⟨79332, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6690⟩⟩, ⟨.program ⟨214⟩, ⟨26339⟩⟩]⟩, (1)⟩)

def eventLeaf4944 : Array AnnotatedEvent := #[
  { event := event79104
    frameStart := 79069 },
  { event := event79105
    frameStart := 79069 },
  { event := event79106
    frameStart := 79069 },
  { event := event79107
    frameStart := 79069 },
  { event := event79108
    frameStart := 79069 },
  { event := event79109
    frameStart := 79069 },
  { event := event79110
    frameStart := 79069 },
  { event := event79111
    frameStart := 79069 },
  { event := event79112
    frameStart := 79069 },
  { event := event79113
    frameStart := 79069 },
  { event := event79114
    frameStart := 79069 },
  { event := event79115
    frameStart := 79069 },
  { event := event79116
    frameStart := 79069 },
  { event := event79117
    frameStart := 79069 },
  { event := event79118
    frameStart := 79069 },
  { event := event79119
    frameStart := 79069 }
]

def eventLeaf4945 : Array AnnotatedEvent := #[
  { event := event79120
    frameStart := 79069 },
  { event := event79121
    frameStart := 79069 },
  { event := event79122
    frameStart := 79069 },
  { event := event79123
    frameStart := 79069 },
  { event := event79124
    frameStart := 79069 },
  { event := event79125
    frameStart := 79069 },
  { event := event79126
    frameStart := 79069 },
  { event := event79127
    frameStart := 79069 },
  { event := event79128
    frameStart := 79069 },
  { event := event79129
    frameStart := 79069 },
  { event := event79130
    frameStart := 79069 },
  { event := event79131
    frameStart := 79069 },
  { event := event79132
    frameStart := 79069 },
  { event := event79133
    frameStart := 79069 },
  { event := event79134
    frameStart := 79069 },
  { event := event79135
    frameStart := 79069 }
]

def eventLeaf4946 : Array AnnotatedEvent := #[
  { event := event79136
    frameStart := 79069 },
  { event := event79137
    frameStart := 79069 },
  { event := event79138
    frameStart := 79069 },
  { event := event79139
    frameStart := 79069 },
  { event := event79140
    frameStart := 79069 },
  { event := event79141
    frameStart := 79069 },
  { event := event79142
    frameStart := 79069 },
  { event := event79143
    frameStart := 79069 },
  { event := event79144
    frameStart := 79069 },
  { event := event79145
    frameStart := 79069 },
  { event := event79146
    frameStart := 79069 },
  { event := event79147
    frameStart := 79069 },
  { event := event79148
    frameStart := 79069 },
  { event := event79149
    frameStart := 79069 },
  { event := event79150
    frameStart := 79069 },
  { event := event79151
    frameStart := 79069 }
]

def eventLeaf4947 : Array AnnotatedEvent := #[
  { event := event79152
    frameStart := 79069 },
  { event := event79153
    frameStart := 79069 },
  { event := event79154
    frameStart := 79069 },
  { event := event79155
    frameStart := 79069 },
  { event := event79156
    frameStart := 79069 },
  { event := event79157
    frameStart := 79069 },
  { event := event79158
    frameStart := 79069 },
  { event := event79159
    frameStart := 79069 },
  { event := event79160
    frameStart := 79069 },
  { event := event79161
    frameStart := 79069 },
  { event := event79162
    frameStart := 79069 },
  { event := event79163
    frameStart := 79069 },
  { event := event79164
    frameStart := 79069 },
  { event := event79165
    frameStart := 79069 },
  { event := event79166
    frameStart := 79069 },
  { event := event79167
    frameStart := 79069 }
]

def eventLeaf4948 : Array AnnotatedEvent := #[
  { event := event79168
    frameStart := 79069 },
  { event := event79169
    frameStart := 79069 },
  { event := event79170
    frameStart := 79069 },
  { event := event79171
    frameStart := 79069 },
  { event := event79172
    frameStart := 79069 },
  { event := event79173
    frameStart := 0 },
  { event := event79174
    frameStart := 0 },
  { event := event79175
    frameStart := 0 },
  { event := event79176
    frameStart := 0 },
  { event := event79177
    frameStart := 0 },
  { event := event79178
    frameStart := 0 },
  { event := event79179
    frameStart := 0 },
  { event := event79180
    frameStart := 0 },
  { event := event79181
    frameStart := 0 },
  { event := event79182
    frameStart := 0 },
  { event := event79183
    frameStart := 0 }
]

def eventLeaf4949 : Array AnnotatedEvent := #[
  { event := event79184
    frameStart := 0 },
  { event := event79185
    frameStart := 0 },
  { event := event79186
    frameStart := 0 },
  { event := event79187
    frameStart := 0 },
  { event := event79188
    frameStart := 0 },
  { event := event79189
    frameStart := 0 },
  { event := event79190
    frameStart := 0 },
  { event := event79191
    frameStart := 0 },
  { event := event79192
    frameStart := 0 },
  { event := event79193
    frameStart := 0 },
  { event := event79194
    frameStart := 0 },
  { event := event79195
    frameStart := 0 },
  { event := event79196
    frameStart := 0 },
  { event := event79197
    frameStart := 0 },
  { event := event79198
    frameStart := 0 },
  { event := event79199
    frameStart := 0 }
]

def eventLeaf4950 : Array AnnotatedEvent := #[
  { event := event79200
    frameStart := 0 },
  { event := event79201
    frameStart := 0 },
  { event := event79202
    frameStart := 0 },
  { event := event79203
    frameStart := 0 },
  { event := event79204
    frameStart := 0 },
  { event := event79205
    frameStart := 0 },
  { event := event79206
    frameStart := 0 },
  { event := event79207
    frameStart := 0 },
  { event := event79208
    frameStart := 0 },
  { event := event79209
    frameStart := 0 },
  { event := event79210
    frameStart := 0 },
  { event := event79211
    frameStart := 0 },
  { event := event79212
    frameStart := 0 },
  { event := event79213
    frameStart := 0 },
  { event := event79214
    frameStart := 0 },
  { event := event79215
    frameStart := 0 }
]

def eventLeaf4951 : Array AnnotatedEvent := #[
  { event := event79216
    frameStart := 0 },
  { event := event79217
    frameStart := 0 },
  { event := event79218
    frameStart := 0 },
  { event := event79219
    frameStart := 0 },
  { event := event79220
    frameStart := 0 },
  { event := event79221
    frameStart := 0 },
  { event := event79222
    frameStart := 0 },
  { event := event79223
    frameStart := 0 },
  { event := event79224
    frameStart := 0 },
  { event := event79225
    frameStart := 0 },
  { event := event79226
    frameStart := 0 },
  { event := event79227
    frameStart := 79227 },
  { event := event79228
    frameStart := 79227 },
  { event := event79229
    frameStart := 79227 },
  { event := event79230
    frameStart := 79227 },
  { event := event79231
    frameStart := 79227 }
]

def eventLeaf4952 : Array AnnotatedEvent := #[
  { event := event79232
    frameStart := 79227 },
  { event := event79233
    frameStart := 79227 },
  { event := event79234
    frameStart := 79227 },
  { event := event79235
    frameStart := 79227 },
  { event := event79236
    frameStart := 79227 },
  { event := event79237
    frameStart := 79227 },
  { event := event79238
    frameStart := 79227 },
  { event := event79239
    frameStart := 79227 },
  { event := event79240
    frameStart := 79227 },
  { event := event79241
    frameStart := 79227 },
  { event := event79242
    frameStart := 79227 },
  { event := event79243
    frameStart := 79227 },
  { event := event79244
    frameStart := 79227 },
  { event := event79245
    frameStart := 79227 },
  { event := event79246
    frameStart := 79227 },
  { event := event79247
    frameStart := 79227 }
]

def eventLeaf4953 : Array AnnotatedEvent := #[
  { event := event79248
    frameStart := 79227 },
  { event := event79249
    frameStart := 79227 },
  { event := event79250
    frameStart := 79227 },
  { event := event79251
    frameStart := 79227 },
  { event := event79252
    frameStart := 79227 },
  { event := event79253
    frameStart := 79227 },
  { event := event79254
    frameStart := 79227 },
  { event := event79255
    frameStart := 79227 },
  { event := event79256
    frameStart := 79227 },
  { event := event79257
    frameStart := 79227 },
  { event := event79258
    frameStart := 79227 },
  { event := event79259
    frameStart := 79227 },
  { event := event79260
    frameStart := 79227 },
  { event := event79261
    frameStart := 79227 },
  { event := event79262
    frameStart := 79227 },
  { event := event79263
    frameStart := 79227 }
]

def eventLeaf4954 : Array AnnotatedEvent := #[
  { event := event79264
    frameStart := 79227 },
  { event := event79265
    frameStart := 79227 },
  { event := event79266
    frameStart := 79227 },
  { event := event79267
    frameStart := 79227 },
  { event := event79268
    frameStart := 79227 },
  { event := event79269
    frameStart := 79227 },
  { event := event79270
    frameStart := 79227 },
  { event := event79271
    frameStart := 79227 },
  { event := event79272
    frameStart := 79227 },
  { event := event79273
    frameStart := 79227 },
  { event := event79274
    frameStart := 79227 },
  { event := event79275
    frameStart := 79227 },
  { event := event79276
    frameStart := 79227 },
  { event := event79277
    frameStart := 79227 },
  { event := event79278
    frameStart := 79227 },
  { event := event79279
    frameStart := 79227 }
]

def eventLeaf4955 : Array AnnotatedEvent := #[
  { event := event79280
    frameStart := 79227 },
  { event := event79281
    frameStart := 79281 },
  { event := event79282
    frameStart := 79281 },
  { event := event79283
    frameStart := 79281 },
  { event := event79284
    frameStart := 79281 },
  { event := event79285
    frameStart := 79281 },
  { event := event79286
    frameStart := 79281 },
  { event := event79287
    frameStart := 79281 },
  { event := event79288
    frameStart := 79281 },
  { event := event79289
    frameStart := 79281 },
  { event := event79290
    frameStart := 79281 },
  { event := event79291
    frameStart := 79281 },
  { event := event79292
    frameStart := 79281 },
  { event := event79293
    frameStart := 79281 },
  { event := event79294
    frameStart := 79281 },
  { event := event79295
    frameStart := 79281 }
]

def eventLeaf4956 : Array AnnotatedEvent := #[
  { event := event79296
    frameStart := 79281 },
  { event := event79297
    frameStart := 79281 },
  { event := event79298
    frameStart := 79281 },
  { event := event79299
    frameStart := 79281 },
  { event := event79300
    frameStart := 79281 },
  { event := event79301
    frameStart := 79281 },
  { event := event79302
    frameStart := 79281 },
  { event := event79303
    frameStart := 79281 },
  { event := event79304
    frameStart := 79281 },
  { event := event79305
    frameStart := 79281 },
  { event := event79306
    frameStart := 79281 },
  { event := event79307
    frameStart := 79281 },
  { event := event79308
    frameStart := 79281 },
  { event := event79309
    frameStart := 79281 },
  { event := event79310
    frameStart := 79281 },
  { event := event79311
    frameStart := 79281 }
]

def eventLeaf4957 : Array AnnotatedEvent := #[
  { event := event79312
    frameStart := 79281 },
  { event := event79313
    frameStart := 79281 },
  { event := event79314
    frameStart := 79281 },
  { event := event79315
    frameStart := 79281 },
  { event := event79316
    frameStart := 79281 },
  { event := event79317
    frameStart := 79281 },
  { event := event79318
    frameStart := 79281 },
  { event := event79319
    frameStart := 79281 },
  { event := event79320
    frameStart := 79281 },
  { event := event79321
    frameStart := 79281 },
  { event := event79322
    frameStart := 79281 },
  { event := event79323
    frameStart := 79281 },
  { event := event79324
    frameStart := 79281 },
  { event := event79325
    frameStart := 79281 },
  { event := event79326
    frameStart := 79281 },
  { event := event79327
    frameStart := 79281 }
]

def eventLeaf4958 : Array AnnotatedEvent := #[
  { event := event79328
    frameStart := 79281 },
  { event := event79329
    frameStart := 79281 },
  { event := event79330
    frameStart := 79281 },
  { event := event79331
    frameStart := 79281 },
  { event := event79332
    frameStart := 79281 },
  { event := event79333
    frameStart := 79281 },
  { event := event79334
    frameStart := 79281 },
  { event := event79335
    frameStart := 79281 },
  { event := event79336
    frameStart := 79281 },
  { event := event79337
    frameStart := 79281 },
  { event := event79338
    frameStart := 79281 },
  { event := event79339
    frameStart := 79281 },
  { event := event79340
    frameStart := 79281 },
  { event := event79341
    frameStart := 79281 },
  { event := event79342
    frameStart := 79281 },
  { event := event79343
    frameStart := 79281 }
]

def eventLeaf4959 : Array AnnotatedEvent := #[
  { event := event79344
    frameStart := 79281 },
  { event := event79345
    frameStart := 79281 },
  { event := event79346
    frameStart := 79281 },
  { event := event79347
    frameStart := 79281 },
  { event := event79348
    frameStart := 79281 },
  { event := event79349
    frameStart := 79281 },
  { event := event79350
    frameStart := 79281 },
  { event := event79351
    frameStart := 79281 },
  { event := event79352
    frameStart := 79281 },
  { event := event79353
    frameStart := 79281 },
  { event := event79354
    frameStart := 79281 },
  { event := event79355
    frameStart := 79281 },
  { event := event79356
    frameStart := 79281 },
  { event := event79357
    frameStart := 79281 },
  { event := event79358
    frameStart := 79281 },
  { event := event79359
    frameStart := 79281 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events309
