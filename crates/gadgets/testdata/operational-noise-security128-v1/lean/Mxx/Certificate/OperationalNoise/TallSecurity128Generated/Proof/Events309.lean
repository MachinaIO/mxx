import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events309

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event79104 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event79105 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event79106 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 79105

def event79107 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 79103

def event79108 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 79106 .coefficient) (.value (.predecessor 1 79107 .coefficient)))

def event79109 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event79110 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10269⟩⟩) 0 ⟨392⟩ 79109

def event79111 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10269⟩⟩) 1 ⟨10267⟩ 79101

def event79112 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10269⟩⟩) (.sum [.predecessor 0 79110 .coefficient, .predecessor 1 79111 .coefficient])

def event79113 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10269⟩⟩) (.finite 655355)

def event79114 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10325⟩⟩) 0 ⟨10269⟩ 79113

def event79115 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10325⟩⟩) 1 ⟨5426⟩ 79099

def event79116 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10325⟩⟩) (.identity (.predecessor 1 79115 .coefficient))

def event79117 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10325⟩⟩) (.finite 655360)

def event79118 : Event := .predecessor (⟨.program ⟨257⟩, ⟨28918⟩⟩) 0 ⟨10325⟩ 79117

def event79119 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨28918⟩⟩) (.authority (.programFamilyFact))

def exact79120RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨28918⟩⟩], []⟩, (1)⟩]

theorem exact79120RawTermsValid :
    exact79120RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79120 : Event := .resultExact (⟨.program ⟨257⟩, ⟨28918⟩⟩) exact79120RawTerms (.finite 36) 79119 .exactZero (none)

def event79121 : Event := .predecessor (⟨.program ⟨257⟩, ⟨13371⟩⟩) 0 ⟨10325⟩ 79117

def event79122 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨13371⟩⟩) (.authority (.programFamilyFact))

def exact79123RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨13371⟩⟩], []⟩, (1)⟩]

theorem exact79123RawTermsValid :
    exact79123RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79123 : Event := .resultExact (⟨.program ⟨257⟩, ⟨13371⟩⟩) exact79123RawTerms (.finite 36) 79122 .exactZero (none)

def event79124 : Event := .predecessor (⟨.program ⟨257⟩, ⟨28919⟩⟩) 0 ⟨13371⟩ 79123

def event79125 : Event := .predecessor (⟨.program ⟨257⟩, ⟨28919⟩⟩) 1 ⟨28918⟩ 79120

def event79126 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨28919⟩⟩) (.product (.predecessor 0 79124 .coefficient) (.predecessor 1 79125 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event79127 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨28919⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨13371⟩⟩, ⟨.program ⟨257⟩, ⟨28918⟩⟩], []⟩) [⟨.result 79123 .coefficient, true, some 1⟩, ⟨.result 79120 .coefficient, true, some 1⟩])

def event79128 : Event := .survivorFold (1) 79127

def exact79129RawTerms : List Term := []

theorem exact79129RawTermsValid :
    exact79129RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79129 : Event := .resultExact (⟨.program ⟨257⟩, ⟨28919⟩⟩) exact79129RawTerms (.finite 1296) 79126 (.finite 1296) (some (79127))

def event79130 : Event := .predecessor (⟨.program ⟨257⟩, ⟨28920⟩⟩) 0 ⟨28919⟩ 79129

def event79131 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨28920⟩⟩) (.identity (.predecessor 0 79130 .coefficient))

def event79132 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨28920⟩⟩) (.finite 1296)

def event79133 : Event := .predecessor (⟨.program ⟨257⟩, ⟨29136⟩⟩) 0 ⟨28920⟩ 79132

def event79134 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨29136⟩⟩) (.authority (.programFamilyFact))

def exact79135RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨29136⟩⟩], []⟩, (1)⟩]

theorem exact79135RawTermsValid :
    exact79135RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79135 : Event := .resultExact (⟨.program ⟨257⟩, ⟨29136⟩⟩) exact79135RawTerms (.finite 36) 79134 .exactZero (none)

def event79136 : Event := .predecessor (⟨.program ⟨257⟩, ⟨29137⟩⟩) 0 ⟨29136⟩ 79135

def event79137 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨29137⟩⟩) (.identity (.predecessor 0 79136 .coefficient))

def event79138 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨29137⟩⟩) (.finite 36)

def event79139 : Event := .predecessor (⟨.program ⟨257⟩, ⟨29956⟩⟩) 0 ⟨29137⟩ 79138

def event79140 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨29956⟩⟩) (.authority (.relationPreimageSource ⟨81⟩))

def exact79141RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨29956⟩⟩]⟩, (1)⟩]

theorem exact79141RawTermsValid :
    exact79141RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79141 : Event := .resultExact (⟨.program ⟨257⟩, ⟨29956⟩⟩) exact79141RawTerms (.finite 5647228698) 79140 .exactZero (none)

def event79142 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35⟩⟩) (.authority (.operator))

def exact79143RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact79143RawTermsValid :
    exact79143RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79143 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35⟩⟩) exact79143RawTerms .large 79142 .exactZero (none)

def event79144 : Event := .predecessor (⟨.program ⟨257⟩, ⟨29957⟩⟩) 0 ⟨35⟩ 79143

def event79145 : Event := .predecessor (⟨.program ⟨257⟩, ⟨29957⟩⟩) 1 ⟨29956⟩ 79141

def event79146 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨29957⟩⟩) (.product (.predecessor 0 79144 .coefficient) (.predecessor 1 79145 .coefficient) (⟨false, false, none, none, none⟩))

def event79147 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨29957⟩⟩, .operator (⟨79143, 0⟩, ⟨79141, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29956⟩⟩]⟩, (1)⟩)

def exact79148RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29956⟩⟩]⟩, (1)⟩]

theorem exact79148RawTermsValid :
    exact79148RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79148 : Event := .resultExact (⟨.program ⟨257⟩, ⟨29957⟩⟩) exact79148RawTerms .large 79146 .exactZero (none)

def event79149 : Event := .preFoldPolynomial 79148 [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29956⟩⟩]⟩, (1)⟩] .exactZero none

def exact79150RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29956⟩⟩]⟩, (1)⟩]

def event79150 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨29957⟩⟩) 79149 exact79150RawTerms .large 79146 .exactZero (none)

def event79151 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨31123⟩⟩)

def event79152 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event79153 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event79154 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10267⟩⟩) (.authority (.operator))

def event79155 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10267⟩⟩) (.finite 15)

def event79156 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event79157 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event79158 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event79159 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event79160 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 79159

def event79161 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 79157

def event79162 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 79160 .coefficient) (.value (.predecessor 1 79161 .coefficient)))

def event79163 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event79164 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10269⟩⟩) 0 ⟨392⟩ 79163

def event79165 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10269⟩⟩) 1 ⟨10267⟩ 79155

def event79166 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10269⟩⟩) (.sum [.predecessor 0 79164 .coefficient, .predecessor 1 79165 .coefficient])

def event79167 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10269⟩⟩) (.finite 655355)

def event79168 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10325⟩⟩) 0 ⟨10269⟩ 79167

def event79169 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10325⟩⟩) 1 ⟨5426⟩ 79153

def event79170 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10325⟩⟩) (.identity (.predecessor 1 79169 .coefficient))

def event79171 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10325⟩⟩) (.finite 655360)

def event79172 : Event := .predecessor (⟨.program ⟨257⟩, ⟨28918⟩⟩) 0 ⟨10325⟩ 79171

def event79173 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨28918⟩⟩) (.authority (.programFamilyFact))

def exact79174RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨28918⟩⟩], []⟩, (1)⟩]

theorem exact79174RawTermsValid :
    exact79174RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79174 : Event := .resultExact (⟨.program ⟨257⟩, ⟨28918⟩⟩) exact79174RawTerms (.finite 36) 79173 .exactZero (none)

def event79175 : Event := .predecessor (⟨.program ⟨257⟩, ⟨13371⟩⟩) 0 ⟨10325⟩ 79171

def event79176 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨13371⟩⟩) (.authority (.programFamilyFact))

def exact79177RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨13371⟩⟩], []⟩, (1)⟩]

theorem exact79177RawTermsValid :
    exact79177RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79177 : Event := .resultExact (⟨.program ⟨257⟩, ⟨13371⟩⟩) exact79177RawTerms (.finite 36) 79176 .exactZero (none)

def event79178 : Event := .predecessor (⟨.program ⟨257⟩, ⟨28919⟩⟩) 0 ⟨13371⟩ 79177

def event79179 : Event := .predecessor (⟨.program ⟨257⟩, ⟨28919⟩⟩) 1 ⟨28918⟩ 79174

def event79180 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨28919⟩⟩) (.product (.predecessor 0 79178 .coefficient) (.predecessor 1 79179 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event79181 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨28919⟩⟩, .operator (⟨79177, 0⟩, ⟨79174, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨13371⟩⟩, ⟨.program ⟨257⟩, ⟨28918⟩⟩], []⟩, (1)⟩)

def exact79182RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨13371⟩⟩, ⟨.program ⟨257⟩, ⟨28918⟩⟩], []⟩, (1)⟩]

theorem exact79182RawTermsValid :
    exact79182RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79182 : Event := .resultExact (⟨.program ⟨257⟩, ⟨28919⟩⟩) exact79182RawTerms (.finite 1296) 79180 .exactZero (none)

def event79183 : Event := .predecessor (⟨.program ⟨257⟩, ⟨28920⟩⟩) 0 ⟨28919⟩ 79182

def event79184 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨28920⟩⟩) (.identity (.predecessor 0 79183 .coefficient))

def event79185 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨28920⟩⟩) (.finite 1296)

def event79186 : Event := .predecessor (⟨.program ⟨257⟩, ⟨29136⟩⟩) 0 ⟨28920⟩ 79185

def event79187 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨29136⟩⟩) (.authority (.programFamilyFact))

def exact79188RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨29136⟩⟩], []⟩, (1)⟩]

theorem exact79188RawTermsValid :
    exact79188RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79188 : Event := .resultExact (⟨.program ⟨257⟩, ⟨29136⟩⟩) exact79188RawTerms (.finite 36) 79187 .exactZero (none)

def event79189 : Event := .predecessor (⟨.program ⟨257⟩, ⟨29137⟩⟩) 0 ⟨29136⟩ 79188

def event79190 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨29137⟩⟩) (.identity (.predecessor 0 79189 .coefficient))

def event79191 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨29137⟩⟩) (.finite 36)

def event79192 : Event := .predecessor (⟨.program ⟨257⟩, ⟨30293⟩⟩) 0 ⟨29137⟩ 79191

def event79193 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨30293⟩⟩) (.authority (.programFamilyFact))

def event79194 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨30293⟩⟩) (.finite 3720)

def event79195 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨7177⟩⟩) .missing

def event79196 : Event := .predecessor (⟨.program ⟨257⟩, ⟨30295⟩⟩) 0 ⟨7177⟩ 79195

def event79197 : Event := .predecessor (⟨.program ⟨257⟩, ⟨30295⟩⟩) 1 ⟨30293⟩ 79194

def event79198 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨30295⟩⟩) (.authority (.operator))

def exact79199RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨30295⟩⟩]⟩, (1)⟩]

theorem exact79199RawTermsValid :
    exact79199RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79199 : Event := .resultExact (⟨.program ⟨257⟩, ⟨30295⟩⟩) exact79199RawTerms .large 79198 .exactZero (none)

def event79200 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31119⟩⟩) 0 ⟨30295⟩ 79199

def event79201 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31119⟩⟩) (.authority (.operator))

def exact79202RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨31119⟩⟩]⟩, (1)⟩]

theorem exact79202RawTermsValid :
    exact79202RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79202 : Event := .resultExact (⟨.program ⟨257⟩, ⟨31119⟩⟩) exact79202RawTerms (.finite 8192) 79201 .exactZero (none)

def event79203 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event79204 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event79205 : Event := .predecessor (⟨.program ⟨257⟩, ⟨30470⟩⟩) 0 ⟨29137⟩ 79191

def event79206 : Event := .predecessor (⟨.program ⟨257⟩, ⟨30470⟩⟩) 1 ⟨136⟩ 79204

def event79207 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨30470⟩⟩) (.sum [.predecessor 0 79205 .coefficient, .predecessor 1 79206 .coefficient])

def event79208 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨30470⟩⟩) (.finite 36)

def event79209 : Event := .predecessor (⟨.program ⟨257⟩, ⟨30471⟩⟩) 0 ⟨30470⟩ 79208

def event79210 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨30471⟩⟩) (.identity (.predecessor 0 79209 .coefficient))

def exact79211RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨29136⟩⟩], []⟩, (1)⟩]

theorem exact79211RawTermsValid :
    exact79211RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79211 : Event := .resultExact (⟨.program ⟨257⟩, ⟨30471⟩⟩) exact79211RawTerms (.finite 36) 79210 .exactZero (none)

def event79212 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact79213RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact79213RawTermsValid :
    exact79213RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79213 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact79213RawTerms .large 79212 .exactZero (none)

def event79214 : Event := .predecessor (⟨.program ⟨257⟩, ⟨30472⟩⟩) 0 ⟨6908⟩ 79213

def event79215 : Event := .predecessor (⟨.program ⟨257⟩, ⟨30472⟩⟩) 1 ⟨30471⟩ 79211

def event79216 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨30472⟩⟩) (.product (.predecessor 0 79214 .coefficient) (.predecessor 1 79215 .coefficient) (⟨false, false, none, none, none⟩))

def event79217 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨30472⟩⟩, .operator (⟨79213, 0⟩, ⟨79211, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨29136⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact79218RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨29136⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact79218RawTermsValid :
    exact79218RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79218 : Event := .resultExact (⟨.program ⟨257⟩, ⟨30472⟩⟩) exact79218RawTerms .large 79216 .exactZero (none)

def event79219 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7190⟩⟩) 0 ⟨7177⟩ 79195

def event79220 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7190⟩⟩) (.authority (.operator))

def exact79221RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7190⟩⟩]⟩, (1)⟩]

theorem exact79221RawTermsValid :
    exact79221RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79221 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7190⟩⟩) exact79221RawTerms .large 79220 .exactZero (none)

def event79222 : Event := .predecessor (⟨.program ⟨257⟩, ⟨30473⟩⟩) 0 ⟨7190⟩ 79221

def event79223 : Event := .predecessor (⟨.program ⟨257⟩, ⟨30473⟩⟩) 1 ⟨30472⟩ 79218

def event79224 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨30473⟩⟩) (.sum [.predecessor 0 79222 .coefficient, .predecessor 1 79223 .coefficient])

def exact79225RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7190⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨29136⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact79225RawTermsValid :
    exact79225RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79225 : Event := .resultExact (⟨.program ⟨257⟩, ⟨30473⟩⟩) exact79225RawTerms .large 79224 .exactZero (none)

def event79226 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31120⟩⟩) 0 ⟨30473⟩ 79225

def event79227 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31120⟩⟩) 1 ⟨31119⟩ 79202

def event79228 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31120⟩⟩) (.product (.predecessor 0 79226 .coefficient) (.predecessor 1 79227 .coefficient) (⟨false, false, none, none, none⟩))

def event79229 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨31120⟩⟩, .operator (⟨79225, 0⟩, ⟨79202, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨31119⟩⟩]⟩, (1)⟩)

def event79230 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨31120⟩⟩, .operator (⟨79225, 1⟩, ⟨79202, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨29136⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨31119⟩⟩]⟩, (-1)⟩)

def event79231 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨31120⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨29136⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨31119⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨31119⟩⟩) ⟨30295⟩ 79199)

def event79232 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨31120⟩⟩, .relation 79231 0, ⟨[⟨.program ⟨257⟩, ⟨29136⟩⟩], [⟨.program ⟨257⟩, ⟨30295⟩⟩]⟩, (-1)⟩)

def exact79233RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨31119⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨29136⟩⟩], [⟨.program ⟨257⟩, ⟨30295⟩⟩]⟩, (-1)⟩]

theorem exact79233RawTermsValid :
    exact79233RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79233 : Event := .resultExact (⟨.program ⟨257⟩, ⟨31120⟩⟩) exact79233RawTerms .large 79228 .exactZero (none)

def event79234 : Event := .predecessor (⟨.program ⟨257⟩, ⟨29377⟩⟩) 0 ⟨29137⟩ 79191

def event79235 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨29377⟩⟩) (.authority (.programFamilyFact))

def exact79236RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨29377⟩⟩], []⟩, (1)⟩]

theorem exact79236RawTermsValid :
    exact79236RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79236 : Event := .resultExact (⟨.program ⟨257⟩, ⟨29377⟩⟩) exact79236RawTerms (.finite 62) 79235 .exactZero (none)

def event79237 : Event := .predecessor (⟨.program ⟨257⟩, ⟨29378⟩⟩) 0 ⟨6908⟩ 79213

def event79238 : Event := .predecessor (⟨.program ⟨257⟩, ⟨29378⟩⟩) 1 ⟨29377⟩ 79236

def event79239 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨29378⟩⟩) (.product (.predecessor 0 79237 .coefficient) (.predecessor 1 79238 .coefficient) (⟨false, true, none, none, some 1⟩))

def event79240 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨29378⟩⟩, .operator (⟨79213, 0⟩, ⟨79236, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨29377⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact79241RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨29377⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact79241RawTermsValid :
    exact79241RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79241 : Event := .resultExact (⟨.program ⟨257⟩, ⟨29378⟩⟩) exact79241RawTerms .large 79239 .exactZero (none)

def event79242 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7220⟩⟩) 0 ⟨7177⟩ 79195

def event79243 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7220⟩⟩) (.authority (.operator))

def exact79244RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩]

theorem exact79244RawTermsValid :
    exact79244RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79244 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7220⟩⟩) exact79244RawTerms .large 79243 .exactZero (none)

def event79245 : Event := .predecessor (⟨.program ⟨257⟩, ⟨29379⟩⟩) 0 ⟨7220⟩ 79244

def event79246 : Event := .predecessor (⟨.program ⟨257⟩, ⟨29379⟩⟩) 1 ⟨29378⟩ 79241

def event79247 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨29379⟩⟩) (.sum [.predecessor 0 79245 .coefficient, .predecessor 1 79246 .coefficient])

def exact79248RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨29377⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact79248RawTermsValid :
    exact79248RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79248 : Event := .resultExact (⟨.program ⟨257⟩, ⟨29379⟩⟩) exact79248RawTerms .large 79247 .exactZero (none)

def event79249 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31123⟩⟩) 0 ⟨29379⟩ 79248

def event79250 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31123⟩⟩) 1 ⟨31120⟩ 79233

def event79251 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31123⟩⟩) (.sum [.predecessor 0 79249 .coefficient, .predecessor 1 79250 .coefficient])

def exact79252RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨31119⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨29136⟩⟩], [⟨.program ⟨257⟩, ⟨30295⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨29377⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact79252RawTermsValid :
    exact79252RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79252 : Event := .resultExact (⟨.program ⟨257⟩, ⟨31123⟩⟩) exact79252RawTerms .large 79251 .exactZero (none)

def event79253 : Event := .preFoldPolynomial 79252 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨31119⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨29136⟩⟩], [⟨.program ⟨257⟩, ⟨30295⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨29377⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact79254RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨31119⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨29136⟩⟩], [⟨.program ⟨257⟩, ⟨30295⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨29377⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event79254 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨31123⟩⟩) 79253 exact79254RawTerms .large 79251 .exactZero (none)

def event79255 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨29137⟩⟩) ⟨⟨99⟩, ⟨81⟩, ⟨135⟩⟩ ⟨79097, 79255⟩

def event79256 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨29959⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29956⟩⟩]⟩) (1) 0 2 (.universal 79255 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29956⟩⟩]⟩) (none) 79254)

def event79257 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨29959⟩⟩, .relation 79256 1, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩)

def event79258 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨29959⟩⟩, .relation 79256 0, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨31119⟩⟩]⟩, (-1)⟩)

def event79259 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨29959⟩⟩, .relation 79256 2, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨29136⟩⟩], [⟨.program ⟨257⟩, ⟨30295⟩⟩]⟩, (1)⟩)

def event79260 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨29959⟩⟩, .relation 79256 3, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨29377⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact79261RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨31119⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨29136⟩⟩], [⟨.program ⟨257⟩, ⟨30295⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨29377⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact79261RawTermsValid :
    exact79261RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79261 : Event := .resultExact (⟨.program ⟨257⟩, ⟨29959⟩⟩) exact79261RawTerms .large 79093 (.finite 202072841853861888) (some (79095))

def event79262 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31122⟩⟩) 0 ⟨29959⟩ 79261

def event79263 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31122⟩⟩) 1 ⟨31121⟩ 79083

def event79264 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31122⟩⟩) (.sum [.predecessor 0 79262 .coefficient, .predecessor 1 79263 .coefficient])

def event79265 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨31122⟩⟩, .operator (⟨79261, 0⟩, ⟨79083, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨31119⟩⟩]⟩, (1)⟩)

def event79266 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨31122⟩⟩, .operator (⟨79261, 2⟩, ⟨79083, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨29136⟩⟩], [⟨.program ⟨257⟩, ⟨30295⟩⟩]⟩, (-1)⟩)

def event79267 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31122⟩⟩) (.sum [.result 79261 .summary, .result 79083 .summary])

def exact79268RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨29377⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact79268RawTermsValid :
    exact79268RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79268 : Event := .resultExact (⟨.program ⟨257⟩, ⟨31122⟩⟩) exact79268RawTerms .large 79264 (.finite 32192146870060392302605751287808) (some (79267))

def event79269 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27613⟩⟩) 0 ⟨26457⟩ 3264

def event79270 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨27613⟩⟩) (.authority (.programFamilyFact))

def event79271 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨27613⟩⟩) (.finite 3720)

def event79272 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27615⟩⟩) 0 ⟨7177⟩ 15500

def event79273 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27615⟩⟩) 1 ⟨27613⟩ 79271

def event79274 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨27615⟩⟩) (.authority (.operator))

def exact79275RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨27615⟩⟩]⟩, (1)⟩]

theorem exact79275RawTermsValid :
    exact79275RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79275 : Event := .resultExact (⟨.program ⟨257⟩, ⟨27615⟩⟩) exact79275RawTerms .large 79274 .exactZero (none)

def event79276 : Event := .predecessor (⟨.program ⟨257⟩, ⟨28439⟩⟩) 0 ⟨27615⟩ 79275

def event79277 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨28439⟩⟩) (.authority (.operator))

def exact79278RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨28439⟩⟩]⟩, (1)⟩]

theorem exact79278RawTermsValid :
    exact79278RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79278 : Event := .resultExact (⟨.program ⟨257⟩, ⟨28439⟩⟩) exact79278RawTerms (.finite 8192) 79277 .exactZero (none)

def event79279 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27444⟩⟩) 0 ⟨26240⟩ 3258

def event79280 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨27444⟩⟩) (.authority (.programFamilyFact))

def event79281 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨27444⟩⟩) (.finite 3720)

def event79282 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27445⟩⟩) 0 ⟨7177⟩ 15500

def event79283 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27445⟩⟩) 1 ⟨27444⟩ 79281

def event79284 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨27445⟩⟩) (.authority (.operator))

def exact79285RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨27445⟩⟩]⟩, (1)⟩]

theorem exact79285RawTermsValid :
    exact79285RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79285 : Event := .resultExact (⟨.program ⟨257⟩, ⟨27445⟩⟩) exact79285RawTerms .large 79284 .exactZero (none)

def event79286 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27985⟩⟩) 0 ⟨27445⟩ 79285

def event79287 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨27985⟩⟩) (.authority (.operator))

def exact79288RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨27985⟩⟩]⟩, (1)⟩]

theorem exact79288RawTermsValid :
    exact79288RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79288 : Event := .resultExact (⟨.program ⟨257⟩, ⟨27985⟩⟩) exact79288RawTerms (.finite 8192) 79287 .exactZero (none)

def event79289 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26241⟩⟩) 0 ⟨26238⟩ 3247

def event79290 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26241⟩⟩) 1 ⟨10328⟩ 75903

def event79291 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨26241⟩⟩) (.tensor (.predecessor 0 79289 .coefficient) (.predecessor 1 79290 .coefficient) true false)

def event79292 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨26241⟩⟩, .operator (⟨3247, 0⟩, ⟨75903, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨26238⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact79293RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨26238⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact79293RawTermsValid :
    exact79293RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79293 : Event := .resultExact (⟨.program ⟨257⟩, ⟨26241⟩⟩) exact79293RawTerms .large 79291 .exactZero (none)

def event79294 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10336⟩⟩) 0 ⟨10327⟩ 75773

def event79295 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10336⟩⟩) 1 ⟨7278⟩ 20587

def event79296 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10336⟩⟩) (.product (.predecessor 0 79294 .coefficient) (.predecessor 1 79295 .coefficient) (⟨false, false, none, none, none⟩))

def event79297 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨10336⟩⟩, .operator (⟨75773, 0⟩, ⟨20587, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7278⟩⟩]⟩, (1)⟩)

def exact79298RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7278⟩⟩]⟩, (1)⟩]

theorem exact79298RawTermsValid :
    exact79298RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79298 : Event := .resultExact (⟨.program ⟨257⟩, ⟨10336⟩⟩) exact79298RawTerms .large 79296 .exactZero (none)

def event79299 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26242⟩⟩) 0 ⟨10336⟩ 79298

def event79300 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26242⟩⟩) 1 ⟨26241⟩ 79293

def event79301 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨26242⟩⟩) (.sum [.predecessor 0 79299 .coefficient, .predecessor 1 79300 .coefficient])

def exact79302RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7278⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨26238⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact79302RawTermsValid :
    exact79302RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79302 : Event := .resultExact (⟨.program ⟨257⟩, ⟨26242⟩⟩) exact79302RawTerms .large 79301 .exactZero (none)

def event79303 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26243⟩⟩) 0 ⟨26242⟩ 79302

def event79304 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26243⟩⟩) 1 ⟨104⟩ 20579

def event79305 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨26243⟩⟩) (.sum [.predecessor 0 79303 .coefficient, .predecessor 1 79304 .coefficient])

def event79306 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨26243⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨104⟩⟩]⟩) [⟨.result 20579 .coefficient, false, none⟩])

def event79307 : Event := .survivorFold (1) 79306

def exact79308RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7278⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨26238⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact79308RawTermsValid :
    exact79308RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79308 : Event := .resultExact (⟨.program ⟨257⟩, ⟨26243⟩⟩) exact79308RawTerms .large 79305 (.finite 26) (some (79306))

def event79309 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26244⟩⟩) 0 ⟨26243⟩ 79308

def event79310 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26244⟩⟩) 1 ⟨13071⟩ 3250

def event79311 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨26244⟩⟩) (.product (.predecessor 0 79309 .coefficient) (.predecessor 1 79310 .coefficient) (⟨false, true, none, none, some 1⟩))

def event79312 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨26244⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨13071⟩⟩], []⟩) [⟨.result 3250 .coefficient, true, some 1⟩])

def event79313 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨26244⟩⟩) (.product (.result 79308 .summary) (.transfer 79312) (⟨false, false, none, none, none⟩))

def event79314 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨26244⟩⟩, .operator (⟨79308, 1⟩, ⟨3250, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨13071⟩⟩, ⟨.program ⟨257⟩, ⟨26238⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event79315 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨26244⟩⟩, .operator (⟨79308, 0⟩, ⟨3250, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨13071⟩⟩], [⟨.program ⟨257⟩, ⟨7278⟩⟩]⟩, (1)⟩)

def exact79316RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨13071⟩⟩], [⟨.program ⟨257⟩, ⟨7278⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨13071⟩⟩, ⟨.program ⟨257⟩, ⟨26238⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact79316RawTermsValid :
    exact79316RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79316 : Event := .resultExact (⟨.program ⟨257⟩, ⟨26244⟩⟩) exact79316RawTerms .large 79311 (.finite 25559040) (some (79313))

def event79317 : Event := .predecessor (⟨.program ⟨257⟩, ⟨13072⟩⟩) 0 ⟨13071⟩ 3250

def event79318 : Event := .predecessor (⟨.program ⟨257⟩, ⟨13072⟩⟩) 1 ⟨10328⟩ 75903

def event79319 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨13072⟩⟩) (.tensor (.predecessor 0 79317 .coefficient) (.predecessor 1 79318 .coefficient) true false)

def event79320 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨13072⟩⟩, .operator (⟨3250, 0⟩, ⟨75903, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨13071⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact79321RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨13071⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact79321RawTermsValid :
    exact79321RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79321 : Event := .resultExact (⟨.program ⟨257⟩, ⟨13072⟩⟩) exact79321RawTerms .large 79319 .exactZero (none)

def event79322 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10353⟩⟩) 0 ⟨10327⟩ 75773

def event79323 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10353⟩⟩) 1 ⟨7295⟩ 20628

def event79324 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10353⟩⟩) (.product (.predecessor 0 79322 .coefficient) (.predecessor 1 79323 .coefficient) (⟨false, false, none, none, none⟩))

def event79325 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨10353⟩⟩, .operator (⟨75773, 0⟩, ⟨20628, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7295⟩⟩]⟩, (1)⟩)

def exact79326RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7295⟩⟩]⟩, (1)⟩]

theorem exact79326RawTermsValid :
    exact79326RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79326 : Event := .resultExact (⟨.program ⟨257⟩, ⟨10353⟩⟩) exact79326RawTerms .large 79324 .exactZero (none)

def event79327 : Event := .predecessor (⟨.program ⟨257⟩, ⟨13073⟩⟩) 0 ⟨10353⟩ 79326

def event79328 : Event := .predecessor (⟨.program ⟨257⟩, ⟨13073⟩⟩) 1 ⟨13072⟩ 79321

def event79329 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨13073⟩⟩) (.sum [.predecessor 0 79327 .coefficient, .predecessor 1 79328 .coefficient])

def exact79330RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7295⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨13071⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact79330RawTermsValid :
    exact79330RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79330 : Event := .resultExact (⟨.program ⟨257⟩, ⟨13073⟩⟩) exact79330RawTerms .large 79329 .exactZero (none)

def event79331 : Event := .predecessor (⟨.program ⟨257⟩, ⟨13074⟩⟩) 0 ⟨13073⟩ 79330

def event79332 : Event := .predecessor (⟨.program ⟨257⟩, ⟨13074⟩⟩) 1 ⟨121⟩ 20620

def event79333 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨13074⟩⟩) (.sum [.predecessor 0 79331 .coefficient, .predecessor 1 79332 .coefficient])

def event79334 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨13074⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨121⟩⟩]⟩) [⟨.result 20620 .coefficient, false, none⟩])

def event79335 : Event := .survivorFold (1) 79334

def exact79336RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7295⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨13071⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact79336RawTermsValid :
    exact79336RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79336 : Event := .resultExact (⟨.program ⟨257⟩, ⟨13074⟩⟩) exact79336RawTerms .large 79333 (.finite 26) (some (79334))

def event79337 : Event := .predecessor (⟨.program ⟨257⟩, ⟨13075⟩⟩) 0 ⟨13074⟩ 79336

def event79338 : Event := .predecessor (⟨.program ⟨257⟩, ⟨13075⟩⟩) 1 ⟨9545⟩ 20617

def event79339 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨13075⟩⟩) (.product (.predecessor 0 79337 .coefficient) (.predecessor 1 79338 .coefficient) (⟨false, false, none, none, none⟩))

def event79340 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨13075⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨9544⟩⟩]⟩) [⟨.result 20613 .coefficient, false, none⟩])

def event79341 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨13075⟩⟩) (.product (.result 79336 .summary) (.transfer 79340) (⟨false, false, none, none, none⟩))

def event79342 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨13075⟩⟩, .operator (⟨79336, 1⟩, ⟨20617, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨13071⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩]⟩, (-1)⟩)

def event79343 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨13075⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨13071⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨9544⟩⟩) ⟨7278⟩ 20587)

def event79344 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨13075⟩⟩, .relation 79343 0, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨13071⟩⟩], [⟨.program ⟨257⟩, ⟨7278⟩⟩]⟩, (-1)⟩)

def event79345 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨13075⟩⟩, .operator (⟨79336, 0⟩, ⟨20617, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩]⟩, (1)⟩)

def exact79346RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨13071⟩⟩], [⟨.program ⟨257⟩, ⟨7278⟩⟩]⟩, (-1)⟩]

theorem exact79346RawTermsValid :
    exact79346RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79346 : Event := .resultExact (⟨.program ⟨257⟩, ⟨13075⟩⟩) exact79346RawTerms .large 79339 (.finite 279172874240) (some (79341))

def event79347 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26245⟩⟩) 0 ⟨13075⟩ 79346

def event79348 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26245⟩⟩) 1 ⟨26244⟩ 79316

def event79349 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨26245⟩⟩) (.sum [.predecessor 0 79347 .coefficient, .predecessor 1 79348 .coefficient])

def event79350 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨26245⟩⟩, .operator (⟨79346, 1⟩, ⟨79316, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨13071⟩⟩], [⟨.program ⟨257⟩, ⟨7278⟩⟩]⟩, (1)⟩)

def event79351 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨26245⟩⟩) (.sum [.result 79346 .summary, .result 79316 .summary])

def exact79352RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨13071⟩⟩, ⟨.program ⟨257⟩, ⟨26238⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact79352RawTermsValid :
    exact79352RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79352 : Event := .resultExact (⟨.program ⟨257⟩, ⟨26245⟩⟩) exact79352RawTerms .large 79349 (.finite 279198433280) (some (79351))

def event79353 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27986⟩⟩) 0 ⟨26245⟩ 79352

def event79354 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27986⟩⟩) 1 ⟨27985⟩ 79288

def event79355 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨27986⟩⟩) (.product (.predecessor 0 79353 .coefficient) (.predecessor 1 79354 .coefficient) (⟨false, false, none, none, none⟩))

def event79356 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨27986⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨27985⟩⟩]⟩) [⟨.result 79288 .coefficient, false, none⟩])

def event79357 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨27986⟩⟩) (.product (.result 79352 .summary) (.transfer 79356) (⟨false, false, none, none, none⟩))

def event79358 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨27986⟩⟩, .operator (⟨79352, 1⟩, ⟨79288, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨13071⟩⟩, ⟨.program ⟨257⟩, ⟨26238⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨27985⟩⟩]⟩, (-1)⟩)

def event79359 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨27986⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨13071⟩⟩, ⟨.program ⟨257⟩, ⟨26238⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨27985⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨27985⟩⟩) ⟨27445⟩ 79285)

def eventLeaf4944 : Array AnnotatedEvent := #[
  { event := event79104
    frameStart := 79097 },
  { event := event79105
    frameStart := 79097 },
  { event := event79106
    frameStart := 79097 },
  { event := event79107
    frameStart := 79097 },
  { event := event79108
    frameStart := 79097 },
  { event := event79109
    frameStart := 79097 },
  { event := event79110
    frameStart := 79097 },
  { event := event79111
    frameStart := 79097 },
  { event := event79112
    frameStart := 79097 },
  { event := event79113
    frameStart := 79097 },
  { event := event79114
    frameStart := 79097 },
  { event := event79115
    frameStart := 79097 },
  { event := event79116
    frameStart := 79097 },
  { event := event79117
    frameStart := 79097 },
  { event := event79118
    frameStart := 79097 },
  { event := event79119
    frameStart := 79097 }
]

def eventLeaf4945 : Array AnnotatedEvent := #[
  { event := event79120
    frameStart := 79097 },
  { event := event79121
    frameStart := 79097 },
  { event := event79122
    frameStart := 79097 },
  { event := event79123
    frameStart := 79097 },
  { event := event79124
    frameStart := 79097 },
  { event := event79125
    frameStart := 79097 },
  { event := event79126
    frameStart := 79097 },
  { event := event79127
    frameStart := 79097 },
  { event := event79128
    frameStart := 79097 },
  { event := event79129
    frameStart := 79097 },
  { event := event79130
    frameStart := 79097 },
  { event := event79131
    frameStart := 79097 },
  { event := event79132
    frameStart := 79097 },
  { event := event79133
    frameStart := 79097 },
  { event := event79134
    frameStart := 79097 },
  { event := event79135
    frameStart := 79097 }
]

def eventLeaf4946 : Array AnnotatedEvent := #[
  { event := event79136
    frameStart := 79097 },
  { event := event79137
    frameStart := 79097 },
  { event := event79138
    frameStart := 79097 },
  { event := event79139
    frameStart := 79097 },
  { event := event79140
    frameStart := 79097 },
  { event := event79141
    frameStart := 79097 },
  { event := event79142
    frameStart := 79097 },
  { event := event79143
    frameStart := 79097 },
  { event := event79144
    frameStart := 79097 },
  { event := event79145
    frameStart := 79097 },
  { event := event79146
    frameStart := 79097 },
  { event := event79147
    frameStart := 79097 },
  { event := event79148
    frameStart := 79097 },
  { event := event79149
    frameStart := 79097 },
  { event := event79150
    frameStart := 79097 },
  { event := event79151
    frameStart := 79151 }
]

def eventLeaf4947 : Array AnnotatedEvent := #[
  { event := event79152
    frameStart := 79151 },
  { event := event79153
    frameStart := 79151 },
  { event := event79154
    frameStart := 79151 },
  { event := event79155
    frameStart := 79151 },
  { event := event79156
    frameStart := 79151 },
  { event := event79157
    frameStart := 79151 },
  { event := event79158
    frameStart := 79151 },
  { event := event79159
    frameStart := 79151 },
  { event := event79160
    frameStart := 79151 },
  { event := event79161
    frameStart := 79151 },
  { event := event79162
    frameStart := 79151 },
  { event := event79163
    frameStart := 79151 },
  { event := event79164
    frameStart := 79151 },
  { event := event79165
    frameStart := 79151 },
  { event := event79166
    frameStart := 79151 },
  { event := event79167
    frameStart := 79151 }
]

def eventLeaf4948 : Array AnnotatedEvent := #[
  { event := event79168
    frameStart := 79151 },
  { event := event79169
    frameStart := 79151 },
  { event := event79170
    frameStart := 79151 },
  { event := event79171
    frameStart := 79151 },
  { event := event79172
    frameStart := 79151 },
  { event := event79173
    frameStart := 79151 },
  { event := event79174
    frameStart := 79151 },
  { event := event79175
    frameStart := 79151 },
  { event := event79176
    frameStart := 79151 },
  { event := event79177
    frameStart := 79151 },
  { event := event79178
    frameStart := 79151 },
  { event := event79179
    frameStart := 79151 },
  { event := event79180
    frameStart := 79151 },
  { event := event79181
    frameStart := 79151 },
  { event := event79182
    frameStart := 79151 },
  { event := event79183
    frameStart := 79151 }
]

def eventLeaf4949 : Array AnnotatedEvent := #[
  { event := event79184
    frameStart := 79151 },
  { event := event79185
    frameStart := 79151 },
  { event := event79186
    frameStart := 79151 },
  { event := event79187
    frameStart := 79151 },
  { event := event79188
    frameStart := 79151 },
  { event := event79189
    frameStart := 79151 },
  { event := event79190
    frameStart := 79151 },
  { event := event79191
    frameStart := 79151 },
  { event := event79192
    frameStart := 79151 },
  { event := event79193
    frameStart := 79151 },
  { event := event79194
    frameStart := 79151 },
  { event := event79195
    frameStart := 79151 },
  { event := event79196
    frameStart := 79151 },
  { event := event79197
    frameStart := 79151 },
  { event := event79198
    frameStart := 79151 },
  { event := event79199
    frameStart := 79151 }
]

def eventLeaf4950 : Array AnnotatedEvent := #[
  { event := event79200
    frameStart := 79151 },
  { event := event79201
    frameStart := 79151 },
  { event := event79202
    frameStart := 79151 },
  { event := event79203
    frameStart := 79151 },
  { event := event79204
    frameStart := 79151 },
  { event := event79205
    frameStart := 79151 },
  { event := event79206
    frameStart := 79151 },
  { event := event79207
    frameStart := 79151 },
  { event := event79208
    frameStart := 79151 },
  { event := event79209
    frameStart := 79151 },
  { event := event79210
    frameStart := 79151 },
  { event := event79211
    frameStart := 79151 },
  { event := event79212
    frameStart := 79151 },
  { event := event79213
    frameStart := 79151 },
  { event := event79214
    frameStart := 79151 },
  { event := event79215
    frameStart := 79151 }
]

def eventLeaf4951 : Array AnnotatedEvent := #[
  { event := event79216
    frameStart := 79151 },
  { event := event79217
    frameStart := 79151 },
  { event := event79218
    frameStart := 79151 },
  { event := event79219
    frameStart := 79151 },
  { event := event79220
    frameStart := 79151 },
  { event := event79221
    frameStart := 79151 },
  { event := event79222
    frameStart := 79151 },
  { event := event79223
    frameStart := 79151 },
  { event := event79224
    frameStart := 79151 },
  { event := event79225
    frameStart := 79151 },
  { event := event79226
    frameStart := 79151 },
  { event := event79227
    frameStart := 79151 },
  { event := event79228
    frameStart := 79151 },
  { event := event79229
    frameStart := 79151 },
  { event := event79230
    frameStart := 79151 },
  { event := event79231
    frameStart := 79151 }
]

def eventLeaf4952 : Array AnnotatedEvent := #[
  { event := event79232
    frameStart := 79151 },
  { event := event79233
    frameStart := 79151 },
  { event := event79234
    frameStart := 79151 },
  { event := event79235
    frameStart := 79151 },
  { event := event79236
    frameStart := 79151 },
  { event := event79237
    frameStart := 79151 },
  { event := event79238
    frameStart := 79151 },
  { event := event79239
    frameStart := 79151 },
  { event := event79240
    frameStart := 79151 },
  { event := event79241
    frameStart := 79151 },
  { event := event79242
    frameStart := 79151 },
  { event := event79243
    frameStart := 79151 },
  { event := event79244
    frameStart := 79151 },
  { event := event79245
    frameStart := 79151 },
  { event := event79246
    frameStart := 79151 },
  { event := event79247
    frameStart := 79151 }
]

def eventLeaf4953 : Array AnnotatedEvent := #[
  { event := event79248
    frameStart := 79151 },
  { event := event79249
    frameStart := 79151 },
  { event := event79250
    frameStart := 79151 },
  { event := event79251
    frameStart := 79151 },
  { event := event79252
    frameStart := 79151 },
  { event := event79253
    frameStart := 79151 },
  { event := event79254
    frameStart := 79151 },
  { event := event79255
    frameStart := 0 },
  { event := event79256
    frameStart := 0 },
  { event := event79257
    frameStart := 0 },
  { event := event79258
    frameStart := 0 },
  { event := event79259
    frameStart := 0 },
  { event := event79260
    frameStart := 0 },
  { event := event79261
    frameStart := 0 },
  { event := event79262
    frameStart := 0 },
  { event := event79263
    frameStart := 0 }
]

def eventLeaf4954 : Array AnnotatedEvent := #[
  { event := event79264
    frameStart := 0 },
  { event := event79265
    frameStart := 0 },
  { event := event79266
    frameStart := 0 },
  { event := event79267
    frameStart := 0 },
  { event := event79268
    frameStart := 0 },
  { event := event79269
    frameStart := 0 },
  { event := event79270
    frameStart := 0 },
  { event := event79271
    frameStart := 0 },
  { event := event79272
    frameStart := 0 },
  { event := event79273
    frameStart := 0 },
  { event := event79274
    frameStart := 0 },
  { event := event79275
    frameStart := 0 },
  { event := event79276
    frameStart := 0 },
  { event := event79277
    frameStart := 0 },
  { event := event79278
    frameStart := 0 },
  { event := event79279
    frameStart := 0 }
]

def eventLeaf4955 : Array AnnotatedEvent := #[
  { event := event79280
    frameStart := 0 },
  { event := event79281
    frameStart := 0 },
  { event := event79282
    frameStart := 0 },
  { event := event79283
    frameStart := 0 },
  { event := event79284
    frameStart := 0 },
  { event := event79285
    frameStart := 0 },
  { event := event79286
    frameStart := 0 },
  { event := event79287
    frameStart := 0 },
  { event := event79288
    frameStart := 0 },
  { event := event79289
    frameStart := 0 },
  { event := event79290
    frameStart := 0 },
  { event := event79291
    frameStart := 0 },
  { event := event79292
    frameStart := 0 },
  { event := event79293
    frameStart := 0 },
  { event := event79294
    frameStart := 0 },
  { event := event79295
    frameStart := 0 }
]

def eventLeaf4956 : Array AnnotatedEvent := #[
  { event := event79296
    frameStart := 0 },
  { event := event79297
    frameStart := 0 },
  { event := event79298
    frameStart := 0 },
  { event := event79299
    frameStart := 0 },
  { event := event79300
    frameStart := 0 },
  { event := event79301
    frameStart := 0 },
  { event := event79302
    frameStart := 0 },
  { event := event79303
    frameStart := 0 },
  { event := event79304
    frameStart := 0 },
  { event := event79305
    frameStart := 0 },
  { event := event79306
    frameStart := 0 },
  { event := event79307
    frameStart := 0 },
  { event := event79308
    frameStart := 0 },
  { event := event79309
    frameStart := 0 },
  { event := event79310
    frameStart := 0 },
  { event := event79311
    frameStart := 0 }
]

def eventLeaf4957 : Array AnnotatedEvent := #[
  { event := event79312
    frameStart := 0 },
  { event := event79313
    frameStart := 0 },
  { event := event79314
    frameStart := 0 },
  { event := event79315
    frameStart := 0 },
  { event := event79316
    frameStart := 0 },
  { event := event79317
    frameStart := 0 },
  { event := event79318
    frameStart := 0 },
  { event := event79319
    frameStart := 0 },
  { event := event79320
    frameStart := 0 },
  { event := event79321
    frameStart := 0 },
  { event := event79322
    frameStart := 0 },
  { event := event79323
    frameStart := 0 },
  { event := event79324
    frameStart := 0 },
  { event := event79325
    frameStart := 0 },
  { event := event79326
    frameStart := 0 },
  { event := event79327
    frameStart := 0 }
]

def eventLeaf4958 : Array AnnotatedEvent := #[
  { event := event79328
    frameStart := 0 },
  { event := event79329
    frameStart := 0 },
  { event := event79330
    frameStart := 0 },
  { event := event79331
    frameStart := 0 },
  { event := event79332
    frameStart := 0 },
  { event := event79333
    frameStart := 0 },
  { event := event79334
    frameStart := 0 },
  { event := event79335
    frameStart := 0 },
  { event := event79336
    frameStart := 0 },
  { event := event79337
    frameStart := 0 },
  { event := event79338
    frameStart := 0 },
  { event := event79339
    frameStart := 0 },
  { event := event79340
    frameStart := 0 },
  { event := event79341
    frameStart := 0 },
  { event := event79342
    frameStart := 0 },
  { event := event79343
    frameStart := 0 }
]

def eventLeaf4959 : Array AnnotatedEvent := #[
  { event := event79344
    frameStart := 0 },
  { event := event79345
    frameStart := 0 },
  { event := event79346
    frameStart := 0 },
  { event := event79347
    frameStart := 0 },
  { event := event79348
    frameStart := 0 },
  { event := event79349
    frameStart := 0 },
  { event := event79350
    frameStart := 0 },
  { event := event79351
    frameStart := 0 },
  { event := event79352
    frameStart := 0 },
  { event := event79353
    frameStart := 0 },
  { event := event79354
    frameStart := 0 },
  { event := event79355
    frameStart := 0 },
  { event := event79356
    frameStart := 0 },
  { event := event79357
    frameStart := 0 },
  { event := event79358
    frameStart := 0 },
  { event := event79359
    frameStart := 0 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events309
