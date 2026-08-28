import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events1020

open Mxx.Certificate.OperationalNoise
open CertificateABI

def exact261120RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨24950⟩⟩, ⟨.program ⟨257⟩, ⟨56370⟩⟩], []⟩, (1)⟩]

theorem exact261120RawTermsValid :
    exact261120RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event261120 : Event := .resultExact (⟨.program ⟨257⟩, ⟨56371⟩⟩) exact261120RawTerms (.finite 256) 261118 .exactZero (none)

def event261121 : Event := .predecessor (⟨.program ⟨257⟩, ⟨56372⟩⟩) 0 ⟨56371⟩ 261120

def event261122 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨56372⟩⟩) (.identity (.predecessor 0 261121 .coefficient))

def event261123 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨56372⟩⟩) (.finite 256)

def event261124 : Event := .predecessor (⟨.program ⟨257⟩, ⟨56808⟩⟩) 0 ⟨56372⟩ 261123

def event261125 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨56808⟩⟩) (.authority (.programFamilyFact))

def exact261126RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨56808⟩⟩], []⟩, (1)⟩]

theorem exact261126RawTermsValid :
    exact261126RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event261126 : Event := .resultExact (⟨.program ⟨257⟩, ⟨56808⟩⟩) exact261126RawTerms (.finite 16) 261125 .exactZero (none)

def event261127 : Event := .predecessor (⟨.program ⟨257⟩, ⟨56809⟩⟩) 0 ⟨56808⟩ 261126

def event261128 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨56809⟩⟩) (.identity (.predecessor 0 261127 .coefficient))

def event261129 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨56809⟩⟩) (.finite 16)

def event261130 : Event := .predecessor (⟨.program ⟨257⟩, ⟨57026⟩⟩) 0 ⟨56809⟩ 261129

def event261131 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨57026⟩⟩) (.authority (.programFamilyFact))

def exact261132RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨57026⟩⟩], []⟩, (1)⟩]

theorem exact261132RawTermsValid :
    exact261132RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event261132 : Event := .resultExact (⟨.program ⟨257⟩, ⟨57026⟩⟩) exact261132RawTerms (.finite 60) 261131 .exactZero (none)

def event261133 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24710⟩⟩) 0 ⟨5505⟩ 260856

def event261134 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨24710⟩⟩) (.authority (.programFamilyFact))

def exact261135RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨24710⟩⟩], []⟩, (1)⟩]

theorem exact261135RawTermsValid :
    exact261135RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event261135 : Event := .resultExact (⟨.program ⟨257⟩, ⟨24710⟩⟩) exact261135RawTerms (.finite 12) 261134 .exactZero (none)

def event261136 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53390⟩⟩) 0 ⟨5505⟩ 260856

def event261137 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53390⟩⟩) (.authority (.programFamilyFact))

def exact261138RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨53390⟩⟩], []⟩, (1)⟩]

theorem exact261138RawTermsValid :
    exact261138RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event261138 : Event := .resultExact (⟨.program ⟨257⟩, ⟨53390⟩⟩) exact261138RawTerms (.finite 12) 261137 .exactZero (none)

def event261139 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53391⟩⟩) 0 ⟨53390⟩ 261138

def event261140 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53391⟩⟩) 1 ⟨24710⟩ 261135

def event261141 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53391⟩⟩) (.product (.predecessor 0 261139 .coefficient) (.predecessor 1 261140 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event261142 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨53391⟩⟩, .operator (⟨261138, 0⟩, ⟨261135, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨24710⟩⟩, ⟨.program ⟨257⟩, ⟨53390⟩⟩], []⟩, (1)⟩)

def exact261143RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨24710⟩⟩, ⟨.program ⟨257⟩, ⟨53390⟩⟩], []⟩, (1)⟩]

theorem exact261143RawTermsValid :
    exact261143RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event261143 : Event := .resultExact (⟨.program ⟨257⟩, ⟨53391⟩⟩) exact261143RawTerms (.finite 144) 261141 .exactZero (none)

def event261144 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53392⟩⟩) 0 ⟨53391⟩ 261143

def event261145 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53392⟩⟩) (.identity (.predecessor 0 261144 .coefficient))

def event261146 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨53392⟩⟩) (.finite 144)

def event261147 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53828⟩⟩) 0 ⟨53392⟩ 261146

def event261148 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53828⟩⟩) (.authority (.programFamilyFact))

def exact261149RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨53828⟩⟩], []⟩, (1)⟩]

theorem exact261149RawTermsValid :
    exact261149RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event261149 : Event := .resultExact (⟨.program ⟨257⟩, ⟨53828⟩⟩) exact261149RawTerms (.finite 12) 261148 .exactZero (none)

def event261150 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53829⟩⟩) 0 ⟨53828⟩ 261149

def event261151 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53829⟩⟩) (.identity (.predecessor 0 261150 .coefficient))

def event261152 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨53829⟩⟩) (.finite 12)

def event261153 : Event := .predecessor (⟨.program ⟨257⟩, ⟨54046⟩⟩) 0 ⟨53829⟩ 261152

def event261154 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨54046⟩⟩) (.authority (.programFamilyFact))

def exact261155RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨54046⟩⟩], []⟩, (1)⟩]

theorem exact261155RawTermsValid :
    exact261155RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event261155 : Event := .resultExact (⟨.program ⟨257⟩, ⟨54046⟩⟩) exact261155RawTerms (.finite 59) 261154 .exactZero (none)

def event261156 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24470⟩⟩) 0 ⟨5505⟩ 260856

def event261157 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨24470⟩⟩) (.authority (.programFamilyFact))

def exact261158RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨24470⟩⟩], []⟩, (1)⟩]

theorem exact261158RawTermsValid :
    exact261158RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event261158 : Event := .resultExact (⟨.program ⟨257⟩, ⟨24470⟩⟩) exact261158RawTerms (.finite 10) 261157 .exactZero (none)

def event261159 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50410⟩⟩) 0 ⟨5505⟩ 260856

def event261160 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50410⟩⟩) (.authority (.programFamilyFact))

def exact261161RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨50410⟩⟩], []⟩, (1)⟩]

theorem exact261161RawTermsValid :
    exact261161RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event261161 : Event := .resultExact (⟨.program ⟨257⟩, ⟨50410⟩⟩) exact261161RawTerms (.finite 10) 261160 .exactZero (none)

def event261162 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50411⟩⟩) 0 ⟨50410⟩ 261161

def event261163 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50411⟩⟩) 1 ⟨24470⟩ 261158

def event261164 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50411⟩⟩) (.product (.predecessor 0 261162 .coefficient) (.predecessor 1 261163 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event261165 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨50411⟩⟩, .operator (⟨261161, 0⟩, ⟨261158, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨24470⟩⟩, ⟨.program ⟨257⟩, ⟨50410⟩⟩], []⟩, (1)⟩)

def exact261166RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨24470⟩⟩, ⟨.program ⟨257⟩, ⟨50410⟩⟩], []⟩, (1)⟩]

theorem exact261166RawTermsValid :
    exact261166RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event261166 : Event := .resultExact (⟨.program ⟨257⟩, ⟨50411⟩⟩) exact261166RawTerms (.finite 100) 261164 .exactZero (none)

def event261167 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50412⟩⟩) 0 ⟨50411⟩ 261166

def event261168 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50412⟩⟩) (.identity (.predecessor 0 261167 .coefficient))

def event261169 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨50412⟩⟩) (.finite 100)

def event261170 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50848⟩⟩) 0 ⟨50412⟩ 261169

def event261171 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50848⟩⟩) (.authority (.programFamilyFact))

def exact261172RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨50848⟩⟩], []⟩, (1)⟩]

theorem exact261172RawTermsValid :
    exact261172RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event261172 : Event := .resultExact (⟨.program ⟨257⟩, ⟨50848⟩⟩) exact261172RawTerms (.finite 10) 261171 .exactZero (none)

def event261173 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50849⟩⟩) 0 ⟨50848⟩ 261172

def event261174 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50849⟩⟩) (.identity (.predecessor 0 261173 .coefficient))

def event261175 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨50849⟩⟩) (.finite 10)

def event261176 : Event := .predecessor (⟨.program ⟨257⟩, ⟨51066⟩⟩) 0 ⟨50849⟩ 261175

def event261177 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨51066⟩⟩) (.authority (.programFamilyFact))

def exact261178RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨51066⟩⟩], []⟩, (1)⟩]

theorem exact261178RawTermsValid :
    exact261178RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event261178 : Event := .resultExact (⟨.program ⟨257⟩, ⟨51066⟩⟩) exact261178RawTerms (.finite 58) 261177 .exactZero (none)

def event261179 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24230⟩⟩) 0 ⟨5505⟩ 260856

def event261180 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨24230⟩⟩) (.authority (.programFamilyFact))

def exact261181RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨24230⟩⟩], []⟩, (1)⟩]

theorem exact261181RawTermsValid :
    exact261181RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event261181 : Event := .resultExact (⟨.program ⟨257⟩, ⟨24230⟩⟩) exact261181RawTerms (.finite 6) 261180 .exactZero (none)

def event261182 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31350⟩⟩) 0 ⟨5505⟩ 260856

def event261183 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31350⟩⟩) (.authority (.programFamilyFact))

def exact261184RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨31350⟩⟩], []⟩, (1)⟩]

theorem exact261184RawTermsValid :
    exact261184RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event261184 : Event := .resultExact (⟨.program ⟨257⟩, ⟨31350⟩⟩) exact261184RawTerms (.finite 6) 261183 .exactZero (none)

def event261185 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31351⟩⟩) 0 ⟨31350⟩ 261184

def event261186 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31351⟩⟩) 1 ⟨24230⟩ 261181

def event261187 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31351⟩⟩) (.product (.predecessor 0 261185 .coefficient) (.predecessor 1 261186 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event261188 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨31351⟩⟩, .operator (⟨261184, 0⟩, ⟨261181, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨24230⟩⟩, ⟨.program ⟨257⟩, ⟨31350⟩⟩], []⟩, (1)⟩)

def exact261189RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨24230⟩⟩, ⟨.program ⟨257⟩, ⟨31350⟩⟩], []⟩, (1)⟩]

theorem exact261189RawTermsValid :
    exact261189RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event261189 : Event := .resultExact (⟨.program ⟨257⟩, ⟨31351⟩⟩) exact261189RawTerms (.finite 36) 261187 .exactZero (none)

def event261190 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31352⟩⟩) 0 ⟨31351⟩ 261189

def event261191 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31352⟩⟩) (.identity (.predecessor 0 261190 .coefficient))

def event261192 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨31352⟩⟩) (.finite 36)

def event261193 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31788⟩⟩) 0 ⟨31352⟩ 261192

def event261194 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31788⟩⟩) (.authority (.programFamilyFact))

def exact261195RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨31788⟩⟩], []⟩, (1)⟩]

theorem exact261195RawTermsValid :
    exact261195RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event261195 : Event := .resultExact (⟨.program ⟨257⟩, ⟨31788⟩⟩) exact261195RawTerms (.finite 6) 261194 .exactZero (none)

def event261196 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31789⟩⟩) 0 ⟨31788⟩ 261195

def event261197 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31789⟩⟩) (.identity (.predecessor 0 261196 .coefficient))

def event261198 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨31789⟩⟩) (.finite 6)

def event261199 : Event := .predecessor (⟨.program ⟨257⟩, ⟨32011⟩⟩) 0 ⟨31789⟩ 261198

def event261200 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨32011⟩⟩) (.authority (.programFamilyFact))

def exact261201RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨32011⟩⟩], []⟩, (1)⟩]

theorem exact261201RawTermsValid :
    exact261201RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event261201 : Event := .resultExact (⟨.program ⟨257⟩, ⟨32011⟩⟩) exact261201RawTerms (.finite 55) 261200 .exactZero (none)

def event261202 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21374⟩⟩) 0 ⟨5505⟩ 260856

def event261203 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨21374⟩⟩) (.authority (.programFamilyFact))

def exact261204RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨21374⟩⟩], []⟩, (1)⟩]

theorem exact261204RawTermsValid :
    exact261204RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event261204 : Event := .resultExact (⟨.program ⟨257⟩, ⟨21374⟩⟩) exact261204RawTerms (.finite 4) 261203 .exactZero (none)

def event261205 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21026⟩⟩) 0 ⟨5505⟩ 260856

def event261206 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨21026⟩⟩) (.authority (.programFamilyFact))

def exact261207RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨21026⟩⟩], []⟩, (1)⟩]

theorem exact261207RawTermsValid :
    exact261207RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event261207 : Event := .resultExact (⟨.program ⟨257⟩, ⟨21026⟩⟩) exact261207RawTerms (.finite 4) 261206 .exactZero (none)

def event261208 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21375⟩⟩) 0 ⟨21026⟩ 261207

def event261209 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21375⟩⟩) 1 ⟨21374⟩ 261204

def event261210 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨21375⟩⟩) (.product (.predecessor 0 261208 .coefficient) (.predecessor 1 261209 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event261211 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨21375⟩⟩, .operator (⟨261207, 0⟩, ⟨261204, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨21026⟩⟩, ⟨.program ⟨257⟩, ⟨21374⟩⟩], []⟩, (1)⟩)

def exact261212RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨21026⟩⟩, ⟨.program ⟨257⟩, ⟨21374⟩⟩], []⟩, (1)⟩]

theorem exact261212RawTermsValid :
    exact261212RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event261212 : Event := .resultExact (⟨.program ⟨257⟩, ⟨21375⟩⟩) exact261212RawTerms (.finite 16) 261210 .exactZero (none)

def event261213 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21376⟩⟩) 0 ⟨21375⟩ 261212

def event261214 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨21376⟩⟩) (.identity (.predecessor 0 261213 .coefficient))

def event261215 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨21376⟩⟩) (.finite 16)

def event261216 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21768⟩⟩) 0 ⟨21376⟩ 261215

def event261217 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨21768⟩⟩) (.authority (.programFamilyFact))

def exact261218RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨21768⟩⟩], []⟩, (1)⟩]

theorem exact261218RawTermsValid :
    exact261218RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event261218 : Event := .resultExact (⟨.program ⟨257⟩, ⟨21768⟩⟩) exact261218RawTerms (.finite 4) 261217 .exactZero (none)

def event261219 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21769⟩⟩) 0 ⟨21768⟩ 261218

def event261220 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨21769⟩⟩) (.identity (.predecessor 0 261219 .coefficient))

def event261221 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨21769⟩⟩) (.finite 4)

def event261222 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21991⟩⟩) 0 ⟨21769⟩ 261221

def event261223 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨21991⟩⟩) (.authority (.programFamilyFact))

def exact261224RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨21991⟩⟩], []⟩, (1)⟩]

theorem exact261224RawTermsValid :
    exact261224RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event261224 : Event := .resultExact (⟨.program ⟨257⟩, ⟨21991⟩⟩) exact261224RawTerms (.finite 51) 261223 .exactZero (none)

def event261225 : Event := .predecessor (⟨.program ⟨257⟩, ⟨18154⟩⟩) 0 ⟨5505⟩ 260856

def event261226 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨18154⟩⟩) (.authority (.programFamilyFact))

def exact261227RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨18154⟩⟩], []⟩, (1)⟩]

theorem exact261227RawTermsValid :
    exact261227RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event261227 : Event := .resultExact (⟨.program ⟨257⟩, ⟨18154⟩⟩) exact261227RawTerms (.finite 3) 261226 .exactZero (none)

def event261228 : Event := .predecessor (⟨.program ⟨257⟩, ⟨12606⟩⟩) 0 ⟨5505⟩ 260856

def event261229 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨12606⟩⟩) (.authority (.programFamilyFact))

def exact261230RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨12606⟩⟩], []⟩, (1)⟩]

theorem exact261230RawTermsValid :
    exact261230RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event261230 : Event := .resultExact (⟨.program ⟨257⟩, ⟨12606⟩⟩) exact261230RawTerms (.finite 3) 261229 .exactZero (none)

def event261231 : Event := .predecessor (⟨.program ⟨257⟩, ⟨18155⟩⟩) 0 ⟨12606⟩ 261230

def event261232 : Event := .predecessor (⟨.program ⟨257⟩, ⟨18155⟩⟩) 1 ⟨18154⟩ 261227

def event261233 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨18155⟩⟩) (.product (.predecessor 0 261231 .coefficient) (.predecessor 1 261232 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event261234 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨18155⟩⟩, .operator (⟨261230, 0⟩, ⟨261227, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨12606⟩⟩, ⟨.program ⟨257⟩, ⟨18154⟩⟩], []⟩, (1)⟩)

def exact261235RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨12606⟩⟩, ⟨.program ⟨257⟩, ⟨18154⟩⟩], []⟩, (1)⟩]

theorem exact261235RawTermsValid :
    exact261235RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event261235 : Event := .resultExact (⟨.program ⟨257⟩, ⟨18155⟩⟩) exact261235RawTerms (.finite 9) 261233 .exactZero (none)

def event261236 : Event := .predecessor (⟨.program ⟨257⟩, ⟨18156⟩⟩) 0 ⟨18155⟩ 261235

def event261237 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨18156⟩⟩) (.identity (.predecessor 0 261236 .coefficient))

def event261238 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨18156⟩⟩) (.finite 9)

def event261239 : Event := .predecessor (⟨.program ⟨257⟩, ⟨18548⟩⟩) 0 ⟨18156⟩ 261238

def event261240 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨18548⟩⟩) (.authority (.programFamilyFact))

def exact261241RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨18548⟩⟩], []⟩, (1)⟩]

theorem exact261241RawTermsValid :
    exact261241RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event261241 : Event := .resultExact (⟨.program ⟨257⟩, ⟨18548⟩⟩) exact261241RawTerms (.finite 3) 261240 .exactZero (none)

def event261242 : Event := .predecessor (⟨.program ⟨257⟩, ⟨18549⟩⟩) 0 ⟨18548⟩ 261241

def event261243 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨18549⟩⟩) (.identity (.predecessor 0 261242 .coefficient))

def event261244 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨18549⟩⟩) (.finite 3)

def event261245 : Event := .predecessor (⟨.program ⟨257⟩, ⟨18771⟩⟩) 0 ⟨18549⟩ 261244

def event261246 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨18771⟩⟩) (.authority (.programFamilyFact))

def exact261247RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨18771⟩⟩], []⟩, (1)⟩]

theorem exact261247RawTermsValid :
    exact261247RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event261247 : Event := .resultExact (⟨.program ⟨257⟩, ⟨18771⟩⟩) exact261247RawTerms (.finite 48) 261246 .exactZero (none)

def event261248 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15354⟩⟩) 0 ⟨5505⟩ 260856

def event261249 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15354⟩⟩) (.authority (.programFamilyFact))

def exact261250RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15354⟩⟩], []⟩, (1)⟩]

theorem exact261250RawTermsValid :
    exact261250RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event261250 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15354⟩⟩) exact261250RawTerms (.finite 2) 261249 .exactZero (none)

def event261251 : Event := .predecessor (⟨.program ⟨257⟩, ⟨12306⟩⟩) 0 ⟨5505⟩ 260856

def event261252 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨12306⟩⟩) (.authority (.programFamilyFact))

def exact261253RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨12306⟩⟩], []⟩, (1)⟩]

theorem exact261253RawTermsValid :
    exact261253RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event261253 : Event := .resultExact (⟨.program ⟨257⟩, ⟨12306⟩⟩) exact261253RawTerms (.finite 2) 261252 .exactZero (none)

def event261254 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15355⟩⟩) 0 ⟨12306⟩ 261253

def event261255 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15355⟩⟩) 1 ⟨15354⟩ 261250

def event261256 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15355⟩⟩) (.product (.predecessor 0 261254 .coefficient) (.predecessor 1 261255 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event261257 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨15355⟩⟩, .operator (⟨261253, 0⟩, ⟨261250, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨12306⟩⟩, ⟨.program ⟨257⟩, ⟨15354⟩⟩], []⟩, (1)⟩)

def exact261258RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨12306⟩⟩, ⟨.program ⟨257⟩, ⟨15354⟩⟩], []⟩, (1)⟩]

theorem exact261258RawTermsValid :
    exact261258RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event261258 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15355⟩⟩) exact261258RawTerms (.finite 4) 261256 .exactZero (none)

def event261259 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15356⟩⟩) 0 ⟨15355⟩ 261258

def event261260 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15356⟩⟩) (.identity (.predecessor 0 261259 .coefficient))

def event261261 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨15356⟩⟩) (.finite 4)

def event261262 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15748⟩⟩) 0 ⟨15356⟩ 261261

def event261263 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15748⟩⟩) (.authority (.programFamilyFact))

def exact261264RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15748⟩⟩], []⟩, (1)⟩]

theorem exact261264RawTermsValid :
    exact261264RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event261264 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15748⟩⟩) exact261264RawTerms (.finite 2) 261263 .exactZero (none)

def event261265 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15749⟩⟩) 0 ⟨15748⟩ 261264

def event261266 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15749⟩⟩) (.identity (.predecessor 0 261265 .coefficient))

def event261267 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨15749⟩⟩) (.finite 2)

def event261268 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15955⟩⟩) 0 ⟨15749⟩ 261267

def event261269 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15955⟩⟩) (.authority (.programFamilyFact))

def exact261270RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15955⟩⟩], []⟩, (1)⟩]

theorem exact261270RawTermsValid :
    exact261270RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event261270 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15955⟩⟩) exact261270RawTerms (.finite 43) 261269 .exactZero (none)

def event261271 : Event := .predecessor (⟨.program ⟨257⟩, ⟨18772⟩⟩) 0 ⟨15955⟩ 261270

def event261272 : Event := .predecessor (⟨.program ⟨257⟩, ⟨18772⟩⟩) 1 ⟨18771⟩ 261247

def event261273 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨18772⟩⟩) (.sum [.predecessor 0 261271 .coefficient, .predecessor 1 261272 .coefficient])

def exact261274RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15955⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨18771⟩⟩], []⟩, (1)⟩]

theorem exact261274RawTermsValid :
    exact261274RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event261274 : Event := .resultExact (⟨.program ⟨257⟩, ⟨18772⟩⟩) exact261274RawTerms (.finite 91) 261273 .exactZero (none)

def event261275 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21992⟩⟩) 0 ⟨18772⟩ 261274

def event261276 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21992⟩⟩) 1 ⟨21991⟩ 261224

def event261277 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨21992⟩⟩) (.sum [.predecessor 0 261275 .coefficient, .predecessor 1 261276 .coefficient])

def exact261278RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15955⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨18771⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨21991⟩⟩], []⟩, (1)⟩]

theorem exact261278RawTermsValid :
    exact261278RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event261278 : Event := .resultExact (⟨.program ⟨257⟩, ⟨21992⟩⟩) exact261278RawTerms (.finite 142) 261277 .exactZero (none)

def event261279 : Event := .predecessor (⟨.program ⟨257⟩, ⟨32012⟩⟩) 0 ⟨21992⟩ 261278

def event261280 : Event := .predecessor (⟨.program ⟨257⟩, ⟨32012⟩⟩) 1 ⟨32011⟩ 261201

def event261281 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨32012⟩⟩) (.sum [.predecessor 0 261279 .coefficient, .predecessor 1 261280 .coefficient])

def exact261282RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15955⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨18771⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨21991⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨32011⟩⟩], []⟩, (1)⟩]

theorem exact261282RawTermsValid :
    exact261282RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event261282 : Event := .resultExact (⟨.program ⟨257⟩, ⟨32012⟩⟩) exact261282RawTerms (.finite 197) 261281 .exactZero (none)

def event261283 : Event := .predecessor (⟨.program ⟨257⟩, ⟨51067⟩⟩) 0 ⟨32012⟩ 261282

def event261284 : Event := .predecessor (⟨.program ⟨257⟩, ⟨51067⟩⟩) 1 ⟨51066⟩ 261178

def event261285 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨51067⟩⟩) (.sum [.predecessor 0 261283 .coefficient, .predecessor 1 261284 .coefficient])

def exact261286RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15955⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨18771⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨21991⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨32011⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨51066⟩⟩], []⟩, (1)⟩]

theorem exact261286RawTermsValid :
    exact261286RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event261286 : Event := .resultExact (⟨.program ⟨257⟩, ⟨51067⟩⟩) exact261286RawTerms (.finite 255) 261285 .exactZero (none)

def event261287 : Event := .predecessor (⟨.program ⟨257⟩, ⟨54047⟩⟩) 0 ⟨51067⟩ 261286

def event261288 : Event := .predecessor (⟨.program ⟨257⟩, ⟨54047⟩⟩) 1 ⟨54046⟩ 261155

def event261289 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨54047⟩⟩) (.sum [.predecessor 0 261287 .coefficient, .predecessor 1 261288 .coefficient])

def exact261290RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15955⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨18771⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨21991⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨32011⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨51066⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨54046⟩⟩], []⟩, (1)⟩]

theorem exact261290RawTermsValid :
    exact261290RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event261290 : Event := .resultExact (⟨.program ⟨257⟩, ⟨54047⟩⟩) exact261290RawTerms (.finite 314) 261289 .exactZero (none)

def event261291 : Event := .predecessor (⟨.program ⟨257⟩, ⟨57027⟩⟩) 0 ⟨54047⟩ 261290

def event261292 : Event := .predecessor (⟨.program ⟨257⟩, ⟨57027⟩⟩) 1 ⟨57026⟩ 261132

def event261293 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨57027⟩⟩) (.sum [.predecessor 0 261291 .coefficient, .predecessor 1 261292 .coefficient])

def exact261294RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15955⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨18771⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨21991⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨32011⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨51066⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨54046⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨57026⟩⟩], []⟩, (1)⟩]

theorem exact261294RawTermsValid :
    exact261294RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event261294 : Event := .resultExact (⟨.program ⟨257⟩, ⟨57027⟩⟩) exact261294RawTerms (.finite 374) 261293 .exactZero (none)

def event261295 : Event := .predecessor (⟨.program ⟨257⟩, ⟨60007⟩⟩) 0 ⟨57027⟩ 261294

def event261296 : Event := .predecessor (⟨.program ⟨257⟩, ⟨60007⟩⟩) 1 ⟨60006⟩ 261109

def event261297 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨60007⟩⟩) (.sum [.predecessor 0 261295 .coefficient, .predecessor 1 261296 .coefficient])

def exact261298RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15955⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨18771⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨21991⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨32011⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨51066⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨54046⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨57026⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨60006⟩⟩], []⟩, (1)⟩]

theorem exact261298RawTermsValid :
    exact261298RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event261298 : Event := .resultExact (⟨.program ⟨257⟩, ⟨60007⟩⟩) exact261298RawTerms (.finite 435) 261297 .exactZero (none)

def event261299 : Event := .predecessor (⟨.program ⟨257⟩, ⟨62987⟩⟩) 0 ⟨60007⟩ 261298

def event261300 : Event := .predecessor (⟨.program ⟨257⟩, ⟨62987⟩⟩) 1 ⟨62986⟩ 261086

def event261301 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨62987⟩⟩) (.sum [.predecessor 0 261299 .coefficient, .predecessor 1 261300 .coefficient])

def exact261302RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15955⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨18771⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨21991⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨32011⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨51066⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨54046⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨57026⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨60006⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨62986⟩⟩], []⟩, (1)⟩]

theorem exact261302RawTermsValid :
    exact261302RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event261302 : Event := .resultExact (⟨.program ⟨257⟩, ⟨62987⟩⟩) exact261302RawTerms (.finite 496) 261301 .exactZero (none)

def event261303 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66252⟩⟩) 0 ⟨62987⟩ 261302

def event261304 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66252⟩⟩) 1 ⟨66251⟩ 261063

def event261305 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨66252⟩⟩) (.sum [.predecessor 0 261303 .coefficient, .predecessor 1 261304 .coefficient])

def exact261306RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15955⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨18771⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨21991⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨32011⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨51066⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨54046⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨57026⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨60006⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨62986⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨66251⟩⟩], []⟩, (1)⟩]

theorem exact261306RawTermsValid :
    exact261306RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event261306 : Event := .resultExact (⟨.program ⟨257⟩, ⟨66252⟩⟩) exact261306RawTerms (.finite 558) 261305 .exactZero (none)

def event261307 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66253⟩⟩) 0 ⟨66252⟩ 261306

def event261308 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66253⟩⟩) 1 ⟨26554⟩ 261040

def event261309 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨66253⟩⟩) (.sum [.predecessor 0 261307 .coefficient, .predecessor 1 261308 .coefficient])

def exact261310RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15955⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨18771⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨21991⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨26554⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨32011⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨51066⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨54046⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨57026⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨60006⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨62986⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨66251⟩⟩], []⟩, (1)⟩]

theorem exact261310RawTermsValid :
    exact261310RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event261310 : Event := .resultExact (⟨.program ⟨257⟩, ⟨66253⟩⟩) exact261310RawTerms (.finite 620) 261309 .exactZero (none)

def event261311 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66254⟩⟩) 0 ⟨66253⟩ 261310

def event261312 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66254⟩⟩) 1 ⟨29234⟩ 261017

def event261313 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨66254⟩⟩) (.sum [.predecessor 0 261311 .coefficient, .predecessor 1 261312 .coefficient])

def exact261314RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15955⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨18771⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨21991⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨26554⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨29234⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨32011⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨51066⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨54046⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨57026⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨60006⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨62986⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨66251⟩⟩], []⟩, (1)⟩]

theorem exact261314RawTermsValid :
    exact261314RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event261314 : Event := .resultExact (⟨.program ⟨257⟩, ⟨66254⟩⟩) exact261314RawTerms (.finite 682) 261313 .exactZero (none)

def event261315 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66255⟩⟩) 0 ⟨66254⟩ 261314

def event261316 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66255⟩⟩) 1 ⟨34898⟩ 260994

def event261317 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨66255⟩⟩) (.sum [.predecessor 0 261315 .coefficient, .predecessor 1 261316 .coefficient])

def exact261318RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15955⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨18771⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨21991⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨26554⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨29234⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨32011⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨34898⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨51066⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨54046⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨57026⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨60006⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨62986⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨66251⟩⟩], []⟩, (1)⟩]

theorem exact261318RawTermsValid :
    exact261318RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event261318 : Event := .resultExact (⟨.program ⟨257⟩, ⟨66255⟩⟩) exact261318RawTerms (.finite 744) 261317 .exactZero (none)

def event261319 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66256⟩⟩) 0 ⟨66255⟩ 261318

def event261320 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66256⟩⟩) 1 ⟨37578⟩ 260971

def event261321 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨66256⟩⟩) (.sum [.predecessor 0 261319 .coefficient, .predecessor 1 261320 .coefficient])

def exact261322RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15955⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨18771⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨21991⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨26554⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨29234⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨32011⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨34898⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨37578⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨51066⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨54046⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨57026⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨60006⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨62986⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨66251⟩⟩], []⟩, (1)⟩]

theorem exact261322RawTermsValid :
    exact261322RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event261322 : Event := .resultExact (⟨.program ⟨257⟩, ⟨66256⟩⟩) exact261322RawTerms (.finite 807) 261321 .exactZero (none)

def event261323 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66257⟩⟩) 0 ⟨66256⟩ 261322

def event261324 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66257⟩⟩) 1 ⟨40254⟩ 260948

def event261325 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨66257⟩⟩) (.sum [.predecessor 0 261323 .coefficient, .predecessor 1 261324 .coefficient])

def exact261326RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15955⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨18771⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨21991⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨26554⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨29234⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨32011⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨34898⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨37578⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨40254⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨51066⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨54046⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨57026⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨60006⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨62986⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨66251⟩⟩], []⟩, (1)⟩]

theorem exact261326RawTermsValid :
    exact261326RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event261326 : Event := .resultExact (⟨.program ⟨257⟩, ⟨66257⟩⟩) exact261326RawTerms (.finite 870) 261325 .exactZero (none)

def event261327 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66258⟩⟩) 0 ⟨66257⟩ 261326

def event261328 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66258⟩⟩) 1 ⟨42934⟩ 260925

def event261329 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨66258⟩⟩) (.sum [.predecessor 0 261327 .coefficient, .predecessor 1 261328 .coefficient])

def exact261330RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15955⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨18771⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨21991⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨26554⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨29234⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨32011⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨34898⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨37578⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨40254⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨42934⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨51066⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨54046⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨57026⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨60006⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨62986⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨66251⟩⟩], []⟩, (1)⟩]

theorem exact261330RawTermsValid :
    exact261330RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event261330 : Event := .resultExact (⟨.program ⟨257⟩, ⟨66258⟩⟩) exact261330RawTerms (.finite 933) 261329 .exactZero (none)

def event261331 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66259⟩⟩) 0 ⟨66258⟩ 261330

def event261332 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66259⟩⟩) 1 ⟨45618⟩ 260902

def event261333 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨66259⟩⟩) (.sum [.predecessor 0 261331 .coefficient, .predecessor 1 261332 .coefficient])

def exact261334RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15955⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨18771⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨21991⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨26554⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨29234⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨32011⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨34898⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨37578⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨40254⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨42934⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨45618⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨51066⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨54046⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨57026⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨60006⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨62986⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨66251⟩⟩], []⟩, (1)⟩]

theorem exact261334RawTermsValid :
    exact261334RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event261334 : Event := .resultExact (⟨.program ⟨257⟩, ⟨66259⟩⟩) exact261334RawTerms (.finite 996) 261333 .exactZero (none)

def event261335 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66260⟩⟩) 0 ⟨66259⟩ 261334

def event261336 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66260⟩⟩) 1 ⟨48298⟩ 260879

def event261337 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨66260⟩⟩) (.sum [.predecessor 0 261335 .coefficient, .predecessor 1 261336 .coefficient])

def exact261338RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15955⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨18771⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨21991⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨26554⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨29234⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨32011⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨34898⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨37578⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨40254⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨42934⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨45618⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨48298⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨51066⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨54046⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨57026⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨60006⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨62986⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨66251⟩⟩], []⟩, (1)⟩]

theorem exact261338RawTermsValid :
    exact261338RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event261338 : Event := .resultExact (⟨.program ⟨257⟩, ⟨66260⟩⟩) exact261338RawTerms (.finite 1059) 261337 .exactZero (none)

def event261339 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66261⟩⟩) 0 ⟨66260⟩ 261338

def event261340 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨66261⟩⟩) (.identity (.predecessor 0 261339 .coefficient))

def event261341 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨66261⟩⟩) (.finite 1059)

def event261342 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68799⟩⟩) 0 ⟨66261⟩ 261341

def event261343 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨68799⟩⟩) (.authority (.programFamilyFact))

def event261344 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨68799⟩⟩) (.finite 1152)

def event261345 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨7177⟩⟩) .missing

def event261346 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68800⟩⟩) 0 ⟨7177⟩ 261345

def event261347 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68800⟩⟩) 1 ⟨68799⟩ 261344

def event261348 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨68800⟩⟩) (.authority (.operator))

def exact261349RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨68800⟩⟩]⟩, (1)⟩]

theorem exact261349RawTermsValid :
    exact261349RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event261349 : Event := .resultExact (⟨.program ⟨257⟩, ⟨68800⟩⟩) exact261349RawTerms .large 261348 .exactZero (none)

def event261350 : Event := .predecessor (⟨.program ⟨257⟩, ⟨71082⟩⟩) 0 ⟨68800⟩ 261349

def event261351 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨71082⟩⟩) (.authority (.operator))

def exact261352RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨71082⟩⟩]⟩, (1)⟩]

theorem exact261352RawTermsValid :
    exact261352RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event261352 : Event := .resultExact (⟨.program ⟨257⟩, ⟨71082⟩⟩) exact261352RawTerms (.finite 8192) 261351 .exactZero (none)

def event261353 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event261354 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event261355 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69067⟩⟩) 0 ⟨66261⟩ 261341

def event261356 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69067⟩⟩) 1 ⟨136⟩ 261354

def event261357 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69067⟩⟩) (.sum [.predecessor 0 261355 .coefficient, .predecessor 1 261356 .coefficient])

def event261358 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨69067⟩⟩) (.finite 1059)

def event261359 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69068⟩⟩) 0 ⟨69067⟩ 261358

def event261360 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69068⟩⟩) (.identity (.predecessor 0 261359 .coefficient))

def exact261361RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15955⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨18771⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨21991⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨26554⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨29234⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨32011⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨34898⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨37578⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨40254⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨42934⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨45618⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨48298⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨51066⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨54046⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨57026⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨60006⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨62986⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨66251⟩⟩], []⟩, (1)⟩]

theorem exact261361RawTermsValid :
    exact261361RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event261361 : Event := .resultExact (⟨.program ⟨257⟩, ⟨69068⟩⟩) exact261361RawTerms (.finite 1059) 261360 .exactZero (none)

def event261362 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact261363RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact261363RawTermsValid :
    exact261363RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event261363 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact261363RawTerms .large 261362 .exactZero (none)

def event261364 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69069⟩⟩) 0 ⟨6908⟩ 261363

def event261365 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69069⟩⟩) 1 ⟨69068⟩ 261361

def event261366 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69069⟩⟩) (.product (.predecessor 0 261364 .coefficient) (.predecessor 1 261365 .coefficient) (⟨false, false, none, none, none⟩))

def event261367 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨69069⟩⟩, .operator (⟨261363, 0⟩, ⟨261361, 11⟩), ⟨[⟨.program ⟨257⟩, ⟨48298⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def event261368 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨69069⟩⟩, .operator (⟨261363, 0⟩, ⟨261361, 10⟩), ⟨[⟨.program ⟨257⟩, ⟨45618⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def event261369 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨69069⟩⟩, .operator (⟨261363, 0⟩, ⟨261361, 9⟩), ⟨[⟨.program ⟨257⟩, ⟨42934⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def event261370 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨69069⟩⟩, .operator (⟨261363, 0⟩, ⟨261361, 8⟩), ⟨[⟨.program ⟨257⟩, ⟨40254⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def event261371 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨69069⟩⟩, .operator (⟨261363, 0⟩, ⟨261361, 7⟩), ⟨[⟨.program ⟨257⟩, ⟨37578⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def event261372 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨69069⟩⟩, .operator (⟨261363, 0⟩, ⟨261361, 6⟩), ⟨[⟨.program ⟨257⟩, ⟨34898⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def event261373 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨69069⟩⟩, .operator (⟨261363, 0⟩, ⟨261361, 4⟩), ⟨[⟨.program ⟨257⟩, ⟨29234⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def event261374 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨69069⟩⟩, .operator (⟨261363, 0⟩, ⟨261361, 3⟩), ⟨[⟨.program ⟨257⟩, ⟨26554⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def event261375 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨69069⟩⟩, .operator (⟨261363, 0⟩, ⟨261361, 17⟩), ⟨[⟨.program ⟨257⟩, ⟨66251⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def eventLeaf16320 : Array AnnotatedEvent := #[
  { event := event261120
    frameStart := 260836 },
  { event := event261121
    frameStart := 260836 },
  { event := event261122
    frameStart := 260836 },
  { event := event261123
    frameStart := 260836 },
  { event := event261124
    frameStart := 260836 },
  { event := event261125
    frameStart := 260836 },
  { event := event261126
    frameStart := 260836 },
  { event := event261127
    frameStart := 260836 },
  { event := event261128
    frameStart := 260836 },
  { event := event261129
    frameStart := 260836 },
  { event := event261130
    frameStart := 260836 },
  { event := event261131
    frameStart := 260836 },
  { event := event261132
    frameStart := 260836 },
  { event := event261133
    frameStart := 260836 },
  { event := event261134
    frameStart := 260836 },
  { event := event261135
    frameStart := 260836 }
]

def eventLeaf16321 : Array AnnotatedEvent := #[
  { event := event261136
    frameStart := 260836 },
  { event := event261137
    frameStart := 260836 },
  { event := event261138
    frameStart := 260836 },
  { event := event261139
    frameStart := 260836 },
  { event := event261140
    frameStart := 260836 },
  { event := event261141
    frameStart := 260836 },
  { event := event261142
    frameStart := 260836 },
  { event := event261143
    frameStart := 260836 },
  { event := event261144
    frameStart := 260836 },
  { event := event261145
    frameStart := 260836 },
  { event := event261146
    frameStart := 260836 },
  { event := event261147
    frameStart := 260836 },
  { event := event261148
    frameStart := 260836 },
  { event := event261149
    frameStart := 260836 },
  { event := event261150
    frameStart := 260836 },
  { event := event261151
    frameStart := 260836 }
]

def eventLeaf16322 : Array AnnotatedEvent := #[
  { event := event261152
    frameStart := 260836 },
  { event := event261153
    frameStart := 260836 },
  { event := event261154
    frameStart := 260836 },
  { event := event261155
    frameStart := 260836 },
  { event := event261156
    frameStart := 260836 },
  { event := event261157
    frameStart := 260836 },
  { event := event261158
    frameStart := 260836 },
  { event := event261159
    frameStart := 260836 },
  { event := event261160
    frameStart := 260836 },
  { event := event261161
    frameStart := 260836 },
  { event := event261162
    frameStart := 260836 },
  { event := event261163
    frameStart := 260836 },
  { event := event261164
    frameStart := 260836 },
  { event := event261165
    frameStart := 260836 },
  { event := event261166
    frameStart := 260836 },
  { event := event261167
    frameStart := 260836 }
]

def eventLeaf16323 : Array AnnotatedEvent := #[
  { event := event261168
    frameStart := 260836 },
  { event := event261169
    frameStart := 260836 },
  { event := event261170
    frameStart := 260836 },
  { event := event261171
    frameStart := 260836 },
  { event := event261172
    frameStart := 260836 },
  { event := event261173
    frameStart := 260836 },
  { event := event261174
    frameStart := 260836 },
  { event := event261175
    frameStart := 260836 },
  { event := event261176
    frameStart := 260836 },
  { event := event261177
    frameStart := 260836 },
  { event := event261178
    frameStart := 260836 },
  { event := event261179
    frameStart := 260836 },
  { event := event261180
    frameStart := 260836 },
  { event := event261181
    frameStart := 260836 },
  { event := event261182
    frameStart := 260836 },
  { event := event261183
    frameStart := 260836 }
]

def eventLeaf16324 : Array AnnotatedEvent := #[
  { event := event261184
    frameStart := 260836 },
  { event := event261185
    frameStart := 260836 },
  { event := event261186
    frameStart := 260836 },
  { event := event261187
    frameStart := 260836 },
  { event := event261188
    frameStart := 260836 },
  { event := event261189
    frameStart := 260836 },
  { event := event261190
    frameStart := 260836 },
  { event := event261191
    frameStart := 260836 },
  { event := event261192
    frameStart := 260836 },
  { event := event261193
    frameStart := 260836 },
  { event := event261194
    frameStart := 260836 },
  { event := event261195
    frameStart := 260836 },
  { event := event261196
    frameStart := 260836 },
  { event := event261197
    frameStart := 260836 },
  { event := event261198
    frameStart := 260836 },
  { event := event261199
    frameStart := 260836 }
]

def eventLeaf16325 : Array AnnotatedEvent := #[
  { event := event261200
    frameStart := 260836 },
  { event := event261201
    frameStart := 260836 },
  { event := event261202
    frameStart := 260836 },
  { event := event261203
    frameStart := 260836 },
  { event := event261204
    frameStart := 260836 },
  { event := event261205
    frameStart := 260836 },
  { event := event261206
    frameStart := 260836 },
  { event := event261207
    frameStart := 260836 },
  { event := event261208
    frameStart := 260836 },
  { event := event261209
    frameStart := 260836 },
  { event := event261210
    frameStart := 260836 },
  { event := event261211
    frameStart := 260836 },
  { event := event261212
    frameStart := 260836 },
  { event := event261213
    frameStart := 260836 },
  { event := event261214
    frameStart := 260836 },
  { event := event261215
    frameStart := 260836 }
]

def eventLeaf16326 : Array AnnotatedEvent := #[
  { event := event261216
    frameStart := 260836 },
  { event := event261217
    frameStart := 260836 },
  { event := event261218
    frameStart := 260836 },
  { event := event261219
    frameStart := 260836 },
  { event := event261220
    frameStart := 260836 },
  { event := event261221
    frameStart := 260836 },
  { event := event261222
    frameStart := 260836 },
  { event := event261223
    frameStart := 260836 },
  { event := event261224
    frameStart := 260836 },
  { event := event261225
    frameStart := 260836 },
  { event := event261226
    frameStart := 260836 },
  { event := event261227
    frameStart := 260836 },
  { event := event261228
    frameStart := 260836 },
  { event := event261229
    frameStart := 260836 },
  { event := event261230
    frameStart := 260836 },
  { event := event261231
    frameStart := 260836 }
]

def eventLeaf16327 : Array AnnotatedEvent := #[
  { event := event261232
    frameStart := 260836 },
  { event := event261233
    frameStart := 260836 },
  { event := event261234
    frameStart := 260836 },
  { event := event261235
    frameStart := 260836 },
  { event := event261236
    frameStart := 260836 },
  { event := event261237
    frameStart := 260836 },
  { event := event261238
    frameStart := 260836 },
  { event := event261239
    frameStart := 260836 },
  { event := event261240
    frameStart := 260836 },
  { event := event261241
    frameStart := 260836 },
  { event := event261242
    frameStart := 260836 },
  { event := event261243
    frameStart := 260836 },
  { event := event261244
    frameStart := 260836 },
  { event := event261245
    frameStart := 260836 },
  { event := event261246
    frameStart := 260836 },
  { event := event261247
    frameStart := 260836 }
]

def eventLeaf16328 : Array AnnotatedEvent := #[
  { event := event261248
    frameStart := 260836 },
  { event := event261249
    frameStart := 260836 },
  { event := event261250
    frameStart := 260836 },
  { event := event261251
    frameStart := 260836 },
  { event := event261252
    frameStart := 260836 },
  { event := event261253
    frameStart := 260836 },
  { event := event261254
    frameStart := 260836 },
  { event := event261255
    frameStart := 260836 },
  { event := event261256
    frameStart := 260836 },
  { event := event261257
    frameStart := 260836 },
  { event := event261258
    frameStart := 260836 },
  { event := event261259
    frameStart := 260836 },
  { event := event261260
    frameStart := 260836 },
  { event := event261261
    frameStart := 260836 },
  { event := event261262
    frameStart := 260836 },
  { event := event261263
    frameStart := 260836 }
]

def eventLeaf16329 : Array AnnotatedEvent := #[
  { event := event261264
    frameStart := 260836 },
  { event := event261265
    frameStart := 260836 },
  { event := event261266
    frameStart := 260836 },
  { event := event261267
    frameStart := 260836 },
  { event := event261268
    frameStart := 260836 },
  { event := event261269
    frameStart := 260836 },
  { event := event261270
    frameStart := 260836 },
  { event := event261271
    frameStart := 260836 },
  { event := event261272
    frameStart := 260836 },
  { event := event261273
    frameStart := 260836 },
  { event := event261274
    frameStart := 260836 },
  { event := event261275
    frameStart := 260836 },
  { event := event261276
    frameStart := 260836 },
  { event := event261277
    frameStart := 260836 },
  { event := event261278
    frameStart := 260836 },
  { event := event261279
    frameStart := 260836 }
]

def eventLeaf16330 : Array AnnotatedEvent := #[
  { event := event261280
    frameStart := 260836 },
  { event := event261281
    frameStart := 260836 },
  { event := event261282
    frameStart := 260836 },
  { event := event261283
    frameStart := 260836 },
  { event := event261284
    frameStart := 260836 },
  { event := event261285
    frameStart := 260836 },
  { event := event261286
    frameStart := 260836 },
  { event := event261287
    frameStart := 260836 },
  { event := event261288
    frameStart := 260836 },
  { event := event261289
    frameStart := 260836 },
  { event := event261290
    frameStart := 260836 },
  { event := event261291
    frameStart := 260836 },
  { event := event261292
    frameStart := 260836 },
  { event := event261293
    frameStart := 260836 },
  { event := event261294
    frameStart := 260836 },
  { event := event261295
    frameStart := 260836 }
]

def eventLeaf16331 : Array AnnotatedEvent := #[
  { event := event261296
    frameStart := 260836 },
  { event := event261297
    frameStart := 260836 },
  { event := event261298
    frameStart := 260836 },
  { event := event261299
    frameStart := 260836 },
  { event := event261300
    frameStart := 260836 },
  { event := event261301
    frameStart := 260836 },
  { event := event261302
    frameStart := 260836 },
  { event := event261303
    frameStart := 260836 },
  { event := event261304
    frameStart := 260836 },
  { event := event261305
    frameStart := 260836 },
  { event := event261306
    frameStart := 260836 },
  { event := event261307
    frameStart := 260836 },
  { event := event261308
    frameStart := 260836 },
  { event := event261309
    frameStart := 260836 },
  { event := event261310
    frameStart := 260836 },
  { event := event261311
    frameStart := 260836 }
]

def eventLeaf16332 : Array AnnotatedEvent := #[
  { event := event261312
    frameStart := 260836 },
  { event := event261313
    frameStart := 260836 },
  { event := event261314
    frameStart := 260836 },
  { event := event261315
    frameStart := 260836 },
  { event := event261316
    frameStart := 260836 },
  { event := event261317
    frameStart := 260836 },
  { event := event261318
    frameStart := 260836 },
  { event := event261319
    frameStart := 260836 },
  { event := event261320
    frameStart := 260836 },
  { event := event261321
    frameStart := 260836 },
  { event := event261322
    frameStart := 260836 },
  { event := event261323
    frameStart := 260836 },
  { event := event261324
    frameStart := 260836 },
  { event := event261325
    frameStart := 260836 },
  { event := event261326
    frameStart := 260836 },
  { event := event261327
    frameStart := 260836 }
]

def eventLeaf16333 : Array AnnotatedEvent := #[
  { event := event261328
    frameStart := 260836 },
  { event := event261329
    frameStart := 260836 },
  { event := event261330
    frameStart := 260836 },
  { event := event261331
    frameStart := 260836 },
  { event := event261332
    frameStart := 260836 },
  { event := event261333
    frameStart := 260836 },
  { event := event261334
    frameStart := 260836 },
  { event := event261335
    frameStart := 260836 },
  { event := event261336
    frameStart := 260836 },
  { event := event261337
    frameStart := 260836 },
  { event := event261338
    frameStart := 260836 },
  { event := event261339
    frameStart := 260836 },
  { event := event261340
    frameStart := 260836 },
  { event := event261341
    frameStart := 260836 },
  { event := event261342
    frameStart := 260836 },
  { event := event261343
    frameStart := 260836 }
]

def eventLeaf16334 : Array AnnotatedEvent := #[
  { event := event261344
    frameStart := 260836 },
  { event := event261345
    frameStart := 260836 },
  { event := event261346
    frameStart := 260836 },
  { event := event261347
    frameStart := 260836 },
  { event := event261348
    frameStart := 260836 },
  { event := event261349
    frameStart := 260836 },
  { event := event261350
    frameStart := 260836 },
  { event := event261351
    frameStart := 260836 },
  { event := event261352
    frameStart := 260836 },
  { event := event261353
    frameStart := 260836 },
  { event := event261354
    frameStart := 260836 },
  { event := event261355
    frameStart := 260836 },
  { event := event261356
    frameStart := 260836 },
  { event := event261357
    frameStart := 260836 },
  { event := event261358
    frameStart := 260836 },
  { event := event261359
    frameStart := 260836 }
]

def eventLeaf16335 : Array AnnotatedEvent := #[
  { event := event261360
    frameStart := 260836 },
  { event := event261361
    frameStart := 260836 },
  { event := event261362
    frameStart := 260836 },
  { event := event261363
    frameStart := 260836 },
  { event := event261364
    frameStart := 260836 },
  { event := event261365
    frameStart := 260836 },
  { event := event261366
    frameStart := 260836 },
  { event := event261367
    frameStart := 260836 },
  { event := event261368
    frameStart := 260836 },
  { event := event261369
    frameStart := 260836 },
  { event := event261370
    frameStart := 260836 },
  { event := event261371
    frameStart := 260836 },
  { event := event261372
    frameStart := 260836 },
  { event := event261373
    frameStart := 260836 },
  { event := event261374
    frameStart := 260836 },
  { event := event261375
    frameStart := 260836 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events1020
