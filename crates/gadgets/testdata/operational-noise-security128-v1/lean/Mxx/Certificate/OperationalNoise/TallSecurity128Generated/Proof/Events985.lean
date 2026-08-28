import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events985

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event252160 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨46926⟩⟩, .operator (⟨252156, 2⟩, ⟨251970, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨14706⟩⟩, ⟨.program ⟨257⟩, ⟨45034⟩⟩], [⟨.program ⟨257⟩, ⟨46439⟩⟩]⟩, (-1)⟩)

def event252161 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨46926⟩⟩, .operator (⟨252156, 1⟩, ⟨251970, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨46924⟩⟩]⟩, (1)⟩)

def event252162 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46926⟩⟩) (.sum [.result 252156 .summary, .result 251970 .summary])

def exact252163RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨7195⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨45428⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact252163RawTermsValid :
    exact252163RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event252163 : Event := .resultExact (⟨.program ⟨257⟩, ⟨46926⟩⟩) exact252163RawTerms .large 252159 (.finite 2998328565150755586048) (some (252162))

def event252164 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47226⟩⟩) 0 ⟨46926⟩ 252163

def event252165 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47226⟩⟩) 1 ⟨47224⟩ 251886

def event252166 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47226⟩⟩) (.product (.predecessor 0 252164 .coefficient) (.predecessor 1 252165 .coefficient) (⟨false, false, none, none, none⟩))

def event252167 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47226⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨47224⟩⟩]⟩) [⟨.result 251886 .coefficient, false, none⟩])

def event252168 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47226⟩⟩) (.product (.result 252163 .summary) (.transfer 252167) (⟨false, false, none, none, none⟩))

def event252169 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨47226⟩⟩, .operator (⟨252163, 0⟩, ⟨251886, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47224⟩⟩]⟩, (1)⟩)

def event252170 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨47226⟩⟩, .operator (⟨252163, 1⟩, ⟨251886, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨45428⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨47224⟩⟩]⟩, (-1)⟩)

def event252171 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨47226⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨45428⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨47224⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨47224⟩⟩) ⟨46576⟩ 251883)

def event252172 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨47226⟩⟩, .relation 252171 0, ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨45428⟩⟩], [⟨.program ⟨257⟩, ⟨46576⟩⟩]⟩, (-1)⟩)

def exact252173RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47224⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨45428⟩⟩], [⟨.program ⟨257⟩, ⟨46576⟩⟩]⟩, (-1)⟩]

theorem exact252173RawTermsValid :
    exact252173RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event252173 : Event := .resultExact (⟨.program ⟨257⟩, ⟨47226⟩⟩) exact252173RawTerms .large 252166 (.finite 32194307824962751379413684715520) (some (252168))

def event252174 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46116⟩⟩) 0 ⟨45429⟩ 12102

def event252175 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46116⟩⟩) (.authority (.relationPreimageSource ⟨92⟩))

def exact252176RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨46116⟩⟩]⟩, (1)⟩]

theorem exact252176RawTermsValid :
    exact252176RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event252176 : Event := .resultExact (⟨.program ⟨257⟩, ⟨46116⟩⟩) exact252176RawTerms (.finite 5647228698) 252175 .exactZero (none)

def event252177 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46118⟩⟩) 0 ⟨46116⟩ 252176

def event252178 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46118⟩⟩) 1 ⟨2370⟩ 4

def event252179 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46118⟩⟩) (.scale (.predecessor 0 252177 .coefficient) (.value (.predecessor 1 252178 .coefficient)))

def exact252180RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨46116⟩⟩]⟩, (1)⟩]

theorem exact252180RawTermsValid :
    exact252180RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event252180 : Event := .resultExact (⟨.program ⟨257⟩, ⟨46118⟩⟩) exact252180RawTerms (.finite 5647228698) 252179 .exactZero (none)

def event252181 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46119⟩⟩) 0 ⟨5509⟩ 251495

def event252182 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46119⟩⟩) 1 ⟨46118⟩ 252180

def event252183 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46119⟩⟩) (.product (.predecessor 0 252181 .coefficient) (.predecessor 1 252182 .coefficient) (⟨false, false, none, none, none⟩))

def event252184 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46119⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨46116⟩⟩]⟩) [⟨.result 252176 .coefficient, false, none⟩])

def event252185 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46119⟩⟩) (.product (.result 251495 .summary) (.transfer 252184) (⟨false, false, none, none, none⟩))

def event252186 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨46119⟩⟩, .operator (⟨251495, 0⟩, ⟨252180, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46116⟩⟩]⟩, (1)⟩)

def event252187 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨46117⟩⟩)

def event252188 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event252189 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event252190 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨2880⟩⟩) (.authority (.operator))

def event252191 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨2880⟩⟩) (.finite 3)

def event252192 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event252193 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event252194 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event252195 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event252196 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 252195

def event252197 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 252193

def event252198 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 252196 .coefficient) (.value (.predecessor 1 252197 .coefficient)))

def event252199 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event252200 : Event := .predecessor (⟨.program ⟨257⟩, ⟨2882⟩⟩) 0 ⟨392⟩ 252199

def event252201 : Event := .predecessor (⟨.program ⟨257⟩, ⟨2882⟩⟩) 1 ⟨2880⟩ 252191

def event252202 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨2882⟩⟩) (.sum [.predecessor 0 252200 .coefficient, .predecessor 1 252201 .coefficient])

def event252203 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨2882⟩⟩) (.finite 655343)

def event252204 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5505⟩⟩) 0 ⟨2882⟩ 252203

def event252205 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5505⟩⟩) 1 ⟨5426⟩ 252189

def event252206 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5505⟩⟩) (.identity (.predecessor 1 252205 .coefficient))

def event252207 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5505⟩⟩) (.finite 655360)

def event252208 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45034⟩⟩) 0 ⟨5505⟩ 252207

def event252209 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45034⟩⟩) (.authority (.programFamilyFact))

def exact252210RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨45034⟩⟩], []⟩, (1)⟩]

theorem exact252210RawTermsValid :
    exact252210RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event252210 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45034⟩⟩) exact252210RawTerms (.finite 58) 252209 .exactZero (none)

def event252211 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14706⟩⟩) 0 ⟨5505⟩ 252207

def event252212 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14706⟩⟩) (.authority (.programFamilyFact))

def exact252213RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14706⟩⟩], []⟩, (1)⟩]

theorem exact252213RawTermsValid :
    exact252213RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event252213 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14706⟩⟩) exact252213RawTerms (.finite 58) 252212 .exactZero (none)

def event252214 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45035⟩⟩) 0 ⟨14706⟩ 252213

def event252215 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45035⟩⟩) 1 ⟨45034⟩ 252210

def event252216 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45035⟩⟩) (.product (.predecessor 0 252214 .coefficient) (.predecessor 1 252215 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event252217 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45035⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨14706⟩⟩, ⟨.program ⟨257⟩, ⟨45034⟩⟩], []⟩) [⟨.result 252213 .coefficient, true, some 1⟩, ⟨.result 252210 .coefficient, true, some 1⟩])

def event252218 : Event := .survivorFold (1) 252217

def exact252219RawTerms : List Term := []

theorem exact252219RawTermsValid :
    exact252219RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event252219 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45035⟩⟩) exact252219RawTerms (.finite 3364) 252216 (.finite 3364) (some (252217))

def event252220 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45036⟩⟩) 0 ⟨45035⟩ 252219

def event252221 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45036⟩⟩) (.identity (.predecessor 0 252220 .coefficient))

def event252222 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨45036⟩⟩) (.finite 3364)

def event252223 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45428⟩⟩) 0 ⟨45036⟩ 252222

def event252224 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45428⟩⟩) (.authority (.programFamilyFact))

def exact252225RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨45428⟩⟩], []⟩, (1)⟩]

theorem exact252225RawTermsValid :
    exact252225RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event252225 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45428⟩⟩) exact252225RawTerms (.finite 58) 252224 .exactZero (none)

def event252226 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45429⟩⟩) 0 ⟨45428⟩ 252225

def event252227 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45429⟩⟩) (.identity (.predecessor 0 252226 .coefficient))

def event252228 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨45429⟩⟩) (.finite 58)

def event252229 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46116⟩⟩) 0 ⟨45429⟩ 252228

def event252230 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46116⟩⟩) (.authority (.relationPreimageSource ⟨92⟩))

def exact252231RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨46116⟩⟩]⟩, (1)⟩]

theorem exact252231RawTermsValid :
    exact252231RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event252231 : Event := .resultExact (⟨.program ⟨257⟩, ⟨46116⟩⟩) exact252231RawTerms (.finite 5647228698) 252230 .exactZero (none)

def event252232 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35⟩⟩) (.authority (.operator))

def exact252233RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact252233RawTermsValid :
    exact252233RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event252233 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35⟩⟩) exact252233RawTerms .large 252232 .exactZero (none)

def event252234 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46117⟩⟩) 0 ⟨35⟩ 252233

def event252235 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46117⟩⟩) 1 ⟨46116⟩ 252231

def event252236 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46117⟩⟩) (.product (.predecessor 0 252234 .coefficient) (.predecessor 1 252235 .coefficient) (⟨false, false, none, none, none⟩))

def event252237 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨46117⟩⟩, .operator (⟨252233, 0⟩, ⟨252231, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46116⟩⟩]⟩, (1)⟩)

def exact252238RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46116⟩⟩]⟩, (1)⟩]

theorem exact252238RawTermsValid :
    exact252238RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event252238 : Event := .resultExact (⟨.program ⟨257⟩, ⟨46117⟩⟩) exact252238RawTerms .large 252236 .exactZero (none)

def event252239 : Event := .preFoldPolynomial 252238 [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46116⟩⟩]⟩, (1)⟩] .exactZero none

def exact252240RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46116⟩⟩]⟩, (1)⟩]

def event252240 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨46117⟩⟩) 252239 exact252240RawTerms .large 252236 .exactZero (none)

def event252241 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨47228⟩⟩)

def event252242 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event252243 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event252244 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨2880⟩⟩) (.authority (.operator))

def event252245 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨2880⟩⟩) (.finite 3)

def event252246 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event252247 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event252248 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event252249 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event252250 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 252249

def event252251 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 252247

def event252252 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 252250 .coefficient) (.value (.predecessor 1 252251 .coefficient)))

def event252253 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event252254 : Event := .predecessor (⟨.program ⟨257⟩, ⟨2882⟩⟩) 0 ⟨392⟩ 252253

def event252255 : Event := .predecessor (⟨.program ⟨257⟩, ⟨2882⟩⟩) 1 ⟨2880⟩ 252245

def event252256 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨2882⟩⟩) (.sum [.predecessor 0 252254 .coefficient, .predecessor 1 252255 .coefficient])

def event252257 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨2882⟩⟩) (.finite 655343)

def event252258 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5505⟩⟩) 0 ⟨2882⟩ 252257

def event252259 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5505⟩⟩) 1 ⟨5426⟩ 252243

def event252260 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5505⟩⟩) (.identity (.predecessor 1 252259 .coefficient))

def event252261 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5505⟩⟩) (.finite 655360)

def event252262 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45034⟩⟩) 0 ⟨5505⟩ 252261

def event252263 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45034⟩⟩) (.authority (.programFamilyFact))

def exact252264RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨45034⟩⟩], []⟩, (1)⟩]

theorem exact252264RawTermsValid :
    exact252264RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event252264 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45034⟩⟩) exact252264RawTerms (.finite 58) 252263 .exactZero (none)

def event252265 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14706⟩⟩) 0 ⟨5505⟩ 252261

def event252266 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14706⟩⟩) (.authority (.programFamilyFact))

def exact252267RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14706⟩⟩], []⟩, (1)⟩]

theorem exact252267RawTermsValid :
    exact252267RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event252267 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14706⟩⟩) exact252267RawTerms (.finite 58) 252266 .exactZero (none)

def event252268 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45035⟩⟩) 0 ⟨14706⟩ 252267

def event252269 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45035⟩⟩) 1 ⟨45034⟩ 252264

def event252270 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45035⟩⟩) (.product (.predecessor 0 252268 .coefficient) (.predecessor 1 252269 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event252271 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨45035⟩⟩, .operator (⟨252267, 0⟩, ⟨252264, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨14706⟩⟩, ⟨.program ⟨257⟩, ⟨45034⟩⟩], []⟩, (1)⟩)

def exact252272RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14706⟩⟩, ⟨.program ⟨257⟩, ⟨45034⟩⟩], []⟩, (1)⟩]

theorem exact252272RawTermsValid :
    exact252272RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event252272 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45035⟩⟩) exact252272RawTerms (.finite 3364) 252270 .exactZero (none)

def event252273 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45036⟩⟩) 0 ⟨45035⟩ 252272

def event252274 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45036⟩⟩) (.identity (.predecessor 0 252273 .coefficient))

def event252275 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨45036⟩⟩) (.finite 3364)

def event252276 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45428⟩⟩) 0 ⟨45036⟩ 252275

def event252277 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45428⟩⟩) (.authority (.programFamilyFact))

def exact252278RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨45428⟩⟩], []⟩, (1)⟩]

theorem exact252278RawTermsValid :
    exact252278RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event252278 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45428⟩⟩) exact252278RawTerms (.finite 58) 252277 .exactZero (none)

def event252279 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45429⟩⟩) 0 ⟨45428⟩ 252278

def event252280 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45429⟩⟩) (.identity (.predecessor 0 252279 .coefficient))

def event252281 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨45429⟩⟩) (.finite 58)

def event252282 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46574⟩⟩) 0 ⟨45429⟩ 252281

def event252283 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46574⟩⟩) (.authority (.programFamilyFact))

def event252284 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨46574⟩⟩) (.finite 3720)

def event252285 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨7177⟩⟩) .missing

def event252286 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46576⟩⟩) 0 ⟨7177⟩ 252285

def event252287 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46576⟩⟩) 1 ⟨46574⟩ 252284

def event252288 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46576⟩⟩) (.authority (.operator))

def exact252289RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨46576⟩⟩]⟩, (1)⟩]

theorem exact252289RawTermsValid :
    exact252289RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event252289 : Event := .resultExact (⟨.program ⟨257⟩, ⟨46576⟩⟩) exact252289RawTerms .large 252288 .exactZero (none)

def event252290 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47224⟩⟩) 0 ⟨46576⟩ 252289

def event252291 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47224⟩⟩) (.authority (.operator))

def exact252292RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨47224⟩⟩]⟩, (1)⟩]

theorem exact252292RawTermsValid :
    exact252292RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event252292 : Event := .resultExact (⟨.program ⟨257⟩, ⟨47224⟩⟩) exact252292RawTerms (.finite 8192) 252291 .exactZero (none)

def event252293 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event252294 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event252295 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46806⟩⟩) 0 ⟨45429⟩ 252281

def event252296 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46806⟩⟩) 1 ⟨136⟩ 252294

def event252297 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46806⟩⟩) (.sum [.predecessor 0 252295 .coefficient, .predecessor 1 252296 .coefficient])

def event252298 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨46806⟩⟩) (.finite 58)

def event252299 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46807⟩⟩) 0 ⟨46806⟩ 252298

def event252300 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46807⟩⟩) (.identity (.predecessor 0 252299 .coefficient))

def exact252301RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨45428⟩⟩], []⟩, (1)⟩]

theorem exact252301RawTermsValid :
    exact252301RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event252301 : Event := .resultExact (⟨.program ⟨257⟩, ⟨46807⟩⟩) exact252301RawTerms (.finite 58) 252300 .exactZero (none)

def event252302 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact252303RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact252303RawTermsValid :
    exact252303RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event252303 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact252303RawTerms .large 252302 .exactZero (none)

def event252304 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46808⟩⟩) 0 ⟨6908⟩ 252303

def event252305 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46808⟩⟩) 1 ⟨46807⟩ 252301

def event252306 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46808⟩⟩) (.product (.predecessor 0 252304 .coefficient) (.predecessor 1 252305 .coefficient) (⟨false, false, none, none, none⟩))

def event252307 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨46808⟩⟩, .operator (⟨252303, 0⟩, ⟨252301, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨45428⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact252308RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨45428⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact252308RawTermsValid :
    exact252308RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event252308 : Event := .resultExact (⟨.program ⟨257⟩, ⟨46808⟩⟩) exact252308RawTerms .large 252306 .exactZero (none)

def event252309 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7195⟩⟩) 0 ⟨7177⟩ 252285

def event252310 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7195⟩⟩) (.authority (.operator))

def exact252311RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7195⟩⟩]⟩, (1)⟩]

theorem exact252311RawTermsValid :
    exact252311RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event252311 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7195⟩⟩) exact252311RawTerms .large 252310 .exactZero (none)

def event252312 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46809⟩⟩) 0 ⟨7195⟩ 252311

def event252313 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46809⟩⟩) 1 ⟨46808⟩ 252308

def event252314 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46809⟩⟩) (.sum [.predecessor 0 252312 .coefficient, .predecessor 1 252313 .coefficient])

def exact252315RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7195⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨45428⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact252315RawTermsValid :
    exact252315RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event252315 : Event := .resultExact (⟨.program ⟨257⟩, ⟨46809⟩⟩) exact252315RawTerms .large 252314 .exactZero (none)

def event252316 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47225⟩⟩) 0 ⟨46809⟩ 252315

def event252317 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47225⟩⟩) 1 ⟨47224⟩ 252292

def event252318 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47225⟩⟩) (.product (.predecessor 0 252316 .coefficient) (.predecessor 1 252317 .coefficient) (⟨false, false, none, none, none⟩))

def event252319 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨47225⟩⟩, .operator (⟨252315, 0⟩, ⟨252292, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47224⟩⟩]⟩, (1)⟩)

def event252320 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨47225⟩⟩, .operator (⟨252315, 1⟩, ⟨252292, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨45428⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨47224⟩⟩]⟩, (-1)⟩)

def event252321 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨47225⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨45428⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨47224⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨47224⟩⟩) ⟨46576⟩ 252289)

def event252322 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨47225⟩⟩, .relation 252321 0, ⟨[⟨.program ⟨257⟩, ⟨45428⟩⟩], [⟨.program ⟨257⟩, ⟨46576⟩⟩]⟩, (-1)⟩)

def exact252323RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47224⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨45428⟩⟩], [⟨.program ⟨257⟩, ⟨46576⟩⟩]⟩, (-1)⟩]

theorem exact252323RawTermsValid :
    exact252323RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event252323 : Event := .resultExact (⟨.program ⟨257⟩, ⟨47225⟩⟩) exact252323RawTerms .large 252318 .exactZero (none)

def event252324 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45618⟩⟩) 0 ⟨45429⟩ 252281

def event252325 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45618⟩⟩) (.authority (.programFamilyFact))

def exact252326RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨45618⟩⟩], []⟩, (1)⟩]

theorem exact252326RawTermsValid :
    exact252326RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event252326 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45618⟩⟩) exact252326RawTerms (.finite 63) 252325 .exactZero (none)

def event252327 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45619⟩⟩) 0 ⟨6908⟩ 252303

def event252328 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45619⟩⟩) 1 ⟨45618⟩ 252326

def event252329 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45619⟩⟩) (.product (.predecessor 0 252327 .coefficient) (.predecessor 1 252328 .coefficient) (⟨false, true, none, none, some 1⟩))

def event252330 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨45619⟩⟩, .operator (⟨252303, 0⟩, ⟨252326, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨45618⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact252331RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨45618⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact252331RawTermsValid :
    exact252331RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event252331 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45619⟩⟩) exact252331RawTerms .large 252329 .exactZero (none)

def event252332 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7230⟩⟩) 0 ⟨7177⟩ 252285

def event252333 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7230⟩⟩) (.authority (.operator))

def exact252334RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7230⟩⟩]⟩, (1)⟩]

theorem exact252334RawTermsValid :
    exact252334RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event252334 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7230⟩⟩) exact252334RawTerms .large 252333 .exactZero (none)

def event252335 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45620⟩⟩) 0 ⟨7230⟩ 252334

def event252336 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45620⟩⟩) 1 ⟨45619⟩ 252331

def event252337 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45620⟩⟩) (.sum [.predecessor 0 252335 .coefficient, .predecessor 1 252336 .coefficient])

def exact252338RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7230⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨45618⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact252338RawTermsValid :
    exact252338RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event252338 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45620⟩⟩) exact252338RawTerms .large 252337 .exactZero (none)

def event252339 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47228⟩⟩) 0 ⟨45620⟩ 252338

def event252340 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47228⟩⟩) 1 ⟨47225⟩ 252323

def event252341 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47228⟩⟩) (.sum [.predecessor 0 252339 .coefficient, .predecessor 1 252340 .coefficient])

def exact252342RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47224⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7230⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨45428⟩⟩], [⟨.program ⟨257⟩, ⟨46576⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨45618⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact252342RawTermsValid :
    exact252342RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event252342 : Event := .resultExact (⟨.program ⟨257⟩, ⟨47228⟩⟩) exact252342RawTerms .large 252341 .exactZero (none)

def event252343 : Event := .preFoldPolynomial 252342 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47224⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7230⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨45428⟩⟩], [⟨.program ⟨257⟩, ⟨46576⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨45618⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact252344RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47224⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7230⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨45428⟩⟩], [⟨.program ⟨257⟩, ⟨46576⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨45618⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event252344 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨47228⟩⟩) 252343 exact252344RawTerms .large 252341 .exactZero (none)

def event252345 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨45429⟩⟩) ⟨⟨109⟩, ⟨92⟩, ⟨135⟩⟩ ⟨252187, 252345⟩

def event252346 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨46119⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46116⟩⟩]⟩) (1) 0 2 (.universal 252345 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46116⟩⟩]⟩) (none) 252344)

def event252347 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨46119⟩⟩, .relation 252346 1, ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨7230⟩⟩]⟩, (1)⟩)

def event252348 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨46119⟩⟩, .relation 252346 0, ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47224⟩⟩]⟩, (-1)⟩)

def event252349 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨46119⟩⟩, .relation 252346 2, ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨45428⟩⟩], [⟨.program ⟨257⟩, ⟨46576⟩⟩]⟩, (1)⟩)

def event252350 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨46119⟩⟩, .relation 252346 3, ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨45618⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact252351RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47224⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨7230⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨45428⟩⟩], [⟨.program ⟨257⟩, ⟨46576⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨45618⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact252351RawTermsValid :
    exact252351RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event252351 : Event := .resultExact (⟨.program ⟨257⟩, ⟨46119⟩⟩) exact252351RawTerms .large 252183 (.finite 202072841853861888) (some (252185))

def event252352 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47227⟩⟩) 0 ⟨46119⟩ 252351

def event252353 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47227⟩⟩) 1 ⟨47226⟩ 252173

def event252354 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47227⟩⟩) (.sum [.predecessor 0 252352 .coefficient, .predecessor 1 252353 .coefficient])

def event252355 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨47227⟩⟩, .operator (⟨252351, 0⟩, ⟨252173, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47224⟩⟩]⟩, (1)⟩)

def event252356 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨47227⟩⟩, .operator (⟨252351, 2⟩, ⟨252173, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨45428⟩⟩], [⟨.program ⟨257⟩, ⟨46576⟩⟩]⟩, (-1)⟩)

def event252357 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47227⟩⟩) (.sum [.result 252351 .summary, .result 252173 .summary])

def exact252358RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨7230⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨45618⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact252358RawTermsValid :
    exact252358RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event252358 : Event := .resultExact (⟨.program ⟨257⟩, ⟨47227⟩⟩) exact252358RawTerms .large 252354 (.finite 32194307824962953452255538577408) (some (252357))

def event252359 : Event := .predecessor (⟨.program ⟨257⟩, ⟨43894⟩⟩) 0 ⟨42749⟩ 12125

def event252360 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨43894⟩⟩) (.authority (.programFamilyFact))

def event252361 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨43894⟩⟩) (.finite 3720)

def event252362 : Event := .predecessor (⟨.program ⟨257⟩, ⟨43896⟩⟩) 0 ⟨7177⟩ 15500

def event252363 : Event := .predecessor (⟨.program ⟨257⟩, ⟨43896⟩⟩) 1 ⟨43894⟩ 252361

def event252364 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨43896⟩⟩) (.authority (.operator))

def exact252365RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨43896⟩⟩]⟩, (1)⟩]

theorem exact252365RawTermsValid :
    exact252365RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event252365 : Event := .resultExact (⟨.program ⟨257⟩, ⟨43896⟩⟩) exact252365RawTerms .large 252364 .exactZero (none)

def event252366 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44544⟩⟩) 0 ⟨43896⟩ 252365

def event252367 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44544⟩⟩) (.authority (.operator))

def exact252368RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨44544⟩⟩]⟩, (1)⟩]

theorem exact252368RawTermsValid :
    exact252368RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event252368 : Event := .resultExact (⟨.program ⟨257⟩, ⟨44544⟩⟩) exact252368RawTerms (.finite 8192) 252367 .exactZero (none)

def event252369 : Event := .predecessor (⟨.program ⟨257⟩, ⟨43758⟩⟩) 0 ⟨42356⟩ 12119

def event252370 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨43758⟩⟩) (.authority (.programFamilyFact))

def event252371 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨43758⟩⟩) (.finite 3720)

def event252372 : Event := .predecessor (⟨.program ⟨257⟩, ⟨43759⟩⟩) 0 ⟨7177⟩ 15500

def event252373 : Event := .predecessor (⟨.program ⟨257⟩, ⟨43759⟩⟩) 1 ⟨43758⟩ 252371

def event252374 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨43759⟩⟩) (.authority (.operator))

def exact252375RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨43759⟩⟩]⟩, (1)⟩]

theorem exact252375RawTermsValid :
    exact252375RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event252375 : Event := .resultExact (⟨.program ⟨257⟩, ⟨43759⟩⟩) exact252375RawTerms .large 252374 .exactZero (none)

def event252376 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44244⟩⟩) 0 ⟨43759⟩ 252375

def event252377 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44244⟩⟩) (.authority (.operator))

def exact252378RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨44244⟩⟩]⟩, (1)⟩]

theorem exact252378RawTermsValid :
    exact252378RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event252378 : Event := .resultExact (⟨.program ⟨257⟩, ⟨44244⟩⟩) exact252378RawTerms (.finite 8192) 252377 .exactZero (none)

def event252379 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42357⟩⟩) 0 ⟨42354⟩ 12108

def event252380 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42357⟩⟩) 1 ⟨6925⟩ 251403

def event252381 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42357⟩⟩) (.tensor (.predecessor 0 252379 .coefficient) (.predecessor 1 252380 .coefficient) true false)

def event252382 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨42357⟩⟩, .operator (⟨12108, 0⟩, ⟨251403, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨42354⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact252383RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨42354⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact252383RawTermsValid :
    exact252383RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event252383 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42357⟩⟩) exact252383RawTerms .large 252381 .exactZero (none)

def event252384 : Event := .predecessor (⟨.program ⟨257⟩, ⟨8019⟩⟩) 0 ⟨5507⟩ 251273

def event252385 : Event := .predecessor (⟨.program ⟨257⟩, ⟨8019⟩⟩) 1 ⟨7283⟩ 18082

def event252386 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨8019⟩⟩) (.product (.predecessor 0 252384 .coefficient) (.predecessor 1 252385 .coefficient) (⟨false, false, none, none, none⟩))

def event252387 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨8019⟩⟩, .operator (⟨251273, 0⟩, ⟨18082, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨7283⟩⟩]⟩, (1)⟩)

def exact252388RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨7283⟩⟩]⟩, (1)⟩]

theorem exact252388RawTermsValid :
    exact252388RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event252388 : Event := .resultExact (⟨.program ⟨257⟩, ⟨8019⟩⟩) exact252388RawTerms .large 252386 .exactZero (none)

def event252389 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42358⟩⟩) 0 ⟨8019⟩ 252388

def event252390 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42358⟩⟩) 1 ⟨42357⟩ 252383

def event252391 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42358⟩⟩) (.sum [.predecessor 0 252389 .coefficient, .predecessor 1 252390 .coefficient])

def exact252392RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨7283⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨42354⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact252392RawTermsValid :
    exact252392RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event252392 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42358⟩⟩) exact252392RawTerms .large 252391 .exactZero (none)

def event252393 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42359⟩⟩) 0 ⟨42358⟩ 252392

def event252394 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42359⟩⟩) 1 ⟨109⟩ 18074

def event252395 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42359⟩⟩) (.sum [.predecessor 0 252393 .coefficient, .predecessor 1 252394 .coefficient])

def event252396 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42359⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨109⟩⟩]⟩) [⟨.result 18074 .coefficient, false, none⟩])

def event252397 : Event := .survivorFold (1) 252396

def exact252398RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨7283⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨42354⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact252398RawTermsValid :
    exact252398RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event252398 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42359⟩⟩) exact252398RawTerms .large 252395 (.finite 26) (some (252396))

def event252399 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42360⟩⟩) 0 ⟨42359⟩ 252398

def event252400 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42360⟩⟩) 1 ⟨14406⟩ 12111

def event252401 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42360⟩⟩) (.product (.predecessor 0 252399 .coefficient) (.predecessor 1 252400 .coefficient) (⟨false, true, none, none, some 1⟩))

def event252402 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42360⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨14406⟩⟩], []⟩) [⟨.result 12111 .coefficient, true, some 1⟩])

def event252403 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42360⟩⟩) (.product (.result 252398 .summary) (.transfer 252402) (⟨false, false, none, none, none⟩))

def event252404 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨42360⟩⟩, .operator (⟨252398, 1⟩, ⟨12111, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨14406⟩⟩, ⟨.program ⟨257⟩, ⟨42354⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event252405 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨42360⟩⟩, .operator (⟨252398, 0⟩, ⟨12111, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨14406⟩⟩], [⟨.program ⟨257⟩, ⟨7283⟩⟩]⟩, (1)⟩)

def exact252406RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨14406⟩⟩], [⟨.program ⟨257⟩, ⟨7283⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨14406⟩⟩, ⟨.program ⟨257⟩, ⟨42354⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact252406RawTermsValid :
    exact252406RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event252406 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42360⟩⟩) exact252406RawTerms .large 252401 (.finite 44302336) (some (252403))

def event252407 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14407⟩⟩) 0 ⟨14406⟩ 12111

def event252408 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14407⟩⟩) 1 ⟨6925⟩ 251403

def event252409 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14407⟩⟩) (.tensor (.predecessor 0 252407 .coefficient) (.predecessor 1 252408 .coefficient) true false)

def event252410 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨14407⟩⟩, .operator (⟨12111, 0⟩, ⟨251403, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨14406⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact252411RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨14406⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact252411RawTermsValid :
    exact252411RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event252411 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14407⟩⟩) exact252411RawTerms .large 252409 .exactZero (none)

def event252412 : Event := .predecessor (⟨.program ⟨257⟩, ⟨8036⟩⟩) 0 ⟨5507⟩ 251273

def event252413 : Event := .predecessor (⟨.program ⟨257⟩, ⟨8036⟩⟩) 1 ⟨7300⟩ 18123

def event252414 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨8036⟩⟩) (.product (.predecessor 0 252412 .coefficient) (.predecessor 1 252413 .coefficient) (⟨false, false, none, none, none⟩))

def event252415 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨8036⟩⟩, .operator (⟨251273, 0⟩, ⟨18123, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨7300⟩⟩]⟩, (1)⟩)

def eventLeaf15760 : Array AnnotatedEvent := #[
  { event := event252160
    frameStart := 0 },
  { event := event252161
    frameStart := 0 },
  { event := event252162
    frameStart := 0 },
  { event := event252163
    frameStart := 0 },
  { event := event252164
    frameStart := 0 },
  { event := event252165
    frameStart := 0 },
  { event := event252166
    frameStart := 0 },
  { event := event252167
    frameStart := 0 },
  { event := event252168
    frameStart := 0 },
  { event := event252169
    frameStart := 0 },
  { event := event252170
    frameStart := 0 },
  { event := event252171
    frameStart := 0 },
  { event := event252172
    frameStart := 0 },
  { event := event252173
    frameStart := 0 },
  { event := event252174
    frameStart := 0 },
  { event := event252175
    frameStart := 0 }
]

def eventLeaf15761 : Array AnnotatedEvent := #[
  { event := event252176
    frameStart := 0 },
  { event := event252177
    frameStart := 0 },
  { event := event252178
    frameStart := 0 },
  { event := event252179
    frameStart := 0 },
  { event := event252180
    frameStart := 0 },
  { event := event252181
    frameStart := 0 },
  { event := event252182
    frameStart := 0 },
  { event := event252183
    frameStart := 0 },
  { event := event252184
    frameStart := 0 },
  { event := event252185
    frameStart := 0 },
  { event := event252186
    frameStart := 0 },
  { event := event252187
    frameStart := 252187 },
  { event := event252188
    frameStart := 252187 },
  { event := event252189
    frameStart := 252187 },
  { event := event252190
    frameStart := 252187 },
  { event := event252191
    frameStart := 252187 }
]

def eventLeaf15762 : Array AnnotatedEvent := #[
  { event := event252192
    frameStart := 252187 },
  { event := event252193
    frameStart := 252187 },
  { event := event252194
    frameStart := 252187 },
  { event := event252195
    frameStart := 252187 },
  { event := event252196
    frameStart := 252187 },
  { event := event252197
    frameStart := 252187 },
  { event := event252198
    frameStart := 252187 },
  { event := event252199
    frameStart := 252187 },
  { event := event252200
    frameStart := 252187 },
  { event := event252201
    frameStart := 252187 },
  { event := event252202
    frameStart := 252187 },
  { event := event252203
    frameStart := 252187 },
  { event := event252204
    frameStart := 252187 },
  { event := event252205
    frameStart := 252187 },
  { event := event252206
    frameStart := 252187 },
  { event := event252207
    frameStart := 252187 }
]

def eventLeaf15763 : Array AnnotatedEvent := #[
  { event := event252208
    frameStart := 252187 },
  { event := event252209
    frameStart := 252187 },
  { event := event252210
    frameStart := 252187 },
  { event := event252211
    frameStart := 252187 },
  { event := event252212
    frameStart := 252187 },
  { event := event252213
    frameStart := 252187 },
  { event := event252214
    frameStart := 252187 },
  { event := event252215
    frameStart := 252187 },
  { event := event252216
    frameStart := 252187 },
  { event := event252217
    frameStart := 252187 },
  { event := event252218
    frameStart := 252187 },
  { event := event252219
    frameStart := 252187 },
  { event := event252220
    frameStart := 252187 },
  { event := event252221
    frameStart := 252187 },
  { event := event252222
    frameStart := 252187 },
  { event := event252223
    frameStart := 252187 }
]

def eventLeaf15764 : Array AnnotatedEvent := #[
  { event := event252224
    frameStart := 252187 },
  { event := event252225
    frameStart := 252187 },
  { event := event252226
    frameStart := 252187 },
  { event := event252227
    frameStart := 252187 },
  { event := event252228
    frameStart := 252187 },
  { event := event252229
    frameStart := 252187 },
  { event := event252230
    frameStart := 252187 },
  { event := event252231
    frameStart := 252187 },
  { event := event252232
    frameStart := 252187 },
  { event := event252233
    frameStart := 252187 },
  { event := event252234
    frameStart := 252187 },
  { event := event252235
    frameStart := 252187 },
  { event := event252236
    frameStart := 252187 },
  { event := event252237
    frameStart := 252187 },
  { event := event252238
    frameStart := 252187 },
  { event := event252239
    frameStart := 252187 }
]

def eventLeaf15765 : Array AnnotatedEvent := #[
  { event := event252240
    frameStart := 252187 },
  { event := event252241
    frameStart := 252241 },
  { event := event252242
    frameStart := 252241 },
  { event := event252243
    frameStart := 252241 },
  { event := event252244
    frameStart := 252241 },
  { event := event252245
    frameStart := 252241 },
  { event := event252246
    frameStart := 252241 },
  { event := event252247
    frameStart := 252241 },
  { event := event252248
    frameStart := 252241 },
  { event := event252249
    frameStart := 252241 },
  { event := event252250
    frameStart := 252241 },
  { event := event252251
    frameStart := 252241 },
  { event := event252252
    frameStart := 252241 },
  { event := event252253
    frameStart := 252241 },
  { event := event252254
    frameStart := 252241 },
  { event := event252255
    frameStart := 252241 }
]

def eventLeaf15766 : Array AnnotatedEvent := #[
  { event := event252256
    frameStart := 252241 },
  { event := event252257
    frameStart := 252241 },
  { event := event252258
    frameStart := 252241 },
  { event := event252259
    frameStart := 252241 },
  { event := event252260
    frameStart := 252241 },
  { event := event252261
    frameStart := 252241 },
  { event := event252262
    frameStart := 252241 },
  { event := event252263
    frameStart := 252241 },
  { event := event252264
    frameStart := 252241 },
  { event := event252265
    frameStart := 252241 },
  { event := event252266
    frameStart := 252241 },
  { event := event252267
    frameStart := 252241 },
  { event := event252268
    frameStart := 252241 },
  { event := event252269
    frameStart := 252241 },
  { event := event252270
    frameStart := 252241 },
  { event := event252271
    frameStart := 252241 }
]

def eventLeaf15767 : Array AnnotatedEvent := #[
  { event := event252272
    frameStart := 252241 },
  { event := event252273
    frameStart := 252241 },
  { event := event252274
    frameStart := 252241 },
  { event := event252275
    frameStart := 252241 },
  { event := event252276
    frameStart := 252241 },
  { event := event252277
    frameStart := 252241 },
  { event := event252278
    frameStart := 252241 },
  { event := event252279
    frameStart := 252241 },
  { event := event252280
    frameStart := 252241 },
  { event := event252281
    frameStart := 252241 },
  { event := event252282
    frameStart := 252241 },
  { event := event252283
    frameStart := 252241 },
  { event := event252284
    frameStart := 252241 },
  { event := event252285
    frameStart := 252241 },
  { event := event252286
    frameStart := 252241 },
  { event := event252287
    frameStart := 252241 }
]

def eventLeaf15768 : Array AnnotatedEvent := #[
  { event := event252288
    frameStart := 252241 },
  { event := event252289
    frameStart := 252241 },
  { event := event252290
    frameStart := 252241 },
  { event := event252291
    frameStart := 252241 },
  { event := event252292
    frameStart := 252241 },
  { event := event252293
    frameStart := 252241 },
  { event := event252294
    frameStart := 252241 },
  { event := event252295
    frameStart := 252241 },
  { event := event252296
    frameStart := 252241 },
  { event := event252297
    frameStart := 252241 },
  { event := event252298
    frameStart := 252241 },
  { event := event252299
    frameStart := 252241 },
  { event := event252300
    frameStart := 252241 },
  { event := event252301
    frameStart := 252241 },
  { event := event252302
    frameStart := 252241 },
  { event := event252303
    frameStart := 252241 }
]

def eventLeaf15769 : Array AnnotatedEvent := #[
  { event := event252304
    frameStart := 252241 },
  { event := event252305
    frameStart := 252241 },
  { event := event252306
    frameStart := 252241 },
  { event := event252307
    frameStart := 252241 },
  { event := event252308
    frameStart := 252241 },
  { event := event252309
    frameStart := 252241 },
  { event := event252310
    frameStart := 252241 },
  { event := event252311
    frameStart := 252241 },
  { event := event252312
    frameStart := 252241 },
  { event := event252313
    frameStart := 252241 },
  { event := event252314
    frameStart := 252241 },
  { event := event252315
    frameStart := 252241 },
  { event := event252316
    frameStart := 252241 },
  { event := event252317
    frameStart := 252241 },
  { event := event252318
    frameStart := 252241 },
  { event := event252319
    frameStart := 252241 }
]

def eventLeaf15770 : Array AnnotatedEvent := #[
  { event := event252320
    frameStart := 252241 },
  { event := event252321
    frameStart := 252241 },
  { event := event252322
    frameStart := 252241 },
  { event := event252323
    frameStart := 252241 },
  { event := event252324
    frameStart := 252241 },
  { event := event252325
    frameStart := 252241 },
  { event := event252326
    frameStart := 252241 },
  { event := event252327
    frameStart := 252241 },
  { event := event252328
    frameStart := 252241 },
  { event := event252329
    frameStart := 252241 },
  { event := event252330
    frameStart := 252241 },
  { event := event252331
    frameStart := 252241 },
  { event := event252332
    frameStart := 252241 },
  { event := event252333
    frameStart := 252241 },
  { event := event252334
    frameStart := 252241 },
  { event := event252335
    frameStart := 252241 }
]

def eventLeaf15771 : Array AnnotatedEvent := #[
  { event := event252336
    frameStart := 252241 },
  { event := event252337
    frameStart := 252241 },
  { event := event252338
    frameStart := 252241 },
  { event := event252339
    frameStart := 252241 },
  { event := event252340
    frameStart := 252241 },
  { event := event252341
    frameStart := 252241 },
  { event := event252342
    frameStart := 252241 },
  { event := event252343
    frameStart := 252241 },
  { event := event252344
    frameStart := 252241 },
  { event := event252345
    frameStart := 0 },
  { event := event252346
    frameStart := 0 },
  { event := event252347
    frameStart := 0 },
  { event := event252348
    frameStart := 0 },
  { event := event252349
    frameStart := 0 },
  { event := event252350
    frameStart := 0 },
  { event := event252351
    frameStart := 0 }
]

def eventLeaf15772 : Array AnnotatedEvent := #[
  { event := event252352
    frameStart := 0 },
  { event := event252353
    frameStart := 0 },
  { event := event252354
    frameStart := 0 },
  { event := event252355
    frameStart := 0 },
  { event := event252356
    frameStart := 0 },
  { event := event252357
    frameStart := 0 },
  { event := event252358
    frameStart := 0 },
  { event := event252359
    frameStart := 0 },
  { event := event252360
    frameStart := 0 },
  { event := event252361
    frameStart := 0 },
  { event := event252362
    frameStart := 0 },
  { event := event252363
    frameStart := 0 },
  { event := event252364
    frameStart := 0 },
  { event := event252365
    frameStart := 0 },
  { event := event252366
    frameStart := 0 },
  { event := event252367
    frameStart := 0 }
]

def eventLeaf15773 : Array AnnotatedEvent := #[
  { event := event252368
    frameStart := 0 },
  { event := event252369
    frameStart := 0 },
  { event := event252370
    frameStart := 0 },
  { event := event252371
    frameStart := 0 },
  { event := event252372
    frameStart := 0 },
  { event := event252373
    frameStart := 0 },
  { event := event252374
    frameStart := 0 },
  { event := event252375
    frameStart := 0 },
  { event := event252376
    frameStart := 0 },
  { event := event252377
    frameStart := 0 },
  { event := event252378
    frameStart := 0 },
  { event := event252379
    frameStart := 0 },
  { event := event252380
    frameStart := 0 },
  { event := event252381
    frameStart := 0 },
  { event := event252382
    frameStart := 0 },
  { event := event252383
    frameStart := 0 }
]

def eventLeaf15774 : Array AnnotatedEvent := #[
  { event := event252384
    frameStart := 0 },
  { event := event252385
    frameStart := 0 },
  { event := event252386
    frameStart := 0 },
  { event := event252387
    frameStart := 0 },
  { event := event252388
    frameStart := 0 },
  { event := event252389
    frameStart := 0 },
  { event := event252390
    frameStart := 0 },
  { event := event252391
    frameStart := 0 },
  { event := event252392
    frameStart := 0 },
  { event := event252393
    frameStart := 0 },
  { event := event252394
    frameStart := 0 },
  { event := event252395
    frameStart := 0 },
  { event := event252396
    frameStart := 0 },
  { event := event252397
    frameStart := 0 },
  { event := event252398
    frameStart := 0 },
  { event := event252399
    frameStart := 0 }
]

def eventLeaf15775 : Array AnnotatedEvent := #[
  { event := event252400
    frameStart := 0 },
  { event := event252401
    frameStart := 0 },
  { event := event252402
    frameStart := 0 },
  { event := event252403
    frameStart := 0 },
  { event := event252404
    frameStart := 0 },
  { event := event252405
    frameStart := 0 },
  { event := event252406
    frameStart := 0 },
  { event := event252407
    frameStart := 0 },
  { event := event252408
    frameStart := 0 },
  { event := event252409
    frameStart := 0 },
  { event := event252410
    frameStart := 0 },
  { event := event252411
    frameStart := 0 },
  { event := event252412
    frameStart := 0 },
  { event := event252413
    frameStart := 0 },
  { event := event252414
    frameStart := 0 },
  { event := event252415
    frameStart := 0 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events985
