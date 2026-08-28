import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events067

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event17152 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48504⟩⟩) 1 ⟨2370⟩ 4

def event17153 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48504⟩⟩) (.scale (.predecessor 0 17151 .coefficient) (.value (.predecessor 1 17152 .coefficient)))

def exact17154RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨48502⟩⟩]⟩, (1)⟩]

theorem exact17154RawTermsValid :
    exact17154RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17154 : Event := .resultExact (⟨.program ⟨257⟩, ⟨48504⟩⟩) exact17154RawTerms (.finite 5647228698) 17153 .exactZero (none)

def event17155 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨22⟩⟩) (.authority (.operator))

def exact17156RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨22⟩⟩]⟩, (1)⟩]

theorem exact17156RawTermsValid :
    exact17156RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17156 : Event := .resultExact (⟨.program ⟨257⟩, ⟨22⟩⟩) exact17156RawTerms (.finite 26) 17155 .exactZero (none)

def event17157 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35⟩⟩) (.authority (.operator))

def exact17158RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact17158RawTermsValid :
    exact17158RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17158 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35⟩⟩) exact17158RawTerms .large 17157 .exactZero (none)

def event17159 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5442⟩⟩) 0 ⟨5441⟩ 16922

def event17160 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5442⟩⟩) 1 ⟨35⟩ 17158

def event17161 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5442⟩⟩) (.product (.predecessor 0 17159 .coefficient) (.predecessor 1 17160 .coefficient) (⟨false, false, none, none, none⟩))

def event17162 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨5442⟩⟩, .operator (⟨16922, 0⟩, ⟨17158, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩)

def exact17163RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact17163RawTermsValid :
    exact17163RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17163 : Event := .resultExact (⟨.program ⟨257⟩, ⟨5442⟩⟩) exact17163RawTerms .large 17161 .exactZero (none)

def event17164 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5443⟩⟩) 0 ⟨5442⟩ 17163

def event17165 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5443⟩⟩) 1 ⟨22⟩ 17156

def event17166 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5443⟩⟩) (.sum [.predecessor 0 17164 .coefficient, .predecessor 1 17165 .coefficient])

def event17167 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5443⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨22⟩⟩]⟩) [⟨.result 17156 .coefficient, false, none⟩])

def event17168 : Event := .survivorFold (1) 17167

def exact17169RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact17169RawTermsValid :
    exact17169RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17169 : Event := .resultExact (⟨.program ⟨257⟩, ⟨5443⟩⟩) exact17169RawTerms .large 17166 (.finite 26) (some (17167))

def event17170 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48505⟩⟩) 0 ⟨5443⟩ 17169

def event17171 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48505⟩⟩) 1 ⟨48504⟩ 17154

def event17172 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48505⟩⟩) (.product (.predecessor 0 17170 .coefficient) (.predecessor 1 17171 .coefficient) (⟨false, false, none, none, none⟩))

def event17173 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48505⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨48502⟩⟩]⟩) [⟨.result 17150 .coefficient, false, none⟩])

def event17174 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48505⟩⟩) (.product (.result 17169 .summary) (.transfer 17173) (⟨false, false, none, none, none⟩))

def event17175 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨48505⟩⟩, .operator (⟨17169, 0⟩, ⟨17154, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48502⟩⟩]⟩, (1)⟩)

def event17176 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨48503⟩⟩)

def event17177 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event17178 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event17179 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨140⟩⟩) (.authority (.operator))

def event17180 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨140⟩⟩) (.finite 19)

def event17181 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event17182 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event17183 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event17184 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event17185 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 17184

def event17186 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 17182

def event17187 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 17185 .coefficient) (.value (.predecessor 1 17186 .coefficient)))

def event17188 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event17189 : Event := .predecessor (⟨.program ⟨257⟩, ⟨393⟩⟩) 0 ⟨392⟩ 17188

def event17190 : Event := .predecessor (⟨.program ⟨257⟩, ⟨393⟩⟩) 1 ⟨140⟩ 17180

def event17191 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨393⟩⟩) (.sum [.predecessor 0 17189 .coefficient, .predecessor 1 17190 .coefficient])

def event17192 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨393⟩⟩) (.finite 655359)

def event17193 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5439⟩⟩) 0 ⟨393⟩ 17192

def event17194 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5439⟩⟩) 1 ⟨5426⟩ 17178

def event17195 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5439⟩⟩) (.identity (.predecessor 1 17194 .coefficient))

def event17196 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5439⟩⟩) (.finite 655360)

def event17197 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47626⟩⟩) 0 ⟨5439⟩ 17196

def event17198 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47626⟩⟩) (.authority (.programFamilyFact))

def exact17199RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨47626⟩⟩], []⟩, (1)⟩]

theorem exact17199RawTermsValid :
    exact17199RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17199 : Event := .resultExact (⟨.program ⟨257⟩, ⟨47626⟩⟩) exact17199RawTerms (.finite 60) 17198 .exactZero (none)

def event17200 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14951⟩⟩) 0 ⟨5439⟩ 17196

def event17201 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14951⟩⟩) (.authority (.programFamilyFact))

def exact17202RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14951⟩⟩], []⟩, (1)⟩]

theorem exact17202RawTermsValid :
    exact17202RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17202 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14951⟩⟩) exact17202RawTerms (.finite 60) 17201 .exactZero (none)

def event17203 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47627⟩⟩) 0 ⟨14951⟩ 17202

def event17204 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47627⟩⟩) 1 ⟨47626⟩ 17199

def event17205 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47627⟩⟩) (.product (.predecessor 0 17203 .coefficient) (.predecessor 1 17204 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event17206 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47627⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨14951⟩⟩, ⟨.program ⟨257⟩, ⟨47626⟩⟩], []⟩) [⟨.result 17202 .coefficient, true, some 1⟩, ⟨.result 17199 .coefficient, true, some 1⟩])

def event17207 : Event := .survivorFold (1) 17206

def exact17208RawTerms : List Term := []

theorem exact17208RawTermsValid :
    exact17208RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17208 : Event := .resultExact (⟨.program ⟨257⟩, ⟨47627⟩⟩) exact17208RawTerms (.finite 3600) 17205 (.finite 3600) (some (17206))

def event17209 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47628⟩⟩) 0 ⟨47627⟩ 17208

def event17210 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47628⟩⟩) (.identity (.predecessor 0 17209 .coefficient))

def event17211 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨47628⟩⟩) (.finite 3600)

def event17212 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48502⟩⟩) 0 ⟨47628⟩ 17211

def event17213 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48502⟩⟩) (.authority (.relationPreimageSource ⟨54⟩))

def exact17214RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨48502⟩⟩]⟩, (1)⟩]

theorem exact17214RawTermsValid :
    exact17214RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17214 : Event := .resultExact (⟨.program ⟨257⟩, ⟨48502⟩⟩) exact17214RawTerms (.finite 5647228698) 17213 .exactZero (none)

def event17215 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35⟩⟩) (.authority (.operator))

def exact17216RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact17216RawTermsValid :
    exact17216RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17216 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35⟩⟩) exact17216RawTerms .large 17215 .exactZero (none)

def event17217 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48503⟩⟩) 0 ⟨35⟩ 17216

def event17218 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48503⟩⟩) 1 ⟨48502⟩ 17214

def event17219 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48503⟩⟩) (.product (.predecessor 0 17217 .coefficient) (.predecessor 1 17218 .coefficient) (⟨false, false, none, none, none⟩))

def event17220 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨48503⟩⟩, .operator (⟨17216, 0⟩, ⟨17214, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48502⟩⟩]⟩, (1)⟩)

def exact17221RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48502⟩⟩]⟩, (1)⟩]

theorem exact17221RawTermsValid :
    exact17221RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17221 : Event := .resultExact (⟨.program ⟨257⟩, ⟨48503⟩⟩) exact17221RawTerms .large 17219 .exactZero (none)

def event17222 : Event := .preFoldPolynomial 17221 [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48502⟩⟩]⟩, (1)⟩] .exactZero none

def exact17223RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48502⟩⟩]⟩, (1)⟩]

def event17223 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨48503⟩⟩) 17222 exact17223RawTerms .large 17219 .exactZero (none)

def event17224 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨49567⟩⟩)

def event17225 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event17226 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event17227 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨140⟩⟩) (.authority (.operator))

def event17228 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨140⟩⟩) (.finite 19)

def event17229 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event17230 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event17231 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event17232 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event17233 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 17232

def event17234 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 17230

def event17235 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 17233 .coefficient) (.value (.predecessor 1 17234 .coefficient)))

def event17236 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event17237 : Event := .predecessor (⟨.program ⟨257⟩, ⟨393⟩⟩) 0 ⟨392⟩ 17236

def event17238 : Event := .predecessor (⟨.program ⟨257⟩, ⟨393⟩⟩) 1 ⟨140⟩ 17228

def event17239 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨393⟩⟩) (.sum [.predecessor 0 17237 .coefficient, .predecessor 1 17238 .coefficient])

def event17240 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨393⟩⟩) (.finite 655359)

def event17241 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5439⟩⟩) 0 ⟨393⟩ 17240

def event17242 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5439⟩⟩) 1 ⟨5426⟩ 17226

def event17243 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5439⟩⟩) (.identity (.predecessor 1 17242 .coefficient))

def event17244 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5439⟩⟩) (.finite 655360)

def event17245 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47626⟩⟩) 0 ⟨5439⟩ 17244

def event17246 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47626⟩⟩) (.authority (.programFamilyFact))

def exact17247RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨47626⟩⟩], []⟩, (1)⟩]

theorem exact17247RawTermsValid :
    exact17247RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17247 : Event := .resultExact (⟨.program ⟨257⟩, ⟨47626⟩⟩) exact17247RawTerms (.finite 60) 17246 .exactZero (none)

def event17248 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14951⟩⟩) 0 ⟨5439⟩ 17244

def event17249 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14951⟩⟩) (.authority (.programFamilyFact))

def exact17250RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14951⟩⟩], []⟩, (1)⟩]

theorem exact17250RawTermsValid :
    exact17250RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17250 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14951⟩⟩) exact17250RawTerms (.finite 60) 17249 .exactZero (none)

def event17251 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47627⟩⟩) 0 ⟨14951⟩ 17250

def event17252 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47627⟩⟩) 1 ⟨47626⟩ 17247

def event17253 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47627⟩⟩) (.product (.predecessor 0 17251 .coefficient) (.predecessor 1 17252 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event17254 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨47627⟩⟩, .operator (⟨17250, 0⟩, ⟨17247, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨14951⟩⟩, ⟨.program ⟨257⟩, ⟨47626⟩⟩], []⟩, (1)⟩)

def exact17255RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14951⟩⟩, ⟨.program ⟨257⟩, ⟨47626⟩⟩], []⟩, (1)⟩]

theorem exact17255RawTermsValid :
    exact17255RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17255 : Event := .resultExact (⟨.program ⟨257⟩, ⟨47627⟩⟩) exact17255RawTerms (.finite 3600) 17253 .exactZero (none)

def event17256 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47628⟩⟩) 0 ⟨47627⟩ 17255

def event17257 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47628⟩⟩) (.identity (.predecessor 0 17256 .coefficient))

def event17258 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨47628⟩⟩) (.finite 3600)

def event17259 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49096⟩⟩) 0 ⟨47628⟩ 17258

def event17260 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49096⟩⟩) (.authority (.programFamilyFact))

def event17261 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨49096⟩⟩) (.finite 3720)

def event17262 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨7177⟩⟩) .missing

def event17263 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49097⟩⟩) 0 ⟨7177⟩ 17262

def event17264 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49097⟩⟩) 1 ⟨49096⟩ 17261

def event17265 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49097⟩⟩) (.authority (.operator))

def exact17266RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨49097⟩⟩]⟩, (1)⟩]

theorem exact17266RawTermsValid :
    exact17266RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17266 : Event := .resultExact (⟨.program ⟨257⟩, ⟨49097⟩⟩) exact17266RawTerms .large 17265 .exactZero (none)

def event17267 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49563⟩⟩) 0 ⟨49097⟩ 17266

def event17268 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49563⟩⟩) (.authority (.operator))

def exact17269RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨49563⟩⟩]⟩, (1)⟩]

theorem exact17269RawTermsValid :
    exact17269RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17269 : Event := .resultExact (⟨.program ⟨257⟩, ⟨49563⟩⟩) exact17269RawTerms (.finite 8192) 17268 .exactZero (none)

def event17270 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event17271 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event17272 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49390⟩⟩) 0 ⟨47628⟩ 17258

def event17273 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49390⟩⟩) 1 ⟨136⟩ 17271

def event17274 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49390⟩⟩) (.sum [.predecessor 0 17272 .coefficient, .predecessor 1 17273 .coefficient])

def event17275 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨49390⟩⟩) (.finite 3600)

def event17276 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49391⟩⟩) 0 ⟨49390⟩ 17275

def event17277 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49391⟩⟩) (.identity (.predecessor 0 17276 .coefficient))

def exact17278RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14951⟩⟩, ⟨.program ⟨257⟩, ⟨47626⟩⟩], []⟩, (1)⟩]

theorem exact17278RawTermsValid :
    exact17278RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17278 : Event := .resultExact (⟨.program ⟨257⟩, ⟨49391⟩⟩) exact17278RawTerms (.finite 3600) 17277 .exactZero (none)

def event17279 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact17280RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact17280RawTermsValid :
    exact17280RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17280 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact17280RawTerms .large 17279 .exactZero (none)

def event17281 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49392⟩⟩) 0 ⟨6908⟩ 17280

def event17282 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49392⟩⟩) 1 ⟨49391⟩ 17278

def event17283 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49392⟩⟩) (.product (.predecessor 0 17281 .coefficient) (.predecessor 1 17282 .coefficient) (⟨false, false, none, none, none⟩))

def event17284 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨49392⟩⟩, .operator (⟨17280, 0⟩, ⟨17278, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨14951⟩⟩, ⟨.program ⟨257⟩, ⟨47626⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact17285RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14951⟩⟩, ⟨.program ⟨257⟩, ⟨47626⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact17285RawTermsValid :
    exact17285RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17285 : Event := .resultExact (⟨.program ⟨257⟩, ⟨49392⟩⟩) exact17285RawTerms .large 17283 .exactZero (none)

def event17286 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨2370⟩⟩) (.authority (.operator))

def event17287 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨2370⟩⟩) (.finite 1)

def event17288 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7178⟩⟩) 0 ⟨7177⟩ 17262

def event17289 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7178⟩⟩) (.authority (.operator))

def exact17290RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7178⟩⟩]⟩, (1)⟩]

theorem exact17290RawTermsValid :
    exact17290RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17290 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7178⟩⟩) exact17290RawTerms .large 17289 .exactZero (none)

def event17291 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7285⟩⟩) 0 ⟨7178⟩ 17290

def event17292 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7285⟩⟩) (.identity (.predecessor 0 17291 .coefficient))

def exact17293RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7285⟩⟩]⟩, (1)⟩]

theorem exact17293RawTermsValid :
    exact17293RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17293 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7285⟩⟩) exact17293RawTerms .large 17292 .exactZero (none)

def event17294 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9565⟩⟩) 0 ⟨7285⟩ 17293

def event17295 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9565⟩⟩) (.authority (.operator))

def exact17296RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨9565⟩⟩]⟩, (1)⟩]

theorem exact17296RawTermsValid :
    exact17296RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17296 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9565⟩⟩) exact17296RawTerms (.finite 8192) 17295 .exactZero (none)

def event17297 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9566⟩⟩) 0 ⟨9565⟩ 17296

def event17298 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9566⟩⟩) 1 ⟨2370⟩ 17287

def event17299 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9566⟩⟩) (.scale (.predecessor 0 17297 .coefficient) (.value (.predecessor 1 17298 .coefficient)))

def exact17300RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨9565⟩⟩]⟩, (1)⟩]

theorem exact17300RawTermsValid :
    exact17300RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17300 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9566⟩⟩) exact17300RawTerms (.finite 8192) 17299 .exactZero (none)

def event17301 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7302⟩⟩) 0 ⟨7178⟩ 17290

def event17302 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7302⟩⟩) (.identity (.predecessor 0 17301 .coefficient))

def exact17303RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7302⟩⟩]⟩, (1)⟩]

theorem exact17303RawTermsValid :
    exact17303RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17303 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7302⟩⟩) exact17303RawTerms .large 17302 .exactZero (none)

def event17304 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9567⟩⟩) 0 ⟨7302⟩ 17303

def event17305 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9567⟩⟩) 1 ⟨9566⟩ 17300

def event17306 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9567⟩⟩) (.product (.predecessor 0 17304 .coefficient) (.predecessor 1 17305 .coefficient) (⟨false, false, none, none, none⟩))

def event17307 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨9567⟩⟩, .operator (⟨17303, 0⟩, ⟨17300, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩]⟩, (1)⟩)

def exact17308RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩]⟩, (1)⟩]

theorem exact17308RawTermsValid :
    exact17308RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17308 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9567⟩⟩) exact17308RawTerms .large 17306 .exactZero (none)

def event17309 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49393⟩⟩) 0 ⟨9567⟩ 17308

def event17310 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49393⟩⟩) 1 ⟨49392⟩ 17285

def event17311 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49393⟩⟩) (.sum [.predecessor 0 17309 .coefficient, .predecessor 1 17310 .coefficient])

def exact17312RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨14951⟩⟩, ⟨.program ⟨257⟩, ⟨47626⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact17312RawTermsValid :
    exact17312RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17312 : Event := .resultExact (⟨.program ⟨257⟩, ⟨49393⟩⟩) exact17312RawTerms .large 17311 .exactZero (none)

def event17313 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49566⟩⟩) 0 ⟨49393⟩ 17312

def event17314 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49566⟩⟩) 1 ⟨49563⟩ 17269

def event17315 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49566⟩⟩) (.product (.predecessor 0 17313 .coefficient) (.predecessor 1 17314 .coefficient) (⟨false, false, none, none, none⟩))

def event17316 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨49566⟩⟩, .operator (⟨17312, 1⟩, ⟨17269, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨14951⟩⟩, ⟨.program ⟨257⟩, ⟨47626⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨49563⟩⟩]⟩, (-1)⟩)

def event17317 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨49566⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨14951⟩⟩, ⟨.program ⟨257⟩, ⟨47626⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨49563⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨49563⟩⟩) ⟨49097⟩ 17266)

def event17318 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨49566⟩⟩, .relation 17317 0, ⟨[⟨.program ⟨257⟩, ⟨14951⟩⟩, ⟨.program ⟨257⟩, ⟨47626⟩⟩], [⟨.program ⟨257⟩, ⟨49097⟩⟩]⟩, (-1)⟩)

def event17319 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨49566⟩⟩, .operator (⟨17312, 0⟩, ⟨17269, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49563⟩⟩]⟩, (1)⟩)

def exact17320RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49563⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨14951⟩⟩, ⟨.program ⟨257⟩, ⟨47626⟩⟩], [⟨.program ⟨257⟩, ⟨49097⟩⟩]⟩, (-1)⟩]

theorem exact17320RawTermsValid :
    exact17320RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17320 : Event := .resultExact (⟨.program ⟨257⟩, ⟨49566⟩⟩) exact17320RawTerms .large 17315 .exactZero (none)

def event17321 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48078⟩⟩) 0 ⟨47628⟩ 17258

def event17322 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48078⟩⟩) (.authority (.programFamilyFact))

def exact17323RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨48078⟩⟩], []⟩, (1)⟩]

theorem exact17323RawTermsValid :
    exact17323RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17323 : Event := .resultExact (⟨.program ⟨257⟩, ⟨48078⟩⟩) exact17323RawTerms (.finite 60) 17322 .exactZero (none)

def event17324 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48080⟩⟩) 0 ⟨6908⟩ 17280

def event17325 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48080⟩⟩) 1 ⟨48078⟩ 17323

def event17326 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48080⟩⟩) (.product (.predecessor 0 17324 .coefficient) (.predecessor 1 17325 .coefficient) (⟨false, true, none, none, some 1⟩))

def event17327 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨48080⟩⟩, .operator (⟨17280, 0⟩, ⟨17323, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨48078⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact17328RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨48078⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact17328RawTermsValid :
    exact17328RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17328 : Event := .resultExact (⟨.program ⟨257⟩, ⟨48080⟩⟩) exact17328RawTerms .large 17326 .exactZero (none)

def event17329 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7196⟩⟩) 0 ⟨7177⟩ 17262

def event17330 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7196⟩⟩) (.authority (.operator))

def exact17331RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7196⟩⟩]⟩, (1)⟩]

theorem exact17331RawTermsValid :
    exact17331RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17331 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7196⟩⟩) exact17331RawTerms .large 17330 .exactZero (none)

def event17332 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48081⟩⟩) 0 ⟨7196⟩ 17331

def event17333 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48081⟩⟩) 1 ⟨48080⟩ 17328

def event17334 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48081⟩⟩) (.sum [.predecessor 0 17332 .coefficient, .predecessor 1 17333 .coefficient])

def exact17335RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7196⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨48078⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact17335RawTermsValid :
    exact17335RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17335 : Event := .resultExact (⟨.program ⟨257⟩, ⟨48081⟩⟩) exact17335RawTerms .large 17334 .exactZero (none)

def event17336 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49567⟩⟩) 0 ⟨48081⟩ 17335

def event17337 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49567⟩⟩) 1 ⟨49566⟩ 17320

def event17338 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49567⟩⟩) (.sum [.predecessor 0 17336 .coefficient, .predecessor 1 17337 .coefficient])

def exact17339RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7196⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49563⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨14951⟩⟩, ⟨.program ⟨257⟩, ⟨47626⟩⟩], [⟨.program ⟨257⟩, ⟨49097⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨48078⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact17339RawTermsValid :
    exact17339RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17339 : Event := .resultExact (⟨.program ⟨257⟩, ⟨49567⟩⟩) exact17339RawTerms .large 17338 .exactZero (none)

def event17340 : Event := .preFoldPolynomial 17339 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7196⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49563⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨14951⟩⟩, ⟨.program ⟨257⟩, ⟨47626⟩⟩], [⟨.program ⟨257⟩, ⟨49097⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨48078⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact17341RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7196⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49563⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨14951⟩⟩, ⟨.program ⟨257⟩, ⟨47626⟩⟩], [⟨.program ⟨257⟩, ⟨49097⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨48078⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event17341 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨49567⟩⟩) 17340 exact17341RawTerms .large 17338 .exactZero (none)

def event17342 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨47628⟩⟩) ⟨⟨75⟩, ⟨54⟩, ⟨135⟩⟩ ⟨17176, 17342⟩

def event17343 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨48505⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48502⟩⟩]⟩) (1) 0 2 (.universal 17342 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48502⟩⟩]⟩) (none) 17341)

def event17344 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨48505⟩⟩, .relation 17343 2, ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨14951⟩⟩, ⟨.program ⟨257⟩, ⟨47626⟩⟩], [⟨.program ⟨257⟩, ⟨49097⟩⟩]⟩, (1)⟩)

def event17345 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨48505⟩⟩, .relation 17343 1, ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49563⟩⟩]⟩, (-1)⟩)

def event17346 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨48505⟩⟩, .relation 17343 3, ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨48078⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event17347 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨48505⟩⟩, .relation 17343 0, ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7196⟩⟩]⟩, (1)⟩)

def exact17348RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7196⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49563⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨14951⟩⟩, ⟨.program ⟨257⟩, ⟨47626⟩⟩], [⟨.program ⟨257⟩, ⟨49097⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨48078⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact17348RawTermsValid :
    exact17348RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17348 : Event := .resultExact (⟨.program ⟨257⟩, ⟨48505⟩⟩) exact17348RawTerms .large 17172 (.finite 202072841853861888) (some (17174))

def event17349 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49565⟩⟩) 0 ⟨48505⟩ 17348

def event17350 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49565⟩⟩) 1 ⟨49564⟩ 17147

def event17351 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49565⟩⟩) (.sum [.predecessor 0 17349 .coefficient, .predecessor 1 17350 .coefficient])

def event17352 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨49565⟩⟩, .operator (⟨17348, 2⟩, ⟨17147, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨14951⟩⟩, ⟨.program ⟨257⟩, ⟨47626⟩⟩], [⟨.program ⟨257⟩, ⟨49097⟩⟩]⟩, (-1)⟩)

def event17353 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨49565⟩⟩, .operator (⟨17348, 1⟩, ⟨17147, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49563⟩⟩]⟩, (1)⟩)

def event17354 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49565⟩⟩) (.sum [.result 17348 .summary, .result 17147 .summary])

def exact17355RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7196⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨48078⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact17355RawTermsValid :
    exact17355RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17355 : Event := .resultExact (⟨.program ⟨257⟩, ⟨49565⟩⟩) exact17355RawTerms .large 17351 (.finite 2998346861024241778688) (some (17354))

def event17356 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49813⟩⟩) 0 ⟨49565⟩ 17355

def event17357 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49813⟩⟩) 1 ⟨49811⟩ 17037

def event17358 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49813⟩⟩) (.product (.predecessor 0 17356 .coefficient) (.predecessor 1 17357 .coefficient) (⟨false, false, none, none, none⟩))

def event17359 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49813⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨49811⟩⟩]⟩) [⟨.result 17037 .coefficient, false, none⟩])

def event17360 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49813⟩⟩) (.product (.result 17355 .summary) (.transfer 17359) (⟨false, false, none, none, none⟩))

def event17361 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨49813⟩⟩, .operator (⟨17355, 1⟩, ⟨17037, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨48078⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨49811⟩⟩]⟩, (-1)⟩)

def event17362 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨49813⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨48078⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨49811⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨49811⟩⟩) ⟨49223⟩ 17034)

def event17363 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨49813⟩⟩, .relation 17362 0, ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨48078⟩⟩], [⟨.program ⟨257⟩, ⟨49223⟩⟩]⟩, (-1)⟩)

def event17364 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨49813⟩⟩, .operator (⟨17355, 0⟩, ⟨17037, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨49811⟩⟩]⟩, (1)⟩)

def exact17365RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨49811⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨48078⟩⟩], [⟨.program ⟨257⟩, ⟨49223⟩⟩]⟩, (-1)⟩]

theorem exact17365RawTermsValid :
    exact17365RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17365 : Event := .resultExact (⟨.program ⟨257⟩, ⟨49813⟩⟩) exact17365RawTerms .large 17358 (.finite 32194504275408438756654574469120) (some (17360))

def event17366 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48722⟩⟩) 0 ⟨48079⟩ 68

def event17367 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48722⟩⟩) (.authority (.relationPreimageSource ⟨94⟩))

def exact17368RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨48722⟩⟩]⟩, (1)⟩]

theorem exact17368RawTermsValid :
    exact17368RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17368 : Event := .resultExact (⟨.program ⟨257⟩, ⟨48722⟩⟩) exact17368RawTerms (.finite 5647228698) 17367 .exactZero (none)

def event17369 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48724⟩⟩) 0 ⟨48722⟩ 17368

def event17370 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48724⟩⟩) 1 ⟨2370⟩ 4

def event17371 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48724⟩⟩) (.scale (.predecessor 0 17369 .coefficient) (.value (.predecessor 1 17370 .coefficient)))

def exact17372RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨48722⟩⟩]⟩, (1)⟩]

theorem exact17372RawTermsValid :
    exact17372RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17372 : Event := .resultExact (⟨.program ⟨257⟩, ⟨48724⟩⟩) exact17372RawTerms (.finite 5647228698) 17371 .exactZero (none)

def event17373 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48725⟩⟩) 0 ⟨5443⟩ 17169

def event17374 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48725⟩⟩) 1 ⟨48724⟩ 17372

def event17375 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48725⟩⟩) (.product (.predecessor 0 17373 .coefficient) (.predecessor 1 17374 .coefficient) (⟨false, false, none, none, none⟩))

def event17376 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48725⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨48722⟩⟩]⟩) [⟨.result 17368 .coefficient, false, none⟩])

def event17377 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48725⟩⟩) (.product (.result 17169 .summary) (.transfer 17376) (⟨false, false, none, none, none⟩))

def event17378 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨48725⟩⟩, .operator (⟨17169, 0⟩, ⟨17372, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48722⟩⟩]⟩, (1)⟩)

def event17379 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨48723⟩⟩)

def event17380 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event17381 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event17382 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨140⟩⟩) (.authority (.operator))

def event17383 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨140⟩⟩) (.finite 19)

def event17384 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event17385 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event17386 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event17387 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event17388 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 17387

def event17389 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 17385

def event17390 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 17388 .coefficient) (.value (.predecessor 1 17389 .coefficient)))

def event17391 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event17392 : Event := .predecessor (⟨.program ⟨257⟩, ⟨393⟩⟩) 0 ⟨392⟩ 17391

def event17393 : Event := .predecessor (⟨.program ⟨257⟩, ⟨393⟩⟩) 1 ⟨140⟩ 17383

def event17394 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨393⟩⟩) (.sum [.predecessor 0 17392 .coefficient, .predecessor 1 17393 .coefficient])

def event17395 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨393⟩⟩) (.finite 655359)

def event17396 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5439⟩⟩) 0 ⟨393⟩ 17395

def event17397 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5439⟩⟩) 1 ⟨5426⟩ 17381

def event17398 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5439⟩⟩) (.identity (.predecessor 1 17397 .coefficient))

def event17399 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5439⟩⟩) (.finite 655360)

def event17400 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47626⟩⟩) 0 ⟨5439⟩ 17399

def event17401 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47626⟩⟩) (.authority (.programFamilyFact))

def exact17402RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨47626⟩⟩], []⟩, (1)⟩]

theorem exact17402RawTermsValid :
    exact17402RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17402 : Event := .resultExact (⟨.program ⟨257⟩, ⟨47626⟩⟩) exact17402RawTerms (.finite 60) 17401 .exactZero (none)

def event17403 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14951⟩⟩) 0 ⟨5439⟩ 17399

def event17404 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14951⟩⟩) (.authority (.programFamilyFact))

def exact17405RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14951⟩⟩], []⟩, (1)⟩]

theorem exact17405RawTermsValid :
    exact17405RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17405 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14951⟩⟩) exact17405RawTerms (.finite 60) 17404 .exactZero (none)

def event17406 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47627⟩⟩) 0 ⟨14951⟩ 17405

def event17407 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47627⟩⟩) 1 ⟨47626⟩ 17402

def eventLeaf1072 : Array AnnotatedEvent := #[
  { event := event17152
    frameStart := 0 },
  { event := event17153
    frameStart := 0 },
  { event := event17154
    frameStart := 0 },
  { event := event17155
    frameStart := 0 },
  { event := event17156
    frameStart := 0 },
  { event := event17157
    frameStart := 0 },
  { event := event17158
    frameStart := 0 },
  { event := event17159
    frameStart := 0 },
  { event := event17160
    frameStart := 0 },
  { event := event17161
    frameStart := 0 },
  { event := event17162
    frameStart := 0 },
  { event := event17163
    frameStart := 0 },
  { event := event17164
    frameStart := 0 },
  { event := event17165
    frameStart := 0 },
  { event := event17166
    frameStart := 0 },
  { event := event17167
    frameStart := 0 }
]

def eventLeaf1073 : Array AnnotatedEvent := #[
  { event := event17168
    frameStart := 0 },
  { event := event17169
    frameStart := 0 },
  { event := event17170
    frameStart := 0 },
  { event := event17171
    frameStart := 0 },
  { event := event17172
    frameStart := 0 },
  { event := event17173
    frameStart := 0 },
  { event := event17174
    frameStart := 0 },
  { event := event17175
    frameStart := 0 },
  { event := event17176
    frameStart := 17176 },
  { event := event17177
    frameStart := 17176 },
  { event := event17178
    frameStart := 17176 },
  { event := event17179
    frameStart := 17176 },
  { event := event17180
    frameStart := 17176 },
  { event := event17181
    frameStart := 17176 },
  { event := event17182
    frameStart := 17176 },
  { event := event17183
    frameStart := 17176 }
]

def eventLeaf1074 : Array AnnotatedEvent := #[
  { event := event17184
    frameStart := 17176 },
  { event := event17185
    frameStart := 17176 },
  { event := event17186
    frameStart := 17176 },
  { event := event17187
    frameStart := 17176 },
  { event := event17188
    frameStart := 17176 },
  { event := event17189
    frameStart := 17176 },
  { event := event17190
    frameStart := 17176 },
  { event := event17191
    frameStart := 17176 },
  { event := event17192
    frameStart := 17176 },
  { event := event17193
    frameStart := 17176 },
  { event := event17194
    frameStart := 17176 },
  { event := event17195
    frameStart := 17176 },
  { event := event17196
    frameStart := 17176 },
  { event := event17197
    frameStart := 17176 },
  { event := event17198
    frameStart := 17176 },
  { event := event17199
    frameStart := 17176 }
]

def eventLeaf1075 : Array AnnotatedEvent := #[
  { event := event17200
    frameStart := 17176 },
  { event := event17201
    frameStart := 17176 },
  { event := event17202
    frameStart := 17176 },
  { event := event17203
    frameStart := 17176 },
  { event := event17204
    frameStart := 17176 },
  { event := event17205
    frameStart := 17176 },
  { event := event17206
    frameStart := 17176 },
  { event := event17207
    frameStart := 17176 },
  { event := event17208
    frameStart := 17176 },
  { event := event17209
    frameStart := 17176 },
  { event := event17210
    frameStart := 17176 },
  { event := event17211
    frameStart := 17176 },
  { event := event17212
    frameStart := 17176 },
  { event := event17213
    frameStart := 17176 },
  { event := event17214
    frameStart := 17176 },
  { event := event17215
    frameStart := 17176 }
]

def eventLeaf1076 : Array AnnotatedEvent := #[
  { event := event17216
    frameStart := 17176 },
  { event := event17217
    frameStart := 17176 },
  { event := event17218
    frameStart := 17176 },
  { event := event17219
    frameStart := 17176 },
  { event := event17220
    frameStart := 17176 },
  { event := event17221
    frameStart := 17176 },
  { event := event17222
    frameStart := 17176 },
  { event := event17223
    frameStart := 17176 },
  { event := event17224
    frameStart := 17224 },
  { event := event17225
    frameStart := 17224 },
  { event := event17226
    frameStart := 17224 },
  { event := event17227
    frameStart := 17224 },
  { event := event17228
    frameStart := 17224 },
  { event := event17229
    frameStart := 17224 },
  { event := event17230
    frameStart := 17224 },
  { event := event17231
    frameStart := 17224 }
]

def eventLeaf1077 : Array AnnotatedEvent := #[
  { event := event17232
    frameStart := 17224 },
  { event := event17233
    frameStart := 17224 },
  { event := event17234
    frameStart := 17224 },
  { event := event17235
    frameStart := 17224 },
  { event := event17236
    frameStart := 17224 },
  { event := event17237
    frameStart := 17224 },
  { event := event17238
    frameStart := 17224 },
  { event := event17239
    frameStart := 17224 },
  { event := event17240
    frameStart := 17224 },
  { event := event17241
    frameStart := 17224 },
  { event := event17242
    frameStart := 17224 },
  { event := event17243
    frameStart := 17224 },
  { event := event17244
    frameStart := 17224 },
  { event := event17245
    frameStart := 17224 },
  { event := event17246
    frameStart := 17224 },
  { event := event17247
    frameStart := 17224 }
]

def eventLeaf1078 : Array AnnotatedEvent := #[
  { event := event17248
    frameStart := 17224 },
  { event := event17249
    frameStart := 17224 },
  { event := event17250
    frameStart := 17224 },
  { event := event17251
    frameStart := 17224 },
  { event := event17252
    frameStart := 17224 },
  { event := event17253
    frameStart := 17224 },
  { event := event17254
    frameStart := 17224 },
  { event := event17255
    frameStart := 17224 },
  { event := event17256
    frameStart := 17224 },
  { event := event17257
    frameStart := 17224 },
  { event := event17258
    frameStart := 17224 },
  { event := event17259
    frameStart := 17224 },
  { event := event17260
    frameStart := 17224 },
  { event := event17261
    frameStart := 17224 },
  { event := event17262
    frameStart := 17224 },
  { event := event17263
    frameStart := 17224 }
]

def eventLeaf1079 : Array AnnotatedEvent := #[
  { event := event17264
    frameStart := 17224 },
  { event := event17265
    frameStart := 17224 },
  { event := event17266
    frameStart := 17224 },
  { event := event17267
    frameStart := 17224 },
  { event := event17268
    frameStart := 17224 },
  { event := event17269
    frameStart := 17224 },
  { event := event17270
    frameStart := 17224 },
  { event := event17271
    frameStart := 17224 },
  { event := event17272
    frameStart := 17224 },
  { event := event17273
    frameStart := 17224 },
  { event := event17274
    frameStart := 17224 },
  { event := event17275
    frameStart := 17224 },
  { event := event17276
    frameStart := 17224 },
  { event := event17277
    frameStart := 17224 },
  { event := event17278
    frameStart := 17224 },
  { event := event17279
    frameStart := 17224 }
]

def eventLeaf1080 : Array AnnotatedEvent := #[
  { event := event17280
    frameStart := 17224 },
  { event := event17281
    frameStart := 17224 },
  { event := event17282
    frameStart := 17224 },
  { event := event17283
    frameStart := 17224 },
  { event := event17284
    frameStart := 17224 },
  { event := event17285
    frameStart := 17224 },
  { event := event17286
    frameStart := 17224 },
  { event := event17287
    frameStart := 17224 },
  { event := event17288
    frameStart := 17224 },
  { event := event17289
    frameStart := 17224 },
  { event := event17290
    frameStart := 17224 },
  { event := event17291
    frameStart := 17224 },
  { event := event17292
    frameStart := 17224 },
  { event := event17293
    frameStart := 17224 },
  { event := event17294
    frameStart := 17224 },
  { event := event17295
    frameStart := 17224 }
]

def eventLeaf1081 : Array AnnotatedEvent := #[
  { event := event17296
    frameStart := 17224 },
  { event := event17297
    frameStart := 17224 },
  { event := event17298
    frameStart := 17224 },
  { event := event17299
    frameStart := 17224 },
  { event := event17300
    frameStart := 17224 },
  { event := event17301
    frameStart := 17224 },
  { event := event17302
    frameStart := 17224 },
  { event := event17303
    frameStart := 17224 },
  { event := event17304
    frameStart := 17224 },
  { event := event17305
    frameStart := 17224 },
  { event := event17306
    frameStart := 17224 },
  { event := event17307
    frameStart := 17224 },
  { event := event17308
    frameStart := 17224 },
  { event := event17309
    frameStart := 17224 },
  { event := event17310
    frameStart := 17224 },
  { event := event17311
    frameStart := 17224 }
]

def eventLeaf1082 : Array AnnotatedEvent := #[
  { event := event17312
    frameStart := 17224 },
  { event := event17313
    frameStart := 17224 },
  { event := event17314
    frameStart := 17224 },
  { event := event17315
    frameStart := 17224 },
  { event := event17316
    frameStart := 17224 },
  { event := event17317
    frameStart := 17224 },
  { event := event17318
    frameStart := 17224 },
  { event := event17319
    frameStart := 17224 },
  { event := event17320
    frameStart := 17224 },
  { event := event17321
    frameStart := 17224 },
  { event := event17322
    frameStart := 17224 },
  { event := event17323
    frameStart := 17224 },
  { event := event17324
    frameStart := 17224 },
  { event := event17325
    frameStart := 17224 },
  { event := event17326
    frameStart := 17224 },
  { event := event17327
    frameStart := 17224 }
]

def eventLeaf1083 : Array AnnotatedEvent := #[
  { event := event17328
    frameStart := 17224 },
  { event := event17329
    frameStart := 17224 },
  { event := event17330
    frameStart := 17224 },
  { event := event17331
    frameStart := 17224 },
  { event := event17332
    frameStart := 17224 },
  { event := event17333
    frameStart := 17224 },
  { event := event17334
    frameStart := 17224 },
  { event := event17335
    frameStart := 17224 },
  { event := event17336
    frameStart := 17224 },
  { event := event17337
    frameStart := 17224 },
  { event := event17338
    frameStart := 17224 },
  { event := event17339
    frameStart := 17224 },
  { event := event17340
    frameStart := 17224 },
  { event := event17341
    frameStart := 17224 },
  { event := event17342
    frameStart := 0 },
  { event := event17343
    frameStart := 0 }
]

def eventLeaf1084 : Array AnnotatedEvent := #[
  { event := event17344
    frameStart := 0 },
  { event := event17345
    frameStart := 0 },
  { event := event17346
    frameStart := 0 },
  { event := event17347
    frameStart := 0 },
  { event := event17348
    frameStart := 0 },
  { event := event17349
    frameStart := 0 },
  { event := event17350
    frameStart := 0 },
  { event := event17351
    frameStart := 0 },
  { event := event17352
    frameStart := 0 },
  { event := event17353
    frameStart := 0 },
  { event := event17354
    frameStart := 0 },
  { event := event17355
    frameStart := 0 },
  { event := event17356
    frameStart := 0 },
  { event := event17357
    frameStart := 0 },
  { event := event17358
    frameStart := 0 },
  { event := event17359
    frameStart := 0 }
]

def eventLeaf1085 : Array AnnotatedEvent := #[
  { event := event17360
    frameStart := 0 },
  { event := event17361
    frameStart := 0 },
  { event := event17362
    frameStart := 0 },
  { event := event17363
    frameStart := 0 },
  { event := event17364
    frameStart := 0 },
  { event := event17365
    frameStart := 0 },
  { event := event17366
    frameStart := 0 },
  { event := event17367
    frameStart := 0 },
  { event := event17368
    frameStart := 0 },
  { event := event17369
    frameStart := 0 },
  { event := event17370
    frameStart := 0 },
  { event := event17371
    frameStart := 0 },
  { event := event17372
    frameStart := 0 },
  { event := event17373
    frameStart := 0 },
  { event := event17374
    frameStart := 0 },
  { event := event17375
    frameStart := 0 }
]

def eventLeaf1086 : Array AnnotatedEvent := #[
  { event := event17376
    frameStart := 0 },
  { event := event17377
    frameStart := 0 },
  { event := event17378
    frameStart := 0 },
  { event := event17379
    frameStart := 17379 },
  { event := event17380
    frameStart := 17379 },
  { event := event17381
    frameStart := 17379 },
  { event := event17382
    frameStart := 17379 },
  { event := event17383
    frameStart := 17379 },
  { event := event17384
    frameStart := 17379 },
  { event := event17385
    frameStart := 17379 },
  { event := event17386
    frameStart := 17379 },
  { event := event17387
    frameStart := 17379 },
  { event := event17388
    frameStart := 17379 },
  { event := event17389
    frameStart := 17379 },
  { event := event17390
    frameStart := 17379 },
  { event := event17391
    frameStart := 17379 }
]

def eventLeaf1087 : Array AnnotatedEvent := #[
  { event := event17392
    frameStart := 17379 },
  { event := event17393
    frameStart := 17379 },
  { event := event17394
    frameStart := 17379 },
  { event := event17395
    frameStart := 17379 },
  { event := event17396
    frameStart := 17379 },
  { event := event17397
    frameStart := 17379 },
  { event := event17398
    frameStart := 17379 },
  { event := event17399
    frameStart := 17379 },
  { event := event17400
    frameStart := 17379 },
  { event := event17401
    frameStart := 17379 },
  { event := event17402
    frameStart := 17379 },
  { event := event17403
    frameStart := 17379 },
  { event := event17404
    frameStart := 17379 },
  { event := event17405
    frameStart := 17379 },
  { event := event17406
    frameStart := 17379 },
  { event := event17407
    frameStart := 17379 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events067
