import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events751

open Mxx.Certificate.OperationalNoise
open CertificateABI

def exact192256RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact192256RawTermsValid :
    exact192256RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event192256 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35⟩⟩) exact192256RawTerms .large 192255 .exactZero (none)

def event192257 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16653⟩⟩) 0 ⟨35⟩ 192256

def event192258 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16653⟩⟩) 1 ⟨16652⟩ 192254

def event192259 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16653⟩⟩) (.product (.predecessor 0 192257 .coefficient) (.predecessor 1 192258 .coefficient) (⟨false, false, none, none, none⟩))

def event192260 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16653⟩⟩, .operator (⟨192256, 0⟩, ⟨192254, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16652⟩⟩]⟩, (1)⟩)

def exact192261RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16652⟩⟩]⟩, (1)⟩]

theorem exact192261RawTermsValid :
    exact192261RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event192261 : Event := .resultExact (⟨.program ⟨257⟩, ⟨16653⟩⟩) exact192261RawTerms .large 192259 .exactZero (none)

def event192262 : Event := .preFoldPolynomial 192261 [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16652⟩⟩]⟩, (1)⟩] .exactZero none

def exact192263RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16652⟩⟩]⟩, (1)⟩]

def event192263 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨16653⟩⟩) 192262 exact192263RawTerms .large 192259 .exactZero (none)

def event192264 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨17844⟩⟩)

def event192265 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event192266 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event192267 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6170⟩⟩) (.authority (.operator))

def event192268 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨6170⟩⟩) (.finite 8)

def event192269 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event192270 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event192271 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event192272 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event192273 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 192272

def event192274 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 192270

def event192275 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 192273 .coefficient) (.value (.predecessor 1 192274 .coefficient)))

def event192276 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event192277 : Event := .predecessor (⟨.program ⟨257⟩, ⟨6172⟩⟩) 0 ⟨392⟩ 192276

def event192278 : Event := .predecessor (⟨.program ⟨257⟩, ⟨6172⟩⟩) 1 ⟨6170⟩ 192268

def event192279 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6172⟩⟩) (.sum [.predecessor 0 192277 .coefficient, .predecessor 1 192278 .coefficient])

def event192280 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨6172⟩⟩) (.finite 655348)

def event192281 : Event := .predecessor (⟨.program ⟨257⟩, ⟨6182⟩⟩) 0 ⟨6172⟩ 192280

def event192282 : Event := .predecessor (⟨.program ⟨257⟩, ⟨6182⟩⟩) 1 ⟨5426⟩ 192266

def event192283 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6182⟩⟩) (.identity (.predecessor 1 192282 .coefficient))

def event192284 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨6182⟩⟩) (.finite 655360)

def event192285 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15546⟩⟩) 0 ⟨6182⟩ 192284

def event192286 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15546⟩⟩) (.authority (.programFamilyFact))

def exact192287RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15546⟩⟩], []⟩, (1)⟩]

theorem exact192287RawTermsValid :
    exact192287RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event192287 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15546⟩⟩) exact192287RawTerms (.finite 2) 192286 .exactZero (none)

def event192288 : Event := .predecessor (⟨.program ⟨257⟩, ⟨12426⟩⟩) 0 ⟨6182⟩ 192284

def event192289 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨12426⟩⟩) (.authority (.programFamilyFact))

def exact192290RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨12426⟩⟩], []⟩, (1)⟩]

theorem exact192290RawTermsValid :
    exact192290RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event192290 : Event := .resultExact (⟨.program ⟨257⟩, ⟨12426⟩⟩) exact192290RawTerms (.finite 2) 192289 .exactZero (none)

def event192291 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15547⟩⟩) 0 ⟨12426⟩ 192290

def event192292 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15547⟩⟩) 1 ⟨15546⟩ 192287

def event192293 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15547⟩⟩) (.product (.predecessor 0 192291 .coefficient) (.predecessor 1 192292 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event192294 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨15547⟩⟩, .operator (⟨192290, 0⟩, ⟨192287, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨12426⟩⟩, ⟨.program ⟨257⟩, ⟨15546⟩⟩], []⟩, (1)⟩)

def exact192295RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨12426⟩⟩, ⟨.program ⟨257⟩, ⟨15546⟩⟩], []⟩, (1)⟩]

theorem exact192295RawTermsValid :
    exact192295RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event192295 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15547⟩⟩) exact192295RawTerms (.finite 4) 192293 .exactZero (none)

def event192296 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15548⟩⟩) 0 ⟨15547⟩ 192295

def event192297 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15548⟩⟩) (.identity (.predecessor 0 192296 .coefficient))

def event192298 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨15548⟩⟩) (.finite 4)

def event192299 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15812⟩⟩) 0 ⟨15548⟩ 192298

def event192300 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15812⟩⟩) (.authority (.programFamilyFact))

def exact192301RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15812⟩⟩], []⟩, (1)⟩]

theorem exact192301RawTermsValid :
    exact192301RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event192301 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15812⟩⟩) exact192301RawTerms (.finite 2) 192300 .exactZero (none)

def event192302 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15813⟩⟩) 0 ⟨15812⟩ 192301

def event192303 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15813⟩⟩) (.identity (.predecessor 0 192302 .coefficient))

def event192304 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨15813⟩⟩) (.finite 2)

def event192305 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17026⟩⟩) 0 ⟨15813⟩ 192304

def event192306 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17026⟩⟩) (.authority (.programFamilyFact))

def event192307 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨17026⟩⟩) (.finite 3720)

def event192308 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨7177⟩⟩) .missing

def event192309 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17027⟩⟩) 0 ⟨7177⟩ 192308

def event192310 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17027⟩⟩) 1 ⟨17026⟩ 192307

def event192311 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17027⟩⟩) (.authority (.operator))

def exact192312RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨17027⟩⟩]⟩, (1)⟩]

theorem exact192312RawTermsValid :
    exact192312RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event192312 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17027⟩⟩) exact192312RawTerms .large 192311 .exactZero (none)

def event192313 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17838⟩⟩) 0 ⟨17027⟩ 192312

def event192314 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17838⟩⟩) (.authority (.operator))

def exact192315RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨17838⟩⟩]⟩, (1)⟩]

theorem exact192315RawTermsValid :
    exact192315RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event192315 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17838⟩⟩) exact192315RawTerms (.finite 8192) 192314 .exactZero (none)

def event192316 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event192317 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event192318 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17218⟩⟩) 0 ⟨15813⟩ 192304

def event192319 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17218⟩⟩) 1 ⟨136⟩ 192317

def event192320 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17218⟩⟩) (.sum [.predecessor 0 192318 .coefficient, .predecessor 1 192319 .coefficient])

def event192321 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨17218⟩⟩) (.finite 2)

def event192322 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17219⟩⟩) 0 ⟨17218⟩ 192321

def event192323 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17219⟩⟩) (.identity (.predecessor 0 192322 .coefficient))

def exact192324RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15812⟩⟩], []⟩, (1)⟩]

theorem exact192324RawTermsValid :
    exact192324RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event192324 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17219⟩⟩) exact192324RawTerms (.finite 2) 192323 .exactZero (none)

def event192325 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact192326RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact192326RawTermsValid :
    exact192326RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event192326 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact192326RawTerms .large 192325 .exactZero (none)

def event192327 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17220⟩⟩) 0 ⟨6908⟩ 192326

def event192328 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17220⟩⟩) 1 ⟨17219⟩ 192324

def event192329 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17220⟩⟩) (.product (.predecessor 0 192327 .coefficient) (.predecessor 1 192328 .coefficient) (⟨false, false, none, none, none⟩))

def event192330 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17220⟩⟩, .operator (⟨192326, 0⟩, ⟨192324, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨15812⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact192331RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15812⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact192331RawTermsValid :
    exact192331RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event192331 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17220⟩⟩) exact192331RawTerms .large 192329 .exactZero (none)

def event192332 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7179⟩⟩) 0 ⟨7177⟩ 192308

def event192333 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7179⟩⟩) (.authority (.operator))

def exact192334RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩]⟩, (1)⟩]

theorem exact192334RawTermsValid :
    exact192334RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event192334 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7179⟩⟩) exact192334RawTerms .large 192333 .exactZero (none)

def event192335 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17221⟩⟩) 0 ⟨7179⟩ 192334

def event192336 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17221⟩⟩) 1 ⟨17220⟩ 192331

def event192337 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17221⟩⟩) (.sum [.predecessor 0 192335 .coefficient, .predecessor 1 192336 .coefficient])

def exact192338RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15812⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact192338RawTermsValid :
    exact192338RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event192338 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17221⟩⟩) exact192338RawTerms .large 192337 .exactZero (none)

def event192339 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17839⟩⟩) 0 ⟨17221⟩ 192338

def event192340 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17839⟩⟩) 1 ⟨17838⟩ 192315

def event192341 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17839⟩⟩) (.product (.predecessor 0 192339 .coefficient) (.predecessor 1 192340 .coefficient) (⟨false, false, none, none, none⟩))

def event192342 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17839⟩⟩, .operator (⟨192338, 0⟩, ⟨192315, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17838⟩⟩]⟩, (1)⟩)

def event192343 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17839⟩⟩, .operator (⟨192338, 1⟩, ⟨192315, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨15812⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17838⟩⟩]⟩, (-1)⟩)

def event192344 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨17839⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨15812⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17838⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨17838⟩⟩) ⟨17027⟩ 192312)

def event192345 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17839⟩⟩, .relation 192344 0, ⟨[⟨.program ⟨257⟩, ⟨15812⟩⟩], [⟨.program ⟨257⟩, ⟨17027⟩⟩]⟩, (-1)⟩)

def exact192346RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17838⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15812⟩⟩], [⟨.program ⟨257⟩, ⟨17027⟩⟩]⟩, (-1)⟩]

theorem exact192346RawTermsValid :
    exact192346RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event192346 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17839⟩⟩) exact192346RawTerms .large 192341 .exactZero (none)

def event192347 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16078⟩⟩) 0 ⟨15813⟩ 192304

def event192348 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16078⟩⟩) (.authority (.programFamilyFact))

def exact192349RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨16078⟩⟩], []⟩, (1)⟩]

theorem exact192349RawTermsValid :
    exact192349RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event192349 : Event := .resultExact (⟨.program ⟨257⟩, ⟨16078⟩⟩) exact192349RawTerms (.finite 2) 192348 .exactZero (none)

def event192350 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16081⟩⟩) 0 ⟨6908⟩ 192326

def event192351 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16081⟩⟩) 1 ⟨16078⟩ 192349

def event192352 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16081⟩⟩) (.product (.predecessor 0 192350 .coefficient) (.predecessor 1 192351 .coefficient) (⟨false, true, none, none, some 1⟩))

def event192353 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16081⟩⟩, .operator (⟨192326, 0⟩, ⟨192349, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨16078⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact192354RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨16078⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact192354RawTermsValid :
    exact192354RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event192354 : Event := .resultExact (⟨.program ⟨257⟩, ⟨16081⟩⟩) exact192354RawTerms .large 192352 .exactZero (none)

def event192355 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7197⟩⟩) 0 ⟨7177⟩ 192308

def event192356 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7197⟩⟩) (.authority (.operator))

def exact192357RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7197⟩⟩]⟩, (1)⟩]

theorem exact192357RawTermsValid :
    exact192357RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event192357 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7197⟩⟩) exact192357RawTerms .large 192356 .exactZero (none)

def event192358 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16082⟩⟩) 0 ⟨7197⟩ 192357

def event192359 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16082⟩⟩) 1 ⟨16081⟩ 192354

def event192360 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16082⟩⟩) (.sum [.predecessor 0 192358 .coefficient, .predecessor 1 192359 .coefficient])

def exact192361RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7197⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨16078⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact192361RawTermsValid :
    exact192361RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event192361 : Event := .resultExact (⟨.program ⟨257⟩, ⟨16082⟩⟩) exact192361RawTerms .large 192360 .exactZero (none)

def event192362 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17844⟩⟩) 0 ⟨16082⟩ 192361

def event192363 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17844⟩⟩) 1 ⟨17839⟩ 192346

def event192364 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17844⟩⟩) (.sum [.predecessor 0 192362 .coefficient, .predecessor 1 192363 .coefficient])

def exact192365RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17838⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7197⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15812⟩⟩], [⟨.program ⟨257⟩, ⟨17027⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨16078⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact192365RawTermsValid :
    exact192365RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event192365 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17844⟩⟩) exact192365RawTerms .large 192364 .exactZero (none)

def event192366 : Event := .preFoldPolynomial 192365 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17838⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7197⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15812⟩⟩], [⟨.program ⟨257⟩, ⟨17027⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨16078⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact192367RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17838⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7197⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15812⟩⟩], [⟨.program ⟨257⟩, ⟨17027⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨16078⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event192367 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨17844⟩⟩) 192366 exact192367RawTerms .large 192364 .exactZero (none)

def event192368 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨15813⟩⟩) ⟨⟨76⟩, ⟨56⟩, ⟨135⟩⟩ ⟨192210, 192368⟩

def event192369 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨16655⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16652⟩⟩]⟩) (1) 0 2 (.universal 192368 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16652⟩⟩]⟩) (none) 192367)

def event192370 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16655⟩⟩, .relation 192369 1, ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩]⟩, (1)⟩)

def event192371 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16655⟩⟩, .relation 192369 0, ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17838⟩⟩]⟩, (-1)⟩)

def event192372 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16655⟩⟩, .relation 192369 2, ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨15812⟩⟩], [⟨.program ⟨257⟩, ⟨17027⟩⟩]⟩, (1)⟩)

def event192373 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16655⟩⟩, .relation 192369 3, ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨16078⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact192374RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17838⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨15812⟩⟩], [⟨.program ⟨257⟩, ⟨17027⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨16078⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact192374RawTermsValid :
    exact192374RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event192374 : Event := .resultExact (⟨.program ⟨257⟩, ⟨16655⟩⟩) exact192374RawTerms .large 192206 (.finite 202072841853861888) (some (192208))

def event192375 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17841⟩⟩) 0 ⟨16655⟩ 192374

def event192376 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17841⟩⟩) 1 ⟨17840⟩ 192196

def event192377 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17841⟩⟩) (.sum [.predecessor 0 192375 .coefficient, .predecessor 1 192376 .coefficient])

def event192378 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17841⟩⟩, .operator (⟨192374, 0⟩, ⟨192196, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17838⟩⟩]⟩, (1)⟩)

def event192379 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17841⟩⟩, .operator (⟨192374, 2⟩, ⟨192196, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨15812⟩⟩], [⟨.program ⟨257⟩, ⟨17027⟩⟩]⟩, (-1)⟩)

def event192380 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17841⟩⟩) (.sum [.result 192374 .summary, .result 192196 .summary])

def exact192381RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨16078⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact192381RawTermsValid :
    exact192381RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event192381 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17841⟩⟩) exact192381RawTerms .large 192377 (.finite 32188807212483706889510625476608) (some (192380))

def event192382 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17842⟩⟩) 0 ⟨17841⟩ 192381

def event192383 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17842⟩⟩) 1 ⟨7172⟩ 15882

def event192384 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17842⟩⟩) (.product (.predecessor 0 192382 .coefficient) (.predecessor 1 192383 .coefficient) (⟨false, false, none, none, none⟩))

def event192385 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17842⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩) [⟨.result 15878 .coefficient, false, none⟩])

def event192386 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17842⟩⟩) (.product (.result 192381 .summary) (.transfer 192385) (⟨false, false, none, none, none⟩))

def event192387 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17842⟩⟩, .operator (⟨192381, 0⟩, ⟨15882, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩)

def event192388 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17842⟩⟩, .operator (⟨192381, 1⟩, ⟨15882, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨16078⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (-1)⟩)

def event192389 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨17842⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨16078⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨7171⟩⟩) ⟨7051⟩ 15875)

def event192390 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17842⟩⟩, .relation 192389 0, ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16078⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact192391RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16078⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact192391RawTermsValid :
    exact192391RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event192391 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17842⟩⟩) exact192391RawTerms .large 192384 (.finite 345624685687166110058245054666339432529920) (some (192386))

def event192392 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7094⟩⟩) 0 ⟨6727⟩ 723

def event192393 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7094⟩⟩) 1 ⟨7004⟩ 178278

def event192394 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7094⟩⟩) (.tensor (.predecessor 0 192392 .coefficient) (.predecessor 1 192393 .coefficient) true false)

def event192395 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨7094⟩⟩, .operator (⟨723, 0⟩, ⟨178278, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6727⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact192396RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6727⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact192396RawTermsValid :
    exact192396RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event192396 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7094⟩⟩) exact192396RawTerms .large 192394 .exactZero (none)

def event192397 : Event := .predecessor (⟨.program ⟨257⟩, ⟨8940⟩⟩) 0 ⟨6184⟩ 178148

def event192398 : Event := .predecessor (⟨.program ⟨257⟩, ⟨8940⟩⟩) 1 ⟨7292⟩ 15896

def event192399 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨8940⟩⟩) (.product (.predecessor 0 192397 .coefficient) (.predecessor 1 192398 .coefficient) (⟨false, false, none, none, none⟩))

def event192400 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨8940⟩⟩, .operator (⟨178148, 0⟩, ⟨15896, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7292⟩⟩]⟩, (1)⟩)

def exact192401RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7292⟩⟩]⟩, (1)⟩]

theorem exact192401RawTermsValid :
    exact192401RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event192401 : Event := .resultExact (⟨.program ⟨257⟩, ⟨8940⟩⟩) exact192401RawTerms .large 192399 .exactZero (none)

def event192402 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9425⟩⟩) 0 ⟨8940⟩ 192401

def event192403 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9425⟩⟩) 1 ⟨7094⟩ 192396

def event192404 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9425⟩⟩) (.sum [.predecessor 0 192402 .coefficient, .predecessor 1 192403 .coefficient])

def exact192405RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7292⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6727⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact192405RawTermsValid :
    exact192405RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event192405 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9425⟩⟩) exact192405RawTerms .large 192404 .exactZero (none)

def event192406 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9426⟩⟩) 0 ⟨9425⟩ 192405

def event192407 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9426⟩⟩) 1 ⟨118⟩ 31516

def event192408 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9426⟩⟩) (.sum [.predecessor 0 192406 .coefficient, .predecessor 1 192407 .coefficient])

def event192409 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9426⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨118⟩⟩]⟩) [⟨.result 31516 .coefficient, false, none⟩])

def event192410 : Event := .survivorFold (1) 192409

def exact192411RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7292⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6727⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact192411RawTermsValid :
    exact192411RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event192411 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9426⟩⟩) exact192411RawTerms .large 192408 (.finite 26) (some (192409))

def event192412 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9486⟩⟩) 0 ⟨9426⟩ 192411

def event192413 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9486⟩⟩) 1 ⟨9426⟩ 192411

def event192414 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9486⟩⟩) (.sum [.predecessor 0 192412 .coefficient, .predecessor 1 192413 .coefficient])

def event192415 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨9486⟩⟩, .operator (⟨192411, 1⟩, ⟨192411, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6727⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def event192416 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨9486⟩⟩, .operator (⟨192411, 0⟩, ⟨192411, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7292⟩⟩]⟩, (-1)⟩)

def event192417 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9486⟩⟩) (.sum [.result 192411 .summary, .result 192411 .summary])

def exact192418RawTerms : List Term := []

theorem exact192418RawTermsValid :
    exact192418RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event192418 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9486⟩⟩) exact192418RawTerms .large 192414 (.finite 52) (some (192417))

def event192419 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17843⟩⟩) 0 ⟨9486⟩ 192418

def event192420 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17843⟩⟩) 1 ⟨17842⟩ 192391

def event192421 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17843⟩⟩) (.sum [.predecessor 0 192419 .coefficient, .predecessor 1 192420 .coefficient])

def event192422 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17843⟩⟩) (.sum [.result 192418 .summary, .result 192391 .summary])

def exact192423RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16078⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact192423RawTermsValid :
    exact192423RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event192423 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17843⟩⟩) exact192423RawTerms .large 192421 (.finite 345624685687166110058245054666339432529972) (some (192422))

def event192424 : Event := .predecessor (⟨.program ⟨257⟩, ⟨20743⟩⟩) 0 ⟨17843⟩ 192423

def event192425 : Event := .predecessor (⟨.program ⟨257⟩, ⟨20743⟩⟩) 1 ⟨20742⟩ 192179

def event192426 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨20743⟩⟩) (.sum [.predecessor 0 192424 .coefficient, .predecessor 1 192425 .coefficient])

def event192427 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨20743⟩⟩) (.sum [.result 192423 .summary, .result 192179 .summary])

def exact192428RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18918⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16078⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact192428RawTermsValid :
    exact192428RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event192428 : Event := .resultExact (⟨.program ⟨257⟩, ⟨20743⟩⟩) exact192428RawTerms .large 192426 (.finite 691250426059631610003352154589745737891892) (some (192427))

def event192429 : Event := .predecessor (⟨.program ⟨257⟩, ⟨23963⟩⟩) 0 ⟨20743⟩ 192428

def event192430 : Event := .predecessor (⟨.program ⟨257⟩, ⟨23963⟩⟩) 1 ⟨23962⟩ 191967

def event192431 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨23963⟩⟩) (.sum [.predecessor 0 192429 .coefficient, .predecessor 1 192430 .coefficient])

def event192432 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨23963⟩⟩) (.sum [.result 192428 .summary, .result 191967 .summary])

def exact192433RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22138⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18918⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16078⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact192433RawTermsValid :
    exact192433RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event192433 : Event := .resultExact (⟨.program ⟨257⟩, ⟨23963⟩⟩) exact192433RawTerms .large 192431 (.finite 1036877221117396499835321299770218916085812) (some (192432))

def event192434 : Event := .predecessor (⟨.program ⟨257⟩, ⟨33983⟩⟩) 0 ⟨23963⟩ 192433

def event192435 : Event := .predecessor (⟨.program ⟨257⟩, ⟨33983⟩⟩) 1 ⟨33982⟩ 191755

def event192436 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨33983⟩⟩) (.sum [.predecessor 0 192434 .coefficient, .predecessor 1 192435 .coefficient])

def event192437 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨33983⟩⟩) (.sum [.result 192433 .summary, .result 191755 .summary])

def exact192438RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32158⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22138⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18918⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16078⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact192438RawTermsValid :
    exact192438RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event192438 : Event := .resultExact (⟨.program ⟨257⟩, ⟨33983⟩⟩) exact192438RawTerms .large 192436 (.finite 1382506125545760169441014535464825839943732) (some (192437))

def event192439 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53043⟩⟩) 0 ⟨33983⟩ 192438

def event192440 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53043⟩⟩) 1 ⟨53042⟩ 191543

def event192441 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53043⟩⟩) (.sum [.predecessor 0 192439 .coefficient, .predecessor 1 192440 .coefficient])

def event192442 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53043⟩⟩) (.sum [.result 192438 .summary, .result 191543 .summary])

def exact192443RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7205⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51222⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32158⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22138⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18918⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16078⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact192443RawTermsValid :
    exact192443RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event192443 : Event := .resultExact (⟨.program ⟨257⟩, ⟨53043⟩⟩) exact192443RawTerms .large 192441 (.finite 1728139248715321398594155952187700255129652) (some (192442))

def event192444 : Event := .predecessor (⟨.program ⟨257⟩, ⟨56023⟩⟩) 0 ⟨53043⟩ 192443

def event192445 : Event := .predecessor (⟨.program ⟨257⟩, ⟨56023⟩⟩) 1 ⟨56022⟩ 191331

def event192446 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨56023⟩⟩) (.sum [.predecessor 0 192444 .coefficient, .predecessor 1 192445 .coefficient])

def event192447 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨56023⟩⟩) (.sum [.result 192443 .summary, .result 191331 .summary])

def exact192448RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7205⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7207⟩⟩, ⟨.program ⟨257⟩, ⟨7125⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54202⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51222⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32158⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22138⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18918⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16078⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact192448RawTermsValid :
    exact192448RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event192448 : Event := .resultExact (⟨.program ⟨257⟩, ⟨56023⟩⟩) exact192448RawTerms .large 192446 (.finite 2073774481255481407521021459424708415979572) (some (192447))

def event192449 : Event := .predecessor (⟨.program ⟨257⟩, ⟨59003⟩⟩) 0 ⟨56023⟩ 192448

def event192450 : Event := .predecessor (⟨.program ⟨257⟩, ⟨59003⟩⟩) 1 ⟨59002⟩ 191119

def event192451 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨59003⟩⟩) (.sum [.predecessor 0 192449 .coefficient, .predecessor 1 192450 .coefficient])

def event192452 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨59003⟩⟩) (.sum [.result 192448 .summary, .result 191119 .summary])

def exact192453RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7205⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7207⟩⟩, ⟨.program ⟨257⟩, ⟨7125⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7209⟩⟩, ⟨.program ⟨257⟩, ⟨7107⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57182⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54202⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51222⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32158⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22138⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18918⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16078⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact192453RawTermsValid :
    exact192453RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event192453 : Event := .resultExact (⟨.program ⟨257⟩, ⟨59003⟩⟩) exact192453RawTerms .large 192451 (.finite 2419413932536838975995335147689984068157492) (some (192452))

def event192454 : Event := .predecessor (⟨.program ⟨257⟩, ⟨61983⟩⟩) 0 ⟨59003⟩ 192453

def event192455 : Event := .predecessor (⟨.program ⟨257⟩, ⟨61983⟩⟩) 1 ⟨61982⟩ 190907

def event192456 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨61983⟩⟩) (.sum [.predecessor 0 192454 .coefficient, .predecessor 1 192455 .coefficient])

def event192457 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨61983⟩⟩) (.sum [.result 192453 .summary, .result 190907 .summary])

def exact192458RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7205⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7207⟩⟩, ⟨.program ⟨257⟩, ⟨7125⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7209⟩⟩, ⟨.program ⟨257⟩, ⟨7107⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7211⟩⟩, ⟨.program ⟨257⟩, ⟨7103⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60162⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57182⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54202⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51222⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32158⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22138⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18918⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16078⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact192458RawTermsValid :
    exact192458RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event192458 : Event := .resultExact (⟨.program ⟨257⟩, ⟨61983⟩⟩) exact192458RawTerms .large 192456 (.finite 2765055493188795324243372926469393465999412) (some (192457))

def event192459 : Event := .predecessor (⟨.program ⟨257⟩, ⟨64963⟩⟩) 0 ⟨61983⟩ 192458

def event192460 : Event := .predecessor (⟨.program ⟨257⟩, ⟨64963⟩⟩) 1 ⟨64962⟩ 190695

def event192461 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨64963⟩⟩) (.sum [.predecessor 0 192459 .coefficient, .predecessor 1 192460 .coefficient])

def event192462 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨64963⟩⟩) (.sum [.result 192458 .summary, .result 190695 .summary])

def exact192463RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7205⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7207⟩⟩, ⟨.program ⟨257⟩, ⟨7125⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7209⟩⟩, ⟨.program ⟨257⟩, ⟨7107⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7211⟩⟩, ⟨.program ⟨257⟩, ⟨7103⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7213⟩⟩, ⟨.program ⟨257⟩, ⟨7099⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63142⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60162⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57182⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54202⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51222⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32158⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22138⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18918⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16078⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact192463RawTermsValid :
    exact192463RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event192463 : Event := .resultExact (⟨.program ⟨257⟩, ⟨64963⟩⟩) exact192463RawTerms .large 192461 (.finite 3110701272581949232038858886277070355169332) (some (192462))

def event192464 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70404⟩⟩) 0 ⟨64963⟩ 192463

def event192465 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70404⟩⟩) 1 ⟨70403⟩ 190483

def event192466 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70404⟩⟩) (.sum [.predecessor 0 192464 .coefficient, .predecessor 1 192465 .coefficient])

def event192467 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70404⟩⟩) (.sum [.result 192463 .summary, .result 190483 .summary])

def exact192468RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7205⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7207⟩⟩, ⟨.program ⟨257⟩, ⟨7125⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7209⟩⟩, ⟨.program ⟨257⟩, ⟨7107⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7211⟩⟩, ⟨.program ⟨257⟩, ⟨7103⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7213⟩⟩, ⟨.program ⟨257⟩, ⟨7099⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7215⟩⟩, ⟨.program ⟨257⟩, ⟨7173⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63142⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60162⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57182⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54202⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51222⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32158⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22138⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18918⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16078⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66798⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact192468RawTermsValid :
    exact192468RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event192468 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70404⟩⟩) exact192468RawTerms .large 192466 (.finite 3456353380086899479155517117627148481331252) (some (192467))

def event192469 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70405⟩⟩) 0 ⟨70404⟩ 192468

def event192470 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70405⟩⟩) 1 ⟨28362⟩ 190271

def event192471 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70405⟩⟩) (.sum [.predecessor 0 192469 .coefficient, .predecessor 1 192470 .coefficient])

def event192472 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70405⟩⟩) (.sum [.result 192468 .summary, .result 190271 .summary])

def exact192473RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7205⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7207⟩⟩, ⟨.program ⟨257⟩, ⟨7125⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7209⟩⟩, ⟨.program ⟨257⟩, ⟨7107⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7211⟩⟩, ⟨.program ⟨257⟩, ⟨7103⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7213⟩⟩, ⟨.program ⟨257⟩, ⟨7099⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7215⟩⟩, ⟨.program ⟨257⟩, ⟨7173⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7217⟩⟩, ⟨.program ⟨257⟩, ⟨7169⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63142⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60162⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57182⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54202⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51222⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32158⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22138⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18918⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26661⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16078⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66798⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact192473RawTermsValid :
    exact192473RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event192473 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70405⟩⟩) exact192473RawTerms .large 192471 (.finite 3802007596962448506045899439491360353157172) (some (192472))

def event192474 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70406⟩⟩) 0 ⟨70405⟩ 192473

def event192475 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70406⟩⟩) 1 ⟨31042⟩ 190059

def event192476 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70406⟩⟩) (.sum [.predecessor 0 192474 .coefficient, .predecessor 1 192475 .coefficient])

def event192477 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70406⟩⟩) (.sum [.result 192473 .summary, .result 190059 .summary])

def exact192478RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7205⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7207⟩⟩, ⟨.program ⟨257⟩, ⟨7125⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7209⟩⟩, ⟨.program ⟨257⟩, ⟨7107⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7211⟩⟩, ⟨.program ⟨257⟩, ⟨7103⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7213⟩⟩, ⟨.program ⟨257⟩, ⟨7099⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7215⟩⟩, ⟨.program ⟨257⟩, ⟨7173⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7217⟩⟩, ⟨.program ⟨257⟩, ⟨7169⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7219⟩⟩, ⟨.program ⟨257⟩, ⟨7167⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63142⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60162⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57182⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54202⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51222⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32158⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22138⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18918⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29341⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26661⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16078⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66798⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact192478RawTermsValid :
    exact192478RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event192478 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70406⟩⟩) exact192478RawTerms .large 192476 (.finite 4147668141949793872257454032897973461975092) (some (192477))

def event192479 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70407⟩⟩) 0 ⟨70406⟩ 192478

def event192480 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70407⟩⟩) 1 ⟨36702⟩ 189847

def event192481 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70407⟩⟩) (.sum [.predecessor 0 192479 .coefficient, .predecessor 1 192480 .coefficient])

def event192482 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70407⟩⟩) (.sum [.result 192478 .summary, .result 189847 .summary])

def exact192483RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7205⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7207⟩⟩, ⟨.program ⟨257⟩, ⟨7125⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7209⟩⟩, ⟨.program ⟨257⟩, ⟨7107⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7211⟩⟩, ⟨.program ⟨257⟩, ⟨7103⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7213⟩⟩, ⟨.program ⟨257⟩, ⟨7099⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7215⟩⟩, ⟨.program ⟨257⟩, ⟨7173⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7217⟩⟩, ⟨.program ⟨257⟩, ⟨7169⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7219⟩⟩, ⟨.program ⟨257⟩, ⟨7167⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7221⟩⟩, ⟨.program ⟨257⟩, ⟨7163⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63142⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60162⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57182⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54202⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51222⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32158⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22138⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34998⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18918⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29341⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26661⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16078⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66798⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact192483RawTermsValid :
    exact192483RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event192483 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70407⟩⟩) exact192483RawTerms .large 192481 (.finite 4493332905678336798016456807332854062121012) (some (192482))

def event192484 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70408⟩⟩) 0 ⟨70407⟩ 192483

def event192485 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70408⟩⟩) 1 ⟨39382⟩ 189635

def event192486 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70408⟩⟩) (.sum [.predecessor 0 192484 .coefficient, .predecessor 1 192485 .coefficient])

def event192487 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70408⟩⟩) (.sum [.result 192483 .summary, .result 189635 .summary])

def exact192488RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7205⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7207⟩⟩, ⟨.program ⟨257⟩, ⟨7125⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7209⟩⟩, ⟨.program ⟨257⟩, ⟨7107⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7211⟩⟩, ⟨.program ⟨257⟩, ⟨7103⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7213⟩⟩, ⟨.program ⟨257⟩, ⟨7099⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7215⟩⟩, ⟨.program ⟨257⟩, ⟨7173⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7217⟩⟩, ⟨.program ⟨257⟩, ⟨7169⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7219⟩⟩, ⟨.program ⟨257⟩, ⟨7167⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7221⟩⟩, ⟨.program ⟨257⟩, ⟨7163⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7223⟩⟩, ⟨.program ⟨257⟩, ⟨7161⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63142⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60162⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57182⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54202⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51222⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32158⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22138⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37678⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34998⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18918⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29341⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26661⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16078⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66798⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact192488RawTermsValid :
    exact192488RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event192488 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70408⟩⟩) exact192488RawTerms .large 192486 (.finite 4838999778777478503549183672281868407930932) (some (192487))

def event192489 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70409⟩⟩) 0 ⟨70408⟩ 192488

def event192490 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70409⟩⟩) 1 ⟨42062⟩ 189423

def event192491 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70409⟩⟩) (.sum [.predecessor 0 192489 .coefficient, .predecessor 1 192490 .coefficient])

def event192492 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70409⟩⟩) (.sum [.result 192488 .summary, .result 189423 .summary])

def exact192493RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7205⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7207⟩⟩, ⟨.program ⟨257⟩, ⟨7125⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7209⟩⟩, ⟨.program ⟨257⟩, ⟨7107⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7211⟩⟩, ⟨.program ⟨257⟩, ⟨7103⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7213⟩⟩, ⟨.program ⟨257⟩, ⟨7099⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7215⟩⟩, ⟨.program ⟨257⟩, ⟨7173⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7217⟩⟩, ⟨.program ⟨257⟩, ⟨7169⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7219⟩⟩, ⟨.program ⟨257⟩, ⟨7167⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7221⟩⟩, ⟨.program ⟨257⟩, ⟨7163⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7223⟩⟩, ⟨.program ⟨257⟩, ⟨7161⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7225⟩⟩, ⟨.program ⟨257⟩, ⟨7159⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63142⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60162⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57182⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54202⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51222⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32158⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22138⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40361⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37678⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34998⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18918⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29341⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26661⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16078⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66798⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact192493RawTermsValid :
    exact192493RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event192493 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70409⟩⟩) exact192493RawTerms .large 192491 (.finite 5184670870617817768629358718259150245068852) (some (192492))

def event192494 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70410⟩⟩) 0 ⟨70409⟩ 192493

def event192495 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70410⟩⟩) 1 ⟨44742⟩ 189211

def event192496 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70410⟩⟩) (.sum [.predecessor 0 192494 .coefficient, .predecessor 1 192495 .coefficient])

def event192497 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70410⟩⟩) (.sum [.result 192493 .summary, .result 189211 .summary])

def exact192498RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7205⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7207⟩⟩, ⟨.program ⟨257⟩, ⟨7125⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7209⟩⟩, ⟨.program ⟨257⟩, ⟨7107⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7211⟩⟩, ⟨.program ⟨257⟩, ⟨7103⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7213⟩⟩, ⟨.program ⟨257⟩, ⟨7099⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7215⟩⟩, ⟨.program ⟨257⟩, ⟨7173⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7217⟩⟩, ⟨.program ⟨257⟩, ⟨7169⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7219⟩⟩, ⟨.program ⟨257⟩, ⟨7167⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7221⟩⟩, ⟨.program ⟨257⟩, ⟨7163⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7223⟩⟩, ⟨.program ⟨257⟩, ⟨7161⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7225⟩⟩, ⟨.program ⟨257⟩, ⟨7159⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7227⟩⟩, ⟨.program ⟨257⟩, ⟨7153⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63142⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60162⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57182⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54202⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51222⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32158⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨43041⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22138⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40361⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37678⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34998⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18918⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29341⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26661⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16078⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66798⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact192498RawTermsValid :
    exact192498RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event192498 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70410⟩⟩) exact192498RawTerms .large 192496 (.finite 5530348290569953373030706035778833319198772) (some (192497))

def event192499 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70411⟩⟩) 0 ⟨70410⟩ 192498

def event192500 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70411⟩⟩) 1 ⟨47422⟩ 188999

def event192501 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70411⟩⟩) (.sum [.predecessor 0 192499 .coefficient, .predecessor 1 192500 .coefficient])

def event192502 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70411⟩⟩) (.sum [.result 192498 .summary, .result 188999 .summary])

def exact192503RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7205⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7207⟩⟩, ⟨.program ⟨257⟩, ⟨7125⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7209⟩⟩, ⟨.program ⟨257⟩, ⟨7107⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7211⟩⟩, ⟨.program ⟨257⟩, ⟨7103⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7213⟩⟩, ⟨.program ⟨257⟩, ⟨7099⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7215⟩⟩, ⟨.program ⟨257⟩, ⟨7173⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7217⟩⟩, ⟨.program ⟨257⟩, ⟨7169⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7219⟩⟩, ⟨.program ⟨257⟩, ⟨7167⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7221⟩⟩, ⟨.program ⟨257⟩, ⟨7163⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7223⟩⟩, ⟨.program ⟨257⟩, ⟨7161⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7225⟩⟩, ⟨.program ⟨257⟩, ⟨7159⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7227⟩⟩, ⟨.program ⟨257⟩, ⟨7153⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7229⟩⟩, ⟨.program ⟨257⟩, ⟨7151⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63142⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60162⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57182⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54202⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51222⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32158⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45718⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨43041⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22138⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40361⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37678⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34998⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18918⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29341⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26661⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16078⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66798⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact192503RawTermsValid :
    exact192503RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event192503 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70411⟩⟩) exact192503RawTerms .large 192501 (.finite 5876032038633885316753225624840917630320692) (some (192502))

def event192504 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70412⟩⟩) 0 ⟨70411⟩ 192503

def event192505 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70412⟩⟩) 1 ⟨50102⟩ 188787

def event192506 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70412⟩⟩) (.sum [.predecessor 0 192504 .coefficient, .predecessor 1 192505 .coefficient])

def event192507 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70412⟩⟩) (.sum [.result 192503 .summary, .result 188787 .summary])

def exact192508RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7205⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7207⟩⟩, ⟨.program ⟨257⟩, ⟨7125⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7209⟩⟩, ⟨.program ⟨257⟩, ⟨7107⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7211⟩⟩, ⟨.program ⟨257⟩, ⟨7103⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7213⟩⟩, ⟨.program ⟨257⟩, ⟨7099⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7215⟩⟩, ⟨.program ⟨257⟩, ⟨7173⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7217⟩⟩, ⟨.program ⟨257⟩, ⟨7169⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7219⟩⟩, ⟨.program ⟨257⟩, ⟨7167⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7221⟩⟩, ⟨.program ⟨257⟩, ⟨7163⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7223⟩⟩, ⟨.program ⟨257⟩, ⟨7161⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7225⟩⟩, ⟨.program ⟨257⟩, ⟨7159⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7227⟩⟩, ⟨.program ⟨257⟩, ⟨7153⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7229⟩⟩, ⟨.program ⟨257⟩, ⟨7151⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7231⟩⟩, ⟨.program ⟨257⟩, ⟨7147⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63142⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60162⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57182⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54202⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51222⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32158⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48398⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45718⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨43041⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22138⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40361⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37678⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34998⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18918⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29341⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26661⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16078⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66798⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact192508RawTermsValid :
    exact192508RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event192508 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70412⟩⟩) exact192508RawTerms .large 192506 (.finite 6221717896068416040249469304417135687106612) (some (192507))

def event192509 : Event := .predecessor (⟨.program ⟨257⟩, ⟨71335⟩⟩) 0 ⟨70412⟩ 192508

def event192510 : Event := .predecessor (⟨.program ⟨257⟩, ⟨71335⟩⟩) 1 ⟨71333⟩ 188575

def event192511 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨71335⟩⟩) (.sum [.predecessor 0 192509 .coefficient, .predecessor 1 192510 .coefficient])

def eventLeaf12016 : Array AnnotatedEvent := #[
  { event := event192256
    frameStart := 192210 },
  { event := event192257
    frameStart := 192210 },
  { event := event192258
    frameStart := 192210 },
  { event := event192259
    frameStart := 192210 },
  { event := event192260
    frameStart := 192210 },
  { event := event192261
    frameStart := 192210 },
  { event := event192262
    frameStart := 192210 },
  { event := event192263
    frameStart := 192210 },
  { event := event192264
    frameStart := 192264 },
  { event := event192265
    frameStart := 192264 },
  { event := event192266
    frameStart := 192264 },
  { event := event192267
    frameStart := 192264 },
  { event := event192268
    frameStart := 192264 },
  { event := event192269
    frameStart := 192264 },
  { event := event192270
    frameStart := 192264 },
  { event := event192271
    frameStart := 192264 }
]

def eventLeaf12017 : Array AnnotatedEvent := #[
  { event := event192272
    frameStart := 192264 },
  { event := event192273
    frameStart := 192264 },
  { event := event192274
    frameStart := 192264 },
  { event := event192275
    frameStart := 192264 },
  { event := event192276
    frameStart := 192264 },
  { event := event192277
    frameStart := 192264 },
  { event := event192278
    frameStart := 192264 },
  { event := event192279
    frameStart := 192264 },
  { event := event192280
    frameStart := 192264 },
  { event := event192281
    frameStart := 192264 },
  { event := event192282
    frameStart := 192264 },
  { event := event192283
    frameStart := 192264 },
  { event := event192284
    frameStart := 192264 },
  { event := event192285
    frameStart := 192264 },
  { event := event192286
    frameStart := 192264 },
  { event := event192287
    frameStart := 192264 }
]

def eventLeaf12018 : Array AnnotatedEvent := #[
  { event := event192288
    frameStart := 192264 },
  { event := event192289
    frameStart := 192264 },
  { event := event192290
    frameStart := 192264 },
  { event := event192291
    frameStart := 192264 },
  { event := event192292
    frameStart := 192264 },
  { event := event192293
    frameStart := 192264 },
  { event := event192294
    frameStart := 192264 },
  { event := event192295
    frameStart := 192264 },
  { event := event192296
    frameStart := 192264 },
  { event := event192297
    frameStart := 192264 },
  { event := event192298
    frameStart := 192264 },
  { event := event192299
    frameStart := 192264 },
  { event := event192300
    frameStart := 192264 },
  { event := event192301
    frameStart := 192264 },
  { event := event192302
    frameStart := 192264 },
  { event := event192303
    frameStart := 192264 }
]

def eventLeaf12019 : Array AnnotatedEvent := #[
  { event := event192304
    frameStart := 192264 },
  { event := event192305
    frameStart := 192264 },
  { event := event192306
    frameStart := 192264 },
  { event := event192307
    frameStart := 192264 },
  { event := event192308
    frameStart := 192264 },
  { event := event192309
    frameStart := 192264 },
  { event := event192310
    frameStart := 192264 },
  { event := event192311
    frameStart := 192264 },
  { event := event192312
    frameStart := 192264 },
  { event := event192313
    frameStart := 192264 },
  { event := event192314
    frameStart := 192264 },
  { event := event192315
    frameStart := 192264 },
  { event := event192316
    frameStart := 192264 },
  { event := event192317
    frameStart := 192264 },
  { event := event192318
    frameStart := 192264 },
  { event := event192319
    frameStart := 192264 }
]

def eventLeaf12020 : Array AnnotatedEvent := #[
  { event := event192320
    frameStart := 192264 },
  { event := event192321
    frameStart := 192264 },
  { event := event192322
    frameStart := 192264 },
  { event := event192323
    frameStart := 192264 },
  { event := event192324
    frameStart := 192264 },
  { event := event192325
    frameStart := 192264 },
  { event := event192326
    frameStart := 192264 },
  { event := event192327
    frameStart := 192264 },
  { event := event192328
    frameStart := 192264 },
  { event := event192329
    frameStart := 192264 },
  { event := event192330
    frameStart := 192264 },
  { event := event192331
    frameStart := 192264 },
  { event := event192332
    frameStart := 192264 },
  { event := event192333
    frameStart := 192264 },
  { event := event192334
    frameStart := 192264 },
  { event := event192335
    frameStart := 192264 }
]

def eventLeaf12021 : Array AnnotatedEvent := #[
  { event := event192336
    frameStart := 192264 },
  { event := event192337
    frameStart := 192264 },
  { event := event192338
    frameStart := 192264 },
  { event := event192339
    frameStart := 192264 },
  { event := event192340
    frameStart := 192264 },
  { event := event192341
    frameStart := 192264 },
  { event := event192342
    frameStart := 192264 },
  { event := event192343
    frameStart := 192264 },
  { event := event192344
    frameStart := 192264 },
  { event := event192345
    frameStart := 192264 },
  { event := event192346
    frameStart := 192264 },
  { event := event192347
    frameStart := 192264 },
  { event := event192348
    frameStart := 192264 },
  { event := event192349
    frameStart := 192264 },
  { event := event192350
    frameStart := 192264 },
  { event := event192351
    frameStart := 192264 }
]

def eventLeaf12022 : Array AnnotatedEvent := #[
  { event := event192352
    frameStart := 192264 },
  { event := event192353
    frameStart := 192264 },
  { event := event192354
    frameStart := 192264 },
  { event := event192355
    frameStart := 192264 },
  { event := event192356
    frameStart := 192264 },
  { event := event192357
    frameStart := 192264 },
  { event := event192358
    frameStart := 192264 },
  { event := event192359
    frameStart := 192264 },
  { event := event192360
    frameStart := 192264 },
  { event := event192361
    frameStart := 192264 },
  { event := event192362
    frameStart := 192264 },
  { event := event192363
    frameStart := 192264 },
  { event := event192364
    frameStart := 192264 },
  { event := event192365
    frameStart := 192264 },
  { event := event192366
    frameStart := 192264 },
  { event := event192367
    frameStart := 192264 }
]

def eventLeaf12023 : Array AnnotatedEvent := #[
  { event := event192368
    frameStart := 0 },
  { event := event192369
    frameStart := 0 },
  { event := event192370
    frameStart := 0 },
  { event := event192371
    frameStart := 0 },
  { event := event192372
    frameStart := 0 },
  { event := event192373
    frameStart := 0 },
  { event := event192374
    frameStart := 0 },
  { event := event192375
    frameStart := 0 },
  { event := event192376
    frameStart := 0 },
  { event := event192377
    frameStart := 0 },
  { event := event192378
    frameStart := 0 },
  { event := event192379
    frameStart := 0 },
  { event := event192380
    frameStart := 0 },
  { event := event192381
    frameStart := 0 },
  { event := event192382
    frameStart := 0 },
  { event := event192383
    frameStart := 0 }
]

def eventLeaf12024 : Array AnnotatedEvent := #[
  { event := event192384
    frameStart := 0 },
  { event := event192385
    frameStart := 0 },
  { event := event192386
    frameStart := 0 },
  { event := event192387
    frameStart := 0 },
  { event := event192388
    frameStart := 0 },
  { event := event192389
    frameStart := 0 },
  { event := event192390
    frameStart := 0 },
  { event := event192391
    frameStart := 0 },
  { event := event192392
    frameStart := 0 },
  { event := event192393
    frameStart := 0 },
  { event := event192394
    frameStart := 0 },
  { event := event192395
    frameStart := 0 },
  { event := event192396
    frameStart := 0 },
  { event := event192397
    frameStart := 0 },
  { event := event192398
    frameStart := 0 },
  { event := event192399
    frameStart := 0 }
]

def eventLeaf12025 : Array AnnotatedEvent := #[
  { event := event192400
    frameStart := 0 },
  { event := event192401
    frameStart := 0 },
  { event := event192402
    frameStart := 0 },
  { event := event192403
    frameStart := 0 },
  { event := event192404
    frameStart := 0 },
  { event := event192405
    frameStart := 0 },
  { event := event192406
    frameStart := 0 },
  { event := event192407
    frameStart := 0 },
  { event := event192408
    frameStart := 0 },
  { event := event192409
    frameStart := 0 },
  { event := event192410
    frameStart := 0 },
  { event := event192411
    frameStart := 0 },
  { event := event192412
    frameStart := 0 },
  { event := event192413
    frameStart := 0 },
  { event := event192414
    frameStart := 0 },
  { event := event192415
    frameStart := 0 }
]

def eventLeaf12026 : Array AnnotatedEvent := #[
  { event := event192416
    frameStart := 0 },
  { event := event192417
    frameStart := 0 },
  { event := event192418
    frameStart := 0 },
  { event := event192419
    frameStart := 0 },
  { event := event192420
    frameStart := 0 },
  { event := event192421
    frameStart := 0 },
  { event := event192422
    frameStart := 0 },
  { event := event192423
    frameStart := 0 },
  { event := event192424
    frameStart := 0 },
  { event := event192425
    frameStart := 0 },
  { event := event192426
    frameStart := 0 },
  { event := event192427
    frameStart := 0 },
  { event := event192428
    frameStart := 0 },
  { event := event192429
    frameStart := 0 },
  { event := event192430
    frameStart := 0 },
  { event := event192431
    frameStart := 0 }
]

def eventLeaf12027 : Array AnnotatedEvent := #[
  { event := event192432
    frameStart := 0 },
  { event := event192433
    frameStart := 0 },
  { event := event192434
    frameStart := 0 },
  { event := event192435
    frameStart := 0 },
  { event := event192436
    frameStart := 0 },
  { event := event192437
    frameStart := 0 },
  { event := event192438
    frameStart := 0 },
  { event := event192439
    frameStart := 0 },
  { event := event192440
    frameStart := 0 },
  { event := event192441
    frameStart := 0 },
  { event := event192442
    frameStart := 0 },
  { event := event192443
    frameStart := 0 },
  { event := event192444
    frameStart := 0 },
  { event := event192445
    frameStart := 0 },
  { event := event192446
    frameStart := 0 },
  { event := event192447
    frameStart := 0 }
]

def eventLeaf12028 : Array AnnotatedEvent := #[
  { event := event192448
    frameStart := 0 },
  { event := event192449
    frameStart := 0 },
  { event := event192450
    frameStart := 0 },
  { event := event192451
    frameStart := 0 },
  { event := event192452
    frameStart := 0 },
  { event := event192453
    frameStart := 0 },
  { event := event192454
    frameStart := 0 },
  { event := event192455
    frameStart := 0 },
  { event := event192456
    frameStart := 0 },
  { event := event192457
    frameStart := 0 },
  { event := event192458
    frameStart := 0 },
  { event := event192459
    frameStart := 0 },
  { event := event192460
    frameStart := 0 },
  { event := event192461
    frameStart := 0 },
  { event := event192462
    frameStart := 0 },
  { event := event192463
    frameStart := 0 }
]

def eventLeaf12029 : Array AnnotatedEvent := #[
  { event := event192464
    frameStart := 0 },
  { event := event192465
    frameStart := 0 },
  { event := event192466
    frameStart := 0 },
  { event := event192467
    frameStart := 0 },
  { event := event192468
    frameStart := 0 },
  { event := event192469
    frameStart := 0 },
  { event := event192470
    frameStart := 0 },
  { event := event192471
    frameStart := 0 },
  { event := event192472
    frameStart := 0 },
  { event := event192473
    frameStart := 0 },
  { event := event192474
    frameStart := 0 },
  { event := event192475
    frameStart := 0 },
  { event := event192476
    frameStart := 0 },
  { event := event192477
    frameStart := 0 },
  { event := event192478
    frameStart := 0 },
  { event := event192479
    frameStart := 0 }
]

def eventLeaf12030 : Array AnnotatedEvent := #[
  { event := event192480
    frameStart := 0 },
  { event := event192481
    frameStart := 0 },
  { event := event192482
    frameStart := 0 },
  { event := event192483
    frameStart := 0 },
  { event := event192484
    frameStart := 0 },
  { event := event192485
    frameStart := 0 },
  { event := event192486
    frameStart := 0 },
  { event := event192487
    frameStart := 0 },
  { event := event192488
    frameStart := 0 },
  { event := event192489
    frameStart := 0 },
  { event := event192490
    frameStart := 0 },
  { event := event192491
    frameStart := 0 },
  { event := event192492
    frameStart := 0 },
  { event := event192493
    frameStart := 0 },
  { event := event192494
    frameStart := 0 },
  { event := event192495
    frameStart := 0 }
]

def eventLeaf12031 : Array AnnotatedEvent := #[
  { event := event192496
    frameStart := 0 },
  { event := event192497
    frameStart := 0 },
  { event := event192498
    frameStart := 0 },
  { event := event192499
    frameStart := 0 },
  { event := event192500
    frameStart := 0 },
  { event := event192501
    frameStart := 0 },
  { event := event192502
    frameStart := 0 },
  { event := event192503
    frameStart := 0 },
  { event := event192504
    frameStart := 0 },
  { event := event192505
    frameStart := 0 },
  { event := event192506
    frameStart := 0 },
  { event := event192507
    frameStart := 0 },
  { event := event192508
    frameStart := 0 },
  { event := event192509
    frameStart := 0 },
  { event := event192510
    frameStart := 0 },
  { event := event192511
    frameStart := 0 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events751
