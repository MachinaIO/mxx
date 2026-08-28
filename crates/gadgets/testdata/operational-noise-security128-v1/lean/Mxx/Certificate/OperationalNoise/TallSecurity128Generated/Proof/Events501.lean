import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events501

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event128256 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17651⟩⟩, .operator (⟨128250, 0⟩, ⟨127973, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17649⟩⟩]⟩, (1)⟩)

def event128257 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17651⟩⟩, .operator (⟨128250, 1⟩, ⟨127973, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨15756⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17649⟩⟩]⟩, (-1)⟩)

def event128258 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨17651⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨15756⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17649⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨17649⟩⟩) ⟨16965⟩ 127970)

def event128259 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17651⟩⟩, .relation 128258 0, ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨15756⟩⟩], [⟨.program ⟨257⟩, ⟨16965⟩⟩]⟩, (-1)⟩)

def exact128260RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17649⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨15756⟩⟩], [⟨.program ⟨257⟩, ⟨16965⟩⟩]⟩, (-1)⟩]

theorem exact128260RawTermsValid :
    exact128260RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event128260 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17651⟩⟩) exact128260RawTerms .large 128253 (.finite 32188807212483504816668771614720) (some (128255))

def event128261 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16516⟩⟩) 0 ⟨15757⟩ 5738

def event128262 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16516⟩⟩) (.authority (.relationPreimageSource ⟨57⟩))

def exact128263RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨16516⟩⟩]⟩, (1)⟩]

theorem exact128263RawTermsValid :
    exact128263RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event128263 : Event := .resultExact (⟨.program ⟨257⟩, ⟨16516⟩⟩) exact128263RawTerms (.finite 5647228698) 128262 .exactZero (none)

def event128264 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16518⟩⟩) 0 ⟨16516⟩ 128263

def event128265 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16518⟩⟩) 1 ⟨2370⟩ 4

def event128266 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16518⟩⟩) (.scale (.predecessor 0 128264 .coefficient) (.value (.predecessor 1 128265 .coefficient)))

def exact128267RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨16516⟩⟩]⟩, (1)⟩]

theorem exact128267RawTermsValid :
    exact128267RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event128267 : Event := .resultExact (⟨.program ⟨257⟩, ⟨16518⟩⟩) exact128267RawTerms (.finite 5647228698) 128266 .exactZero (none)

def event128268 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16519⟩⟩) 0 ⟨5527⟩ 119870

def event128269 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16519⟩⟩) 1 ⟨16518⟩ 128267

def event128270 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16519⟩⟩) (.product (.predecessor 0 128268 .coefficient) (.predecessor 1 128269 .coefficient) (⟨false, false, none, none, none⟩))

def event128271 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16519⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨16516⟩⟩]⟩) [⟨.result 128263 .coefficient, false, none⟩])

def event128272 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16519⟩⟩) (.product (.result 119870 .summary) (.transfer 128271) (⟨false, false, none, none, none⟩))

def event128273 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16519⟩⟩, .operator (⟨119870, 0⟩, ⟨128267, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16516⟩⟩]⟩, (1)⟩)

def event128274 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨16517⟩⟩)

def event128275 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event128276 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event128277 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨2942⟩⟩) (.authority (.operator))

def event128278 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨2942⟩⟩) (.finite 12)

def event128279 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event128280 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event128281 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event128282 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event128283 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 128282

def event128284 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 128280

def event128285 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 128283 .coefficient) (.value (.predecessor 1 128284 .coefficient)))

def event128286 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event128287 : Event := .predecessor (⟨.program ⟨257⟩, ⟨2944⟩⟩) 0 ⟨392⟩ 128286

def event128288 : Event := .predecessor (⟨.program ⟨257⟩, ⟨2944⟩⟩) 1 ⟨2942⟩ 128278

def event128289 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨2944⟩⟩) (.sum [.predecessor 0 128287 .coefficient, .predecessor 1 128288 .coefficient])

def event128290 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨2944⟩⟩) (.finite 655352)

def event128291 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5523⟩⟩) 0 ⟨2944⟩ 128290

def event128292 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5523⟩⟩) 1 ⟨5426⟩ 128276

def event128293 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5523⟩⟩) (.identity (.predecessor 1 128292 .coefficient))

def event128294 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5523⟩⟩) (.finite 655360)

def event128295 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15378⟩⟩) 0 ⟨5523⟩ 128294

def event128296 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15378⟩⟩) (.authority (.programFamilyFact))

def exact128297RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15378⟩⟩], []⟩, (1)⟩]

theorem exact128297RawTermsValid :
    exact128297RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event128297 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15378⟩⟩) exact128297RawTerms (.finite 2) 128296 .exactZero (none)

def event128298 : Event := .predecessor (⟨.program ⟨257⟩, ⟨12321⟩⟩) 0 ⟨5523⟩ 128294

def event128299 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨12321⟩⟩) (.authority (.programFamilyFact))

def exact128300RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨12321⟩⟩], []⟩, (1)⟩]

theorem exact128300RawTermsValid :
    exact128300RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event128300 : Event := .resultExact (⟨.program ⟨257⟩, ⟨12321⟩⟩) exact128300RawTerms (.finite 2) 128299 .exactZero (none)

def event128301 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15379⟩⟩) 0 ⟨12321⟩ 128300

def event128302 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15379⟩⟩) 1 ⟨15378⟩ 128297

def event128303 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15379⟩⟩) (.product (.predecessor 0 128301 .coefficient) (.predecessor 1 128302 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event128304 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15379⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨12321⟩⟩, ⟨.program ⟨257⟩, ⟨15378⟩⟩], []⟩) [⟨.result 128300 .coefficient, true, some 1⟩, ⟨.result 128297 .coefficient, true, some 1⟩])

def event128305 : Event := .survivorFold (1) 128304

def exact128306RawTerms : List Term := []

theorem exact128306RawTermsValid :
    exact128306RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event128306 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15379⟩⟩) exact128306RawTerms (.finite 4) 128303 (.finite 4) (some (128304))

def event128307 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15380⟩⟩) 0 ⟨15379⟩ 128306

def event128308 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15380⟩⟩) (.identity (.predecessor 0 128307 .coefficient))

def event128309 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨15380⟩⟩) (.finite 4)

def event128310 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15756⟩⟩) 0 ⟨15380⟩ 128309

def event128311 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15756⟩⟩) (.authority (.programFamilyFact))

def exact128312RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15756⟩⟩], []⟩, (1)⟩]

theorem exact128312RawTermsValid :
    exact128312RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event128312 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15756⟩⟩) exact128312RawTerms (.finite 2) 128311 .exactZero (none)

def event128313 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15757⟩⟩) 0 ⟨15756⟩ 128312

def event128314 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15757⟩⟩) (.identity (.predecessor 0 128313 .coefficient))

def event128315 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨15757⟩⟩) (.finite 2)

def event128316 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16516⟩⟩) 0 ⟨15757⟩ 128315

def event128317 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16516⟩⟩) (.authority (.relationPreimageSource ⟨57⟩))

def exact128318RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨16516⟩⟩]⟩, (1)⟩]

theorem exact128318RawTermsValid :
    exact128318RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event128318 : Event := .resultExact (⟨.program ⟨257⟩, ⟨16516⟩⟩) exact128318RawTerms (.finite 5647228698) 128317 .exactZero (none)

def event128319 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35⟩⟩) (.authority (.operator))

def exact128320RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact128320RawTermsValid :
    exact128320RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event128320 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35⟩⟩) exact128320RawTerms .large 128319 .exactZero (none)

def event128321 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16517⟩⟩) 0 ⟨35⟩ 128320

def event128322 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16517⟩⟩) 1 ⟨16516⟩ 128318

def event128323 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16517⟩⟩) (.product (.predecessor 0 128321 .coefficient) (.predecessor 1 128322 .coefficient) (⟨false, false, none, none, none⟩))

def event128324 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16517⟩⟩, .operator (⟨128320, 0⟩, ⟨128318, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16516⟩⟩]⟩, (1)⟩)

def exact128325RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16516⟩⟩]⟩, (1)⟩]

theorem exact128325RawTermsValid :
    exact128325RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event128325 : Event := .resultExact (⟨.program ⟨257⟩, ⟨16517⟩⟩) exact128325RawTerms .large 128323 .exactZero (none)

def event128326 : Event := .preFoldPolynomial 128325 [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16516⟩⟩]⟩, (1)⟩] .exactZero none

def exact128327RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16516⟩⟩]⟩, (1)⟩]

def event128327 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨16517⟩⟩) 128326 exact128327RawTerms .large 128323 .exactZero (none)

def event128328 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨17653⟩⟩)

def event128329 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event128330 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event128331 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨2942⟩⟩) (.authority (.operator))

def event128332 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨2942⟩⟩) (.finite 12)

def event128333 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event128334 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event128335 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event128336 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event128337 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 128336

def event128338 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 128334

def event128339 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 128337 .coefficient) (.value (.predecessor 1 128338 .coefficient)))

def event128340 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event128341 : Event := .predecessor (⟨.program ⟨257⟩, ⟨2944⟩⟩) 0 ⟨392⟩ 128340

def event128342 : Event := .predecessor (⟨.program ⟨257⟩, ⟨2944⟩⟩) 1 ⟨2942⟩ 128332

def event128343 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨2944⟩⟩) (.sum [.predecessor 0 128341 .coefficient, .predecessor 1 128342 .coefficient])

def event128344 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨2944⟩⟩) (.finite 655352)

def event128345 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5523⟩⟩) 0 ⟨2944⟩ 128344

def event128346 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5523⟩⟩) 1 ⟨5426⟩ 128330

def event128347 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5523⟩⟩) (.identity (.predecessor 1 128346 .coefficient))

def event128348 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5523⟩⟩) (.finite 655360)

def event128349 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15378⟩⟩) 0 ⟨5523⟩ 128348

def event128350 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15378⟩⟩) (.authority (.programFamilyFact))

def exact128351RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15378⟩⟩], []⟩, (1)⟩]

theorem exact128351RawTermsValid :
    exact128351RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event128351 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15378⟩⟩) exact128351RawTerms (.finite 2) 128350 .exactZero (none)

def event128352 : Event := .predecessor (⟨.program ⟨257⟩, ⟨12321⟩⟩) 0 ⟨5523⟩ 128348

def event128353 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨12321⟩⟩) (.authority (.programFamilyFact))

def exact128354RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨12321⟩⟩], []⟩, (1)⟩]

theorem exact128354RawTermsValid :
    exact128354RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event128354 : Event := .resultExact (⟨.program ⟨257⟩, ⟨12321⟩⟩) exact128354RawTerms (.finite 2) 128353 .exactZero (none)

def event128355 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15379⟩⟩) 0 ⟨12321⟩ 128354

def event128356 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15379⟩⟩) 1 ⟨15378⟩ 128351

def event128357 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15379⟩⟩) (.product (.predecessor 0 128355 .coefficient) (.predecessor 1 128356 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event128358 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨15379⟩⟩, .operator (⟨128354, 0⟩, ⟨128351, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨12321⟩⟩, ⟨.program ⟨257⟩, ⟨15378⟩⟩], []⟩, (1)⟩)

def exact128359RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨12321⟩⟩, ⟨.program ⟨257⟩, ⟨15378⟩⟩], []⟩, (1)⟩]

theorem exact128359RawTermsValid :
    exact128359RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event128359 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15379⟩⟩) exact128359RawTerms (.finite 4) 128357 .exactZero (none)

def event128360 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15380⟩⟩) 0 ⟨15379⟩ 128359

def event128361 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15380⟩⟩) (.identity (.predecessor 0 128360 .coefficient))

def event128362 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨15380⟩⟩) (.finite 4)

def event128363 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15756⟩⟩) 0 ⟨15380⟩ 128362

def event128364 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15756⟩⟩) (.authority (.programFamilyFact))

def exact128365RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15756⟩⟩], []⟩, (1)⟩]

theorem exact128365RawTermsValid :
    exact128365RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event128365 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15756⟩⟩) exact128365RawTerms (.finite 2) 128364 .exactZero (none)

def event128366 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15757⟩⟩) 0 ⟨15756⟩ 128365

def event128367 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15757⟩⟩) (.identity (.predecessor 0 128366 .coefficient))

def event128368 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨15757⟩⟩) (.finite 2)

def event128369 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16963⟩⟩) 0 ⟨15757⟩ 128368

def event128370 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16963⟩⟩) (.authority (.programFamilyFact))

def event128371 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨16963⟩⟩) (.finite 3720)

def event128372 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨7177⟩⟩) .missing

def event128373 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16965⟩⟩) 0 ⟨7177⟩ 128372

def event128374 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16965⟩⟩) 1 ⟨16963⟩ 128371

def event128375 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16965⟩⟩) (.authority (.operator))

def exact128376RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨16965⟩⟩]⟩, (1)⟩]

theorem exact128376RawTermsValid :
    exact128376RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event128376 : Event := .resultExact (⟨.program ⟨257⟩, ⟨16965⟩⟩) exact128376RawTerms .large 128375 .exactZero (none)

def event128377 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17649⟩⟩) 0 ⟨16965⟩ 128376

def event128378 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17649⟩⟩) (.authority (.operator))

def exact128379RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨17649⟩⟩]⟩, (1)⟩]

theorem exact128379RawTermsValid :
    exact128379RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event128379 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17649⟩⟩) exact128379RawTerms (.finite 8192) 128378 .exactZero (none)

def event128380 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event128381 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event128382 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17190⟩⟩) 0 ⟨15757⟩ 128368

def event128383 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17190⟩⟩) 1 ⟨136⟩ 128381

def event128384 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17190⟩⟩) (.sum [.predecessor 0 128382 .coefficient, .predecessor 1 128383 .coefficient])

def event128385 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨17190⟩⟩) (.finite 2)

def event128386 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17191⟩⟩) 0 ⟨17190⟩ 128385

def event128387 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17191⟩⟩) (.identity (.predecessor 0 128386 .coefficient))

def exact128388RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15756⟩⟩], []⟩, (1)⟩]

theorem exact128388RawTermsValid :
    exact128388RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event128388 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17191⟩⟩) exact128388RawTerms (.finite 2) 128387 .exactZero (none)

def event128389 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact128390RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact128390RawTermsValid :
    exact128390RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event128390 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact128390RawTerms .large 128389 .exactZero (none)

def event128391 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17192⟩⟩) 0 ⟨6908⟩ 128390

def event128392 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17192⟩⟩) 1 ⟨17191⟩ 128388

def event128393 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17192⟩⟩) (.product (.predecessor 0 128391 .coefficient) (.predecessor 1 128392 .coefficient) (⟨false, false, none, none, none⟩))

def event128394 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17192⟩⟩, .operator (⟨128390, 0⟩, ⟨128388, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨15756⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact128395RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15756⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact128395RawTermsValid :
    exact128395RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event128395 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17192⟩⟩) exact128395RawTerms .large 128393 .exactZero (none)

def event128396 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7179⟩⟩) 0 ⟨7177⟩ 128372

def event128397 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7179⟩⟩) (.authority (.operator))

def exact128398RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩]⟩, (1)⟩]

theorem exact128398RawTermsValid :
    exact128398RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event128398 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7179⟩⟩) exact128398RawTerms .large 128397 .exactZero (none)

def event128399 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17193⟩⟩) 0 ⟨7179⟩ 128398

def event128400 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17193⟩⟩) 1 ⟨17192⟩ 128395

def event128401 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17193⟩⟩) (.sum [.predecessor 0 128399 .coefficient, .predecessor 1 128400 .coefficient])

def exact128402RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15756⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact128402RawTermsValid :
    exact128402RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event128402 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17193⟩⟩) exact128402RawTerms .large 128401 .exactZero (none)

def event128403 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17650⟩⟩) 0 ⟨17193⟩ 128402

def event128404 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17650⟩⟩) 1 ⟨17649⟩ 128379

def event128405 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17650⟩⟩) (.product (.predecessor 0 128403 .coefficient) (.predecessor 1 128404 .coefficient) (⟨false, false, none, none, none⟩))

def event128406 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17650⟩⟩, .operator (⟨128402, 0⟩, ⟨128379, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17649⟩⟩]⟩, (1)⟩)

def event128407 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17650⟩⟩, .operator (⟨128402, 1⟩, ⟨128379, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨15756⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17649⟩⟩]⟩, (-1)⟩)

def event128408 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨17650⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨15756⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17649⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨17649⟩⟩) ⟨16965⟩ 128376)

def event128409 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17650⟩⟩, .relation 128408 0, ⟨[⟨.program ⟨257⟩, ⟨15756⟩⟩], [⟨.program ⟨257⟩, ⟨16965⟩⟩]⟩, (-1)⟩)

def exact128410RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17649⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15756⟩⟩], [⟨.program ⟨257⟩, ⟨16965⟩⟩]⟩, (-1)⟩]

theorem exact128410RawTermsValid :
    exact128410RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event128410 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17650⟩⟩) exact128410RawTerms .large 128405 .exactZero (none)

def event128411 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15971⟩⟩) 0 ⟨15757⟩ 128368

def event128412 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15971⟩⟩) (.authority (.programFamilyFact))

def exact128413RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15971⟩⟩], []⟩, (1)⟩]

theorem exact128413RawTermsValid :
    exact128413RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event128413 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15971⟩⟩) exact128413RawTerms (.finite 43) 128412 .exactZero (none)

def event128414 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15972⟩⟩) 0 ⟨6908⟩ 128390

def event128415 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15972⟩⟩) 1 ⟨15971⟩ 128413

def event128416 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15972⟩⟩) (.product (.predecessor 0 128414 .coefficient) (.predecessor 1 128415 .coefficient) (⟨false, true, none, none, some 1⟩))

def event128417 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨15972⟩⟩, .operator (⟨128390, 0⟩, ⟨128413, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨15971⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact128418RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15971⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact128418RawTermsValid :
    exact128418RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event128418 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15972⟩⟩) exact128418RawTerms .large 128416 .exactZero (none)

def event128419 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7198⟩⟩) 0 ⟨7177⟩ 128372

def event128420 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7198⟩⟩) (.authority (.operator))

def exact128421RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩]

theorem exact128421RawTermsValid :
    exact128421RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event128421 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7198⟩⟩) exact128421RawTerms .large 128420 .exactZero (none)

def event128422 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15973⟩⟩) 0 ⟨7198⟩ 128421

def event128423 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15973⟩⟩) 1 ⟨15972⟩ 128418

def event128424 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15973⟩⟩) (.sum [.predecessor 0 128422 .coefficient, .predecessor 1 128423 .coefficient])

def exact128425RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15971⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact128425RawTermsValid :
    exact128425RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event128425 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15973⟩⟩) exact128425RawTerms .large 128424 .exactZero (none)

def event128426 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17653⟩⟩) 0 ⟨15973⟩ 128425

def event128427 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17653⟩⟩) 1 ⟨17650⟩ 128410

def event128428 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17653⟩⟩) (.sum [.predecessor 0 128426 .coefficient, .predecessor 1 128427 .coefficient])

def exact128429RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17649⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15756⟩⟩], [⟨.program ⟨257⟩, ⟨16965⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15971⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact128429RawTermsValid :
    exact128429RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event128429 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17653⟩⟩) exact128429RawTerms .large 128428 .exactZero (none)

def event128430 : Event := .preFoldPolynomial 128429 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17649⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15756⟩⟩], [⟨.program ⟨257⟩, ⟨16965⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15971⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact128431RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17649⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15756⟩⟩], [⟨.program ⟨257⟩, ⟨16965⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15971⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event128431 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨17653⟩⟩) 128430 exact128431RawTerms .large 128428 .exactZero (none)

def event128432 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨15757⟩⟩) ⟨⟨77⟩, ⟨57⟩, ⟨135⟩⟩ ⟨128274, 128432⟩

def event128433 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨16519⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16516⟩⟩]⟩) (1) 0 2 (.universal 128432 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16516⟩⟩]⟩) (none) 128431)

def event128434 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16519⟩⟩, .relation 128433 1, ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩)

def event128435 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16519⟩⟩, .relation 128433 0, ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17649⟩⟩]⟩, (-1)⟩)

def event128436 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16519⟩⟩, .relation 128433 2, ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨15756⟩⟩], [⟨.program ⟨257⟩, ⟨16965⟩⟩]⟩, (1)⟩)

def event128437 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16519⟩⟩, .relation 128433 3, ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨15971⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact128438RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17649⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨15756⟩⟩], [⟨.program ⟨257⟩, ⟨16965⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨15971⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact128438RawTermsValid :
    exact128438RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event128438 : Event := .resultExact (⟨.program ⟨257⟩, ⟨16519⟩⟩) exact128438RawTerms .large 128270 (.finite 202072841853861888) (some (128272))

def event128439 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17652⟩⟩) 0 ⟨16519⟩ 128438

def event128440 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17652⟩⟩) 1 ⟨17651⟩ 128260

def event128441 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17652⟩⟩) (.sum [.predecessor 0 128439 .coefficient, .predecessor 1 128440 .coefficient])

def event128442 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17652⟩⟩, .operator (⟨128438, 0⟩, ⟨128260, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17649⟩⟩]⟩, (1)⟩)

def event128443 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17652⟩⟩, .operator (⟨128438, 2⟩, ⟨128260, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨15756⟩⟩], [⟨.program ⟨257⟩, ⟨16965⟩⟩]⟩, (-1)⟩)

def event128444 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17652⟩⟩) (.sum [.result 128438 .summary, .result 128260 .summary])

def exact128445RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨15971⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact128445RawTermsValid :
    exact128445RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event128445 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17652⟩⟩) exact128445RawTerms .large 128441 (.finite 32188807212483706889510625476608) (some (128444))

def event128446 : Event := .predecessor (⟨.program ⟨257⟩, ⟨20532⟩⟩) 0 ⟨17652⟩ 128445

def event128447 : Event := .predecessor (⟨.program ⟨257⟩, ⟨20532⟩⟩) 1 ⟨20531⟩ 127963

def event128448 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨20532⟩⟩) (.sum [.predecessor 0 128446 .coefficient, .predecessor 1 128447 .coefficient])

def event128449 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨20532⟩⟩) (.sum [.result 128445 .summary, .result 127963 .summary])

def exact128450RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨15971⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨18790⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact128450RawTermsValid :
    exact128450RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event128450 : Event := .resultExact (⟨.program ⟨257⟩, ⟨20532⟩⟩) exact128450RawTerms .large 128448 (.finite 64377712650190257467641695830016) (some (128449))

def event128451 : Event := .predecessor (⟨.program ⟨257⟩, ⟨23752⟩⟩) 0 ⟨20532⟩ 128450

def event128452 : Event := .predecessor (⟨.program ⟨257⟩, ⟨23752⟩⟩) 1 ⟨23751⟩ 127481

def event128453 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨23752⟩⟩) (.sum [.predecessor 0 128451 .coefficient, .predecessor 1 128452 .coefficient])

def event128454 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨23752⟩⟩) (.sum [.result 128450 .summary, .result 127481 .summary])

def exact128455RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨15971⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨18790⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨22010⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact128455RawTermsValid :
    exact128455RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event128455 : Event := .resultExact (⟨.program ⟨257⟩, ⟨23752⟩⟩) exact128455RawTerms .large 128453 (.finite 96566716313119651734393211060224) (some (128454))

def event128456 : Event := .predecessor (⟨.program ⟨257⟩, ⟨33772⟩⟩) 0 ⟨23752⟩ 128455

def event128457 : Event := .predecessor (⟨.program ⟨257⟩, ⟨33772⟩⟩) 1 ⟨33771⟩ 126999

def event128458 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨33772⟩⟩) (.sum [.predecessor 0 128456 .coefficient, .predecessor 1 128457 .coefficient])

def event128459 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨33772⟩⟩) (.sum [.result 128455 .summary, .result 126999 .summary])

def exact128460RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨15971⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨18790⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨22010⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨32030⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact128460RawTermsValid :
    exact128460RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event128460 : Event := .resultExact (⟨.program ⟨257⟩, ⟨33772⟩⟩) exact128460RawTerms .large 128458 (.finite 128755916426494733378385616044032) (some (128459))

def event128461 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52832⟩⟩) 0 ⟨33772⟩ 128460

def event128462 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52832⟩⟩) 1 ⟨52831⟩ 126517

def event128463 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨52832⟩⟩) (.sum [.predecessor 0 128461 .coefficient, .predecessor 1 128462 .coefficient])

def event128464 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨52832⟩⟩) (.sum [.result 128460 .summary, .result 126517 .summary])

def exact128465RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨15971⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨18790⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨22010⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨32030⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨51085⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact128465RawTermsValid :
    exact128465RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event128465 : Event := .resultExact (⟨.program ⟨257⟩, ⟨52832⟩⟩) exact128465RawTerms .large 128463 (.finite 160945509440761189776859800535040) (some (128464))

def event128466 : Event := .predecessor (⟨.program ⟨257⟩, ⟨55812⟩⟩) 0 ⟨52832⟩ 128465

def event128467 : Event := .predecessor (⟨.program ⟨257⟩, ⟨55812⟩⟩) 1 ⟨55811⟩ 126035

def event128468 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨55812⟩⟩) (.sum [.predecessor 0 128466 .coefficient, .predecessor 1 128467 .coefficient])

def event128469 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨55812⟩⟩) (.sum [.result 128465 .summary, .result 126035 .summary])

def exact128470RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨15971⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨18790⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨22010⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨32030⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨51085⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨54065⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact128470RawTermsValid :
    exact128470RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event128470 : Event := .resultExact (⟨.program ⟨257⟩, ⟨55812⟩⟩) exact128470RawTerms .large 128468 (.finite 193135298905473333552574874779648) (some (128469))

def event128471 : Event := .predecessor (⟨.program ⟨257⟩, ⟨58792⟩⟩) 0 ⟨55812⟩ 128470

def event128472 : Event := .predecessor (⟨.program ⟨257⟩, ⟨58792⟩⟩) 1 ⟨58791⟩ 125553

def event128473 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨58792⟩⟩) (.sum [.predecessor 0 128471 .coefficient, .predecessor 1 128472 .coefficient])

def event128474 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨58792⟩⟩) (.sum [.result 128470 .summary, .result 125553 .summary])

def exact128475RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨15971⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨18790⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨22010⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨32030⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨51085⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨54065⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨57045⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact128475RawTermsValid :
    exact128475RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event128475 : Event := .resultExact (⟨.program ⟨257⟩, ⟨58792⟩⟩) exact128475RawTerms .large 128473 (.finite 225325481271076852082771728531456) (some (128474))

def event128476 : Event := .predecessor (⟨.program ⟨257⟩, ⟨61772⟩⟩) 0 ⟨58792⟩ 128475

def event128477 : Event := .predecessor (⟨.program ⟨257⟩, ⟨61772⟩⟩) 1 ⟨61771⟩ 125071

def event128478 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨61772⟩⟩) (.sum [.predecessor 0 128476 .coefficient, .predecessor 1 128477 .coefficient])

def event128479 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨61772⟩⟩) (.sum [.result 128475 .summary, .result 125071 .summary])

def exact128480RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨15971⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨18790⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨22010⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨32030⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨51085⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨54065⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨57045⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨60025⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact128480RawTermsValid :
    exact128480RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event128480 : Event := .resultExact (⟨.program ⟨257⟩, ⟨61772⟩⟩) exact128480RawTerms .large 128478 (.finite 257515860087126057990209472036864) (some (128479))

def event128481 : Event := .predecessor (⟨.program ⟨257⟩, ⟨64752⟩⟩) 0 ⟨61772⟩ 128480

def event128482 : Event := .predecessor (⟨.program ⟨257⟩, ⟨64752⟩⟩) 1 ⟨64751⟩ 124589

def event128483 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨64752⟩⟩) (.sum [.predecessor 0 128481 .coefficient, .predecessor 1 128482 .coefficient])

def event128484 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨64752⟩⟩) (.sum [.result 128480 .summary, .result 124589 .summary])

def exact128485RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨15971⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨18790⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨22010⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨32030⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨51085⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨54065⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨57045⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨60025⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨63005⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact128485RawTermsValid :
    exact128485RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event128485 : Event := .resultExact (⟨.program ⟨257⟩, ⟨64752⟩⟩) exact128485RawTerms .large 128483 (.finite 289706631804066638652128995049472) (some (128484))

def event128486 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69865⟩⟩) 0 ⟨64752⟩ 128485

def event128487 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69865⟩⟩) 1 ⟨69864⟩ 124107

def event128488 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69865⟩⟩) (.sum [.predecessor 0 128486 .coefficient, .predecessor 1 128487 .coefficient])

def event128489 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69865⟩⟩) (.sum [.result 128485 .summary, .result 124107 .summary])

def exact128490RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨15971⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨18790⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨22010⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨32030⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨51085⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨54065⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨57045⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨60025⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨63005⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨66321⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact128490RawTermsValid :
    exact128490RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event128490 : Event := .resultExact (⟨.program ⟨257⟩, ⟨69865⟩⟩) exact128490RawTerms .large 128488 (.finite 321897992872344281445771187322880) (some (128489))

def event128491 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69866⟩⟩) 0 ⟨69865⟩ 128490

def event128492 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69866⟩⟩) 1 ⟨28192⟩ 123625

def event128493 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69866⟩⟩) (.sum [.predecessor 0 128491 .coefficient, .predecessor 1 128492 .coefficient])

def event128494 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69866⟩⟩) (.sum [.result 128490 .summary, .result 123625 .summary])

def exact128495RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨15971⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨18790⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨22010⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨26567⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨32030⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨51085⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨54065⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨57045⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨60025⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨63005⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨66321⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact128495RawTermsValid :
    exact128495RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event128495 : Event := .resultExact (⟨.program ⟨257⟩, ⟨69866⟩⟩) exact128495RawTerms .large 128493 (.finite 354089550391067611616654269349888) (some (128494))

def event128496 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69867⟩⟩) 0 ⟨69866⟩ 128495

def event128497 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69867⟩⟩) 1 ⟨30872⟩ 123143

def event128498 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69867⟩⟩) (.sum [.predecessor 0 128496 .coefficient, .predecessor 1 128497 .coefficient])

def event128499 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69867⟩⟩) (.sum [.result 128495 .summary, .result 123143 .summary])

def exact128500RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨15971⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨18790⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨22010⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨26567⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨29247⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨32030⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨51085⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨54065⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨57045⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨60025⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨63005⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨66321⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact128500RawTermsValid :
    exact128500RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event128500 : Event := .resultExact (⟨.program ⟨257⟩, ⟨69867⟩⟩) exact128500RawTerms .large 128498 (.finite 386281697261128003919260020637696) (some (128499))

def event128501 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69868⟩⟩) 0 ⟨69867⟩ 128500

def event128502 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69868⟩⟩) 1 ⟨36532⟩ 122661

def event128503 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69868⟩⟩) (.sum [.predecessor 0 128501 .coefficient, .predecessor 1 128502 .coefficient])

def event128504 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69868⟩⟩) (.sum [.result 128500 .summary, .result 122661 .summary])

def exact128505RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨15971⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨18790⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨22010⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨26567⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨29247⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨32030⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨34911⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨51085⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨54065⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨57045⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨60025⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨63005⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨66321⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact128505RawTermsValid :
    exact128505RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event128505 : Event := .resultExact (⟨.program ⟨257⟩, ⟨69868⟩⟩) exact128505RawTerms .large 128503 (.finite 418474237032079770976347551432704) (some (128504))

def event128506 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69869⟩⟩) 0 ⟨69868⟩ 128505

def event128507 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69869⟩⟩) 1 ⟨39212⟩ 122179

def event128508 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69869⟩⟩) (.sum [.predecessor 0 128506 .coefficient, .predecessor 1 128507 .coefficient])

def event128509 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69869⟩⟩) (.sum [.result 128505 .summary, .result 122179 .summary])

def exact128510RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7224⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨15971⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨18790⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨22010⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨26567⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨29247⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨32030⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨34911⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨37591⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨51085⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨54065⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨57045⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨60025⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨63005⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨66321⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact128510RawTermsValid :
    exact128510RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event128510 : Event := .resultExact (⟨.program ⟨257⟩, ⟨69869⟩⟩) exact128510RawTerms .large 128508 (.finite 450666973253477225410675971981312) (some (128509))

def event128511 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69870⟩⟩) 0 ⟨69869⟩ 128510

def eventLeaf8016 : Array AnnotatedEvent := #[
  { event := event128256
    frameStart := 0 },
  { event := event128257
    frameStart := 0 },
  { event := event128258
    frameStart := 0 },
  { event := event128259
    frameStart := 0 },
  { event := event128260
    frameStart := 0 },
  { event := event128261
    frameStart := 0 },
  { event := event128262
    frameStart := 0 },
  { event := event128263
    frameStart := 0 },
  { event := event128264
    frameStart := 0 },
  { event := event128265
    frameStart := 0 },
  { event := event128266
    frameStart := 0 },
  { event := event128267
    frameStart := 0 },
  { event := event128268
    frameStart := 0 },
  { event := event128269
    frameStart := 0 },
  { event := event128270
    frameStart := 0 },
  { event := event128271
    frameStart := 0 }
]

def eventLeaf8017 : Array AnnotatedEvent := #[
  { event := event128272
    frameStart := 0 },
  { event := event128273
    frameStart := 0 },
  { event := event128274
    frameStart := 128274 },
  { event := event128275
    frameStart := 128274 },
  { event := event128276
    frameStart := 128274 },
  { event := event128277
    frameStart := 128274 },
  { event := event128278
    frameStart := 128274 },
  { event := event128279
    frameStart := 128274 },
  { event := event128280
    frameStart := 128274 },
  { event := event128281
    frameStart := 128274 },
  { event := event128282
    frameStart := 128274 },
  { event := event128283
    frameStart := 128274 },
  { event := event128284
    frameStart := 128274 },
  { event := event128285
    frameStart := 128274 },
  { event := event128286
    frameStart := 128274 },
  { event := event128287
    frameStart := 128274 }
]

def eventLeaf8018 : Array AnnotatedEvent := #[
  { event := event128288
    frameStart := 128274 },
  { event := event128289
    frameStart := 128274 },
  { event := event128290
    frameStart := 128274 },
  { event := event128291
    frameStart := 128274 },
  { event := event128292
    frameStart := 128274 },
  { event := event128293
    frameStart := 128274 },
  { event := event128294
    frameStart := 128274 },
  { event := event128295
    frameStart := 128274 },
  { event := event128296
    frameStart := 128274 },
  { event := event128297
    frameStart := 128274 },
  { event := event128298
    frameStart := 128274 },
  { event := event128299
    frameStart := 128274 },
  { event := event128300
    frameStart := 128274 },
  { event := event128301
    frameStart := 128274 },
  { event := event128302
    frameStart := 128274 },
  { event := event128303
    frameStart := 128274 }
]

def eventLeaf8019 : Array AnnotatedEvent := #[
  { event := event128304
    frameStart := 128274 },
  { event := event128305
    frameStart := 128274 },
  { event := event128306
    frameStart := 128274 },
  { event := event128307
    frameStart := 128274 },
  { event := event128308
    frameStart := 128274 },
  { event := event128309
    frameStart := 128274 },
  { event := event128310
    frameStart := 128274 },
  { event := event128311
    frameStart := 128274 },
  { event := event128312
    frameStart := 128274 },
  { event := event128313
    frameStart := 128274 },
  { event := event128314
    frameStart := 128274 },
  { event := event128315
    frameStart := 128274 },
  { event := event128316
    frameStart := 128274 },
  { event := event128317
    frameStart := 128274 },
  { event := event128318
    frameStart := 128274 },
  { event := event128319
    frameStart := 128274 }
]

def eventLeaf8020 : Array AnnotatedEvent := #[
  { event := event128320
    frameStart := 128274 },
  { event := event128321
    frameStart := 128274 },
  { event := event128322
    frameStart := 128274 },
  { event := event128323
    frameStart := 128274 },
  { event := event128324
    frameStart := 128274 },
  { event := event128325
    frameStart := 128274 },
  { event := event128326
    frameStart := 128274 },
  { event := event128327
    frameStart := 128274 },
  { event := event128328
    frameStart := 128328 },
  { event := event128329
    frameStart := 128328 },
  { event := event128330
    frameStart := 128328 },
  { event := event128331
    frameStart := 128328 },
  { event := event128332
    frameStart := 128328 },
  { event := event128333
    frameStart := 128328 },
  { event := event128334
    frameStart := 128328 },
  { event := event128335
    frameStart := 128328 }
]

def eventLeaf8021 : Array AnnotatedEvent := #[
  { event := event128336
    frameStart := 128328 },
  { event := event128337
    frameStart := 128328 },
  { event := event128338
    frameStart := 128328 },
  { event := event128339
    frameStart := 128328 },
  { event := event128340
    frameStart := 128328 },
  { event := event128341
    frameStart := 128328 },
  { event := event128342
    frameStart := 128328 },
  { event := event128343
    frameStart := 128328 },
  { event := event128344
    frameStart := 128328 },
  { event := event128345
    frameStart := 128328 },
  { event := event128346
    frameStart := 128328 },
  { event := event128347
    frameStart := 128328 },
  { event := event128348
    frameStart := 128328 },
  { event := event128349
    frameStart := 128328 },
  { event := event128350
    frameStart := 128328 },
  { event := event128351
    frameStart := 128328 }
]

def eventLeaf8022 : Array AnnotatedEvent := #[
  { event := event128352
    frameStart := 128328 },
  { event := event128353
    frameStart := 128328 },
  { event := event128354
    frameStart := 128328 },
  { event := event128355
    frameStart := 128328 },
  { event := event128356
    frameStart := 128328 },
  { event := event128357
    frameStart := 128328 },
  { event := event128358
    frameStart := 128328 },
  { event := event128359
    frameStart := 128328 },
  { event := event128360
    frameStart := 128328 },
  { event := event128361
    frameStart := 128328 },
  { event := event128362
    frameStart := 128328 },
  { event := event128363
    frameStart := 128328 },
  { event := event128364
    frameStart := 128328 },
  { event := event128365
    frameStart := 128328 },
  { event := event128366
    frameStart := 128328 },
  { event := event128367
    frameStart := 128328 }
]

def eventLeaf8023 : Array AnnotatedEvent := #[
  { event := event128368
    frameStart := 128328 },
  { event := event128369
    frameStart := 128328 },
  { event := event128370
    frameStart := 128328 },
  { event := event128371
    frameStart := 128328 },
  { event := event128372
    frameStart := 128328 },
  { event := event128373
    frameStart := 128328 },
  { event := event128374
    frameStart := 128328 },
  { event := event128375
    frameStart := 128328 },
  { event := event128376
    frameStart := 128328 },
  { event := event128377
    frameStart := 128328 },
  { event := event128378
    frameStart := 128328 },
  { event := event128379
    frameStart := 128328 },
  { event := event128380
    frameStart := 128328 },
  { event := event128381
    frameStart := 128328 },
  { event := event128382
    frameStart := 128328 },
  { event := event128383
    frameStart := 128328 }
]

def eventLeaf8024 : Array AnnotatedEvent := #[
  { event := event128384
    frameStart := 128328 },
  { event := event128385
    frameStart := 128328 },
  { event := event128386
    frameStart := 128328 },
  { event := event128387
    frameStart := 128328 },
  { event := event128388
    frameStart := 128328 },
  { event := event128389
    frameStart := 128328 },
  { event := event128390
    frameStart := 128328 },
  { event := event128391
    frameStart := 128328 },
  { event := event128392
    frameStart := 128328 },
  { event := event128393
    frameStart := 128328 },
  { event := event128394
    frameStart := 128328 },
  { event := event128395
    frameStart := 128328 },
  { event := event128396
    frameStart := 128328 },
  { event := event128397
    frameStart := 128328 },
  { event := event128398
    frameStart := 128328 },
  { event := event128399
    frameStart := 128328 }
]

def eventLeaf8025 : Array AnnotatedEvent := #[
  { event := event128400
    frameStart := 128328 },
  { event := event128401
    frameStart := 128328 },
  { event := event128402
    frameStart := 128328 },
  { event := event128403
    frameStart := 128328 },
  { event := event128404
    frameStart := 128328 },
  { event := event128405
    frameStart := 128328 },
  { event := event128406
    frameStart := 128328 },
  { event := event128407
    frameStart := 128328 },
  { event := event128408
    frameStart := 128328 },
  { event := event128409
    frameStart := 128328 },
  { event := event128410
    frameStart := 128328 },
  { event := event128411
    frameStart := 128328 },
  { event := event128412
    frameStart := 128328 },
  { event := event128413
    frameStart := 128328 },
  { event := event128414
    frameStart := 128328 },
  { event := event128415
    frameStart := 128328 }
]

def eventLeaf8026 : Array AnnotatedEvent := #[
  { event := event128416
    frameStart := 128328 },
  { event := event128417
    frameStart := 128328 },
  { event := event128418
    frameStart := 128328 },
  { event := event128419
    frameStart := 128328 },
  { event := event128420
    frameStart := 128328 },
  { event := event128421
    frameStart := 128328 },
  { event := event128422
    frameStart := 128328 },
  { event := event128423
    frameStart := 128328 },
  { event := event128424
    frameStart := 128328 },
  { event := event128425
    frameStart := 128328 },
  { event := event128426
    frameStart := 128328 },
  { event := event128427
    frameStart := 128328 },
  { event := event128428
    frameStart := 128328 },
  { event := event128429
    frameStart := 128328 },
  { event := event128430
    frameStart := 128328 },
  { event := event128431
    frameStart := 128328 }
]

def eventLeaf8027 : Array AnnotatedEvent := #[
  { event := event128432
    frameStart := 0 },
  { event := event128433
    frameStart := 0 },
  { event := event128434
    frameStart := 0 },
  { event := event128435
    frameStart := 0 },
  { event := event128436
    frameStart := 0 },
  { event := event128437
    frameStart := 0 },
  { event := event128438
    frameStart := 0 },
  { event := event128439
    frameStart := 0 },
  { event := event128440
    frameStart := 0 },
  { event := event128441
    frameStart := 0 },
  { event := event128442
    frameStart := 0 },
  { event := event128443
    frameStart := 0 },
  { event := event128444
    frameStart := 0 },
  { event := event128445
    frameStart := 0 },
  { event := event128446
    frameStart := 0 },
  { event := event128447
    frameStart := 0 }
]

def eventLeaf8028 : Array AnnotatedEvent := #[
  { event := event128448
    frameStart := 0 },
  { event := event128449
    frameStart := 0 },
  { event := event128450
    frameStart := 0 },
  { event := event128451
    frameStart := 0 },
  { event := event128452
    frameStart := 0 },
  { event := event128453
    frameStart := 0 },
  { event := event128454
    frameStart := 0 },
  { event := event128455
    frameStart := 0 },
  { event := event128456
    frameStart := 0 },
  { event := event128457
    frameStart := 0 },
  { event := event128458
    frameStart := 0 },
  { event := event128459
    frameStart := 0 },
  { event := event128460
    frameStart := 0 },
  { event := event128461
    frameStart := 0 },
  { event := event128462
    frameStart := 0 },
  { event := event128463
    frameStart := 0 }
]

def eventLeaf8029 : Array AnnotatedEvent := #[
  { event := event128464
    frameStart := 0 },
  { event := event128465
    frameStart := 0 },
  { event := event128466
    frameStart := 0 },
  { event := event128467
    frameStart := 0 },
  { event := event128468
    frameStart := 0 },
  { event := event128469
    frameStart := 0 },
  { event := event128470
    frameStart := 0 },
  { event := event128471
    frameStart := 0 },
  { event := event128472
    frameStart := 0 },
  { event := event128473
    frameStart := 0 },
  { event := event128474
    frameStart := 0 },
  { event := event128475
    frameStart := 0 },
  { event := event128476
    frameStart := 0 },
  { event := event128477
    frameStart := 0 },
  { event := event128478
    frameStart := 0 },
  { event := event128479
    frameStart := 0 }
]

def eventLeaf8030 : Array AnnotatedEvent := #[
  { event := event128480
    frameStart := 0 },
  { event := event128481
    frameStart := 0 },
  { event := event128482
    frameStart := 0 },
  { event := event128483
    frameStart := 0 },
  { event := event128484
    frameStart := 0 },
  { event := event128485
    frameStart := 0 },
  { event := event128486
    frameStart := 0 },
  { event := event128487
    frameStart := 0 },
  { event := event128488
    frameStart := 0 },
  { event := event128489
    frameStart := 0 },
  { event := event128490
    frameStart := 0 },
  { event := event128491
    frameStart := 0 },
  { event := event128492
    frameStart := 0 },
  { event := event128493
    frameStart := 0 },
  { event := event128494
    frameStart := 0 },
  { event := event128495
    frameStart := 0 }
]

def eventLeaf8031 : Array AnnotatedEvent := #[
  { event := event128496
    frameStart := 0 },
  { event := event128497
    frameStart := 0 },
  { event := event128498
    frameStart := 0 },
  { event := event128499
    frameStart := 0 },
  { event := event128500
    frameStart := 0 },
  { event := event128501
    frameStart := 0 },
  { event := event128502
    frameStart := 0 },
  { event := event128503
    frameStart := 0 },
  { event := event128504
    frameStart := 0 },
  { event := event128505
    frameStart := 0 },
  { event := event128506
    frameStart := 0 },
  { event := event128507
    frameStart := 0 },
  { event := event128508
    frameStart := 0 },
  { event := event128509
    frameStart := 0 },
  { event := event128510
    frameStart := 0 },
  { event := event128511
    frameStart := 0 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events501
