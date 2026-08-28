import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events966

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event247296 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47295⟩⟩) 1 ⟨47293⟩ 247294

def event247297 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47295⟩⟩) (.product (.predecessor 0 247295 .coefficient) (.predecessor 1 247296 .coefficient) (⟨false, false, none, none, none⟩))

def event247298 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47295⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨47293⟩⟩]⟩) [⟨.result 247294 .coefficient, false, none⟩])

def event247299 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47295⟩⟩) (.product (.result 237538 .summary) (.transfer 247298) (⟨false, false, none, none, none⟩))

def event247300 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨47295⟩⟩, .operator (⟨237538, 0⟩, ⟨247294, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47293⟩⟩]⟩, (1)⟩)

def event247301 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨47295⟩⟩, .operator (⟨237538, 1⟩, ⟨247294, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨45452⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨47293⟩⟩]⟩, (-1)⟩)

def event247302 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨47295⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨45452⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨47293⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨47293⟩⟩) ⟨46602⟩ 247291)

def event247303 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨47295⟩⟩, .relation 247302 0, ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨45452⟩⟩], [⟨.program ⟨257⟩, ⟨46602⟩⟩]⟩, (-1)⟩)

def exact247304RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47293⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨45452⟩⟩], [⟨.program ⟨257⟩, ⟨46602⟩⟩]⟩, (-1)⟩]

theorem exact247304RawTermsValid :
    exact247304RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event247304 : Event := .resultExact (⟨.program ⟨257⟩, ⟨47295⟩⟩) exact247304RawTerms .large 247297 (.finite 32194307824962751379413684715520) (some (247299))

def event247305 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46172⟩⟩) 0 ⟨45453⟩ 11354

def event247306 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46172⟩⟩) (.authority (.relationPreimageSource ⟨91⟩))

def exact247307RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨46172⟩⟩]⟩, (1)⟩]

theorem exact247307RawTermsValid :
    exact247307RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event247307 : Event := .resultExact (⟨.program ⟨257⟩, ⟨46172⟩⟩) exact247307RawTerms (.finite 5647228698) 247306 .exactZero (none)

def event247308 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46174⟩⟩) 0 ⟨46172⟩ 247307

def event247309 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46174⟩⟩) 1 ⟨2370⟩ 4

def event247310 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46174⟩⟩) (.scale (.predecessor 0 247308 .coefficient) (.value (.predecessor 1 247309 .coefficient)))

def exact247311RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨46172⟩⟩]⟩, (1)⟩]

theorem exact247311RawTermsValid :
    exact247311RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event247311 : Event := .resultExact (⟨.program ⟨257⟩, ⟨46174⟩⟩) exact247311RawTerms (.finite 5647228698) 247310 .exactZero (none)

def event247312 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46175⟩⟩) 0 ⟨5563⟩ 236870

def event247313 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46175⟩⟩) 1 ⟨46174⟩ 247311

def event247314 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46175⟩⟩) (.product (.predecessor 0 247312 .coefficient) (.predecessor 1 247313 .coefficient) (⟨false, false, none, none, none⟩))

def event247315 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46175⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨46172⟩⟩]⟩) [⟨.result 247307 .coefficient, false, none⟩])

def event247316 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46175⟩⟩) (.product (.result 236870 .summary) (.transfer 247315) (⟨false, false, none, none, none⟩))

def event247317 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨46175⟩⟩, .operator (⟨236870, 0⟩, ⟨247311, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46172⟩⟩]⟩, (1)⟩)

def event247318 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨46173⟩⟩)

def event247319 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event247320 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event247321 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨4740⟩⟩) (.authority (.operator))

def event247322 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨4740⟩⟩) (.finite 4)

def event247323 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event247324 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event247325 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event247326 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event247327 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 247326

def event247328 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 247324

def event247329 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 247327 .coefficient) (.value (.predecessor 1 247328 .coefficient)))

def event247330 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event247331 : Event := .predecessor (⟨.program ⟨257⟩, ⟨4742⟩⟩) 0 ⟨392⟩ 247330

def event247332 : Event := .predecessor (⟨.program ⟨257⟩, ⟨4742⟩⟩) 1 ⟨4740⟩ 247322

def event247333 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨4742⟩⟩) (.sum [.predecessor 0 247331 .coefficient, .predecessor 1 247332 .coefficient])

def event247334 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨4742⟩⟩) (.finite 655344)

def event247335 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5559⟩⟩) 0 ⟨4742⟩ 247334

def event247336 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5559⟩⟩) 1 ⟨5426⟩ 247320

def event247337 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5559⟩⟩) (.identity (.predecessor 1 247336 .coefficient))

def event247338 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5559⟩⟩) (.finite 655360)

def event247339 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45106⟩⟩) 0 ⟨5559⟩ 247338

def event247340 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45106⟩⟩) (.authority (.programFamilyFact))

def exact247341RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨45106⟩⟩], []⟩, (1)⟩]

theorem exact247341RawTermsValid :
    exact247341RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event247341 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45106⟩⟩) exact247341RawTerms (.finite 58) 247340 .exactZero (none)

def event247342 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14751⟩⟩) 0 ⟨5559⟩ 247338

def event247343 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14751⟩⟩) (.authority (.programFamilyFact))

def exact247344RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14751⟩⟩], []⟩, (1)⟩]

theorem exact247344RawTermsValid :
    exact247344RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event247344 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14751⟩⟩) exact247344RawTerms (.finite 58) 247343 .exactZero (none)

def event247345 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45107⟩⟩) 0 ⟨14751⟩ 247344

def event247346 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45107⟩⟩) 1 ⟨45106⟩ 247341

def event247347 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45107⟩⟩) (.product (.predecessor 0 247345 .coefficient) (.predecessor 1 247346 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event247348 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45107⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨14751⟩⟩, ⟨.program ⟨257⟩, ⟨45106⟩⟩], []⟩) [⟨.result 247344 .coefficient, true, some 1⟩, ⟨.result 247341 .coefficient, true, some 1⟩])

def event247349 : Event := .survivorFold (1) 247348

def exact247350RawTerms : List Term := []

theorem exact247350RawTermsValid :
    exact247350RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event247350 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45107⟩⟩) exact247350RawTerms (.finite 3364) 247347 (.finite 3364) (some (247348))

def event247351 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45108⟩⟩) 0 ⟨45107⟩ 247350

def event247352 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45108⟩⟩) (.identity (.predecessor 0 247351 .coefficient))

def event247353 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨45108⟩⟩) (.finite 3364)

def event247354 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45452⟩⟩) 0 ⟨45108⟩ 247353

def event247355 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45452⟩⟩) (.authority (.programFamilyFact))

def exact247356RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨45452⟩⟩], []⟩, (1)⟩]

theorem exact247356RawTermsValid :
    exact247356RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event247356 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45452⟩⟩) exact247356RawTerms (.finite 58) 247355 .exactZero (none)

def event247357 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45453⟩⟩) 0 ⟨45452⟩ 247356

def event247358 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45453⟩⟩) (.identity (.predecessor 0 247357 .coefficient))

def event247359 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨45453⟩⟩) (.finite 58)

def event247360 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46172⟩⟩) 0 ⟨45453⟩ 247359

def event247361 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46172⟩⟩) (.authority (.relationPreimageSource ⟨91⟩))

def exact247362RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨46172⟩⟩]⟩, (1)⟩]

theorem exact247362RawTermsValid :
    exact247362RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event247362 : Event := .resultExact (⟨.program ⟨257⟩, ⟨46172⟩⟩) exact247362RawTerms (.finite 5647228698) 247361 .exactZero (none)

def event247363 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35⟩⟩) (.authority (.operator))

def exact247364RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact247364RawTermsValid :
    exact247364RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event247364 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35⟩⟩) exact247364RawTerms .large 247363 .exactZero (none)

def event247365 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46173⟩⟩) 0 ⟨35⟩ 247364

def event247366 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46173⟩⟩) 1 ⟨46172⟩ 247362

def event247367 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46173⟩⟩) (.product (.predecessor 0 247365 .coefficient) (.predecessor 1 247366 .coefficient) (⟨false, false, none, none, none⟩))

def event247368 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨46173⟩⟩, .operator (⟨247364, 0⟩, ⟨247362, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46172⟩⟩]⟩, (1)⟩)

def exact247369RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46172⟩⟩]⟩, (1)⟩]

theorem exact247369RawTermsValid :
    exact247369RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event247369 : Event := .resultExact (⟨.program ⟨257⟩, ⟨46173⟩⟩) exact247369RawTerms .large 247367 .exactZero (none)

def event247370 : Event := .preFoldPolynomial 247369 [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46172⟩⟩]⟩, (1)⟩] .exactZero none

def exact247371RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46172⟩⟩]⟩, (1)⟩]

def event247371 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨46173⟩⟩) 247370 exact247371RawTerms .large 247367 .exactZero (none)

def event247372 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨47298⟩⟩)

def event247373 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event247374 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event247375 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨4740⟩⟩) (.authority (.operator))

def event247376 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨4740⟩⟩) (.finite 4)

def event247377 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event247378 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event247379 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event247380 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event247381 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 247380

def event247382 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 247378

def event247383 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 247381 .coefficient) (.value (.predecessor 1 247382 .coefficient)))

def event247384 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event247385 : Event := .predecessor (⟨.program ⟨257⟩, ⟨4742⟩⟩) 0 ⟨392⟩ 247384

def event247386 : Event := .predecessor (⟨.program ⟨257⟩, ⟨4742⟩⟩) 1 ⟨4740⟩ 247376

def event247387 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨4742⟩⟩) (.sum [.predecessor 0 247385 .coefficient, .predecessor 1 247386 .coefficient])

def event247388 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨4742⟩⟩) (.finite 655344)

def event247389 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5559⟩⟩) 0 ⟨4742⟩ 247388

def event247390 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5559⟩⟩) 1 ⟨5426⟩ 247374

def event247391 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5559⟩⟩) (.identity (.predecessor 1 247390 .coefficient))

def event247392 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5559⟩⟩) (.finite 655360)

def event247393 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45106⟩⟩) 0 ⟨5559⟩ 247392

def event247394 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45106⟩⟩) (.authority (.programFamilyFact))

def exact247395RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨45106⟩⟩], []⟩, (1)⟩]

theorem exact247395RawTermsValid :
    exact247395RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event247395 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45106⟩⟩) exact247395RawTerms (.finite 58) 247394 .exactZero (none)

def event247396 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14751⟩⟩) 0 ⟨5559⟩ 247392

def event247397 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14751⟩⟩) (.authority (.programFamilyFact))

def exact247398RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14751⟩⟩], []⟩, (1)⟩]

theorem exact247398RawTermsValid :
    exact247398RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event247398 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14751⟩⟩) exact247398RawTerms (.finite 58) 247397 .exactZero (none)

def event247399 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45107⟩⟩) 0 ⟨14751⟩ 247398

def event247400 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45107⟩⟩) 1 ⟨45106⟩ 247395

def event247401 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45107⟩⟩) (.product (.predecessor 0 247399 .coefficient) (.predecessor 1 247400 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event247402 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨45107⟩⟩, .operator (⟨247398, 0⟩, ⟨247395, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨14751⟩⟩, ⟨.program ⟨257⟩, ⟨45106⟩⟩], []⟩, (1)⟩)

def exact247403RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14751⟩⟩, ⟨.program ⟨257⟩, ⟨45106⟩⟩], []⟩, (1)⟩]

theorem exact247403RawTermsValid :
    exact247403RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event247403 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45107⟩⟩) exact247403RawTerms (.finite 3364) 247401 .exactZero (none)

def event247404 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45108⟩⟩) 0 ⟨45107⟩ 247403

def event247405 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45108⟩⟩) (.identity (.predecessor 0 247404 .coefficient))

def event247406 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨45108⟩⟩) (.finite 3364)

def event247407 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45452⟩⟩) 0 ⟨45108⟩ 247406

def event247408 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45452⟩⟩) (.authority (.programFamilyFact))

def exact247409RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨45452⟩⟩], []⟩, (1)⟩]

theorem exact247409RawTermsValid :
    exact247409RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event247409 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45452⟩⟩) exact247409RawTerms (.finite 58) 247408 .exactZero (none)

def event247410 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45453⟩⟩) 0 ⟨45452⟩ 247409

def event247411 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45453⟩⟩) (.identity (.predecessor 0 247410 .coefficient))

def event247412 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨45453⟩⟩) (.finite 58)

def event247413 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46601⟩⟩) 0 ⟨45453⟩ 247412

def event247414 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46601⟩⟩) (.authority (.programFamilyFact))

def event247415 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨46601⟩⟩) (.finite 3720)

def event247416 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨7177⟩⟩) .missing

def event247417 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46602⟩⟩) 0 ⟨7177⟩ 247416

def event247418 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46602⟩⟩) 1 ⟨46601⟩ 247415

def event247419 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46602⟩⟩) (.authority (.operator))

def exact247420RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨46602⟩⟩]⟩, (1)⟩]

theorem exact247420RawTermsValid :
    exact247420RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event247420 : Event := .resultExact (⟨.program ⟨257⟩, ⟨46602⟩⟩) exact247420RawTerms .large 247419 .exactZero (none)

def event247421 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47293⟩⟩) 0 ⟨46602⟩ 247420

def event247422 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47293⟩⟩) (.authority (.operator))

def exact247423RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨47293⟩⟩]⟩, (1)⟩]

theorem exact247423RawTermsValid :
    exact247423RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event247423 : Event := .resultExact (⟨.program ⟨257⟩, ⟨47293⟩⟩) exact247423RawTerms (.finite 8192) 247422 .exactZero (none)

def event247424 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event247425 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event247426 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46818⟩⟩) 0 ⟨45453⟩ 247412

def event247427 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46818⟩⟩) 1 ⟨136⟩ 247425

def event247428 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46818⟩⟩) (.sum [.predecessor 0 247426 .coefficient, .predecessor 1 247427 .coefficient])

def event247429 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨46818⟩⟩) (.finite 58)

def event247430 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46819⟩⟩) 0 ⟨46818⟩ 247429

def event247431 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46819⟩⟩) (.identity (.predecessor 0 247430 .coefficient))

def exact247432RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨45452⟩⟩], []⟩, (1)⟩]

theorem exact247432RawTermsValid :
    exact247432RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event247432 : Event := .resultExact (⟨.program ⟨257⟩, ⟨46819⟩⟩) exact247432RawTerms (.finite 58) 247431 .exactZero (none)

def event247433 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact247434RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact247434RawTermsValid :
    exact247434RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event247434 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact247434RawTerms .large 247433 .exactZero (none)

def event247435 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46820⟩⟩) 0 ⟨6908⟩ 247434

def event247436 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46820⟩⟩) 1 ⟨46819⟩ 247432

def event247437 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46820⟩⟩) (.product (.predecessor 0 247435 .coefficient) (.predecessor 1 247436 .coefficient) (⟨false, false, none, none, none⟩))

def event247438 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨46820⟩⟩, .operator (⟨247434, 0⟩, ⟨247432, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨45452⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact247439RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨45452⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact247439RawTermsValid :
    exact247439RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event247439 : Event := .resultExact (⟨.program ⟨257⟩, ⟨46820⟩⟩) exact247439RawTerms .large 247437 .exactZero (none)

def event247440 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7195⟩⟩) 0 ⟨7177⟩ 247416

def event247441 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7195⟩⟩) (.authority (.operator))

def exact247442RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7195⟩⟩]⟩, (1)⟩]

theorem exact247442RawTermsValid :
    exact247442RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event247442 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7195⟩⟩) exact247442RawTerms .large 247441 .exactZero (none)

def event247443 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46821⟩⟩) 0 ⟨7195⟩ 247442

def event247444 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46821⟩⟩) 1 ⟨46820⟩ 247439

def event247445 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46821⟩⟩) (.sum [.predecessor 0 247443 .coefficient, .predecessor 1 247444 .coefficient])

def exact247446RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7195⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨45452⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact247446RawTermsValid :
    exact247446RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event247446 : Event := .resultExact (⟨.program ⟨257⟩, ⟨46821⟩⟩) exact247446RawTerms .large 247445 .exactZero (none)

def event247447 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47294⟩⟩) 0 ⟨46821⟩ 247446

def event247448 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47294⟩⟩) 1 ⟨47293⟩ 247423

def event247449 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47294⟩⟩) (.product (.predecessor 0 247447 .coefficient) (.predecessor 1 247448 .coefficient) (⟨false, false, none, none, none⟩))

def event247450 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨47294⟩⟩, .operator (⟨247446, 0⟩, ⟨247423, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47293⟩⟩]⟩, (1)⟩)

def event247451 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨47294⟩⟩, .operator (⟨247446, 1⟩, ⟨247423, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨45452⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨47293⟩⟩]⟩, (-1)⟩)

def event247452 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨47294⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨45452⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨47293⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨47293⟩⟩) ⟨46602⟩ 247420)

def event247453 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨47294⟩⟩, .relation 247452 0, ⟨[⟨.program ⟨257⟩, ⟨45452⟩⟩], [⟨.program ⟨257⟩, ⟨46602⟩⟩]⟩, (-1)⟩)

def exact247454RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47293⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨45452⟩⟩], [⟨.program ⟨257⟩, ⟨46602⟩⟩]⟩, (-1)⟩]

theorem exact247454RawTermsValid :
    exact247454RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event247454 : Event := .resultExact (⟨.program ⟨257⟩, ⟨47294⟩⟩) exact247454RawTerms .large 247449 .exactZero (none)

def event247455 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45653⟩⟩) 0 ⟨45453⟩ 247412

def event247456 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45653⟩⟩) (.authority (.programFamilyFact))

def exact247457RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨45653⟩⟩], []⟩, (1)⟩]

theorem exact247457RawTermsValid :
    exact247457RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event247457 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45653⟩⟩) exact247457RawTerms (.finite 58) 247456 .exactZero (none)

def event247458 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45655⟩⟩) 0 ⟨6908⟩ 247434

def event247459 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45655⟩⟩) 1 ⟨45653⟩ 247457

def event247460 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45655⟩⟩) (.product (.predecessor 0 247458 .coefficient) (.predecessor 1 247459 .coefficient) (⟨false, true, none, none, some 1⟩))

def event247461 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨45655⟩⟩, .operator (⟨247434, 0⟩, ⟨247457, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨45653⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact247462RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨45653⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact247462RawTermsValid :
    exact247462RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event247462 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45655⟩⟩) exact247462RawTerms .large 247460 .exactZero (none)

def event247463 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7229⟩⟩) 0 ⟨7177⟩ 247416

def event247464 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7229⟩⟩) (.authority (.operator))

def exact247465RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7229⟩⟩]⟩, (1)⟩]

theorem exact247465RawTermsValid :
    exact247465RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event247465 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7229⟩⟩) exact247465RawTerms .large 247464 .exactZero (none)

def event247466 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45656⟩⟩) 0 ⟨7229⟩ 247465

def event247467 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45656⟩⟩) 1 ⟨45655⟩ 247462

def event247468 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45656⟩⟩) (.sum [.predecessor 0 247466 .coefficient, .predecessor 1 247467 .coefficient])

def exact247469RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7229⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨45653⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact247469RawTermsValid :
    exact247469RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event247469 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45656⟩⟩) exact247469RawTerms .large 247468 .exactZero (none)

def event247470 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47298⟩⟩) 0 ⟨45656⟩ 247469

def event247471 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47298⟩⟩) 1 ⟨47294⟩ 247454

def event247472 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47298⟩⟩) (.sum [.predecessor 0 247470 .coefficient, .predecessor 1 247471 .coefficient])

def exact247473RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47293⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7229⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨45452⟩⟩], [⟨.program ⟨257⟩, ⟨46602⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨45653⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact247473RawTermsValid :
    exact247473RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event247473 : Event := .resultExact (⟨.program ⟨257⟩, ⟨47298⟩⟩) exact247473RawTerms .large 247472 .exactZero (none)

def event247474 : Event := .preFoldPolynomial 247473 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47293⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7229⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨45452⟩⟩], [⟨.program ⟨257⟩, ⟨46602⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨45653⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact247475RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47293⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7229⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨45452⟩⟩], [⟨.program ⟨257⟩, ⟨46602⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨45653⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event247475 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨47298⟩⟩) 247474 exact247475RawTerms .large 247472 .exactZero (none)

def event247476 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨45453⟩⟩) ⟨⟨108⟩, ⟨91⟩, ⟨135⟩⟩ ⟨247318, 247476⟩

def event247477 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨46175⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46172⟩⟩]⟩) (1) 0 2 (.universal 247476 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46172⟩⟩]⟩) (none) 247475)

def event247478 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨46175⟩⟩, .relation 247477 1, ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7229⟩⟩]⟩, (1)⟩)

def event247479 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨46175⟩⟩, .relation 247477 0, ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47293⟩⟩]⟩, (-1)⟩)

def event247480 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨46175⟩⟩, .relation 247477 2, ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨45452⟩⟩], [⟨.program ⟨257⟩, ⟨46602⟩⟩]⟩, (1)⟩)

def event247481 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨46175⟩⟩, .relation 247477 3, ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨45653⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact247482RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47293⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7229⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨45452⟩⟩], [⟨.program ⟨257⟩, ⟨46602⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨45653⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact247482RawTermsValid :
    exact247482RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event247482 : Event := .resultExact (⟨.program ⟨257⟩, ⟨46175⟩⟩) exact247482RawTerms .large 247314 (.finite 202072841853861888) (some (247316))

def event247483 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47296⟩⟩) 0 ⟨46175⟩ 247482

def event247484 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47296⟩⟩) 1 ⟨47295⟩ 247304

def event247485 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47296⟩⟩) (.sum [.predecessor 0 247483 .coefficient, .predecessor 1 247484 .coefficient])

def event247486 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨47296⟩⟩, .operator (⟨247482, 0⟩, ⟨247304, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47293⟩⟩]⟩, (1)⟩)

def event247487 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨47296⟩⟩, .operator (⟨247482, 2⟩, ⟨247304, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨45452⟩⟩], [⟨.program ⟨257⟩, ⟨46602⟩⟩]⟩, (-1)⟩)

def event247488 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47296⟩⟩) (.sum [.result 247482 .summary, .result 247304 .summary])

def exact247489RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7229⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨45653⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact247489RawTermsValid :
    exact247489RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event247489 : Event := .resultExact (⟨.program ⟨257⟩, ⟨47296⟩⟩) exact247489RawTerms .large 247485 (.finite 32194307824962953452255538577408) (some (247488))

def event247490 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47297⟩⟩) 0 ⟨47296⟩ 247489

def event247491 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47297⟩⟩) 1 ⟨7152⟩ 15562

def event247492 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47297⟩⟩) (.product (.predecessor 0 247490 .coefficient) (.predecessor 1 247491 .coefficient) (⟨false, false, none, none, none⟩))

def event247493 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47297⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨7151⟩⟩]⟩) [⟨.result 15558 .coefficient, false, none⟩])

def event247494 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47297⟩⟩) (.product (.result 247489 .summary) (.transfer 247493) (⟨false, false, none, none, none⟩))

def event247495 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨47297⟩⟩, .operator (⟨247489, 0⟩, ⟨15562, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7229⟩⟩, ⟨.program ⟨257⟩, ⟨7151⟩⟩]⟩, (1)⟩)

def event247496 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨47297⟩⟩, .operator (⟨247489, 1⟩, ⟨15562, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨45653⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7151⟩⟩]⟩, (-1)⟩)

def event247497 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨47297⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨45653⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7151⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨7151⟩⟩) ⟨7041⟩ 15555)

def event247498 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨47297⟩⟩, .relation 247497 0, ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45653⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact247499RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7229⟩⟩, ⟨.program ⟨257⟩, ⟨7151⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45653⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact247499RawTermsValid :
    exact247499RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event247499 : Event := .resultExact (⟨.program ⟨257⟩, ⟨47297⟩⟩) exact247499RawTerms .large 247492 (.finite 345683748063931943722519589062084311121920) (some (247494))

def event247500 : Event := .predecessor (⟨.program ⟨257⟩, ⟨43922⟩⟩) 0 ⟨7177⟩ 15500

def event247501 : Event := .predecessor (⟨.program ⟨257⟩, ⟨43922⟩⟩) 1 ⟨43921⟩ 237736

def event247502 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨43922⟩⟩) (.authority (.operator))

def exact247503RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨43922⟩⟩]⟩, (1)⟩]

theorem exact247503RawTermsValid :
    exact247503RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event247503 : Event := .resultExact (⟨.program ⟨257⟩, ⟨43922⟩⟩) exact247503RawTerms .large 247502 .exactZero (none)

def event247504 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44613⟩⟩) 0 ⟨43922⟩ 247503

def event247505 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44613⟩⟩) (.authority (.operator))

def exact247506RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨44613⟩⟩]⟩, (1)⟩]

theorem exact247506RawTermsValid :
    exact247506RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event247506 : Event := .resultExact (⟨.program ⟨257⟩, ⟨44613⟩⟩) exact247506RawTerms (.finite 8192) 247505 .exactZero (none)

def event247507 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44615⟩⟩) 0 ⟨44279⟩ 238020

def event247508 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44615⟩⟩) 1 ⟨44613⟩ 247506

def event247509 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44615⟩⟩) (.product (.predecessor 0 247507 .coefficient) (.predecessor 1 247508 .coefficient) (⟨false, false, none, none, none⟩))

def event247510 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44615⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨44613⟩⟩]⟩) [⟨.result 247506 .coefficient, false, none⟩])

def event247511 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44615⟩⟩) (.product (.result 238020 .summary) (.transfer 247510) (⟨false, false, none, none, none⟩))

def event247512 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨44615⟩⟩, .operator (⟨238020, 0⟩, ⟨247506, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44613⟩⟩]⟩, (1)⟩)

def event247513 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨44615⟩⟩, .operator (⟨238020, 1⟩, ⟨247506, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨42772⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨44613⟩⟩]⟩, (-1)⟩)

def event247514 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨44615⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨42772⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨44613⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨44613⟩⟩) ⟨43922⟩ 247503)

def event247515 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨44615⟩⟩, .relation 247514 0, ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨42772⟩⟩], [⟨.program ⟨257⟩, ⟨43922⟩⟩]⟩, (-1)⟩)

def exact247516RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44613⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨42772⟩⟩], [⟨.program ⟨257⟩, ⟨43922⟩⟩]⟩, (-1)⟩]

theorem exact247516RawTermsValid :
    exact247516RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event247516 : Event := .resultExact (⟨.program ⟨257⟩, ⟨44615⟩⟩) exact247516RawTerms .large 247509 (.finite 32193718473625689247691015454720) (some (247511))

def event247517 : Event := .predecessor (⟨.program ⟨257⟩, ⟨43492⟩⟩) 0 ⟨42773⟩ 11377

def event247518 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨43492⟩⟩) (.authority (.relationPreimageSource ⟨89⟩))

def exact247519RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨43492⟩⟩]⟩, (1)⟩]

theorem exact247519RawTermsValid :
    exact247519RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event247519 : Event := .resultExact (⟨.program ⟨257⟩, ⟨43492⟩⟩) exact247519RawTerms (.finite 5647228698) 247518 .exactZero (none)

def event247520 : Event := .predecessor (⟨.program ⟨257⟩, ⟨43494⟩⟩) 0 ⟨43492⟩ 247519

def event247521 : Event := .predecessor (⟨.program ⟨257⟩, ⟨43494⟩⟩) 1 ⟨2370⟩ 4

def event247522 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨43494⟩⟩) (.scale (.predecessor 0 247520 .coefficient) (.value (.predecessor 1 247521 .coefficient)))

def exact247523RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨43492⟩⟩]⟩, (1)⟩]

theorem exact247523RawTermsValid :
    exact247523RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event247523 : Event := .resultExact (⟨.program ⟨257⟩, ⟨43494⟩⟩) exact247523RawTerms (.finite 5647228698) 247522 .exactZero (none)

def event247524 : Event := .predecessor (⟨.program ⟨257⟩, ⟨43495⟩⟩) 0 ⟨5563⟩ 236870

def event247525 : Event := .predecessor (⟨.program ⟨257⟩, ⟨43495⟩⟩) 1 ⟨43494⟩ 247523

def event247526 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨43495⟩⟩) (.product (.predecessor 0 247524 .coefficient) (.predecessor 1 247525 .coefficient) (⟨false, false, none, none, none⟩))

def event247527 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨43495⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨43492⟩⟩]⟩) [⟨.result 247519 .coefficient, false, none⟩])

def event247528 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨43495⟩⟩) (.product (.result 236870 .summary) (.transfer 247527) (⟨false, false, none, none, none⟩))

def event247529 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨43495⟩⟩, .operator (⟨236870, 0⟩, ⟨247523, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43492⟩⟩]⟩, (1)⟩)

def event247530 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨43493⟩⟩)

def event247531 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event247532 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event247533 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨4740⟩⟩) (.authority (.operator))

def event247534 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨4740⟩⟩) (.finite 4)

def event247535 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event247536 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event247537 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event247538 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event247539 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 247538

def event247540 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 247536

def event247541 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 247539 .coefficient) (.value (.predecessor 1 247540 .coefficient)))

def event247542 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event247543 : Event := .predecessor (⟨.program ⟨257⟩, ⟨4742⟩⟩) 0 ⟨392⟩ 247542

def event247544 : Event := .predecessor (⟨.program ⟨257⟩, ⟨4742⟩⟩) 1 ⟨4740⟩ 247534

def event247545 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨4742⟩⟩) (.sum [.predecessor 0 247543 .coefficient, .predecessor 1 247544 .coefficient])

def event247546 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨4742⟩⟩) (.finite 655344)

def event247547 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5559⟩⟩) 0 ⟨4742⟩ 247546

def event247548 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5559⟩⟩) 1 ⟨5426⟩ 247532

def event247549 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5559⟩⟩) (.identity (.predecessor 1 247548 .coefficient))

def event247550 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5559⟩⟩) (.finite 655360)

def event247551 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42426⟩⟩) 0 ⟨5559⟩ 247550

def eventLeaf15456 : Array AnnotatedEvent := #[
  { event := event247296
    frameStart := 0 },
  { event := event247297
    frameStart := 0 },
  { event := event247298
    frameStart := 0 },
  { event := event247299
    frameStart := 0 },
  { event := event247300
    frameStart := 0 },
  { event := event247301
    frameStart := 0 },
  { event := event247302
    frameStart := 0 },
  { event := event247303
    frameStart := 0 },
  { event := event247304
    frameStart := 0 },
  { event := event247305
    frameStart := 0 },
  { event := event247306
    frameStart := 0 },
  { event := event247307
    frameStart := 0 },
  { event := event247308
    frameStart := 0 },
  { event := event247309
    frameStart := 0 },
  { event := event247310
    frameStart := 0 },
  { event := event247311
    frameStart := 0 }
]

def eventLeaf15457 : Array AnnotatedEvent := #[
  { event := event247312
    frameStart := 0 },
  { event := event247313
    frameStart := 0 },
  { event := event247314
    frameStart := 0 },
  { event := event247315
    frameStart := 0 },
  { event := event247316
    frameStart := 0 },
  { event := event247317
    frameStart := 0 },
  { event := event247318
    frameStart := 247318 },
  { event := event247319
    frameStart := 247318 },
  { event := event247320
    frameStart := 247318 },
  { event := event247321
    frameStart := 247318 },
  { event := event247322
    frameStart := 247318 },
  { event := event247323
    frameStart := 247318 },
  { event := event247324
    frameStart := 247318 },
  { event := event247325
    frameStart := 247318 },
  { event := event247326
    frameStart := 247318 },
  { event := event247327
    frameStart := 247318 }
]

def eventLeaf15458 : Array AnnotatedEvent := #[
  { event := event247328
    frameStart := 247318 },
  { event := event247329
    frameStart := 247318 },
  { event := event247330
    frameStart := 247318 },
  { event := event247331
    frameStart := 247318 },
  { event := event247332
    frameStart := 247318 },
  { event := event247333
    frameStart := 247318 },
  { event := event247334
    frameStart := 247318 },
  { event := event247335
    frameStart := 247318 },
  { event := event247336
    frameStart := 247318 },
  { event := event247337
    frameStart := 247318 },
  { event := event247338
    frameStart := 247318 },
  { event := event247339
    frameStart := 247318 },
  { event := event247340
    frameStart := 247318 },
  { event := event247341
    frameStart := 247318 },
  { event := event247342
    frameStart := 247318 },
  { event := event247343
    frameStart := 247318 }
]

def eventLeaf15459 : Array AnnotatedEvent := #[
  { event := event247344
    frameStart := 247318 },
  { event := event247345
    frameStart := 247318 },
  { event := event247346
    frameStart := 247318 },
  { event := event247347
    frameStart := 247318 },
  { event := event247348
    frameStart := 247318 },
  { event := event247349
    frameStart := 247318 },
  { event := event247350
    frameStart := 247318 },
  { event := event247351
    frameStart := 247318 },
  { event := event247352
    frameStart := 247318 },
  { event := event247353
    frameStart := 247318 },
  { event := event247354
    frameStart := 247318 },
  { event := event247355
    frameStart := 247318 },
  { event := event247356
    frameStart := 247318 },
  { event := event247357
    frameStart := 247318 },
  { event := event247358
    frameStart := 247318 },
  { event := event247359
    frameStart := 247318 }
]

def eventLeaf15460 : Array AnnotatedEvent := #[
  { event := event247360
    frameStart := 247318 },
  { event := event247361
    frameStart := 247318 },
  { event := event247362
    frameStart := 247318 },
  { event := event247363
    frameStart := 247318 },
  { event := event247364
    frameStart := 247318 },
  { event := event247365
    frameStart := 247318 },
  { event := event247366
    frameStart := 247318 },
  { event := event247367
    frameStart := 247318 },
  { event := event247368
    frameStart := 247318 },
  { event := event247369
    frameStart := 247318 },
  { event := event247370
    frameStart := 247318 },
  { event := event247371
    frameStart := 247318 },
  { event := event247372
    frameStart := 247372 },
  { event := event247373
    frameStart := 247372 },
  { event := event247374
    frameStart := 247372 },
  { event := event247375
    frameStart := 247372 }
]

def eventLeaf15461 : Array AnnotatedEvent := #[
  { event := event247376
    frameStart := 247372 },
  { event := event247377
    frameStart := 247372 },
  { event := event247378
    frameStart := 247372 },
  { event := event247379
    frameStart := 247372 },
  { event := event247380
    frameStart := 247372 },
  { event := event247381
    frameStart := 247372 },
  { event := event247382
    frameStart := 247372 },
  { event := event247383
    frameStart := 247372 },
  { event := event247384
    frameStart := 247372 },
  { event := event247385
    frameStart := 247372 },
  { event := event247386
    frameStart := 247372 },
  { event := event247387
    frameStart := 247372 },
  { event := event247388
    frameStart := 247372 },
  { event := event247389
    frameStart := 247372 },
  { event := event247390
    frameStart := 247372 },
  { event := event247391
    frameStart := 247372 }
]

def eventLeaf15462 : Array AnnotatedEvent := #[
  { event := event247392
    frameStart := 247372 },
  { event := event247393
    frameStart := 247372 },
  { event := event247394
    frameStart := 247372 },
  { event := event247395
    frameStart := 247372 },
  { event := event247396
    frameStart := 247372 },
  { event := event247397
    frameStart := 247372 },
  { event := event247398
    frameStart := 247372 },
  { event := event247399
    frameStart := 247372 },
  { event := event247400
    frameStart := 247372 },
  { event := event247401
    frameStart := 247372 },
  { event := event247402
    frameStart := 247372 },
  { event := event247403
    frameStart := 247372 },
  { event := event247404
    frameStart := 247372 },
  { event := event247405
    frameStart := 247372 },
  { event := event247406
    frameStart := 247372 },
  { event := event247407
    frameStart := 247372 }
]

def eventLeaf15463 : Array AnnotatedEvent := #[
  { event := event247408
    frameStart := 247372 },
  { event := event247409
    frameStart := 247372 },
  { event := event247410
    frameStart := 247372 },
  { event := event247411
    frameStart := 247372 },
  { event := event247412
    frameStart := 247372 },
  { event := event247413
    frameStart := 247372 },
  { event := event247414
    frameStart := 247372 },
  { event := event247415
    frameStart := 247372 },
  { event := event247416
    frameStart := 247372 },
  { event := event247417
    frameStart := 247372 },
  { event := event247418
    frameStart := 247372 },
  { event := event247419
    frameStart := 247372 },
  { event := event247420
    frameStart := 247372 },
  { event := event247421
    frameStart := 247372 },
  { event := event247422
    frameStart := 247372 },
  { event := event247423
    frameStart := 247372 }
]

def eventLeaf15464 : Array AnnotatedEvent := #[
  { event := event247424
    frameStart := 247372 },
  { event := event247425
    frameStart := 247372 },
  { event := event247426
    frameStart := 247372 },
  { event := event247427
    frameStart := 247372 },
  { event := event247428
    frameStart := 247372 },
  { event := event247429
    frameStart := 247372 },
  { event := event247430
    frameStart := 247372 },
  { event := event247431
    frameStart := 247372 },
  { event := event247432
    frameStart := 247372 },
  { event := event247433
    frameStart := 247372 },
  { event := event247434
    frameStart := 247372 },
  { event := event247435
    frameStart := 247372 },
  { event := event247436
    frameStart := 247372 },
  { event := event247437
    frameStart := 247372 },
  { event := event247438
    frameStart := 247372 },
  { event := event247439
    frameStart := 247372 }
]

def eventLeaf15465 : Array AnnotatedEvent := #[
  { event := event247440
    frameStart := 247372 },
  { event := event247441
    frameStart := 247372 },
  { event := event247442
    frameStart := 247372 },
  { event := event247443
    frameStart := 247372 },
  { event := event247444
    frameStart := 247372 },
  { event := event247445
    frameStart := 247372 },
  { event := event247446
    frameStart := 247372 },
  { event := event247447
    frameStart := 247372 },
  { event := event247448
    frameStart := 247372 },
  { event := event247449
    frameStart := 247372 },
  { event := event247450
    frameStart := 247372 },
  { event := event247451
    frameStart := 247372 },
  { event := event247452
    frameStart := 247372 },
  { event := event247453
    frameStart := 247372 },
  { event := event247454
    frameStart := 247372 },
  { event := event247455
    frameStart := 247372 }
]

def eventLeaf15466 : Array AnnotatedEvent := #[
  { event := event247456
    frameStart := 247372 },
  { event := event247457
    frameStart := 247372 },
  { event := event247458
    frameStart := 247372 },
  { event := event247459
    frameStart := 247372 },
  { event := event247460
    frameStart := 247372 },
  { event := event247461
    frameStart := 247372 },
  { event := event247462
    frameStart := 247372 },
  { event := event247463
    frameStart := 247372 },
  { event := event247464
    frameStart := 247372 },
  { event := event247465
    frameStart := 247372 },
  { event := event247466
    frameStart := 247372 },
  { event := event247467
    frameStart := 247372 },
  { event := event247468
    frameStart := 247372 },
  { event := event247469
    frameStart := 247372 },
  { event := event247470
    frameStart := 247372 },
  { event := event247471
    frameStart := 247372 }
]

def eventLeaf15467 : Array AnnotatedEvent := #[
  { event := event247472
    frameStart := 247372 },
  { event := event247473
    frameStart := 247372 },
  { event := event247474
    frameStart := 247372 },
  { event := event247475
    frameStart := 247372 },
  { event := event247476
    frameStart := 0 },
  { event := event247477
    frameStart := 0 },
  { event := event247478
    frameStart := 0 },
  { event := event247479
    frameStart := 0 },
  { event := event247480
    frameStart := 0 },
  { event := event247481
    frameStart := 0 },
  { event := event247482
    frameStart := 0 },
  { event := event247483
    frameStart := 0 },
  { event := event247484
    frameStart := 0 },
  { event := event247485
    frameStart := 0 },
  { event := event247486
    frameStart := 0 },
  { event := event247487
    frameStart := 0 }
]

def eventLeaf15468 : Array AnnotatedEvent := #[
  { event := event247488
    frameStart := 0 },
  { event := event247489
    frameStart := 0 },
  { event := event247490
    frameStart := 0 },
  { event := event247491
    frameStart := 0 },
  { event := event247492
    frameStart := 0 },
  { event := event247493
    frameStart := 0 },
  { event := event247494
    frameStart := 0 },
  { event := event247495
    frameStart := 0 },
  { event := event247496
    frameStart := 0 },
  { event := event247497
    frameStart := 0 },
  { event := event247498
    frameStart := 0 },
  { event := event247499
    frameStart := 0 },
  { event := event247500
    frameStart := 0 },
  { event := event247501
    frameStart := 0 },
  { event := event247502
    frameStart := 0 },
  { event := event247503
    frameStart := 0 }
]

def eventLeaf15469 : Array AnnotatedEvent := #[
  { event := event247504
    frameStart := 0 },
  { event := event247505
    frameStart := 0 },
  { event := event247506
    frameStart := 0 },
  { event := event247507
    frameStart := 0 },
  { event := event247508
    frameStart := 0 },
  { event := event247509
    frameStart := 0 },
  { event := event247510
    frameStart := 0 },
  { event := event247511
    frameStart := 0 },
  { event := event247512
    frameStart := 0 },
  { event := event247513
    frameStart := 0 },
  { event := event247514
    frameStart := 0 },
  { event := event247515
    frameStart := 0 },
  { event := event247516
    frameStart := 0 },
  { event := event247517
    frameStart := 0 },
  { event := event247518
    frameStart := 0 },
  { event := event247519
    frameStart := 0 }
]

def eventLeaf15470 : Array AnnotatedEvent := #[
  { event := event247520
    frameStart := 0 },
  { event := event247521
    frameStart := 0 },
  { event := event247522
    frameStart := 0 },
  { event := event247523
    frameStart := 0 },
  { event := event247524
    frameStart := 0 },
  { event := event247525
    frameStart := 0 },
  { event := event247526
    frameStart := 0 },
  { event := event247527
    frameStart := 0 },
  { event := event247528
    frameStart := 0 },
  { event := event247529
    frameStart := 0 },
  { event := event247530
    frameStart := 247530 },
  { event := event247531
    frameStart := 247530 },
  { event := event247532
    frameStart := 247530 },
  { event := event247533
    frameStart := 247530 },
  { event := event247534
    frameStart := 247530 },
  { event := event247535
    frameStart := 247530 }
]

def eventLeaf15471 : Array AnnotatedEvent := #[
  { event := event247536
    frameStart := 247530 },
  { event := event247537
    frameStart := 247530 },
  { event := event247538
    frameStart := 247530 },
  { event := event247539
    frameStart := 247530 },
  { event := event247540
    frameStart := 247530 },
  { event := event247541
    frameStart := 247530 },
  { event := event247542
    frameStart := 247530 },
  { event := event247543
    frameStart := 247530 },
  { event := event247544
    frameStart := 247530 },
  { event := event247545
    frameStart := 247530 },
  { event := event247546
    frameStart := 247530 },
  { event := event247547
    frameStart := 247530 },
  { event := event247548
    frameStart := 247530 },
  { event := event247549
    frameStart := 247530 },
  { event := event247550
    frameStart := 247530 },
  { event := event247551
    frameStart := 247530 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events966
