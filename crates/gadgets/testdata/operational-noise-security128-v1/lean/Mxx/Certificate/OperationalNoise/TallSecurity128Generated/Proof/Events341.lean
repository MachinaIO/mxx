import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events341

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event87296 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event87297 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event87298 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event87299 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event87300 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 87299

def event87301 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 87297

def event87302 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 87300 .coefficient) (.value (.predecessor 1 87301 .coefficient)))

def event87303 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event87304 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10269⟩⟩) 0 ⟨392⟩ 87303

def event87305 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10269⟩⟩) 1 ⟨10267⟩ 87295

def event87306 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10269⟩⟩) (.sum [.predecessor 0 87304 .coefficient, .predecessor 1 87305 .coefficient])

def event87307 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10269⟩⟩) (.finite 655355)

def event87308 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10325⟩⟩) 0 ⟨10269⟩ 87307

def event87309 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10325⟩⟩) 1 ⟨5426⟩ 87293

def event87310 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10325⟩⟩) (.identity (.predecessor 1 87309 .coefficient))

def event87311 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10325⟩⟩) (.finite 655360)

def event87312 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34578⟩⟩) 0 ⟨10325⟩ 87311

def event87313 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34578⟩⟩) (.authority (.programFamilyFact))

def exact87314RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨34578⟩⟩], []⟩, (1)⟩]

theorem exact87314RawTermsValid :
    exact87314RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87314 : Event := .resultExact (⟨.program ⟨257⟩, ⟨34578⟩⟩) exact87314RawTerms (.finite 40) 87313 .exactZero (none)

def event87315 : Event := .predecessor (⟨.program ⟨257⟩, ⟨13671⟩⟩) 0 ⟨10325⟩ 87311

def event87316 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨13671⟩⟩) (.authority (.programFamilyFact))

def exact87317RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨13671⟩⟩], []⟩, (1)⟩]

theorem exact87317RawTermsValid :
    exact87317RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87317 : Event := .resultExact (⟨.program ⟨257⟩, ⟨13671⟩⟩) exact87317RawTerms (.finite 40) 87316 .exactZero (none)

def event87318 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34579⟩⟩) 0 ⟨13671⟩ 87317

def event87319 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34579⟩⟩) 1 ⟨34578⟩ 87314

def event87320 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34579⟩⟩) (.product (.predecessor 0 87318 .coefficient) (.predecessor 1 87319 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event87321 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34579⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨13671⟩⟩, ⟨.program ⟨257⟩, ⟨34578⟩⟩], []⟩) [⟨.result 87317 .coefficient, true, some 1⟩, ⟨.result 87314 .coefficient, true, some 1⟩])

def event87322 : Event := .survivorFold (1) 87321

def exact87323RawTerms : List Term := []

theorem exact87323RawTermsValid :
    exact87323RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87323 : Event := .resultExact (⟨.program ⟨257⟩, ⟨34579⟩⟩) exact87323RawTerms (.finite 1600) 87320 (.finite 1600) (some (87321))

def event87324 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34580⟩⟩) 0 ⟨34579⟩ 87323

def event87325 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34580⟩⟩) (.identity (.predecessor 0 87324 .coefficient))

def event87326 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨34580⟩⟩) (.finite 1600)

def event87327 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34796⟩⟩) 0 ⟨34580⟩ 87326

def event87328 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34796⟩⟩) (.authority (.programFamilyFact))

def exact87329RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨34796⟩⟩], []⟩, (1)⟩]

theorem exact87329RawTermsValid :
    exact87329RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87329 : Event := .resultExact (⟨.program ⟨257⟩, ⟨34796⟩⟩) exact87329RawTerms (.finite 40) 87328 .exactZero (none)

def event87330 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34797⟩⟩) 0 ⟨34796⟩ 87329

def event87331 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34797⟩⟩) (.identity (.predecessor 0 87330 .coefficient))

def event87332 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨34797⟩⟩) (.finite 40)

def event87333 : Event := .predecessor (⟨.program ⟨257⟩, ⟨35612⟩⟩) 0 ⟨34797⟩ 87332

def event87334 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35612⟩⟩) (.authority (.relationPreimageSource ⟨82⟩))

def exact87335RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35612⟩⟩]⟩, (1)⟩]

theorem exact87335RawTermsValid :
    exact87335RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87335 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35612⟩⟩) exact87335RawTerms (.finite 5647228698) 87334 .exactZero (none)

def event87336 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35⟩⟩) (.authority (.operator))

def exact87337RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact87337RawTermsValid :
    exact87337RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87337 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35⟩⟩) exact87337RawTerms .large 87336 .exactZero (none)

def event87338 : Event := .predecessor (⟨.program ⟨257⟩, ⟨35613⟩⟩) 0 ⟨35⟩ 87337

def event87339 : Event := .predecessor (⟨.program ⟨257⟩, ⟨35613⟩⟩) 1 ⟨35612⟩ 87335

def event87340 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35613⟩⟩) (.product (.predecessor 0 87338 .coefficient) (.predecessor 1 87339 .coefficient) (⟨false, false, none, none, none⟩))

def event87341 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨35613⟩⟩, .operator (⟨87337, 0⟩, ⟨87335, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35612⟩⟩]⟩, (1)⟩)

def exact87342RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35612⟩⟩]⟩, (1)⟩]

theorem exact87342RawTermsValid :
    exact87342RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87342 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35613⟩⟩) exact87342RawTerms .large 87340 .exactZero (none)

def event87343 : Event := .preFoldPolynomial 87342 [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35612⟩⟩]⟩, (1)⟩] .exactZero none

def exact87344RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35612⟩⟩]⟩, (1)⟩]

def event87344 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨35613⟩⟩) 87343 exact87344RawTerms .large 87340 .exactZero (none)

def event87345 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨36778⟩⟩)

def event87346 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event87347 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event87348 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10267⟩⟩) (.authority (.operator))

def event87349 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10267⟩⟩) (.finite 15)

def event87350 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event87351 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event87352 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event87353 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event87354 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 87353

def event87355 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 87351

def event87356 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 87354 .coefficient) (.value (.predecessor 1 87355 .coefficient)))

def event87357 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event87358 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10269⟩⟩) 0 ⟨392⟩ 87357

def event87359 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10269⟩⟩) 1 ⟨10267⟩ 87349

def event87360 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10269⟩⟩) (.sum [.predecessor 0 87358 .coefficient, .predecessor 1 87359 .coefficient])

def event87361 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10269⟩⟩) (.finite 655355)

def event87362 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10325⟩⟩) 0 ⟨10269⟩ 87361

def event87363 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10325⟩⟩) 1 ⟨5426⟩ 87347

def event87364 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10325⟩⟩) (.identity (.predecessor 1 87363 .coefficient))

def event87365 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10325⟩⟩) (.finite 655360)

def event87366 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34578⟩⟩) 0 ⟨10325⟩ 87365

def event87367 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34578⟩⟩) (.authority (.programFamilyFact))

def exact87368RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨34578⟩⟩], []⟩, (1)⟩]

theorem exact87368RawTermsValid :
    exact87368RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87368 : Event := .resultExact (⟨.program ⟨257⟩, ⟨34578⟩⟩) exact87368RawTerms (.finite 40) 87367 .exactZero (none)

def event87369 : Event := .predecessor (⟨.program ⟨257⟩, ⟨13671⟩⟩) 0 ⟨10325⟩ 87365

def event87370 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨13671⟩⟩) (.authority (.programFamilyFact))

def exact87371RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨13671⟩⟩], []⟩, (1)⟩]

theorem exact87371RawTermsValid :
    exact87371RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87371 : Event := .resultExact (⟨.program ⟨257⟩, ⟨13671⟩⟩) exact87371RawTerms (.finite 40) 87370 .exactZero (none)

def event87372 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34579⟩⟩) 0 ⟨13671⟩ 87371

def event87373 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34579⟩⟩) 1 ⟨34578⟩ 87368

def event87374 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34579⟩⟩) (.product (.predecessor 0 87372 .coefficient) (.predecessor 1 87373 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event87375 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨34579⟩⟩, .operator (⟨87371, 0⟩, ⟨87368, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨13671⟩⟩, ⟨.program ⟨257⟩, ⟨34578⟩⟩], []⟩, (1)⟩)

def exact87376RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨13671⟩⟩, ⟨.program ⟨257⟩, ⟨34578⟩⟩], []⟩, (1)⟩]

theorem exact87376RawTermsValid :
    exact87376RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87376 : Event := .resultExact (⟨.program ⟨257⟩, ⟨34579⟩⟩) exact87376RawTerms (.finite 1600) 87374 .exactZero (none)

def event87377 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34580⟩⟩) 0 ⟨34579⟩ 87376

def event87378 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34580⟩⟩) (.identity (.predecessor 0 87377 .coefficient))

def event87379 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨34580⟩⟩) (.finite 1600)

def event87380 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34796⟩⟩) 0 ⟨34580⟩ 87379

def event87381 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34796⟩⟩) (.authority (.programFamilyFact))

def exact87382RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨34796⟩⟩], []⟩, (1)⟩]

theorem exact87382RawTermsValid :
    exact87382RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87382 : Event := .resultExact (⟨.program ⟨257⟩, ⟨34796⟩⟩) exact87382RawTerms (.finite 40) 87381 .exactZero (none)

def event87383 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34797⟩⟩) 0 ⟨34796⟩ 87382

def event87384 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34797⟩⟩) (.identity (.predecessor 0 87383 .coefficient))

def event87385 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨34797⟩⟩) (.finite 40)

def event87386 : Event := .predecessor (⟨.program ⟨257⟩, ⟨35953⟩⟩) 0 ⟨34797⟩ 87385

def event87387 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35953⟩⟩) (.authority (.programFamilyFact))

def event87388 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨35953⟩⟩) (.finite 3720)

def event87389 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨7177⟩⟩) .missing

def event87390 : Event := .predecessor (⟨.program ⟨257⟩, ⟨35954⟩⟩) 0 ⟨7177⟩ 87389

def event87391 : Event := .predecessor (⟨.program ⟨257⟩, ⟨35954⟩⟩) 1 ⟨35953⟩ 87388

def event87392 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35954⟩⟩) (.authority (.operator))

def exact87393RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35954⟩⟩]⟩, (1)⟩]

theorem exact87393RawTermsValid :
    exact87393RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87393 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35954⟩⟩) exact87393RawTerms .large 87392 .exactZero (none)

def event87394 : Event := .predecessor (⟨.program ⟨257⟩, ⟨36773⟩⟩) 0 ⟨35954⟩ 87393

def event87395 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨36773⟩⟩) (.authority (.operator))

def exact87396RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨36773⟩⟩]⟩, (1)⟩]

theorem exact87396RawTermsValid :
    exact87396RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87396 : Event := .resultExact (⟨.program ⟨257⟩, ⟨36773⟩⟩) exact87396RawTerms (.finite 8192) 87395 .exactZero (none)

def event87397 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event87398 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event87399 : Event := .predecessor (⟨.program ⟨257⟩, ⟨36130⟩⟩) 0 ⟨34797⟩ 87385

def event87400 : Event := .predecessor (⟨.program ⟨257⟩, ⟨36130⟩⟩) 1 ⟨136⟩ 87398

def event87401 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨36130⟩⟩) (.sum [.predecessor 0 87399 .coefficient, .predecessor 1 87400 .coefficient])

def event87402 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨36130⟩⟩) (.finite 40)

def event87403 : Event := .predecessor (⟨.program ⟨257⟩, ⟨36131⟩⟩) 0 ⟨36130⟩ 87402

def event87404 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨36131⟩⟩) (.identity (.predecessor 0 87403 .coefficient))

def exact87405RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨34796⟩⟩], []⟩, (1)⟩]

theorem exact87405RawTermsValid :
    exact87405RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87405 : Event := .resultExact (⟨.program ⟨257⟩, ⟨36131⟩⟩) exact87405RawTerms (.finite 40) 87404 .exactZero (none)

def event87406 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact87407RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact87407RawTermsValid :
    exact87407RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87407 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact87407RawTerms .large 87406 .exactZero (none)

def event87408 : Event := .predecessor (⟨.program ⟨257⟩, ⟨36132⟩⟩) 0 ⟨6908⟩ 87407

def event87409 : Event := .predecessor (⟨.program ⟨257⟩, ⟨36132⟩⟩) 1 ⟨36131⟩ 87405

def event87410 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨36132⟩⟩) (.product (.predecessor 0 87408 .coefficient) (.predecessor 1 87409 .coefficient) (⟨false, false, none, none, none⟩))

def event87411 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨36132⟩⟩, .operator (⟨87407, 0⟩, ⟨87405, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨34796⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact87412RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨34796⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact87412RawTermsValid :
    exact87412RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87412 : Event := .resultExact (⟨.program ⟨257⟩, ⟨36132⟩⟩) exact87412RawTerms .large 87410 .exactZero (none)

def event87413 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7191⟩⟩) 0 ⟨7177⟩ 87389

def event87414 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7191⟩⟩) (.authority (.operator))

def exact87415RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7191⟩⟩]⟩, (1)⟩]

theorem exact87415RawTermsValid :
    exact87415RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87415 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7191⟩⟩) exact87415RawTerms .large 87414 .exactZero (none)

def event87416 : Event := .predecessor (⟨.program ⟨257⟩, ⟨36133⟩⟩) 0 ⟨7191⟩ 87415

def event87417 : Event := .predecessor (⟨.program ⟨257⟩, ⟨36133⟩⟩) 1 ⟨36132⟩ 87412

def event87418 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨36133⟩⟩) (.sum [.predecessor 0 87416 .coefficient, .predecessor 1 87417 .coefficient])

def exact87419RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7191⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨34796⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact87419RawTermsValid :
    exact87419RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87419 : Event := .resultExact (⟨.program ⟨257⟩, ⟨36133⟩⟩) exact87419RawTerms .large 87418 .exactZero (none)

def event87420 : Event := .predecessor (⟨.program ⟨257⟩, ⟨36774⟩⟩) 0 ⟨36133⟩ 87419

def event87421 : Event := .predecessor (⟨.program ⟨257⟩, ⟨36774⟩⟩) 1 ⟨36773⟩ 87396

def event87422 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨36774⟩⟩) (.product (.predecessor 0 87420 .coefficient) (.predecessor 1 87421 .coefficient) (⟨false, false, none, none, none⟩))

def event87423 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨36774⟩⟩, .operator (⟨87419, 0⟩, ⟨87396, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7191⟩⟩, ⟨.program ⟨257⟩, ⟨36773⟩⟩]⟩, (1)⟩)

def event87424 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨36774⟩⟩, .operator (⟨87419, 1⟩, ⟨87396, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨34796⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36773⟩⟩]⟩, (-1)⟩)

def event87425 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨36774⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨34796⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36773⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨36773⟩⟩) ⟨35954⟩ 87393)

def event87426 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨36774⟩⟩, .relation 87425 0, ⟨[⟨.program ⟨257⟩, ⟨34796⟩⟩], [⟨.program ⟨257⟩, ⟨35954⟩⟩]⟩, (-1)⟩)

def exact87427RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7191⟩⟩, ⟨.program ⟨257⟩, ⟨36773⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨34796⟩⟩], [⟨.program ⟨257⟩, ⟨35954⟩⟩]⟩, (-1)⟩]

theorem exact87427RawTermsValid :
    exact87427RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87427 : Event := .resultExact (⟨.program ⟨257⟩, ⟨36774⟩⟩) exact87427RawTerms .large 87422 .exactZero (none)

def event87428 : Event := .predecessor (⟨.program ⟨257⟩, ⟨35037⟩⟩) 0 ⟨34797⟩ 87385

def event87429 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35037⟩⟩) (.authority (.programFamilyFact))

def exact87430RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨35037⟩⟩], []⟩, (1)⟩]

theorem exact87430RawTermsValid :
    exact87430RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87430 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35037⟩⟩) exact87430RawTerms (.finite 40) 87429 .exactZero (none)

def event87431 : Event := .predecessor (⟨.program ⟨257⟩, ⟨35039⟩⟩) 0 ⟨6908⟩ 87407

def event87432 : Event := .predecessor (⟨.program ⟨257⟩, ⟨35039⟩⟩) 1 ⟨35037⟩ 87430

def event87433 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35039⟩⟩) (.product (.predecessor 0 87431 .coefficient) (.predecessor 1 87432 .coefficient) (⟨false, true, none, none, some 1⟩))

def event87434 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨35039⟩⟩, .operator (⟨87407, 0⟩, ⟨87430, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨35037⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact87435RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨35037⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact87435RawTermsValid :
    exact87435RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87435 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35039⟩⟩) exact87435RawTerms .large 87433 .exactZero (none)

def event87436 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7221⟩⟩) 0 ⟨7177⟩ 87389

def event87437 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7221⟩⟩) (.authority (.operator))

def exact87438RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7221⟩⟩]⟩, (1)⟩]

theorem exact87438RawTermsValid :
    exact87438RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87438 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7221⟩⟩) exact87438RawTerms .large 87437 .exactZero (none)

def event87439 : Event := .predecessor (⟨.program ⟨257⟩, ⟨35040⟩⟩) 0 ⟨7221⟩ 87438

def event87440 : Event := .predecessor (⟨.program ⟨257⟩, ⟨35040⟩⟩) 1 ⟨35039⟩ 87435

def event87441 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35040⟩⟩) (.sum [.predecessor 0 87439 .coefficient, .predecessor 1 87440 .coefficient])

def exact87442RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7221⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨35037⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact87442RawTermsValid :
    exact87442RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87442 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35040⟩⟩) exact87442RawTerms .large 87441 .exactZero (none)

def event87443 : Event := .predecessor (⟨.program ⟨257⟩, ⟨36778⟩⟩) 0 ⟨35040⟩ 87442

def event87444 : Event := .predecessor (⟨.program ⟨257⟩, ⟨36778⟩⟩) 1 ⟨36774⟩ 87427

def event87445 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨36778⟩⟩) (.sum [.predecessor 0 87443 .coefficient, .predecessor 1 87444 .coefficient])

def exact87446RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7191⟩⟩, ⟨.program ⟨257⟩, ⟨36773⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7221⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨34796⟩⟩], [⟨.program ⟨257⟩, ⟨35954⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨35037⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact87446RawTermsValid :
    exact87446RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87446 : Event := .resultExact (⟨.program ⟨257⟩, ⟨36778⟩⟩) exact87446RawTerms .large 87445 .exactZero (none)

def event87447 : Event := .preFoldPolynomial 87446 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7191⟩⟩, ⟨.program ⟨257⟩, ⟨36773⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7221⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨34796⟩⟩], [⟨.program ⟨257⟩, ⟨35954⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨35037⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact87448RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7191⟩⟩, ⟨.program ⟨257⟩, ⟨36773⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7221⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨34796⟩⟩], [⟨.program ⟨257⟩, ⟨35954⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨35037⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event87448 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨36778⟩⟩) 87447 exact87448RawTerms .large 87445 .exactZero (none)

def event87449 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨34797⟩⟩) ⟨⟨100⟩, ⟨82⟩, ⟨135⟩⟩ ⟨87291, 87449⟩

def event87450 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨35615⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35612⟩⟩]⟩) (1) 0 2 (.universal 87449 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35612⟩⟩]⟩) (none) 87448)

def event87451 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨35615⟩⟩, .relation 87450 1, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7221⟩⟩]⟩, (1)⟩)

def event87452 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨35615⟩⟩, .relation 87450 0, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7191⟩⟩, ⟨.program ⟨257⟩, ⟨36773⟩⟩]⟩, (-1)⟩)

def event87453 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨35615⟩⟩, .relation 87450 2, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨34796⟩⟩], [⟨.program ⟨257⟩, ⟨35954⟩⟩]⟩, (1)⟩)

def event87454 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨35615⟩⟩, .relation 87450 3, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨35037⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact87455RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7191⟩⟩, ⟨.program ⟨257⟩, ⟨36773⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7221⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨34796⟩⟩], [⟨.program ⟨257⟩, ⟨35954⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨35037⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact87455RawTermsValid :
    exact87455RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87455 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35615⟩⟩) exact87455RawTerms .large 87287 (.finite 202072841853861888) (some (87289))

def event87456 : Event := .predecessor (⟨.program ⟨257⟩, ⟨36776⟩⟩) 0 ⟨35615⟩ 87455

def event87457 : Event := .predecessor (⟨.program ⟨257⟩, ⟨36776⟩⟩) 1 ⟨36775⟩ 87277

def event87458 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨36776⟩⟩) (.sum [.predecessor 0 87456 .coefficient, .predecessor 1 87457 .coefficient])

def event87459 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨36776⟩⟩, .operator (⟨87455, 0⟩, ⟨87277, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7191⟩⟩, ⟨.program ⟨257⟩, ⟨36773⟩⟩]⟩, (1)⟩)

def event87460 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨36776⟩⟩, .operator (⟨87455, 2⟩, ⟨87277, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨34796⟩⟩], [⟨.program ⟨257⟩, ⟨35954⟩⟩]⟩, (-1)⟩)

def event87461 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨36776⟩⟩) (.sum [.result 87455 .summary, .result 87277 .summary])

def exact87462RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7221⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨35037⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact87462RawTermsValid :
    exact87462RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87462 : Event := .resultExact (⟨.program ⟨257⟩, ⟨36776⟩⟩) exact87462RawTerms .large 87458 (.finite 32192539770951767057087530795008) (some (87461))

def event87463 : Event := .predecessor (⟨.program ⟨257⟩, ⟨36777⟩⟩) 0 ⟨36776⟩ 87462

def event87464 : Event := .predecessor (⟨.program ⟨257⟩, ⟨36777⟩⟩) 1 ⟨7164⟩ 15642

def event87465 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨36777⟩⟩) (.product (.predecessor 0 87463 .coefficient) (.predecessor 1 87464 .coefficient) (⟨false, false, none, none, none⟩))

def event87466 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨36777⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨7163⟩⟩]⟩) [⟨.result 15638 .coefficient, false, none⟩])

def event87467 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨36777⟩⟩) (.product (.result 87462 .summary) (.transfer 87466) (⟨false, false, none, none, none⟩))

def event87468 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨36777⟩⟩, .operator (⟨87462, 0⟩, ⟨15642, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7221⟩⟩, ⟨.program ⟨257⟩, ⟨7163⟩⟩]⟩, (1)⟩)

def event87469 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨36777⟩⟩, .operator (⟨87462, 1⟩, ⟨15642, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨35037⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7163⟩⟩]⟩, (-1)⟩)

def event87470 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨36777⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨35037⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7163⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨7163⟩⟩) ⟨7047⟩ 15635)

def event87471 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨36777⟩⟩, .relation 87470 0, ⟨[⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨35037⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact87472RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨35037⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7221⟩⟩, ⟨.program ⟨257⟩, ⟨7163⟩⟩]⟩, (1)⟩]

theorem exact87472RawTermsValid :
    exact87472RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87472 : Event := .resultExact (⟨.program ⟨257⟩, ⟨36777⟩⟩) exact87472RawTerms .large 87465 (.finite 345664763728542925759002774434880600145920) (some (87467))

def event87473 : Event := .predecessor (⟨.program ⟨257⟩, ⟨30294⟩⟩) 0 ⟨7177⟩ 15500

def event87474 : Event := .predecessor (⟨.program ⟨257⟩, ⟨30294⟩⟩) 1 ⟨30293⟩ 78789

def event87475 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨30294⟩⟩) (.authority (.operator))

def exact87476RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨30294⟩⟩]⟩, (1)⟩]

theorem exact87476RawTermsValid :
    exact87476RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87476 : Event := .resultExact (⟨.program ⟨257⟩, ⟨30294⟩⟩) exact87476RawTerms .large 87475 .exactZero (none)

def event87477 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31113⟩⟩) 0 ⟨30294⟩ 87476

def event87478 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31113⟩⟩) (.authority (.operator))

def exact87479RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨31113⟩⟩]⟩, (1)⟩]

theorem exact87479RawTermsValid :
    exact87479RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87479 : Event := .resultExact (⟨.program ⟨257⟩, ⟨31113⟩⟩) exact87479RawTerms (.finite 8192) 87478 .exactZero (none)

def event87480 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31115⟩⟩) 0 ⟨30667⟩ 79073

def event87481 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31115⟩⟩) 1 ⟨31113⟩ 87479

def event87482 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31115⟩⟩) (.product (.predecessor 0 87480 .coefficient) (.predecessor 1 87481 .coefficient) (⟨false, false, none, none, none⟩))

def event87483 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31115⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨31113⟩⟩]⟩) [⟨.result 87479 .coefficient, false, none⟩])

def event87484 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31115⟩⟩) (.product (.result 79073 .summary) (.transfer 87483) (⟨false, false, none, none, none⟩))

def event87485 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨31115⟩⟩, .operator (⟨79073, 0⟩, ⟨87479, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨31113⟩⟩]⟩, (1)⟩)

def event87486 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨31115⟩⟩, .operator (⟨79073, 1⟩, ⟨87479, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨29136⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨31113⟩⟩]⟩, (-1)⟩)

def event87487 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨31115⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨29136⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨31113⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨31113⟩⟩) ⟨30294⟩ 87476)

def event87488 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨31115⟩⟩, .relation 87487 0, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨29136⟩⟩], [⟨.program ⟨257⟩, ⟨30294⟩⟩]⟩, (-1)⟩)

def exact87489RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨31113⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨29136⟩⟩], [⟨.program ⟨257⟩, ⟨30294⟩⟩]⟩, (-1)⟩]

theorem exact87489RawTermsValid :
    exact87489RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87489 : Event := .resultExact (⟨.program ⟨257⟩, ⟨31115⟩⟩) exact87489RawTerms .large 87482 (.finite 32192146870060190229763897425920) (some (87484))

def event87490 : Event := .predecessor (⟨.program ⟨257⟩, ⟨29952⟩⟩) 0 ⟨29137⟩ 3241

def event87491 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨29952⟩⟩) (.authority (.relationPreimageSource ⟨80⟩))

def exact87492RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨29952⟩⟩]⟩, (1)⟩]

theorem exact87492RawTermsValid :
    exact87492RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87492 : Event := .resultExact (⟨.program ⟨257⟩, ⟨29952⟩⟩) exact87492RawTerms (.finite 5647228698) 87491 .exactZero (none)

def event87493 : Event := .predecessor (⟨.program ⟨257⟩, ⟨29954⟩⟩) 0 ⟨29952⟩ 87492

def event87494 : Event := .predecessor (⟨.program ⟨257⟩, ⟨29954⟩⟩) 1 ⟨2370⟩ 4

def event87495 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨29954⟩⟩) (.scale (.predecessor 0 87493 .coefficient) (.value (.predecessor 1 87494 .coefficient)))

def exact87496RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨29952⟩⟩]⟩, (1)⟩]

theorem exact87496RawTermsValid :
    exact87496RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87496 : Event := .resultExact (⟨.program ⟨257⟩, ⟨29954⟩⟩) exact87496RawTerms (.finite 5647228698) 87495 .exactZero (none)

def event87497 : Event := .predecessor (⟨.program ⟨257⟩, ⟨29955⟩⟩) 0 ⟨10368⟩ 75995

def event87498 : Event := .predecessor (⟨.program ⟨257⟩, ⟨29955⟩⟩) 1 ⟨29954⟩ 87496

def event87499 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨29955⟩⟩) (.product (.predecessor 0 87497 .coefficient) (.predecessor 1 87498 .coefficient) (⟨false, false, none, none, none⟩))

def event87500 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨29955⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨29952⟩⟩]⟩) [⟨.result 87492 .coefficient, false, none⟩])

def event87501 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨29955⟩⟩) (.product (.result 75995 .summary) (.transfer 87500) (⟨false, false, none, none, none⟩))

def event87502 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨29955⟩⟩, .operator (⟨75995, 0⟩, ⟨87496, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29952⟩⟩]⟩, (1)⟩)

def event87503 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨29953⟩⟩)

def event87504 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event87505 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event87506 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10267⟩⟩) (.authority (.operator))

def event87507 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10267⟩⟩) (.finite 15)

def event87508 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event87509 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event87510 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event87511 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event87512 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 87511

def event87513 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 87509

def event87514 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 87512 .coefficient) (.value (.predecessor 1 87513 .coefficient)))

def event87515 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event87516 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10269⟩⟩) 0 ⟨392⟩ 87515

def event87517 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10269⟩⟩) 1 ⟨10267⟩ 87507

def event87518 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10269⟩⟩) (.sum [.predecessor 0 87516 .coefficient, .predecessor 1 87517 .coefficient])

def event87519 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10269⟩⟩) (.finite 655355)

def event87520 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10325⟩⟩) 0 ⟨10269⟩ 87519

def event87521 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10325⟩⟩) 1 ⟨5426⟩ 87505

def event87522 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10325⟩⟩) (.identity (.predecessor 1 87521 .coefficient))

def event87523 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10325⟩⟩) (.finite 655360)

def event87524 : Event := .predecessor (⟨.program ⟨257⟩, ⟨28918⟩⟩) 0 ⟨10325⟩ 87523

def event87525 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨28918⟩⟩) (.authority (.programFamilyFact))

def exact87526RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨28918⟩⟩], []⟩, (1)⟩]

theorem exact87526RawTermsValid :
    exact87526RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87526 : Event := .resultExact (⟨.program ⟨257⟩, ⟨28918⟩⟩) exact87526RawTerms (.finite 36) 87525 .exactZero (none)

def event87527 : Event := .predecessor (⟨.program ⟨257⟩, ⟨13371⟩⟩) 0 ⟨10325⟩ 87523

def event87528 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨13371⟩⟩) (.authority (.programFamilyFact))

def exact87529RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨13371⟩⟩], []⟩, (1)⟩]

theorem exact87529RawTermsValid :
    exact87529RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87529 : Event := .resultExact (⟨.program ⟨257⟩, ⟨13371⟩⟩) exact87529RawTerms (.finite 36) 87528 .exactZero (none)

def event87530 : Event := .predecessor (⟨.program ⟨257⟩, ⟨28919⟩⟩) 0 ⟨13371⟩ 87529

def event87531 : Event := .predecessor (⟨.program ⟨257⟩, ⟨28919⟩⟩) 1 ⟨28918⟩ 87526

def event87532 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨28919⟩⟩) (.product (.predecessor 0 87530 .coefficient) (.predecessor 1 87531 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event87533 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨28919⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨13371⟩⟩, ⟨.program ⟨257⟩, ⟨28918⟩⟩], []⟩) [⟨.result 87529 .coefficient, true, some 1⟩, ⟨.result 87526 .coefficient, true, some 1⟩])

def event87534 : Event := .survivorFold (1) 87533

def exact87535RawTerms : List Term := []

theorem exact87535RawTermsValid :
    exact87535RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87535 : Event := .resultExact (⟨.program ⟨257⟩, ⟨28919⟩⟩) exact87535RawTerms (.finite 1296) 87532 (.finite 1296) (some (87533))

def event87536 : Event := .predecessor (⟨.program ⟨257⟩, ⟨28920⟩⟩) 0 ⟨28919⟩ 87535

def event87537 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨28920⟩⟩) (.identity (.predecessor 0 87536 .coefficient))

def event87538 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨28920⟩⟩) (.finite 1296)

def event87539 : Event := .predecessor (⟨.program ⟨257⟩, ⟨29136⟩⟩) 0 ⟨28920⟩ 87538

def event87540 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨29136⟩⟩) (.authority (.programFamilyFact))

def exact87541RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨29136⟩⟩], []⟩, (1)⟩]

theorem exact87541RawTermsValid :
    exact87541RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87541 : Event := .resultExact (⟨.program ⟨257⟩, ⟨29136⟩⟩) exact87541RawTerms (.finite 36) 87540 .exactZero (none)

def event87542 : Event := .predecessor (⟨.program ⟨257⟩, ⟨29137⟩⟩) 0 ⟨29136⟩ 87541

def event87543 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨29137⟩⟩) (.identity (.predecessor 0 87542 .coefficient))

def event87544 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨29137⟩⟩) (.finite 36)

def event87545 : Event := .predecessor (⟨.program ⟨257⟩, ⟨29952⟩⟩) 0 ⟨29137⟩ 87544

def event87546 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨29952⟩⟩) (.authority (.relationPreimageSource ⟨80⟩))

def exact87547RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨29952⟩⟩]⟩, (1)⟩]

theorem exact87547RawTermsValid :
    exact87547RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87547 : Event := .resultExact (⟨.program ⟨257⟩, ⟨29952⟩⟩) exact87547RawTerms (.finite 5647228698) 87546 .exactZero (none)

def event87548 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35⟩⟩) (.authority (.operator))

def exact87549RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact87549RawTermsValid :
    exact87549RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87549 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35⟩⟩) exact87549RawTerms .large 87548 .exactZero (none)

def event87550 : Event := .predecessor (⟨.program ⟨257⟩, ⟨29953⟩⟩) 0 ⟨35⟩ 87549

def event87551 : Event := .predecessor (⟨.program ⟨257⟩, ⟨29953⟩⟩) 1 ⟨29952⟩ 87547

def eventLeaf5456 : Array AnnotatedEvent := #[
  { event := event87296
    frameStart := 87291 },
  { event := event87297
    frameStart := 87291 },
  { event := event87298
    frameStart := 87291 },
  { event := event87299
    frameStart := 87291 },
  { event := event87300
    frameStart := 87291 },
  { event := event87301
    frameStart := 87291 },
  { event := event87302
    frameStart := 87291 },
  { event := event87303
    frameStart := 87291 },
  { event := event87304
    frameStart := 87291 },
  { event := event87305
    frameStart := 87291 },
  { event := event87306
    frameStart := 87291 },
  { event := event87307
    frameStart := 87291 },
  { event := event87308
    frameStart := 87291 },
  { event := event87309
    frameStart := 87291 },
  { event := event87310
    frameStart := 87291 },
  { event := event87311
    frameStart := 87291 }
]

def eventLeaf5457 : Array AnnotatedEvent := #[
  { event := event87312
    frameStart := 87291 },
  { event := event87313
    frameStart := 87291 },
  { event := event87314
    frameStart := 87291 },
  { event := event87315
    frameStart := 87291 },
  { event := event87316
    frameStart := 87291 },
  { event := event87317
    frameStart := 87291 },
  { event := event87318
    frameStart := 87291 },
  { event := event87319
    frameStart := 87291 },
  { event := event87320
    frameStart := 87291 },
  { event := event87321
    frameStart := 87291 },
  { event := event87322
    frameStart := 87291 },
  { event := event87323
    frameStart := 87291 },
  { event := event87324
    frameStart := 87291 },
  { event := event87325
    frameStart := 87291 },
  { event := event87326
    frameStart := 87291 },
  { event := event87327
    frameStart := 87291 }
]

def eventLeaf5458 : Array AnnotatedEvent := #[
  { event := event87328
    frameStart := 87291 },
  { event := event87329
    frameStart := 87291 },
  { event := event87330
    frameStart := 87291 },
  { event := event87331
    frameStart := 87291 },
  { event := event87332
    frameStart := 87291 },
  { event := event87333
    frameStart := 87291 },
  { event := event87334
    frameStart := 87291 },
  { event := event87335
    frameStart := 87291 },
  { event := event87336
    frameStart := 87291 },
  { event := event87337
    frameStart := 87291 },
  { event := event87338
    frameStart := 87291 },
  { event := event87339
    frameStart := 87291 },
  { event := event87340
    frameStart := 87291 },
  { event := event87341
    frameStart := 87291 },
  { event := event87342
    frameStart := 87291 },
  { event := event87343
    frameStart := 87291 }
]

def eventLeaf5459 : Array AnnotatedEvent := #[
  { event := event87344
    frameStart := 87291 },
  { event := event87345
    frameStart := 87345 },
  { event := event87346
    frameStart := 87345 },
  { event := event87347
    frameStart := 87345 },
  { event := event87348
    frameStart := 87345 },
  { event := event87349
    frameStart := 87345 },
  { event := event87350
    frameStart := 87345 },
  { event := event87351
    frameStart := 87345 },
  { event := event87352
    frameStart := 87345 },
  { event := event87353
    frameStart := 87345 },
  { event := event87354
    frameStart := 87345 },
  { event := event87355
    frameStart := 87345 },
  { event := event87356
    frameStart := 87345 },
  { event := event87357
    frameStart := 87345 },
  { event := event87358
    frameStart := 87345 },
  { event := event87359
    frameStart := 87345 }
]

def eventLeaf5460 : Array AnnotatedEvent := #[
  { event := event87360
    frameStart := 87345 },
  { event := event87361
    frameStart := 87345 },
  { event := event87362
    frameStart := 87345 },
  { event := event87363
    frameStart := 87345 },
  { event := event87364
    frameStart := 87345 },
  { event := event87365
    frameStart := 87345 },
  { event := event87366
    frameStart := 87345 },
  { event := event87367
    frameStart := 87345 },
  { event := event87368
    frameStart := 87345 },
  { event := event87369
    frameStart := 87345 },
  { event := event87370
    frameStart := 87345 },
  { event := event87371
    frameStart := 87345 },
  { event := event87372
    frameStart := 87345 },
  { event := event87373
    frameStart := 87345 },
  { event := event87374
    frameStart := 87345 },
  { event := event87375
    frameStart := 87345 }
]

def eventLeaf5461 : Array AnnotatedEvent := #[
  { event := event87376
    frameStart := 87345 },
  { event := event87377
    frameStart := 87345 },
  { event := event87378
    frameStart := 87345 },
  { event := event87379
    frameStart := 87345 },
  { event := event87380
    frameStart := 87345 },
  { event := event87381
    frameStart := 87345 },
  { event := event87382
    frameStart := 87345 },
  { event := event87383
    frameStart := 87345 },
  { event := event87384
    frameStart := 87345 },
  { event := event87385
    frameStart := 87345 },
  { event := event87386
    frameStart := 87345 },
  { event := event87387
    frameStart := 87345 },
  { event := event87388
    frameStart := 87345 },
  { event := event87389
    frameStart := 87345 },
  { event := event87390
    frameStart := 87345 },
  { event := event87391
    frameStart := 87345 }
]

def eventLeaf5462 : Array AnnotatedEvent := #[
  { event := event87392
    frameStart := 87345 },
  { event := event87393
    frameStart := 87345 },
  { event := event87394
    frameStart := 87345 },
  { event := event87395
    frameStart := 87345 },
  { event := event87396
    frameStart := 87345 },
  { event := event87397
    frameStart := 87345 },
  { event := event87398
    frameStart := 87345 },
  { event := event87399
    frameStart := 87345 },
  { event := event87400
    frameStart := 87345 },
  { event := event87401
    frameStart := 87345 },
  { event := event87402
    frameStart := 87345 },
  { event := event87403
    frameStart := 87345 },
  { event := event87404
    frameStart := 87345 },
  { event := event87405
    frameStart := 87345 },
  { event := event87406
    frameStart := 87345 },
  { event := event87407
    frameStart := 87345 }
]

def eventLeaf5463 : Array AnnotatedEvent := #[
  { event := event87408
    frameStart := 87345 },
  { event := event87409
    frameStart := 87345 },
  { event := event87410
    frameStart := 87345 },
  { event := event87411
    frameStart := 87345 },
  { event := event87412
    frameStart := 87345 },
  { event := event87413
    frameStart := 87345 },
  { event := event87414
    frameStart := 87345 },
  { event := event87415
    frameStart := 87345 },
  { event := event87416
    frameStart := 87345 },
  { event := event87417
    frameStart := 87345 },
  { event := event87418
    frameStart := 87345 },
  { event := event87419
    frameStart := 87345 },
  { event := event87420
    frameStart := 87345 },
  { event := event87421
    frameStart := 87345 },
  { event := event87422
    frameStart := 87345 },
  { event := event87423
    frameStart := 87345 }
]

def eventLeaf5464 : Array AnnotatedEvent := #[
  { event := event87424
    frameStart := 87345 },
  { event := event87425
    frameStart := 87345 },
  { event := event87426
    frameStart := 87345 },
  { event := event87427
    frameStart := 87345 },
  { event := event87428
    frameStart := 87345 },
  { event := event87429
    frameStart := 87345 },
  { event := event87430
    frameStart := 87345 },
  { event := event87431
    frameStart := 87345 },
  { event := event87432
    frameStart := 87345 },
  { event := event87433
    frameStart := 87345 },
  { event := event87434
    frameStart := 87345 },
  { event := event87435
    frameStart := 87345 },
  { event := event87436
    frameStart := 87345 },
  { event := event87437
    frameStart := 87345 },
  { event := event87438
    frameStart := 87345 },
  { event := event87439
    frameStart := 87345 }
]

def eventLeaf5465 : Array AnnotatedEvent := #[
  { event := event87440
    frameStart := 87345 },
  { event := event87441
    frameStart := 87345 },
  { event := event87442
    frameStart := 87345 },
  { event := event87443
    frameStart := 87345 },
  { event := event87444
    frameStart := 87345 },
  { event := event87445
    frameStart := 87345 },
  { event := event87446
    frameStart := 87345 },
  { event := event87447
    frameStart := 87345 },
  { event := event87448
    frameStart := 87345 },
  { event := event87449
    frameStart := 0 },
  { event := event87450
    frameStart := 0 },
  { event := event87451
    frameStart := 0 },
  { event := event87452
    frameStart := 0 },
  { event := event87453
    frameStart := 0 },
  { event := event87454
    frameStart := 0 },
  { event := event87455
    frameStart := 0 }
]

def eventLeaf5466 : Array AnnotatedEvent := #[
  { event := event87456
    frameStart := 0 },
  { event := event87457
    frameStart := 0 },
  { event := event87458
    frameStart := 0 },
  { event := event87459
    frameStart := 0 },
  { event := event87460
    frameStart := 0 },
  { event := event87461
    frameStart := 0 },
  { event := event87462
    frameStart := 0 },
  { event := event87463
    frameStart := 0 },
  { event := event87464
    frameStart := 0 },
  { event := event87465
    frameStart := 0 },
  { event := event87466
    frameStart := 0 },
  { event := event87467
    frameStart := 0 },
  { event := event87468
    frameStart := 0 },
  { event := event87469
    frameStart := 0 },
  { event := event87470
    frameStart := 0 },
  { event := event87471
    frameStart := 0 }
]

def eventLeaf5467 : Array AnnotatedEvent := #[
  { event := event87472
    frameStart := 0 },
  { event := event87473
    frameStart := 0 },
  { event := event87474
    frameStart := 0 },
  { event := event87475
    frameStart := 0 },
  { event := event87476
    frameStart := 0 },
  { event := event87477
    frameStart := 0 },
  { event := event87478
    frameStart := 0 },
  { event := event87479
    frameStart := 0 },
  { event := event87480
    frameStart := 0 },
  { event := event87481
    frameStart := 0 },
  { event := event87482
    frameStart := 0 },
  { event := event87483
    frameStart := 0 },
  { event := event87484
    frameStart := 0 },
  { event := event87485
    frameStart := 0 },
  { event := event87486
    frameStart := 0 },
  { event := event87487
    frameStart := 0 }
]

def eventLeaf5468 : Array AnnotatedEvent := #[
  { event := event87488
    frameStart := 0 },
  { event := event87489
    frameStart := 0 },
  { event := event87490
    frameStart := 0 },
  { event := event87491
    frameStart := 0 },
  { event := event87492
    frameStart := 0 },
  { event := event87493
    frameStart := 0 },
  { event := event87494
    frameStart := 0 },
  { event := event87495
    frameStart := 0 },
  { event := event87496
    frameStart := 0 },
  { event := event87497
    frameStart := 0 },
  { event := event87498
    frameStart := 0 },
  { event := event87499
    frameStart := 0 },
  { event := event87500
    frameStart := 0 },
  { event := event87501
    frameStart := 0 },
  { event := event87502
    frameStart := 0 },
  { event := event87503
    frameStart := 87503 }
]

def eventLeaf5469 : Array AnnotatedEvent := #[
  { event := event87504
    frameStart := 87503 },
  { event := event87505
    frameStart := 87503 },
  { event := event87506
    frameStart := 87503 },
  { event := event87507
    frameStart := 87503 },
  { event := event87508
    frameStart := 87503 },
  { event := event87509
    frameStart := 87503 },
  { event := event87510
    frameStart := 87503 },
  { event := event87511
    frameStart := 87503 },
  { event := event87512
    frameStart := 87503 },
  { event := event87513
    frameStart := 87503 },
  { event := event87514
    frameStart := 87503 },
  { event := event87515
    frameStart := 87503 },
  { event := event87516
    frameStart := 87503 },
  { event := event87517
    frameStart := 87503 },
  { event := event87518
    frameStart := 87503 },
  { event := event87519
    frameStart := 87503 }
]

def eventLeaf5470 : Array AnnotatedEvent := #[
  { event := event87520
    frameStart := 87503 },
  { event := event87521
    frameStart := 87503 },
  { event := event87522
    frameStart := 87503 },
  { event := event87523
    frameStart := 87503 },
  { event := event87524
    frameStart := 87503 },
  { event := event87525
    frameStart := 87503 },
  { event := event87526
    frameStart := 87503 },
  { event := event87527
    frameStart := 87503 },
  { event := event87528
    frameStart := 87503 },
  { event := event87529
    frameStart := 87503 },
  { event := event87530
    frameStart := 87503 },
  { event := event87531
    frameStart := 87503 },
  { event := event87532
    frameStart := 87503 },
  { event := event87533
    frameStart := 87503 },
  { event := event87534
    frameStart := 87503 },
  { event := event87535
    frameStart := 87503 }
]

def eventLeaf5471 : Array AnnotatedEvent := #[
  { event := event87536
    frameStart := 87503 },
  { event := event87537
    frameStart := 87503 },
  { event := event87538
    frameStart := 87503 },
  { event := event87539
    frameStart := 87503 },
  { event := event87540
    frameStart := 87503 },
  { event := event87541
    frameStart := 87503 },
  { event := event87542
    frameStart := 87503 },
  { event := event87543
    frameStart := 87503 },
  { event := event87544
    frameStart := 87503 },
  { event := event87545
    frameStart := 87503 },
  { event := event87546
    frameStart := 87503 },
  { event := event87547
    frameStart := 87503 },
  { event := event87548
    frameStart := 87503 },
  { event := event87549
    frameStart := 87503 },
  { event := event87550
    frameStart := 87503 },
  { event := event87551
    frameStart := 87503 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events341
