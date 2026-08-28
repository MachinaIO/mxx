import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events298

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event76288 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12755⟩⟩) (.product (.predecessor 0 76286 .coefficient) (.predecessor 1 76287 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event76289 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12755⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨10025⟩⟩, ⟨.program ⟨214⟩, ⟨12754⟩⟩], []⟩) [⟨.result 76285 .coefficient, true, some 1⟩, ⟨.result 76282 .coefficient, true, some 1⟩])

def event76290 : Event := .survivorFold (1) 76289

def exact76291RawTerms : List Term := []

theorem exact76291RawTermsValid :
    exact76291RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76291 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12755⟩⟩) exact76291RawTerms (.finite 2116) 76288 (.finite 2116) (some (76289))

def event76292 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12756⟩⟩) 0 ⟨12755⟩ 76291

def event76293 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12756⟩⟩) (.identity (.predecessor 0 76292 .coefficient))

def event76294 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨12756⟩⟩) (.finite 2116)

def event76295 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16629⟩⟩) 0 ⟨12756⟩ 76294

def event76296 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16629⟩⟩) (.authority (.programFamilyFact))

def exact76297RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16629⟩⟩], []⟩, (1)⟩]

theorem exact76297RawTermsValid :
    exact76297RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76297 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16629⟩⟩) exact76297RawTerms (.finite 46) 76296 .exactZero (none)

def event76298 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16630⟩⟩) 0 ⟨16629⟩ 76297

def event76299 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16630⟩⟩) (.identity (.predecessor 0 76298 .coefficient))

def event76300 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨16630⟩⟩) (.finite 46)

def event76301 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22332⟩⟩) 0 ⟨16630⟩ 76300

def event76302 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22332⟩⟩) (.authority (.relationPreimageSource ⟨58⟩))

def exact76303RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨22332⟩⟩]⟩, (1)⟩]

theorem exact76303RawTermsValid :
    exact76303RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76303 : Event := .resultExact (⟨.program ⟨214⟩, ⟨22332⟩⟩) exact76303RawTerms (.finite 136065468) 76302 .exactZero (none)

def event76304 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6⟩⟩) (.authority (.operator))

def exact76305RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩]⟩, (1)⟩]

theorem exact76305RawTermsValid :
    exact76305RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76305 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6⟩⟩) exact76305RawTerms .large 76304 .exactZero (none)

def event76306 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22333⟩⟩) 0 ⟨6⟩ 76305

def event76307 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22333⟩⟩) 1 ⟨22332⟩ 76303

def event76308 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22333⟩⟩) (.product (.predecessor 0 76306 .coefficient) (.predecessor 1 76307 .coefficient) (⟨false, false, none, none, none⟩))

def event76309 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨22333⟩⟩, .operator (⟨76305, 0⟩, ⟨76303, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22332⟩⟩]⟩, (1)⟩)

def exact76310RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22332⟩⟩]⟩, (1)⟩]

theorem exact76310RawTermsValid :
    exact76310RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76310 : Event := .resultExact (⟨.program ⟨214⟩, ⟨22333⟩⟩) exact76310RawTerms .large 76308 .exactZero (none)

def event76311 : Event := .preFoldPolynomial 76310 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22332⟩⟩]⟩, (1)⟩] .exactZero none

def exact76312RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22332⟩⟩]⟩, (1)⟩]

def event76312 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨22333⟩⟩) 76311 exact76312RawTerms .large 76308 .exactZero (none)

def event76313 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨29371⟩⟩)

def event76314 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event76315 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event76316 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨360⟩⟩) (.authority (.operator))

def event76317 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨360⟩⟩) (.finite 2)

def event76318 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event76319 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event76320 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event76321 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event76322 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 76321

def event76323 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 76319

def event76324 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 76322 .coefficient) (.value (.predecessor 1 76323 .coefficient)))

def event76325 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event76326 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5504⟩⟩) 0 ⟨5503⟩ 76325

def event76327 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5504⟩⟩) 1 ⟨360⟩ 76317

def event76328 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5504⟩⟩) (.sum [.predecessor 0 76326 .coefficient, .predecessor 1 76327 .coefficient])

def event76329 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5504⟩⟩) (.finite 219)

def event76330 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5530⟩⟩) 0 ⟨5504⟩ 76329

def event76331 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5530⟩⟩) 1 ⟨961⟩ 76315

def event76332 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5530⟩⟩) (.identity (.predecessor 1 76331 .coefficient))

def event76333 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5530⟩⟩) (.finite 224)

def event76334 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12754⟩⟩) 0 ⟨5530⟩ 76333

def event76335 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12754⟩⟩) (.authority (.programFamilyFact))

def exact76336RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨12754⟩⟩], []⟩, (1)⟩]

theorem exact76336RawTermsValid :
    exact76336RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76336 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12754⟩⟩) exact76336RawTerms (.finite 46) 76335 .exactZero (none)

def event76337 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10025⟩⟩) 0 ⟨5530⟩ 76333

def event76338 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10025⟩⟩) (.authority (.programFamilyFact))

def exact76339RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨10025⟩⟩], []⟩, (1)⟩]

theorem exact76339RawTermsValid :
    exact76339RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76339 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10025⟩⟩) exact76339RawTerms (.finite 46) 76338 .exactZero (none)

def event76340 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12755⟩⟩) 0 ⟨10025⟩ 76339

def event76341 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12755⟩⟩) 1 ⟨12754⟩ 76336

def event76342 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12755⟩⟩) (.product (.predecessor 0 76340 .coefficient) (.predecessor 1 76341 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event76343 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨12755⟩⟩, .operator (⟨76339, 0⟩, ⟨76336, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨10025⟩⟩, ⟨.program ⟨214⟩, ⟨12754⟩⟩], []⟩, (1)⟩)

def exact76344RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨10025⟩⟩, ⟨.program ⟨214⟩, ⟨12754⟩⟩], []⟩, (1)⟩]

theorem exact76344RawTermsValid :
    exact76344RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76344 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12755⟩⟩) exact76344RawTerms (.finite 2116) 76342 .exactZero (none)

def event76345 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12756⟩⟩) 0 ⟨12755⟩ 76344

def event76346 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12756⟩⟩) (.identity (.predecessor 0 76345 .coefficient))

def event76347 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨12756⟩⟩) (.finite 2116)

def event76348 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16629⟩⟩) 0 ⟨12756⟩ 76347

def event76349 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16629⟩⟩) (.authority (.programFamilyFact))

def exact76350RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16629⟩⟩], []⟩, (1)⟩]

theorem exact76350RawTermsValid :
    exact76350RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76350 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16629⟩⟩) exact76350RawTerms (.finite 46) 76349 .exactZero (none)

def event76351 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16630⟩⟩) 0 ⟨16629⟩ 76350

def event76352 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16630⟩⟩) (.identity (.predecessor 0 76351 .coefficient))

def event76353 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨16630⟩⟩) (.finite 46)

def event76354 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24598⟩⟩) 0 ⟨16630⟩ 76353

def event76355 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24598⟩⟩) (.authority (.programFamilyFact))

def event76356 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨24598⟩⟩) (.finite 3720)

def event76357 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨6689⟩⟩) .missing

def event76358 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24599⟩⟩) 0 ⟨6689⟩ 76357

def event76359 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24599⟩⟩) 1 ⟨24598⟩ 76356

def event76360 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24599⟩⟩) (.authority (.operator))

def exact76361RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨24599⟩⟩]⟩, (1)⟩]

theorem exact76361RawTermsValid :
    exact76361RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76361 : Event := .resultExact (⟨.program ⟨214⟩, ⟨24599⟩⟩) exact76361RawTerms .large 76360 .exactZero (none)

def event76362 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29365⟩⟩) 0 ⟨24599⟩ 76361

def event76363 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29365⟩⟩) (.authority (.operator))

def exact76364RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨29365⟩⟩]⟩, (1)⟩]

theorem exact76364RawTermsValid :
    exact76364RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76364 : Event := .resultExact (⟨.program ⟨214⟩, ⟨29365⟩⟩) exact76364RawTerms (.finite 8192) 76363 .exactZero (none)

def event76365 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨110⟩⟩) (.authority (.operator))

def event76366 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨110⟩⟩) .exactZero

def event76367 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16704⟩⟩) 0 ⟨16630⟩ 76353

def event76368 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16704⟩⟩) 1 ⟨110⟩ 76366

def event76369 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16704⟩⟩) (.sum [.predecessor 0 76367 .coefficient, .predecessor 1 76368 .coefficient])

def event76370 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨16704⟩⟩) (.finite 46)

def event76371 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16705⟩⟩) 0 ⟨16704⟩ 76370

def event76372 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16705⟩⟩) (.identity (.predecessor 0 76371 .coefficient))

def exact76373RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16629⟩⟩], []⟩, (1)⟩]

theorem exact76373RawTermsValid :
    exact76373RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76373 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16705⟩⟩) exact76373RawTerms (.finite 46) 76372 .exactZero (none)

def event76374 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6544⟩⟩) (.authority (.factStore))

def exact76375RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact76375RawTermsValid :
    exact76375RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76375 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6544⟩⟩) exact76375RawTerms .large 76374 .exactZero (none)

def event76376 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16706⟩⟩) 0 ⟨6544⟩ 76375

def event76377 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16706⟩⟩) 1 ⟨16705⟩ 76373

def event76378 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16706⟩⟩) (.product (.predecessor 0 76376 .coefficient) (.predecessor 1 76377 .coefficient) (⟨false, false, none, none, none⟩))

def event76379 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨16706⟩⟩, .operator (⟨76375, 0⟩, ⟨76373, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨16629⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact76380RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16629⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact76380RawTermsValid :
    exact76380RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76380 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16706⟩⟩) exact76380RawTerms .large 76378 .exactZero (none)

def event76381 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6704⟩⟩) 0 ⟨6689⟩ 76357

def event76382 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6704⟩⟩) (.authority (.operator))

def exact76383RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6704⟩⟩]⟩, (1)⟩]

theorem exact76383RawTermsValid :
    exact76383RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76383 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6704⟩⟩) exact76383RawTerms .large 76382 .exactZero (none)

def event76384 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16707⟩⟩) 0 ⟨6704⟩ 76383

def event76385 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16707⟩⟩) 1 ⟨16706⟩ 76380

def event76386 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16707⟩⟩) (.sum [.predecessor 0 76384 .coefficient, .predecessor 1 76385 .coefficient])

def exact76387RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6704⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16629⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact76387RawTermsValid :
    exact76387RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76387 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16707⟩⟩) exact76387RawTerms .large 76386 .exactZero (none)

def event76388 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29366⟩⟩) 0 ⟨16707⟩ 76387

def event76389 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29366⟩⟩) 1 ⟨29365⟩ 76364

def event76390 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29366⟩⟩) (.product (.predecessor 0 76388 .coefficient) (.predecessor 1 76389 .coefficient) (⟨false, false, none, none, none⟩))

def event76391 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨29366⟩⟩, .operator (⟨76387, 0⟩, ⟨76364, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6704⟩⟩, ⟨.program ⟨214⟩, ⟨29365⟩⟩]⟩, (1)⟩)

def event76392 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨29366⟩⟩, .operator (⟨76387, 1⟩, ⟨76364, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨16629⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29365⟩⟩]⟩, (-1)⟩)

def event76393 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨29366⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨16629⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29365⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨29365⟩⟩) ⟨24599⟩ 76361)

def event76394 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨29366⟩⟩, .relation 76393 0, ⟨[⟨.program ⟨214⟩, ⟨16629⟩⟩], [⟨.program ⟨214⟩, ⟨24599⟩⟩]⟩, (-1)⟩)

def exact76395RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6704⟩⟩, ⟨.program ⟨214⟩, ⟨29365⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16629⟩⟩], [⟨.program ⟨214⟩, ⟨24599⟩⟩]⟩, (-1)⟩]

theorem exact76395RawTermsValid :
    exact76395RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76395 : Event := .resultExact (⟨.program ⟨214⟩, ⟨29366⟩⟩) exact76395RawTerms .large 76390 .exactZero (none)

def event76396 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17714⟩⟩) 0 ⟨16630⟩ 76353

def event76397 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨17714⟩⟩) (.authority (.programFamilyFact))

def exact76398RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨17714⟩⟩], []⟩, (1)⟩]

theorem exact76398RawTermsValid :
    exact76398RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76398 : Event := .resultExact (⟨.program ⟨214⟩, ⟨17714⟩⟩) exact76398RawTerms (.finite 46) 76397 .exactZero (none)

def event76399 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17716⟩⟩) 0 ⟨6544⟩ 76375

def event76400 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17716⟩⟩) 1 ⟨17714⟩ 76398

def event76401 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨17716⟩⟩) (.product (.predecessor 0 76399 .coefficient) (.predecessor 1 76400 .coefficient) (⟨false, true, none, none, some 1⟩))

def event76402 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨17716⟩⟩, .operator (⟨76375, 0⟩, ⟨76398, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨17714⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact76403RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨17714⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact76403RawTermsValid :
    exact76403RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76403 : Event := .resultExact (⟨.program ⟨214⟩, ⟨17716⟩⟩) exact76403RawTerms .large 76401 .exactZero (none)

def event76404 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6736⟩⟩) 0 ⟨6689⟩ 76357

def event76405 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6736⟩⟩) (.authority (.operator))

def exact76406RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6736⟩⟩]⟩, (1)⟩]

theorem exact76406RawTermsValid :
    exact76406RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76406 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6736⟩⟩) exact76406RawTerms .large 76405 .exactZero (none)

def event76407 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17717⟩⟩) 0 ⟨6736⟩ 76406

def event76408 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17717⟩⟩) 1 ⟨17716⟩ 76403

def event76409 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨17717⟩⟩) (.sum [.predecessor 0 76407 .coefficient, .predecessor 1 76408 .coefficient])

def exact76410RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6736⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17714⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact76410RawTermsValid :
    exact76410RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76410 : Event := .resultExact (⟨.program ⟨214⟩, ⟨17717⟩⟩) exact76410RawTerms .large 76409 .exactZero (none)

def event76411 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29371⟩⟩) 0 ⟨17717⟩ 76410

def event76412 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29371⟩⟩) 1 ⟨29366⟩ 76395

def event76413 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29371⟩⟩) (.sum [.predecessor 0 76411 .coefficient, .predecessor 1 76412 .coefficient])

def exact76414RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6704⟩⟩, ⟨.program ⟨214⟩, ⟨29365⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6736⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16629⟩⟩], [⟨.program ⟨214⟩, ⟨24599⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17714⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact76414RawTermsValid :
    exact76414RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76414 : Event := .resultExact (⟨.program ⟨214⟩, ⟨29371⟩⟩) exact76414RawTerms .large 76413 .exactZero (none)

def event76415 : Event := .preFoldPolynomial 76414 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6704⟩⟩, ⟨.program ⟨214⟩, ⟨29365⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6736⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16629⟩⟩], [⟨.program ⟨214⟩, ⟨24599⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17714⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩] .exactZero none

def exact76416RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6704⟩⟩, ⟨.program ⟨214⟩, ⟨29365⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6736⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16629⟩⟩], [⟨.program ⟨214⟩, ⟨24599⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17714⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

def event76416 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨29371⟩⟩) 76415 exact76416RawTerms .large 76413 .exactZero (none)

def event76417 : Event := .specializationComputed (⟨.program ⟨214⟩, ⟨16630⟩⟩) ⟨⟨149⟩, ⟨58⟩, ⟨109⟩⟩ ⟨76259, 76417⟩

def event76418 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨22335⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22332⟩⟩]⟩) (1) 0 2 (.universal 76417 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22332⟩⟩]⟩) (none) 76416)

def event76419 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨22335⟩⟩, .relation 76418 1, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6736⟩⟩]⟩, (1)⟩)

def event76420 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨22335⟩⟩, .relation 76418 0, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6704⟩⟩, ⟨.program ⟨214⟩, ⟨29365⟩⟩]⟩, (-1)⟩)

def event76421 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨22335⟩⟩, .relation 76418 2, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16629⟩⟩], [⟨.program ⟨214⟩, ⟨24599⟩⟩]⟩, (1)⟩)

def event76422 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨22335⟩⟩, .relation 76418 3, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨17714⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact76423RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6704⟩⟩, ⟨.program ⟨214⟩, ⟨29365⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6736⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16629⟩⟩], [⟨.program ⟨214⟩, ⟨24599⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨17714⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact76423RawTermsValid :
    exact76423RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76423 : Event := .resultExact (⟨.program ⟨214⟩, ⟨22335⟩⟩) exact76423RawTerms .large 76255 (.finite 1811303510016) (some (76257))

def event76424 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29368⟩⟩) 0 ⟨22335⟩ 76423

def event76425 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29368⟩⟩) 1 ⟨29367⟩ 76245

def event76426 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29368⟩⟩) (.sum [.predecessor 0 76424 .coefficient, .predecessor 1 76425 .coefficient])

def event76427 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨29368⟩⟩, .operator (⟨76423, 0⟩, ⟨76245, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6704⟩⟩, ⟨.program ⟨214⟩, ⟨29365⟩⟩]⟩, (1)⟩)

def event76428 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨29368⟩⟩, .operator (⟨76423, 2⟩, ⟨76245, 1⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16629⟩⟩], [⟨.program ⟨214⟩, ⟨24599⟩⟩]⟩, (-1)⟩)

def event76429 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29368⟩⟩) (.sum [.result 76423 .summary, .result 76245 .summary])

def exact76430RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6736⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨17714⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact76430RawTermsValid :
    exact76430RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76430 : Event := .resultExact (⟨.program ⟨214⟩, ⟨29368⟩⟩) exact76430RawTerms .large 76426 (.finite 1292382248169874534400) (some (76429))

def event76431 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29369⟩⟩) 0 ⟨29368⟩ 76430

def event76432 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29369⟩⟩) 1 ⟨6666⟩ 5579

def event76433 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29369⟩⟩) (.product (.predecessor 0 76431 .coefficient) (.predecessor 1 76432 .coefficient) (⟨false, false, none, none, none⟩))

def event76434 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29369⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨6665⟩⟩]⟩) [⟨.result 5575 .coefficient, false, none⟩])

def event76435 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29369⟩⟩) (.product (.result 76430 .summary) (.transfer 76434) (⟨false, false, none, none, none⟩))

def event76436 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨29369⟩⟩, .operator (⟨76430, 0⟩, ⟨5579, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6736⟩⟩, ⟨.program ⟨214⟩, ⟨6665⟩⟩]⟩, (1)⟩)

def event76437 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨29369⟩⟩, .operator (⟨76430, 1⟩, ⟨5579, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨17714⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨6665⟩⟩]⟩, (-1)⟩)

def event76438 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨29369⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨17714⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨6665⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨6665⟩⟩) ⟨6604⟩ 5572)

def event76439 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨29369⟩⟩, .relation 76438 0, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨6459⟩⟩, ⟨.program ⟨214⟩, ⟨17714⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact76440RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6736⟩⟩, ⟨.program ⟨214⟩, ⟨6665⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨6459⟩⟩, ⟨.program ⟨214⟩, ⟨17714⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact76440RawTermsValid :
    exact76440RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76440 : Event := .resultExact (⟨.program ⟨214⟩, ⟨29369⟩⟩) exact76440RawTerms .large 76433 (.finite 4743063528899410259240550400) (some (76435))

def event76441 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24536⟩⟩) 0 ⟨6689⟩ 5477

def event76442 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24536⟩⟩) 1 ⟨24535⟩ 67217

def event76443 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24536⟩⟩) (.authority (.operator))

def exact76444RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨24536⟩⟩]⟩, (1)⟩]

theorem exact76444RawTermsValid :
    exact76444RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76444 : Event := .resultExact (⟨.program ⟨214⟩, ⟨24536⟩⟩) exact76444RawTerms .large 76443 .exactZero (none)

def event76445 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29148⟩⟩) 0 ⟨24536⟩ 76444

def event76446 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29148⟩⟩) (.authority (.operator))

def exact76447RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨29148⟩⟩]⟩, (1)⟩]

theorem exact76447RawTermsValid :
    exact76447RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76447 : Event := .resultExact (⟨.program ⟨214⟩, ⟨29148⟩⟩) exact76447RawTerms (.finite 8192) 76446 .exactZero (none)

def event76448 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29150⟩⟩) 0 ⟨25447⟩ 67501

def event76449 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29150⟩⟩) 1 ⟨29148⟩ 76447

def event76450 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29150⟩⟩) (.product (.predecessor 0 76448 .coefficient) (.predecessor 1 76449 .coefficient) (⟨false, false, none, none, none⟩))

def event76451 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29150⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨29148⟩⟩]⟩) [⟨.result 76447 .coefficient, false, none⟩])

def event76452 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29150⟩⟩) (.product (.result 67501 .summary) (.transfer 76451) (⟨false, false, none, none, none⟩))

def event76453 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨29150⟩⟩, .operator (⟨67501, 0⟩, ⟨76447, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6703⟩⟩, ⟨.program ⟨214⟩, ⟨29148⟩⟩]⟩, (1)⟩)

def event76454 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨29150⟩⟩, .operator (⟨67501, 1⟩, ⟨76447, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16545⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29148⟩⟩]⟩, (-1)⟩)

def event76455 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨29150⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16545⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29148⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨29148⟩⟩) ⟨24536⟩ 76444)

def event76456 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨29150⟩⟩, .relation 76455 0, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16545⟩⟩], [⟨.program ⟨214⟩, ⟨24536⟩⟩]⟩, (-1)⟩)

def exact76457RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6703⟩⟩, ⟨.program ⟨214⟩, ⟨29148⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16545⟩⟩], [⟨.program ⟨214⟩, ⟨24536⟩⟩]⟩, (-1)⟩]

theorem exact76457RawTermsValid :
    exact76457RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76457 : Event := .resultExact (⟨.program ⟨214⟩, ⟨29150⟩⟩) exact76457RawTerms .large 76450 (.finite 1292337421468529852416) (some (76452))

def event76458 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22188⟩⟩) 0 ⟨16546⟩ 3195

def event76459 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22188⟩⟩) (.authority (.relationPreimageSource ⟨56⟩))

def exact76460RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨22188⟩⟩]⟩, (1)⟩]

theorem exact76460RawTermsValid :
    exact76460RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76460 : Event := .resultExact (⟨.program ⟨214⟩, ⟨22188⟩⟩) exact76460RawTerms (.finite 136065468) 76459 .exactZero (none)

def event76461 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22190⟩⟩) 0 ⟨22188⟩ 76460

def event76462 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22190⟩⟩) 1 ⟨2348⟩ 4

def event76463 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22190⟩⟩) (.scale (.predecessor 0 76461 .coefficient) (.value (.predecessor 1 76462 .coefficient)))

def exact76464RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨22188⟩⟩]⟩, (1)⟩]

theorem exact76464RawTermsValid :
    exact76464RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76464 : Event := .resultExact (⟨.program ⟨214⟩, ⟨22190⟩⟩) exact76464RawTerms (.finite 136065468) 76463 .exactZero (none)

def event76465 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22191⟩⟩) 0 ⟨5535⟩ 65387

def event76466 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22191⟩⟩) 1 ⟨22190⟩ 76464

def event76467 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22191⟩⟩) (.product (.predecessor 0 76465 .coefficient) (.predecessor 1 76466 .coefficient) (⟨false, false, none, none, none⟩))

def event76468 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22191⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨22188⟩⟩]⟩) [⟨.result 76460 .coefficient, false, none⟩])

def event76469 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22191⟩⟩) (.product (.result 65387 .summary) (.transfer 76468) (⟨false, false, none, none, none⟩))

def event76470 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨22191⟩⟩, .operator (⟨65387, 0⟩, ⟨76464, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22188⟩⟩]⟩, (1)⟩)

def event76471 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨22189⟩⟩)

def event76472 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event76473 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event76474 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨360⟩⟩) (.authority (.operator))

def event76475 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨360⟩⟩) (.finite 2)

def event76476 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event76477 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event76478 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event76479 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event76480 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 76479

def event76481 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 76477

def event76482 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 76480 .coefficient) (.value (.predecessor 1 76481 .coefficient)))

def event76483 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event76484 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5504⟩⟩) 0 ⟨5503⟩ 76483

def event76485 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5504⟩⟩) 1 ⟨360⟩ 76475

def event76486 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5504⟩⟩) (.sum [.predecessor 0 76484 .coefficient, .predecessor 1 76485 .coefficient])

def event76487 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5504⟩⟩) (.finite 219)

def event76488 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5530⟩⟩) 0 ⟨5504⟩ 76487

def event76489 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5530⟩⟩) 1 ⟨961⟩ 76473

def event76490 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5530⟩⟩) (.identity (.predecessor 1 76489 .coefficient))

def event76491 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5530⟩⟩) (.finite 224)

def event76492 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12558⟩⟩) 0 ⟨5530⟩ 76491

def event76493 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12558⟩⟩) (.authority (.programFamilyFact))

def exact76494RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨12558⟩⟩], []⟩, (1)⟩]

theorem exact76494RawTermsValid :
    exact76494RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76494 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12558⟩⟩) exact76494RawTerms (.finite 42) 76493 .exactZero (none)

def event76495 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9920⟩⟩) 0 ⟨5530⟩ 76491

def event76496 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨9920⟩⟩) (.authority (.programFamilyFact))

def exact76497RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨9920⟩⟩], []⟩, (1)⟩]

theorem exact76497RawTermsValid :
    exact76497RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76497 : Event := .resultExact (⟨.program ⟨214⟩, ⟨9920⟩⟩) exact76497RawTerms (.finite 42) 76496 .exactZero (none)

def event76498 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12559⟩⟩) 0 ⟨9920⟩ 76497

def event76499 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12559⟩⟩) 1 ⟨12558⟩ 76494

def event76500 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12559⟩⟩) (.product (.predecessor 0 76498 .coefficient) (.predecessor 1 76499 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event76501 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12559⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨9920⟩⟩, ⟨.program ⟨214⟩, ⟨12558⟩⟩], []⟩) [⟨.result 76497 .coefficient, true, some 1⟩, ⟨.result 76494 .coefficient, true, some 1⟩])

def event76502 : Event := .survivorFold (1) 76501

def exact76503RawTerms : List Term := []

theorem exact76503RawTermsValid :
    exact76503RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76503 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12559⟩⟩) exact76503RawTerms (.finite 1764) 76500 (.finite 1764) (some (76501))

def event76504 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12560⟩⟩) 0 ⟨12559⟩ 76503

def event76505 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12560⟩⟩) (.identity (.predecessor 0 76504 .coefficient))

def event76506 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨12560⟩⟩) (.finite 1764)

def event76507 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16545⟩⟩) 0 ⟨12560⟩ 76506

def event76508 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16545⟩⟩) (.authority (.programFamilyFact))

def exact76509RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16545⟩⟩], []⟩, (1)⟩]

theorem exact76509RawTermsValid :
    exact76509RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76509 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16545⟩⟩) exact76509RawTerms (.finite 42) 76508 .exactZero (none)

def event76510 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16546⟩⟩) 0 ⟨16545⟩ 76509

def event76511 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16546⟩⟩) (.identity (.predecessor 0 76510 .coefficient))

def event76512 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨16546⟩⟩) (.finite 42)

def event76513 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22188⟩⟩) 0 ⟨16546⟩ 76512

def event76514 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22188⟩⟩) (.authority (.relationPreimageSource ⟨56⟩))

def exact76515RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨22188⟩⟩]⟩, (1)⟩]

theorem exact76515RawTermsValid :
    exact76515RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76515 : Event := .resultExact (⟨.program ⟨214⟩, ⟨22188⟩⟩) exact76515RawTerms (.finite 136065468) 76514 .exactZero (none)

def event76516 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6⟩⟩) (.authority (.operator))

def exact76517RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩]⟩, (1)⟩]

theorem exact76517RawTermsValid :
    exact76517RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76517 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6⟩⟩) exact76517RawTerms .large 76516 .exactZero (none)

def event76518 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22189⟩⟩) 0 ⟨6⟩ 76517

def event76519 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22189⟩⟩) 1 ⟨22188⟩ 76515

def event76520 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22189⟩⟩) (.product (.predecessor 0 76518 .coefficient) (.predecessor 1 76519 .coefficient) (⟨false, false, none, none, none⟩))

def event76521 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨22189⟩⟩, .operator (⟨76517, 0⟩, ⟨76515, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22188⟩⟩]⟩, (1)⟩)

def exact76522RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22188⟩⟩]⟩, (1)⟩]

theorem exact76522RawTermsValid :
    exact76522RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76522 : Event := .resultExact (⟨.program ⟨214⟩, ⟨22189⟩⟩) exact76522RawTerms .large 76520 .exactZero (none)

def event76523 : Event := .preFoldPolynomial 76522 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22188⟩⟩]⟩, (1)⟩] .exactZero none

def exact76524RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22188⟩⟩]⟩, (1)⟩]

def event76524 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨22189⟩⟩) 76523 exact76524RawTerms .large 76520 .exactZero (none)

def event76525 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨29154⟩⟩)

def event76526 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event76527 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event76528 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨360⟩⟩) (.authority (.operator))

def event76529 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨360⟩⟩) (.finite 2)

def event76530 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event76531 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event76532 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event76533 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event76534 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 76533

def event76535 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 76531

def event76536 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 76534 .coefficient) (.value (.predecessor 1 76535 .coefficient)))

def event76537 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event76538 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5504⟩⟩) 0 ⟨5503⟩ 76537

def event76539 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5504⟩⟩) 1 ⟨360⟩ 76529

def event76540 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5504⟩⟩) (.sum [.predecessor 0 76538 .coefficient, .predecessor 1 76539 .coefficient])

def event76541 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5504⟩⟩) (.finite 219)

def event76542 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5530⟩⟩) 0 ⟨5504⟩ 76541

def event76543 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5530⟩⟩) 1 ⟨961⟩ 76527

def eventLeaf4768 : Array AnnotatedEvent := #[
  { event := event76288
    frameStart := 76259 },
  { event := event76289
    frameStart := 76259 },
  { event := event76290
    frameStart := 76259 },
  { event := event76291
    frameStart := 76259 },
  { event := event76292
    frameStart := 76259 },
  { event := event76293
    frameStart := 76259 },
  { event := event76294
    frameStart := 76259 },
  { event := event76295
    frameStart := 76259 },
  { event := event76296
    frameStart := 76259 },
  { event := event76297
    frameStart := 76259 },
  { event := event76298
    frameStart := 76259 },
  { event := event76299
    frameStart := 76259 },
  { event := event76300
    frameStart := 76259 },
  { event := event76301
    frameStart := 76259 },
  { event := event76302
    frameStart := 76259 },
  { event := event76303
    frameStart := 76259 }
]

def eventLeaf4769 : Array AnnotatedEvent := #[
  { event := event76304
    frameStart := 76259 },
  { event := event76305
    frameStart := 76259 },
  { event := event76306
    frameStart := 76259 },
  { event := event76307
    frameStart := 76259 },
  { event := event76308
    frameStart := 76259 },
  { event := event76309
    frameStart := 76259 },
  { event := event76310
    frameStart := 76259 },
  { event := event76311
    frameStart := 76259 },
  { event := event76312
    frameStart := 76259 },
  { event := event76313
    frameStart := 76313 },
  { event := event76314
    frameStart := 76313 },
  { event := event76315
    frameStart := 76313 },
  { event := event76316
    frameStart := 76313 },
  { event := event76317
    frameStart := 76313 },
  { event := event76318
    frameStart := 76313 },
  { event := event76319
    frameStart := 76313 }
]

def eventLeaf4770 : Array AnnotatedEvent := #[
  { event := event76320
    frameStart := 76313 },
  { event := event76321
    frameStart := 76313 },
  { event := event76322
    frameStart := 76313 },
  { event := event76323
    frameStart := 76313 },
  { event := event76324
    frameStart := 76313 },
  { event := event76325
    frameStart := 76313 },
  { event := event76326
    frameStart := 76313 },
  { event := event76327
    frameStart := 76313 },
  { event := event76328
    frameStart := 76313 },
  { event := event76329
    frameStart := 76313 },
  { event := event76330
    frameStart := 76313 },
  { event := event76331
    frameStart := 76313 },
  { event := event76332
    frameStart := 76313 },
  { event := event76333
    frameStart := 76313 },
  { event := event76334
    frameStart := 76313 },
  { event := event76335
    frameStart := 76313 }
]

def eventLeaf4771 : Array AnnotatedEvent := #[
  { event := event76336
    frameStart := 76313 },
  { event := event76337
    frameStart := 76313 },
  { event := event76338
    frameStart := 76313 },
  { event := event76339
    frameStart := 76313 },
  { event := event76340
    frameStart := 76313 },
  { event := event76341
    frameStart := 76313 },
  { event := event76342
    frameStart := 76313 },
  { event := event76343
    frameStart := 76313 },
  { event := event76344
    frameStart := 76313 },
  { event := event76345
    frameStart := 76313 },
  { event := event76346
    frameStart := 76313 },
  { event := event76347
    frameStart := 76313 },
  { event := event76348
    frameStart := 76313 },
  { event := event76349
    frameStart := 76313 },
  { event := event76350
    frameStart := 76313 },
  { event := event76351
    frameStart := 76313 }
]

def eventLeaf4772 : Array AnnotatedEvent := #[
  { event := event76352
    frameStart := 76313 },
  { event := event76353
    frameStart := 76313 },
  { event := event76354
    frameStart := 76313 },
  { event := event76355
    frameStart := 76313 },
  { event := event76356
    frameStart := 76313 },
  { event := event76357
    frameStart := 76313 },
  { event := event76358
    frameStart := 76313 },
  { event := event76359
    frameStart := 76313 },
  { event := event76360
    frameStart := 76313 },
  { event := event76361
    frameStart := 76313 },
  { event := event76362
    frameStart := 76313 },
  { event := event76363
    frameStart := 76313 },
  { event := event76364
    frameStart := 76313 },
  { event := event76365
    frameStart := 76313 },
  { event := event76366
    frameStart := 76313 },
  { event := event76367
    frameStart := 76313 }
]

def eventLeaf4773 : Array AnnotatedEvent := #[
  { event := event76368
    frameStart := 76313 },
  { event := event76369
    frameStart := 76313 },
  { event := event76370
    frameStart := 76313 },
  { event := event76371
    frameStart := 76313 },
  { event := event76372
    frameStart := 76313 },
  { event := event76373
    frameStart := 76313 },
  { event := event76374
    frameStart := 76313 },
  { event := event76375
    frameStart := 76313 },
  { event := event76376
    frameStart := 76313 },
  { event := event76377
    frameStart := 76313 },
  { event := event76378
    frameStart := 76313 },
  { event := event76379
    frameStart := 76313 },
  { event := event76380
    frameStart := 76313 },
  { event := event76381
    frameStart := 76313 },
  { event := event76382
    frameStart := 76313 },
  { event := event76383
    frameStart := 76313 }
]

def eventLeaf4774 : Array AnnotatedEvent := #[
  { event := event76384
    frameStart := 76313 },
  { event := event76385
    frameStart := 76313 },
  { event := event76386
    frameStart := 76313 },
  { event := event76387
    frameStart := 76313 },
  { event := event76388
    frameStart := 76313 },
  { event := event76389
    frameStart := 76313 },
  { event := event76390
    frameStart := 76313 },
  { event := event76391
    frameStart := 76313 },
  { event := event76392
    frameStart := 76313 },
  { event := event76393
    frameStart := 76313 },
  { event := event76394
    frameStart := 76313 },
  { event := event76395
    frameStart := 76313 },
  { event := event76396
    frameStart := 76313 },
  { event := event76397
    frameStart := 76313 },
  { event := event76398
    frameStart := 76313 },
  { event := event76399
    frameStart := 76313 }
]

def eventLeaf4775 : Array AnnotatedEvent := #[
  { event := event76400
    frameStart := 76313 },
  { event := event76401
    frameStart := 76313 },
  { event := event76402
    frameStart := 76313 },
  { event := event76403
    frameStart := 76313 },
  { event := event76404
    frameStart := 76313 },
  { event := event76405
    frameStart := 76313 },
  { event := event76406
    frameStart := 76313 },
  { event := event76407
    frameStart := 76313 },
  { event := event76408
    frameStart := 76313 },
  { event := event76409
    frameStart := 76313 },
  { event := event76410
    frameStart := 76313 },
  { event := event76411
    frameStart := 76313 },
  { event := event76412
    frameStart := 76313 },
  { event := event76413
    frameStart := 76313 },
  { event := event76414
    frameStart := 76313 },
  { event := event76415
    frameStart := 76313 }
]

def eventLeaf4776 : Array AnnotatedEvent := #[
  { event := event76416
    frameStart := 76313 },
  { event := event76417
    frameStart := 0 },
  { event := event76418
    frameStart := 0 },
  { event := event76419
    frameStart := 0 },
  { event := event76420
    frameStart := 0 },
  { event := event76421
    frameStart := 0 },
  { event := event76422
    frameStart := 0 },
  { event := event76423
    frameStart := 0 },
  { event := event76424
    frameStart := 0 },
  { event := event76425
    frameStart := 0 },
  { event := event76426
    frameStart := 0 },
  { event := event76427
    frameStart := 0 },
  { event := event76428
    frameStart := 0 },
  { event := event76429
    frameStart := 0 },
  { event := event76430
    frameStart := 0 },
  { event := event76431
    frameStart := 0 }
]

def eventLeaf4777 : Array AnnotatedEvent := #[
  { event := event76432
    frameStart := 0 },
  { event := event76433
    frameStart := 0 },
  { event := event76434
    frameStart := 0 },
  { event := event76435
    frameStart := 0 },
  { event := event76436
    frameStart := 0 },
  { event := event76437
    frameStart := 0 },
  { event := event76438
    frameStart := 0 },
  { event := event76439
    frameStart := 0 },
  { event := event76440
    frameStart := 0 },
  { event := event76441
    frameStart := 0 },
  { event := event76442
    frameStart := 0 },
  { event := event76443
    frameStart := 0 },
  { event := event76444
    frameStart := 0 },
  { event := event76445
    frameStart := 0 },
  { event := event76446
    frameStart := 0 },
  { event := event76447
    frameStart := 0 }
]

def eventLeaf4778 : Array AnnotatedEvent := #[
  { event := event76448
    frameStart := 0 },
  { event := event76449
    frameStart := 0 },
  { event := event76450
    frameStart := 0 },
  { event := event76451
    frameStart := 0 },
  { event := event76452
    frameStart := 0 },
  { event := event76453
    frameStart := 0 },
  { event := event76454
    frameStart := 0 },
  { event := event76455
    frameStart := 0 },
  { event := event76456
    frameStart := 0 },
  { event := event76457
    frameStart := 0 },
  { event := event76458
    frameStart := 0 },
  { event := event76459
    frameStart := 0 },
  { event := event76460
    frameStart := 0 },
  { event := event76461
    frameStart := 0 },
  { event := event76462
    frameStart := 0 },
  { event := event76463
    frameStart := 0 }
]

def eventLeaf4779 : Array AnnotatedEvent := #[
  { event := event76464
    frameStart := 0 },
  { event := event76465
    frameStart := 0 },
  { event := event76466
    frameStart := 0 },
  { event := event76467
    frameStart := 0 },
  { event := event76468
    frameStart := 0 },
  { event := event76469
    frameStart := 0 },
  { event := event76470
    frameStart := 0 },
  { event := event76471
    frameStart := 76471 },
  { event := event76472
    frameStart := 76471 },
  { event := event76473
    frameStart := 76471 },
  { event := event76474
    frameStart := 76471 },
  { event := event76475
    frameStart := 76471 },
  { event := event76476
    frameStart := 76471 },
  { event := event76477
    frameStart := 76471 },
  { event := event76478
    frameStart := 76471 },
  { event := event76479
    frameStart := 76471 }
]

def eventLeaf4780 : Array AnnotatedEvent := #[
  { event := event76480
    frameStart := 76471 },
  { event := event76481
    frameStart := 76471 },
  { event := event76482
    frameStart := 76471 },
  { event := event76483
    frameStart := 76471 },
  { event := event76484
    frameStart := 76471 },
  { event := event76485
    frameStart := 76471 },
  { event := event76486
    frameStart := 76471 },
  { event := event76487
    frameStart := 76471 },
  { event := event76488
    frameStart := 76471 },
  { event := event76489
    frameStart := 76471 },
  { event := event76490
    frameStart := 76471 },
  { event := event76491
    frameStart := 76471 },
  { event := event76492
    frameStart := 76471 },
  { event := event76493
    frameStart := 76471 },
  { event := event76494
    frameStart := 76471 },
  { event := event76495
    frameStart := 76471 }
]

def eventLeaf4781 : Array AnnotatedEvent := #[
  { event := event76496
    frameStart := 76471 },
  { event := event76497
    frameStart := 76471 },
  { event := event76498
    frameStart := 76471 },
  { event := event76499
    frameStart := 76471 },
  { event := event76500
    frameStart := 76471 },
  { event := event76501
    frameStart := 76471 },
  { event := event76502
    frameStart := 76471 },
  { event := event76503
    frameStart := 76471 },
  { event := event76504
    frameStart := 76471 },
  { event := event76505
    frameStart := 76471 },
  { event := event76506
    frameStart := 76471 },
  { event := event76507
    frameStart := 76471 },
  { event := event76508
    frameStart := 76471 },
  { event := event76509
    frameStart := 76471 },
  { event := event76510
    frameStart := 76471 },
  { event := event76511
    frameStart := 76471 }
]

def eventLeaf4782 : Array AnnotatedEvent := #[
  { event := event76512
    frameStart := 76471 },
  { event := event76513
    frameStart := 76471 },
  { event := event76514
    frameStart := 76471 },
  { event := event76515
    frameStart := 76471 },
  { event := event76516
    frameStart := 76471 },
  { event := event76517
    frameStart := 76471 },
  { event := event76518
    frameStart := 76471 },
  { event := event76519
    frameStart := 76471 },
  { event := event76520
    frameStart := 76471 },
  { event := event76521
    frameStart := 76471 },
  { event := event76522
    frameStart := 76471 },
  { event := event76523
    frameStart := 76471 },
  { event := event76524
    frameStart := 76471 },
  { event := event76525
    frameStart := 76525 },
  { event := event76526
    frameStart := 76525 },
  { event := event76527
    frameStart := 76525 }
]

def eventLeaf4783 : Array AnnotatedEvent := #[
  { event := event76528
    frameStart := 76525 },
  { event := event76529
    frameStart := 76525 },
  { event := event76530
    frameStart := 76525 },
  { event := event76531
    frameStart := 76525 },
  { event := event76532
    frameStart := 76525 },
  { event := event76533
    frameStart := 76525 },
  { event := event76534
    frameStart := 76525 },
  { event := event76535
    frameStart := 76525 },
  { event := event76536
    frameStart := 76525 },
  { event := event76537
    frameStart := 76525 },
  { event := event76538
    frameStart := 76525 },
  { event := event76539
    frameStart := 76525 },
  { event := event76540
    frameStart := 76525 },
  { event := event76541
    frameStart := 76525 },
  { event := event76542
    frameStart := 76525 },
  { event := event76543
    frameStart := 76525 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events298
