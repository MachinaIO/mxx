import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events302

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event77312 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨43078⟩⟩, .operator (⟨77285, 0⟩, ⟨77308, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨43077⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact77313RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨43077⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact77313RawTermsValid :
    exact77313RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77313 : Event := .resultExact (⟨.program ⟨257⟩, ⟨43078⟩⟩) exact77313RawTerms .large 77311 .exactZero (none)

def event77314 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7228⟩⟩) 0 ⟨7177⟩ 77267

def event77315 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7228⟩⟩) (.authority (.operator))

def exact77316RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7228⟩⟩]⟩, (1)⟩]

theorem exact77316RawTermsValid :
    exact77316RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77316 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7228⟩⟩) exact77316RawTerms .large 77315 .exactZero (none)

def event77317 : Event := .predecessor (⟨.program ⟨257⟩, ⟨43079⟩⟩) 0 ⟨7228⟩ 77316

def event77318 : Event := .predecessor (⟨.program ⟨257⟩, ⟨43079⟩⟩) 1 ⟨43078⟩ 77313

def event77319 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨43079⟩⟩) (.sum [.predecessor 0 77317 .coefficient, .predecessor 1 77318 .coefficient])

def exact77320RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7228⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨43077⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact77320RawTermsValid :
    exact77320RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77320 : Event := .resultExact (⟨.program ⟨257⟩, ⟨43079⟩⟩) exact77320RawTerms .large 77319 .exactZero (none)

def event77321 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44823⟩⟩) 0 ⟨43079⟩ 77320

def event77322 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44823⟩⟩) 1 ⟨44820⟩ 77305

def event77323 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44823⟩⟩) (.sum [.predecessor 0 77321 .coefficient, .predecessor 1 77322 .coefficient])

def exact77324RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44819⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7228⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨42836⟩⟩], [⟨.program ⟨257⟩, ⟨43995⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨43077⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact77324RawTermsValid :
    exact77324RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77324 : Event := .resultExact (⟨.program ⟨257⟩, ⟨44823⟩⟩) exact77324RawTerms .large 77323 .exactZero (none)

def event77325 : Event := .preFoldPolynomial 77324 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44819⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7228⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨42836⟩⟩], [⟨.program ⟨257⟩, ⟨43995⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨43077⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact77326RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44819⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7228⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨42836⟩⟩], [⟨.program ⟨257⟩, ⟨43995⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨43077⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event77326 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨44823⟩⟩) 77325 exact77326RawTerms .large 77323 .exactZero (none)

def event77327 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨42837⟩⟩) ⟨⟨107⟩, ⟨90⟩, ⟨135⟩⟩ ⟨77169, 77327⟩

def event77328 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨43659⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43656⟩⟩]⟩) (1) 0 2 (.universal 77327 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43656⟩⟩]⟩) (none) 77326)

def event77329 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨43659⟩⟩, .relation 77328 1, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7228⟩⟩]⟩, (1)⟩)

def event77330 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨43659⟩⟩, .relation 77328 0, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44819⟩⟩]⟩, (-1)⟩)

def event77331 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨43659⟩⟩, .relation 77328 2, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨42836⟩⟩], [⟨.program ⟨257⟩, ⟨43995⟩⟩]⟩, (1)⟩)

def event77332 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨43659⟩⟩, .relation 77328 3, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨43077⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact77333RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44819⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7228⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨42836⟩⟩], [⟨.program ⟨257⟩, ⟨43995⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨43077⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact77333RawTermsValid :
    exact77333RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77333 : Event := .resultExact (⟨.program ⟨257⟩, ⟨43659⟩⟩) exact77333RawTerms .large 77165 (.finite 202072841853861888) (some (77167))

def event77334 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44822⟩⟩) 0 ⟨43659⟩ 77333

def event77335 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44822⟩⟩) 1 ⟨44821⟩ 77155

def event77336 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44822⟩⟩) (.sum [.predecessor 0 77334 .coefficient, .predecessor 1 77335 .coefficient])

def event77337 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨44822⟩⟩, .operator (⟨77333, 0⟩, ⟨77155, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44819⟩⟩]⟩, (1)⟩)

def event77338 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨44822⟩⟩, .operator (⟨77333, 2⟩, ⟨77155, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨42836⟩⟩], [⟨.program ⟨257⟩, ⟨43995⟩⟩]⟩, (-1)⟩)

def event77339 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44822⟩⟩) (.sum [.result 77333 .summary, .result 77155 .summary])

def exact77340RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7228⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨43077⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact77340RawTermsValid :
    exact77340RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77340 : Event := .resultExact (⟨.program ⟨257⟩, ⟨44822⟩⟩) exact77340RawTerms .large 77336 (.finite 32193718473625891320532869316608) (some (77339))

def event77341 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41313⟩⟩) 0 ⟨40157⟩ 3172

def event77342 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41313⟩⟩) (.authority (.programFamilyFact))

def event77343 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨41313⟩⟩) (.finite 3720)

def event77344 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41315⟩⟩) 0 ⟨7177⟩ 15500

def event77345 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41315⟩⟩) 1 ⟨41313⟩ 77343

def event77346 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41315⟩⟩) (.authority (.operator))

def exact77347RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨41315⟩⟩]⟩, (1)⟩]

theorem exact77347RawTermsValid :
    exact77347RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77347 : Event := .resultExact (⟨.program ⟨257⟩, ⟨41315⟩⟩) exact77347RawTerms .large 77346 .exactZero (none)

def event77348 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42139⟩⟩) 0 ⟨41315⟩ 77347

def event77349 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42139⟩⟩) (.authority (.operator))

def exact77350RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨42139⟩⟩]⟩, (1)⟩]

theorem exact77350RawTermsValid :
    exact77350RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77350 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42139⟩⟩) exact77350RawTerms (.finite 8192) 77349 .exactZero (none)

def event77351 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41144⟩⟩) 0 ⟨39940⟩ 3166

def event77352 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41144⟩⟩) (.authority (.programFamilyFact))

def event77353 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨41144⟩⟩) (.finite 3720)

def event77354 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41145⟩⟩) 0 ⟨7177⟩ 15500

def event77355 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41145⟩⟩) 1 ⟨41144⟩ 77353

def event77356 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41145⟩⟩) (.authority (.operator))

def exact77357RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨41145⟩⟩]⟩, (1)⟩]

theorem exact77357RawTermsValid :
    exact77357RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77357 : Event := .resultExact (⟨.program ⟨257⟩, ⟨41145⟩⟩) exact77357RawTerms .large 77356 .exactZero (none)

def event77358 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41685⟩⟩) 0 ⟨41145⟩ 77357

def event77359 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41685⟩⟩) (.authority (.operator))

def exact77360RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨41685⟩⟩]⟩, (1)⟩]

theorem exact77360RawTermsValid :
    exact77360RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77360 : Event := .resultExact (⟨.program ⟨257⟩, ⟨41685⟩⟩) exact77360RawTerms (.finite 8192) 77359 .exactZero (none)

def event77361 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39941⟩⟩) 0 ⟨39938⟩ 3155

def event77362 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39941⟩⟩) 1 ⟨10328⟩ 75903

def event77363 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39941⟩⟩) (.tensor (.predecessor 0 77361 .coefficient) (.predecessor 1 77362 .coefficient) true false)

def event77364 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨39941⟩⟩, .operator (⟨3155, 0⟩, ⟨75903, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨39938⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact77365RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨39938⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact77365RawTermsValid :
    exact77365RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77365 : Event := .resultExact (⟨.program ⟨257⟩, ⟨39941⟩⟩) exact77365RawTerms .large 77363 .exactZero (none)

def event77366 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10340⟩⟩) 0 ⟨10327⟩ 75773

def event77367 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10340⟩⟩) 1 ⟨7282⟩ 18583

def event77368 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10340⟩⟩) (.product (.predecessor 0 77366 .coefficient) (.predecessor 1 77367 .coefficient) (⟨false, false, none, none, none⟩))

def event77369 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨10340⟩⟩, .operator (⟨75773, 0⟩, ⟨18583, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7282⟩⟩]⟩, (1)⟩)

def exact77370RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7282⟩⟩]⟩, (1)⟩]

theorem exact77370RawTermsValid :
    exact77370RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77370 : Event := .resultExact (⟨.program ⟨257⟩, ⟨10340⟩⟩) exact77370RawTerms .large 77368 .exactZero (none)

def event77371 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39942⟩⟩) 0 ⟨10340⟩ 77370

def event77372 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39942⟩⟩) 1 ⟨39941⟩ 77365

def event77373 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39942⟩⟩) (.sum [.predecessor 0 77371 .coefficient, .predecessor 1 77372 .coefficient])

def exact77374RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7282⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨39938⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact77374RawTermsValid :
    exact77374RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77374 : Event := .resultExact (⟨.program ⟨257⟩, ⟨39942⟩⟩) exact77374RawTerms .large 77373 .exactZero (none)

def event77375 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39943⟩⟩) 0 ⟨39942⟩ 77374

def event77376 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39943⟩⟩) 1 ⟨108⟩ 18575

def event77377 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39943⟩⟩) (.sum [.predecessor 0 77375 .coefficient, .predecessor 1 77376 .coefficient])

def event77378 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39943⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨108⟩⟩]⟩) [⟨.result 18575 .coefficient, false, none⟩])

def event77379 : Event := .survivorFold (1) 77378

def exact77380RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7282⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨39938⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact77380RawTermsValid :
    exact77380RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77380 : Event := .resultExact (⟨.program ⟨257⟩, ⟨39943⟩⟩) exact77380RawTerms .large 77377 (.finite 26) (some (77378))

def event77381 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39944⟩⟩) 0 ⟨39943⟩ 77380

def event77382 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39944⟩⟩) 1 ⟨14271⟩ 3158

def event77383 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39944⟩⟩) (.product (.predecessor 0 77381 .coefficient) (.predecessor 1 77382 .coefficient) (⟨false, true, none, none, some 1⟩))

def event77384 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39944⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨14271⟩⟩], []⟩) [⟨.result 3158 .coefficient, true, some 1⟩])

def event77385 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39944⟩⟩) (.product (.result 77380 .summary) (.transfer 77384) (⟨false, false, none, none, none⟩))

def event77386 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨39944⟩⟩, .operator (⟨77380, 1⟩, ⟨3158, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨14271⟩⟩, ⟨.program ⟨257⟩, ⟨39938⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event77387 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨39944⟩⟩, .operator (⟨77380, 0⟩, ⟨3158, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨14271⟩⟩], [⟨.program ⟨257⟩, ⟨7282⟩⟩]⟩, (1)⟩)

def exact77388RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨14271⟩⟩], [⟨.program ⟨257⟩, ⟨7282⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨14271⟩⟩, ⟨.program ⟨257⟩, ⟨39938⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact77388RawTermsValid :
    exact77388RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77388 : Event := .resultExact (⟨.program ⟨257⟩, ⟨39944⟩⟩) exact77388RawTerms .large 77383 (.finite 39190528) (some (77385))

def event77389 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14272⟩⟩) 0 ⟨14271⟩ 3158

def event77390 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14272⟩⟩) 1 ⟨10328⟩ 75903

def event77391 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14272⟩⟩) (.tensor (.predecessor 0 77389 .coefficient) (.predecessor 1 77390 .coefficient) true false)

def event77392 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨14272⟩⟩, .operator (⟨3158, 0⟩, ⟨75903, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨14271⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact77393RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨14271⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact77393RawTermsValid :
    exact77393RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77393 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14272⟩⟩) exact77393RawTerms .large 77391 .exactZero (none)

def event77394 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10357⟩⟩) 0 ⟨10327⟩ 75773

def event77395 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10357⟩⟩) 1 ⟨7299⟩ 18624

def event77396 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10357⟩⟩) (.product (.predecessor 0 77394 .coefficient) (.predecessor 1 77395 .coefficient) (⟨false, false, none, none, none⟩))

def event77397 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨10357⟩⟩, .operator (⟨75773, 0⟩, ⟨18624, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7299⟩⟩]⟩, (1)⟩)

def exact77398RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7299⟩⟩]⟩, (1)⟩]

theorem exact77398RawTermsValid :
    exact77398RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77398 : Event := .resultExact (⟨.program ⟨257⟩, ⟨10357⟩⟩) exact77398RawTerms .large 77396 .exactZero (none)

def event77399 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14273⟩⟩) 0 ⟨10357⟩ 77398

def event77400 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14273⟩⟩) 1 ⟨14272⟩ 77393

def event77401 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14273⟩⟩) (.sum [.predecessor 0 77399 .coefficient, .predecessor 1 77400 .coefficient])

def exact77402RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7299⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨14271⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact77402RawTermsValid :
    exact77402RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77402 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14273⟩⟩) exact77402RawTerms .large 77401 .exactZero (none)

def event77403 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14274⟩⟩) 0 ⟨14273⟩ 77402

def event77404 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14274⟩⟩) 1 ⟨125⟩ 18616

def event77405 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14274⟩⟩) (.sum [.predecessor 0 77403 .coefficient, .predecessor 1 77404 .coefficient])

def event77406 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14274⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨125⟩⟩]⟩) [⟨.result 18616 .coefficient, false, none⟩])

def event77407 : Event := .survivorFold (1) 77406

def exact77408RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7299⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨14271⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact77408RawTermsValid :
    exact77408RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77408 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14274⟩⟩) exact77408RawTerms .large 77405 (.finite 26) (some (77406))

def event77409 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14275⟩⟩) 0 ⟨14274⟩ 77408

def event77410 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14275⟩⟩) 1 ⟨9557⟩ 18613

def event77411 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14275⟩⟩) (.product (.predecessor 0 77409 .coefficient) (.predecessor 1 77410 .coefficient) (⟨false, false, none, none, none⟩))

def event77412 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14275⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨9556⟩⟩]⟩) [⟨.result 18609 .coefficient, false, none⟩])

def event77413 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14275⟩⟩) (.product (.result 77408 .summary) (.transfer 77412) (⟨false, false, none, none, none⟩))

def event77414 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨14275⟩⟩, .operator (⟨77408, 1⟩, ⟨18613, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨14271⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩]⟩, (-1)⟩)

def event77415 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨14275⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨14271⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨9556⟩⟩) ⟨7282⟩ 18583)

def event77416 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨14275⟩⟩, .relation 77415 0, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨14271⟩⟩], [⟨.program ⟨257⟩, ⟨7282⟩⟩]⟩, (-1)⟩)

def event77417 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨14275⟩⟩, .operator (⟨77408, 0⟩, ⟨18613, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩]⟩, (1)⟩)

def exact77418RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨14271⟩⟩], [⟨.program ⟨257⟩, ⟨7282⟩⟩]⟩, (-1)⟩]

theorem exact77418RawTermsValid :
    exact77418RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77418 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14275⟩⟩) exact77418RawTerms .large 77411 (.finite 279172874240) (some (77413))

def event77419 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39945⟩⟩) 0 ⟨14275⟩ 77418

def event77420 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39945⟩⟩) 1 ⟨39944⟩ 77388

def event77421 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39945⟩⟩) (.sum [.predecessor 0 77419 .coefficient, .predecessor 1 77420 .coefficient])

def event77422 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨39945⟩⟩, .operator (⟨77418, 1⟩, ⟨77388, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨14271⟩⟩], [⟨.program ⟨257⟩, ⟨7282⟩⟩]⟩, (1)⟩)

def event77423 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39945⟩⟩) (.sum [.result 77418 .summary, .result 77388 .summary])

def exact77424RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨14271⟩⟩, ⟨.program ⟨257⟩, ⟨39938⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact77424RawTermsValid :
    exact77424RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77424 : Event := .resultExact (⟨.program ⟨257⟩, ⟨39945⟩⟩) exact77424RawTerms .large 77421 (.finite 279212064768) (some (77423))

def event77425 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41686⟩⟩) 0 ⟨39945⟩ 77424

def event77426 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41686⟩⟩) 1 ⟨41685⟩ 77360

def event77427 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41686⟩⟩) (.product (.predecessor 0 77425 .coefficient) (.predecessor 1 77426 .coefficient) (⟨false, false, none, none, none⟩))

def event77428 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41686⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨41685⟩⟩]⟩) [⟨.result 77360 .coefficient, false, none⟩])

def event77429 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41686⟩⟩) (.product (.result 77424 .summary) (.transfer 77428) (⟨false, false, none, none, none⟩))

def event77430 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨41686⟩⟩, .operator (⟨77424, 1⟩, ⟨77360, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨14271⟩⟩, ⟨.program ⟨257⟩, ⟨39938⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨41685⟩⟩]⟩, (-1)⟩)

def event77431 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨41686⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨14271⟩⟩, ⟨.program ⟨257⟩, ⟨39938⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨41685⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨41685⟩⟩) ⟨41145⟩ 77357)

def event77432 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨41686⟩⟩, .relation 77431 0, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨14271⟩⟩, ⟨.program ⟨257⟩, ⟨39938⟩⟩], [⟨.program ⟨257⟩, ⟨41145⟩⟩]⟩, (-1)⟩)

def event77433 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨41686⟩⟩, .operator (⟨77424, 0⟩, ⟨77360, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41685⟩⟩]⟩, (1)⟩)

def exact77434RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41685⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨14271⟩⟩, ⟨.program ⟨257⟩, ⟨39938⟩⟩], [⟨.program ⟨257⟩, ⟨41145⟩⟩]⟩, (-1)⟩]

theorem exact77434RawTermsValid :
    exact77434RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77434 : Event := .resultExact (⟨.program ⟨257⟩, ⟨41686⟩⟩) exact77434RawTerms .large 77427 (.finite 2998016717067984568320) (some (77429))

def event77435 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40609⟩⟩) 0 ⟨39940⟩ 3166

def event77436 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40609⟩⟩) (.authority (.relationPreimageSource ⟨51⟩))

def exact77437RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨40609⟩⟩]⟩, (1)⟩]

theorem exact77437RawTermsValid :
    exact77437RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77437 : Event := .resultExact (⟨.program ⟨257⟩, ⟨40609⟩⟩) exact77437RawTerms (.finite 5647228698) 77436 .exactZero (none)

def event77438 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40611⟩⟩) 0 ⟨40609⟩ 77437

def event77439 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40611⟩⟩) 1 ⟨2370⟩ 4

def event77440 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40611⟩⟩) (.scale (.predecessor 0 77438 .coefficient) (.value (.predecessor 1 77439 .coefficient)))

def exact77441RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨40609⟩⟩]⟩, (1)⟩]

theorem exact77441RawTermsValid :
    exact77441RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77441 : Event := .resultExact (⟨.program ⟨257⟩, ⟨40611⟩⟩) exact77441RawTerms (.finite 5647228698) 77440 .exactZero (none)

def event77442 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40612⟩⟩) 0 ⟨10368⟩ 75995

def event77443 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40612⟩⟩) 1 ⟨40611⟩ 77441

def event77444 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40612⟩⟩) (.product (.predecessor 0 77442 .coefficient) (.predecessor 1 77443 .coefficient) (⟨false, false, none, none, none⟩))

def event77445 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40612⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨40609⟩⟩]⟩) [⟨.result 77437 .coefficient, false, none⟩])

def event77446 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40612⟩⟩) (.product (.result 75995 .summary) (.transfer 77445) (⟨false, false, none, none, none⟩))

def event77447 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨40612⟩⟩, .operator (⟨75995, 0⟩, ⟨77441, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40609⟩⟩]⟩, (1)⟩)

def event77448 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨40610⟩⟩)

def event77449 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event77450 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event77451 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10267⟩⟩) (.authority (.operator))

def event77452 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10267⟩⟩) (.finite 15)

def event77453 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event77454 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event77455 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event77456 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event77457 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 77456

def event77458 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 77454

def event77459 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 77457 .coefficient) (.value (.predecessor 1 77458 .coefficient)))

def event77460 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event77461 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10269⟩⟩) 0 ⟨392⟩ 77460

def event77462 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10269⟩⟩) 1 ⟨10267⟩ 77452

def event77463 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10269⟩⟩) (.sum [.predecessor 0 77461 .coefficient, .predecessor 1 77462 .coefficient])

def event77464 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10269⟩⟩) (.finite 655355)

def event77465 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10325⟩⟩) 0 ⟨10269⟩ 77464

def event77466 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10325⟩⟩) 1 ⟨5426⟩ 77450

def event77467 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10325⟩⟩) (.identity (.predecessor 1 77466 .coefficient))

def event77468 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10325⟩⟩) (.finite 655360)

def event77469 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39938⟩⟩) 0 ⟨10325⟩ 77468

def event77470 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39938⟩⟩) (.authority (.programFamilyFact))

def exact77471RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨39938⟩⟩], []⟩, (1)⟩]

theorem exact77471RawTermsValid :
    exact77471RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77471 : Event := .resultExact (⟨.program ⟨257⟩, ⟨39938⟩⟩) exact77471RawTerms (.finite 46) 77470 .exactZero (none)

def event77472 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14271⟩⟩) 0 ⟨10325⟩ 77468

def event77473 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14271⟩⟩) (.authority (.programFamilyFact))

def exact77474RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14271⟩⟩], []⟩, (1)⟩]

theorem exact77474RawTermsValid :
    exact77474RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77474 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14271⟩⟩) exact77474RawTerms (.finite 46) 77473 .exactZero (none)

def event77475 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39939⟩⟩) 0 ⟨14271⟩ 77474

def event77476 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39939⟩⟩) 1 ⟨39938⟩ 77471

def event77477 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39939⟩⟩) (.product (.predecessor 0 77475 .coefficient) (.predecessor 1 77476 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event77478 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39939⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨14271⟩⟩, ⟨.program ⟨257⟩, ⟨39938⟩⟩], []⟩) [⟨.result 77474 .coefficient, true, some 1⟩, ⟨.result 77471 .coefficient, true, some 1⟩])

def event77479 : Event := .survivorFold (1) 77478

def exact77480RawTerms : List Term := []

theorem exact77480RawTermsValid :
    exact77480RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77480 : Event := .resultExact (⟨.program ⟨257⟩, ⟨39939⟩⟩) exact77480RawTerms (.finite 2116) 77477 (.finite 2116) (some (77478))

def event77481 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39940⟩⟩) 0 ⟨39939⟩ 77480

def event77482 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39940⟩⟩) (.identity (.predecessor 0 77481 .coefficient))

def event77483 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨39940⟩⟩) (.finite 2116)

def event77484 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40609⟩⟩) 0 ⟨39940⟩ 77483

def event77485 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40609⟩⟩) (.authority (.relationPreimageSource ⟨51⟩))

def exact77486RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨40609⟩⟩]⟩, (1)⟩]

theorem exact77486RawTermsValid :
    exact77486RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77486 : Event := .resultExact (⟨.program ⟨257⟩, ⟨40609⟩⟩) exact77486RawTerms (.finite 5647228698) 77485 .exactZero (none)

def event77487 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35⟩⟩) (.authority (.operator))

def exact77488RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact77488RawTermsValid :
    exact77488RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77488 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35⟩⟩) exact77488RawTerms .large 77487 .exactZero (none)

def event77489 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40610⟩⟩) 0 ⟨35⟩ 77488

def event77490 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40610⟩⟩) 1 ⟨40609⟩ 77486

def event77491 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40610⟩⟩) (.product (.predecessor 0 77489 .coefficient) (.predecessor 1 77490 .coefficient) (⟨false, false, none, none, none⟩))

def event77492 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨40610⟩⟩, .operator (⟨77488, 0⟩, ⟨77486, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40609⟩⟩]⟩, (1)⟩)

def exact77493RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40609⟩⟩]⟩, (1)⟩]

theorem exact77493RawTermsValid :
    exact77493RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77493 : Event := .resultExact (⟨.program ⟨257⟩, ⟨40610⟩⟩) exact77493RawTerms .large 77491 .exactZero (none)

def event77494 : Event := .preFoldPolynomial 77493 [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40609⟩⟩]⟩, (1)⟩] .exactZero none

def exact77495RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40609⟩⟩]⟩, (1)⟩]

def event77495 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨40610⟩⟩) 77494 exact77495RawTerms .large 77491 .exactZero (none)

def event77496 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨41689⟩⟩)

def event77497 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event77498 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event77499 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10267⟩⟩) (.authority (.operator))

def event77500 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10267⟩⟩) (.finite 15)

def event77501 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event77502 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event77503 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event77504 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event77505 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 77504

def event77506 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 77502

def event77507 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 77505 .coefficient) (.value (.predecessor 1 77506 .coefficient)))

def event77508 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event77509 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10269⟩⟩) 0 ⟨392⟩ 77508

def event77510 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10269⟩⟩) 1 ⟨10267⟩ 77500

def event77511 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10269⟩⟩) (.sum [.predecessor 0 77509 .coefficient, .predecessor 1 77510 .coefficient])

def event77512 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10269⟩⟩) (.finite 655355)

def event77513 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10325⟩⟩) 0 ⟨10269⟩ 77512

def event77514 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10325⟩⟩) 1 ⟨5426⟩ 77498

def event77515 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10325⟩⟩) (.identity (.predecessor 1 77514 .coefficient))

def event77516 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10325⟩⟩) (.finite 655360)

def event77517 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39938⟩⟩) 0 ⟨10325⟩ 77516

def event77518 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39938⟩⟩) (.authority (.programFamilyFact))

def exact77519RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨39938⟩⟩], []⟩, (1)⟩]

theorem exact77519RawTermsValid :
    exact77519RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77519 : Event := .resultExact (⟨.program ⟨257⟩, ⟨39938⟩⟩) exact77519RawTerms (.finite 46) 77518 .exactZero (none)

def event77520 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14271⟩⟩) 0 ⟨10325⟩ 77516

def event77521 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14271⟩⟩) (.authority (.programFamilyFact))

def exact77522RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14271⟩⟩], []⟩, (1)⟩]

theorem exact77522RawTermsValid :
    exact77522RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77522 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14271⟩⟩) exact77522RawTerms (.finite 46) 77521 .exactZero (none)

def event77523 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39939⟩⟩) 0 ⟨14271⟩ 77522

def event77524 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39939⟩⟩) 1 ⟨39938⟩ 77519

def event77525 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39939⟩⟩) (.product (.predecessor 0 77523 .coefficient) (.predecessor 1 77524 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event77526 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨39939⟩⟩, .operator (⟨77522, 0⟩, ⟨77519, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨14271⟩⟩, ⟨.program ⟨257⟩, ⟨39938⟩⟩], []⟩, (1)⟩)

def exact77527RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14271⟩⟩, ⟨.program ⟨257⟩, ⟨39938⟩⟩], []⟩, (1)⟩]

theorem exact77527RawTermsValid :
    exact77527RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77527 : Event := .resultExact (⟨.program ⟨257⟩, ⟨39939⟩⟩) exact77527RawTerms (.finite 2116) 77525 .exactZero (none)

def event77528 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39940⟩⟩) 0 ⟨39939⟩ 77527

def event77529 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39940⟩⟩) (.identity (.predecessor 0 77528 .coefficient))

def event77530 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨39940⟩⟩) (.finite 2116)

def event77531 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41144⟩⟩) 0 ⟨39940⟩ 77530

def event77532 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41144⟩⟩) (.authority (.programFamilyFact))

def event77533 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨41144⟩⟩) (.finite 3720)

def event77534 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨7177⟩⟩) .missing

def event77535 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41145⟩⟩) 0 ⟨7177⟩ 77534

def event77536 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41145⟩⟩) 1 ⟨41144⟩ 77533

def event77537 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41145⟩⟩) (.authority (.operator))

def exact77538RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨41145⟩⟩]⟩, (1)⟩]

theorem exact77538RawTermsValid :
    exact77538RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77538 : Event := .resultExact (⟨.program ⟨257⟩, ⟨41145⟩⟩) exact77538RawTerms .large 77537 .exactZero (none)

def event77539 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41685⟩⟩) 0 ⟨41145⟩ 77538

def event77540 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41685⟩⟩) (.authority (.operator))

def exact77541RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨41685⟩⟩]⟩, (1)⟩]

theorem exact77541RawTermsValid :
    exact77541RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77541 : Event := .resultExact (⟨.program ⟨257⟩, ⟨41685⟩⟩) exact77541RawTerms (.finite 8192) 77540 .exactZero (none)

def event77542 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event77543 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event77544 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41410⟩⟩) 0 ⟨39940⟩ 77530

def event77545 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41410⟩⟩) 1 ⟨136⟩ 77543

def event77546 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41410⟩⟩) (.sum [.predecessor 0 77544 .coefficient, .predecessor 1 77545 .coefficient])

def event77547 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨41410⟩⟩) (.finite 2116)

def event77548 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41411⟩⟩) 0 ⟨41410⟩ 77547

def event77549 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41411⟩⟩) (.identity (.predecessor 0 77548 .coefficient))

def exact77550RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14271⟩⟩, ⟨.program ⟨257⟩, ⟨39938⟩⟩], []⟩, (1)⟩]

theorem exact77550RawTermsValid :
    exact77550RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77550 : Event := .resultExact (⟨.program ⟨257⟩, ⟨41411⟩⟩) exact77550RawTerms (.finite 2116) 77549 .exactZero (none)

def event77551 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact77552RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact77552RawTermsValid :
    exact77552RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77552 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact77552RawTerms .large 77551 .exactZero (none)

def event77553 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41412⟩⟩) 0 ⟨6908⟩ 77552

def event77554 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41412⟩⟩) 1 ⟨41411⟩ 77550

def event77555 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41412⟩⟩) (.product (.predecessor 0 77553 .coefficient) (.predecessor 1 77554 .coefficient) (⟨false, false, none, none, none⟩))

def event77556 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨41412⟩⟩, .operator (⟨77552, 0⟩, ⟨77550, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨14271⟩⟩, ⟨.program ⟨257⟩, ⟨39938⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact77557RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14271⟩⟩, ⟨.program ⟨257⟩, ⟨39938⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact77557RawTermsValid :
    exact77557RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77557 : Event := .resultExact (⟨.program ⟨257⟩, ⟨41412⟩⟩) exact77557RawTerms .large 77555 .exactZero (none)

def event77558 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨2370⟩⟩) (.authority (.operator))

def event77559 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨2370⟩⟩) (.finite 1)

def event77560 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7178⟩⟩) 0 ⟨7177⟩ 77534

def event77561 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7178⟩⟩) (.authority (.operator))

def exact77562RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7178⟩⟩]⟩, (1)⟩]

theorem exact77562RawTermsValid :
    exact77562RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77562 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7178⟩⟩) exact77562RawTerms .large 77561 .exactZero (none)

def event77563 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7282⟩⟩) 0 ⟨7178⟩ 77562

def event77564 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7282⟩⟩) (.identity (.predecessor 0 77563 .coefficient))

def exact77565RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7282⟩⟩]⟩, (1)⟩]

theorem exact77565RawTermsValid :
    exact77565RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77565 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7282⟩⟩) exact77565RawTerms .large 77564 .exactZero (none)

def event77566 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9556⟩⟩) 0 ⟨7282⟩ 77565

def event77567 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9556⟩⟩) (.authority (.operator))

def eventLeaf4832 : Array AnnotatedEvent := #[
  { event := event77312
    frameStart := 77223 },
  { event := event77313
    frameStart := 77223 },
  { event := event77314
    frameStart := 77223 },
  { event := event77315
    frameStart := 77223 },
  { event := event77316
    frameStart := 77223 },
  { event := event77317
    frameStart := 77223 },
  { event := event77318
    frameStart := 77223 },
  { event := event77319
    frameStart := 77223 },
  { event := event77320
    frameStart := 77223 },
  { event := event77321
    frameStart := 77223 },
  { event := event77322
    frameStart := 77223 },
  { event := event77323
    frameStart := 77223 },
  { event := event77324
    frameStart := 77223 },
  { event := event77325
    frameStart := 77223 },
  { event := event77326
    frameStart := 77223 },
  { event := event77327
    frameStart := 0 }
]

def eventLeaf4833 : Array AnnotatedEvent := #[
  { event := event77328
    frameStart := 0 },
  { event := event77329
    frameStart := 0 },
  { event := event77330
    frameStart := 0 },
  { event := event77331
    frameStart := 0 },
  { event := event77332
    frameStart := 0 },
  { event := event77333
    frameStart := 0 },
  { event := event77334
    frameStart := 0 },
  { event := event77335
    frameStart := 0 },
  { event := event77336
    frameStart := 0 },
  { event := event77337
    frameStart := 0 },
  { event := event77338
    frameStart := 0 },
  { event := event77339
    frameStart := 0 },
  { event := event77340
    frameStart := 0 },
  { event := event77341
    frameStart := 0 },
  { event := event77342
    frameStart := 0 },
  { event := event77343
    frameStart := 0 }
]

def eventLeaf4834 : Array AnnotatedEvent := #[
  { event := event77344
    frameStart := 0 },
  { event := event77345
    frameStart := 0 },
  { event := event77346
    frameStart := 0 },
  { event := event77347
    frameStart := 0 },
  { event := event77348
    frameStart := 0 },
  { event := event77349
    frameStart := 0 },
  { event := event77350
    frameStart := 0 },
  { event := event77351
    frameStart := 0 },
  { event := event77352
    frameStart := 0 },
  { event := event77353
    frameStart := 0 },
  { event := event77354
    frameStart := 0 },
  { event := event77355
    frameStart := 0 },
  { event := event77356
    frameStart := 0 },
  { event := event77357
    frameStart := 0 },
  { event := event77358
    frameStart := 0 },
  { event := event77359
    frameStart := 0 }
]

def eventLeaf4835 : Array AnnotatedEvent := #[
  { event := event77360
    frameStart := 0 },
  { event := event77361
    frameStart := 0 },
  { event := event77362
    frameStart := 0 },
  { event := event77363
    frameStart := 0 },
  { event := event77364
    frameStart := 0 },
  { event := event77365
    frameStart := 0 },
  { event := event77366
    frameStart := 0 },
  { event := event77367
    frameStart := 0 },
  { event := event77368
    frameStart := 0 },
  { event := event77369
    frameStart := 0 },
  { event := event77370
    frameStart := 0 },
  { event := event77371
    frameStart := 0 },
  { event := event77372
    frameStart := 0 },
  { event := event77373
    frameStart := 0 },
  { event := event77374
    frameStart := 0 },
  { event := event77375
    frameStart := 0 }
]

def eventLeaf4836 : Array AnnotatedEvent := #[
  { event := event77376
    frameStart := 0 },
  { event := event77377
    frameStart := 0 },
  { event := event77378
    frameStart := 0 },
  { event := event77379
    frameStart := 0 },
  { event := event77380
    frameStart := 0 },
  { event := event77381
    frameStart := 0 },
  { event := event77382
    frameStart := 0 },
  { event := event77383
    frameStart := 0 },
  { event := event77384
    frameStart := 0 },
  { event := event77385
    frameStart := 0 },
  { event := event77386
    frameStart := 0 },
  { event := event77387
    frameStart := 0 },
  { event := event77388
    frameStart := 0 },
  { event := event77389
    frameStart := 0 },
  { event := event77390
    frameStart := 0 },
  { event := event77391
    frameStart := 0 }
]

def eventLeaf4837 : Array AnnotatedEvent := #[
  { event := event77392
    frameStart := 0 },
  { event := event77393
    frameStart := 0 },
  { event := event77394
    frameStart := 0 },
  { event := event77395
    frameStart := 0 },
  { event := event77396
    frameStart := 0 },
  { event := event77397
    frameStart := 0 },
  { event := event77398
    frameStart := 0 },
  { event := event77399
    frameStart := 0 },
  { event := event77400
    frameStart := 0 },
  { event := event77401
    frameStart := 0 },
  { event := event77402
    frameStart := 0 },
  { event := event77403
    frameStart := 0 },
  { event := event77404
    frameStart := 0 },
  { event := event77405
    frameStart := 0 },
  { event := event77406
    frameStart := 0 },
  { event := event77407
    frameStart := 0 }
]

def eventLeaf4838 : Array AnnotatedEvent := #[
  { event := event77408
    frameStart := 0 },
  { event := event77409
    frameStart := 0 },
  { event := event77410
    frameStart := 0 },
  { event := event77411
    frameStart := 0 },
  { event := event77412
    frameStart := 0 },
  { event := event77413
    frameStart := 0 },
  { event := event77414
    frameStart := 0 },
  { event := event77415
    frameStart := 0 },
  { event := event77416
    frameStart := 0 },
  { event := event77417
    frameStart := 0 },
  { event := event77418
    frameStart := 0 },
  { event := event77419
    frameStart := 0 },
  { event := event77420
    frameStart := 0 },
  { event := event77421
    frameStart := 0 },
  { event := event77422
    frameStart := 0 },
  { event := event77423
    frameStart := 0 }
]

def eventLeaf4839 : Array AnnotatedEvent := #[
  { event := event77424
    frameStart := 0 },
  { event := event77425
    frameStart := 0 },
  { event := event77426
    frameStart := 0 },
  { event := event77427
    frameStart := 0 },
  { event := event77428
    frameStart := 0 },
  { event := event77429
    frameStart := 0 },
  { event := event77430
    frameStart := 0 },
  { event := event77431
    frameStart := 0 },
  { event := event77432
    frameStart := 0 },
  { event := event77433
    frameStart := 0 },
  { event := event77434
    frameStart := 0 },
  { event := event77435
    frameStart := 0 },
  { event := event77436
    frameStart := 0 },
  { event := event77437
    frameStart := 0 },
  { event := event77438
    frameStart := 0 },
  { event := event77439
    frameStart := 0 }
]

def eventLeaf4840 : Array AnnotatedEvent := #[
  { event := event77440
    frameStart := 0 },
  { event := event77441
    frameStart := 0 },
  { event := event77442
    frameStart := 0 },
  { event := event77443
    frameStart := 0 },
  { event := event77444
    frameStart := 0 },
  { event := event77445
    frameStart := 0 },
  { event := event77446
    frameStart := 0 },
  { event := event77447
    frameStart := 0 },
  { event := event77448
    frameStart := 77448 },
  { event := event77449
    frameStart := 77448 },
  { event := event77450
    frameStart := 77448 },
  { event := event77451
    frameStart := 77448 },
  { event := event77452
    frameStart := 77448 },
  { event := event77453
    frameStart := 77448 },
  { event := event77454
    frameStart := 77448 },
  { event := event77455
    frameStart := 77448 }
]

def eventLeaf4841 : Array AnnotatedEvent := #[
  { event := event77456
    frameStart := 77448 },
  { event := event77457
    frameStart := 77448 },
  { event := event77458
    frameStart := 77448 },
  { event := event77459
    frameStart := 77448 },
  { event := event77460
    frameStart := 77448 },
  { event := event77461
    frameStart := 77448 },
  { event := event77462
    frameStart := 77448 },
  { event := event77463
    frameStart := 77448 },
  { event := event77464
    frameStart := 77448 },
  { event := event77465
    frameStart := 77448 },
  { event := event77466
    frameStart := 77448 },
  { event := event77467
    frameStart := 77448 },
  { event := event77468
    frameStart := 77448 },
  { event := event77469
    frameStart := 77448 },
  { event := event77470
    frameStart := 77448 },
  { event := event77471
    frameStart := 77448 }
]

def eventLeaf4842 : Array AnnotatedEvent := #[
  { event := event77472
    frameStart := 77448 },
  { event := event77473
    frameStart := 77448 },
  { event := event77474
    frameStart := 77448 },
  { event := event77475
    frameStart := 77448 },
  { event := event77476
    frameStart := 77448 },
  { event := event77477
    frameStart := 77448 },
  { event := event77478
    frameStart := 77448 },
  { event := event77479
    frameStart := 77448 },
  { event := event77480
    frameStart := 77448 },
  { event := event77481
    frameStart := 77448 },
  { event := event77482
    frameStart := 77448 },
  { event := event77483
    frameStart := 77448 },
  { event := event77484
    frameStart := 77448 },
  { event := event77485
    frameStart := 77448 },
  { event := event77486
    frameStart := 77448 },
  { event := event77487
    frameStart := 77448 }
]

def eventLeaf4843 : Array AnnotatedEvent := #[
  { event := event77488
    frameStart := 77448 },
  { event := event77489
    frameStart := 77448 },
  { event := event77490
    frameStart := 77448 },
  { event := event77491
    frameStart := 77448 },
  { event := event77492
    frameStart := 77448 },
  { event := event77493
    frameStart := 77448 },
  { event := event77494
    frameStart := 77448 },
  { event := event77495
    frameStart := 77448 },
  { event := event77496
    frameStart := 77496 },
  { event := event77497
    frameStart := 77496 },
  { event := event77498
    frameStart := 77496 },
  { event := event77499
    frameStart := 77496 },
  { event := event77500
    frameStart := 77496 },
  { event := event77501
    frameStart := 77496 },
  { event := event77502
    frameStart := 77496 },
  { event := event77503
    frameStart := 77496 }
]

def eventLeaf4844 : Array AnnotatedEvent := #[
  { event := event77504
    frameStart := 77496 },
  { event := event77505
    frameStart := 77496 },
  { event := event77506
    frameStart := 77496 },
  { event := event77507
    frameStart := 77496 },
  { event := event77508
    frameStart := 77496 },
  { event := event77509
    frameStart := 77496 },
  { event := event77510
    frameStart := 77496 },
  { event := event77511
    frameStart := 77496 },
  { event := event77512
    frameStart := 77496 },
  { event := event77513
    frameStart := 77496 },
  { event := event77514
    frameStart := 77496 },
  { event := event77515
    frameStart := 77496 },
  { event := event77516
    frameStart := 77496 },
  { event := event77517
    frameStart := 77496 },
  { event := event77518
    frameStart := 77496 },
  { event := event77519
    frameStart := 77496 }
]

def eventLeaf4845 : Array AnnotatedEvent := #[
  { event := event77520
    frameStart := 77496 },
  { event := event77521
    frameStart := 77496 },
  { event := event77522
    frameStart := 77496 },
  { event := event77523
    frameStart := 77496 },
  { event := event77524
    frameStart := 77496 },
  { event := event77525
    frameStart := 77496 },
  { event := event77526
    frameStart := 77496 },
  { event := event77527
    frameStart := 77496 },
  { event := event77528
    frameStart := 77496 },
  { event := event77529
    frameStart := 77496 },
  { event := event77530
    frameStart := 77496 },
  { event := event77531
    frameStart := 77496 },
  { event := event77532
    frameStart := 77496 },
  { event := event77533
    frameStart := 77496 },
  { event := event77534
    frameStart := 77496 },
  { event := event77535
    frameStart := 77496 }
]

def eventLeaf4846 : Array AnnotatedEvent := #[
  { event := event77536
    frameStart := 77496 },
  { event := event77537
    frameStart := 77496 },
  { event := event77538
    frameStart := 77496 },
  { event := event77539
    frameStart := 77496 },
  { event := event77540
    frameStart := 77496 },
  { event := event77541
    frameStart := 77496 },
  { event := event77542
    frameStart := 77496 },
  { event := event77543
    frameStart := 77496 },
  { event := event77544
    frameStart := 77496 },
  { event := event77545
    frameStart := 77496 },
  { event := event77546
    frameStart := 77496 },
  { event := event77547
    frameStart := 77496 },
  { event := event77548
    frameStart := 77496 },
  { event := event77549
    frameStart := 77496 },
  { event := event77550
    frameStart := 77496 },
  { event := event77551
    frameStart := 77496 }
]

def eventLeaf4847 : Array AnnotatedEvent := #[
  { event := event77552
    frameStart := 77496 },
  { event := event77553
    frameStart := 77496 },
  { event := event77554
    frameStart := 77496 },
  { event := event77555
    frameStart := 77496 },
  { event := event77556
    frameStart := 77496 },
  { event := event77557
    frameStart := 77496 },
  { event := event77558
    frameStart := 77496 },
  { event := event77559
    frameStart := 77496 },
  { event := event77560
    frameStart := 77496 },
  { event := event77561
    frameStart := 77496 },
  { event := event77562
    frameStart := 77496 },
  { event := event77563
    frameStart := 77496 },
  { event := event77564
    frameStart := 77496 },
  { event := event77565
    frameStart := 77496 },
  { event := event77566
    frameStart := 77496 },
  { event := event77567
    frameStart := 77496 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events302
