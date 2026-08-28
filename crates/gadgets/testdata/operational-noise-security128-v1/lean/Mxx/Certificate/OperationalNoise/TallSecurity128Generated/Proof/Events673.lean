import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events673

open Mxx.Certificate.OperationalNoise
open CertificateABI

def exact172288RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨16099⟩⟩], []⟩, (1)⟩]

theorem exact172288RawTermsValid :
    exact172288RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event172288 : Event := .resultExact (⟨.program ⟨257⟩, ⟨16099⟩⟩) exact172288RawTerms (.finite 43) 172287 .exactZero (none)

def event172289 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16100⟩⟩) 0 ⟨6908⟩ 172265

def event172290 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16100⟩⟩) 1 ⟨16099⟩ 172288

def event172291 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16100⟩⟩) (.product (.predecessor 0 172289 .coefficient) (.predecessor 1 172290 .coefficient) (⟨false, true, none, none, some 1⟩))

def event172292 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16100⟩⟩, .operator (⟨172265, 0⟩, ⟨172288, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨16099⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact172293RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨16099⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact172293RawTermsValid :
    exact172293RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event172293 : Event := .resultExact (⟨.program ⟨257⟩, ⟨16100⟩⟩) exact172293RawTerms .large 172291 .exactZero (none)

def event172294 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7198⟩⟩) 0 ⟨7177⟩ 172247

def event172295 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7198⟩⟩) (.authority (.operator))

def exact172296RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩]

theorem exact172296RawTermsValid :
    exact172296RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event172296 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7198⟩⟩) exact172296RawTerms .large 172295 .exactZero (none)

def event172297 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16101⟩⟩) 0 ⟨7198⟩ 172296

def event172298 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16101⟩⟩) 1 ⟨16100⟩ 172293

def event172299 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16101⟩⟩) (.sum [.predecessor 0 172297 .coefficient, .predecessor 1 172298 .coefficient])

def exact172300RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨16099⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact172300RawTermsValid :
    exact172300RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event172300 : Event := .resultExact (⟨.program ⟨257⟩, ⟨16101⟩⟩) exact172300RawTerms .large 172299 .exactZero (none)

def event172301 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17877⟩⟩) 0 ⟨16101⟩ 172300

def event172302 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17877⟩⟩) 1 ⟨17874⟩ 172285

def event172303 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17877⟩⟩) (.sum [.predecessor 0 172301 .coefficient, .predecessor 1 172302 .coefficient])

def exact172304RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17873⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15820⟩⟩], [⟨.program ⟨257⟩, ⟨17037⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨16099⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact172304RawTermsValid :
    exact172304RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event172304 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17877⟩⟩) exact172304RawTerms .large 172303 .exactZero (none)

def event172305 : Event := .preFoldPolynomial 172304 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17873⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15820⟩⟩], [⟨.program ⟨257⟩, ⟨17037⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨16099⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact172306RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17873⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15820⟩⟩], [⟨.program ⟨257⟩, ⟨17037⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨16099⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event172306 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨17877⟩⟩) 172305 exact172306RawTerms .large 172303 .exactZero (none)

def event172307 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨15821⟩⟩) ⟨⟨77⟩, ⟨57⟩, ⟨135⟩⟩ ⟨172149, 172307⟩

def event172308 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨16679⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16676⟩⟩]⟩) (1) 0 2 (.universal 172307 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16676⟩⟩]⟩) (none) 172306)

def event172309 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16679⟩⟩, .relation 172308 1, ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩)

def event172310 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16679⟩⟩, .relation 172308 0, ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17873⟩⟩]⟩, (-1)⟩)

def event172311 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16679⟩⟩, .relation 172308 2, ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨15820⟩⟩], [⟨.program ⟨257⟩, ⟨17037⟩⟩]⟩, (1)⟩)

def event172312 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16679⟩⟩, .relation 172308 3, ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨16099⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact172313RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17873⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨15820⟩⟩], [⟨.program ⟨257⟩, ⟨17037⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨16099⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact172313RawTermsValid :
    exact172313RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event172313 : Event := .resultExact (⟨.program ⟨257⟩, ⟨16679⟩⟩) exact172313RawTerms .large 172145 (.finite 202072841853861888) (some (172147))

def event172314 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17876⟩⟩) 0 ⟨16679⟩ 172313

def event172315 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17876⟩⟩) 1 ⟨17875⟩ 172135

def event172316 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17876⟩⟩) (.sum [.predecessor 0 172314 .coefficient, .predecessor 1 172315 .coefficient])

def event172317 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17876⟩⟩, .operator (⟨172313, 0⟩, ⟨172135, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17873⟩⟩]⟩, (1)⟩)

def event172318 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17876⟩⟩, .operator (⟨172313, 2⟩, ⟨172135, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨15820⟩⟩], [⟨.program ⟨257⟩, ⟨17037⟩⟩]⟩, (-1)⟩)

def event172319 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17876⟩⟩) (.sum [.result 172313 .summary, .result 172135 .summary])

def exact172320RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨16099⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact172320RawTermsValid :
    exact172320RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event172320 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17876⟩⟩) exact172320RawTerms .large 172316 (.finite 32188807212483706889510625476608) (some (172319))

def event172321 : Event := .predecessor (⟨.program ⟨257⟩, ⟨20780⟩⟩) 0 ⟨17876⟩ 172320

def event172322 : Event := .predecessor (⟨.program ⟨257⟩, ⟨20780⟩⟩) 1 ⟨20779⟩ 171838

def event172323 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨20780⟩⟩) (.sum [.predecessor 0 172321 .coefficient, .predecessor 1 172322 .coefficient])

def event172324 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨20780⟩⟩) (.sum [.result 172320 .summary, .result 171838 .summary])

def exact172325RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨16099⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨18942⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact172325RawTermsValid :
    exact172325RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event172325 : Event := .resultExact (⟨.program ⟨257⟩, ⟨20780⟩⟩) exact172325RawTerms .large 172323 (.finite 64377712650190257467641695830016) (some (172324))

def event172326 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24000⟩⟩) 0 ⟨20780⟩ 172325

def event172327 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24000⟩⟩) 1 ⟨23999⟩ 171356

def event172328 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨24000⟩⟩) (.sum [.predecessor 0 172326 .coefficient, .predecessor 1 172327 .coefficient])

def event172329 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨24000⟩⟩) (.sum [.result 172325 .summary, .result 171356 .summary])

def exact172330RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨16099⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨18942⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨22162⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact172330RawTermsValid :
    exact172330RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event172330 : Event := .resultExact (⟨.program ⟨257⟩, ⟨24000⟩⟩) exact172330RawTerms .large 172328 (.finite 96566716313119651734393211060224) (some (172329))

def event172331 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34020⟩⟩) 0 ⟨24000⟩ 172330

def event172332 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34020⟩⟩) 1 ⟨34019⟩ 170874

def event172333 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34020⟩⟩) (.sum [.predecessor 0 172331 .coefficient, .predecessor 1 172332 .coefficient])

def event172334 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34020⟩⟩) (.sum [.result 172330 .summary, .result 170874 .summary])

def exact172335RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨16099⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨18942⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨22162⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨32182⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact172335RawTermsValid :
    exact172335RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event172335 : Event := .resultExact (⟨.program ⟨257⟩, ⟨34020⟩⟩) exact172335RawTerms .large 172333 (.finite 128755916426494733378385616044032) (some (172334))

def event172336 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53080⟩⟩) 0 ⟨34020⟩ 172335

def event172337 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53080⟩⟩) 1 ⟨53079⟩ 170392

def event172338 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53080⟩⟩) (.sum [.predecessor 0 172336 .coefficient, .predecessor 1 172337 .coefficient])

def event172339 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53080⟩⟩) (.sum [.result 172335 .summary, .result 170392 .summary])

def exact172340RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨16099⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨18942⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨22162⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨32182⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨51237⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact172340RawTermsValid :
    exact172340RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event172340 : Event := .resultExact (⟨.program ⟨257⟩, ⟨53080⟩⟩) exact172340RawTerms .large 172338 (.finite 160945509440761189776859800535040) (some (172339))

def event172341 : Event := .predecessor (⟨.program ⟨257⟩, ⟨56060⟩⟩) 0 ⟨53080⟩ 172340

def event172342 : Event := .predecessor (⟨.program ⟨257⟩, ⟨56060⟩⟩) 1 ⟨56059⟩ 169910

def event172343 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨56060⟩⟩) (.sum [.predecessor 0 172341 .coefficient, .predecessor 1 172342 .coefficient])

def event172344 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨56060⟩⟩) (.sum [.result 172340 .summary, .result 169910 .summary])

def exact172345RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨16099⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨18942⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨22162⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨32182⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨51237⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨54217⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact172345RawTermsValid :
    exact172345RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event172345 : Event := .resultExact (⟨.program ⟨257⟩, ⟨56060⟩⟩) exact172345RawTerms .large 172343 (.finite 193135298905473333552574874779648) (some (172344))

def event172346 : Event := .predecessor (⟨.program ⟨257⟩, ⟨59040⟩⟩) 0 ⟨56060⟩ 172345

def event172347 : Event := .predecessor (⟨.program ⟨257⟩, ⟨59040⟩⟩) 1 ⟨59039⟩ 169428

def event172348 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨59040⟩⟩) (.sum [.predecessor 0 172346 .coefficient, .predecessor 1 172347 .coefficient])

def event172349 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨59040⟩⟩) (.sum [.result 172345 .summary, .result 169428 .summary])

def exact172350RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨16099⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨18942⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨22162⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨32182⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨51237⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨54217⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨57197⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact172350RawTermsValid :
    exact172350RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event172350 : Event := .resultExact (⟨.program ⟨257⟩, ⟨59040⟩⟩) exact172350RawTerms .large 172348 (.finite 225325481271076852082771728531456) (some (172349))

def event172351 : Event := .predecessor (⟨.program ⟨257⟩, ⟨62020⟩⟩) 0 ⟨59040⟩ 172350

def event172352 : Event := .predecessor (⟨.program ⟨257⟩, ⟨62020⟩⟩) 1 ⟨62019⟩ 168946

def event172353 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨62020⟩⟩) (.sum [.predecessor 0 172351 .coefficient, .predecessor 1 172352 .coefficient])

def event172354 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨62020⟩⟩) (.sum [.result 172350 .summary, .result 168946 .summary])

def exact172355RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨16099⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨18942⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨22162⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨32182⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨51237⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨54217⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨57197⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨60177⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact172355RawTermsValid :
    exact172355RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event172355 : Event := .resultExact (⟨.program ⟨257⟩, ⟨62020⟩⟩) exact172355RawTerms .large 172353 (.finite 257515860087126057990209472036864) (some (172354))

def event172356 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65000⟩⟩) 0 ⟨62020⟩ 172355

def event172357 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65000⟩⟩) 1 ⟨64999⟩ 168464

def event172358 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨65000⟩⟩) (.sum [.predecessor 0 172356 .coefficient, .predecessor 1 172357 .coefficient])

def event172359 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨65000⟩⟩) (.sum [.result 172355 .summary, .result 168464 .summary])

def exact172360RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨16099⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨18942⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨22162⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨32182⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨51237⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨54217⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨57197⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨60177⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨63157⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact172360RawTermsValid :
    exact172360RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event172360 : Event := .resultExact (⟨.program ⟨257⟩, ⟨65000⟩⟩) exact172360RawTerms .large 172358 (.finite 289706631804066638652128995049472) (some (172359))

def event172361 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70497⟩⟩) 0 ⟨65000⟩ 172360

def event172362 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70497⟩⟩) 1 ⟨70496⟩ 167982

def event172363 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70497⟩⟩) (.sum [.predecessor 0 172361 .coefficient, .predecessor 1 172362 .coefficient])

def event172364 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70497⟩⟩) (.sum [.result 172360 .summary, .result 167982 .summary])

def exact172365RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨16099⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨18942⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨22162⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨32182⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨51237⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨54217⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨57197⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨60177⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨63157⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨66881⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact172365RawTermsValid :
    exact172365RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event172365 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70497⟩⟩) exact172365RawTerms .large 172363 (.finite 321897992872344281445771187322880) (some (172364))

def event172366 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70498⟩⟩) 0 ⟨70497⟩ 172365

def event172367 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70498⟩⟩) 1 ⟨28392⟩ 167500

def event172368 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70498⟩⟩) (.sum [.predecessor 0 172366 .coefficient, .predecessor 1 172367 .coefficient])

def event172369 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70498⟩⟩) (.sum [.result 172365 .summary, .result 167500 .summary])

def exact172370RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨16099⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨18942⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨22162⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨26671⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨32182⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨51237⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨54217⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨57197⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨60177⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨63157⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨66881⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact172370RawTermsValid :
    exact172370RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event172370 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70498⟩⟩) exact172370RawTerms .large 172368 (.finite 354089550391067611616654269349888) (some (172369))

def event172371 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70499⟩⟩) 0 ⟨70498⟩ 172370

def event172372 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70499⟩⟩) 1 ⟨31072⟩ 167018

def event172373 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70499⟩⟩) (.sum [.predecessor 0 172371 .coefficient, .predecessor 1 172372 .coefficient])

def event172374 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70499⟩⟩) (.sum [.result 172370 .summary, .result 167018 .summary])

def exact172375RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨16099⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨18942⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨22162⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨26671⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨29351⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨32182⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨51237⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨54217⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨57197⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨60177⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨63157⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨66881⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact172375RawTermsValid :
    exact172375RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event172375 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70499⟩⟩) exact172375RawTerms .large 172373 (.finite 386281697261128003919260020637696) (some (172374))

def event172376 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70500⟩⟩) 0 ⟨70499⟩ 172375

def event172377 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70500⟩⟩) 1 ⟨36732⟩ 166536

def event172378 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70500⟩⟩) (.sum [.predecessor 0 172376 .coefficient, .predecessor 1 172377 .coefficient])

def event172379 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70500⟩⟩) (.sum [.result 172375 .summary, .result 166536 .summary])

def exact172380RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨16099⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨18942⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨22162⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨26671⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨29351⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨32182⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨35015⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨51237⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨54217⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨57197⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨60177⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨63157⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨66881⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact172380RawTermsValid :
    exact172380RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event172380 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70500⟩⟩) exact172380RawTerms .large 172378 (.finite 418474237032079770976347551432704) (some (172379))

def event172381 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70501⟩⟩) 0 ⟨70500⟩ 172380

def event172382 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70501⟩⟩) 1 ⟨39412⟩ 166054

def event172383 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70501⟩⟩) (.sum [.predecessor 0 172381 .coefficient, .predecessor 1 172382 .coefficient])

def event172384 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70501⟩⟩) (.sum [.result 172380 .summary, .result 166054 .summary])

def exact172385RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7224⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨16099⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨18942⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨22162⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨26671⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨29351⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨32182⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨35015⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨37695⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨51237⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨54217⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨57197⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨60177⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨63157⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨66881⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact172385RawTermsValid :
    exact172385RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event172385 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70501⟩⟩) exact172385RawTerms .large 172383 (.finite 450666973253477225410675971981312) (some (172384))

def event172386 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70502⟩⟩) 0 ⟨70501⟩ 172385

def event172387 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70502⟩⟩) 1 ⟨42092⟩ 165572

def event172388 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70502⟩⟩) (.sum [.predecessor 0 172386 .coefficient, .predecessor 1 172387 .coefficient])

def event172389 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70502⟩⟩) (.sum [.result 172385 .summary, .result 165572 .summary])

def exact172390RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7224⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7226⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨16099⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨18942⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨22162⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨26671⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨29351⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨32182⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨35015⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨37695⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨40371⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨51237⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨54217⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨57197⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨60177⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨63157⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨66881⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact172390RawTermsValid :
    exact172390RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event172390 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70502⟩⟩) exact172390RawTerms .large 172388 (.finite 482860102375766054599486172037120) (some (172389))

def event172391 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70503⟩⟩) 0 ⟨70502⟩ 172390

def event172392 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70503⟩⟩) 1 ⟨44772⟩ 165090

def event172393 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70503⟩⟩) (.sum [.predecessor 0 172391 .coefficient, .predecessor 1 172392 .coefficient])

def event172394 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70503⟩⟩) (.sum [.result 172390 .summary, .result 165090 .summary])

def exact172395RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7224⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7226⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7228⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨16099⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨18942⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨22162⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨26671⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨29351⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨32182⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨35015⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨37695⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨40371⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨43051⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨51237⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨54217⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨57197⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨60177⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨63157⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨66881⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact172395RawTermsValid :
    exact172395RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event172395 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70503⟩⟩) exact172395RawTerms .large 172393 (.finite 515053820849391945920019041353728) (some (172394))

def event172396 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70504⟩⟩) 0 ⟨70503⟩ 172395

def event172397 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70504⟩⟩) 1 ⟨47452⟩ 164608

def event172398 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70504⟩⟩) (.sum [.predecessor 0 172396 .coefficient, .predecessor 1 172397 .coefficient])

def event172399 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70504⟩⟩) (.sum [.result 172395 .summary, .result 164608 .summary])

def exact172400RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7224⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7226⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7228⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7230⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨16099⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨18942⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨22162⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨26671⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨29351⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨32182⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨35015⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨37695⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨40371⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨43051⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨45735⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨51237⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨54217⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨57197⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨60177⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨63157⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨66881⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact172400RawTermsValid :
    exact172400RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event172400 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70504⟩⟩) exact172400RawTerms .large 172398 (.finite 547248128674354899372274579931136) (some (172399))

def event172401 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70505⟩⟩) 0 ⟨70504⟩ 172400

def event172402 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70505⟩⟩) 1 ⟨50132⟩ 164126

def event172403 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70505⟩⟩) (.sum [.predecessor 0 172401 .coefficient, .predecessor 1 172402 .coefficient])

def event172404 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70505⟩⟩) (.sum [.result 172400 .summary, .result 164126 .summary])

def exact172405RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7224⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7226⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7228⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7230⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7232⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨16099⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨18942⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨22162⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨26671⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨29351⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨32182⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨35015⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨37695⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨40371⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨43051⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨45735⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨48415⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨51237⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨54217⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨57197⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨60177⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨63157⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨66881⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact172405RawTermsValid :
    exact172405RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event172405 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70505⟩⟩) exact172405RawTerms .large 172403 (.finite 579442632949763540201771008262144) (some (172404))

def event172406 : Event := .predecessor (⟨.program ⟨257⟩, ⟨71367⟩⟩) 0 ⟨70505⟩ 172405

def event172407 : Event := .predecessor (⟨.program ⟨257⟩, ⟨71367⟩⟩) 1 ⟨71365⟩ 163628

def event172408 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨71367⟩⟩) (.product (.predecessor 0 172406 .coefficient) (.predecessor 1 172407 .coefficient) (⟨false, false, none, none, none⟩))

def event172409 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨71367⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨71365⟩⟩]⟩) [⟨.result 163628 .coefficient, false, none⟩])

def event172410 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨71367⟩⟩) (.product (.result 172405 .summary) (.transfer 172409) (⟨false, false, none, none, none⟩))

def event172411 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71367⟩⟩, .operator (⟨172405, 17⟩, ⟨163628, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7232⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩]⟩, (1)⟩)

def event172412 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71367⟩⟩, .operator (⟨172405, 29⟩, ⟨163628, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨48415⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩]⟩, (-1)⟩)

def event172413 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71367⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨48415⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71365⟩⟩) ⟨68854⟩ 163625)

def event172414 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71367⟩⟩, .relation 172413 0, ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨48415⟩⟩], [⟨.program ⟨257⟩, ⟨68854⟩⟩]⟩, (-1)⟩)

def event172415 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71367⟩⟩, .operator (⟨172405, 16⟩, ⟨163628, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩]⟩, (1)⟩)

def event172416 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71367⟩⟩, .operator (⟨172405, 28⟩, ⟨163628, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨45735⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩]⟩, (-1)⟩)

def event172417 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71367⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨45735⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71365⟩⟩) ⟨68854⟩ 163625)

def event172418 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71367⟩⟩, .relation 172417 0, ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨45735⟩⟩], [⟨.program ⟨257⟩, ⟨68854⟩⟩]⟩, (-1)⟩)

def event172419 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71367⟩⟩, .operator (⟨172405, 15⟩, ⟨163628, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩]⟩, (1)⟩)

def event172420 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71367⟩⟩, .operator (⟨172405, 27⟩, ⟨163628, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨43051⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩]⟩, (-1)⟩)

def event172421 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71367⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨43051⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71365⟩⟩) ⟨68854⟩ 163625)

def event172422 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71367⟩⟩, .relation 172421 0, ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨43051⟩⟩], [⟨.program ⟨257⟩, ⟨68854⟩⟩]⟩, (-1)⟩)

def event172423 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71367⟩⟩, .operator (⟨172405, 14⟩, ⟨163628, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩]⟩, (1)⟩)

def event172424 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71367⟩⟩, .operator (⟨172405, 26⟩, ⟨163628, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨40371⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩]⟩, (-1)⟩)

def event172425 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71367⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨40371⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71365⟩⟩) ⟨68854⟩ 163625)

def event172426 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71367⟩⟩, .relation 172425 0, ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨40371⟩⟩], [⟨.program ⟨257⟩, ⟨68854⟩⟩]⟩, (-1)⟩)

def event172427 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71367⟩⟩, .operator (⟨172405, 13⟩, ⟨163628, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩]⟩, (1)⟩)

def event172428 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71367⟩⟩, .operator (⟨172405, 25⟩, ⟨163628, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨37695⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩]⟩, (-1)⟩)

def event172429 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71367⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨37695⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71365⟩⟩) ⟨68854⟩ 163625)

def event172430 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71367⟩⟩, .relation 172429 0, ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨37695⟩⟩], [⟨.program ⟨257⟩, ⟨68854⟩⟩]⟩, (-1)⟩)

def event172431 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71367⟩⟩, .operator (⟨172405, 12⟩, ⟨163628, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7222⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩]⟩, (1)⟩)

def event172432 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71367⟩⟩, .operator (⟨172405, 24⟩, ⟨163628, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨35015⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩]⟩, (-1)⟩)

def event172433 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71367⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨35015⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71365⟩⟩) ⟨68854⟩ 163625)

def event172434 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71367⟩⟩, .relation 172433 0, ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨35015⟩⟩], [⟨.program ⟨257⟩, ⟨68854⟩⟩]⟩, (-1)⟩)

def event172435 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71367⟩⟩, .operator (⟨172405, 11⟩, ⟨163628, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩]⟩, (1)⟩)

def event172436 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71367⟩⟩, .operator (⟨172405, 22⟩, ⟨163628, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨29351⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩]⟩, (-1)⟩)

def event172437 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71367⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨29351⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71365⟩⟩) ⟨68854⟩ 163625)

def event172438 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71367⟩⟩, .relation 172437 0, ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨29351⟩⟩], [⟨.program ⟨257⟩, ⟨68854⟩⟩]⟩, (-1)⟩)

def event172439 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71367⟩⟩, .operator (⟨172405, 10⟩, ⟨163628, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩]⟩, (1)⟩)

def event172440 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71367⟩⟩, .operator (⟨172405, 21⟩, ⟨163628, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨26671⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩]⟩, (-1)⟩)

def event172441 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71367⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨26671⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71365⟩⟩) ⟨68854⟩ 163625)

def event172442 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71367⟩⟩, .relation 172441 0, ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨26671⟩⟩], [⟨.program ⟨257⟩, ⟨68854⟩⟩]⟩, (-1)⟩)

def event172443 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71367⟩⟩, .operator (⟨172405, 9⟩, ⟨163628, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩]⟩, (1)⟩)

def event172444 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71367⟩⟩, .operator (⟨172405, 35⟩, ⟨163628, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨66881⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩]⟩, (-1)⟩)

def event172445 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71367⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨66881⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71365⟩⟩) ⟨68854⟩ 163625)

def event172446 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71367⟩⟩, .relation 172445 0, ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨66881⟩⟩], [⟨.program ⟨257⟩, ⟨68854⟩⟩]⟩, (-1)⟩)

def event172447 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71367⟩⟩, .operator (⟨172405, 8⟩, ⟨163628, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩]⟩, (1)⟩)

def event172448 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71367⟩⟩, .operator (⟨172405, 34⟩, ⟨163628, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨63157⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩]⟩, (-1)⟩)

def event172449 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71367⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨63157⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71365⟩⟩) ⟨68854⟩ 163625)

def event172450 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71367⟩⟩, .relation 172449 0, ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨63157⟩⟩], [⟨.program ⟨257⟩, ⟨68854⟩⟩]⟩, (-1)⟩)

def event172451 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71367⟩⟩, .operator (⟨172405, 7⟩, ⟨163628, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩]⟩, (1)⟩)

def event172452 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71367⟩⟩, .operator (⟨172405, 33⟩, ⟨163628, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨60177⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩]⟩, (-1)⟩)

def event172453 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71367⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨60177⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71365⟩⟩) ⟨68854⟩ 163625)

def event172454 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71367⟩⟩, .relation 172453 0, ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨60177⟩⟩], [⟨.program ⟨257⟩, ⟨68854⟩⟩]⟩, (-1)⟩)

def event172455 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71367⟩⟩, .operator (⟨172405, 6⟩, ⟨163628, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩]⟩, (1)⟩)

def event172456 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71367⟩⟩, .operator (⟨172405, 32⟩, ⟨163628, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨57197⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩]⟩, (-1)⟩)

def event172457 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71367⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨57197⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71365⟩⟩) ⟨68854⟩ 163625)

def event172458 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71367⟩⟩, .relation 172457 0, ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨57197⟩⟩], [⟨.program ⟨257⟩, ⟨68854⟩⟩]⟩, (-1)⟩)

def event172459 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71367⟩⟩, .operator (⟨172405, 5⟩, ⟨163628, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩]⟩, (1)⟩)

def event172460 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71367⟩⟩, .operator (⟨172405, 31⟩, ⟨163628, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨54217⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩]⟩, (-1)⟩)

def event172461 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71367⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨54217⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71365⟩⟩) ⟨68854⟩ 163625)

def event172462 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71367⟩⟩, .relation 172461 0, ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨54217⟩⟩], [⟨.program ⟨257⟩, ⟨68854⟩⟩]⟩, (-1)⟩)

def event172463 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71367⟩⟩, .operator (⟨172405, 4⟩, ⟨163628, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩]⟩, (1)⟩)

def event172464 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71367⟩⟩, .operator (⟨172405, 30⟩, ⟨163628, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨51237⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩]⟩, (-1)⟩)

def event172465 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71367⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨51237⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71365⟩⟩) ⟨68854⟩ 163625)

def event172466 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71367⟩⟩, .relation 172465 0, ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨51237⟩⟩], [⟨.program ⟨257⟩, ⟨68854⟩⟩]⟩, (-1)⟩)

def event172467 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71367⟩⟩, .operator (⟨172405, 3⟩, ⟨163628, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩]⟩, (1)⟩)

def event172468 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71367⟩⟩, .operator (⟨172405, 23⟩, ⟨163628, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨32182⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩]⟩, (-1)⟩)

def event172469 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71367⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨32182⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71365⟩⟩) ⟨68854⟩ 163625)

def event172470 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71367⟩⟩, .relation 172469 0, ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨32182⟩⟩], [⟨.program ⟨257⟩, ⟨68854⟩⟩]⟩, (-1)⟩)

def event172471 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71367⟩⟩, .operator (⟨172405, 2⟩, ⟨163628, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩]⟩, (1)⟩)

def event172472 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71367⟩⟩, .operator (⟨172405, 20⟩, ⟨163628, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨22162⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩]⟩, (-1)⟩)

def event172473 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71367⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨22162⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71365⟩⟩) ⟨68854⟩ 163625)

def event172474 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71367⟩⟩, .relation 172473 0, ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨22162⟩⟩], [⟨.program ⟨257⟩, ⟨68854⟩⟩]⟩, (-1)⟩)

def event172475 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71367⟩⟩, .operator (⟨172405, 1⟩, ⟨163628, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩]⟩, (1)⟩)

def event172476 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71367⟩⟩, .operator (⟨172405, 19⟩, ⟨163628, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨18942⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩]⟩, (-1)⟩)

def event172477 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71367⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨18942⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71365⟩⟩) ⟨68854⟩ 163625)

def event172478 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71367⟩⟩, .relation 172477 0, ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨18942⟩⟩], [⟨.program ⟨257⟩, ⟨68854⟩⟩]⟩, (-1)⟩)

def event172479 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71367⟩⟩, .operator (⟨172405, 0⟩, ⟨163628, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩]⟩, (1)⟩)

def event172480 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71367⟩⟩, .operator (⟨172405, 18⟩, ⟨163628, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨16099⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩]⟩, (-1)⟩)

def event172481 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71367⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨16099⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71365⟩⟩) ⟨68854⟩ 163625)

def event172482 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71367⟩⟩, .relation 172481 0, ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨16099⟩⟩], [⟨.program ⟨257⟩, ⟨68854⟩⟩]⟩, (-1)⟩)

def exact172483RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7222⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7232⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨16099⟩⟩], [⟨.program ⟨257⟩, ⟨68854⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨18942⟩⟩], [⟨.program ⟨257⟩, ⟨68854⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨22162⟩⟩], [⟨.program ⟨257⟩, ⟨68854⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨26671⟩⟩], [⟨.program ⟨257⟩, ⟨68854⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨29351⟩⟩], [⟨.program ⟨257⟩, ⟨68854⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨32182⟩⟩], [⟨.program ⟨257⟩, ⟨68854⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨35015⟩⟩], [⟨.program ⟨257⟩, ⟨68854⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨37695⟩⟩], [⟨.program ⟨257⟩, ⟨68854⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨40371⟩⟩], [⟨.program ⟨257⟩, ⟨68854⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨43051⟩⟩], [⟨.program ⟨257⟩, ⟨68854⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨45735⟩⟩], [⟨.program ⟨257⟩, ⟨68854⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨48415⟩⟩], [⟨.program ⟨257⟩, ⟨68854⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨51237⟩⟩], [⟨.program ⟨257⟩, ⟨68854⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨54217⟩⟩], [⟨.program ⟨257⟩, ⟨68854⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨57197⟩⟩], [⟨.program ⟨257⟩, ⟨68854⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨60177⟩⟩], [⟨.program ⟨257⟩, ⟨68854⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨63157⟩⟩], [⟨.program ⟨257⟩, ⟨68854⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨66881⟩⟩], [⟨.program ⟨257⟩, ⟨68854⟩⟩]⟩, (-1)⟩]

theorem exact172483RawTermsValid :
    exact172483RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event172483 : Event := .resultExact (⟨.program ⟨257⟩, ⟨71367⟩⟩) exact172483RawTerms .large 172408 (.finite 6221717896068416040249469304417135687106560) (some (172410))

def event172484 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68410⟩⟩) 0 ⟨66891⟩ 8056

def event172485 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨68410⟩⟩) (.authority (.relationPreimageSource ⟨95⟩))

def exact172486RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨68410⟩⟩]⟩, (1)⟩]

theorem exact172486RawTermsValid :
    exact172486RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event172486 : Event := .resultExact (⟨.program ⟨257⟩, ⟨68410⟩⟩) exact172486RawTerms (.finite 5647228698) 172485 .exactZero (none)

def event172487 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68412⟩⟩) 0 ⟨68410⟩ 172486

def event172488 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68412⟩⟩) 1 ⟨2370⟩ 4

def event172489 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨68412⟩⟩) (.scale (.predecessor 0 172487 .coefficient) (.value (.predecessor 1 172488 .coefficient)))

def exact172490RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨68410⟩⟩]⟩, (1)⟩]

theorem exact172490RawTermsValid :
    exact172490RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event172490 : Event := .resultExact (⟨.program ⟨257⟩, ⟨68412⟩⟩) exact172490RawTerms (.finite 5647228698) 172489 .exactZero (none)

def event172491 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68413⟩⟩) 0 ⟨6466⟩ 163745

def event172492 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68413⟩⟩) 1 ⟨68412⟩ 172490

def event172493 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨68413⟩⟩) (.product (.predecessor 0 172491 .coefficient) (.predecessor 1 172492 .coefficient) (⟨false, false, none, none, none⟩))

def event172494 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨68413⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨68410⟩⟩]⟩) [⟨.result 172486 .coefficient, false, none⟩])

def event172495 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨68413⟩⟩) (.product (.result 163745 .summary) (.transfer 172494) (⟨false, false, none, none, none⟩))

def event172496 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68413⟩⟩, .operator (⟨163745, 0⟩, ⟨172490, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68410⟩⟩]⟩, (1)⟩)

def event172497 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨68411⟩⟩)

def event172498 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event172499 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event172500 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6449⟩⟩) (.authority (.operator))

def event172501 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨6449⟩⟩) (.finite 9)

def event172502 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event172503 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event172504 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event172505 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event172506 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 172505

def event172507 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 172503

def event172508 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 172506 .coefficient) (.value (.predecessor 1 172507 .coefficient)))

def event172509 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event172510 : Event := .predecessor (⟨.program ⟨257⟩, ⟨6451⟩⟩) 0 ⟨392⟩ 172509

def event172511 : Event := .predecessor (⟨.program ⟨257⟩, ⟨6451⟩⟩) 1 ⟨6449⟩ 172501

def event172512 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6451⟩⟩) (.sum [.predecessor 0 172510 .coefficient, .predecessor 1 172511 .coefficient])

def event172513 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨6451⟩⟩) (.finite 655349)

def event172514 : Event := .predecessor (⟨.program ⟨257⟩, ⟨6462⟩⟩) 0 ⟨6451⟩ 172513

def event172515 : Event := .predecessor (⟨.program ⟨257⟩, ⟨6462⟩⟩) 1 ⟨5426⟩ 172499

def event172516 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6462⟩⟩) (.identity (.predecessor 1 172515 .coefficient))

def event172517 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨6462⟩⟩) (.finite 655360)

def event172518 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47930⟩⟩) 0 ⟨6462⟩ 172517

def event172519 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47930⟩⟩) (.authority (.programFamilyFact))

def exact172520RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨47930⟩⟩], []⟩, (1)⟩]

theorem exact172520RawTermsValid :
    exact172520RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event172520 : Event := .resultExact (⟨.program ⟨257⟩, ⟨47930⟩⟩) exact172520RawTerms (.finite 60) 172519 .exactZero (none)

def event172521 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15141⟩⟩) 0 ⟨6462⟩ 172517

def event172522 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15141⟩⟩) (.authority (.programFamilyFact))

def exact172523RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15141⟩⟩], []⟩, (1)⟩]

theorem exact172523RawTermsValid :
    exact172523RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event172523 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15141⟩⟩) exact172523RawTerms (.finite 60) 172522 .exactZero (none)

def event172524 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47931⟩⟩) 0 ⟨15141⟩ 172523

def event172525 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47931⟩⟩) 1 ⟨47930⟩ 172520

def event172526 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47931⟩⟩) (.product (.predecessor 0 172524 .coefficient) (.predecessor 1 172525 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event172527 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47931⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨15141⟩⟩, ⟨.program ⟨257⟩, ⟨47930⟩⟩], []⟩) [⟨.result 172523 .coefficient, true, some 1⟩, ⟨.result 172520 .coefficient, true, some 1⟩])

def event172528 : Event := .survivorFold (1) 172527

def exact172529RawTerms : List Term := []

theorem exact172529RawTermsValid :
    exact172529RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event172529 : Event := .resultExact (⟨.program ⟨257⟩, ⟨47931⟩⟩) exact172529RawTerms (.finite 3600) 172526 (.finite 3600) (some (172527))

def event172530 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47932⟩⟩) 0 ⟨47931⟩ 172529

def event172531 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47932⟩⟩) (.identity (.predecessor 0 172530 .coefficient))

def event172532 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨47932⟩⟩) (.finite 3600)

def event172533 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48180⟩⟩) 0 ⟨47932⟩ 172532

def event172534 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48180⟩⟩) (.authority (.programFamilyFact))

def exact172535RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨48180⟩⟩], []⟩, (1)⟩]

theorem exact172535RawTermsValid :
    exact172535RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event172535 : Event := .resultExact (⟨.program ⟨257⟩, ⟨48180⟩⟩) exact172535RawTerms (.finite 60) 172534 .exactZero (none)

def event172536 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48181⟩⟩) 0 ⟨48180⟩ 172535

def event172537 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48181⟩⟩) (.identity (.predecessor 0 172536 .coefficient))

def event172538 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨48181⟩⟩) (.finite 60)

def event172539 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48415⟩⟩) 0 ⟨48181⟩ 172538

def event172540 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48415⟩⟩) (.authority (.programFamilyFact))

def exact172541RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨48415⟩⟩], []⟩, (1)⟩]

theorem exact172541RawTermsValid :
    exact172541RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event172541 : Event := .resultExact (⟨.program ⟨257⟩, ⟨48415⟩⟩) exact172541RawTerms (.finite 63) 172540 .exactZero (none)

def event172542 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45250⟩⟩) 0 ⟨6462⟩ 172517

def event172543 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45250⟩⟩) (.authority (.programFamilyFact))

def eventLeaf10768 : Array AnnotatedEvent := #[
  { event := event172288
    frameStart := 172203 },
  { event := event172289
    frameStart := 172203 },
  { event := event172290
    frameStart := 172203 },
  { event := event172291
    frameStart := 172203 },
  { event := event172292
    frameStart := 172203 },
  { event := event172293
    frameStart := 172203 },
  { event := event172294
    frameStart := 172203 },
  { event := event172295
    frameStart := 172203 },
  { event := event172296
    frameStart := 172203 },
  { event := event172297
    frameStart := 172203 },
  { event := event172298
    frameStart := 172203 },
  { event := event172299
    frameStart := 172203 },
  { event := event172300
    frameStart := 172203 },
  { event := event172301
    frameStart := 172203 },
  { event := event172302
    frameStart := 172203 },
  { event := event172303
    frameStart := 172203 }
]

def eventLeaf10769 : Array AnnotatedEvent := #[
  { event := event172304
    frameStart := 172203 },
  { event := event172305
    frameStart := 172203 },
  { event := event172306
    frameStart := 172203 },
  { event := event172307
    frameStart := 0 },
  { event := event172308
    frameStart := 0 },
  { event := event172309
    frameStart := 0 },
  { event := event172310
    frameStart := 0 },
  { event := event172311
    frameStart := 0 },
  { event := event172312
    frameStart := 0 },
  { event := event172313
    frameStart := 0 },
  { event := event172314
    frameStart := 0 },
  { event := event172315
    frameStart := 0 },
  { event := event172316
    frameStart := 0 },
  { event := event172317
    frameStart := 0 },
  { event := event172318
    frameStart := 0 },
  { event := event172319
    frameStart := 0 }
]

def eventLeaf10770 : Array AnnotatedEvent := #[
  { event := event172320
    frameStart := 0 },
  { event := event172321
    frameStart := 0 },
  { event := event172322
    frameStart := 0 },
  { event := event172323
    frameStart := 0 },
  { event := event172324
    frameStart := 0 },
  { event := event172325
    frameStart := 0 },
  { event := event172326
    frameStart := 0 },
  { event := event172327
    frameStart := 0 },
  { event := event172328
    frameStart := 0 },
  { event := event172329
    frameStart := 0 },
  { event := event172330
    frameStart := 0 },
  { event := event172331
    frameStart := 0 },
  { event := event172332
    frameStart := 0 },
  { event := event172333
    frameStart := 0 },
  { event := event172334
    frameStart := 0 },
  { event := event172335
    frameStart := 0 }
]

def eventLeaf10771 : Array AnnotatedEvent := #[
  { event := event172336
    frameStart := 0 },
  { event := event172337
    frameStart := 0 },
  { event := event172338
    frameStart := 0 },
  { event := event172339
    frameStart := 0 },
  { event := event172340
    frameStart := 0 },
  { event := event172341
    frameStart := 0 },
  { event := event172342
    frameStart := 0 },
  { event := event172343
    frameStart := 0 },
  { event := event172344
    frameStart := 0 },
  { event := event172345
    frameStart := 0 },
  { event := event172346
    frameStart := 0 },
  { event := event172347
    frameStart := 0 },
  { event := event172348
    frameStart := 0 },
  { event := event172349
    frameStart := 0 },
  { event := event172350
    frameStart := 0 },
  { event := event172351
    frameStart := 0 }
]

def eventLeaf10772 : Array AnnotatedEvent := #[
  { event := event172352
    frameStart := 0 },
  { event := event172353
    frameStart := 0 },
  { event := event172354
    frameStart := 0 },
  { event := event172355
    frameStart := 0 },
  { event := event172356
    frameStart := 0 },
  { event := event172357
    frameStart := 0 },
  { event := event172358
    frameStart := 0 },
  { event := event172359
    frameStart := 0 },
  { event := event172360
    frameStart := 0 },
  { event := event172361
    frameStart := 0 },
  { event := event172362
    frameStart := 0 },
  { event := event172363
    frameStart := 0 },
  { event := event172364
    frameStart := 0 },
  { event := event172365
    frameStart := 0 },
  { event := event172366
    frameStart := 0 },
  { event := event172367
    frameStart := 0 }
]

def eventLeaf10773 : Array AnnotatedEvent := #[
  { event := event172368
    frameStart := 0 },
  { event := event172369
    frameStart := 0 },
  { event := event172370
    frameStart := 0 },
  { event := event172371
    frameStart := 0 },
  { event := event172372
    frameStart := 0 },
  { event := event172373
    frameStart := 0 },
  { event := event172374
    frameStart := 0 },
  { event := event172375
    frameStart := 0 },
  { event := event172376
    frameStart := 0 },
  { event := event172377
    frameStart := 0 },
  { event := event172378
    frameStart := 0 },
  { event := event172379
    frameStart := 0 },
  { event := event172380
    frameStart := 0 },
  { event := event172381
    frameStart := 0 },
  { event := event172382
    frameStart := 0 },
  { event := event172383
    frameStart := 0 }
]

def eventLeaf10774 : Array AnnotatedEvent := #[
  { event := event172384
    frameStart := 0 },
  { event := event172385
    frameStart := 0 },
  { event := event172386
    frameStart := 0 },
  { event := event172387
    frameStart := 0 },
  { event := event172388
    frameStart := 0 },
  { event := event172389
    frameStart := 0 },
  { event := event172390
    frameStart := 0 },
  { event := event172391
    frameStart := 0 },
  { event := event172392
    frameStart := 0 },
  { event := event172393
    frameStart := 0 },
  { event := event172394
    frameStart := 0 },
  { event := event172395
    frameStart := 0 },
  { event := event172396
    frameStart := 0 },
  { event := event172397
    frameStart := 0 },
  { event := event172398
    frameStart := 0 },
  { event := event172399
    frameStart := 0 }
]

def eventLeaf10775 : Array AnnotatedEvent := #[
  { event := event172400
    frameStart := 0 },
  { event := event172401
    frameStart := 0 },
  { event := event172402
    frameStart := 0 },
  { event := event172403
    frameStart := 0 },
  { event := event172404
    frameStart := 0 },
  { event := event172405
    frameStart := 0 },
  { event := event172406
    frameStart := 0 },
  { event := event172407
    frameStart := 0 },
  { event := event172408
    frameStart := 0 },
  { event := event172409
    frameStart := 0 },
  { event := event172410
    frameStart := 0 },
  { event := event172411
    frameStart := 0 },
  { event := event172412
    frameStart := 0 },
  { event := event172413
    frameStart := 0 },
  { event := event172414
    frameStart := 0 },
  { event := event172415
    frameStart := 0 }
]

def eventLeaf10776 : Array AnnotatedEvent := #[
  { event := event172416
    frameStart := 0 },
  { event := event172417
    frameStart := 0 },
  { event := event172418
    frameStart := 0 },
  { event := event172419
    frameStart := 0 },
  { event := event172420
    frameStart := 0 },
  { event := event172421
    frameStart := 0 },
  { event := event172422
    frameStart := 0 },
  { event := event172423
    frameStart := 0 },
  { event := event172424
    frameStart := 0 },
  { event := event172425
    frameStart := 0 },
  { event := event172426
    frameStart := 0 },
  { event := event172427
    frameStart := 0 },
  { event := event172428
    frameStart := 0 },
  { event := event172429
    frameStart := 0 },
  { event := event172430
    frameStart := 0 },
  { event := event172431
    frameStart := 0 }
]

def eventLeaf10777 : Array AnnotatedEvent := #[
  { event := event172432
    frameStart := 0 },
  { event := event172433
    frameStart := 0 },
  { event := event172434
    frameStart := 0 },
  { event := event172435
    frameStart := 0 },
  { event := event172436
    frameStart := 0 },
  { event := event172437
    frameStart := 0 },
  { event := event172438
    frameStart := 0 },
  { event := event172439
    frameStart := 0 },
  { event := event172440
    frameStart := 0 },
  { event := event172441
    frameStart := 0 },
  { event := event172442
    frameStart := 0 },
  { event := event172443
    frameStart := 0 },
  { event := event172444
    frameStart := 0 },
  { event := event172445
    frameStart := 0 },
  { event := event172446
    frameStart := 0 },
  { event := event172447
    frameStart := 0 }
]

def eventLeaf10778 : Array AnnotatedEvent := #[
  { event := event172448
    frameStart := 0 },
  { event := event172449
    frameStart := 0 },
  { event := event172450
    frameStart := 0 },
  { event := event172451
    frameStart := 0 },
  { event := event172452
    frameStart := 0 },
  { event := event172453
    frameStart := 0 },
  { event := event172454
    frameStart := 0 },
  { event := event172455
    frameStart := 0 },
  { event := event172456
    frameStart := 0 },
  { event := event172457
    frameStart := 0 },
  { event := event172458
    frameStart := 0 },
  { event := event172459
    frameStart := 0 },
  { event := event172460
    frameStart := 0 },
  { event := event172461
    frameStart := 0 },
  { event := event172462
    frameStart := 0 },
  { event := event172463
    frameStart := 0 }
]

def eventLeaf10779 : Array AnnotatedEvent := #[
  { event := event172464
    frameStart := 0 },
  { event := event172465
    frameStart := 0 },
  { event := event172466
    frameStart := 0 },
  { event := event172467
    frameStart := 0 },
  { event := event172468
    frameStart := 0 },
  { event := event172469
    frameStart := 0 },
  { event := event172470
    frameStart := 0 },
  { event := event172471
    frameStart := 0 },
  { event := event172472
    frameStart := 0 },
  { event := event172473
    frameStart := 0 },
  { event := event172474
    frameStart := 0 },
  { event := event172475
    frameStart := 0 },
  { event := event172476
    frameStart := 0 },
  { event := event172477
    frameStart := 0 },
  { event := event172478
    frameStart := 0 },
  { event := event172479
    frameStart := 0 }
]

def eventLeaf10780 : Array AnnotatedEvent := #[
  { event := event172480
    frameStart := 0 },
  { event := event172481
    frameStart := 0 },
  { event := event172482
    frameStart := 0 },
  { event := event172483
    frameStart := 0 },
  { event := event172484
    frameStart := 0 },
  { event := event172485
    frameStart := 0 },
  { event := event172486
    frameStart := 0 },
  { event := event172487
    frameStart := 0 },
  { event := event172488
    frameStart := 0 },
  { event := event172489
    frameStart := 0 },
  { event := event172490
    frameStart := 0 },
  { event := event172491
    frameStart := 0 },
  { event := event172492
    frameStart := 0 },
  { event := event172493
    frameStart := 0 },
  { event := event172494
    frameStart := 0 },
  { event := event172495
    frameStart := 0 }
]

def eventLeaf10781 : Array AnnotatedEvent := #[
  { event := event172496
    frameStart := 0 },
  { event := event172497
    frameStart := 172497 },
  { event := event172498
    frameStart := 172497 },
  { event := event172499
    frameStart := 172497 },
  { event := event172500
    frameStart := 172497 },
  { event := event172501
    frameStart := 172497 },
  { event := event172502
    frameStart := 172497 },
  { event := event172503
    frameStart := 172497 },
  { event := event172504
    frameStart := 172497 },
  { event := event172505
    frameStart := 172497 },
  { event := event172506
    frameStart := 172497 },
  { event := event172507
    frameStart := 172497 },
  { event := event172508
    frameStart := 172497 },
  { event := event172509
    frameStart := 172497 },
  { event := event172510
    frameStart := 172497 },
  { event := event172511
    frameStart := 172497 }
]

def eventLeaf10782 : Array AnnotatedEvent := #[
  { event := event172512
    frameStart := 172497 },
  { event := event172513
    frameStart := 172497 },
  { event := event172514
    frameStart := 172497 },
  { event := event172515
    frameStart := 172497 },
  { event := event172516
    frameStart := 172497 },
  { event := event172517
    frameStart := 172497 },
  { event := event172518
    frameStart := 172497 },
  { event := event172519
    frameStart := 172497 },
  { event := event172520
    frameStart := 172497 },
  { event := event172521
    frameStart := 172497 },
  { event := event172522
    frameStart := 172497 },
  { event := event172523
    frameStart := 172497 },
  { event := event172524
    frameStart := 172497 },
  { event := event172525
    frameStart := 172497 },
  { event := event172526
    frameStart := 172497 },
  { event := event172527
    frameStart := 172497 }
]

def eventLeaf10783 : Array AnnotatedEvent := #[
  { event := event172528
    frameStart := 172497 },
  { event := event172529
    frameStart := 172497 },
  { event := event172530
    frameStart := 172497 },
  { event := event172531
    frameStart := 172497 },
  { event := event172532
    frameStart := 172497 },
  { event := event172533
    frameStart := 172497 },
  { event := event172534
    frameStart := 172497 },
  { event := event172535
    frameStart := 172497 },
  { event := event172536
    frameStart := 172497 },
  { event := event172537
    frameStart := 172497 },
  { event := event172538
    frameStart := 172497 },
  { event := event172539
    frameStart := 172497 },
  { event := event172540
    frameStart := 172497 },
  { event := event172541
    frameStart := 172497 },
  { event := event172542
    frameStart := 172497 },
  { event := event172543
    frameStart := 172497 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events673
