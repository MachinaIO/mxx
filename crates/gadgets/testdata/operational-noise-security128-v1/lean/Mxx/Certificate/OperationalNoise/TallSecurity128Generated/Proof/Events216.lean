import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events216

open Mxx.Certificate.OperationalNoise
open CertificateABI

def exact55296RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩]

theorem exact55296RawTermsValid :
    exact55296RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event55296 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7198⟩⟩) exact55296RawTerms .large 55295 .exactZero (none)

def event55297 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16165⟩⟩) 0 ⟨7198⟩ 55296

def event55298 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16165⟩⟩) 1 ⟨16164⟩ 55293

def event55299 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16165⟩⟩) (.sum [.predecessor 0 55297 .coefficient, .predecessor 1 55298 .coefficient])

def exact55300RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨16163⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact55300RawTermsValid :
    exact55300RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event55300 : Event := .resultExact (⟨.program ⟨257⟩, ⟨16165⟩⟩) exact55300RawTerms .large 55299 .exactZero (none)

def event55301 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17989⟩⟩) 0 ⟨16165⟩ 55300

def event55302 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17989⟩⟩) 1 ⟨17986⟩ 55285

def event55303 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17989⟩⟩) (.sum [.predecessor 0 55301 .coefficient, .predecessor 1 55302 .coefficient])

def exact55304RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17985⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15852⟩⟩], [⟨.program ⟨257⟩, ⟨17073⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨16163⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact55304RawTermsValid :
    exact55304RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event55304 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17989⟩⟩) exact55304RawTerms .large 55303 .exactZero (none)

def event55305 : Event := .preFoldPolynomial 55304 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17985⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15852⟩⟩], [⟨.program ⟨257⟩, ⟨17073⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨16163⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact55306RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17985⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15852⟩⟩], [⟨.program ⟨257⟩, ⟨17073⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨16163⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event55306 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨17989⟩⟩) 55305 exact55306RawTerms .large 55303 .exactZero (none)

def event55307 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨15853⟩⟩) ⟨⟨77⟩, ⟨57⟩, ⟨135⟩⟩ ⟨55149, 55307⟩

def event55308 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨16759⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16756⟩⟩]⟩) (1) 0 2 (.universal 55307 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16756⟩⟩]⟩) (none) 55306)

def event55309 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16759⟩⟩, .relation 55308 1, ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩)

def event55310 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16759⟩⟩, .relation 55308 0, ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17985⟩⟩]⟩, (-1)⟩)

def event55311 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16759⟩⟩, .relation 55308 2, ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨15852⟩⟩], [⟨.program ⟨257⟩, ⟨17073⟩⟩]⟩, (1)⟩)

def event55312 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16759⟩⟩, .relation 55308 3, ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨16163⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact55313RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17985⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨15852⟩⟩], [⟨.program ⟨257⟩, ⟨17073⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨16163⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact55313RawTermsValid :
    exact55313RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event55313 : Event := .resultExact (⟨.program ⟨257⟩, ⟨16759⟩⟩) exact55313RawTerms .large 55145 (.finite 202072841853861888) (some (55147))

def event55314 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17988⟩⟩) 0 ⟨16759⟩ 55313

def event55315 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17988⟩⟩) 1 ⟨17987⟩ 55135

def event55316 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17988⟩⟩) (.sum [.predecessor 0 55314 .coefficient, .predecessor 1 55315 .coefficient])

def event55317 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17988⟩⟩, .operator (⟨55313, 0⟩, ⟨55135, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17985⟩⟩]⟩, (1)⟩)

def event55318 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17988⟩⟩, .operator (⟨55313, 2⟩, ⟨55135, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨15852⟩⟩], [⟨.program ⟨257⟩, ⟨17073⟩⟩]⟩, (-1)⟩)

def event55319 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17988⟩⟩) (.sum [.result 55313 .summary, .result 55135 .summary])

def exact55320RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨16163⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact55320RawTermsValid :
    exact55320RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event55320 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17988⟩⟩) exact55320RawTerms .large 55316 (.finite 32188807212483706889510625476608) (some (55319))

def event55321 : Event := .predecessor (⟨.program ⟨257⟩, ⟨20904⟩⟩) 0 ⟨17988⟩ 55320

def event55322 : Event := .predecessor (⟨.program ⟨257⟩, ⟨20904⟩⟩) 1 ⟨20903⟩ 54838

def event55323 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨20904⟩⟩) (.sum [.predecessor 0 55321 .coefficient, .predecessor 1 55322 .coefficient])

def event55324 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨20904⟩⟩) (.sum [.result 55320 .summary, .result 54838 .summary])

def exact55325RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨16163⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨19018⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact55325RawTermsValid :
    exact55325RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event55325 : Event := .resultExact (⟨.program ⟨257⟩, ⟨20904⟩⟩) exact55325RawTerms .large 55323 (.finite 64377712650190257467641695830016) (some (55324))

def event55326 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24124⟩⟩) 0 ⟨20904⟩ 55325

def event55327 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24124⟩⟩) 1 ⟨24123⟩ 54356

def event55328 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨24124⟩⟩) (.sum [.predecessor 0 55326 .coefficient, .predecessor 1 55327 .coefficient])

def event55329 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨24124⟩⟩) (.sum [.result 55325 .summary, .result 54356 .summary])

def exact55330RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨16163⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨19018⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨22238⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact55330RawTermsValid :
    exact55330RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event55330 : Event := .resultExact (⟨.program ⟨257⟩, ⟨24124⟩⟩) exact55330RawTerms .large 55328 (.finite 96566716313119651734393211060224) (some (55329))

def event55331 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34144⟩⟩) 0 ⟨24124⟩ 55330

def event55332 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34144⟩⟩) 1 ⟨34143⟩ 53874

def event55333 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34144⟩⟩) (.sum [.predecessor 0 55331 .coefficient, .predecessor 1 55332 .coefficient])

def event55334 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34144⟩⟩) (.sum [.result 55330 .summary, .result 53874 .summary])

def exact55335RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨16163⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨19018⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨22238⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨32258⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact55335RawTermsValid :
    exact55335RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event55335 : Event := .resultExact (⟨.program ⟨257⟩, ⟨34144⟩⟩) exact55335RawTerms .large 55333 (.finite 128755916426494733378385616044032) (some (55334))

def event55336 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53204⟩⟩) 0 ⟨34144⟩ 55335

def event55337 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53204⟩⟩) 1 ⟨53203⟩ 53392

def event55338 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53204⟩⟩) (.sum [.predecessor 0 55336 .coefficient, .predecessor 1 55337 .coefficient])

def event55339 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53204⟩⟩) (.sum [.result 55335 .summary, .result 53392 .summary])

def exact55340RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨16163⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨19018⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨22238⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨32258⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨51313⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact55340RawTermsValid :
    exact55340RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event55340 : Event := .resultExact (⟨.program ⟨257⟩, ⟨53204⟩⟩) exact55340RawTerms .large 55338 (.finite 160945509440761189776859800535040) (some (55339))

def event55341 : Event := .predecessor (⟨.program ⟨257⟩, ⟨56184⟩⟩) 0 ⟨53204⟩ 55340

def event55342 : Event := .predecessor (⟨.program ⟨257⟩, ⟨56184⟩⟩) 1 ⟨56183⟩ 52910

def event55343 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨56184⟩⟩) (.sum [.predecessor 0 55341 .coefficient, .predecessor 1 55342 .coefficient])

def event55344 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨56184⟩⟩) (.sum [.result 55340 .summary, .result 52910 .summary])

def exact55345RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨16163⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨19018⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨22238⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨32258⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨51313⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨54293⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact55345RawTermsValid :
    exact55345RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event55345 : Event := .resultExact (⟨.program ⟨257⟩, ⟨56184⟩⟩) exact55345RawTerms .large 55343 (.finite 193135298905473333552574874779648) (some (55344))

def event55346 : Event := .predecessor (⟨.program ⟨257⟩, ⟨59164⟩⟩) 0 ⟨56184⟩ 55345

def event55347 : Event := .predecessor (⟨.program ⟨257⟩, ⟨59164⟩⟩) 1 ⟨59163⟩ 52428

def event55348 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨59164⟩⟩) (.sum [.predecessor 0 55346 .coefficient, .predecessor 1 55347 .coefficient])

def event55349 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨59164⟩⟩) (.sum [.result 55345 .summary, .result 52428 .summary])

def exact55350RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨16163⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨19018⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨22238⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨32258⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨51313⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨54293⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨57273⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact55350RawTermsValid :
    exact55350RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event55350 : Event := .resultExact (⟨.program ⟨257⟩, ⟨59164⟩⟩) exact55350RawTerms .large 55348 (.finite 225325481271076852082771728531456) (some (55349))

def event55351 : Event := .predecessor (⟨.program ⟨257⟩, ⟨62144⟩⟩) 0 ⟨59164⟩ 55350

def event55352 : Event := .predecessor (⟨.program ⟨257⟩, ⟨62144⟩⟩) 1 ⟨62143⟩ 51946

def event55353 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨62144⟩⟩) (.sum [.predecessor 0 55351 .coefficient, .predecessor 1 55352 .coefficient])

def event55354 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨62144⟩⟩) (.sum [.result 55350 .summary, .result 51946 .summary])

def exact55355RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨16163⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨19018⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨22238⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨32258⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨51313⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨54293⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨57273⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨60253⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact55355RawTermsValid :
    exact55355RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event55355 : Event := .resultExact (⟨.program ⟨257⟩, ⟨62144⟩⟩) exact55355RawTerms .large 55353 (.finite 257515860087126057990209472036864) (some (55354))

def event55356 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65124⟩⟩) 0 ⟨62144⟩ 55355

def event55357 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65124⟩⟩) 1 ⟨65123⟩ 51464

def event55358 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨65124⟩⟩) (.sum [.predecessor 0 55356 .coefficient, .predecessor 1 55357 .coefficient])

def event55359 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨65124⟩⟩) (.sum [.result 55355 .summary, .result 51464 .summary])

def exact55360RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨16163⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨19018⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨22238⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨32258⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨51313⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨54293⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨57273⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨60253⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨63233⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact55360RawTermsValid :
    exact55360RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event55360 : Event := .resultExact (⟨.program ⟨257⟩, ⟨65124⟩⟩) exact55360RawTerms .large 55358 (.finite 289706631804066638652128995049472) (some (55359))

def event55361 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70813⟩⟩) 0 ⟨65124⟩ 55360

def event55362 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70813⟩⟩) 1 ⟨70812⟩ 50982

def event55363 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70813⟩⟩) (.sum [.predecessor 0 55361 .coefficient, .predecessor 1 55362 .coefficient])

def event55364 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70813⟩⟩) (.sum [.result 55360 .summary, .result 50982 .summary])

def exact55365RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨16163⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨19018⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨22238⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨32258⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨51313⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨54293⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨57273⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨60253⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨63233⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨67161⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact55365RawTermsValid :
    exact55365RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event55365 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70813⟩⟩) exact55365RawTerms .large 55363 (.finite 321897992872344281445771187322880) (some (55364))

def event55366 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70814⟩⟩) 0 ⟨70813⟩ 55365

def event55367 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70814⟩⟩) 1 ⟨28492⟩ 50500

def event55368 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70814⟩⟩) (.sum [.predecessor 0 55366 .coefficient, .predecessor 1 55367 .coefficient])

def event55369 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70814⟩⟩) (.sum [.result 55365 .summary, .result 50500 .summary])

def exact55370RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨16163⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨19018⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨22238⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨26723⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨32258⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨51313⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨54293⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨57273⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨60253⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨63233⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨67161⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact55370RawTermsValid :
    exact55370RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event55370 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70814⟩⟩) exact55370RawTerms .large 55368 (.finite 354089550391067611616654269349888) (some (55369))

def event55371 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70815⟩⟩) 0 ⟨70814⟩ 55370

def event55372 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70815⟩⟩) 1 ⟨31172⟩ 50018

def event55373 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70815⟩⟩) (.sum [.predecessor 0 55371 .coefficient, .predecessor 1 55372 .coefficient])

def event55374 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70815⟩⟩) (.sum [.result 55370 .summary, .result 50018 .summary])

def exact55375RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨16163⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨19018⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨22238⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨26723⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨29403⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨32258⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨51313⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨54293⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨57273⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨60253⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨63233⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨67161⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact55375RawTermsValid :
    exact55375RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event55375 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70815⟩⟩) exact55375RawTerms .large 55373 (.finite 386281697261128003919260020637696) (some (55374))

def event55376 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70816⟩⟩) 0 ⟨70815⟩ 55375

def event55377 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70816⟩⟩) 1 ⟨36832⟩ 49536

def event55378 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70816⟩⟩) (.sum [.predecessor 0 55376 .coefficient, .predecessor 1 55377 .coefficient])

def event55379 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70816⟩⟩) (.sum [.result 55375 .summary, .result 49536 .summary])

def exact55380RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨16163⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨19018⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨22238⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨26723⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨29403⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨32258⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨35067⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨51313⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨54293⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨57273⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨60253⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨63233⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨67161⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact55380RawTermsValid :
    exact55380RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event55380 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70816⟩⟩) exact55380RawTerms .large 55378 (.finite 418474237032079770976347551432704) (some (55379))

def event55381 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70817⟩⟩) 0 ⟨70816⟩ 55380

def event55382 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70817⟩⟩) 1 ⟨39512⟩ 49054

def event55383 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70817⟩⟩) (.sum [.predecessor 0 55381 .coefficient, .predecessor 1 55382 .coefficient])

def event55384 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70817⟩⟩) (.sum [.result 55380 .summary, .result 49054 .summary])

def exact55385RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7224⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨16163⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨19018⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨22238⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨26723⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨29403⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨32258⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨35067⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨37747⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨51313⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨54293⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨57273⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨60253⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨63233⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨67161⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact55385RawTermsValid :
    exact55385RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event55385 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70817⟩⟩) exact55385RawTerms .large 55383 (.finite 450666973253477225410675971981312) (some (55384))

def event55386 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70818⟩⟩) 0 ⟨70817⟩ 55385

def event55387 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70818⟩⟩) 1 ⟨42192⟩ 48572

def event55388 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70818⟩⟩) (.sum [.predecessor 0 55386 .coefficient, .predecessor 1 55387 .coefficient])

def event55389 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70818⟩⟩) (.sum [.result 55385 .summary, .result 48572 .summary])

def exact55390RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7224⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7226⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨16163⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨19018⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨22238⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨26723⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨29403⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨32258⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨35067⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨37747⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨40423⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨51313⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨54293⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨57273⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨60253⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨63233⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨67161⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact55390RawTermsValid :
    exact55390RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event55390 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70818⟩⟩) exact55390RawTerms .large 55388 (.finite 482860102375766054599486172037120) (some (55389))

def event55391 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70819⟩⟩) 0 ⟨70818⟩ 55390

def event55392 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70819⟩⟩) 1 ⟨44872⟩ 48090

def event55393 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70819⟩⟩) (.sum [.predecessor 0 55391 .coefficient, .predecessor 1 55392 .coefficient])

def event55394 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70819⟩⟩) (.sum [.result 55390 .summary, .result 48090 .summary])

def exact55395RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7224⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7226⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7228⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨16163⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨19018⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨22238⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨26723⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨29403⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨32258⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨35067⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨37747⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨40423⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨43103⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨51313⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨54293⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨57273⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨60253⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨63233⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨67161⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact55395RawTermsValid :
    exact55395RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event55395 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70819⟩⟩) exact55395RawTerms .large 55393 (.finite 515053820849391945920019041353728) (some (55394))

def event55396 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70820⟩⟩) 0 ⟨70819⟩ 55395

def event55397 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70820⟩⟩) 1 ⟨47552⟩ 47608

def event55398 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70820⟩⟩) (.sum [.predecessor 0 55396 .coefficient, .predecessor 1 55397 .coefficient])

def event55399 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70820⟩⟩) (.sum [.result 55395 .summary, .result 47608 .summary])

def exact55400RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7224⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7226⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7228⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7230⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨16163⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨19018⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨22238⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨26723⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨29403⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨32258⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨35067⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨37747⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨40423⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨43103⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨45787⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨51313⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨54293⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨57273⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨60253⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨63233⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨67161⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact55400RawTermsValid :
    exact55400RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event55400 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70820⟩⟩) exact55400RawTerms .large 55398 (.finite 547248128674354899372274579931136) (some (55399))

def event55401 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70821⟩⟩) 0 ⟨70820⟩ 55400

def event55402 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70821⟩⟩) 1 ⟨50232⟩ 47126

def event55403 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70821⟩⟩) (.sum [.predecessor 0 55401 .coefficient, .predecessor 1 55402 .coefficient])

def event55404 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70821⟩⟩) (.sum [.result 55400 .summary, .result 47126 .summary])

def exact55405RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7224⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7226⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7228⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7230⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7232⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨16163⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨19018⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨22238⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨26723⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨29403⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨32258⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨35067⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨37747⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨40423⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨43103⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨45787⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨48467⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨51313⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨54293⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨57273⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨60253⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨63233⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨67161⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact55405RawTermsValid :
    exact55405RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event55405 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70821⟩⟩) exact55405RawTerms .large 55403 (.finite 579442632949763540201771008262144) (some (55404))

def event55406 : Event := .predecessor (⟨.program ⟨257⟩, ⟨71503⟩⟩) 0 ⟨70821⟩ 55405

def event55407 : Event := .predecessor (⟨.program ⟨257⟩, ⟨71503⟩⟩) 1 ⟨71501⟩ 46628

def event55408 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨71503⟩⟩) (.product (.predecessor 0 55406 .coefficient) (.predecessor 1 55407 .coefficient) (⟨false, false, none, none, none⟩))

def event55409 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨71503⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨71501⟩⟩]⟩) [⟨.result 46628 .coefficient, false, none⟩])

def event55410 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨71503⟩⟩) (.product (.result 55405 .summary) (.transfer 55409) (⟨false, false, none, none, none⟩))

def event55411 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71503⟩⟩, .operator (⟨55405, 17⟩, ⟨46628, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7232⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩]⟩, (1)⟩)

def event55412 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71503⟩⟩, .operator (⟨55405, 29⟩, ⟨46628, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨48467⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩]⟩, (-1)⟩)

def event55413 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71503⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨48467⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71501⟩⟩) ⟨68878⟩ 46625)

def event55414 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71503⟩⟩, .relation 55413 0, ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨48467⟩⟩], [⟨.program ⟨257⟩, ⟨68878⟩⟩]⟩, (-1)⟩)

def event55415 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71503⟩⟩, .operator (⟨55405, 16⟩, ⟨46628, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩]⟩, (1)⟩)

def event55416 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71503⟩⟩, .operator (⟨55405, 28⟩, ⟨46628, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨45787⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩]⟩, (-1)⟩)

def event55417 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71503⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨45787⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71501⟩⟩) ⟨68878⟩ 46625)

def event55418 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71503⟩⟩, .relation 55417 0, ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨45787⟩⟩], [⟨.program ⟨257⟩, ⟨68878⟩⟩]⟩, (-1)⟩)

def event55419 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71503⟩⟩, .operator (⟨55405, 15⟩, ⟨46628, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩]⟩, (1)⟩)

def event55420 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71503⟩⟩, .operator (⟨55405, 27⟩, ⟨46628, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨43103⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩]⟩, (-1)⟩)

def event55421 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71503⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨43103⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71501⟩⟩) ⟨68878⟩ 46625)

def event55422 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71503⟩⟩, .relation 55421 0, ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨43103⟩⟩], [⟨.program ⟨257⟩, ⟨68878⟩⟩]⟩, (-1)⟩)

def event55423 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71503⟩⟩, .operator (⟨55405, 14⟩, ⟨46628, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩]⟩, (1)⟩)

def event55424 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71503⟩⟩, .operator (⟨55405, 26⟩, ⟨46628, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨40423⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩]⟩, (-1)⟩)

def event55425 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71503⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨40423⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71501⟩⟩) ⟨68878⟩ 46625)

def event55426 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71503⟩⟩, .relation 55425 0, ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨40423⟩⟩], [⟨.program ⟨257⟩, ⟨68878⟩⟩]⟩, (-1)⟩)

def event55427 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71503⟩⟩, .operator (⟨55405, 13⟩, ⟨46628, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩]⟩, (1)⟩)

def event55428 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71503⟩⟩, .operator (⟨55405, 25⟩, ⟨46628, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨37747⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩]⟩, (-1)⟩)

def event55429 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71503⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨37747⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71501⟩⟩) ⟨68878⟩ 46625)

def event55430 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71503⟩⟩, .relation 55429 0, ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨37747⟩⟩], [⟨.program ⟨257⟩, ⟨68878⟩⟩]⟩, (-1)⟩)

def event55431 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71503⟩⟩, .operator (⟨55405, 12⟩, ⟨46628, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7222⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩]⟩, (1)⟩)

def event55432 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71503⟩⟩, .operator (⟨55405, 24⟩, ⟨46628, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨35067⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩]⟩, (-1)⟩)

def event55433 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71503⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨35067⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71501⟩⟩) ⟨68878⟩ 46625)

def event55434 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71503⟩⟩, .relation 55433 0, ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨35067⟩⟩], [⟨.program ⟨257⟩, ⟨68878⟩⟩]⟩, (-1)⟩)

def event55435 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71503⟩⟩, .operator (⟨55405, 11⟩, ⟨46628, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩]⟩, (1)⟩)

def event55436 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71503⟩⟩, .operator (⟨55405, 22⟩, ⟨46628, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨29403⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩]⟩, (-1)⟩)

def event55437 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71503⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨29403⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71501⟩⟩) ⟨68878⟩ 46625)

def event55438 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71503⟩⟩, .relation 55437 0, ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨29403⟩⟩], [⟨.program ⟨257⟩, ⟨68878⟩⟩]⟩, (-1)⟩)

def event55439 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71503⟩⟩, .operator (⟨55405, 10⟩, ⟨46628, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩]⟩, (1)⟩)

def event55440 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71503⟩⟩, .operator (⟨55405, 21⟩, ⟨46628, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨26723⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩]⟩, (-1)⟩)

def event55441 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71503⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨26723⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71501⟩⟩) ⟨68878⟩ 46625)

def event55442 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71503⟩⟩, .relation 55441 0, ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨26723⟩⟩], [⟨.program ⟨257⟩, ⟨68878⟩⟩]⟩, (-1)⟩)

def event55443 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71503⟩⟩, .operator (⟨55405, 9⟩, ⟨46628, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩]⟩, (1)⟩)

def event55444 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71503⟩⟩, .operator (⟨55405, 35⟩, ⟨46628, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨67161⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩]⟩, (-1)⟩)

def event55445 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71503⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨67161⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71501⟩⟩) ⟨68878⟩ 46625)

def event55446 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71503⟩⟩, .relation 55445 0, ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨67161⟩⟩], [⟨.program ⟨257⟩, ⟨68878⟩⟩]⟩, (-1)⟩)

def event55447 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71503⟩⟩, .operator (⟨55405, 8⟩, ⟨46628, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩]⟩, (1)⟩)

def event55448 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71503⟩⟩, .operator (⟨55405, 34⟩, ⟨46628, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨63233⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩]⟩, (-1)⟩)

def event55449 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71503⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨63233⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71501⟩⟩) ⟨68878⟩ 46625)

def event55450 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71503⟩⟩, .relation 55449 0, ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨63233⟩⟩], [⟨.program ⟨257⟩, ⟨68878⟩⟩]⟩, (-1)⟩)

def event55451 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71503⟩⟩, .operator (⟨55405, 7⟩, ⟨46628, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩]⟩, (1)⟩)

def event55452 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71503⟩⟩, .operator (⟨55405, 33⟩, ⟨46628, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨60253⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩]⟩, (-1)⟩)

def event55453 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71503⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨60253⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71501⟩⟩) ⟨68878⟩ 46625)

def event55454 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71503⟩⟩, .relation 55453 0, ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨60253⟩⟩], [⟨.program ⟨257⟩, ⟨68878⟩⟩]⟩, (-1)⟩)

def event55455 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71503⟩⟩, .operator (⟨55405, 6⟩, ⟨46628, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩]⟩, (1)⟩)

def event55456 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71503⟩⟩, .operator (⟨55405, 32⟩, ⟨46628, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨57273⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩]⟩, (-1)⟩)

def event55457 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71503⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨57273⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71501⟩⟩) ⟨68878⟩ 46625)

def event55458 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71503⟩⟩, .relation 55457 0, ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨57273⟩⟩], [⟨.program ⟨257⟩, ⟨68878⟩⟩]⟩, (-1)⟩)

def event55459 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71503⟩⟩, .operator (⟨55405, 5⟩, ⟨46628, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩]⟩, (1)⟩)

def event55460 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71503⟩⟩, .operator (⟨55405, 31⟩, ⟨46628, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨54293⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩]⟩, (-1)⟩)

def event55461 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71503⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨54293⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71501⟩⟩) ⟨68878⟩ 46625)

def event55462 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71503⟩⟩, .relation 55461 0, ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨54293⟩⟩], [⟨.program ⟨257⟩, ⟨68878⟩⟩]⟩, (-1)⟩)

def event55463 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71503⟩⟩, .operator (⟨55405, 4⟩, ⟨46628, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩]⟩, (1)⟩)

def event55464 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71503⟩⟩, .operator (⟨55405, 30⟩, ⟨46628, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨51313⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩]⟩, (-1)⟩)

def event55465 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71503⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨51313⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71501⟩⟩) ⟨68878⟩ 46625)

def event55466 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71503⟩⟩, .relation 55465 0, ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨51313⟩⟩], [⟨.program ⟨257⟩, ⟨68878⟩⟩]⟩, (-1)⟩)

def event55467 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71503⟩⟩, .operator (⟨55405, 3⟩, ⟨46628, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩]⟩, (1)⟩)

def event55468 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71503⟩⟩, .operator (⟨55405, 23⟩, ⟨46628, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨32258⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩]⟩, (-1)⟩)

def event55469 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71503⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨32258⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71501⟩⟩) ⟨68878⟩ 46625)

def event55470 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71503⟩⟩, .relation 55469 0, ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨32258⟩⟩], [⟨.program ⟨257⟩, ⟨68878⟩⟩]⟩, (-1)⟩)

def event55471 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71503⟩⟩, .operator (⟨55405, 2⟩, ⟨46628, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩]⟩, (1)⟩)

def event55472 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71503⟩⟩, .operator (⟨55405, 20⟩, ⟨46628, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨22238⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩]⟩, (-1)⟩)

def event55473 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71503⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨22238⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71501⟩⟩) ⟨68878⟩ 46625)

def event55474 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71503⟩⟩, .relation 55473 0, ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨22238⟩⟩], [⟨.program ⟨257⟩, ⟨68878⟩⟩]⟩, (-1)⟩)

def event55475 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71503⟩⟩, .operator (⟨55405, 1⟩, ⟨46628, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩]⟩, (1)⟩)

def event55476 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71503⟩⟩, .operator (⟨55405, 19⟩, ⟨46628, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨19018⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩]⟩, (-1)⟩)

def event55477 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71503⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨19018⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71501⟩⟩) ⟨68878⟩ 46625)

def event55478 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71503⟩⟩, .relation 55477 0, ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨19018⟩⟩], [⟨.program ⟨257⟩, ⟨68878⟩⟩]⟩, (-1)⟩)

def event55479 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71503⟩⟩, .operator (⟨55405, 0⟩, ⟨46628, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩]⟩, (1)⟩)

def event55480 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71503⟩⟩, .operator (⟨55405, 18⟩, ⟨46628, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨16163⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩]⟩, (-1)⟩)

def event55481 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71503⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨16163⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71501⟩⟩) ⟨68878⟩ 46625)

def event55482 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71503⟩⟩, .relation 55481 0, ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨16163⟩⟩], [⟨.program ⟨257⟩, ⟨68878⟩⟩]⟩, (-1)⟩)

def exact55483RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7222⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7232⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨16163⟩⟩], [⟨.program ⟨257⟩, ⟨68878⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨19018⟩⟩], [⟨.program ⟨257⟩, ⟨68878⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨22238⟩⟩], [⟨.program ⟨257⟩, ⟨68878⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨26723⟩⟩], [⟨.program ⟨257⟩, ⟨68878⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨29403⟩⟩], [⟨.program ⟨257⟩, ⟨68878⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨32258⟩⟩], [⟨.program ⟨257⟩, ⟨68878⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨35067⟩⟩], [⟨.program ⟨257⟩, ⟨68878⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨37747⟩⟩], [⟨.program ⟨257⟩, ⟨68878⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨40423⟩⟩], [⟨.program ⟨257⟩, ⟨68878⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨43103⟩⟩], [⟨.program ⟨257⟩, ⟨68878⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨45787⟩⟩], [⟨.program ⟨257⟩, ⟨68878⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨48467⟩⟩], [⟨.program ⟨257⟩, ⟨68878⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨51313⟩⟩], [⟨.program ⟨257⟩, ⟨68878⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨54293⟩⟩], [⟨.program ⟨257⟩, ⟨68878⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨57273⟩⟩], [⟨.program ⟨257⟩, ⟨68878⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨60253⟩⟩], [⟨.program ⟨257⟩, ⟨68878⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨63233⟩⟩], [⟨.program ⟨257⟩, ⟨68878⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨67161⟩⟩], [⟨.program ⟨257⟩, ⟨68878⟩⟩]⟩, (-1)⟩]

theorem exact55483RawTermsValid :
    exact55483RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event55483 : Event := .resultExact (⟨.program ⟨257⟩, ⟨71503⟩⟩) exact55483RawTerms .large 55408 (.finite 6221717896068416040249469304417135687106560) (some (55410))

def event55484 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68450⟩⟩) 0 ⟨67171⟩ 2072

def event55485 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨68450⟩⟩) (.authority (.relationPreimageSource ⟨95⟩))

def exact55486RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨68450⟩⟩]⟩, (1)⟩]

theorem exact55486RawTermsValid :
    exact55486RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event55486 : Event := .resultExact (⟨.program ⟨257⟩, ⟨68450⟩⟩) exact55486RawTerms (.finite 5647228698) 55485 .exactZero (none)

def event55487 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68452⟩⟩) 0 ⟨68450⟩ 55486

def event55488 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68452⟩⟩) 1 ⟨2370⟩ 4

def event55489 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨68452⟩⟩) (.scale (.predecessor 0 55487 .coefficient) (.value (.predecessor 1 55488 .coefficient)))

def exact55490RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨68450⟩⟩]⟩, (1)⟩]

theorem exact55490RawTermsValid :
    exact55490RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event55490 : Event := .resultExact (⟨.program ⟨257⟩, ⟨68452⟩⟩) exact55490RawTerms (.finite 5647228698) 55489 .exactZero (none)

def event55491 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68453⟩⟩) 0 ⟨11216⟩ 46745

def event55492 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68453⟩⟩) 1 ⟨68452⟩ 55490

def event55493 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨68453⟩⟩) (.product (.predecessor 0 55491 .coefficient) (.predecessor 1 55492 .coefficient) (⟨false, false, none, none, none⟩))

def event55494 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨68453⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨68450⟩⟩]⟩) [⟨.result 55486 .coefficient, false, none⟩])

def event55495 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨68453⟩⟩) (.product (.result 46745 .summary) (.transfer 55494) (⟨false, false, none, none, none⟩))

def event55496 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68453⟩⟩, .operator (⟨46745, 0⟩, ⟨55490, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68450⟩⟩]⟩, (1)⟩)

def event55497 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨68451⟩⟩)

def event55498 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event55499 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event55500 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨11115⟩⟩) (.authority (.operator))

def event55501 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨11115⟩⟩) (.finite 17)

def event55502 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event55503 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event55504 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event55505 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event55506 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 55505

def event55507 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 55503

def event55508 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 55506 .coefficient) (.value (.predecessor 1 55507 .coefficient)))

def event55509 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event55510 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11117⟩⟩) 0 ⟨392⟩ 55509

def event55511 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11117⟩⟩) 1 ⟨11115⟩ 55501

def event55512 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨11117⟩⟩) (.sum [.predecessor 0 55510 .coefficient, .predecessor 1 55511 .coefficient])

def event55513 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨11117⟩⟩) (.finite 655357)

def event55514 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11173⟩⟩) 0 ⟨11117⟩ 55513

def event55515 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11173⟩⟩) 1 ⟨5426⟩ 55499

def event55516 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨11173⟩⟩) (.identity (.predecessor 1 55515 .coefficient))

def event55517 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨11173⟩⟩) (.finite 655360)

def event55518 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48026⟩⟩) 0 ⟨11173⟩ 55517

def event55519 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48026⟩⟩) (.authority (.programFamilyFact))

def exact55520RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨48026⟩⟩], []⟩, (1)⟩]

theorem exact55520RawTermsValid :
    exact55520RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event55520 : Event := .resultExact (⟨.program ⟨257⟩, ⟨48026⟩⟩) exact55520RawTerms (.finite 60) 55519 .exactZero (none)

def event55521 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15201⟩⟩) 0 ⟨11173⟩ 55517

def event55522 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15201⟩⟩) (.authority (.programFamilyFact))

def exact55523RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15201⟩⟩], []⟩, (1)⟩]

theorem exact55523RawTermsValid :
    exact55523RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event55523 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15201⟩⟩) exact55523RawTerms (.finite 60) 55522 .exactZero (none)

def event55524 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48027⟩⟩) 0 ⟨15201⟩ 55523

def event55525 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48027⟩⟩) 1 ⟨48026⟩ 55520

def event55526 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48027⟩⟩) (.product (.predecessor 0 55524 .coefficient) (.predecessor 1 55525 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event55527 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48027⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨15201⟩⟩, ⟨.program ⟨257⟩, ⟨48026⟩⟩], []⟩) [⟨.result 55523 .coefficient, true, some 1⟩, ⟨.result 55520 .coefficient, true, some 1⟩])

def event55528 : Event := .survivorFold (1) 55527

def exact55529RawTerms : List Term := []

theorem exact55529RawTermsValid :
    exact55529RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event55529 : Event := .resultExact (⟨.program ⟨257⟩, ⟨48027⟩⟩) exact55529RawTerms (.finite 3600) 55526 (.finite 3600) (some (55527))

def event55530 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48028⟩⟩) 0 ⟨48027⟩ 55529

def event55531 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48028⟩⟩) (.identity (.predecessor 0 55530 .coefficient))

def event55532 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨48028⟩⟩) (.finite 3600)

def event55533 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48212⟩⟩) 0 ⟨48028⟩ 55532

def event55534 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48212⟩⟩) (.authority (.programFamilyFact))

def exact55535RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨48212⟩⟩], []⟩, (1)⟩]

theorem exact55535RawTermsValid :
    exact55535RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event55535 : Event := .resultExact (⟨.program ⟨257⟩, ⟨48212⟩⟩) exact55535RawTerms (.finite 60) 55534 .exactZero (none)

def event55536 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48213⟩⟩) 0 ⟨48212⟩ 55535

def event55537 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48213⟩⟩) (.identity (.predecessor 0 55536 .coefficient))

def event55538 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨48213⟩⟩) (.finite 60)

def event55539 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48467⟩⟩) 0 ⟨48213⟩ 55538

def event55540 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48467⟩⟩) (.authority (.programFamilyFact))

def exact55541RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨48467⟩⟩], []⟩, (1)⟩]

theorem exact55541RawTermsValid :
    exact55541RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event55541 : Event := .resultExact (⟨.program ⟨257⟩, ⟨48467⟩⟩) exact55541RawTerms (.finite 63) 55540 .exactZero (none)

def event55542 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45346⟩⟩) 0 ⟨11173⟩ 55517

def event55543 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45346⟩⟩) (.authority (.programFamilyFact))

def exact55544RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨45346⟩⟩], []⟩, (1)⟩]

theorem exact55544RawTermsValid :
    exact55544RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event55544 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45346⟩⟩) exact55544RawTerms (.finite 58) 55543 .exactZero (none)

def event55545 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14901⟩⟩) 0 ⟨11173⟩ 55517

def event55546 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14901⟩⟩) (.authority (.programFamilyFact))

def exact55547RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14901⟩⟩], []⟩, (1)⟩]

theorem exact55547RawTermsValid :
    exact55547RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event55547 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14901⟩⟩) exact55547RawTerms (.finite 58) 55546 .exactZero (none)

def event55548 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45347⟩⟩) 0 ⟨14901⟩ 55547

def event55549 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45347⟩⟩) 1 ⟨45346⟩ 55544

def event55550 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45347⟩⟩) (.product (.predecessor 0 55548 .coefficient) (.predecessor 1 55549 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event55551 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45347⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨14901⟩⟩, ⟨.program ⟨257⟩, ⟨45346⟩⟩], []⟩) [⟨.result 55547 .coefficient, true, some 1⟩, ⟨.result 55544 .coefficient, true, some 1⟩])

def eventLeaf3456 : Array AnnotatedEvent := #[
  { event := event55296
    frameStart := 55203 },
  { event := event55297
    frameStart := 55203 },
  { event := event55298
    frameStart := 55203 },
  { event := event55299
    frameStart := 55203 },
  { event := event55300
    frameStart := 55203 },
  { event := event55301
    frameStart := 55203 },
  { event := event55302
    frameStart := 55203 },
  { event := event55303
    frameStart := 55203 },
  { event := event55304
    frameStart := 55203 },
  { event := event55305
    frameStart := 55203 },
  { event := event55306
    frameStart := 55203 },
  { event := event55307
    frameStart := 0 },
  { event := event55308
    frameStart := 0 },
  { event := event55309
    frameStart := 0 },
  { event := event55310
    frameStart := 0 },
  { event := event55311
    frameStart := 0 }
]

def eventLeaf3457 : Array AnnotatedEvent := #[
  { event := event55312
    frameStart := 0 },
  { event := event55313
    frameStart := 0 },
  { event := event55314
    frameStart := 0 },
  { event := event55315
    frameStart := 0 },
  { event := event55316
    frameStart := 0 },
  { event := event55317
    frameStart := 0 },
  { event := event55318
    frameStart := 0 },
  { event := event55319
    frameStart := 0 },
  { event := event55320
    frameStart := 0 },
  { event := event55321
    frameStart := 0 },
  { event := event55322
    frameStart := 0 },
  { event := event55323
    frameStart := 0 },
  { event := event55324
    frameStart := 0 },
  { event := event55325
    frameStart := 0 },
  { event := event55326
    frameStart := 0 },
  { event := event55327
    frameStart := 0 }
]

def eventLeaf3458 : Array AnnotatedEvent := #[
  { event := event55328
    frameStart := 0 },
  { event := event55329
    frameStart := 0 },
  { event := event55330
    frameStart := 0 },
  { event := event55331
    frameStart := 0 },
  { event := event55332
    frameStart := 0 },
  { event := event55333
    frameStart := 0 },
  { event := event55334
    frameStart := 0 },
  { event := event55335
    frameStart := 0 },
  { event := event55336
    frameStart := 0 },
  { event := event55337
    frameStart := 0 },
  { event := event55338
    frameStart := 0 },
  { event := event55339
    frameStart := 0 },
  { event := event55340
    frameStart := 0 },
  { event := event55341
    frameStart := 0 },
  { event := event55342
    frameStart := 0 },
  { event := event55343
    frameStart := 0 }
]

def eventLeaf3459 : Array AnnotatedEvent := #[
  { event := event55344
    frameStart := 0 },
  { event := event55345
    frameStart := 0 },
  { event := event55346
    frameStart := 0 },
  { event := event55347
    frameStart := 0 },
  { event := event55348
    frameStart := 0 },
  { event := event55349
    frameStart := 0 },
  { event := event55350
    frameStart := 0 },
  { event := event55351
    frameStart := 0 },
  { event := event55352
    frameStart := 0 },
  { event := event55353
    frameStart := 0 },
  { event := event55354
    frameStart := 0 },
  { event := event55355
    frameStart := 0 },
  { event := event55356
    frameStart := 0 },
  { event := event55357
    frameStart := 0 },
  { event := event55358
    frameStart := 0 },
  { event := event55359
    frameStart := 0 }
]

def eventLeaf3460 : Array AnnotatedEvent := #[
  { event := event55360
    frameStart := 0 },
  { event := event55361
    frameStart := 0 },
  { event := event55362
    frameStart := 0 },
  { event := event55363
    frameStart := 0 },
  { event := event55364
    frameStart := 0 },
  { event := event55365
    frameStart := 0 },
  { event := event55366
    frameStart := 0 },
  { event := event55367
    frameStart := 0 },
  { event := event55368
    frameStart := 0 },
  { event := event55369
    frameStart := 0 },
  { event := event55370
    frameStart := 0 },
  { event := event55371
    frameStart := 0 },
  { event := event55372
    frameStart := 0 },
  { event := event55373
    frameStart := 0 },
  { event := event55374
    frameStart := 0 },
  { event := event55375
    frameStart := 0 }
]

def eventLeaf3461 : Array AnnotatedEvent := #[
  { event := event55376
    frameStart := 0 },
  { event := event55377
    frameStart := 0 },
  { event := event55378
    frameStart := 0 },
  { event := event55379
    frameStart := 0 },
  { event := event55380
    frameStart := 0 },
  { event := event55381
    frameStart := 0 },
  { event := event55382
    frameStart := 0 },
  { event := event55383
    frameStart := 0 },
  { event := event55384
    frameStart := 0 },
  { event := event55385
    frameStart := 0 },
  { event := event55386
    frameStart := 0 },
  { event := event55387
    frameStart := 0 },
  { event := event55388
    frameStart := 0 },
  { event := event55389
    frameStart := 0 },
  { event := event55390
    frameStart := 0 },
  { event := event55391
    frameStart := 0 }
]

def eventLeaf3462 : Array AnnotatedEvent := #[
  { event := event55392
    frameStart := 0 },
  { event := event55393
    frameStart := 0 },
  { event := event55394
    frameStart := 0 },
  { event := event55395
    frameStart := 0 },
  { event := event55396
    frameStart := 0 },
  { event := event55397
    frameStart := 0 },
  { event := event55398
    frameStart := 0 },
  { event := event55399
    frameStart := 0 },
  { event := event55400
    frameStart := 0 },
  { event := event55401
    frameStart := 0 },
  { event := event55402
    frameStart := 0 },
  { event := event55403
    frameStart := 0 },
  { event := event55404
    frameStart := 0 },
  { event := event55405
    frameStart := 0 },
  { event := event55406
    frameStart := 0 },
  { event := event55407
    frameStart := 0 }
]

def eventLeaf3463 : Array AnnotatedEvent := #[
  { event := event55408
    frameStart := 0 },
  { event := event55409
    frameStart := 0 },
  { event := event55410
    frameStart := 0 },
  { event := event55411
    frameStart := 0 },
  { event := event55412
    frameStart := 0 },
  { event := event55413
    frameStart := 0 },
  { event := event55414
    frameStart := 0 },
  { event := event55415
    frameStart := 0 },
  { event := event55416
    frameStart := 0 },
  { event := event55417
    frameStart := 0 },
  { event := event55418
    frameStart := 0 },
  { event := event55419
    frameStart := 0 },
  { event := event55420
    frameStart := 0 },
  { event := event55421
    frameStart := 0 },
  { event := event55422
    frameStart := 0 },
  { event := event55423
    frameStart := 0 }
]

def eventLeaf3464 : Array AnnotatedEvent := #[
  { event := event55424
    frameStart := 0 },
  { event := event55425
    frameStart := 0 },
  { event := event55426
    frameStart := 0 },
  { event := event55427
    frameStart := 0 },
  { event := event55428
    frameStart := 0 },
  { event := event55429
    frameStart := 0 },
  { event := event55430
    frameStart := 0 },
  { event := event55431
    frameStart := 0 },
  { event := event55432
    frameStart := 0 },
  { event := event55433
    frameStart := 0 },
  { event := event55434
    frameStart := 0 },
  { event := event55435
    frameStart := 0 },
  { event := event55436
    frameStart := 0 },
  { event := event55437
    frameStart := 0 },
  { event := event55438
    frameStart := 0 },
  { event := event55439
    frameStart := 0 }
]

def eventLeaf3465 : Array AnnotatedEvent := #[
  { event := event55440
    frameStart := 0 },
  { event := event55441
    frameStart := 0 },
  { event := event55442
    frameStart := 0 },
  { event := event55443
    frameStart := 0 },
  { event := event55444
    frameStart := 0 },
  { event := event55445
    frameStart := 0 },
  { event := event55446
    frameStart := 0 },
  { event := event55447
    frameStart := 0 },
  { event := event55448
    frameStart := 0 },
  { event := event55449
    frameStart := 0 },
  { event := event55450
    frameStart := 0 },
  { event := event55451
    frameStart := 0 },
  { event := event55452
    frameStart := 0 },
  { event := event55453
    frameStart := 0 },
  { event := event55454
    frameStart := 0 },
  { event := event55455
    frameStart := 0 }
]

def eventLeaf3466 : Array AnnotatedEvent := #[
  { event := event55456
    frameStart := 0 },
  { event := event55457
    frameStart := 0 },
  { event := event55458
    frameStart := 0 },
  { event := event55459
    frameStart := 0 },
  { event := event55460
    frameStart := 0 },
  { event := event55461
    frameStart := 0 },
  { event := event55462
    frameStart := 0 },
  { event := event55463
    frameStart := 0 },
  { event := event55464
    frameStart := 0 },
  { event := event55465
    frameStart := 0 },
  { event := event55466
    frameStart := 0 },
  { event := event55467
    frameStart := 0 },
  { event := event55468
    frameStart := 0 },
  { event := event55469
    frameStart := 0 },
  { event := event55470
    frameStart := 0 },
  { event := event55471
    frameStart := 0 }
]

def eventLeaf3467 : Array AnnotatedEvent := #[
  { event := event55472
    frameStart := 0 },
  { event := event55473
    frameStart := 0 },
  { event := event55474
    frameStart := 0 },
  { event := event55475
    frameStart := 0 },
  { event := event55476
    frameStart := 0 },
  { event := event55477
    frameStart := 0 },
  { event := event55478
    frameStart := 0 },
  { event := event55479
    frameStart := 0 },
  { event := event55480
    frameStart := 0 },
  { event := event55481
    frameStart := 0 },
  { event := event55482
    frameStart := 0 },
  { event := event55483
    frameStart := 0 },
  { event := event55484
    frameStart := 0 },
  { event := event55485
    frameStart := 0 },
  { event := event55486
    frameStart := 0 },
  { event := event55487
    frameStart := 0 }
]

def eventLeaf3468 : Array AnnotatedEvent := #[
  { event := event55488
    frameStart := 0 },
  { event := event55489
    frameStart := 0 },
  { event := event55490
    frameStart := 0 },
  { event := event55491
    frameStart := 0 },
  { event := event55492
    frameStart := 0 },
  { event := event55493
    frameStart := 0 },
  { event := event55494
    frameStart := 0 },
  { event := event55495
    frameStart := 0 },
  { event := event55496
    frameStart := 0 },
  { event := event55497
    frameStart := 55497 },
  { event := event55498
    frameStart := 55497 },
  { event := event55499
    frameStart := 55497 },
  { event := event55500
    frameStart := 55497 },
  { event := event55501
    frameStart := 55497 },
  { event := event55502
    frameStart := 55497 },
  { event := event55503
    frameStart := 55497 }
]

def eventLeaf3469 : Array AnnotatedEvent := #[
  { event := event55504
    frameStart := 55497 },
  { event := event55505
    frameStart := 55497 },
  { event := event55506
    frameStart := 55497 },
  { event := event55507
    frameStart := 55497 },
  { event := event55508
    frameStart := 55497 },
  { event := event55509
    frameStart := 55497 },
  { event := event55510
    frameStart := 55497 },
  { event := event55511
    frameStart := 55497 },
  { event := event55512
    frameStart := 55497 },
  { event := event55513
    frameStart := 55497 },
  { event := event55514
    frameStart := 55497 },
  { event := event55515
    frameStart := 55497 },
  { event := event55516
    frameStart := 55497 },
  { event := event55517
    frameStart := 55497 },
  { event := event55518
    frameStart := 55497 },
  { event := event55519
    frameStart := 55497 }
]

def eventLeaf3470 : Array AnnotatedEvent := #[
  { event := event55520
    frameStart := 55497 },
  { event := event55521
    frameStart := 55497 },
  { event := event55522
    frameStart := 55497 },
  { event := event55523
    frameStart := 55497 },
  { event := event55524
    frameStart := 55497 },
  { event := event55525
    frameStart := 55497 },
  { event := event55526
    frameStart := 55497 },
  { event := event55527
    frameStart := 55497 },
  { event := event55528
    frameStart := 55497 },
  { event := event55529
    frameStart := 55497 },
  { event := event55530
    frameStart := 55497 },
  { event := event55531
    frameStart := 55497 },
  { event := event55532
    frameStart := 55497 },
  { event := event55533
    frameStart := 55497 },
  { event := event55534
    frameStart := 55497 },
  { event := event55535
    frameStart := 55497 }
]

def eventLeaf3471 : Array AnnotatedEvent := #[
  { event := event55536
    frameStart := 55497 },
  { event := event55537
    frameStart := 55497 },
  { event := event55538
    frameStart := 55497 },
  { event := event55539
    frameStart := 55497 },
  { event := event55540
    frameStart := 55497 },
  { event := event55541
    frameStart := 55497 },
  { event := event55542
    frameStart := 55497 },
  { event := event55543
    frameStart := 55497 },
  { event := event55544
    frameStart := 55497 },
  { event := event55545
    frameStart := 55497 },
  { event := event55546
    frameStart := 55497 },
  { event := event55547
    frameStart := 55497 },
  { event := event55548
    frameStart := 55497 },
  { event := event55549
    frameStart := 55497 },
  { event := event55550
    frameStart := 55497 },
  { event := event55551
    frameStart := 55497 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events216
