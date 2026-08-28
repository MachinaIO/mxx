import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events287

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event73472 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20535⟩⟩, .relation 73468 3, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15306⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact73473RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6691⟩⟩, ⟨.program ⟨214⟩, ⟨26551⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6711⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨14949⟩⟩], [⟨.program ⟨214⟩, ⟨23781⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15306⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact73473RawTermsValid :
    exact73473RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event73473 : Event := .resultExact (⟨.program ⟨214⟩, ⟨20535⟩⟩) exact73473RawTerms .large 73305 (.finite 1811303510016) (some (73307))

def event73474 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26554⟩⟩) 0 ⟨20535⟩ 73473

def event73475 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26554⟩⟩) 1 ⟨26553⟩ 73295

def event73476 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26554⟩⟩) (.sum [.predecessor 0 73474 .coefficient, .predecessor 1 73475 .coefficient])

def event73477 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26554⟩⟩, .operator (⟨73473, 0⟩, ⟨73295, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6691⟩⟩, ⟨.program ⟨214⟩, ⟨26551⟩⟩]⟩, (1)⟩)

def event73478 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26554⟩⟩, .operator (⟨73473, 2⟩, ⟨73295, 1⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨14949⟩⟩], [⟨.program ⟨214⟩, ⟨23781⟩⟩]⟩, (-1)⟩)

def event73479 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26554⟩⟩) (.sum [.result 73473 .summary, .result 73295 .summary])

def exact73480RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6711⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15306⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact73480RawTermsValid :
    exact73480RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event73480 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26554⟩⟩) exact73480RawTerms .large 73476 (.finite 1291900380601931935744) (some (73479))

def event73481 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23716⟩⟩) 0 ⟨14789⟩ 3494

def event73482 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23716⟩⟩) (.authority (.programFamilyFact))

def event73483 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨23716⟩⟩) (.finite 3720)

def event73484 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23718⟩⟩) 0 ⟨6689⟩ 5477

def event73485 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23718⟩⟩) 1 ⟨23716⟩ 73483

def event73486 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23718⟩⟩) (.authority (.operator))

def exact73487RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨23718⟩⟩]⟩, (1)⟩]

theorem exact73487RawTermsValid :
    exact73487RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event73487 : Event := .resultExact (⟨.program ⟨214⟩, ⟨23718⟩⟩) exact73487RawTerms .large 73486 .exactZero (none)

def event73488 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26346⟩⟩) 0 ⟨23718⟩ 73487

def event73489 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26346⟩⟩) (.authority (.operator))

def exact73490RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨26346⟩⟩]⟩, (1)⟩]

theorem exact73490RawTermsValid :
    exact73490RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event73490 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26346⟩⟩) exact73490RawTerms (.finite 8192) 73489 .exactZero (none)

def event73491 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22951⟩⟩) 0 ⟨10474⟩ 3488

def event73492 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22951⟩⟩) (.authority (.programFamilyFact))

def event73493 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨22951⟩⟩) (.finite 3720)

def event73494 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22952⟩⟩) 0 ⟨6689⟩ 5477

def event73495 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22952⟩⟩) 1 ⟨22951⟩ 73493

def event73496 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22952⟩⟩) (.authority (.operator))

def exact73497RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨22952⟩⟩]⟩, (1)⟩]

theorem exact73497RawTermsValid :
    exact73497RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event73497 : Event := .resultExact (⟨.program ⟨214⟩, ⟨22952⟩⟩) exact73497RawTerms .large 73496 .exactZero (none)

def event73498 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24906⟩⟩) 0 ⟨22952⟩ 73497

def event73499 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24906⟩⟩) (.authority (.operator))

def exact73500RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨24906⟩⟩]⟩, (1)⟩]

theorem exact73500RawTermsValid :
    exact73500RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event73500 : Event := .resultExact (⟨.program ⟨214⟩, ⟨24906⟩⟩) exact73500RawTerms (.finite 8192) 73499 .exactZero (none)

def event73501 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10475⟩⟩) 0 ⟨10472⟩ 3477

def event73502 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10475⟩⟩) 1 ⟨6566⟩ 65295

def event73503 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10475⟩⟩) (.tensor (.predecessor 0 73501 .coefficient) (.predecessor 1 73502 .coefficient) true false)

def event73504 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨10475⟩⟩, .operator (⟨3477, 0⟩, ⟨65295, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨10472⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact73505RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨10472⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact73505RawTermsValid :
    exact73505RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event73505 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10475⟩⟩) exact73505RawTerms .large 73503 .exactZero (none)

def event73506 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7190⟩⟩) 0 ⟨5533⟩ 65165

def event73507 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7190⟩⟩) 1 ⟨6772⟩ 14989

def event73508 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7190⟩⟩) (.product (.predecessor 0 73506 .coefficient) (.predecessor 1 73507 .coefficient) (⟨false, false, none, none, none⟩))

def event73509 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7190⟩⟩, .operator (⟨65165, 0⟩, ⟨14989, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6772⟩⟩]⟩, (1)⟩)

def exact73510RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6772⟩⟩]⟩, (1)⟩]

theorem exact73510RawTermsValid :
    exact73510RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event73510 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7190⟩⟩) exact73510RawTerms .large 73508 .exactZero (none)

def event73511 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10476⟩⟩) 0 ⟨7190⟩ 73510

def event73512 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10476⟩⟩) 1 ⟨10475⟩ 73505

def event73513 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10476⟩⟩) (.sum [.predecessor 0 73511 .coefficient, .predecessor 1 73512 .coefficient])

def exact73514RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6772⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨10472⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact73514RawTermsValid :
    exact73514RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event73514 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10476⟩⟩) exact73514RawTerms .large 73513 .exactZero (none)

def event73515 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10477⟩⟩) 0 ⟨10476⟩ 73514

def event73516 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10477⟩⟩) 1 ⟨86⟩ 14981

def event73517 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10477⟩⟩) (.sum [.predecessor 0 73515 .coefficient, .predecessor 1 73516 .coefficient])

def event73518 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10477⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨86⟩⟩]⟩) [⟨.result 14981 .coefficient, false, none⟩])

def event73519 : Event := .survivorFold (1) 73518

def exact73520RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6772⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨10472⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact73520RawTermsValid :
    exact73520RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event73520 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10477⟩⟩) exact73520RawTerms .large 73517 (.finite 26) (some (73518))

def event73521 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10478⟩⟩) 0 ⟨10477⟩ 73520

def event73522 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10478⟩⟩) 1 ⟨9395⟩ 3480

def event73523 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10478⟩⟩) (.product (.predecessor 0 73521 .coefficient) (.predecessor 1 73522 .coefficient) (⟨false, true, none, none, some 1⟩))

def event73524 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10478⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨9395⟩⟩], []⟩) [⟨.result 3480 .coefficient, true, some 1⟩])

def event73525 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10478⟩⟩) (.product (.result 73520 .summary) (.transfer 73524) (⟨false, false, none, none, none⟩))

def event73526 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨10478⟩⟩, .operator (⟨73520, 1⟩, ⟨3480, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9395⟩⟩, ⟨.program ⟨214⟩, ⟨10472⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def event73527 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨10478⟩⟩, .operator (⟨73520, 0⟩, ⟨3480, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9395⟩⟩], [⟨.program ⟨214⟩, ⟨6772⟩⟩]⟩, (1)⟩)

def exact73528RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9395⟩⟩], [⟨.program ⟨214⟩, ⟨6772⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9395⟩⟩, ⟨.program ⟨214⟩, ⟨10472⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact73528RawTermsValid :
    exact73528RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event73528 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10478⟩⟩) exact73528RawTerms .large 73523 (.finite 1664) (some (73525))

def event73529 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9396⟩⟩) 0 ⟨9395⟩ 3480

def event73530 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9396⟩⟩) 1 ⟨6566⟩ 65295

def event73531 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨9396⟩⟩) (.tensor (.predecessor 0 73529 .coefficient) (.predecessor 1 73530 .coefficient) true false)

def event73532 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨9396⟩⟩, .operator (⟨3480, 0⟩, ⟨65295, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9395⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact73533RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9395⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact73533RawTermsValid :
    exact73533RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event73533 : Event := .resultExact (⟨.program ⟨214⟩, ⟨9396⟩⟩) exact73533RawTerms .large 73531 .exactZero (none)

def event73534 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7189⟩⟩) 0 ⟨5533⟩ 65165

def event73535 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7189⟩⟩) 1 ⟨6771⟩ 15030

def event73536 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7189⟩⟩) (.product (.predecessor 0 73534 .coefficient) (.predecessor 1 73535 .coefficient) (⟨false, false, none, none, none⟩))

def event73537 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7189⟩⟩, .operator (⟨65165, 0⟩, ⟨15030, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6771⟩⟩]⟩, (1)⟩)

def exact73538RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6771⟩⟩]⟩, (1)⟩]

theorem exact73538RawTermsValid :
    exact73538RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event73538 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7189⟩⟩) exact73538RawTerms .large 73536 .exactZero (none)

def event73539 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9397⟩⟩) 0 ⟨7189⟩ 73538

def event73540 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9397⟩⟩) 1 ⟨9396⟩ 73533

def event73541 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨9397⟩⟩) (.sum [.predecessor 0 73539 .coefficient, .predecessor 1 73540 .coefficient])

def exact73542RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6771⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9395⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact73542RawTermsValid :
    exact73542RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event73542 : Event := .resultExact (⟨.program ⟨214⟩, ⟨9397⟩⟩) exact73542RawTerms .large 73541 .exactZero (none)

def event73543 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9398⟩⟩) 0 ⟨9397⟩ 73542

def event73544 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9398⟩⟩) 1 ⟨85⟩ 15022

def event73545 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨9398⟩⟩) (.sum [.predecessor 0 73543 .coefficient, .predecessor 1 73544 .coefficient])

def event73546 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨9398⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨85⟩⟩]⟩) [⟨.result 15022 .coefficient, false, none⟩])

def event73547 : Event := .survivorFold (1) 73546

def exact73548RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6771⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9395⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact73548RawTermsValid :
    exact73548RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event73548 : Event := .resultExact (⟨.program ⟨214⟩, ⟨9398⟩⟩) exact73548RawTerms .large 73545 (.finite 26) (some (73546))

def event73549 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9399⟩⟩) 0 ⟨9398⟩ 73548

def event73550 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9399⟩⟩) 1 ⟨7832⟩ 15019

def event73551 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨9399⟩⟩) (.product (.predecessor 0 73549 .coefficient) (.predecessor 1 73550 .coefficient) (⟨false, false, none, none, none⟩))

def event73552 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨9399⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨7831⟩⟩]⟩) [⟨.result 15015 .coefficient, false, none⟩])

def event73553 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨9399⟩⟩) (.product (.result 73548 .summary) (.transfer 73552) (⟨false, false, none, none, none⟩))

def event73554 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨9399⟩⟩, .operator (⟨73548, 1⟩, ⟨15019, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9395⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7831⟩⟩]⟩, (-1)⟩)

def event73555 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨9399⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9395⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7831⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨7831⟩⟩) ⟨6772⟩ 14989)

def event73556 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨9399⟩⟩, .relation 73555 0, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9395⟩⟩], [⟨.program ⟨214⟩, ⟨6772⟩⟩]⟩, (-1)⟩)

def event73557 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨9399⟩⟩, .operator (⟨73548, 0⟩, ⟨15019, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6771⟩⟩, ⟨.program ⟨214⟩, ⟨7831⟩⟩]⟩, (1)⟩)

def exact73558RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6771⟩⟩, ⟨.program ⟨214⟩, ⟨7831⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9395⟩⟩], [⟨.program ⟨214⟩, ⟨6772⟩⟩]⟩, (-1)⟩]

theorem exact73558RawTermsValid :
    exact73558RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event73558 : Event := .resultExact (⟨.program ⟨214⟩, ⟨9399⟩⟩) exact73558RawTerms .large 73551 (.finite 95420416) (some (73553))

def event73559 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10479⟩⟩) 0 ⟨9399⟩ 73558

def event73560 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10479⟩⟩) 1 ⟨10478⟩ 73528

def event73561 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10479⟩⟩) (.sum [.predecessor 0 73559 .coefficient, .predecessor 1 73560 .coefficient])

def event73562 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨10479⟩⟩, .operator (⟨73558, 1⟩, ⟨73528, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9395⟩⟩], [⟨.program ⟨214⟩, ⟨6772⟩⟩]⟩, (1)⟩)

def event73563 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10479⟩⟩) (.sum [.result 73558 .summary, .result 73528 .summary])

def exact73564RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6771⟩⟩, ⟨.program ⟨214⟩, ⟨7831⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9395⟩⟩, ⟨.program ⟨214⟩, ⟨10472⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact73564RawTermsValid :
    exact73564RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event73564 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10479⟩⟩) exact73564RawTerms .large 73561 (.finite 95422080) (some (73563))

def event73565 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24907⟩⟩) 0 ⟨10479⟩ 73564

def event73566 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24907⟩⟩) 1 ⟨24906⟩ 73500

def event73567 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24907⟩⟩) (.product (.predecessor 0 73565 .coefficient) (.predecessor 1 73566 .coefficient) (⟨false, false, none, none, none⟩))

def event73568 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24907⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨24906⟩⟩]⟩) [⟨.result 73500 .coefficient, false, none⟩])

def event73569 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24907⟩⟩) (.product (.result 73564 .summary) (.transfer 73568) (⟨false, false, none, none, none⟩))

def event73570 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨24907⟩⟩, .operator (⟨73564, 1⟩, ⟨73500, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9395⟩⟩, ⟨.program ⟨214⟩, ⟨10472⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨24906⟩⟩]⟩, (-1)⟩)

def event73571 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨24907⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9395⟩⟩, ⟨.program ⟨214⟩, ⟨10472⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨24906⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨24906⟩⟩) ⟨22952⟩ 73497)

def event73572 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨24907⟩⟩, .relation 73571 0, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9395⟩⟩, ⟨.program ⟨214⟩, ⟨10472⟩⟩], [⟨.program ⟨214⟩, ⟨22952⟩⟩]⟩, (-1)⟩)

def event73573 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨24907⟩⟩, .operator (⟨73564, 0⟩, ⟨73500, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6771⟩⟩, ⟨.program ⟨214⟩, ⟨7831⟩⟩, ⟨.program ⟨214⟩, ⟨24906⟩⟩]⟩, (1)⟩)

def exact73574RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6771⟩⟩, ⟨.program ⟨214⟩, ⟨7831⟩⟩, ⟨.program ⟨214⟩, ⟨24906⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9395⟩⟩, ⟨.program ⟨214⟩, ⟨10472⟩⟩], [⟨.program ⟨214⟩, ⟨22952⟩⟩]⟩, (-1)⟩]

theorem exact73574RawTermsValid :
    exact73574RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event73574 : Event := .resultExact (⟨.program ⟨214⟩, ⟨24907⟩⟩) exact73574RawTerms .large 73567 (.finite 350200560353280) (some (73569))

def event73575 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19020⟩⟩) 0 ⟨10474⟩ 3488

def event73576 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19020⟩⟩) (.authority (.relationPreimageSource ⟨7⟩))

def exact73577RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨19020⟩⟩]⟩, (1)⟩]

theorem exact73577RawTermsValid :
    exact73577RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event73577 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19020⟩⟩) exact73577RawTerms (.finite 136065468) 73576 .exactZero (none)

def event73578 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19022⟩⟩) 0 ⟨19020⟩ 73577

def event73579 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19022⟩⟩) 1 ⟨2348⟩ 4

def event73580 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19022⟩⟩) (.scale (.predecessor 0 73578 .coefficient) (.value (.predecessor 1 73579 .coefficient)))

def exact73581RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨19020⟩⟩]⟩, (1)⟩]

theorem exact73581RawTermsValid :
    exact73581RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event73581 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19022⟩⟩) exact73581RawTerms (.finite 136065468) 73580 .exactZero (none)

def event73582 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19023⟩⟩) 0 ⟨5535⟩ 65387

def event73583 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19023⟩⟩) 1 ⟨19022⟩ 73581

def event73584 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19023⟩⟩) (.product (.predecessor 0 73582 .coefficient) (.predecessor 1 73583 .coefficient) (⟨false, false, none, none, none⟩))

def event73585 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19023⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨19020⟩⟩]⟩) [⟨.result 73577 .coefficient, false, none⟩])

def event73586 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19023⟩⟩) (.product (.result 65387 .summary) (.transfer 73585) (⟨false, false, none, none, none⟩))

def event73587 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19023⟩⟩, .operator (⟨65387, 0⟩, ⟨73581, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19020⟩⟩]⟩, (1)⟩)

def event73588 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨19021⟩⟩)

def event73589 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event73590 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event73591 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨360⟩⟩) (.authority (.operator))

def event73592 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨360⟩⟩) (.finite 2)

def event73593 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event73594 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event73595 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event73596 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event73597 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 73596

def event73598 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 73594

def event73599 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 73597 .coefficient) (.value (.predecessor 1 73598 .coefficient)))

def event73600 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event73601 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5504⟩⟩) 0 ⟨5503⟩ 73600

def event73602 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5504⟩⟩) 1 ⟨360⟩ 73592

def event73603 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5504⟩⟩) (.sum [.predecessor 0 73601 .coefficient, .predecessor 1 73602 .coefficient])

def event73604 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5504⟩⟩) (.finite 219)

def event73605 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5530⟩⟩) 0 ⟨5504⟩ 73604

def event73606 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5530⟩⟩) 1 ⟨961⟩ 73590

def event73607 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5530⟩⟩) (.identity (.predecessor 1 73606 .coefficient))

def event73608 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5530⟩⟩) (.finite 224)

def event73609 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10472⟩⟩) 0 ⟨5530⟩ 73608

def event73610 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10472⟩⟩) (.authority (.programFamilyFact))

def exact73611RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨10472⟩⟩], []⟩, (1)⟩]

theorem exact73611RawTermsValid :
    exact73611RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event73611 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10472⟩⟩) exact73611RawTerms (.finite 2) 73610 .exactZero (none)

def event73612 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9395⟩⟩) 0 ⟨5530⟩ 73608

def event73613 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨9395⟩⟩) (.authority (.programFamilyFact))

def exact73614RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨9395⟩⟩], []⟩, (1)⟩]

theorem exact73614RawTermsValid :
    exact73614RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event73614 : Event := .resultExact (⟨.program ⟨214⟩, ⟨9395⟩⟩) exact73614RawTerms (.finite 2) 73613 .exactZero (none)

def event73615 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10473⟩⟩) 0 ⟨9395⟩ 73614

def event73616 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10473⟩⟩) 1 ⟨10472⟩ 73611

def event73617 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10473⟩⟩) (.product (.predecessor 0 73615 .coefficient) (.predecessor 1 73616 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event73618 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10473⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨9395⟩⟩, ⟨.program ⟨214⟩, ⟨10472⟩⟩], []⟩) [⟨.result 73614 .coefficient, true, some 1⟩, ⟨.result 73611 .coefficient, true, some 1⟩])

def event73619 : Event := .survivorFold (1) 73618

def exact73620RawTerms : List Term := []

theorem exact73620RawTermsValid :
    exact73620RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event73620 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10473⟩⟩) exact73620RawTerms (.finite 4) 73617 (.finite 4) (some (73618))

def event73621 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10474⟩⟩) 0 ⟨10473⟩ 73620

def event73622 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10474⟩⟩) (.identity (.predecessor 0 73621 .coefficient))

def event73623 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨10474⟩⟩) (.finite 4)

def event73624 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19020⟩⟩) 0 ⟨10474⟩ 73623

def event73625 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19020⟩⟩) (.authority (.relationPreimageSource ⟨7⟩))

def exact73626RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨19020⟩⟩]⟩, (1)⟩]

theorem exact73626RawTermsValid :
    exact73626RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event73626 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19020⟩⟩) exact73626RawTerms (.finite 136065468) 73625 .exactZero (none)

def event73627 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6⟩⟩) (.authority (.operator))

def exact73628RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩]⟩, (1)⟩]

theorem exact73628RawTermsValid :
    exact73628RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event73628 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6⟩⟩) exact73628RawTerms .large 73627 .exactZero (none)

def event73629 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19021⟩⟩) 0 ⟨6⟩ 73628

def event73630 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19021⟩⟩) 1 ⟨19020⟩ 73626

def event73631 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19021⟩⟩) (.product (.predecessor 0 73629 .coefficient) (.predecessor 1 73630 .coefficient) (⟨false, false, none, none, none⟩))

def event73632 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19021⟩⟩, .operator (⟨73628, 0⟩, ⟨73626, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19020⟩⟩]⟩, (1)⟩)

def exact73633RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19020⟩⟩]⟩, (1)⟩]

theorem exact73633RawTermsValid :
    exact73633RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event73633 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19021⟩⟩) exact73633RawTerms .large 73631 .exactZero (none)

def event73634 : Event := .preFoldPolynomial 73633 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19020⟩⟩]⟩, (1)⟩] .exactZero none

def exact73635RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19020⟩⟩]⟩, (1)⟩]

def event73635 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨19021⟩⟩) 73634 exact73635RawTerms .large 73631 .exactZero (none)

def event73636 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨24910⟩⟩)

def event73637 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event73638 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event73639 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨360⟩⟩) (.authority (.operator))

def event73640 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨360⟩⟩) (.finite 2)

def event73641 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event73642 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event73643 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event73644 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event73645 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 73644

def event73646 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 73642

def event73647 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 73645 .coefficient) (.value (.predecessor 1 73646 .coefficient)))

def event73648 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event73649 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5504⟩⟩) 0 ⟨5503⟩ 73648

def event73650 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5504⟩⟩) 1 ⟨360⟩ 73640

def event73651 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5504⟩⟩) (.sum [.predecessor 0 73649 .coefficient, .predecessor 1 73650 .coefficient])

def event73652 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5504⟩⟩) (.finite 219)

def event73653 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5530⟩⟩) 0 ⟨5504⟩ 73652

def event73654 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5530⟩⟩) 1 ⟨961⟩ 73638

def event73655 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5530⟩⟩) (.identity (.predecessor 1 73654 .coefficient))

def event73656 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5530⟩⟩) (.finite 224)

def event73657 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10472⟩⟩) 0 ⟨5530⟩ 73656

def event73658 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10472⟩⟩) (.authority (.programFamilyFact))

def exact73659RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨10472⟩⟩], []⟩, (1)⟩]

theorem exact73659RawTermsValid :
    exact73659RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event73659 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10472⟩⟩) exact73659RawTerms (.finite 2) 73658 .exactZero (none)

def event73660 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9395⟩⟩) 0 ⟨5530⟩ 73656

def event73661 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨9395⟩⟩) (.authority (.programFamilyFact))

def exact73662RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨9395⟩⟩], []⟩, (1)⟩]

theorem exact73662RawTermsValid :
    exact73662RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event73662 : Event := .resultExact (⟨.program ⟨214⟩, ⟨9395⟩⟩) exact73662RawTerms (.finite 2) 73661 .exactZero (none)

def event73663 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10473⟩⟩) 0 ⟨9395⟩ 73662

def event73664 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10473⟩⟩) 1 ⟨10472⟩ 73659

def event73665 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10473⟩⟩) (.product (.predecessor 0 73663 .coefficient) (.predecessor 1 73664 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event73666 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨10473⟩⟩, .operator (⟨73662, 0⟩, ⟨73659, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨9395⟩⟩, ⟨.program ⟨214⟩, ⟨10472⟩⟩], []⟩, (1)⟩)

def exact73667RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨9395⟩⟩, ⟨.program ⟨214⟩, ⟨10472⟩⟩], []⟩, (1)⟩]

theorem exact73667RawTermsValid :
    exact73667RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event73667 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10473⟩⟩) exact73667RawTerms (.finite 4) 73665 .exactZero (none)

def event73668 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10474⟩⟩) 0 ⟨10473⟩ 73667

def event73669 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10474⟩⟩) (.identity (.predecessor 0 73668 .coefficient))

def event73670 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨10474⟩⟩) (.finite 4)

def event73671 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22951⟩⟩) 0 ⟨10474⟩ 73670

def event73672 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22951⟩⟩) (.authority (.programFamilyFact))

def event73673 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨22951⟩⟩) (.finite 3720)

def event73674 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨6689⟩⟩) .missing

def event73675 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22952⟩⟩) 0 ⟨6689⟩ 73674

def event73676 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22952⟩⟩) 1 ⟨22951⟩ 73673

def event73677 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22952⟩⟩) (.authority (.operator))

def exact73678RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨22952⟩⟩]⟩, (1)⟩]

theorem exact73678RawTermsValid :
    exact73678RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event73678 : Event := .resultExact (⟨.program ⟨214⟩, ⟨22952⟩⟩) exact73678RawTerms .large 73677 .exactZero (none)

def event73679 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24906⟩⟩) 0 ⟨22952⟩ 73678

def event73680 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24906⟩⟩) (.authority (.operator))

def exact73681RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨24906⟩⟩]⟩, (1)⟩]

theorem exact73681RawTermsValid :
    exact73681RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event73681 : Event := .resultExact (⟨.program ⟨214⟩, ⟨24906⟩⟩) exact73681RawTerms (.finite 8192) 73680 .exactZero (none)

def event73682 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨110⟩⟩) (.authority (.operator))

def event73683 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨110⟩⟩) .exactZero

def event73684 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10572⟩⟩) 0 ⟨10474⟩ 73670

def event73685 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10572⟩⟩) 1 ⟨110⟩ 73683

def event73686 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10572⟩⟩) (.sum [.predecessor 0 73684 .coefficient, .predecessor 1 73685 .coefficient])

def event73687 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨10572⟩⟩) (.finite 4)

def event73688 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10573⟩⟩) 0 ⟨10572⟩ 73687

def event73689 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10573⟩⟩) (.identity (.predecessor 0 73688 .coefficient))

def exact73690RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨9395⟩⟩, ⟨.program ⟨214⟩, ⟨10472⟩⟩], []⟩, (1)⟩]

theorem exact73690RawTermsValid :
    exact73690RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event73690 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10573⟩⟩) exact73690RawTerms (.finite 4) 73689 .exactZero (none)

def event73691 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6544⟩⟩) (.authority (.factStore))

def exact73692RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact73692RawTermsValid :
    exact73692RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event73692 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6544⟩⟩) exact73692RawTerms .large 73691 .exactZero (none)

def event73693 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10574⟩⟩) 0 ⟨6544⟩ 73692

def event73694 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10574⟩⟩) 1 ⟨10573⟩ 73690

def event73695 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10574⟩⟩) (.product (.predecessor 0 73693 .coefficient) (.predecessor 1 73694 .coefficient) (⟨false, false, none, none, none⟩))

def event73696 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨10574⟩⟩, .operator (⟨73692, 0⟩, ⟨73690, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨9395⟩⟩, ⟨.program ⟨214⟩, ⟨10472⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact73697RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨9395⟩⟩, ⟨.program ⟨214⟩, ⟨10472⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact73697RawTermsValid :
    exact73697RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event73697 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10574⟩⟩) exact73697RawTerms .large 73695 .exactZero (none)

def event73698 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.authority (.operator))

def event73699 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.finite 1)

def event73700 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6757⟩⟩) 0 ⟨6689⟩ 73674

def event73701 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6757⟩⟩) (.authority (.operator))

def exact73702RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6757⟩⟩]⟩, (1)⟩]

theorem exact73702RawTermsValid :
    exact73702RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event73702 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6757⟩⟩) exact73702RawTerms .large 73701 .exactZero (none)

def event73703 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6772⟩⟩) 0 ⟨6757⟩ 73702

def event73704 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6772⟩⟩) (.identity (.predecessor 0 73703 .coefficient))

def exact73705RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6772⟩⟩]⟩, (1)⟩]

theorem exact73705RawTermsValid :
    exact73705RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event73705 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6772⟩⟩) exact73705RawTerms .large 73704 .exactZero (none)

def event73706 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7831⟩⟩) 0 ⟨6772⟩ 73705

def event73707 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7831⟩⟩) (.authority (.operator))

def exact73708RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨7831⟩⟩]⟩, (1)⟩]

theorem exact73708RawTermsValid :
    exact73708RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event73708 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7831⟩⟩) exact73708RawTerms (.finite 8192) 73707 .exactZero (none)

def event73709 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7832⟩⟩) 0 ⟨7831⟩ 73708

def event73710 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7832⟩⟩) 1 ⟨2348⟩ 73699

def event73711 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7832⟩⟩) (.scale (.predecessor 0 73709 .coefficient) (.value (.predecessor 1 73710 .coefficient)))

def exact73712RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨7831⟩⟩]⟩, (1)⟩]

theorem exact73712RawTermsValid :
    exact73712RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event73712 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7832⟩⟩) exact73712RawTerms (.finite 8192) 73711 .exactZero (none)

def event73713 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6771⟩⟩) 0 ⟨6757⟩ 73702

def event73714 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6771⟩⟩) (.identity (.predecessor 0 73713 .coefficient))

def exact73715RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6771⟩⟩]⟩, (1)⟩]

theorem exact73715RawTermsValid :
    exact73715RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event73715 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6771⟩⟩) exact73715RawTerms .large 73714 .exactZero (none)

def event73716 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7833⟩⟩) 0 ⟨6771⟩ 73715

def event73717 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7833⟩⟩) 1 ⟨7832⟩ 73712

def event73718 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7833⟩⟩) (.product (.predecessor 0 73716 .coefficient) (.predecessor 1 73717 .coefficient) (⟨false, false, none, none, none⟩))

def event73719 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7833⟩⟩, .operator (⟨73715, 0⟩, ⟨73712, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6771⟩⟩, ⟨.program ⟨214⟩, ⟨7831⟩⟩]⟩, (1)⟩)

def exact73720RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6771⟩⟩, ⟨.program ⟨214⟩, ⟨7831⟩⟩]⟩, (1)⟩]

theorem exact73720RawTermsValid :
    exact73720RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event73720 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7833⟩⟩) exact73720RawTerms .large 73718 .exactZero (none)

def event73721 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10575⟩⟩) 0 ⟨7833⟩ 73720

def event73722 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10575⟩⟩) 1 ⟨10574⟩ 73697

def event73723 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10575⟩⟩) (.sum [.predecessor 0 73721 .coefficient, .predecessor 1 73722 .coefficient])

def exact73724RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6771⟩⟩, ⟨.program ⟨214⟩, ⟨7831⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨9395⟩⟩, ⟨.program ⟨214⟩, ⟨10472⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact73724RawTermsValid :
    exact73724RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event73724 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10575⟩⟩) exact73724RawTerms .large 73723 .exactZero (none)

def event73725 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24909⟩⟩) 0 ⟨10575⟩ 73724

def event73726 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24909⟩⟩) 1 ⟨24906⟩ 73681

def event73727 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24909⟩⟩) (.product (.predecessor 0 73725 .coefficient) (.predecessor 1 73726 .coefficient) (⟨false, false, none, none, none⟩))

def eventLeaf4592 : Array AnnotatedEvent := #[
  { event := event73472
    frameStart := 0 },
  { event := event73473
    frameStart := 0 },
  { event := event73474
    frameStart := 0 },
  { event := event73475
    frameStart := 0 },
  { event := event73476
    frameStart := 0 },
  { event := event73477
    frameStart := 0 },
  { event := event73478
    frameStart := 0 },
  { event := event73479
    frameStart := 0 },
  { event := event73480
    frameStart := 0 },
  { event := event73481
    frameStart := 0 },
  { event := event73482
    frameStart := 0 },
  { event := event73483
    frameStart := 0 },
  { event := event73484
    frameStart := 0 },
  { event := event73485
    frameStart := 0 },
  { event := event73486
    frameStart := 0 },
  { event := event73487
    frameStart := 0 }
]

def eventLeaf4593 : Array AnnotatedEvent := #[
  { event := event73488
    frameStart := 0 },
  { event := event73489
    frameStart := 0 },
  { event := event73490
    frameStart := 0 },
  { event := event73491
    frameStart := 0 },
  { event := event73492
    frameStart := 0 },
  { event := event73493
    frameStart := 0 },
  { event := event73494
    frameStart := 0 },
  { event := event73495
    frameStart := 0 },
  { event := event73496
    frameStart := 0 },
  { event := event73497
    frameStart := 0 },
  { event := event73498
    frameStart := 0 },
  { event := event73499
    frameStart := 0 },
  { event := event73500
    frameStart := 0 },
  { event := event73501
    frameStart := 0 },
  { event := event73502
    frameStart := 0 },
  { event := event73503
    frameStart := 0 }
]

def eventLeaf4594 : Array AnnotatedEvent := #[
  { event := event73504
    frameStart := 0 },
  { event := event73505
    frameStart := 0 },
  { event := event73506
    frameStart := 0 },
  { event := event73507
    frameStart := 0 },
  { event := event73508
    frameStart := 0 },
  { event := event73509
    frameStart := 0 },
  { event := event73510
    frameStart := 0 },
  { event := event73511
    frameStart := 0 },
  { event := event73512
    frameStart := 0 },
  { event := event73513
    frameStart := 0 },
  { event := event73514
    frameStart := 0 },
  { event := event73515
    frameStart := 0 },
  { event := event73516
    frameStart := 0 },
  { event := event73517
    frameStart := 0 },
  { event := event73518
    frameStart := 0 },
  { event := event73519
    frameStart := 0 }
]

def eventLeaf4595 : Array AnnotatedEvent := #[
  { event := event73520
    frameStart := 0 },
  { event := event73521
    frameStart := 0 },
  { event := event73522
    frameStart := 0 },
  { event := event73523
    frameStart := 0 },
  { event := event73524
    frameStart := 0 },
  { event := event73525
    frameStart := 0 },
  { event := event73526
    frameStart := 0 },
  { event := event73527
    frameStart := 0 },
  { event := event73528
    frameStart := 0 },
  { event := event73529
    frameStart := 0 },
  { event := event73530
    frameStart := 0 },
  { event := event73531
    frameStart := 0 },
  { event := event73532
    frameStart := 0 },
  { event := event73533
    frameStart := 0 },
  { event := event73534
    frameStart := 0 },
  { event := event73535
    frameStart := 0 }
]

def eventLeaf4596 : Array AnnotatedEvent := #[
  { event := event73536
    frameStart := 0 },
  { event := event73537
    frameStart := 0 },
  { event := event73538
    frameStart := 0 },
  { event := event73539
    frameStart := 0 },
  { event := event73540
    frameStart := 0 },
  { event := event73541
    frameStart := 0 },
  { event := event73542
    frameStart := 0 },
  { event := event73543
    frameStart := 0 },
  { event := event73544
    frameStart := 0 },
  { event := event73545
    frameStart := 0 },
  { event := event73546
    frameStart := 0 },
  { event := event73547
    frameStart := 0 },
  { event := event73548
    frameStart := 0 },
  { event := event73549
    frameStart := 0 },
  { event := event73550
    frameStart := 0 },
  { event := event73551
    frameStart := 0 }
]

def eventLeaf4597 : Array AnnotatedEvent := #[
  { event := event73552
    frameStart := 0 },
  { event := event73553
    frameStart := 0 },
  { event := event73554
    frameStart := 0 },
  { event := event73555
    frameStart := 0 },
  { event := event73556
    frameStart := 0 },
  { event := event73557
    frameStart := 0 },
  { event := event73558
    frameStart := 0 },
  { event := event73559
    frameStart := 0 },
  { event := event73560
    frameStart := 0 },
  { event := event73561
    frameStart := 0 },
  { event := event73562
    frameStart := 0 },
  { event := event73563
    frameStart := 0 },
  { event := event73564
    frameStart := 0 },
  { event := event73565
    frameStart := 0 },
  { event := event73566
    frameStart := 0 },
  { event := event73567
    frameStart := 0 }
]

def eventLeaf4598 : Array AnnotatedEvent := #[
  { event := event73568
    frameStart := 0 },
  { event := event73569
    frameStart := 0 },
  { event := event73570
    frameStart := 0 },
  { event := event73571
    frameStart := 0 },
  { event := event73572
    frameStart := 0 },
  { event := event73573
    frameStart := 0 },
  { event := event73574
    frameStart := 0 },
  { event := event73575
    frameStart := 0 },
  { event := event73576
    frameStart := 0 },
  { event := event73577
    frameStart := 0 },
  { event := event73578
    frameStart := 0 },
  { event := event73579
    frameStart := 0 },
  { event := event73580
    frameStart := 0 },
  { event := event73581
    frameStart := 0 },
  { event := event73582
    frameStart := 0 },
  { event := event73583
    frameStart := 0 }
]

def eventLeaf4599 : Array AnnotatedEvent := #[
  { event := event73584
    frameStart := 0 },
  { event := event73585
    frameStart := 0 },
  { event := event73586
    frameStart := 0 },
  { event := event73587
    frameStart := 0 },
  { event := event73588
    frameStart := 73588 },
  { event := event73589
    frameStart := 73588 },
  { event := event73590
    frameStart := 73588 },
  { event := event73591
    frameStart := 73588 },
  { event := event73592
    frameStart := 73588 },
  { event := event73593
    frameStart := 73588 },
  { event := event73594
    frameStart := 73588 },
  { event := event73595
    frameStart := 73588 },
  { event := event73596
    frameStart := 73588 },
  { event := event73597
    frameStart := 73588 },
  { event := event73598
    frameStart := 73588 },
  { event := event73599
    frameStart := 73588 }
]

def eventLeaf4600 : Array AnnotatedEvent := #[
  { event := event73600
    frameStart := 73588 },
  { event := event73601
    frameStart := 73588 },
  { event := event73602
    frameStart := 73588 },
  { event := event73603
    frameStart := 73588 },
  { event := event73604
    frameStart := 73588 },
  { event := event73605
    frameStart := 73588 },
  { event := event73606
    frameStart := 73588 },
  { event := event73607
    frameStart := 73588 },
  { event := event73608
    frameStart := 73588 },
  { event := event73609
    frameStart := 73588 },
  { event := event73610
    frameStart := 73588 },
  { event := event73611
    frameStart := 73588 },
  { event := event73612
    frameStart := 73588 },
  { event := event73613
    frameStart := 73588 },
  { event := event73614
    frameStart := 73588 },
  { event := event73615
    frameStart := 73588 }
]

def eventLeaf4601 : Array AnnotatedEvent := #[
  { event := event73616
    frameStart := 73588 },
  { event := event73617
    frameStart := 73588 },
  { event := event73618
    frameStart := 73588 },
  { event := event73619
    frameStart := 73588 },
  { event := event73620
    frameStart := 73588 },
  { event := event73621
    frameStart := 73588 },
  { event := event73622
    frameStart := 73588 },
  { event := event73623
    frameStart := 73588 },
  { event := event73624
    frameStart := 73588 },
  { event := event73625
    frameStart := 73588 },
  { event := event73626
    frameStart := 73588 },
  { event := event73627
    frameStart := 73588 },
  { event := event73628
    frameStart := 73588 },
  { event := event73629
    frameStart := 73588 },
  { event := event73630
    frameStart := 73588 },
  { event := event73631
    frameStart := 73588 }
]

def eventLeaf4602 : Array AnnotatedEvent := #[
  { event := event73632
    frameStart := 73588 },
  { event := event73633
    frameStart := 73588 },
  { event := event73634
    frameStart := 73588 },
  { event := event73635
    frameStart := 73588 },
  { event := event73636
    frameStart := 73636 },
  { event := event73637
    frameStart := 73636 },
  { event := event73638
    frameStart := 73636 },
  { event := event73639
    frameStart := 73636 },
  { event := event73640
    frameStart := 73636 },
  { event := event73641
    frameStart := 73636 },
  { event := event73642
    frameStart := 73636 },
  { event := event73643
    frameStart := 73636 },
  { event := event73644
    frameStart := 73636 },
  { event := event73645
    frameStart := 73636 },
  { event := event73646
    frameStart := 73636 },
  { event := event73647
    frameStart := 73636 }
]

def eventLeaf4603 : Array AnnotatedEvent := #[
  { event := event73648
    frameStart := 73636 },
  { event := event73649
    frameStart := 73636 },
  { event := event73650
    frameStart := 73636 },
  { event := event73651
    frameStart := 73636 },
  { event := event73652
    frameStart := 73636 },
  { event := event73653
    frameStart := 73636 },
  { event := event73654
    frameStart := 73636 },
  { event := event73655
    frameStart := 73636 },
  { event := event73656
    frameStart := 73636 },
  { event := event73657
    frameStart := 73636 },
  { event := event73658
    frameStart := 73636 },
  { event := event73659
    frameStart := 73636 },
  { event := event73660
    frameStart := 73636 },
  { event := event73661
    frameStart := 73636 },
  { event := event73662
    frameStart := 73636 },
  { event := event73663
    frameStart := 73636 }
]

def eventLeaf4604 : Array AnnotatedEvent := #[
  { event := event73664
    frameStart := 73636 },
  { event := event73665
    frameStart := 73636 },
  { event := event73666
    frameStart := 73636 },
  { event := event73667
    frameStart := 73636 },
  { event := event73668
    frameStart := 73636 },
  { event := event73669
    frameStart := 73636 },
  { event := event73670
    frameStart := 73636 },
  { event := event73671
    frameStart := 73636 },
  { event := event73672
    frameStart := 73636 },
  { event := event73673
    frameStart := 73636 },
  { event := event73674
    frameStart := 73636 },
  { event := event73675
    frameStart := 73636 },
  { event := event73676
    frameStart := 73636 },
  { event := event73677
    frameStart := 73636 },
  { event := event73678
    frameStart := 73636 },
  { event := event73679
    frameStart := 73636 }
]

def eventLeaf4605 : Array AnnotatedEvent := #[
  { event := event73680
    frameStart := 73636 },
  { event := event73681
    frameStart := 73636 },
  { event := event73682
    frameStart := 73636 },
  { event := event73683
    frameStart := 73636 },
  { event := event73684
    frameStart := 73636 },
  { event := event73685
    frameStart := 73636 },
  { event := event73686
    frameStart := 73636 },
  { event := event73687
    frameStart := 73636 },
  { event := event73688
    frameStart := 73636 },
  { event := event73689
    frameStart := 73636 },
  { event := event73690
    frameStart := 73636 },
  { event := event73691
    frameStart := 73636 },
  { event := event73692
    frameStart := 73636 },
  { event := event73693
    frameStart := 73636 },
  { event := event73694
    frameStart := 73636 },
  { event := event73695
    frameStart := 73636 }
]

def eventLeaf4606 : Array AnnotatedEvent := #[
  { event := event73696
    frameStart := 73636 },
  { event := event73697
    frameStart := 73636 },
  { event := event73698
    frameStart := 73636 },
  { event := event73699
    frameStart := 73636 },
  { event := event73700
    frameStart := 73636 },
  { event := event73701
    frameStart := 73636 },
  { event := event73702
    frameStart := 73636 },
  { event := event73703
    frameStart := 73636 },
  { event := event73704
    frameStart := 73636 },
  { event := event73705
    frameStart := 73636 },
  { event := event73706
    frameStart := 73636 },
  { event := event73707
    frameStart := 73636 },
  { event := event73708
    frameStart := 73636 },
  { event := event73709
    frameStart := 73636 },
  { event := event73710
    frameStart := 73636 },
  { event := event73711
    frameStart := 73636 }
]

def eventLeaf4607 : Array AnnotatedEvent := #[
  { event := event73712
    frameStart := 73636 },
  { event := event73713
    frameStart := 73636 },
  { event := event73714
    frameStart := 73636 },
  { event := event73715
    frameStart := 73636 },
  { event := event73716
    frameStart := 73636 },
  { event := event73717
    frameStart := 73636 },
  { event := event73718
    frameStart := 73636 },
  { event := event73719
    frameStart := 73636 },
  { event := event73720
    frameStart := 73636 },
  { event := event73721
    frameStart := 73636 },
  { event := event73722
    frameStart := 73636 },
  { event := event73723
    frameStart := 73636 },
  { event := event73724
    frameStart := 73636 },
  { event := event73725
    frameStart := 73636 },
  { event := event73726
    frameStart := 73636 },
  { event := event73727
    frameStart := 73636 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events287
