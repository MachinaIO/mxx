import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events564

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event144384 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨69061⟩⟩, .operator (⟨144363, 0⟩, ⟨144361, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨15923⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact144385RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15923⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨18733⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨21953⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨26528⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨29208⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨31973⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨34872⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨37552⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨40228⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨42908⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨45592⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨48272⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨51028⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨54008⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨56988⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨59968⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨62948⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨66111⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact144385RawTermsValid :
    exact144385RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event144385 : Event := .resultExact (⟨.program ⟨257⟩, ⟨69061⟩⟩) exact144385RawTerms .large 144366 .exactZero (none)

def event144386 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7232⟩⟩) 0 ⟨7177⟩ 144345

def event144387 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7232⟩⟩) (.authority (.operator))

def exact144388RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7232⟩⟩]⟩, (1)⟩]

theorem exact144388RawTermsValid :
    exact144388RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event144388 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7232⟩⟩) exact144388RawTerms .large 144387 .exactZero (none)

def event144389 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7230⟩⟩) 0 ⟨7177⟩ 144345

def event144390 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7230⟩⟩) (.authority (.operator))

def exact144391RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7230⟩⟩]⟩, (1)⟩]

theorem exact144391RawTermsValid :
    exact144391RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event144391 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7230⟩⟩) exact144391RawTerms .large 144390 .exactZero (none)

def event144392 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7228⟩⟩) 0 ⟨7177⟩ 144345

def event144393 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7228⟩⟩) (.authority (.operator))

def exact144394RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7228⟩⟩]⟩, (1)⟩]

theorem exact144394RawTermsValid :
    exact144394RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event144394 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7228⟩⟩) exact144394RawTerms .large 144393 .exactZero (none)

def event144395 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7226⟩⟩) 0 ⟨7177⟩ 144345

def event144396 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7226⟩⟩) (.authority (.operator))

def exact144397RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7226⟩⟩]⟩, (1)⟩]

theorem exact144397RawTermsValid :
    exact144397RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event144397 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7226⟩⟩) exact144397RawTerms .large 144396 .exactZero (none)

def event144398 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7224⟩⟩) 0 ⟨7177⟩ 144345

def event144399 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7224⟩⟩) (.authority (.operator))

def exact144400RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7224⟩⟩]⟩, (1)⟩]

theorem exact144400RawTermsValid :
    exact144400RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event144400 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7224⟩⟩) exact144400RawTerms .large 144399 .exactZero (none)

def event144401 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7222⟩⟩) 0 ⟨7177⟩ 144345

def event144402 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7222⟩⟩) (.authority (.operator))

def exact144403RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩]

theorem exact144403RawTermsValid :
    exact144403RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event144403 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7222⟩⟩) exact144403RawTerms .large 144402 .exactZero (none)

def event144404 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7220⟩⟩) 0 ⟨7177⟩ 144345

def event144405 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7220⟩⟩) (.authority (.operator))

def exact144406RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩]

theorem exact144406RawTermsValid :
    exact144406RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event144406 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7220⟩⟩) exact144406RawTerms .large 144405 .exactZero (none)

def event144407 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7218⟩⟩) 0 ⟨7177⟩ 144345

def event144408 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7218⟩⟩) (.authority (.operator))

def exact144409RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩]

theorem exact144409RawTermsValid :
    exact144409RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event144409 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7218⟩⟩) exact144409RawTerms .large 144408 .exactZero (none)

def event144410 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7216⟩⟩) 0 ⟨7177⟩ 144345

def event144411 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7216⟩⟩) (.authority (.operator))

def exact144412RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩]

theorem exact144412RawTermsValid :
    exact144412RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event144412 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7216⟩⟩) exact144412RawTerms .large 144411 .exactZero (none)

def event144413 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7214⟩⟩) 0 ⟨7177⟩ 144345

def event144414 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7214⟩⟩) (.authority (.operator))

def exact144415RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩]

theorem exact144415RawTermsValid :
    exact144415RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event144415 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7214⟩⟩) exact144415RawTerms .large 144414 .exactZero (none)

def event144416 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7212⟩⟩) 0 ⟨7177⟩ 144345

def event144417 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7212⟩⟩) (.authority (.operator))

def exact144418RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩]

theorem exact144418RawTermsValid :
    exact144418RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event144418 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7212⟩⟩) exact144418RawTerms .large 144417 .exactZero (none)

def event144419 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7210⟩⟩) 0 ⟨7177⟩ 144345

def event144420 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7210⟩⟩) (.authority (.operator))

def exact144421RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩]

theorem exact144421RawTermsValid :
    exact144421RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event144421 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7210⟩⟩) exact144421RawTerms .large 144420 .exactZero (none)

def event144422 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7208⟩⟩) 0 ⟨7177⟩ 144345

def event144423 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7208⟩⟩) (.authority (.operator))

def exact144424RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩]

theorem exact144424RawTermsValid :
    exact144424RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event144424 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7208⟩⟩) exact144424RawTerms .large 144423 .exactZero (none)

def event144425 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7206⟩⟩) 0 ⟨7177⟩ 144345

def event144426 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7206⟩⟩) (.authority (.operator))

def exact144427RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩]

theorem exact144427RawTermsValid :
    exact144427RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event144427 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7206⟩⟩) exact144427RawTerms .large 144426 .exactZero (none)

def event144428 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7204⟩⟩) 0 ⟨7177⟩ 144345

def event144429 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7204⟩⟩) (.authority (.operator))

def exact144430RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩]

theorem exact144430RawTermsValid :
    exact144430RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event144430 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7204⟩⟩) exact144430RawTerms .large 144429 .exactZero (none)

def event144431 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7202⟩⟩) 0 ⟨7177⟩ 144345

def event144432 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7202⟩⟩) (.authority (.operator))

def exact144433RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩]

theorem exact144433RawTermsValid :
    exact144433RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event144433 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7202⟩⟩) exact144433RawTerms .large 144432 .exactZero (none)

def event144434 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7200⟩⟩) 0 ⟨7177⟩ 144345

def event144435 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7200⟩⟩) (.authority (.operator))

def exact144436RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩]

theorem exact144436RawTermsValid :
    exact144436RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event144436 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7200⟩⟩) exact144436RawTerms .large 144435 .exactZero (none)

def event144437 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7198⟩⟩) 0 ⟨7177⟩ 144345

def event144438 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7198⟩⟩) (.authority (.operator))

def exact144439RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩]

theorem exact144439RawTermsValid :
    exact144439RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event144439 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7198⟩⟩) exact144439RawTerms .large 144438 .exactZero (none)

def event144440 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7309⟩⟩) 0 ⟨7198⟩ 144439

def event144441 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7309⟩⟩) 1 ⟨7200⟩ 144436

def event144442 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7309⟩⟩) (.sum [.predecessor 0 144440 .coefficient, .predecessor 1 144441 .coefficient])

def exact144443RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩]

theorem exact144443RawTermsValid :
    exact144443RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event144443 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7309⟩⟩) exact144443RawTerms .large 144442 .exactZero (none)

def event144444 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7310⟩⟩) 0 ⟨7309⟩ 144443

def event144445 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7310⟩⟩) 1 ⟨7202⟩ 144433

def event144446 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7310⟩⟩) (.sum [.predecessor 0 144444 .coefficient, .predecessor 1 144445 .coefficient])

def exact144447RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩]

theorem exact144447RawTermsValid :
    exact144447RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event144447 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7310⟩⟩) exact144447RawTerms .large 144446 .exactZero (none)

def event144448 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7311⟩⟩) 0 ⟨7310⟩ 144447

def event144449 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7311⟩⟩) 1 ⟨7204⟩ 144430

def event144450 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7311⟩⟩) (.sum [.predecessor 0 144448 .coefficient, .predecessor 1 144449 .coefficient])

def exact144451RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩]

theorem exact144451RawTermsValid :
    exact144451RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event144451 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7311⟩⟩) exact144451RawTerms .large 144450 .exactZero (none)

def event144452 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7312⟩⟩) 0 ⟨7311⟩ 144451

def event144453 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7312⟩⟩) 1 ⟨7206⟩ 144427

def event144454 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7312⟩⟩) (.sum [.predecessor 0 144452 .coefficient, .predecessor 1 144453 .coefficient])

def exact144455RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩]

theorem exact144455RawTermsValid :
    exact144455RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event144455 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7312⟩⟩) exact144455RawTerms .large 144454 .exactZero (none)

def event144456 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7313⟩⟩) 0 ⟨7312⟩ 144455

def event144457 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7313⟩⟩) 1 ⟨7208⟩ 144424

def event144458 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7313⟩⟩) (.sum [.predecessor 0 144456 .coefficient, .predecessor 1 144457 .coefficient])

def exact144459RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩]

theorem exact144459RawTermsValid :
    exact144459RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event144459 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7313⟩⟩) exact144459RawTerms .large 144458 .exactZero (none)

def event144460 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7314⟩⟩) 0 ⟨7313⟩ 144459

def event144461 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7314⟩⟩) 1 ⟨7210⟩ 144421

def event144462 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7314⟩⟩) (.sum [.predecessor 0 144460 .coefficient, .predecessor 1 144461 .coefficient])

def exact144463RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩]

theorem exact144463RawTermsValid :
    exact144463RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event144463 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7314⟩⟩) exact144463RawTerms .large 144462 .exactZero (none)

def event144464 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7315⟩⟩) 0 ⟨7314⟩ 144463

def event144465 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7315⟩⟩) 1 ⟨7212⟩ 144418

def event144466 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7315⟩⟩) (.sum [.predecessor 0 144464 .coefficient, .predecessor 1 144465 .coefficient])

def exact144467RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩]

theorem exact144467RawTermsValid :
    exact144467RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event144467 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7315⟩⟩) exact144467RawTerms .large 144466 .exactZero (none)

def event144468 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7316⟩⟩) 0 ⟨7315⟩ 144467

def event144469 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7316⟩⟩) 1 ⟨7214⟩ 144415

def event144470 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7316⟩⟩) (.sum [.predecessor 0 144468 .coefficient, .predecessor 1 144469 .coefficient])

def exact144471RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩]

theorem exact144471RawTermsValid :
    exact144471RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event144471 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7316⟩⟩) exact144471RawTerms .large 144470 .exactZero (none)

def event144472 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7317⟩⟩) 0 ⟨7316⟩ 144471

def event144473 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7317⟩⟩) 1 ⟨7216⟩ 144412

def event144474 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7317⟩⟩) (.sum [.predecessor 0 144472 .coefficient, .predecessor 1 144473 .coefficient])

def exact144475RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩]

theorem exact144475RawTermsValid :
    exact144475RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event144475 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7317⟩⟩) exact144475RawTerms .large 144474 .exactZero (none)

def event144476 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7318⟩⟩) 0 ⟨7317⟩ 144475

def event144477 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7318⟩⟩) 1 ⟨7218⟩ 144409

def event144478 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7318⟩⟩) (.sum [.predecessor 0 144476 .coefficient, .predecessor 1 144477 .coefficient])

def exact144479RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩]

theorem exact144479RawTermsValid :
    exact144479RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event144479 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7318⟩⟩) exact144479RawTerms .large 144478 .exactZero (none)

def event144480 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7319⟩⟩) 0 ⟨7318⟩ 144479

def event144481 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7319⟩⟩) 1 ⟨7220⟩ 144406

def event144482 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7319⟩⟩) (.sum [.predecessor 0 144480 .coefficient, .predecessor 1 144481 .coefficient])

def exact144483RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩]

theorem exact144483RawTermsValid :
    exact144483RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event144483 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7319⟩⟩) exact144483RawTerms .large 144482 .exactZero (none)

def event144484 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7320⟩⟩) 0 ⟨7319⟩ 144483

def event144485 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7320⟩⟩) 1 ⟨7222⟩ 144403

def event144486 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7320⟩⟩) (.sum [.predecessor 0 144484 .coefficient, .predecessor 1 144485 .coefficient])

def exact144487RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩]

theorem exact144487RawTermsValid :
    exact144487RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event144487 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7320⟩⟩) exact144487RawTerms .large 144486 .exactZero (none)

def event144488 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7321⟩⟩) 0 ⟨7320⟩ 144487

def event144489 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7321⟩⟩) 1 ⟨7224⟩ 144400

def event144490 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7321⟩⟩) (.sum [.predecessor 0 144488 .coefficient, .predecessor 1 144489 .coefficient])

def exact144491RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7224⟩⟩]⟩, (1)⟩]

theorem exact144491RawTermsValid :
    exact144491RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event144491 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7321⟩⟩) exact144491RawTerms .large 144490 .exactZero (none)

def event144492 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7322⟩⟩) 0 ⟨7321⟩ 144491

def event144493 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7322⟩⟩) 1 ⟨7226⟩ 144397

def event144494 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7322⟩⟩) (.sum [.predecessor 0 144492 .coefficient, .predecessor 1 144493 .coefficient])

def exact144495RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7224⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7226⟩⟩]⟩, (1)⟩]

theorem exact144495RawTermsValid :
    exact144495RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event144495 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7322⟩⟩) exact144495RawTerms .large 144494 .exactZero (none)

def event144496 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7323⟩⟩) 0 ⟨7322⟩ 144495

def event144497 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7323⟩⟩) 1 ⟨7228⟩ 144394

def event144498 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7323⟩⟩) (.sum [.predecessor 0 144496 .coefficient, .predecessor 1 144497 .coefficient])

def exact144499RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7224⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7226⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7228⟩⟩]⟩, (1)⟩]

theorem exact144499RawTermsValid :
    exact144499RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event144499 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7323⟩⟩) exact144499RawTerms .large 144498 .exactZero (none)

def event144500 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7324⟩⟩) 0 ⟨7323⟩ 144499

def event144501 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7324⟩⟩) 1 ⟨7230⟩ 144391

def event144502 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7324⟩⟩) (.sum [.predecessor 0 144500 .coefficient, .predecessor 1 144501 .coefficient])

def exact144503RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7224⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7226⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7228⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7230⟩⟩]⟩, (1)⟩]

theorem exact144503RawTermsValid :
    exact144503RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event144503 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7324⟩⟩) exact144503RawTerms .large 144502 .exactZero (none)

def event144504 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7325⟩⟩) 0 ⟨7324⟩ 144503

def event144505 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7325⟩⟩) 1 ⟨7232⟩ 144388

def event144506 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7325⟩⟩) (.sum [.predecessor 0 144504 .coefficient, .predecessor 1 144505 .coefficient])

def exact144507RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7224⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7226⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7228⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7230⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7232⟩⟩]⟩, (1)⟩]

theorem exact144507RawTermsValid :
    exact144507RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event144507 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7325⟩⟩) exact144507RawTerms .large 144506 .exactZero (none)

def event144508 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69062⟩⟩) 0 ⟨7325⟩ 144507

def event144509 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69062⟩⟩) 1 ⟨69061⟩ 144385

def event144510 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69062⟩⟩) (.sum [.predecessor 0 144508 .coefficient, .predecessor 1 144509 .coefficient])

def exact144511RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7224⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7226⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7228⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7230⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7232⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15923⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨18733⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨21953⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨26528⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨29208⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨31973⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨34872⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨37552⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨40228⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨42908⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨45592⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨48272⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨51028⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨54008⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨56988⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨59968⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨62948⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨66111⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact144511RawTermsValid :
    exact144511RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event144511 : Event := .resultExact (⟨.program ⟨257⟩, ⟨69062⟩⟩) exact144511RawTerms .large 144510 .exactZero (none)

def event144512 : Event := .predecessor (⟨.program ⟨257⟩, ⟨71018⟩⟩) 0 ⟨69062⟩ 144511

def event144513 : Event := .predecessor (⟨.program ⟨257⟩, ⟨71018⟩⟩) 1 ⟨71017⟩ 144352

def event144514 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨71018⟩⟩) (.product (.predecessor 0 144512 .coefficient) (.predecessor 1 144513 .coefficient) (⟨false, false, none, none, none⟩))

def event144515 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71018⟩⟩, .operator (⟨144511, 17⟩, ⟨144352, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7232⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (1)⟩)

def event144516 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71018⟩⟩, .operator (⟨144511, 16⟩, ⟨144352, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (1)⟩)

def event144517 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71018⟩⟩, .operator (⟨144511, 15⟩, ⟨144352, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (1)⟩)

def event144518 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71018⟩⟩, .operator (⟨144511, 14⟩, ⟨144352, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (1)⟩)

def event144519 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71018⟩⟩, .operator (⟨144511, 13⟩, ⟨144352, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (1)⟩)

def event144520 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71018⟩⟩, .operator (⟨144511, 12⟩, ⟨144352, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7222⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (1)⟩)

def event144521 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71018⟩⟩, .operator (⟨144511, 11⟩, ⟨144352, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7220⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (1)⟩)

def event144522 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71018⟩⟩, .operator (⟨144511, 10⟩, ⟨144352, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7218⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (1)⟩)

def event144523 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71018⟩⟩, .operator (⟨144511, 9⟩, ⟨144352, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7216⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (1)⟩)

def event144524 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71018⟩⟩, .operator (⟨144511, 8⟩, ⟨144352, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7214⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (1)⟩)

def event144525 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71018⟩⟩, .operator (⟨144511, 7⟩, ⟨144352, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (1)⟩)

def event144526 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71018⟩⟩, .operator (⟨144511, 6⟩, ⟨144352, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (1)⟩)

def event144527 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71018⟩⟩, .operator (⟨144511, 5⟩, ⟨144352, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (1)⟩)

def event144528 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71018⟩⟩, .operator (⟨144511, 4⟩, ⟨144352, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (1)⟩)

def event144529 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71018⟩⟩, .operator (⟨144511, 3⟩, ⟨144352, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (1)⟩)

def event144530 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71018⟩⟩, .operator (⟨144511, 2⟩, ⟨144352, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (1)⟩)

def event144531 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71018⟩⟩, .operator (⟨144511, 1⟩, ⟨144352, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (1)⟩)

def event144532 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71018⟩⟩, .operator (⟨144511, 0⟩, ⟨144352, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (1)⟩)

def event144533 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71018⟩⟩, .operator (⟨144511, 29⟩, ⟨144352, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨48272⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (-1)⟩)

def event144534 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71018⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨48272⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71017⟩⟩) ⟨68788⟩ 144349)

def event144535 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71018⟩⟩, .relation 144534 0, ⟨[⟨.program ⟨257⟩, ⟨48272⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩, (-1)⟩)

def event144536 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71018⟩⟩, .operator (⟨144511, 28⟩, ⟨144352, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨45592⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (-1)⟩)

def event144537 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71018⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨45592⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71017⟩⟩) ⟨68788⟩ 144349)

def event144538 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71018⟩⟩, .relation 144537 0, ⟨[⟨.program ⟨257⟩, ⟨45592⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩, (-1)⟩)

def event144539 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71018⟩⟩, .operator (⟨144511, 27⟩, ⟨144352, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨42908⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (-1)⟩)

def event144540 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71018⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨42908⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71017⟩⟩) ⟨68788⟩ 144349)

def event144541 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71018⟩⟩, .relation 144540 0, ⟨[⟨.program ⟨257⟩, ⟨42908⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩, (-1)⟩)

def event144542 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71018⟩⟩, .operator (⟨144511, 26⟩, ⟨144352, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨40228⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (-1)⟩)

def event144543 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71018⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨40228⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71017⟩⟩) ⟨68788⟩ 144349)

def event144544 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71018⟩⟩, .relation 144543 0, ⟨[⟨.program ⟨257⟩, ⟨40228⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩, (-1)⟩)

def event144545 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71018⟩⟩, .operator (⟨144511, 25⟩, ⟨144352, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨37552⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (-1)⟩)

def event144546 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71018⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨37552⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71017⟩⟩) ⟨68788⟩ 144349)

def event144547 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71018⟩⟩, .relation 144546 0, ⟨[⟨.program ⟨257⟩, ⟨37552⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩, (-1)⟩)

def event144548 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71018⟩⟩, .operator (⟨144511, 24⟩, ⟨144352, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨34872⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (-1)⟩)

def event144549 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71018⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨34872⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71017⟩⟩) ⟨68788⟩ 144349)

def event144550 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71018⟩⟩, .relation 144549 0, ⟨[⟨.program ⟨257⟩, ⟨34872⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩, (-1)⟩)

def event144551 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71018⟩⟩, .operator (⟨144511, 22⟩, ⟨144352, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨29208⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (-1)⟩)

def event144552 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71018⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨29208⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71017⟩⟩) ⟨68788⟩ 144349)

def event144553 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71018⟩⟩, .relation 144552 0, ⟨[⟨.program ⟨257⟩, ⟨29208⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩, (-1)⟩)

def event144554 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71018⟩⟩, .operator (⟨144511, 21⟩, ⟨144352, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨26528⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (-1)⟩)

def event144555 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71018⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨26528⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71017⟩⟩) ⟨68788⟩ 144349)

def event144556 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71018⟩⟩, .relation 144555 0, ⟨[⟨.program ⟨257⟩, ⟨26528⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩, (-1)⟩)

def event144557 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71018⟩⟩, .operator (⟨144511, 35⟩, ⟨144352, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨66111⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (-1)⟩)

def event144558 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71018⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨66111⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71017⟩⟩) ⟨68788⟩ 144349)

def event144559 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71018⟩⟩, .relation 144558 0, ⟨[⟨.program ⟨257⟩, ⟨66111⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩, (-1)⟩)

def event144560 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71018⟩⟩, .operator (⟨144511, 34⟩, ⟨144352, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨62948⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (-1)⟩)

def event144561 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71018⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨62948⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71017⟩⟩) ⟨68788⟩ 144349)

def event144562 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71018⟩⟩, .relation 144561 0, ⟨[⟨.program ⟨257⟩, ⟨62948⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩, (-1)⟩)

def event144563 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71018⟩⟩, .operator (⟨144511, 33⟩, ⟨144352, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨59968⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (-1)⟩)

def event144564 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71018⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨59968⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71017⟩⟩) ⟨68788⟩ 144349)

def event144565 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71018⟩⟩, .relation 144564 0, ⟨[⟨.program ⟨257⟩, ⟨59968⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩, (-1)⟩)

def event144566 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71018⟩⟩, .operator (⟨144511, 32⟩, ⟨144352, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨56988⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (-1)⟩)

def event144567 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71018⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨56988⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71017⟩⟩) ⟨68788⟩ 144349)

def event144568 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71018⟩⟩, .relation 144567 0, ⟨[⟨.program ⟨257⟩, ⟨56988⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩, (-1)⟩)

def event144569 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71018⟩⟩, .operator (⟨144511, 31⟩, ⟨144352, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨54008⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (-1)⟩)

def event144570 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71018⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨54008⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71017⟩⟩) ⟨68788⟩ 144349)

def event144571 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71018⟩⟩, .relation 144570 0, ⟨[⟨.program ⟨257⟩, ⟨54008⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩, (-1)⟩)

def event144572 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71018⟩⟩, .operator (⟨144511, 30⟩, ⟨144352, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨51028⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (-1)⟩)

def event144573 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71018⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨51028⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71017⟩⟩) ⟨68788⟩ 144349)

def event144574 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71018⟩⟩, .relation 144573 0, ⟨[⟨.program ⟨257⟩, ⟨51028⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩, (-1)⟩)

def event144575 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71018⟩⟩, .operator (⟨144511, 23⟩, ⟨144352, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨31973⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (-1)⟩)

def event144576 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71018⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨31973⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71017⟩⟩) ⟨68788⟩ 144349)

def event144577 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71018⟩⟩, .relation 144576 0, ⟨[⟨.program ⟨257⟩, ⟨31973⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩, (-1)⟩)

def event144578 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71018⟩⟩, .operator (⟨144511, 20⟩, ⟨144352, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨21953⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (-1)⟩)

def event144579 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71018⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨21953⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71017⟩⟩) ⟨68788⟩ 144349)

def event144580 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71018⟩⟩, .relation 144579 0, ⟨[⟨.program ⟨257⟩, ⟨21953⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩, (-1)⟩)

def event144581 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71018⟩⟩, .operator (⟨144511, 19⟩, ⟨144352, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨18733⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (-1)⟩)

def event144582 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71018⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨18733⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71017⟩⟩) ⟨68788⟩ 144349)

def event144583 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71018⟩⟩, .relation 144582 0, ⟨[⟨.program ⟨257⟩, ⟨18733⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩, (-1)⟩)

def event144584 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71018⟩⟩, .operator (⟨144511, 18⟩, ⟨144352, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨15923⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (-1)⟩)

def event144585 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71018⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨15923⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71017⟩⟩) ⟨68788⟩ 144349)

def event144586 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71018⟩⟩, .relation 144585 0, ⟨[⟨.program ⟨257⟩, ⟨15923⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩, (-1)⟩)

def exact144587RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7214⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7216⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7218⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7220⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7222⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7232⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15923⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨18733⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨21953⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨26528⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨29208⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨31973⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨34872⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨37552⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨40228⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨42908⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨45592⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨48272⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨51028⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨54008⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨56988⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨59968⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨62948⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨66111⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩, (-1)⟩]

theorem exact144587RawTermsValid :
    exact144587RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event144587 : Event := .resultExact (⟨.program ⟨257⟩, ⟨71018⟩⟩) exact144587RawTerms .large 144514 .exactZero (none)

def event144588 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67322⟩⟩) 0 ⟨66121⟩ 144341

def event144589 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67322⟩⟩) (.authority (.programFamilyFact))

def exact144590RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨67322⟩⟩], []⟩, (1)⟩]

theorem exact144590RawTermsValid :
    exact144590RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event144590 : Event := .resultExact (⟨.program ⟨257⟩, ⟨67322⟩⟩) exact144590RawTerms (.finite 18) 144589 .exactZero (none)

def event144591 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67324⟩⟩) 0 ⟨6908⟩ 144363

def event144592 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67324⟩⟩) 1 ⟨67322⟩ 144590

def event144593 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67324⟩⟩) (.product (.predecessor 0 144591 .coefficient) (.predecessor 1 144592 .coefficient) (⟨false, true, none, none, some 1⟩))

def event144594 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67324⟩⟩, .operator (⟨144363, 0⟩, ⟨144590, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨67322⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact144595RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨67322⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact144595RawTermsValid :
    exact144595RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event144595 : Event := .resultExact (⟨.program ⟨257⟩, ⟨67324⟩⟩) exact144595RawTerms .large 144593 .exactZero (none)

def event144596 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7233⟩⟩) 0 ⟨7177⟩ 144345

def event144597 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7233⟩⟩) (.authority (.operator))

def exact144598RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7233⟩⟩]⟩, (1)⟩]

theorem exact144598RawTermsValid :
    exact144598RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event144598 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7233⟩⟩) exact144598RawTerms .large 144597 .exactZero (none)

def event144599 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67328⟩⟩) 0 ⟨7233⟩ 144598

def event144600 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67328⟩⟩) 1 ⟨67324⟩ 144595

def event144601 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67328⟩⟩) (.sum [.predecessor 0 144599 .coefficient, .predecessor 1 144600 .coefficient])

def exact144602RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7233⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨67322⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact144602RawTermsValid :
    exact144602RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event144602 : Event := .resultExact (⟨.program ⟨257⟩, ⟨67328⟩⟩) exact144602RawTerms .large 144601 .exactZero (none)

def event144603 : Event := .predecessor (⟨.program ⟨257⟩, ⟨71022⟩⟩) 0 ⟨67328⟩ 144602

def event144604 : Event := .predecessor (⟨.program ⟨257⟩, ⟨71022⟩⟩) 1 ⟨71018⟩ 144587

def event144605 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨71022⟩⟩) (.sum [.predecessor 0 144603 .coefficient, .predecessor 1 144604 .coefficient])

def exact144606RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7214⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7216⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7218⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7220⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7222⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7232⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7233⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15923⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨18733⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨21953⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨26528⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨29208⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨31973⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨34872⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨37552⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨40228⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨42908⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨45592⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨48272⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨51028⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨54008⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨56988⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨59968⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨62948⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨66111⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨67322⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact144606RawTermsValid :
    exact144606RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event144606 : Event := .resultExact (⟨.program ⟨257⟩, ⟨71022⟩⟩) exact144606RawTerms .large 144605 .exactZero (none)

def event144607 : Event := .preFoldPolynomial 144606 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7214⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7216⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7218⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7220⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7222⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7232⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7233⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15923⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨18733⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨21953⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨26528⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨29208⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨31973⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨34872⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨37552⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨40228⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨42908⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨45592⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨48272⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨51028⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨54008⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨56988⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨59968⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨62948⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨66111⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨67322⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact144608RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7214⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7216⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7218⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7220⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7222⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7232⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7233⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15923⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨18733⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨21953⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨26528⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨29208⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨31973⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨34872⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨37552⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨40228⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨42908⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨45592⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨48272⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨51028⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨54008⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨56988⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨59968⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨62948⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨66111⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨67322⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event144608 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨71022⟩⟩) 144607 exact144608RawTerms .large 144605 .exactZero (none)

def event144609 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨66121⟩⟩) ⟨⟨1⟩, ⟨95⟩, ⟨135⟩⟩ ⟨143247, 144609⟩

def event144610 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨68303⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68300⟩⟩]⟩) (1) 0 2 (.universal 144609 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68300⟩⟩]⟩) (none) 144608)

def event144611 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68303⟩⟩, .relation 144610 18, ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7233⟩⟩]⟩, (1)⟩)

def event144612 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68303⟩⟩, .relation 144610 17, ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7232⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (-1)⟩)

def event144613 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68303⟩⟩, .relation 144610 16, ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (-1)⟩)

def event144614 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68303⟩⟩, .relation 144610 15, ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (-1)⟩)

def event144615 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68303⟩⟩, .relation 144610 14, ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (-1)⟩)

def event144616 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68303⟩⟩, .relation 144610 13, ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (-1)⟩)

def event144617 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68303⟩⟩, .relation 144610 12, ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7222⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (-1)⟩)

def event144618 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68303⟩⟩, .relation 144610 11, ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (-1)⟩)

def event144619 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68303⟩⟩, .relation 144610 10, ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (-1)⟩)

def event144620 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68303⟩⟩, .relation 144610 9, ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (-1)⟩)

def event144621 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68303⟩⟩, .relation 144610 8, ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (-1)⟩)

def event144622 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68303⟩⟩, .relation 144610 7, ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (-1)⟩)

def event144623 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68303⟩⟩, .relation 144610 6, ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (-1)⟩)

def event144624 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68303⟩⟩, .relation 144610 5, ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (-1)⟩)

def event144625 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68303⟩⟩, .relation 144610 4, ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (-1)⟩)

def event144626 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68303⟩⟩, .relation 144610 3, ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (-1)⟩)

def event144627 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68303⟩⟩, .relation 144610 2, ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (-1)⟩)

def event144628 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68303⟩⟩, .relation 144610 1, ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (-1)⟩)

def event144629 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68303⟩⟩, .relation 144610 0, ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩, (-1)⟩)

def event144630 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68303⟩⟩, .relation 144610 30, ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨48272⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩, (1)⟩)

def event144631 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68303⟩⟩, .relation 144610 29, ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨45592⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩, (1)⟩)

def event144632 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68303⟩⟩, .relation 144610 28, ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨42908⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩, (1)⟩)

def event144633 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68303⟩⟩, .relation 144610 27, ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨40228⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩, (1)⟩)

def event144634 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68303⟩⟩, .relation 144610 26, ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨37552⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩, (1)⟩)

def event144635 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68303⟩⟩, .relation 144610 25, ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨34872⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩, (1)⟩)

def event144636 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68303⟩⟩, .relation 144610 23, ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨29208⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩, (1)⟩)

def event144637 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68303⟩⟩, .relation 144610 22, ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨26528⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩, (1)⟩)

def event144638 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68303⟩⟩, .relation 144610 36, ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨66111⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩, (1)⟩)

def event144639 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68303⟩⟩, .relation 144610 35, ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨62948⟩⟩], [⟨.program ⟨257⟩, ⟨68788⟩⟩]⟩, (1)⟩)

def eventLeaf9024 : Array AnnotatedEvent := #[
  { event := event144384
    frameStart := 143836 },
  { event := event144385
    frameStart := 143836 },
  { event := event144386
    frameStart := 143836 },
  { event := event144387
    frameStart := 143836 },
  { event := event144388
    frameStart := 143836 },
  { event := event144389
    frameStart := 143836 },
  { event := event144390
    frameStart := 143836 },
  { event := event144391
    frameStart := 143836 },
  { event := event144392
    frameStart := 143836 },
  { event := event144393
    frameStart := 143836 },
  { event := event144394
    frameStart := 143836 },
  { event := event144395
    frameStart := 143836 },
  { event := event144396
    frameStart := 143836 },
  { event := event144397
    frameStart := 143836 },
  { event := event144398
    frameStart := 143836 },
  { event := event144399
    frameStart := 143836 }
]

def eventLeaf9025 : Array AnnotatedEvent := #[
  { event := event144400
    frameStart := 143836 },
  { event := event144401
    frameStart := 143836 },
  { event := event144402
    frameStart := 143836 },
  { event := event144403
    frameStart := 143836 },
  { event := event144404
    frameStart := 143836 },
  { event := event144405
    frameStart := 143836 },
  { event := event144406
    frameStart := 143836 },
  { event := event144407
    frameStart := 143836 },
  { event := event144408
    frameStart := 143836 },
  { event := event144409
    frameStart := 143836 },
  { event := event144410
    frameStart := 143836 },
  { event := event144411
    frameStart := 143836 },
  { event := event144412
    frameStart := 143836 },
  { event := event144413
    frameStart := 143836 },
  { event := event144414
    frameStart := 143836 },
  { event := event144415
    frameStart := 143836 }
]

def eventLeaf9026 : Array AnnotatedEvent := #[
  { event := event144416
    frameStart := 143836 },
  { event := event144417
    frameStart := 143836 },
  { event := event144418
    frameStart := 143836 },
  { event := event144419
    frameStart := 143836 },
  { event := event144420
    frameStart := 143836 },
  { event := event144421
    frameStart := 143836 },
  { event := event144422
    frameStart := 143836 },
  { event := event144423
    frameStart := 143836 },
  { event := event144424
    frameStart := 143836 },
  { event := event144425
    frameStart := 143836 },
  { event := event144426
    frameStart := 143836 },
  { event := event144427
    frameStart := 143836 },
  { event := event144428
    frameStart := 143836 },
  { event := event144429
    frameStart := 143836 },
  { event := event144430
    frameStart := 143836 },
  { event := event144431
    frameStart := 143836 }
]

def eventLeaf9027 : Array AnnotatedEvent := #[
  { event := event144432
    frameStart := 143836 },
  { event := event144433
    frameStart := 143836 },
  { event := event144434
    frameStart := 143836 },
  { event := event144435
    frameStart := 143836 },
  { event := event144436
    frameStart := 143836 },
  { event := event144437
    frameStart := 143836 },
  { event := event144438
    frameStart := 143836 },
  { event := event144439
    frameStart := 143836 },
  { event := event144440
    frameStart := 143836 },
  { event := event144441
    frameStart := 143836 },
  { event := event144442
    frameStart := 143836 },
  { event := event144443
    frameStart := 143836 },
  { event := event144444
    frameStart := 143836 },
  { event := event144445
    frameStart := 143836 },
  { event := event144446
    frameStart := 143836 },
  { event := event144447
    frameStart := 143836 }
]

def eventLeaf9028 : Array AnnotatedEvent := #[
  { event := event144448
    frameStart := 143836 },
  { event := event144449
    frameStart := 143836 },
  { event := event144450
    frameStart := 143836 },
  { event := event144451
    frameStart := 143836 },
  { event := event144452
    frameStart := 143836 },
  { event := event144453
    frameStart := 143836 },
  { event := event144454
    frameStart := 143836 },
  { event := event144455
    frameStart := 143836 },
  { event := event144456
    frameStart := 143836 },
  { event := event144457
    frameStart := 143836 },
  { event := event144458
    frameStart := 143836 },
  { event := event144459
    frameStart := 143836 },
  { event := event144460
    frameStart := 143836 },
  { event := event144461
    frameStart := 143836 },
  { event := event144462
    frameStart := 143836 },
  { event := event144463
    frameStart := 143836 }
]

def eventLeaf9029 : Array AnnotatedEvent := #[
  { event := event144464
    frameStart := 143836 },
  { event := event144465
    frameStart := 143836 },
  { event := event144466
    frameStart := 143836 },
  { event := event144467
    frameStart := 143836 },
  { event := event144468
    frameStart := 143836 },
  { event := event144469
    frameStart := 143836 },
  { event := event144470
    frameStart := 143836 },
  { event := event144471
    frameStart := 143836 },
  { event := event144472
    frameStart := 143836 },
  { event := event144473
    frameStart := 143836 },
  { event := event144474
    frameStart := 143836 },
  { event := event144475
    frameStart := 143836 },
  { event := event144476
    frameStart := 143836 },
  { event := event144477
    frameStart := 143836 },
  { event := event144478
    frameStart := 143836 },
  { event := event144479
    frameStart := 143836 }
]

def eventLeaf9030 : Array AnnotatedEvent := #[
  { event := event144480
    frameStart := 143836 },
  { event := event144481
    frameStart := 143836 },
  { event := event144482
    frameStart := 143836 },
  { event := event144483
    frameStart := 143836 },
  { event := event144484
    frameStart := 143836 },
  { event := event144485
    frameStart := 143836 },
  { event := event144486
    frameStart := 143836 },
  { event := event144487
    frameStart := 143836 },
  { event := event144488
    frameStart := 143836 },
  { event := event144489
    frameStart := 143836 },
  { event := event144490
    frameStart := 143836 },
  { event := event144491
    frameStart := 143836 },
  { event := event144492
    frameStart := 143836 },
  { event := event144493
    frameStart := 143836 },
  { event := event144494
    frameStart := 143836 },
  { event := event144495
    frameStart := 143836 }
]

def eventLeaf9031 : Array AnnotatedEvent := #[
  { event := event144496
    frameStart := 143836 },
  { event := event144497
    frameStart := 143836 },
  { event := event144498
    frameStart := 143836 },
  { event := event144499
    frameStart := 143836 },
  { event := event144500
    frameStart := 143836 },
  { event := event144501
    frameStart := 143836 },
  { event := event144502
    frameStart := 143836 },
  { event := event144503
    frameStart := 143836 },
  { event := event144504
    frameStart := 143836 },
  { event := event144505
    frameStart := 143836 },
  { event := event144506
    frameStart := 143836 },
  { event := event144507
    frameStart := 143836 },
  { event := event144508
    frameStart := 143836 },
  { event := event144509
    frameStart := 143836 },
  { event := event144510
    frameStart := 143836 },
  { event := event144511
    frameStart := 143836 }
]

def eventLeaf9032 : Array AnnotatedEvent := #[
  { event := event144512
    frameStart := 143836 },
  { event := event144513
    frameStart := 143836 },
  { event := event144514
    frameStart := 143836 },
  { event := event144515
    frameStart := 143836 },
  { event := event144516
    frameStart := 143836 },
  { event := event144517
    frameStart := 143836 },
  { event := event144518
    frameStart := 143836 },
  { event := event144519
    frameStart := 143836 },
  { event := event144520
    frameStart := 143836 },
  { event := event144521
    frameStart := 143836 },
  { event := event144522
    frameStart := 143836 },
  { event := event144523
    frameStart := 143836 },
  { event := event144524
    frameStart := 143836 },
  { event := event144525
    frameStart := 143836 },
  { event := event144526
    frameStart := 143836 },
  { event := event144527
    frameStart := 143836 }
]

def eventLeaf9033 : Array AnnotatedEvent := #[
  { event := event144528
    frameStart := 143836 },
  { event := event144529
    frameStart := 143836 },
  { event := event144530
    frameStart := 143836 },
  { event := event144531
    frameStart := 143836 },
  { event := event144532
    frameStart := 143836 },
  { event := event144533
    frameStart := 143836 },
  { event := event144534
    frameStart := 143836 },
  { event := event144535
    frameStart := 143836 },
  { event := event144536
    frameStart := 143836 },
  { event := event144537
    frameStart := 143836 },
  { event := event144538
    frameStart := 143836 },
  { event := event144539
    frameStart := 143836 },
  { event := event144540
    frameStart := 143836 },
  { event := event144541
    frameStart := 143836 },
  { event := event144542
    frameStart := 143836 },
  { event := event144543
    frameStart := 143836 }
]

def eventLeaf9034 : Array AnnotatedEvent := #[
  { event := event144544
    frameStart := 143836 },
  { event := event144545
    frameStart := 143836 },
  { event := event144546
    frameStart := 143836 },
  { event := event144547
    frameStart := 143836 },
  { event := event144548
    frameStart := 143836 },
  { event := event144549
    frameStart := 143836 },
  { event := event144550
    frameStart := 143836 },
  { event := event144551
    frameStart := 143836 },
  { event := event144552
    frameStart := 143836 },
  { event := event144553
    frameStart := 143836 },
  { event := event144554
    frameStart := 143836 },
  { event := event144555
    frameStart := 143836 },
  { event := event144556
    frameStart := 143836 },
  { event := event144557
    frameStart := 143836 },
  { event := event144558
    frameStart := 143836 },
  { event := event144559
    frameStart := 143836 }
]

def eventLeaf9035 : Array AnnotatedEvent := #[
  { event := event144560
    frameStart := 143836 },
  { event := event144561
    frameStart := 143836 },
  { event := event144562
    frameStart := 143836 },
  { event := event144563
    frameStart := 143836 },
  { event := event144564
    frameStart := 143836 },
  { event := event144565
    frameStart := 143836 },
  { event := event144566
    frameStart := 143836 },
  { event := event144567
    frameStart := 143836 },
  { event := event144568
    frameStart := 143836 },
  { event := event144569
    frameStart := 143836 },
  { event := event144570
    frameStart := 143836 },
  { event := event144571
    frameStart := 143836 },
  { event := event144572
    frameStart := 143836 },
  { event := event144573
    frameStart := 143836 },
  { event := event144574
    frameStart := 143836 },
  { event := event144575
    frameStart := 143836 }
]

def eventLeaf9036 : Array AnnotatedEvent := #[
  { event := event144576
    frameStart := 143836 },
  { event := event144577
    frameStart := 143836 },
  { event := event144578
    frameStart := 143836 },
  { event := event144579
    frameStart := 143836 },
  { event := event144580
    frameStart := 143836 },
  { event := event144581
    frameStart := 143836 },
  { event := event144582
    frameStart := 143836 },
  { event := event144583
    frameStart := 143836 },
  { event := event144584
    frameStart := 143836 },
  { event := event144585
    frameStart := 143836 },
  { event := event144586
    frameStart := 143836 },
  { event := event144587
    frameStart := 143836 },
  { event := event144588
    frameStart := 143836 },
  { event := event144589
    frameStart := 143836 },
  { event := event144590
    frameStart := 143836 },
  { event := event144591
    frameStart := 143836 }
]

def eventLeaf9037 : Array AnnotatedEvent := #[
  { event := event144592
    frameStart := 143836 },
  { event := event144593
    frameStart := 143836 },
  { event := event144594
    frameStart := 143836 },
  { event := event144595
    frameStart := 143836 },
  { event := event144596
    frameStart := 143836 },
  { event := event144597
    frameStart := 143836 },
  { event := event144598
    frameStart := 143836 },
  { event := event144599
    frameStart := 143836 },
  { event := event144600
    frameStart := 143836 },
  { event := event144601
    frameStart := 143836 },
  { event := event144602
    frameStart := 143836 },
  { event := event144603
    frameStart := 143836 },
  { event := event144604
    frameStart := 143836 },
  { event := event144605
    frameStart := 143836 },
  { event := event144606
    frameStart := 143836 },
  { event := event144607
    frameStart := 143836 }
]

def eventLeaf9038 : Array AnnotatedEvent := #[
  { event := event144608
    frameStart := 143836 },
  { event := event144609
    frameStart := 0 },
  { event := event144610
    frameStart := 0 },
  { event := event144611
    frameStart := 0 },
  { event := event144612
    frameStart := 0 },
  { event := event144613
    frameStart := 0 },
  { event := event144614
    frameStart := 0 },
  { event := event144615
    frameStart := 0 },
  { event := event144616
    frameStart := 0 },
  { event := event144617
    frameStart := 0 },
  { event := event144618
    frameStart := 0 },
  { event := event144619
    frameStart := 0 },
  { event := event144620
    frameStart := 0 },
  { event := event144621
    frameStart := 0 },
  { event := event144622
    frameStart := 0 },
  { event := event144623
    frameStart := 0 }
]

def eventLeaf9039 : Array AnnotatedEvent := #[
  { event := event144624
    frameStart := 0 },
  { event := event144625
    frameStart := 0 },
  { event := event144626
    frameStart := 0 },
  { event := event144627
    frameStart := 0 },
  { event := event144628
    frameStart := 0 },
  { event := event144629
    frameStart := 0 },
  { event := event144630
    frameStart := 0 },
  { event := event144631
    frameStart := 0 },
  { event := event144632
    frameStart := 0 },
  { event := event144633
    frameStart := 0 },
  { event := event144634
    frameStart := 0 },
  { event := event144635
    frameStart := 0 },
  { event := event144636
    frameStart := 0 },
  { event := event144637
    frameStart := 0 },
  { event := event144638
    frameStart := 0 },
  { event := event144639
    frameStart := 0 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events564
