import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events111

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event28416 : Event := .preFoldPolynomial 28415 [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40678⟩⟩]⟩, (1)⟩] .exactZero none

def exact28417RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40678⟩⟩]⟩, (1)⟩]

def event28417 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨40679⟩⟩) 28416 exact28417RawTerms .large 28413 .exactZero (none)

def event28418 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨41770⟩⟩)

def event28419 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event28420 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event28421 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨140⟩⟩) (.authority (.operator))

def event28422 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨140⟩⟩) (.finite 19)

def event28423 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event28424 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event28425 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event28426 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event28427 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 28426

def event28428 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 28424

def event28429 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 28427 .coefficient) (.value (.predecessor 1 28428 .coefficient)))

def event28430 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event28431 : Event := .predecessor (⟨.program ⟨257⟩, ⟨393⟩⟩) 0 ⟨392⟩ 28430

def event28432 : Event := .predecessor (⟨.program ⟨257⟩, ⟨393⟩⟩) 1 ⟨140⟩ 28422

def event28433 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨393⟩⟩) (.sum [.predecessor 0 28431 .coefficient, .predecessor 1 28432 .coefficient])

def event28434 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨393⟩⟩) (.finite 655359)

def event28435 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5439⟩⟩) 0 ⟨393⟩ 28434

def event28436 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5439⟩⟩) 1 ⟨5426⟩ 28420

def event28437 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5439⟩⟩) (.identity (.predecessor 1 28436 .coefficient))

def event28438 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5439⟩⟩) (.finite 655360)

def event28439 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39586⟩⟩) 0 ⟨5439⟩ 28438

def event28440 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39586⟩⟩) (.authority (.programFamilyFact))

def exact28441RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨39586⟩⟩], []⟩, (1)⟩]

theorem exact28441RawTermsValid :
    exact28441RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28441 : Event := .resultExact (⟨.program ⟨257⟩, ⟨39586⟩⟩) exact28441RawTerms (.finite 46) 28440 .exactZero (none)

def event28442 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14051⟩⟩) 0 ⟨5439⟩ 28438

def event28443 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14051⟩⟩) (.authority (.programFamilyFact))

def exact28444RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14051⟩⟩], []⟩, (1)⟩]

theorem exact28444RawTermsValid :
    exact28444RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28444 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14051⟩⟩) exact28444RawTerms (.finite 46) 28443 .exactZero (none)

def event28445 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39587⟩⟩) 0 ⟨14051⟩ 28444

def event28446 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39587⟩⟩) 1 ⟨39586⟩ 28441

def event28447 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39587⟩⟩) (.product (.predecessor 0 28445 .coefficient) (.predecessor 1 28446 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event28448 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨39587⟩⟩, .operator (⟨28444, 0⟩, ⟨28441, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨14051⟩⟩, ⟨.program ⟨257⟩, ⟨39586⟩⟩], []⟩, (1)⟩)

def exact28449RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14051⟩⟩, ⟨.program ⟨257⟩, ⟨39586⟩⟩], []⟩, (1)⟩]

theorem exact28449RawTermsValid :
    exact28449RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28449 : Event := .resultExact (⟨.program ⟨257⟩, ⟨39587⟩⟩) exact28449RawTerms (.finite 2116) 28447 .exactZero (none)

def event28450 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39588⟩⟩) 0 ⟨39587⟩ 28449

def event28451 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39588⟩⟩) (.identity (.predecessor 0 28450 .coefficient))

def event28452 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨39588⟩⟩) (.finite 2116)

def event28453 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40038⟩⟩) 0 ⟨39588⟩ 28452

def event28454 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40038⟩⟩) (.authority (.programFamilyFact))

def exact28455RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨40038⟩⟩], []⟩, (1)⟩]

theorem exact28455RawTermsValid :
    exact28455RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28455 : Event := .resultExact (⟨.program ⟨257⟩, ⟨40038⟩⟩) exact28455RawTerms (.finite 46) 28454 .exactZero (none)

def event28456 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40039⟩⟩) 0 ⟨40038⟩ 28455

def event28457 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40039⟩⟩) (.identity (.predecessor 0 28456 .coefficient))

def event28458 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨40039⟩⟩) (.finite 46)

def event28459 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41181⟩⟩) 0 ⟨40039⟩ 28458

def event28460 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41181⟩⟩) (.authority (.programFamilyFact))

def event28461 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨41181⟩⟩) (.finite 3720)

def event28462 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨7177⟩⟩) .missing

def event28463 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41182⟩⟩) 0 ⟨7177⟩ 28462

def event28464 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41182⟩⟩) 1 ⟨41181⟩ 28461

def event28465 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41182⟩⟩) (.authority (.operator))

def exact28466RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨41182⟩⟩]⟩, (1)⟩]

theorem exact28466RawTermsValid :
    exact28466RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28466 : Event := .resultExact (⟨.program ⟨257⟩, ⟨41182⟩⟩) exact28466RawTerms .large 28465 .exactZero (none)

def event28467 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41765⟩⟩) 0 ⟨41182⟩ 28466

def event28468 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41765⟩⟩) (.authority (.operator))

def exact28469RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨41765⟩⟩]⟩, (1)⟩]

theorem exact28469RawTermsValid :
    exact28469RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28469 : Event := .resultExact (⟨.program ⟨257⟩, ⟨41765⟩⟩) exact28469RawTerms (.finite 8192) 28468 .exactZero (none)

def event28470 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event28471 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event28472 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41430⟩⟩) 0 ⟨40039⟩ 28458

def event28473 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41430⟩⟩) 1 ⟨136⟩ 28471

def event28474 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41430⟩⟩) (.sum [.predecessor 0 28472 .coefficient, .predecessor 1 28473 .coefficient])

def event28475 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨41430⟩⟩) (.finite 46)

def event28476 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41431⟩⟩) 0 ⟨41430⟩ 28475

def event28477 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41431⟩⟩) (.identity (.predecessor 0 28476 .coefficient))

def exact28478RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨40038⟩⟩], []⟩, (1)⟩]

theorem exact28478RawTermsValid :
    exact28478RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28478 : Event := .resultExact (⟨.program ⟨257⟩, ⟨41431⟩⟩) exact28478RawTerms (.finite 46) 28477 .exactZero (none)

def event28479 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact28480RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact28480RawTermsValid :
    exact28480RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28480 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact28480RawTerms .large 28479 .exactZero (none)

def event28481 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41432⟩⟩) 0 ⟨6908⟩ 28480

def event28482 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41432⟩⟩) 1 ⟨41431⟩ 28478

def event28483 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41432⟩⟩) (.product (.predecessor 0 28481 .coefficient) (.predecessor 1 28482 .coefficient) (⟨false, false, none, none, none⟩))

def event28484 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨41432⟩⟩, .operator (⟨28480, 0⟩, ⟨28478, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨40038⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact28485RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨40038⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact28485RawTermsValid :
    exact28485RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28485 : Event := .resultExact (⟨.program ⟨257⟩, ⟨41432⟩⟩) exact28485RawTerms .large 28483 .exactZero (none)

def event28486 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7193⟩⟩) 0 ⟨7177⟩ 28462

def event28487 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7193⟩⟩) (.authority (.operator))

def exact28488RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7193⟩⟩]⟩, (1)⟩]

theorem exact28488RawTermsValid :
    exact28488RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28488 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7193⟩⟩) exact28488RawTerms .large 28487 .exactZero (none)

def event28489 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41433⟩⟩) 0 ⟨7193⟩ 28488

def event28490 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41433⟩⟩) 1 ⟨41432⟩ 28485

def event28491 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41433⟩⟩) (.sum [.predecessor 0 28489 .coefficient, .predecessor 1 28490 .coefficient])

def exact28492RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7193⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨40038⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact28492RawTermsValid :
    exact28492RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28492 : Event := .resultExact (⟨.program ⟨257⟩, ⟨41433⟩⟩) exact28492RawTerms .large 28491 .exactZero (none)

def event28493 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41766⟩⟩) 0 ⟨41433⟩ 28492

def event28494 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41766⟩⟩) 1 ⟨41765⟩ 28469

def event28495 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41766⟩⟩) (.product (.predecessor 0 28493 .coefficient) (.predecessor 1 28494 .coefficient) (⟨false, false, none, none, none⟩))

def event28496 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨41766⟩⟩, .operator (⟨28492, 1⟩, ⟨28469, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨40038⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨41765⟩⟩]⟩, (-1)⟩)

def event28497 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨41766⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨40038⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨41765⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨41765⟩⟩) ⟨41182⟩ 28466)

def event28498 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨41766⟩⟩, .relation 28497 0, ⟨[⟨.program ⟨257⟩, ⟨40038⟩⟩], [⟨.program ⟨257⟩, ⟨41182⟩⟩]⟩, (-1)⟩)

def event28499 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨41766⟩⟩, .operator (⟨28492, 0⟩, ⟨28469, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨41765⟩⟩]⟩, (1)⟩)

def exact28500RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨41765⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨40038⟩⟩], [⟨.program ⟨257⟩, ⟨41182⟩⟩]⟩, (-1)⟩]

theorem exact28500RawTermsValid :
    exact28500RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28500 : Event := .resultExact (⟨.program ⟨257⟩, ⟨41766⟩⟩) exact28500RawTerms .large 28495 .exactZero (none)

def event28501 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40208⟩⟩) 0 ⟨40039⟩ 28458

def event28502 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40208⟩⟩) (.authority (.programFamilyFact))

def exact28503RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨40208⟩⟩], []⟩, (1)⟩]

theorem exact28503RawTermsValid :
    exact28503RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28503 : Event := .resultExact (⟨.program ⟨257⟩, ⟨40208⟩⟩) exact28503RawTerms (.finite 46) 28502 .exactZero (none)

def event28504 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40210⟩⟩) 0 ⟨6908⟩ 28480

def event28505 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40210⟩⟩) 1 ⟨40208⟩ 28503

def event28506 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40210⟩⟩) (.product (.predecessor 0 28504 .coefficient) (.predecessor 1 28505 .coefficient) (⟨false, true, none, none, some 1⟩))

def event28507 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨40210⟩⟩, .operator (⟨28480, 0⟩, ⟨28503, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨40208⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact28508RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨40208⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact28508RawTermsValid :
    exact28508RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28508 : Event := .resultExact (⟨.program ⟨257⟩, ⟨40210⟩⟩) exact28508RawTerms .large 28506 .exactZero (none)

def event28509 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7225⟩⟩) 0 ⟨7177⟩ 28462

def event28510 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7225⟩⟩) (.authority (.operator))

def exact28511RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7225⟩⟩]⟩, (1)⟩]

theorem exact28511RawTermsValid :
    exact28511RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28511 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7225⟩⟩) exact28511RawTerms .large 28510 .exactZero (none)

def event28512 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40211⟩⟩) 0 ⟨7225⟩ 28511

def event28513 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40211⟩⟩) 1 ⟨40210⟩ 28508

def event28514 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40211⟩⟩) (.sum [.predecessor 0 28512 .coefficient, .predecessor 1 28513 .coefficient])

def exact28515RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7225⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨40208⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact28515RawTermsValid :
    exact28515RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28515 : Event := .resultExact (⟨.program ⟨257⟩, ⟨40211⟩⟩) exact28515RawTerms .large 28514 .exactZero (none)

def event28516 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41770⟩⟩) 0 ⟨40211⟩ 28515

def event28517 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41770⟩⟩) 1 ⟨41766⟩ 28500

def event28518 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41770⟩⟩) (.sum [.predecessor 0 28516 .coefficient, .predecessor 1 28517 .coefficient])

def exact28519RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨41765⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7225⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨40038⟩⟩], [⟨.program ⟨257⟩, ⟨41182⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨40208⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact28519RawTermsValid :
    exact28519RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28519 : Event := .resultExact (⟨.program ⟨257⟩, ⟨41770⟩⟩) exact28519RawTerms .large 28518 .exactZero (none)

def event28520 : Event := .preFoldPolynomial 28519 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨41765⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7225⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨40038⟩⟩], [⟨.program ⟨257⟩, ⟨41182⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨40208⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact28521RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨41765⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7225⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨40038⟩⟩], [⟨.program ⟨257⟩, ⟨41182⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨40208⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event28521 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨41770⟩⟩) 28520 exact28521RawTerms .large 28518 .exactZero (none)

def event28522 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨40039⟩⟩) ⟨⟨104⟩, ⟨86⟩, ⟨135⟩⟩ ⟨28364, 28522⟩

def event28523 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨40681⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40678⟩⟩]⟩) (1) 0 2 (.universal 28522 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40678⟩⟩]⟩) (none) 28521)

def event28524 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨40681⟩⟩, .relation 28523 1, ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7225⟩⟩]⟩, (1)⟩)

def event28525 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨40681⟩⟩, .relation 28523 2, ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨40038⟩⟩], [⟨.program ⟨257⟩, ⟨41182⟩⟩]⟩, (1)⟩)

def event28526 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨40681⟩⟩, .relation 28523 0, ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨41765⟩⟩]⟩, (-1)⟩)

def event28527 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨40681⟩⟩, .relation 28523 3, ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨40208⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact28528RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨41765⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7225⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨40038⟩⟩], [⟨.program ⟨257⟩, ⟨41182⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨40208⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact28528RawTermsValid :
    exact28528RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28528 : Event := .resultExact (⟨.program ⟨257⟩, ⟨40681⟩⟩) exact28528RawTerms .large 28360 (.finite 202072841853861888) (some (28362))

def event28529 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41768⟩⟩) 0 ⟨40681⟩ 28528

def event28530 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41768⟩⟩) 1 ⟨41767⟩ 28350

def event28531 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41768⟩⟩) (.sum [.predecessor 0 28529 .coefficient, .predecessor 1 28530 .coefficient])

def event28532 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨41768⟩⟩, .operator (⟨28528, 2⟩, ⟨28350, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨40038⟩⟩], [⟨.program ⟨257⟩, ⟨41182⟩⟩]⟩, (-1)⟩)

def event28533 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨41768⟩⟩, .operator (⟨28528, 0⟩, ⟨28350, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨41765⟩⟩]⟩, (1)⟩)

def event28534 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41768⟩⟩) (.sum [.result 28528 .summary, .result 28350 .summary])

def exact28535RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7225⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨40208⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact28535RawTermsValid :
    exact28535RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28535 : Event := .resultExact (⟨.program ⟨257⟩, ⟨41768⟩⟩) exact28535RawTerms .large 28531 (.finite 32193129122288829188810200055808) (some (28534))

def event28536 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41769⟩⟩) 0 ⟨41768⟩ 28535

def event28537 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41769⟩⟩) 1 ⟨7160⟩ 15602

def event28538 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41769⟩⟩) (.product (.predecessor 0 28536 .coefficient) (.predecessor 1 28537 .coefficient) (⟨false, false, none, none, none⟩))

def event28539 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41769⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨7159⟩⟩]⟩) [⟨.result 15598 .coefficient, false, none⟩])

def event28540 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41769⟩⟩) (.product (.result 28535 .summary) (.transfer 28539) (⟨false, false, none, none, none⟩))

def event28541 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨41769⟩⟩, .operator (⟨28535, 0⟩, ⟨15602, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7225⟩⟩, ⟨.program ⟨257⟩, ⟨7159⟩⟩]⟩, (1)⟩)

def event28542 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨41769⟩⟩, .operator (⟨28535, 1⟩, ⟨15602, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨40208⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7159⟩⟩]⟩, (-1)⟩)

def event28543 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨41769⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨40208⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7159⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨7159⟩⟩) ⟨7045⟩ 15595)

def event28544 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨41769⟩⟩, .relation 28543 0, ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40208⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact28545RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7225⟩⟩, ⟨.program ⟨257⟩, ⟨7159⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40208⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact28545RawTermsValid :
    exact28545RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28545 : Event := .resultExact (⟨.program ⟨257⟩, ⟨41769⟩⟩) exact28545RawTerms .large 28538 (.finite 345671091840339265080175045977281837137920) (some (28540))

def event28546 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38502⟩⟩) 0 ⟨7177⟩ 15500

def event28547 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38502⟩⟩) 1 ⟨38501⟩ 19056

def event28548 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨38502⟩⟩) (.authority (.operator))

def exact28549RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨38502⟩⟩]⟩, (1)⟩]

theorem exact28549RawTermsValid :
    exact28549RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28549 : Event := .resultExact (⟨.program ⟨257⟩, ⟨38502⟩⟩) exact28549RawTerms .large 28548 .exactZero (none)

def event28550 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39085⟩⟩) 0 ⟨38502⟩ 28549

def event28551 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39085⟩⟩) (.authority (.operator))

def exact28552RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨39085⟩⟩]⟩, (1)⟩]

theorem exact28552RawTermsValid :
    exact28552RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28552 : Event := .resultExact (⟨.program ⟨257⟩, ⟨39085⟩⟩) exact28552RawTerms (.finite 8192) 28551 .exactZero (none)

def event28553 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39087⟩⟩) 0 ⟨38845⟩ 19359

def event28554 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39087⟩⟩) 1 ⟨39085⟩ 28552

def event28555 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39087⟩⟩) (.product (.predecessor 0 28553 .coefficient) (.predecessor 1 28554 .coefficient) (⟨false, false, none, none, none⟩))

def event28556 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39087⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨39085⟩⟩]⟩) [⟨.result 28552 .coefficient, false, none⟩])

def event28557 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39087⟩⟩) (.product (.result 19359 .summary) (.transfer 28556) (⟨false, false, none, none, none⟩))

def event28558 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨39087⟩⟩, .operator (⟨19359, 1⟩, ⟨28552, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨37358⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨39085⟩⟩]⟩, (-1)⟩)

def event28559 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨39087⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨37358⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨39085⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨39085⟩⟩) ⟨38502⟩ 28549)

def event28560 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨39087⟩⟩, .relation 28559 0, ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨37358⟩⟩], [⟨.program ⟨257⟩, ⟨38502⟩⟩]⟩, (-1)⟩)

def event28561 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨39087⟩⟩, .operator (⟨19359, 0⟩, ⟨28552, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7192⟩⟩, ⟨.program ⟨257⟩, ⟨39085⟩⟩]⟩, (1)⟩)

def exact28562RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7192⟩⟩, ⟨.program ⟨257⟩, ⟨39085⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨37358⟩⟩], [⟨.program ⟨257⟩, ⟨38502⟩⟩]⟩, (-1)⟩]

theorem exact28562RawTermsValid :
    exact28562RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28562 : Event := .resultExact (⟨.program ⟨257⟩, ⟨39087⟩⟩) exact28562RawTerms .large 28555 (.finite 32192736221397252361486566686720) (some (28557))

def event28563 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37998⟩⟩) 0 ⟨37359⟩ 160

def event28564 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37998⟩⟩) (.authority (.relationPreimageSource ⟨84⟩))

def exact28565RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨37998⟩⟩]⟩, (1)⟩]

theorem exact28565RawTermsValid :
    exact28565RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28565 : Event := .resultExact (⟨.program ⟨257⟩, ⟨37998⟩⟩) exact28565RawTerms (.finite 5647228698) 28564 .exactZero (none)

def event28566 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38000⟩⟩) 0 ⟨37998⟩ 28565

def event28567 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38000⟩⟩) 1 ⟨2370⟩ 4

def event28568 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨38000⟩⟩) (.scale (.predecessor 0 28566 .coefficient) (.value (.predecessor 1 28567 .coefficient)))

def exact28569RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨37998⟩⟩]⟩, (1)⟩]

theorem exact28569RawTermsValid :
    exact28569RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28569 : Event := .resultExact (⟨.program ⟨257⟩, ⟨38000⟩⟩) exact28569RawTerms (.finite 5647228698) 28568 .exactZero (none)

def event28570 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38001⟩⟩) 0 ⟨5443⟩ 17169

def event28571 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38001⟩⟩) 1 ⟨38000⟩ 28569

def event28572 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨38001⟩⟩) (.product (.predecessor 0 28570 .coefficient) (.predecessor 1 28571 .coefficient) (⟨false, false, none, none, none⟩))

def event28573 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨38001⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨37998⟩⟩]⟩) [⟨.result 28565 .coefficient, false, none⟩])

def event28574 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨38001⟩⟩) (.product (.result 17169 .summary) (.transfer 28573) (⟨false, false, none, none, none⟩))

def event28575 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨38001⟩⟩, .operator (⟨17169, 0⟩, ⟨28569, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37998⟩⟩]⟩, (1)⟩)

def event28576 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨37999⟩⟩)

def event28577 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event28578 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event28579 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨140⟩⟩) (.authority (.operator))

def event28580 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨140⟩⟩) (.finite 19)

def event28581 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event28582 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event28583 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event28584 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event28585 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 28584

def event28586 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 28582

def event28587 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 28585 .coefficient) (.value (.predecessor 1 28586 .coefficient)))

def event28588 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event28589 : Event := .predecessor (⟨.program ⟨257⟩, ⟨393⟩⟩) 0 ⟨392⟩ 28588

def event28590 : Event := .predecessor (⟨.program ⟨257⟩, ⟨393⟩⟩) 1 ⟨140⟩ 28580

def event28591 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨393⟩⟩) (.sum [.predecessor 0 28589 .coefficient, .predecessor 1 28590 .coefficient])

def event28592 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨393⟩⟩) (.finite 655359)

def event28593 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5439⟩⟩) 0 ⟨393⟩ 28592

def event28594 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5439⟩⟩) 1 ⟨5426⟩ 28578

def event28595 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5439⟩⟩) (.identity (.predecessor 1 28594 .coefficient))

def event28596 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5439⟩⟩) (.finite 655360)

def event28597 : Event := .predecessor (⟨.program ⟨257⟩, ⟨36906⟩⟩) 0 ⟨5439⟩ 28596

def event28598 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨36906⟩⟩) (.authority (.programFamilyFact))

def exact28599RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨36906⟩⟩], []⟩, (1)⟩]

theorem exact28599RawTermsValid :
    exact28599RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28599 : Event := .resultExact (⟨.program ⟨257⟩, ⟨36906⟩⟩) exact28599RawTerms (.finite 42) 28598 .exactZero (none)

def event28600 : Event := .predecessor (⟨.program ⟨257⟩, ⟨13751⟩⟩) 0 ⟨5439⟩ 28596

def event28601 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨13751⟩⟩) (.authority (.programFamilyFact))

def exact28602RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨13751⟩⟩], []⟩, (1)⟩]

theorem exact28602RawTermsValid :
    exact28602RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28602 : Event := .resultExact (⟨.program ⟨257⟩, ⟨13751⟩⟩) exact28602RawTerms (.finite 42) 28601 .exactZero (none)

def event28603 : Event := .predecessor (⟨.program ⟨257⟩, ⟨36907⟩⟩) 0 ⟨13751⟩ 28602

def event28604 : Event := .predecessor (⟨.program ⟨257⟩, ⟨36907⟩⟩) 1 ⟨36906⟩ 28599

def event28605 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨36907⟩⟩) (.product (.predecessor 0 28603 .coefficient) (.predecessor 1 28604 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event28606 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨36907⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨13751⟩⟩, ⟨.program ⟨257⟩, ⟨36906⟩⟩], []⟩) [⟨.result 28602 .coefficient, true, some 1⟩, ⟨.result 28599 .coefficient, true, some 1⟩])

def event28607 : Event := .survivorFold (1) 28606

def exact28608RawTerms : List Term := []

theorem exact28608RawTermsValid :
    exact28608RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28608 : Event := .resultExact (⟨.program ⟨257⟩, ⟨36907⟩⟩) exact28608RawTerms (.finite 1764) 28605 (.finite 1764) (some (28606))

def event28609 : Event := .predecessor (⟨.program ⟨257⟩, ⟨36908⟩⟩) 0 ⟨36907⟩ 28608

def event28610 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨36908⟩⟩) (.identity (.predecessor 0 28609 .coefficient))

def event28611 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨36908⟩⟩) (.finite 1764)

def event28612 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37358⟩⟩) 0 ⟨36908⟩ 28611

def event28613 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37358⟩⟩) (.authority (.programFamilyFact))

def exact28614RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨37358⟩⟩], []⟩, (1)⟩]

theorem exact28614RawTermsValid :
    exact28614RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28614 : Event := .resultExact (⟨.program ⟨257⟩, ⟨37358⟩⟩) exact28614RawTerms (.finite 42) 28613 .exactZero (none)

def event28615 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37359⟩⟩) 0 ⟨37358⟩ 28614

def event28616 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37359⟩⟩) (.identity (.predecessor 0 28615 .coefficient))

def event28617 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨37359⟩⟩) (.finite 42)

def event28618 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37998⟩⟩) 0 ⟨37359⟩ 28617

def event28619 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37998⟩⟩) (.authority (.relationPreimageSource ⟨84⟩))

def exact28620RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨37998⟩⟩]⟩, (1)⟩]

theorem exact28620RawTermsValid :
    exact28620RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28620 : Event := .resultExact (⟨.program ⟨257⟩, ⟨37998⟩⟩) exact28620RawTerms (.finite 5647228698) 28619 .exactZero (none)

def event28621 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35⟩⟩) (.authority (.operator))

def exact28622RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact28622RawTermsValid :
    exact28622RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28622 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35⟩⟩) exact28622RawTerms .large 28621 .exactZero (none)

def event28623 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37999⟩⟩) 0 ⟨35⟩ 28622

def event28624 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37999⟩⟩) 1 ⟨37998⟩ 28620

def event28625 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37999⟩⟩) (.product (.predecessor 0 28623 .coefficient) (.predecessor 1 28624 .coefficient) (⟨false, false, none, none, none⟩))

def event28626 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨37999⟩⟩, .operator (⟨28622, 0⟩, ⟨28620, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37998⟩⟩]⟩, (1)⟩)

def exact28627RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37998⟩⟩]⟩, (1)⟩]

theorem exact28627RawTermsValid :
    exact28627RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28627 : Event := .resultExact (⟨.program ⟨257⟩, ⟨37999⟩⟩) exact28627RawTerms .large 28625 .exactZero (none)

def event28628 : Event := .preFoldPolynomial 28627 [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37998⟩⟩]⟩, (1)⟩] .exactZero none

def exact28629RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37998⟩⟩]⟩, (1)⟩]

def event28629 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨37999⟩⟩) 28628 exact28629RawTerms .large 28625 .exactZero (none)

def event28630 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨39090⟩⟩)

def event28631 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event28632 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event28633 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨140⟩⟩) (.authority (.operator))

def event28634 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨140⟩⟩) (.finite 19)

def event28635 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event28636 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event28637 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event28638 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event28639 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 28638

def event28640 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 28636

def event28641 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 28639 .coefficient) (.value (.predecessor 1 28640 .coefficient)))

def event28642 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event28643 : Event := .predecessor (⟨.program ⟨257⟩, ⟨393⟩⟩) 0 ⟨392⟩ 28642

def event28644 : Event := .predecessor (⟨.program ⟨257⟩, ⟨393⟩⟩) 1 ⟨140⟩ 28634

def event28645 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨393⟩⟩) (.sum [.predecessor 0 28643 .coefficient, .predecessor 1 28644 .coefficient])

def event28646 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨393⟩⟩) (.finite 655359)

def event28647 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5439⟩⟩) 0 ⟨393⟩ 28646

def event28648 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5439⟩⟩) 1 ⟨5426⟩ 28632

def event28649 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5439⟩⟩) (.identity (.predecessor 1 28648 .coefficient))

def event28650 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5439⟩⟩) (.finite 655360)

def event28651 : Event := .predecessor (⟨.program ⟨257⟩, ⟨36906⟩⟩) 0 ⟨5439⟩ 28650

def event28652 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨36906⟩⟩) (.authority (.programFamilyFact))

def exact28653RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨36906⟩⟩], []⟩, (1)⟩]

theorem exact28653RawTermsValid :
    exact28653RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28653 : Event := .resultExact (⟨.program ⟨257⟩, ⟨36906⟩⟩) exact28653RawTerms (.finite 42) 28652 .exactZero (none)

def event28654 : Event := .predecessor (⟨.program ⟨257⟩, ⟨13751⟩⟩) 0 ⟨5439⟩ 28650

def event28655 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨13751⟩⟩) (.authority (.programFamilyFact))

def exact28656RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨13751⟩⟩], []⟩, (1)⟩]

theorem exact28656RawTermsValid :
    exact28656RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28656 : Event := .resultExact (⟨.program ⟨257⟩, ⟨13751⟩⟩) exact28656RawTerms (.finite 42) 28655 .exactZero (none)

def event28657 : Event := .predecessor (⟨.program ⟨257⟩, ⟨36907⟩⟩) 0 ⟨13751⟩ 28656

def event28658 : Event := .predecessor (⟨.program ⟨257⟩, ⟨36907⟩⟩) 1 ⟨36906⟩ 28653

def event28659 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨36907⟩⟩) (.product (.predecessor 0 28657 .coefficient) (.predecessor 1 28658 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event28660 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨36907⟩⟩, .operator (⟨28656, 0⟩, ⟨28653, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨13751⟩⟩, ⟨.program ⟨257⟩, ⟨36906⟩⟩], []⟩, (1)⟩)

def exact28661RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨13751⟩⟩, ⟨.program ⟨257⟩, ⟨36906⟩⟩], []⟩, (1)⟩]

theorem exact28661RawTermsValid :
    exact28661RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28661 : Event := .resultExact (⟨.program ⟨257⟩, ⟨36907⟩⟩) exact28661RawTerms (.finite 1764) 28659 .exactZero (none)

def event28662 : Event := .predecessor (⟨.program ⟨257⟩, ⟨36908⟩⟩) 0 ⟨36907⟩ 28661

def event28663 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨36908⟩⟩) (.identity (.predecessor 0 28662 .coefficient))

def event28664 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨36908⟩⟩) (.finite 1764)

def event28665 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37358⟩⟩) 0 ⟨36908⟩ 28664

def event28666 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37358⟩⟩) (.authority (.programFamilyFact))

def exact28667RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨37358⟩⟩], []⟩, (1)⟩]

theorem exact28667RawTermsValid :
    exact28667RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28667 : Event := .resultExact (⟨.program ⟨257⟩, ⟨37358⟩⟩) exact28667RawTerms (.finite 42) 28666 .exactZero (none)

def event28668 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37359⟩⟩) 0 ⟨37358⟩ 28667

def event28669 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37359⟩⟩) (.identity (.predecessor 0 28668 .coefficient))

def event28670 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨37359⟩⟩) (.finite 42)

def event28671 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38501⟩⟩) 0 ⟨37359⟩ 28670

def eventLeaf1776 : Array AnnotatedEvent := #[
  { event := event28416
    frameStart := 28364 },
  { event := event28417
    frameStart := 28364 },
  { event := event28418
    frameStart := 28418 },
  { event := event28419
    frameStart := 28418 },
  { event := event28420
    frameStart := 28418 },
  { event := event28421
    frameStart := 28418 },
  { event := event28422
    frameStart := 28418 },
  { event := event28423
    frameStart := 28418 },
  { event := event28424
    frameStart := 28418 },
  { event := event28425
    frameStart := 28418 },
  { event := event28426
    frameStart := 28418 },
  { event := event28427
    frameStart := 28418 },
  { event := event28428
    frameStart := 28418 },
  { event := event28429
    frameStart := 28418 },
  { event := event28430
    frameStart := 28418 },
  { event := event28431
    frameStart := 28418 }
]

def eventLeaf1777 : Array AnnotatedEvent := #[
  { event := event28432
    frameStart := 28418 },
  { event := event28433
    frameStart := 28418 },
  { event := event28434
    frameStart := 28418 },
  { event := event28435
    frameStart := 28418 },
  { event := event28436
    frameStart := 28418 },
  { event := event28437
    frameStart := 28418 },
  { event := event28438
    frameStart := 28418 },
  { event := event28439
    frameStart := 28418 },
  { event := event28440
    frameStart := 28418 },
  { event := event28441
    frameStart := 28418 },
  { event := event28442
    frameStart := 28418 },
  { event := event28443
    frameStart := 28418 },
  { event := event28444
    frameStart := 28418 },
  { event := event28445
    frameStart := 28418 },
  { event := event28446
    frameStart := 28418 },
  { event := event28447
    frameStart := 28418 }
]

def eventLeaf1778 : Array AnnotatedEvent := #[
  { event := event28448
    frameStart := 28418 },
  { event := event28449
    frameStart := 28418 },
  { event := event28450
    frameStart := 28418 },
  { event := event28451
    frameStart := 28418 },
  { event := event28452
    frameStart := 28418 },
  { event := event28453
    frameStart := 28418 },
  { event := event28454
    frameStart := 28418 },
  { event := event28455
    frameStart := 28418 },
  { event := event28456
    frameStart := 28418 },
  { event := event28457
    frameStart := 28418 },
  { event := event28458
    frameStart := 28418 },
  { event := event28459
    frameStart := 28418 },
  { event := event28460
    frameStart := 28418 },
  { event := event28461
    frameStart := 28418 },
  { event := event28462
    frameStart := 28418 },
  { event := event28463
    frameStart := 28418 }
]

def eventLeaf1779 : Array AnnotatedEvent := #[
  { event := event28464
    frameStart := 28418 },
  { event := event28465
    frameStart := 28418 },
  { event := event28466
    frameStart := 28418 },
  { event := event28467
    frameStart := 28418 },
  { event := event28468
    frameStart := 28418 },
  { event := event28469
    frameStart := 28418 },
  { event := event28470
    frameStart := 28418 },
  { event := event28471
    frameStart := 28418 },
  { event := event28472
    frameStart := 28418 },
  { event := event28473
    frameStart := 28418 },
  { event := event28474
    frameStart := 28418 },
  { event := event28475
    frameStart := 28418 },
  { event := event28476
    frameStart := 28418 },
  { event := event28477
    frameStart := 28418 },
  { event := event28478
    frameStart := 28418 },
  { event := event28479
    frameStart := 28418 }
]

def eventLeaf1780 : Array AnnotatedEvent := #[
  { event := event28480
    frameStart := 28418 },
  { event := event28481
    frameStart := 28418 },
  { event := event28482
    frameStart := 28418 },
  { event := event28483
    frameStart := 28418 },
  { event := event28484
    frameStart := 28418 },
  { event := event28485
    frameStart := 28418 },
  { event := event28486
    frameStart := 28418 },
  { event := event28487
    frameStart := 28418 },
  { event := event28488
    frameStart := 28418 },
  { event := event28489
    frameStart := 28418 },
  { event := event28490
    frameStart := 28418 },
  { event := event28491
    frameStart := 28418 },
  { event := event28492
    frameStart := 28418 },
  { event := event28493
    frameStart := 28418 },
  { event := event28494
    frameStart := 28418 },
  { event := event28495
    frameStart := 28418 }
]

def eventLeaf1781 : Array AnnotatedEvent := #[
  { event := event28496
    frameStart := 28418 },
  { event := event28497
    frameStart := 28418 },
  { event := event28498
    frameStart := 28418 },
  { event := event28499
    frameStart := 28418 },
  { event := event28500
    frameStart := 28418 },
  { event := event28501
    frameStart := 28418 },
  { event := event28502
    frameStart := 28418 },
  { event := event28503
    frameStart := 28418 },
  { event := event28504
    frameStart := 28418 },
  { event := event28505
    frameStart := 28418 },
  { event := event28506
    frameStart := 28418 },
  { event := event28507
    frameStart := 28418 },
  { event := event28508
    frameStart := 28418 },
  { event := event28509
    frameStart := 28418 },
  { event := event28510
    frameStart := 28418 },
  { event := event28511
    frameStart := 28418 }
]

def eventLeaf1782 : Array AnnotatedEvent := #[
  { event := event28512
    frameStart := 28418 },
  { event := event28513
    frameStart := 28418 },
  { event := event28514
    frameStart := 28418 },
  { event := event28515
    frameStart := 28418 },
  { event := event28516
    frameStart := 28418 },
  { event := event28517
    frameStart := 28418 },
  { event := event28518
    frameStart := 28418 },
  { event := event28519
    frameStart := 28418 },
  { event := event28520
    frameStart := 28418 },
  { event := event28521
    frameStart := 28418 },
  { event := event28522
    frameStart := 0 },
  { event := event28523
    frameStart := 0 },
  { event := event28524
    frameStart := 0 },
  { event := event28525
    frameStart := 0 },
  { event := event28526
    frameStart := 0 },
  { event := event28527
    frameStart := 0 }
]

def eventLeaf1783 : Array AnnotatedEvent := #[
  { event := event28528
    frameStart := 0 },
  { event := event28529
    frameStart := 0 },
  { event := event28530
    frameStart := 0 },
  { event := event28531
    frameStart := 0 },
  { event := event28532
    frameStart := 0 },
  { event := event28533
    frameStart := 0 },
  { event := event28534
    frameStart := 0 },
  { event := event28535
    frameStart := 0 },
  { event := event28536
    frameStart := 0 },
  { event := event28537
    frameStart := 0 },
  { event := event28538
    frameStart := 0 },
  { event := event28539
    frameStart := 0 },
  { event := event28540
    frameStart := 0 },
  { event := event28541
    frameStart := 0 },
  { event := event28542
    frameStart := 0 },
  { event := event28543
    frameStart := 0 }
]

def eventLeaf1784 : Array AnnotatedEvent := #[
  { event := event28544
    frameStart := 0 },
  { event := event28545
    frameStart := 0 },
  { event := event28546
    frameStart := 0 },
  { event := event28547
    frameStart := 0 },
  { event := event28548
    frameStart := 0 },
  { event := event28549
    frameStart := 0 },
  { event := event28550
    frameStart := 0 },
  { event := event28551
    frameStart := 0 },
  { event := event28552
    frameStart := 0 },
  { event := event28553
    frameStart := 0 },
  { event := event28554
    frameStart := 0 },
  { event := event28555
    frameStart := 0 },
  { event := event28556
    frameStart := 0 },
  { event := event28557
    frameStart := 0 },
  { event := event28558
    frameStart := 0 },
  { event := event28559
    frameStart := 0 }
]

def eventLeaf1785 : Array AnnotatedEvent := #[
  { event := event28560
    frameStart := 0 },
  { event := event28561
    frameStart := 0 },
  { event := event28562
    frameStart := 0 },
  { event := event28563
    frameStart := 0 },
  { event := event28564
    frameStart := 0 },
  { event := event28565
    frameStart := 0 },
  { event := event28566
    frameStart := 0 },
  { event := event28567
    frameStart := 0 },
  { event := event28568
    frameStart := 0 },
  { event := event28569
    frameStart := 0 },
  { event := event28570
    frameStart := 0 },
  { event := event28571
    frameStart := 0 },
  { event := event28572
    frameStart := 0 },
  { event := event28573
    frameStart := 0 },
  { event := event28574
    frameStart := 0 },
  { event := event28575
    frameStart := 0 }
]

def eventLeaf1786 : Array AnnotatedEvent := #[
  { event := event28576
    frameStart := 28576 },
  { event := event28577
    frameStart := 28576 },
  { event := event28578
    frameStart := 28576 },
  { event := event28579
    frameStart := 28576 },
  { event := event28580
    frameStart := 28576 },
  { event := event28581
    frameStart := 28576 },
  { event := event28582
    frameStart := 28576 },
  { event := event28583
    frameStart := 28576 },
  { event := event28584
    frameStart := 28576 },
  { event := event28585
    frameStart := 28576 },
  { event := event28586
    frameStart := 28576 },
  { event := event28587
    frameStart := 28576 },
  { event := event28588
    frameStart := 28576 },
  { event := event28589
    frameStart := 28576 },
  { event := event28590
    frameStart := 28576 },
  { event := event28591
    frameStart := 28576 }
]

def eventLeaf1787 : Array AnnotatedEvent := #[
  { event := event28592
    frameStart := 28576 },
  { event := event28593
    frameStart := 28576 },
  { event := event28594
    frameStart := 28576 },
  { event := event28595
    frameStart := 28576 },
  { event := event28596
    frameStart := 28576 },
  { event := event28597
    frameStart := 28576 },
  { event := event28598
    frameStart := 28576 },
  { event := event28599
    frameStart := 28576 },
  { event := event28600
    frameStart := 28576 },
  { event := event28601
    frameStart := 28576 },
  { event := event28602
    frameStart := 28576 },
  { event := event28603
    frameStart := 28576 },
  { event := event28604
    frameStart := 28576 },
  { event := event28605
    frameStart := 28576 },
  { event := event28606
    frameStart := 28576 },
  { event := event28607
    frameStart := 28576 }
]

def eventLeaf1788 : Array AnnotatedEvent := #[
  { event := event28608
    frameStart := 28576 },
  { event := event28609
    frameStart := 28576 },
  { event := event28610
    frameStart := 28576 },
  { event := event28611
    frameStart := 28576 },
  { event := event28612
    frameStart := 28576 },
  { event := event28613
    frameStart := 28576 },
  { event := event28614
    frameStart := 28576 },
  { event := event28615
    frameStart := 28576 },
  { event := event28616
    frameStart := 28576 },
  { event := event28617
    frameStart := 28576 },
  { event := event28618
    frameStart := 28576 },
  { event := event28619
    frameStart := 28576 },
  { event := event28620
    frameStart := 28576 },
  { event := event28621
    frameStart := 28576 },
  { event := event28622
    frameStart := 28576 },
  { event := event28623
    frameStart := 28576 }
]

def eventLeaf1789 : Array AnnotatedEvent := #[
  { event := event28624
    frameStart := 28576 },
  { event := event28625
    frameStart := 28576 },
  { event := event28626
    frameStart := 28576 },
  { event := event28627
    frameStart := 28576 },
  { event := event28628
    frameStart := 28576 },
  { event := event28629
    frameStart := 28576 },
  { event := event28630
    frameStart := 28630 },
  { event := event28631
    frameStart := 28630 },
  { event := event28632
    frameStart := 28630 },
  { event := event28633
    frameStart := 28630 },
  { event := event28634
    frameStart := 28630 },
  { event := event28635
    frameStart := 28630 },
  { event := event28636
    frameStart := 28630 },
  { event := event28637
    frameStart := 28630 },
  { event := event28638
    frameStart := 28630 },
  { event := event28639
    frameStart := 28630 }
]

def eventLeaf1790 : Array AnnotatedEvent := #[
  { event := event28640
    frameStart := 28630 },
  { event := event28641
    frameStart := 28630 },
  { event := event28642
    frameStart := 28630 },
  { event := event28643
    frameStart := 28630 },
  { event := event28644
    frameStart := 28630 },
  { event := event28645
    frameStart := 28630 },
  { event := event28646
    frameStart := 28630 },
  { event := event28647
    frameStart := 28630 },
  { event := event28648
    frameStart := 28630 },
  { event := event28649
    frameStart := 28630 },
  { event := event28650
    frameStart := 28630 },
  { event := event28651
    frameStart := 28630 },
  { event := event28652
    frameStart := 28630 },
  { event := event28653
    frameStart := 28630 },
  { event := event28654
    frameStart := 28630 },
  { event := event28655
    frameStart := 28630 }
]

def eventLeaf1791 : Array AnnotatedEvent := #[
  { event := event28656
    frameStart := 28630 },
  { event := event28657
    frameStart := 28630 },
  { event := event28658
    frameStart := 28630 },
  { event := event28659
    frameStart := 28630 },
  { event := event28660
    frameStart := 28630 },
  { event := event28661
    frameStart := 28630 },
  { event := event28662
    frameStart := 28630 },
  { event := event28663
    frameStart := 28630 },
  { event := event28664
    frameStart := 28630 },
  { event := event28665
    frameStart := 28630 },
  { event := event28666
    frameStart := 28630 },
  { event := event28667
    frameStart := 28630 },
  { event := event28668
    frameStart := 28630 },
  { event := event28669
    frameStart := 28630 },
  { event := event28670
    frameStart := 28630 },
  { event := event28671
    frameStart := 28630 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events111
