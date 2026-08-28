import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events303

open Mxx.Certificate.OperationalNoise
open CertificateABI

def exact77568RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨9556⟩⟩]⟩, (1)⟩]

theorem exact77568RawTermsValid :
    exact77568RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77568 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9556⟩⟩) exact77568RawTerms (.finite 8192) 77567 .exactZero (none)

def event77569 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9557⟩⟩) 0 ⟨9556⟩ 77568

def event77570 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9557⟩⟩) 1 ⟨2370⟩ 77559

def event77571 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9557⟩⟩) (.scale (.predecessor 0 77569 .coefficient) (.value (.predecessor 1 77570 .coefficient)))

def exact77572RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨9556⟩⟩]⟩, (1)⟩]

theorem exact77572RawTermsValid :
    exact77572RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77572 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9557⟩⟩) exact77572RawTerms (.finite 8192) 77571 .exactZero (none)

def event77573 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7299⟩⟩) 0 ⟨7178⟩ 77562

def event77574 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7299⟩⟩) (.identity (.predecessor 0 77573 .coefficient))

def exact77575RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7299⟩⟩]⟩, (1)⟩]

theorem exact77575RawTermsValid :
    exact77575RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77575 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7299⟩⟩) exact77575RawTerms .large 77574 .exactZero (none)

def event77576 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9558⟩⟩) 0 ⟨7299⟩ 77575

def event77577 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9558⟩⟩) 1 ⟨9557⟩ 77572

def event77578 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9558⟩⟩) (.product (.predecessor 0 77576 .coefficient) (.predecessor 1 77577 .coefficient) (⟨false, false, none, none, none⟩))

def event77579 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨9558⟩⟩, .operator (⟨77575, 0⟩, ⟨77572, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩]⟩, (1)⟩)

def exact77580RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩]⟩, (1)⟩]

theorem exact77580RawTermsValid :
    exact77580RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77580 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9558⟩⟩) exact77580RawTerms .large 77578 .exactZero (none)

def event77581 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41413⟩⟩) 0 ⟨9558⟩ 77580

def event77582 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41413⟩⟩) 1 ⟨41412⟩ 77557

def event77583 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41413⟩⟩) (.sum [.predecessor 0 77581 .coefficient, .predecessor 1 77582 .coefficient])

def exact77584RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨14271⟩⟩, ⟨.program ⟨257⟩, ⟨39938⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact77584RawTermsValid :
    exact77584RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77584 : Event := .resultExact (⟨.program ⟨257⟩, ⟨41413⟩⟩) exact77584RawTerms .large 77583 .exactZero (none)

def event77585 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41688⟩⟩) 0 ⟨41413⟩ 77584

def event77586 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41688⟩⟩) 1 ⟨41685⟩ 77541

def event77587 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41688⟩⟩) (.product (.predecessor 0 77585 .coefficient) (.predecessor 1 77586 .coefficient) (⟨false, false, none, none, none⟩))

def event77588 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨41688⟩⟩, .operator (⟨77584, 0⟩, ⟨77541, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41685⟩⟩]⟩, (1)⟩)

def event77589 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨41688⟩⟩, .operator (⟨77584, 1⟩, ⟨77541, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨14271⟩⟩, ⟨.program ⟨257⟩, ⟨39938⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨41685⟩⟩]⟩, (-1)⟩)

def event77590 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨41688⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨14271⟩⟩, ⟨.program ⟨257⟩, ⟨39938⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨41685⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨41685⟩⟩) ⟨41145⟩ 77538)

def event77591 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨41688⟩⟩, .relation 77590 0, ⟨[⟨.program ⟨257⟩, ⟨14271⟩⟩, ⟨.program ⟨257⟩, ⟨39938⟩⟩], [⟨.program ⟨257⟩, ⟨41145⟩⟩]⟩, (-1)⟩)

def exact77592RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41685⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨14271⟩⟩, ⟨.program ⟨257⟩, ⟨39938⟩⟩], [⟨.program ⟨257⟩, ⟨41145⟩⟩]⟩, (-1)⟩]

theorem exact77592RawTermsValid :
    exact77592RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77592 : Event := .resultExact (⟨.program ⟨257⟩, ⟨41688⟩⟩) exact77592RawTerms .large 77587 .exactZero (none)

def event77593 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40156⟩⟩) 0 ⟨39940⟩ 77530

def event77594 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40156⟩⟩) (.authority (.programFamilyFact))

def exact77595RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨40156⟩⟩], []⟩, (1)⟩]

theorem exact77595RawTermsValid :
    exact77595RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77595 : Event := .resultExact (⟨.program ⟨257⟩, ⟨40156⟩⟩) exact77595RawTerms (.finite 46) 77594 .exactZero (none)

def event77596 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40158⟩⟩) 0 ⟨6908⟩ 77552

def event77597 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40158⟩⟩) 1 ⟨40156⟩ 77595

def event77598 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40158⟩⟩) (.product (.predecessor 0 77596 .coefficient) (.predecessor 1 77597 .coefficient) (⟨false, true, none, none, some 1⟩))

def event77599 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨40158⟩⟩, .operator (⟨77552, 0⟩, ⟨77595, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨40156⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact77600RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨40156⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact77600RawTermsValid :
    exact77600RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77600 : Event := .resultExact (⟨.program ⟨257⟩, ⟨40158⟩⟩) exact77600RawTerms .large 77598 .exactZero (none)

def event77601 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7193⟩⟩) 0 ⟨7177⟩ 77534

def event77602 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7193⟩⟩) (.authority (.operator))

def exact77603RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7193⟩⟩]⟩, (1)⟩]

theorem exact77603RawTermsValid :
    exact77603RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77603 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7193⟩⟩) exact77603RawTerms .large 77602 .exactZero (none)

def event77604 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40159⟩⟩) 0 ⟨7193⟩ 77603

def event77605 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40159⟩⟩) 1 ⟨40158⟩ 77600

def event77606 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40159⟩⟩) (.sum [.predecessor 0 77604 .coefficient, .predecessor 1 77605 .coefficient])

def exact77607RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7193⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨40156⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact77607RawTermsValid :
    exact77607RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77607 : Event := .resultExact (⟨.program ⟨257⟩, ⟨40159⟩⟩) exact77607RawTerms .large 77606 .exactZero (none)

def event77608 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41689⟩⟩) 0 ⟨40159⟩ 77607

def event77609 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41689⟩⟩) 1 ⟨41688⟩ 77592

def event77610 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41689⟩⟩) (.sum [.predecessor 0 77608 .coefficient, .predecessor 1 77609 .coefficient])

def exact77611RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7193⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41685⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨14271⟩⟩, ⟨.program ⟨257⟩, ⟨39938⟩⟩], [⟨.program ⟨257⟩, ⟨41145⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨40156⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact77611RawTermsValid :
    exact77611RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77611 : Event := .resultExact (⟨.program ⟨257⟩, ⟨41689⟩⟩) exact77611RawTerms .large 77610 .exactZero (none)

def event77612 : Event := .preFoldPolynomial 77611 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7193⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41685⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨14271⟩⟩, ⟨.program ⟨257⟩, ⟨39938⟩⟩], [⟨.program ⟨257⟩, ⟨41145⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨40156⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact77613RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7193⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41685⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨14271⟩⟩, ⟨.program ⟨257⟩, ⟨39938⟩⟩], [⟨.program ⟨257⟩, ⟨41145⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨40156⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event77613 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨41689⟩⟩) 77612 exact77613RawTerms .large 77610 .exactZero (none)

def event77614 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨39940⟩⟩) ⟨⟨72⟩, ⟨51⟩, ⟨135⟩⟩ ⟨77448, 77614⟩

def event77615 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨40612⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40609⟩⟩]⟩) (1) 0 2 (.universal 77614 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40609⟩⟩]⟩) (none) 77613)

def event77616 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨40612⟩⟩, .relation 77615 0, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7193⟩⟩]⟩, (1)⟩)

def event77617 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨40612⟩⟩, .relation 77615 1, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41685⟩⟩]⟩, (-1)⟩)

def event77618 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨40612⟩⟩, .relation 77615 2, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨14271⟩⟩, ⟨.program ⟨257⟩, ⟨39938⟩⟩], [⟨.program ⟨257⟩, ⟨41145⟩⟩]⟩, (1)⟩)

def event77619 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨40612⟩⟩, .relation 77615 3, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨40156⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact77620RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7193⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41685⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨14271⟩⟩, ⟨.program ⟨257⟩, ⟨39938⟩⟩], [⟨.program ⟨257⟩, ⟨41145⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨40156⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact77620RawTermsValid :
    exact77620RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77620 : Event := .resultExact (⟨.program ⟨257⟩, ⟨40612⟩⟩) exact77620RawTerms .large 77444 (.finite 202072841853861888) (some (77446))

def event77621 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41687⟩⟩) 0 ⟨40612⟩ 77620

def event77622 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41687⟩⟩) 1 ⟨41686⟩ 77434

def event77623 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41687⟩⟩) (.sum [.predecessor 0 77621 .coefficient, .predecessor 1 77622 .coefficient])

def event77624 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨41687⟩⟩, .operator (⟨77620, 2⟩, ⟨77434, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨14271⟩⟩, ⟨.program ⟨257⟩, ⟨39938⟩⟩], [⟨.program ⟨257⟩, ⟨41145⟩⟩]⟩, (-1)⟩)

def event77625 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨41687⟩⟩, .operator (⟨77620, 1⟩, ⟨77434, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41685⟩⟩]⟩, (1)⟩)

def event77626 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41687⟩⟩) (.sum [.result 77620 .summary, .result 77434 .summary])

def exact77627RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7193⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨40156⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact77627RawTermsValid :
    exact77627RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77627 : Event := .resultExact (⟨.program ⟨257⟩, ⟨41687⟩⟩) exact77627RawTerms .large 77623 (.finite 2998218789909838430208) (some (77626))

def event77628 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42141⟩⟩) 0 ⟨41687⟩ 77627

def event77629 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42141⟩⟩) 1 ⟨42139⟩ 77350

def event77630 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42141⟩⟩) (.product (.predecessor 0 77628 .coefficient) (.predecessor 1 77629 .coefficient) (⟨false, false, none, none, none⟩))

def event77631 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42141⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨42139⟩⟩]⟩) [⟨.result 77350 .coefficient, false, none⟩])

def event77632 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42141⟩⟩) (.product (.result 77627 .summary) (.transfer 77631) (⟨false, false, none, none, none⟩))

def event77633 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨42141⟩⟩, .operator (⟨77627, 0⟩, ⟨77350, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨42139⟩⟩]⟩, (1)⟩)

def event77634 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨42141⟩⟩, .operator (⟨77627, 1⟩, ⟨77350, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨40156⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨42139⟩⟩]⟩, (-1)⟩)

def event77635 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨42141⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨40156⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨42139⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨42139⟩⟩) ⟨41315⟩ 77347)

def event77636 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨42141⟩⟩, .relation 77635 0, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨40156⟩⟩], [⟨.program ⟨257⟩, ⟨41315⟩⟩]⟩, (-1)⟩)

def exact77637RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨42139⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨40156⟩⟩], [⟨.program ⟨257⟩, ⟨41315⟩⟩]⟩, (-1)⟩]

theorem exact77637RawTermsValid :
    exact77637RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77637 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42141⟩⟩) exact77637RawTerms .large 77630 (.finite 32193129122288627115968346193920) (some (77632))

def event77638 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40976⟩⟩) 0 ⟨40157⟩ 3172

def event77639 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40976⟩⟩) (.authority (.relationPreimageSource ⟨87⟩))

def exact77640RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨40976⟩⟩]⟩, (1)⟩]

theorem exact77640RawTermsValid :
    exact77640RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77640 : Event := .resultExact (⟨.program ⟨257⟩, ⟨40976⟩⟩) exact77640RawTerms (.finite 5647228698) 77639 .exactZero (none)

def event77641 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40978⟩⟩) 0 ⟨40976⟩ 77640

def event77642 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40978⟩⟩) 1 ⟨2370⟩ 4

def event77643 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40978⟩⟩) (.scale (.predecessor 0 77641 .coefficient) (.value (.predecessor 1 77642 .coefficient)))

def exact77644RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨40976⟩⟩]⟩, (1)⟩]

theorem exact77644RawTermsValid :
    exact77644RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77644 : Event := .resultExact (⟨.program ⟨257⟩, ⟨40978⟩⟩) exact77644RawTerms (.finite 5647228698) 77643 .exactZero (none)

def event77645 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40979⟩⟩) 0 ⟨10368⟩ 75995

def event77646 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40979⟩⟩) 1 ⟨40978⟩ 77644

def event77647 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40979⟩⟩) (.product (.predecessor 0 77645 .coefficient) (.predecessor 1 77646 .coefficient) (⟨false, false, none, none, none⟩))

def event77648 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40979⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨40976⟩⟩]⟩) [⟨.result 77640 .coefficient, false, none⟩])

def event77649 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40979⟩⟩) (.product (.result 75995 .summary) (.transfer 77648) (⟨false, false, none, none, none⟩))

def event77650 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨40979⟩⟩, .operator (⟨75995, 0⟩, ⟨77644, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40976⟩⟩]⟩, (1)⟩)

def event77651 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨40977⟩⟩)

def event77652 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event77653 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event77654 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10267⟩⟩) (.authority (.operator))

def event77655 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10267⟩⟩) (.finite 15)

def event77656 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event77657 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event77658 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event77659 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event77660 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 77659

def event77661 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 77657

def event77662 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 77660 .coefficient) (.value (.predecessor 1 77661 .coefficient)))

def event77663 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event77664 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10269⟩⟩) 0 ⟨392⟩ 77663

def event77665 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10269⟩⟩) 1 ⟨10267⟩ 77655

def event77666 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10269⟩⟩) (.sum [.predecessor 0 77664 .coefficient, .predecessor 1 77665 .coefficient])

def event77667 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10269⟩⟩) (.finite 655355)

def event77668 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10325⟩⟩) 0 ⟨10269⟩ 77667

def event77669 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10325⟩⟩) 1 ⟨5426⟩ 77653

def event77670 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10325⟩⟩) (.identity (.predecessor 1 77669 .coefficient))

def event77671 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10325⟩⟩) (.finite 655360)

def event77672 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39938⟩⟩) 0 ⟨10325⟩ 77671

def event77673 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39938⟩⟩) (.authority (.programFamilyFact))

def exact77674RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨39938⟩⟩], []⟩, (1)⟩]

theorem exact77674RawTermsValid :
    exact77674RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77674 : Event := .resultExact (⟨.program ⟨257⟩, ⟨39938⟩⟩) exact77674RawTerms (.finite 46) 77673 .exactZero (none)

def event77675 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14271⟩⟩) 0 ⟨10325⟩ 77671

def event77676 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14271⟩⟩) (.authority (.programFamilyFact))

def exact77677RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14271⟩⟩], []⟩, (1)⟩]

theorem exact77677RawTermsValid :
    exact77677RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77677 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14271⟩⟩) exact77677RawTerms (.finite 46) 77676 .exactZero (none)

def event77678 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39939⟩⟩) 0 ⟨14271⟩ 77677

def event77679 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39939⟩⟩) 1 ⟨39938⟩ 77674

def event77680 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39939⟩⟩) (.product (.predecessor 0 77678 .coefficient) (.predecessor 1 77679 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event77681 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39939⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨14271⟩⟩, ⟨.program ⟨257⟩, ⟨39938⟩⟩], []⟩) [⟨.result 77677 .coefficient, true, some 1⟩, ⟨.result 77674 .coefficient, true, some 1⟩])

def event77682 : Event := .survivorFold (1) 77681

def exact77683RawTerms : List Term := []

theorem exact77683RawTermsValid :
    exact77683RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77683 : Event := .resultExact (⟨.program ⟨257⟩, ⟨39939⟩⟩) exact77683RawTerms (.finite 2116) 77680 (.finite 2116) (some (77681))

def event77684 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39940⟩⟩) 0 ⟨39939⟩ 77683

def event77685 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39940⟩⟩) (.identity (.predecessor 0 77684 .coefficient))

def event77686 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨39940⟩⟩) (.finite 2116)

def event77687 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40156⟩⟩) 0 ⟨39940⟩ 77686

def event77688 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40156⟩⟩) (.authority (.programFamilyFact))

def exact77689RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨40156⟩⟩], []⟩, (1)⟩]

theorem exact77689RawTermsValid :
    exact77689RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77689 : Event := .resultExact (⟨.program ⟨257⟩, ⟨40156⟩⟩) exact77689RawTerms (.finite 46) 77688 .exactZero (none)

def event77690 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40157⟩⟩) 0 ⟨40156⟩ 77689

def event77691 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40157⟩⟩) (.identity (.predecessor 0 77690 .coefficient))

def event77692 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨40157⟩⟩) (.finite 46)

def event77693 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40976⟩⟩) 0 ⟨40157⟩ 77692

def event77694 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40976⟩⟩) (.authority (.relationPreimageSource ⟨87⟩))

def exact77695RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨40976⟩⟩]⟩, (1)⟩]

theorem exact77695RawTermsValid :
    exact77695RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77695 : Event := .resultExact (⟨.program ⟨257⟩, ⟨40976⟩⟩) exact77695RawTerms (.finite 5647228698) 77694 .exactZero (none)

def event77696 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35⟩⟩) (.authority (.operator))

def exact77697RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact77697RawTermsValid :
    exact77697RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77697 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35⟩⟩) exact77697RawTerms .large 77696 .exactZero (none)

def event77698 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40977⟩⟩) 0 ⟨35⟩ 77697

def event77699 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40977⟩⟩) 1 ⟨40976⟩ 77695

def event77700 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40977⟩⟩) (.product (.predecessor 0 77698 .coefficient) (.predecessor 1 77699 .coefficient) (⟨false, false, none, none, none⟩))

def event77701 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨40977⟩⟩, .operator (⟨77697, 0⟩, ⟨77695, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40976⟩⟩]⟩, (1)⟩)

def exact77702RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40976⟩⟩]⟩, (1)⟩]

theorem exact77702RawTermsValid :
    exact77702RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77702 : Event := .resultExact (⟨.program ⟨257⟩, ⟨40977⟩⟩) exact77702RawTerms .large 77700 .exactZero (none)

def event77703 : Event := .preFoldPolynomial 77702 [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40976⟩⟩]⟩, (1)⟩] .exactZero none

def exact77704RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40976⟩⟩]⟩, (1)⟩]

def event77704 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨40977⟩⟩) 77703 exact77704RawTerms .large 77700 .exactZero (none)

def event77705 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨42143⟩⟩)

def event77706 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event77707 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event77708 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10267⟩⟩) (.authority (.operator))

def event77709 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10267⟩⟩) (.finite 15)

def event77710 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event77711 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event77712 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event77713 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event77714 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 77713

def event77715 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 77711

def event77716 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 77714 .coefficient) (.value (.predecessor 1 77715 .coefficient)))

def event77717 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event77718 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10269⟩⟩) 0 ⟨392⟩ 77717

def event77719 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10269⟩⟩) 1 ⟨10267⟩ 77709

def event77720 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10269⟩⟩) (.sum [.predecessor 0 77718 .coefficient, .predecessor 1 77719 .coefficient])

def event77721 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10269⟩⟩) (.finite 655355)

def event77722 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10325⟩⟩) 0 ⟨10269⟩ 77721

def event77723 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10325⟩⟩) 1 ⟨5426⟩ 77707

def event77724 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10325⟩⟩) (.identity (.predecessor 1 77723 .coefficient))

def event77725 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10325⟩⟩) (.finite 655360)

def event77726 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39938⟩⟩) 0 ⟨10325⟩ 77725

def event77727 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39938⟩⟩) (.authority (.programFamilyFact))

def exact77728RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨39938⟩⟩], []⟩, (1)⟩]

theorem exact77728RawTermsValid :
    exact77728RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77728 : Event := .resultExact (⟨.program ⟨257⟩, ⟨39938⟩⟩) exact77728RawTerms (.finite 46) 77727 .exactZero (none)

def event77729 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14271⟩⟩) 0 ⟨10325⟩ 77725

def event77730 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14271⟩⟩) (.authority (.programFamilyFact))

def exact77731RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14271⟩⟩], []⟩, (1)⟩]

theorem exact77731RawTermsValid :
    exact77731RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77731 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14271⟩⟩) exact77731RawTerms (.finite 46) 77730 .exactZero (none)

def event77732 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39939⟩⟩) 0 ⟨14271⟩ 77731

def event77733 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39939⟩⟩) 1 ⟨39938⟩ 77728

def event77734 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39939⟩⟩) (.product (.predecessor 0 77732 .coefficient) (.predecessor 1 77733 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event77735 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨39939⟩⟩, .operator (⟨77731, 0⟩, ⟨77728, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨14271⟩⟩, ⟨.program ⟨257⟩, ⟨39938⟩⟩], []⟩, (1)⟩)

def exact77736RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14271⟩⟩, ⟨.program ⟨257⟩, ⟨39938⟩⟩], []⟩, (1)⟩]

theorem exact77736RawTermsValid :
    exact77736RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77736 : Event := .resultExact (⟨.program ⟨257⟩, ⟨39939⟩⟩) exact77736RawTerms (.finite 2116) 77734 .exactZero (none)

def event77737 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39940⟩⟩) 0 ⟨39939⟩ 77736

def event77738 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39940⟩⟩) (.identity (.predecessor 0 77737 .coefficient))

def event77739 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨39940⟩⟩) (.finite 2116)

def event77740 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40156⟩⟩) 0 ⟨39940⟩ 77739

def event77741 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40156⟩⟩) (.authority (.programFamilyFact))

def exact77742RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨40156⟩⟩], []⟩, (1)⟩]

theorem exact77742RawTermsValid :
    exact77742RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77742 : Event := .resultExact (⟨.program ⟨257⟩, ⟨40156⟩⟩) exact77742RawTerms (.finite 46) 77741 .exactZero (none)

def event77743 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40157⟩⟩) 0 ⟨40156⟩ 77742

def event77744 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40157⟩⟩) (.identity (.predecessor 0 77743 .coefficient))

def event77745 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨40157⟩⟩) (.finite 46)

def event77746 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41313⟩⟩) 0 ⟨40157⟩ 77745

def event77747 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41313⟩⟩) (.authority (.programFamilyFact))

def event77748 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨41313⟩⟩) (.finite 3720)

def event77749 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨7177⟩⟩) .missing

def event77750 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41315⟩⟩) 0 ⟨7177⟩ 77749

def event77751 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41315⟩⟩) 1 ⟨41313⟩ 77748

def event77752 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41315⟩⟩) (.authority (.operator))

def exact77753RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨41315⟩⟩]⟩, (1)⟩]

theorem exact77753RawTermsValid :
    exact77753RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77753 : Event := .resultExact (⟨.program ⟨257⟩, ⟨41315⟩⟩) exact77753RawTerms .large 77752 .exactZero (none)

def event77754 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42139⟩⟩) 0 ⟨41315⟩ 77753

def event77755 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42139⟩⟩) (.authority (.operator))

def exact77756RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨42139⟩⟩]⟩, (1)⟩]

theorem exact77756RawTermsValid :
    exact77756RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77756 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42139⟩⟩) exact77756RawTerms (.finite 8192) 77755 .exactZero (none)

def event77757 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event77758 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event77759 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41490⟩⟩) 0 ⟨40157⟩ 77745

def event77760 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41490⟩⟩) 1 ⟨136⟩ 77758

def event77761 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41490⟩⟩) (.sum [.predecessor 0 77759 .coefficient, .predecessor 1 77760 .coefficient])

def event77762 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨41490⟩⟩) (.finite 46)

def event77763 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41491⟩⟩) 0 ⟨41490⟩ 77762

def event77764 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41491⟩⟩) (.identity (.predecessor 0 77763 .coefficient))

def exact77765RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨40156⟩⟩], []⟩, (1)⟩]

theorem exact77765RawTermsValid :
    exact77765RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77765 : Event := .resultExact (⟨.program ⟨257⟩, ⟨41491⟩⟩) exact77765RawTerms (.finite 46) 77764 .exactZero (none)

def event77766 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact77767RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact77767RawTermsValid :
    exact77767RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77767 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact77767RawTerms .large 77766 .exactZero (none)

def event77768 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41492⟩⟩) 0 ⟨6908⟩ 77767

def event77769 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41492⟩⟩) 1 ⟨41491⟩ 77765

def event77770 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41492⟩⟩) (.product (.predecessor 0 77768 .coefficient) (.predecessor 1 77769 .coefficient) (⟨false, false, none, none, none⟩))

def event77771 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨41492⟩⟩, .operator (⟨77767, 0⟩, ⟨77765, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨40156⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact77772RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨40156⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact77772RawTermsValid :
    exact77772RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77772 : Event := .resultExact (⟨.program ⟨257⟩, ⟨41492⟩⟩) exact77772RawTerms .large 77770 .exactZero (none)

def event77773 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7193⟩⟩) 0 ⟨7177⟩ 77749

def event77774 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7193⟩⟩) (.authority (.operator))

def exact77775RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7193⟩⟩]⟩, (1)⟩]

theorem exact77775RawTermsValid :
    exact77775RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77775 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7193⟩⟩) exact77775RawTerms .large 77774 .exactZero (none)

def event77776 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41493⟩⟩) 0 ⟨7193⟩ 77775

def event77777 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41493⟩⟩) 1 ⟨41492⟩ 77772

def event77778 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41493⟩⟩) (.sum [.predecessor 0 77776 .coefficient, .predecessor 1 77777 .coefficient])

def exact77779RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7193⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨40156⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact77779RawTermsValid :
    exact77779RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77779 : Event := .resultExact (⟨.program ⟨257⟩, ⟨41493⟩⟩) exact77779RawTerms .large 77778 .exactZero (none)

def event77780 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42140⟩⟩) 0 ⟨41493⟩ 77779

def event77781 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42140⟩⟩) 1 ⟨42139⟩ 77756

def event77782 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42140⟩⟩) (.product (.predecessor 0 77780 .coefficient) (.predecessor 1 77781 .coefficient) (⟨false, false, none, none, none⟩))

def event77783 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨42140⟩⟩, .operator (⟨77779, 0⟩, ⟨77756, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨42139⟩⟩]⟩, (1)⟩)

def event77784 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨42140⟩⟩, .operator (⟨77779, 1⟩, ⟨77756, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨40156⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨42139⟩⟩]⟩, (-1)⟩)

def event77785 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨42140⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨40156⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨42139⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨42139⟩⟩) ⟨41315⟩ 77753)

def event77786 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨42140⟩⟩, .relation 77785 0, ⟨[⟨.program ⟨257⟩, ⟨40156⟩⟩], [⟨.program ⟨257⟩, ⟨41315⟩⟩]⟩, (-1)⟩)

def exact77787RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨42139⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨40156⟩⟩], [⟨.program ⟨257⟩, ⟨41315⟩⟩]⟩, (-1)⟩]

theorem exact77787RawTermsValid :
    exact77787RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77787 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42140⟩⟩) exact77787RawTerms .large 77782 .exactZero (none)

def event77788 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40397⟩⟩) 0 ⟨40157⟩ 77745

def event77789 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40397⟩⟩) (.authority (.programFamilyFact))

def exact77790RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨40397⟩⟩], []⟩, (1)⟩]

theorem exact77790RawTermsValid :
    exact77790RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77790 : Event := .resultExact (⟨.program ⟨257⟩, ⟨40397⟩⟩) exact77790RawTerms (.finite 63) 77789 .exactZero (none)

def event77791 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40398⟩⟩) 0 ⟨6908⟩ 77767

def event77792 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40398⟩⟩) 1 ⟨40397⟩ 77790

def event77793 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40398⟩⟩) (.product (.predecessor 0 77791 .coefficient) (.predecessor 1 77792 .coefficient) (⟨false, true, none, none, some 1⟩))

def event77794 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨40398⟩⟩, .operator (⟨77767, 0⟩, ⟨77790, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨40397⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact77795RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨40397⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact77795RawTermsValid :
    exact77795RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77795 : Event := .resultExact (⟨.program ⟨257⟩, ⟨40398⟩⟩) exact77795RawTerms .large 77793 .exactZero (none)

def event77796 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7226⟩⟩) 0 ⟨7177⟩ 77749

def event77797 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7226⟩⟩) (.authority (.operator))

def exact77798RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7226⟩⟩]⟩, (1)⟩]

theorem exact77798RawTermsValid :
    exact77798RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77798 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7226⟩⟩) exact77798RawTerms .large 77797 .exactZero (none)

def event77799 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40399⟩⟩) 0 ⟨7226⟩ 77798

def event77800 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40399⟩⟩) 1 ⟨40398⟩ 77795

def event77801 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40399⟩⟩) (.sum [.predecessor 0 77799 .coefficient, .predecessor 1 77800 .coefficient])

def exact77802RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7226⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨40397⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact77802RawTermsValid :
    exact77802RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77802 : Event := .resultExact (⟨.program ⟨257⟩, ⟨40399⟩⟩) exact77802RawTerms .large 77801 .exactZero (none)

def event77803 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42143⟩⟩) 0 ⟨40399⟩ 77802

def event77804 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42143⟩⟩) 1 ⟨42140⟩ 77787

def event77805 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42143⟩⟩) (.sum [.predecessor 0 77803 .coefficient, .predecessor 1 77804 .coefficient])

def exact77806RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨42139⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7226⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨40156⟩⟩], [⟨.program ⟨257⟩, ⟨41315⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨40397⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact77806RawTermsValid :
    exact77806RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77806 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42143⟩⟩) exact77806RawTerms .large 77805 .exactZero (none)

def event77807 : Event := .preFoldPolynomial 77806 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨42139⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7226⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨40156⟩⟩], [⟨.program ⟨257⟩, ⟨41315⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨40397⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact77808RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨42139⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7226⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨40156⟩⟩], [⟨.program ⟨257⟩, ⟨41315⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨40397⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event77808 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨42143⟩⟩) 77807 exact77808RawTerms .large 77805 .exactZero (none)

def event77809 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨40157⟩⟩) ⟨⟨105⟩, ⟨87⟩, ⟨135⟩⟩ ⟨77651, 77809⟩

def event77810 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨40979⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40976⟩⟩]⟩) (1) 0 2 (.universal 77809 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40976⟩⟩]⟩) (none) 77808)

def event77811 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨40979⟩⟩, .relation 77810 1, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7226⟩⟩]⟩, (1)⟩)

def event77812 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨40979⟩⟩, .relation 77810 0, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨42139⟩⟩]⟩, (-1)⟩)

def event77813 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨40979⟩⟩, .relation 77810 2, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨40156⟩⟩], [⟨.program ⟨257⟩, ⟨41315⟩⟩]⟩, (1)⟩)

def event77814 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨40979⟩⟩, .relation 77810 3, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨40397⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact77815RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨42139⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7226⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨40156⟩⟩], [⟨.program ⟨257⟩, ⟨41315⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨40397⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact77815RawTermsValid :
    exact77815RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77815 : Event := .resultExact (⟨.program ⟨257⟩, ⟨40979⟩⟩) exact77815RawTerms .large 77647 (.finite 202072841853861888) (some (77649))

def event77816 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42142⟩⟩) 0 ⟨40979⟩ 77815

def event77817 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42142⟩⟩) 1 ⟨42141⟩ 77637

def event77818 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42142⟩⟩) (.sum [.predecessor 0 77816 .coefficient, .predecessor 1 77817 .coefficient])

def event77819 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨42142⟩⟩, .operator (⟨77815, 0⟩, ⟨77637, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨42139⟩⟩]⟩, (1)⟩)

def event77820 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨42142⟩⟩, .operator (⟨77815, 2⟩, ⟨77637, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨40156⟩⟩], [⟨.program ⟨257⟩, ⟨41315⟩⟩]⟩, (-1)⟩)

def event77821 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42142⟩⟩) (.sum [.result 77815 .summary, .result 77637 .summary])

def exact77822RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7226⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨40397⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact77822RawTermsValid :
    exact77822RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77822 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42142⟩⟩) exact77822RawTerms .large 77818 (.finite 32193129122288829188810200055808) (some (77821))

def event77823 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38633⟩⟩) 0 ⟨37477⟩ 3195

def eventLeaf4848 : Array AnnotatedEvent := #[
  { event := event77568
    frameStart := 77496 },
  { event := event77569
    frameStart := 77496 },
  { event := event77570
    frameStart := 77496 },
  { event := event77571
    frameStart := 77496 },
  { event := event77572
    frameStart := 77496 },
  { event := event77573
    frameStart := 77496 },
  { event := event77574
    frameStart := 77496 },
  { event := event77575
    frameStart := 77496 },
  { event := event77576
    frameStart := 77496 },
  { event := event77577
    frameStart := 77496 },
  { event := event77578
    frameStart := 77496 },
  { event := event77579
    frameStart := 77496 },
  { event := event77580
    frameStart := 77496 },
  { event := event77581
    frameStart := 77496 },
  { event := event77582
    frameStart := 77496 },
  { event := event77583
    frameStart := 77496 }
]

def eventLeaf4849 : Array AnnotatedEvent := #[
  { event := event77584
    frameStart := 77496 },
  { event := event77585
    frameStart := 77496 },
  { event := event77586
    frameStart := 77496 },
  { event := event77587
    frameStart := 77496 },
  { event := event77588
    frameStart := 77496 },
  { event := event77589
    frameStart := 77496 },
  { event := event77590
    frameStart := 77496 },
  { event := event77591
    frameStart := 77496 },
  { event := event77592
    frameStart := 77496 },
  { event := event77593
    frameStart := 77496 },
  { event := event77594
    frameStart := 77496 },
  { event := event77595
    frameStart := 77496 },
  { event := event77596
    frameStart := 77496 },
  { event := event77597
    frameStart := 77496 },
  { event := event77598
    frameStart := 77496 },
  { event := event77599
    frameStart := 77496 }
]

def eventLeaf4850 : Array AnnotatedEvent := #[
  { event := event77600
    frameStart := 77496 },
  { event := event77601
    frameStart := 77496 },
  { event := event77602
    frameStart := 77496 },
  { event := event77603
    frameStart := 77496 },
  { event := event77604
    frameStart := 77496 },
  { event := event77605
    frameStart := 77496 },
  { event := event77606
    frameStart := 77496 },
  { event := event77607
    frameStart := 77496 },
  { event := event77608
    frameStart := 77496 },
  { event := event77609
    frameStart := 77496 },
  { event := event77610
    frameStart := 77496 },
  { event := event77611
    frameStart := 77496 },
  { event := event77612
    frameStart := 77496 },
  { event := event77613
    frameStart := 77496 },
  { event := event77614
    frameStart := 0 },
  { event := event77615
    frameStart := 0 }
]

def eventLeaf4851 : Array AnnotatedEvent := #[
  { event := event77616
    frameStart := 0 },
  { event := event77617
    frameStart := 0 },
  { event := event77618
    frameStart := 0 },
  { event := event77619
    frameStart := 0 },
  { event := event77620
    frameStart := 0 },
  { event := event77621
    frameStart := 0 },
  { event := event77622
    frameStart := 0 },
  { event := event77623
    frameStart := 0 },
  { event := event77624
    frameStart := 0 },
  { event := event77625
    frameStart := 0 },
  { event := event77626
    frameStart := 0 },
  { event := event77627
    frameStart := 0 },
  { event := event77628
    frameStart := 0 },
  { event := event77629
    frameStart := 0 },
  { event := event77630
    frameStart := 0 },
  { event := event77631
    frameStart := 0 }
]

def eventLeaf4852 : Array AnnotatedEvent := #[
  { event := event77632
    frameStart := 0 },
  { event := event77633
    frameStart := 0 },
  { event := event77634
    frameStart := 0 },
  { event := event77635
    frameStart := 0 },
  { event := event77636
    frameStart := 0 },
  { event := event77637
    frameStart := 0 },
  { event := event77638
    frameStart := 0 },
  { event := event77639
    frameStart := 0 },
  { event := event77640
    frameStart := 0 },
  { event := event77641
    frameStart := 0 },
  { event := event77642
    frameStart := 0 },
  { event := event77643
    frameStart := 0 },
  { event := event77644
    frameStart := 0 },
  { event := event77645
    frameStart := 0 },
  { event := event77646
    frameStart := 0 },
  { event := event77647
    frameStart := 0 }
]

def eventLeaf4853 : Array AnnotatedEvent := #[
  { event := event77648
    frameStart := 0 },
  { event := event77649
    frameStart := 0 },
  { event := event77650
    frameStart := 0 },
  { event := event77651
    frameStart := 77651 },
  { event := event77652
    frameStart := 77651 },
  { event := event77653
    frameStart := 77651 },
  { event := event77654
    frameStart := 77651 },
  { event := event77655
    frameStart := 77651 },
  { event := event77656
    frameStart := 77651 },
  { event := event77657
    frameStart := 77651 },
  { event := event77658
    frameStart := 77651 },
  { event := event77659
    frameStart := 77651 },
  { event := event77660
    frameStart := 77651 },
  { event := event77661
    frameStart := 77651 },
  { event := event77662
    frameStart := 77651 },
  { event := event77663
    frameStart := 77651 }
]

def eventLeaf4854 : Array AnnotatedEvent := #[
  { event := event77664
    frameStart := 77651 },
  { event := event77665
    frameStart := 77651 },
  { event := event77666
    frameStart := 77651 },
  { event := event77667
    frameStart := 77651 },
  { event := event77668
    frameStart := 77651 },
  { event := event77669
    frameStart := 77651 },
  { event := event77670
    frameStart := 77651 },
  { event := event77671
    frameStart := 77651 },
  { event := event77672
    frameStart := 77651 },
  { event := event77673
    frameStart := 77651 },
  { event := event77674
    frameStart := 77651 },
  { event := event77675
    frameStart := 77651 },
  { event := event77676
    frameStart := 77651 },
  { event := event77677
    frameStart := 77651 },
  { event := event77678
    frameStart := 77651 },
  { event := event77679
    frameStart := 77651 }
]

def eventLeaf4855 : Array AnnotatedEvent := #[
  { event := event77680
    frameStart := 77651 },
  { event := event77681
    frameStart := 77651 },
  { event := event77682
    frameStart := 77651 },
  { event := event77683
    frameStart := 77651 },
  { event := event77684
    frameStart := 77651 },
  { event := event77685
    frameStart := 77651 },
  { event := event77686
    frameStart := 77651 },
  { event := event77687
    frameStart := 77651 },
  { event := event77688
    frameStart := 77651 },
  { event := event77689
    frameStart := 77651 },
  { event := event77690
    frameStart := 77651 },
  { event := event77691
    frameStart := 77651 },
  { event := event77692
    frameStart := 77651 },
  { event := event77693
    frameStart := 77651 },
  { event := event77694
    frameStart := 77651 },
  { event := event77695
    frameStart := 77651 }
]

def eventLeaf4856 : Array AnnotatedEvent := #[
  { event := event77696
    frameStart := 77651 },
  { event := event77697
    frameStart := 77651 },
  { event := event77698
    frameStart := 77651 },
  { event := event77699
    frameStart := 77651 },
  { event := event77700
    frameStart := 77651 },
  { event := event77701
    frameStart := 77651 },
  { event := event77702
    frameStart := 77651 },
  { event := event77703
    frameStart := 77651 },
  { event := event77704
    frameStart := 77651 },
  { event := event77705
    frameStart := 77705 },
  { event := event77706
    frameStart := 77705 },
  { event := event77707
    frameStart := 77705 },
  { event := event77708
    frameStart := 77705 },
  { event := event77709
    frameStart := 77705 },
  { event := event77710
    frameStart := 77705 },
  { event := event77711
    frameStart := 77705 }
]

def eventLeaf4857 : Array AnnotatedEvent := #[
  { event := event77712
    frameStart := 77705 },
  { event := event77713
    frameStart := 77705 },
  { event := event77714
    frameStart := 77705 },
  { event := event77715
    frameStart := 77705 },
  { event := event77716
    frameStart := 77705 },
  { event := event77717
    frameStart := 77705 },
  { event := event77718
    frameStart := 77705 },
  { event := event77719
    frameStart := 77705 },
  { event := event77720
    frameStart := 77705 },
  { event := event77721
    frameStart := 77705 },
  { event := event77722
    frameStart := 77705 },
  { event := event77723
    frameStart := 77705 },
  { event := event77724
    frameStart := 77705 },
  { event := event77725
    frameStart := 77705 },
  { event := event77726
    frameStart := 77705 },
  { event := event77727
    frameStart := 77705 }
]

def eventLeaf4858 : Array AnnotatedEvent := #[
  { event := event77728
    frameStart := 77705 },
  { event := event77729
    frameStart := 77705 },
  { event := event77730
    frameStart := 77705 },
  { event := event77731
    frameStart := 77705 },
  { event := event77732
    frameStart := 77705 },
  { event := event77733
    frameStart := 77705 },
  { event := event77734
    frameStart := 77705 },
  { event := event77735
    frameStart := 77705 },
  { event := event77736
    frameStart := 77705 },
  { event := event77737
    frameStart := 77705 },
  { event := event77738
    frameStart := 77705 },
  { event := event77739
    frameStart := 77705 },
  { event := event77740
    frameStart := 77705 },
  { event := event77741
    frameStart := 77705 },
  { event := event77742
    frameStart := 77705 },
  { event := event77743
    frameStart := 77705 }
]

def eventLeaf4859 : Array AnnotatedEvent := #[
  { event := event77744
    frameStart := 77705 },
  { event := event77745
    frameStart := 77705 },
  { event := event77746
    frameStart := 77705 },
  { event := event77747
    frameStart := 77705 },
  { event := event77748
    frameStart := 77705 },
  { event := event77749
    frameStart := 77705 },
  { event := event77750
    frameStart := 77705 },
  { event := event77751
    frameStart := 77705 },
  { event := event77752
    frameStart := 77705 },
  { event := event77753
    frameStart := 77705 },
  { event := event77754
    frameStart := 77705 },
  { event := event77755
    frameStart := 77705 },
  { event := event77756
    frameStart := 77705 },
  { event := event77757
    frameStart := 77705 },
  { event := event77758
    frameStart := 77705 },
  { event := event77759
    frameStart := 77705 }
]

def eventLeaf4860 : Array AnnotatedEvent := #[
  { event := event77760
    frameStart := 77705 },
  { event := event77761
    frameStart := 77705 },
  { event := event77762
    frameStart := 77705 },
  { event := event77763
    frameStart := 77705 },
  { event := event77764
    frameStart := 77705 },
  { event := event77765
    frameStart := 77705 },
  { event := event77766
    frameStart := 77705 },
  { event := event77767
    frameStart := 77705 },
  { event := event77768
    frameStart := 77705 },
  { event := event77769
    frameStart := 77705 },
  { event := event77770
    frameStart := 77705 },
  { event := event77771
    frameStart := 77705 },
  { event := event77772
    frameStart := 77705 },
  { event := event77773
    frameStart := 77705 },
  { event := event77774
    frameStart := 77705 },
  { event := event77775
    frameStart := 77705 }
]

def eventLeaf4861 : Array AnnotatedEvent := #[
  { event := event77776
    frameStart := 77705 },
  { event := event77777
    frameStart := 77705 },
  { event := event77778
    frameStart := 77705 },
  { event := event77779
    frameStart := 77705 },
  { event := event77780
    frameStart := 77705 },
  { event := event77781
    frameStart := 77705 },
  { event := event77782
    frameStart := 77705 },
  { event := event77783
    frameStart := 77705 },
  { event := event77784
    frameStart := 77705 },
  { event := event77785
    frameStart := 77705 },
  { event := event77786
    frameStart := 77705 },
  { event := event77787
    frameStart := 77705 },
  { event := event77788
    frameStart := 77705 },
  { event := event77789
    frameStart := 77705 },
  { event := event77790
    frameStart := 77705 },
  { event := event77791
    frameStart := 77705 }
]

def eventLeaf4862 : Array AnnotatedEvent := #[
  { event := event77792
    frameStart := 77705 },
  { event := event77793
    frameStart := 77705 },
  { event := event77794
    frameStart := 77705 },
  { event := event77795
    frameStart := 77705 },
  { event := event77796
    frameStart := 77705 },
  { event := event77797
    frameStart := 77705 },
  { event := event77798
    frameStart := 77705 },
  { event := event77799
    frameStart := 77705 },
  { event := event77800
    frameStart := 77705 },
  { event := event77801
    frameStart := 77705 },
  { event := event77802
    frameStart := 77705 },
  { event := event77803
    frameStart := 77705 },
  { event := event77804
    frameStart := 77705 },
  { event := event77805
    frameStart := 77705 },
  { event := event77806
    frameStart := 77705 },
  { event := event77807
    frameStart := 77705 }
]

def eventLeaf4863 : Array AnnotatedEvent := #[
  { event := event77808
    frameStart := 77705 },
  { event := event77809
    frameStart := 0 },
  { event := event77810
    frameStart := 0 },
  { event := event77811
    frameStart := 0 },
  { event := event77812
    frameStart := 0 },
  { event := event77813
    frameStart := 0 },
  { event := event77814
    frameStart := 0 },
  { event := event77815
    frameStart := 0 },
  { event := event77816
    frameStart := 0 },
  { event := event77817
    frameStart := 0 },
  { event := event77818
    frameStart := 0 },
  { event := event77819
    frameStart := 0 },
  { event := event77820
    frameStart := 0 },
  { event := event77821
    frameStart := 0 },
  { event := event77822
    frameStart := 0 },
  { event := event77823
    frameStart := 0 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events303
