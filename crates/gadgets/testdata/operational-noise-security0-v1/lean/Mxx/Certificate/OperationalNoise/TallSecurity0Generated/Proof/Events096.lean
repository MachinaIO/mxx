import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events096

open Mxx.Certificate.OperationalNoise
open CertificateABI

def exact24576RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6701⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6764⟩⟩, ⟨.program ⟨214⟩, ⟨7864⟩⟩, ⟨.program ⟨214⟩, ⟨25234⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨9730⟩⟩, ⟨.program ⟨214⟩, ⟨11981⟩⟩], [⟨.program ⟨214⟩, ⟨23128⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16393⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

def event24576 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨25238⟩⟩) 24575 exact24576RawTerms .large 24573 .exactZero (none)

def event24577 : Event := .specializationComputed (⟨.program ⟨214⟩, ⟨11983⟩⟩) ⟨⟨114⟩, ⟨19⟩, ⟨109⟩⟩ ⟨24411, 24577⟩

def event24578 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨19831⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19828⟩⟩]⟩) (1) 0 2 (.universal 24577 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19828⟩⟩]⟩) (none) 24576)

def event24579 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19831⟩⟩, .relation 24578 0, ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6701⟩⟩]⟩, (1)⟩)

def event24580 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19831⟩⟩, .relation 24578 1, ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6764⟩⟩, ⟨.program ⟨214⟩, ⟨7864⟩⟩, ⟨.program ⟨214⟩, ⟨25234⟩⟩]⟩, (-1)⟩)

def event24581 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19831⟩⟩, .relation 24578 2, ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨9730⟩⟩, ⟨.program ⟨214⟩, ⟨11981⟩⟩], [⟨.program ⟨214⟩, ⟨23128⟩⟩]⟩, (1)⟩)

def event24582 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19831⟩⟩, .relation 24578 3, ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨16393⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact24583RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6701⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6764⟩⟩, ⟨.program ⟨214⟩, ⟨7864⟩⟩, ⟨.program ⟨214⟩, ⟨25234⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨9730⟩⟩, ⟨.program ⟨214⟩, ⟨11981⟩⟩], [⟨.program ⟨214⟩, ⟨23128⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨16393⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact24583RawTermsValid :
    exact24583RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event24583 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19831⟩⟩) exact24583RawTerms .large 24407 (.finite 1811303510016) (some (24409))

def event24584 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25236⟩⟩) 0 ⟨19831⟩ 24583

def event24585 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25236⟩⟩) 1 ⟨25235⟩ 24397

def event24586 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25236⟩⟩) (.sum [.predecessor 0 24584 .coefficient, .predecessor 1 24585 .coefficient])

def event24587 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25236⟩⟩, .operator (⟨24583, 2⟩, ⟨24397, 1⟩), ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨9730⟩⟩, ⟨.program ⟨214⟩, ⟨11981⟩⟩], [⟨.program ⟨214⟩, ⟨23128⟩⟩]⟩, (-1)⟩)

def event24588 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25236⟩⟩, .operator (⟨24583, 1⟩, ⟨24397, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6764⟩⟩, ⟨.program ⟨214⟩, ⟨7864⟩⟩, ⟨.program ⟨214⟩, ⟨25234⟩⟩]⟩, (1)⟩)

def event24589 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25236⟩⟩) (.sum [.result 24583 .summary, .result 24397 .summary])

def exact24590RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6701⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨16393⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact24590RawTermsValid :
    exact24590RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event24590 : Event := .resultExact (⟨.program ⟨214⟩, ⟨25236⟩⟩) exact24590RawTerms .large 24586 (.finite 352115681275904) (some (24589))

def event24591 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28775⟩⟩) 0 ⟨25236⟩ 24590

def event24592 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28775⟩⟩) 1 ⟨28773⟩ 24313

def event24593 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28775⟩⟩) (.product (.predecessor 0 24591 .coefficient) (.predecessor 1 24592 .coefficient) (⟨false, false, none, none, none⟩))

def event24594 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28775⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨28773⟩⟩]⟩) [⟨.result 24313 .coefficient, false, none⟩])

def event24595 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28775⟩⟩) (.product (.result 24590 .summary) (.transfer 24594) (⟨false, false, none, none, none⟩))

def event24596 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28775⟩⟩, .operator (⟨24590, 0⟩, ⟨24313, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6701⟩⟩, ⟨.program ⟨214⟩, ⟨28773⟩⟩]⟩, (1)⟩)

def event24597 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28775⟩⟩, .operator (⟨24590, 1⟩, ⟨24313, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨16393⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28773⟩⟩]⟩, (-1)⟩)

def event24598 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨28775⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨16393⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28773⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨28773⟩⟩) ⟨24423⟩ 24310)

def event24599 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28775⟩⟩, .relation 24598 0, ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨16393⟩⟩], [⟨.program ⟨214⟩, ⟨24423⟩⟩]⟩, (-1)⟩)

def exact24600RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6701⟩⟩, ⟨.program ⟨214⟩, ⟨28773⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨16393⟩⟩], [⟨.program ⟨214⟩, ⟨24423⟩⟩]⟩, (-1)⟩]

theorem exact24600RawTermsValid :
    exact24600RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event24600 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28775⟩⟩) exact24600RawTerms .large 24593 (.finite 1292270184133468094464) (some (24595))

def event24601 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21988⟩⟩) 0 ⟨16394⟩ 997

def event24602 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21988⟩⟩) (.authority (.relationPreimageSource ⟨52⟩))

def exact24603RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨21988⟩⟩]⟩, (1)⟩]

theorem exact24603RawTermsValid :
    exact24603RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event24603 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21988⟩⟩) exact24603RawTerms (.finite 136065468) 24602 .exactZero (none)

def event24604 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21990⟩⟩) 0 ⟨21988⟩ 24603

def event24605 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21990⟩⟩) 1 ⟨2348⟩ 4

def event24606 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21990⟩⟩) (.scale (.predecessor 0 24604 .coefficient) (.value (.predecessor 1 24605 .coefficient)))

def exact24607RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨21988⟩⟩]⟩, (1)⟩]

theorem exact24607RawTermsValid :
    exact24607RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event24607 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21990⟩⟩) exact24607RawTerms (.finite 136065468) 24606 .exactZero (none)

def event24608 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21991⟩⟩) 0 ⟨5559⟩ 21512

def event24609 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21991⟩⟩) 1 ⟨21990⟩ 24607

def event24610 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21991⟩⟩) (.product (.predecessor 0 24608 .coefficient) (.predecessor 1 24609 .coefficient) (⟨false, false, none, none, none⟩))

def event24611 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21991⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨21988⟩⟩]⟩) [⟨.result 24603 .coefficient, false, none⟩])

def event24612 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21991⟩⟩) (.product (.result 21512 .summary) (.transfer 24611) (⟨false, false, none, none, none⟩))

def event24613 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21991⟩⟩, .operator (⟨21512, 0⟩, ⟨24607, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21988⟩⟩]⟩, (1)⟩)

def event24614 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨21989⟩⟩)

def event24615 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event24616 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event24617 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨4989⟩⟩) (.authority (.operator))

def event24618 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨4989⟩⟩) (.finite 5)

def event24619 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event24620 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event24621 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event24622 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event24623 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 24622

def event24624 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 24620

def event24625 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 24623 .coefficient) (.value (.predecessor 1 24624 .coefficient)))

def event24626 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event24627 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5514⟩⟩) 0 ⟨5503⟩ 24626

def event24628 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5514⟩⟩) 1 ⟨4989⟩ 24618

def event24629 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5514⟩⟩) (.sum [.predecessor 0 24627 .coefficient, .predecessor 1 24628 .coefficient])

def event24630 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5514⟩⟩) (.finite 222)

def event24631 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5554⟩⟩) 0 ⟨5514⟩ 24630

def event24632 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5554⟩⟩) 1 ⟨961⟩ 24616

def event24633 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5554⟩⟩) (.identity (.predecessor 1 24632 .coefficient))

def event24634 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5554⟩⟩) (.finite 224)

def event24635 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11981⟩⟩) 0 ⟨5554⟩ 24634

def event24636 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11981⟩⟩) (.authority (.programFamilyFact))

def exact24637RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11981⟩⟩], []⟩, (1)⟩]

theorem exact24637RawTermsValid :
    exact24637RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event24637 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11981⟩⟩) exact24637RawTerms (.finite 36) 24636 .exactZero (none)

def event24638 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9730⟩⟩) 0 ⟨5554⟩ 24634

def event24639 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨9730⟩⟩) (.authority (.programFamilyFact))

def exact24640RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨9730⟩⟩], []⟩, (1)⟩]

theorem exact24640RawTermsValid :
    exact24640RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event24640 : Event := .resultExact (⟨.program ⟨214⟩, ⟨9730⟩⟩) exact24640RawTerms (.finite 36) 24639 .exactZero (none)

def event24641 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11982⟩⟩) 0 ⟨9730⟩ 24640

def event24642 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11982⟩⟩) 1 ⟨11981⟩ 24637

def event24643 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11982⟩⟩) (.product (.predecessor 0 24641 .coefficient) (.predecessor 1 24642 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event24644 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11982⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨9730⟩⟩, ⟨.program ⟨214⟩, ⟨11981⟩⟩], []⟩) [⟨.result 24640 .coefficient, true, some 1⟩, ⟨.result 24637 .coefficient, true, some 1⟩])

def event24645 : Event := .survivorFold (1) 24644

def exact24646RawTerms : List Term := []

theorem exact24646RawTermsValid :
    exact24646RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event24646 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11982⟩⟩) exact24646RawTerms (.finite 1296) 24643 (.finite 1296) (some (24644))

def event24647 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11983⟩⟩) 0 ⟨11982⟩ 24646

def event24648 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11983⟩⟩) (.identity (.predecessor 0 24647 .coefficient))

def event24649 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨11983⟩⟩) (.finite 1296)

def event24650 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16393⟩⟩) 0 ⟨11983⟩ 24649

def event24651 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16393⟩⟩) (.authority (.programFamilyFact))

def exact24652RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16393⟩⟩], []⟩, (1)⟩]

theorem exact24652RawTermsValid :
    exact24652RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event24652 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16393⟩⟩) exact24652RawTerms (.finite 36) 24651 .exactZero (none)

def event24653 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16394⟩⟩) 0 ⟨16393⟩ 24652

def event24654 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16394⟩⟩) (.identity (.predecessor 0 24653 .coefficient))

def event24655 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨16394⟩⟩) (.finite 36)

def event24656 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21988⟩⟩) 0 ⟨16394⟩ 24655

def event24657 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21988⟩⟩) (.authority (.relationPreimageSource ⟨52⟩))

def exact24658RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨21988⟩⟩]⟩, (1)⟩]

theorem exact24658RawTermsValid :
    exact24658RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event24658 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21988⟩⟩) exact24658RawTerms (.finite 136065468) 24657 .exactZero (none)

def event24659 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6⟩⟩) (.authority (.operator))

def exact24660RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩]⟩, (1)⟩]

theorem exact24660RawTermsValid :
    exact24660RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event24660 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6⟩⟩) exact24660RawTerms .large 24659 .exactZero (none)

def event24661 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21989⟩⟩) 0 ⟨6⟩ 24660

def event24662 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21989⟩⟩) 1 ⟨21988⟩ 24658

def event24663 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21989⟩⟩) (.product (.predecessor 0 24661 .coefficient) (.predecessor 1 24662 .coefficient) (⟨false, false, none, none, none⟩))

def event24664 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21989⟩⟩, .operator (⟨24660, 0⟩, ⟨24658, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21988⟩⟩]⟩, (1)⟩)

def exact24665RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21988⟩⟩]⟩, (1)⟩]

theorem exact24665RawTermsValid :
    exact24665RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event24665 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21989⟩⟩) exact24665RawTerms .large 24663 .exactZero (none)

def event24666 : Event := .preFoldPolynomial 24665 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21988⟩⟩]⟩, (1)⟩] .exactZero none

def exact24667RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21988⟩⟩]⟩, (1)⟩]

def event24667 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨21989⟩⟩) 24666 exact24667RawTerms .large 24663 .exactZero (none)

def event24668 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨28778⟩⟩)

def event24669 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event24670 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event24671 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨4989⟩⟩) (.authority (.operator))

def event24672 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨4989⟩⟩) (.finite 5)

def event24673 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event24674 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event24675 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event24676 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event24677 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 24676

def event24678 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 24674

def event24679 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 24677 .coefficient) (.value (.predecessor 1 24678 .coefficient)))

def event24680 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event24681 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5514⟩⟩) 0 ⟨5503⟩ 24680

def event24682 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5514⟩⟩) 1 ⟨4989⟩ 24672

def event24683 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5514⟩⟩) (.sum [.predecessor 0 24681 .coefficient, .predecessor 1 24682 .coefficient])

def event24684 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5514⟩⟩) (.finite 222)

def event24685 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5554⟩⟩) 0 ⟨5514⟩ 24684

def event24686 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5554⟩⟩) 1 ⟨961⟩ 24670

def event24687 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5554⟩⟩) (.identity (.predecessor 1 24686 .coefficient))

def event24688 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5554⟩⟩) (.finite 224)

def event24689 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11981⟩⟩) 0 ⟨5554⟩ 24688

def event24690 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11981⟩⟩) (.authority (.programFamilyFact))

def exact24691RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11981⟩⟩], []⟩, (1)⟩]

theorem exact24691RawTermsValid :
    exact24691RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event24691 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11981⟩⟩) exact24691RawTerms (.finite 36) 24690 .exactZero (none)

def event24692 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9730⟩⟩) 0 ⟨5554⟩ 24688

def event24693 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨9730⟩⟩) (.authority (.programFamilyFact))

def exact24694RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨9730⟩⟩], []⟩, (1)⟩]

theorem exact24694RawTermsValid :
    exact24694RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event24694 : Event := .resultExact (⟨.program ⟨214⟩, ⟨9730⟩⟩) exact24694RawTerms (.finite 36) 24693 .exactZero (none)

def event24695 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11982⟩⟩) 0 ⟨9730⟩ 24694

def event24696 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11982⟩⟩) 1 ⟨11981⟩ 24691

def event24697 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11982⟩⟩) (.product (.predecessor 0 24695 .coefficient) (.predecessor 1 24696 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event24698 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨11982⟩⟩, .operator (⟨24694, 0⟩, ⟨24691, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨9730⟩⟩, ⟨.program ⟨214⟩, ⟨11981⟩⟩], []⟩, (1)⟩)

def exact24699RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨9730⟩⟩, ⟨.program ⟨214⟩, ⟨11981⟩⟩], []⟩, (1)⟩]

theorem exact24699RawTermsValid :
    exact24699RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event24699 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11982⟩⟩) exact24699RawTerms (.finite 1296) 24697 .exactZero (none)

def event24700 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11983⟩⟩) 0 ⟨11982⟩ 24699

def event24701 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11983⟩⟩) (.identity (.predecessor 0 24700 .coefficient))

def event24702 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨11983⟩⟩) (.finite 1296)

def event24703 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16393⟩⟩) 0 ⟨11983⟩ 24702

def event24704 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16393⟩⟩) (.authority (.programFamilyFact))

def exact24705RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16393⟩⟩], []⟩, (1)⟩]

theorem exact24705RawTermsValid :
    exact24705RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event24705 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16393⟩⟩) exact24705RawTerms (.finite 36) 24704 .exactZero (none)

def event24706 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16394⟩⟩) 0 ⟨16393⟩ 24705

def event24707 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16394⟩⟩) (.identity (.predecessor 0 24706 .coefficient))

def event24708 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨16394⟩⟩) (.finite 36)

def event24709 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24421⟩⟩) 0 ⟨16394⟩ 24708

def event24710 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24421⟩⟩) (.authority (.programFamilyFact))

def event24711 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨24421⟩⟩) (.finite 3720)

def event24712 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨6689⟩⟩) .missing

def event24713 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24423⟩⟩) 0 ⟨6689⟩ 24712

def event24714 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24423⟩⟩) 1 ⟨24421⟩ 24711

def event24715 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24423⟩⟩) (.authority (.operator))

def exact24716RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨24423⟩⟩]⟩, (1)⟩]

theorem exact24716RawTermsValid :
    exact24716RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event24716 : Event := .resultExact (⟨.program ⟨214⟩, ⟨24423⟩⟩) exact24716RawTerms .large 24715 .exactZero (none)

def event24717 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28773⟩⟩) 0 ⟨24423⟩ 24716

def event24718 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28773⟩⟩) (.authority (.operator))

def exact24719RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨28773⟩⟩]⟩, (1)⟩]

theorem exact24719RawTermsValid :
    exact24719RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event24719 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28773⟩⟩) exact24719RawTerms (.finite 8192) 24718 .exactZero (none)

def event24720 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨110⟩⟩) (.authority (.operator))

def event24721 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨110⟩⟩) .exactZero

def event24722 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16433⟩⟩) 0 ⟨16394⟩ 24708

def event24723 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16433⟩⟩) 1 ⟨110⟩ 24721

def event24724 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16433⟩⟩) (.sum [.predecessor 0 24722 .coefficient, .predecessor 1 24723 .coefficient])

def event24725 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨16433⟩⟩) (.finite 36)

def event24726 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16434⟩⟩) 0 ⟨16433⟩ 24725

def event24727 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16434⟩⟩) (.identity (.predecessor 0 24726 .coefficient))

def exact24728RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16393⟩⟩], []⟩, (1)⟩]

theorem exact24728RawTermsValid :
    exact24728RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event24728 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16434⟩⟩) exact24728RawTerms (.finite 36) 24727 .exactZero (none)

def event24729 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6544⟩⟩) (.authority (.factStore))

def exact24730RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact24730RawTermsValid :
    exact24730RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event24730 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6544⟩⟩) exact24730RawTerms .large 24729 .exactZero (none)

def event24731 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16435⟩⟩) 0 ⟨6544⟩ 24730

def event24732 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16435⟩⟩) 1 ⟨16434⟩ 24728

def event24733 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16435⟩⟩) (.product (.predecessor 0 24731 .coefficient) (.predecessor 1 24732 .coefficient) (⟨false, false, none, none, none⟩))

def event24734 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨16435⟩⟩, .operator (⟨24730, 0⟩, ⟨24728, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨16393⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact24735RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16393⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact24735RawTermsValid :
    exact24735RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event24735 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16435⟩⟩) exact24735RawTerms .large 24733 .exactZero (none)

def event24736 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6701⟩⟩) 0 ⟨6689⟩ 24712

def event24737 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6701⟩⟩) (.authority (.operator))

def exact24738RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6701⟩⟩]⟩, (1)⟩]

theorem exact24738RawTermsValid :
    exact24738RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event24738 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6701⟩⟩) exact24738RawTerms .large 24737 .exactZero (none)

def event24739 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16436⟩⟩) 0 ⟨6701⟩ 24738

def event24740 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16436⟩⟩) 1 ⟨16435⟩ 24735

def event24741 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16436⟩⟩) (.sum [.predecessor 0 24739 .coefficient, .predecessor 1 24740 .coefficient])

def exact24742RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6701⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16393⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact24742RawTermsValid :
    exact24742RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event24742 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16436⟩⟩) exact24742RawTerms .large 24741 .exactZero (none)

def event24743 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28774⟩⟩) 0 ⟨16436⟩ 24742

def event24744 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28774⟩⟩) 1 ⟨28773⟩ 24719

def event24745 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28774⟩⟩) (.product (.predecessor 0 24743 .coefficient) (.predecessor 1 24744 .coefficient) (⟨false, false, none, none, none⟩))

def event24746 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28774⟩⟩, .operator (⟨24742, 0⟩, ⟨24719, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6701⟩⟩, ⟨.program ⟨214⟩, ⟨28773⟩⟩]⟩, (1)⟩)

def event24747 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28774⟩⟩, .operator (⟨24742, 1⟩, ⟨24719, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨16393⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28773⟩⟩]⟩, (-1)⟩)

def event24748 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨28774⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨16393⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28773⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨28773⟩⟩) ⟨24423⟩ 24716)

def event24749 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28774⟩⟩, .relation 24748 0, ⟨[⟨.program ⟨214⟩, ⟨16393⟩⟩], [⟨.program ⟨214⟩, ⟨24423⟩⟩]⟩, (-1)⟩)

def exact24750RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6701⟩⟩, ⟨.program ⟨214⟩, ⟨28773⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16393⟩⟩], [⟨.program ⟨214⟩, ⟨24423⟩⟩]⟩, (-1)⟩]

theorem exact24750RawTermsValid :
    exact24750RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event24750 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28774⟩⟩) exact24750RawTerms .large 24745 .exactZero (none)

def event24751 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17129⟩⟩) 0 ⟨16394⟩ 24708

def event24752 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨17129⟩⟩) (.authority (.programFamilyFact))

def exact24753RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨17129⟩⟩], []⟩, (1)⟩]

theorem exact24753RawTermsValid :
    exact24753RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event24753 : Event := .resultExact (⟨.program ⟨214⟩, ⟨17129⟩⟩) exact24753RawTerms (.finite 62) 24752 .exactZero (none)

def event24754 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17130⟩⟩) 0 ⟨6544⟩ 24730

def event24755 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17130⟩⟩) 1 ⟨17129⟩ 24753

def event24756 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨17130⟩⟩) (.product (.predecessor 0 24754 .coefficient) (.predecessor 1 24755 .coefficient) (⟨false, true, none, none, some 1⟩))

def event24757 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨17130⟩⟩, .operator (⟨24730, 0⟩, ⟨24753, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨17129⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact24758RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨17129⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact24758RawTermsValid :
    exact24758RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event24758 : Event := .resultExact (⟨.program ⟨214⟩, ⟨17130⟩⟩) exact24758RawTerms .large 24756 .exactZero (none)

def event24759 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6731⟩⟩) 0 ⟨6689⟩ 24712

def event24760 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6731⟩⟩) (.authority (.operator))

def exact24761RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6731⟩⟩]⟩, (1)⟩]

theorem exact24761RawTermsValid :
    exact24761RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event24761 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6731⟩⟩) exact24761RawTerms .large 24760 .exactZero (none)

def event24762 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17131⟩⟩) 0 ⟨6731⟩ 24761

def event24763 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17131⟩⟩) 1 ⟨17130⟩ 24758

def event24764 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨17131⟩⟩) (.sum [.predecessor 0 24762 .coefficient, .predecessor 1 24763 .coefficient])

def exact24765RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6731⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17129⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact24765RawTermsValid :
    exact24765RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event24765 : Event := .resultExact (⟨.program ⟨214⟩, ⟨17131⟩⟩) exact24765RawTerms .large 24764 .exactZero (none)

def event24766 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28778⟩⟩) 0 ⟨17131⟩ 24765

def event24767 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28778⟩⟩) 1 ⟨28774⟩ 24750

def event24768 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28778⟩⟩) (.sum [.predecessor 0 24766 .coefficient, .predecessor 1 24767 .coefficient])

def exact24769RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6701⟩⟩, ⟨.program ⟨214⟩, ⟨28773⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6731⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16393⟩⟩], [⟨.program ⟨214⟩, ⟨24423⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17129⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact24769RawTermsValid :
    exact24769RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event24769 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28778⟩⟩) exact24769RawTerms .large 24768 .exactZero (none)

def event24770 : Event := .preFoldPolynomial 24769 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6701⟩⟩, ⟨.program ⟨214⟩, ⟨28773⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6731⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16393⟩⟩], [⟨.program ⟨214⟩, ⟨24423⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17129⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩] .exactZero none

def exact24771RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6701⟩⟩, ⟨.program ⟨214⟩, ⟨28773⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6731⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16393⟩⟩], [⟨.program ⟨214⟩, ⟨24423⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17129⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

def event24771 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨28778⟩⟩) 24770 exact24771RawTerms .large 24768 .exactZero (none)

def event24772 : Event := .specializationComputed (⟨.program ⟨214⟩, ⟨16394⟩⟩) ⟨⟨144⟩, ⟨52⟩, ⟨109⟩⟩ ⟨24614, 24772⟩

def event24773 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨21991⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21988⟩⟩]⟩) (1) 0 2 (.universal 24772 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21988⟩⟩]⟩) (none) 24771)

def event24774 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21991⟩⟩, .relation 24773 1, ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6731⟩⟩]⟩, (1)⟩)

def event24775 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21991⟩⟩, .relation 24773 0, ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6701⟩⟩, ⟨.program ⟨214⟩, ⟨28773⟩⟩]⟩, (-1)⟩)

def event24776 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21991⟩⟩, .relation 24773 2, ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨16393⟩⟩], [⟨.program ⟨214⟩, ⟨24423⟩⟩]⟩, (1)⟩)

def event24777 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21991⟩⟩, .relation 24773 3, ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨17129⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact24778RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6701⟩⟩, ⟨.program ⟨214⟩, ⟨28773⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6731⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨16393⟩⟩], [⟨.program ⟨214⟩, ⟨24423⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨17129⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact24778RawTermsValid :
    exact24778RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event24778 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21991⟩⟩) exact24778RawTerms .large 24610 (.finite 1811303510016) (some (24612))

def event24779 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28776⟩⟩) 0 ⟨21991⟩ 24778

def event24780 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28776⟩⟩) 1 ⟨28775⟩ 24600

def event24781 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28776⟩⟩) (.sum [.predecessor 0 24779 .coefficient, .predecessor 1 24780 .coefficient])

def event24782 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28776⟩⟩, .operator (⟨24778, 0⟩, ⟨24600, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6701⟩⟩, ⟨.program ⟨214⟩, ⟨28773⟩⟩]⟩, (1)⟩)

def event24783 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28776⟩⟩, .operator (⟨24778, 2⟩, ⟨24600, 1⟩), ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨16393⟩⟩], [⟨.program ⟨214⟩, ⟨24423⟩⟩]⟩, (-1)⟩)

def event24784 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28776⟩⟩) (.sum [.result 24778 .summary, .result 24600 .summary])

def exact24785RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6731⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨17129⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact24785RawTermsValid :
    exact24785RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event24785 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28776⟩⟩) exact24785RawTerms .large 24781 (.finite 1292270185944771604480) (some (24784))

def event24786 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24358⟩⟩) 0 ⟨16275⟩ 1020

def event24787 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24358⟩⟩) (.authority (.programFamilyFact))

def event24788 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨24358⟩⟩) (.finite 3720)

def event24789 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24360⟩⟩) 0 ⟨6689⟩ 5477

def event24790 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24360⟩⟩) 1 ⟨24358⟩ 24788

def event24791 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24360⟩⟩) (.authority (.operator))

def exact24792RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨24360⟩⟩]⟩, (1)⟩]

theorem exact24792RawTermsValid :
    exact24792RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event24792 : Event := .resultExact (⟨.program ⟨214⟩, ⟨24360⟩⟩) exact24792RawTerms .large 24791 .exactZero (none)

def event24793 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28556⟩⟩) 0 ⟨24360⟩ 24792

def event24794 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28556⟩⟩) (.authority (.operator))

def exact24795RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨28556⟩⟩]⟩, (1)⟩]

theorem exact24795RawTermsValid :
    exact24795RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event24795 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28556⟩⟩) exact24795RawTerms (.finite 8192) 24794 .exactZero (none)

def event24796 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23085⟩⟩) 0 ⟨11787⟩ 1014

def event24797 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23085⟩⟩) (.authority (.programFamilyFact))

def event24798 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨23085⟩⟩) (.finite 3720)

def event24799 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23086⟩⟩) 0 ⟨6689⟩ 5477

def event24800 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23086⟩⟩) 1 ⟨23085⟩ 24798

def event24801 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23086⟩⟩) (.authority (.operator))

def exact24802RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨23086⟩⟩]⟩, (1)⟩]

theorem exact24802RawTermsValid :
    exact24802RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event24802 : Event := .resultExact (⟨.program ⟨214⟩, ⟨23086⟩⟩) exact24802RawTerms .large 24801 .exactZero (none)

def event24803 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25157⟩⟩) 0 ⟨23086⟩ 24802

def event24804 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25157⟩⟩) (.authority (.operator))

def exact24805RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨25157⟩⟩]⟩, (1)⟩]

theorem exact24805RawTermsValid :
    exact24805RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event24805 : Event := .resultExact (⟨.program ⟨214⟩, ⟨25157⟩⟩) exact24805RawTerms (.finite 8192) 24804 .exactZero (none)

def event24806 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11788⟩⟩) 0 ⟨11785⟩ 1003

def event24807 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11788⟩⟩) 1 ⟨6570⟩ 21420

def event24808 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11788⟩⟩) (.tensor (.predecessor 0 24806 .coefficient) (.predecessor 1 24807 .coefficient) true false)

def event24809 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨11788⟩⟩, .operator (⟨1003, 0⟩, ⟨21420, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨11785⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact24810RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨11785⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact24810RawTermsValid :
    exact24810RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event24810 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11788⟩⟩) exact24810RawTerms .large 24808 .exactZero (none)

def event24811 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7353⟩⟩) 0 ⟨5557⟩ 21290

def event24812 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7353⟩⟩) 1 ⟨6783⟩ 9979

def event24813 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7353⟩⟩) (.product (.predecessor 0 24811 .coefficient) (.predecessor 1 24812 .coefficient) (⟨false, false, none, none, none⟩))

def event24814 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7353⟩⟩, .operator (⟨21290, 0⟩, ⟨9979, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6783⟩⟩]⟩, (1)⟩)

def exact24815RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6783⟩⟩]⟩, (1)⟩]

theorem exact24815RawTermsValid :
    exact24815RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event24815 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7353⟩⟩) exact24815RawTerms .large 24813 .exactZero (none)

def event24816 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11789⟩⟩) 0 ⟨7353⟩ 24815

def event24817 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11789⟩⟩) 1 ⟨11788⟩ 24810

def event24818 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11789⟩⟩) (.sum [.predecessor 0 24816 .coefficient, .predecessor 1 24817 .coefficient])

def exact24819RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6783⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨11785⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact24819RawTermsValid :
    exact24819RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event24819 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11789⟩⟩) exact24819RawTerms .large 24818 .exactZero (none)

def event24820 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11790⟩⟩) 0 ⟨11789⟩ 24819

def event24821 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11790⟩⟩) 1 ⟨97⟩ 9971

def event24822 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11790⟩⟩) (.sum [.predecessor 0 24820 .coefficient, .predecessor 1 24821 .coefficient])

def event24823 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11790⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨97⟩⟩]⟩) [⟨.result 9971 .coefficient, false, none⟩])

def event24824 : Event := .survivorFold (1) 24823

def exact24825RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6783⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨11785⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact24825RawTermsValid :
    exact24825RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event24825 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11790⟩⟩) exact24825RawTerms .large 24822 (.finite 26) (some (24823))

def event24826 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11791⟩⟩) 0 ⟨11790⟩ 24825

def event24827 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11791⟩⟩) 1 ⟨9625⟩ 1006

def event24828 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11791⟩⟩) (.product (.predecessor 0 24826 .coefficient) (.predecessor 1 24827 .coefficient) (⟨false, true, none, none, some 1⟩))

def event24829 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11791⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨9625⟩⟩], []⟩) [⟨.result 1006 .coefficient, true, some 1⟩])

def event24830 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11791⟩⟩) (.product (.result 24825 .summary) (.transfer 24829) (⟨false, false, none, none, none⟩))

def event24831 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨11791⟩⟩, .operator (⟨24825, 1⟩, ⟨1006, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨9625⟩⟩, ⟨.program ⟨214⟩, ⟨11785⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def eventLeaf1536 : Array AnnotatedEvent := #[
  { event := event24576
    frameStart := 24459 },
  { event := event24577
    frameStart := 0 },
  { event := event24578
    frameStart := 0 },
  { event := event24579
    frameStart := 0 },
  { event := event24580
    frameStart := 0 },
  { event := event24581
    frameStart := 0 },
  { event := event24582
    frameStart := 0 },
  { event := event24583
    frameStart := 0 },
  { event := event24584
    frameStart := 0 },
  { event := event24585
    frameStart := 0 },
  { event := event24586
    frameStart := 0 },
  { event := event24587
    frameStart := 0 },
  { event := event24588
    frameStart := 0 },
  { event := event24589
    frameStart := 0 },
  { event := event24590
    frameStart := 0 },
  { event := event24591
    frameStart := 0 }
]

def eventLeaf1537 : Array AnnotatedEvent := #[
  { event := event24592
    frameStart := 0 },
  { event := event24593
    frameStart := 0 },
  { event := event24594
    frameStart := 0 },
  { event := event24595
    frameStart := 0 },
  { event := event24596
    frameStart := 0 },
  { event := event24597
    frameStart := 0 },
  { event := event24598
    frameStart := 0 },
  { event := event24599
    frameStart := 0 },
  { event := event24600
    frameStart := 0 },
  { event := event24601
    frameStart := 0 },
  { event := event24602
    frameStart := 0 },
  { event := event24603
    frameStart := 0 },
  { event := event24604
    frameStart := 0 },
  { event := event24605
    frameStart := 0 },
  { event := event24606
    frameStart := 0 },
  { event := event24607
    frameStart := 0 }
]

def eventLeaf1538 : Array AnnotatedEvent := #[
  { event := event24608
    frameStart := 0 },
  { event := event24609
    frameStart := 0 },
  { event := event24610
    frameStart := 0 },
  { event := event24611
    frameStart := 0 },
  { event := event24612
    frameStart := 0 },
  { event := event24613
    frameStart := 0 },
  { event := event24614
    frameStart := 24614 },
  { event := event24615
    frameStart := 24614 },
  { event := event24616
    frameStart := 24614 },
  { event := event24617
    frameStart := 24614 },
  { event := event24618
    frameStart := 24614 },
  { event := event24619
    frameStart := 24614 },
  { event := event24620
    frameStart := 24614 },
  { event := event24621
    frameStart := 24614 },
  { event := event24622
    frameStart := 24614 },
  { event := event24623
    frameStart := 24614 }
]

def eventLeaf1539 : Array AnnotatedEvent := #[
  { event := event24624
    frameStart := 24614 },
  { event := event24625
    frameStart := 24614 },
  { event := event24626
    frameStart := 24614 },
  { event := event24627
    frameStart := 24614 },
  { event := event24628
    frameStart := 24614 },
  { event := event24629
    frameStart := 24614 },
  { event := event24630
    frameStart := 24614 },
  { event := event24631
    frameStart := 24614 },
  { event := event24632
    frameStart := 24614 },
  { event := event24633
    frameStart := 24614 },
  { event := event24634
    frameStart := 24614 },
  { event := event24635
    frameStart := 24614 },
  { event := event24636
    frameStart := 24614 },
  { event := event24637
    frameStart := 24614 },
  { event := event24638
    frameStart := 24614 },
  { event := event24639
    frameStart := 24614 }
]

def eventLeaf1540 : Array AnnotatedEvent := #[
  { event := event24640
    frameStart := 24614 },
  { event := event24641
    frameStart := 24614 },
  { event := event24642
    frameStart := 24614 },
  { event := event24643
    frameStart := 24614 },
  { event := event24644
    frameStart := 24614 },
  { event := event24645
    frameStart := 24614 },
  { event := event24646
    frameStart := 24614 },
  { event := event24647
    frameStart := 24614 },
  { event := event24648
    frameStart := 24614 },
  { event := event24649
    frameStart := 24614 },
  { event := event24650
    frameStart := 24614 },
  { event := event24651
    frameStart := 24614 },
  { event := event24652
    frameStart := 24614 },
  { event := event24653
    frameStart := 24614 },
  { event := event24654
    frameStart := 24614 },
  { event := event24655
    frameStart := 24614 }
]

def eventLeaf1541 : Array AnnotatedEvent := #[
  { event := event24656
    frameStart := 24614 },
  { event := event24657
    frameStart := 24614 },
  { event := event24658
    frameStart := 24614 },
  { event := event24659
    frameStart := 24614 },
  { event := event24660
    frameStart := 24614 },
  { event := event24661
    frameStart := 24614 },
  { event := event24662
    frameStart := 24614 },
  { event := event24663
    frameStart := 24614 },
  { event := event24664
    frameStart := 24614 },
  { event := event24665
    frameStart := 24614 },
  { event := event24666
    frameStart := 24614 },
  { event := event24667
    frameStart := 24614 },
  { event := event24668
    frameStart := 24668 },
  { event := event24669
    frameStart := 24668 },
  { event := event24670
    frameStart := 24668 },
  { event := event24671
    frameStart := 24668 }
]

def eventLeaf1542 : Array AnnotatedEvent := #[
  { event := event24672
    frameStart := 24668 },
  { event := event24673
    frameStart := 24668 },
  { event := event24674
    frameStart := 24668 },
  { event := event24675
    frameStart := 24668 },
  { event := event24676
    frameStart := 24668 },
  { event := event24677
    frameStart := 24668 },
  { event := event24678
    frameStart := 24668 },
  { event := event24679
    frameStart := 24668 },
  { event := event24680
    frameStart := 24668 },
  { event := event24681
    frameStart := 24668 },
  { event := event24682
    frameStart := 24668 },
  { event := event24683
    frameStart := 24668 },
  { event := event24684
    frameStart := 24668 },
  { event := event24685
    frameStart := 24668 },
  { event := event24686
    frameStart := 24668 },
  { event := event24687
    frameStart := 24668 }
]

def eventLeaf1543 : Array AnnotatedEvent := #[
  { event := event24688
    frameStart := 24668 },
  { event := event24689
    frameStart := 24668 },
  { event := event24690
    frameStart := 24668 },
  { event := event24691
    frameStart := 24668 },
  { event := event24692
    frameStart := 24668 },
  { event := event24693
    frameStart := 24668 },
  { event := event24694
    frameStart := 24668 },
  { event := event24695
    frameStart := 24668 },
  { event := event24696
    frameStart := 24668 },
  { event := event24697
    frameStart := 24668 },
  { event := event24698
    frameStart := 24668 },
  { event := event24699
    frameStart := 24668 },
  { event := event24700
    frameStart := 24668 },
  { event := event24701
    frameStart := 24668 },
  { event := event24702
    frameStart := 24668 },
  { event := event24703
    frameStart := 24668 }
]

def eventLeaf1544 : Array AnnotatedEvent := #[
  { event := event24704
    frameStart := 24668 },
  { event := event24705
    frameStart := 24668 },
  { event := event24706
    frameStart := 24668 },
  { event := event24707
    frameStart := 24668 },
  { event := event24708
    frameStart := 24668 },
  { event := event24709
    frameStart := 24668 },
  { event := event24710
    frameStart := 24668 },
  { event := event24711
    frameStart := 24668 },
  { event := event24712
    frameStart := 24668 },
  { event := event24713
    frameStart := 24668 },
  { event := event24714
    frameStart := 24668 },
  { event := event24715
    frameStart := 24668 },
  { event := event24716
    frameStart := 24668 },
  { event := event24717
    frameStart := 24668 },
  { event := event24718
    frameStart := 24668 },
  { event := event24719
    frameStart := 24668 }
]

def eventLeaf1545 : Array AnnotatedEvent := #[
  { event := event24720
    frameStart := 24668 },
  { event := event24721
    frameStart := 24668 },
  { event := event24722
    frameStart := 24668 },
  { event := event24723
    frameStart := 24668 },
  { event := event24724
    frameStart := 24668 },
  { event := event24725
    frameStart := 24668 },
  { event := event24726
    frameStart := 24668 },
  { event := event24727
    frameStart := 24668 },
  { event := event24728
    frameStart := 24668 },
  { event := event24729
    frameStart := 24668 },
  { event := event24730
    frameStart := 24668 },
  { event := event24731
    frameStart := 24668 },
  { event := event24732
    frameStart := 24668 },
  { event := event24733
    frameStart := 24668 },
  { event := event24734
    frameStart := 24668 },
  { event := event24735
    frameStart := 24668 }
]

def eventLeaf1546 : Array AnnotatedEvent := #[
  { event := event24736
    frameStart := 24668 },
  { event := event24737
    frameStart := 24668 },
  { event := event24738
    frameStart := 24668 },
  { event := event24739
    frameStart := 24668 },
  { event := event24740
    frameStart := 24668 },
  { event := event24741
    frameStart := 24668 },
  { event := event24742
    frameStart := 24668 },
  { event := event24743
    frameStart := 24668 },
  { event := event24744
    frameStart := 24668 },
  { event := event24745
    frameStart := 24668 },
  { event := event24746
    frameStart := 24668 },
  { event := event24747
    frameStart := 24668 },
  { event := event24748
    frameStart := 24668 },
  { event := event24749
    frameStart := 24668 },
  { event := event24750
    frameStart := 24668 },
  { event := event24751
    frameStart := 24668 }
]

def eventLeaf1547 : Array AnnotatedEvent := #[
  { event := event24752
    frameStart := 24668 },
  { event := event24753
    frameStart := 24668 },
  { event := event24754
    frameStart := 24668 },
  { event := event24755
    frameStart := 24668 },
  { event := event24756
    frameStart := 24668 },
  { event := event24757
    frameStart := 24668 },
  { event := event24758
    frameStart := 24668 },
  { event := event24759
    frameStart := 24668 },
  { event := event24760
    frameStart := 24668 },
  { event := event24761
    frameStart := 24668 },
  { event := event24762
    frameStart := 24668 },
  { event := event24763
    frameStart := 24668 },
  { event := event24764
    frameStart := 24668 },
  { event := event24765
    frameStart := 24668 },
  { event := event24766
    frameStart := 24668 },
  { event := event24767
    frameStart := 24668 }
]

def eventLeaf1548 : Array AnnotatedEvent := #[
  { event := event24768
    frameStart := 24668 },
  { event := event24769
    frameStart := 24668 },
  { event := event24770
    frameStart := 24668 },
  { event := event24771
    frameStart := 24668 },
  { event := event24772
    frameStart := 0 },
  { event := event24773
    frameStart := 0 },
  { event := event24774
    frameStart := 0 },
  { event := event24775
    frameStart := 0 },
  { event := event24776
    frameStart := 0 },
  { event := event24777
    frameStart := 0 },
  { event := event24778
    frameStart := 0 },
  { event := event24779
    frameStart := 0 },
  { event := event24780
    frameStart := 0 },
  { event := event24781
    frameStart := 0 },
  { event := event24782
    frameStart := 0 },
  { event := event24783
    frameStart := 0 }
]

def eventLeaf1549 : Array AnnotatedEvent := #[
  { event := event24784
    frameStart := 0 },
  { event := event24785
    frameStart := 0 },
  { event := event24786
    frameStart := 0 },
  { event := event24787
    frameStart := 0 },
  { event := event24788
    frameStart := 0 },
  { event := event24789
    frameStart := 0 },
  { event := event24790
    frameStart := 0 },
  { event := event24791
    frameStart := 0 },
  { event := event24792
    frameStart := 0 },
  { event := event24793
    frameStart := 0 },
  { event := event24794
    frameStart := 0 },
  { event := event24795
    frameStart := 0 },
  { event := event24796
    frameStart := 0 },
  { event := event24797
    frameStart := 0 },
  { event := event24798
    frameStart := 0 },
  { event := event24799
    frameStart := 0 }
]

def eventLeaf1550 : Array AnnotatedEvent := #[
  { event := event24800
    frameStart := 0 },
  { event := event24801
    frameStart := 0 },
  { event := event24802
    frameStart := 0 },
  { event := event24803
    frameStart := 0 },
  { event := event24804
    frameStart := 0 },
  { event := event24805
    frameStart := 0 },
  { event := event24806
    frameStart := 0 },
  { event := event24807
    frameStart := 0 },
  { event := event24808
    frameStart := 0 },
  { event := event24809
    frameStart := 0 },
  { event := event24810
    frameStart := 0 },
  { event := event24811
    frameStart := 0 },
  { event := event24812
    frameStart := 0 },
  { event := event24813
    frameStart := 0 },
  { event := event24814
    frameStart := 0 },
  { event := event24815
    frameStart := 0 }
]

def eventLeaf1551 : Array AnnotatedEvent := #[
  { event := event24816
    frameStart := 0 },
  { event := event24817
    frameStart := 0 },
  { event := event24818
    frameStart := 0 },
  { event := event24819
    frameStart := 0 },
  { event := event24820
    frameStart := 0 },
  { event := event24821
    frameStart := 0 },
  { event := event24822
    frameStart := 0 },
  { event := event24823
    frameStart := 0 },
  { event := event24824
    frameStart := 0 },
  { event := event24825
    frameStart := 0 },
  { event := event24826
    frameStart := 0 },
  { event := event24827
    frameStart := 0 },
  { event := event24828
    frameStart := 0 },
  { event := event24829
    frameStart := 0 },
  { event := event24830
    frameStart := 0 },
  { event := event24831
    frameStart := 0 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events096
