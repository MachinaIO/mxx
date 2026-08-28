import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events303

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event77568 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16055⟩⟩) (.authority (.programFamilyFact))

def exact77569RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16055⟩⟩], []⟩, (1)⟩]

theorem exact77569RawTermsValid :
    exact77569RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77569 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16055⟩⟩) exact77569RawTerms (.finite 22) 77568 .exactZero (none)

def event77570 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16056⟩⟩) 0 ⟨16055⟩ 77569

def event77571 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16056⟩⟩) (.identity (.predecessor 0 77570 .coefficient))

def event77572 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨16056⟩⟩) (.finite 22)

def event77573 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21468⟩⟩) 0 ⟨16056⟩ 77572

def event77574 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21468⟩⟩) (.authority (.relationPreimageSource ⟨45⟩))

def exact77575RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨21468⟩⟩]⟩, (1)⟩]

theorem exact77575RawTermsValid :
    exact77575RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77575 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21468⟩⟩) exact77575RawTerms (.finite 136065468) 77574 .exactZero (none)

def event77576 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6⟩⟩) (.authority (.operator))

def exact77577RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩]⟩, (1)⟩]

theorem exact77577RawTermsValid :
    exact77577RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77577 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6⟩⟩) exact77577RawTerms .large 77576 .exactZero (none)

def event77578 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21469⟩⟩) 0 ⟨6⟩ 77577

def event77579 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21469⟩⟩) 1 ⟨21468⟩ 77575

def event77580 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21469⟩⟩) (.product (.predecessor 0 77578 .coefficient) (.predecessor 1 77579 .coefficient) (⟨false, false, none, none, none⟩))

def event77581 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21469⟩⟩, .operator (⟨77577, 0⟩, ⟨77575, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21468⟩⟩]⟩, (1)⟩)

def exact77582RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21468⟩⟩]⟩, (1)⟩]

theorem exact77582RawTermsValid :
    exact77582RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77582 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21469⟩⟩) exact77582RawTerms .large 77580 .exactZero (none)

def event77583 : Event := .preFoldPolynomial 77582 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21468⟩⟩]⟩, (1)⟩] .exactZero none

def exact77584RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21468⟩⟩]⟩, (1)⟩]

def event77584 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨21469⟩⟩) 77583 exact77584RawTerms .large 77580 .exactZero (none)

def event77585 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨28069⟩⟩)

def event77586 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event77587 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event77588 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨360⟩⟩) (.authority (.operator))

def event77589 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨360⟩⟩) (.finite 2)

def event77590 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event77591 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event77592 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event77593 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event77594 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 77593

def event77595 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 77591

def event77596 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 77594 .coefficient) (.value (.predecessor 1 77595 .coefficient)))

def event77597 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event77598 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5504⟩⟩) 0 ⟨5503⟩ 77597

def event77599 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5504⟩⟩) 1 ⟨360⟩ 77589

def event77600 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5504⟩⟩) (.sum [.predecessor 0 77598 .coefficient, .predecessor 1 77599 .coefficient])

def event77601 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5504⟩⟩) (.finite 219)

def event77602 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5530⟩⟩) 0 ⟨5504⟩ 77601

def event77603 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5530⟩⟩) 1 ⟨961⟩ 77587

def event77604 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5530⟩⟩) (.identity (.predecessor 1 77603 .coefficient))

def event77605 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5530⟩⟩) (.finite 224)

def event77606 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11549⟩⟩) 0 ⟨5530⟩ 77605

def event77607 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11549⟩⟩) (.authority (.programFamilyFact))

def exact77608RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11549⟩⟩], []⟩, (1)⟩]

theorem exact77608RawTermsValid :
    exact77608RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77608 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11549⟩⟩) exact77608RawTerms (.finite 22) 77607 .exactZero (none)

def event77609 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14415⟩⟩) 0 ⟨5530⟩ 77605

def event77610 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14415⟩⟩) (.authority (.programFamilyFact))

def exact77611RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨14415⟩⟩], []⟩, (1)⟩]

theorem exact77611RawTermsValid :
    exact77611RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77611 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14415⟩⟩) exact77611RawTerms (.finite 22) 77610 .exactZero (none)

def event77612 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14416⟩⟩) 0 ⟨14415⟩ 77611

def event77613 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14416⟩⟩) 1 ⟨11549⟩ 77608

def event77614 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14416⟩⟩) (.product (.predecessor 0 77612 .coefficient) (.predecessor 1 77613 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event77615 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨14416⟩⟩, .operator (⟨77611, 0⟩, ⟨77608, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨11549⟩⟩, ⟨.program ⟨214⟩, ⟨14415⟩⟩], []⟩, (1)⟩)

def exact77616RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11549⟩⟩, ⟨.program ⟨214⟩, ⟨14415⟩⟩], []⟩, (1)⟩]

theorem exact77616RawTermsValid :
    exact77616RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77616 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14416⟩⟩) exact77616RawTerms (.finite 484) 77614 .exactZero (none)

def event77617 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14417⟩⟩) 0 ⟨14416⟩ 77616

def event77618 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14417⟩⟩) (.identity (.predecessor 0 77617 .coefficient))

def event77619 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨14417⟩⟩) (.finite 484)

def event77620 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16055⟩⟩) 0 ⟨14417⟩ 77619

def event77621 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16055⟩⟩) (.authority (.programFamilyFact))

def exact77622RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16055⟩⟩], []⟩, (1)⟩]

theorem exact77622RawTermsValid :
    exact77622RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77622 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16055⟩⟩) exact77622RawTerms (.finite 22) 77621 .exactZero (none)

def event77623 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16056⟩⟩) 0 ⟨16055⟩ 77622

def event77624 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16056⟩⟩) (.identity (.predecessor 0 77623 .coefficient))

def event77625 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨16056⟩⟩) (.finite 22)

def event77626 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24220⟩⟩) 0 ⟨16056⟩ 77625

def event77627 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24220⟩⟩) (.authority (.programFamilyFact))

def event77628 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨24220⟩⟩) (.finite 3720)

def event77629 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨6689⟩⟩) .missing

def event77630 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24221⟩⟩) 0 ⟨6689⟩ 77629

def event77631 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24221⟩⟩) 1 ⟨24220⟩ 77628

def event77632 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24221⟩⟩) (.authority (.operator))

def exact77633RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨24221⟩⟩]⟩, (1)⟩]

theorem exact77633RawTermsValid :
    exact77633RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77633 : Event := .resultExact (⟨.program ⟨214⟩, ⟨24221⟩⟩) exact77633RawTerms .large 77632 .exactZero (none)

def event77634 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28063⟩⟩) 0 ⟨24221⟩ 77633

def event77635 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28063⟩⟩) (.authority (.operator))

def exact77636RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨28063⟩⟩]⟩, (1)⟩]

theorem exact77636RawTermsValid :
    exact77636RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77636 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28063⟩⟩) exact77636RawTerms (.finite 8192) 77635 .exactZero (none)

def event77637 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨110⟩⟩) (.authority (.operator))

def event77638 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨110⟩⟩) .exactZero

def event77639 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16130⟩⟩) 0 ⟨16056⟩ 77625

def event77640 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16130⟩⟩) 1 ⟨110⟩ 77638

def event77641 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16130⟩⟩) (.sum [.predecessor 0 77639 .coefficient, .predecessor 1 77640 .coefficient])

def event77642 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨16130⟩⟩) (.finite 22)

def event77643 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16131⟩⟩) 0 ⟨16130⟩ 77642

def event77644 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16131⟩⟩) (.identity (.predecessor 0 77643 .coefficient))

def exact77645RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16055⟩⟩], []⟩, (1)⟩]

theorem exact77645RawTermsValid :
    exact77645RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77645 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16131⟩⟩) exact77645RawTerms (.finite 22) 77644 .exactZero (none)

def event77646 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6544⟩⟩) (.authority (.factStore))

def exact77647RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact77647RawTermsValid :
    exact77647RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77647 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6544⟩⟩) exact77647RawTerms .large 77646 .exactZero (none)

def event77648 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16132⟩⟩) 0 ⟨6544⟩ 77647

def event77649 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16132⟩⟩) 1 ⟨16131⟩ 77645

def event77650 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16132⟩⟩) (.product (.predecessor 0 77648 .coefficient) (.predecessor 1 77649 .coefficient) (⟨false, false, none, none, none⟩))

def event77651 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨16132⟩⟩, .operator (⟨77647, 0⟩, ⟨77645, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨16055⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact77652RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16055⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact77652RawTermsValid :
    exact77652RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77652 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16132⟩⟩) exact77652RawTerms .large 77650 .exactZero (none)

def event77653 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6698⟩⟩) 0 ⟨6689⟩ 77629

def event77654 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6698⟩⟩) (.authority (.operator))

def exact77655RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6698⟩⟩]⟩, (1)⟩]

theorem exact77655RawTermsValid :
    exact77655RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77655 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6698⟩⟩) exact77655RawTerms .large 77654 .exactZero (none)

def event77656 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16133⟩⟩) 0 ⟨6698⟩ 77655

def event77657 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16133⟩⟩) 1 ⟨16132⟩ 77652

def event77658 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16133⟩⟩) (.sum [.predecessor 0 77656 .coefficient, .predecessor 1 77657 .coefficient])

def exact77659RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6698⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16055⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact77659RawTermsValid :
    exact77659RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77659 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16133⟩⟩) exact77659RawTerms .large 77658 .exactZero (none)

def event77660 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28064⟩⟩) 0 ⟨16133⟩ 77659

def event77661 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28064⟩⟩) 1 ⟨28063⟩ 77636

def event77662 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28064⟩⟩) (.product (.predecessor 0 77660 .coefficient) (.predecessor 1 77661 .coefficient) (⟨false, false, none, none, none⟩))

def event77663 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28064⟩⟩, .operator (⟨77659, 0⟩, ⟨77636, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6698⟩⟩, ⟨.program ⟨214⟩, ⟨28063⟩⟩]⟩, (1)⟩)

def event77664 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28064⟩⟩, .operator (⟨77659, 1⟩, ⟨77636, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨16055⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28063⟩⟩]⟩, (-1)⟩)

def event77665 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨28064⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨16055⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28063⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨28063⟩⟩) ⟨24221⟩ 77633)

def event77666 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28064⟩⟩, .relation 77665 0, ⟨[⟨.program ⟨214⟩, ⟨16055⟩⟩], [⟨.program ⟨214⟩, ⟨24221⟩⟩]⟩, (-1)⟩)

def exact77667RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6698⟩⟩, ⟨.program ⟨214⟩, ⟨28063⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16055⟩⟩], [⟨.program ⟨214⟩, ⟨24221⟩⟩]⟩, (-1)⟩]

theorem exact77667RawTermsValid :
    exact77667RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77667 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28064⟩⟩) exact77667RawTerms .large 77662 .exactZero (none)

def event77668 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18028⟩⟩) 0 ⟨16056⟩ 77625

def event77669 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18028⟩⟩) (.authority (.programFamilyFact))

def exact77670RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨18028⟩⟩], []⟩, (1)⟩]

theorem exact77670RawTermsValid :
    exact77670RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77670 : Event := .resultExact (⟨.program ⟨214⟩, ⟨18028⟩⟩) exact77670RawTerms (.finite 22) 77669 .exactZero (none)

def event77671 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18033⟩⟩) 0 ⟨6544⟩ 77647

def event77672 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18033⟩⟩) 1 ⟨18028⟩ 77670

def event77673 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18033⟩⟩) (.product (.predecessor 0 77671 .coefficient) (.predecessor 1 77672 .coefficient) (⟨false, true, none, none, some 1⟩))

def event77674 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18033⟩⟩, .operator (⟨77647, 0⟩, ⟨77670, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨18028⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact77675RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨18028⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact77675RawTermsValid :
    exact77675RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77675 : Event := .resultExact (⟨.program ⟨214⟩, ⟨18033⟩⟩) exact77675RawTerms .large 77673 .exactZero (none)

def event77676 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6724⟩⟩) 0 ⟨6689⟩ 77629

def event77677 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6724⟩⟩) (.authority (.operator))

def exact77678RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6724⟩⟩]⟩, (1)⟩]

theorem exact77678RawTermsValid :
    exact77678RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77678 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6724⟩⟩) exact77678RawTerms .large 77677 .exactZero (none)

def event77679 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18034⟩⟩) 0 ⟨6724⟩ 77678

def event77680 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18034⟩⟩) 1 ⟨18033⟩ 77675

def event77681 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18034⟩⟩) (.sum [.predecessor 0 77679 .coefficient, .predecessor 1 77680 .coefficient])

def exact77682RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6724⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨18028⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact77682RawTermsValid :
    exact77682RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77682 : Event := .resultExact (⟨.program ⟨214⟩, ⟨18034⟩⟩) exact77682RawTerms .large 77681 .exactZero (none)

def event77683 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28069⟩⟩) 0 ⟨18034⟩ 77682

def event77684 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28069⟩⟩) 1 ⟨28064⟩ 77667

def event77685 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28069⟩⟩) (.sum [.predecessor 0 77683 .coefficient, .predecessor 1 77684 .coefficient])

def exact77686RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6698⟩⟩, ⟨.program ⟨214⟩, ⟨28063⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6724⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16055⟩⟩], [⟨.program ⟨214⟩, ⟨24221⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨18028⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact77686RawTermsValid :
    exact77686RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77686 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28069⟩⟩) exact77686RawTerms .large 77685 .exactZero (none)

def event77687 : Event := .preFoldPolynomial 77686 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6698⟩⟩, ⟨.program ⟨214⟩, ⟨28063⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6724⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16055⟩⟩], [⟨.program ⟨214⟩, ⟨24221⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨18028⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩] .exactZero none

def exact77688RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6698⟩⟩, ⟨.program ⟨214⟩, ⟨28063⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6724⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16055⟩⟩], [⟨.program ⟨214⟩, ⟨24221⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨18028⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

def event77688 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨28069⟩⟩) 77687 exact77688RawTerms .large 77685 .exactZero (none)

def event77689 : Event := .specializationComputed (⟨.program ⟨214⟩, ⟨16056⟩⟩) ⟨⟨137⟩, ⟨45⟩, ⟨109⟩⟩ ⟨77531, 77689⟩

def event77690 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨21471⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21468⟩⟩]⟩) (1) 0 2 (.universal 77689 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21468⟩⟩]⟩) (none) 77688)

def event77691 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21471⟩⟩, .relation 77690 1, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6724⟩⟩]⟩, (1)⟩)

def event77692 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21471⟩⟩, .relation 77690 0, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6698⟩⟩, ⟨.program ⟨214⟩, ⟨28063⟩⟩]⟩, (-1)⟩)

def event77693 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21471⟩⟩, .relation 77690 2, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16055⟩⟩], [⟨.program ⟨214⟩, ⟨24221⟩⟩]⟩, (1)⟩)

def event77694 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21471⟩⟩, .relation 77690 3, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨18028⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact77695RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6698⟩⟩, ⟨.program ⟨214⟩, ⟨28063⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6724⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16055⟩⟩], [⟨.program ⟨214⟩, ⟨24221⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨18028⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact77695RawTermsValid :
    exact77695RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77695 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21471⟩⟩) exact77695RawTerms .large 77527 (.finite 1811303510016) (some (77529))

def event77696 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28066⟩⟩) 0 ⟨21471⟩ 77695

def event77697 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28066⟩⟩) 1 ⟨28065⟩ 77517

def event77698 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28066⟩⟩) (.sum [.predecessor 0 77696 .coefficient, .predecessor 1 77697 .coefficient])

def event77699 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28066⟩⟩, .operator (⟨77695, 0⟩, ⟨77517, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6698⟩⟩, ⟨.program ⟨214⟩, ⟨28063⟩⟩]⟩, (1)⟩)

def event77700 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28066⟩⟩, .operator (⟨77695, 2⟩, ⟨77517, 1⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16055⟩⟩], [⟨.program ⟨214⟩, ⟨24221⟩⟩]⟩, (-1)⟩)

def event77701 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28066⟩⟩) (.sum [.result 77695 .summary, .result 77517 .summary])

def exact77702RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6724⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨18028⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact77702RawTermsValid :
    exact77702RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77702 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28066⟩⟩) exact77702RawTerms .large 77698 (.finite 1292113298829627502592) (some (77701))

def event77703 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28067⟩⟩) 0 ⟨28066⟩ 77702

def event77704 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28067⟩⟩) 1 ⟨6638⟩ 5699

def event77705 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28067⟩⟩) (.product (.predecessor 0 77703 .coefficient) (.predecessor 1 77704 .coefficient) (⟨false, false, none, none, none⟩))

def event77706 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28067⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨6637⟩⟩]⟩) [⟨.result 5695 .coefficient, false, none⟩])

def event77707 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28067⟩⟩) (.product (.result 77702 .summary) (.transfer 77706) (⟨false, false, none, none, none⟩))

def event77708 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28067⟩⟩, .operator (⟨77702, 0⟩, ⟨5699, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6724⟩⟩, ⟨.program ⟨214⟩, ⟨6637⟩⟩]⟩, (1)⟩)

def event77709 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28067⟩⟩, .operator (⟨77702, 1⟩, ⟨5699, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨18028⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨6637⟩⟩]⟩, (-1)⟩)

def event77710 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨28067⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨18028⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨6637⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨6637⟩⟩) ⟨6590⟩ 5692)

def event77711 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28067⟩⟩, .relation 77710 0, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨6383⟩⟩, ⟨.program ⟨214⟩, ⟨18028⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact77712RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6724⟩⟩, ⟨.program ⟨214⟩, ⟨6637⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨6383⟩⟩, ⟨.program ⟨214⟩, ⟨18028⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact77712RawTermsValid :
    exact77712RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77712 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28067⟩⟩) exact77712RawTerms .large 77705 (.finite 4742076480517514208552681472) (some (77707))

def event77713 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24158⟩⟩) 0 ⟨6689⟩ 5477

def event77714 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24158⟩⟩) 1 ⟨24157⟩ 70109

def event77715 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24158⟩⟩) (.authority (.operator))

def exact77716RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨24158⟩⟩]⟩, (1)⟩]

theorem exact77716RawTermsValid :
    exact77716RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77716 : Event := .resultExact (⟨.program ⟨214⟩, ⟨24158⟩⟩) exact77716RawTerms .large 77715 .exactZero (none)

def event77717 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27846⟩⟩) 0 ⟨24158⟩ 77716

def event77718 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27846⟩⟩) (.authority (.operator))

def exact77719RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨27846⟩⟩]⟩, (1)⟩]

theorem exact77719RawTermsValid :
    exact77719RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77719 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27846⟩⟩) exact77719RawTerms (.finite 8192) 77718 .exactZero (none)

def event77720 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27848⟩⟩) 0 ⟨26063⟩ 70393

def event77721 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27848⟩⟩) 1 ⟨27846⟩ 77719

def event77722 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27848⟩⟩) (.product (.predecessor 0 77720 .coefficient) (.predecessor 1 77721 .coefficient) (⟨false, false, none, none, none⟩))

def event77723 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27848⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨27846⟩⟩]⟩) [⟨.result 77719 .coefficient, false, none⟩])

def event77724 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27848⟩⟩) (.product (.result 70393 .summary) (.transfer 77723) (⟨false, false, none, none, none⟩))

def event77725 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27848⟩⟩, .operator (⟨70393, 0⟩, ⟨77719, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6697⟩⟩, ⟨.program ⟨214⟩, ⟨27846⟩⟩]⟩, (1)⟩)

def event77726 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27848⟩⟩, .operator (⟨70393, 1⟩, ⟨77719, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15936⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27846⟩⟩]⟩, (-1)⟩)

def event77727 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨27848⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15936⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27846⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨27846⟩⟩) ⟨24158⟩ 77716)

def event77728 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27848⟩⟩, .relation 77727 0, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15936⟩⟩], [⟨.program ⟨214⟩, ⟨24158⟩⟩]⟩, (-1)⟩)

def exact77729RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6697⟩⟩, ⟨.program ⟨214⟩, ⟨27846⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15936⟩⟩], [⟨.program ⟨214⟩, ⟨24158⟩⟩]⟩, (-1)⟩]

theorem exact77729RawTermsValid :
    exact77729RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77729 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27848⟩⟩) exact77729RawTerms .large 77722 (.finite 1292068472128282820608) (some (77724))

def event77730 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21324⟩⟩) 0 ⟨15937⟩ 3333

def event77731 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21324⟩⟩) (.authority (.relationPreimageSource ⟨42⟩))

def exact77732RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨21324⟩⟩]⟩, (1)⟩]

theorem exact77732RawTermsValid :
    exact77732RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77732 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21324⟩⟩) exact77732RawTerms (.finite 136065468) 77731 .exactZero (none)

def event77733 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21326⟩⟩) 0 ⟨21324⟩ 77732

def event77734 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21326⟩⟩) 1 ⟨2348⟩ 4

def event77735 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21326⟩⟩) (.scale (.predecessor 0 77733 .coefficient) (.value (.predecessor 1 77734 .coefficient)))

def exact77736RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨21324⟩⟩]⟩, (1)⟩]

theorem exact77736RawTermsValid :
    exact77736RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77736 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21326⟩⟩) exact77736RawTerms (.finite 136065468) 77735 .exactZero (none)

def event77737 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21327⟩⟩) 0 ⟨5535⟩ 65387

def event77738 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21327⟩⟩) 1 ⟨21326⟩ 77736

def event77739 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21327⟩⟩) (.product (.predecessor 0 77737 .coefficient) (.predecessor 1 77738 .coefficient) (⟨false, false, none, none, none⟩))

def event77740 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21327⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨21324⟩⟩]⟩) [⟨.result 77732 .coefficient, false, none⟩])

def event77741 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21327⟩⟩) (.product (.result 65387 .summary) (.transfer 77740) (⟨false, false, none, none, none⟩))

def event77742 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21327⟩⟩, .operator (⟨65387, 0⟩, ⟨77736, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21324⟩⟩]⟩, (1)⟩)

def event77743 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨21325⟩⟩)

def event77744 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event77745 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event77746 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨360⟩⟩) (.authority (.operator))

def event77747 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨360⟩⟩) (.finite 2)

def event77748 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event77749 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event77750 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event77751 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event77752 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 77751

def event77753 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 77749

def event77754 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 77752 .coefficient) (.value (.predecessor 1 77753 .coefficient)))

def event77755 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event77756 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5504⟩⟩) 0 ⟨5503⟩ 77755

def event77757 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5504⟩⟩) 1 ⟨360⟩ 77747

def event77758 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5504⟩⟩) (.sum [.predecessor 0 77756 .coefficient, .predecessor 1 77757 .coefficient])

def event77759 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5504⟩⟩) (.finite 219)

def event77760 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5530⟩⟩) 0 ⟨5504⟩ 77759

def event77761 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5530⟩⟩) 1 ⟨961⟩ 77745

def event77762 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5530⟩⟩) (.identity (.predecessor 1 77761 .coefficient))

def event77763 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5530⟩⟩) (.finite 224)

def event77764 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11465⟩⟩) 0 ⟨5530⟩ 77763

def event77765 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11465⟩⟩) (.authority (.programFamilyFact))

def exact77766RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11465⟩⟩], []⟩, (1)⟩]

theorem exact77766RawTermsValid :
    exact77766RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77766 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11465⟩⟩) exact77766RawTerms (.finite 18) 77765 .exactZero (none)

def event77767 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14198⟩⟩) 0 ⟨5530⟩ 77763

def event77768 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14198⟩⟩) (.authority (.programFamilyFact))

def exact77769RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨14198⟩⟩], []⟩, (1)⟩]

theorem exact77769RawTermsValid :
    exact77769RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77769 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14198⟩⟩) exact77769RawTerms (.finite 18) 77768 .exactZero (none)

def event77770 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14199⟩⟩) 0 ⟨14198⟩ 77769

def event77771 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14199⟩⟩) 1 ⟨11465⟩ 77766

def event77772 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14199⟩⟩) (.product (.predecessor 0 77770 .coefficient) (.predecessor 1 77771 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event77773 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14199⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨11465⟩⟩, ⟨.program ⟨214⟩, ⟨14198⟩⟩], []⟩) [⟨.result 77769 .coefficient, true, some 1⟩, ⟨.result 77766 .coefficient, true, some 1⟩])

def event77774 : Event := .survivorFold (1) 77773

def exact77775RawTerms : List Term := []

theorem exact77775RawTermsValid :
    exact77775RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77775 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14199⟩⟩) exact77775RawTerms (.finite 324) 77772 (.finite 324) (some (77773))

def event77776 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14200⟩⟩) 0 ⟨14199⟩ 77775

def event77777 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14200⟩⟩) (.identity (.predecessor 0 77776 .coefficient))

def event77778 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨14200⟩⟩) (.finite 324)

def event77779 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15936⟩⟩) 0 ⟨14200⟩ 77778

def event77780 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15936⟩⟩) (.authority (.programFamilyFact))

def exact77781RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15936⟩⟩], []⟩, (1)⟩]

theorem exact77781RawTermsValid :
    exact77781RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77781 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15936⟩⟩) exact77781RawTerms (.finite 18) 77780 .exactZero (none)

def event77782 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15937⟩⟩) 0 ⟨15936⟩ 77781

def event77783 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15937⟩⟩) (.identity (.predecessor 0 77782 .coefficient))

def event77784 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨15937⟩⟩) (.finite 18)

def event77785 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21324⟩⟩) 0 ⟨15937⟩ 77784

def event77786 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21324⟩⟩) (.authority (.relationPreimageSource ⟨42⟩))

def exact77787RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨21324⟩⟩]⟩, (1)⟩]

theorem exact77787RawTermsValid :
    exact77787RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77787 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21324⟩⟩) exact77787RawTerms (.finite 136065468) 77786 .exactZero (none)

def event77788 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6⟩⟩) (.authority (.operator))

def exact77789RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩]⟩, (1)⟩]

theorem exact77789RawTermsValid :
    exact77789RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77789 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6⟩⟩) exact77789RawTerms .large 77788 .exactZero (none)

def event77790 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21325⟩⟩) 0 ⟨6⟩ 77789

def event77791 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21325⟩⟩) 1 ⟨21324⟩ 77787

def event77792 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21325⟩⟩) (.product (.predecessor 0 77790 .coefficient) (.predecessor 1 77791 .coefficient) (⟨false, false, none, none, none⟩))

def event77793 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21325⟩⟩, .operator (⟨77789, 0⟩, ⟨77787, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21324⟩⟩]⟩, (1)⟩)

def exact77794RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21324⟩⟩]⟩, (1)⟩]

theorem exact77794RawTermsValid :
    exact77794RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77794 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21325⟩⟩) exact77794RawTerms .large 77792 .exactZero (none)

def event77795 : Event := .preFoldPolynomial 77794 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21324⟩⟩]⟩, (1)⟩] .exactZero none

def exact77796RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21324⟩⟩]⟩, (1)⟩]

def event77796 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨21325⟩⟩) 77795 exact77796RawTerms .large 77792 .exactZero (none)

def event77797 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨27852⟩⟩)

def event77798 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event77799 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event77800 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨360⟩⟩) (.authority (.operator))

def event77801 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨360⟩⟩) (.finite 2)

def event77802 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event77803 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event77804 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event77805 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event77806 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 77805

def event77807 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 77803

def event77808 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 77806 .coefficient) (.value (.predecessor 1 77807 .coefficient)))

def event77809 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event77810 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5504⟩⟩) 0 ⟨5503⟩ 77809

def event77811 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5504⟩⟩) 1 ⟨360⟩ 77801

def event77812 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5504⟩⟩) (.sum [.predecessor 0 77810 .coefficient, .predecessor 1 77811 .coefficient])

def event77813 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5504⟩⟩) (.finite 219)

def event77814 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5530⟩⟩) 0 ⟨5504⟩ 77813

def event77815 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5530⟩⟩) 1 ⟨961⟩ 77799

def event77816 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5530⟩⟩) (.identity (.predecessor 1 77815 .coefficient))

def event77817 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5530⟩⟩) (.finite 224)

def event77818 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11465⟩⟩) 0 ⟨5530⟩ 77817

def event77819 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11465⟩⟩) (.authority (.programFamilyFact))

def exact77820RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11465⟩⟩], []⟩, (1)⟩]

theorem exact77820RawTermsValid :
    exact77820RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77820 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11465⟩⟩) exact77820RawTerms (.finite 18) 77819 .exactZero (none)

def event77821 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14198⟩⟩) 0 ⟨5530⟩ 77817

def event77822 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14198⟩⟩) (.authority (.programFamilyFact))

def exact77823RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨14198⟩⟩], []⟩, (1)⟩]

theorem exact77823RawTermsValid :
    exact77823RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77823 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14198⟩⟩) exact77823RawTerms (.finite 18) 77822 .exactZero (none)

def eventLeaf4848 : Array AnnotatedEvent := #[
  { event := event77568
    frameStart := 77531 },
  { event := event77569
    frameStart := 77531 },
  { event := event77570
    frameStart := 77531 },
  { event := event77571
    frameStart := 77531 },
  { event := event77572
    frameStart := 77531 },
  { event := event77573
    frameStart := 77531 },
  { event := event77574
    frameStart := 77531 },
  { event := event77575
    frameStart := 77531 },
  { event := event77576
    frameStart := 77531 },
  { event := event77577
    frameStart := 77531 },
  { event := event77578
    frameStart := 77531 },
  { event := event77579
    frameStart := 77531 },
  { event := event77580
    frameStart := 77531 },
  { event := event77581
    frameStart := 77531 },
  { event := event77582
    frameStart := 77531 },
  { event := event77583
    frameStart := 77531 }
]

def eventLeaf4849 : Array AnnotatedEvent := #[
  { event := event77584
    frameStart := 77531 },
  { event := event77585
    frameStart := 77585 },
  { event := event77586
    frameStart := 77585 },
  { event := event77587
    frameStart := 77585 },
  { event := event77588
    frameStart := 77585 },
  { event := event77589
    frameStart := 77585 },
  { event := event77590
    frameStart := 77585 },
  { event := event77591
    frameStart := 77585 },
  { event := event77592
    frameStart := 77585 },
  { event := event77593
    frameStart := 77585 },
  { event := event77594
    frameStart := 77585 },
  { event := event77595
    frameStart := 77585 },
  { event := event77596
    frameStart := 77585 },
  { event := event77597
    frameStart := 77585 },
  { event := event77598
    frameStart := 77585 },
  { event := event77599
    frameStart := 77585 }
]

def eventLeaf4850 : Array AnnotatedEvent := #[
  { event := event77600
    frameStart := 77585 },
  { event := event77601
    frameStart := 77585 },
  { event := event77602
    frameStart := 77585 },
  { event := event77603
    frameStart := 77585 },
  { event := event77604
    frameStart := 77585 },
  { event := event77605
    frameStart := 77585 },
  { event := event77606
    frameStart := 77585 },
  { event := event77607
    frameStart := 77585 },
  { event := event77608
    frameStart := 77585 },
  { event := event77609
    frameStart := 77585 },
  { event := event77610
    frameStart := 77585 },
  { event := event77611
    frameStart := 77585 },
  { event := event77612
    frameStart := 77585 },
  { event := event77613
    frameStart := 77585 },
  { event := event77614
    frameStart := 77585 },
  { event := event77615
    frameStart := 77585 }
]

def eventLeaf4851 : Array AnnotatedEvent := #[
  { event := event77616
    frameStart := 77585 },
  { event := event77617
    frameStart := 77585 },
  { event := event77618
    frameStart := 77585 },
  { event := event77619
    frameStart := 77585 },
  { event := event77620
    frameStart := 77585 },
  { event := event77621
    frameStart := 77585 },
  { event := event77622
    frameStart := 77585 },
  { event := event77623
    frameStart := 77585 },
  { event := event77624
    frameStart := 77585 },
  { event := event77625
    frameStart := 77585 },
  { event := event77626
    frameStart := 77585 },
  { event := event77627
    frameStart := 77585 },
  { event := event77628
    frameStart := 77585 },
  { event := event77629
    frameStart := 77585 },
  { event := event77630
    frameStart := 77585 },
  { event := event77631
    frameStart := 77585 }
]

def eventLeaf4852 : Array AnnotatedEvent := #[
  { event := event77632
    frameStart := 77585 },
  { event := event77633
    frameStart := 77585 },
  { event := event77634
    frameStart := 77585 },
  { event := event77635
    frameStart := 77585 },
  { event := event77636
    frameStart := 77585 },
  { event := event77637
    frameStart := 77585 },
  { event := event77638
    frameStart := 77585 },
  { event := event77639
    frameStart := 77585 },
  { event := event77640
    frameStart := 77585 },
  { event := event77641
    frameStart := 77585 },
  { event := event77642
    frameStart := 77585 },
  { event := event77643
    frameStart := 77585 },
  { event := event77644
    frameStart := 77585 },
  { event := event77645
    frameStart := 77585 },
  { event := event77646
    frameStart := 77585 },
  { event := event77647
    frameStart := 77585 }
]

def eventLeaf4853 : Array AnnotatedEvent := #[
  { event := event77648
    frameStart := 77585 },
  { event := event77649
    frameStart := 77585 },
  { event := event77650
    frameStart := 77585 },
  { event := event77651
    frameStart := 77585 },
  { event := event77652
    frameStart := 77585 },
  { event := event77653
    frameStart := 77585 },
  { event := event77654
    frameStart := 77585 },
  { event := event77655
    frameStart := 77585 },
  { event := event77656
    frameStart := 77585 },
  { event := event77657
    frameStart := 77585 },
  { event := event77658
    frameStart := 77585 },
  { event := event77659
    frameStart := 77585 },
  { event := event77660
    frameStart := 77585 },
  { event := event77661
    frameStart := 77585 },
  { event := event77662
    frameStart := 77585 },
  { event := event77663
    frameStart := 77585 }
]

def eventLeaf4854 : Array AnnotatedEvent := #[
  { event := event77664
    frameStart := 77585 },
  { event := event77665
    frameStart := 77585 },
  { event := event77666
    frameStart := 77585 },
  { event := event77667
    frameStart := 77585 },
  { event := event77668
    frameStart := 77585 },
  { event := event77669
    frameStart := 77585 },
  { event := event77670
    frameStart := 77585 },
  { event := event77671
    frameStart := 77585 },
  { event := event77672
    frameStart := 77585 },
  { event := event77673
    frameStart := 77585 },
  { event := event77674
    frameStart := 77585 },
  { event := event77675
    frameStart := 77585 },
  { event := event77676
    frameStart := 77585 },
  { event := event77677
    frameStart := 77585 },
  { event := event77678
    frameStart := 77585 },
  { event := event77679
    frameStart := 77585 }
]

def eventLeaf4855 : Array AnnotatedEvent := #[
  { event := event77680
    frameStart := 77585 },
  { event := event77681
    frameStart := 77585 },
  { event := event77682
    frameStart := 77585 },
  { event := event77683
    frameStart := 77585 },
  { event := event77684
    frameStart := 77585 },
  { event := event77685
    frameStart := 77585 },
  { event := event77686
    frameStart := 77585 },
  { event := event77687
    frameStart := 77585 },
  { event := event77688
    frameStart := 77585 },
  { event := event77689
    frameStart := 0 },
  { event := event77690
    frameStart := 0 },
  { event := event77691
    frameStart := 0 },
  { event := event77692
    frameStart := 0 },
  { event := event77693
    frameStart := 0 },
  { event := event77694
    frameStart := 0 },
  { event := event77695
    frameStart := 0 }
]

def eventLeaf4856 : Array AnnotatedEvent := #[
  { event := event77696
    frameStart := 0 },
  { event := event77697
    frameStart := 0 },
  { event := event77698
    frameStart := 0 },
  { event := event77699
    frameStart := 0 },
  { event := event77700
    frameStart := 0 },
  { event := event77701
    frameStart := 0 },
  { event := event77702
    frameStart := 0 },
  { event := event77703
    frameStart := 0 },
  { event := event77704
    frameStart := 0 },
  { event := event77705
    frameStart := 0 },
  { event := event77706
    frameStart := 0 },
  { event := event77707
    frameStart := 0 },
  { event := event77708
    frameStart := 0 },
  { event := event77709
    frameStart := 0 },
  { event := event77710
    frameStart := 0 },
  { event := event77711
    frameStart := 0 }
]

def eventLeaf4857 : Array AnnotatedEvent := #[
  { event := event77712
    frameStart := 0 },
  { event := event77713
    frameStart := 0 },
  { event := event77714
    frameStart := 0 },
  { event := event77715
    frameStart := 0 },
  { event := event77716
    frameStart := 0 },
  { event := event77717
    frameStart := 0 },
  { event := event77718
    frameStart := 0 },
  { event := event77719
    frameStart := 0 },
  { event := event77720
    frameStart := 0 },
  { event := event77721
    frameStart := 0 },
  { event := event77722
    frameStart := 0 },
  { event := event77723
    frameStart := 0 },
  { event := event77724
    frameStart := 0 },
  { event := event77725
    frameStart := 0 },
  { event := event77726
    frameStart := 0 },
  { event := event77727
    frameStart := 0 }
]

def eventLeaf4858 : Array AnnotatedEvent := #[
  { event := event77728
    frameStart := 0 },
  { event := event77729
    frameStart := 0 },
  { event := event77730
    frameStart := 0 },
  { event := event77731
    frameStart := 0 },
  { event := event77732
    frameStart := 0 },
  { event := event77733
    frameStart := 0 },
  { event := event77734
    frameStart := 0 },
  { event := event77735
    frameStart := 0 },
  { event := event77736
    frameStart := 0 },
  { event := event77737
    frameStart := 0 },
  { event := event77738
    frameStart := 0 },
  { event := event77739
    frameStart := 0 },
  { event := event77740
    frameStart := 0 },
  { event := event77741
    frameStart := 0 },
  { event := event77742
    frameStart := 0 },
  { event := event77743
    frameStart := 77743 }
]

def eventLeaf4859 : Array AnnotatedEvent := #[
  { event := event77744
    frameStart := 77743 },
  { event := event77745
    frameStart := 77743 },
  { event := event77746
    frameStart := 77743 },
  { event := event77747
    frameStart := 77743 },
  { event := event77748
    frameStart := 77743 },
  { event := event77749
    frameStart := 77743 },
  { event := event77750
    frameStart := 77743 },
  { event := event77751
    frameStart := 77743 },
  { event := event77752
    frameStart := 77743 },
  { event := event77753
    frameStart := 77743 },
  { event := event77754
    frameStart := 77743 },
  { event := event77755
    frameStart := 77743 },
  { event := event77756
    frameStart := 77743 },
  { event := event77757
    frameStart := 77743 },
  { event := event77758
    frameStart := 77743 },
  { event := event77759
    frameStart := 77743 }
]

def eventLeaf4860 : Array AnnotatedEvent := #[
  { event := event77760
    frameStart := 77743 },
  { event := event77761
    frameStart := 77743 },
  { event := event77762
    frameStart := 77743 },
  { event := event77763
    frameStart := 77743 },
  { event := event77764
    frameStart := 77743 },
  { event := event77765
    frameStart := 77743 },
  { event := event77766
    frameStart := 77743 },
  { event := event77767
    frameStart := 77743 },
  { event := event77768
    frameStart := 77743 },
  { event := event77769
    frameStart := 77743 },
  { event := event77770
    frameStart := 77743 },
  { event := event77771
    frameStart := 77743 },
  { event := event77772
    frameStart := 77743 },
  { event := event77773
    frameStart := 77743 },
  { event := event77774
    frameStart := 77743 },
  { event := event77775
    frameStart := 77743 }
]

def eventLeaf4861 : Array AnnotatedEvent := #[
  { event := event77776
    frameStart := 77743 },
  { event := event77777
    frameStart := 77743 },
  { event := event77778
    frameStart := 77743 },
  { event := event77779
    frameStart := 77743 },
  { event := event77780
    frameStart := 77743 },
  { event := event77781
    frameStart := 77743 },
  { event := event77782
    frameStart := 77743 },
  { event := event77783
    frameStart := 77743 },
  { event := event77784
    frameStart := 77743 },
  { event := event77785
    frameStart := 77743 },
  { event := event77786
    frameStart := 77743 },
  { event := event77787
    frameStart := 77743 },
  { event := event77788
    frameStart := 77743 },
  { event := event77789
    frameStart := 77743 },
  { event := event77790
    frameStart := 77743 },
  { event := event77791
    frameStart := 77743 }
]

def eventLeaf4862 : Array AnnotatedEvent := #[
  { event := event77792
    frameStart := 77743 },
  { event := event77793
    frameStart := 77743 },
  { event := event77794
    frameStart := 77743 },
  { event := event77795
    frameStart := 77743 },
  { event := event77796
    frameStart := 77743 },
  { event := event77797
    frameStart := 77797 },
  { event := event77798
    frameStart := 77797 },
  { event := event77799
    frameStart := 77797 },
  { event := event77800
    frameStart := 77797 },
  { event := event77801
    frameStart := 77797 },
  { event := event77802
    frameStart := 77797 },
  { event := event77803
    frameStart := 77797 },
  { event := event77804
    frameStart := 77797 },
  { event := event77805
    frameStart := 77797 },
  { event := event77806
    frameStart := 77797 },
  { event := event77807
    frameStart := 77797 }
]

def eventLeaf4863 : Array AnnotatedEvent := #[
  { event := event77808
    frameStart := 77797 },
  { event := event77809
    frameStart := 77797 },
  { event := event77810
    frameStart := 77797 },
  { event := event77811
    frameStart := 77797 },
  { event := event77812
    frameStart := 77797 },
  { event := event77813
    frameStart := 77797 },
  { event := event77814
    frameStart := 77797 },
  { event := event77815
    frameStart := 77797 },
  { event := event77816
    frameStart := 77797 },
  { event := event77817
    frameStart := 77797 },
  { event := event77818
    frameStart := 77797 },
  { event := event77819
    frameStart := 77797 },
  { event := event77820
    frameStart := 77797 },
  { event := event77821
    frameStart := 77797 },
  { event := event77822
    frameStart := 77797 },
  { event := event77823
    frameStart := 77797 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events303
