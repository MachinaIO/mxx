import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events108

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event27648 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15758⟩⟩) (.product (.predecessor 0 27646 .coefficient) (.predecessor 1 27647 .coefficient) (⟨false, true, none, none, some 1⟩))

def event27649 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨15758⟩⟩, .operator (⟨27622, 0⟩, ⟨27645, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨15757⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact27650RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15757⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact27650RawTermsValid :
    exact27650RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event27650 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15758⟩⟩) exact27650RawTerms .large 27648 .exactZero (none)

def event27651 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6719⟩⟩) 0 ⟨6689⟩ 27604

def event27652 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6719⟩⟩) (.authority (.operator))

def exact27653RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6719⟩⟩]⟩, (1)⟩]

theorem exact27653RawTermsValid :
    exact27653RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event27653 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6719⟩⟩) exact27653RawTerms .large 27652 .exactZero (none)

def event27654 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15759⟩⟩) 0 ⟨6719⟩ 27653

def event27655 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15759⟩⟩) 1 ⟨15758⟩ 27650

def event27656 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15759⟩⟩) (.sum [.predecessor 0 27654 .coefficient, .predecessor 1 27655 .coefficient])

def exact27657RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6719⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15757⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact27657RawTermsValid :
    exact27657RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event27657 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15759⟩⟩) exact27657RawTerms .large 27656 .exactZero (none)

def event27658 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27476⟩⟩) 0 ⟨15759⟩ 27657

def event27659 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27476⟩⟩) 1 ⟨27472⟩ 27642

def event27660 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27476⟩⟩) (.sum [.predecessor 0 27658 .coefficient, .predecessor 1 27659 .coefficient])

def exact27661RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6695⟩⟩, ⟨.program ⟨214⟩, ⟨27471⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6719⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15714⟩⟩], [⟨.program ⟨214⟩, ⟨24045⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15757⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact27661RawTermsValid :
    exact27661RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event27661 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27476⟩⟩) exact27661RawTerms .large 27660 .exactZero (none)

def event27662 : Event := .preFoldPolynomial 27661 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6695⟩⟩, ⟨.program ⟨214⟩, ⟨27471⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6719⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15714⟩⟩], [⟨.program ⟨214⟩, ⟨24045⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15757⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩] .exactZero none

def exact27663RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6695⟩⟩, ⟨.program ⟨214⟩, ⟨27471⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6719⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15714⟩⟩], [⟨.program ⟨214⟩, ⟨24045⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15757⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

def event27663 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨27476⟩⟩) 27662 exact27663RawTerms .large 27660 .exactZero (none)

def event27664 : Event := .specializationComputed (⟨.program ⟨214⟩, ⟨15715⟩⟩) ⟨⟨132⟩, ⟨39⟩, ⟨109⟩⟩ ⟨27506, 27664⟩

def event27665 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨21127⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21124⟩⟩]⟩) (1) 0 2 (.universal 27664 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21124⟩⟩]⟩) (none) 27663)

def event27666 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21127⟩⟩, .relation 27665 1, ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6719⟩⟩]⟩, (1)⟩)

def event27667 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21127⟩⟩, .relation 27665 0, ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6695⟩⟩, ⟨.program ⟨214⟩, ⟨27471⟩⟩]⟩, (-1)⟩)

def event27668 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21127⟩⟩, .relation 27665 2, ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15714⟩⟩], [⟨.program ⟨214⟩, ⟨24045⟩⟩]⟩, (1)⟩)

def event27669 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21127⟩⟩, .relation 27665 3, ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15757⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact27670RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6695⟩⟩, ⟨.program ⟨214⟩, ⟨27471⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6719⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15714⟩⟩], [⟨.program ⟨214⟩, ⟨24045⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15757⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact27670RawTermsValid :
    exact27670RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event27670 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21127⟩⟩) exact27670RawTerms .large 27502 (.finite 1811303510016) (some (27504))

def event27671 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27474⟩⟩) 0 ⟨21127⟩ 27670

def event27672 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27474⟩⟩) 1 ⟨27473⟩ 27492

def event27673 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27474⟩⟩) (.sum [.predecessor 0 27671 .coefficient, .predecessor 1 27672 .coefficient])

def event27674 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27474⟩⟩, .operator (⟨27670, 0⟩, ⟨27492, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6695⟩⟩, ⟨.program ⟨214⟩, ⟨27471⟩⟩]⟩, (1)⟩)

def event27675 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27474⟩⟩, .operator (⟨27670, 2⟩, ⟨27492, 1⟩), ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15714⟩⟩], [⟨.program ⟨214⟩, ⟨24045⟩⟩]⟩, (-1)⟩)

def event27676 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27474⟩⟩) (.sum [.result 27670 .summary, .result 27492 .summary])

def exact27677RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6719⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15757⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact27677RawTermsValid :
    exact27677RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event27677 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27474⟩⟩) exact27677RawTerms .large 27673 (.finite 1292001236604524572672) (some (27676))

def event27678 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23980⟩⟩) 0 ⟨15596⟩ 1158

def event27679 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23980⟩⟩) (.authority (.programFamilyFact))

def event27680 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨23980⟩⟩) (.finite 3720)

def event27681 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23982⟩⟩) 0 ⟨6689⟩ 5477

def event27682 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23982⟩⟩) 1 ⟨23980⟩ 27680

def event27683 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23982⟩⟩) (.authority (.operator))

def exact27684RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨23982⟩⟩]⟩, (1)⟩]

theorem exact27684RawTermsValid :
    exact27684RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event27684 : Event := .resultExact (⟨.program ⟨214⟩, ⟨23982⟩⟩) exact27684RawTerms .large 27683 .exactZero (none)

def event27685 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27254⟩⟩) 0 ⟨23982⟩ 27684

def event27686 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27254⟩⟩) (.authority (.operator))

def exact27687RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨27254⟩⟩]⟩, (1)⟩]

theorem exact27687RawTermsValid :
    exact27687RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event27687 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27254⟩⟩) exact27687RawTerms (.finite 8192) 27686 .exactZero (none)

def event27688 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23463⟩⟩) 0 ⟨13585⟩ 1152

def event27689 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23463⟩⟩) (.authority (.programFamilyFact))

def event27690 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨23463⟩⟩) (.finite 3720)

def event27691 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23464⟩⟩) 0 ⟨6689⟩ 5477

def event27692 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23464⟩⟩) 1 ⟨23463⟩ 27690

def event27693 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23464⟩⟩) (.authority (.operator))

def exact27694RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨23464⟩⟩]⟩, (1)⟩]

theorem exact27694RawTermsValid :
    exact27694RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event27694 : Event := .resultExact (⟨.program ⟨214⟩, ⟨23464⟩⟩) exact27694RawTerms .large 27693 .exactZero (none)

def event27695 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25850⟩⟩) 0 ⟨23464⟩ 27694

def event27696 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25850⟩⟩) (.authority (.operator))

def exact27697RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨25850⟩⟩]⟩, (1)⟩]

theorem exact27697RawTermsValid :
    exact27697RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event27697 : Event := .resultExact (⟨.program ⟨214⟩, ⟨25850⟩⟩) exact27697RawTerms (.finite 8192) 27696 .exactZero (none)

def event27698 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11230⟩⟩) 0 ⟨11229⟩ 1141

def event27699 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11230⟩⟩) 1 ⟨6570⟩ 21420

def event27700 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11230⟩⟩) (.tensor (.predecessor 0 27698 .coefficient) (.predecessor 1 27699 .coefficient) true false)

def event27701 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨11230⟩⟩, .operator (⟨1141, 0⟩, ⟨21420, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨11229⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact27702RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨11229⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact27702RawTermsValid :
    exact27702RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event27702 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11230⟩⟩) exact27702RawTerms .large 27700 .exactZero (none)

def event27703 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7346⟩⟩) 0 ⟨5557⟩ 21290

def event27704 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7346⟩⟩) 1 ⟨6776⟩ 12985

def event27705 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7346⟩⟩) (.product (.predecessor 0 27703 .coefficient) (.predecessor 1 27704 .coefficient) (⟨false, false, none, none, none⟩))

def event27706 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7346⟩⟩, .operator (⟨21290, 0⟩, ⟨12985, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6776⟩⟩]⟩, (1)⟩)

def exact27707RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6776⟩⟩]⟩, (1)⟩]

theorem exact27707RawTermsValid :
    exact27707RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event27707 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7346⟩⟩) exact27707RawTerms .large 27705 .exactZero (none)

def event27708 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11231⟩⟩) 0 ⟨7346⟩ 27707

def event27709 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11231⟩⟩) 1 ⟨11230⟩ 27702

def event27710 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11231⟩⟩) (.sum [.predecessor 0 27708 .coefficient, .predecessor 1 27709 .coefficient])

def exact27711RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6776⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨11229⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact27711RawTermsValid :
    exact27711RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event27711 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11231⟩⟩) exact27711RawTerms .large 27710 .exactZero (none)

def event27712 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11232⟩⟩) 0 ⟨11231⟩ 27711

def event27713 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11232⟩⟩) 1 ⟨90⟩ 12977

def event27714 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11232⟩⟩) (.sum [.predecessor 0 27712 .coefficient, .predecessor 1 27713 .coefficient])

def event27715 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11232⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨90⟩⟩]⟩) [⟨.result 12977 .coefficient, false, none⟩])

def event27716 : Event := .survivorFold (1) 27715

def exact27717RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6776⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨11229⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact27717RawTermsValid :
    exact27717RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event27717 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11232⟩⟩) exact27717RawTerms .large 27714 (.finite 26) (some (27715))

def event27718 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13586⟩⟩) 0 ⟨11232⟩ 27717

def event27719 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13586⟩⟩) 1 ⟨13583⟩ 1144

def event27720 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13586⟩⟩) (.product (.predecessor 0 27718 .coefficient) (.predecessor 1 27719 .coefficient) (⟨false, true, none, none, some 1⟩))

def event27721 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13586⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨13583⟩⟩], []⟩) [⟨.result 1144 .coefficient, true, some 1⟩])

def event27722 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13586⟩⟩) (.product (.result 27717 .summary) (.transfer 27721) (⟨false, false, none, none, none⟩))

def event27723 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨13586⟩⟩, .operator (⟨27717, 1⟩, ⟨1144, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨11229⟩⟩, ⟨.program ⟨214⟩, ⟨13583⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def event27724 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨13586⟩⟩, .operator (⟨27717, 0⟩, ⟨1144, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨13583⟩⟩], [⟨.program ⟨214⟩, ⟨6776⟩⟩]⟩, (1)⟩)

def exact27725RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨11229⟩⟩, ⟨.program ⟨214⟩, ⟨13583⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨13583⟩⟩], [⟨.program ⟨214⟩, ⟨6776⟩⟩]⟩, (1)⟩]

theorem exact27725RawTermsValid :
    exact27725RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event27725 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13586⟩⟩) exact27725RawTerms .large 27720 (.finite 8320) (some (27722))

def event27726 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13587⟩⟩) 0 ⟨13583⟩ 1144

def event27727 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13587⟩⟩) 1 ⟨6570⟩ 21420

def event27728 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13587⟩⟩) (.tensor (.predecessor 0 27726 .coefficient) (.predecessor 1 27727 .coefficient) true false)

def event27729 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨13587⟩⟩, .operator (⟨1144, 0⟩, ⟨21420, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨13583⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact27730RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨13583⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact27730RawTermsValid :
    exact27730RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event27730 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13587⟩⟩) exact27730RawTerms .large 27728 .exactZero (none)

def event27731 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7363⟩⟩) 0 ⟨5557⟩ 21290

def event27732 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7363⟩⟩) 1 ⟨6793⟩ 13026

def event27733 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7363⟩⟩) (.product (.predecessor 0 27731 .coefficient) (.predecessor 1 27732 .coefficient) (⟨false, false, none, none, none⟩))

def event27734 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7363⟩⟩, .operator (⟨21290, 0⟩, ⟨13026, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6793⟩⟩]⟩, (1)⟩)

def exact27735RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6793⟩⟩]⟩, (1)⟩]

theorem exact27735RawTermsValid :
    exact27735RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event27735 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7363⟩⟩) exact27735RawTerms .large 27733 .exactZero (none)

def event27736 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13588⟩⟩) 0 ⟨7363⟩ 27735

def event27737 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13588⟩⟩) 1 ⟨13587⟩ 27730

def event27738 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13588⟩⟩) (.sum [.predecessor 0 27736 .coefficient, .predecessor 1 27737 .coefficient])

def exact27739RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6793⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨13583⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact27739RawTermsValid :
    exact27739RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event27739 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13588⟩⟩) exact27739RawTerms .large 27738 .exactZero (none)

def event27740 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13589⟩⟩) 0 ⟨13588⟩ 27739

def event27741 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13589⟩⟩) 1 ⟨107⟩ 13018

def event27742 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13589⟩⟩) (.sum [.predecessor 0 27740 .coefficient, .predecessor 1 27741 .coefficient])

def event27743 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13589⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨107⟩⟩]⟩) [⟨.result 13018 .coefficient, false, none⟩])

def event27744 : Event := .survivorFold (1) 27743

def exact27745RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6793⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨13583⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact27745RawTermsValid :
    exact27745RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event27745 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13589⟩⟩) exact27745RawTerms .large 27742 (.finite 26) (some (27743))

def event27746 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13590⟩⟩) 0 ⟨13589⟩ 27745

def event27747 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13590⟩⟩) 1 ⟨7844⟩ 13015

def event27748 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13590⟩⟩) (.product (.predecessor 0 27746 .coefficient) (.predecessor 1 27747 .coefficient) (⟨false, false, none, none, none⟩))

def event27749 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13590⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨7843⟩⟩]⟩) [⟨.result 13011 .coefficient, false, none⟩])

def event27750 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13590⟩⟩) (.product (.result 27745 .summary) (.transfer 27749) (⟨false, false, none, none, none⟩))

def event27751 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨13590⟩⟩, .operator (⟨27745, 1⟩, ⟨13015, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨13583⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7843⟩⟩]⟩, (-1)⟩)

def event27752 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨13590⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨13583⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7843⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨7843⟩⟩) ⟨6776⟩ 12985)

def event27753 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨13590⟩⟩, .relation 27752 0, ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨13583⟩⟩], [⟨.program ⟨214⟩, ⟨6776⟩⟩]⟩, (-1)⟩)

def event27754 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨13590⟩⟩, .operator (⟨27745, 0⟩, ⟨13015, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6793⟩⟩, ⟨.program ⟨214⟩, ⟨7843⟩⟩]⟩, (1)⟩)

def exact27755RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6793⟩⟩, ⟨.program ⟨214⟩, ⟨7843⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨13583⟩⟩], [⟨.program ⟨214⟩, ⟨6776⟩⟩]⟩, (-1)⟩]

theorem exact27755RawTermsValid :
    exact27755RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event27755 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13590⟩⟩) exact27755RawTerms .large 27748 (.finite 95420416) (some (27750))

def event27756 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13591⟩⟩) 0 ⟨13590⟩ 27755

def event27757 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13591⟩⟩) 1 ⟨13586⟩ 27725

def event27758 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13591⟩⟩) (.sum [.predecessor 0 27756 .coefficient, .predecessor 1 27757 .coefficient])

def event27759 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨13591⟩⟩, .operator (⟨27755, 1⟩, ⟨27725, 1⟩), ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨13583⟩⟩], [⟨.program ⟨214⟩, ⟨6776⟩⟩]⟩, (1)⟩)

def event27760 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13591⟩⟩) (.sum [.result 27755 .summary, .result 27725 .summary])

def exact27761RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6793⟩⟩, ⟨.program ⟨214⟩, ⟨7843⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨11229⟩⟩, ⟨.program ⟨214⟩, ⟨13583⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact27761RawTermsValid :
    exact27761RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event27761 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13591⟩⟩) exact27761RawTerms .large 27758 (.finite 95428736) (some (27760))

def event27762 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25851⟩⟩) 0 ⟨13591⟩ 27761

def event27763 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25851⟩⟩) 1 ⟨25850⟩ 27697

def event27764 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25851⟩⟩) (.product (.predecessor 0 27762 .coefficient) (.predecessor 1 27763 .coefficient) (⟨false, false, none, none, none⟩))

def event27765 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25851⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨25850⟩⟩]⟩) [⟨.result 27697 .coefficient, false, none⟩])

def event27766 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25851⟩⟩) (.product (.result 27761 .summary) (.transfer 27765) (⟨false, false, none, none, none⟩))

def event27767 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25851⟩⟩, .operator (⟨27761, 1⟩, ⟨27697, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨11229⟩⟩, ⟨.program ⟨214⟩, ⟨13583⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25850⟩⟩]⟩, (-1)⟩)

def event27768 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨25851⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨11229⟩⟩, ⟨.program ⟨214⟩, ⟨13583⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25850⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨25850⟩⟩) ⟨23464⟩ 27694)

def event27769 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25851⟩⟩, .relation 27768 0, ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨11229⟩⟩, ⟨.program ⟨214⟩, ⟨13583⟩⟩], [⟨.program ⟨214⟩, ⟨23464⟩⟩]⟩, (-1)⟩)

def event27770 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25851⟩⟩, .operator (⟨27761, 0⟩, ⟨27697, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6793⟩⟩, ⟨.program ⟨214⟩, ⟨7843⟩⟩, ⟨.program ⟨214⟩, ⟨25850⟩⟩]⟩, (1)⟩)

def exact27771RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6793⟩⟩, ⟨.program ⟨214⟩, ⟨7843⟩⟩, ⟨.program ⟨214⟩, ⟨25850⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨11229⟩⟩, ⟨.program ⟨214⟩, ⟨13583⟩⟩], [⟨.program ⟨214⟩, ⟨23464⟩⟩]⟩, (-1)⟩]

theorem exact27771RawTermsValid :
    exact27771RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event27771 : Event := .resultExact (⟨.program ⟨214⟩, ⟨25851⟩⟩) exact27771RawTerms .large 27764 (.finite 350224987979776) (some (27766))

def event27772 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19324⟩⟩) 0 ⟨13585⟩ 1152

def event27773 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19324⟩⟩) (.authority (.relationPreimageSource ⟨12⟩))

def exact27774RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨19324⟩⟩]⟩, (1)⟩]

theorem exact27774RawTermsValid :
    exact27774RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event27774 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19324⟩⟩) exact27774RawTerms (.finite 136065468) 27773 .exactZero (none)

def event27775 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19326⟩⟩) 0 ⟨19324⟩ 27774

def event27776 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19326⟩⟩) 1 ⟨2348⟩ 4

def event27777 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19326⟩⟩) (.scale (.predecessor 0 27775 .coefficient) (.value (.predecessor 1 27776 .coefficient)))

def exact27778RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨19324⟩⟩]⟩, (1)⟩]

theorem exact27778RawTermsValid :
    exact27778RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event27778 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19326⟩⟩) exact27778RawTerms (.finite 136065468) 27777 .exactZero (none)

def event27779 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19327⟩⟩) 0 ⟨5559⟩ 21512

def event27780 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19327⟩⟩) 1 ⟨19326⟩ 27778

def event27781 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19327⟩⟩) (.product (.predecessor 0 27779 .coefficient) (.predecessor 1 27780 .coefficient) (⟨false, false, none, none, none⟩))

def event27782 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19327⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨19324⟩⟩]⟩) [⟨.result 27774 .coefficient, false, none⟩])

def event27783 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19327⟩⟩) (.product (.result 21512 .summary) (.transfer 27782) (⟨false, false, none, none, none⟩))

def event27784 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19327⟩⟩, .operator (⟨21512, 0⟩, ⟨27778, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19324⟩⟩]⟩, (1)⟩)

def event27785 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨19325⟩⟩)

def event27786 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event27787 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event27788 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨4989⟩⟩) (.authority (.operator))

def event27789 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨4989⟩⟩) (.finite 5)

def event27790 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event27791 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event27792 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event27793 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event27794 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 27793

def event27795 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 27791

def event27796 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 27794 .coefficient) (.value (.predecessor 1 27795 .coefficient)))

def event27797 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event27798 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5514⟩⟩) 0 ⟨5503⟩ 27797

def event27799 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5514⟩⟩) 1 ⟨4989⟩ 27789

def event27800 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5514⟩⟩) (.sum [.predecessor 0 27798 .coefficient, .predecessor 1 27799 .coefficient])

def event27801 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5514⟩⟩) (.finite 222)

def event27802 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5554⟩⟩) 0 ⟨5514⟩ 27801

def event27803 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5554⟩⟩) 1 ⟨961⟩ 27787

def event27804 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5554⟩⟩) (.identity (.predecessor 1 27803 .coefficient))

def event27805 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5554⟩⟩) (.finite 224)

def event27806 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11229⟩⟩) 0 ⟨5554⟩ 27805

def event27807 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11229⟩⟩) (.authority (.programFamilyFact))

def exact27808RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11229⟩⟩], []⟩, (1)⟩]

theorem exact27808RawTermsValid :
    exact27808RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event27808 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11229⟩⟩) exact27808RawTerms (.finite 10) 27807 .exactZero (none)

def event27809 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13583⟩⟩) 0 ⟨5554⟩ 27805

def event27810 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13583⟩⟩) (.authority (.programFamilyFact))

def exact27811RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨13583⟩⟩], []⟩, (1)⟩]

theorem exact27811RawTermsValid :
    exact27811RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event27811 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13583⟩⟩) exact27811RawTerms (.finite 10) 27810 .exactZero (none)

def event27812 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13584⟩⟩) 0 ⟨13583⟩ 27811

def event27813 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13584⟩⟩) 1 ⟨11229⟩ 27808

def event27814 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13584⟩⟩) (.product (.predecessor 0 27812 .coefficient) (.predecessor 1 27813 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event27815 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13584⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨11229⟩⟩, ⟨.program ⟨214⟩, ⟨13583⟩⟩], []⟩) [⟨.result 27811 .coefficient, true, some 1⟩, ⟨.result 27808 .coefficient, true, some 1⟩])

def event27816 : Event := .survivorFold (1) 27815

def exact27817RawTerms : List Term := []

theorem exact27817RawTermsValid :
    exact27817RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event27817 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13584⟩⟩) exact27817RawTerms (.finite 100) 27814 (.finite 100) (some (27815))

def event27818 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13585⟩⟩) 0 ⟨13584⟩ 27817

def event27819 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13585⟩⟩) (.identity (.predecessor 0 27818 .coefficient))

def event27820 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨13585⟩⟩) (.finite 100)

def event27821 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19324⟩⟩) 0 ⟨13585⟩ 27820

def event27822 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19324⟩⟩) (.authority (.relationPreimageSource ⟨12⟩))

def exact27823RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨19324⟩⟩]⟩, (1)⟩]

theorem exact27823RawTermsValid :
    exact27823RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event27823 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19324⟩⟩) exact27823RawTerms (.finite 136065468) 27822 .exactZero (none)

def event27824 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6⟩⟩) (.authority (.operator))

def exact27825RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩]⟩, (1)⟩]

theorem exact27825RawTermsValid :
    exact27825RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event27825 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6⟩⟩) exact27825RawTerms .large 27824 .exactZero (none)

def event27826 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19325⟩⟩) 0 ⟨6⟩ 27825

def event27827 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19325⟩⟩) 1 ⟨19324⟩ 27823

def event27828 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19325⟩⟩) (.product (.predecessor 0 27826 .coefficient) (.predecessor 1 27827 .coefficient) (⟨false, false, none, none, none⟩))

def event27829 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19325⟩⟩, .operator (⟨27825, 0⟩, ⟨27823, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19324⟩⟩]⟩, (1)⟩)

def exact27830RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19324⟩⟩]⟩, (1)⟩]

theorem exact27830RawTermsValid :
    exact27830RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event27830 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19325⟩⟩) exact27830RawTerms .large 27828 .exactZero (none)

def event27831 : Event := .preFoldPolynomial 27830 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19324⟩⟩]⟩, (1)⟩] .exactZero none

def exact27832RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19324⟩⟩]⟩, (1)⟩]

def event27832 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨19325⟩⟩) 27831 exact27832RawTerms .large 27828 .exactZero (none)

def event27833 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨25854⟩⟩)

def event27834 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event27835 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event27836 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨4989⟩⟩) (.authority (.operator))

def event27837 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨4989⟩⟩) (.finite 5)

def event27838 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event27839 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event27840 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event27841 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event27842 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 27841

def event27843 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 27839

def event27844 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 27842 .coefficient) (.value (.predecessor 1 27843 .coefficient)))

def event27845 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event27846 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5514⟩⟩) 0 ⟨5503⟩ 27845

def event27847 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5514⟩⟩) 1 ⟨4989⟩ 27837

def event27848 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5514⟩⟩) (.sum [.predecessor 0 27846 .coefficient, .predecessor 1 27847 .coefficient])

def event27849 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5514⟩⟩) (.finite 222)

def event27850 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5554⟩⟩) 0 ⟨5514⟩ 27849

def event27851 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5554⟩⟩) 1 ⟨961⟩ 27835

def event27852 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5554⟩⟩) (.identity (.predecessor 1 27851 .coefficient))

def event27853 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5554⟩⟩) (.finite 224)

def event27854 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11229⟩⟩) 0 ⟨5554⟩ 27853

def event27855 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11229⟩⟩) (.authority (.programFamilyFact))

def exact27856RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11229⟩⟩], []⟩, (1)⟩]

theorem exact27856RawTermsValid :
    exact27856RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event27856 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11229⟩⟩) exact27856RawTerms (.finite 10) 27855 .exactZero (none)

def event27857 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13583⟩⟩) 0 ⟨5554⟩ 27853

def event27858 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13583⟩⟩) (.authority (.programFamilyFact))

def exact27859RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨13583⟩⟩], []⟩, (1)⟩]

theorem exact27859RawTermsValid :
    exact27859RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event27859 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13583⟩⟩) exact27859RawTerms (.finite 10) 27858 .exactZero (none)

def event27860 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13584⟩⟩) 0 ⟨13583⟩ 27859

def event27861 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13584⟩⟩) 1 ⟨11229⟩ 27856

def event27862 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13584⟩⟩) (.product (.predecessor 0 27860 .coefficient) (.predecessor 1 27861 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event27863 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨13584⟩⟩, .operator (⟨27859, 0⟩, ⟨27856, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨11229⟩⟩, ⟨.program ⟨214⟩, ⟨13583⟩⟩], []⟩, (1)⟩)

def exact27864RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11229⟩⟩, ⟨.program ⟨214⟩, ⟨13583⟩⟩], []⟩, (1)⟩]

theorem exact27864RawTermsValid :
    exact27864RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event27864 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13584⟩⟩) exact27864RawTerms (.finite 100) 27862 .exactZero (none)

def event27865 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13585⟩⟩) 0 ⟨13584⟩ 27864

def event27866 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13585⟩⟩) (.identity (.predecessor 0 27865 .coefficient))

def event27867 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨13585⟩⟩) (.finite 100)

def event27868 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23463⟩⟩) 0 ⟨13585⟩ 27867

def event27869 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23463⟩⟩) (.authority (.programFamilyFact))

def event27870 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨23463⟩⟩) (.finite 3720)

def event27871 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨6689⟩⟩) .missing

def event27872 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23464⟩⟩) 0 ⟨6689⟩ 27871

def event27873 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23464⟩⟩) 1 ⟨23463⟩ 27870

def event27874 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23464⟩⟩) (.authority (.operator))

def exact27875RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨23464⟩⟩]⟩, (1)⟩]

theorem exact27875RawTermsValid :
    exact27875RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event27875 : Event := .resultExact (⟨.program ⟨214⟩, ⟨23464⟩⟩) exact27875RawTerms .large 27874 .exactZero (none)

def event27876 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25850⟩⟩) 0 ⟨23464⟩ 27875

def event27877 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25850⟩⟩) (.authority (.operator))

def exact27878RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨25850⟩⟩]⟩, (1)⟩]

theorem exact27878RawTermsValid :
    exact27878RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event27878 : Event := .resultExact (⟨.program ⟨214⟩, ⟨25850⟩⟩) exact27878RawTerms (.finite 8192) 27877 .exactZero (none)

def event27879 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨110⟩⟩) (.authority (.operator))

def event27880 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨110⟩⟩) .exactZero

def event27881 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13675⟩⟩) 0 ⟨13585⟩ 27867

def event27882 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13675⟩⟩) 1 ⟨110⟩ 27880

def event27883 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13675⟩⟩) (.sum [.predecessor 0 27881 .coefficient, .predecessor 1 27882 .coefficient])

def event27884 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨13675⟩⟩) (.finite 100)

def event27885 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13676⟩⟩) 0 ⟨13675⟩ 27884

def event27886 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13676⟩⟩) (.identity (.predecessor 0 27885 .coefficient))

def exact27887RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11229⟩⟩, ⟨.program ⟨214⟩, ⟨13583⟩⟩], []⟩, (1)⟩]

theorem exact27887RawTermsValid :
    exact27887RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event27887 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13676⟩⟩) exact27887RawTerms (.finite 100) 27886 .exactZero (none)

def event27888 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6544⟩⟩) (.authority (.factStore))

def exact27889RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact27889RawTermsValid :
    exact27889RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event27889 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6544⟩⟩) exact27889RawTerms .large 27888 .exactZero (none)

def event27890 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13677⟩⟩) 0 ⟨6544⟩ 27889

def event27891 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13677⟩⟩) 1 ⟨13676⟩ 27887

def event27892 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13677⟩⟩) (.product (.predecessor 0 27890 .coefficient) (.predecessor 1 27891 .coefficient) (⟨false, false, none, none, none⟩))

def event27893 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨13677⟩⟩, .operator (⟨27889, 0⟩, ⟨27887, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨11229⟩⟩, ⟨.program ⟨214⟩, ⟨13583⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact27894RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11229⟩⟩, ⟨.program ⟨214⟩, ⟨13583⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact27894RawTermsValid :
    exact27894RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event27894 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13677⟩⟩) exact27894RawTerms .large 27892 .exactZero (none)

def event27895 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.authority (.operator))

def event27896 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.finite 1)

def event27897 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6757⟩⟩) 0 ⟨6689⟩ 27871

def event27898 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6757⟩⟩) (.authority (.operator))

def exact27899RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6757⟩⟩]⟩, (1)⟩]

theorem exact27899RawTermsValid :
    exact27899RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event27899 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6757⟩⟩) exact27899RawTerms .large 27898 .exactZero (none)

def event27900 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6776⟩⟩) 0 ⟨6757⟩ 27899

def event27901 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6776⟩⟩) (.identity (.predecessor 0 27900 .coefficient))

def exact27902RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6776⟩⟩]⟩, (1)⟩]

theorem exact27902RawTermsValid :
    exact27902RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event27902 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6776⟩⟩) exact27902RawTerms .large 27901 .exactZero (none)

def event27903 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7843⟩⟩) 0 ⟨6776⟩ 27902

def eventLeaf1728 : Array AnnotatedEvent := #[
  { event := event27648
    frameStart := 27560 },
  { event := event27649
    frameStart := 27560 },
  { event := event27650
    frameStart := 27560 },
  { event := event27651
    frameStart := 27560 },
  { event := event27652
    frameStart := 27560 },
  { event := event27653
    frameStart := 27560 },
  { event := event27654
    frameStart := 27560 },
  { event := event27655
    frameStart := 27560 },
  { event := event27656
    frameStart := 27560 },
  { event := event27657
    frameStart := 27560 },
  { event := event27658
    frameStart := 27560 },
  { event := event27659
    frameStart := 27560 },
  { event := event27660
    frameStart := 27560 },
  { event := event27661
    frameStart := 27560 },
  { event := event27662
    frameStart := 27560 },
  { event := event27663
    frameStart := 27560 }
]

def eventLeaf1729 : Array AnnotatedEvent := #[
  { event := event27664
    frameStart := 0 },
  { event := event27665
    frameStart := 0 },
  { event := event27666
    frameStart := 0 },
  { event := event27667
    frameStart := 0 },
  { event := event27668
    frameStart := 0 },
  { event := event27669
    frameStart := 0 },
  { event := event27670
    frameStart := 0 },
  { event := event27671
    frameStart := 0 },
  { event := event27672
    frameStart := 0 },
  { event := event27673
    frameStart := 0 },
  { event := event27674
    frameStart := 0 },
  { event := event27675
    frameStart := 0 },
  { event := event27676
    frameStart := 0 },
  { event := event27677
    frameStart := 0 },
  { event := event27678
    frameStart := 0 },
  { event := event27679
    frameStart := 0 }
]

def eventLeaf1730 : Array AnnotatedEvent := #[
  { event := event27680
    frameStart := 0 },
  { event := event27681
    frameStart := 0 },
  { event := event27682
    frameStart := 0 },
  { event := event27683
    frameStart := 0 },
  { event := event27684
    frameStart := 0 },
  { event := event27685
    frameStart := 0 },
  { event := event27686
    frameStart := 0 },
  { event := event27687
    frameStart := 0 },
  { event := event27688
    frameStart := 0 },
  { event := event27689
    frameStart := 0 },
  { event := event27690
    frameStart := 0 },
  { event := event27691
    frameStart := 0 },
  { event := event27692
    frameStart := 0 },
  { event := event27693
    frameStart := 0 },
  { event := event27694
    frameStart := 0 },
  { event := event27695
    frameStart := 0 }
]

def eventLeaf1731 : Array AnnotatedEvent := #[
  { event := event27696
    frameStart := 0 },
  { event := event27697
    frameStart := 0 },
  { event := event27698
    frameStart := 0 },
  { event := event27699
    frameStart := 0 },
  { event := event27700
    frameStart := 0 },
  { event := event27701
    frameStart := 0 },
  { event := event27702
    frameStart := 0 },
  { event := event27703
    frameStart := 0 },
  { event := event27704
    frameStart := 0 },
  { event := event27705
    frameStart := 0 },
  { event := event27706
    frameStart := 0 },
  { event := event27707
    frameStart := 0 },
  { event := event27708
    frameStart := 0 },
  { event := event27709
    frameStart := 0 },
  { event := event27710
    frameStart := 0 },
  { event := event27711
    frameStart := 0 }
]

def eventLeaf1732 : Array AnnotatedEvent := #[
  { event := event27712
    frameStart := 0 },
  { event := event27713
    frameStart := 0 },
  { event := event27714
    frameStart := 0 },
  { event := event27715
    frameStart := 0 },
  { event := event27716
    frameStart := 0 },
  { event := event27717
    frameStart := 0 },
  { event := event27718
    frameStart := 0 },
  { event := event27719
    frameStart := 0 },
  { event := event27720
    frameStart := 0 },
  { event := event27721
    frameStart := 0 },
  { event := event27722
    frameStart := 0 },
  { event := event27723
    frameStart := 0 },
  { event := event27724
    frameStart := 0 },
  { event := event27725
    frameStart := 0 },
  { event := event27726
    frameStart := 0 },
  { event := event27727
    frameStart := 0 }
]

def eventLeaf1733 : Array AnnotatedEvent := #[
  { event := event27728
    frameStart := 0 },
  { event := event27729
    frameStart := 0 },
  { event := event27730
    frameStart := 0 },
  { event := event27731
    frameStart := 0 },
  { event := event27732
    frameStart := 0 },
  { event := event27733
    frameStart := 0 },
  { event := event27734
    frameStart := 0 },
  { event := event27735
    frameStart := 0 },
  { event := event27736
    frameStart := 0 },
  { event := event27737
    frameStart := 0 },
  { event := event27738
    frameStart := 0 },
  { event := event27739
    frameStart := 0 },
  { event := event27740
    frameStart := 0 },
  { event := event27741
    frameStart := 0 },
  { event := event27742
    frameStart := 0 },
  { event := event27743
    frameStart := 0 }
]

def eventLeaf1734 : Array AnnotatedEvent := #[
  { event := event27744
    frameStart := 0 },
  { event := event27745
    frameStart := 0 },
  { event := event27746
    frameStart := 0 },
  { event := event27747
    frameStart := 0 },
  { event := event27748
    frameStart := 0 },
  { event := event27749
    frameStart := 0 },
  { event := event27750
    frameStart := 0 },
  { event := event27751
    frameStart := 0 },
  { event := event27752
    frameStart := 0 },
  { event := event27753
    frameStart := 0 },
  { event := event27754
    frameStart := 0 },
  { event := event27755
    frameStart := 0 },
  { event := event27756
    frameStart := 0 },
  { event := event27757
    frameStart := 0 },
  { event := event27758
    frameStart := 0 },
  { event := event27759
    frameStart := 0 }
]

def eventLeaf1735 : Array AnnotatedEvent := #[
  { event := event27760
    frameStart := 0 },
  { event := event27761
    frameStart := 0 },
  { event := event27762
    frameStart := 0 },
  { event := event27763
    frameStart := 0 },
  { event := event27764
    frameStart := 0 },
  { event := event27765
    frameStart := 0 },
  { event := event27766
    frameStart := 0 },
  { event := event27767
    frameStart := 0 },
  { event := event27768
    frameStart := 0 },
  { event := event27769
    frameStart := 0 },
  { event := event27770
    frameStart := 0 },
  { event := event27771
    frameStart := 0 },
  { event := event27772
    frameStart := 0 },
  { event := event27773
    frameStart := 0 },
  { event := event27774
    frameStart := 0 },
  { event := event27775
    frameStart := 0 }
]

def eventLeaf1736 : Array AnnotatedEvent := #[
  { event := event27776
    frameStart := 0 },
  { event := event27777
    frameStart := 0 },
  { event := event27778
    frameStart := 0 },
  { event := event27779
    frameStart := 0 },
  { event := event27780
    frameStart := 0 },
  { event := event27781
    frameStart := 0 },
  { event := event27782
    frameStart := 0 },
  { event := event27783
    frameStart := 0 },
  { event := event27784
    frameStart := 0 },
  { event := event27785
    frameStart := 27785 },
  { event := event27786
    frameStart := 27785 },
  { event := event27787
    frameStart := 27785 },
  { event := event27788
    frameStart := 27785 },
  { event := event27789
    frameStart := 27785 },
  { event := event27790
    frameStart := 27785 },
  { event := event27791
    frameStart := 27785 }
]

def eventLeaf1737 : Array AnnotatedEvent := #[
  { event := event27792
    frameStart := 27785 },
  { event := event27793
    frameStart := 27785 },
  { event := event27794
    frameStart := 27785 },
  { event := event27795
    frameStart := 27785 },
  { event := event27796
    frameStart := 27785 },
  { event := event27797
    frameStart := 27785 },
  { event := event27798
    frameStart := 27785 },
  { event := event27799
    frameStart := 27785 },
  { event := event27800
    frameStart := 27785 },
  { event := event27801
    frameStart := 27785 },
  { event := event27802
    frameStart := 27785 },
  { event := event27803
    frameStart := 27785 },
  { event := event27804
    frameStart := 27785 },
  { event := event27805
    frameStart := 27785 },
  { event := event27806
    frameStart := 27785 },
  { event := event27807
    frameStart := 27785 }
]

def eventLeaf1738 : Array AnnotatedEvent := #[
  { event := event27808
    frameStart := 27785 },
  { event := event27809
    frameStart := 27785 },
  { event := event27810
    frameStart := 27785 },
  { event := event27811
    frameStart := 27785 },
  { event := event27812
    frameStart := 27785 },
  { event := event27813
    frameStart := 27785 },
  { event := event27814
    frameStart := 27785 },
  { event := event27815
    frameStart := 27785 },
  { event := event27816
    frameStart := 27785 },
  { event := event27817
    frameStart := 27785 },
  { event := event27818
    frameStart := 27785 },
  { event := event27819
    frameStart := 27785 },
  { event := event27820
    frameStart := 27785 },
  { event := event27821
    frameStart := 27785 },
  { event := event27822
    frameStart := 27785 },
  { event := event27823
    frameStart := 27785 }
]

def eventLeaf1739 : Array AnnotatedEvent := #[
  { event := event27824
    frameStart := 27785 },
  { event := event27825
    frameStart := 27785 },
  { event := event27826
    frameStart := 27785 },
  { event := event27827
    frameStart := 27785 },
  { event := event27828
    frameStart := 27785 },
  { event := event27829
    frameStart := 27785 },
  { event := event27830
    frameStart := 27785 },
  { event := event27831
    frameStart := 27785 },
  { event := event27832
    frameStart := 27785 },
  { event := event27833
    frameStart := 27833 },
  { event := event27834
    frameStart := 27833 },
  { event := event27835
    frameStart := 27833 },
  { event := event27836
    frameStart := 27833 },
  { event := event27837
    frameStart := 27833 },
  { event := event27838
    frameStart := 27833 },
  { event := event27839
    frameStart := 27833 }
]

def eventLeaf1740 : Array AnnotatedEvent := #[
  { event := event27840
    frameStart := 27833 },
  { event := event27841
    frameStart := 27833 },
  { event := event27842
    frameStart := 27833 },
  { event := event27843
    frameStart := 27833 },
  { event := event27844
    frameStart := 27833 },
  { event := event27845
    frameStart := 27833 },
  { event := event27846
    frameStart := 27833 },
  { event := event27847
    frameStart := 27833 },
  { event := event27848
    frameStart := 27833 },
  { event := event27849
    frameStart := 27833 },
  { event := event27850
    frameStart := 27833 },
  { event := event27851
    frameStart := 27833 },
  { event := event27852
    frameStart := 27833 },
  { event := event27853
    frameStart := 27833 },
  { event := event27854
    frameStart := 27833 },
  { event := event27855
    frameStart := 27833 }
]

def eventLeaf1741 : Array AnnotatedEvent := #[
  { event := event27856
    frameStart := 27833 },
  { event := event27857
    frameStart := 27833 },
  { event := event27858
    frameStart := 27833 },
  { event := event27859
    frameStart := 27833 },
  { event := event27860
    frameStart := 27833 },
  { event := event27861
    frameStart := 27833 },
  { event := event27862
    frameStart := 27833 },
  { event := event27863
    frameStart := 27833 },
  { event := event27864
    frameStart := 27833 },
  { event := event27865
    frameStart := 27833 },
  { event := event27866
    frameStart := 27833 },
  { event := event27867
    frameStart := 27833 },
  { event := event27868
    frameStart := 27833 },
  { event := event27869
    frameStart := 27833 },
  { event := event27870
    frameStart := 27833 },
  { event := event27871
    frameStart := 27833 }
]

def eventLeaf1742 : Array AnnotatedEvent := #[
  { event := event27872
    frameStart := 27833 },
  { event := event27873
    frameStart := 27833 },
  { event := event27874
    frameStart := 27833 },
  { event := event27875
    frameStart := 27833 },
  { event := event27876
    frameStart := 27833 },
  { event := event27877
    frameStart := 27833 },
  { event := event27878
    frameStart := 27833 },
  { event := event27879
    frameStart := 27833 },
  { event := event27880
    frameStart := 27833 },
  { event := event27881
    frameStart := 27833 },
  { event := event27882
    frameStart := 27833 },
  { event := event27883
    frameStart := 27833 },
  { event := event27884
    frameStart := 27833 },
  { event := event27885
    frameStart := 27833 },
  { event := event27886
    frameStart := 27833 },
  { event := event27887
    frameStart := 27833 }
]

def eventLeaf1743 : Array AnnotatedEvent := #[
  { event := event27888
    frameStart := 27833 },
  { event := event27889
    frameStart := 27833 },
  { event := event27890
    frameStart := 27833 },
  { event := event27891
    frameStart := 27833 },
  { event := event27892
    frameStart := 27833 },
  { event := event27893
    frameStart := 27833 },
  { event := event27894
    frameStart := 27833 },
  { event := event27895
    frameStart := 27833 },
  { event := event27896
    frameStart := 27833 },
  { event := event27897
    frameStart := 27833 },
  { event := event27898
    frameStart := 27833 },
  { event := event27899
    frameStart := 27833 },
  { event := event27900
    frameStart := 27833 },
  { event := event27901
    frameStart := 27833 },
  { event := event27902
    frameStart := 27833 },
  { event := event27903
    frameStart := 27833 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events108
