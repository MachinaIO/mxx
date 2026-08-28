import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events268

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event68608 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16419⟩⟩) (.product (.predecessor 0 68606 .coefficient) (.predecessor 1 68607 .coefficient) (⟨false, false, none, none, none⟩))

def event68609 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨16419⟩⟩, .operator (⟨68605, 0⟩, ⟨68603, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨16377⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact68610RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16377⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact68610RawTermsValid :
    exact68610RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68610 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16419⟩⟩) exact68610RawTerms .large 68608 .exactZero (none)

def event68611 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6701⟩⟩) 0 ⟨6689⟩ 68587

def event68612 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6701⟩⟩) (.authority (.operator))

def exact68613RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6701⟩⟩]⟩, (1)⟩]

theorem exact68613RawTermsValid :
    exact68613RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68613 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6701⟩⟩) exact68613RawTerms .large 68612 .exactZero (none)

def event68614 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16420⟩⟩) 0 ⟨6701⟩ 68613

def event68615 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16420⟩⟩) 1 ⟨16419⟩ 68610

def event68616 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16420⟩⟩) (.sum [.predecessor 0 68614 .coefficient, .predecessor 1 68615 .coefficient])

def exact68617RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6701⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16377⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact68617RawTermsValid :
    exact68617RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68617 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16420⟩⟩) exact68617RawTerms .large 68616 .exactZero (none)

def event68618 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28722⟩⟩) 0 ⟨16420⟩ 68617

def event68619 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28722⟩⟩) 1 ⟨28721⟩ 68594

def event68620 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28722⟩⟩) (.product (.predecessor 0 68618 .coefficient) (.predecessor 1 68619 .coefficient) (⟨false, false, none, none, none⟩))

def event68621 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28722⟩⟩, .operator (⟨68617, 0⟩, ⟨68594, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6701⟩⟩, ⟨.program ⟨214⟩, ⟨28721⟩⟩]⟩, (1)⟩)

def event68622 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28722⟩⟩, .operator (⟨68617, 1⟩, ⟨68594, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨16377⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28721⟩⟩]⟩, (-1)⟩)

def event68623 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨28722⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨16377⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28721⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨28721⟩⟩) ⟨24411⟩ 68591)

def event68624 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28722⟩⟩, .relation 68623 0, ⟨[⟨.program ⟨214⟩, ⟨16377⟩⟩], [⟨.program ⟨214⟩, ⟨24411⟩⟩]⟩, (-1)⟩)

def exact68625RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6701⟩⟩, ⟨.program ⟨214⟩, ⟨28721⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16377⟩⟩], [⟨.program ⟨214⟩, ⟨24411⟩⟩]⟩, (-1)⟩]

theorem exact68625RawTermsValid :
    exact68625RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68625 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28722⟩⟩) exact68625RawTerms .large 68620 .exactZero (none)

def event68626 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17117⟩⟩) 0 ⟨16378⟩ 68583

def event68627 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨17117⟩⟩) (.authority (.programFamilyFact))

def exact68628RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨17117⟩⟩], []⟩, (1)⟩]

theorem exact68628RawTermsValid :
    exact68628RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68628 : Event := .resultExact (⟨.program ⟨214⟩, ⟨17117⟩⟩) exact68628RawTerms (.finite 62) 68627 .exactZero (none)

def event68629 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17118⟩⟩) 0 ⟨6544⟩ 68605

def event68630 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17118⟩⟩) 1 ⟨17117⟩ 68628

def event68631 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨17118⟩⟩) (.product (.predecessor 0 68629 .coefficient) (.predecessor 1 68630 .coefficient) (⟨false, true, none, none, some 1⟩))

def event68632 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨17118⟩⟩, .operator (⟨68605, 0⟩, ⟨68628, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨17117⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact68633RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨17117⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact68633RawTermsValid :
    exact68633RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68633 : Event := .resultExact (⟨.program ⟨214⟩, ⟨17118⟩⟩) exact68633RawTerms .large 68631 .exactZero (none)

def event68634 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6731⟩⟩) 0 ⟨6689⟩ 68587

def event68635 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6731⟩⟩) (.authority (.operator))

def exact68636RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6731⟩⟩]⟩, (1)⟩]

theorem exact68636RawTermsValid :
    exact68636RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68636 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6731⟩⟩) exact68636RawTerms .large 68635 .exactZero (none)

def event68637 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17119⟩⟩) 0 ⟨6731⟩ 68636

def event68638 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17119⟩⟩) 1 ⟨17118⟩ 68633

def event68639 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨17119⟩⟩) (.sum [.predecessor 0 68637 .coefficient, .predecessor 1 68638 .coefficient])

def exact68640RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6731⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17117⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact68640RawTermsValid :
    exact68640RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68640 : Event := .resultExact (⟨.program ⟨214⟩, ⟨17119⟩⟩) exact68640RawTerms .large 68639 .exactZero (none)

def event68641 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28726⟩⟩) 0 ⟨17119⟩ 68640

def event68642 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28726⟩⟩) 1 ⟨28722⟩ 68625

def event68643 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28726⟩⟩) (.sum [.predecessor 0 68641 .coefficient, .predecessor 1 68642 .coefficient])

def exact68644RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6701⟩⟩, ⟨.program ⟨214⟩, ⟨28721⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6731⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16377⟩⟩], [⟨.program ⟨214⟩, ⟨24411⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17117⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact68644RawTermsValid :
    exact68644RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68644 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28726⟩⟩) exact68644RawTerms .large 68643 .exactZero (none)

def event68645 : Event := .preFoldPolynomial 68644 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6701⟩⟩, ⟨.program ⟨214⟩, ⟨28721⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6731⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16377⟩⟩], [⟨.program ⟨214⟩, ⟨24411⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17117⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩] .exactZero none

def exact68646RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6701⟩⟩, ⟨.program ⟨214⟩, ⟨28721⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6731⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16377⟩⟩], [⟨.program ⟨214⟩, ⟨24411⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17117⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

def event68646 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨28726⟩⟩) 68645 exact68646RawTerms .large 68643 .exactZero (none)

def event68647 : Event := .specializationComputed (⟨.program ⟨214⟩, ⟨16378⟩⟩) ⟨⟨144⟩, ⟨52⟩, ⟨109⟩⟩ ⟨68489, 68647⟩

def event68648 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨21975⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21972⟩⟩]⟩) (1) 0 2 (.universal 68647 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21972⟩⟩]⟩) (none) 68646)

def event68649 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21975⟩⟩, .relation 68648 1, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6731⟩⟩]⟩, (1)⟩)

def event68650 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21975⟩⟩, .relation 68648 0, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6701⟩⟩, ⟨.program ⟨214⟩, ⟨28721⟩⟩]⟩, (-1)⟩)

def event68651 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21975⟩⟩, .relation 68648 2, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16377⟩⟩], [⟨.program ⟨214⟩, ⟨24411⟩⟩]⟩, (1)⟩)

def event68652 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21975⟩⟩, .relation 68648 3, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨17117⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact68653RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6701⟩⟩, ⟨.program ⟨214⟩, ⟨28721⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6731⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16377⟩⟩], [⟨.program ⟨214⟩, ⟨24411⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨17117⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact68653RawTermsValid :
    exact68653RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68653 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21975⟩⟩) exact68653RawTerms .large 68485 (.finite 1811303510016) (some (68487))

def event68654 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28724⟩⟩) 0 ⟨21975⟩ 68653

def event68655 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28724⟩⟩) 1 ⟨28723⟩ 68475

def event68656 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28724⟩⟩) (.sum [.predecessor 0 68654 .coefficient, .predecessor 1 68655 .coefficient])

def event68657 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28724⟩⟩, .operator (⟨68653, 0⟩, ⟨68475, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6701⟩⟩, ⟨.program ⟨214⟩, ⟨28721⟩⟩]⟩, (1)⟩)

def event68658 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28724⟩⟩, .operator (⟨68653, 2⟩, ⟨68475, 1⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16377⟩⟩], [⟨.program ⟨214⟩, ⟨24411⟩⟩]⟩, (-1)⟩)

def event68659 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28724⟩⟩) (.sum [.result 68653 .summary, .result 68475 .summary])

def exact68660RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6731⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨17117⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact68660RawTermsValid :
    exact68660RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68660 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28724⟩⟩) exact68660RawTerms .large 68656 (.finite 1292270185944771604480) (some (68659))

def event68661 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24346⟩⟩) 0 ⟨16259⟩ 3264

def event68662 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24346⟩⟩) (.authority (.programFamilyFact))

def event68663 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨24346⟩⟩) (.finite 3720)

def event68664 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24348⟩⟩) 0 ⟨6689⟩ 5477

def event68665 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24348⟩⟩) 1 ⟨24346⟩ 68663

def event68666 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24348⟩⟩) (.authority (.operator))

def exact68667RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨24348⟩⟩]⟩, (1)⟩]

theorem exact68667RawTermsValid :
    exact68667RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68667 : Event := .resultExact (⟨.program ⟨214⟩, ⟨24348⟩⟩) exact68667RawTerms .large 68666 .exactZero (none)

def event68668 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28504⟩⟩) 0 ⟨24348⟩ 68667

def event68669 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28504⟩⟩) (.authority (.operator))

def exact68670RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨28504⟩⟩]⟩, (1)⟩]

theorem exact68670RawTermsValid :
    exact68670RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68670 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28504⟩⟩) exact68670RawTerms (.finite 8192) 68669 .exactZero (none)

def event68671 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23077⟩⟩) 0 ⟨11755⟩ 3258

def event68672 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23077⟩⟩) (.authority (.programFamilyFact))

def event68673 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨23077⟩⟩) (.finite 3720)

def event68674 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23078⟩⟩) 0 ⟨6689⟩ 5477

def event68675 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23078⟩⟩) 1 ⟨23077⟩ 68673

def event68676 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23078⟩⟩) (.authority (.operator))

def exact68677RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨23078⟩⟩]⟩, (1)⟩]

theorem exact68677RawTermsValid :
    exact68677RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68677 : Event := .resultExact (⟨.program ⟨214⟩, ⟨23078⟩⟩) exact68677RawTerms .large 68676 .exactZero (none)

def event68678 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25137⟩⟩) 0 ⟨23078⟩ 68677

def event68679 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25137⟩⟩) (.authority (.operator))

def exact68680RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨25137⟩⟩]⟩, (1)⟩]

theorem exact68680RawTermsValid :
    exact68680RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68680 : Event := .resultExact (⟨.program ⟨214⟩, ⟨25137⟩⟩) exact68680RawTerms (.finite 8192) 68679 .exactZero (none)

def event68681 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11756⟩⟩) 0 ⟨11753⟩ 3247

def event68682 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11756⟩⟩) 1 ⟨6566⟩ 65295

def event68683 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11756⟩⟩) (.tensor (.predecessor 0 68681 .coefficient) (.predecessor 1 68682 .coefficient) true false)

def event68684 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨11756⟩⟩, .operator (⟨3247, 0⟩, ⟨65295, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨11753⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact68685RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨11753⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact68685RawTermsValid :
    exact68685RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68685 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11756⟩⟩) exact68685RawTerms .large 68683 .exactZero (none)

def event68686 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7201⟩⟩) 0 ⟨5533⟩ 65165

def event68687 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7201⟩⟩) 1 ⟨6783⟩ 9979

def event68688 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7201⟩⟩) (.product (.predecessor 0 68686 .coefficient) (.predecessor 1 68687 .coefficient) (⟨false, false, none, none, none⟩))

def event68689 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7201⟩⟩, .operator (⟨65165, 0⟩, ⟨9979, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6783⟩⟩]⟩, (1)⟩)

def exact68690RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6783⟩⟩]⟩, (1)⟩]

theorem exact68690RawTermsValid :
    exact68690RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68690 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7201⟩⟩) exact68690RawTerms .large 68688 .exactZero (none)

def event68691 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11757⟩⟩) 0 ⟨7201⟩ 68690

def event68692 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11757⟩⟩) 1 ⟨11756⟩ 68685

def event68693 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11757⟩⟩) (.sum [.predecessor 0 68691 .coefficient, .predecessor 1 68692 .coefficient])

def exact68694RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6783⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨11753⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact68694RawTermsValid :
    exact68694RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68694 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11757⟩⟩) exact68694RawTerms .large 68693 .exactZero (none)

def event68695 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11758⟩⟩) 0 ⟨11757⟩ 68694

def event68696 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11758⟩⟩) 1 ⟨97⟩ 9971

def event68697 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11758⟩⟩) (.sum [.predecessor 0 68695 .coefficient, .predecessor 1 68696 .coefficient])

def event68698 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11758⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨97⟩⟩]⟩) [⟨.result 9971 .coefficient, false, none⟩])

def event68699 : Event := .survivorFold (1) 68698

def exact68700RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6783⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨11753⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact68700RawTermsValid :
    exact68700RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68700 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11758⟩⟩) exact68700RawTerms .large 68697 (.finite 26) (some (68698))

def event68701 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11759⟩⟩) 0 ⟨11758⟩ 68700

def event68702 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11759⟩⟩) 1 ⟨9605⟩ 3250

def event68703 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11759⟩⟩) (.product (.predecessor 0 68701 .coefficient) (.predecessor 1 68702 .coefficient) (⟨false, true, none, none, some 1⟩))

def event68704 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11759⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨9605⟩⟩], []⟩) [⟨.result 3250 .coefficient, true, some 1⟩])

def event68705 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11759⟩⟩) (.product (.result 68700 .summary) (.transfer 68704) (⟨false, false, none, none, none⟩))

def event68706 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨11759⟩⟩, .operator (⟨68700, 1⟩, ⟨3250, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9605⟩⟩, ⟨.program ⟨214⟩, ⟨11753⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def event68707 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨11759⟩⟩, .operator (⟨68700, 0⟩, ⟨3250, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9605⟩⟩], [⟨.program ⟨214⟩, ⟨6783⟩⟩]⟩, (1)⟩)

def exact68708RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9605⟩⟩], [⟨.program ⟨214⟩, ⟨6783⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9605⟩⟩, ⟨.program ⟨214⟩, ⟨11753⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact68708RawTermsValid :
    exact68708RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68708 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11759⟩⟩) exact68708RawTerms .large 68703 (.finite 24960) (some (68705))

def event68709 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9606⟩⟩) 0 ⟨9605⟩ 3250

def event68710 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9606⟩⟩) 1 ⟨6566⟩ 65295

def event68711 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨9606⟩⟩) (.tensor (.predecessor 0 68709 .coefficient) (.predecessor 1 68710 .coefficient) true false)

def event68712 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨9606⟩⟩, .operator (⟨3250, 0⟩, ⟨65295, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9605⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact68713RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9605⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact68713RawTermsValid :
    exact68713RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68713 : Event := .resultExact (⟨.program ⟨214⟩, ⟨9606⟩⟩) exact68713RawTerms .large 68711 .exactZero (none)

def event68714 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7181⟩⟩) 0 ⟨5533⟩ 65165

def event68715 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7181⟩⟩) 1 ⟨6763⟩ 10020

def event68716 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7181⟩⟩) (.product (.predecessor 0 68714 .coefficient) (.predecessor 1 68715 .coefficient) (⟨false, false, none, none, none⟩))

def event68717 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7181⟩⟩, .operator (⟨65165, 0⟩, ⟨10020, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6763⟩⟩]⟩, (1)⟩)

def exact68718RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6763⟩⟩]⟩, (1)⟩]

theorem exact68718RawTermsValid :
    exact68718RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68718 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7181⟩⟩) exact68718RawTerms .large 68716 .exactZero (none)

def event68719 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9607⟩⟩) 0 ⟨7181⟩ 68718

def event68720 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9607⟩⟩) 1 ⟨9606⟩ 68713

def event68721 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨9607⟩⟩) (.sum [.predecessor 0 68719 .coefficient, .predecessor 1 68720 .coefficient])

def exact68722RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6763⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9605⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact68722RawTermsValid :
    exact68722RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68722 : Event := .resultExact (⟨.program ⟨214⟩, ⟨9607⟩⟩) exact68722RawTerms .large 68721 .exactZero (none)

def event68723 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9608⟩⟩) 0 ⟨9607⟩ 68722

def event68724 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9608⟩⟩) 1 ⟨77⟩ 10012

def event68725 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨9608⟩⟩) (.sum [.predecessor 0 68723 .coefficient, .predecessor 1 68724 .coefficient])

def event68726 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨9608⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨77⟩⟩]⟩) [⟨.result 10012 .coefficient, false, none⟩])

def event68727 : Event := .survivorFold (1) 68726

def exact68728RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6763⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9605⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact68728RawTermsValid :
    exact68728RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68728 : Event := .resultExact (⟨.program ⟨214⟩, ⟨9608⟩⟩) exact68728RawTerms .large 68725 (.finite 26) (some (68726))

def event68729 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9609⟩⟩) 0 ⟨9608⟩ 68728

def event68730 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9609⟩⟩) 1 ⟨7862⟩ 10009

def event68731 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨9609⟩⟩) (.product (.predecessor 0 68729 .coefficient) (.predecessor 1 68730 .coefficient) (⟨false, false, none, none, none⟩))

def event68732 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨9609⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨7861⟩⟩]⟩) [⟨.result 10005 .coefficient, false, none⟩])

def event68733 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨9609⟩⟩) (.product (.result 68728 .summary) (.transfer 68732) (⟨false, false, none, none, none⟩))

def event68734 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨9609⟩⟩, .operator (⟨68728, 1⟩, ⟨10009, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9605⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7861⟩⟩]⟩, (-1)⟩)

def event68735 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨9609⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9605⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7861⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨7861⟩⟩) ⟨6783⟩ 9979)

def event68736 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨9609⟩⟩, .relation 68735 0, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9605⟩⟩], [⟨.program ⟨214⟩, ⟨6783⟩⟩]⟩, (-1)⟩)

def event68737 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨9609⟩⟩, .operator (⟨68728, 0⟩, ⟨10009, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6763⟩⟩, ⟨.program ⟨214⟩, ⟨7861⟩⟩]⟩, (1)⟩)

def exact68738RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6763⟩⟩, ⟨.program ⟨214⟩, ⟨7861⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9605⟩⟩], [⟨.program ⟨214⟩, ⟨6783⟩⟩]⟩, (-1)⟩]

theorem exact68738RawTermsValid :
    exact68738RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68738 : Event := .resultExact (⟨.program ⟨214⟩, ⟨9609⟩⟩) exact68738RawTerms .large 68731 (.finite 95420416) (some (68733))

def event68739 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11760⟩⟩) 0 ⟨9609⟩ 68738

def event68740 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11760⟩⟩) 1 ⟨11759⟩ 68708

def event68741 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11760⟩⟩) (.sum [.predecessor 0 68739 .coefficient, .predecessor 1 68740 .coefficient])

def event68742 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨11760⟩⟩, .operator (⟨68738, 1⟩, ⟨68708, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9605⟩⟩], [⟨.program ⟨214⟩, ⟨6783⟩⟩]⟩, (1)⟩)

def event68743 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11760⟩⟩) (.sum [.result 68738 .summary, .result 68708 .summary])

def exact68744RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6763⟩⟩, ⟨.program ⟨214⟩, ⟨7861⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9605⟩⟩, ⟨.program ⟨214⟩, ⟨11753⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact68744RawTermsValid :
    exact68744RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68744 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11760⟩⟩) exact68744RawTerms .large 68741 (.finite 95445376) (some (68743))

def event68745 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25138⟩⟩) 0 ⟨11760⟩ 68744

def event68746 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25138⟩⟩) 1 ⟨25137⟩ 68680

def event68747 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25138⟩⟩) (.product (.predecessor 0 68745 .coefficient) (.predecessor 1 68746 .coefficient) (⟨false, false, none, none, none⟩))

def event68748 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25138⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨25137⟩⟩]⟩) [⟨.result 68680 .coefficient, false, none⟩])

def event68749 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25138⟩⟩) (.product (.result 68744 .summary) (.transfer 68748) (⟨false, false, none, none, none⟩))

def event68750 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25138⟩⟩, .operator (⟨68744, 1⟩, ⟨68680, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9605⟩⟩, ⟨.program ⟨214⟩, ⟨11753⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25137⟩⟩]⟩, (-1)⟩)

def event68751 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨25138⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9605⟩⟩, ⟨.program ⟨214⟩, ⟨11753⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25137⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨25137⟩⟩) ⟨23078⟩ 68677)

def event68752 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25138⟩⟩, .relation 68751 0, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9605⟩⟩, ⟨.program ⟨214⟩, ⟨11753⟩⟩], [⟨.program ⟨214⟩, ⟨23078⟩⟩]⟩, (-1)⟩)

def event68753 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25138⟩⟩, .operator (⟨68744, 0⟩, ⟨68680, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6763⟩⟩, ⟨.program ⟨214⟩, ⟨7861⟩⟩, ⟨.program ⟨214⟩, ⟨25137⟩⟩]⟩, (1)⟩)

def exact68754RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6763⟩⟩, ⟨.program ⟨214⟩, ⟨7861⟩⟩, ⟨.program ⟨214⟩, ⟨25137⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9605⟩⟩, ⟨.program ⟨214⟩, ⟨11753⟩⟩], [⟨.program ⟨214⟩, ⟨23078⟩⟩]⟩, (-1)⟩]

theorem exact68754RawTermsValid :
    exact68754RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68754 : Event := .resultExact (⟨.program ⟨214⟩, ⟨25138⟩⟩) exact68754RawTerms .large 68747 (.finite 350286057046016) (some (68749))

def event68755 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19740⟩⟩) 0 ⟨11755⟩ 3258

def event68756 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19740⟩⟩) (.authority (.relationPreimageSource ⟨18⟩))

def exact68757RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨19740⟩⟩]⟩, (1)⟩]

theorem exact68757RawTermsValid :
    exact68757RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68757 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19740⟩⟩) exact68757RawTerms (.finite 136065468) 68756 .exactZero (none)

def event68758 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19742⟩⟩) 0 ⟨19740⟩ 68757

def event68759 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19742⟩⟩) 1 ⟨2348⟩ 4

def event68760 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19742⟩⟩) (.scale (.predecessor 0 68758 .coefficient) (.value (.predecessor 1 68759 .coefficient)))

def exact68761RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨19740⟩⟩]⟩, (1)⟩]

theorem exact68761RawTermsValid :
    exact68761RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68761 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19742⟩⟩) exact68761RawTerms (.finite 136065468) 68760 .exactZero (none)

def event68762 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19743⟩⟩) 0 ⟨5535⟩ 65387

def event68763 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19743⟩⟩) 1 ⟨19742⟩ 68761

def event68764 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19743⟩⟩) (.product (.predecessor 0 68762 .coefficient) (.predecessor 1 68763 .coefficient) (⟨false, false, none, none, none⟩))

def event68765 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19743⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨19740⟩⟩]⟩) [⟨.result 68757 .coefficient, false, none⟩])

def event68766 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19743⟩⟩) (.product (.result 65387 .summary) (.transfer 68765) (⟨false, false, none, none, none⟩))

def event68767 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19743⟩⟩, .operator (⟨65387, 0⟩, ⟨68761, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19740⟩⟩]⟩, (1)⟩)

def event68768 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨19741⟩⟩)

def event68769 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event68770 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event68771 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨360⟩⟩) (.authority (.operator))

def event68772 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨360⟩⟩) (.finite 2)

def event68773 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event68774 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event68775 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event68776 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event68777 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 68776

def event68778 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 68774

def event68779 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 68777 .coefficient) (.value (.predecessor 1 68778 .coefficient)))

def event68780 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event68781 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5504⟩⟩) 0 ⟨5503⟩ 68780

def event68782 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5504⟩⟩) 1 ⟨360⟩ 68772

def event68783 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5504⟩⟩) (.sum [.predecessor 0 68781 .coefficient, .predecessor 1 68782 .coefficient])

def event68784 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5504⟩⟩) (.finite 219)

def event68785 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5530⟩⟩) 0 ⟨5504⟩ 68784

def event68786 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5530⟩⟩) 1 ⟨961⟩ 68770

def event68787 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5530⟩⟩) (.identity (.predecessor 1 68786 .coefficient))

def event68788 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5530⟩⟩) (.finite 224)

def event68789 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11753⟩⟩) 0 ⟨5530⟩ 68788

def event68790 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11753⟩⟩) (.authority (.programFamilyFact))

def exact68791RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11753⟩⟩], []⟩, (1)⟩]

theorem exact68791RawTermsValid :
    exact68791RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68791 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11753⟩⟩) exact68791RawTerms (.finite 30) 68790 .exactZero (none)

def event68792 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9605⟩⟩) 0 ⟨5530⟩ 68788

def event68793 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨9605⟩⟩) (.authority (.programFamilyFact))

def exact68794RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨9605⟩⟩], []⟩, (1)⟩]

theorem exact68794RawTermsValid :
    exact68794RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68794 : Event := .resultExact (⟨.program ⟨214⟩, ⟨9605⟩⟩) exact68794RawTerms (.finite 30) 68793 .exactZero (none)

def event68795 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11754⟩⟩) 0 ⟨9605⟩ 68794

def event68796 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11754⟩⟩) 1 ⟨11753⟩ 68791

def event68797 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11754⟩⟩) (.product (.predecessor 0 68795 .coefficient) (.predecessor 1 68796 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event68798 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11754⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨9605⟩⟩, ⟨.program ⟨214⟩, ⟨11753⟩⟩], []⟩) [⟨.result 68794 .coefficient, true, some 1⟩, ⟨.result 68791 .coefficient, true, some 1⟩])

def event68799 : Event := .survivorFold (1) 68798

def exact68800RawTerms : List Term := []

theorem exact68800RawTermsValid :
    exact68800RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68800 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11754⟩⟩) exact68800RawTerms (.finite 900) 68797 (.finite 900) (some (68798))

def event68801 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11755⟩⟩) 0 ⟨11754⟩ 68800

def event68802 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11755⟩⟩) (.identity (.predecessor 0 68801 .coefficient))

def event68803 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨11755⟩⟩) (.finite 900)

def event68804 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19740⟩⟩) 0 ⟨11755⟩ 68803

def event68805 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19740⟩⟩) (.authority (.relationPreimageSource ⟨18⟩))

def exact68806RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨19740⟩⟩]⟩, (1)⟩]

theorem exact68806RawTermsValid :
    exact68806RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68806 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19740⟩⟩) exact68806RawTerms (.finite 136065468) 68805 .exactZero (none)

def event68807 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6⟩⟩) (.authority (.operator))

def exact68808RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩]⟩, (1)⟩]

theorem exact68808RawTermsValid :
    exact68808RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68808 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6⟩⟩) exact68808RawTerms .large 68807 .exactZero (none)

def event68809 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19741⟩⟩) 0 ⟨6⟩ 68808

def event68810 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19741⟩⟩) 1 ⟨19740⟩ 68806

def event68811 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19741⟩⟩) (.product (.predecessor 0 68809 .coefficient) (.predecessor 1 68810 .coefficient) (⟨false, false, none, none, none⟩))

def event68812 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19741⟩⟩, .operator (⟨68808, 0⟩, ⟨68806, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19740⟩⟩]⟩, (1)⟩)

def exact68813RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19740⟩⟩]⟩, (1)⟩]

theorem exact68813RawTermsValid :
    exact68813RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68813 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19741⟩⟩) exact68813RawTerms .large 68811 .exactZero (none)

def event68814 : Event := .preFoldPolynomial 68813 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19740⟩⟩]⟩, (1)⟩] .exactZero none

def exact68815RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19740⟩⟩]⟩, (1)⟩]

def event68815 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨19741⟩⟩) 68814 exact68815RawTerms .large 68811 .exactZero (none)

def event68816 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨25141⟩⟩)

def event68817 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event68818 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event68819 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨360⟩⟩) (.authority (.operator))

def event68820 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨360⟩⟩) (.finite 2)

def event68821 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event68822 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event68823 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event68824 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event68825 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 68824

def event68826 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 68822

def event68827 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 68825 .coefficient) (.value (.predecessor 1 68826 .coefficient)))

def event68828 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event68829 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5504⟩⟩) 0 ⟨5503⟩ 68828

def event68830 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5504⟩⟩) 1 ⟨360⟩ 68820

def event68831 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5504⟩⟩) (.sum [.predecessor 0 68829 .coefficient, .predecessor 1 68830 .coefficient])

def event68832 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5504⟩⟩) (.finite 219)

def event68833 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5530⟩⟩) 0 ⟨5504⟩ 68832

def event68834 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5530⟩⟩) 1 ⟨961⟩ 68818

def event68835 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5530⟩⟩) (.identity (.predecessor 1 68834 .coefficient))

def event68836 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5530⟩⟩) (.finite 224)

def event68837 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11753⟩⟩) 0 ⟨5530⟩ 68836

def event68838 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11753⟩⟩) (.authority (.programFamilyFact))

def exact68839RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11753⟩⟩], []⟩, (1)⟩]

theorem exact68839RawTermsValid :
    exact68839RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68839 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11753⟩⟩) exact68839RawTerms (.finite 30) 68838 .exactZero (none)

def event68840 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9605⟩⟩) 0 ⟨5530⟩ 68836

def event68841 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨9605⟩⟩) (.authority (.programFamilyFact))

def exact68842RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨9605⟩⟩], []⟩, (1)⟩]

theorem exact68842RawTermsValid :
    exact68842RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68842 : Event := .resultExact (⟨.program ⟨214⟩, ⟨9605⟩⟩) exact68842RawTerms (.finite 30) 68841 .exactZero (none)

def event68843 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11754⟩⟩) 0 ⟨9605⟩ 68842

def event68844 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11754⟩⟩) 1 ⟨11753⟩ 68839

def event68845 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11754⟩⟩) (.product (.predecessor 0 68843 .coefficient) (.predecessor 1 68844 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event68846 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨11754⟩⟩, .operator (⟨68842, 0⟩, ⟨68839, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨9605⟩⟩, ⟨.program ⟨214⟩, ⟨11753⟩⟩], []⟩, (1)⟩)

def exact68847RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨9605⟩⟩, ⟨.program ⟨214⟩, ⟨11753⟩⟩], []⟩, (1)⟩]

theorem exact68847RawTermsValid :
    exact68847RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68847 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11754⟩⟩) exact68847RawTerms (.finite 900) 68845 .exactZero (none)

def event68848 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11755⟩⟩) 0 ⟨11754⟩ 68847

def event68849 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11755⟩⟩) (.identity (.predecessor 0 68848 .coefficient))

def event68850 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨11755⟩⟩) (.finite 900)

def event68851 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23077⟩⟩) 0 ⟨11755⟩ 68850

def event68852 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23077⟩⟩) (.authority (.programFamilyFact))

def event68853 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨23077⟩⟩) (.finite 3720)

def event68854 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨6689⟩⟩) .missing

def event68855 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23078⟩⟩) 0 ⟨6689⟩ 68854

def event68856 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23078⟩⟩) 1 ⟨23077⟩ 68853

def event68857 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23078⟩⟩) (.authority (.operator))

def exact68858RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨23078⟩⟩]⟩, (1)⟩]

theorem exact68858RawTermsValid :
    exact68858RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68858 : Event := .resultExact (⟨.program ⟨214⟩, ⟨23078⟩⟩) exact68858RawTerms .large 68857 .exactZero (none)

def event68859 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25137⟩⟩) 0 ⟨23078⟩ 68858

def event68860 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25137⟩⟩) (.authority (.operator))

def exact68861RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨25137⟩⟩]⟩, (1)⟩]

theorem exact68861RawTermsValid :
    exact68861RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68861 : Event := .resultExact (⟨.program ⟨214⟩, ⟨25137⟩⟩) exact68861RawTerms (.finite 8192) 68860 .exactZero (none)

def event68862 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨110⟩⟩) (.authority (.operator))

def event68863 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨110⟩⟩) .exactZero

def eventLeaf4288 : Array AnnotatedEvent := #[
  { event := event68608
    frameStart := 68543 },
  { event := event68609
    frameStart := 68543 },
  { event := event68610
    frameStart := 68543 },
  { event := event68611
    frameStart := 68543 },
  { event := event68612
    frameStart := 68543 },
  { event := event68613
    frameStart := 68543 },
  { event := event68614
    frameStart := 68543 },
  { event := event68615
    frameStart := 68543 },
  { event := event68616
    frameStart := 68543 },
  { event := event68617
    frameStart := 68543 },
  { event := event68618
    frameStart := 68543 },
  { event := event68619
    frameStart := 68543 },
  { event := event68620
    frameStart := 68543 },
  { event := event68621
    frameStart := 68543 },
  { event := event68622
    frameStart := 68543 },
  { event := event68623
    frameStart := 68543 }
]

def eventLeaf4289 : Array AnnotatedEvent := #[
  { event := event68624
    frameStart := 68543 },
  { event := event68625
    frameStart := 68543 },
  { event := event68626
    frameStart := 68543 },
  { event := event68627
    frameStart := 68543 },
  { event := event68628
    frameStart := 68543 },
  { event := event68629
    frameStart := 68543 },
  { event := event68630
    frameStart := 68543 },
  { event := event68631
    frameStart := 68543 },
  { event := event68632
    frameStart := 68543 },
  { event := event68633
    frameStart := 68543 },
  { event := event68634
    frameStart := 68543 },
  { event := event68635
    frameStart := 68543 },
  { event := event68636
    frameStart := 68543 },
  { event := event68637
    frameStart := 68543 },
  { event := event68638
    frameStart := 68543 },
  { event := event68639
    frameStart := 68543 }
]

def eventLeaf4290 : Array AnnotatedEvent := #[
  { event := event68640
    frameStart := 68543 },
  { event := event68641
    frameStart := 68543 },
  { event := event68642
    frameStart := 68543 },
  { event := event68643
    frameStart := 68543 },
  { event := event68644
    frameStart := 68543 },
  { event := event68645
    frameStart := 68543 },
  { event := event68646
    frameStart := 68543 },
  { event := event68647
    frameStart := 0 },
  { event := event68648
    frameStart := 0 },
  { event := event68649
    frameStart := 0 },
  { event := event68650
    frameStart := 0 },
  { event := event68651
    frameStart := 0 },
  { event := event68652
    frameStart := 0 },
  { event := event68653
    frameStart := 0 },
  { event := event68654
    frameStart := 0 },
  { event := event68655
    frameStart := 0 }
]

def eventLeaf4291 : Array AnnotatedEvent := #[
  { event := event68656
    frameStart := 0 },
  { event := event68657
    frameStart := 0 },
  { event := event68658
    frameStart := 0 },
  { event := event68659
    frameStart := 0 },
  { event := event68660
    frameStart := 0 },
  { event := event68661
    frameStart := 0 },
  { event := event68662
    frameStart := 0 },
  { event := event68663
    frameStart := 0 },
  { event := event68664
    frameStart := 0 },
  { event := event68665
    frameStart := 0 },
  { event := event68666
    frameStart := 0 },
  { event := event68667
    frameStart := 0 },
  { event := event68668
    frameStart := 0 },
  { event := event68669
    frameStart := 0 },
  { event := event68670
    frameStart := 0 },
  { event := event68671
    frameStart := 0 }
]

def eventLeaf4292 : Array AnnotatedEvent := #[
  { event := event68672
    frameStart := 0 },
  { event := event68673
    frameStart := 0 },
  { event := event68674
    frameStart := 0 },
  { event := event68675
    frameStart := 0 },
  { event := event68676
    frameStart := 0 },
  { event := event68677
    frameStart := 0 },
  { event := event68678
    frameStart := 0 },
  { event := event68679
    frameStart := 0 },
  { event := event68680
    frameStart := 0 },
  { event := event68681
    frameStart := 0 },
  { event := event68682
    frameStart := 0 },
  { event := event68683
    frameStart := 0 },
  { event := event68684
    frameStart := 0 },
  { event := event68685
    frameStart := 0 },
  { event := event68686
    frameStart := 0 },
  { event := event68687
    frameStart := 0 }
]

def eventLeaf4293 : Array AnnotatedEvent := #[
  { event := event68688
    frameStart := 0 },
  { event := event68689
    frameStart := 0 },
  { event := event68690
    frameStart := 0 },
  { event := event68691
    frameStart := 0 },
  { event := event68692
    frameStart := 0 },
  { event := event68693
    frameStart := 0 },
  { event := event68694
    frameStart := 0 },
  { event := event68695
    frameStart := 0 },
  { event := event68696
    frameStart := 0 },
  { event := event68697
    frameStart := 0 },
  { event := event68698
    frameStart := 0 },
  { event := event68699
    frameStart := 0 },
  { event := event68700
    frameStart := 0 },
  { event := event68701
    frameStart := 0 },
  { event := event68702
    frameStart := 0 },
  { event := event68703
    frameStart := 0 }
]

def eventLeaf4294 : Array AnnotatedEvent := #[
  { event := event68704
    frameStart := 0 },
  { event := event68705
    frameStart := 0 },
  { event := event68706
    frameStart := 0 },
  { event := event68707
    frameStart := 0 },
  { event := event68708
    frameStart := 0 },
  { event := event68709
    frameStart := 0 },
  { event := event68710
    frameStart := 0 },
  { event := event68711
    frameStart := 0 },
  { event := event68712
    frameStart := 0 },
  { event := event68713
    frameStart := 0 },
  { event := event68714
    frameStart := 0 },
  { event := event68715
    frameStart := 0 },
  { event := event68716
    frameStart := 0 },
  { event := event68717
    frameStart := 0 },
  { event := event68718
    frameStart := 0 },
  { event := event68719
    frameStart := 0 }
]

def eventLeaf4295 : Array AnnotatedEvent := #[
  { event := event68720
    frameStart := 0 },
  { event := event68721
    frameStart := 0 },
  { event := event68722
    frameStart := 0 },
  { event := event68723
    frameStart := 0 },
  { event := event68724
    frameStart := 0 },
  { event := event68725
    frameStart := 0 },
  { event := event68726
    frameStart := 0 },
  { event := event68727
    frameStart := 0 },
  { event := event68728
    frameStart := 0 },
  { event := event68729
    frameStart := 0 },
  { event := event68730
    frameStart := 0 },
  { event := event68731
    frameStart := 0 },
  { event := event68732
    frameStart := 0 },
  { event := event68733
    frameStart := 0 },
  { event := event68734
    frameStart := 0 },
  { event := event68735
    frameStart := 0 }
]

def eventLeaf4296 : Array AnnotatedEvent := #[
  { event := event68736
    frameStart := 0 },
  { event := event68737
    frameStart := 0 },
  { event := event68738
    frameStart := 0 },
  { event := event68739
    frameStart := 0 },
  { event := event68740
    frameStart := 0 },
  { event := event68741
    frameStart := 0 },
  { event := event68742
    frameStart := 0 },
  { event := event68743
    frameStart := 0 },
  { event := event68744
    frameStart := 0 },
  { event := event68745
    frameStart := 0 },
  { event := event68746
    frameStart := 0 },
  { event := event68747
    frameStart := 0 },
  { event := event68748
    frameStart := 0 },
  { event := event68749
    frameStart := 0 },
  { event := event68750
    frameStart := 0 },
  { event := event68751
    frameStart := 0 }
]

def eventLeaf4297 : Array AnnotatedEvent := #[
  { event := event68752
    frameStart := 0 },
  { event := event68753
    frameStart := 0 },
  { event := event68754
    frameStart := 0 },
  { event := event68755
    frameStart := 0 },
  { event := event68756
    frameStart := 0 },
  { event := event68757
    frameStart := 0 },
  { event := event68758
    frameStart := 0 },
  { event := event68759
    frameStart := 0 },
  { event := event68760
    frameStart := 0 },
  { event := event68761
    frameStart := 0 },
  { event := event68762
    frameStart := 0 },
  { event := event68763
    frameStart := 0 },
  { event := event68764
    frameStart := 0 },
  { event := event68765
    frameStart := 0 },
  { event := event68766
    frameStart := 0 },
  { event := event68767
    frameStart := 0 }
]

def eventLeaf4298 : Array AnnotatedEvent := #[
  { event := event68768
    frameStart := 68768 },
  { event := event68769
    frameStart := 68768 },
  { event := event68770
    frameStart := 68768 },
  { event := event68771
    frameStart := 68768 },
  { event := event68772
    frameStart := 68768 },
  { event := event68773
    frameStart := 68768 },
  { event := event68774
    frameStart := 68768 },
  { event := event68775
    frameStart := 68768 },
  { event := event68776
    frameStart := 68768 },
  { event := event68777
    frameStart := 68768 },
  { event := event68778
    frameStart := 68768 },
  { event := event68779
    frameStart := 68768 },
  { event := event68780
    frameStart := 68768 },
  { event := event68781
    frameStart := 68768 },
  { event := event68782
    frameStart := 68768 },
  { event := event68783
    frameStart := 68768 }
]

def eventLeaf4299 : Array AnnotatedEvent := #[
  { event := event68784
    frameStart := 68768 },
  { event := event68785
    frameStart := 68768 },
  { event := event68786
    frameStart := 68768 },
  { event := event68787
    frameStart := 68768 },
  { event := event68788
    frameStart := 68768 },
  { event := event68789
    frameStart := 68768 },
  { event := event68790
    frameStart := 68768 },
  { event := event68791
    frameStart := 68768 },
  { event := event68792
    frameStart := 68768 },
  { event := event68793
    frameStart := 68768 },
  { event := event68794
    frameStart := 68768 },
  { event := event68795
    frameStart := 68768 },
  { event := event68796
    frameStart := 68768 },
  { event := event68797
    frameStart := 68768 },
  { event := event68798
    frameStart := 68768 },
  { event := event68799
    frameStart := 68768 }
]

def eventLeaf4300 : Array AnnotatedEvent := #[
  { event := event68800
    frameStart := 68768 },
  { event := event68801
    frameStart := 68768 },
  { event := event68802
    frameStart := 68768 },
  { event := event68803
    frameStart := 68768 },
  { event := event68804
    frameStart := 68768 },
  { event := event68805
    frameStart := 68768 },
  { event := event68806
    frameStart := 68768 },
  { event := event68807
    frameStart := 68768 },
  { event := event68808
    frameStart := 68768 },
  { event := event68809
    frameStart := 68768 },
  { event := event68810
    frameStart := 68768 },
  { event := event68811
    frameStart := 68768 },
  { event := event68812
    frameStart := 68768 },
  { event := event68813
    frameStart := 68768 },
  { event := event68814
    frameStart := 68768 },
  { event := event68815
    frameStart := 68768 }
]

def eventLeaf4301 : Array AnnotatedEvent := #[
  { event := event68816
    frameStart := 68816 },
  { event := event68817
    frameStart := 68816 },
  { event := event68818
    frameStart := 68816 },
  { event := event68819
    frameStart := 68816 },
  { event := event68820
    frameStart := 68816 },
  { event := event68821
    frameStart := 68816 },
  { event := event68822
    frameStart := 68816 },
  { event := event68823
    frameStart := 68816 },
  { event := event68824
    frameStart := 68816 },
  { event := event68825
    frameStart := 68816 },
  { event := event68826
    frameStart := 68816 },
  { event := event68827
    frameStart := 68816 },
  { event := event68828
    frameStart := 68816 },
  { event := event68829
    frameStart := 68816 },
  { event := event68830
    frameStart := 68816 },
  { event := event68831
    frameStart := 68816 }
]

def eventLeaf4302 : Array AnnotatedEvent := #[
  { event := event68832
    frameStart := 68816 },
  { event := event68833
    frameStart := 68816 },
  { event := event68834
    frameStart := 68816 },
  { event := event68835
    frameStart := 68816 },
  { event := event68836
    frameStart := 68816 },
  { event := event68837
    frameStart := 68816 },
  { event := event68838
    frameStart := 68816 },
  { event := event68839
    frameStart := 68816 },
  { event := event68840
    frameStart := 68816 },
  { event := event68841
    frameStart := 68816 },
  { event := event68842
    frameStart := 68816 },
  { event := event68843
    frameStart := 68816 },
  { event := event68844
    frameStart := 68816 },
  { event := event68845
    frameStart := 68816 },
  { event := event68846
    frameStart := 68816 },
  { event := event68847
    frameStart := 68816 }
]

def eventLeaf4303 : Array AnnotatedEvent := #[
  { event := event68848
    frameStart := 68816 },
  { event := event68849
    frameStart := 68816 },
  { event := event68850
    frameStart := 68816 },
  { event := event68851
    frameStart := 68816 },
  { event := event68852
    frameStart := 68816 },
  { event := event68853
    frameStart := 68816 },
  { event := event68854
    frameStart := 68816 },
  { event := event68855
    frameStart := 68816 },
  { event := event68856
    frameStart := 68816 },
  { event := event68857
    frameStart := 68816 },
  { event := event68858
    frameStart := 68816 },
  { event := event68859
    frameStart := 68816 },
  { event := event68860
    frameStart := 68816 },
  { event := event68861
    frameStart := 68816 },
  { event := event68862
    frameStart := 68816 },
  { event := event68863
    frameStart := 68816 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events268
