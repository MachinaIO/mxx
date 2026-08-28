import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events268

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event68608 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event68609 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event68610 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10691⟩⟩) (.authority (.operator))

def event68611 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10691⟩⟩) (.finite 16)

def event68612 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event68613 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event68614 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event68615 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event68616 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 68615

def event68617 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 68613

def event68618 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 68616 .coefficient) (.value (.predecessor 1 68617 .coefficient)))

def event68619 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event68620 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10693⟩⟩) 0 ⟨392⟩ 68619

def event68621 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10693⟩⟩) 1 ⟨10691⟩ 68611

def event68622 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10693⟩⟩) (.sum [.predecessor 0 68620 .coefficient, .predecessor 1 68621 .coefficient])

def event68623 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10693⟩⟩) (.finite 655356)

def event68624 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10749⟩⟩) 0 ⟨10693⟩ 68623

def event68625 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10749⟩⟩) 1 ⟨5426⟩ 68609

def event68626 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10749⟩⟩) (.identity (.predecessor 1 68625 .coefficient))

def event68627 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10749⟩⟩) (.finite 655360)

def event68628 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21662⟩⟩) 0 ⟨10749⟩ 68627

def event68629 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨21662⟩⟩) (.authority (.programFamilyFact))

def exact68630RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨21662⟩⟩], []⟩, (1)⟩]

theorem exact68630RawTermsValid :
    exact68630RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68630 : Event := .resultExact (⟨.program ⟨257⟩, ⟨21662⟩⟩) exact68630RawTerms (.finite 4) 68629 .exactZero (none)

def event68631 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21206⟩⟩) 0 ⟨10749⟩ 68627

def event68632 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨21206⟩⟩) (.authority (.programFamilyFact))

def exact68633RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨21206⟩⟩], []⟩, (1)⟩]

theorem exact68633RawTermsValid :
    exact68633RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68633 : Event := .resultExact (⟨.program ⟨257⟩, ⟨21206⟩⟩) exact68633RawTerms (.finite 4) 68632 .exactZero (none)

def event68634 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21663⟩⟩) 0 ⟨21206⟩ 68633

def event68635 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21663⟩⟩) 1 ⟨21662⟩ 68630

def event68636 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨21663⟩⟩) (.product (.predecessor 0 68634 .coefficient) (.predecessor 1 68635 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event68637 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨21663⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨21206⟩⟩, ⟨.program ⟨257⟩, ⟨21662⟩⟩], []⟩) [⟨.result 68633 .coefficient, true, some 1⟩, ⟨.result 68630 .coefficient, true, some 1⟩])

def event68638 : Event := .survivorFold (1) 68637

def exact68639RawTerms : List Term := []

theorem exact68639RawTermsValid :
    exact68639RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68639 : Event := .resultExact (⟨.program ⟨257⟩, ⟨21663⟩⟩) exact68639RawTerms (.finite 16) 68636 (.finite 16) (some (68637))

def event68640 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21664⟩⟩) 0 ⟨21663⟩ 68639

def event68641 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨21664⟩⟩) (.identity (.predecessor 0 68640 .coefficient))

def event68642 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨21664⟩⟩) (.finite 16)

def event68643 : Event := .predecessor (⟨.program ⟨257⟩, ⟨22439⟩⟩) 0 ⟨21664⟩ 68642

def event68644 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨22439⟩⟩) (.authority (.relationPreimageSource ⟨38⟩))

def exact68645RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨22439⟩⟩]⟩, (1)⟩]

theorem exact68645RawTermsValid :
    exact68645RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68645 : Event := .resultExact (⟨.program ⟨257⟩, ⟨22439⟩⟩) exact68645RawTerms (.finite 5647228698) 68644 .exactZero (none)

def event68646 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35⟩⟩) (.authority (.operator))

def exact68647RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact68647RawTermsValid :
    exact68647RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68647 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35⟩⟩) exact68647RawTerms .large 68646 .exactZero (none)

def event68648 : Event := .predecessor (⟨.program ⟨257⟩, ⟨22440⟩⟩) 0 ⟨35⟩ 68647

def event68649 : Event := .predecessor (⟨.program ⟨257⟩, ⟨22440⟩⟩) 1 ⟨22439⟩ 68645

def event68650 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨22440⟩⟩) (.product (.predecessor 0 68648 .coefficient) (.predecessor 1 68649 .coefficient) (⟨false, false, none, none, none⟩))

def event68651 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨22440⟩⟩, .operator (⟨68647, 0⟩, ⟨68645, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22439⟩⟩]⟩, (1)⟩)

def exact68652RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22439⟩⟩]⟩, (1)⟩]

theorem exact68652RawTermsValid :
    exact68652RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68652 : Event := .resultExact (⟨.program ⟨257⟩, ⟨22440⟩⟩) exact68652RawTerms .large 68650 .exactZero (none)

def event68653 : Event := .preFoldPolynomial 68652 [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22439⟩⟩]⟩, (1)⟩] .exactZero none

def exact68654RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22439⟩⟩]⟩, (1)⟩]

def event68654 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨22440⟩⟩) 68653 exact68654RawTerms .large 68650 .exactZero (none)

def event68655 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨23520⟩⟩)

def event68656 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event68657 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event68658 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10691⟩⟩) (.authority (.operator))

def event68659 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10691⟩⟩) (.finite 16)

def event68660 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event68661 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event68662 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event68663 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event68664 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 68663

def event68665 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 68661

def event68666 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 68664 .coefficient) (.value (.predecessor 1 68665 .coefficient)))

def event68667 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event68668 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10693⟩⟩) 0 ⟨392⟩ 68667

def event68669 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10693⟩⟩) 1 ⟨10691⟩ 68659

def event68670 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10693⟩⟩) (.sum [.predecessor 0 68668 .coefficient, .predecessor 1 68669 .coefficient])

def event68671 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10693⟩⟩) (.finite 655356)

def event68672 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10749⟩⟩) 0 ⟨10693⟩ 68671

def event68673 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10749⟩⟩) 1 ⟨5426⟩ 68657

def event68674 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10749⟩⟩) (.identity (.predecessor 1 68673 .coefficient))

def event68675 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10749⟩⟩) (.finite 655360)

def event68676 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21662⟩⟩) 0 ⟨10749⟩ 68675

def event68677 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨21662⟩⟩) (.authority (.programFamilyFact))

def exact68678RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨21662⟩⟩], []⟩, (1)⟩]

theorem exact68678RawTermsValid :
    exact68678RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68678 : Event := .resultExact (⟨.program ⟨257⟩, ⟨21662⟩⟩) exact68678RawTerms (.finite 4) 68677 .exactZero (none)

def event68679 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21206⟩⟩) 0 ⟨10749⟩ 68675

def event68680 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨21206⟩⟩) (.authority (.programFamilyFact))

def exact68681RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨21206⟩⟩], []⟩, (1)⟩]

theorem exact68681RawTermsValid :
    exact68681RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68681 : Event := .resultExact (⟨.program ⟨257⟩, ⟨21206⟩⟩) exact68681RawTerms (.finite 4) 68680 .exactZero (none)

def event68682 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21663⟩⟩) 0 ⟨21206⟩ 68681

def event68683 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21663⟩⟩) 1 ⟨21662⟩ 68678

def event68684 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨21663⟩⟩) (.product (.predecessor 0 68682 .coefficient) (.predecessor 1 68683 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event68685 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨21663⟩⟩, .operator (⟨68681, 0⟩, ⟨68678, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨21206⟩⟩, ⟨.program ⟨257⟩, ⟨21662⟩⟩], []⟩, (1)⟩)

def exact68686RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨21206⟩⟩, ⟨.program ⟨257⟩, ⟨21662⟩⟩], []⟩, (1)⟩]

theorem exact68686RawTermsValid :
    exact68686RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68686 : Event := .resultExact (⟨.program ⟨257⟩, ⟨21663⟩⟩) exact68686RawTerms (.finite 16) 68684 .exactZero (none)

def event68687 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21664⟩⟩) 0 ⟨21663⟩ 68686

def event68688 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨21664⟩⟩) (.identity (.predecessor 0 68687 .coefficient))

def event68689 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨21664⟩⟩) (.finite 16)

def event68690 : Event := .predecessor (⟨.program ⟨257⟩, ⟨22970⟩⟩) 0 ⟨21664⟩ 68689

def event68691 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨22970⟩⟩) (.authority (.programFamilyFact))

def event68692 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨22970⟩⟩) (.finite 3720)

def event68693 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨7177⟩⟩) .missing

def event68694 : Event := .predecessor (⟨.program ⟨257⟩, ⟨22971⟩⟩) 0 ⟨7177⟩ 68693

def event68695 : Event := .predecessor (⟨.program ⟨257⟩, ⟨22971⟩⟩) 1 ⟨22970⟩ 68692

def event68696 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨22971⟩⟩) (.authority (.operator))

def exact68697RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨22971⟩⟩]⟩, (1)⟩]

theorem exact68697RawTermsValid :
    exact68697RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68697 : Event := .resultExact (⟨.program ⟨257⟩, ⟨22971⟩⟩) exact68697RawTerms .large 68696 .exactZero (none)

def event68698 : Event := .predecessor (⟨.program ⟨257⟩, ⟨23516⟩⟩) 0 ⟨22971⟩ 68697

def event68699 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨23516⟩⟩) (.authority (.operator))

def exact68700RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨23516⟩⟩]⟩, (1)⟩]

theorem exact68700RawTermsValid :
    exact68700RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68700 : Event := .resultExact (⟨.program ⟨257⟩, ⟨23516⟩⟩) exact68700RawTerms (.finite 8192) 68699 .exactZero (none)

def event68701 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event68702 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event68703 : Event := .predecessor (⟨.program ⟨257⟩, ⟨23234⟩⟩) 0 ⟨21664⟩ 68689

def event68704 : Event := .predecessor (⟨.program ⟨257⟩, ⟨23234⟩⟩) 1 ⟨136⟩ 68702

def event68705 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨23234⟩⟩) (.sum [.predecessor 0 68703 .coefficient, .predecessor 1 68704 .coefficient])

def event68706 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨23234⟩⟩) (.finite 16)

def event68707 : Event := .predecessor (⟨.program ⟨257⟩, ⟨23235⟩⟩) 0 ⟨23234⟩ 68706

def event68708 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨23235⟩⟩) (.identity (.predecessor 0 68707 .coefficient))

def exact68709RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨21206⟩⟩, ⟨.program ⟨257⟩, ⟨21662⟩⟩], []⟩, (1)⟩]

theorem exact68709RawTermsValid :
    exact68709RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68709 : Event := .resultExact (⟨.program ⟨257⟩, ⟨23235⟩⟩) exact68709RawTerms (.finite 16) 68708 .exactZero (none)

def event68710 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact68711RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact68711RawTermsValid :
    exact68711RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68711 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact68711RawTerms .large 68710 .exactZero (none)

def event68712 : Event := .predecessor (⟨.program ⟨257⟩, ⟨23236⟩⟩) 0 ⟨6908⟩ 68711

def event68713 : Event := .predecessor (⟨.program ⟨257⟩, ⟨23236⟩⟩) 1 ⟨23235⟩ 68709

def event68714 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨23236⟩⟩) (.product (.predecessor 0 68712 .coefficient) (.predecessor 1 68713 .coefficient) (⟨false, false, none, none, none⟩))

def event68715 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨23236⟩⟩, .operator (⟨68711, 0⟩, ⟨68709, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨21206⟩⟩, ⟨.program ⟨257⟩, ⟨21662⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact68716RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨21206⟩⟩, ⟨.program ⟨257⟩, ⟨21662⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact68716RawTermsValid :
    exact68716RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68716 : Event := .resultExact (⟨.program ⟨257⟩, ⟨23236⟩⟩) exact68716RawTerms .large 68714 .exactZero (none)

def event68717 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨2370⟩⟩) (.authority (.operator))

def event68718 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨2370⟩⟩) (.finite 1)

def event68719 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7178⟩⟩) 0 ⟨7177⟩ 68693

def event68720 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7178⟩⟩) (.authority (.operator))

def exact68721RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7178⟩⟩]⟩, (1)⟩]

theorem exact68721RawTermsValid :
    exact68721RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68721 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7178⟩⟩) exact68721RawTerms .large 68720 .exactZero (none)

def event68722 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7306⟩⟩) 0 ⟨7178⟩ 68721

def event68723 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7306⟩⟩) (.identity (.predecessor 0 68722 .coefficient))

def exact68724RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7306⟩⟩]⟩, (1)⟩]

theorem exact68724RawTermsValid :
    exact68724RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68724 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7306⟩⟩) exact68724RawTerms .large 68723 .exactZero (none)

def event68725 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9574⟩⟩) 0 ⟨7306⟩ 68724

def event68726 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9574⟩⟩) (.authority (.operator))

def exact68727RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨9574⟩⟩]⟩, (1)⟩]

theorem exact68727RawTermsValid :
    exact68727RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68727 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9574⟩⟩) exact68727RawTerms (.finite 8192) 68726 .exactZero (none)

def event68728 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9575⟩⟩) 0 ⟨9574⟩ 68727

def event68729 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9575⟩⟩) 1 ⟨2370⟩ 68718

def event68730 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9575⟩⟩) (.scale (.predecessor 0 68728 .coefficient) (.value (.predecessor 1 68729 .coefficient)))

def exact68731RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨9574⟩⟩]⟩, (1)⟩]

theorem exact68731RawTermsValid :
    exact68731RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68731 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9575⟩⟩) exact68731RawTerms (.finite 8192) 68730 .exactZero (none)

def event68732 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7286⟩⟩) 0 ⟨7178⟩ 68721

def event68733 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7286⟩⟩) (.identity (.predecessor 0 68732 .coefficient))

def exact68734RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7286⟩⟩]⟩, (1)⟩]

theorem exact68734RawTermsValid :
    exact68734RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68734 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7286⟩⟩) exact68734RawTerms .large 68733 .exactZero (none)

def event68735 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9576⟩⟩) 0 ⟨7286⟩ 68734

def event68736 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9576⟩⟩) 1 ⟨9575⟩ 68731

def event68737 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9576⟩⟩) (.product (.predecessor 0 68735 .coefficient) (.predecessor 1 68736 .coefficient) (⟨false, false, none, none, none⟩))

def event68738 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨9576⟩⟩, .operator (⟨68734, 0⟩, ⟨68731, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩]⟩, (1)⟩)

def exact68739RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩]⟩, (1)⟩]

theorem exact68739RawTermsValid :
    exact68739RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68739 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9576⟩⟩) exact68739RawTerms .large 68737 .exactZero (none)

def event68740 : Event := .predecessor (⟨.program ⟨257⟩, ⟨23237⟩⟩) 0 ⟨9576⟩ 68739

def event68741 : Event := .predecessor (⟨.program ⟨257⟩, ⟨23237⟩⟩) 1 ⟨23236⟩ 68716

def event68742 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨23237⟩⟩) (.sum [.predecessor 0 68740 .coefficient, .predecessor 1 68741 .coefficient])

def exact68743RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨21206⟩⟩, ⟨.program ⟨257⟩, ⟨21662⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact68743RawTermsValid :
    exact68743RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68743 : Event := .resultExact (⟨.program ⟨257⟩, ⟨23237⟩⟩) exact68743RawTerms .large 68742 .exactZero (none)

def event68744 : Event := .predecessor (⟨.program ⟨257⟩, ⟨23519⟩⟩) 0 ⟨23237⟩ 68743

def event68745 : Event := .predecessor (⟨.program ⟨257⟩, ⟨23519⟩⟩) 1 ⟨23516⟩ 68700

def event68746 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨23519⟩⟩) (.product (.predecessor 0 68744 .coefficient) (.predecessor 1 68745 .coefficient) (⟨false, false, none, none, none⟩))

def event68747 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨23519⟩⟩, .operator (⟨68743, 0⟩, ⟨68700, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23516⟩⟩]⟩, (1)⟩)

def event68748 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨23519⟩⟩, .operator (⟨68743, 1⟩, ⟨68700, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨21206⟩⟩, ⟨.program ⟨257⟩, ⟨21662⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨23516⟩⟩]⟩, (-1)⟩)

def event68749 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨23519⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨21206⟩⟩, ⟨.program ⟨257⟩, ⟨21662⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨23516⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨23516⟩⟩) ⟨22971⟩ 68697)

def event68750 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨23519⟩⟩, .relation 68749 0, ⟨[⟨.program ⟨257⟩, ⟨21206⟩⟩, ⟨.program ⟨257⟩, ⟨21662⟩⟩], [⟨.program ⟨257⟩, ⟨22971⟩⟩]⟩, (-1)⟩)

def exact68751RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23516⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨21206⟩⟩, ⟨.program ⟨257⟩, ⟨21662⟩⟩], [⟨.program ⟨257⟩, ⟨22971⟩⟩]⟩, (-1)⟩]

theorem exact68751RawTermsValid :
    exact68751RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68751 : Event := .resultExact (⟨.program ⟨257⟩, ⟨23519⟩⟩) exact68751RawTerms .large 68746 .exactZero (none)

def event68752 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21864⟩⟩) 0 ⟨21664⟩ 68689

def event68753 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨21864⟩⟩) (.authority (.programFamilyFact))

def exact68754RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨21864⟩⟩], []⟩, (1)⟩]

theorem exact68754RawTermsValid :
    exact68754RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68754 : Event := .resultExact (⟨.program ⟨257⟩, ⟨21864⟩⟩) exact68754RawTerms (.finite 4) 68753 .exactZero (none)

def event68755 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21866⟩⟩) 0 ⟨6908⟩ 68711

def event68756 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21866⟩⟩) 1 ⟨21864⟩ 68754

def event68757 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨21866⟩⟩) (.product (.predecessor 0 68755 .coefficient) (.predecessor 1 68756 .coefficient) (⟨false, true, none, none, some 1⟩))

def event68758 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨21866⟩⟩, .operator (⟨68711, 0⟩, ⟨68754, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨21864⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact68759RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨21864⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact68759RawTermsValid :
    exact68759RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68759 : Event := .resultExact (⟨.program ⟨257⟩, ⟨21866⟩⟩) exact68759RawTerms .large 68757 .exactZero (none)

def event68760 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7181⟩⟩) 0 ⟨7177⟩ 68693

def event68761 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7181⟩⟩) (.authority (.operator))

def exact68762RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7181⟩⟩]⟩, (1)⟩]

theorem exact68762RawTermsValid :
    exact68762RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68762 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7181⟩⟩) exact68762RawTerms .large 68761 .exactZero (none)

def event68763 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21867⟩⟩) 0 ⟨7181⟩ 68762

def event68764 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21867⟩⟩) 1 ⟨21866⟩ 68759

def event68765 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨21867⟩⟩) (.sum [.predecessor 0 68763 .coefficient, .predecessor 1 68764 .coefficient])

def exact68766RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7181⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨21864⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact68766RawTermsValid :
    exact68766RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68766 : Event := .resultExact (⟨.program ⟨257⟩, ⟨21867⟩⟩) exact68766RawTerms .large 68765 .exactZero (none)

def event68767 : Event := .predecessor (⟨.program ⟨257⟩, ⟨23520⟩⟩) 0 ⟨21867⟩ 68766

def event68768 : Event := .predecessor (⟨.program ⟨257⟩, ⟨23520⟩⟩) 1 ⟨23519⟩ 68751

def event68769 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨23520⟩⟩) (.sum [.predecessor 0 68767 .coefficient, .predecessor 1 68768 .coefficient])

def exact68770RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7181⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23516⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨21206⟩⟩, ⟨.program ⟨257⟩, ⟨21662⟩⟩], [⟨.program ⟨257⟩, ⟨22971⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨21864⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact68770RawTermsValid :
    exact68770RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68770 : Event := .resultExact (⟨.program ⟨257⟩, ⟨23520⟩⟩) exact68770RawTerms .large 68769 .exactZero (none)

def event68771 : Event := .preFoldPolynomial 68770 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7181⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23516⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨21206⟩⟩, ⟨.program ⟨257⟩, ⟨21662⟩⟩], [⟨.program ⟨257⟩, ⟨22971⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨21864⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact68772RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7181⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23516⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨21206⟩⟩, ⟨.program ⟨257⟩, ⟨21662⟩⟩], [⟨.program ⟨257⟩, ⟨22971⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨21864⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event68772 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨23520⟩⟩) 68771 exact68772RawTerms .large 68769 .exactZero (none)

def event68773 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨21664⟩⟩) ⟨⟨60⟩, ⟨38⟩, ⟨135⟩⟩ ⟨68607, 68773⟩

def event68774 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨22442⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22439⟩⟩]⟩) (1) 0 2 (.universal 68773 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22439⟩⟩]⟩) (none) 68772)

def event68775 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨22442⟩⟩, .relation 68774 0, ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7181⟩⟩]⟩, (1)⟩)

def event68776 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨22442⟩⟩, .relation 68774 1, ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23516⟩⟩]⟩, (-1)⟩)

def event68777 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨22442⟩⟩, .relation 68774 2, ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨21206⟩⟩, ⟨.program ⟨257⟩, ⟨21662⟩⟩], [⟨.program ⟨257⟩, ⟨22971⟩⟩]⟩, (1)⟩)

def event68778 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨22442⟩⟩, .relation 68774 3, ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨21864⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact68779RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7181⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23516⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨21206⟩⟩, ⟨.program ⟨257⟩, ⟨21662⟩⟩], [⟨.program ⟨257⟩, ⟨22971⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨21864⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact68779RawTermsValid :
    exact68779RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68779 : Event := .resultExact (⟨.program ⟨257⟩, ⟨22442⟩⟩) exact68779RawTerms .large 68603 (.finite 202072841853861888) (some (68605))

def event68780 : Event := .predecessor (⟨.program ⟨257⟩, ⟨23518⟩⟩) 0 ⟨22442⟩ 68779

def event68781 : Event := .predecessor (⟨.program ⟨257⟩, ⟨23518⟩⟩) 1 ⟨23517⟩ 68593

def event68782 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨23518⟩⟩) (.sum [.predecessor 0 68780 .coefficient, .predecessor 1 68781 .coefficient])

def event68783 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨23518⟩⟩, .operator (⟨68779, 2⟩, ⟨68593, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨21206⟩⟩, ⟨.program ⟨257⟩, ⟨21662⟩⟩], [⟨.program ⟨257⟩, ⟨22971⟩⟩]⟩, (-1)⟩)

def event68784 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨23518⟩⟩, .operator (⟨68779, 1⟩, ⟨68593, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23516⟩⟩]⟩, (1)⟩)

def event68785 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨23518⟩⟩) (.sum [.result 68779 .summary, .result 68593 .summary])

def exact68786RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7181⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨21864⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact68786RawTermsValid :
    exact68786RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68786 : Event := .resultExact (⟨.program ⟨257⟩, ⟨23518⟩⟩) exact68786RawTerms .large 68782 (.finite 2997834576566628384768) (some (68785))

def event68787 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24091⟩⟩) 0 ⟨23518⟩ 68786

def event68788 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24091⟩⟩) 1 ⟨24089⟩ 68509

def event68789 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨24091⟩⟩) (.product (.predecessor 0 68787 .coefficient) (.predecessor 1 68788 .coefficient) (⟨false, false, none, none, none⟩))

def event68790 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨24091⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨24089⟩⟩]⟩) [⟨.result 68509 .coefficient, false, none⟩])

def event68791 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨24091⟩⟩) (.product (.result 68786 .summary) (.transfer 68790) (⟨false, false, none, none, none⟩))

def event68792 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨24091⟩⟩, .operator (⟨68786, 0⟩, ⟨68509, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7181⟩⟩, ⟨.program ⟨257⟩, ⟨24089⟩⟩]⟩, (1)⟩)

def event68793 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨24091⟩⟩, .operator (⟨68786, 1⟩, ⟨68509, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨21864⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨24089⟩⟩]⟩, (-1)⟩)

def event68794 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨24091⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨21864⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨24089⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨24089⟩⟩) ⟨23144⟩ 68506)

def event68795 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨24091⟩⟩, .relation 68794 0, ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨21864⟩⟩], [⟨.program ⟨257⟩, ⟨23144⟩⟩]⟩, (-1)⟩)

def exact68796RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7181⟩⟩, ⟨.program ⟨257⟩, ⟨24089⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨21864⟩⟩], [⟨.program ⟨257⟩, ⟨23144⟩⟩]⟩, (-1)⟩]

theorem exact68796RawTermsValid :
    exact68796RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68796 : Event := .resultExact (⟨.program ⟨257⟩, ⟨24091⟩⟩) exact68796RawTerms .large 68789 (.finite 32189003662929192193909661368320) (some (68791))

def event68797 : Event := .predecessor (⟨.program ⟨257⟩, ⟨22816⟩⟩) 0 ⟨21865⟩ 2700

def event68798 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨22816⟩⟩) (.authority (.relationPreimageSource ⟨61⟩))

def exact68799RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨22816⟩⟩]⟩, (1)⟩]

theorem exact68799RawTermsValid :
    exact68799RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68799 : Event := .resultExact (⟨.program ⟨257⟩, ⟨22816⟩⟩) exact68799RawTerms (.finite 5647228698) 68798 .exactZero (none)

def event68800 : Event := .predecessor (⟨.program ⟨257⟩, ⟨22818⟩⟩) 0 ⟨22816⟩ 68799

def event68801 : Event := .predecessor (⟨.program ⟨257⟩, ⟨22818⟩⟩) 1 ⟨2370⟩ 4

def event68802 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨22818⟩⟩) (.scale (.predecessor 0 68800 .coefficient) (.value (.predecessor 1 68801 .coefficient)))

def exact68803RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨22816⟩⟩]⟩, (1)⟩]

theorem exact68803RawTermsValid :
    exact68803RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68803 : Event := .resultExact (⟨.program ⟨257⟩, ⟨22818⟩⟩) exact68803RawTerms (.finite 5647228698) 68802 .exactZero (none)

def event68804 : Event := .predecessor (⟨.program ⟨257⟩, ⟨22819⟩⟩) 0 ⟨10792⟩ 61370

def event68805 : Event := .predecessor (⟨.program ⟨257⟩, ⟨22819⟩⟩) 1 ⟨22818⟩ 68803

def event68806 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨22819⟩⟩) (.product (.predecessor 0 68804 .coefficient) (.predecessor 1 68805 .coefficient) (⟨false, false, none, none, none⟩))

def event68807 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨22819⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨22816⟩⟩]⟩) [⟨.result 68799 .coefficient, false, none⟩])

def event68808 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨22819⟩⟩) (.product (.result 61370 .summary) (.transfer 68807) (⟨false, false, none, none, none⟩))

def event68809 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨22819⟩⟩, .operator (⟨61370, 0⟩, ⟨68803, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22816⟩⟩]⟩, (1)⟩)

def event68810 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨22817⟩⟩)

def event68811 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event68812 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event68813 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10691⟩⟩) (.authority (.operator))

def event68814 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10691⟩⟩) (.finite 16)

def event68815 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event68816 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event68817 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event68818 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event68819 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 68818

def event68820 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 68816

def event68821 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 68819 .coefficient) (.value (.predecessor 1 68820 .coefficient)))

def event68822 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event68823 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10693⟩⟩) 0 ⟨392⟩ 68822

def event68824 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10693⟩⟩) 1 ⟨10691⟩ 68814

def event68825 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10693⟩⟩) (.sum [.predecessor 0 68823 .coefficient, .predecessor 1 68824 .coefficient])

def event68826 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10693⟩⟩) (.finite 655356)

def event68827 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10749⟩⟩) 0 ⟨10693⟩ 68826

def event68828 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10749⟩⟩) 1 ⟨5426⟩ 68812

def event68829 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10749⟩⟩) (.identity (.predecessor 1 68828 .coefficient))

def event68830 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10749⟩⟩) (.finite 655360)

def event68831 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21662⟩⟩) 0 ⟨10749⟩ 68830

def event68832 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨21662⟩⟩) (.authority (.programFamilyFact))

def exact68833RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨21662⟩⟩], []⟩, (1)⟩]

theorem exact68833RawTermsValid :
    exact68833RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68833 : Event := .resultExact (⟨.program ⟨257⟩, ⟨21662⟩⟩) exact68833RawTerms (.finite 4) 68832 .exactZero (none)

def event68834 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21206⟩⟩) 0 ⟨10749⟩ 68830

def event68835 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨21206⟩⟩) (.authority (.programFamilyFact))

def exact68836RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨21206⟩⟩], []⟩, (1)⟩]

theorem exact68836RawTermsValid :
    exact68836RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68836 : Event := .resultExact (⟨.program ⟨257⟩, ⟨21206⟩⟩) exact68836RawTerms (.finite 4) 68835 .exactZero (none)

def event68837 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21663⟩⟩) 0 ⟨21206⟩ 68836

def event68838 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21663⟩⟩) 1 ⟨21662⟩ 68833

def event68839 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨21663⟩⟩) (.product (.predecessor 0 68837 .coefficient) (.predecessor 1 68838 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event68840 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨21663⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨21206⟩⟩, ⟨.program ⟨257⟩, ⟨21662⟩⟩], []⟩) [⟨.result 68836 .coefficient, true, some 1⟩, ⟨.result 68833 .coefficient, true, some 1⟩])

def event68841 : Event := .survivorFold (1) 68840

def exact68842RawTerms : List Term := []

theorem exact68842RawTermsValid :
    exact68842RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68842 : Event := .resultExact (⟨.program ⟨257⟩, ⟨21663⟩⟩) exact68842RawTerms (.finite 16) 68839 (.finite 16) (some (68840))

def event68843 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21664⟩⟩) 0 ⟨21663⟩ 68842

def event68844 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨21664⟩⟩) (.identity (.predecessor 0 68843 .coefficient))

def event68845 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨21664⟩⟩) (.finite 16)

def event68846 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21864⟩⟩) 0 ⟨21664⟩ 68845

def event68847 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨21864⟩⟩) (.authority (.programFamilyFact))

def exact68848RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨21864⟩⟩], []⟩, (1)⟩]

theorem exact68848RawTermsValid :
    exact68848RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68848 : Event := .resultExact (⟨.program ⟨257⟩, ⟨21864⟩⟩) exact68848RawTerms (.finite 4) 68847 .exactZero (none)

def event68849 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21865⟩⟩) 0 ⟨21864⟩ 68848

def event68850 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨21865⟩⟩) (.identity (.predecessor 0 68849 .coefficient))

def event68851 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨21865⟩⟩) (.finite 4)

def event68852 : Event := .predecessor (⟨.program ⟨257⟩, ⟨22816⟩⟩) 0 ⟨21865⟩ 68851

def event68853 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨22816⟩⟩) (.authority (.relationPreimageSource ⟨61⟩))

def exact68854RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨22816⟩⟩]⟩, (1)⟩]

theorem exact68854RawTermsValid :
    exact68854RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68854 : Event := .resultExact (⟨.program ⟨257⟩, ⟨22816⟩⟩) exact68854RawTerms (.finite 5647228698) 68853 .exactZero (none)

def event68855 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35⟩⟩) (.authority (.operator))

def exact68856RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact68856RawTermsValid :
    exact68856RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68856 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35⟩⟩) exact68856RawTerms .large 68855 .exactZero (none)

def event68857 : Event := .predecessor (⟨.program ⟨257⟩, ⟨22817⟩⟩) 0 ⟨35⟩ 68856

def event68858 : Event := .predecessor (⟨.program ⟨257⟩, ⟨22817⟩⟩) 1 ⟨22816⟩ 68854

def event68859 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨22817⟩⟩) (.product (.predecessor 0 68857 .coefficient) (.predecessor 1 68858 .coefficient) (⟨false, false, none, none, none⟩))

def event68860 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨22817⟩⟩, .operator (⟨68856, 0⟩, ⟨68854, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22816⟩⟩]⟩, (1)⟩)

def exact68861RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22816⟩⟩]⟩, (1)⟩]

theorem exact68861RawTermsValid :
    exact68861RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68861 : Event := .resultExact (⟨.program ⟨257⟩, ⟨22817⟩⟩) exact68861RawTerms .large 68859 .exactZero (none)

def event68862 : Event := .preFoldPolynomial 68861 [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22816⟩⟩]⟩, (1)⟩] .exactZero none

def exact68863RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22816⟩⟩]⟩, (1)⟩]

def event68863 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨22817⟩⟩) 68862 exact68863RawTerms .large 68859 .exactZero (none)

def eventLeaf4288 : Array AnnotatedEvent := #[
  { event := event68608
    frameStart := 68607 },
  { event := event68609
    frameStart := 68607 },
  { event := event68610
    frameStart := 68607 },
  { event := event68611
    frameStart := 68607 },
  { event := event68612
    frameStart := 68607 },
  { event := event68613
    frameStart := 68607 },
  { event := event68614
    frameStart := 68607 },
  { event := event68615
    frameStart := 68607 },
  { event := event68616
    frameStart := 68607 },
  { event := event68617
    frameStart := 68607 },
  { event := event68618
    frameStart := 68607 },
  { event := event68619
    frameStart := 68607 },
  { event := event68620
    frameStart := 68607 },
  { event := event68621
    frameStart := 68607 },
  { event := event68622
    frameStart := 68607 },
  { event := event68623
    frameStart := 68607 }
]

def eventLeaf4289 : Array AnnotatedEvent := #[
  { event := event68624
    frameStart := 68607 },
  { event := event68625
    frameStart := 68607 },
  { event := event68626
    frameStart := 68607 },
  { event := event68627
    frameStart := 68607 },
  { event := event68628
    frameStart := 68607 },
  { event := event68629
    frameStart := 68607 },
  { event := event68630
    frameStart := 68607 },
  { event := event68631
    frameStart := 68607 },
  { event := event68632
    frameStart := 68607 },
  { event := event68633
    frameStart := 68607 },
  { event := event68634
    frameStart := 68607 },
  { event := event68635
    frameStart := 68607 },
  { event := event68636
    frameStart := 68607 },
  { event := event68637
    frameStart := 68607 },
  { event := event68638
    frameStart := 68607 },
  { event := event68639
    frameStart := 68607 }
]

def eventLeaf4290 : Array AnnotatedEvent := #[
  { event := event68640
    frameStart := 68607 },
  { event := event68641
    frameStart := 68607 },
  { event := event68642
    frameStart := 68607 },
  { event := event68643
    frameStart := 68607 },
  { event := event68644
    frameStart := 68607 },
  { event := event68645
    frameStart := 68607 },
  { event := event68646
    frameStart := 68607 },
  { event := event68647
    frameStart := 68607 },
  { event := event68648
    frameStart := 68607 },
  { event := event68649
    frameStart := 68607 },
  { event := event68650
    frameStart := 68607 },
  { event := event68651
    frameStart := 68607 },
  { event := event68652
    frameStart := 68607 },
  { event := event68653
    frameStart := 68607 },
  { event := event68654
    frameStart := 68607 },
  { event := event68655
    frameStart := 68655 }
]

def eventLeaf4291 : Array AnnotatedEvent := #[
  { event := event68656
    frameStart := 68655 },
  { event := event68657
    frameStart := 68655 },
  { event := event68658
    frameStart := 68655 },
  { event := event68659
    frameStart := 68655 },
  { event := event68660
    frameStart := 68655 },
  { event := event68661
    frameStart := 68655 },
  { event := event68662
    frameStart := 68655 },
  { event := event68663
    frameStart := 68655 },
  { event := event68664
    frameStart := 68655 },
  { event := event68665
    frameStart := 68655 },
  { event := event68666
    frameStart := 68655 },
  { event := event68667
    frameStart := 68655 },
  { event := event68668
    frameStart := 68655 },
  { event := event68669
    frameStart := 68655 },
  { event := event68670
    frameStart := 68655 },
  { event := event68671
    frameStart := 68655 }
]

def eventLeaf4292 : Array AnnotatedEvent := #[
  { event := event68672
    frameStart := 68655 },
  { event := event68673
    frameStart := 68655 },
  { event := event68674
    frameStart := 68655 },
  { event := event68675
    frameStart := 68655 },
  { event := event68676
    frameStart := 68655 },
  { event := event68677
    frameStart := 68655 },
  { event := event68678
    frameStart := 68655 },
  { event := event68679
    frameStart := 68655 },
  { event := event68680
    frameStart := 68655 },
  { event := event68681
    frameStart := 68655 },
  { event := event68682
    frameStart := 68655 },
  { event := event68683
    frameStart := 68655 },
  { event := event68684
    frameStart := 68655 },
  { event := event68685
    frameStart := 68655 },
  { event := event68686
    frameStart := 68655 },
  { event := event68687
    frameStart := 68655 }
]

def eventLeaf4293 : Array AnnotatedEvent := #[
  { event := event68688
    frameStart := 68655 },
  { event := event68689
    frameStart := 68655 },
  { event := event68690
    frameStart := 68655 },
  { event := event68691
    frameStart := 68655 },
  { event := event68692
    frameStart := 68655 },
  { event := event68693
    frameStart := 68655 },
  { event := event68694
    frameStart := 68655 },
  { event := event68695
    frameStart := 68655 },
  { event := event68696
    frameStart := 68655 },
  { event := event68697
    frameStart := 68655 },
  { event := event68698
    frameStart := 68655 },
  { event := event68699
    frameStart := 68655 },
  { event := event68700
    frameStart := 68655 },
  { event := event68701
    frameStart := 68655 },
  { event := event68702
    frameStart := 68655 },
  { event := event68703
    frameStart := 68655 }
]

def eventLeaf4294 : Array AnnotatedEvent := #[
  { event := event68704
    frameStart := 68655 },
  { event := event68705
    frameStart := 68655 },
  { event := event68706
    frameStart := 68655 },
  { event := event68707
    frameStart := 68655 },
  { event := event68708
    frameStart := 68655 },
  { event := event68709
    frameStart := 68655 },
  { event := event68710
    frameStart := 68655 },
  { event := event68711
    frameStart := 68655 },
  { event := event68712
    frameStart := 68655 },
  { event := event68713
    frameStart := 68655 },
  { event := event68714
    frameStart := 68655 },
  { event := event68715
    frameStart := 68655 },
  { event := event68716
    frameStart := 68655 },
  { event := event68717
    frameStart := 68655 },
  { event := event68718
    frameStart := 68655 },
  { event := event68719
    frameStart := 68655 }
]

def eventLeaf4295 : Array AnnotatedEvent := #[
  { event := event68720
    frameStart := 68655 },
  { event := event68721
    frameStart := 68655 },
  { event := event68722
    frameStart := 68655 },
  { event := event68723
    frameStart := 68655 },
  { event := event68724
    frameStart := 68655 },
  { event := event68725
    frameStart := 68655 },
  { event := event68726
    frameStart := 68655 },
  { event := event68727
    frameStart := 68655 },
  { event := event68728
    frameStart := 68655 },
  { event := event68729
    frameStart := 68655 },
  { event := event68730
    frameStart := 68655 },
  { event := event68731
    frameStart := 68655 },
  { event := event68732
    frameStart := 68655 },
  { event := event68733
    frameStart := 68655 },
  { event := event68734
    frameStart := 68655 },
  { event := event68735
    frameStart := 68655 }
]

def eventLeaf4296 : Array AnnotatedEvent := #[
  { event := event68736
    frameStart := 68655 },
  { event := event68737
    frameStart := 68655 },
  { event := event68738
    frameStart := 68655 },
  { event := event68739
    frameStart := 68655 },
  { event := event68740
    frameStart := 68655 },
  { event := event68741
    frameStart := 68655 },
  { event := event68742
    frameStart := 68655 },
  { event := event68743
    frameStart := 68655 },
  { event := event68744
    frameStart := 68655 },
  { event := event68745
    frameStart := 68655 },
  { event := event68746
    frameStart := 68655 },
  { event := event68747
    frameStart := 68655 },
  { event := event68748
    frameStart := 68655 },
  { event := event68749
    frameStart := 68655 },
  { event := event68750
    frameStart := 68655 },
  { event := event68751
    frameStart := 68655 }
]

def eventLeaf4297 : Array AnnotatedEvent := #[
  { event := event68752
    frameStart := 68655 },
  { event := event68753
    frameStart := 68655 },
  { event := event68754
    frameStart := 68655 },
  { event := event68755
    frameStart := 68655 },
  { event := event68756
    frameStart := 68655 },
  { event := event68757
    frameStart := 68655 },
  { event := event68758
    frameStart := 68655 },
  { event := event68759
    frameStart := 68655 },
  { event := event68760
    frameStart := 68655 },
  { event := event68761
    frameStart := 68655 },
  { event := event68762
    frameStart := 68655 },
  { event := event68763
    frameStart := 68655 },
  { event := event68764
    frameStart := 68655 },
  { event := event68765
    frameStart := 68655 },
  { event := event68766
    frameStart := 68655 },
  { event := event68767
    frameStart := 68655 }
]

def eventLeaf4298 : Array AnnotatedEvent := #[
  { event := event68768
    frameStart := 68655 },
  { event := event68769
    frameStart := 68655 },
  { event := event68770
    frameStart := 68655 },
  { event := event68771
    frameStart := 68655 },
  { event := event68772
    frameStart := 68655 },
  { event := event68773
    frameStart := 0 },
  { event := event68774
    frameStart := 0 },
  { event := event68775
    frameStart := 0 },
  { event := event68776
    frameStart := 0 },
  { event := event68777
    frameStart := 0 },
  { event := event68778
    frameStart := 0 },
  { event := event68779
    frameStart := 0 },
  { event := event68780
    frameStart := 0 },
  { event := event68781
    frameStart := 0 },
  { event := event68782
    frameStart := 0 },
  { event := event68783
    frameStart := 0 }
]

def eventLeaf4299 : Array AnnotatedEvent := #[
  { event := event68784
    frameStart := 0 },
  { event := event68785
    frameStart := 0 },
  { event := event68786
    frameStart := 0 },
  { event := event68787
    frameStart := 0 },
  { event := event68788
    frameStart := 0 },
  { event := event68789
    frameStart := 0 },
  { event := event68790
    frameStart := 0 },
  { event := event68791
    frameStart := 0 },
  { event := event68792
    frameStart := 0 },
  { event := event68793
    frameStart := 0 },
  { event := event68794
    frameStart := 0 },
  { event := event68795
    frameStart := 0 },
  { event := event68796
    frameStart := 0 },
  { event := event68797
    frameStart := 0 },
  { event := event68798
    frameStart := 0 },
  { event := event68799
    frameStart := 0 }
]

def eventLeaf4300 : Array AnnotatedEvent := #[
  { event := event68800
    frameStart := 0 },
  { event := event68801
    frameStart := 0 },
  { event := event68802
    frameStart := 0 },
  { event := event68803
    frameStart := 0 },
  { event := event68804
    frameStart := 0 },
  { event := event68805
    frameStart := 0 },
  { event := event68806
    frameStart := 0 },
  { event := event68807
    frameStart := 0 },
  { event := event68808
    frameStart := 0 },
  { event := event68809
    frameStart := 0 },
  { event := event68810
    frameStart := 68810 },
  { event := event68811
    frameStart := 68810 },
  { event := event68812
    frameStart := 68810 },
  { event := event68813
    frameStart := 68810 },
  { event := event68814
    frameStart := 68810 },
  { event := event68815
    frameStart := 68810 }
]

def eventLeaf4301 : Array AnnotatedEvent := #[
  { event := event68816
    frameStart := 68810 },
  { event := event68817
    frameStart := 68810 },
  { event := event68818
    frameStart := 68810 },
  { event := event68819
    frameStart := 68810 },
  { event := event68820
    frameStart := 68810 },
  { event := event68821
    frameStart := 68810 },
  { event := event68822
    frameStart := 68810 },
  { event := event68823
    frameStart := 68810 },
  { event := event68824
    frameStart := 68810 },
  { event := event68825
    frameStart := 68810 },
  { event := event68826
    frameStart := 68810 },
  { event := event68827
    frameStart := 68810 },
  { event := event68828
    frameStart := 68810 },
  { event := event68829
    frameStart := 68810 },
  { event := event68830
    frameStart := 68810 },
  { event := event68831
    frameStart := 68810 }
]

def eventLeaf4302 : Array AnnotatedEvent := #[
  { event := event68832
    frameStart := 68810 },
  { event := event68833
    frameStart := 68810 },
  { event := event68834
    frameStart := 68810 },
  { event := event68835
    frameStart := 68810 },
  { event := event68836
    frameStart := 68810 },
  { event := event68837
    frameStart := 68810 },
  { event := event68838
    frameStart := 68810 },
  { event := event68839
    frameStart := 68810 },
  { event := event68840
    frameStart := 68810 },
  { event := event68841
    frameStart := 68810 },
  { event := event68842
    frameStart := 68810 },
  { event := event68843
    frameStart := 68810 },
  { event := event68844
    frameStart := 68810 },
  { event := event68845
    frameStart := 68810 },
  { event := event68846
    frameStart := 68810 },
  { event := event68847
    frameStart := 68810 }
]

def eventLeaf4303 : Array AnnotatedEvent := #[
  { event := event68848
    frameStart := 68810 },
  { event := event68849
    frameStart := 68810 },
  { event := event68850
    frameStart := 68810 },
  { event := event68851
    frameStart := 68810 },
  { event := event68852
    frameStart := 68810 },
  { event := event68853
    frameStart := 68810 },
  { event := event68854
    frameStart := 68810 },
  { event := event68855
    frameStart := 68810 },
  { event := event68856
    frameStart := 68810 },
  { event := event68857
    frameStart := 68810 },
  { event := event68858
    frameStart := 68810 },
  { event := event68859
    frameStart := 68810 },
  { event := event68860
    frameStart := 68810 },
  { event := event68861
    frameStart := 68810 },
  { event := event68862
    frameStart := 68810 },
  { event := event68863
    frameStart := 68810 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events268
