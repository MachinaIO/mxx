import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events346

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event88576 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10269⟩⟩) 0 ⟨392⟩ 88575

def event88577 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10269⟩⟩) 1 ⟨10267⟩ 88567

def event88578 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10269⟩⟩) (.sum [.predecessor 0 88576 .coefficient, .predecessor 1 88577 .coefficient])

def event88579 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10269⟩⟩) (.finite 655355)

def event88580 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10325⟩⟩) 0 ⟨10269⟩ 88579

def event88581 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10325⟩⟩) 1 ⟨5426⟩ 88565

def event88582 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10325⟩⟩) (.identity (.predecessor 1 88581 .coefficient))

def event88583 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10325⟩⟩) (.finite 655360)

def event88584 : Event := .predecessor (⟨.program ⟨257⟩, ⟨25082⟩⟩) 0 ⟨10325⟩ 88583

def event88585 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨25082⟩⟩) (.authority (.programFamilyFact))

def exact88586RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨25082⟩⟩], []⟩, (1)⟩]

theorem exact88586RawTermsValid :
    exact88586RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88586 : Event := .resultExact (⟨.program ⟨257⟩, ⟨25082⟩⟩) exact88586RawTerms (.finite 16) 88585 .exactZero (none)

def event88587 : Event := .predecessor (⟨.program ⟨257⟩, ⟨56667⟩⟩) 0 ⟨10325⟩ 88583

def event88588 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨56667⟩⟩) (.authority (.programFamilyFact))

def exact88589RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨56667⟩⟩], []⟩, (1)⟩]

theorem exact88589RawTermsValid :
    exact88589RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88589 : Event := .resultExact (⟨.program ⟨257⟩, ⟨56667⟩⟩) exact88589RawTerms (.finite 16) 88588 .exactZero (none)

def event88590 : Event := .predecessor (⟨.program ⟨257⟩, ⟨56668⟩⟩) 0 ⟨56667⟩ 88589

def event88591 : Event := .predecessor (⟨.program ⟨257⟩, ⟨56668⟩⟩) 1 ⟨25082⟩ 88586

def event88592 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨56668⟩⟩) (.product (.predecessor 0 88590 .coefficient) (.predecessor 1 88591 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event88593 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨56668⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨25082⟩⟩, ⟨.program ⟨257⟩, ⟨56667⟩⟩], []⟩) [⟨.result 88589 .coefficient, true, some 1⟩, ⟨.result 88586 .coefficient, true, some 1⟩])

def event88594 : Event := .survivorFold (1) 88593

def exact88595RawTerms : List Term := []

theorem exact88595RawTermsValid :
    exact88595RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88595 : Event := .resultExact (⟨.program ⟨257⟩, ⟨56668⟩⟩) exact88595RawTerms (.finite 256) 88592 (.finite 256) (some (88593))

def event88596 : Event := .predecessor (⟨.program ⟨257⟩, ⟨56669⟩⟩) 0 ⟨56668⟩ 88595

def event88597 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨56669⟩⟩) (.identity (.predecessor 0 88596 .coefficient))

def event88598 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨56669⟩⟩) (.finite 256)

def event88599 : Event := .predecessor (⟨.program ⟨257⟩, ⟨56896⟩⟩) 0 ⟨56669⟩ 88598

def event88600 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨56896⟩⟩) (.authority (.programFamilyFact))

def exact88601RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨56896⟩⟩], []⟩, (1)⟩]

theorem exact88601RawTermsValid :
    exact88601RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88601 : Event := .resultExact (⟨.program ⟨257⟩, ⟨56896⟩⟩) exact88601RawTerms (.finite 16) 88600 .exactZero (none)

def event88602 : Event := .predecessor (⟨.program ⟨257⟩, ⟨56897⟩⟩) 0 ⟨56896⟩ 88601

def event88603 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨56897⟩⟩) (.identity (.predecessor 0 88602 .coefficient))

def event88604 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨56897⟩⟩) (.finite 16)

def event88605 : Event := .predecessor (⟨.program ⟨257⟩, ⟨57832⟩⟩) 0 ⟨56897⟩ 88604

def event88606 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨57832⟩⟩) (.authority (.relationPreimageSource ⟨69⟩))

def exact88607RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨57832⟩⟩]⟩, (1)⟩]

theorem exact88607RawTermsValid :
    exact88607RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88607 : Event := .resultExact (⟨.program ⟨257⟩, ⟨57832⟩⟩) exact88607RawTerms (.finite 5647228698) 88606 .exactZero (none)

def event88608 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35⟩⟩) (.authority (.operator))

def exact88609RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact88609RawTermsValid :
    exact88609RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88609 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35⟩⟩) exact88609RawTerms .large 88608 .exactZero (none)

def event88610 : Event := .predecessor (⟨.program ⟨257⟩, ⟨57833⟩⟩) 0 ⟨35⟩ 88609

def event88611 : Event := .predecessor (⟨.program ⟨257⟩, ⟨57833⟩⟩) 1 ⟨57832⟩ 88607

def event88612 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨57833⟩⟩) (.product (.predecessor 0 88610 .coefficient) (.predecessor 1 88611 .coefficient) (⟨false, false, none, none, none⟩))

def event88613 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨57833⟩⟩, .operator (⟨88609, 0⟩, ⟨88607, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57832⟩⟩]⟩, (1)⟩)

def exact88614RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57832⟩⟩]⟩, (1)⟩]

theorem exact88614RawTermsValid :
    exact88614RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88614 : Event := .resultExact (⟨.program ⟨257⟩, ⟨57833⟩⟩) exact88614RawTerms .large 88612 .exactZero (none)

def event88615 : Event := .preFoldPolynomial 88614 [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57832⟩⟩]⟩, (1)⟩] .exactZero none

def exact88616RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57832⟩⟩]⟩, (1)⟩]

def event88616 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨57833⟩⟩) 88615 exact88616RawTerms .large 88612 .exactZero (none)

def event88617 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨59097⟩⟩)

def event88618 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event88619 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event88620 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10267⟩⟩) (.authority (.operator))

def event88621 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10267⟩⟩) (.finite 15)

def event88622 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event88623 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event88624 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event88625 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event88626 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 88625

def event88627 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 88623

def event88628 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 88626 .coefficient) (.value (.predecessor 1 88627 .coefficient)))

def event88629 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event88630 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10269⟩⟩) 0 ⟨392⟩ 88629

def event88631 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10269⟩⟩) 1 ⟨10267⟩ 88621

def event88632 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10269⟩⟩) (.sum [.predecessor 0 88630 .coefficient, .predecessor 1 88631 .coefficient])

def event88633 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10269⟩⟩) (.finite 655355)

def event88634 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10325⟩⟩) 0 ⟨10269⟩ 88633

def event88635 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10325⟩⟩) 1 ⟨5426⟩ 88619

def event88636 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10325⟩⟩) (.identity (.predecessor 1 88635 .coefficient))

def event88637 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10325⟩⟩) (.finite 655360)

def event88638 : Event := .predecessor (⟨.program ⟨257⟩, ⟨25082⟩⟩) 0 ⟨10325⟩ 88637

def event88639 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨25082⟩⟩) (.authority (.programFamilyFact))

def exact88640RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨25082⟩⟩], []⟩, (1)⟩]

theorem exact88640RawTermsValid :
    exact88640RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88640 : Event := .resultExact (⟨.program ⟨257⟩, ⟨25082⟩⟩) exact88640RawTerms (.finite 16) 88639 .exactZero (none)

def event88641 : Event := .predecessor (⟨.program ⟨257⟩, ⟨56667⟩⟩) 0 ⟨10325⟩ 88637

def event88642 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨56667⟩⟩) (.authority (.programFamilyFact))

def exact88643RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨56667⟩⟩], []⟩, (1)⟩]

theorem exact88643RawTermsValid :
    exact88643RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88643 : Event := .resultExact (⟨.program ⟨257⟩, ⟨56667⟩⟩) exact88643RawTerms (.finite 16) 88642 .exactZero (none)

def event88644 : Event := .predecessor (⟨.program ⟨257⟩, ⟨56668⟩⟩) 0 ⟨56667⟩ 88643

def event88645 : Event := .predecessor (⟨.program ⟨257⟩, ⟨56668⟩⟩) 1 ⟨25082⟩ 88640

def event88646 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨56668⟩⟩) (.product (.predecessor 0 88644 .coefficient) (.predecessor 1 88645 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event88647 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨56668⟩⟩, .operator (⟨88643, 0⟩, ⟨88640, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨25082⟩⟩, ⟨.program ⟨257⟩, ⟨56667⟩⟩], []⟩, (1)⟩)

def exact88648RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨25082⟩⟩, ⟨.program ⟨257⟩, ⟨56667⟩⟩], []⟩, (1)⟩]

theorem exact88648RawTermsValid :
    exact88648RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88648 : Event := .resultExact (⟨.program ⟨257⟩, ⟨56668⟩⟩) exact88648RawTerms (.finite 256) 88646 .exactZero (none)

def event88649 : Event := .predecessor (⟨.program ⟨257⟩, ⟨56669⟩⟩) 0 ⟨56668⟩ 88648

def event88650 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨56669⟩⟩) (.identity (.predecessor 0 88649 .coefficient))

def event88651 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨56669⟩⟩) (.finite 256)

def event88652 : Event := .predecessor (⟨.program ⟨257⟩, ⟨56896⟩⟩) 0 ⟨56669⟩ 88651

def event88653 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨56896⟩⟩) (.authority (.programFamilyFact))

def exact88654RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨56896⟩⟩], []⟩, (1)⟩]

theorem exact88654RawTermsValid :
    exact88654RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88654 : Event := .resultExact (⟨.program ⟨257⟩, ⟨56896⟩⟩) exact88654RawTerms (.finite 16) 88653 .exactZero (none)

def event88655 : Event := .predecessor (⟨.program ⟨257⟩, ⟨56897⟩⟩) 0 ⟨56896⟩ 88654

def event88656 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨56897⟩⟩) (.identity (.predecessor 0 88655 .coefficient))

def event88657 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨56897⟩⟩) (.finite 16)

def event88658 : Event := .predecessor (⟨.program ⟨257⟩, ⟨58173⟩⟩) 0 ⟨56897⟩ 88657

def event88659 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨58173⟩⟩) (.authority (.programFamilyFact))

def event88660 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨58173⟩⟩) (.finite 3720)

def event88661 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨7177⟩⟩) .missing

def event88662 : Event := .predecessor (⟨.program ⟨257⟩, ⟨58174⟩⟩) 0 ⟨7177⟩ 88661

def event88663 : Event := .predecessor (⟨.program ⟨257⟩, ⟨58174⟩⟩) 1 ⟨58173⟩ 88660

def event88664 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨58174⟩⟩) (.authority (.operator))

def exact88665RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨58174⟩⟩]⟩, (1)⟩]

theorem exact88665RawTermsValid :
    exact88665RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88665 : Event := .resultExact (⟨.program ⟨257⟩, ⟨58174⟩⟩) exact88665RawTerms .large 88664 .exactZero (none)

def event88666 : Event := .predecessor (⟨.program ⟨257⟩, ⟨59091⟩⟩) 0 ⟨58174⟩ 88665

def event88667 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨59091⟩⟩) (.authority (.operator))

def exact88668RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨59091⟩⟩]⟩, (1)⟩]

theorem exact88668RawTermsValid :
    exact88668RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88668 : Event := .resultExact (⟨.program ⟨257⟩, ⟨59091⟩⟩) exact88668RawTerms (.finite 8192) 88667 .exactZero (none)

def event88669 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event88670 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event88671 : Event := .predecessor (⟨.program ⟨257⟩, ⟨58350⟩⟩) 0 ⟨56897⟩ 88657

def event88672 : Event := .predecessor (⟨.program ⟨257⟩, ⟨58350⟩⟩) 1 ⟨136⟩ 88670

def event88673 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨58350⟩⟩) (.sum [.predecessor 0 88671 .coefficient, .predecessor 1 88672 .coefficient])

def event88674 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨58350⟩⟩) (.finite 16)

def event88675 : Event := .predecessor (⟨.program ⟨257⟩, ⟨58351⟩⟩) 0 ⟨58350⟩ 88674

def event88676 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨58351⟩⟩) (.identity (.predecessor 0 88675 .coefficient))

def exact88677RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨56896⟩⟩], []⟩, (1)⟩]

theorem exact88677RawTermsValid :
    exact88677RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88677 : Event := .resultExact (⟨.program ⟨257⟩, ⟨58351⟩⟩) exact88677RawTerms (.finite 16) 88676 .exactZero (none)

def event88678 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact88679RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact88679RawTermsValid :
    exact88679RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88679 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact88679RawTerms .large 88678 .exactZero (none)

def event88680 : Event := .predecessor (⟨.program ⟨257⟩, ⟨58352⟩⟩) 0 ⟨6908⟩ 88679

def event88681 : Event := .predecessor (⟨.program ⟨257⟩, ⟨58352⟩⟩) 1 ⟨58351⟩ 88677

def event88682 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨58352⟩⟩) (.product (.predecessor 0 88680 .coefficient) (.predecessor 1 88681 .coefficient) (⟨false, false, none, none, none⟩))

def event88683 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨58352⟩⟩, .operator (⟨88679, 0⟩, ⟨88677, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨56896⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact88684RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨56896⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact88684RawTermsValid :
    exact88684RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88684 : Event := .resultExact (⟨.program ⟨257⟩, ⟨58352⟩⟩) exact88684RawTerms .large 88682 .exactZero (none)

def event88685 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7185⟩⟩) 0 ⟨7177⟩ 88661

def event88686 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7185⟩⟩) (.authority (.operator))

def exact88687RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7185⟩⟩]⟩, (1)⟩]

theorem exact88687RawTermsValid :
    exact88687RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88687 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7185⟩⟩) exact88687RawTerms .large 88686 .exactZero (none)

def event88688 : Event := .predecessor (⟨.program ⟨257⟩, ⟨58353⟩⟩) 0 ⟨7185⟩ 88687

def event88689 : Event := .predecessor (⟨.program ⟨257⟩, ⟨58353⟩⟩) 1 ⟨58352⟩ 88684

def event88690 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨58353⟩⟩) (.sum [.predecessor 0 88688 .coefficient, .predecessor 1 88689 .coefficient])

def exact88691RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7185⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨56896⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact88691RawTermsValid :
    exact88691RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88691 : Event := .resultExact (⟨.program ⟨257⟩, ⟨58353⟩⟩) exact88691RawTerms .large 88690 .exactZero (none)

def event88692 : Event := .predecessor (⟨.program ⟨257⟩, ⟨59092⟩⟩) 0 ⟨58353⟩ 88691

def event88693 : Event := .predecessor (⟨.program ⟨257⟩, ⟨59092⟩⟩) 1 ⟨59091⟩ 88668

def event88694 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨59092⟩⟩) (.product (.predecessor 0 88692 .coefficient) (.predecessor 1 88693 .coefficient) (⟨false, false, none, none, none⟩))

def event88695 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨59092⟩⟩, .operator (⟨88691, 0⟩, ⟨88668, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7185⟩⟩, ⟨.program ⟨257⟩, ⟨59091⟩⟩]⟩, (1)⟩)

def event88696 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨59092⟩⟩, .operator (⟨88691, 1⟩, ⟨88668, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨56896⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨59091⟩⟩]⟩, (-1)⟩)

def event88697 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨59092⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨56896⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨59091⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨59091⟩⟩) ⟨58174⟩ 88665)

def event88698 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨59092⟩⟩, .relation 88697 0, ⟨[⟨.program ⟨257⟩, ⟨56896⟩⟩], [⟨.program ⟨257⟩, ⟨58174⟩⟩]⟩, (-1)⟩)

def exact88699RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7185⟩⟩, ⟨.program ⟨257⟩, ⟨59091⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨56896⟩⟩], [⟨.program ⟨257⟩, ⟨58174⟩⟩]⟩, (-1)⟩]

theorem exact88699RawTermsValid :
    exact88699RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88699 : Event := .resultExact (⟨.program ⟨257⟩, ⟨59092⟩⟩) exact88699RawTerms .large 88694 .exactZero (none)

def event88700 : Event := .predecessor (⟨.program ⟨257⟩, ⟨57239⟩⟩) 0 ⟨56897⟩ 88657

def event88701 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨57239⟩⟩) (.authority (.programFamilyFact))

def exact88702RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨57239⟩⟩], []⟩, (1)⟩]

theorem exact88702RawTermsValid :
    exact88702RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88702 : Event := .resultExact (⟨.program ⟨257⟩, ⟨57239⟩⟩) exact88702RawTerms (.finite 16) 88701 .exactZero (none)

def event88703 : Event := .predecessor (⟨.program ⟨257⟩, ⟨57242⟩⟩) 0 ⟨6908⟩ 88679

def event88704 : Event := .predecessor (⟨.program ⟨257⟩, ⟨57242⟩⟩) 1 ⟨57239⟩ 88702

def event88705 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨57242⟩⟩) (.product (.predecessor 0 88703 .coefficient) (.predecessor 1 88704 .coefficient) (⟨false, true, none, none, some 1⟩))

def event88706 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨57242⟩⟩, .operator (⟨88679, 0⟩, ⟨88702, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨57239⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact88707RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨57239⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact88707RawTermsValid :
    exact88707RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88707 : Event := .resultExact (⟨.program ⟨257⟩, ⟨57242⟩⟩) exact88707RawTerms .large 88705 .exactZero (none)

def event88708 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7209⟩⟩) 0 ⟨7177⟩ 88661

def event88709 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7209⟩⟩) (.authority (.operator))

def exact88710RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7209⟩⟩]⟩, (1)⟩]

theorem exact88710RawTermsValid :
    exact88710RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88710 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7209⟩⟩) exact88710RawTerms .large 88709 .exactZero (none)

def event88711 : Event := .predecessor (⟨.program ⟨257⟩, ⟨57243⟩⟩) 0 ⟨7209⟩ 88710

def event88712 : Event := .predecessor (⟨.program ⟨257⟩, ⟨57243⟩⟩) 1 ⟨57242⟩ 88707

def event88713 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨57243⟩⟩) (.sum [.predecessor 0 88711 .coefficient, .predecessor 1 88712 .coefficient])

def exact88714RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7209⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨57239⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact88714RawTermsValid :
    exact88714RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88714 : Event := .resultExact (⟨.program ⟨257⟩, ⟨57243⟩⟩) exact88714RawTerms .large 88713 .exactZero (none)

def event88715 : Event := .predecessor (⟨.program ⟨257⟩, ⟨59097⟩⟩) 0 ⟨57243⟩ 88714

def event88716 : Event := .predecessor (⟨.program ⟨257⟩, ⟨59097⟩⟩) 1 ⟨59092⟩ 88699

def event88717 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨59097⟩⟩) (.sum [.predecessor 0 88715 .coefficient, .predecessor 1 88716 .coefficient])

def exact88718RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7185⟩⟩, ⟨.program ⟨257⟩, ⟨59091⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7209⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨56896⟩⟩], [⟨.program ⟨257⟩, ⟨58174⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨57239⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact88718RawTermsValid :
    exact88718RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88718 : Event := .resultExact (⟨.program ⟨257⟩, ⟨59097⟩⟩) exact88718RawTerms .large 88717 .exactZero (none)

def event88719 : Event := .preFoldPolynomial 88718 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7185⟩⟩, ⟨.program ⟨257⟩, ⟨59091⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7209⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨56896⟩⟩], [⟨.program ⟨257⟩, ⟨58174⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨57239⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact88720RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7185⟩⟩, ⟨.program ⟨257⟩, ⟨59091⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7209⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨56896⟩⟩], [⟨.program ⟨257⟩, ⟨58174⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨57239⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event88720 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨59097⟩⟩) 88719 exact88720RawTerms .large 88717 .exactZero (none)

def event88721 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨56897⟩⟩) ⟨⟨88⟩, ⟨69⟩, ⟨135⟩⟩ ⟨88563, 88721⟩

def event88722 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨57835⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57832⟩⟩]⟩) (1) 0 2 (.universal 88721 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57832⟩⟩]⟩) (none) 88720)

def event88723 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨57835⟩⟩, .relation 88722 1, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7209⟩⟩]⟩, (1)⟩)

def event88724 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨57835⟩⟩, .relation 88722 0, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7185⟩⟩, ⟨.program ⟨257⟩, ⟨59091⟩⟩]⟩, (-1)⟩)

def event88725 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨57835⟩⟩, .relation 88722 2, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨56896⟩⟩], [⟨.program ⟨257⟩, ⟨58174⟩⟩]⟩, (1)⟩)

def event88726 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨57835⟩⟩, .relation 88722 3, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨57239⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact88727RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7185⟩⟩, ⟨.program ⟨257⟩, ⟨59091⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7209⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨56896⟩⟩], [⟨.program ⟨257⟩, ⟨58174⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨57239⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact88727RawTermsValid :
    exact88727RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88727 : Event := .resultExact (⟨.program ⟨257⟩, ⟨57835⟩⟩) exact88727RawTerms .large 88559 (.finite 202072841853861888) (some (88561))

def event88728 : Event := .predecessor (⟨.program ⟨257⟩, ⟨59094⟩⟩) 0 ⟨57835⟩ 88727

def event88729 : Event := .predecessor (⟨.program ⟨257⟩, ⟨59094⟩⟩) 1 ⟨59093⟩ 88549

def event88730 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨59094⟩⟩) (.sum [.predecessor 0 88728 .coefficient, .predecessor 1 88729 .coefficient])

def event88731 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨59094⟩⟩, .operator (⟨88727, 0⟩, ⟨88549, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7185⟩⟩, ⟨.program ⟨257⟩, ⟨59091⟩⟩]⟩, (1)⟩)

def event88732 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨59094⟩⟩, .operator (⟨88727, 2⟩, ⟨88549, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨56896⟩⟩], [⟨.program ⟨257⟩, ⟨58174⟩⟩]⟩, (-1)⟩)

def event88733 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨59094⟩⟩) (.sum [.result 88727 .summary, .result 88549 .summary])

def exact88734RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7209⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨57239⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact88734RawTermsValid :
    exact88734RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88734 : Event := .resultExact (⟨.program ⟨257⟩, ⟨59094⟩⟩) exact88734RawTerms .large 88730 (.finite 32190182365603518530196853751808) (some (88733))

def event88735 : Event := .predecessor (⟨.program ⟨257⟩, ⟨59095⟩⟩) 0 ⟨59094⟩ 88734

def event88736 : Event := .predecessor (⟨.program ⟨257⟩, ⟨59095⟩⟩) 1 ⟨7108⟩ 15762

def event88737 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨59095⟩⟩) (.product (.predecessor 0 88735 .coefficient) (.predecessor 1 88736 .coefficient) (⟨false, false, none, none, none⟩))

def event88738 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨59095⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨7107⟩⟩]⟩) [⟨.result 15758 .coefficient, false, none⟩])

def event88739 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨59095⟩⟩) (.product (.result 88734 .summary) (.transfer 88738) (⟨false, false, none, none, none⟩))

def event88740 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨59095⟩⟩, .operator (⟨88734, 0⟩, ⟨15762, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7209⟩⟩, ⟨.program ⟨257⟩, ⟨7107⟩⟩]⟩, (1)⟩)

def event88741 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨59095⟩⟩, .operator (⟨88734, 1⟩, ⟨15762, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨57239⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7107⟩⟩]⟩, (-1)⟩)

def event88742 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨59095⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨57239⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7107⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨7107⟩⟩) ⟨7019⟩ 15755)

def event88743 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨59095⟩⟩, .relation 88742 0, ⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨57239⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact88744RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨57239⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7209⟩⟩, ⟨.program ⟨257⟩, ⟨7107⟩⟩]⟩, (1)⟩]

theorem exact88744RawTermsValid :
    exact88744RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88744 : Event := .resultExact (⟨.program ⟨257⟩, ⟨59095⟩⟩) exact88744RawTerms .large 88737 (.finite 345639451281357568474313688265275652177920) (some (88739))

def event88745 : Event := .predecessor (⟨.program ⟨257⟩, ⟨55194⟩⟩) 0 ⟨7177⟩ 15500

def event88746 : Event := .predecessor (⟨.program ⟨257⟩, ⟨55194⟩⟩) 1 ⟨55193⟩ 81681

def event88747 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨55194⟩⟩) (.authority (.operator))

def exact88748RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨55194⟩⟩]⟩, (1)⟩]

theorem exact88748RawTermsValid :
    exact88748RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88748 : Event := .resultExact (⟨.program ⟨257⟩, ⟨55194⟩⟩) exact88748RawTerms .large 88747 .exactZero (none)

def event88749 : Event := .predecessor (⟨.program ⟨257⟩, ⟨56111⟩⟩) 0 ⟨55194⟩ 88748

def event88750 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨56111⟩⟩) (.authority (.operator))

def exact88751RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨56111⟩⟩]⟩, (1)⟩]

theorem exact88751RawTermsValid :
    exact88751RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88751 : Event := .resultExact (⟨.program ⟨257⟩, ⟨56111⟩⟩) exact88751RawTerms (.finite 8192) 88750 .exactZero (none)

def event88752 : Event := .predecessor (⟨.program ⟨257⟩, ⟨56113⟩⟩) 0 ⟨55567⟩ 81965

def event88753 : Event := .predecessor (⟨.program ⟨257⟩, ⟨56113⟩⟩) 1 ⟨56111⟩ 88751

def event88754 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨56113⟩⟩) (.product (.predecessor 0 88752 .coefficient) (.predecessor 1 88753 .coefficient) (⟨false, false, none, none, none⟩))

def event88755 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨56113⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨56111⟩⟩]⟩) [⟨.result 88751 .coefficient, false, none⟩])

def event88756 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨56113⟩⟩) (.product (.result 81965 .summary) (.transfer 88755) (⟨false, false, none, none, none⟩))

def event88757 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨56113⟩⟩, .operator (⟨81965, 0⟩, ⟨88751, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨56111⟩⟩]⟩, (1)⟩)

def event88758 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨56113⟩⟩, .operator (⟨81965, 1⟩, ⟨88751, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨53916⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨56111⟩⟩]⟩, (-1)⟩)

def event88759 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨56113⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨53916⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨56111⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨56111⟩⟩) ⟨55194⟩ 88748)

def event88760 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨56113⟩⟩, .relation 88759 0, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨53916⟩⟩], [⟨.program ⟨257⟩, ⟨55194⟩⟩]⟩, (-1)⟩)

def exact88761RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨56111⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨53916⟩⟩], [⟨.program ⟨257⟩, ⟨55194⟩⟩]⟩, (-1)⟩]

theorem exact88761RawTermsValid :
    exact88761RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88761 : Event := .resultExact (⟨.program ⟨257⟩, ⟨56113⟩⟩) exact88761RawTerms .large 88754 (.finite 32189789464711941702873220382720) (some (88756))

def event88762 : Event := .predecessor (⟨.program ⟨257⟩, ⟨54852⟩⟩) 0 ⟨53917⟩ 3379

def event88763 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨54852⟩⟩) (.authority (.relationPreimageSource ⟨67⟩))

def exact88764RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨54852⟩⟩]⟩, (1)⟩]

theorem exact88764RawTermsValid :
    exact88764RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88764 : Event := .resultExact (⟨.program ⟨257⟩, ⟨54852⟩⟩) exact88764RawTerms (.finite 5647228698) 88763 .exactZero (none)

def event88765 : Event := .predecessor (⟨.program ⟨257⟩, ⟨54854⟩⟩) 0 ⟨54852⟩ 88764

def event88766 : Event := .predecessor (⟨.program ⟨257⟩, ⟨54854⟩⟩) 1 ⟨2370⟩ 4

def event88767 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨54854⟩⟩) (.scale (.predecessor 0 88765 .coefficient) (.value (.predecessor 1 88766 .coefficient)))

def exact88768RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨54852⟩⟩]⟩, (1)⟩]

theorem exact88768RawTermsValid :
    exact88768RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88768 : Event := .resultExact (⟨.program ⟨257⟩, ⟨54854⟩⟩) exact88768RawTerms (.finite 5647228698) 88767 .exactZero (none)

def event88769 : Event := .predecessor (⟨.program ⟨257⟩, ⟨54855⟩⟩) 0 ⟨10368⟩ 75995

def event88770 : Event := .predecessor (⟨.program ⟨257⟩, ⟨54855⟩⟩) 1 ⟨54854⟩ 88768

def event88771 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨54855⟩⟩) (.product (.predecessor 0 88769 .coefficient) (.predecessor 1 88770 .coefficient) (⟨false, false, none, none, none⟩))

def event88772 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨54855⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨54852⟩⟩]⟩) [⟨.result 88764 .coefficient, false, none⟩])

def event88773 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨54855⟩⟩) (.product (.result 75995 .summary) (.transfer 88772) (⟨false, false, none, none, none⟩))

def event88774 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨54855⟩⟩, .operator (⟨75995, 0⟩, ⟨88768, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54852⟩⟩]⟩, (1)⟩)

def event88775 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨54853⟩⟩)

def event88776 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event88777 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event88778 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10267⟩⟩) (.authority (.operator))

def event88779 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10267⟩⟩) (.finite 15)

def event88780 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event88781 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event88782 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event88783 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event88784 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 88783

def event88785 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 88781

def event88786 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 88784 .coefficient) (.value (.predecessor 1 88785 .coefficient)))

def event88787 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event88788 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10269⟩⟩) 0 ⟨392⟩ 88787

def event88789 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10269⟩⟩) 1 ⟨10267⟩ 88779

def event88790 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10269⟩⟩) (.sum [.predecessor 0 88788 .coefficient, .predecessor 1 88789 .coefficient])

def event88791 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10269⟩⟩) (.finite 655355)

def event88792 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10325⟩⟩) 0 ⟨10269⟩ 88791

def event88793 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10325⟩⟩) 1 ⟨5426⟩ 88777

def event88794 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10325⟩⟩) (.identity (.predecessor 1 88793 .coefficient))

def event88795 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10325⟩⟩) (.finite 655360)

def event88796 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24842⟩⟩) 0 ⟨10325⟩ 88795

def event88797 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨24842⟩⟩) (.authority (.programFamilyFact))

def exact88798RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨24842⟩⟩], []⟩, (1)⟩]

theorem exact88798RawTermsValid :
    exact88798RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88798 : Event := .resultExact (⟨.program ⟨257⟩, ⟨24842⟩⟩) exact88798RawTerms (.finite 12) 88797 .exactZero (none)

def event88799 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53687⟩⟩) 0 ⟨10325⟩ 88795

def event88800 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53687⟩⟩) (.authority (.programFamilyFact))

def exact88801RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨53687⟩⟩], []⟩, (1)⟩]

theorem exact88801RawTermsValid :
    exact88801RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88801 : Event := .resultExact (⟨.program ⟨257⟩, ⟨53687⟩⟩) exact88801RawTerms (.finite 12) 88800 .exactZero (none)

def event88802 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53688⟩⟩) 0 ⟨53687⟩ 88801

def event88803 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53688⟩⟩) 1 ⟨24842⟩ 88798

def event88804 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53688⟩⟩) (.product (.predecessor 0 88802 .coefficient) (.predecessor 1 88803 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event88805 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53688⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨24842⟩⟩, ⟨.program ⟨257⟩, ⟨53687⟩⟩], []⟩) [⟨.result 88801 .coefficient, true, some 1⟩, ⟨.result 88798 .coefficient, true, some 1⟩])

def event88806 : Event := .survivorFold (1) 88805

def exact88807RawTerms : List Term := []

theorem exact88807RawTermsValid :
    exact88807RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88807 : Event := .resultExact (⟨.program ⟨257⟩, ⟨53688⟩⟩) exact88807RawTerms (.finite 144) 88804 (.finite 144) (some (88805))

def event88808 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53689⟩⟩) 0 ⟨53688⟩ 88807

def event88809 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53689⟩⟩) (.identity (.predecessor 0 88808 .coefficient))

def event88810 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨53689⟩⟩) (.finite 144)

def event88811 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53916⟩⟩) 0 ⟨53689⟩ 88810

def event88812 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53916⟩⟩) (.authority (.programFamilyFact))

def exact88813RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨53916⟩⟩], []⟩, (1)⟩]

theorem exact88813RawTermsValid :
    exact88813RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88813 : Event := .resultExact (⟨.program ⟨257⟩, ⟨53916⟩⟩) exact88813RawTerms (.finite 12) 88812 .exactZero (none)

def event88814 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53917⟩⟩) 0 ⟨53916⟩ 88813

def event88815 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53917⟩⟩) (.identity (.predecessor 0 88814 .coefficient))

def event88816 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨53917⟩⟩) (.finite 12)

def event88817 : Event := .predecessor (⟨.program ⟨257⟩, ⟨54852⟩⟩) 0 ⟨53917⟩ 88816

def event88818 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨54852⟩⟩) (.authority (.relationPreimageSource ⟨67⟩))

def exact88819RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨54852⟩⟩]⟩, (1)⟩]

theorem exact88819RawTermsValid :
    exact88819RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88819 : Event := .resultExact (⟨.program ⟨257⟩, ⟨54852⟩⟩) exact88819RawTerms (.finite 5647228698) 88818 .exactZero (none)

def event88820 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35⟩⟩) (.authority (.operator))

def exact88821RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact88821RawTermsValid :
    exact88821RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88821 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35⟩⟩) exact88821RawTerms .large 88820 .exactZero (none)

def event88822 : Event := .predecessor (⟨.program ⟨257⟩, ⟨54853⟩⟩) 0 ⟨35⟩ 88821

def event88823 : Event := .predecessor (⟨.program ⟨257⟩, ⟨54853⟩⟩) 1 ⟨54852⟩ 88819

def event88824 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨54853⟩⟩) (.product (.predecessor 0 88822 .coefficient) (.predecessor 1 88823 .coefficient) (⟨false, false, none, none, none⟩))

def event88825 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨54853⟩⟩, .operator (⟨88821, 0⟩, ⟨88819, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54852⟩⟩]⟩, (1)⟩)

def exact88826RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54852⟩⟩]⟩, (1)⟩]

theorem exact88826RawTermsValid :
    exact88826RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88826 : Event := .resultExact (⟨.program ⟨257⟩, ⟨54853⟩⟩) exact88826RawTerms .large 88824 .exactZero (none)

def event88827 : Event := .preFoldPolynomial 88826 [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54852⟩⟩]⟩, (1)⟩] .exactZero none

def exact88828RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54852⟩⟩]⟩, (1)⟩]

def event88828 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨54853⟩⟩) 88827 exact88828RawTerms .large 88824 .exactZero (none)

def event88829 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨56117⟩⟩)

def event88830 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event88831 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def eventLeaf5536 : Array AnnotatedEvent := #[
  { event := event88576
    frameStart := 88563 },
  { event := event88577
    frameStart := 88563 },
  { event := event88578
    frameStart := 88563 },
  { event := event88579
    frameStart := 88563 },
  { event := event88580
    frameStart := 88563 },
  { event := event88581
    frameStart := 88563 },
  { event := event88582
    frameStart := 88563 },
  { event := event88583
    frameStart := 88563 },
  { event := event88584
    frameStart := 88563 },
  { event := event88585
    frameStart := 88563 },
  { event := event88586
    frameStart := 88563 },
  { event := event88587
    frameStart := 88563 },
  { event := event88588
    frameStart := 88563 },
  { event := event88589
    frameStart := 88563 },
  { event := event88590
    frameStart := 88563 },
  { event := event88591
    frameStart := 88563 }
]

def eventLeaf5537 : Array AnnotatedEvent := #[
  { event := event88592
    frameStart := 88563 },
  { event := event88593
    frameStart := 88563 },
  { event := event88594
    frameStart := 88563 },
  { event := event88595
    frameStart := 88563 },
  { event := event88596
    frameStart := 88563 },
  { event := event88597
    frameStart := 88563 },
  { event := event88598
    frameStart := 88563 },
  { event := event88599
    frameStart := 88563 },
  { event := event88600
    frameStart := 88563 },
  { event := event88601
    frameStart := 88563 },
  { event := event88602
    frameStart := 88563 },
  { event := event88603
    frameStart := 88563 },
  { event := event88604
    frameStart := 88563 },
  { event := event88605
    frameStart := 88563 },
  { event := event88606
    frameStart := 88563 },
  { event := event88607
    frameStart := 88563 }
]

def eventLeaf5538 : Array AnnotatedEvent := #[
  { event := event88608
    frameStart := 88563 },
  { event := event88609
    frameStart := 88563 },
  { event := event88610
    frameStart := 88563 },
  { event := event88611
    frameStart := 88563 },
  { event := event88612
    frameStart := 88563 },
  { event := event88613
    frameStart := 88563 },
  { event := event88614
    frameStart := 88563 },
  { event := event88615
    frameStart := 88563 },
  { event := event88616
    frameStart := 88563 },
  { event := event88617
    frameStart := 88617 },
  { event := event88618
    frameStart := 88617 },
  { event := event88619
    frameStart := 88617 },
  { event := event88620
    frameStart := 88617 },
  { event := event88621
    frameStart := 88617 },
  { event := event88622
    frameStart := 88617 },
  { event := event88623
    frameStart := 88617 }
]

def eventLeaf5539 : Array AnnotatedEvent := #[
  { event := event88624
    frameStart := 88617 },
  { event := event88625
    frameStart := 88617 },
  { event := event88626
    frameStart := 88617 },
  { event := event88627
    frameStart := 88617 },
  { event := event88628
    frameStart := 88617 },
  { event := event88629
    frameStart := 88617 },
  { event := event88630
    frameStart := 88617 },
  { event := event88631
    frameStart := 88617 },
  { event := event88632
    frameStart := 88617 },
  { event := event88633
    frameStart := 88617 },
  { event := event88634
    frameStart := 88617 },
  { event := event88635
    frameStart := 88617 },
  { event := event88636
    frameStart := 88617 },
  { event := event88637
    frameStart := 88617 },
  { event := event88638
    frameStart := 88617 },
  { event := event88639
    frameStart := 88617 }
]

def eventLeaf5540 : Array AnnotatedEvent := #[
  { event := event88640
    frameStart := 88617 },
  { event := event88641
    frameStart := 88617 },
  { event := event88642
    frameStart := 88617 },
  { event := event88643
    frameStart := 88617 },
  { event := event88644
    frameStart := 88617 },
  { event := event88645
    frameStart := 88617 },
  { event := event88646
    frameStart := 88617 },
  { event := event88647
    frameStart := 88617 },
  { event := event88648
    frameStart := 88617 },
  { event := event88649
    frameStart := 88617 },
  { event := event88650
    frameStart := 88617 },
  { event := event88651
    frameStart := 88617 },
  { event := event88652
    frameStart := 88617 },
  { event := event88653
    frameStart := 88617 },
  { event := event88654
    frameStart := 88617 },
  { event := event88655
    frameStart := 88617 }
]

def eventLeaf5541 : Array AnnotatedEvent := #[
  { event := event88656
    frameStart := 88617 },
  { event := event88657
    frameStart := 88617 },
  { event := event88658
    frameStart := 88617 },
  { event := event88659
    frameStart := 88617 },
  { event := event88660
    frameStart := 88617 },
  { event := event88661
    frameStart := 88617 },
  { event := event88662
    frameStart := 88617 },
  { event := event88663
    frameStart := 88617 },
  { event := event88664
    frameStart := 88617 },
  { event := event88665
    frameStart := 88617 },
  { event := event88666
    frameStart := 88617 },
  { event := event88667
    frameStart := 88617 },
  { event := event88668
    frameStart := 88617 },
  { event := event88669
    frameStart := 88617 },
  { event := event88670
    frameStart := 88617 },
  { event := event88671
    frameStart := 88617 }
]

def eventLeaf5542 : Array AnnotatedEvent := #[
  { event := event88672
    frameStart := 88617 },
  { event := event88673
    frameStart := 88617 },
  { event := event88674
    frameStart := 88617 },
  { event := event88675
    frameStart := 88617 },
  { event := event88676
    frameStart := 88617 },
  { event := event88677
    frameStart := 88617 },
  { event := event88678
    frameStart := 88617 },
  { event := event88679
    frameStart := 88617 },
  { event := event88680
    frameStart := 88617 },
  { event := event88681
    frameStart := 88617 },
  { event := event88682
    frameStart := 88617 },
  { event := event88683
    frameStart := 88617 },
  { event := event88684
    frameStart := 88617 },
  { event := event88685
    frameStart := 88617 },
  { event := event88686
    frameStart := 88617 },
  { event := event88687
    frameStart := 88617 }
]

def eventLeaf5543 : Array AnnotatedEvent := #[
  { event := event88688
    frameStart := 88617 },
  { event := event88689
    frameStart := 88617 },
  { event := event88690
    frameStart := 88617 },
  { event := event88691
    frameStart := 88617 },
  { event := event88692
    frameStart := 88617 },
  { event := event88693
    frameStart := 88617 },
  { event := event88694
    frameStart := 88617 },
  { event := event88695
    frameStart := 88617 },
  { event := event88696
    frameStart := 88617 },
  { event := event88697
    frameStart := 88617 },
  { event := event88698
    frameStart := 88617 },
  { event := event88699
    frameStart := 88617 },
  { event := event88700
    frameStart := 88617 },
  { event := event88701
    frameStart := 88617 },
  { event := event88702
    frameStart := 88617 },
  { event := event88703
    frameStart := 88617 }
]

def eventLeaf5544 : Array AnnotatedEvent := #[
  { event := event88704
    frameStart := 88617 },
  { event := event88705
    frameStart := 88617 },
  { event := event88706
    frameStart := 88617 },
  { event := event88707
    frameStart := 88617 },
  { event := event88708
    frameStart := 88617 },
  { event := event88709
    frameStart := 88617 },
  { event := event88710
    frameStart := 88617 },
  { event := event88711
    frameStart := 88617 },
  { event := event88712
    frameStart := 88617 },
  { event := event88713
    frameStart := 88617 },
  { event := event88714
    frameStart := 88617 },
  { event := event88715
    frameStart := 88617 },
  { event := event88716
    frameStart := 88617 },
  { event := event88717
    frameStart := 88617 },
  { event := event88718
    frameStart := 88617 },
  { event := event88719
    frameStart := 88617 }
]

def eventLeaf5545 : Array AnnotatedEvent := #[
  { event := event88720
    frameStart := 88617 },
  { event := event88721
    frameStart := 0 },
  { event := event88722
    frameStart := 0 },
  { event := event88723
    frameStart := 0 },
  { event := event88724
    frameStart := 0 },
  { event := event88725
    frameStart := 0 },
  { event := event88726
    frameStart := 0 },
  { event := event88727
    frameStart := 0 },
  { event := event88728
    frameStart := 0 },
  { event := event88729
    frameStart := 0 },
  { event := event88730
    frameStart := 0 },
  { event := event88731
    frameStart := 0 },
  { event := event88732
    frameStart := 0 },
  { event := event88733
    frameStart := 0 },
  { event := event88734
    frameStart := 0 },
  { event := event88735
    frameStart := 0 }
]

def eventLeaf5546 : Array AnnotatedEvent := #[
  { event := event88736
    frameStart := 0 },
  { event := event88737
    frameStart := 0 },
  { event := event88738
    frameStart := 0 },
  { event := event88739
    frameStart := 0 },
  { event := event88740
    frameStart := 0 },
  { event := event88741
    frameStart := 0 },
  { event := event88742
    frameStart := 0 },
  { event := event88743
    frameStart := 0 },
  { event := event88744
    frameStart := 0 },
  { event := event88745
    frameStart := 0 },
  { event := event88746
    frameStart := 0 },
  { event := event88747
    frameStart := 0 },
  { event := event88748
    frameStart := 0 },
  { event := event88749
    frameStart := 0 },
  { event := event88750
    frameStart := 0 },
  { event := event88751
    frameStart := 0 }
]

def eventLeaf5547 : Array AnnotatedEvent := #[
  { event := event88752
    frameStart := 0 },
  { event := event88753
    frameStart := 0 },
  { event := event88754
    frameStart := 0 },
  { event := event88755
    frameStart := 0 },
  { event := event88756
    frameStart := 0 },
  { event := event88757
    frameStart := 0 },
  { event := event88758
    frameStart := 0 },
  { event := event88759
    frameStart := 0 },
  { event := event88760
    frameStart := 0 },
  { event := event88761
    frameStart := 0 },
  { event := event88762
    frameStart := 0 },
  { event := event88763
    frameStart := 0 },
  { event := event88764
    frameStart := 0 },
  { event := event88765
    frameStart := 0 },
  { event := event88766
    frameStart := 0 },
  { event := event88767
    frameStart := 0 }
]

def eventLeaf5548 : Array AnnotatedEvent := #[
  { event := event88768
    frameStart := 0 },
  { event := event88769
    frameStart := 0 },
  { event := event88770
    frameStart := 0 },
  { event := event88771
    frameStart := 0 },
  { event := event88772
    frameStart := 0 },
  { event := event88773
    frameStart := 0 },
  { event := event88774
    frameStart := 0 },
  { event := event88775
    frameStart := 88775 },
  { event := event88776
    frameStart := 88775 },
  { event := event88777
    frameStart := 88775 },
  { event := event88778
    frameStart := 88775 },
  { event := event88779
    frameStart := 88775 },
  { event := event88780
    frameStart := 88775 },
  { event := event88781
    frameStart := 88775 },
  { event := event88782
    frameStart := 88775 },
  { event := event88783
    frameStart := 88775 }
]

def eventLeaf5549 : Array AnnotatedEvent := #[
  { event := event88784
    frameStart := 88775 },
  { event := event88785
    frameStart := 88775 },
  { event := event88786
    frameStart := 88775 },
  { event := event88787
    frameStart := 88775 },
  { event := event88788
    frameStart := 88775 },
  { event := event88789
    frameStart := 88775 },
  { event := event88790
    frameStart := 88775 },
  { event := event88791
    frameStart := 88775 },
  { event := event88792
    frameStart := 88775 },
  { event := event88793
    frameStart := 88775 },
  { event := event88794
    frameStart := 88775 },
  { event := event88795
    frameStart := 88775 },
  { event := event88796
    frameStart := 88775 },
  { event := event88797
    frameStart := 88775 },
  { event := event88798
    frameStart := 88775 },
  { event := event88799
    frameStart := 88775 }
]

def eventLeaf5550 : Array AnnotatedEvent := #[
  { event := event88800
    frameStart := 88775 },
  { event := event88801
    frameStart := 88775 },
  { event := event88802
    frameStart := 88775 },
  { event := event88803
    frameStart := 88775 },
  { event := event88804
    frameStart := 88775 },
  { event := event88805
    frameStart := 88775 },
  { event := event88806
    frameStart := 88775 },
  { event := event88807
    frameStart := 88775 },
  { event := event88808
    frameStart := 88775 },
  { event := event88809
    frameStart := 88775 },
  { event := event88810
    frameStart := 88775 },
  { event := event88811
    frameStart := 88775 },
  { event := event88812
    frameStart := 88775 },
  { event := event88813
    frameStart := 88775 },
  { event := event88814
    frameStart := 88775 },
  { event := event88815
    frameStart := 88775 }
]

def eventLeaf5551 : Array AnnotatedEvent := #[
  { event := event88816
    frameStart := 88775 },
  { event := event88817
    frameStart := 88775 },
  { event := event88818
    frameStart := 88775 },
  { event := event88819
    frameStart := 88775 },
  { event := event88820
    frameStart := 88775 },
  { event := event88821
    frameStart := 88775 },
  { event := event88822
    frameStart := 88775 },
  { event := event88823
    frameStart := 88775 },
  { event := event88824
    frameStart := 88775 },
  { event := event88825
    frameStart := 88775 },
  { event := event88826
    frameStart := 88775 },
  { event := event88827
    frameStart := 88775 },
  { event := event88828
    frameStart := 88775 },
  { event := event88829
    frameStart := 88829 },
  { event := event88830
    frameStart := 88829 },
  { event := event88831
    frameStart := 88829 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events346
