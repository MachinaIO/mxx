import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events928

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event237568 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event237569 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event237570 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event237571 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 237570

def event237572 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 237568

def event237573 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 237571 .coefficient) (.value (.predecessor 1 237572 .coefficient)))

def event237574 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event237575 : Event := .predecessor (⟨.program ⟨257⟩, ⟨4742⟩⟩) 0 ⟨392⟩ 237574

def event237576 : Event := .predecessor (⟨.program ⟨257⟩, ⟨4742⟩⟩) 1 ⟨4740⟩ 237566

def event237577 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨4742⟩⟩) (.sum [.predecessor 0 237575 .coefficient, .predecessor 1 237576 .coefficient])

def event237578 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨4742⟩⟩) (.finite 655344)

def event237579 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5559⟩⟩) 0 ⟨4742⟩ 237578

def event237580 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5559⟩⟩) 1 ⟨5426⟩ 237564

def event237581 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5559⟩⟩) (.identity (.predecessor 1 237580 .coefficient))

def event237582 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5559⟩⟩) (.finite 655360)

def event237583 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45106⟩⟩) 0 ⟨5559⟩ 237582

def event237584 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45106⟩⟩) (.authority (.programFamilyFact))

def exact237585RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨45106⟩⟩], []⟩, (1)⟩]

theorem exact237585RawTermsValid :
    exact237585RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event237585 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45106⟩⟩) exact237585RawTerms (.finite 58) 237584 .exactZero (none)

def event237586 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14751⟩⟩) 0 ⟨5559⟩ 237582

def event237587 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14751⟩⟩) (.authority (.programFamilyFact))

def exact237588RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14751⟩⟩], []⟩, (1)⟩]

theorem exact237588RawTermsValid :
    exact237588RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event237588 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14751⟩⟩) exact237588RawTerms (.finite 58) 237587 .exactZero (none)

def event237589 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45107⟩⟩) 0 ⟨14751⟩ 237588

def event237590 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45107⟩⟩) 1 ⟨45106⟩ 237585

def event237591 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45107⟩⟩) (.product (.predecessor 0 237589 .coefficient) (.predecessor 1 237590 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event237592 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45107⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨14751⟩⟩, ⟨.program ⟨257⟩, ⟨45106⟩⟩], []⟩) [⟨.result 237588 .coefficient, true, some 1⟩, ⟨.result 237585 .coefficient, true, some 1⟩])

def event237593 : Event := .survivorFold (1) 237592

def exact237594RawTerms : List Term := []

theorem exact237594RawTermsValid :
    exact237594RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event237594 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45107⟩⟩) exact237594RawTerms (.finite 3364) 237591 (.finite 3364) (some (237592))

def event237595 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45108⟩⟩) 0 ⟨45107⟩ 237594

def event237596 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45108⟩⟩) (.identity (.predecessor 0 237595 .coefficient))

def event237597 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨45108⟩⟩) (.finite 3364)

def event237598 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45452⟩⟩) 0 ⟨45108⟩ 237597

def event237599 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45452⟩⟩) (.authority (.programFamilyFact))

def exact237600RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨45452⟩⟩], []⟩, (1)⟩]

theorem exact237600RawTermsValid :
    exact237600RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event237600 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45452⟩⟩) exact237600RawTerms (.finite 58) 237599 .exactZero (none)

def event237601 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45453⟩⟩) 0 ⟨45452⟩ 237600

def event237602 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45453⟩⟩) (.identity (.predecessor 0 237601 .coefficient))

def event237603 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨45453⟩⟩) (.finite 58)

def event237604 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46176⟩⟩) 0 ⟨45453⟩ 237603

def event237605 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46176⟩⟩) (.authority (.relationPreimageSource ⟨92⟩))

def exact237606RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨46176⟩⟩]⟩, (1)⟩]

theorem exact237606RawTermsValid :
    exact237606RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event237606 : Event := .resultExact (⟨.program ⟨257⟩, ⟨46176⟩⟩) exact237606RawTerms (.finite 5647228698) 237605 .exactZero (none)

def event237607 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35⟩⟩) (.authority (.operator))

def exact237608RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact237608RawTermsValid :
    exact237608RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event237608 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35⟩⟩) exact237608RawTerms .large 237607 .exactZero (none)

def event237609 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46177⟩⟩) 0 ⟨35⟩ 237608

def event237610 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46177⟩⟩) 1 ⟨46176⟩ 237606

def event237611 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46177⟩⟩) (.product (.predecessor 0 237609 .coefficient) (.predecessor 1 237610 .coefficient) (⟨false, false, none, none, none⟩))

def event237612 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨46177⟩⟩, .operator (⟨237608, 0⟩, ⟨237606, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46176⟩⟩]⟩, (1)⟩)

def exact237613RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46176⟩⟩]⟩, (1)⟩]

theorem exact237613RawTermsValid :
    exact237613RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event237613 : Event := .resultExact (⟨.program ⟨257⟩, ⟨46177⟩⟩) exact237613RawTerms .large 237611 .exactZero (none)

def event237614 : Event := .preFoldPolynomial 237613 [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46176⟩⟩]⟩, (1)⟩] .exactZero none

def exact237615RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46176⟩⟩]⟩, (1)⟩]

def event237615 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨46177⟩⟩) 237614 exact237615RawTerms .large 237611 .exactZero (none)

def event237616 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨47303⟩⟩)

def event237617 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event237618 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event237619 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨4740⟩⟩) (.authority (.operator))

def event237620 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨4740⟩⟩) (.finite 4)

def event237621 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event237622 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event237623 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event237624 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event237625 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 237624

def event237626 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 237622

def event237627 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 237625 .coefficient) (.value (.predecessor 1 237626 .coefficient)))

def event237628 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event237629 : Event := .predecessor (⟨.program ⟨257⟩, ⟨4742⟩⟩) 0 ⟨392⟩ 237628

def event237630 : Event := .predecessor (⟨.program ⟨257⟩, ⟨4742⟩⟩) 1 ⟨4740⟩ 237620

def event237631 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨4742⟩⟩) (.sum [.predecessor 0 237629 .coefficient, .predecessor 1 237630 .coefficient])

def event237632 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨4742⟩⟩) (.finite 655344)

def event237633 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5559⟩⟩) 0 ⟨4742⟩ 237632

def event237634 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5559⟩⟩) 1 ⟨5426⟩ 237618

def event237635 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5559⟩⟩) (.identity (.predecessor 1 237634 .coefficient))

def event237636 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5559⟩⟩) (.finite 655360)

def event237637 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45106⟩⟩) 0 ⟨5559⟩ 237636

def event237638 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45106⟩⟩) (.authority (.programFamilyFact))

def exact237639RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨45106⟩⟩], []⟩, (1)⟩]

theorem exact237639RawTermsValid :
    exact237639RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event237639 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45106⟩⟩) exact237639RawTerms (.finite 58) 237638 .exactZero (none)

def event237640 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14751⟩⟩) 0 ⟨5559⟩ 237636

def event237641 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14751⟩⟩) (.authority (.programFamilyFact))

def exact237642RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14751⟩⟩], []⟩, (1)⟩]

theorem exact237642RawTermsValid :
    exact237642RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event237642 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14751⟩⟩) exact237642RawTerms (.finite 58) 237641 .exactZero (none)

def event237643 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45107⟩⟩) 0 ⟨14751⟩ 237642

def event237644 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45107⟩⟩) 1 ⟨45106⟩ 237639

def event237645 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45107⟩⟩) (.product (.predecessor 0 237643 .coefficient) (.predecessor 1 237644 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event237646 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨45107⟩⟩, .operator (⟨237642, 0⟩, ⟨237639, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨14751⟩⟩, ⟨.program ⟨257⟩, ⟨45106⟩⟩], []⟩, (1)⟩)

def exact237647RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14751⟩⟩, ⟨.program ⟨257⟩, ⟨45106⟩⟩], []⟩, (1)⟩]

theorem exact237647RawTermsValid :
    exact237647RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event237647 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45107⟩⟩) exact237647RawTerms (.finite 3364) 237645 .exactZero (none)

def event237648 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45108⟩⟩) 0 ⟨45107⟩ 237647

def event237649 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45108⟩⟩) (.identity (.predecessor 0 237648 .coefficient))

def event237650 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨45108⟩⟩) (.finite 3364)

def event237651 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45452⟩⟩) 0 ⟨45108⟩ 237650

def event237652 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45452⟩⟩) (.authority (.programFamilyFact))

def exact237653RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨45452⟩⟩], []⟩, (1)⟩]

theorem exact237653RawTermsValid :
    exact237653RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event237653 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45452⟩⟩) exact237653RawTerms (.finite 58) 237652 .exactZero (none)

def event237654 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45453⟩⟩) 0 ⟨45452⟩ 237653

def event237655 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45453⟩⟩) (.identity (.predecessor 0 237654 .coefficient))

def event237656 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨45453⟩⟩) (.finite 58)

def event237657 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46601⟩⟩) 0 ⟨45453⟩ 237656

def event237658 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46601⟩⟩) (.authority (.programFamilyFact))

def event237659 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨46601⟩⟩) (.finite 3720)

def event237660 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨7177⟩⟩) .missing

def event237661 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46603⟩⟩) 0 ⟨7177⟩ 237660

def event237662 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46603⟩⟩) 1 ⟨46601⟩ 237659

def event237663 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46603⟩⟩) (.authority (.operator))

def exact237664RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨46603⟩⟩]⟩, (1)⟩]

theorem exact237664RawTermsValid :
    exact237664RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event237664 : Event := .resultExact (⟨.program ⟨257⟩, ⟨46603⟩⟩) exact237664RawTerms .large 237663 .exactZero (none)

def event237665 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47299⟩⟩) 0 ⟨46603⟩ 237664

def event237666 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47299⟩⟩) (.authority (.operator))

def exact237667RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨47299⟩⟩]⟩, (1)⟩]

theorem exact237667RawTermsValid :
    exact237667RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event237667 : Event := .resultExact (⟨.program ⟨257⟩, ⟨47299⟩⟩) exact237667RawTerms (.finite 8192) 237666 .exactZero (none)

def event237668 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event237669 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event237670 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46818⟩⟩) 0 ⟨45453⟩ 237656

def event237671 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46818⟩⟩) 1 ⟨136⟩ 237669

def event237672 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46818⟩⟩) (.sum [.predecessor 0 237670 .coefficient, .predecessor 1 237671 .coefficient])

def event237673 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨46818⟩⟩) (.finite 58)

def event237674 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46819⟩⟩) 0 ⟨46818⟩ 237673

def event237675 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46819⟩⟩) (.identity (.predecessor 0 237674 .coefficient))

def exact237676RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨45452⟩⟩], []⟩, (1)⟩]

theorem exact237676RawTermsValid :
    exact237676RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event237676 : Event := .resultExact (⟨.program ⟨257⟩, ⟨46819⟩⟩) exact237676RawTerms (.finite 58) 237675 .exactZero (none)

def event237677 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact237678RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact237678RawTermsValid :
    exact237678RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event237678 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact237678RawTerms .large 237677 .exactZero (none)

def event237679 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46820⟩⟩) 0 ⟨6908⟩ 237678

def event237680 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46820⟩⟩) 1 ⟨46819⟩ 237676

def event237681 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46820⟩⟩) (.product (.predecessor 0 237679 .coefficient) (.predecessor 1 237680 .coefficient) (⟨false, false, none, none, none⟩))

def event237682 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨46820⟩⟩, .operator (⟨237678, 0⟩, ⟨237676, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨45452⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact237683RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨45452⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact237683RawTermsValid :
    exact237683RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event237683 : Event := .resultExact (⟨.program ⟨257⟩, ⟨46820⟩⟩) exact237683RawTerms .large 237681 .exactZero (none)

def event237684 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7195⟩⟩) 0 ⟨7177⟩ 237660

def event237685 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7195⟩⟩) (.authority (.operator))

def exact237686RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7195⟩⟩]⟩, (1)⟩]

theorem exact237686RawTermsValid :
    exact237686RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event237686 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7195⟩⟩) exact237686RawTerms .large 237685 .exactZero (none)

def event237687 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46821⟩⟩) 0 ⟨7195⟩ 237686

def event237688 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46821⟩⟩) 1 ⟨46820⟩ 237683

def event237689 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46821⟩⟩) (.sum [.predecessor 0 237687 .coefficient, .predecessor 1 237688 .coefficient])

def exact237690RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7195⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨45452⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact237690RawTermsValid :
    exact237690RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event237690 : Event := .resultExact (⟨.program ⟨257⟩, ⟨46821⟩⟩) exact237690RawTerms .large 237689 .exactZero (none)

def event237691 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47300⟩⟩) 0 ⟨46821⟩ 237690

def event237692 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47300⟩⟩) 1 ⟨47299⟩ 237667

def event237693 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47300⟩⟩) (.product (.predecessor 0 237691 .coefficient) (.predecessor 1 237692 .coefficient) (⟨false, false, none, none, none⟩))

def event237694 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨47300⟩⟩, .operator (⟨237690, 0⟩, ⟨237667, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47299⟩⟩]⟩, (1)⟩)

def event237695 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨47300⟩⟩, .operator (⟨237690, 1⟩, ⟨237667, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨45452⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨47299⟩⟩]⟩, (-1)⟩)

def event237696 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨47300⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨45452⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨47299⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨47299⟩⟩) ⟨46603⟩ 237664)

def event237697 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨47300⟩⟩, .relation 237696 0, ⟨[⟨.program ⟨257⟩, ⟨45452⟩⟩], [⟨.program ⟨257⟩, ⟨46603⟩⟩]⟩, (-1)⟩)

def exact237698RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47299⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨45452⟩⟩], [⟨.program ⟨257⟩, ⟨46603⟩⟩]⟩, (-1)⟩]

theorem exact237698RawTermsValid :
    exact237698RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event237698 : Event := .resultExact (⟨.program ⟨257⟩, ⟨47300⟩⟩) exact237698RawTerms .large 237693 .exactZero (none)

def event237699 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45657⟩⟩) 0 ⟨45453⟩ 237656

def event237700 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45657⟩⟩) (.authority (.programFamilyFact))

def exact237701RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨45657⟩⟩], []⟩, (1)⟩]

theorem exact237701RawTermsValid :
    exact237701RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event237701 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45657⟩⟩) exact237701RawTerms (.finite 63) 237700 .exactZero (none)

def event237702 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45658⟩⟩) 0 ⟨6908⟩ 237678

def event237703 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45658⟩⟩) 1 ⟨45657⟩ 237701

def event237704 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45658⟩⟩) (.product (.predecessor 0 237702 .coefficient) (.predecessor 1 237703 .coefficient) (⟨false, true, none, none, some 1⟩))

def event237705 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨45658⟩⟩, .operator (⟨237678, 0⟩, ⟨237701, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨45657⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact237706RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨45657⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact237706RawTermsValid :
    exact237706RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event237706 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45658⟩⟩) exact237706RawTerms .large 237704 .exactZero (none)

def event237707 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7230⟩⟩) 0 ⟨7177⟩ 237660

def event237708 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7230⟩⟩) (.authority (.operator))

def exact237709RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7230⟩⟩]⟩, (1)⟩]

theorem exact237709RawTermsValid :
    exact237709RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event237709 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7230⟩⟩) exact237709RawTerms .large 237708 .exactZero (none)

def event237710 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45659⟩⟩) 0 ⟨7230⟩ 237709

def event237711 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45659⟩⟩) 1 ⟨45658⟩ 237706

def event237712 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45659⟩⟩) (.sum [.predecessor 0 237710 .coefficient, .predecessor 1 237711 .coefficient])

def exact237713RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7230⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨45657⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact237713RawTermsValid :
    exact237713RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event237713 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45659⟩⟩) exact237713RawTerms .large 237712 .exactZero (none)

def event237714 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47303⟩⟩) 0 ⟨45659⟩ 237713

def event237715 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47303⟩⟩) 1 ⟨47300⟩ 237698

def event237716 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47303⟩⟩) (.sum [.predecessor 0 237714 .coefficient, .predecessor 1 237715 .coefficient])

def exact237717RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47299⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7230⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨45452⟩⟩], [⟨.program ⟨257⟩, ⟨46603⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨45657⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact237717RawTermsValid :
    exact237717RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event237717 : Event := .resultExact (⟨.program ⟨257⟩, ⟨47303⟩⟩) exact237717RawTerms .large 237716 .exactZero (none)

def event237718 : Event := .preFoldPolynomial 237717 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47299⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7230⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨45452⟩⟩], [⟨.program ⟨257⟩, ⟨46603⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨45657⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact237719RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47299⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7230⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨45452⟩⟩], [⟨.program ⟨257⟩, ⟨46603⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨45657⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event237719 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨47303⟩⟩) 237718 exact237719RawTerms .large 237716 .exactZero (none)

def event237720 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨45453⟩⟩) ⟨⟨109⟩, ⟨92⟩, ⟨135⟩⟩ ⟨237562, 237720⟩

def event237721 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨46179⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46176⟩⟩]⟩) (1) 0 2 (.universal 237720 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46176⟩⟩]⟩) (none) 237719)

def event237722 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨46179⟩⟩, .relation 237721 1, ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7230⟩⟩]⟩, (1)⟩)

def event237723 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨46179⟩⟩, .relation 237721 0, ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47299⟩⟩]⟩, (-1)⟩)

def event237724 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨46179⟩⟩, .relation 237721 2, ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨45452⟩⟩], [⟨.program ⟨257⟩, ⟨46603⟩⟩]⟩, (1)⟩)

def event237725 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨46179⟩⟩, .relation 237721 3, ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨45657⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact237726RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47299⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7230⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨45452⟩⟩], [⟨.program ⟨257⟩, ⟨46603⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨45657⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact237726RawTermsValid :
    exact237726RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event237726 : Event := .resultExact (⟨.program ⟨257⟩, ⟨46179⟩⟩) exact237726RawTerms .large 237558 (.finite 202072841853861888) (some (237560))

def event237727 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47302⟩⟩) 0 ⟨46179⟩ 237726

def event237728 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47302⟩⟩) 1 ⟨47301⟩ 237548

def event237729 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47302⟩⟩) (.sum [.predecessor 0 237727 .coefficient, .predecessor 1 237728 .coefficient])

def event237730 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨47302⟩⟩, .operator (⟨237726, 0⟩, ⟨237548, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47299⟩⟩]⟩, (1)⟩)

def event237731 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨47302⟩⟩, .operator (⟨237726, 2⟩, ⟨237548, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨45452⟩⟩], [⟨.program ⟨257⟩, ⟨46603⟩⟩]⟩, (-1)⟩)

def event237732 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47302⟩⟩) (.sum [.result 237726 .summary, .result 237548 .summary])

def exact237733RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7230⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨45657⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact237733RawTermsValid :
    exact237733RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event237733 : Event := .resultExact (⟨.program ⟨257⟩, ⟨47302⟩⟩) exact237733RawTerms .large 237729 (.finite 32194307824962953452255538577408) (some (237732))

def event237734 : Event := .predecessor (⟨.program ⟨257⟩, ⟨43921⟩⟩) 0 ⟨42773⟩ 11377

def event237735 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨43921⟩⟩) (.authority (.programFamilyFact))

def event237736 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨43921⟩⟩) (.finite 3720)

def event237737 : Event := .predecessor (⟨.program ⟨257⟩, ⟨43923⟩⟩) 0 ⟨7177⟩ 15500

def event237738 : Event := .predecessor (⟨.program ⟨257⟩, ⟨43923⟩⟩) 1 ⟨43921⟩ 237736

def event237739 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨43923⟩⟩) (.authority (.operator))

def exact237740RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨43923⟩⟩]⟩, (1)⟩]

theorem exact237740RawTermsValid :
    exact237740RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event237740 : Event := .resultExact (⟨.program ⟨257⟩, ⟨43923⟩⟩) exact237740RawTerms .large 237739 .exactZero (none)

def event237741 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44619⟩⟩) 0 ⟨43923⟩ 237740

def event237742 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44619⟩⟩) (.authority (.operator))

def exact237743RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨44619⟩⟩]⟩, (1)⟩]

theorem exact237743RawTermsValid :
    exact237743RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event237743 : Event := .resultExact (⟨.program ⟨257⟩, ⟨44619⟩⟩) exact237743RawTerms (.finite 8192) 237742 .exactZero (none)

def event237744 : Event := .predecessor (⟨.program ⟨257⟩, ⟨43776⟩⟩) 0 ⟨42428⟩ 11371

def event237745 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨43776⟩⟩) (.authority (.programFamilyFact))

def event237746 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨43776⟩⟩) (.finite 3720)

def event237747 : Event := .predecessor (⟨.program ⟨257⟩, ⟨43777⟩⟩) 0 ⟨7177⟩ 15500

def event237748 : Event := .predecessor (⟨.program ⟨257⟩, ⟨43777⟩⟩) 1 ⟨43776⟩ 237746

def event237749 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨43777⟩⟩) (.authority (.operator))

def exact237750RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨43777⟩⟩]⟩, (1)⟩]

theorem exact237750RawTermsValid :
    exact237750RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event237750 : Event := .resultExact (⟨.program ⟨257⟩, ⟨43777⟩⟩) exact237750RawTerms .large 237749 .exactZero (none)

def event237751 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44277⟩⟩) 0 ⟨43777⟩ 237750

def event237752 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44277⟩⟩) (.authority (.operator))

def exact237753RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨44277⟩⟩]⟩, (1)⟩]

theorem exact237753RawTermsValid :
    exact237753RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event237753 : Event := .resultExact (⟨.program ⟨257⟩, ⟨44277⟩⟩) exact237753RawTerms (.finite 8192) 237752 .exactZero (none)

def event237754 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42429⟩⟩) 0 ⟨42426⟩ 11360

def event237755 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42429⟩⟩) 1 ⟨6934⟩ 236778

def event237756 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42429⟩⟩) (.tensor (.predecessor 0 237754 .coefficient) (.predecessor 1 237755 .coefficient) true false)

def event237757 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨42429⟩⟩, .operator (⟨11360, 0⟩, ⟨236778, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨42426⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact237758RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨42426⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact237758RawTermsValid :
    exact237758RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event237758 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42429⟩⟩) exact237758RawTerms .large 237756 .exactZero (none)

def event237759 : Event := .predecessor (⟨.program ⟨257⟩, ⟨8361⟩⟩) 0 ⟨5561⟩ 236648

def event237760 : Event := .predecessor (⟨.program ⟨257⟩, ⟨8361⟩⟩) 1 ⟨7283⟩ 18082

def event237761 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨8361⟩⟩) (.product (.predecessor 0 237759 .coefficient) (.predecessor 1 237760 .coefficient) (⟨false, false, none, none, none⟩))

def event237762 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨8361⟩⟩, .operator (⟨236648, 0⟩, ⟨18082, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7283⟩⟩]⟩, (1)⟩)

def exact237763RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7283⟩⟩]⟩, (1)⟩]

theorem exact237763RawTermsValid :
    exact237763RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event237763 : Event := .resultExact (⟨.program ⟨257⟩, ⟨8361⟩⟩) exact237763RawTerms .large 237761 .exactZero (none)

def event237764 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42430⟩⟩) 0 ⟨8361⟩ 237763

def event237765 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42430⟩⟩) 1 ⟨42429⟩ 237758

def event237766 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42430⟩⟩) (.sum [.predecessor 0 237764 .coefficient, .predecessor 1 237765 .coefficient])

def exact237767RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7283⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨42426⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact237767RawTermsValid :
    exact237767RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event237767 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42430⟩⟩) exact237767RawTerms .large 237766 .exactZero (none)

def event237768 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42431⟩⟩) 0 ⟨42430⟩ 237767

def event237769 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42431⟩⟩) 1 ⟨109⟩ 18074

def event237770 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42431⟩⟩) (.sum [.predecessor 0 237768 .coefficient, .predecessor 1 237769 .coefficient])

def event237771 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42431⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨109⟩⟩]⟩) [⟨.result 18074 .coefficient, false, none⟩])

def event237772 : Event := .survivorFold (1) 237771

def exact237773RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7283⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨42426⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact237773RawTermsValid :
    exact237773RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event237773 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42431⟩⟩) exact237773RawTerms .large 237770 (.finite 26) (some (237771))

def event237774 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42432⟩⟩) 0 ⟨42431⟩ 237773

def event237775 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42432⟩⟩) 1 ⟨14451⟩ 11363

def event237776 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42432⟩⟩) (.product (.predecessor 0 237774 .coefficient) (.predecessor 1 237775 .coefficient) (⟨false, true, none, none, some 1⟩))

def event237777 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42432⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨14451⟩⟩], []⟩) [⟨.result 11363 .coefficient, true, some 1⟩])

def event237778 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42432⟩⟩) (.product (.result 237773 .summary) (.transfer 237777) (⟨false, false, none, none, none⟩))

def event237779 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨42432⟩⟩, .operator (⟨237773, 1⟩, ⟨11363, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨14451⟩⟩, ⟨.program ⟨257⟩, ⟨42426⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event237780 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨42432⟩⟩, .operator (⟨237773, 0⟩, ⟨11363, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨14451⟩⟩], [⟨.program ⟨257⟩, ⟨7283⟩⟩]⟩, (1)⟩)

def exact237781RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨14451⟩⟩], [⟨.program ⟨257⟩, ⟨7283⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨14451⟩⟩, ⟨.program ⟨257⟩, ⟨42426⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact237781RawTermsValid :
    exact237781RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event237781 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42432⟩⟩) exact237781RawTerms .large 237776 (.finite 44302336) (some (237778))

def event237782 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14452⟩⟩) 0 ⟨14451⟩ 11363

def event237783 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14452⟩⟩) 1 ⟨6934⟩ 236778

def event237784 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14452⟩⟩) (.tensor (.predecessor 0 237782 .coefficient) (.predecessor 1 237783 .coefficient) true false)

def event237785 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨14452⟩⟩, .operator (⟨11363, 0⟩, ⟨236778, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨14451⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact237786RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨14451⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact237786RawTermsValid :
    exact237786RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event237786 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14452⟩⟩) exact237786RawTerms .large 237784 .exactZero (none)

def event237787 : Event := .predecessor (⟨.program ⟨257⟩, ⟨8378⟩⟩) 0 ⟨5561⟩ 236648

def event237788 : Event := .predecessor (⟨.program ⟨257⟩, ⟨8378⟩⟩) 1 ⟨7300⟩ 18123

def event237789 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨8378⟩⟩) (.product (.predecessor 0 237787 .coefficient) (.predecessor 1 237788 .coefficient) (⟨false, false, none, none, none⟩))

def event237790 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨8378⟩⟩, .operator (⟨236648, 0⟩, ⟨18123, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7300⟩⟩]⟩, (1)⟩)

def exact237791RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7300⟩⟩]⟩, (1)⟩]

theorem exact237791RawTermsValid :
    exact237791RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event237791 : Event := .resultExact (⟨.program ⟨257⟩, ⟨8378⟩⟩) exact237791RawTerms .large 237789 .exactZero (none)

def event237792 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14453⟩⟩) 0 ⟨8378⟩ 237791

def event237793 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14453⟩⟩) 1 ⟨14452⟩ 237786

def event237794 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14453⟩⟩) (.sum [.predecessor 0 237792 .coefficient, .predecessor 1 237793 .coefficient])

def exact237795RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7300⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨14451⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact237795RawTermsValid :
    exact237795RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event237795 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14453⟩⟩) exact237795RawTerms .large 237794 .exactZero (none)

def event237796 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14454⟩⟩) 0 ⟨14453⟩ 237795

def event237797 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14454⟩⟩) 1 ⟨126⟩ 18115

def event237798 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14454⟩⟩) (.sum [.predecessor 0 237796 .coefficient, .predecessor 1 237797 .coefficient])

def event237799 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14454⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨126⟩⟩]⟩) [⟨.result 18115 .coefficient, false, none⟩])

def event237800 : Event := .survivorFold (1) 237799

def exact237801RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7300⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨14451⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact237801RawTermsValid :
    exact237801RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event237801 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14454⟩⟩) exact237801RawTerms .large 237798 (.finite 26) (some (237799))

def event237802 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14455⟩⟩) 0 ⟨14454⟩ 237801

def event237803 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14455⟩⟩) 1 ⟨9560⟩ 18112

def event237804 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14455⟩⟩) (.product (.predecessor 0 237802 .coefficient) (.predecessor 1 237803 .coefficient) (⟨false, false, none, none, none⟩))

def event237805 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14455⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨9559⟩⟩]⟩) [⟨.result 18108 .coefficient, false, none⟩])

def event237806 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14455⟩⟩) (.product (.result 237801 .summary) (.transfer 237805) (⟨false, false, none, none, none⟩))

def event237807 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨14455⟩⟩, .operator (⟨237801, 1⟩, ⟨18112, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨14451⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩]⟩, (-1)⟩)

def event237808 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨14455⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨14451⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨9559⟩⟩) ⟨7283⟩ 18082)

def event237809 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨14455⟩⟩, .relation 237808 0, ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨14451⟩⟩], [⟨.program ⟨257⟩, ⟨7283⟩⟩]⟩, (-1)⟩)

def event237810 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨14455⟩⟩, .operator (⟨237801, 0⟩, ⟨18112, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩]⟩, (1)⟩)

def exact237811RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨14451⟩⟩], [⟨.program ⟨257⟩, ⟨7283⟩⟩]⟩, (-1)⟩]

theorem exact237811RawTermsValid :
    exact237811RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event237811 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14455⟩⟩) exact237811RawTerms .large 237804 (.finite 279172874240) (some (237806))

def event237812 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42433⟩⟩) 0 ⟨14455⟩ 237811

def event237813 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42433⟩⟩) 1 ⟨42432⟩ 237781

def event237814 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42433⟩⟩) (.sum [.predecessor 0 237812 .coefficient, .predecessor 1 237813 .coefficient])

def event237815 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨42433⟩⟩, .operator (⟨237811, 1⟩, ⟨237781, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨14451⟩⟩], [⟨.program ⟨257⟩, ⟨7283⟩⟩]⟩, (1)⟩)

def event237816 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42433⟩⟩) (.sum [.result 237811 .summary, .result 237781 .summary])

def exact237817RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨14451⟩⟩, ⟨.program ⟨257⟩, ⟨42426⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact237817RawTermsValid :
    exact237817RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event237817 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42433⟩⟩) exact237817RawTerms .large 237814 (.finite 279217176576) (some (237816))

def event237818 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44278⟩⟩) 0 ⟨42433⟩ 237817

def event237819 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44278⟩⟩) 1 ⟨44277⟩ 237753

def event237820 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44278⟩⟩) (.product (.predecessor 0 237818 .coefficient) (.predecessor 1 237819 .coefficient) (⟨false, false, none, none, none⟩))

def event237821 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44278⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨44277⟩⟩]⟩) [⟨.result 237753 .coefficient, false, none⟩])

def event237822 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44278⟩⟩) (.product (.result 237817 .summary) (.transfer 237821) (⟨false, false, none, none, none⟩))

def event237823 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨44278⟩⟩, .operator (⟨237817, 1⟩, ⟨237753, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨14451⟩⟩, ⟨.program ⟨257⟩, ⟨42426⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨44277⟩⟩]⟩, (-1)⟩)

def eventLeaf14848 : Array AnnotatedEvent := #[
  { event := event237568
    frameStart := 237562 },
  { event := event237569
    frameStart := 237562 },
  { event := event237570
    frameStart := 237562 },
  { event := event237571
    frameStart := 237562 },
  { event := event237572
    frameStart := 237562 },
  { event := event237573
    frameStart := 237562 },
  { event := event237574
    frameStart := 237562 },
  { event := event237575
    frameStart := 237562 },
  { event := event237576
    frameStart := 237562 },
  { event := event237577
    frameStart := 237562 },
  { event := event237578
    frameStart := 237562 },
  { event := event237579
    frameStart := 237562 },
  { event := event237580
    frameStart := 237562 },
  { event := event237581
    frameStart := 237562 },
  { event := event237582
    frameStart := 237562 },
  { event := event237583
    frameStart := 237562 }
]

def eventLeaf14849 : Array AnnotatedEvent := #[
  { event := event237584
    frameStart := 237562 },
  { event := event237585
    frameStart := 237562 },
  { event := event237586
    frameStart := 237562 },
  { event := event237587
    frameStart := 237562 },
  { event := event237588
    frameStart := 237562 },
  { event := event237589
    frameStart := 237562 },
  { event := event237590
    frameStart := 237562 },
  { event := event237591
    frameStart := 237562 },
  { event := event237592
    frameStart := 237562 },
  { event := event237593
    frameStart := 237562 },
  { event := event237594
    frameStart := 237562 },
  { event := event237595
    frameStart := 237562 },
  { event := event237596
    frameStart := 237562 },
  { event := event237597
    frameStart := 237562 },
  { event := event237598
    frameStart := 237562 },
  { event := event237599
    frameStart := 237562 }
]

def eventLeaf14850 : Array AnnotatedEvent := #[
  { event := event237600
    frameStart := 237562 },
  { event := event237601
    frameStart := 237562 },
  { event := event237602
    frameStart := 237562 },
  { event := event237603
    frameStart := 237562 },
  { event := event237604
    frameStart := 237562 },
  { event := event237605
    frameStart := 237562 },
  { event := event237606
    frameStart := 237562 },
  { event := event237607
    frameStart := 237562 },
  { event := event237608
    frameStart := 237562 },
  { event := event237609
    frameStart := 237562 },
  { event := event237610
    frameStart := 237562 },
  { event := event237611
    frameStart := 237562 },
  { event := event237612
    frameStart := 237562 },
  { event := event237613
    frameStart := 237562 },
  { event := event237614
    frameStart := 237562 },
  { event := event237615
    frameStart := 237562 }
]

def eventLeaf14851 : Array AnnotatedEvent := #[
  { event := event237616
    frameStart := 237616 },
  { event := event237617
    frameStart := 237616 },
  { event := event237618
    frameStart := 237616 },
  { event := event237619
    frameStart := 237616 },
  { event := event237620
    frameStart := 237616 },
  { event := event237621
    frameStart := 237616 },
  { event := event237622
    frameStart := 237616 },
  { event := event237623
    frameStart := 237616 },
  { event := event237624
    frameStart := 237616 },
  { event := event237625
    frameStart := 237616 },
  { event := event237626
    frameStart := 237616 },
  { event := event237627
    frameStart := 237616 },
  { event := event237628
    frameStart := 237616 },
  { event := event237629
    frameStart := 237616 },
  { event := event237630
    frameStart := 237616 },
  { event := event237631
    frameStart := 237616 }
]

def eventLeaf14852 : Array AnnotatedEvent := #[
  { event := event237632
    frameStart := 237616 },
  { event := event237633
    frameStart := 237616 },
  { event := event237634
    frameStart := 237616 },
  { event := event237635
    frameStart := 237616 },
  { event := event237636
    frameStart := 237616 },
  { event := event237637
    frameStart := 237616 },
  { event := event237638
    frameStart := 237616 },
  { event := event237639
    frameStart := 237616 },
  { event := event237640
    frameStart := 237616 },
  { event := event237641
    frameStart := 237616 },
  { event := event237642
    frameStart := 237616 },
  { event := event237643
    frameStart := 237616 },
  { event := event237644
    frameStart := 237616 },
  { event := event237645
    frameStart := 237616 },
  { event := event237646
    frameStart := 237616 },
  { event := event237647
    frameStart := 237616 }
]

def eventLeaf14853 : Array AnnotatedEvent := #[
  { event := event237648
    frameStart := 237616 },
  { event := event237649
    frameStart := 237616 },
  { event := event237650
    frameStart := 237616 },
  { event := event237651
    frameStart := 237616 },
  { event := event237652
    frameStart := 237616 },
  { event := event237653
    frameStart := 237616 },
  { event := event237654
    frameStart := 237616 },
  { event := event237655
    frameStart := 237616 },
  { event := event237656
    frameStart := 237616 },
  { event := event237657
    frameStart := 237616 },
  { event := event237658
    frameStart := 237616 },
  { event := event237659
    frameStart := 237616 },
  { event := event237660
    frameStart := 237616 },
  { event := event237661
    frameStart := 237616 },
  { event := event237662
    frameStart := 237616 },
  { event := event237663
    frameStart := 237616 }
]

def eventLeaf14854 : Array AnnotatedEvent := #[
  { event := event237664
    frameStart := 237616 },
  { event := event237665
    frameStart := 237616 },
  { event := event237666
    frameStart := 237616 },
  { event := event237667
    frameStart := 237616 },
  { event := event237668
    frameStart := 237616 },
  { event := event237669
    frameStart := 237616 },
  { event := event237670
    frameStart := 237616 },
  { event := event237671
    frameStart := 237616 },
  { event := event237672
    frameStart := 237616 },
  { event := event237673
    frameStart := 237616 },
  { event := event237674
    frameStart := 237616 },
  { event := event237675
    frameStart := 237616 },
  { event := event237676
    frameStart := 237616 },
  { event := event237677
    frameStart := 237616 },
  { event := event237678
    frameStart := 237616 },
  { event := event237679
    frameStart := 237616 }
]

def eventLeaf14855 : Array AnnotatedEvent := #[
  { event := event237680
    frameStart := 237616 },
  { event := event237681
    frameStart := 237616 },
  { event := event237682
    frameStart := 237616 },
  { event := event237683
    frameStart := 237616 },
  { event := event237684
    frameStart := 237616 },
  { event := event237685
    frameStart := 237616 },
  { event := event237686
    frameStart := 237616 },
  { event := event237687
    frameStart := 237616 },
  { event := event237688
    frameStart := 237616 },
  { event := event237689
    frameStart := 237616 },
  { event := event237690
    frameStart := 237616 },
  { event := event237691
    frameStart := 237616 },
  { event := event237692
    frameStart := 237616 },
  { event := event237693
    frameStart := 237616 },
  { event := event237694
    frameStart := 237616 },
  { event := event237695
    frameStart := 237616 }
]

def eventLeaf14856 : Array AnnotatedEvent := #[
  { event := event237696
    frameStart := 237616 },
  { event := event237697
    frameStart := 237616 },
  { event := event237698
    frameStart := 237616 },
  { event := event237699
    frameStart := 237616 },
  { event := event237700
    frameStart := 237616 },
  { event := event237701
    frameStart := 237616 },
  { event := event237702
    frameStart := 237616 },
  { event := event237703
    frameStart := 237616 },
  { event := event237704
    frameStart := 237616 },
  { event := event237705
    frameStart := 237616 },
  { event := event237706
    frameStart := 237616 },
  { event := event237707
    frameStart := 237616 },
  { event := event237708
    frameStart := 237616 },
  { event := event237709
    frameStart := 237616 },
  { event := event237710
    frameStart := 237616 },
  { event := event237711
    frameStart := 237616 }
]

def eventLeaf14857 : Array AnnotatedEvent := #[
  { event := event237712
    frameStart := 237616 },
  { event := event237713
    frameStart := 237616 },
  { event := event237714
    frameStart := 237616 },
  { event := event237715
    frameStart := 237616 },
  { event := event237716
    frameStart := 237616 },
  { event := event237717
    frameStart := 237616 },
  { event := event237718
    frameStart := 237616 },
  { event := event237719
    frameStart := 237616 },
  { event := event237720
    frameStart := 0 },
  { event := event237721
    frameStart := 0 },
  { event := event237722
    frameStart := 0 },
  { event := event237723
    frameStart := 0 },
  { event := event237724
    frameStart := 0 },
  { event := event237725
    frameStart := 0 },
  { event := event237726
    frameStart := 0 },
  { event := event237727
    frameStart := 0 }
]

def eventLeaf14858 : Array AnnotatedEvent := #[
  { event := event237728
    frameStart := 0 },
  { event := event237729
    frameStart := 0 },
  { event := event237730
    frameStart := 0 },
  { event := event237731
    frameStart := 0 },
  { event := event237732
    frameStart := 0 },
  { event := event237733
    frameStart := 0 },
  { event := event237734
    frameStart := 0 },
  { event := event237735
    frameStart := 0 },
  { event := event237736
    frameStart := 0 },
  { event := event237737
    frameStart := 0 },
  { event := event237738
    frameStart := 0 },
  { event := event237739
    frameStart := 0 },
  { event := event237740
    frameStart := 0 },
  { event := event237741
    frameStart := 0 },
  { event := event237742
    frameStart := 0 },
  { event := event237743
    frameStart := 0 }
]

def eventLeaf14859 : Array AnnotatedEvent := #[
  { event := event237744
    frameStart := 0 },
  { event := event237745
    frameStart := 0 },
  { event := event237746
    frameStart := 0 },
  { event := event237747
    frameStart := 0 },
  { event := event237748
    frameStart := 0 },
  { event := event237749
    frameStart := 0 },
  { event := event237750
    frameStart := 0 },
  { event := event237751
    frameStart := 0 },
  { event := event237752
    frameStart := 0 },
  { event := event237753
    frameStart := 0 },
  { event := event237754
    frameStart := 0 },
  { event := event237755
    frameStart := 0 },
  { event := event237756
    frameStart := 0 },
  { event := event237757
    frameStart := 0 },
  { event := event237758
    frameStart := 0 },
  { event := event237759
    frameStart := 0 }
]

def eventLeaf14860 : Array AnnotatedEvent := #[
  { event := event237760
    frameStart := 0 },
  { event := event237761
    frameStart := 0 },
  { event := event237762
    frameStart := 0 },
  { event := event237763
    frameStart := 0 },
  { event := event237764
    frameStart := 0 },
  { event := event237765
    frameStart := 0 },
  { event := event237766
    frameStart := 0 },
  { event := event237767
    frameStart := 0 },
  { event := event237768
    frameStart := 0 },
  { event := event237769
    frameStart := 0 },
  { event := event237770
    frameStart := 0 },
  { event := event237771
    frameStart := 0 },
  { event := event237772
    frameStart := 0 },
  { event := event237773
    frameStart := 0 },
  { event := event237774
    frameStart := 0 },
  { event := event237775
    frameStart := 0 }
]

def eventLeaf14861 : Array AnnotatedEvent := #[
  { event := event237776
    frameStart := 0 },
  { event := event237777
    frameStart := 0 },
  { event := event237778
    frameStart := 0 },
  { event := event237779
    frameStart := 0 },
  { event := event237780
    frameStart := 0 },
  { event := event237781
    frameStart := 0 },
  { event := event237782
    frameStart := 0 },
  { event := event237783
    frameStart := 0 },
  { event := event237784
    frameStart := 0 },
  { event := event237785
    frameStart := 0 },
  { event := event237786
    frameStart := 0 },
  { event := event237787
    frameStart := 0 },
  { event := event237788
    frameStart := 0 },
  { event := event237789
    frameStart := 0 },
  { event := event237790
    frameStart := 0 },
  { event := event237791
    frameStart := 0 }
]

def eventLeaf14862 : Array AnnotatedEvent := #[
  { event := event237792
    frameStart := 0 },
  { event := event237793
    frameStart := 0 },
  { event := event237794
    frameStart := 0 },
  { event := event237795
    frameStart := 0 },
  { event := event237796
    frameStart := 0 },
  { event := event237797
    frameStart := 0 },
  { event := event237798
    frameStart := 0 },
  { event := event237799
    frameStart := 0 },
  { event := event237800
    frameStart := 0 },
  { event := event237801
    frameStart := 0 },
  { event := event237802
    frameStart := 0 },
  { event := event237803
    frameStart := 0 },
  { event := event237804
    frameStart := 0 },
  { event := event237805
    frameStart := 0 },
  { event := event237806
    frameStart := 0 },
  { event := event237807
    frameStart := 0 }
]

def eventLeaf14863 : Array AnnotatedEvent := #[
  { event := event237808
    frameStart := 0 },
  { event := event237809
    frameStart := 0 },
  { event := event237810
    frameStart := 0 },
  { event := event237811
    frameStart := 0 },
  { event := event237812
    frameStart := 0 },
  { event := event237813
    frameStart := 0 },
  { event := event237814
    frameStart := 0 },
  { event := event237815
    frameStart := 0 },
  { event := event237816
    frameStart := 0 },
  { event := event237817
    frameStart := 0 },
  { event := event237818
    frameStart := 0 },
  { event := event237819
    frameStart := 0 },
  { event := event237820
    frameStart := 0 },
  { event := event237821
    frameStart := 0 },
  { event := event237822
    frameStart := 0 },
  { event := event237823
    frameStart := 0 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events928
