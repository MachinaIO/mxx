import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events026

open Mxx.Certificate.OperationalNoise
open CertificateABI

def exact6656RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨59972⟩⟩], []⟩, (1)⟩]

theorem exact6656RawTermsValid :
    exact6656RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event6656 : Event := .resultExact (⟨.program ⟨257⟩, ⟨59973⟩⟩) exact6656RawTerms (.finite 222230617312560576599880) 6654 .exactZero (none)

def event6657 : Event := .predecessor (⟨.program ⟨257⟩, ⟨56992⟩⟩) 0 ⟨56793⟩ 6348

def event6658 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨56992⟩⟩) (.authority (.programFamilyFact))

def exact6659RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨56992⟩⟩], []⟩, (1)⟩]

theorem exact6659RawTermsValid :
    exact6659RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event6659 : Event := .resultExact (⟨.program ⟨257⟩, ⟨56992⟩⟩) exact6659RawTerms (.finite 16) 6658 .exactZero (none)

def event6660 : Event := .predecessor (⟨.program ⟨257⟩, ⟨56993⟩⟩) 0 ⟨56992⟩ 6659

def event6661 : Event := .predecessor (⟨.program ⟨257⟩, ⟨56993⟩⟩) 1 ⟨6741⟩ 653

def event6662 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨56993⟩⟩) (.product (.predecessor 0 6660 .coefficient) (.predecessor 1 6661 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event6663 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨56993⟩⟩, .operator (⟨6659, 0⟩, ⟨653, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨56992⟩⟩], []⟩, (1)⟩)

def exact6664RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨56992⟩⟩], []⟩, (1)⟩]

theorem exact6664RawTermsValid :
    exact6664RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event6664 : Event := .resultExact (⟨.program ⟨257⟩, ⟨56993⟩⟩) exact6664RawTerms (.finite 220778129617707239497920) 6662 .exactZero (none)

def event6665 : Event := .predecessor (⟨.program ⟨257⟩, ⟨54012⟩⟩) 0 ⟨53813⟩ 6371

def event6666 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨54012⟩⟩) (.authority (.programFamilyFact))

def exact6667RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨54012⟩⟩], []⟩, (1)⟩]

theorem exact6667RawTermsValid :
    exact6667RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event6667 : Event := .resultExact (⟨.program ⟨257⟩, ⟨54012⟩⟩) exact6667RawTerms (.finite 12) 6666 .exactZero (none)

def event6668 : Event := .predecessor (⟨.program ⟨257⟩, ⟨54013⟩⟩) 0 ⟨54012⟩ 6667

def event6669 : Event := .predecessor (⟨.program ⟨257⟩, ⟨54013⟩⟩) 1 ⟨6757⟩ 663

def event6670 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨54013⟩⟩) (.product (.predecessor 0 6668 .coefficient) (.predecessor 1 6669 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event6671 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨54013⟩⟩, .operator (⟨6667, 0⟩, ⟨663, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54012⟩⟩], []⟩, (1)⟩)

def exact6672RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54012⟩⟩], []⟩, (1)⟩]

theorem exact6672RawTermsValid :
    exact6672RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event6672 : Event := .resultExact (⟨.program ⟨257⟩, ⟨54013⟩⟩) exact6672RawTerms (.finite 216532396355828254122960) 6670 .exactZero (none)

def event6673 : Event := .predecessor (⟨.program ⟨257⟩, ⟨51032⟩⟩) 0 ⟨50833⟩ 6394

def event6674 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨51032⟩⟩) (.authority (.programFamilyFact))

def exact6675RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨51032⟩⟩], []⟩, (1)⟩]

theorem exact6675RawTermsValid :
    exact6675RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event6675 : Event := .resultExact (⟨.program ⟨257⟩, ⟨51032⟩⟩) exact6675RawTerms (.finite 10) 6674 .exactZero (none)

def event6676 : Event := .predecessor (⟨.program ⟨257⟩, ⟨51033⟩⟩) 0 ⟨51032⟩ 6675

def event6677 : Event := .predecessor (⟨.program ⟨257⟩, ⟨51033⟩⟩) 1 ⟨6768⟩ 673

def event6678 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨51033⟩⟩) (.product (.predecessor 0 6676 .coefficient) (.predecessor 1 6677 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event6679 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨51033⟩⟩, .operator (⟨6675, 0⟩, ⟨673, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51032⟩⟩], []⟩, (1)⟩)

def exact6680RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51032⟩⟩], []⟩, (1)⟩]

theorem exact6680RawTermsValid :
    exact6680RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event6680 : Event := .resultExact (⟨.program ⟨257⟩, ⟨51033⟩⟩) exact6680RawTerms (.finite 213251602471649038151400) 6678 .exactZero (none)

def event6681 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31968⟩⟩) 0 ⟨31773⟩ 6417

def event6682 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31968⟩⟩) (.authority (.programFamilyFact))

def exact6683RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨31968⟩⟩], []⟩, (1)⟩]

theorem exact6683RawTermsValid :
    exact6683RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event6683 : Event := .resultExact (⟨.program ⟨257⟩, ⟨31968⟩⟩) exact6683RawTerms (.finite 6) 6682 .exactZero (none)

def event6684 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31969⟩⟩) 0 ⟨31968⟩ 6683

def event6685 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31969⟩⟩) 1 ⟨6794⟩ 683

def event6686 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31969⟩⟩) (.product (.predecessor 0 6684 .coefficient) (.predecessor 1 6685 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event6687 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨31969⟩⟩, .operator (⟨6683, 0⟩, ⟨683, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨31968⟩⟩], []⟩, (1)⟩)

def exact6688RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨31968⟩⟩], []⟩, (1)⟩]

theorem exact6688RawTermsValid :
    exact6688RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event6688 : Event := .resultExact (⟨.program ⟨257⟩, ⟨31969⟩⟩) exact6688RawTerms (.finite 201065796616126235971320) 6686 .exactZero (none)

def event6689 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21948⟩⟩) 0 ⟨21753⟩ 6440

def event6690 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨21948⟩⟩) (.authority (.programFamilyFact))

def exact6691RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨21948⟩⟩], []⟩, (1)⟩]

theorem exact6691RawTermsValid :
    exact6691RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event6691 : Event := .resultExact (⟨.program ⟨257⟩, ⟨21948⟩⟩) exact6691RawTerms (.finite 4) 6690 .exactZero (none)

def event6692 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21949⟩⟩) 0 ⟨21948⟩ 6691

def event6693 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21949⟩⟩) 1 ⟨6822⟩ 693

def event6694 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨21949⟩⟩) (.product (.predecessor 0 6692 .coefficient) (.predecessor 1 6693 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event6695 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨21949⟩⟩, .operator (⟨6691, 0⟩, ⟨693, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨21948⟩⟩], []⟩, (1)⟩)

def exact6696RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨21948⟩⟩], []⟩, (1)⟩]

theorem exact6696RawTermsValid :
    exact6696RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event6696 : Event := .resultExact (⟨.program ⟨257⟩, ⟨21949⟩⟩) exact6696RawTerms (.finite 187661410175051153573232) 6694 .exactZero (none)

def event6697 : Event := .predecessor (⟨.program ⟨257⟩, ⟨18728⟩⟩) 0 ⟨18533⟩ 6463

def event6698 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨18728⟩⟩) (.authority (.programFamilyFact))

def exact6699RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨18728⟩⟩], []⟩, (1)⟩]

theorem exact6699RawTermsValid :
    exact6699RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event6699 : Event := .resultExact (⟨.program ⟨257⟩, ⟨18728⟩⟩) exact6699RawTerms (.finite 3) 6698 .exactZero (none)

def event6700 : Event := .predecessor (⟨.program ⟨257⟩, ⟨18729⟩⟩) 0 ⟨18728⟩ 6699

def event6701 : Event := .predecessor (⟨.program ⟨257⟩, ⟨18729⟩⟩) 1 ⟨6846⟩ 703

def event6702 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨18729⟩⟩) (.product (.predecessor 0 6700 .coefficient) (.predecessor 1 6701 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event6703 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨18729⟩⟩, .operator (⟨6699, 0⟩, ⟨703, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18728⟩⟩], []⟩, (1)⟩)

def exact6704RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18728⟩⟩], []⟩, (1)⟩]

theorem exact6704RawTermsValid :
    exact6704RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event6704 : Event := .resultExact (⟨.program ⟨257⟩, ⟨18729⟩⟩) exact6704RawTerms (.finite 175932572039110456474905) 6702 .exactZero (none)

def event6705 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15918⟩⟩) 0 ⟨15733⟩ 6486

def event6706 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15918⟩⟩) (.authority (.programFamilyFact))

def exact6707RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15918⟩⟩], []⟩, (1)⟩]

theorem exact6707RawTermsValid :
    exact6707RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event6707 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15918⟩⟩) exact6707RawTerms (.finite 2) 6706 .exactZero (none)

def event6708 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15919⟩⟩) 0 ⟨15918⟩ 6707

def event6709 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15919⟩⟩) 1 ⟨6863⟩ 713

def event6710 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15919⟩⟩) (.product (.predecessor 0 6708 .coefficient) (.predecessor 1 6709 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event6711 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨15919⟩⟩, .operator (⟨6707, 0⟩, ⟨713, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15918⟩⟩], []⟩, (1)⟩)

def exact6712RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15918⟩⟩], []⟩, (1)⟩]

theorem exact6712RawTermsValid :
    exact6712RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event6712 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15919⟩⟩) exact6712RawTerms (.finite 156384508479209294644360) 6710 .exactZero (none)

def event6713 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15920⟩⟩) 0 ⟨6728⟩ 728

def event6714 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15920⟩⟩) 1 ⟨15919⟩ 6712

def event6715 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15920⟩⟩) (.sum [.predecessor 0 6713 .coefficient, .predecessor 1 6714 .coefficient])

def exact6716RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15918⟩⟩], []⟩, (1)⟩]

theorem exact6716RawTermsValid :
    exact6716RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event6716 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15920⟩⟩) exact6716RawTerms (.finite 156384508479209294644360) 6715 .exactZero (none)

def event6717 : Event := .predecessor (⟨.program ⟨257⟩, ⟨18730⟩⟩) 0 ⟨15920⟩ 6716

def event6718 : Event := .predecessor (⟨.program ⟨257⟩, ⟨18730⟩⟩) 1 ⟨18729⟩ 6704

def event6719 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨18730⟩⟩) (.sum [.predecessor 0 6717 .coefficient, .predecessor 1 6718 .coefficient])

def exact6720RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18728⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15918⟩⟩], []⟩, (1)⟩]

theorem exact6720RawTermsValid :
    exact6720RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event6720 : Event := .resultExact (⟨.program ⟨257⟩, ⟨18730⟩⟩) exact6720RawTerms (.finite 332317080518319751119265) 6719 .exactZero (none)

def event6721 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21950⟩⟩) 0 ⟨18730⟩ 6720

def event6722 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21950⟩⟩) 1 ⟨21949⟩ 6696

def event6723 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨21950⟩⟩) (.sum [.predecessor 0 6721 .coefficient, .predecessor 1 6722 .coefficient])

def exact6724RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨21948⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18728⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15918⟩⟩], []⟩, (1)⟩]

theorem exact6724RawTermsValid :
    exact6724RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event6724 : Event := .resultExact (⟨.program ⟨257⟩, ⟨21950⟩⟩) exact6724RawTerms (.finite 519978490693370904692497) 6723 .exactZero (none)

def event6725 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31970⟩⟩) 0 ⟨21950⟩ 6724

def event6726 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31970⟩⟩) 1 ⟨31969⟩ 6688

def event6727 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31970⟩⟩) (.sum [.predecessor 0 6725 .coefficient, .predecessor 1 6726 .coefficient])

def exact6728RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨31968⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨21948⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18728⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15918⟩⟩], []⟩, (1)⟩]

theorem exact6728RawTermsValid :
    exact6728RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event6728 : Event := .resultExact (⟨.program ⟨257⟩, ⟨31970⟩⟩) exact6728RawTerms (.finite 721044287309497140663817) 6727 .exactZero (none)

def event6729 : Event := .predecessor (⟨.program ⟨257⟩, ⟨51034⟩⟩) 0 ⟨31970⟩ 6728

def event6730 : Event := .predecessor (⟨.program ⟨257⟩, ⟨51034⟩⟩) 1 ⟨51033⟩ 6680

def event6731 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨51034⟩⟩) (.sum [.predecessor 0 6729 .coefficient, .predecessor 1 6730 .coefficient])

def exact6732RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51032⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨31968⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨21948⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18728⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15918⟩⟩], []⟩, (1)⟩]

theorem exact6732RawTermsValid :
    exact6732RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event6732 : Event := .resultExact (⟨.program ⟨257⟩, ⟨51034⟩⟩) exact6732RawTerms (.finite 934295889781146178815217) 6731 .exactZero (none)

def event6733 : Event := .predecessor (⟨.program ⟨257⟩, ⟨54014⟩⟩) 0 ⟨51034⟩ 6732

def event6734 : Event := .predecessor (⟨.program ⟨257⟩, ⟨54014⟩⟩) 1 ⟨54013⟩ 6672

def event6735 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨54014⟩⟩) (.sum [.predecessor 0 6733 .coefficient, .predecessor 1 6734 .coefficient])

def exact6736RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54012⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51032⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨31968⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨21948⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18728⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15918⟩⟩], []⟩, (1)⟩]

theorem exact6736RawTermsValid :
    exact6736RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event6736 : Event := .resultExact (⟨.program ⟨257⟩, ⟨54014⟩⟩) exact6736RawTerms (.finite 1150828286136974432938177) 6735 .exactZero (none)

def event6737 : Event := .predecessor (⟨.program ⟨257⟩, ⟨56994⟩⟩) 0 ⟨54014⟩ 6736

def event6738 : Event := .predecessor (⟨.program ⟨257⟩, ⟨56994⟩⟩) 1 ⟨56993⟩ 6664

def event6739 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨56994⟩⟩) (.sum [.predecessor 0 6737 .coefficient, .predecessor 1 6738 .coefficient])

def exact6740RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨56992⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54012⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51032⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨31968⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨21948⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18728⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15918⟩⟩], []⟩, (1)⟩]

theorem exact6740RawTermsValid :
    exact6740RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event6740 : Event := .resultExact (⟨.program ⟨257⟩, ⟨56994⟩⟩) exact6740RawTerms (.finite 1371606415754681672436097) 6739 .exactZero (none)

def event6741 : Event := .predecessor (⟨.program ⟨257⟩, ⟨59974⟩⟩) 0 ⟨56994⟩ 6740

def event6742 : Event := .predecessor (⟨.program ⟨257⟩, ⟨59974⟩⟩) 1 ⟨59973⟩ 6656

def event6743 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨59974⟩⟩) (.sum [.predecessor 0 6741 .coefficient, .predecessor 1 6742 .coefficient])

def exact6744RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨59972⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨56992⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54012⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51032⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨31968⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨21948⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18728⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15918⟩⟩], []⟩, (1)⟩]

theorem exact6744RawTermsValid :
    exact6744RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event6744 : Event := .resultExact (⟨.program ⟨257⟩, ⟨59974⟩⟩) exact6744RawTerms (.finite 1593837033067242249035977) 6743 .exactZero (none)

def event6745 : Event := .predecessor (⟨.program ⟨257⟩, ⟨62954⟩⟩) 0 ⟨59974⟩ 6744

def event6746 : Event := .predecessor (⟨.program ⟨257⟩, ⟨62954⟩⟩) 1 ⟨62953⟩ 6648

def event6747 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨62954⟩⟩) (.sum [.predecessor 0 6745 .coefficient, .predecessor 1 6746 .coefficient])

def exact6748RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨62952⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨59972⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨56992⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54012⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51032⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨31968⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨21948⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18728⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15918⟩⟩], []⟩, (1)⟩]

theorem exact6748RawTermsValid :
    exact6748RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event6748 : Event := .resultExact (⟨.program ⟨257⟩, ⟨62954⟩⟩) exact6748RawTerms (.finite 1818214806102629497873537) 6747 .exactZero (none)

def event6749 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66100⟩⟩) 0 ⟨62954⟩ 6748

def event6750 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66100⟩⟩) 1 ⟨66099⟩ 6640

def event6751 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨66100⟩⟩) (.sum [.predecessor 0 6749 .coefficient, .predecessor 1 6750 .coefficient])

def exact6752RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨62952⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨59972⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨56992⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54012⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51032⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨31968⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨21948⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18728⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15918⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66098⟩⟩], []⟩, (1)⟩]

theorem exact6752RawTermsValid :
    exact6752RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event6752 : Event := .resultExact (⟨.program ⟨257⟩, ⟨66100⟩⟩) exact6752RawTerms (.finite 2044702714934587786668817) 6751 .exactZero (none)

def event6753 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66101⟩⟩) 0 ⟨66100⟩ 6752

def event6754 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66101⟩⟩) 1 ⟨26532⟩ 6632

def event6755 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨66101⟩⟩) (.sum [.predecessor 0 6753 .coefficient, .predecessor 1 6754 .coefficient])

def exact6756RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨62952⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨59972⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨56992⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54012⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51032⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨31968⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨21948⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18728⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26531⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15918⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66098⟩⟩], []⟩, (1)⟩]

theorem exact6756RawTermsValid :
    exact6756RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event6756 : Event := .resultExact (⟨.program ⟨257⟩, ⟨66101⟩⟩) exact6756RawTerms (.finite 2271712485307633536959017) 6755 .exactZero (none)

def event6757 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66102⟩⟩) 0 ⟨66101⟩ 6756

def event6758 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66102⟩⟩) 1 ⟨29212⟩ 6624

def event6759 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨66102⟩⟩) (.sum [.predecessor 0 6757 .coefficient, .predecessor 1 6758 .coefficient])

def exact6760RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨62952⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨59972⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨56992⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54012⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51032⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨31968⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨21948⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18728⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29211⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26531⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15918⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66098⟩⟩], []⟩, (1)⟩]

theorem exact6760RawTermsValid :
    exact6760RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event6760 : Event := .resultExact (⟨.program ⟨257⟩, ⟨66102⟩⟩) exact6760RawTerms (.finite 2499949335520533588602137) 6759 .exactZero (none)

def event6761 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66103⟩⟩) 0 ⟨66102⟩ 6760

def event6762 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66103⟩⟩) 1 ⟨34869⟩ 6616

def event6763 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨66103⟩⟩) (.sum [.predecessor 0 6761 .coefficient, .predecessor 1 6762 .coefficient])

def exact6764RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨62952⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨59972⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨56992⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54012⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51032⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨31968⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨21948⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34868⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18728⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29211⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26531⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15918⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66098⟩⟩], []⟩, (1)⟩]

theorem exact6764RawTermsValid :
    exact6764RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event6764 : Event := .resultExact (⟨.program ⟨257⟩, ⟨66103⟩⟩) exact6764RawTerms (.finite 2728804713782791092959737) 6763 .exactZero (none)

def event6765 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66104⟩⟩) 0 ⟨66103⟩ 6764

def event6766 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66104⟩⟩) 1 ⟨37549⟩ 6608

def event6767 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨66104⟩⟩) (.sum [.predecessor 0 6765 .coefficient, .predecessor 1 6766 .coefficient])

def exact6768RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨62952⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨59972⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨56992⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54012⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51032⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨31968⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨21948⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37548⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34868⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18728⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29211⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26531⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15918⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66098⟩⟩], []⟩, (1)⟩]

theorem exact6768RawTermsValid :
    exact6768RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event6768 : Event := .resultExact (⟨.program ⟨257⟩, ⟨66104⟩⟩) exact6768RawTerms (.finite 2957926202950004710694497) 6767 .exactZero (none)

def event6769 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66105⟩⟩) 0 ⟨66104⟩ 6768

def event6770 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66105⟩⟩) 1 ⟨40232⟩ 6600

def event6771 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨66105⟩⟩) (.sum [.predecessor 0 6769 .coefficient, .predecessor 1 6770 .coefficient])

def exact6772RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨62952⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨59972⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨56992⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54012⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51032⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨31968⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨21948⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40231⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37548⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34868⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18728⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29211⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26531⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15918⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66098⟩⟩], []⟩, (1)⟩]

theorem exact6772RawTermsValid :
    exact6772RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event6772 : Event := .resultExact (⟨.program ⟨257⟩, ⟨66105⟩⟩) exact6772RawTerms (.finite 3187511970717354526236217) 6771 .exactZero (none)

def event6773 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66106⟩⟩) 0 ⟨66105⟩ 6772

def event6774 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66106⟩⟩) 1 ⟨42912⟩ 6592

def event6775 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨66106⟩⟩) (.sum [.predecessor 0 6773 .coefficient, .predecessor 1 6774 .coefficient])

def exact6776RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨62952⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨59972⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨56992⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54012⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51032⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨31968⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨42911⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨21948⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40231⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37548⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34868⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18728⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29211⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26531⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15918⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66098⟩⟩], []⟩, (1)⟩]

theorem exact6776RawTermsValid :
    exact6776RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event6776 : Event := .resultExact (⟨.program ⟨257⟩, ⟨66106⟩⟩) exact6776RawTerms (.finite 3417662756781096507033577) 6775 .exactZero (none)

def event6777 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66107⟩⟩) 0 ⟨66106⟩ 6776

def event6778 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66107⟩⟩) 1 ⟨45589⟩ 6584

def event6779 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨66107⟩⟩) (.sum [.predecessor 0 6777 .coefficient, .predecessor 1 6778 .coefficient])

def exact6780RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨62952⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨59972⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨56992⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54012⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51032⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨31968⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45588⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨42911⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨21948⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40231⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37548⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34868⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18728⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29211⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26531⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15918⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66098⟩⟩], []⟩, (1)⟩]

theorem exact6780RawTermsValid :
    exact6780RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event6780 : Event := .resultExact (⟨.program ⟨257⟩, ⟨66107⟩⟩) exact6780RawTerms (.finite 3648263642165693263543057) 6779 .exactZero (none)

def event6781 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66108⟩⟩) 0 ⟨66107⟩ 6780

def event6782 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66108⟩⟩) 1 ⟨48269⟩ 6576

def event6783 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨66108⟩⟩) (.sum [.predecessor 0 6781 .coefficient, .predecessor 1 6782 .coefficient])

def exact6784RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨62952⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨59972⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨56992⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54012⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51032⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨31968⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48268⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45588⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨42911⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨21948⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40231⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37548⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34868⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18728⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29211⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26531⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15918⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66098⟩⟩], []⟩, (1)⟩]

theorem exact6784RawTermsValid :
    exact6784RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event6784 : Event := .resultExact (⟨.program ⟨257⟩, ⟨66108⟩⟩) exact6784RawTerms (.finite 3878994884184198780231457) 6783 .exactZero (none)

def event6785 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67325⟩⟩) 0 ⟨66108⟩ 6784

def event6786 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67325⟩⟩) 1 ⟨67323⟩ 6568

def event6787 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67325⟩⟩) (.sum [.predecessor 0 6785 .coefficient, .predecessor 1 6786 .coefficient])

def exact6788RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨62952⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨59972⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨56992⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54012⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51032⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67322⟩⟩], []⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨31968⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48268⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45588⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨42911⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨21948⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40231⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37548⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34868⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18728⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29211⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26531⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15918⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66098⟩⟩], []⟩, (1)⟩]

theorem exact6788RawTermsValid :
    exact6788RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event6788 : Event := .resultExact (⟨.program ⟨257⟩, ⟨67325⟩⟩) exact6788RawTerms (.finite 8101376613122849735629177) 6787 .exactZero (none)

def event6789 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67326⟩⟩) 0 ⟨67325⟩ 6788

def event6790 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67326⟩⟩) 1 ⟨6751⟩ 6065

def event6791 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67326⟩⟩) (.product (.predecessor 0 6789 .coefficient) (.predecessor 1 6790 .coefficient) (⟨false, true, none, none, some 1⟩))

def event6792 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67326⟩⟩, .operator (⟨6788, 5⟩, ⟨6065, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6751⟩⟩, ⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67322⟩⟩], []⟩, (-1)⟩)

def event6793 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67326⟩⟩, .operator (⟨6788, 7⟩, ⟨6065, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6751⟩⟩, ⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48268⟩⟩], []⟩, (1)⟩)

def event6794 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67326⟩⟩, .operator (⟨6788, 8⟩, ⟨6065, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6751⟩⟩, ⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45588⟩⟩], []⟩, (1)⟩)

def event6795 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67326⟩⟩, .operator (⟨6788, 9⟩, ⟨6065, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6751⟩⟩, ⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨42911⟩⟩], []⟩, (1)⟩)

def event6796 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67326⟩⟩, .operator (⟨6788, 11⟩, ⟨6065, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6751⟩⟩, ⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40231⟩⟩], []⟩, (1)⟩)

def event6797 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67326⟩⟩, .operator (⟨6788, 12⟩, ⟨6065, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6751⟩⟩, ⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37548⟩⟩], []⟩, (1)⟩)

def event6798 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67326⟩⟩, .operator (⟨6788, 13⟩, ⟨6065, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6751⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34868⟩⟩], []⟩, (1)⟩)

def event6799 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67326⟩⟩, .operator (⟨6788, 15⟩, ⟨6065, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6751⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29211⟩⟩], []⟩, (1)⟩)

def event6800 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67326⟩⟩, .operator (⟨6788, 16⟩, ⟨6065, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6751⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26531⟩⟩], []⟩, (1)⟩)

def event6801 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67326⟩⟩, .operator (⟨6788, 18⟩, ⟨6065, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6751⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66098⟩⟩], []⟩, (1)⟩)

def event6802 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67326⟩⟩, .operator (⟨6788, 0⟩, ⟨6065, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨6751⟩⟩, ⟨.program ⟨257⟩, ⟨62952⟩⟩], []⟩, (1)⟩)

def event6803 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67326⟩⟩, .operator (⟨6788, 1⟩, ⟨6065, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨6751⟩⟩, ⟨.program ⟨257⟩, ⟨59972⟩⟩], []⟩, (1)⟩)

def event6804 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67326⟩⟩, .operator (⟨6788, 2⟩, ⟨6065, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨6751⟩⟩, ⟨.program ⟨257⟩, ⟨56992⟩⟩], []⟩, (1)⟩)

def event6805 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67326⟩⟩, .operator (⟨6788, 3⟩, ⟨6065, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6751⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54012⟩⟩], []⟩, (1)⟩)

def event6806 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67326⟩⟩, .operator (⟨6788, 4⟩, ⟨6065, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6751⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51032⟩⟩], []⟩, (1)⟩)

def event6807 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67326⟩⟩, .operator (⟨6788, 6⟩, ⟨6065, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6751⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨31968⟩⟩], []⟩, (1)⟩)

def event6808 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67326⟩⟩, .operator (⟨6788, 10⟩, ⟨6065, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6751⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨21948⟩⟩], []⟩, (1)⟩)

def event6809 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67326⟩⟩, .operator (⟨6788, 14⟩, ⟨6065, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6751⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18728⟩⟩], []⟩, (1)⟩)

def event6810 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67326⟩⟩, .operator (⟨6788, 17⟩, ⟨6065, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6751⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15918⟩⟩], []⟩, (1)⟩)

def exact6811RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨6751⟩⟩, ⟨.program ⟨257⟩, ⟨62952⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨6751⟩⟩, ⟨.program ⟨257⟩, ⟨59972⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨6751⟩⟩, ⟨.program ⟨257⟩, ⟨56992⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6751⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54012⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6751⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51032⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6751⟩⟩, ⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67322⟩⟩], []⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6751⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨31968⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6751⟩⟩, ⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48268⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6751⟩⟩, ⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45588⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6751⟩⟩, ⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨42911⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6751⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨21948⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6751⟩⟩, ⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40231⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6751⟩⟩, ⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37548⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6751⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34868⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6751⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18728⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6751⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29211⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6751⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26531⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6751⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15918⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6751⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66098⟩⟩], []⟩, (1)⟩]

theorem exact6811RawTermsValid :
    exact6811RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event6811 : Event := .resultExact (⟨.program ⟨257⟩, ⟨67326⟩⟩) exact6811RawTerms (.finite 242439265663414481461807438261659125549024385957747627321007901649791422734527837612598556531990428564442786651523598220427935283195601994622870932882522772408482364584924167384489915188362411016192) 6791 .exactZero (none)

def event6812 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6771⟩⟩) (.authority (.factStore))

def exact6813RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6771⟩⟩], []⟩, (1)⟩]

theorem exact6813RawTermsValid :
    exact6813RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event6813 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6771⟩⟩) exact6813RawTerms (.finite 338309437981014441096759724635630572701369121440190406863515829483373066956979144836552849054411963346472131188989532524765423391376866138657337362246117665517647600019) 6812 .exactZero (none)

def event6814 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨4614⟩⟩) (.authority (.operator))

def event6815 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨4614⟩⟩) (.finite 10)

def event6816 : Event := .predecessor (⟨.program ⟨257⟩, ⟨4616⟩⟩) 0 ⟨392⟩ 14

def event6817 : Event := .predecessor (⟨.program ⟨257⟩, ⟨4616⟩⟩) 1 ⟨4614⟩ 6815

def event6818 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨4616⟩⟩) (.sum [.predecessor 0 6816 .coefficient, .predecessor 1 6817 .coefficient])

def event6819 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨4616⟩⟩) (.finite 655350)

def event6820 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5541⟩⟩) 0 ⟨4616⟩ 6819

def event6821 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5541⟩⟩) 1 ⟨5426⟩ 38

def event6822 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5541⟩⟩) (.identity (.predecessor 1 6821 .coefficient))

def event6823 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5541⟩⟩) (.finite 655360)

def event6824 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47762⟩⟩) 0 ⟨5541⟩ 6823

def event6825 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47762⟩⟩) (.authority (.programFamilyFact))

def exact6826RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨47762⟩⟩], []⟩, (1)⟩]

theorem exact6826RawTermsValid :
    exact6826RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event6826 : Event := .resultExact (⟨.program ⟨257⟩, ⟨47762⟩⟩) exact6826RawTerms (.finite 60) 6825 .exactZero (none)

def event6827 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15036⟩⟩) 0 ⟨5541⟩ 6823

def event6828 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15036⟩⟩) (.authority (.programFamilyFact))

def exact6829RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15036⟩⟩], []⟩, (1)⟩]

theorem exact6829RawTermsValid :
    exact6829RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event6829 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15036⟩⟩) exact6829RawTerms (.finite 60) 6828 .exactZero (none)

def event6830 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47763⟩⟩) 0 ⟨15036⟩ 6829

def event6831 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47763⟩⟩) 1 ⟨47762⟩ 6826

def event6832 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47763⟩⟩) (.product (.predecessor 0 6830 .coefficient) (.predecessor 1 6831 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event6833 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨47763⟩⟩, .operator (⟨6829, 0⟩, ⟨6826, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨15036⟩⟩, ⟨.program ⟨257⟩, ⟨47762⟩⟩], []⟩, (1)⟩)

def exact6834RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15036⟩⟩, ⟨.program ⟨257⟩, ⟨47762⟩⟩], []⟩, (1)⟩]

theorem exact6834RawTermsValid :
    exact6834RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event6834 : Event := .resultExact (⟨.program ⟨257⟩, ⟨47763⟩⟩) exact6834RawTerms (.finite 3600) 6832 .exactZero (none)

def event6835 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47764⟩⟩) 0 ⟨47763⟩ 6834

def event6836 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47764⟩⟩) (.identity (.predecessor 0 6835 .coefficient))

def event6837 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨47764⟩⟩) (.finite 3600)

def event6838 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48124⟩⟩) 0 ⟨47764⟩ 6837

def event6839 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48124⟩⟩) (.authority (.programFamilyFact))

def exact6840RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨48124⟩⟩], []⟩, (1)⟩]

theorem exact6840RawTermsValid :
    exact6840RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event6840 : Event := .resultExact (⟨.program ⟨257⟩, ⟨48124⟩⟩) exact6840RawTerms (.finite 60) 6839 .exactZero (none)

def event6841 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48125⟩⟩) 0 ⟨48124⟩ 6840

def event6842 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48125⟩⟩) (.identity (.predecessor 0 6841 .coefficient))

def event6843 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨48125⟩⟩) (.finite 60)

def event6844 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48324⟩⟩) 0 ⟨48125⟩ 6843

def event6845 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48324⟩⟩) (.authority (.programFamilyFact))

def exact6846RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨48324⟩⟩], []⟩, (1)⟩]

theorem exact6846RawTermsValid :
    exact6846RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event6846 : Event := .resultExact (⟨.program ⟨257⟩, ⟨48324⟩⟩) exact6846RawTerms (.finite 63) 6845 .exactZero (none)

def event6847 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45082⟩⟩) 0 ⟨5541⟩ 6823

def event6848 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45082⟩⟩) (.authority (.programFamilyFact))

def exact6849RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨45082⟩⟩], []⟩, (1)⟩]

theorem exact6849RawTermsValid :
    exact6849RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event6849 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45082⟩⟩) exact6849RawTerms (.finite 58) 6848 .exactZero (none)

def event6850 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14736⟩⟩) 0 ⟨5541⟩ 6823

def event6851 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14736⟩⟩) (.authority (.programFamilyFact))

def exact6852RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14736⟩⟩], []⟩, (1)⟩]

theorem exact6852RawTermsValid :
    exact6852RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event6852 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14736⟩⟩) exact6852RawTerms (.finite 58) 6851 .exactZero (none)

def event6853 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45083⟩⟩) 0 ⟨14736⟩ 6852

def event6854 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45083⟩⟩) 1 ⟨45082⟩ 6849

def event6855 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45083⟩⟩) (.product (.predecessor 0 6853 .coefficient) (.predecessor 1 6854 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event6856 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨45083⟩⟩, .operator (⟨6852, 0⟩, ⟨6849, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨14736⟩⟩, ⟨.program ⟨257⟩, ⟨45082⟩⟩], []⟩, (1)⟩)

def exact6857RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14736⟩⟩, ⟨.program ⟨257⟩, ⟨45082⟩⟩], []⟩, (1)⟩]

theorem exact6857RawTermsValid :
    exact6857RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event6857 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45083⟩⟩) exact6857RawTerms (.finite 3364) 6855 .exactZero (none)

def event6858 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45084⟩⟩) 0 ⟨45083⟩ 6857

def event6859 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45084⟩⟩) (.identity (.predecessor 0 6858 .coefficient))

def event6860 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨45084⟩⟩) (.finite 3364)

def event6861 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45444⟩⟩) 0 ⟨45084⟩ 6860

def event6862 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45444⟩⟩) (.authority (.programFamilyFact))

def exact6863RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨45444⟩⟩], []⟩, (1)⟩]

theorem exact6863RawTermsValid :
    exact6863RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event6863 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45444⟩⟩) exact6863RawTerms (.finite 58) 6862 .exactZero (none)

def event6864 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45445⟩⟩) 0 ⟨45444⟩ 6863

def event6865 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45445⟩⟩) (.identity (.predecessor 0 6864 .coefficient))

def event6866 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨45445⟩⟩) (.finite 58)

def event6867 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45644⟩⟩) 0 ⟨45445⟩ 6866

def event6868 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45644⟩⟩) (.authority (.programFamilyFact))

def exact6869RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨45644⟩⟩], []⟩, (1)⟩]

theorem exact6869RawTermsValid :
    exact6869RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event6869 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45644⟩⟩) exact6869RawTerms (.finite 63) 6868 .exactZero (none)

def event6870 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42402⟩⟩) 0 ⟨5541⟩ 6823

def event6871 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42402⟩⟩) (.authority (.programFamilyFact))

def exact6872RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨42402⟩⟩], []⟩, (1)⟩]

theorem exact6872RawTermsValid :
    exact6872RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event6872 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42402⟩⟩) exact6872RawTerms (.finite 52) 6871 .exactZero (none)

def event6873 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14436⟩⟩) 0 ⟨5541⟩ 6823

def event6874 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14436⟩⟩) (.authority (.programFamilyFact))

def exact6875RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14436⟩⟩], []⟩, (1)⟩]

theorem exact6875RawTermsValid :
    exact6875RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event6875 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14436⟩⟩) exact6875RawTerms (.finite 52) 6874 .exactZero (none)

def event6876 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42403⟩⟩) 0 ⟨14436⟩ 6875

def event6877 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42403⟩⟩) 1 ⟨42402⟩ 6872

def event6878 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42403⟩⟩) (.product (.predecessor 0 6876 .coefficient) (.predecessor 1 6877 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event6879 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨42403⟩⟩, .operator (⟨6875, 0⟩, ⟨6872, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨14436⟩⟩, ⟨.program ⟨257⟩, ⟨42402⟩⟩], []⟩, (1)⟩)

def exact6880RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14436⟩⟩, ⟨.program ⟨257⟩, ⟨42402⟩⟩], []⟩, (1)⟩]

theorem exact6880RawTermsValid :
    exact6880RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event6880 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42403⟩⟩) exact6880RawTerms (.finite 2704) 6878 .exactZero (none)

def event6881 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42404⟩⟩) 0 ⟨42403⟩ 6880

def event6882 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42404⟩⟩) (.identity (.predecessor 0 6881 .coefficient))

def event6883 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨42404⟩⟩) (.finite 2704)

def event6884 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42764⟩⟩) 0 ⟨42404⟩ 6883

def event6885 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42764⟩⟩) (.authority (.programFamilyFact))

def exact6886RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨42764⟩⟩], []⟩, (1)⟩]

theorem exact6886RawTermsValid :
    exact6886RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event6886 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42764⟩⟩) exact6886RawTerms (.finite 52) 6885 .exactZero (none)

def event6887 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42765⟩⟩) 0 ⟨42764⟩ 6886

def event6888 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42765⟩⟩) (.identity (.predecessor 0 6887 .coefficient))

def event6889 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨42765⟩⟩) (.finite 52)

def event6890 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42960⟩⟩) 0 ⟨42765⟩ 6889

def event6891 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42960⟩⟩) (.authority (.programFamilyFact))

def exact6892RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨42960⟩⟩], []⟩, (1)⟩]

theorem exact6892RawTermsValid :
    exact6892RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event6892 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42960⟩⟩) exact6892RawTerms (.finite 63) 6891 .exactZero (none)

def event6893 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39722⟩⟩) 0 ⟨5541⟩ 6823

def event6894 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39722⟩⟩) (.authority (.programFamilyFact))

def exact6895RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨39722⟩⟩], []⟩, (1)⟩]

theorem exact6895RawTermsValid :
    exact6895RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event6895 : Event := .resultExact (⟨.program ⟨257⟩, ⟨39722⟩⟩) exact6895RawTerms (.finite 46) 6894 .exactZero (none)

def event6896 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14136⟩⟩) 0 ⟨5541⟩ 6823

def event6897 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14136⟩⟩) (.authority (.programFamilyFact))

def exact6898RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14136⟩⟩], []⟩, (1)⟩]

theorem exact6898RawTermsValid :
    exact6898RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event6898 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14136⟩⟩) exact6898RawTerms (.finite 46) 6897 .exactZero (none)

def event6899 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39723⟩⟩) 0 ⟨14136⟩ 6898

def event6900 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39723⟩⟩) 1 ⟨39722⟩ 6895

def event6901 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39723⟩⟩) (.product (.predecessor 0 6899 .coefficient) (.predecessor 1 6900 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event6902 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨39723⟩⟩, .operator (⟨6898, 0⟩, ⟨6895, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨14136⟩⟩, ⟨.program ⟨257⟩, ⟨39722⟩⟩], []⟩, (1)⟩)

def exact6903RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14136⟩⟩, ⟨.program ⟨257⟩, ⟨39722⟩⟩], []⟩, (1)⟩]

theorem exact6903RawTermsValid :
    exact6903RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event6903 : Event := .resultExact (⟨.program ⟨257⟩, ⟨39723⟩⟩) exact6903RawTerms (.finite 2116) 6901 .exactZero (none)

def event6904 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39724⟩⟩) 0 ⟨39723⟩ 6903

def event6905 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39724⟩⟩) (.identity (.predecessor 0 6904 .coefficient))

def event6906 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨39724⟩⟩) (.finite 2116)

def event6907 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40084⟩⟩) 0 ⟨39724⟩ 6906

def event6908 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40084⟩⟩) (.authority (.programFamilyFact))

def exact6909RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨40084⟩⟩], []⟩, (1)⟩]

theorem exact6909RawTermsValid :
    exact6909RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event6909 : Event := .resultExact (⟨.program ⟨257⟩, ⟨40084⟩⟩) exact6909RawTerms (.finite 46) 6908 .exactZero (none)

def event6910 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40085⟩⟩) 0 ⟨40084⟩ 6909

def event6911 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40085⟩⟩) (.identity (.predecessor 0 6910 .coefficient))

def eventLeaf416 : Array AnnotatedEvent := #[
  { event := event6656
    frameStart := 0 },
  { event := event6657
    frameStart := 0 },
  { event := event6658
    frameStart := 0 },
  { event := event6659
    frameStart := 0 },
  { event := event6660
    frameStart := 0 },
  { event := event6661
    frameStart := 0 },
  { event := event6662
    frameStart := 0 },
  { event := event6663
    frameStart := 0 },
  { event := event6664
    frameStart := 0 },
  { event := event6665
    frameStart := 0 },
  { event := event6666
    frameStart := 0 },
  { event := event6667
    frameStart := 0 },
  { event := event6668
    frameStart := 0 },
  { event := event6669
    frameStart := 0 },
  { event := event6670
    frameStart := 0 },
  { event := event6671
    frameStart := 0 }
]

def eventLeaf417 : Array AnnotatedEvent := #[
  { event := event6672
    frameStart := 0 },
  { event := event6673
    frameStart := 0 },
  { event := event6674
    frameStart := 0 },
  { event := event6675
    frameStart := 0 },
  { event := event6676
    frameStart := 0 },
  { event := event6677
    frameStart := 0 },
  { event := event6678
    frameStart := 0 },
  { event := event6679
    frameStart := 0 },
  { event := event6680
    frameStart := 0 },
  { event := event6681
    frameStart := 0 },
  { event := event6682
    frameStart := 0 },
  { event := event6683
    frameStart := 0 },
  { event := event6684
    frameStart := 0 },
  { event := event6685
    frameStart := 0 },
  { event := event6686
    frameStart := 0 },
  { event := event6687
    frameStart := 0 }
]

def eventLeaf418 : Array AnnotatedEvent := #[
  { event := event6688
    frameStart := 0 },
  { event := event6689
    frameStart := 0 },
  { event := event6690
    frameStart := 0 },
  { event := event6691
    frameStart := 0 },
  { event := event6692
    frameStart := 0 },
  { event := event6693
    frameStart := 0 },
  { event := event6694
    frameStart := 0 },
  { event := event6695
    frameStart := 0 },
  { event := event6696
    frameStart := 0 },
  { event := event6697
    frameStart := 0 },
  { event := event6698
    frameStart := 0 },
  { event := event6699
    frameStart := 0 },
  { event := event6700
    frameStart := 0 },
  { event := event6701
    frameStart := 0 },
  { event := event6702
    frameStart := 0 },
  { event := event6703
    frameStart := 0 }
]

def eventLeaf419 : Array AnnotatedEvent := #[
  { event := event6704
    frameStart := 0 },
  { event := event6705
    frameStart := 0 },
  { event := event6706
    frameStart := 0 },
  { event := event6707
    frameStart := 0 },
  { event := event6708
    frameStart := 0 },
  { event := event6709
    frameStart := 0 },
  { event := event6710
    frameStart := 0 },
  { event := event6711
    frameStart := 0 },
  { event := event6712
    frameStart := 0 },
  { event := event6713
    frameStart := 0 },
  { event := event6714
    frameStart := 0 },
  { event := event6715
    frameStart := 0 },
  { event := event6716
    frameStart := 0 },
  { event := event6717
    frameStart := 0 },
  { event := event6718
    frameStart := 0 },
  { event := event6719
    frameStart := 0 }
]

def eventLeaf420 : Array AnnotatedEvent := #[
  { event := event6720
    frameStart := 0 },
  { event := event6721
    frameStart := 0 },
  { event := event6722
    frameStart := 0 },
  { event := event6723
    frameStart := 0 },
  { event := event6724
    frameStart := 0 },
  { event := event6725
    frameStart := 0 },
  { event := event6726
    frameStart := 0 },
  { event := event6727
    frameStart := 0 },
  { event := event6728
    frameStart := 0 },
  { event := event6729
    frameStart := 0 },
  { event := event6730
    frameStart := 0 },
  { event := event6731
    frameStart := 0 },
  { event := event6732
    frameStart := 0 },
  { event := event6733
    frameStart := 0 },
  { event := event6734
    frameStart := 0 },
  { event := event6735
    frameStart := 0 }
]

def eventLeaf421 : Array AnnotatedEvent := #[
  { event := event6736
    frameStart := 0 },
  { event := event6737
    frameStart := 0 },
  { event := event6738
    frameStart := 0 },
  { event := event6739
    frameStart := 0 },
  { event := event6740
    frameStart := 0 },
  { event := event6741
    frameStart := 0 },
  { event := event6742
    frameStart := 0 },
  { event := event6743
    frameStart := 0 },
  { event := event6744
    frameStart := 0 },
  { event := event6745
    frameStart := 0 },
  { event := event6746
    frameStart := 0 },
  { event := event6747
    frameStart := 0 },
  { event := event6748
    frameStart := 0 },
  { event := event6749
    frameStart := 0 },
  { event := event6750
    frameStart := 0 },
  { event := event6751
    frameStart := 0 }
]

def eventLeaf422 : Array AnnotatedEvent := #[
  { event := event6752
    frameStart := 0 },
  { event := event6753
    frameStart := 0 },
  { event := event6754
    frameStart := 0 },
  { event := event6755
    frameStart := 0 },
  { event := event6756
    frameStart := 0 },
  { event := event6757
    frameStart := 0 },
  { event := event6758
    frameStart := 0 },
  { event := event6759
    frameStart := 0 },
  { event := event6760
    frameStart := 0 },
  { event := event6761
    frameStart := 0 },
  { event := event6762
    frameStart := 0 },
  { event := event6763
    frameStart := 0 },
  { event := event6764
    frameStart := 0 },
  { event := event6765
    frameStart := 0 },
  { event := event6766
    frameStart := 0 },
  { event := event6767
    frameStart := 0 }
]

def eventLeaf423 : Array AnnotatedEvent := #[
  { event := event6768
    frameStart := 0 },
  { event := event6769
    frameStart := 0 },
  { event := event6770
    frameStart := 0 },
  { event := event6771
    frameStart := 0 },
  { event := event6772
    frameStart := 0 },
  { event := event6773
    frameStart := 0 },
  { event := event6774
    frameStart := 0 },
  { event := event6775
    frameStart := 0 },
  { event := event6776
    frameStart := 0 },
  { event := event6777
    frameStart := 0 },
  { event := event6778
    frameStart := 0 },
  { event := event6779
    frameStart := 0 },
  { event := event6780
    frameStart := 0 },
  { event := event6781
    frameStart := 0 },
  { event := event6782
    frameStart := 0 },
  { event := event6783
    frameStart := 0 }
]

def eventLeaf424 : Array AnnotatedEvent := #[
  { event := event6784
    frameStart := 0 },
  { event := event6785
    frameStart := 0 },
  { event := event6786
    frameStart := 0 },
  { event := event6787
    frameStart := 0 },
  { event := event6788
    frameStart := 0 },
  { event := event6789
    frameStart := 0 },
  { event := event6790
    frameStart := 0 },
  { event := event6791
    frameStart := 0 },
  { event := event6792
    frameStart := 0 },
  { event := event6793
    frameStart := 0 },
  { event := event6794
    frameStart := 0 },
  { event := event6795
    frameStart := 0 },
  { event := event6796
    frameStart := 0 },
  { event := event6797
    frameStart := 0 },
  { event := event6798
    frameStart := 0 },
  { event := event6799
    frameStart := 0 }
]

def eventLeaf425 : Array AnnotatedEvent := #[
  { event := event6800
    frameStart := 0 },
  { event := event6801
    frameStart := 0 },
  { event := event6802
    frameStart := 0 },
  { event := event6803
    frameStart := 0 },
  { event := event6804
    frameStart := 0 },
  { event := event6805
    frameStart := 0 },
  { event := event6806
    frameStart := 0 },
  { event := event6807
    frameStart := 0 },
  { event := event6808
    frameStart := 0 },
  { event := event6809
    frameStart := 0 },
  { event := event6810
    frameStart := 0 },
  { event := event6811
    frameStart := 0 },
  { event := event6812
    frameStart := 0 },
  { event := event6813
    frameStart := 0 },
  { event := event6814
    frameStart := 0 },
  { event := event6815
    frameStart := 0 }
]

def eventLeaf426 : Array AnnotatedEvent := #[
  { event := event6816
    frameStart := 0 },
  { event := event6817
    frameStart := 0 },
  { event := event6818
    frameStart := 0 },
  { event := event6819
    frameStart := 0 },
  { event := event6820
    frameStart := 0 },
  { event := event6821
    frameStart := 0 },
  { event := event6822
    frameStart := 0 },
  { event := event6823
    frameStart := 0 },
  { event := event6824
    frameStart := 0 },
  { event := event6825
    frameStart := 0 },
  { event := event6826
    frameStart := 0 },
  { event := event6827
    frameStart := 0 },
  { event := event6828
    frameStart := 0 },
  { event := event6829
    frameStart := 0 },
  { event := event6830
    frameStart := 0 },
  { event := event6831
    frameStart := 0 }
]

def eventLeaf427 : Array AnnotatedEvent := #[
  { event := event6832
    frameStart := 0 },
  { event := event6833
    frameStart := 0 },
  { event := event6834
    frameStart := 0 },
  { event := event6835
    frameStart := 0 },
  { event := event6836
    frameStart := 0 },
  { event := event6837
    frameStart := 0 },
  { event := event6838
    frameStart := 0 },
  { event := event6839
    frameStart := 0 },
  { event := event6840
    frameStart := 0 },
  { event := event6841
    frameStart := 0 },
  { event := event6842
    frameStart := 0 },
  { event := event6843
    frameStart := 0 },
  { event := event6844
    frameStart := 0 },
  { event := event6845
    frameStart := 0 },
  { event := event6846
    frameStart := 0 },
  { event := event6847
    frameStart := 0 }
]

def eventLeaf428 : Array AnnotatedEvent := #[
  { event := event6848
    frameStart := 0 },
  { event := event6849
    frameStart := 0 },
  { event := event6850
    frameStart := 0 },
  { event := event6851
    frameStart := 0 },
  { event := event6852
    frameStart := 0 },
  { event := event6853
    frameStart := 0 },
  { event := event6854
    frameStart := 0 },
  { event := event6855
    frameStart := 0 },
  { event := event6856
    frameStart := 0 },
  { event := event6857
    frameStart := 0 },
  { event := event6858
    frameStart := 0 },
  { event := event6859
    frameStart := 0 },
  { event := event6860
    frameStart := 0 },
  { event := event6861
    frameStart := 0 },
  { event := event6862
    frameStart := 0 },
  { event := event6863
    frameStart := 0 }
]

def eventLeaf429 : Array AnnotatedEvent := #[
  { event := event6864
    frameStart := 0 },
  { event := event6865
    frameStart := 0 },
  { event := event6866
    frameStart := 0 },
  { event := event6867
    frameStart := 0 },
  { event := event6868
    frameStart := 0 },
  { event := event6869
    frameStart := 0 },
  { event := event6870
    frameStart := 0 },
  { event := event6871
    frameStart := 0 },
  { event := event6872
    frameStart := 0 },
  { event := event6873
    frameStart := 0 },
  { event := event6874
    frameStart := 0 },
  { event := event6875
    frameStart := 0 },
  { event := event6876
    frameStart := 0 },
  { event := event6877
    frameStart := 0 },
  { event := event6878
    frameStart := 0 },
  { event := event6879
    frameStart := 0 }
]

def eventLeaf430 : Array AnnotatedEvent := #[
  { event := event6880
    frameStart := 0 },
  { event := event6881
    frameStart := 0 },
  { event := event6882
    frameStart := 0 },
  { event := event6883
    frameStart := 0 },
  { event := event6884
    frameStart := 0 },
  { event := event6885
    frameStart := 0 },
  { event := event6886
    frameStart := 0 },
  { event := event6887
    frameStart := 0 },
  { event := event6888
    frameStart := 0 },
  { event := event6889
    frameStart := 0 },
  { event := event6890
    frameStart := 0 },
  { event := event6891
    frameStart := 0 },
  { event := event6892
    frameStart := 0 },
  { event := event6893
    frameStart := 0 },
  { event := event6894
    frameStart := 0 },
  { event := event6895
    frameStart := 0 }
]

def eventLeaf431 : Array AnnotatedEvent := #[
  { event := event6896
    frameStart := 0 },
  { event := event6897
    frameStart := 0 },
  { event := event6898
    frameStart := 0 },
  { event := event6899
    frameStart := 0 },
  { event := event6900
    frameStart := 0 },
  { event := event6901
    frameStart := 0 },
  { event := event6902
    frameStart := 0 },
  { event := event6903
    frameStart := 0 },
  { event := event6904
    frameStart := 0 },
  { event := event6905
    frameStart := 0 },
  { event := event6906
    frameStart := 0 },
  { event := event6907
    frameStart := 0 },
  { event := event6908
    frameStart := 0 },
  { event := event6909
    frameStart := 0 },
  { event := event6910
    frameStart := 0 },
  { event := event6911
    frameStart := 0 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events026
