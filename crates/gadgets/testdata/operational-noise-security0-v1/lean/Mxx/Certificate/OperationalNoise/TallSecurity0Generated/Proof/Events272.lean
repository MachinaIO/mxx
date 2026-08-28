import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events272

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event69632 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28070⟩⟩) 0 ⟨24222⟩ 69631

def event69633 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28070⟩⟩) (.authority (.operator))

def exact69634RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨28070⟩⟩]⟩, (1)⟩]

theorem exact69634RawTermsValid :
    exact69634RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69634 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28070⟩⟩) exact69634RawTerms (.finite 8192) 69633 .exactZero (none)

def event69635 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23623⟩⟩) 0 ⟨14417⟩ 3304

def event69636 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23623⟩⟩) (.authority (.programFamilyFact))

def event69637 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨23623⟩⟩) (.finite 3720)

def event69638 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23624⟩⟩) 0 ⟨6689⟩ 5477

def event69639 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23624⟩⟩) 1 ⟨23623⟩ 69637

def event69640 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23624⟩⟩) (.authority (.operator))

def exact69641RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨23624⟩⟩]⟩, (1)⟩]

theorem exact69641RawTermsValid :
    exact69641RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69641 : Event := .resultExact (⟨.program ⟨214⟩, ⟨23624⟩⟩) exact69641RawTerms .large 69640 .exactZero (none)

def event69642 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26138⟩⟩) 0 ⟨23624⟩ 69641

def event69643 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26138⟩⟩) (.authority (.operator))

def exact69644RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨26138⟩⟩]⟩, (1)⟩]

theorem exact69644RawTermsValid :
    exact69644RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69644 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26138⟩⟩) exact69644RawTerms (.finite 8192) 69643 .exactZero (none)

def event69645 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11550⟩⟩) 0 ⟨11549⟩ 3293

def event69646 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11550⟩⟩) 1 ⟨6566⟩ 65295

def event69647 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11550⟩⟩) (.tensor (.predecessor 0 69645 .coefficient) (.predecessor 1 69646 .coefficient) true false)

def event69648 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨11550⟩⟩, .operator (⟨3293, 0⟩, ⟨65295, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨11549⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact69649RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨11549⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact69649RawTermsValid :
    exact69649RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69649 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11550⟩⟩) exact69649RawTerms .large 69647 .exactZero (none)

def event69650 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7198⟩⟩) 0 ⟨5533⟩ 65165

def event69651 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7198⟩⟩) 1 ⟨6780⟩ 10981

def event69652 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7198⟩⟩) (.product (.predecessor 0 69650 .coefficient) (.predecessor 1 69651 .coefficient) (⟨false, false, none, none, none⟩))

def event69653 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7198⟩⟩, .operator (⟨65165, 0⟩, ⟨10981, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6780⟩⟩]⟩, (1)⟩)

def exact69654RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6780⟩⟩]⟩, (1)⟩]

theorem exact69654RawTermsValid :
    exact69654RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69654 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7198⟩⟩) exact69654RawTerms .large 69652 .exactZero (none)

def event69655 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11551⟩⟩) 0 ⟨7198⟩ 69654

def event69656 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11551⟩⟩) 1 ⟨11550⟩ 69649

def event69657 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11551⟩⟩) (.sum [.predecessor 0 69655 .coefficient, .predecessor 1 69656 .coefficient])

def exact69658RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6780⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨11549⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact69658RawTermsValid :
    exact69658RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69658 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11551⟩⟩) exact69658RawTerms .large 69657 .exactZero (none)

def event69659 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11552⟩⟩) 0 ⟨11551⟩ 69658

def event69660 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11552⟩⟩) 1 ⟨94⟩ 10973

def event69661 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11552⟩⟩) (.sum [.predecessor 0 69659 .coefficient, .predecessor 1 69660 .coefficient])

def event69662 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11552⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨94⟩⟩]⟩) [⟨.result 10973 .coefficient, false, none⟩])

def event69663 : Event := .survivorFold (1) 69662

def exact69664RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6780⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨11549⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact69664RawTermsValid :
    exact69664RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69664 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11552⟩⟩) exact69664RawTerms .large 69661 (.finite 26) (some (69662))

def event69665 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14418⟩⟩) 0 ⟨11552⟩ 69664

def event69666 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14418⟩⟩) 1 ⟨14415⟩ 3296

def event69667 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14418⟩⟩) (.product (.predecessor 0 69665 .coefficient) (.predecessor 1 69666 .coefficient) (⟨false, true, none, none, some 1⟩))

def event69668 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14418⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨14415⟩⟩], []⟩) [⟨.result 3296 .coefficient, true, some 1⟩])

def event69669 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14418⟩⟩) (.product (.result 69664 .summary) (.transfer 69668) (⟨false, false, none, none, none⟩))

def event69670 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨14418⟩⟩, .operator (⟨69664, 1⟩, ⟨3296, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨11549⟩⟩, ⟨.program ⟨214⟩, ⟨14415⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def event69671 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨14418⟩⟩, .operator (⟨69664, 0⟩, ⟨3296, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨14415⟩⟩], [⟨.program ⟨214⟩, ⟨6780⟩⟩]⟩, (1)⟩)

def exact69672RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨11549⟩⟩, ⟨.program ⟨214⟩, ⟨14415⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨14415⟩⟩], [⟨.program ⟨214⟩, ⟨6780⟩⟩]⟩, (1)⟩]

theorem exact69672RawTermsValid :
    exact69672RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69672 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14418⟩⟩) exact69672RawTerms .large 69667 (.finite 18304) (some (69669))

def event69673 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14419⟩⟩) 0 ⟨14415⟩ 3296

def event69674 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14419⟩⟩) 1 ⟨6566⟩ 65295

def event69675 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14419⟩⟩) (.tensor (.predecessor 0 69673 .coefficient) (.predecessor 1 69674 .coefficient) true false)

def event69676 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨14419⟩⟩, .operator (⟨3296, 0⟩, ⟨65295, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨14415⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact69677RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨14415⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact69677RawTermsValid :
    exact69677RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69677 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14419⟩⟩) exact69677RawTerms .large 69675 .exactZero (none)

def event69678 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7179⟩⟩) 0 ⟨5533⟩ 65165

def event69679 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7179⟩⟩) 1 ⟨6761⟩ 11022

def event69680 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7179⟩⟩) (.product (.predecessor 0 69678 .coefficient) (.predecessor 1 69679 .coefficient) (⟨false, false, none, none, none⟩))

def event69681 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7179⟩⟩, .operator (⟨65165, 0⟩, ⟨11022, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6761⟩⟩]⟩, (1)⟩)

def exact69682RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6761⟩⟩]⟩, (1)⟩]

theorem exact69682RawTermsValid :
    exact69682RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69682 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7179⟩⟩) exact69682RawTerms .large 69680 .exactZero (none)

def event69683 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14420⟩⟩) 0 ⟨7179⟩ 69682

def event69684 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14420⟩⟩) 1 ⟨14419⟩ 69677

def event69685 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14420⟩⟩) (.sum [.predecessor 0 69683 .coefficient, .predecessor 1 69684 .coefficient])

def exact69686RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6761⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨14415⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact69686RawTermsValid :
    exact69686RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69686 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14420⟩⟩) exact69686RawTerms .large 69685 .exactZero (none)

def event69687 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14421⟩⟩) 0 ⟨14420⟩ 69686

def event69688 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14421⟩⟩) 1 ⟨75⟩ 11014

def event69689 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14421⟩⟩) (.sum [.predecessor 0 69687 .coefficient, .predecessor 1 69688 .coefficient])

def event69690 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14421⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨75⟩⟩]⟩) [⟨.result 11014 .coefficient, false, none⟩])

def event69691 : Event := .survivorFold (1) 69690

def exact69692RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6761⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨14415⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact69692RawTermsValid :
    exact69692RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69692 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14421⟩⟩) exact69692RawTerms .large 69689 (.finite 26) (some (69690))

def event69693 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14422⟩⟩) 0 ⟨14421⟩ 69692

def event69694 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14422⟩⟩) 1 ⟨7856⟩ 11011

def event69695 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14422⟩⟩) (.product (.predecessor 0 69693 .coefficient) (.predecessor 1 69694 .coefficient) (⟨false, false, none, none, none⟩))

def event69696 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14422⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨7855⟩⟩]⟩) [⟨.result 11007 .coefficient, false, none⟩])

def event69697 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14422⟩⟩) (.product (.result 69692 .summary) (.transfer 69696) (⟨false, false, none, none, none⟩))

def event69698 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨14422⟩⟩, .operator (⟨69692, 1⟩, ⟨11011, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨14415⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7855⟩⟩]⟩, (-1)⟩)

def event69699 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨14422⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨14415⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7855⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨7855⟩⟩) ⟨6780⟩ 10981)

def event69700 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨14422⟩⟩, .relation 69699 0, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨14415⟩⟩], [⟨.program ⟨214⟩, ⟨6780⟩⟩]⟩, (-1)⟩)

def event69701 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨14422⟩⟩, .operator (⟨69692, 0⟩, ⟨11011, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6761⟩⟩, ⟨.program ⟨214⟩, ⟨7855⟩⟩]⟩, (1)⟩)

def exact69702RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6761⟩⟩, ⟨.program ⟨214⟩, ⟨7855⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨14415⟩⟩], [⟨.program ⟨214⟩, ⟨6780⟩⟩]⟩, (-1)⟩]

theorem exact69702RawTermsValid :
    exact69702RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69702 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14422⟩⟩) exact69702RawTerms .large 69695 (.finite 95420416) (some (69697))

def event69703 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14423⟩⟩) 0 ⟨14422⟩ 69702

def event69704 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14423⟩⟩) 1 ⟨14418⟩ 69672

def event69705 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14423⟩⟩) (.sum [.predecessor 0 69703 .coefficient, .predecessor 1 69704 .coefficient])

def event69706 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨14423⟩⟩, .operator (⟨69702, 1⟩, ⟨69672, 1⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨14415⟩⟩], [⟨.program ⟨214⟩, ⟨6780⟩⟩]⟩, (1)⟩)

def event69707 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14423⟩⟩) (.sum [.result 69702 .summary, .result 69672 .summary])

def exact69708RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6761⟩⟩, ⟨.program ⟨214⟩, ⟨7855⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨11549⟩⟩, ⟨.program ⟨214⟩, ⟨14415⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact69708RawTermsValid :
    exact69708RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69708 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14423⟩⟩) exact69708RawTerms .large 69705 (.finite 95438720) (some (69707))

def event69709 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26139⟩⟩) 0 ⟨14423⟩ 69708

def event69710 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26139⟩⟩) 1 ⟨26138⟩ 69644

def event69711 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26139⟩⟩) (.product (.predecessor 0 69709 .coefficient) (.predecessor 1 69710 .coefficient) (⟨false, false, none, none, none⟩))

def event69712 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26139⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨26138⟩⟩]⟩) [⟨.result 69644 .coefficient, false, none⟩])

def event69713 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26139⟩⟩) (.product (.result 69708 .summary) (.transfer 69712) (⟨false, false, none, none, none⟩))

def event69714 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26139⟩⟩, .operator (⟨69708, 1⟩, ⟨69644, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨11549⟩⟩, ⟨.program ⟨214⟩, ⟨14415⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26138⟩⟩]⟩, (-1)⟩)

def event69715 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨26139⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨11549⟩⟩, ⟨.program ⟨214⟩, ⟨14415⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26138⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨26138⟩⟩) ⟨23624⟩ 69641)

def event69716 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26139⟩⟩, .relation 69715 0, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨11549⟩⟩, ⟨.program ⟨214⟩, ⟨14415⟩⟩], [⟨.program ⟨214⟩, ⟨23624⟩⟩]⟩, (-1)⟩)

def event69717 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26139⟩⟩, .operator (⟨69708, 0⟩, ⟨69644, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6761⟩⟩, ⟨.program ⟨214⟩, ⟨7855⟩⟩, ⟨.program ⟨214⟩, ⟨26138⟩⟩]⟩, (1)⟩)

def exact69718RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6761⟩⟩, ⟨.program ⟨214⟩, ⟨7855⟩⟩, ⟨.program ⟨214⟩, ⟨26138⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨11549⟩⟩, ⟨.program ⟨214⟩, ⟨14415⟩⟩], [⟨.program ⟨214⟩, ⟨23624⟩⟩]⟩, (-1)⟩]

theorem exact69718RawTermsValid :
    exact69718RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69718 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26139⟩⟩) exact69718RawTerms .large 69711 (.finite 350261629419520) (some (69713))

def event69719 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19596⟩⟩) 0 ⟨14417⟩ 3304

def event69720 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19596⟩⟩) (.authority (.relationPreimageSource ⟨16⟩))

def exact69721RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨19596⟩⟩]⟩, (1)⟩]

theorem exact69721RawTermsValid :
    exact69721RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69721 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19596⟩⟩) exact69721RawTerms (.finite 136065468) 69720 .exactZero (none)

def event69722 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19598⟩⟩) 0 ⟨19596⟩ 69721

def event69723 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19598⟩⟩) 1 ⟨2348⟩ 4

def event69724 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19598⟩⟩) (.scale (.predecessor 0 69722 .coefficient) (.value (.predecessor 1 69723 .coefficient)))

def exact69725RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨19596⟩⟩]⟩, (1)⟩]

theorem exact69725RawTermsValid :
    exact69725RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69725 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19598⟩⟩) exact69725RawTerms (.finite 136065468) 69724 .exactZero (none)

def event69726 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19599⟩⟩) 0 ⟨5535⟩ 65387

def event69727 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19599⟩⟩) 1 ⟨19598⟩ 69725

def event69728 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19599⟩⟩) (.product (.predecessor 0 69726 .coefficient) (.predecessor 1 69727 .coefficient) (⟨false, false, none, none, none⟩))

def event69729 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19599⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨19596⟩⟩]⟩) [⟨.result 69721 .coefficient, false, none⟩])

def event69730 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19599⟩⟩) (.product (.result 65387 .summary) (.transfer 69729) (⟨false, false, none, none, none⟩))

def event69731 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19599⟩⟩, .operator (⟨65387, 0⟩, ⟨69725, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19596⟩⟩]⟩, (1)⟩)

def event69732 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨19597⟩⟩)

def event69733 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event69734 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event69735 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨360⟩⟩) (.authority (.operator))

def event69736 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨360⟩⟩) (.finite 2)

def event69737 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event69738 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event69739 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event69740 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event69741 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 69740

def event69742 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 69738

def event69743 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 69741 .coefficient) (.value (.predecessor 1 69742 .coefficient)))

def event69744 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event69745 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5504⟩⟩) 0 ⟨5503⟩ 69744

def event69746 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5504⟩⟩) 1 ⟨360⟩ 69736

def event69747 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5504⟩⟩) (.sum [.predecessor 0 69745 .coefficient, .predecessor 1 69746 .coefficient])

def event69748 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5504⟩⟩) (.finite 219)

def event69749 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5530⟩⟩) 0 ⟨5504⟩ 69748

def event69750 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5530⟩⟩) 1 ⟨961⟩ 69734

def event69751 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5530⟩⟩) (.identity (.predecessor 1 69750 .coefficient))

def event69752 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5530⟩⟩) (.finite 224)

def event69753 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11549⟩⟩) 0 ⟨5530⟩ 69752

def event69754 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11549⟩⟩) (.authority (.programFamilyFact))

def exact69755RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11549⟩⟩], []⟩, (1)⟩]

theorem exact69755RawTermsValid :
    exact69755RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69755 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11549⟩⟩) exact69755RawTerms (.finite 22) 69754 .exactZero (none)

def event69756 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14415⟩⟩) 0 ⟨5530⟩ 69752

def event69757 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14415⟩⟩) (.authority (.programFamilyFact))

def exact69758RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨14415⟩⟩], []⟩, (1)⟩]

theorem exact69758RawTermsValid :
    exact69758RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69758 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14415⟩⟩) exact69758RawTerms (.finite 22) 69757 .exactZero (none)

def event69759 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14416⟩⟩) 0 ⟨14415⟩ 69758

def event69760 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14416⟩⟩) 1 ⟨11549⟩ 69755

def event69761 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14416⟩⟩) (.product (.predecessor 0 69759 .coefficient) (.predecessor 1 69760 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event69762 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14416⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨11549⟩⟩, ⟨.program ⟨214⟩, ⟨14415⟩⟩], []⟩) [⟨.result 69758 .coefficient, true, some 1⟩, ⟨.result 69755 .coefficient, true, some 1⟩])

def event69763 : Event := .survivorFold (1) 69762

def exact69764RawTerms : List Term := []

theorem exact69764RawTermsValid :
    exact69764RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69764 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14416⟩⟩) exact69764RawTerms (.finite 484) 69761 (.finite 484) (some (69762))

def event69765 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14417⟩⟩) 0 ⟨14416⟩ 69764

def event69766 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14417⟩⟩) (.identity (.predecessor 0 69765 .coefficient))

def event69767 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨14417⟩⟩) (.finite 484)

def event69768 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19596⟩⟩) 0 ⟨14417⟩ 69767

def event69769 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19596⟩⟩) (.authority (.relationPreimageSource ⟨16⟩))

def exact69770RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨19596⟩⟩]⟩, (1)⟩]

theorem exact69770RawTermsValid :
    exact69770RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69770 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19596⟩⟩) exact69770RawTerms (.finite 136065468) 69769 .exactZero (none)

def event69771 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6⟩⟩) (.authority (.operator))

def exact69772RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩]⟩, (1)⟩]

theorem exact69772RawTermsValid :
    exact69772RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69772 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6⟩⟩) exact69772RawTerms .large 69771 .exactZero (none)

def event69773 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19597⟩⟩) 0 ⟨6⟩ 69772

def event69774 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19597⟩⟩) 1 ⟨19596⟩ 69770

def event69775 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19597⟩⟩) (.product (.predecessor 0 69773 .coefficient) (.predecessor 1 69774 .coefficient) (⟨false, false, none, none, none⟩))

def event69776 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19597⟩⟩, .operator (⟨69772, 0⟩, ⟨69770, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19596⟩⟩]⟩, (1)⟩)

def exact69777RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19596⟩⟩]⟩, (1)⟩]

theorem exact69777RawTermsValid :
    exact69777RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69777 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19597⟩⟩) exact69777RawTerms .large 69775 .exactZero (none)

def event69778 : Event := .preFoldPolynomial 69777 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19596⟩⟩]⟩, (1)⟩] .exactZero none

def exact69779RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19596⟩⟩]⟩, (1)⟩]

def event69779 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨19597⟩⟩) 69778 exact69779RawTerms .large 69775 .exactZero (none)

def event69780 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨26142⟩⟩)

def event69781 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event69782 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event69783 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨360⟩⟩) (.authority (.operator))

def event69784 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨360⟩⟩) (.finite 2)

def event69785 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event69786 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event69787 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event69788 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event69789 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 69788

def event69790 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 69786

def event69791 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 69789 .coefficient) (.value (.predecessor 1 69790 .coefficient)))

def event69792 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event69793 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5504⟩⟩) 0 ⟨5503⟩ 69792

def event69794 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5504⟩⟩) 1 ⟨360⟩ 69784

def event69795 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5504⟩⟩) (.sum [.predecessor 0 69793 .coefficient, .predecessor 1 69794 .coefficient])

def event69796 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5504⟩⟩) (.finite 219)

def event69797 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5530⟩⟩) 0 ⟨5504⟩ 69796

def event69798 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5530⟩⟩) 1 ⟨961⟩ 69782

def event69799 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5530⟩⟩) (.identity (.predecessor 1 69798 .coefficient))

def event69800 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5530⟩⟩) (.finite 224)

def event69801 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11549⟩⟩) 0 ⟨5530⟩ 69800

def event69802 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11549⟩⟩) (.authority (.programFamilyFact))

def exact69803RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11549⟩⟩], []⟩, (1)⟩]

theorem exact69803RawTermsValid :
    exact69803RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69803 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11549⟩⟩) exact69803RawTerms (.finite 22) 69802 .exactZero (none)

def event69804 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14415⟩⟩) 0 ⟨5530⟩ 69800

def event69805 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14415⟩⟩) (.authority (.programFamilyFact))

def exact69806RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨14415⟩⟩], []⟩, (1)⟩]

theorem exact69806RawTermsValid :
    exact69806RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69806 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14415⟩⟩) exact69806RawTerms (.finite 22) 69805 .exactZero (none)

def event69807 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14416⟩⟩) 0 ⟨14415⟩ 69806

def event69808 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14416⟩⟩) 1 ⟨11549⟩ 69803

def event69809 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14416⟩⟩) (.product (.predecessor 0 69807 .coefficient) (.predecessor 1 69808 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event69810 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨14416⟩⟩, .operator (⟨69806, 0⟩, ⟨69803, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨11549⟩⟩, ⟨.program ⟨214⟩, ⟨14415⟩⟩], []⟩, (1)⟩)

def exact69811RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11549⟩⟩, ⟨.program ⟨214⟩, ⟨14415⟩⟩], []⟩, (1)⟩]

theorem exact69811RawTermsValid :
    exact69811RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69811 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14416⟩⟩) exact69811RawTerms (.finite 484) 69809 .exactZero (none)

def event69812 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14417⟩⟩) 0 ⟨14416⟩ 69811

def event69813 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14417⟩⟩) (.identity (.predecessor 0 69812 .coefficient))

def event69814 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨14417⟩⟩) (.finite 484)

def event69815 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23623⟩⟩) 0 ⟨14417⟩ 69814

def event69816 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23623⟩⟩) (.authority (.programFamilyFact))

def event69817 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨23623⟩⟩) (.finite 3720)

def event69818 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨6689⟩⟩) .missing

def event69819 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23624⟩⟩) 0 ⟨6689⟩ 69818

def event69820 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23624⟩⟩) 1 ⟨23623⟩ 69817

def event69821 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23624⟩⟩) (.authority (.operator))

def exact69822RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨23624⟩⟩]⟩, (1)⟩]

theorem exact69822RawTermsValid :
    exact69822RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69822 : Event := .resultExact (⟨.program ⟨214⟩, ⟨23624⟩⟩) exact69822RawTerms .large 69821 .exactZero (none)

def event69823 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26138⟩⟩) 0 ⟨23624⟩ 69822

def event69824 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26138⟩⟩) (.authority (.operator))

def exact69825RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨26138⟩⟩]⟩, (1)⟩]

theorem exact69825RawTermsValid :
    exact69825RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69825 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26138⟩⟩) exact69825RawTerms (.finite 8192) 69824 .exactZero (none)

def event69826 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨110⟩⟩) (.authority (.operator))

def event69827 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨110⟩⟩) .exactZero

def event69828 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14527⟩⟩) 0 ⟨14417⟩ 69814

def event69829 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14527⟩⟩) 1 ⟨110⟩ 69827

def event69830 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14527⟩⟩) (.sum [.predecessor 0 69828 .coefficient, .predecessor 1 69829 .coefficient])

def event69831 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨14527⟩⟩) (.finite 484)

def event69832 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14528⟩⟩) 0 ⟨14527⟩ 69831

def event69833 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14528⟩⟩) (.identity (.predecessor 0 69832 .coefficient))

def exact69834RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11549⟩⟩, ⟨.program ⟨214⟩, ⟨14415⟩⟩], []⟩, (1)⟩]

theorem exact69834RawTermsValid :
    exact69834RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69834 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14528⟩⟩) exact69834RawTerms (.finite 484) 69833 .exactZero (none)

def event69835 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6544⟩⟩) (.authority (.factStore))

def exact69836RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact69836RawTermsValid :
    exact69836RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69836 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6544⟩⟩) exact69836RawTerms .large 69835 .exactZero (none)

def event69837 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14529⟩⟩) 0 ⟨6544⟩ 69836

def event69838 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14529⟩⟩) 1 ⟨14528⟩ 69834

def event69839 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14529⟩⟩) (.product (.predecessor 0 69837 .coefficient) (.predecessor 1 69838 .coefficient) (⟨false, false, none, none, none⟩))

def event69840 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨14529⟩⟩, .operator (⟨69836, 0⟩, ⟨69834, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨11549⟩⟩, ⟨.program ⟨214⟩, ⟨14415⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact69841RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11549⟩⟩, ⟨.program ⟨214⟩, ⟨14415⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact69841RawTermsValid :
    exact69841RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69841 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14529⟩⟩) exact69841RawTerms .large 69839 .exactZero (none)

def event69842 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.authority (.operator))

def event69843 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.finite 1)

def event69844 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6757⟩⟩) 0 ⟨6689⟩ 69818

def event69845 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6757⟩⟩) (.authority (.operator))

def exact69846RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6757⟩⟩]⟩, (1)⟩]

theorem exact69846RawTermsValid :
    exact69846RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69846 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6757⟩⟩) exact69846RawTerms .large 69845 .exactZero (none)

def event69847 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6780⟩⟩) 0 ⟨6757⟩ 69846

def event69848 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6780⟩⟩) (.identity (.predecessor 0 69847 .coefficient))

def exact69849RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6780⟩⟩]⟩, (1)⟩]

theorem exact69849RawTermsValid :
    exact69849RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69849 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6780⟩⟩) exact69849RawTerms .large 69848 .exactZero (none)

def event69850 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7855⟩⟩) 0 ⟨6780⟩ 69849

def event69851 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7855⟩⟩) (.authority (.operator))

def exact69852RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨7855⟩⟩]⟩, (1)⟩]

theorem exact69852RawTermsValid :
    exact69852RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69852 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7855⟩⟩) exact69852RawTerms (.finite 8192) 69851 .exactZero (none)

def event69853 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7856⟩⟩) 0 ⟨7855⟩ 69852

def event69854 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7856⟩⟩) 1 ⟨2348⟩ 69843

def event69855 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7856⟩⟩) (.scale (.predecessor 0 69853 .coefficient) (.value (.predecessor 1 69854 .coefficient)))

def exact69856RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨7855⟩⟩]⟩, (1)⟩]

theorem exact69856RawTermsValid :
    exact69856RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69856 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7856⟩⟩) exact69856RawTerms (.finite 8192) 69855 .exactZero (none)

def event69857 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6761⟩⟩) 0 ⟨6757⟩ 69846

def event69858 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6761⟩⟩) (.identity (.predecessor 0 69857 .coefficient))

def exact69859RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6761⟩⟩]⟩, (1)⟩]

theorem exact69859RawTermsValid :
    exact69859RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69859 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6761⟩⟩) exact69859RawTerms .large 69858 .exactZero (none)

def event69860 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7857⟩⟩) 0 ⟨6761⟩ 69859

def event69861 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7857⟩⟩) 1 ⟨7856⟩ 69856

def event69862 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7857⟩⟩) (.product (.predecessor 0 69860 .coefficient) (.predecessor 1 69861 .coefficient) (⟨false, false, none, none, none⟩))

def event69863 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7857⟩⟩, .operator (⟨69859, 0⟩, ⟨69856, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6761⟩⟩, ⟨.program ⟨214⟩, ⟨7855⟩⟩]⟩, (1)⟩)

def exact69864RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6761⟩⟩, ⟨.program ⟨214⟩, ⟨7855⟩⟩]⟩, (1)⟩]

theorem exact69864RawTermsValid :
    exact69864RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69864 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7857⟩⟩) exact69864RawTerms .large 69862 .exactZero (none)

def event69865 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14530⟩⟩) 0 ⟨7857⟩ 69864

def event69866 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14530⟩⟩) 1 ⟨14529⟩ 69841

def event69867 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14530⟩⟩) (.sum [.predecessor 0 69865 .coefficient, .predecessor 1 69866 .coefficient])

def exact69868RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6761⟩⟩, ⟨.program ⟨214⟩, ⟨7855⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨11549⟩⟩, ⟨.program ⟨214⟩, ⟨14415⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact69868RawTermsValid :
    exact69868RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69868 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14530⟩⟩) exact69868RawTerms .large 69867 .exactZero (none)

def event69869 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26141⟩⟩) 0 ⟨14530⟩ 69868

def event69870 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26141⟩⟩) 1 ⟨26138⟩ 69825

def event69871 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26141⟩⟩) (.product (.predecessor 0 69869 .coefficient) (.predecessor 1 69870 .coefficient) (⟨false, false, none, none, none⟩))

def event69872 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26141⟩⟩, .operator (⟨69868, 0⟩, ⟨69825, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6761⟩⟩, ⟨.program ⟨214⟩, ⟨7855⟩⟩, ⟨.program ⟨214⟩, ⟨26138⟩⟩]⟩, (1)⟩)

def event69873 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26141⟩⟩, .operator (⟨69868, 1⟩, ⟨69825, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨11549⟩⟩, ⟨.program ⟨214⟩, ⟨14415⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26138⟩⟩]⟩, (-1)⟩)

def event69874 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨26141⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨11549⟩⟩, ⟨.program ⟨214⟩, ⟨14415⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26138⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨26138⟩⟩) ⟨23624⟩ 69822)

def event69875 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26141⟩⟩, .relation 69874 0, ⟨[⟨.program ⟨214⟩, ⟨11549⟩⟩, ⟨.program ⟨214⟩, ⟨14415⟩⟩], [⟨.program ⟨214⟩, ⟨23624⟩⟩]⟩, (-1)⟩)

def exact69876RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6761⟩⟩, ⟨.program ⟨214⟩, ⟨7855⟩⟩, ⟨.program ⟨214⟩, ⟨26138⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨11549⟩⟩, ⟨.program ⟨214⟩, ⟨14415⟩⟩], [⟨.program ⟨214⟩, ⟨23624⟩⟩]⟩, (-1)⟩]

theorem exact69876RawTermsValid :
    exact69876RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69876 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26141⟩⟩) exact69876RawTerms .large 69871 .exactZero (none)

def event69877 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16055⟩⟩) 0 ⟨14417⟩ 69814

def event69878 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16055⟩⟩) (.authority (.programFamilyFact))

def exact69879RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16055⟩⟩], []⟩, (1)⟩]

theorem exact69879RawTermsValid :
    exact69879RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69879 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16055⟩⟩) exact69879RawTerms (.finite 22) 69878 .exactZero (none)

def event69880 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16057⟩⟩) 0 ⟨6544⟩ 69836

def event69881 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16057⟩⟩) 1 ⟨16055⟩ 69879

def event69882 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16057⟩⟩) (.product (.predecessor 0 69880 .coefficient) (.predecessor 1 69881 .coefficient) (⟨false, true, none, none, some 1⟩))

def event69883 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨16057⟩⟩, .operator (⟨69836, 0⟩, ⟨69879, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨16055⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact69884RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16055⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact69884RawTermsValid :
    exact69884RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69884 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16057⟩⟩) exact69884RawTerms .large 69882 .exactZero (none)

def event69885 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6698⟩⟩) 0 ⟨6689⟩ 69818

def event69886 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6698⟩⟩) (.authority (.operator))

def exact69887RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6698⟩⟩]⟩, (1)⟩]

theorem exact69887RawTermsValid :
    exact69887RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69887 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6698⟩⟩) exact69887RawTerms .large 69886 .exactZero (none)

def eventLeaf4352 : Array AnnotatedEvent := #[
  { event := event69632
    frameStart := 0 },
  { event := event69633
    frameStart := 0 },
  { event := event69634
    frameStart := 0 },
  { event := event69635
    frameStart := 0 },
  { event := event69636
    frameStart := 0 },
  { event := event69637
    frameStart := 0 },
  { event := event69638
    frameStart := 0 },
  { event := event69639
    frameStart := 0 },
  { event := event69640
    frameStart := 0 },
  { event := event69641
    frameStart := 0 },
  { event := event69642
    frameStart := 0 },
  { event := event69643
    frameStart := 0 },
  { event := event69644
    frameStart := 0 },
  { event := event69645
    frameStart := 0 },
  { event := event69646
    frameStart := 0 },
  { event := event69647
    frameStart := 0 }
]

def eventLeaf4353 : Array AnnotatedEvent := #[
  { event := event69648
    frameStart := 0 },
  { event := event69649
    frameStart := 0 },
  { event := event69650
    frameStart := 0 },
  { event := event69651
    frameStart := 0 },
  { event := event69652
    frameStart := 0 },
  { event := event69653
    frameStart := 0 },
  { event := event69654
    frameStart := 0 },
  { event := event69655
    frameStart := 0 },
  { event := event69656
    frameStart := 0 },
  { event := event69657
    frameStart := 0 },
  { event := event69658
    frameStart := 0 },
  { event := event69659
    frameStart := 0 },
  { event := event69660
    frameStart := 0 },
  { event := event69661
    frameStart := 0 },
  { event := event69662
    frameStart := 0 },
  { event := event69663
    frameStart := 0 }
]

def eventLeaf4354 : Array AnnotatedEvent := #[
  { event := event69664
    frameStart := 0 },
  { event := event69665
    frameStart := 0 },
  { event := event69666
    frameStart := 0 },
  { event := event69667
    frameStart := 0 },
  { event := event69668
    frameStart := 0 },
  { event := event69669
    frameStart := 0 },
  { event := event69670
    frameStart := 0 },
  { event := event69671
    frameStart := 0 },
  { event := event69672
    frameStart := 0 },
  { event := event69673
    frameStart := 0 },
  { event := event69674
    frameStart := 0 },
  { event := event69675
    frameStart := 0 },
  { event := event69676
    frameStart := 0 },
  { event := event69677
    frameStart := 0 },
  { event := event69678
    frameStart := 0 },
  { event := event69679
    frameStart := 0 }
]

def eventLeaf4355 : Array AnnotatedEvent := #[
  { event := event69680
    frameStart := 0 },
  { event := event69681
    frameStart := 0 },
  { event := event69682
    frameStart := 0 },
  { event := event69683
    frameStart := 0 },
  { event := event69684
    frameStart := 0 },
  { event := event69685
    frameStart := 0 },
  { event := event69686
    frameStart := 0 },
  { event := event69687
    frameStart := 0 },
  { event := event69688
    frameStart := 0 },
  { event := event69689
    frameStart := 0 },
  { event := event69690
    frameStart := 0 },
  { event := event69691
    frameStart := 0 },
  { event := event69692
    frameStart := 0 },
  { event := event69693
    frameStart := 0 },
  { event := event69694
    frameStart := 0 },
  { event := event69695
    frameStart := 0 }
]

def eventLeaf4356 : Array AnnotatedEvent := #[
  { event := event69696
    frameStart := 0 },
  { event := event69697
    frameStart := 0 },
  { event := event69698
    frameStart := 0 },
  { event := event69699
    frameStart := 0 },
  { event := event69700
    frameStart := 0 },
  { event := event69701
    frameStart := 0 },
  { event := event69702
    frameStart := 0 },
  { event := event69703
    frameStart := 0 },
  { event := event69704
    frameStart := 0 },
  { event := event69705
    frameStart := 0 },
  { event := event69706
    frameStart := 0 },
  { event := event69707
    frameStart := 0 },
  { event := event69708
    frameStart := 0 },
  { event := event69709
    frameStart := 0 },
  { event := event69710
    frameStart := 0 },
  { event := event69711
    frameStart := 0 }
]

def eventLeaf4357 : Array AnnotatedEvent := #[
  { event := event69712
    frameStart := 0 },
  { event := event69713
    frameStart := 0 },
  { event := event69714
    frameStart := 0 },
  { event := event69715
    frameStart := 0 },
  { event := event69716
    frameStart := 0 },
  { event := event69717
    frameStart := 0 },
  { event := event69718
    frameStart := 0 },
  { event := event69719
    frameStart := 0 },
  { event := event69720
    frameStart := 0 },
  { event := event69721
    frameStart := 0 },
  { event := event69722
    frameStart := 0 },
  { event := event69723
    frameStart := 0 },
  { event := event69724
    frameStart := 0 },
  { event := event69725
    frameStart := 0 },
  { event := event69726
    frameStart := 0 },
  { event := event69727
    frameStart := 0 }
]

def eventLeaf4358 : Array AnnotatedEvent := #[
  { event := event69728
    frameStart := 0 },
  { event := event69729
    frameStart := 0 },
  { event := event69730
    frameStart := 0 },
  { event := event69731
    frameStart := 0 },
  { event := event69732
    frameStart := 69732 },
  { event := event69733
    frameStart := 69732 },
  { event := event69734
    frameStart := 69732 },
  { event := event69735
    frameStart := 69732 },
  { event := event69736
    frameStart := 69732 },
  { event := event69737
    frameStart := 69732 },
  { event := event69738
    frameStart := 69732 },
  { event := event69739
    frameStart := 69732 },
  { event := event69740
    frameStart := 69732 },
  { event := event69741
    frameStart := 69732 },
  { event := event69742
    frameStart := 69732 },
  { event := event69743
    frameStart := 69732 }
]

def eventLeaf4359 : Array AnnotatedEvent := #[
  { event := event69744
    frameStart := 69732 },
  { event := event69745
    frameStart := 69732 },
  { event := event69746
    frameStart := 69732 },
  { event := event69747
    frameStart := 69732 },
  { event := event69748
    frameStart := 69732 },
  { event := event69749
    frameStart := 69732 },
  { event := event69750
    frameStart := 69732 },
  { event := event69751
    frameStart := 69732 },
  { event := event69752
    frameStart := 69732 },
  { event := event69753
    frameStart := 69732 },
  { event := event69754
    frameStart := 69732 },
  { event := event69755
    frameStart := 69732 },
  { event := event69756
    frameStart := 69732 },
  { event := event69757
    frameStart := 69732 },
  { event := event69758
    frameStart := 69732 },
  { event := event69759
    frameStart := 69732 }
]

def eventLeaf4360 : Array AnnotatedEvent := #[
  { event := event69760
    frameStart := 69732 },
  { event := event69761
    frameStart := 69732 },
  { event := event69762
    frameStart := 69732 },
  { event := event69763
    frameStart := 69732 },
  { event := event69764
    frameStart := 69732 },
  { event := event69765
    frameStart := 69732 },
  { event := event69766
    frameStart := 69732 },
  { event := event69767
    frameStart := 69732 },
  { event := event69768
    frameStart := 69732 },
  { event := event69769
    frameStart := 69732 },
  { event := event69770
    frameStart := 69732 },
  { event := event69771
    frameStart := 69732 },
  { event := event69772
    frameStart := 69732 },
  { event := event69773
    frameStart := 69732 },
  { event := event69774
    frameStart := 69732 },
  { event := event69775
    frameStart := 69732 }
]

def eventLeaf4361 : Array AnnotatedEvent := #[
  { event := event69776
    frameStart := 69732 },
  { event := event69777
    frameStart := 69732 },
  { event := event69778
    frameStart := 69732 },
  { event := event69779
    frameStart := 69732 },
  { event := event69780
    frameStart := 69780 },
  { event := event69781
    frameStart := 69780 },
  { event := event69782
    frameStart := 69780 },
  { event := event69783
    frameStart := 69780 },
  { event := event69784
    frameStart := 69780 },
  { event := event69785
    frameStart := 69780 },
  { event := event69786
    frameStart := 69780 },
  { event := event69787
    frameStart := 69780 },
  { event := event69788
    frameStart := 69780 },
  { event := event69789
    frameStart := 69780 },
  { event := event69790
    frameStart := 69780 },
  { event := event69791
    frameStart := 69780 }
]

def eventLeaf4362 : Array AnnotatedEvent := #[
  { event := event69792
    frameStart := 69780 },
  { event := event69793
    frameStart := 69780 },
  { event := event69794
    frameStart := 69780 },
  { event := event69795
    frameStart := 69780 },
  { event := event69796
    frameStart := 69780 },
  { event := event69797
    frameStart := 69780 },
  { event := event69798
    frameStart := 69780 },
  { event := event69799
    frameStart := 69780 },
  { event := event69800
    frameStart := 69780 },
  { event := event69801
    frameStart := 69780 },
  { event := event69802
    frameStart := 69780 },
  { event := event69803
    frameStart := 69780 },
  { event := event69804
    frameStart := 69780 },
  { event := event69805
    frameStart := 69780 },
  { event := event69806
    frameStart := 69780 },
  { event := event69807
    frameStart := 69780 }
]

def eventLeaf4363 : Array AnnotatedEvent := #[
  { event := event69808
    frameStart := 69780 },
  { event := event69809
    frameStart := 69780 },
  { event := event69810
    frameStart := 69780 },
  { event := event69811
    frameStart := 69780 },
  { event := event69812
    frameStart := 69780 },
  { event := event69813
    frameStart := 69780 },
  { event := event69814
    frameStart := 69780 },
  { event := event69815
    frameStart := 69780 },
  { event := event69816
    frameStart := 69780 },
  { event := event69817
    frameStart := 69780 },
  { event := event69818
    frameStart := 69780 },
  { event := event69819
    frameStart := 69780 },
  { event := event69820
    frameStart := 69780 },
  { event := event69821
    frameStart := 69780 },
  { event := event69822
    frameStart := 69780 },
  { event := event69823
    frameStart := 69780 }
]

def eventLeaf4364 : Array AnnotatedEvent := #[
  { event := event69824
    frameStart := 69780 },
  { event := event69825
    frameStart := 69780 },
  { event := event69826
    frameStart := 69780 },
  { event := event69827
    frameStart := 69780 },
  { event := event69828
    frameStart := 69780 },
  { event := event69829
    frameStart := 69780 },
  { event := event69830
    frameStart := 69780 },
  { event := event69831
    frameStart := 69780 },
  { event := event69832
    frameStart := 69780 },
  { event := event69833
    frameStart := 69780 },
  { event := event69834
    frameStart := 69780 },
  { event := event69835
    frameStart := 69780 },
  { event := event69836
    frameStart := 69780 },
  { event := event69837
    frameStart := 69780 },
  { event := event69838
    frameStart := 69780 },
  { event := event69839
    frameStart := 69780 }
]

def eventLeaf4365 : Array AnnotatedEvent := #[
  { event := event69840
    frameStart := 69780 },
  { event := event69841
    frameStart := 69780 },
  { event := event69842
    frameStart := 69780 },
  { event := event69843
    frameStart := 69780 },
  { event := event69844
    frameStart := 69780 },
  { event := event69845
    frameStart := 69780 },
  { event := event69846
    frameStart := 69780 },
  { event := event69847
    frameStart := 69780 },
  { event := event69848
    frameStart := 69780 },
  { event := event69849
    frameStart := 69780 },
  { event := event69850
    frameStart := 69780 },
  { event := event69851
    frameStart := 69780 },
  { event := event69852
    frameStart := 69780 },
  { event := event69853
    frameStart := 69780 },
  { event := event69854
    frameStart := 69780 },
  { event := event69855
    frameStart := 69780 }
]

def eventLeaf4366 : Array AnnotatedEvent := #[
  { event := event69856
    frameStart := 69780 },
  { event := event69857
    frameStart := 69780 },
  { event := event69858
    frameStart := 69780 },
  { event := event69859
    frameStart := 69780 },
  { event := event69860
    frameStart := 69780 },
  { event := event69861
    frameStart := 69780 },
  { event := event69862
    frameStart := 69780 },
  { event := event69863
    frameStart := 69780 },
  { event := event69864
    frameStart := 69780 },
  { event := event69865
    frameStart := 69780 },
  { event := event69866
    frameStart := 69780 },
  { event := event69867
    frameStart := 69780 },
  { event := event69868
    frameStart := 69780 },
  { event := event69869
    frameStart := 69780 },
  { event := event69870
    frameStart := 69780 },
  { event := event69871
    frameStart := 69780 }
]

def eventLeaf4367 : Array AnnotatedEvent := #[
  { event := event69872
    frameStart := 69780 },
  { event := event69873
    frameStart := 69780 },
  { event := event69874
    frameStart := 69780 },
  { event := event69875
    frameStart := 69780 },
  { event := event69876
    frameStart := 69780 },
  { event := event69877
    frameStart := 69780 },
  { event := event69878
    frameStart := 69780 },
  { event := event69879
    frameStart := 69780 },
  { event := event69880
    frameStart := 69780 },
  { event := event69881
    frameStart := 69780 },
  { event := event69882
    frameStart := 69780 },
  { event := event69883
    frameStart := 69780 },
  { event := event69884
    frameStart := 69780 },
  { event := event69885
    frameStart := 69780 },
  { event := event69886
    frameStart := 69780 },
  { event := event69887
    frameStart := 69780 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events272
