import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events190

open Mxx.Certificate.OperationalNoise
open CertificateABI

def exact48640RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7298⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨14001⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact48640RawTermsValid :
    exact48640RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48640 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14004⟩⟩) exact48640RawTerms .large 48637 (.finite 26) (some (48638))

def event48641 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14005⟩⟩) 0 ⟨14004⟩ 48640

def event48642 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14005⟩⟩) 1 ⟨9554⟩ 19114

def event48643 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14005⟩⟩) (.product (.predecessor 0 48641 .coefficient) (.predecessor 1 48642 .coefficient) (⟨false, false, none, none, none⟩))

def event48644 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14005⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨9553⟩⟩]⟩) [⟨.result 19110 .coefficient, false, none⟩])

def event48645 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14005⟩⟩) (.product (.result 48640 .summary) (.transfer 48644) (⟨false, false, none, none, none⟩))

def event48646 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨14005⟩⟩, .operator (⟨48640, 1⟩, ⟨19114, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨14001⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩]⟩, (-1)⟩)

def event48647 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨14005⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨14001⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨9553⟩⟩) ⟨7281⟩ 19084)

def event48648 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨14005⟩⟩, .relation 48647 0, ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨14001⟩⟩], [⟨.program ⟨257⟩, ⟨7281⟩⟩]⟩, (-1)⟩)

def event48649 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨14005⟩⟩, .operator (⟨48640, 0⟩, ⟨19114, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩]⟩, (1)⟩)

def exact48650RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨14001⟩⟩], [⟨.program ⟨257⟩, ⟨7281⟩⟩]⟩, (-1)⟩]

theorem exact48650RawTermsValid :
    exact48650RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48650 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14005⟩⟩) exact48650RawTerms .large 48643 (.finite 279172874240) (some (48645))

def event48651 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37313⟩⟩) 0 ⟨14005⟩ 48650

def event48652 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37313⟩⟩) 1 ⟨37312⟩ 48620

def event48653 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37313⟩⟩) (.sum [.predecessor 0 48651 .coefficient, .predecessor 1 48652 .coefficient])

def event48654 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨37313⟩⟩, .operator (⟨48650, 1⟩, ⟨48620, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨14001⟩⟩], [⟨.program ⟨257⟩, ⟨7281⟩⟩]⟩, (1)⟩)

def event48655 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37313⟩⟩) (.sum [.result 48650 .summary, .result 48620 .summary])

def exact48656RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨14001⟩⟩, ⟨.program ⟨257⟩, ⟨37306⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact48656RawTermsValid :
    exact48656RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48656 : Event := .resultExact (⟨.program ⟨257⟩, ⟨37313⟩⟩) exact48656RawTerms .large 48653 (.finite 279208656896) (some (48655))

def event48657 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39028⟩⟩) 0 ⟨37313⟩ 48656

def event48658 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39028⟩⟩) 1 ⟨39027⟩ 48592

def event48659 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39028⟩⟩) (.product (.predecessor 0 48657 .coefficient) (.predecessor 1 48658 .coefficient) (⟨false, false, none, none, none⟩))

def event48660 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39028⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨39027⟩⟩]⟩) [⟨.result 48592 .coefficient, false, none⟩])

def event48661 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39028⟩⟩) (.product (.result 48656 .summary) (.transfer 48660) (⟨false, false, none, none, none⟩))

def event48662 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨39028⟩⟩, .operator (⟨48656, 1⟩, ⟨48592, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨14001⟩⟩, ⟨.program ⟨257⟩, ⟨37306⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨39027⟩⟩]⟩, (-1)⟩)

def event48663 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨39028⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨14001⟩⟩, ⟨.program ⟨257⟩, ⟨37306⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨39027⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨39027⟩⟩) ⟨38477⟩ 48589)

def event48664 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨39028⟩⟩, .relation 48663 0, ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨14001⟩⟩, ⟨.program ⟨257⟩, ⟨37306⟩⟩], [⟨.program ⟨257⟩, ⟨38477⟩⟩]⟩, (-1)⟩)

def event48665 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨39028⟩⟩, .operator (⟨48656, 0⟩, ⟨48592, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨39027⟩⟩]⟩, (1)⟩)

def exact48666RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨39027⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨14001⟩⟩, ⟨.program ⟨257⟩, ⟨37306⟩⟩], [⟨.program ⟨257⟩, ⟨38477⟩⟩]⟩, (-1)⟩]

theorem exact48666RawTermsValid :
    exact48666RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48666 : Event := .resultExact (⟨.program ⟨257⟩, ⟨39028⟩⟩) exact48666RawTerms .large 48659 (.finite 2997980125321012183040) (some (48661))

def event48667 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37949⟩⟩) 0 ⟨37308⟩ 1693

def event48668 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37949⟩⟩) (.authority (.relationPreimageSource ⟨50⟩))

def exact48669RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨37949⟩⟩]⟩, (1)⟩]

theorem exact48669RawTermsValid :
    exact48669RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48669 : Event := .resultExact (⟨.program ⟨257⟩, ⟨37949⟩⟩) exact48669RawTerms (.finite 5647228698) 48668 .exactZero (none)

def event48670 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37951⟩⟩) 0 ⟨37949⟩ 48669

def event48671 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37951⟩⟩) 1 ⟨2370⟩ 4

def event48672 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37951⟩⟩) (.scale (.predecessor 0 48670 .coefficient) (.value (.predecessor 1 48671 .coefficient)))

def exact48673RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨37949⟩⟩]⟩, (1)⟩]

theorem exact48673RawTermsValid :
    exact48673RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48673 : Event := .resultExact (⟨.program ⟨257⟩, ⟨37951⟩⟩) exact48673RawTerms (.finite 5647228698) 48672 .exactZero (none)

def event48674 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37952⟩⟩) 0 ⟨11216⟩ 46745

def event48675 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37952⟩⟩) 1 ⟨37951⟩ 48673

def event48676 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37952⟩⟩) (.product (.predecessor 0 48674 .coefficient) (.predecessor 1 48675 .coefficient) (⟨false, false, none, none, none⟩))

def event48677 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37952⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨37949⟩⟩]⟩) [⟨.result 48669 .coefficient, false, none⟩])

def event48678 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37952⟩⟩) (.product (.result 46745 .summary) (.transfer 48677) (⟨false, false, none, none, none⟩))

def event48679 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨37952⟩⟩, .operator (⟨46745, 0⟩, ⟨48673, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37949⟩⟩]⟩, (1)⟩)

def event48680 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨37950⟩⟩)

def event48681 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event48682 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event48683 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨11115⟩⟩) (.authority (.operator))

def event48684 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨11115⟩⟩) (.finite 17)

def event48685 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event48686 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event48687 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event48688 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event48689 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 48688

def event48690 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 48686

def event48691 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 48689 .coefficient) (.value (.predecessor 1 48690 .coefficient)))

def event48692 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event48693 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11117⟩⟩) 0 ⟨392⟩ 48692

def event48694 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11117⟩⟩) 1 ⟨11115⟩ 48684

def event48695 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨11117⟩⟩) (.sum [.predecessor 0 48693 .coefficient, .predecessor 1 48694 .coefficient])

def event48696 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨11117⟩⟩) (.finite 655357)

def event48697 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11173⟩⟩) 0 ⟨11117⟩ 48696

def event48698 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11173⟩⟩) 1 ⟨5426⟩ 48682

def event48699 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨11173⟩⟩) (.identity (.predecessor 1 48698 .coefficient))

def event48700 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨11173⟩⟩) (.finite 655360)

def event48701 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37306⟩⟩) 0 ⟨11173⟩ 48700

def event48702 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37306⟩⟩) (.authority (.programFamilyFact))

def exact48703RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨37306⟩⟩], []⟩, (1)⟩]

theorem exact48703RawTermsValid :
    exact48703RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48703 : Event := .resultExact (⟨.program ⟨257⟩, ⟨37306⟩⟩) exact48703RawTerms (.finite 42) 48702 .exactZero (none)

def event48704 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14001⟩⟩) 0 ⟨11173⟩ 48700

def event48705 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14001⟩⟩) (.authority (.programFamilyFact))

def exact48706RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14001⟩⟩], []⟩, (1)⟩]

theorem exact48706RawTermsValid :
    exact48706RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48706 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14001⟩⟩) exact48706RawTerms (.finite 42) 48705 .exactZero (none)

def event48707 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37307⟩⟩) 0 ⟨14001⟩ 48706

def event48708 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37307⟩⟩) 1 ⟨37306⟩ 48703

def event48709 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37307⟩⟩) (.product (.predecessor 0 48707 .coefficient) (.predecessor 1 48708 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event48710 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37307⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨14001⟩⟩, ⟨.program ⟨257⟩, ⟨37306⟩⟩], []⟩) [⟨.result 48706 .coefficient, true, some 1⟩, ⟨.result 48703 .coefficient, true, some 1⟩])

def event48711 : Event := .survivorFold (1) 48710

def exact48712RawTerms : List Term := []

theorem exact48712RawTermsValid :
    exact48712RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48712 : Event := .resultExact (⟨.program ⟨257⟩, ⟨37307⟩⟩) exact48712RawTerms (.finite 1764) 48709 (.finite 1764) (some (48710))

def event48713 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37308⟩⟩) 0 ⟨37307⟩ 48712

def event48714 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37308⟩⟩) (.identity (.predecessor 0 48713 .coefficient))

def event48715 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨37308⟩⟩) (.finite 1764)

def event48716 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37949⟩⟩) 0 ⟨37308⟩ 48715

def event48717 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37949⟩⟩) (.authority (.relationPreimageSource ⟨50⟩))

def exact48718RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨37949⟩⟩]⟩, (1)⟩]

theorem exact48718RawTermsValid :
    exact48718RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48718 : Event := .resultExact (⟨.program ⟨257⟩, ⟨37949⟩⟩) exact48718RawTerms (.finite 5647228698) 48717 .exactZero (none)

def event48719 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35⟩⟩) (.authority (.operator))

def exact48720RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact48720RawTermsValid :
    exact48720RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48720 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35⟩⟩) exact48720RawTerms .large 48719 .exactZero (none)

def event48721 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37950⟩⟩) 0 ⟨35⟩ 48720

def event48722 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37950⟩⟩) 1 ⟨37949⟩ 48718

def event48723 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37950⟩⟩) (.product (.predecessor 0 48721 .coefficient) (.predecessor 1 48722 .coefficient) (⟨false, false, none, none, none⟩))

def event48724 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨37950⟩⟩, .operator (⟨48720, 0⟩, ⟨48718, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37949⟩⟩]⟩, (1)⟩)

def exact48725RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37949⟩⟩]⟩, (1)⟩]

theorem exact48725RawTermsValid :
    exact48725RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48725 : Event := .resultExact (⟨.program ⟨257⟩, ⟨37950⟩⟩) exact48725RawTerms .large 48723 .exactZero (none)

def event48726 : Event := .preFoldPolynomial 48725 [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37949⟩⟩]⟩, (1)⟩] .exactZero none

def exact48727RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37949⟩⟩]⟩, (1)⟩]

def event48727 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨37950⟩⟩) 48726 exact48727RawTerms .large 48723 .exactZero (none)

def event48728 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨39031⟩⟩)

def event48729 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event48730 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event48731 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨11115⟩⟩) (.authority (.operator))

def event48732 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨11115⟩⟩) (.finite 17)

def event48733 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event48734 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event48735 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event48736 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event48737 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 48736

def event48738 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 48734

def event48739 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 48737 .coefficient) (.value (.predecessor 1 48738 .coefficient)))

def event48740 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event48741 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11117⟩⟩) 0 ⟨392⟩ 48740

def event48742 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11117⟩⟩) 1 ⟨11115⟩ 48732

def event48743 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨11117⟩⟩) (.sum [.predecessor 0 48741 .coefficient, .predecessor 1 48742 .coefficient])

def event48744 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨11117⟩⟩) (.finite 655357)

def event48745 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11173⟩⟩) 0 ⟨11117⟩ 48744

def event48746 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11173⟩⟩) 1 ⟨5426⟩ 48730

def event48747 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨11173⟩⟩) (.identity (.predecessor 1 48746 .coefficient))

def event48748 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨11173⟩⟩) (.finite 655360)

def event48749 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37306⟩⟩) 0 ⟨11173⟩ 48748

def event48750 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37306⟩⟩) (.authority (.programFamilyFact))

def exact48751RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨37306⟩⟩], []⟩, (1)⟩]

theorem exact48751RawTermsValid :
    exact48751RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48751 : Event := .resultExact (⟨.program ⟨257⟩, ⟨37306⟩⟩) exact48751RawTerms (.finite 42) 48750 .exactZero (none)

def event48752 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14001⟩⟩) 0 ⟨11173⟩ 48748

def event48753 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14001⟩⟩) (.authority (.programFamilyFact))

def exact48754RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14001⟩⟩], []⟩, (1)⟩]

theorem exact48754RawTermsValid :
    exact48754RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48754 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14001⟩⟩) exact48754RawTerms (.finite 42) 48753 .exactZero (none)

def event48755 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37307⟩⟩) 0 ⟨14001⟩ 48754

def event48756 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37307⟩⟩) 1 ⟨37306⟩ 48751

def event48757 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37307⟩⟩) (.product (.predecessor 0 48755 .coefficient) (.predecessor 1 48756 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event48758 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨37307⟩⟩, .operator (⟨48754, 0⟩, ⟨48751, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨14001⟩⟩, ⟨.program ⟨257⟩, ⟨37306⟩⟩], []⟩, (1)⟩)

def exact48759RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14001⟩⟩, ⟨.program ⟨257⟩, ⟨37306⟩⟩], []⟩, (1)⟩]

theorem exact48759RawTermsValid :
    exact48759RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48759 : Event := .resultExact (⟨.program ⟨257⟩, ⟨37307⟩⟩) exact48759RawTerms (.finite 1764) 48757 .exactZero (none)

def event48760 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37308⟩⟩) 0 ⟨37307⟩ 48759

def event48761 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37308⟩⟩) (.identity (.predecessor 0 48760 .coefficient))

def event48762 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨37308⟩⟩) (.finite 1764)

def event48763 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38476⟩⟩) 0 ⟨37308⟩ 48762

def event48764 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨38476⟩⟩) (.authority (.programFamilyFact))

def event48765 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨38476⟩⟩) (.finite 3720)

def event48766 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨7177⟩⟩) .missing

def event48767 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38477⟩⟩) 0 ⟨7177⟩ 48766

def event48768 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38477⟩⟩) 1 ⟨38476⟩ 48765

def event48769 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨38477⟩⟩) (.authority (.operator))

def exact48770RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨38477⟩⟩]⟩, (1)⟩]

theorem exact48770RawTermsValid :
    exact48770RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48770 : Event := .resultExact (⟨.program ⟨257⟩, ⟨38477⟩⟩) exact48770RawTerms .large 48769 .exactZero (none)

def event48771 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39027⟩⟩) 0 ⟨38477⟩ 48770

def event48772 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39027⟩⟩) (.authority (.operator))

def exact48773RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨39027⟩⟩]⟩, (1)⟩]

theorem exact48773RawTermsValid :
    exact48773RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48773 : Event := .resultExact (⟨.program ⟨257⟩, ⟨39027⟩⟩) exact48773RawTerms (.finite 8192) 48772 .exactZero (none)

def event48774 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event48775 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event48776 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38738⟩⟩) 0 ⟨37308⟩ 48762

def event48777 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38738⟩⟩) 1 ⟨136⟩ 48775

def event48778 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨38738⟩⟩) (.sum [.predecessor 0 48776 .coefficient, .predecessor 1 48777 .coefficient])

def event48779 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨38738⟩⟩) (.finite 1764)

def event48780 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38739⟩⟩) 0 ⟨38738⟩ 48779

def event48781 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨38739⟩⟩) (.identity (.predecessor 0 48780 .coefficient))

def exact48782RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14001⟩⟩, ⟨.program ⟨257⟩, ⟨37306⟩⟩], []⟩, (1)⟩]

theorem exact48782RawTermsValid :
    exact48782RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48782 : Event := .resultExact (⟨.program ⟨257⟩, ⟨38739⟩⟩) exact48782RawTerms (.finite 1764) 48781 .exactZero (none)

def event48783 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact48784RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact48784RawTermsValid :
    exact48784RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48784 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact48784RawTerms .large 48783 .exactZero (none)

def event48785 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38740⟩⟩) 0 ⟨6908⟩ 48784

def event48786 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38740⟩⟩) 1 ⟨38739⟩ 48782

def event48787 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨38740⟩⟩) (.product (.predecessor 0 48785 .coefficient) (.predecessor 1 48786 .coefficient) (⟨false, false, none, none, none⟩))

def event48788 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨38740⟩⟩, .operator (⟨48784, 0⟩, ⟨48782, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨14001⟩⟩, ⟨.program ⟨257⟩, ⟨37306⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact48789RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14001⟩⟩, ⟨.program ⟨257⟩, ⟨37306⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact48789RawTermsValid :
    exact48789RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48789 : Event := .resultExact (⟨.program ⟨257⟩, ⟨38740⟩⟩) exact48789RawTerms .large 48787 .exactZero (none)

def event48790 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨2370⟩⟩) (.authority (.operator))

def event48791 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨2370⟩⟩) (.finite 1)

def event48792 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7178⟩⟩) 0 ⟨7177⟩ 48766

def event48793 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7178⟩⟩) (.authority (.operator))

def exact48794RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7178⟩⟩]⟩, (1)⟩]

theorem exact48794RawTermsValid :
    exact48794RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48794 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7178⟩⟩) exact48794RawTerms .large 48793 .exactZero (none)

def event48795 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7281⟩⟩) 0 ⟨7178⟩ 48794

def event48796 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7281⟩⟩) (.identity (.predecessor 0 48795 .coefficient))

def exact48797RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7281⟩⟩]⟩, (1)⟩]

theorem exact48797RawTermsValid :
    exact48797RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48797 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7281⟩⟩) exact48797RawTerms .large 48796 .exactZero (none)

def event48798 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9553⟩⟩) 0 ⟨7281⟩ 48797

def event48799 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9553⟩⟩) (.authority (.operator))

def exact48800RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨9553⟩⟩]⟩, (1)⟩]

theorem exact48800RawTermsValid :
    exact48800RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48800 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9553⟩⟩) exact48800RawTerms (.finite 8192) 48799 .exactZero (none)

def event48801 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9554⟩⟩) 0 ⟨9553⟩ 48800

def event48802 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9554⟩⟩) 1 ⟨2370⟩ 48791

def event48803 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9554⟩⟩) (.scale (.predecessor 0 48801 .coefficient) (.value (.predecessor 1 48802 .coefficient)))

def exact48804RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨9553⟩⟩]⟩, (1)⟩]

theorem exact48804RawTermsValid :
    exact48804RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48804 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9554⟩⟩) exact48804RawTerms (.finite 8192) 48803 .exactZero (none)

def event48805 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7298⟩⟩) 0 ⟨7178⟩ 48794

def event48806 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7298⟩⟩) (.identity (.predecessor 0 48805 .coefficient))

def exact48807RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7298⟩⟩]⟩, (1)⟩]

theorem exact48807RawTermsValid :
    exact48807RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48807 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7298⟩⟩) exact48807RawTerms .large 48806 .exactZero (none)

def event48808 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9555⟩⟩) 0 ⟨7298⟩ 48807

def event48809 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9555⟩⟩) 1 ⟨9554⟩ 48804

def event48810 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9555⟩⟩) (.product (.predecessor 0 48808 .coefficient) (.predecessor 1 48809 .coefficient) (⟨false, false, none, none, none⟩))

def event48811 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨9555⟩⟩, .operator (⟨48807, 0⟩, ⟨48804, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩]⟩, (1)⟩)

def exact48812RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩]⟩, (1)⟩]

theorem exact48812RawTermsValid :
    exact48812RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48812 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9555⟩⟩) exact48812RawTerms .large 48810 .exactZero (none)

def event48813 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38741⟩⟩) 0 ⟨9555⟩ 48812

def event48814 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38741⟩⟩) 1 ⟨38740⟩ 48789

def event48815 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨38741⟩⟩) (.sum [.predecessor 0 48813 .coefficient, .predecessor 1 48814 .coefficient])

def exact48816RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨14001⟩⟩, ⟨.program ⟨257⟩, ⟨37306⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact48816RawTermsValid :
    exact48816RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48816 : Event := .resultExact (⟨.program ⟨257⟩, ⟨38741⟩⟩) exact48816RawTerms .large 48815 .exactZero (none)

def event48817 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39030⟩⟩) 0 ⟨38741⟩ 48816

def event48818 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39030⟩⟩) 1 ⟨39027⟩ 48773

def event48819 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39030⟩⟩) (.product (.predecessor 0 48817 .coefficient) (.predecessor 1 48818 .coefficient) (⟨false, false, none, none, none⟩))

def event48820 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨39030⟩⟩, .operator (⟨48816, 0⟩, ⟨48773, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨39027⟩⟩]⟩, (1)⟩)

def event48821 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨39030⟩⟩, .operator (⟨48816, 1⟩, ⟨48773, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨14001⟩⟩, ⟨.program ⟨257⟩, ⟨37306⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨39027⟩⟩]⟩, (-1)⟩)

def event48822 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨39030⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨14001⟩⟩, ⟨.program ⟨257⟩, ⟨37306⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨39027⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨39027⟩⟩) ⟨38477⟩ 48770)

def event48823 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨39030⟩⟩, .relation 48822 0, ⟨[⟨.program ⟨257⟩, ⟨14001⟩⟩, ⟨.program ⟨257⟩, ⟨37306⟩⟩], [⟨.program ⟨257⟩, ⟨38477⟩⟩]⟩, (-1)⟩)

def exact48824RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨39027⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨14001⟩⟩, ⟨.program ⟨257⟩, ⟨37306⟩⟩], [⟨.program ⟨257⟩, ⟨38477⟩⟩]⟩, (-1)⟩]

theorem exact48824RawTermsValid :
    exact48824RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48824 : Event := .resultExact (⟨.program ⟨257⟩, ⟨39030⟩⟩) exact48824RawTerms .large 48819 .exactZero (none)

def event48825 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37492⟩⟩) 0 ⟨37308⟩ 48762

def event48826 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37492⟩⟩) (.authority (.programFamilyFact))

def exact48827RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨37492⟩⟩], []⟩, (1)⟩]

theorem exact48827RawTermsValid :
    exact48827RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48827 : Event := .resultExact (⟨.program ⟨257⟩, ⟨37492⟩⟩) exact48827RawTerms (.finite 42) 48826 .exactZero (none)

def event48828 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37494⟩⟩) 0 ⟨6908⟩ 48784

def event48829 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37494⟩⟩) 1 ⟨37492⟩ 48827

def event48830 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37494⟩⟩) (.product (.predecessor 0 48828 .coefficient) (.predecessor 1 48829 .coefficient) (⟨false, true, none, none, some 1⟩))

def event48831 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨37494⟩⟩, .operator (⟨48784, 0⟩, ⟨48827, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨37492⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact48832RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨37492⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact48832RawTermsValid :
    exact48832RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48832 : Event := .resultExact (⟨.program ⟨257⟩, ⟨37494⟩⟩) exact48832RawTerms .large 48830 .exactZero (none)

def event48833 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7192⟩⟩) 0 ⟨7177⟩ 48766

def event48834 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7192⟩⟩) (.authority (.operator))

def exact48835RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7192⟩⟩]⟩, (1)⟩]

theorem exact48835RawTermsValid :
    exact48835RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48835 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7192⟩⟩) exact48835RawTerms .large 48834 .exactZero (none)

def event48836 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37495⟩⟩) 0 ⟨7192⟩ 48835

def event48837 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37495⟩⟩) 1 ⟨37494⟩ 48832

def event48838 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37495⟩⟩) (.sum [.predecessor 0 48836 .coefficient, .predecessor 1 48837 .coefficient])

def exact48839RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7192⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨37492⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact48839RawTermsValid :
    exact48839RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48839 : Event := .resultExact (⟨.program ⟨257⟩, ⟨37495⟩⟩) exact48839RawTerms .large 48838 .exactZero (none)

def event48840 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39031⟩⟩) 0 ⟨37495⟩ 48839

def event48841 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39031⟩⟩) 1 ⟨39030⟩ 48824

def event48842 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39031⟩⟩) (.sum [.predecessor 0 48840 .coefficient, .predecessor 1 48841 .coefficient])

def exact48843RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7192⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨39027⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨14001⟩⟩, ⟨.program ⟨257⟩, ⟨37306⟩⟩], [⟨.program ⟨257⟩, ⟨38477⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨37492⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact48843RawTermsValid :
    exact48843RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48843 : Event := .resultExact (⟨.program ⟨257⟩, ⟨39031⟩⟩) exact48843RawTerms .large 48842 .exactZero (none)

def event48844 : Event := .preFoldPolynomial 48843 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7192⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨39027⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨14001⟩⟩, ⟨.program ⟨257⟩, ⟨37306⟩⟩], [⟨.program ⟨257⟩, ⟨38477⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨37492⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact48845RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7192⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨39027⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨14001⟩⟩, ⟨.program ⟨257⟩, ⟨37306⟩⟩], [⟨.program ⟨257⟩, ⟨38477⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨37492⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event48845 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨39031⟩⟩) 48844 exact48845RawTerms .large 48842 .exactZero (none)

def event48846 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨37308⟩⟩) ⟨⟨71⟩, ⟨50⟩, ⟨135⟩⟩ ⟨48680, 48846⟩

def event48847 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨37952⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37949⟩⟩]⟩) (1) 0 2 (.universal 48846 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37949⟩⟩]⟩) (none) 48845)

def event48848 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨37952⟩⟩, .relation 48847 0, ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7192⟩⟩]⟩, (1)⟩)

def event48849 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨37952⟩⟩, .relation 48847 1, ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨39027⟩⟩]⟩, (-1)⟩)

def event48850 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨37952⟩⟩, .relation 48847 2, ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨14001⟩⟩, ⟨.program ⟨257⟩, ⟨37306⟩⟩], [⟨.program ⟨257⟩, ⟨38477⟩⟩]⟩, (1)⟩)

def event48851 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨37952⟩⟩, .relation 48847 3, ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨37492⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact48852RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7192⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨39027⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨14001⟩⟩, ⟨.program ⟨257⟩, ⟨37306⟩⟩], [⟨.program ⟨257⟩, ⟨38477⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨37492⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact48852RawTermsValid :
    exact48852RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48852 : Event := .resultExact (⟨.program ⟨257⟩, ⟨37952⟩⟩) exact48852RawTerms .large 48676 (.finite 202072841853861888) (some (48678))

def event48853 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39029⟩⟩) 0 ⟨37952⟩ 48852

def event48854 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39029⟩⟩) 1 ⟨39028⟩ 48666

def event48855 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39029⟩⟩) (.sum [.predecessor 0 48853 .coefficient, .predecessor 1 48854 .coefficient])

def event48856 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨39029⟩⟩, .operator (⟨48852, 2⟩, ⟨48666, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨14001⟩⟩, ⟨.program ⟨257⟩, ⟨37306⟩⟩], [⟨.program ⟨257⟩, ⟨38477⟩⟩]⟩, (-1)⟩)

def event48857 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨39029⟩⟩, .operator (⟨48852, 1⟩, ⟨48666, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨39027⟩⟩]⟩, (1)⟩)

def event48858 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39029⟩⟩) (.sum [.result 48852 .summary, .result 48666 .summary])

def exact48859RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7192⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨37492⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact48859RawTermsValid :
    exact48859RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48859 : Event := .resultExact (⟨.program ⟨257⟩, ⟨39029⟩⟩) exact48859RawTerms .large 48855 (.finite 2998182198162866044928) (some (48858))

def event48860 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39511⟩⟩) 0 ⟨39029⟩ 48859

def event48861 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39511⟩⟩) 1 ⟨39509⟩ 48582

def event48862 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39511⟩⟩) (.product (.predecessor 0 48860 .coefficient) (.predecessor 1 48861 .coefficient) (⟨false, false, none, none, none⟩))

def event48863 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39511⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨39509⟩⟩]⟩) [⟨.result 48582 .coefficient, false, none⟩])

def event48864 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39511⟩⟩) (.product (.result 48859 .summary) (.transfer 48863) (⟨false, false, none, none, none⟩))

def event48865 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨39511⟩⟩, .operator (⟨48859, 0⟩, ⟨48582, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7192⟩⟩, ⟨.program ⟨257⟩, ⟨39509⟩⟩]⟩, (1)⟩)

def event48866 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨39511⟩⟩, .operator (⟨48859, 1⟩, ⟨48582, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨37492⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨39509⟩⟩]⟩, (-1)⟩)

def event48867 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨39511⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨37492⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨39509⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨39509⟩⟩) ⟨38653⟩ 48579)

def event48868 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨39511⟩⟩, .relation 48867 0, ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨37492⟩⟩], [⟨.program ⟨257⟩, ⟨38653⟩⟩]⟩, (-1)⟩)

def exact48869RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7192⟩⟩, ⟨.program ⟨257⟩, ⟨39509⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨37492⟩⟩], [⟨.program ⟨257⟩, ⟨38653⟩⟩]⟩, (-1)⟩]

theorem exact48869RawTermsValid :
    exact48869RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48869 : Event := .resultExact (⟨.program ⟨257⟩, ⟨39511⟩⟩) exact48869RawTerms .large 48862 (.finite 32192736221397252361486566686720) (some (48864))

def event48870 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38336⟩⟩) 0 ⟨37493⟩ 1699

def event48871 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨38336⟩⟩) (.authority (.relationPreimageSource ⟨85⟩))

def exact48872RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨38336⟩⟩]⟩, (1)⟩]

theorem exact48872RawTermsValid :
    exact48872RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48872 : Event := .resultExact (⟨.program ⟨257⟩, ⟨38336⟩⟩) exact48872RawTerms (.finite 5647228698) 48871 .exactZero (none)

def event48873 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38338⟩⟩) 0 ⟨38336⟩ 48872

def event48874 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38338⟩⟩) 1 ⟨2370⟩ 4

def event48875 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨38338⟩⟩) (.scale (.predecessor 0 48873 .coefficient) (.value (.predecessor 1 48874 .coefficient)))

def exact48876RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨38336⟩⟩]⟩, (1)⟩]

theorem exact48876RawTermsValid :
    exact48876RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48876 : Event := .resultExact (⟨.program ⟨257⟩, ⟨38338⟩⟩) exact48876RawTerms (.finite 5647228698) 48875 .exactZero (none)

def event48877 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38339⟩⟩) 0 ⟨11216⟩ 46745

def event48878 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38339⟩⟩) 1 ⟨38338⟩ 48876

def event48879 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨38339⟩⟩) (.product (.predecessor 0 48877 .coefficient) (.predecessor 1 48878 .coefficient) (⟨false, false, none, none, none⟩))

def event48880 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨38339⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨38336⟩⟩]⟩) [⟨.result 48872 .coefficient, false, none⟩])

def event48881 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨38339⟩⟩) (.product (.result 46745 .summary) (.transfer 48880) (⟨false, false, none, none, none⟩))

def event48882 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨38339⟩⟩, .operator (⟨46745, 0⟩, ⟨48876, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨38336⟩⟩]⟩, (1)⟩)

def event48883 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨38337⟩⟩)

def event48884 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event48885 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event48886 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨11115⟩⟩) (.authority (.operator))

def event48887 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨11115⟩⟩) (.finite 17)

def event48888 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event48889 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event48890 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event48891 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event48892 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 48891

def event48893 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 48889

def event48894 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 48892 .coefficient) (.value (.predecessor 1 48893 .coefficient)))

def event48895 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def eventLeaf3040 : Array AnnotatedEvent := #[
  { event := event48640
    frameStart := 0 },
  { event := event48641
    frameStart := 0 },
  { event := event48642
    frameStart := 0 },
  { event := event48643
    frameStart := 0 },
  { event := event48644
    frameStart := 0 },
  { event := event48645
    frameStart := 0 },
  { event := event48646
    frameStart := 0 },
  { event := event48647
    frameStart := 0 },
  { event := event48648
    frameStart := 0 },
  { event := event48649
    frameStart := 0 },
  { event := event48650
    frameStart := 0 },
  { event := event48651
    frameStart := 0 },
  { event := event48652
    frameStart := 0 },
  { event := event48653
    frameStart := 0 },
  { event := event48654
    frameStart := 0 },
  { event := event48655
    frameStart := 0 }
]

def eventLeaf3041 : Array AnnotatedEvent := #[
  { event := event48656
    frameStart := 0 },
  { event := event48657
    frameStart := 0 },
  { event := event48658
    frameStart := 0 },
  { event := event48659
    frameStart := 0 },
  { event := event48660
    frameStart := 0 },
  { event := event48661
    frameStart := 0 },
  { event := event48662
    frameStart := 0 },
  { event := event48663
    frameStart := 0 },
  { event := event48664
    frameStart := 0 },
  { event := event48665
    frameStart := 0 },
  { event := event48666
    frameStart := 0 },
  { event := event48667
    frameStart := 0 },
  { event := event48668
    frameStart := 0 },
  { event := event48669
    frameStart := 0 },
  { event := event48670
    frameStart := 0 },
  { event := event48671
    frameStart := 0 }
]

def eventLeaf3042 : Array AnnotatedEvent := #[
  { event := event48672
    frameStart := 0 },
  { event := event48673
    frameStart := 0 },
  { event := event48674
    frameStart := 0 },
  { event := event48675
    frameStart := 0 },
  { event := event48676
    frameStart := 0 },
  { event := event48677
    frameStart := 0 },
  { event := event48678
    frameStart := 0 },
  { event := event48679
    frameStart := 0 },
  { event := event48680
    frameStart := 48680 },
  { event := event48681
    frameStart := 48680 },
  { event := event48682
    frameStart := 48680 },
  { event := event48683
    frameStart := 48680 },
  { event := event48684
    frameStart := 48680 },
  { event := event48685
    frameStart := 48680 },
  { event := event48686
    frameStart := 48680 },
  { event := event48687
    frameStart := 48680 }
]

def eventLeaf3043 : Array AnnotatedEvent := #[
  { event := event48688
    frameStart := 48680 },
  { event := event48689
    frameStart := 48680 },
  { event := event48690
    frameStart := 48680 },
  { event := event48691
    frameStart := 48680 },
  { event := event48692
    frameStart := 48680 },
  { event := event48693
    frameStart := 48680 },
  { event := event48694
    frameStart := 48680 },
  { event := event48695
    frameStart := 48680 },
  { event := event48696
    frameStart := 48680 },
  { event := event48697
    frameStart := 48680 },
  { event := event48698
    frameStart := 48680 },
  { event := event48699
    frameStart := 48680 },
  { event := event48700
    frameStart := 48680 },
  { event := event48701
    frameStart := 48680 },
  { event := event48702
    frameStart := 48680 },
  { event := event48703
    frameStart := 48680 }
]

def eventLeaf3044 : Array AnnotatedEvent := #[
  { event := event48704
    frameStart := 48680 },
  { event := event48705
    frameStart := 48680 },
  { event := event48706
    frameStart := 48680 },
  { event := event48707
    frameStart := 48680 },
  { event := event48708
    frameStart := 48680 },
  { event := event48709
    frameStart := 48680 },
  { event := event48710
    frameStart := 48680 },
  { event := event48711
    frameStart := 48680 },
  { event := event48712
    frameStart := 48680 },
  { event := event48713
    frameStart := 48680 },
  { event := event48714
    frameStart := 48680 },
  { event := event48715
    frameStart := 48680 },
  { event := event48716
    frameStart := 48680 },
  { event := event48717
    frameStart := 48680 },
  { event := event48718
    frameStart := 48680 },
  { event := event48719
    frameStart := 48680 }
]

def eventLeaf3045 : Array AnnotatedEvent := #[
  { event := event48720
    frameStart := 48680 },
  { event := event48721
    frameStart := 48680 },
  { event := event48722
    frameStart := 48680 },
  { event := event48723
    frameStart := 48680 },
  { event := event48724
    frameStart := 48680 },
  { event := event48725
    frameStart := 48680 },
  { event := event48726
    frameStart := 48680 },
  { event := event48727
    frameStart := 48680 },
  { event := event48728
    frameStart := 48728 },
  { event := event48729
    frameStart := 48728 },
  { event := event48730
    frameStart := 48728 },
  { event := event48731
    frameStart := 48728 },
  { event := event48732
    frameStart := 48728 },
  { event := event48733
    frameStart := 48728 },
  { event := event48734
    frameStart := 48728 },
  { event := event48735
    frameStart := 48728 }
]

def eventLeaf3046 : Array AnnotatedEvent := #[
  { event := event48736
    frameStart := 48728 },
  { event := event48737
    frameStart := 48728 },
  { event := event48738
    frameStart := 48728 },
  { event := event48739
    frameStart := 48728 },
  { event := event48740
    frameStart := 48728 },
  { event := event48741
    frameStart := 48728 },
  { event := event48742
    frameStart := 48728 },
  { event := event48743
    frameStart := 48728 },
  { event := event48744
    frameStart := 48728 },
  { event := event48745
    frameStart := 48728 },
  { event := event48746
    frameStart := 48728 },
  { event := event48747
    frameStart := 48728 },
  { event := event48748
    frameStart := 48728 },
  { event := event48749
    frameStart := 48728 },
  { event := event48750
    frameStart := 48728 },
  { event := event48751
    frameStart := 48728 }
]

def eventLeaf3047 : Array AnnotatedEvent := #[
  { event := event48752
    frameStart := 48728 },
  { event := event48753
    frameStart := 48728 },
  { event := event48754
    frameStart := 48728 },
  { event := event48755
    frameStart := 48728 },
  { event := event48756
    frameStart := 48728 },
  { event := event48757
    frameStart := 48728 },
  { event := event48758
    frameStart := 48728 },
  { event := event48759
    frameStart := 48728 },
  { event := event48760
    frameStart := 48728 },
  { event := event48761
    frameStart := 48728 },
  { event := event48762
    frameStart := 48728 },
  { event := event48763
    frameStart := 48728 },
  { event := event48764
    frameStart := 48728 },
  { event := event48765
    frameStart := 48728 },
  { event := event48766
    frameStart := 48728 },
  { event := event48767
    frameStart := 48728 }
]

def eventLeaf3048 : Array AnnotatedEvent := #[
  { event := event48768
    frameStart := 48728 },
  { event := event48769
    frameStart := 48728 },
  { event := event48770
    frameStart := 48728 },
  { event := event48771
    frameStart := 48728 },
  { event := event48772
    frameStart := 48728 },
  { event := event48773
    frameStart := 48728 },
  { event := event48774
    frameStart := 48728 },
  { event := event48775
    frameStart := 48728 },
  { event := event48776
    frameStart := 48728 },
  { event := event48777
    frameStart := 48728 },
  { event := event48778
    frameStart := 48728 },
  { event := event48779
    frameStart := 48728 },
  { event := event48780
    frameStart := 48728 },
  { event := event48781
    frameStart := 48728 },
  { event := event48782
    frameStart := 48728 },
  { event := event48783
    frameStart := 48728 }
]

def eventLeaf3049 : Array AnnotatedEvent := #[
  { event := event48784
    frameStart := 48728 },
  { event := event48785
    frameStart := 48728 },
  { event := event48786
    frameStart := 48728 },
  { event := event48787
    frameStart := 48728 },
  { event := event48788
    frameStart := 48728 },
  { event := event48789
    frameStart := 48728 },
  { event := event48790
    frameStart := 48728 },
  { event := event48791
    frameStart := 48728 },
  { event := event48792
    frameStart := 48728 },
  { event := event48793
    frameStart := 48728 },
  { event := event48794
    frameStart := 48728 },
  { event := event48795
    frameStart := 48728 },
  { event := event48796
    frameStart := 48728 },
  { event := event48797
    frameStart := 48728 },
  { event := event48798
    frameStart := 48728 },
  { event := event48799
    frameStart := 48728 }
]

def eventLeaf3050 : Array AnnotatedEvent := #[
  { event := event48800
    frameStart := 48728 },
  { event := event48801
    frameStart := 48728 },
  { event := event48802
    frameStart := 48728 },
  { event := event48803
    frameStart := 48728 },
  { event := event48804
    frameStart := 48728 },
  { event := event48805
    frameStart := 48728 },
  { event := event48806
    frameStart := 48728 },
  { event := event48807
    frameStart := 48728 },
  { event := event48808
    frameStart := 48728 },
  { event := event48809
    frameStart := 48728 },
  { event := event48810
    frameStart := 48728 },
  { event := event48811
    frameStart := 48728 },
  { event := event48812
    frameStart := 48728 },
  { event := event48813
    frameStart := 48728 },
  { event := event48814
    frameStart := 48728 },
  { event := event48815
    frameStart := 48728 }
]

def eventLeaf3051 : Array AnnotatedEvent := #[
  { event := event48816
    frameStart := 48728 },
  { event := event48817
    frameStart := 48728 },
  { event := event48818
    frameStart := 48728 },
  { event := event48819
    frameStart := 48728 },
  { event := event48820
    frameStart := 48728 },
  { event := event48821
    frameStart := 48728 },
  { event := event48822
    frameStart := 48728 },
  { event := event48823
    frameStart := 48728 },
  { event := event48824
    frameStart := 48728 },
  { event := event48825
    frameStart := 48728 },
  { event := event48826
    frameStart := 48728 },
  { event := event48827
    frameStart := 48728 },
  { event := event48828
    frameStart := 48728 },
  { event := event48829
    frameStart := 48728 },
  { event := event48830
    frameStart := 48728 },
  { event := event48831
    frameStart := 48728 }
]

def eventLeaf3052 : Array AnnotatedEvent := #[
  { event := event48832
    frameStart := 48728 },
  { event := event48833
    frameStart := 48728 },
  { event := event48834
    frameStart := 48728 },
  { event := event48835
    frameStart := 48728 },
  { event := event48836
    frameStart := 48728 },
  { event := event48837
    frameStart := 48728 },
  { event := event48838
    frameStart := 48728 },
  { event := event48839
    frameStart := 48728 },
  { event := event48840
    frameStart := 48728 },
  { event := event48841
    frameStart := 48728 },
  { event := event48842
    frameStart := 48728 },
  { event := event48843
    frameStart := 48728 },
  { event := event48844
    frameStart := 48728 },
  { event := event48845
    frameStart := 48728 },
  { event := event48846
    frameStart := 0 },
  { event := event48847
    frameStart := 0 }
]

def eventLeaf3053 : Array AnnotatedEvent := #[
  { event := event48848
    frameStart := 0 },
  { event := event48849
    frameStart := 0 },
  { event := event48850
    frameStart := 0 },
  { event := event48851
    frameStart := 0 },
  { event := event48852
    frameStart := 0 },
  { event := event48853
    frameStart := 0 },
  { event := event48854
    frameStart := 0 },
  { event := event48855
    frameStart := 0 },
  { event := event48856
    frameStart := 0 },
  { event := event48857
    frameStart := 0 },
  { event := event48858
    frameStart := 0 },
  { event := event48859
    frameStart := 0 },
  { event := event48860
    frameStart := 0 },
  { event := event48861
    frameStart := 0 },
  { event := event48862
    frameStart := 0 },
  { event := event48863
    frameStart := 0 }
]

def eventLeaf3054 : Array AnnotatedEvent := #[
  { event := event48864
    frameStart := 0 },
  { event := event48865
    frameStart := 0 },
  { event := event48866
    frameStart := 0 },
  { event := event48867
    frameStart := 0 },
  { event := event48868
    frameStart := 0 },
  { event := event48869
    frameStart := 0 },
  { event := event48870
    frameStart := 0 },
  { event := event48871
    frameStart := 0 },
  { event := event48872
    frameStart := 0 },
  { event := event48873
    frameStart := 0 },
  { event := event48874
    frameStart := 0 },
  { event := event48875
    frameStart := 0 },
  { event := event48876
    frameStart := 0 },
  { event := event48877
    frameStart := 0 },
  { event := event48878
    frameStart := 0 },
  { event := event48879
    frameStart := 0 }
]

def eventLeaf3055 : Array AnnotatedEvent := #[
  { event := event48880
    frameStart := 0 },
  { event := event48881
    frameStart := 0 },
  { event := event48882
    frameStart := 0 },
  { event := event48883
    frameStart := 48883 },
  { event := event48884
    frameStart := 48883 },
  { event := event48885
    frameStart := 48883 },
  { event := event48886
    frameStart := 48883 },
  { event := event48887
    frameStart := 48883 },
  { event := event48888
    frameStart := 48883 },
  { event := event48889
    frameStart := 48883 },
  { event := event48890
    frameStart := 48883 },
  { event := event48891
    frameStart := 48883 },
  { event := event48892
    frameStart := 48883 },
  { event := event48893
    frameStart := 48883 },
  { event := event48894
    frameStart := 48883 },
  { event := event48895
    frameStart := 48883 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events190
