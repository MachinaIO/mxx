import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events366

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event93696 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨30656⟩⟩, .operator (⟨93691, 1⟩, ⟨93505, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩, ⟨.program ⟨257⟩, ⟨30654⟩⟩]⟩, (1)⟩)

def event93697 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨30656⟩⟩) (.sum [.result 93691 .summary, .result 93505 .summary])

def exact93698RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7190⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨29128⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact93698RawTermsValid :
    exact93698RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event93698 : Event := .resultExact (⟨.program ⟨257⟩, ⟨30656⟩⟩) exact93698RawTerms .large 93694 (.finite 2998127310542407467008) (some (93697))

def event93699 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31096⟩⟩) 0 ⟨30656⟩ 93698

def event93700 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31096⟩⟩) 1 ⟨31094⟩ 93421

def event93701 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31096⟩⟩) (.product (.predecessor 0 93699 .coefficient) (.predecessor 1 93700 .coefficient) (⟨false, false, none, none, none⟩))

def event93702 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31096⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨31094⟩⟩]⟩) [⟨.result 93421 .coefficient, false, none⟩])

def event93703 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31096⟩⟩) (.product (.result 93698 .summary) (.transfer 93702) (⟨false, false, none, none, none⟩))

def event93704 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨31096⟩⟩, .operator (⟨93698, 0⟩, ⟨93421, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨31094⟩⟩]⟩, (1)⟩)

def event93705 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨31096⟩⟩, .operator (⟨93698, 1⟩, ⟨93421, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨29128⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨31094⟩⟩]⟩, (-1)⟩)

def event93706 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨31096⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨29128⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨31094⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨31094⟩⟩) ⟨30286⟩ 93418)

def event93707 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨31096⟩⟩, .relation 93706 0, ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨29128⟩⟩], [⟨.program ⟨257⟩, ⟨30286⟩⟩]⟩, (-1)⟩)

def exact93708RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨31094⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨29128⟩⟩], [⟨.program ⟨257⟩, ⟨30286⟩⟩]⟩, (-1)⟩]

theorem exact93708RawTermsValid :
    exact93708RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event93708 : Event := .resultExact (⟨.program ⟨257⟩, ⟨31096⟩⟩) exact93708RawTerms .large 93701 (.finite 32192146870060190229763897425920) (some (93703))

def event93709 : Event := .predecessor (⟨.program ⟨257⟩, ⟨29936⟩⟩) 0 ⟨29129⟩ 3989

def event93710 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨29936⟩⟩) (.authority (.relationPreimageSource ⟨81⟩))

def exact93711RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨29936⟩⟩]⟩, (1)⟩]

theorem exact93711RawTermsValid :
    exact93711RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event93711 : Event := .resultExact (⟨.program ⟨257⟩, ⟨29936⟩⟩) exact93711RawTerms (.finite 5647228698) 93710 .exactZero (none)

def event93712 : Event := .predecessor (⟨.program ⟨257⟩, ⟨29938⟩⟩) 0 ⟨29936⟩ 93711

def event93713 : Event := .predecessor (⟨.program ⟨257⟩, ⟨29938⟩⟩) 1 ⟨2370⟩ 4

def event93714 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨29938⟩⟩) (.scale (.predecessor 0 93712 .coefficient) (.value (.predecessor 1 93713 .coefficient)))

def exact93715RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨29936⟩⟩]⟩, (1)⟩]

theorem exact93715RawTermsValid :
    exact93715RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event93715 : Event := .resultExact (⟨.program ⟨257⟩, ⟨29938⟩⟩) exact93715RawTerms (.finite 5647228698) 93714 .exactZero (none)

def event93716 : Event := .predecessor (⟨.program ⟨257⟩, ⟨29939⟩⟩) 0 ⟨9944⟩ 90620

def event93717 : Event := .predecessor (⟨.program ⟨257⟩, ⟨29939⟩⟩) 1 ⟨29938⟩ 93715

def event93718 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨29939⟩⟩) (.product (.predecessor 0 93716 .coefficient) (.predecessor 1 93717 .coefficient) (⟨false, false, none, none, none⟩))

def event93719 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨29939⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨29936⟩⟩]⟩) [⟨.result 93711 .coefficient, false, none⟩])

def event93720 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨29939⟩⟩) (.product (.result 90620 .summary) (.transfer 93719) (⟨false, false, none, none, none⟩))

def event93721 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨29939⟩⟩, .operator (⟨90620, 0⟩, ⟨93715, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29936⟩⟩]⟩, (1)⟩)

def event93722 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨29937⟩⟩)

def event93723 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event93724 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event93725 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9843⟩⟩) (.authority (.operator))

def event93726 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨9843⟩⟩) (.finite 14)

def event93727 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event93728 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event93729 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event93730 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event93731 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 93730

def event93732 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 93728

def event93733 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 93731 .coefficient) (.value (.predecessor 1 93732 .coefficient)))

def event93734 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event93735 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9845⟩⟩) 0 ⟨392⟩ 93734

def event93736 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9845⟩⟩) 1 ⟨9843⟩ 93726

def event93737 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9845⟩⟩) (.sum [.predecessor 0 93735 .coefficient, .predecessor 1 93736 .coefficient])

def event93738 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨9845⟩⟩) (.finite 655354)

def event93739 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9901⟩⟩) 0 ⟨9845⟩ 93738

def event93740 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9901⟩⟩) 1 ⟨5426⟩ 93724

def event93741 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9901⟩⟩) (.identity (.predecessor 1 93740 .coefficient))

def event93742 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨9901⟩⟩) (.finite 655360)

def event93743 : Event := .predecessor (⟨.program ⟨257⟩, ⟨28894⟩⟩) 0 ⟨9901⟩ 93742

def event93744 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨28894⟩⟩) (.authority (.programFamilyFact))

def exact93745RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨28894⟩⟩], []⟩, (1)⟩]

theorem exact93745RawTermsValid :
    exact93745RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event93745 : Event := .resultExact (⟨.program ⟨257⟩, ⟨28894⟩⟩) exact93745RawTerms (.finite 36) 93744 .exactZero (none)

def event93746 : Event := .predecessor (⟨.program ⟨257⟩, ⟨13356⟩⟩) 0 ⟨9901⟩ 93742

def event93747 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨13356⟩⟩) (.authority (.programFamilyFact))

def exact93748RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨13356⟩⟩], []⟩, (1)⟩]

theorem exact93748RawTermsValid :
    exact93748RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event93748 : Event := .resultExact (⟨.program ⟨257⟩, ⟨13356⟩⟩) exact93748RawTerms (.finite 36) 93747 .exactZero (none)

def event93749 : Event := .predecessor (⟨.program ⟨257⟩, ⟨28895⟩⟩) 0 ⟨13356⟩ 93748

def event93750 : Event := .predecessor (⟨.program ⟨257⟩, ⟨28895⟩⟩) 1 ⟨28894⟩ 93745

def event93751 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨28895⟩⟩) (.product (.predecessor 0 93749 .coefficient) (.predecessor 1 93750 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event93752 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨28895⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨13356⟩⟩, ⟨.program ⟨257⟩, ⟨28894⟩⟩], []⟩) [⟨.result 93748 .coefficient, true, some 1⟩, ⟨.result 93745 .coefficient, true, some 1⟩])

def event93753 : Event := .survivorFold (1) 93752

def exact93754RawTerms : List Term := []

theorem exact93754RawTermsValid :
    exact93754RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event93754 : Event := .resultExact (⟨.program ⟨257⟩, ⟨28895⟩⟩) exact93754RawTerms (.finite 1296) 93751 (.finite 1296) (some (93752))

def event93755 : Event := .predecessor (⟨.program ⟨257⟩, ⟨28896⟩⟩) 0 ⟨28895⟩ 93754

def event93756 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨28896⟩⟩) (.identity (.predecessor 0 93755 .coefficient))

def event93757 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨28896⟩⟩) (.finite 1296)

def event93758 : Event := .predecessor (⟨.program ⟨257⟩, ⟨29128⟩⟩) 0 ⟨28896⟩ 93757

def event93759 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨29128⟩⟩) (.authority (.programFamilyFact))

def exact93760RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨29128⟩⟩], []⟩, (1)⟩]

theorem exact93760RawTermsValid :
    exact93760RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event93760 : Event := .resultExact (⟨.program ⟨257⟩, ⟨29128⟩⟩) exact93760RawTerms (.finite 36) 93759 .exactZero (none)

def event93761 : Event := .predecessor (⟨.program ⟨257⟩, ⟨29129⟩⟩) 0 ⟨29128⟩ 93760

def event93762 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨29129⟩⟩) (.identity (.predecessor 0 93761 .coefficient))

def event93763 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨29129⟩⟩) (.finite 36)

def event93764 : Event := .predecessor (⟨.program ⟨257⟩, ⟨29936⟩⟩) 0 ⟨29129⟩ 93763

def event93765 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨29936⟩⟩) (.authority (.relationPreimageSource ⟨81⟩))

def exact93766RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨29936⟩⟩]⟩, (1)⟩]

theorem exact93766RawTermsValid :
    exact93766RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event93766 : Event := .resultExact (⟨.program ⟨257⟩, ⟨29936⟩⟩) exact93766RawTerms (.finite 5647228698) 93765 .exactZero (none)

def event93767 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35⟩⟩) (.authority (.operator))

def exact93768RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact93768RawTermsValid :
    exact93768RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event93768 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35⟩⟩) exact93768RawTerms .large 93767 .exactZero (none)

def event93769 : Event := .predecessor (⟨.program ⟨257⟩, ⟨29937⟩⟩) 0 ⟨35⟩ 93768

def event93770 : Event := .predecessor (⟨.program ⟨257⟩, ⟨29937⟩⟩) 1 ⟨29936⟩ 93766

def event93771 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨29937⟩⟩) (.product (.predecessor 0 93769 .coefficient) (.predecessor 1 93770 .coefficient) (⟨false, false, none, none, none⟩))

def event93772 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨29937⟩⟩, .operator (⟨93768, 0⟩, ⟨93766, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29936⟩⟩]⟩, (1)⟩)

def exact93773RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29936⟩⟩]⟩, (1)⟩]

theorem exact93773RawTermsValid :
    exact93773RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event93773 : Event := .resultExact (⟨.program ⟨257⟩, ⟨29937⟩⟩) exact93773RawTerms .large 93771 .exactZero (none)

def event93774 : Event := .preFoldPolynomial 93773 [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29936⟩⟩]⟩, (1)⟩] .exactZero none

def exact93775RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29936⟩⟩]⟩, (1)⟩]

def event93775 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨29937⟩⟩) 93774 exact93775RawTerms .large 93771 .exactZero (none)

def event93776 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨31098⟩⟩)

def event93777 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event93778 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event93779 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9843⟩⟩) (.authority (.operator))

def event93780 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨9843⟩⟩) (.finite 14)

def event93781 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event93782 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event93783 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event93784 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event93785 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 93784

def event93786 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 93782

def event93787 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 93785 .coefficient) (.value (.predecessor 1 93786 .coefficient)))

def event93788 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event93789 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9845⟩⟩) 0 ⟨392⟩ 93788

def event93790 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9845⟩⟩) 1 ⟨9843⟩ 93780

def event93791 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9845⟩⟩) (.sum [.predecessor 0 93789 .coefficient, .predecessor 1 93790 .coefficient])

def event93792 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨9845⟩⟩) (.finite 655354)

def event93793 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9901⟩⟩) 0 ⟨9845⟩ 93792

def event93794 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9901⟩⟩) 1 ⟨5426⟩ 93778

def event93795 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9901⟩⟩) (.identity (.predecessor 1 93794 .coefficient))

def event93796 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨9901⟩⟩) (.finite 655360)

def event93797 : Event := .predecessor (⟨.program ⟨257⟩, ⟨28894⟩⟩) 0 ⟨9901⟩ 93796

def event93798 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨28894⟩⟩) (.authority (.programFamilyFact))

def exact93799RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨28894⟩⟩], []⟩, (1)⟩]

theorem exact93799RawTermsValid :
    exact93799RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event93799 : Event := .resultExact (⟨.program ⟨257⟩, ⟨28894⟩⟩) exact93799RawTerms (.finite 36) 93798 .exactZero (none)

def event93800 : Event := .predecessor (⟨.program ⟨257⟩, ⟨13356⟩⟩) 0 ⟨9901⟩ 93796

def event93801 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨13356⟩⟩) (.authority (.programFamilyFact))

def exact93802RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨13356⟩⟩], []⟩, (1)⟩]

theorem exact93802RawTermsValid :
    exact93802RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event93802 : Event := .resultExact (⟨.program ⟨257⟩, ⟨13356⟩⟩) exact93802RawTerms (.finite 36) 93801 .exactZero (none)

def event93803 : Event := .predecessor (⟨.program ⟨257⟩, ⟨28895⟩⟩) 0 ⟨13356⟩ 93802

def event93804 : Event := .predecessor (⟨.program ⟨257⟩, ⟨28895⟩⟩) 1 ⟨28894⟩ 93799

def event93805 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨28895⟩⟩) (.product (.predecessor 0 93803 .coefficient) (.predecessor 1 93804 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event93806 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨28895⟩⟩, .operator (⟨93802, 0⟩, ⟨93799, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨13356⟩⟩, ⟨.program ⟨257⟩, ⟨28894⟩⟩], []⟩, (1)⟩)

def exact93807RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨13356⟩⟩, ⟨.program ⟨257⟩, ⟨28894⟩⟩], []⟩, (1)⟩]

theorem exact93807RawTermsValid :
    exact93807RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event93807 : Event := .resultExact (⟨.program ⟨257⟩, ⟨28895⟩⟩) exact93807RawTerms (.finite 1296) 93805 .exactZero (none)

def event93808 : Event := .predecessor (⟨.program ⟨257⟩, ⟨28896⟩⟩) 0 ⟨28895⟩ 93807

def event93809 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨28896⟩⟩) (.identity (.predecessor 0 93808 .coefficient))

def event93810 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨28896⟩⟩) (.finite 1296)

def event93811 : Event := .predecessor (⟨.program ⟨257⟩, ⟨29128⟩⟩) 0 ⟨28896⟩ 93810

def event93812 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨29128⟩⟩) (.authority (.programFamilyFact))

def exact93813RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨29128⟩⟩], []⟩, (1)⟩]

theorem exact93813RawTermsValid :
    exact93813RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event93813 : Event := .resultExact (⟨.program ⟨257⟩, ⟨29128⟩⟩) exact93813RawTerms (.finite 36) 93812 .exactZero (none)

def event93814 : Event := .predecessor (⟨.program ⟨257⟩, ⟨29129⟩⟩) 0 ⟨29128⟩ 93813

def event93815 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨29129⟩⟩) (.identity (.predecessor 0 93814 .coefficient))

def event93816 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨29129⟩⟩) (.finite 36)

def event93817 : Event := .predecessor (⟨.program ⟨257⟩, ⟨30284⟩⟩) 0 ⟨29129⟩ 93816

def event93818 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨30284⟩⟩) (.authority (.programFamilyFact))

def event93819 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨30284⟩⟩) (.finite 3720)

def event93820 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨7177⟩⟩) .missing

def event93821 : Event := .predecessor (⟨.program ⟨257⟩, ⟨30286⟩⟩) 0 ⟨7177⟩ 93820

def event93822 : Event := .predecessor (⟨.program ⟨257⟩, ⟨30286⟩⟩) 1 ⟨30284⟩ 93819

def event93823 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨30286⟩⟩) (.authority (.operator))

def exact93824RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨30286⟩⟩]⟩, (1)⟩]

theorem exact93824RawTermsValid :
    exact93824RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event93824 : Event := .resultExact (⟨.program ⟨257⟩, ⟨30286⟩⟩) exact93824RawTerms .large 93823 .exactZero (none)

def event93825 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31094⟩⟩) 0 ⟨30286⟩ 93824

def event93826 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31094⟩⟩) (.authority (.operator))

def exact93827RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨31094⟩⟩]⟩, (1)⟩]

theorem exact93827RawTermsValid :
    exact93827RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event93827 : Event := .resultExact (⟨.program ⟨257⟩, ⟨31094⟩⟩) exact93827RawTerms (.finite 8192) 93826 .exactZero (none)

def event93828 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event93829 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event93830 : Event := .predecessor (⟨.program ⟨257⟩, ⟨30466⟩⟩) 0 ⟨29129⟩ 93816

def event93831 : Event := .predecessor (⟨.program ⟨257⟩, ⟨30466⟩⟩) 1 ⟨136⟩ 93829

def event93832 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨30466⟩⟩) (.sum [.predecessor 0 93830 .coefficient, .predecessor 1 93831 .coefficient])

def event93833 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨30466⟩⟩) (.finite 36)

def event93834 : Event := .predecessor (⟨.program ⟨257⟩, ⟨30467⟩⟩) 0 ⟨30466⟩ 93833

def event93835 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨30467⟩⟩) (.identity (.predecessor 0 93834 .coefficient))

def exact93836RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨29128⟩⟩], []⟩, (1)⟩]

theorem exact93836RawTermsValid :
    exact93836RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event93836 : Event := .resultExact (⟨.program ⟨257⟩, ⟨30467⟩⟩) exact93836RawTerms (.finite 36) 93835 .exactZero (none)

def event93837 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact93838RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact93838RawTermsValid :
    exact93838RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event93838 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact93838RawTerms .large 93837 .exactZero (none)

def event93839 : Event := .predecessor (⟨.program ⟨257⟩, ⟨30468⟩⟩) 0 ⟨6908⟩ 93838

def event93840 : Event := .predecessor (⟨.program ⟨257⟩, ⟨30468⟩⟩) 1 ⟨30467⟩ 93836

def event93841 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨30468⟩⟩) (.product (.predecessor 0 93839 .coefficient) (.predecessor 1 93840 .coefficient) (⟨false, false, none, none, none⟩))

def event93842 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨30468⟩⟩, .operator (⟨93838, 0⟩, ⟨93836, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨29128⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact93843RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨29128⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact93843RawTermsValid :
    exact93843RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event93843 : Event := .resultExact (⟨.program ⟨257⟩, ⟨30468⟩⟩) exact93843RawTerms .large 93841 .exactZero (none)

def event93844 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7190⟩⟩) 0 ⟨7177⟩ 93820

def event93845 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7190⟩⟩) (.authority (.operator))

def exact93846RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7190⟩⟩]⟩, (1)⟩]

theorem exact93846RawTermsValid :
    exact93846RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event93846 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7190⟩⟩) exact93846RawTerms .large 93845 .exactZero (none)

def event93847 : Event := .predecessor (⟨.program ⟨257⟩, ⟨30469⟩⟩) 0 ⟨7190⟩ 93846

def event93848 : Event := .predecessor (⟨.program ⟨257⟩, ⟨30469⟩⟩) 1 ⟨30468⟩ 93843

def event93849 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨30469⟩⟩) (.sum [.predecessor 0 93847 .coefficient, .predecessor 1 93848 .coefficient])

def exact93850RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7190⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨29128⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact93850RawTermsValid :
    exact93850RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event93850 : Event := .resultExact (⟨.program ⟨257⟩, ⟨30469⟩⟩) exact93850RawTerms .large 93849 .exactZero (none)

def event93851 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31095⟩⟩) 0 ⟨30469⟩ 93850

def event93852 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31095⟩⟩) 1 ⟨31094⟩ 93827

def event93853 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31095⟩⟩) (.product (.predecessor 0 93851 .coefficient) (.predecessor 1 93852 .coefficient) (⟨false, false, none, none, none⟩))

def event93854 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨31095⟩⟩, .operator (⟨93850, 0⟩, ⟨93827, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨31094⟩⟩]⟩, (1)⟩)

def event93855 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨31095⟩⟩, .operator (⟨93850, 1⟩, ⟨93827, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨29128⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨31094⟩⟩]⟩, (-1)⟩)

def event93856 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨31095⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨29128⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨31094⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨31094⟩⟩) ⟨30286⟩ 93824)

def event93857 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨31095⟩⟩, .relation 93856 0, ⟨[⟨.program ⟨257⟩, ⟨29128⟩⟩], [⟨.program ⟨257⟩, ⟨30286⟩⟩]⟩, (-1)⟩)

def exact93858RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨31094⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨29128⟩⟩], [⟨.program ⟨257⟩, ⟨30286⟩⟩]⟩, (-1)⟩]

theorem exact93858RawTermsValid :
    exact93858RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event93858 : Event := .resultExact (⟨.program ⟨257⟩, ⟨31095⟩⟩) exact93858RawTerms .large 93853 .exactZero (none)

def event93859 : Event := .predecessor (⟨.program ⟨257⟩, ⟨29364⟩⟩) 0 ⟨29129⟩ 93816

def event93860 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨29364⟩⟩) (.authority (.programFamilyFact))

def exact93861RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨29364⟩⟩], []⟩, (1)⟩]

theorem exact93861RawTermsValid :
    exact93861RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event93861 : Event := .resultExact (⟨.program ⟨257⟩, ⟨29364⟩⟩) exact93861RawTerms (.finite 62) 93860 .exactZero (none)

def event93862 : Event := .predecessor (⟨.program ⟨257⟩, ⟨29365⟩⟩) 0 ⟨6908⟩ 93838

def event93863 : Event := .predecessor (⟨.program ⟨257⟩, ⟨29365⟩⟩) 1 ⟨29364⟩ 93861

def event93864 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨29365⟩⟩) (.product (.predecessor 0 93862 .coefficient) (.predecessor 1 93863 .coefficient) (⟨false, true, none, none, some 1⟩))

def event93865 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨29365⟩⟩, .operator (⟨93838, 0⟩, ⟨93861, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨29364⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact93866RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨29364⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact93866RawTermsValid :
    exact93866RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event93866 : Event := .resultExact (⟨.program ⟨257⟩, ⟨29365⟩⟩) exact93866RawTerms .large 93864 .exactZero (none)

def event93867 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7220⟩⟩) 0 ⟨7177⟩ 93820

def event93868 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7220⟩⟩) (.authority (.operator))

def exact93869RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩]

theorem exact93869RawTermsValid :
    exact93869RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event93869 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7220⟩⟩) exact93869RawTerms .large 93868 .exactZero (none)

def event93870 : Event := .predecessor (⟨.program ⟨257⟩, ⟨29366⟩⟩) 0 ⟨7220⟩ 93869

def event93871 : Event := .predecessor (⟨.program ⟨257⟩, ⟨29366⟩⟩) 1 ⟨29365⟩ 93866

def event93872 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨29366⟩⟩) (.sum [.predecessor 0 93870 .coefficient, .predecessor 1 93871 .coefficient])

def exact93873RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨29364⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact93873RawTermsValid :
    exact93873RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event93873 : Event := .resultExact (⟨.program ⟨257⟩, ⟨29366⟩⟩) exact93873RawTerms .large 93872 .exactZero (none)

def event93874 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31098⟩⟩) 0 ⟨29366⟩ 93873

def event93875 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31098⟩⟩) 1 ⟨31095⟩ 93858

def event93876 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31098⟩⟩) (.sum [.predecessor 0 93874 .coefficient, .predecessor 1 93875 .coefficient])

def exact93877RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨31094⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨29128⟩⟩], [⟨.program ⟨257⟩, ⟨30286⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨29364⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact93877RawTermsValid :
    exact93877RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event93877 : Event := .resultExact (⟨.program ⟨257⟩, ⟨31098⟩⟩) exact93877RawTerms .large 93876 .exactZero (none)

def event93878 : Event := .preFoldPolynomial 93877 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨31094⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨29128⟩⟩], [⟨.program ⟨257⟩, ⟨30286⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨29364⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact93879RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨31094⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨29128⟩⟩], [⟨.program ⟨257⟩, ⟨30286⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨29364⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event93879 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨31098⟩⟩) 93878 exact93879RawTerms .large 93876 .exactZero (none)

def event93880 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨29129⟩⟩) ⟨⟨99⟩, ⟨81⟩, ⟨135⟩⟩ ⟨93722, 93880⟩

def event93881 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨29939⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29936⟩⟩]⟩) (1) 0 2 (.universal 93880 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29936⟩⟩]⟩) (none) 93879)

def event93882 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨29939⟩⟩, .relation 93881 1, ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩)

def event93883 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨29939⟩⟩, .relation 93881 0, ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨31094⟩⟩]⟩, (-1)⟩)

def event93884 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨29939⟩⟩, .relation 93881 2, ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨29128⟩⟩], [⟨.program ⟨257⟩, ⟨30286⟩⟩]⟩, (1)⟩)

def event93885 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨29939⟩⟩, .relation 93881 3, ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨29364⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact93886RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨31094⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨29128⟩⟩], [⟨.program ⟨257⟩, ⟨30286⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨29364⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact93886RawTermsValid :
    exact93886RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event93886 : Event := .resultExact (⟨.program ⟨257⟩, ⟨29939⟩⟩) exact93886RawTerms .large 93718 (.finite 202072841853861888) (some (93720))

def event93887 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31097⟩⟩) 0 ⟨29939⟩ 93886

def event93888 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31097⟩⟩) 1 ⟨31096⟩ 93708

def event93889 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31097⟩⟩) (.sum [.predecessor 0 93887 .coefficient, .predecessor 1 93888 .coefficient])

def event93890 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨31097⟩⟩, .operator (⟨93886, 0⟩, ⟨93708, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨31094⟩⟩]⟩, (1)⟩)

def event93891 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨31097⟩⟩, .operator (⟨93886, 2⟩, ⟨93708, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨29128⟩⟩], [⟨.program ⟨257⟩, ⟨30286⟩⟩]⟩, (-1)⟩)

def event93892 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31097⟩⟩) (.sum [.result 93886 .summary, .result 93708 .summary])

def exact93893RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨29364⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact93893RawTermsValid :
    exact93893RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event93893 : Event := .resultExact (⟨.program ⟨257⟩, ⟨31097⟩⟩) exact93893RawTerms .large 93889 (.finite 32192146870060392302605751287808) (some (93892))

def event93894 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27604⟩⟩) 0 ⟨26449⟩ 4012

def event93895 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨27604⟩⟩) (.authority (.programFamilyFact))

def event93896 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨27604⟩⟩) (.finite 3720)

def event93897 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27606⟩⟩) 0 ⟨7177⟩ 15500

def event93898 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27606⟩⟩) 1 ⟨27604⟩ 93896

def event93899 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨27606⟩⟩) (.authority (.operator))

def exact93900RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨27606⟩⟩]⟩, (1)⟩]

theorem exact93900RawTermsValid :
    exact93900RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event93900 : Event := .resultExact (⟨.program ⟨257⟩, ⟨27606⟩⟩) exact93900RawTerms .large 93899 .exactZero (none)

def event93901 : Event := .predecessor (⟨.program ⟨257⟩, ⟨28414⟩⟩) 0 ⟨27606⟩ 93900

def event93902 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨28414⟩⟩) (.authority (.operator))

def exact93903RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨28414⟩⟩]⟩, (1)⟩]

theorem exact93903RawTermsValid :
    exact93903RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event93903 : Event := .resultExact (⟨.program ⟨257⟩, ⟨28414⟩⟩) exact93903RawTerms (.finite 8192) 93902 .exactZero (none)

def event93904 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27438⟩⟩) 0 ⟨26216⟩ 4006

def event93905 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨27438⟩⟩) (.authority (.programFamilyFact))

def event93906 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨27438⟩⟩) (.finite 3720)

def event93907 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27439⟩⟩) 0 ⟨7177⟩ 15500

def event93908 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27439⟩⟩) 1 ⟨27438⟩ 93906

def event93909 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨27439⟩⟩) (.authority (.operator))

def exact93910RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨27439⟩⟩]⟩, (1)⟩]

theorem exact93910RawTermsValid :
    exact93910RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event93910 : Event := .resultExact (⟨.program ⟨257⟩, ⟨27439⟩⟩) exact93910RawTerms .large 93909 .exactZero (none)

def event93911 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27974⟩⟩) 0 ⟨27439⟩ 93910

def event93912 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨27974⟩⟩) (.authority (.operator))

def exact93913RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨27974⟩⟩]⟩, (1)⟩]

theorem exact93913RawTermsValid :
    exact93913RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event93913 : Event := .resultExact (⟨.program ⟨257⟩, ⟨27974⟩⟩) exact93913RawTerms (.finite 8192) 93912 .exactZero (none)

def event93914 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26217⟩⟩) 0 ⟨26214⟩ 3995

def event93915 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26217⟩⟩) 1 ⟨9904⟩ 90528

def event93916 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨26217⟩⟩) (.tensor (.predecessor 0 93914 .coefficient) (.predecessor 1 93915 .coefficient) true false)

def event93917 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨26217⟩⟩, .operator (⟨3995, 0⟩, ⟨90528, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨26214⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact93918RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨26214⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact93918RawTermsValid :
    exact93918RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event93918 : Event := .resultExact (⟨.program ⟨257⟩, ⟨26217⟩⟩) exact93918RawTerms .large 93916 .exactZero (none)

def event93919 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9912⟩⟩) 0 ⟨9903⟩ 90398

def event93920 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9912⟩⟩) 1 ⟨7278⟩ 20587

def event93921 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9912⟩⟩) (.product (.predecessor 0 93919 .coefficient) (.predecessor 1 93920 .coefficient) (⟨false, false, none, none, none⟩))

def event93922 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨9912⟩⟩, .operator (⟨90398, 0⟩, ⟨20587, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7278⟩⟩]⟩, (1)⟩)

def exact93923RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7278⟩⟩]⟩, (1)⟩]

theorem exact93923RawTermsValid :
    exact93923RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event93923 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9912⟩⟩) exact93923RawTerms .large 93921 .exactZero (none)

def event93924 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26218⟩⟩) 0 ⟨9912⟩ 93923

def event93925 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26218⟩⟩) 1 ⟨26217⟩ 93918

def event93926 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨26218⟩⟩) (.sum [.predecessor 0 93924 .coefficient, .predecessor 1 93925 .coefficient])

def exact93927RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7278⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨26214⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact93927RawTermsValid :
    exact93927RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event93927 : Event := .resultExact (⟨.program ⟨257⟩, ⟨26218⟩⟩) exact93927RawTerms .large 93926 .exactZero (none)

def event93928 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26219⟩⟩) 0 ⟨26218⟩ 93927

def event93929 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26219⟩⟩) 1 ⟨104⟩ 20579

def event93930 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨26219⟩⟩) (.sum [.predecessor 0 93928 .coefficient, .predecessor 1 93929 .coefficient])

def event93931 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨26219⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨104⟩⟩]⟩) [⟨.result 20579 .coefficient, false, none⟩])

def event93932 : Event := .survivorFold (1) 93931

def exact93933RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7278⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨26214⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact93933RawTermsValid :
    exact93933RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event93933 : Event := .resultExact (⟨.program ⟨257⟩, ⟨26219⟩⟩) exact93933RawTerms .large 93930 (.finite 26) (some (93931))

def event93934 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26220⟩⟩) 0 ⟨26219⟩ 93933

def event93935 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26220⟩⟩) 1 ⟨13056⟩ 3998

def event93936 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨26220⟩⟩) (.product (.predecessor 0 93934 .coefficient) (.predecessor 1 93935 .coefficient) (⟨false, true, none, none, some 1⟩))

def event93937 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨26220⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨13056⟩⟩], []⟩) [⟨.result 3998 .coefficient, true, some 1⟩])

def event93938 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨26220⟩⟩) (.product (.result 93933 .summary) (.transfer 93937) (⟨false, false, none, none, none⟩))

def event93939 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨26220⟩⟩, .operator (⟨93933, 1⟩, ⟨3998, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨13056⟩⟩, ⟨.program ⟨257⟩, ⟨26214⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event93940 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨26220⟩⟩, .operator (⟨93933, 0⟩, ⟨3998, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨13056⟩⟩], [⟨.program ⟨257⟩, ⟨7278⟩⟩]⟩, (1)⟩)

def exact93941RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨13056⟩⟩], [⟨.program ⟨257⟩, ⟨7278⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨13056⟩⟩, ⟨.program ⟨257⟩, ⟨26214⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact93941RawTermsValid :
    exact93941RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event93941 : Event := .resultExact (⟨.program ⟨257⟩, ⟨26220⟩⟩) exact93941RawTerms .large 93936 (.finite 25559040) (some (93938))

def event93942 : Event := .predecessor (⟨.program ⟨257⟩, ⟨13057⟩⟩) 0 ⟨13056⟩ 3998

def event93943 : Event := .predecessor (⟨.program ⟨257⟩, ⟨13057⟩⟩) 1 ⟨9904⟩ 90528

def event93944 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨13057⟩⟩) (.tensor (.predecessor 0 93942 .coefficient) (.predecessor 1 93943 .coefficient) true false)

def event93945 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨13057⟩⟩, .operator (⟨3998, 0⟩, ⟨90528, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨13056⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact93946RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨13056⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact93946RawTermsValid :
    exact93946RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event93946 : Event := .resultExact (⟨.program ⟨257⟩, ⟨13057⟩⟩) exact93946RawTerms .large 93944 .exactZero (none)

def event93947 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9929⟩⟩) 0 ⟨9903⟩ 90398

def event93948 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9929⟩⟩) 1 ⟨7295⟩ 20628

def event93949 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9929⟩⟩) (.product (.predecessor 0 93947 .coefficient) (.predecessor 1 93948 .coefficient) (⟨false, false, none, none, none⟩))

def event93950 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨9929⟩⟩, .operator (⟨90398, 0⟩, ⟨20628, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7295⟩⟩]⟩, (1)⟩)

def exact93951RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7295⟩⟩]⟩, (1)⟩]

theorem exact93951RawTermsValid :
    exact93951RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event93951 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9929⟩⟩) exact93951RawTerms .large 93949 .exactZero (none)

def eventLeaf5856 : Array AnnotatedEvent := #[
  { event := event93696
    frameStart := 0 },
  { event := event93697
    frameStart := 0 },
  { event := event93698
    frameStart := 0 },
  { event := event93699
    frameStart := 0 },
  { event := event93700
    frameStart := 0 },
  { event := event93701
    frameStart := 0 },
  { event := event93702
    frameStart := 0 },
  { event := event93703
    frameStart := 0 },
  { event := event93704
    frameStart := 0 },
  { event := event93705
    frameStart := 0 },
  { event := event93706
    frameStart := 0 },
  { event := event93707
    frameStart := 0 },
  { event := event93708
    frameStart := 0 },
  { event := event93709
    frameStart := 0 },
  { event := event93710
    frameStart := 0 },
  { event := event93711
    frameStart := 0 }
]

def eventLeaf5857 : Array AnnotatedEvent := #[
  { event := event93712
    frameStart := 0 },
  { event := event93713
    frameStart := 0 },
  { event := event93714
    frameStart := 0 },
  { event := event93715
    frameStart := 0 },
  { event := event93716
    frameStart := 0 },
  { event := event93717
    frameStart := 0 },
  { event := event93718
    frameStart := 0 },
  { event := event93719
    frameStart := 0 },
  { event := event93720
    frameStart := 0 },
  { event := event93721
    frameStart := 0 },
  { event := event93722
    frameStart := 93722 },
  { event := event93723
    frameStart := 93722 },
  { event := event93724
    frameStart := 93722 },
  { event := event93725
    frameStart := 93722 },
  { event := event93726
    frameStart := 93722 },
  { event := event93727
    frameStart := 93722 }
]

def eventLeaf5858 : Array AnnotatedEvent := #[
  { event := event93728
    frameStart := 93722 },
  { event := event93729
    frameStart := 93722 },
  { event := event93730
    frameStart := 93722 },
  { event := event93731
    frameStart := 93722 },
  { event := event93732
    frameStart := 93722 },
  { event := event93733
    frameStart := 93722 },
  { event := event93734
    frameStart := 93722 },
  { event := event93735
    frameStart := 93722 },
  { event := event93736
    frameStart := 93722 },
  { event := event93737
    frameStart := 93722 },
  { event := event93738
    frameStart := 93722 },
  { event := event93739
    frameStart := 93722 },
  { event := event93740
    frameStart := 93722 },
  { event := event93741
    frameStart := 93722 },
  { event := event93742
    frameStart := 93722 },
  { event := event93743
    frameStart := 93722 }
]

def eventLeaf5859 : Array AnnotatedEvent := #[
  { event := event93744
    frameStart := 93722 },
  { event := event93745
    frameStart := 93722 },
  { event := event93746
    frameStart := 93722 },
  { event := event93747
    frameStart := 93722 },
  { event := event93748
    frameStart := 93722 },
  { event := event93749
    frameStart := 93722 },
  { event := event93750
    frameStart := 93722 },
  { event := event93751
    frameStart := 93722 },
  { event := event93752
    frameStart := 93722 },
  { event := event93753
    frameStart := 93722 },
  { event := event93754
    frameStart := 93722 },
  { event := event93755
    frameStart := 93722 },
  { event := event93756
    frameStart := 93722 },
  { event := event93757
    frameStart := 93722 },
  { event := event93758
    frameStart := 93722 },
  { event := event93759
    frameStart := 93722 }
]

def eventLeaf5860 : Array AnnotatedEvent := #[
  { event := event93760
    frameStart := 93722 },
  { event := event93761
    frameStart := 93722 },
  { event := event93762
    frameStart := 93722 },
  { event := event93763
    frameStart := 93722 },
  { event := event93764
    frameStart := 93722 },
  { event := event93765
    frameStart := 93722 },
  { event := event93766
    frameStart := 93722 },
  { event := event93767
    frameStart := 93722 },
  { event := event93768
    frameStart := 93722 },
  { event := event93769
    frameStart := 93722 },
  { event := event93770
    frameStart := 93722 },
  { event := event93771
    frameStart := 93722 },
  { event := event93772
    frameStart := 93722 },
  { event := event93773
    frameStart := 93722 },
  { event := event93774
    frameStart := 93722 },
  { event := event93775
    frameStart := 93722 }
]

def eventLeaf5861 : Array AnnotatedEvent := #[
  { event := event93776
    frameStart := 93776 },
  { event := event93777
    frameStart := 93776 },
  { event := event93778
    frameStart := 93776 },
  { event := event93779
    frameStart := 93776 },
  { event := event93780
    frameStart := 93776 },
  { event := event93781
    frameStart := 93776 },
  { event := event93782
    frameStart := 93776 },
  { event := event93783
    frameStart := 93776 },
  { event := event93784
    frameStart := 93776 },
  { event := event93785
    frameStart := 93776 },
  { event := event93786
    frameStart := 93776 },
  { event := event93787
    frameStart := 93776 },
  { event := event93788
    frameStart := 93776 },
  { event := event93789
    frameStart := 93776 },
  { event := event93790
    frameStart := 93776 },
  { event := event93791
    frameStart := 93776 }
]

def eventLeaf5862 : Array AnnotatedEvent := #[
  { event := event93792
    frameStart := 93776 },
  { event := event93793
    frameStart := 93776 },
  { event := event93794
    frameStart := 93776 },
  { event := event93795
    frameStart := 93776 },
  { event := event93796
    frameStart := 93776 },
  { event := event93797
    frameStart := 93776 },
  { event := event93798
    frameStart := 93776 },
  { event := event93799
    frameStart := 93776 },
  { event := event93800
    frameStart := 93776 },
  { event := event93801
    frameStart := 93776 },
  { event := event93802
    frameStart := 93776 },
  { event := event93803
    frameStart := 93776 },
  { event := event93804
    frameStart := 93776 },
  { event := event93805
    frameStart := 93776 },
  { event := event93806
    frameStart := 93776 },
  { event := event93807
    frameStart := 93776 }
]

def eventLeaf5863 : Array AnnotatedEvent := #[
  { event := event93808
    frameStart := 93776 },
  { event := event93809
    frameStart := 93776 },
  { event := event93810
    frameStart := 93776 },
  { event := event93811
    frameStart := 93776 },
  { event := event93812
    frameStart := 93776 },
  { event := event93813
    frameStart := 93776 },
  { event := event93814
    frameStart := 93776 },
  { event := event93815
    frameStart := 93776 },
  { event := event93816
    frameStart := 93776 },
  { event := event93817
    frameStart := 93776 },
  { event := event93818
    frameStart := 93776 },
  { event := event93819
    frameStart := 93776 },
  { event := event93820
    frameStart := 93776 },
  { event := event93821
    frameStart := 93776 },
  { event := event93822
    frameStart := 93776 },
  { event := event93823
    frameStart := 93776 }
]

def eventLeaf5864 : Array AnnotatedEvent := #[
  { event := event93824
    frameStart := 93776 },
  { event := event93825
    frameStart := 93776 },
  { event := event93826
    frameStart := 93776 },
  { event := event93827
    frameStart := 93776 },
  { event := event93828
    frameStart := 93776 },
  { event := event93829
    frameStart := 93776 },
  { event := event93830
    frameStart := 93776 },
  { event := event93831
    frameStart := 93776 },
  { event := event93832
    frameStart := 93776 },
  { event := event93833
    frameStart := 93776 },
  { event := event93834
    frameStart := 93776 },
  { event := event93835
    frameStart := 93776 },
  { event := event93836
    frameStart := 93776 },
  { event := event93837
    frameStart := 93776 },
  { event := event93838
    frameStart := 93776 },
  { event := event93839
    frameStart := 93776 }
]

def eventLeaf5865 : Array AnnotatedEvent := #[
  { event := event93840
    frameStart := 93776 },
  { event := event93841
    frameStart := 93776 },
  { event := event93842
    frameStart := 93776 },
  { event := event93843
    frameStart := 93776 },
  { event := event93844
    frameStart := 93776 },
  { event := event93845
    frameStart := 93776 },
  { event := event93846
    frameStart := 93776 },
  { event := event93847
    frameStart := 93776 },
  { event := event93848
    frameStart := 93776 },
  { event := event93849
    frameStart := 93776 },
  { event := event93850
    frameStart := 93776 },
  { event := event93851
    frameStart := 93776 },
  { event := event93852
    frameStart := 93776 },
  { event := event93853
    frameStart := 93776 },
  { event := event93854
    frameStart := 93776 },
  { event := event93855
    frameStart := 93776 }
]

def eventLeaf5866 : Array AnnotatedEvent := #[
  { event := event93856
    frameStart := 93776 },
  { event := event93857
    frameStart := 93776 },
  { event := event93858
    frameStart := 93776 },
  { event := event93859
    frameStart := 93776 },
  { event := event93860
    frameStart := 93776 },
  { event := event93861
    frameStart := 93776 },
  { event := event93862
    frameStart := 93776 },
  { event := event93863
    frameStart := 93776 },
  { event := event93864
    frameStart := 93776 },
  { event := event93865
    frameStart := 93776 },
  { event := event93866
    frameStart := 93776 },
  { event := event93867
    frameStart := 93776 },
  { event := event93868
    frameStart := 93776 },
  { event := event93869
    frameStart := 93776 },
  { event := event93870
    frameStart := 93776 },
  { event := event93871
    frameStart := 93776 }
]

def eventLeaf5867 : Array AnnotatedEvent := #[
  { event := event93872
    frameStart := 93776 },
  { event := event93873
    frameStart := 93776 },
  { event := event93874
    frameStart := 93776 },
  { event := event93875
    frameStart := 93776 },
  { event := event93876
    frameStart := 93776 },
  { event := event93877
    frameStart := 93776 },
  { event := event93878
    frameStart := 93776 },
  { event := event93879
    frameStart := 93776 },
  { event := event93880
    frameStart := 0 },
  { event := event93881
    frameStart := 0 },
  { event := event93882
    frameStart := 0 },
  { event := event93883
    frameStart := 0 },
  { event := event93884
    frameStart := 0 },
  { event := event93885
    frameStart := 0 },
  { event := event93886
    frameStart := 0 },
  { event := event93887
    frameStart := 0 }
]

def eventLeaf5868 : Array AnnotatedEvent := #[
  { event := event93888
    frameStart := 0 },
  { event := event93889
    frameStart := 0 },
  { event := event93890
    frameStart := 0 },
  { event := event93891
    frameStart := 0 },
  { event := event93892
    frameStart := 0 },
  { event := event93893
    frameStart := 0 },
  { event := event93894
    frameStart := 0 },
  { event := event93895
    frameStart := 0 },
  { event := event93896
    frameStart := 0 },
  { event := event93897
    frameStart := 0 },
  { event := event93898
    frameStart := 0 },
  { event := event93899
    frameStart := 0 },
  { event := event93900
    frameStart := 0 },
  { event := event93901
    frameStart := 0 },
  { event := event93902
    frameStart := 0 },
  { event := event93903
    frameStart := 0 }
]

def eventLeaf5869 : Array AnnotatedEvent := #[
  { event := event93904
    frameStart := 0 },
  { event := event93905
    frameStart := 0 },
  { event := event93906
    frameStart := 0 },
  { event := event93907
    frameStart := 0 },
  { event := event93908
    frameStart := 0 },
  { event := event93909
    frameStart := 0 },
  { event := event93910
    frameStart := 0 },
  { event := event93911
    frameStart := 0 },
  { event := event93912
    frameStart := 0 },
  { event := event93913
    frameStart := 0 },
  { event := event93914
    frameStart := 0 },
  { event := event93915
    frameStart := 0 },
  { event := event93916
    frameStart := 0 },
  { event := event93917
    frameStart := 0 },
  { event := event93918
    frameStart := 0 },
  { event := event93919
    frameStart := 0 }
]

def eventLeaf5870 : Array AnnotatedEvent := #[
  { event := event93920
    frameStart := 0 },
  { event := event93921
    frameStart := 0 },
  { event := event93922
    frameStart := 0 },
  { event := event93923
    frameStart := 0 },
  { event := event93924
    frameStart := 0 },
  { event := event93925
    frameStart := 0 },
  { event := event93926
    frameStart := 0 },
  { event := event93927
    frameStart := 0 },
  { event := event93928
    frameStart := 0 },
  { event := event93929
    frameStart := 0 },
  { event := event93930
    frameStart := 0 },
  { event := event93931
    frameStart := 0 },
  { event := event93932
    frameStart := 0 },
  { event := event93933
    frameStart := 0 },
  { event := event93934
    frameStart := 0 },
  { event := event93935
    frameStart := 0 }
]

def eventLeaf5871 : Array AnnotatedEvent := #[
  { event := event93936
    frameStart := 0 },
  { event := event93937
    frameStart := 0 },
  { event := event93938
    frameStart := 0 },
  { event := event93939
    frameStart := 0 },
  { event := event93940
    frameStart := 0 },
  { event := event93941
    frameStart := 0 },
  { event := event93942
    frameStart := 0 },
  { event := event93943
    frameStart := 0 },
  { event := event93944
    frameStart := 0 },
  { event := event93945
    frameStart := 0 },
  { event := event93946
    frameStart := 0 },
  { event := event93947
    frameStart := 0 },
  { event := event93948
    frameStart := 0 },
  { event := event93949
    frameStart := 0 },
  { event := event93950
    frameStart := 0 },
  { event := event93951
    frameStart := 0 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events366
