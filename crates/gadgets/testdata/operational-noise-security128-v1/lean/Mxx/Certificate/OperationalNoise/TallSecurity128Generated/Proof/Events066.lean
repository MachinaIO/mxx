import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events066

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event16896 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5671⟩⟩) 0 ⟨5670⟩ 16895

def event16897 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5671⟩⟩) 1 ⟨2370⟩ 4

def event16898 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5671⟩⟩) (.sum [.predecessor 0 16896 .coefficient, .predecessor 1 16897 .coefficient])

def event16899 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5671⟩⟩) (.finite 655361)

def event16900 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5672⟩⟩) 0 ⟨0⟩ 20

def event16901 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5672⟩⟩) 1 ⟨5670⟩ 16895

def event16902 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5672⟩⟩) 2 ⟨5671⟩ 16899

def event16903 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5672⟩⟩) 3 ⟨136⟩ 6

def event16904 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5672⟩⟩) 4 ⟨2370⟩ 4

def event16905 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5672⟩⟩) (.identity (.predecessor 0 16900 .coefficient))

def exact16906RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨2377⟩⟩]⟩, (1)⟩]

theorem exact16906RawTermsValid :
    exact16906RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event16906 : Event := .resultExact (⟨.program ⟨257⟩, ⟨5672⟩⟩) exact16906RawTerms (.finite 1) 16905 .exactZero (none)

def event16907 : Event := .predecessor (⟨.program ⟨257⟩, ⟨6963⟩⟩) 0 ⟨5672⟩ 16906

def event16908 : Event := .predecessor (⟨.program ⟨257⟩, ⟨6963⟩⟩) 1 ⟨6908⟩ 2

def event16909 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6963⟩⟩) (.product (.predecessor 0 16907 .coefficient) (.predecessor 1 16908 .coefficient) (⟨false, false, none, none, none⟩))

def event16910 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨6963⟩⟩, .operator (⟨16906, 0⟩, ⟨2, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact16911RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact16911RawTermsValid :
    exact16911RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event16911 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6963⟩⟩) exact16911RawTerms .large 16909 .exactZero (none)

def event16912 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5440⟩⟩) 0 ⟨5439⟩ 48

def event16913 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5440⟩⟩) 1 ⟨2370⟩ 4

def event16914 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5440⟩⟩) (.sum [.predecessor 0 16912 .coefficient, .predecessor 1 16913 .coefficient])

def event16915 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5440⟩⟩) (.finite 655361)

def event16916 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5441⟩⟩) 0 ⟨0⟩ 20

def event16917 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5441⟩⟩) 1 ⟨5439⟩ 48

def event16918 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5441⟩⟩) 2 ⟨5440⟩ 16915

def event16919 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5441⟩⟩) 3 ⟨136⟩ 6

def event16920 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5441⟩⟩) 4 ⟨2370⟩ 4

def event16921 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5441⟩⟩) (.identity (.predecessor 0 16916 .coefficient))

def exact16922RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨2371⟩⟩]⟩, (1)⟩]

theorem exact16922RawTermsValid :
    exact16922RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event16922 : Event := .resultExact (⟨.program ⟨257⟩, ⟨5441⟩⟩) exact16922RawTerms (.finite 1) 16921 .exactZero (none)

def event16923 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7589⟩⟩) 0 ⟨5441⟩ 16922

def event16924 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7589⟩⟩) 1 ⟨7235⟩ 15503

def event16925 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7589⟩⟩) (.product (.predecessor 0 16923 .coefficient) (.predecessor 1 16924 .coefficient) (⟨false, false, none, none, none⟩))

def event16926 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨7589⟩⟩, .operator (⟨16922, 0⟩, ⟨15503, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7235⟩⟩]⟩, (1)⟩)

def exact16927RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7235⟩⟩]⟩, (1)⟩]

theorem exact16927RawTermsValid :
    exact16927RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event16927 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7589⟩⟩) exact16927RawTerms .large 16925 .exactZero (none)

def event16928 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9283⟩⟩) 0 ⟨7589⟩ 16927

def event16929 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9283⟩⟩) 1 ⟨6963⟩ 16911

def event16930 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9283⟩⟩) (.sum [.predecessor 0 16928 .coefficient, .predecessor 1 16929 .coefficient])

def exact16931RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7235⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact16931RawTermsValid :
    exact16931RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event16931 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9283⟩⟩) exact16931RawTerms .large 16930 .exactZero (none)

def event16932 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9284⟩⟩) 0 ⟨9283⟩ 16931

def event16933 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9284⟩⟩) 1 ⟨34⟩ 16885

def event16934 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9284⟩⟩) (.sum [.predecessor 0 16932 .coefficient, .predecessor 1 16933 .coefficient])

def event16935 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9284⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨34⟩⟩]⟩) [⟨.result 16885 .coefficient, false, none⟩])

def event16936 : Event := .survivorFold (1) 16935

def exact16937RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7235⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact16937RawTermsValid :
    exact16937RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event16937 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9284⟩⟩) exact16937RawTerms .large 16934 (.finite 26) (some (16935))

def event16938 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67298⟩⟩) 0 ⟨9284⟩ 16937

def event16939 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67298⟩⟩) 1 ⟨67296⟩ 804

def event16940 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67298⟩⟩) (.product (.predecessor 0 16938 .coefficient) (.predecessor 1 16939 .coefficient) (⟨false, false, none, none, none⟩))

def event16941 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67298⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67293⟩⟩], []⟩) [⟨.result 36 .coefficient, true, some 1⟩, ⟨.result 536 .coefficient, true, some 1⟩])

def event16942 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67298⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48245⟩⟩], []⟩) [⟨.result 543 .coefficient, true, some 1⟩, ⟨.result 546 .coefficient, true, some 1⟩])

def event16943 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67298⟩⟩) (.sum [.transfer 16941, .transfer 16942])

def event16944 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67298⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45565⟩⟩], []⟩) [⟨.result 553 .coefficient, true, some 1⟩, ⟨.result 556 .coefficient, true, some 1⟩])

def event16945 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67298⟩⟩) (.sum [.transfer 16943, .transfer 16944])

def event16946 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67298⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨42888⟩⟩], []⟩) [⟨.result 563 .coefficient, true, some 1⟩, ⟨.result 566 .coefficient, true, some 1⟩])

def event16947 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67298⟩⟩) (.sum [.transfer 16945, .transfer 16946])

def event16948 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67298⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40208⟩⟩], []⟩) [⟨.result 573 .coefficient, true, some 1⟩, ⟨.result 576 .coefficient, true, some 1⟩])

def event16949 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67298⟩⟩) (.sum [.transfer 16947, .transfer 16948])

def event16950 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67298⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37525⟩⟩], []⟩) [⟨.result 583 .coefficient, true, some 1⟩, ⟨.result 586 .coefficient, true, some 1⟩])

def event16951 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67298⟩⟩) (.sum [.transfer 16949, .transfer 16950])

def event16952 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67298⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34845⟩⟩], []⟩) [⟨.result 593 .coefficient, true, some 1⟩, ⟨.result 596 .coefficient, true, some 1⟩])

def event16953 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67298⟩⟩) (.sum [.transfer 16951, .transfer 16952])

def event16954 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67298⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29188⟩⟩], []⟩) [⟨.result 603 .coefficient, true, some 1⟩, ⟨.result 606 .coefficient, true, some 1⟩])

def event16955 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67298⟩⟩) (.sum [.transfer 16953, .transfer 16954])

def event16956 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67298⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26508⟩⟩], []⟩) [⟨.result 613 .coefficient, true, some 1⟩, ⟨.result 616 .coefficient, true, some 1⟩])

def event16957 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67298⟩⟩) (.sum [.transfer 16955, .transfer 16956])

def event16958 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67298⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨65980⟩⟩], []⟩) [⟨.result 623 .coefficient, true, some 1⟩, ⟨.result 626 .coefficient, true, some 1⟩])

def event16959 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67298⟩⟩) (.sum [.transfer 16957, .transfer 16958])

def event16960 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67298⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨62919⟩⟩], []⟩) [⟨.result 633 .coefficient, true, some 1⟩, ⟨.result 636 .coefficient, true, some 1⟩])

def event16961 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67298⟩⟩) (.sum [.transfer 16959, .transfer 16960])

def event16962 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67298⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨59939⟩⟩], []⟩) [⟨.result 643 .coefficient, true, some 1⟩, ⟨.result 646 .coefficient, true, some 1⟩])

def event16963 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67298⟩⟩) (.sum [.transfer 16961, .transfer 16962])

def event16964 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67298⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨56959⟩⟩], []⟩) [⟨.result 653 .coefficient, true, some 1⟩, ⟨.result 656 .coefficient, true, some 1⟩])

def event16965 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67298⟩⟩) (.sum [.transfer 16963, .transfer 16964])

def event16966 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67298⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨53979⟩⟩], []⟩) [⟨.result 663 .coefficient, true, some 1⟩, ⟨.result 666 .coefficient, true, some 1⟩])

def event16967 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67298⟩⟩) (.sum [.transfer 16965, .transfer 16966])

def event16968 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67298⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨50999⟩⟩], []⟩) [⟨.result 673 .coefficient, true, some 1⟩, ⟨.result 676 .coefficient, true, some 1⟩])

def event16969 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67298⟩⟩) (.sum [.transfer 16967, .transfer 16968])

def event16970 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67298⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨31935⟩⟩], []⟩) [⟨.result 683 .coefficient, true, some 1⟩, ⟨.result 686 .coefficient, true, some 1⟩])

def event16971 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67298⟩⟩) (.sum [.transfer 16969, .transfer 16970])

def event16972 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67298⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨21915⟩⟩], []⟩) [⟨.result 693 .coefficient, true, some 1⟩, ⟨.result 696 .coefficient, true, some 1⟩])

def event16973 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67298⟩⟩) (.sum [.transfer 16971, .transfer 16972])

def event16974 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67298⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18695⟩⟩], []⟩) [⟨.result 703 .coefficient, true, some 1⟩, ⟨.result 706 .coefficient, true, some 1⟩])

def event16975 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67298⟩⟩) (.sum [.transfer 16973, .transfer 16974])

def event16976 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67298⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15890⟩⟩], []⟩) [⟨.result 713 .coefficient, true, some 1⟩, ⟨.result 716 .coefficient, true, some 1⟩])

def event16977 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67298⟩⟩) (.sum [.transfer 16975, .transfer 16976])

def event16978 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67298⟩⟩) (.product (.result 16937 .summary) (.transfer 16977) (⟨false, false, none, none, none⟩))

def event16979 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67298⟩⟩, .operator (⟨16937, 1⟩, ⟨804, 5⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67293⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def event16980 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67298⟩⟩, .operator (⟨16937, 1⟩, ⟨804, 7⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48245⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event16981 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67298⟩⟩, .operator (⟨16937, 1⟩, ⟨804, 8⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45565⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event16982 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67298⟩⟩, .operator (⟨16937, 1⟩, ⟨804, 9⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨42888⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event16983 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67298⟩⟩, .operator (⟨16937, 1⟩, ⟨804, 11⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40208⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event16984 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67298⟩⟩, .operator (⟨16937, 1⟩, ⟨804, 12⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37525⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event16985 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67298⟩⟩, .operator (⟨16937, 1⟩, ⟨804, 13⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34845⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event16986 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67298⟩⟩, .operator (⟨16937, 1⟩, ⟨804, 15⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29188⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event16987 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67298⟩⟩, .operator (⟨16937, 1⟩, ⟨804, 16⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26508⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event16988 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67298⟩⟩, .operator (⟨16937, 1⟩, ⟨804, 18⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨65980⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event16989 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67298⟩⟩, .operator (⟨16937, 1⟩, ⟨804, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨62919⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event16990 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67298⟩⟩, .operator (⟨16937, 1⟩, ⟨804, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨59939⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event16991 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67298⟩⟩, .operator (⟨16937, 1⟩, ⟨804, 2⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨56959⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event16992 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67298⟩⟩, .operator (⟨16937, 1⟩, ⟨804, 3⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨53979⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event16993 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67298⟩⟩, .operator (⟨16937, 1⟩, ⟨804, 4⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨50999⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event16994 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67298⟩⟩, .operator (⟨16937, 1⟩, ⟨804, 6⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨31935⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event16995 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67298⟩⟩, .operator (⟨16937, 1⟩, ⟨804, 10⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨21915⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event16996 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67298⟩⟩, .operator (⟨16937, 1⟩, ⟨804, 14⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18695⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event16997 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67298⟩⟩, .operator (⟨16937, 1⟩, ⟨804, 17⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15890⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event16998 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67298⟩⟩, .operator (⟨16937, 0⟩, ⟨804, 5⟩), ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67293⟩⟩], [⟨.program ⟨257⟩, ⟨7235⟩⟩]⟩, (-1)⟩)

def event16999 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67298⟩⟩, .operator (⟨16937, 0⟩, ⟨804, 7⟩), ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48245⟩⟩], [⟨.program ⟨257⟩, ⟨7235⟩⟩]⟩, (1)⟩)

def event17000 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67298⟩⟩, .operator (⟨16937, 0⟩, ⟨804, 8⟩), ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45565⟩⟩], [⟨.program ⟨257⟩, ⟨7235⟩⟩]⟩, (1)⟩)

def event17001 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67298⟩⟩, .operator (⟨16937, 0⟩, ⟨804, 9⟩), ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨42888⟩⟩], [⟨.program ⟨257⟩, ⟨7235⟩⟩]⟩, (1)⟩)

def event17002 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67298⟩⟩, .operator (⟨16937, 0⟩, ⟨804, 11⟩), ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40208⟩⟩], [⟨.program ⟨257⟩, ⟨7235⟩⟩]⟩, (1)⟩)

def event17003 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67298⟩⟩, .operator (⟨16937, 0⟩, ⟨804, 12⟩), ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37525⟩⟩], [⟨.program ⟨257⟩, ⟨7235⟩⟩]⟩, (1)⟩)

def event17004 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67298⟩⟩, .operator (⟨16937, 0⟩, ⟨804, 13⟩), ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34845⟩⟩], [⟨.program ⟨257⟩, ⟨7235⟩⟩]⟩, (1)⟩)

def event17005 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67298⟩⟩, .operator (⟨16937, 0⟩, ⟨804, 15⟩), ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29188⟩⟩], [⟨.program ⟨257⟩, ⟨7235⟩⟩]⟩, (1)⟩)

def event17006 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67298⟩⟩, .operator (⟨16937, 0⟩, ⟨804, 16⟩), ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26508⟩⟩], [⟨.program ⟨257⟩, ⟨7235⟩⟩]⟩, (1)⟩)

def event17007 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67298⟩⟩, .operator (⟨16937, 0⟩, ⟨804, 18⟩), ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨65980⟩⟩], [⟨.program ⟨257⟩, ⟨7235⟩⟩]⟩, (1)⟩)

def event17008 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67298⟩⟩, .operator (⟨16937, 0⟩, ⟨804, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨62919⟩⟩], [⟨.program ⟨257⟩, ⟨7235⟩⟩]⟩, (1)⟩)

def event17009 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67298⟩⟩, .operator (⟨16937, 0⟩, ⟨804, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨59939⟩⟩], [⟨.program ⟨257⟩, ⟨7235⟩⟩]⟩, (1)⟩)

def event17010 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67298⟩⟩, .operator (⟨16937, 0⟩, ⟨804, 2⟩), ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨56959⟩⟩], [⟨.program ⟨257⟩, ⟨7235⟩⟩]⟩, (1)⟩)

def event17011 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67298⟩⟩, .operator (⟨16937, 0⟩, ⟨804, 3⟩), ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨53979⟩⟩], [⟨.program ⟨257⟩, ⟨7235⟩⟩]⟩, (1)⟩)

def event17012 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67298⟩⟩, .operator (⟨16937, 0⟩, ⟨804, 4⟩), ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨50999⟩⟩], [⟨.program ⟨257⟩, ⟨7235⟩⟩]⟩, (1)⟩)

def event17013 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67298⟩⟩, .operator (⟨16937, 0⟩, ⟨804, 6⟩), ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨31935⟩⟩], [⟨.program ⟨257⟩, ⟨7235⟩⟩]⟩, (1)⟩)

def event17014 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67298⟩⟩, .operator (⟨16937, 0⟩, ⟨804, 10⟩), ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨21915⟩⟩], [⟨.program ⟨257⟩, ⟨7235⟩⟩]⟩, (1)⟩)

def event17015 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67298⟩⟩, .operator (⟨16937, 0⟩, ⟨804, 14⟩), ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18695⟩⟩], [⟨.program ⟨257⟩, ⟨7235⟩⟩]⟩, (1)⟩)

def event17016 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67298⟩⟩, .operator (⟨16937, 0⟩, ⟨804, 17⟩), ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15890⟩⟩], [⟨.program ⟨257⟩, ⟨7235⟩⟩]⟩, (1)⟩)

def exact17017RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨62919⟩⟩], [⟨.program ⟨257⟩, ⟨7235⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨59939⟩⟩], [⟨.program ⟨257⟩, ⟨7235⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨56959⟩⟩], [⟨.program ⟨257⟩, ⟨7235⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨53979⟩⟩], [⟨.program ⟨257⟩, ⟨7235⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨50999⟩⟩], [⟨.program ⟨257⟩, ⟨7235⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67293⟩⟩], [⟨.program ⟨257⟩, ⟨7235⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨31935⟩⟩], [⟨.program ⟨257⟩, ⟨7235⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48245⟩⟩], [⟨.program ⟨257⟩, ⟨7235⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45565⟩⟩], [⟨.program ⟨257⟩, ⟨7235⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨42888⟩⟩], [⟨.program ⟨257⟩, ⟨7235⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨21915⟩⟩], [⟨.program ⟨257⟩, ⟨7235⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40208⟩⟩], [⟨.program ⟨257⟩, ⟨7235⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37525⟩⟩], [⟨.program ⟨257⟩, ⟨7235⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34845⟩⟩], [⟨.program ⟨257⟩, ⟨7235⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18695⟩⟩], [⟨.program ⟨257⟩, ⟨7235⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29188⟩⟩], [⟨.program ⟨257⟩, ⟨7235⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26508⟩⟩], [⟨.program ⟨257⟩, ⟨7235⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15890⟩⟩], [⟨.program ⟨257⟩, ⟨7235⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨65980⟩⟩], [⟨.program ⟨257⟩, ⟨7235⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨62919⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨59939⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨56959⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨53979⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨50999⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67293⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨31935⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48245⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45565⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨42888⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨21915⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40208⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37525⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34845⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18695⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29188⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26508⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15890⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨65980⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact17017RawTermsValid :
    exact17017RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17017 : Event := .resultExact (⟨.program ⟨257⟩, ⟨67298⟩⟩) exact17017RawTerms .large 16940 (.finite 6902113630329048043564518670336) (some (16978))

def event17018 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68777⟩⟩) 0 ⟨66003⟩ 533

def event17019 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨68777⟩⟩) (.authority (.programFamilyFact))

def event17020 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨68777⟩⟩) (.finite 1152)

def event17021 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68778⟩⟩) 0 ⟨7177⟩ 15500

def event17022 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68778⟩⟩) 1 ⟨68777⟩ 17020

def event17023 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨68778⟩⟩) (.authority (.operator))

def exact17024RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨68778⟩⟩]⟩, (1)⟩]

theorem exact17024RawTermsValid :
    exact17024RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17024 : Event := .resultExact (⟨.program ⟨257⟩, ⟨68778⟩⟩) exact17024RawTerms .large 17023 .exactZero (none)

def event17025 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70968⟩⟩) 0 ⟨68778⟩ 17024

def event17026 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70968⟩⟩) (.authority (.operator))

def exact17027RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨70968⟩⟩]⟩, (1)⟩]

theorem exact17027RawTermsValid :
    exact17027RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17027 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70968⟩⟩) exact17027RawTerms (.finite 8192) 17026 .exactZero (none)

def event17028 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49221⟩⟩) 0 ⟨48079⟩ 68

def event17029 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49221⟩⟩) (.authority (.programFamilyFact))

def event17030 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨49221⟩⟩) (.finite 3720)

def event17031 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49223⟩⟩) 0 ⟨7177⟩ 15500

def event17032 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49223⟩⟩) 1 ⟨49221⟩ 17030

def event17033 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49223⟩⟩) (.authority (.operator))

def exact17034RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨49223⟩⟩]⟩, (1)⟩]

theorem exact17034RawTermsValid :
    exact17034RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17034 : Event := .resultExact (⟨.program ⟨257⟩, ⟨49223⟩⟩) exact17034RawTerms .large 17033 .exactZero (none)

def event17035 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49811⟩⟩) 0 ⟨49223⟩ 17034

def event17036 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49811⟩⟩) (.authority (.operator))

def exact17037RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨49811⟩⟩]⟩, (1)⟩]

theorem exact17037RawTermsValid :
    exact17037RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17037 : Event := .resultExact (⟨.program ⟨257⟩, ⟨49811⟩⟩) exact17037RawTerms (.finite 8192) 17036 .exactZero (none)

def event17038 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49096⟩⟩) 0 ⟨47628⟩ 62

def event17039 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49096⟩⟩) (.authority (.programFamilyFact))

def event17040 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨49096⟩⟩) (.finite 3720)

def event17041 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49097⟩⟩) 0 ⟨7177⟩ 15500

def event17042 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49097⟩⟩) 1 ⟨49096⟩ 17040

def event17043 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49097⟩⟩) (.authority (.operator))

def exact17044RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨49097⟩⟩]⟩, (1)⟩]

theorem exact17044RawTermsValid :
    exact17044RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17044 : Event := .resultExact (⟨.program ⟨257⟩, ⟨49097⟩⟩) exact17044RawTerms .large 17043 .exactZero (none)

def event17045 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49563⟩⟩) 0 ⟨49097⟩ 17044

def event17046 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49563⟩⟩) (.authority (.operator))

def exact17047RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨49563⟩⟩]⟩, (1)⟩]

theorem exact17047RawTermsValid :
    exact17047RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17047 : Event := .resultExact (⟨.program ⟨257⟩, ⟨49563⟩⟩) exact17047RawTerms (.finite 8192) 17046 .exactZero (none)

def event17048 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨11⟩⟩) (.authority (.operator))

def exact17049RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨11⟩⟩]⟩, (1)⟩]

theorem exact17049RawTermsValid :
    exact17049RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17049 : Event := .resultExact (⟨.program ⟨257⟩, ⟨11⟩⟩) exact17049RawTerms (.finite 26) 17048 .exactZero (none)

def event17050 : Event := .predecessor (⟨.program ⟨257⟩, ⟨111⟩⟩) 0 ⟨11⟩ 17049

def event17051 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨111⟩⟩) (.identity (.predecessor 0 17050 .coefficient))

def exact17052RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨111⟩⟩]⟩, (1)⟩]

theorem exact17052RawTermsValid :
    exact17052RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17052 : Event := .resultExact (⟨.program ⟨257⟩, ⟨111⟩⟩) exact17052RawTerms (.finite 26) 17051 .exactZero (none)

def event17053 : Event := .predecessor (⟨.program ⟨257⟩, ⟨6914⟩⟩) 0 ⟨5441⟩ 16922

def event17054 : Event := .predecessor (⟨.program ⟨257⟩, ⟨6914⟩⟩) 1 ⟨6908⟩ 2

def event17055 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6914⟩⟩) (.product (.predecessor 0 17053 .coefficient) (.predecessor 1 17054 .coefficient) (⟨false, false, none, none, none⟩))

def event17056 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨6914⟩⟩, .operator (⟨16922, 0⟩, ⟨2, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact17057RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact17057RawTermsValid :
    exact17057RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17057 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6914⟩⟩) exact17057RawTerms .large 17055 .exactZero (none)

def event17058 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47629⟩⟩) 0 ⟨47626⟩ 51

def event17059 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47629⟩⟩) 1 ⟨6914⟩ 17057

def event17060 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47629⟩⟩) (.tensor (.predecessor 0 17058 .coefficient) (.predecessor 1 17059 .coefficient) true false)

def event17061 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨47629⟩⟩, .operator (⟨51, 0⟩, ⟨17057, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨47626⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact17062RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨47626⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact17062RawTermsValid :
    exact17062RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17062 : Event := .resultExact (⟨.program ⟨257⟩, ⟨47629⟩⟩) exact17062RawTerms .large 17060 .exactZero (none)

def event17063 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7285⟩⟩) 0 ⟨7178⟩ 15893

def event17064 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7285⟩⟩) (.identity (.predecessor 0 17063 .coefficient))

def exact17065RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7285⟩⟩]⟩, (1)⟩]

theorem exact17065RawTermsValid :
    exact17065RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17065 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7285⟩⟩) exact17065RawTerms .large 17064 .exactZero (none)

def event17066 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7603⟩⟩) 0 ⟨5441⟩ 16922

def event17067 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7603⟩⟩) 1 ⟨7285⟩ 17065

def event17068 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7603⟩⟩) (.product (.predecessor 0 17066 .coefficient) (.predecessor 1 17067 .coefficient) (⟨false, false, none, none, none⟩))

def event17069 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨7603⟩⟩, .operator (⟨16922, 0⟩, ⟨17065, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7285⟩⟩]⟩, (1)⟩)

def exact17070RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7285⟩⟩]⟩, (1)⟩]

theorem exact17070RawTermsValid :
    exact17070RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17070 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7603⟩⟩) exact17070RawTerms .large 17068 .exactZero (none)

def event17071 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47630⟩⟩) 0 ⟨7603⟩ 17070

def event17072 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47630⟩⟩) 1 ⟨47629⟩ 17062

def event17073 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47630⟩⟩) (.sum [.predecessor 0 17071 .coefficient, .predecessor 1 17072 .coefficient])

def exact17074RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7285⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨47626⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact17074RawTermsValid :
    exact17074RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17074 : Event := .resultExact (⟨.program ⟨257⟩, ⟨47630⟩⟩) exact17074RawTerms .large 17073 .exactZero (none)

def event17075 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47631⟩⟩) 0 ⟨47630⟩ 17074

def event17076 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47631⟩⟩) 1 ⟨111⟩ 17052

def event17077 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47631⟩⟩) (.sum [.predecessor 0 17075 .coefficient, .predecessor 1 17076 .coefficient])

def event17078 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47631⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨111⟩⟩]⟩) [⟨.result 17052 .coefficient, false, none⟩])

def event17079 : Event := .survivorFold (1) 17078

def exact17080RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7285⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨47626⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact17080RawTermsValid :
    exact17080RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17080 : Event := .resultExact (⟨.program ⟨257⟩, ⟨47631⟩⟩) exact17080RawTerms .large 17077 (.finite 26) (some (17078))

def event17081 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47632⟩⟩) 0 ⟨47631⟩ 17080

def event17082 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47632⟩⟩) 1 ⟨14951⟩ 54

def event17083 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47632⟩⟩) (.product (.predecessor 0 17081 .coefficient) (.predecessor 1 17082 .coefficient) (⟨false, true, none, none, some 1⟩))

def event17084 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47632⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨14951⟩⟩], []⟩) [⟨.result 54 .coefficient, true, some 1⟩])

def event17085 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47632⟩⟩) (.product (.result 17080 .summary) (.transfer 17084) (⟨false, false, none, none, none⟩))

def event17086 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨47632⟩⟩, .operator (⟨17080, 1⟩, ⟨54, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨14951⟩⟩, ⟨.program ⟨257⟩, ⟨47626⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event17087 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨47632⟩⟩, .operator (⟨17080, 0⟩, ⟨54, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨14951⟩⟩], [⟨.program ⟨257⟩, ⟨7285⟩⟩]⟩, (1)⟩)

def exact17088RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨14951⟩⟩], [⟨.program ⟨257⟩, ⟨7285⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨14951⟩⟩, ⟨.program ⟨257⟩, ⟨47626⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact17088RawTermsValid :
    exact17088RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17088 : Event := .resultExact (⟨.program ⟨257⟩, ⟨47632⟩⟩) exact17088RawTerms .large 17083 (.finite 51118080) (some (17085))

def event17089 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9565⟩⟩) 0 ⟨7285⟩ 17065

def event17090 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9565⟩⟩) (.authority (.operator))

def exact17091RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨9565⟩⟩]⟩, (1)⟩]

theorem exact17091RawTermsValid :
    exact17091RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17091 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9565⟩⟩) exact17091RawTerms (.finite 8192) 17090 .exactZero (none)

def event17092 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9566⟩⟩) 0 ⟨9565⟩ 17091

def event17093 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9566⟩⟩) 1 ⟨2370⟩ 4

def event17094 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9566⟩⟩) (.scale (.predecessor 0 17092 .coefficient) (.value (.predecessor 1 17093 .coefficient)))

def exact17095RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨9565⟩⟩]⟩, (1)⟩]

theorem exact17095RawTermsValid :
    exact17095RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17095 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9566⟩⟩) exact17095RawTerms (.finite 8192) 17094 .exactZero (none)

def event17096 : Event := .predecessor (⟨.program ⟨257⟩, ⟨128⟩⟩) 0 ⟨11⟩ 17049

def event17097 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨128⟩⟩) (.identity (.predecessor 0 17096 .coefficient))

def exact17098RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨128⟩⟩]⟩, (1)⟩]

theorem exact17098RawTermsValid :
    exact17098RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17098 : Event := .resultExact (⟨.program ⟨257⟩, ⟨128⟩⟩) exact17098RawTerms (.finite 26) 17097 .exactZero (none)

def event17099 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14952⟩⟩) 0 ⟨14951⟩ 54

def event17100 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14952⟩⟩) 1 ⟨6914⟩ 17057

def event17101 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14952⟩⟩) (.tensor (.predecessor 0 17099 .coefficient) (.predecessor 1 17100 .coefficient) true false)

def event17102 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨14952⟩⟩, .operator (⟨54, 0⟩, ⟨17057, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨14951⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact17103RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨14951⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact17103RawTermsValid :
    exact17103RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17103 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14952⟩⟩) exact17103RawTerms .large 17101 .exactZero (none)

def event17104 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7302⟩⟩) 0 ⟨7178⟩ 15893

def event17105 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7302⟩⟩) (.identity (.predecessor 0 17104 .coefficient))

def exact17106RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7302⟩⟩]⟩, (1)⟩]

theorem exact17106RawTermsValid :
    exact17106RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17106 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7302⟩⟩) exact17106RawTerms .large 17105 .exactZero (none)

def event17107 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7620⟩⟩) 0 ⟨5441⟩ 16922

def event17108 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7620⟩⟩) 1 ⟨7302⟩ 17106

def event17109 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7620⟩⟩) (.product (.predecessor 0 17107 .coefficient) (.predecessor 1 17108 .coefficient) (⟨false, false, none, none, none⟩))

def event17110 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨7620⟩⟩, .operator (⟨16922, 0⟩, ⟨17106, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7302⟩⟩]⟩, (1)⟩)

def exact17111RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7302⟩⟩]⟩, (1)⟩]

theorem exact17111RawTermsValid :
    exact17111RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17111 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7620⟩⟩) exact17111RawTerms .large 17109 .exactZero (none)

def event17112 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14953⟩⟩) 0 ⟨7620⟩ 17111

def event17113 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14953⟩⟩) 1 ⟨14952⟩ 17103

def event17114 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14953⟩⟩) (.sum [.predecessor 0 17112 .coefficient, .predecessor 1 17113 .coefficient])

def exact17115RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7302⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨14951⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact17115RawTermsValid :
    exact17115RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17115 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14953⟩⟩) exact17115RawTerms .large 17114 .exactZero (none)

def event17116 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14954⟩⟩) 0 ⟨14953⟩ 17115

def event17117 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14954⟩⟩) 1 ⟨128⟩ 17098

def event17118 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14954⟩⟩) (.sum [.predecessor 0 17116 .coefficient, .predecessor 1 17117 .coefficient])

def event17119 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14954⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨128⟩⟩]⟩) [⟨.result 17098 .coefficient, false, none⟩])

def event17120 : Event := .survivorFold (1) 17119

def exact17121RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7302⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨14951⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact17121RawTermsValid :
    exact17121RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17121 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14954⟩⟩) exact17121RawTerms .large 17118 (.finite 26) (some (17119))

def event17122 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14955⟩⟩) 0 ⟨14954⟩ 17121

def event17123 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14955⟩⟩) 1 ⟨9566⟩ 17095

def event17124 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14955⟩⟩) (.product (.predecessor 0 17122 .coefficient) (.predecessor 1 17123 .coefficient) (⟨false, false, none, none, none⟩))

def event17125 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14955⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨9565⟩⟩]⟩) [⟨.result 17091 .coefficient, false, none⟩])

def event17126 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14955⟩⟩) (.product (.result 17121 .summary) (.transfer 17125) (⟨false, false, none, none, none⟩))

def event17127 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨14955⟩⟩, .operator (⟨17121, 1⟩, ⟨17095, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨14951⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩]⟩, (-1)⟩)

def event17128 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨14955⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨14951⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨9565⟩⟩) ⟨7285⟩ 17065)

def event17129 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨14955⟩⟩, .relation 17128 0, ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨14951⟩⟩], [⟨.program ⟨257⟩, ⟨7285⟩⟩]⟩, (-1)⟩)

def event17130 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨14955⟩⟩, .operator (⟨17121, 0⟩, ⟨17095, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩]⟩, (1)⟩)

def exact17131RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨14951⟩⟩], [⟨.program ⟨257⟩, ⟨7285⟩⟩]⟩, (-1)⟩]

theorem exact17131RawTermsValid :
    exact17131RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17131 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14955⟩⟩) exact17131RawTerms .large 17124 (.finite 279172874240) (some (17126))

def event17132 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47633⟩⟩) 0 ⟨14955⟩ 17131

def event17133 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47633⟩⟩) 1 ⟨47632⟩ 17088

def event17134 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47633⟩⟩) (.sum [.predecessor 0 17132 .coefficient, .predecessor 1 17133 .coefficient])

def event17135 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨47633⟩⟩, .operator (⟨17131, 1⟩, ⟨17088, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨14951⟩⟩], [⟨.program ⟨257⟩, ⟨7285⟩⟩]⟩, (1)⟩)

def event17136 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47633⟩⟩) (.sum [.result 17131 .summary, .result 17088 .summary])

def exact17137RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨14951⟩⟩, ⟨.program ⟨257⟩, ⟨47626⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact17137RawTermsValid :
    exact17137RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17137 : Event := .resultExact (⟨.program ⟨257⟩, ⟨47633⟩⟩) exact17137RawTerms .large 17134 (.finite 279223992320) (some (17136))

def event17138 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49564⟩⟩) 0 ⟨47633⟩ 17137

def event17139 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49564⟩⟩) 1 ⟨49563⟩ 17047

def event17140 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49564⟩⟩) (.product (.predecessor 0 17138 .coefficient) (.predecessor 1 17139 .coefficient) (⟨false, false, none, none, none⟩))

def event17141 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49564⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨49563⟩⟩]⟩) [⟨.result 17047 .coefficient, false, none⟩])

def event17142 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49564⟩⟩) (.product (.result 17137 .summary) (.transfer 17141) (⟨false, false, none, none, none⟩))

def event17143 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨49564⟩⟩, .operator (⟨17137, 1⟩, ⟨17047, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨14951⟩⟩, ⟨.program ⟨257⟩, ⟨47626⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨49563⟩⟩]⟩, (-1)⟩)

def event17144 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨49564⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨14951⟩⟩, ⟨.program ⟨257⟩, ⟨47626⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨49563⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨49563⟩⟩) ⟨49097⟩ 17044)

def event17145 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨49564⟩⟩, .relation 17144 0, ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨14951⟩⟩, ⟨.program ⟨257⟩, ⟨47626⟩⟩], [⟨.program ⟨257⟩, ⟨49097⟩⟩]⟩, (-1)⟩)

def event17146 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨49564⟩⟩, .operator (⟨17137, 0⟩, ⟨17047, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49563⟩⟩]⟩, (1)⟩)

def exact17147RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49563⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨14951⟩⟩, ⟨.program ⟨257⟩, ⟨47626⟩⟩], [⟨.program ⟨257⟩, ⟨49097⟩⟩]⟩, (-1)⟩]

theorem exact17147RawTermsValid :
    exact17147RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17147 : Event := .resultExact (⟨.program ⟨257⟩, ⟨49564⟩⟩) exact17147RawTerms .large 17140 (.finite 2998144788182387916800) (some (17142))

def event17148 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48502⟩⟩) 0 ⟨47628⟩ 62

def event17149 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48502⟩⟩) (.authority (.relationPreimageSource ⟨54⟩))

def exact17150RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨48502⟩⟩]⟩, (1)⟩]

theorem exact17150RawTermsValid :
    exact17150RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17150 : Event := .resultExact (⟨.program ⟨257⟩, ⟨48502⟩⟩) exact17150RawTerms (.finite 5647228698) 17149 .exactZero (none)

def event17151 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48504⟩⟩) 0 ⟨48502⟩ 17150

def eventLeaf1056 : Array AnnotatedEvent := #[
  { event := event16896
    frameStart := 0 },
  { event := event16897
    frameStart := 0 },
  { event := event16898
    frameStart := 0 },
  { event := event16899
    frameStart := 0 },
  { event := event16900
    frameStart := 0 },
  { event := event16901
    frameStart := 0 },
  { event := event16902
    frameStart := 0 },
  { event := event16903
    frameStart := 0 },
  { event := event16904
    frameStart := 0 },
  { event := event16905
    frameStart := 0 },
  { event := event16906
    frameStart := 0 },
  { event := event16907
    frameStart := 0 },
  { event := event16908
    frameStart := 0 },
  { event := event16909
    frameStart := 0 },
  { event := event16910
    frameStart := 0 },
  { event := event16911
    frameStart := 0 }
]

def eventLeaf1057 : Array AnnotatedEvent := #[
  { event := event16912
    frameStart := 0 },
  { event := event16913
    frameStart := 0 },
  { event := event16914
    frameStart := 0 },
  { event := event16915
    frameStart := 0 },
  { event := event16916
    frameStart := 0 },
  { event := event16917
    frameStart := 0 },
  { event := event16918
    frameStart := 0 },
  { event := event16919
    frameStart := 0 },
  { event := event16920
    frameStart := 0 },
  { event := event16921
    frameStart := 0 },
  { event := event16922
    frameStart := 0 },
  { event := event16923
    frameStart := 0 },
  { event := event16924
    frameStart := 0 },
  { event := event16925
    frameStart := 0 },
  { event := event16926
    frameStart := 0 },
  { event := event16927
    frameStart := 0 }
]

def eventLeaf1058 : Array AnnotatedEvent := #[
  { event := event16928
    frameStart := 0 },
  { event := event16929
    frameStart := 0 },
  { event := event16930
    frameStart := 0 },
  { event := event16931
    frameStart := 0 },
  { event := event16932
    frameStart := 0 },
  { event := event16933
    frameStart := 0 },
  { event := event16934
    frameStart := 0 },
  { event := event16935
    frameStart := 0 },
  { event := event16936
    frameStart := 0 },
  { event := event16937
    frameStart := 0 },
  { event := event16938
    frameStart := 0 },
  { event := event16939
    frameStart := 0 },
  { event := event16940
    frameStart := 0 },
  { event := event16941
    frameStart := 0 },
  { event := event16942
    frameStart := 0 },
  { event := event16943
    frameStart := 0 }
]

def eventLeaf1059 : Array AnnotatedEvent := #[
  { event := event16944
    frameStart := 0 },
  { event := event16945
    frameStart := 0 },
  { event := event16946
    frameStart := 0 },
  { event := event16947
    frameStart := 0 },
  { event := event16948
    frameStart := 0 },
  { event := event16949
    frameStart := 0 },
  { event := event16950
    frameStart := 0 },
  { event := event16951
    frameStart := 0 },
  { event := event16952
    frameStart := 0 },
  { event := event16953
    frameStart := 0 },
  { event := event16954
    frameStart := 0 },
  { event := event16955
    frameStart := 0 },
  { event := event16956
    frameStart := 0 },
  { event := event16957
    frameStart := 0 },
  { event := event16958
    frameStart := 0 },
  { event := event16959
    frameStart := 0 }
]

def eventLeaf1060 : Array AnnotatedEvent := #[
  { event := event16960
    frameStart := 0 },
  { event := event16961
    frameStart := 0 },
  { event := event16962
    frameStart := 0 },
  { event := event16963
    frameStart := 0 },
  { event := event16964
    frameStart := 0 },
  { event := event16965
    frameStart := 0 },
  { event := event16966
    frameStart := 0 },
  { event := event16967
    frameStart := 0 },
  { event := event16968
    frameStart := 0 },
  { event := event16969
    frameStart := 0 },
  { event := event16970
    frameStart := 0 },
  { event := event16971
    frameStart := 0 },
  { event := event16972
    frameStart := 0 },
  { event := event16973
    frameStart := 0 },
  { event := event16974
    frameStart := 0 },
  { event := event16975
    frameStart := 0 }
]

def eventLeaf1061 : Array AnnotatedEvent := #[
  { event := event16976
    frameStart := 0 },
  { event := event16977
    frameStart := 0 },
  { event := event16978
    frameStart := 0 },
  { event := event16979
    frameStart := 0 },
  { event := event16980
    frameStart := 0 },
  { event := event16981
    frameStart := 0 },
  { event := event16982
    frameStart := 0 },
  { event := event16983
    frameStart := 0 },
  { event := event16984
    frameStart := 0 },
  { event := event16985
    frameStart := 0 },
  { event := event16986
    frameStart := 0 },
  { event := event16987
    frameStart := 0 },
  { event := event16988
    frameStart := 0 },
  { event := event16989
    frameStart := 0 },
  { event := event16990
    frameStart := 0 },
  { event := event16991
    frameStart := 0 }
]

def eventLeaf1062 : Array AnnotatedEvent := #[
  { event := event16992
    frameStart := 0 },
  { event := event16993
    frameStart := 0 },
  { event := event16994
    frameStart := 0 },
  { event := event16995
    frameStart := 0 },
  { event := event16996
    frameStart := 0 },
  { event := event16997
    frameStart := 0 },
  { event := event16998
    frameStart := 0 },
  { event := event16999
    frameStart := 0 },
  { event := event17000
    frameStart := 0 },
  { event := event17001
    frameStart := 0 },
  { event := event17002
    frameStart := 0 },
  { event := event17003
    frameStart := 0 },
  { event := event17004
    frameStart := 0 },
  { event := event17005
    frameStart := 0 },
  { event := event17006
    frameStart := 0 },
  { event := event17007
    frameStart := 0 }
]

def eventLeaf1063 : Array AnnotatedEvent := #[
  { event := event17008
    frameStart := 0 },
  { event := event17009
    frameStart := 0 },
  { event := event17010
    frameStart := 0 },
  { event := event17011
    frameStart := 0 },
  { event := event17012
    frameStart := 0 },
  { event := event17013
    frameStart := 0 },
  { event := event17014
    frameStart := 0 },
  { event := event17015
    frameStart := 0 },
  { event := event17016
    frameStart := 0 },
  { event := event17017
    frameStart := 0 },
  { event := event17018
    frameStart := 0 },
  { event := event17019
    frameStart := 0 },
  { event := event17020
    frameStart := 0 },
  { event := event17021
    frameStart := 0 },
  { event := event17022
    frameStart := 0 },
  { event := event17023
    frameStart := 0 }
]

def eventLeaf1064 : Array AnnotatedEvent := #[
  { event := event17024
    frameStart := 0 },
  { event := event17025
    frameStart := 0 },
  { event := event17026
    frameStart := 0 },
  { event := event17027
    frameStart := 0 },
  { event := event17028
    frameStart := 0 },
  { event := event17029
    frameStart := 0 },
  { event := event17030
    frameStart := 0 },
  { event := event17031
    frameStart := 0 },
  { event := event17032
    frameStart := 0 },
  { event := event17033
    frameStart := 0 },
  { event := event17034
    frameStart := 0 },
  { event := event17035
    frameStart := 0 },
  { event := event17036
    frameStart := 0 },
  { event := event17037
    frameStart := 0 },
  { event := event17038
    frameStart := 0 },
  { event := event17039
    frameStart := 0 }
]

def eventLeaf1065 : Array AnnotatedEvent := #[
  { event := event17040
    frameStart := 0 },
  { event := event17041
    frameStart := 0 },
  { event := event17042
    frameStart := 0 },
  { event := event17043
    frameStart := 0 },
  { event := event17044
    frameStart := 0 },
  { event := event17045
    frameStart := 0 },
  { event := event17046
    frameStart := 0 },
  { event := event17047
    frameStart := 0 },
  { event := event17048
    frameStart := 0 },
  { event := event17049
    frameStart := 0 },
  { event := event17050
    frameStart := 0 },
  { event := event17051
    frameStart := 0 },
  { event := event17052
    frameStart := 0 },
  { event := event17053
    frameStart := 0 },
  { event := event17054
    frameStart := 0 },
  { event := event17055
    frameStart := 0 }
]

def eventLeaf1066 : Array AnnotatedEvent := #[
  { event := event17056
    frameStart := 0 },
  { event := event17057
    frameStart := 0 },
  { event := event17058
    frameStart := 0 },
  { event := event17059
    frameStart := 0 },
  { event := event17060
    frameStart := 0 },
  { event := event17061
    frameStart := 0 },
  { event := event17062
    frameStart := 0 },
  { event := event17063
    frameStart := 0 },
  { event := event17064
    frameStart := 0 },
  { event := event17065
    frameStart := 0 },
  { event := event17066
    frameStart := 0 },
  { event := event17067
    frameStart := 0 },
  { event := event17068
    frameStart := 0 },
  { event := event17069
    frameStart := 0 },
  { event := event17070
    frameStart := 0 },
  { event := event17071
    frameStart := 0 }
]

def eventLeaf1067 : Array AnnotatedEvent := #[
  { event := event17072
    frameStart := 0 },
  { event := event17073
    frameStart := 0 },
  { event := event17074
    frameStart := 0 },
  { event := event17075
    frameStart := 0 },
  { event := event17076
    frameStart := 0 },
  { event := event17077
    frameStart := 0 },
  { event := event17078
    frameStart := 0 },
  { event := event17079
    frameStart := 0 },
  { event := event17080
    frameStart := 0 },
  { event := event17081
    frameStart := 0 },
  { event := event17082
    frameStart := 0 },
  { event := event17083
    frameStart := 0 },
  { event := event17084
    frameStart := 0 },
  { event := event17085
    frameStart := 0 },
  { event := event17086
    frameStart := 0 },
  { event := event17087
    frameStart := 0 }
]

def eventLeaf1068 : Array AnnotatedEvent := #[
  { event := event17088
    frameStart := 0 },
  { event := event17089
    frameStart := 0 },
  { event := event17090
    frameStart := 0 },
  { event := event17091
    frameStart := 0 },
  { event := event17092
    frameStart := 0 },
  { event := event17093
    frameStart := 0 },
  { event := event17094
    frameStart := 0 },
  { event := event17095
    frameStart := 0 },
  { event := event17096
    frameStart := 0 },
  { event := event17097
    frameStart := 0 },
  { event := event17098
    frameStart := 0 },
  { event := event17099
    frameStart := 0 },
  { event := event17100
    frameStart := 0 },
  { event := event17101
    frameStart := 0 },
  { event := event17102
    frameStart := 0 },
  { event := event17103
    frameStart := 0 }
]

def eventLeaf1069 : Array AnnotatedEvent := #[
  { event := event17104
    frameStart := 0 },
  { event := event17105
    frameStart := 0 },
  { event := event17106
    frameStart := 0 },
  { event := event17107
    frameStart := 0 },
  { event := event17108
    frameStart := 0 },
  { event := event17109
    frameStart := 0 },
  { event := event17110
    frameStart := 0 },
  { event := event17111
    frameStart := 0 },
  { event := event17112
    frameStart := 0 },
  { event := event17113
    frameStart := 0 },
  { event := event17114
    frameStart := 0 },
  { event := event17115
    frameStart := 0 },
  { event := event17116
    frameStart := 0 },
  { event := event17117
    frameStart := 0 },
  { event := event17118
    frameStart := 0 },
  { event := event17119
    frameStart := 0 }
]

def eventLeaf1070 : Array AnnotatedEvent := #[
  { event := event17120
    frameStart := 0 },
  { event := event17121
    frameStart := 0 },
  { event := event17122
    frameStart := 0 },
  { event := event17123
    frameStart := 0 },
  { event := event17124
    frameStart := 0 },
  { event := event17125
    frameStart := 0 },
  { event := event17126
    frameStart := 0 },
  { event := event17127
    frameStart := 0 },
  { event := event17128
    frameStart := 0 },
  { event := event17129
    frameStart := 0 },
  { event := event17130
    frameStart := 0 },
  { event := event17131
    frameStart := 0 },
  { event := event17132
    frameStart := 0 },
  { event := event17133
    frameStart := 0 },
  { event := event17134
    frameStart := 0 },
  { event := event17135
    frameStart := 0 }
]

def eventLeaf1071 : Array AnnotatedEvent := #[
  { event := event17136
    frameStart := 0 },
  { event := event17137
    frameStart := 0 },
  { event := event17138
    frameStart := 0 },
  { event := event17139
    frameStart := 0 },
  { event := event17140
    frameStart := 0 },
  { event := event17141
    frameStart := 0 },
  { event := event17142
    frameStart := 0 },
  { event := event17143
    frameStart := 0 },
  { event := event17144
    frameStart := 0 },
  { event := event17145
    frameStart := 0 },
  { event := event17146
    frameStart := 0 },
  { event := event17147
    frameStart := 0 },
  { event := event17148
    frameStart := 0 },
  { event := event17149
    frameStart := 0 },
  { event := event17150
    frameStart := 0 },
  { event := event17151
    frameStart := 0 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events066
