import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events308

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event78848 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6⟩⟩) (.authority (.operator))

def exact78849RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩]⟩, (1)⟩]

theorem exact78849RawTermsValid :
    exact78849RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78849 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6⟩⟩) exact78849RawTerms .large 78848 .exactZero (none)

def event78850 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20605⟩⟩) 0 ⟨6⟩ 78849

def event78851 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20605⟩⟩) 1 ⟨20604⟩ 78847

def event78852 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20605⟩⟩) (.product (.predecessor 0 78850 .coefficient) (.predecessor 1 78851 .coefficient) (⟨false, false, none, none, none⟩))

def event78853 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20605⟩⟩, .operator (⟨78849, 0⟩, ⟨78847, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20604⟩⟩]⟩, (1)⟩)

def exact78854RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20604⟩⟩]⟩, (1)⟩]

theorem exact78854RawTermsValid :
    exact78854RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78854 : Event := .resultExact (⟨.program ⟨214⟩, ⟨20605⟩⟩) exact78854RawTerms .large 78852 .exactZero (none)

def event78855 : Event := .preFoldPolynomial 78854 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20604⟩⟩]⟩, (1)⟩] .exactZero none

def exact78856RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20604⟩⟩]⟩, (1)⟩]

def event78856 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨20605⟩⟩) 78855 exact78856RawTerms .large 78852 .exactZero (none)

def event78857 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨26767⟩⟩)

def event78858 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event78859 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event78860 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨360⟩⟩) (.authority (.operator))

def event78861 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨360⟩⟩) (.finite 2)

def event78862 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event78863 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event78864 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event78865 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event78866 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 78865

def event78867 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 78863

def event78868 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 78866 .coefficient) (.value (.predecessor 1 78867 .coefficient)))

def event78869 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event78870 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5504⟩⟩) 0 ⟨5503⟩ 78869

def event78871 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5504⟩⟩) 1 ⟨360⟩ 78861

def event78872 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5504⟩⟩) (.sum [.predecessor 0 78870 .coefficient, .predecessor 1 78871 .coefficient])

def event78873 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5504⟩⟩) (.finite 219)

def event78874 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5530⟩⟩) 0 ⟨5504⟩ 78873

def event78875 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5530⟩⟩) 1 ⟨961⟩ 78859

def event78876 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5530⟩⟩) (.identity (.predecessor 1 78875 .coefficient))

def event78877 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5530⟩⟩) (.finite 224)

def event78878 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10969⟩⟩) 0 ⟨5530⟩ 78877

def event78879 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10969⟩⟩) (.authority (.programFamilyFact))

def exact78880RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨10969⟩⟩], []⟩, (1)⟩]

theorem exact78880RawTermsValid :
    exact78880RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78880 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10969⟩⟩) exact78880RawTerms (.finite 4) 78879 .exactZero (none)

def event78881 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10837⟩⟩) 0 ⟨5530⟩ 78877

def event78882 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10837⟩⟩) (.authority (.programFamilyFact))

def exact78883RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨10837⟩⟩], []⟩, (1)⟩]

theorem exact78883RawTermsValid :
    exact78883RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78883 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10837⟩⟩) exact78883RawTerms (.finite 4) 78882 .exactZero (none)

def event78884 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10970⟩⟩) 0 ⟨10837⟩ 78883

def event78885 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10970⟩⟩) 1 ⟨10969⟩ 78880

def event78886 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10970⟩⟩) (.product (.predecessor 0 78884 .coefficient) (.predecessor 1 78885 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event78887 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨10970⟩⟩, .operator (⟨78883, 0⟩, ⟨78880, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨10837⟩⟩, ⟨.program ⟨214⟩, ⟨10969⟩⟩], []⟩, (1)⟩)

def exact78888RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨10837⟩⟩, ⟨.program ⟨214⟩, ⟨10969⟩⟩], []⟩, (1)⟩]

theorem exact78888RawTermsValid :
    exact78888RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78888 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10970⟩⟩) exact78888RawTerms (.finite 16) 78886 .exactZero (none)

def event78889 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10971⟩⟩) 0 ⟨10970⟩ 78888

def event78890 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10971⟩⟩) (.identity (.predecessor 0 78889 .coefficient))

def event78891 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨10971⟩⟩) (.finite 16)

def event78892 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15110⟩⟩) 0 ⟨10971⟩ 78891

def event78893 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15110⟩⟩) (.authority (.programFamilyFact))

def exact78894RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15110⟩⟩], []⟩, (1)⟩]

theorem exact78894RawTermsValid :
    exact78894RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78894 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15110⟩⟩) exact78894RawTerms (.finite 4) 78893 .exactZero (none)

def event78895 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15111⟩⟩) 0 ⟨15110⟩ 78894

def event78896 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15111⟩⟩) (.identity (.predecessor 0 78895 .coefficient))

def event78897 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨15111⟩⟩) (.finite 4)

def event78898 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23842⟩⟩) 0 ⟨15111⟩ 78897

def event78899 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23842⟩⟩) (.authority (.programFamilyFact))

def event78900 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨23842⟩⟩) (.finite 3720)

def event78901 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨6689⟩⟩) .missing

def event78902 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23843⟩⟩) 0 ⟨6689⟩ 78901

def event78903 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23843⟩⟩) 1 ⟨23842⟩ 78900

def event78904 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23843⟩⟩) (.authority (.operator))

def exact78905RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨23843⟩⟩]⟩, (1)⟩]

theorem exact78905RawTermsValid :
    exact78905RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78905 : Event := .resultExact (⟨.program ⟨214⟩, ⟨23843⟩⟩) exact78905RawTerms .large 78904 .exactZero (none)

def event78906 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26761⟩⟩) 0 ⟨23843⟩ 78905

def event78907 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26761⟩⟩) (.authority (.operator))

def exact78908RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨26761⟩⟩]⟩, (1)⟩]

theorem exact78908RawTermsValid :
    exact78908RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78908 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26761⟩⟩) exact78908RawTerms (.finite 8192) 78907 .exactZero (none)

def event78909 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨110⟩⟩) (.authority (.operator))

def event78910 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨110⟩⟩) .exactZero

def event78911 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15150⟩⟩) 0 ⟨15111⟩ 78897

def event78912 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15150⟩⟩) 1 ⟨110⟩ 78910

def event78913 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15150⟩⟩) (.sum [.predecessor 0 78911 .coefficient, .predecessor 1 78912 .coefficient])

def event78914 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨15150⟩⟩) (.finite 4)

def event78915 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15151⟩⟩) 0 ⟨15150⟩ 78914

def event78916 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15151⟩⟩) (.identity (.predecessor 0 78915 .coefficient))

def exact78917RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15110⟩⟩], []⟩, (1)⟩]

theorem exact78917RawTermsValid :
    exact78917RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78917 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15151⟩⟩) exact78917RawTerms (.finite 4) 78916 .exactZero (none)

def event78918 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6544⟩⟩) (.authority (.factStore))

def exact78919RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact78919RawTermsValid :
    exact78919RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78919 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6544⟩⟩) exact78919RawTerms .large 78918 .exactZero (none)

def event78920 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15152⟩⟩) 0 ⟨6544⟩ 78919

def event78921 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15152⟩⟩) 1 ⟨15151⟩ 78917

def event78922 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15152⟩⟩) (.product (.predecessor 0 78920 .coefficient) (.predecessor 1 78921 .coefficient) (⟨false, false, none, none, none⟩))

def event78923 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨15152⟩⟩, .operator (⟨78919, 0⟩, ⟨78917, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨15110⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact78924RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15110⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact78924RawTermsValid :
    exact78924RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78924 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15152⟩⟩) exact78924RawTerms .large 78922 .exactZero (none)

def event78925 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6692⟩⟩) 0 ⟨6689⟩ 78901

def event78926 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6692⟩⟩) (.authority (.operator))

def exact78927RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6692⟩⟩]⟩, (1)⟩]

theorem exact78927RawTermsValid :
    exact78927RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78927 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6692⟩⟩) exact78927RawTerms .large 78926 .exactZero (none)

def event78928 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15153⟩⟩) 0 ⟨6692⟩ 78927

def event78929 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15153⟩⟩) 1 ⟨15152⟩ 78924

def event78930 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15153⟩⟩) (.sum [.predecessor 0 78928 .coefficient, .predecessor 1 78929 .coefficient])

def exact78931RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6692⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15110⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact78931RawTermsValid :
    exact78931RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78931 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15153⟩⟩) exact78931RawTerms .large 78930 .exactZero (none)

def event78932 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26762⟩⟩) 0 ⟨15153⟩ 78931

def event78933 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26762⟩⟩) 1 ⟨26761⟩ 78908

def event78934 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26762⟩⟩) (.product (.predecessor 0 78932 .coefficient) (.predecessor 1 78933 .coefficient) (⟨false, false, none, none, none⟩))

def event78935 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26762⟩⟩, .operator (⟨78931, 0⟩, ⟨78908, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6692⟩⟩, ⟨.program ⟨214⟩, ⟨26761⟩⟩]⟩, (1)⟩)

def event78936 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26762⟩⟩, .operator (⟨78931, 1⟩, ⟨78908, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨15110⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26761⟩⟩]⟩, (-1)⟩)

def event78937 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨26762⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨15110⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26761⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨26761⟩⟩) ⟨23843⟩ 78905)

def event78938 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26762⟩⟩, .relation 78937 0, ⟨[⟨.program ⟨214⟩, ⟨15110⟩⟩], [⟨.program ⟨214⟩, ⟨23843⟩⟩]⟩, (-1)⟩)

def exact78939RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6692⟩⟩, ⟨.program ⟨214⟩, ⟨26761⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15110⟩⟩], [⟨.program ⟨214⟩, ⟨23843⟩⟩]⟩, (-1)⟩]

theorem exact78939RawTermsValid :
    exact78939RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78939 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26762⟩⟩) exact78939RawTerms .large 78934 .exactZero (none)

def event78940 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15203⟩⟩) 0 ⟨15111⟩ 78897

def event78941 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15203⟩⟩) (.authority (.programFamilyFact))

def exact78942RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15203⟩⟩], []⟩, (1)⟩]

theorem exact78942RawTermsValid :
    exact78942RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78942 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15203⟩⟩) exact78942RawTerms (.finite 4) 78941 .exactZero (none)

def event78943 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15206⟩⟩) 0 ⟨6544⟩ 78919

def event78944 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15206⟩⟩) 1 ⟨15203⟩ 78942

def event78945 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15206⟩⟩) (.product (.predecessor 0 78943 .coefficient) (.predecessor 1 78944 .coefficient) (⟨false, true, none, none, some 1⟩))

def event78946 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨15206⟩⟩, .operator (⟨78919, 0⟩, ⟨78942, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨15203⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact78947RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15203⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact78947RawTermsValid :
    exact78947RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78947 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15206⟩⟩) exact78947RawTerms .large 78945 .exactZero (none)

def event78948 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6712⟩⟩) 0 ⟨6689⟩ 78901

def event78949 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6712⟩⟩) (.authority (.operator))

def exact78950RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6712⟩⟩]⟩, (1)⟩]

theorem exact78950RawTermsValid :
    exact78950RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78950 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6712⟩⟩) exact78950RawTerms .large 78949 .exactZero (none)

def event78951 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15207⟩⟩) 0 ⟨6712⟩ 78950

def event78952 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15207⟩⟩) 1 ⟨15206⟩ 78947

def event78953 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15207⟩⟩) (.sum [.predecessor 0 78951 .coefficient, .predecessor 1 78952 .coefficient])

def exact78954RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6712⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15203⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact78954RawTermsValid :
    exact78954RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78954 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15207⟩⟩) exact78954RawTerms .large 78953 .exactZero (none)

def event78955 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26767⟩⟩) 0 ⟨15207⟩ 78954

def event78956 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26767⟩⟩) 1 ⟨26762⟩ 78939

def event78957 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26767⟩⟩) (.sum [.predecessor 0 78955 .coefficient, .predecessor 1 78956 .coefficient])

def exact78958RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6692⟩⟩, ⟨.program ⟨214⟩, ⟨26761⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6712⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15110⟩⟩], [⟨.program ⟨214⟩, ⟨23843⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15203⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact78958RawTermsValid :
    exact78958RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78958 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26767⟩⟩) exact78958RawTerms .large 78957 .exactZero (none)

def event78959 : Event := .preFoldPolynomial 78958 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6692⟩⟩, ⟨.program ⟨214⟩, ⟨26761⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6712⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15110⟩⟩], [⟨.program ⟨214⟩, ⟨23843⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15203⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩] .exactZero none

def exact78960RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6692⟩⟩, ⟨.program ⟨214⟩, ⟨26761⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6712⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15110⟩⟩], [⟨.program ⟨214⟩, ⟨23843⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15203⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

def event78960 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨26767⟩⟩) 78959 exact78960RawTerms .large 78957 .exactZero (none)

def event78961 : Event := .specializationComputed (⟨.program ⟨214⟩, ⟨15111⟩⟩) ⟨⟨125⟩, ⟨31⟩, ⟨109⟩⟩ ⟨78803, 78961⟩

def event78962 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨20607⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20604⟩⟩]⟩) (1) 0 2 (.universal 78961 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20604⟩⟩]⟩) (none) 78960)

def event78963 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20607⟩⟩, .relation 78962 1, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6712⟩⟩]⟩, (1)⟩)

def event78964 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20607⟩⟩, .relation 78962 0, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6692⟩⟩, ⟨.program ⟨214⟩, ⟨26761⟩⟩]⟩, (-1)⟩)

def event78965 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20607⟩⟩, .relation 78962 2, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15110⟩⟩], [⟨.program ⟨214⟩, ⟨23843⟩⟩]⟩, (1)⟩)

def event78966 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20607⟩⟩, .relation 78962 3, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15203⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact78967RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6692⟩⟩, ⟨.program ⟨214⟩, ⟨26761⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6712⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15110⟩⟩], [⟨.program ⟨214⟩, ⟨23843⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15203⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact78967RawTermsValid :
    exact78967RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78967 : Event := .resultExact (⟨.program ⟨214⟩, ⟨20607⟩⟩) exact78967RawTerms .large 78799 (.finite 1811303510016) (some (78801))

def event78968 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26764⟩⟩) 0 ⟨20607⟩ 78967

def event78969 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26764⟩⟩) 1 ⟨26763⟩ 78789

def event78970 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26764⟩⟩) (.sum [.predecessor 0 78968 .coefficient, .predecessor 1 78969 .coefficient])

def event78971 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26764⟩⟩, .operator (⟨78967, 0⟩, ⟨78789, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6692⟩⟩, ⟨.program ⟨214⟩, ⟨26761⟩⟩]⟩, (1)⟩)

def event78972 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26764⟩⟩, .operator (⟨78967, 2⟩, ⟨78789, 1⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15110⟩⟩], [⟨.program ⟨214⟩, ⟨23843⟩⟩]⟩, (-1)⟩)

def event78973 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26764⟩⟩) (.sum [.result 78967 .summary, .result 78789 .summary])

def exact78974RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6712⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15203⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact78974RawTermsValid :
    exact78974RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78974 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26764⟩⟩) exact78974RawTerms .large 78970 (.finite 1291911586824442228736) (some (78973))

def event78975 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26765⟩⟩) 0 ⟨26764⟩ 78974

def event78976 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26765⟩⟩) 1 ⟨6664⟩ 5819

def event78977 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26765⟩⟩) (.product (.predecessor 0 78975 .coefficient) (.predecessor 1 78976 .coefficient) (⟨false, false, none, none, none⟩))

def event78978 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26765⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨6663⟩⟩]⟩) [⟨.result 5815 .coefficient, false, none⟩])

def event78979 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26765⟩⟩) (.product (.result 78974 .summary) (.transfer 78978) (⟨false, false, none, none, none⟩))

def event78980 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26765⟩⟩, .operator (⟨78974, 0⟩, ⟨5819, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6712⟩⟩, ⟨.program ⟨214⟩, ⟨6663⟩⟩]⟩, (1)⟩)

def event78981 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26765⟩⟩, .operator (⟨78974, 1⟩, ⟨5819, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15203⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨6663⟩⟩]⟩, (-1)⟩)

def event78982 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨26765⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15203⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨6663⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨6663⟩⟩) ⟨6603⟩ 5812)

def event78983 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26765⟩⟩, .relation 78982 0, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨6452⟩⟩, ⟨.program ⟨214⟩, ⟨15203⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact78984RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6712⟩⟩, ⟨.program ⟨214⟩, ⟨6663⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨6452⟩⟩, ⟨.program ⟨214⟩, ⟨15203⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact78984RawTermsValid :
    exact78984RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78984 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26765⟩⟩) exact78984RawTerms .large 78977 (.finite 4741336194231092170536779776) (some (78979))

def event78985 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23780⟩⟩) 0 ⟨6689⟩ 5477

def event78986 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23780⟩⟩) 1 ⟨23779⟩ 73001

def event78987 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23780⟩⟩) (.authority (.operator))

def exact78988RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨23780⟩⟩]⟩, (1)⟩]

theorem exact78988RawTermsValid :
    exact78988RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78988 : Event := .resultExact (⟨.program ⟨214⟩, ⟨23780⟩⟩) exact78988RawTerms .large 78987 .exactZero (none)

def event78989 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26544⟩⟩) 0 ⟨23780⟩ 78988

def event78990 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26544⟩⟩) (.authority (.operator))

def exact78991RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨26544⟩⟩]⟩, (1)⟩]

theorem exact78991RawTermsValid :
    exact78991RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78991 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26544⟩⟩) exact78991RawTerms (.finite 8192) 78990 .exactZero (none)

def event78992 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26546⟩⟩) 0 ⟨24985⟩ 73285

def event78993 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26546⟩⟩) 1 ⟨26544⟩ 78991

def event78994 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26546⟩⟩) (.product (.predecessor 0 78992 .coefficient) (.predecessor 1 78993 .coefficient) (⟨false, false, none, none, none⟩))

def event78995 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26546⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨26544⟩⟩]⟩) [⟨.result 78991 .coefficient, false, none⟩])

def event78996 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26546⟩⟩) (.product (.result 73285 .summary) (.transfer 78995) (⟨false, false, none, none, none⟩))

def event78997 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26546⟩⟩, .operator (⟨73285, 0⟩, ⟨78991, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6691⟩⟩, ⟨.program ⟨214⟩, ⟨26544⟩⟩]⟩, (1)⟩)

def event78998 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26546⟩⟩, .operator (⟨73285, 1⟩, ⟨78991, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨14949⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26544⟩⟩]⟩, (-1)⟩)

def event78999 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨26546⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨14949⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26544⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨26544⟩⟩) ⟨23780⟩ 78988)

def event79000 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26546⟩⟩, .relation 78999 0, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨14949⟩⟩], [⟨.program ⟨214⟩, ⟨23780⟩⟩]⟩, (-1)⟩)

def exact79001RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6691⟩⟩, ⟨.program ⟨214⟩, ⟨26544⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨14949⟩⟩], [⟨.program ⟨214⟩, ⟨23780⟩⟩]⟩, (-1)⟩]

theorem exact79001RawTermsValid :
    exact79001RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79001 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26546⟩⟩) exact79001RawTerms .large 78994 (.finite 1291900378790628425728) (some (78996))

def event79002 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20460⟩⟩) 0 ⟨14950⟩ 3471

def event79003 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20460⟩⟩) (.authority (.relationPreimageSource ⟨29⟩))

def exact79004RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨20460⟩⟩]⟩, (1)⟩]

theorem exact79004RawTermsValid :
    exact79004RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79004 : Event := .resultExact (⟨.program ⟨214⟩, ⟨20460⟩⟩) exact79004RawTerms (.finite 136065468) 79003 .exactZero (none)

def event79005 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20462⟩⟩) 0 ⟨20460⟩ 79004

def event79006 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20462⟩⟩) 1 ⟨2348⟩ 4

def event79007 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20462⟩⟩) (.scale (.predecessor 0 79005 .coefficient) (.value (.predecessor 1 79006 .coefficient)))

def exact79008RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨20460⟩⟩]⟩, (1)⟩]

theorem exact79008RawTermsValid :
    exact79008RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79008 : Event := .resultExact (⟨.program ⟨214⟩, ⟨20462⟩⟩) exact79008RawTerms (.finite 136065468) 79007 .exactZero (none)

def event79009 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20463⟩⟩) 0 ⟨5535⟩ 65387

def event79010 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20463⟩⟩) 1 ⟨20462⟩ 79008

def event79011 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20463⟩⟩) (.product (.predecessor 0 79009 .coefficient) (.predecessor 1 79010 .coefficient) (⟨false, false, none, none, none⟩))

def event79012 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20463⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨20460⟩⟩]⟩) [⟨.result 79004 .coefficient, false, none⟩])

def event79013 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20463⟩⟩) (.product (.result 65387 .summary) (.transfer 79012) (⟨false, false, none, none, none⟩))

def event79014 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20463⟩⟩, .operator (⟨65387, 0⟩, ⟨79008, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20460⟩⟩]⟩, (1)⟩)

def event79015 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨20461⟩⟩)

def event79016 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event79017 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event79018 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨360⟩⟩) (.authority (.operator))

def event79019 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨360⟩⟩) (.finite 2)

def event79020 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event79021 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event79022 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event79023 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event79024 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 79023

def event79025 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 79021

def event79026 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 79024 .coefficient) (.value (.predecessor 1 79025 .coefficient)))

def event79027 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event79028 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5504⟩⟩) 0 ⟨5503⟩ 79027

def event79029 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5504⟩⟩) 1 ⟨360⟩ 79019

def event79030 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5504⟩⟩) (.sum [.predecessor 0 79028 .coefficient, .predecessor 1 79029 .coefficient])

def event79031 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5504⟩⟩) (.finite 219)

def event79032 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5530⟩⟩) 0 ⟨5504⟩ 79031

def event79033 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5530⟩⟩) 1 ⟨961⟩ 79017

def event79034 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5530⟩⟩) (.identity (.predecessor 1 79033 .coefficient))

def event79035 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5530⟩⟩) (.finite 224)

def event79036 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10668⟩⟩) 0 ⟨5530⟩ 79035

def event79037 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10668⟩⟩) (.authority (.programFamilyFact))

def exact79038RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨10668⟩⟩], []⟩, (1)⟩]

theorem exact79038RawTermsValid :
    exact79038RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79038 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10668⟩⟩) exact79038RawTerms (.finite 3) 79037 .exactZero (none)

def event79039 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9500⟩⟩) 0 ⟨5530⟩ 79035

def event79040 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨9500⟩⟩) (.authority (.programFamilyFact))

def exact79041RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨9500⟩⟩], []⟩, (1)⟩]

theorem exact79041RawTermsValid :
    exact79041RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79041 : Event := .resultExact (⟨.program ⟨214⟩, ⟨9500⟩⟩) exact79041RawTerms (.finite 3) 79040 .exactZero (none)

def event79042 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10669⟩⟩) 0 ⟨9500⟩ 79041

def event79043 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10669⟩⟩) 1 ⟨10668⟩ 79038

def event79044 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10669⟩⟩) (.product (.predecessor 0 79042 .coefficient) (.predecessor 1 79043 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event79045 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10669⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨9500⟩⟩, ⟨.program ⟨214⟩, ⟨10668⟩⟩], []⟩) [⟨.result 79041 .coefficient, true, some 1⟩, ⟨.result 79038 .coefficient, true, some 1⟩])

def event79046 : Event := .survivorFold (1) 79045

def exact79047RawTerms : List Term := []

theorem exact79047RawTermsValid :
    exact79047RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79047 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10669⟩⟩) exact79047RawTerms (.finite 9) 79044 (.finite 9) (some (79045))

def event79048 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10670⟩⟩) 0 ⟨10669⟩ 79047

def event79049 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10670⟩⟩) (.identity (.predecessor 0 79048 .coefficient))

def event79050 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨10670⟩⟩) (.finite 9)

def event79051 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14949⟩⟩) 0 ⟨10670⟩ 79050

def event79052 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14949⟩⟩) (.authority (.programFamilyFact))

def exact79053RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨14949⟩⟩], []⟩, (1)⟩]

theorem exact79053RawTermsValid :
    exact79053RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79053 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14949⟩⟩) exact79053RawTerms (.finite 3) 79052 .exactZero (none)

def event79054 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14950⟩⟩) 0 ⟨14949⟩ 79053

def event79055 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14950⟩⟩) (.identity (.predecessor 0 79054 .coefficient))

def event79056 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨14950⟩⟩) (.finite 3)

def event79057 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20460⟩⟩) 0 ⟨14950⟩ 79056

def event79058 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20460⟩⟩) (.authority (.relationPreimageSource ⟨29⟩))

def exact79059RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨20460⟩⟩]⟩, (1)⟩]

theorem exact79059RawTermsValid :
    exact79059RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79059 : Event := .resultExact (⟨.program ⟨214⟩, ⟨20460⟩⟩) exact79059RawTerms (.finite 136065468) 79058 .exactZero (none)

def event79060 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6⟩⟩) (.authority (.operator))

def exact79061RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩]⟩, (1)⟩]

theorem exact79061RawTermsValid :
    exact79061RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79061 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6⟩⟩) exact79061RawTerms .large 79060 .exactZero (none)

def event79062 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20461⟩⟩) 0 ⟨6⟩ 79061

def event79063 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20461⟩⟩) 1 ⟨20460⟩ 79059

def event79064 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20461⟩⟩) (.product (.predecessor 0 79062 .coefficient) (.predecessor 1 79063 .coefficient) (⟨false, false, none, none, none⟩))

def event79065 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20461⟩⟩, .operator (⟨79061, 0⟩, ⟨79059, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20460⟩⟩]⟩, (1)⟩)

def exact79066RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20460⟩⟩]⟩, (1)⟩]

theorem exact79066RawTermsValid :
    exact79066RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79066 : Event := .resultExact (⟨.program ⟨214⟩, ⟨20461⟩⟩) exact79066RawTerms .large 79064 .exactZero (none)

def event79067 : Event := .preFoldPolynomial 79066 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20460⟩⟩]⟩, (1)⟩] .exactZero none

def exact79068RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20460⟩⟩]⟩, (1)⟩]

def event79068 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨20461⟩⟩) 79067 exact79068RawTerms .large 79064 .exactZero (none)

def event79069 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨26550⟩⟩)

def event79070 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event79071 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event79072 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨360⟩⟩) (.authority (.operator))

def event79073 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨360⟩⟩) (.finite 2)

def event79074 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event79075 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event79076 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event79077 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event79078 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 79077

def event79079 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 79075

def event79080 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 79078 .coefficient) (.value (.predecessor 1 79079 .coefficient)))

def event79081 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event79082 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5504⟩⟩) 0 ⟨5503⟩ 79081

def event79083 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5504⟩⟩) 1 ⟨360⟩ 79073

def event79084 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5504⟩⟩) (.sum [.predecessor 0 79082 .coefficient, .predecessor 1 79083 .coefficient])

def event79085 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5504⟩⟩) (.finite 219)

def event79086 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5530⟩⟩) 0 ⟨5504⟩ 79085

def event79087 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5530⟩⟩) 1 ⟨961⟩ 79071

def event79088 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5530⟩⟩) (.identity (.predecessor 1 79087 .coefficient))

def event79089 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5530⟩⟩) (.finite 224)

def event79090 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10668⟩⟩) 0 ⟨5530⟩ 79089

def event79091 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10668⟩⟩) (.authority (.programFamilyFact))

def exact79092RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨10668⟩⟩], []⟩, (1)⟩]

theorem exact79092RawTermsValid :
    exact79092RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79092 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10668⟩⟩) exact79092RawTerms (.finite 3) 79091 .exactZero (none)

def event79093 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9500⟩⟩) 0 ⟨5530⟩ 79089

def event79094 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨9500⟩⟩) (.authority (.programFamilyFact))

def exact79095RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨9500⟩⟩], []⟩, (1)⟩]

theorem exact79095RawTermsValid :
    exact79095RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79095 : Event := .resultExact (⟨.program ⟨214⟩, ⟨9500⟩⟩) exact79095RawTerms (.finite 3) 79094 .exactZero (none)

def event79096 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10669⟩⟩) 0 ⟨9500⟩ 79095

def event79097 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10669⟩⟩) 1 ⟨10668⟩ 79092

def event79098 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10669⟩⟩) (.product (.predecessor 0 79096 .coefficient) (.predecessor 1 79097 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event79099 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨10669⟩⟩, .operator (⟨79095, 0⟩, ⟨79092, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨9500⟩⟩, ⟨.program ⟨214⟩, ⟨10668⟩⟩], []⟩, (1)⟩)

def exact79100RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨9500⟩⟩, ⟨.program ⟨214⟩, ⟨10668⟩⟩], []⟩, (1)⟩]

theorem exact79100RawTermsValid :
    exact79100RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79100 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10669⟩⟩) exact79100RawTerms (.finite 9) 79098 .exactZero (none)

def event79101 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10670⟩⟩) 0 ⟨10669⟩ 79100

def event79102 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10670⟩⟩) (.identity (.predecessor 0 79101 .coefficient))

def event79103 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨10670⟩⟩) (.finite 9)

def eventLeaf4928 : Array AnnotatedEvent := #[
  { event := event78848
    frameStart := 78803 },
  { event := event78849
    frameStart := 78803 },
  { event := event78850
    frameStart := 78803 },
  { event := event78851
    frameStart := 78803 },
  { event := event78852
    frameStart := 78803 },
  { event := event78853
    frameStart := 78803 },
  { event := event78854
    frameStart := 78803 },
  { event := event78855
    frameStart := 78803 },
  { event := event78856
    frameStart := 78803 },
  { event := event78857
    frameStart := 78857 },
  { event := event78858
    frameStart := 78857 },
  { event := event78859
    frameStart := 78857 },
  { event := event78860
    frameStart := 78857 },
  { event := event78861
    frameStart := 78857 },
  { event := event78862
    frameStart := 78857 },
  { event := event78863
    frameStart := 78857 }
]

def eventLeaf4929 : Array AnnotatedEvent := #[
  { event := event78864
    frameStart := 78857 },
  { event := event78865
    frameStart := 78857 },
  { event := event78866
    frameStart := 78857 },
  { event := event78867
    frameStart := 78857 },
  { event := event78868
    frameStart := 78857 },
  { event := event78869
    frameStart := 78857 },
  { event := event78870
    frameStart := 78857 },
  { event := event78871
    frameStart := 78857 },
  { event := event78872
    frameStart := 78857 },
  { event := event78873
    frameStart := 78857 },
  { event := event78874
    frameStart := 78857 },
  { event := event78875
    frameStart := 78857 },
  { event := event78876
    frameStart := 78857 },
  { event := event78877
    frameStart := 78857 },
  { event := event78878
    frameStart := 78857 },
  { event := event78879
    frameStart := 78857 }
]

def eventLeaf4930 : Array AnnotatedEvent := #[
  { event := event78880
    frameStart := 78857 },
  { event := event78881
    frameStart := 78857 },
  { event := event78882
    frameStart := 78857 },
  { event := event78883
    frameStart := 78857 },
  { event := event78884
    frameStart := 78857 },
  { event := event78885
    frameStart := 78857 },
  { event := event78886
    frameStart := 78857 },
  { event := event78887
    frameStart := 78857 },
  { event := event78888
    frameStart := 78857 },
  { event := event78889
    frameStart := 78857 },
  { event := event78890
    frameStart := 78857 },
  { event := event78891
    frameStart := 78857 },
  { event := event78892
    frameStart := 78857 },
  { event := event78893
    frameStart := 78857 },
  { event := event78894
    frameStart := 78857 },
  { event := event78895
    frameStart := 78857 }
]

def eventLeaf4931 : Array AnnotatedEvent := #[
  { event := event78896
    frameStart := 78857 },
  { event := event78897
    frameStart := 78857 },
  { event := event78898
    frameStart := 78857 },
  { event := event78899
    frameStart := 78857 },
  { event := event78900
    frameStart := 78857 },
  { event := event78901
    frameStart := 78857 },
  { event := event78902
    frameStart := 78857 },
  { event := event78903
    frameStart := 78857 },
  { event := event78904
    frameStart := 78857 },
  { event := event78905
    frameStart := 78857 },
  { event := event78906
    frameStart := 78857 },
  { event := event78907
    frameStart := 78857 },
  { event := event78908
    frameStart := 78857 },
  { event := event78909
    frameStart := 78857 },
  { event := event78910
    frameStart := 78857 },
  { event := event78911
    frameStart := 78857 }
]

def eventLeaf4932 : Array AnnotatedEvent := #[
  { event := event78912
    frameStart := 78857 },
  { event := event78913
    frameStart := 78857 },
  { event := event78914
    frameStart := 78857 },
  { event := event78915
    frameStart := 78857 },
  { event := event78916
    frameStart := 78857 },
  { event := event78917
    frameStart := 78857 },
  { event := event78918
    frameStart := 78857 },
  { event := event78919
    frameStart := 78857 },
  { event := event78920
    frameStart := 78857 },
  { event := event78921
    frameStart := 78857 },
  { event := event78922
    frameStart := 78857 },
  { event := event78923
    frameStart := 78857 },
  { event := event78924
    frameStart := 78857 },
  { event := event78925
    frameStart := 78857 },
  { event := event78926
    frameStart := 78857 },
  { event := event78927
    frameStart := 78857 }
]

def eventLeaf4933 : Array AnnotatedEvent := #[
  { event := event78928
    frameStart := 78857 },
  { event := event78929
    frameStart := 78857 },
  { event := event78930
    frameStart := 78857 },
  { event := event78931
    frameStart := 78857 },
  { event := event78932
    frameStart := 78857 },
  { event := event78933
    frameStart := 78857 },
  { event := event78934
    frameStart := 78857 },
  { event := event78935
    frameStart := 78857 },
  { event := event78936
    frameStart := 78857 },
  { event := event78937
    frameStart := 78857 },
  { event := event78938
    frameStart := 78857 },
  { event := event78939
    frameStart := 78857 },
  { event := event78940
    frameStart := 78857 },
  { event := event78941
    frameStart := 78857 },
  { event := event78942
    frameStart := 78857 },
  { event := event78943
    frameStart := 78857 }
]

def eventLeaf4934 : Array AnnotatedEvent := #[
  { event := event78944
    frameStart := 78857 },
  { event := event78945
    frameStart := 78857 },
  { event := event78946
    frameStart := 78857 },
  { event := event78947
    frameStart := 78857 },
  { event := event78948
    frameStart := 78857 },
  { event := event78949
    frameStart := 78857 },
  { event := event78950
    frameStart := 78857 },
  { event := event78951
    frameStart := 78857 },
  { event := event78952
    frameStart := 78857 },
  { event := event78953
    frameStart := 78857 },
  { event := event78954
    frameStart := 78857 },
  { event := event78955
    frameStart := 78857 },
  { event := event78956
    frameStart := 78857 },
  { event := event78957
    frameStart := 78857 },
  { event := event78958
    frameStart := 78857 },
  { event := event78959
    frameStart := 78857 }
]

def eventLeaf4935 : Array AnnotatedEvent := #[
  { event := event78960
    frameStart := 78857 },
  { event := event78961
    frameStart := 0 },
  { event := event78962
    frameStart := 0 },
  { event := event78963
    frameStart := 0 },
  { event := event78964
    frameStart := 0 },
  { event := event78965
    frameStart := 0 },
  { event := event78966
    frameStart := 0 },
  { event := event78967
    frameStart := 0 },
  { event := event78968
    frameStart := 0 },
  { event := event78969
    frameStart := 0 },
  { event := event78970
    frameStart := 0 },
  { event := event78971
    frameStart := 0 },
  { event := event78972
    frameStart := 0 },
  { event := event78973
    frameStart := 0 },
  { event := event78974
    frameStart := 0 },
  { event := event78975
    frameStart := 0 }
]

def eventLeaf4936 : Array AnnotatedEvent := #[
  { event := event78976
    frameStart := 0 },
  { event := event78977
    frameStart := 0 },
  { event := event78978
    frameStart := 0 },
  { event := event78979
    frameStart := 0 },
  { event := event78980
    frameStart := 0 },
  { event := event78981
    frameStart := 0 },
  { event := event78982
    frameStart := 0 },
  { event := event78983
    frameStart := 0 },
  { event := event78984
    frameStart := 0 },
  { event := event78985
    frameStart := 0 },
  { event := event78986
    frameStart := 0 },
  { event := event78987
    frameStart := 0 },
  { event := event78988
    frameStart := 0 },
  { event := event78989
    frameStart := 0 },
  { event := event78990
    frameStart := 0 },
  { event := event78991
    frameStart := 0 }
]

def eventLeaf4937 : Array AnnotatedEvent := #[
  { event := event78992
    frameStart := 0 },
  { event := event78993
    frameStart := 0 },
  { event := event78994
    frameStart := 0 },
  { event := event78995
    frameStart := 0 },
  { event := event78996
    frameStart := 0 },
  { event := event78997
    frameStart := 0 },
  { event := event78998
    frameStart := 0 },
  { event := event78999
    frameStart := 0 },
  { event := event79000
    frameStart := 0 },
  { event := event79001
    frameStart := 0 },
  { event := event79002
    frameStart := 0 },
  { event := event79003
    frameStart := 0 },
  { event := event79004
    frameStart := 0 },
  { event := event79005
    frameStart := 0 },
  { event := event79006
    frameStart := 0 },
  { event := event79007
    frameStart := 0 }
]

def eventLeaf4938 : Array AnnotatedEvent := #[
  { event := event79008
    frameStart := 0 },
  { event := event79009
    frameStart := 0 },
  { event := event79010
    frameStart := 0 },
  { event := event79011
    frameStart := 0 },
  { event := event79012
    frameStart := 0 },
  { event := event79013
    frameStart := 0 },
  { event := event79014
    frameStart := 0 },
  { event := event79015
    frameStart := 79015 },
  { event := event79016
    frameStart := 79015 },
  { event := event79017
    frameStart := 79015 },
  { event := event79018
    frameStart := 79015 },
  { event := event79019
    frameStart := 79015 },
  { event := event79020
    frameStart := 79015 },
  { event := event79021
    frameStart := 79015 },
  { event := event79022
    frameStart := 79015 },
  { event := event79023
    frameStart := 79015 }
]

def eventLeaf4939 : Array AnnotatedEvent := #[
  { event := event79024
    frameStart := 79015 },
  { event := event79025
    frameStart := 79015 },
  { event := event79026
    frameStart := 79015 },
  { event := event79027
    frameStart := 79015 },
  { event := event79028
    frameStart := 79015 },
  { event := event79029
    frameStart := 79015 },
  { event := event79030
    frameStart := 79015 },
  { event := event79031
    frameStart := 79015 },
  { event := event79032
    frameStart := 79015 },
  { event := event79033
    frameStart := 79015 },
  { event := event79034
    frameStart := 79015 },
  { event := event79035
    frameStart := 79015 },
  { event := event79036
    frameStart := 79015 },
  { event := event79037
    frameStart := 79015 },
  { event := event79038
    frameStart := 79015 },
  { event := event79039
    frameStart := 79015 }
]

def eventLeaf4940 : Array AnnotatedEvent := #[
  { event := event79040
    frameStart := 79015 },
  { event := event79041
    frameStart := 79015 },
  { event := event79042
    frameStart := 79015 },
  { event := event79043
    frameStart := 79015 },
  { event := event79044
    frameStart := 79015 },
  { event := event79045
    frameStart := 79015 },
  { event := event79046
    frameStart := 79015 },
  { event := event79047
    frameStart := 79015 },
  { event := event79048
    frameStart := 79015 },
  { event := event79049
    frameStart := 79015 },
  { event := event79050
    frameStart := 79015 },
  { event := event79051
    frameStart := 79015 },
  { event := event79052
    frameStart := 79015 },
  { event := event79053
    frameStart := 79015 },
  { event := event79054
    frameStart := 79015 },
  { event := event79055
    frameStart := 79015 }
]

def eventLeaf4941 : Array AnnotatedEvent := #[
  { event := event79056
    frameStart := 79015 },
  { event := event79057
    frameStart := 79015 },
  { event := event79058
    frameStart := 79015 },
  { event := event79059
    frameStart := 79015 },
  { event := event79060
    frameStart := 79015 },
  { event := event79061
    frameStart := 79015 },
  { event := event79062
    frameStart := 79015 },
  { event := event79063
    frameStart := 79015 },
  { event := event79064
    frameStart := 79015 },
  { event := event79065
    frameStart := 79015 },
  { event := event79066
    frameStart := 79015 },
  { event := event79067
    frameStart := 79015 },
  { event := event79068
    frameStart := 79015 },
  { event := event79069
    frameStart := 79069 },
  { event := event79070
    frameStart := 79069 },
  { event := event79071
    frameStart := 79069 }
]

def eventLeaf4942 : Array AnnotatedEvent := #[
  { event := event79072
    frameStart := 79069 },
  { event := event79073
    frameStart := 79069 },
  { event := event79074
    frameStart := 79069 },
  { event := event79075
    frameStart := 79069 },
  { event := event79076
    frameStart := 79069 },
  { event := event79077
    frameStart := 79069 },
  { event := event79078
    frameStart := 79069 },
  { event := event79079
    frameStart := 79069 },
  { event := event79080
    frameStart := 79069 },
  { event := event79081
    frameStart := 79069 },
  { event := event79082
    frameStart := 79069 },
  { event := event79083
    frameStart := 79069 },
  { event := event79084
    frameStart := 79069 },
  { event := event79085
    frameStart := 79069 },
  { event := event79086
    frameStart := 79069 },
  { event := event79087
    frameStart := 79069 }
]

def eventLeaf4943 : Array AnnotatedEvent := #[
  { event := event79088
    frameStart := 79069 },
  { event := event79089
    frameStart := 79069 },
  { event := event79090
    frameStart := 79069 },
  { event := event79091
    frameStart := 79069 },
  { event := event79092
    frameStart := 79069 },
  { event := event79093
    frameStart := 79069 },
  { event := event79094
    frameStart := 79069 },
  { event := event79095
    frameStart := 79069 },
  { event := event79096
    frameStart := 79069 },
  { event := event79097
    frameStart := 79069 },
  { event := event79098
    frameStart := 79069 },
  { event := event79099
    frameStart := 79069 },
  { event := event79100
    frameStart := 79069 },
  { event := event79101
    frameStart := 79069 },
  { event := event79102
    frameStart := 79069 },
  { event := event79103
    frameStart := 79069 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events308
