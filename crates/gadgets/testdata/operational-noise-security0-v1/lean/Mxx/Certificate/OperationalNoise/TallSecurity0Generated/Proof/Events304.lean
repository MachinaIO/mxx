import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events304

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event77824 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14199⟩⟩) 0 ⟨14198⟩ 77823

def event77825 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14199⟩⟩) 1 ⟨11465⟩ 77820

def event77826 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14199⟩⟩) (.product (.predecessor 0 77824 .coefficient) (.predecessor 1 77825 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event77827 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨14199⟩⟩, .operator (⟨77823, 0⟩, ⟨77820, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨11465⟩⟩, ⟨.program ⟨214⟩, ⟨14198⟩⟩], []⟩, (1)⟩)

def exact77828RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11465⟩⟩, ⟨.program ⟨214⟩, ⟨14198⟩⟩], []⟩, (1)⟩]

theorem exact77828RawTermsValid :
    exact77828RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77828 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14199⟩⟩) exact77828RawTerms (.finite 324) 77826 .exactZero (none)

def event77829 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14200⟩⟩) 0 ⟨14199⟩ 77828

def event77830 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14200⟩⟩) (.identity (.predecessor 0 77829 .coefficient))

def event77831 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨14200⟩⟩) (.finite 324)

def event77832 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15936⟩⟩) 0 ⟨14200⟩ 77831

def event77833 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15936⟩⟩) (.authority (.programFamilyFact))

def exact77834RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15936⟩⟩], []⟩, (1)⟩]

theorem exact77834RawTermsValid :
    exact77834RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77834 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15936⟩⟩) exact77834RawTerms (.finite 18) 77833 .exactZero (none)

def event77835 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15937⟩⟩) 0 ⟨15936⟩ 77834

def event77836 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15937⟩⟩) (.identity (.predecessor 0 77835 .coefficient))

def event77837 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨15937⟩⟩) (.finite 18)

def event77838 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24157⟩⟩) 0 ⟨15937⟩ 77837

def event77839 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24157⟩⟩) (.authority (.programFamilyFact))

def event77840 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨24157⟩⟩) (.finite 3720)

def event77841 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨6689⟩⟩) .missing

def event77842 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24158⟩⟩) 0 ⟨6689⟩ 77841

def event77843 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24158⟩⟩) 1 ⟨24157⟩ 77840

def event77844 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24158⟩⟩) (.authority (.operator))

def exact77845RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨24158⟩⟩]⟩, (1)⟩]

theorem exact77845RawTermsValid :
    exact77845RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77845 : Event := .resultExact (⟨.program ⟨214⟩, ⟨24158⟩⟩) exact77845RawTerms .large 77844 .exactZero (none)

def event77846 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27846⟩⟩) 0 ⟨24158⟩ 77845

def event77847 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27846⟩⟩) (.authority (.operator))

def exact77848RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨27846⟩⟩]⟩, (1)⟩]

theorem exact77848RawTermsValid :
    exact77848RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77848 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27846⟩⟩) exact77848RawTerms (.finite 8192) 77847 .exactZero (none)

def event77849 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨110⟩⟩) (.authority (.operator))

def event77850 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨110⟩⟩) .exactZero

def event77851 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16011⟩⟩) 0 ⟨15937⟩ 77837

def event77852 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16011⟩⟩) 1 ⟨110⟩ 77850

def event77853 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16011⟩⟩) (.sum [.predecessor 0 77851 .coefficient, .predecessor 1 77852 .coefficient])

def event77854 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨16011⟩⟩) (.finite 18)

def event77855 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16012⟩⟩) 0 ⟨16011⟩ 77854

def event77856 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16012⟩⟩) (.identity (.predecessor 0 77855 .coefficient))

def exact77857RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15936⟩⟩], []⟩, (1)⟩]

theorem exact77857RawTermsValid :
    exact77857RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77857 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16012⟩⟩) exact77857RawTerms (.finite 18) 77856 .exactZero (none)

def event77858 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6544⟩⟩) (.authority (.factStore))

def exact77859RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact77859RawTermsValid :
    exact77859RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77859 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6544⟩⟩) exact77859RawTerms .large 77858 .exactZero (none)

def event77860 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16013⟩⟩) 0 ⟨6544⟩ 77859

def event77861 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16013⟩⟩) 1 ⟨16012⟩ 77857

def event77862 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16013⟩⟩) (.product (.predecessor 0 77860 .coefficient) (.predecessor 1 77861 .coefficient) (⟨false, false, none, none, none⟩))

def event77863 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨16013⟩⟩, .operator (⟨77859, 0⟩, ⟨77857, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨15936⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact77864RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15936⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact77864RawTermsValid :
    exact77864RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77864 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16013⟩⟩) exact77864RawTerms .large 77862 .exactZero (none)

def event77865 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6697⟩⟩) 0 ⟨6689⟩ 77841

def event77866 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6697⟩⟩) (.authority (.operator))

def exact77867RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6697⟩⟩]⟩, (1)⟩]

theorem exact77867RawTermsValid :
    exact77867RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77867 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6697⟩⟩) exact77867RawTerms .large 77866 .exactZero (none)

def event77868 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16014⟩⟩) 0 ⟨6697⟩ 77867

def event77869 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16014⟩⟩) 1 ⟨16013⟩ 77864

def event77870 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16014⟩⟩) (.sum [.predecessor 0 77868 .coefficient, .predecessor 1 77869 .coefficient])

def exact77871RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6697⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15936⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact77871RawTermsValid :
    exact77871RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77871 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16014⟩⟩) exact77871RawTerms .large 77870 .exactZero (none)

def event77872 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27847⟩⟩) 0 ⟨16014⟩ 77871

def event77873 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27847⟩⟩) 1 ⟨27846⟩ 77848

def event77874 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27847⟩⟩) (.product (.predecessor 0 77872 .coefficient) (.predecessor 1 77873 .coefficient) (⟨false, false, none, none, none⟩))

def event77875 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27847⟩⟩, .operator (⟨77871, 0⟩, ⟨77848, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6697⟩⟩, ⟨.program ⟨214⟩, ⟨27846⟩⟩]⟩, (1)⟩)

def event77876 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27847⟩⟩, .operator (⟨77871, 1⟩, ⟨77848, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨15936⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27846⟩⟩]⟩, (-1)⟩)

def event77877 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨27847⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨15936⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27846⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨27846⟩⟩) ⟨24158⟩ 77845)

def event77878 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27847⟩⟩, .relation 77877 0, ⟨[⟨.program ⟨214⟩, ⟨15936⟩⟩], [⟨.program ⟨214⟩, ⟨24158⟩⟩]⟩, (-1)⟩)

def exact77879RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6697⟩⟩, ⟨.program ⟨214⟩, ⟨27846⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15936⟩⟩], [⟨.program ⟨214⟩, ⟨24158⟩⟩]⟩, (-1)⟩]

theorem exact77879RawTermsValid :
    exact77879RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77879 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27847⟩⟩) exact77879RawTerms .large 77874 .exactZero (none)

def event77880 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17161⟩⟩) 0 ⟨15937⟩ 77837

def event77881 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨17161⟩⟩) (.authority (.programFamilyFact))

def exact77882RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨17161⟩⟩], []⟩, (1)⟩]

theorem exact77882RawTermsValid :
    exact77882RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77882 : Event := .resultExact (⟨.program ⟨214⟩, ⟨17161⟩⟩) exact77882RawTerms (.finite 18) 77881 .exactZero (none)

def event77883 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17163⟩⟩) 0 ⟨6544⟩ 77859

def event77884 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17163⟩⟩) 1 ⟨17161⟩ 77882

def event77885 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨17163⟩⟩) (.product (.predecessor 0 77883 .coefficient) (.predecessor 1 77884 .coefficient) (⟨false, true, none, none, some 1⟩))

def event77886 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨17163⟩⟩, .operator (⟨77859, 0⟩, ⟨77882, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨17161⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact77887RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨17161⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact77887RawTermsValid :
    exact77887RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77887 : Event := .resultExact (⟨.program ⟨214⟩, ⟨17163⟩⟩) exact77887RawTerms .large 77885 .exactZero (none)

def event77888 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6722⟩⟩) 0 ⟨6689⟩ 77841

def event77889 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6722⟩⟩) (.authority (.operator))

def exact77890RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6722⟩⟩]⟩, (1)⟩]

theorem exact77890RawTermsValid :
    exact77890RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77890 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6722⟩⟩) exact77890RawTerms .large 77889 .exactZero (none)

def event77891 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17164⟩⟩) 0 ⟨6722⟩ 77890

def event77892 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17164⟩⟩) 1 ⟨17163⟩ 77887

def event77893 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨17164⟩⟩) (.sum [.predecessor 0 77891 .coefficient, .predecessor 1 77892 .coefficient])

def exact77894RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6722⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17161⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact77894RawTermsValid :
    exact77894RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77894 : Event := .resultExact (⟨.program ⟨214⟩, ⟨17164⟩⟩) exact77894RawTerms .large 77893 .exactZero (none)

def event77895 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27852⟩⟩) 0 ⟨17164⟩ 77894

def event77896 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27852⟩⟩) 1 ⟨27847⟩ 77879

def event77897 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27852⟩⟩) (.sum [.predecessor 0 77895 .coefficient, .predecessor 1 77896 .coefficient])

def exact77898RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6697⟩⟩, ⟨.program ⟨214⟩, ⟨27846⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6722⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15936⟩⟩], [⟨.program ⟨214⟩, ⟨24158⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17161⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact77898RawTermsValid :
    exact77898RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77898 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27852⟩⟩) exact77898RawTerms .large 77897 .exactZero (none)

def event77899 : Event := .preFoldPolynomial 77898 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6697⟩⟩, ⟨.program ⟨214⟩, ⟨27846⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6722⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15936⟩⟩], [⟨.program ⟨214⟩, ⟨24158⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17161⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩] .exactZero none

def exact77900RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6697⟩⟩, ⟨.program ⟨214⟩, ⟨27846⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6722⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15936⟩⟩], [⟨.program ⟨214⟩, ⟨24158⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17161⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

def event77900 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨27852⟩⟩) 77899 exact77900RawTerms .large 77897 .exactZero (none)

def event77901 : Event := .specializationComputed (⟨.program ⟨214⟩, ⟨15937⟩⟩) ⟨⟨135⟩, ⟨42⟩, ⟨109⟩⟩ ⟨77743, 77901⟩

def event77902 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨21327⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21324⟩⟩]⟩) (1) 0 2 (.universal 77901 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21324⟩⟩]⟩) (none) 77900)

def event77903 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21327⟩⟩, .relation 77902 1, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6722⟩⟩]⟩, (1)⟩)

def event77904 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21327⟩⟩, .relation 77902 0, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6697⟩⟩, ⟨.program ⟨214⟩, ⟨27846⟩⟩]⟩, (-1)⟩)

def event77905 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21327⟩⟩, .relation 77902 2, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15936⟩⟩], [⟨.program ⟨214⟩, ⟨24158⟩⟩]⟩, (1)⟩)

def event77906 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21327⟩⟩, .relation 77902 3, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨17161⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact77907RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6697⟩⟩, ⟨.program ⟨214⟩, ⟨27846⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6722⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15936⟩⟩], [⟨.program ⟨214⟩, ⟨24158⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨17161⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact77907RawTermsValid :
    exact77907RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77907 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21327⟩⟩) exact77907RawTerms .large 77739 (.finite 1811303510016) (some (77741))

def event77908 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27849⟩⟩) 0 ⟨21327⟩ 77907

def event77909 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27849⟩⟩) 1 ⟨27848⟩ 77729

def event77910 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27849⟩⟩) (.sum [.predecessor 0 77908 .coefficient, .predecessor 1 77909 .coefficient])

def event77911 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27849⟩⟩, .operator (⟨77907, 0⟩, ⟨77729, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6697⟩⟩, ⟨.program ⟨214⟩, ⟨27846⟩⟩]⟩, (1)⟩)

def event77912 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27849⟩⟩, .operator (⟨77907, 2⟩, ⟨77729, 1⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15936⟩⟩], [⟨.program ⟨214⟩, ⟨24158⟩⟩]⟩, (-1)⟩)

def event77913 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27849⟩⟩) (.sum [.result 77907 .summary, .result 77729 .summary])

def exact77914RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6722⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨17161⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact77914RawTermsValid :
    exact77914RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77914 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27849⟩⟩) exact77914RawTerms .large 77910 (.finite 1292068473939586330624) (some (77913))

def event77915 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27850⟩⟩) 0 ⟨27849⟩ 77914

def event77916 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27850⟩⟩) 1 ⟨6642⟩ 5719

def event77917 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27850⟩⟩) (.product (.predecessor 0 77915 .coefficient) (.predecessor 1 77916 .coefficient) (⟨false, false, none, none, none⟩))

def event77918 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27850⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨6641⟩⟩]⟩) [⟨.result 5715 .coefficient, false, none⟩])

def event77919 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27850⟩⟩) (.product (.result 77914 .summary) (.transfer 77918) (⟨false, false, none, none, none⟩))

def event77920 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27850⟩⟩, .operator (⟨77914, 0⟩, ⟨5719, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6722⟩⟩, ⟨.program ⟨214⟩, ⟨6641⟩⟩]⟩, (1)⟩)

def event77921 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27850⟩⟩, .operator (⟨77914, 1⟩, ⟨5719, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨17161⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨6641⟩⟩]⟩, (-1)⟩)

def event77922 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨27850⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨17161⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨6641⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨6641⟩⟩) ⟨6592⟩ 5712)

def event77923 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27850⟩⟩, .relation 77922 0, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨6387⟩⟩, ⟨.program ⟨214⟩, ⟨17161⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact77924RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6722⟩⟩, ⟨.program ⟨214⟩, ⟨6641⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨6387⟩⟩, ⟨.program ⟨214⟩, ⟨17161⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact77924RawTermsValid :
    exact77924RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77924 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27850⟩⟩) exact77924RawTerms .large 77917 (.finite 4741911972453864866771369984) (some (77919))

def event77925 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24095⟩⟩) 0 ⟨6689⟩ 5477

def event77926 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24095⟩⟩) 1 ⟨24094⟩ 70591

def event77927 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24095⟩⟩) (.authority (.operator))

def exact77928RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨24095⟩⟩]⟩, (1)⟩]

theorem exact77928RawTermsValid :
    exact77928RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77928 : Event := .resultExact (⟨.program ⟨214⟩, ⟨24095⟩⟩) exact77928RawTerms .large 77927 .exactZero (none)

def event77929 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27629⟩⟩) 0 ⟨24095⟩ 77928

def event77930 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27629⟩⟩) (.authority (.operator))

def exact77931RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨27629⟩⟩]⟩, (1)⟩]

theorem exact77931RawTermsValid :
    exact77931RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77931 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27629⟩⟩) exact77931RawTerms (.finite 8192) 77930 .exactZero (none)

def event77932 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27631⟩⟩) 0 ⟨25986⟩ 70875

def event77933 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27631⟩⟩) 1 ⟨27629⟩ 77931

def event77934 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27631⟩⟩) (.product (.predecessor 0 77932 .coefficient) (.predecessor 1 77933 .coefficient) (⟨false, false, none, none, none⟩))

def event77935 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27631⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨27629⟩⟩]⟩) [⟨.result 77931 .coefficient, false, none⟩])

def event77936 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27631⟩⟩) (.product (.result 70875 .summary) (.transfer 77935) (⟨false, false, none, none, none⟩))

def event77937 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27631⟩⟩, .operator (⟨70875, 0⟩, ⟨77931, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6696⟩⟩, ⟨.program ⟨214⟩, ⟨27629⟩⟩]⟩, (1)⟩)

def event77938 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27631⟩⟩, .operator (⟨70875, 1⟩, ⟨77931, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15817⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27629⟩⟩]⟩, (-1)⟩)

def event77939 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨27631⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15817⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27629⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨27629⟩⟩) ⟨24095⟩ 77928)

def event77940 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27631⟩⟩, .relation 77939 0, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15817⟩⟩], [⟨.program ⟨214⟩, ⟨24095⟩⟩]⟩, (-1)⟩)

def exact77941RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6696⟩⟩, ⟨.program ⟨214⟩, ⟨27629⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15817⟩⟩], [⟨.program ⟨214⟩, ⟨24095⟩⟩]⟩, (-1)⟩]

theorem exact77941RawTermsValid :
    exact77941RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77941 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27631⟩⟩) exact77941RawTerms .large 77934 (.finite 1292046059683262234624) (some (77936))

def event77942 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21180⟩⟩) 0 ⟨15818⟩ 3356

def event77943 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21180⟩⟩) (.authority (.relationPreimageSource ⟨40⟩))

def exact77944RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨21180⟩⟩]⟩, (1)⟩]

theorem exact77944RawTermsValid :
    exact77944RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77944 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21180⟩⟩) exact77944RawTerms (.finite 136065468) 77943 .exactZero (none)

def event77945 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21182⟩⟩) 0 ⟨21180⟩ 77944

def event77946 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21182⟩⟩) 1 ⟨2348⟩ 4

def event77947 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21182⟩⟩) (.scale (.predecessor 0 77945 .coefficient) (.value (.predecessor 1 77946 .coefficient)))

def exact77948RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨21180⟩⟩]⟩, (1)⟩]

theorem exact77948RawTermsValid :
    exact77948RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77948 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21182⟩⟩) exact77948RawTerms (.finite 136065468) 77947 .exactZero (none)

def event77949 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21183⟩⟩) 0 ⟨5535⟩ 65387

def event77950 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21183⟩⟩) 1 ⟨21182⟩ 77948

def event77951 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21183⟩⟩) (.product (.predecessor 0 77949 .coefficient) (.predecessor 1 77950 .coefficient) (⟨false, false, none, none, none⟩))

def event77952 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21183⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨21180⟩⟩]⟩) [⟨.result 77944 .coefficient, false, none⟩])

def event77953 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21183⟩⟩) (.product (.result 65387 .summary) (.transfer 77952) (⟨false, false, none, none, none⟩))

def event77954 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21183⟩⟩, .operator (⟨65387, 0⟩, ⟨77948, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21180⟩⟩]⟩, (1)⟩)

def event77955 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨21181⟩⟩)

def event77956 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event77957 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event77958 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨360⟩⟩) (.authority (.operator))

def event77959 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨360⟩⟩) (.finite 2)

def event77960 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event77961 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event77962 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event77963 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event77964 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 77963

def event77965 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 77961

def event77966 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 77964 .coefficient) (.value (.predecessor 1 77965 .coefficient)))

def event77967 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event77968 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5504⟩⟩) 0 ⟨5503⟩ 77967

def event77969 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5504⟩⟩) 1 ⟨360⟩ 77959

def event77970 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5504⟩⟩) (.sum [.predecessor 0 77968 .coefficient, .predecessor 1 77969 .coefficient])

def event77971 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5504⟩⟩) (.finite 219)

def event77972 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5530⟩⟩) 0 ⟨5504⟩ 77971

def event77973 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5530⟩⟩) 1 ⟨961⟩ 77957

def event77974 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5530⟩⟩) (.identity (.predecessor 1 77973 .coefficient))

def event77975 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5530⟩⟩) (.finite 224)

def event77976 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11381⟩⟩) 0 ⟨5530⟩ 77975

def event77977 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11381⟩⟩) (.authority (.programFamilyFact))

def exact77978RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11381⟩⟩], []⟩, (1)⟩]

theorem exact77978RawTermsValid :
    exact77978RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77978 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11381⟩⟩) exact77978RawTerms (.finite 16) 77977 .exactZero (none)

def event77979 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13981⟩⟩) 0 ⟨5530⟩ 77975

def event77980 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13981⟩⟩) (.authority (.programFamilyFact))

def exact77981RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨13981⟩⟩], []⟩, (1)⟩]

theorem exact77981RawTermsValid :
    exact77981RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77981 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13981⟩⟩) exact77981RawTerms (.finite 16) 77980 .exactZero (none)

def event77982 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13982⟩⟩) 0 ⟨13981⟩ 77981

def event77983 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13982⟩⟩) 1 ⟨11381⟩ 77978

def event77984 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13982⟩⟩) (.product (.predecessor 0 77982 .coefficient) (.predecessor 1 77983 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event77985 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13982⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨11381⟩⟩, ⟨.program ⟨214⟩, ⟨13981⟩⟩], []⟩) [⟨.result 77981 .coefficient, true, some 1⟩, ⟨.result 77978 .coefficient, true, some 1⟩])

def event77986 : Event := .survivorFold (1) 77985

def exact77987RawTerms : List Term := []

theorem exact77987RawTermsValid :
    exact77987RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77987 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13982⟩⟩) exact77987RawTerms (.finite 256) 77984 (.finite 256) (some (77985))

def event77988 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13983⟩⟩) 0 ⟨13982⟩ 77987

def event77989 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13983⟩⟩) (.identity (.predecessor 0 77988 .coefficient))

def event77990 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨13983⟩⟩) (.finite 256)

def event77991 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15817⟩⟩) 0 ⟨13983⟩ 77990

def event77992 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15817⟩⟩) (.authority (.programFamilyFact))

def exact77993RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15817⟩⟩], []⟩, (1)⟩]

theorem exact77993RawTermsValid :
    exact77993RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77993 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15817⟩⟩) exact77993RawTerms (.finite 16) 77992 .exactZero (none)

def event77994 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15818⟩⟩) 0 ⟨15817⟩ 77993

def event77995 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15818⟩⟩) (.identity (.predecessor 0 77994 .coefficient))

def event77996 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨15818⟩⟩) (.finite 16)

def event77997 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21180⟩⟩) 0 ⟨15818⟩ 77996

def event77998 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21180⟩⟩) (.authority (.relationPreimageSource ⟨40⟩))

def exact77999RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨21180⟩⟩]⟩, (1)⟩]

theorem exact77999RawTermsValid :
    exact77999RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77999 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21180⟩⟩) exact77999RawTerms (.finite 136065468) 77998 .exactZero (none)

def event78000 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6⟩⟩) (.authority (.operator))

def exact78001RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩]⟩, (1)⟩]

theorem exact78001RawTermsValid :
    exact78001RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78001 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6⟩⟩) exact78001RawTerms .large 78000 .exactZero (none)

def event78002 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21181⟩⟩) 0 ⟨6⟩ 78001

def event78003 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21181⟩⟩) 1 ⟨21180⟩ 77999

def event78004 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21181⟩⟩) (.product (.predecessor 0 78002 .coefficient) (.predecessor 1 78003 .coefficient) (⟨false, false, none, none, none⟩))

def event78005 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21181⟩⟩, .operator (⟨78001, 0⟩, ⟨77999, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21180⟩⟩]⟩, (1)⟩)

def exact78006RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21180⟩⟩]⟩, (1)⟩]

theorem exact78006RawTermsValid :
    exact78006RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78006 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21181⟩⟩) exact78006RawTerms .large 78004 .exactZero (none)

def event78007 : Event := .preFoldPolynomial 78006 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21180⟩⟩]⟩, (1)⟩] .exactZero none

def exact78008RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21180⟩⟩]⟩, (1)⟩]

def event78008 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨21181⟩⟩) 78007 exact78008RawTerms .large 78004 .exactZero (none)

def event78009 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨27635⟩⟩)

def event78010 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event78011 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event78012 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨360⟩⟩) (.authority (.operator))

def event78013 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨360⟩⟩) (.finite 2)

def event78014 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event78015 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event78016 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event78017 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event78018 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 78017

def event78019 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 78015

def event78020 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 78018 .coefficient) (.value (.predecessor 1 78019 .coefficient)))

def event78021 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event78022 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5504⟩⟩) 0 ⟨5503⟩ 78021

def event78023 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5504⟩⟩) 1 ⟨360⟩ 78013

def event78024 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5504⟩⟩) (.sum [.predecessor 0 78022 .coefficient, .predecessor 1 78023 .coefficient])

def event78025 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5504⟩⟩) (.finite 219)

def event78026 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5530⟩⟩) 0 ⟨5504⟩ 78025

def event78027 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5530⟩⟩) 1 ⟨961⟩ 78011

def event78028 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5530⟩⟩) (.identity (.predecessor 1 78027 .coefficient))

def event78029 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5530⟩⟩) (.finite 224)

def event78030 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11381⟩⟩) 0 ⟨5530⟩ 78029

def event78031 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11381⟩⟩) (.authority (.programFamilyFact))

def exact78032RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11381⟩⟩], []⟩, (1)⟩]

theorem exact78032RawTermsValid :
    exact78032RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78032 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11381⟩⟩) exact78032RawTerms (.finite 16) 78031 .exactZero (none)

def event78033 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13981⟩⟩) 0 ⟨5530⟩ 78029

def event78034 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13981⟩⟩) (.authority (.programFamilyFact))

def exact78035RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨13981⟩⟩], []⟩, (1)⟩]

theorem exact78035RawTermsValid :
    exact78035RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78035 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13981⟩⟩) exact78035RawTerms (.finite 16) 78034 .exactZero (none)

def event78036 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13982⟩⟩) 0 ⟨13981⟩ 78035

def event78037 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13982⟩⟩) 1 ⟨11381⟩ 78032

def event78038 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13982⟩⟩) (.product (.predecessor 0 78036 .coefficient) (.predecessor 1 78037 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event78039 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨13982⟩⟩, .operator (⟨78035, 0⟩, ⟨78032, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨11381⟩⟩, ⟨.program ⟨214⟩, ⟨13981⟩⟩], []⟩, (1)⟩)

def exact78040RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11381⟩⟩, ⟨.program ⟨214⟩, ⟨13981⟩⟩], []⟩, (1)⟩]

theorem exact78040RawTermsValid :
    exact78040RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78040 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13982⟩⟩) exact78040RawTerms (.finite 256) 78038 .exactZero (none)

def event78041 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13983⟩⟩) 0 ⟨13982⟩ 78040

def event78042 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13983⟩⟩) (.identity (.predecessor 0 78041 .coefficient))

def event78043 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨13983⟩⟩) (.finite 256)

def event78044 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15817⟩⟩) 0 ⟨13983⟩ 78043

def event78045 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15817⟩⟩) (.authority (.programFamilyFact))

def exact78046RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15817⟩⟩], []⟩, (1)⟩]

theorem exact78046RawTermsValid :
    exact78046RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78046 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15817⟩⟩) exact78046RawTerms (.finite 16) 78045 .exactZero (none)

def event78047 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15818⟩⟩) 0 ⟨15817⟩ 78046

def event78048 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15818⟩⟩) (.identity (.predecessor 0 78047 .coefficient))

def event78049 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨15818⟩⟩) (.finite 16)

def event78050 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24094⟩⟩) 0 ⟨15818⟩ 78049

def event78051 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24094⟩⟩) (.authority (.programFamilyFact))

def event78052 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨24094⟩⟩) (.finite 3720)

def event78053 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨6689⟩⟩) .missing

def event78054 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24095⟩⟩) 0 ⟨6689⟩ 78053

def event78055 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24095⟩⟩) 1 ⟨24094⟩ 78052

def event78056 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24095⟩⟩) (.authority (.operator))

def exact78057RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨24095⟩⟩]⟩, (1)⟩]

theorem exact78057RawTermsValid :
    exact78057RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78057 : Event := .resultExact (⟨.program ⟨214⟩, ⟨24095⟩⟩) exact78057RawTerms .large 78056 .exactZero (none)

def event78058 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27629⟩⟩) 0 ⟨24095⟩ 78057

def event78059 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27629⟩⟩) (.authority (.operator))

def exact78060RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨27629⟩⟩]⟩, (1)⟩]

theorem exact78060RawTermsValid :
    exact78060RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78060 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27629⟩⟩) exact78060RawTerms (.finite 8192) 78059 .exactZero (none)

def event78061 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨110⟩⟩) (.authority (.operator))

def event78062 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨110⟩⟩) .exactZero

def event78063 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15892⟩⟩) 0 ⟨15818⟩ 78049

def event78064 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15892⟩⟩) 1 ⟨110⟩ 78062

def event78065 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15892⟩⟩) (.sum [.predecessor 0 78063 .coefficient, .predecessor 1 78064 .coefficient])

def event78066 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨15892⟩⟩) (.finite 16)

def event78067 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15893⟩⟩) 0 ⟨15892⟩ 78066

def event78068 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15893⟩⟩) (.identity (.predecessor 0 78067 .coefficient))

def exact78069RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15817⟩⟩], []⟩, (1)⟩]

theorem exact78069RawTermsValid :
    exact78069RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78069 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15893⟩⟩) exact78069RawTerms (.finite 16) 78068 .exactZero (none)

def event78070 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6544⟩⟩) (.authority (.factStore))

def exact78071RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact78071RawTermsValid :
    exact78071RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78071 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6544⟩⟩) exact78071RawTerms .large 78070 .exactZero (none)

def event78072 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15894⟩⟩) 0 ⟨6544⟩ 78071

def event78073 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15894⟩⟩) 1 ⟨15893⟩ 78069

def event78074 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15894⟩⟩) (.product (.predecessor 0 78072 .coefficient) (.predecessor 1 78073 .coefficient) (⟨false, false, none, none, none⟩))

def event78075 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨15894⟩⟩, .operator (⟨78071, 0⟩, ⟨78069, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨15817⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact78076RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15817⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact78076RawTermsValid :
    exact78076RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78076 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15894⟩⟩) exact78076RawTerms .large 78074 .exactZero (none)

def event78077 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6696⟩⟩) 0 ⟨6689⟩ 78053

def event78078 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6696⟩⟩) (.authority (.operator))

def exact78079RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6696⟩⟩]⟩, (1)⟩]

theorem exact78079RawTermsValid :
    exact78079RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78079 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6696⟩⟩) exact78079RawTerms .large 78078 .exactZero (none)

def eventLeaf4864 : Array AnnotatedEvent := #[
  { event := event77824
    frameStart := 77797 },
  { event := event77825
    frameStart := 77797 },
  { event := event77826
    frameStart := 77797 },
  { event := event77827
    frameStart := 77797 },
  { event := event77828
    frameStart := 77797 },
  { event := event77829
    frameStart := 77797 },
  { event := event77830
    frameStart := 77797 },
  { event := event77831
    frameStart := 77797 },
  { event := event77832
    frameStart := 77797 },
  { event := event77833
    frameStart := 77797 },
  { event := event77834
    frameStart := 77797 },
  { event := event77835
    frameStart := 77797 },
  { event := event77836
    frameStart := 77797 },
  { event := event77837
    frameStart := 77797 },
  { event := event77838
    frameStart := 77797 },
  { event := event77839
    frameStart := 77797 }
]

def eventLeaf4865 : Array AnnotatedEvent := #[
  { event := event77840
    frameStart := 77797 },
  { event := event77841
    frameStart := 77797 },
  { event := event77842
    frameStart := 77797 },
  { event := event77843
    frameStart := 77797 },
  { event := event77844
    frameStart := 77797 },
  { event := event77845
    frameStart := 77797 },
  { event := event77846
    frameStart := 77797 },
  { event := event77847
    frameStart := 77797 },
  { event := event77848
    frameStart := 77797 },
  { event := event77849
    frameStart := 77797 },
  { event := event77850
    frameStart := 77797 },
  { event := event77851
    frameStart := 77797 },
  { event := event77852
    frameStart := 77797 },
  { event := event77853
    frameStart := 77797 },
  { event := event77854
    frameStart := 77797 },
  { event := event77855
    frameStart := 77797 }
]

def eventLeaf4866 : Array AnnotatedEvent := #[
  { event := event77856
    frameStart := 77797 },
  { event := event77857
    frameStart := 77797 },
  { event := event77858
    frameStart := 77797 },
  { event := event77859
    frameStart := 77797 },
  { event := event77860
    frameStart := 77797 },
  { event := event77861
    frameStart := 77797 },
  { event := event77862
    frameStart := 77797 },
  { event := event77863
    frameStart := 77797 },
  { event := event77864
    frameStart := 77797 },
  { event := event77865
    frameStart := 77797 },
  { event := event77866
    frameStart := 77797 },
  { event := event77867
    frameStart := 77797 },
  { event := event77868
    frameStart := 77797 },
  { event := event77869
    frameStart := 77797 },
  { event := event77870
    frameStart := 77797 },
  { event := event77871
    frameStart := 77797 }
]

def eventLeaf4867 : Array AnnotatedEvent := #[
  { event := event77872
    frameStart := 77797 },
  { event := event77873
    frameStart := 77797 },
  { event := event77874
    frameStart := 77797 },
  { event := event77875
    frameStart := 77797 },
  { event := event77876
    frameStart := 77797 },
  { event := event77877
    frameStart := 77797 },
  { event := event77878
    frameStart := 77797 },
  { event := event77879
    frameStart := 77797 },
  { event := event77880
    frameStart := 77797 },
  { event := event77881
    frameStart := 77797 },
  { event := event77882
    frameStart := 77797 },
  { event := event77883
    frameStart := 77797 },
  { event := event77884
    frameStart := 77797 },
  { event := event77885
    frameStart := 77797 },
  { event := event77886
    frameStart := 77797 },
  { event := event77887
    frameStart := 77797 }
]

def eventLeaf4868 : Array AnnotatedEvent := #[
  { event := event77888
    frameStart := 77797 },
  { event := event77889
    frameStart := 77797 },
  { event := event77890
    frameStart := 77797 },
  { event := event77891
    frameStart := 77797 },
  { event := event77892
    frameStart := 77797 },
  { event := event77893
    frameStart := 77797 },
  { event := event77894
    frameStart := 77797 },
  { event := event77895
    frameStart := 77797 },
  { event := event77896
    frameStart := 77797 },
  { event := event77897
    frameStart := 77797 },
  { event := event77898
    frameStart := 77797 },
  { event := event77899
    frameStart := 77797 },
  { event := event77900
    frameStart := 77797 },
  { event := event77901
    frameStart := 0 },
  { event := event77902
    frameStart := 0 },
  { event := event77903
    frameStart := 0 }
]

def eventLeaf4869 : Array AnnotatedEvent := #[
  { event := event77904
    frameStart := 0 },
  { event := event77905
    frameStart := 0 },
  { event := event77906
    frameStart := 0 },
  { event := event77907
    frameStart := 0 },
  { event := event77908
    frameStart := 0 },
  { event := event77909
    frameStart := 0 },
  { event := event77910
    frameStart := 0 },
  { event := event77911
    frameStart := 0 },
  { event := event77912
    frameStart := 0 },
  { event := event77913
    frameStart := 0 },
  { event := event77914
    frameStart := 0 },
  { event := event77915
    frameStart := 0 },
  { event := event77916
    frameStart := 0 },
  { event := event77917
    frameStart := 0 },
  { event := event77918
    frameStart := 0 },
  { event := event77919
    frameStart := 0 }
]

def eventLeaf4870 : Array AnnotatedEvent := #[
  { event := event77920
    frameStart := 0 },
  { event := event77921
    frameStart := 0 },
  { event := event77922
    frameStart := 0 },
  { event := event77923
    frameStart := 0 },
  { event := event77924
    frameStart := 0 },
  { event := event77925
    frameStart := 0 },
  { event := event77926
    frameStart := 0 },
  { event := event77927
    frameStart := 0 },
  { event := event77928
    frameStart := 0 },
  { event := event77929
    frameStart := 0 },
  { event := event77930
    frameStart := 0 },
  { event := event77931
    frameStart := 0 },
  { event := event77932
    frameStart := 0 },
  { event := event77933
    frameStart := 0 },
  { event := event77934
    frameStart := 0 },
  { event := event77935
    frameStart := 0 }
]

def eventLeaf4871 : Array AnnotatedEvent := #[
  { event := event77936
    frameStart := 0 },
  { event := event77937
    frameStart := 0 },
  { event := event77938
    frameStart := 0 },
  { event := event77939
    frameStart := 0 },
  { event := event77940
    frameStart := 0 },
  { event := event77941
    frameStart := 0 },
  { event := event77942
    frameStart := 0 },
  { event := event77943
    frameStart := 0 },
  { event := event77944
    frameStart := 0 },
  { event := event77945
    frameStart := 0 },
  { event := event77946
    frameStart := 0 },
  { event := event77947
    frameStart := 0 },
  { event := event77948
    frameStart := 0 },
  { event := event77949
    frameStart := 0 },
  { event := event77950
    frameStart := 0 },
  { event := event77951
    frameStart := 0 }
]

def eventLeaf4872 : Array AnnotatedEvent := #[
  { event := event77952
    frameStart := 0 },
  { event := event77953
    frameStart := 0 },
  { event := event77954
    frameStart := 0 },
  { event := event77955
    frameStart := 77955 },
  { event := event77956
    frameStart := 77955 },
  { event := event77957
    frameStart := 77955 },
  { event := event77958
    frameStart := 77955 },
  { event := event77959
    frameStart := 77955 },
  { event := event77960
    frameStart := 77955 },
  { event := event77961
    frameStart := 77955 },
  { event := event77962
    frameStart := 77955 },
  { event := event77963
    frameStart := 77955 },
  { event := event77964
    frameStart := 77955 },
  { event := event77965
    frameStart := 77955 },
  { event := event77966
    frameStart := 77955 },
  { event := event77967
    frameStart := 77955 }
]

def eventLeaf4873 : Array AnnotatedEvent := #[
  { event := event77968
    frameStart := 77955 },
  { event := event77969
    frameStart := 77955 },
  { event := event77970
    frameStart := 77955 },
  { event := event77971
    frameStart := 77955 },
  { event := event77972
    frameStart := 77955 },
  { event := event77973
    frameStart := 77955 },
  { event := event77974
    frameStart := 77955 },
  { event := event77975
    frameStart := 77955 },
  { event := event77976
    frameStart := 77955 },
  { event := event77977
    frameStart := 77955 },
  { event := event77978
    frameStart := 77955 },
  { event := event77979
    frameStart := 77955 },
  { event := event77980
    frameStart := 77955 },
  { event := event77981
    frameStart := 77955 },
  { event := event77982
    frameStart := 77955 },
  { event := event77983
    frameStart := 77955 }
]

def eventLeaf4874 : Array AnnotatedEvent := #[
  { event := event77984
    frameStart := 77955 },
  { event := event77985
    frameStart := 77955 },
  { event := event77986
    frameStart := 77955 },
  { event := event77987
    frameStart := 77955 },
  { event := event77988
    frameStart := 77955 },
  { event := event77989
    frameStart := 77955 },
  { event := event77990
    frameStart := 77955 },
  { event := event77991
    frameStart := 77955 },
  { event := event77992
    frameStart := 77955 },
  { event := event77993
    frameStart := 77955 },
  { event := event77994
    frameStart := 77955 },
  { event := event77995
    frameStart := 77955 },
  { event := event77996
    frameStart := 77955 },
  { event := event77997
    frameStart := 77955 },
  { event := event77998
    frameStart := 77955 },
  { event := event77999
    frameStart := 77955 }
]

def eventLeaf4875 : Array AnnotatedEvent := #[
  { event := event78000
    frameStart := 77955 },
  { event := event78001
    frameStart := 77955 },
  { event := event78002
    frameStart := 77955 },
  { event := event78003
    frameStart := 77955 },
  { event := event78004
    frameStart := 77955 },
  { event := event78005
    frameStart := 77955 },
  { event := event78006
    frameStart := 77955 },
  { event := event78007
    frameStart := 77955 },
  { event := event78008
    frameStart := 77955 },
  { event := event78009
    frameStart := 78009 },
  { event := event78010
    frameStart := 78009 },
  { event := event78011
    frameStart := 78009 },
  { event := event78012
    frameStart := 78009 },
  { event := event78013
    frameStart := 78009 },
  { event := event78014
    frameStart := 78009 },
  { event := event78015
    frameStart := 78009 }
]

def eventLeaf4876 : Array AnnotatedEvent := #[
  { event := event78016
    frameStart := 78009 },
  { event := event78017
    frameStart := 78009 },
  { event := event78018
    frameStart := 78009 },
  { event := event78019
    frameStart := 78009 },
  { event := event78020
    frameStart := 78009 },
  { event := event78021
    frameStart := 78009 },
  { event := event78022
    frameStart := 78009 },
  { event := event78023
    frameStart := 78009 },
  { event := event78024
    frameStart := 78009 },
  { event := event78025
    frameStart := 78009 },
  { event := event78026
    frameStart := 78009 },
  { event := event78027
    frameStart := 78009 },
  { event := event78028
    frameStart := 78009 },
  { event := event78029
    frameStart := 78009 },
  { event := event78030
    frameStart := 78009 },
  { event := event78031
    frameStart := 78009 }
]

def eventLeaf4877 : Array AnnotatedEvent := #[
  { event := event78032
    frameStart := 78009 },
  { event := event78033
    frameStart := 78009 },
  { event := event78034
    frameStart := 78009 },
  { event := event78035
    frameStart := 78009 },
  { event := event78036
    frameStart := 78009 },
  { event := event78037
    frameStart := 78009 },
  { event := event78038
    frameStart := 78009 },
  { event := event78039
    frameStart := 78009 },
  { event := event78040
    frameStart := 78009 },
  { event := event78041
    frameStart := 78009 },
  { event := event78042
    frameStart := 78009 },
  { event := event78043
    frameStart := 78009 },
  { event := event78044
    frameStart := 78009 },
  { event := event78045
    frameStart := 78009 },
  { event := event78046
    frameStart := 78009 },
  { event := event78047
    frameStart := 78009 }
]

def eventLeaf4878 : Array AnnotatedEvent := #[
  { event := event78048
    frameStart := 78009 },
  { event := event78049
    frameStart := 78009 },
  { event := event78050
    frameStart := 78009 },
  { event := event78051
    frameStart := 78009 },
  { event := event78052
    frameStart := 78009 },
  { event := event78053
    frameStart := 78009 },
  { event := event78054
    frameStart := 78009 },
  { event := event78055
    frameStart := 78009 },
  { event := event78056
    frameStart := 78009 },
  { event := event78057
    frameStart := 78009 },
  { event := event78058
    frameStart := 78009 },
  { event := event78059
    frameStart := 78009 },
  { event := event78060
    frameStart := 78009 },
  { event := event78061
    frameStart := 78009 },
  { event := event78062
    frameStart := 78009 },
  { event := event78063
    frameStart := 78009 }
]

def eventLeaf4879 : Array AnnotatedEvent := #[
  { event := event78064
    frameStart := 78009 },
  { event := event78065
    frameStart := 78009 },
  { event := event78066
    frameStart := 78009 },
  { event := event78067
    frameStart := 78009 },
  { event := event78068
    frameStart := 78009 },
  { event := event78069
    frameStart := 78009 },
  { event := event78070
    frameStart := 78009 },
  { event := event78071
    frameStart := 78009 },
  { event := event78072
    frameStart := 78009 },
  { event := event78073
    frameStart := 78009 },
  { event := event78074
    frameStart := 78009 },
  { event := event78075
    frameStart := 78009 },
  { event := event78076
    frameStart := 78009 },
  { event := event78077
    frameStart := 78009 },
  { event := event78078
    frameStart := 78009 },
  { event := event78079
    frameStart := 78009 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events304
