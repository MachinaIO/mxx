import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events101

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event25856 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19615⟩⟩, .operator (⟨21512, 0⟩, ⟨25850, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19612⟩⟩]⟩, (1)⟩)

def event25857 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨19613⟩⟩)

def event25858 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event25859 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event25860 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨4989⟩⟩) (.authority (.operator))

def event25861 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨4989⟩⟩) (.finite 5)

def event25862 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event25863 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event25864 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event25865 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event25866 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 25865

def event25867 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 25863

def event25868 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 25866 .coefficient) (.value (.predecessor 1 25867 .coefficient)))

def event25869 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event25870 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5514⟩⟩) 0 ⟨5503⟩ 25869

def event25871 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5514⟩⟩) 1 ⟨4989⟩ 25861

def event25872 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5514⟩⟩) (.sum [.predecessor 0 25870 .coefficient, .predecessor 1 25871 .coefficient])

def event25873 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5514⟩⟩) (.finite 222)

def event25874 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5554⟩⟩) 0 ⟨5514⟩ 25873

def event25875 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5554⟩⟩) 1 ⟨961⟩ 25859

def event25876 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5554⟩⟩) (.identity (.predecessor 1 25875 .coefficient))

def event25877 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5554⟩⟩) (.finite 224)

def event25878 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11565⟩⟩) 0 ⟨5554⟩ 25877

def event25879 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11565⟩⟩) (.authority (.programFamilyFact))

def exact25880RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11565⟩⟩], []⟩, (1)⟩]

theorem exact25880RawTermsValid :
    exact25880RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event25880 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11565⟩⟩) exact25880RawTerms (.finite 22) 25879 .exactZero (none)

def event25881 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14451⟩⟩) 0 ⟨5554⟩ 25877

def event25882 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14451⟩⟩) (.authority (.programFamilyFact))

def exact25883RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨14451⟩⟩], []⟩, (1)⟩]

theorem exact25883RawTermsValid :
    exact25883RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event25883 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14451⟩⟩) exact25883RawTerms (.finite 22) 25882 .exactZero (none)

def event25884 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14452⟩⟩) 0 ⟨14451⟩ 25883

def event25885 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14452⟩⟩) 1 ⟨11565⟩ 25880

def event25886 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14452⟩⟩) (.product (.predecessor 0 25884 .coefficient) (.predecessor 1 25885 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event25887 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14452⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨11565⟩⟩, ⟨.program ⟨214⟩, ⟨14451⟩⟩], []⟩) [⟨.result 25883 .coefficient, true, some 1⟩, ⟨.result 25880 .coefficient, true, some 1⟩])

def event25888 : Event := .survivorFold (1) 25887

def exact25889RawTerms : List Term := []

theorem exact25889RawTermsValid :
    exact25889RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event25889 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14452⟩⟩) exact25889RawTerms (.finite 484) 25886 (.finite 484) (some (25887))

def event25890 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14453⟩⟩) 0 ⟨14452⟩ 25889

def event25891 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14453⟩⟩) (.identity (.predecessor 0 25890 .coefficient))

def event25892 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨14453⟩⟩) (.finite 484)

def event25893 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19612⟩⟩) 0 ⟨14453⟩ 25892

def event25894 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19612⟩⟩) (.authority (.relationPreimageSource ⟨16⟩))

def exact25895RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨19612⟩⟩]⟩, (1)⟩]

theorem exact25895RawTermsValid :
    exact25895RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event25895 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19612⟩⟩) exact25895RawTerms (.finite 136065468) 25894 .exactZero (none)

def event25896 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6⟩⟩) (.authority (.operator))

def exact25897RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩]⟩, (1)⟩]

theorem exact25897RawTermsValid :
    exact25897RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event25897 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6⟩⟩) exact25897RawTerms .large 25896 .exactZero (none)

def event25898 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19613⟩⟩) 0 ⟨6⟩ 25897

def event25899 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19613⟩⟩) 1 ⟨19612⟩ 25895

def event25900 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19613⟩⟩) (.product (.predecessor 0 25898 .coefficient) (.predecessor 1 25899 .coefficient) (⟨false, false, none, none, none⟩))

def event25901 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19613⟩⟩, .operator (⟨25897, 0⟩, ⟨25895, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19612⟩⟩]⟩, (1)⟩)

def exact25902RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19612⟩⟩]⟩, (1)⟩]

theorem exact25902RawTermsValid :
    exact25902RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event25902 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19613⟩⟩) exact25902RawTerms .large 25900 .exactZero (none)

def event25903 : Event := .preFoldPolynomial 25902 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19612⟩⟩]⟩, (1)⟩] .exactZero none

def exact25904RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19612⟩⟩]⟩, (1)⟩]

def event25904 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨19613⟩⟩) 25903 exact25904RawTerms .large 25900 .exactZero (none)

def event25905 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨26162⟩⟩)

def event25906 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event25907 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event25908 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨4989⟩⟩) (.authority (.operator))

def event25909 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨4989⟩⟩) (.finite 5)

def event25910 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event25911 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event25912 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event25913 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event25914 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 25913

def event25915 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 25911

def event25916 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 25914 .coefficient) (.value (.predecessor 1 25915 .coefficient)))

def event25917 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event25918 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5514⟩⟩) 0 ⟨5503⟩ 25917

def event25919 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5514⟩⟩) 1 ⟨4989⟩ 25909

def event25920 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5514⟩⟩) (.sum [.predecessor 0 25918 .coefficient, .predecessor 1 25919 .coefficient])

def event25921 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5514⟩⟩) (.finite 222)

def event25922 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5554⟩⟩) 0 ⟨5514⟩ 25921

def event25923 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5554⟩⟩) 1 ⟨961⟩ 25907

def event25924 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5554⟩⟩) (.identity (.predecessor 1 25923 .coefficient))

def event25925 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5554⟩⟩) (.finite 224)

def event25926 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11565⟩⟩) 0 ⟨5554⟩ 25925

def event25927 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11565⟩⟩) (.authority (.programFamilyFact))

def exact25928RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11565⟩⟩], []⟩, (1)⟩]

theorem exact25928RawTermsValid :
    exact25928RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event25928 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11565⟩⟩) exact25928RawTerms (.finite 22) 25927 .exactZero (none)

def event25929 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14451⟩⟩) 0 ⟨5554⟩ 25925

def event25930 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14451⟩⟩) (.authority (.programFamilyFact))

def exact25931RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨14451⟩⟩], []⟩, (1)⟩]

theorem exact25931RawTermsValid :
    exact25931RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event25931 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14451⟩⟩) exact25931RawTerms (.finite 22) 25930 .exactZero (none)

def event25932 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14452⟩⟩) 0 ⟨14451⟩ 25931

def event25933 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14452⟩⟩) 1 ⟨11565⟩ 25928

def event25934 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14452⟩⟩) (.product (.predecessor 0 25932 .coefficient) (.predecessor 1 25933 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event25935 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨14452⟩⟩, .operator (⟨25931, 0⟩, ⟨25928, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨11565⟩⟩, ⟨.program ⟨214⟩, ⟨14451⟩⟩], []⟩, (1)⟩)

def exact25936RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11565⟩⟩, ⟨.program ⟨214⟩, ⟨14451⟩⟩], []⟩, (1)⟩]

theorem exact25936RawTermsValid :
    exact25936RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event25936 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14452⟩⟩) exact25936RawTerms (.finite 484) 25934 .exactZero (none)

def event25937 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14453⟩⟩) 0 ⟨14452⟩ 25936

def event25938 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14453⟩⟩) (.identity (.predecessor 0 25937 .coefficient))

def event25939 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨14453⟩⟩) (.finite 484)

def event25940 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23631⟩⟩) 0 ⟨14453⟩ 25939

def event25941 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23631⟩⟩) (.authority (.programFamilyFact))

def event25942 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨23631⟩⟩) (.finite 3720)

def event25943 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨6689⟩⟩) .missing

def event25944 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23632⟩⟩) 0 ⟨6689⟩ 25943

def event25945 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23632⟩⟩) 1 ⟨23631⟩ 25942

def event25946 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23632⟩⟩) (.authority (.operator))

def exact25947RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨23632⟩⟩]⟩, (1)⟩]

theorem exact25947RawTermsValid :
    exact25947RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event25947 : Event := .resultExact (⟨.program ⟨214⟩, ⟨23632⟩⟩) exact25947RawTerms .large 25946 .exactZero (none)

def event25948 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26158⟩⟩) 0 ⟨23632⟩ 25947

def event25949 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26158⟩⟩) (.authority (.operator))

def exact25950RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨26158⟩⟩]⟩, (1)⟩]

theorem exact25950RawTermsValid :
    exact25950RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event25950 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26158⟩⟩) exact25950RawTerms (.finite 8192) 25949 .exactZero (none)

def event25951 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨110⟩⟩) (.authority (.operator))

def event25952 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨110⟩⟩) .exactZero

def event25953 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14543⟩⟩) 0 ⟨14453⟩ 25939

def event25954 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14543⟩⟩) 1 ⟨110⟩ 25952

def event25955 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14543⟩⟩) (.sum [.predecessor 0 25953 .coefficient, .predecessor 1 25954 .coefficient])

def event25956 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨14543⟩⟩) (.finite 484)

def event25957 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14544⟩⟩) 0 ⟨14543⟩ 25956

def event25958 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14544⟩⟩) (.identity (.predecessor 0 25957 .coefficient))

def exact25959RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11565⟩⟩, ⟨.program ⟨214⟩, ⟨14451⟩⟩], []⟩, (1)⟩]

theorem exact25959RawTermsValid :
    exact25959RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event25959 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14544⟩⟩) exact25959RawTerms (.finite 484) 25958 .exactZero (none)

def event25960 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6544⟩⟩) (.authority (.factStore))

def exact25961RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact25961RawTermsValid :
    exact25961RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event25961 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6544⟩⟩) exact25961RawTerms .large 25960 .exactZero (none)

def event25962 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14545⟩⟩) 0 ⟨6544⟩ 25961

def event25963 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14545⟩⟩) 1 ⟨14544⟩ 25959

def event25964 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14545⟩⟩) (.product (.predecessor 0 25962 .coefficient) (.predecessor 1 25963 .coefficient) (⟨false, false, none, none, none⟩))

def event25965 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨14545⟩⟩, .operator (⟨25961, 0⟩, ⟨25959, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨11565⟩⟩, ⟨.program ⟨214⟩, ⟨14451⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact25966RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11565⟩⟩, ⟨.program ⟨214⟩, ⟨14451⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact25966RawTermsValid :
    exact25966RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event25966 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14545⟩⟩) exact25966RawTerms .large 25964 .exactZero (none)

def event25967 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.authority (.operator))

def event25968 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.finite 1)

def event25969 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6757⟩⟩) 0 ⟨6689⟩ 25943

def event25970 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6757⟩⟩) (.authority (.operator))

def exact25971RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6757⟩⟩]⟩, (1)⟩]

theorem exact25971RawTermsValid :
    exact25971RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event25971 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6757⟩⟩) exact25971RawTerms .large 25970 .exactZero (none)

def event25972 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6780⟩⟩) 0 ⟨6757⟩ 25971

def event25973 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6780⟩⟩) (.identity (.predecessor 0 25972 .coefficient))

def exact25974RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6780⟩⟩]⟩, (1)⟩]

theorem exact25974RawTermsValid :
    exact25974RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event25974 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6780⟩⟩) exact25974RawTerms .large 25973 .exactZero (none)

def event25975 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7855⟩⟩) 0 ⟨6780⟩ 25974

def event25976 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7855⟩⟩) (.authority (.operator))

def exact25977RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨7855⟩⟩]⟩, (1)⟩]

theorem exact25977RawTermsValid :
    exact25977RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event25977 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7855⟩⟩) exact25977RawTerms (.finite 8192) 25976 .exactZero (none)

def event25978 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7856⟩⟩) 0 ⟨7855⟩ 25977

def event25979 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7856⟩⟩) 1 ⟨2348⟩ 25968

def event25980 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7856⟩⟩) (.scale (.predecessor 0 25978 .coefficient) (.value (.predecessor 1 25979 .coefficient)))

def exact25981RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨7855⟩⟩]⟩, (1)⟩]

theorem exact25981RawTermsValid :
    exact25981RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event25981 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7856⟩⟩) exact25981RawTerms (.finite 8192) 25980 .exactZero (none)

def event25982 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6761⟩⟩) 0 ⟨6757⟩ 25971

def event25983 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6761⟩⟩) (.identity (.predecessor 0 25982 .coefficient))

def exact25984RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6761⟩⟩]⟩, (1)⟩]

theorem exact25984RawTermsValid :
    exact25984RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event25984 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6761⟩⟩) exact25984RawTerms .large 25983 .exactZero (none)

def event25985 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7857⟩⟩) 0 ⟨6761⟩ 25984

def event25986 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7857⟩⟩) 1 ⟨7856⟩ 25981

def event25987 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7857⟩⟩) (.product (.predecessor 0 25985 .coefficient) (.predecessor 1 25986 .coefficient) (⟨false, false, none, none, none⟩))

def event25988 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7857⟩⟩, .operator (⟨25984, 0⟩, ⟨25981, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6761⟩⟩, ⟨.program ⟨214⟩, ⟨7855⟩⟩]⟩, (1)⟩)

def exact25989RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6761⟩⟩, ⟨.program ⟨214⟩, ⟨7855⟩⟩]⟩, (1)⟩]

theorem exact25989RawTermsValid :
    exact25989RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event25989 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7857⟩⟩) exact25989RawTerms .large 25987 .exactZero (none)

def event25990 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14546⟩⟩) 0 ⟨7857⟩ 25989

def event25991 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14546⟩⟩) 1 ⟨14545⟩ 25966

def event25992 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14546⟩⟩) (.sum [.predecessor 0 25990 .coefficient, .predecessor 1 25991 .coefficient])

def exact25993RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6761⟩⟩, ⟨.program ⟨214⟩, ⟨7855⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨11565⟩⟩, ⟨.program ⟨214⟩, ⟨14451⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact25993RawTermsValid :
    exact25993RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event25993 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14546⟩⟩) exact25993RawTerms .large 25992 .exactZero (none)

def event25994 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26161⟩⟩) 0 ⟨14546⟩ 25993

def event25995 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26161⟩⟩) 1 ⟨26158⟩ 25950

def event25996 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26161⟩⟩) (.product (.predecessor 0 25994 .coefficient) (.predecessor 1 25995 .coefficient) (⟨false, false, none, none, none⟩))

def event25997 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26161⟩⟩, .operator (⟨25993, 0⟩, ⟨25950, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6761⟩⟩, ⟨.program ⟨214⟩, ⟨7855⟩⟩, ⟨.program ⟨214⟩, ⟨26158⟩⟩]⟩, (1)⟩)

def event25998 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26161⟩⟩, .operator (⟨25993, 1⟩, ⟨25950, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨11565⟩⟩, ⟨.program ⟨214⟩, ⟨14451⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26158⟩⟩]⟩, (-1)⟩)

def event25999 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨26161⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨11565⟩⟩, ⟨.program ⟨214⟩, ⟨14451⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26158⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨26158⟩⟩) ⟨23632⟩ 25947)

def event26000 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26161⟩⟩, .relation 25999 0, ⟨[⟨.program ⟨214⟩, ⟨11565⟩⟩, ⟨.program ⟨214⟩, ⟨14451⟩⟩], [⟨.program ⟨214⟩, ⟨23632⟩⟩]⟩, (-1)⟩)

def exact26001RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6761⟩⟩, ⟨.program ⟨214⟩, ⟨7855⟩⟩, ⟨.program ⟨214⟩, ⟨26158⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨11565⟩⟩, ⟨.program ⟨214⟩, ⟨14451⟩⟩], [⟨.program ⟨214⟩, ⟨23632⟩⟩]⟩, (-1)⟩]

theorem exact26001RawTermsValid :
    exact26001RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26001 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26161⟩⟩) exact26001RawTerms .large 25996 .exactZero (none)

def event26002 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16071⟩⟩) 0 ⟨14453⟩ 25939

def event26003 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16071⟩⟩) (.authority (.programFamilyFact))

def exact26004RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16071⟩⟩], []⟩, (1)⟩]

theorem exact26004RawTermsValid :
    exact26004RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26004 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16071⟩⟩) exact26004RawTerms (.finite 22) 26003 .exactZero (none)

def event26005 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16073⟩⟩) 0 ⟨6544⟩ 25961

def event26006 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16073⟩⟩) 1 ⟨16071⟩ 26004

def event26007 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16073⟩⟩) (.product (.predecessor 0 26005 .coefficient) (.predecessor 1 26006 .coefficient) (⟨false, true, none, none, some 1⟩))

def event26008 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨16073⟩⟩, .operator (⟨25961, 0⟩, ⟨26004, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨16071⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact26009RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16071⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact26009RawTermsValid :
    exact26009RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26009 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16073⟩⟩) exact26009RawTerms .large 26007 .exactZero (none)

def event26010 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6698⟩⟩) 0 ⟨6689⟩ 25943

def event26011 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6698⟩⟩) (.authority (.operator))

def exact26012RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6698⟩⟩]⟩, (1)⟩]

theorem exact26012RawTermsValid :
    exact26012RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26012 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6698⟩⟩) exact26012RawTerms .large 26011 .exactZero (none)

def event26013 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16074⟩⟩) 0 ⟨6698⟩ 26012

def event26014 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16074⟩⟩) 1 ⟨16073⟩ 26009

def event26015 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16074⟩⟩) (.sum [.predecessor 0 26013 .coefficient, .predecessor 1 26014 .coefficient])

def exact26016RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6698⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16071⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact26016RawTermsValid :
    exact26016RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26016 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16074⟩⟩) exact26016RawTerms .large 26015 .exactZero (none)

def event26017 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26162⟩⟩) 0 ⟨16074⟩ 26016

def event26018 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26162⟩⟩) 1 ⟨26161⟩ 26001

def event26019 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26162⟩⟩) (.sum [.predecessor 0 26017 .coefficient, .predecessor 1 26018 .coefficient])

def exact26020RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6698⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6761⟩⟩, ⟨.program ⟨214⟩, ⟨7855⟩⟩, ⟨.program ⟨214⟩, ⟨26158⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨11565⟩⟩, ⟨.program ⟨214⟩, ⟨14451⟩⟩], [⟨.program ⟨214⟩, ⟨23632⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16071⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact26020RawTermsValid :
    exact26020RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26020 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26162⟩⟩) exact26020RawTerms .large 26019 .exactZero (none)

def event26021 : Event := .preFoldPolynomial 26020 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6698⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6761⟩⟩, ⟨.program ⟨214⟩, ⟨7855⟩⟩, ⟨.program ⟨214⟩, ⟨26158⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨11565⟩⟩, ⟨.program ⟨214⟩, ⟨14451⟩⟩], [⟨.program ⟨214⟩, ⟨23632⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16071⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩] .exactZero none

def exact26022RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6698⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6761⟩⟩, ⟨.program ⟨214⟩, ⟨7855⟩⟩, ⟨.program ⟨214⟩, ⟨26158⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨11565⟩⟩, ⟨.program ⟨214⟩, ⟨14451⟩⟩], [⟨.program ⟨214⟩, ⟨23632⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16071⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

def event26022 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨26162⟩⟩) 26021 exact26022RawTerms .large 26019 .exactZero (none)

def event26023 : Event := .specializationComputed (⟨.program ⟨214⟩, ⟨14453⟩⟩) ⟨⟨111⟩, ⟨16⟩, ⟨109⟩⟩ ⟨25857, 26023⟩

def event26024 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨19615⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19612⟩⟩]⟩) (1) 0 2 (.universal 26023 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19612⟩⟩]⟩) (none) 26022)

def event26025 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19615⟩⟩, .relation 26024 0, ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6698⟩⟩]⟩, (1)⟩)

def event26026 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19615⟩⟩, .relation 26024 1, ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6761⟩⟩, ⟨.program ⟨214⟩, ⟨7855⟩⟩, ⟨.program ⟨214⟩, ⟨26158⟩⟩]⟩, (-1)⟩)

def event26027 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19615⟩⟩, .relation 26024 2, ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨11565⟩⟩, ⟨.program ⟨214⟩, ⟨14451⟩⟩], [⟨.program ⟨214⟩, ⟨23632⟩⟩]⟩, (1)⟩)

def event26028 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19615⟩⟩, .relation 26024 3, ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨16071⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact26029RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6698⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6761⟩⟩, ⟨.program ⟨214⟩, ⟨7855⟩⟩, ⟨.program ⟨214⟩, ⟨26158⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨11565⟩⟩, ⟨.program ⟨214⟩, ⟨14451⟩⟩], [⟨.program ⟨214⟩, ⟨23632⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨16071⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact26029RawTermsValid :
    exact26029RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26029 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19615⟩⟩) exact26029RawTerms .large 25853 (.finite 1811303510016) (some (25855))

def event26030 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26160⟩⟩) 0 ⟨19615⟩ 26029

def event26031 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26160⟩⟩) 1 ⟨26159⟩ 25843

def event26032 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26160⟩⟩) (.sum [.predecessor 0 26030 .coefficient, .predecessor 1 26031 .coefficient])

def event26033 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26160⟩⟩, .operator (⟨26029, 2⟩, ⟨25843, 1⟩), ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨11565⟩⟩, ⟨.program ⟨214⟩, ⟨14451⟩⟩], [⟨.program ⟨214⟩, ⟨23632⟩⟩]⟩, (-1)⟩)

def event26034 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26160⟩⟩, .operator (⟨26029, 1⟩, ⟨25843, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6761⟩⟩, ⟨.program ⟨214⟩, ⟨7855⟩⟩, ⟨.program ⟨214⟩, ⟨26158⟩⟩]⟩, (1)⟩)

def event26035 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26160⟩⟩) (.sum [.result 26029 .summary, .result 25843 .summary])

def exact26036RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6698⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨16071⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact26036RawTermsValid :
    exact26036RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26036 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26160⟩⟩) exact26036RawTerms .large 26032 (.finite 352072932929536) (some (26035))

def event26037 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28124⟩⟩) 0 ⟨26160⟩ 26036

def event26038 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28124⟩⟩) 1 ⟨28122⟩ 25759

def event26039 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28124⟩⟩) (.product (.predecessor 0 26037 .coefficient) (.predecessor 1 26038 .coefficient) (⟨false, false, none, none, none⟩))

def event26040 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28124⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨28122⟩⟩]⟩) [⟨.result 25759 .coefficient, false, none⟩])

def event26041 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28124⟩⟩) (.product (.result 26036 .summary) (.transfer 26040) (⟨false, false, none, none, none⟩))

def event26042 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28124⟩⟩, .operator (⟨26036, 0⟩, ⟨25759, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6698⟩⟩, ⟨.program ⟨214⟩, ⟨28122⟩⟩]⟩, (1)⟩)

def event26043 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28124⟩⟩, .operator (⟨26036, 1⟩, ⟨25759, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨16071⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28122⟩⟩]⟩, (-1)⟩)

def event26044 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨28124⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨16071⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28122⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨28122⟩⟩) ⟨24234⟩ 25756)

def event26045 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28124⟩⟩, .relation 26044 0, ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨16071⟩⟩], [⟨.program ⟨214⟩, ⟨24234⟩⟩]⟩, (-1)⟩)

def exact26046RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6698⟩⟩, ⟨.program ⟨214⟩, ⟨28122⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨16071⟩⟩], [⟨.program ⟨214⟩, ⟨24234⟩⟩]⟩, (-1)⟩]

theorem exact26046RawTermsValid :
    exact26046RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26046 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28124⟩⟩) exact26046RawTerms .large 26039 (.finite 1292113297018323992576) (some (26041))

def event26047 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21556⟩⟩) 0 ⟨16072⟩ 1066

def event26048 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21556⟩⟩) (.authority (.relationPreimageSource ⟨46⟩))

def exact26049RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨21556⟩⟩]⟩, (1)⟩]

theorem exact26049RawTermsValid :
    exact26049RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26049 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21556⟩⟩) exact26049RawTerms (.finite 136065468) 26048 .exactZero (none)

def event26050 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21558⟩⟩) 0 ⟨21556⟩ 26049

def event26051 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21558⟩⟩) 1 ⟨2348⟩ 4

def event26052 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21558⟩⟩) (.scale (.predecessor 0 26050 .coefficient) (.value (.predecessor 1 26051 .coefficient)))

def exact26053RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨21556⟩⟩]⟩, (1)⟩]

theorem exact26053RawTermsValid :
    exact26053RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26053 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21558⟩⟩) exact26053RawTerms (.finite 136065468) 26052 .exactZero (none)

def event26054 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21559⟩⟩) 0 ⟨5559⟩ 21512

def event26055 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21559⟩⟩) 1 ⟨21558⟩ 26053

def event26056 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21559⟩⟩) (.product (.predecessor 0 26054 .coefficient) (.predecessor 1 26055 .coefficient) (⟨false, false, none, none, none⟩))

def event26057 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21559⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨21556⟩⟩]⟩) [⟨.result 26049 .coefficient, false, none⟩])

def event26058 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21559⟩⟩) (.product (.result 21512 .summary) (.transfer 26057) (⟨false, false, none, none, none⟩))

def event26059 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21559⟩⟩, .operator (⟨21512, 0⟩, ⟨26053, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21556⟩⟩]⟩, (1)⟩)

def event26060 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨21557⟩⟩)

def event26061 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event26062 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event26063 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨4989⟩⟩) (.authority (.operator))

def event26064 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨4989⟩⟩) (.finite 5)

def event26065 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event26066 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event26067 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event26068 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event26069 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 26068

def event26070 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 26066

def event26071 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 26069 .coefficient) (.value (.predecessor 1 26070 .coefficient)))

def event26072 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event26073 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5514⟩⟩) 0 ⟨5503⟩ 26072

def event26074 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5514⟩⟩) 1 ⟨4989⟩ 26064

def event26075 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5514⟩⟩) (.sum [.predecessor 0 26073 .coefficient, .predecessor 1 26074 .coefficient])

def event26076 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5514⟩⟩) (.finite 222)

def event26077 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5554⟩⟩) 0 ⟨5514⟩ 26076

def event26078 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5554⟩⟩) 1 ⟨961⟩ 26062

def event26079 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5554⟩⟩) (.identity (.predecessor 1 26078 .coefficient))

def event26080 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5554⟩⟩) (.finite 224)

def event26081 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11565⟩⟩) 0 ⟨5554⟩ 26080

def event26082 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11565⟩⟩) (.authority (.programFamilyFact))

def exact26083RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11565⟩⟩], []⟩, (1)⟩]

theorem exact26083RawTermsValid :
    exact26083RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26083 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11565⟩⟩) exact26083RawTerms (.finite 22) 26082 .exactZero (none)

def event26084 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14451⟩⟩) 0 ⟨5554⟩ 26080

def event26085 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14451⟩⟩) (.authority (.programFamilyFact))

def exact26086RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨14451⟩⟩], []⟩, (1)⟩]

theorem exact26086RawTermsValid :
    exact26086RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26086 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14451⟩⟩) exact26086RawTerms (.finite 22) 26085 .exactZero (none)

def event26087 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14452⟩⟩) 0 ⟨14451⟩ 26086

def event26088 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14452⟩⟩) 1 ⟨11565⟩ 26083

def event26089 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14452⟩⟩) (.product (.predecessor 0 26087 .coefficient) (.predecessor 1 26088 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event26090 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14452⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨11565⟩⟩, ⟨.program ⟨214⟩, ⟨14451⟩⟩], []⟩) [⟨.result 26086 .coefficient, true, some 1⟩, ⟨.result 26083 .coefficient, true, some 1⟩])

def event26091 : Event := .survivorFold (1) 26090

def exact26092RawTerms : List Term := []

theorem exact26092RawTermsValid :
    exact26092RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26092 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14452⟩⟩) exact26092RawTerms (.finite 484) 26089 (.finite 484) (some (26090))

def event26093 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14453⟩⟩) 0 ⟨14452⟩ 26092

def event26094 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14453⟩⟩) (.identity (.predecessor 0 26093 .coefficient))

def event26095 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨14453⟩⟩) (.finite 484)

def event26096 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16071⟩⟩) 0 ⟨14453⟩ 26095

def event26097 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16071⟩⟩) (.authority (.programFamilyFact))

def exact26098RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16071⟩⟩], []⟩, (1)⟩]

theorem exact26098RawTermsValid :
    exact26098RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26098 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16071⟩⟩) exact26098RawTerms (.finite 22) 26097 .exactZero (none)

def event26099 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16072⟩⟩) 0 ⟨16071⟩ 26098

def event26100 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16072⟩⟩) (.identity (.predecessor 0 26099 .coefficient))

def event26101 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨16072⟩⟩) (.finite 22)

def event26102 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21556⟩⟩) 0 ⟨16072⟩ 26101

def event26103 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21556⟩⟩) (.authority (.relationPreimageSource ⟨46⟩))

def exact26104RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨21556⟩⟩]⟩, (1)⟩]

theorem exact26104RawTermsValid :
    exact26104RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26104 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21556⟩⟩) exact26104RawTerms (.finite 136065468) 26103 .exactZero (none)

def event26105 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6⟩⟩) (.authority (.operator))

def exact26106RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩]⟩, (1)⟩]

theorem exact26106RawTermsValid :
    exact26106RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26106 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6⟩⟩) exact26106RawTerms .large 26105 .exactZero (none)

def event26107 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21557⟩⟩) 0 ⟨6⟩ 26106

def event26108 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21557⟩⟩) 1 ⟨21556⟩ 26104

def event26109 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21557⟩⟩) (.product (.predecessor 0 26107 .coefficient) (.predecessor 1 26108 .coefficient) (⟨false, false, none, none, none⟩))

def event26110 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21557⟩⟩, .operator (⟨26106, 0⟩, ⟨26104, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21556⟩⟩]⟩, (1)⟩)

def exact26111RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21556⟩⟩]⟩, (1)⟩]

theorem exact26111RawTermsValid :
    exact26111RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26111 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21557⟩⟩) exact26111RawTerms .large 26109 .exactZero (none)

def eventLeaf1616 : Array AnnotatedEvent := #[
  { event := event25856
    frameStart := 0 },
  { event := event25857
    frameStart := 25857 },
  { event := event25858
    frameStart := 25857 },
  { event := event25859
    frameStart := 25857 },
  { event := event25860
    frameStart := 25857 },
  { event := event25861
    frameStart := 25857 },
  { event := event25862
    frameStart := 25857 },
  { event := event25863
    frameStart := 25857 },
  { event := event25864
    frameStart := 25857 },
  { event := event25865
    frameStart := 25857 },
  { event := event25866
    frameStart := 25857 },
  { event := event25867
    frameStart := 25857 },
  { event := event25868
    frameStart := 25857 },
  { event := event25869
    frameStart := 25857 },
  { event := event25870
    frameStart := 25857 },
  { event := event25871
    frameStart := 25857 }
]

def eventLeaf1617 : Array AnnotatedEvent := #[
  { event := event25872
    frameStart := 25857 },
  { event := event25873
    frameStart := 25857 },
  { event := event25874
    frameStart := 25857 },
  { event := event25875
    frameStart := 25857 },
  { event := event25876
    frameStart := 25857 },
  { event := event25877
    frameStart := 25857 },
  { event := event25878
    frameStart := 25857 },
  { event := event25879
    frameStart := 25857 },
  { event := event25880
    frameStart := 25857 },
  { event := event25881
    frameStart := 25857 },
  { event := event25882
    frameStart := 25857 },
  { event := event25883
    frameStart := 25857 },
  { event := event25884
    frameStart := 25857 },
  { event := event25885
    frameStart := 25857 },
  { event := event25886
    frameStart := 25857 },
  { event := event25887
    frameStart := 25857 }
]

def eventLeaf1618 : Array AnnotatedEvent := #[
  { event := event25888
    frameStart := 25857 },
  { event := event25889
    frameStart := 25857 },
  { event := event25890
    frameStart := 25857 },
  { event := event25891
    frameStart := 25857 },
  { event := event25892
    frameStart := 25857 },
  { event := event25893
    frameStart := 25857 },
  { event := event25894
    frameStart := 25857 },
  { event := event25895
    frameStart := 25857 },
  { event := event25896
    frameStart := 25857 },
  { event := event25897
    frameStart := 25857 },
  { event := event25898
    frameStart := 25857 },
  { event := event25899
    frameStart := 25857 },
  { event := event25900
    frameStart := 25857 },
  { event := event25901
    frameStart := 25857 },
  { event := event25902
    frameStart := 25857 },
  { event := event25903
    frameStart := 25857 }
]

def eventLeaf1619 : Array AnnotatedEvent := #[
  { event := event25904
    frameStart := 25857 },
  { event := event25905
    frameStart := 25905 },
  { event := event25906
    frameStart := 25905 },
  { event := event25907
    frameStart := 25905 },
  { event := event25908
    frameStart := 25905 },
  { event := event25909
    frameStart := 25905 },
  { event := event25910
    frameStart := 25905 },
  { event := event25911
    frameStart := 25905 },
  { event := event25912
    frameStart := 25905 },
  { event := event25913
    frameStart := 25905 },
  { event := event25914
    frameStart := 25905 },
  { event := event25915
    frameStart := 25905 },
  { event := event25916
    frameStart := 25905 },
  { event := event25917
    frameStart := 25905 },
  { event := event25918
    frameStart := 25905 },
  { event := event25919
    frameStart := 25905 }
]

def eventLeaf1620 : Array AnnotatedEvent := #[
  { event := event25920
    frameStart := 25905 },
  { event := event25921
    frameStart := 25905 },
  { event := event25922
    frameStart := 25905 },
  { event := event25923
    frameStart := 25905 },
  { event := event25924
    frameStart := 25905 },
  { event := event25925
    frameStart := 25905 },
  { event := event25926
    frameStart := 25905 },
  { event := event25927
    frameStart := 25905 },
  { event := event25928
    frameStart := 25905 },
  { event := event25929
    frameStart := 25905 },
  { event := event25930
    frameStart := 25905 },
  { event := event25931
    frameStart := 25905 },
  { event := event25932
    frameStart := 25905 },
  { event := event25933
    frameStart := 25905 },
  { event := event25934
    frameStart := 25905 },
  { event := event25935
    frameStart := 25905 }
]

def eventLeaf1621 : Array AnnotatedEvent := #[
  { event := event25936
    frameStart := 25905 },
  { event := event25937
    frameStart := 25905 },
  { event := event25938
    frameStart := 25905 },
  { event := event25939
    frameStart := 25905 },
  { event := event25940
    frameStart := 25905 },
  { event := event25941
    frameStart := 25905 },
  { event := event25942
    frameStart := 25905 },
  { event := event25943
    frameStart := 25905 },
  { event := event25944
    frameStart := 25905 },
  { event := event25945
    frameStart := 25905 },
  { event := event25946
    frameStart := 25905 },
  { event := event25947
    frameStart := 25905 },
  { event := event25948
    frameStart := 25905 },
  { event := event25949
    frameStart := 25905 },
  { event := event25950
    frameStart := 25905 },
  { event := event25951
    frameStart := 25905 }
]

def eventLeaf1622 : Array AnnotatedEvent := #[
  { event := event25952
    frameStart := 25905 },
  { event := event25953
    frameStart := 25905 },
  { event := event25954
    frameStart := 25905 },
  { event := event25955
    frameStart := 25905 },
  { event := event25956
    frameStart := 25905 },
  { event := event25957
    frameStart := 25905 },
  { event := event25958
    frameStart := 25905 },
  { event := event25959
    frameStart := 25905 },
  { event := event25960
    frameStart := 25905 },
  { event := event25961
    frameStart := 25905 },
  { event := event25962
    frameStart := 25905 },
  { event := event25963
    frameStart := 25905 },
  { event := event25964
    frameStart := 25905 },
  { event := event25965
    frameStart := 25905 },
  { event := event25966
    frameStart := 25905 },
  { event := event25967
    frameStart := 25905 }
]

def eventLeaf1623 : Array AnnotatedEvent := #[
  { event := event25968
    frameStart := 25905 },
  { event := event25969
    frameStart := 25905 },
  { event := event25970
    frameStart := 25905 },
  { event := event25971
    frameStart := 25905 },
  { event := event25972
    frameStart := 25905 },
  { event := event25973
    frameStart := 25905 },
  { event := event25974
    frameStart := 25905 },
  { event := event25975
    frameStart := 25905 },
  { event := event25976
    frameStart := 25905 },
  { event := event25977
    frameStart := 25905 },
  { event := event25978
    frameStart := 25905 },
  { event := event25979
    frameStart := 25905 },
  { event := event25980
    frameStart := 25905 },
  { event := event25981
    frameStart := 25905 },
  { event := event25982
    frameStart := 25905 },
  { event := event25983
    frameStart := 25905 }
]

def eventLeaf1624 : Array AnnotatedEvent := #[
  { event := event25984
    frameStart := 25905 },
  { event := event25985
    frameStart := 25905 },
  { event := event25986
    frameStart := 25905 },
  { event := event25987
    frameStart := 25905 },
  { event := event25988
    frameStart := 25905 },
  { event := event25989
    frameStart := 25905 },
  { event := event25990
    frameStart := 25905 },
  { event := event25991
    frameStart := 25905 },
  { event := event25992
    frameStart := 25905 },
  { event := event25993
    frameStart := 25905 },
  { event := event25994
    frameStart := 25905 },
  { event := event25995
    frameStart := 25905 },
  { event := event25996
    frameStart := 25905 },
  { event := event25997
    frameStart := 25905 },
  { event := event25998
    frameStart := 25905 },
  { event := event25999
    frameStart := 25905 }
]

def eventLeaf1625 : Array AnnotatedEvent := #[
  { event := event26000
    frameStart := 25905 },
  { event := event26001
    frameStart := 25905 },
  { event := event26002
    frameStart := 25905 },
  { event := event26003
    frameStart := 25905 },
  { event := event26004
    frameStart := 25905 },
  { event := event26005
    frameStart := 25905 },
  { event := event26006
    frameStart := 25905 },
  { event := event26007
    frameStart := 25905 },
  { event := event26008
    frameStart := 25905 },
  { event := event26009
    frameStart := 25905 },
  { event := event26010
    frameStart := 25905 },
  { event := event26011
    frameStart := 25905 },
  { event := event26012
    frameStart := 25905 },
  { event := event26013
    frameStart := 25905 },
  { event := event26014
    frameStart := 25905 },
  { event := event26015
    frameStart := 25905 }
]

def eventLeaf1626 : Array AnnotatedEvent := #[
  { event := event26016
    frameStart := 25905 },
  { event := event26017
    frameStart := 25905 },
  { event := event26018
    frameStart := 25905 },
  { event := event26019
    frameStart := 25905 },
  { event := event26020
    frameStart := 25905 },
  { event := event26021
    frameStart := 25905 },
  { event := event26022
    frameStart := 25905 },
  { event := event26023
    frameStart := 0 },
  { event := event26024
    frameStart := 0 },
  { event := event26025
    frameStart := 0 },
  { event := event26026
    frameStart := 0 },
  { event := event26027
    frameStart := 0 },
  { event := event26028
    frameStart := 0 },
  { event := event26029
    frameStart := 0 },
  { event := event26030
    frameStart := 0 },
  { event := event26031
    frameStart := 0 }
]

def eventLeaf1627 : Array AnnotatedEvent := #[
  { event := event26032
    frameStart := 0 },
  { event := event26033
    frameStart := 0 },
  { event := event26034
    frameStart := 0 },
  { event := event26035
    frameStart := 0 },
  { event := event26036
    frameStart := 0 },
  { event := event26037
    frameStart := 0 },
  { event := event26038
    frameStart := 0 },
  { event := event26039
    frameStart := 0 },
  { event := event26040
    frameStart := 0 },
  { event := event26041
    frameStart := 0 },
  { event := event26042
    frameStart := 0 },
  { event := event26043
    frameStart := 0 },
  { event := event26044
    frameStart := 0 },
  { event := event26045
    frameStart := 0 },
  { event := event26046
    frameStart := 0 },
  { event := event26047
    frameStart := 0 }
]

def eventLeaf1628 : Array AnnotatedEvent := #[
  { event := event26048
    frameStart := 0 },
  { event := event26049
    frameStart := 0 },
  { event := event26050
    frameStart := 0 },
  { event := event26051
    frameStart := 0 },
  { event := event26052
    frameStart := 0 },
  { event := event26053
    frameStart := 0 },
  { event := event26054
    frameStart := 0 },
  { event := event26055
    frameStart := 0 },
  { event := event26056
    frameStart := 0 },
  { event := event26057
    frameStart := 0 },
  { event := event26058
    frameStart := 0 },
  { event := event26059
    frameStart := 0 },
  { event := event26060
    frameStart := 26060 },
  { event := event26061
    frameStart := 26060 },
  { event := event26062
    frameStart := 26060 },
  { event := event26063
    frameStart := 26060 }
]

def eventLeaf1629 : Array AnnotatedEvent := #[
  { event := event26064
    frameStart := 26060 },
  { event := event26065
    frameStart := 26060 },
  { event := event26066
    frameStart := 26060 },
  { event := event26067
    frameStart := 26060 },
  { event := event26068
    frameStart := 26060 },
  { event := event26069
    frameStart := 26060 },
  { event := event26070
    frameStart := 26060 },
  { event := event26071
    frameStart := 26060 },
  { event := event26072
    frameStart := 26060 },
  { event := event26073
    frameStart := 26060 },
  { event := event26074
    frameStart := 26060 },
  { event := event26075
    frameStart := 26060 },
  { event := event26076
    frameStart := 26060 },
  { event := event26077
    frameStart := 26060 },
  { event := event26078
    frameStart := 26060 },
  { event := event26079
    frameStart := 26060 }
]

def eventLeaf1630 : Array AnnotatedEvent := #[
  { event := event26080
    frameStart := 26060 },
  { event := event26081
    frameStart := 26060 },
  { event := event26082
    frameStart := 26060 },
  { event := event26083
    frameStart := 26060 },
  { event := event26084
    frameStart := 26060 },
  { event := event26085
    frameStart := 26060 },
  { event := event26086
    frameStart := 26060 },
  { event := event26087
    frameStart := 26060 },
  { event := event26088
    frameStart := 26060 },
  { event := event26089
    frameStart := 26060 },
  { event := event26090
    frameStart := 26060 },
  { event := event26091
    frameStart := 26060 },
  { event := event26092
    frameStart := 26060 },
  { event := event26093
    frameStart := 26060 },
  { event := event26094
    frameStart := 26060 },
  { event := event26095
    frameStart := 26060 }
]

def eventLeaf1631 : Array AnnotatedEvent := #[
  { event := event26096
    frameStart := 26060 },
  { event := event26097
    frameStart := 26060 },
  { event := event26098
    frameStart := 26060 },
  { event := event26099
    frameStart := 26060 },
  { event := event26100
    frameStart := 26060 },
  { event := event26101
    frameStart := 26060 },
  { event := event26102
    frameStart := 26060 },
  { event := event26103
    frameStart := 26060 },
  { event := event26104
    frameStart := 26060 },
  { event := event26105
    frameStart := 26060 },
  { event := event26106
    frameStart := 26060 },
  { event := event26107
    frameStart := 26060 },
  { event := event26108
    frameStart := 26060 },
  { event := event26109
    frameStart := 26060 },
  { event := event26110
    frameStart := 26060 },
  { event := event26111
    frameStart := 26060 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events101
