import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events320

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event81920 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25451⟩⟩) (.product (.result 81915 .summary) (.transfer 81919) (⟨false, false, none, none, none⟩))

def event81921 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25451⟩⟩, .operator (⟨81915, 1⟩, ⟨81851, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨9925⟩⟩, ⟨.program ⟨214⟩, ⟨12566⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25450⟩⟩]⟩, (-1)⟩)

def event81922 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨25451⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨9925⟩⟩, ⟨.program ⟨214⟩, ⟨12566⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25450⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨25450⟩⟩) ⟨23248⟩ 81848)

def event81923 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25451⟩⟩, .relation 81922 0, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨9925⟩⟩, ⟨.program ⟨214⟩, ⟨12566⟩⟩], [⟨.program ⟨214⟩, ⟨23248⟩⟩]⟩, (-1)⟩)

def event81924 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25451⟩⟩, .operator (⟨81915, 0⟩, ⟨81851, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6766⟩⟩, ⟨.program ⟨214⟩, ⟨7870⟩⟩, ⟨.program ⟨214⟩, ⟨25450⟩⟩]⟩, (1)⟩)

def exact81925RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6766⟩⟩, ⟨.program ⟨214⟩, ⟨7870⟩⟩, ⟨.program ⟨214⟩, ⟨25450⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨9925⟩⟩, ⟨.program ⟨214⟩, ⟨12566⟩⟩], [⟨.program ⟨214⟩, ⟨23248⟩⟩]⟩, (-1)⟩]

theorem exact81925RawTermsValid :
    exact81925RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event81925 : Event := .resultExact (⟨.program ⟨214⟩, ⟨25451⟩⟩) exact81925RawTerms .large 81918 (.finite 350322698485760) (some (81920))

def event81926 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19960⟩⟩) 0 ⟨12568⟩ 3931

def event81927 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19960⟩⟩) (.authority (.relationPreimageSource ⟨21⟩))

def exact81928RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨19960⟩⟩]⟩, (1)⟩]

theorem exact81928RawTermsValid :
    exact81928RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event81928 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19960⟩⟩) exact81928RawTerms (.finite 136065468) 81927 .exactZero (none)

def event81929 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19962⟩⟩) 0 ⟨19960⟩ 81928

def event81930 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19962⟩⟩) 1 ⟨2348⟩ 4

def event81931 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19962⟩⟩) (.scale (.predecessor 0 81929 .coefficient) (.value (.predecessor 1 81930 .coefficient)))

def exact81932RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨19960⟩⟩]⟩, (1)⟩]

theorem exact81932RawTermsValid :
    exact81932RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event81932 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19962⟩⟩) exact81932RawTerms (.finite 136065468) 81931 .exactZero (none)

def event81933 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19963⟩⟩) 0 ⟨5541⟩ 80012

def event81934 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19963⟩⟩) 1 ⟨19962⟩ 81932

def event81935 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19963⟩⟩) (.product (.predecessor 0 81933 .coefficient) (.predecessor 1 81934 .coefficient) (⟨false, false, none, none, none⟩))

def event81936 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19963⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨19960⟩⟩]⟩) [⟨.result 81928 .coefficient, false, none⟩])

def event81937 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19963⟩⟩) (.product (.result 80012 .summary) (.transfer 81936) (⟨false, false, none, none, none⟩))

def event81938 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19963⟩⟩, .operator (⟨80012, 0⟩, ⟨81932, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19960⟩⟩]⟩, (1)⟩)

def event81939 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨19961⟩⟩)

def event81940 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event81941 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event81942 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.authority (.operator))

def event81943 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.finite 1)

def event81944 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event81945 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event81946 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event81947 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event81948 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 81947

def event81949 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 81945

def event81950 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 81948 .coefficient) (.value (.predecessor 1 81949 .coefficient)))

def event81951 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event81952 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5505⟩⟩) 0 ⟨5503⟩ 81951

def event81953 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5505⟩⟩) 1 ⟨2348⟩ 81943

def event81954 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5505⟩⟩) (.sum [.predecessor 0 81952 .coefficient, .predecessor 1 81953 .coefficient])

def event81955 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5505⟩⟩) (.finite 218)

def event81956 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5536⟩⟩) 0 ⟨5505⟩ 81955

def event81957 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5536⟩⟩) 1 ⟨961⟩ 81941

def event81958 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5536⟩⟩) (.identity (.predecessor 1 81957 .coefficient))

def event81959 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5536⟩⟩) (.finite 224)

def event81960 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12566⟩⟩) 0 ⟨5536⟩ 81959

def event81961 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12566⟩⟩) (.authority (.programFamilyFact))

def exact81962RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨12566⟩⟩], []⟩, (1)⟩]

theorem exact81962RawTermsValid :
    exact81962RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event81962 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12566⟩⟩) exact81962RawTerms (.finite 42) 81961 .exactZero (none)

def event81963 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9925⟩⟩) 0 ⟨5536⟩ 81959

def event81964 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨9925⟩⟩) (.authority (.programFamilyFact))

def exact81965RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨9925⟩⟩], []⟩, (1)⟩]

theorem exact81965RawTermsValid :
    exact81965RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event81965 : Event := .resultExact (⟨.program ⟨214⟩, ⟨9925⟩⟩) exact81965RawTerms (.finite 42) 81964 .exactZero (none)

def event81966 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12567⟩⟩) 0 ⟨9925⟩ 81965

def event81967 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12567⟩⟩) 1 ⟨12566⟩ 81962

def event81968 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12567⟩⟩) (.product (.predecessor 0 81966 .coefficient) (.predecessor 1 81967 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event81969 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12567⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨9925⟩⟩, ⟨.program ⟨214⟩, ⟨12566⟩⟩], []⟩) [⟨.result 81965 .coefficient, true, some 1⟩, ⟨.result 81962 .coefficient, true, some 1⟩])

def event81970 : Event := .survivorFold (1) 81969

def exact81971RawTerms : List Term := []

theorem exact81971RawTermsValid :
    exact81971RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event81971 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12567⟩⟩) exact81971RawTerms (.finite 1764) 81968 (.finite 1764) (some (81969))

def event81972 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12568⟩⟩) 0 ⟨12567⟩ 81971

def event81973 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12568⟩⟩) (.identity (.predecessor 0 81972 .coefficient))

def event81974 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨12568⟩⟩) (.finite 1764)

def event81975 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19960⟩⟩) 0 ⟨12568⟩ 81974

def event81976 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19960⟩⟩) (.authority (.relationPreimageSource ⟨21⟩))

def exact81977RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨19960⟩⟩]⟩, (1)⟩]

theorem exact81977RawTermsValid :
    exact81977RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event81977 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19960⟩⟩) exact81977RawTerms (.finite 136065468) 81976 .exactZero (none)

def event81978 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6⟩⟩) (.authority (.operator))

def exact81979RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩]⟩, (1)⟩]

theorem exact81979RawTermsValid :
    exact81979RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event81979 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6⟩⟩) exact81979RawTerms .large 81978 .exactZero (none)

def event81980 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19961⟩⟩) 0 ⟨6⟩ 81979

def event81981 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19961⟩⟩) 1 ⟨19960⟩ 81977

def event81982 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19961⟩⟩) (.product (.predecessor 0 81980 .coefficient) (.predecessor 1 81981 .coefficient) (⟨false, false, none, none, none⟩))

def event81983 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19961⟩⟩, .operator (⟨81979, 0⟩, ⟨81977, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19960⟩⟩]⟩, (1)⟩)

def exact81984RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19960⟩⟩]⟩, (1)⟩]

theorem exact81984RawTermsValid :
    exact81984RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event81984 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19961⟩⟩) exact81984RawTerms .large 81982 .exactZero (none)

def event81985 : Event := .preFoldPolynomial 81984 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19960⟩⟩]⟩, (1)⟩] .exactZero none

def exact81986RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19960⟩⟩]⟩, (1)⟩]

def event81986 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨19961⟩⟩) 81985 exact81986RawTerms .large 81982 .exactZero (none)

def event81987 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨25454⟩⟩)

def event81988 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event81989 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event81990 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.authority (.operator))

def event81991 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.finite 1)

def event81992 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event81993 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event81994 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event81995 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event81996 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 81995

def event81997 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 81993

def event81998 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 81996 .coefficient) (.value (.predecessor 1 81997 .coefficient)))

def event81999 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event82000 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5505⟩⟩) 0 ⟨5503⟩ 81999

def event82001 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5505⟩⟩) 1 ⟨2348⟩ 81991

def event82002 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5505⟩⟩) (.sum [.predecessor 0 82000 .coefficient, .predecessor 1 82001 .coefficient])

def event82003 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5505⟩⟩) (.finite 218)

def event82004 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5536⟩⟩) 0 ⟨5505⟩ 82003

def event82005 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5536⟩⟩) 1 ⟨961⟩ 81989

def event82006 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5536⟩⟩) (.identity (.predecessor 1 82005 .coefficient))

def event82007 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5536⟩⟩) (.finite 224)

def event82008 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12566⟩⟩) 0 ⟨5536⟩ 82007

def event82009 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12566⟩⟩) (.authority (.programFamilyFact))

def exact82010RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨12566⟩⟩], []⟩, (1)⟩]

theorem exact82010RawTermsValid :
    exact82010RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82010 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12566⟩⟩) exact82010RawTerms (.finite 42) 82009 .exactZero (none)

def event82011 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9925⟩⟩) 0 ⟨5536⟩ 82007

def event82012 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨9925⟩⟩) (.authority (.programFamilyFact))

def exact82013RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨9925⟩⟩], []⟩, (1)⟩]

theorem exact82013RawTermsValid :
    exact82013RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82013 : Event := .resultExact (⟨.program ⟨214⟩, ⟨9925⟩⟩) exact82013RawTerms (.finite 42) 82012 .exactZero (none)

def event82014 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12567⟩⟩) 0 ⟨9925⟩ 82013

def event82015 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12567⟩⟩) 1 ⟨12566⟩ 82010

def event82016 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12567⟩⟩) (.product (.predecessor 0 82014 .coefficient) (.predecessor 1 82015 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event82017 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨12567⟩⟩, .operator (⟨82013, 0⟩, ⟨82010, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨9925⟩⟩, ⟨.program ⟨214⟩, ⟨12566⟩⟩], []⟩, (1)⟩)

def exact82018RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨9925⟩⟩, ⟨.program ⟨214⟩, ⟨12566⟩⟩], []⟩, (1)⟩]

theorem exact82018RawTermsValid :
    exact82018RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82018 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12567⟩⟩) exact82018RawTerms (.finite 1764) 82016 .exactZero (none)

def event82019 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12568⟩⟩) 0 ⟨12567⟩ 82018

def event82020 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12568⟩⟩) (.identity (.predecessor 0 82019 .coefficient))

def event82021 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨12568⟩⟩) (.finite 1764)

def event82022 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23247⟩⟩) 0 ⟨12568⟩ 82021

def event82023 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23247⟩⟩) (.authority (.programFamilyFact))

def event82024 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨23247⟩⟩) (.finite 3720)

def event82025 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨6689⟩⟩) .missing

def event82026 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23248⟩⟩) 0 ⟨6689⟩ 82025

def event82027 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23248⟩⟩) 1 ⟨23247⟩ 82024

def event82028 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23248⟩⟩) (.authority (.operator))

def exact82029RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨23248⟩⟩]⟩, (1)⟩]

theorem exact82029RawTermsValid :
    exact82029RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82029 : Event := .resultExact (⟨.program ⟨214⟩, ⟨23248⟩⟩) exact82029RawTerms .large 82028 .exactZero (none)

def event82030 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25450⟩⟩) 0 ⟨23248⟩ 82029

def event82031 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25450⟩⟩) (.authority (.operator))

def exact82032RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨25450⟩⟩]⟩, (1)⟩]

theorem exact82032RawTermsValid :
    exact82032RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82032 : Event := .resultExact (⟨.program ⟨214⟩, ⟨25450⟩⟩) exact82032RawTerms (.finite 8192) 82031 .exactZero (none)

def event82033 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨110⟩⟩) (.authority (.operator))

def event82034 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨110⟩⟩) .exactZero

def event82035 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12662⟩⟩) 0 ⟨12568⟩ 82021

def event82036 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12662⟩⟩) 1 ⟨110⟩ 82034

def event82037 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12662⟩⟩) (.sum [.predecessor 0 82035 .coefficient, .predecessor 1 82036 .coefficient])

def event82038 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨12662⟩⟩) (.finite 1764)

def event82039 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12663⟩⟩) 0 ⟨12662⟩ 82038

def event82040 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12663⟩⟩) (.identity (.predecessor 0 82039 .coefficient))

def exact82041RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨9925⟩⟩, ⟨.program ⟨214⟩, ⟨12566⟩⟩], []⟩, (1)⟩]

theorem exact82041RawTermsValid :
    exact82041RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82041 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12663⟩⟩) exact82041RawTerms (.finite 1764) 82040 .exactZero (none)

def event82042 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6544⟩⟩) (.authority (.factStore))

def exact82043RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact82043RawTermsValid :
    exact82043RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82043 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6544⟩⟩) exact82043RawTerms .large 82042 .exactZero (none)

def event82044 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12664⟩⟩) 0 ⟨6544⟩ 82043

def event82045 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12664⟩⟩) 1 ⟨12663⟩ 82041

def event82046 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12664⟩⟩) (.product (.predecessor 0 82044 .coefficient) (.predecessor 1 82045 .coefficient) (⟨false, false, none, none, none⟩))

def event82047 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨12664⟩⟩, .operator (⟨82043, 0⟩, ⟨82041, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨9925⟩⟩, ⟨.program ⟨214⟩, ⟨12566⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact82048RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨9925⟩⟩, ⟨.program ⟨214⟩, ⟨12566⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact82048RawTermsValid :
    exact82048RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82048 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12664⟩⟩) exact82048RawTerms .large 82046 .exactZero (none)

def event82049 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6757⟩⟩) 0 ⟨6689⟩ 82025

def event82050 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6757⟩⟩) (.authority (.operator))

def exact82051RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6757⟩⟩]⟩, (1)⟩]

theorem exact82051RawTermsValid :
    exact82051RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82051 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6757⟩⟩) exact82051RawTerms .large 82050 .exactZero (none)

def event82052 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6786⟩⟩) 0 ⟨6757⟩ 82051

def event82053 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6786⟩⟩) (.identity (.predecessor 0 82052 .coefficient))

def exact82054RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6786⟩⟩]⟩, (1)⟩]

theorem exact82054RawTermsValid :
    exact82054RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82054 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6786⟩⟩) exact82054RawTerms .large 82053 .exactZero (none)

def event82055 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7870⟩⟩) 0 ⟨6786⟩ 82054

def event82056 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7870⟩⟩) (.authority (.operator))

def exact82057RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨7870⟩⟩]⟩, (1)⟩]

theorem exact82057RawTermsValid :
    exact82057RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82057 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7870⟩⟩) exact82057RawTerms (.finite 8192) 82056 .exactZero (none)

def event82058 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7871⟩⟩) 0 ⟨7870⟩ 82057

def event82059 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7871⟩⟩) 1 ⟨2348⟩ 81991

def event82060 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7871⟩⟩) (.scale (.predecessor 0 82058 .coefficient) (.value (.predecessor 1 82059 .coefficient)))

def exact82061RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨7870⟩⟩]⟩, (1)⟩]

theorem exact82061RawTermsValid :
    exact82061RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82061 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7871⟩⟩) exact82061RawTerms (.finite 8192) 82060 .exactZero (none)

def event82062 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6766⟩⟩) 0 ⟨6757⟩ 82051

def event82063 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6766⟩⟩) (.identity (.predecessor 0 82062 .coefficient))

def exact82064RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6766⟩⟩]⟩, (1)⟩]

theorem exact82064RawTermsValid :
    exact82064RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82064 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6766⟩⟩) exact82064RawTerms .large 82063 .exactZero (none)

def event82065 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7872⟩⟩) 0 ⟨6766⟩ 82064

def event82066 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7872⟩⟩) 1 ⟨7871⟩ 82061

def event82067 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7872⟩⟩) (.product (.predecessor 0 82065 .coefficient) (.predecessor 1 82066 .coefficient) (⟨false, false, none, none, none⟩))

def event82068 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7872⟩⟩, .operator (⟨82064, 0⟩, ⟨82061, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6766⟩⟩, ⟨.program ⟨214⟩, ⟨7870⟩⟩]⟩, (1)⟩)

def exact82069RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6766⟩⟩, ⟨.program ⟨214⟩, ⟨7870⟩⟩]⟩, (1)⟩]

theorem exact82069RawTermsValid :
    exact82069RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82069 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7872⟩⟩) exact82069RawTerms .large 82067 .exactZero (none)

def event82070 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12665⟩⟩) 0 ⟨7872⟩ 82069

def event82071 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12665⟩⟩) 1 ⟨12664⟩ 82048

def event82072 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12665⟩⟩) (.sum [.predecessor 0 82070 .coefficient, .predecessor 1 82071 .coefficient])

def exact82073RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6766⟩⟩, ⟨.program ⟨214⟩, ⟨7870⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨9925⟩⟩, ⟨.program ⟨214⟩, ⟨12566⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact82073RawTermsValid :
    exact82073RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82073 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12665⟩⟩) exact82073RawTerms .large 82072 .exactZero (none)

def event82074 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25453⟩⟩) 0 ⟨12665⟩ 82073

def event82075 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25453⟩⟩) 1 ⟨25450⟩ 82032

def event82076 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25453⟩⟩) (.product (.predecessor 0 82074 .coefficient) (.predecessor 1 82075 .coefficient) (⟨false, false, none, none, none⟩))

def event82077 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25453⟩⟩, .operator (⟨82073, 0⟩, ⟨82032, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6766⟩⟩, ⟨.program ⟨214⟩, ⟨7870⟩⟩, ⟨.program ⟨214⟩, ⟨25450⟩⟩]⟩, (1)⟩)

def event82078 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25453⟩⟩, .operator (⟨82073, 1⟩, ⟨82032, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨9925⟩⟩, ⟨.program ⟨214⟩, ⟨12566⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25450⟩⟩]⟩, (-1)⟩)

def event82079 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨25453⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨9925⟩⟩, ⟨.program ⟨214⟩, ⟨12566⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25450⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨25450⟩⟩) ⟨23248⟩ 82029)

def event82080 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25453⟩⟩, .relation 82079 0, ⟨[⟨.program ⟨214⟩, ⟨9925⟩⟩, ⟨.program ⟨214⟩, ⟨12566⟩⟩], [⟨.program ⟨214⟩, ⟨23248⟩⟩]⟩, (-1)⟩)

def exact82081RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6766⟩⟩, ⟨.program ⟨214⟩, ⟨7870⟩⟩, ⟨.program ⟨214⟩, ⟨25450⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨9925⟩⟩, ⟨.program ⟨214⟩, ⟨12566⟩⟩], [⟨.program ⟨214⟩, ⟨23248⟩⟩]⟩, (-1)⟩]

theorem exact82081RawTermsValid :
    exact82081RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82081 : Event := .resultExact (⟨.program ⟨214⟩, ⟨25453⟩⟩) exact82081RawTerms .large 82076 .exactZero (none)

def event82082 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16549⟩⟩) 0 ⟨12568⟩ 82021

def event82083 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16549⟩⟩) (.authority (.programFamilyFact))

def exact82084RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16549⟩⟩], []⟩, (1)⟩]

theorem exact82084RawTermsValid :
    exact82084RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82084 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16549⟩⟩) exact82084RawTerms (.finite 42) 82083 .exactZero (none)

def event82085 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16551⟩⟩) 0 ⟨6544⟩ 82043

def event82086 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16551⟩⟩) 1 ⟨16549⟩ 82084

def event82087 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16551⟩⟩) (.product (.predecessor 0 82085 .coefficient) (.predecessor 1 82086 .coefficient) (⟨false, true, none, none, some 1⟩))

def event82088 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨16551⟩⟩, .operator (⟨82043, 0⟩, ⟨82084, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨16549⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact82089RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16549⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact82089RawTermsValid :
    exact82089RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82089 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16551⟩⟩) exact82089RawTerms .large 82087 .exactZero (none)

def event82090 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6703⟩⟩) 0 ⟨6689⟩ 82025

def event82091 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6703⟩⟩) (.authority (.operator))

def exact82092RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6703⟩⟩]⟩, (1)⟩]

theorem exact82092RawTermsValid :
    exact82092RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82092 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6703⟩⟩) exact82092RawTerms .large 82091 .exactZero (none)

def event82093 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16552⟩⟩) 0 ⟨6703⟩ 82092

def event82094 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16552⟩⟩) 1 ⟨16551⟩ 82089

def event82095 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16552⟩⟩) (.sum [.predecessor 0 82093 .coefficient, .predecessor 1 82094 .coefficient])

def exact82096RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6703⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16549⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact82096RawTermsValid :
    exact82096RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82096 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16552⟩⟩) exact82096RawTerms .large 82095 .exactZero (none)

def event82097 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25454⟩⟩) 0 ⟨16552⟩ 82096

def event82098 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25454⟩⟩) 1 ⟨25453⟩ 82081

def event82099 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25454⟩⟩) (.sum [.predecessor 0 82097 .coefficient, .predecessor 1 82098 .coefficient])

def exact82100RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6703⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6766⟩⟩, ⟨.program ⟨214⟩, ⟨7870⟩⟩, ⟨.program ⟨214⟩, ⟨25450⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨9925⟩⟩, ⟨.program ⟨214⟩, ⟨12566⟩⟩], [⟨.program ⟨214⟩, ⟨23248⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16549⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact82100RawTermsValid :
    exact82100RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82100 : Event := .resultExact (⟨.program ⟨214⟩, ⟨25454⟩⟩) exact82100RawTerms .large 82099 .exactZero (none)

def event82101 : Event := .preFoldPolynomial 82100 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6703⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6766⟩⟩, ⟨.program ⟨214⟩, ⟨7870⟩⟩, ⟨.program ⟨214⟩, ⟨25450⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨9925⟩⟩, ⟨.program ⟨214⟩, ⟨12566⟩⟩], [⟨.program ⟨214⟩, ⟨23248⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16549⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩] .exactZero none

def exact82102RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6703⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6766⟩⟩, ⟨.program ⟨214⟩, ⟨7870⟩⟩, ⟨.program ⟨214⟩, ⟨25450⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨9925⟩⟩, ⟨.program ⟨214⟩, ⟨12566⟩⟩], [⟨.program ⟨214⟩, ⟨23248⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16549⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

def event82102 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨25454⟩⟩) 82101 exact82102RawTerms .large 82099 .exactZero (none)

def event82103 : Event := .specializationComputed (⟨.program ⟨214⟩, ⟨12568⟩⟩) ⟨⟨116⟩, ⟨21⟩, ⟨109⟩⟩ ⟨81939, 82103⟩

def event82104 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨19963⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19960⟩⟩]⟩) (1) 0 2 (.universal 82103 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19960⟩⟩]⟩) (none) 82102)

def event82105 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19963⟩⟩, .relation 82104 0, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6703⟩⟩]⟩, (1)⟩)

def event82106 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19963⟩⟩, .relation 82104 1, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6766⟩⟩, ⟨.program ⟨214⟩, ⟨7870⟩⟩, ⟨.program ⟨214⟩, ⟨25450⟩⟩]⟩, (-1)⟩)

def event82107 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19963⟩⟩, .relation 82104 2, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨9925⟩⟩, ⟨.program ⟨214⟩, ⟨12566⟩⟩], [⟨.program ⟨214⟩, ⟨23248⟩⟩]⟩, (1)⟩)

def event82108 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19963⟩⟩, .relation 82104 3, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16549⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact82109RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6703⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6766⟩⟩, ⟨.program ⟨214⟩, ⟨7870⟩⟩, ⟨.program ⟨214⟩, ⟨25450⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨9925⟩⟩, ⟨.program ⟨214⟩, ⟨12566⟩⟩], [⟨.program ⟨214⟩, ⟨23248⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16549⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact82109RawTermsValid :
    exact82109RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82109 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19963⟩⟩) exact82109RawTerms .large 81935 (.finite 1811303510016) (some (81937))

def event82110 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25452⟩⟩) 0 ⟨19963⟩ 82109

def event82111 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25452⟩⟩) 1 ⟨25451⟩ 81925

def event82112 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25452⟩⟩) (.sum [.predecessor 0 82110 .coefficient, .predecessor 1 82111 .coefficient])

def event82113 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25452⟩⟩, .operator (⟨82109, 2⟩, ⟨81925, 1⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨9925⟩⟩, ⟨.program ⟨214⟩, ⟨12566⟩⟩], [⟨.program ⟨214⟩, ⟨23248⟩⟩]⟩, (-1)⟩)

def event82114 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25452⟩⟩, .operator (⟨82109, 1⟩, ⟨81925, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6766⟩⟩, ⟨.program ⟨214⟩, ⟨7870⟩⟩, ⟨.program ⟨214⟩, ⟨25450⟩⟩]⟩, (1)⟩)

def event82115 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25452⟩⟩) (.sum [.result 82109 .summary, .result 81925 .summary])

def exact82116RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6703⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16549⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact82116RawTermsValid :
    exact82116RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82116 : Event := .resultExact (⟨.program ⟨214⟩, ⟨25452⟩⟩) exact82116RawTerms .large 82112 (.finite 352134001995776) (some (82115))

def event82117 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29170⟩⟩) 0 ⟨25452⟩ 82116

def event82118 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29170⟩⟩) 1 ⟨29168⟩ 81841

def event82119 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29170⟩⟩) (.product (.predecessor 0 82117 .coefficient) (.predecessor 1 82118 .coefficient) (⟨false, false, none, none, none⟩))

def event82120 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29170⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨29168⟩⟩]⟩) [⟨.result 81841 .coefficient, false, none⟩])

def event82121 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29170⟩⟩) (.product (.result 82116 .summary) (.transfer 82120) (⟨false, false, none, none, none⟩))

def event82122 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨29170⟩⟩, .operator (⟨82116, 0⟩, ⟨81841, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6703⟩⟩, ⟨.program ⟨214⟩, ⟨29168⟩⟩]⟩, (1)⟩)

def event82123 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨29170⟩⟩, .operator (⟨82116, 1⟩, ⟨81841, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16549⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29168⟩⟩]⟩, (-1)⟩)

def event82124 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨29170⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16549⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29168⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨29168⟩⟩) ⟨24540⟩ 81838)

def event82125 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨29170⟩⟩, .relation 82124 0, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16549⟩⟩], [⟨.program ⟨214⟩, ⟨24540⟩⟩]⟩, (-1)⟩)

def exact82126RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6703⟩⟩, ⟨.program ⟨214⟩, ⟨29168⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16549⟩⟩], [⟨.program ⟨214⟩, ⟨24540⟩⟩]⟩, (-1)⟩]

theorem exact82126RawTermsValid :
    exact82126RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82126 : Event := .resultExact (⟨.program ⟨214⟩, ⟨29170⟩⟩) exact82126RawTerms .large 82119 (.finite 1292337421468529852416) (some (82121))

def event82127 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22264⟩⟩) 0 ⟨16550⟩ 3937

def event82128 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22264⟩⟩) (.authority (.relationPreimageSource ⟨57⟩))

def exact82129RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨22264⟩⟩]⟩, (1)⟩]

theorem exact82129RawTermsValid :
    exact82129RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82129 : Event := .resultExact (⟨.program ⟨214⟩, ⟨22264⟩⟩) exact82129RawTerms (.finite 136065468) 82128 .exactZero (none)

def event82130 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22266⟩⟩) 0 ⟨22264⟩ 82129

def event82131 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22266⟩⟩) 1 ⟨2348⟩ 4

def event82132 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22266⟩⟩) (.scale (.predecessor 0 82130 .coefficient) (.value (.predecessor 1 82131 .coefficient)))

def exact82133RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨22264⟩⟩]⟩, (1)⟩]

theorem exact82133RawTermsValid :
    exact82133RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82133 : Event := .resultExact (⟨.program ⟨214⟩, ⟨22266⟩⟩) exact82133RawTerms (.finite 136065468) 82132 .exactZero (none)

def event82134 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22267⟩⟩) 0 ⟨5541⟩ 80012

def event82135 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22267⟩⟩) 1 ⟨22266⟩ 82133

def event82136 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22267⟩⟩) (.product (.predecessor 0 82134 .coefficient) (.predecessor 1 82135 .coefficient) (⟨false, false, none, none, none⟩))

def event82137 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22267⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨22264⟩⟩]⟩) [⟨.result 82129 .coefficient, false, none⟩])

def event82138 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22267⟩⟩) (.product (.result 80012 .summary) (.transfer 82137) (⟨false, false, none, none, none⟩))

def event82139 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨22267⟩⟩, .operator (⟨80012, 0⟩, ⟨82133, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22264⟩⟩]⟩, (1)⟩)

def event82140 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨22265⟩⟩)

def event82141 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event82142 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event82143 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.authority (.operator))

def event82144 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.finite 1)

def event82145 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event82146 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event82147 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event82148 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event82149 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 82148

def event82150 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 82146

def event82151 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 82149 .coefficient) (.value (.predecessor 1 82150 .coefficient)))

def event82152 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event82153 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5505⟩⟩) 0 ⟨5503⟩ 82152

def event82154 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5505⟩⟩) 1 ⟨2348⟩ 82144

def event82155 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5505⟩⟩) (.sum [.predecessor 0 82153 .coefficient, .predecessor 1 82154 .coefficient])

def event82156 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5505⟩⟩) (.finite 218)

def event82157 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5536⟩⟩) 0 ⟨5505⟩ 82156

def event82158 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5536⟩⟩) 1 ⟨961⟩ 82142

def event82159 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5536⟩⟩) (.identity (.predecessor 1 82158 .coefficient))

def event82160 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5536⟩⟩) (.finite 224)

def event82161 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12566⟩⟩) 0 ⟨5536⟩ 82160

def event82162 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12566⟩⟩) (.authority (.programFamilyFact))

def exact82163RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨12566⟩⟩], []⟩, (1)⟩]

theorem exact82163RawTermsValid :
    exact82163RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82163 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12566⟩⟩) exact82163RawTerms (.finite 42) 82162 .exactZero (none)

def event82164 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9925⟩⟩) 0 ⟨5536⟩ 82160

def event82165 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨9925⟩⟩) (.authority (.programFamilyFact))

def exact82166RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨9925⟩⟩], []⟩, (1)⟩]

theorem exact82166RawTermsValid :
    exact82166RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82166 : Event := .resultExact (⟨.program ⟨214⟩, ⟨9925⟩⟩) exact82166RawTerms (.finite 42) 82165 .exactZero (none)

def event82167 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12567⟩⟩) 0 ⟨9925⟩ 82166

def event82168 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12567⟩⟩) 1 ⟨12566⟩ 82163

def event82169 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12567⟩⟩) (.product (.predecessor 0 82167 .coefficient) (.predecessor 1 82168 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event82170 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12567⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨9925⟩⟩, ⟨.program ⟨214⟩, ⟨12566⟩⟩], []⟩) [⟨.result 82166 .coefficient, true, some 1⟩, ⟨.result 82163 .coefficient, true, some 1⟩])

def event82171 : Event := .survivorFold (1) 82170

def exact82172RawTerms : List Term := []

theorem exact82172RawTermsValid :
    exact82172RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82172 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12567⟩⟩) exact82172RawTerms (.finite 1764) 82169 (.finite 1764) (some (82170))

def event82173 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12568⟩⟩) 0 ⟨12567⟩ 82172

def event82174 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12568⟩⟩) (.identity (.predecessor 0 82173 .coefficient))

def event82175 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨12568⟩⟩) (.finite 1764)

def eventLeaf5120 : Array AnnotatedEvent := #[
  { event := event81920
    frameStart := 0 },
  { event := event81921
    frameStart := 0 },
  { event := event81922
    frameStart := 0 },
  { event := event81923
    frameStart := 0 },
  { event := event81924
    frameStart := 0 },
  { event := event81925
    frameStart := 0 },
  { event := event81926
    frameStart := 0 },
  { event := event81927
    frameStart := 0 },
  { event := event81928
    frameStart := 0 },
  { event := event81929
    frameStart := 0 },
  { event := event81930
    frameStart := 0 },
  { event := event81931
    frameStart := 0 },
  { event := event81932
    frameStart := 0 },
  { event := event81933
    frameStart := 0 },
  { event := event81934
    frameStart := 0 },
  { event := event81935
    frameStart := 0 }
]

def eventLeaf5121 : Array AnnotatedEvent := #[
  { event := event81936
    frameStart := 0 },
  { event := event81937
    frameStart := 0 },
  { event := event81938
    frameStart := 0 },
  { event := event81939
    frameStart := 81939 },
  { event := event81940
    frameStart := 81939 },
  { event := event81941
    frameStart := 81939 },
  { event := event81942
    frameStart := 81939 },
  { event := event81943
    frameStart := 81939 },
  { event := event81944
    frameStart := 81939 },
  { event := event81945
    frameStart := 81939 },
  { event := event81946
    frameStart := 81939 },
  { event := event81947
    frameStart := 81939 },
  { event := event81948
    frameStart := 81939 },
  { event := event81949
    frameStart := 81939 },
  { event := event81950
    frameStart := 81939 },
  { event := event81951
    frameStart := 81939 }
]

def eventLeaf5122 : Array AnnotatedEvent := #[
  { event := event81952
    frameStart := 81939 },
  { event := event81953
    frameStart := 81939 },
  { event := event81954
    frameStart := 81939 },
  { event := event81955
    frameStart := 81939 },
  { event := event81956
    frameStart := 81939 },
  { event := event81957
    frameStart := 81939 },
  { event := event81958
    frameStart := 81939 },
  { event := event81959
    frameStart := 81939 },
  { event := event81960
    frameStart := 81939 },
  { event := event81961
    frameStart := 81939 },
  { event := event81962
    frameStart := 81939 },
  { event := event81963
    frameStart := 81939 },
  { event := event81964
    frameStart := 81939 },
  { event := event81965
    frameStart := 81939 },
  { event := event81966
    frameStart := 81939 },
  { event := event81967
    frameStart := 81939 }
]

def eventLeaf5123 : Array AnnotatedEvent := #[
  { event := event81968
    frameStart := 81939 },
  { event := event81969
    frameStart := 81939 },
  { event := event81970
    frameStart := 81939 },
  { event := event81971
    frameStart := 81939 },
  { event := event81972
    frameStart := 81939 },
  { event := event81973
    frameStart := 81939 },
  { event := event81974
    frameStart := 81939 },
  { event := event81975
    frameStart := 81939 },
  { event := event81976
    frameStart := 81939 },
  { event := event81977
    frameStart := 81939 },
  { event := event81978
    frameStart := 81939 },
  { event := event81979
    frameStart := 81939 },
  { event := event81980
    frameStart := 81939 },
  { event := event81981
    frameStart := 81939 },
  { event := event81982
    frameStart := 81939 },
  { event := event81983
    frameStart := 81939 }
]

def eventLeaf5124 : Array AnnotatedEvent := #[
  { event := event81984
    frameStart := 81939 },
  { event := event81985
    frameStart := 81939 },
  { event := event81986
    frameStart := 81939 },
  { event := event81987
    frameStart := 81987 },
  { event := event81988
    frameStart := 81987 },
  { event := event81989
    frameStart := 81987 },
  { event := event81990
    frameStart := 81987 },
  { event := event81991
    frameStart := 81987 },
  { event := event81992
    frameStart := 81987 },
  { event := event81993
    frameStart := 81987 },
  { event := event81994
    frameStart := 81987 },
  { event := event81995
    frameStart := 81987 },
  { event := event81996
    frameStart := 81987 },
  { event := event81997
    frameStart := 81987 },
  { event := event81998
    frameStart := 81987 },
  { event := event81999
    frameStart := 81987 }
]

def eventLeaf5125 : Array AnnotatedEvent := #[
  { event := event82000
    frameStart := 81987 },
  { event := event82001
    frameStart := 81987 },
  { event := event82002
    frameStart := 81987 },
  { event := event82003
    frameStart := 81987 },
  { event := event82004
    frameStart := 81987 },
  { event := event82005
    frameStart := 81987 },
  { event := event82006
    frameStart := 81987 },
  { event := event82007
    frameStart := 81987 },
  { event := event82008
    frameStart := 81987 },
  { event := event82009
    frameStart := 81987 },
  { event := event82010
    frameStart := 81987 },
  { event := event82011
    frameStart := 81987 },
  { event := event82012
    frameStart := 81987 },
  { event := event82013
    frameStart := 81987 },
  { event := event82014
    frameStart := 81987 },
  { event := event82015
    frameStart := 81987 }
]

def eventLeaf5126 : Array AnnotatedEvent := #[
  { event := event82016
    frameStart := 81987 },
  { event := event82017
    frameStart := 81987 },
  { event := event82018
    frameStart := 81987 },
  { event := event82019
    frameStart := 81987 },
  { event := event82020
    frameStart := 81987 },
  { event := event82021
    frameStart := 81987 },
  { event := event82022
    frameStart := 81987 },
  { event := event82023
    frameStart := 81987 },
  { event := event82024
    frameStart := 81987 },
  { event := event82025
    frameStart := 81987 },
  { event := event82026
    frameStart := 81987 },
  { event := event82027
    frameStart := 81987 },
  { event := event82028
    frameStart := 81987 },
  { event := event82029
    frameStart := 81987 },
  { event := event82030
    frameStart := 81987 },
  { event := event82031
    frameStart := 81987 }
]

def eventLeaf5127 : Array AnnotatedEvent := #[
  { event := event82032
    frameStart := 81987 },
  { event := event82033
    frameStart := 81987 },
  { event := event82034
    frameStart := 81987 },
  { event := event82035
    frameStart := 81987 },
  { event := event82036
    frameStart := 81987 },
  { event := event82037
    frameStart := 81987 },
  { event := event82038
    frameStart := 81987 },
  { event := event82039
    frameStart := 81987 },
  { event := event82040
    frameStart := 81987 },
  { event := event82041
    frameStart := 81987 },
  { event := event82042
    frameStart := 81987 },
  { event := event82043
    frameStart := 81987 },
  { event := event82044
    frameStart := 81987 },
  { event := event82045
    frameStart := 81987 },
  { event := event82046
    frameStart := 81987 },
  { event := event82047
    frameStart := 81987 }
]

def eventLeaf5128 : Array AnnotatedEvent := #[
  { event := event82048
    frameStart := 81987 },
  { event := event82049
    frameStart := 81987 },
  { event := event82050
    frameStart := 81987 },
  { event := event82051
    frameStart := 81987 },
  { event := event82052
    frameStart := 81987 },
  { event := event82053
    frameStart := 81987 },
  { event := event82054
    frameStart := 81987 },
  { event := event82055
    frameStart := 81987 },
  { event := event82056
    frameStart := 81987 },
  { event := event82057
    frameStart := 81987 },
  { event := event82058
    frameStart := 81987 },
  { event := event82059
    frameStart := 81987 },
  { event := event82060
    frameStart := 81987 },
  { event := event82061
    frameStart := 81987 },
  { event := event82062
    frameStart := 81987 },
  { event := event82063
    frameStart := 81987 }
]

def eventLeaf5129 : Array AnnotatedEvent := #[
  { event := event82064
    frameStart := 81987 },
  { event := event82065
    frameStart := 81987 },
  { event := event82066
    frameStart := 81987 },
  { event := event82067
    frameStart := 81987 },
  { event := event82068
    frameStart := 81987 },
  { event := event82069
    frameStart := 81987 },
  { event := event82070
    frameStart := 81987 },
  { event := event82071
    frameStart := 81987 },
  { event := event82072
    frameStart := 81987 },
  { event := event82073
    frameStart := 81987 },
  { event := event82074
    frameStart := 81987 },
  { event := event82075
    frameStart := 81987 },
  { event := event82076
    frameStart := 81987 },
  { event := event82077
    frameStart := 81987 },
  { event := event82078
    frameStart := 81987 },
  { event := event82079
    frameStart := 81987 }
]

def eventLeaf5130 : Array AnnotatedEvent := #[
  { event := event82080
    frameStart := 81987 },
  { event := event82081
    frameStart := 81987 },
  { event := event82082
    frameStart := 81987 },
  { event := event82083
    frameStart := 81987 },
  { event := event82084
    frameStart := 81987 },
  { event := event82085
    frameStart := 81987 },
  { event := event82086
    frameStart := 81987 },
  { event := event82087
    frameStart := 81987 },
  { event := event82088
    frameStart := 81987 },
  { event := event82089
    frameStart := 81987 },
  { event := event82090
    frameStart := 81987 },
  { event := event82091
    frameStart := 81987 },
  { event := event82092
    frameStart := 81987 },
  { event := event82093
    frameStart := 81987 },
  { event := event82094
    frameStart := 81987 },
  { event := event82095
    frameStart := 81987 }
]

def eventLeaf5131 : Array AnnotatedEvent := #[
  { event := event82096
    frameStart := 81987 },
  { event := event82097
    frameStart := 81987 },
  { event := event82098
    frameStart := 81987 },
  { event := event82099
    frameStart := 81987 },
  { event := event82100
    frameStart := 81987 },
  { event := event82101
    frameStart := 81987 },
  { event := event82102
    frameStart := 81987 },
  { event := event82103
    frameStart := 0 },
  { event := event82104
    frameStart := 0 },
  { event := event82105
    frameStart := 0 },
  { event := event82106
    frameStart := 0 },
  { event := event82107
    frameStart := 0 },
  { event := event82108
    frameStart := 0 },
  { event := event82109
    frameStart := 0 },
  { event := event82110
    frameStart := 0 },
  { event := event82111
    frameStart := 0 }
]

def eventLeaf5132 : Array AnnotatedEvent := #[
  { event := event82112
    frameStart := 0 },
  { event := event82113
    frameStart := 0 },
  { event := event82114
    frameStart := 0 },
  { event := event82115
    frameStart := 0 },
  { event := event82116
    frameStart := 0 },
  { event := event82117
    frameStart := 0 },
  { event := event82118
    frameStart := 0 },
  { event := event82119
    frameStart := 0 },
  { event := event82120
    frameStart := 0 },
  { event := event82121
    frameStart := 0 },
  { event := event82122
    frameStart := 0 },
  { event := event82123
    frameStart := 0 },
  { event := event82124
    frameStart := 0 },
  { event := event82125
    frameStart := 0 },
  { event := event82126
    frameStart := 0 },
  { event := event82127
    frameStart := 0 }
]

def eventLeaf5133 : Array AnnotatedEvent := #[
  { event := event82128
    frameStart := 0 },
  { event := event82129
    frameStart := 0 },
  { event := event82130
    frameStart := 0 },
  { event := event82131
    frameStart := 0 },
  { event := event82132
    frameStart := 0 },
  { event := event82133
    frameStart := 0 },
  { event := event82134
    frameStart := 0 },
  { event := event82135
    frameStart := 0 },
  { event := event82136
    frameStart := 0 },
  { event := event82137
    frameStart := 0 },
  { event := event82138
    frameStart := 0 },
  { event := event82139
    frameStart := 0 },
  { event := event82140
    frameStart := 82140 },
  { event := event82141
    frameStart := 82140 },
  { event := event82142
    frameStart := 82140 },
  { event := event82143
    frameStart := 82140 }
]

def eventLeaf5134 : Array AnnotatedEvent := #[
  { event := event82144
    frameStart := 82140 },
  { event := event82145
    frameStart := 82140 },
  { event := event82146
    frameStart := 82140 },
  { event := event82147
    frameStart := 82140 },
  { event := event82148
    frameStart := 82140 },
  { event := event82149
    frameStart := 82140 },
  { event := event82150
    frameStart := 82140 },
  { event := event82151
    frameStart := 82140 },
  { event := event82152
    frameStart := 82140 },
  { event := event82153
    frameStart := 82140 },
  { event := event82154
    frameStart := 82140 },
  { event := event82155
    frameStart := 82140 },
  { event := event82156
    frameStart := 82140 },
  { event := event82157
    frameStart := 82140 },
  { event := event82158
    frameStart := 82140 },
  { event := event82159
    frameStart := 82140 }
]

def eventLeaf5135 : Array AnnotatedEvent := #[
  { event := event82160
    frameStart := 82140 },
  { event := event82161
    frameStart := 82140 },
  { event := event82162
    frameStart := 82140 },
  { event := event82163
    frameStart := 82140 },
  { event := event82164
    frameStart := 82140 },
  { event := event82165
    frameStart := 82140 },
  { event := event82166
    frameStart := 82140 },
  { event := event82167
    frameStart := 82140 },
  { event := event82168
    frameStart := 82140 },
  { event := event82169
    frameStart := 82140 },
  { event := event82170
    frameStart := 82140 },
  { event := event82171
    frameStart := 82140 },
  { event := event82172
    frameStart := 82140 },
  { event := event82173
    frameStart := 82140 },
  { event := event82174
    frameStart := 82140 },
  { event := event82175
    frameStart := 82140 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events320
