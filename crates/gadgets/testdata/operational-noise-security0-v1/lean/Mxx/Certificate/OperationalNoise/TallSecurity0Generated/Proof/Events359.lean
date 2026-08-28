import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events359

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event91904 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21619⟩⟩) (.product (.predecessor 0 91902 .coefficient) (.predecessor 1 91903 .coefficient) (⟨false, false, none, none, none⟩))

def event91905 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21619⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨21616⟩⟩]⟩) [⟨.result 91897 .coefficient, false, none⟩])

def event91906 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21619⟩⟩) (.product (.result 80012 .summary) (.transfer 91905) (⟨false, false, none, none, none⟩))

def event91907 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21619⟩⟩, .operator (⟨80012, 0⟩, ⟨91901, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21616⟩⟩]⟩, (1)⟩)

def event91908 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨21617⟩⟩)

def event91909 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event91910 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event91911 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.authority (.operator))

def event91912 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.finite 1)

def event91913 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event91914 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event91915 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event91916 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event91917 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 91916

def event91918 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 91914

def event91919 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 91917 .coefficient) (.value (.predecessor 1 91918 .coefficient)))

def event91920 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event91921 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5505⟩⟩) 0 ⟨5503⟩ 91920

def event91922 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5505⟩⟩) 1 ⟨2348⟩ 91912

def event91923 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5505⟩⟩) (.sum [.predecessor 0 91921 .coefficient, .predecessor 1 91922 .coefficient])

def event91924 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5505⟩⟩) (.finite 218)

def event91925 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5536⟩⟩) 0 ⟨5505⟩ 91924

def event91926 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5536⟩⟩) 1 ⟨961⟩ 91910

def event91927 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5536⟩⟩) (.identity (.predecessor 1 91926 .coefficient))

def event91928 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5536⟩⟩) (.finite 224)

def event91929 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11637⟩⟩) 0 ⟨5536⟩ 91928

def event91930 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11637⟩⟩) (.authority (.programFamilyFact))

def exact91931RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11637⟩⟩], []⟩, (1)⟩]

theorem exact91931RawTermsValid :
    exact91931RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event91931 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11637⟩⟩) exact91931RawTerms (.finite 28) 91930 .exactZero (none)

def event91932 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14641⟩⟩) 0 ⟨5536⟩ 91928

def event91933 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14641⟩⟩) (.authority (.programFamilyFact))

def exact91934RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨14641⟩⟩], []⟩, (1)⟩]

theorem exact91934RawTermsValid :
    exact91934RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event91934 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14641⟩⟩) exact91934RawTerms (.finite 28) 91933 .exactZero (none)

def event91935 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14642⟩⟩) 0 ⟨14641⟩ 91934

def event91936 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14642⟩⟩) 1 ⟨11637⟩ 91931

def event91937 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14642⟩⟩) (.product (.predecessor 0 91935 .coefficient) (.predecessor 1 91936 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event91938 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14642⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨11637⟩⟩, ⟨.program ⟨214⟩, ⟨14641⟩⟩], []⟩) [⟨.result 91934 .coefficient, true, some 1⟩, ⟨.result 91931 .coefficient, true, some 1⟩])

def event91939 : Event := .survivorFold (1) 91938

def exact91940RawTerms : List Term := []

theorem exact91940RawTermsValid :
    exact91940RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event91940 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14642⟩⟩) exact91940RawTerms (.finite 784) 91937 (.finite 784) (some (91938))

def event91941 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14643⟩⟩) 0 ⟨14642⟩ 91940

def event91942 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14643⟩⟩) (.identity (.predecessor 0 91941 .coefficient))

def event91943 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨14643⟩⟩) (.finite 784)

def event91944 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16178⟩⟩) 0 ⟨14643⟩ 91943

def event91945 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16178⟩⟩) (.authority (.programFamilyFact))

def exact91946RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16178⟩⟩], []⟩, (1)⟩]

theorem exact91946RawTermsValid :
    exact91946RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event91946 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16178⟩⟩) exact91946RawTerms (.finite 28) 91945 .exactZero (none)

def event91947 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16179⟩⟩) 0 ⟨16178⟩ 91946

def event91948 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16179⟩⟩) (.identity (.predecessor 0 91947 .coefficient))

def event91949 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨16179⟩⟩) (.finite 28)

def event91950 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21616⟩⟩) 0 ⟨16179⟩ 91949

def event91951 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21616⟩⟩) (.authority (.relationPreimageSource ⟨47⟩))

def exact91952RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨21616⟩⟩]⟩, (1)⟩]

theorem exact91952RawTermsValid :
    exact91952RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event91952 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21616⟩⟩) exact91952RawTerms (.finite 136065468) 91951 .exactZero (none)

def event91953 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6⟩⟩) (.authority (.operator))

def exact91954RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩]⟩, (1)⟩]

theorem exact91954RawTermsValid :
    exact91954RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event91954 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6⟩⟩) exact91954RawTerms .large 91953 .exactZero (none)

def event91955 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21617⟩⟩) 0 ⟨6⟩ 91954

def event91956 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21617⟩⟩) 1 ⟨21616⟩ 91952

def event91957 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21617⟩⟩) (.product (.predecessor 0 91955 .coefficient) (.predecessor 1 91956 .coefficient) (⟨false, false, none, none, none⟩))

def event91958 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21617⟩⟩, .operator (⟨91954, 0⟩, ⟨91952, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21616⟩⟩]⟩, (1)⟩)

def exact91959RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21616⟩⟩]⟩, (1)⟩]

theorem exact91959RawTermsValid :
    exact91959RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event91959 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21617⟩⟩) exact91959RawTerms .large 91957 .exactZero (none)

def event91960 : Event := .preFoldPolynomial 91959 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21616⟩⟩]⟩, (1)⟩] .exactZero none

def exact91961RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21616⟩⟩]⟩, (1)⟩]

def event91961 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨21617⟩⟩) 91960 exact91961RawTerms .large 91957 .exactZero (none)

def event91962 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨28299⟩⟩)

def event91963 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event91964 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event91965 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.authority (.operator))

def event91966 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.finite 1)

def event91967 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event91968 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event91969 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event91970 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event91971 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 91970

def event91972 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 91968

def event91973 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 91971 .coefficient) (.value (.predecessor 1 91972 .coefficient)))

def event91974 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event91975 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5505⟩⟩) 0 ⟨5503⟩ 91974

def event91976 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5505⟩⟩) 1 ⟨2348⟩ 91966

def event91977 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5505⟩⟩) (.sum [.predecessor 0 91975 .coefficient, .predecessor 1 91976 .coefficient])

def event91978 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5505⟩⟩) (.finite 218)

def event91979 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5536⟩⟩) 0 ⟨5505⟩ 91978

def event91980 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5536⟩⟩) 1 ⟨961⟩ 91964

def event91981 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5536⟩⟩) (.identity (.predecessor 1 91980 .coefficient))

def event91982 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5536⟩⟩) (.finite 224)

def event91983 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11637⟩⟩) 0 ⟨5536⟩ 91982

def event91984 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11637⟩⟩) (.authority (.programFamilyFact))

def exact91985RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11637⟩⟩], []⟩, (1)⟩]

theorem exact91985RawTermsValid :
    exact91985RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event91985 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11637⟩⟩) exact91985RawTerms (.finite 28) 91984 .exactZero (none)

def event91986 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14641⟩⟩) 0 ⟨5536⟩ 91982

def event91987 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14641⟩⟩) (.authority (.programFamilyFact))

def exact91988RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨14641⟩⟩], []⟩, (1)⟩]

theorem exact91988RawTermsValid :
    exact91988RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event91988 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14641⟩⟩) exact91988RawTerms (.finite 28) 91987 .exactZero (none)

def event91989 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14642⟩⟩) 0 ⟨14641⟩ 91988

def event91990 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14642⟩⟩) 1 ⟨11637⟩ 91985

def event91991 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14642⟩⟩) (.product (.predecessor 0 91989 .coefficient) (.predecessor 1 91990 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event91992 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨14642⟩⟩, .operator (⟨91988, 0⟩, ⟨91985, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨11637⟩⟩, ⟨.program ⟨214⟩, ⟨14641⟩⟩], []⟩, (1)⟩)

def exact91993RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11637⟩⟩, ⟨.program ⟨214⟩, ⟨14641⟩⟩], []⟩, (1)⟩]

theorem exact91993RawTermsValid :
    exact91993RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event91993 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14642⟩⟩) exact91993RawTerms (.finite 784) 91991 .exactZero (none)

def event91994 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14643⟩⟩) 0 ⟨14642⟩ 91993

def event91995 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14643⟩⟩) (.identity (.predecessor 0 91994 .coefficient))

def event91996 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨14643⟩⟩) (.finite 784)

def event91997 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16178⟩⟩) 0 ⟨14643⟩ 91996

def event91998 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16178⟩⟩) (.authority (.programFamilyFact))

def exact91999RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16178⟩⟩], []⟩, (1)⟩]

theorem exact91999RawTermsValid :
    exact91999RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event91999 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16178⟩⟩) exact91999RawTerms (.finite 28) 91998 .exactZero (none)

def event92000 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16179⟩⟩) 0 ⟨16178⟩ 91999

def event92001 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16179⟩⟩) (.identity (.predecessor 0 92000 .coefficient))

def event92002 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨16179⟩⟩) (.finite 28)

def event92003 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24286⟩⟩) 0 ⟨16179⟩ 92002

def event92004 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24286⟩⟩) (.authority (.programFamilyFact))

def event92005 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨24286⟩⟩) (.finite 3720)

def event92006 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨6689⟩⟩) .missing

def event92007 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24287⟩⟩) 0 ⟨6689⟩ 92006

def event92008 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24287⟩⟩) 1 ⟨24286⟩ 92005

def event92009 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24287⟩⟩) (.authority (.operator))

def exact92010RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨24287⟩⟩]⟩, (1)⟩]

theorem exact92010RawTermsValid :
    exact92010RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event92010 : Event := .resultExact (⟨.program ⟨214⟩, ⟨24287⟩⟩) exact92010RawTerms .large 92009 .exactZero (none)

def event92011 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28293⟩⟩) 0 ⟨24287⟩ 92010

def event92012 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28293⟩⟩) (.authority (.operator))

def exact92013RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨28293⟩⟩]⟩, (1)⟩]

theorem exact92013RawTermsValid :
    exact92013RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event92013 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28293⟩⟩) exact92013RawTerms (.finite 8192) 92012 .exactZero (none)

def event92014 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨110⟩⟩) (.authority (.operator))

def event92015 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨110⟩⟩) .exactZero

def event92016 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16218⟩⟩) 0 ⟨16179⟩ 92002

def event92017 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16218⟩⟩) 1 ⟨110⟩ 92015

def event92018 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16218⟩⟩) (.sum [.predecessor 0 92016 .coefficient, .predecessor 1 92017 .coefficient])

def event92019 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨16218⟩⟩) (.finite 28)

def event92020 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16219⟩⟩) 0 ⟨16218⟩ 92019

def event92021 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16219⟩⟩) (.identity (.predecessor 0 92020 .coefficient))

def exact92022RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16178⟩⟩], []⟩, (1)⟩]

theorem exact92022RawTermsValid :
    exact92022RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event92022 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16219⟩⟩) exact92022RawTerms (.finite 28) 92021 .exactZero (none)

def event92023 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6544⟩⟩) (.authority (.factStore))

def exact92024RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact92024RawTermsValid :
    exact92024RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event92024 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6544⟩⟩) exact92024RawTerms .large 92023 .exactZero (none)

def event92025 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16220⟩⟩) 0 ⟨6544⟩ 92024

def event92026 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16220⟩⟩) 1 ⟨16219⟩ 92022

def event92027 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16220⟩⟩) (.product (.predecessor 0 92025 .coefficient) (.predecessor 1 92026 .coefficient) (⟨false, false, none, none, none⟩))

def event92028 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨16220⟩⟩, .operator (⟨92024, 0⟩, ⟨92022, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨16178⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact92029RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16178⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact92029RawTermsValid :
    exact92029RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event92029 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16220⟩⟩) exact92029RawTerms .large 92027 .exactZero (none)

def event92030 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6699⟩⟩) 0 ⟨6689⟩ 92006

def event92031 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6699⟩⟩) (.authority (.operator))

def exact92032RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6699⟩⟩]⟩, (1)⟩]

theorem exact92032RawTermsValid :
    exact92032RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event92032 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6699⟩⟩) exact92032RawTerms .large 92031 .exactZero (none)

def event92033 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16221⟩⟩) 0 ⟨6699⟩ 92032

def event92034 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16221⟩⟩) 1 ⟨16220⟩ 92029

def event92035 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16221⟩⟩) (.sum [.predecessor 0 92033 .coefficient, .predecessor 1 92034 .coefficient])

def exact92036RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6699⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16178⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact92036RawTermsValid :
    exact92036RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event92036 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16221⟩⟩) exact92036RawTerms .large 92035 .exactZero (none)

def event92037 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28294⟩⟩) 0 ⟨16221⟩ 92036

def event92038 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28294⟩⟩) 1 ⟨28293⟩ 92013

def event92039 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28294⟩⟩) (.product (.predecessor 0 92037 .coefficient) (.predecessor 1 92038 .coefficient) (⟨false, false, none, none, none⟩))

def event92040 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28294⟩⟩, .operator (⟨92036, 0⟩, ⟨92013, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28293⟩⟩]⟩, (1)⟩)

def event92041 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28294⟩⟩, .operator (⟨92036, 1⟩, ⟨92013, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨16178⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28293⟩⟩]⟩, (-1)⟩)

def event92042 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨28294⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨16178⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28293⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨28293⟩⟩) ⟨24287⟩ 92010)

def event92043 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28294⟩⟩, .relation 92042 0, ⟨[⟨.program ⟨214⟩, ⟨16178⟩⟩], [⟨.program ⟨214⟩, ⟨24287⟩⟩]⟩, (-1)⟩)

def exact92044RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28293⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16178⟩⟩], [⟨.program ⟨214⟩, ⟨24287⟩⟩]⟩, (-1)⟩]

theorem exact92044RawTermsValid :
    exact92044RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event92044 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28294⟩⟩) exact92044RawTerms .large 92039 .exactZero (none)

def event92045 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17662⟩⟩) 0 ⟨16179⟩ 92002

def event92046 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨17662⟩⟩) (.authority (.programFamilyFact))

def exact92047RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨17662⟩⟩], []⟩, (1)⟩]

theorem exact92047RawTermsValid :
    exact92047RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event92047 : Event := .resultExact (⟨.program ⟨214⟩, ⟨17662⟩⟩) exact92047RawTerms (.finite 28) 92046 .exactZero (none)

def event92048 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17664⟩⟩) 0 ⟨6544⟩ 92024

def event92049 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17664⟩⟩) 1 ⟨17662⟩ 92047

def event92050 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨17664⟩⟩) (.product (.predecessor 0 92048 .coefficient) (.predecessor 1 92049 .coefficient) (⟨false, true, none, none, some 1⟩))

def event92051 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨17664⟩⟩, .operator (⟨92024, 0⟩, ⟨92047, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨17662⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact92052RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨17662⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact92052RawTermsValid :
    exact92052RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event92052 : Event := .resultExact (⟨.program ⟨214⟩, ⟨17664⟩⟩) exact92052RawTerms .large 92050 .exactZero (none)

def event92053 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6726⟩⟩) 0 ⟨6689⟩ 92006

def event92054 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6726⟩⟩) (.authority (.operator))

def exact92055RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6726⟩⟩]⟩, (1)⟩]

theorem exact92055RawTermsValid :
    exact92055RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event92055 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6726⟩⟩) exact92055RawTerms .large 92054 .exactZero (none)

def event92056 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17665⟩⟩) 0 ⟨6726⟩ 92055

def event92057 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17665⟩⟩) 1 ⟨17664⟩ 92052

def event92058 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨17665⟩⟩) (.sum [.predecessor 0 92056 .coefficient, .predecessor 1 92057 .coefficient])

def exact92059RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6726⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17662⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact92059RawTermsValid :
    exact92059RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event92059 : Event := .resultExact (⟨.program ⟨214⟩, ⟨17665⟩⟩) exact92059RawTerms .large 92058 .exactZero (none)

def event92060 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28299⟩⟩) 0 ⟨17665⟩ 92059

def event92061 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28299⟩⟩) 1 ⟨28294⟩ 92044

def event92062 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28299⟩⟩) (.sum [.predecessor 0 92060 .coefficient, .predecessor 1 92061 .coefficient])

def exact92063RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28293⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6726⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16178⟩⟩], [⟨.program ⟨214⟩, ⟨24287⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17662⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact92063RawTermsValid :
    exact92063RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event92063 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28299⟩⟩) exact92063RawTerms .large 92062 .exactZero (none)

def event92064 : Event := .preFoldPolynomial 92063 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28293⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6726⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16178⟩⟩], [⟨.program ⟨214⟩, ⟨24287⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17662⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩] .exactZero none

def exact92065RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28293⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6726⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16178⟩⟩], [⟨.program ⟨214⟩, ⟨24287⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17662⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

def event92065 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨28299⟩⟩) 92064 exact92065RawTerms .large 92062 .exactZero (none)

def event92066 : Event := .specializationComputed (⟨.program ⟨214⟩, ⟨16179⟩⟩) ⟨⟨139⟩, ⟨47⟩, ⟨109⟩⟩ ⟨91908, 92066⟩

def event92067 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨21619⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21616⟩⟩]⟩) (1) 0 2 (.universal 92066 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21616⟩⟩]⟩) (none) 92065)

def event92068 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21619⟩⟩, .relation 92067 1, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6726⟩⟩]⟩, (1)⟩)

def event92069 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21619⟩⟩, .relation 92067 0, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28293⟩⟩]⟩, (-1)⟩)

def event92070 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21619⟩⟩, .relation 92067 2, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16178⟩⟩], [⟨.program ⟨214⟩, ⟨24287⟩⟩]⟩, (1)⟩)

def event92071 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21619⟩⟩, .relation 92067 3, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨17662⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact92072RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28293⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6726⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16178⟩⟩], [⟨.program ⟨214⟩, ⟨24287⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨17662⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact92072RawTermsValid :
    exact92072RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event92072 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21619⟩⟩) exact92072RawTerms .large 91904 (.finite 1811303510016) (some (91906))

def event92073 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28296⟩⟩) 0 ⟨21619⟩ 92072

def event92074 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28296⟩⟩) 1 ⟨28295⟩ 91894

def event92075 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28296⟩⟩) (.sum [.predecessor 0 92073 .coefficient, .predecessor 1 92074 .coefficient])

def event92076 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28296⟩⟩, .operator (⟨92072, 0⟩, ⟨91894, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28293⟩⟩]⟩, (1)⟩)

def event92077 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28296⟩⟩, .operator (⟨92072, 2⟩, ⟨91894, 1⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16178⟩⟩], [⟨.program ⟨214⟩, ⟨24287⟩⟩]⟩, (-1)⟩)

def event92078 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28296⟩⟩) (.sum [.result 92072 .summary, .result 91894 .summary])

def exact92079RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6726⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨17662⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact92079RawTermsValid :
    exact92079RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event92079 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28296⟩⟩) exact92079RawTerms .large 92075 (.finite 1292180536164689260544) (some (92078))

def event92080 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28297⟩⟩) 0 ⟨28296⟩ 92079

def event92081 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28297⟩⟩) 1 ⟨6682⟩ 5679

def event92082 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28297⟩⟩) (.product (.predecessor 0 92080 .coefficient) (.predecessor 1 92081 .coefficient) (⟨false, false, none, none, none⟩))

def event92083 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28297⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨6681⟩⟩]⟩) [⟨.result 5675 .coefficient, false, none⟩])

def event92084 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28297⟩⟩) (.product (.result 92079 .summary) (.transfer 92083) (⟨false, false, none, none, none⟩))

def event92085 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28297⟩⟩, .operator (⟨92079, 0⟩, ⟨5679, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6726⟩⟩, ⟨.program ⟨214⟩, ⟨6681⟩⟩]⟩, (1)⟩)

def event92086 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28297⟩⟩, .operator (⟨92079, 1⟩, ⟨5679, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨17662⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨6681⟩⟩]⟩, (-1)⟩)

def event92087 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨28297⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨17662⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨6681⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨6681⟩⟩) ⟨6612⟩ 5672)

def event92088 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28297⟩⟩, .relation 92087 0, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨6502⟩⟩, ⟨.program ⟨214⟩, ⟨17662⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact92089RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6726⟩⟩, ⟨.program ⟨214⟩, ⟨6681⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨6502⟩⟩, ⟨.program ⟨214⟩, ⟨17662⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact92089RawTermsValid :
    exact92089RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event92089 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28297⟩⟩) exact92089RawTerms .large 92082 (.finite 4742323242612988221224648704) (some (92084))

def event92090 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24224⟩⟩) 0 ⟨6689⟩ 5477

def event92091 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24224⟩⟩) 1 ⟨24223⟩ 84234

def event92092 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24224⟩⟩) (.authority (.operator))

def exact92093RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨24224⟩⟩]⟩, (1)⟩]

theorem exact92093RawTermsValid :
    exact92093RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event92093 : Event := .resultExact (⟨.program ⟨214⟩, ⟨24224⟩⟩) exact92093RawTerms .large 92092 .exactZero (none)

def event92094 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28076⟩⟩) 0 ⟨24224⟩ 92093

def event92095 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28076⟩⟩) (.authority (.operator))

def exact92096RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨28076⟩⟩]⟩, (1)⟩]

theorem exact92096RawTermsValid :
    exact92096RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event92096 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28076⟩⟩) exact92096RawTerms (.finite 8192) 92095 .exactZero (none)

def event92097 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28078⟩⟩) 0 ⟨26145⟩ 84516

def event92098 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28078⟩⟩) 1 ⟨28076⟩ 92096

def event92099 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28078⟩⟩) (.product (.predecessor 0 92097 .coefficient) (.predecessor 1 92098 .coefficient) (⟨false, false, none, none, none⟩))

def event92100 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28078⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨28076⟩⟩]⟩) [⟨.result 92096 .coefficient, false, none⟩])

def event92101 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28078⟩⟩) (.product (.result 84516 .summary) (.transfer 92100) (⟨false, false, none, none, none⟩))

def event92102 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28078⟩⟩, .operator (⟨84516, 0⟩, ⟨92096, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6698⟩⟩, ⟨.program ⟨214⟩, ⟨28076⟩⟩]⟩, (1)⟩)

def event92103 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28078⟩⟩, .operator (⟨84516, 1⟩, ⟨92096, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16059⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28076⟩⟩]⟩, (-1)⟩)

def event92104 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨28078⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16059⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28076⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨28076⟩⟩) ⟨24224⟩ 92093)

def event92105 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28078⟩⟩, .relation 92104 0, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16059⟩⟩], [⟨.program ⟨214⟩, ⟨24224⟩⟩]⟩, (-1)⟩)

def exact92106RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6698⟩⟩, ⟨.program ⟨214⟩, ⟨28076⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16059⟩⟩], [⟨.program ⟨214⟩, ⟨24224⟩⟩]⟩, (-1)⟩]

theorem exact92106RawTermsValid :
    exact92106RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event92106 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28078⟩⟩) exact92106RawTerms .large 92099 (.finite 1292113297018323992576) (some (92101))

def event92107 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21472⟩⟩) 0 ⟨16060⟩ 4052

def event92108 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21472⟩⟩) (.authority (.relationPreimageSource ⟨45⟩))

def exact92109RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨21472⟩⟩]⟩, (1)⟩]

theorem exact92109RawTermsValid :
    exact92109RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event92109 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21472⟩⟩) exact92109RawTerms (.finite 136065468) 92108 .exactZero (none)

def event92110 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21474⟩⟩) 0 ⟨21472⟩ 92109

def event92111 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21474⟩⟩) 1 ⟨2348⟩ 4

def event92112 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21474⟩⟩) (.scale (.predecessor 0 92110 .coefficient) (.value (.predecessor 1 92111 .coefficient)))

def exact92113RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨21472⟩⟩]⟩, (1)⟩]

theorem exact92113RawTermsValid :
    exact92113RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event92113 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21474⟩⟩) exact92113RawTerms (.finite 136065468) 92112 .exactZero (none)

def event92114 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21475⟩⟩) 0 ⟨5541⟩ 80012

def event92115 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21475⟩⟩) 1 ⟨21474⟩ 92113

def event92116 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21475⟩⟩) (.product (.predecessor 0 92114 .coefficient) (.predecessor 1 92115 .coefficient) (⟨false, false, none, none, none⟩))

def event92117 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21475⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨21472⟩⟩]⟩) [⟨.result 92109 .coefficient, false, none⟩])

def event92118 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21475⟩⟩) (.product (.result 80012 .summary) (.transfer 92117) (⟨false, false, none, none, none⟩))

def event92119 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21475⟩⟩, .operator (⟨80012, 0⟩, ⟨92113, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21472⟩⟩]⟩, (1)⟩)

def event92120 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨21473⟩⟩)

def event92121 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event92122 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event92123 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.authority (.operator))

def event92124 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.finite 1)

def event92125 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event92126 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event92127 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event92128 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event92129 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 92128

def event92130 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 92126

def event92131 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 92129 .coefficient) (.value (.predecessor 1 92130 .coefficient)))

def event92132 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event92133 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5505⟩⟩) 0 ⟨5503⟩ 92132

def event92134 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5505⟩⟩) 1 ⟨2348⟩ 92124

def event92135 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5505⟩⟩) (.sum [.predecessor 0 92133 .coefficient, .predecessor 1 92134 .coefficient])

def event92136 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5505⟩⟩) (.finite 218)

def event92137 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5536⟩⟩) 0 ⟨5505⟩ 92136

def event92138 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5536⟩⟩) 1 ⟨961⟩ 92122

def event92139 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5536⟩⟩) (.identity (.predecessor 1 92138 .coefficient))

def event92140 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5536⟩⟩) (.finite 224)

def event92141 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11553⟩⟩) 0 ⟨5536⟩ 92140

def event92142 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11553⟩⟩) (.authority (.programFamilyFact))

def exact92143RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11553⟩⟩], []⟩, (1)⟩]

theorem exact92143RawTermsValid :
    exact92143RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event92143 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11553⟩⟩) exact92143RawTerms (.finite 22) 92142 .exactZero (none)

def event92144 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14424⟩⟩) 0 ⟨5536⟩ 92140

def event92145 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14424⟩⟩) (.authority (.programFamilyFact))

def exact92146RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨14424⟩⟩], []⟩, (1)⟩]

theorem exact92146RawTermsValid :
    exact92146RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event92146 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14424⟩⟩) exact92146RawTerms (.finite 22) 92145 .exactZero (none)

def event92147 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14425⟩⟩) 0 ⟨14424⟩ 92146

def event92148 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14425⟩⟩) 1 ⟨11553⟩ 92143

def event92149 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14425⟩⟩) (.product (.predecessor 0 92147 .coefficient) (.predecessor 1 92148 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event92150 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14425⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨11553⟩⟩, ⟨.program ⟨214⟩, ⟨14424⟩⟩], []⟩) [⟨.result 92146 .coefficient, true, some 1⟩, ⟨.result 92143 .coefficient, true, some 1⟩])

def event92151 : Event := .survivorFold (1) 92150

def exact92152RawTerms : List Term := []

theorem exact92152RawTermsValid :
    exact92152RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event92152 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14425⟩⟩) exact92152RawTerms (.finite 484) 92149 (.finite 484) (some (92150))

def event92153 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14426⟩⟩) 0 ⟨14425⟩ 92152

def event92154 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14426⟩⟩) (.identity (.predecessor 0 92153 .coefficient))

def event92155 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨14426⟩⟩) (.finite 484)

def event92156 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16059⟩⟩) 0 ⟨14426⟩ 92155

def event92157 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16059⟩⟩) (.authority (.programFamilyFact))

def exact92158RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16059⟩⟩], []⟩, (1)⟩]

theorem exact92158RawTermsValid :
    exact92158RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event92158 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16059⟩⟩) exact92158RawTerms (.finite 22) 92157 .exactZero (none)

def event92159 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16060⟩⟩) 0 ⟨16059⟩ 92158

def eventLeaf5744 : Array AnnotatedEvent := #[
  { event := event91904
    frameStart := 0 },
  { event := event91905
    frameStart := 0 },
  { event := event91906
    frameStart := 0 },
  { event := event91907
    frameStart := 0 },
  { event := event91908
    frameStart := 91908 },
  { event := event91909
    frameStart := 91908 },
  { event := event91910
    frameStart := 91908 },
  { event := event91911
    frameStart := 91908 },
  { event := event91912
    frameStart := 91908 },
  { event := event91913
    frameStart := 91908 },
  { event := event91914
    frameStart := 91908 },
  { event := event91915
    frameStart := 91908 },
  { event := event91916
    frameStart := 91908 },
  { event := event91917
    frameStart := 91908 },
  { event := event91918
    frameStart := 91908 },
  { event := event91919
    frameStart := 91908 }
]

def eventLeaf5745 : Array AnnotatedEvent := #[
  { event := event91920
    frameStart := 91908 },
  { event := event91921
    frameStart := 91908 },
  { event := event91922
    frameStart := 91908 },
  { event := event91923
    frameStart := 91908 },
  { event := event91924
    frameStart := 91908 },
  { event := event91925
    frameStart := 91908 },
  { event := event91926
    frameStart := 91908 },
  { event := event91927
    frameStart := 91908 },
  { event := event91928
    frameStart := 91908 },
  { event := event91929
    frameStart := 91908 },
  { event := event91930
    frameStart := 91908 },
  { event := event91931
    frameStart := 91908 },
  { event := event91932
    frameStart := 91908 },
  { event := event91933
    frameStart := 91908 },
  { event := event91934
    frameStart := 91908 },
  { event := event91935
    frameStart := 91908 }
]

def eventLeaf5746 : Array AnnotatedEvent := #[
  { event := event91936
    frameStart := 91908 },
  { event := event91937
    frameStart := 91908 },
  { event := event91938
    frameStart := 91908 },
  { event := event91939
    frameStart := 91908 },
  { event := event91940
    frameStart := 91908 },
  { event := event91941
    frameStart := 91908 },
  { event := event91942
    frameStart := 91908 },
  { event := event91943
    frameStart := 91908 },
  { event := event91944
    frameStart := 91908 },
  { event := event91945
    frameStart := 91908 },
  { event := event91946
    frameStart := 91908 },
  { event := event91947
    frameStart := 91908 },
  { event := event91948
    frameStart := 91908 },
  { event := event91949
    frameStart := 91908 },
  { event := event91950
    frameStart := 91908 },
  { event := event91951
    frameStart := 91908 }
]

def eventLeaf5747 : Array AnnotatedEvent := #[
  { event := event91952
    frameStart := 91908 },
  { event := event91953
    frameStart := 91908 },
  { event := event91954
    frameStart := 91908 },
  { event := event91955
    frameStart := 91908 },
  { event := event91956
    frameStart := 91908 },
  { event := event91957
    frameStart := 91908 },
  { event := event91958
    frameStart := 91908 },
  { event := event91959
    frameStart := 91908 },
  { event := event91960
    frameStart := 91908 },
  { event := event91961
    frameStart := 91908 },
  { event := event91962
    frameStart := 91962 },
  { event := event91963
    frameStart := 91962 },
  { event := event91964
    frameStart := 91962 },
  { event := event91965
    frameStart := 91962 },
  { event := event91966
    frameStart := 91962 },
  { event := event91967
    frameStart := 91962 }
]

def eventLeaf5748 : Array AnnotatedEvent := #[
  { event := event91968
    frameStart := 91962 },
  { event := event91969
    frameStart := 91962 },
  { event := event91970
    frameStart := 91962 },
  { event := event91971
    frameStart := 91962 },
  { event := event91972
    frameStart := 91962 },
  { event := event91973
    frameStart := 91962 },
  { event := event91974
    frameStart := 91962 },
  { event := event91975
    frameStart := 91962 },
  { event := event91976
    frameStart := 91962 },
  { event := event91977
    frameStart := 91962 },
  { event := event91978
    frameStart := 91962 },
  { event := event91979
    frameStart := 91962 },
  { event := event91980
    frameStart := 91962 },
  { event := event91981
    frameStart := 91962 },
  { event := event91982
    frameStart := 91962 },
  { event := event91983
    frameStart := 91962 }
]

def eventLeaf5749 : Array AnnotatedEvent := #[
  { event := event91984
    frameStart := 91962 },
  { event := event91985
    frameStart := 91962 },
  { event := event91986
    frameStart := 91962 },
  { event := event91987
    frameStart := 91962 },
  { event := event91988
    frameStart := 91962 },
  { event := event91989
    frameStart := 91962 },
  { event := event91990
    frameStart := 91962 },
  { event := event91991
    frameStart := 91962 },
  { event := event91992
    frameStart := 91962 },
  { event := event91993
    frameStart := 91962 },
  { event := event91994
    frameStart := 91962 },
  { event := event91995
    frameStart := 91962 },
  { event := event91996
    frameStart := 91962 },
  { event := event91997
    frameStart := 91962 },
  { event := event91998
    frameStart := 91962 },
  { event := event91999
    frameStart := 91962 }
]

def eventLeaf5750 : Array AnnotatedEvent := #[
  { event := event92000
    frameStart := 91962 },
  { event := event92001
    frameStart := 91962 },
  { event := event92002
    frameStart := 91962 },
  { event := event92003
    frameStart := 91962 },
  { event := event92004
    frameStart := 91962 },
  { event := event92005
    frameStart := 91962 },
  { event := event92006
    frameStart := 91962 },
  { event := event92007
    frameStart := 91962 },
  { event := event92008
    frameStart := 91962 },
  { event := event92009
    frameStart := 91962 },
  { event := event92010
    frameStart := 91962 },
  { event := event92011
    frameStart := 91962 },
  { event := event92012
    frameStart := 91962 },
  { event := event92013
    frameStart := 91962 },
  { event := event92014
    frameStart := 91962 },
  { event := event92015
    frameStart := 91962 }
]

def eventLeaf5751 : Array AnnotatedEvent := #[
  { event := event92016
    frameStart := 91962 },
  { event := event92017
    frameStart := 91962 },
  { event := event92018
    frameStart := 91962 },
  { event := event92019
    frameStart := 91962 },
  { event := event92020
    frameStart := 91962 },
  { event := event92021
    frameStart := 91962 },
  { event := event92022
    frameStart := 91962 },
  { event := event92023
    frameStart := 91962 },
  { event := event92024
    frameStart := 91962 },
  { event := event92025
    frameStart := 91962 },
  { event := event92026
    frameStart := 91962 },
  { event := event92027
    frameStart := 91962 },
  { event := event92028
    frameStart := 91962 },
  { event := event92029
    frameStart := 91962 },
  { event := event92030
    frameStart := 91962 },
  { event := event92031
    frameStart := 91962 }
]

def eventLeaf5752 : Array AnnotatedEvent := #[
  { event := event92032
    frameStart := 91962 },
  { event := event92033
    frameStart := 91962 },
  { event := event92034
    frameStart := 91962 },
  { event := event92035
    frameStart := 91962 },
  { event := event92036
    frameStart := 91962 },
  { event := event92037
    frameStart := 91962 },
  { event := event92038
    frameStart := 91962 },
  { event := event92039
    frameStart := 91962 },
  { event := event92040
    frameStart := 91962 },
  { event := event92041
    frameStart := 91962 },
  { event := event92042
    frameStart := 91962 },
  { event := event92043
    frameStart := 91962 },
  { event := event92044
    frameStart := 91962 },
  { event := event92045
    frameStart := 91962 },
  { event := event92046
    frameStart := 91962 },
  { event := event92047
    frameStart := 91962 }
]

def eventLeaf5753 : Array AnnotatedEvent := #[
  { event := event92048
    frameStart := 91962 },
  { event := event92049
    frameStart := 91962 },
  { event := event92050
    frameStart := 91962 },
  { event := event92051
    frameStart := 91962 },
  { event := event92052
    frameStart := 91962 },
  { event := event92053
    frameStart := 91962 },
  { event := event92054
    frameStart := 91962 },
  { event := event92055
    frameStart := 91962 },
  { event := event92056
    frameStart := 91962 },
  { event := event92057
    frameStart := 91962 },
  { event := event92058
    frameStart := 91962 },
  { event := event92059
    frameStart := 91962 },
  { event := event92060
    frameStart := 91962 },
  { event := event92061
    frameStart := 91962 },
  { event := event92062
    frameStart := 91962 },
  { event := event92063
    frameStart := 91962 }
]

def eventLeaf5754 : Array AnnotatedEvent := #[
  { event := event92064
    frameStart := 91962 },
  { event := event92065
    frameStart := 91962 },
  { event := event92066
    frameStart := 0 },
  { event := event92067
    frameStart := 0 },
  { event := event92068
    frameStart := 0 },
  { event := event92069
    frameStart := 0 },
  { event := event92070
    frameStart := 0 },
  { event := event92071
    frameStart := 0 },
  { event := event92072
    frameStart := 0 },
  { event := event92073
    frameStart := 0 },
  { event := event92074
    frameStart := 0 },
  { event := event92075
    frameStart := 0 },
  { event := event92076
    frameStart := 0 },
  { event := event92077
    frameStart := 0 },
  { event := event92078
    frameStart := 0 },
  { event := event92079
    frameStart := 0 }
]

def eventLeaf5755 : Array AnnotatedEvent := #[
  { event := event92080
    frameStart := 0 },
  { event := event92081
    frameStart := 0 },
  { event := event92082
    frameStart := 0 },
  { event := event92083
    frameStart := 0 },
  { event := event92084
    frameStart := 0 },
  { event := event92085
    frameStart := 0 },
  { event := event92086
    frameStart := 0 },
  { event := event92087
    frameStart := 0 },
  { event := event92088
    frameStart := 0 },
  { event := event92089
    frameStart := 0 },
  { event := event92090
    frameStart := 0 },
  { event := event92091
    frameStart := 0 },
  { event := event92092
    frameStart := 0 },
  { event := event92093
    frameStart := 0 },
  { event := event92094
    frameStart := 0 },
  { event := event92095
    frameStart := 0 }
]

def eventLeaf5756 : Array AnnotatedEvent := #[
  { event := event92096
    frameStart := 0 },
  { event := event92097
    frameStart := 0 },
  { event := event92098
    frameStart := 0 },
  { event := event92099
    frameStart := 0 },
  { event := event92100
    frameStart := 0 },
  { event := event92101
    frameStart := 0 },
  { event := event92102
    frameStart := 0 },
  { event := event92103
    frameStart := 0 },
  { event := event92104
    frameStart := 0 },
  { event := event92105
    frameStart := 0 },
  { event := event92106
    frameStart := 0 },
  { event := event92107
    frameStart := 0 },
  { event := event92108
    frameStart := 0 },
  { event := event92109
    frameStart := 0 },
  { event := event92110
    frameStart := 0 },
  { event := event92111
    frameStart := 0 }
]

def eventLeaf5757 : Array AnnotatedEvent := #[
  { event := event92112
    frameStart := 0 },
  { event := event92113
    frameStart := 0 },
  { event := event92114
    frameStart := 0 },
  { event := event92115
    frameStart := 0 },
  { event := event92116
    frameStart := 0 },
  { event := event92117
    frameStart := 0 },
  { event := event92118
    frameStart := 0 },
  { event := event92119
    frameStart := 0 },
  { event := event92120
    frameStart := 92120 },
  { event := event92121
    frameStart := 92120 },
  { event := event92122
    frameStart := 92120 },
  { event := event92123
    frameStart := 92120 },
  { event := event92124
    frameStart := 92120 },
  { event := event92125
    frameStart := 92120 },
  { event := event92126
    frameStart := 92120 },
  { event := event92127
    frameStart := 92120 }
]

def eventLeaf5758 : Array AnnotatedEvent := #[
  { event := event92128
    frameStart := 92120 },
  { event := event92129
    frameStart := 92120 },
  { event := event92130
    frameStart := 92120 },
  { event := event92131
    frameStart := 92120 },
  { event := event92132
    frameStart := 92120 },
  { event := event92133
    frameStart := 92120 },
  { event := event92134
    frameStart := 92120 },
  { event := event92135
    frameStart := 92120 },
  { event := event92136
    frameStart := 92120 },
  { event := event92137
    frameStart := 92120 },
  { event := event92138
    frameStart := 92120 },
  { event := event92139
    frameStart := 92120 },
  { event := event92140
    frameStart := 92120 },
  { event := event92141
    frameStart := 92120 },
  { event := event92142
    frameStart := 92120 },
  { event := event92143
    frameStart := 92120 }
]

def eventLeaf5759 : Array AnnotatedEvent := #[
  { event := event92144
    frameStart := 92120 },
  { event := event92145
    frameStart := 92120 },
  { event := event92146
    frameStart := 92120 },
  { event := event92147
    frameStart := 92120 },
  { event := event92148
    frameStart := 92120 },
  { event := event92149
    frameStart := 92120 },
  { event := event92150
    frameStart := 92120 },
  { event := event92151
    frameStart := 92120 },
  { event := event92152
    frameStart := 92120 },
  { event := event92153
    frameStart := 92120 },
  { event := event92154
    frameStart := 92120 },
  { event := event92155
    frameStart := 92120 },
  { event := event92156
    frameStart := 92120 },
  { event := event92157
    frameStart := 92120 },
  { event := event92158
    frameStart := 92120 },
  { event := event92159
    frameStart := 92120 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events359
