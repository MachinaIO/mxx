import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events406

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event103936 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18117⟩⟩) (.sum [.predecessor 0 103934 .coefficient, .predecessor 1 103935 .coefficient])

def exact103937RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6742⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨18114⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact103937RawTermsValid :
    exact103937RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event103937 : Event := .resultExact (⟨.program ⟨214⟩, ⟨18117⟩⟩) exact103937RawTerms .large 103936 .exactZero (none)

def event103938 : Event := .predecessor (⟨.program ⟨214⟩, ⟨30060⟩⟩) 0 ⟨18117⟩ 103937

def event103939 : Event := .predecessor (⟨.program ⟨214⟩, ⟨30060⟩⟩) 1 ⟨30055⟩ 103922

def event103940 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨30060⟩⟩) (.sum [.predecessor 0 103938 .coefficient, .predecessor 1 103939 .coefficient])

def exact103941RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6707⟩⟩, ⟨.program ⟨214⟩, ⟨30054⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6742⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17001⟩⟩], [⟨.program ⟨214⟩, ⟨24782⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨18114⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact103941RawTermsValid :
    exact103941RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event103941 : Event := .resultExact (⟨.program ⟨214⟩, ⟨30060⟩⟩) exact103941RawTerms .large 103940 .exactZero (none)

def event103942 : Event := .preFoldPolynomial 103941 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6707⟩⟩, ⟨.program ⟨214⟩, ⟨30054⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6742⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17001⟩⟩], [⟨.program ⟨214⟩, ⟨24782⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨18114⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩] .exactZero none

def exact103943RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6707⟩⟩, ⟨.program ⟨214⟩, ⟨30054⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6742⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17001⟩⟩], [⟨.program ⟨214⟩, ⟨24782⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨18114⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

def event103943 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨30060⟩⟩) 103942 exact103943RawTerms .large 103940 .exactZero (none)

def event103944 : Event := .specializationComputed (⟨.program ⟨214⟩, ⟨17002⟩⟩) ⟨⟨155⟩, ⟨64⟩, ⟨109⟩⟩ ⟨103810, 103944⟩

def event103945 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨22760⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22757⟩⟩]⟩) (1) 0 2 (.universal 103944 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22757⟩⟩]⟩) (none) 103943)

def event103946 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨22760⟩⟩, .relation 103945 1, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6742⟩⟩]⟩, (1)⟩)

def event103947 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨22760⟩⟩, .relation 103945 0, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6707⟩⟩, ⟨.program ⟨214⟩, ⟨30054⟩⟩]⟩, (-1)⟩)

def event103948 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨22760⟩⟩, .relation 103945 2, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨17001⟩⟩], [⟨.program ⟨214⟩, ⟨24782⟩⟩]⟩, (1)⟩)

def event103949 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨22760⟩⟩, .relation 103945 3, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨18114⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact103950RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6707⟩⟩, ⟨.program ⟨214⟩, ⟨30054⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6742⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨17001⟩⟩], [⟨.program ⟨214⟩, ⟨24782⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨18114⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact103950RawTermsValid :
    exact103950RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event103950 : Event := .resultExact (⟨.program ⟨214⟩, ⟨22760⟩⟩) exact103950RawTerms .large 103806 (.finite 1811303510016) (some (103808))

def event103951 : Event := .predecessor (⟨.program ⟨214⟩, ⟨30057⟩⟩) 0 ⟨22760⟩ 103950

def event103952 : Event := .predecessor (⟨.program ⟨214⟩, ⟨30057⟩⟩) 1 ⟨30056⟩ 103796

def event103953 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨30057⟩⟩) (.sum [.predecessor 0 103951 .coefficient, .predecessor 1 103952 .coefficient])

def event103954 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30057⟩⟩, .operator (⟨103950, 0⟩, ⟨103796, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6707⟩⟩, ⟨.program ⟨214⟩, ⟨30054⟩⟩]⟩, (1)⟩)

def event103955 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30057⟩⟩, .operator (⟨103950, 2⟩, ⟨103796, 1⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨17001⟩⟩], [⟨.program ⟨214⟩, ⟨24782⟩⟩]⟩, (-1)⟩)

def event103956 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨30057⟩⟩) (.sum [.result 103950 .summary, .result 103796 .summary])

def exact103957RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6742⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨18114⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact103957RawTermsValid :
    exact103957RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event103957 : Event := .resultExact (⟨.program ⟨214⟩, ⟨30057⟩⟩) exact103957RawTerms .large 103953 (.finite 1292539135285018636288) (some (103956))

def event103958 : Event := .predecessor (⟨.program ⟨214⟩, ⟨30058⟩⟩) 0 ⟨30057⟩ 103957

def event103959 : Event := .predecessor (⟨.program ⟨214⟩, ⟨30058⟩⟩) 1 ⟨6658⟩ 5519

def event103960 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨30058⟩⟩) (.product (.predecessor 0 103958 .coefficient) (.predecessor 1 103959 .coefficient) (⟨false, false, none, none, none⟩))

def event103961 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨30058⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨6657⟩⟩]⟩) [⟨.result 5515 .coefficient, false, none⟩])

def event103962 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨30058⟩⟩) (.product (.result 103957 .summary) (.transfer 103961) (⟨false, false, none, none, none⟩))

def event103963 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30058⟩⟩, .operator (⟨103957, 0⟩, ⟨5519, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6742⟩⟩, ⟨.program ⟨214⟩, ⟨6657⟩⟩]⟩, (1)⟩)

def event103964 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30058⟩⟩, .operator (⟨103957, 1⟩, ⟨5519, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨18114⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨6657⟩⟩]⟩, (-1)⟩)

def event103965 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨30058⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨18114⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨6657⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨6657⟩⟩) ⟨6600⟩ 5512)

def event103966 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30058⟩⟩, .relation 103965 0, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6435⟩⟩, ⟨.program ⟨214⟩, ⟨18114⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact103967RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6742⟩⟩, ⟨.program ⟨214⟩, ⟨6657⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6435⟩⟩, ⟨.program ⟨214⟩, ⟨18114⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact103967RawTermsValid :
    exact103967RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event103967 : Event := .resultExact (⟨.program ⟨214⟩, ⟨30058⟩⟩) exact103967RawTerms .large 103960 (.finite 4743639307122182955475140608) (some (103962))

def event103968 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24719⟩⟩) 0 ⟨6689⟩ 5477

def event103969 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24719⟩⟩) 1 ⟨24718⟩ 94798

def event103970 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24719⟩⟩) (.authority (.operator))

def exact103971RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨24719⟩⟩]⟩, (1)⟩]

theorem exact103971RawTermsValid :
    exact103971RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event103971 : Event := .resultExact (⟨.program ⟨214⟩, ⟨24719⟩⟩) exact103971RawTerms .large 103970 .exactZero (none)

def event103972 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29777⟩⟩) 0 ⟨24719⟩ 103971

def event103973 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29777⟩⟩) (.authority (.operator))

def exact103974RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨29777⟩⟩]⟩, (1)⟩]

theorem exact103974RawTermsValid :
    exact103974RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event103974 : Event := .resultExact (⟨.program ⟨214⟩, ⟨29777⟩⟩) exact103974RawTerms (.finite 8192) 103973 .exactZero (none)

def event103975 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29779⟩⟩) 0 ⟨25670⟩ 95058

def event103976 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29779⟩⟩) 1 ⟨29777⟩ 103974

def event103977 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29779⟩⟩) (.product (.predecessor 0 103975 .coefficient) (.predecessor 1 103976 .coefficient) (⟨false, false, none, none, none⟩))

def event103978 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29779⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨29777⟩⟩]⟩) [⟨.result 103974 .coefficient, false, none⟩])

def event103979 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29779⟩⟩) (.product (.result 95058 .summary) (.transfer 103978) (⟨false, false, none, none, none⟩))

def event103980 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨29779⟩⟩, .operator (⟨95058, 0⟩, ⟨103974, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6706⟩⟩, ⟨.program ⟨214⟩, ⟨29777⟩⟩]⟩, (1)⟩)

def event103981 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨29779⟩⟩, .operator (⟨95058, 1⟩, ⟨103974, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16861⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29777⟩⟩]⟩, (-1)⟩)

def event103982 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨29779⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16861⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29777⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨29777⟩⟩) ⟨24719⟩ 103971)

def event103983 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨29779⟩⟩, .relation 103982 0, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16861⟩⟩], [⟨.program ⟨214⟩, ⟨24719⟩⟩]⟩, (-1)⟩)

def exact103984RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6706⟩⟩, ⟨.program ⟨214⟩, ⟨29777⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16861⟩⟩], [⟨.program ⟨214⟩, ⟨24719⟩⟩]⟩, (-1)⟩]

theorem exact103984RawTermsValid :
    exact103984RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event103984 : Event := .resultExact (⟨.program ⟨214⟩, ⟨29779⟩⟩) exact103984RawTerms .large 103977 (.finite 1292516721028694540288) (some (103979))

def event103985 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22613⟩⟩) 0 ⟨16862⟩ 4606

def event103986 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22613⟩⟩) (.authority (.relationPreimageSource ⟨62⟩))

def exact103987RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨22613⟩⟩]⟩, (1)⟩]

theorem exact103987RawTermsValid :
    exact103987RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event103987 : Event := .resultExact (⟨.program ⟨214⟩, ⟨22613⟩⟩) exact103987RawTerms (.finite 136065468) 103986 .exactZero (none)

def event103988 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22615⟩⟩) 0 ⟨22613⟩ 103987

def event103989 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22615⟩⟩) 1 ⟨2348⟩ 4

def event103990 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22615⟩⟩) (.scale (.predecessor 0 103988 .coefficient) (.value (.predecessor 1 103989 .coefficient)))

def exact103991RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨22613⟩⟩]⟩, (1)⟩]

theorem exact103991RawTermsValid :
    exact103991RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event103991 : Event := .resultExact (⟨.program ⟨214⟩, ⟨22615⟩⟩) exact103991RawTerms (.finite 136065468) 103990 .exactZero (none)

def event103992 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22616⟩⟩) 0 ⟨5509⟩ 94462

def event103993 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22616⟩⟩) 1 ⟨22615⟩ 103991

def event103994 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22616⟩⟩) (.product (.predecessor 0 103992 .coefficient) (.predecessor 1 103993 .coefficient) (⟨false, false, none, none, none⟩))

def event103995 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22616⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨22613⟩⟩]⟩) [⟨.result 103987 .coefficient, false, none⟩])

def event103996 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22616⟩⟩) (.product (.result 94462 .summary) (.transfer 103995) (⟨false, false, none, none, none⟩))

def event103997 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨22616⟩⟩, .operator (⟨94462, 0⟩, ⟨103991, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22613⟩⟩]⟩, (1)⟩)

def event103998 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨22614⟩⟩)

def event103999 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event104000 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event104001 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event104002 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event104003 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 104002

def event104004 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 104000

def event104005 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 104003 .coefficient) (.value (.predecessor 1 104004 .coefficient)))

def event104006 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event104007 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13130⟩⟩) 0 ⟨5503⟩ 104006

def event104008 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13130⟩⟩) (.authority (.programFamilyFact))

def exact104009RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨13130⟩⟩], []⟩, (1)⟩]

theorem exact104009RawTermsValid :
    exact104009RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event104009 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13130⟩⟩) exact104009RawTerms (.finite 58) 104008 .exactZero (none)

def event104010 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10225⟩⟩) 0 ⟨5503⟩ 104006

def event104011 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10225⟩⟩) (.authority (.programFamilyFact))

def exact104012RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨10225⟩⟩], []⟩, (1)⟩]

theorem exact104012RawTermsValid :
    exact104012RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event104012 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10225⟩⟩) exact104012RawTerms (.finite 58) 104011 .exactZero (none)

def event104013 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13131⟩⟩) 0 ⟨10225⟩ 104012

def event104014 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13131⟩⟩) 1 ⟨13130⟩ 104009

def event104015 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13131⟩⟩) (.product (.predecessor 0 104013 .coefficient) (.predecessor 1 104014 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event104016 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13131⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨10225⟩⟩, ⟨.program ⟨214⟩, ⟨13130⟩⟩], []⟩) [⟨.result 104012 .coefficient, true, some 1⟩, ⟨.result 104009 .coefficient, true, some 1⟩])

def event104017 : Event := .survivorFold (1) 104016

def exact104018RawTerms : List Term := []

theorem exact104018RawTermsValid :
    exact104018RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event104018 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13131⟩⟩) exact104018RawTerms (.finite 3364) 104015 (.finite 3364) (some (104016))

def event104019 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13132⟩⟩) 0 ⟨13131⟩ 104018

def event104020 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13132⟩⟩) (.identity (.predecessor 0 104019 .coefficient))

def event104021 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨13132⟩⟩) (.finite 3364)

def event104022 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16861⟩⟩) 0 ⟨13132⟩ 104021

def event104023 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16861⟩⟩) (.authority (.programFamilyFact))

def exact104024RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16861⟩⟩], []⟩, (1)⟩]

theorem exact104024RawTermsValid :
    exact104024RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event104024 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16861⟩⟩) exact104024RawTerms (.finite 58) 104023 .exactZero (none)

def event104025 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16862⟩⟩) 0 ⟨16861⟩ 104024

def event104026 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16862⟩⟩) (.identity (.predecessor 0 104025 .coefficient))

def event104027 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨16862⟩⟩) (.finite 58)

def event104028 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22613⟩⟩) 0 ⟨16862⟩ 104027

def event104029 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22613⟩⟩) (.authority (.relationPreimageSource ⟨62⟩))

def exact104030RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨22613⟩⟩]⟩, (1)⟩]

theorem exact104030RawTermsValid :
    exact104030RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event104030 : Event := .resultExact (⟨.program ⟨214⟩, ⟨22613⟩⟩) exact104030RawTerms (.finite 136065468) 104029 .exactZero (none)

def event104031 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6⟩⟩) (.authority (.operator))

def exact104032RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩]⟩, (1)⟩]

theorem exact104032RawTermsValid :
    exact104032RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event104032 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6⟩⟩) exact104032RawTerms .large 104031 .exactZero (none)

def event104033 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22614⟩⟩) 0 ⟨6⟩ 104032

def event104034 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22614⟩⟩) 1 ⟨22613⟩ 104030

def event104035 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22614⟩⟩) (.product (.predecessor 0 104033 .coefficient) (.predecessor 1 104034 .coefficient) (⟨false, false, none, none, none⟩))

def event104036 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨22614⟩⟩, .operator (⟨104032, 0⟩, ⟨104030, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22613⟩⟩]⟩, (1)⟩)

def exact104037RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22613⟩⟩]⟩, (1)⟩]

theorem exact104037RawTermsValid :
    exact104037RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event104037 : Event := .resultExact (⟨.program ⟨214⟩, ⟨22614⟩⟩) exact104037RawTerms .large 104035 .exactZero (none)

def event104038 : Event := .preFoldPolynomial 104037 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22613⟩⟩]⟩, (1)⟩] .exactZero none

def exact104039RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22613⟩⟩]⟩, (1)⟩]

def event104039 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨22614⟩⟩) 104038 exact104039RawTerms .large 104035 .exactZero (none)

def event104040 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨29783⟩⟩)

def event104041 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event104042 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event104043 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event104044 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event104045 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 104044

def event104046 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 104042

def event104047 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 104045 .coefficient) (.value (.predecessor 1 104046 .coefficient)))

def event104048 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event104049 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13130⟩⟩) 0 ⟨5503⟩ 104048

def event104050 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13130⟩⟩) (.authority (.programFamilyFact))

def exact104051RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨13130⟩⟩], []⟩, (1)⟩]

theorem exact104051RawTermsValid :
    exact104051RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event104051 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13130⟩⟩) exact104051RawTerms (.finite 58) 104050 .exactZero (none)

def event104052 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10225⟩⟩) 0 ⟨5503⟩ 104048

def event104053 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10225⟩⟩) (.authority (.programFamilyFact))

def exact104054RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨10225⟩⟩], []⟩, (1)⟩]

theorem exact104054RawTermsValid :
    exact104054RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event104054 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10225⟩⟩) exact104054RawTerms (.finite 58) 104053 .exactZero (none)

def event104055 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13131⟩⟩) 0 ⟨10225⟩ 104054

def event104056 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13131⟩⟩) 1 ⟨13130⟩ 104051

def event104057 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13131⟩⟩) (.product (.predecessor 0 104055 .coefficient) (.predecessor 1 104056 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event104058 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨13131⟩⟩, .operator (⟨104054, 0⟩, ⟨104051, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨10225⟩⟩, ⟨.program ⟨214⟩, ⟨13130⟩⟩], []⟩, (1)⟩)

def exact104059RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨10225⟩⟩, ⟨.program ⟨214⟩, ⟨13130⟩⟩], []⟩, (1)⟩]

theorem exact104059RawTermsValid :
    exact104059RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event104059 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13131⟩⟩) exact104059RawTerms (.finite 3364) 104057 .exactZero (none)

def event104060 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13132⟩⟩) 0 ⟨13131⟩ 104059

def event104061 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13132⟩⟩) (.identity (.predecessor 0 104060 .coefficient))

def event104062 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨13132⟩⟩) (.finite 3364)

def event104063 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16861⟩⟩) 0 ⟨13132⟩ 104062

def event104064 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16861⟩⟩) (.authority (.programFamilyFact))

def exact104065RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16861⟩⟩], []⟩, (1)⟩]

theorem exact104065RawTermsValid :
    exact104065RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event104065 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16861⟩⟩) exact104065RawTerms (.finite 58) 104064 .exactZero (none)

def event104066 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16862⟩⟩) 0 ⟨16861⟩ 104065

def event104067 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16862⟩⟩) (.identity (.predecessor 0 104066 .coefficient))

def event104068 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨16862⟩⟩) (.finite 58)

def event104069 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24718⟩⟩) 0 ⟨16862⟩ 104068

def event104070 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24718⟩⟩) (.authority (.programFamilyFact))

def event104071 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨24718⟩⟩) (.finite 3720)

def event104072 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨6689⟩⟩) .missing

def event104073 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24719⟩⟩) 0 ⟨6689⟩ 104072

def event104074 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24719⟩⟩) 1 ⟨24718⟩ 104071

def event104075 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24719⟩⟩) (.authority (.operator))

def exact104076RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨24719⟩⟩]⟩, (1)⟩]

theorem exact104076RawTermsValid :
    exact104076RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event104076 : Event := .resultExact (⟨.program ⟨214⟩, ⟨24719⟩⟩) exact104076RawTerms .large 104075 .exactZero (none)

def event104077 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29777⟩⟩) 0 ⟨24719⟩ 104076

def event104078 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29777⟩⟩) (.authority (.operator))

def exact104079RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨29777⟩⟩]⟩, (1)⟩]

theorem exact104079RawTermsValid :
    exact104079RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event104079 : Event := .resultExact (⟨.program ⟨214⟩, ⟨29777⟩⟩) exact104079RawTerms (.finite 8192) 104078 .exactZero (none)

def event104080 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨110⟩⟩) (.authority (.operator))

def event104081 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨110⟩⟩) .exactZero

def event104082 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16959⟩⟩) 0 ⟨16862⟩ 104068

def event104083 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16959⟩⟩) 1 ⟨110⟩ 104081

def event104084 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16959⟩⟩) (.sum [.predecessor 0 104082 .coefficient, .predecessor 1 104083 .coefficient])

def event104085 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨16959⟩⟩) (.finite 58)

def event104086 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16960⟩⟩) 0 ⟨16959⟩ 104085

def event104087 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16960⟩⟩) (.identity (.predecessor 0 104086 .coefficient))

def exact104088RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16861⟩⟩], []⟩, (1)⟩]

theorem exact104088RawTermsValid :
    exact104088RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event104088 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16960⟩⟩) exact104088RawTerms (.finite 58) 104087 .exactZero (none)

def event104089 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6544⟩⟩) (.authority (.factStore))

def exact104090RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact104090RawTermsValid :
    exact104090RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event104090 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6544⟩⟩) exact104090RawTerms .large 104089 .exactZero (none)

def event104091 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16961⟩⟩) 0 ⟨6544⟩ 104090

def event104092 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16961⟩⟩) 1 ⟨16960⟩ 104088

def event104093 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16961⟩⟩) (.product (.predecessor 0 104091 .coefficient) (.predecessor 1 104092 .coefficient) (⟨false, false, none, none, none⟩))

def event104094 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨16961⟩⟩, .operator (⟨104090, 0⟩, ⟨104088, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨16861⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact104095RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16861⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact104095RawTermsValid :
    exact104095RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event104095 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16961⟩⟩) exact104095RawTerms .large 104093 .exactZero (none)

def event104096 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6706⟩⟩) 0 ⟨6689⟩ 104072

def event104097 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6706⟩⟩) (.authority (.operator))

def exact104098RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6706⟩⟩]⟩, (1)⟩]

theorem exact104098RawTermsValid :
    exact104098RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event104098 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6706⟩⟩) exact104098RawTerms .large 104097 .exactZero (none)

def event104099 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16962⟩⟩) 0 ⟨6706⟩ 104098

def event104100 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16962⟩⟩) 1 ⟨16961⟩ 104095

def event104101 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16962⟩⟩) (.sum [.predecessor 0 104099 .coefficient, .predecessor 1 104100 .coefficient])

def exact104102RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6706⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16861⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact104102RawTermsValid :
    exact104102RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event104102 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16962⟩⟩) exact104102RawTerms .large 104101 .exactZero (none)

def event104103 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29778⟩⟩) 0 ⟨16962⟩ 104102

def event104104 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29778⟩⟩) 1 ⟨29777⟩ 104079

def event104105 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29778⟩⟩) (.product (.predecessor 0 104103 .coefficient) (.predecessor 1 104104 .coefficient) (⟨false, false, none, none, none⟩))

def event104106 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨29778⟩⟩, .operator (⟨104102, 0⟩, ⟨104079, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6706⟩⟩, ⟨.program ⟨214⟩, ⟨29777⟩⟩]⟩, (1)⟩)

def event104107 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨29778⟩⟩, .operator (⟨104102, 1⟩, ⟨104079, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨16861⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29777⟩⟩]⟩, (-1)⟩)

def event104108 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨29778⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨16861⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29777⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨29777⟩⟩) ⟨24719⟩ 104076)

def event104109 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨29778⟩⟩, .relation 104108 0, ⟨[⟨.program ⟨214⟩, ⟨16861⟩⟩], [⟨.program ⟨214⟩, ⟨24719⟩⟩]⟩, (-1)⟩)

def exact104110RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6706⟩⟩, ⟨.program ⟨214⟩, ⟨29777⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16861⟩⟩], [⟨.program ⟨214⟩, ⟨24719⟩⟩]⟩, (-1)⟩]

theorem exact104110RawTermsValid :
    exact104110RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event104110 : Event := .resultExact (⟨.program ⟨214⟩, ⟨29778⟩⟩) exact104110RawTerms .large 104105 .exactZero (none)

def event104111 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16917⟩⟩) 0 ⟨16862⟩ 104068

def event104112 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16917⟩⟩) (.authority (.programFamilyFact))

def exact104113RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16917⟩⟩], []⟩, (1)⟩]

theorem exact104113RawTermsValid :
    exact104113RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event104113 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16917⟩⟩) exact104113RawTerms (.finite 58) 104112 .exactZero (none)

def event104114 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16919⟩⟩) 0 ⟨6544⟩ 104090

def event104115 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16919⟩⟩) 1 ⟨16917⟩ 104113

def event104116 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16919⟩⟩) (.product (.predecessor 0 104114 .coefficient) (.predecessor 1 104115 .coefficient) (⟨false, true, none, none, some 1⟩))

def event104117 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨16919⟩⟩, .operator (⟨104090, 0⟩, ⟨104113, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨16917⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact104118RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16917⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact104118RawTermsValid :
    exact104118RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event104118 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16919⟩⟩) exact104118RawTerms .large 104116 .exactZero (none)

def event104119 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6740⟩⟩) 0 ⟨6689⟩ 104072

def event104120 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6740⟩⟩) (.authority (.operator))

def exact104121RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6740⟩⟩]⟩, (1)⟩]

theorem exact104121RawTermsValid :
    exact104121RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event104121 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6740⟩⟩) exact104121RawTerms .large 104120 .exactZero (none)

def event104122 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16920⟩⟩) 0 ⟨6740⟩ 104121

def event104123 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16920⟩⟩) 1 ⟨16919⟩ 104118

def event104124 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16920⟩⟩) (.sum [.predecessor 0 104122 .coefficient, .predecessor 1 104123 .coefficient])

def exact104125RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6740⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16917⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact104125RawTermsValid :
    exact104125RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event104125 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16920⟩⟩) exact104125RawTerms .large 104124 .exactZero (none)

def event104126 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29783⟩⟩) 0 ⟨16920⟩ 104125

def event104127 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29783⟩⟩) 1 ⟨29778⟩ 104110

def event104128 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29783⟩⟩) (.sum [.predecessor 0 104126 .coefficient, .predecessor 1 104127 .coefficient])

def exact104129RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6706⟩⟩, ⟨.program ⟨214⟩, ⟨29777⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6740⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16861⟩⟩], [⟨.program ⟨214⟩, ⟨24719⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16917⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact104129RawTermsValid :
    exact104129RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event104129 : Event := .resultExact (⟨.program ⟨214⟩, ⟨29783⟩⟩) exact104129RawTerms .large 104128 .exactZero (none)

def event104130 : Event := .preFoldPolynomial 104129 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6706⟩⟩, ⟨.program ⟨214⟩, ⟨29777⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6740⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16861⟩⟩], [⟨.program ⟨214⟩, ⟨24719⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16917⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩] .exactZero none

def exact104131RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6706⟩⟩, ⟨.program ⟨214⟩, ⟨29777⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6740⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16861⟩⟩], [⟨.program ⟨214⟩, ⟨24719⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16917⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

def event104131 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨29783⟩⟩) 104130 exact104131RawTerms .large 104128 .exactZero (none)

def event104132 : Event := .specializationComputed (⟨.program ⟨214⟩, ⟨16862⟩⟩) ⟨⟨153⟩, ⟨62⟩, ⟨109⟩⟩ ⟨103998, 104132⟩

def event104133 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨22616⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22613⟩⟩]⟩) (1) 0 2 (.universal 104132 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22613⟩⟩]⟩) (none) 104131)

def event104134 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨22616⟩⟩, .relation 104133 1, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6740⟩⟩]⟩, (1)⟩)

def event104135 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨22616⟩⟩, .relation 104133 0, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6706⟩⟩, ⟨.program ⟨214⟩, ⟨29777⟩⟩]⟩, (-1)⟩)

def event104136 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨22616⟩⟩, .relation 104133 2, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16861⟩⟩], [⟨.program ⟨214⟩, ⟨24719⟩⟩]⟩, (1)⟩)

def event104137 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨22616⟩⟩, .relation 104133 3, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16917⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact104138RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6706⟩⟩, ⟨.program ⟨214⟩, ⟨29777⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6740⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16861⟩⟩], [⟨.program ⟨214⟩, ⟨24719⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16917⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact104138RawTermsValid :
    exact104138RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event104138 : Event := .resultExact (⟨.program ⟨214⟩, ⟨22616⟩⟩) exact104138RawTerms .large 103994 (.finite 1811303510016) (some (103996))

def event104139 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29780⟩⟩) 0 ⟨22616⟩ 104138

def event104140 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29780⟩⟩) 1 ⟨29779⟩ 103984

def event104141 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29780⟩⟩) (.sum [.predecessor 0 104139 .coefficient, .predecessor 1 104140 .coefficient])

def event104142 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨29780⟩⟩, .operator (⟨104138, 0⟩, ⟨103984, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6706⟩⟩, ⟨.program ⟨214⟩, ⟨29777⟩⟩]⟩, (1)⟩)

def event104143 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨29780⟩⟩, .operator (⟨104138, 2⟩, ⟨103984, 1⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16861⟩⟩], [⟨.program ⟨214⟩, ⟨24719⟩⟩]⟩, (-1)⟩)

def event104144 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29780⟩⟩) (.sum [.result 104138 .summary, .result 103984 .summary])

def exact104145RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6740⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16917⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact104145RawTermsValid :
    exact104145RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event104145 : Event := .resultExact (⟨.program ⟨214⟩, ⟨29780⟩⟩) exact104145RawTerms .large 104141 (.finite 1292516722839998050304) (some (104144))

def event104146 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29781⟩⟩) 0 ⟨29780⟩ 104145

def event104147 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29781⟩⟩) 1 ⟨6660⟩ 5539

def event104148 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29781⟩⟩) (.product (.predecessor 0 104146 .coefficient) (.predecessor 1 104147 .coefficient) (⟨false, false, none, none, none⟩))

def event104149 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29781⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨6659⟩⟩]⟩) [⟨.result 5535 .coefficient, false, none⟩])

def event104150 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29781⟩⟩) (.product (.result 104145 .summary) (.transfer 104149) (⟨false, false, none, none, none⟩))

def event104151 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨29781⟩⟩, .operator (⟨104145, 0⟩, ⟨5539, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6740⟩⟩, ⟨.program ⟨214⟩, ⟨6659⟩⟩]⟩, (1)⟩)

def event104152 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨29781⟩⟩, .operator (⟨104145, 1⟩, ⟨5539, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16917⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨6659⟩⟩]⟩, (-1)⟩)

def event104153 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨29781⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16917⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨6659⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨6659⟩⟩) ⟨6601⟩ 5532)

def event104154 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨29781⟩⟩, .relation 104153 0, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6437⟩⟩, ⟨.program ⟨214⟩, ⟨16917⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact104155RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6740⟩⟩, ⟨.program ⟨214⟩, ⟨6659⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6437⟩⟩, ⟨.program ⟨214⟩, ⟨16917⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact104155RawTermsValid :
    exact104155RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event104155 : Event := .resultExact (⟨.program ⟨214⟩, ⟨29781⟩⟩) exact104155RawTerms .large 104148 (.finite 4743557053090358284584484864) (some (104150))

def event104156 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24656⟩⟩) 0 ⟨6689⟩ 5477

def event104157 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24656⟩⟩) 1 ⟨24655⟩ 95232

def event104158 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24656⟩⟩) (.authority (.operator))

def exact104159RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨24656⟩⟩]⟩, (1)⟩]

theorem exact104159RawTermsValid :
    exact104159RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event104159 : Event := .resultExact (⟨.program ⟨214⟩, ⟨24656⟩⟩) exact104159RawTerms .large 104158 .exactZero (none)

def event104160 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29560⟩⟩) 0 ⟨24656⟩ 104159

def event104161 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29560⟩⟩) (.authority (.operator))

def exact104162RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨29560⟩⟩]⟩, (1)⟩]

theorem exact104162RawTermsValid :
    exact104162RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event104162 : Event := .resultExact (⟨.program ⟨214⟩, ⟨29560⟩⟩) exact104162RawTerms (.finite 8192) 104161 .exactZero (none)

def event104163 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29562⟩⟩) 0 ⟨25593⟩ 95492

def event104164 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29562⟩⟩) 1 ⟨29560⟩ 104162

def event104165 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29562⟩⟩) (.product (.predecessor 0 104163 .coefficient) (.predecessor 1 104164 .coefficient) (⟨false, false, none, none, none⟩))

def event104166 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29562⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨29560⟩⟩]⟩) [⟨.result 104162 .coefficient, false, none⟩])

def event104167 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29562⟩⟩) (.product (.result 95492 .summary) (.transfer 104166) (⟨false, false, none, none, none⟩))

def event104168 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨29562⟩⟩, .operator (⟨95492, 0⟩, ⟨104162, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6705⟩⟩, ⟨.program ⟨214⟩, ⟨29560⟩⟩]⟩, (1)⟩)

def event104169 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨29562⟩⟩, .operator (⟨95492, 1⟩, ⟨104162, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16742⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29560⟩⟩]⟩, (-1)⟩)

def event104170 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨29562⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16742⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29560⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨29560⟩⟩) ⟨24656⟩ 104159)

def event104171 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨29562⟩⟩, .relation 104170 0, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16742⟩⟩], [⟨.program ⟨214⟩, ⟨24656⟩⟩]⟩, (-1)⟩)

def exact104172RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6705⟩⟩, ⟨.program ⟨214⟩, ⟨29560⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16742⟩⟩], [⟨.program ⟨214⟩, ⟨24656⟩⟩]⟩, (-1)⟩]

theorem exact104172RawTermsValid :
    exact104172RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event104172 : Event := .resultExact (⟨.program ⟨214⟩, ⟨29562⟩⟩) exact104172RawTerms .large 104165 (.finite 1292449483693632782336) (some (104167))

def event104173 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22469⟩⟩) 0 ⟨16743⟩ 4629

def event104174 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22469⟩⟩) (.authority (.relationPreimageSource ⟨60⟩))

def exact104175RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨22469⟩⟩]⟩, (1)⟩]

theorem exact104175RawTermsValid :
    exact104175RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event104175 : Event := .resultExact (⟨.program ⟨214⟩, ⟨22469⟩⟩) exact104175RawTerms (.finite 136065468) 104174 .exactZero (none)

def event104176 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22471⟩⟩) 0 ⟨22469⟩ 104175

def event104177 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22471⟩⟩) 1 ⟨2348⟩ 4

def event104178 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22471⟩⟩) (.scale (.predecessor 0 104176 .coefficient) (.value (.predecessor 1 104177 .coefficient)))

def exact104179RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨22469⟩⟩]⟩, (1)⟩]

theorem exact104179RawTermsValid :
    exact104179RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event104179 : Event := .resultExact (⟨.program ⟨214⟩, ⟨22471⟩⟩) exact104179RawTerms (.finite 136065468) 104178 .exactZero (none)

def event104180 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22472⟩⟩) 0 ⟨5509⟩ 94462

def event104181 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22472⟩⟩) 1 ⟨22471⟩ 104179

def event104182 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22472⟩⟩) (.product (.predecessor 0 104180 .coefficient) (.predecessor 1 104181 .coefficient) (⟨false, false, none, none, none⟩))

def event104183 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22472⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨22469⟩⟩]⟩) [⟨.result 104175 .coefficient, false, none⟩])

def event104184 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22472⟩⟩) (.product (.result 94462 .summary) (.transfer 104183) (⟨false, false, none, none, none⟩))

def event104185 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨22472⟩⟩, .operator (⟨94462, 0⟩, ⟨104179, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22469⟩⟩]⟩, (1)⟩)

def event104186 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨22470⟩⟩)

def event104187 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event104188 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event104189 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event104190 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event104191 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 104190

def eventLeaf6496 : Array AnnotatedEvent := #[
  { event := event103936
    frameStart := 103852 },
  { event := event103937
    frameStart := 103852 },
  { event := event103938
    frameStart := 103852 },
  { event := event103939
    frameStart := 103852 },
  { event := event103940
    frameStart := 103852 },
  { event := event103941
    frameStart := 103852 },
  { event := event103942
    frameStart := 103852 },
  { event := event103943
    frameStart := 103852 },
  { event := event103944
    frameStart := 0 },
  { event := event103945
    frameStart := 0 },
  { event := event103946
    frameStart := 0 },
  { event := event103947
    frameStart := 0 },
  { event := event103948
    frameStart := 0 },
  { event := event103949
    frameStart := 0 },
  { event := event103950
    frameStart := 0 },
  { event := event103951
    frameStart := 0 }
]

def eventLeaf6497 : Array AnnotatedEvent := #[
  { event := event103952
    frameStart := 0 },
  { event := event103953
    frameStart := 0 },
  { event := event103954
    frameStart := 0 },
  { event := event103955
    frameStart := 0 },
  { event := event103956
    frameStart := 0 },
  { event := event103957
    frameStart := 0 },
  { event := event103958
    frameStart := 0 },
  { event := event103959
    frameStart := 0 },
  { event := event103960
    frameStart := 0 },
  { event := event103961
    frameStart := 0 },
  { event := event103962
    frameStart := 0 },
  { event := event103963
    frameStart := 0 },
  { event := event103964
    frameStart := 0 },
  { event := event103965
    frameStart := 0 },
  { event := event103966
    frameStart := 0 },
  { event := event103967
    frameStart := 0 }
]

def eventLeaf6498 : Array AnnotatedEvent := #[
  { event := event103968
    frameStart := 0 },
  { event := event103969
    frameStart := 0 },
  { event := event103970
    frameStart := 0 },
  { event := event103971
    frameStart := 0 },
  { event := event103972
    frameStart := 0 },
  { event := event103973
    frameStart := 0 },
  { event := event103974
    frameStart := 0 },
  { event := event103975
    frameStart := 0 },
  { event := event103976
    frameStart := 0 },
  { event := event103977
    frameStart := 0 },
  { event := event103978
    frameStart := 0 },
  { event := event103979
    frameStart := 0 },
  { event := event103980
    frameStart := 0 },
  { event := event103981
    frameStart := 0 },
  { event := event103982
    frameStart := 0 },
  { event := event103983
    frameStart := 0 }
]

def eventLeaf6499 : Array AnnotatedEvent := #[
  { event := event103984
    frameStart := 0 },
  { event := event103985
    frameStart := 0 },
  { event := event103986
    frameStart := 0 },
  { event := event103987
    frameStart := 0 },
  { event := event103988
    frameStart := 0 },
  { event := event103989
    frameStart := 0 },
  { event := event103990
    frameStart := 0 },
  { event := event103991
    frameStart := 0 },
  { event := event103992
    frameStart := 0 },
  { event := event103993
    frameStart := 0 },
  { event := event103994
    frameStart := 0 },
  { event := event103995
    frameStart := 0 },
  { event := event103996
    frameStart := 0 },
  { event := event103997
    frameStart := 0 },
  { event := event103998
    frameStart := 103998 },
  { event := event103999
    frameStart := 103998 }
]

def eventLeaf6500 : Array AnnotatedEvent := #[
  { event := event104000
    frameStart := 103998 },
  { event := event104001
    frameStart := 103998 },
  { event := event104002
    frameStart := 103998 },
  { event := event104003
    frameStart := 103998 },
  { event := event104004
    frameStart := 103998 },
  { event := event104005
    frameStart := 103998 },
  { event := event104006
    frameStart := 103998 },
  { event := event104007
    frameStart := 103998 },
  { event := event104008
    frameStart := 103998 },
  { event := event104009
    frameStart := 103998 },
  { event := event104010
    frameStart := 103998 },
  { event := event104011
    frameStart := 103998 },
  { event := event104012
    frameStart := 103998 },
  { event := event104013
    frameStart := 103998 },
  { event := event104014
    frameStart := 103998 },
  { event := event104015
    frameStart := 103998 }
]

def eventLeaf6501 : Array AnnotatedEvent := #[
  { event := event104016
    frameStart := 103998 },
  { event := event104017
    frameStart := 103998 },
  { event := event104018
    frameStart := 103998 },
  { event := event104019
    frameStart := 103998 },
  { event := event104020
    frameStart := 103998 },
  { event := event104021
    frameStart := 103998 },
  { event := event104022
    frameStart := 103998 },
  { event := event104023
    frameStart := 103998 },
  { event := event104024
    frameStart := 103998 },
  { event := event104025
    frameStart := 103998 },
  { event := event104026
    frameStart := 103998 },
  { event := event104027
    frameStart := 103998 },
  { event := event104028
    frameStart := 103998 },
  { event := event104029
    frameStart := 103998 },
  { event := event104030
    frameStart := 103998 },
  { event := event104031
    frameStart := 103998 }
]

def eventLeaf6502 : Array AnnotatedEvent := #[
  { event := event104032
    frameStart := 103998 },
  { event := event104033
    frameStart := 103998 },
  { event := event104034
    frameStart := 103998 },
  { event := event104035
    frameStart := 103998 },
  { event := event104036
    frameStart := 103998 },
  { event := event104037
    frameStart := 103998 },
  { event := event104038
    frameStart := 103998 },
  { event := event104039
    frameStart := 103998 },
  { event := event104040
    frameStart := 104040 },
  { event := event104041
    frameStart := 104040 },
  { event := event104042
    frameStart := 104040 },
  { event := event104043
    frameStart := 104040 },
  { event := event104044
    frameStart := 104040 },
  { event := event104045
    frameStart := 104040 },
  { event := event104046
    frameStart := 104040 },
  { event := event104047
    frameStart := 104040 }
]

def eventLeaf6503 : Array AnnotatedEvent := #[
  { event := event104048
    frameStart := 104040 },
  { event := event104049
    frameStart := 104040 },
  { event := event104050
    frameStart := 104040 },
  { event := event104051
    frameStart := 104040 },
  { event := event104052
    frameStart := 104040 },
  { event := event104053
    frameStart := 104040 },
  { event := event104054
    frameStart := 104040 },
  { event := event104055
    frameStart := 104040 },
  { event := event104056
    frameStart := 104040 },
  { event := event104057
    frameStart := 104040 },
  { event := event104058
    frameStart := 104040 },
  { event := event104059
    frameStart := 104040 },
  { event := event104060
    frameStart := 104040 },
  { event := event104061
    frameStart := 104040 },
  { event := event104062
    frameStart := 104040 },
  { event := event104063
    frameStart := 104040 }
]

def eventLeaf6504 : Array AnnotatedEvent := #[
  { event := event104064
    frameStart := 104040 },
  { event := event104065
    frameStart := 104040 },
  { event := event104066
    frameStart := 104040 },
  { event := event104067
    frameStart := 104040 },
  { event := event104068
    frameStart := 104040 },
  { event := event104069
    frameStart := 104040 },
  { event := event104070
    frameStart := 104040 },
  { event := event104071
    frameStart := 104040 },
  { event := event104072
    frameStart := 104040 },
  { event := event104073
    frameStart := 104040 },
  { event := event104074
    frameStart := 104040 },
  { event := event104075
    frameStart := 104040 },
  { event := event104076
    frameStart := 104040 },
  { event := event104077
    frameStart := 104040 },
  { event := event104078
    frameStart := 104040 },
  { event := event104079
    frameStart := 104040 }
]

def eventLeaf6505 : Array AnnotatedEvent := #[
  { event := event104080
    frameStart := 104040 },
  { event := event104081
    frameStart := 104040 },
  { event := event104082
    frameStart := 104040 },
  { event := event104083
    frameStart := 104040 },
  { event := event104084
    frameStart := 104040 },
  { event := event104085
    frameStart := 104040 },
  { event := event104086
    frameStart := 104040 },
  { event := event104087
    frameStart := 104040 },
  { event := event104088
    frameStart := 104040 },
  { event := event104089
    frameStart := 104040 },
  { event := event104090
    frameStart := 104040 },
  { event := event104091
    frameStart := 104040 },
  { event := event104092
    frameStart := 104040 },
  { event := event104093
    frameStart := 104040 },
  { event := event104094
    frameStart := 104040 },
  { event := event104095
    frameStart := 104040 }
]

def eventLeaf6506 : Array AnnotatedEvent := #[
  { event := event104096
    frameStart := 104040 },
  { event := event104097
    frameStart := 104040 },
  { event := event104098
    frameStart := 104040 },
  { event := event104099
    frameStart := 104040 },
  { event := event104100
    frameStart := 104040 },
  { event := event104101
    frameStart := 104040 },
  { event := event104102
    frameStart := 104040 },
  { event := event104103
    frameStart := 104040 },
  { event := event104104
    frameStart := 104040 },
  { event := event104105
    frameStart := 104040 },
  { event := event104106
    frameStart := 104040 },
  { event := event104107
    frameStart := 104040 },
  { event := event104108
    frameStart := 104040 },
  { event := event104109
    frameStart := 104040 },
  { event := event104110
    frameStart := 104040 },
  { event := event104111
    frameStart := 104040 }
]

def eventLeaf6507 : Array AnnotatedEvent := #[
  { event := event104112
    frameStart := 104040 },
  { event := event104113
    frameStart := 104040 },
  { event := event104114
    frameStart := 104040 },
  { event := event104115
    frameStart := 104040 },
  { event := event104116
    frameStart := 104040 },
  { event := event104117
    frameStart := 104040 },
  { event := event104118
    frameStart := 104040 },
  { event := event104119
    frameStart := 104040 },
  { event := event104120
    frameStart := 104040 },
  { event := event104121
    frameStart := 104040 },
  { event := event104122
    frameStart := 104040 },
  { event := event104123
    frameStart := 104040 },
  { event := event104124
    frameStart := 104040 },
  { event := event104125
    frameStart := 104040 },
  { event := event104126
    frameStart := 104040 },
  { event := event104127
    frameStart := 104040 }
]

def eventLeaf6508 : Array AnnotatedEvent := #[
  { event := event104128
    frameStart := 104040 },
  { event := event104129
    frameStart := 104040 },
  { event := event104130
    frameStart := 104040 },
  { event := event104131
    frameStart := 104040 },
  { event := event104132
    frameStart := 0 },
  { event := event104133
    frameStart := 0 },
  { event := event104134
    frameStart := 0 },
  { event := event104135
    frameStart := 0 },
  { event := event104136
    frameStart := 0 },
  { event := event104137
    frameStart := 0 },
  { event := event104138
    frameStart := 0 },
  { event := event104139
    frameStart := 0 },
  { event := event104140
    frameStart := 0 },
  { event := event104141
    frameStart := 0 },
  { event := event104142
    frameStart := 0 },
  { event := event104143
    frameStart := 0 }
]

def eventLeaf6509 : Array AnnotatedEvent := #[
  { event := event104144
    frameStart := 0 },
  { event := event104145
    frameStart := 0 },
  { event := event104146
    frameStart := 0 },
  { event := event104147
    frameStart := 0 },
  { event := event104148
    frameStart := 0 },
  { event := event104149
    frameStart := 0 },
  { event := event104150
    frameStart := 0 },
  { event := event104151
    frameStart := 0 },
  { event := event104152
    frameStart := 0 },
  { event := event104153
    frameStart := 0 },
  { event := event104154
    frameStart := 0 },
  { event := event104155
    frameStart := 0 },
  { event := event104156
    frameStart := 0 },
  { event := event104157
    frameStart := 0 },
  { event := event104158
    frameStart := 0 },
  { event := event104159
    frameStart := 0 }
]

def eventLeaf6510 : Array AnnotatedEvent := #[
  { event := event104160
    frameStart := 0 },
  { event := event104161
    frameStart := 0 },
  { event := event104162
    frameStart := 0 },
  { event := event104163
    frameStart := 0 },
  { event := event104164
    frameStart := 0 },
  { event := event104165
    frameStart := 0 },
  { event := event104166
    frameStart := 0 },
  { event := event104167
    frameStart := 0 },
  { event := event104168
    frameStart := 0 },
  { event := event104169
    frameStart := 0 },
  { event := event104170
    frameStart := 0 },
  { event := event104171
    frameStart := 0 },
  { event := event104172
    frameStart := 0 },
  { event := event104173
    frameStart := 0 },
  { event := event104174
    frameStart := 0 },
  { event := event104175
    frameStart := 0 }
]

def eventLeaf6511 : Array AnnotatedEvent := #[
  { event := event104176
    frameStart := 0 },
  { event := event104177
    frameStart := 0 },
  { event := event104178
    frameStart := 0 },
  { event := event104179
    frameStart := 0 },
  { event := event104180
    frameStart := 0 },
  { event := event104181
    frameStart := 0 },
  { event := event104182
    frameStart := 0 },
  { event := event104183
    frameStart := 0 },
  { event := event104184
    frameStart := 0 },
  { event := event104185
    frameStart := 0 },
  { event := event104186
    frameStart := 104186 },
  { event := event104187
    frameStart := 104186 },
  { event := event104188
    frameStart := 104186 },
  { event := event104189
    frameStart := 104186 },
  { event := event104190
    frameStart := 104186 },
  { event := event104191
    frameStart := 104186 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events406
