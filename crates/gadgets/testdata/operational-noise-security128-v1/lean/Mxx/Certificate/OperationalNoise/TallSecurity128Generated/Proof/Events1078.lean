import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events1078

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event275968 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨68779⟩⟩) (.authority (.programFamilyFact))

def event275969 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨68779⟩⟩) (.finite 1152)

def event275970 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨7177⟩⟩) .missing

def event275971 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68780⟩⟩) 0 ⟨7177⟩ 275970

def event275972 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68780⟩⟩) 1 ⟨68779⟩ 275969

def event275973 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨68780⟩⟩) (.authority (.operator))

def exact275974RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (1)⟩]

theorem exact275974RawTermsValid :
    exact275974RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event275974 : Event := .resultExact (⟨.program ⟨257⟩, ⟨68780⟩⟩) exact275974RawTerms .large 275973 .exactZero (none)

def event275975 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70979⟩⟩) 0 ⟨68780⟩ 275974

def event275976 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70979⟩⟩) (.authority (.operator))

def exact275977RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (1)⟩]

theorem exact275977RawTermsValid :
    exact275977RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event275977 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70979⟩⟩) exact275977RawTerms (.finite 8192) 275976 .exactZero (none)

def event275978 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event275979 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event275980 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69055⟩⟩) 0 ⟨66029⟩ 275966

def event275981 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69055⟩⟩) 1 ⟨136⟩ 275979

def event275982 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69055⟩⟩) (.sum [.predecessor 0 275980 .coefficient, .predecessor 1 275981 .coefficient])

def event275983 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨69055⟩⟩) (.finite 1059)

def event275984 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69056⟩⟩) 0 ⟨69055⟩ 275983

def event275985 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69056⟩⟩) (.identity (.predecessor 0 275984 .coefficient))

def exact275986RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15903⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨18709⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨21929⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨26512⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨29192⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨31949⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨34856⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨37536⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨40212⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨42892⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨45576⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨48256⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨51004⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨53984⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨56964⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨59944⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨62924⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨66019⟩⟩], []⟩, (1)⟩]

theorem exact275986RawTermsValid :
    exact275986RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event275986 : Event := .resultExact (⟨.program ⟨257⟩, ⟨69056⟩⟩) exact275986RawTerms (.finite 1059) 275985 .exactZero (none)

def event275987 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact275988RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact275988RawTermsValid :
    exact275988RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event275988 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact275988RawTerms .large 275987 .exactZero (none)

def event275989 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69057⟩⟩) 0 ⟨6908⟩ 275988

def event275990 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69057⟩⟩) 1 ⟨69056⟩ 275986

def event275991 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69057⟩⟩) (.product (.predecessor 0 275989 .coefficient) (.predecessor 1 275990 .coefficient) (⟨false, false, none, none, none⟩))

def event275992 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨69057⟩⟩, .operator (⟨275988, 0⟩, ⟨275986, 11⟩), ⟨[⟨.program ⟨257⟩, ⟨48256⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def event275993 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨69057⟩⟩, .operator (⟨275988, 0⟩, ⟨275986, 10⟩), ⟨[⟨.program ⟨257⟩, ⟨45576⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def event275994 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨69057⟩⟩, .operator (⟨275988, 0⟩, ⟨275986, 9⟩), ⟨[⟨.program ⟨257⟩, ⟨42892⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def event275995 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨69057⟩⟩, .operator (⟨275988, 0⟩, ⟨275986, 8⟩), ⟨[⟨.program ⟨257⟩, ⟨40212⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def event275996 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨69057⟩⟩, .operator (⟨275988, 0⟩, ⟨275986, 7⟩), ⟨[⟨.program ⟨257⟩, ⟨37536⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def event275997 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨69057⟩⟩, .operator (⟨275988, 0⟩, ⟨275986, 6⟩), ⟨[⟨.program ⟨257⟩, ⟨34856⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def event275998 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨69057⟩⟩, .operator (⟨275988, 0⟩, ⟨275986, 4⟩), ⟨[⟨.program ⟨257⟩, ⟨29192⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def event275999 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨69057⟩⟩, .operator (⟨275988, 0⟩, ⟨275986, 3⟩), ⟨[⟨.program ⟨257⟩, ⟨26512⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def event276000 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨69057⟩⟩, .operator (⟨275988, 0⟩, ⟨275986, 17⟩), ⟨[⟨.program ⟨257⟩, ⟨66019⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def event276001 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨69057⟩⟩, .operator (⟨275988, 0⟩, ⟨275986, 16⟩), ⟨[⟨.program ⟨257⟩, ⟨62924⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def event276002 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨69057⟩⟩, .operator (⟨275988, 0⟩, ⟨275986, 15⟩), ⟨[⟨.program ⟨257⟩, ⟨59944⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def event276003 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨69057⟩⟩, .operator (⟨275988, 0⟩, ⟨275986, 14⟩), ⟨[⟨.program ⟨257⟩, ⟨56964⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def event276004 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨69057⟩⟩, .operator (⟨275988, 0⟩, ⟨275986, 13⟩), ⟨[⟨.program ⟨257⟩, ⟨53984⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def event276005 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨69057⟩⟩, .operator (⟨275988, 0⟩, ⟨275986, 12⟩), ⟨[⟨.program ⟨257⟩, ⟨51004⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def event276006 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨69057⟩⟩, .operator (⟨275988, 0⟩, ⟨275986, 5⟩), ⟨[⟨.program ⟨257⟩, ⟨31949⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def event276007 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨69057⟩⟩, .operator (⟨275988, 0⟩, ⟨275986, 2⟩), ⟨[⟨.program ⟨257⟩, ⟨21929⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def event276008 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨69057⟩⟩, .operator (⟨275988, 0⟩, ⟨275986, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨18709⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def event276009 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨69057⟩⟩, .operator (⟨275988, 0⟩, ⟨275986, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨15903⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact276010RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15903⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨18709⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨21929⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨26512⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨29192⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨31949⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨34856⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨37536⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨40212⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨42892⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨45576⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨48256⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨51004⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨53984⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨56964⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨59944⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨62924⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨66019⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact276010RawTermsValid :
    exact276010RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event276010 : Event := .resultExact (⟨.program ⟨257⟩, ⟨69057⟩⟩) exact276010RawTerms .large 275991 .exactZero (none)

def event276011 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7232⟩⟩) 0 ⟨7177⟩ 275970

def event276012 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7232⟩⟩) (.authority (.operator))

def exact276013RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7232⟩⟩]⟩, (1)⟩]

theorem exact276013RawTermsValid :
    exact276013RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event276013 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7232⟩⟩) exact276013RawTerms .large 276012 .exactZero (none)

def event276014 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7230⟩⟩) 0 ⟨7177⟩ 275970

def event276015 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7230⟩⟩) (.authority (.operator))

def exact276016RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7230⟩⟩]⟩, (1)⟩]

theorem exact276016RawTermsValid :
    exact276016RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event276016 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7230⟩⟩) exact276016RawTerms .large 276015 .exactZero (none)

def event276017 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7228⟩⟩) 0 ⟨7177⟩ 275970

def event276018 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7228⟩⟩) (.authority (.operator))

def exact276019RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7228⟩⟩]⟩, (1)⟩]

theorem exact276019RawTermsValid :
    exact276019RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event276019 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7228⟩⟩) exact276019RawTerms .large 276018 .exactZero (none)

def event276020 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7226⟩⟩) 0 ⟨7177⟩ 275970

def event276021 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7226⟩⟩) (.authority (.operator))

def exact276022RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7226⟩⟩]⟩, (1)⟩]

theorem exact276022RawTermsValid :
    exact276022RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event276022 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7226⟩⟩) exact276022RawTerms .large 276021 .exactZero (none)

def event276023 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7224⟩⟩) 0 ⟨7177⟩ 275970

def event276024 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7224⟩⟩) (.authority (.operator))

def exact276025RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7224⟩⟩]⟩, (1)⟩]

theorem exact276025RawTermsValid :
    exact276025RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event276025 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7224⟩⟩) exact276025RawTerms .large 276024 .exactZero (none)

def event276026 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7222⟩⟩) 0 ⟨7177⟩ 275970

def event276027 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7222⟩⟩) (.authority (.operator))

def exact276028RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩]

theorem exact276028RawTermsValid :
    exact276028RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event276028 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7222⟩⟩) exact276028RawTerms .large 276027 .exactZero (none)

def event276029 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7220⟩⟩) 0 ⟨7177⟩ 275970

def event276030 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7220⟩⟩) (.authority (.operator))

def exact276031RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩]

theorem exact276031RawTermsValid :
    exact276031RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event276031 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7220⟩⟩) exact276031RawTerms .large 276030 .exactZero (none)

def event276032 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7218⟩⟩) 0 ⟨7177⟩ 275970

def event276033 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7218⟩⟩) (.authority (.operator))

def exact276034RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩]

theorem exact276034RawTermsValid :
    exact276034RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event276034 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7218⟩⟩) exact276034RawTerms .large 276033 .exactZero (none)

def event276035 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7216⟩⟩) 0 ⟨7177⟩ 275970

def event276036 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7216⟩⟩) (.authority (.operator))

def exact276037RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩]

theorem exact276037RawTermsValid :
    exact276037RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event276037 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7216⟩⟩) exact276037RawTerms .large 276036 .exactZero (none)

def event276038 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7214⟩⟩) 0 ⟨7177⟩ 275970

def event276039 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7214⟩⟩) (.authority (.operator))

def exact276040RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩]

theorem exact276040RawTermsValid :
    exact276040RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event276040 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7214⟩⟩) exact276040RawTerms .large 276039 .exactZero (none)

def event276041 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7212⟩⟩) 0 ⟨7177⟩ 275970

def event276042 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7212⟩⟩) (.authority (.operator))

def exact276043RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩]

theorem exact276043RawTermsValid :
    exact276043RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event276043 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7212⟩⟩) exact276043RawTerms .large 276042 .exactZero (none)

def event276044 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7210⟩⟩) 0 ⟨7177⟩ 275970

def event276045 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7210⟩⟩) (.authority (.operator))

def exact276046RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩]

theorem exact276046RawTermsValid :
    exact276046RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event276046 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7210⟩⟩) exact276046RawTerms .large 276045 .exactZero (none)

def event276047 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7208⟩⟩) 0 ⟨7177⟩ 275970

def event276048 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7208⟩⟩) (.authority (.operator))

def exact276049RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩]

theorem exact276049RawTermsValid :
    exact276049RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event276049 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7208⟩⟩) exact276049RawTerms .large 276048 .exactZero (none)

def event276050 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7206⟩⟩) 0 ⟨7177⟩ 275970

def event276051 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7206⟩⟩) (.authority (.operator))

def exact276052RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩]

theorem exact276052RawTermsValid :
    exact276052RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event276052 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7206⟩⟩) exact276052RawTerms .large 276051 .exactZero (none)

def event276053 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7204⟩⟩) 0 ⟨7177⟩ 275970

def event276054 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7204⟩⟩) (.authority (.operator))

def exact276055RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩]

theorem exact276055RawTermsValid :
    exact276055RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event276055 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7204⟩⟩) exact276055RawTerms .large 276054 .exactZero (none)

def event276056 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7202⟩⟩) 0 ⟨7177⟩ 275970

def event276057 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7202⟩⟩) (.authority (.operator))

def exact276058RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩]

theorem exact276058RawTermsValid :
    exact276058RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event276058 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7202⟩⟩) exact276058RawTerms .large 276057 .exactZero (none)

def event276059 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7200⟩⟩) 0 ⟨7177⟩ 275970

def event276060 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7200⟩⟩) (.authority (.operator))

def exact276061RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩]

theorem exact276061RawTermsValid :
    exact276061RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event276061 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7200⟩⟩) exact276061RawTerms .large 276060 .exactZero (none)

def event276062 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7198⟩⟩) 0 ⟨7177⟩ 275970

def event276063 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7198⟩⟩) (.authority (.operator))

def exact276064RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩]

theorem exact276064RawTermsValid :
    exact276064RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event276064 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7198⟩⟩) exact276064RawTerms .large 276063 .exactZero (none)

def event276065 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7309⟩⟩) 0 ⟨7198⟩ 276064

def event276066 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7309⟩⟩) 1 ⟨7200⟩ 276061

def event276067 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7309⟩⟩) (.sum [.predecessor 0 276065 .coefficient, .predecessor 1 276066 .coefficient])

def exact276068RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩]

theorem exact276068RawTermsValid :
    exact276068RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event276068 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7309⟩⟩) exact276068RawTerms .large 276067 .exactZero (none)

def event276069 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7310⟩⟩) 0 ⟨7309⟩ 276068

def event276070 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7310⟩⟩) 1 ⟨7202⟩ 276058

def event276071 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7310⟩⟩) (.sum [.predecessor 0 276069 .coefficient, .predecessor 1 276070 .coefficient])

def exact276072RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩]

theorem exact276072RawTermsValid :
    exact276072RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event276072 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7310⟩⟩) exact276072RawTerms .large 276071 .exactZero (none)

def event276073 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7311⟩⟩) 0 ⟨7310⟩ 276072

def event276074 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7311⟩⟩) 1 ⟨7204⟩ 276055

def event276075 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7311⟩⟩) (.sum [.predecessor 0 276073 .coefficient, .predecessor 1 276074 .coefficient])

def exact276076RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩]

theorem exact276076RawTermsValid :
    exact276076RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event276076 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7311⟩⟩) exact276076RawTerms .large 276075 .exactZero (none)

def event276077 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7312⟩⟩) 0 ⟨7311⟩ 276076

def event276078 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7312⟩⟩) 1 ⟨7206⟩ 276052

def event276079 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7312⟩⟩) (.sum [.predecessor 0 276077 .coefficient, .predecessor 1 276078 .coefficient])

def exact276080RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩]

theorem exact276080RawTermsValid :
    exact276080RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event276080 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7312⟩⟩) exact276080RawTerms .large 276079 .exactZero (none)

def event276081 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7313⟩⟩) 0 ⟨7312⟩ 276080

def event276082 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7313⟩⟩) 1 ⟨7208⟩ 276049

def event276083 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7313⟩⟩) (.sum [.predecessor 0 276081 .coefficient, .predecessor 1 276082 .coefficient])

def exact276084RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩]

theorem exact276084RawTermsValid :
    exact276084RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event276084 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7313⟩⟩) exact276084RawTerms .large 276083 .exactZero (none)

def event276085 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7314⟩⟩) 0 ⟨7313⟩ 276084

def event276086 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7314⟩⟩) 1 ⟨7210⟩ 276046

def event276087 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7314⟩⟩) (.sum [.predecessor 0 276085 .coefficient, .predecessor 1 276086 .coefficient])

def exact276088RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩]

theorem exact276088RawTermsValid :
    exact276088RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event276088 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7314⟩⟩) exact276088RawTerms .large 276087 .exactZero (none)

def event276089 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7315⟩⟩) 0 ⟨7314⟩ 276088

def event276090 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7315⟩⟩) 1 ⟨7212⟩ 276043

def event276091 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7315⟩⟩) (.sum [.predecessor 0 276089 .coefficient, .predecessor 1 276090 .coefficient])

def exact276092RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩]

theorem exact276092RawTermsValid :
    exact276092RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event276092 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7315⟩⟩) exact276092RawTerms .large 276091 .exactZero (none)

def event276093 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7316⟩⟩) 0 ⟨7315⟩ 276092

def event276094 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7316⟩⟩) 1 ⟨7214⟩ 276040

def event276095 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7316⟩⟩) (.sum [.predecessor 0 276093 .coefficient, .predecessor 1 276094 .coefficient])

def exact276096RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩]

theorem exact276096RawTermsValid :
    exact276096RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event276096 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7316⟩⟩) exact276096RawTerms .large 276095 .exactZero (none)

def event276097 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7317⟩⟩) 0 ⟨7316⟩ 276096

def event276098 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7317⟩⟩) 1 ⟨7216⟩ 276037

def event276099 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7317⟩⟩) (.sum [.predecessor 0 276097 .coefficient, .predecessor 1 276098 .coefficient])

def exact276100RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩]

theorem exact276100RawTermsValid :
    exact276100RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event276100 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7317⟩⟩) exact276100RawTerms .large 276099 .exactZero (none)

def event276101 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7318⟩⟩) 0 ⟨7317⟩ 276100

def event276102 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7318⟩⟩) 1 ⟨7218⟩ 276034

def event276103 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7318⟩⟩) (.sum [.predecessor 0 276101 .coefficient, .predecessor 1 276102 .coefficient])

def exact276104RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩]

theorem exact276104RawTermsValid :
    exact276104RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event276104 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7318⟩⟩) exact276104RawTerms .large 276103 .exactZero (none)

def event276105 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7319⟩⟩) 0 ⟨7318⟩ 276104

def event276106 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7319⟩⟩) 1 ⟨7220⟩ 276031

def event276107 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7319⟩⟩) (.sum [.predecessor 0 276105 .coefficient, .predecessor 1 276106 .coefficient])

def exact276108RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩]

theorem exact276108RawTermsValid :
    exact276108RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event276108 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7319⟩⟩) exact276108RawTerms .large 276107 .exactZero (none)

def event276109 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7320⟩⟩) 0 ⟨7319⟩ 276108

def event276110 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7320⟩⟩) 1 ⟨7222⟩ 276028

def event276111 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7320⟩⟩) (.sum [.predecessor 0 276109 .coefficient, .predecessor 1 276110 .coefficient])

def exact276112RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩]

theorem exact276112RawTermsValid :
    exact276112RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event276112 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7320⟩⟩) exact276112RawTerms .large 276111 .exactZero (none)

def event276113 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7321⟩⟩) 0 ⟨7320⟩ 276112

def event276114 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7321⟩⟩) 1 ⟨7224⟩ 276025

def event276115 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7321⟩⟩) (.sum [.predecessor 0 276113 .coefficient, .predecessor 1 276114 .coefficient])

def exact276116RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7224⟩⟩]⟩, (1)⟩]

theorem exact276116RawTermsValid :
    exact276116RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event276116 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7321⟩⟩) exact276116RawTerms .large 276115 .exactZero (none)

def event276117 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7322⟩⟩) 0 ⟨7321⟩ 276116

def event276118 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7322⟩⟩) 1 ⟨7226⟩ 276022

def event276119 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7322⟩⟩) (.sum [.predecessor 0 276117 .coefficient, .predecessor 1 276118 .coefficient])

def exact276120RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7224⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7226⟩⟩]⟩, (1)⟩]

theorem exact276120RawTermsValid :
    exact276120RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event276120 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7322⟩⟩) exact276120RawTerms .large 276119 .exactZero (none)

def event276121 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7323⟩⟩) 0 ⟨7322⟩ 276120

def event276122 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7323⟩⟩) 1 ⟨7228⟩ 276019

def event276123 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7323⟩⟩) (.sum [.predecessor 0 276121 .coefficient, .predecessor 1 276122 .coefficient])

def exact276124RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7224⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7226⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7228⟩⟩]⟩, (1)⟩]

theorem exact276124RawTermsValid :
    exact276124RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event276124 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7323⟩⟩) exact276124RawTerms .large 276123 .exactZero (none)

def event276125 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7324⟩⟩) 0 ⟨7323⟩ 276124

def event276126 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7324⟩⟩) 1 ⟨7230⟩ 276016

def event276127 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7324⟩⟩) (.sum [.predecessor 0 276125 .coefficient, .predecessor 1 276126 .coefficient])

def exact276128RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7224⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7226⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7228⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7230⟩⟩]⟩, (1)⟩]

theorem exact276128RawTermsValid :
    exact276128RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event276128 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7324⟩⟩) exact276128RawTerms .large 276127 .exactZero (none)

def event276129 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7325⟩⟩) 0 ⟨7324⟩ 276128

def event276130 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7325⟩⟩) 1 ⟨7232⟩ 276013

def event276131 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7325⟩⟩) (.sum [.predecessor 0 276129 .coefficient, .predecessor 1 276130 .coefficient])

def exact276132RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7224⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7226⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7228⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7230⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7232⟩⟩]⟩, (1)⟩]

theorem exact276132RawTermsValid :
    exact276132RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event276132 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7325⟩⟩) exact276132RawTerms .large 276131 .exactZero (none)

def event276133 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69058⟩⟩) 0 ⟨7325⟩ 276132

def event276134 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69058⟩⟩) 1 ⟨69057⟩ 276010

def event276135 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69058⟩⟩) (.sum [.predecessor 0 276133 .coefficient, .predecessor 1 276134 .coefficient])

def exact276136RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7224⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7226⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7228⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7230⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7232⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15903⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨18709⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨21929⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨26512⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨29192⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨31949⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨34856⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨37536⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨40212⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨42892⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨45576⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨48256⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨51004⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨53984⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨56964⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨59944⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨62924⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨66019⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact276136RawTermsValid :
    exact276136RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event276136 : Event := .resultExact (⟨.program ⟨257⟩, ⟨69058⟩⟩) exact276136RawTerms .large 276135 .exactZero (none)

def event276137 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70980⟩⟩) 0 ⟨69058⟩ 276136

def event276138 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70980⟩⟩) 1 ⟨70979⟩ 275977

def event276139 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70980⟩⟩) (.product (.predecessor 0 276137 .coefficient) (.predecessor 1 276138 .coefficient) (⟨false, false, none, none, none⟩))

def event276140 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70980⟩⟩, .operator (⟨276136, 17⟩, ⟨275977, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7232⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (1)⟩)

def event276141 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70980⟩⟩, .operator (⟨276136, 16⟩, ⟨275977, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (1)⟩)

def event276142 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70980⟩⟩, .operator (⟨276136, 15⟩, ⟨275977, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (1)⟩)

def event276143 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70980⟩⟩, .operator (⟨276136, 14⟩, ⟨275977, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (1)⟩)

def event276144 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70980⟩⟩, .operator (⟨276136, 13⟩, ⟨275977, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (1)⟩)

def event276145 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70980⟩⟩, .operator (⟨276136, 12⟩, ⟨275977, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7222⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (1)⟩)

def event276146 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70980⟩⟩, .operator (⟨276136, 11⟩, ⟨275977, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7220⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (1)⟩)

def event276147 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70980⟩⟩, .operator (⟨276136, 10⟩, ⟨275977, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7218⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (1)⟩)

def event276148 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70980⟩⟩, .operator (⟨276136, 9⟩, ⟨275977, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7216⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (1)⟩)

def event276149 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70980⟩⟩, .operator (⟨276136, 8⟩, ⟨275977, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7214⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (1)⟩)

def event276150 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70980⟩⟩, .operator (⟨276136, 7⟩, ⟨275977, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (1)⟩)

def event276151 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70980⟩⟩, .operator (⟨276136, 6⟩, ⟨275977, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (1)⟩)

def event276152 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70980⟩⟩, .operator (⟨276136, 5⟩, ⟨275977, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (1)⟩)

def event276153 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70980⟩⟩, .operator (⟨276136, 4⟩, ⟨275977, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (1)⟩)

def event276154 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70980⟩⟩, .operator (⟨276136, 3⟩, ⟨275977, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (1)⟩)

def event276155 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70980⟩⟩, .operator (⟨276136, 2⟩, ⟨275977, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (1)⟩)

def event276156 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70980⟩⟩, .operator (⟨276136, 1⟩, ⟨275977, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (1)⟩)

def event276157 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70980⟩⟩, .operator (⟨276136, 0⟩, ⟨275977, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (1)⟩)

def event276158 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70980⟩⟩, .operator (⟨276136, 29⟩, ⟨275977, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨48256⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (-1)⟩)

def event276159 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨70980⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨48256⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨70979⟩⟩) ⟨68780⟩ 275974)

def event276160 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70980⟩⟩, .relation 276159 0, ⟨[⟨.program ⟨257⟩, ⟨48256⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (-1)⟩)

def event276161 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70980⟩⟩, .operator (⟨276136, 28⟩, ⟨275977, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨45576⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (-1)⟩)

def event276162 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨70980⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨45576⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨70979⟩⟩) ⟨68780⟩ 275974)

def event276163 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70980⟩⟩, .relation 276162 0, ⟨[⟨.program ⟨257⟩, ⟨45576⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (-1)⟩)

def event276164 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70980⟩⟩, .operator (⟨276136, 27⟩, ⟨275977, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨42892⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (-1)⟩)

def event276165 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨70980⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨42892⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨70979⟩⟩) ⟨68780⟩ 275974)

def event276166 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70980⟩⟩, .relation 276165 0, ⟨[⟨.program ⟨257⟩, ⟨42892⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (-1)⟩)

def event276167 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70980⟩⟩, .operator (⟨276136, 26⟩, ⟨275977, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨40212⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (-1)⟩)

def event276168 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨70980⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨40212⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨70979⟩⟩) ⟨68780⟩ 275974)

def event276169 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70980⟩⟩, .relation 276168 0, ⟨[⟨.program ⟨257⟩, ⟨40212⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (-1)⟩)

def event276170 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70980⟩⟩, .operator (⟨276136, 25⟩, ⟨275977, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨37536⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (-1)⟩)

def event276171 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨70980⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨37536⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨70979⟩⟩) ⟨68780⟩ 275974)

def event276172 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70980⟩⟩, .relation 276171 0, ⟨[⟨.program ⟨257⟩, ⟨37536⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (-1)⟩)

def event276173 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70980⟩⟩, .operator (⟨276136, 24⟩, ⟨275977, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨34856⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (-1)⟩)

def event276174 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨70980⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨34856⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨70979⟩⟩) ⟨68780⟩ 275974)

def event276175 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70980⟩⟩, .relation 276174 0, ⟨[⟨.program ⟨257⟩, ⟨34856⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (-1)⟩)

def event276176 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70980⟩⟩, .operator (⟨276136, 22⟩, ⟨275977, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨29192⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (-1)⟩)

def event276177 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨70980⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨29192⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨70979⟩⟩) ⟨68780⟩ 275974)

def event276178 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70980⟩⟩, .relation 276177 0, ⟨[⟨.program ⟨257⟩, ⟨29192⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (-1)⟩)

def event276179 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70980⟩⟩, .operator (⟨276136, 21⟩, ⟨275977, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨26512⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (-1)⟩)

def event276180 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨70980⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨26512⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨70979⟩⟩) ⟨68780⟩ 275974)

def event276181 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70980⟩⟩, .relation 276180 0, ⟨[⟨.program ⟨257⟩, ⟨26512⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (-1)⟩)

def event276182 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70980⟩⟩, .operator (⟨276136, 35⟩, ⟨275977, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨66019⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (-1)⟩)

def event276183 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨70980⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨66019⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨70979⟩⟩) ⟨68780⟩ 275974)

def event276184 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70980⟩⟩, .relation 276183 0, ⟨[⟨.program ⟨257⟩, ⟨66019⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (-1)⟩)

def event276185 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70980⟩⟩, .operator (⟨276136, 34⟩, ⟨275977, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨62924⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (-1)⟩)

def event276186 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨70980⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨62924⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨70979⟩⟩) ⟨68780⟩ 275974)

def event276187 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70980⟩⟩, .relation 276186 0, ⟨[⟨.program ⟨257⟩, ⟨62924⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (-1)⟩)

def event276188 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70980⟩⟩, .operator (⟨276136, 33⟩, ⟨275977, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨59944⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (-1)⟩)

def event276189 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨70980⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨59944⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨70979⟩⟩) ⟨68780⟩ 275974)

def event276190 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70980⟩⟩, .relation 276189 0, ⟨[⟨.program ⟨257⟩, ⟨59944⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (-1)⟩)

def event276191 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70980⟩⟩, .operator (⟨276136, 32⟩, ⟨275977, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨56964⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (-1)⟩)

def event276192 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨70980⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨56964⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨70979⟩⟩) ⟨68780⟩ 275974)

def event276193 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70980⟩⟩, .relation 276192 0, ⟨[⟨.program ⟨257⟩, ⟨56964⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (-1)⟩)

def event276194 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70980⟩⟩, .operator (⟨276136, 31⟩, ⟨275977, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨53984⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (-1)⟩)

def event276195 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨70980⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨53984⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨70979⟩⟩) ⟨68780⟩ 275974)

def event276196 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70980⟩⟩, .relation 276195 0, ⟨[⟨.program ⟨257⟩, ⟨53984⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (-1)⟩)

def event276197 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70980⟩⟩, .operator (⟨276136, 30⟩, ⟨275977, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨51004⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (-1)⟩)

def event276198 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨70980⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨51004⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨70979⟩⟩) ⟨68780⟩ 275974)

def event276199 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70980⟩⟩, .relation 276198 0, ⟨[⟨.program ⟨257⟩, ⟨51004⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (-1)⟩)

def event276200 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70980⟩⟩, .operator (⟨276136, 23⟩, ⟨275977, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨31949⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (-1)⟩)

def event276201 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨70980⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨31949⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨70979⟩⟩) ⟨68780⟩ 275974)

def event276202 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70980⟩⟩, .relation 276201 0, ⟨[⟨.program ⟨257⟩, ⟨31949⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (-1)⟩)

def event276203 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70980⟩⟩, .operator (⟨276136, 20⟩, ⟨275977, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨21929⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (-1)⟩)

def event276204 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨70980⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨21929⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨70979⟩⟩) ⟨68780⟩ 275974)

def event276205 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70980⟩⟩, .relation 276204 0, ⟨[⟨.program ⟨257⟩, ⟨21929⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (-1)⟩)

def event276206 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70980⟩⟩, .operator (⟨276136, 19⟩, ⟨275977, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨18709⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (-1)⟩)

def event276207 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨70980⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨18709⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨70979⟩⟩) ⟨68780⟩ 275974)

def event276208 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70980⟩⟩, .relation 276207 0, ⟨[⟨.program ⟨257⟩, ⟨18709⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (-1)⟩)

def event276209 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70980⟩⟩, .operator (⟨276136, 18⟩, ⟨275977, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨15903⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (-1)⟩)

def event276210 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨70980⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨15903⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨70979⟩⟩) ⟨68780⟩ 275974)

def event276211 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70980⟩⟩, .relation 276210 0, ⟨[⟨.program ⟨257⟩, ⟨15903⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (-1)⟩)

def exact276212RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7214⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7216⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7218⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7220⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7222⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7232⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15903⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨18709⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨21929⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨26512⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨29192⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨31949⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨34856⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨37536⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨40212⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨42892⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨45576⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨48256⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨51004⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨53984⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨56964⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨59944⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨62924⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨66019⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (-1)⟩]

theorem exact276212RawTermsValid :
    exact276212RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event276212 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70980⟩⟩) exact276212RawTerms .large 276139 .exactZero (none)

def event276213 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67300⟩⟩) 0 ⟨66029⟩ 275966

def event276214 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67300⟩⟩) (.authority (.programFamilyFact))

def exact276215RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨67300⟩⟩], []⟩, (1)⟩]

theorem exact276215RawTermsValid :
    exact276215RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event276215 : Event := .resultExact (⟨.program ⟨257⟩, ⟨67300⟩⟩) exact276215RawTerms (.finite 18) 276214 .exactZero (none)

def event276216 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67302⟩⟩) 0 ⟨6908⟩ 275988

def event276217 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67302⟩⟩) 1 ⟨67300⟩ 276215

def event276218 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67302⟩⟩) (.product (.predecessor 0 276216 .coefficient) (.predecessor 1 276217 .coefficient) (⟨false, true, none, none, some 1⟩))

def event276219 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67302⟩⟩, .operator (⟨275988, 0⟩, ⟨276215, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨67300⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact276220RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨67300⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact276220RawTermsValid :
    exact276220RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event276220 : Event := .resultExact (⟨.program ⟨257⟩, ⟨67302⟩⟩) exact276220RawTerms .large 276218 .exactZero (none)

def event276221 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7233⟩⟩) 0 ⟨7177⟩ 275970

def event276222 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7233⟩⟩) (.authority (.operator))

def exact276223RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7233⟩⟩]⟩, (1)⟩]

theorem exact276223RawTermsValid :
    exact276223RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event276223 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7233⟩⟩) exact276223RawTerms .large 276222 .exactZero (none)

def eventLeaf17248 : Array AnnotatedEvent := #[
  { event := event275968
    frameStart := 275461 },
  { event := event275969
    frameStart := 275461 },
  { event := event275970
    frameStart := 275461 },
  { event := event275971
    frameStart := 275461 },
  { event := event275972
    frameStart := 275461 },
  { event := event275973
    frameStart := 275461 },
  { event := event275974
    frameStart := 275461 },
  { event := event275975
    frameStart := 275461 },
  { event := event275976
    frameStart := 275461 },
  { event := event275977
    frameStart := 275461 },
  { event := event275978
    frameStart := 275461 },
  { event := event275979
    frameStart := 275461 },
  { event := event275980
    frameStart := 275461 },
  { event := event275981
    frameStart := 275461 },
  { event := event275982
    frameStart := 275461 },
  { event := event275983
    frameStart := 275461 }
]

def eventLeaf17249 : Array AnnotatedEvent := #[
  { event := event275984
    frameStart := 275461 },
  { event := event275985
    frameStart := 275461 },
  { event := event275986
    frameStart := 275461 },
  { event := event275987
    frameStart := 275461 },
  { event := event275988
    frameStart := 275461 },
  { event := event275989
    frameStart := 275461 },
  { event := event275990
    frameStart := 275461 },
  { event := event275991
    frameStart := 275461 },
  { event := event275992
    frameStart := 275461 },
  { event := event275993
    frameStart := 275461 },
  { event := event275994
    frameStart := 275461 },
  { event := event275995
    frameStart := 275461 },
  { event := event275996
    frameStart := 275461 },
  { event := event275997
    frameStart := 275461 },
  { event := event275998
    frameStart := 275461 },
  { event := event275999
    frameStart := 275461 }
]

def eventLeaf17250 : Array AnnotatedEvent := #[
  { event := event276000
    frameStart := 275461 },
  { event := event276001
    frameStart := 275461 },
  { event := event276002
    frameStart := 275461 },
  { event := event276003
    frameStart := 275461 },
  { event := event276004
    frameStart := 275461 },
  { event := event276005
    frameStart := 275461 },
  { event := event276006
    frameStart := 275461 },
  { event := event276007
    frameStart := 275461 },
  { event := event276008
    frameStart := 275461 },
  { event := event276009
    frameStart := 275461 },
  { event := event276010
    frameStart := 275461 },
  { event := event276011
    frameStart := 275461 },
  { event := event276012
    frameStart := 275461 },
  { event := event276013
    frameStart := 275461 },
  { event := event276014
    frameStart := 275461 },
  { event := event276015
    frameStart := 275461 }
]

def eventLeaf17251 : Array AnnotatedEvent := #[
  { event := event276016
    frameStart := 275461 },
  { event := event276017
    frameStart := 275461 },
  { event := event276018
    frameStart := 275461 },
  { event := event276019
    frameStart := 275461 },
  { event := event276020
    frameStart := 275461 },
  { event := event276021
    frameStart := 275461 },
  { event := event276022
    frameStart := 275461 },
  { event := event276023
    frameStart := 275461 },
  { event := event276024
    frameStart := 275461 },
  { event := event276025
    frameStart := 275461 },
  { event := event276026
    frameStart := 275461 },
  { event := event276027
    frameStart := 275461 },
  { event := event276028
    frameStart := 275461 },
  { event := event276029
    frameStart := 275461 },
  { event := event276030
    frameStart := 275461 },
  { event := event276031
    frameStart := 275461 }
]

def eventLeaf17252 : Array AnnotatedEvent := #[
  { event := event276032
    frameStart := 275461 },
  { event := event276033
    frameStart := 275461 },
  { event := event276034
    frameStart := 275461 },
  { event := event276035
    frameStart := 275461 },
  { event := event276036
    frameStart := 275461 },
  { event := event276037
    frameStart := 275461 },
  { event := event276038
    frameStart := 275461 },
  { event := event276039
    frameStart := 275461 },
  { event := event276040
    frameStart := 275461 },
  { event := event276041
    frameStart := 275461 },
  { event := event276042
    frameStart := 275461 },
  { event := event276043
    frameStart := 275461 },
  { event := event276044
    frameStart := 275461 },
  { event := event276045
    frameStart := 275461 },
  { event := event276046
    frameStart := 275461 },
  { event := event276047
    frameStart := 275461 }
]

def eventLeaf17253 : Array AnnotatedEvent := #[
  { event := event276048
    frameStart := 275461 },
  { event := event276049
    frameStart := 275461 },
  { event := event276050
    frameStart := 275461 },
  { event := event276051
    frameStart := 275461 },
  { event := event276052
    frameStart := 275461 },
  { event := event276053
    frameStart := 275461 },
  { event := event276054
    frameStart := 275461 },
  { event := event276055
    frameStart := 275461 },
  { event := event276056
    frameStart := 275461 },
  { event := event276057
    frameStart := 275461 },
  { event := event276058
    frameStart := 275461 },
  { event := event276059
    frameStart := 275461 },
  { event := event276060
    frameStart := 275461 },
  { event := event276061
    frameStart := 275461 },
  { event := event276062
    frameStart := 275461 },
  { event := event276063
    frameStart := 275461 }
]

def eventLeaf17254 : Array AnnotatedEvent := #[
  { event := event276064
    frameStart := 275461 },
  { event := event276065
    frameStart := 275461 },
  { event := event276066
    frameStart := 275461 },
  { event := event276067
    frameStart := 275461 },
  { event := event276068
    frameStart := 275461 },
  { event := event276069
    frameStart := 275461 },
  { event := event276070
    frameStart := 275461 },
  { event := event276071
    frameStart := 275461 },
  { event := event276072
    frameStart := 275461 },
  { event := event276073
    frameStart := 275461 },
  { event := event276074
    frameStart := 275461 },
  { event := event276075
    frameStart := 275461 },
  { event := event276076
    frameStart := 275461 },
  { event := event276077
    frameStart := 275461 },
  { event := event276078
    frameStart := 275461 },
  { event := event276079
    frameStart := 275461 }
]

def eventLeaf17255 : Array AnnotatedEvent := #[
  { event := event276080
    frameStart := 275461 },
  { event := event276081
    frameStart := 275461 },
  { event := event276082
    frameStart := 275461 },
  { event := event276083
    frameStart := 275461 },
  { event := event276084
    frameStart := 275461 },
  { event := event276085
    frameStart := 275461 },
  { event := event276086
    frameStart := 275461 },
  { event := event276087
    frameStart := 275461 },
  { event := event276088
    frameStart := 275461 },
  { event := event276089
    frameStart := 275461 },
  { event := event276090
    frameStart := 275461 },
  { event := event276091
    frameStart := 275461 },
  { event := event276092
    frameStart := 275461 },
  { event := event276093
    frameStart := 275461 },
  { event := event276094
    frameStart := 275461 },
  { event := event276095
    frameStart := 275461 }
]

def eventLeaf17256 : Array AnnotatedEvent := #[
  { event := event276096
    frameStart := 275461 },
  { event := event276097
    frameStart := 275461 },
  { event := event276098
    frameStart := 275461 },
  { event := event276099
    frameStart := 275461 },
  { event := event276100
    frameStart := 275461 },
  { event := event276101
    frameStart := 275461 },
  { event := event276102
    frameStart := 275461 },
  { event := event276103
    frameStart := 275461 },
  { event := event276104
    frameStart := 275461 },
  { event := event276105
    frameStart := 275461 },
  { event := event276106
    frameStart := 275461 },
  { event := event276107
    frameStart := 275461 },
  { event := event276108
    frameStart := 275461 },
  { event := event276109
    frameStart := 275461 },
  { event := event276110
    frameStart := 275461 },
  { event := event276111
    frameStart := 275461 }
]

def eventLeaf17257 : Array AnnotatedEvent := #[
  { event := event276112
    frameStart := 275461 },
  { event := event276113
    frameStart := 275461 },
  { event := event276114
    frameStart := 275461 },
  { event := event276115
    frameStart := 275461 },
  { event := event276116
    frameStart := 275461 },
  { event := event276117
    frameStart := 275461 },
  { event := event276118
    frameStart := 275461 },
  { event := event276119
    frameStart := 275461 },
  { event := event276120
    frameStart := 275461 },
  { event := event276121
    frameStart := 275461 },
  { event := event276122
    frameStart := 275461 },
  { event := event276123
    frameStart := 275461 },
  { event := event276124
    frameStart := 275461 },
  { event := event276125
    frameStart := 275461 },
  { event := event276126
    frameStart := 275461 },
  { event := event276127
    frameStart := 275461 }
]

def eventLeaf17258 : Array AnnotatedEvent := #[
  { event := event276128
    frameStart := 275461 },
  { event := event276129
    frameStart := 275461 },
  { event := event276130
    frameStart := 275461 },
  { event := event276131
    frameStart := 275461 },
  { event := event276132
    frameStart := 275461 },
  { event := event276133
    frameStart := 275461 },
  { event := event276134
    frameStart := 275461 },
  { event := event276135
    frameStart := 275461 },
  { event := event276136
    frameStart := 275461 },
  { event := event276137
    frameStart := 275461 },
  { event := event276138
    frameStart := 275461 },
  { event := event276139
    frameStart := 275461 },
  { event := event276140
    frameStart := 275461 },
  { event := event276141
    frameStart := 275461 },
  { event := event276142
    frameStart := 275461 },
  { event := event276143
    frameStart := 275461 }
]

def eventLeaf17259 : Array AnnotatedEvent := #[
  { event := event276144
    frameStart := 275461 },
  { event := event276145
    frameStart := 275461 },
  { event := event276146
    frameStart := 275461 },
  { event := event276147
    frameStart := 275461 },
  { event := event276148
    frameStart := 275461 },
  { event := event276149
    frameStart := 275461 },
  { event := event276150
    frameStart := 275461 },
  { event := event276151
    frameStart := 275461 },
  { event := event276152
    frameStart := 275461 },
  { event := event276153
    frameStart := 275461 },
  { event := event276154
    frameStart := 275461 },
  { event := event276155
    frameStart := 275461 },
  { event := event276156
    frameStart := 275461 },
  { event := event276157
    frameStart := 275461 },
  { event := event276158
    frameStart := 275461 },
  { event := event276159
    frameStart := 275461 }
]

def eventLeaf17260 : Array AnnotatedEvent := #[
  { event := event276160
    frameStart := 275461 },
  { event := event276161
    frameStart := 275461 },
  { event := event276162
    frameStart := 275461 },
  { event := event276163
    frameStart := 275461 },
  { event := event276164
    frameStart := 275461 },
  { event := event276165
    frameStart := 275461 },
  { event := event276166
    frameStart := 275461 },
  { event := event276167
    frameStart := 275461 },
  { event := event276168
    frameStart := 275461 },
  { event := event276169
    frameStart := 275461 },
  { event := event276170
    frameStart := 275461 },
  { event := event276171
    frameStart := 275461 },
  { event := event276172
    frameStart := 275461 },
  { event := event276173
    frameStart := 275461 },
  { event := event276174
    frameStart := 275461 },
  { event := event276175
    frameStart := 275461 }
]

def eventLeaf17261 : Array AnnotatedEvent := #[
  { event := event276176
    frameStart := 275461 },
  { event := event276177
    frameStart := 275461 },
  { event := event276178
    frameStart := 275461 },
  { event := event276179
    frameStart := 275461 },
  { event := event276180
    frameStart := 275461 },
  { event := event276181
    frameStart := 275461 },
  { event := event276182
    frameStart := 275461 },
  { event := event276183
    frameStart := 275461 },
  { event := event276184
    frameStart := 275461 },
  { event := event276185
    frameStart := 275461 },
  { event := event276186
    frameStart := 275461 },
  { event := event276187
    frameStart := 275461 },
  { event := event276188
    frameStart := 275461 },
  { event := event276189
    frameStart := 275461 },
  { event := event276190
    frameStart := 275461 },
  { event := event276191
    frameStart := 275461 }
]

def eventLeaf17262 : Array AnnotatedEvent := #[
  { event := event276192
    frameStart := 275461 },
  { event := event276193
    frameStart := 275461 },
  { event := event276194
    frameStart := 275461 },
  { event := event276195
    frameStart := 275461 },
  { event := event276196
    frameStart := 275461 },
  { event := event276197
    frameStart := 275461 },
  { event := event276198
    frameStart := 275461 },
  { event := event276199
    frameStart := 275461 },
  { event := event276200
    frameStart := 275461 },
  { event := event276201
    frameStart := 275461 },
  { event := event276202
    frameStart := 275461 },
  { event := event276203
    frameStart := 275461 },
  { event := event276204
    frameStart := 275461 },
  { event := event276205
    frameStart := 275461 },
  { event := event276206
    frameStart := 275461 },
  { event := event276207
    frameStart := 275461 }
]

def eventLeaf17263 : Array AnnotatedEvent := #[
  { event := event276208
    frameStart := 275461 },
  { event := event276209
    frameStart := 275461 },
  { event := event276210
    frameStart := 275461 },
  { event := event276211
    frameStart := 275461 },
  { event := event276212
    frameStart := 275461 },
  { event := event276213
    frameStart := 275461 },
  { event := event276214
    frameStart := 275461 },
  { event := event276215
    frameStart := 275461 },
  { event := event276216
    frameStart := 275461 },
  { event := event276217
    frameStart := 275461 },
  { event := event276218
    frameStart := 275461 },
  { event := event276219
    frameStart := 275461 },
  { event := event276220
    frameStart := 275461 },
  { event := event276221
    frameStart := 275461 },
  { event := event276222
    frameStart := 275461 },
  { event := event276223
    frameStart := 275461 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events1078
