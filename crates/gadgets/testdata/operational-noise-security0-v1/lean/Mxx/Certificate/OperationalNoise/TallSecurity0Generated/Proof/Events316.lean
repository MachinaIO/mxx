import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events316

open Mxx.Certificate.OperationalNoise
open CertificateABI

def exact80896RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨12958⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact80896RawTermsValid :
    exact80896RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80896 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12961⟩⟩) exact80896RawTerms .large 80894 .exactZero (none)

def event80897 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7244⟩⟩) 0 ⟨5539⟩ 79790

def event80898 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7244⟩⟩) 1 ⟨6788⟩ 7474

def event80899 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7244⟩⟩) (.product (.predecessor 0 80897 .coefficient) (.predecessor 1 80898 .coefficient) (⟨false, false, none, none, none⟩))

def event80900 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7244⟩⟩, .operator (⟨79790, 0⟩, ⟨7474, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6788⟩⟩]⟩, (1)⟩)

def exact80901RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6788⟩⟩]⟩, (1)⟩]

theorem exact80901RawTermsValid :
    exact80901RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80901 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7244⟩⟩) exact80901RawTerms .large 80899 .exactZero (none)

def event80902 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12962⟩⟩) 0 ⟨7244⟩ 80901

def event80903 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12962⟩⟩) 1 ⟨12961⟩ 80896

def event80904 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12962⟩⟩) (.sum [.predecessor 0 80902 .coefficient, .predecessor 1 80903 .coefficient])

def exact80905RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6788⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨12958⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact80905RawTermsValid :
    exact80905RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80905 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12962⟩⟩) exact80905RawTerms .large 80904 .exactZero (none)

def event80906 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12963⟩⟩) 0 ⟨12962⟩ 80905

def event80907 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12963⟩⟩) 1 ⟨102⟩ 7466

def event80908 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12963⟩⟩) (.sum [.predecessor 0 80906 .coefficient, .predecessor 1 80907 .coefficient])

def event80909 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12963⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨102⟩⟩]⟩) [⟨.result 7466 .coefficient, false, none⟩])

def event80910 : Event := .survivorFold (1) 80909

def exact80911RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6788⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨12958⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact80911RawTermsValid :
    exact80911RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80911 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12963⟩⟩) exact80911RawTerms .large 80908 (.finite 26) (some (80909))

def event80912 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12964⟩⟩) 0 ⟨12963⟩ 80911

def event80913 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12964⟩⟩) 1 ⟨10135⟩ 3877

def event80914 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12964⟩⟩) (.product (.predecessor 0 80912 .coefficient) (.predecessor 1 80913 .coefficient) (⟨false, true, none, none, some 1⟩))

def event80915 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12964⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨10135⟩⟩], []⟩) [⟨.result 3877 .coefficient, true, some 1⟩])

def event80916 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12964⟩⟩) (.product (.result 80911 .summary) (.transfer 80915) (⟨false, false, none, none, none⟩))

def event80917 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨12964⟩⟩, .operator (⟨80911, 1⟩, ⟨3877, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨10135⟩⟩, ⟨.program ⟨214⟩, ⟨12958⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def event80918 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨12964⟩⟩, .operator (⟨80911, 0⟩, ⟨3877, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨10135⟩⟩], [⟨.program ⟨214⟩, ⟨6788⟩⟩]⟩, (1)⟩)

def exact80919RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨10135⟩⟩], [⟨.program ⟨214⟩, ⟨6788⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨10135⟩⟩, ⟨.program ⟨214⟩, ⟨12958⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact80919RawTermsValid :
    exact80919RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80919 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12964⟩⟩) exact80919RawTerms .large 80914 (.finite 43264) (some (80916))

def event80920 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10136⟩⟩) 0 ⟨10135⟩ 3877

def event80921 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10136⟩⟩) 1 ⟨6567⟩ 79920

def event80922 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10136⟩⟩) (.tensor (.predecessor 0 80920 .coefficient) (.predecessor 1 80921 .coefficient) true false)

def event80923 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨10136⟩⟩, .operator (⟨3877, 0⟩, ⟨79920, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨10135⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact80924RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨10135⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact80924RawTermsValid :
    exact80924RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80924 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10136⟩⟩) exact80924RawTerms .large 80922 .exactZero (none)

def event80925 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7224⟩⟩) 0 ⟨5539⟩ 79790

def event80926 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7224⟩⟩) 1 ⟨6768⟩ 7515

def event80927 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7224⟩⟩) (.product (.predecessor 0 80925 .coefficient) (.predecessor 1 80926 .coefficient) (⟨false, false, none, none, none⟩))

def event80928 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7224⟩⟩, .operator (⟨79790, 0⟩, ⟨7515, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6768⟩⟩]⟩, (1)⟩)

def exact80929RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6768⟩⟩]⟩, (1)⟩]

theorem exact80929RawTermsValid :
    exact80929RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80929 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7224⟩⟩) exact80929RawTerms .large 80927 .exactZero (none)

def event80930 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10137⟩⟩) 0 ⟨7224⟩ 80929

def event80931 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10137⟩⟩) 1 ⟨10136⟩ 80924

def event80932 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10137⟩⟩) (.sum [.predecessor 0 80930 .coefficient, .predecessor 1 80931 .coefficient])

def exact80933RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6768⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨10135⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact80933RawTermsValid :
    exact80933RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80933 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10137⟩⟩) exact80933RawTerms .large 80932 .exactZero (none)

def event80934 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10138⟩⟩) 0 ⟨10137⟩ 80933

def event80935 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10138⟩⟩) 1 ⟨82⟩ 7507

def event80936 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10138⟩⟩) (.sum [.predecessor 0 80934 .coefficient, .predecessor 1 80935 .coefficient])

def event80937 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10138⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨82⟩⟩]⟩) [⟨.result 7507 .coefficient, false, none⟩])

def event80938 : Event := .survivorFold (1) 80937

def exact80939RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6768⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨10135⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact80939RawTermsValid :
    exact80939RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80939 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10138⟩⟩) exact80939RawTerms .large 80936 (.finite 26) (some (80937))

def event80940 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10139⟩⟩) 0 ⟨10138⟩ 80939

def event80941 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10139⟩⟩) 1 ⟨7877⟩ 7504

def event80942 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10139⟩⟩) (.product (.predecessor 0 80940 .coefficient) (.predecessor 1 80941 .coefficient) (⟨false, false, none, none, none⟩))

def event80943 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10139⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨7876⟩⟩]⟩) [⟨.result 7500 .coefficient, false, none⟩])

def event80944 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10139⟩⟩) (.product (.result 80939 .summary) (.transfer 80943) (⟨false, false, none, none, none⟩))

def event80945 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨10139⟩⟩, .operator (⟨80939, 1⟩, ⟨7504, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨10135⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7876⟩⟩]⟩, (-1)⟩)

def event80946 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨10139⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨10135⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7876⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨7876⟩⟩) ⟨6788⟩ 7474)

def event80947 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨10139⟩⟩, .relation 80946 0, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨10135⟩⟩], [⟨.program ⟨214⟩, ⟨6788⟩⟩]⟩, (-1)⟩)

def event80948 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨10139⟩⟩, .operator (⟨80939, 0⟩, ⟨7504, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6768⟩⟩, ⟨.program ⟨214⟩, ⟨7876⟩⟩]⟩, (1)⟩)

def exact80949RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6768⟩⟩, ⟨.program ⟨214⟩, ⟨7876⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨10135⟩⟩], [⟨.program ⟨214⟩, ⟨6788⟩⟩]⟩, (-1)⟩]

theorem exact80949RawTermsValid :
    exact80949RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80949 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10139⟩⟩) exact80949RawTerms .large 80942 (.finite 95420416) (some (80944))

def event80950 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12965⟩⟩) 0 ⟨10139⟩ 80949

def event80951 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12965⟩⟩) 1 ⟨12964⟩ 80919

def event80952 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12965⟩⟩) (.sum [.predecessor 0 80950 .coefficient, .predecessor 1 80951 .coefficient])

def event80953 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨12965⟩⟩, .operator (⟨80949, 1⟩, ⟨80919, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨10135⟩⟩], [⟨.program ⟨214⟩, ⟨6788⟩⟩]⟩, (1)⟩)

def event80954 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12965⟩⟩) (.sum [.result 80949 .summary, .result 80919 .summary])

def exact80955RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6768⟩⟩, ⟨.program ⟨214⟩, ⟨7876⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨10135⟩⟩, ⟨.program ⟨214⟩, ⟨12958⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact80955RawTermsValid :
    exact80955RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80955 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12965⟩⟩) exact80955RawTerms .large 80952 (.finite 95463680) (some (80954))

def event80956 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25605⟩⟩) 0 ⟨12965⟩ 80955

def event80957 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25605⟩⟩) 1 ⟨25604⟩ 80891

def event80958 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25605⟩⟩) (.product (.predecessor 0 80956 .coefficient) (.predecessor 1 80957 .coefficient) (⟨false, false, none, none, none⟩))

def event80959 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25605⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨25604⟩⟩]⟩) [⟨.result 80891 .coefficient, false, none⟩])

def event80960 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25605⟩⟩) (.product (.result 80955 .summary) (.transfer 80959) (⟨false, false, none, none, none⟩))

def event80961 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25605⟩⟩, .operator (⟨80955, 1⟩, ⟨80891, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨10135⟩⟩, ⟨.program ⟨214⟩, ⟨12958⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25604⟩⟩]⟩, (-1)⟩)

def event80962 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨25605⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨10135⟩⟩, ⟨.program ⟨214⟩, ⟨12958⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25604⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨25604⟩⟩) ⟨23332⟩ 80888)

def event80963 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25605⟩⟩, .relation 80962 0, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨10135⟩⟩, ⟨.program ⟨214⟩, ⟨12958⟩⟩], [⟨.program ⟨214⟩, ⟨23332⟩⟩]⟩, (-1)⟩)

def event80964 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25605⟩⟩, .operator (⟨80955, 0⟩, ⟨80891, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6768⟩⟩, ⟨.program ⟨214⟩, ⟨7876⟩⟩, ⟨.program ⟨214⟩, ⟨25604⟩⟩]⟩, (1)⟩)

def exact80965RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6768⟩⟩, ⟨.program ⟨214⟩, ⟨7876⟩⟩, ⟨.program ⟨214⟩, ⟨25604⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨10135⟩⟩, ⟨.program ⟨214⟩, ⟨12958⟩⟩], [⟨.program ⟨214⟩, ⟨23332⟩⟩]⟩, (-1)⟩]

theorem exact80965RawTermsValid :
    exact80965RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80965 : Event := .resultExact (⟨.program ⟨214⟩, ⟨25605⟩⟩) exact80965RawTerms .large 80958 (.finite 350353233018880) (some (80960))

def event80966 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20104⟩⟩) 0 ⟨12960⟩ 3885

def event80967 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20104⟩⟩) (.authority (.relationPreimageSource ⟨24⟩))

def exact80968RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨20104⟩⟩]⟩, (1)⟩]

theorem exact80968RawTermsValid :
    exact80968RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80968 : Event := .resultExact (⟨.program ⟨214⟩, ⟨20104⟩⟩) exact80968RawTerms (.finite 136065468) 80967 .exactZero (none)

def event80969 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20106⟩⟩) 0 ⟨20104⟩ 80968

def event80970 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20106⟩⟩) 1 ⟨2348⟩ 4

def event80971 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20106⟩⟩) (.scale (.predecessor 0 80969 .coefficient) (.value (.predecessor 1 80970 .coefficient)))

def exact80972RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨20104⟩⟩]⟩, (1)⟩]

theorem exact80972RawTermsValid :
    exact80972RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80972 : Event := .resultExact (⟨.program ⟨214⟩, ⟨20106⟩⟩) exact80972RawTerms (.finite 136065468) 80971 .exactZero (none)

def event80973 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20107⟩⟩) 0 ⟨5541⟩ 80012

def event80974 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20107⟩⟩) 1 ⟨20106⟩ 80972

def event80975 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20107⟩⟩) (.product (.predecessor 0 80973 .coefficient) (.predecessor 1 80974 .coefficient) (⟨false, false, none, none, none⟩))

def event80976 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20107⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨20104⟩⟩]⟩) [⟨.result 80968 .coefficient, false, none⟩])

def event80977 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20107⟩⟩) (.product (.result 80012 .summary) (.transfer 80976) (⟨false, false, none, none, none⟩))

def event80978 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20107⟩⟩, .operator (⟨80012, 0⟩, ⟨80972, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20104⟩⟩]⟩, (1)⟩)

def event80979 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨20105⟩⟩)

def event80980 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event80981 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event80982 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.authority (.operator))

def event80983 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.finite 1)

def event80984 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event80985 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event80986 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event80987 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event80988 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 80987

def event80989 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 80985

def event80990 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 80988 .coefficient) (.value (.predecessor 1 80989 .coefficient)))

def event80991 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event80992 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5505⟩⟩) 0 ⟨5503⟩ 80991

def event80993 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5505⟩⟩) 1 ⟨2348⟩ 80983

def event80994 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5505⟩⟩) (.sum [.predecessor 0 80992 .coefficient, .predecessor 1 80993 .coefficient])

def event80995 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5505⟩⟩) (.finite 218)

def event80996 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5536⟩⟩) 0 ⟨5505⟩ 80995

def event80997 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5536⟩⟩) 1 ⟨961⟩ 80981

def event80998 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5536⟩⟩) (.identity (.predecessor 1 80997 .coefficient))

def event80999 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5536⟩⟩) (.finite 224)

def event81000 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12958⟩⟩) 0 ⟨5536⟩ 80999

def event81001 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12958⟩⟩) (.authority (.programFamilyFact))

def exact81002RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨12958⟩⟩], []⟩, (1)⟩]

theorem exact81002RawTermsValid :
    exact81002RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event81002 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12958⟩⟩) exact81002RawTerms (.finite 52) 81001 .exactZero (none)

def event81003 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10135⟩⟩) 0 ⟨5536⟩ 80999

def event81004 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10135⟩⟩) (.authority (.programFamilyFact))

def exact81005RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨10135⟩⟩], []⟩, (1)⟩]

theorem exact81005RawTermsValid :
    exact81005RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event81005 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10135⟩⟩) exact81005RawTerms (.finite 52) 81004 .exactZero (none)

def event81006 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12959⟩⟩) 0 ⟨10135⟩ 81005

def event81007 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12959⟩⟩) 1 ⟨12958⟩ 81002

def event81008 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12959⟩⟩) (.product (.predecessor 0 81006 .coefficient) (.predecessor 1 81007 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event81009 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12959⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨10135⟩⟩, ⟨.program ⟨214⟩, ⟨12958⟩⟩], []⟩) [⟨.result 81005 .coefficient, true, some 1⟩, ⟨.result 81002 .coefficient, true, some 1⟩])

def event81010 : Event := .survivorFold (1) 81009

def exact81011RawTerms : List Term := []

theorem exact81011RawTermsValid :
    exact81011RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event81011 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12959⟩⟩) exact81011RawTerms (.finite 2704) 81008 (.finite 2704) (some (81009))

def event81012 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12960⟩⟩) 0 ⟨12959⟩ 81011

def event81013 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12960⟩⟩) (.identity (.predecessor 0 81012 .coefficient))

def event81014 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨12960⟩⟩) (.finite 2704)

def event81015 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20104⟩⟩) 0 ⟨12960⟩ 81014

def event81016 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20104⟩⟩) (.authority (.relationPreimageSource ⟨24⟩))

def exact81017RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨20104⟩⟩]⟩, (1)⟩]

theorem exact81017RawTermsValid :
    exact81017RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event81017 : Event := .resultExact (⟨.program ⟨214⟩, ⟨20104⟩⟩) exact81017RawTerms (.finite 136065468) 81016 .exactZero (none)

def event81018 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6⟩⟩) (.authority (.operator))

def exact81019RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩]⟩, (1)⟩]

theorem exact81019RawTermsValid :
    exact81019RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event81019 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6⟩⟩) exact81019RawTerms .large 81018 .exactZero (none)

def event81020 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20105⟩⟩) 0 ⟨6⟩ 81019

def event81021 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20105⟩⟩) 1 ⟨20104⟩ 81017

def event81022 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20105⟩⟩) (.product (.predecessor 0 81020 .coefficient) (.predecessor 1 81021 .coefficient) (⟨false, false, none, none, none⟩))

def event81023 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20105⟩⟩, .operator (⟨81019, 0⟩, ⟨81017, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20104⟩⟩]⟩, (1)⟩)

def exact81024RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20104⟩⟩]⟩, (1)⟩]

theorem exact81024RawTermsValid :
    exact81024RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event81024 : Event := .resultExact (⟨.program ⟨214⟩, ⟨20105⟩⟩) exact81024RawTerms .large 81022 .exactZero (none)

def event81025 : Event := .preFoldPolynomial 81024 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20104⟩⟩]⟩, (1)⟩] .exactZero none

def exact81026RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20104⟩⟩]⟩, (1)⟩]

def event81026 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨20105⟩⟩) 81025 exact81026RawTerms .large 81022 .exactZero (none)

def event81027 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨25608⟩⟩)

def event81028 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event81029 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event81030 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.authority (.operator))

def event81031 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.finite 1)

def event81032 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event81033 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event81034 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event81035 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event81036 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 81035

def event81037 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 81033

def event81038 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 81036 .coefficient) (.value (.predecessor 1 81037 .coefficient)))

def event81039 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event81040 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5505⟩⟩) 0 ⟨5503⟩ 81039

def event81041 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5505⟩⟩) 1 ⟨2348⟩ 81031

def event81042 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5505⟩⟩) (.sum [.predecessor 0 81040 .coefficient, .predecessor 1 81041 .coefficient])

def event81043 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5505⟩⟩) (.finite 218)

def event81044 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5536⟩⟩) 0 ⟨5505⟩ 81043

def event81045 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5536⟩⟩) 1 ⟨961⟩ 81029

def event81046 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5536⟩⟩) (.identity (.predecessor 1 81045 .coefficient))

def event81047 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5536⟩⟩) (.finite 224)

def event81048 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12958⟩⟩) 0 ⟨5536⟩ 81047

def event81049 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12958⟩⟩) (.authority (.programFamilyFact))

def exact81050RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨12958⟩⟩], []⟩, (1)⟩]

theorem exact81050RawTermsValid :
    exact81050RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event81050 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12958⟩⟩) exact81050RawTerms (.finite 52) 81049 .exactZero (none)

def event81051 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10135⟩⟩) 0 ⟨5536⟩ 81047

def event81052 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10135⟩⟩) (.authority (.programFamilyFact))

def exact81053RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨10135⟩⟩], []⟩, (1)⟩]

theorem exact81053RawTermsValid :
    exact81053RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event81053 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10135⟩⟩) exact81053RawTerms (.finite 52) 81052 .exactZero (none)

def event81054 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12959⟩⟩) 0 ⟨10135⟩ 81053

def event81055 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12959⟩⟩) 1 ⟨12958⟩ 81050

def event81056 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12959⟩⟩) (.product (.predecessor 0 81054 .coefficient) (.predecessor 1 81055 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event81057 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨12959⟩⟩, .operator (⟨81053, 0⟩, ⟨81050, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨10135⟩⟩, ⟨.program ⟨214⟩, ⟨12958⟩⟩], []⟩, (1)⟩)

def exact81058RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨10135⟩⟩, ⟨.program ⟨214⟩, ⟨12958⟩⟩], []⟩, (1)⟩]

theorem exact81058RawTermsValid :
    exact81058RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event81058 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12959⟩⟩) exact81058RawTerms (.finite 2704) 81056 .exactZero (none)

def event81059 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12960⟩⟩) 0 ⟨12959⟩ 81058

def event81060 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12960⟩⟩) (.identity (.predecessor 0 81059 .coefficient))

def event81061 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨12960⟩⟩) (.finite 2704)

def event81062 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23331⟩⟩) 0 ⟨12960⟩ 81061

def event81063 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23331⟩⟩) (.authority (.programFamilyFact))

def event81064 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨23331⟩⟩) (.finite 3720)

def event81065 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨6689⟩⟩) .missing

def event81066 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23332⟩⟩) 0 ⟨6689⟩ 81065

def event81067 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23332⟩⟩) 1 ⟨23331⟩ 81064

def event81068 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23332⟩⟩) (.authority (.operator))

def exact81069RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨23332⟩⟩]⟩, (1)⟩]

theorem exact81069RawTermsValid :
    exact81069RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event81069 : Event := .resultExact (⟨.program ⟨214⟩, ⟨23332⟩⟩) exact81069RawTerms .large 81068 .exactZero (none)

def event81070 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25604⟩⟩) 0 ⟨23332⟩ 81069

def event81071 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25604⟩⟩) (.authority (.operator))

def exact81072RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨25604⟩⟩]⟩, (1)⟩]

theorem exact81072RawTermsValid :
    exact81072RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event81072 : Event := .resultExact (⟨.program ⟨214⟩, ⟨25604⟩⟩) exact81072RawTerms (.finite 8192) 81071 .exactZero (none)

def event81073 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨110⟩⟩) (.authority (.operator))

def event81074 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨110⟩⟩) .exactZero

def event81075 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13054⟩⟩) 0 ⟨12960⟩ 81061

def event81076 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13054⟩⟩) 1 ⟨110⟩ 81074

def event81077 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13054⟩⟩) (.sum [.predecessor 0 81075 .coefficient, .predecessor 1 81076 .coefficient])

def event81078 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨13054⟩⟩) (.finite 2704)

def event81079 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13055⟩⟩) 0 ⟨13054⟩ 81078

def event81080 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13055⟩⟩) (.identity (.predecessor 0 81079 .coefficient))

def exact81081RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨10135⟩⟩, ⟨.program ⟨214⟩, ⟨12958⟩⟩], []⟩, (1)⟩]

theorem exact81081RawTermsValid :
    exact81081RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event81081 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13055⟩⟩) exact81081RawTerms (.finite 2704) 81080 .exactZero (none)

def event81082 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6544⟩⟩) (.authority (.factStore))

def exact81083RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact81083RawTermsValid :
    exact81083RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event81083 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6544⟩⟩) exact81083RawTerms .large 81082 .exactZero (none)

def event81084 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13056⟩⟩) 0 ⟨6544⟩ 81083

def event81085 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13056⟩⟩) 1 ⟨13055⟩ 81081

def event81086 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13056⟩⟩) (.product (.predecessor 0 81084 .coefficient) (.predecessor 1 81085 .coefficient) (⟨false, false, none, none, none⟩))

def event81087 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨13056⟩⟩, .operator (⟨81083, 0⟩, ⟨81081, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨10135⟩⟩, ⟨.program ⟨214⟩, ⟨12958⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact81088RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨10135⟩⟩, ⟨.program ⟨214⟩, ⟨12958⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact81088RawTermsValid :
    exact81088RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event81088 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13056⟩⟩) exact81088RawTerms .large 81086 .exactZero (none)

def event81089 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6757⟩⟩) 0 ⟨6689⟩ 81065

def event81090 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6757⟩⟩) (.authority (.operator))

def exact81091RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6757⟩⟩]⟩, (1)⟩]

theorem exact81091RawTermsValid :
    exact81091RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event81091 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6757⟩⟩) exact81091RawTerms .large 81090 .exactZero (none)

def event81092 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6788⟩⟩) 0 ⟨6757⟩ 81091

def event81093 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6788⟩⟩) (.identity (.predecessor 0 81092 .coefficient))

def exact81094RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6788⟩⟩]⟩, (1)⟩]

theorem exact81094RawTermsValid :
    exact81094RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event81094 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6788⟩⟩) exact81094RawTerms .large 81093 .exactZero (none)

def event81095 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7876⟩⟩) 0 ⟨6788⟩ 81094

def event81096 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7876⟩⟩) (.authority (.operator))

def exact81097RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨7876⟩⟩]⟩, (1)⟩]

theorem exact81097RawTermsValid :
    exact81097RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event81097 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7876⟩⟩) exact81097RawTerms (.finite 8192) 81096 .exactZero (none)

def event81098 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7877⟩⟩) 0 ⟨7876⟩ 81097

def event81099 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7877⟩⟩) 1 ⟨2348⟩ 81031

def event81100 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7877⟩⟩) (.scale (.predecessor 0 81098 .coefficient) (.value (.predecessor 1 81099 .coefficient)))

def exact81101RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨7876⟩⟩]⟩, (1)⟩]

theorem exact81101RawTermsValid :
    exact81101RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event81101 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7877⟩⟩) exact81101RawTerms (.finite 8192) 81100 .exactZero (none)

def event81102 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6768⟩⟩) 0 ⟨6757⟩ 81091

def event81103 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6768⟩⟩) (.identity (.predecessor 0 81102 .coefficient))

def exact81104RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6768⟩⟩]⟩, (1)⟩]

theorem exact81104RawTermsValid :
    exact81104RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event81104 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6768⟩⟩) exact81104RawTerms .large 81103 .exactZero (none)

def event81105 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7878⟩⟩) 0 ⟨6768⟩ 81104

def event81106 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7878⟩⟩) 1 ⟨7877⟩ 81101

def event81107 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7878⟩⟩) (.product (.predecessor 0 81105 .coefficient) (.predecessor 1 81106 .coefficient) (⟨false, false, none, none, none⟩))

def event81108 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7878⟩⟩, .operator (⟨81104, 0⟩, ⟨81101, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6768⟩⟩, ⟨.program ⟨214⟩, ⟨7876⟩⟩]⟩, (1)⟩)

def exact81109RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6768⟩⟩, ⟨.program ⟨214⟩, ⟨7876⟩⟩]⟩, (1)⟩]

theorem exact81109RawTermsValid :
    exact81109RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event81109 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7878⟩⟩) exact81109RawTerms .large 81107 .exactZero (none)

def event81110 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13057⟩⟩) 0 ⟨7878⟩ 81109

def event81111 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13057⟩⟩) 1 ⟨13056⟩ 81088

def event81112 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13057⟩⟩) (.sum [.predecessor 0 81110 .coefficient, .predecessor 1 81111 .coefficient])

def exact81113RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6768⟩⟩, ⟨.program ⟨214⟩, ⟨7876⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨10135⟩⟩, ⟨.program ⟨214⟩, ⟨12958⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact81113RawTermsValid :
    exact81113RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event81113 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13057⟩⟩) exact81113RawTerms .large 81112 .exactZero (none)

def event81114 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25607⟩⟩) 0 ⟨13057⟩ 81113

def event81115 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25607⟩⟩) 1 ⟨25604⟩ 81072

def event81116 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25607⟩⟩) (.product (.predecessor 0 81114 .coefficient) (.predecessor 1 81115 .coefficient) (⟨false, false, none, none, none⟩))

def event81117 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25607⟩⟩, .operator (⟨81113, 0⟩, ⟨81072, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6768⟩⟩, ⟨.program ⟨214⟩, ⟨7876⟩⟩, ⟨.program ⟨214⟩, ⟨25604⟩⟩]⟩, (1)⟩)

def event81118 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25607⟩⟩, .operator (⟨81113, 1⟩, ⟨81072, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨10135⟩⟩, ⟨.program ⟨214⟩, ⟨12958⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25604⟩⟩]⟩, (-1)⟩)

def event81119 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨25607⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨10135⟩⟩, ⟨.program ⟨214⟩, ⟨12958⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25604⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨25604⟩⟩) ⟨23332⟩ 81069)

def event81120 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25607⟩⟩, .relation 81119 0, ⟨[⟨.program ⟨214⟩, ⟨10135⟩⟩, ⟨.program ⟨214⟩, ⟨12958⟩⟩], [⟨.program ⟨214⟩, ⟨23332⟩⟩]⟩, (-1)⟩)

def exact81121RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6768⟩⟩, ⟨.program ⟨214⟩, ⟨7876⟩⟩, ⟨.program ⟨214⟩, ⟨25604⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨10135⟩⟩, ⟨.program ⟨214⟩, ⟨12958⟩⟩], [⟨.program ⟨214⟩, ⟨23332⟩⟩]⟩, (-1)⟩]

theorem exact81121RawTermsValid :
    exact81121RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event81121 : Event := .resultExact (⟨.program ⟨214⟩, ⟨25607⟩⟩) exact81121RawTerms .large 81116 .exactZero (none)

def event81122 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16752⟩⟩) 0 ⟨12960⟩ 81061

def event81123 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16752⟩⟩) (.authority (.programFamilyFact))

def exact81124RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16752⟩⟩], []⟩, (1)⟩]

theorem exact81124RawTermsValid :
    exact81124RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event81124 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16752⟩⟩) exact81124RawTerms (.finite 52) 81123 .exactZero (none)

def event81125 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16754⟩⟩) 0 ⟨6544⟩ 81083

def event81126 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16754⟩⟩) 1 ⟨16752⟩ 81124

def event81127 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16754⟩⟩) (.product (.predecessor 0 81125 .coefficient) (.predecessor 1 81126 .coefficient) (⟨false, true, none, none, some 1⟩))

def event81128 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨16754⟩⟩, .operator (⟨81083, 0⟩, ⟨81124, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨16752⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact81129RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16752⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact81129RawTermsValid :
    exact81129RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event81129 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16754⟩⟩) exact81129RawTerms .large 81127 .exactZero (none)

def event81130 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6705⟩⟩) 0 ⟨6689⟩ 81065

def event81131 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6705⟩⟩) (.authority (.operator))

def exact81132RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6705⟩⟩]⟩, (1)⟩]

theorem exact81132RawTermsValid :
    exact81132RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event81132 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6705⟩⟩) exact81132RawTerms .large 81131 .exactZero (none)

def event81133 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16755⟩⟩) 0 ⟨6705⟩ 81132

def event81134 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16755⟩⟩) 1 ⟨16754⟩ 81129

def event81135 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16755⟩⟩) (.sum [.predecessor 0 81133 .coefficient, .predecessor 1 81134 .coefficient])

def exact81136RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6705⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16752⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact81136RawTermsValid :
    exact81136RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event81136 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16755⟩⟩) exact81136RawTerms .large 81135 .exactZero (none)

def event81137 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25608⟩⟩) 0 ⟨16755⟩ 81136

def event81138 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25608⟩⟩) 1 ⟨25607⟩ 81121

def event81139 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25608⟩⟩) (.sum [.predecessor 0 81137 .coefficient, .predecessor 1 81138 .coefficient])

def exact81140RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6705⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6768⟩⟩, ⟨.program ⟨214⟩, ⟨7876⟩⟩, ⟨.program ⟨214⟩, ⟨25604⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨10135⟩⟩, ⟨.program ⟨214⟩, ⟨12958⟩⟩], [⟨.program ⟨214⟩, ⟨23332⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16752⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact81140RawTermsValid :
    exact81140RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event81140 : Event := .resultExact (⟨.program ⟨214⟩, ⟨25608⟩⟩) exact81140RawTerms .large 81139 .exactZero (none)

def event81141 : Event := .preFoldPolynomial 81140 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6705⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6768⟩⟩, ⟨.program ⟨214⟩, ⟨7876⟩⟩, ⟨.program ⟨214⟩, ⟨25604⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨10135⟩⟩, ⟨.program ⟨214⟩, ⟨12958⟩⟩], [⟨.program ⟨214⟩, ⟨23332⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16752⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩] .exactZero none

def exact81142RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6705⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6768⟩⟩, ⟨.program ⟨214⟩, ⟨7876⟩⟩, ⟨.program ⟨214⟩, ⟨25604⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨10135⟩⟩, ⟨.program ⟨214⟩, ⟨12958⟩⟩], [⟨.program ⟨214⟩, ⟨23332⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16752⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

def event81142 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨25608⟩⟩) 81141 exact81142RawTerms .large 81139 .exactZero (none)

def event81143 : Event := .specializationComputed (⟨.program ⟨214⟩, ⟨12960⟩⟩) ⟨⟨118⟩, ⟨24⟩, ⟨109⟩⟩ ⟨80979, 81143⟩

def event81144 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨20107⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20104⟩⟩]⟩) (1) 0 2 (.universal 81143 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20104⟩⟩]⟩) (none) 81142)

def event81145 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20107⟩⟩, .relation 81144 0, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6705⟩⟩]⟩, (1)⟩)

def event81146 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20107⟩⟩, .relation 81144 1, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6768⟩⟩, ⟨.program ⟨214⟩, ⟨7876⟩⟩, ⟨.program ⟨214⟩, ⟨25604⟩⟩]⟩, (-1)⟩)

def event81147 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20107⟩⟩, .relation 81144 2, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨10135⟩⟩, ⟨.program ⟨214⟩, ⟨12958⟩⟩], [⟨.program ⟨214⟩, ⟨23332⟩⟩]⟩, (1)⟩)

def event81148 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20107⟩⟩, .relation 81144 3, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16752⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact81149RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6705⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6768⟩⟩, ⟨.program ⟨214⟩, ⟨7876⟩⟩, ⟨.program ⟨214⟩, ⟨25604⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨10135⟩⟩, ⟨.program ⟨214⟩, ⟨12958⟩⟩], [⟨.program ⟨214⟩, ⟨23332⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16752⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact81149RawTermsValid :
    exact81149RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event81149 : Event := .resultExact (⟨.program ⟨214⟩, ⟨20107⟩⟩) exact81149RawTerms .large 80975 (.finite 1811303510016) (some (80977))

def event81150 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25606⟩⟩) 0 ⟨20107⟩ 81149

def event81151 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25606⟩⟩) 1 ⟨25605⟩ 80965

def eventLeaf5056 : Array AnnotatedEvent := #[
  { event := event80896
    frameStart := 0 },
  { event := event80897
    frameStart := 0 },
  { event := event80898
    frameStart := 0 },
  { event := event80899
    frameStart := 0 },
  { event := event80900
    frameStart := 0 },
  { event := event80901
    frameStart := 0 },
  { event := event80902
    frameStart := 0 },
  { event := event80903
    frameStart := 0 },
  { event := event80904
    frameStart := 0 },
  { event := event80905
    frameStart := 0 },
  { event := event80906
    frameStart := 0 },
  { event := event80907
    frameStart := 0 },
  { event := event80908
    frameStart := 0 },
  { event := event80909
    frameStart := 0 },
  { event := event80910
    frameStart := 0 },
  { event := event80911
    frameStart := 0 }
]

def eventLeaf5057 : Array AnnotatedEvent := #[
  { event := event80912
    frameStart := 0 },
  { event := event80913
    frameStart := 0 },
  { event := event80914
    frameStart := 0 },
  { event := event80915
    frameStart := 0 },
  { event := event80916
    frameStart := 0 },
  { event := event80917
    frameStart := 0 },
  { event := event80918
    frameStart := 0 },
  { event := event80919
    frameStart := 0 },
  { event := event80920
    frameStart := 0 },
  { event := event80921
    frameStart := 0 },
  { event := event80922
    frameStart := 0 },
  { event := event80923
    frameStart := 0 },
  { event := event80924
    frameStart := 0 },
  { event := event80925
    frameStart := 0 },
  { event := event80926
    frameStart := 0 },
  { event := event80927
    frameStart := 0 }
]

def eventLeaf5058 : Array AnnotatedEvent := #[
  { event := event80928
    frameStart := 0 },
  { event := event80929
    frameStart := 0 },
  { event := event80930
    frameStart := 0 },
  { event := event80931
    frameStart := 0 },
  { event := event80932
    frameStart := 0 },
  { event := event80933
    frameStart := 0 },
  { event := event80934
    frameStart := 0 },
  { event := event80935
    frameStart := 0 },
  { event := event80936
    frameStart := 0 },
  { event := event80937
    frameStart := 0 },
  { event := event80938
    frameStart := 0 },
  { event := event80939
    frameStart := 0 },
  { event := event80940
    frameStart := 0 },
  { event := event80941
    frameStart := 0 },
  { event := event80942
    frameStart := 0 },
  { event := event80943
    frameStart := 0 }
]

def eventLeaf5059 : Array AnnotatedEvent := #[
  { event := event80944
    frameStart := 0 },
  { event := event80945
    frameStart := 0 },
  { event := event80946
    frameStart := 0 },
  { event := event80947
    frameStart := 0 },
  { event := event80948
    frameStart := 0 },
  { event := event80949
    frameStart := 0 },
  { event := event80950
    frameStart := 0 },
  { event := event80951
    frameStart := 0 },
  { event := event80952
    frameStart := 0 },
  { event := event80953
    frameStart := 0 },
  { event := event80954
    frameStart := 0 },
  { event := event80955
    frameStart := 0 },
  { event := event80956
    frameStart := 0 },
  { event := event80957
    frameStart := 0 },
  { event := event80958
    frameStart := 0 },
  { event := event80959
    frameStart := 0 }
]

def eventLeaf5060 : Array AnnotatedEvent := #[
  { event := event80960
    frameStart := 0 },
  { event := event80961
    frameStart := 0 },
  { event := event80962
    frameStart := 0 },
  { event := event80963
    frameStart := 0 },
  { event := event80964
    frameStart := 0 },
  { event := event80965
    frameStart := 0 },
  { event := event80966
    frameStart := 0 },
  { event := event80967
    frameStart := 0 },
  { event := event80968
    frameStart := 0 },
  { event := event80969
    frameStart := 0 },
  { event := event80970
    frameStart := 0 },
  { event := event80971
    frameStart := 0 },
  { event := event80972
    frameStart := 0 },
  { event := event80973
    frameStart := 0 },
  { event := event80974
    frameStart := 0 },
  { event := event80975
    frameStart := 0 }
]

def eventLeaf5061 : Array AnnotatedEvent := #[
  { event := event80976
    frameStart := 0 },
  { event := event80977
    frameStart := 0 },
  { event := event80978
    frameStart := 0 },
  { event := event80979
    frameStart := 80979 },
  { event := event80980
    frameStart := 80979 },
  { event := event80981
    frameStart := 80979 },
  { event := event80982
    frameStart := 80979 },
  { event := event80983
    frameStart := 80979 },
  { event := event80984
    frameStart := 80979 },
  { event := event80985
    frameStart := 80979 },
  { event := event80986
    frameStart := 80979 },
  { event := event80987
    frameStart := 80979 },
  { event := event80988
    frameStart := 80979 },
  { event := event80989
    frameStart := 80979 },
  { event := event80990
    frameStart := 80979 },
  { event := event80991
    frameStart := 80979 }
]

def eventLeaf5062 : Array AnnotatedEvent := #[
  { event := event80992
    frameStart := 80979 },
  { event := event80993
    frameStart := 80979 },
  { event := event80994
    frameStart := 80979 },
  { event := event80995
    frameStart := 80979 },
  { event := event80996
    frameStart := 80979 },
  { event := event80997
    frameStart := 80979 },
  { event := event80998
    frameStart := 80979 },
  { event := event80999
    frameStart := 80979 },
  { event := event81000
    frameStart := 80979 },
  { event := event81001
    frameStart := 80979 },
  { event := event81002
    frameStart := 80979 },
  { event := event81003
    frameStart := 80979 },
  { event := event81004
    frameStart := 80979 },
  { event := event81005
    frameStart := 80979 },
  { event := event81006
    frameStart := 80979 },
  { event := event81007
    frameStart := 80979 }
]

def eventLeaf5063 : Array AnnotatedEvent := #[
  { event := event81008
    frameStart := 80979 },
  { event := event81009
    frameStart := 80979 },
  { event := event81010
    frameStart := 80979 },
  { event := event81011
    frameStart := 80979 },
  { event := event81012
    frameStart := 80979 },
  { event := event81013
    frameStart := 80979 },
  { event := event81014
    frameStart := 80979 },
  { event := event81015
    frameStart := 80979 },
  { event := event81016
    frameStart := 80979 },
  { event := event81017
    frameStart := 80979 },
  { event := event81018
    frameStart := 80979 },
  { event := event81019
    frameStart := 80979 },
  { event := event81020
    frameStart := 80979 },
  { event := event81021
    frameStart := 80979 },
  { event := event81022
    frameStart := 80979 },
  { event := event81023
    frameStart := 80979 }
]

def eventLeaf5064 : Array AnnotatedEvent := #[
  { event := event81024
    frameStart := 80979 },
  { event := event81025
    frameStart := 80979 },
  { event := event81026
    frameStart := 80979 },
  { event := event81027
    frameStart := 81027 },
  { event := event81028
    frameStart := 81027 },
  { event := event81029
    frameStart := 81027 },
  { event := event81030
    frameStart := 81027 },
  { event := event81031
    frameStart := 81027 },
  { event := event81032
    frameStart := 81027 },
  { event := event81033
    frameStart := 81027 },
  { event := event81034
    frameStart := 81027 },
  { event := event81035
    frameStart := 81027 },
  { event := event81036
    frameStart := 81027 },
  { event := event81037
    frameStart := 81027 },
  { event := event81038
    frameStart := 81027 },
  { event := event81039
    frameStart := 81027 }
]

def eventLeaf5065 : Array AnnotatedEvent := #[
  { event := event81040
    frameStart := 81027 },
  { event := event81041
    frameStart := 81027 },
  { event := event81042
    frameStart := 81027 },
  { event := event81043
    frameStart := 81027 },
  { event := event81044
    frameStart := 81027 },
  { event := event81045
    frameStart := 81027 },
  { event := event81046
    frameStart := 81027 },
  { event := event81047
    frameStart := 81027 },
  { event := event81048
    frameStart := 81027 },
  { event := event81049
    frameStart := 81027 },
  { event := event81050
    frameStart := 81027 },
  { event := event81051
    frameStart := 81027 },
  { event := event81052
    frameStart := 81027 },
  { event := event81053
    frameStart := 81027 },
  { event := event81054
    frameStart := 81027 },
  { event := event81055
    frameStart := 81027 }
]

def eventLeaf5066 : Array AnnotatedEvent := #[
  { event := event81056
    frameStart := 81027 },
  { event := event81057
    frameStart := 81027 },
  { event := event81058
    frameStart := 81027 },
  { event := event81059
    frameStart := 81027 },
  { event := event81060
    frameStart := 81027 },
  { event := event81061
    frameStart := 81027 },
  { event := event81062
    frameStart := 81027 },
  { event := event81063
    frameStart := 81027 },
  { event := event81064
    frameStart := 81027 },
  { event := event81065
    frameStart := 81027 },
  { event := event81066
    frameStart := 81027 },
  { event := event81067
    frameStart := 81027 },
  { event := event81068
    frameStart := 81027 },
  { event := event81069
    frameStart := 81027 },
  { event := event81070
    frameStart := 81027 },
  { event := event81071
    frameStart := 81027 }
]

def eventLeaf5067 : Array AnnotatedEvent := #[
  { event := event81072
    frameStart := 81027 },
  { event := event81073
    frameStart := 81027 },
  { event := event81074
    frameStart := 81027 },
  { event := event81075
    frameStart := 81027 },
  { event := event81076
    frameStart := 81027 },
  { event := event81077
    frameStart := 81027 },
  { event := event81078
    frameStart := 81027 },
  { event := event81079
    frameStart := 81027 },
  { event := event81080
    frameStart := 81027 },
  { event := event81081
    frameStart := 81027 },
  { event := event81082
    frameStart := 81027 },
  { event := event81083
    frameStart := 81027 },
  { event := event81084
    frameStart := 81027 },
  { event := event81085
    frameStart := 81027 },
  { event := event81086
    frameStart := 81027 },
  { event := event81087
    frameStart := 81027 }
]

def eventLeaf5068 : Array AnnotatedEvent := #[
  { event := event81088
    frameStart := 81027 },
  { event := event81089
    frameStart := 81027 },
  { event := event81090
    frameStart := 81027 },
  { event := event81091
    frameStart := 81027 },
  { event := event81092
    frameStart := 81027 },
  { event := event81093
    frameStart := 81027 },
  { event := event81094
    frameStart := 81027 },
  { event := event81095
    frameStart := 81027 },
  { event := event81096
    frameStart := 81027 },
  { event := event81097
    frameStart := 81027 },
  { event := event81098
    frameStart := 81027 },
  { event := event81099
    frameStart := 81027 },
  { event := event81100
    frameStart := 81027 },
  { event := event81101
    frameStart := 81027 },
  { event := event81102
    frameStart := 81027 },
  { event := event81103
    frameStart := 81027 }
]

def eventLeaf5069 : Array AnnotatedEvent := #[
  { event := event81104
    frameStart := 81027 },
  { event := event81105
    frameStart := 81027 },
  { event := event81106
    frameStart := 81027 },
  { event := event81107
    frameStart := 81027 },
  { event := event81108
    frameStart := 81027 },
  { event := event81109
    frameStart := 81027 },
  { event := event81110
    frameStart := 81027 },
  { event := event81111
    frameStart := 81027 },
  { event := event81112
    frameStart := 81027 },
  { event := event81113
    frameStart := 81027 },
  { event := event81114
    frameStart := 81027 },
  { event := event81115
    frameStart := 81027 },
  { event := event81116
    frameStart := 81027 },
  { event := event81117
    frameStart := 81027 },
  { event := event81118
    frameStart := 81027 },
  { event := event81119
    frameStart := 81027 }
]

def eventLeaf5070 : Array AnnotatedEvent := #[
  { event := event81120
    frameStart := 81027 },
  { event := event81121
    frameStart := 81027 },
  { event := event81122
    frameStart := 81027 },
  { event := event81123
    frameStart := 81027 },
  { event := event81124
    frameStart := 81027 },
  { event := event81125
    frameStart := 81027 },
  { event := event81126
    frameStart := 81027 },
  { event := event81127
    frameStart := 81027 },
  { event := event81128
    frameStart := 81027 },
  { event := event81129
    frameStart := 81027 },
  { event := event81130
    frameStart := 81027 },
  { event := event81131
    frameStart := 81027 },
  { event := event81132
    frameStart := 81027 },
  { event := event81133
    frameStart := 81027 },
  { event := event81134
    frameStart := 81027 },
  { event := event81135
    frameStart := 81027 }
]

def eventLeaf5071 : Array AnnotatedEvent := #[
  { event := event81136
    frameStart := 81027 },
  { event := event81137
    frameStart := 81027 },
  { event := event81138
    frameStart := 81027 },
  { event := event81139
    frameStart := 81027 },
  { event := event81140
    frameStart := 81027 },
  { event := event81141
    frameStart := 81027 },
  { event := event81142
    frameStart := 81027 },
  { event := event81143
    frameStart := 0 },
  { event := event81144
    frameStart := 0 },
  { event := event81145
    frameStart := 0 },
  { event := event81146
    frameStart := 0 },
  { event := event81147
    frameStart := 0 },
  { event := event81148
    frameStart := 0 },
  { event := event81149
    frameStart := 0 },
  { event := event81150
    frameStart := 0 },
  { event := event81151
    frameStart := 0 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events316
