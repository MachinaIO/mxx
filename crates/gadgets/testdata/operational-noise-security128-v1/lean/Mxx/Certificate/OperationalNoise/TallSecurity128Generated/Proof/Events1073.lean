import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events1073

open Mxx.Certificate.OperationalNoise
open CertificateABI

def exact274688RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17529⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨15722⟩⟩], [⟨.program ⟨257⟩, ⟨16926⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨15903⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact274688RawTermsValid :
    exact274688RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event274688 : Event := .resultExact (⟨.program ⟨257⟩, ⟨16433⟩⟩) exact274688RawTerms .large 274520 (.finite 202072841853861888) (some (274522))

def event274689 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17532⟩⟩) 0 ⟨16433⟩ 274688

def event274690 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17532⟩⟩) 1 ⟨17531⟩ 274510

def event274691 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17532⟩⟩) (.sum [.predecessor 0 274689 .coefficient, .predecessor 1 274690 .coefficient])

def event274692 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17532⟩⟩, .operator (⟨274688, 0⟩, ⟨274510, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17529⟩⟩]⟩, (1)⟩)

def event274693 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17532⟩⟩, .operator (⟨274688, 2⟩, ⟨274510, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨15722⟩⟩], [⟨.program ⟨257⟩, ⟨16926⟩⟩]⟩, (-1)⟩)

def event274694 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17532⟩⟩) (.sum [.result 274688 .summary, .result 274510 .summary])

def exact274695RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨15903⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact274695RawTermsValid :
    exact274695RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event274695 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17532⟩⟩) exact274695RawTerms .large 274691 (.finite 32188807212483706889510625476608) (some (274694))

def event274696 : Event := .predecessor (⟨.program ⟨257⟩, ⟨20399⟩⟩) 0 ⟨17532⟩ 274695

def event274697 : Event := .predecessor (⟨.program ⟨257⟩, ⟨20399⟩⟩) 1 ⟨20398⟩ 274213

def event274698 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨20399⟩⟩) (.sum [.predecessor 0 274696 .coefficient, .predecessor 1 274697 .coefficient])

def event274699 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨20399⟩⟩) (.sum [.result 274695 .summary, .result 274213 .summary])

def exact274700RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨15903⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨18709⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact274700RawTermsValid :
    exact274700RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event274700 : Event := .resultExact (⟨.program ⟨257⟩, ⟨20399⟩⟩) exact274700RawTerms .large 274698 (.finite 64377712650190257467641695830016) (some (274699))

def event274701 : Event := .predecessor (⟨.program ⟨257⟩, ⟨23619⟩⟩) 0 ⟨20399⟩ 274700

def event274702 : Event := .predecessor (⟨.program ⟨257⟩, ⟨23619⟩⟩) 1 ⟨23618⟩ 273731

def event274703 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨23619⟩⟩) (.sum [.predecessor 0 274701 .coefficient, .predecessor 1 274702 .coefficient])

def event274704 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨23619⟩⟩) (.sum [.result 274700 .summary, .result 273731 .summary])

def exact274705RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨15903⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨18709⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨21929⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact274705RawTermsValid :
    exact274705RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event274705 : Event := .resultExact (⟨.program ⟨257⟩, ⟨23619⟩⟩) exact274705RawTerms .large 274703 (.finite 96566716313119651734393211060224) (some (274704))

def event274706 : Event := .predecessor (⟨.program ⟨257⟩, ⟨33639⟩⟩) 0 ⟨23619⟩ 274705

def event274707 : Event := .predecessor (⟨.program ⟨257⟩, ⟨33639⟩⟩) 1 ⟨33638⟩ 273249

def event274708 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨33639⟩⟩) (.sum [.predecessor 0 274706 .coefficient, .predecessor 1 274707 .coefficient])

def event274709 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨33639⟩⟩) (.sum [.result 274705 .summary, .result 273249 .summary])

def exact274710RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨15903⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨18709⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨21929⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨31949⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact274710RawTermsValid :
    exact274710RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event274710 : Event := .resultExact (⟨.program ⟨257⟩, ⟨33639⟩⟩) exact274710RawTerms .large 274708 (.finite 128755916426494733378385616044032) (some (274709))

def event274711 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52699⟩⟩) 0 ⟨33639⟩ 274710

def event274712 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52699⟩⟩) 1 ⟨52698⟩ 272767

def event274713 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨52699⟩⟩) (.sum [.predecessor 0 274711 .coefficient, .predecessor 1 274712 .coefficient])

def event274714 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨52699⟩⟩) (.sum [.result 274710 .summary, .result 272767 .summary])

def exact274715RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨15903⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨18709⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨21929⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨31949⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨51004⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact274715RawTermsValid :
    exact274715RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event274715 : Event := .resultExact (⟨.program ⟨257⟩, ⟨52699⟩⟩) exact274715RawTerms .large 274713 (.finite 160945509440761189776859800535040) (some (274714))

def event274716 : Event := .predecessor (⟨.program ⟨257⟩, ⟨55679⟩⟩) 0 ⟨52699⟩ 274715

def event274717 : Event := .predecessor (⟨.program ⟨257⟩, ⟨55679⟩⟩) 1 ⟨55678⟩ 272285

def event274718 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨55679⟩⟩) (.sum [.predecessor 0 274716 .coefficient, .predecessor 1 274717 .coefficient])

def event274719 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨55679⟩⟩) (.sum [.result 274715 .summary, .result 272285 .summary])

def exact274720RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨15903⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨18709⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨21929⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨31949⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨51004⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨53984⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact274720RawTermsValid :
    exact274720RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event274720 : Event := .resultExact (⟨.program ⟨257⟩, ⟨55679⟩⟩) exact274720RawTerms .large 274718 (.finite 193135298905473333552574874779648) (some (274719))

def event274721 : Event := .predecessor (⟨.program ⟨257⟩, ⟨58659⟩⟩) 0 ⟨55679⟩ 274720

def event274722 : Event := .predecessor (⟨.program ⟨257⟩, ⟨58659⟩⟩) 1 ⟨58658⟩ 271803

def event274723 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨58659⟩⟩) (.sum [.predecessor 0 274721 .coefficient, .predecessor 1 274722 .coefficient])

def event274724 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨58659⟩⟩) (.sum [.result 274720 .summary, .result 271803 .summary])

def exact274725RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨15903⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨18709⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨21929⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨31949⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨51004⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨53984⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨56964⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact274725RawTermsValid :
    exact274725RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event274725 : Event := .resultExact (⟨.program ⟨257⟩, ⟨58659⟩⟩) exact274725RawTerms .large 274723 (.finite 225325481271076852082771728531456) (some (274724))

def event274726 : Event := .predecessor (⟨.program ⟨257⟩, ⟨61639⟩⟩) 0 ⟨58659⟩ 274725

def event274727 : Event := .predecessor (⟨.program ⟨257⟩, ⟨61639⟩⟩) 1 ⟨61638⟩ 271321

def event274728 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨61639⟩⟩) (.sum [.predecessor 0 274726 .coefficient, .predecessor 1 274727 .coefficient])

def event274729 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨61639⟩⟩) (.sum [.result 274725 .summary, .result 271321 .summary])

def exact274730RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨15903⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨18709⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨21929⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨31949⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨51004⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨53984⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨56964⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨59944⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact274730RawTermsValid :
    exact274730RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event274730 : Event := .resultExact (⟨.program ⟨257⟩, ⟨61639⟩⟩) exact274730RawTerms .large 274728 (.finite 257515860087126057990209472036864) (some (274729))

def event274731 : Event := .predecessor (⟨.program ⟨257⟩, ⟨64619⟩⟩) 0 ⟨61639⟩ 274730

def event274732 : Event := .predecessor (⟨.program ⟨257⟩, ⟨64619⟩⟩) 1 ⟨64618⟩ 270839

def event274733 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨64619⟩⟩) (.sum [.predecessor 0 274731 .coefficient, .predecessor 1 274732 .coefficient])

def event274734 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨64619⟩⟩) (.sum [.result 274730 .summary, .result 270839 .summary])

def exact274735RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨15903⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨18709⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨21929⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨31949⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨51004⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨53984⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨56964⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨59944⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨62924⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact274735RawTermsValid :
    exact274735RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event274735 : Event := .resultExact (⟨.program ⟨257⟩, ⟨64619⟩⟩) exact274735RawTerms .large 274733 (.finite 289706631804066638652128995049472) (some (274734))

def event274736 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69524⟩⟩) 0 ⟨64619⟩ 274735

def event274737 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69524⟩⟩) 1 ⟨69523⟩ 270357

def event274738 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69524⟩⟩) (.sum [.predecessor 0 274736 .coefficient, .predecessor 1 274737 .coefficient])

def event274739 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69524⟩⟩) (.sum [.result 274735 .summary, .result 270357 .summary])

def exact274740RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨15903⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨18709⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨21929⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨31949⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨51004⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨53984⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨56964⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨59944⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨62924⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨66019⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact274740RawTermsValid :
    exact274740RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event274740 : Event := .resultExact (⟨.program ⟨257⟩, ⟨69524⟩⟩) exact274740RawTerms .large 274738 (.finite 321897992872344281445771187322880) (some (274739))

def event274741 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69525⟩⟩) 0 ⟨69524⟩ 274740

def event274742 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69525⟩⟩) 1 ⟨28085⟩ 269875

def event274743 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69525⟩⟩) (.sum [.predecessor 0 274741 .coefficient, .predecessor 1 274742 .coefficient])

def event274744 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69525⟩⟩) (.sum [.result 274740 .summary, .result 269875 .summary])

def exact274745RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨15903⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨18709⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨21929⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨26512⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨31949⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨51004⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨53984⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨56964⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨59944⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨62924⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨66019⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact274745RawTermsValid :
    exact274745RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event274745 : Event := .resultExact (⟨.program ⟨257⟩, ⟨69525⟩⟩) exact274745RawTerms .large 274743 (.finite 354089550391067611616654269349888) (some (274744))

def event274746 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69526⟩⟩) 0 ⟨69525⟩ 274745

def event274747 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69526⟩⟩) 1 ⟨30765⟩ 269393

def event274748 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69526⟩⟩) (.sum [.predecessor 0 274746 .coefficient, .predecessor 1 274747 .coefficient])

def event274749 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69526⟩⟩) (.sum [.result 274745 .summary, .result 269393 .summary])

def exact274750RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨15903⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨18709⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨21929⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨26512⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨29192⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨31949⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨51004⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨53984⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨56964⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨59944⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨62924⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨66019⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact274750RawTermsValid :
    exact274750RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event274750 : Event := .resultExact (⟨.program ⟨257⟩, ⟨69526⟩⟩) exact274750RawTerms .large 274748 (.finite 386281697261128003919260020637696) (some (274749))

def event274751 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69527⟩⟩) 0 ⟨69526⟩ 274750

def event274752 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69527⟩⟩) 1 ⟨36425⟩ 268911

def event274753 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69527⟩⟩) (.sum [.predecessor 0 274751 .coefficient, .predecessor 1 274752 .coefficient])

def event274754 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69527⟩⟩) (.sum [.result 274750 .summary, .result 268911 .summary])

def exact274755RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨15903⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨18709⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨21929⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨26512⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨29192⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨31949⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨34856⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨51004⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨53984⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨56964⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨59944⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨62924⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨66019⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact274755RawTermsValid :
    exact274755RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event274755 : Event := .resultExact (⟨.program ⟨257⟩, ⟨69527⟩⟩) exact274755RawTerms .large 274753 (.finite 418474237032079770976347551432704) (some (274754))

def event274756 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69528⟩⟩) 0 ⟨69527⟩ 274755

def event274757 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69528⟩⟩) 1 ⟨39105⟩ 268429

def event274758 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69528⟩⟩) (.sum [.predecessor 0 274756 .coefficient, .predecessor 1 274757 .coefficient])

def event274759 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69528⟩⟩) (.sum [.result 274755 .summary, .result 268429 .summary])

def exact274760RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7224⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨15903⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨18709⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨21929⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨26512⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨29192⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨31949⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨34856⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨37536⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨51004⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨53984⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨56964⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨59944⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨62924⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨66019⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact274760RawTermsValid :
    exact274760RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event274760 : Event := .resultExact (⟨.program ⟨257⟩, ⟨69528⟩⟩) exact274760RawTerms .large 274758 (.finite 450666973253477225410675971981312) (some (274759))

def event274761 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69529⟩⟩) 0 ⟨69528⟩ 274760

def event274762 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69529⟩⟩) 1 ⟨41785⟩ 267947

def event274763 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69529⟩⟩) (.sum [.predecessor 0 274761 .coefficient, .predecessor 1 274762 .coefficient])

def event274764 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69529⟩⟩) (.sum [.result 274760 .summary, .result 267947 .summary])

def exact274765RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7224⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7226⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨15903⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨18709⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨21929⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨26512⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨29192⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨31949⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨34856⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨37536⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨40212⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨51004⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨53984⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨56964⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨59944⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨62924⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨66019⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact274765RawTermsValid :
    exact274765RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event274765 : Event := .resultExact (⟨.program ⟨257⟩, ⟨69529⟩⟩) exact274765RawTerms .large 274763 (.finite 482860102375766054599486172037120) (some (274764))

def event274766 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69530⟩⟩) 0 ⟨69529⟩ 274765

def event274767 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69530⟩⟩) 1 ⟨44465⟩ 267465

def event274768 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69530⟩⟩) (.sum [.predecessor 0 274766 .coefficient, .predecessor 1 274767 .coefficient])

def event274769 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69530⟩⟩) (.sum [.result 274765 .summary, .result 267465 .summary])

def exact274770RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7224⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7226⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7228⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨15903⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨18709⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨21929⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨26512⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨29192⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨31949⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨34856⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨37536⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨40212⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨42892⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨51004⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨53984⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨56964⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨59944⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨62924⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨66019⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact274770RawTermsValid :
    exact274770RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event274770 : Event := .resultExact (⟨.program ⟨257⟩, ⟨69530⟩⟩) exact274770RawTerms .large 274768 (.finite 515053820849391945920019041353728) (some (274769))

def event274771 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69531⟩⟩) 0 ⟨69530⟩ 274770

def event274772 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69531⟩⟩) 1 ⟨47145⟩ 266983

def event274773 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69531⟩⟩) (.sum [.predecessor 0 274771 .coefficient, .predecessor 1 274772 .coefficient])

def event274774 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69531⟩⟩) (.sum [.result 274770 .summary, .result 266983 .summary])

def exact274775RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7224⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7226⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7228⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7230⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨15903⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨18709⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨21929⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨26512⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨29192⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨31949⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨34856⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨37536⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨40212⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨42892⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨45576⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨51004⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨53984⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨56964⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨59944⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨62924⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨66019⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact274775RawTermsValid :
    exact274775RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event274775 : Event := .resultExact (⟨.program ⟨257⟩, ⟨69531⟩⟩) exact274775RawTerms .large 274773 (.finite 547248128674354899372274579931136) (some (274774))

def event274776 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69532⟩⟩) 0 ⟨69531⟩ 274775

def event274777 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69532⟩⟩) 1 ⟨49825⟩ 266501

def event274778 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69532⟩⟩) (.sum [.predecessor 0 274776 .coefficient, .predecessor 1 274777 .coefficient])

def event274779 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69532⟩⟩) (.sum [.result 274775 .summary, .result 266501 .summary])

def exact274780RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7224⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7226⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7228⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7230⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7232⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨15903⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨18709⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨21929⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨26512⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨29192⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨31949⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨34856⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨37536⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨40212⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨42892⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨45576⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨48256⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨51004⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨53984⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨56964⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨59944⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨62924⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨66019⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact274780RawTermsValid :
    exact274780RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event274780 : Event := .resultExact (⟨.program ⟨257⟩, ⟨69532⟩⟩) exact274780RawTerms .large 274778 (.finite 579442632949763540201771008262144) (some (274779))

def event274781 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70981⟩⟩) 0 ⟨69532⟩ 274780

def event274782 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70981⟩⟩) 1 ⟨70979⟩ 266003

def event274783 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70981⟩⟩) (.product (.predecessor 0 274781 .coefficient) (.predecessor 1 274782 .coefficient) (⟨false, false, none, none, none⟩))

def event274784 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70981⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩) [⟨.result 266003 .coefficient, false, none⟩])

def event274785 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70981⟩⟩) (.product (.result 274780 .summary) (.transfer 274784) (⟨false, false, none, none, none⟩))

def event274786 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70981⟩⟩, .operator (⟨274780, 17⟩, ⟨266003, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7232⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (1)⟩)

def event274787 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70981⟩⟩, .operator (⟨274780, 29⟩, ⟨266003, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨48256⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (-1)⟩)

def event274788 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨70981⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨48256⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨70979⟩⟩) ⟨68780⟩ 266000)

def event274789 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70981⟩⟩, .relation 274788 0, ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨48256⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (-1)⟩)

def event274790 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70981⟩⟩, .operator (⟨274780, 16⟩, ⟨266003, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (1)⟩)

def event274791 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70981⟩⟩, .operator (⟨274780, 28⟩, ⟨266003, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨45576⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (-1)⟩)

def event274792 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨70981⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨45576⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨70979⟩⟩) ⟨68780⟩ 266000)

def event274793 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70981⟩⟩, .relation 274792 0, ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨45576⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (-1)⟩)

def event274794 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70981⟩⟩, .operator (⟨274780, 15⟩, ⟨266003, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (1)⟩)

def event274795 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70981⟩⟩, .operator (⟨274780, 27⟩, ⟨266003, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨42892⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (-1)⟩)

def event274796 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨70981⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨42892⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨70979⟩⟩) ⟨68780⟩ 266000)

def event274797 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70981⟩⟩, .relation 274796 0, ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨42892⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (-1)⟩)

def event274798 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70981⟩⟩, .operator (⟨274780, 14⟩, ⟨266003, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (1)⟩)

def event274799 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70981⟩⟩, .operator (⟨274780, 26⟩, ⟨266003, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨40212⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (-1)⟩)

def event274800 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨70981⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨40212⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨70979⟩⟩) ⟨68780⟩ 266000)

def event274801 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70981⟩⟩, .relation 274800 0, ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨40212⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (-1)⟩)

def event274802 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70981⟩⟩, .operator (⟨274780, 13⟩, ⟨266003, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (1)⟩)

def event274803 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70981⟩⟩, .operator (⟨274780, 25⟩, ⟨266003, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨37536⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (-1)⟩)

def event274804 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨70981⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨37536⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨70979⟩⟩) ⟨68780⟩ 266000)

def event274805 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70981⟩⟩, .relation 274804 0, ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨37536⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (-1)⟩)

def event274806 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70981⟩⟩, .operator (⟨274780, 12⟩, ⟨266003, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7222⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (1)⟩)

def event274807 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70981⟩⟩, .operator (⟨274780, 24⟩, ⟨266003, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨34856⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (-1)⟩)

def event274808 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨70981⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨34856⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨70979⟩⟩) ⟨68780⟩ 266000)

def event274809 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70981⟩⟩, .relation 274808 0, ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨34856⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (-1)⟩)

def event274810 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70981⟩⟩, .operator (⟨274780, 11⟩, ⟨266003, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (1)⟩)

def event274811 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70981⟩⟩, .operator (⟨274780, 22⟩, ⟨266003, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨29192⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (-1)⟩)

def event274812 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨70981⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨29192⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨70979⟩⟩) ⟨68780⟩ 266000)

def event274813 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70981⟩⟩, .relation 274812 0, ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨29192⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (-1)⟩)

def event274814 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70981⟩⟩, .operator (⟨274780, 10⟩, ⟨266003, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (1)⟩)

def event274815 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70981⟩⟩, .operator (⟨274780, 21⟩, ⟨266003, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨26512⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (-1)⟩)

def event274816 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨70981⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨26512⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨70979⟩⟩) ⟨68780⟩ 266000)

def event274817 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70981⟩⟩, .relation 274816 0, ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨26512⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (-1)⟩)

def event274818 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70981⟩⟩, .operator (⟨274780, 9⟩, ⟨266003, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (1)⟩)

def event274819 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70981⟩⟩, .operator (⟨274780, 35⟩, ⟨266003, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨66019⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (-1)⟩)

def event274820 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨70981⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨66019⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨70979⟩⟩) ⟨68780⟩ 266000)

def event274821 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70981⟩⟩, .relation 274820 0, ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨66019⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (-1)⟩)

def event274822 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70981⟩⟩, .operator (⟨274780, 8⟩, ⟨266003, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (1)⟩)

def event274823 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70981⟩⟩, .operator (⟨274780, 34⟩, ⟨266003, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨62924⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (-1)⟩)

def event274824 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨70981⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨62924⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨70979⟩⟩) ⟨68780⟩ 266000)

def event274825 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70981⟩⟩, .relation 274824 0, ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨62924⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (-1)⟩)

def event274826 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70981⟩⟩, .operator (⟨274780, 7⟩, ⟨266003, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (1)⟩)

def event274827 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70981⟩⟩, .operator (⟨274780, 33⟩, ⟨266003, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨59944⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (-1)⟩)

def event274828 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨70981⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨59944⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨70979⟩⟩) ⟨68780⟩ 266000)

def event274829 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70981⟩⟩, .relation 274828 0, ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨59944⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (-1)⟩)

def event274830 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70981⟩⟩, .operator (⟨274780, 6⟩, ⟨266003, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (1)⟩)

def event274831 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70981⟩⟩, .operator (⟨274780, 32⟩, ⟨266003, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨56964⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (-1)⟩)

def event274832 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨70981⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨56964⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨70979⟩⟩) ⟨68780⟩ 266000)

def event274833 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70981⟩⟩, .relation 274832 0, ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨56964⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (-1)⟩)

def event274834 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70981⟩⟩, .operator (⟨274780, 5⟩, ⟨266003, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (1)⟩)

def event274835 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70981⟩⟩, .operator (⟨274780, 31⟩, ⟨266003, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨53984⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (-1)⟩)

def event274836 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨70981⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨53984⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨70979⟩⟩) ⟨68780⟩ 266000)

def event274837 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70981⟩⟩, .relation 274836 0, ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨53984⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (-1)⟩)

def event274838 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70981⟩⟩, .operator (⟨274780, 4⟩, ⟨266003, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (1)⟩)

def event274839 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70981⟩⟩, .operator (⟨274780, 30⟩, ⟨266003, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨51004⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (-1)⟩)

def event274840 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨70981⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨51004⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨70979⟩⟩) ⟨68780⟩ 266000)

def event274841 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70981⟩⟩, .relation 274840 0, ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨51004⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (-1)⟩)

def event274842 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70981⟩⟩, .operator (⟨274780, 3⟩, ⟨266003, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (1)⟩)

def event274843 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70981⟩⟩, .operator (⟨274780, 23⟩, ⟨266003, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨31949⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (-1)⟩)

def event274844 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨70981⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨31949⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨70979⟩⟩) ⟨68780⟩ 266000)

def event274845 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70981⟩⟩, .relation 274844 0, ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨31949⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (-1)⟩)

def event274846 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70981⟩⟩, .operator (⟨274780, 2⟩, ⟨266003, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (1)⟩)

def event274847 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70981⟩⟩, .operator (⟨274780, 20⟩, ⟨266003, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨21929⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (-1)⟩)

def event274848 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨70981⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨21929⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨70979⟩⟩) ⟨68780⟩ 266000)

def event274849 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70981⟩⟩, .relation 274848 0, ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨21929⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (-1)⟩)

def event274850 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70981⟩⟩, .operator (⟨274780, 1⟩, ⟨266003, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (1)⟩)

def event274851 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70981⟩⟩, .operator (⟨274780, 19⟩, ⟨266003, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨18709⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (-1)⟩)

def event274852 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨70981⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨18709⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨70979⟩⟩) ⟨68780⟩ 266000)

def event274853 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70981⟩⟩, .relation 274852 0, ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨18709⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (-1)⟩)

def event274854 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70981⟩⟩, .operator (⟨274780, 0⟩, ⟨266003, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (1)⟩)

def event274855 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70981⟩⟩, .operator (⟨274780, 18⟩, ⟨266003, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨15903⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (-1)⟩)

def event274856 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨70981⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨15903⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨70979⟩⟩) ⟨68780⟩ 266000)

def event274857 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70981⟩⟩, .relation 274856 0, ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨15903⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (-1)⟩)

def exact274858RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7222⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7232⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨15903⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨18709⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨21929⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨26512⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨29192⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨31949⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨34856⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨37536⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨40212⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨42892⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨45576⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨48256⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨51004⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨53984⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨56964⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨59944⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨62924⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨66019⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩, (-1)⟩]

theorem exact274858RawTermsValid :
    exact274858RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event274858 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70981⟩⟩) exact274858RawTerms .large 274783 (.finite 6221717896068416040249469304417135687106560) (some (274785))

def event274859 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68287⟩⟩) 0 ⟨66029⟩ 13292

def event274860 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨68287⟩⟩) (.authority (.relationPreimageSource ⟨95⟩))

def exact274861RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨68287⟩⟩]⟩, (1)⟩]

theorem exact274861RawTermsValid :
    exact274861RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event274861 : Event := .resultExact (⟨.program ⟨257⟩, ⟨68287⟩⟩) exact274861RawTerms (.finite 5647228698) 274860 .exactZero (none)

def event274862 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68289⟩⟩) 0 ⟨68287⟩ 274861

def event274863 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68289⟩⟩) 1 ⟨2370⟩ 4

def event274864 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨68289⟩⟩) (.scale (.predecessor 0 274862 .coefficient) (.value (.predecessor 1 274863 .coefficient)))

def exact274865RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨68287⟩⟩]⟩, (1)⟩]

theorem exact274865RawTermsValid :
    exact274865RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event274865 : Event := .resultExact (⟨.program ⟨257⟩, ⟨68289⟩⟩) exact274865RawTerms (.finite 5647228698) 274864 .exactZero (none)

def event274866 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68290⟩⟩) 0 ⟨5449⟩ 266120

def event274867 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68290⟩⟩) 1 ⟨68289⟩ 274865

def event274868 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨68290⟩⟩) (.product (.predecessor 0 274866 .coefficient) (.predecessor 1 274867 .coefficient) (⟨false, false, none, none, none⟩))

def event274869 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨68290⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨68287⟩⟩]⟩) [⟨.result 274861 .coefficient, false, none⟩])

def event274870 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨68290⟩⟩) (.product (.result 266120 .summary) (.transfer 274869) (⟨false, false, none, none, none⟩))

def event274871 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68290⟩⟩, .operator (⟨266120, 0⟩, ⟨274865, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68287⟩⟩]⟩, (1)⟩)

def event274872 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨68288⟩⟩)

def event274873 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event274874 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event274875 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨387⟩⟩) (.authority (.operator))

def event274876 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨387⟩⟩) (.finite 2)

def event274877 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event274878 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event274879 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event274880 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event274881 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 274880

def event274882 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 274878

def event274883 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 274881 .coefficient) (.value (.predecessor 1 274882 .coefficient)))

def event274884 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event274885 : Event := .predecessor (⟨.program ⟨257⟩, ⟨394⟩⟩) 0 ⟨392⟩ 274884

def event274886 : Event := .predecessor (⟨.program ⟨257⟩, ⟨394⟩⟩) 1 ⟨387⟩ 274876

def event274887 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨394⟩⟩) (.sum [.predecessor 0 274885 .coefficient, .predecessor 1 274886 .coefficient])

def event274888 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨394⟩⟩) (.finite 655342)

def event274889 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5445⟩⟩) 0 ⟨394⟩ 274888

def event274890 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5445⟩⟩) 1 ⟨5426⟩ 274874

def event274891 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5445⟩⟩) (.identity (.predecessor 1 274890 .coefficient))

def event274892 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5445⟩⟩) (.finite 655360)

def event274893 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47634⟩⟩) 0 ⟨5445⟩ 274892

def event274894 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47634⟩⟩) (.authority (.programFamilyFact))

def exact274895RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨47634⟩⟩], []⟩, (1)⟩]

theorem exact274895RawTermsValid :
    exact274895RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event274895 : Event := .resultExact (⟨.program ⟨257⟩, ⟨47634⟩⟩) exact274895RawTerms (.finite 60) 274894 .exactZero (none)

def event274896 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14956⟩⟩) 0 ⟨5445⟩ 274892

def event274897 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14956⟩⟩) (.authority (.programFamilyFact))

def exact274898RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14956⟩⟩], []⟩, (1)⟩]

theorem exact274898RawTermsValid :
    exact274898RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event274898 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14956⟩⟩) exact274898RawTerms (.finite 60) 274897 .exactZero (none)

def event274899 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47635⟩⟩) 0 ⟨14956⟩ 274898

def event274900 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47635⟩⟩) 1 ⟨47634⟩ 274895

def event274901 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47635⟩⟩) (.product (.predecessor 0 274899 .coefficient) (.predecessor 1 274900 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event274902 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47635⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨14956⟩⟩, ⟨.program ⟨257⟩, ⟨47634⟩⟩], []⟩) [⟨.result 274898 .coefficient, true, some 1⟩, ⟨.result 274895 .coefficient, true, some 1⟩])

def event274903 : Event := .survivorFold (1) 274902

def exact274904RawTerms : List Term := []

theorem exact274904RawTermsValid :
    exact274904RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event274904 : Event := .resultExact (⟨.program ⟨257⟩, ⟨47635⟩⟩) exact274904RawTerms (.finite 3600) 274901 (.finite 3600) (some (274902))

def event274905 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47636⟩⟩) 0 ⟨47635⟩ 274904

def event274906 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47636⟩⟩) (.identity (.predecessor 0 274905 .coefficient))

def event274907 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨47636⟩⟩) (.finite 3600)

def event274908 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48082⟩⟩) 0 ⟨47636⟩ 274907

def event274909 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48082⟩⟩) (.authority (.programFamilyFact))

def exact274910RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨48082⟩⟩], []⟩, (1)⟩]

theorem exact274910RawTermsValid :
    exact274910RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event274910 : Event := .resultExact (⟨.program ⟨257⟩, ⟨48082⟩⟩) exact274910RawTerms (.finite 60) 274909 .exactZero (none)

def event274911 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48083⟩⟩) 0 ⟨48082⟩ 274910

def event274912 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48083⟩⟩) (.identity (.predecessor 0 274911 .coefficient))

def event274913 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨48083⟩⟩) (.finite 60)

def event274914 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48256⟩⟩) 0 ⟨48083⟩ 274913

def event274915 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48256⟩⟩) (.authority (.programFamilyFact))

def exact274916RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨48256⟩⟩], []⟩, (1)⟩]

theorem exact274916RawTermsValid :
    exact274916RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event274916 : Event := .resultExact (⟨.program ⟨257⟩, ⟨48256⟩⟩) exact274916RawTerms (.finite 63) 274915 .exactZero (none)

def event274917 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44954⟩⟩) 0 ⟨5445⟩ 274892

def event274918 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44954⟩⟩) (.authority (.programFamilyFact))

def exact274919RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨44954⟩⟩], []⟩, (1)⟩]

theorem exact274919RawTermsValid :
    exact274919RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event274919 : Event := .resultExact (⟨.program ⟨257⟩, ⟨44954⟩⟩) exact274919RawTerms (.finite 58) 274918 .exactZero (none)

def event274920 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14656⟩⟩) 0 ⟨5445⟩ 274892

def event274921 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14656⟩⟩) (.authority (.programFamilyFact))

def exact274922RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14656⟩⟩], []⟩, (1)⟩]

theorem exact274922RawTermsValid :
    exact274922RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event274922 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14656⟩⟩) exact274922RawTerms (.finite 58) 274921 .exactZero (none)

def event274923 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44955⟩⟩) 0 ⟨14656⟩ 274922

def event274924 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44955⟩⟩) 1 ⟨44954⟩ 274919

def event274925 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44955⟩⟩) (.product (.predecessor 0 274923 .coefficient) (.predecessor 1 274924 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event274926 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44955⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨14656⟩⟩, ⟨.program ⟨257⟩, ⟨44954⟩⟩], []⟩) [⟨.result 274922 .coefficient, true, some 1⟩, ⟨.result 274919 .coefficient, true, some 1⟩])

def event274927 : Event := .survivorFold (1) 274926

def exact274928RawTerms : List Term := []

theorem exact274928RawTermsValid :
    exact274928RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event274928 : Event := .resultExact (⟨.program ⟨257⟩, ⟨44955⟩⟩) exact274928RawTerms (.finite 3364) 274925 (.finite 3364) (some (274926))

def event274929 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44956⟩⟩) 0 ⟨44955⟩ 274928

def event274930 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44956⟩⟩) (.identity (.predecessor 0 274929 .coefficient))

def event274931 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨44956⟩⟩) (.finite 3364)

def event274932 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45402⟩⟩) 0 ⟨44956⟩ 274931

def event274933 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45402⟩⟩) (.authority (.programFamilyFact))

def exact274934RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨45402⟩⟩], []⟩, (1)⟩]

theorem exact274934RawTermsValid :
    exact274934RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event274934 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45402⟩⟩) exact274934RawTerms (.finite 58) 274933 .exactZero (none)

def event274935 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45403⟩⟩) 0 ⟨45402⟩ 274934

def event274936 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45403⟩⟩) (.identity (.predecessor 0 274935 .coefficient))

def event274937 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨45403⟩⟩) (.finite 58)

def event274938 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45576⟩⟩) 0 ⟨45403⟩ 274937

def event274939 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45576⟩⟩) (.authority (.programFamilyFact))

def exact274940RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨45576⟩⟩], []⟩, (1)⟩]

theorem exact274940RawTermsValid :
    exact274940RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event274940 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45576⟩⟩) exact274940RawTerms (.finite 63) 274939 .exactZero (none)

def event274941 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42274⟩⟩) 0 ⟨5445⟩ 274892

def event274942 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42274⟩⟩) (.authority (.programFamilyFact))

def exact274943RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨42274⟩⟩], []⟩, (1)⟩]

theorem exact274943RawTermsValid :
    exact274943RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event274943 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42274⟩⟩) exact274943RawTerms (.finite 52) 274942 .exactZero (none)

def eventLeaf17168 : Array AnnotatedEvent := #[
  { event := event274688
    frameStart := 0 },
  { event := event274689
    frameStart := 0 },
  { event := event274690
    frameStart := 0 },
  { event := event274691
    frameStart := 0 },
  { event := event274692
    frameStart := 0 },
  { event := event274693
    frameStart := 0 },
  { event := event274694
    frameStart := 0 },
  { event := event274695
    frameStart := 0 },
  { event := event274696
    frameStart := 0 },
  { event := event274697
    frameStart := 0 },
  { event := event274698
    frameStart := 0 },
  { event := event274699
    frameStart := 0 },
  { event := event274700
    frameStart := 0 },
  { event := event274701
    frameStart := 0 },
  { event := event274702
    frameStart := 0 },
  { event := event274703
    frameStart := 0 }
]

def eventLeaf17169 : Array AnnotatedEvent := #[
  { event := event274704
    frameStart := 0 },
  { event := event274705
    frameStart := 0 },
  { event := event274706
    frameStart := 0 },
  { event := event274707
    frameStart := 0 },
  { event := event274708
    frameStart := 0 },
  { event := event274709
    frameStart := 0 },
  { event := event274710
    frameStart := 0 },
  { event := event274711
    frameStart := 0 },
  { event := event274712
    frameStart := 0 },
  { event := event274713
    frameStart := 0 },
  { event := event274714
    frameStart := 0 },
  { event := event274715
    frameStart := 0 },
  { event := event274716
    frameStart := 0 },
  { event := event274717
    frameStart := 0 },
  { event := event274718
    frameStart := 0 },
  { event := event274719
    frameStart := 0 }
]

def eventLeaf17170 : Array AnnotatedEvent := #[
  { event := event274720
    frameStart := 0 },
  { event := event274721
    frameStart := 0 },
  { event := event274722
    frameStart := 0 },
  { event := event274723
    frameStart := 0 },
  { event := event274724
    frameStart := 0 },
  { event := event274725
    frameStart := 0 },
  { event := event274726
    frameStart := 0 },
  { event := event274727
    frameStart := 0 },
  { event := event274728
    frameStart := 0 },
  { event := event274729
    frameStart := 0 },
  { event := event274730
    frameStart := 0 },
  { event := event274731
    frameStart := 0 },
  { event := event274732
    frameStart := 0 },
  { event := event274733
    frameStart := 0 },
  { event := event274734
    frameStart := 0 },
  { event := event274735
    frameStart := 0 }
]

def eventLeaf17171 : Array AnnotatedEvent := #[
  { event := event274736
    frameStart := 0 },
  { event := event274737
    frameStart := 0 },
  { event := event274738
    frameStart := 0 },
  { event := event274739
    frameStart := 0 },
  { event := event274740
    frameStart := 0 },
  { event := event274741
    frameStart := 0 },
  { event := event274742
    frameStart := 0 },
  { event := event274743
    frameStart := 0 },
  { event := event274744
    frameStart := 0 },
  { event := event274745
    frameStart := 0 },
  { event := event274746
    frameStart := 0 },
  { event := event274747
    frameStart := 0 },
  { event := event274748
    frameStart := 0 },
  { event := event274749
    frameStart := 0 },
  { event := event274750
    frameStart := 0 },
  { event := event274751
    frameStart := 0 }
]

def eventLeaf17172 : Array AnnotatedEvent := #[
  { event := event274752
    frameStart := 0 },
  { event := event274753
    frameStart := 0 },
  { event := event274754
    frameStart := 0 },
  { event := event274755
    frameStart := 0 },
  { event := event274756
    frameStart := 0 },
  { event := event274757
    frameStart := 0 },
  { event := event274758
    frameStart := 0 },
  { event := event274759
    frameStart := 0 },
  { event := event274760
    frameStart := 0 },
  { event := event274761
    frameStart := 0 },
  { event := event274762
    frameStart := 0 },
  { event := event274763
    frameStart := 0 },
  { event := event274764
    frameStart := 0 },
  { event := event274765
    frameStart := 0 },
  { event := event274766
    frameStart := 0 },
  { event := event274767
    frameStart := 0 }
]

def eventLeaf17173 : Array AnnotatedEvent := #[
  { event := event274768
    frameStart := 0 },
  { event := event274769
    frameStart := 0 },
  { event := event274770
    frameStart := 0 },
  { event := event274771
    frameStart := 0 },
  { event := event274772
    frameStart := 0 },
  { event := event274773
    frameStart := 0 },
  { event := event274774
    frameStart := 0 },
  { event := event274775
    frameStart := 0 },
  { event := event274776
    frameStart := 0 },
  { event := event274777
    frameStart := 0 },
  { event := event274778
    frameStart := 0 },
  { event := event274779
    frameStart := 0 },
  { event := event274780
    frameStart := 0 },
  { event := event274781
    frameStart := 0 },
  { event := event274782
    frameStart := 0 },
  { event := event274783
    frameStart := 0 }
]

def eventLeaf17174 : Array AnnotatedEvent := #[
  { event := event274784
    frameStart := 0 },
  { event := event274785
    frameStart := 0 },
  { event := event274786
    frameStart := 0 },
  { event := event274787
    frameStart := 0 },
  { event := event274788
    frameStart := 0 },
  { event := event274789
    frameStart := 0 },
  { event := event274790
    frameStart := 0 },
  { event := event274791
    frameStart := 0 },
  { event := event274792
    frameStart := 0 },
  { event := event274793
    frameStart := 0 },
  { event := event274794
    frameStart := 0 },
  { event := event274795
    frameStart := 0 },
  { event := event274796
    frameStart := 0 },
  { event := event274797
    frameStart := 0 },
  { event := event274798
    frameStart := 0 },
  { event := event274799
    frameStart := 0 }
]

def eventLeaf17175 : Array AnnotatedEvent := #[
  { event := event274800
    frameStart := 0 },
  { event := event274801
    frameStart := 0 },
  { event := event274802
    frameStart := 0 },
  { event := event274803
    frameStart := 0 },
  { event := event274804
    frameStart := 0 },
  { event := event274805
    frameStart := 0 },
  { event := event274806
    frameStart := 0 },
  { event := event274807
    frameStart := 0 },
  { event := event274808
    frameStart := 0 },
  { event := event274809
    frameStart := 0 },
  { event := event274810
    frameStart := 0 },
  { event := event274811
    frameStart := 0 },
  { event := event274812
    frameStart := 0 },
  { event := event274813
    frameStart := 0 },
  { event := event274814
    frameStart := 0 },
  { event := event274815
    frameStart := 0 }
]

def eventLeaf17176 : Array AnnotatedEvent := #[
  { event := event274816
    frameStart := 0 },
  { event := event274817
    frameStart := 0 },
  { event := event274818
    frameStart := 0 },
  { event := event274819
    frameStart := 0 },
  { event := event274820
    frameStart := 0 },
  { event := event274821
    frameStart := 0 },
  { event := event274822
    frameStart := 0 },
  { event := event274823
    frameStart := 0 },
  { event := event274824
    frameStart := 0 },
  { event := event274825
    frameStart := 0 },
  { event := event274826
    frameStart := 0 },
  { event := event274827
    frameStart := 0 },
  { event := event274828
    frameStart := 0 },
  { event := event274829
    frameStart := 0 },
  { event := event274830
    frameStart := 0 },
  { event := event274831
    frameStart := 0 }
]

def eventLeaf17177 : Array AnnotatedEvent := #[
  { event := event274832
    frameStart := 0 },
  { event := event274833
    frameStart := 0 },
  { event := event274834
    frameStart := 0 },
  { event := event274835
    frameStart := 0 },
  { event := event274836
    frameStart := 0 },
  { event := event274837
    frameStart := 0 },
  { event := event274838
    frameStart := 0 },
  { event := event274839
    frameStart := 0 },
  { event := event274840
    frameStart := 0 },
  { event := event274841
    frameStart := 0 },
  { event := event274842
    frameStart := 0 },
  { event := event274843
    frameStart := 0 },
  { event := event274844
    frameStart := 0 },
  { event := event274845
    frameStart := 0 },
  { event := event274846
    frameStart := 0 },
  { event := event274847
    frameStart := 0 }
]

def eventLeaf17178 : Array AnnotatedEvent := #[
  { event := event274848
    frameStart := 0 },
  { event := event274849
    frameStart := 0 },
  { event := event274850
    frameStart := 0 },
  { event := event274851
    frameStart := 0 },
  { event := event274852
    frameStart := 0 },
  { event := event274853
    frameStart := 0 },
  { event := event274854
    frameStart := 0 },
  { event := event274855
    frameStart := 0 },
  { event := event274856
    frameStart := 0 },
  { event := event274857
    frameStart := 0 },
  { event := event274858
    frameStart := 0 },
  { event := event274859
    frameStart := 0 },
  { event := event274860
    frameStart := 0 },
  { event := event274861
    frameStart := 0 },
  { event := event274862
    frameStart := 0 },
  { event := event274863
    frameStart := 0 }
]

def eventLeaf17179 : Array AnnotatedEvent := #[
  { event := event274864
    frameStart := 0 },
  { event := event274865
    frameStart := 0 },
  { event := event274866
    frameStart := 0 },
  { event := event274867
    frameStart := 0 },
  { event := event274868
    frameStart := 0 },
  { event := event274869
    frameStart := 0 },
  { event := event274870
    frameStart := 0 },
  { event := event274871
    frameStart := 0 },
  { event := event274872
    frameStart := 274872 },
  { event := event274873
    frameStart := 274872 },
  { event := event274874
    frameStart := 274872 },
  { event := event274875
    frameStart := 274872 },
  { event := event274876
    frameStart := 274872 },
  { event := event274877
    frameStart := 274872 },
  { event := event274878
    frameStart := 274872 },
  { event := event274879
    frameStart := 274872 }
]

def eventLeaf17180 : Array AnnotatedEvent := #[
  { event := event274880
    frameStart := 274872 },
  { event := event274881
    frameStart := 274872 },
  { event := event274882
    frameStart := 274872 },
  { event := event274883
    frameStart := 274872 },
  { event := event274884
    frameStart := 274872 },
  { event := event274885
    frameStart := 274872 },
  { event := event274886
    frameStart := 274872 },
  { event := event274887
    frameStart := 274872 },
  { event := event274888
    frameStart := 274872 },
  { event := event274889
    frameStart := 274872 },
  { event := event274890
    frameStart := 274872 },
  { event := event274891
    frameStart := 274872 },
  { event := event274892
    frameStart := 274872 },
  { event := event274893
    frameStart := 274872 },
  { event := event274894
    frameStart := 274872 },
  { event := event274895
    frameStart := 274872 }
]

def eventLeaf17181 : Array AnnotatedEvent := #[
  { event := event274896
    frameStart := 274872 },
  { event := event274897
    frameStart := 274872 },
  { event := event274898
    frameStart := 274872 },
  { event := event274899
    frameStart := 274872 },
  { event := event274900
    frameStart := 274872 },
  { event := event274901
    frameStart := 274872 },
  { event := event274902
    frameStart := 274872 },
  { event := event274903
    frameStart := 274872 },
  { event := event274904
    frameStart := 274872 },
  { event := event274905
    frameStart := 274872 },
  { event := event274906
    frameStart := 274872 },
  { event := event274907
    frameStart := 274872 },
  { event := event274908
    frameStart := 274872 },
  { event := event274909
    frameStart := 274872 },
  { event := event274910
    frameStart := 274872 },
  { event := event274911
    frameStart := 274872 }
]

def eventLeaf17182 : Array AnnotatedEvent := #[
  { event := event274912
    frameStart := 274872 },
  { event := event274913
    frameStart := 274872 },
  { event := event274914
    frameStart := 274872 },
  { event := event274915
    frameStart := 274872 },
  { event := event274916
    frameStart := 274872 },
  { event := event274917
    frameStart := 274872 },
  { event := event274918
    frameStart := 274872 },
  { event := event274919
    frameStart := 274872 },
  { event := event274920
    frameStart := 274872 },
  { event := event274921
    frameStart := 274872 },
  { event := event274922
    frameStart := 274872 },
  { event := event274923
    frameStart := 274872 },
  { event := event274924
    frameStart := 274872 },
  { event := event274925
    frameStart := 274872 },
  { event := event274926
    frameStart := 274872 },
  { event := event274927
    frameStart := 274872 }
]

def eventLeaf17183 : Array AnnotatedEvent := #[
  { event := event274928
    frameStart := 274872 },
  { event := event274929
    frameStart := 274872 },
  { event := event274930
    frameStart := 274872 },
  { event := event274931
    frameStart := 274872 },
  { event := event274932
    frameStart := 274872 },
  { event := event274933
    frameStart := 274872 },
  { event := event274934
    frameStart := 274872 },
  { event := event274935
    frameStart := 274872 },
  { event := event274936
    frameStart := 274872 },
  { event := event274937
    frameStart := 274872 },
  { event := event274938
    frameStart := 274872 },
  { event := event274939
    frameStart := 274872 },
  { event := event274940
    frameStart := 274872 },
  { event := event274941
    frameStart := 274872 },
  { event := event274942
    frameStart := 274872 },
  { event := event274943
    frameStart := 274872 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events1073
