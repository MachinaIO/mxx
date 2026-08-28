import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events1136

open Mxx.Certificate.OperationalNoise
open CertificateABI

def exact290816RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7233⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨67341⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact290816RawTermsValid :
    exact290816RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event290816 : Event := .resultExact (⟨.program ⟨257⟩, ⟨67349⟩⟩) exact290816RawTerms .large 290815 .exactZero (none)

def event290817 : Event := .predecessor (⟨.program ⟨257⟩, ⟨71053⟩⟩) 0 ⟨67349⟩ 290816

def event290818 : Event := .predecessor (⟨.program ⟨257⟩, ⟨71053⟩⟩) 1 ⟨71049⟩ 290801

def event290819 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨71053⟩⟩) (.sum [.predecessor 0 290817 .coefficient, .predecessor 1 290818 .coefficient])

def exact290820RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7214⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7216⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7218⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7220⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7222⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7232⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7233⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15939⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨18752⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨21972⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨26541⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨29221⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨31992⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨34885⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨37565⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨40241⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨42921⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨45605⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨48285⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨51047⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨54027⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨57007⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨59987⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨62967⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨66181⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨67341⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact290820RawTermsValid :
    exact290820RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event290820 : Event := .resultExact (⟨.program ⟨257⟩, ⟨71053⟩⟩) exact290820RawTerms .large 290819 .exactZero (none)

def event290821 : Event := .preFoldPolynomial 290820 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7214⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7216⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7218⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7220⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7222⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7232⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7233⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15939⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨18752⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨21972⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨26541⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨29221⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨31992⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨34885⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨37565⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨40241⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨42921⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨45605⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨48285⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨51047⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨54027⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨57007⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨59987⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨62967⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨66181⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨67341⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact290822RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7214⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7216⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7218⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7220⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7222⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7232⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7233⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15939⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨18752⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨21972⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨26541⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨29221⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨31992⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨34885⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨37565⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨40241⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨42921⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨45605⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨48285⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨51047⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨54027⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨57007⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨59987⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨62967⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨66181⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨67341⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event290822 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨71053⟩⟩) 290821 exact290822RawTerms .large 290819 .exactZero (none)

def event290823 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨66191⟩⟩) ⟨⟨1⟩, ⟨95⟩, ⟨135⟩⟩ ⟨289461, 290823⟩

def event290824 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨68313⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68310⟩⟩]⟩) (1) 0 2 (.universal 290823 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68310⟩⟩]⟩) (none) 290822)

def event290825 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68313⟩⟩, .relation 290824 18, ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7233⟩⟩]⟩, (1)⟩)

def event290826 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68313⟩⟩, .relation 290824 17, ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7232⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (-1)⟩)

def event290827 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68313⟩⟩, .relation 290824 16, ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (-1)⟩)

def event290828 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68313⟩⟩, .relation 290824 15, ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (-1)⟩)

def event290829 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68313⟩⟩, .relation 290824 14, ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (-1)⟩)

def event290830 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68313⟩⟩, .relation 290824 13, ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (-1)⟩)

def event290831 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68313⟩⟩, .relation 290824 12, ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7222⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (-1)⟩)

def event290832 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68313⟩⟩, .relation 290824 11, ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (-1)⟩)

def event290833 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68313⟩⟩, .relation 290824 10, ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (-1)⟩)

def event290834 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68313⟩⟩, .relation 290824 9, ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (-1)⟩)

def event290835 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68313⟩⟩, .relation 290824 8, ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (-1)⟩)

def event290836 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68313⟩⟩, .relation 290824 7, ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (-1)⟩)

def event290837 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68313⟩⟩, .relation 290824 6, ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (-1)⟩)

def event290838 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68313⟩⟩, .relation 290824 5, ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (-1)⟩)

def event290839 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68313⟩⟩, .relation 290824 4, ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (-1)⟩)

def event290840 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68313⟩⟩, .relation 290824 3, ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (-1)⟩)

def event290841 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68313⟩⟩, .relation 290824 2, ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (-1)⟩)

def event290842 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68313⟩⟩, .relation 290824 1, ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (-1)⟩)

def event290843 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68313⟩⟩, .relation 290824 0, ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (-1)⟩)

def event290844 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68313⟩⟩, .relation 290824 30, ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨48285⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (1)⟩)

def event290845 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68313⟩⟩, .relation 290824 29, ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨45605⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (1)⟩)

def event290846 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68313⟩⟩, .relation 290824 28, ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨42921⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (1)⟩)

def event290847 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68313⟩⟩, .relation 290824 27, ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨40241⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (1)⟩)

def event290848 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68313⟩⟩, .relation 290824 26, ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨37565⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (1)⟩)

def event290849 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68313⟩⟩, .relation 290824 25, ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨34885⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (1)⟩)

def event290850 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68313⟩⟩, .relation 290824 23, ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨29221⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (1)⟩)

def event290851 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68313⟩⟩, .relation 290824 22, ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨26541⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (1)⟩)

def event290852 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68313⟩⟩, .relation 290824 36, ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨66181⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (1)⟩)

def event290853 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68313⟩⟩, .relation 290824 35, ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨62967⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (1)⟩)

def event290854 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68313⟩⟩, .relation 290824 34, ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨59987⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (1)⟩)

def event290855 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68313⟩⟩, .relation 290824 33, ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨57007⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (1)⟩)

def event290856 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68313⟩⟩, .relation 290824 32, ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨54027⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (1)⟩)

def event290857 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68313⟩⟩, .relation 290824 31, ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨51047⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (1)⟩)

def event290858 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68313⟩⟩, .relation 290824 24, ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨31992⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (1)⟩)

def event290859 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68313⟩⟩, .relation 290824 21, ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨21972⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (1)⟩)

def event290860 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68313⟩⟩, .relation 290824 20, ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨18752⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (1)⟩)

def event290861 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68313⟩⟩, .relation 290824 19, ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨15939⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (1)⟩)

def event290862 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68313⟩⟩, .relation 290824 37, ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨67341⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact290863RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7222⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7232⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7233⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨15939⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨18752⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨21972⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨26541⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨29221⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨31992⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨34885⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨37565⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨40241⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨42921⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨45605⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨48285⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨51047⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨54027⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨57007⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨59987⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨62967⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨66181⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨67341⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact290863RawTermsValid :
    exact290863RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event290863 : Event := .resultExact (⟨.program ⟨257⟩, ⟨68313⟩⟩) exact290863RawTerms .large 289457 (.finite 202072841853861888) (some (289459))

def event290864 : Event := .predecessor (⟨.program ⟨257⟩, ⟨71051⟩⟩) 0 ⟨68313⟩ 290863

def event290865 : Event := .predecessor (⟨.program ⟨257⟩, ⟨71051⟩⟩) 1 ⟨71050⟩ 289447

def event290866 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨71051⟩⟩) (.sum [.predecessor 0 290864 .coefficient, .predecessor 1 290865 .coefficient])

def event290867 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71051⟩⟩, .operator (⟨290863, 17⟩, ⟨289447, 17⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7232⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (1)⟩)

def event290868 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71051⟩⟩, .operator (⟨290863, 30⟩, ⟨289447, 29⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨48285⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (-1)⟩)

def event290869 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71051⟩⟩, .operator (⟨290863, 16⟩, ⟨289447, 16⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (1)⟩)

def event290870 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71051⟩⟩, .operator (⟨290863, 29⟩, ⟨289447, 28⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨45605⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (-1)⟩)

def event290871 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71051⟩⟩, .operator (⟨290863, 15⟩, ⟨289447, 15⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (1)⟩)

def event290872 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71051⟩⟩, .operator (⟨290863, 28⟩, ⟨289447, 27⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨42921⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (-1)⟩)

def event290873 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71051⟩⟩, .operator (⟨290863, 14⟩, ⟨289447, 14⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (1)⟩)

def event290874 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71051⟩⟩, .operator (⟨290863, 27⟩, ⟨289447, 26⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨40241⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (-1)⟩)

def event290875 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71051⟩⟩, .operator (⟨290863, 13⟩, ⟨289447, 13⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (1)⟩)

def event290876 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71051⟩⟩, .operator (⟨290863, 26⟩, ⟨289447, 25⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨37565⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (-1)⟩)

def event290877 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71051⟩⟩, .operator (⟨290863, 12⟩, ⟨289447, 12⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7222⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (1)⟩)

def event290878 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71051⟩⟩, .operator (⟨290863, 25⟩, ⟨289447, 24⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨34885⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (-1)⟩)

def event290879 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71051⟩⟩, .operator (⟨290863, 11⟩, ⟨289447, 11⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (1)⟩)

def event290880 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71051⟩⟩, .operator (⟨290863, 23⟩, ⟨289447, 22⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨29221⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (-1)⟩)

def event290881 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71051⟩⟩, .operator (⟨290863, 10⟩, ⟨289447, 10⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (1)⟩)

def event290882 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71051⟩⟩, .operator (⟨290863, 22⟩, ⟨289447, 21⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨26541⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (-1)⟩)

def event290883 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71051⟩⟩, .operator (⟨290863, 9⟩, ⟨289447, 9⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (1)⟩)

def event290884 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71051⟩⟩, .operator (⟨290863, 36⟩, ⟨289447, 35⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨66181⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (-1)⟩)

def event290885 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71051⟩⟩, .operator (⟨290863, 8⟩, ⟨289447, 8⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (1)⟩)

def event290886 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71051⟩⟩, .operator (⟨290863, 35⟩, ⟨289447, 34⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨62967⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (-1)⟩)

def event290887 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71051⟩⟩, .operator (⟨290863, 7⟩, ⟨289447, 7⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (1)⟩)

def event290888 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71051⟩⟩, .operator (⟨290863, 34⟩, ⟨289447, 33⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨59987⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (-1)⟩)

def event290889 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71051⟩⟩, .operator (⟨290863, 6⟩, ⟨289447, 6⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (1)⟩)

def event290890 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71051⟩⟩, .operator (⟨290863, 33⟩, ⟨289447, 32⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨57007⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (-1)⟩)

def event290891 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71051⟩⟩, .operator (⟨290863, 5⟩, ⟨289447, 5⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (1)⟩)

def event290892 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71051⟩⟩, .operator (⟨290863, 32⟩, ⟨289447, 31⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨54027⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (-1)⟩)

def event290893 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71051⟩⟩, .operator (⟨290863, 4⟩, ⟨289447, 4⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (1)⟩)

def event290894 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71051⟩⟩, .operator (⟨290863, 31⟩, ⟨289447, 30⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨51047⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (-1)⟩)

def event290895 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71051⟩⟩, .operator (⟨290863, 3⟩, ⟨289447, 3⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (1)⟩)

def event290896 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71051⟩⟩, .operator (⟨290863, 24⟩, ⟨289447, 23⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨31992⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (-1)⟩)

def event290897 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71051⟩⟩, .operator (⟨290863, 2⟩, ⟨289447, 2⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (1)⟩)

def event290898 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71051⟩⟩, .operator (⟨290863, 21⟩, ⟨289447, 20⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨21972⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (-1)⟩)

def event290899 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71051⟩⟩, .operator (⟨290863, 1⟩, ⟨289447, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (1)⟩)

def event290900 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71051⟩⟩, .operator (⟨290863, 20⟩, ⟨289447, 19⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨18752⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (-1)⟩)

def event290901 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71051⟩⟩, .operator (⟨290863, 0⟩, ⟨289447, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩, (1)⟩)

def event290902 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71051⟩⟩, .operator (⟨290863, 19⟩, ⟨289447, 18⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨15939⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩, (-1)⟩)

def event290903 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨71051⟩⟩) (.sum [.result 290863 .summary, .result 289447 .summary])

def exact290904RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7233⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨67341⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact290904RawTermsValid :
    exact290904RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event290904 : Event := .resultExact (⟨.program ⟨257⟩, ⟨71051⟩⟩) exact290904RawTerms .large 290866 (.finite 6221717896068416040249469506489977540968448) (some (290903))

def event290905 : Event := .predecessor (⟨.program ⟨257⟩, ⟨71052⟩⟩) 0 ⟨71051⟩ 290904

def event290906 : Event := .predecessor (⟨.program ⟨257⟩, ⟨71052⟩⟩) 1 ⟨7140⟩ 15522

def event290907 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨71052⟩⟩) (.product (.predecessor 0 290905 .coefficient) (.predecessor 1 290906 .coefficient) (⟨false, false, none, none, none⟩))

def event290908 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨71052⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨7139⟩⟩]⟩) [⟨.result 15518 .coefficient, false, none⟩])

def event290909 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨71052⟩⟩) (.product (.result 290904 .summary) (.transfer 290908) (⟨false, false, none, none, none⟩))

def event290910 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71052⟩⟩, .operator (⟨290904, 0⟩, ⟨15522, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7233⟩⟩, ⟨.program ⟨257⟩, ⟨7139⟩⟩]⟩, (1)⟩)

def event290911 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71052⟩⟩, .operator (⟨290904, 1⟩, ⟨15522, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨67341⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7139⟩⟩]⟩, (-1)⟩)

def event290912 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71052⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨67341⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7139⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨7139⟩⟩) ⟨7035⟩ 15515)

def event290913 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71052⟩⟩, .relation 290912 0, ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67341⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact290914RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7233⟩⟩, ⟨.program ⟨257⟩, ⟨7139⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67341⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact290914RawTermsValid :
    exact290914RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event290914 : Event := .resultExact (⟨.program ⟨257⟩, ⟨71052⟩⟩) exact290914RawTerms .large 290907 (.finite 66805187221379434678483228029309283225584960819691520) (some (290909))

def event290915 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49246⟩⟩) 0 ⟨7177⟩ 15500

def event290916 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49246⟩⟩) 1 ⟨49245⟩ 280631

def event290917 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49246⟩⟩) (.authority (.operator))

def exact290918RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨49246⟩⟩]⟩, (1)⟩]

theorem exact290918RawTermsValid :
    exact290918RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event290918 : Event := .resultExact (⟨.program ⟨257⟩, ⟨49246⟩⟩) exact290918RawTerms .large 290917 .exactZero (none)

def event290919 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49873⟩⟩) 0 ⟨49246⟩ 290918

def event290920 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49873⟩⟩) (.authority (.operator))

def exact290921RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨49873⟩⟩]⟩, (1)⟩]

theorem exact290921RawTermsValid :
    exact290921RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event290921 : Event := .resultExact (⟨.program ⟨257⟩, ⟨49873⟩⟩) exact290921RawTerms (.finite 8192) 290920 .exactZero (none)

def event290922 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49875⟩⟩) 0 ⟨49595⟩ 280929

def event290923 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49875⟩⟩) 1 ⟨49873⟩ 290921

def event290924 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49875⟩⟩) (.product (.predecessor 0 290922 .coefficient) (.predecessor 1 290923 .coefficient) (⟨false, false, none, none, none⟩))

def event290925 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49875⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨49873⟩⟩]⟩) [⟨.result 290921 .coefficient, false, none⟩])

def event290926 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49875⟩⟩) (.product (.result 280929 .summary) (.transfer 290925) (⟨false, false, none, none, none⟩))

def event290927 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨49875⟩⟩, .operator (⟨280929, 0⟩, ⟨290921, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨49873⟩⟩]⟩, (1)⟩)

def event290928 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨49875⟩⟩, .operator (⟨280929, 1⟩, ⟨290921, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨48100⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨49873⟩⟩]⟩, (-1)⟩)

def event290929 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨49875⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨48100⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨49873⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨49873⟩⟩) ⟨49246⟩ 290918)

def event290930 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨49875⟩⟩, .relation 290929 0, ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨48100⟩⟩], [⟨.program ⟨257⟩, ⟨49246⟩⟩]⟩, (-1)⟩)

def exact290931RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨49873⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨48100⟩⟩], [⟨.program ⟨257⟩, ⟨49246⟩⟩]⟩, (-1)⟩]

theorem exact290931RawTermsValid :
    exact290931RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event290931 : Event := .resultExact (⟨.program ⟨257⟩, ⟨49875⟩⟩) exact290931RawTerms .large 290924 (.finite 32194504275408438756654574469120) (some (290926))

def event290932 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48772⟩⟩) 0 ⟨48101⟩ 13569

def event290933 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48772⟩⟩) (.authority (.relationPreimageSource ⟨93⟩))

def exact290934RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨48772⟩⟩]⟩, (1)⟩]

theorem exact290934RawTermsValid :
    exact290934RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event290934 : Event := .resultExact (⟨.program ⟨257⟩, ⟨48772⟩⟩) exact290934RawTerms (.finite 5647228698) 290933 .exactZero (none)

def event290935 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48774⟩⟩) 0 ⟨48772⟩ 290934

def event290936 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48774⟩⟩) 1 ⟨2370⟩ 4

def event290937 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48774⟩⟩) (.scale (.predecessor 0 290935 .coefficient) (.value (.predecessor 1 290936 .coefficient)))

def exact290938RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨48772⟩⟩]⟩, (1)⟩]

theorem exact290938RawTermsValid :
    exact290938RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event290938 : Event := .resultExact (⟨.program ⟨257⟩, ⟨48774⟩⟩) exact290938RawTerms (.finite 5647228698) 290937 .exactZero (none)

def event290939 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48775⟩⟩) 0 ⟨5491⟩ 280745

def event290940 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48775⟩⟩) 1 ⟨48774⟩ 290938

def event290941 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48775⟩⟩) (.product (.predecessor 0 290939 .coefficient) (.predecessor 1 290940 .coefficient) (⟨false, false, none, none, none⟩))

def event290942 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48775⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨48772⟩⟩]⟩) [⟨.result 290934 .coefficient, false, none⟩])

def event290943 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48775⟩⟩) (.product (.result 280745 .summary) (.transfer 290942) (⟨false, false, none, none, none⟩))

def event290944 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨48775⟩⟩, .operator (⟨280745, 0⟩, ⟨290938, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48772⟩⟩]⟩, (1)⟩)

def event290945 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨48773⟩⟩)

def event290946 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event290947 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event290948 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨2370⟩⟩) (.authority (.operator))

def event290949 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨2370⟩⟩) (.finite 1)

def event290950 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event290951 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event290952 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event290953 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event290954 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 290953

def event290955 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 290951

def event290956 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 290954 .coefficient) (.value (.predecessor 1 290955 .coefficient)))

def event290957 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event290958 : Event := .predecessor (⟨.program ⟨257⟩, ⟨2376⟩⟩) 0 ⟨392⟩ 290957

def event290959 : Event := .predecessor (⟨.program ⟨257⟩, ⟨2376⟩⟩) 1 ⟨2370⟩ 290949

def event290960 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨2376⟩⟩) (.sum [.predecessor 0 290958 .coefficient, .predecessor 1 290959 .coefficient])

def event290961 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨2376⟩⟩) (.finite 655341)

def event290962 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5487⟩⟩) 0 ⟨2376⟩ 290961

def event290963 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5487⟩⟩) 1 ⟨5426⟩ 290947

def event290964 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5487⟩⟩) (.identity (.predecessor 1 290963 .coefficient))

def event290965 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5487⟩⟩) (.finite 655360)

def event290966 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47690⟩⟩) 0 ⟨5487⟩ 290965

def event290967 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47690⟩⟩) (.authority (.programFamilyFact))

def exact290968RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨47690⟩⟩], []⟩, (1)⟩]

theorem exact290968RawTermsValid :
    exact290968RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event290968 : Event := .resultExact (⟨.program ⟨257⟩, ⟨47690⟩⟩) exact290968RawTerms (.finite 60) 290967 .exactZero (none)

def event290969 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14991⟩⟩) 0 ⟨5487⟩ 290965

def event290970 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14991⟩⟩) (.authority (.programFamilyFact))

def exact290971RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14991⟩⟩], []⟩, (1)⟩]

theorem exact290971RawTermsValid :
    exact290971RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event290971 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14991⟩⟩) exact290971RawTerms (.finite 60) 290970 .exactZero (none)

def event290972 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47691⟩⟩) 0 ⟨14991⟩ 290971

def event290973 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47691⟩⟩) 1 ⟨47690⟩ 290968

def event290974 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47691⟩⟩) (.product (.predecessor 0 290972 .coefficient) (.predecessor 1 290973 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event290975 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47691⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨14991⟩⟩, ⟨.program ⟨257⟩, ⟨47690⟩⟩], []⟩) [⟨.result 290971 .coefficient, true, some 1⟩, ⟨.result 290968 .coefficient, true, some 1⟩])

def event290976 : Event := .survivorFold (1) 290975

def exact290977RawTerms : List Term := []

theorem exact290977RawTermsValid :
    exact290977RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event290977 : Event := .resultExact (⟨.program ⟨257⟩, ⟨47691⟩⟩) exact290977RawTerms (.finite 3600) 290974 (.finite 3600) (some (290975))

def event290978 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47692⟩⟩) 0 ⟨47691⟩ 290977

def event290979 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47692⟩⟩) (.identity (.predecessor 0 290978 .coefficient))

def event290980 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨47692⟩⟩) (.finite 3600)

def event290981 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48100⟩⟩) 0 ⟨47692⟩ 290980

def event290982 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48100⟩⟩) (.authority (.programFamilyFact))

def exact290983RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨48100⟩⟩], []⟩, (1)⟩]

theorem exact290983RawTermsValid :
    exact290983RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event290983 : Event := .resultExact (⟨.program ⟨257⟩, ⟨48100⟩⟩) exact290983RawTerms (.finite 60) 290982 .exactZero (none)

def event290984 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48101⟩⟩) 0 ⟨48100⟩ 290983

def event290985 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48101⟩⟩) (.identity (.predecessor 0 290984 .coefficient))

def event290986 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨48101⟩⟩) (.finite 60)

def event290987 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48772⟩⟩) 0 ⟨48101⟩ 290986

def event290988 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48772⟩⟩) (.authority (.relationPreimageSource ⟨93⟩))

def exact290989RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨48772⟩⟩]⟩, (1)⟩]

theorem exact290989RawTermsValid :
    exact290989RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event290989 : Event := .resultExact (⟨.program ⟨257⟩, ⟨48772⟩⟩) exact290989RawTerms (.finite 5647228698) 290988 .exactZero (none)

def event290990 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35⟩⟩) (.authority (.operator))

def exact290991RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact290991RawTermsValid :
    exact290991RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event290991 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35⟩⟩) exact290991RawTerms .large 290990 .exactZero (none)

def event290992 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48773⟩⟩) 0 ⟨35⟩ 290991

def event290993 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48773⟩⟩) 1 ⟨48772⟩ 290989

def event290994 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48773⟩⟩) (.product (.predecessor 0 290992 .coefficient) (.predecessor 1 290993 .coefficient) (⟨false, false, none, none, none⟩))

def event290995 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨48773⟩⟩, .operator (⟨290991, 0⟩, ⟨290989, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48772⟩⟩]⟩, (1)⟩)

def exact290996RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48772⟩⟩]⟩, (1)⟩]

theorem exact290996RawTermsValid :
    exact290996RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event290996 : Event := .resultExact (⟨.program ⟨257⟩, ⟨48773⟩⟩) exact290996RawTerms .large 290994 .exactZero (none)

def event290997 : Event := .preFoldPolynomial 290996 [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48772⟩⟩]⟩, (1)⟩] .exactZero none

def exact290998RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48772⟩⟩]⟩, (1)⟩]

def event290998 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨48773⟩⟩) 290997 exact290998RawTerms .large 290994 .exactZero (none)

def event290999 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨49878⟩⟩)

def event291000 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event291001 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event291002 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨2370⟩⟩) (.authority (.operator))

def event291003 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨2370⟩⟩) (.finite 1)

def event291004 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event291005 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event291006 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event291007 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event291008 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 291007

def event291009 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 291005

def event291010 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 291008 .coefficient) (.value (.predecessor 1 291009 .coefficient)))

def event291011 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event291012 : Event := .predecessor (⟨.program ⟨257⟩, ⟨2376⟩⟩) 0 ⟨392⟩ 291011

def event291013 : Event := .predecessor (⟨.program ⟨257⟩, ⟨2376⟩⟩) 1 ⟨2370⟩ 291003

def event291014 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨2376⟩⟩) (.sum [.predecessor 0 291012 .coefficient, .predecessor 1 291013 .coefficient])

def event291015 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨2376⟩⟩) (.finite 655341)

def event291016 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5487⟩⟩) 0 ⟨2376⟩ 291015

def event291017 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5487⟩⟩) 1 ⟨5426⟩ 291001

def event291018 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5487⟩⟩) (.identity (.predecessor 1 291017 .coefficient))

def event291019 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5487⟩⟩) (.finite 655360)

def event291020 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47690⟩⟩) 0 ⟨5487⟩ 291019

def event291021 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47690⟩⟩) (.authority (.programFamilyFact))

def exact291022RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨47690⟩⟩], []⟩, (1)⟩]

theorem exact291022RawTermsValid :
    exact291022RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event291022 : Event := .resultExact (⟨.program ⟨257⟩, ⟨47690⟩⟩) exact291022RawTerms (.finite 60) 291021 .exactZero (none)

def event291023 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14991⟩⟩) 0 ⟨5487⟩ 291019

def event291024 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14991⟩⟩) (.authority (.programFamilyFact))

def exact291025RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14991⟩⟩], []⟩, (1)⟩]

theorem exact291025RawTermsValid :
    exact291025RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event291025 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14991⟩⟩) exact291025RawTerms (.finite 60) 291024 .exactZero (none)

def event291026 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47691⟩⟩) 0 ⟨14991⟩ 291025

def event291027 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47691⟩⟩) 1 ⟨47690⟩ 291022

def event291028 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47691⟩⟩) (.product (.predecessor 0 291026 .coefficient) (.predecessor 1 291027 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event291029 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨47691⟩⟩, .operator (⟨291025, 0⟩, ⟨291022, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨14991⟩⟩, ⟨.program ⟨257⟩, ⟨47690⟩⟩], []⟩, (1)⟩)

def exact291030RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14991⟩⟩, ⟨.program ⟨257⟩, ⟨47690⟩⟩], []⟩, (1)⟩]

theorem exact291030RawTermsValid :
    exact291030RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event291030 : Event := .resultExact (⟨.program ⟨257⟩, ⟨47691⟩⟩) exact291030RawTerms (.finite 3600) 291028 .exactZero (none)

def event291031 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47692⟩⟩) 0 ⟨47691⟩ 291030

def event291032 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47692⟩⟩) (.identity (.predecessor 0 291031 .coefficient))

def event291033 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨47692⟩⟩) (.finite 3600)

def event291034 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48100⟩⟩) 0 ⟨47692⟩ 291033

def event291035 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48100⟩⟩) (.authority (.programFamilyFact))

def exact291036RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨48100⟩⟩], []⟩, (1)⟩]

theorem exact291036RawTermsValid :
    exact291036RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event291036 : Event := .resultExact (⟨.program ⟨257⟩, ⟨48100⟩⟩) exact291036RawTerms (.finite 60) 291035 .exactZero (none)

def event291037 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48101⟩⟩) 0 ⟨48100⟩ 291036

def event291038 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48101⟩⟩) (.identity (.predecessor 0 291037 .coefficient))

def event291039 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨48101⟩⟩) (.finite 60)

def event291040 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49245⟩⟩) 0 ⟨48101⟩ 291039

def event291041 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49245⟩⟩) (.authority (.programFamilyFact))

def event291042 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨49245⟩⟩) (.finite 3720)

def event291043 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨7177⟩⟩) .missing

def event291044 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49246⟩⟩) 0 ⟨7177⟩ 291043

def event291045 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49246⟩⟩) 1 ⟨49245⟩ 291042

def event291046 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49246⟩⟩) (.authority (.operator))

def exact291047RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨49246⟩⟩]⟩, (1)⟩]

theorem exact291047RawTermsValid :
    exact291047RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event291047 : Event := .resultExact (⟨.program ⟨257⟩, ⟨49246⟩⟩) exact291047RawTerms .large 291046 .exactZero (none)

def event291048 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49873⟩⟩) 0 ⟨49246⟩ 291047

def event291049 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49873⟩⟩) (.authority (.operator))

def exact291050RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨49873⟩⟩]⟩, (1)⟩]

theorem exact291050RawTermsValid :
    exact291050RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event291050 : Event := .resultExact (⟨.program ⟨257⟩, ⟨49873⟩⟩) exact291050RawTerms (.finite 8192) 291049 .exactZero (none)

def event291051 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event291052 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event291053 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49482⟩⟩) 0 ⟨48101⟩ 291039

def event291054 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49482⟩⟩) 1 ⟨136⟩ 291052

def event291055 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49482⟩⟩) (.sum [.predecessor 0 291053 .coefficient, .predecessor 1 291054 .coefficient])

def event291056 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨49482⟩⟩) (.finite 60)

def event291057 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49483⟩⟩) 0 ⟨49482⟩ 291056

def event291058 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49483⟩⟩) (.identity (.predecessor 0 291057 .coefficient))

def exact291059RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨48100⟩⟩], []⟩, (1)⟩]

theorem exact291059RawTermsValid :
    exact291059RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event291059 : Event := .resultExact (⟨.program ⟨257⟩, ⟨49483⟩⟩) exact291059RawTerms (.finite 60) 291058 .exactZero (none)

def event291060 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact291061RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact291061RawTermsValid :
    exact291061RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event291061 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact291061RawTerms .large 291060 .exactZero (none)

def event291062 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49484⟩⟩) 0 ⟨6908⟩ 291061

def event291063 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49484⟩⟩) 1 ⟨49483⟩ 291059

def event291064 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49484⟩⟩) (.product (.predecessor 0 291062 .coefficient) (.predecessor 1 291063 .coefficient) (⟨false, false, none, none, none⟩))

def event291065 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨49484⟩⟩, .operator (⟨291061, 0⟩, ⟨291059, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨48100⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact291066RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨48100⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact291066RawTermsValid :
    exact291066RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event291066 : Event := .resultExact (⟨.program ⟨257⟩, ⟨49484⟩⟩) exact291066RawTerms .large 291064 .exactZero (none)

def event291067 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7196⟩⟩) 0 ⟨7177⟩ 291043

def event291068 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7196⟩⟩) (.authority (.operator))

def exact291069RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7196⟩⟩]⟩, (1)⟩]

theorem exact291069RawTermsValid :
    exact291069RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event291069 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7196⟩⟩) exact291069RawTerms .large 291068 .exactZero (none)

def event291070 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49485⟩⟩) 0 ⟨7196⟩ 291069

def event291071 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49485⟩⟩) 1 ⟨49484⟩ 291066

def eventLeaf18176 : Array AnnotatedEvent := #[
  { event := event290816
    frameStart := 290050 },
  { event := event290817
    frameStart := 290050 },
  { event := event290818
    frameStart := 290050 },
  { event := event290819
    frameStart := 290050 },
  { event := event290820
    frameStart := 290050 },
  { event := event290821
    frameStart := 290050 },
  { event := event290822
    frameStart := 290050 },
  { event := event290823
    frameStart := 0 },
  { event := event290824
    frameStart := 0 },
  { event := event290825
    frameStart := 0 },
  { event := event290826
    frameStart := 0 },
  { event := event290827
    frameStart := 0 },
  { event := event290828
    frameStart := 0 },
  { event := event290829
    frameStart := 0 },
  { event := event290830
    frameStart := 0 },
  { event := event290831
    frameStart := 0 }
]

def eventLeaf18177 : Array AnnotatedEvent := #[
  { event := event290832
    frameStart := 0 },
  { event := event290833
    frameStart := 0 },
  { event := event290834
    frameStart := 0 },
  { event := event290835
    frameStart := 0 },
  { event := event290836
    frameStart := 0 },
  { event := event290837
    frameStart := 0 },
  { event := event290838
    frameStart := 0 },
  { event := event290839
    frameStart := 0 },
  { event := event290840
    frameStart := 0 },
  { event := event290841
    frameStart := 0 },
  { event := event290842
    frameStart := 0 },
  { event := event290843
    frameStart := 0 },
  { event := event290844
    frameStart := 0 },
  { event := event290845
    frameStart := 0 },
  { event := event290846
    frameStart := 0 },
  { event := event290847
    frameStart := 0 }
]

def eventLeaf18178 : Array AnnotatedEvent := #[
  { event := event290848
    frameStart := 0 },
  { event := event290849
    frameStart := 0 },
  { event := event290850
    frameStart := 0 },
  { event := event290851
    frameStart := 0 },
  { event := event290852
    frameStart := 0 },
  { event := event290853
    frameStart := 0 },
  { event := event290854
    frameStart := 0 },
  { event := event290855
    frameStart := 0 },
  { event := event290856
    frameStart := 0 },
  { event := event290857
    frameStart := 0 },
  { event := event290858
    frameStart := 0 },
  { event := event290859
    frameStart := 0 },
  { event := event290860
    frameStart := 0 },
  { event := event290861
    frameStart := 0 },
  { event := event290862
    frameStart := 0 },
  { event := event290863
    frameStart := 0 }
]

def eventLeaf18179 : Array AnnotatedEvent := #[
  { event := event290864
    frameStart := 0 },
  { event := event290865
    frameStart := 0 },
  { event := event290866
    frameStart := 0 },
  { event := event290867
    frameStart := 0 },
  { event := event290868
    frameStart := 0 },
  { event := event290869
    frameStart := 0 },
  { event := event290870
    frameStart := 0 },
  { event := event290871
    frameStart := 0 },
  { event := event290872
    frameStart := 0 },
  { event := event290873
    frameStart := 0 },
  { event := event290874
    frameStart := 0 },
  { event := event290875
    frameStart := 0 },
  { event := event290876
    frameStart := 0 },
  { event := event290877
    frameStart := 0 },
  { event := event290878
    frameStart := 0 },
  { event := event290879
    frameStart := 0 }
]

def eventLeaf18180 : Array AnnotatedEvent := #[
  { event := event290880
    frameStart := 0 },
  { event := event290881
    frameStart := 0 },
  { event := event290882
    frameStart := 0 },
  { event := event290883
    frameStart := 0 },
  { event := event290884
    frameStart := 0 },
  { event := event290885
    frameStart := 0 },
  { event := event290886
    frameStart := 0 },
  { event := event290887
    frameStart := 0 },
  { event := event290888
    frameStart := 0 },
  { event := event290889
    frameStart := 0 },
  { event := event290890
    frameStart := 0 },
  { event := event290891
    frameStart := 0 },
  { event := event290892
    frameStart := 0 },
  { event := event290893
    frameStart := 0 },
  { event := event290894
    frameStart := 0 },
  { event := event290895
    frameStart := 0 }
]

def eventLeaf18181 : Array AnnotatedEvent := #[
  { event := event290896
    frameStart := 0 },
  { event := event290897
    frameStart := 0 },
  { event := event290898
    frameStart := 0 },
  { event := event290899
    frameStart := 0 },
  { event := event290900
    frameStart := 0 },
  { event := event290901
    frameStart := 0 },
  { event := event290902
    frameStart := 0 },
  { event := event290903
    frameStart := 0 },
  { event := event290904
    frameStart := 0 },
  { event := event290905
    frameStart := 0 },
  { event := event290906
    frameStart := 0 },
  { event := event290907
    frameStart := 0 },
  { event := event290908
    frameStart := 0 },
  { event := event290909
    frameStart := 0 },
  { event := event290910
    frameStart := 0 },
  { event := event290911
    frameStart := 0 }
]

def eventLeaf18182 : Array AnnotatedEvent := #[
  { event := event290912
    frameStart := 0 },
  { event := event290913
    frameStart := 0 },
  { event := event290914
    frameStart := 0 },
  { event := event290915
    frameStart := 0 },
  { event := event290916
    frameStart := 0 },
  { event := event290917
    frameStart := 0 },
  { event := event290918
    frameStart := 0 },
  { event := event290919
    frameStart := 0 },
  { event := event290920
    frameStart := 0 },
  { event := event290921
    frameStart := 0 },
  { event := event290922
    frameStart := 0 },
  { event := event290923
    frameStart := 0 },
  { event := event290924
    frameStart := 0 },
  { event := event290925
    frameStart := 0 },
  { event := event290926
    frameStart := 0 },
  { event := event290927
    frameStart := 0 }
]

def eventLeaf18183 : Array AnnotatedEvent := #[
  { event := event290928
    frameStart := 0 },
  { event := event290929
    frameStart := 0 },
  { event := event290930
    frameStart := 0 },
  { event := event290931
    frameStart := 0 },
  { event := event290932
    frameStart := 0 },
  { event := event290933
    frameStart := 0 },
  { event := event290934
    frameStart := 0 },
  { event := event290935
    frameStart := 0 },
  { event := event290936
    frameStart := 0 },
  { event := event290937
    frameStart := 0 },
  { event := event290938
    frameStart := 0 },
  { event := event290939
    frameStart := 0 },
  { event := event290940
    frameStart := 0 },
  { event := event290941
    frameStart := 0 },
  { event := event290942
    frameStart := 0 },
  { event := event290943
    frameStart := 0 }
]

def eventLeaf18184 : Array AnnotatedEvent := #[
  { event := event290944
    frameStart := 0 },
  { event := event290945
    frameStart := 290945 },
  { event := event290946
    frameStart := 290945 },
  { event := event290947
    frameStart := 290945 },
  { event := event290948
    frameStart := 290945 },
  { event := event290949
    frameStart := 290945 },
  { event := event290950
    frameStart := 290945 },
  { event := event290951
    frameStart := 290945 },
  { event := event290952
    frameStart := 290945 },
  { event := event290953
    frameStart := 290945 },
  { event := event290954
    frameStart := 290945 },
  { event := event290955
    frameStart := 290945 },
  { event := event290956
    frameStart := 290945 },
  { event := event290957
    frameStart := 290945 },
  { event := event290958
    frameStart := 290945 },
  { event := event290959
    frameStart := 290945 }
]

def eventLeaf18185 : Array AnnotatedEvent := #[
  { event := event290960
    frameStart := 290945 },
  { event := event290961
    frameStart := 290945 },
  { event := event290962
    frameStart := 290945 },
  { event := event290963
    frameStart := 290945 },
  { event := event290964
    frameStart := 290945 },
  { event := event290965
    frameStart := 290945 },
  { event := event290966
    frameStart := 290945 },
  { event := event290967
    frameStart := 290945 },
  { event := event290968
    frameStart := 290945 },
  { event := event290969
    frameStart := 290945 },
  { event := event290970
    frameStart := 290945 },
  { event := event290971
    frameStart := 290945 },
  { event := event290972
    frameStart := 290945 },
  { event := event290973
    frameStart := 290945 },
  { event := event290974
    frameStart := 290945 },
  { event := event290975
    frameStart := 290945 }
]

def eventLeaf18186 : Array AnnotatedEvent := #[
  { event := event290976
    frameStart := 290945 },
  { event := event290977
    frameStart := 290945 },
  { event := event290978
    frameStart := 290945 },
  { event := event290979
    frameStart := 290945 },
  { event := event290980
    frameStart := 290945 },
  { event := event290981
    frameStart := 290945 },
  { event := event290982
    frameStart := 290945 },
  { event := event290983
    frameStart := 290945 },
  { event := event290984
    frameStart := 290945 },
  { event := event290985
    frameStart := 290945 },
  { event := event290986
    frameStart := 290945 },
  { event := event290987
    frameStart := 290945 },
  { event := event290988
    frameStart := 290945 },
  { event := event290989
    frameStart := 290945 },
  { event := event290990
    frameStart := 290945 },
  { event := event290991
    frameStart := 290945 }
]

def eventLeaf18187 : Array AnnotatedEvent := #[
  { event := event290992
    frameStart := 290945 },
  { event := event290993
    frameStart := 290945 },
  { event := event290994
    frameStart := 290945 },
  { event := event290995
    frameStart := 290945 },
  { event := event290996
    frameStart := 290945 },
  { event := event290997
    frameStart := 290945 },
  { event := event290998
    frameStart := 290945 },
  { event := event290999
    frameStart := 290999 },
  { event := event291000
    frameStart := 290999 },
  { event := event291001
    frameStart := 290999 },
  { event := event291002
    frameStart := 290999 },
  { event := event291003
    frameStart := 290999 },
  { event := event291004
    frameStart := 290999 },
  { event := event291005
    frameStart := 290999 },
  { event := event291006
    frameStart := 290999 },
  { event := event291007
    frameStart := 290999 }
]

def eventLeaf18188 : Array AnnotatedEvent := #[
  { event := event291008
    frameStart := 290999 },
  { event := event291009
    frameStart := 290999 },
  { event := event291010
    frameStart := 290999 },
  { event := event291011
    frameStart := 290999 },
  { event := event291012
    frameStart := 290999 },
  { event := event291013
    frameStart := 290999 },
  { event := event291014
    frameStart := 290999 },
  { event := event291015
    frameStart := 290999 },
  { event := event291016
    frameStart := 290999 },
  { event := event291017
    frameStart := 290999 },
  { event := event291018
    frameStart := 290999 },
  { event := event291019
    frameStart := 290999 },
  { event := event291020
    frameStart := 290999 },
  { event := event291021
    frameStart := 290999 },
  { event := event291022
    frameStart := 290999 },
  { event := event291023
    frameStart := 290999 }
]

def eventLeaf18189 : Array AnnotatedEvent := #[
  { event := event291024
    frameStart := 290999 },
  { event := event291025
    frameStart := 290999 },
  { event := event291026
    frameStart := 290999 },
  { event := event291027
    frameStart := 290999 },
  { event := event291028
    frameStart := 290999 },
  { event := event291029
    frameStart := 290999 },
  { event := event291030
    frameStart := 290999 },
  { event := event291031
    frameStart := 290999 },
  { event := event291032
    frameStart := 290999 },
  { event := event291033
    frameStart := 290999 },
  { event := event291034
    frameStart := 290999 },
  { event := event291035
    frameStart := 290999 },
  { event := event291036
    frameStart := 290999 },
  { event := event291037
    frameStart := 290999 },
  { event := event291038
    frameStart := 290999 },
  { event := event291039
    frameStart := 290999 }
]

def eventLeaf18190 : Array AnnotatedEvent := #[
  { event := event291040
    frameStart := 290999 },
  { event := event291041
    frameStart := 290999 },
  { event := event291042
    frameStart := 290999 },
  { event := event291043
    frameStart := 290999 },
  { event := event291044
    frameStart := 290999 },
  { event := event291045
    frameStart := 290999 },
  { event := event291046
    frameStart := 290999 },
  { event := event291047
    frameStart := 290999 },
  { event := event291048
    frameStart := 290999 },
  { event := event291049
    frameStart := 290999 },
  { event := event291050
    frameStart := 290999 },
  { event := event291051
    frameStart := 290999 },
  { event := event291052
    frameStart := 290999 },
  { event := event291053
    frameStart := 290999 },
  { event := event291054
    frameStart := 290999 },
  { event := event291055
    frameStart := 290999 }
]

def eventLeaf18191 : Array AnnotatedEvent := #[
  { event := event291056
    frameStart := 290999 },
  { event := event291057
    frameStart := 290999 },
  { event := event291058
    frameStart := 290999 },
  { event := event291059
    frameStart := 290999 },
  { event := event291060
    frameStart := 290999 },
  { event := event291061
    frameStart := 290999 },
  { event := event291062
    frameStart := 290999 },
  { event := event291063
    frameStart := 290999 },
  { event := event291064
    frameStart := 290999 },
  { event := event291065
    frameStart := 290999 },
  { event := event291066
    frameStart := 290999 },
  { event := event291067
    frameStart := 290999 },
  { event := event291068
    frameStart := 290999 },
  { event := event291069
    frameStart := 290999 },
  { event := event291070
    frameStart := 290999 },
  { event := event291071
    frameStart := 290999 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events1136
