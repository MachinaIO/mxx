import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events850

open Mxx.Certificate.OperationalNoise
open CertificateABI

def exact217600RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩]

theorem exact217600RawTermsValid :
    exact217600RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event217600 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7317⟩⟩) exact217600RawTerms .large 217599 .exactZero (none)

def event217601 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7318⟩⟩) 0 ⟨7317⟩ 217600

def event217602 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7318⟩⟩) 1 ⟨7218⟩ 217534

def event217603 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7318⟩⟩) (.sum [.predecessor 0 217601 .coefficient, .predecessor 1 217602 .coefficient])

def exact217604RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩]

theorem exact217604RawTermsValid :
    exact217604RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event217604 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7318⟩⟩) exact217604RawTerms .large 217603 .exactZero (none)

def event217605 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7319⟩⟩) 0 ⟨7318⟩ 217604

def event217606 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7319⟩⟩) 1 ⟨7220⟩ 217531

def event217607 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7319⟩⟩) (.sum [.predecessor 0 217605 .coefficient, .predecessor 1 217606 .coefficient])

def exact217608RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩]

theorem exact217608RawTermsValid :
    exact217608RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event217608 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7319⟩⟩) exact217608RawTerms .large 217607 .exactZero (none)

def event217609 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7320⟩⟩) 0 ⟨7319⟩ 217608

def event217610 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7320⟩⟩) 1 ⟨7222⟩ 217528

def event217611 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7320⟩⟩) (.sum [.predecessor 0 217609 .coefficient, .predecessor 1 217610 .coefficient])

def exact217612RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩]

theorem exact217612RawTermsValid :
    exact217612RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event217612 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7320⟩⟩) exact217612RawTerms .large 217611 .exactZero (none)

def event217613 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7321⟩⟩) 0 ⟨7320⟩ 217612

def event217614 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7321⟩⟩) 1 ⟨7224⟩ 217525

def event217615 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7321⟩⟩) (.sum [.predecessor 0 217613 .coefficient, .predecessor 1 217614 .coefficient])

def exact217616RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7224⟩⟩]⟩, (1)⟩]

theorem exact217616RawTermsValid :
    exact217616RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event217616 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7321⟩⟩) exact217616RawTerms .large 217615 .exactZero (none)

def event217617 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7322⟩⟩) 0 ⟨7321⟩ 217616

def event217618 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7322⟩⟩) 1 ⟨7226⟩ 217522

def event217619 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7322⟩⟩) (.sum [.predecessor 0 217617 .coefficient, .predecessor 1 217618 .coefficient])

def exact217620RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7224⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7226⟩⟩]⟩, (1)⟩]

theorem exact217620RawTermsValid :
    exact217620RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event217620 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7322⟩⟩) exact217620RawTerms .large 217619 .exactZero (none)

def event217621 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7323⟩⟩) 0 ⟨7322⟩ 217620

def event217622 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7323⟩⟩) 1 ⟨7228⟩ 217519

def event217623 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7323⟩⟩) (.sum [.predecessor 0 217621 .coefficient, .predecessor 1 217622 .coefficient])

def exact217624RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7224⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7226⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7228⟩⟩]⟩, (1)⟩]

theorem exact217624RawTermsValid :
    exact217624RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event217624 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7323⟩⟩) exact217624RawTerms .large 217623 .exactZero (none)

def event217625 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7324⟩⟩) 0 ⟨7323⟩ 217624

def event217626 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7324⟩⟩) 1 ⟨7230⟩ 217516

def event217627 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7324⟩⟩) (.sum [.predecessor 0 217625 .coefficient, .predecessor 1 217626 .coefficient])

def exact217628RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7224⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7226⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7228⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7230⟩⟩]⟩, (1)⟩]

theorem exact217628RawTermsValid :
    exact217628RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event217628 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7324⟩⟩) exact217628RawTerms .large 217627 .exactZero (none)

def event217629 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7325⟩⟩) 0 ⟨7324⟩ 217628

def event217630 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7325⟩⟩) 1 ⟨7232⟩ 217513

def event217631 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7325⟩⟩) (.sum [.predecessor 0 217629 .coefficient, .predecessor 1 217630 .coefficient])

def exact217632RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7224⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7226⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7228⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7230⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7232⟩⟩]⟩, (1)⟩]

theorem exact217632RawTermsValid :
    exact217632RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event217632 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7325⟩⟩) exact217632RawTerms .large 217631 .exactZero (none)

def event217633 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69090⟩⟩) 0 ⟨7325⟩ 217632

def event217634 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69090⟩⟩) 1 ⟨69089⟩ 217510

def event217635 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69090⟩⟩) (.sum [.predecessor 0 217633 .coefficient, .predecessor 1 217634 .coefficient])

def exact217636RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7224⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7226⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7228⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7230⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7232⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨16035⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨18866⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨22086⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨26619⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨29299⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨32106⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨34963⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨37643⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨40319⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨42999⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨45683⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨48363⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨51161⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨54141⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨57121⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨60101⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨63081⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨66601⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact217636RawTermsValid :
    exact217636RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event217636 : Event := .resultExact (⟨.program ⟨257⟩, ⟨69090⟩⟩) exact217636RawTerms .large 217635 .exactZero (none)

def event217637 : Event := .predecessor (⟨.program ⟨257⟩, ⟨71237⟩⟩) 0 ⟨69090⟩ 217636

def event217638 : Event := .predecessor (⟨.program ⟨257⟩, ⟨71237⟩⟩) 1 ⟨71236⟩ 217477

def event217639 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨71237⟩⟩) (.product (.predecessor 0 217637 .coefficient) (.predecessor 1 217638 .coefficient) (⟨false, false, none, none, none⟩))

def event217640 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71237⟩⟩, .operator (⟨217636, 17⟩, ⟨217477, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7232⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (1)⟩)

def event217641 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71237⟩⟩, .operator (⟨217636, 16⟩, ⟨217477, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (1)⟩)

def event217642 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71237⟩⟩, .operator (⟨217636, 15⟩, ⟨217477, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (1)⟩)

def event217643 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71237⟩⟩, .operator (⟨217636, 14⟩, ⟨217477, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (1)⟩)

def event217644 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71237⟩⟩, .operator (⟨217636, 13⟩, ⟨217477, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (1)⟩)

def event217645 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71237⟩⟩, .operator (⟨217636, 12⟩, ⟨217477, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7222⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (1)⟩)

def event217646 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71237⟩⟩, .operator (⟨217636, 11⟩, ⟨217477, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7220⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (1)⟩)

def event217647 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71237⟩⟩, .operator (⟨217636, 10⟩, ⟨217477, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7218⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (1)⟩)

def event217648 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71237⟩⟩, .operator (⟨217636, 9⟩, ⟨217477, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7216⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (1)⟩)

def event217649 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71237⟩⟩, .operator (⟨217636, 8⟩, ⟨217477, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7214⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (1)⟩)

def event217650 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71237⟩⟩, .operator (⟨217636, 7⟩, ⟨217477, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (1)⟩)

def event217651 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71237⟩⟩, .operator (⟨217636, 6⟩, ⟨217477, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (1)⟩)

def event217652 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71237⟩⟩, .operator (⟨217636, 5⟩, ⟨217477, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (1)⟩)

def event217653 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71237⟩⟩, .operator (⟨217636, 4⟩, ⟨217477, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (1)⟩)

def event217654 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71237⟩⟩, .operator (⟨217636, 3⟩, ⟨217477, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (1)⟩)

def event217655 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71237⟩⟩, .operator (⟨217636, 2⟩, ⟨217477, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (1)⟩)

def event217656 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71237⟩⟩, .operator (⟨217636, 1⟩, ⟨217477, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (1)⟩)

def event217657 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71237⟩⟩, .operator (⟨217636, 0⟩, ⟨217477, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (1)⟩)

def event217658 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71237⟩⟩, .operator (⟨217636, 29⟩, ⟨217477, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨48363⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (-1)⟩)

def event217659 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71237⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨48363⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71236⟩⟩) ⟨68830⟩ 217474)

def event217660 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71237⟩⟩, .relation 217659 0, ⟨[⟨.program ⟨257⟩, ⟨48363⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩, (-1)⟩)

def event217661 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71237⟩⟩, .operator (⟨217636, 28⟩, ⟨217477, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨45683⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (-1)⟩)

def event217662 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71237⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨45683⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71236⟩⟩) ⟨68830⟩ 217474)

def event217663 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71237⟩⟩, .relation 217662 0, ⟨[⟨.program ⟨257⟩, ⟨45683⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩, (-1)⟩)

def event217664 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71237⟩⟩, .operator (⟨217636, 27⟩, ⟨217477, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨42999⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (-1)⟩)

def event217665 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71237⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨42999⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71236⟩⟩) ⟨68830⟩ 217474)

def event217666 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71237⟩⟩, .relation 217665 0, ⟨[⟨.program ⟨257⟩, ⟨42999⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩, (-1)⟩)

def event217667 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71237⟩⟩, .operator (⟨217636, 26⟩, ⟨217477, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨40319⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (-1)⟩)

def event217668 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71237⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨40319⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71236⟩⟩) ⟨68830⟩ 217474)

def event217669 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71237⟩⟩, .relation 217668 0, ⟨[⟨.program ⟨257⟩, ⟨40319⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩, (-1)⟩)

def event217670 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71237⟩⟩, .operator (⟨217636, 25⟩, ⟨217477, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨37643⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (-1)⟩)

def event217671 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71237⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨37643⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71236⟩⟩) ⟨68830⟩ 217474)

def event217672 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71237⟩⟩, .relation 217671 0, ⟨[⟨.program ⟨257⟩, ⟨37643⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩, (-1)⟩)

def event217673 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71237⟩⟩, .operator (⟨217636, 24⟩, ⟨217477, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨34963⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (-1)⟩)

def event217674 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71237⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨34963⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71236⟩⟩) ⟨68830⟩ 217474)

def event217675 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71237⟩⟩, .relation 217674 0, ⟨[⟨.program ⟨257⟩, ⟨34963⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩, (-1)⟩)

def event217676 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71237⟩⟩, .operator (⟨217636, 22⟩, ⟨217477, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨29299⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (-1)⟩)

def event217677 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71237⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨29299⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71236⟩⟩) ⟨68830⟩ 217474)

def event217678 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71237⟩⟩, .relation 217677 0, ⟨[⟨.program ⟨257⟩, ⟨29299⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩, (-1)⟩)

def event217679 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71237⟩⟩, .operator (⟨217636, 21⟩, ⟨217477, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨26619⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (-1)⟩)

def event217680 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71237⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨26619⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71236⟩⟩) ⟨68830⟩ 217474)

def event217681 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71237⟩⟩, .relation 217680 0, ⟨[⟨.program ⟨257⟩, ⟨26619⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩, (-1)⟩)

def event217682 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71237⟩⟩, .operator (⟨217636, 35⟩, ⟨217477, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨66601⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (-1)⟩)

def event217683 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71237⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨66601⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71236⟩⟩) ⟨68830⟩ 217474)

def event217684 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71237⟩⟩, .relation 217683 0, ⟨[⟨.program ⟨257⟩, ⟨66601⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩, (-1)⟩)

def event217685 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71237⟩⟩, .operator (⟨217636, 34⟩, ⟨217477, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨63081⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (-1)⟩)

def event217686 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71237⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨63081⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71236⟩⟩) ⟨68830⟩ 217474)

def event217687 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71237⟩⟩, .relation 217686 0, ⟨[⟨.program ⟨257⟩, ⟨63081⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩, (-1)⟩)

def event217688 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71237⟩⟩, .operator (⟨217636, 33⟩, ⟨217477, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨60101⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (-1)⟩)

def event217689 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71237⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨60101⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71236⟩⟩) ⟨68830⟩ 217474)

def event217690 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71237⟩⟩, .relation 217689 0, ⟨[⟨.program ⟨257⟩, ⟨60101⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩, (-1)⟩)

def event217691 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71237⟩⟩, .operator (⟨217636, 32⟩, ⟨217477, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨57121⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (-1)⟩)

def event217692 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71237⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨57121⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71236⟩⟩) ⟨68830⟩ 217474)

def event217693 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71237⟩⟩, .relation 217692 0, ⟨[⟨.program ⟨257⟩, ⟨57121⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩, (-1)⟩)

def event217694 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71237⟩⟩, .operator (⟨217636, 31⟩, ⟨217477, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨54141⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (-1)⟩)

def event217695 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71237⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨54141⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71236⟩⟩) ⟨68830⟩ 217474)

def event217696 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71237⟩⟩, .relation 217695 0, ⟨[⟨.program ⟨257⟩, ⟨54141⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩, (-1)⟩)

def event217697 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71237⟩⟩, .operator (⟨217636, 30⟩, ⟨217477, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨51161⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (-1)⟩)

def event217698 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71237⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨51161⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71236⟩⟩) ⟨68830⟩ 217474)

def event217699 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71237⟩⟩, .relation 217698 0, ⟨[⟨.program ⟨257⟩, ⟨51161⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩, (-1)⟩)

def event217700 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71237⟩⟩, .operator (⟨217636, 23⟩, ⟨217477, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨32106⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (-1)⟩)

def event217701 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71237⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨32106⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71236⟩⟩) ⟨68830⟩ 217474)

def event217702 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71237⟩⟩, .relation 217701 0, ⟨[⟨.program ⟨257⟩, ⟨32106⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩, (-1)⟩)

def event217703 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71237⟩⟩, .operator (⟨217636, 20⟩, ⟨217477, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨22086⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (-1)⟩)

def event217704 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71237⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨22086⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71236⟩⟩) ⟨68830⟩ 217474)

def event217705 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71237⟩⟩, .relation 217704 0, ⟨[⟨.program ⟨257⟩, ⟨22086⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩, (-1)⟩)

def event217706 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71237⟩⟩, .operator (⟨217636, 19⟩, ⟨217477, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨18866⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (-1)⟩)

def event217707 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71237⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨18866⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71236⟩⟩) ⟨68830⟩ 217474)

def event217708 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71237⟩⟩, .relation 217707 0, ⟨[⟨.program ⟨257⟩, ⟨18866⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩, (-1)⟩)

def event217709 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71237⟩⟩, .operator (⟨217636, 18⟩, ⟨217477, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨16035⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (-1)⟩)

def event217710 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71237⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨16035⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71236⟩⟩) ⟨68830⟩ 217474)

def event217711 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71237⟩⟩, .relation 217710 0, ⟨[⟨.program ⟨257⟩, ⟨16035⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩, (-1)⟩)

def exact217712RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7214⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7216⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7218⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7220⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7222⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7232⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨16035⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨18866⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨22086⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨26619⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨29299⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨32106⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨34963⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨37643⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨40319⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨42999⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨45683⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨48363⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨51161⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨54141⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨57121⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨60101⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨63081⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨66601⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩, (-1)⟩]

theorem exact217712RawTermsValid :
    exact217712RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event217712 : Event := .resultExact (⟨.program ⟨257⟩, ⟨71237⟩⟩) exact217712RawTerms .large 217639 .exactZero (none)

def event217713 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67457⟩⟩) 0 ⟨66611⟩ 217466

def event217714 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67457⟩⟩) (.authority (.programFamilyFact))

def exact217715RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨67457⟩⟩], []⟩, (1)⟩]

theorem exact217715RawTermsValid :
    exact217715RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event217715 : Event := .resultExact (⟨.program ⟨257⟩, ⟨67457⟩⟩) exact217715RawTerms (.finite 18) 217714 .exactZero (none)

def event217716 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67459⟩⟩) 0 ⟨6908⟩ 217488

def event217717 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67459⟩⟩) 1 ⟨67457⟩ 217715

def event217718 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67459⟩⟩) (.product (.predecessor 0 217716 .coefficient) (.predecessor 1 217717 .coefficient) (⟨false, true, none, none, some 1⟩))

def event217719 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67459⟩⟩, .operator (⟨217488, 0⟩, ⟨217715, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨67457⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact217720RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨67457⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact217720RawTermsValid :
    exact217720RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event217720 : Event := .resultExact (⟨.program ⟨257⟩, ⟨67459⟩⟩) exact217720RawTerms .large 217718 .exactZero (none)

def event217721 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7233⟩⟩) 0 ⟨7177⟩ 217470

def event217722 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7233⟩⟩) (.authority (.operator))

def exact217723RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7233⟩⟩]⟩, (1)⟩]

theorem exact217723RawTermsValid :
    exact217723RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event217723 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7233⟩⟩) exact217723RawTerms .large 217722 .exactZero (none)

def event217724 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67464⟩⟩) 0 ⟨7233⟩ 217723

def event217725 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67464⟩⟩) 1 ⟨67459⟩ 217720

def event217726 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67464⟩⟩) (.sum [.predecessor 0 217724 .coefficient, .predecessor 1 217725 .coefficient])

def exact217727RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7233⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨67457⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact217727RawTermsValid :
    exact217727RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event217727 : Event := .resultExact (⟨.program ⟨257⟩, ⟨67464⟩⟩) exact217727RawTerms .large 217726 .exactZero (none)

def event217728 : Event := .predecessor (⟨.program ⟨257⟩, ⟨71241⟩⟩) 0 ⟨67464⟩ 217727

def event217729 : Event := .predecessor (⟨.program ⟨257⟩, ⟨71241⟩⟩) 1 ⟨71237⟩ 217712

def event217730 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨71241⟩⟩) (.sum [.predecessor 0 217728 .coefficient, .predecessor 1 217729 .coefficient])

def exact217731RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7214⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7216⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7218⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7220⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7222⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7232⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7233⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨16035⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨18866⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨22086⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨26619⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨29299⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨32106⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨34963⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨37643⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨40319⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨42999⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨45683⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨48363⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨51161⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨54141⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨57121⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨60101⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨63081⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨66601⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨67457⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact217731RawTermsValid :
    exact217731RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event217731 : Event := .resultExact (⟨.program ⟨257⟩, ⟨71241⟩⟩) exact217731RawTerms .large 217730 .exactZero (none)

def event217732 : Event := .preFoldPolynomial 217731 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7214⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7216⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7218⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7220⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7222⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7232⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7233⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨16035⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨18866⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨22086⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨26619⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨29299⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨32106⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨34963⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨37643⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨40319⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨42999⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨45683⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨48363⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨51161⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨54141⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨57121⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨60101⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨63081⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨66601⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨67457⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact217733RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7214⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7216⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7218⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7220⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7222⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7232⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7233⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨16035⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨18866⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨22086⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨26619⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨29299⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨32106⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨34963⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨37643⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨40319⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨42999⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨45683⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨48363⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨51161⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨54141⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨57121⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨60101⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨63081⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨66601⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨67457⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event217733 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨71241⟩⟩) 217732 exact217733RawTerms .large 217730 .exactZero (none)

def event217734 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨66611⟩⟩) ⟨⟨1⟩, ⟨95⟩, ⟨135⟩⟩ ⟨216372, 217734⟩

def event217735 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨68373⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68370⟩⟩]⟩) (1) 0 2 (.universal 217734 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68370⟩⟩]⟩) (none) 217733)

def event217736 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68373⟩⟩, .relation 217735 18, ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7233⟩⟩]⟩, (1)⟩)

def event217737 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68373⟩⟩, .relation 217735 17, ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7232⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (-1)⟩)

def event217738 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68373⟩⟩, .relation 217735 16, ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (-1)⟩)

def event217739 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68373⟩⟩, .relation 217735 15, ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (-1)⟩)

def event217740 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68373⟩⟩, .relation 217735 14, ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (-1)⟩)

def event217741 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68373⟩⟩, .relation 217735 13, ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (-1)⟩)

def event217742 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68373⟩⟩, .relation 217735 12, ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7222⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (-1)⟩)

def event217743 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68373⟩⟩, .relation 217735 11, ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (-1)⟩)

def event217744 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68373⟩⟩, .relation 217735 10, ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (-1)⟩)

def event217745 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68373⟩⟩, .relation 217735 9, ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (-1)⟩)

def event217746 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68373⟩⟩, .relation 217735 8, ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (-1)⟩)

def event217747 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68373⟩⟩, .relation 217735 7, ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (-1)⟩)

def event217748 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68373⟩⟩, .relation 217735 6, ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (-1)⟩)

def event217749 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68373⟩⟩, .relation 217735 5, ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (-1)⟩)

def event217750 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68373⟩⟩, .relation 217735 4, ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (-1)⟩)

def event217751 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68373⟩⟩, .relation 217735 3, ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (-1)⟩)

def event217752 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68373⟩⟩, .relation 217735 2, ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (-1)⟩)

def event217753 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68373⟩⟩, .relation 217735 1, ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (-1)⟩)

def event217754 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68373⟩⟩, .relation 217735 0, ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (-1)⟩)

def event217755 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68373⟩⟩, .relation 217735 30, ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨48363⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩, (1)⟩)

def event217756 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68373⟩⟩, .relation 217735 29, ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨45683⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩, (1)⟩)

def event217757 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68373⟩⟩, .relation 217735 28, ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨42999⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩, (1)⟩)

def event217758 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68373⟩⟩, .relation 217735 27, ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨40319⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩, (1)⟩)

def event217759 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68373⟩⟩, .relation 217735 26, ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨37643⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩, (1)⟩)

def event217760 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68373⟩⟩, .relation 217735 25, ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨34963⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩, (1)⟩)

def event217761 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68373⟩⟩, .relation 217735 23, ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨29299⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩, (1)⟩)

def event217762 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68373⟩⟩, .relation 217735 22, ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨26619⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩, (1)⟩)

def event217763 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68373⟩⟩, .relation 217735 36, ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨66601⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩, (1)⟩)

def event217764 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68373⟩⟩, .relation 217735 35, ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨63081⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩, (1)⟩)

def event217765 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68373⟩⟩, .relation 217735 34, ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨60101⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩, (1)⟩)

def event217766 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68373⟩⟩, .relation 217735 33, ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨57121⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩, (1)⟩)

def event217767 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68373⟩⟩, .relation 217735 32, ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨54141⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩, (1)⟩)

def event217768 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68373⟩⟩, .relation 217735 31, ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨51161⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩, (1)⟩)

def event217769 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68373⟩⟩, .relation 217735 24, ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨32106⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩, (1)⟩)

def event217770 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68373⟩⟩, .relation 217735 21, ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨22086⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩, (1)⟩)

def event217771 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68373⟩⟩, .relation 217735 20, ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨18866⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩, (1)⟩)

def event217772 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68373⟩⟩, .relation 217735 19, ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨16035⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩, (1)⟩)

def event217773 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68373⟩⟩, .relation 217735 37, ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨67457⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact217774RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7222⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7232⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7233⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨16035⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨18866⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨22086⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨26619⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨29299⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨32106⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨34963⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨37643⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨40319⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨42999⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨45683⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨48363⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨51161⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨54141⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨57121⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨60101⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨63081⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨66601⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨67457⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact217774RawTermsValid :
    exact217774RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event217774 : Event := .resultExact (⟨.program ⟨257⟩, ⟨68373⟩⟩) exact217774RawTerms .large 216368 (.finite 202072841853861888) (some (216370))

def event217775 : Event := .predecessor (⟨.program ⟨257⟩, ⟨71239⟩⟩) 0 ⟨68373⟩ 217774

def event217776 : Event := .predecessor (⟨.program ⟨257⟩, ⟨71239⟩⟩) 1 ⟨71238⟩ 216358

def event217777 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨71239⟩⟩) (.sum [.predecessor 0 217775 .coefficient, .predecessor 1 217776 .coefficient])

def event217778 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71239⟩⟩, .operator (⟨217774, 17⟩, ⟨216358, 17⟩), ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7232⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (1)⟩)

def event217779 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71239⟩⟩, .operator (⟨217774, 30⟩, ⟨216358, 29⟩), ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨48363⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩, (-1)⟩)

def event217780 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71239⟩⟩, .operator (⟨217774, 16⟩, ⟨216358, 16⟩), ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (1)⟩)

def event217781 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71239⟩⟩, .operator (⟨217774, 29⟩, ⟨216358, 28⟩), ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨45683⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩, (-1)⟩)

def event217782 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71239⟩⟩, .operator (⟨217774, 15⟩, ⟨216358, 15⟩), ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (1)⟩)

def event217783 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71239⟩⟩, .operator (⟨217774, 28⟩, ⟨216358, 27⟩), ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨42999⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩, (-1)⟩)

def event217784 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71239⟩⟩, .operator (⟨217774, 14⟩, ⟨216358, 14⟩), ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (1)⟩)

def event217785 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71239⟩⟩, .operator (⟨217774, 27⟩, ⟨216358, 26⟩), ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨40319⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩, (-1)⟩)

def event217786 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71239⟩⟩, .operator (⟨217774, 13⟩, ⟨216358, 13⟩), ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (1)⟩)

def event217787 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71239⟩⟩, .operator (⟨217774, 26⟩, ⟨216358, 25⟩), ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨37643⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩, (-1)⟩)

def event217788 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71239⟩⟩, .operator (⟨217774, 12⟩, ⟨216358, 12⟩), ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7222⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (1)⟩)

def event217789 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71239⟩⟩, .operator (⟨217774, 25⟩, ⟨216358, 24⟩), ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨34963⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩, (-1)⟩)

def event217790 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71239⟩⟩, .operator (⟨217774, 11⟩, ⟨216358, 11⟩), ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (1)⟩)

def event217791 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71239⟩⟩, .operator (⟨217774, 23⟩, ⟨216358, 22⟩), ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨29299⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩, (-1)⟩)

def event217792 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71239⟩⟩, .operator (⟨217774, 10⟩, ⟨216358, 10⟩), ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (1)⟩)

def event217793 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71239⟩⟩, .operator (⟨217774, 22⟩, ⟨216358, 21⟩), ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨26619⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩, (-1)⟩)

def event217794 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71239⟩⟩, .operator (⟨217774, 9⟩, ⟨216358, 9⟩), ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (1)⟩)

def event217795 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71239⟩⟩, .operator (⟨217774, 36⟩, ⟨216358, 35⟩), ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨66601⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩, (-1)⟩)

def event217796 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71239⟩⟩, .operator (⟨217774, 8⟩, ⟨216358, 8⟩), ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (1)⟩)

def event217797 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71239⟩⟩, .operator (⟨217774, 35⟩, ⟨216358, 34⟩), ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨63081⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩, (-1)⟩)

def event217798 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71239⟩⟩, .operator (⟨217774, 7⟩, ⟨216358, 7⟩), ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (1)⟩)

def event217799 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71239⟩⟩, .operator (⟨217774, 34⟩, ⟨216358, 33⟩), ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨60101⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩, (-1)⟩)

def event217800 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71239⟩⟩, .operator (⟨217774, 6⟩, ⟨216358, 6⟩), ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (1)⟩)

def event217801 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71239⟩⟩, .operator (⟨217774, 33⟩, ⟨216358, 32⟩), ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨57121⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩, (-1)⟩)

def event217802 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71239⟩⟩, .operator (⟨217774, 5⟩, ⟨216358, 5⟩), ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (1)⟩)

def event217803 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71239⟩⟩, .operator (⟨217774, 32⟩, ⟨216358, 31⟩), ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨54141⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩, (-1)⟩)

def event217804 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71239⟩⟩, .operator (⟨217774, 4⟩, ⟨216358, 4⟩), ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (1)⟩)

def event217805 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71239⟩⟩, .operator (⟨217774, 31⟩, ⟨216358, 30⟩), ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨51161⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩, (-1)⟩)

def event217806 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71239⟩⟩, .operator (⟨217774, 3⟩, ⟨216358, 3⟩), ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (1)⟩)

def event217807 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71239⟩⟩, .operator (⟨217774, 24⟩, ⟨216358, 23⟩), ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨32106⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩, (-1)⟩)

def event217808 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71239⟩⟩, .operator (⟨217774, 2⟩, ⟨216358, 2⟩), ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (1)⟩)

def event217809 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71239⟩⟩, .operator (⟨217774, 21⟩, ⟨216358, 20⟩), ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨22086⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩, (-1)⟩)

def event217810 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71239⟩⟩, .operator (⟨217774, 1⟩, ⟨216358, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (1)⟩)

def event217811 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71239⟩⟩, .operator (⟨217774, 20⟩, ⟨216358, 19⟩), ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨18866⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩, (-1)⟩)

def event217812 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71239⟩⟩, .operator (⟨217774, 0⟩, ⟨216358, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (1)⟩)

def event217813 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71239⟩⟩, .operator (⟨217774, 19⟩, ⟨216358, 18⟩), ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨16035⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩, (-1)⟩)

def event217814 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨71239⟩⟩) (.sum [.result 217774 .summary, .result 216358 .summary])

def exact217815RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7233⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨67457⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact217815RawTermsValid :
    exact217815RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event217815 : Event := .resultExact (⟨.program ⟨257⟩, ⟨71239⟩⟩) exact217815RawTerms .large 217777 (.finite 6221717896068416040249469506489977540968448) (some (217814))

def event217816 : Event := .predecessor (⟨.program ⟨257⟩, ⟨71240⟩⟩) 0 ⟨71239⟩ 217815

def event217817 : Event := .predecessor (⟨.program ⟨257⟩, ⟨71240⟩⟩) 1 ⟨7140⟩ 15522

def event217818 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨71240⟩⟩) (.product (.predecessor 0 217816 .coefficient) (.predecessor 1 217817 .coefficient) (⟨false, false, none, none, none⟩))

def event217819 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨71240⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨7139⟩⟩]⟩) [⟨.result 15518 .coefficient, false, none⟩])

def event217820 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨71240⟩⟩) (.product (.result 217815 .summary) (.transfer 217819) (⟨false, false, none, none, none⟩))

def event217821 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71240⟩⟩, .operator (⟨217815, 0⟩, ⟨15522, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7233⟩⟩, ⟨.program ⟨257⟩, ⟨7139⟩⟩]⟩, (1)⟩)

def event217822 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71240⟩⟩, .operator (⟨217815, 1⟩, ⟨15522, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨67457⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7139⟩⟩]⟩, (-1)⟩)

def event217823 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71240⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨67457⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7139⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨7139⟩⟩) ⟨7035⟩ 15515)

def event217824 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71240⟩⟩, .relation 217823 0, ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67457⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact217825RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7233⟩⟩, ⟨.program ⟨257⟩, ⟨7139⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67457⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact217825RawTermsValid :
    exact217825RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event217825 : Event := .resultExact (⟨.program ⟨257⟩, ⟨71240⟩⟩) exact217825RawTerms .large 217818 (.finite 66805187221379434678483228029309283225584960819691520) (some (217820))

def event217826 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49300⟩⟩) 0 ⟨7177⟩ 15500

def event217827 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49300⟩⟩) 1 ⟨49299⟩ 207506

def event217828 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49300⟩⟩) (.authority (.operator))

def exact217829RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨49300⟩⟩]⟩, (1)⟩]

theorem exact217829RawTermsValid :
    exact217829RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event217829 : Event := .resultExact (⟨.program ⟨257⟩, ⟨49300⟩⟩) exact217829RawTerms .large 217828 .exactZero (none)

def event217830 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50023⟩⟩) 0 ⟨49300⟩ 217829

def event217831 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50023⟩⟩) (.authority (.operator))

def exact217832RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨50023⟩⟩]⟩, (1)⟩]

theorem exact217832RawTermsValid :
    exact217832RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event217832 : Event := .resultExact (⟨.program ⟨257⟩, ⟨50023⟩⟩) exact217832RawTerms (.finite 8192) 217831 .exactZero (none)

def event217833 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50025⟩⟩) 0 ⟨49661⟩ 207806

def event217834 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50025⟩⟩) 1 ⟨50023⟩ 217832

def event217835 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50025⟩⟩) (.product (.predecessor 0 217833 .coefficient) (.predecessor 1 217834 .coefficient) (⟨false, false, none, none, none⟩))

def event217836 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50025⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨50023⟩⟩]⟩) [⟨.result 217832 .coefficient, false, none⟩])

def event217837 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50025⟩⟩) (.product (.result 207806 .summary) (.transfer 217836) (⟨false, false, none, none, none⟩))

def event217838 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨50025⟩⟩, .operator (⟨207806, 0⟩, ⟨217832, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨50023⟩⟩]⟩, (1)⟩)

def event217839 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨50025⟩⟩, .operator (⟨207806, 1⟩, ⟨217832, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨48148⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨50023⟩⟩]⟩, (-1)⟩)

def event217840 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨50025⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨48148⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨50023⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨50023⟩⟩) ⟨49300⟩ 217829)

def event217841 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨50025⟩⟩, .relation 217840 0, ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨48148⟩⟩], [⟨.program ⟨257⟩, ⟨49300⟩⟩]⟩, (-1)⟩)

def exact217842RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨50023⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨48148⟩⟩], [⟨.program ⟨257⟩, ⟨49300⟩⟩]⟩, (-1)⟩]

theorem exact217842RawTermsValid :
    exact217842RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event217842 : Event := .resultExact (⟨.program ⟨257⟩, ⟨50025⟩⟩) exact217842RawTerms .large 217835 (.finite 32194504275408438756654574469120) (some (217837))

def event217843 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48892⟩⟩) 0 ⟨48149⟩ 9835

def event217844 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48892⟩⟩) (.authority (.relationPreimageSource ⟨93⟩))

def exact217845RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨48892⟩⟩]⟩, (1)⟩]

theorem exact217845RawTermsValid :
    exact217845RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event217845 : Event := .resultExact (⟨.program ⟨257⟩, ⟨48892⟩⟩) exact217845RawTerms (.finite 5647228698) 217844 .exactZero (none)

def event217846 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48894⟩⟩) 0 ⟨48892⟩ 217845

def event217847 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48894⟩⟩) 1 ⟨2370⟩ 4

def event217848 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48894⟩⟩) (.scale (.predecessor 0 217846 .coefficient) (.value (.predecessor 1 217847 .coefficient)))

def exact217849RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨48892⟩⟩]⟩, (1)⟩]

theorem exact217849RawTermsValid :
    exact217849RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event217849 : Event := .resultExact (⟨.program ⟨257⟩, ⟨48894⟩⟩) exact217849RawTerms (.finite 5647228698) 217848 .exactZero (none)

def event217850 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48895⟩⟩) 0 ⟨5599⟩ 207620

def event217851 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48895⟩⟩) 1 ⟨48894⟩ 217849

def event217852 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48895⟩⟩) (.product (.predecessor 0 217850 .coefficient) (.predecessor 1 217851 .coefficient) (⟨false, false, none, none, none⟩))

def event217853 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48895⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨48892⟩⟩]⟩) [⟨.result 217845 .coefficient, false, none⟩])

def event217854 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48895⟩⟩) (.product (.result 207620 .summary) (.transfer 217853) (⟨false, false, none, none, none⟩))

def event217855 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨48895⟩⟩, .operator (⟨207620, 0⟩, ⟨217849, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48892⟩⟩]⟩, (1)⟩)

def eventLeaf13600 : Array AnnotatedEvent := #[
  { event := event217600
    frameStart := 216961 },
  { event := event217601
    frameStart := 216961 },
  { event := event217602
    frameStart := 216961 },
  { event := event217603
    frameStart := 216961 },
  { event := event217604
    frameStart := 216961 },
  { event := event217605
    frameStart := 216961 },
  { event := event217606
    frameStart := 216961 },
  { event := event217607
    frameStart := 216961 },
  { event := event217608
    frameStart := 216961 },
  { event := event217609
    frameStart := 216961 },
  { event := event217610
    frameStart := 216961 },
  { event := event217611
    frameStart := 216961 },
  { event := event217612
    frameStart := 216961 },
  { event := event217613
    frameStart := 216961 },
  { event := event217614
    frameStart := 216961 },
  { event := event217615
    frameStart := 216961 }
]

def eventLeaf13601 : Array AnnotatedEvent := #[
  { event := event217616
    frameStart := 216961 },
  { event := event217617
    frameStart := 216961 },
  { event := event217618
    frameStart := 216961 },
  { event := event217619
    frameStart := 216961 },
  { event := event217620
    frameStart := 216961 },
  { event := event217621
    frameStart := 216961 },
  { event := event217622
    frameStart := 216961 },
  { event := event217623
    frameStart := 216961 },
  { event := event217624
    frameStart := 216961 },
  { event := event217625
    frameStart := 216961 },
  { event := event217626
    frameStart := 216961 },
  { event := event217627
    frameStart := 216961 },
  { event := event217628
    frameStart := 216961 },
  { event := event217629
    frameStart := 216961 },
  { event := event217630
    frameStart := 216961 },
  { event := event217631
    frameStart := 216961 }
]

def eventLeaf13602 : Array AnnotatedEvent := #[
  { event := event217632
    frameStart := 216961 },
  { event := event217633
    frameStart := 216961 },
  { event := event217634
    frameStart := 216961 },
  { event := event217635
    frameStart := 216961 },
  { event := event217636
    frameStart := 216961 },
  { event := event217637
    frameStart := 216961 },
  { event := event217638
    frameStart := 216961 },
  { event := event217639
    frameStart := 216961 },
  { event := event217640
    frameStart := 216961 },
  { event := event217641
    frameStart := 216961 },
  { event := event217642
    frameStart := 216961 },
  { event := event217643
    frameStart := 216961 },
  { event := event217644
    frameStart := 216961 },
  { event := event217645
    frameStart := 216961 },
  { event := event217646
    frameStart := 216961 },
  { event := event217647
    frameStart := 216961 }
]

def eventLeaf13603 : Array AnnotatedEvent := #[
  { event := event217648
    frameStart := 216961 },
  { event := event217649
    frameStart := 216961 },
  { event := event217650
    frameStart := 216961 },
  { event := event217651
    frameStart := 216961 },
  { event := event217652
    frameStart := 216961 },
  { event := event217653
    frameStart := 216961 },
  { event := event217654
    frameStart := 216961 },
  { event := event217655
    frameStart := 216961 },
  { event := event217656
    frameStart := 216961 },
  { event := event217657
    frameStart := 216961 },
  { event := event217658
    frameStart := 216961 },
  { event := event217659
    frameStart := 216961 },
  { event := event217660
    frameStart := 216961 },
  { event := event217661
    frameStart := 216961 },
  { event := event217662
    frameStart := 216961 },
  { event := event217663
    frameStart := 216961 }
]

def eventLeaf13604 : Array AnnotatedEvent := #[
  { event := event217664
    frameStart := 216961 },
  { event := event217665
    frameStart := 216961 },
  { event := event217666
    frameStart := 216961 },
  { event := event217667
    frameStart := 216961 },
  { event := event217668
    frameStart := 216961 },
  { event := event217669
    frameStart := 216961 },
  { event := event217670
    frameStart := 216961 },
  { event := event217671
    frameStart := 216961 },
  { event := event217672
    frameStart := 216961 },
  { event := event217673
    frameStart := 216961 },
  { event := event217674
    frameStart := 216961 },
  { event := event217675
    frameStart := 216961 },
  { event := event217676
    frameStart := 216961 },
  { event := event217677
    frameStart := 216961 },
  { event := event217678
    frameStart := 216961 },
  { event := event217679
    frameStart := 216961 }
]

def eventLeaf13605 : Array AnnotatedEvent := #[
  { event := event217680
    frameStart := 216961 },
  { event := event217681
    frameStart := 216961 },
  { event := event217682
    frameStart := 216961 },
  { event := event217683
    frameStart := 216961 },
  { event := event217684
    frameStart := 216961 },
  { event := event217685
    frameStart := 216961 },
  { event := event217686
    frameStart := 216961 },
  { event := event217687
    frameStart := 216961 },
  { event := event217688
    frameStart := 216961 },
  { event := event217689
    frameStart := 216961 },
  { event := event217690
    frameStart := 216961 },
  { event := event217691
    frameStart := 216961 },
  { event := event217692
    frameStart := 216961 },
  { event := event217693
    frameStart := 216961 },
  { event := event217694
    frameStart := 216961 },
  { event := event217695
    frameStart := 216961 }
]

def eventLeaf13606 : Array AnnotatedEvent := #[
  { event := event217696
    frameStart := 216961 },
  { event := event217697
    frameStart := 216961 },
  { event := event217698
    frameStart := 216961 },
  { event := event217699
    frameStart := 216961 },
  { event := event217700
    frameStart := 216961 },
  { event := event217701
    frameStart := 216961 },
  { event := event217702
    frameStart := 216961 },
  { event := event217703
    frameStart := 216961 },
  { event := event217704
    frameStart := 216961 },
  { event := event217705
    frameStart := 216961 },
  { event := event217706
    frameStart := 216961 },
  { event := event217707
    frameStart := 216961 },
  { event := event217708
    frameStart := 216961 },
  { event := event217709
    frameStart := 216961 },
  { event := event217710
    frameStart := 216961 },
  { event := event217711
    frameStart := 216961 }
]

def eventLeaf13607 : Array AnnotatedEvent := #[
  { event := event217712
    frameStart := 216961 },
  { event := event217713
    frameStart := 216961 },
  { event := event217714
    frameStart := 216961 },
  { event := event217715
    frameStart := 216961 },
  { event := event217716
    frameStart := 216961 },
  { event := event217717
    frameStart := 216961 },
  { event := event217718
    frameStart := 216961 },
  { event := event217719
    frameStart := 216961 },
  { event := event217720
    frameStart := 216961 },
  { event := event217721
    frameStart := 216961 },
  { event := event217722
    frameStart := 216961 },
  { event := event217723
    frameStart := 216961 },
  { event := event217724
    frameStart := 216961 },
  { event := event217725
    frameStart := 216961 },
  { event := event217726
    frameStart := 216961 },
  { event := event217727
    frameStart := 216961 }
]

def eventLeaf13608 : Array AnnotatedEvent := #[
  { event := event217728
    frameStart := 216961 },
  { event := event217729
    frameStart := 216961 },
  { event := event217730
    frameStart := 216961 },
  { event := event217731
    frameStart := 216961 },
  { event := event217732
    frameStart := 216961 },
  { event := event217733
    frameStart := 216961 },
  { event := event217734
    frameStart := 0 },
  { event := event217735
    frameStart := 0 },
  { event := event217736
    frameStart := 0 },
  { event := event217737
    frameStart := 0 },
  { event := event217738
    frameStart := 0 },
  { event := event217739
    frameStart := 0 },
  { event := event217740
    frameStart := 0 },
  { event := event217741
    frameStart := 0 },
  { event := event217742
    frameStart := 0 },
  { event := event217743
    frameStart := 0 }
]

def eventLeaf13609 : Array AnnotatedEvent := #[
  { event := event217744
    frameStart := 0 },
  { event := event217745
    frameStart := 0 },
  { event := event217746
    frameStart := 0 },
  { event := event217747
    frameStart := 0 },
  { event := event217748
    frameStart := 0 },
  { event := event217749
    frameStart := 0 },
  { event := event217750
    frameStart := 0 },
  { event := event217751
    frameStart := 0 },
  { event := event217752
    frameStart := 0 },
  { event := event217753
    frameStart := 0 },
  { event := event217754
    frameStart := 0 },
  { event := event217755
    frameStart := 0 },
  { event := event217756
    frameStart := 0 },
  { event := event217757
    frameStart := 0 },
  { event := event217758
    frameStart := 0 },
  { event := event217759
    frameStart := 0 }
]

def eventLeaf13610 : Array AnnotatedEvent := #[
  { event := event217760
    frameStart := 0 },
  { event := event217761
    frameStart := 0 },
  { event := event217762
    frameStart := 0 },
  { event := event217763
    frameStart := 0 },
  { event := event217764
    frameStart := 0 },
  { event := event217765
    frameStart := 0 },
  { event := event217766
    frameStart := 0 },
  { event := event217767
    frameStart := 0 },
  { event := event217768
    frameStart := 0 },
  { event := event217769
    frameStart := 0 },
  { event := event217770
    frameStart := 0 },
  { event := event217771
    frameStart := 0 },
  { event := event217772
    frameStart := 0 },
  { event := event217773
    frameStart := 0 },
  { event := event217774
    frameStart := 0 },
  { event := event217775
    frameStart := 0 }
]

def eventLeaf13611 : Array AnnotatedEvent := #[
  { event := event217776
    frameStart := 0 },
  { event := event217777
    frameStart := 0 },
  { event := event217778
    frameStart := 0 },
  { event := event217779
    frameStart := 0 },
  { event := event217780
    frameStart := 0 },
  { event := event217781
    frameStart := 0 },
  { event := event217782
    frameStart := 0 },
  { event := event217783
    frameStart := 0 },
  { event := event217784
    frameStart := 0 },
  { event := event217785
    frameStart := 0 },
  { event := event217786
    frameStart := 0 },
  { event := event217787
    frameStart := 0 },
  { event := event217788
    frameStart := 0 },
  { event := event217789
    frameStart := 0 },
  { event := event217790
    frameStart := 0 },
  { event := event217791
    frameStart := 0 }
]

def eventLeaf13612 : Array AnnotatedEvent := #[
  { event := event217792
    frameStart := 0 },
  { event := event217793
    frameStart := 0 },
  { event := event217794
    frameStart := 0 },
  { event := event217795
    frameStart := 0 },
  { event := event217796
    frameStart := 0 },
  { event := event217797
    frameStart := 0 },
  { event := event217798
    frameStart := 0 },
  { event := event217799
    frameStart := 0 },
  { event := event217800
    frameStart := 0 },
  { event := event217801
    frameStart := 0 },
  { event := event217802
    frameStart := 0 },
  { event := event217803
    frameStart := 0 },
  { event := event217804
    frameStart := 0 },
  { event := event217805
    frameStart := 0 },
  { event := event217806
    frameStart := 0 },
  { event := event217807
    frameStart := 0 }
]

def eventLeaf13613 : Array AnnotatedEvent := #[
  { event := event217808
    frameStart := 0 },
  { event := event217809
    frameStart := 0 },
  { event := event217810
    frameStart := 0 },
  { event := event217811
    frameStart := 0 },
  { event := event217812
    frameStart := 0 },
  { event := event217813
    frameStart := 0 },
  { event := event217814
    frameStart := 0 },
  { event := event217815
    frameStart := 0 },
  { event := event217816
    frameStart := 0 },
  { event := event217817
    frameStart := 0 },
  { event := event217818
    frameStart := 0 },
  { event := event217819
    frameStart := 0 },
  { event := event217820
    frameStart := 0 },
  { event := event217821
    frameStart := 0 },
  { event := event217822
    frameStart := 0 },
  { event := event217823
    frameStart := 0 }
]

def eventLeaf13614 : Array AnnotatedEvent := #[
  { event := event217824
    frameStart := 0 },
  { event := event217825
    frameStart := 0 },
  { event := event217826
    frameStart := 0 },
  { event := event217827
    frameStart := 0 },
  { event := event217828
    frameStart := 0 },
  { event := event217829
    frameStart := 0 },
  { event := event217830
    frameStart := 0 },
  { event := event217831
    frameStart := 0 },
  { event := event217832
    frameStart := 0 },
  { event := event217833
    frameStart := 0 },
  { event := event217834
    frameStart := 0 },
  { event := event217835
    frameStart := 0 },
  { event := event217836
    frameStart := 0 },
  { event := event217837
    frameStart := 0 },
  { event := event217838
    frameStart := 0 },
  { event := event217839
    frameStart := 0 }
]

def eventLeaf13615 : Array AnnotatedEvent := #[
  { event := event217840
    frameStart := 0 },
  { event := event217841
    frameStart := 0 },
  { event := event217842
    frameStart := 0 },
  { event := event217843
    frameStart := 0 },
  { event := event217844
    frameStart := 0 },
  { event := event217845
    frameStart := 0 },
  { event := event217846
    frameStart := 0 },
  { event := event217847
    frameStart := 0 },
  { event := event217848
    frameStart := 0 },
  { event := event217849
    frameStart := 0 },
  { event := event217850
    frameStart := 0 },
  { event := event217851
    frameStart := 0 },
  { event := event217852
    frameStart := 0 },
  { event := event217853
    frameStart := 0 },
  { event := event217854
    frameStart := 0 },
  { event := event217855
    frameStart := 0 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events850
