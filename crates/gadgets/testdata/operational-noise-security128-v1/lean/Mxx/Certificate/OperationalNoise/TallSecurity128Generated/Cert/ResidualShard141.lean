import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard128
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard129
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard134
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard135
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard136
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard137
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard138
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard139
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard140

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace ResidualResult16730
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  witness.honestTerminalActual 16730
end ResidualResult16730

namespace ResidualResult16734
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  witness.honestTerminalActual 16734
end ResidualResult16734

namespace ResidualResult16757
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult15977.actual selector witness *
    ResidualResult16734.actual selector witness
end ResidualResult16757

namespace ResidualResult16761
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult15901.actual selector witness +
    ResidualResult16757.actual selector witness
end ResidualResult16761

namespace ResidualResult16765
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult16761.actual selector witness +
    ResidualResult16722.actual selector witness
end ResidualResult16765

namespace ResidualResult16769
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult16765.actual selector witness +
    ResidualResult16682.actual selector witness
end ResidualResult16769

namespace ResidualResult16773
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult16769.actual selector witness +
    ResidualResult16642.actual selector witness
end ResidualResult16773

namespace ResidualResult16777
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult16773.actual selector witness +
    ResidualResult16602.actual selector witness
end ResidualResult16777

namespace ResidualResult16781
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult16777.actual selector witness +
    ResidualResult16562.actual selector witness
end ResidualResult16781

namespace ResidualResult16785
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult16781.actual selector witness +
    ResidualResult16522.actual selector witness
end ResidualResult16785

namespace ResidualResult16789
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult16785.actual selector witness +
    ResidualResult16482.actual selector witness
end ResidualResult16789

namespace ResidualResult16793
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult16789.actual selector witness +
    ResidualResult16442.actual selector witness
end ResidualResult16793

namespace ResidualResult16797
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult16793.actual selector witness +
    ResidualResult16402.actual selector witness
end ResidualResult16797

namespace ResidualResult16801
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult16797.actual selector witness +
    ResidualResult16362.actual selector witness
end ResidualResult16801

namespace ResidualResult16805
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult16801.actual selector witness +
    ResidualResult16322.actual selector witness
end ResidualResult16805

namespace ResidualResult16809
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult16805.actual selector witness +
    ResidualResult16282.actual selector witness
end ResidualResult16809

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert
