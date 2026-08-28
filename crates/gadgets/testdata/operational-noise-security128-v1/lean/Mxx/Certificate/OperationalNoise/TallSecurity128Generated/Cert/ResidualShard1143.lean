import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard125
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard1056
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard1098
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard1142

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace ResidualResult161615
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  witness.honestTerminalActual 161615
end ResidualResult161615

namespace ResidualResult161620
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult161592.actual selector witness *
    ResidualResult161615.actual selector witness
end ResidualResult161620

namespace ResidualResult161623
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  witness.honestTerminalActual 161623
end ResidualResult161623

namespace ResidualResult161627
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult161623.actual selector witness -
    ResidualResult161620.actual selector witness
end ResidualResult161627

namespace ResidualResult161631
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult161627.actual selector witness -
    ResidualResult161612.actual selector witness
end ResidualResult161631

namespace ResidualResult161640
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult149120.actual selector witness *
    ResidualResult161469.actual selector witness
end ResidualResult161640

namespace ResidualResult161647
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult161640.actual selector witness +
    ResidualResult161462.actual selector witness
end ResidualResult161647

namespace ResidualResult161657
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult161647.actual selector witness *
    ResidualResult15742.actual selector witness
end ResidualResult161657

namespace ResidualResult161661
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  witness.honestTerminalActual 161661
end ResidualResult161661

namespace ResidualResult161664
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  witness.honestTerminalActual 161664
end ResidualResult161664

namespace ResidualResult161674
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult154608.actual selector witness *
    ResidualResult161664.actual selector witness
end ResidualResult161674

namespace ResidualResult161677
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  witness.honestTerminalActual 161677
end ResidualResult161677

namespace ResidualResult161681
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  witness.honestTerminalActual 161681
end ResidualResult161681

namespace ResidualResult161779
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  witness.honestTerminalActual 161779
end ResidualResult161779

namespace ResidualResult161790
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  witness.honestTerminalActual 161790
end ResidualResult161790

namespace ResidualResult161793
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  witness.honestTerminalActual 161793
end ResidualResult161793

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert
