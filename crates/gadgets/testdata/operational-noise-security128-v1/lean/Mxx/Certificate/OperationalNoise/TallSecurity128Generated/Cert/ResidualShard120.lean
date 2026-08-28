import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard006
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard060
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard066
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard072
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard078
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard084
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard090
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard096
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard102
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard108
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard114
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard116
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard117
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard119

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace ResidualResult14988
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult14984.actual selector witness +
    ResidualResult14804.actual selector witness
end ResidualResult14988

namespace ResidualResult14992
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult14988.actual selector witness +
    ResidualResult14796.actual selector witness
end ResidualResult14992

namespace ResidualResult14996
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult14992.actual selector witness +
    ResidualResult14788.actual selector witness
end ResidualResult14996

namespace ResidualResult15000
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult14996.actual selector witness -
    ResidualResult14780.actual selector witness
end ResidualResult15000

namespace ResidualResult15023
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult15000.actual selector witness *
    ResidualResult14287.actual selector witness
end ResidualResult15023

namespace ResidualResult15027
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult728.actual selector witness +
    ResidualResult15023.actual selector witness
end ResidualResult15027

namespace ResidualResult15031
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult15027.actual selector witness +
    ResidualResult14285.actual selector witness
end ResidualResult15031

namespace ResidualResult15035
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult15031.actual selector witness +
    ResidualResult13543.actual selector witness
end ResidualResult15035

namespace ResidualResult15039
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult15035.actual selector witness +
    ResidualResult12795.actual selector witness
end ResidualResult15039

namespace ResidualResult15043
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult15039.actual selector witness +
    ResidualResult12047.actual selector witness
end ResidualResult15043

namespace ResidualResult15047
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult15043.actual selector witness +
    ResidualResult11299.actual selector witness
end ResidualResult15047

namespace ResidualResult15051
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult15047.actual selector witness +
    ResidualResult10551.actual selector witness
end ResidualResult15051

namespace ResidualResult15055
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult15051.actual selector witness +
    ResidualResult9803.actual selector witness
end ResidualResult15055

namespace ResidualResult15059
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult15055.actual selector witness +
    ResidualResult9055.actual selector witness
end ResidualResult15059

namespace ResidualResult15063
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult15059.actual selector witness +
    ResidualResult8307.actual selector witness
end ResidualResult15063

namespace ResidualResult15067
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult15063.actual selector witness +
    ResidualResult7559.actual selector witness
end ResidualResult15067

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert
