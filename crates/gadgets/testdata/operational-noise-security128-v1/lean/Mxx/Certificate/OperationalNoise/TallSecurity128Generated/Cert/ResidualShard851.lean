import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard006
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard127
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard128
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard248
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard753
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard754
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard847
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard848
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard850

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace ResidualResult119229
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult119201.actual selector witness *
    ResidualResult119224.actual selector witness
end ResidualResult119229

namespace ResidualResult119232
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  witness.honestTerminalActual 119232
end ResidualResult119232

namespace ResidualResult119236
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult119232.actual selector witness -
    ResidualResult119229.actual selector witness
end ResidualResult119236

namespace ResidualResult119240
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult119236.actual selector witness -
    ResidualResult119221.actual selector witness
end ResidualResult119240

namespace ResidualResult119249
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult105245.actual selector witness *
    ResidualResult119078.actual selector witness
end ResidualResult119249

namespace ResidualResult119256
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult119249.actual selector witness +
    ResidualResult119071.actual selector witness
end ResidualResult119256

namespace ResidualResult119266
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult119256.actual selector witness *
    ResidualResult15882.actual selector witness
end ResidualResult119266

namespace ResidualResult119271
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult723.actual selector witness *
    ResidualResult105153.actual selector witness
end ResidualResult119271

namespace ResidualResult119276
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult105023.actual selector witness *
    ResidualResult15896.actual selector witness
end ResidualResult119276

namespace ResidualResult119280
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult119276.actual selector witness -
    ResidualResult119271.actual selector witness
end ResidualResult119280

namespace ResidualResult119286
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult119280.actual selector witness +
    ResidualResult31516.actual selector witness
end ResidualResult119286

namespace ResidualResult119293
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult119286.actual selector witness -
    ResidualResult119286.actual selector witness
end ResidualResult119293

namespace ResidualResult119298
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult119293.actual selector witness +
    ResidualResult119266.actual selector witness
end ResidualResult119298

namespace ResidualResult119303
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult119298.actual selector witness +
    ResidualResult119054.actual selector witness
end ResidualResult119303

namespace ResidualResult119308
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult119303.actual selector witness +
    ResidualResult118842.actual selector witness
end ResidualResult119308

namespace ResidualResult119313
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult119308.actual selector witness +
    ResidualResult118630.actual selector witness
end ResidualResult119313

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert
