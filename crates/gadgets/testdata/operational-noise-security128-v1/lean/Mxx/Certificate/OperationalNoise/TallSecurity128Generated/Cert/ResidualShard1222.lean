import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard1157
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard1185
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard1189
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard1192
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard1196
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard1200
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard1203
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard1207
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard1211
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard1214
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard1218
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard1221

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace ResidualResult172293
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult172265.actual selector witness *
    ResidualResult172288.actual selector witness
end ResidualResult172293

namespace ResidualResult172296
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  witness.honestTerminalActual 172296
end ResidualResult172296

namespace ResidualResult172300
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult172296.actual selector witness -
    ResidualResult172293.actual selector witness
end ResidualResult172300

namespace ResidualResult172304
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult172300.actual selector witness -
    ResidualResult172285.actual selector witness
end ResidualResult172304

namespace ResidualResult172313
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult163745.actual selector witness *
    ResidualResult172142.actual selector witness
end ResidualResult172313

namespace ResidualResult172320
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult172313.actual selector witness +
    ResidualResult172135.actual selector witness
end ResidualResult172320

namespace ResidualResult172325
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult172320.actual selector witness +
    ResidualResult171838.actual selector witness
end ResidualResult172325

namespace ResidualResult172330
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult172325.actual selector witness +
    ResidualResult171356.actual selector witness
end ResidualResult172330

namespace ResidualResult172335
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult172330.actual selector witness +
    ResidualResult170874.actual selector witness
end ResidualResult172335

namespace ResidualResult172340
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult172335.actual selector witness +
    ResidualResult170392.actual selector witness
end ResidualResult172340

namespace ResidualResult172345
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult172340.actual selector witness +
    ResidualResult169910.actual selector witness
end ResidualResult172345

namespace ResidualResult172350
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult172345.actual selector witness +
    ResidualResult169428.actual selector witness
end ResidualResult172350

namespace ResidualResult172355
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult172350.actual selector witness +
    ResidualResult168946.actual selector witness
end ResidualResult172355

namespace ResidualResult172360
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult172355.actual selector witness +
    ResidualResult168464.actual selector witness
end ResidualResult172360

namespace ResidualResult172365
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult172360.actual selector witness +
    ResidualResult167982.actual selector witness
end ResidualResult172365

namespace ResidualResult172370
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult172365.actual selector witness +
    ResidualResult167500.actual selector witness
end ResidualResult172370

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert
