import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard006
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard128
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard142
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard143
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard235
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard236
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard238
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard239
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard241
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard242
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard243
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard245
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard246
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard247

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace ResidualResult31516
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  witness.honestTerminalActual 31516
end ResidualResult31516

namespace ResidualResult31521
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult723.actual selector witness *
    ResidualResult17057.actual selector witness
end ResidualResult31521

namespace ResidualResult31526
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult16922.actual selector witness *
    ResidualResult15896.actual selector witness
end ResidualResult31526

namespace ResidualResult31530
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult31526.actual selector witness -
    ResidualResult31521.actual selector witness
end ResidualResult31530

namespace ResidualResult31536
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult31530.actual selector witness +
    ResidualResult31516.actual selector witness
end ResidualResult31536

namespace ResidualResult31543
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult31536.actual selector witness -
    ResidualResult31536.actual selector witness
end ResidualResult31543

namespace ResidualResult31548
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult31543.actual selector witness +
    ResidualResult31513.actual selector witness
end ResidualResult31548

namespace ResidualResult31553
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult31548.actual selector witness +
    ResidualResult31301.actual selector witness
end ResidualResult31553

namespace ResidualResult31558
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult31553.actual selector witness +
    ResidualResult31089.actual selector witness
end ResidualResult31558

namespace ResidualResult31563
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult31558.actual selector witness +
    ResidualResult30877.actual selector witness
end ResidualResult31563

namespace ResidualResult31568
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult31563.actual selector witness +
    ResidualResult30665.actual selector witness
end ResidualResult31568

namespace ResidualResult31573
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult31568.actual selector witness +
    ResidualResult30453.actual selector witness
end ResidualResult31573

namespace ResidualResult31578
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult31573.actual selector witness +
    ResidualResult30241.actual selector witness
end ResidualResult31578

namespace ResidualResult31583
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult31578.actual selector witness +
    ResidualResult30029.actual selector witness
end ResidualResult31583

namespace ResidualResult31588
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult31583.actual selector witness +
    ResidualResult29817.actual selector witness
end ResidualResult31588

namespace ResidualResult31593
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult31588.actual selector witness +
    ResidualResult29605.actual selector witness
end ResidualResult31593

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert
