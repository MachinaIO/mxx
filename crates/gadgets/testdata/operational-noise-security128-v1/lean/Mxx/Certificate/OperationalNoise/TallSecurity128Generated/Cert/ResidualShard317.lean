import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard252
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard276
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard280
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard284
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard287
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard291
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard295
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard298
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard302
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard306
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard309
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard313
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard316

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace ResidualResult40671
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  witness.honestTerminalActual 40671
end ResidualResult40671

namespace ResidualResult40675
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult40671.actual selector witness -
    ResidualResult40668.actual selector witness
end ResidualResult40675

namespace ResidualResult40679
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult40675.actual selector witness -
    ResidualResult40660.actual selector witness
end ResidualResult40679

namespace ResidualResult40688
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult32120.actual selector witness *
    ResidualResult40517.actual selector witness
end ResidualResult40688

namespace ResidualResult40695
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult40688.actual selector witness +
    ResidualResult40510.actual selector witness
end ResidualResult40695

namespace ResidualResult40700
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult40695.actual selector witness +
    ResidualResult40213.actual selector witness
end ResidualResult40700

namespace ResidualResult40705
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult40700.actual selector witness +
    ResidualResult39731.actual selector witness
end ResidualResult40705

namespace ResidualResult40710
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult40705.actual selector witness +
    ResidualResult39249.actual selector witness
end ResidualResult40710

namespace ResidualResult40715
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult40710.actual selector witness +
    ResidualResult38767.actual selector witness
end ResidualResult40715

namespace ResidualResult40720
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult40715.actual selector witness +
    ResidualResult38285.actual selector witness
end ResidualResult40720

namespace ResidualResult40725
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult40720.actual selector witness +
    ResidualResult37803.actual selector witness
end ResidualResult40725

namespace ResidualResult40730
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult40725.actual selector witness +
    ResidualResult37321.actual selector witness
end ResidualResult40730

namespace ResidualResult40735
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult40730.actual selector witness +
    ResidualResult36839.actual selector witness
end ResidualResult40735

namespace ResidualResult40740
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult40735.actual selector witness +
    ResidualResult36357.actual selector witness
end ResidualResult40740

namespace ResidualResult40745
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult40740.actual selector witness +
    ResidualResult35875.actual selector witness
end ResidualResult40745

namespace ResidualResult40750
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult40745.actual selector witness +
    ResidualResult35393.actual selector witness
end ResidualResult40750

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert
