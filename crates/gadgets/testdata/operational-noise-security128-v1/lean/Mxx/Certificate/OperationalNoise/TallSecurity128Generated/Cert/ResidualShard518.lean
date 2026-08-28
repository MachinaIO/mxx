import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard453
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard485
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard488
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard492
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard496
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard499
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard503
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard507
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard511
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard514
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard517

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace ResidualResult69913
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  witness.honestTerminalActual 69913
end ResidualResult69913

namespace ResidualResult69918
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult69890.actual selector witness *
    ResidualResult69913.actual selector witness
end ResidualResult69918

namespace ResidualResult69921
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  witness.honestTerminalActual 69921
end ResidualResult69921

namespace ResidualResult69925
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult69921.actual selector witness -
    ResidualResult69918.actual selector witness
end ResidualResult69925

namespace ResidualResult69929
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult69925.actual selector witness -
    ResidualResult69910.actual selector witness
end ResidualResult69929

namespace ResidualResult69938
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult61370.actual selector witness *
    ResidualResult69767.actual selector witness
end ResidualResult69938

namespace ResidualResult69945
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult69938.actual selector witness +
    ResidualResult69760.actual selector witness
end ResidualResult69945

namespace ResidualResult69950
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult69945.actual selector witness +
    ResidualResult69463.actual selector witness
end ResidualResult69950

namespace ResidualResult69955
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult69950.actual selector witness +
    ResidualResult68981.actual selector witness
end ResidualResult69955

namespace ResidualResult69960
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult69955.actual selector witness +
    ResidualResult68499.actual selector witness
end ResidualResult69960

namespace ResidualResult69965
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult69960.actual selector witness +
    ResidualResult68017.actual selector witness
end ResidualResult69965

namespace ResidualResult69970
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult69965.actual selector witness +
    ResidualResult67535.actual selector witness
end ResidualResult69970

namespace ResidualResult69975
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult69970.actual selector witness +
    ResidualResult67053.actual selector witness
end ResidualResult69975

namespace ResidualResult69980
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult69975.actual selector witness +
    ResidualResult66571.actual selector witness
end ResidualResult69980

namespace ResidualResult69985
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult69980.actual selector witness +
    ResidualResult66089.actual selector witness
end ResidualResult69985

namespace ResidualResult69990
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult69985.actual selector witness +
    ResidualResult65607.actual selector witness
end ResidualResult69990

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert
