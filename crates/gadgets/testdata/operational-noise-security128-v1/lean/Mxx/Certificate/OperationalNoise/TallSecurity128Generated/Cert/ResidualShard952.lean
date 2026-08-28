import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard006
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard128
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard248
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard853
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard854
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard938
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard939
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard940
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard942
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard943
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard945
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard946
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard947
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard949
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard950
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard951

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace ResidualResult133896
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult723.actual selector witness *
    ResidualResult119778.actual selector witness
end ResidualResult133896

namespace ResidualResult133901
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult119648.actual selector witness *
    ResidualResult15896.actual selector witness
end ResidualResult133901

namespace ResidualResult133905
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult133901.actual selector witness -
    ResidualResult133896.actual selector witness
end ResidualResult133905

namespace ResidualResult133911
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult133905.actual selector witness +
    ResidualResult31516.actual selector witness
end ResidualResult133911

namespace ResidualResult133918
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult133911.actual selector witness -
    ResidualResult133911.actual selector witness
end ResidualResult133918

namespace ResidualResult133923
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult133918.actual selector witness +
    ResidualResult133891.actual selector witness
end ResidualResult133923

namespace ResidualResult133928
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult133923.actual selector witness +
    ResidualResult133679.actual selector witness
end ResidualResult133928

namespace ResidualResult133933
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult133928.actual selector witness +
    ResidualResult133467.actual selector witness
end ResidualResult133933

namespace ResidualResult133938
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult133933.actual selector witness +
    ResidualResult133255.actual selector witness
end ResidualResult133938

namespace ResidualResult133943
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult133938.actual selector witness +
    ResidualResult133043.actual selector witness
end ResidualResult133943

namespace ResidualResult133948
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult133943.actual selector witness +
    ResidualResult132831.actual selector witness
end ResidualResult133948

namespace ResidualResult133953
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult133948.actual selector witness +
    ResidualResult132619.actual selector witness
end ResidualResult133953

namespace ResidualResult133958
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult133953.actual selector witness +
    ResidualResult132407.actual selector witness
end ResidualResult133958

namespace ResidualResult133963
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult133958.actual selector witness +
    ResidualResult132195.actual selector witness
end ResidualResult133963

namespace ResidualResult133968
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult133963.actual selector witness +
    ResidualResult131983.actual selector witness
end ResidualResult133968

namespace ResidualResult133973
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  ResidualResult133968.actual selector witness +
    ResidualResult131771.actual selector witness
end ResidualResult133973

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert
