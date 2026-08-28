import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard021
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard118
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard364
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard365
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard420

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace ResidualResult57944
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult2686.actual selector witness *
    ResidualResult50670.actual selector witness
end ResidualResult57944

namespace ResidualResult57949
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult50540.actual selector witness *
    ResidualResult14028.actual selector witness
end ResidualResult57949

namespace ResidualResult57953
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult57949.actual selector witness -
    ResidualResult57944.actual selector witness
end ResidualResult57953

namespace ResidualResult57959
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult57953.actual selector witness +
    ResidualResult14020.actual selector witness
end ResidualResult57959

namespace ResidualResult57969
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult57959.actual selector witness *
    ResidualResult14017.actual selector witness
end ResidualResult57969

namespace ResidualResult57975
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult57969.actual selector witness +
    ResidualResult57939.actual selector witness
end ResidualResult57975

namespace ResidualResult57985
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult57975.actual selector witness *
    ResidualResult57911.actual selector witness
end ResidualResult57985

namespace ResidualResult57988
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 57988
end ResidualResult57988

namespace ResidualResult57992
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 57992
end ResidualResult57992

namespace ResidualResult58070
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 58070
end ResidualResult58070

namespace ResidualResult58073
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 58073
end ResidualResult58073

namespace ResidualResult58078
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult58073.actual selector witness *
    ResidualResult58070.actual selector witness
end ResidualResult58078

namespace ResidualResult58089
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 58089
end ResidualResult58089

namespace ResidualResult58092
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 58092
end ResidualResult58092

namespace ResidualResult58101
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 58101
end ResidualResult58101

namespace ResidualResult58103
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 58103
end ResidualResult58103

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert
