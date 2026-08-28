import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard020
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard101
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard102
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard364
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard365
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard405

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace ResidualResult55993
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult50540.actual selector witness *
    ResidualResult11983.actual selector witness
end ResidualResult55993

namespace ResidualResult55997
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult55993.actual selector witness -
    ResidualResult55988.actual selector witness
end ResidualResult55997

namespace ResidualResult56003
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult55997.actual selector witness +
    ResidualResult11975.actual selector witness
end ResidualResult56003

namespace ResidualResult56011
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult56003.actual selector witness *
    ResidualResult2594.actual selector witness
end ResidualResult56011

namespace ResidualResult56016
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult2594.actual selector witness *
    ResidualResult50670.actual selector witness
end ResidualResult56016

namespace ResidualResult56021
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult50540.actual selector witness *
    ResidualResult12024.actual selector witness
end ResidualResult56021

namespace ResidualResult56025
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult56021.actual selector witness -
    ResidualResult56016.actual selector witness
end ResidualResult56025

namespace ResidualResult56031
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult56025.actual selector witness +
    ResidualResult12016.actual selector witness
end ResidualResult56031

namespace ResidualResult56041
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult56031.actual selector witness *
    ResidualResult12013.actual selector witness
end ResidualResult56041

namespace ResidualResult56047
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult56041.actual selector witness +
    ResidualResult56011.actual selector witness
end ResidualResult56047

namespace ResidualResult56057
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult56047.actual selector witness *
    ResidualResult55983.actual selector witness
end ResidualResult56057

namespace ResidualResult56060
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 56060
end ResidualResult56060

namespace ResidualResult56064
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 56064
end ResidualResult56064

namespace ResidualResult56142
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 56142
end ResidualResult56142

namespace ResidualResult56145
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 56145
end ResidualResult56145

namespace ResidualResult56150
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult56145.actual selector witness *
    ResidualResult56142.actual selector witness
end ResidualResult56150

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert
