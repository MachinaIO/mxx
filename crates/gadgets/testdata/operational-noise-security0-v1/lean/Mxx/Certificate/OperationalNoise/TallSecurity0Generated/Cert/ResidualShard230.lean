import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard165
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard193
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard197
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard200
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard204
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard208
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard211
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard215
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard219
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard222
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard226
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard229

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace ResidualResult30060
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult30032.actual selector witness *
    ResidualResult30055.actual selector witness
end ResidualResult30060

namespace ResidualResult30063
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 30063
end ResidualResult30063

namespace ResidualResult30067
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult30063.actual selector witness -
    ResidualResult30060.actual selector witness
end ResidualResult30067

namespace ResidualResult30071
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult30067.actual selector witness -
    ResidualResult30052.actual selector witness
end ResidualResult30071

namespace ResidualResult30080
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult21512.actual selector witness *
    ResidualResult29909.actual selector witness
end ResidualResult30080

namespace ResidualResult30087
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult30080.actual selector witness +
    ResidualResult29902.actual selector witness
end ResidualResult30087

namespace ResidualResult30092
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult30087.actual selector witness +
    ResidualResult29605.actual selector witness
end ResidualResult30092

namespace ResidualResult30097
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult30092.actual selector witness +
    ResidualResult29123.actual selector witness
end ResidualResult30097

namespace ResidualResult30102
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult30097.actual selector witness +
    ResidualResult28641.actual selector witness
end ResidualResult30102

namespace ResidualResult30107
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult30102.actual selector witness +
    ResidualResult28159.actual selector witness
end ResidualResult30107

namespace ResidualResult30112
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult30107.actual selector witness +
    ResidualResult27677.actual selector witness
end ResidualResult30112

namespace ResidualResult30117
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult30112.actual selector witness +
    ResidualResult27195.actual selector witness
end ResidualResult30117

namespace ResidualResult30122
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult30117.actual selector witness +
    ResidualResult26713.actual selector witness
end ResidualResult30122

namespace ResidualResult30127
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult30122.actual selector witness +
    ResidualResult26231.actual selector witness
end ResidualResult30127

namespace ResidualResult30132
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult30127.actual selector witness +
    ResidualResult25749.actual selector witness
end ResidualResult30132

namespace ResidualResult30137
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult30132.actual selector witness +
    ResidualResult25267.actual selector witness
end ResidualResult30137

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert
