import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard058
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard085
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard086

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace ResidualResult10184
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult10179.actual selector witness *
    ResidualResult10177.actual selector witness
end ResidualResult10184

namespace ResidualResult10189
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 10189
end ResidualResult10189

namespace ResidualResult10195
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 10195
end ResidualResult10195

namespace ResidualResult10199
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 10199
end ResidualResult10199

namespace ResidualResult10202
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 10202
end ResidualResult10202

namespace ResidualResult10207
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult10202.actual selector witness *
    ResidualResult10199.actual selector witness
end ResidualResult10207

namespace ResidualResult10211
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult10207.actual selector witness -
    ResidualResult10184.actual selector witness
end ResidualResult10211

namespace ResidualResult10219
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult10211.actual selector witness *
    ResidualResult10168.actual selector witness
end ResidualResult10219

namespace ResidualResult10222
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 10222
end ResidualResult10222

namespace ResidualResult10227
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult10179.actual selector witness *
    ResidualResult10222.actual selector witness
end ResidualResult10227

namespace ResidualResult10230
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 10230
end ResidualResult10230

namespace ResidualResult10234
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult10230.actual selector witness -
    ResidualResult10227.actual selector witness
end ResidualResult10234

namespace ResidualResult10238
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult10234.actual selector witness -
    ResidualResult10219.actual selector witness
end ResidualResult10238

namespace ResidualResult10247
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult6561.actual selector witness *
    ResidualResult10068.actual selector witness
end ResidualResult10247

namespace ResidualResult10254
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult10247.actual selector witness +
    ResidualResult10061.actual selector witness
end ResidualResult10254

namespace ResidualResult10264
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult10254.actual selector witness *
    ResidualResult9958.actual selector witness
end ResidualResult10264

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert
