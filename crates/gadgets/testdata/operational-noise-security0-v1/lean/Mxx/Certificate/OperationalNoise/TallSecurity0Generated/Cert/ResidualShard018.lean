import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard005
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard006
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard016
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard017

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace ResidualResult2219
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 2219
end ResidualResult2219

namespace ResidualResult2224
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult2219.actual selector witness *
    ResidualResult713.actual selector witness
end ResidualResult2224

namespace ResidualResult2228
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult728.actual selector witness +
    ResidualResult2224.actual selector witness
end ResidualResult2228

namespace ResidualResult2232
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult2228.actual selector witness +
    ResidualResult2216.actual selector witness
end ResidualResult2232

namespace ResidualResult2236
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult2232.actual selector witness +
    ResidualResult2208.actual selector witness
end ResidualResult2236

namespace ResidualResult2240
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult2236.actual selector witness +
    ResidualResult2200.actual selector witness
end ResidualResult2240

namespace ResidualResult2244
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult2240.actual selector witness +
    ResidualResult2192.actual selector witness
end ResidualResult2244

namespace ResidualResult2248
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult2244.actual selector witness +
    ResidualResult2184.actual selector witness
end ResidualResult2248

namespace ResidualResult2252
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult2248.actual selector witness +
    ResidualResult2176.actual selector witness
end ResidualResult2252

namespace ResidualResult2256
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult2252.actual selector witness +
    ResidualResult2168.actual selector witness
end ResidualResult2256

namespace ResidualResult2260
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult2256.actual selector witness +
    ResidualResult2160.actual selector witness
end ResidualResult2260

namespace ResidualResult2264
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult2260.actual selector witness +
    ResidualResult2152.actual selector witness
end ResidualResult2264

namespace ResidualResult2268
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult2264.actual selector witness +
    ResidualResult2144.actual selector witness
end ResidualResult2268

namespace ResidualResult2272
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult2268.actual selector witness +
    ResidualResult2136.actual selector witness
end ResidualResult2272

namespace ResidualResult2276
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult2272.actual selector witness +
    ResidualResult2128.actual selector witness
end ResidualResult2276

namespace ResidualResult2280
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult2276.actual selector witness +
    ResidualResult2120.actual selector witness
end ResidualResult2280

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert
