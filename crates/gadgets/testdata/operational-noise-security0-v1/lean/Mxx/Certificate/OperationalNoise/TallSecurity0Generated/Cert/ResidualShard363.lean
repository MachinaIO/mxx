import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard000
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard052
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard264
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard337
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard338
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard340
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard341
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard342
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard344
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard345
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard347
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard348
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard349
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard351
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard352
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard362

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace ResidualResult50225
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult50220.actual selector witness +
    ResidualResult48674.actual selector witness
end ResidualResult50225

namespace ResidualResult50230
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult50225.actual selector witness +
    ResidualResult48462.actual selector witness
end ResidualResult50230

namespace ResidualResult50235
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult50230.actual selector witness +
    ResidualResult48250.actual selector witness
end ResidualResult50235

namespace ResidualResult50240
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult50235.actual selector witness +
    ResidualResult48038.actual selector witness
end ResidualResult50240

namespace ResidualResult50245
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult50240.actual selector witness +
    ResidualResult47826.actual selector witness
end ResidualResult50245

namespace ResidualResult50250
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult50245.actual selector witness +
    ResidualResult47614.actual selector witness
end ResidualResult50250

namespace ResidualResult50255
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult50250.actual selector witness +
    ResidualResult47402.actual selector witness
end ResidualResult50255

namespace ResidualResult50260
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult50255.actual selector witness +
    ResidualResult47190.actual selector witness
end ResidualResult50260

namespace ResidualResult50265
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult50260.actual selector witness +
    ResidualResult46978.actual selector witness
end ResidualResult50265

namespace ResidualResult50270
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult50265.actual selector witness +
    ResidualResult46766.actual selector witness
end ResidualResult50270

namespace ResidualResult50275
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult50270.actual selector witness +
    ResidualResult46554.actual selector witness
end ResidualResult50275

namespace ResidualResult50280
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult50275.actual selector witness -
    ResidualResult46342.actual selector witness
end ResidualResult50280

namespace ResidualResult50282
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 50282
end ResidualResult50282

namespace ResidualResult50287
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult27.actual selector witness *
    ResidualResult6044.actual selector witness
end ResidualResult50287

namespace ResidualResult50291
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult50287.actual selector witness -
    ResidualResult36045.actual selector witness
end ResidualResult50291

namespace ResidualResult50297
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult50291.actual selector witness +
    ResidualResult50282.actual selector witness
end ResidualResult50297

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert
