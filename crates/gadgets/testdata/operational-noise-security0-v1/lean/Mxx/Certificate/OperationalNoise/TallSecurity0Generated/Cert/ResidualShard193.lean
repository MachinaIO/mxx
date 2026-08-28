import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard008
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard089
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard163
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard164
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard165
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard192

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace ResidualResult25232
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult25224.actual selector witness *
    ResidualResult25201.actual selector witness
end ResidualResult25232

namespace ResidualResult25235
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 25235
end ResidualResult25235

namespace ResidualResult25240
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult25212.actual selector witness *
    ResidualResult25235.actual selector witness
end ResidualResult25240

namespace ResidualResult25243
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 25243
end ResidualResult25243

namespace ResidualResult25247
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult25243.actual selector witness -
    ResidualResult25240.actual selector witness
end ResidualResult25247

namespace ResidualResult25251
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult25247.actual selector witness -
    ResidualResult25232.actual selector witness
end ResidualResult25251

namespace ResidualResult25260
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult21512.actual selector witness *
    ResidualResult25089.actual selector witness
end ResidualResult25260

namespace ResidualResult25267
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult25260.actual selector witness +
    ResidualResult25082.actual selector witness
end ResidualResult25267

namespace ResidualResult25274
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 25274
end ResidualResult25274

namespace ResidualResult25277
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 25277
end ResidualResult25277

namespace ResidualResult25284
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 25284
end ResidualResult25284

namespace ResidualResult25287
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 25287
end ResidualResult25287

namespace ResidualResult25292
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult1026.actual selector witness *
    ResidualResult21420.actual selector witness
end ResidualResult25292

namespace ResidualResult25297
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult21290.actual selector witness *
    ResidualResult10480.actual selector witness
end ResidualResult25297

namespace ResidualResult25301
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult25297.actual selector witness -
    ResidualResult25292.actual selector witness
end ResidualResult25301

namespace ResidualResult25307
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult25301.actual selector witness +
    ResidualResult10472.actual selector witness
end ResidualResult25307

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert
