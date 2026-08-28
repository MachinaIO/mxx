import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard026
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard081
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard465

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace ResidualResult68198
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 68198
end ResidualResult68198

namespace ResidualResult68203
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult3224.actual selector witness *
    ResidualResult65295.actual selector witness
end ResidualResult68203

namespace ResidualResult68208
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult65165.actual selector witness *
    ResidualResult9478.actual selector witness
end ResidualResult68208

namespace ResidualResult68212
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult68208.actual selector witness -
    ResidualResult68203.actual selector witness
end ResidualResult68212

namespace ResidualResult68218
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult68212.actual selector witness +
    ResidualResult9470.actual selector witness
end ResidualResult68218

namespace ResidualResult68226
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult68218.actual selector witness *
    ResidualResult3227.actual selector witness
end ResidualResult68226

namespace ResidualResult68231
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult3227.actual selector witness *
    ResidualResult65295.actual selector witness
end ResidualResult68231

namespace ResidualResult68236
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult65165.actual selector witness *
    ResidualResult9519.actual selector witness
end ResidualResult68236

namespace ResidualResult68240
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult68236.actual selector witness -
    ResidualResult68231.actual selector witness
end ResidualResult68240

namespace ResidualResult68246
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult68240.actual selector witness +
    ResidualResult9511.actual selector witness
end ResidualResult68246

namespace ResidualResult68256
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult68246.actual selector witness *
    ResidualResult9508.actual selector witness
end ResidualResult68256

namespace ResidualResult68262
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult68256.actual selector witness +
    ResidualResult68226.actual selector witness
end ResidualResult68262

namespace ResidualResult68272
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult68262.actual selector witness *
    ResidualResult68198.actual selector witness
end ResidualResult68272

namespace ResidualResult68275
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 68275
end ResidualResult68275

namespace ResidualResult68279
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 68279
end ResidualResult68279

namespace ResidualResult68357
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 68357
end ResidualResult68357

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert
