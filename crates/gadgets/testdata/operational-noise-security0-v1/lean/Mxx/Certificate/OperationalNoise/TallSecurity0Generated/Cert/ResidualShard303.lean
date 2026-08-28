import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard265
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard301
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard302

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace ResidualResult41073
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult41068.actual selector witness *
    ResidualResult41066.actual selector witness
end ResidualResult41073

namespace ResidualResult41078
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 41078
end ResidualResult41078

namespace ResidualResult41084
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 41084
end ResidualResult41084

namespace ResidualResult41088
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 41088
end ResidualResult41088

namespace ResidualResult41091
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 41091
end ResidualResult41091

namespace ResidualResult41096
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult41091.actual selector witness *
    ResidualResult41088.actual selector witness
end ResidualResult41096

namespace ResidualResult41100
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult41096.actual selector witness -
    ResidualResult41073.actual selector witness
end ResidualResult41100

namespace ResidualResult41108
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult41100.actual selector witness *
    ResidualResult41057.actual selector witness
end ResidualResult41108

namespace ResidualResult41111
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 41111
end ResidualResult41111

namespace ResidualResult41116
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult41068.actual selector witness *
    ResidualResult41111.actual selector witness
end ResidualResult41116

namespace ResidualResult41119
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 41119
end ResidualResult41119

namespace ResidualResult41123
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult41119.actual selector witness -
    ResidualResult41116.actual selector witness
end ResidualResult41123

namespace ResidualResult41127
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult41123.actual selector witness -
    ResidualResult41108.actual selector witness
end ResidualResult41127

namespace ResidualResult41136
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult36137.actual selector witness *
    ResidualResult40957.actual selector witness
end ResidualResult41136

namespace ResidualResult41143
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult41136.actual selector witness +
    ResidualResult40950.actual selector witness
end ResidualResult41143

namespace ResidualResult41153
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult41143.actual selector witness *
    ResidualResult40866.actual selector witness
end ResidualResult41153

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert
