import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard366
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard367

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace ResidualResult51086
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 51086
end ResidualResult51086

namespace ResidualResult51088
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 51088
end ResidualResult51088

namespace ResidualResult51093
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult51088.actual selector witness *
    ResidualResult51086.actual selector witness
end ResidualResult51093

namespace ResidualResult51096
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 51096
end ResidualResult51096

namespace ResidualResult51100
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult51096.actual selector witness -
    ResidualResult51093.actual selector witness
end ResidualResult51100

namespace ResidualResult51108
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult51100.actual selector witness *
    ResidualResult51077.actual selector witness
end ResidualResult51108

namespace ResidualResult51111
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 51111
end ResidualResult51111

namespace ResidualResult51116
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult51088.actual selector witness *
    ResidualResult51111.actual selector witness
end ResidualResult51116

namespace ResidualResult51119
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 51119
end ResidualResult51119

namespace ResidualResult51123
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult51119.actual selector witness -
    ResidualResult51116.actual selector witness
end ResidualResult51123

namespace ResidualResult51127
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult51123.actual selector witness -
    ResidualResult51108.actual selector witness
end ResidualResult51127

namespace ResidualResult51136
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult50762.actual selector witness *
    ResidualResult50965.actual selector witness
end ResidualResult51136

namespace ResidualResult51143
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult51136.actual selector witness +
    ResidualResult50958.actual selector witness
end ResidualResult51143

namespace ResidualResult51150
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 51150
end ResidualResult51150

namespace ResidualResult51153
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 51153
end ResidualResult51153

namespace ResidualResult51160
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 51160
end ResidualResult51160

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert
