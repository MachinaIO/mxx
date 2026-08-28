import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard033
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard125
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard126
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard565
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard566

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace ResidualResult88081
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 88081
end ResidualResult88081

namespace ResidualResult88088
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 88088
end ResidualResult88088

namespace ResidualResult88091
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 88091
end ResidualResult88091

namespace ResidualResult88096
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult4219.actual selector witness *
    ResidualResult79920.actual selector witness
end ResidualResult88096

namespace ResidualResult88101
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult79790.actual selector witness *
    ResidualResult14989.actual selector witness
end ResidualResult88101

namespace ResidualResult88105
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult88101.actual selector witness -
    ResidualResult88096.actual selector witness
end ResidualResult88105

namespace ResidualResult88111
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult88105.actual selector witness +
    ResidualResult14981.actual selector witness
end ResidualResult88111

namespace ResidualResult88119
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult88111.actual selector witness *
    ResidualResult4222.actual selector witness
end ResidualResult88119

namespace ResidualResult88124
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult4222.actual selector witness *
    ResidualResult79920.actual selector witness
end ResidualResult88124

namespace ResidualResult88129
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult79790.actual selector witness *
    ResidualResult15030.actual selector witness
end ResidualResult88129

namespace ResidualResult88133
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult88129.actual selector witness -
    ResidualResult88124.actual selector witness
end ResidualResult88133

namespace ResidualResult88139
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult88133.actual selector witness +
    ResidualResult15022.actual selector witness
end ResidualResult88139

namespace ResidualResult88149
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult88139.actual selector witness *
    ResidualResult15019.actual selector witness
end ResidualResult88149

namespace ResidualResult88155
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult88149.actual selector witness +
    ResidualResult88119.actual selector witness
end ResidualResult88155

namespace ResidualResult88165
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult88155.actual selector witness *
    ResidualResult88091.actual selector witness
end ResidualResult88165

namespace ResidualResult88168
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 88168
end ResidualResult88168

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert
