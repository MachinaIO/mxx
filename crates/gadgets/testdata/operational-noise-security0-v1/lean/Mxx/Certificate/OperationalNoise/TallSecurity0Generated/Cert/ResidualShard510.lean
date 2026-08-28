import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard026
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard105
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard106
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard465

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace ResidualResult71080
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 71080
end ResidualResult71080

namespace ResidualResult71087
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 71087
end ResidualResult71087

namespace ResidualResult71090
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 71090
end ResidualResult71090

namespace ResidualResult71095
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult3362.actual selector witness *
    ResidualResult65295.actual selector witness
end ResidualResult71095

namespace ResidualResult71100
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult65165.actual selector witness *
    ResidualResult12484.actual selector witness
end ResidualResult71100

namespace ResidualResult71104
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult71100.actual selector witness -
    ResidualResult71095.actual selector witness
end ResidualResult71104

namespace ResidualResult71110
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult71104.actual selector witness +
    ResidualResult12476.actual selector witness
end ResidualResult71110

namespace ResidualResult71118
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult71110.actual selector witness *
    ResidualResult3365.actual selector witness
end ResidualResult71118

namespace ResidualResult71123
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult3365.actual selector witness *
    ResidualResult65295.actual selector witness
end ResidualResult71123

namespace ResidualResult71128
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult65165.actual selector witness *
    ResidualResult12525.actual selector witness
end ResidualResult71128

namespace ResidualResult71132
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult71128.actual selector witness -
    ResidualResult71123.actual selector witness
end ResidualResult71132

namespace ResidualResult71138
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult71132.actual selector witness +
    ResidualResult12517.actual selector witness
end ResidualResult71138

namespace ResidualResult71148
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult71138.actual selector witness *
    ResidualResult12514.actual selector witness
end ResidualResult71148

namespace ResidualResult71154
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult71148.actual selector witness +
    ResidualResult71118.actual selector witness
end ResidualResult71154

namespace ResidualResult71164
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult71154.actual selector witness *
    ResidualResult71090.actual selector witness
end ResidualResult71164

namespace ResidualResult71167
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 71167
end ResidualResult71167

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert
