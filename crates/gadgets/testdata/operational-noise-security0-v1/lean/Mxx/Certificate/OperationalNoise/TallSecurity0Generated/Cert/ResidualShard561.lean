import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard049
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard466
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard560

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace ResidualResult79117
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 79117
end ResidualResult79117

namespace ResidualResult79120
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 79120
end ResidualResult79120

namespace ResidualResult79129
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 79129
end ResidualResult79129

namespace ResidualResult79131
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 79131
end ResidualResult79131

namespace ResidualResult79136
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult79131.actual selector witness *
    ResidualResult79129.actual selector witness
end ResidualResult79136

namespace ResidualResult79139
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 79139
end ResidualResult79139

namespace ResidualResult79143
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult79139.actual selector witness -
    ResidualResult79136.actual selector witness
end ResidualResult79143

namespace ResidualResult79151
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult79143.actual selector witness *
    ResidualResult79120.actual selector witness
end ResidualResult79151

namespace ResidualResult79154
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 79154
end ResidualResult79154

namespace ResidualResult79159
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult79131.actual selector witness *
    ResidualResult79154.actual selector witness
end ResidualResult79159

namespace ResidualResult79162
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 79162
end ResidualResult79162

namespace ResidualResult79166
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult79162.actual selector witness -
    ResidualResult79159.actual selector witness
end ResidualResult79166

namespace ResidualResult79170
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult79166.actual selector witness -
    ResidualResult79151.actual selector witness
end ResidualResult79170

namespace ResidualResult79179
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult65387.actual selector witness *
    ResidualResult79008.actual selector witness
end ResidualResult79179

namespace ResidualResult79186
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult79179.actual selector witness +
    ResidualResult79001.actual selector witness
end ResidualResult79186

namespace ResidualResult79196
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult79186.actual selector witness *
    ResidualResult5839.actual selector witness
end ResidualResult79196

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert
