import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard058
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard148

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace ResidualResult19119
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 19119
end ResidualResult19119

namespace ResidualResult19130
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 19130
end ResidualResult19130

namespace ResidualResult19133
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 19133
end ResidualResult19133

namespace ResidualResult19142
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 19142
end ResidualResult19142

namespace ResidualResult19144
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 19144
end ResidualResult19144

namespace ResidualResult19149
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult19144.actual selector witness *
    ResidualResult19142.actual selector witness
end ResidualResult19149

namespace ResidualResult19152
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 19152
end ResidualResult19152

namespace ResidualResult19156
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult19152.actual selector witness -
    ResidualResult19149.actual selector witness
end ResidualResult19156

namespace ResidualResult19164
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult19156.actual selector witness *
    ResidualResult19133.actual selector witness
end ResidualResult19164

namespace ResidualResult19167
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 19167
end ResidualResult19167

namespace ResidualResult19172
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult19144.actual selector witness *
    ResidualResult19167.actual selector witness
end ResidualResult19172

namespace ResidualResult19175
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 19175
end ResidualResult19175

namespace ResidualResult19179
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult19175.actual selector witness -
    ResidualResult19172.actual selector witness
end ResidualResult19179

namespace ResidualResult19183
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult19179.actual selector witness -
    ResidualResult19164.actual selector witness
end ResidualResult19183

namespace ResidualResult19192
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult6561.actual selector witness *
    ResidualResult19021.actual selector witness
end ResidualResult19192

namespace ResidualResult19199
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult19192.actual selector witness +
    ResidualResult19014.actual selector witness
end ResidualResult19199

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert
