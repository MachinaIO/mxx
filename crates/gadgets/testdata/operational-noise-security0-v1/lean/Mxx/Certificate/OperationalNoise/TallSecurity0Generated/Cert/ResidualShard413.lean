import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard021
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard109
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard110
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard364
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard365
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard411
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard412

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace ResidualResult56927
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult56920.actual selector witness +
    ResidualResult56742.actual selector witness
end ResidualResult56927

namespace ResidualResult56934
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 56934
end ResidualResult56934

namespace ResidualResult56937
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 56937
end ResidualResult56937

namespace ResidualResult56944
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 56944
end ResidualResult56944

namespace ResidualResult56947
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 56947
end ResidualResult56947

namespace ResidualResult56952
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult2637.actual selector witness *
    ResidualResult50670.actual selector witness
end ResidualResult56952

namespace ResidualResult56957
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult50540.actual selector witness *
    ResidualResult12985.actual selector witness
end ResidualResult56957

namespace ResidualResult56961
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult56957.actual selector witness -
    ResidualResult56952.actual selector witness
end ResidualResult56961

namespace ResidualResult56967
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult56961.actual selector witness +
    ResidualResult12977.actual selector witness
end ResidualResult56967

namespace ResidualResult56975
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult56967.actual selector witness *
    ResidualResult2640.actual selector witness
end ResidualResult56975

namespace ResidualResult56980
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult2640.actual selector witness *
    ResidualResult50670.actual selector witness
end ResidualResult56980

namespace ResidualResult56985
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult50540.actual selector witness *
    ResidualResult13026.actual selector witness
end ResidualResult56985

namespace ResidualResult56989
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult56985.actual selector witness -
    ResidualResult56980.actual selector witness
end ResidualResult56989

namespace ResidualResult56995
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult56989.actual selector witness +
    ResidualResult13018.actual selector witness
end ResidualResult56995

namespace ResidualResult57005
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult56995.actual selector witness *
    ResidualResult13015.actual selector witness
end ResidualResult57005

namespace ResidualResult57011
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult57005.actual selector witness +
    ResidualResult56975.actual selector witness
end ResidualResult57011

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert
