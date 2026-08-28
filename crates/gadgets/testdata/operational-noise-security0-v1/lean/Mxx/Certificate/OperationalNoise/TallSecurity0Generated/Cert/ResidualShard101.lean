import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard001
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard055
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard056
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard058
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard100

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace ResidualResult11928
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 11928
end ResidualResult11928

namespace ResidualResult11932
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult11928.actual selector witness -
    ResidualResult11925.actual selector witness
end ResidualResult11932

namespace ResidualResult11936
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult11932.actual selector witness -
    ResidualResult11917.actual selector witness
end ResidualResult11936

namespace ResidualResult11945
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult6561.actual selector witness *
    ResidualResult11774.actual selector witness
end ResidualResult11945

namespace ResidualResult11952
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult11945.actual selector witness +
    ResidualResult11767.actual selector witness
end ResidualResult11952

namespace ResidualResult11959
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 11959
end ResidualResult11959

namespace ResidualResult11962
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 11962
end ResidualResult11962

namespace ResidualResult11969
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 11969
end ResidualResult11969

namespace ResidualResult11972
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 11972
end ResidualResult11972

namespace ResidualResult11975
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 11975
end ResidualResult11975

namespace ResidualResult11980
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult304.actual selector witness *
    ResidualResult6449.actual selector witness
end ResidualResult11980

namespace ResidualResult11983
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 11983
end ResidualResult11983

namespace ResidualResult11988
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult6314.actual selector witness *
    ResidualResult11983.actual selector witness
end ResidualResult11988

namespace ResidualResult11992
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult11988.actual selector witness -
    ResidualResult11980.actual selector witness
end ResidualResult11992

namespace ResidualResult11998
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult11992.actual selector witness +
    ResidualResult11975.actual selector witness
end ResidualResult11998

namespace ResidualResult12006
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult11998.actual selector witness *
    ResidualResult307.actual selector witness
end ResidualResult12006

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert
