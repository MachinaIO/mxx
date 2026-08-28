import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard006
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard049
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard050
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard055
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard056
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard150
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard151
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard152
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard154
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard155
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard156
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard158
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard159
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard160

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace ResidualResult20905
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult20895.actual selector witness *
    ResidualResult5859.actual selector witness
end ResidualResult20905

namespace ResidualResult20908
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 20908
end ResidualResult20908

namespace ResidualResult20913
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult723.actual selector witness *
    ResidualResult6449.actual selector witness
end ResidualResult20913

namespace ResidualResult20918
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult6314.actual selector witness *
    ResidualResult5873.actual selector witness
end ResidualResult20918

namespace ResidualResult20922
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult20918.actual selector witness -
    ResidualResult20913.actual selector witness
end ResidualResult20922

namespace ResidualResult20928
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult20922.actual selector witness +
    ResidualResult20908.actual selector witness
end ResidualResult20928

namespace ResidualResult20935
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult20928.actual selector witness -
    ResidualResult20928.actual selector witness
end ResidualResult20935

namespace ResidualResult20940
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult20935.actual selector witness +
    ResidualResult20905.actual selector witness
end ResidualResult20940

namespace ResidualResult20945
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult20940.actual selector witness +
    ResidualResult20693.actual selector witness
end ResidualResult20945

namespace ResidualResult20950
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult20945.actual selector witness +
    ResidualResult20481.actual selector witness
end ResidualResult20950

namespace ResidualResult20955
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult20950.actual selector witness +
    ResidualResult20269.actual selector witness
end ResidualResult20955

namespace ResidualResult20960
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult20955.actual selector witness +
    ResidualResult20057.actual selector witness
end ResidualResult20960

namespace ResidualResult20965
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult20960.actual selector witness +
    ResidualResult19845.actual selector witness
end ResidualResult20965

namespace ResidualResult20970
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult20965.actual selector witness +
    ResidualResult19633.actual selector witness
end ResidualResult20970

namespace ResidualResult20975
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult20970.actual selector witness +
    ResidualResult19421.actual selector witness
end ResidualResult20975

namespace ResidualResult20980
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult20975.actual selector witness +
    ResidualResult19209.actual selector witness
end ResidualResult20980

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert
