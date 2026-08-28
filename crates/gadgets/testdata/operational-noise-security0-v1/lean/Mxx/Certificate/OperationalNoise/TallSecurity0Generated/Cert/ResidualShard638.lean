import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard636
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard637

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace ResidualResult89956
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult89952.actual selector witness +
    ResidualResult89893.actual selector witness
end ResidualResult89956

namespace ResidualResult89960
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult89956.actual selector witness +
    ResidualResult89890.actual selector witness
end ResidualResult89960

namespace ResidualResult89964
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult89960.actual selector witness +
    ResidualResult89887.actual selector witness
end ResidualResult89964

namespace ResidualResult89968
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult89964.actual selector witness +
    ResidualResult89884.actual selector witness
end ResidualResult89968

namespace ResidualResult89972
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult89968.actual selector witness +
    ResidualResult89881.actual selector witness
end ResidualResult89972

namespace ResidualResult89976
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult89972.actual selector witness +
    ResidualResult89878.actual selector witness
end ResidualResult89976

namespace ResidualResult89980
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult89976.actual selector witness +
    ResidualResult89875.actual selector witness
end ResidualResult89980

namespace ResidualResult89984
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult89980.actual selector witness +
    ResidualResult89872.actual selector witness
end ResidualResult89984

namespace ResidualResult89988
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult89984.actual selector witness +
    ResidualResult89869.actual selector witness
end ResidualResult89988

namespace ResidualResult89992
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult89988.actual selector witness -
    ResidualResult89866.actual selector witness
end ResidualResult89992

namespace ResidualResult90068
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult89992.actual selector witness *
    ResidualResult89833.actual selector witness
end ResidualResult90068

namespace ResidualResult90071
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 90071
end ResidualResult90071

namespace ResidualResult90076
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult89844.actual selector witness *
    ResidualResult90071.actual selector witness
end ResidualResult90076

namespace ResidualResult90079
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 90079
end ResidualResult90079

namespace ResidualResult90083
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult90079.actual selector witness -
    ResidualResult90076.actual selector witness
end ResidualResult90083

namespace ResidualResult90087
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult90083.actual selector witness -
    ResidualResult90068.actual selector witness
end ResidualResult90087

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert
