import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard031
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard065
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard565
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard566
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard573

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace ResidualResult80905
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult80901.actual selector witness -
    ResidualResult80896.actual selector witness
end ResidualResult80905

namespace ResidualResult80911
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult80905.actual selector witness +
    ResidualResult7466.actual selector witness
end ResidualResult80911

namespace ResidualResult80919
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult80911.actual selector witness *
    ResidualResult3877.actual selector witness
end ResidualResult80919

namespace ResidualResult80924
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult3877.actual selector witness *
    ResidualResult79920.actual selector witness
end ResidualResult80924

namespace ResidualResult80929
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult79790.actual selector witness *
    ResidualResult7515.actual selector witness
end ResidualResult80929

namespace ResidualResult80933
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult80929.actual selector witness -
    ResidualResult80924.actual selector witness
end ResidualResult80933

namespace ResidualResult80939
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult80933.actual selector witness +
    ResidualResult7507.actual selector witness
end ResidualResult80939

namespace ResidualResult80949
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult80939.actual selector witness *
    ResidualResult7504.actual selector witness
end ResidualResult80949

namespace ResidualResult80955
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult80949.actual selector witness +
    ResidualResult80919.actual selector witness
end ResidualResult80955

namespace ResidualResult80965
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult80955.actual selector witness *
    ResidualResult80891.actual selector witness
end ResidualResult80965

namespace ResidualResult80968
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 80968
end ResidualResult80968

namespace ResidualResult80972
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 80972
end ResidualResult80972

namespace ResidualResult81050
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 81050
end ResidualResult81050

namespace ResidualResult81053
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 81053
end ResidualResult81053

namespace ResidualResult81058
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult81053.actual selector witness *
    ResidualResult81050.actual selector witness
end ResidualResult81058

namespace ResidualResult81069
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 81069
end ResidualResult81069

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert
