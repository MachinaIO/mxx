import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard058
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard124

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace ResidualResult14908
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult14903.actual selector witness *
    ResidualResult14901.actual selector witness
end ResidualResult14908

namespace ResidualResult14911
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 14911
end ResidualResult14911

namespace ResidualResult14915
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult14911.actual selector witness -
    ResidualResult14908.actual selector witness
end ResidualResult14915

namespace ResidualResult14923
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult14915.actual selector witness *
    ResidualResult14892.actual selector witness
end ResidualResult14923

namespace ResidualResult14926
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 14926
end ResidualResult14926

namespace ResidualResult14931
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult14903.actual selector witness *
    ResidualResult14926.actual selector witness
end ResidualResult14931

namespace ResidualResult14934
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 14934
end ResidualResult14934

namespace ResidualResult14938
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult14934.actual selector witness -
    ResidualResult14931.actual selector witness
end ResidualResult14938

namespace ResidualResult14942
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult14938.actual selector witness -
    ResidualResult14923.actual selector witness
end ResidualResult14942

namespace ResidualResult14951
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult6561.actual selector witness *
    ResidualResult14780.actual selector witness
end ResidualResult14951

namespace ResidualResult14958
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult14951.actual selector witness +
    ResidualResult14773.actual selector witness
end ResidualResult14958

namespace ResidualResult14965
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 14965
end ResidualResult14965

namespace ResidualResult14968
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 14968
end ResidualResult14968

namespace ResidualResult14975
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 14975
end ResidualResult14975

namespace ResidualResult14978
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 14978
end ResidualResult14978

namespace ResidualResult14981
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 14981
end ResidualResult14981

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert
