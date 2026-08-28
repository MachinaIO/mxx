import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard000
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard039
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard117
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard118
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard721

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace ResidualResult100911
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult100905.actual selector witness +
    ResidualResult13979.actual selector witness
end ResidualResult100911

namespace ResidualResult100919
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult100911.actual selector witness *
    ResidualResult4914.actual selector witness
end ResidualResult100919

namespace ResidualResult100924
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult4914.actual selector witness *
    ResidualResult32.actual selector witness
end ResidualResult100924

namespace ResidualResult100929
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult27.actual selector witness *
    ResidualResult14028.actual selector witness
end ResidualResult100929

namespace ResidualResult100933
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult100929.actual selector witness -
    ResidualResult100924.actual selector witness
end ResidualResult100933

namespace ResidualResult100939
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult100933.actual selector witness +
    ResidualResult14020.actual selector witness
end ResidualResult100939

namespace ResidualResult100949
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult100939.actual selector witness *
    ResidualResult14017.actual selector witness
end ResidualResult100949

namespace ResidualResult100955
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult100949.actual selector witness +
    ResidualResult100919.actual selector witness
end ResidualResult100955

namespace ResidualResult100965
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult100955.actual selector witness *
    ResidualResult100891.actual selector witness
end ResidualResult100965

namespace ResidualResult100968
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 100968
end ResidualResult100968

namespace ResidualResult100972
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 100972
end ResidualResult100972

namespace ResidualResult101026
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 101026
end ResidualResult101026

namespace ResidualResult101029
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 101029
end ResidualResult101029

namespace ResidualResult101034
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult101029.actual selector witness *
    ResidualResult101026.actual selector witness
end ResidualResult101034

namespace ResidualResult101045
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 101045
end ResidualResult101045

namespace ResidualResult101048
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 101048
end ResidualResult101048

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert
