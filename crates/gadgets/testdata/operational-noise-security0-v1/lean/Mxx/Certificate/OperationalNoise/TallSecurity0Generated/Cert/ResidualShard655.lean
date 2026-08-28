import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard047
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard567
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard612
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard654

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace ResidualResult92668
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 92668
end ResidualResult92668

namespace ResidualResult92672
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult92668.actual selector witness -
    ResidualResult92665.actual selector witness
end ResidualResult92672

namespace ResidualResult92680
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult92672.actual selector witness *
    ResidualResult92649.actual selector witness
end ResidualResult92680

namespace ResidualResult92683
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 92683
end ResidualResult92683

namespace ResidualResult92688
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult92660.actual selector witness *
    ResidualResult92683.actual selector witness
end ResidualResult92688

namespace ResidualResult92691
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 92691
end ResidualResult92691

namespace ResidualResult92695
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult92691.actual selector witness -
    ResidualResult92688.actual selector witness
end ResidualResult92695

namespace ResidualResult92699
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult92695.actual selector witness -
    ResidualResult92680.actual selector witness
end ResidualResult92699

namespace ResidualResult92708
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult80012.actual selector witness *
    ResidualResult92537.actual selector witness
end ResidualResult92708

namespace ResidualResult92715
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult92708.actual selector witness +
    ResidualResult92530.actual selector witness
end ResidualResult92715

namespace ResidualResult92725
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult92715.actual selector witness *
    ResidualResult5739.actual selector witness
end ResidualResult92725

namespace ResidualResult92729
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 92729
end ResidualResult92729

namespace ResidualResult92732
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 92732
end ResidualResult92732

namespace ResidualResult92742
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult85956.actual selector witness *
    ResidualResult92732.actual selector witness
end ResidualResult92742

namespace ResidualResult92745
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 92745
end ResidualResult92745

namespace ResidualResult92749
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 92749
end ResidualResult92749

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert
