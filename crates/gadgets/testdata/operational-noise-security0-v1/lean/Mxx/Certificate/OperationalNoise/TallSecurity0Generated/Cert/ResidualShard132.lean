import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard131

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace ResidualResult16636
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 16636
end ResidualResult16636

namespace ResidualResult16659
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 16659
end ResidualResult16659

namespace ResidualResult16663
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult16659.actual selector witness +
    ResidualResult16636.actual selector witness
end ResidualResult16663

namespace ResidualResult16667
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult16663.actual selector witness +
    ResidualResult16613.actual selector witness
end ResidualResult16667

namespace ResidualResult16671
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult16667.actual selector witness +
    ResidualResult16590.actual selector witness
end ResidualResult16671

namespace ResidualResult16675
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult16671.actual selector witness +
    ResidualResult16567.actual selector witness
end ResidualResult16675

namespace ResidualResult16679
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult16675.actual selector witness +
    ResidualResult16544.actual selector witness
end ResidualResult16679

namespace ResidualResult16683
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult16679.actual selector witness +
    ResidualResult16521.actual selector witness
end ResidualResult16683

namespace ResidualResult16687
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult16683.actual selector witness +
    ResidualResult16498.actual selector witness
end ResidualResult16687

namespace ResidualResult16691
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult16687.actual selector witness +
    ResidualResult16475.actual selector witness
end ResidualResult16691

namespace ResidualResult16695
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult16691.actual selector witness +
    ResidualResult16452.actual selector witness
end ResidualResult16695

namespace ResidualResult16699
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult16695.actual selector witness +
    ResidualResult16429.actual selector witness
end ResidualResult16699

namespace ResidualResult16703
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult16699.actual selector witness +
    ResidualResult16406.actual selector witness
end ResidualResult16703

namespace ResidualResult16707
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult16703.actual selector witness +
    ResidualResult16383.actual selector witness
end ResidualResult16707

namespace ResidualResult16711
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult16707.actual selector witness +
    ResidualResult16360.actual selector witness
end ResidualResult16711

namespace ResidualResult16715
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult16711.actual selector witness +
    ResidualResult16337.actual selector witness
end ResidualResult16715

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert
