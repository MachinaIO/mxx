import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard366
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard426

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace ResidualResult58798
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 58798
end ResidualResult58798

namespace ResidualResult58800
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 58800
end ResidualResult58800

namespace ResidualResult58805
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult58800.actual selector witness *
    ResidualResult58798.actual selector witness
end ResidualResult58805

namespace ResidualResult58808
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 58808
end ResidualResult58808

namespace ResidualResult58812
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult58808.actual selector witness -
    ResidualResult58805.actual selector witness
end ResidualResult58812

namespace ResidualResult58820
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult58812.actual selector witness *
    ResidualResult58789.actual selector witness
end ResidualResult58820

namespace ResidualResult58823
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 58823
end ResidualResult58823

namespace ResidualResult58828
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult58800.actual selector witness *
    ResidualResult58823.actual selector witness
end ResidualResult58828

namespace ResidualResult58831
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 58831
end ResidualResult58831

namespace ResidualResult58835
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult58831.actual selector witness -
    ResidualResult58828.actual selector witness
end ResidualResult58835

namespace ResidualResult58839
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult58835.actual selector witness -
    ResidualResult58820.actual selector witness
end ResidualResult58839

namespace ResidualResult58848
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult50762.actual selector witness *
    ResidualResult58677.actual selector witness
end ResidualResult58848

namespace ResidualResult58855
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult58848.actual selector witness +
    ResidualResult58670.actual selector witness
end ResidualResult58855

namespace ResidualResult58862
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 58862
end ResidualResult58862

namespace ResidualResult58865
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 58865
end ResidualResult58865

namespace ResidualResult58872
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 58872
end ResidualResult58872

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert
