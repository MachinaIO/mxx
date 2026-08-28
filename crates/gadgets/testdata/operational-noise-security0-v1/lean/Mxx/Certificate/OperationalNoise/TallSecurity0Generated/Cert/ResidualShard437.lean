import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard366
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard432
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard435
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard436

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace ResidualResult60750
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult60746.actual selector witness +
    ResidualResult60673.actual selector witness
end ResidualResult60750

namespace ResidualResult60754
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult60750.actual selector witness +
    ResidualResult60670.actual selector witness
end ResidualResult60754

namespace ResidualResult60758
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult60754.actual selector witness +
    ResidualResult60667.actual selector witness
end ResidualResult60758

namespace ResidualResult60762
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult60758.actual selector witness +
    ResidualResult60664.actual selector witness
end ResidualResult60762

namespace ResidualResult60766
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult60762.actual selector witness +
    ResidualResult60661.actual selector witness
end ResidualResult60766

namespace ResidualResult60770
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult60766.actual selector witness +
    ResidualResult60658.actual selector witness
end ResidualResult60770

namespace ResidualResult60774
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult60770.actual selector witness +
    ResidualResult60655.actual selector witness
end ResidualResult60774

namespace ResidualResult60778
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult60774.actual selector witness -
    ResidualResult60652.actual selector witness
end ResidualResult60778

namespace ResidualResult60854
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult60778.actual selector witness *
    ResidualResult60619.actual selector witness
end ResidualResult60854

namespace ResidualResult60857
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 60857
end ResidualResult60857

namespace ResidualResult60862
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult60630.actual selector witness *
    ResidualResult60857.actual selector witness
end ResidualResult60862

namespace ResidualResult60865
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 60865
end ResidualResult60865

namespace ResidualResult60869
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult60865.actual selector witness -
    ResidualResult60862.actual selector witness
end ResidualResult60869

namespace ResidualResult60873
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult60869.actual selector witness -
    ResidualResult60854.actual selector witness
end ResidualResult60873

namespace ResidualResult60916
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult50762.actual selector witness *
    ResidualResult59507.actual selector witness
end ResidualResult60916

namespace ResidualResult60957
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult60916.actual selector witness +
    ResidualResult59500.actual selector witness
end ResidualResult60957

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert
