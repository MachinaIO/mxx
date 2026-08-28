import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard045
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard046
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard047
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard048
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard049

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace ResidualResult5873
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 5873
end ResidualResult5873

namespace ResidualResult5878
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult5873.actual selector witness -
    ResidualResult5873.actual selector witness
end ResidualResult5878

namespace ResidualResult5882
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult5878.actual selector witness +
    ResidualResult5867.actual selector witness
end ResidualResult5882

namespace ResidualResult5886
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult5882.actual selector witness +
    ResidualResult5847.actual selector witness
end ResidualResult5886

namespace ResidualResult5890
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult5886.actual selector witness +
    ResidualResult5827.actual selector witness
end ResidualResult5890

namespace ResidualResult5894
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult5890.actual selector witness +
    ResidualResult5807.actual selector witness
end ResidualResult5894

namespace ResidualResult5898
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult5894.actual selector witness +
    ResidualResult5787.actual selector witness
end ResidualResult5898

namespace ResidualResult5902
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult5898.actual selector witness +
    ResidualResult5767.actual selector witness
end ResidualResult5902

namespace ResidualResult5906
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult5902.actual selector witness +
    ResidualResult5747.actual selector witness
end ResidualResult5906

namespace ResidualResult5910
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult5906.actual selector witness +
    ResidualResult5727.actual selector witness
end ResidualResult5910

namespace ResidualResult5914
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult5910.actual selector witness +
    ResidualResult5707.actual selector witness
end ResidualResult5914

namespace ResidualResult5918
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult5914.actual selector witness +
    ResidualResult5687.actual selector witness
end ResidualResult5918

namespace ResidualResult5922
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult5918.actual selector witness +
    ResidualResult5667.actual selector witness
end ResidualResult5922

namespace ResidualResult5926
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult5922.actual selector witness +
    ResidualResult5647.actual selector witness
end ResidualResult5926

namespace ResidualResult5930
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult5926.actual selector witness +
    ResidualResult5627.actual selector witness
end ResidualResult5930

namespace ResidualResult5934
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult5930.actual selector witness +
    ResidualResult5607.actual selector witness
end ResidualResult5934

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert
