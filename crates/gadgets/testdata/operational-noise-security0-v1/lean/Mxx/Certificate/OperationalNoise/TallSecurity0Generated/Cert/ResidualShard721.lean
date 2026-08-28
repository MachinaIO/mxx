import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard000
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard039
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard117
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard667
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard720

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace ResidualResult100828
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult100824.actual selector witness -
    ResidualResult100821.actual selector witness
end ResidualResult100828

namespace ResidualResult100836
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult100828.actual selector witness *
    ResidualResult100805.actual selector witness
end ResidualResult100836

namespace ResidualResult100839
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 100839
end ResidualResult100839

namespace ResidualResult100844
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult100816.actual selector witness *
    ResidualResult100839.actual selector witness
end ResidualResult100844

namespace ResidualResult100847
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 100847
end ResidualResult100847

namespace ResidualResult100851
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult100847.actual selector witness -
    ResidualResult100844.actual selector witness
end ResidualResult100851

namespace ResidualResult100855
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult100851.actual selector witness -
    ResidualResult100836.actual selector witness
end ResidualResult100855

namespace ResidualResult100864
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult94462.actual selector witness *
    ResidualResult100717.actual selector witness
end ResidualResult100864

namespace ResidualResult100871
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult100864.actual selector witness +
    ResidualResult100710.actual selector witness
end ResidualResult100871

namespace ResidualResult100878
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 100878
end ResidualResult100878

namespace ResidualResult100881
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 100881
end ResidualResult100881

namespace ResidualResult100888
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 100888
end ResidualResult100888

namespace ResidualResult100891
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 100891
end ResidualResult100891

namespace ResidualResult100896
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult4911.actual selector witness *
    ResidualResult32.actual selector witness
end ResidualResult100896

namespace ResidualResult100901
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult27.actual selector witness *
    ResidualResult13987.actual selector witness
end ResidualResult100901

namespace ResidualResult100905
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult100901.actual selector witness -
    ResidualResult100896.actual selector witness
end ResidualResult100905

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert
