import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard014
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard264
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard265
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard285

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace ResidualResult38878
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult38873.actual selector witness *
    ResidualResult38871.actual selector witness
end ResidualResult38878

namespace ResidualResult38881
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 38881
end ResidualResult38881

namespace ResidualResult38885
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult38881.actual selector witness -
    ResidualResult38878.actual selector witness
end ResidualResult38885

namespace ResidualResult38893
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult38885.actual selector witness *
    ResidualResult38862.actual selector witness
end ResidualResult38893

namespace ResidualResult38896
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 38896
end ResidualResult38896

namespace ResidualResult38901
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult38873.actual selector witness *
    ResidualResult38896.actual selector witness
end ResidualResult38901

namespace ResidualResult38904
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 38904
end ResidualResult38904

namespace ResidualResult38908
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult38904.actual selector witness -
    ResidualResult38901.actual selector witness
end ResidualResult38908

namespace ResidualResult38912
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult38908.actual selector witness -
    ResidualResult38893.actual selector witness
end ResidualResult38912

namespace ResidualResult38921
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult36137.actual selector witness *
    ResidualResult38750.actual selector witness
end ResidualResult38921

namespace ResidualResult38928
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult38921.actual selector witness +
    ResidualResult38743.actual selector witness
end ResidualResult38928

namespace ResidualResult38935
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 38935
end ResidualResult38935

namespace ResidualResult38938
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 38938
end ResidualResult38938

namespace ResidualResult38945
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 38945
end ResidualResult38945

namespace ResidualResult38948
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 38948
end ResidualResult38948

namespace ResidualResult38953
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult1728.actual selector witness *
    ResidualResult36045.actual selector witness
end ResidualResult38953

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert
