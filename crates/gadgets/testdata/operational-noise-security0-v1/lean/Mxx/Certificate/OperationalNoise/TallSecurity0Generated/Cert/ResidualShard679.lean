import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard667
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard677
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard678

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace ResidualResult95867
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 95867
end ResidualResult95867

namespace ResidualResult95871
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 95871
end ResidualResult95871

namespace ResidualResult95874
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 95874
end ResidualResult95874

namespace ResidualResult95879
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult95874.actual selector witness *
    ResidualResult95871.actual selector witness
end ResidualResult95879

namespace ResidualResult95883
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult95879.actual selector witness -
    ResidualResult95856.actual selector witness
end ResidualResult95883

namespace ResidualResult95891
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult95883.actual selector witness *
    ResidualResult95840.actual selector witness
end ResidualResult95891

namespace ResidualResult95894
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 95894
end ResidualResult95894

namespace ResidualResult95899
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult95851.actual selector witness *
    ResidualResult95894.actual selector witness
end ResidualResult95899

namespace ResidualResult95902
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 95902
end ResidualResult95902

namespace ResidualResult95906
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult95902.actual selector witness -
    ResidualResult95899.actual selector witness
end ResidualResult95906

namespace ResidualResult95910
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult95906.actual selector witness -
    ResidualResult95891.actual selector witness
end ResidualResult95910

namespace ResidualResult95919
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult94462.actual selector witness *
    ResidualResult95764.actual selector witness
end ResidualResult95919

namespace ResidualResult95926
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult95919.actual selector witness +
    ResidualResult95757.actual selector witness
end ResidualResult95926

namespace ResidualResult95936
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult95926.actual selector witness *
    ResidualResult95673.actual selector witness
end ResidualResult95936

namespace ResidualResult95939
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 95939
end ResidualResult95939

namespace ResidualResult95943
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 95943
end ResidualResult95943

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert
