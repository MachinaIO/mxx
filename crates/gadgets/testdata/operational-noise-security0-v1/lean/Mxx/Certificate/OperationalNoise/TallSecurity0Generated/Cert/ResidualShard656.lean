import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard567
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard655

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace ResidualResult92847
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 92847
end ResidualResult92847

namespace ResidualResult92858
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 92858
end ResidualResult92858

namespace ResidualResult92861
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 92861
end ResidualResult92861

namespace ResidualResult92870
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 92870
end ResidualResult92870

namespace ResidualResult92872
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 92872
end ResidualResult92872

namespace ResidualResult92877
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult92872.actual selector witness *
    ResidualResult92870.actual selector witness
end ResidualResult92877

namespace ResidualResult92880
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 92880
end ResidualResult92880

namespace ResidualResult92884
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult92880.actual selector witness -
    ResidualResult92877.actual selector witness
end ResidualResult92884

namespace ResidualResult92892
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult92884.actual selector witness *
    ResidualResult92861.actual selector witness
end ResidualResult92892

namespace ResidualResult92895
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 92895
end ResidualResult92895

namespace ResidualResult92900
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult92872.actual selector witness *
    ResidualResult92895.actual selector witness
end ResidualResult92900

namespace ResidualResult92903
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 92903
end ResidualResult92903

namespace ResidualResult92907
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult92903.actual selector witness -
    ResidualResult92900.actual selector witness
end ResidualResult92907

namespace ResidualResult92911
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult92907.actual selector witness -
    ResidualResult92892.actual selector witness
end ResidualResult92911

namespace ResidualResult92920
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult80012.actual selector witness *
    ResidualResult92749.actual selector witness
end ResidualResult92920

namespace ResidualResult92927
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult92920.actual selector witness +
    ResidualResult92742.actual selector witness
end ResidualResult92927

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert
