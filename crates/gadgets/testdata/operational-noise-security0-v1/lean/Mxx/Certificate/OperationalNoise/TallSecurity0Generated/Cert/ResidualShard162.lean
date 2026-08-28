import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard000
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard051
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard056
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard136
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard137
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard139
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard140
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard141
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard143
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard144
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard145
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard147
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard148
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard161

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace ResidualResult20985
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult20980.actual selector witness +
    ResidualResult18997.actual selector witness
end ResidualResult20985

namespace ResidualResult20990
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult20985.actual selector witness +
    ResidualResult18785.actual selector witness
end ResidualResult20990

namespace ResidualResult20995
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult20990.actual selector witness +
    ResidualResult18573.actual selector witness
end ResidualResult20995

namespace ResidualResult21000
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult20995.actual selector witness +
    ResidualResult18361.actual selector witness
end ResidualResult21000

namespace ResidualResult21005
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult21000.actual selector witness +
    ResidualResult18149.actual selector witness
end ResidualResult21005

namespace ResidualResult21010
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult21005.actual selector witness +
    ResidualResult17937.actual selector witness
end ResidualResult21010

namespace ResidualResult21015
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult21010.actual selector witness +
    ResidualResult17725.actual selector witness
end ResidualResult21015

namespace ResidualResult21020
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult21015.actual selector witness +
    ResidualResult17513.actual selector witness
end ResidualResult21020

namespace ResidualResult21025
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult21020.actual selector witness +
    ResidualResult17301.actual selector witness
end ResidualResult21025

namespace ResidualResult21030
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult21025.actual selector witness -
    ResidualResult17089.actual selector witness
end ResidualResult21030

namespace ResidualResult21032
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 21032
end ResidualResult21032

namespace ResidualResult21037
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult27.actual selector witness *
    ResidualResult5964.actual selector witness
end ResidualResult21037

namespace ResidualResult21041
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult21037.actual selector witness -
    ResidualResult6449.actual selector witness
end ResidualResult21041

namespace ResidualResult21047
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult21041.actual selector witness +
    ResidualResult21032.actual selector witness
end ResidualResult21047

namespace ResidualResult21075
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult21047.actual selector witness *
    ResidualResult5961.actual selector witness
end ResidualResult21075

namespace ResidualResult21099
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult21075.actual selector witness +
    ResidualResult21030.actual selector witness
end ResidualResult21099

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert
