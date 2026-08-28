import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard006
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard049
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard050
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard161
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard465
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard466
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard557
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard559
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard560
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard561
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard562

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace ResidualResult79374
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 79374
end ResidualResult79374

namespace ResidualResult79378
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult79374.actual selector witness -
    ResidualResult79371.actual selector witness
end ResidualResult79378

namespace ResidualResult79382
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult79378.actual selector witness -
    ResidualResult79363.actual selector witness
end ResidualResult79382

namespace ResidualResult79391
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult65387.actual selector witness *
    ResidualResult79220.actual selector witness
end ResidualResult79391

namespace ResidualResult79398
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult79391.actual selector witness +
    ResidualResult79213.actual selector witness
end ResidualResult79398

namespace ResidualResult79408
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult79398.actual selector witness *
    ResidualResult5859.actual selector witness
end ResidualResult79408

namespace ResidualResult79413
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult723.actual selector witness *
    ResidualResult65295.actual selector witness
end ResidualResult79413

namespace ResidualResult79418
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult65165.actual selector witness *
    ResidualResult5873.actual selector witness
end ResidualResult79418

namespace ResidualResult79422
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult79418.actual selector witness -
    ResidualResult79413.actual selector witness
end ResidualResult79422

namespace ResidualResult79428
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult79422.actual selector witness +
    ResidualResult20908.actual selector witness
end ResidualResult79428

namespace ResidualResult79435
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult79428.actual selector witness -
    ResidualResult79428.actual selector witness
end ResidualResult79435

namespace ResidualResult79440
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult79435.actual selector witness +
    ResidualResult79408.actual selector witness
end ResidualResult79440

namespace ResidualResult79445
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult79440.actual selector witness +
    ResidualResult79196.actual selector witness
end ResidualResult79445

namespace ResidualResult79450
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult79445.actual selector witness +
    ResidualResult78984.actual selector witness
end ResidualResult79450

namespace ResidualResult79455
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult79450.actual selector witness +
    ResidualResult78772.actual selector witness
end ResidualResult79455

namespace ResidualResult79460
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult79455.actual selector witness +
    ResidualResult78560.actual selector witness
end ResidualResult79460

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert
