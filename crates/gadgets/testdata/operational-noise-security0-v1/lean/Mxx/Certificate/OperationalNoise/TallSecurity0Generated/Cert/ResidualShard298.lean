import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard014
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard093
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard094
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard263
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard264
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard297

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace ResidualResult40399
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult1797.actual selector witness *
    ResidualResult36045.actual selector witness
end ResidualResult40399

namespace ResidualResult40404
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult35915.actual selector witness *
    ResidualResult10981.actual selector witness
end ResidualResult40404

namespace ResidualResult40408
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult40404.actual selector witness -
    ResidualResult40399.actual selector witness
end ResidualResult40408

namespace ResidualResult40414
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult40408.actual selector witness +
    ResidualResult10973.actual selector witness
end ResidualResult40414

namespace ResidualResult40422
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult40414.actual selector witness *
    ResidualResult1800.actual selector witness
end ResidualResult40422

namespace ResidualResult40427
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult1800.actual selector witness *
    ResidualResult36045.actual selector witness
end ResidualResult40427

namespace ResidualResult40432
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult35915.actual selector witness *
    ResidualResult11022.actual selector witness
end ResidualResult40432

namespace ResidualResult40436
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult40432.actual selector witness -
    ResidualResult40427.actual selector witness
end ResidualResult40436

namespace ResidualResult40442
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult40436.actual selector witness +
    ResidualResult11014.actual selector witness
end ResidualResult40442

namespace ResidualResult40452
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult40442.actual selector witness *
    ResidualResult11011.actual selector witness
end ResidualResult40452

namespace ResidualResult40458
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult40452.actual selector witness +
    ResidualResult40422.actual selector witness
end ResidualResult40458

namespace ResidualResult40468
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  ResidualResult40458.actual selector witness *
    ResidualResult40394.actual selector witness
end ResidualResult40468

namespace ResidualResult40471
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 40471
end ResidualResult40471

namespace ResidualResult40475
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 40475
end ResidualResult40475

namespace ResidualResult40553
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 40553
end ResidualResult40553

namespace ResidualResult40556
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  witness.honestTerminalActual 40556
end ResidualResult40556

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert
