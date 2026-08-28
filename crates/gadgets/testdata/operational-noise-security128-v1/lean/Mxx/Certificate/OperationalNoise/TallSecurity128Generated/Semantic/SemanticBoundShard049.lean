import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard009
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard012
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard014
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard017
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard019
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard022
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard024
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard026
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard029
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard031
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard034
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard036
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard039
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard041
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard044
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard046
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard048

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound15030
def owner : Owner := ⟨.program ⟨257⟩, ⟨67346⟩⟩
def transferEvent : Nat := 15030
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 15028 .coefficient, .predecessor 1 15029 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 15028 .coefficient)
      LeftBound15026.bound (LeftBound15026.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events058.exact15027RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15026.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15026.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 15029 .coefficient)
      LeftBound14265.bound (LeftBound14265.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events055.exact14285RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14265.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14265.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound15026.bound, LeftBound14265.bound]
def bound : CoeffClass := .finite ⟨542996768254952255020851616852483020531835568544753660967213398454193031574323094960458778515416684044481311503083260709592248761543369027792001023704585528406437625778210199105819490027153207492610, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound15026.bound, LeftBound14265.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound15026.actual selector witness, LeftBound14265.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound15030

namespace LeftBound15034
def owner : Owner := ⟨.program ⟨257⟩, ⟨67347⟩⟩
def transferEvent : Nat := 15034
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 15032 .coefficient, .predecessor 1 15033 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 15032 .coefficient)
      LeftBound15030.bound (LeftBound15030.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events058.exact15031RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15030.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15030.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 15033 .coefficient)
      LeftBound13523.bound (LeftBound13523.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events052.exact13543RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13523.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13523.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound15030.bound, LeftBound13523.bound]
def bound : CoeffClass := .finite ⟨672051163655048040366784807771125549463785661682259136508761121041087654171544178995973686472904601175218589475292148683874637201218983362140214212875221140987861628042887277578433992331977720299522, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound15030.bound, LeftBound13523.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound15030.actual selector witness, LeftBound13523.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound15034

namespace LeftBound15038
def owner : Owner := ⟨.program ⟨257⟩, ⟨67368⟩⟩
def transferEvent : Nat := 15038
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 15036 .coefficient, .predecessor 1 15037 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 15036 .coefficient)
      LeftBound15034.bound (LeftBound15034.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events058.exact15035RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15034.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15034.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 15037 .coefficient)
      LeftBound12775.bound (LeftBound12775.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events049.exact12795RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12775.bound, RecordedBoundRefines] <;> decide)
      (LeftBound12775.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound15034.bound, LeftBound12775.bound]
def bound : CoeffClass := .finite ⟨1009340526455529770115193168811409414647223797085836577678740446060854967759939516809695636530864772923790524128840001741433231032866468496852258397354412763951513341801266967929874168097887345573890, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound15034.bound, LeftBound12775.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound15034.actual selector witness, LeftBound12775.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound15038

namespace LeftBound15042
def owner : Owner := ⟨.program ⟨257⟩, ⟨67422⟩⟩
def transferEvent : Nat := 15042
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 15040 .coefficient, .predecessor 1 15041 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 15040 .coefficient)
      LeftBound15038.bound (LeftBound15038.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events058.exact15039RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15038.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15038.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 15041 .coefficient)
      LeftBound12027.bound (LeftBound12027.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events047.exact12047RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12027.bound, RecordedBoundRefines] <;> decide)
      (LeftBound12027.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound15038.bound, LeftBound12027.bound]
def bound : CoeffClass := .finite ⟨1079819613360031692290171131083543967257142223715911994549405513653755774395529374827883839331049581689656537143507366842527675697641084942726005258570374396267066556647993621900327137154558071799810, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound15038.bound, LeftBound12027.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound15038.actual selector witness, LeftBound12027.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound15042

namespace LeftBound15046
def owner : Owner := ⟨.program ⟨257⟩, ⟨67442⟩⟩
def transferEvent : Nat := 15046
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 15044 .coefficient, .predecessor 1 15045 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 15044 .coefficient)
      LeftBound15042.bound (LeftBound15042.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events058.exact15043RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15042.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15042.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 15045 .coefficient)
      LeftBound11279.bound (LeftBound11279.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events044.exact11299RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11279.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11279.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound15042.bound, LeftBound11279.bound]
def bound : CoeffClass := .finite ⟨1325684961847789477574708794703737870133863097145124865446189800320713994915088164677296682346086480286889034931935330118343710733230146578985490422063958612874508423786943165780995997288739414638594, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound15042.bound, LeftBound11279.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound15042.actual selector witness, LeftBound11279.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound15046

namespace LeftBound15050
def owner : Owner := ⟨.program ⟨257⟩, ⟨67462⟩⟩
def transferEvent : Nat := 15050
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 15048 .coefficient, .predecessor 1 15049 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 15048 .coefficient)
      LeftBound15046.bound (LeftBound15046.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events058.exact15047RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15046.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15046.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 15049 .coefficient)
      LeftBound10531.bound (LeftBound10531.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events041.exact10551RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10531.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10531.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound15046.bound, LeftBound10531.bound]
def bound : CoeffClass := .finite ⟨1420855841597677493273520414039300523635134860535819585694335006226637093514043024215732907198382451211963158231510879467883696393448368597512610766097274546188427166091959586902412865788582088441858, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound15046.bound, LeftBound10531.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound15046.actual selector witness, LeftBound10531.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound15050

namespace LeftBound15054
def owner : Owner := ⟨.program ⟨257⟩, ⟨67499⟩⟩
def transferEvent : Nat := 15054
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 15052 .coefficient, .predecessor 1 15053 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 15052 .coefficient)
      LeftBound15050.bound (LeftBound15050.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events058.exact15051RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15050.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15050.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 15053 .coefficient)
      LeftBound9783.bound (LeftBound9783.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events038.exact9803RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9783.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9783.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound15050.bound, LeftBound9783.bound]
def bound : CoeffClass := .finite ⟨1672986196047277504415903645253145265975473571869851273452016558508992584912513759845307592913166009679183152955058229658770947327979905647155015946143581415136103410341048531871556999959292127969282, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound15050.bound, LeftBound9783.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound15050.actual selector witness, LeftBound9783.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound15054

namespace LeftBound15058
def owner : Owner := ⟨.program ⟨257⟩, ⟨67519⟩⟩
def transferEvent : Nat := 15058
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 15056 .coefficient, .predecessor 1 15057 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 15056 .coefficient)
      LeftBound15054.bound (LeftBound15054.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events058.exact15055RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15054.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15054.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 15057 .coefficient)
      LeftBound9035.bound (LeftBound9035.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events035.exact9055RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9035.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9035.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound15054.bound, LeftBound9035.bound]
def bound : CoeffClass := .finite ⟨1777104291035577475984992214239650504155650380041269651391577294487908072874963080883826886728781055530876982055212121000122687581115049153559655783956871732501031747454204709752802552171414715957250, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound15054.bound, LeftBound9035.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound15054.actual selector witness, LeftBound9035.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound15058

namespace LeftBound15062
def owner : Owner := ⟨.program ⟨257⟩, ⟨67543⟩⟩
def transferEvent : Nat := 15062
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 15060 .coefficient, .predecessor 1 15061 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 15060 .coefficient)
      LeftBound15058.bound (LeftBound15058.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events058.exact15059RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15058.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15058.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 15061 .coefficient)
      LeftBound8287.bound (LeftBound8287.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events032.exact8307RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8287.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8287.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound15058.bound, LeftBound8287.bound]
def bound : CoeffClass := .finite ⟨1832732058535653329923076036421640366475622284292410768404193077949087223060441478934442323864945716306956322453590240543876465263420259873052772368389124606186609812402569350307654360864199731970050, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound15058.bound, LeftBound8287.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound15058.actual selector witness, LeftBound8287.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound15062

namespace LeftBound15066
def owner : Owner := ⟨.program ⟨257⟩, ⟨67544⟩⟩
def transferEvent : Nat := 15066
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 15064 .coefficient, .predecessor 1 15065 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 15064 .coefficient)
      LeftBound15062.bound (LeftBound15062.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events058.exact15063RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15062.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15062.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 15065 .coefficient)
      LeftBound7539.bound (LeftBound7539.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events029.exact7559RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound7539.bound, RecordedBoundRefines] <;> decide)
      (LeftBound7539.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound15062.bound, LeftBound7539.bound]
def bound : CoeffClass := .finite ⟨1922541680964796388146454601336603918969375305780471537618290953097010256281796475150018008120085296178564156175287229271934650004832869838809778249457199670432313164060584206878303285656535440982018, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound15062.bound, LeftBound7539.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound15062.actual selector witness, LeftBound7539.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound15066

namespace LeftBound15070
def owner : Owner := ⟨.program ⟨257⟩, ⟨67545⟩⟩
def transferEvent : Nat := 15070
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 15068 .coefficient, .predecessor 1 15069 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 15068 .coefficient)
      LeftBound15066.bound (LeftBound15066.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events058.exact15067RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15066.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15066.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 15069 .coefficient)
      LeftBound6791.bound (LeftBound6791.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events026.exact6811RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6791.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6791.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound15066.bound, LeftBound6791.bound]
def bound : CoeffClass := .finite ⟨2164980946628210869608262099449638246778267437374965761896878484276103520662752490914784404053661154834457178660865621927125102158372141617799099742141813904148270285023666057419776133840033716633602, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound15066.bound, LeftBound6791.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound15066.actual selector witness, LeftBound6791.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound15070

namespace LeftBound15074
def owner : Owner := ⟨.program ⟨257⟩, ⟨67546⟩⟩
def transferEvent : Nat := 15074
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 15072 .coefficient, .predecessor 1 15073 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 15072 .coefficient)
      LeftBound15070.bound (LeftBound15070.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events058.exact15071RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15070.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15070.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 15073 .coefficient)
      LeftBound6043.bound (LeftBound6043.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events023.exact6063RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6043.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6043.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound15070.bound, LeftBound6043.bound]
def bound : CoeffClass := .finite ⟨2205836211728078693439313735396398006575771274702137645468335288434297318322923080969875978068874632537684938027691199939457790363105904747873161888285988176998507008067079513735676819889025034715138, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound15070.bound, LeftBound6043.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound15070.actual selector witness, LeftBound6043.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound15074

namespace LeftBound15078
def owner : Owner := ⟨.program ⟨257⟩, ⟨67547⟩⟩
def transferEvent : Nat := 15078
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 15076 .coefficient, .predecessor 1 15077 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 15076 .coefficient)
      LeftBound15074.bound (LeftBound15074.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events058.exact15075RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15074.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15074.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 15077 .coefficient)
      LeftBound5295.bound (LeftBound5295.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events020.exact5315RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5295.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5295.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound15074.bound, LeftBound5295.bound]
def bound : CoeffClass := .finite ⟨2251396666591234913773189408255292390831419282760066217599244081746853741033999420182258357824265347472516131093190483942800926089554221834355935596654156562821733627702192817638959874190934013050882, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound15074.bound, LeftBound5295.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound15074.actual selector witness, LeftBound5295.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound15078

namespace LeftBound15082
def owner : Owner := ⟨.program ⟨257⟩, ⟨67571⟩⟩
def transferEvent : Nat := 15082
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 15080 .coefficient, .predecessor 1 15081 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 15080 .coefficient)
      LeftBound15078.bound (LeftBound15078.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events058.exact15079RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15078.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15078.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 15081 .coefficient)
      LeftBound4547.bound (LeftBound4547.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events017.exact4567RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound4547.bound, RecordedBoundRefines] <;> decide)
      (LeftBound4547.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound15078.bound, LeftBound4547.bound]
def bound : CoeffClass := .finite ⟨2474926557939653212570695213508739993033097895487676114072865025011765011850916335034021877569031622520275201323303567957541729077687280872741127429341747363277178902569497136566448970823578661289986, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound15078.bound, LeftBound4547.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound15078.actual selector witness, LeftBound4547.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound15082

namespace LeftBound15086
def owner : Owner := ⟨.program ⟨257⟩, ⟨67591⟩⟩
def transferEvent : Nat := 15086
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 15084 .coefficient, .predecessor 1 15085 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 15084 .coefficient)
      LeftBound15082.bound (LeftBound15082.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events058.exact15083RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15082.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15082.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 15085 .coefficient)
      LeftBound3799.bound (LeftBound3799.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events014.exact3819RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound3799.bound, RecordedBoundRefines] <;> decide)
      (LeftBound3799.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound15082.bound, LeftBound3799.bound]
def bound : CoeffClass := .finite ⟨2763363672149895599070144338931460826934561460724899158174325180219463274366259749111835974812292816237189443143534341441304364691790507241781313147496677637232224419076158662118069594571613028810754, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound15082.bound, LeftBound3799.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound15082.actual selector witness, LeftBound3799.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound15086

namespace LeftBound15090
def owner : Owner := ⟨.program ⟨257⟩, ⟨67611⟩⟩
def transferEvent : Nat := 15090
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 15088 .coefficient, .predecessor 1 15089 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 15088 .coefficient)
      LeftBound15086.bound (LeftBound15086.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events058.exact15087RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15086.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15086.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 15089 .coefficient)
      LeftBound3051.bound (LeftBound3051.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events011.exact3071RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound3051.bound, RecordedBoundRefines] <;> decide)
      (LeftBound3051.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound15086.bound, LeftBound3051.bound]
def bound : CoeffClass := .finite ⟨2776143336333402191585603384732445063544877149822978657382464754357584615014465340192800653194311466380060288905681282245991871384451325977333181056757177593705271138062888880087473554300375197417474, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound15086.bound, LeftBound3051.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound15086.actual selector witness, LeftBound3051.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound15090

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
