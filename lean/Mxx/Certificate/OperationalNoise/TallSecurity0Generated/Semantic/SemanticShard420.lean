import Mxx.Certificate.OperationalNoise.TallSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticShard420

open Mxx.Certificate.OperationalNoise
open TallSecurity0ABI
open TallSemantics

def shardIndex : Nat := 420
def shardStartEvent : Nat := 107520
def shardEndEvent : Nat := 107567
def rawSemanticCount : Nat := 3
def rawBoundTransferCount : Nat := 0
def rawResultCount : Nat := 1
def rawRelationCount : Nat := 0
def rawSurvivorFoldCount : Nat := 0
def rawPreFoldCount : Nat := 1
def rawInvocationEndCount : Nat := 1
def canonicalWork : Nat := 278

namespace Bound0
def selectedEvent : Nat := 107566
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨30220⟩⟩
def rootResultEvent : Nat := 107564
def prefoldEvent : Nat := 107565
def endEvent : Nat := 107566
def survivorEvents : List Nat := [6328, 6471, 6512, 6560, 6987, 7028, 7488, 7529, 7989, 8030, 8490, 8531, 8991, 9032, 9492, 9533, 9993, 10034, 10494, 10535, 10995, 11036, 11496, 11537, 11997, 12038, 12498, 12539, 12999, 13040, 13500, 13541, 14001, 14042, 14502, 14543, 15003, 15044, 20927, 21046, 21304, 21439, 21467, 21511, 21932, 21960, 22414, 22442, 22896, 22924, 23378, 23406, 23860, 23888, 24342, 24370, 24824, 24852, 25306, 25334, 25788, 25816, 26270, 26298, 26752, 26780, 27234, 27262, 27716, 27744, 28198, 28226, 28680, 28708, 29162, 29190, 29644, 29672, 35552, 35671, 35929, 36064, 36092, 36136, 36557, 36585, 37039, 37067, 37521, 37549, 38003, 38031, 38485, 38513, 38967, 38995, 39449, 39477, 39931, 39959, 40413, 40441, 40895, 40923, 41377, 41405, 41859, 41887, 42341, 42369, 42823, 42851, 43305, 43333, 43787, 43815, 44269, 44297, 50177, 50296, 50554, 50689, 50717, 50761, 51182, 51210, 51664, 51692, 52146, 52174, 52628, 52656, 53110, 53138, 53592, 53620, 54074, 54102, 54556, 54584, 55038, 55066, 55520, 55548, 56002, 56030, 56484, 56512, 56966, 56994, 57448, 57476, 57930, 57958, 58412, 58440, 58894, 58922, 64802, 64921, 65179, 65314, 65342, 65386, 65807, 65835, 66289, 66317, 66771, 66799, 67253, 67281, 67735, 67763, 68217, 68245, 68699, 68727, 69181, 69209, 69663, 69691, 70145, 70173, 70627, 70655, 71109, 71137, 71591, 71619, 72073, 72101, 72555, 72583, 73037, 73065, 73519, 73547, 79427, 79546, 79804, 79939, 79967, 80011, 80430, 80458, 80910, 80938, 81390, 81418, 81870, 81898, 82350, 82378, 82830, 82858, 83310, 83338, 83790, 83818, 84270, 84298, 84750, 84778, 85230, 85258, 85710, 85738, 86190, 86218, 86670, 86698, 87150, 87178, 87630, 87658, 88110, 88138, 94016, 94135, 94389, 94417, 94461, 94834, 94862, 95268, 95296, 95702, 95730, 96136, 96164, 96570, 96598, 97004, 97032, 97438, 97466, 97872, 97900, 98306, 98334, 98740, 98768, 99174, 99202, 99608, 99636, 100042, 100070, 100476, 100504, 100910, 100938, 101344, 101372, 101778, 101806, 107182]
def rootRaw : List Term := []
def prefoldRaw : List Term := []
def endRaw : List Term := []
def rootTerms : Polynomial Owner := []
def prefoldTerms : Polynomial Owner := []
def endTerms : Polynomial Owner := []
def rootSummary : Bound := (.finite 25317157507886064950797272225391822339692950454324)
def prefoldSummary : Bound := (.finite 25317157507886064950797272225391822339692950454324)
def endSummary : Bound := (.finite 25317157507886064950797272225391822339692950454324)
def rootBound : Nat := 25317157507886064950797272225391822339692950454324
def prefoldBound : Nat := 25317157507886064950797272225391822339692950454324
def survivorContributionsChunk0 : List Nat := [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
def survivorBoundsChunk0 : List Nat := [6327, 6470, 6511, 6559, 6986, 7027, 7487, 7528, 7988, 8029, 8489, 8530, 8990, 9031, 9491, 9532]
theorem survivorBoundsSoundChunk0 : List.Forall₂ (fun actual bound => actual ≤ bound) survivorContributionsChunk0 survivorBoundsChunk0 :=
by
  constructor
  · omega
  ·
    constructor
    · omega
    ·
      constructor
      · omega
      ·
        constructor
        · omega
        ·
          constructor
          · omega
          ·
            constructor
            · omega
            ·
              constructor
              · omega
              ·
                constructor
                · omega
                ·
                  constructor
                  · omega
                  ·
                    constructor
                    · omega
                    ·
                      constructor
                      · omega
                      ·
                        constructor
                        · omega
                        ·
                          constructor
                          · omega
                          ·
                            constructor
                            · omega
                            ·
                              constructor
                              · omega
                              ·
                                constructor
                                · omega
                                ·
                                  exact List.Forall₂.nil

def survivorContributionsChunk1 : List Nat := [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
def survivorBoundsChunk1 : List Nat := [9992, 10033, 10493, 10534, 10994, 11035, 11495, 11536, 11996, 12037, 12497, 12538, 12998, 13039, 13499, 13540]
theorem survivorBoundsSoundChunk1 : List.Forall₂ (fun actual bound => actual ≤ bound) survivorContributionsChunk1 survivorBoundsChunk1 :=
by
  constructor
  · omega
  ·
    constructor
    · omega
    ·
      constructor
      · omega
      ·
        constructor
        · omega
        ·
          constructor
          · omega
          ·
            constructor
            · omega
            ·
              constructor
              · omega
              ·
                constructor
                · omega
                ·
                  constructor
                  · omega
                  ·
                    constructor
                    · omega
                    ·
                      constructor
                      · omega
                      ·
                        constructor
                        · omega
                        ·
                          constructor
                          · omega
                          ·
                            constructor
                            · omega
                            ·
                              constructor
                              · omega
                              ·
                                constructor
                                · omega
                                ·
                                  exact List.Forall₂.nil

def survivorContributionsChunk2 : List Nat := [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
def survivorBoundsChunk2 : List Nat := [14000, 14041, 14501, 14542, 15002, 15043, 20926, 21045, 21303, 21438, 21466, 21510, 21931, 21959, 22413, 22441]
theorem survivorBoundsSoundChunk2 : List.Forall₂ (fun actual bound => actual ≤ bound) survivorContributionsChunk2 survivorBoundsChunk2 :=
by
  constructor
  · omega
  ·
    constructor
    · omega
    ·
      constructor
      · omega
      ·
        constructor
        · omega
        ·
          constructor
          · omega
          ·
            constructor
            · omega
            ·
              constructor
              · omega
              ·
                constructor
                · omega
                ·
                  constructor
                  · omega
                  ·
                    constructor
                    · omega
                    ·
                      constructor
                      · omega
                      ·
                        constructor
                        · omega
                        ·
                          constructor
                          · omega
                          ·
                            constructor
                            · omega
                            ·
                              constructor
                              · omega
                              ·
                                constructor
                                · omega
                                ·
                                  exact List.Forall₂.nil

def survivorContributionsChunk3 : List Nat := [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
def survivorBoundsChunk3 : List Nat := [22895, 22923, 23377, 23405, 23859, 23887, 24341, 24369, 24823, 24851, 25305, 25333, 25787, 25815, 26269, 26297]
theorem survivorBoundsSoundChunk3 : List.Forall₂ (fun actual bound => actual ≤ bound) survivorContributionsChunk3 survivorBoundsChunk3 :=
by
  constructor
  · omega
  ·
    constructor
    · omega
    ·
      constructor
      · omega
      ·
        constructor
        · omega
        ·
          constructor
          · omega
          ·
            constructor
            · omega
            ·
              constructor
              · omega
              ·
                constructor
                · omega
                ·
                  constructor
                  · omega
                  ·
                    constructor
                    · omega
                    ·
                      constructor
                      · omega
                      ·
                        constructor
                        · omega
                        ·
                          constructor
                          · omega
                          ·
                            constructor
                            · omega
                            ·
                              constructor
                              · omega
                              ·
                                constructor
                                · omega
                                ·
                                  exact List.Forall₂.nil

def survivorContributionsChunk4 : List Nat := [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
def survivorBoundsChunk4 : List Nat := [26751, 26779, 27233, 27261, 27715, 27743, 28197, 28225, 28679, 28707, 29161, 29189, 29643, 29671, 35551, 35670]
theorem survivorBoundsSoundChunk4 : List.Forall₂ (fun actual bound => actual ≤ bound) survivorContributionsChunk4 survivorBoundsChunk4 :=
by
  constructor
  · omega
  ·
    constructor
    · omega
    ·
      constructor
      · omega
      ·
        constructor
        · omega
        ·
          constructor
          · omega
          ·
            constructor
            · omega
            ·
              constructor
              · omega
              ·
                constructor
                · omega
                ·
                  constructor
                  · omega
                  ·
                    constructor
                    · omega
                    ·
                      constructor
                      · omega
                      ·
                        constructor
                        · omega
                        ·
                          constructor
                          · omega
                          ·
                            constructor
                            · omega
                            ·
                              constructor
                              · omega
                              ·
                                constructor
                                · omega
                                ·
                                  exact List.Forall₂.nil

def survivorContributionsChunk5 : List Nat := [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
def survivorBoundsChunk5 : List Nat := [35928, 36063, 36091, 36135, 36556, 36584, 37038, 37066, 37520, 37548, 38002, 38030, 38484, 38512, 38966, 38994]
theorem survivorBoundsSoundChunk5 : List.Forall₂ (fun actual bound => actual ≤ bound) survivorContributionsChunk5 survivorBoundsChunk5 :=
by
  constructor
  · omega
  ·
    constructor
    · omega
    ·
      constructor
      · omega
      ·
        constructor
        · omega
        ·
          constructor
          · omega
          ·
            constructor
            · omega
            ·
              constructor
              · omega
              ·
                constructor
                · omega
                ·
                  constructor
                  · omega
                  ·
                    constructor
                    · omega
                    ·
                      constructor
                      · omega
                      ·
                        constructor
                        · omega
                        ·
                          constructor
                          · omega
                          ·
                            constructor
                            · omega
                            ·
                              constructor
                              · omega
                              ·
                                constructor
                                · omega
                                ·
                                  exact List.Forall₂.nil

def survivorContributionsChunk6 : List Nat := [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
def survivorBoundsChunk6 : List Nat := [39448, 39476, 39930, 39958, 40412, 40440, 40894, 40922, 41376, 41404, 41858, 41886, 42340, 42368, 42822, 42850]
theorem survivorBoundsSoundChunk6 : List.Forall₂ (fun actual bound => actual ≤ bound) survivorContributionsChunk6 survivorBoundsChunk6 :=
by
  constructor
  · omega
  ·
    constructor
    · omega
    ·
      constructor
      · omega
      ·
        constructor
        · omega
        ·
          constructor
          · omega
          ·
            constructor
            · omega
            ·
              constructor
              · omega
              ·
                constructor
                · omega
                ·
                  constructor
                  · omega
                  ·
                    constructor
                    · omega
                    ·
                      constructor
                      · omega
                      ·
                        constructor
                        · omega
                        ·
                          constructor
                          · omega
                          ·
                            constructor
                            · omega
                            ·
                              constructor
                              · omega
                              ·
                                constructor
                                · omega
                                ·
                                  exact List.Forall₂.nil

def survivorContributionsChunk7 : List Nat := [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
def survivorBoundsChunk7 : List Nat := [43304, 43332, 43786, 43814, 44268, 44296, 50176, 50295, 50553, 50688, 50716, 50760, 51181, 51209, 51663, 51691]
theorem survivorBoundsSoundChunk7 : List.Forall₂ (fun actual bound => actual ≤ bound) survivorContributionsChunk7 survivorBoundsChunk7 :=
by
  constructor
  · omega
  ·
    constructor
    · omega
    ·
      constructor
      · omega
      ·
        constructor
        · omega
        ·
          constructor
          · omega
          ·
            constructor
            · omega
            ·
              constructor
              · omega
              ·
                constructor
                · omega
                ·
                  constructor
                  · omega
                  ·
                    constructor
                    · omega
                    ·
                      constructor
                      · omega
                      ·
                        constructor
                        · omega
                        ·
                          constructor
                          · omega
                          ·
                            constructor
                            · omega
                            ·
                              constructor
                              · omega
                              ·
                                constructor
                                · omega
                                ·
                                  exact List.Forall₂.nil

def survivorContributionsChunk8 : List Nat := [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
def survivorBoundsChunk8 : List Nat := [52145, 52173, 52627, 52655, 53109, 53137, 53591, 53619, 54073, 54101, 54555, 54583, 55037, 55065, 55519, 55547]
theorem survivorBoundsSoundChunk8 : List.Forall₂ (fun actual bound => actual ≤ bound) survivorContributionsChunk8 survivorBoundsChunk8 :=
by
  constructor
  · omega
  ·
    constructor
    · omega
    ·
      constructor
      · omega
      ·
        constructor
        · omega
        ·
          constructor
          · omega
          ·
            constructor
            · omega
            ·
              constructor
              · omega
              ·
                constructor
                · omega
                ·
                  constructor
                  · omega
                  ·
                    constructor
                    · omega
                    ·
                      constructor
                      · omega
                      ·
                        constructor
                        · omega
                        ·
                          constructor
                          · omega
                          ·
                            constructor
                            · omega
                            ·
                              constructor
                              · omega
                              ·
                                constructor
                                · omega
                                ·
                                  exact List.Forall₂.nil

def survivorContributionsChunk9 : List Nat := [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
def survivorBoundsChunk9 : List Nat := [56001, 56029, 56483, 56511, 56965, 56993, 57447, 57475, 57929, 57957, 58411, 58439, 58893, 58921, 64801, 64920]
theorem survivorBoundsSoundChunk9 : List.Forall₂ (fun actual bound => actual ≤ bound) survivorContributionsChunk9 survivorBoundsChunk9 :=
by
  constructor
  · omega
  ·
    constructor
    · omega
    ·
      constructor
      · omega
      ·
        constructor
        · omega
        ·
          constructor
          · omega
          ·
            constructor
            · omega
            ·
              constructor
              · omega
              ·
                constructor
                · omega
                ·
                  constructor
                  · omega
                  ·
                    constructor
                    · omega
                    ·
                      constructor
                      · omega
                      ·
                        constructor
                        · omega
                        ·
                          constructor
                          · omega
                          ·
                            constructor
                            · omega
                            ·
                              constructor
                              · omega
                              ·
                                constructor
                                · omega
                                ·
                                  exact List.Forall₂.nil

def survivorContributionsChunk10 : List Nat := [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
def survivorBoundsChunk10 : List Nat := [65178, 65313, 65341, 65385, 65806, 65834, 66288, 66316, 66770, 66798, 67252, 67280, 67734, 67762, 68216, 68244]
theorem survivorBoundsSoundChunk10 : List.Forall₂ (fun actual bound => actual ≤ bound) survivorContributionsChunk10 survivorBoundsChunk10 :=
by
  constructor
  · omega
  ·
    constructor
    · omega
    ·
      constructor
      · omega
      ·
        constructor
        · omega
        ·
          constructor
          · omega
          ·
            constructor
            · omega
            ·
              constructor
              · omega
              ·
                constructor
                · omega
                ·
                  constructor
                  · omega
                  ·
                    constructor
                    · omega
                    ·
                      constructor
                      · omega
                      ·
                        constructor
                        · omega
                        ·
                          constructor
                          · omega
                          ·
                            constructor
                            · omega
                            ·
                              constructor
                              · omega
                              ·
                                constructor
                                · omega
                                ·
                                  exact List.Forall₂.nil

def survivorContributionsChunk11 : List Nat := [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
def survivorBoundsChunk11 : List Nat := [68698, 68726, 69180, 69208, 69662, 69690, 70144, 70172, 70626, 70654, 71108, 71136, 71590, 71618, 72072, 72100]
theorem survivorBoundsSoundChunk11 : List.Forall₂ (fun actual bound => actual ≤ bound) survivorContributionsChunk11 survivorBoundsChunk11 :=
by
  constructor
  · omega
  ·
    constructor
    · omega
    ·
      constructor
      · omega
      ·
        constructor
        · omega
        ·
          constructor
          · omega
          ·
            constructor
            · omega
            ·
              constructor
              · omega
              ·
                constructor
                · omega
                ·
                  constructor
                  · omega
                  ·
                    constructor
                    · omega
                    ·
                      constructor
                      · omega
                      ·
                        constructor
                        · omega
                        ·
                          constructor
                          · omega
                          ·
                            constructor
                            · omega
                            ·
                              constructor
                              · omega
                              ·
                                constructor
                                · omega
                                ·
                                  exact List.Forall₂.nil

def survivorContributionsChunk12 : List Nat := [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
def survivorBoundsChunk12 : List Nat := [72554, 72582, 73036, 73064, 73518, 73546, 79426, 79545, 79803, 79938, 79966, 80010, 80429, 80457, 80909, 80937]
theorem survivorBoundsSoundChunk12 : List.Forall₂ (fun actual bound => actual ≤ bound) survivorContributionsChunk12 survivorBoundsChunk12 :=
by
  constructor
  · omega
  ·
    constructor
    · omega
    ·
      constructor
      · omega
      ·
        constructor
        · omega
        ·
          constructor
          · omega
          ·
            constructor
            · omega
            ·
              constructor
              · omega
              ·
                constructor
                · omega
                ·
                  constructor
                  · omega
                  ·
                    constructor
                    · omega
                    ·
                      constructor
                      · omega
                      ·
                        constructor
                        · omega
                        ·
                          constructor
                          · omega
                          ·
                            constructor
                            · omega
                            ·
                              constructor
                              · omega
                              ·
                                constructor
                                · omega
                                ·
                                  exact List.Forall₂.nil

def survivorContributionsChunk13 : List Nat := [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
def survivorBoundsChunk13 : List Nat := [81389, 81417, 81869, 81897, 82349, 82377, 82829, 82857, 83309, 83337, 83789, 83817, 84269, 84297, 84749, 84777]
theorem survivorBoundsSoundChunk13 : List.Forall₂ (fun actual bound => actual ≤ bound) survivorContributionsChunk13 survivorBoundsChunk13 :=
by
  constructor
  · omega
  ·
    constructor
    · omega
    ·
      constructor
      · omega
      ·
        constructor
        · omega
        ·
          constructor
          · omega
          ·
            constructor
            · omega
            ·
              constructor
              · omega
              ·
                constructor
                · omega
                ·
                  constructor
                  · omega
                  ·
                    constructor
                    · omega
                    ·
                      constructor
                      · omega
                      ·
                        constructor
                        · omega
                        ·
                          constructor
                          · omega
                          ·
                            constructor
                            · omega
                            ·
                              constructor
                              · omega
                              ·
                                constructor
                                · omega
                                ·
                                  exact List.Forall₂.nil

def survivorContributionsChunk14 : List Nat := [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
def survivorBoundsChunk14 : List Nat := [85229, 85257, 85709, 85737, 86189, 86217, 86669, 86697, 87149, 87177, 87629, 87657, 88109, 88137, 94015, 94134]
theorem survivorBoundsSoundChunk14 : List.Forall₂ (fun actual bound => actual ≤ bound) survivorContributionsChunk14 survivorBoundsChunk14 :=
by
  constructor
  · omega
  ·
    constructor
    · omega
    ·
      constructor
      · omega
      ·
        constructor
        · omega
        ·
          constructor
          · omega
          ·
            constructor
            · omega
            ·
              constructor
              · omega
              ·
                constructor
                · omega
                ·
                  constructor
                  · omega
                  ·
                    constructor
                    · omega
                    ·
                      constructor
                      · omega
                      ·
                        constructor
                        · omega
                        ·
                          constructor
                          · omega
                          ·
                            constructor
                            · omega
                            ·
                              constructor
                              · omega
                              ·
                                constructor
                                · omega
                                ·
                                  exact List.Forall₂.nil

def survivorContributionsChunk15 : List Nat := [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
def survivorBoundsChunk15 : List Nat := [94388, 94416, 94460, 94833, 94861, 95267, 95295, 95701, 95729, 96135, 96163, 96569, 96597, 97003, 97031, 97437]
theorem survivorBoundsSoundChunk15 : List.Forall₂ (fun actual bound => actual ≤ bound) survivorContributionsChunk15 survivorBoundsChunk15 :=
by
  constructor
  · omega
  ·
    constructor
    · omega
    ·
      constructor
      · omega
      ·
        constructor
        · omega
        ·
          constructor
          · omega
          ·
            constructor
            · omega
            ·
              constructor
              · omega
              ·
                constructor
                · omega
                ·
                  constructor
                  · omega
                  ·
                    constructor
                    · omega
                    ·
                      constructor
                      · omega
                      ·
                        constructor
                        · omega
                        ·
                          constructor
                          · omega
                          ·
                            constructor
                            · omega
                            ·
                              constructor
                              · omega
                              ·
                                constructor
                                · omega
                                ·
                                  exact List.Forall₂.nil

def survivorContributionsChunk16 : List Nat := [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
def survivorBoundsChunk16 : List Nat := [97465, 97871, 97899, 98305, 98333, 98739, 98767, 99173, 99201, 99607, 99635, 100041, 100069, 100475, 100503, 100909]
theorem survivorBoundsSoundChunk16 : List.Forall₂ (fun actual bound => actual ≤ bound) survivorContributionsChunk16 survivorBoundsChunk16 :=
by
  constructor
  · omega
  ·
    constructor
    · omega
    ·
      constructor
      · omega
      ·
        constructor
        · omega
        ·
          constructor
          · omega
          ·
            constructor
            · omega
            ·
              constructor
              · omega
              ·
                constructor
                · omega
                ·
                  constructor
                  · omega
                  ·
                    constructor
                    · omega
                    ·
                      constructor
                      · omega
                      ·
                        constructor
                        · omega
                        ·
                          constructor
                          · omega
                          ·
                            constructor
                            · omega
                            ·
                              constructor
                              · omega
                              ·
                                constructor
                                · omega
                                ·
                                  exact List.Forall₂.nil

def survivorContributionsChunk17 : List Nat := [1, 1, 1, 1, 1, 1]
def survivorBoundsChunk17 : List Nat := [100937, 101343, 101371, 101777, 101805, 107181]
theorem survivorBoundsSoundChunk17 : List.Forall₂ (fun actual bound => actual ≤ bound) survivorContributionsChunk17 survivorBoundsChunk17 :=
by
  constructor
  · omega
  ·
    constructor
    · omega
    ·
      constructor
      · omega
      ·
        constructor
        · omega
        ·
          constructor
          · omega
          ·
            constructor
            · omega
            ·
              exact List.Forall₂.nil

def survivorContributionsTree0_0 : List Nat := survivorContributionsChunk0 ++ survivorContributionsChunk1
def survivorBoundsTree0_0 : List Nat := survivorBoundsChunk0 ++ survivorBoundsChunk1
theorem survivorBoundsSoundTree0_0 : List.Forall₂ (fun actual bound => actual ≤ bound) survivorContributionsTree0_0 survivorBoundsTree0_0 := by
  exact forall₂_append survivorBoundsSoundChunk0 survivorBoundsSoundChunk1
def survivorContributionsTree0_1 : List Nat := survivorContributionsChunk2 ++ survivorContributionsChunk3
def survivorBoundsTree0_1 : List Nat := survivorBoundsChunk2 ++ survivorBoundsChunk3
theorem survivorBoundsSoundTree0_1 : List.Forall₂ (fun actual bound => actual ≤ bound) survivorContributionsTree0_1 survivorBoundsTree0_1 := by
  exact forall₂_append survivorBoundsSoundChunk2 survivorBoundsSoundChunk3
def survivorContributionsTree0_2 : List Nat := survivorContributionsChunk4 ++ survivorContributionsChunk5
def survivorBoundsTree0_2 : List Nat := survivorBoundsChunk4 ++ survivorBoundsChunk5
theorem survivorBoundsSoundTree0_2 : List.Forall₂ (fun actual bound => actual ≤ bound) survivorContributionsTree0_2 survivorBoundsTree0_2 := by
  exact forall₂_append survivorBoundsSoundChunk4 survivorBoundsSoundChunk5
def survivorContributionsTree0_3 : List Nat := survivorContributionsChunk6 ++ survivorContributionsChunk7
def survivorBoundsTree0_3 : List Nat := survivorBoundsChunk6 ++ survivorBoundsChunk7
theorem survivorBoundsSoundTree0_3 : List.Forall₂ (fun actual bound => actual ≤ bound) survivorContributionsTree0_3 survivorBoundsTree0_3 := by
  exact forall₂_append survivorBoundsSoundChunk6 survivorBoundsSoundChunk7
def survivorContributionsTree0_4 : List Nat := survivorContributionsChunk8 ++ survivorContributionsChunk9
def survivorBoundsTree0_4 : List Nat := survivorBoundsChunk8 ++ survivorBoundsChunk9
theorem survivorBoundsSoundTree0_4 : List.Forall₂ (fun actual bound => actual ≤ bound) survivorContributionsTree0_4 survivorBoundsTree0_4 := by
  exact forall₂_append survivorBoundsSoundChunk8 survivorBoundsSoundChunk9
def survivorContributionsTree0_5 : List Nat := survivorContributionsChunk10 ++ survivorContributionsChunk11
def survivorBoundsTree0_5 : List Nat := survivorBoundsChunk10 ++ survivorBoundsChunk11
theorem survivorBoundsSoundTree0_5 : List.Forall₂ (fun actual bound => actual ≤ bound) survivorContributionsTree0_5 survivorBoundsTree0_5 := by
  exact forall₂_append survivorBoundsSoundChunk10 survivorBoundsSoundChunk11
def survivorContributionsTree0_6 : List Nat := survivorContributionsChunk12 ++ survivorContributionsChunk13
def survivorBoundsTree0_6 : List Nat := survivorBoundsChunk12 ++ survivorBoundsChunk13
theorem survivorBoundsSoundTree0_6 : List.Forall₂ (fun actual bound => actual ≤ bound) survivorContributionsTree0_6 survivorBoundsTree0_6 := by
  exact forall₂_append survivorBoundsSoundChunk12 survivorBoundsSoundChunk13
def survivorContributionsTree0_7 : List Nat := survivorContributionsChunk14 ++ survivorContributionsChunk15
def survivorBoundsTree0_7 : List Nat := survivorBoundsChunk14 ++ survivorBoundsChunk15
theorem survivorBoundsSoundTree0_7 : List.Forall₂ (fun actual bound => actual ≤ bound) survivorContributionsTree0_7 survivorBoundsTree0_7 := by
  exact forall₂_append survivorBoundsSoundChunk14 survivorBoundsSoundChunk15
def survivorContributionsTree0_8 : List Nat := survivorContributionsChunk16 ++ survivorContributionsChunk17
def survivorBoundsTree0_8 : List Nat := survivorBoundsChunk16 ++ survivorBoundsChunk17
theorem survivorBoundsSoundTree0_8 : List.Forall₂ (fun actual bound => actual ≤ bound) survivorContributionsTree0_8 survivorBoundsTree0_8 := by
  exact forall₂_append survivorBoundsSoundChunk16 survivorBoundsSoundChunk17
def survivorContributionsTree1_0 : List Nat := survivorContributionsTree0_0 ++ survivorContributionsTree0_1
def survivorBoundsTree1_0 : List Nat := survivorBoundsTree0_0 ++ survivorBoundsTree0_1
theorem survivorBoundsSoundTree1_0 : List.Forall₂ (fun actual bound => actual ≤ bound) survivorContributionsTree1_0 survivorBoundsTree1_0 := by
  exact forall₂_append survivorBoundsSoundTree0_0 survivorBoundsSoundTree0_1
def survivorContributionsTree1_1 : List Nat := survivorContributionsTree0_2 ++ survivorContributionsTree0_3
def survivorBoundsTree1_1 : List Nat := survivorBoundsTree0_2 ++ survivorBoundsTree0_3
theorem survivorBoundsSoundTree1_1 : List.Forall₂ (fun actual bound => actual ≤ bound) survivorContributionsTree1_1 survivorBoundsTree1_1 := by
  exact forall₂_append survivorBoundsSoundTree0_2 survivorBoundsSoundTree0_3
def survivorContributionsTree1_2 : List Nat := survivorContributionsTree0_4 ++ survivorContributionsTree0_5
def survivorBoundsTree1_2 : List Nat := survivorBoundsTree0_4 ++ survivorBoundsTree0_5
theorem survivorBoundsSoundTree1_2 : List.Forall₂ (fun actual bound => actual ≤ bound) survivorContributionsTree1_2 survivorBoundsTree1_2 := by
  exact forall₂_append survivorBoundsSoundTree0_4 survivorBoundsSoundTree0_5
def survivorContributionsTree1_3 : List Nat := survivorContributionsTree0_6 ++ survivorContributionsTree0_7
def survivorBoundsTree1_3 : List Nat := survivorBoundsTree0_6 ++ survivorBoundsTree0_7
theorem survivorBoundsSoundTree1_3 : List.Forall₂ (fun actual bound => actual ≤ bound) survivorContributionsTree1_3 survivorBoundsTree1_3 := by
  exact forall₂_append survivorBoundsSoundTree0_6 survivorBoundsSoundTree0_7
def survivorContributionsTree2_0 : List Nat := survivorContributionsTree1_0 ++ survivorContributionsTree1_1
def survivorBoundsTree2_0 : List Nat := survivorBoundsTree1_0 ++ survivorBoundsTree1_1
theorem survivorBoundsSoundTree2_0 : List.Forall₂ (fun actual bound => actual ≤ bound) survivorContributionsTree2_0 survivorBoundsTree2_0 := by
  exact forall₂_append survivorBoundsSoundTree1_0 survivorBoundsSoundTree1_1
def survivorContributionsTree2_1 : List Nat := survivorContributionsTree1_2 ++ survivorContributionsTree1_3
def survivorBoundsTree2_1 : List Nat := survivorBoundsTree1_2 ++ survivorBoundsTree1_3
theorem survivorBoundsSoundTree2_1 : List.Forall₂ (fun actual bound => actual ≤ bound) survivorContributionsTree2_1 survivorBoundsTree2_1 := by
  exact forall₂_append survivorBoundsSoundTree1_2 survivorBoundsSoundTree1_3
def survivorContributionsTree3_0 : List Nat := survivorContributionsTree2_0 ++ survivorContributionsTree2_1
def survivorBoundsTree3_0 : List Nat := survivorBoundsTree2_0 ++ survivorBoundsTree2_1
theorem survivorBoundsSoundTree3_0 : List.Forall₂ (fun actual bound => actual ≤ bound) survivorContributionsTree3_0 survivorBoundsTree3_0 := by
  exact forall₂_append survivorBoundsSoundTree2_0 survivorBoundsSoundTree2_1
def survivorContributionsTree4_0 : List Nat := survivorContributionsTree3_0 ++ survivorContributionsTree0_8
def survivorBoundsTree4_0 : List Nat := survivorBoundsTree3_0 ++ survivorBoundsTree0_8
theorem survivorBoundsSoundTree4_0 : List.Forall₂ (fun actual bound => actual ≤ bound) survivorContributionsTree4_0 survivorBoundsTree4_0 := by
  exact forall₂_append survivorBoundsSoundTree3_0 survivorBoundsSoundTree0_8
def survivorContributions : List Nat := survivorContributionsTree4_0
def survivorBounds : List Nat := survivorBoundsTree4_0
theorem survivorBoundsSound : List.Forall₂ (fun actual bound => actual ≤ bound) survivorContributions survivorBounds := by
  exact survivorBoundsSoundTree4_0

theorem prefoldResult : prefoldTerms = rootTerms := by rfl

theorem prefoldBoundSound : rootBound ≤ prefoldBound := by decide +kernel


theorem prefoldSound :
  preFoldBound rootBound prefoldBound survivorContributions survivorBounds := by
  exact (preFoldSound rootTerms prefoldTerms prefoldResult prefoldBoundSound survivorBoundsSound).2

theorem endResult : endTerms = prefoldTerms := by rfl

theorem endSummaryResult : endSummary = prefoldSummary := by rfl

theorem endSound :
  endTerms = prefoldTerms ∧ endSummary = prefoldSummary := by
  exact ⟨endResult, endSummaryResult⟩

theorem invocationEndClaimSound (env : Env Owner) (actual : Int)
    (claim : ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 env actual (.exact rootTerms rootSummary)) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 env actual (.exact endTerms endSummary) := by
  exact invocationEndSound 100418593683253592432016548326729029359133068138294319235841 env actual rootTerms endTerms rootSummary endSummary
    claim endResult endSummaryResult

theorem selectedRootResultAt : (history.lookup rootResultEvent).map AnnotatedEvent.event = some (.resultExact selectedOwner rootRaw rootSummary) := by
  rfl

theorem selectedPreFoldAt : (history.lookup prefoldEvent).map AnnotatedEvent.event = some (.preFoldPolynomial rootResultEvent prefoldRaw prefoldSummary (some (.result 107564 .summary))) := by
  rfl

theorem selectedInvocationEndAt : (history.lookup endEvent).map AnnotatedEvent.event = some (.invocationEndExact selectedOwner prefoldEvent endRaw endSummary) := by
  rfl

end Bound0

def theoremCount : Nat := 46

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticShard420
