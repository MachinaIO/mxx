import MxxIR
import MxxRuntime

set_option maxRecDepth 16384
set_option maxHeartbeats 2000000

namespace Stage_nand

structure Params where
  unit : Unit

/-- The packed coefficients of node 2 in `generatedRoot`. -/
def generatedRoot.table_2 : Nat := 26690917011920845083239682246246085285890080641497476984368583986199229026977992866419362096317221648308883837536175778598273460070876657951137201386574463801773638447222378603836230826737511964072364082421451285867669881813841778868591615901439412138875015251046305742714148456088521452339381583951809131762427070322305576950417750770325460451194004769013605809208162290105902810246707965763981103003328413769294174316366269682372250360739177412042766023470139551884195535077142283002006258776518354348328955771221252124556716179634330265746630274832360976474528657466334495639431986969868348186323981358615115376345770620152518449275547014409221979342581191621623645781257141391861158841989255907215339423373064698698642551805983724870508854954195941366930844733698701149690977917644603865477369852282281630072581265932696755165132868450410267389709254595679952478292470686354469090054708485293205656926016219078857819114827912974152342220612408172231606322441705067713572180954455760007174701005744455496703262916664370711595479181785860351953540753876222936952252555873400912234517306864882601415433827970926796373712595618187911122948973380426152850368576251949724214207235478173425342767166994942001988568257912185397944038106757288635555823523923942992438225453113482281157928834639352077468830309742049682532327095483315173740943431525098146335535303831020554450193294467401880125975967719855559812102344063455584405655025211990514583138549545227042624237631045744046243487535286221783163167823016277063465762102420676313679488984297559914548974789920535159724107502181390502112893089841773596093940692256436496869965549586624496803493699811324237106503262204494194039744348575199253081063384674168126988338718836068366583508044979756663431170176857932238873553048713775076361948705548118116314731415724952178795194602335703273868141195018635072767971066806137031879639192725866528697805689628102737067471268178572498421402008085568737861288439520657041974593284803037820027218918764679311932940741336726532543256422281911008254718111226141034207641063090386252246371019606833672432440942609570233057322155481951227262311690199904616680774292599717759081722765825601066451697185483439260656469297136555631386128422074034568224447527313533243991214128967494725931042124354828116380247940074422415478642468641194484786634576043558380369465355098052834511882151258602839987878313373697942189925305862026378662463638043305018578805771673592298598513346734624276501059425825088967269382587518195375182492338845675875009088146023923650555728119683428529592243588356126357837219346700344395691152295746527041341871168546231032259323288068827307630483692181075265712169264503387632389957231206024666333697004117670268266018759716113405937386993691361600348574617474044081103799290722346823817588000997766630339362643206082389041134126356911243882136568359296740018315346416492943954669998417459006227864284609066768144578151635656626475661778136500263573435834712891938500337175376260347050106929934097835406598671037026406850738988540285082700187242045388227386102827845664147945318616452712042081653914629491796433231649182587008378025811825224229628257268627245688520471729988754166612408932204401116074922686128407653809266196638159906938573565152627381058024330667928030168285154448403371687493134106377141295662794020237865050550939119013289699766814665356117001161540748210739194293753056890242781669419809375153552720644563377083540346318598462449668201012662366031341338587908386254175391195636598214626710629840889367304190763087414793948660313683470413682862365084413690382579743934716597426043946876658791906846111228205080415904514923261405931576273041140898914325375802686030379558277382143046015608182459340061297594949123542860678627184778851066516834805355439879426320687623680235071406705707905160760928952842846447687691372628682298636241901695063859125859457026997839193410473568440011746067834844103528299422465241446770430450789698390401008818908902632172084193236452774712846636593369210188848436999998895471577420977691372254935964730483908083178644725260677655802936982831925357369796084604429149867695015787312245809860427530673965409413698797872967377846618228511697567043646023298309735384556447965075348644740831494721010014701629204444609769736712411157264143965411866987197911346703170617048507080561205314147951600216430935220409005980931988064446910522824333970808383124865545266349686992783178810092047216506000151942143406955313374026408138039904899252898684752604465655468183982057461893158836584250690195847952100623437562457623978939283094703406607885811829300953425126293489604771606277337565507472902404108032154539026988384048340765711661330159133078491191593385318311196380842191033143112229863277464439363678182649118819611283355244150782557963518397027508932479946817801103687707956007576456323010448636447172396476922418798950745231561880784521262840689385957207844380689961604499665428917180054447843096535943882488303185955088568237889211644378102092426940611029510311333128877925644120211987258263870089639420358962072055048726320383596053005747087714688188994000963212607802955680630498867146046407854007800485148298765690668959839274636471233249311045662807656399617674869919783734278328832859151249930490997645739618859295216454061866635239728537269164551954370793651550430869808603826849622603403595032225204541490978444135114727230235219096322142027628969052346525000200359658845182837346115271074003047424831356471292821106156586405882430623886861645226303887281266121459845430640992380787429123545585937500985436190865810042607734465834546242500035683260332200376099738706474646272487760551567500668600495468174746800985482277104129402211592104071779891173254922756437838276045841290215358938663894906208891685233253061995430539339807608791942966258582230320523336213204178596461705659690553651960042245475444575343559965510824159054661244213497424957929750132842885527674931489437799832165197155761859078533886230689010582603971743528814581342116946023676000768140917187152620899210787485155326822620091136012644748822441192639064235159767429502592792689015919486121082836254568730834969901258063444485485359057148373016369399132506409219664639237251186684080033931829241465238397028797512715505953043524830236345208173651082593314087157070564660357922061124327214417840925304747984996517826158129821115206203054548177297540718320884891445048554581194052665210419619435193976102272362356996946569353632089198837611355307013035235878274548929795363243477837371392263034049379567404944865093714341649126797313385535653188428819464038374106799707753676323235647789113398560941000819550506135539710762483105538136749938305846893726457438516725463089329205874267215570916880030533645992206583191700959349679736136246083038897403069450650474202421936331271739186840462035560244557278577436149102784454470941247107190848687076796251504756787002189541266587731060154759862993377432068599055664637614719692988020999602051211674879008742982336764445943423021813204647138323238427393799298104654138470661009312875713724278920585846109915405455093062323960570603349987559172763976965909346864873548268752510343435741478160108232240640968385014650478755812334452715937754424674431344002785564443913777361546445227804410109933134379016731852158209934505348302086859367289676466796152230892636305989933175793214820300385910046371397203488022342580672580664088028324676311805936564128360349162787955025701977126451983110118796027990042102845505024843675501599058396823650569281947471336299184942433805871467484565619542593731776917410650171885637931007466581773237391398646050882812734532049500188913342822974777407394646956190101154997921750435258924104113181094433260726015654900594557945114886713865319652460984896433973934748888575396550744212234544555380211072280358934102822588090745868622384198654334730429936335816708127192281415869237794442670983368275575639387267510460551801529906471264930916109467692341087603068463399912815778187289367112639479512162630541914853699514887325816830364944864808186866115847288025205131233219269470080319316617531292546325496751234205538253057462780701102158875133185079384803168352351042138679397920650726779885344152111175701706912858626332356375596097322266045104394858425593281827557731083153780196982312393997507448892208951529400067267900882653277041231133123768307807151851183182226076688102010486354478663632913073909510498874339642746710425910564913318558651137297238744462187807180809993720776905896932839723005360070827032358686114896234438798904177910751624060560840967575828946882349587050111776617615416285242684638602623856054613766990563319675612801428141331508421812110190101960326614096388328035056100603288070530341119010085646492731038956413814902413706269296743375097952985104749218567821884505599413899166940447796960414724219833844282797569796868834669527177424087262856416900003269975454799490181243615415319938564009056532466124425600494609698298246747437641359569689291478515707333952118361031918670049275828796777569974928549580512961962961957983799464126368217605591571304828339194065399463062587439327375347214341951063539104206258786449357929513901592590060620799507912181876873053063200440552484903036588878671209431437737701958240000

abbrev parallel_generatedRoot_31.constraints_0 (w_3_0 : Int) (w_4_0 : Int) (outputs : Int) : Prop :=
  w_3_0 ≠ 0 ∧
  outputs = w_4_0

def parallel_generatedRoot_31 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (inputs : Int × Int × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : Int := inputs.1
  let w_1_0 : Int := inputs.2.1
  let w_3_0 : Int := inputs.2.2.1
  let w_2_0 : Int := (w_0_0 + w_1_0)
  let w_4_0 : Int := (w_2_0 % w_3_0)
  parallel_generatedRoot_31.constraints_0 w_3_0 w_4_0 outputs

abbrev parallel_generatedRoot_32.constraints_0 (w_3_0 : Int) (w_4_0 : Int) (outputs : Int) : Prop :=
  w_3_0 ≠ 0 ∧
  outputs = w_4_0

def parallel_generatedRoot_32 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (inputs : Int × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let _backend := backend
  let w_1_0 : Int := inputs.1
  let w_3_0 : Int := inputs.2.1
  let w_0_0 : Int := 0
  let w_2_0 : Int := (w_0_0 - w_1_0)
  let w_4_0 : Int := (w_2_0 % w_3_0)
  parallel_generatedRoot_32.constraints_0 w_3_0 w_4_0 outputs

abbrev parallel_scope_tfhe_blind_rotation_6.constraints_0 (w_5_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  w_5_0 ≠ 0 ∧
  w_7_0 ≠ 0 ∧
  outputs = w_8_0

def parallel_scope_tfhe_blind_rotation_6 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (inputs : Int) (outputs : Int) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : Int := inputs
  let w_1_0 : Int := 2048
  let w_2_0 : Int := (w_0_0 * w_1_0)
  let w_3_0 : Int := 2147483648
  let w_4_0 : Int := (w_2_0 + w_3_0)
  let w_5_0 : Int := 4294967296
  let w_6_0 : Int := (w_4_0 / w_5_0)
  let w_7_0 : Int := 2048
  let w_8_0 : Int := (w_6_0 % w_7_0)
  parallel_scope_tfhe_blind_rotation_6.constraints_0 w_5_0 w_7_0 w_8_0 outputs

abbrev sequential_scope_tfhe_blind_rotation_7.constraints_0 (backend : MxxRuntime.BackendContext) (w_1_0 : Fin 630 → Mxx.Primitives.ExactMatrix 5234636801 1024 1 12) (w_4_0 : Fin 630 → Mxx.Primitives.ExactMatrix 5234636801 1024 1 12) (w_7_0 : Fin 630 → Int) (w_2_0 : Int) (w_3_0 : Mxx.Primitives.ExactMatrix 5234636801 1024 1 12) (w_5_0 : Mxx.Primitives.ExactMatrix 5234636801 1024 1 12) (w_6_0 : Mxx.Primitives.ExactMatrix 5234636801 1024 2 12) (w_8_0 : Int) (w_10_0 : Mxx.Primitives.ExactMatrix 5234636801 1024 2 1) (w_11_0 : Mxx.Primitives.ExactMatrix 5234636801 1024 12 1) (w_13_0 : Mxx.Primitives.ExactMatrix 5234636801 1024 2 1) (outputs : Mxx.Primitives.ExactMatrix 5234636801 1024 2 1) : Prop :=
  0 ≤ w_2_0 ∧
  w_2_0 < 630 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_2_0 w_3_0 ∧
  0 ≤ w_2_0 ∧
  w_2_0 < 630 ∧
  MxxRuntime.familyGetDynamic w_4_0 w_2_0 w_5_0 ∧
  MxxRuntime.concatRows w_3_0 w_5_0 w_6_0 ∧
  0 ≤ w_2_0 ∧
  w_2_0 < 630 ∧
  MxxRuntime.familyGetDynamic w_7_0 w_2_0 w_8_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 64 6 w_10_0 w_11_0 ∧
  outputs = w_13_0

def sequential_scope_tfhe_blind_rotation_7 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (inputs : Mxx.Primitives.ExactMatrix 5234636801 1024 2 1 × (Fin 630 → Mxx.Primitives.ExactMatrix 5234636801 1024 1 12) × (Fin 630 → Mxx.Primitives.ExactMatrix 5234636801 1024 1 12) × (Fin 630 → Int) × Unit) (outputs : Mxx.Primitives.ExactMatrix 5234636801 1024 2 1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : Mxx.Primitives.ExactMatrix 5234636801 1024 2 1 := inputs.1
  let w_1_0 : Fin 630 → Mxx.Primitives.ExactMatrix 5234636801 1024 1 12 := inputs.2.1
  let w_4_0 : Fin 630 → Mxx.Primitives.ExactMatrix 5234636801 1024 1 12 := inputs.2.2.1
  let w_7_0 : Fin 630 → Int := inputs.2.2.2.1
  let w_2_0 : Int := (Int.ofNat i_0)
  ∃ (w_3_0 : Mxx.Primitives.ExactMatrix 5234636801 1024 1 12),
    ∃ (w_5_0 : Mxx.Primitives.ExactMatrix 5234636801 1024 1 12),
      ∃ (w_6_0 : Mxx.Primitives.ExactMatrix 5234636801 1024 2 12),
        ∃ (w_8_0 : Int),
          let w_9_0 : Mxx.Primitives.ExactMatrix 5234636801 1024 2 1 := MxxRuntime.multiplyMonomial w_0_0 w_8_0
          let w_10_0 : Mxx.Primitives.ExactMatrix 5234636801 1024 2 1 := MxxRuntime.matrixSub w_9_0 w_0_0
          ∃ (w_11_0 : Mxx.Primitives.ExactMatrix 5234636801 1024 12 1),
            let w_12_0 : Mxx.Primitives.ExactMatrix 5234636801 1024 2 1 := MxxRuntime.matrixMul w_6_0 w_11_0
            let w_13_0 : Mxx.Primitives.ExactMatrix 5234636801 1024 2 1 := MxxRuntime.matrixAdd w_0_0 w_12_0
            sequential_scope_tfhe_blind_rotation_7.constraints_0 backend w_1_0 w_4_0 w_7_0 w_2_0 w_3_0 w_5_0 w_6_0 w_8_0 w_10_0 w_11_0 w_13_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure scope_tfhe_blind_rotation.Witness where
  w_2_0 : Mxx.Primitives.ExactMatrix 5234636801 1024 2 1
  w_6_0 : Fin 630 → Int
  w_7_0 : Mxx.Primitives.ExactMatrix 5234636801 1024 2 1
  w_8_0 : Mxx.Primitives.ExactMatrix 5234636801 1024 1 1
  w_9_0 : Mxx.Primitives.ExactMatrix 5234636801 1024 1 1

abbrev scope_tfhe_blind_rotation.constraints_0 (backend : MxxRuntime.BackendContext) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (w_0_0 : Mxx.Primitives.ExactMatrix 5234636801 1024 1 1) (w_1_0 : Mxx.Primitives.ExactMatrix 5234636801 1024 1 1) (w_5_0 : Fin 630 → Int) (w_3_0 : Fin 630 → Mxx.Primitives.ExactMatrix 5234636801 1024 1 12) (w_4_0 : Fin 630 → Mxx.Primitives.ExactMatrix 5234636801 1024 1 12) (w_2_0 : Mxx.Primitives.ExactMatrix 5234636801 1024 2 1) (w_6_0 : Fin 630 → Int) (w_7_0 : Mxx.Primitives.ExactMatrix 5234636801 1024 2 1) (w_8_0 : Mxx.Primitives.ExactMatrix 5234636801 1024 1 1) (w_9_0 : Mxx.Primitives.ExactMatrix 5234636801 1024 1 1) (outputs : Mxx.Primitives.ExactMatrix 5234636801 1024 1 1 × Mxx.Primitives.ExactMatrix 5234636801 1024 1 1 × Unit) : Prop :=
  MxxRuntime.concatRows w_0_0 w_1_0 w_2_0 ∧
  (630) = 630 ∧
  (∀ i : Fin 630, parallel_scope_tfhe_blind_rotation_6 backend tape (path ++ [6, i.val]) params i (w_5_0 i) (w_6_0 i)) ∧
  0 ≤ (630) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Mxx.Primitives.ExactMatrix 5234636801 1024 2 1) => sequential_scope_tfhe_blind_rotation_7 backend tape (path ++ [7, i]) params i (current, (w_3_0, (w_4_0, (w_6_0, ())))) next) (Int.toNat (630)) w_2_0 w_7_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 2 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  MxxRuntime.sliceMatrix w_7_0 0 1 0 1 w_8_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 2 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  MxxRuntime.sliceMatrix w_7_0 1 2 0 1 w_9_0 ∧
  outputs = (w_8_0, (w_9_0, ()))

abbrev scope_tfhe_blind_rotation.body (backend : MxxRuntime.BackendContext) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (inputs : Mxx.Primitives.ExactMatrix 5234636801 1024 1 1 × Mxx.Primitives.ExactMatrix 5234636801 1024 1 1 × (Fin 630 → Int) × (Fin 630 → Mxx.Primitives.ExactMatrix 5234636801 1024 1 12) × (Fin 630 → Mxx.Primitives.ExactMatrix 5234636801 1024 1 12) × Unit) (outputs : Mxx.Primitives.ExactMatrix 5234636801 1024 1 1 × Mxx.Primitives.ExactMatrix 5234636801 1024 1 1 × Unit) (witness : scope_tfhe_blind_rotation.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : Mxx.Primitives.ExactMatrix 5234636801 1024 1 1 := inputs.1
  let w_1_0 : Mxx.Primitives.ExactMatrix 5234636801 1024 1 1 := inputs.2.1
  let w_5_0 : Fin 630 → Int := inputs.2.2.1
  let w_3_0 : Fin 630 → Mxx.Primitives.ExactMatrix 5234636801 1024 1 12 := inputs.2.2.2.1
  let w_4_0 : Fin 630 → Mxx.Primitives.ExactMatrix 5234636801 1024 1 12 := inputs.2.2.2.2.1
  let w_2_0 := witness.w_2_0
  let w_6_0 := witness.w_6_0
  let w_7_0 := witness.w_7_0
  let w_8_0 := witness.w_8_0
  let w_9_0 := witness.w_9_0
  scope_tfhe_blind_rotation.constraints_0 backend tape path params w_0_0 w_1_0 w_5_0 w_3_0 w_4_0 w_2_0 w_6_0 w_7_0 w_8_0 w_9_0 outputs

def scope_tfhe_blind_rotation (backend : MxxRuntime.BackendContext) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (inputs : Mxx.Primitives.ExactMatrix 5234636801 1024 1 1 × Mxx.Primitives.ExactMatrix 5234636801 1024 1 1 × (Fin 630 → Int) × (Fin 630 → Mxx.Primitives.ExactMatrix 5234636801 1024 1 12) × (Fin 630 → Mxx.Primitives.ExactMatrix 5234636801 1024 1 12) × Unit) (outputs : Mxx.Primitives.ExactMatrix 5234636801 1024 1 1 × Mxx.Primitives.ExactMatrix 5234636801 1024 1 1 × Unit) : Prop :=
  ∃ witness : scope_tfhe_blind_rotation.Witness,
    scope_tfhe_blind_rotation.body backend tape path params inputs outputs witness

abbrev parallel_generatedRoot_38.constraints_0 (w_0_0 : Fin 1024 → Int) (w_23_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_6_0 : Int) (w_15_0 : Int) (w_21_0 : Int) (w_24_0 : Int) (outputs : Int) : Prop :=
  w_4_0 ≠ 0 ∧
  0 ≤ w_5_0 ∧
  w_5_0 < 1024 ∧
  MxxRuntime.familyGetDynamic w_0_0 w_5_0 w_6_0 ∧
  w_15_0 ≠ 0 ∧
  w_21_0 ≠ 0 ∧
  w_23_0 ≠ 0 ∧
  outputs = w_24_0

def parallel_generatedRoot_38 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (inputs : (Fin 1024 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : Fin 1024 → Int := inputs.1
  let w_23_0 : Int := inputs.2.1
  let w_1_0 : Int := 1024
  let w_2_0 : Int := (Int.ofNat i_0)
  let w_3_0 : Int := (w_1_0 - w_2_0)
  let w_4_0 : Int := 1024
  let w_5_0 : Int := (w_3_0 % w_4_0)
  ∃ (w_6_0 : Int),
    let w_7_0 : Int := 0
    let w_8_0 : Bool := decide (w_2_0 = w_7_0)
    let w_9_0 : Int := if w_8_0 then 1 else 0
    let w_10_0 : Int := 2
    let w_11_0 : Int := (w_9_0 * w_10_0)
    let w_12_0 : Int := 1
    let w_13_0 : Int := (w_11_0 - w_12_0)
    let w_14_0 : Int := (w_6_0 * w_13_0)
    let w_15_0 : Int := 5234636801
    let w_16_0 : Int := (w_14_0 % w_15_0)
    let w_17_0 : Int := 4294967296
    let w_18_0 : Int := (w_16_0 * w_17_0)
    let w_19_0 : Int := 2617318400
    let w_20_0 : Int := (w_18_0 + w_19_0)
    let w_21_0 : Int := 5234636801
    let w_22_0 : Int := (w_20_0 / w_21_0)
    let w_24_0 : Int := (w_22_0 % w_23_0)
    parallel_generatedRoot_38.constraints_0 w_0_0 w_23_0 w_4_0 w_5_0 w_6_0 w_15_0 w_21_0 w_24_0 outputs

abbrev parallel_generatedRoot_47.constraints_0 (w_0_0 : Fin 1024 → Int) (w_2_0 : Int) (w_3_0 : Int) (w_4_0 : Int) (w_7_0 : Int) (w_9_0 : Int) (w_11_0 : Int) (w_12_0 : Int) (w_15_0 : Int) (w_18_0 : Int) (w_21_0 : Int) (w_24_0 : Int) (w_27_0 : Int) (w_30_0 : Int) (w_33_0 : Int) (w_36_0 : Int) (w_37_0 : Int) (w_39_0 : Int) (w_40_0 : Int) (outputs : Int) : Prop :=
  w_2_0 ≠ 0 ∧
  0 ≤ w_3_0 ∧
  w_3_0 < 1024 ∧
  MxxRuntime.familyGetDynamic w_0_0 w_3_0 w_4_0 ∧
  w_7_0 ≠ 0 ∧
  w_9_0 ≠ 0 ∧
  w_11_0 ≠ 0 ∧
  0 ≤ w_12_0 ∧
  w_12_0 < 8 ∧
  8 = 8 ∧
  MxxRuntime.select w_12_0 [w_15_0, w_18_0, w_21_0, w_24_0, w_27_0, w_30_0, w_33_0, w_36_0] w_37_0 ∧
  w_37_0 ≠ 0 ∧
  w_39_0 ≠ 0 ∧
  outputs = w_40_0

def parallel_generatedRoot_47 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (inputs : (Fin 1024 → Int) × Int × Int × Int × Int × Int × Int × Int × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : Fin 1024 → Int := inputs.1
  let w_13_0 : Int := inputs.2.1
  let w_16_0 : Int := inputs.2.2.1
  let w_19_0 : Int := inputs.2.2.2.1
  let w_22_0 : Int := inputs.2.2.2.2.1
  let w_25_0 : Int := inputs.2.2.2.2.2.1
  let w_28_0 : Int := inputs.2.2.2.2.2.2.1
  let w_31_0 : Int := inputs.2.2.2.2.2.2.2.1
  let w_34_0 : Int := inputs.2.2.2.2.2.2.2.2.1
  let w_1_0 : Int := (Int.ofNat i_0)
  let w_2_0 : Int := 8
  let w_3_0 : Int := (w_1_0 / w_2_0)
  ∃ (w_4_0 : Int),
    let w_5_0 : Int := 32768
    let w_6_0 : Int := (w_4_0 + w_5_0)
    let w_7_0 : Int := 65536
    let w_8_0 : Int := (w_6_0 / w_7_0)
    let w_9_0 : Int := 65536
    let w_10_0 : Int := (w_8_0 % w_9_0)
    let w_11_0 : Int := 8
    let w_12_0 : Int := (w_1_0 % w_11_0)
    let w_14_0 : Int := 0
    let w_15_0 : Int := (w_13_0 + w_14_0)
    let w_17_0 : Int := 0
    let w_18_0 : Int := (w_16_0 + w_17_0)
    let w_20_0 : Int := 0
    let w_21_0 : Int := (w_19_0 + w_20_0)
    let w_23_0 : Int := 0
    let w_24_0 : Int := (w_22_0 + w_23_0)
    let w_26_0 : Int := 0
    let w_27_0 : Int := (w_25_0 + w_26_0)
    let w_29_0 : Int := 0
    let w_30_0 : Int := (w_28_0 + w_29_0)
    let w_32_0 : Int := 0
    let w_33_0 : Int := (w_31_0 + w_32_0)
    let w_35_0 : Int := 0
    let w_36_0 : Int := (w_34_0 + w_35_0)
    ∃ (w_37_0 : Int),
      let w_38_0 : Int := (w_10_0 / w_37_0)
      let w_39_0 : Int := 4
      let w_40_0 : Int := (w_38_0 % w_39_0)
      parallel_generatedRoot_47.constraints_0 w_0_0 w_2_0 w_3_0 w_4_0 w_7_0 w_9_0 w_11_0 w_12_0 w_15_0 w_18_0 w_21_0 w_24_0 w_27_0 w_30_0 w_33_0 w_36_0 w_37_0 w_39_0 w_40_0 outputs

abbrev parallel_generatedRoot_50.constraints_0 (w_3_0 : Int) (w_4_0 : Int) (outputs : Int) : Prop :=
  w_3_0 ≠ 0 ∧
  outputs = w_4_0

def parallel_generatedRoot_50 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (inputs : Int × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let _backend := backend
  let w_1_0 : Int := inputs.1
  let w_3_0 : Int := inputs.2.1
  let w_0_0 : Int := 0
  let w_2_0 : Int := (w_0_0 - w_1_0)
  let w_4_0 : Int := (w_2_0 % w_3_0)
  parallel_generatedRoot_50.constraints_0 w_3_0 w_4_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure generatedRoot.Witness where
  w_31_0 : Fin 630 → Int
  w_32_0 : Fin 630 → Int
  w_35_0 : Mxx.Primitives.ExactMatrix 5234636801 1024 1 1
  w_35_1 : Mxx.Primitives.ExactMatrix 5234636801 1024 1 1
  w_36_0 : Fin 1024 → Int
  w_38_0 : Fin 1024 → Int
  w_47_0 : Fin 8192 → Int
  w_50_0 : Fin 630 → Int
  w_51_0 : Fin 1024 → Int
  w_52_0 : Int
  w_64_0 : Int

abbrev generatedRoot.constraints_0 (backend : MxxRuntime.BackendContext) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (w_29_0 : Fin 630 → Int) (w_30_0 : Fin 630 → Int) (w_33_0 : Fin 630 → Mxx.Primitives.ExactMatrix 5234636801 1024 1 12) (w_34_0 : Fin 630 → Mxx.Primitives.ExactMatrix 5234636801 1024 1 12) (w_1_0 : Mxx.Primitives.ExactMatrix 5234636801 1024 1 1) (w_8_0 : Int) (w_11_0 : Int) (w_15_0 : Int) (w_21_0 : Int) (w_23_0 : Int) (w_26_0 : Int) (w_28_0 : Mxx.Primitives.ExactMatrix 5234636801 1024 1 1) (w_31_0 : Fin 630 → Int) (w_32_0 : Fin 630 → Int) (w_35_0 : Mxx.Primitives.ExactMatrix 5234636801 1024 1 1) (w_35_1 : Mxx.Primitives.ExactMatrix 5234636801 1024 1 1) (w_36_0 : Fin 1024 → Int) (w_37_0 : Int) (w_38_0 : Fin 1024 → Int) (w_39_0 : Int) (w_40_0 : Int) (w_41_0 : Int) (w_42_0 : Int) (w_43_0 : Int) (w_44_0 : Int) (w_45_0 : Int) (w_46_0 : Int) (w_47_0 : Fin 8192 → Int) (w_48_0 : Fin 630 → Int) (w_49_0 : Int) (w_50_0 : Fin 630 → Int) (w_51_0 : Fin 1024 → Int) (w_52_0 : Int) (w_53_0 : Int) (w_59_0 : Int) (w_63_0 : Fin 1 → Int) (w_64_0 : Int) (w_66_0 : Int) (outputs : (Fin 630 → Int) × Int × Unit) : Prop :=
  w_8_0 ≠ 0 ∧
  w_11_0 ≠ 0 ∧
  w_15_0 ≠ 0 ∧
  w_21_0 ≠ 0 ∧
  w_23_0 ≠ 0 ∧
  w_26_0 ≠ 0 ∧
  (630) = 630 ∧
  (∀ i : Fin 630, parallel_generatedRoot_31 backend tape (path ++ [31, i.val]) params i ((w_29_0 i), ((w_30_0 i), (w_8_0, ()))) (w_31_0 i)) ∧
  (630) = 630 ∧
  (∀ i : Fin 630, parallel_generatedRoot_32 backend tape (path ++ [32, i.val]) params i ((w_31_0 i), (w_11_0, ())) (w_32_0 i)) ∧
  scope_tfhe_blind_rotation backend tape (path ++ [35]) params (w_1_0, (w_28_0, (w_32_0, (w_33_0, (w_34_0, ()))))) (w_35_0, (w_35_1, ())) ∧
  MxxRuntime.polynomialValues false w_35_0 w_36_0 ∧
  (1024) = 1024 ∧
  (∀ i : Fin 1024, parallel_generatedRoot_38 backend tape (path ++ [38, i.val]) params i (w_36_0, (w_37_0, ())) (w_38_0 i)) ∧
  (8192) = 8192 ∧
  (∀ i : Fin 8192, parallel_generatedRoot_47 backend tape (path ++ [47, i.val]) params i (w_38_0, (w_39_0, (w_40_0, (w_41_0, (w_42_0, (w_43_0, (w_44_0, (w_45_0, (w_46_0, ()))))))))) (w_47_0 i)) ∧
  (630) = 630 ∧
  (∀ i : Fin 630, parallel_generatedRoot_50 backend tape (path ++ [50, i.val]) params i ((w_48_0 i), (w_49_0, ())) (w_50_0 i)) ∧
  MxxRuntime.polynomialValues false w_35_1 w_51_0 ∧
  MxxRuntime.familyGetStatic w_51_0 (0) w_52_0 ∧
  w_53_0 ≠ 0 ∧
  w_59_0 ≠ 0 ∧
  w_37_0 ≠ 0 ∧
  MxxRuntime.familyGetStatic w_63_0 (0) w_64_0 ∧
  w_49_0 ≠ 0 ∧
  outputs = (w_50_0, (w_66_0, ()))

abbrev generatedRoot.body (backend : MxxRuntime.BackendContext) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (inputs : (Fin 5160960 → Int) × Int × Int × (Fin 630 → Int) × (Fin 630 → Int) × (Fin 630 → Mxx.Primitives.ExactMatrix 5234636801 1024 1 12) × (Fin 630 → Mxx.Primitives.ExactMatrix 5234636801 1024 1 12) × (Fin 8192 → Int) × Unit) (outputs : (Fin 630 → Int) × Int × Unit) (witness : generatedRoot.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : Fin 5160960 → Int := inputs.1
  let w_5_0 : Int := inputs.2.1
  let w_6_0 : Int := inputs.2.2.1
  let w_29_0 : Fin 630 → Int := inputs.2.2.2.1
  let w_30_0 : Fin 630 → Int := inputs.2.2.2.2.1
  let w_33_0 : Fin 630 → Mxx.Primitives.ExactMatrix 5234636801 1024 1 12 := inputs.2.2.2.2.2.1
  let w_34_0 : Fin 630 → Mxx.Primitives.ExactMatrix 5234636801 1024 1 12 := inputs.2.2.2.2.2.2.1
  let w_62_0 : Fin 8192 → Int := inputs.2.2.2.2.2.2.2.1
  let w_1_0 : Mxx.Primitives.ExactMatrix 5234636801 1024 1 1 := (0 : Mxx.Primitives.ExactMatrix 5234636801 1024 1 1)
  let w_2_0 : Mxx.Primitives.ExactMatrix 5234636801 1024 1 1 := (MxxRuntime.packedPolynomial 30 1024 generatedRoot.table_2 : Mxx.Primitives.ExactMatrix 5234636801 1024 1 1)
  let w_3_0 : Int := 0
  let w_4_0 : Int := 0
  let w_7_0 : Int := (w_5_0 + w_6_0)
  let w_8_0 : Int := 4294967296
  let w_9_0 : Int := (w_7_0 % w_8_0)
  let w_10_0 : Int := (w_4_0 - w_9_0)
  let w_11_0 : Int := 4294967296
  let w_12_0 : Int := (w_10_0 % w_11_0)
  let w_13_0 : Int := 536870912
  let w_14_0 : Int := (w_12_0 + w_13_0)
  let w_15_0 : Int := 4294967296
  let w_16_0 : Int := (w_14_0 % w_15_0)
  let w_17_0 : Int := 2048
  let w_18_0 : Int := (w_16_0 * w_17_0)
  let w_19_0 : Int := 2147483648
  let w_20_0 : Int := (w_18_0 + w_19_0)
  let w_21_0 : Int := 4294967296
  let w_22_0 : Int := (w_20_0 / w_21_0)
  let w_23_0 : Int := 2048
  let w_24_0 : Int := (w_22_0 % w_23_0)
  let w_25_0 : Int := (w_3_0 - w_24_0)
  let w_26_0 : Int := 2048
  let w_27_0 : Int := (w_25_0 % w_26_0)
  let w_28_0 : Mxx.Primitives.ExactMatrix 5234636801 1024 1 1 := MxxRuntime.multiplyMonomial w_2_0 w_27_0
  let w_31_0 := witness.w_31_0
  let w_32_0 := witness.w_32_0
  let w_35_0 := witness.w_35_0
  let w_35_1 := witness.w_35_1
  let w_36_0 := witness.w_36_0
  let w_37_0 : Int := 4294967296
  let w_38_0 := witness.w_38_0
  let w_39_0 : Int := 1
  let w_40_0 : Int := 4
  let w_41_0 : Int := 16
  let w_42_0 : Int := 64
  let w_43_0 : Int := 256
  let w_44_0 : Int := 1024
  let w_45_0 : Int := 4096
  let w_46_0 : Int := 16384
  let w_47_0 := witness.w_47_0
  let w_48_0 : Fin 630 → Int := MxxRuntime.intMatrixVectorProduct true w_0_0 w_47_0
  let w_49_0 : Int := 4294967296
  let w_50_0 := witness.w_50_0
  let w_51_0 := witness.w_51_0
  let w_52_0 := witness.w_52_0
  let w_53_0 : Int := 5234636801
  let w_54_0 : Int := (w_52_0 % w_53_0)
  let w_55_0 : Int := 4294967296
  let w_56_0 : Int := (w_54_0 * w_55_0)
  let w_57_0 : Int := 2617318400
  let w_58_0 : Int := (w_56_0 + w_57_0)
  let w_59_0 : Int := 5234636801
  let w_60_0 : Int := (w_58_0 / w_59_0)
  let w_61_0 : Int := (w_60_0 % w_37_0)
  let w_63_0 : Fin 1 → Int := MxxRuntime.intMatrixVectorProduct true w_62_0 w_47_0
  let w_64_0 := witness.w_64_0
  let w_65_0 : Int := (w_61_0 - w_64_0)
  let w_66_0 : Int := (w_65_0 % w_49_0)
  generatedRoot.constraints_0 backend tape path params w_29_0 w_30_0 w_33_0 w_34_0 w_1_0 w_8_0 w_11_0 w_15_0 w_21_0 w_23_0 w_26_0 w_28_0 w_31_0 w_32_0 w_35_0 w_35_1 w_36_0 w_37_0 w_38_0 w_39_0 w_40_0 w_41_0 w_42_0 w_43_0 w_44_0 w_45_0 w_46_0 w_47_0 w_48_0 w_49_0 w_50_0 w_51_0 w_52_0 w_53_0 w_59_0 w_63_0 w_64_0 w_66_0 outputs

def generatedRoot (backend : MxxRuntime.BackendContext) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (inputs : (Fin 5160960 → Int) × Int × Int × (Fin 630 → Int) × (Fin 630 → Int) × (Fin 630 → Mxx.Primitives.ExactMatrix 5234636801 1024 1 12) × (Fin 630 → Mxx.Primitives.ExactMatrix 5234636801 1024 1 12) × (Fin 8192 → Int) × Unit) (outputs : (Fin 630 → Int) × Int × Unit) : Prop :=
  ∃ witness : generatedRoot.Witness,
    generatedRoot.body backend tape path params inputs outputs witness


end Stage_nand
