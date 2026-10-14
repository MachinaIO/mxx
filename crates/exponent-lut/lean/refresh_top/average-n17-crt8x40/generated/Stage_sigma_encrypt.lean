import MxxIR
import MxxRuntime

set_option maxRecDepth 16384
set_option maxHeartbeats 2000000

namespace Stage_sigma_encrypt

structure Params where
  unit : Unit

local notation "mxxNat0" => 58939296492691138878702773516453407894262516600999710477297696781780197428886147183580532930842779245363751414241558374714389584615895775645170098790957301481710406198150777982291393965342211790429209140300582802320489682732584091465620426149778680085678654673775272619957085140939019627938941188568589418040655422102526426364406009365202076927674220764118972834074665636427689815578523772328795531861434460551114477692852200267973892039145338281935459716928817045069623711041256489953177312097154832269467014174343169
local notation "mxxNat1" => 53611356076003898413213065075597162215754334196546171573119847117485967770603342141865710594970714126436975754319627813722360382223771204493444257953152133185581399438071429528888721158426124878908768631739375612998825631828962215329564210352462464652101278868494205117710292217630936246817238130188252229743312974787821112749889617659713757824353460415063287202874104835850195903282250326372433858169114185278130346940276065173492686123846178009010257062390029214606904319295847429618832278950170460684289
local notation "mxxNat2" => 48764999414468870004284786596132215334791253665407392213082376954163828662227469646783002495440071126480869572992061373463936942554766993076566775382939094085676568434038710091995898213864265673287200274882137337694092340972812009131390058615139651139540328066628908784089153378793885815792305309627034079947865027868062173311727808255284644763712149466633415361389358591839538931541735542321735270131526579157493250028741077951533512361876653298515986594441047363218654163044177691198952308737
local notation "mxxNat3" => 22849706627100411279805053781123438698169391302742684849719104667188008502058309689050316536705688625878338989141265536630661955351795235935179221031938693603967556815149715824757179268754537685751702762869283827625153106965932116101709342637883606957203067672483050460698051749173490309062706278803086068157824735058705067159681224669660763775735423320310634814252166524151227697567329196937576449
local notation "mxxNat4" => 20783565130672126330897984915270246305190718191134759245162164538239682133685594496549997172715885125225153665826597806359325755457451809047912810881894345428066821779034879929070437569028761057244809996653054486132501254138989274082620642513088966386219569386609692905229299422932494659524585399265843564376603106981871602858286289822573810649329058455160048185531671309436090991771649
local notation "mxxNat5" => 18904232540379663606298671617222323272326738038390029396917824777746969582045167858404151562874829160926414758198062640031724523819329642633677342252406968964798152526469533620526817961691577118545119336836976607577612946619880114513560241379804216574981139379779325820613839568974862527139956458102788878455983699609552819892163294046567535706628946510510275803584369524737
local notation "mxxNat6" => 17194779275841340066094911582660118468150022575780817946956761424016171971975644626612993759657783559033812747229612023541500953633995164564387606070818297194210498119397913119070984614359409232233058467631090401698164636312143628944282056978346654853962441965445507118894925452267578916248174069761705821782719146070044905207880195296125044373224260617321054209
local notation "mxxNat7" => 15639891845680862471273391501028538700541437717538409679095559405445804528070626983357193625897380999208378607381195148167160916902901604581105204366284610038309228524796037314276898909933691568859596974598220317428101056377947215041991765519754580648561888623368771509456317222487602091898587287192370009158658697626230010316495535089043219399311361
local notation "mxxNat8" => 14225599413043849874313110955476320826847714068103435018255579625040844987614592814921216522176466278368146264280922420657454902124234157630410932078675892426402796415678081679458699869279729102080225575172667164578333615322351009590667667588488650942207148756107026898955916986009550662438661825954286719355032762293978803607517001154561
local notation "mxxNat9" => 12939193196003583247700745219127459276538103829878228255452964825429565744738496943340136285757373009365008027512352428281163371523086712086335462845897380008297825183853731232340529655567580283203101783098765504342130209377805468698205625341896587740691110471279395485602344847745227903941887898078012727529355788540377038849
local notation "mxxNat10" => 11769053624114879662112160167819918571087756078475517226025140209950000697438445616103855951934845758042851220004268835280083723078935285065206709652167357816684063235671681742884779135887118938377642504598868343721508787284556232775611560993407551544197361730665816986991555988904817833153884745654587506510790657
local notation "mxxNat11" => 10704718830052645159558437338680613290428270391588260726941401141685843313697367701755877287966105991387779044078883002840225637249711548487927800644301887928350976281363775850073673132136882262680409334771844414563541630439900275818010930920182987500791214865537269603350940622377042848318639613935617
local notation "mxxNat12" => 9736632548792324873927854075713791405494493690291289930484105488647225447794833419707664394946183376006772118405827319456147558326407337915145682114769134320430781498751623334684336078475478387746643054210897714560839787891586099028101663247881380430122397333390018389907135708925263347713
local notation "mxxNat13" => 8856063931433349353777749199840300423201273833919512788866693017246937325932836637550898012786540038926285538828926565061772997670687700554247604366721692215549678434755586777820506522592266645331915489522880346391132557219398623724334598008315111111510902908700756335271346177
local notation "mxxNat14" => 44356584807662390582974036183718532592948938341515613905271330982211363991910147909785928777943624044679453441181946633266767017576573299092915986421441193664382504781582965701086753303911852745815725937621403943838130197376954354157239211429065130811017205011948503532732543592298255862727045349566508849228074814484020667870885329977778176618133725817440073479402205553934479588944792292455106586694802429764522075146408317146864987968842541117659987345168921017335784298220093441
local notation "mxxNat15" => 8055080958861527156612166159716893751603793734049033618335598088738794598397393641001068374575492900107506707131983323245369217953846867605177044540907874818104518793379941017137176073249795397424141950359933660080606670194087562361642881104810188164043751152680961
local notation "mxxNat16" => 7326532072524307707736124693260693059380887668417833042585390925513960938386451154286953732300195746996919587908289417507723765261060713789310297311903088628393866898092564725098602757961958364722064196042521885373766844705011047636563943576968163491841
local notation "mxxNat17" => 6663863131350690869489159208043323566981896159568684606109881332596928508518424264401057919741590526470213548611216807099538371695864300991355765901857308226933207608673095853646724326006066368768674120411140203584000765149558518557124329473
local notation "mxxNat18" => 6061122488108641973420157184386247391232670344957471793731902779915874104395629789434972185573721480099586866973302753874247151633542098410564240286537836798644378563363486222202490534664914003616194598189542092777905816942149633
local notation "mxxNat19" => 5512867666694373219115871970903288029215211960191862388120690159144039078598177602465227991599035013137010102790205942412194741109497094431335932405884417480548526382712638318441099737268915905780303389446013788356609
local notation "mxxNat20" => 5014201285498287044571829504092395859170014061982570370064233881086809855983593086449481212078216073968919420320109342876444280020212232236090324687438347215264479417937802431763227897342933251975830568961
local notation "mxxNat21" => 4560640678659212200413092730406177900813074834916767428238891371420899921997321429202131260369104208213068131427321389208876415881159494552801258246307209689917770935612845481671273649224810497
local notation "mxxNat22" => 4148098088297657826837312901609192179974413451366515355467911990942123970073947713581974860453445204725757011898563957078540577681475412311926879316781507522205850799831022751711233
local notation "mxxNat23" => 3772870229134357736045297237853950864238360940177929599688916703412106100584926896283922827652492396242283864160111436141630884689863793038510772437184293348033193050113
local notation "mxxNat24" => 3431582204909317882500015710448701810725839055401161206051444022296236030028164797979279485468090427211570548863247937663363459158370600335115995717139169281
local notation "mxxNat25" => 40346609599881356220837572389148745492707889782157253985114074622870119635384542256138256733871534125776362551190437393425805460665682516605934793689054934288066150602580962837156609210318793863796292818736937592444729847837652128872220271346655239817773261191870912264956811067225654055923213873834449562352202400123154847258845594914024823959485374540059295492961815622998264934217811307699517460289652992967872013240202921822953204208082202860547685542392991864324097
local notation "mxxNat26" => 3121147220562692655871736294535619050331812260122680288780092738349874994418875705303353159054301821854920518431232183649214520770396688421486593
local notation "mxxNat27" => 2838775151230559413418818746480280063980232358065862815445807092727729241399019106007725155181821623092536294263146717901932973785089
local notation "mxxNat28" => 2581927291416572864619368372858661854095731200741641349970442421348128330067118131310025446354327075029741340727792631809
local notation "mxxNat29" => 2348314698081035712001987778321114647878826645963489881374862414216186885241767220677921135674985115419934721
local notation "mxxNat30" => 2135831716638487324604933616719561441822403870837824617769981942184435508314978270190567457292289
local notation "mxxNat31" => 36699131300532040202874628738679301745518768276247572676519885810347852166710508198556317553594263393881423524846705965941306027447195207701038301786978650701994670692271368316959997088574206884048956363261104691724308307125262752936938046346790725509939983486181983675166625920850163171862018995585496653690398557863058658278827413168537073757020849305611889872365768236786446005228456427241498072220878073004066452310651408739976654318283976266949824086017
local notation "mxxNat32" => 33381302613763707714135659081702312941925506087577779744270390358422997362703355583611015868571841781788422225821881454754441155027925933889062895360669899811348641011021937656518288360359063741576299345796092247689499077123972604540572587453823129310188487679751815285677139119780827117968376090423801396942470634017006035025170844051714535686565063235939758897934886926067297909673513293061677173305121261440307169006263462037641070806150152193
local notation "mxxNat33" => 30363353782355777582384487781334120088284938185520381890151379293848957055739265559829706084500606745584608702056286710988677995620438714513014119398057034375242069238015528072008609331178770145968039498528670570712762270552862174661072859407755959856057359535225700630920529199192209743000769240318846450399043660992504408833190807605159643093718012042290284451525514829476242284008230606321999771457299176261596538145946630967525377
local notation "mxxNat34" => 27618200072880679526437612227137758925486614570755974628973663818473313316718148748077679186040510556794019871056463700519941291164245344254403798419188869692953458979003048081968323154250241619763697119930042500271788581642850440935834777529629104057541246958715714472583956417189444392667294168516056463817138447505264164967403500518589294624534337729242348361371473145183817822873266337958912739627935059321466132627457
local notation "mxxNat35" => 25121086222911491376137526004410590836769979497088927555272808019255531684072937530321622927622024410743315114289554001729959736229342837692089099576636198916589402880010211193953045204278395560167945164738032236537231331765715592871763384219377950815511978161075119752391877135014765897903977948537270828236374669272087592998962253286129875243636685957149695068415190242022213866695200245861582071256866881537

/-- The modulus of a matrix type. -/
abbrev modulus_0 : Nat := mxxNat0

/-- The modulus of a matrix type. -/
abbrev modulus_1 : Nat := mxxNat1

/-- The modulus of a matrix type. -/
abbrev modulus_2 : Nat := mxxNat2

/-- The modulus of a matrix type. -/
abbrev modulus_3 : Nat := mxxNat3

/-- The modulus of a matrix type. -/
abbrev modulus_4 : Nat := mxxNat4

/-- The modulus of a matrix type. -/
abbrev modulus_5 : Nat := mxxNat5

/-- The modulus of a matrix type. -/
abbrev modulus_6 : Nat := mxxNat6

/-- The modulus of a matrix type. -/
abbrev modulus_7 : Nat := mxxNat7

/-- The modulus of a matrix type. -/
abbrev modulus_8 : Nat := mxxNat8

/-- The modulus of a matrix type. -/
abbrev modulus_9 : Nat := mxxNat9

/-- The modulus of a matrix type. -/
abbrev modulus_10 : Nat := mxxNat10

/-- The modulus of a matrix type. -/
abbrev modulus_11 : Nat := mxxNat11

/-- The modulus of a matrix type. -/
abbrev modulus_12 : Nat := mxxNat12

/-- The modulus of a matrix type. -/
abbrev modulus_13 : Nat := mxxNat13

/-- The modulus of a matrix type. -/
abbrev modulus_14 : Nat := mxxNat14

/-- The modulus of a matrix type. -/
abbrev modulus_15 : Nat := mxxNat15

/-- The modulus of a matrix type. -/
abbrev modulus_16 : Nat := mxxNat16

/-- The modulus of a matrix type. -/
abbrev modulus_17 : Nat := mxxNat17

/-- The modulus of a matrix type. -/
abbrev modulus_18 : Nat := mxxNat18

/-- The modulus of a matrix type. -/
abbrev modulus_19 : Nat := mxxNat19

/-- The modulus of a matrix type. -/
abbrev modulus_20 : Nat := mxxNat20

/-- The modulus of a matrix type. -/
abbrev modulus_21 : Nat := mxxNat21

/-- The modulus of a matrix type. -/
abbrev modulus_22 : Nat := mxxNat22

/-- The modulus of a matrix type. -/
abbrev modulus_23 : Nat := mxxNat23

/-- The modulus of a matrix type. -/
abbrev modulus_24 : Nat := mxxNat24

/-- The modulus of a matrix type. -/
abbrev modulus_25 : Nat := mxxNat25

/-- The modulus of a matrix type. -/
abbrev modulus_26 : Nat := mxxNat26

/-- The modulus of a matrix type. -/
abbrev modulus_27 : Nat := mxxNat27

/-- The modulus of a matrix type. -/
abbrev modulus_28 : Nat := mxxNat28

/-- The modulus of a matrix type. -/
abbrev modulus_29 : Nat := mxxNat29

/-- The modulus of a matrix type. -/
abbrev modulus_30 : Nat := mxxNat30

/-- The modulus of a matrix type. -/
abbrev modulus_31 : Nat := mxxNat31

/-- The modulus of a matrix type. -/
abbrev modulus_32 : Nat := mxxNat32

/-- The modulus of a matrix type. -/
abbrev modulus_33 : Nat := mxxNat33

/-- The modulus of a matrix type. -/
abbrev modulus_34 : Nat := mxxNat34

/-- The modulus of a matrix type. -/
abbrev modulus_35 : Nat := mxxNat35

local notation "mxxMat0" => Mxx.Primitives.ExactMatrix modulus_0 131072 1 172
local notation "mxxMat1" => Mxx.Primitives.ExactMatrix modulus_0 131072 1 1
local notation "mxxMat2" => Mxx.Primitives.ExactMatrix modulus_0 131072 1 168
local notation "mxxMat3" => Mxx.Primitives.ExactMatrix modulus_1 131072 1 168
local notation "mxxMat4" => Mxx.Primitives.ExactMatrix modulus_1 131072 1 1
local notation "mxxMat5" => Mxx.Primitives.ExactMatrix modulus_1 131072 1 164
local notation "mxxMat6" => Mxx.Primitives.ExactMatrix modulus_2 131072 1 164
local notation "mxxMat7" => Mxx.Primitives.ExactMatrix modulus_3 131072 1 1
local notation "mxxMat8" => Mxx.Primitives.ExactMatrix modulus_3 131072 1 128
local notation "mxxMat9" => Mxx.Primitives.ExactMatrix modulus_4 131072 1 128
local notation "mxxMat10" => Mxx.Primitives.ExactMatrix modulus_4 131072 1 1
local notation "mxxMat11" => Mxx.Primitives.ExactMatrix modulus_4 131072 1 124
local notation "mxxMat12" => Mxx.Primitives.ExactMatrix modulus_5 131072 1 124
local notation "mxxMat13" => Mxx.Primitives.ExactMatrix modulus_5 131072 1 1
local notation "mxxMat14" => Mxx.Primitives.ExactMatrix modulus_5 131072 1 120
local notation "mxxMat15" => Mxx.Primitives.ExactMatrix modulus_6 131072 1 120
local notation "mxxMat16" => Mxx.Primitives.ExactMatrix modulus_6 131072 1 1
local notation "mxxMat17" => Mxx.Primitives.ExactMatrix modulus_6 131072 1 116
local notation "mxxMat18" => Mxx.Primitives.ExactMatrix modulus_7 131072 1 116
local notation "mxxMat19" => Mxx.Primitives.ExactMatrix modulus_7 131072 1 1
local notation "mxxMat20" => Mxx.Primitives.ExactMatrix modulus_7 131072 1 112
local notation "mxxMat21" => Mxx.Primitives.ExactMatrix modulus_8 131072 1 112
local notation "mxxMat22" => Mxx.Primitives.ExactMatrix modulus_8 131072 1 1
local notation "mxxMat23" => Mxx.Primitives.ExactMatrix modulus_8 131072 1 108
local notation "mxxMat24" => Mxx.Primitives.ExactMatrix modulus_9 131072 1 108
local notation "mxxMat25" => Mxx.Primitives.ExactMatrix modulus_9 131072 1 1
local notation "mxxMat26" => Mxx.Primitives.ExactMatrix modulus_9 131072 1 104
local notation "mxxMat27" => Mxx.Primitives.ExactMatrix modulus_10 131072 1 104
local notation "mxxMat28" => Mxx.Primitives.ExactMatrix modulus_10 131072 1 1
local notation "mxxMat29" => Mxx.Primitives.ExactMatrix modulus_10 131072 1 100
local notation "mxxMat30" => Mxx.Primitives.ExactMatrix modulus_11 131072 1 100
local notation "mxxMat31" => Mxx.Primitives.ExactMatrix modulus_11 131072 1 1
local notation "mxxMat32" => Mxx.Primitives.ExactMatrix modulus_11 131072 1 96
local notation "mxxMat33" => Mxx.Primitives.ExactMatrix modulus_12 131072 1 96
local notation "mxxMat34" => Mxx.Primitives.ExactMatrix modulus_12 131072 1 1
local notation "mxxMat35" => Mxx.Primitives.ExactMatrix modulus_12 131072 1 92
local notation "mxxMat36" => Mxx.Primitives.ExactMatrix modulus_13 131072 1 92
local notation "mxxMat37" => Mxx.Primitives.ExactMatrix modulus_2 131072 1 1
local notation "mxxMat38" => Mxx.Primitives.ExactMatrix modulus_2 131072 1 160
local notation "mxxMat39" => Mxx.Primitives.ExactMatrix modulus_14 131072 1 160
local notation "mxxMat40" => Mxx.Primitives.ExactMatrix modulus_13 131072 1 1
local notation "mxxMat41" => Mxx.Primitives.ExactMatrix modulus_13 131072 1 88
local notation "mxxMat42" => Mxx.Primitives.ExactMatrix modulus_15 131072 1 88
local notation "mxxMat43" => Mxx.Primitives.ExactMatrix modulus_15 131072 1 1
local notation "mxxMat44" => Mxx.Primitives.ExactMatrix modulus_15 131072 1 84
local notation "mxxMat45" => Mxx.Primitives.ExactMatrix modulus_16 131072 1 84
local notation "mxxMat46" => Mxx.Primitives.ExactMatrix modulus_16 131072 1 1
local notation "mxxMat47" => Mxx.Primitives.ExactMatrix modulus_16 131072 1 80
local notation "mxxMat48" => Mxx.Primitives.ExactMatrix modulus_17 131072 1 80
local notation "mxxMat49" => Mxx.Primitives.ExactMatrix modulus_17 131072 1 1
local notation "mxxMat50" => Mxx.Primitives.ExactMatrix modulus_17 131072 1 76
local notation "mxxMat51" => Mxx.Primitives.ExactMatrix modulus_18 131072 1 76
local notation "mxxMat52" => Mxx.Primitives.ExactMatrix modulus_18 131072 1 1
local notation "mxxMat53" => Mxx.Primitives.ExactMatrix modulus_18 131072 1 72
local notation "mxxMat54" => Mxx.Primitives.ExactMatrix modulus_19 131072 1 72
local notation "mxxMat55" => Mxx.Primitives.ExactMatrix modulus_19 131072 1 1
local notation "mxxMat56" => Mxx.Primitives.ExactMatrix modulus_19 131072 1 68
local notation "mxxMat57" => Mxx.Primitives.ExactMatrix modulus_20 131072 1 68
local notation "mxxMat58" => Mxx.Primitives.ExactMatrix modulus_20 131072 1 1
local notation "mxxMat59" => Mxx.Primitives.ExactMatrix modulus_20 131072 1 64
local notation "mxxMat60" => Mxx.Primitives.ExactMatrix modulus_21 131072 1 64
local notation "mxxMat61" => Mxx.Primitives.ExactMatrix modulus_21 131072 1 1
local notation "mxxMat62" => Mxx.Primitives.ExactMatrix modulus_21 131072 1 60
local notation "mxxMat63" => Mxx.Primitives.ExactMatrix modulus_22 131072 1 60
local notation "mxxMat64" => Mxx.Primitives.ExactMatrix modulus_22 131072 1 1
local notation "mxxMat65" => Mxx.Primitives.ExactMatrix modulus_22 131072 1 56
local notation "mxxMat66" => Mxx.Primitives.ExactMatrix modulus_23 131072 1 56
local notation "mxxMat67" => Mxx.Primitives.ExactMatrix modulus_23 131072 1 1
local notation "mxxMat68" => Mxx.Primitives.ExactMatrix modulus_23 131072 1 52
local notation "mxxMat69" => Mxx.Primitives.ExactMatrix modulus_24 131072 1 52
local notation "mxxMat70" => Mxx.Primitives.ExactMatrix modulus_14 131072 1 1
local notation "mxxMat71" => Mxx.Primitives.ExactMatrix modulus_14 131072 1 156
local notation "mxxMat72" => Mxx.Primitives.ExactMatrix modulus_25 131072 1 156
local notation "mxxMat73" => Mxx.Primitives.ExactMatrix modulus_24 131072 1 1
local notation "mxxMat74" => Mxx.Primitives.ExactMatrix modulus_24 131072 1 48
local notation "mxxMat75" => Mxx.Primitives.ExactMatrix modulus_26 131072 1 48
local notation "mxxMat76" => Mxx.Primitives.ExactMatrix modulus_26 131072 1 1
local notation "mxxMat77" => Mxx.Primitives.ExactMatrix modulus_26 131072 1 44
local notation "mxxMat78" => Mxx.Primitives.ExactMatrix modulus_27 131072 1 44
local notation "mxxMat79" => Mxx.Primitives.ExactMatrix modulus_27 131072 1 1
local notation "mxxMat80" => Mxx.Primitives.ExactMatrix modulus_27 131072 1 40
local notation "mxxMat81" => Mxx.Primitives.ExactMatrix modulus_28 131072 1 40
local notation "mxxMat82" => Mxx.Primitives.ExactMatrix modulus_28 131072 1 1
local notation "mxxMat83" => Mxx.Primitives.ExactMatrix modulus_28 131072 1 36
local notation "mxxMat84" => Mxx.Primitives.ExactMatrix modulus_29 131072 1 36
local notation "mxxMat85" => Mxx.Primitives.ExactMatrix modulus_29 131072 1 1
local notation "mxxMat86" => Mxx.Primitives.ExactMatrix modulus_29 131072 1 32
local notation "mxxMat87" => Mxx.Primitives.ExactMatrix modulus_30 131072 1 32
local notation "mxxMat88" => Mxx.Primitives.ExactMatrix modulus_30 131072 1 1
local notation "mxxMat89" => Mxx.Primitives.ExactMatrix modulus_30 131072 1 28
local notation "mxxMat90" => Mxx.Primitives.ExactMatrix modulus_25 131072 1 1
local notation "mxxMat91" => Mxx.Primitives.ExactMatrix modulus_25 131072 1 152
local notation "mxxMat92" => Mxx.Primitives.ExactMatrix modulus_31 131072 1 152
local notation "mxxMat93" => Mxx.Primitives.ExactMatrix modulus_31 131072 1 1
local notation "mxxMat94" => Mxx.Primitives.ExactMatrix modulus_31 131072 1 148
local notation "mxxMat95" => Mxx.Primitives.ExactMatrix modulus_32 131072 1 148
local notation "mxxMat96" => Mxx.Primitives.ExactMatrix modulus_32 131072 1 1
local notation "mxxMat97" => Mxx.Primitives.ExactMatrix modulus_32 131072 1 144
local notation "mxxMat98" => Mxx.Primitives.ExactMatrix modulus_33 131072 1 144
local notation "mxxMat99" => Mxx.Primitives.ExactMatrix modulus_33 131072 1 1
local notation "mxxMat100" => Mxx.Primitives.ExactMatrix modulus_33 131072 1 140
local notation "mxxMat101" => Mxx.Primitives.ExactMatrix modulus_34 131072 1 140
local notation "mxxMat102" => Mxx.Primitives.ExactMatrix modulus_34 131072 1 1
local notation "mxxMat103" => Mxx.Primitives.ExactMatrix modulus_34 131072 1 136
local notation "mxxMat104" => Mxx.Primitives.ExactMatrix modulus_35 131072 1 136
local notation "mxxMat105" => Mxx.Primitives.ExactMatrix modulus_35 131072 1 1
local notation "mxxMat106" => Mxx.Primitives.ExactMatrix modulus_35 131072 1 132
local notation "mxxMat107" => Mxx.Primitives.ExactMatrix modulus_3 131072 1 132

abbrev parallel_parallel_generatedRoot_7_39.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_1 : Nat) (w_1_0 : ByteArray) (w_2_0 : mxxMat0) (w_5_0 : mxxMat0) (w_11_0 : mxxMat1) (w_12_0 : mxxMat0) (w_16_0 : mxxMat0) (w_17_0 : mxxMat0) (w_18_0 : mxxMat0) (w_20_0 : mxxMat0) (w_25_0 : mxxMat0) (w_26_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × mxxMat1 × Unit) : Prop :=
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 105, 110, 112, 117, 116, 47, 87, 105, 114, 101, 73, 100, 40, 49, 41, 47, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le ((Int.ofNat i_1)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_2_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 105, 110, 112, 117, 116, 47, 87, 105, 114, 101, 73, 100, 40, 49, 41, 47, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le ((Int.ofNat i_1)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_5_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 1024 172 w_12_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [16]) ((4/1)) (26) w_16_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 105, 110, 112, 117, 116, 47, 87, 105, 114, 101, 73, 100, 40, 49, 41, 47, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le ((Int.ofNat i_1)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_18_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 105, 110, 112, 117, 116, 47, 87, 105, 114, 101, 73, 100, 40, 49, 41, 47, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le ((Int.ofNat i_1)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_20_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [25]) ((4/1)) (26) w_25_0 ∧
  outputs = (w_17_0, (w_26_0, (w_11_0, ())))

def parallel_parallel_generatedRoot_7_39 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat1 × ByteArray × mxxMat1 × Int × Int × Unit) (outputs : mxxMat0 × mxxMat0 × mxxMat1 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_4_0 : mxxMat1 := inputs.2.2.1
  let w_8_0 : Int := inputs.2.2.2.1
  let w_10_0 : Int := inputs.2.2.2.2.1
  ∃ (w_2_0 : mxxMat0),
    let w_3_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_0_0 w_2_0
    ∃ (w_5_0 : mxxMat0),
      let w_6_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_4_0 w_5_0
      let w_7_0 : mxxMat0 := MxxRuntime.matrixAdd w_3_0 w_6_0
      let w_9_0 : mxxMat1 := MxxRuntime.liftInteger w_8_0
      let w_11_0 : mxxMat1 := MxxRuntime.multiplyMonomial w_9_0 w_10_0
      ∃ (w_12_0 : mxxMat0),
        let w_13_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_11_0 w_12_0
        let w_14_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_0_0 w_13_0
        let w_15_0 : mxxMat0 := MxxRuntime.matrixSub w_7_0 w_14_0
        ∃ (w_16_0 : mxxMat0),
          let w_17_0 : mxxMat0 := MxxRuntime.matrixAdd w_15_0 w_16_0
          ∃ (w_18_0 : mxxMat0),
            let w_19_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_0_0 w_18_0
            ∃ (w_20_0 : mxxMat0),
              let w_21_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_4_0 w_20_0
              let w_22_0 : mxxMat0 := MxxRuntime.matrixAdd w_19_0 w_21_0
              let w_23_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_4_0 w_13_0
              let w_24_0 : mxxMat0 := MxxRuntime.matrixSub w_22_0 w_23_0
              ∃ (w_25_0 : mxxMat0),
                let w_26_0 : mxxMat0 := MxxRuntime.matrixAdd w_24_0 w_25_0
                parallel_parallel_generatedRoot_7_39.constraints_0 backend hashModel tape path i_1 w_1_0 w_2_0 w_5_0 w_11_0 w_12_0 w_16_0 w_17_0 w_18_0 w_20_0 w_25_0 w_26_0 outputs

abbrev parallel_parallel_generatedRoot_7_40.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_1 : Nat) (w_0_0 : mxxMat1) (w_1_0 : ByteArray) (w_5_0 : mxxMat1) (w_2_0 : Int) (w_3_0 : mxxMat2) (w_6_0 : mxxMat2) (w_10_0 : mxxMat1) (w_12_0 : mxxMat3) (w_13_0 : mxxMat2) (w_16_0 : mxxMat2) (w_17_0 : mxxMat2) (w_18_0 : mxxMat2) (w_20_0 : mxxMat2) (w_23_0 : mxxMat1) (w_27_0 : mxxMat2) (w_28_0 : mxxMat2) (outputs : mxxMat2 × mxxMat2 × Unit) : Prop :=
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 128 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (0), .integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (0), .integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_6_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 128 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (65), (85), (99), (129), (135), (153), (165), (195), (255), (297), (325), (387), (459), (495), (561), (585), (645), (765), (891), (975), (1105), (1161), (1485), (1683), (1755), (1935), (2145), (2193), (2295), (2805), (2925), (3315), (3483), (4257), (4455), (5049), (5525), (5805), (6435), (6579), (8385), (8415), (8775), (9945), (10725), (10965), (12771), (14561), (15147), (16575), (17415), (19305), (19737), (21285), (22687), (25155), (25245), (27431), (29835), (32175), (32895), (33911), (36465), (36959), (38313), (41023), (41925), (43039), (43683), (49725), (50407), (57915), (59211), (63855), (66041), (68061), (70487), (72369), (72805), (75465), (75735), (83399), (96525), (98685), (99701), (107063), (109395), (110877), (114939), (115181), (118955), (123069), (125207), (125775), (127033), (129117), (129899), (130681), (130727), (130957), (131003), (131049), (142545), (149175), (165491), (178699), (182325), (188437), (189109), (191565), (198123), (204183), (217107), (218323), (218415), (226395), (234329), (247537)])) w_0_0 w_10_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 1024 168 w_12_0 ∧
  MxxRuntime.centeredRebaseRuns w_12_0 w_13_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [16]) ((4/1)) (26) w_16_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (0), .integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_18_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (0), .integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_20_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 128 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (65), (85), (99), (129), (135), (153), (165), (195), (255), (297), (325), (387), (459), (495), (561), (585), (645), (765), (891), (975), (1105), (1161), (1485), (1683), (1755), (1935), (2145), (2193), (2295), (2805), (2925), (3315), (3483), (4257), (4455), (5049), (5525), (5805), (6435), (6579), (8385), (8415), (8775), (9945), (10725), (10965), (12771), (14561), (15147), (16575), (17415), (19305), (19737), (21285), (22687), (25155), (25245), (27431), (29835), (32175), (32895), (33911), (36465), (36959), (38313), (41023), (41925), (43039), (43683), (49725), (50407), (57915), (59211), (63855), (66041), (68061), (70487), (72369), (72805), (75465), (75735), (83399), (96525), (98685), (99701), (107063), (109395), (110877), (114939), (115181), (118955), (123069), (125207), (125775), (127033), (129117), (129899), (130681), (130727), (130957), (131003), (131049), (142545), (149175), (165491), (178699), (182325), (188437), (189109), (191565), (198123), (204183), (217107), (218323), (218415), (226395), (234329), (247537)])) w_5_0 w_23_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [27]) ((4/1)) (26) w_27_0 ∧
  outputs = (w_17_0, (w_28_0, ()))

def parallel_parallel_generatedRoot_7_40 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat1 × ByteArray × mxxMat1 × Unit) (outputs : mxxMat2 × mxxMat2 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat1 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (65), (85), (99), (129), (135), (153), (165), (195), (255), (297), (325), (387), (459), (495), (561), (585), (645), (765), (891), (975), (1105), (1161), (1485), (1683), (1755), (1935), (2145), (2193), (2295), (2805), (2925), (3315), (3483), (4257), (4455), (5049), (5525), (5805), (6435), (6579), (8385), (8415), (8775), (9945), (10725), (10965), (12771), (14561), (15147), (16575), (17415), (19305), (19737), (21285), (22687), (25155), (25245), (27431), (29835), (32175), (32895), (33911), (36465), (36959), (38313), (41023), (41925), (43039), (43683), (49725), (50407), (57915), (59211), (63855), (66041), (68061), (70487), (72369), (72805), (75465), (75735), (83399), (96525), (98685), (99701), (107063), (109395), (110877), (114939), (115181), (118955), (123069), (125207), (125775), (127033), (129117), (129899), (130681), (130727), (130957), (131003), (131049), (142545), (149175), (165491), (178699), (182325), (188437), (189109), (191565), (198123), (204183), (217107), (218323), (218415), (226395), (234329), (247537)])
  ∃ (w_3_0 : mxxMat2),
    let w_4_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat2),
      let w_7_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat2 := MxxRuntime.matrixAdd w_4_0 w_7_0
      let w_9_0 : mxxMat1 := (MxxRuntime.matrixPolynomial [(1099380817921)] : mxxMat1)
      ∃ (w_10_0 : mxxMat1),
        let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
        ∃ (w_12_0 : mxxMat3),
          ∃ (w_13_0 : mxxMat2),
            let w_14_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_11_0 w_13_0
            let w_15_0 : mxxMat2 := MxxRuntime.matrixSub w_8_0 w_14_0
            ∃ (w_16_0 : mxxMat2),
              let w_17_0 : mxxMat2 := MxxRuntime.matrixAdd w_15_0 w_16_0
              ∃ (w_18_0 : mxxMat2),
                let w_19_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_0_0 w_18_0
                ∃ (w_20_0 : mxxMat2),
                  let w_21_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_5_0 w_20_0
                  let w_22_0 : mxxMat2 := MxxRuntime.matrixAdd w_19_0 w_21_0
                  ∃ (w_23_0 : mxxMat1),
                    let w_24_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_23_0
                    let w_25_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_24_0 w_13_0
                    let w_26_0 : mxxMat2 := MxxRuntime.matrixSub w_22_0 w_25_0
                    ∃ (w_27_0 : mxxMat2),
                      let w_28_0 : mxxMat2 := MxxRuntime.matrixAdd w_26_0 w_27_0
                      parallel_parallel_generatedRoot_7_40.constraints_0 backend hashModel tape path i_1 w_0_0 w_1_0 w_5_0 w_2_0 w_3_0 w_6_0 w_10_0 w_12_0 w_13_0 w_16_0 w_17_0 w_18_0 w_20_0 w_23_0 w_27_0 w_28_0 outputs

abbrev parallel_parallel_generatedRoot_7_41.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_1 : Nat) (w_0_0 : mxxMat1) (w_1_0 : ByteArray) (w_5_0 : mxxMat1) (w_2_0 : Int) (w_3_0 : mxxMat2) (w_6_0 : mxxMat2) (w_10_0 : mxxMat1) (w_12_0 : mxxMat3) (w_13_0 : mxxMat2) (w_16_0 : mxxMat2) (w_17_0 : mxxMat2) (w_18_0 : mxxMat2) (w_20_0 : mxxMat2) (w_23_0 : mxxMat1) (w_27_0 : mxxMat2) (w_28_0 : mxxMat2) (outputs : mxxMat2 × mxxMat2 × Unit) : Prop :=
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 32 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (0), .integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (0), .integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_6_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 32 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (85), (99), (135), (153), (165), (255), (297), (459), (495), (561), (765), (891), (1485), (1683), (2295), (2805), (4455), (5049), (8415), (15147), (25245), (75735)])) w_0_0 w_10_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 1024 168 w_12_0 ∧
  MxxRuntime.centeredRebaseRuns w_12_0 w_13_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [16]) ((4/1)) (26) w_16_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (0), .integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_18_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (0), .integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_20_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 32 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (85), (99), (135), (153), (165), (255), (297), (459), (495), (561), (765), (891), (1485), (1683), (2295), (2805), (4455), (5049), (8415), (15147), (25245), (75735)])) w_5_0 w_23_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [27]) ((4/1)) (26) w_27_0 ∧
  outputs = (w_17_0, (w_28_0, ()))

def parallel_parallel_generatedRoot_7_41 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat1 × ByteArray × mxxMat1 × Unit) (outputs : mxxMat2 × mxxMat2 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat1 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (85), (99), (135), (153), (165), (255), (297), (459), (495), (561), (765), (891), (1485), (1683), (2295), (2805), (4455), (5049), (8415), (15147), (25245), (75735)])
  ∃ (w_3_0 : mxxMat2),
    let w_4_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat2),
      let w_7_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat2 := MxxRuntime.matrixAdd w_4_0 w_7_0
      let w_9_0 : mxxMat1 := (MxxRuntime.matrixPolynomial [(1099380817921)] : mxxMat1)
      ∃ (w_10_0 : mxxMat1),
        let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
        ∃ (w_12_0 : mxxMat3),
          ∃ (w_13_0 : mxxMat2),
            let w_14_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_11_0 w_13_0
            let w_15_0 : mxxMat2 := MxxRuntime.matrixSub w_8_0 w_14_0
            ∃ (w_16_0 : mxxMat2),
              let w_17_0 : mxxMat2 := MxxRuntime.matrixAdd w_15_0 w_16_0
              ∃ (w_18_0 : mxxMat2),
                let w_19_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_0_0 w_18_0
                ∃ (w_20_0 : mxxMat2),
                  let w_21_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_5_0 w_20_0
                  let w_22_0 : mxxMat2 := MxxRuntime.matrixAdd w_19_0 w_21_0
                  ∃ (w_23_0 : mxxMat1),
                    let w_24_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_23_0
                    let w_25_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_24_0 w_13_0
                    let w_26_0 : mxxMat2 := MxxRuntime.matrixSub w_22_0 w_25_0
                    ∃ (w_27_0 : mxxMat2),
                      let w_28_0 : mxxMat2 := MxxRuntime.matrixAdd w_26_0 w_27_0
                      parallel_parallel_generatedRoot_7_41.constraints_0 backend hashModel tape path i_1 w_0_0 w_1_0 w_5_0 w_2_0 w_3_0 w_6_0 w_10_0 w_12_0 w_13_0 w_16_0 w_17_0 w_18_0 w_20_0 w_23_0 w_27_0 w_28_0 outputs

abbrev parallel_parallel_generatedRoot_7_42.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_1 : Nat) (w_0_0 : mxxMat1) (w_2_0 : ByteArray) (w_6_0 : mxxMat1) (w_1_0 : mxxMat4) (w_3_0 : Int) (w_4_0 : mxxMat5) (w_7_0 : mxxMat4) (w_8_0 : mxxMat5) (w_12_0 : mxxMat4) (w_14_0 : mxxMat6) (w_15_0 : mxxMat5) (w_18_0 : mxxMat5) (w_19_0 : mxxMat5) (w_20_0 : mxxMat5) (w_22_0 : mxxMat5) (w_25_0 : mxxMat4) (w_29_0 : mxxMat5) (w_30_0 : mxxMat5) (outputs : mxxMat5 × mxxMat5 × Unit) : Prop :=
  MxxRuntime.modulusReduceRuns w_0_0 w_1_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 128 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (1), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_2_0) w_4_0 ∧
  MxxRuntime.modulusReduceRuns w_6_0 w_7_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (1), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_2_0) w_8_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 128 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (65), (85), (99), (129), (135), (153), (165), (195), (255), (297), (325), (387), (459), (495), (561), (585), (645), (765), (891), (975), (1105), (1161), (1485), (1683), (1755), (1935), (2145), (2193), (2295), (2805), (2925), (3315), (3483), (4257), (4455), (5049), (5525), (5805), (6435), (6579), (8385), (8415), (8775), (9945), (10725), (10965), (12771), (14561), (15147), (16575), (17415), (19305), (19737), (21285), (22687), (25155), (25245), (27431), (29835), (32175), (32895), (33911), (36465), (36959), (38313), (41023), (41925), (43039), (43683), (49725), (50407), (57915), (59211), (63855), (66041), (68061), (70487), (72369), (72805), (75465), (75735), (83399), (96525), (98685), (99701), (107063), (109395), (110877), (114939), (115181), (118955), (123069), (125207), (125775), (127033), (129117), (129899), (130681), (130727), (130957), (131003), (131049), (142545), (149175), (165491), (178699), (182325), (188437), (189109), (191565), (198123), (204183), (217107), (218323), (218415), (226395), (234329), (247537)])) w_1_0 w_12_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 1024 164 w_14_0 ∧
  MxxRuntime.centeredRebaseRuns w_14_0 w_15_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [18]) ((4/1)) (26) w_18_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (1), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_2_0) w_20_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (1), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_2_0) w_22_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 128 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (65), (85), (99), (129), (135), (153), (165), (195), (255), (297), (325), (387), (459), (495), (561), (585), (645), (765), (891), (975), (1105), (1161), (1485), (1683), (1755), (1935), (2145), (2193), (2295), (2805), (2925), (3315), (3483), (4257), (4455), (5049), (5525), (5805), (6435), (6579), (8385), (8415), (8775), (9945), (10725), (10965), (12771), (14561), (15147), (16575), (17415), (19305), (19737), (21285), (22687), (25155), (25245), (27431), (29835), (32175), (32895), (33911), (36465), (36959), (38313), (41023), (41925), (43039), (43683), (49725), (50407), (57915), (59211), (63855), (66041), (68061), (70487), (72369), (72805), (75465), (75735), (83399), (96525), (98685), (99701), (107063), (109395), (110877), (114939), (115181), (118955), (123069), (125207), (125775), (127033), (129117), (129899), (130681), (130727), (130957), (131003), (131049), (142545), (149175), (165491), (178699), (182325), (188437), (189109), (191565), (198123), (204183), (217107), (218323), (218415), (226395), (234329), (247537)])) w_7_0 w_25_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [29]) ((4/1)) (26) w_29_0 ∧
  outputs = (w_19_0, (w_30_0, ()))

def parallel_parallel_generatedRoot_7_42 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat1 × ByteArray × mxxMat1 × Unit) (outputs : mxxMat5 × mxxMat5 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_2_0 : ByteArray := inputs.2.1
  let w_6_0 : mxxMat1 := inputs.2.2.1
  ∃ (w_1_0 : mxxMat4),
    let w_3_0 : Int := (MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (65), (85), (99), (129), (135), (153), (165), (195), (255), (297), (325), (387), (459), (495), (561), (585), (645), (765), (891), (975), (1105), (1161), (1485), (1683), (1755), (1935), (2145), (2193), (2295), (2805), (2925), (3315), (3483), (4257), (4455), (5049), (5525), (5805), (6435), (6579), (8385), (8415), (8775), (9945), (10725), (10965), (12771), (14561), (15147), (16575), (17415), (19305), (19737), (21285), (22687), (25155), (25245), (27431), (29835), (32175), (32895), (33911), (36465), (36959), (38313), (41023), (41925), (43039), (43683), (49725), (50407), (57915), (59211), (63855), (66041), (68061), (70487), (72369), (72805), (75465), (75735), (83399), (96525), (98685), (99701), (107063), (109395), (110877), (114939), (115181), (118955), (123069), (125207), (125775), (127033), (129117), (129899), (130681), (130727), (130957), (131003), (131049), (142545), (149175), (165491), (178699), (182325), (188437), (189109), (191565), (198123), (204183), (217107), (218323), (218415), (226395), (234329), (247537)])
    ∃ (w_4_0 : mxxMat5),
      let w_5_0 : mxxMat5 := MxxRuntime.matrixMulScalarLeft w_1_0 w_4_0
      ∃ (w_7_0 : mxxMat4),
        ∃ (w_8_0 : mxxMat5),
          let w_9_0 : mxxMat5 := MxxRuntime.matrixMulScalarLeft w_7_0 w_8_0
          let w_10_0 : mxxMat5 := MxxRuntime.matrixAdd w_5_0 w_9_0
          let w_11_0 : mxxMat4 := (MxxRuntime.matrixPolynomial [(1099381866497)] : mxxMat4)
          ∃ (w_12_0 : mxxMat4),
            let w_13_0 : mxxMat4 := MxxRuntime.matrixMulScalarLeft w_11_0 w_12_0
            ∃ (w_14_0 : mxxMat6),
              ∃ (w_15_0 : mxxMat5),
                let w_16_0 : mxxMat5 := MxxRuntime.matrixMulScalarLeft w_13_0 w_15_0
                let w_17_0 : mxxMat5 := MxxRuntime.matrixSub w_10_0 w_16_0
                ∃ (w_18_0 : mxxMat5),
                  let w_19_0 : mxxMat5 := MxxRuntime.matrixAdd w_17_0 w_18_0
                  ∃ (w_20_0 : mxxMat5),
                    let w_21_0 : mxxMat5 := MxxRuntime.matrixMulScalarLeft w_1_0 w_20_0
                    ∃ (w_22_0 : mxxMat5),
                      let w_23_0 : mxxMat5 := MxxRuntime.matrixMulScalarLeft w_7_0 w_22_0
                      let w_24_0 : mxxMat5 := MxxRuntime.matrixAdd w_21_0 w_23_0
                      ∃ (w_25_0 : mxxMat4),
                        let w_26_0 : mxxMat4 := MxxRuntime.matrixMulScalarLeft w_11_0 w_25_0
                        let w_27_0 : mxxMat5 := MxxRuntime.matrixMulScalarLeft w_26_0 w_15_0
                        let w_28_0 : mxxMat5 := MxxRuntime.matrixSub w_24_0 w_27_0
                        ∃ (w_29_0 : mxxMat5),
                          let w_30_0 : mxxMat5 := MxxRuntime.matrixAdd w_28_0 w_29_0
                          parallel_parallel_generatedRoot_7_42.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs

abbrev parallel_parallel_generatedRoot_7_43.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_1 : Nat) (w_0_0 : mxxMat1) (w_2_0 : ByteArray) (w_6_0 : mxxMat1) (w_1_0 : mxxMat4) (w_3_0 : Int) (w_4_0 : mxxMat5) (w_7_0 : mxxMat4) (w_8_0 : mxxMat5) (w_12_0 : mxxMat4) (w_14_0 : mxxMat6) (w_15_0 : mxxMat5) (w_18_0 : mxxMat5) (w_19_0 : mxxMat5) (w_20_0 : mxxMat5) (w_22_0 : mxxMat5) (w_25_0 : mxxMat4) (w_29_0 : mxxMat5) (w_30_0 : mxxMat5) (outputs : mxxMat5 × mxxMat5 × Unit) : Prop :=
  MxxRuntime.modulusReduceRuns w_0_0 w_1_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 32 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (1), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_2_0) w_4_0 ∧
  MxxRuntime.modulusReduceRuns w_6_0 w_7_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (1), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_2_0) w_8_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 32 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (85), (99), (135), (153), (165), (255), (297), (459), (495), (561), (765), (891), (1485), (1683), (2295), (2805), (4455), (5049), (8415), (15147), (25245), (75735)])) w_1_0 w_12_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 1024 164 w_14_0 ∧
  MxxRuntime.centeredRebaseRuns w_14_0 w_15_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [18]) ((4/1)) (26) w_18_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (1), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_2_0) w_20_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (1), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_2_0) w_22_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 32 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (85), (99), (135), (153), (165), (255), (297), (459), (495), (561), (765), (891), (1485), (1683), (2295), (2805), (4455), (5049), (8415), (15147), (25245), (75735)])) w_7_0 w_25_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [29]) ((4/1)) (26) w_29_0 ∧
  outputs = (w_19_0, (w_30_0, ()))

def parallel_parallel_generatedRoot_7_43 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat1 × ByteArray × mxxMat1 × Unit) (outputs : mxxMat5 × mxxMat5 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_2_0 : ByteArray := inputs.2.1
  let w_6_0 : mxxMat1 := inputs.2.2.1
  ∃ (w_1_0 : mxxMat4),
    let w_3_0 : Int := (MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (85), (99), (135), (153), (165), (255), (297), (459), (495), (561), (765), (891), (1485), (1683), (2295), (2805), (4455), (5049), (8415), (15147), (25245), (75735)])
    ∃ (w_4_0 : mxxMat5),
      let w_5_0 : mxxMat5 := MxxRuntime.matrixMulScalarLeft w_1_0 w_4_0
      ∃ (w_7_0 : mxxMat4),
        ∃ (w_8_0 : mxxMat5),
          let w_9_0 : mxxMat5 := MxxRuntime.matrixMulScalarLeft w_7_0 w_8_0
          let w_10_0 : mxxMat5 := MxxRuntime.matrixAdd w_5_0 w_9_0
          let w_11_0 : mxxMat4 := (MxxRuntime.matrixPolynomial [(1099381866497)] : mxxMat4)
          ∃ (w_12_0 : mxxMat4),
            let w_13_0 : mxxMat4 := MxxRuntime.matrixMulScalarLeft w_11_0 w_12_0
            ∃ (w_14_0 : mxxMat6),
              ∃ (w_15_0 : mxxMat5),
                let w_16_0 : mxxMat5 := MxxRuntime.matrixMulScalarLeft w_13_0 w_15_0
                let w_17_0 : mxxMat5 := MxxRuntime.matrixSub w_10_0 w_16_0
                ∃ (w_18_0 : mxxMat5),
                  let w_19_0 : mxxMat5 := MxxRuntime.matrixAdd w_17_0 w_18_0
                  ∃ (w_20_0 : mxxMat5),
                    let w_21_0 : mxxMat5 := MxxRuntime.matrixMulScalarLeft w_1_0 w_20_0
                    ∃ (w_22_0 : mxxMat5),
                      let w_23_0 : mxxMat5 := MxxRuntime.matrixMulScalarLeft w_7_0 w_22_0
                      let w_24_0 : mxxMat5 := MxxRuntime.matrixAdd w_21_0 w_23_0
                      ∃ (w_25_0 : mxxMat4),
                        let w_26_0 : mxxMat4 := MxxRuntime.matrixMulScalarLeft w_11_0 w_25_0
                        let w_27_0 : mxxMat5 := MxxRuntime.matrixMulScalarLeft w_26_0 w_15_0
                        let w_28_0 : mxxMat5 := MxxRuntime.matrixSub w_24_0 w_27_0
                        ∃ (w_29_0 : mxxMat5),
                          let w_30_0 : mxxMat5 := MxxRuntime.matrixAdd w_28_0 w_29_0
                          parallel_parallel_generatedRoot_7_43.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs

abbrev parallel_parallel_generatedRoot_7_44.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_1 : Nat) (w_0_0 : mxxMat1) (w_2_0 : ByteArray) (w_6_0 : mxxMat1) (w_1_0 : mxxMat7) (w_3_0 : Int) (w_4_0 : mxxMat8) (w_7_0 : mxxMat7) (w_8_0 : mxxMat8) (w_12_0 : mxxMat7) (w_14_0 : mxxMat9) (w_15_0 : mxxMat8) (w_18_0 : mxxMat8) (w_19_0 : mxxMat8) (w_20_0 : mxxMat8) (w_22_0 : mxxMat8) (w_25_0 : mxxMat7) (w_29_0 : mxxMat8) (w_30_0 : mxxMat8) (outputs : mxxMat8 × mxxMat8 × Unit) : Prop :=
  MxxRuntime.modulusReduceRuns w_0_0 w_1_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 128 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (10), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_2_0) w_4_0 ∧
  MxxRuntime.modulusReduceRuns w_6_0 w_7_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (10), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_2_0) w_8_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 128 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (65), (85), (99), (129), (135), (153), (165), (195), (255), (297), (325), (387), (459), (495), (561), (585), (645), (765), (891), (975), (1105), (1161), (1485), (1683), (1755), (1935), (2145), (2193), (2295), (2805), (2925), (3315), (3483), (4257), (4455), (5049), (5525), (5805), (6435), (6579), (8385), (8415), (8775), (9945), (10725), (10965), (12771), (14561), (15147), (16575), (17415), (19305), (19737), (21285), (22687), (25155), (25245), (27431), (29835), (32175), (32895), (33911), (36465), (36959), (38313), (41023), (41925), (43039), (43683), (49725), (50407), (57915), (59211), (63855), (66041), (68061), (70487), (72369), (72805), (75465), (75735), (83399), (96525), (98685), (99701), (107063), (109395), (110877), (114939), (115181), (118955), (123069), (125207), (125775), (127033), (129117), (129899), (130681), (130727), (130957), (131003), (131049), (142545), (149175), (165491), (178699), (182325), (188437), (189109), (191565), (198123), (204183), (217107), (218323), (218415), (226395), (234329), (247537)])) w_1_0 w_12_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 1024 128 w_14_0 ∧
  MxxRuntime.centeredRebaseRuns w_14_0 w_15_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [18]) ((4/1)) (26) w_18_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (10), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_2_0) w_20_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (10), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_2_0) w_22_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 128 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (65), (85), (99), (129), (135), (153), (165), (195), (255), (297), (325), (387), (459), (495), (561), (585), (645), (765), (891), (975), (1105), (1161), (1485), (1683), (1755), (1935), (2145), (2193), (2295), (2805), (2925), (3315), (3483), (4257), (4455), (5049), (5525), (5805), (6435), (6579), (8385), (8415), (8775), (9945), (10725), (10965), (12771), (14561), (15147), (16575), (17415), (19305), (19737), (21285), (22687), (25155), (25245), (27431), (29835), (32175), (32895), (33911), (36465), (36959), (38313), (41023), (41925), (43039), (43683), (49725), (50407), (57915), (59211), (63855), (66041), (68061), (70487), (72369), (72805), (75465), (75735), (83399), (96525), (98685), (99701), (107063), (109395), (110877), (114939), (115181), (118955), (123069), (125207), (125775), (127033), (129117), (129899), (130681), (130727), (130957), (131003), (131049), (142545), (149175), (165491), (178699), (182325), (188437), (189109), (191565), (198123), (204183), (217107), (218323), (218415), (226395), (234329), (247537)])) w_7_0 w_25_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [29]) ((4/1)) (26) w_29_0 ∧
  outputs = (w_19_0, (w_30_0, ()))

def parallel_parallel_generatedRoot_7_44 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat1 × ByteArray × mxxMat1 × Unit) (outputs : mxxMat8 × mxxMat8 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_2_0 : ByteArray := inputs.2.1
  let w_6_0 : mxxMat1 := inputs.2.2.1
  ∃ (w_1_0 : mxxMat7),
    let w_3_0 : Int := (MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (65), (85), (99), (129), (135), (153), (165), (195), (255), (297), (325), (387), (459), (495), (561), (585), (645), (765), (891), (975), (1105), (1161), (1485), (1683), (1755), (1935), (2145), (2193), (2295), (2805), (2925), (3315), (3483), (4257), (4455), (5049), (5525), (5805), (6435), (6579), (8385), (8415), (8775), (9945), (10725), (10965), (12771), (14561), (15147), (16575), (17415), (19305), (19737), (21285), (22687), (25155), (25245), (27431), (29835), (32175), (32895), (33911), (36465), (36959), (38313), (41023), (41925), (43039), (43683), (49725), (50407), (57915), (59211), (63855), (66041), (68061), (70487), (72369), (72805), (75465), (75735), (83399), (96525), (98685), (99701), (107063), (109395), (110877), (114939), (115181), (118955), (123069), (125207), (125775), (127033), (129117), (129899), (130681), (130727), (130957), (131003), (131049), (142545), (149175), (165491), (178699), (182325), (188437), (189109), (191565), (198123), (204183), (217107), (218323), (218415), (226395), (234329), (247537)])
    ∃ (w_4_0 : mxxMat8),
      let w_5_0 : mxxMat8 := MxxRuntime.matrixMulScalarLeft w_1_0 w_4_0
      ∃ (w_7_0 : mxxMat7),
        ∃ (w_8_0 : mxxMat8),
          let w_9_0 : mxxMat8 := MxxRuntime.matrixMulScalarLeft w_7_0 w_8_0
          let w_10_0 : mxxMat8 := MxxRuntime.matrixAdd w_5_0 w_9_0
          let w_11_0 : mxxMat7 := (MxxRuntime.matrixPolynomial [(1099412275201)] : mxxMat7)
          ∃ (w_12_0 : mxxMat7),
            let w_13_0 : mxxMat7 := MxxRuntime.matrixMulScalarLeft w_11_0 w_12_0
            ∃ (w_14_0 : mxxMat9),
              ∃ (w_15_0 : mxxMat8),
                let w_16_0 : mxxMat8 := MxxRuntime.matrixMulScalarLeft w_13_0 w_15_0
                let w_17_0 : mxxMat8 := MxxRuntime.matrixSub w_10_0 w_16_0
                ∃ (w_18_0 : mxxMat8),
                  let w_19_0 : mxxMat8 := MxxRuntime.matrixAdd w_17_0 w_18_0
                  ∃ (w_20_0 : mxxMat8),
                    let w_21_0 : mxxMat8 := MxxRuntime.matrixMulScalarLeft w_1_0 w_20_0
                    ∃ (w_22_0 : mxxMat8),
                      let w_23_0 : mxxMat8 := MxxRuntime.matrixMulScalarLeft w_7_0 w_22_0
                      let w_24_0 : mxxMat8 := MxxRuntime.matrixAdd w_21_0 w_23_0
                      ∃ (w_25_0 : mxxMat7),
                        let w_26_0 : mxxMat7 := MxxRuntime.matrixMulScalarLeft w_11_0 w_25_0
                        let w_27_0 : mxxMat8 := MxxRuntime.matrixMulScalarLeft w_26_0 w_15_0
                        let w_28_0 : mxxMat8 := MxxRuntime.matrixSub w_24_0 w_27_0
                        ∃ (w_29_0 : mxxMat8),
                          let w_30_0 : mxxMat8 := MxxRuntime.matrixAdd w_28_0 w_29_0
                          parallel_parallel_generatedRoot_7_44.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs

abbrev parallel_parallel_generatedRoot_7_45.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_1 : Nat) (w_0_0 : mxxMat1) (w_2_0 : ByteArray) (w_6_0 : mxxMat1) (w_1_0 : mxxMat7) (w_3_0 : Int) (w_4_0 : mxxMat8) (w_7_0 : mxxMat7) (w_8_0 : mxxMat8) (w_12_0 : mxxMat7) (w_14_0 : mxxMat9) (w_15_0 : mxxMat8) (w_18_0 : mxxMat8) (w_19_0 : mxxMat8) (w_20_0 : mxxMat8) (w_22_0 : mxxMat8) (w_25_0 : mxxMat7) (w_29_0 : mxxMat8) (w_30_0 : mxxMat8) (outputs : mxxMat8 × mxxMat8 × Unit) : Prop :=
  MxxRuntime.modulusReduceRuns w_0_0 w_1_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 32 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (10), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_2_0) w_4_0 ∧
  MxxRuntime.modulusReduceRuns w_6_0 w_7_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (10), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_2_0) w_8_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 32 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (85), (99), (135), (153), (165), (255), (297), (459), (495), (561), (765), (891), (1485), (1683), (2295), (2805), (4455), (5049), (8415), (15147), (25245), (75735)])) w_1_0 w_12_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 1024 128 w_14_0 ∧
  MxxRuntime.centeredRebaseRuns w_14_0 w_15_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [18]) ((4/1)) (26) w_18_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (10), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_2_0) w_20_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (10), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_2_0) w_22_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 32 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (85), (99), (135), (153), (165), (255), (297), (459), (495), (561), (765), (891), (1485), (1683), (2295), (2805), (4455), (5049), (8415), (15147), (25245), (75735)])) w_7_0 w_25_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [29]) ((4/1)) (26) w_29_0 ∧
  outputs = (w_19_0, (w_30_0, ()))

def parallel_parallel_generatedRoot_7_45 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat1 × ByteArray × mxxMat1 × Unit) (outputs : mxxMat8 × mxxMat8 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_2_0 : ByteArray := inputs.2.1
  let w_6_0 : mxxMat1 := inputs.2.2.1
  ∃ (w_1_0 : mxxMat7),
    let w_3_0 : Int := (MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (85), (99), (135), (153), (165), (255), (297), (459), (495), (561), (765), (891), (1485), (1683), (2295), (2805), (4455), (5049), (8415), (15147), (25245), (75735)])
    ∃ (w_4_0 : mxxMat8),
      let w_5_0 : mxxMat8 := MxxRuntime.matrixMulScalarLeft w_1_0 w_4_0
      ∃ (w_7_0 : mxxMat7),
        ∃ (w_8_0 : mxxMat8),
          let w_9_0 : mxxMat8 := MxxRuntime.matrixMulScalarLeft w_7_0 w_8_0
          let w_10_0 : mxxMat8 := MxxRuntime.matrixAdd w_5_0 w_9_0
          let w_11_0 : mxxMat7 := (MxxRuntime.matrixPolynomial [(1099412275201)] : mxxMat7)
          ∃ (w_12_0 : mxxMat7),
            let w_13_0 : mxxMat7 := MxxRuntime.matrixMulScalarLeft w_11_0 w_12_0
            ∃ (w_14_0 : mxxMat9),
              ∃ (w_15_0 : mxxMat8),
                let w_16_0 : mxxMat8 := MxxRuntime.matrixMulScalarLeft w_13_0 w_15_0
                let w_17_0 : mxxMat8 := MxxRuntime.matrixSub w_10_0 w_16_0
                ∃ (w_18_0 : mxxMat8),
                  let w_19_0 : mxxMat8 := MxxRuntime.matrixAdd w_17_0 w_18_0
                  ∃ (w_20_0 : mxxMat8),
                    let w_21_0 : mxxMat8 := MxxRuntime.matrixMulScalarLeft w_1_0 w_20_0
                    ∃ (w_22_0 : mxxMat8),
                      let w_23_0 : mxxMat8 := MxxRuntime.matrixMulScalarLeft w_7_0 w_22_0
                      let w_24_0 : mxxMat8 := MxxRuntime.matrixAdd w_21_0 w_23_0
                      ∃ (w_25_0 : mxxMat7),
                        let w_26_0 : mxxMat7 := MxxRuntime.matrixMulScalarLeft w_11_0 w_25_0
                        let w_27_0 : mxxMat8 := MxxRuntime.matrixMulScalarLeft w_26_0 w_15_0
                        let w_28_0 : mxxMat8 := MxxRuntime.matrixSub w_24_0 w_27_0
                        ∃ (w_29_0 : mxxMat8),
                          let w_30_0 : mxxMat8 := MxxRuntime.matrixAdd w_28_0 w_29_0
                          parallel_parallel_generatedRoot_7_45.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs

abbrev parallel_parallel_generatedRoot_7_46.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_1 : Nat) (w_0_0 : mxxMat1) (w_2_0 : ByteArray) (w_6_0 : mxxMat1) (w_1_0 : mxxMat10) (w_3_0 : Int) (w_4_0 : mxxMat11) (w_7_0 : mxxMat10) (w_8_0 : mxxMat11) (w_12_0 : mxxMat10) (w_14_0 : mxxMat12) (w_15_0 : mxxMat11) (w_18_0 : mxxMat11) (w_19_0 : mxxMat11) (w_20_0 : mxxMat11) (w_22_0 : mxxMat11) (w_25_0 : mxxMat10) (w_29_0 : mxxMat11) (w_30_0 : mxxMat11) (outputs : mxxMat11 × mxxMat11 × Unit) : Prop :=
  MxxRuntime.modulusReduceRuns w_0_0 w_1_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 128 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (11), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_2_0) w_4_0 ∧
  MxxRuntime.modulusReduceRuns w_6_0 w_7_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (11), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_2_0) w_8_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 128 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (65), (85), (99), (129), (135), (153), (165), (195), (255), (297), (325), (387), (459), (495), (561), (585), (645), (765), (891), (975), (1105), (1161), (1485), (1683), (1755), (1935), (2145), (2193), (2295), (2805), (2925), (3315), (3483), (4257), (4455), (5049), (5525), (5805), (6435), (6579), (8385), (8415), (8775), (9945), (10725), (10965), (12771), (14561), (15147), (16575), (17415), (19305), (19737), (21285), (22687), (25155), (25245), (27431), (29835), (32175), (32895), (33911), (36465), (36959), (38313), (41023), (41925), (43039), (43683), (49725), (50407), (57915), (59211), (63855), (66041), (68061), (70487), (72369), (72805), (75465), (75735), (83399), (96525), (98685), (99701), (107063), (109395), (110877), (114939), (115181), (118955), (123069), (125207), (125775), (127033), (129117), (129899), (130681), (130727), (130957), (131003), (131049), (142545), (149175), (165491), (178699), (182325), (188437), (189109), (191565), (198123), (204183), (217107), (218323), (218415), (226395), (234329), (247537)])) w_1_0 w_12_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 1024 124 w_14_0 ∧
  MxxRuntime.centeredRebaseRuns w_14_0 w_15_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [18]) ((4/1)) (26) w_18_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (11), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_2_0) w_20_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (11), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_2_0) w_22_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 128 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (65), (85), (99), (129), (135), (153), (165), (195), (255), (297), (325), (387), (459), (495), (561), (585), (645), (765), (891), (975), (1105), (1161), (1485), (1683), (1755), (1935), (2145), (2193), (2295), (2805), (2925), (3315), (3483), (4257), (4455), (5049), (5525), (5805), (6435), (6579), (8385), (8415), (8775), (9945), (10725), (10965), (12771), (14561), (15147), (16575), (17415), (19305), (19737), (21285), (22687), (25155), (25245), (27431), (29835), (32175), (32895), (33911), (36465), (36959), (38313), (41023), (41925), (43039), (43683), (49725), (50407), (57915), (59211), (63855), (66041), (68061), (70487), (72369), (72805), (75465), (75735), (83399), (96525), (98685), (99701), (107063), (109395), (110877), (114939), (115181), (118955), (123069), (125207), (125775), (127033), (129117), (129899), (130681), (130727), (130957), (131003), (131049), (142545), (149175), (165491), (178699), (182325), (188437), (189109), (191565), (198123), (204183), (217107), (218323), (218415), (226395), (234329), (247537)])) w_7_0 w_25_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [29]) ((4/1)) (26) w_29_0 ∧
  outputs = (w_19_0, (w_30_0, ()))

def parallel_parallel_generatedRoot_7_46 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat1 × ByteArray × mxxMat1 × Unit) (outputs : mxxMat11 × mxxMat11 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_2_0 : ByteArray := inputs.2.1
  let w_6_0 : mxxMat1 := inputs.2.2.1
  ∃ (w_1_0 : mxxMat10),
    let w_3_0 : Int := (MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (65), (85), (99), (129), (135), (153), (165), (195), (255), (297), (325), (387), (459), (495), (561), (585), (645), (765), (891), (975), (1105), (1161), (1485), (1683), (1755), (1935), (2145), (2193), (2295), (2805), (2925), (3315), (3483), (4257), (4455), (5049), (5525), (5805), (6435), (6579), (8385), (8415), (8775), (9945), (10725), (10965), (12771), (14561), (15147), (16575), (17415), (19305), (19737), (21285), (22687), (25155), (25245), (27431), (29835), (32175), (32895), (33911), (36465), (36959), (38313), (41023), (41925), (43039), (43683), (49725), (50407), (57915), (59211), (63855), (66041), (68061), (70487), (72369), (72805), (75465), (75735), (83399), (96525), (98685), (99701), (107063), (109395), (110877), (114939), (115181), (118955), (123069), (125207), (125775), (127033), (129117), (129899), (130681), (130727), (130957), (131003), (131049), (142545), (149175), (165491), (178699), (182325), (188437), (189109), (191565), (198123), (204183), (217107), (218323), (218415), (226395), (234329), (247537)])
    ∃ (w_4_0 : mxxMat11),
      let w_5_0 : mxxMat11 := MxxRuntime.matrixMulScalarLeft w_1_0 w_4_0
      ∃ (w_7_0 : mxxMat10),
        ∃ (w_8_0 : mxxMat11),
          let w_9_0 : mxxMat11 := MxxRuntime.matrixMulScalarLeft w_7_0 w_8_0
          let w_10_0 : mxxMat11 := MxxRuntime.matrixAdd w_5_0 w_9_0
          let w_11_0 : mxxMat10 := (MxxRuntime.matrixPolynomial [(1099413323777)] : mxxMat10)
          ∃ (w_12_0 : mxxMat10),
            let w_13_0 : mxxMat10 := MxxRuntime.matrixMulScalarLeft w_11_0 w_12_0
            ∃ (w_14_0 : mxxMat12),
              ∃ (w_15_0 : mxxMat11),
                let w_16_0 : mxxMat11 := MxxRuntime.matrixMulScalarLeft w_13_0 w_15_0
                let w_17_0 : mxxMat11 := MxxRuntime.matrixSub w_10_0 w_16_0
                ∃ (w_18_0 : mxxMat11),
                  let w_19_0 : mxxMat11 := MxxRuntime.matrixAdd w_17_0 w_18_0
                  ∃ (w_20_0 : mxxMat11),
                    let w_21_0 : mxxMat11 := MxxRuntime.matrixMulScalarLeft w_1_0 w_20_0
                    ∃ (w_22_0 : mxxMat11),
                      let w_23_0 : mxxMat11 := MxxRuntime.matrixMulScalarLeft w_7_0 w_22_0
                      let w_24_0 : mxxMat11 := MxxRuntime.matrixAdd w_21_0 w_23_0
                      ∃ (w_25_0 : mxxMat10),
                        let w_26_0 : mxxMat10 := MxxRuntime.matrixMulScalarLeft w_11_0 w_25_0
                        let w_27_0 : mxxMat11 := MxxRuntime.matrixMulScalarLeft w_26_0 w_15_0
                        let w_28_0 : mxxMat11 := MxxRuntime.matrixSub w_24_0 w_27_0
                        ∃ (w_29_0 : mxxMat11),
                          let w_30_0 : mxxMat11 := MxxRuntime.matrixAdd w_28_0 w_29_0
                          parallel_parallel_generatedRoot_7_46.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs

abbrev parallel_parallel_generatedRoot_7_47.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_1 : Nat) (w_0_0 : mxxMat1) (w_2_0 : ByteArray) (w_6_0 : mxxMat1) (w_1_0 : mxxMat10) (w_3_0 : Int) (w_4_0 : mxxMat11) (w_7_0 : mxxMat10) (w_8_0 : mxxMat11) (w_12_0 : mxxMat10) (w_14_0 : mxxMat12) (w_15_0 : mxxMat11) (w_18_0 : mxxMat11) (w_19_0 : mxxMat11) (w_20_0 : mxxMat11) (w_22_0 : mxxMat11) (w_25_0 : mxxMat10) (w_29_0 : mxxMat11) (w_30_0 : mxxMat11) (outputs : mxxMat11 × mxxMat11 × Unit) : Prop :=
  MxxRuntime.modulusReduceRuns w_0_0 w_1_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 32 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (11), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_2_0) w_4_0 ∧
  MxxRuntime.modulusReduceRuns w_6_0 w_7_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (11), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_2_0) w_8_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 32 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (85), (99), (135), (153), (165), (255), (297), (459), (495), (561), (765), (891), (1485), (1683), (2295), (2805), (4455), (5049), (8415), (15147), (25245), (75735)])) w_1_0 w_12_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 1024 124 w_14_0 ∧
  MxxRuntime.centeredRebaseRuns w_14_0 w_15_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [18]) ((4/1)) (26) w_18_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (11), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_2_0) w_20_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (11), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_2_0) w_22_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 32 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (85), (99), (135), (153), (165), (255), (297), (459), (495), (561), (765), (891), (1485), (1683), (2295), (2805), (4455), (5049), (8415), (15147), (25245), (75735)])) w_7_0 w_25_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [29]) ((4/1)) (26) w_29_0 ∧
  outputs = (w_19_0, (w_30_0, ()))

def parallel_parallel_generatedRoot_7_47 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat1 × ByteArray × mxxMat1 × Unit) (outputs : mxxMat11 × mxxMat11 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_2_0 : ByteArray := inputs.2.1
  let w_6_0 : mxxMat1 := inputs.2.2.1
  ∃ (w_1_0 : mxxMat10),
    let w_3_0 : Int := (MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (85), (99), (135), (153), (165), (255), (297), (459), (495), (561), (765), (891), (1485), (1683), (2295), (2805), (4455), (5049), (8415), (15147), (25245), (75735)])
    ∃ (w_4_0 : mxxMat11),
      let w_5_0 : mxxMat11 := MxxRuntime.matrixMulScalarLeft w_1_0 w_4_0
      ∃ (w_7_0 : mxxMat10),
        ∃ (w_8_0 : mxxMat11),
          let w_9_0 : mxxMat11 := MxxRuntime.matrixMulScalarLeft w_7_0 w_8_0
          let w_10_0 : mxxMat11 := MxxRuntime.matrixAdd w_5_0 w_9_0
          let w_11_0 : mxxMat10 := (MxxRuntime.matrixPolynomial [(1099413323777)] : mxxMat10)
          ∃ (w_12_0 : mxxMat10),
            let w_13_0 : mxxMat10 := MxxRuntime.matrixMulScalarLeft w_11_0 w_12_0
            ∃ (w_14_0 : mxxMat12),
              ∃ (w_15_0 : mxxMat11),
                let w_16_0 : mxxMat11 := MxxRuntime.matrixMulScalarLeft w_13_0 w_15_0
                let w_17_0 : mxxMat11 := MxxRuntime.matrixSub w_10_0 w_16_0
                ∃ (w_18_0 : mxxMat11),
                  let w_19_0 : mxxMat11 := MxxRuntime.matrixAdd w_17_0 w_18_0
                  ∃ (w_20_0 : mxxMat11),
                    let w_21_0 : mxxMat11 := MxxRuntime.matrixMulScalarLeft w_1_0 w_20_0
                    ∃ (w_22_0 : mxxMat11),
                      let w_23_0 : mxxMat11 := MxxRuntime.matrixMulScalarLeft w_7_0 w_22_0
                      let w_24_0 : mxxMat11 := MxxRuntime.matrixAdd w_21_0 w_23_0
                      ∃ (w_25_0 : mxxMat10),
                        let w_26_0 : mxxMat10 := MxxRuntime.matrixMulScalarLeft w_11_0 w_25_0
                        let w_27_0 : mxxMat11 := MxxRuntime.matrixMulScalarLeft w_26_0 w_15_0
                        let w_28_0 : mxxMat11 := MxxRuntime.matrixSub w_24_0 w_27_0
                        ∃ (w_29_0 : mxxMat11),
                          let w_30_0 : mxxMat11 := MxxRuntime.matrixAdd w_28_0 w_29_0
                          parallel_parallel_generatedRoot_7_47.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs

abbrev parallel_parallel_generatedRoot_7_48.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_1 : Nat) (w_0_0 : mxxMat1) (w_2_0 : ByteArray) (w_6_0 : mxxMat1) (w_1_0 : mxxMat13) (w_3_0 : Int) (w_4_0 : mxxMat14) (w_7_0 : mxxMat13) (w_8_0 : mxxMat14) (w_12_0 : mxxMat13) (w_14_0 : mxxMat15) (w_15_0 : mxxMat14) (w_18_0 : mxxMat14) (w_19_0 : mxxMat14) (w_20_0 : mxxMat14) (w_22_0 : mxxMat14) (w_25_0 : mxxMat13) (w_29_0 : mxxMat14) (w_30_0 : mxxMat14) (outputs : mxxMat14 × mxxMat14 × Unit) : Prop :=
  MxxRuntime.modulusReduceRuns w_0_0 w_1_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 128 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (12), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_2_0) w_4_0 ∧
  MxxRuntime.modulusReduceRuns w_6_0 w_7_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (12), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_2_0) w_8_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 128 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (65), (85), (99), (129), (135), (153), (165), (195), (255), (297), (325), (387), (459), (495), (561), (585), (645), (765), (891), (975), (1105), (1161), (1485), (1683), (1755), (1935), (2145), (2193), (2295), (2805), (2925), (3315), (3483), (4257), (4455), (5049), (5525), (5805), (6435), (6579), (8385), (8415), (8775), (9945), (10725), (10965), (12771), (14561), (15147), (16575), (17415), (19305), (19737), (21285), (22687), (25155), (25245), (27431), (29835), (32175), (32895), (33911), (36465), (36959), (38313), (41023), (41925), (43039), (43683), (49725), (50407), (57915), (59211), (63855), (66041), (68061), (70487), (72369), (72805), (75465), (75735), (83399), (96525), (98685), (99701), (107063), (109395), (110877), (114939), (115181), (118955), (123069), (125207), (125775), (127033), (129117), (129899), (130681), (130727), (130957), (131003), (131049), (142545), (149175), (165491), (178699), (182325), (188437), (189109), (191565), (198123), (204183), (217107), (218323), (218415), (226395), (234329), (247537)])) w_1_0 w_12_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 1024 120 w_14_0 ∧
  MxxRuntime.centeredRebaseRuns w_14_0 w_15_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [18]) ((4/1)) (26) w_18_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (12), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_2_0) w_20_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (12), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_2_0) w_22_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 128 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (65), (85), (99), (129), (135), (153), (165), (195), (255), (297), (325), (387), (459), (495), (561), (585), (645), (765), (891), (975), (1105), (1161), (1485), (1683), (1755), (1935), (2145), (2193), (2295), (2805), (2925), (3315), (3483), (4257), (4455), (5049), (5525), (5805), (6435), (6579), (8385), (8415), (8775), (9945), (10725), (10965), (12771), (14561), (15147), (16575), (17415), (19305), (19737), (21285), (22687), (25155), (25245), (27431), (29835), (32175), (32895), (33911), (36465), (36959), (38313), (41023), (41925), (43039), (43683), (49725), (50407), (57915), (59211), (63855), (66041), (68061), (70487), (72369), (72805), (75465), (75735), (83399), (96525), (98685), (99701), (107063), (109395), (110877), (114939), (115181), (118955), (123069), (125207), (125775), (127033), (129117), (129899), (130681), (130727), (130957), (131003), (131049), (142545), (149175), (165491), (178699), (182325), (188437), (189109), (191565), (198123), (204183), (217107), (218323), (218415), (226395), (234329), (247537)])) w_7_0 w_25_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [29]) ((4/1)) (26) w_29_0 ∧
  outputs = (w_19_0, (w_30_0, ()))

def parallel_parallel_generatedRoot_7_48 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat1 × ByteArray × mxxMat1 × Unit) (outputs : mxxMat14 × mxxMat14 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_2_0 : ByteArray := inputs.2.1
  let w_6_0 : mxxMat1 := inputs.2.2.1
  ∃ (w_1_0 : mxxMat13),
    let w_3_0 : Int := (MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (65), (85), (99), (129), (135), (153), (165), (195), (255), (297), (325), (387), (459), (495), (561), (585), (645), (765), (891), (975), (1105), (1161), (1485), (1683), (1755), (1935), (2145), (2193), (2295), (2805), (2925), (3315), (3483), (4257), (4455), (5049), (5525), (5805), (6435), (6579), (8385), (8415), (8775), (9945), (10725), (10965), (12771), (14561), (15147), (16575), (17415), (19305), (19737), (21285), (22687), (25155), (25245), (27431), (29835), (32175), (32895), (33911), (36465), (36959), (38313), (41023), (41925), (43039), (43683), (49725), (50407), (57915), (59211), (63855), (66041), (68061), (70487), (72369), (72805), (75465), (75735), (83399), (96525), (98685), (99701), (107063), (109395), (110877), (114939), (115181), (118955), (123069), (125207), (125775), (127033), (129117), (129899), (130681), (130727), (130957), (131003), (131049), (142545), (149175), (165491), (178699), (182325), (188437), (189109), (191565), (198123), (204183), (217107), (218323), (218415), (226395), (234329), (247537)])
    ∃ (w_4_0 : mxxMat14),
      let w_5_0 : mxxMat14 := MxxRuntime.matrixMulScalarLeft w_1_0 w_4_0
      ∃ (w_7_0 : mxxMat13),
        ∃ (w_8_0 : mxxMat14),
          let w_9_0 : mxxMat14 := MxxRuntime.matrixMulScalarLeft w_7_0 w_8_0
          let w_10_0 : mxxMat14 := MxxRuntime.matrixAdd w_5_0 w_9_0
          let w_11_0 : mxxMat13 := (MxxRuntime.matrixPolynomial [(1099416993793)] : mxxMat13)
          ∃ (w_12_0 : mxxMat13),
            let w_13_0 : mxxMat13 := MxxRuntime.matrixMulScalarLeft w_11_0 w_12_0
            ∃ (w_14_0 : mxxMat15),
              ∃ (w_15_0 : mxxMat14),
                let w_16_0 : mxxMat14 := MxxRuntime.matrixMulScalarLeft w_13_0 w_15_0
                let w_17_0 : mxxMat14 := MxxRuntime.matrixSub w_10_0 w_16_0
                ∃ (w_18_0 : mxxMat14),
                  let w_19_0 : mxxMat14 := MxxRuntime.matrixAdd w_17_0 w_18_0
                  ∃ (w_20_0 : mxxMat14),
                    let w_21_0 : mxxMat14 := MxxRuntime.matrixMulScalarLeft w_1_0 w_20_0
                    ∃ (w_22_0 : mxxMat14),
                      let w_23_0 : mxxMat14 := MxxRuntime.matrixMulScalarLeft w_7_0 w_22_0
                      let w_24_0 : mxxMat14 := MxxRuntime.matrixAdd w_21_0 w_23_0
                      ∃ (w_25_0 : mxxMat13),
                        let w_26_0 : mxxMat13 := MxxRuntime.matrixMulScalarLeft w_11_0 w_25_0
                        let w_27_0 : mxxMat14 := MxxRuntime.matrixMulScalarLeft w_26_0 w_15_0
                        let w_28_0 : mxxMat14 := MxxRuntime.matrixSub w_24_0 w_27_0
                        ∃ (w_29_0 : mxxMat14),
                          let w_30_0 : mxxMat14 := MxxRuntime.matrixAdd w_28_0 w_29_0
                          parallel_parallel_generatedRoot_7_48.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs

abbrev parallel_parallel_generatedRoot_7_49.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_1 : Nat) (w_0_0 : mxxMat1) (w_2_0 : ByteArray) (w_6_0 : mxxMat1) (w_1_0 : mxxMat13) (w_3_0 : Int) (w_4_0 : mxxMat14) (w_7_0 : mxxMat13) (w_8_0 : mxxMat14) (w_12_0 : mxxMat13) (w_14_0 : mxxMat15) (w_15_0 : mxxMat14) (w_18_0 : mxxMat14) (w_19_0 : mxxMat14) (w_20_0 : mxxMat14) (w_22_0 : mxxMat14) (w_25_0 : mxxMat13) (w_29_0 : mxxMat14) (w_30_0 : mxxMat14) (outputs : mxxMat14 × mxxMat14 × Unit) : Prop :=
  MxxRuntime.modulusReduceRuns w_0_0 w_1_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 32 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (12), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_2_0) w_4_0 ∧
  MxxRuntime.modulusReduceRuns w_6_0 w_7_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (12), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_2_0) w_8_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 32 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (85), (99), (135), (153), (165), (255), (297), (459), (495), (561), (765), (891), (1485), (1683), (2295), (2805), (4455), (5049), (8415), (15147), (25245), (75735)])) w_1_0 w_12_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 1024 120 w_14_0 ∧
  MxxRuntime.centeredRebaseRuns w_14_0 w_15_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [18]) ((4/1)) (26) w_18_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (12), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_2_0) w_20_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (12), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_2_0) w_22_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 32 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (85), (99), (135), (153), (165), (255), (297), (459), (495), (561), (765), (891), (1485), (1683), (2295), (2805), (4455), (5049), (8415), (15147), (25245), (75735)])) w_7_0 w_25_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [29]) ((4/1)) (26) w_29_0 ∧
  outputs = (w_19_0, (w_30_0, ()))

def parallel_parallel_generatedRoot_7_49 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat1 × ByteArray × mxxMat1 × Unit) (outputs : mxxMat14 × mxxMat14 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_2_0 : ByteArray := inputs.2.1
  let w_6_0 : mxxMat1 := inputs.2.2.1
  ∃ (w_1_0 : mxxMat13),
    let w_3_0 : Int := (MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (85), (99), (135), (153), (165), (255), (297), (459), (495), (561), (765), (891), (1485), (1683), (2295), (2805), (4455), (5049), (8415), (15147), (25245), (75735)])
    ∃ (w_4_0 : mxxMat14),
      let w_5_0 : mxxMat14 := MxxRuntime.matrixMulScalarLeft w_1_0 w_4_0
      ∃ (w_7_0 : mxxMat13),
        ∃ (w_8_0 : mxxMat14),
          let w_9_0 : mxxMat14 := MxxRuntime.matrixMulScalarLeft w_7_0 w_8_0
          let w_10_0 : mxxMat14 := MxxRuntime.matrixAdd w_5_0 w_9_0
          let w_11_0 : mxxMat13 := (MxxRuntime.matrixPolynomial [(1099416993793)] : mxxMat13)
          ∃ (w_12_0 : mxxMat13),
            let w_13_0 : mxxMat13 := MxxRuntime.matrixMulScalarLeft w_11_0 w_12_0
            ∃ (w_14_0 : mxxMat15),
              ∃ (w_15_0 : mxxMat14),
                let w_16_0 : mxxMat14 := MxxRuntime.matrixMulScalarLeft w_13_0 w_15_0
                let w_17_0 : mxxMat14 := MxxRuntime.matrixSub w_10_0 w_16_0
                ∃ (w_18_0 : mxxMat14),
                  let w_19_0 : mxxMat14 := MxxRuntime.matrixAdd w_17_0 w_18_0
                  ∃ (w_20_0 : mxxMat14),
                    let w_21_0 : mxxMat14 := MxxRuntime.matrixMulScalarLeft w_1_0 w_20_0
                    ∃ (w_22_0 : mxxMat14),
                      let w_23_0 : mxxMat14 := MxxRuntime.matrixMulScalarLeft w_7_0 w_22_0
                      let w_24_0 : mxxMat14 := MxxRuntime.matrixAdd w_21_0 w_23_0
                      ∃ (w_25_0 : mxxMat13),
                        let w_26_0 : mxxMat13 := MxxRuntime.matrixMulScalarLeft w_11_0 w_25_0
                        let w_27_0 : mxxMat14 := MxxRuntime.matrixMulScalarLeft w_26_0 w_15_0
                        let w_28_0 : mxxMat14 := MxxRuntime.matrixSub w_24_0 w_27_0
                        ∃ (w_29_0 : mxxMat14),
                          let w_30_0 : mxxMat14 := MxxRuntime.matrixAdd w_28_0 w_29_0
                          parallel_parallel_generatedRoot_7_49.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs

abbrev parallel_parallel_generatedRoot_7_50.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_1 : Nat) (w_0_0 : mxxMat1) (w_2_0 : ByteArray) (w_6_0 : mxxMat1) (w_1_0 : mxxMat16) (w_3_0 : Int) (w_4_0 : mxxMat17) (w_7_0 : mxxMat16) (w_8_0 : mxxMat17) (w_12_0 : mxxMat16) (w_14_0 : mxxMat18) (w_15_0 : mxxMat17) (w_18_0 : mxxMat17) (w_19_0 : mxxMat17) (w_20_0 : mxxMat17) (w_22_0 : mxxMat17) (w_25_0 : mxxMat16) (w_29_0 : mxxMat17) (w_30_0 : mxxMat17) (outputs : mxxMat17 × mxxMat17 × Unit) : Prop :=
  MxxRuntime.modulusReduceRuns w_0_0 w_1_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 128 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (13), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_2_0) w_4_0 ∧
  MxxRuntime.modulusReduceRuns w_6_0 w_7_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (13), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_2_0) w_8_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 128 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (65), (85), (99), (129), (135), (153), (165), (195), (255), (297), (325), (387), (459), (495), (561), (585), (645), (765), (891), (975), (1105), (1161), (1485), (1683), (1755), (1935), (2145), (2193), (2295), (2805), (2925), (3315), (3483), (4257), (4455), (5049), (5525), (5805), (6435), (6579), (8385), (8415), (8775), (9945), (10725), (10965), (12771), (14561), (15147), (16575), (17415), (19305), (19737), (21285), (22687), (25155), (25245), (27431), (29835), (32175), (32895), (33911), (36465), (36959), (38313), (41023), (41925), (43039), (43683), (49725), (50407), (57915), (59211), (63855), (66041), (68061), (70487), (72369), (72805), (75465), (75735), (83399), (96525), (98685), (99701), (107063), (109395), (110877), (114939), (115181), (118955), (123069), (125207), (125775), (127033), (129117), (129899), (130681), (130727), (130957), (131003), (131049), (142545), (149175), (165491), (178699), (182325), (188437), (189109), (191565), (198123), (204183), (217107), (218323), (218415), (226395), (234329), (247537)])) w_1_0 w_12_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 1024 116 w_14_0 ∧
  MxxRuntime.centeredRebaseRuns w_14_0 w_15_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [18]) ((4/1)) (26) w_18_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (13), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_2_0) w_20_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (13), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_2_0) w_22_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 128 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (65), (85), (99), (129), (135), (153), (165), (195), (255), (297), (325), (387), (459), (495), (561), (585), (645), (765), (891), (975), (1105), (1161), (1485), (1683), (1755), (1935), (2145), (2193), (2295), (2805), (2925), (3315), (3483), (4257), (4455), (5049), (5525), (5805), (6435), (6579), (8385), (8415), (8775), (9945), (10725), (10965), (12771), (14561), (15147), (16575), (17415), (19305), (19737), (21285), (22687), (25155), (25245), (27431), (29835), (32175), (32895), (33911), (36465), (36959), (38313), (41023), (41925), (43039), (43683), (49725), (50407), (57915), (59211), (63855), (66041), (68061), (70487), (72369), (72805), (75465), (75735), (83399), (96525), (98685), (99701), (107063), (109395), (110877), (114939), (115181), (118955), (123069), (125207), (125775), (127033), (129117), (129899), (130681), (130727), (130957), (131003), (131049), (142545), (149175), (165491), (178699), (182325), (188437), (189109), (191565), (198123), (204183), (217107), (218323), (218415), (226395), (234329), (247537)])) w_7_0 w_25_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [29]) ((4/1)) (26) w_29_0 ∧
  outputs = (w_19_0, (w_30_0, ()))

def parallel_parallel_generatedRoot_7_50 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat1 × ByteArray × mxxMat1 × Unit) (outputs : mxxMat17 × mxxMat17 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_2_0 : ByteArray := inputs.2.1
  let w_6_0 : mxxMat1 := inputs.2.2.1
  ∃ (w_1_0 : mxxMat16),
    let w_3_0 : Int := (MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (65), (85), (99), (129), (135), (153), (165), (195), (255), (297), (325), (387), (459), (495), (561), (585), (645), (765), (891), (975), (1105), (1161), (1485), (1683), (1755), (1935), (2145), (2193), (2295), (2805), (2925), (3315), (3483), (4257), (4455), (5049), (5525), (5805), (6435), (6579), (8385), (8415), (8775), (9945), (10725), (10965), (12771), (14561), (15147), (16575), (17415), (19305), (19737), (21285), (22687), (25155), (25245), (27431), (29835), (32175), (32895), (33911), (36465), (36959), (38313), (41023), (41925), (43039), (43683), (49725), (50407), (57915), (59211), (63855), (66041), (68061), (70487), (72369), (72805), (75465), (75735), (83399), (96525), (98685), (99701), (107063), (109395), (110877), (114939), (115181), (118955), (123069), (125207), (125775), (127033), (129117), (129899), (130681), (130727), (130957), (131003), (131049), (142545), (149175), (165491), (178699), (182325), (188437), (189109), (191565), (198123), (204183), (217107), (218323), (218415), (226395), (234329), (247537)])
    ∃ (w_4_0 : mxxMat17),
      let w_5_0 : mxxMat17 := MxxRuntime.matrixMulScalarLeft w_1_0 w_4_0
      ∃ (w_7_0 : mxxMat16),
        ∃ (w_8_0 : mxxMat17),
          let w_9_0 : mxxMat17 := MxxRuntime.matrixMulScalarLeft w_7_0 w_8_0
          let w_10_0 : mxxMat17 := MxxRuntime.matrixAdd w_5_0 w_9_0
          let w_11_0 : mxxMat16 := (MxxRuntime.matrixPolynomial [(1099418042369)] : mxxMat16)
          ∃ (w_12_0 : mxxMat16),
            let w_13_0 : mxxMat16 := MxxRuntime.matrixMulScalarLeft w_11_0 w_12_0
            ∃ (w_14_0 : mxxMat18),
              ∃ (w_15_0 : mxxMat17),
                let w_16_0 : mxxMat17 := MxxRuntime.matrixMulScalarLeft w_13_0 w_15_0
                let w_17_0 : mxxMat17 := MxxRuntime.matrixSub w_10_0 w_16_0
                ∃ (w_18_0 : mxxMat17),
                  let w_19_0 : mxxMat17 := MxxRuntime.matrixAdd w_17_0 w_18_0
                  ∃ (w_20_0 : mxxMat17),
                    let w_21_0 : mxxMat17 := MxxRuntime.matrixMulScalarLeft w_1_0 w_20_0
                    ∃ (w_22_0 : mxxMat17),
                      let w_23_0 : mxxMat17 := MxxRuntime.matrixMulScalarLeft w_7_0 w_22_0
                      let w_24_0 : mxxMat17 := MxxRuntime.matrixAdd w_21_0 w_23_0
                      ∃ (w_25_0 : mxxMat16),
                        let w_26_0 : mxxMat16 := MxxRuntime.matrixMulScalarLeft w_11_0 w_25_0
                        let w_27_0 : mxxMat17 := MxxRuntime.matrixMulScalarLeft w_26_0 w_15_0
                        let w_28_0 : mxxMat17 := MxxRuntime.matrixSub w_24_0 w_27_0
                        ∃ (w_29_0 : mxxMat17),
                          let w_30_0 : mxxMat17 := MxxRuntime.matrixAdd w_28_0 w_29_0
                          parallel_parallel_generatedRoot_7_50.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs

abbrev parallel_parallel_generatedRoot_7_51.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_1 : Nat) (w_0_0 : mxxMat1) (w_2_0 : ByteArray) (w_6_0 : mxxMat1) (w_1_0 : mxxMat16) (w_3_0 : Int) (w_4_0 : mxxMat17) (w_7_0 : mxxMat16) (w_8_0 : mxxMat17) (w_12_0 : mxxMat16) (w_14_0 : mxxMat18) (w_15_0 : mxxMat17) (w_18_0 : mxxMat17) (w_19_0 : mxxMat17) (w_20_0 : mxxMat17) (w_22_0 : mxxMat17) (w_25_0 : mxxMat16) (w_29_0 : mxxMat17) (w_30_0 : mxxMat17) (outputs : mxxMat17 × mxxMat17 × Unit) : Prop :=
  MxxRuntime.modulusReduceRuns w_0_0 w_1_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 32 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (13), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_2_0) w_4_0 ∧
  MxxRuntime.modulusReduceRuns w_6_0 w_7_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (13), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_2_0) w_8_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 32 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (85), (99), (135), (153), (165), (255), (297), (459), (495), (561), (765), (891), (1485), (1683), (2295), (2805), (4455), (5049), (8415), (15147), (25245), (75735)])) w_1_0 w_12_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 1024 116 w_14_0 ∧
  MxxRuntime.centeredRebaseRuns w_14_0 w_15_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [18]) ((4/1)) (26) w_18_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (13), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_2_0) w_20_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (13), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_2_0) w_22_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 32 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (85), (99), (135), (153), (165), (255), (297), (459), (495), (561), (765), (891), (1485), (1683), (2295), (2805), (4455), (5049), (8415), (15147), (25245), (75735)])) w_7_0 w_25_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [29]) ((4/1)) (26) w_29_0 ∧
  outputs = (w_19_0, (w_30_0, ()))

def parallel_parallel_generatedRoot_7_51 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat1 × ByteArray × mxxMat1 × Unit) (outputs : mxxMat17 × mxxMat17 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_2_0 : ByteArray := inputs.2.1
  let w_6_0 : mxxMat1 := inputs.2.2.1
  ∃ (w_1_0 : mxxMat16),
    let w_3_0 : Int := (MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (85), (99), (135), (153), (165), (255), (297), (459), (495), (561), (765), (891), (1485), (1683), (2295), (2805), (4455), (5049), (8415), (15147), (25245), (75735)])
    ∃ (w_4_0 : mxxMat17),
      let w_5_0 : mxxMat17 := MxxRuntime.matrixMulScalarLeft w_1_0 w_4_0
      ∃ (w_7_0 : mxxMat16),
        ∃ (w_8_0 : mxxMat17),
          let w_9_0 : mxxMat17 := MxxRuntime.matrixMulScalarLeft w_7_0 w_8_0
          let w_10_0 : mxxMat17 := MxxRuntime.matrixAdd w_5_0 w_9_0
          let w_11_0 : mxxMat16 := (MxxRuntime.matrixPolynomial [(1099418042369)] : mxxMat16)
          ∃ (w_12_0 : mxxMat16),
            let w_13_0 : mxxMat16 := MxxRuntime.matrixMulScalarLeft w_11_0 w_12_0
            ∃ (w_14_0 : mxxMat18),
              ∃ (w_15_0 : mxxMat17),
                let w_16_0 : mxxMat17 := MxxRuntime.matrixMulScalarLeft w_13_0 w_15_0
                let w_17_0 : mxxMat17 := MxxRuntime.matrixSub w_10_0 w_16_0
                ∃ (w_18_0 : mxxMat17),
                  let w_19_0 : mxxMat17 := MxxRuntime.matrixAdd w_17_0 w_18_0
                  ∃ (w_20_0 : mxxMat17),
                    let w_21_0 : mxxMat17 := MxxRuntime.matrixMulScalarLeft w_1_0 w_20_0
                    ∃ (w_22_0 : mxxMat17),
                      let w_23_0 : mxxMat17 := MxxRuntime.matrixMulScalarLeft w_7_0 w_22_0
                      let w_24_0 : mxxMat17 := MxxRuntime.matrixAdd w_21_0 w_23_0
                      ∃ (w_25_0 : mxxMat16),
                        let w_26_0 : mxxMat16 := MxxRuntime.matrixMulScalarLeft w_11_0 w_25_0
                        let w_27_0 : mxxMat17 := MxxRuntime.matrixMulScalarLeft w_26_0 w_15_0
                        let w_28_0 : mxxMat17 := MxxRuntime.matrixSub w_24_0 w_27_0
                        ∃ (w_29_0 : mxxMat17),
                          let w_30_0 : mxxMat17 := MxxRuntime.matrixAdd w_28_0 w_29_0
                          parallel_parallel_generatedRoot_7_51.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs

abbrev parallel_parallel_generatedRoot_7_52.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_1 : Nat) (w_0_0 : mxxMat1) (w_2_0 : ByteArray) (w_6_0 : mxxMat1) (w_1_0 : mxxMat19) (w_3_0 : Int) (w_4_0 : mxxMat20) (w_7_0 : mxxMat19) (w_8_0 : mxxMat20) (w_12_0 : mxxMat19) (w_14_0 : mxxMat21) (w_15_0 : mxxMat20) (w_18_0 : mxxMat20) (w_19_0 : mxxMat20) (w_20_0 : mxxMat20) (w_22_0 : mxxMat20) (w_25_0 : mxxMat19) (w_29_0 : mxxMat20) (w_30_0 : mxxMat20) (outputs : mxxMat20 × mxxMat20 × Unit) : Prop :=
  MxxRuntime.modulusReduceRuns w_0_0 w_1_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 128 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (14), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_2_0) w_4_0 ∧
  MxxRuntime.modulusReduceRuns w_6_0 w_7_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (14), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_2_0) w_8_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 128 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (65), (85), (99), (129), (135), (153), (165), (195), (255), (297), (325), (387), (459), (495), (561), (585), (645), (765), (891), (975), (1105), (1161), (1485), (1683), (1755), (1935), (2145), (2193), (2295), (2805), (2925), (3315), (3483), (4257), (4455), (5049), (5525), (5805), (6435), (6579), (8385), (8415), (8775), (9945), (10725), (10965), (12771), (14561), (15147), (16575), (17415), (19305), (19737), (21285), (22687), (25155), (25245), (27431), (29835), (32175), (32895), (33911), (36465), (36959), (38313), (41023), (41925), (43039), (43683), (49725), (50407), (57915), (59211), (63855), (66041), (68061), (70487), (72369), (72805), (75465), (75735), (83399), (96525), (98685), (99701), (107063), (109395), (110877), (114939), (115181), (118955), (123069), (125207), (125775), (127033), (129117), (129899), (130681), (130727), (130957), (131003), (131049), (142545), (149175), (165491), (178699), (182325), (188437), (189109), (191565), (198123), (204183), (217107), (218323), (218415), (226395), (234329), (247537)])) w_1_0 w_12_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 1024 112 w_14_0 ∧
  MxxRuntime.centeredRebaseRuns w_14_0 w_15_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [18]) ((4/1)) (26) w_18_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (14), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_2_0) w_20_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (14), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_2_0) w_22_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 128 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (65), (85), (99), (129), (135), (153), (165), (195), (255), (297), (325), (387), (459), (495), (561), (585), (645), (765), (891), (975), (1105), (1161), (1485), (1683), (1755), (1935), (2145), (2193), (2295), (2805), (2925), (3315), (3483), (4257), (4455), (5049), (5525), (5805), (6435), (6579), (8385), (8415), (8775), (9945), (10725), (10965), (12771), (14561), (15147), (16575), (17415), (19305), (19737), (21285), (22687), (25155), (25245), (27431), (29835), (32175), (32895), (33911), (36465), (36959), (38313), (41023), (41925), (43039), (43683), (49725), (50407), (57915), (59211), (63855), (66041), (68061), (70487), (72369), (72805), (75465), (75735), (83399), (96525), (98685), (99701), (107063), (109395), (110877), (114939), (115181), (118955), (123069), (125207), (125775), (127033), (129117), (129899), (130681), (130727), (130957), (131003), (131049), (142545), (149175), (165491), (178699), (182325), (188437), (189109), (191565), (198123), (204183), (217107), (218323), (218415), (226395), (234329), (247537)])) w_7_0 w_25_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [29]) ((4/1)) (26) w_29_0 ∧
  outputs = (w_19_0, (w_30_0, ()))

def parallel_parallel_generatedRoot_7_52 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat1 × ByteArray × mxxMat1 × Unit) (outputs : mxxMat20 × mxxMat20 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_2_0 : ByteArray := inputs.2.1
  let w_6_0 : mxxMat1 := inputs.2.2.1
  ∃ (w_1_0 : mxxMat19),
    let w_3_0 : Int := (MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (65), (85), (99), (129), (135), (153), (165), (195), (255), (297), (325), (387), (459), (495), (561), (585), (645), (765), (891), (975), (1105), (1161), (1485), (1683), (1755), (1935), (2145), (2193), (2295), (2805), (2925), (3315), (3483), (4257), (4455), (5049), (5525), (5805), (6435), (6579), (8385), (8415), (8775), (9945), (10725), (10965), (12771), (14561), (15147), (16575), (17415), (19305), (19737), (21285), (22687), (25155), (25245), (27431), (29835), (32175), (32895), (33911), (36465), (36959), (38313), (41023), (41925), (43039), (43683), (49725), (50407), (57915), (59211), (63855), (66041), (68061), (70487), (72369), (72805), (75465), (75735), (83399), (96525), (98685), (99701), (107063), (109395), (110877), (114939), (115181), (118955), (123069), (125207), (125775), (127033), (129117), (129899), (130681), (130727), (130957), (131003), (131049), (142545), (149175), (165491), (178699), (182325), (188437), (189109), (191565), (198123), (204183), (217107), (218323), (218415), (226395), (234329), (247537)])
    ∃ (w_4_0 : mxxMat20),
      let w_5_0 : mxxMat20 := MxxRuntime.matrixMulScalarLeft w_1_0 w_4_0
      ∃ (w_7_0 : mxxMat19),
        ∃ (w_8_0 : mxxMat20),
          let w_9_0 : mxxMat20 := MxxRuntime.matrixMulScalarLeft w_7_0 w_8_0
          let w_10_0 : mxxMat20 := MxxRuntime.matrixAdd w_5_0 w_9_0
          let w_11_0 : mxxMat19 := (MxxRuntime.matrixPolynomial [(1099418828801)] : mxxMat19)
          ∃ (w_12_0 : mxxMat19),
            let w_13_0 : mxxMat19 := MxxRuntime.matrixMulScalarLeft w_11_0 w_12_0
            ∃ (w_14_0 : mxxMat21),
              ∃ (w_15_0 : mxxMat20),
                let w_16_0 : mxxMat20 := MxxRuntime.matrixMulScalarLeft w_13_0 w_15_0
                let w_17_0 : mxxMat20 := MxxRuntime.matrixSub w_10_0 w_16_0
                ∃ (w_18_0 : mxxMat20),
                  let w_19_0 : mxxMat20 := MxxRuntime.matrixAdd w_17_0 w_18_0
                  ∃ (w_20_0 : mxxMat20),
                    let w_21_0 : mxxMat20 := MxxRuntime.matrixMulScalarLeft w_1_0 w_20_0
                    ∃ (w_22_0 : mxxMat20),
                      let w_23_0 : mxxMat20 := MxxRuntime.matrixMulScalarLeft w_7_0 w_22_0
                      let w_24_0 : mxxMat20 := MxxRuntime.matrixAdd w_21_0 w_23_0
                      ∃ (w_25_0 : mxxMat19),
                        let w_26_0 : mxxMat19 := MxxRuntime.matrixMulScalarLeft w_11_0 w_25_0
                        let w_27_0 : mxxMat20 := MxxRuntime.matrixMulScalarLeft w_26_0 w_15_0
                        let w_28_0 : mxxMat20 := MxxRuntime.matrixSub w_24_0 w_27_0
                        ∃ (w_29_0 : mxxMat20),
                          let w_30_0 : mxxMat20 := MxxRuntime.matrixAdd w_28_0 w_29_0
                          parallel_parallel_generatedRoot_7_52.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs

abbrev parallel_parallel_generatedRoot_7_53.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_1 : Nat) (w_0_0 : mxxMat1) (w_2_0 : ByteArray) (w_6_0 : mxxMat1) (w_1_0 : mxxMat19) (w_3_0 : Int) (w_4_0 : mxxMat20) (w_7_0 : mxxMat19) (w_8_0 : mxxMat20) (w_12_0 : mxxMat19) (w_14_0 : mxxMat21) (w_15_0 : mxxMat20) (w_18_0 : mxxMat20) (w_19_0 : mxxMat20) (w_20_0 : mxxMat20) (w_22_0 : mxxMat20) (w_25_0 : mxxMat19) (w_29_0 : mxxMat20) (w_30_0 : mxxMat20) (outputs : mxxMat20 × mxxMat20 × Unit) : Prop :=
  MxxRuntime.modulusReduceRuns w_0_0 w_1_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 32 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (14), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_2_0) w_4_0 ∧
  MxxRuntime.modulusReduceRuns w_6_0 w_7_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (14), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_2_0) w_8_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 32 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (85), (99), (135), (153), (165), (255), (297), (459), (495), (561), (765), (891), (1485), (1683), (2295), (2805), (4455), (5049), (8415), (15147), (25245), (75735)])) w_1_0 w_12_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 1024 112 w_14_0 ∧
  MxxRuntime.centeredRebaseRuns w_14_0 w_15_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [18]) ((4/1)) (26) w_18_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (14), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_2_0) w_20_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (14), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_2_0) w_22_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 32 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (85), (99), (135), (153), (165), (255), (297), (459), (495), (561), (765), (891), (1485), (1683), (2295), (2805), (4455), (5049), (8415), (15147), (25245), (75735)])) w_7_0 w_25_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [29]) ((4/1)) (26) w_29_0 ∧
  outputs = (w_19_0, (w_30_0, ()))

def parallel_parallel_generatedRoot_7_53 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat1 × ByteArray × mxxMat1 × Unit) (outputs : mxxMat20 × mxxMat20 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_2_0 : ByteArray := inputs.2.1
  let w_6_0 : mxxMat1 := inputs.2.2.1
  ∃ (w_1_0 : mxxMat19),
    let w_3_0 : Int := (MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (85), (99), (135), (153), (165), (255), (297), (459), (495), (561), (765), (891), (1485), (1683), (2295), (2805), (4455), (5049), (8415), (15147), (25245), (75735)])
    ∃ (w_4_0 : mxxMat20),
      let w_5_0 : mxxMat20 := MxxRuntime.matrixMulScalarLeft w_1_0 w_4_0
      ∃ (w_7_0 : mxxMat19),
        ∃ (w_8_0 : mxxMat20),
          let w_9_0 : mxxMat20 := MxxRuntime.matrixMulScalarLeft w_7_0 w_8_0
          let w_10_0 : mxxMat20 := MxxRuntime.matrixAdd w_5_0 w_9_0
          let w_11_0 : mxxMat19 := (MxxRuntime.matrixPolynomial [(1099418828801)] : mxxMat19)
          ∃ (w_12_0 : mxxMat19),
            let w_13_0 : mxxMat19 := MxxRuntime.matrixMulScalarLeft w_11_0 w_12_0
            ∃ (w_14_0 : mxxMat21),
              ∃ (w_15_0 : mxxMat20),
                let w_16_0 : mxxMat20 := MxxRuntime.matrixMulScalarLeft w_13_0 w_15_0
                let w_17_0 : mxxMat20 := MxxRuntime.matrixSub w_10_0 w_16_0
                ∃ (w_18_0 : mxxMat20),
                  let w_19_0 : mxxMat20 := MxxRuntime.matrixAdd w_17_0 w_18_0
                  ∃ (w_20_0 : mxxMat20),
                    let w_21_0 : mxxMat20 := MxxRuntime.matrixMulScalarLeft w_1_0 w_20_0
                    ∃ (w_22_0 : mxxMat20),
                      let w_23_0 : mxxMat20 := MxxRuntime.matrixMulScalarLeft w_7_0 w_22_0
                      let w_24_0 : mxxMat20 := MxxRuntime.matrixAdd w_21_0 w_23_0
                      ∃ (w_25_0 : mxxMat19),
                        let w_26_0 : mxxMat19 := MxxRuntime.matrixMulScalarLeft w_11_0 w_25_0
                        let w_27_0 : mxxMat20 := MxxRuntime.matrixMulScalarLeft w_26_0 w_15_0
                        let w_28_0 : mxxMat20 := MxxRuntime.matrixSub w_24_0 w_27_0
                        ∃ (w_29_0 : mxxMat20),
                          let w_30_0 : mxxMat20 := MxxRuntime.matrixAdd w_28_0 w_29_0
                          parallel_parallel_generatedRoot_7_53.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs

abbrev parallel_parallel_generatedRoot_7_54.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_1 : Nat) (w_0_0 : mxxMat1) (w_2_0 : ByteArray) (w_6_0 : mxxMat1) (w_1_0 : mxxMat22) (w_3_0 : Int) (w_4_0 : mxxMat23) (w_7_0 : mxxMat22) (w_8_0 : mxxMat23) (w_12_0 : mxxMat22) (w_14_0 : mxxMat24) (w_15_0 : mxxMat23) (w_18_0 : mxxMat23) (w_19_0 : mxxMat23) (w_20_0 : mxxMat23) (w_22_0 : mxxMat23) (w_25_0 : mxxMat22) (w_29_0 : mxxMat23) (w_30_0 : mxxMat23) (outputs : mxxMat23 × mxxMat23 × Unit) : Prop :=
  MxxRuntime.modulusReduceRuns w_0_0 w_1_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 128 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (15), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_2_0) w_4_0 ∧
  MxxRuntime.modulusReduceRuns w_6_0 w_7_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (15), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_2_0) w_8_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 128 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (65), (85), (99), (129), (135), (153), (165), (195), (255), (297), (325), (387), (459), (495), (561), (585), (645), (765), (891), (975), (1105), (1161), (1485), (1683), (1755), (1935), (2145), (2193), (2295), (2805), (2925), (3315), (3483), (4257), (4455), (5049), (5525), (5805), (6435), (6579), (8385), (8415), (8775), (9945), (10725), (10965), (12771), (14561), (15147), (16575), (17415), (19305), (19737), (21285), (22687), (25155), (25245), (27431), (29835), (32175), (32895), (33911), (36465), (36959), (38313), (41023), (41925), (43039), (43683), (49725), (50407), (57915), (59211), (63855), (66041), (68061), (70487), (72369), (72805), (75465), (75735), (83399), (96525), (98685), (99701), (107063), (109395), (110877), (114939), (115181), (118955), (123069), (125207), (125775), (127033), (129117), (129899), (130681), (130727), (130957), (131003), (131049), (142545), (149175), (165491), (178699), (182325), (188437), (189109), (191565), (198123), (204183), (217107), (218323), (218415), (226395), (234329), (247537)])) w_1_0 w_12_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 1024 108 w_14_0 ∧
  MxxRuntime.centeredRebaseRuns w_14_0 w_15_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [18]) ((4/1)) (26) w_18_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (15), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_2_0) w_20_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (15), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_2_0) w_22_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 128 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (65), (85), (99), (129), (135), (153), (165), (195), (255), (297), (325), (387), (459), (495), (561), (585), (645), (765), (891), (975), (1105), (1161), (1485), (1683), (1755), (1935), (2145), (2193), (2295), (2805), (2925), (3315), (3483), (4257), (4455), (5049), (5525), (5805), (6435), (6579), (8385), (8415), (8775), (9945), (10725), (10965), (12771), (14561), (15147), (16575), (17415), (19305), (19737), (21285), (22687), (25155), (25245), (27431), (29835), (32175), (32895), (33911), (36465), (36959), (38313), (41023), (41925), (43039), (43683), (49725), (50407), (57915), (59211), (63855), (66041), (68061), (70487), (72369), (72805), (75465), (75735), (83399), (96525), (98685), (99701), (107063), (109395), (110877), (114939), (115181), (118955), (123069), (125207), (125775), (127033), (129117), (129899), (130681), (130727), (130957), (131003), (131049), (142545), (149175), (165491), (178699), (182325), (188437), (189109), (191565), (198123), (204183), (217107), (218323), (218415), (226395), (234329), (247537)])) w_7_0 w_25_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [29]) ((4/1)) (26) w_29_0 ∧
  outputs = (w_19_0, (w_30_0, ()))

def parallel_parallel_generatedRoot_7_54 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat1 × ByteArray × mxxMat1 × Unit) (outputs : mxxMat23 × mxxMat23 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_2_0 : ByteArray := inputs.2.1
  let w_6_0 : mxxMat1 := inputs.2.2.1
  ∃ (w_1_0 : mxxMat22),
    let w_3_0 : Int := (MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (65), (85), (99), (129), (135), (153), (165), (195), (255), (297), (325), (387), (459), (495), (561), (585), (645), (765), (891), (975), (1105), (1161), (1485), (1683), (1755), (1935), (2145), (2193), (2295), (2805), (2925), (3315), (3483), (4257), (4455), (5049), (5525), (5805), (6435), (6579), (8385), (8415), (8775), (9945), (10725), (10965), (12771), (14561), (15147), (16575), (17415), (19305), (19737), (21285), (22687), (25155), (25245), (27431), (29835), (32175), (32895), (33911), (36465), (36959), (38313), (41023), (41925), (43039), (43683), (49725), (50407), (57915), (59211), (63855), (66041), (68061), (70487), (72369), (72805), (75465), (75735), (83399), (96525), (98685), (99701), (107063), (109395), (110877), (114939), (115181), (118955), (123069), (125207), (125775), (127033), (129117), (129899), (130681), (130727), (130957), (131003), (131049), (142545), (149175), (165491), (178699), (182325), (188437), (189109), (191565), (198123), (204183), (217107), (218323), (218415), (226395), (234329), (247537)])
    ∃ (w_4_0 : mxxMat23),
      let w_5_0 : mxxMat23 := MxxRuntime.matrixMulScalarLeft w_1_0 w_4_0
      ∃ (w_7_0 : mxxMat22),
        ∃ (w_8_0 : mxxMat23),
          let w_9_0 : mxxMat23 := MxxRuntime.matrixMulScalarLeft w_7_0 w_8_0
          let w_10_0 : mxxMat23 := MxxRuntime.matrixAdd w_5_0 w_9_0
          let w_11_0 : mxxMat22 := (MxxRuntime.matrixPolynomial [(1099419353089)] : mxxMat22)
          ∃ (w_12_0 : mxxMat22),
            let w_13_0 : mxxMat22 := MxxRuntime.matrixMulScalarLeft w_11_0 w_12_0
            ∃ (w_14_0 : mxxMat24),
              ∃ (w_15_0 : mxxMat23),
                let w_16_0 : mxxMat23 := MxxRuntime.matrixMulScalarLeft w_13_0 w_15_0
                let w_17_0 : mxxMat23 := MxxRuntime.matrixSub w_10_0 w_16_0
                ∃ (w_18_0 : mxxMat23),
                  let w_19_0 : mxxMat23 := MxxRuntime.matrixAdd w_17_0 w_18_0
                  ∃ (w_20_0 : mxxMat23),
                    let w_21_0 : mxxMat23 := MxxRuntime.matrixMulScalarLeft w_1_0 w_20_0
                    ∃ (w_22_0 : mxxMat23),
                      let w_23_0 : mxxMat23 := MxxRuntime.matrixMulScalarLeft w_7_0 w_22_0
                      let w_24_0 : mxxMat23 := MxxRuntime.matrixAdd w_21_0 w_23_0
                      ∃ (w_25_0 : mxxMat22),
                        let w_26_0 : mxxMat22 := MxxRuntime.matrixMulScalarLeft w_11_0 w_25_0
                        let w_27_0 : mxxMat23 := MxxRuntime.matrixMulScalarLeft w_26_0 w_15_0
                        let w_28_0 : mxxMat23 := MxxRuntime.matrixSub w_24_0 w_27_0
                        ∃ (w_29_0 : mxxMat23),
                          let w_30_0 : mxxMat23 := MxxRuntime.matrixAdd w_28_0 w_29_0
                          parallel_parallel_generatedRoot_7_54.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs

abbrev parallel_parallel_generatedRoot_7_55.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_1 : Nat) (w_0_0 : mxxMat1) (w_2_0 : ByteArray) (w_6_0 : mxxMat1) (w_1_0 : mxxMat22) (w_3_0 : Int) (w_4_0 : mxxMat23) (w_7_0 : mxxMat22) (w_8_0 : mxxMat23) (w_12_0 : mxxMat22) (w_14_0 : mxxMat24) (w_15_0 : mxxMat23) (w_18_0 : mxxMat23) (w_19_0 : mxxMat23) (w_20_0 : mxxMat23) (w_22_0 : mxxMat23) (w_25_0 : mxxMat22) (w_29_0 : mxxMat23) (w_30_0 : mxxMat23) (outputs : mxxMat23 × mxxMat23 × Unit) : Prop :=
  MxxRuntime.modulusReduceRuns w_0_0 w_1_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 32 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (15), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_2_0) w_4_0 ∧
  MxxRuntime.modulusReduceRuns w_6_0 w_7_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (15), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_2_0) w_8_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 32 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (85), (99), (135), (153), (165), (255), (297), (459), (495), (561), (765), (891), (1485), (1683), (2295), (2805), (4455), (5049), (8415), (15147), (25245), (75735)])) w_1_0 w_12_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 1024 108 w_14_0 ∧
  MxxRuntime.centeredRebaseRuns w_14_0 w_15_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [18]) ((4/1)) (26) w_18_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (15), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_2_0) w_20_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (15), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_2_0) w_22_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 32 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (85), (99), (135), (153), (165), (255), (297), (459), (495), (561), (765), (891), (1485), (1683), (2295), (2805), (4455), (5049), (8415), (15147), (25245), (75735)])) w_7_0 w_25_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [29]) ((4/1)) (26) w_29_0 ∧
  outputs = (w_19_0, (w_30_0, ()))

def parallel_parallel_generatedRoot_7_55 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat1 × ByteArray × mxxMat1 × Unit) (outputs : mxxMat23 × mxxMat23 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_2_0 : ByteArray := inputs.2.1
  let w_6_0 : mxxMat1 := inputs.2.2.1
  ∃ (w_1_0 : mxxMat22),
    let w_3_0 : Int := (MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (85), (99), (135), (153), (165), (255), (297), (459), (495), (561), (765), (891), (1485), (1683), (2295), (2805), (4455), (5049), (8415), (15147), (25245), (75735)])
    ∃ (w_4_0 : mxxMat23),
      let w_5_0 : mxxMat23 := MxxRuntime.matrixMulScalarLeft w_1_0 w_4_0
      ∃ (w_7_0 : mxxMat22),
        ∃ (w_8_0 : mxxMat23),
          let w_9_0 : mxxMat23 := MxxRuntime.matrixMulScalarLeft w_7_0 w_8_0
          let w_10_0 : mxxMat23 := MxxRuntime.matrixAdd w_5_0 w_9_0
          let w_11_0 : mxxMat22 := (MxxRuntime.matrixPolynomial [(1099419353089)] : mxxMat22)
          ∃ (w_12_0 : mxxMat22),
            let w_13_0 : mxxMat22 := MxxRuntime.matrixMulScalarLeft w_11_0 w_12_0
            ∃ (w_14_0 : mxxMat24),
              ∃ (w_15_0 : mxxMat23),
                let w_16_0 : mxxMat23 := MxxRuntime.matrixMulScalarLeft w_13_0 w_15_0
                let w_17_0 : mxxMat23 := MxxRuntime.matrixSub w_10_0 w_16_0
                ∃ (w_18_0 : mxxMat23),
                  let w_19_0 : mxxMat23 := MxxRuntime.matrixAdd w_17_0 w_18_0
                  ∃ (w_20_0 : mxxMat23),
                    let w_21_0 : mxxMat23 := MxxRuntime.matrixMulScalarLeft w_1_0 w_20_0
                    ∃ (w_22_0 : mxxMat23),
                      let w_23_0 : mxxMat23 := MxxRuntime.matrixMulScalarLeft w_7_0 w_22_0
                      let w_24_0 : mxxMat23 := MxxRuntime.matrixAdd w_21_0 w_23_0
                      ∃ (w_25_0 : mxxMat22),
                        let w_26_0 : mxxMat22 := MxxRuntime.matrixMulScalarLeft w_11_0 w_25_0
                        let w_27_0 : mxxMat23 := MxxRuntime.matrixMulScalarLeft w_26_0 w_15_0
                        let w_28_0 : mxxMat23 := MxxRuntime.matrixSub w_24_0 w_27_0
                        ∃ (w_29_0 : mxxMat23),
                          let w_30_0 : mxxMat23 := MxxRuntime.matrixAdd w_28_0 w_29_0
                          parallel_parallel_generatedRoot_7_55.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs

abbrev parallel_parallel_generatedRoot_7_56.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_1 : Nat) (w_0_0 : mxxMat1) (w_2_0 : ByteArray) (w_6_0 : mxxMat1) (w_1_0 : mxxMat25) (w_3_0 : Int) (w_4_0 : mxxMat26) (w_7_0 : mxxMat25) (w_8_0 : mxxMat26) (w_12_0 : mxxMat25) (w_14_0 : mxxMat27) (w_15_0 : mxxMat26) (w_18_0 : mxxMat26) (w_19_0 : mxxMat26) (w_20_0 : mxxMat26) (w_22_0 : mxxMat26) (w_25_0 : mxxMat25) (w_29_0 : mxxMat26) (w_30_0 : mxxMat26) (outputs : mxxMat26 × mxxMat26 × Unit) : Prop :=
  MxxRuntime.modulusReduceRuns w_0_0 w_1_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 128 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (16), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_2_0) w_4_0 ∧
  MxxRuntime.modulusReduceRuns w_6_0 w_7_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (16), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_2_0) w_8_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 128 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (65), (85), (99), (129), (135), (153), (165), (195), (255), (297), (325), (387), (459), (495), (561), (585), (645), (765), (891), (975), (1105), (1161), (1485), (1683), (1755), (1935), (2145), (2193), (2295), (2805), (2925), (3315), (3483), (4257), (4455), (5049), (5525), (5805), (6435), (6579), (8385), (8415), (8775), (9945), (10725), (10965), (12771), (14561), (15147), (16575), (17415), (19305), (19737), (21285), (22687), (25155), (25245), (27431), (29835), (32175), (32895), (33911), (36465), (36959), (38313), (41023), (41925), (43039), (43683), (49725), (50407), (57915), (59211), (63855), (66041), (68061), (70487), (72369), (72805), (75465), (75735), (83399), (96525), (98685), (99701), (107063), (109395), (110877), (114939), (115181), (118955), (123069), (125207), (125775), (127033), (129117), (129899), (130681), (130727), (130957), (131003), (131049), (142545), (149175), (165491), (178699), (182325), (188437), (189109), (191565), (198123), (204183), (217107), (218323), (218415), (226395), (234329), (247537)])) w_1_0 w_12_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 1024 104 w_14_0 ∧
  MxxRuntime.centeredRebaseRuns w_14_0 w_15_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [18]) ((4/1)) (26) w_18_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (16), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_2_0) w_20_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (16), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_2_0) w_22_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 128 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (65), (85), (99), (129), (135), (153), (165), (195), (255), (297), (325), (387), (459), (495), (561), (585), (645), (765), (891), (975), (1105), (1161), (1485), (1683), (1755), (1935), (2145), (2193), (2295), (2805), (2925), (3315), (3483), (4257), (4455), (5049), (5525), (5805), (6435), (6579), (8385), (8415), (8775), (9945), (10725), (10965), (12771), (14561), (15147), (16575), (17415), (19305), (19737), (21285), (22687), (25155), (25245), (27431), (29835), (32175), (32895), (33911), (36465), (36959), (38313), (41023), (41925), (43039), (43683), (49725), (50407), (57915), (59211), (63855), (66041), (68061), (70487), (72369), (72805), (75465), (75735), (83399), (96525), (98685), (99701), (107063), (109395), (110877), (114939), (115181), (118955), (123069), (125207), (125775), (127033), (129117), (129899), (130681), (130727), (130957), (131003), (131049), (142545), (149175), (165491), (178699), (182325), (188437), (189109), (191565), (198123), (204183), (217107), (218323), (218415), (226395), (234329), (247537)])) w_7_0 w_25_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [29]) ((4/1)) (26) w_29_0 ∧
  outputs = (w_19_0, (w_30_0, ()))

def parallel_parallel_generatedRoot_7_56 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat1 × ByteArray × mxxMat1 × Unit) (outputs : mxxMat26 × mxxMat26 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_2_0 : ByteArray := inputs.2.1
  let w_6_0 : mxxMat1 := inputs.2.2.1
  ∃ (w_1_0 : mxxMat25),
    let w_3_0 : Int := (MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (65), (85), (99), (129), (135), (153), (165), (195), (255), (297), (325), (387), (459), (495), (561), (585), (645), (765), (891), (975), (1105), (1161), (1485), (1683), (1755), (1935), (2145), (2193), (2295), (2805), (2925), (3315), (3483), (4257), (4455), (5049), (5525), (5805), (6435), (6579), (8385), (8415), (8775), (9945), (10725), (10965), (12771), (14561), (15147), (16575), (17415), (19305), (19737), (21285), (22687), (25155), (25245), (27431), (29835), (32175), (32895), (33911), (36465), (36959), (38313), (41023), (41925), (43039), (43683), (49725), (50407), (57915), (59211), (63855), (66041), (68061), (70487), (72369), (72805), (75465), (75735), (83399), (96525), (98685), (99701), (107063), (109395), (110877), (114939), (115181), (118955), (123069), (125207), (125775), (127033), (129117), (129899), (130681), (130727), (130957), (131003), (131049), (142545), (149175), (165491), (178699), (182325), (188437), (189109), (191565), (198123), (204183), (217107), (218323), (218415), (226395), (234329), (247537)])
    ∃ (w_4_0 : mxxMat26),
      let w_5_0 : mxxMat26 := MxxRuntime.matrixMulScalarLeft w_1_0 w_4_0
      ∃ (w_7_0 : mxxMat25),
        ∃ (w_8_0 : mxxMat26),
          let w_9_0 : mxxMat26 := MxxRuntime.matrixMulScalarLeft w_7_0 w_8_0
          let w_10_0 : mxxMat26 := MxxRuntime.matrixAdd w_5_0 w_9_0
          let w_11_0 : mxxMat25 := (MxxRuntime.matrixPolynomial [(1099425120257)] : mxxMat25)
          ∃ (w_12_0 : mxxMat25),
            let w_13_0 : mxxMat25 := MxxRuntime.matrixMulScalarLeft w_11_0 w_12_0
            ∃ (w_14_0 : mxxMat27),
              ∃ (w_15_0 : mxxMat26),
                let w_16_0 : mxxMat26 := MxxRuntime.matrixMulScalarLeft w_13_0 w_15_0
                let w_17_0 : mxxMat26 := MxxRuntime.matrixSub w_10_0 w_16_0
                ∃ (w_18_0 : mxxMat26),
                  let w_19_0 : mxxMat26 := MxxRuntime.matrixAdd w_17_0 w_18_0
                  ∃ (w_20_0 : mxxMat26),
                    let w_21_0 : mxxMat26 := MxxRuntime.matrixMulScalarLeft w_1_0 w_20_0
                    ∃ (w_22_0 : mxxMat26),
                      let w_23_0 : mxxMat26 := MxxRuntime.matrixMulScalarLeft w_7_0 w_22_0
                      let w_24_0 : mxxMat26 := MxxRuntime.matrixAdd w_21_0 w_23_0
                      ∃ (w_25_0 : mxxMat25),
                        let w_26_0 : mxxMat25 := MxxRuntime.matrixMulScalarLeft w_11_0 w_25_0
                        let w_27_0 : mxxMat26 := MxxRuntime.matrixMulScalarLeft w_26_0 w_15_0
                        let w_28_0 : mxxMat26 := MxxRuntime.matrixSub w_24_0 w_27_0
                        ∃ (w_29_0 : mxxMat26),
                          let w_30_0 : mxxMat26 := MxxRuntime.matrixAdd w_28_0 w_29_0
                          parallel_parallel_generatedRoot_7_56.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs

abbrev parallel_parallel_generatedRoot_7_57.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_1 : Nat) (w_0_0 : mxxMat1) (w_2_0 : ByteArray) (w_6_0 : mxxMat1) (w_1_0 : mxxMat25) (w_3_0 : Int) (w_4_0 : mxxMat26) (w_7_0 : mxxMat25) (w_8_0 : mxxMat26) (w_12_0 : mxxMat25) (w_14_0 : mxxMat27) (w_15_0 : mxxMat26) (w_18_0 : mxxMat26) (w_19_0 : mxxMat26) (w_20_0 : mxxMat26) (w_22_0 : mxxMat26) (w_25_0 : mxxMat25) (w_29_0 : mxxMat26) (w_30_0 : mxxMat26) (outputs : mxxMat26 × mxxMat26 × Unit) : Prop :=
  MxxRuntime.modulusReduceRuns w_0_0 w_1_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 32 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (16), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_2_0) w_4_0 ∧
  MxxRuntime.modulusReduceRuns w_6_0 w_7_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (16), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_2_0) w_8_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 32 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (85), (99), (135), (153), (165), (255), (297), (459), (495), (561), (765), (891), (1485), (1683), (2295), (2805), (4455), (5049), (8415), (15147), (25245), (75735)])) w_1_0 w_12_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 1024 104 w_14_0 ∧
  MxxRuntime.centeredRebaseRuns w_14_0 w_15_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [18]) ((4/1)) (26) w_18_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (16), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_2_0) w_20_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (16), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_2_0) w_22_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 32 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (85), (99), (135), (153), (165), (255), (297), (459), (495), (561), (765), (891), (1485), (1683), (2295), (2805), (4455), (5049), (8415), (15147), (25245), (75735)])) w_7_0 w_25_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [29]) ((4/1)) (26) w_29_0 ∧
  outputs = (w_19_0, (w_30_0, ()))

def parallel_parallel_generatedRoot_7_57 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat1 × ByteArray × mxxMat1 × Unit) (outputs : mxxMat26 × mxxMat26 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_2_0 : ByteArray := inputs.2.1
  let w_6_0 : mxxMat1 := inputs.2.2.1
  ∃ (w_1_0 : mxxMat25),
    let w_3_0 : Int := (MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (85), (99), (135), (153), (165), (255), (297), (459), (495), (561), (765), (891), (1485), (1683), (2295), (2805), (4455), (5049), (8415), (15147), (25245), (75735)])
    ∃ (w_4_0 : mxxMat26),
      let w_5_0 : mxxMat26 := MxxRuntime.matrixMulScalarLeft w_1_0 w_4_0
      ∃ (w_7_0 : mxxMat25),
        ∃ (w_8_0 : mxxMat26),
          let w_9_0 : mxxMat26 := MxxRuntime.matrixMulScalarLeft w_7_0 w_8_0
          let w_10_0 : mxxMat26 := MxxRuntime.matrixAdd w_5_0 w_9_0
          let w_11_0 : mxxMat25 := (MxxRuntime.matrixPolynomial [(1099425120257)] : mxxMat25)
          ∃ (w_12_0 : mxxMat25),
            let w_13_0 : mxxMat25 := MxxRuntime.matrixMulScalarLeft w_11_0 w_12_0
            ∃ (w_14_0 : mxxMat27),
              ∃ (w_15_0 : mxxMat26),
                let w_16_0 : mxxMat26 := MxxRuntime.matrixMulScalarLeft w_13_0 w_15_0
                let w_17_0 : mxxMat26 := MxxRuntime.matrixSub w_10_0 w_16_0
                ∃ (w_18_0 : mxxMat26),
                  let w_19_0 : mxxMat26 := MxxRuntime.matrixAdd w_17_0 w_18_0
                  ∃ (w_20_0 : mxxMat26),
                    let w_21_0 : mxxMat26 := MxxRuntime.matrixMulScalarLeft w_1_0 w_20_0
                    ∃ (w_22_0 : mxxMat26),
                      let w_23_0 : mxxMat26 := MxxRuntime.matrixMulScalarLeft w_7_0 w_22_0
                      let w_24_0 : mxxMat26 := MxxRuntime.matrixAdd w_21_0 w_23_0
                      ∃ (w_25_0 : mxxMat25),
                        let w_26_0 : mxxMat25 := MxxRuntime.matrixMulScalarLeft w_11_0 w_25_0
                        let w_27_0 : mxxMat26 := MxxRuntime.matrixMulScalarLeft w_26_0 w_15_0
                        let w_28_0 : mxxMat26 := MxxRuntime.matrixSub w_24_0 w_27_0
                        ∃ (w_29_0 : mxxMat26),
                          let w_30_0 : mxxMat26 := MxxRuntime.matrixAdd w_28_0 w_29_0
                          parallel_parallel_generatedRoot_7_57.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs

abbrev parallel_parallel_generatedRoot_7_58.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_1 : Nat) (w_0_0 : mxxMat1) (w_2_0 : ByteArray) (w_6_0 : mxxMat1) (w_1_0 : mxxMat28) (w_3_0 : Int) (w_4_0 : mxxMat29) (w_7_0 : mxxMat28) (w_8_0 : mxxMat29) (w_12_0 : mxxMat28) (w_14_0 : mxxMat30) (w_15_0 : mxxMat29) (w_18_0 : mxxMat29) (w_19_0 : mxxMat29) (w_20_0 : mxxMat29) (w_22_0 : mxxMat29) (w_25_0 : mxxMat28) (w_29_0 : mxxMat29) (w_30_0 : mxxMat29) (outputs : mxxMat29 × mxxMat29 × Unit) : Prop :=
  MxxRuntime.modulusReduceRuns w_0_0 w_1_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 128 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (17), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_2_0) w_4_0 ∧
  MxxRuntime.modulusReduceRuns w_6_0 w_7_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (17), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_2_0) w_8_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 128 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (65), (85), (99), (129), (135), (153), (165), (195), (255), (297), (325), (387), (459), (495), (561), (585), (645), (765), (891), (975), (1105), (1161), (1485), (1683), (1755), (1935), (2145), (2193), (2295), (2805), (2925), (3315), (3483), (4257), (4455), (5049), (5525), (5805), (6435), (6579), (8385), (8415), (8775), (9945), (10725), (10965), (12771), (14561), (15147), (16575), (17415), (19305), (19737), (21285), (22687), (25155), (25245), (27431), (29835), (32175), (32895), (33911), (36465), (36959), (38313), (41023), (41925), (43039), (43683), (49725), (50407), (57915), (59211), (63855), (66041), (68061), (70487), (72369), (72805), (75465), (75735), (83399), (96525), (98685), (99701), (107063), (109395), (110877), (114939), (115181), (118955), (123069), (125207), (125775), (127033), (129117), (129899), (130681), (130727), (130957), (131003), (131049), (142545), (149175), (165491), (178699), (182325), (188437), (189109), (191565), (198123), (204183), (217107), (218323), (218415), (226395), (234329), (247537)])) w_1_0 w_12_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 1024 100 w_14_0 ∧
  MxxRuntime.centeredRebaseRuns w_14_0 w_15_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [18]) ((4/1)) (26) w_18_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (17), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_2_0) w_20_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (17), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_2_0) w_22_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 128 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (65), (85), (99), (129), (135), (153), (165), (195), (255), (297), (325), (387), (459), (495), (561), (585), (645), (765), (891), (975), (1105), (1161), (1485), (1683), (1755), (1935), (2145), (2193), (2295), (2805), (2925), (3315), (3483), (4257), (4455), (5049), (5525), (5805), (6435), (6579), (8385), (8415), (8775), (9945), (10725), (10965), (12771), (14561), (15147), (16575), (17415), (19305), (19737), (21285), (22687), (25155), (25245), (27431), (29835), (32175), (32895), (33911), (36465), (36959), (38313), (41023), (41925), (43039), (43683), (49725), (50407), (57915), (59211), (63855), (66041), (68061), (70487), (72369), (72805), (75465), (75735), (83399), (96525), (98685), (99701), (107063), (109395), (110877), (114939), (115181), (118955), (123069), (125207), (125775), (127033), (129117), (129899), (130681), (130727), (130957), (131003), (131049), (142545), (149175), (165491), (178699), (182325), (188437), (189109), (191565), (198123), (204183), (217107), (218323), (218415), (226395), (234329), (247537)])) w_7_0 w_25_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [29]) ((4/1)) (26) w_29_0 ∧
  outputs = (w_19_0, (w_30_0, ()))

def parallel_parallel_generatedRoot_7_58 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat1 × ByteArray × mxxMat1 × Unit) (outputs : mxxMat29 × mxxMat29 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_2_0 : ByteArray := inputs.2.1
  let w_6_0 : mxxMat1 := inputs.2.2.1
  ∃ (w_1_0 : mxxMat28),
    let w_3_0 : Int := (MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (65), (85), (99), (129), (135), (153), (165), (195), (255), (297), (325), (387), (459), (495), (561), (585), (645), (765), (891), (975), (1105), (1161), (1485), (1683), (1755), (1935), (2145), (2193), (2295), (2805), (2925), (3315), (3483), (4257), (4455), (5049), (5525), (5805), (6435), (6579), (8385), (8415), (8775), (9945), (10725), (10965), (12771), (14561), (15147), (16575), (17415), (19305), (19737), (21285), (22687), (25155), (25245), (27431), (29835), (32175), (32895), (33911), (36465), (36959), (38313), (41023), (41925), (43039), (43683), (49725), (50407), (57915), (59211), (63855), (66041), (68061), (70487), (72369), (72805), (75465), (75735), (83399), (96525), (98685), (99701), (107063), (109395), (110877), (114939), (115181), (118955), (123069), (125207), (125775), (127033), (129117), (129899), (130681), (130727), (130957), (131003), (131049), (142545), (149175), (165491), (178699), (182325), (188437), (189109), (191565), (198123), (204183), (217107), (218323), (218415), (226395), (234329), (247537)])
    ∃ (w_4_0 : mxxMat29),
      let w_5_0 : mxxMat29 := MxxRuntime.matrixMulScalarLeft w_1_0 w_4_0
      ∃ (w_7_0 : mxxMat28),
        ∃ (w_8_0 : mxxMat29),
          let w_9_0 : mxxMat29 := MxxRuntime.matrixMulScalarLeft w_7_0 w_8_0
          let w_10_0 : mxxMat29 := MxxRuntime.matrixAdd w_5_0 w_9_0
          let w_11_0 : mxxMat28 := (MxxRuntime.matrixPolynomial [(1099426693121)] : mxxMat28)
          ∃ (w_12_0 : mxxMat28),
            let w_13_0 : mxxMat28 := MxxRuntime.matrixMulScalarLeft w_11_0 w_12_0
            ∃ (w_14_0 : mxxMat30),
              ∃ (w_15_0 : mxxMat29),
                let w_16_0 : mxxMat29 := MxxRuntime.matrixMulScalarLeft w_13_0 w_15_0
                let w_17_0 : mxxMat29 := MxxRuntime.matrixSub w_10_0 w_16_0
                ∃ (w_18_0 : mxxMat29),
                  let w_19_0 : mxxMat29 := MxxRuntime.matrixAdd w_17_0 w_18_0
                  ∃ (w_20_0 : mxxMat29),
                    let w_21_0 : mxxMat29 := MxxRuntime.matrixMulScalarLeft w_1_0 w_20_0
                    ∃ (w_22_0 : mxxMat29),
                      let w_23_0 : mxxMat29 := MxxRuntime.matrixMulScalarLeft w_7_0 w_22_0
                      let w_24_0 : mxxMat29 := MxxRuntime.matrixAdd w_21_0 w_23_0
                      ∃ (w_25_0 : mxxMat28),
                        let w_26_0 : mxxMat28 := MxxRuntime.matrixMulScalarLeft w_11_0 w_25_0
                        let w_27_0 : mxxMat29 := MxxRuntime.matrixMulScalarLeft w_26_0 w_15_0
                        let w_28_0 : mxxMat29 := MxxRuntime.matrixSub w_24_0 w_27_0
                        ∃ (w_29_0 : mxxMat29),
                          let w_30_0 : mxxMat29 := MxxRuntime.matrixAdd w_28_0 w_29_0
                          parallel_parallel_generatedRoot_7_58.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs

abbrev parallel_parallel_generatedRoot_7_59.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_1 : Nat) (w_0_0 : mxxMat1) (w_2_0 : ByteArray) (w_6_0 : mxxMat1) (w_1_0 : mxxMat28) (w_3_0 : Int) (w_4_0 : mxxMat29) (w_7_0 : mxxMat28) (w_8_0 : mxxMat29) (w_12_0 : mxxMat28) (w_14_0 : mxxMat30) (w_15_0 : mxxMat29) (w_18_0 : mxxMat29) (w_19_0 : mxxMat29) (w_20_0 : mxxMat29) (w_22_0 : mxxMat29) (w_25_0 : mxxMat28) (w_29_0 : mxxMat29) (w_30_0 : mxxMat29) (outputs : mxxMat29 × mxxMat29 × Unit) : Prop :=
  MxxRuntime.modulusReduceRuns w_0_0 w_1_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 32 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (17), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_2_0) w_4_0 ∧
  MxxRuntime.modulusReduceRuns w_6_0 w_7_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (17), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_2_0) w_8_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 32 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (85), (99), (135), (153), (165), (255), (297), (459), (495), (561), (765), (891), (1485), (1683), (2295), (2805), (4455), (5049), (8415), (15147), (25245), (75735)])) w_1_0 w_12_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 1024 100 w_14_0 ∧
  MxxRuntime.centeredRebaseRuns w_14_0 w_15_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [18]) ((4/1)) (26) w_18_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (17), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_2_0) w_20_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (17), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_2_0) w_22_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 32 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (85), (99), (135), (153), (165), (255), (297), (459), (495), (561), (765), (891), (1485), (1683), (2295), (2805), (4455), (5049), (8415), (15147), (25245), (75735)])) w_7_0 w_25_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [29]) ((4/1)) (26) w_29_0 ∧
  outputs = (w_19_0, (w_30_0, ()))

def parallel_parallel_generatedRoot_7_59 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat1 × ByteArray × mxxMat1 × Unit) (outputs : mxxMat29 × mxxMat29 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_2_0 : ByteArray := inputs.2.1
  let w_6_0 : mxxMat1 := inputs.2.2.1
  ∃ (w_1_0 : mxxMat28),
    let w_3_0 : Int := (MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (85), (99), (135), (153), (165), (255), (297), (459), (495), (561), (765), (891), (1485), (1683), (2295), (2805), (4455), (5049), (8415), (15147), (25245), (75735)])
    ∃ (w_4_0 : mxxMat29),
      let w_5_0 : mxxMat29 := MxxRuntime.matrixMulScalarLeft w_1_0 w_4_0
      ∃ (w_7_0 : mxxMat28),
        ∃ (w_8_0 : mxxMat29),
          let w_9_0 : mxxMat29 := MxxRuntime.matrixMulScalarLeft w_7_0 w_8_0
          let w_10_0 : mxxMat29 := MxxRuntime.matrixAdd w_5_0 w_9_0
          let w_11_0 : mxxMat28 := (MxxRuntime.matrixPolynomial [(1099426693121)] : mxxMat28)
          ∃ (w_12_0 : mxxMat28),
            let w_13_0 : mxxMat28 := MxxRuntime.matrixMulScalarLeft w_11_0 w_12_0
            ∃ (w_14_0 : mxxMat30),
              ∃ (w_15_0 : mxxMat29),
                let w_16_0 : mxxMat29 := MxxRuntime.matrixMulScalarLeft w_13_0 w_15_0
                let w_17_0 : mxxMat29 := MxxRuntime.matrixSub w_10_0 w_16_0
                ∃ (w_18_0 : mxxMat29),
                  let w_19_0 : mxxMat29 := MxxRuntime.matrixAdd w_17_0 w_18_0
                  ∃ (w_20_0 : mxxMat29),
                    let w_21_0 : mxxMat29 := MxxRuntime.matrixMulScalarLeft w_1_0 w_20_0
                    ∃ (w_22_0 : mxxMat29),
                      let w_23_0 : mxxMat29 := MxxRuntime.matrixMulScalarLeft w_7_0 w_22_0
                      let w_24_0 : mxxMat29 := MxxRuntime.matrixAdd w_21_0 w_23_0
                      ∃ (w_25_0 : mxxMat28),
                        let w_26_0 : mxxMat28 := MxxRuntime.matrixMulScalarLeft w_11_0 w_25_0
                        let w_27_0 : mxxMat29 := MxxRuntime.matrixMulScalarLeft w_26_0 w_15_0
                        let w_28_0 : mxxMat29 := MxxRuntime.matrixSub w_24_0 w_27_0
                        ∃ (w_29_0 : mxxMat29),
                          let w_30_0 : mxxMat29 := MxxRuntime.matrixAdd w_28_0 w_29_0
                          parallel_parallel_generatedRoot_7_59.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs

abbrev parallel_parallel_generatedRoot_7_60.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_1 : Nat) (w_0_0 : mxxMat1) (w_2_0 : ByteArray) (w_6_0 : mxxMat1) (w_1_0 : mxxMat31) (w_3_0 : Int) (w_4_0 : mxxMat32) (w_7_0 : mxxMat31) (w_8_0 : mxxMat32) (w_12_0 : mxxMat31) (w_14_0 : mxxMat33) (w_15_0 : mxxMat32) (w_18_0 : mxxMat32) (w_19_0 : mxxMat32) (w_20_0 : mxxMat32) (w_22_0 : mxxMat32) (w_25_0 : mxxMat31) (w_29_0 : mxxMat32) (w_30_0 : mxxMat32) (outputs : mxxMat32 × mxxMat32 × Unit) : Prop :=
  MxxRuntime.modulusReduceRuns w_0_0 w_1_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 128 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (18), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_2_0) w_4_0 ∧
  MxxRuntime.modulusReduceRuns w_6_0 w_7_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (18), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_2_0) w_8_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 128 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (65), (85), (99), (129), (135), (153), (165), (195), (255), (297), (325), (387), (459), (495), (561), (585), (645), (765), (891), (975), (1105), (1161), (1485), (1683), (1755), (1935), (2145), (2193), (2295), (2805), (2925), (3315), (3483), (4257), (4455), (5049), (5525), (5805), (6435), (6579), (8385), (8415), (8775), (9945), (10725), (10965), (12771), (14561), (15147), (16575), (17415), (19305), (19737), (21285), (22687), (25155), (25245), (27431), (29835), (32175), (32895), (33911), (36465), (36959), (38313), (41023), (41925), (43039), (43683), (49725), (50407), (57915), (59211), (63855), (66041), (68061), (70487), (72369), (72805), (75465), (75735), (83399), (96525), (98685), (99701), (107063), (109395), (110877), (114939), (115181), (118955), (123069), (125207), (125775), (127033), (129117), (129899), (130681), (130727), (130957), (131003), (131049), (142545), (149175), (165491), (178699), (182325), (188437), (189109), (191565), (198123), (204183), (217107), (218323), (218415), (226395), (234329), (247537)])) w_1_0 w_12_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 1024 96 w_14_0 ∧
  MxxRuntime.centeredRebaseRuns w_14_0 w_15_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [18]) ((4/1)) (26) w_18_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (18), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_2_0) w_20_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (18), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_2_0) w_22_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 128 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (65), (85), (99), (129), (135), (153), (165), (195), (255), (297), (325), (387), (459), (495), (561), (585), (645), (765), (891), (975), (1105), (1161), (1485), (1683), (1755), (1935), (2145), (2193), (2295), (2805), (2925), (3315), (3483), (4257), (4455), (5049), (5525), (5805), (6435), (6579), (8385), (8415), (8775), (9945), (10725), (10965), (12771), (14561), (15147), (16575), (17415), (19305), (19737), (21285), (22687), (25155), (25245), (27431), (29835), (32175), (32895), (33911), (36465), (36959), (38313), (41023), (41925), (43039), (43683), (49725), (50407), (57915), (59211), (63855), (66041), (68061), (70487), (72369), (72805), (75465), (75735), (83399), (96525), (98685), (99701), (107063), (109395), (110877), (114939), (115181), (118955), (123069), (125207), (125775), (127033), (129117), (129899), (130681), (130727), (130957), (131003), (131049), (142545), (149175), (165491), (178699), (182325), (188437), (189109), (191565), (198123), (204183), (217107), (218323), (218415), (226395), (234329), (247537)])) w_7_0 w_25_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [29]) ((4/1)) (26) w_29_0 ∧
  outputs = (w_19_0, (w_30_0, ()))

def parallel_parallel_generatedRoot_7_60 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat1 × ByteArray × mxxMat1 × Unit) (outputs : mxxMat32 × mxxMat32 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_2_0 : ByteArray := inputs.2.1
  let w_6_0 : mxxMat1 := inputs.2.2.1
  ∃ (w_1_0 : mxxMat31),
    let w_3_0 : Int := (MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (65), (85), (99), (129), (135), (153), (165), (195), (255), (297), (325), (387), (459), (495), (561), (585), (645), (765), (891), (975), (1105), (1161), (1485), (1683), (1755), (1935), (2145), (2193), (2295), (2805), (2925), (3315), (3483), (4257), (4455), (5049), (5525), (5805), (6435), (6579), (8385), (8415), (8775), (9945), (10725), (10965), (12771), (14561), (15147), (16575), (17415), (19305), (19737), (21285), (22687), (25155), (25245), (27431), (29835), (32175), (32895), (33911), (36465), (36959), (38313), (41023), (41925), (43039), (43683), (49725), (50407), (57915), (59211), (63855), (66041), (68061), (70487), (72369), (72805), (75465), (75735), (83399), (96525), (98685), (99701), (107063), (109395), (110877), (114939), (115181), (118955), (123069), (125207), (125775), (127033), (129117), (129899), (130681), (130727), (130957), (131003), (131049), (142545), (149175), (165491), (178699), (182325), (188437), (189109), (191565), (198123), (204183), (217107), (218323), (218415), (226395), (234329), (247537)])
    ∃ (w_4_0 : mxxMat32),
      let w_5_0 : mxxMat32 := MxxRuntime.matrixMulScalarLeft w_1_0 w_4_0
      ∃ (w_7_0 : mxxMat31),
        ∃ (w_8_0 : mxxMat32),
          let w_9_0 : mxxMat32 := MxxRuntime.matrixMulScalarLeft w_7_0 w_8_0
          let w_10_0 : mxxMat32 := MxxRuntime.matrixAdd w_5_0 w_9_0
          let w_11_0 : mxxMat31 := (MxxRuntime.matrixPolynomial [(1099427217409)] : mxxMat31)
          ∃ (w_12_0 : mxxMat31),
            let w_13_0 : mxxMat31 := MxxRuntime.matrixMulScalarLeft w_11_0 w_12_0
            ∃ (w_14_0 : mxxMat33),
              ∃ (w_15_0 : mxxMat32),
                let w_16_0 : mxxMat32 := MxxRuntime.matrixMulScalarLeft w_13_0 w_15_0
                let w_17_0 : mxxMat32 := MxxRuntime.matrixSub w_10_0 w_16_0
                ∃ (w_18_0 : mxxMat32),
                  let w_19_0 : mxxMat32 := MxxRuntime.matrixAdd w_17_0 w_18_0
                  ∃ (w_20_0 : mxxMat32),
                    let w_21_0 : mxxMat32 := MxxRuntime.matrixMulScalarLeft w_1_0 w_20_0
                    ∃ (w_22_0 : mxxMat32),
                      let w_23_0 : mxxMat32 := MxxRuntime.matrixMulScalarLeft w_7_0 w_22_0
                      let w_24_0 : mxxMat32 := MxxRuntime.matrixAdd w_21_0 w_23_0
                      ∃ (w_25_0 : mxxMat31),
                        let w_26_0 : mxxMat31 := MxxRuntime.matrixMulScalarLeft w_11_0 w_25_0
                        let w_27_0 : mxxMat32 := MxxRuntime.matrixMulScalarLeft w_26_0 w_15_0
                        let w_28_0 : mxxMat32 := MxxRuntime.matrixSub w_24_0 w_27_0
                        ∃ (w_29_0 : mxxMat32),
                          let w_30_0 : mxxMat32 := MxxRuntime.matrixAdd w_28_0 w_29_0
                          parallel_parallel_generatedRoot_7_60.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs

abbrev parallel_parallel_generatedRoot_7_61.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_1 : Nat) (w_0_0 : mxxMat1) (w_2_0 : ByteArray) (w_6_0 : mxxMat1) (w_1_0 : mxxMat31) (w_3_0 : Int) (w_4_0 : mxxMat32) (w_7_0 : mxxMat31) (w_8_0 : mxxMat32) (w_12_0 : mxxMat31) (w_14_0 : mxxMat33) (w_15_0 : mxxMat32) (w_18_0 : mxxMat32) (w_19_0 : mxxMat32) (w_20_0 : mxxMat32) (w_22_0 : mxxMat32) (w_25_0 : mxxMat31) (w_29_0 : mxxMat32) (w_30_0 : mxxMat32) (outputs : mxxMat32 × mxxMat32 × Unit) : Prop :=
  MxxRuntime.modulusReduceRuns w_0_0 w_1_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 32 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (18), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_2_0) w_4_0 ∧
  MxxRuntime.modulusReduceRuns w_6_0 w_7_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (18), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_2_0) w_8_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 32 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (85), (99), (135), (153), (165), (255), (297), (459), (495), (561), (765), (891), (1485), (1683), (2295), (2805), (4455), (5049), (8415), (15147), (25245), (75735)])) w_1_0 w_12_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 1024 96 w_14_0 ∧
  MxxRuntime.centeredRebaseRuns w_14_0 w_15_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [18]) ((4/1)) (26) w_18_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (18), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_2_0) w_20_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (18), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_2_0) w_22_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 32 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (85), (99), (135), (153), (165), (255), (297), (459), (495), (561), (765), (891), (1485), (1683), (2295), (2805), (4455), (5049), (8415), (15147), (25245), (75735)])) w_7_0 w_25_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [29]) ((4/1)) (26) w_29_0 ∧
  outputs = (w_19_0, (w_30_0, ()))

def parallel_parallel_generatedRoot_7_61 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat1 × ByteArray × mxxMat1 × Unit) (outputs : mxxMat32 × mxxMat32 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_2_0 : ByteArray := inputs.2.1
  let w_6_0 : mxxMat1 := inputs.2.2.1
  ∃ (w_1_0 : mxxMat31),
    let w_3_0 : Int := (MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (85), (99), (135), (153), (165), (255), (297), (459), (495), (561), (765), (891), (1485), (1683), (2295), (2805), (4455), (5049), (8415), (15147), (25245), (75735)])
    ∃ (w_4_0 : mxxMat32),
      let w_5_0 : mxxMat32 := MxxRuntime.matrixMulScalarLeft w_1_0 w_4_0
      ∃ (w_7_0 : mxxMat31),
        ∃ (w_8_0 : mxxMat32),
          let w_9_0 : mxxMat32 := MxxRuntime.matrixMulScalarLeft w_7_0 w_8_0
          let w_10_0 : mxxMat32 := MxxRuntime.matrixAdd w_5_0 w_9_0
          let w_11_0 : mxxMat31 := (MxxRuntime.matrixPolynomial [(1099427217409)] : mxxMat31)
          ∃ (w_12_0 : mxxMat31),
            let w_13_0 : mxxMat31 := MxxRuntime.matrixMulScalarLeft w_11_0 w_12_0
            ∃ (w_14_0 : mxxMat33),
              ∃ (w_15_0 : mxxMat32),
                let w_16_0 : mxxMat32 := MxxRuntime.matrixMulScalarLeft w_13_0 w_15_0
                let w_17_0 : mxxMat32 := MxxRuntime.matrixSub w_10_0 w_16_0
                ∃ (w_18_0 : mxxMat32),
                  let w_19_0 : mxxMat32 := MxxRuntime.matrixAdd w_17_0 w_18_0
                  ∃ (w_20_0 : mxxMat32),
                    let w_21_0 : mxxMat32 := MxxRuntime.matrixMulScalarLeft w_1_0 w_20_0
                    ∃ (w_22_0 : mxxMat32),
                      let w_23_0 : mxxMat32 := MxxRuntime.matrixMulScalarLeft w_7_0 w_22_0
                      let w_24_0 : mxxMat32 := MxxRuntime.matrixAdd w_21_0 w_23_0
                      ∃ (w_25_0 : mxxMat31),
                        let w_26_0 : mxxMat31 := MxxRuntime.matrixMulScalarLeft w_11_0 w_25_0
                        let w_27_0 : mxxMat32 := MxxRuntime.matrixMulScalarLeft w_26_0 w_15_0
                        let w_28_0 : mxxMat32 := MxxRuntime.matrixSub w_24_0 w_27_0
                        ∃ (w_29_0 : mxxMat32),
                          let w_30_0 : mxxMat32 := MxxRuntime.matrixAdd w_28_0 w_29_0
                          parallel_parallel_generatedRoot_7_61.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs

abbrev parallel_parallel_generatedRoot_7_62.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_1 : Nat) (w_0_0 : mxxMat1) (w_2_0 : ByteArray) (w_6_0 : mxxMat1) (w_1_0 : mxxMat34) (w_3_0 : Int) (w_4_0 : mxxMat35) (w_7_0 : mxxMat34) (w_8_0 : mxxMat35) (w_12_0 : mxxMat34) (w_14_0 : mxxMat36) (w_15_0 : mxxMat35) (w_18_0 : mxxMat35) (w_19_0 : mxxMat35) (w_20_0 : mxxMat35) (w_22_0 : mxxMat35) (w_25_0 : mxxMat34) (w_29_0 : mxxMat35) (w_30_0 : mxxMat35) (outputs : mxxMat35 × mxxMat35 × Unit) : Prop :=
  MxxRuntime.modulusReduceRuns w_0_0 w_1_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 128 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (19), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_2_0) w_4_0 ∧
  MxxRuntime.modulusReduceRuns w_6_0 w_7_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (19), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_2_0) w_8_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 128 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (65), (85), (99), (129), (135), (153), (165), (195), (255), (297), (325), (387), (459), (495), (561), (585), (645), (765), (891), (975), (1105), (1161), (1485), (1683), (1755), (1935), (2145), (2193), (2295), (2805), (2925), (3315), (3483), (4257), (4455), (5049), (5525), (5805), (6435), (6579), (8385), (8415), (8775), (9945), (10725), (10965), (12771), (14561), (15147), (16575), (17415), (19305), (19737), (21285), (22687), (25155), (25245), (27431), (29835), (32175), (32895), (33911), (36465), (36959), (38313), (41023), (41925), (43039), (43683), (49725), (50407), (57915), (59211), (63855), (66041), (68061), (70487), (72369), (72805), (75465), (75735), (83399), (96525), (98685), (99701), (107063), (109395), (110877), (114939), (115181), (118955), (123069), (125207), (125775), (127033), (129117), (129899), (130681), (130727), (130957), (131003), (131049), (142545), (149175), (165491), (178699), (182325), (188437), (189109), (191565), (198123), (204183), (217107), (218323), (218415), (226395), (234329), (247537)])) w_1_0 w_12_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 1024 92 w_14_0 ∧
  MxxRuntime.centeredRebaseRuns w_14_0 w_15_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [18]) ((4/1)) (26) w_18_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (19), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_2_0) w_20_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (19), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_2_0) w_22_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 128 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (65), (85), (99), (129), (135), (153), (165), (195), (255), (297), (325), (387), (459), (495), (561), (585), (645), (765), (891), (975), (1105), (1161), (1485), (1683), (1755), (1935), (2145), (2193), (2295), (2805), (2925), (3315), (3483), (4257), (4455), (5049), (5525), (5805), (6435), (6579), (8385), (8415), (8775), (9945), (10725), (10965), (12771), (14561), (15147), (16575), (17415), (19305), (19737), (21285), (22687), (25155), (25245), (27431), (29835), (32175), (32895), (33911), (36465), (36959), (38313), (41023), (41925), (43039), (43683), (49725), (50407), (57915), (59211), (63855), (66041), (68061), (70487), (72369), (72805), (75465), (75735), (83399), (96525), (98685), (99701), (107063), (109395), (110877), (114939), (115181), (118955), (123069), (125207), (125775), (127033), (129117), (129899), (130681), (130727), (130957), (131003), (131049), (142545), (149175), (165491), (178699), (182325), (188437), (189109), (191565), (198123), (204183), (217107), (218323), (218415), (226395), (234329), (247537)])) w_7_0 w_25_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [29]) ((4/1)) (26) w_29_0 ∧
  outputs = (w_19_0, (w_30_0, ()))

def parallel_parallel_generatedRoot_7_62 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat1 × ByteArray × mxxMat1 × Unit) (outputs : mxxMat35 × mxxMat35 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_2_0 : ByteArray := inputs.2.1
  let w_6_0 : mxxMat1 := inputs.2.2.1
  ∃ (w_1_0 : mxxMat34),
    let w_3_0 : Int := (MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (65), (85), (99), (129), (135), (153), (165), (195), (255), (297), (325), (387), (459), (495), (561), (585), (645), (765), (891), (975), (1105), (1161), (1485), (1683), (1755), (1935), (2145), (2193), (2295), (2805), (2925), (3315), (3483), (4257), (4455), (5049), (5525), (5805), (6435), (6579), (8385), (8415), (8775), (9945), (10725), (10965), (12771), (14561), (15147), (16575), (17415), (19305), (19737), (21285), (22687), (25155), (25245), (27431), (29835), (32175), (32895), (33911), (36465), (36959), (38313), (41023), (41925), (43039), (43683), (49725), (50407), (57915), (59211), (63855), (66041), (68061), (70487), (72369), (72805), (75465), (75735), (83399), (96525), (98685), (99701), (107063), (109395), (110877), (114939), (115181), (118955), (123069), (125207), (125775), (127033), (129117), (129899), (130681), (130727), (130957), (131003), (131049), (142545), (149175), (165491), (178699), (182325), (188437), (189109), (191565), (198123), (204183), (217107), (218323), (218415), (226395), (234329), (247537)])
    ∃ (w_4_0 : mxxMat35),
      let w_5_0 : mxxMat35 := MxxRuntime.matrixMulScalarLeft w_1_0 w_4_0
      ∃ (w_7_0 : mxxMat34),
        ∃ (w_8_0 : mxxMat35),
          let w_9_0 : mxxMat35 := MxxRuntime.matrixMulScalarLeft w_7_0 w_8_0
          let w_10_0 : mxxMat35 := MxxRuntime.matrixAdd w_5_0 w_9_0
          let w_11_0 : mxxMat34 := (MxxRuntime.matrixPolynomial [(1099431149569)] : mxxMat34)
          ∃ (w_12_0 : mxxMat34),
            let w_13_0 : mxxMat34 := MxxRuntime.matrixMulScalarLeft w_11_0 w_12_0
            ∃ (w_14_0 : mxxMat36),
              ∃ (w_15_0 : mxxMat35),
                let w_16_0 : mxxMat35 := MxxRuntime.matrixMulScalarLeft w_13_0 w_15_0
                let w_17_0 : mxxMat35 := MxxRuntime.matrixSub w_10_0 w_16_0
                ∃ (w_18_0 : mxxMat35),
                  let w_19_0 : mxxMat35 := MxxRuntime.matrixAdd w_17_0 w_18_0
                  ∃ (w_20_0 : mxxMat35),
                    let w_21_0 : mxxMat35 := MxxRuntime.matrixMulScalarLeft w_1_0 w_20_0
                    ∃ (w_22_0 : mxxMat35),
                      let w_23_0 : mxxMat35 := MxxRuntime.matrixMulScalarLeft w_7_0 w_22_0
                      let w_24_0 : mxxMat35 := MxxRuntime.matrixAdd w_21_0 w_23_0
                      ∃ (w_25_0 : mxxMat34),
                        let w_26_0 : mxxMat34 := MxxRuntime.matrixMulScalarLeft w_11_0 w_25_0
                        let w_27_0 : mxxMat35 := MxxRuntime.matrixMulScalarLeft w_26_0 w_15_0
                        let w_28_0 : mxxMat35 := MxxRuntime.matrixSub w_24_0 w_27_0
                        ∃ (w_29_0 : mxxMat35),
                          let w_30_0 : mxxMat35 := MxxRuntime.matrixAdd w_28_0 w_29_0
                          parallel_parallel_generatedRoot_7_62.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs

abbrev parallel_parallel_generatedRoot_7_63.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_1 : Nat) (w_0_0 : mxxMat1) (w_2_0 : ByteArray) (w_6_0 : mxxMat1) (w_1_0 : mxxMat34) (w_3_0 : Int) (w_4_0 : mxxMat35) (w_7_0 : mxxMat34) (w_8_0 : mxxMat35) (w_12_0 : mxxMat34) (w_14_0 : mxxMat36) (w_15_0 : mxxMat35) (w_18_0 : mxxMat35) (w_19_0 : mxxMat35) (w_20_0 : mxxMat35) (w_22_0 : mxxMat35) (w_25_0 : mxxMat34) (w_29_0 : mxxMat35) (w_30_0 : mxxMat35) (outputs : mxxMat35 × mxxMat35 × Unit) : Prop :=
  MxxRuntime.modulusReduceRuns w_0_0 w_1_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 32 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (19), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_2_0) w_4_0 ∧
  MxxRuntime.modulusReduceRuns w_6_0 w_7_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (19), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_2_0) w_8_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 32 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (85), (99), (135), (153), (165), (255), (297), (459), (495), (561), (765), (891), (1485), (1683), (2295), (2805), (4455), (5049), (8415), (15147), (25245), (75735)])) w_1_0 w_12_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 1024 92 w_14_0 ∧
  MxxRuntime.centeredRebaseRuns w_14_0 w_15_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [18]) ((4/1)) (26) w_18_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (19), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_2_0) w_20_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (19), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_2_0) w_22_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 32 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (85), (99), (135), (153), (165), (255), (297), (459), (495), (561), (765), (891), (1485), (1683), (2295), (2805), (4455), (5049), (8415), (15147), (25245), (75735)])) w_7_0 w_25_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [29]) ((4/1)) (26) w_29_0 ∧
  outputs = (w_19_0, (w_30_0, ()))

def parallel_parallel_generatedRoot_7_63 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat1 × ByteArray × mxxMat1 × Unit) (outputs : mxxMat35 × mxxMat35 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_2_0 : ByteArray := inputs.2.1
  let w_6_0 : mxxMat1 := inputs.2.2.1
  ∃ (w_1_0 : mxxMat34),
    let w_3_0 : Int := (MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (85), (99), (135), (153), (165), (255), (297), (459), (495), (561), (765), (891), (1485), (1683), (2295), (2805), (4455), (5049), (8415), (15147), (25245), (75735)])
    ∃ (w_4_0 : mxxMat35),
      let w_5_0 : mxxMat35 := MxxRuntime.matrixMulScalarLeft w_1_0 w_4_0
      ∃ (w_7_0 : mxxMat34),
        ∃ (w_8_0 : mxxMat35),
          let w_9_0 : mxxMat35 := MxxRuntime.matrixMulScalarLeft w_7_0 w_8_0
          let w_10_0 : mxxMat35 := MxxRuntime.matrixAdd w_5_0 w_9_0
          let w_11_0 : mxxMat34 := (MxxRuntime.matrixPolynomial [(1099431149569)] : mxxMat34)
          ∃ (w_12_0 : mxxMat34),
            let w_13_0 : mxxMat34 := MxxRuntime.matrixMulScalarLeft w_11_0 w_12_0
            ∃ (w_14_0 : mxxMat36),
              ∃ (w_15_0 : mxxMat35),
                let w_16_0 : mxxMat35 := MxxRuntime.matrixMulScalarLeft w_13_0 w_15_0
                let w_17_0 : mxxMat35 := MxxRuntime.matrixSub w_10_0 w_16_0
                ∃ (w_18_0 : mxxMat35),
                  let w_19_0 : mxxMat35 := MxxRuntime.matrixAdd w_17_0 w_18_0
                  ∃ (w_20_0 : mxxMat35),
                    let w_21_0 : mxxMat35 := MxxRuntime.matrixMulScalarLeft w_1_0 w_20_0
                    ∃ (w_22_0 : mxxMat35),
                      let w_23_0 : mxxMat35 := MxxRuntime.matrixMulScalarLeft w_7_0 w_22_0
                      let w_24_0 : mxxMat35 := MxxRuntime.matrixAdd w_21_0 w_23_0
                      ∃ (w_25_0 : mxxMat34),
                        let w_26_0 : mxxMat34 := MxxRuntime.matrixMulScalarLeft w_11_0 w_25_0
                        let w_27_0 : mxxMat35 := MxxRuntime.matrixMulScalarLeft w_26_0 w_15_0
                        let w_28_0 : mxxMat35 := MxxRuntime.matrixSub w_24_0 w_27_0
                        ∃ (w_29_0 : mxxMat35),
                          let w_30_0 : mxxMat35 := MxxRuntime.matrixAdd w_28_0 w_29_0
                          parallel_parallel_generatedRoot_7_63.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs

abbrev parallel_parallel_generatedRoot_7_64.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_1 : Nat) (w_0_0 : mxxMat1) (w_2_0 : ByteArray) (w_6_0 : mxxMat1) (w_1_0 : mxxMat37) (w_3_0 : Int) (w_4_0 : mxxMat38) (w_7_0 : mxxMat37) (w_8_0 : mxxMat38) (w_12_0 : mxxMat37) (w_14_0 : mxxMat39) (w_15_0 : mxxMat38) (w_18_0 : mxxMat38) (w_19_0 : mxxMat38) (w_20_0 : mxxMat38) (w_22_0 : mxxMat38) (w_25_0 : mxxMat37) (w_29_0 : mxxMat38) (w_30_0 : mxxMat38) (outputs : mxxMat38 × mxxMat38 × Unit) : Prop :=
  MxxRuntime.modulusReduceRuns w_0_0 w_1_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 128 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (2), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_2_0) w_4_0 ∧
  MxxRuntime.modulusReduceRuns w_6_0 w_7_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (2), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_2_0) w_8_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 128 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (65), (85), (99), (129), (135), (153), (165), (195), (255), (297), (325), (387), (459), (495), (561), (585), (645), (765), (891), (975), (1105), (1161), (1485), (1683), (1755), (1935), (2145), (2193), (2295), (2805), (2925), (3315), (3483), (4257), (4455), (5049), (5525), (5805), (6435), (6579), (8385), (8415), (8775), (9945), (10725), (10965), (12771), (14561), (15147), (16575), (17415), (19305), (19737), (21285), (22687), (25155), (25245), (27431), (29835), (32175), (32895), (33911), (36465), (36959), (38313), (41023), (41925), (43039), (43683), (49725), (50407), (57915), (59211), (63855), (66041), (68061), (70487), (72369), (72805), (75465), (75735), (83399), (96525), (98685), (99701), (107063), (109395), (110877), (114939), (115181), (118955), (123069), (125207), (125775), (127033), (129117), (129899), (130681), (130727), (130957), (131003), (131049), (142545), (149175), (165491), (178699), (182325), (188437), (189109), (191565), (198123), (204183), (217107), (218323), (218415), (226395), (234329), (247537)])) w_1_0 w_12_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 1024 160 w_14_0 ∧
  MxxRuntime.centeredRebaseRuns w_14_0 w_15_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [18]) ((4/1)) (26) w_18_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (2), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_2_0) w_20_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (2), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_2_0) w_22_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 128 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (65), (85), (99), (129), (135), (153), (165), (195), (255), (297), (325), (387), (459), (495), (561), (585), (645), (765), (891), (975), (1105), (1161), (1485), (1683), (1755), (1935), (2145), (2193), (2295), (2805), (2925), (3315), (3483), (4257), (4455), (5049), (5525), (5805), (6435), (6579), (8385), (8415), (8775), (9945), (10725), (10965), (12771), (14561), (15147), (16575), (17415), (19305), (19737), (21285), (22687), (25155), (25245), (27431), (29835), (32175), (32895), (33911), (36465), (36959), (38313), (41023), (41925), (43039), (43683), (49725), (50407), (57915), (59211), (63855), (66041), (68061), (70487), (72369), (72805), (75465), (75735), (83399), (96525), (98685), (99701), (107063), (109395), (110877), (114939), (115181), (118955), (123069), (125207), (125775), (127033), (129117), (129899), (130681), (130727), (130957), (131003), (131049), (142545), (149175), (165491), (178699), (182325), (188437), (189109), (191565), (198123), (204183), (217107), (218323), (218415), (226395), (234329), (247537)])) w_7_0 w_25_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [29]) ((4/1)) (26) w_29_0 ∧
  outputs = (w_19_0, (w_30_0, ()))

def parallel_parallel_generatedRoot_7_64 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat1 × ByteArray × mxxMat1 × Unit) (outputs : mxxMat38 × mxxMat38 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_2_0 : ByteArray := inputs.2.1
  let w_6_0 : mxxMat1 := inputs.2.2.1
  ∃ (w_1_0 : mxxMat37),
    let w_3_0 : Int := (MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (65), (85), (99), (129), (135), (153), (165), (195), (255), (297), (325), (387), (459), (495), (561), (585), (645), (765), (891), (975), (1105), (1161), (1485), (1683), (1755), (1935), (2145), (2193), (2295), (2805), (2925), (3315), (3483), (4257), (4455), (5049), (5525), (5805), (6435), (6579), (8385), (8415), (8775), (9945), (10725), (10965), (12771), (14561), (15147), (16575), (17415), (19305), (19737), (21285), (22687), (25155), (25245), (27431), (29835), (32175), (32895), (33911), (36465), (36959), (38313), (41023), (41925), (43039), (43683), (49725), (50407), (57915), (59211), (63855), (66041), (68061), (70487), (72369), (72805), (75465), (75735), (83399), (96525), (98685), (99701), (107063), (109395), (110877), (114939), (115181), (118955), (123069), (125207), (125775), (127033), (129117), (129899), (130681), (130727), (130957), (131003), (131049), (142545), (149175), (165491), (178699), (182325), (188437), (189109), (191565), (198123), (204183), (217107), (218323), (218415), (226395), (234329), (247537)])
    ∃ (w_4_0 : mxxMat38),
      let w_5_0 : mxxMat38 := MxxRuntime.matrixMulScalarLeft w_1_0 w_4_0
      ∃ (w_7_0 : mxxMat37),
        ∃ (w_8_0 : mxxMat38),
          let w_9_0 : mxxMat38 := MxxRuntime.matrixMulScalarLeft w_7_0 w_8_0
          let w_10_0 : mxxMat38 := MxxRuntime.matrixAdd w_5_0 w_9_0
          let w_11_0 : mxxMat37 := (MxxRuntime.matrixPolynomial [(1099385798657)] : mxxMat37)
          ∃ (w_12_0 : mxxMat37),
            let w_13_0 : mxxMat37 := MxxRuntime.matrixMulScalarLeft w_11_0 w_12_0
            ∃ (w_14_0 : mxxMat39),
              ∃ (w_15_0 : mxxMat38),
                let w_16_0 : mxxMat38 := MxxRuntime.matrixMulScalarLeft w_13_0 w_15_0
                let w_17_0 : mxxMat38 := MxxRuntime.matrixSub w_10_0 w_16_0
                ∃ (w_18_0 : mxxMat38),
                  let w_19_0 : mxxMat38 := MxxRuntime.matrixAdd w_17_0 w_18_0
                  ∃ (w_20_0 : mxxMat38),
                    let w_21_0 : mxxMat38 := MxxRuntime.matrixMulScalarLeft w_1_0 w_20_0
                    ∃ (w_22_0 : mxxMat38),
                      let w_23_0 : mxxMat38 := MxxRuntime.matrixMulScalarLeft w_7_0 w_22_0
                      let w_24_0 : mxxMat38 := MxxRuntime.matrixAdd w_21_0 w_23_0
                      ∃ (w_25_0 : mxxMat37),
                        let w_26_0 : mxxMat37 := MxxRuntime.matrixMulScalarLeft w_11_0 w_25_0
                        let w_27_0 : mxxMat38 := MxxRuntime.matrixMulScalarLeft w_26_0 w_15_0
                        let w_28_0 : mxxMat38 := MxxRuntime.matrixSub w_24_0 w_27_0
                        ∃ (w_29_0 : mxxMat38),
                          let w_30_0 : mxxMat38 := MxxRuntime.matrixAdd w_28_0 w_29_0
                          parallel_parallel_generatedRoot_7_64.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs

abbrev parallel_parallel_generatedRoot_7_65.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_1 : Nat) (w_0_0 : mxxMat1) (w_2_0 : ByteArray) (w_6_0 : mxxMat1) (w_1_0 : mxxMat37) (w_3_0 : Int) (w_4_0 : mxxMat38) (w_7_0 : mxxMat37) (w_8_0 : mxxMat38) (w_12_0 : mxxMat37) (w_14_0 : mxxMat39) (w_15_0 : mxxMat38) (w_18_0 : mxxMat38) (w_19_0 : mxxMat38) (w_20_0 : mxxMat38) (w_22_0 : mxxMat38) (w_25_0 : mxxMat37) (w_29_0 : mxxMat38) (w_30_0 : mxxMat38) (outputs : mxxMat38 × mxxMat38 × Unit) : Prop :=
  MxxRuntime.modulusReduceRuns w_0_0 w_1_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 32 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (2), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_2_0) w_4_0 ∧
  MxxRuntime.modulusReduceRuns w_6_0 w_7_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (2), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_2_0) w_8_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 32 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (85), (99), (135), (153), (165), (255), (297), (459), (495), (561), (765), (891), (1485), (1683), (2295), (2805), (4455), (5049), (8415), (15147), (25245), (75735)])) w_1_0 w_12_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 1024 160 w_14_0 ∧
  MxxRuntime.centeredRebaseRuns w_14_0 w_15_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [18]) ((4/1)) (26) w_18_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (2), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_2_0) w_20_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (2), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_2_0) w_22_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 32 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (85), (99), (135), (153), (165), (255), (297), (459), (495), (561), (765), (891), (1485), (1683), (2295), (2805), (4455), (5049), (8415), (15147), (25245), (75735)])) w_7_0 w_25_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [29]) ((4/1)) (26) w_29_0 ∧
  outputs = (w_19_0, (w_30_0, ()))

def parallel_parallel_generatedRoot_7_65 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat1 × ByteArray × mxxMat1 × Unit) (outputs : mxxMat38 × mxxMat38 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_2_0 : ByteArray := inputs.2.1
  let w_6_0 : mxxMat1 := inputs.2.2.1
  ∃ (w_1_0 : mxxMat37),
    let w_3_0 : Int := (MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (85), (99), (135), (153), (165), (255), (297), (459), (495), (561), (765), (891), (1485), (1683), (2295), (2805), (4455), (5049), (8415), (15147), (25245), (75735)])
    ∃ (w_4_0 : mxxMat38),
      let w_5_0 : mxxMat38 := MxxRuntime.matrixMulScalarLeft w_1_0 w_4_0
      ∃ (w_7_0 : mxxMat37),
        ∃ (w_8_0 : mxxMat38),
          let w_9_0 : mxxMat38 := MxxRuntime.matrixMulScalarLeft w_7_0 w_8_0
          let w_10_0 : mxxMat38 := MxxRuntime.matrixAdd w_5_0 w_9_0
          let w_11_0 : mxxMat37 := (MxxRuntime.matrixPolynomial [(1099385798657)] : mxxMat37)
          ∃ (w_12_0 : mxxMat37),
            let w_13_0 : mxxMat37 := MxxRuntime.matrixMulScalarLeft w_11_0 w_12_0
            ∃ (w_14_0 : mxxMat39),
              ∃ (w_15_0 : mxxMat38),
                let w_16_0 : mxxMat38 := MxxRuntime.matrixMulScalarLeft w_13_0 w_15_0
                let w_17_0 : mxxMat38 := MxxRuntime.matrixSub w_10_0 w_16_0
                ∃ (w_18_0 : mxxMat38),
                  let w_19_0 : mxxMat38 := MxxRuntime.matrixAdd w_17_0 w_18_0
                  ∃ (w_20_0 : mxxMat38),
                    let w_21_0 : mxxMat38 := MxxRuntime.matrixMulScalarLeft w_1_0 w_20_0
                    ∃ (w_22_0 : mxxMat38),
                      let w_23_0 : mxxMat38 := MxxRuntime.matrixMulScalarLeft w_7_0 w_22_0
                      let w_24_0 : mxxMat38 := MxxRuntime.matrixAdd w_21_0 w_23_0
                      ∃ (w_25_0 : mxxMat37),
                        let w_26_0 : mxxMat37 := MxxRuntime.matrixMulScalarLeft w_11_0 w_25_0
                        let w_27_0 : mxxMat38 := MxxRuntime.matrixMulScalarLeft w_26_0 w_15_0
                        let w_28_0 : mxxMat38 := MxxRuntime.matrixSub w_24_0 w_27_0
                        ∃ (w_29_0 : mxxMat38),
                          let w_30_0 : mxxMat38 := MxxRuntime.matrixAdd w_28_0 w_29_0
                          parallel_parallel_generatedRoot_7_65.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs

abbrev parallel_parallel_generatedRoot_7_66.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_1 : Nat) (w_0_0 : mxxMat1) (w_2_0 : ByteArray) (w_6_0 : mxxMat1) (w_1_0 : mxxMat40) (w_3_0 : Int) (w_4_0 : mxxMat41) (w_7_0 : mxxMat40) (w_8_0 : mxxMat41) (w_12_0 : mxxMat40) (w_14_0 : mxxMat42) (w_15_0 : mxxMat41) (w_18_0 : mxxMat41) (w_19_0 : mxxMat41) (w_20_0 : mxxMat41) (w_22_0 : mxxMat41) (w_25_0 : mxxMat40) (w_29_0 : mxxMat41) (w_30_0 : mxxMat41) (outputs : mxxMat41 × mxxMat41 × Unit) : Prop :=
  MxxRuntime.modulusReduceRuns w_0_0 w_1_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 128 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (20), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_2_0) w_4_0 ∧
  MxxRuntime.modulusReduceRuns w_6_0 w_7_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (20), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_2_0) w_8_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 128 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (65), (85), (99), (129), (135), (153), (165), (195), (255), (297), (325), (387), (459), (495), (561), (585), (645), (765), (891), (975), (1105), (1161), (1485), (1683), (1755), (1935), (2145), (2193), (2295), (2805), (2925), (3315), (3483), (4257), (4455), (5049), (5525), (5805), (6435), (6579), (8385), (8415), (8775), (9945), (10725), (10965), (12771), (14561), (15147), (16575), (17415), (19305), (19737), (21285), (22687), (25155), (25245), (27431), (29835), (32175), (32895), (33911), (36465), (36959), (38313), (41023), (41925), (43039), (43683), (49725), (50407), (57915), (59211), (63855), (66041), (68061), (70487), (72369), (72805), (75465), (75735), (83399), (96525), (98685), (99701), (107063), (109395), (110877), (114939), (115181), (118955), (123069), (125207), (125775), (127033), (129117), (129899), (130681), (130727), (130957), (131003), (131049), (142545), (149175), (165491), (178699), (182325), (188437), (189109), (191565), (198123), (204183), (217107), (218323), (218415), (226395), (234329), (247537)])) w_1_0 w_12_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 1024 88 w_14_0 ∧
  MxxRuntime.centeredRebaseRuns w_14_0 w_15_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [18]) ((4/1)) (26) w_18_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (20), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_2_0) w_20_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (20), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_2_0) w_22_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 128 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (65), (85), (99), (129), (135), (153), (165), (195), (255), (297), (325), (387), (459), (495), (561), (585), (645), (765), (891), (975), (1105), (1161), (1485), (1683), (1755), (1935), (2145), (2193), (2295), (2805), (2925), (3315), (3483), (4257), (4455), (5049), (5525), (5805), (6435), (6579), (8385), (8415), (8775), (9945), (10725), (10965), (12771), (14561), (15147), (16575), (17415), (19305), (19737), (21285), (22687), (25155), (25245), (27431), (29835), (32175), (32895), (33911), (36465), (36959), (38313), (41023), (41925), (43039), (43683), (49725), (50407), (57915), (59211), (63855), (66041), (68061), (70487), (72369), (72805), (75465), (75735), (83399), (96525), (98685), (99701), (107063), (109395), (110877), (114939), (115181), (118955), (123069), (125207), (125775), (127033), (129117), (129899), (130681), (130727), (130957), (131003), (131049), (142545), (149175), (165491), (178699), (182325), (188437), (189109), (191565), (198123), (204183), (217107), (218323), (218415), (226395), (234329), (247537)])) w_7_0 w_25_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [29]) ((4/1)) (26) w_29_0 ∧
  outputs = (w_19_0, (w_30_0, ()))

def parallel_parallel_generatedRoot_7_66 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat1 × ByteArray × mxxMat1 × Unit) (outputs : mxxMat41 × mxxMat41 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_2_0 : ByteArray := inputs.2.1
  let w_6_0 : mxxMat1 := inputs.2.2.1
  ∃ (w_1_0 : mxxMat40),
    let w_3_0 : Int := (MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (65), (85), (99), (129), (135), (153), (165), (195), (255), (297), (325), (387), (459), (495), (561), (585), (645), (765), (891), (975), (1105), (1161), (1485), (1683), (1755), (1935), (2145), (2193), (2295), (2805), (2925), (3315), (3483), (4257), (4455), (5049), (5525), (5805), (6435), (6579), (8385), (8415), (8775), (9945), (10725), (10965), (12771), (14561), (15147), (16575), (17415), (19305), (19737), (21285), (22687), (25155), (25245), (27431), (29835), (32175), (32895), (33911), (36465), (36959), (38313), (41023), (41925), (43039), (43683), (49725), (50407), (57915), (59211), (63855), (66041), (68061), (70487), (72369), (72805), (75465), (75735), (83399), (96525), (98685), (99701), (107063), (109395), (110877), (114939), (115181), (118955), (123069), (125207), (125775), (127033), (129117), (129899), (130681), (130727), (130957), (131003), (131049), (142545), (149175), (165491), (178699), (182325), (188437), (189109), (191565), (198123), (204183), (217107), (218323), (218415), (226395), (234329), (247537)])
    ∃ (w_4_0 : mxxMat41),
      let w_5_0 : mxxMat41 := MxxRuntime.matrixMulScalarLeft w_1_0 w_4_0
      ∃ (w_7_0 : mxxMat40),
        ∃ (w_8_0 : mxxMat41),
          let w_9_0 : mxxMat41 := MxxRuntime.matrixMulScalarLeft w_7_0 w_8_0
          let w_10_0 : mxxMat41 := MxxRuntime.matrixAdd w_5_0 w_9_0
          let w_11_0 : mxxMat40 := (MxxRuntime.matrixPolynomial [(1099438227457)] : mxxMat40)
          ∃ (w_12_0 : mxxMat40),
            let w_13_0 : mxxMat40 := MxxRuntime.matrixMulScalarLeft w_11_0 w_12_0
            ∃ (w_14_0 : mxxMat42),
              ∃ (w_15_0 : mxxMat41),
                let w_16_0 : mxxMat41 := MxxRuntime.matrixMulScalarLeft w_13_0 w_15_0
                let w_17_0 : mxxMat41 := MxxRuntime.matrixSub w_10_0 w_16_0
                ∃ (w_18_0 : mxxMat41),
                  let w_19_0 : mxxMat41 := MxxRuntime.matrixAdd w_17_0 w_18_0
                  ∃ (w_20_0 : mxxMat41),
                    let w_21_0 : mxxMat41 := MxxRuntime.matrixMulScalarLeft w_1_0 w_20_0
                    ∃ (w_22_0 : mxxMat41),
                      let w_23_0 : mxxMat41 := MxxRuntime.matrixMulScalarLeft w_7_0 w_22_0
                      let w_24_0 : mxxMat41 := MxxRuntime.matrixAdd w_21_0 w_23_0
                      ∃ (w_25_0 : mxxMat40),
                        let w_26_0 : mxxMat40 := MxxRuntime.matrixMulScalarLeft w_11_0 w_25_0
                        let w_27_0 : mxxMat41 := MxxRuntime.matrixMulScalarLeft w_26_0 w_15_0
                        let w_28_0 : mxxMat41 := MxxRuntime.matrixSub w_24_0 w_27_0
                        ∃ (w_29_0 : mxxMat41),
                          let w_30_0 : mxxMat41 := MxxRuntime.matrixAdd w_28_0 w_29_0
                          parallel_parallel_generatedRoot_7_66.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs

abbrev parallel_parallel_generatedRoot_7_67.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_1 : Nat) (w_0_0 : mxxMat1) (w_2_0 : ByteArray) (w_6_0 : mxxMat1) (w_1_0 : mxxMat40) (w_3_0 : Int) (w_4_0 : mxxMat41) (w_7_0 : mxxMat40) (w_8_0 : mxxMat41) (w_12_0 : mxxMat40) (w_14_0 : mxxMat42) (w_15_0 : mxxMat41) (w_18_0 : mxxMat41) (w_19_0 : mxxMat41) (w_20_0 : mxxMat41) (w_22_0 : mxxMat41) (w_25_0 : mxxMat40) (w_29_0 : mxxMat41) (w_30_0 : mxxMat41) (outputs : mxxMat41 × mxxMat41 × Unit) : Prop :=
  MxxRuntime.modulusReduceRuns w_0_0 w_1_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 32 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (20), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_2_0) w_4_0 ∧
  MxxRuntime.modulusReduceRuns w_6_0 w_7_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (20), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_2_0) w_8_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 32 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (85), (99), (135), (153), (165), (255), (297), (459), (495), (561), (765), (891), (1485), (1683), (2295), (2805), (4455), (5049), (8415), (15147), (25245), (75735)])) w_1_0 w_12_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 1024 88 w_14_0 ∧
  MxxRuntime.centeredRebaseRuns w_14_0 w_15_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [18]) ((4/1)) (26) w_18_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (20), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_2_0) w_20_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (20), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_2_0) w_22_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 32 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (85), (99), (135), (153), (165), (255), (297), (459), (495), (561), (765), (891), (1485), (1683), (2295), (2805), (4455), (5049), (8415), (15147), (25245), (75735)])) w_7_0 w_25_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [29]) ((4/1)) (26) w_29_0 ∧
  outputs = (w_19_0, (w_30_0, ()))

def parallel_parallel_generatedRoot_7_67 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat1 × ByteArray × mxxMat1 × Unit) (outputs : mxxMat41 × mxxMat41 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_2_0 : ByteArray := inputs.2.1
  let w_6_0 : mxxMat1 := inputs.2.2.1
  ∃ (w_1_0 : mxxMat40),
    let w_3_0 : Int := (MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (85), (99), (135), (153), (165), (255), (297), (459), (495), (561), (765), (891), (1485), (1683), (2295), (2805), (4455), (5049), (8415), (15147), (25245), (75735)])
    ∃ (w_4_0 : mxxMat41),
      let w_5_0 : mxxMat41 := MxxRuntime.matrixMulScalarLeft w_1_0 w_4_0
      ∃ (w_7_0 : mxxMat40),
        ∃ (w_8_0 : mxxMat41),
          let w_9_0 : mxxMat41 := MxxRuntime.matrixMulScalarLeft w_7_0 w_8_0
          let w_10_0 : mxxMat41 := MxxRuntime.matrixAdd w_5_0 w_9_0
          let w_11_0 : mxxMat40 := (MxxRuntime.matrixPolynomial [(1099438227457)] : mxxMat40)
          ∃ (w_12_0 : mxxMat40),
            let w_13_0 : mxxMat40 := MxxRuntime.matrixMulScalarLeft w_11_0 w_12_0
            ∃ (w_14_0 : mxxMat42),
              ∃ (w_15_0 : mxxMat41),
                let w_16_0 : mxxMat41 := MxxRuntime.matrixMulScalarLeft w_13_0 w_15_0
                let w_17_0 : mxxMat41 := MxxRuntime.matrixSub w_10_0 w_16_0
                ∃ (w_18_0 : mxxMat41),
                  let w_19_0 : mxxMat41 := MxxRuntime.matrixAdd w_17_0 w_18_0
                  ∃ (w_20_0 : mxxMat41),
                    let w_21_0 : mxxMat41 := MxxRuntime.matrixMulScalarLeft w_1_0 w_20_0
                    ∃ (w_22_0 : mxxMat41),
                      let w_23_0 : mxxMat41 := MxxRuntime.matrixMulScalarLeft w_7_0 w_22_0
                      let w_24_0 : mxxMat41 := MxxRuntime.matrixAdd w_21_0 w_23_0
                      ∃ (w_25_0 : mxxMat40),
                        let w_26_0 : mxxMat40 := MxxRuntime.matrixMulScalarLeft w_11_0 w_25_0
                        let w_27_0 : mxxMat41 := MxxRuntime.matrixMulScalarLeft w_26_0 w_15_0
                        let w_28_0 : mxxMat41 := MxxRuntime.matrixSub w_24_0 w_27_0
                        ∃ (w_29_0 : mxxMat41),
                          let w_30_0 : mxxMat41 := MxxRuntime.matrixAdd w_28_0 w_29_0
                          parallel_parallel_generatedRoot_7_67.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs

abbrev parallel_parallel_generatedRoot_7_68.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_1 : Nat) (w_0_0 : mxxMat1) (w_2_0 : ByteArray) (w_6_0 : mxxMat1) (w_1_0 : mxxMat43) (w_3_0 : Int) (w_4_0 : mxxMat44) (w_7_0 : mxxMat43) (w_8_0 : mxxMat44) (w_12_0 : mxxMat43) (w_14_0 : mxxMat45) (w_15_0 : mxxMat44) (w_18_0 : mxxMat44) (w_19_0 : mxxMat44) (w_20_0 : mxxMat44) (w_22_0 : mxxMat44) (w_25_0 : mxxMat43) (w_29_0 : mxxMat44) (w_30_0 : mxxMat44) (outputs : mxxMat44 × mxxMat44 × Unit) : Prop :=
  MxxRuntime.modulusReduceRuns w_0_0 w_1_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 128 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (21), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_2_0) w_4_0 ∧
  MxxRuntime.modulusReduceRuns w_6_0 w_7_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (21), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_2_0) w_8_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 128 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (65), (85), (99), (129), (135), (153), (165), (195), (255), (297), (325), (387), (459), (495), (561), (585), (645), (765), (891), (975), (1105), (1161), (1485), (1683), (1755), (1935), (2145), (2193), (2295), (2805), (2925), (3315), (3483), (4257), (4455), (5049), (5525), (5805), (6435), (6579), (8385), (8415), (8775), (9945), (10725), (10965), (12771), (14561), (15147), (16575), (17415), (19305), (19737), (21285), (22687), (25155), (25245), (27431), (29835), (32175), (32895), (33911), (36465), (36959), (38313), (41023), (41925), (43039), (43683), (49725), (50407), (57915), (59211), (63855), (66041), (68061), (70487), (72369), (72805), (75465), (75735), (83399), (96525), (98685), (99701), (107063), (109395), (110877), (114939), (115181), (118955), (123069), (125207), (125775), (127033), (129117), (129899), (130681), (130727), (130957), (131003), (131049), (142545), (149175), (165491), (178699), (182325), (188437), (189109), (191565), (198123), (204183), (217107), (218323), (218415), (226395), (234329), (247537)])) w_1_0 w_12_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 1024 84 w_14_0 ∧
  MxxRuntime.centeredRebaseRuns w_14_0 w_15_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [18]) ((4/1)) (26) w_18_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (21), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_2_0) w_20_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (21), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_2_0) w_22_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 128 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (65), (85), (99), (129), (135), (153), (165), (195), (255), (297), (325), (387), (459), (495), (561), (585), (645), (765), (891), (975), (1105), (1161), (1485), (1683), (1755), (1935), (2145), (2193), (2295), (2805), (2925), (3315), (3483), (4257), (4455), (5049), (5525), (5805), (6435), (6579), (8385), (8415), (8775), (9945), (10725), (10965), (12771), (14561), (15147), (16575), (17415), (19305), (19737), (21285), (22687), (25155), (25245), (27431), (29835), (32175), (32895), (33911), (36465), (36959), (38313), (41023), (41925), (43039), (43683), (49725), (50407), (57915), (59211), (63855), (66041), (68061), (70487), (72369), (72805), (75465), (75735), (83399), (96525), (98685), (99701), (107063), (109395), (110877), (114939), (115181), (118955), (123069), (125207), (125775), (127033), (129117), (129899), (130681), (130727), (130957), (131003), (131049), (142545), (149175), (165491), (178699), (182325), (188437), (189109), (191565), (198123), (204183), (217107), (218323), (218415), (226395), (234329), (247537)])) w_7_0 w_25_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [29]) ((4/1)) (26) w_29_0 ∧
  outputs = (w_19_0, (w_30_0, ()))

def parallel_parallel_generatedRoot_7_68 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat1 × ByteArray × mxxMat1 × Unit) (outputs : mxxMat44 × mxxMat44 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_2_0 : ByteArray := inputs.2.1
  let w_6_0 : mxxMat1 := inputs.2.2.1
  ∃ (w_1_0 : mxxMat43),
    let w_3_0 : Int := (MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (65), (85), (99), (129), (135), (153), (165), (195), (255), (297), (325), (387), (459), (495), (561), (585), (645), (765), (891), (975), (1105), (1161), (1485), (1683), (1755), (1935), (2145), (2193), (2295), (2805), (2925), (3315), (3483), (4257), (4455), (5049), (5525), (5805), (6435), (6579), (8385), (8415), (8775), (9945), (10725), (10965), (12771), (14561), (15147), (16575), (17415), (19305), (19737), (21285), (22687), (25155), (25245), (27431), (29835), (32175), (32895), (33911), (36465), (36959), (38313), (41023), (41925), (43039), (43683), (49725), (50407), (57915), (59211), (63855), (66041), (68061), (70487), (72369), (72805), (75465), (75735), (83399), (96525), (98685), (99701), (107063), (109395), (110877), (114939), (115181), (118955), (123069), (125207), (125775), (127033), (129117), (129899), (130681), (130727), (130957), (131003), (131049), (142545), (149175), (165491), (178699), (182325), (188437), (189109), (191565), (198123), (204183), (217107), (218323), (218415), (226395), (234329), (247537)])
    ∃ (w_4_0 : mxxMat44),
      let w_5_0 : mxxMat44 := MxxRuntime.matrixMulScalarLeft w_1_0 w_4_0
      ∃ (w_7_0 : mxxMat43),
        ∃ (w_8_0 : mxxMat44),
          let w_9_0 : mxxMat44 := MxxRuntime.matrixMulScalarLeft w_7_0 w_8_0
          let w_10_0 : mxxMat44 := MxxRuntime.matrixAdd w_5_0 w_9_0
          let w_11_0 : mxxMat43 := (MxxRuntime.matrixPolynomial [(1099439800321)] : mxxMat43)
          ∃ (w_12_0 : mxxMat43),
            let w_13_0 : mxxMat43 := MxxRuntime.matrixMulScalarLeft w_11_0 w_12_0
            ∃ (w_14_0 : mxxMat45),
              ∃ (w_15_0 : mxxMat44),
                let w_16_0 : mxxMat44 := MxxRuntime.matrixMulScalarLeft w_13_0 w_15_0
                let w_17_0 : mxxMat44 := MxxRuntime.matrixSub w_10_0 w_16_0
                ∃ (w_18_0 : mxxMat44),
                  let w_19_0 : mxxMat44 := MxxRuntime.matrixAdd w_17_0 w_18_0
                  ∃ (w_20_0 : mxxMat44),
                    let w_21_0 : mxxMat44 := MxxRuntime.matrixMulScalarLeft w_1_0 w_20_0
                    ∃ (w_22_0 : mxxMat44),
                      let w_23_0 : mxxMat44 := MxxRuntime.matrixMulScalarLeft w_7_0 w_22_0
                      let w_24_0 : mxxMat44 := MxxRuntime.matrixAdd w_21_0 w_23_0
                      ∃ (w_25_0 : mxxMat43),
                        let w_26_0 : mxxMat43 := MxxRuntime.matrixMulScalarLeft w_11_0 w_25_0
                        let w_27_0 : mxxMat44 := MxxRuntime.matrixMulScalarLeft w_26_0 w_15_0
                        let w_28_0 : mxxMat44 := MxxRuntime.matrixSub w_24_0 w_27_0
                        ∃ (w_29_0 : mxxMat44),
                          let w_30_0 : mxxMat44 := MxxRuntime.matrixAdd w_28_0 w_29_0
                          parallel_parallel_generatedRoot_7_68.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs

abbrev parallel_parallel_generatedRoot_7_69.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_1 : Nat) (w_0_0 : mxxMat1) (w_2_0 : ByteArray) (w_6_0 : mxxMat1) (w_1_0 : mxxMat43) (w_3_0 : Int) (w_4_0 : mxxMat44) (w_7_0 : mxxMat43) (w_8_0 : mxxMat44) (w_12_0 : mxxMat43) (w_14_0 : mxxMat45) (w_15_0 : mxxMat44) (w_18_0 : mxxMat44) (w_19_0 : mxxMat44) (w_20_0 : mxxMat44) (w_22_0 : mxxMat44) (w_25_0 : mxxMat43) (w_29_0 : mxxMat44) (w_30_0 : mxxMat44) (outputs : mxxMat44 × mxxMat44 × Unit) : Prop :=
  MxxRuntime.modulusReduceRuns w_0_0 w_1_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 32 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (21), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_2_0) w_4_0 ∧
  MxxRuntime.modulusReduceRuns w_6_0 w_7_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (21), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_2_0) w_8_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 32 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (85), (99), (135), (153), (165), (255), (297), (459), (495), (561), (765), (891), (1485), (1683), (2295), (2805), (4455), (5049), (8415), (15147), (25245), (75735)])) w_1_0 w_12_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 1024 84 w_14_0 ∧
  MxxRuntime.centeredRebaseRuns w_14_0 w_15_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [18]) ((4/1)) (26) w_18_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (21), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_2_0) w_20_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (21), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_2_0) w_22_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 32 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (85), (99), (135), (153), (165), (255), (297), (459), (495), (561), (765), (891), (1485), (1683), (2295), (2805), (4455), (5049), (8415), (15147), (25245), (75735)])) w_7_0 w_25_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [29]) ((4/1)) (26) w_29_0 ∧
  outputs = (w_19_0, (w_30_0, ()))

def parallel_parallel_generatedRoot_7_69 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat1 × ByteArray × mxxMat1 × Unit) (outputs : mxxMat44 × mxxMat44 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_2_0 : ByteArray := inputs.2.1
  let w_6_0 : mxxMat1 := inputs.2.2.1
  ∃ (w_1_0 : mxxMat43),
    let w_3_0 : Int := (MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (85), (99), (135), (153), (165), (255), (297), (459), (495), (561), (765), (891), (1485), (1683), (2295), (2805), (4455), (5049), (8415), (15147), (25245), (75735)])
    ∃ (w_4_0 : mxxMat44),
      let w_5_0 : mxxMat44 := MxxRuntime.matrixMulScalarLeft w_1_0 w_4_0
      ∃ (w_7_0 : mxxMat43),
        ∃ (w_8_0 : mxxMat44),
          let w_9_0 : mxxMat44 := MxxRuntime.matrixMulScalarLeft w_7_0 w_8_0
          let w_10_0 : mxxMat44 := MxxRuntime.matrixAdd w_5_0 w_9_0
          let w_11_0 : mxxMat43 := (MxxRuntime.matrixPolynomial [(1099439800321)] : mxxMat43)
          ∃ (w_12_0 : mxxMat43),
            let w_13_0 : mxxMat43 := MxxRuntime.matrixMulScalarLeft w_11_0 w_12_0
            ∃ (w_14_0 : mxxMat45),
              ∃ (w_15_0 : mxxMat44),
                let w_16_0 : mxxMat44 := MxxRuntime.matrixMulScalarLeft w_13_0 w_15_0
                let w_17_0 : mxxMat44 := MxxRuntime.matrixSub w_10_0 w_16_0
                ∃ (w_18_0 : mxxMat44),
                  let w_19_0 : mxxMat44 := MxxRuntime.matrixAdd w_17_0 w_18_0
                  ∃ (w_20_0 : mxxMat44),
                    let w_21_0 : mxxMat44 := MxxRuntime.matrixMulScalarLeft w_1_0 w_20_0
                    ∃ (w_22_0 : mxxMat44),
                      let w_23_0 : mxxMat44 := MxxRuntime.matrixMulScalarLeft w_7_0 w_22_0
                      let w_24_0 : mxxMat44 := MxxRuntime.matrixAdd w_21_0 w_23_0
                      ∃ (w_25_0 : mxxMat43),
                        let w_26_0 : mxxMat43 := MxxRuntime.matrixMulScalarLeft w_11_0 w_25_0
                        let w_27_0 : mxxMat44 := MxxRuntime.matrixMulScalarLeft w_26_0 w_15_0
                        let w_28_0 : mxxMat44 := MxxRuntime.matrixSub w_24_0 w_27_0
                        ∃ (w_29_0 : mxxMat44),
                          let w_30_0 : mxxMat44 := MxxRuntime.matrixAdd w_28_0 w_29_0
                          parallel_parallel_generatedRoot_7_69.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs

abbrev parallel_parallel_generatedRoot_7_70.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_1 : Nat) (w_0_0 : mxxMat1) (w_2_0 : ByteArray) (w_6_0 : mxxMat1) (w_1_0 : mxxMat46) (w_3_0 : Int) (w_4_0 : mxxMat47) (w_7_0 : mxxMat46) (w_8_0 : mxxMat47) (w_12_0 : mxxMat46) (w_14_0 : mxxMat48) (w_15_0 : mxxMat47) (w_18_0 : mxxMat47) (w_19_0 : mxxMat47) (w_20_0 : mxxMat47) (w_22_0 : mxxMat47) (w_25_0 : mxxMat46) (w_29_0 : mxxMat47) (w_30_0 : mxxMat47) (outputs : mxxMat47 × mxxMat47 × Unit) : Prop :=
  MxxRuntime.modulusReduceRuns w_0_0 w_1_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 128 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (22), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_2_0) w_4_0 ∧
  MxxRuntime.modulusReduceRuns w_6_0 w_7_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (22), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_2_0) w_8_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 128 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (65), (85), (99), (129), (135), (153), (165), (195), (255), (297), (325), (387), (459), (495), (561), (585), (645), (765), (891), (975), (1105), (1161), (1485), (1683), (1755), (1935), (2145), (2193), (2295), (2805), (2925), (3315), (3483), (4257), (4455), (5049), (5525), (5805), (6435), (6579), (8385), (8415), (8775), (9945), (10725), (10965), (12771), (14561), (15147), (16575), (17415), (19305), (19737), (21285), (22687), (25155), (25245), (27431), (29835), (32175), (32895), (33911), (36465), (36959), (38313), (41023), (41925), (43039), (43683), (49725), (50407), (57915), (59211), (63855), (66041), (68061), (70487), (72369), (72805), (75465), (75735), (83399), (96525), (98685), (99701), (107063), (109395), (110877), (114939), (115181), (118955), (123069), (125207), (125775), (127033), (129117), (129899), (130681), (130727), (130957), (131003), (131049), (142545), (149175), (165491), (178699), (182325), (188437), (189109), (191565), (198123), (204183), (217107), (218323), (218415), (226395), (234329), (247537)])) w_1_0 w_12_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 1024 80 w_14_0 ∧
  MxxRuntime.centeredRebaseRuns w_14_0 w_15_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [18]) ((4/1)) (26) w_18_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (22), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_2_0) w_20_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (22), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_2_0) w_22_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 128 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (65), (85), (99), (129), (135), (153), (165), (195), (255), (297), (325), (387), (459), (495), (561), (585), (645), (765), (891), (975), (1105), (1161), (1485), (1683), (1755), (1935), (2145), (2193), (2295), (2805), (2925), (3315), (3483), (4257), (4455), (5049), (5525), (5805), (6435), (6579), (8385), (8415), (8775), (9945), (10725), (10965), (12771), (14561), (15147), (16575), (17415), (19305), (19737), (21285), (22687), (25155), (25245), (27431), (29835), (32175), (32895), (33911), (36465), (36959), (38313), (41023), (41925), (43039), (43683), (49725), (50407), (57915), (59211), (63855), (66041), (68061), (70487), (72369), (72805), (75465), (75735), (83399), (96525), (98685), (99701), (107063), (109395), (110877), (114939), (115181), (118955), (123069), (125207), (125775), (127033), (129117), (129899), (130681), (130727), (130957), (131003), (131049), (142545), (149175), (165491), (178699), (182325), (188437), (189109), (191565), (198123), (204183), (217107), (218323), (218415), (226395), (234329), (247537)])) w_7_0 w_25_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [29]) ((4/1)) (26) w_29_0 ∧
  outputs = (w_19_0, (w_30_0, ()))

def parallel_parallel_generatedRoot_7_70 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat1 × ByteArray × mxxMat1 × Unit) (outputs : mxxMat47 × mxxMat47 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_2_0 : ByteArray := inputs.2.1
  let w_6_0 : mxxMat1 := inputs.2.2.1
  ∃ (w_1_0 : mxxMat46),
    let w_3_0 : Int := (MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (65), (85), (99), (129), (135), (153), (165), (195), (255), (297), (325), (387), (459), (495), (561), (585), (645), (765), (891), (975), (1105), (1161), (1485), (1683), (1755), (1935), (2145), (2193), (2295), (2805), (2925), (3315), (3483), (4257), (4455), (5049), (5525), (5805), (6435), (6579), (8385), (8415), (8775), (9945), (10725), (10965), (12771), (14561), (15147), (16575), (17415), (19305), (19737), (21285), (22687), (25155), (25245), (27431), (29835), (32175), (32895), (33911), (36465), (36959), (38313), (41023), (41925), (43039), (43683), (49725), (50407), (57915), (59211), (63855), (66041), (68061), (70487), (72369), (72805), (75465), (75735), (83399), (96525), (98685), (99701), (107063), (109395), (110877), (114939), (115181), (118955), (123069), (125207), (125775), (127033), (129117), (129899), (130681), (130727), (130957), (131003), (131049), (142545), (149175), (165491), (178699), (182325), (188437), (189109), (191565), (198123), (204183), (217107), (218323), (218415), (226395), (234329), (247537)])
    ∃ (w_4_0 : mxxMat47),
      let w_5_0 : mxxMat47 := MxxRuntime.matrixMulScalarLeft w_1_0 w_4_0
      ∃ (w_7_0 : mxxMat46),
        ∃ (w_8_0 : mxxMat47),
          let w_9_0 : mxxMat47 := MxxRuntime.matrixMulScalarLeft w_7_0 w_8_0
          let w_10_0 : mxxMat47 := MxxRuntime.matrixAdd w_5_0 w_9_0
          let w_11_0 : mxxMat46 := (MxxRuntime.matrixPolynomial [(1099442159617)] : mxxMat46)
          ∃ (w_12_0 : mxxMat46),
            let w_13_0 : mxxMat46 := MxxRuntime.matrixMulScalarLeft w_11_0 w_12_0
            ∃ (w_14_0 : mxxMat48),
              ∃ (w_15_0 : mxxMat47),
                let w_16_0 : mxxMat47 := MxxRuntime.matrixMulScalarLeft w_13_0 w_15_0
                let w_17_0 : mxxMat47 := MxxRuntime.matrixSub w_10_0 w_16_0
                ∃ (w_18_0 : mxxMat47),
                  let w_19_0 : mxxMat47 := MxxRuntime.matrixAdd w_17_0 w_18_0
                  ∃ (w_20_0 : mxxMat47),
                    let w_21_0 : mxxMat47 := MxxRuntime.matrixMulScalarLeft w_1_0 w_20_0
                    ∃ (w_22_0 : mxxMat47),
                      let w_23_0 : mxxMat47 := MxxRuntime.matrixMulScalarLeft w_7_0 w_22_0
                      let w_24_0 : mxxMat47 := MxxRuntime.matrixAdd w_21_0 w_23_0
                      ∃ (w_25_0 : mxxMat46),
                        let w_26_0 : mxxMat46 := MxxRuntime.matrixMulScalarLeft w_11_0 w_25_0
                        let w_27_0 : mxxMat47 := MxxRuntime.matrixMulScalarLeft w_26_0 w_15_0
                        let w_28_0 : mxxMat47 := MxxRuntime.matrixSub w_24_0 w_27_0
                        ∃ (w_29_0 : mxxMat47),
                          let w_30_0 : mxxMat47 := MxxRuntime.matrixAdd w_28_0 w_29_0
                          parallel_parallel_generatedRoot_7_70.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs

abbrev parallel_parallel_generatedRoot_7_71.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_1 : Nat) (w_0_0 : mxxMat1) (w_2_0 : ByteArray) (w_6_0 : mxxMat1) (w_1_0 : mxxMat46) (w_3_0 : Int) (w_4_0 : mxxMat47) (w_7_0 : mxxMat46) (w_8_0 : mxxMat47) (w_12_0 : mxxMat46) (w_14_0 : mxxMat48) (w_15_0 : mxxMat47) (w_18_0 : mxxMat47) (w_19_0 : mxxMat47) (w_20_0 : mxxMat47) (w_22_0 : mxxMat47) (w_25_0 : mxxMat46) (w_29_0 : mxxMat47) (w_30_0 : mxxMat47) (outputs : mxxMat47 × mxxMat47 × Unit) : Prop :=
  MxxRuntime.modulusReduceRuns w_0_0 w_1_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 32 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (22), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_2_0) w_4_0 ∧
  MxxRuntime.modulusReduceRuns w_6_0 w_7_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (22), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_2_0) w_8_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 32 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (85), (99), (135), (153), (165), (255), (297), (459), (495), (561), (765), (891), (1485), (1683), (2295), (2805), (4455), (5049), (8415), (15147), (25245), (75735)])) w_1_0 w_12_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 1024 80 w_14_0 ∧
  MxxRuntime.centeredRebaseRuns w_14_0 w_15_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [18]) ((4/1)) (26) w_18_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (22), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_2_0) w_20_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (22), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_2_0) w_22_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 32 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (85), (99), (135), (153), (165), (255), (297), (459), (495), (561), (765), (891), (1485), (1683), (2295), (2805), (4455), (5049), (8415), (15147), (25245), (75735)])) w_7_0 w_25_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [29]) ((4/1)) (26) w_29_0 ∧
  outputs = (w_19_0, (w_30_0, ()))

def parallel_parallel_generatedRoot_7_71 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat1 × ByteArray × mxxMat1 × Unit) (outputs : mxxMat47 × mxxMat47 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_2_0 : ByteArray := inputs.2.1
  let w_6_0 : mxxMat1 := inputs.2.2.1
  ∃ (w_1_0 : mxxMat46),
    let w_3_0 : Int := (MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (85), (99), (135), (153), (165), (255), (297), (459), (495), (561), (765), (891), (1485), (1683), (2295), (2805), (4455), (5049), (8415), (15147), (25245), (75735)])
    ∃ (w_4_0 : mxxMat47),
      let w_5_0 : mxxMat47 := MxxRuntime.matrixMulScalarLeft w_1_0 w_4_0
      ∃ (w_7_0 : mxxMat46),
        ∃ (w_8_0 : mxxMat47),
          let w_9_0 : mxxMat47 := MxxRuntime.matrixMulScalarLeft w_7_0 w_8_0
          let w_10_0 : mxxMat47 := MxxRuntime.matrixAdd w_5_0 w_9_0
          let w_11_0 : mxxMat46 := (MxxRuntime.matrixPolynomial [(1099442159617)] : mxxMat46)
          ∃ (w_12_0 : mxxMat46),
            let w_13_0 : mxxMat46 := MxxRuntime.matrixMulScalarLeft w_11_0 w_12_0
            ∃ (w_14_0 : mxxMat48),
              ∃ (w_15_0 : mxxMat47),
                let w_16_0 : mxxMat47 := MxxRuntime.matrixMulScalarLeft w_13_0 w_15_0
                let w_17_0 : mxxMat47 := MxxRuntime.matrixSub w_10_0 w_16_0
                ∃ (w_18_0 : mxxMat47),
                  let w_19_0 : mxxMat47 := MxxRuntime.matrixAdd w_17_0 w_18_0
                  ∃ (w_20_0 : mxxMat47),
                    let w_21_0 : mxxMat47 := MxxRuntime.matrixMulScalarLeft w_1_0 w_20_0
                    ∃ (w_22_0 : mxxMat47),
                      let w_23_0 : mxxMat47 := MxxRuntime.matrixMulScalarLeft w_7_0 w_22_0
                      let w_24_0 : mxxMat47 := MxxRuntime.matrixAdd w_21_0 w_23_0
                      ∃ (w_25_0 : mxxMat46),
                        let w_26_0 : mxxMat46 := MxxRuntime.matrixMulScalarLeft w_11_0 w_25_0
                        let w_27_0 : mxxMat47 := MxxRuntime.matrixMulScalarLeft w_26_0 w_15_0
                        let w_28_0 : mxxMat47 := MxxRuntime.matrixSub w_24_0 w_27_0
                        ∃ (w_29_0 : mxxMat47),
                          let w_30_0 : mxxMat47 := MxxRuntime.matrixAdd w_28_0 w_29_0
                          parallel_parallel_generatedRoot_7_71.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs

abbrev parallel_parallel_generatedRoot_7_72.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_1 : Nat) (w_0_0 : mxxMat1) (w_2_0 : ByteArray) (w_6_0 : mxxMat1) (w_1_0 : mxxMat49) (w_3_0 : Int) (w_4_0 : mxxMat50) (w_7_0 : mxxMat49) (w_8_0 : mxxMat50) (w_12_0 : mxxMat49) (w_14_0 : mxxMat51) (w_15_0 : mxxMat50) (w_18_0 : mxxMat50) (w_19_0 : mxxMat50) (w_20_0 : mxxMat50) (w_22_0 : mxxMat50) (w_25_0 : mxxMat49) (w_29_0 : mxxMat50) (w_30_0 : mxxMat50) (outputs : mxxMat50 × mxxMat50 × Unit) : Prop :=
  MxxRuntime.modulusReduceRuns w_0_0 w_1_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 128 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (23), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_2_0) w_4_0 ∧
  MxxRuntime.modulusReduceRuns w_6_0 w_7_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (23), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_2_0) w_8_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 128 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (65), (85), (99), (129), (135), (153), (165), (195), (255), (297), (325), (387), (459), (495), (561), (585), (645), (765), (891), (975), (1105), (1161), (1485), (1683), (1755), (1935), (2145), (2193), (2295), (2805), (2925), (3315), (3483), (4257), (4455), (5049), (5525), (5805), (6435), (6579), (8385), (8415), (8775), (9945), (10725), (10965), (12771), (14561), (15147), (16575), (17415), (19305), (19737), (21285), (22687), (25155), (25245), (27431), (29835), (32175), (32895), (33911), (36465), (36959), (38313), (41023), (41925), (43039), (43683), (49725), (50407), (57915), (59211), (63855), (66041), (68061), (70487), (72369), (72805), (75465), (75735), (83399), (96525), (98685), (99701), (107063), (109395), (110877), (114939), (115181), (118955), (123069), (125207), (125775), (127033), (129117), (129899), (130681), (130727), (130957), (131003), (131049), (142545), (149175), (165491), (178699), (182325), (188437), (189109), (191565), (198123), (204183), (217107), (218323), (218415), (226395), (234329), (247537)])) w_1_0 w_12_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 1024 76 w_14_0 ∧
  MxxRuntime.centeredRebaseRuns w_14_0 w_15_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [18]) ((4/1)) (26) w_18_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (23), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_2_0) w_20_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (23), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_2_0) w_22_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 128 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (65), (85), (99), (129), (135), (153), (165), (195), (255), (297), (325), (387), (459), (495), (561), (585), (645), (765), (891), (975), (1105), (1161), (1485), (1683), (1755), (1935), (2145), (2193), (2295), (2805), (2925), (3315), (3483), (4257), (4455), (5049), (5525), (5805), (6435), (6579), (8385), (8415), (8775), (9945), (10725), (10965), (12771), (14561), (15147), (16575), (17415), (19305), (19737), (21285), (22687), (25155), (25245), (27431), (29835), (32175), (32895), (33911), (36465), (36959), (38313), (41023), (41925), (43039), (43683), (49725), (50407), (57915), (59211), (63855), (66041), (68061), (70487), (72369), (72805), (75465), (75735), (83399), (96525), (98685), (99701), (107063), (109395), (110877), (114939), (115181), (118955), (123069), (125207), (125775), (127033), (129117), (129899), (130681), (130727), (130957), (131003), (131049), (142545), (149175), (165491), (178699), (182325), (188437), (189109), (191565), (198123), (204183), (217107), (218323), (218415), (226395), (234329), (247537)])) w_7_0 w_25_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [29]) ((4/1)) (26) w_29_0 ∧
  outputs = (w_19_0, (w_30_0, ()))

def parallel_parallel_generatedRoot_7_72 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat1 × ByteArray × mxxMat1 × Unit) (outputs : mxxMat50 × mxxMat50 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_2_0 : ByteArray := inputs.2.1
  let w_6_0 : mxxMat1 := inputs.2.2.1
  ∃ (w_1_0 : mxxMat49),
    let w_3_0 : Int := (MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (65), (85), (99), (129), (135), (153), (165), (195), (255), (297), (325), (387), (459), (495), (561), (585), (645), (765), (891), (975), (1105), (1161), (1485), (1683), (1755), (1935), (2145), (2193), (2295), (2805), (2925), (3315), (3483), (4257), (4455), (5049), (5525), (5805), (6435), (6579), (8385), (8415), (8775), (9945), (10725), (10965), (12771), (14561), (15147), (16575), (17415), (19305), (19737), (21285), (22687), (25155), (25245), (27431), (29835), (32175), (32895), (33911), (36465), (36959), (38313), (41023), (41925), (43039), (43683), (49725), (50407), (57915), (59211), (63855), (66041), (68061), (70487), (72369), (72805), (75465), (75735), (83399), (96525), (98685), (99701), (107063), (109395), (110877), (114939), (115181), (118955), (123069), (125207), (125775), (127033), (129117), (129899), (130681), (130727), (130957), (131003), (131049), (142545), (149175), (165491), (178699), (182325), (188437), (189109), (191565), (198123), (204183), (217107), (218323), (218415), (226395), (234329), (247537)])
    ∃ (w_4_0 : mxxMat50),
      let w_5_0 : mxxMat50 := MxxRuntime.matrixMulScalarLeft w_1_0 w_4_0
      ∃ (w_7_0 : mxxMat49),
        ∃ (w_8_0 : mxxMat50),
          let w_9_0 : mxxMat50 := MxxRuntime.matrixMulScalarLeft w_7_0 w_8_0
          let w_10_0 : mxxMat50 := MxxRuntime.matrixAdd w_5_0 w_9_0
          let w_11_0 : mxxMat49 := (MxxRuntime.matrixPolynomial [(1099443732481)] : mxxMat49)
          ∃ (w_12_0 : mxxMat49),
            let w_13_0 : mxxMat49 := MxxRuntime.matrixMulScalarLeft w_11_0 w_12_0
            ∃ (w_14_0 : mxxMat51),
              ∃ (w_15_0 : mxxMat50),
                let w_16_0 : mxxMat50 := MxxRuntime.matrixMulScalarLeft w_13_0 w_15_0
                let w_17_0 : mxxMat50 := MxxRuntime.matrixSub w_10_0 w_16_0
                ∃ (w_18_0 : mxxMat50),
                  let w_19_0 : mxxMat50 := MxxRuntime.matrixAdd w_17_0 w_18_0
                  ∃ (w_20_0 : mxxMat50),
                    let w_21_0 : mxxMat50 := MxxRuntime.matrixMulScalarLeft w_1_0 w_20_0
                    ∃ (w_22_0 : mxxMat50),
                      let w_23_0 : mxxMat50 := MxxRuntime.matrixMulScalarLeft w_7_0 w_22_0
                      let w_24_0 : mxxMat50 := MxxRuntime.matrixAdd w_21_0 w_23_0
                      ∃ (w_25_0 : mxxMat49),
                        let w_26_0 : mxxMat49 := MxxRuntime.matrixMulScalarLeft w_11_0 w_25_0
                        let w_27_0 : mxxMat50 := MxxRuntime.matrixMulScalarLeft w_26_0 w_15_0
                        let w_28_0 : mxxMat50 := MxxRuntime.matrixSub w_24_0 w_27_0
                        ∃ (w_29_0 : mxxMat50),
                          let w_30_0 : mxxMat50 := MxxRuntime.matrixAdd w_28_0 w_29_0
                          parallel_parallel_generatedRoot_7_72.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs

abbrev parallel_parallel_generatedRoot_7_73.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_1 : Nat) (w_0_0 : mxxMat1) (w_2_0 : ByteArray) (w_6_0 : mxxMat1) (w_1_0 : mxxMat49) (w_3_0 : Int) (w_4_0 : mxxMat50) (w_7_0 : mxxMat49) (w_8_0 : mxxMat50) (w_12_0 : mxxMat49) (w_14_0 : mxxMat51) (w_15_0 : mxxMat50) (w_18_0 : mxxMat50) (w_19_0 : mxxMat50) (w_20_0 : mxxMat50) (w_22_0 : mxxMat50) (w_25_0 : mxxMat49) (w_29_0 : mxxMat50) (w_30_0 : mxxMat50) (outputs : mxxMat50 × mxxMat50 × Unit) : Prop :=
  MxxRuntime.modulusReduceRuns w_0_0 w_1_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 32 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (23), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_2_0) w_4_0 ∧
  MxxRuntime.modulusReduceRuns w_6_0 w_7_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (23), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_2_0) w_8_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 32 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (85), (99), (135), (153), (165), (255), (297), (459), (495), (561), (765), (891), (1485), (1683), (2295), (2805), (4455), (5049), (8415), (15147), (25245), (75735)])) w_1_0 w_12_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 1024 76 w_14_0 ∧
  MxxRuntime.centeredRebaseRuns w_14_0 w_15_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [18]) ((4/1)) (26) w_18_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (23), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_2_0) w_20_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (23), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_2_0) w_22_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 32 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (85), (99), (135), (153), (165), (255), (297), (459), (495), (561), (765), (891), (1485), (1683), (2295), (2805), (4455), (5049), (8415), (15147), (25245), (75735)])) w_7_0 w_25_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [29]) ((4/1)) (26) w_29_0 ∧
  outputs = (w_19_0, (w_30_0, ()))

def parallel_parallel_generatedRoot_7_73 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat1 × ByteArray × mxxMat1 × Unit) (outputs : mxxMat50 × mxxMat50 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_2_0 : ByteArray := inputs.2.1
  let w_6_0 : mxxMat1 := inputs.2.2.1
  ∃ (w_1_0 : mxxMat49),
    let w_3_0 : Int := (MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (85), (99), (135), (153), (165), (255), (297), (459), (495), (561), (765), (891), (1485), (1683), (2295), (2805), (4455), (5049), (8415), (15147), (25245), (75735)])
    ∃ (w_4_0 : mxxMat50),
      let w_5_0 : mxxMat50 := MxxRuntime.matrixMulScalarLeft w_1_0 w_4_0
      ∃ (w_7_0 : mxxMat49),
        ∃ (w_8_0 : mxxMat50),
          let w_9_0 : mxxMat50 := MxxRuntime.matrixMulScalarLeft w_7_0 w_8_0
          let w_10_0 : mxxMat50 := MxxRuntime.matrixAdd w_5_0 w_9_0
          let w_11_0 : mxxMat49 := (MxxRuntime.matrixPolynomial [(1099443732481)] : mxxMat49)
          ∃ (w_12_0 : mxxMat49),
            let w_13_0 : mxxMat49 := MxxRuntime.matrixMulScalarLeft w_11_0 w_12_0
            ∃ (w_14_0 : mxxMat51),
              ∃ (w_15_0 : mxxMat50),
                let w_16_0 : mxxMat50 := MxxRuntime.matrixMulScalarLeft w_13_0 w_15_0
                let w_17_0 : mxxMat50 := MxxRuntime.matrixSub w_10_0 w_16_0
                ∃ (w_18_0 : mxxMat50),
                  let w_19_0 : mxxMat50 := MxxRuntime.matrixAdd w_17_0 w_18_0
                  ∃ (w_20_0 : mxxMat50),
                    let w_21_0 : mxxMat50 := MxxRuntime.matrixMulScalarLeft w_1_0 w_20_0
                    ∃ (w_22_0 : mxxMat50),
                      let w_23_0 : mxxMat50 := MxxRuntime.matrixMulScalarLeft w_7_0 w_22_0
                      let w_24_0 : mxxMat50 := MxxRuntime.matrixAdd w_21_0 w_23_0
                      ∃ (w_25_0 : mxxMat49),
                        let w_26_0 : mxxMat49 := MxxRuntime.matrixMulScalarLeft w_11_0 w_25_0
                        let w_27_0 : mxxMat50 := MxxRuntime.matrixMulScalarLeft w_26_0 w_15_0
                        let w_28_0 : mxxMat50 := MxxRuntime.matrixSub w_24_0 w_27_0
                        ∃ (w_29_0 : mxxMat50),
                          let w_30_0 : mxxMat50 := MxxRuntime.matrixAdd w_28_0 w_29_0
                          parallel_parallel_generatedRoot_7_73.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs

abbrev parallel_parallel_generatedRoot_7_74.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_1 : Nat) (w_0_0 : mxxMat1) (w_2_0 : ByteArray) (w_6_0 : mxxMat1) (w_1_0 : mxxMat52) (w_3_0 : Int) (w_4_0 : mxxMat53) (w_7_0 : mxxMat52) (w_8_0 : mxxMat53) (w_12_0 : mxxMat52) (w_14_0 : mxxMat54) (w_15_0 : mxxMat53) (w_18_0 : mxxMat53) (w_19_0 : mxxMat53) (w_20_0 : mxxMat53) (w_22_0 : mxxMat53) (w_25_0 : mxxMat52) (w_29_0 : mxxMat53) (w_30_0 : mxxMat53) (outputs : mxxMat53 × mxxMat53 × Unit) : Prop :=
  MxxRuntime.modulusReduceRuns w_0_0 w_1_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 128 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (24), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_2_0) w_4_0 ∧
  MxxRuntime.modulusReduceRuns w_6_0 w_7_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (24), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_2_0) w_8_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 128 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (65), (85), (99), (129), (135), (153), (165), (195), (255), (297), (325), (387), (459), (495), (561), (585), (645), (765), (891), (975), (1105), (1161), (1485), (1683), (1755), (1935), (2145), (2193), (2295), (2805), (2925), (3315), (3483), (4257), (4455), (5049), (5525), (5805), (6435), (6579), (8385), (8415), (8775), (9945), (10725), (10965), (12771), (14561), (15147), (16575), (17415), (19305), (19737), (21285), (22687), (25155), (25245), (27431), (29835), (32175), (32895), (33911), (36465), (36959), (38313), (41023), (41925), (43039), (43683), (49725), (50407), (57915), (59211), (63855), (66041), (68061), (70487), (72369), (72805), (75465), (75735), (83399), (96525), (98685), (99701), (107063), (109395), (110877), (114939), (115181), (118955), (123069), (125207), (125775), (127033), (129117), (129899), (130681), (130727), (130957), (131003), (131049), (142545), (149175), (165491), (178699), (182325), (188437), (189109), (191565), (198123), (204183), (217107), (218323), (218415), (226395), (234329), (247537)])) w_1_0 w_12_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 1024 72 w_14_0 ∧
  MxxRuntime.centeredRebaseRuns w_14_0 w_15_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [18]) ((4/1)) (26) w_18_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (24), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_2_0) w_20_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (24), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_2_0) w_22_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 128 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (65), (85), (99), (129), (135), (153), (165), (195), (255), (297), (325), (387), (459), (495), (561), (585), (645), (765), (891), (975), (1105), (1161), (1485), (1683), (1755), (1935), (2145), (2193), (2295), (2805), (2925), (3315), (3483), (4257), (4455), (5049), (5525), (5805), (6435), (6579), (8385), (8415), (8775), (9945), (10725), (10965), (12771), (14561), (15147), (16575), (17415), (19305), (19737), (21285), (22687), (25155), (25245), (27431), (29835), (32175), (32895), (33911), (36465), (36959), (38313), (41023), (41925), (43039), (43683), (49725), (50407), (57915), (59211), (63855), (66041), (68061), (70487), (72369), (72805), (75465), (75735), (83399), (96525), (98685), (99701), (107063), (109395), (110877), (114939), (115181), (118955), (123069), (125207), (125775), (127033), (129117), (129899), (130681), (130727), (130957), (131003), (131049), (142545), (149175), (165491), (178699), (182325), (188437), (189109), (191565), (198123), (204183), (217107), (218323), (218415), (226395), (234329), (247537)])) w_7_0 w_25_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [29]) ((4/1)) (26) w_29_0 ∧
  outputs = (w_19_0, (w_30_0, ()))

def parallel_parallel_generatedRoot_7_74 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat1 × ByteArray × mxxMat1 × Unit) (outputs : mxxMat53 × mxxMat53 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_2_0 : ByteArray := inputs.2.1
  let w_6_0 : mxxMat1 := inputs.2.2.1
  ∃ (w_1_0 : mxxMat52),
    let w_3_0 : Int := (MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (65), (85), (99), (129), (135), (153), (165), (195), (255), (297), (325), (387), (459), (495), (561), (585), (645), (765), (891), (975), (1105), (1161), (1485), (1683), (1755), (1935), (2145), (2193), (2295), (2805), (2925), (3315), (3483), (4257), (4455), (5049), (5525), (5805), (6435), (6579), (8385), (8415), (8775), (9945), (10725), (10965), (12771), (14561), (15147), (16575), (17415), (19305), (19737), (21285), (22687), (25155), (25245), (27431), (29835), (32175), (32895), (33911), (36465), (36959), (38313), (41023), (41925), (43039), (43683), (49725), (50407), (57915), (59211), (63855), (66041), (68061), (70487), (72369), (72805), (75465), (75735), (83399), (96525), (98685), (99701), (107063), (109395), (110877), (114939), (115181), (118955), (123069), (125207), (125775), (127033), (129117), (129899), (130681), (130727), (130957), (131003), (131049), (142545), (149175), (165491), (178699), (182325), (188437), (189109), (191565), (198123), (204183), (217107), (218323), (218415), (226395), (234329), (247537)])
    ∃ (w_4_0 : mxxMat53),
      let w_5_0 : mxxMat53 := MxxRuntime.matrixMulScalarLeft w_1_0 w_4_0
      ∃ (w_7_0 : mxxMat52),
        ∃ (w_8_0 : mxxMat53),
          let w_9_0 : mxxMat53 := MxxRuntime.matrixMulScalarLeft w_7_0 w_8_0
          let w_10_0 : mxxMat53 := MxxRuntime.matrixAdd w_5_0 w_9_0
          let w_11_0 : mxxMat52 := (MxxRuntime.matrixPolynomial [(1099450023937)] : mxxMat52)
          ∃ (w_12_0 : mxxMat52),
            let w_13_0 : mxxMat52 := MxxRuntime.matrixMulScalarLeft w_11_0 w_12_0
            ∃ (w_14_0 : mxxMat54),
              ∃ (w_15_0 : mxxMat53),
                let w_16_0 : mxxMat53 := MxxRuntime.matrixMulScalarLeft w_13_0 w_15_0
                let w_17_0 : mxxMat53 := MxxRuntime.matrixSub w_10_0 w_16_0
                ∃ (w_18_0 : mxxMat53),
                  let w_19_0 : mxxMat53 := MxxRuntime.matrixAdd w_17_0 w_18_0
                  ∃ (w_20_0 : mxxMat53),
                    let w_21_0 : mxxMat53 := MxxRuntime.matrixMulScalarLeft w_1_0 w_20_0
                    ∃ (w_22_0 : mxxMat53),
                      let w_23_0 : mxxMat53 := MxxRuntime.matrixMulScalarLeft w_7_0 w_22_0
                      let w_24_0 : mxxMat53 := MxxRuntime.matrixAdd w_21_0 w_23_0
                      ∃ (w_25_0 : mxxMat52),
                        let w_26_0 : mxxMat52 := MxxRuntime.matrixMulScalarLeft w_11_0 w_25_0
                        let w_27_0 : mxxMat53 := MxxRuntime.matrixMulScalarLeft w_26_0 w_15_0
                        let w_28_0 : mxxMat53 := MxxRuntime.matrixSub w_24_0 w_27_0
                        ∃ (w_29_0 : mxxMat53),
                          let w_30_0 : mxxMat53 := MxxRuntime.matrixAdd w_28_0 w_29_0
                          parallel_parallel_generatedRoot_7_74.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs

abbrev parallel_parallel_generatedRoot_7_75.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_1 : Nat) (w_0_0 : mxxMat1) (w_2_0 : ByteArray) (w_6_0 : mxxMat1) (w_1_0 : mxxMat52) (w_3_0 : Int) (w_4_0 : mxxMat53) (w_7_0 : mxxMat52) (w_8_0 : mxxMat53) (w_12_0 : mxxMat52) (w_14_0 : mxxMat54) (w_15_0 : mxxMat53) (w_18_0 : mxxMat53) (w_19_0 : mxxMat53) (w_20_0 : mxxMat53) (w_22_0 : mxxMat53) (w_25_0 : mxxMat52) (w_29_0 : mxxMat53) (w_30_0 : mxxMat53) (outputs : mxxMat53 × mxxMat53 × Unit) : Prop :=
  MxxRuntime.modulusReduceRuns w_0_0 w_1_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 32 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (24), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_2_0) w_4_0 ∧
  MxxRuntime.modulusReduceRuns w_6_0 w_7_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (24), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_2_0) w_8_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 32 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (85), (99), (135), (153), (165), (255), (297), (459), (495), (561), (765), (891), (1485), (1683), (2295), (2805), (4455), (5049), (8415), (15147), (25245), (75735)])) w_1_0 w_12_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 1024 72 w_14_0 ∧
  MxxRuntime.centeredRebaseRuns w_14_0 w_15_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [18]) ((4/1)) (26) w_18_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (24), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_2_0) w_20_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (24), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_2_0) w_22_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 32 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (85), (99), (135), (153), (165), (255), (297), (459), (495), (561), (765), (891), (1485), (1683), (2295), (2805), (4455), (5049), (8415), (15147), (25245), (75735)])) w_7_0 w_25_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [29]) ((4/1)) (26) w_29_0 ∧
  outputs = (w_19_0, (w_30_0, ()))

def parallel_parallel_generatedRoot_7_75 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat1 × ByteArray × mxxMat1 × Unit) (outputs : mxxMat53 × mxxMat53 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_2_0 : ByteArray := inputs.2.1
  let w_6_0 : mxxMat1 := inputs.2.2.1
  ∃ (w_1_0 : mxxMat52),
    let w_3_0 : Int := (MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (85), (99), (135), (153), (165), (255), (297), (459), (495), (561), (765), (891), (1485), (1683), (2295), (2805), (4455), (5049), (8415), (15147), (25245), (75735)])
    ∃ (w_4_0 : mxxMat53),
      let w_5_0 : mxxMat53 := MxxRuntime.matrixMulScalarLeft w_1_0 w_4_0
      ∃ (w_7_0 : mxxMat52),
        ∃ (w_8_0 : mxxMat53),
          let w_9_0 : mxxMat53 := MxxRuntime.matrixMulScalarLeft w_7_0 w_8_0
          let w_10_0 : mxxMat53 := MxxRuntime.matrixAdd w_5_0 w_9_0
          let w_11_0 : mxxMat52 := (MxxRuntime.matrixPolynomial [(1099450023937)] : mxxMat52)
          ∃ (w_12_0 : mxxMat52),
            let w_13_0 : mxxMat52 := MxxRuntime.matrixMulScalarLeft w_11_0 w_12_0
            ∃ (w_14_0 : mxxMat54),
              ∃ (w_15_0 : mxxMat53),
                let w_16_0 : mxxMat53 := MxxRuntime.matrixMulScalarLeft w_13_0 w_15_0
                let w_17_0 : mxxMat53 := MxxRuntime.matrixSub w_10_0 w_16_0
                ∃ (w_18_0 : mxxMat53),
                  let w_19_0 : mxxMat53 := MxxRuntime.matrixAdd w_17_0 w_18_0
                  ∃ (w_20_0 : mxxMat53),
                    let w_21_0 : mxxMat53 := MxxRuntime.matrixMulScalarLeft w_1_0 w_20_0
                    ∃ (w_22_0 : mxxMat53),
                      let w_23_0 : mxxMat53 := MxxRuntime.matrixMulScalarLeft w_7_0 w_22_0
                      let w_24_0 : mxxMat53 := MxxRuntime.matrixAdd w_21_0 w_23_0
                      ∃ (w_25_0 : mxxMat52),
                        let w_26_0 : mxxMat52 := MxxRuntime.matrixMulScalarLeft w_11_0 w_25_0
                        let w_27_0 : mxxMat53 := MxxRuntime.matrixMulScalarLeft w_26_0 w_15_0
                        let w_28_0 : mxxMat53 := MxxRuntime.matrixSub w_24_0 w_27_0
                        ∃ (w_29_0 : mxxMat53),
                          let w_30_0 : mxxMat53 := MxxRuntime.matrixAdd w_28_0 w_29_0
                          parallel_parallel_generatedRoot_7_75.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs

abbrev parallel_parallel_generatedRoot_7_76.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_1 : Nat) (w_0_0 : mxxMat1) (w_2_0 : ByteArray) (w_6_0 : mxxMat1) (w_1_0 : mxxMat55) (w_3_0 : Int) (w_4_0 : mxxMat56) (w_7_0 : mxxMat55) (w_8_0 : mxxMat56) (w_12_0 : mxxMat55) (w_14_0 : mxxMat57) (w_15_0 : mxxMat56) (w_18_0 : mxxMat56) (w_19_0 : mxxMat56) (w_20_0 : mxxMat56) (w_22_0 : mxxMat56) (w_25_0 : mxxMat55) (w_29_0 : mxxMat56) (w_30_0 : mxxMat56) (outputs : mxxMat56 × mxxMat56 × Unit) : Prop :=
  MxxRuntime.modulusReduceRuns w_0_0 w_1_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 128 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (25), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_2_0) w_4_0 ∧
  MxxRuntime.modulusReduceRuns w_6_0 w_7_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (25), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_2_0) w_8_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 128 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (65), (85), (99), (129), (135), (153), (165), (195), (255), (297), (325), (387), (459), (495), (561), (585), (645), (765), (891), (975), (1105), (1161), (1485), (1683), (1755), (1935), (2145), (2193), (2295), (2805), (2925), (3315), (3483), (4257), (4455), (5049), (5525), (5805), (6435), (6579), (8385), (8415), (8775), (9945), (10725), (10965), (12771), (14561), (15147), (16575), (17415), (19305), (19737), (21285), (22687), (25155), (25245), (27431), (29835), (32175), (32895), (33911), (36465), (36959), (38313), (41023), (41925), (43039), (43683), (49725), (50407), (57915), (59211), (63855), (66041), (68061), (70487), (72369), (72805), (75465), (75735), (83399), (96525), (98685), (99701), (107063), (109395), (110877), (114939), (115181), (118955), (123069), (125207), (125775), (127033), (129117), (129899), (130681), (130727), (130957), (131003), (131049), (142545), (149175), (165491), (178699), (182325), (188437), (189109), (191565), (198123), (204183), (217107), (218323), (218415), (226395), (234329), (247537)])) w_1_0 w_12_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 1024 68 w_14_0 ∧
  MxxRuntime.centeredRebaseRuns w_14_0 w_15_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [18]) ((4/1)) (26) w_18_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (25), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_2_0) w_20_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (25), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_2_0) w_22_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 128 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (65), (85), (99), (129), (135), (153), (165), (195), (255), (297), (325), (387), (459), (495), (561), (585), (645), (765), (891), (975), (1105), (1161), (1485), (1683), (1755), (1935), (2145), (2193), (2295), (2805), (2925), (3315), (3483), (4257), (4455), (5049), (5525), (5805), (6435), (6579), (8385), (8415), (8775), (9945), (10725), (10965), (12771), (14561), (15147), (16575), (17415), (19305), (19737), (21285), (22687), (25155), (25245), (27431), (29835), (32175), (32895), (33911), (36465), (36959), (38313), (41023), (41925), (43039), (43683), (49725), (50407), (57915), (59211), (63855), (66041), (68061), (70487), (72369), (72805), (75465), (75735), (83399), (96525), (98685), (99701), (107063), (109395), (110877), (114939), (115181), (118955), (123069), (125207), (125775), (127033), (129117), (129899), (130681), (130727), (130957), (131003), (131049), (142545), (149175), (165491), (178699), (182325), (188437), (189109), (191565), (198123), (204183), (217107), (218323), (218415), (226395), (234329), (247537)])) w_7_0 w_25_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [29]) ((4/1)) (26) w_29_0 ∧
  outputs = (w_19_0, (w_30_0, ()))

def parallel_parallel_generatedRoot_7_76 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat1 × ByteArray × mxxMat1 × Unit) (outputs : mxxMat56 × mxxMat56 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_2_0 : ByteArray := inputs.2.1
  let w_6_0 : mxxMat1 := inputs.2.2.1
  ∃ (w_1_0 : mxxMat55),
    let w_3_0 : Int := (MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (65), (85), (99), (129), (135), (153), (165), (195), (255), (297), (325), (387), (459), (495), (561), (585), (645), (765), (891), (975), (1105), (1161), (1485), (1683), (1755), (1935), (2145), (2193), (2295), (2805), (2925), (3315), (3483), (4257), (4455), (5049), (5525), (5805), (6435), (6579), (8385), (8415), (8775), (9945), (10725), (10965), (12771), (14561), (15147), (16575), (17415), (19305), (19737), (21285), (22687), (25155), (25245), (27431), (29835), (32175), (32895), (33911), (36465), (36959), (38313), (41023), (41925), (43039), (43683), (49725), (50407), (57915), (59211), (63855), (66041), (68061), (70487), (72369), (72805), (75465), (75735), (83399), (96525), (98685), (99701), (107063), (109395), (110877), (114939), (115181), (118955), (123069), (125207), (125775), (127033), (129117), (129899), (130681), (130727), (130957), (131003), (131049), (142545), (149175), (165491), (178699), (182325), (188437), (189109), (191565), (198123), (204183), (217107), (218323), (218415), (226395), (234329), (247537)])
    ∃ (w_4_0 : mxxMat56),
      let w_5_0 : mxxMat56 := MxxRuntime.matrixMulScalarLeft w_1_0 w_4_0
      ∃ (w_7_0 : mxxMat55),
        ∃ (w_8_0 : mxxMat56),
          let w_9_0 : mxxMat56 := MxxRuntime.matrixMulScalarLeft w_7_0 w_8_0
          let w_10_0 : mxxMat56 := MxxRuntime.matrixAdd w_5_0 w_9_0
          let w_11_0 : mxxMat55 := (MxxRuntime.matrixPolynomial [(1099450810369)] : mxxMat55)
          ∃ (w_12_0 : mxxMat55),
            let w_13_0 : mxxMat55 := MxxRuntime.matrixMulScalarLeft w_11_0 w_12_0
            ∃ (w_14_0 : mxxMat57),
              ∃ (w_15_0 : mxxMat56),
                let w_16_0 : mxxMat56 := MxxRuntime.matrixMulScalarLeft w_13_0 w_15_0
                let w_17_0 : mxxMat56 := MxxRuntime.matrixSub w_10_0 w_16_0
                ∃ (w_18_0 : mxxMat56),
                  let w_19_0 : mxxMat56 := MxxRuntime.matrixAdd w_17_0 w_18_0
                  ∃ (w_20_0 : mxxMat56),
                    let w_21_0 : mxxMat56 := MxxRuntime.matrixMulScalarLeft w_1_0 w_20_0
                    ∃ (w_22_0 : mxxMat56),
                      let w_23_0 : mxxMat56 := MxxRuntime.matrixMulScalarLeft w_7_0 w_22_0
                      let w_24_0 : mxxMat56 := MxxRuntime.matrixAdd w_21_0 w_23_0
                      ∃ (w_25_0 : mxxMat55),
                        let w_26_0 : mxxMat55 := MxxRuntime.matrixMulScalarLeft w_11_0 w_25_0
                        let w_27_0 : mxxMat56 := MxxRuntime.matrixMulScalarLeft w_26_0 w_15_0
                        let w_28_0 : mxxMat56 := MxxRuntime.matrixSub w_24_0 w_27_0
                        ∃ (w_29_0 : mxxMat56),
                          let w_30_0 : mxxMat56 := MxxRuntime.matrixAdd w_28_0 w_29_0
                          parallel_parallel_generatedRoot_7_76.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs

abbrev parallel_parallel_generatedRoot_7_77.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_1 : Nat) (w_0_0 : mxxMat1) (w_2_0 : ByteArray) (w_6_0 : mxxMat1) (w_1_0 : mxxMat55) (w_3_0 : Int) (w_4_0 : mxxMat56) (w_7_0 : mxxMat55) (w_8_0 : mxxMat56) (w_12_0 : mxxMat55) (w_14_0 : mxxMat57) (w_15_0 : mxxMat56) (w_18_0 : mxxMat56) (w_19_0 : mxxMat56) (w_20_0 : mxxMat56) (w_22_0 : mxxMat56) (w_25_0 : mxxMat55) (w_29_0 : mxxMat56) (w_30_0 : mxxMat56) (outputs : mxxMat56 × mxxMat56 × Unit) : Prop :=
  MxxRuntime.modulusReduceRuns w_0_0 w_1_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 32 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (25), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_2_0) w_4_0 ∧
  MxxRuntime.modulusReduceRuns w_6_0 w_7_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (25), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_2_0) w_8_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 32 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (85), (99), (135), (153), (165), (255), (297), (459), (495), (561), (765), (891), (1485), (1683), (2295), (2805), (4455), (5049), (8415), (15147), (25245), (75735)])) w_1_0 w_12_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 1024 68 w_14_0 ∧
  MxxRuntime.centeredRebaseRuns w_14_0 w_15_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [18]) ((4/1)) (26) w_18_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (25), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_2_0) w_20_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (25), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_2_0) w_22_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 32 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (85), (99), (135), (153), (165), (255), (297), (459), (495), (561), (765), (891), (1485), (1683), (2295), (2805), (4455), (5049), (8415), (15147), (25245), (75735)])) w_7_0 w_25_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [29]) ((4/1)) (26) w_29_0 ∧
  outputs = (w_19_0, (w_30_0, ()))

def parallel_parallel_generatedRoot_7_77 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat1 × ByteArray × mxxMat1 × Unit) (outputs : mxxMat56 × mxxMat56 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_2_0 : ByteArray := inputs.2.1
  let w_6_0 : mxxMat1 := inputs.2.2.1
  ∃ (w_1_0 : mxxMat55),
    let w_3_0 : Int := (MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (85), (99), (135), (153), (165), (255), (297), (459), (495), (561), (765), (891), (1485), (1683), (2295), (2805), (4455), (5049), (8415), (15147), (25245), (75735)])
    ∃ (w_4_0 : mxxMat56),
      let w_5_0 : mxxMat56 := MxxRuntime.matrixMulScalarLeft w_1_0 w_4_0
      ∃ (w_7_0 : mxxMat55),
        ∃ (w_8_0 : mxxMat56),
          let w_9_0 : mxxMat56 := MxxRuntime.matrixMulScalarLeft w_7_0 w_8_0
          let w_10_0 : mxxMat56 := MxxRuntime.matrixAdd w_5_0 w_9_0
          let w_11_0 : mxxMat55 := (MxxRuntime.matrixPolynomial [(1099450810369)] : mxxMat55)
          ∃ (w_12_0 : mxxMat55),
            let w_13_0 : mxxMat55 := MxxRuntime.matrixMulScalarLeft w_11_0 w_12_0
            ∃ (w_14_0 : mxxMat57),
              ∃ (w_15_0 : mxxMat56),
                let w_16_0 : mxxMat56 := MxxRuntime.matrixMulScalarLeft w_13_0 w_15_0
                let w_17_0 : mxxMat56 := MxxRuntime.matrixSub w_10_0 w_16_0
                ∃ (w_18_0 : mxxMat56),
                  let w_19_0 : mxxMat56 := MxxRuntime.matrixAdd w_17_0 w_18_0
                  ∃ (w_20_0 : mxxMat56),
                    let w_21_0 : mxxMat56 := MxxRuntime.matrixMulScalarLeft w_1_0 w_20_0
                    ∃ (w_22_0 : mxxMat56),
                      let w_23_0 : mxxMat56 := MxxRuntime.matrixMulScalarLeft w_7_0 w_22_0
                      let w_24_0 : mxxMat56 := MxxRuntime.matrixAdd w_21_0 w_23_0
                      ∃ (w_25_0 : mxxMat55),
                        let w_26_0 : mxxMat55 := MxxRuntime.matrixMulScalarLeft w_11_0 w_25_0
                        let w_27_0 : mxxMat56 := MxxRuntime.matrixMulScalarLeft w_26_0 w_15_0
                        let w_28_0 : mxxMat56 := MxxRuntime.matrixSub w_24_0 w_27_0
                        ∃ (w_29_0 : mxxMat56),
                          let w_30_0 : mxxMat56 := MxxRuntime.matrixAdd w_28_0 w_29_0
                          parallel_parallel_generatedRoot_7_77.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs

abbrev parallel_parallel_generatedRoot_7_78.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_1 : Nat) (w_0_0 : mxxMat1) (w_2_0 : ByteArray) (w_6_0 : mxxMat1) (w_1_0 : mxxMat58) (w_3_0 : Int) (w_4_0 : mxxMat59) (w_7_0 : mxxMat58) (w_8_0 : mxxMat59) (w_12_0 : mxxMat58) (w_14_0 : mxxMat60) (w_15_0 : mxxMat59) (w_18_0 : mxxMat59) (w_19_0 : mxxMat59) (w_20_0 : mxxMat59) (w_22_0 : mxxMat59) (w_25_0 : mxxMat58) (w_29_0 : mxxMat59) (w_30_0 : mxxMat59) (outputs : mxxMat59 × mxxMat59 × Unit) : Prop :=
  MxxRuntime.modulusReduceRuns w_0_0 w_1_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 128 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (26), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_2_0) w_4_0 ∧
  MxxRuntime.modulusReduceRuns w_6_0 w_7_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (26), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_2_0) w_8_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 128 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (65), (85), (99), (129), (135), (153), (165), (195), (255), (297), (325), (387), (459), (495), (561), (585), (645), (765), (891), (975), (1105), (1161), (1485), (1683), (1755), (1935), (2145), (2193), (2295), (2805), (2925), (3315), (3483), (4257), (4455), (5049), (5525), (5805), (6435), (6579), (8385), (8415), (8775), (9945), (10725), (10965), (12771), (14561), (15147), (16575), (17415), (19305), (19737), (21285), (22687), (25155), (25245), (27431), (29835), (32175), (32895), (33911), (36465), (36959), (38313), (41023), (41925), (43039), (43683), (49725), (50407), (57915), (59211), (63855), (66041), (68061), (70487), (72369), (72805), (75465), (75735), (83399), (96525), (98685), (99701), (107063), (109395), (110877), (114939), (115181), (118955), (123069), (125207), (125775), (127033), (129117), (129899), (130681), (130727), (130957), (131003), (131049), (142545), (149175), (165491), (178699), (182325), (188437), (189109), (191565), (198123), (204183), (217107), (218323), (218415), (226395), (234329), (247537)])) w_1_0 w_12_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 1024 64 w_14_0 ∧
  MxxRuntime.centeredRebaseRuns w_14_0 w_15_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [18]) ((4/1)) (26) w_18_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (26), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_2_0) w_20_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (26), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_2_0) w_22_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 128 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (65), (85), (99), (129), (135), (153), (165), (195), (255), (297), (325), (387), (459), (495), (561), (585), (645), (765), (891), (975), (1105), (1161), (1485), (1683), (1755), (1935), (2145), (2193), (2295), (2805), (2925), (3315), (3483), (4257), (4455), (5049), (5525), (5805), (6435), (6579), (8385), (8415), (8775), (9945), (10725), (10965), (12771), (14561), (15147), (16575), (17415), (19305), (19737), (21285), (22687), (25155), (25245), (27431), (29835), (32175), (32895), (33911), (36465), (36959), (38313), (41023), (41925), (43039), (43683), (49725), (50407), (57915), (59211), (63855), (66041), (68061), (70487), (72369), (72805), (75465), (75735), (83399), (96525), (98685), (99701), (107063), (109395), (110877), (114939), (115181), (118955), (123069), (125207), (125775), (127033), (129117), (129899), (130681), (130727), (130957), (131003), (131049), (142545), (149175), (165491), (178699), (182325), (188437), (189109), (191565), (198123), (204183), (217107), (218323), (218415), (226395), (234329), (247537)])) w_7_0 w_25_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [29]) ((4/1)) (26) w_29_0 ∧
  outputs = (w_19_0, (w_30_0, ()))

def parallel_parallel_generatedRoot_7_78 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat1 × ByteArray × mxxMat1 × Unit) (outputs : mxxMat59 × mxxMat59 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_2_0 : ByteArray := inputs.2.1
  let w_6_0 : mxxMat1 := inputs.2.2.1
  ∃ (w_1_0 : mxxMat58),
    let w_3_0 : Int := (MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (65), (85), (99), (129), (135), (153), (165), (195), (255), (297), (325), (387), (459), (495), (561), (585), (645), (765), (891), (975), (1105), (1161), (1485), (1683), (1755), (1935), (2145), (2193), (2295), (2805), (2925), (3315), (3483), (4257), (4455), (5049), (5525), (5805), (6435), (6579), (8385), (8415), (8775), (9945), (10725), (10965), (12771), (14561), (15147), (16575), (17415), (19305), (19737), (21285), (22687), (25155), (25245), (27431), (29835), (32175), (32895), (33911), (36465), (36959), (38313), (41023), (41925), (43039), (43683), (49725), (50407), (57915), (59211), (63855), (66041), (68061), (70487), (72369), (72805), (75465), (75735), (83399), (96525), (98685), (99701), (107063), (109395), (110877), (114939), (115181), (118955), (123069), (125207), (125775), (127033), (129117), (129899), (130681), (130727), (130957), (131003), (131049), (142545), (149175), (165491), (178699), (182325), (188437), (189109), (191565), (198123), (204183), (217107), (218323), (218415), (226395), (234329), (247537)])
    ∃ (w_4_0 : mxxMat59),
      let w_5_0 : mxxMat59 := MxxRuntime.matrixMulScalarLeft w_1_0 w_4_0
      ∃ (w_7_0 : mxxMat58),
        ∃ (w_8_0 : mxxMat59),
          let w_9_0 : mxxMat59 := MxxRuntime.matrixMulScalarLeft w_7_0 w_8_0
          let w_10_0 : mxxMat59 := MxxRuntime.matrixAdd w_5_0 w_9_0
          let w_11_0 : mxxMat58 := (MxxRuntime.matrixPolynomial [(1099451072513)] : mxxMat58)
          ∃ (w_12_0 : mxxMat58),
            let w_13_0 : mxxMat58 := MxxRuntime.matrixMulScalarLeft w_11_0 w_12_0
            ∃ (w_14_0 : mxxMat60),
              ∃ (w_15_0 : mxxMat59),
                let w_16_0 : mxxMat59 := MxxRuntime.matrixMulScalarLeft w_13_0 w_15_0
                let w_17_0 : mxxMat59 := MxxRuntime.matrixSub w_10_0 w_16_0
                ∃ (w_18_0 : mxxMat59),
                  let w_19_0 : mxxMat59 := MxxRuntime.matrixAdd w_17_0 w_18_0
                  ∃ (w_20_0 : mxxMat59),
                    let w_21_0 : mxxMat59 := MxxRuntime.matrixMulScalarLeft w_1_0 w_20_0
                    ∃ (w_22_0 : mxxMat59),
                      let w_23_0 : mxxMat59 := MxxRuntime.matrixMulScalarLeft w_7_0 w_22_0
                      let w_24_0 : mxxMat59 := MxxRuntime.matrixAdd w_21_0 w_23_0
                      ∃ (w_25_0 : mxxMat58),
                        let w_26_0 : mxxMat58 := MxxRuntime.matrixMulScalarLeft w_11_0 w_25_0
                        let w_27_0 : mxxMat59 := MxxRuntime.matrixMulScalarLeft w_26_0 w_15_0
                        let w_28_0 : mxxMat59 := MxxRuntime.matrixSub w_24_0 w_27_0
                        ∃ (w_29_0 : mxxMat59),
                          let w_30_0 : mxxMat59 := MxxRuntime.matrixAdd w_28_0 w_29_0
                          parallel_parallel_generatedRoot_7_78.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs

abbrev parallel_parallel_generatedRoot_7_79.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_1 : Nat) (w_0_0 : mxxMat1) (w_2_0 : ByteArray) (w_6_0 : mxxMat1) (w_1_0 : mxxMat58) (w_3_0 : Int) (w_4_0 : mxxMat59) (w_7_0 : mxxMat58) (w_8_0 : mxxMat59) (w_12_0 : mxxMat58) (w_14_0 : mxxMat60) (w_15_0 : mxxMat59) (w_18_0 : mxxMat59) (w_19_0 : mxxMat59) (w_20_0 : mxxMat59) (w_22_0 : mxxMat59) (w_25_0 : mxxMat58) (w_29_0 : mxxMat59) (w_30_0 : mxxMat59) (outputs : mxxMat59 × mxxMat59 × Unit) : Prop :=
  MxxRuntime.modulusReduceRuns w_0_0 w_1_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 32 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (26), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_2_0) w_4_0 ∧
  MxxRuntime.modulusReduceRuns w_6_0 w_7_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (26), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_2_0) w_8_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 32 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (85), (99), (135), (153), (165), (255), (297), (459), (495), (561), (765), (891), (1485), (1683), (2295), (2805), (4455), (5049), (8415), (15147), (25245), (75735)])) w_1_0 w_12_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 1024 64 w_14_0 ∧
  MxxRuntime.centeredRebaseRuns w_14_0 w_15_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [18]) ((4/1)) (26) w_18_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (26), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_2_0) w_20_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (26), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_2_0) w_22_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 32 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (85), (99), (135), (153), (165), (255), (297), (459), (495), (561), (765), (891), (1485), (1683), (2295), (2805), (4455), (5049), (8415), (15147), (25245), (75735)])) w_7_0 w_25_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [29]) ((4/1)) (26) w_29_0 ∧
  outputs = (w_19_0, (w_30_0, ()))

def parallel_parallel_generatedRoot_7_79 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat1 × ByteArray × mxxMat1 × Unit) (outputs : mxxMat59 × mxxMat59 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_2_0 : ByteArray := inputs.2.1
  let w_6_0 : mxxMat1 := inputs.2.2.1
  ∃ (w_1_0 : mxxMat58),
    let w_3_0 : Int := (MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (85), (99), (135), (153), (165), (255), (297), (459), (495), (561), (765), (891), (1485), (1683), (2295), (2805), (4455), (5049), (8415), (15147), (25245), (75735)])
    ∃ (w_4_0 : mxxMat59),
      let w_5_0 : mxxMat59 := MxxRuntime.matrixMulScalarLeft w_1_0 w_4_0
      ∃ (w_7_0 : mxxMat58),
        ∃ (w_8_0 : mxxMat59),
          let w_9_0 : mxxMat59 := MxxRuntime.matrixMulScalarLeft w_7_0 w_8_0
          let w_10_0 : mxxMat59 := MxxRuntime.matrixAdd w_5_0 w_9_0
          let w_11_0 : mxxMat58 := (MxxRuntime.matrixPolynomial [(1099451072513)] : mxxMat58)
          ∃ (w_12_0 : mxxMat58),
            let w_13_0 : mxxMat58 := MxxRuntime.matrixMulScalarLeft w_11_0 w_12_0
            ∃ (w_14_0 : mxxMat60),
              ∃ (w_15_0 : mxxMat59),
                let w_16_0 : mxxMat59 := MxxRuntime.matrixMulScalarLeft w_13_0 w_15_0
                let w_17_0 : mxxMat59 := MxxRuntime.matrixSub w_10_0 w_16_0
                ∃ (w_18_0 : mxxMat59),
                  let w_19_0 : mxxMat59 := MxxRuntime.matrixAdd w_17_0 w_18_0
                  ∃ (w_20_0 : mxxMat59),
                    let w_21_0 : mxxMat59 := MxxRuntime.matrixMulScalarLeft w_1_0 w_20_0
                    ∃ (w_22_0 : mxxMat59),
                      let w_23_0 : mxxMat59 := MxxRuntime.matrixMulScalarLeft w_7_0 w_22_0
                      let w_24_0 : mxxMat59 := MxxRuntime.matrixAdd w_21_0 w_23_0
                      ∃ (w_25_0 : mxxMat58),
                        let w_26_0 : mxxMat58 := MxxRuntime.matrixMulScalarLeft w_11_0 w_25_0
                        let w_27_0 : mxxMat59 := MxxRuntime.matrixMulScalarLeft w_26_0 w_15_0
                        let w_28_0 : mxxMat59 := MxxRuntime.matrixSub w_24_0 w_27_0
                        ∃ (w_29_0 : mxxMat59),
                          let w_30_0 : mxxMat59 := MxxRuntime.matrixAdd w_28_0 w_29_0
                          parallel_parallel_generatedRoot_7_79.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs

abbrev parallel_parallel_generatedRoot_7_80.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_1 : Nat) (w_0_0 : mxxMat1) (w_2_0 : ByteArray) (w_6_0 : mxxMat1) (w_1_0 : mxxMat61) (w_3_0 : Int) (w_4_0 : mxxMat62) (w_7_0 : mxxMat61) (w_8_0 : mxxMat62) (w_12_0 : mxxMat61) (w_14_0 : mxxMat63) (w_15_0 : mxxMat62) (w_18_0 : mxxMat62) (w_19_0 : mxxMat62) (w_20_0 : mxxMat62) (w_22_0 : mxxMat62) (w_25_0 : mxxMat61) (w_29_0 : mxxMat62) (w_30_0 : mxxMat62) (outputs : mxxMat62 × mxxMat62 × Unit) : Prop :=
  MxxRuntime.modulusReduceRuns w_0_0 w_1_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 128 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (27), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_2_0) w_4_0 ∧
  MxxRuntime.modulusReduceRuns w_6_0 w_7_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (27), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_2_0) w_8_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 128 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (65), (85), (99), (129), (135), (153), (165), (195), (255), (297), (325), (387), (459), (495), (561), (585), (645), (765), (891), (975), (1105), (1161), (1485), (1683), (1755), (1935), (2145), (2193), (2295), (2805), (2925), (3315), (3483), (4257), (4455), (5049), (5525), (5805), (6435), (6579), (8385), (8415), (8775), (9945), (10725), (10965), (12771), (14561), (15147), (16575), (17415), (19305), (19737), (21285), (22687), (25155), (25245), (27431), (29835), (32175), (32895), (33911), (36465), (36959), (38313), (41023), (41925), (43039), (43683), (49725), (50407), (57915), (59211), (63855), (66041), (68061), (70487), (72369), (72805), (75465), (75735), (83399), (96525), (98685), (99701), (107063), (109395), (110877), (114939), (115181), (118955), (123069), (125207), (125775), (127033), (129117), (129899), (130681), (130727), (130957), (131003), (131049), (142545), (149175), (165491), (178699), (182325), (188437), (189109), (191565), (198123), (204183), (217107), (218323), (218415), (226395), (234329), (247537)])) w_1_0 w_12_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 1024 60 w_14_0 ∧
  MxxRuntime.centeredRebaseRuns w_14_0 w_15_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [18]) ((4/1)) (26) w_18_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (27), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_2_0) w_20_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (27), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_2_0) w_22_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 128 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (65), (85), (99), (129), (135), (153), (165), (195), (255), (297), (325), (387), (459), (495), (561), (585), (645), (765), (891), (975), (1105), (1161), (1485), (1683), (1755), (1935), (2145), (2193), (2295), (2805), (2925), (3315), (3483), (4257), (4455), (5049), (5525), (5805), (6435), (6579), (8385), (8415), (8775), (9945), (10725), (10965), (12771), (14561), (15147), (16575), (17415), (19305), (19737), (21285), (22687), (25155), (25245), (27431), (29835), (32175), (32895), (33911), (36465), (36959), (38313), (41023), (41925), (43039), (43683), (49725), (50407), (57915), (59211), (63855), (66041), (68061), (70487), (72369), (72805), (75465), (75735), (83399), (96525), (98685), (99701), (107063), (109395), (110877), (114939), (115181), (118955), (123069), (125207), (125775), (127033), (129117), (129899), (130681), (130727), (130957), (131003), (131049), (142545), (149175), (165491), (178699), (182325), (188437), (189109), (191565), (198123), (204183), (217107), (218323), (218415), (226395), (234329), (247537)])) w_7_0 w_25_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [29]) ((4/1)) (26) w_29_0 ∧
  outputs = (w_19_0, (w_30_0, ()))

def parallel_parallel_generatedRoot_7_80 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat1 × ByteArray × mxxMat1 × Unit) (outputs : mxxMat62 × mxxMat62 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_2_0 : ByteArray := inputs.2.1
  let w_6_0 : mxxMat1 := inputs.2.2.1
  ∃ (w_1_0 : mxxMat61),
    let w_3_0 : Int := (MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (65), (85), (99), (129), (135), (153), (165), (195), (255), (297), (325), (387), (459), (495), (561), (585), (645), (765), (891), (975), (1105), (1161), (1485), (1683), (1755), (1935), (2145), (2193), (2295), (2805), (2925), (3315), (3483), (4257), (4455), (5049), (5525), (5805), (6435), (6579), (8385), (8415), (8775), (9945), (10725), (10965), (12771), (14561), (15147), (16575), (17415), (19305), (19737), (21285), (22687), (25155), (25245), (27431), (29835), (32175), (32895), (33911), (36465), (36959), (38313), (41023), (41925), (43039), (43683), (49725), (50407), (57915), (59211), (63855), (66041), (68061), (70487), (72369), (72805), (75465), (75735), (83399), (96525), (98685), (99701), (107063), (109395), (110877), (114939), (115181), (118955), (123069), (125207), (125775), (127033), (129117), (129899), (130681), (130727), (130957), (131003), (131049), (142545), (149175), (165491), (178699), (182325), (188437), (189109), (191565), (198123), (204183), (217107), (218323), (218415), (226395), (234329), (247537)])
    ∃ (w_4_0 : mxxMat62),
      let w_5_0 : mxxMat62 := MxxRuntime.matrixMulScalarLeft w_1_0 w_4_0
      ∃ (w_7_0 : mxxMat61),
        ∃ (w_8_0 : mxxMat62),
          let w_9_0 : mxxMat62 := MxxRuntime.matrixMulScalarLeft w_7_0 w_8_0
          let w_10_0 : mxxMat62 := MxxRuntime.matrixAdd w_5_0 w_9_0
          let w_11_0 : mxxMat61 := (MxxRuntime.matrixPolynomial [(1099453431809)] : mxxMat61)
          ∃ (w_12_0 : mxxMat61),
            let w_13_0 : mxxMat61 := MxxRuntime.matrixMulScalarLeft w_11_0 w_12_0
            ∃ (w_14_0 : mxxMat63),
              ∃ (w_15_0 : mxxMat62),
                let w_16_0 : mxxMat62 := MxxRuntime.matrixMulScalarLeft w_13_0 w_15_0
                let w_17_0 : mxxMat62 := MxxRuntime.matrixSub w_10_0 w_16_0
                ∃ (w_18_0 : mxxMat62),
                  let w_19_0 : mxxMat62 := MxxRuntime.matrixAdd w_17_0 w_18_0
                  ∃ (w_20_0 : mxxMat62),
                    let w_21_0 : mxxMat62 := MxxRuntime.matrixMulScalarLeft w_1_0 w_20_0
                    ∃ (w_22_0 : mxxMat62),
                      let w_23_0 : mxxMat62 := MxxRuntime.matrixMulScalarLeft w_7_0 w_22_0
                      let w_24_0 : mxxMat62 := MxxRuntime.matrixAdd w_21_0 w_23_0
                      ∃ (w_25_0 : mxxMat61),
                        let w_26_0 : mxxMat61 := MxxRuntime.matrixMulScalarLeft w_11_0 w_25_0
                        let w_27_0 : mxxMat62 := MxxRuntime.matrixMulScalarLeft w_26_0 w_15_0
                        let w_28_0 : mxxMat62 := MxxRuntime.matrixSub w_24_0 w_27_0
                        ∃ (w_29_0 : mxxMat62),
                          let w_30_0 : mxxMat62 := MxxRuntime.matrixAdd w_28_0 w_29_0
                          parallel_parallel_generatedRoot_7_80.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs

abbrev parallel_parallel_generatedRoot_7_81.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_1 : Nat) (w_0_0 : mxxMat1) (w_2_0 : ByteArray) (w_6_0 : mxxMat1) (w_1_0 : mxxMat61) (w_3_0 : Int) (w_4_0 : mxxMat62) (w_7_0 : mxxMat61) (w_8_0 : mxxMat62) (w_12_0 : mxxMat61) (w_14_0 : mxxMat63) (w_15_0 : mxxMat62) (w_18_0 : mxxMat62) (w_19_0 : mxxMat62) (w_20_0 : mxxMat62) (w_22_0 : mxxMat62) (w_25_0 : mxxMat61) (w_29_0 : mxxMat62) (w_30_0 : mxxMat62) (outputs : mxxMat62 × mxxMat62 × Unit) : Prop :=
  MxxRuntime.modulusReduceRuns w_0_0 w_1_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 32 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (27), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_2_0) w_4_0 ∧
  MxxRuntime.modulusReduceRuns w_6_0 w_7_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (27), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_2_0) w_8_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 32 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (85), (99), (135), (153), (165), (255), (297), (459), (495), (561), (765), (891), (1485), (1683), (2295), (2805), (4455), (5049), (8415), (15147), (25245), (75735)])) w_1_0 w_12_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 1024 60 w_14_0 ∧
  MxxRuntime.centeredRebaseRuns w_14_0 w_15_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [18]) ((4/1)) (26) w_18_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (27), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_2_0) w_20_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (27), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_2_0) w_22_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 32 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (85), (99), (135), (153), (165), (255), (297), (459), (495), (561), (765), (891), (1485), (1683), (2295), (2805), (4455), (5049), (8415), (15147), (25245), (75735)])) w_7_0 w_25_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [29]) ((4/1)) (26) w_29_0 ∧
  outputs = (w_19_0, (w_30_0, ()))

def parallel_parallel_generatedRoot_7_81 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat1 × ByteArray × mxxMat1 × Unit) (outputs : mxxMat62 × mxxMat62 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_2_0 : ByteArray := inputs.2.1
  let w_6_0 : mxxMat1 := inputs.2.2.1
  ∃ (w_1_0 : mxxMat61),
    let w_3_0 : Int := (MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (85), (99), (135), (153), (165), (255), (297), (459), (495), (561), (765), (891), (1485), (1683), (2295), (2805), (4455), (5049), (8415), (15147), (25245), (75735)])
    ∃ (w_4_0 : mxxMat62),
      let w_5_0 : mxxMat62 := MxxRuntime.matrixMulScalarLeft w_1_0 w_4_0
      ∃ (w_7_0 : mxxMat61),
        ∃ (w_8_0 : mxxMat62),
          let w_9_0 : mxxMat62 := MxxRuntime.matrixMulScalarLeft w_7_0 w_8_0
          let w_10_0 : mxxMat62 := MxxRuntime.matrixAdd w_5_0 w_9_0
          let w_11_0 : mxxMat61 := (MxxRuntime.matrixPolynomial [(1099453431809)] : mxxMat61)
          ∃ (w_12_0 : mxxMat61),
            let w_13_0 : mxxMat61 := MxxRuntime.matrixMulScalarLeft w_11_0 w_12_0
            ∃ (w_14_0 : mxxMat63),
              ∃ (w_15_0 : mxxMat62),
                let w_16_0 : mxxMat62 := MxxRuntime.matrixMulScalarLeft w_13_0 w_15_0
                let w_17_0 : mxxMat62 := MxxRuntime.matrixSub w_10_0 w_16_0
                ∃ (w_18_0 : mxxMat62),
                  let w_19_0 : mxxMat62 := MxxRuntime.matrixAdd w_17_0 w_18_0
                  ∃ (w_20_0 : mxxMat62),
                    let w_21_0 : mxxMat62 := MxxRuntime.matrixMulScalarLeft w_1_0 w_20_0
                    ∃ (w_22_0 : mxxMat62),
                      let w_23_0 : mxxMat62 := MxxRuntime.matrixMulScalarLeft w_7_0 w_22_0
                      let w_24_0 : mxxMat62 := MxxRuntime.matrixAdd w_21_0 w_23_0
                      ∃ (w_25_0 : mxxMat61),
                        let w_26_0 : mxxMat61 := MxxRuntime.matrixMulScalarLeft w_11_0 w_25_0
                        let w_27_0 : mxxMat62 := MxxRuntime.matrixMulScalarLeft w_26_0 w_15_0
                        let w_28_0 : mxxMat62 := MxxRuntime.matrixSub w_24_0 w_27_0
                        ∃ (w_29_0 : mxxMat62),
                          let w_30_0 : mxxMat62 := MxxRuntime.matrixAdd w_28_0 w_29_0
                          parallel_parallel_generatedRoot_7_81.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs

abbrev parallel_parallel_generatedRoot_7_82.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_1 : Nat) (w_0_0 : mxxMat1) (w_2_0 : ByteArray) (w_6_0 : mxxMat1) (w_1_0 : mxxMat64) (w_3_0 : Int) (w_4_0 : mxxMat65) (w_7_0 : mxxMat64) (w_8_0 : mxxMat65) (w_12_0 : mxxMat64) (w_14_0 : mxxMat66) (w_15_0 : mxxMat65) (w_18_0 : mxxMat65) (w_19_0 : mxxMat65) (w_20_0 : mxxMat65) (w_22_0 : mxxMat65) (w_25_0 : mxxMat64) (w_29_0 : mxxMat65) (w_30_0 : mxxMat65) (outputs : mxxMat65 × mxxMat65 × Unit) : Prop :=
  MxxRuntime.modulusReduceRuns w_0_0 w_1_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 128 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (28), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_2_0) w_4_0 ∧
  MxxRuntime.modulusReduceRuns w_6_0 w_7_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (28), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_2_0) w_8_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 128 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (65), (85), (99), (129), (135), (153), (165), (195), (255), (297), (325), (387), (459), (495), (561), (585), (645), (765), (891), (975), (1105), (1161), (1485), (1683), (1755), (1935), (2145), (2193), (2295), (2805), (2925), (3315), (3483), (4257), (4455), (5049), (5525), (5805), (6435), (6579), (8385), (8415), (8775), (9945), (10725), (10965), (12771), (14561), (15147), (16575), (17415), (19305), (19737), (21285), (22687), (25155), (25245), (27431), (29835), (32175), (32895), (33911), (36465), (36959), (38313), (41023), (41925), (43039), (43683), (49725), (50407), (57915), (59211), (63855), (66041), (68061), (70487), (72369), (72805), (75465), (75735), (83399), (96525), (98685), (99701), (107063), (109395), (110877), (114939), (115181), (118955), (123069), (125207), (125775), (127033), (129117), (129899), (130681), (130727), (130957), (131003), (131049), (142545), (149175), (165491), (178699), (182325), (188437), (189109), (191565), (198123), (204183), (217107), (218323), (218415), (226395), (234329), (247537)])) w_1_0 w_12_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 1024 56 w_14_0 ∧
  MxxRuntime.centeredRebaseRuns w_14_0 w_15_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [18]) ((4/1)) (26) w_18_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (28), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_2_0) w_20_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (28), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_2_0) w_22_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 128 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (65), (85), (99), (129), (135), (153), (165), (195), (255), (297), (325), (387), (459), (495), (561), (585), (645), (765), (891), (975), (1105), (1161), (1485), (1683), (1755), (1935), (2145), (2193), (2295), (2805), (2925), (3315), (3483), (4257), (4455), (5049), (5525), (5805), (6435), (6579), (8385), (8415), (8775), (9945), (10725), (10965), (12771), (14561), (15147), (16575), (17415), (19305), (19737), (21285), (22687), (25155), (25245), (27431), (29835), (32175), (32895), (33911), (36465), (36959), (38313), (41023), (41925), (43039), (43683), (49725), (50407), (57915), (59211), (63855), (66041), (68061), (70487), (72369), (72805), (75465), (75735), (83399), (96525), (98685), (99701), (107063), (109395), (110877), (114939), (115181), (118955), (123069), (125207), (125775), (127033), (129117), (129899), (130681), (130727), (130957), (131003), (131049), (142545), (149175), (165491), (178699), (182325), (188437), (189109), (191565), (198123), (204183), (217107), (218323), (218415), (226395), (234329), (247537)])) w_7_0 w_25_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [29]) ((4/1)) (26) w_29_0 ∧
  outputs = (w_19_0, (w_30_0, ()))

def parallel_parallel_generatedRoot_7_82 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat1 × ByteArray × mxxMat1 × Unit) (outputs : mxxMat65 × mxxMat65 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_2_0 : ByteArray := inputs.2.1
  let w_6_0 : mxxMat1 := inputs.2.2.1
  ∃ (w_1_0 : mxxMat64),
    let w_3_0 : Int := (MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (65), (85), (99), (129), (135), (153), (165), (195), (255), (297), (325), (387), (459), (495), (561), (585), (645), (765), (891), (975), (1105), (1161), (1485), (1683), (1755), (1935), (2145), (2193), (2295), (2805), (2925), (3315), (3483), (4257), (4455), (5049), (5525), (5805), (6435), (6579), (8385), (8415), (8775), (9945), (10725), (10965), (12771), (14561), (15147), (16575), (17415), (19305), (19737), (21285), (22687), (25155), (25245), (27431), (29835), (32175), (32895), (33911), (36465), (36959), (38313), (41023), (41925), (43039), (43683), (49725), (50407), (57915), (59211), (63855), (66041), (68061), (70487), (72369), (72805), (75465), (75735), (83399), (96525), (98685), (99701), (107063), (109395), (110877), (114939), (115181), (118955), (123069), (125207), (125775), (127033), (129117), (129899), (130681), (130727), (130957), (131003), (131049), (142545), (149175), (165491), (178699), (182325), (188437), (189109), (191565), (198123), (204183), (217107), (218323), (218415), (226395), (234329), (247537)])
    ∃ (w_4_0 : mxxMat65),
      let w_5_0 : mxxMat65 := MxxRuntime.matrixMulScalarLeft w_1_0 w_4_0
      ∃ (w_7_0 : mxxMat64),
        ∃ (w_8_0 : mxxMat65),
          let w_9_0 : mxxMat65 := MxxRuntime.matrixMulScalarLeft w_7_0 w_8_0
          let w_10_0 : mxxMat65 := MxxRuntime.matrixAdd w_5_0 w_9_0
          let w_11_0 : mxxMat64 := (MxxRuntime.matrixPolynomial [(1099454218241)] : mxxMat64)
          ∃ (w_12_0 : mxxMat64),
            let w_13_0 : mxxMat64 := MxxRuntime.matrixMulScalarLeft w_11_0 w_12_0
            ∃ (w_14_0 : mxxMat66),
              ∃ (w_15_0 : mxxMat65),
                let w_16_0 : mxxMat65 := MxxRuntime.matrixMulScalarLeft w_13_0 w_15_0
                let w_17_0 : mxxMat65 := MxxRuntime.matrixSub w_10_0 w_16_0
                ∃ (w_18_0 : mxxMat65),
                  let w_19_0 : mxxMat65 := MxxRuntime.matrixAdd w_17_0 w_18_0
                  ∃ (w_20_0 : mxxMat65),
                    let w_21_0 : mxxMat65 := MxxRuntime.matrixMulScalarLeft w_1_0 w_20_0
                    ∃ (w_22_0 : mxxMat65),
                      let w_23_0 : mxxMat65 := MxxRuntime.matrixMulScalarLeft w_7_0 w_22_0
                      let w_24_0 : mxxMat65 := MxxRuntime.matrixAdd w_21_0 w_23_0
                      ∃ (w_25_0 : mxxMat64),
                        let w_26_0 : mxxMat64 := MxxRuntime.matrixMulScalarLeft w_11_0 w_25_0
                        let w_27_0 : mxxMat65 := MxxRuntime.matrixMulScalarLeft w_26_0 w_15_0
                        let w_28_0 : mxxMat65 := MxxRuntime.matrixSub w_24_0 w_27_0
                        ∃ (w_29_0 : mxxMat65),
                          let w_30_0 : mxxMat65 := MxxRuntime.matrixAdd w_28_0 w_29_0
                          parallel_parallel_generatedRoot_7_82.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs

abbrev parallel_parallel_generatedRoot_7_83.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_1 : Nat) (w_0_0 : mxxMat1) (w_2_0 : ByteArray) (w_6_0 : mxxMat1) (w_1_0 : mxxMat64) (w_3_0 : Int) (w_4_0 : mxxMat65) (w_7_0 : mxxMat64) (w_8_0 : mxxMat65) (w_12_0 : mxxMat64) (w_14_0 : mxxMat66) (w_15_0 : mxxMat65) (w_18_0 : mxxMat65) (w_19_0 : mxxMat65) (w_20_0 : mxxMat65) (w_22_0 : mxxMat65) (w_25_0 : mxxMat64) (w_29_0 : mxxMat65) (w_30_0 : mxxMat65) (outputs : mxxMat65 × mxxMat65 × Unit) : Prop :=
  MxxRuntime.modulusReduceRuns w_0_0 w_1_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 32 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (28), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_2_0) w_4_0 ∧
  MxxRuntime.modulusReduceRuns w_6_0 w_7_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (28), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_2_0) w_8_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 32 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (85), (99), (135), (153), (165), (255), (297), (459), (495), (561), (765), (891), (1485), (1683), (2295), (2805), (4455), (5049), (8415), (15147), (25245), (75735)])) w_1_0 w_12_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 1024 56 w_14_0 ∧
  MxxRuntime.centeredRebaseRuns w_14_0 w_15_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [18]) ((4/1)) (26) w_18_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (28), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_2_0) w_20_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (28), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_2_0) w_22_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 32 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (85), (99), (135), (153), (165), (255), (297), (459), (495), (561), (765), (891), (1485), (1683), (2295), (2805), (4455), (5049), (8415), (15147), (25245), (75735)])) w_7_0 w_25_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [29]) ((4/1)) (26) w_29_0 ∧
  outputs = (w_19_0, (w_30_0, ()))

def parallel_parallel_generatedRoot_7_83 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat1 × ByteArray × mxxMat1 × Unit) (outputs : mxxMat65 × mxxMat65 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_2_0 : ByteArray := inputs.2.1
  let w_6_0 : mxxMat1 := inputs.2.2.1
  ∃ (w_1_0 : mxxMat64),
    let w_3_0 : Int := (MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (85), (99), (135), (153), (165), (255), (297), (459), (495), (561), (765), (891), (1485), (1683), (2295), (2805), (4455), (5049), (8415), (15147), (25245), (75735)])
    ∃ (w_4_0 : mxxMat65),
      let w_5_0 : mxxMat65 := MxxRuntime.matrixMulScalarLeft w_1_0 w_4_0
      ∃ (w_7_0 : mxxMat64),
        ∃ (w_8_0 : mxxMat65),
          let w_9_0 : mxxMat65 := MxxRuntime.matrixMulScalarLeft w_7_0 w_8_0
          let w_10_0 : mxxMat65 := MxxRuntime.matrixAdd w_5_0 w_9_0
          let w_11_0 : mxxMat64 := (MxxRuntime.matrixPolynomial [(1099454218241)] : mxxMat64)
          ∃ (w_12_0 : mxxMat64),
            let w_13_0 : mxxMat64 := MxxRuntime.matrixMulScalarLeft w_11_0 w_12_0
            ∃ (w_14_0 : mxxMat66),
              ∃ (w_15_0 : mxxMat65),
                let w_16_0 : mxxMat65 := MxxRuntime.matrixMulScalarLeft w_13_0 w_15_0
                let w_17_0 : mxxMat65 := MxxRuntime.matrixSub w_10_0 w_16_0
                ∃ (w_18_0 : mxxMat65),
                  let w_19_0 : mxxMat65 := MxxRuntime.matrixAdd w_17_0 w_18_0
                  ∃ (w_20_0 : mxxMat65),
                    let w_21_0 : mxxMat65 := MxxRuntime.matrixMulScalarLeft w_1_0 w_20_0
                    ∃ (w_22_0 : mxxMat65),
                      let w_23_0 : mxxMat65 := MxxRuntime.matrixMulScalarLeft w_7_0 w_22_0
                      let w_24_0 : mxxMat65 := MxxRuntime.matrixAdd w_21_0 w_23_0
                      ∃ (w_25_0 : mxxMat64),
                        let w_26_0 : mxxMat64 := MxxRuntime.matrixMulScalarLeft w_11_0 w_25_0
                        let w_27_0 : mxxMat65 := MxxRuntime.matrixMulScalarLeft w_26_0 w_15_0
                        let w_28_0 : mxxMat65 := MxxRuntime.matrixSub w_24_0 w_27_0
                        ∃ (w_29_0 : mxxMat65),
                          let w_30_0 : mxxMat65 := MxxRuntime.matrixAdd w_28_0 w_29_0
                          parallel_parallel_generatedRoot_7_83.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs

abbrev parallel_parallel_generatedRoot_7_84.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_1 : Nat) (w_0_0 : mxxMat1) (w_2_0 : ByteArray) (w_6_0 : mxxMat1) (w_1_0 : mxxMat67) (w_3_0 : Int) (w_4_0 : mxxMat68) (w_7_0 : mxxMat67) (w_8_0 : mxxMat68) (w_12_0 : mxxMat67) (w_14_0 : mxxMat69) (w_15_0 : mxxMat68) (w_18_0 : mxxMat68) (w_19_0 : mxxMat68) (w_20_0 : mxxMat68) (w_22_0 : mxxMat68) (w_25_0 : mxxMat67) (w_29_0 : mxxMat68) (w_30_0 : mxxMat68) (outputs : mxxMat68 × mxxMat68 × Unit) : Prop :=
  MxxRuntime.modulusReduceRuns w_0_0 w_1_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 128 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (29), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_2_0) w_4_0 ∧
  MxxRuntime.modulusReduceRuns w_6_0 w_7_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (29), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_2_0) w_8_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 128 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (65), (85), (99), (129), (135), (153), (165), (195), (255), (297), (325), (387), (459), (495), (561), (585), (645), (765), (891), (975), (1105), (1161), (1485), (1683), (1755), (1935), (2145), (2193), (2295), (2805), (2925), (3315), (3483), (4257), (4455), (5049), (5525), (5805), (6435), (6579), (8385), (8415), (8775), (9945), (10725), (10965), (12771), (14561), (15147), (16575), (17415), (19305), (19737), (21285), (22687), (25155), (25245), (27431), (29835), (32175), (32895), (33911), (36465), (36959), (38313), (41023), (41925), (43039), (43683), (49725), (50407), (57915), (59211), (63855), (66041), (68061), (70487), (72369), (72805), (75465), (75735), (83399), (96525), (98685), (99701), (107063), (109395), (110877), (114939), (115181), (118955), (123069), (125207), (125775), (127033), (129117), (129899), (130681), (130727), (130957), (131003), (131049), (142545), (149175), (165491), (178699), (182325), (188437), (189109), (191565), (198123), (204183), (217107), (218323), (218415), (226395), (234329), (247537)])) w_1_0 w_12_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 1024 52 w_14_0 ∧
  MxxRuntime.centeredRebaseRuns w_14_0 w_15_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [18]) ((4/1)) (26) w_18_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (29), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_2_0) w_20_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (29), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_2_0) w_22_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 128 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (65), (85), (99), (129), (135), (153), (165), (195), (255), (297), (325), (387), (459), (495), (561), (585), (645), (765), (891), (975), (1105), (1161), (1485), (1683), (1755), (1935), (2145), (2193), (2295), (2805), (2925), (3315), (3483), (4257), (4455), (5049), (5525), (5805), (6435), (6579), (8385), (8415), (8775), (9945), (10725), (10965), (12771), (14561), (15147), (16575), (17415), (19305), (19737), (21285), (22687), (25155), (25245), (27431), (29835), (32175), (32895), (33911), (36465), (36959), (38313), (41023), (41925), (43039), (43683), (49725), (50407), (57915), (59211), (63855), (66041), (68061), (70487), (72369), (72805), (75465), (75735), (83399), (96525), (98685), (99701), (107063), (109395), (110877), (114939), (115181), (118955), (123069), (125207), (125775), (127033), (129117), (129899), (130681), (130727), (130957), (131003), (131049), (142545), (149175), (165491), (178699), (182325), (188437), (189109), (191565), (198123), (204183), (217107), (218323), (218415), (226395), (234329), (247537)])) w_7_0 w_25_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [29]) ((4/1)) (26) w_29_0 ∧
  outputs = (w_19_0, (w_30_0, ()))

def parallel_parallel_generatedRoot_7_84 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat1 × ByteArray × mxxMat1 × Unit) (outputs : mxxMat68 × mxxMat68 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_2_0 : ByteArray := inputs.2.1
  let w_6_0 : mxxMat1 := inputs.2.2.1
  ∃ (w_1_0 : mxxMat67),
    let w_3_0 : Int := (MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (65), (85), (99), (129), (135), (153), (165), (195), (255), (297), (325), (387), (459), (495), (561), (585), (645), (765), (891), (975), (1105), (1161), (1485), (1683), (1755), (1935), (2145), (2193), (2295), (2805), (2925), (3315), (3483), (4257), (4455), (5049), (5525), (5805), (6435), (6579), (8385), (8415), (8775), (9945), (10725), (10965), (12771), (14561), (15147), (16575), (17415), (19305), (19737), (21285), (22687), (25155), (25245), (27431), (29835), (32175), (32895), (33911), (36465), (36959), (38313), (41023), (41925), (43039), (43683), (49725), (50407), (57915), (59211), (63855), (66041), (68061), (70487), (72369), (72805), (75465), (75735), (83399), (96525), (98685), (99701), (107063), (109395), (110877), (114939), (115181), (118955), (123069), (125207), (125775), (127033), (129117), (129899), (130681), (130727), (130957), (131003), (131049), (142545), (149175), (165491), (178699), (182325), (188437), (189109), (191565), (198123), (204183), (217107), (218323), (218415), (226395), (234329), (247537)])
    ∃ (w_4_0 : mxxMat68),
      let w_5_0 : mxxMat68 := MxxRuntime.matrixMulScalarLeft w_1_0 w_4_0
      ∃ (w_7_0 : mxxMat67),
        ∃ (w_8_0 : mxxMat68),
          let w_9_0 : mxxMat68 := MxxRuntime.matrixMulScalarLeft w_7_0 w_8_0
          let w_10_0 : mxxMat68 := MxxRuntime.matrixAdd w_5_0 w_9_0
          let w_11_0 : mxxMat67 := (MxxRuntime.matrixPolynomial [(1099455004673)] : mxxMat67)
          ∃ (w_12_0 : mxxMat67),
            let w_13_0 : mxxMat67 := MxxRuntime.matrixMulScalarLeft w_11_0 w_12_0
            ∃ (w_14_0 : mxxMat69),
              ∃ (w_15_0 : mxxMat68),
                let w_16_0 : mxxMat68 := MxxRuntime.matrixMulScalarLeft w_13_0 w_15_0
                let w_17_0 : mxxMat68 := MxxRuntime.matrixSub w_10_0 w_16_0
                ∃ (w_18_0 : mxxMat68),
                  let w_19_0 : mxxMat68 := MxxRuntime.matrixAdd w_17_0 w_18_0
                  ∃ (w_20_0 : mxxMat68),
                    let w_21_0 : mxxMat68 := MxxRuntime.matrixMulScalarLeft w_1_0 w_20_0
                    ∃ (w_22_0 : mxxMat68),
                      let w_23_0 : mxxMat68 := MxxRuntime.matrixMulScalarLeft w_7_0 w_22_0
                      let w_24_0 : mxxMat68 := MxxRuntime.matrixAdd w_21_0 w_23_0
                      ∃ (w_25_0 : mxxMat67),
                        let w_26_0 : mxxMat67 := MxxRuntime.matrixMulScalarLeft w_11_0 w_25_0
                        let w_27_0 : mxxMat68 := MxxRuntime.matrixMulScalarLeft w_26_0 w_15_0
                        let w_28_0 : mxxMat68 := MxxRuntime.matrixSub w_24_0 w_27_0
                        ∃ (w_29_0 : mxxMat68),
                          let w_30_0 : mxxMat68 := MxxRuntime.matrixAdd w_28_0 w_29_0
                          parallel_parallel_generatedRoot_7_84.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs

abbrev parallel_parallel_generatedRoot_7_85.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_1 : Nat) (w_0_0 : mxxMat1) (w_2_0 : ByteArray) (w_6_0 : mxxMat1) (w_1_0 : mxxMat67) (w_3_0 : Int) (w_4_0 : mxxMat68) (w_7_0 : mxxMat67) (w_8_0 : mxxMat68) (w_12_0 : mxxMat67) (w_14_0 : mxxMat69) (w_15_0 : mxxMat68) (w_18_0 : mxxMat68) (w_19_0 : mxxMat68) (w_20_0 : mxxMat68) (w_22_0 : mxxMat68) (w_25_0 : mxxMat67) (w_29_0 : mxxMat68) (w_30_0 : mxxMat68) (outputs : mxxMat68 × mxxMat68 × Unit) : Prop :=
  MxxRuntime.modulusReduceRuns w_0_0 w_1_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 32 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (29), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_2_0) w_4_0 ∧
  MxxRuntime.modulusReduceRuns w_6_0 w_7_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (29), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_2_0) w_8_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 32 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (85), (99), (135), (153), (165), (255), (297), (459), (495), (561), (765), (891), (1485), (1683), (2295), (2805), (4455), (5049), (8415), (15147), (25245), (75735)])) w_1_0 w_12_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 1024 52 w_14_0 ∧
  MxxRuntime.centeredRebaseRuns w_14_0 w_15_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [18]) ((4/1)) (26) w_18_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (29), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_2_0) w_20_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (29), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_2_0) w_22_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 32 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (85), (99), (135), (153), (165), (255), (297), (459), (495), (561), (765), (891), (1485), (1683), (2295), (2805), (4455), (5049), (8415), (15147), (25245), (75735)])) w_7_0 w_25_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [29]) ((4/1)) (26) w_29_0 ∧
  outputs = (w_19_0, (w_30_0, ()))

def parallel_parallel_generatedRoot_7_85 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat1 × ByteArray × mxxMat1 × Unit) (outputs : mxxMat68 × mxxMat68 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_2_0 : ByteArray := inputs.2.1
  let w_6_0 : mxxMat1 := inputs.2.2.1
  ∃ (w_1_0 : mxxMat67),
    let w_3_0 : Int := (MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (85), (99), (135), (153), (165), (255), (297), (459), (495), (561), (765), (891), (1485), (1683), (2295), (2805), (4455), (5049), (8415), (15147), (25245), (75735)])
    ∃ (w_4_0 : mxxMat68),
      let w_5_0 : mxxMat68 := MxxRuntime.matrixMulScalarLeft w_1_0 w_4_0
      ∃ (w_7_0 : mxxMat67),
        ∃ (w_8_0 : mxxMat68),
          let w_9_0 : mxxMat68 := MxxRuntime.matrixMulScalarLeft w_7_0 w_8_0
          let w_10_0 : mxxMat68 := MxxRuntime.matrixAdd w_5_0 w_9_0
          let w_11_0 : mxxMat67 := (MxxRuntime.matrixPolynomial [(1099455004673)] : mxxMat67)
          ∃ (w_12_0 : mxxMat67),
            let w_13_0 : mxxMat67 := MxxRuntime.matrixMulScalarLeft w_11_0 w_12_0
            ∃ (w_14_0 : mxxMat69),
              ∃ (w_15_0 : mxxMat68),
                let w_16_0 : mxxMat68 := MxxRuntime.matrixMulScalarLeft w_13_0 w_15_0
                let w_17_0 : mxxMat68 := MxxRuntime.matrixSub w_10_0 w_16_0
                ∃ (w_18_0 : mxxMat68),
                  let w_19_0 : mxxMat68 := MxxRuntime.matrixAdd w_17_0 w_18_0
                  ∃ (w_20_0 : mxxMat68),
                    let w_21_0 : mxxMat68 := MxxRuntime.matrixMulScalarLeft w_1_0 w_20_0
                    ∃ (w_22_0 : mxxMat68),
                      let w_23_0 : mxxMat68 := MxxRuntime.matrixMulScalarLeft w_7_0 w_22_0
                      let w_24_0 : mxxMat68 := MxxRuntime.matrixAdd w_21_0 w_23_0
                      ∃ (w_25_0 : mxxMat67),
                        let w_26_0 : mxxMat67 := MxxRuntime.matrixMulScalarLeft w_11_0 w_25_0
                        let w_27_0 : mxxMat68 := MxxRuntime.matrixMulScalarLeft w_26_0 w_15_0
                        let w_28_0 : mxxMat68 := MxxRuntime.matrixSub w_24_0 w_27_0
                        ∃ (w_29_0 : mxxMat68),
                          let w_30_0 : mxxMat68 := MxxRuntime.matrixAdd w_28_0 w_29_0
                          parallel_parallel_generatedRoot_7_85.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs

abbrev parallel_parallel_generatedRoot_7_86.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_1 : Nat) (w_0_0 : mxxMat1) (w_2_0 : ByteArray) (w_6_0 : mxxMat1) (w_1_0 : mxxMat70) (w_3_0 : Int) (w_4_0 : mxxMat71) (w_7_0 : mxxMat70) (w_8_0 : mxxMat71) (w_12_0 : mxxMat70) (w_14_0 : mxxMat72) (w_15_0 : mxxMat71) (w_18_0 : mxxMat71) (w_19_0 : mxxMat71) (w_20_0 : mxxMat71) (w_22_0 : mxxMat71) (w_25_0 : mxxMat70) (w_29_0 : mxxMat71) (w_30_0 : mxxMat71) (outputs : mxxMat71 × mxxMat71 × Unit) : Prop :=
  MxxRuntime.modulusReduceRuns w_0_0 w_1_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 128 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (3), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_2_0) w_4_0 ∧
  MxxRuntime.modulusReduceRuns w_6_0 w_7_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (3), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_2_0) w_8_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 128 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (65), (85), (99), (129), (135), (153), (165), (195), (255), (297), (325), (387), (459), (495), (561), (585), (645), (765), (891), (975), (1105), (1161), (1485), (1683), (1755), (1935), (2145), (2193), (2295), (2805), (2925), (3315), (3483), (4257), (4455), (5049), (5525), (5805), (6435), (6579), (8385), (8415), (8775), (9945), (10725), (10965), (12771), (14561), (15147), (16575), (17415), (19305), (19737), (21285), (22687), (25155), (25245), (27431), (29835), (32175), (32895), (33911), (36465), (36959), (38313), (41023), (41925), (43039), (43683), (49725), (50407), (57915), (59211), (63855), (66041), (68061), (70487), (72369), (72805), (75465), (75735), (83399), (96525), (98685), (99701), (107063), (109395), (110877), (114939), (115181), (118955), (123069), (125207), (125775), (127033), (129117), (129899), (130681), (130727), (130957), (131003), (131049), (142545), (149175), (165491), (178699), (182325), (188437), (189109), (191565), (198123), (204183), (217107), (218323), (218415), (226395), (234329), (247537)])) w_1_0 w_12_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 1024 156 w_14_0 ∧
  MxxRuntime.centeredRebaseRuns w_14_0 w_15_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [18]) ((4/1)) (26) w_18_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (3), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_2_0) w_20_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (3), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_2_0) w_22_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 128 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (65), (85), (99), (129), (135), (153), (165), (195), (255), (297), (325), (387), (459), (495), (561), (585), (645), (765), (891), (975), (1105), (1161), (1485), (1683), (1755), (1935), (2145), (2193), (2295), (2805), (2925), (3315), (3483), (4257), (4455), (5049), (5525), (5805), (6435), (6579), (8385), (8415), (8775), (9945), (10725), (10965), (12771), (14561), (15147), (16575), (17415), (19305), (19737), (21285), (22687), (25155), (25245), (27431), (29835), (32175), (32895), (33911), (36465), (36959), (38313), (41023), (41925), (43039), (43683), (49725), (50407), (57915), (59211), (63855), (66041), (68061), (70487), (72369), (72805), (75465), (75735), (83399), (96525), (98685), (99701), (107063), (109395), (110877), (114939), (115181), (118955), (123069), (125207), (125775), (127033), (129117), (129899), (130681), (130727), (130957), (131003), (131049), (142545), (149175), (165491), (178699), (182325), (188437), (189109), (191565), (198123), (204183), (217107), (218323), (218415), (226395), (234329), (247537)])) w_7_0 w_25_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [29]) ((4/1)) (26) w_29_0 ∧
  outputs = (w_19_0, (w_30_0, ()))

def parallel_parallel_generatedRoot_7_86 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat1 × ByteArray × mxxMat1 × Unit) (outputs : mxxMat71 × mxxMat71 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_2_0 : ByteArray := inputs.2.1
  let w_6_0 : mxxMat1 := inputs.2.2.1
  ∃ (w_1_0 : mxxMat70),
    let w_3_0 : Int := (MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (65), (85), (99), (129), (135), (153), (165), (195), (255), (297), (325), (387), (459), (495), (561), (585), (645), (765), (891), (975), (1105), (1161), (1485), (1683), (1755), (1935), (2145), (2193), (2295), (2805), (2925), (3315), (3483), (4257), (4455), (5049), (5525), (5805), (6435), (6579), (8385), (8415), (8775), (9945), (10725), (10965), (12771), (14561), (15147), (16575), (17415), (19305), (19737), (21285), (22687), (25155), (25245), (27431), (29835), (32175), (32895), (33911), (36465), (36959), (38313), (41023), (41925), (43039), (43683), (49725), (50407), (57915), (59211), (63855), (66041), (68061), (70487), (72369), (72805), (75465), (75735), (83399), (96525), (98685), (99701), (107063), (109395), (110877), (114939), (115181), (118955), (123069), (125207), (125775), (127033), (129117), (129899), (130681), (130727), (130957), (131003), (131049), (142545), (149175), (165491), (178699), (182325), (188437), (189109), (191565), (198123), (204183), (217107), (218323), (218415), (226395), (234329), (247537)])
    ∃ (w_4_0 : mxxMat71),
      let w_5_0 : mxxMat71 := MxxRuntime.matrixMulScalarLeft w_1_0 w_4_0
      ∃ (w_7_0 : mxxMat70),
        ∃ (w_8_0 : mxxMat71),
          let w_9_0 : mxxMat71 := MxxRuntime.matrixMulScalarLeft w_7_0 w_8_0
          let w_10_0 : mxxMat71 := MxxRuntime.matrixAdd w_5_0 w_9_0
          let w_11_0 : mxxMat70 := (MxxRuntime.matrixPolynomial [(1099388157953)] : mxxMat70)
          ∃ (w_12_0 : mxxMat70),
            let w_13_0 : mxxMat70 := MxxRuntime.matrixMulScalarLeft w_11_0 w_12_0
            ∃ (w_14_0 : mxxMat72),
              ∃ (w_15_0 : mxxMat71),
                let w_16_0 : mxxMat71 := MxxRuntime.matrixMulScalarLeft w_13_0 w_15_0
                let w_17_0 : mxxMat71 := MxxRuntime.matrixSub w_10_0 w_16_0
                ∃ (w_18_0 : mxxMat71),
                  let w_19_0 : mxxMat71 := MxxRuntime.matrixAdd w_17_0 w_18_0
                  ∃ (w_20_0 : mxxMat71),
                    let w_21_0 : mxxMat71 := MxxRuntime.matrixMulScalarLeft w_1_0 w_20_0
                    ∃ (w_22_0 : mxxMat71),
                      let w_23_0 : mxxMat71 := MxxRuntime.matrixMulScalarLeft w_7_0 w_22_0
                      let w_24_0 : mxxMat71 := MxxRuntime.matrixAdd w_21_0 w_23_0
                      ∃ (w_25_0 : mxxMat70),
                        let w_26_0 : mxxMat70 := MxxRuntime.matrixMulScalarLeft w_11_0 w_25_0
                        let w_27_0 : mxxMat71 := MxxRuntime.matrixMulScalarLeft w_26_0 w_15_0
                        let w_28_0 : mxxMat71 := MxxRuntime.matrixSub w_24_0 w_27_0
                        ∃ (w_29_0 : mxxMat71),
                          let w_30_0 : mxxMat71 := MxxRuntime.matrixAdd w_28_0 w_29_0
                          parallel_parallel_generatedRoot_7_86.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs

abbrev parallel_parallel_generatedRoot_7_87.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_1 : Nat) (w_0_0 : mxxMat1) (w_2_0 : ByteArray) (w_6_0 : mxxMat1) (w_1_0 : mxxMat70) (w_3_0 : Int) (w_4_0 : mxxMat71) (w_7_0 : mxxMat70) (w_8_0 : mxxMat71) (w_12_0 : mxxMat70) (w_14_0 : mxxMat72) (w_15_0 : mxxMat71) (w_18_0 : mxxMat71) (w_19_0 : mxxMat71) (w_20_0 : mxxMat71) (w_22_0 : mxxMat71) (w_25_0 : mxxMat70) (w_29_0 : mxxMat71) (w_30_0 : mxxMat71) (outputs : mxxMat71 × mxxMat71 × Unit) : Prop :=
  MxxRuntime.modulusReduceRuns w_0_0 w_1_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 32 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (3), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_2_0) w_4_0 ∧
  MxxRuntime.modulusReduceRuns w_6_0 w_7_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (3), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_2_0) w_8_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 32 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (85), (99), (135), (153), (165), (255), (297), (459), (495), (561), (765), (891), (1485), (1683), (2295), (2805), (4455), (5049), (8415), (15147), (25245), (75735)])) w_1_0 w_12_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 1024 156 w_14_0 ∧
  MxxRuntime.centeredRebaseRuns w_14_0 w_15_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [18]) ((4/1)) (26) w_18_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (3), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_2_0) w_20_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (3), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_2_0) w_22_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 32 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (85), (99), (135), (153), (165), (255), (297), (459), (495), (561), (765), (891), (1485), (1683), (2295), (2805), (4455), (5049), (8415), (15147), (25245), (75735)])) w_7_0 w_25_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [29]) ((4/1)) (26) w_29_0 ∧
  outputs = (w_19_0, (w_30_0, ()))

def parallel_parallel_generatedRoot_7_87 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat1 × ByteArray × mxxMat1 × Unit) (outputs : mxxMat71 × mxxMat71 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_2_0 : ByteArray := inputs.2.1
  let w_6_0 : mxxMat1 := inputs.2.2.1
  ∃ (w_1_0 : mxxMat70),
    let w_3_0 : Int := (MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (85), (99), (135), (153), (165), (255), (297), (459), (495), (561), (765), (891), (1485), (1683), (2295), (2805), (4455), (5049), (8415), (15147), (25245), (75735)])
    ∃ (w_4_0 : mxxMat71),
      let w_5_0 : mxxMat71 := MxxRuntime.matrixMulScalarLeft w_1_0 w_4_0
      ∃ (w_7_0 : mxxMat70),
        ∃ (w_8_0 : mxxMat71),
          let w_9_0 : mxxMat71 := MxxRuntime.matrixMulScalarLeft w_7_0 w_8_0
          let w_10_0 : mxxMat71 := MxxRuntime.matrixAdd w_5_0 w_9_0
          let w_11_0 : mxxMat70 := (MxxRuntime.matrixPolynomial [(1099388157953)] : mxxMat70)
          ∃ (w_12_0 : mxxMat70),
            let w_13_0 : mxxMat70 := MxxRuntime.matrixMulScalarLeft w_11_0 w_12_0
            ∃ (w_14_0 : mxxMat72),
              ∃ (w_15_0 : mxxMat71),
                let w_16_0 : mxxMat71 := MxxRuntime.matrixMulScalarLeft w_13_0 w_15_0
                let w_17_0 : mxxMat71 := MxxRuntime.matrixSub w_10_0 w_16_0
                ∃ (w_18_0 : mxxMat71),
                  let w_19_0 : mxxMat71 := MxxRuntime.matrixAdd w_17_0 w_18_0
                  ∃ (w_20_0 : mxxMat71),
                    let w_21_0 : mxxMat71 := MxxRuntime.matrixMulScalarLeft w_1_0 w_20_0
                    ∃ (w_22_0 : mxxMat71),
                      let w_23_0 : mxxMat71 := MxxRuntime.matrixMulScalarLeft w_7_0 w_22_0
                      let w_24_0 : mxxMat71 := MxxRuntime.matrixAdd w_21_0 w_23_0
                      ∃ (w_25_0 : mxxMat70),
                        let w_26_0 : mxxMat70 := MxxRuntime.matrixMulScalarLeft w_11_0 w_25_0
                        let w_27_0 : mxxMat71 := MxxRuntime.matrixMulScalarLeft w_26_0 w_15_0
                        let w_28_0 : mxxMat71 := MxxRuntime.matrixSub w_24_0 w_27_0
                        ∃ (w_29_0 : mxxMat71),
                          let w_30_0 : mxxMat71 := MxxRuntime.matrixAdd w_28_0 w_29_0
                          parallel_parallel_generatedRoot_7_87.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs

abbrev parallel_parallel_generatedRoot_7_88.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_1 : Nat) (w_0_0 : mxxMat1) (w_2_0 : ByteArray) (w_6_0 : mxxMat1) (w_1_0 : mxxMat73) (w_3_0 : Int) (w_4_0 : mxxMat74) (w_7_0 : mxxMat73) (w_8_0 : mxxMat74) (w_12_0 : mxxMat73) (w_14_0 : mxxMat75) (w_15_0 : mxxMat74) (w_18_0 : mxxMat74) (w_19_0 : mxxMat74) (w_20_0 : mxxMat74) (w_22_0 : mxxMat74) (w_25_0 : mxxMat73) (w_29_0 : mxxMat74) (w_30_0 : mxxMat74) (outputs : mxxMat74 × mxxMat74 × Unit) : Prop :=
  MxxRuntime.modulusReduceRuns w_0_0 w_1_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 128 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (30), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_2_0) w_4_0 ∧
  MxxRuntime.modulusReduceRuns w_6_0 w_7_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (30), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_2_0) w_8_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 128 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (65), (85), (99), (129), (135), (153), (165), (195), (255), (297), (325), (387), (459), (495), (561), (585), (645), (765), (891), (975), (1105), (1161), (1485), (1683), (1755), (1935), (2145), (2193), (2295), (2805), (2925), (3315), (3483), (4257), (4455), (5049), (5525), (5805), (6435), (6579), (8385), (8415), (8775), (9945), (10725), (10965), (12771), (14561), (15147), (16575), (17415), (19305), (19737), (21285), (22687), (25155), (25245), (27431), (29835), (32175), (32895), (33911), (36465), (36959), (38313), (41023), (41925), (43039), (43683), (49725), (50407), (57915), (59211), (63855), (66041), (68061), (70487), (72369), (72805), (75465), (75735), (83399), (96525), (98685), (99701), (107063), (109395), (110877), (114939), (115181), (118955), (123069), (125207), (125775), (127033), (129117), (129899), (130681), (130727), (130957), (131003), (131049), (142545), (149175), (165491), (178699), (182325), (188437), (189109), (191565), (198123), (204183), (217107), (218323), (218415), (226395), (234329), (247537)])) w_1_0 w_12_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 1024 48 w_14_0 ∧
  MxxRuntime.centeredRebaseRuns w_14_0 w_15_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [18]) ((4/1)) (26) w_18_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (30), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_2_0) w_20_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (30), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_2_0) w_22_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 128 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (65), (85), (99), (129), (135), (153), (165), (195), (255), (297), (325), (387), (459), (495), (561), (585), (645), (765), (891), (975), (1105), (1161), (1485), (1683), (1755), (1935), (2145), (2193), (2295), (2805), (2925), (3315), (3483), (4257), (4455), (5049), (5525), (5805), (6435), (6579), (8385), (8415), (8775), (9945), (10725), (10965), (12771), (14561), (15147), (16575), (17415), (19305), (19737), (21285), (22687), (25155), (25245), (27431), (29835), (32175), (32895), (33911), (36465), (36959), (38313), (41023), (41925), (43039), (43683), (49725), (50407), (57915), (59211), (63855), (66041), (68061), (70487), (72369), (72805), (75465), (75735), (83399), (96525), (98685), (99701), (107063), (109395), (110877), (114939), (115181), (118955), (123069), (125207), (125775), (127033), (129117), (129899), (130681), (130727), (130957), (131003), (131049), (142545), (149175), (165491), (178699), (182325), (188437), (189109), (191565), (198123), (204183), (217107), (218323), (218415), (226395), (234329), (247537)])) w_7_0 w_25_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [29]) ((4/1)) (26) w_29_0 ∧
  outputs = (w_19_0, (w_30_0, ()))

def parallel_parallel_generatedRoot_7_88 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat1 × ByteArray × mxxMat1 × Unit) (outputs : mxxMat74 × mxxMat74 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_2_0 : ByteArray := inputs.2.1
  let w_6_0 : mxxMat1 := inputs.2.2.1
  ∃ (w_1_0 : mxxMat73),
    let w_3_0 : Int := (MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (65), (85), (99), (129), (135), (153), (165), (195), (255), (297), (325), (387), (459), (495), (561), (585), (645), (765), (891), (975), (1105), (1161), (1485), (1683), (1755), (1935), (2145), (2193), (2295), (2805), (2925), (3315), (3483), (4257), (4455), (5049), (5525), (5805), (6435), (6579), (8385), (8415), (8775), (9945), (10725), (10965), (12771), (14561), (15147), (16575), (17415), (19305), (19737), (21285), (22687), (25155), (25245), (27431), (29835), (32175), (32895), (33911), (36465), (36959), (38313), (41023), (41925), (43039), (43683), (49725), (50407), (57915), (59211), (63855), (66041), (68061), (70487), (72369), (72805), (75465), (75735), (83399), (96525), (98685), (99701), (107063), (109395), (110877), (114939), (115181), (118955), (123069), (125207), (125775), (127033), (129117), (129899), (130681), (130727), (130957), (131003), (131049), (142545), (149175), (165491), (178699), (182325), (188437), (189109), (191565), (198123), (204183), (217107), (218323), (218415), (226395), (234329), (247537)])
    ∃ (w_4_0 : mxxMat74),
      let w_5_0 : mxxMat74 := MxxRuntime.matrixMulScalarLeft w_1_0 w_4_0
      ∃ (w_7_0 : mxxMat73),
        ∃ (w_8_0 : mxxMat74),
          let w_9_0 : mxxMat74 := MxxRuntime.matrixMulScalarLeft w_7_0 w_8_0
          let w_10_0 : mxxMat74 := MxxRuntime.matrixAdd w_5_0 w_9_0
          let w_11_0 : mxxMat73 := (MxxRuntime.matrixPolynomial [(1099461820417)] : mxxMat73)
          ∃ (w_12_0 : mxxMat73),
            let w_13_0 : mxxMat73 := MxxRuntime.matrixMulScalarLeft w_11_0 w_12_0
            ∃ (w_14_0 : mxxMat75),
              ∃ (w_15_0 : mxxMat74),
                let w_16_0 : mxxMat74 := MxxRuntime.matrixMulScalarLeft w_13_0 w_15_0
                let w_17_0 : mxxMat74 := MxxRuntime.matrixSub w_10_0 w_16_0
                ∃ (w_18_0 : mxxMat74),
                  let w_19_0 : mxxMat74 := MxxRuntime.matrixAdd w_17_0 w_18_0
                  ∃ (w_20_0 : mxxMat74),
                    let w_21_0 : mxxMat74 := MxxRuntime.matrixMulScalarLeft w_1_0 w_20_0
                    ∃ (w_22_0 : mxxMat74),
                      let w_23_0 : mxxMat74 := MxxRuntime.matrixMulScalarLeft w_7_0 w_22_0
                      let w_24_0 : mxxMat74 := MxxRuntime.matrixAdd w_21_0 w_23_0
                      ∃ (w_25_0 : mxxMat73),
                        let w_26_0 : mxxMat73 := MxxRuntime.matrixMulScalarLeft w_11_0 w_25_0
                        let w_27_0 : mxxMat74 := MxxRuntime.matrixMulScalarLeft w_26_0 w_15_0
                        let w_28_0 : mxxMat74 := MxxRuntime.matrixSub w_24_0 w_27_0
                        ∃ (w_29_0 : mxxMat74),
                          let w_30_0 : mxxMat74 := MxxRuntime.matrixAdd w_28_0 w_29_0
                          parallel_parallel_generatedRoot_7_88.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs

abbrev parallel_parallel_generatedRoot_7_89.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_1 : Nat) (w_0_0 : mxxMat1) (w_2_0 : ByteArray) (w_6_0 : mxxMat1) (w_1_0 : mxxMat73) (w_3_0 : Int) (w_4_0 : mxxMat74) (w_7_0 : mxxMat73) (w_8_0 : mxxMat74) (w_12_0 : mxxMat73) (w_14_0 : mxxMat75) (w_15_0 : mxxMat74) (w_18_0 : mxxMat74) (w_19_0 : mxxMat74) (w_20_0 : mxxMat74) (w_22_0 : mxxMat74) (w_25_0 : mxxMat73) (w_29_0 : mxxMat74) (w_30_0 : mxxMat74) (outputs : mxxMat74 × mxxMat74 × Unit) : Prop :=
  MxxRuntime.modulusReduceRuns w_0_0 w_1_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 32 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (30), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_2_0) w_4_0 ∧
  MxxRuntime.modulusReduceRuns w_6_0 w_7_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (30), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_2_0) w_8_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 32 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (85), (99), (135), (153), (165), (255), (297), (459), (495), (561), (765), (891), (1485), (1683), (2295), (2805), (4455), (5049), (8415), (15147), (25245), (75735)])) w_1_0 w_12_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 1024 48 w_14_0 ∧
  MxxRuntime.centeredRebaseRuns w_14_0 w_15_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [18]) ((4/1)) (26) w_18_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (30), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_2_0) w_20_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (30), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_2_0) w_22_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 32 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (85), (99), (135), (153), (165), (255), (297), (459), (495), (561), (765), (891), (1485), (1683), (2295), (2805), (4455), (5049), (8415), (15147), (25245), (75735)])) w_7_0 w_25_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [29]) ((4/1)) (26) w_29_0 ∧
  outputs = (w_19_0, (w_30_0, ()))

def parallel_parallel_generatedRoot_7_89 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat1 × ByteArray × mxxMat1 × Unit) (outputs : mxxMat74 × mxxMat74 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_2_0 : ByteArray := inputs.2.1
  let w_6_0 : mxxMat1 := inputs.2.2.1
  ∃ (w_1_0 : mxxMat73),
    let w_3_0 : Int := (MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (85), (99), (135), (153), (165), (255), (297), (459), (495), (561), (765), (891), (1485), (1683), (2295), (2805), (4455), (5049), (8415), (15147), (25245), (75735)])
    ∃ (w_4_0 : mxxMat74),
      let w_5_0 : mxxMat74 := MxxRuntime.matrixMulScalarLeft w_1_0 w_4_0
      ∃ (w_7_0 : mxxMat73),
        ∃ (w_8_0 : mxxMat74),
          let w_9_0 : mxxMat74 := MxxRuntime.matrixMulScalarLeft w_7_0 w_8_0
          let w_10_0 : mxxMat74 := MxxRuntime.matrixAdd w_5_0 w_9_0
          let w_11_0 : mxxMat73 := (MxxRuntime.matrixPolynomial [(1099461820417)] : mxxMat73)
          ∃ (w_12_0 : mxxMat73),
            let w_13_0 : mxxMat73 := MxxRuntime.matrixMulScalarLeft w_11_0 w_12_0
            ∃ (w_14_0 : mxxMat75),
              ∃ (w_15_0 : mxxMat74),
                let w_16_0 : mxxMat74 := MxxRuntime.matrixMulScalarLeft w_13_0 w_15_0
                let w_17_0 : mxxMat74 := MxxRuntime.matrixSub w_10_0 w_16_0
                ∃ (w_18_0 : mxxMat74),
                  let w_19_0 : mxxMat74 := MxxRuntime.matrixAdd w_17_0 w_18_0
                  ∃ (w_20_0 : mxxMat74),
                    let w_21_0 : mxxMat74 := MxxRuntime.matrixMulScalarLeft w_1_0 w_20_0
                    ∃ (w_22_0 : mxxMat74),
                      let w_23_0 : mxxMat74 := MxxRuntime.matrixMulScalarLeft w_7_0 w_22_0
                      let w_24_0 : mxxMat74 := MxxRuntime.matrixAdd w_21_0 w_23_0
                      ∃ (w_25_0 : mxxMat73),
                        let w_26_0 : mxxMat73 := MxxRuntime.matrixMulScalarLeft w_11_0 w_25_0
                        let w_27_0 : mxxMat74 := MxxRuntime.matrixMulScalarLeft w_26_0 w_15_0
                        let w_28_0 : mxxMat74 := MxxRuntime.matrixSub w_24_0 w_27_0
                        ∃ (w_29_0 : mxxMat74),
                          let w_30_0 : mxxMat74 := MxxRuntime.matrixAdd w_28_0 w_29_0
                          parallel_parallel_generatedRoot_7_89.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs

abbrev parallel_parallel_generatedRoot_7_90.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_1 : Nat) (w_0_0 : mxxMat1) (w_2_0 : ByteArray) (w_6_0 : mxxMat1) (w_1_0 : mxxMat76) (w_3_0 : Int) (w_4_0 : mxxMat77) (w_7_0 : mxxMat76) (w_8_0 : mxxMat77) (w_12_0 : mxxMat76) (w_14_0 : mxxMat78) (w_15_0 : mxxMat77) (w_18_0 : mxxMat77) (w_19_0 : mxxMat77) (w_20_0 : mxxMat77) (w_22_0 : mxxMat77) (w_25_0 : mxxMat76) (w_29_0 : mxxMat77) (w_30_0 : mxxMat77) (outputs : mxxMat77 × mxxMat77 × Unit) : Prop :=
  MxxRuntime.modulusReduceRuns w_0_0 w_1_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 128 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (31), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_2_0) w_4_0 ∧
  MxxRuntime.modulusReduceRuns w_6_0 w_7_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (31), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_2_0) w_8_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 128 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (65), (85), (99), (129), (135), (153), (165), (195), (255), (297), (325), (387), (459), (495), (561), (585), (645), (765), (891), (975), (1105), (1161), (1485), (1683), (1755), (1935), (2145), (2193), (2295), (2805), (2925), (3315), (3483), (4257), (4455), (5049), (5525), (5805), (6435), (6579), (8385), (8415), (8775), (9945), (10725), (10965), (12771), (14561), (15147), (16575), (17415), (19305), (19737), (21285), (22687), (25155), (25245), (27431), (29835), (32175), (32895), (33911), (36465), (36959), (38313), (41023), (41925), (43039), (43683), (49725), (50407), (57915), (59211), (63855), (66041), (68061), (70487), (72369), (72805), (75465), (75735), (83399), (96525), (98685), (99701), (107063), (109395), (110877), (114939), (115181), (118955), (123069), (125207), (125775), (127033), (129117), (129899), (130681), (130727), (130957), (131003), (131049), (142545), (149175), (165491), (178699), (182325), (188437), (189109), (191565), (198123), (204183), (217107), (218323), (218415), (226395), (234329), (247537)])) w_1_0 w_12_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 1024 44 w_14_0 ∧
  MxxRuntime.centeredRebaseRuns w_14_0 w_15_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [18]) ((4/1)) (26) w_18_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (31), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_2_0) w_20_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (31), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_2_0) w_22_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 128 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (65), (85), (99), (129), (135), (153), (165), (195), (255), (297), (325), (387), (459), (495), (561), (585), (645), (765), (891), (975), (1105), (1161), (1485), (1683), (1755), (1935), (2145), (2193), (2295), (2805), (2925), (3315), (3483), (4257), (4455), (5049), (5525), (5805), (6435), (6579), (8385), (8415), (8775), (9945), (10725), (10965), (12771), (14561), (15147), (16575), (17415), (19305), (19737), (21285), (22687), (25155), (25245), (27431), (29835), (32175), (32895), (33911), (36465), (36959), (38313), (41023), (41925), (43039), (43683), (49725), (50407), (57915), (59211), (63855), (66041), (68061), (70487), (72369), (72805), (75465), (75735), (83399), (96525), (98685), (99701), (107063), (109395), (110877), (114939), (115181), (118955), (123069), (125207), (125775), (127033), (129117), (129899), (130681), (130727), (130957), (131003), (131049), (142545), (149175), (165491), (178699), (182325), (188437), (189109), (191565), (198123), (204183), (217107), (218323), (218415), (226395), (234329), (247537)])) w_7_0 w_25_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [29]) ((4/1)) (26) w_29_0 ∧
  outputs = (w_19_0, (w_30_0, ()))

def parallel_parallel_generatedRoot_7_90 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat1 × ByteArray × mxxMat1 × Unit) (outputs : mxxMat77 × mxxMat77 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_2_0 : ByteArray := inputs.2.1
  let w_6_0 : mxxMat1 := inputs.2.2.1
  ∃ (w_1_0 : mxxMat76),
    let w_3_0 : Int := (MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (65), (85), (99), (129), (135), (153), (165), (195), (255), (297), (325), (387), (459), (495), (561), (585), (645), (765), (891), (975), (1105), (1161), (1485), (1683), (1755), (1935), (2145), (2193), (2295), (2805), (2925), (3315), (3483), (4257), (4455), (5049), (5525), (5805), (6435), (6579), (8385), (8415), (8775), (9945), (10725), (10965), (12771), (14561), (15147), (16575), (17415), (19305), (19737), (21285), (22687), (25155), (25245), (27431), (29835), (32175), (32895), (33911), (36465), (36959), (38313), (41023), (41925), (43039), (43683), (49725), (50407), (57915), (59211), (63855), (66041), (68061), (70487), (72369), (72805), (75465), (75735), (83399), (96525), (98685), (99701), (107063), (109395), (110877), (114939), (115181), (118955), (123069), (125207), (125775), (127033), (129117), (129899), (130681), (130727), (130957), (131003), (131049), (142545), (149175), (165491), (178699), (182325), (188437), (189109), (191565), (198123), (204183), (217107), (218323), (218415), (226395), (234329), (247537)])
    ∃ (w_4_0 : mxxMat77),
      let w_5_0 : mxxMat77 := MxxRuntime.matrixMulScalarLeft w_1_0 w_4_0
      ∃ (w_7_0 : mxxMat76),
        ∃ (w_8_0 : mxxMat77),
          let w_9_0 : mxxMat77 := MxxRuntime.matrixMulScalarLeft w_7_0 w_8_0
          let w_10_0 : mxxMat77 := MxxRuntime.matrixAdd w_5_0 w_9_0
          let w_11_0 : mxxMat76 := (MxxRuntime.matrixPolynomial [(1099469684737)] : mxxMat76)
          ∃ (w_12_0 : mxxMat76),
            let w_13_0 : mxxMat76 := MxxRuntime.matrixMulScalarLeft w_11_0 w_12_0
            ∃ (w_14_0 : mxxMat78),
              ∃ (w_15_0 : mxxMat77),
                let w_16_0 : mxxMat77 := MxxRuntime.matrixMulScalarLeft w_13_0 w_15_0
                let w_17_0 : mxxMat77 := MxxRuntime.matrixSub w_10_0 w_16_0
                ∃ (w_18_0 : mxxMat77),
                  let w_19_0 : mxxMat77 := MxxRuntime.matrixAdd w_17_0 w_18_0
                  ∃ (w_20_0 : mxxMat77),
                    let w_21_0 : mxxMat77 := MxxRuntime.matrixMulScalarLeft w_1_0 w_20_0
                    ∃ (w_22_0 : mxxMat77),
                      let w_23_0 : mxxMat77 := MxxRuntime.matrixMulScalarLeft w_7_0 w_22_0
                      let w_24_0 : mxxMat77 := MxxRuntime.matrixAdd w_21_0 w_23_0
                      ∃ (w_25_0 : mxxMat76),
                        let w_26_0 : mxxMat76 := MxxRuntime.matrixMulScalarLeft w_11_0 w_25_0
                        let w_27_0 : mxxMat77 := MxxRuntime.matrixMulScalarLeft w_26_0 w_15_0
                        let w_28_0 : mxxMat77 := MxxRuntime.matrixSub w_24_0 w_27_0
                        ∃ (w_29_0 : mxxMat77),
                          let w_30_0 : mxxMat77 := MxxRuntime.matrixAdd w_28_0 w_29_0
                          parallel_parallel_generatedRoot_7_90.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs

abbrev parallel_parallel_generatedRoot_7_91.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_1 : Nat) (w_0_0 : mxxMat1) (w_2_0 : ByteArray) (w_6_0 : mxxMat1) (w_1_0 : mxxMat76) (w_3_0 : Int) (w_4_0 : mxxMat77) (w_7_0 : mxxMat76) (w_8_0 : mxxMat77) (w_12_0 : mxxMat76) (w_14_0 : mxxMat78) (w_15_0 : mxxMat77) (w_18_0 : mxxMat77) (w_19_0 : mxxMat77) (w_20_0 : mxxMat77) (w_22_0 : mxxMat77) (w_25_0 : mxxMat76) (w_29_0 : mxxMat77) (w_30_0 : mxxMat77) (outputs : mxxMat77 × mxxMat77 × Unit) : Prop :=
  MxxRuntime.modulusReduceRuns w_0_0 w_1_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 32 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (31), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_2_0) w_4_0 ∧
  MxxRuntime.modulusReduceRuns w_6_0 w_7_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (31), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_2_0) w_8_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 32 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (85), (99), (135), (153), (165), (255), (297), (459), (495), (561), (765), (891), (1485), (1683), (2295), (2805), (4455), (5049), (8415), (15147), (25245), (75735)])) w_1_0 w_12_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 1024 44 w_14_0 ∧
  MxxRuntime.centeredRebaseRuns w_14_0 w_15_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [18]) ((4/1)) (26) w_18_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (31), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_2_0) w_20_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (31), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_2_0) w_22_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 32 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (85), (99), (135), (153), (165), (255), (297), (459), (495), (561), (765), (891), (1485), (1683), (2295), (2805), (4455), (5049), (8415), (15147), (25245), (75735)])) w_7_0 w_25_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [29]) ((4/1)) (26) w_29_0 ∧
  outputs = (w_19_0, (w_30_0, ()))

def parallel_parallel_generatedRoot_7_91 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat1 × ByteArray × mxxMat1 × Unit) (outputs : mxxMat77 × mxxMat77 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_2_0 : ByteArray := inputs.2.1
  let w_6_0 : mxxMat1 := inputs.2.2.1
  ∃ (w_1_0 : mxxMat76),
    let w_3_0 : Int := (MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (85), (99), (135), (153), (165), (255), (297), (459), (495), (561), (765), (891), (1485), (1683), (2295), (2805), (4455), (5049), (8415), (15147), (25245), (75735)])
    ∃ (w_4_0 : mxxMat77),
      let w_5_0 : mxxMat77 := MxxRuntime.matrixMulScalarLeft w_1_0 w_4_0
      ∃ (w_7_0 : mxxMat76),
        ∃ (w_8_0 : mxxMat77),
          let w_9_0 : mxxMat77 := MxxRuntime.matrixMulScalarLeft w_7_0 w_8_0
          let w_10_0 : mxxMat77 := MxxRuntime.matrixAdd w_5_0 w_9_0
          let w_11_0 : mxxMat76 := (MxxRuntime.matrixPolynomial [(1099469684737)] : mxxMat76)
          ∃ (w_12_0 : mxxMat76),
            let w_13_0 : mxxMat76 := MxxRuntime.matrixMulScalarLeft w_11_0 w_12_0
            ∃ (w_14_0 : mxxMat78),
              ∃ (w_15_0 : mxxMat77),
                let w_16_0 : mxxMat77 := MxxRuntime.matrixMulScalarLeft w_13_0 w_15_0
                let w_17_0 : mxxMat77 := MxxRuntime.matrixSub w_10_0 w_16_0
                ∃ (w_18_0 : mxxMat77),
                  let w_19_0 : mxxMat77 := MxxRuntime.matrixAdd w_17_0 w_18_0
                  ∃ (w_20_0 : mxxMat77),
                    let w_21_0 : mxxMat77 := MxxRuntime.matrixMulScalarLeft w_1_0 w_20_0
                    ∃ (w_22_0 : mxxMat77),
                      let w_23_0 : mxxMat77 := MxxRuntime.matrixMulScalarLeft w_7_0 w_22_0
                      let w_24_0 : mxxMat77 := MxxRuntime.matrixAdd w_21_0 w_23_0
                      ∃ (w_25_0 : mxxMat76),
                        let w_26_0 : mxxMat76 := MxxRuntime.matrixMulScalarLeft w_11_0 w_25_0
                        let w_27_0 : mxxMat77 := MxxRuntime.matrixMulScalarLeft w_26_0 w_15_0
                        let w_28_0 : mxxMat77 := MxxRuntime.matrixSub w_24_0 w_27_0
                        ∃ (w_29_0 : mxxMat77),
                          let w_30_0 : mxxMat77 := MxxRuntime.matrixAdd w_28_0 w_29_0
                          parallel_parallel_generatedRoot_7_91.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs

abbrev parallel_parallel_generatedRoot_7_92.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_1 : Nat) (w_0_0 : mxxMat1) (w_2_0 : ByteArray) (w_6_0 : mxxMat1) (w_1_0 : mxxMat79) (w_3_0 : Int) (w_4_0 : mxxMat80) (w_7_0 : mxxMat79) (w_8_0 : mxxMat80) (w_12_0 : mxxMat79) (w_14_0 : mxxMat81) (w_15_0 : mxxMat80) (w_18_0 : mxxMat80) (w_19_0 : mxxMat80) (w_20_0 : mxxMat80) (w_22_0 : mxxMat80) (w_25_0 : mxxMat79) (w_29_0 : mxxMat80) (w_30_0 : mxxMat80) (outputs : mxxMat80 × mxxMat80 × Unit) : Prop :=
  MxxRuntime.modulusReduceRuns w_0_0 w_1_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 128 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (32), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_2_0) w_4_0 ∧
  MxxRuntime.modulusReduceRuns w_6_0 w_7_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (32), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_2_0) w_8_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 128 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (65), (85), (99), (129), (135), (153), (165), (195), (255), (297), (325), (387), (459), (495), (561), (585), (645), (765), (891), (975), (1105), (1161), (1485), (1683), (1755), (1935), (2145), (2193), (2295), (2805), (2925), (3315), (3483), (4257), (4455), (5049), (5525), (5805), (6435), (6579), (8385), (8415), (8775), (9945), (10725), (10965), (12771), (14561), (15147), (16575), (17415), (19305), (19737), (21285), (22687), (25155), (25245), (27431), (29835), (32175), (32895), (33911), (36465), (36959), (38313), (41023), (41925), (43039), (43683), (49725), (50407), (57915), (59211), (63855), (66041), (68061), (70487), (72369), (72805), (75465), (75735), (83399), (96525), (98685), (99701), (107063), (109395), (110877), (114939), (115181), (118955), (123069), (125207), (125775), (127033), (129117), (129899), (130681), (130727), (130957), (131003), (131049), (142545), (149175), (165491), (178699), (182325), (188437), (189109), (191565), (198123), (204183), (217107), (218323), (218415), (226395), (234329), (247537)])) w_1_0 w_12_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 1024 40 w_14_0 ∧
  MxxRuntime.centeredRebaseRuns w_14_0 w_15_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [18]) ((4/1)) (26) w_18_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (32), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_2_0) w_20_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (32), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_2_0) w_22_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 128 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (65), (85), (99), (129), (135), (153), (165), (195), (255), (297), (325), (387), (459), (495), (561), (585), (645), (765), (891), (975), (1105), (1161), (1485), (1683), (1755), (1935), (2145), (2193), (2295), (2805), (2925), (3315), (3483), (4257), (4455), (5049), (5525), (5805), (6435), (6579), (8385), (8415), (8775), (9945), (10725), (10965), (12771), (14561), (15147), (16575), (17415), (19305), (19737), (21285), (22687), (25155), (25245), (27431), (29835), (32175), (32895), (33911), (36465), (36959), (38313), (41023), (41925), (43039), (43683), (49725), (50407), (57915), (59211), (63855), (66041), (68061), (70487), (72369), (72805), (75465), (75735), (83399), (96525), (98685), (99701), (107063), (109395), (110877), (114939), (115181), (118955), (123069), (125207), (125775), (127033), (129117), (129899), (130681), (130727), (130957), (131003), (131049), (142545), (149175), (165491), (178699), (182325), (188437), (189109), (191565), (198123), (204183), (217107), (218323), (218415), (226395), (234329), (247537)])) w_7_0 w_25_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [29]) ((4/1)) (26) w_29_0 ∧
  outputs = (w_19_0, (w_30_0, ()))

def parallel_parallel_generatedRoot_7_92 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat1 × ByteArray × mxxMat1 × Unit) (outputs : mxxMat80 × mxxMat80 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_2_0 : ByteArray := inputs.2.1
  let w_6_0 : mxxMat1 := inputs.2.2.1
  ∃ (w_1_0 : mxxMat79),
    let w_3_0 : Int := (MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (65), (85), (99), (129), (135), (153), (165), (195), (255), (297), (325), (387), (459), (495), (561), (585), (645), (765), (891), (975), (1105), (1161), (1485), (1683), (1755), (1935), (2145), (2193), (2295), (2805), (2925), (3315), (3483), (4257), (4455), (5049), (5525), (5805), (6435), (6579), (8385), (8415), (8775), (9945), (10725), (10965), (12771), (14561), (15147), (16575), (17415), (19305), (19737), (21285), (22687), (25155), (25245), (27431), (29835), (32175), (32895), (33911), (36465), (36959), (38313), (41023), (41925), (43039), (43683), (49725), (50407), (57915), (59211), (63855), (66041), (68061), (70487), (72369), (72805), (75465), (75735), (83399), (96525), (98685), (99701), (107063), (109395), (110877), (114939), (115181), (118955), (123069), (125207), (125775), (127033), (129117), (129899), (130681), (130727), (130957), (131003), (131049), (142545), (149175), (165491), (178699), (182325), (188437), (189109), (191565), (198123), (204183), (217107), (218323), (218415), (226395), (234329), (247537)])
    ∃ (w_4_0 : mxxMat80),
      let w_5_0 : mxxMat80 := MxxRuntime.matrixMulScalarLeft w_1_0 w_4_0
      ∃ (w_7_0 : mxxMat79),
        ∃ (w_8_0 : mxxMat80),
          let w_9_0 : mxxMat80 := MxxRuntime.matrixMulScalarLeft w_7_0 w_8_0
          let w_10_0 : mxxMat80 := MxxRuntime.matrixAdd w_5_0 w_9_0
          let w_11_0 : mxxMat79 := (MxxRuntime.matrixPolynomial [(1099479121921)] : mxxMat79)
          ∃ (w_12_0 : mxxMat79),
            let w_13_0 : mxxMat79 := MxxRuntime.matrixMulScalarLeft w_11_0 w_12_0
            ∃ (w_14_0 : mxxMat81),
              ∃ (w_15_0 : mxxMat80),
                let w_16_0 : mxxMat80 := MxxRuntime.matrixMulScalarLeft w_13_0 w_15_0
                let w_17_0 : mxxMat80 := MxxRuntime.matrixSub w_10_0 w_16_0
                ∃ (w_18_0 : mxxMat80),
                  let w_19_0 : mxxMat80 := MxxRuntime.matrixAdd w_17_0 w_18_0
                  ∃ (w_20_0 : mxxMat80),
                    let w_21_0 : mxxMat80 := MxxRuntime.matrixMulScalarLeft w_1_0 w_20_0
                    ∃ (w_22_0 : mxxMat80),
                      let w_23_0 : mxxMat80 := MxxRuntime.matrixMulScalarLeft w_7_0 w_22_0
                      let w_24_0 : mxxMat80 := MxxRuntime.matrixAdd w_21_0 w_23_0
                      ∃ (w_25_0 : mxxMat79),
                        let w_26_0 : mxxMat79 := MxxRuntime.matrixMulScalarLeft w_11_0 w_25_0
                        let w_27_0 : mxxMat80 := MxxRuntime.matrixMulScalarLeft w_26_0 w_15_0
                        let w_28_0 : mxxMat80 := MxxRuntime.matrixSub w_24_0 w_27_0
                        ∃ (w_29_0 : mxxMat80),
                          let w_30_0 : mxxMat80 := MxxRuntime.matrixAdd w_28_0 w_29_0
                          parallel_parallel_generatedRoot_7_92.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs

abbrev parallel_parallel_generatedRoot_7_93.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_1 : Nat) (w_0_0 : mxxMat1) (w_2_0 : ByteArray) (w_6_0 : mxxMat1) (w_1_0 : mxxMat79) (w_3_0 : Int) (w_4_0 : mxxMat80) (w_7_0 : mxxMat79) (w_8_0 : mxxMat80) (w_12_0 : mxxMat79) (w_14_0 : mxxMat81) (w_15_0 : mxxMat80) (w_18_0 : mxxMat80) (w_19_0 : mxxMat80) (w_20_0 : mxxMat80) (w_22_0 : mxxMat80) (w_25_0 : mxxMat79) (w_29_0 : mxxMat80) (w_30_0 : mxxMat80) (outputs : mxxMat80 × mxxMat80 × Unit) : Prop :=
  MxxRuntime.modulusReduceRuns w_0_0 w_1_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 32 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (32), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_2_0) w_4_0 ∧
  MxxRuntime.modulusReduceRuns w_6_0 w_7_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (32), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_2_0) w_8_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 32 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (85), (99), (135), (153), (165), (255), (297), (459), (495), (561), (765), (891), (1485), (1683), (2295), (2805), (4455), (5049), (8415), (15147), (25245), (75735)])) w_1_0 w_12_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 1024 40 w_14_0 ∧
  MxxRuntime.centeredRebaseRuns w_14_0 w_15_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [18]) ((4/1)) (26) w_18_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (32), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_2_0) w_20_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (32), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_2_0) w_22_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 32 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (85), (99), (135), (153), (165), (255), (297), (459), (495), (561), (765), (891), (1485), (1683), (2295), (2805), (4455), (5049), (8415), (15147), (25245), (75735)])) w_7_0 w_25_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [29]) ((4/1)) (26) w_29_0 ∧
  outputs = (w_19_0, (w_30_0, ()))

def parallel_parallel_generatedRoot_7_93 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat1 × ByteArray × mxxMat1 × Unit) (outputs : mxxMat80 × mxxMat80 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_2_0 : ByteArray := inputs.2.1
  let w_6_0 : mxxMat1 := inputs.2.2.1
  ∃ (w_1_0 : mxxMat79),
    let w_3_0 : Int := (MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (85), (99), (135), (153), (165), (255), (297), (459), (495), (561), (765), (891), (1485), (1683), (2295), (2805), (4455), (5049), (8415), (15147), (25245), (75735)])
    ∃ (w_4_0 : mxxMat80),
      let w_5_0 : mxxMat80 := MxxRuntime.matrixMulScalarLeft w_1_0 w_4_0
      ∃ (w_7_0 : mxxMat79),
        ∃ (w_8_0 : mxxMat80),
          let w_9_0 : mxxMat80 := MxxRuntime.matrixMulScalarLeft w_7_0 w_8_0
          let w_10_0 : mxxMat80 := MxxRuntime.matrixAdd w_5_0 w_9_0
          let w_11_0 : mxxMat79 := (MxxRuntime.matrixPolynomial [(1099479121921)] : mxxMat79)
          ∃ (w_12_0 : mxxMat79),
            let w_13_0 : mxxMat79 := MxxRuntime.matrixMulScalarLeft w_11_0 w_12_0
            ∃ (w_14_0 : mxxMat81),
              ∃ (w_15_0 : mxxMat80),
                let w_16_0 : mxxMat80 := MxxRuntime.matrixMulScalarLeft w_13_0 w_15_0
                let w_17_0 : mxxMat80 := MxxRuntime.matrixSub w_10_0 w_16_0
                ∃ (w_18_0 : mxxMat80),
                  let w_19_0 : mxxMat80 := MxxRuntime.matrixAdd w_17_0 w_18_0
                  ∃ (w_20_0 : mxxMat80),
                    let w_21_0 : mxxMat80 := MxxRuntime.matrixMulScalarLeft w_1_0 w_20_0
                    ∃ (w_22_0 : mxxMat80),
                      let w_23_0 : mxxMat80 := MxxRuntime.matrixMulScalarLeft w_7_0 w_22_0
                      let w_24_0 : mxxMat80 := MxxRuntime.matrixAdd w_21_0 w_23_0
                      ∃ (w_25_0 : mxxMat79),
                        let w_26_0 : mxxMat79 := MxxRuntime.matrixMulScalarLeft w_11_0 w_25_0
                        let w_27_0 : mxxMat80 := MxxRuntime.matrixMulScalarLeft w_26_0 w_15_0
                        let w_28_0 : mxxMat80 := MxxRuntime.matrixSub w_24_0 w_27_0
                        ∃ (w_29_0 : mxxMat80),
                          let w_30_0 : mxxMat80 := MxxRuntime.matrixAdd w_28_0 w_29_0
                          parallel_parallel_generatedRoot_7_93.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs

abbrev parallel_parallel_generatedRoot_7_94.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_1 : Nat) (w_0_0 : mxxMat1) (w_2_0 : ByteArray) (w_6_0 : mxxMat1) (w_1_0 : mxxMat82) (w_3_0 : Int) (w_4_0 : mxxMat83) (w_7_0 : mxxMat82) (w_8_0 : mxxMat83) (w_12_0 : mxxMat82) (w_14_0 : mxxMat84) (w_15_0 : mxxMat83) (w_18_0 : mxxMat83) (w_19_0 : mxxMat83) (w_20_0 : mxxMat83) (w_22_0 : mxxMat83) (w_25_0 : mxxMat82) (w_29_0 : mxxMat83) (w_30_0 : mxxMat83) (outputs : mxxMat83 × mxxMat83 × Unit) : Prop :=
  MxxRuntime.modulusReduceRuns w_0_0 w_1_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 128 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (33), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_2_0) w_4_0 ∧
  MxxRuntime.modulusReduceRuns w_6_0 w_7_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (33), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_2_0) w_8_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 128 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (65), (85), (99), (129), (135), (153), (165), (195), (255), (297), (325), (387), (459), (495), (561), (585), (645), (765), (891), (975), (1105), (1161), (1485), (1683), (1755), (1935), (2145), (2193), (2295), (2805), (2925), (3315), (3483), (4257), (4455), (5049), (5525), (5805), (6435), (6579), (8385), (8415), (8775), (9945), (10725), (10965), (12771), (14561), (15147), (16575), (17415), (19305), (19737), (21285), (22687), (25155), (25245), (27431), (29835), (32175), (32895), (33911), (36465), (36959), (38313), (41023), (41925), (43039), (43683), (49725), (50407), (57915), (59211), (63855), (66041), (68061), (70487), (72369), (72805), (75465), (75735), (83399), (96525), (98685), (99701), (107063), (109395), (110877), (114939), (115181), (118955), (123069), (125207), (125775), (127033), (129117), (129899), (130681), (130727), (130957), (131003), (131049), (142545), (149175), (165491), (178699), (182325), (188437), (189109), (191565), (198123), (204183), (217107), (218323), (218415), (226395), (234329), (247537)])) w_1_0 w_12_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 1024 36 w_14_0 ∧
  MxxRuntime.centeredRebaseRuns w_14_0 w_15_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [18]) ((4/1)) (26) w_18_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (33), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_2_0) w_20_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (33), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_2_0) w_22_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 128 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (65), (85), (99), (129), (135), (153), (165), (195), (255), (297), (325), (387), (459), (495), (561), (585), (645), (765), (891), (975), (1105), (1161), (1485), (1683), (1755), (1935), (2145), (2193), (2295), (2805), (2925), (3315), (3483), (4257), (4455), (5049), (5525), (5805), (6435), (6579), (8385), (8415), (8775), (9945), (10725), (10965), (12771), (14561), (15147), (16575), (17415), (19305), (19737), (21285), (22687), (25155), (25245), (27431), (29835), (32175), (32895), (33911), (36465), (36959), (38313), (41023), (41925), (43039), (43683), (49725), (50407), (57915), (59211), (63855), (66041), (68061), (70487), (72369), (72805), (75465), (75735), (83399), (96525), (98685), (99701), (107063), (109395), (110877), (114939), (115181), (118955), (123069), (125207), (125775), (127033), (129117), (129899), (130681), (130727), (130957), (131003), (131049), (142545), (149175), (165491), (178699), (182325), (188437), (189109), (191565), (198123), (204183), (217107), (218323), (218415), (226395), (234329), (247537)])) w_7_0 w_25_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [29]) ((4/1)) (26) w_29_0 ∧
  outputs = (w_19_0, (w_30_0, ()))

def parallel_parallel_generatedRoot_7_94 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat1 × ByteArray × mxxMat1 × Unit) (outputs : mxxMat83 × mxxMat83 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_2_0 : ByteArray := inputs.2.1
  let w_6_0 : mxxMat1 := inputs.2.2.1
  ∃ (w_1_0 : mxxMat82),
    let w_3_0 : Int := (MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (65), (85), (99), (129), (135), (153), (165), (195), (255), (297), (325), (387), (459), (495), (561), (585), (645), (765), (891), (975), (1105), (1161), (1485), (1683), (1755), (1935), (2145), (2193), (2295), (2805), (2925), (3315), (3483), (4257), (4455), (5049), (5525), (5805), (6435), (6579), (8385), (8415), (8775), (9945), (10725), (10965), (12771), (14561), (15147), (16575), (17415), (19305), (19737), (21285), (22687), (25155), (25245), (27431), (29835), (32175), (32895), (33911), (36465), (36959), (38313), (41023), (41925), (43039), (43683), (49725), (50407), (57915), (59211), (63855), (66041), (68061), (70487), (72369), (72805), (75465), (75735), (83399), (96525), (98685), (99701), (107063), (109395), (110877), (114939), (115181), (118955), (123069), (125207), (125775), (127033), (129117), (129899), (130681), (130727), (130957), (131003), (131049), (142545), (149175), (165491), (178699), (182325), (188437), (189109), (191565), (198123), (204183), (217107), (218323), (218415), (226395), (234329), (247537)])
    ∃ (w_4_0 : mxxMat83),
      let w_5_0 : mxxMat83 := MxxRuntime.matrixMulScalarLeft w_1_0 w_4_0
      ∃ (w_7_0 : mxxMat82),
        ∃ (w_8_0 : mxxMat83),
          let w_9_0 : mxxMat83 := MxxRuntime.matrixMulScalarLeft w_7_0 w_8_0
          let w_10_0 : mxxMat83 := MxxRuntime.matrixAdd w_5_0 w_9_0
          let w_11_0 : mxxMat82 := (MxxRuntime.matrixPolynomial [(1099480956929)] : mxxMat82)
          ∃ (w_12_0 : mxxMat82),
            let w_13_0 : mxxMat82 := MxxRuntime.matrixMulScalarLeft w_11_0 w_12_0
            ∃ (w_14_0 : mxxMat84),
              ∃ (w_15_0 : mxxMat83),
                let w_16_0 : mxxMat83 := MxxRuntime.matrixMulScalarLeft w_13_0 w_15_0
                let w_17_0 : mxxMat83 := MxxRuntime.matrixSub w_10_0 w_16_0
                ∃ (w_18_0 : mxxMat83),
                  let w_19_0 : mxxMat83 := MxxRuntime.matrixAdd w_17_0 w_18_0
                  ∃ (w_20_0 : mxxMat83),
                    let w_21_0 : mxxMat83 := MxxRuntime.matrixMulScalarLeft w_1_0 w_20_0
                    ∃ (w_22_0 : mxxMat83),
                      let w_23_0 : mxxMat83 := MxxRuntime.matrixMulScalarLeft w_7_0 w_22_0
                      let w_24_0 : mxxMat83 := MxxRuntime.matrixAdd w_21_0 w_23_0
                      ∃ (w_25_0 : mxxMat82),
                        let w_26_0 : mxxMat82 := MxxRuntime.matrixMulScalarLeft w_11_0 w_25_0
                        let w_27_0 : mxxMat83 := MxxRuntime.matrixMulScalarLeft w_26_0 w_15_0
                        let w_28_0 : mxxMat83 := MxxRuntime.matrixSub w_24_0 w_27_0
                        ∃ (w_29_0 : mxxMat83),
                          let w_30_0 : mxxMat83 := MxxRuntime.matrixAdd w_28_0 w_29_0
                          parallel_parallel_generatedRoot_7_94.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs

abbrev parallel_parallel_generatedRoot_7_95.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_1 : Nat) (w_0_0 : mxxMat1) (w_2_0 : ByteArray) (w_6_0 : mxxMat1) (w_1_0 : mxxMat82) (w_3_0 : Int) (w_4_0 : mxxMat83) (w_7_0 : mxxMat82) (w_8_0 : mxxMat83) (w_12_0 : mxxMat82) (w_14_0 : mxxMat84) (w_15_0 : mxxMat83) (w_18_0 : mxxMat83) (w_19_0 : mxxMat83) (w_20_0 : mxxMat83) (w_22_0 : mxxMat83) (w_25_0 : mxxMat82) (w_29_0 : mxxMat83) (w_30_0 : mxxMat83) (outputs : mxxMat83 × mxxMat83 × Unit) : Prop :=
  MxxRuntime.modulusReduceRuns w_0_0 w_1_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 32 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (33), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_2_0) w_4_0 ∧
  MxxRuntime.modulusReduceRuns w_6_0 w_7_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (33), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_2_0) w_8_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 32 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (85), (99), (135), (153), (165), (255), (297), (459), (495), (561), (765), (891), (1485), (1683), (2295), (2805), (4455), (5049), (8415), (15147), (25245), (75735)])) w_1_0 w_12_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 1024 36 w_14_0 ∧
  MxxRuntime.centeredRebaseRuns w_14_0 w_15_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [18]) ((4/1)) (26) w_18_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (33), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_2_0) w_20_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (33), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_2_0) w_22_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 32 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (85), (99), (135), (153), (165), (255), (297), (459), (495), (561), (765), (891), (1485), (1683), (2295), (2805), (4455), (5049), (8415), (15147), (25245), (75735)])) w_7_0 w_25_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [29]) ((4/1)) (26) w_29_0 ∧
  outputs = (w_19_0, (w_30_0, ()))

def parallel_parallel_generatedRoot_7_95 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat1 × ByteArray × mxxMat1 × Unit) (outputs : mxxMat83 × mxxMat83 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_2_0 : ByteArray := inputs.2.1
  let w_6_0 : mxxMat1 := inputs.2.2.1
  ∃ (w_1_0 : mxxMat82),
    let w_3_0 : Int := (MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (85), (99), (135), (153), (165), (255), (297), (459), (495), (561), (765), (891), (1485), (1683), (2295), (2805), (4455), (5049), (8415), (15147), (25245), (75735)])
    ∃ (w_4_0 : mxxMat83),
      let w_5_0 : mxxMat83 := MxxRuntime.matrixMulScalarLeft w_1_0 w_4_0
      ∃ (w_7_0 : mxxMat82),
        ∃ (w_8_0 : mxxMat83),
          let w_9_0 : mxxMat83 := MxxRuntime.matrixMulScalarLeft w_7_0 w_8_0
          let w_10_0 : mxxMat83 := MxxRuntime.matrixAdd w_5_0 w_9_0
          let w_11_0 : mxxMat82 := (MxxRuntime.matrixPolynomial [(1099480956929)] : mxxMat82)
          ∃ (w_12_0 : mxxMat82),
            let w_13_0 : mxxMat82 := MxxRuntime.matrixMulScalarLeft w_11_0 w_12_0
            ∃ (w_14_0 : mxxMat84),
              ∃ (w_15_0 : mxxMat83),
                let w_16_0 : mxxMat83 := MxxRuntime.matrixMulScalarLeft w_13_0 w_15_0
                let w_17_0 : mxxMat83 := MxxRuntime.matrixSub w_10_0 w_16_0
                ∃ (w_18_0 : mxxMat83),
                  let w_19_0 : mxxMat83 := MxxRuntime.matrixAdd w_17_0 w_18_0
                  ∃ (w_20_0 : mxxMat83),
                    let w_21_0 : mxxMat83 := MxxRuntime.matrixMulScalarLeft w_1_0 w_20_0
                    ∃ (w_22_0 : mxxMat83),
                      let w_23_0 : mxxMat83 := MxxRuntime.matrixMulScalarLeft w_7_0 w_22_0
                      let w_24_0 : mxxMat83 := MxxRuntime.matrixAdd w_21_0 w_23_0
                      ∃ (w_25_0 : mxxMat82),
                        let w_26_0 : mxxMat82 := MxxRuntime.matrixMulScalarLeft w_11_0 w_25_0
                        let w_27_0 : mxxMat83 := MxxRuntime.matrixMulScalarLeft w_26_0 w_15_0
                        let w_28_0 : mxxMat83 := MxxRuntime.matrixSub w_24_0 w_27_0
                        ∃ (w_29_0 : mxxMat83),
                          let w_30_0 : mxxMat83 := MxxRuntime.matrixAdd w_28_0 w_29_0
                          parallel_parallel_generatedRoot_7_95.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs

abbrev parallel_parallel_generatedRoot_7_96.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_1 : Nat) (w_0_0 : mxxMat1) (w_2_0 : ByteArray) (w_6_0 : mxxMat1) (w_1_0 : mxxMat85) (w_3_0 : Int) (w_4_0 : mxxMat86) (w_7_0 : mxxMat85) (w_8_0 : mxxMat86) (w_12_0 : mxxMat85) (w_14_0 : mxxMat87) (w_15_0 : mxxMat86) (w_18_0 : mxxMat86) (w_19_0 : mxxMat86) (w_20_0 : mxxMat86) (w_22_0 : mxxMat86) (w_25_0 : mxxMat85) (w_29_0 : mxxMat86) (w_30_0 : mxxMat86) (outputs : mxxMat86 × mxxMat86 × Unit) : Prop :=
  MxxRuntime.modulusReduceRuns w_0_0 w_1_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 128 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (34), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_2_0) w_4_0 ∧
  MxxRuntime.modulusReduceRuns w_6_0 w_7_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (34), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_2_0) w_8_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 128 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (65), (85), (99), (129), (135), (153), (165), (195), (255), (297), (325), (387), (459), (495), (561), (585), (645), (765), (891), (975), (1105), (1161), (1485), (1683), (1755), (1935), (2145), (2193), (2295), (2805), (2925), (3315), (3483), (4257), (4455), (5049), (5525), (5805), (6435), (6579), (8385), (8415), (8775), (9945), (10725), (10965), (12771), (14561), (15147), (16575), (17415), (19305), (19737), (21285), (22687), (25155), (25245), (27431), (29835), (32175), (32895), (33911), (36465), (36959), (38313), (41023), (41925), (43039), (43683), (49725), (50407), (57915), (59211), (63855), (66041), (68061), (70487), (72369), (72805), (75465), (75735), (83399), (96525), (98685), (99701), (107063), (109395), (110877), (114939), (115181), (118955), (123069), (125207), (125775), (127033), (129117), (129899), (130681), (130727), (130957), (131003), (131049), (142545), (149175), (165491), (178699), (182325), (188437), (189109), (191565), (198123), (204183), (217107), (218323), (218415), (226395), (234329), (247537)])) w_1_0 w_12_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 1024 32 w_14_0 ∧
  MxxRuntime.centeredRebaseRuns w_14_0 w_15_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [18]) ((4/1)) (26) w_18_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (34), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_2_0) w_20_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (34), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_2_0) w_22_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 128 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (65), (85), (99), (129), (135), (153), (165), (195), (255), (297), (325), (387), (459), (495), (561), (585), (645), (765), (891), (975), (1105), (1161), (1485), (1683), (1755), (1935), (2145), (2193), (2295), (2805), (2925), (3315), (3483), (4257), (4455), (5049), (5525), (5805), (6435), (6579), (8385), (8415), (8775), (9945), (10725), (10965), (12771), (14561), (15147), (16575), (17415), (19305), (19737), (21285), (22687), (25155), (25245), (27431), (29835), (32175), (32895), (33911), (36465), (36959), (38313), (41023), (41925), (43039), (43683), (49725), (50407), (57915), (59211), (63855), (66041), (68061), (70487), (72369), (72805), (75465), (75735), (83399), (96525), (98685), (99701), (107063), (109395), (110877), (114939), (115181), (118955), (123069), (125207), (125775), (127033), (129117), (129899), (130681), (130727), (130957), (131003), (131049), (142545), (149175), (165491), (178699), (182325), (188437), (189109), (191565), (198123), (204183), (217107), (218323), (218415), (226395), (234329), (247537)])) w_7_0 w_25_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [29]) ((4/1)) (26) w_29_0 ∧
  outputs = (w_19_0, (w_30_0, ()))

def parallel_parallel_generatedRoot_7_96 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat1 × ByteArray × mxxMat1 × Unit) (outputs : mxxMat86 × mxxMat86 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_2_0 : ByteArray := inputs.2.1
  let w_6_0 : mxxMat1 := inputs.2.2.1
  ∃ (w_1_0 : mxxMat85),
    let w_3_0 : Int := (MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (65), (85), (99), (129), (135), (153), (165), (195), (255), (297), (325), (387), (459), (495), (561), (585), (645), (765), (891), (975), (1105), (1161), (1485), (1683), (1755), (1935), (2145), (2193), (2295), (2805), (2925), (3315), (3483), (4257), (4455), (5049), (5525), (5805), (6435), (6579), (8385), (8415), (8775), (9945), (10725), (10965), (12771), (14561), (15147), (16575), (17415), (19305), (19737), (21285), (22687), (25155), (25245), (27431), (29835), (32175), (32895), (33911), (36465), (36959), (38313), (41023), (41925), (43039), (43683), (49725), (50407), (57915), (59211), (63855), (66041), (68061), (70487), (72369), (72805), (75465), (75735), (83399), (96525), (98685), (99701), (107063), (109395), (110877), (114939), (115181), (118955), (123069), (125207), (125775), (127033), (129117), (129899), (130681), (130727), (130957), (131003), (131049), (142545), (149175), (165491), (178699), (182325), (188437), (189109), (191565), (198123), (204183), (217107), (218323), (218415), (226395), (234329), (247537)])
    ∃ (w_4_0 : mxxMat86),
      let w_5_0 : mxxMat86 := MxxRuntime.matrixMulScalarLeft w_1_0 w_4_0
      ∃ (w_7_0 : mxxMat85),
        ∃ (w_8_0 : mxxMat86),
          let w_9_0 : mxxMat86 := MxxRuntime.matrixMulScalarLeft w_7_0 w_8_0
          let w_10_0 : mxxMat86 := MxxRuntime.matrixAdd w_5_0 w_9_0
          let w_11_0 : mxxMat85 := (MxxRuntime.matrixPolynomial [(1099484889089)] : mxxMat85)
          ∃ (w_12_0 : mxxMat85),
            let w_13_0 : mxxMat85 := MxxRuntime.matrixMulScalarLeft w_11_0 w_12_0
            ∃ (w_14_0 : mxxMat87),
              ∃ (w_15_0 : mxxMat86),
                let w_16_0 : mxxMat86 := MxxRuntime.matrixMulScalarLeft w_13_0 w_15_0
                let w_17_0 : mxxMat86 := MxxRuntime.matrixSub w_10_0 w_16_0
                ∃ (w_18_0 : mxxMat86),
                  let w_19_0 : mxxMat86 := MxxRuntime.matrixAdd w_17_0 w_18_0
                  ∃ (w_20_0 : mxxMat86),
                    let w_21_0 : mxxMat86 := MxxRuntime.matrixMulScalarLeft w_1_0 w_20_0
                    ∃ (w_22_0 : mxxMat86),
                      let w_23_0 : mxxMat86 := MxxRuntime.matrixMulScalarLeft w_7_0 w_22_0
                      let w_24_0 : mxxMat86 := MxxRuntime.matrixAdd w_21_0 w_23_0
                      ∃ (w_25_0 : mxxMat85),
                        let w_26_0 : mxxMat85 := MxxRuntime.matrixMulScalarLeft w_11_0 w_25_0
                        let w_27_0 : mxxMat86 := MxxRuntime.matrixMulScalarLeft w_26_0 w_15_0
                        let w_28_0 : mxxMat86 := MxxRuntime.matrixSub w_24_0 w_27_0
                        ∃ (w_29_0 : mxxMat86),
                          let w_30_0 : mxxMat86 := MxxRuntime.matrixAdd w_28_0 w_29_0
                          parallel_parallel_generatedRoot_7_96.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs

abbrev parallel_parallel_generatedRoot_7_97.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_1 : Nat) (w_0_0 : mxxMat1) (w_2_0 : ByteArray) (w_6_0 : mxxMat1) (w_1_0 : mxxMat85) (w_3_0 : Int) (w_4_0 : mxxMat86) (w_7_0 : mxxMat85) (w_8_0 : mxxMat86) (w_12_0 : mxxMat85) (w_14_0 : mxxMat87) (w_15_0 : mxxMat86) (w_18_0 : mxxMat86) (w_19_0 : mxxMat86) (w_20_0 : mxxMat86) (w_22_0 : mxxMat86) (w_25_0 : mxxMat85) (w_29_0 : mxxMat86) (w_30_0 : mxxMat86) (outputs : mxxMat86 × mxxMat86 × Unit) : Prop :=
  MxxRuntime.modulusReduceRuns w_0_0 w_1_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 32 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (34), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_2_0) w_4_0 ∧
  MxxRuntime.modulusReduceRuns w_6_0 w_7_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (34), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_2_0) w_8_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 32 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (85), (99), (135), (153), (165), (255), (297), (459), (495), (561), (765), (891), (1485), (1683), (2295), (2805), (4455), (5049), (8415), (15147), (25245), (75735)])) w_1_0 w_12_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 1024 32 w_14_0 ∧
  MxxRuntime.centeredRebaseRuns w_14_0 w_15_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [18]) ((4/1)) (26) w_18_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (34), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_2_0) w_20_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (34), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_2_0) w_22_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 32 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (85), (99), (135), (153), (165), (255), (297), (459), (495), (561), (765), (891), (1485), (1683), (2295), (2805), (4455), (5049), (8415), (15147), (25245), (75735)])) w_7_0 w_25_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [29]) ((4/1)) (26) w_29_0 ∧
  outputs = (w_19_0, (w_30_0, ()))

def parallel_parallel_generatedRoot_7_97 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat1 × ByteArray × mxxMat1 × Unit) (outputs : mxxMat86 × mxxMat86 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_2_0 : ByteArray := inputs.2.1
  let w_6_0 : mxxMat1 := inputs.2.2.1
  ∃ (w_1_0 : mxxMat85),
    let w_3_0 : Int := (MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (85), (99), (135), (153), (165), (255), (297), (459), (495), (561), (765), (891), (1485), (1683), (2295), (2805), (4455), (5049), (8415), (15147), (25245), (75735)])
    ∃ (w_4_0 : mxxMat86),
      let w_5_0 : mxxMat86 := MxxRuntime.matrixMulScalarLeft w_1_0 w_4_0
      ∃ (w_7_0 : mxxMat85),
        ∃ (w_8_0 : mxxMat86),
          let w_9_0 : mxxMat86 := MxxRuntime.matrixMulScalarLeft w_7_0 w_8_0
          let w_10_0 : mxxMat86 := MxxRuntime.matrixAdd w_5_0 w_9_0
          let w_11_0 : mxxMat85 := (MxxRuntime.matrixPolynomial [(1099484889089)] : mxxMat85)
          ∃ (w_12_0 : mxxMat85),
            let w_13_0 : mxxMat85 := MxxRuntime.matrixMulScalarLeft w_11_0 w_12_0
            ∃ (w_14_0 : mxxMat87),
              ∃ (w_15_0 : mxxMat86),
                let w_16_0 : mxxMat86 := MxxRuntime.matrixMulScalarLeft w_13_0 w_15_0
                let w_17_0 : mxxMat86 := MxxRuntime.matrixSub w_10_0 w_16_0
                ∃ (w_18_0 : mxxMat86),
                  let w_19_0 : mxxMat86 := MxxRuntime.matrixAdd w_17_0 w_18_0
                  ∃ (w_20_0 : mxxMat86),
                    let w_21_0 : mxxMat86 := MxxRuntime.matrixMulScalarLeft w_1_0 w_20_0
                    ∃ (w_22_0 : mxxMat86),
                      let w_23_0 : mxxMat86 := MxxRuntime.matrixMulScalarLeft w_7_0 w_22_0
                      let w_24_0 : mxxMat86 := MxxRuntime.matrixAdd w_21_0 w_23_0
                      ∃ (w_25_0 : mxxMat85),
                        let w_26_0 : mxxMat85 := MxxRuntime.matrixMulScalarLeft w_11_0 w_25_0
                        let w_27_0 : mxxMat86 := MxxRuntime.matrixMulScalarLeft w_26_0 w_15_0
                        let w_28_0 : mxxMat86 := MxxRuntime.matrixSub w_24_0 w_27_0
                        ∃ (w_29_0 : mxxMat86),
                          let w_30_0 : mxxMat86 := MxxRuntime.matrixAdd w_28_0 w_29_0
                          parallel_parallel_generatedRoot_7_97.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs

abbrev parallel_parallel_generatedRoot_7_98.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (i_1 : Nat) (w_0_0 : mxxMat1) (w_2_0 : ByteArray) (w_6_0 : mxxMat1) (w_1_0 : mxxMat88) (w_3_0 : Int) (w_4_0 : mxxMat89) (w_7_0 : mxxMat88) (w_8_0 : mxxMat89) (w_12_0 : mxxMat88) (w_14_0 : Mxx.Primitives.ExactMatrix ((1099510054913) * (1099507695617) * (1099504549889) * (1099502714881) * (1099500617729) * (1099499569153) * ((MxxIR.selectInt ((Int.ofNat i_0)) [(1099498258433), (1099498258433), (1099498258433), (1099498258433), (1099498258433), (1099498258433), (1099498258433), (1099489607681)]))).toNat 131072 1 28) (w_15_0 : mxxMat89) (w_18_0 : mxxMat89) (w_19_0 : mxxMat89) (w_20_0 : mxxMat89) (w_22_0 : mxxMat89) (w_25_0 : mxxMat88) (w_29_0 : mxxMat89) (w_30_0 : mxxMat89) (outputs : mxxMat89 × mxxMat89 × Unit) : Prop :=
  MxxRuntime.modulusReduceRuns w_0_0 w_1_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 128 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (35), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_2_0) w_4_0 ∧
  MxxRuntime.modulusReduceRuns w_6_0 w_7_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (35), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_2_0) w_8_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 8 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 128 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (65), (85), (99), (129), (135), (153), (165), (195), (255), (297), (325), (387), (459), (495), (561), (585), (645), (765), (891), (975), (1105), (1161), (1485), (1683), (1755), (1935), (2145), (2193), (2295), (2805), (2925), (3315), (3483), (4257), (4455), (5049), (5525), (5805), (6435), (6579), (8385), (8415), (8775), (9945), (10725), (10965), (12771), (14561), (15147), (16575), (17415), (19305), (19737), (21285), (22687), (25155), (25245), (27431), (29835), (32175), (32895), (33911), (36465), (36959), (38313), (41023), (41925), (43039), (43683), (49725), (50407), (57915), (59211), (63855), (66041), (68061), (70487), (72369), (72805), (75465), (75735), (83399), (96525), (98685), (99701), (107063), (109395), (110877), (114939), (115181), (118955), (123069), (125207), (125775), (127033), (129117), (129899), (130681), (130727), (130957), (131003), (131049), (142545), (149175), (165491), (178699), (182325), (188437), (189109), (191565), (198123), (204183), (217107), (218323), (218415), (226395), (234329), (247537)])) w_1_0 w_12_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 1024 28 w_14_0 ∧
  MxxRuntime.centeredRebaseRuns w_14_0 w_15_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [18]) ((4/1)) (26) w_18_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (35), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_2_0) w_20_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (35), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_2_0) w_22_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 128 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (65), (85), (99), (129), (135), (153), (165), (195), (255), (297), (325), (387), (459), (495), (561), (585), (645), (765), (891), (975), (1105), (1161), (1485), (1683), (1755), (1935), (2145), (2193), (2295), (2805), (2925), (3315), (3483), (4257), (4455), (5049), (5525), (5805), (6435), (6579), (8385), (8415), (8775), (9945), (10725), (10965), (12771), (14561), (15147), (16575), (17415), (19305), (19737), (21285), (22687), (25155), (25245), (27431), (29835), (32175), (32895), (33911), (36465), (36959), (38313), (41023), (41925), (43039), (43683), (49725), (50407), (57915), (59211), (63855), (66041), (68061), (70487), (72369), (72805), (75465), (75735), (83399), (96525), (98685), (99701), (107063), (109395), (110877), (114939), (115181), (118955), (123069), (125207), (125775), (127033), (129117), (129899), (130681), (130727), (130957), (131003), (131049), (142545), (149175), (165491), (178699), (182325), (188437), (189109), (191565), (198123), (204183), (217107), (218323), (218415), (226395), (234329), (247537)])) w_7_0 w_25_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [29]) ((4/1)) (26) w_29_0 ∧
  outputs = (w_19_0, (w_30_0, ()))

def parallel_parallel_generatedRoot_7_98 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (i_1 : Nat) (inputs : mxxMat1 × ByteArray × mxxMat1 × Unit) (outputs : mxxMat89 × mxxMat89 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_2_0 : ByteArray := inputs.2.1
  let w_6_0 : mxxMat1 := inputs.2.2.1
  ∃ (w_1_0 : mxxMat88),
    let w_3_0 : Int := (MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (65), (85), (99), (129), (135), (153), (165), (195), (255), (297), (325), (387), (459), (495), (561), (585), (645), (765), (891), (975), (1105), (1161), (1485), (1683), (1755), (1935), (2145), (2193), (2295), (2805), (2925), (3315), (3483), (4257), (4455), (5049), (5525), (5805), (6435), (6579), (8385), (8415), (8775), (9945), (10725), (10965), (12771), (14561), (15147), (16575), (17415), (19305), (19737), (21285), (22687), (25155), (25245), (27431), (29835), (32175), (32895), (33911), (36465), (36959), (38313), (41023), (41925), (43039), (43683), (49725), (50407), (57915), (59211), (63855), (66041), (68061), (70487), (72369), (72805), (75465), (75735), (83399), (96525), (98685), (99701), (107063), (109395), (110877), (114939), (115181), (118955), (123069), (125207), (125775), (127033), (129117), (129899), (130681), (130727), (130957), (131003), (131049), (142545), (149175), (165491), (178699), (182325), (188437), (189109), (191565), (198123), (204183), (217107), (218323), (218415), (226395), (234329), (247537)])
    ∃ (w_4_0 : mxxMat89),
      let w_5_0 : mxxMat89 := MxxRuntime.matrixMulScalarLeft w_1_0 w_4_0
      ∃ (w_7_0 : mxxMat88),
        ∃ (w_8_0 : mxxMat89),
          let w_9_0 : mxxMat89 := MxxRuntime.matrixMulScalarLeft w_7_0 w_8_0
          let w_10_0 : mxxMat89 := MxxRuntime.matrixAdd w_5_0 w_9_0
          let w_11_0 : mxxMat88 := (MxxRuntime.matrixPolynomial [((MxxIR.selectInt ((Int.ofNat i_0)) [(1099489607681), (1099489607681), (1099489607681), (1099489607681), (1099489607681), (1099489607681), (1099489607681), (1099498258433)]))] : mxxMat88)
          ∃ (w_12_0 : mxxMat88),
            let w_13_0 : mxxMat88 := MxxRuntime.matrixMulScalarLeft w_11_0 w_12_0
            ∃ (w_14_0 : Mxx.Primitives.ExactMatrix ((1099510054913) * (1099507695617) * (1099504549889) * (1099502714881) * (1099500617729) * (1099499569153) * ((MxxIR.selectInt ((Int.ofNat i_0)) [(1099498258433), (1099498258433), (1099498258433), (1099498258433), (1099498258433), (1099498258433), (1099498258433), (1099489607681)]))).toNat 131072 1 28),
              ∃ (w_15_0 : mxxMat89),
                let w_16_0 : mxxMat89 := MxxRuntime.matrixMulScalarLeft w_13_0 w_15_0
                let w_17_0 : mxxMat89 := MxxRuntime.matrixSub w_10_0 w_16_0
                ∃ (w_18_0 : mxxMat89),
                  let w_19_0 : mxxMat89 := MxxRuntime.matrixAdd w_17_0 w_18_0
                  ∃ (w_20_0 : mxxMat89),
                    let w_21_0 : mxxMat89 := MxxRuntime.matrixMulScalarLeft w_1_0 w_20_0
                    ∃ (w_22_0 : mxxMat89),
                      let w_23_0 : mxxMat89 := MxxRuntime.matrixMulScalarLeft w_7_0 w_22_0
                      let w_24_0 : mxxMat89 := MxxRuntime.matrixAdd w_21_0 w_23_0
                      ∃ (w_25_0 : mxxMat88),
                        let w_26_0 : mxxMat88 := MxxRuntime.matrixMulScalarLeft w_11_0 w_25_0
                        let w_27_0 : mxxMat89 := MxxRuntime.matrixMulScalarLeft w_26_0 w_15_0
                        let w_28_0 : mxxMat89 := MxxRuntime.matrixSub w_24_0 w_27_0
                        ∃ (w_29_0 : mxxMat89),
                          let w_30_0 : mxxMat89 := MxxRuntime.matrixAdd w_28_0 w_29_0
                          parallel_parallel_generatedRoot_7_98.constraints_0 backend hashModel tape path i_0 i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs

abbrev parallel_parallel_generatedRoot_7_99.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (i_1 : Nat) (w_0_0 : mxxMat1) (w_2_0 : ByteArray) (w_6_0 : mxxMat1) (w_1_0 : mxxMat88) (w_3_0 : Int) (w_4_0 : mxxMat89) (w_7_0 : mxxMat88) (w_8_0 : mxxMat89) (w_12_0 : mxxMat88) (w_14_0 : Mxx.Primitives.ExactMatrix ((1099510054913) * (1099507695617) * (1099504549889) * (1099502714881) * (1099500617729) * (1099499569153) * ((MxxIR.selectInt ((Int.ofNat i_0)) [(1099498258433), (1099498258433), (1099498258433), (1099498258433), (1099498258433), (1099498258433), (1099498258433), (1099489607681)]))).toNat 131072 1 28) (w_15_0 : mxxMat89) (w_18_0 : mxxMat89) (w_19_0 : mxxMat89) (w_20_0 : mxxMat89) (w_22_0 : mxxMat89) (w_25_0 : mxxMat88) (w_29_0 : mxxMat89) (w_30_0 : mxxMat89) (outputs : mxxMat89 × mxxMat89 × Unit) : Prop :=
  MxxRuntime.modulusReduceRuns w_0_0 w_1_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 32 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (35), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_2_0) w_4_0 ∧
  MxxRuntime.modulusReduceRuns w_6_0 w_7_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (35), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_2_0) w_8_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 8 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 32 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (85), (99), (135), (153), (165), (255), (297), (459), (495), (561), (765), (891), (1485), (1683), (2295), (2805), (4455), (5049), (8415), (15147), (25245), (75735)])) w_1_0 w_12_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 1024 28 w_14_0 ∧
  MxxRuntime.centeredRebaseRuns w_14_0 w_15_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [18]) ((4/1)) (26) w_18_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (35), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_2_0) w_20_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (35), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_2_0) w_22_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 32 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (85), (99), (135), (153), (165), (255), (297), (459), (495), (561), (765), (891), (1485), (1683), (2295), (2805), (4455), (5049), (8415), (15147), (25245), (75735)])) w_7_0 w_25_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [29]) ((4/1)) (26) w_29_0 ∧
  outputs = (w_19_0, (w_30_0, ()))

def parallel_parallel_generatedRoot_7_99 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (i_1 : Nat) (inputs : mxxMat1 × ByteArray × mxxMat1 × Unit) (outputs : mxxMat89 × mxxMat89 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_2_0 : ByteArray := inputs.2.1
  let w_6_0 : mxxMat1 := inputs.2.2.1
  ∃ (w_1_0 : mxxMat88),
    let w_3_0 : Int := (MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (85), (99), (135), (153), (165), (255), (297), (459), (495), (561), (765), (891), (1485), (1683), (2295), (2805), (4455), (5049), (8415), (15147), (25245), (75735)])
    ∃ (w_4_0 : mxxMat89),
      let w_5_0 : mxxMat89 := MxxRuntime.matrixMulScalarLeft w_1_0 w_4_0
      ∃ (w_7_0 : mxxMat88),
        ∃ (w_8_0 : mxxMat89),
          let w_9_0 : mxxMat89 := MxxRuntime.matrixMulScalarLeft w_7_0 w_8_0
          let w_10_0 : mxxMat89 := MxxRuntime.matrixAdd w_5_0 w_9_0
          let w_11_0 : mxxMat88 := (MxxRuntime.matrixPolynomial [((MxxIR.selectInt ((Int.ofNat i_0)) [(1099489607681), (1099489607681), (1099489607681), (1099489607681), (1099489607681), (1099489607681), (1099489607681), (1099498258433)]))] : mxxMat88)
          ∃ (w_12_0 : mxxMat88),
            let w_13_0 : mxxMat88 := MxxRuntime.matrixMulScalarLeft w_11_0 w_12_0
            ∃ (w_14_0 : Mxx.Primitives.ExactMatrix ((1099510054913) * (1099507695617) * (1099504549889) * (1099502714881) * (1099500617729) * (1099499569153) * ((MxxIR.selectInt ((Int.ofNat i_0)) [(1099498258433), (1099498258433), (1099498258433), (1099498258433), (1099498258433), (1099498258433), (1099498258433), (1099489607681)]))).toNat 131072 1 28),
              ∃ (w_15_0 : mxxMat89),
                let w_16_0 : mxxMat89 := MxxRuntime.matrixMulScalarLeft w_13_0 w_15_0
                let w_17_0 : mxxMat89 := MxxRuntime.matrixSub w_10_0 w_16_0
                ∃ (w_18_0 : mxxMat89),
                  let w_19_0 : mxxMat89 := MxxRuntime.matrixAdd w_17_0 w_18_0
                  ∃ (w_20_0 : mxxMat89),
                    let w_21_0 : mxxMat89 := MxxRuntime.matrixMulScalarLeft w_1_0 w_20_0
                    ∃ (w_22_0 : mxxMat89),
                      let w_23_0 : mxxMat89 := MxxRuntime.matrixMulScalarLeft w_7_0 w_22_0
                      let w_24_0 : mxxMat89 := MxxRuntime.matrixAdd w_21_0 w_23_0
                      ∃ (w_25_0 : mxxMat88),
                        let w_26_0 : mxxMat88 := MxxRuntime.matrixMulScalarLeft w_11_0 w_25_0
                        let w_27_0 : mxxMat89 := MxxRuntime.matrixMulScalarLeft w_26_0 w_15_0
                        let w_28_0 : mxxMat89 := MxxRuntime.matrixSub w_24_0 w_27_0
                        ∃ (w_29_0 : mxxMat89),
                          let w_30_0 : mxxMat89 := MxxRuntime.matrixAdd w_28_0 w_29_0
                          parallel_parallel_generatedRoot_7_99.constraints_0 backend hashModel tape path i_0 i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs

abbrev parallel_parallel_generatedRoot_7_100.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_1 : Nat) (w_0_0 : mxxMat1) (w_2_0 : ByteArray) (w_6_0 : mxxMat1) (w_1_0 : mxxMat90) (w_3_0 : Int) (w_4_0 : mxxMat91) (w_7_0 : mxxMat90) (w_8_0 : mxxMat91) (w_12_0 : mxxMat90) (w_14_0 : mxxMat92) (w_15_0 : mxxMat91) (w_18_0 : mxxMat91) (w_19_0 : mxxMat91) (w_20_0 : mxxMat91) (w_22_0 : mxxMat91) (w_25_0 : mxxMat90) (w_29_0 : mxxMat91) (w_30_0 : mxxMat91) (outputs : mxxMat91 × mxxMat91 × Unit) : Prop :=
  MxxRuntime.modulusReduceRuns w_0_0 w_1_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 128 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (4), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_2_0) w_4_0 ∧
  MxxRuntime.modulusReduceRuns w_6_0 w_7_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (4), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_2_0) w_8_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 128 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (65), (85), (99), (129), (135), (153), (165), (195), (255), (297), (325), (387), (459), (495), (561), (585), (645), (765), (891), (975), (1105), (1161), (1485), (1683), (1755), (1935), (2145), (2193), (2295), (2805), (2925), (3315), (3483), (4257), (4455), (5049), (5525), (5805), (6435), (6579), (8385), (8415), (8775), (9945), (10725), (10965), (12771), (14561), (15147), (16575), (17415), (19305), (19737), (21285), (22687), (25155), (25245), (27431), (29835), (32175), (32895), (33911), (36465), (36959), (38313), (41023), (41925), (43039), (43683), (49725), (50407), (57915), (59211), (63855), (66041), (68061), (70487), (72369), (72805), (75465), (75735), (83399), (96525), (98685), (99701), (107063), (109395), (110877), (114939), (115181), (118955), (123069), (125207), (125775), (127033), (129117), (129899), (130681), (130727), (130957), (131003), (131049), (142545), (149175), (165491), (178699), (182325), (188437), (189109), (191565), (198123), (204183), (217107), (218323), (218415), (226395), (234329), (247537)])) w_1_0 w_12_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 1024 152 w_14_0 ∧
  MxxRuntime.centeredRebaseRuns w_14_0 w_15_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [18]) ((4/1)) (26) w_18_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (4), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_2_0) w_20_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (4), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_2_0) w_22_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 128 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (65), (85), (99), (129), (135), (153), (165), (195), (255), (297), (325), (387), (459), (495), (561), (585), (645), (765), (891), (975), (1105), (1161), (1485), (1683), (1755), (1935), (2145), (2193), (2295), (2805), (2925), (3315), (3483), (4257), (4455), (5049), (5525), (5805), (6435), (6579), (8385), (8415), (8775), (9945), (10725), (10965), (12771), (14561), (15147), (16575), (17415), (19305), (19737), (21285), (22687), (25155), (25245), (27431), (29835), (32175), (32895), (33911), (36465), (36959), (38313), (41023), (41925), (43039), (43683), (49725), (50407), (57915), (59211), (63855), (66041), (68061), (70487), (72369), (72805), (75465), (75735), (83399), (96525), (98685), (99701), (107063), (109395), (110877), (114939), (115181), (118955), (123069), (125207), (125775), (127033), (129117), (129899), (130681), (130727), (130957), (131003), (131049), (142545), (149175), (165491), (178699), (182325), (188437), (189109), (191565), (198123), (204183), (217107), (218323), (218415), (226395), (234329), (247537)])) w_7_0 w_25_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [29]) ((4/1)) (26) w_29_0 ∧
  outputs = (w_19_0, (w_30_0, ()))

def parallel_parallel_generatedRoot_7_100 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat1 × ByteArray × mxxMat1 × Unit) (outputs : mxxMat91 × mxxMat91 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_2_0 : ByteArray := inputs.2.1
  let w_6_0 : mxxMat1 := inputs.2.2.1
  ∃ (w_1_0 : mxxMat90),
    let w_3_0 : Int := (MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (65), (85), (99), (129), (135), (153), (165), (195), (255), (297), (325), (387), (459), (495), (561), (585), (645), (765), (891), (975), (1105), (1161), (1485), (1683), (1755), (1935), (2145), (2193), (2295), (2805), (2925), (3315), (3483), (4257), (4455), (5049), (5525), (5805), (6435), (6579), (8385), (8415), (8775), (9945), (10725), (10965), (12771), (14561), (15147), (16575), (17415), (19305), (19737), (21285), (22687), (25155), (25245), (27431), (29835), (32175), (32895), (33911), (36465), (36959), (38313), (41023), (41925), (43039), (43683), (49725), (50407), (57915), (59211), (63855), (66041), (68061), (70487), (72369), (72805), (75465), (75735), (83399), (96525), (98685), (99701), (107063), (109395), (110877), (114939), (115181), (118955), (123069), (125207), (125775), (127033), (129117), (129899), (130681), (130727), (130957), (131003), (131049), (142545), (149175), (165491), (178699), (182325), (188437), (189109), (191565), (198123), (204183), (217107), (218323), (218415), (226395), (234329), (247537)])
    ∃ (w_4_0 : mxxMat91),
      let w_5_0 : mxxMat91 := MxxRuntime.matrixMulScalarLeft w_1_0 w_4_0
      ∃ (w_7_0 : mxxMat90),
        ∃ (w_8_0 : mxxMat91),
          let w_9_0 : mxxMat91 := MxxRuntime.matrixMulScalarLeft w_7_0 w_8_0
          let w_10_0 : mxxMat91 := MxxRuntime.matrixAdd w_5_0 w_9_0
          let w_11_0 : mxxMat90 := (MxxRuntime.matrixPolynomial [(1099388682241)] : mxxMat90)
          ∃ (w_12_0 : mxxMat90),
            let w_13_0 : mxxMat90 := MxxRuntime.matrixMulScalarLeft w_11_0 w_12_0
            ∃ (w_14_0 : mxxMat92),
              ∃ (w_15_0 : mxxMat91),
                let w_16_0 : mxxMat91 := MxxRuntime.matrixMulScalarLeft w_13_0 w_15_0
                let w_17_0 : mxxMat91 := MxxRuntime.matrixSub w_10_0 w_16_0
                ∃ (w_18_0 : mxxMat91),
                  let w_19_0 : mxxMat91 := MxxRuntime.matrixAdd w_17_0 w_18_0
                  ∃ (w_20_0 : mxxMat91),
                    let w_21_0 : mxxMat91 := MxxRuntime.matrixMulScalarLeft w_1_0 w_20_0
                    ∃ (w_22_0 : mxxMat91),
                      let w_23_0 : mxxMat91 := MxxRuntime.matrixMulScalarLeft w_7_0 w_22_0
                      let w_24_0 : mxxMat91 := MxxRuntime.matrixAdd w_21_0 w_23_0
                      ∃ (w_25_0 : mxxMat90),
                        let w_26_0 : mxxMat90 := MxxRuntime.matrixMulScalarLeft w_11_0 w_25_0
                        let w_27_0 : mxxMat91 := MxxRuntime.matrixMulScalarLeft w_26_0 w_15_0
                        let w_28_0 : mxxMat91 := MxxRuntime.matrixSub w_24_0 w_27_0
                        ∃ (w_29_0 : mxxMat91),
                          let w_30_0 : mxxMat91 := MxxRuntime.matrixAdd w_28_0 w_29_0
                          parallel_parallel_generatedRoot_7_100.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs

abbrev parallel_parallel_generatedRoot_7_101.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_1 : Nat) (w_0_0 : mxxMat1) (w_2_0 : ByteArray) (w_6_0 : mxxMat1) (w_1_0 : mxxMat90) (w_3_0 : Int) (w_4_0 : mxxMat91) (w_7_0 : mxxMat90) (w_8_0 : mxxMat91) (w_12_0 : mxxMat90) (w_14_0 : mxxMat92) (w_15_0 : mxxMat91) (w_18_0 : mxxMat91) (w_19_0 : mxxMat91) (w_20_0 : mxxMat91) (w_22_0 : mxxMat91) (w_25_0 : mxxMat90) (w_29_0 : mxxMat91) (w_30_0 : mxxMat91) (outputs : mxxMat91 × mxxMat91 × Unit) : Prop :=
  MxxRuntime.modulusReduceRuns w_0_0 w_1_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 32 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (4), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_2_0) w_4_0 ∧
  MxxRuntime.modulusReduceRuns w_6_0 w_7_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (4), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_2_0) w_8_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 32 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (85), (99), (135), (153), (165), (255), (297), (459), (495), (561), (765), (891), (1485), (1683), (2295), (2805), (4455), (5049), (8415), (15147), (25245), (75735)])) w_1_0 w_12_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 1024 152 w_14_0 ∧
  MxxRuntime.centeredRebaseRuns w_14_0 w_15_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [18]) ((4/1)) (26) w_18_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (4), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_2_0) w_20_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (4), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_2_0) w_22_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 32 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (85), (99), (135), (153), (165), (255), (297), (459), (495), (561), (765), (891), (1485), (1683), (2295), (2805), (4455), (5049), (8415), (15147), (25245), (75735)])) w_7_0 w_25_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [29]) ((4/1)) (26) w_29_0 ∧
  outputs = (w_19_0, (w_30_0, ()))

def parallel_parallel_generatedRoot_7_101 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat1 × ByteArray × mxxMat1 × Unit) (outputs : mxxMat91 × mxxMat91 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_2_0 : ByteArray := inputs.2.1
  let w_6_0 : mxxMat1 := inputs.2.2.1
  ∃ (w_1_0 : mxxMat90),
    let w_3_0 : Int := (MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (85), (99), (135), (153), (165), (255), (297), (459), (495), (561), (765), (891), (1485), (1683), (2295), (2805), (4455), (5049), (8415), (15147), (25245), (75735)])
    ∃ (w_4_0 : mxxMat91),
      let w_5_0 : mxxMat91 := MxxRuntime.matrixMulScalarLeft w_1_0 w_4_0
      ∃ (w_7_0 : mxxMat90),
        ∃ (w_8_0 : mxxMat91),
          let w_9_0 : mxxMat91 := MxxRuntime.matrixMulScalarLeft w_7_0 w_8_0
          let w_10_0 : mxxMat91 := MxxRuntime.matrixAdd w_5_0 w_9_0
          let w_11_0 : mxxMat90 := (MxxRuntime.matrixPolynomial [(1099388682241)] : mxxMat90)
          ∃ (w_12_0 : mxxMat90),
            let w_13_0 : mxxMat90 := MxxRuntime.matrixMulScalarLeft w_11_0 w_12_0
            ∃ (w_14_0 : mxxMat92),
              ∃ (w_15_0 : mxxMat91),
                let w_16_0 : mxxMat91 := MxxRuntime.matrixMulScalarLeft w_13_0 w_15_0
                let w_17_0 : mxxMat91 := MxxRuntime.matrixSub w_10_0 w_16_0
                ∃ (w_18_0 : mxxMat91),
                  let w_19_0 : mxxMat91 := MxxRuntime.matrixAdd w_17_0 w_18_0
                  ∃ (w_20_0 : mxxMat91),
                    let w_21_0 : mxxMat91 := MxxRuntime.matrixMulScalarLeft w_1_0 w_20_0
                    ∃ (w_22_0 : mxxMat91),
                      let w_23_0 : mxxMat91 := MxxRuntime.matrixMulScalarLeft w_7_0 w_22_0
                      let w_24_0 : mxxMat91 := MxxRuntime.matrixAdd w_21_0 w_23_0
                      ∃ (w_25_0 : mxxMat90),
                        let w_26_0 : mxxMat90 := MxxRuntime.matrixMulScalarLeft w_11_0 w_25_0
                        let w_27_0 : mxxMat91 := MxxRuntime.matrixMulScalarLeft w_26_0 w_15_0
                        let w_28_0 : mxxMat91 := MxxRuntime.matrixSub w_24_0 w_27_0
                        ∃ (w_29_0 : mxxMat91),
                          let w_30_0 : mxxMat91 := MxxRuntime.matrixAdd w_28_0 w_29_0
                          parallel_parallel_generatedRoot_7_101.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs

abbrev parallel_parallel_generatedRoot_7_102.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_1 : Nat) (w_0_0 : mxxMat1) (w_2_0 : ByteArray) (w_6_0 : mxxMat1) (w_1_0 : mxxMat93) (w_3_0 : Int) (w_4_0 : mxxMat94) (w_7_0 : mxxMat93) (w_8_0 : mxxMat94) (w_12_0 : mxxMat93) (w_14_0 : mxxMat95) (w_15_0 : mxxMat94) (w_18_0 : mxxMat94) (w_19_0 : mxxMat94) (w_20_0 : mxxMat94) (w_22_0 : mxxMat94) (w_25_0 : mxxMat93) (w_29_0 : mxxMat94) (w_30_0 : mxxMat94) (outputs : mxxMat94 × mxxMat94 × Unit) : Prop :=
  MxxRuntime.modulusReduceRuns w_0_0 w_1_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 128 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (5), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_2_0) w_4_0 ∧
  MxxRuntime.modulusReduceRuns w_6_0 w_7_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (5), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_2_0) w_8_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 128 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (65), (85), (99), (129), (135), (153), (165), (195), (255), (297), (325), (387), (459), (495), (561), (585), (645), (765), (891), (975), (1105), (1161), (1485), (1683), (1755), (1935), (2145), (2193), (2295), (2805), (2925), (3315), (3483), (4257), (4455), (5049), (5525), (5805), (6435), (6579), (8385), (8415), (8775), (9945), (10725), (10965), (12771), (14561), (15147), (16575), (17415), (19305), (19737), (21285), (22687), (25155), (25245), (27431), (29835), (32175), (32895), (33911), (36465), (36959), (38313), (41023), (41925), (43039), (43683), (49725), (50407), (57915), (59211), (63855), (66041), (68061), (70487), (72369), (72805), (75465), (75735), (83399), (96525), (98685), (99701), (107063), (109395), (110877), (114939), (115181), (118955), (123069), (125207), (125775), (127033), (129117), (129899), (130681), (130727), (130957), (131003), (131049), (142545), (149175), (165491), (178699), (182325), (188437), (189109), (191565), (198123), (204183), (217107), (218323), (218415), (226395), (234329), (247537)])) w_1_0 w_12_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 1024 148 w_14_0 ∧
  MxxRuntime.centeredRebaseRuns w_14_0 w_15_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [18]) ((4/1)) (26) w_18_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (5), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_2_0) w_20_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (5), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_2_0) w_22_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 128 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (65), (85), (99), (129), (135), (153), (165), (195), (255), (297), (325), (387), (459), (495), (561), (585), (645), (765), (891), (975), (1105), (1161), (1485), (1683), (1755), (1935), (2145), (2193), (2295), (2805), (2925), (3315), (3483), (4257), (4455), (5049), (5525), (5805), (6435), (6579), (8385), (8415), (8775), (9945), (10725), (10965), (12771), (14561), (15147), (16575), (17415), (19305), (19737), (21285), (22687), (25155), (25245), (27431), (29835), (32175), (32895), (33911), (36465), (36959), (38313), (41023), (41925), (43039), (43683), (49725), (50407), (57915), (59211), (63855), (66041), (68061), (70487), (72369), (72805), (75465), (75735), (83399), (96525), (98685), (99701), (107063), (109395), (110877), (114939), (115181), (118955), (123069), (125207), (125775), (127033), (129117), (129899), (130681), (130727), (130957), (131003), (131049), (142545), (149175), (165491), (178699), (182325), (188437), (189109), (191565), (198123), (204183), (217107), (218323), (218415), (226395), (234329), (247537)])) w_7_0 w_25_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [29]) ((4/1)) (26) w_29_0 ∧
  outputs = (w_19_0, (w_30_0, ()))

def parallel_parallel_generatedRoot_7_102 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat1 × ByteArray × mxxMat1 × Unit) (outputs : mxxMat94 × mxxMat94 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_2_0 : ByteArray := inputs.2.1
  let w_6_0 : mxxMat1 := inputs.2.2.1
  ∃ (w_1_0 : mxxMat93),
    let w_3_0 : Int := (MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (65), (85), (99), (129), (135), (153), (165), (195), (255), (297), (325), (387), (459), (495), (561), (585), (645), (765), (891), (975), (1105), (1161), (1485), (1683), (1755), (1935), (2145), (2193), (2295), (2805), (2925), (3315), (3483), (4257), (4455), (5049), (5525), (5805), (6435), (6579), (8385), (8415), (8775), (9945), (10725), (10965), (12771), (14561), (15147), (16575), (17415), (19305), (19737), (21285), (22687), (25155), (25245), (27431), (29835), (32175), (32895), (33911), (36465), (36959), (38313), (41023), (41925), (43039), (43683), (49725), (50407), (57915), (59211), (63855), (66041), (68061), (70487), (72369), (72805), (75465), (75735), (83399), (96525), (98685), (99701), (107063), (109395), (110877), (114939), (115181), (118955), (123069), (125207), (125775), (127033), (129117), (129899), (130681), (130727), (130957), (131003), (131049), (142545), (149175), (165491), (178699), (182325), (188437), (189109), (191565), (198123), (204183), (217107), (218323), (218415), (226395), (234329), (247537)])
    ∃ (w_4_0 : mxxMat94),
      let w_5_0 : mxxMat94 := MxxRuntime.matrixMulScalarLeft w_1_0 w_4_0
      ∃ (w_7_0 : mxxMat93),
        ∃ (w_8_0 : mxxMat94),
          let w_9_0 : mxxMat94 := MxxRuntime.matrixMulScalarLeft w_7_0 w_8_0
          let w_10_0 : mxxMat94 := MxxRuntime.matrixAdd w_5_0 w_9_0
          let w_11_0 : mxxMat93 := (MxxRuntime.matrixPolynomial [(1099391827969)] : mxxMat93)
          ∃ (w_12_0 : mxxMat93),
            let w_13_0 : mxxMat93 := MxxRuntime.matrixMulScalarLeft w_11_0 w_12_0
            ∃ (w_14_0 : mxxMat95),
              ∃ (w_15_0 : mxxMat94),
                let w_16_0 : mxxMat94 := MxxRuntime.matrixMulScalarLeft w_13_0 w_15_0
                let w_17_0 : mxxMat94 := MxxRuntime.matrixSub w_10_0 w_16_0
                ∃ (w_18_0 : mxxMat94),
                  let w_19_0 : mxxMat94 := MxxRuntime.matrixAdd w_17_0 w_18_0
                  ∃ (w_20_0 : mxxMat94),
                    let w_21_0 : mxxMat94 := MxxRuntime.matrixMulScalarLeft w_1_0 w_20_0
                    ∃ (w_22_0 : mxxMat94),
                      let w_23_0 : mxxMat94 := MxxRuntime.matrixMulScalarLeft w_7_0 w_22_0
                      let w_24_0 : mxxMat94 := MxxRuntime.matrixAdd w_21_0 w_23_0
                      ∃ (w_25_0 : mxxMat93),
                        let w_26_0 : mxxMat93 := MxxRuntime.matrixMulScalarLeft w_11_0 w_25_0
                        let w_27_0 : mxxMat94 := MxxRuntime.matrixMulScalarLeft w_26_0 w_15_0
                        let w_28_0 : mxxMat94 := MxxRuntime.matrixSub w_24_0 w_27_0
                        ∃ (w_29_0 : mxxMat94),
                          let w_30_0 : mxxMat94 := MxxRuntime.matrixAdd w_28_0 w_29_0
                          parallel_parallel_generatedRoot_7_102.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs

abbrev parallel_parallel_generatedRoot_7_103.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_1 : Nat) (w_0_0 : mxxMat1) (w_2_0 : ByteArray) (w_6_0 : mxxMat1) (w_1_0 : mxxMat93) (w_3_0 : Int) (w_4_0 : mxxMat94) (w_7_0 : mxxMat93) (w_8_0 : mxxMat94) (w_12_0 : mxxMat93) (w_14_0 : mxxMat95) (w_15_0 : mxxMat94) (w_18_0 : mxxMat94) (w_19_0 : mxxMat94) (w_20_0 : mxxMat94) (w_22_0 : mxxMat94) (w_25_0 : mxxMat93) (w_29_0 : mxxMat94) (w_30_0 : mxxMat94) (outputs : mxxMat94 × mxxMat94 × Unit) : Prop :=
  MxxRuntime.modulusReduceRuns w_0_0 w_1_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 32 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (5), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_2_0) w_4_0 ∧
  MxxRuntime.modulusReduceRuns w_6_0 w_7_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (5), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_2_0) w_8_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 32 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (85), (99), (135), (153), (165), (255), (297), (459), (495), (561), (765), (891), (1485), (1683), (2295), (2805), (4455), (5049), (8415), (15147), (25245), (75735)])) w_1_0 w_12_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 1024 148 w_14_0 ∧
  MxxRuntime.centeredRebaseRuns w_14_0 w_15_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [18]) ((4/1)) (26) w_18_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (5), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_2_0) w_20_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (5), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_2_0) w_22_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 32 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (85), (99), (135), (153), (165), (255), (297), (459), (495), (561), (765), (891), (1485), (1683), (2295), (2805), (4455), (5049), (8415), (15147), (25245), (75735)])) w_7_0 w_25_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [29]) ((4/1)) (26) w_29_0 ∧
  outputs = (w_19_0, (w_30_0, ()))

def parallel_parallel_generatedRoot_7_103 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat1 × ByteArray × mxxMat1 × Unit) (outputs : mxxMat94 × mxxMat94 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_2_0 : ByteArray := inputs.2.1
  let w_6_0 : mxxMat1 := inputs.2.2.1
  ∃ (w_1_0 : mxxMat93),
    let w_3_0 : Int := (MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (85), (99), (135), (153), (165), (255), (297), (459), (495), (561), (765), (891), (1485), (1683), (2295), (2805), (4455), (5049), (8415), (15147), (25245), (75735)])
    ∃ (w_4_0 : mxxMat94),
      let w_5_0 : mxxMat94 := MxxRuntime.matrixMulScalarLeft w_1_0 w_4_0
      ∃ (w_7_0 : mxxMat93),
        ∃ (w_8_0 : mxxMat94),
          let w_9_0 : mxxMat94 := MxxRuntime.matrixMulScalarLeft w_7_0 w_8_0
          let w_10_0 : mxxMat94 := MxxRuntime.matrixAdd w_5_0 w_9_0
          let w_11_0 : mxxMat93 := (MxxRuntime.matrixPolynomial [(1099391827969)] : mxxMat93)
          ∃ (w_12_0 : mxxMat93),
            let w_13_0 : mxxMat93 := MxxRuntime.matrixMulScalarLeft w_11_0 w_12_0
            ∃ (w_14_0 : mxxMat95),
              ∃ (w_15_0 : mxxMat94),
                let w_16_0 : mxxMat94 := MxxRuntime.matrixMulScalarLeft w_13_0 w_15_0
                let w_17_0 : mxxMat94 := MxxRuntime.matrixSub w_10_0 w_16_0
                ∃ (w_18_0 : mxxMat94),
                  let w_19_0 : mxxMat94 := MxxRuntime.matrixAdd w_17_0 w_18_0
                  ∃ (w_20_0 : mxxMat94),
                    let w_21_0 : mxxMat94 := MxxRuntime.matrixMulScalarLeft w_1_0 w_20_0
                    ∃ (w_22_0 : mxxMat94),
                      let w_23_0 : mxxMat94 := MxxRuntime.matrixMulScalarLeft w_7_0 w_22_0
                      let w_24_0 : mxxMat94 := MxxRuntime.matrixAdd w_21_0 w_23_0
                      ∃ (w_25_0 : mxxMat93),
                        let w_26_0 : mxxMat93 := MxxRuntime.matrixMulScalarLeft w_11_0 w_25_0
                        let w_27_0 : mxxMat94 := MxxRuntime.matrixMulScalarLeft w_26_0 w_15_0
                        let w_28_0 : mxxMat94 := MxxRuntime.matrixSub w_24_0 w_27_0
                        ∃ (w_29_0 : mxxMat94),
                          let w_30_0 : mxxMat94 := MxxRuntime.matrixAdd w_28_0 w_29_0
                          parallel_parallel_generatedRoot_7_103.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs

abbrev parallel_parallel_generatedRoot_7_104.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_1 : Nat) (w_0_0 : mxxMat1) (w_2_0 : ByteArray) (w_6_0 : mxxMat1) (w_1_0 : mxxMat96) (w_3_0 : Int) (w_4_0 : mxxMat97) (w_7_0 : mxxMat96) (w_8_0 : mxxMat97) (w_12_0 : mxxMat96) (w_14_0 : mxxMat98) (w_15_0 : mxxMat97) (w_18_0 : mxxMat97) (w_19_0 : mxxMat97) (w_20_0 : mxxMat97) (w_22_0 : mxxMat97) (w_25_0 : mxxMat96) (w_29_0 : mxxMat97) (w_30_0 : mxxMat97) (outputs : mxxMat97 × mxxMat97 × Unit) : Prop :=
  MxxRuntime.modulusReduceRuns w_0_0 w_1_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 128 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (6), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_2_0) w_4_0 ∧
  MxxRuntime.modulusReduceRuns w_6_0 w_7_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (6), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_2_0) w_8_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 128 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (65), (85), (99), (129), (135), (153), (165), (195), (255), (297), (325), (387), (459), (495), (561), (585), (645), (765), (891), (975), (1105), (1161), (1485), (1683), (1755), (1935), (2145), (2193), (2295), (2805), (2925), (3315), (3483), (4257), (4455), (5049), (5525), (5805), (6435), (6579), (8385), (8415), (8775), (9945), (10725), (10965), (12771), (14561), (15147), (16575), (17415), (19305), (19737), (21285), (22687), (25155), (25245), (27431), (29835), (32175), (32895), (33911), (36465), (36959), (38313), (41023), (41925), (43039), (43683), (49725), (50407), (57915), (59211), (63855), (66041), (68061), (70487), (72369), (72805), (75465), (75735), (83399), (96525), (98685), (99701), (107063), (109395), (110877), (114939), (115181), (118955), (123069), (125207), (125775), (127033), (129117), (129899), (130681), (130727), (130957), (131003), (131049), (142545), (149175), (165491), (178699), (182325), (188437), (189109), (191565), (198123), (204183), (217107), (218323), (218415), (226395), (234329), (247537)])) w_1_0 w_12_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 1024 144 w_14_0 ∧
  MxxRuntime.centeredRebaseRuns w_14_0 w_15_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [18]) ((4/1)) (26) w_18_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (6), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_2_0) w_20_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (6), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_2_0) w_22_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 128 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (65), (85), (99), (129), (135), (153), (165), (195), (255), (297), (325), (387), (459), (495), (561), (585), (645), (765), (891), (975), (1105), (1161), (1485), (1683), (1755), (1935), (2145), (2193), (2295), (2805), (2925), (3315), (3483), (4257), (4455), (5049), (5525), (5805), (6435), (6579), (8385), (8415), (8775), (9945), (10725), (10965), (12771), (14561), (15147), (16575), (17415), (19305), (19737), (21285), (22687), (25155), (25245), (27431), (29835), (32175), (32895), (33911), (36465), (36959), (38313), (41023), (41925), (43039), (43683), (49725), (50407), (57915), (59211), (63855), (66041), (68061), (70487), (72369), (72805), (75465), (75735), (83399), (96525), (98685), (99701), (107063), (109395), (110877), (114939), (115181), (118955), (123069), (125207), (125775), (127033), (129117), (129899), (130681), (130727), (130957), (131003), (131049), (142545), (149175), (165491), (178699), (182325), (188437), (189109), (191565), (198123), (204183), (217107), (218323), (218415), (226395), (234329), (247537)])) w_7_0 w_25_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [29]) ((4/1)) (26) w_29_0 ∧
  outputs = (w_19_0, (w_30_0, ()))

def parallel_parallel_generatedRoot_7_104 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat1 × ByteArray × mxxMat1 × Unit) (outputs : mxxMat97 × mxxMat97 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_2_0 : ByteArray := inputs.2.1
  let w_6_0 : mxxMat1 := inputs.2.2.1
  ∃ (w_1_0 : mxxMat96),
    let w_3_0 : Int := (MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (65), (85), (99), (129), (135), (153), (165), (195), (255), (297), (325), (387), (459), (495), (561), (585), (645), (765), (891), (975), (1105), (1161), (1485), (1683), (1755), (1935), (2145), (2193), (2295), (2805), (2925), (3315), (3483), (4257), (4455), (5049), (5525), (5805), (6435), (6579), (8385), (8415), (8775), (9945), (10725), (10965), (12771), (14561), (15147), (16575), (17415), (19305), (19737), (21285), (22687), (25155), (25245), (27431), (29835), (32175), (32895), (33911), (36465), (36959), (38313), (41023), (41925), (43039), (43683), (49725), (50407), (57915), (59211), (63855), (66041), (68061), (70487), (72369), (72805), (75465), (75735), (83399), (96525), (98685), (99701), (107063), (109395), (110877), (114939), (115181), (118955), (123069), (125207), (125775), (127033), (129117), (129899), (130681), (130727), (130957), (131003), (131049), (142545), (149175), (165491), (178699), (182325), (188437), (189109), (191565), (198123), (204183), (217107), (218323), (218415), (226395), (234329), (247537)])
    ∃ (w_4_0 : mxxMat97),
      let w_5_0 : mxxMat97 := MxxRuntime.matrixMulScalarLeft w_1_0 w_4_0
      ∃ (w_7_0 : mxxMat96),
        ∃ (w_8_0 : mxxMat97),
          let w_9_0 : mxxMat97 := MxxRuntime.matrixMulScalarLeft w_7_0 w_8_0
          let w_10_0 : mxxMat97 := MxxRuntime.matrixAdd w_5_0 w_9_0
          let w_11_0 : mxxMat96 := (MxxRuntime.matrixPolynomial [(1099394449409)] : mxxMat96)
          ∃ (w_12_0 : mxxMat96),
            let w_13_0 : mxxMat96 := MxxRuntime.matrixMulScalarLeft w_11_0 w_12_0
            ∃ (w_14_0 : mxxMat98),
              ∃ (w_15_0 : mxxMat97),
                let w_16_0 : mxxMat97 := MxxRuntime.matrixMulScalarLeft w_13_0 w_15_0
                let w_17_0 : mxxMat97 := MxxRuntime.matrixSub w_10_0 w_16_0
                ∃ (w_18_0 : mxxMat97),
                  let w_19_0 : mxxMat97 := MxxRuntime.matrixAdd w_17_0 w_18_0
                  ∃ (w_20_0 : mxxMat97),
                    let w_21_0 : mxxMat97 := MxxRuntime.matrixMulScalarLeft w_1_0 w_20_0
                    ∃ (w_22_0 : mxxMat97),
                      let w_23_0 : mxxMat97 := MxxRuntime.matrixMulScalarLeft w_7_0 w_22_0
                      let w_24_0 : mxxMat97 := MxxRuntime.matrixAdd w_21_0 w_23_0
                      ∃ (w_25_0 : mxxMat96),
                        let w_26_0 : mxxMat96 := MxxRuntime.matrixMulScalarLeft w_11_0 w_25_0
                        let w_27_0 : mxxMat97 := MxxRuntime.matrixMulScalarLeft w_26_0 w_15_0
                        let w_28_0 : mxxMat97 := MxxRuntime.matrixSub w_24_0 w_27_0
                        ∃ (w_29_0 : mxxMat97),
                          let w_30_0 : mxxMat97 := MxxRuntime.matrixAdd w_28_0 w_29_0
                          parallel_parallel_generatedRoot_7_104.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs

abbrev parallel_parallel_generatedRoot_7_105.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_1 : Nat) (w_0_0 : mxxMat1) (w_2_0 : ByteArray) (w_6_0 : mxxMat1) (w_1_0 : mxxMat96) (w_3_0 : Int) (w_4_0 : mxxMat97) (w_7_0 : mxxMat96) (w_8_0 : mxxMat97) (w_12_0 : mxxMat96) (w_14_0 : mxxMat98) (w_15_0 : mxxMat97) (w_18_0 : mxxMat97) (w_19_0 : mxxMat97) (w_20_0 : mxxMat97) (w_22_0 : mxxMat97) (w_25_0 : mxxMat96) (w_29_0 : mxxMat97) (w_30_0 : mxxMat97) (outputs : mxxMat97 × mxxMat97 × Unit) : Prop :=
  MxxRuntime.modulusReduceRuns w_0_0 w_1_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 32 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (6), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_2_0) w_4_0 ∧
  MxxRuntime.modulusReduceRuns w_6_0 w_7_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (6), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_2_0) w_8_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 32 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (85), (99), (135), (153), (165), (255), (297), (459), (495), (561), (765), (891), (1485), (1683), (2295), (2805), (4455), (5049), (8415), (15147), (25245), (75735)])) w_1_0 w_12_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 1024 144 w_14_0 ∧
  MxxRuntime.centeredRebaseRuns w_14_0 w_15_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [18]) ((4/1)) (26) w_18_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (6), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_2_0) w_20_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (6), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_2_0) w_22_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 32 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (85), (99), (135), (153), (165), (255), (297), (459), (495), (561), (765), (891), (1485), (1683), (2295), (2805), (4455), (5049), (8415), (15147), (25245), (75735)])) w_7_0 w_25_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [29]) ((4/1)) (26) w_29_0 ∧
  outputs = (w_19_0, (w_30_0, ()))

def parallel_parallel_generatedRoot_7_105 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat1 × ByteArray × mxxMat1 × Unit) (outputs : mxxMat97 × mxxMat97 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_2_0 : ByteArray := inputs.2.1
  let w_6_0 : mxxMat1 := inputs.2.2.1
  ∃ (w_1_0 : mxxMat96),
    let w_3_0 : Int := (MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (85), (99), (135), (153), (165), (255), (297), (459), (495), (561), (765), (891), (1485), (1683), (2295), (2805), (4455), (5049), (8415), (15147), (25245), (75735)])
    ∃ (w_4_0 : mxxMat97),
      let w_5_0 : mxxMat97 := MxxRuntime.matrixMulScalarLeft w_1_0 w_4_0
      ∃ (w_7_0 : mxxMat96),
        ∃ (w_8_0 : mxxMat97),
          let w_9_0 : mxxMat97 := MxxRuntime.matrixMulScalarLeft w_7_0 w_8_0
          let w_10_0 : mxxMat97 := MxxRuntime.matrixAdd w_5_0 w_9_0
          let w_11_0 : mxxMat96 := (MxxRuntime.matrixPolynomial [(1099394449409)] : mxxMat96)
          ∃ (w_12_0 : mxxMat96),
            let w_13_0 : mxxMat96 := MxxRuntime.matrixMulScalarLeft w_11_0 w_12_0
            ∃ (w_14_0 : mxxMat98),
              ∃ (w_15_0 : mxxMat97),
                let w_16_0 : mxxMat97 := MxxRuntime.matrixMulScalarLeft w_13_0 w_15_0
                let w_17_0 : mxxMat97 := MxxRuntime.matrixSub w_10_0 w_16_0
                ∃ (w_18_0 : mxxMat97),
                  let w_19_0 : mxxMat97 := MxxRuntime.matrixAdd w_17_0 w_18_0
                  ∃ (w_20_0 : mxxMat97),
                    let w_21_0 : mxxMat97 := MxxRuntime.matrixMulScalarLeft w_1_0 w_20_0
                    ∃ (w_22_0 : mxxMat97),
                      let w_23_0 : mxxMat97 := MxxRuntime.matrixMulScalarLeft w_7_0 w_22_0
                      let w_24_0 : mxxMat97 := MxxRuntime.matrixAdd w_21_0 w_23_0
                      ∃ (w_25_0 : mxxMat96),
                        let w_26_0 : mxxMat96 := MxxRuntime.matrixMulScalarLeft w_11_0 w_25_0
                        let w_27_0 : mxxMat97 := MxxRuntime.matrixMulScalarLeft w_26_0 w_15_0
                        let w_28_0 : mxxMat97 := MxxRuntime.matrixSub w_24_0 w_27_0
                        ∃ (w_29_0 : mxxMat97),
                          let w_30_0 : mxxMat97 := MxxRuntime.matrixAdd w_28_0 w_29_0
                          parallel_parallel_generatedRoot_7_105.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs

abbrev parallel_parallel_generatedRoot_7_106.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_1 : Nat) (w_0_0 : mxxMat1) (w_2_0 : ByteArray) (w_6_0 : mxxMat1) (w_1_0 : mxxMat99) (w_3_0 : Int) (w_4_0 : mxxMat100) (w_7_0 : mxxMat99) (w_8_0 : mxxMat100) (w_12_0 : mxxMat99) (w_14_0 : mxxMat101) (w_15_0 : mxxMat100) (w_18_0 : mxxMat100) (w_19_0 : mxxMat100) (w_20_0 : mxxMat100) (w_22_0 : mxxMat100) (w_25_0 : mxxMat99) (w_29_0 : mxxMat100) (w_30_0 : mxxMat100) (outputs : mxxMat100 × mxxMat100 × Unit) : Prop :=
  MxxRuntime.modulusReduceRuns w_0_0 w_1_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 128 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (7), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_2_0) w_4_0 ∧
  MxxRuntime.modulusReduceRuns w_6_0 w_7_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (7), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_2_0) w_8_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 128 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (65), (85), (99), (129), (135), (153), (165), (195), (255), (297), (325), (387), (459), (495), (561), (585), (645), (765), (891), (975), (1105), (1161), (1485), (1683), (1755), (1935), (2145), (2193), (2295), (2805), (2925), (3315), (3483), (4257), (4455), (5049), (5525), (5805), (6435), (6579), (8385), (8415), (8775), (9945), (10725), (10965), (12771), (14561), (15147), (16575), (17415), (19305), (19737), (21285), (22687), (25155), (25245), (27431), (29835), (32175), (32895), (33911), (36465), (36959), (38313), (41023), (41925), (43039), (43683), (49725), (50407), (57915), (59211), (63855), (66041), (68061), (70487), (72369), (72805), (75465), (75735), (83399), (96525), (98685), (99701), (107063), (109395), (110877), (114939), (115181), (118955), (123069), (125207), (125775), (127033), (129117), (129899), (130681), (130727), (130957), (131003), (131049), (142545), (149175), (165491), (178699), (182325), (188437), (189109), (191565), (198123), (204183), (217107), (218323), (218415), (226395), (234329), (247537)])) w_1_0 w_12_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 1024 140 w_14_0 ∧
  MxxRuntime.centeredRebaseRuns w_14_0 w_15_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [18]) ((4/1)) (26) w_18_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (7), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_2_0) w_20_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (7), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_2_0) w_22_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 128 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (65), (85), (99), (129), (135), (153), (165), (195), (255), (297), (325), (387), (459), (495), (561), (585), (645), (765), (891), (975), (1105), (1161), (1485), (1683), (1755), (1935), (2145), (2193), (2295), (2805), (2925), (3315), (3483), (4257), (4455), (5049), (5525), (5805), (6435), (6579), (8385), (8415), (8775), (9945), (10725), (10965), (12771), (14561), (15147), (16575), (17415), (19305), (19737), (21285), (22687), (25155), (25245), (27431), (29835), (32175), (32895), (33911), (36465), (36959), (38313), (41023), (41925), (43039), (43683), (49725), (50407), (57915), (59211), (63855), (66041), (68061), (70487), (72369), (72805), (75465), (75735), (83399), (96525), (98685), (99701), (107063), (109395), (110877), (114939), (115181), (118955), (123069), (125207), (125775), (127033), (129117), (129899), (130681), (130727), (130957), (131003), (131049), (142545), (149175), (165491), (178699), (182325), (188437), (189109), (191565), (198123), (204183), (217107), (218323), (218415), (226395), (234329), (247537)])) w_7_0 w_25_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [29]) ((4/1)) (26) w_29_0 ∧
  outputs = (w_19_0, (w_30_0, ()))

def parallel_parallel_generatedRoot_7_106 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat1 × ByteArray × mxxMat1 × Unit) (outputs : mxxMat100 × mxxMat100 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_2_0 : ByteArray := inputs.2.1
  let w_6_0 : mxxMat1 := inputs.2.2.1
  ∃ (w_1_0 : mxxMat99),
    let w_3_0 : Int := (MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (65), (85), (99), (129), (135), (153), (165), (195), (255), (297), (325), (387), (459), (495), (561), (585), (645), (765), (891), (975), (1105), (1161), (1485), (1683), (1755), (1935), (2145), (2193), (2295), (2805), (2925), (3315), (3483), (4257), (4455), (5049), (5525), (5805), (6435), (6579), (8385), (8415), (8775), (9945), (10725), (10965), (12771), (14561), (15147), (16575), (17415), (19305), (19737), (21285), (22687), (25155), (25245), (27431), (29835), (32175), (32895), (33911), (36465), (36959), (38313), (41023), (41925), (43039), (43683), (49725), (50407), (57915), (59211), (63855), (66041), (68061), (70487), (72369), (72805), (75465), (75735), (83399), (96525), (98685), (99701), (107063), (109395), (110877), (114939), (115181), (118955), (123069), (125207), (125775), (127033), (129117), (129899), (130681), (130727), (130957), (131003), (131049), (142545), (149175), (165491), (178699), (182325), (188437), (189109), (191565), (198123), (204183), (217107), (218323), (218415), (226395), (234329), (247537)])
    ∃ (w_4_0 : mxxMat100),
      let w_5_0 : mxxMat100 := MxxRuntime.matrixMulScalarLeft w_1_0 w_4_0
      ∃ (w_7_0 : mxxMat99),
        ∃ (w_8_0 : mxxMat100),
          let w_9_0 : mxxMat100 := MxxRuntime.matrixMulScalarLeft w_7_0 w_8_0
          let w_10_0 : mxxMat100 := MxxRuntime.matrixAdd w_5_0 w_9_0
          let w_11_0 : mxxMat99 := (MxxRuntime.matrixPolynomial [(1099396546561)] : mxxMat99)
          ∃ (w_12_0 : mxxMat99),
            let w_13_0 : mxxMat99 := MxxRuntime.matrixMulScalarLeft w_11_0 w_12_0
            ∃ (w_14_0 : mxxMat101),
              ∃ (w_15_0 : mxxMat100),
                let w_16_0 : mxxMat100 := MxxRuntime.matrixMulScalarLeft w_13_0 w_15_0
                let w_17_0 : mxxMat100 := MxxRuntime.matrixSub w_10_0 w_16_0
                ∃ (w_18_0 : mxxMat100),
                  let w_19_0 : mxxMat100 := MxxRuntime.matrixAdd w_17_0 w_18_0
                  ∃ (w_20_0 : mxxMat100),
                    let w_21_0 : mxxMat100 := MxxRuntime.matrixMulScalarLeft w_1_0 w_20_0
                    ∃ (w_22_0 : mxxMat100),
                      let w_23_0 : mxxMat100 := MxxRuntime.matrixMulScalarLeft w_7_0 w_22_0
                      let w_24_0 : mxxMat100 := MxxRuntime.matrixAdd w_21_0 w_23_0
                      ∃ (w_25_0 : mxxMat99),
                        let w_26_0 : mxxMat99 := MxxRuntime.matrixMulScalarLeft w_11_0 w_25_0
                        let w_27_0 : mxxMat100 := MxxRuntime.matrixMulScalarLeft w_26_0 w_15_0
                        let w_28_0 : mxxMat100 := MxxRuntime.matrixSub w_24_0 w_27_0
                        ∃ (w_29_0 : mxxMat100),
                          let w_30_0 : mxxMat100 := MxxRuntime.matrixAdd w_28_0 w_29_0
                          parallel_parallel_generatedRoot_7_106.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs

abbrev parallel_parallel_generatedRoot_7_107.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_1 : Nat) (w_0_0 : mxxMat1) (w_2_0 : ByteArray) (w_6_0 : mxxMat1) (w_1_0 : mxxMat99) (w_3_0 : Int) (w_4_0 : mxxMat100) (w_7_0 : mxxMat99) (w_8_0 : mxxMat100) (w_12_0 : mxxMat99) (w_14_0 : mxxMat101) (w_15_0 : mxxMat100) (w_18_0 : mxxMat100) (w_19_0 : mxxMat100) (w_20_0 : mxxMat100) (w_22_0 : mxxMat100) (w_25_0 : mxxMat99) (w_29_0 : mxxMat100) (w_30_0 : mxxMat100) (outputs : mxxMat100 × mxxMat100 × Unit) : Prop :=
  MxxRuntime.modulusReduceRuns w_0_0 w_1_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 32 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (7), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_2_0) w_4_0 ∧
  MxxRuntime.modulusReduceRuns w_6_0 w_7_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (7), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_2_0) w_8_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 32 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (85), (99), (135), (153), (165), (255), (297), (459), (495), (561), (765), (891), (1485), (1683), (2295), (2805), (4455), (5049), (8415), (15147), (25245), (75735)])) w_1_0 w_12_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 1024 140 w_14_0 ∧
  MxxRuntime.centeredRebaseRuns w_14_0 w_15_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [18]) ((4/1)) (26) w_18_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (7), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_2_0) w_20_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (7), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_2_0) w_22_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 32 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (85), (99), (135), (153), (165), (255), (297), (459), (495), (561), (765), (891), (1485), (1683), (2295), (2805), (4455), (5049), (8415), (15147), (25245), (75735)])) w_7_0 w_25_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [29]) ((4/1)) (26) w_29_0 ∧
  outputs = (w_19_0, (w_30_0, ()))

def parallel_parallel_generatedRoot_7_107 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat1 × ByteArray × mxxMat1 × Unit) (outputs : mxxMat100 × mxxMat100 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_2_0 : ByteArray := inputs.2.1
  let w_6_0 : mxxMat1 := inputs.2.2.1
  ∃ (w_1_0 : mxxMat99),
    let w_3_0 : Int := (MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (85), (99), (135), (153), (165), (255), (297), (459), (495), (561), (765), (891), (1485), (1683), (2295), (2805), (4455), (5049), (8415), (15147), (25245), (75735)])
    ∃ (w_4_0 : mxxMat100),
      let w_5_0 : mxxMat100 := MxxRuntime.matrixMulScalarLeft w_1_0 w_4_0
      ∃ (w_7_0 : mxxMat99),
        ∃ (w_8_0 : mxxMat100),
          let w_9_0 : mxxMat100 := MxxRuntime.matrixMulScalarLeft w_7_0 w_8_0
          let w_10_0 : mxxMat100 := MxxRuntime.matrixAdd w_5_0 w_9_0
          let w_11_0 : mxxMat99 := (MxxRuntime.matrixPolynomial [(1099396546561)] : mxxMat99)
          ∃ (w_12_0 : mxxMat99),
            let w_13_0 : mxxMat99 := MxxRuntime.matrixMulScalarLeft w_11_0 w_12_0
            ∃ (w_14_0 : mxxMat101),
              ∃ (w_15_0 : mxxMat100),
                let w_16_0 : mxxMat100 := MxxRuntime.matrixMulScalarLeft w_13_0 w_15_0
                let w_17_0 : mxxMat100 := MxxRuntime.matrixSub w_10_0 w_16_0
                ∃ (w_18_0 : mxxMat100),
                  let w_19_0 : mxxMat100 := MxxRuntime.matrixAdd w_17_0 w_18_0
                  ∃ (w_20_0 : mxxMat100),
                    let w_21_0 : mxxMat100 := MxxRuntime.matrixMulScalarLeft w_1_0 w_20_0
                    ∃ (w_22_0 : mxxMat100),
                      let w_23_0 : mxxMat100 := MxxRuntime.matrixMulScalarLeft w_7_0 w_22_0
                      let w_24_0 : mxxMat100 := MxxRuntime.matrixAdd w_21_0 w_23_0
                      ∃ (w_25_0 : mxxMat99),
                        let w_26_0 : mxxMat99 := MxxRuntime.matrixMulScalarLeft w_11_0 w_25_0
                        let w_27_0 : mxxMat100 := MxxRuntime.matrixMulScalarLeft w_26_0 w_15_0
                        let w_28_0 : mxxMat100 := MxxRuntime.matrixSub w_24_0 w_27_0
                        ∃ (w_29_0 : mxxMat100),
                          let w_30_0 : mxxMat100 := MxxRuntime.matrixAdd w_28_0 w_29_0
                          parallel_parallel_generatedRoot_7_107.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs

abbrev parallel_parallel_generatedRoot_7_108.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_1 : Nat) (w_0_0 : mxxMat1) (w_2_0 : ByteArray) (w_6_0 : mxxMat1) (w_1_0 : mxxMat102) (w_3_0 : Int) (w_4_0 : mxxMat103) (w_7_0 : mxxMat102) (w_8_0 : mxxMat103) (w_12_0 : mxxMat102) (w_14_0 : mxxMat104) (w_15_0 : mxxMat103) (w_18_0 : mxxMat103) (w_19_0 : mxxMat103) (w_20_0 : mxxMat103) (w_22_0 : mxxMat103) (w_25_0 : mxxMat102) (w_29_0 : mxxMat103) (w_30_0 : mxxMat103) (outputs : mxxMat103 × mxxMat103 × Unit) : Prop :=
  MxxRuntime.modulusReduceRuns w_0_0 w_1_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 128 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (8), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_2_0) w_4_0 ∧
  MxxRuntime.modulusReduceRuns w_6_0 w_7_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (8), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_2_0) w_8_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 128 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (65), (85), (99), (129), (135), (153), (165), (195), (255), (297), (325), (387), (459), (495), (561), (585), (645), (765), (891), (975), (1105), (1161), (1485), (1683), (1755), (1935), (2145), (2193), (2295), (2805), (2925), (3315), (3483), (4257), (4455), (5049), (5525), (5805), (6435), (6579), (8385), (8415), (8775), (9945), (10725), (10965), (12771), (14561), (15147), (16575), (17415), (19305), (19737), (21285), (22687), (25155), (25245), (27431), (29835), (32175), (32895), (33911), (36465), (36959), (38313), (41023), (41925), (43039), (43683), (49725), (50407), (57915), (59211), (63855), (66041), (68061), (70487), (72369), (72805), (75465), (75735), (83399), (96525), (98685), (99701), (107063), (109395), (110877), (114939), (115181), (118955), (123069), (125207), (125775), (127033), (129117), (129899), (130681), (130727), (130957), (131003), (131049), (142545), (149175), (165491), (178699), (182325), (188437), (189109), (191565), (198123), (204183), (217107), (218323), (218415), (226395), (234329), (247537)])) w_1_0 w_12_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 1024 136 w_14_0 ∧
  MxxRuntime.centeredRebaseRuns w_14_0 w_15_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [18]) ((4/1)) (26) w_18_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (8), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_2_0) w_20_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (8), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_2_0) w_22_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 128 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (65), (85), (99), (129), (135), (153), (165), (195), (255), (297), (325), (387), (459), (495), (561), (585), (645), (765), (891), (975), (1105), (1161), (1485), (1683), (1755), (1935), (2145), (2193), (2295), (2805), (2925), (3315), (3483), (4257), (4455), (5049), (5525), (5805), (6435), (6579), (8385), (8415), (8775), (9945), (10725), (10965), (12771), (14561), (15147), (16575), (17415), (19305), (19737), (21285), (22687), (25155), (25245), (27431), (29835), (32175), (32895), (33911), (36465), (36959), (38313), (41023), (41925), (43039), (43683), (49725), (50407), (57915), (59211), (63855), (66041), (68061), (70487), (72369), (72805), (75465), (75735), (83399), (96525), (98685), (99701), (107063), (109395), (110877), (114939), (115181), (118955), (123069), (125207), (125775), (127033), (129117), (129899), (130681), (130727), (130957), (131003), (131049), (142545), (149175), (165491), (178699), (182325), (188437), (189109), (191565), (198123), (204183), (217107), (218323), (218415), (226395), (234329), (247537)])) w_7_0 w_25_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [29]) ((4/1)) (26) w_29_0 ∧
  outputs = (w_19_0, (w_30_0, ()))

def parallel_parallel_generatedRoot_7_108 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat1 × ByteArray × mxxMat1 × Unit) (outputs : mxxMat103 × mxxMat103 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_2_0 : ByteArray := inputs.2.1
  let w_6_0 : mxxMat1 := inputs.2.2.1
  ∃ (w_1_0 : mxxMat102),
    let w_3_0 : Int := (MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (65), (85), (99), (129), (135), (153), (165), (195), (255), (297), (325), (387), (459), (495), (561), (585), (645), (765), (891), (975), (1105), (1161), (1485), (1683), (1755), (1935), (2145), (2193), (2295), (2805), (2925), (3315), (3483), (4257), (4455), (5049), (5525), (5805), (6435), (6579), (8385), (8415), (8775), (9945), (10725), (10965), (12771), (14561), (15147), (16575), (17415), (19305), (19737), (21285), (22687), (25155), (25245), (27431), (29835), (32175), (32895), (33911), (36465), (36959), (38313), (41023), (41925), (43039), (43683), (49725), (50407), (57915), (59211), (63855), (66041), (68061), (70487), (72369), (72805), (75465), (75735), (83399), (96525), (98685), (99701), (107063), (109395), (110877), (114939), (115181), (118955), (123069), (125207), (125775), (127033), (129117), (129899), (130681), (130727), (130957), (131003), (131049), (142545), (149175), (165491), (178699), (182325), (188437), (189109), (191565), (198123), (204183), (217107), (218323), (218415), (226395), (234329), (247537)])
    ∃ (w_4_0 : mxxMat103),
      let w_5_0 : mxxMat103 := MxxRuntime.matrixMulScalarLeft w_1_0 w_4_0
      ∃ (w_7_0 : mxxMat102),
        ∃ (w_8_0 : mxxMat103),
          let w_9_0 : mxxMat103 := MxxRuntime.matrixMulScalarLeft w_7_0 w_8_0
          let w_10_0 : mxxMat103 := MxxRuntime.matrixAdd w_5_0 w_9_0
          let w_11_0 : mxxMat102 := (MxxRuntime.matrixPolynomial [(1099403100161)] : mxxMat102)
          ∃ (w_12_0 : mxxMat102),
            let w_13_0 : mxxMat102 := MxxRuntime.matrixMulScalarLeft w_11_0 w_12_0
            ∃ (w_14_0 : mxxMat104),
              ∃ (w_15_0 : mxxMat103),
                let w_16_0 : mxxMat103 := MxxRuntime.matrixMulScalarLeft w_13_0 w_15_0
                let w_17_0 : mxxMat103 := MxxRuntime.matrixSub w_10_0 w_16_0
                ∃ (w_18_0 : mxxMat103),
                  let w_19_0 : mxxMat103 := MxxRuntime.matrixAdd w_17_0 w_18_0
                  ∃ (w_20_0 : mxxMat103),
                    let w_21_0 : mxxMat103 := MxxRuntime.matrixMulScalarLeft w_1_0 w_20_0
                    ∃ (w_22_0 : mxxMat103),
                      let w_23_0 : mxxMat103 := MxxRuntime.matrixMulScalarLeft w_7_0 w_22_0
                      let w_24_0 : mxxMat103 := MxxRuntime.matrixAdd w_21_0 w_23_0
                      ∃ (w_25_0 : mxxMat102),
                        let w_26_0 : mxxMat102 := MxxRuntime.matrixMulScalarLeft w_11_0 w_25_0
                        let w_27_0 : mxxMat103 := MxxRuntime.matrixMulScalarLeft w_26_0 w_15_0
                        let w_28_0 : mxxMat103 := MxxRuntime.matrixSub w_24_0 w_27_0
                        ∃ (w_29_0 : mxxMat103),
                          let w_30_0 : mxxMat103 := MxxRuntime.matrixAdd w_28_0 w_29_0
                          parallel_parallel_generatedRoot_7_108.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs

abbrev parallel_parallel_generatedRoot_7_109.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_1 : Nat) (w_0_0 : mxxMat1) (w_2_0 : ByteArray) (w_6_0 : mxxMat1) (w_1_0 : mxxMat102) (w_3_0 : Int) (w_4_0 : mxxMat103) (w_7_0 : mxxMat102) (w_8_0 : mxxMat103) (w_12_0 : mxxMat102) (w_14_0 : mxxMat104) (w_15_0 : mxxMat103) (w_18_0 : mxxMat103) (w_19_0 : mxxMat103) (w_20_0 : mxxMat103) (w_22_0 : mxxMat103) (w_25_0 : mxxMat102) (w_29_0 : mxxMat103) (w_30_0 : mxxMat103) (outputs : mxxMat103 × mxxMat103 × Unit) : Prop :=
  MxxRuntime.modulusReduceRuns w_0_0 w_1_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 32 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (8), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_2_0) w_4_0 ∧
  MxxRuntime.modulusReduceRuns w_6_0 w_7_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (8), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_2_0) w_8_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 32 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (85), (99), (135), (153), (165), (255), (297), (459), (495), (561), (765), (891), (1485), (1683), (2295), (2805), (4455), (5049), (8415), (15147), (25245), (75735)])) w_1_0 w_12_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 1024 136 w_14_0 ∧
  MxxRuntime.centeredRebaseRuns w_14_0 w_15_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [18]) ((4/1)) (26) w_18_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (8), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_2_0) w_20_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (8), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_2_0) w_22_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 32 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (85), (99), (135), (153), (165), (255), (297), (459), (495), (561), (765), (891), (1485), (1683), (2295), (2805), (4455), (5049), (8415), (15147), (25245), (75735)])) w_7_0 w_25_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [29]) ((4/1)) (26) w_29_0 ∧
  outputs = (w_19_0, (w_30_0, ()))

def parallel_parallel_generatedRoot_7_109 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat1 × ByteArray × mxxMat1 × Unit) (outputs : mxxMat103 × mxxMat103 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_2_0 : ByteArray := inputs.2.1
  let w_6_0 : mxxMat1 := inputs.2.2.1
  ∃ (w_1_0 : mxxMat102),
    let w_3_0 : Int := (MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (85), (99), (135), (153), (165), (255), (297), (459), (495), (561), (765), (891), (1485), (1683), (2295), (2805), (4455), (5049), (8415), (15147), (25245), (75735)])
    ∃ (w_4_0 : mxxMat103),
      let w_5_0 : mxxMat103 := MxxRuntime.matrixMulScalarLeft w_1_0 w_4_0
      ∃ (w_7_0 : mxxMat102),
        ∃ (w_8_0 : mxxMat103),
          let w_9_0 : mxxMat103 := MxxRuntime.matrixMulScalarLeft w_7_0 w_8_0
          let w_10_0 : mxxMat103 := MxxRuntime.matrixAdd w_5_0 w_9_0
          let w_11_0 : mxxMat102 := (MxxRuntime.matrixPolynomial [(1099403100161)] : mxxMat102)
          ∃ (w_12_0 : mxxMat102),
            let w_13_0 : mxxMat102 := MxxRuntime.matrixMulScalarLeft w_11_0 w_12_0
            ∃ (w_14_0 : mxxMat104),
              ∃ (w_15_0 : mxxMat103),
                let w_16_0 : mxxMat103 := MxxRuntime.matrixMulScalarLeft w_13_0 w_15_0
                let w_17_0 : mxxMat103 := MxxRuntime.matrixSub w_10_0 w_16_0
                ∃ (w_18_0 : mxxMat103),
                  let w_19_0 : mxxMat103 := MxxRuntime.matrixAdd w_17_0 w_18_0
                  ∃ (w_20_0 : mxxMat103),
                    let w_21_0 : mxxMat103 := MxxRuntime.matrixMulScalarLeft w_1_0 w_20_0
                    ∃ (w_22_0 : mxxMat103),
                      let w_23_0 : mxxMat103 := MxxRuntime.matrixMulScalarLeft w_7_0 w_22_0
                      let w_24_0 : mxxMat103 := MxxRuntime.matrixAdd w_21_0 w_23_0
                      ∃ (w_25_0 : mxxMat102),
                        let w_26_0 : mxxMat102 := MxxRuntime.matrixMulScalarLeft w_11_0 w_25_0
                        let w_27_0 : mxxMat103 := MxxRuntime.matrixMulScalarLeft w_26_0 w_15_0
                        let w_28_0 : mxxMat103 := MxxRuntime.matrixSub w_24_0 w_27_0
                        ∃ (w_29_0 : mxxMat103),
                          let w_30_0 : mxxMat103 := MxxRuntime.matrixAdd w_28_0 w_29_0
                          parallel_parallel_generatedRoot_7_109.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs

abbrev parallel_parallel_generatedRoot_7_110.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_1 : Nat) (w_0_0 : mxxMat1) (w_2_0 : ByteArray) (w_6_0 : mxxMat1) (w_1_0 : mxxMat105) (w_3_0 : Int) (w_4_0 : mxxMat106) (w_7_0 : mxxMat105) (w_8_0 : mxxMat106) (w_12_0 : mxxMat105) (w_14_0 : mxxMat107) (w_15_0 : mxxMat106) (w_18_0 : mxxMat106) (w_19_0 : mxxMat106) (w_20_0 : mxxMat106) (w_22_0 : mxxMat106) (w_25_0 : mxxMat105) (w_29_0 : mxxMat106) (w_30_0 : mxxMat106) (outputs : mxxMat106 × mxxMat106 × Unit) : Prop :=
  MxxRuntime.modulusReduceRuns w_0_0 w_1_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 128 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (9), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_2_0) w_4_0 ∧
  MxxRuntime.modulusReduceRuns w_6_0 w_7_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (9), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_2_0) w_8_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 128 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (65), (85), (99), (129), (135), (153), (165), (195), (255), (297), (325), (387), (459), (495), (561), (585), (645), (765), (891), (975), (1105), (1161), (1485), (1683), (1755), (1935), (2145), (2193), (2295), (2805), (2925), (3315), (3483), (4257), (4455), (5049), (5525), (5805), (6435), (6579), (8385), (8415), (8775), (9945), (10725), (10965), (12771), (14561), (15147), (16575), (17415), (19305), (19737), (21285), (22687), (25155), (25245), (27431), (29835), (32175), (32895), (33911), (36465), (36959), (38313), (41023), (41925), (43039), (43683), (49725), (50407), (57915), (59211), (63855), (66041), (68061), (70487), (72369), (72805), (75465), (75735), (83399), (96525), (98685), (99701), (107063), (109395), (110877), (114939), (115181), (118955), (123069), (125207), (125775), (127033), (129117), (129899), (130681), (130727), (130957), (131003), (131049), (142545), (149175), (165491), (178699), (182325), (188437), (189109), (191565), (198123), (204183), (217107), (218323), (218415), (226395), (234329), (247537)])) w_1_0 w_12_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 1024 132 w_14_0 ∧
  MxxRuntime.centeredRebaseRuns w_14_0 w_15_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [18]) ((4/1)) (26) w_18_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (9), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_2_0) w_20_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (9), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_2_0) w_22_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 128 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (65), (85), (99), (129), (135), (153), (165), (195), (255), (297), (325), (387), (459), (495), (561), (585), (645), (765), (891), (975), (1105), (1161), (1485), (1683), (1755), (1935), (2145), (2193), (2295), (2805), (2925), (3315), (3483), (4257), (4455), (5049), (5525), (5805), (6435), (6579), (8385), (8415), (8775), (9945), (10725), (10965), (12771), (14561), (15147), (16575), (17415), (19305), (19737), (21285), (22687), (25155), (25245), (27431), (29835), (32175), (32895), (33911), (36465), (36959), (38313), (41023), (41925), (43039), (43683), (49725), (50407), (57915), (59211), (63855), (66041), (68061), (70487), (72369), (72805), (75465), (75735), (83399), (96525), (98685), (99701), (107063), (109395), (110877), (114939), (115181), (118955), (123069), (125207), (125775), (127033), (129117), (129899), (130681), (130727), (130957), (131003), (131049), (142545), (149175), (165491), (178699), (182325), (188437), (189109), (191565), (198123), (204183), (217107), (218323), (218415), (226395), (234329), (247537)])) w_7_0 w_25_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [29]) ((4/1)) (26) w_29_0 ∧
  outputs = (w_19_0, (w_30_0, ()))

def parallel_parallel_generatedRoot_7_110 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat1 × ByteArray × mxxMat1 × Unit) (outputs : mxxMat106 × mxxMat106 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_2_0 : ByteArray := inputs.2.1
  let w_6_0 : mxxMat1 := inputs.2.2.1
  ∃ (w_1_0 : mxxMat105),
    let w_3_0 : Int := (MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (65), (85), (99), (129), (135), (153), (165), (195), (255), (297), (325), (387), (459), (495), (561), (585), (645), (765), (891), (975), (1105), (1161), (1485), (1683), (1755), (1935), (2145), (2193), (2295), (2805), (2925), (3315), (3483), (4257), (4455), (5049), (5525), (5805), (6435), (6579), (8385), (8415), (8775), (9945), (10725), (10965), (12771), (14561), (15147), (16575), (17415), (19305), (19737), (21285), (22687), (25155), (25245), (27431), (29835), (32175), (32895), (33911), (36465), (36959), (38313), (41023), (41925), (43039), (43683), (49725), (50407), (57915), (59211), (63855), (66041), (68061), (70487), (72369), (72805), (75465), (75735), (83399), (96525), (98685), (99701), (107063), (109395), (110877), (114939), (115181), (118955), (123069), (125207), (125775), (127033), (129117), (129899), (130681), (130727), (130957), (131003), (131049), (142545), (149175), (165491), (178699), (182325), (188437), (189109), (191565), (198123), (204183), (217107), (218323), (218415), (226395), (234329), (247537)])
    ∃ (w_4_0 : mxxMat106),
      let w_5_0 : mxxMat106 := MxxRuntime.matrixMulScalarLeft w_1_0 w_4_0
      ∃ (w_7_0 : mxxMat105),
        ∃ (w_8_0 : mxxMat106),
          let w_9_0 : mxxMat106 := MxxRuntime.matrixMulScalarLeft w_7_0 w_8_0
          let w_10_0 : mxxMat106 := MxxRuntime.matrixAdd w_5_0 w_9_0
          let w_11_0 : mxxMat105 := (MxxRuntime.matrixPolynomial [(1099405197313)] : mxxMat105)
          ∃ (w_12_0 : mxxMat105),
            let w_13_0 : mxxMat105 := MxxRuntime.matrixMulScalarLeft w_11_0 w_12_0
            ∃ (w_14_0 : mxxMat107),
              ∃ (w_15_0 : mxxMat106),
                let w_16_0 : mxxMat106 := MxxRuntime.matrixMulScalarLeft w_13_0 w_15_0
                let w_17_0 : mxxMat106 := MxxRuntime.matrixSub w_10_0 w_16_0
                ∃ (w_18_0 : mxxMat106),
                  let w_19_0 : mxxMat106 := MxxRuntime.matrixAdd w_17_0 w_18_0
                  ∃ (w_20_0 : mxxMat106),
                    let w_21_0 : mxxMat106 := MxxRuntime.matrixMulScalarLeft w_1_0 w_20_0
                    ∃ (w_22_0 : mxxMat106),
                      let w_23_0 : mxxMat106 := MxxRuntime.matrixMulScalarLeft w_7_0 w_22_0
                      let w_24_0 : mxxMat106 := MxxRuntime.matrixAdd w_21_0 w_23_0
                      ∃ (w_25_0 : mxxMat105),
                        let w_26_0 : mxxMat105 := MxxRuntime.matrixMulScalarLeft w_11_0 w_25_0
                        let w_27_0 : mxxMat106 := MxxRuntime.matrixMulScalarLeft w_26_0 w_15_0
                        let w_28_0 : mxxMat106 := MxxRuntime.matrixSub w_24_0 w_27_0
                        ∃ (w_29_0 : mxxMat106),
                          let w_30_0 : mxxMat106 := MxxRuntime.matrixAdd w_28_0 w_29_0
                          parallel_parallel_generatedRoot_7_110.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs

abbrev parallel_parallel_generatedRoot_7_111.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_1 : Nat) (w_0_0 : mxxMat1) (w_2_0 : ByteArray) (w_6_0 : mxxMat1) (w_1_0 : mxxMat105) (w_3_0 : Int) (w_4_0 : mxxMat106) (w_7_0 : mxxMat105) (w_8_0 : mxxMat106) (w_12_0 : mxxMat105) (w_14_0 : mxxMat107) (w_15_0 : mxxMat106) (w_18_0 : mxxMat106) (w_19_0 : mxxMat106) (w_20_0 : mxxMat106) (w_22_0 : mxxMat106) (w_25_0 : mxxMat105) (w_29_0 : mxxMat106) (w_30_0 : mxxMat106) (outputs : mxxMat106 × mxxMat106 × Unit) : Prop :=
  MxxRuntime.modulusReduceRuns w_0_0 w_1_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 32 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (9), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_2_0) w_4_0 ∧
  MxxRuntime.modulusReduceRuns w_6_0 w_7_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (9), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_2_0) w_8_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 32 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (85), (99), (135), (153), (165), (255), (297), (459), (495), (561), (765), (891), (1485), (1683), (2295), (2805), (4455), (5049), (8415), (15147), (25245), (75735)])) w_1_0 w_12_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 1024 132 w_14_0 ∧
  MxxRuntime.centeredRebaseRuns w_14_0 w_15_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [18]) ((4/1)) (26) w_18_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (9), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_2_0) w_20_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (9), .integer w_3_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_2_0) w_22_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 32 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (85), (99), (135), (153), (165), (255), (297), (459), (495), (561), (765), (891), (1485), (1683), (2295), (2805), (4455), (5049), (8415), (15147), (25245), (75735)])) w_7_0 w_25_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [29]) ((4/1)) (26) w_29_0 ∧
  outputs = (w_19_0, (w_30_0, ()))

def parallel_parallel_generatedRoot_7_111 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat1 × ByteArray × mxxMat1 × Unit) (outputs : mxxMat106 × mxxMat106 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_2_0 : ByteArray := inputs.2.1
  let w_6_0 : mxxMat1 := inputs.2.2.1
  ∃ (w_1_0 : mxxMat105),
    let w_3_0 : Int := (MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (85), (99), (135), (153), (165), (255), (297), (459), (495), (561), (765), (891), (1485), (1683), (2295), (2805), (4455), (5049), (8415), (15147), (25245), (75735)])
    ∃ (w_4_0 : mxxMat106),
      let w_5_0 : mxxMat106 := MxxRuntime.matrixMulScalarLeft w_1_0 w_4_0
      ∃ (w_7_0 : mxxMat105),
        ∃ (w_8_0 : mxxMat106),
          let w_9_0 : mxxMat106 := MxxRuntime.matrixMulScalarLeft w_7_0 w_8_0
          let w_10_0 : mxxMat106 := MxxRuntime.matrixAdd w_5_0 w_9_0
          let w_11_0 : mxxMat105 := (MxxRuntime.matrixPolynomial [(1099405197313)] : mxxMat105)
          ∃ (w_12_0 : mxxMat105),
            let w_13_0 : mxxMat105 := MxxRuntime.matrixMulScalarLeft w_11_0 w_12_0
            ∃ (w_14_0 : mxxMat107),
              ∃ (w_15_0 : mxxMat106),
                let w_16_0 : mxxMat106 := MxxRuntime.matrixMulScalarLeft w_13_0 w_15_0
                let w_17_0 : mxxMat106 := MxxRuntime.matrixSub w_10_0 w_16_0
                ∃ (w_18_0 : mxxMat106),
                  let w_19_0 : mxxMat106 := MxxRuntime.matrixAdd w_17_0 w_18_0
                  ∃ (w_20_0 : mxxMat106),
                    let w_21_0 : mxxMat106 := MxxRuntime.matrixMulScalarLeft w_1_0 w_20_0
                    ∃ (w_22_0 : mxxMat106),
                      let w_23_0 : mxxMat106 := MxxRuntime.matrixMulScalarLeft w_7_0 w_22_0
                      let w_24_0 : mxxMat106 := MxxRuntime.matrixAdd w_21_0 w_23_0
                      ∃ (w_25_0 : mxxMat105),
                        let w_26_0 : mxxMat105 := MxxRuntime.matrixMulScalarLeft w_11_0 w_25_0
                        let w_27_0 : mxxMat106 := MxxRuntime.matrixMulScalarLeft w_26_0 w_15_0
                        let w_28_0 : mxxMat106 := MxxRuntime.matrixSub w_24_0 w_27_0
                        ∃ (w_29_0 : mxxMat106),
                          let w_30_0 : mxxMat106 := MxxRuntime.matrixAdd w_28_0 w_29_0
                          parallel_parallel_generatedRoot_7_111.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_7.Witness where
  w_1_0 : mxxMat1
  w_3_0 : mxxMat2
  w_6_0 : mxxMat1
  w_7_0 : mxxMat2
  w_11_0 : mxxMat1
  w_12_0 : mxxMat3
  w_13_0 : mxxMat2
  w_23_0 : mxxMat2
  w_25_0 : mxxMat2
  w_27_0 : mxxMat2
  w_31_0 : mxxMat1
  w_35_0 : mxxMat2
  w_39_0 : Fin 1025 → mxxMat0
  w_39_1 : Fin 1025 → mxxMat0
  w_39_2 : Fin 1025 → mxxMat1
  w_40_0 : Fin 128 → mxxMat2
  w_40_1 : Fin 128 → mxxMat2
  w_41_0 : Fin 32 → mxxMat2
  w_41_1 : Fin 32 → mxxMat2
  w_42_0 : Fin 128 → mxxMat5
  w_42_1 : Fin 128 → mxxMat5
  w_43_0 : Fin 32 → mxxMat5
  w_43_1 : Fin 32 → mxxMat5
  w_44_0 : Fin 128 → mxxMat8
  w_44_1 : Fin 128 → mxxMat8
  w_45_0 : Fin 32 → mxxMat8
  w_45_1 : Fin 32 → mxxMat8
  w_46_0 : Fin 128 → mxxMat11
  w_46_1 : Fin 128 → mxxMat11
  w_47_0 : Fin 32 → mxxMat11
  w_47_1 : Fin 32 → mxxMat11
  w_48_0 : Fin 128 → mxxMat14
  w_48_1 : Fin 128 → mxxMat14
  w_49_0 : Fin 32 → mxxMat14
  w_49_1 : Fin 32 → mxxMat14
  w_50_0 : Fin 128 → mxxMat17
  w_50_1 : Fin 128 → mxxMat17
  w_51_0 : Fin 32 → mxxMat17
  w_51_1 : Fin 32 → mxxMat17
  w_52_0 : Fin 128 → mxxMat20
  w_52_1 : Fin 128 → mxxMat20
  w_53_0 : Fin 32 → mxxMat20
  w_53_1 : Fin 32 → mxxMat20
  w_54_0 : Fin 128 → mxxMat23
  w_54_1 : Fin 128 → mxxMat23
  w_55_0 : Fin 32 → mxxMat23
  w_55_1 : Fin 32 → mxxMat23
  w_56_0 : Fin 128 → mxxMat26
  w_56_1 : Fin 128 → mxxMat26
  w_57_0 : Fin 32 → mxxMat26
  w_57_1 : Fin 32 → mxxMat26
  w_58_0 : Fin 128 → mxxMat29
  w_58_1 : Fin 128 → mxxMat29
  w_59_0 : Fin 32 → mxxMat29
  w_59_1 : Fin 32 → mxxMat29
  w_60_0 : Fin 128 → mxxMat32
  w_60_1 : Fin 128 → mxxMat32
  w_61_0 : Fin 32 → mxxMat32
  w_61_1 : Fin 32 → mxxMat32
  w_62_0 : Fin 128 → mxxMat35
  w_62_1 : Fin 128 → mxxMat35
  w_63_0 : Fin 32 → mxxMat35
  w_63_1 : Fin 32 → mxxMat35
  w_64_0 : Fin 128 → mxxMat38
  w_64_1 : Fin 128 → mxxMat38
  w_65_0 : Fin 32 → mxxMat38
  w_65_1 : Fin 32 → mxxMat38
  w_66_0 : Fin 128 → mxxMat41
  w_66_1 : Fin 128 → mxxMat41
  w_67_0 : Fin 32 → mxxMat41
  w_67_1 : Fin 32 → mxxMat41
  w_68_0 : Fin 128 → mxxMat44
  w_68_1 : Fin 128 → mxxMat44
  w_69_0 : Fin 32 → mxxMat44
  w_69_1 : Fin 32 → mxxMat44
  w_70_0 : Fin 128 → mxxMat47
  w_70_1 : Fin 128 → mxxMat47
  w_71_0 : Fin 32 → mxxMat47
  w_71_1 : Fin 32 → mxxMat47
  w_72_0 : Fin 128 → mxxMat50
  w_72_1 : Fin 128 → mxxMat50
  w_73_0 : Fin 32 → mxxMat50
  w_73_1 : Fin 32 → mxxMat50
  w_74_0 : Fin 128 → mxxMat53
  w_74_1 : Fin 128 → mxxMat53
  w_75_0 : Fin 32 → mxxMat53
  w_75_1 : Fin 32 → mxxMat53
  w_76_0 : Fin 128 → mxxMat56
  w_76_1 : Fin 128 → mxxMat56
  w_77_0 : Fin 32 → mxxMat56
  w_77_1 : Fin 32 → mxxMat56
  w_78_0 : Fin 128 → mxxMat59
  w_78_1 : Fin 128 → mxxMat59
  w_79_0 : Fin 32 → mxxMat59
  w_79_1 : Fin 32 → mxxMat59
  w_80_0 : Fin 128 → mxxMat62
  w_80_1 : Fin 128 → mxxMat62
  w_81_0 : Fin 32 → mxxMat62
  w_81_1 : Fin 32 → mxxMat62
  w_82_0 : Fin 128 → mxxMat65
  w_82_1 : Fin 128 → mxxMat65
  w_83_0 : Fin 32 → mxxMat65
  w_83_1 : Fin 32 → mxxMat65
  w_84_0 : Fin 128 → mxxMat68
  w_84_1 : Fin 128 → mxxMat68
  w_85_0 : Fin 32 → mxxMat68
  w_85_1 : Fin 32 → mxxMat68
  w_86_0 : Fin 128 → mxxMat71
  w_86_1 : Fin 128 → mxxMat71
  w_87_0 : Fin 32 → mxxMat71
  w_87_1 : Fin 32 → mxxMat71
  w_88_0 : Fin 128 → mxxMat74
  w_88_1 : Fin 128 → mxxMat74
  w_89_0 : Fin 32 → mxxMat74
  w_89_1 : Fin 32 → mxxMat74
  w_90_0 : Fin 128 → mxxMat77
  w_90_1 : Fin 128 → mxxMat77
  w_91_0 : Fin 32 → mxxMat77
  w_91_1 : Fin 32 → mxxMat77
  w_92_0 : Fin 128 → mxxMat80
  w_92_1 : Fin 128 → mxxMat80
  w_93_0 : Fin 32 → mxxMat80
  w_93_1 : Fin 32 → mxxMat80
  w_94_0 : Fin 128 → mxxMat83
  w_94_1 : Fin 128 → mxxMat83
  w_95_0 : Fin 32 → mxxMat83
  w_95_1 : Fin 32 → mxxMat83
  w_96_0 : Fin 128 → mxxMat86
  w_96_1 : Fin 128 → mxxMat86
  w_97_0 : Fin 32 → mxxMat86
  w_97_1 : Fin 32 → mxxMat86
  w_98_0 : Fin 128 → mxxMat89
  w_98_1 : Fin 128 → mxxMat89
  w_99_0 : Fin 32 → mxxMat89
  w_99_1 : Fin 32 → mxxMat89
  w_100_0 : Fin 128 → mxxMat91
  w_100_1 : Fin 128 → mxxMat91
  w_101_0 : Fin 32 → mxxMat91
  w_101_1 : Fin 32 → mxxMat91
  w_102_0 : Fin 128 → mxxMat94
  w_102_1 : Fin 128 → mxxMat94
  w_103_0 : Fin 32 → mxxMat94
  w_103_1 : Fin 32 → mxxMat94
  w_104_0 : Fin 128 → mxxMat97
  w_104_1 : Fin 128 → mxxMat97
  w_105_0 : Fin 32 → mxxMat97
  w_105_1 : Fin 32 → mxxMat97
  w_106_0 : Fin 128 → mxxMat100
  w_106_1 : Fin 128 → mxxMat100
  w_107_0 : Fin 32 → mxxMat100
  w_107_1 : Fin 32 → mxxMat100
  w_108_0 : Fin 128 → mxxMat103
  w_108_1 : Fin 128 → mxxMat103
  w_109_0 : Fin 32 → mxxMat103
  w_109_1 : Fin 32 → mxxMat103
  w_110_0 : Fin 128 → mxxMat106
  w_110_1 : Fin 128 → mxxMat106
  w_111_0 : Fin 32 → mxxMat106
  w_111_1 : Fin 32 → mxxMat106

abbrev parallel_generatedRoot_7.constraints_4 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_2_0 : ByteArray) (w_1_0 : mxxMat1) (w_6_0 : mxxMat1) (w_24_0 : mxxMat2) (w_36_0 : mxxMat2) (w_39_0 : Fin 1025 → mxxMat0) (w_39_1 : Fin 1025 → mxxMat0) (w_39_2 : Fin 1025 → mxxMat1) (w_40_0 : Fin 128 → mxxMat2) (w_40_1 : Fin 128 → mxxMat2) (w_41_0 : Fin 32 → mxxMat2) (w_41_1 : Fin 32 → mxxMat2) (w_42_0 : Fin 128 → mxxMat5) (w_42_1 : Fin 128 → mxxMat5) (w_43_0 : Fin 32 → mxxMat5) (w_43_1 : Fin 32 → mxxMat5) (w_44_0 : Fin 128 → mxxMat8) (w_44_1 : Fin 128 → mxxMat8) (w_45_0 : Fin 32 → mxxMat8) (w_45_1 : Fin 32 → mxxMat8) (w_46_0 : Fin 128 → mxxMat11) (w_46_1 : Fin 128 → mxxMat11) (w_47_0 : Fin 32 → mxxMat11) (w_47_1 : Fin 32 → mxxMat11) (w_48_0 : Fin 128 → mxxMat14) (w_48_1 : Fin 128 → mxxMat14) (w_49_0 : Fin 32 → mxxMat14) (w_49_1 : Fin 32 → mxxMat14) (w_50_0 : Fin 128 → mxxMat17) (w_50_1 : Fin 128 → mxxMat17) (w_51_0 : Fin 32 → mxxMat17) (w_51_1 : Fin 32 → mxxMat17) (w_52_0 : Fin 128 → mxxMat20) (w_52_1 : Fin 128 → mxxMat20) (w_53_0 : Fin 32 → mxxMat20) (w_53_1 : Fin 32 → mxxMat20) (w_54_0 : Fin 128 → mxxMat23) (w_54_1 : Fin 128 → mxxMat23) (w_55_0 : Fin 32 → mxxMat23) (w_55_1 : Fin 32 → mxxMat23) (w_56_0 : Fin 128 → mxxMat26) (w_56_1 : Fin 128 → mxxMat26) (w_57_0 : Fin 32 → mxxMat26) (w_57_1 : Fin 32 → mxxMat26) (w_58_0 : Fin 128 → mxxMat29) (w_58_1 : Fin 128 → mxxMat29) (w_59_0 : Fin 32 → mxxMat29) (w_59_1 : Fin 32 → mxxMat29) (w_60_0 : Fin 128 → mxxMat32) (w_60_1 : Fin 128 → mxxMat32) (w_61_0 : Fin 32 → mxxMat32) (w_61_1 : Fin 32 → mxxMat32) (w_62_0 : Fin 128 → mxxMat35) (w_62_1 : Fin 128 → mxxMat35) (w_63_0 : Fin 32 → mxxMat35) (w_63_1 : Fin 32 → mxxMat35) (w_64_0 : Fin 128 → mxxMat38) (w_64_1 : Fin 128 → mxxMat38) (w_65_0 : Fin 32 → mxxMat38) (w_65_1 : Fin 32 → mxxMat38) (w_66_0 : Fin 128 → mxxMat41) (w_66_1 : Fin 128 → mxxMat41) (w_67_0 : Fin 32 → mxxMat41) (w_67_1 : Fin 32 → mxxMat41) (w_68_0 : Fin 128 → mxxMat44) (w_68_1 : Fin 128 → mxxMat44) (w_69_0 : Fin 32 → mxxMat44) (w_69_1 : Fin 32 → mxxMat44) (w_70_0 : Fin 128 → mxxMat47) (w_70_1 : Fin 128 → mxxMat47) (w_71_0 : Fin 32 → mxxMat47) (w_71_1 : Fin 32 → mxxMat47) (w_72_0 : Fin 128 → mxxMat50) (w_72_1 : Fin 128 → mxxMat50) (w_73_0 : Fin 32 → mxxMat50) (w_73_1 : Fin 32 → mxxMat50) (w_74_0 : Fin 128 → mxxMat53) (w_74_1 : Fin 128 → mxxMat53) (w_75_0 : Fin 32 → mxxMat53) (w_75_1 : Fin 32 → mxxMat53) (w_76_0 : Fin 128 → mxxMat56) (w_76_1 : Fin 128 → mxxMat56) (w_77_0 : Fin 32 → mxxMat56) (w_77_1 : Fin 32 → mxxMat56) (w_78_0 : Fin 128 → mxxMat59) (w_78_1 : Fin 128 → mxxMat59) (w_79_0 : Fin 32 → mxxMat59) (w_79_1 : Fin 32 → mxxMat59) (w_80_0 : Fin 128 → mxxMat62) (w_80_1 : Fin 128 → mxxMat62) (w_81_0 : Fin 32 → mxxMat62) (w_81_1 : Fin 32 → mxxMat62) (w_82_0 : Fin 128 → mxxMat65) (w_82_1 : Fin 128 → mxxMat65) (w_83_0 : Fin 32 → mxxMat65) (w_83_1 : Fin 32 → mxxMat65) (w_84_0 : Fin 128 → mxxMat68) (w_84_1 : Fin 128 → mxxMat68) (w_85_0 : Fin 32 → mxxMat68) (w_85_1 : Fin 32 → mxxMat68) (w_86_0 : Fin 128 → mxxMat71) (w_86_1 : Fin 128 → mxxMat71) (w_87_0 : Fin 32 → mxxMat71) (w_87_1 : Fin 32 → mxxMat71) (w_88_0 : Fin 128 → mxxMat74) (w_88_1 : Fin 128 → mxxMat74) (w_89_0 : Fin 32 → mxxMat74) (w_89_1 : Fin 32 → mxxMat74) (w_90_0 : Fin 128 → mxxMat77) (w_90_1 : Fin 128 → mxxMat77) (w_91_0 : Fin 32 → mxxMat77) (w_91_1 : Fin 32 → mxxMat77) (w_92_0 : Fin 128 → mxxMat80) (w_92_1 : Fin 128 → mxxMat80) (w_93_0 : Fin 32 → mxxMat80) (w_93_1 : Fin 32 → mxxMat80) (w_94_0 : Fin 128 → mxxMat83) (w_94_1 : Fin 128 → mxxMat83) (w_95_0 : Fin 32 → mxxMat83) (w_95_1 : Fin 32 → mxxMat83) (w_96_0 : Fin 128 → mxxMat86) (w_96_1 : Fin 128 → mxxMat86) (w_97_0 : Fin 32 → mxxMat86) (w_97_1 : Fin 32 → mxxMat86) (w_98_0 : Fin 128 → mxxMat89) (w_98_1 : Fin 128 → mxxMat89) (w_99_0 : Fin 32 → mxxMat89) (w_99_1 : Fin 32 → mxxMat89) (w_100_0 : Fin 128 → mxxMat91) (w_100_1 : Fin 128 → mxxMat91) (w_101_0 : Fin 32 → mxxMat91) (w_101_1 : Fin 32 → mxxMat91) (w_102_0 : Fin 128 → mxxMat94) (w_102_1 : Fin 128 → mxxMat94) (w_103_0 : Fin 32 → mxxMat94) (w_103_1 : Fin 32 → mxxMat94) (w_104_0 : Fin 128 → mxxMat97) (w_104_1 : Fin 128 → mxxMat97) (w_105_0 : Fin 32 → mxxMat97) (w_105_1 : Fin 32 → mxxMat97) (w_106_0 : Fin 128 → mxxMat100) (w_106_1 : Fin 128 → mxxMat100) (w_107_0 : Fin 32 → mxxMat100) (w_107_1 : Fin 32 → mxxMat100) (w_108_0 : Fin 128 → mxxMat103) (w_108_1 : Fin 128 → mxxMat103) (w_109_0 : Fin 32 → mxxMat103) (w_109_1 : Fin 32 → mxxMat103) (w_110_0 : Fin 128 → mxxMat106) (w_110_1 : Fin 128 → mxxMat106) (w_111_0 : Fin 32 → mxxMat106) (w_111_1 : Fin 32 → mxxMat106) (outputs : mxxMat2 × mxxMat2 × (Fin 1025 → mxxMat0) × (Fin 1025 → mxxMat0) × (Fin 1025 → mxxMat1) × (Fin 128 → mxxMat2) × (Fin 128 → mxxMat2) × (Fin 32 → mxxMat2) × (Fin 32 → mxxMat2) × (Fin 128 → mxxMat5) × (Fin 128 → mxxMat5) × (Fin 32 → mxxMat5) × (Fin 32 → mxxMat5) × (Fin 128 → mxxMat8) × (Fin 128 → mxxMat8) × (Fin 32 → mxxMat8) × (Fin 32 → mxxMat8) × (Fin 128 → mxxMat11) × (Fin 128 → mxxMat11) × (Fin 32 → mxxMat11) × (Fin 32 → mxxMat11) × (Fin 128 → mxxMat14) × (Fin 128 → mxxMat14) × (Fin 32 → mxxMat14) × (Fin 32 → mxxMat14) × (Fin 128 → mxxMat17) × (Fin 128 → mxxMat17) × (Fin 32 → mxxMat17) × (Fin 32 → mxxMat17) × (Fin 128 → mxxMat20) × (Fin 128 → mxxMat20) × (Fin 32 → mxxMat20) × (Fin 32 → mxxMat20) × (Fin 128 → mxxMat23) × (Fin 128 → mxxMat23) × (Fin 32 → mxxMat23) × (Fin 32 → mxxMat23) × (Fin 128 → mxxMat26) × (Fin 128 → mxxMat26) × (Fin 32 → mxxMat26) × (Fin 32 → mxxMat26) × (Fin 128 → mxxMat29) × (Fin 128 → mxxMat29) × (Fin 32 → mxxMat29) × (Fin 32 → mxxMat29) × (Fin 128 → mxxMat32) × (Fin 128 → mxxMat32) × (Fin 32 → mxxMat32) × (Fin 32 → mxxMat32) × (Fin 128 → mxxMat35) × (Fin 128 → mxxMat35) × (Fin 32 → mxxMat35) × (Fin 32 → mxxMat35) × (Fin 128 → mxxMat38) × (Fin 128 → mxxMat38) × (Fin 32 → mxxMat38) × (Fin 32 → mxxMat38) × (Fin 128 → mxxMat41) × (Fin 128 → mxxMat41) × (Fin 32 → mxxMat41) × (Fin 32 → mxxMat41) × (Fin 128 → mxxMat44) × (Fin 128 → mxxMat44) × (Fin 32 → mxxMat44) × (Fin 32 → mxxMat44) × (Fin 128 → mxxMat47) × (Fin 128 → mxxMat47) × (Fin 32 → mxxMat47) × (Fin 32 → mxxMat47) × (Fin 128 → mxxMat50) × (Fin 128 → mxxMat50) × (Fin 32 → mxxMat50) × (Fin 32 → mxxMat50) × (Fin 128 → mxxMat53) × (Fin 128 → mxxMat53) × (Fin 32 → mxxMat53) × (Fin 32 → mxxMat53) × (Fin 128 → mxxMat56) × (Fin 128 → mxxMat56) × (Fin 32 → mxxMat56) × (Fin 32 → mxxMat56) × (Fin 128 → mxxMat59) × (Fin 128 → mxxMat59) × (Fin 32 → mxxMat59) × (Fin 32 → mxxMat59) × (Fin 128 → mxxMat62) × (Fin 128 → mxxMat62) × (Fin 32 → mxxMat62) × (Fin 32 → mxxMat62) × (Fin 128 → mxxMat65) × (Fin 128 → mxxMat65) × (Fin 32 → mxxMat65) × (Fin 32 → mxxMat65) × (Fin 128 → mxxMat68) × (Fin 128 → mxxMat68) × (Fin 32 → mxxMat68) × (Fin 32 → mxxMat68) × (Fin 128 → mxxMat71) × (Fin 128 → mxxMat71) × (Fin 32 → mxxMat71) × (Fin 32 → mxxMat71) × (Fin 128 → mxxMat74) × (Fin 128 → mxxMat74) × (Fin 32 → mxxMat74) × (Fin 32 → mxxMat74) × (Fin 128 → mxxMat77) × (Fin 128 → mxxMat77) × (Fin 32 → mxxMat77) × (Fin 32 → mxxMat77) × (Fin 128 → mxxMat80) × (Fin 128 → mxxMat80) × (Fin 32 → mxxMat80) × (Fin 32 → mxxMat80) × (Fin 128 → mxxMat83) × (Fin 128 → mxxMat83) × (Fin 32 → mxxMat83) × (Fin 32 → mxxMat83) × (Fin 128 → mxxMat86) × (Fin 128 → mxxMat86) × (Fin 32 → mxxMat86) × (Fin 32 → mxxMat86) × (Fin 128 → mxxMat89) × (Fin 128 → mxxMat89) × (Fin 32 → mxxMat89) × (Fin 32 → mxxMat89) × (Fin 128 → mxxMat91) × (Fin 128 → mxxMat91) × (Fin 32 → mxxMat91) × (Fin 32 → mxxMat91) × (Fin 128 → mxxMat94) × (Fin 128 → mxxMat94) × (Fin 32 → mxxMat94) × (Fin 32 → mxxMat94) × (Fin 128 → mxxMat97) × (Fin 128 → mxxMat97) × (Fin 32 → mxxMat97) × (Fin 32 → mxxMat97) × (Fin 128 → mxxMat100) × (Fin 128 → mxxMat100) × (Fin 32 → mxxMat100) × (Fin 32 → mxxMat100) × (Fin 128 → mxxMat103) × (Fin 128 → mxxMat103) × (Fin 32 → mxxMat103) × (Fin 32 → mxxMat103) × (Fin 128 → mxxMat106) × (Fin 128 → mxxMat106) × (Fin 32 → mxxMat106) × (Fin 32 → mxxMat106) × Unit) : Prop :=
  (32) = 32 ∧
  (∀ i : Fin 32, parallel_parallel_generatedRoot_7_97 backend hashModel tape (path ++ [97, i.val]) params i_0 i (w_1_0, (w_2_0, (w_6_0, ()))) ((w_97_0 i), ((w_97_1 i), ()))) ∧
  (128) = 128 ∧
  (∀ i : Fin 128, parallel_parallel_generatedRoot_7_98 backend hashModel tape (path ++ [98, i.val]) params i_0 i (w_1_0, (w_2_0, (w_6_0, ()))) ((w_98_0 i), ((w_98_1 i), ()))) ∧
  (32) = 32 ∧
  (∀ i : Fin 32, parallel_parallel_generatedRoot_7_99 backend hashModel tape (path ++ [99, i.val]) params i_0 i (w_1_0, (w_2_0, (w_6_0, ()))) ((w_99_0 i), ((w_99_1 i), ()))) ∧
  (128) = 128 ∧
  (∀ i : Fin 128, parallel_parallel_generatedRoot_7_100 backend hashModel tape (path ++ [100, i.val]) params i_0 i (w_1_0, (w_2_0, (w_6_0, ()))) ((w_100_0 i), ((w_100_1 i), ()))) ∧
  (32) = 32 ∧
  (∀ i : Fin 32, parallel_parallel_generatedRoot_7_101 backend hashModel tape (path ++ [101, i.val]) params i_0 i (w_1_0, (w_2_0, (w_6_0, ()))) ((w_101_0 i), ((w_101_1 i), ()))) ∧
  (128) = 128 ∧
  (∀ i : Fin 128, parallel_parallel_generatedRoot_7_102 backend hashModel tape (path ++ [102, i.val]) params i_0 i (w_1_0, (w_2_0, (w_6_0, ()))) ((w_102_0 i), ((w_102_1 i), ()))) ∧
  (32) = 32 ∧
  (∀ i : Fin 32, parallel_parallel_generatedRoot_7_103 backend hashModel tape (path ++ [103, i.val]) params i_0 i (w_1_0, (w_2_0, (w_6_0, ()))) ((w_103_0 i), ((w_103_1 i), ()))) ∧
  (128) = 128 ∧
  (∀ i : Fin 128, parallel_parallel_generatedRoot_7_104 backend hashModel tape (path ++ [104, i.val]) params i_0 i (w_1_0, (w_2_0, (w_6_0, ()))) ((w_104_0 i), ((w_104_1 i), ()))) ∧
  (32) = 32 ∧
  (∀ i : Fin 32, parallel_parallel_generatedRoot_7_105 backend hashModel tape (path ++ [105, i.val]) params i_0 i (w_1_0, (w_2_0, (w_6_0, ()))) ((w_105_0 i), ((w_105_1 i), ()))) ∧
  (128) = 128 ∧
  (∀ i : Fin 128, parallel_parallel_generatedRoot_7_106 backend hashModel tape (path ++ [106, i.val]) params i_0 i (w_1_0, (w_2_0, (w_6_0, ()))) ((w_106_0 i), ((w_106_1 i), ()))) ∧
  (32) = 32 ∧
  (∀ i : Fin 32, parallel_parallel_generatedRoot_7_107 backend hashModel tape (path ++ [107, i.val]) params i_0 i (w_1_0, (w_2_0, (w_6_0, ()))) ((w_107_0 i), ((w_107_1 i), ()))) ∧
  (128) = 128 ∧
  (∀ i : Fin 128, parallel_parallel_generatedRoot_7_108 backend hashModel tape (path ++ [108, i.val]) params i_0 i (w_1_0, (w_2_0, (w_6_0, ()))) ((w_108_0 i), ((w_108_1 i), ()))) ∧
  (32) = 32 ∧
  (∀ i : Fin 32, parallel_parallel_generatedRoot_7_109 backend hashModel tape (path ++ [109, i.val]) params i_0 i (w_1_0, (w_2_0, (w_6_0, ()))) ((w_109_0 i), ((w_109_1 i), ()))) ∧
  (128) = 128 ∧
  (∀ i : Fin 128, parallel_parallel_generatedRoot_7_110 backend hashModel tape (path ++ [110, i.val]) params i_0 i (w_1_0, (w_2_0, (w_6_0, ()))) ((w_110_0 i), ((w_110_1 i), ()))) ∧
  (32) = 32 ∧
  (∀ i : Fin 32, parallel_parallel_generatedRoot_7_111 backend hashModel tape (path ++ [111, i.val]) params i_0 i (w_1_0, (w_2_0, (w_6_0, ()))) ((w_111_0 i), ((w_111_1 i), ()))) ∧
  outputs = (w_24_0, (w_36_0, (w_39_0, (w_39_1, (w_39_2, (w_40_0, (w_40_1, (w_41_0, (w_41_1, (w_42_0, (w_42_1, (w_43_0, (w_43_1, (w_44_0, (w_44_1, (w_45_0, (w_45_1, (w_46_0, (w_46_1, (w_47_0, (w_47_1, (w_48_0, (w_48_1, (w_49_0, (w_49_1, (w_50_0, (w_50_1, (w_51_0, (w_51_1, (w_52_0, (w_52_1, (w_53_0, (w_53_1, (w_54_0, (w_54_1, (w_55_0, (w_55_1, (w_56_0, (w_56_1, (w_57_0, (w_57_1, (w_58_0, (w_58_1, (w_59_0, (w_59_1, (w_60_0, (w_60_1, (w_61_0, (w_61_1, (w_62_0, (w_62_1, (w_63_0, (w_63_1, (w_64_0, (w_64_1, (w_65_0, (w_65_1, (w_66_0, (w_66_1, (w_67_0, (w_67_1, (w_68_0, (w_68_1, (w_69_0, (w_69_1, (w_70_0, (w_70_1, (w_71_0, (w_71_1, (w_72_0, (w_72_1, (w_73_0, (w_73_1, (w_74_0, (w_74_1, (w_75_0, (w_75_1, (w_76_0, (w_76_1, (w_77_0, (w_77_1, (w_78_0, (w_78_1, (w_79_0, (w_79_1, (w_80_0, (w_80_1, (w_81_0, (w_81_1, (w_82_0, (w_82_1, (w_83_0, (w_83_1, (w_84_0, (w_84_1, (w_85_0, (w_85_1, (w_86_0, (w_86_1, (w_87_0, (w_87_1, (w_88_0, (w_88_1, (w_89_0, (w_89_1, (w_90_0, (w_90_1, (w_91_0, (w_91_1, (w_92_0, (w_92_1, (w_93_0, (w_93_1, (w_94_0, (w_94_1, (w_95_0, (w_95_1, (w_96_0, (w_96_1, (w_97_0, (w_97_1, (w_98_0, (w_98_1, (w_99_0, (w_99_1, (w_100_0, (w_100_1, (w_101_0, (w_101_1, (w_102_0, (w_102_1, (w_103_0, (w_103_1, (w_104_0, (w_104_1, (w_105_0, (w_105_1, (w_106_0, (w_106_1, (w_107_0, (w_107_1, (w_108_0, (w_108_1, (w_109_0, (w_109_1, (w_110_0, (w_110_1, (w_111_0, (w_111_1, ())))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))

abbrev parallel_generatedRoot_7.constraints_3 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_2_0 : ByteArray) (w_1_0 : mxxMat1) (w_6_0 : mxxMat1) (w_81_0 : Fin 32 → mxxMat62) (w_81_1 : Fin 32 → mxxMat62) (w_82_0 : Fin 128 → mxxMat65) (w_82_1 : Fin 128 → mxxMat65) (w_83_0 : Fin 32 → mxxMat65) (w_83_1 : Fin 32 → mxxMat65) (w_84_0 : Fin 128 → mxxMat68) (w_84_1 : Fin 128 → mxxMat68) (w_85_0 : Fin 32 → mxxMat68) (w_85_1 : Fin 32 → mxxMat68) (w_86_0 : Fin 128 → mxxMat71) (w_86_1 : Fin 128 → mxxMat71) (w_87_0 : Fin 32 → mxxMat71) (w_87_1 : Fin 32 → mxxMat71) (w_88_0 : Fin 128 → mxxMat74) (w_88_1 : Fin 128 → mxxMat74) (w_89_0 : Fin 32 → mxxMat74) (w_89_1 : Fin 32 → mxxMat74) (w_90_0 : Fin 128 → mxxMat77) (w_90_1 : Fin 128 → mxxMat77) (w_91_0 : Fin 32 → mxxMat77) (w_91_1 : Fin 32 → mxxMat77) (w_92_0 : Fin 128 → mxxMat80) (w_92_1 : Fin 128 → mxxMat80) (w_93_0 : Fin 32 → mxxMat80) (w_93_1 : Fin 32 → mxxMat80) (w_94_0 : Fin 128 → mxxMat83) (w_94_1 : Fin 128 → mxxMat83) (w_95_0 : Fin 32 → mxxMat83) (w_95_1 : Fin 32 → mxxMat83) (w_96_0 : Fin 128 → mxxMat86) (w_96_1 : Fin 128 → mxxMat86) : Prop :=
  (32) = 32 ∧
  (∀ i : Fin 32, parallel_parallel_generatedRoot_7_81 backend hashModel tape (path ++ [81, i.val]) params i_0 i (w_1_0, (w_2_0, (w_6_0, ()))) ((w_81_0 i), ((w_81_1 i), ()))) ∧
  (128) = 128 ∧
  (∀ i : Fin 128, parallel_parallel_generatedRoot_7_82 backend hashModel tape (path ++ [82, i.val]) params i_0 i (w_1_0, (w_2_0, (w_6_0, ()))) ((w_82_0 i), ((w_82_1 i), ()))) ∧
  (32) = 32 ∧
  (∀ i : Fin 32, parallel_parallel_generatedRoot_7_83 backend hashModel tape (path ++ [83, i.val]) params i_0 i (w_1_0, (w_2_0, (w_6_0, ()))) ((w_83_0 i), ((w_83_1 i), ()))) ∧
  (128) = 128 ∧
  (∀ i : Fin 128, parallel_parallel_generatedRoot_7_84 backend hashModel tape (path ++ [84, i.val]) params i_0 i (w_1_0, (w_2_0, (w_6_0, ()))) ((w_84_0 i), ((w_84_1 i), ()))) ∧
  (32) = 32 ∧
  (∀ i : Fin 32, parallel_parallel_generatedRoot_7_85 backend hashModel tape (path ++ [85, i.val]) params i_0 i (w_1_0, (w_2_0, (w_6_0, ()))) ((w_85_0 i), ((w_85_1 i), ()))) ∧
  (128) = 128 ∧
  (∀ i : Fin 128, parallel_parallel_generatedRoot_7_86 backend hashModel tape (path ++ [86, i.val]) params i_0 i (w_1_0, (w_2_0, (w_6_0, ()))) ((w_86_0 i), ((w_86_1 i), ()))) ∧
  (32) = 32 ∧
  (∀ i : Fin 32, parallel_parallel_generatedRoot_7_87 backend hashModel tape (path ++ [87, i.val]) params i_0 i (w_1_0, (w_2_0, (w_6_0, ()))) ((w_87_0 i), ((w_87_1 i), ()))) ∧
  (128) = 128 ∧
  (∀ i : Fin 128, parallel_parallel_generatedRoot_7_88 backend hashModel tape (path ++ [88, i.val]) params i_0 i (w_1_0, (w_2_0, (w_6_0, ()))) ((w_88_0 i), ((w_88_1 i), ()))) ∧
  (32) = 32 ∧
  (∀ i : Fin 32, parallel_parallel_generatedRoot_7_89 backend hashModel tape (path ++ [89, i.val]) params i_0 i (w_1_0, (w_2_0, (w_6_0, ()))) ((w_89_0 i), ((w_89_1 i), ()))) ∧
  (128) = 128 ∧
  (∀ i : Fin 128, parallel_parallel_generatedRoot_7_90 backend hashModel tape (path ++ [90, i.val]) params i_0 i (w_1_0, (w_2_0, (w_6_0, ()))) ((w_90_0 i), ((w_90_1 i), ()))) ∧
  (32) = 32 ∧
  (∀ i : Fin 32, parallel_parallel_generatedRoot_7_91 backend hashModel tape (path ++ [91, i.val]) params i_0 i (w_1_0, (w_2_0, (w_6_0, ()))) ((w_91_0 i), ((w_91_1 i), ()))) ∧
  (128) = 128 ∧
  (∀ i : Fin 128, parallel_parallel_generatedRoot_7_92 backend hashModel tape (path ++ [92, i.val]) params i_0 i (w_1_0, (w_2_0, (w_6_0, ()))) ((w_92_0 i), ((w_92_1 i), ()))) ∧
  (32) = 32 ∧
  (∀ i : Fin 32, parallel_parallel_generatedRoot_7_93 backend hashModel tape (path ++ [93, i.val]) params i_0 i (w_1_0, (w_2_0, (w_6_0, ()))) ((w_93_0 i), ((w_93_1 i), ()))) ∧
  (128) = 128 ∧
  (∀ i : Fin 128, parallel_parallel_generatedRoot_7_94 backend hashModel tape (path ++ [94, i.val]) params i_0 i (w_1_0, (w_2_0, (w_6_0, ()))) ((w_94_0 i), ((w_94_1 i), ()))) ∧
  (32) = 32 ∧
  (∀ i : Fin 32, parallel_parallel_generatedRoot_7_95 backend hashModel tape (path ++ [95, i.val]) params i_0 i (w_1_0, (w_2_0, (w_6_0, ()))) ((w_95_0 i), ((w_95_1 i), ()))) ∧
  (128) = 128 ∧
  (∀ i : Fin 128, parallel_parallel_generatedRoot_7_96 backend hashModel tape (path ++ [96, i.val]) params i_0 i (w_1_0, (w_2_0, (w_6_0, ()))) ((w_96_0 i), ((w_96_1 i), ())))

abbrev parallel_generatedRoot_7.constraints_2 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_2_0 : ByteArray) (w_1_0 : mxxMat1) (w_6_0 : mxxMat1) (w_65_0 : Fin 32 → mxxMat38) (w_65_1 : Fin 32 → mxxMat38) (w_66_0 : Fin 128 → mxxMat41) (w_66_1 : Fin 128 → mxxMat41) (w_67_0 : Fin 32 → mxxMat41) (w_67_1 : Fin 32 → mxxMat41) (w_68_0 : Fin 128 → mxxMat44) (w_68_1 : Fin 128 → mxxMat44) (w_69_0 : Fin 32 → mxxMat44) (w_69_1 : Fin 32 → mxxMat44) (w_70_0 : Fin 128 → mxxMat47) (w_70_1 : Fin 128 → mxxMat47) (w_71_0 : Fin 32 → mxxMat47) (w_71_1 : Fin 32 → mxxMat47) (w_72_0 : Fin 128 → mxxMat50) (w_72_1 : Fin 128 → mxxMat50) (w_73_0 : Fin 32 → mxxMat50) (w_73_1 : Fin 32 → mxxMat50) (w_74_0 : Fin 128 → mxxMat53) (w_74_1 : Fin 128 → mxxMat53) (w_75_0 : Fin 32 → mxxMat53) (w_75_1 : Fin 32 → mxxMat53) (w_76_0 : Fin 128 → mxxMat56) (w_76_1 : Fin 128 → mxxMat56) (w_77_0 : Fin 32 → mxxMat56) (w_77_1 : Fin 32 → mxxMat56) (w_78_0 : Fin 128 → mxxMat59) (w_78_1 : Fin 128 → mxxMat59) (w_79_0 : Fin 32 → mxxMat59) (w_79_1 : Fin 32 → mxxMat59) (w_80_0 : Fin 128 → mxxMat62) (w_80_1 : Fin 128 → mxxMat62) : Prop :=
  (32) = 32 ∧
  (∀ i : Fin 32, parallel_parallel_generatedRoot_7_65 backend hashModel tape (path ++ [65, i.val]) params i_0 i (w_1_0, (w_2_0, (w_6_0, ()))) ((w_65_0 i), ((w_65_1 i), ()))) ∧
  (128) = 128 ∧
  (∀ i : Fin 128, parallel_parallel_generatedRoot_7_66 backend hashModel tape (path ++ [66, i.val]) params i_0 i (w_1_0, (w_2_0, (w_6_0, ()))) ((w_66_0 i), ((w_66_1 i), ()))) ∧
  (32) = 32 ∧
  (∀ i : Fin 32, parallel_parallel_generatedRoot_7_67 backend hashModel tape (path ++ [67, i.val]) params i_0 i (w_1_0, (w_2_0, (w_6_0, ()))) ((w_67_0 i), ((w_67_1 i), ()))) ∧
  (128) = 128 ∧
  (∀ i : Fin 128, parallel_parallel_generatedRoot_7_68 backend hashModel tape (path ++ [68, i.val]) params i_0 i (w_1_0, (w_2_0, (w_6_0, ()))) ((w_68_0 i), ((w_68_1 i), ()))) ∧
  (32) = 32 ∧
  (∀ i : Fin 32, parallel_parallel_generatedRoot_7_69 backend hashModel tape (path ++ [69, i.val]) params i_0 i (w_1_0, (w_2_0, (w_6_0, ()))) ((w_69_0 i), ((w_69_1 i), ()))) ∧
  (128) = 128 ∧
  (∀ i : Fin 128, parallel_parallel_generatedRoot_7_70 backend hashModel tape (path ++ [70, i.val]) params i_0 i (w_1_0, (w_2_0, (w_6_0, ()))) ((w_70_0 i), ((w_70_1 i), ()))) ∧
  (32) = 32 ∧
  (∀ i : Fin 32, parallel_parallel_generatedRoot_7_71 backend hashModel tape (path ++ [71, i.val]) params i_0 i (w_1_0, (w_2_0, (w_6_0, ()))) ((w_71_0 i), ((w_71_1 i), ()))) ∧
  (128) = 128 ∧
  (∀ i : Fin 128, parallel_parallel_generatedRoot_7_72 backend hashModel tape (path ++ [72, i.val]) params i_0 i (w_1_0, (w_2_0, (w_6_0, ()))) ((w_72_0 i), ((w_72_1 i), ()))) ∧
  (32) = 32 ∧
  (∀ i : Fin 32, parallel_parallel_generatedRoot_7_73 backend hashModel tape (path ++ [73, i.val]) params i_0 i (w_1_0, (w_2_0, (w_6_0, ()))) ((w_73_0 i), ((w_73_1 i), ()))) ∧
  (128) = 128 ∧
  (∀ i : Fin 128, parallel_parallel_generatedRoot_7_74 backend hashModel tape (path ++ [74, i.val]) params i_0 i (w_1_0, (w_2_0, (w_6_0, ()))) ((w_74_0 i), ((w_74_1 i), ()))) ∧
  (32) = 32 ∧
  (∀ i : Fin 32, parallel_parallel_generatedRoot_7_75 backend hashModel tape (path ++ [75, i.val]) params i_0 i (w_1_0, (w_2_0, (w_6_0, ()))) ((w_75_0 i), ((w_75_1 i), ()))) ∧
  (128) = 128 ∧
  (∀ i : Fin 128, parallel_parallel_generatedRoot_7_76 backend hashModel tape (path ++ [76, i.val]) params i_0 i (w_1_0, (w_2_0, (w_6_0, ()))) ((w_76_0 i), ((w_76_1 i), ()))) ∧
  (32) = 32 ∧
  (∀ i : Fin 32, parallel_parallel_generatedRoot_7_77 backend hashModel tape (path ++ [77, i.val]) params i_0 i (w_1_0, (w_2_0, (w_6_0, ()))) ((w_77_0 i), ((w_77_1 i), ()))) ∧
  (128) = 128 ∧
  (∀ i : Fin 128, parallel_parallel_generatedRoot_7_78 backend hashModel tape (path ++ [78, i.val]) params i_0 i (w_1_0, (w_2_0, (w_6_0, ()))) ((w_78_0 i), ((w_78_1 i), ()))) ∧
  (32) = 32 ∧
  (∀ i : Fin 32, parallel_parallel_generatedRoot_7_79 backend hashModel tape (path ++ [79, i.val]) params i_0 i (w_1_0, (w_2_0, (w_6_0, ()))) ((w_79_0 i), ((w_79_1 i), ()))) ∧
  (128) = 128 ∧
  (∀ i : Fin 128, parallel_parallel_generatedRoot_7_80 backend hashModel tape (path ++ [80, i.val]) params i_0 i (w_1_0, (w_2_0, (w_6_0, ()))) ((w_80_0 i), ((w_80_1 i), ())))

abbrev parallel_generatedRoot_7.constraints_1 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_2_0 : ByteArray) (w_1_0 : mxxMat1) (w_6_0 : mxxMat1) (w_49_0 : Fin 32 → mxxMat14) (w_49_1 : Fin 32 → mxxMat14) (w_50_0 : Fin 128 → mxxMat17) (w_50_1 : Fin 128 → mxxMat17) (w_51_0 : Fin 32 → mxxMat17) (w_51_1 : Fin 32 → mxxMat17) (w_52_0 : Fin 128 → mxxMat20) (w_52_1 : Fin 128 → mxxMat20) (w_53_0 : Fin 32 → mxxMat20) (w_53_1 : Fin 32 → mxxMat20) (w_54_0 : Fin 128 → mxxMat23) (w_54_1 : Fin 128 → mxxMat23) (w_55_0 : Fin 32 → mxxMat23) (w_55_1 : Fin 32 → mxxMat23) (w_56_0 : Fin 128 → mxxMat26) (w_56_1 : Fin 128 → mxxMat26) (w_57_0 : Fin 32 → mxxMat26) (w_57_1 : Fin 32 → mxxMat26) (w_58_0 : Fin 128 → mxxMat29) (w_58_1 : Fin 128 → mxxMat29) (w_59_0 : Fin 32 → mxxMat29) (w_59_1 : Fin 32 → mxxMat29) (w_60_0 : Fin 128 → mxxMat32) (w_60_1 : Fin 128 → mxxMat32) (w_61_0 : Fin 32 → mxxMat32) (w_61_1 : Fin 32 → mxxMat32) (w_62_0 : Fin 128 → mxxMat35) (w_62_1 : Fin 128 → mxxMat35) (w_63_0 : Fin 32 → mxxMat35) (w_63_1 : Fin 32 → mxxMat35) (w_64_0 : Fin 128 → mxxMat38) (w_64_1 : Fin 128 → mxxMat38) : Prop :=
  (32) = 32 ∧
  (∀ i : Fin 32, parallel_parallel_generatedRoot_7_49 backend hashModel tape (path ++ [49, i.val]) params i_0 i (w_1_0, (w_2_0, (w_6_0, ()))) ((w_49_0 i), ((w_49_1 i), ()))) ∧
  (128) = 128 ∧
  (∀ i : Fin 128, parallel_parallel_generatedRoot_7_50 backend hashModel tape (path ++ [50, i.val]) params i_0 i (w_1_0, (w_2_0, (w_6_0, ()))) ((w_50_0 i), ((w_50_1 i), ()))) ∧
  (32) = 32 ∧
  (∀ i : Fin 32, parallel_parallel_generatedRoot_7_51 backend hashModel tape (path ++ [51, i.val]) params i_0 i (w_1_0, (w_2_0, (w_6_0, ()))) ((w_51_0 i), ((w_51_1 i), ()))) ∧
  (128) = 128 ∧
  (∀ i : Fin 128, parallel_parallel_generatedRoot_7_52 backend hashModel tape (path ++ [52, i.val]) params i_0 i (w_1_0, (w_2_0, (w_6_0, ()))) ((w_52_0 i), ((w_52_1 i), ()))) ∧
  (32) = 32 ∧
  (∀ i : Fin 32, parallel_parallel_generatedRoot_7_53 backend hashModel tape (path ++ [53, i.val]) params i_0 i (w_1_0, (w_2_0, (w_6_0, ()))) ((w_53_0 i), ((w_53_1 i), ()))) ∧
  (128) = 128 ∧
  (∀ i : Fin 128, parallel_parallel_generatedRoot_7_54 backend hashModel tape (path ++ [54, i.val]) params i_0 i (w_1_0, (w_2_0, (w_6_0, ()))) ((w_54_0 i), ((w_54_1 i), ()))) ∧
  (32) = 32 ∧
  (∀ i : Fin 32, parallel_parallel_generatedRoot_7_55 backend hashModel tape (path ++ [55, i.val]) params i_0 i (w_1_0, (w_2_0, (w_6_0, ()))) ((w_55_0 i), ((w_55_1 i), ()))) ∧
  (128) = 128 ∧
  (∀ i : Fin 128, parallel_parallel_generatedRoot_7_56 backend hashModel tape (path ++ [56, i.val]) params i_0 i (w_1_0, (w_2_0, (w_6_0, ()))) ((w_56_0 i), ((w_56_1 i), ()))) ∧
  (32) = 32 ∧
  (∀ i : Fin 32, parallel_parallel_generatedRoot_7_57 backend hashModel tape (path ++ [57, i.val]) params i_0 i (w_1_0, (w_2_0, (w_6_0, ()))) ((w_57_0 i), ((w_57_1 i), ()))) ∧
  (128) = 128 ∧
  (∀ i : Fin 128, parallel_parallel_generatedRoot_7_58 backend hashModel tape (path ++ [58, i.val]) params i_0 i (w_1_0, (w_2_0, (w_6_0, ()))) ((w_58_0 i), ((w_58_1 i), ()))) ∧
  (32) = 32 ∧
  (∀ i : Fin 32, parallel_parallel_generatedRoot_7_59 backend hashModel tape (path ++ [59, i.val]) params i_0 i (w_1_0, (w_2_0, (w_6_0, ()))) ((w_59_0 i), ((w_59_1 i), ()))) ∧
  (128) = 128 ∧
  (∀ i : Fin 128, parallel_parallel_generatedRoot_7_60 backend hashModel tape (path ++ [60, i.val]) params i_0 i (w_1_0, (w_2_0, (w_6_0, ()))) ((w_60_0 i), ((w_60_1 i), ()))) ∧
  (32) = 32 ∧
  (∀ i : Fin 32, parallel_parallel_generatedRoot_7_61 backend hashModel tape (path ++ [61, i.val]) params i_0 i (w_1_0, (w_2_0, (w_6_0, ()))) ((w_61_0 i), ((w_61_1 i), ()))) ∧
  (128) = 128 ∧
  (∀ i : Fin 128, parallel_parallel_generatedRoot_7_62 backend hashModel tape (path ++ [62, i.val]) params i_0 i (w_1_0, (w_2_0, (w_6_0, ()))) ((w_62_0 i), ((w_62_1 i), ()))) ∧
  (32) = 32 ∧
  (∀ i : Fin 32, parallel_parallel_generatedRoot_7_63 backend hashModel tape (path ++ [63, i.val]) params i_0 i (w_1_0, (w_2_0, (w_6_0, ()))) ((w_63_0 i), ((w_63_1 i), ()))) ∧
  (128) = 128 ∧
  (∀ i : Fin 128, parallel_parallel_generatedRoot_7_64 backend hashModel tape (path ++ [64, i.val]) params i_0 i (w_1_0, (w_2_0, (w_6_0, ()))) ((w_64_0 i), ((w_64_1 i), ())))

abbrev parallel_generatedRoot_7.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_0_0 : mxxMat88) (w_2_0 : ByteArray) (w_10_0 : mxxMat88) (w_37_0 : Fin 1025 → Int) (w_38_0 : Fin 1025 → Int) (w_1_0 : mxxMat1) (w_3_0 : mxxMat2) (w_5_0 : mxxMat88) (w_6_0 : mxxMat1) (w_7_0 : mxxMat2) (w_11_0 : mxxMat1) (w_12_0 : mxxMat3) (w_13_0 : mxxMat2) (w_23_0 : mxxMat2) (w_25_0 : mxxMat2) (w_27_0 : mxxMat2) (w_30_0 : mxxMat88) (w_31_0 : mxxMat1) (w_35_0 : mxxMat2) (w_39_0 : Fin 1025 → mxxMat0) (w_39_1 : Fin 1025 → mxxMat0) (w_39_2 : Fin 1025 → mxxMat1) (w_40_0 : Fin 128 → mxxMat2) (w_40_1 : Fin 128 → mxxMat2) (w_41_0 : Fin 32 → mxxMat2) (w_41_1 : Fin 32 → mxxMat2) (w_42_0 : Fin 128 → mxxMat5) (w_42_1 : Fin 128 → mxxMat5) (w_43_0 : Fin 32 → mxxMat5) (w_43_1 : Fin 32 → mxxMat5) (w_44_0 : Fin 128 → mxxMat8) (w_44_1 : Fin 128 → mxxMat8) (w_45_0 : Fin 32 → mxxMat8) (w_45_1 : Fin 32 → mxxMat8) (w_46_0 : Fin 128 → mxxMat11) (w_46_1 : Fin 128 → mxxMat11) (w_47_0 : Fin 32 → mxxMat11) (w_47_1 : Fin 32 → mxxMat11) (w_48_0 : Fin 128 → mxxMat14) (w_48_1 : Fin 128 → mxxMat14) : Prop :=
  MxxRuntime.centeredRebaseRuns w_0_0 w_1_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 105, 110, 112, 117, 116, 47, 87, 105, 114, 101, 73, 100, 40, 48, 41, 47, 48]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_2_0) w_3_0 ∧
  MxxRuntime.centeredRebaseRuns w_5_0 w_6_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 105, 110, 112, 117, 116, 47, 87, 105, 114, 101, 73, 100, 40, 48, 41, 47, 48]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_2_0) w_7_0 ∧
  MxxRuntime.centeredRebaseRuns w_10_0 w_11_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 1024 168 w_12_0 ∧
  MxxRuntime.centeredRebaseRuns w_12_0 w_13_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [23]) ((4/1)) (26) w_23_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 105, 110, 112, 117, 116, 47, 87, 105, 114, 101, 73, 100, 40, 48, 41, 47, 48]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_2_0) w_25_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 105, 110, 112, 117, 116, 47, 87, 105, 114, 101, 73, 100, 40, 48, 41, 47, 48]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_2_0) w_27_0 ∧
  MxxRuntime.centeredRebaseRuns w_30_0 w_31_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [35]) ((4/1)) (26) w_35_0 ∧
  (1025) = 1025 ∧
  (∀ i : Fin 1025, parallel_parallel_generatedRoot_7_39 backend hashModel tape (path ++ [39, i.val]) params i_0 i (w_1_0, (w_2_0, (w_6_0, ((w_37_0 i), ((w_38_0 i), ()))))) ((w_39_0 i), ((w_39_1 i), ((w_39_2 i), ())))) ∧
  (128) = 128 ∧
  (∀ i : Fin 128, parallel_parallel_generatedRoot_7_40 backend hashModel tape (path ++ [40, i.val]) params i_0 i (w_1_0, (w_2_0, (w_6_0, ()))) ((w_40_0 i), ((w_40_1 i), ()))) ∧
  (32) = 32 ∧
  (∀ i : Fin 32, parallel_parallel_generatedRoot_7_41 backend hashModel tape (path ++ [41, i.val]) params i_0 i (w_1_0, (w_2_0, (w_6_0, ()))) ((w_41_0 i), ((w_41_1 i), ()))) ∧
  (128) = 128 ∧
  (∀ i : Fin 128, parallel_parallel_generatedRoot_7_42 backend hashModel tape (path ++ [42, i.val]) params i_0 i (w_1_0, (w_2_0, (w_6_0, ()))) ((w_42_0 i), ((w_42_1 i), ()))) ∧
  (32) = 32 ∧
  (∀ i : Fin 32, parallel_parallel_generatedRoot_7_43 backend hashModel tape (path ++ [43, i.val]) params i_0 i (w_1_0, (w_2_0, (w_6_0, ()))) ((w_43_0 i), ((w_43_1 i), ()))) ∧
  (128) = 128 ∧
  (∀ i : Fin 128, parallel_parallel_generatedRoot_7_44 backend hashModel tape (path ++ [44, i.val]) params i_0 i (w_1_0, (w_2_0, (w_6_0, ()))) ((w_44_0 i), ((w_44_1 i), ()))) ∧
  (32) = 32 ∧
  (∀ i : Fin 32, parallel_parallel_generatedRoot_7_45 backend hashModel tape (path ++ [45, i.val]) params i_0 i (w_1_0, (w_2_0, (w_6_0, ()))) ((w_45_0 i), ((w_45_1 i), ()))) ∧
  (128) = 128 ∧
  (∀ i : Fin 128, parallel_parallel_generatedRoot_7_46 backend hashModel tape (path ++ [46, i.val]) params i_0 i (w_1_0, (w_2_0, (w_6_0, ()))) ((w_46_0 i), ((w_46_1 i), ()))) ∧
  (32) = 32 ∧
  (∀ i : Fin 32, parallel_parallel_generatedRoot_7_47 backend hashModel tape (path ++ [47, i.val]) params i_0 i (w_1_0, (w_2_0, (w_6_0, ()))) ((w_47_0 i), ((w_47_1 i), ()))) ∧
  (128) = 128 ∧
  (∀ i : Fin 128, parallel_parallel_generatedRoot_7_48 backend hashModel tape (path ++ [48, i.val]) params i_0 i (w_1_0, (w_2_0, (w_6_0, ()))) ((w_48_0 i), ((w_48_1 i), ())))

abbrev parallel_generatedRoot_7.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat88 × ByteArray × mxxMat88 × Int × Int × (Fin 1025 → Int) × (Fin 1025 → Int) × Unit) (outputs : mxxMat2 × mxxMat2 × (Fin 1025 → mxxMat0) × (Fin 1025 → mxxMat0) × (Fin 1025 → mxxMat1) × (Fin 128 → mxxMat2) × (Fin 128 → mxxMat2) × (Fin 32 → mxxMat2) × (Fin 32 → mxxMat2) × (Fin 128 → mxxMat5) × (Fin 128 → mxxMat5) × (Fin 32 → mxxMat5) × (Fin 32 → mxxMat5) × (Fin 128 → mxxMat8) × (Fin 128 → mxxMat8) × (Fin 32 → mxxMat8) × (Fin 32 → mxxMat8) × (Fin 128 → mxxMat11) × (Fin 128 → mxxMat11) × (Fin 32 → mxxMat11) × (Fin 32 → mxxMat11) × (Fin 128 → mxxMat14) × (Fin 128 → mxxMat14) × (Fin 32 → mxxMat14) × (Fin 32 → mxxMat14) × (Fin 128 → mxxMat17) × (Fin 128 → mxxMat17) × (Fin 32 → mxxMat17) × (Fin 32 → mxxMat17) × (Fin 128 → mxxMat20) × (Fin 128 → mxxMat20) × (Fin 32 → mxxMat20) × (Fin 32 → mxxMat20) × (Fin 128 → mxxMat23) × (Fin 128 → mxxMat23) × (Fin 32 → mxxMat23) × (Fin 32 → mxxMat23) × (Fin 128 → mxxMat26) × (Fin 128 → mxxMat26) × (Fin 32 → mxxMat26) × (Fin 32 → mxxMat26) × (Fin 128 → mxxMat29) × (Fin 128 → mxxMat29) × (Fin 32 → mxxMat29) × (Fin 32 → mxxMat29) × (Fin 128 → mxxMat32) × (Fin 128 → mxxMat32) × (Fin 32 → mxxMat32) × (Fin 32 → mxxMat32) × (Fin 128 → mxxMat35) × (Fin 128 → mxxMat35) × (Fin 32 → mxxMat35) × (Fin 32 → mxxMat35) × (Fin 128 → mxxMat38) × (Fin 128 → mxxMat38) × (Fin 32 → mxxMat38) × (Fin 32 → mxxMat38) × (Fin 128 → mxxMat41) × (Fin 128 → mxxMat41) × (Fin 32 → mxxMat41) × (Fin 32 → mxxMat41) × (Fin 128 → mxxMat44) × (Fin 128 → mxxMat44) × (Fin 32 → mxxMat44) × (Fin 32 → mxxMat44) × (Fin 128 → mxxMat47) × (Fin 128 → mxxMat47) × (Fin 32 → mxxMat47) × (Fin 32 → mxxMat47) × (Fin 128 → mxxMat50) × (Fin 128 → mxxMat50) × (Fin 32 → mxxMat50) × (Fin 32 → mxxMat50) × (Fin 128 → mxxMat53) × (Fin 128 → mxxMat53) × (Fin 32 → mxxMat53) × (Fin 32 → mxxMat53) × (Fin 128 → mxxMat56) × (Fin 128 → mxxMat56) × (Fin 32 → mxxMat56) × (Fin 32 → mxxMat56) × (Fin 128 → mxxMat59) × (Fin 128 → mxxMat59) × (Fin 32 → mxxMat59) × (Fin 32 → mxxMat59) × (Fin 128 → mxxMat62) × (Fin 128 → mxxMat62) × (Fin 32 → mxxMat62) × (Fin 32 → mxxMat62) × (Fin 128 → mxxMat65) × (Fin 128 → mxxMat65) × (Fin 32 → mxxMat65) × (Fin 32 → mxxMat65) × (Fin 128 → mxxMat68) × (Fin 128 → mxxMat68) × (Fin 32 → mxxMat68) × (Fin 32 → mxxMat68) × (Fin 128 → mxxMat71) × (Fin 128 → mxxMat71) × (Fin 32 → mxxMat71) × (Fin 32 → mxxMat71) × (Fin 128 → mxxMat74) × (Fin 128 → mxxMat74) × (Fin 32 → mxxMat74) × (Fin 32 → mxxMat74) × (Fin 128 → mxxMat77) × (Fin 128 → mxxMat77) × (Fin 32 → mxxMat77) × (Fin 32 → mxxMat77) × (Fin 128 → mxxMat80) × (Fin 128 → mxxMat80) × (Fin 32 → mxxMat80) × (Fin 32 → mxxMat80) × (Fin 128 → mxxMat83) × (Fin 128 → mxxMat83) × (Fin 32 → mxxMat83) × (Fin 32 → mxxMat83) × (Fin 128 → mxxMat86) × (Fin 128 → mxxMat86) × (Fin 32 → mxxMat86) × (Fin 32 → mxxMat86) × (Fin 128 → mxxMat89) × (Fin 128 → mxxMat89) × (Fin 32 → mxxMat89) × (Fin 32 → mxxMat89) × (Fin 128 → mxxMat91) × (Fin 128 → mxxMat91) × (Fin 32 → mxxMat91) × (Fin 32 → mxxMat91) × (Fin 128 → mxxMat94) × (Fin 128 → mxxMat94) × (Fin 32 → mxxMat94) × (Fin 32 → mxxMat94) × (Fin 128 → mxxMat97) × (Fin 128 → mxxMat97) × (Fin 32 → mxxMat97) × (Fin 32 → mxxMat97) × (Fin 128 → mxxMat100) × (Fin 128 → mxxMat100) × (Fin 32 → mxxMat100) × (Fin 32 → mxxMat100) × (Fin 128 → mxxMat103) × (Fin 128 → mxxMat103) × (Fin 32 → mxxMat103) × (Fin 32 → mxxMat103) × (Fin 128 → mxxMat106) × (Fin 128 → mxxMat106) × (Fin 32 → mxxMat106) × (Fin 32 → mxxMat106) × Unit) (witness : parallel_generatedRoot_7.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat88 := inputs.1
  let w_2_0 : ByteArray := inputs.2.1
  let w_10_0 : mxxMat88 := inputs.2.2.1
  let w_16_0 : Int := inputs.2.2.2.1
  let w_18_0 : Int := inputs.2.2.2.2.1
  let w_37_0 : Fin 1025 → Int := inputs.2.2.2.2.2.1
  let w_38_0 : Fin 1025 → Int := inputs.2.2.2.2.2.2.1
  let w_1_0 := witness.w_1_0
  let w_3_0 := witness.w_3_0
  let w_4_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_1_0 w_3_0
  let w_5_0 : mxxMat88 := (1 : mxxMat88)
  let w_6_0 := witness.w_6_0
  let w_7_0 := witness.w_7_0
  let w_8_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_6_0 w_7_0
  let w_9_0 : mxxMat2 := MxxRuntime.matrixAdd w_4_0 w_8_0
  let w_11_0 := witness.w_11_0
  let w_12_0 := witness.w_12_0
  let w_13_0 := witness.w_13_0
  let w_14_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_11_0 w_13_0
  let w_15_0 : mxxMat1 := (MxxRuntime.matrixPolynomial [(1099380817921)] : mxxMat1)
  let w_17_0 : mxxMat1 := MxxRuntime.liftInteger w_16_0
  let w_19_0 : mxxMat1 := MxxRuntime.multiplyMonomial w_17_0 w_18_0
  let w_20_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_15_0 w_19_0
  let w_21_0 : mxxMat2 := MxxRuntime.matrixMulScalarRight w_14_0 w_20_0
  let w_22_0 : mxxMat2 := MxxRuntime.matrixSub w_9_0 w_21_0
  let w_23_0 := witness.w_23_0
  let w_24_0 : mxxMat2 := MxxRuntime.matrixAdd w_22_0 w_23_0
  let w_25_0 := witness.w_25_0
  let w_26_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_1_0 w_25_0
  let w_27_0 := witness.w_27_0
  let w_28_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_6_0 w_27_0
  let w_29_0 : mxxMat2 := MxxRuntime.matrixAdd w_26_0 w_28_0
  let w_30_0 : mxxMat88 := (1 : mxxMat88)
  let w_31_0 := witness.w_31_0
  let w_32_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_31_0 w_13_0
  let w_33_0 : mxxMat2 := MxxRuntime.matrixMulScalarRight w_32_0 w_20_0
  let w_34_0 : mxxMat2 := MxxRuntime.matrixSub w_29_0 w_33_0
  let w_35_0 := witness.w_35_0
  let w_36_0 : mxxMat2 := MxxRuntime.matrixAdd w_34_0 w_35_0
  let w_39_0 := witness.w_39_0
  let w_39_1 := witness.w_39_1
  let w_39_2 := witness.w_39_2
  let w_40_0 := witness.w_40_0
  let w_40_1 := witness.w_40_1
  let w_41_0 := witness.w_41_0
  let w_41_1 := witness.w_41_1
  let w_42_0 := witness.w_42_0
  let w_42_1 := witness.w_42_1
  let w_43_0 := witness.w_43_0
  let w_43_1 := witness.w_43_1
  let w_44_0 := witness.w_44_0
  let w_44_1 := witness.w_44_1
  let w_45_0 := witness.w_45_0
  let w_45_1 := witness.w_45_1
  let w_46_0 := witness.w_46_0
  let w_46_1 := witness.w_46_1
  let w_47_0 := witness.w_47_0
  let w_47_1 := witness.w_47_1
  let w_48_0 := witness.w_48_0
  let w_48_1 := witness.w_48_1
  let w_49_0 := witness.w_49_0
  let w_49_1 := witness.w_49_1
  let w_50_0 := witness.w_50_0
  let w_50_1 := witness.w_50_1
  let w_51_0 := witness.w_51_0
  let w_51_1 := witness.w_51_1
  let w_52_0 := witness.w_52_0
  let w_52_1 := witness.w_52_1
  let w_53_0 := witness.w_53_0
  let w_53_1 := witness.w_53_1
  let w_54_0 := witness.w_54_0
  let w_54_1 := witness.w_54_1
  let w_55_0 := witness.w_55_0
  let w_55_1 := witness.w_55_1
  let w_56_0 := witness.w_56_0
  let w_56_1 := witness.w_56_1
  let w_57_0 := witness.w_57_0
  let w_57_1 := witness.w_57_1
  let w_58_0 := witness.w_58_0
  let w_58_1 := witness.w_58_1
  let w_59_0 := witness.w_59_0
  let w_59_1 := witness.w_59_1
  let w_60_0 := witness.w_60_0
  let w_60_1 := witness.w_60_1
  let w_61_0 := witness.w_61_0
  let w_61_1 := witness.w_61_1
  let w_62_0 := witness.w_62_0
  let w_62_1 := witness.w_62_1
  let w_63_0 := witness.w_63_0
  let w_63_1 := witness.w_63_1
  let w_64_0 := witness.w_64_0
  let w_64_1 := witness.w_64_1
  let w_65_0 := witness.w_65_0
  let w_65_1 := witness.w_65_1
  let w_66_0 := witness.w_66_0
  let w_66_1 := witness.w_66_1
  let w_67_0 := witness.w_67_0
  let w_67_1 := witness.w_67_1
  let w_68_0 := witness.w_68_0
  let w_68_1 := witness.w_68_1
  let w_69_0 := witness.w_69_0
  let w_69_1 := witness.w_69_1
  let w_70_0 := witness.w_70_0
  let w_70_1 := witness.w_70_1
  let w_71_0 := witness.w_71_0
  let w_71_1 := witness.w_71_1
  let w_72_0 := witness.w_72_0
  let w_72_1 := witness.w_72_1
  let w_73_0 := witness.w_73_0
  let w_73_1 := witness.w_73_1
  let w_74_0 := witness.w_74_0
  let w_74_1 := witness.w_74_1
  let w_75_0 := witness.w_75_0
  let w_75_1 := witness.w_75_1
  let w_76_0 := witness.w_76_0
  let w_76_1 := witness.w_76_1
  let w_77_0 := witness.w_77_0
  let w_77_1 := witness.w_77_1
  let w_78_0 := witness.w_78_0
  let w_78_1 := witness.w_78_1
  let w_79_0 := witness.w_79_0
  let w_79_1 := witness.w_79_1
  let w_80_0 := witness.w_80_0
  let w_80_1 := witness.w_80_1
  let w_81_0 := witness.w_81_0
  let w_81_1 := witness.w_81_1
  let w_82_0 := witness.w_82_0
  let w_82_1 := witness.w_82_1
  let w_83_0 := witness.w_83_0
  let w_83_1 := witness.w_83_1
  let w_84_0 := witness.w_84_0
  let w_84_1 := witness.w_84_1
  let w_85_0 := witness.w_85_0
  let w_85_1 := witness.w_85_1
  let w_86_0 := witness.w_86_0
  let w_86_1 := witness.w_86_1
  let w_87_0 := witness.w_87_0
  let w_87_1 := witness.w_87_1
  let w_88_0 := witness.w_88_0
  let w_88_1 := witness.w_88_1
  let w_89_0 := witness.w_89_0
  let w_89_1 := witness.w_89_1
  let w_90_0 := witness.w_90_0
  let w_90_1 := witness.w_90_1
  let w_91_0 := witness.w_91_0
  let w_91_1 := witness.w_91_1
  let w_92_0 := witness.w_92_0
  let w_92_1 := witness.w_92_1
  let w_93_0 := witness.w_93_0
  let w_93_1 := witness.w_93_1
  let w_94_0 := witness.w_94_0
  let w_94_1 := witness.w_94_1
  let w_95_0 := witness.w_95_0
  let w_95_1 := witness.w_95_1
  let w_96_0 := witness.w_96_0
  let w_96_1 := witness.w_96_1
  let w_97_0 := witness.w_97_0
  let w_97_1 := witness.w_97_1
  let w_98_0 := witness.w_98_0
  let w_98_1 := witness.w_98_1
  let w_99_0 := witness.w_99_0
  let w_99_1 := witness.w_99_1
  let w_100_0 := witness.w_100_0
  let w_100_1 := witness.w_100_1
  let w_101_0 := witness.w_101_0
  let w_101_1 := witness.w_101_1
  let w_102_0 := witness.w_102_0
  let w_102_1 := witness.w_102_1
  let w_103_0 := witness.w_103_0
  let w_103_1 := witness.w_103_1
  let w_104_0 := witness.w_104_0
  let w_104_1 := witness.w_104_1
  let w_105_0 := witness.w_105_0
  let w_105_1 := witness.w_105_1
  let w_106_0 := witness.w_106_0
  let w_106_1 := witness.w_106_1
  let w_107_0 := witness.w_107_0
  let w_107_1 := witness.w_107_1
  let w_108_0 := witness.w_108_0
  let w_108_1 := witness.w_108_1
  let w_109_0 := witness.w_109_0
  let w_109_1 := witness.w_109_1
  let w_110_0 := witness.w_110_0
  let w_110_1 := witness.w_110_1
  let w_111_0 := witness.w_111_0
  let w_111_1 := witness.w_111_1
  parallel_generatedRoot_7.constraints_0 backend hashModel tape path params i_0 w_0_0 w_2_0 w_10_0 w_37_0 w_38_0 w_1_0 w_3_0 w_5_0 w_6_0 w_7_0 w_11_0 w_12_0 w_13_0 w_23_0 w_25_0 w_27_0 w_30_0 w_31_0 w_35_0 w_39_0 w_39_1 w_39_2 w_40_0 w_40_1 w_41_0 w_41_1 w_42_0 w_42_1 w_43_0 w_43_1 w_44_0 w_44_1 w_45_0 w_45_1 w_46_0 w_46_1 w_47_0 w_47_1 w_48_0 w_48_1 ∧
    parallel_generatedRoot_7.constraints_1 backend hashModel tape path params i_0 w_2_0 w_1_0 w_6_0 w_49_0 w_49_1 w_50_0 w_50_1 w_51_0 w_51_1 w_52_0 w_52_1 w_53_0 w_53_1 w_54_0 w_54_1 w_55_0 w_55_1 w_56_0 w_56_1 w_57_0 w_57_1 w_58_0 w_58_1 w_59_0 w_59_1 w_60_0 w_60_1 w_61_0 w_61_1 w_62_0 w_62_1 w_63_0 w_63_1 w_64_0 w_64_1 ∧
    parallel_generatedRoot_7.constraints_2 backend hashModel tape path params i_0 w_2_0 w_1_0 w_6_0 w_65_0 w_65_1 w_66_0 w_66_1 w_67_0 w_67_1 w_68_0 w_68_1 w_69_0 w_69_1 w_70_0 w_70_1 w_71_0 w_71_1 w_72_0 w_72_1 w_73_0 w_73_1 w_74_0 w_74_1 w_75_0 w_75_1 w_76_0 w_76_1 w_77_0 w_77_1 w_78_0 w_78_1 w_79_0 w_79_1 w_80_0 w_80_1 ∧
    parallel_generatedRoot_7.constraints_3 backend hashModel tape path params i_0 w_2_0 w_1_0 w_6_0 w_81_0 w_81_1 w_82_0 w_82_1 w_83_0 w_83_1 w_84_0 w_84_1 w_85_0 w_85_1 w_86_0 w_86_1 w_87_0 w_87_1 w_88_0 w_88_1 w_89_0 w_89_1 w_90_0 w_90_1 w_91_0 w_91_1 w_92_0 w_92_1 w_93_0 w_93_1 w_94_0 w_94_1 w_95_0 w_95_1 w_96_0 w_96_1 ∧
    parallel_generatedRoot_7.constraints_4 backend hashModel tape path params i_0 w_2_0 w_1_0 w_6_0 w_24_0 w_36_0 w_39_0 w_39_1 w_39_2 w_40_0 w_40_1 w_41_0 w_41_1 w_42_0 w_42_1 w_43_0 w_43_1 w_44_0 w_44_1 w_45_0 w_45_1 w_46_0 w_46_1 w_47_0 w_47_1 w_48_0 w_48_1 w_49_0 w_49_1 w_50_0 w_50_1 w_51_0 w_51_1 w_52_0 w_52_1 w_53_0 w_53_1 w_54_0 w_54_1 w_55_0 w_55_1 w_56_0 w_56_1 w_57_0 w_57_1 w_58_0 w_58_1 w_59_0 w_59_1 w_60_0 w_60_1 w_61_0 w_61_1 w_62_0 w_62_1 w_63_0 w_63_1 w_64_0 w_64_1 w_65_0 w_65_1 w_66_0 w_66_1 w_67_0 w_67_1 w_68_0 w_68_1 w_69_0 w_69_1 w_70_0 w_70_1 w_71_0 w_71_1 w_72_0 w_72_1 w_73_0 w_73_1 w_74_0 w_74_1 w_75_0 w_75_1 w_76_0 w_76_1 w_77_0 w_77_1 w_78_0 w_78_1 w_79_0 w_79_1 w_80_0 w_80_1 w_81_0 w_81_1 w_82_0 w_82_1 w_83_0 w_83_1 w_84_0 w_84_1 w_85_0 w_85_1 w_86_0 w_86_1 w_87_0 w_87_1 w_88_0 w_88_1 w_89_0 w_89_1 w_90_0 w_90_1 w_91_0 w_91_1 w_92_0 w_92_1 w_93_0 w_93_1 w_94_0 w_94_1 w_95_0 w_95_1 w_96_0 w_96_1 w_97_0 w_97_1 w_98_0 w_98_1 w_99_0 w_99_1 w_100_0 w_100_1 w_101_0 w_101_1 w_102_0 w_102_1 w_103_0 w_103_1 w_104_0 w_104_1 w_105_0 w_105_1 w_106_0 w_106_1 w_107_0 w_107_1 w_108_0 w_108_1 w_109_0 w_109_1 w_110_0 w_110_1 w_111_0 w_111_1 outputs

def parallel_generatedRoot_7 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat88 × ByteArray × mxxMat88 × Int × Int × (Fin 1025 → Int) × (Fin 1025 → Int) × Unit) (outputs : mxxMat2 × mxxMat2 × (Fin 1025 → mxxMat0) × (Fin 1025 → mxxMat0) × (Fin 1025 → mxxMat1) × (Fin 128 → mxxMat2) × (Fin 128 → mxxMat2) × (Fin 32 → mxxMat2) × (Fin 32 → mxxMat2) × (Fin 128 → mxxMat5) × (Fin 128 → mxxMat5) × (Fin 32 → mxxMat5) × (Fin 32 → mxxMat5) × (Fin 128 → mxxMat8) × (Fin 128 → mxxMat8) × (Fin 32 → mxxMat8) × (Fin 32 → mxxMat8) × (Fin 128 → mxxMat11) × (Fin 128 → mxxMat11) × (Fin 32 → mxxMat11) × (Fin 32 → mxxMat11) × (Fin 128 → mxxMat14) × (Fin 128 → mxxMat14) × (Fin 32 → mxxMat14) × (Fin 32 → mxxMat14) × (Fin 128 → mxxMat17) × (Fin 128 → mxxMat17) × (Fin 32 → mxxMat17) × (Fin 32 → mxxMat17) × (Fin 128 → mxxMat20) × (Fin 128 → mxxMat20) × (Fin 32 → mxxMat20) × (Fin 32 → mxxMat20) × (Fin 128 → mxxMat23) × (Fin 128 → mxxMat23) × (Fin 32 → mxxMat23) × (Fin 32 → mxxMat23) × (Fin 128 → mxxMat26) × (Fin 128 → mxxMat26) × (Fin 32 → mxxMat26) × (Fin 32 → mxxMat26) × (Fin 128 → mxxMat29) × (Fin 128 → mxxMat29) × (Fin 32 → mxxMat29) × (Fin 32 → mxxMat29) × (Fin 128 → mxxMat32) × (Fin 128 → mxxMat32) × (Fin 32 → mxxMat32) × (Fin 32 → mxxMat32) × (Fin 128 → mxxMat35) × (Fin 128 → mxxMat35) × (Fin 32 → mxxMat35) × (Fin 32 → mxxMat35) × (Fin 128 → mxxMat38) × (Fin 128 → mxxMat38) × (Fin 32 → mxxMat38) × (Fin 32 → mxxMat38) × (Fin 128 → mxxMat41) × (Fin 128 → mxxMat41) × (Fin 32 → mxxMat41) × (Fin 32 → mxxMat41) × (Fin 128 → mxxMat44) × (Fin 128 → mxxMat44) × (Fin 32 → mxxMat44) × (Fin 32 → mxxMat44) × (Fin 128 → mxxMat47) × (Fin 128 → mxxMat47) × (Fin 32 → mxxMat47) × (Fin 32 → mxxMat47) × (Fin 128 → mxxMat50) × (Fin 128 → mxxMat50) × (Fin 32 → mxxMat50) × (Fin 32 → mxxMat50) × (Fin 128 → mxxMat53) × (Fin 128 → mxxMat53) × (Fin 32 → mxxMat53) × (Fin 32 → mxxMat53) × (Fin 128 → mxxMat56) × (Fin 128 → mxxMat56) × (Fin 32 → mxxMat56) × (Fin 32 → mxxMat56) × (Fin 128 → mxxMat59) × (Fin 128 → mxxMat59) × (Fin 32 → mxxMat59) × (Fin 32 → mxxMat59) × (Fin 128 → mxxMat62) × (Fin 128 → mxxMat62) × (Fin 32 → mxxMat62) × (Fin 32 → mxxMat62) × (Fin 128 → mxxMat65) × (Fin 128 → mxxMat65) × (Fin 32 → mxxMat65) × (Fin 32 → mxxMat65) × (Fin 128 → mxxMat68) × (Fin 128 → mxxMat68) × (Fin 32 → mxxMat68) × (Fin 32 → mxxMat68) × (Fin 128 → mxxMat71) × (Fin 128 → mxxMat71) × (Fin 32 → mxxMat71) × (Fin 32 → mxxMat71) × (Fin 128 → mxxMat74) × (Fin 128 → mxxMat74) × (Fin 32 → mxxMat74) × (Fin 32 → mxxMat74) × (Fin 128 → mxxMat77) × (Fin 128 → mxxMat77) × (Fin 32 → mxxMat77) × (Fin 32 → mxxMat77) × (Fin 128 → mxxMat80) × (Fin 128 → mxxMat80) × (Fin 32 → mxxMat80) × (Fin 32 → mxxMat80) × (Fin 128 → mxxMat83) × (Fin 128 → mxxMat83) × (Fin 32 → mxxMat83) × (Fin 32 → mxxMat83) × (Fin 128 → mxxMat86) × (Fin 128 → mxxMat86) × (Fin 32 → mxxMat86) × (Fin 32 → mxxMat86) × (Fin 128 → mxxMat89) × (Fin 128 → mxxMat89) × (Fin 32 → mxxMat89) × (Fin 32 → mxxMat89) × (Fin 128 → mxxMat91) × (Fin 128 → mxxMat91) × (Fin 32 → mxxMat91) × (Fin 32 → mxxMat91) × (Fin 128 → mxxMat94) × (Fin 128 → mxxMat94) × (Fin 32 → mxxMat94) × (Fin 32 → mxxMat94) × (Fin 128 → mxxMat97) × (Fin 128 → mxxMat97) × (Fin 32 → mxxMat97) × (Fin 32 → mxxMat97) × (Fin 128 → mxxMat100) × (Fin 128 → mxxMat100) × (Fin 32 → mxxMat100) × (Fin 32 → mxxMat100) × (Fin 128 → mxxMat103) × (Fin 128 → mxxMat103) × (Fin 32 → mxxMat103) × (Fin 32 → mxxMat103) × (Fin 128 → mxxMat106) × (Fin 128 → mxxMat106) × (Fin 32 → mxxMat106) × (Fin 32 → mxxMat106) × Unit) : Prop :=
  ∃ witness : parallel_generatedRoot_7.Witness,
    parallel_generatedRoot_7.body backend hashModel tape path params i_0 inputs outputs witness

set_option genInjectivity false in
set_option genSizeOf false in
structure generatedRoot.Witness where
  w_7_0 : Fin 8 → mxxMat2
  w_7_1 : Fin 8 → mxxMat2
  w_7_2 : Fin 8 → Fin 1025 → mxxMat0
  w_7_3 : Fin 8 → Fin 1025 → mxxMat0
  w_7_4 : Fin 8 → Fin 1025 → mxxMat1
  w_7_5 : Fin 8 → Fin 128 → mxxMat2
  w_7_6 : Fin 8 → Fin 128 → mxxMat2
  w_7_7 : Fin 8 → Fin 32 → mxxMat2
  w_7_8 : Fin 8 → Fin 32 → mxxMat2
  w_7_9 : Fin 8 → Fin 128 → mxxMat5
  w_7_10 : Fin 8 → Fin 128 → mxxMat5
  w_7_11 : Fin 8 → Fin 32 → mxxMat5
  w_7_12 : Fin 8 → Fin 32 → mxxMat5
  w_7_13 : Fin 8 → Fin 128 → mxxMat8
  w_7_14 : Fin 8 → Fin 128 → mxxMat8
  w_7_15 : Fin 8 → Fin 32 → mxxMat8
  w_7_16 : Fin 8 → Fin 32 → mxxMat8
  w_7_17 : Fin 8 → Fin 128 → mxxMat11
  w_7_18 : Fin 8 → Fin 128 → mxxMat11
  w_7_19 : Fin 8 → Fin 32 → mxxMat11
  w_7_20 : Fin 8 → Fin 32 → mxxMat11
  w_7_21 : Fin 8 → Fin 128 → mxxMat14
  w_7_22 : Fin 8 → Fin 128 → mxxMat14
  w_7_23 : Fin 8 → Fin 32 → mxxMat14
  w_7_24 : Fin 8 → Fin 32 → mxxMat14
  w_7_25 : Fin 8 → Fin 128 → mxxMat17
  w_7_26 : Fin 8 → Fin 128 → mxxMat17
  w_7_27 : Fin 8 → Fin 32 → mxxMat17
  w_7_28 : Fin 8 → Fin 32 → mxxMat17
  w_7_29 : Fin 8 → Fin 128 → mxxMat20
  w_7_30 : Fin 8 → Fin 128 → mxxMat20
  w_7_31 : Fin 8 → Fin 32 → mxxMat20
  w_7_32 : Fin 8 → Fin 32 → mxxMat20
  w_7_33 : Fin 8 → Fin 128 → mxxMat23
  w_7_34 : Fin 8 → Fin 128 → mxxMat23
  w_7_35 : Fin 8 → Fin 32 → mxxMat23
  w_7_36 : Fin 8 → Fin 32 → mxxMat23
  w_7_37 : Fin 8 → Fin 128 → mxxMat26
  w_7_38 : Fin 8 → Fin 128 → mxxMat26
  w_7_39 : Fin 8 → Fin 32 → mxxMat26
  w_7_40 : Fin 8 → Fin 32 → mxxMat26
  w_7_41 : Fin 8 → Fin 128 → mxxMat29
  w_7_42 : Fin 8 → Fin 128 → mxxMat29
  w_7_43 : Fin 8 → Fin 32 → mxxMat29
  w_7_44 : Fin 8 → Fin 32 → mxxMat29
  w_7_45 : Fin 8 → Fin 128 → mxxMat32
  w_7_46 : Fin 8 → Fin 128 → mxxMat32
  w_7_47 : Fin 8 → Fin 32 → mxxMat32
  w_7_48 : Fin 8 → Fin 32 → mxxMat32
  w_7_49 : Fin 8 → Fin 128 → mxxMat35
  w_7_50 : Fin 8 → Fin 128 → mxxMat35
  w_7_51 : Fin 8 → Fin 32 → mxxMat35
  w_7_52 : Fin 8 → Fin 32 → mxxMat35
  w_7_53 : Fin 8 → Fin 128 → mxxMat38
  w_7_54 : Fin 8 → Fin 128 → mxxMat38
  w_7_55 : Fin 8 → Fin 32 → mxxMat38
  w_7_56 : Fin 8 → Fin 32 → mxxMat38
  w_7_57 : Fin 8 → Fin 128 → mxxMat41
  w_7_58 : Fin 8 → Fin 128 → mxxMat41
  w_7_59 : Fin 8 → Fin 32 → mxxMat41
  w_7_60 : Fin 8 → Fin 32 → mxxMat41
  w_7_61 : Fin 8 → Fin 128 → mxxMat44
  w_7_62 : Fin 8 → Fin 128 → mxxMat44
  w_7_63 : Fin 8 → Fin 32 → mxxMat44
  w_7_64 : Fin 8 → Fin 32 → mxxMat44
  w_7_65 : Fin 8 → Fin 128 → mxxMat47
  w_7_66 : Fin 8 → Fin 128 → mxxMat47
  w_7_67 : Fin 8 → Fin 32 → mxxMat47
  w_7_68 : Fin 8 → Fin 32 → mxxMat47
  w_7_69 : Fin 8 → Fin 128 → mxxMat50
  w_7_70 : Fin 8 → Fin 128 → mxxMat50
  w_7_71 : Fin 8 → Fin 32 → mxxMat50
  w_7_72 : Fin 8 → Fin 32 → mxxMat50
  w_7_73 : Fin 8 → Fin 128 → mxxMat53
  w_7_74 : Fin 8 → Fin 128 → mxxMat53
  w_7_75 : Fin 8 → Fin 32 → mxxMat53
  w_7_76 : Fin 8 → Fin 32 → mxxMat53
  w_7_77 : Fin 8 → Fin 128 → mxxMat56
  w_7_78 : Fin 8 → Fin 128 → mxxMat56
  w_7_79 : Fin 8 → Fin 32 → mxxMat56
  w_7_80 : Fin 8 → Fin 32 → mxxMat56
  w_7_81 : Fin 8 → Fin 128 → mxxMat59
  w_7_82 : Fin 8 → Fin 128 → mxxMat59
  w_7_83 : Fin 8 → Fin 32 → mxxMat59
  w_7_84 : Fin 8 → Fin 32 → mxxMat59
  w_7_85 : Fin 8 → Fin 128 → mxxMat62
  w_7_86 : Fin 8 → Fin 128 → mxxMat62
  w_7_87 : Fin 8 → Fin 32 → mxxMat62
  w_7_88 : Fin 8 → Fin 32 → mxxMat62
  w_7_89 : Fin 8 → Fin 128 → mxxMat65
  w_7_90 : Fin 8 → Fin 128 → mxxMat65
  w_7_91 : Fin 8 → Fin 32 → mxxMat65
  w_7_92 : Fin 8 → Fin 32 → mxxMat65
  w_7_93 : Fin 8 → Fin 128 → mxxMat68
  w_7_94 : Fin 8 → Fin 128 → mxxMat68
  w_7_95 : Fin 8 → Fin 32 → mxxMat68
  w_7_96 : Fin 8 → Fin 32 → mxxMat68
  w_7_97 : Fin 8 → Fin 128 → mxxMat71
  w_7_98 : Fin 8 → Fin 128 → mxxMat71
  w_7_99 : Fin 8 → Fin 32 → mxxMat71
  w_7_100 : Fin 8 → Fin 32 → mxxMat71
  w_7_101 : Fin 8 → Fin 128 → mxxMat74
  w_7_102 : Fin 8 → Fin 128 → mxxMat74
  w_7_103 : Fin 8 → Fin 32 → mxxMat74
  w_7_104 : Fin 8 → Fin 32 → mxxMat74
  w_7_105 : Fin 8 → Fin 128 → mxxMat77
  w_7_106 : Fin 8 → Fin 128 → mxxMat77
  w_7_107 : Fin 8 → Fin 32 → mxxMat77
  w_7_108 : Fin 8 → Fin 32 → mxxMat77
  w_7_109 : Fin 8 → Fin 128 → mxxMat80
  w_7_110 : Fin 8 → Fin 128 → mxxMat80
  w_7_111 : Fin 8 → Fin 32 → mxxMat80
  w_7_112 : Fin 8 → Fin 32 → mxxMat80
  w_7_113 : Fin 8 → Fin 128 → mxxMat83
  w_7_114 : Fin 8 → Fin 128 → mxxMat83
  w_7_115 : Fin 8 → Fin 32 → mxxMat83
  w_7_116 : Fin 8 → Fin 32 → mxxMat83
  w_7_117 : Fin 8 → Fin 128 → mxxMat86
  w_7_118 : Fin 8 → Fin 128 → mxxMat86
  w_7_119 : Fin 8 → Fin 32 → mxxMat86
  w_7_120 : Fin 8 → Fin 32 → mxxMat86
  w_7_121 : Fin 8 → Fin 128 → mxxMat89
  w_7_122 : Fin 8 → Fin 128 → mxxMat89
  w_7_123 : Fin 8 → Fin 32 → mxxMat89
  w_7_124 : Fin 8 → Fin 32 → mxxMat89
  w_7_125 : Fin 8 → Fin 128 → mxxMat91
  w_7_126 : Fin 8 → Fin 128 → mxxMat91
  w_7_127 : Fin 8 → Fin 32 → mxxMat91
  w_7_128 : Fin 8 → Fin 32 → mxxMat91
  w_7_129 : Fin 8 → Fin 128 → mxxMat94
  w_7_130 : Fin 8 → Fin 128 → mxxMat94
  w_7_131 : Fin 8 → Fin 32 → mxxMat94
  w_7_132 : Fin 8 → Fin 32 → mxxMat94
  w_7_133 : Fin 8 → Fin 128 → mxxMat97
  w_7_134 : Fin 8 → Fin 128 → mxxMat97
  w_7_135 : Fin 8 → Fin 32 → mxxMat97
  w_7_136 : Fin 8 → Fin 32 → mxxMat97
  w_7_137 : Fin 8 → Fin 128 → mxxMat100
  w_7_138 : Fin 8 → Fin 128 → mxxMat100
  w_7_139 : Fin 8 → Fin 32 → mxxMat100
  w_7_140 : Fin 8 → Fin 32 → mxxMat100
  w_7_141 : Fin 8 → Fin 128 → mxxMat103
  w_7_142 : Fin 8 → Fin 128 → mxxMat103
  w_7_143 : Fin 8 → Fin 32 → mxxMat103
  w_7_144 : Fin 8 → Fin 32 → mxxMat103
  w_7_145 : Fin 8 → Fin 128 → mxxMat106
  w_7_146 : Fin 8 → Fin 128 → mxxMat106
  w_7_147 : Fin 8 → Fin 32 → mxxMat106
  w_7_148 : Fin 8 → Fin 32 → mxxMat106

abbrev generatedRoot.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (w_0_0 : mxxMat88) (w_1_0 : Fin 8 → ByteArray) (w_2_0 : mxxMat88) (w_3_0 : Fin 8 → Int) (w_4_0 : Fin 8 → Int) (w_5_0 : Fin 8 → Fin 1025 → Int) (w_6_0 : Fin 8 → Fin 1025 → Int) (w_7_0 : Fin 8 → mxxMat2) (w_7_1 : Fin 8 → mxxMat2) (w_7_2 : Fin 8 → Fin 1025 → mxxMat0) (w_7_3 : Fin 8 → Fin 1025 → mxxMat0) (w_7_4 : Fin 8 → Fin 1025 → mxxMat1) (w_7_5 : Fin 8 → Fin 128 → mxxMat2) (w_7_6 : Fin 8 → Fin 128 → mxxMat2) (w_7_7 : Fin 8 → Fin 32 → mxxMat2) (w_7_8 : Fin 8 → Fin 32 → mxxMat2) (w_7_9 : Fin 8 → Fin 128 → mxxMat5) (w_7_10 : Fin 8 → Fin 128 → mxxMat5) (w_7_11 : Fin 8 → Fin 32 → mxxMat5) (w_7_12 : Fin 8 → Fin 32 → mxxMat5) (w_7_13 : Fin 8 → Fin 128 → mxxMat8) (w_7_14 : Fin 8 → Fin 128 → mxxMat8) (w_7_15 : Fin 8 → Fin 32 → mxxMat8) (w_7_16 : Fin 8 → Fin 32 → mxxMat8) (w_7_17 : Fin 8 → Fin 128 → mxxMat11) (w_7_18 : Fin 8 → Fin 128 → mxxMat11) (w_7_19 : Fin 8 → Fin 32 → mxxMat11) (w_7_20 : Fin 8 → Fin 32 → mxxMat11) (w_7_21 : Fin 8 → Fin 128 → mxxMat14) (w_7_22 : Fin 8 → Fin 128 → mxxMat14) (w_7_23 : Fin 8 → Fin 32 → mxxMat14) (w_7_24 : Fin 8 → Fin 32 → mxxMat14) (w_7_25 : Fin 8 → Fin 128 → mxxMat17) (w_7_26 : Fin 8 → Fin 128 → mxxMat17) (w_7_27 : Fin 8 → Fin 32 → mxxMat17) (w_7_28 : Fin 8 → Fin 32 → mxxMat17) (w_7_29 : Fin 8 → Fin 128 → mxxMat20) (w_7_30 : Fin 8 → Fin 128 → mxxMat20) (w_7_31 : Fin 8 → Fin 32 → mxxMat20) (w_7_32 : Fin 8 → Fin 32 → mxxMat20) (w_7_33 : Fin 8 → Fin 128 → mxxMat23) (w_7_34 : Fin 8 → Fin 128 → mxxMat23) (w_7_35 : Fin 8 → Fin 32 → mxxMat23) (w_7_36 : Fin 8 → Fin 32 → mxxMat23) (w_7_37 : Fin 8 → Fin 128 → mxxMat26) (w_7_38 : Fin 8 → Fin 128 → mxxMat26) (w_7_39 : Fin 8 → Fin 32 → mxxMat26) (w_7_40 : Fin 8 → Fin 32 → mxxMat26) (w_7_41 : Fin 8 → Fin 128 → mxxMat29) (w_7_42 : Fin 8 → Fin 128 → mxxMat29) (w_7_43 : Fin 8 → Fin 32 → mxxMat29) (w_7_44 : Fin 8 → Fin 32 → mxxMat29) (w_7_45 : Fin 8 → Fin 128 → mxxMat32) (w_7_46 : Fin 8 → Fin 128 → mxxMat32) (w_7_47 : Fin 8 → Fin 32 → mxxMat32) (w_7_48 : Fin 8 → Fin 32 → mxxMat32) (w_7_49 : Fin 8 → Fin 128 → mxxMat35) (w_7_50 : Fin 8 → Fin 128 → mxxMat35) (w_7_51 : Fin 8 → Fin 32 → mxxMat35) (w_7_52 : Fin 8 → Fin 32 → mxxMat35) (w_7_53 : Fin 8 → Fin 128 → mxxMat38) (w_7_54 : Fin 8 → Fin 128 → mxxMat38) (w_7_55 : Fin 8 → Fin 32 → mxxMat38) (w_7_56 : Fin 8 → Fin 32 → mxxMat38) (w_7_57 : Fin 8 → Fin 128 → mxxMat41) (w_7_58 : Fin 8 → Fin 128 → mxxMat41) (w_7_59 : Fin 8 → Fin 32 → mxxMat41) (w_7_60 : Fin 8 → Fin 32 → mxxMat41) (w_7_61 : Fin 8 → Fin 128 → mxxMat44) (w_7_62 : Fin 8 → Fin 128 → mxxMat44) (w_7_63 : Fin 8 → Fin 32 → mxxMat44) (w_7_64 : Fin 8 → Fin 32 → mxxMat44) (w_7_65 : Fin 8 → Fin 128 → mxxMat47) (w_7_66 : Fin 8 → Fin 128 → mxxMat47) (w_7_67 : Fin 8 → Fin 32 → mxxMat47) (w_7_68 : Fin 8 → Fin 32 → mxxMat47) (w_7_69 : Fin 8 → Fin 128 → mxxMat50) (w_7_70 : Fin 8 → Fin 128 → mxxMat50) (w_7_71 : Fin 8 → Fin 32 → mxxMat50) (w_7_72 : Fin 8 → Fin 32 → mxxMat50) (w_7_73 : Fin 8 → Fin 128 → mxxMat53) (w_7_74 : Fin 8 → Fin 128 → mxxMat53) (w_7_75 : Fin 8 → Fin 32 → mxxMat53) (w_7_76 : Fin 8 → Fin 32 → mxxMat53) (w_7_77 : Fin 8 → Fin 128 → mxxMat56) (w_7_78 : Fin 8 → Fin 128 → mxxMat56) (w_7_79 : Fin 8 → Fin 32 → mxxMat56) (w_7_80 : Fin 8 → Fin 32 → mxxMat56) (w_7_81 : Fin 8 → Fin 128 → mxxMat59) (w_7_82 : Fin 8 → Fin 128 → mxxMat59) (w_7_83 : Fin 8 → Fin 32 → mxxMat59) (w_7_84 : Fin 8 → Fin 32 → mxxMat59) (w_7_85 : Fin 8 → Fin 128 → mxxMat62) (w_7_86 : Fin 8 → Fin 128 → mxxMat62) (w_7_87 : Fin 8 → Fin 32 → mxxMat62) (w_7_88 : Fin 8 → Fin 32 → mxxMat62) (w_7_89 : Fin 8 → Fin 128 → mxxMat65) (w_7_90 : Fin 8 → Fin 128 → mxxMat65) (w_7_91 : Fin 8 → Fin 32 → mxxMat65) (w_7_92 : Fin 8 → Fin 32 → mxxMat65) (w_7_93 : Fin 8 → Fin 128 → mxxMat68) (w_7_94 : Fin 8 → Fin 128 → mxxMat68) (w_7_95 : Fin 8 → Fin 32 → mxxMat68) (w_7_96 : Fin 8 → Fin 32 → mxxMat68) (w_7_97 : Fin 8 → Fin 128 → mxxMat71) (w_7_98 : Fin 8 → Fin 128 → mxxMat71) (w_7_99 : Fin 8 → Fin 32 → mxxMat71) (w_7_100 : Fin 8 → Fin 32 → mxxMat71) (w_7_101 : Fin 8 → Fin 128 → mxxMat74) (w_7_102 : Fin 8 → Fin 128 → mxxMat74) (w_7_103 : Fin 8 → Fin 32 → mxxMat74) (w_7_104 : Fin 8 → Fin 32 → mxxMat74) (w_7_105 : Fin 8 → Fin 128 → mxxMat77) (w_7_106 : Fin 8 → Fin 128 → mxxMat77) (w_7_107 : Fin 8 → Fin 32 → mxxMat77) (w_7_108 : Fin 8 → Fin 32 → mxxMat77) (w_7_109 : Fin 8 → Fin 128 → mxxMat80) (w_7_110 : Fin 8 → Fin 128 → mxxMat80) (w_7_111 : Fin 8 → Fin 32 → mxxMat80) (w_7_112 : Fin 8 → Fin 32 → mxxMat80) (w_7_113 : Fin 8 → Fin 128 → mxxMat83) (w_7_114 : Fin 8 → Fin 128 → mxxMat83) (w_7_115 : Fin 8 → Fin 32 → mxxMat83) (w_7_116 : Fin 8 → Fin 32 → mxxMat83) (w_7_117 : Fin 8 → Fin 128 → mxxMat86) (w_7_118 : Fin 8 → Fin 128 → mxxMat86) (w_7_119 : Fin 8 → Fin 32 → mxxMat86) (w_7_120 : Fin 8 → Fin 32 → mxxMat86) (w_7_121 : Fin 8 → Fin 128 → mxxMat89) (w_7_122 : Fin 8 → Fin 128 → mxxMat89) (w_7_123 : Fin 8 → Fin 32 → mxxMat89) (w_7_124 : Fin 8 → Fin 32 → mxxMat89) (w_7_125 : Fin 8 → Fin 128 → mxxMat91) (w_7_126 : Fin 8 → Fin 128 → mxxMat91) (w_7_127 : Fin 8 → Fin 32 → mxxMat91) (w_7_128 : Fin 8 → Fin 32 → mxxMat91) (w_7_129 : Fin 8 → Fin 128 → mxxMat94) (w_7_130 : Fin 8 → Fin 128 → mxxMat94) (w_7_131 : Fin 8 → Fin 32 → mxxMat94) (w_7_132 : Fin 8 → Fin 32 → mxxMat94) (w_7_133 : Fin 8 → Fin 128 → mxxMat97) (w_7_134 : Fin 8 → Fin 128 → mxxMat97) (w_7_135 : Fin 8 → Fin 32 → mxxMat97) (w_7_136 : Fin 8 → Fin 32 → mxxMat97) (w_7_137 : Fin 8 → Fin 128 → mxxMat100) (w_7_138 : Fin 8 → Fin 128 → mxxMat100) (w_7_139 : Fin 8 → Fin 32 → mxxMat100) (w_7_140 : Fin 8 → Fin 32 → mxxMat100) (w_7_141 : Fin 8 → Fin 128 → mxxMat103) (w_7_142 : Fin 8 → Fin 128 → mxxMat103) (w_7_143 : Fin 8 → Fin 32 → mxxMat103) (w_7_144 : Fin 8 → Fin 32 → mxxMat103) (w_7_145 : Fin 8 → Fin 128 → mxxMat106) (w_7_146 : Fin 8 → Fin 128 → mxxMat106) (w_7_147 : Fin 8 → Fin 32 → mxxMat106) (w_7_148 : Fin 8 → Fin 32 → mxxMat106) (outputs : (Fin 8 → mxxMat2) × (Fin 8 → mxxMat2) × (Fin 8 → Fin 1025 → mxxMat0) × (Fin 8 → Fin 1025 → mxxMat0) × (Fin 8 → Fin 1025 → mxxMat1) × (Fin 8 → Fin 128 → mxxMat2) × (Fin 8 → Fin 128 → mxxMat2) × (Fin 8 → Fin 32 → mxxMat2) × (Fin 8 → Fin 32 → mxxMat2) × (Fin 8 → Fin 128 → mxxMat5) × (Fin 8 → Fin 128 → mxxMat5) × (Fin 8 → Fin 32 → mxxMat5) × (Fin 8 → Fin 32 → mxxMat5) × (Fin 8 → Fin 128 → mxxMat8) × (Fin 8 → Fin 128 → mxxMat8) × (Fin 8 → Fin 32 → mxxMat8) × (Fin 8 → Fin 32 → mxxMat8) × (Fin 8 → Fin 128 → mxxMat11) × (Fin 8 → Fin 128 → mxxMat11) × (Fin 8 → Fin 32 → mxxMat11) × (Fin 8 → Fin 32 → mxxMat11) × (Fin 8 → Fin 128 → mxxMat14) × (Fin 8 → Fin 128 → mxxMat14) × (Fin 8 → Fin 32 → mxxMat14) × (Fin 8 → Fin 32 → mxxMat14) × (Fin 8 → Fin 128 → mxxMat17) × (Fin 8 → Fin 128 → mxxMat17) × (Fin 8 → Fin 32 → mxxMat17) × (Fin 8 → Fin 32 → mxxMat17) × (Fin 8 → Fin 128 → mxxMat20) × (Fin 8 → Fin 128 → mxxMat20) × (Fin 8 → Fin 32 → mxxMat20) × (Fin 8 → Fin 32 → mxxMat20) × (Fin 8 → Fin 128 → mxxMat23) × (Fin 8 → Fin 128 → mxxMat23) × (Fin 8 → Fin 32 → mxxMat23) × (Fin 8 → Fin 32 → mxxMat23) × (Fin 8 → Fin 128 → mxxMat26) × (Fin 8 → Fin 128 → mxxMat26) × (Fin 8 → Fin 32 → mxxMat26) × (Fin 8 → Fin 32 → mxxMat26) × (Fin 8 → Fin 128 → mxxMat29) × (Fin 8 → Fin 128 → mxxMat29) × (Fin 8 → Fin 32 → mxxMat29) × (Fin 8 → Fin 32 → mxxMat29) × (Fin 8 → Fin 128 → mxxMat32) × (Fin 8 → Fin 128 → mxxMat32) × (Fin 8 → Fin 32 → mxxMat32) × (Fin 8 → Fin 32 → mxxMat32) × (Fin 8 → Fin 128 → mxxMat35) × (Fin 8 → Fin 128 → mxxMat35) × (Fin 8 → Fin 32 → mxxMat35) × (Fin 8 → Fin 32 → mxxMat35) × (Fin 8 → Fin 128 → mxxMat38) × (Fin 8 → Fin 128 → mxxMat38) × (Fin 8 → Fin 32 → mxxMat38) × (Fin 8 → Fin 32 → mxxMat38) × (Fin 8 → Fin 128 → mxxMat41) × (Fin 8 → Fin 128 → mxxMat41) × (Fin 8 → Fin 32 → mxxMat41) × (Fin 8 → Fin 32 → mxxMat41) × (Fin 8 → Fin 128 → mxxMat44) × (Fin 8 → Fin 128 → mxxMat44) × (Fin 8 → Fin 32 → mxxMat44) × (Fin 8 → Fin 32 → mxxMat44) × (Fin 8 → Fin 128 → mxxMat47) × (Fin 8 → Fin 128 → mxxMat47) × (Fin 8 → Fin 32 → mxxMat47) × (Fin 8 → Fin 32 → mxxMat47) × (Fin 8 → Fin 128 → mxxMat50) × (Fin 8 → Fin 128 → mxxMat50) × (Fin 8 → Fin 32 → mxxMat50) × (Fin 8 → Fin 32 → mxxMat50) × (Fin 8 → Fin 128 → mxxMat53) × (Fin 8 → Fin 128 → mxxMat53) × (Fin 8 → Fin 32 → mxxMat53) × (Fin 8 → Fin 32 → mxxMat53) × (Fin 8 → Fin 128 → mxxMat56) × (Fin 8 → Fin 128 → mxxMat56) × (Fin 8 → Fin 32 → mxxMat56) × (Fin 8 → Fin 32 → mxxMat56) × (Fin 8 → Fin 128 → mxxMat59) × (Fin 8 → Fin 128 → mxxMat59) × (Fin 8 → Fin 32 → mxxMat59) × (Fin 8 → Fin 32 → mxxMat59) × (Fin 8 → Fin 128 → mxxMat62) × (Fin 8 → Fin 128 → mxxMat62) × (Fin 8 → Fin 32 → mxxMat62) × (Fin 8 → Fin 32 → mxxMat62) × (Fin 8 → Fin 128 → mxxMat65) × (Fin 8 → Fin 128 → mxxMat65) × (Fin 8 → Fin 32 → mxxMat65) × (Fin 8 → Fin 32 → mxxMat65) × (Fin 8 → Fin 128 → mxxMat68) × (Fin 8 → Fin 128 → mxxMat68) × (Fin 8 → Fin 32 → mxxMat68) × (Fin 8 → Fin 32 → mxxMat68) × (Fin 8 → Fin 128 → mxxMat71) × (Fin 8 → Fin 128 → mxxMat71) × (Fin 8 → Fin 32 → mxxMat71) × (Fin 8 → Fin 32 → mxxMat71) × (Fin 8 → Fin 128 → mxxMat74) × (Fin 8 → Fin 128 → mxxMat74) × (Fin 8 → Fin 32 → mxxMat74) × (Fin 8 → Fin 32 → mxxMat74) × (Fin 8 → Fin 128 → mxxMat77) × (Fin 8 → Fin 128 → mxxMat77) × (Fin 8 → Fin 32 → mxxMat77) × (Fin 8 → Fin 32 → mxxMat77) × (Fin 8 → Fin 128 → mxxMat80) × (Fin 8 → Fin 128 → mxxMat80) × (Fin 8 → Fin 32 → mxxMat80) × (Fin 8 → Fin 32 → mxxMat80) × (Fin 8 → Fin 128 → mxxMat83) × (Fin 8 → Fin 128 → mxxMat83) × (Fin 8 → Fin 32 → mxxMat83) × (Fin 8 → Fin 32 → mxxMat83) × (Fin 8 → Fin 128 → mxxMat86) × (Fin 8 → Fin 128 → mxxMat86) × (Fin 8 → Fin 32 → mxxMat86) × (Fin 8 → Fin 32 → mxxMat86) × (Fin 8 → Fin 128 → mxxMat89) × (Fin 8 → Fin 128 → mxxMat89) × (Fin 8 → Fin 32 → mxxMat89) × (Fin 8 → Fin 32 → mxxMat89) × (Fin 8 → Fin 128 → mxxMat91) × (Fin 8 → Fin 128 → mxxMat91) × (Fin 8 → Fin 32 → mxxMat91) × (Fin 8 → Fin 32 → mxxMat91) × (Fin 8 → Fin 128 → mxxMat94) × (Fin 8 → Fin 128 → mxxMat94) × (Fin 8 → Fin 32 → mxxMat94) × (Fin 8 → Fin 32 → mxxMat94) × (Fin 8 → Fin 128 → mxxMat97) × (Fin 8 → Fin 128 → mxxMat97) × (Fin 8 → Fin 32 → mxxMat97) × (Fin 8 → Fin 32 → mxxMat97) × (Fin 8 → Fin 128 → mxxMat100) × (Fin 8 → Fin 128 → mxxMat100) × (Fin 8 → Fin 32 → mxxMat100) × (Fin 8 → Fin 32 → mxxMat100) × (Fin 8 → Fin 128 → mxxMat103) × (Fin 8 → Fin 128 → mxxMat103) × (Fin 8 → Fin 32 → mxxMat103) × (Fin 8 → Fin 32 → mxxMat103) × (Fin 8 → Fin 128 → mxxMat106) × (Fin 8 → Fin 128 → mxxMat106) × (Fin 8 → Fin 32 → mxxMat106) × (Fin 8 → Fin 32 → mxxMat106) × Unit) : Prop :=
  (8) = 8 ∧
  (∀ i : Fin 8, parallel_generatedRoot_7 backend hashModel tape (path ++ [7, i.val]) params i (w_0_0, ((w_1_0 i), (w_2_0, ((w_3_0 i), ((w_4_0 i), ((w_5_0 i), ((w_6_0 i), ()))))))) ((w_7_0 i), ((w_7_1 i), ((w_7_2 i), ((w_7_3 i), ((w_7_4 i), ((w_7_5 i), ((w_7_6 i), ((w_7_7 i), ((w_7_8 i), ((w_7_9 i), ((w_7_10 i), ((w_7_11 i), ((w_7_12 i), ((w_7_13 i), ((w_7_14 i), ((w_7_15 i), ((w_7_16 i), ((w_7_17 i), ((w_7_18 i), ((w_7_19 i), ((w_7_20 i), ((w_7_21 i), ((w_7_22 i), ((w_7_23 i), ((w_7_24 i), ((w_7_25 i), ((w_7_26 i), ((w_7_27 i), ((w_7_28 i), ((w_7_29 i), ((w_7_30 i), ((w_7_31 i), ((w_7_32 i), ((w_7_33 i), ((w_7_34 i), ((w_7_35 i), ((w_7_36 i), ((w_7_37 i), ((w_7_38 i), ((w_7_39 i), ((w_7_40 i), ((w_7_41 i), ((w_7_42 i), ((w_7_43 i), ((w_7_44 i), ((w_7_45 i), ((w_7_46 i), ((w_7_47 i), ((w_7_48 i), ((w_7_49 i), ((w_7_50 i), ((w_7_51 i), ((w_7_52 i), ((w_7_53 i), ((w_7_54 i), ((w_7_55 i), ((w_7_56 i), ((w_7_57 i), ((w_7_58 i), ((w_7_59 i), ((w_7_60 i), ((w_7_61 i), ((w_7_62 i), ((w_7_63 i), ((w_7_64 i), ((w_7_65 i), ((w_7_66 i), ((w_7_67 i), ((w_7_68 i), ((w_7_69 i), ((w_7_70 i), ((w_7_71 i), ((w_7_72 i), ((w_7_73 i), ((w_7_74 i), ((w_7_75 i), ((w_7_76 i), ((w_7_77 i), ((w_7_78 i), ((w_7_79 i), ((w_7_80 i), ((w_7_81 i), ((w_7_82 i), ((w_7_83 i), ((w_7_84 i), ((w_7_85 i), ((w_7_86 i), ((w_7_87 i), ((w_7_88 i), ((w_7_89 i), ((w_7_90 i), ((w_7_91 i), ((w_7_92 i), ((w_7_93 i), ((w_7_94 i), ((w_7_95 i), ((w_7_96 i), ((w_7_97 i), ((w_7_98 i), ((w_7_99 i), ((w_7_100 i), ((w_7_101 i), ((w_7_102 i), ((w_7_103 i), ((w_7_104 i), ((w_7_105 i), ((w_7_106 i), ((w_7_107 i), ((w_7_108 i), ((w_7_109 i), ((w_7_110 i), ((w_7_111 i), ((w_7_112 i), ((w_7_113 i), ((w_7_114 i), ((w_7_115 i), ((w_7_116 i), ((w_7_117 i), ((w_7_118 i), ((w_7_119 i), ((w_7_120 i), ((w_7_121 i), ((w_7_122 i), ((w_7_123 i), ((w_7_124 i), ((w_7_125 i), ((w_7_126 i), ((w_7_127 i), ((w_7_128 i), ((w_7_129 i), ((w_7_130 i), ((w_7_131 i), ((w_7_132 i), ((w_7_133 i), ((w_7_134 i), ((w_7_135 i), ((w_7_136 i), ((w_7_137 i), ((w_7_138 i), ((w_7_139 i), ((w_7_140 i), ((w_7_141 i), ((w_7_142 i), ((w_7_143 i), ((w_7_144 i), ((w_7_145 i), ((w_7_146 i), ((w_7_147 i), ((w_7_148 i), ())))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))) ∧
  outputs = (w_7_0, (w_7_1, (w_7_2, (w_7_3, (w_7_4, (w_7_5, (w_7_6, (w_7_7, (w_7_8, (w_7_9, (w_7_10, (w_7_11, (w_7_12, (w_7_13, (w_7_14, (w_7_15, (w_7_16, (w_7_17, (w_7_18, (w_7_19, (w_7_20, (w_7_21, (w_7_22, (w_7_23, (w_7_24, (w_7_25, (w_7_26, (w_7_27, (w_7_28, (w_7_29, (w_7_30, (w_7_31, (w_7_32, (w_7_33, (w_7_34, (w_7_35, (w_7_36, (w_7_37, (w_7_38, (w_7_39, (w_7_40, (w_7_41, (w_7_42, (w_7_43, (w_7_44, (w_7_45, (w_7_46, (w_7_47, (w_7_48, (w_7_49, (w_7_50, (w_7_51, (w_7_52, (w_7_53, (w_7_54, (w_7_55, (w_7_56, (w_7_57, (w_7_58, (w_7_59, (w_7_60, (w_7_61, (w_7_62, (w_7_63, (w_7_64, (w_7_65, (w_7_66, (w_7_67, (w_7_68, (w_7_69, (w_7_70, (w_7_71, (w_7_72, (w_7_73, (w_7_74, (w_7_75, (w_7_76, (w_7_77, (w_7_78, (w_7_79, (w_7_80, (w_7_81, (w_7_82, (w_7_83, (w_7_84, (w_7_85, (w_7_86, (w_7_87, (w_7_88, (w_7_89, (w_7_90, (w_7_91, (w_7_92, (w_7_93, (w_7_94, (w_7_95, (w_7_96, (w_7_97, (w_7_98, (w_7_99, (w_7_100, (w_7_101, (w_7_102, (w_7_103, (w_7_104, (w_7_105, (w_7_106, (w_7_107, (w_7_108, (w_7_109, (w_7_110, (w_7_111, (w_7_112, (w_7_113, (w_7_114, (w_7_115, (w_7_116, (w_7_117, (w_7_118, (w_7_119, (w_7_120, (w_7_121, (w_7_122, (w_7_123, (w_7_124, (w_7_125, (w_7_126, (w_7_127, (w_7_128, (w_7_129, (w_7_130, (w_7_131, (w_7_132, (w_7_133, (w_7_134, (w_7_135, (w_7_136, (w_7_137, (w_7_138, (w_7_139, (w_7_140, (w_7_141, (w_7_142, (w_7_143, (w_7_144, (w_7_145, (w_7_146, (w_7_147, (w_7_148, ())))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))

abbrev generatedRoot.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (inputs : mxxMat88 × (Fin 8 → ByteArray) × mxxMat88 × (Fin 8 → Int) × (Fin 8 → Int) × (Fin 8 → Fin 1025 → Int) × (Fin 8 → Fin 1025 → Int) × Unit) (outputs : (Fin 8 → mxxMat2) × (Fin 8 → mxxMat2) × (Fin 8 → Fin 1025 → mxxMat0) × (Fin 8 → Fin 1025 → mxxMat0) × (Fin 8 → Fin 1025 → mxxMat1) × (Fin 8 → Fin 128 → mxxMat2) × (Fin 8 → Fin 128 → mxxMat2) × (Fin 8 → Fin 32 → mxxMat2) × (Fin 8 → Fin 32 → mxxMat2) × (Fin 8 → Fin 128 → mxxMat5) × (Fin 8 → Fin 128 → mxxMat5) × (Fin 8 → Fin 32 → mxxMat5) × (Fin 8 → Fin 32 → mxxMat5) × (Fin 8 → Fin 128 → mxxMat8) × (Fin 8 → Fin 128 → mxxMat8) × (Fin 8 → Fin 32 → mxxMat8) × (Fin 8 → Fin 32 → mxxMat8) × (Fin 8 → Fin 128 → mxxMat11) × (Fin 8 → Fin 128 → mxxMat11) × (Fin 8 → Fin 32 → mxxMat11) × (Fin 8 → Fin 32 → mxxMat11) × (Fin 8 → Fin 128 → mxxMat14) × (Fin 8 → Fin 128 → mxxMat14) × (Fin 8 → Fin 32 → mxxMat14) × (Fin 8 → Fin 32 → mxxMat14) × (Fin 8 → Fin 128 → mxxMat17) × (Fin 8 → Fin 128 → mxxMat17) × (Fin 8 → Fin 32 → mxxMat17) × (Fin 8 → Fin 32 → mxxMat17) × (Fin 8 → Fin 128 → mxxMat20) × (Fin 8 → Fin 128 → mxxMat20) × (Fin 8 → Fin 32 → mxxMat20) × (Fin 8 → Fin 32 → mxxMat20) × (Fin 8 → Fin 128 → mxxMat23) × (Fin 8 → Fin 128 → mxxMat23) × (Fin 8 → Fin 32 → mxxMat23) × (Fin 8 → Fin 32 → mxxMat23) × (Fin 8 → Fin 128 → mxxMat26) × (Fin 8 → Fin 128 → mxxMat26) × (Fin 8 → Fin 32 → mxxMat26) × (Fin 8 → Fin 32 → mxxMat26) × (Fin 8 → Fin 128 → mxxMat29) × (Fin 8 → Fin 128 → mxxMat29) × (Fin 8 → Fin 32 → mxxMat29) × (Fin 8 → Fin 32 → mxxMat29) × (Fin 8 → Fin 128 → mxxMat32) × (Fin 8 → Fin 128 → mxxMat32) × (Fin 8 → Fin 32 → mxxMat32) × (Fin 8 → Fin 32 → mxxMat32) × (Fin 8 → Fin 128 → mxxMat35) × (Fin 8 → Fin 128 → mxxMat35) × (Fin 8 → Fin 32 → mxxMat35) × (Fin 8 → Fin 32 → mxxMat35) × (Fin 8 → Fin 128 → mxxMat38) × (Fin 8 → Fin 128 → mxxMat38) × (Fin 8 → Fin 32 → mxxMat38) × (Fin 8 → Fin 32 → mxxMat38) × (Fin 8 → Fin 128 → mxxMat41) × (Fin 8 → Fin 128 → mxxMat41) × (Fin 8 → Fin 32 → mxxMat41) × (Fin 8 → Fin 32 → mxxMat41) × (Fin 8 → Fin 128 → mxxMat44) × (Fin 8 → Fin 128 → mxxMat44) × (Fin 8 → Fin 32 → mxxMat44) × (Fin 8 → Fin 32 → mxxMat44) × (Fin 8 → Fin 128 → mxxMat47) × (Fin 8 → Fin 128 → mxxMat47) × (Fin 8 → Fin 32 → mxxMat47) × (Fin 8 → Fin 32 → mxxMat47) × (Fin 8 → Fin 128 → mxxMat50) × (Fin 8 → Fin 128 → mxxMat50) × (Fin 8 → Fin 32 → mxxMat50) × (Fin 8 → Fin 32 → mxxMat50) × (Fin 8 → Fin 128 → mxxMat53) × (Fin 8 → Fin 128 → mxxMat53) × (Fin 8 → Fin 32 → mxxMat53) × (Fin 8 → Fin 32 → mxxMat53) × (Fin 8 → Fin 128 → mxxMat56) × (Fin 8 → Fin 128 → mxxMat56) × (Fin 8 → Fin 32 → mxxMat56) × (Fin 8 → Fin 32 → mxxMat56) × (Fin 8 → Fin 128 → mxxMat59) × (Fin 8 → Fin 128 → mxxMat59) × (Fin 8 → Fin 32 → mxxMat59) × (Fin 8 → Fin 32 → mxxMat59) × (Fin 8 → Fin 128 → mxxMat62) × (Fin 8 → Fin 128 → mxxMat62) × (Fin 8 → Fin 32 → mxxMat62) × (Fin 8 → Fin 32 → mxxMat62) × (Fin 8 → Fin 128 → mxxMat65) × (Fin 8 → Fin 128 → mxxMat65) × (Fin 8 → Fin 32 → mxxMat65) × (Fin 8 → Fin 32 → mxxMat65) × (Fin 8 → Fin 128 → mxxMat68) × (Fin 8 → Fin 128 → mxxMat68) × (Fin 8 → Fin 32 → mxxMat68) × (Fin 8 → Fin 32 → mxxMat68) × (Fin 8 → Fin 128 → mxxMat71) × (Fin 8 → Fin 128 → mxxMat71) × (Fin 8 → Fin 32 → mxxMat71) × (Fin 8 → Fin 32 → mxxMat71) × (Fin 8 → Fin 128 → mxxMat74) × (Fin 8 → Fin 128 → mxxMat74) × (Fin 8 → Fin 32 → mxxMat74) × (Fin 8 → Fin 32 → mxxMat74) × (Fin 8 → Fin 128 → mxxMat77) × (Fin 8 → Fin 128 → mxxMat77) × (Fin 8 → Fin 32 → mxxMat77) × (Fin 8 → Fin 32 → mxxMat77) × (Fin 8 → Fin 128 → mxxMat80) × (Fin 8 → Fin 128 → mxxMat80) × (Fin 8 → Fin 32 → mxxMat80) × (Fin 8 → Fin 32 → mxxMat80) × (Fin 8 → Fin 128 → mxxMat83) × (Fin 8 → Fin 128 → mxxMat83) × (Fin 8 → Fin 32 → mxxMat83) × (Fin 8 → Fin 32 → mxxMat83) × (Fin 8 → Fin 128 → mxxMat86) × (Fin 8 → Fin 128 → mxxMat86) × (Fin 8 → Fin 32 → mxxMat86) × (Fin 8 → Fin 32 → mxxMat86) × (Fin 8 → Fin 128 → mxxMat89) × (Fin 8 → Fin 128 → mxxMat89) × (Fin 8 → Fin 32 → mxxMat89) × (Fin 8 → Fin 32 → mxxMat89) × (Fin 8 → Fin 128 → mxxMat91) × (Fin 8 → Fin 128 → mxxMat91) × (Fin 8 → Fin 32 → mxxMat91) × (Fin 8 → Fin 32 → mxxMat91) × (Fin 8 → Fin 128 → mxxMat94) × (Fin 8 → Fin 128 → mxxMat94) × (Fin 8 → Fin 32 → mxxMat94) × (Fin 8 → Fin 32 → mxxMat94) × (Fin 8 → Fin 128 → mxxMat97) × (Fin 8 → Fin 128 → mxxMat97) × (Fin 8 → Fin 32 → mxxMat97) × (Fin 8 → Fin 32 → mxxMat97) × (Fin 8 → Fin 128 → mxxMat100) × (Fin 8 → Fin 128 → mxxMat100) × (Fin 8 → Fin 32 → mxxMat100) × (Fin 8 → Fin 32 → mxxMat100) × (Fin 8 → Fin 128 → mxxMat103) × (Fin 8 → Fin 128 → mxxMat103) × (Fin 8 → Fin 32 → mxxMat103) × (Fin 8 → Fin 32 → mxxMat103) × (Fin 8 → Fin 128 → mxxMat106) × (Fin 8 → Fin 128 → mxxMat106) × (Fin 8 → Fin 32 → mxxMat106) × (Fin 8 → Fin 32 → mxxMat106) × Unit) (witness : generatedRoot.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat88 := inputs.1
  let w_1_0 : Fin 8 → ByteArray := inputs.2.1
  let w_2_0 : mxxMat88 := inputs.2.2.1
  let w_3_0 : Fin 8 → Int := inputs.2.2.2.1
  let w_4_0 : Fin 8 → Int := inputs.2.2.2.2.1
  let w_5_0 : Fin 8 → Fin 1025 → Int := inputs.2.2.2.2.2.1
  let w_6_0 : Fin 8 → Fin 1025 → Int := inputs.2.2.2.2.2.2.1
  let w_7_0 := witness.w_7_0
  let w_7_1 := witness.w_7_1
  let w_7_2 := witness.w_7_2
  let w_7_3 := witness.w_7_3
  let w_7_4 := witness.w_7_4
  let w_7_5 := witness.w_7_5
  let w_7_6 := witness.w_7_6
  let w_7_7 := witness.w_7_7
  let w_7_8 := witness.w_7_8
  let w_7_9 := witness.w_7_9
  let w_7_10 := witness.w_7_10
  let w_7_11 := witness.w_7_11
  let w_7_12 := witness.w_7_12
  let w_7_13 := witness.w_7_13
  let w_7_14 := witness.w_7_14
  let w_7_15 := witness.w_7_15
  let w_7_16 := witness.w_7_16
  let w_7_17 := witness.w_7_17
  let w_7_18 := witness.w_7_18
  let w_7_19 := witness.w_7_19
  let w_7_20 := witness.w_7_20
  let w_7_21 := witness.w_7_21
  let w_7_22 := witness.w_7_22
  let w_7_23 := witness.w_7_23
  let w_7_24 := witness.w_7_24
  let w_7_25 := witness.w_7_25
  let w_7_26 := witness.w_7_26
  let w_7_27 := witness.w_7_27
  let w_7_28 := witness.w_7_28
  let w_7_29 := witness.w_7_29
  let w_7_30 := witness.w_7_30
  let w_7_31 := witness.w_7_31
  let w_7_32 := witness.w_7_32
  let w_7_33 := witness.w_7_33
  let w_7_34 := witness.w_7_34
  let w_7_35 := witness.w_7_35
  let w_7_36 := witness.w_7_36
  let w_7_37 := witness.w_7_37
  let w_7_38 := witness.w_7_38
  let w_7_39 := witness.w_7_39
  let w_7_40 := witness.w_7_40
  let w_7_41 := witness.w_7_41
  let w_7_42 := witness.w_7_42
  let w_7_43 := witness.w_7_43
  let w_7_44 := witness.w_7_44
  let w_7_45 := witness.w_7_45
  let w_7_46 := witness.w_7_46
  let w_7_47 := witness.w_7_47
  let w_7_48 := witness.w_7_48
  let w_7_49 := witness.w_7_49
  let w_7_50 := witness.w_7_50
  let w_7_51 := witness.w_7_51
  let w_7_52 := witness.w_7_52
  let w_7_53 := witness.w_7_53
  let w_7_54 := witness.w_7_54
  let w_7_55 := witness.w_7_55
  let w_7_56 := witness.w_7_56
  let w_7_57 := witness.w_7_57
  let w_7_58 := witness.w_7_58
  let w_7_59 := witness.w_7_59
  let w_7_60 := witness.w_7_60
  let w_7_61 := witness.w_7_61
  let w_7_62 := witness.w_7_62
  let w_7_63 := witness.w_7_63
  let w_7_64 := witness.w_7_64
  let w_7_65 := witness.w_7_65
  let w_7_66 := witness.w_7_66
  let w_7_67 := witness.w_7_67
  let w_7_68 := witness.w_7_68
  let w_7_69 := witness.w_7_69
  let w_7_70 := witness.w_7_70
  let w_7_71 := witness.w_7_71
  let w_7_72 := witness.w_7_72
  let w_7_73 := witness.w_7_73
  let w_7_74 := witness.w_7_74
  let w_7_75 := witness.w_7_75
  let w_7_76 := witness.w_7_76
  let w_7_77 := witness.w_7_77
  let w_7_78 := witness.w_7_78
  let w_7_79 := witness.w_7_79
  let w_7_80 := witness.w_7_80
  let w_7_81 := witness.w_7_81
  let w_7_82 := witness.w_7_82
  let w_7_83 := witness.w_7_83
  let w_7_84 := witness.w_7_84
  let w_7_85 := witness.w_7_85
  let w_7_86 := witness.w_7_86
  let w_7_87 := witness.w_7_87
  let w_7_88 := witness.w_7_88
  let w_7_89 := witness.w_7_89
  let w_7_90 := witness.w_7_90
  let w_7_91 := witness.w_7_91
  let w_7_92 := witness.w_7_92
  let w_7_93 := witness.w_7_93
  let w_7_94 := witness.w_7_94
  let w_7_95 := witness.w_7_95
  let w_7_96 := witness.w_7_96
  let w_7_97 := witness.w_7_97
  let w_7_98 := witness.w_7_98
  let w_7_99 := witness.w_7_99
  let w_7_100 := witness.w_7_100
  let w_7_101 := witness.w_7_101
  let w_7_102 := witness.w_7_102
  let w_7_103 := witness.w_7_103
  let w_7_104 := witness.w_7_104
  let w_7_105 := witness.w_7_105
  let w_7_106 := witness.w_7_106
  let w_7_107 := witness.w_7_107
  let w_7_108 := witness.w_7_108
  let w_7_109 := witness.w_7_109
  let w_7_110 := witness.w_7_110
  let w_7_111 := witness.w_7_111
  let w_7_112 := witness.w_7_112
  let w_7_113 := witness.w_7_113
  let w_7_114 := witness.w_7_114
  let w_7_115 := witness.w_7_115
  let w_7_116 := witness.w_7_116
  let w_7_117 := witness.w_7_117
  let w_7_118 := witness.w_7_118
  let w_7_119 := witness.w_7_119
  let w_7_120 := witness.w_7_120
  let w_7_121 := witness.w_7_121
  let w_7_122 := witness.w_7_122
  let w_7_123 := witness.w_7_123
  let w_7_124 := witness.w_7_124
  let w_7_125 := witness.w_7_125
  let w_7_126 := witness.w_7_126
  let w_7_127 := witness.w_7_127
  let w_7_128 := witness.w_7_128
  let w_7_129 := witness.w_7_129
  let w_7_130 := witness.w_7_130
  let w_7_131 := witness.w_7_131
  let w_7_132 := witness.w_7_132
  let w_7_133 := witness.w_7_133
  let w_7_134 := witness.w_7_134
  let w_7_135 := witness.w_7_135
  let w_7_136 := witness.w_7_136
  let w_7_137 := witness.w_7_137
  let w_7_138 := witness.w_7_138
  let w_7_139 := witness.w_7_139
  let w_7_140 := witness.w_7_140
  let w_7_141 := witness.w_7_141
  let w_7_142 := witness.w_7_142
  let w_7_143 := witness.w_7_143
  let w_7_144 := witness.w_7_144
  let w_7_145 := witness.w_7_145
  let w_7_146 := witness.w_7_146
  let w_7_147 := witness.w_7_147
  let w_7_148 := witness.w_7_148
  generatedRoot.constraints_0 backend hashModel tape path params w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_7_0 w_7_1 w_7_2 w_7_3 w_7_4 w_7_5 w_7_6 w_7_7 w_7_8 w_7_9 w_7_10 w_7_11 w_7_12 w_7_13 w_7_14 w_7_15 w_7_16 w_7_17 w_7_18 w_7_19 w_7_20 w_7_21 w_7_22 w_7_23 w_7_24 w_7_25 w_7_26 w_7_27 w_7_28 w_7_29 w_7_30 w_7_31 w_7_32 w_7_33 w_7_34 w_7_35 w_7_36 w_7_37 w_7_38 w_7_39 w_7_40 w_7_41 w_7_42 w_7_43 w_7_44 w_7_45 w_7_46 w_7_47 w_7_48 w_7_49 w_7_50 w_7_51 w_7_52 w_7_53 w_7_54 w_7_55 w_7_56 w_7_57 w_7_58 w_7_59 w_7_60 w_7_61 w_7_62 w_7_63 w_7_64 w_7_65 w_7_66 w_7_67 w_7_68 w_7_69 w_7_70 w_7_71 w_7_72 w_7_73 w_7_74 w_7_75 w_7_76 w_7_77 w_7_78 w_7_79 w_7_80 w_7_81 w_7_82 w_7_83 w_7_84 w_7_85 w_7_86 w_7_87 w_7_88 w_7_89 w_7_90 w_7_91 w_7_92 w_7_93 w_7_94 w_7_95 w_7_96 w_7_97 w_7_98 w_7_99 w_7_100 w_7_101 w_7_102 w_7_103 w_7_104 w_7_105 w_7_106 w_7_107 w_7_108 w_7_109 w_7_110 w_7_111 w_7_112 w_7_113 w_7_114 w_7_115 w_7_116 w_7_117 w_7_118 w_7_119 w_7_120 w_7_121 w_7_122 w_7_123 w_7_124 w_7_125 w_7_126 w_7_127 w_7_128 w_7_129 w_7_130 w_7_131 w_7_132 w_7_133 w_7_134 w_7_135 w_7_136 w_7_137 w_7_138 w_7_139 w_7_140 w_7_141 w_7_142 w_7_143 w_7_144 w_7_145 w_7_146 w_7_147 w_7_148 outputs

def generatedRoot (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (inputs : mxxMat88 × (Fin 8 → ByteArray) × mxxMat88 × (Fin 8 → Int) × (Fin 8 → Int) × (Fin 8 → Fin 1025 → Int) × (Fin 8 → Fin 1025 → Int) × Unit) (outputs : (Fin 8 → mxxMat2) × (Fin 8 → mxxMat2) × (Fin 8 → Fin 1025 → mxxMat0) × (Fin 8 → Fin 1025 → mxxMat0) × (Fin 8 → Fin 1025 → mxxMat1) × (Fin 8 → Fin 128 → mxxMat2) × (Fin 8 → Fin 128 → mxxMat2) × (Fin 8 → Fin 32 → mxxMat2) × (Fin 8 → Fin 32 → mxxMat2) × (Fin 8 → Fin 128 → mxxMat5) × (Fin 8 → Fin 128 → mxxMat5) × (Fin 8 → Fin 32 → mxxMat5) × (Fin 8 → Fin 32 → mxxMat5) × (Fin 8 → Fin 128 → mxxMat8) × (Fin 8 → Fin 128 → mxxMat8) × (Fin 8 → Fin 32 → mxxMat8) × (Fin 8 → Fin 32 → mxxMat8) × (Fin 8 → Fin 128 → mxxMat11) × (Fin 8 → Fin 128 → mxxMat11) × (Fin 8 → Fin 32 → mxxMat11) × (Fin 8 → Fin 32 → mxxMat11) × (Fin 8 → Fin 128 → mxxMat14) × (Fin 8 → Fin 128 → mxxMat14) × (Fin 8 → Fin 32 → mxxMat14) × (Fin 8 → Fin 32 → mxxMat14) × (Fin 8 → Fin 128 → mxxMat17) × (Fin 8 → Fin 128 → mxxMat17) × (Fin 8 → Fin 32 → mxxMat17) × (Fin 8 → Fin 32 → mxxMat17) × (Fin 8 → Fin 128 → mxxMat20) × (Fin 8 → Fin 128 → mxxMat20) × (Fin 8 → Fin 32 → mxxMat20) × (Fin 8 → Fin 32 → mxxMat20) × (Fin 8 → Fin 128 → mxxMat23) × (Fin 8 → Fin 128 → mxxMat23) × (Fin 8 → Fin 32 → mxxMat23) × (Fin 8 → Fin 32 → mxxMat23) × (Fin 8 → Fin 128 → mxxMat26) × (Fin 8 → Fin 128 → mxxMat26) × (Fin 8 → Fin 32 → mxxMat26) × (Fin 8 → Fin 32 → mxxMat26) × (Fin 8 → Fin 128 → mxxMat29) × (Fin 8 → Fin 128 → mxxMat29) × (Fin 8 → Fin 32 → mxxMat29) × (Fin 8 → Fin 32 → mxxMat29) × (Fin 8 → Fin 128 → mxxMat32) × (Fin 8 → Fin 128 → mxxMat32) × (Fin 8 → Fin 32 → mxxMat32) × (Fin 8 → Fin 32 → mxxMat32) × (Fin 8 → Fin 128 → mxxMat35) × (Fin 8 → Fin 128 → mxxMat35) × (Fin 8 → Fin 32 → mxxMat35) × (Fin 8 → Fin 32 → mxxMat35) × (Fin 8 → Fin 128 → mxxMat38) × (Fin 8 → Fin 128 → mxxMat38) × (Fin 8 → Fin 32 → mxxMat38) × (Fin 8 → Fin 32 → mxxMat38) × (Fin 8 → Fin 128 → mxxMat41) × (Fin 8 → Fin 128 → mxxMat41) × (Fin 8 → Fin 32 → mxxMat41) × (Fin 8 → Fin 32 → mxxMat41) × (Fin 8 → Fin 128 → mxxMat44) × (Fin 8 → Fin 128 → mxxMat44) × (Fin 8 → Fin 32 → mxxMat44) × (Fin 8 → Fin 32 → mxxMat44) × (Fin 8 → Fin 128 → mxxMat47) × (Fin 8 → Fin 128 → mxxMat47) × (Fin 8 → Fin 32 → mxxMat47) × (Fin 8 → Fin 32 → mxxMat47) × (Fin 8 → Fin 128 → mxxMat50) × (Fin 8 → Fin 128 → mxxMat50) × (Fin 8 → Fin 32 → mxxMat50) × (Fin 8 → Fin 32 → mxxMat50) × (Fin 8 → Fin 128 → mxxMat53) × (Fin 8 → Fin 128 → mxxMat53) × (Fin 8 → Fin 32 → mxxMat53) × (Fin 8 → Fin 32 → mxxMat53) × (Fin 8 → Fin 128 → mxxMat56) × (Fin 8 → Fin 128 → mxxMat56) × (Fin 8 → Fin 32 → mxxMat56) × (Fin 8 → Fin 32 → mxxMat56) × (Fin 8 → Fin 128 → mxxMat59) × (Fin 8 → Fin 128 → mxxMat59) × (Fin 8 → Fin 32 → mxxMat59) × (Fin 8 → Fin 32 → mxxMat59) × (Fin 8 → Fin 128 → mxxMat62) × (Fin 8 → Fin 128 → mxxMat62) × (Fin 8 → Fin 32 → mxxMat62) × (Fin 8 → Fin 32 → mxxMat62) × (Fin 8 → Fin 128 → mxxMat65) × (Fin 8 → Fin 128 → mxxMat65) × (Fin 8 → Fin 32 → mxxMat65) × (Fin 8 → Fin 32 → mxxMat65) × (Fin 8 → Fin 128 → mxxMat68) × (Fin 8 → Fin 128 → mxxMat68) × (Fin 8 → Fin 32 → mxxMat68) × (Fin 8 → Fin 32 → mxxMat68) × (Fin 8 → Fin 128 → mxxMat71) × (Fin 8 → Fin 128 → mxxMat71) × (Fin 8 → Fin 32 → mxxMat71) × (Fin 8 → Fin 32 → mxxMat71) × (Fin 8 → Fin 128 → mxxMat74) × (Fin 8 → Fin 128 → mxxMat74) × (Fin 8 → Fin 32 → mxxMat74) × (Fin 8 → Fin 32 → mxxMat74) × (Fin 8 → Fin 128 → mxxMat77) × (Fin 8 → Fin 128 → mxxMat77) × (Fin 8 → Fin 32 → mxxMat77) × (Fin 8 → Fin 32 → mxxMat77) × (Fin 8 → Fin 128 → mxxMat80) × (Fin 8 → Fin 128 → mxxMat80) × (Fin 8 → Fin 32 → mxxMat80) × (Fin 8 → Fin 32 → mxxMat80) × (Fin 8 → Fin 128 → mxxMat83) × (Fin 8 → Fin 128 → mxxMat83) × (Fin 8 → Fin 32 → mxxMat83) × (Fin 8 → Fin 32 → mxxMat83) × (Fin 8 → Fin 128 → mxxMat86) × (Fin 8 → Fin 128 → mxxMat86) × (Fin 8 → Fin 32 → mxxMat86) × (Fin 8 → Fin 32 → mxxMat86) × (Fin 8 → Fin 128 → mxxMat89) × (Fin 8 → Fin 128 → mxxMat89) × (Fin 8 → Fin 32 → mxxMat89) × (Fin 8 → Fin 32 → mxxMat89) × (Fin 8 → Fin 128 → mxxMat91) × (Fin 8 → Fin 128 → mxxMat91) × (Fin 8 → Fin 32 → mxxMat91) × (Fin 8 → Fin 32 → mxxMat91) × (Fin 8 → Fin 128 → mxxMat94) × (Fin 8 → Fin 128 → mxxMat94) × (Fin 8 → Fin 32 → mxxMat94) × (Fin 8 → Fin 32 → mxxMat94) × (Fin 8 → Fin 128 → mxxMat97) × (Fin 8 → Fin 128 → mxxMat97) × (Fin 8 → Fin 32 → mxxMat97) × (Fin 8 → Fin 32 → mxxMat97) × (Fin 8 → Fin 128 → mxxMat100) × (Fin 8 → Fin 128 → mxxMat100) × (Fin 8 → Fin 32 → mxxMat100) × (Fin 8 → Fin 32 → mxxMat100) × (Fin 8 → Fin 128 → mxxMat103) × (Fin 8 → Fin 128 → mxxMat103) × (Fin 8 → Fin 32 → mxxMat103) × (Fin 8 → Fin 32 → mxxMat103) × (Fin 8 → Fin 128 → mxxMat106) × (Fin 8 → Fin 128 → mxxMat106) × (Fin 8 → Fin 32 → mxxMat106) × (Fin 8 → Fin 32 → mxxMat106) × Unit) : Prop :=
  ∃ witness : generatedRoot.Witness,
    generatedRoot.body backend hashModel tape path params inputs outputs witness


end Stage_sigma_encrypt
