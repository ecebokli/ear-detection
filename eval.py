import numpy as np
import matplotlib.pyplot as plt

def main():
    #number of ground truth bounding boxes in test dataset
    gt_num = 288

    #Intersections over unions and confidence scores for the three trained YOLOv3 models
    yolo_lr_es =[0.7468036389586635, 0.7796984620572044, 0.9281654077774176, 0.9065744694617782, 0, 0.8004535982618131, 0.8395510251087362, 0.6640686754243529, 0.8494285546379179, 0.7861060162500721, 0.7720991106867505, 0.7982269456873723, 0.9027444799643859, 0.9642664484636707, 0.8890486852607872, 0.7968101983163784, 0.8223812269153057, 0.8559988206203256, 0.8564505541949663, 0.7147029169429704, 0.7787913528511907, 0.9084062781492834, 0.8874948278535157, 0.8824566474839641, 0.8750103350894197, 0.8344461106663752, 0.8459761107368331, 0.7974277780136051, 0.8983596305083804, 0.8872262814897525, 0.825195150800522, 0.8646218163817436, 0.8410671569943532, 0.749196381262343, 0.8320943844628141, 0.7000082944734416, 0.9365147500377234, 0.6521513762446175, 0.8191470360107355, 0.6731714531824494, 0.8169080947119735, 0.7319825012821582, 0.7751005449491419, 0.8267994631353575, 0.8786129644759798, 0.8207576310698427, 0.826296184960419, 0.7562850778439668, 0.8814790770236357, 0.9006032089798732, 0.7732781693109477, 0.8317888510386686, 0.7585025564236662, 0.8809349638101377, 0.8259208444265477, 0.8254586245192931, 0.8602268031500311, 0.9105462003082873, 0.7899385208045254, 0.8473058574604245, 0.7446772157530924, 0.9158064099268568, 0.7082235463154678, 0.6310487167990884, 0.9022465122596232, 0.8189497584462988, 0.8326996938036648, 0.4378481599962769, 0.8676230601651919, 0.9128697807291907, 0.7849218164374472, 0.7893129664544005, 0.8698083239255509, 0.7922376219251512, 0.8694345996391608, 0.8315012708270146, 0.7956367162545215, 0.8268903255789175, 0.856327301164563, 0.7998517351086458, 0.8301699121774783, 0.8388350741090165, 0.8968831617376505, 0.8231368576409139, 0.8428405013836783, 0.6782495750325119, 0.7750815495448711, 0.8488581155478575, 0.8946589553915538, 0.7769111529316591, 0.8425608183881472, 0.7454258661540452, 0.7425038916358083, 0.7840199411887238, 0.8332148154908143, 0.7310345801646777, 0.9436121814097685, 0.8642277406674065, 0.8450641598855354, 0.7768790364970822, 0.8247194919121196, 0.6472588566506573, 0.8393306142764765, 0.8035108742800092, 0.8260194556721882, 0.8656149471500774, 0.8505658100700607, 0.8177053236225718, 0.885794396699301, 0.8070275767478593, 0.9125057152129482, 0.9009645268803216, 0.9336094898181029, 0.8672396758425268, 0.8349255819166437, 0.8836914007886966, 0.840824799699226, 0.8837508536649418, 0.8634896940712409, 0.8099585282452727, 0, 0.8883161768715855, 0.8246849204285793, 0.82643602662128, 0.7602181657037181, 0.8615213538522132, 0.8988389984841365, 0.9274724671293866, 0.8179629047009994, 0.7234568085152601, 0.8603732193488235, 0.843545516513711, 0.7898181650370103, 0.8555780659862612, 0, 0.7429416060350561, 0.7917123227333372, 0.88018271659884, 0.9187421120720999, 0.9101620153562797, 0.797566353315257, 0.8646855278885185, 0.8830008581812658, 0.8545802174236684, 0.9384863688961644, 0.9046902763551389, 0.8482413725082433, 0.9199792570248778, 0.7898253850524924, 0.751849387175484, 0.8854384750351046, 0.792335171872861, 0.9415344967266401, 0.8987577698020067, 0.846461164278914, 0.8158735297884994, 0.8865274475972859, 0.8505165309742766, 0.8031663163268425, 0.7048370067411557, 0.9421952194471721, 0.8363344827610689, 0.8590443662093902, 0.9011666820561055, 0.8518521734181816, 0, 0.8706509654001174, 0.8885663809232124, 0.8937559465442955, 0.8820182865519909, 0.8756402627189698, 0.8188945745843602, 0.8048743201840903, 0, 0.8513267457307454, 0.8279669883175109, 0.8858877315881395, 0.8880938405343245, 0.9632259317813929, 0.8563731944866341, 0.8643800157645056, 0.9332639574478135, 0.8438982693632842, 0.8789218505674806, 0.9213095205487236, 0.8466907710327641, 0.8803578658795321, 0.9177135205456903, 0.8427027461603047, 0.8177163182142881, 0.9207128050071609, 0.8413874392836074, 0.8946290871234963, 0.8275558988573091, 0.8601856917189735, 0.8307691821353282, 0.7645596288565689, 0.7855518982456643, 0.8219222698832167, 0.7565781450623168, 0.8700018590630393, 0.8059696166072827, 0.7720875236837184, 0.7102486241966736, 0.7880524119333937, 0.7591076993973113, 0.7958345236628315, 0.8532362475189346, 0.8612940596076526, 0.9078548810039719, 0.8029598582566172, 0.8797903969808194, 0.6681876721971837, 0.8353387061241536, 0.8678999926658935, 0.606348160893516, 0.7852735594035076, 0.7435572435333344, 0.582019928276055, 0.8091148898152636, 0.8623441759376386, 0.7670900794085073, 0.8074172100756578, 0.8458166487236827, 0.8492651680535935, 0.7319850145169385, 0.8881774232076224, 0.8426748298635248, 0.7763175717247551, 0.8354735771940786, 0.7693914061939714, 0.8203664051131481, 0.6990895998693661, 0.7219574625423377, 0, 0.8889172240089195, 0.8605773045762369, 0.8469784706562162, 0.7648266745916146, 0.8957226825497486, 0.7770149894653964, 0.7489185363530165, 0.8571520820957195, 0.8338862388623843, 0.9262365042452548, 0.6449868956143544, 0.7727358365304235, 0.8031452505578403, 0.9101739460410482, 0.852519496191693, 0.9309209165626631, 0.8454700230531575, 0.7250721827931391, 0.8239087921545303, 0.8463324530383454, 0.8998124890022599, 0.789314257477607, 0.7686956890539085, 0.8941048560552752, 0.8609389921716408, 0.8752601471686051, 0.8144276953226778, 0.8640594563944594, 0.8695241335360319, 0.7511693039909175, 0.8445596795728363,0.7816516025814731, 0.778695550754881, 0.8742022788552575, 0.7692988540216557, 0.783871598149032, 0, 0.7723985846053797, 0.7744447507511479, 0.8196407712245354, 0.7834752937691407, 0.8990022518628189, 0.8548529879105362, 0.7299708564532759, 0.8316253790813565, 0.9029182842527455, 0.7681878472702345, 0.8785205824141583, 0.6726907232323232]
    yolo_lr_es_scores = [0.99054897, 0.99536407, 0.9613749, 0.9821023, 0.9996376, 0.5144961, 0.98126626, 0.7902535, 0.97066015, 0.9994577, 0.9575331, 0.9864766, 0.9977576, 0.9969629, 0.98585427, 0.9983167, 0.95803785, 0.86218524, 0.9177365, 0.9422687, 0.931429, 0.98539, 0.99943316, 0.98217005, 0.9894336, 0.94813114, 0.9387961, 0.9829007, 0.9986594, 0.93519557, 0.99630314, 0.9327528, 0.6931553, 0.7171897, 0.8697516, 0.9994653, 0.5674416, 0.9922158, 0.97013175, 0.9173753, 0.9947848, 0.9845523, 0.9573113, 0.9871268, 0.98170984, 0.6059774, 0.9759757, 0.9719301, 0.99503696, 0.9922571, 0.9311782, 0.9984085, 0.999703, 0.99900717, 0.4311603, 0.9905381, 0.99782336, 0.9733763, 0.35029835, 0.93527377, 0.970763, 0.7443583, 0.9779867, 0.99658096, 0.99863404, 0.7587971, 0.34273487, 0.9951314, 0.99958134, 0.8139322, 0.969789, 0.78503764, 0.8947927, 0.8491608, 0.9993314, 0.75718147, 0.95674807, 0.9862421, 0.6800103, 0.96149784, 0.98657906, 0.9951789, 0.9825326, 0.9807425, 0.9976625, 0.9599203, 0.9952067, 0.99962974, 0.9982572, 0.9981125, 0.9268329, 0.93046093, 0.99731344, 0.996673, 0.99913365, 0.9984636, 0.9797862, 0.9939328, 0.9977292, 0.86608416, 0.9781083, 0.9941326, 0.851851, 0.98974794, 0.9980271, 0.99296856, 0.5553207, 0.5444035, 0.8541724, 0.9519443, 0.9960104, 0.99676144, 0.9155553, 0.99297744, 0.35171223, 0.6371136, 0.9975091, 0.7319596, 0.78406274, 0.60695213, 0.58616555, 0.8762508, 0.8267581, 0.95840573, 0.9217015, 0.9675522, 0.9991148, 0.99689806, 0.96183103, 0.97311205, 0.37566176, 0.99600726, 0.9913765, 0.96517503, 0.9395768, 0.99087584, 0.9823765, 0.9992478, 0.7180801, 0.999522, 0.96583676, 0.9979071, 0.99971366, 0.89394736, 0.9892963, 0.76874983, 0.99934185, 0.9387086, 0.9962058, 0.99827284, 0.96507907, 0.9852429, 0.98394674, 0.9661099, 0.98548055, 0.61887443, 0.8575851, 0.34742853, 0.99712807, 0.99530536, 0.9656918, 0.99918216, 0.99271154, 0.94767076, 0.9899627, 0.9979715, 0.99534535, 0.9978427, 0.94669855, 0.9723649, 0.99508965, 0.8633814, 0.9918468, 0.99485964, 0.99014837, 0.5740365, 0.9779146, 0.9978588, 0.99834895, 0.99558645, 0.991288, 0.9817171, 0.99962074, 0.9916037, 0.8309996, 0.9722415, 0.9993379, 0.8130723, 0.99647677, 0.9752703, 0.8587085, 0.98549145, 0.72035676, 0.43717635, 0.9609556, 0.72629005, 0.97138274, 0.9855177, 0.83913416, 0.95043683, 0.9910399, 0.33673215, 0.34580588, 0.9888252, 0.9841765, 0.8473533, 0.90054196, 0.9837049, 0.5454355, 0.99582344, 0.9968992, 0.80480486, 0.9721925, 0.896753, 0.4477108, 0.7988349, 0.94988555, 0.70251286, 0.45947027, 0.9784249, 0.9871646, 0.98152494, 0.9541503, 0.95674694, 0.9798314, 0.9827536, 0.98827374, 0.96632993, 0.598553, 0.86958104, 0.9991787, 0.9727449, 0.9383415, 0.9811764, 0.807036, 0.9450266, 0.88498384, 0.99684906, 0.8838827, 0.99939334, 0.5527639, 0.75144094, 0.92167735, 0.999624, 0.94733477, 0.99631876, 0.97897375, 0.9961335, 0.998859, 0.99935734, 0.99589235, 0.6521407, 0.8908003, 0.93872166, 0.97861147, 0.99682677, 0.9871433, 0.99176764, 0.9997375, 0.9688876, 0.6684348, 0.9690184, 0.9928068, 0.97676224, 0.9064261, 0.70797354, 0.4071559, 0.99782944, 0.9996206, 0.6165757, 0.9905748, 0.9588344, 0.82383054, 0.9602863, 0.8991897, 0.9860146, 0.9985188, 0.9605265]


    yolo_lr = [0.8562363600987852, 0.774691462996695, 0.7414484996860153, 0.8694327078447058, 0.6507869997299969, 0.7750313707550561, 0.8302430082663979, 0.659355096538561, 0.9111805662140302, 0.7766958651976152, 0.8471547275061215, 0.7805892277483574, 0.8953239582751394, 0.9322736216058747, 0.9138471741040121, 0.9827580425593203, 0.8457266492052001, 0.8351485224381417, 0.8793981092965502, 0.9225860565355977, 0.8884983727244031, 0.9254961847309244, 0.80849806376968, 0.8470311543885455, 0.8488556895175604, 0.9159541593253561, 0.8753359167235588, 0.760490727546238, 0.8674611438125421, 0.936119180454153, 0.8786206343407664, 0.8534074580632136, 0.8925789862440984, 0.709607137979412, 0.8203648756530428, 0.7993361048081105, 0.8432903219417868, 0.8098769905428179, 0.7872221372162963, 0.8068438982815758, 0.8593896044236179, 0.7389313936827511, 0.8107240799141853, 0.7825250588559188, 0.838941657772563, 0.8344693406504666, 0.769113707446064, 0.827280099441999, 0.8949650442038148, 0.8857668908206813, 0.8122557806529347, 0.8153011510237124, 0.8016985946916265, 0.875776623571488, 0.8555399543022983, 0.8887746250884818, 0.6907621372143129, 0.796421976875631, 0.9007286815006585, 0.8625429407398241, 0.832759959103439, 0.85496556560979, 0.8558219987206442, 0.8368853105357378, 0.7357991434357246, 0.8960307439386923, 0.8266710848960784, 0.7305033471932136, 0.3559046705376398, 0.8362665236808865, 0.8736422734173779, 0.8562212734695226, 0.7214408121626587, 0.9164375926782475, 0.8211154756768527, 0.8671571494479546, 0.8225451986673017, 0.8024951876400386, 0.8456160963392948, 0.7893716349059173, 0.7494470231254033, 0.8677905731080303, 0.9155104801945025, 0.8172267295207826, 0.9158463715600043, 0.8411728103713927, 0.8383443610898624, 0.8761450032611817, 0.9398682510225285, 0.9219321495850419, 0.7373496476988249, 0.9318287615670594, 0.7157605606807657, 0.6862555300050224, 0.8860934995137518, 0.7915393124514948, 0.6757731620619134, 0.940832155603641, 0.8330299046169067, 0.8371645582555879, 0.8594941729974661, 0.8102708199270787, 0.8356825927018613, 0.8459066385302061, 0.7751301465621916, 0.8461427744988373, 0.8518895370004727, 0.8976479317533708, 0.9409981448332296, 0.8440455139042086, 0.9123932616069927, 0.9327353867440616, 0.9177999321530902, 0.9215016729168806, 0.7725155986760159, 0.8995718864847119, 0.8417185960688827, 0.8335345121265031, 0.871377488701475, 0.8846953968514427, 0.8358615641459856, 0, 0.8336668550941749, 0.8725227195736247, 0.779857368316996, 0.7105698756832581, 0.7952518726319358, 0.7381059526621534, 0.802802347232585, 0.8908727094681208, 0.7515494490128692, 0.8950416780975592, 0.8292368142094724, 0.7864095424787388, 0.7991353907690175, 0, 0.7887006910215358, 0.8202239369773028, 0.7551232824662709, 0.8553296482187802, 0.9181275878715045, 0.8450932970051837, 0.8608047739529043, 0.8484121450209393, 0.80403183605336, 0.8687461405707416, 0.9191569608438942, 0.8619041409965943, 0.7433689948697986, 0.8005425546419422, 0.6736186360014357, 0.8778925497941948, 0.8354802535766834, 0.8607562460458541, 0.8716569839154972, 0.8805836000338716, 0.8495979750600803, 0.909731787624486, 0.8302835867401683, 0.7652469445206371, 0.815756942986463, 0.8021238113414699, 0.9188604384951891, 0.8645943227905968, 0.8017016387922609, 0.8936696803145453, 0.8893575337384446, 0.9494889976583917, 0.9052560009406424, 0.8205730169002354, 0.9395156721625166, 0.8540270602475293, 0.7758394349181461, 0.7072620477049819, 0.7871243754356128, 0.8629875771548139, 0.7460234760565851, 0.8566976036835927, 0.8085084561379862, 0.8429937778902802, 0.8410286518457554, 0.8829992736783644, 0.8817721277079617, 0.88713454782866, 0.9672780917574936, 0.9106248400921948, 0.7760536935228676, 0.9080049141875961, 0.8832489175578712, 0.8163280692178643, 0.7452098585682337, 0.8963729954407014, 0.9329395597338774, 0.9494591287400723, 0.8672702933676358, 0.8713788856176146, 0.9460869915496753, 0.6832604594955202, 0.7831536754286708, 0, 0.8037907084092226, 0.7933542704990345, 0.9073533393181907, 0.8098815220527043, 0.828170710439795, 0.7072246504805432, 0.8359435356589695, 0.7603500868299284, 0.8428652307389062, 0.7424690752962323, 0.8592481117649827, 0.6290947354674323, 0.8770064043036792, 0.913519839328051, 0.7999291115852339, 0.9198275681672019, 0.9514634642187483, 0.8137862174291725, 0.8238497540108366, 0.806966778645917, 0.789029286501969, 0.8546225812896855, 0.8298600262097872, 0.8126866809978273, 0.8397745375889127, 0.917443104839415, 0.890830106975702, 0.7161841474443865, 0.8540448110947284, 0.8418537064404303, 0.7312508259057404, 0.8525230440407802, 0.7644430354902468, 0.804801385726935, 0.7094466267201875, 0.747044033015259, 0, 0.9188782948974532, 0.8027803922127872, 0.860244766000226, 0.7954292216165173, 0.8819890945884874, 0.6598450737954411, 0.6946389412848113, 0.7704268240090852, 0.7737091598707596, 0.8911894646076968, 0.8761646831710028, 0.8130719995074042, 0.7874744718393126, 0.8652846979997068, 0.80886725640621, 0.8894783976216026, 0.898569441473769, 0.8544653660240663, 0.7904203641939609, 0.9306086099991102, 0.8009935833883839, 0.7838541364741617, 0.7852064117001286, 0.9008454128747886, 0.8736785958818694, 0.9185586150194615, 0.8483439067899051, 0.8745063642475236, 0.8779856163103521, 0.7714928811186931, 0, 0.8583780125497094, 0.8878006128588646, 0.8236838084832123, 0.8272962741167869, 0.7203762394253334, 0.8929812014603291, 0.8383383630332427, 0, 0.716001186113709, 0.8662861501968373, 0.8431048384878517, 0.9046377080542923, 0.8838052473071204, 0.8412633363339793, 0.7773805287747342, 0.8842140828166438, 0.9225313662796116, 0.8670330716280094, 0.91416776769851, 0.791765202528781]
    yolo_lr_scores = [0.98929584, 0.9969983, 0.98512065, 0.9825307, 0.62662935, 0.99948853, 0.85075426, 0.97816783, 0.89882237, 0.9555808, 0.99888706, 0.94660383, 0.9845796, 0.99488914, 0.99855626, 0.9731981, 0.9993866, 0.93290687, 0.8959653, 0.9098706, 0.91123915, 0.9070074, 0.9281663, 0.9986679, 0.9750132, 0.9960303, 0.95193297, 0.9670266, 0.9867834, 0.99944204, 0.9953827, 0.99852484, 0.93553185, 0.9608824, 0.6752372, 0.97682744, 0.999039, 0.7784319, 0.9874541, 0.95018744, 0.9815812, 0.9982365, 0.9935783, 0.96619946, 0.98991627, 0.9812899, 0.73855937, 0.9094298, 0.96271884, 0.37850752, 0.99325335, 0.9867775, 0.89335877, 0.9971845, 0.99935937, 0.99910283, 0.5736795, 0.9970777, 0.99901986, 0.98365057, 0.67574656, 0.9814061, 0.9939533, 0.9387949, 0.9625909, 0.99488735, 0.997487, 0.892047, 0.5474651, 0.9972946, 0.9998165, 0.9611177, 0.90675706, 0.8108297, 0.96174306, 0.95412093, 0.99811065, 0.81827563, 0.9794961, 0.99799234, 0.93447506, 0.99531984, 0.9910751, 0.98249364, 0.9975289, 0.96767735, 0.9987017, 0.97516227, 0.9987557, 0.9995463, 0.99938226, 0.9991415, 0.93282145, 0.9783887, 0.9968802, 0.9984831, 0.9985068, 0.9929019, 0.9886824, 0.99892724, 0.9953616, 0.6862252, 0.97574615, 0.99398124, 0.95285946, 0.9414191, 0.99886674, 0.99800235, 0.7632503, 0.5780219, 0.62271637, 0.9866033, 0.99601424, 0.98420703, 0.891387, 0.9817442, 0.8103455, 0.94987226, 0.99854773, 0.8985139, 0.90675205, 0.7827681, 0.9815379, 0.9615268, 0.72624654, 0.9641896, 0.5398372, 0.98717445, 0.99768996, 0.99742186, 0.9867066, 0.9882257, 0.48730147, 0.9980035, 0.99885446, 0.94917583, 0.8060985, 0.9976168, 0.9928854, 0.9986865, 0.6555919, 0.9996319, 0.9588078, 0.999264, 0.9998634, 0.9765718, 0.9828695, 0.9714451, 0.9994852, 0.8977641, 0.99847233, 0.99888724, 0.93081033, 0.9768464, 0.9638458, 0.9927713, 0.9825619, 0.491009, 0.90986234, 0.86691797, 0.99704546, 0.98729, 0.93892866, 0.9996043, 0.828995, 0.9957347, 0.99100894, 0.98964787, 0.99804574, 0.99630326, 0.98616177, 0.94551265, 0.9785708, 0.31089467, 0.9843043, 0.76519394, 0.9969116, 0.99321365, 0.99182445, 0.7733787, 0.9906278, 0.99675906, 0.99938786, 0.9962206, 0.9900875, 0.98221767, 0.99936014, 0.9835294, 0.8896631, 0.8355621, 0.9993647, 0.7592481, 0.9997324, 0.9818857, 0.92365795, 0.99684215, 0.8947759, 0.9848766, 0.6775042, 0.33852053, 0.9144841, 0.98155844, 0.6574388, 0.99256086, 0.98130345, 0.8475033, 0.7360633, 0.9871708, 0.992548, 0.9015854, 0.98414385, 0.9954489, 0.7166449, 0.99181944, 0.9963145, 0.8335065, 0.9892047, 0.88946366, 0.35301644, 0.9082727, 0.9789643, 0.80587304, 0.6142062, 0.9903855, 0.99282074, 0.977713, 0.9779571, 0.9421985, 0.992018, 0.97199565, 0.99440855, 0.96857494, 0.5243485, 0.9199697, 0.99933463, 0.8336704, 0.970919, 0.9958745, 0.8671189, 0.8580684, 0.94089925, 0.99667937, 0.90202993, 0.999184, 0.5268125, 0.97717965, 0.9950932, 0.9996047, 0.96035933, 0.99952817, 0.9989157, 0.9934524, 0.9992572, 0.99962085, 0.99786776, 0.8147092, 0.95837295, 0.9857214, 0.9196094, 0.9963527, 0.9948947, 0.9755083, 0.99975175, 0.5268308, 0.9954242, 0.8864482, 0.9951461, 0.99165934, 0.98132837, 0.9591439, 0.9285232, 0.7589003, 0.99716425, 0.9994635, 0.7975098, 0.99625784, 0.96241397, 0.91439205, 0.97020894, 0.78301615, 0.9680551, 0.9969811, 0.991082]
    
    
    yolo = [0.8367889976420607, 0.8463796965856981, 0.8699571068372828, 0.89813950262034, 0.8237689158213102, 0.9185911371051431, 0.8231634594145084, 0.7833623458644671, 0.8298628343959102, 0.794112440172472, 0.8406901489248375, 0.880665603152525, 0.9016372815708036, 0.8863832831516516, 0.765088950061675, 0.8563002161383498, 0.8981095335941234, 0.9570632807020388, 0.8432189495112761, 0.7523124782561228, 0.8938840550530283, 0.8833553385539431, 0.8271303159907698, 0.8850562403829622, 0.8194912971660203, 0.9235327657486548, 0.8412553999218542, 0.9172308921174319, 0.9286500069001226, 0.8006685754788924, 0.8831490068423133, 0.827317402889357, 0.7912545236678348, 0.7874745372494165, 0.8389627256601402, 0.8800651529315922, 0.9444959826822718, 0.8753234093070541, 0.9277289586732621, 0.8163096511832624, 0.8526239133404486, 0.7847516412131784, 0.8461306768118326, 0.8875171581682433, 0.8617130580971243, 0.8607586225329985, 0.845878891707108, 0.827180412225527, 0.8337543739171479, 0.8440976744205031, 0.8135632980078397, 0.8886194390396517, 0.8521821929031618, 0.7878168022290296, 0.8308790860982198, 0.8954856827153713, 0.8783425373186435, 0.8279007434874682, 0.9028457420726967, 0.7965619872400589, 0.7772564965918205, 0.8637671138504522, 0.8573316098661641, 0.8003285554863705, 0.9133749811466506, 0.8749767325685969, 0.8647931583614957, 0.8578750295817322, 0.44093789376604803, 0.8562415310645732, 0.9513998410379649, 0.9072353458457344, 0.7740215259165387, 0.7736807602789331, 0.8972502913369579, 0.9288443810191864, 0.8741837774377619, 0.8175410963258718, 0.8601153412720555, 0.8945737911203128, 0.8447154954227866, 0.8665869217425516, 0.874595228161437, 0.8825766593359543, 0.8791936183733036, 0.7855275922237562, 0.8619172506474487, 0.8557705901350703, 0.9150762358278223, 0.9107921859917816, 0.8406123861391184, 0.8948577700877005, 0.8385349538836047, 0.8303694271788328, 0.8468535185856784, 0.9163209485851767, 0.9277969394443888, 0.9192318556342565, 0.9197266885823836, 0.891704674976666, 0.9095117323072581, 0.8868352946152233, 0.8645264501989631, 0.8666106380760464, 0.8294355510523558, 0.8277659286394828, 0.9291394165925746, 0.9220890731631497, 0.6586452789667524, 0.721464020997172, 0.8842717961715577, 0.8994199258357907, 0.8442559571533201, 0.9546863077809863, 0.7457744182497525, 0.860046694181177, 0.9120364283912374, 0.7161836081233612, 0.960760000938381, 0.8623987825154376, 0.8512387209021571, 0.9247771403960778, 0.8402949932497439, 0.8629932138556043, 0.7795489272254312, 0, 0.8242691352586157, 0.9173640393550452, 0.8453898794734301, 0.7272399638962546, 0.9426035865303463, 0.8597880132126703, 0.9253970520701349, 0.947230649517148, 0, 0.7713462873501845, 0.7186121774485035, 0.8999828839853292, 0.9007784143602031, 0.8917922556803033, 0.8985573559453948, 0.8409159614413396, 0.9548788689395746, 0.8681840411948322, 0.8813267986038749, 0.7858633326240719, 0.8939489353438953, 0.9078345536508428, 0.8348051155459774, 0.8587979417057048, 0.8344117168857952, 0.8448741989683639, 0.8926206596310045, 0.8472430443594666, 0.8464301661811837, 0.8533794661024957, 0.9535898880614972, 0.7237598057222419, 0.8045496860271345, 0.8603334431929982, 0.7575704002257725, 0.8420031185372024, 0.8776003484978531, 0.932725619991043, 0.9308069014012923, 0.7099205278646866, 0.9058074886558485, 0.8928778766659247, 0.9591683099275842, 0.8912857880122351, 0.882291895444914, 0.8510233440997746, 0.7964126442588347, 0.8286631953820572, 0.900124984382254, 0.8368476820166623, 0.9326875431774668, 0.8613447426923968, 0.9050913202561819, 0.8512618678916967, 0.8466836008428107, 0.9600154960339188, 0.889127479581908, 0.7938141401218057, 0.8350421459879833, 0.7572163257453706, 0.9237787172006565, 0.7583854222589714, 0.7682186131590886, 0.8916072584119524, 0.8315870544048356, 0.9216699881022563, 0.9096590455324369, 0.8486320651584479, 0.8484209294942429, 0.8470612795194475, 0.8825343724602601, 0.9000551109868631, 0.8375990878197661, 0.7540984843489548, 0.8197951260958476, 0.9416551951907343, 0.8779103286523632, 0.8441142096294446, 0.8743675605223659, 0.7751644299850536, 0.8055796099780037, 0.8935744450044245, 0.8707664703572069, 0.878055287348062, 0.8202646869929598, 0.889949653553877, 0.762203922320469, 0.8988653465577862, 0.8906518317329686, 0.655502658968868, 0.8737239826383406, 0.880971724341214, 0.9018199165962802, 0.8073076639593932, 0.7971745060532133, 0.8499463355061594, 0.6914429957344813, 0.793459939727701, 0.9263461408657863, 0.9065741455826333, 0.851653447352053, 0.8116939964881478, 0.8641153039046848, 0.9275222247774491, 0.909350145480528, 0.8189598880310258, 0.7261368108082678, 0.7075120286565595, 0, 0.8770334257609855, 0.8945647341953559, 0.8453653349953391, 0.8231317648657439, 0.9239537806385428, 0.7546317731441744, 0.8419367220799988, 0.8640920135806506, 0.8351119171165696, 0.9517536162030177, 0.6081533814914226, 0.8488050591136502, 0.8424742691275227, 0.9573378857315525, 0.8269381582877311, 0.8454376015843883, 0.9694664095353351, 0.8754901596461377, 0.8992205340826231, 0.8637849207345643, 0.8497491874269992, 0.8936226179233443, 0.80665246859722, 0.8156119543397423, 0.8623931183306407, 0.8846582934240733, 0.748014400035313, 0.7536146265276448, 0.8900729620190875, 0.8682068435235195, 0.8245884414610123, 0.876086064035318, 0.9095724424331472, 0.7688409193750873, 0.9517836665906607, 0.880214429284531, 0.9086714888927092, 0.7927825884292817, 0.7921098397233888, 0.8367586982407395, 0.9148112124580062, 0.9469009162516672, 0.7803354475272102, 0.7611211023877074, 0.7549631757873424, 0.7964073794177533, 0.9121911484519464, 0.8101244644279618, 0.9522339443914942, 0.9340953246900974, 0.8452151007394262]
    yolo_scores = [0.9800375, 0.99920464, 0.9980228, 0.9891806, 0.843177, 0.999106, 0.81066275, 0.98887765, 0.8020839, 0.85965633, 0.9990127, 0.992897, 0.99949545, 0.9976573, 0.99745697, 0.69188225, 0.999256, 0.8243778, 0.98038507, 0.95170844, 0.9794164, 0.94476587, 0.94889796, 0.9981426, 0.9962908, 0.9981814, 0.9966925, 0.99927384, 0.8458836, 0.9976736, 0.98504966, 0.9922662, 0.938634, 0.98831594, 0.9728439, 0.908698, 0.9976833, 0.97434366, 0.9959667, 0.9537092, 0.99191254, 0.99565935, 0.95307916, 0.9837871, 0.74262714, 0.6309386, 0.8208099, 0.96157664, 0.91298604, 0.9779224, 0.9956238, 0.990229, 0.8862578, 0.98555624, 0.99937075, 0.9995565, 0.70496434, 0.9973029, 0.99935377, 0.9961204, 0.9642922, 0.99579316, 0.9976162, 0.8886252, 0.9376037, 0.9290422, 0.98846173, 0.9943308, 0.37674722, 0.981936, 0.9996301, 0.8970996, 0.9661181, 0.9166641, 0.57519937, 0.81479234, 0.99822176, 0.5033402, 0.99648106, 0.9894824, 0.9928181, 0.9987184, 0.95184565, 0.9959475, 0.99537295, 0.9933804, 0.98995113, 0.87428504, 0.9988691, 0.9994765, 0.9992769, 0.9990833, 0.9956059, 0.996149, 0.9996565, 0.9984626, 0.99250054, 0.99606836, 0.98807055, 0.99925286, 0.99835634, 0.865398, 0.99473375, 0.9977679, 0.90871805, 0.96548295, 0.9992706, 0.99862677, 0.3622237, 0.5739001, 0.99384034, 0.9958283, 0.99369985, 0.9690178, 0.32566342, 0.9737824, 0.9792374, 0.99181634, 0.99269086, 0.39487082, 0.99517465, 0.75212216, 0.9573931, 0.7441217, 0.9957151, 0.99055475, 0.9985853, 0.999842, 0.9918411, 0.9977379, 0.3961015, 0.9961724, 0.99700063, 0.9053411, 0.697679, 0.9976663, 0.99543816, 0.9992928, 0.9546985, 0.99981856, 0.9817247, 0.9968423, 0.99990654, 0.8972826, 0.88207525, 0.9899293, 0.9995923, 0.72656506, 0.99936396, 0.9993018, 0.8890583, 0.9893445, 0.9931397, 0.99296063, 0.95826054, 0.6624865, 0.8027039, 0.99679315, 0.9965022, 0.9832371, 0.9694302, 0.99984515, 0.8536701, 0.9993355, 0.9622759, 0.9943502, 0.9955625, 0.9866631, 0.9913917, 0.5476865, 0.98891777, 0.9271005, 0.99387145, 0.99517804, 0.9926141, 0.7149877, 0.98804826, 0.62136316, 0.9905617, 0.9967505, 0.999273, 0.9870514, 0.97724056, 0.996589, 0.99746495, 0.98005414, 0.99962115, 0.99967134, 0.50998396, 0.9996786, 0.98168504, 0.97465587, 0.998223, 0.8255795, 0.9813866, 0.98437905, 0.63516104, 0.90497476, 0.9849681, 0.99551135, 0.7096011, 0.9933585, 0.7576469, 0.6618665, 0.90568835, 0.996172, 0.9758199, 0.99493825, 0.9327086, 0.9420124, 0.6014878, 0.99823916, 0.993885, 0.5815328, 0.81099945, 0.8224703, 0.98303485, 0.99277353, 0.99301064, 0.99327564, 0.88956386, 0.9956712, 0.99755055, 0.9944926, 0.7786022, 0.78493524, 0.65240014, 0.99706423, 0.99330294, 0.39697567, 0.60660344, 0.9941154, 0.99950993, 0.72869134, 0.94729406, 0.94809216, 0.976937, 0.9760336, 0.94282335, 0.9921182, 0.96674204, 0.99934065, 0.6254838, 0.43891343, 0.9898647, 0.99979234, 0.98766553, 0.99814206, 0.9978147, 0.9531678, 0.99890614, 0.9994636, 0.9958848, 0.97106093, 0.9986042, 0.9997633, 0.6532721, 0.9968529, 0.99596924, 0.9943748, 0.9997813, 0.9738223, 0.9739771, 0.9946005, 0.86920863, 0.97883284, 0.99646217, 0.99816644, 0.987163, 0.3742521, 0.45634654, 0.8804215, 0.9992599, 0.99970704, 0.8909027, 0.99262536, 0.9923227, 0.81604755, 0.9804028, 0.67584765, 0.9795798, 0.9956914, 0.9993321]
   
    yolo_versions = [yolo_lr_es, yolo_lr, yolo]
    yolo_versions_scores = [yolo_lr_es_scores, yolo_lr_scores, yolo_scores]
    fig, axs = plt.subplots(3)
    plt_count = 0
    for i in range(0, len(yolo_versions)):
        #sort IOUs by confidence score
        yolo = [x for y, x in sorted(zip(yolo_versions_scores[i], yolo_versions[i]))]
        
        precisions = []
        recalls = []
        thresh = 0.5
        TP = 0
        FP = 0
        total_detections = 0
        length = len(yolo)
        for iou in yolo:     
            if iou > thresh:
                TP += 1
            else:
                FP += 1
            total_detections += 1
            precisions.append(TP/total_detections)
            recalls.append(TP/gt_num)
        
      
        axs[plt_count].plot(recalls, precisions)
        f_measure = 2 * ((recalls[-1] * precisions[-1]) / (precisions[-1] + recalls[-1]))
        print("F-measure:", f_measure)
        print("Precision:", precisions[-1])
        print("Recall:", recalls[-1])
        plt_count += 1
    plt.show()


if __name__ == '__main__':
    main()
