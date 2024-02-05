import numpy as np
from ml.plot.newer.column_strategy_sim.plotlatex_lib import plot_list_latex
from ml.plot.newer.column_strategy_sim.plotlatex_lib import plot_list
from ml.plot.newer.column_strategy_sim.plotlatex_lib import plot_integral
from ml.plot.newer.column_strategy_sim.plotlatex_lib import plot_integral_latex
from ml.plot.newer.column_strategy_sim.plotlatex_lib import plot_outperform
from ml.plot.newer.column_strategy_sim.plotlatex_lib import plot_outperform_latex


labels_all = [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 78, 88, 98, 108, 118, 128, 138, 148, 158, 168, 178, 188, 198, 208, 218, 228, 238, 248, 258, 268, 278, 288, 298, 308, 318, 328, 338, 348, 358, 368, 378, 388, 398, 408, 418, 428, 438, 448, 458, 468, 478, 488, 498, 508, 518, 528, 538, 548, 558, 568, 578, 588, 598, 608, 618, 628, 638, 648, 658, 668, 678, 688, 698, 708, 718, 728, 738, 748, 758, 768, 778, 788, 798, 808, 818, 828, 838, 848, 858, 868, 878, 888, 898, 908, 918, 928, 938, 948, 958, 968, 978, 988, 998, 1008, 1018, 1028, 1038]

char_unigram_sim = []
char_unigram_sim.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.043517602802151945, 0.04631947217211431, 0.0477633239787417, 0.05083253438205718, 0.06265164954221444, 0.07594373487637654, 0.10812937990256008, 0.1400010990507087, 0.1411633623042388, 0.14185089479227567, 0.14255067681597597, 0.14346252525427905, 0.1438569638755137, 0.24785465828885594, 0.28609723841482015, 0.330266499251367, 0.3807189546949078, 0.40746589910864534, 0.4270509116096912, 0.4391837977790357, 0.4598659693833259, 0.48639479794628154, 0.5135439494548205, 0.5210565726236057, 0.53718369817327, 0.5475148431015543, 0.5573811453692403, 0.5798530519800347, 0.5873882228704665, 0.6049157636170926, 0.6185108635531524, 0.632583652719464, 0.6581796614380673, 0.6534953676825372, 0.6742421015482811, 0.6856731394718405, 0.6903037718999928, 0.7012695871787089, 0.711672772919282, 0.7285609519572016, 0.7503845925439561, 0.7708777467600958, 0.7835214852249317, 0.8032675851242198, 0.8136011146279347, 0.8236143738857311, 0.833456430104399, 0.8427245923760779, 0.8461452256359012, 0.8630944804696339, 0.8758568405157747, 0.8794174759765067, 0.8817566599505688, 0.8890487915870994, 0.8948656433054707, 0.9053065237408955, 0.9067839704007824, 0.9067839704007824, 0.911578778444414, 0.911578778444414, 0.911578778444414, 0.9144729219767258, 0.9154501054017464, 0.9179634531464769, 0.9237152203073521, 0.9257141622749689, 0.9283969781442932, 0.9283969781442932, 0.92919447145464, 0.9318042112143793, 0.9348006597102818, 0.9386778255816113, 0.9408513561289599, 0.9443218362392812, 0.9443218362392812, 0.9463629669974397, 0.9492345653663266, 0.9510356638506305, 0.956117787041814, 0.9574599084928404, 0.9577750013802426, 0.9577750013802426, 0.9577750013802426, 0.9601072402399906, 0.9601072402399906, 0.9601072402399906, 0.9632958159238221, 0.9654021348806928, 0.9671805566418502, 0.9671805566418502, 0.9695764843856928, 0.9695764843856928, 0.9735532146821535])
average_char_unigram_sim = list(np.mean(np.matrix(char_unigram_sim), axis=0).A1)

char_bigram_sim = []
char_bigram_sim.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.03627924352817077, 0.04006130064007123, 0.0479271979300633, 0.04894648187243182, 0.06167572133571869, 0.07023213787861171, 0.09781969024342155, 0.13223406031209922, 0.13309977982901464, 0.1337405356223031, 0.13417907434310267, 0.1361412604343192, 0.13873855889612782, 0.2793457563925738, 0.2940844802641428, 0.31589145994078005, 0.3437975211026523, 0.38613202631956944, 0.426627800702005, 0.4707841023972687, 0.4959316150406556, 0.5231979502503452, 0.552967994405315, 0.5658699643392009, 0.598622919728142, 0.6156882693963036, 0.6279330148398317, 0.6364278647045533, 0.6623400413927284, 0.669095233521217, 0.6808722156562812, 0.695480018994232, 0.7081768610989154, 0.7120710243594018, 0.7180068800600836, 0.7247130489030218, 0.7435684398552146, 0.7509762376146459, 0.7643568847881933, 0.7725960551937499, 0.7816496300596502, 0.8026529219119501, 0.8123216405946685, 0.8226658137900905, 0.8267317179784148, 0.8324771030354576, 0.8394461145291764, 0.8477688071792817, 0.8561111441791805, 0.8663977287753231, 0.8673295511772438, 0.8679827573200336, 0.8713477115637673, 0.8757074368033988, 0.8836766845402788, 0.8928074886473949, 0.8928074886473949, 0.8969260748188839, 0.9026097604314123, 0.900814456577254, 0.9058242298785484, 0.90718644751891, 0.9109322595513429, 0.9144804554433439, 0.9158740881951516, 0.9246907743136145, 0.9273669844506779, 0.9317093705472754, 0.9320524433744246, 0.9333512217851572, 0.9400581239845487, 0.9514493798764526, 0.9514493798764526, 0.955126616922526, 0.9565274668884948, 0.959644751132474, 0.96250301413531, 0.9640061599956841, 0.9643853015021199, 0.9670902010225182, 0.9670902010225182, 0.9686270403565732, 0.973865039639211, 0.973865039639211, 0.9774650917799862, 0.9787715083643889, 0.9787715083643889, 0.9787715083643889, 0.9787715083643889, 0.9787715083643889, 0.9800650407026973, 0.9800650407026973, 0.9839337448075647])
average_char_bigram_sim = list(np.mean(np.matrix(char_bigram_sim), axis=0).A1)

word_unigram_sim = []
word_unigram_sim.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05399119556574751, 0.054107177833200924, 0.05771439318417946, 0.05766293757096741, 0.07296285935992827, 0.07390193247666921, 0.0851329969825082, 0.08188199403168359, 0.08175547652267177, 0.08193120472384853, 0.08178011137536498, 0.0826011228879573, 0.08277440955462173, 0.11477301350292804, 0.12218954482946617, 0.12177656821397242, 0.12782786767201712, 0.1325045521739387, 0.13505428638112588, 0.13982091761875648, 0.1448054761957105, 0.1526218487487871, 0.15777230341195653, 0.16259360656698724, 0.16553554733485015, 0.16899999278495703, 0.17566657684009485, 0.1791546136235437, 0.18880155248002334, 0.1927997587330015, 0.20324744216412377, 0.2103890710161617, 0.21469763066234324, 0.2155234376874763, 0.21778380307714743, 0.22114064605949713, 0.22855593175023042, 0.23052251407687568, 0.24214680713322584, 0.25001809033350275, 0.2619850313819911, 0.2674824719874598, 0.2758523068464573, 0.2886626051207582, 0.29893058586987975, 0.30536356596580894, 0.30961646846441104, 0.3103843785289741, 0.3195055951415585, 0.3236439347502147, 0.33461241421486826, 0.344289671462206, 0.3600115979424783, 0.36474026011446786, 0.3714426102167213, 0.37346427387156467, 0.37700628531761715, 0.3842229243381011, 0.388621797424321, 0.3955210615389286, 0.4008047627737487, 0.40783088246160304, 0.41691370441347686, 0.42033687178308876, 0.4251747961063529, 0.4320503780718724, 0.43775315871299386, 0.4458420768862618, 0.45141203282925063, 0.45636132861503337, 0.46188023426446, 0.46351967930981813, 0.4679661684252541, 0.47045659574893417, 0.4777540235473299, 0.47987681114476155, 0.48514021205254493, 0.48684113363244047, 0.48899764262096496, 0.49315411821009525, 0.4972950858452972, 0.5007151453507241, 0.5013150718283781, 0.505337268951237, 0.5108238296736908, 0.5108678681346105, 0.5137525510516525, 0.5192050750270816, 0.5211716110987327, 0.5243333635891115, 0.5258478073322663, 0.5259584280412187, 0.5275724933181716])
average_word_unigram_sim = list(np.mean(np.matrix(word_unigram_sim), axis=0).A1)

metadata_sim = []
metadata_sim.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.06452882356197942, 0.07359858320092778, 0.07975544380275477, 0.08823180033946036, 0.07378210302416008, 0.08990398518543001, 0.12257307928336711, 0.15448021959291572, 0.15533123936906787, 0.1576197064292544, 0.1581492822111757, 0.1587269006809109, 0.15907538335000684, 0.2822753429461935, 0.3707353814514233, 0.4080802331776008, 0.44888404034050133, 0.47614659580640784, 0.517028429971534, 0.5397774090592502, 0.5522813652202262, 0.5876148894981389, 0.6151500423651094, 0.6263716345804365, 0.6453387351556483, 0.6572301557792122, 0.6798703849870389, 0.692277820523662, 0.7101351223472467, 0.718700342660365, 0.7309331363933416, 0.7543987347282279, 0.7661443895665324, 0.7886664214054382, 0.7963972920957028, 0.8034776542979131, 0.8086809080383326, 0.8090374775579461, 0.818100729109088, 0.8240324719534853, 0.8292420393175515, 0.8344818725890664, 0.8368880981304437, 0.8376330380650423, 0.8412808892807047, 0.8444557691719691, 0.8494923086307985, 0.8621698341724293, 0.8634203506070988, 0.870982893123896, 0.877786478049414, 0.8829110743961686, 0.8865431073704915, 0.888704443456749, 0.8931747836531259, 0.8987217132603288, 0.900647960314795, 0.9079722540280398, 0.9088331030531667, 0.9133529097753694, 0.913399518596288, 0.9186149626372305, 0.9192104626005321, 0.9213030984021042, 0.9218466836218994, 0.9219604696642923, 0.9237712614212434, 0.9235253122896976, 0.9251018368578334, 0.9253253614059286, 0.9271359012246393, 0.9298762541051054, 0.9260167243103437, 0.9319290544079731, 0.9314306597166254, 0.9328741681426205, 0.9333599147585117, 0.9335531195263904, 0.9361177346684835, 0.9361177346684835, 0.9390365353081476, 0.9402370951850637, 0.9404515679395423, 0.9401184308969318, 0.942309631521057, 0.9413712368330744, 0.9437555073631418, 0.9488141941957376, 0.9502247793220754, 0.9509561901814717, 0.9509561901814717, 0.9509561901814717, 0.9510524618058824])
average_metadata_sim = list(np.mean(np.matrix(metadata_sim), axis=0).A1)

metadata_char_unigrams_sim = []
metadata_char_unigrams_sim.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.044857332637578615, 0.05662233258418616, 0.0642949778866368, 0.06664898909402753, 0.06665370915477775, 0.0772816368715474, 0.10936508954335603, 0.14568148637618317, 0.1463533753128166, 0.1480140298587729, 0.1483425131965938, 0.14862242221387587, 0.14978652618388633, 0.30929437737462845, 0.3309329897411624, 0.384595861529624, 0.42812409779209604, 0.46439608551607864, 0.4890517667567627, 0.5230540349079116, 0.5647015686100992, 0.589838698425196, 0.6193318644799407, 0.6443620533373531, 0.6509912106829165, 0.6677159788606052, 0.6732687071208256, 0.6883365880394229, 0.7053069821711484, 0.7154222166065335, 0.7345369307381042, 0.735977373869396, 0.7368039094428388, 0.7415838832463553, 0.749762890609291, 0.7592027570642216, 0.772961667012409, 0.783847583582652, 0.7978724325638074, 0.8180352093863797, 0.8247291136112439, 0.8389998537006195, 0.8518416920049319, 0.8582438997472457, 0.8728306471426824, 0.874660127860684, 0.8795493845198546, 0.8951300610676938, 0.903883395494528, 0.9177761761930036, 0.923599824952049, 0.9304209211605766, 0.9387895681766603, 0.9437868594098532, 0.9468163283423403, 0.9506960131604375, 0.9540854301345586, 0.9541950517899741, 0.9561140357531125, 0.9644673713416434, 0.9650942934119175, 0.9673752664606091, 0.9678904400799642, 0.969771584288701, 0.9707168164265937, 0.9707168164265937, 0.971438780298006, 0.9736831383370486, 0.975891973678414, 0.975891973678414, 0.976099696063556, 0.976099696063556, 0.9763993689257674, 0.9783152976712856, 0.9783152976712856, 0.9783152976712856, 0.9783152976712856, 0.9791073768792063, 0.9816038588415354, 0.9816038588415354, 0.9816038588415354, 0.9818124348391096, 0.9818124348391096, 0.9818124348391096, 0.9818124348391096, 0.9839187537959801, 0.9839187537959801, 0.9839187537959801, 0.9838090947515866, 0.984525624897804, 0.984525624897804, 0.984525624897804, 0.984525624897804])
average_metadata_char_unigrams_sim = list(np.mean(np.matrix(metadata_char_unigrams_sim), axis=0).A1)



metadata_lstm = []
metadata_lstm.append([0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, 0.0351372381010472, 0.038974359960347954, 0.042999860288064254, 0.047216369406770715, 0.0595369119672125, 0.0688810701270657, 0.10095483973215472, 0.13761323317800897, 0.13791859512496663, 0.13894326180857813, 0.13994820311122153, 0.14062581367202404, 0.14130391106188875, 0.2124937080303264, 0.25709873427702934, 0.306059221197831, 0.3296400451603819, 0.39541596288284336, 0.4199781184733141, 0.47347410768295684, 0.4863999784580121, 0.5071546923208845, 0.5396237733677763, 0.5587896054237286, 0.5804471026099078, 0.6113460871668147, 0.6239369989877618, 0.6382459307912145, 0.6506948045487128, 0.668472738762341, 0.6892447669866987, 0.7046594911947872, 0.7078388420920124, 0.7185027036559918, 0.7241950880822822, 0.7330594392718053, 0.7358656812767567, 0.7472741827869865, 0.7542260675074087, 0.7669421407831127, 0.7750455096561807, 0.7860935592339985, 0.7928659835862822, 0.8012448996561797, 0.8151032104205458, 0.8196734832580315, 0.8287594166041734, 0.8436992217750419, 0.8565351479308643, 0.8644509245141213, 0.8739912901672942, 0.8769622031594194, 0.881699114038579, 0.8887603888626987, 0.8935559092929749, 0.8973568244581365, 0.8993105993093836, 0.8995301227015627, 0.9062839190111205, 0.9063999187897469, 0.9108881983611348, 0.9165406756604095, 0.9196775837903719, 0.9233703451909188, 0.9258871433879676, 0.9282708758696054, 0.9300009005061792, 0.930962863885517, 0.9361082681923982, 0.9398283892135032, 0.940530658816704, 0.9423213096843662, 0.9443183345304403, 0.9504523269760465, 0.9550295799765989, 0.956398823154373, 0.957905110982816, 0.9607821708648115, 0.963196287725045, 0.9639814236417598, 0.9656001443862741, 0.9717453108597851, 0.9717453108597851, 0.9726754241072493, 0.9754971438544475, 0.979135129781221, 0.9799372261611523, 0.981264824784609, 0.9814733480194018, 0.9827759819680757, 0.985394628200362, 0.9859801633541727, 0.9868699300209292])
average_metadata_lstm = list(np.mean(np.matrix(metadata_lstm), axis=0).A1)


metadata_error_corr = []
metadata_error_corr.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.045310844926372475, 0.05254358254577969, 0.05755593328172466, 0.06390711444838998, 0.06845102567219524, 0.08121533263640202, 0.11313108669514076, 0.149682441503678, 0.15106575503526432, 0.15190448197972292, 0.15258060132700205, 0.1530210841723744, 0.1549571500029006, 0.3050234551028387, 0.3511357880877969, 0.381976799703264, 0.41134272752216317, 0.4465634758284082, 0.4788285494603784, 0.5292704783322795, 0.5469690028573682, 0.5707627522345027, 0.6057809742389455, 0.6411086607146644, 0.6559901038388741, 0.6778157585391482, 0.6922149664717165, 0.7226910108726236, 0.7401497713281454, 0.7562804369615821, 0.767909966461126, 0.7706345508092928, 0.7742030749931599, 0.7663467185688653, 0.7708361872531377, 0.7731023039639252, 0.7828971057491516, 0.7937039790428779, 0.8002191332105445, 0.8160332873673841, 0.8334929084865786, 0.8392440850225675, 0.8473673405619871, 0.8546246119269114, 0.8603643537004935, 0.8661114481522375, 0.896391010015145, 0.904391948492537, 0.916403432143699, 0.9244121654302451, 0.9292893338508268, 0.934536873724688, 0.9403059149280786, 0.9403059149280786, 0.9404212291026779, 0.9444004613682706, 0.9444004613682706, 0.9459507151753771, 0.9459507151753771, 0.9459507151753771, 0.9459507151753771, 0.9459507151753771, 0.946838104742012, 0.9480408821722085, 0.9499465559619946, 0.9513231789603573, 0.9533897070180004, 0.9533897070180004, 0.9606794266016895, 0.9630069843279074, 0.9630069843279074, 0.964805449812437, 0.964805449812437, 0.964805449812437, 0.964805449812437, 0.964805449812437, 0.9652073609715014, 0.9693563623749041, 0.971969564221002, 0.9781505616246603, 0.9781505616246603, 0.9781505616246603, 0.9781505616246603, 0.9787801121577374, 0.9787801121577374, 0.9793729975332314, 0.9793729975332314, 0.9839020587232881, 0.9839020587232881, 0.9839020587232881, 0.9858817074999017, 0.9858817074999017, 0.9885947744754897])
average_metadata_error_corr = list(np.mean(np.matrix(metadata_error_corr), axis=0).A1)




ranges = [labels_all,
		  labels_all,
          labels_all,
          labels_all,
          labels_all,
		  labels_all,
		  labels_all
		  ]
list = [average_metadata_char_unigrams_sim,
		average_metadata_error_corr,
		average_metadata_lstm,
		average_char_unigram_sim,
		average_char_bigram_sim,
		average_metadata_sim,
		average_word_unigram_sim

		]
names = [
		 "Char unigrams + Metadata (UM)",
		 "Char unigrams + Metadata + Error correlation (UME)",
		 "LSTM + Metadata (LM)",
		 "Char unigrams (U)",
		 "Char unigrams + bigrams (UB)",
		 "Metadata (M)",
		 "Word unigrams (WU)"
		 ]



plot_list_latex(ranges, list, names, "Hospital", x_max=600)
plot_list(ranges, list, names, "Hospital", x_max=600, end_of_round=238)
plot_integral(ranges, list, names, "Hospital", x_max=800, x_min=238)
#plot_integral_latex(ranges, list, names, "Hospital", x_max=800, sorted=False)
#plot_outperform(ranges, list, names, "Hospital", 1.0, x_max=800)




