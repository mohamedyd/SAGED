import numpy as np

from ml.datasets.hospital.HospitalHoloClean import HospitalHoloClean
from ml.plot.old.user_effort_all_potential.PlotterLatex import PlotterLatex

data = HospitalHoloClean()


fscore_metadata_no_svd_more_data = []
fscore_metadata_no_svd_more_data.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.08727272727272724, 0.1640488656195462, 0.18211920529801318, 0.2762951334379905, 0.26878612716763, 0.2743988684582744, 0.36461126005361927, 0.43388960205391525, 0.2321016166281755, 0.25824800910125145, 0.284593837535014, 0.23195661243220692, 0.23195661243220692, 0.23783783783783785, 0.23783783783783785, 0.2377343438372557, 0.2377343438372557, 0.3175921120913337, 0.3175921120913337, 0.3247676325861127, 0.3533333333333334, 0.35851183765501693, 0.4282828282828283, 0.7235494880546075, 0.762954796030871, 0.7643171806167401, 0.814977973568282, 0.8357541899441341, 0.8357541899441341, 0.8357541899441341, 0.8791208791208792, 0.880351262349067, 0.880351262349067, 0.8815789473684211, 0.8815789473684211, 0.8815789473684211, 0.9137380191693292, 0.9171974522292994, 0.9319371727748692, 0.9319371727748692, 0.9319371727748692, 0.9319371727748692, 0.9319371727748692, 0.9319371727748692, 0.9495365602471678, 0.970736629667003, 0.970736629667003, 0.970736629667003, 0.970736629667003, 0.970736629667003, 0.970736629667003, 0.970736629667003, 0.970736629667003, 0.970736629667003, 0.970736629667003, 0.970736629667003, 0.970736629667003, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
fscore_metadata_no_svd_more_data.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.09090909090909087, 0.11674347158218124, 0.12735166425470332, 0.20718232044198892, 0.19732034104750304, 0.2064825930372149, 0.286697247706422, 0.34551495016611294, 0.20346320346320348, 0.22933333333333336, 0.23604309500489715, 0.26262626262626265, 0.26262626262626265, 0.2667946257197697, 0.2667946257197697, 0.2715231788079471, 0.2715231788079471, 0.2721669037458512, 0.29253428136109705, 0.5454545454545455, 0.6197757390417942, 0.6197757390417942, 0.6620553359683796, 0.7152847152847153, 0.7453416149068324, 0.7812500000000001, 0.7812500000000001, 0.7854166666666668, 0.8220858895705522, 0.8970747562296859, 0.8970747562296859, 0.900647948164147, 0.8977871443624869, 0.9012605042016807, 0.9012605042016807, 0.9012605042016807, 0.9012605042016807, 0.9012605042016807, 0.9058577405857742, 0.9058577405857742, 0.9058577405857742, 0.9395833333333333, 0.9395833333333333, 0.9395833333333333, 0.9395833333333333, 0.9675456389452333, 0.9696356275303644, 0.9768844221105527, 0.9768844221105527, 0.9841269841269841, 0.9841269841269841, 0.9841269841269841, 0.9841269841269841, 0.9841269841269841, 0.9841269841269841, 0.9841269841269841, 0.9841269841269841, 0.9841269841269841, 0.9841269841269841, 0.9841269841269841, 0.9841269841269841, 0.9841269841269841, 0.9841269841269841, 0.9841269841269841, 0.9841269841269841, 0.9841269841269841, 0.9841269841269841, 0.9841269841269841, 0.9841269841269841, 0.9841269841269841, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
fscore_metadata_no_svd_more_data.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0910683012259194, 0.1165644171779141, 0.1253731343283582, 0.21337126600284492, 0.19612590799031476, 0.25934579439252337, 0.33519553072625696, 0.39612486544671693, 0.22784810126582283, 0.24908806670140698, 0.2724935732647815, 0.2964071856287426, 0.2964071856287426, 0.30233714569865744, 0.30233714569865744, 0.3047711781888997, 0.3047711781888997, 0.30883078441045886, 0.34336099585062246, 0.6666666666666667, 0.6666666666666667, 0.6680080482897385, 0.6958333333333333, 0.743099787685775, 0.7484143763213531, 0.7484143763213531, 0.7805907172995782, 0.8167860798362334, 0.8167860798362334, 0.8934782608695653, 0.8934782608695653, 0.8934782608695653, 0.8934782608695653, 0.9041980624327234, 0.9041980624327234, 0.9004237288135594, 0.9096638655462186, 0.9204665959703076, 0.9373695198329853, 0.9373695198329853, 0.9373695198329853, 0.9373695198329853, 0.9373695198329853, 0.9373695198329853, 0.9395833333333333, 0.9395833333333333, 0.9612244897959183, 0.9612244897959183, 0.9612244897959183, 0.9612244897959183, 0.9612244897959183, 0.9675456389452333, 0.9675456389452333, 0.9675456389452333, 0.9675456389452333, 0.9675456389452333, 0.9675456389452333, 0.9675456389452333, 0.9675456389452333, 0.9789368104312939, 0.9789368104312939, 0.9789368104312939, 0.9789368104312939, 0.9789368104312939, 0.9789368104312939, 0.9789368104312939, 0.9789368104312939, 0.9789368104312939, 0.9789368104312939, 0.9789368104312939, 0.9789368104312939, 0.9789368104312939, 0.9789368104312939, 0.9789368104312939, 0.9789368104312939, 0.9789368104312939, 0.9789368104312939])
fscore_metadata_no_svd_more_data.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.09122807017543856, 0.11746031746031746, 0.1308411214953271, 0.2222222222222222, 0.24147727272727268, 0.31335149863760214, 0.39844760672703744, 0.4582814445828144, 0.24283305227655988, 0.2671840354767184, 0.2927362097214637, 0.2588543944031483, 0.2588543944031483, 0.2649237472766885, 0.2649237472766885, 0.26770969143850504, 0.26770969143850504, 0.33244539158231223, 0.6688102893890676, 0.6688102893890676, 0.7053763440860216, 0.7053763440860216, 0.7432150313152401, 0.8004338394793927, 0.8004338394793927, 0.8346456692913387, 0.8744493392070485, 0.8744493392070485, 0.8862144420131292, 0.888646288209607, 0.888646288209607, 0.888646288209607, 0.8934782608695653, 0.8934782608695653, 0.8934782608695653, 0.9272918861959959, 0.9272918861959959, 0.9272918861959959, 0.9272918861959959, 0.9272918861959959, 0.9272918861959959, 0.9272918861959959, 0.9272918861959959, 0.9272918861959959, 0.9272918861959959, 0.9272918861959959, 0.9308176100628931, 0.9308176100628931, 0.9308176100628931, 0.9308176100628931, 0.9308176100628931, 0.9308176100628931, 0.9516957862281603, 0.9516957862281603, 0.9516957862281603, 0.9516957862281603, 0.9516957862281603, 0.9516957862281603, 0.9516957862281603, 0.9516957862281603, 0.9516957862281603, 0.9516957862281603, 0.9516957862281603, 0.9516957862281603, 0.9516957862281603, 0.9516957862281603, 0.9516957862281603, 0.9516957862281603, 0.9516957862281603, 0.9516957862281603, 0.9516957862281603, 0.9516957862281603, 0.9516957862281603, 0.98, 1.0, 1.0, 1.0])
fscore_metadata_no_svd_more_data.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.09574468085106379, 0.16723549488054604, 0.19095477386934667, 0.2829888712241653, 0.27920227920227914, 0.34133333333333327, 0.42332065906210387, 0.46798029556650245, 0.24860022396416573, 0.27373068432671077, 0.29907558455682437, 0.3276414087513341, 0.3276414087513341, 0.3338649654439128, 0.3338649654439128, 0.33598304186539485, 0.33598304186539485, 0.33598304186539485, 0.35768811341330425, 0.354758345789736, 0.38325281803542677, 0.38888888888888895, 0.38952536824877254, 0.7777777777777779, 0.7777777777777779, 0.8035914702581369, 0.8062709966405375, 0.8228571428571428, 0.8228571428571428, 0.8558659217877095, 0.8571428571428572, 0.8710033076074973, 0.8722466960352423, 0.8893709327548808, 0.9232386961093586, 0.9232386961093586, 0.9232386961093586, 0.9232386961093586, 0.9340314136125655, 0.9340314136125655, 0.9340314136125655, 0.9340314136125655, 0.9340314136125655, 0.9340314136125655, 0.9408099688473521, 0.9473684210526315, 0.9473684210526315, 0.9473684210526315, 0.9473684210526315, 0.9473684210526315, 0.9473684210526315, 0.9613034623217922, 0.9613034623217922, 0.9613034623217922, 0.9613034623217922, 0.9613034623217922, 0.9613034623217922, 0.9613034623217922, 0.9613034623217922, 0.9613034623217922, 0.9613034623217922, 0.9613034623217922, 0.9613034623217922, 0.9613034623217922, 0.9613034623217922, 0.9830508474576273, 0.9830508474576273, 0.9830508474576273, 0.9830508474576273, 0.9830508474576273, 0.9830508474576273, 0.9830508474576273, 0.9830508474576273, 0.9830508474576273, 0.9830508474576273, 0.9830508474576273, 0.9830508474576273])
fscore_metadata_no_svd_more_data.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.09090909090909087, 0.13013698630136983, 0.1333333333333333, 0.2182410423452768, 0.19346405228758168, 0.2572509457755359, 0.33894230769230765, 0.4041570438799076, 0.22476813966175674, 0.25053763440860216, 0.27556968733439324, 0.30290456431535273, 0.30290456431535273, 0.29910269192422734, 0.29910269192422734, 0.30338733431516945, 0.30338733431516945, 0.3430962343096235, 0.34471886495007886, 0.38259000537345517, 0.7710583153347733, 0.7710583153347733, 0.7710583153347733, 0.8058361391694725, 0.8134078212290503, 0.8134078212290503, 0.8262370540851554, 0.8262370540851554, 0.8262370540851554, 0.8289322617680827, 0.8448471121177803, 0.8448471121177803, 0.8565022421524664, 0.893709327548807, 0.8963282937365011, 0.8963282937365011, 0.8963282937365011, 0.7885304659498209, 0.9298429319371728, 0.9298429319371728, 0.9298429319371728, 0.9484536082474226, 0.9484536082474226, 0.9484536082474226, 0.9484536082474226, 0.9484536082474226, 0.9484536082474226, 0.9484536082474226, 0.9676113360323887, 0.9676113360323887, 0.9676113360323887, 0.9676113360323887, 0.9676113360323887, 0.9676113360323887, 0.9717741935483871, 0.9717741935483871, 0.9717741935483871, 0.9717741935483871, 0.9717741935483871, 0.9717741935483871, 0.9717741935483871, 0.9717741935483871, 0.9717741935483871, 0.9717741935483871, 0.9717741935483871, 0.9717741935483871, 0.9717741935483871, 0.9717741935483871, 0.9717741935483871, 0.9717741935483871, 0.9717741935483871, 0.9717741935483871, 0.9717741935483871, 0.9717741935483871, 0.9717741935483871, 0.9717741935483871, 0.9717741935483871])
fscore_metadata_no_svd_more_data.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.09154929577464785, 0.16271186440677962, 0.18800648298217174, 0.2799999999999999, 0.2691218130311614, 0.2786206896551724, 0.36649214659685864, 0.42821158690176325, 0.22734946539110862, 0.25388026607538805, 0.2796286182413982, 0.23139820114472615, 0.23139820114472615, 0.23603960396039603, 0.23603960396039603, 0.2405956112852665, 0.2405956112852665, 0.250394944707741, 0.3304258594150847, 0.3304258594150847, 0.3585200625325691, 0.3602290473711609, 0.38302277432712223, 0.40165631469979307, 0.41801990570979586, 0.41801990570979586, 0.4391008886565605, 0.4484590860786398, 0.4484590860786398, 0.45174973488865333, 0.4605405405405406, 0.46481178396072026, 0.9112299465240642, 0.9250263991552271, 0.9250263991552271, 0.9250263991552271, 0.9569672131147541, 0.9617706237424547, 0.9617706237424547, 0.9617706237424547, 0.9617706237424547, 0.980980980980981, 0.980980980980981, 0.9960552268244576, 0.9960552268244576, 0.9960552268244576, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
fscore_metadata_no_svd_more_data.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.09090909090909087, 0.1677852348993288, 0.16320474777448069, 0.2489391796322489, 0.179372197309417, 0.21145374449339202, 0.2708688245315162, 0.31395348837209297, 0.20712277413308341, 0.22952336881073576, 0.25137111517367466, 0.27657657657657664, 0.27657657657657664, 0.2804494382022472, 0.2804494382022472, 0.2893692104102338, 0.2893692104102338, 0.36286472148541116, 0.36286472148541116, 0.6930091185410334, 0.6930091185410334, 0.6930091185410334, 0.7328094302554028, 0.7328094302554028, 0.7886435331230284, 0.7886435331230284, 0.8231441048034934, 0.8231441048034934, 0.8358531317494601, 0.8358531317494601, 0.8408602150537635, 0.8408602150537635, 0.8823529411764707, 0.8823529411764707, 0.8898488120950324, 0.8898488120950324, 0.9075630252100841, 0.9075630252100841, 0.9075630252100841, 0.9075630252100841, 0.9075630252100841, 0.9075630252100841, 0.9382716049382716, 0.9382716049382716, 0.9382716049382716, 0.9538461538461538, 0.9758551307847082, 0.9768844221105527, 0.9768844221105527, 0.9768844221105527, 0.9768844221105527, 0.9768844221105527, 0.9789368104312939, 0.9789368104312939, 0.9789368104312939, 0.9789368104312939, 0.9789368104312939, 0.9789368104312939, 0.9789368104312939, 0.9789368104312939, 0.9789368104312939, 0.9789368104312939, 0.9789368104312939, 0.9789368104312939, 0.9789368104312939, 0.9789368104312939, 0.9789368104312939, 0.9789368104312939, 0.9789368104312939, 0.9789368104312939, 0.9789368104312939, 0.9789368104312939, 0.9789368104312939, 0.9789368104312939, 0.9789368104312939, 0.9789368104312939, 0.9789368104312939])
fscore_metadata_no_svd_more_data.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.09090909090909087, 0.13694267515923564, 0.14992503748125935, 0.2270114942528735, 0.19954648526077093, 0.20560747663551399, 0.2754491017964071, 0.3304347826086956, 0.22332233223322331, 0.24837310195227766, 0.27365045430251206, 0.23452768729641696, 0.23452768729641696, 0.24311183144246357, 0.24311183144246357, 0.2507961783439491, 0.2507961783439491, 0.35775127768313464, 0.5215231788079471, 0.6217516843118384, 0.6554121151936445, 0.6867119301648885, 0.6867119301648885, 0.6867119301648885, 0.7390029325513198, 0.7439143135345668, 0.8217213114754098, 0.8229273285568066, 0.8636836628511967, 0.8636836628511967, 0.8636836628511967, 0.900647948164147, 0.9018338727076591, 0.9204665959703076, 0.9204665959703076, 0.9204665959703076, 0.9496402877697842, 0.9496402877697842, 0.9496402877697842, 0.9496402877697842, 0.9496402877697842, 0.9592668024439919, 0.9592668024439919, 0.9622833843017329, 0.9737903225806451, 0.9737903225806451, 0.9758551307847082, 0.9789368104312939, 0.9789368104312939, 0.9789368104312939, 0.9789368104312939, 0.9789368104312939, 0.9789368104312939, 0.9789368104312939, 0.9789368104312939, 0.9789368104312939, 0.9789368104312939, 0.9789368104312939, 0.9789368104312939, 0.9789368104312939, 0.9789368104312939, 0.9789368104312939, 0.9789368104312939, 0.9789368104312939, 0.9789368104312939, 0.9789368104312939, 0.9789368104312939, 0.9789368104312939, 0.9789368104312939, 0.9789368104312939, 0.9789368104312939, 0.9789368104312939, 0.9789368104312939, 0.9789368104312939, 0.9789368104312939, 0.9789368104312939, 0.9789368104312939])
fscore_metadata_no_svd_more_data.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.09090909090909087, 0.12218649517684883, 0.1301939058171745, 0.21192052980132448, 0.17884615384615382, 0.22990654205607472, 0.2916291629162916, 0.3423580786026201, 0.21673003802281368, 0.2360131640808651, 0.25812441968430827, 0.2325056433408578, 0.2325056433408578, 0.23276501111934772, 0.23276501111934772, 0.23398741206960394, 0.23398741206960394, 0.2694512694512695, 0.33502538071065996, 0.35369774919614155, 0.6327900287631831, 0.6948989412897018, 0.7568710359408034, 0.7568710359408034, 0.8043478260869567, 0.8061002178649238, 0.8061002178649238, 0.8425821064552662, 0.8478015783540023, 0.8478015783540023, 0.8744493392070485, 0.8867102396514162, 0.8963282937365011, 0.8963282937365011, 0.8653648509763618, 0.9001074113856069, 0.9001074113856069, 0.9001074113856069, 0.9001074113856069, 0.9001074113856069, 0.919831223628692, 0.919831223628692, 0.919831223628692, 0.919831223628692, 0.919831223628692, 0.919831223628692, 0.919831223628692, 0.919831223628692, 0.9288702928870293, 0.9528688524590164, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98])


nadeef_fscore = 0.05564746578432847
openrefine_fscore = 1.0


dboost_models = ["Gaussian", "Histogram", "Mixture"]
dboost_sizes = [200, 400, 600, 800]
dboost_fscore_all = [
                        # Gaussian
                        [
                            [0.576354679803, 0.382590005373, 0.382590005373, 0.382590005373, 0.576354679803],
                            [0.576354679803, 0.688362919132, 0.729957805907, 0.748917748918, 0.729957805907],
                            [0.729957805907, 0.748917748918, 0.748917748918, 0.748917748918, 0.748917748918],
                            [0.748917748918, 0.748917748918, 0.748917748918, 0.748917748918, 0.729957805907]
                        ],
                        # Histogram
                        [
                            [0.414327202323, 0.3899543379, 0.478968031408, 0.279025624802, 0.478968031408],
                            [0.778188539741, 0.812741312741, 0.478968031408, 0.625557206538, 0.337745687926],
                            [0.812741312741, 0.812741312741, 0.812741312741, 0.812741312741, 0.812741312741],
                            [0.812741312741, 0.812741312741, 0.812741312741, 0.812741312741, 0.812741312741]
                        ],
                        # Mixture
                        [
                            [0.0266940451745, 0.0341637010676, 0.0266940451745, 0.0266940451745, 0.049766718507],
                            [0.0341637010676, 0.0341637010676, 0.0341637010676, 0.0341637010676, 0.0266940451745],
                            [0.0341637010676, 0.0341637010676, 0.0341637010676, 0.0341637010676, 0.0266940451745],
                            [0.0341637010676, 0.0341637010676, 0.0341637010676, 0.0266940451745, 0.0341637010676]
                        ]
                    ]

dboost_matrix_f = np.array(dboost_fscore_all)
dboost_avg_f = np.mean(dboost_matrix_f, axis = 2)

label_potential = [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 78, 88, 98, 108, 118, 128, 138, 148, 158, 168, 178, 188, 198, 208, 218, 228, 238, 248, 258, 268, 278, 288, 298, 308, 318, 328, 338, 348, 358, 368, 378, 388, 398, 408, 418, 428, 438, 448, 458, 468, 478, 488, 498, 508, 518, 528, 538, 548, 558, 568, 578, 588, 598, 608, 618, 628, 638, 648, 658, 668, 678, 688, 698, 708, 718, 728, 738, 748, 758, 768, 778, 788, 798, 808, 818, 828, 838, 848, 858, 868, 878, 888, 898, 908, 918]


PlotterLatex(data, label_potential, fscore_metadata_no_svd_more_data,
         dboost_models, dboost_sizes, dboost_avg_f,
         nadeef_fscore,
         openrefine_fscore,
         None, xmax=600, filename="Hospital")
