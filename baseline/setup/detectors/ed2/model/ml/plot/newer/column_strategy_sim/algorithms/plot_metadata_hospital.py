import numpy as np
from ml.plot.newer.column_strategy_sim.plotlatex_lib import plot_list_latex
from ml.plot.newer.column_strategy_sim.plotlatex_lib import plot_list
from ml.plot.newer.column_strategy_sim.plotlatex_lib import plot_integral
from ml.plot.newer.column_strategy_sim.plotlatex_lib import plot_integral_latex
from ml.plot.newer.column_strategy_sim.plotlatex_lib import plot_outperform
from ml.plot.newer.column_strategy_sim.plotlatex_lib import plot_outperform_latex


labels_all = [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 78, 88, 98, 108, 118, 128, 138, 148, 158, 168, 178, 188, 198, 208, 218, 228, 238, 248, 258, 268, 278, 288, 298, 308, 318, 328, 338, 348, 358, 368, 378, 388, 398, 408, 418, 428, 438, 448, 458, 468, 478, 488, 498, 508, 518, 528, 538, 548, 558, 568, 578, 588, 598, 608, 618, 628, 638, 648, 658, 668, 678, 688, 698, 708, 718, 728, 738, 748, 758, 768, 778, 788, 798, 808, 818]

svm_sim = []
svm_sim.append([0.024081264909229287, 0.02905305542563778, 0.034435541489074316, 0.03959061381151784, 0.04026680194086891, 0.0399897157783886, 0.04412207353958928, 0.04542024228621504, 0.04453797103935726, 0.0444045837158296, 0.044990481345100426, 0.04511856214619202, 0.045032928374616726, 0.046241254391729596, 0.04549729835304375, 0.045747821745046324, 0.04580081252399491, 0.04656470964832686, 0.047036004194577215, 0.046739739689973164, 0.04589352991928646, 0.0460948221036805, 0.046810045419554934, 0.044843642171567846, 0.04369942315070092, 0.04766316894991234, 0.04808619483647476, 0.04781922097861392, 0.04928479675390757, 0.04963000959276529, 0.04800436736879716, 0.05001248087834098, 0.05049803636234076, 0.05091625579636777, 0.052199548355727786, 0.05144841891405381, 0.05144591970608222, 0.050609795519166355, 0.050185817431844096, 0.050185817431844096, 0.04978174491027557, 0.04974592685835845, 0.04974592685835845, 0.04924468517844297, 0.04924468517844297, 0.04924468517844297, 0.0490008176482529, 0.04926870837307492, 0.049181001646572776, 0.049181001646572776, 0.048959370073545054, 0.04886304600814882, 0.04869897030944891, 0.048701917765077755, 0.048701917765077755, 0.04526668224964174, 0.0454483316974393, 0.0454483316974393, 0.0454483316974393, 0.0454483316974393, 0.0455485254494416, 0.0455485254494416, 0.0455485254494416, 0.04463536911832861, 0.04350263480736579, 0.04350263480736579, 0.04350263480736579, 0.04770762823706355, 0.043549571735720154, 0.043549571735720154, 0.043503094073143414, 0.043503094073143414, 0.043503094073143414, 0.04345831191899294, 0.04352440545025784, 0.04352440545025784, 0.04273343013721788, 0.04273343013721788, 0.04280422551430424, 0.03906578406155034, 0.04293837233420365, 0.039016732437381754, 0.039016732437381754, 0.039016732437381754, 0.039956860237077535, 0.039016732437381754, 0.039016732437381754, 0.039016732437381754, 0.0390197105756034, 0.03909700944394797, 0.039184716170450114, 0.03986532824536503])
average_svm_sim = list(np.mean(np.matrix(svm_sim), axis=0).A1)

bayes_sim = []
bayes_sim.append([0.03289684469689279, 0.039289811216727984, 0.04754197667245656, 0.052197202578155175, 0.05417065466520358, 0.05791050585569616, 0.06262483526061068, 0.06575538169346115, 0.06605387896105933, 0.06535477262172962, 0.06524407278540986, 0.06573776931248154, 0.06562777408894697, 0.06727792981928632, 0.06590697460090045, 0.06605901077581894, 0.06619038740900461, 0.06625374095506506, 0.06779144246934374, 0.06810695223425484, 0.0685212965416368, 0.06926961601291308, 0.06927017170150622, 0.0684325278802668, 0.06751832429475221, 0.06706718338970954, 0.06917654137991315, 0.06980401064237081, 0.07021465860867628, 0.07147401550060917, 0.07090011699326625, 0.07376411245906431, 0.07526417513906916, 0.07658916185399688, 0.07735919352518161, 0.07955697800904446, 0.08079675485563033, 0.08178188710626372, 0.08290818876966524, 0.08398054379445018, 0.08545142334954647, 0.08639918223050608, 0.08794575890457898, 0.0890480971844919, 0.08943743579179894, 0.09073492841947828, 0.09173653071708661, 0.09192148401262408, 0.09355045558428286, 0.09492946638941892, 0.09593826185532627, 0.09833287876703609, 0.09942044074136536, 0.10079170230774044, 0.10184164607428406, 0.10271966027014885, 0.10396771840209511, 0.10534375702957641, 0.10642033580811709, 0.107712493583462, 0.10871822332093395, 0.10975471525737898, 0.11102884055280582, 0.11121633587608179, 0.11213155396867844, 0.1129648531967024, 0.11443512303348242, 0.11562090788876753, 0.11640681109226128, 0.11700049067466183, 0.11774191176467416, 0.11838684733901157, 0.11896049556230037, 0.12008234396611157, 0.1226438937030904, 0.1240985073155666, 0.12449545621605233, 0.12520408855594853, 0.12648788344114692, 0.1278751352789493, 0.12882708978617693, 0.12936348194483213, 0.1294945173956069, 0.13005056740691878, 0.1301338221850382, 0.1309392037043371, 0.1312430478964352, 0.13148674073181227, 0.131842661993028, 0.13221184112384238, 0.1325181874636677, 0.13288370343152348])
average_bayes_sim = list(np.mean(np.matrix(bayes_sim), axis=0).A1)

trees_sim = []
#trees_sim.append()
#average_trees_sim = list(np.mean(np.matrix(trees_sim), axis=0).A1)



ranges = [#labels_all,
		  labels_all,
          labels_all
		  ]
list = [#average_trees_sim,
		average_svm_sim,
		average_bayes_sim

		]
names = [
		 #"Gradient Tree Boosting",
		 "Linear SVM",
		 "Naive Bayes"
		 ]




plot_list_latex(ranges, list, names, "Hospital", x_max=600)
plot_list(ranges, list, names, "Hospital", x_max=600, end_of_round=238)
#plot_integral(ranges, list, names, "Hospital", x_max=800, x_min=238)
#plot_integral_latex(ranges, list, names, "Hospital", x_max=800, sorted=False)
#plot_outperform(ranges, list, names, "Hospital", 1.0, x_max=800)




