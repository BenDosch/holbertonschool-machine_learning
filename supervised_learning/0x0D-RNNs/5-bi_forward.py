#!/usr/bin/env python3
"""Module containing the class BidirectionalCell that represents a
bidirectional cell of an RNN."""

import numpy as np


class BidirectionalCell():
    """Class that represents a bidirectional cell of an RNN."""

    def __init__(self, i, h, o):
        """Class constructor that creates the public instance attributes
        Whf, Whb, Wy, bhf, bhb, by that represent the weights and biases of the
        cell. Whf and bhf are for the hidden states in the forward direction.
        Whb and bhb are for the hidden states in the backward direction. Wy and
        by are for the outputs. The weights are initialized using a random
        normal distribution in the order listed previously. The weights will be
        used on the right side for matrix multiplication. The biases are
        initialized as zeros.

        Args:
            i (int): The dimensionality of the data.
            h (int): The dimensionality of the hidden state.
            o (int): The dimensionality of the outputs.
        """
        pass

    def forward(self, h_prev, x_t):
        """Public instance method that calculates the hidden state in the
        forward direction for one time step.

        Args:
            h_prev (numpy.ndarray): Tensor of shape (m, h) containing the
                previous hidden state, where m is the batch size for the data
                and h is the dimensionality of the hidden state.
            x_t (numpy.ndarray): Tensor of shape (m, i) that contains the data
                input for the cell, where m is the batch size for the data and
                i is the dimensionality of the data.

        Returns:
            h_next(numpy.ndarray): Tensor of shape ( ,  ) contaiing the next
                hidden state.
        """
        pass


# Testing
if __name__ == "__main__":
    np.random.seed(5)
    bi_cell = BidirectionalCell(10, 15, 5)
    print("Whf:", bi_cell.Whf)
    print("Whb:", bi_cell.Whb)
    print("Wy:", bi_cell.Wy)
    print("bhf:", bi_cell.bhf)
    print("bhb:", bi_cell.bhb)
    print("by:", bi_cell.by)
    bi_cell.bhf = np.random.randn(1, 15)
    h_prev = np.random.randn(8, 15)
    x_t = np.random.randn(8, 10)
    h = bi_cell.forward(h_prev, x_t)
    print(h.shape)
    print(h)

# Expected Output
"""
Whf: [[ 0.44122749 -0.33087015  2.43077119 -0.25209213  0.10960984  1.58248112
  -0.9092324  -0.59163666  0.18760323 -0.32986996 -1.19276461 -0.20487651
  -0.35882895  0.6034716  -1.66478853]
 [-0.70017904  1.15139101  1.85733101 -1.51117956  0.64484751 -0.98060789
  -0.85685315 -0.87187918 -0.42250793  0.99643983  0.71242127  0.05914424
  -0.36331088  0.00328884 -0.10593044]
 [ 0.79305332 -0.63157163 -0.00619491 -0.10106761 -0.05230815  0.24921766
   0.19766009  1.33484857 -0.08687561  1.56153229 -0.30585302 -0.47773142
   0.10073819  0.35543847  0.26961241]
 [ 1.29196338  1.13934298  0.4944404  -0.33633626 -0.10061435  1.41339802
   0.22125412 -1.31077313 -0.68956523 -0.57751323  1.15220477 -0.10716398
   2.26010677  0.65661947  0.12480683]
 [-0.43570392  0.97217931 -0.24071114 -0.82412345  0.56813272  0.01275832
   1.18906073 -0.07359332 -2.85968797  0.7893664  -1.87774088  1.53875615
   1.82136474 -0.42703139 -1.16470191]
 [-1.39707402  0.87265462 -0.20211818 -0.59835993 -0.2434197   2.08851469
   0.34691933  0.74572695  0.77690759  1.01842113  1.06135144 -0.71046645
  -0.2151878  -0.76076031 -0.71116323]
 [ 1.14150774 -0.50175555 -0.07915136 -0.69282634 -0.59340277  0.78823794
  -0.44542999 -0.48212019  0.49355766  0.50048733  0.79242262  0.17076445
  -1.75374086  0.63029648  0.49832921]
 [ 1.01813761 -0.84646862  2.52080763 -1.23238611  0.72695326  0.04595522
  -0.48713265  0.81613236 -0.28143012 -2.33562182 -1.16727845  0.45765807
   2.23796561 -1.4812592  -0.01694532]
 [ 1.45073354  0.60687032 -0.37562084 -1.42192455 -1.7811513  -0.74790579
  -0.36840953 -2.24911813 -1.69367504  0.30364847 -0.40899234 -0.75483059
  -0.40751917 -0.81262476  0.92751621]
 [ 1.63995407  2.07361553  0.70979786  0.74715259  1.46309548  1.73844881
   1.46520488  1.21228341 -0.6346525  -1.5996985   0.87715281 -0.09383245
  -0.05567103 -0.88942073 -1.30095145]
 [ 1.40216662  0.46510099 -1.06503262  0.39042061  0.30560017  0.52184949
   2.23327081 -0.0347021  -1.27962318  0.03654264 -0.64635659  0.54856784
   0.21054246  0.34650175 -0.56705117]
 [ 0.41367881 -0.51025606  0.51725935 -0.30100513 -1.11840643  0.49852362
  -0.70609387  1.4438811   0.44295626  0.46770521  0.10134479 -0.05935198
  -2.38669774  1.22217056 -0.81391201]
 [ 0.95626186 -0.63851056 -0.14312642 -0.22418983 -1.03849524 -0.17170905
   0.47634618 -0.41417827 -1.26408334 -0.57321556  0.24981732  1.14720208
   0.83594396  0.28740365 -0.9955963 ]
 [ 0.90688947  0.02421074 -0.23998173  0.91011056  0.61784475  0.49961804
  -1.15154425 -0.6105164  -1.70388541  0.19443738  0.02824125  0.93256051
   0.21204332 -0.36794457  2.1114884 ]
 [-1.02957349 -1.33628031 -0.61056736  0.52469426 -0.34930813 -0.44073846
  -1.1212876   1.47284473 -0.62337224 -1.08070195 -0.12253009 -0.8077431
  -0.23255622  1.33515034 -0.44645673]
 [-0.04978868 -0.36854478 -0.19173957  0.81967992  0.53163372 -0.34161504
  -0.93090048 -0.13421699  0.83259361 -0.01735327 -0.12765822 -1.80791662
   0.99396898 -1.49112886 -1.28210748]
 [-0.37570741  0.03464388  0.04507816 -0.76374689 -0.31313851 -0.60698954
  -1.80955123 -0.25551774 -0.69379935  0.41919776 -0.14520019  0.9638013
   0.69622199  0.89940546  1.20837807]
 [ 0.6932537  -0.16636061  1.35311311 -0.92862651 -0.03547249  0.85964595
  -0.28749661  0.71494995 -0.8034526  -0.54048196  0.54617743  0.71188926
   1.19715449 -0.07006703  0.29822712]
 [ 0.62619261  0.46743206 -1.30262143 -0.57008965  1.44295001 -1.24399513
   0.62888033 -0.42559213  1.00320956 -0.77817761  0.04894463 -2.02640189
  -0.04193635  1.07454278 -1.5008594 ]
 [ 1.18574443 -0.71508124 -0.05123853 -2.77458336  1.07862813 -0.87568592
  -0.53810932 -1.2782157  -0.99276945  1.14342789 -0.5090726   0.89500094
  -0.17620337  0.34608347 -0.50631013]
 [ 0.42716402  2.58856959  0.65289301  0.50583979 -0.47595083  1.01090874
   1.35920097 -1.70208997 -1.38033223  2.10177668  0.42589917  0.12920023
   0.56296251  1.09676472  0.80081885]
 [-0.22308327  2.06367066  0.0126235  -0.8747738  -0.55707938 -0.13230195
  -0.37922499 -0.18779371  0.31546615 -3.28391545 -0.77869325  0.95034471
   0.5630013  -0.68065407 -0.62450339]
 [ 1.14049594 -0.24772894 -0.53020527  1.8557144  -0.36987213  0.68424682
  -0.0456703   0.05078665 -0.94722556 -0.82698742  1.25807361 -1.13889026
   0.27736012 -1.19444596 -0.24043683]
 [-0.03720827 -1.6296784   1.13486338 -0.18379943  1.21473773 -0.93427859
   0.91186241  2.3342401   0.21653196 -0.64706848  0.47870605  0.14082715
  -0.2099986  -0.12050664 -0.57882578]
 [ 0.42386759 -0.38733136 -0.85686815  0.81531389 -0.16581602  2.64535345
  -0.24946988 -0.71733789 -0.54949733  0.37108695 -0.69734581 -1.26330116
   1.63921233 -1.24014464  1.51364577]]
Whb: [[ 0.14105657 -1.06209796  1.6663804  -0.2034536  -1.00754753  0.06540956
   1.28644574  0.68374332  0.8262448   1.75433632  0.21456398  0.37581479
  -0.22598417 -1.45469387 -0.14453466]
 [ 1.61697881 -1.73105363  1.34394613  0.26153957 -0.91051935  0.06546949
   1.77632213 -0.57313319  0.79059361  1.13151397 -0.897094    0.63271186
   0.53515515 -0.47415241  0.68498591]
 [-0.36119419 -0.57742993 -1.2347295   0.38547989 -0.42918999 -0.55892627
  -1.14899998 -1.36515578 -0.78923902  0.72995982 -0.81388187  1.4448595
   0.40825946  0.15806514 -1.20324067]
 [ 1.95358868 -1.4406335   0.53407511  1.69432832 -0.19894722 -0.68352568
  -0.01899812  0.9156626   1.35870723  0.60443768 -1.06941562 -0.6741898
   0.20340805 -1.27616516 -0.24030333]
 [ 2.24095357 -1.05746192  1.16055901 -0.93298444 -0.34072389 -0.07013113
  -1.50552315 -0.10507983  1.29682083  0.7171925   0.69777111 -0.80449784
  -0.14505178  0.2023229   0.67869955]
 [ 1.34251188 -0.99933073  1.69954809 -0.28621623 -0.25163697 -1.20844686
  -0.06779508 -0.22818598 -1.23450433 -0.29138373  0.12135718 -0.41143386
   0.77926035 -1.02468459  0.88988217]
 [-0.18598247  0.37226978 -1.84518514  0.12914587 -0.06190023  0.9357079
  -1.17990317  1.36404151 -1.08263117  1.31669419  0.57819563 -0.7544614
   2.16976159 -1.19562434 -0.17197421]
 [ 0.20706706  0.52178374  0.22638929  0.79913028  0.45924581  0.03269967
  -0.92956292 -0.345037    0.90247952 -1.16649931  0.11099181 -2.04839658
  -0.69561095 -1.62316059  1.24454078]
 [-1.82274919 -0.2396064   0.72844306  0.60888427  0.77318471  1.06235383
   0.47350502  0.83459787 -0.05414128 -0.02563969 -1.76040064  0.16870521
   1.26727682 -0.7479485  -1.16974715]
 [ 0.09123447  1.13441899  1.20657434 -1.50046524  0.4500207  -0.65624934
   0.27747097 -2.25770286  0.92178771 -0.39559162  0.20500967 -0.17965102
  -1.03396596  0.52553754 -0.4816705 ]
 [-0.68841613  0.65919311 -0.08360162 -0.69951327  0.93928815 -0.59041732
  -0.03135099  0.73422269  0.94370965  1.20016024  1.65674391  0.04498319
  -0.79817635  1.64803402  1.39160763]
 [-0.29936128  0.70120491 -0.37760156 -1.55730107  1.03817562  0.34605186
   0.12238178  1.69000604  1.43271214 -0.64162362 -1.10852606 -1.89470326
   0.00622229 -0.38612473  0.36319715]
 [ 0.01284875 -0.65633238 -0.4043497  -1.55800633  1.22838004 -0.53255212
   0.61235606 -0.6577483   0.07085223  0.6300668  -0.27380171 -0.81522014
   2.32521039 -0.19848482  0.56827305]
 [-0.08094039 -0.77322808  1.31425929  0.39116235  0.34091217 -2.27565438
   0.28715275  0.83956179  0.28299387 -0.14843387  1.19284859  0.24610793
   0.58060241  1.03666838  1.00088984]
 [-1.31423951  0.33629778 -0.76162197 -2.00778407  0.28542468 -1.73792109
  -0.46648121 -2.50923485 -1.02102236 -0.46980863 -0.00775845 -0.53986039
  -0.92443751 -0.9010276   1.53221299]
 [-0.47064751 -0.09790047  1.88416925 -1.74263228 -2.06889481 -1.02220276
  -0.39572583 -1.14749445 -0.04310505 -0.08929766 -1.55142087  0.16077208
  -1.43689987  0.69805756  0.21929225]
 [ 2.18675817  0.82094283 -0.21077917  0.33744299  0.13975271 -0.74726193
  -1.46058749  1.09165138  1.51186998 -0.47078551 -0.26528618 -0.49168291
   1.00403618  1.20373126 -0.87839139]
 [-0.34658297 -0.54238642  0.20799802  0.76253508  1.064387   -0.28186182
  -0.9898337   0.53390418  0.58935885 -0.35492432  0.16434875  0.35470849
   0.8773426   0.06850118  0.48434005]
 [ 0.17669481 -0.96828046  0.70237531  0.09819412  1.05312589 -1.18723557
  -0.70287886  0.32574333 -1.71659973 -0.13395585 -0.44814267 -0.41525209
  -0.25240672 -2.22946145  0.04152343]
 [ 0.07414338  1.04951431  1.4438256   0.3547056   0.92535967 -2.39387928
   0.54710378  1.10518889  0.87015652  0.23723286 -0.35581647 -0.36467461
   0.02324046  0.42891668  0.57164015]
 [-0.09454521 -1.25646725  1.82316345 -0.86538249 -2.67845581  1.42602349
   1.01310701  0.18859568  0.97292534  0.44483385 -0.83268662  0.11627968
  -1.23583276 -0.64983948 -0.86673766]
 [ 0.15620962  1.06333106 -0.64545791  0.46029288 -2.13208562 -1.07207642
   0.53533974  0.95781866  0.94326939  1.0694166  -1.6387506  -0.52063558
  -2.57986902 -0.15588174  0.23488451]
 [-0.33955213 -0.92692654  0.42948423  0.53165925  0.13359522 -0.60333099
   0.04305156  0.0852979   0.26729695  0.6221842  -0.02690954 -1.44530212
  -0.27398421  0.37117917 -0.4395953 ]
 [ 0.21816989 -0.68479494  1.34306059  0.4039188   0.13998     0.75699763
   0.76350626  0.09020354  1.00106093  0.65539631  0.21439974  0.00278747
  -1.10824865  0.19314825  0.17739336]
 [-0.8335227   1.36142303 -0.00426734 -1.07221509 -0.4178816  -0.04222753
  -0.03548137 -0.7803113   0.03513728 -1.45721923 -0.42969079  0.43467306
  -0.67350954  0.64650631  0.80079913]]
Wy: [[-0.5951396  -2.39931533  0.8805501   1.34738633 -1.59084192]
 [ 0.20994277  0.32645628 -1.61497389 -0.53523219  1.25703151]
 [ 0.23099738  0.02967206  1.17133152 -0.57382667 -1.78471539]
 [ 1.36940498 -0.76179628  0.01777266  1.18663241  0.30666377]
 [ 0.19639886 -0.57633484  0.61272754  2.2319718  -0.54104594]
 [-1.07834573  0.34178248 -0.69099613  0.16632997  0.06489149]
 [ 0.53806244  0.24238798  0.35846257  1.317401   -0.92252859]
 [ 0.09544892  0.36973682 -1.16863734 -0.8972123  -0.28381505]
 [-0.48653688  1.1682517   0.18185217  2.98140467 -0.32478874]
 [-0.26088823  2.30685413 -0.50824667  2.24781083 -3.06178812]
 [-0.60951027  1.45498546  1.64824094  0.51341349 -0.59298498]
 [-0.08211555  0.8374637   0.37550555  1.01242074 -0.66058459]
 [ 0.89638904  1.34789585  3.15523734  0.79102195  0.77291163]
 [-0.77559619 -1.27286902 -2.22189774  0.53179091  2.09546601]
 [-0.02953905 -2.41892714  0.26206331 -0.1377095  -0.30449584]
 [ 1.83116972  1.31156728  1.56940838 -0.717684    0.43875262]
 [-1.02332951  1.22266729 -1.04255577  1.33993536  0.32960521]
 [ 0.30942228 -1.36258955  0.78763334 -1.74759067 -1.17086109]
 [-0.86199111  2.14702066  0.18271962 -0.60880629  0.59885942]
 [ 2.45365815 -0.0848358  -0.98697267 -1.04397312  0.01487557]
 [ 1.75040717 -0.07217679 -0.1716691  -0.92579127 -1.07683994]
 [ 1.11155634  0.41267393  0.47996896 -1.31493026  0.62496552]
 [ 0.38682202  1.58309294  1.12518776 -0.17742593  0.3912004 ]
 [ 0.37869257 -0.04140653 -0.81968842  0.95714003  2.31940246]
 [-0.04087018  1.70145051 -0.27158407 -0.29013405  0.02527117]
 [-0.24662111  1.02746419  1.21701085 -2.52211938  0.86986643]
 [ 0.44870563  1.80474122 -0.84473908  0.18360829  1.27472362]
 [ 0.95183795  1.59129629 -0.7303816  -0.62133282  0.64605196]
 [-1.32268974  0.50138097  0.9912463   0.39834356 -1.30110969]
 [-0.52717844 -0.26840324  0.46754785  0.29400002 -0.39042771]]
bhf: [[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]]
bhb: [[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]]
by: [[0. 0. 0. 0. 0.]]
(8, 15)
[[-0.99999937 -0.99999989  0.99999384 -0.99957492  0.90439429 -0.99998325
  -0.9998646   0.96307672 -0.69899914  0.75434899 -0.99385354  0.88835127
   0.99999678 -0.99983086 -0.35515913]
 [-0.99738092 -0.99974649 -0.99998586  0.99999592 -0.88287139 -0.77018356
   0.99995386  0.99999503  0.90501843 -0.99999958  0.85342148 -1.
   0.99993909 -0.99999999 -0.99999821]
 [ 0.99907058 -0.99928351  0.88255065 -0.99902455  0.99248207 -0.99924151
  -0.89724959  0.65801635  0.99994381 -0.99999746  0.99998816 -0.9999992
  -1.         -0.91392091 -0.98539153]
 [-0.92316268  0.90413746 -0.97959663 -0.99973523 -0.99999705 -0.3831155
  -0.9988017  -0.99999845 -0.99972609  0.24719108 -0.999933   -0.99998234
   0.2033747  -0.99999792  0.99906718]
 [ 0.99998061  0.9999645   0.99999997 -0.99999997  0.99911737  0.9999847
   0.99817353 -0.9981396  -0.99999959 -0.99980671  0.854733   -0.46623082
   1.         -1.          0.98487869]
 [ 0.98512282  0.99968729 -0.53826753 -0.99998979 -0.98391832  0.99940808
   0.99983417 -0.99992639 -0.99999933  0.99872946  0.88353371  0.99724532
   0.95142483  0.73378345 -0.38139447]
 [-0.99999489  0.99566188  0.99628535  0.99612299 -0.77940446  0.95183775
  -0.9999883  -0.99998999 -0.99999208  0.99999134  0.99992409 -0.99800957
   0.76234982  0.22799726 -0.99995051]
 [-1.          0.9413251  -0.99999583 -0.99128995 -0.99440344 -0.99999999
   0.95319879 -0.96915356  0.96669961 -0.99964141 -0.69485207 -0.99340705
  -0.80712677  0.98892046 -0.99772255]]
"""