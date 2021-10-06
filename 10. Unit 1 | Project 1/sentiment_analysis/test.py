import os
import sys
import time
import traceback
import project1 as p1
import numpy as np

verbose = False

def green(s):
    return '\033[1;32m%s\033[m' % s

def yellow(s):
    return '\033[1;33m%s\033[m' % s

def red(s):
    return '\033[1;31m%s\033[m' % s

def log(*m):
    print(" ".join(map(str, m)))

def log_exit(*m):
    log(red("ERROR:"), *m)
    exit(1)


def check_real(ex_name, f, exp_res, *args):
    try:
        res = f(*args)
    except NotImplementedError:
        log(red("FAIL"), ex_name, ": not implemented")
        return True
    if not np.isreal(res):
        log(red("FAIL"), ex_name, ": does not return a real number, type: ", type(res))
        return True
    if res != exp_res:
        log(red("FAIL"), ex_name, ": incorrect answer. Expected", exp_res, ", got: ", res)
        return True


def equals(x, y):
    if type(y) == np.ndarray:
        return (x == y).all()
    return x == y

def check_tuple(ex_name, f, exp_res, *args, **kwargs):
    try:
        res = f(*args, **kwargs)
    except NotImplementedError:
        log(red("FAIL"), ex_name, ": not implemented")
        return True
    if not type(res) == tuple:
        log(red("FAIL"), ex_name, ": does not return a tuple, type: ", type(res))
        return True
    if not len(res) == len(exp_res):
        log(red("FAIL"), ex_name, ": expected a tuple of size ", len(exp_res), " but got tuple of size", len(res))
        return True
    if not all(equals(x, y) for x, y in zip(res, exp_res)):
        log(red("FAIL"), ex_name, ": incorrect answer. Expected", exp_res, ", got: ", res)
        return True

def check_array(ex_name, f, exp_res, *args):
    try:
        res = f(*args)
    except NotImplementedError:
        log(red("FAIL"), ex_name, ": not implemented")
        return True
    if not type(res) == np.ndarray:
        log(red("FAIL"), ex_name, ": does not return a numpy array, type: ", type(res))
        return True
    if not len(res) == len(exp_res):
        log(red("FAIL"), ex_name, ": expected an array of shape ", exp_res.shape, " but got array of shape", res.shape)
        return True
    if not all(equals(x, y) for x, y in zip(res, exp_res)):
        log(red("FAIL"), ex_name, ": incorrect answer. Expected", exp_res, ", got: ", res)
        return True

def check_list(ex_name, f, exp_res, *args):
    try:
        res = f(*args)
    except NotImplementedError:
        log(red("FAIL"), ex_name, ": not implemented")
        return True
    if not type(res) == list:
        log(red("FAIL"), ex_name, ": does not return a list, type: ", type(res))
        return True
    if not len(res) == len(exp_res):
        log(red("FAIL"), ex_name, ": expected a list of size ", len(exp_res), " but got list of size", len(res))
        return True
    if not all(equals(x, y) for x, y in zip(res, exp_res)):
        log(red("FAIL"), ex_name, ": incorrect answer. Expected", exp_res, ", got: ", res)
        return True


def check_get_order():
    ex_name = "Get order"
    if check_list(
            ex_name, p1.get_order,
            [0], 1):
        log("You should revert `get_order` to its original implementation for this test to pass")
        return
    if check_list(
            ex_name, p1.get_order,
            [1, 0], 2):
        log("You should revert `get_order` to its original implementation for this test to pass")
        return
    log(green("PASS"), ex_name, "")


def check_hinge_loss_single():
    ex_name = "Hinge loss single"

    feature_vector = np.array([1, 2])
    label, theta, theta_0 = 1, np.array([-1, 1]), -0.2
    exp_res = 1 - 0.8
    if check_real(
            ex_name, p1.hinge_loss_single,
            exp_res, feature_vector, label, theta, theta_0):
        return
    log(green("PASS"), ex_name, "")


def check_hinge_loss_full():
    ex_name = "Hinge loss full"

    feature_vector = np.array([[1, 2], [1, 2]])
    label, theta, theta_0 = np.array([1, 1]), np.array([-1, 1]), -0.2
    exp_res = 1 - 0.8
    if check_real(
            ex_name, p1.hinge_loss_full,
            exp_res, feature_vector, label, theta, theta_0):
        return

    log(green("PASS"), ex_name, "")

def check_hinge_loss_full_2():
    ex_name = "Hinge loss full 2"

    feature_vector = np.array([[1, 2], [1, 2], [1, 2]])
    label, theta, theta_0 = np.array([1, 1, 1]), np.array([-1, 1]), -0.2
    exp_res = 1 - 0.8
    if check_real(
            ex_name, p1.hinge_loss_full,
            exp_res, feature_vector, label, theta, theta_0):
        return

    log(green("PASS"), ex_name, "")


def check_hinge_loss_full_4():
    ex_name = "Hinge loss full: Test 4"

    feature_vector = np.array(
            [
                [-0.18472826 ,-3.27551449 , 3.9109146  , 4.0114707  ,-4.37379166],
                [-1.37956983 , 1.91327236 , 4.80435275 , 6.09625954 ,-2.07265713],
                [ 3.04198902 ,-3.07178737 ,-1.52057115 , 3.3385423  , 0.54610877],
                [ 5.39100329 ,-0.49406049 ,-2.52078092 ,-0.45947861 , 6.70556531],
                [ 4.05378793 , 5.93132543 , 3.23167942 ,-4.38850912 , 6.75472453],
                [ 6.9162901  ,-1.85141089 ,-1.69416199 , 4.55784365 ,-3.55049711],
                [-3.86241827 , 4.62693769 , 6.19761984 , 5.06163544 ,-1.11978026],
                [ 4.55913232 , 6.30362193 , 3.7535506  , 6.72269519 , 2.22327723],
                [-1.04733321 ,-2.33002512 , 2.3200751  ,-3.321577   , 4.17192468],
                [ 2.995456   ,-3.68137797 ,-3.6864026  ,-2.85922505 , 1.4804962 ],
                [ 6.42084371 ,-1.68343244 ,-1.09284653 ,-1.64111652 ,-4.74029965],
                [ 2.94280857 ,-0.77494597 , 2.82745767 ,-2.45817565 ,-0.75311432],
                [-3.35238494 , 3.84833403 ,-0.92923066 , 6.96256688 , 1.14271564],
                [-0.69151667 , 2.65114568 , 0.80228413 ,-0.70576065 ,-3.77167494],
                [-2.61930758 , 5.59319241 , 2.5834452  , 6.43903521 , 4.61741086],
                [ 3.77957128 ,-2.06180702 ,-0.20458118 ,-0.906144   ,-1.2175165 ],
                [ 1.25280172 , 4.55291367 , 5.2023342  ,-1.28921724 , 1.87728942],
            ]
        )
    label = np.array([ 1., -1., -1.,  1., -1.,  1., -1.,  1.,  1., -1.,  1., -1., -1.,  1.,  1., -1.,  1.])
    theta = np.array([-1.47729689 , 0.80575773 , 1.36302739 , 0.56973226 , -0.24280451])
    theta_0 = -1
    exp_res = 5.2973172
    if check_real(
            ex_name, p1.hinge_loss_full,
            exp_res, feature_vector, label, theta, theta_0):
        return

    log(green("PASS"), ex_name, "")



def check_perceptron_single_update():
    ex_name = "Perceptron single update"

    feature_vector = np.array([1, 2])
    label, theta, theta_0 = 1, np.array([-1, 1]), -1.5
    exp_res = (np.array([0, 3]), -0.5)
    if check_tuple(
            ex_name, p1.perceptron_single_step_update,
            exp_res, feature_vector, label, theta, theta_0):
        return

    feature_vector = np.array([1, 2])
    label, theta, theta_0 = 1, np.array([-1, 1]), -1
    exp_res = (np.array([0, 3]), 0)
    if check_tuple(
            ex_name + " (boundary case)", p1.perceptron_single_step_update,
            exp_res, feature_vector, label, theta, theta_0):
        return

    log(green("PASS"), ex_name, "")


def check_perceptron():
    ex_name = "Perceptron"

    feature_matrix = np.array([[1, 2]])
    labels = np.array([1])
    T = 1
    exp_res = (np.array([1, 2]), 1)
    if check_tuple(
            ex_name, p1.perceptron,
            exp_res, feature_matrix, labels, T):
        return

    feature_matrix = np.array([[1, 2], [-1, 0]])
    labels = np.array([1, 1])
    T = 1
    exp_res = (np.array([0, 2]), 2)
    if check_tuple(
            ex_name, p1.perceptron,
            exp_res, feature_matrix, labels, T):
        return

    feature_matrix = np.array([[1, 2]])
    labels = np.array([1])
    T = 2
    exp_res = (np.array([1, 2]), 1)
    if check_tuple(
            ex_name, p1.perceptron,
            exp_res, feature_matrix, labels, T):
        return

    feature_matrix = np.array([[1, 2], [-1, 0]])
    labels = np.array([1, 1])
    T = 2
    exp_res = (np.array([0, 2]), 2)
    if check_tuple(
            ex_name, p1.perceptron,
            exp_res, feature_matrix, labels, T):
        return

    log(green("PASS"), ex_name, "")

    '''
    TEST 1
    perceptron input:
    feature_matrix: [[-0.31255827  0.09540529  0.14263542  0.32640921 -0.07153749  0.26586351
    0.11491641 -0.29498473 -0.09630707  0.33812675]
    [-0.41849847 -0.34817643 -0.02284925  0.09093783 -0.10852288 -0.02451863
    -0.19025624 -0.32764926  0.47121641 -0.41851072]
    [-0.15909151 -0.06036342 -0.43346989  0.04251016  0.20662617 -0.4616565
    0.05016029  0.07560336 -0.43477031 -0.14547461]
    [ 0.28743941  0.39208044 -0.07796436  0.49254833 -0.25074881 -0.31444749
    -0.19941556  0.39989347  0.26935477 -0.20676832]
    [ 0.11732656  0.1957919   0.06268382 -0.14590629  0.40598747  0.40537257
    -0.42909546 -0.27983778 -0.39795199  0.08421531]]
    labels: [-1  1 -1 -1  1]
    T: 5
    perceptron output is ['-0.5354600', '-0.9276833', '0.3857841', '-0.8254983', '0.3046018', '0.8665758', '-0.7752691', '-1.1156484', '0.8062034', '-0.7386899']

    TEST 2
    perceptron input:
    feature_matrix: [[ 0.03382818 -0.37611207 -0.44277131  0.06322097  0.20076461]
    [ 0.28218903  0.42171964  0.23078218  0.20915468 -0.19667706]
    [ 0.46882715 -0.20363387 -0.07959366 -0.07212465 -0.16561266]
    [-0.09176815  0.17742056 -0.31850755  0.2764336   0.35381107]
    [ 0.21480336  0.3041882   0.03826913  0.16837212  0.01565019]
    [ 0.37775911 -0.20755822 -0.25520446  0.47247715 -0.13106034]
    [-0.0750656  -0.3228039  -0.318215    0.44272866  0.42592634]
    [-0.16772774  0.10264783 -0.38045584  0.17448901 -0.07368992]
    [-0.44716282  0.08824979  0.31330974 -0.34623079 -0.3768383 ]
    [-0.11010644  0.30758845 -0.21459909  0.07415223 -0.04606715]]
    labels: [-1  1  1  1 -1 -1 -1 -1 -1 -1]
    T: 5
    perceptron output is ['1.1019446', '1.2847810', '0.3108594', '0.6383593', '1.0691919']

    TEST 3
    perceptron input:
    feature_matrix: [[-0.06509464  0.13712114  0.00557293  0.14497962 -0.19795269  0.23609977
    -0.16369244  0.46463837  0.35647425 -0.36253263]
    [ 0.38294949 -0.43167648  0.35600611 -0.30552651 -0.07586525  0.03239405
    -0.49336263 -0.01668423  0.46097149 -0.49287697]
    [ 0.39902591 -0.21236341 -0.23935457  0.45174255 -0.30544718 -0.30024431
    0.34445116  0.02989331  0.46163021  0.16245283]
    [-0.16040475 -0.02626438 -0.29245247 -0.24003465  0.07745426  0.20141987
    0.1637068  -0.15884246  0.39169105  0.24137702]
    [ 0.41435249 -0.47228054 -0.42054895  0.28978785 -0.04734768  0.36944331
    0.34315375  0.1618467   0.30698683  0.016677  ]
    [-0.19551983  0.02450823 -0.42412088 -0.12208617 -0.35414264  0.37544714
    0.26157332  0.04226237 -0.19910126 -0.46301782]
    [-0.00877675  0.18241835 -0.20590345  0.38219876 -0.06302104 -0.42394651
    0.1537848  -0.36429491  0.16041357 -0.4442892 ]
    [-0.16677351  0.35188174  0.40316852  0.43750996  0.07181857 -0.29916056
    -0.30772216  0.09464418  0.49520936  0.45498171]
    [-0.00877675  0.18241835 -0.20590345  0.38219876 -0.06302104 -0.42394651
    0.1537848  -0.36429491  0.16041357 -0.4442892 ]
    [-0.30478497 -0.39248538 -0.43503794 -0.04266835 -0.09933402  0.39074609
    -0.43902264 -0.36769524 -0.09629405  0.30224581]]
    labels: [-1 -1 -1  1 -1  1 -1 -1  1 -1]
    T: 1
    perceptron output is ['-0.2424007', '0.5564252', '-0.4756313', '-0.8095311', '-0.0668936', '-0.5442081', '1.1463484', '-0.8343090', '-0.7093730', '-1.0773019']

    TEST 4
    perceptron input:
    feature_matrix: [[ 0.          0.          0.          0.          0.          0.
    0.          0.          0.          0.        ]
    [-0.29312287 -0.76363642 -0.10473864 -0.70730159 -0.46090859 -0.01503123
    -0.60395173 -0.5309103  -0.2937571  -0.90040334]
    [-0.20648588 -0.04196859 -0.1310821  -0.78347216 -0.40042506 -0.35131155
    -0.90115917 -0.47352425 -0.91439308 -0.59925564]
    [ 0.07797285  0.77246976  0.94395617  0.29478272  0.93105688  0.50554407
    0.3520392   0.34442195  0.61750074  0.06461107]
    [ 0.30270572  0.95484059  0.24249719  0.32784258  0.86588569  0.69744376
    0.98111963  0.31740784  0.38734277  0.0187272 ]
    [-0.92266847 -0.09679031 -0.32334123 -0.79030906 -0.64871187 -0.59744891
    -0.42793238 -0.74235979 -0.7296044  -0.16913392]
    [ 0.92676608  0.70189291  0.90878608  0.95103641  0.45918089  0.80707773
    0.62002113  0.51784546  0.68358502  0.01781545]
    [-0.53855711 -0.57188017 -0.80343514 -0.05051181 -0.06478022 -0.08516316
    -0.81946405 -0.29118727 -0.35681378 -0.1653781 ]
    [ 0.32396609  0.55885251  0.36801491  0.24681444  0.61836972  0.57242144
    0.64462605  0.72348712  0.62610868  0.94214654]
    [ 0.60573664  0.94876097  0.48715015  0.27348288  0.85355904  0.90113733
    0.12221302  0.30518562  0.25942379  0.80243617]]
    labels: [-1 -1 -1  1  1 -1  1 -1  1  1]
    T: 100
    perceptron output is ['0.6057366', '0.9487610', '0.4871501', '0.2734829', '0.8535590', '0.9011373', '0.1222130', '0.3051856', '0.2594238', '0.8024362']

    TEST 5
    perceptron input:
    feature_matrix: [[ 0.3536791  -0.29675647  0.12821809 -0.46894874  0.08591424 -0.41172674
    0.17661347  0.02248075  0.37472662  0.02819403]
    [ 0.06994128  0.21592172  0.01086115 -0.35102022  0.08806846 -0.04582551
    -0.26842663 -0.38676149  0.36787679 -0.40510382]
    [ 0.05189222  0.34954765 -0.4442803   0.22338105  0.08551743  0.05148504
    -0.37734994 -0.2080192   0.26476877 -0.28414112]
    [ 0.06994128  0.21592172  0.01086115 -0.35102022  0.08806846 -0.04582551
    -0.26842663 -0.38676149  0.36787679 -0.40510382]
    [-0.03854452 -0.30860836 -0.1105806   0.45829219  0.11581223 -0.37634994
    -0.46379085  0.14758206 -0.11462745  0.27809934]
    [-0.4823957   0.16890254  0.0761655   0.49121843 -0.05687969 -0.41739487
    -0.33371577 -0.37525056 -0.21806456 -0.16422973]
    [-0.3581763  -0.28624086 -0.06173441 -0.14159161  0.19323096  0.412065
    0.29107479  0.06500099  0.33366361 -0.46632806]
    [-0.44540085 -0.160975    0.41353127  0.0251927   0.00703459 -0.35814648
    -0.22934947 -0.2112058  -0.2716968   0.3624724 ]
    [-0.3332655   0.32236912  0.0738052  -0.10576798 -0.46477038 -0.34853945
    -0.19651855  0.25216025  0.0113481  -0.27639909]
    [-0.29101082  0.49243618  0.14537639 -0.23291846 -0.12875096 -0.12351288
    0.1696267   0.21370381 -0.32811218  0.4105592 ]]
    labels: [ 1  1 -1 -1 -1  1  1  1  1  1]
    T: 100
    perceptron output is ['0.1012128', '0.5042881', '0.3841751', '-1.1601594', '-0.1586490', '-0.1588897', '0.8100310', '0.0886025', '0.1612419', '0.1606539']
    '''


def check_average_perceptron():
    ex_name = "Average perceptron"

    feature_matrix = np.array([[1, 2]])
    labels = np.array([1])
    T = 1
    exp_res = (np.array([1, 2]), 1)
    if check_tuple(
            ex_name, p1.average_perceptron,
            exp_res, feature_matrix, labels, T):
        return

    feature_matrix = np.array([[1, 2], [-1, 0]])
    labels = np.array([1, 1])
    T = 1
    exp_res = (np.array([-0.5, 1]), 1.5)
    if check_tuple(
            ex_name, p1.average_perceptron,
            exp_res, feature_matrix, labels, T):
        return

    feature_matrix = np.array([[1, 2]])
    labels = np.array([1])
    T = 2
    exp_res = (np.array([1, 2]), 1)
    if check_tuple(
            ex_name, p1.average_perceptron,
            exp_res, feature_matrix, labels, T):
        return

    feature_matrix = np.array([[1, 2], [-1, 0]])
    labels = np.array([1, 1])
    T = 2
    exp_res = (np.array([-0.25, 1.5]), 1.75)
    if check_tuple(
            ex_name, p1.average_perceptron,
            exp_res, feature_matrix, labels, T):
        return

    log(green("PASS"), ex_name, "")


def check_pegasos_single_update():
    ex_name = "Pegasos single update"

    feature_vector = np.array([1, 2])
    label, theta, theta_0 = 1, np.array([-1, 1]), -1.5
    L = 0.2
    eta = 0.1
    exp_res = (np.array([-0.88, 1.18]), -1.4)
    if check_tuple(
            ex_name, p1.pegasos_single_step_update,
            exp_res,
            feature_vector, label, L, eta, theta, theta_0):
        return

    feature_vector = np.array([1, 1])
    label, theta, theta_0 = 1, np.array([-1, 1]), 1
    L = 0.2
    eta = 0.1
    exp_res = (np.array([-0.88, 1.08]), 1.1)
    if check_tuple(
            ex_name +  " (boundary case)", p1.pegasos_single_step_update,
            exp_res,
            feature_vector, label, L, eta, theta, theta_0):
        return

    feature_vector = np.array([1, 2])
    label, theta, theta_0 = 1, np.array([-1, 1]), -2
    L = 0.2
    eta = 0.1
    exp_res = (np.array([-0.88, 1.18]), -1.9)
    if check_tuple(
            ex_name, p1.pegasos_single_step_update,
            exp_res,
            feature_vector, label, L, eta, theta, theta_0):
        return

    log(green("PASS"), ex_name, "")


def check_pegasos():
    ex_name = "Pegasos"

    feature_matrix = np.array([[1, 2]])
    labels = np.array([1])
    T = 1
    L = 0.2
    exp_res = (np.array([1, 2]), 1)
    if check_tuple(
            ex_name, p1.pegasos,
            exp_res, feature_matrix, labels, T, L):
        return

    feature_matrix = np.array([[1, 1], [1, 1]])
    labels = np.array([1, 1])
    T = 1
    L = 1
    exp_res = (np.array([1-1/np.sqrt(2), 1-1/np.sqrt(2)]), 1)
    if check_tuple(
            ex_name, p1.pegasos,
            exp_res, feature_matrix, labels, T, L):
        return

    log(green("PASS"), ex_name, "")


def check_classify():
    ex_name = "Classify"

    feature_matrix = np.array([[1, 1], [1, 1], [1, 1]])
    theta = np.array([1, 1])
    theta_0 = 0
    exp_res = np.array([1, 1, 1])
    if check_array(
            ex_name, p1.classify,
            exp_res, feature_matrix, theta, theta_0):
        return

    feature_matrix = np.array([[-1, 1]])
    theta = np.array([1, 1])
    theta_0 = 0
    exp_res = np.array([-1])
    if check_array(
            ex_name + " (boundary case)", p1.classify,
            exp_res, feature_matrix, theta, theta_0):
        return

    log(green("PASS"), ex_name, "")

def check_classifier_accuracy():
    ex_name = "Classifier accuracy"

    train_feature_matrix = np.array([[1, 0], [1, -1], [2, 3]])
    val_feature_matrix = np.array([[1, 1], [2, -1]])
    train_labels = np.array([1, -1, 1])
    val_labels = np.array([-1, 1])
    exp_res = 1, 0
    T=1
    if check_tuple(
            ex_name, p1.classifier_accuracy,
            exp_res,
            p1.perceptron,
            train_feature_matrix, val_feature_matrix,
            train_labels, val_labels,
            T=T):
        return

    train_feature_matrix = np.array([[1, 0], [1, -1], [2, 3]])
    val_feature_matrix = np.array([[1, 1], [2, -1]])
    train_labels = np.array([1, -1, 1])
    val_labels = np.array([-1, 1])
    exp_res = 1, 0
    T=1
    L=0.2
    if check_tuple(
            ex_name, p1.classifier_accuracy,
            exp_res,
            p1.pegasos,
            train_feature_matrix, val_feature_matrix,
            train_labels, val_labels,
            T=T, L=L):
        return

    log(green("PASS"), ex_name, "")

def check_bag_of_words():
    ex_name = "Bag of words"

    texts = [
        "He loves to walk on the beach",
        "There is nothing better"]

    try:
        res = p1.bag_of_words(texts)
    except NotImplementedError:
        log(red("FAIL"), ex_name, ": not implemented")
        return
    if not type(res) == dict:
        log(red("FAIL"), ex_name, ": does not return a tuple, type: ", type(res))
        return

    vals = sorted(res.values())
    exp_vals = list(range(len(res.keys())))
    if not vals == exp_vals:
        log(red("FAIL"), ex_name, ": wrong set of indices. Expected: ", exp_vals, " got ", vals)
        return

    log(green("PASS"), ex_name, "")

    keys = sorted(res.keys())
    exp_keys = ['beach', 'better', 'he', 'is', 'loves', 'nothing', 'on', 'the', 'there', 'to', 'walk']
    stop_keys = ['beach', 'better', 'loves', 'nothing', 'walk']

    if keys == exp_keys:
        log(yellow("WARN"), ex_name, ": does not remove stopwords:", [k for k in keys if k not in stop_keys])
    elif keys == stop_keys:
        log(green("PASS"), ex_name, " stopwords removed")
    else:
        log(red("FAIL"), ex_name, ": keys are missing:", [k for k in stop_keys if k not in keys], " or are not unexpected:", [k for k in keys if k not in stop_keys])


def check_extract_bow_feature_vectors():
    ex_name = "Extract bow feature vectors"
    texts = [
        "He loves her ",
        "He really really loves her"]
    keys = ["he", "loves", "her", "really"]
    dictionary = {k:i for i, k in enumerate(keys)}
    exp_res = np.array(
        [[1, 1, 1, 0],
        [1, 1, 1, 1]])
    non_bin_res = np.array(
        [[1, 1, 1, 0],
        [1, 1, 1, 2]])


    try:
        res = p1.extract_bow_feature_vectors(texts, dictionary)
    except NotImplementedError:
        log(red("FAIL"), ex_name, ": not implemented")
        return

    if not type(res) == np.ndarray:
        log(red("FAIL"), ex_name, ": does not return a numpy array, type: ", type(res))
        return
    if not len(res) == len(exp_res):
        log(red("FAIL"), ex_name, ": expected an array of shape ", exp_res.shape, " but got array of shape", res.shape)
        return

    log(green("PASS"), ex_name)

    if (res == exp_res).all():
        log(yellow("WARN"), ex_name, ": uses binary indicators as features")
    elif (res == non_bin_res).all():
        log(green("PASS"), ex_name, ": correct non binary features")
    else:
        log(red("FAIL"), ex_name, ": unexpected feature matrix")
        return

def main():
    log(green("PASS"), "Import project1")
    try:
        check_get_order()
        check_hinge_loss_single()
        check_hinge_loss_full()
        check_hinge_loss_full_2()
        check_hinge_loss_full_4()
        check_perceptron_single_update()
        check_perceptron()
        check_average_perceptron()
        check_pegasos_single_update()
        check_pegasos()
        check_classify()
        check_classifier_accuracy()
        check_bag_of_words()
        check_extract_bow_feature_vectors()
    except Exception:
        log_exit(traceback.format_exc())

if __name__ == "__main__":
    main()
