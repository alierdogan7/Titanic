import math
import matplotlib.pyplot as plt
import numpy as np


"""
şimdi bu gradient descenti ben direk copy paste yaptım internettwen
geri kalanları ben yazdım


logistic reg. implement etmek için hangi adımları takip etmek lazım acep
şimdi bana bi J(theta) cost fonks. hesaplayan metod lazım
bi gradient hesaplayan metod lazım
bi de converge edene kadar iterate eden bi şey lazım

senin matlab kodunu inceleyebilir miyim
bı bakıyım duruyorsa

yalnız onda hatırladıgım kadarıyla, gradıent descentı hazır olarak kullanmıstık


hadi ya izin verdiler mi ona??!
courserada andrew ng zaten onu kullandırtıyordu
hmm..
NEYSE ÖNEMLİ DEĞİL

sen  capslock acık devam eyle

ANLADIM TAMAM

ÖMER DE DAHA BAŞLAMADI ÖDEVE O DA SIKINTIDA

YOLLAYAYIM......

MAIL YOLLADIM
FEATURE'LAR CINSIYET TICKET CLASS VE YAŞ. 3 TANE PARAMETER VAR
EVET

sadece x1 x2 x3 olacak degıl mı. x1*x2 veya x1*x2(??) kullanmayacaksın?
boundaryfonksıyonu lıneer mı olacak yoksa lıneer olmayabılır mı

HMM.. İŞTE ONDAN EMİN DEĞİLİM

ÖNCE DATANIN VISUALIZE EDİLMİŞ Bİ HALİNİ GÖRSEK TAHMİN EDEBİLİRİZ SANKİ.

BU BENİM HYPTHOESIS FONKSIYONUM BAK:
BURADA LINEER IMPLEMENT ETTIM ASLINDA. SENIN İKİNCİ DEDİĞİN GİBİ YAPMAK AKLIMA GELMEMMİŞTİ..


def hypothesis(**kwargs?????????): #pythonın özelliği bu. istediğin kadar arguman pass edebiliyorsun sonra dictionaryden çekiyorsun argümanları
# mesela bu fonk. şöyle çağırıyorsun. hypothesis(theta_set=[1,2,3], param_set=[1,2,3]) gibi işte.. anladım sanırım.
    sum = 0
    for theta, x in zip(kwargs['theta_set'], kwargs['param_set']):
        sum += theta * x
    return sum # bu mu? evet odevı nerden attın kenk
    # gmail bozulmuş hacı. sende açılıyor mu

    # GMAILE YOLLADIM

## bu neyin sum'ı ? h thetanın içine aldığı parametrenin sum'ı mı?
def classify(sum):
	sum buyuk 0 sa ...
  kucuk 0 sa


def cost (weıghts, x, y):


h(X) = w1*x1 + w2 *x2 + w3*x3 ...... ?
bencu bunu deneyelım.
h(X) = w1*x1 + w2 *x2 + w3*x3 + w4 * x1* x2 ..... ...... ?
ÖDEV DOKUMANINI YOLLAYAYIM Bİ DE SANA



sımdı featureların neler ?

kanka
gradient descenti kendin implement etmek zorunda oldugun nerede yazıyor hoca mı dedı
hemmmm.

ZOR MU YAV :) NASI ETSEKKİNE
sen sımdı gradıent descentı ımplement etmek ıstıyosun sadece degıl mı

ABİ ASLINDA FORMÜLE BAKTIĞINDA GAYET KOLAY GİBİ DURUYOR AMA NEREDEN BAŞLAYACAĞIMI BİLMİYORUM
O KISMI ÇÖZSEM IMPLEMENT EDERIM GİBİ GELİYOR.  o zaman buraya bı algosunu yazak

BAK HEMEN AŞAĞIDA Bİ TANE VAR İNTERNETTEN BULDUĞUM. AMA TAM ANLAYAMADIM KODU

ŞÖYLE Kİ:
Dear all,

For Q5 (logistic regression), the question says "stochastic gradient descent", where "stochastic" is a typo.
You are expected to use batch gradient descent, using the procedure given in the slides for Fisher's LDA and Logistic Regression.
(An extra note: in this procedure, (t) corresponds to the iteration index.)

best,
Gokberk


Dear all,

Just a reminder: you are not allowed to use any 3rd party software, public/private source code, libraries or toolboxes
in your solutions to programming assignments in the CS464 homeworks, unless otherwise specified.

However, you are allowed to use linear algebra primitives, which includes operations like matrix addition,
multiplication, element-wise operators, matrix inversion and eigenvalue decomposition. When you are in doubt, please don't hesitate to contact us.

best,
Gokberk

"""

def gradient_descent(alpha, x, y, ep=0.0001, max_iter=10000):
    """
    BAKIYOR MUSUN?
    bakıyorum

    o zaman dırek bunu sızın ıstedıgınız sekle sokalım
    """

    converged = False
    iter = 0
    m = x.shape[0] # number of samples

    # initial theta
    t0 = np.random.random(x.shape[1]) #bu x.shape nedir tam anlayamadım da ??... xın dimensionu
    theta_set = np.random.random(x.shape[1]) # bı tanesı yetıyor adam generıc yazmış
                                      # bız bızım x ı verınce t1 ın uc tane degero olacak
  																		# T1 MATRİS DEĞİL Mİ? YANİ THETA SET ASLINDA.. VEKTÖR GİBİ EVET.. BENCE TEK BİR t OLSA YETER.. 1'E 3'LÜK MATRIX YANİ
        # BIAS TERM'Ü BİLMİYORUM. HAAA ANLADIM. THETA0 + THETA1 * X1 + THETA2 * X2 GİBİ YANİ.. evet
        # tek t yapmak ıstıyorsan... YOK YA ÖNEMLİ DEĞİL O DETAY.
        # x e bır tane daha column ekleyebılırık sadece 1 lerden olusan
        # bence de detay
        		# bıas termu ıgnore mu edıyım dıyon
          # t0 bıas term
          # o lazım ya.
          # yanı boundary functıon ne kadar yuksekten gececek gıbı.
          # ben sahsen 3 tane deger almoası
            # anladıgımkadarıyla t1 1 e 3 luk bı array evet. t0 a nıye 3 tane deger lazım *????

    # total error, J(theta)
    # bu kısmı daha elegant bı sekılde cozebılırık OLABİLİR
    # KODUN TAMAMI BURADA : http://www.bogotobogo.com/python/python_numpy_batch_gradient_descent_algorithm.php
    # ıstersen sunu for suz yazmaya calısalım
    # senın fonksıyonları da kullanarak. ÇOK VAKİT HARCAMAYA GEREK YOK YA BASİT GİBİ MANTIĞI. BU İYİ BENCE. senın cost_functıon obsolete oluyor :D
    # ısraf
    J = sum([(t0 + t1*x[i] - y[i])**2 for i in range(m)])

    # Iterate Loop
    # OLUR.
    # YA ASLINDA BENİM YAZDIĞIM KOD ÇALIŞMA İHTİMALİ YOK MUDUR SENCE? :)
    # Bİ KAÇ EKLEME ÇIKARMA YAPINCA VS. senın gradıent descent nerde
    # O YOK İŞTE :) ONU DA YAZSAM BİRLEŞTİRSEM OLMAZ MI ? YA ŞİMDİ ALİ ASLINDA KODU BERABER BAŞTAN SONA YAZMAYA GEREK YOK DA
    # BANA ŞU LAZIM: KODUN ÇALIŞMA ALGORİTMASI MANTIĞI YANİ..
    # MESELA BEN BU CSV'Yİ OKUDUKTAN SONRA DATAYI ÖNCE HANGİ FONKSİYONA VERCEM VS.
    # csv yı okuy
    # normalıze et yenı x ler olustur.
    #


    # TAMAM BURAYA KADAR. SONRA?
    # sonra dırek gradıent descente ver.
    # o ılk rastgele weıght atayıp onları optımıze edecek.
    #	buranın ayrıntısına gırelım mı
    #
    # GRADİENT DESCENT AŞAĞIDAKİ METODDA ANLAMADIĞIM YERLER VAR ASLINDA. MESELA İKİ TANE grad variable'I var onlar nedir?!
    # bızım ıkı tane thetamız var ya namely, theta 0 and theta1
    # sımdı bı kere bu baslan
    # burdan devam edek o zaman baskan
    while not converged:
      	# buraya grad ların senın fonksıyonu kullanan sekılde yazmak ıstıyorum YALNIZ BU PARAMETLERİ BEN LİST OLARAK ALDIRDIM BİR WEB DEVELOPER OLARAK :)
        # t1 MATRIX MİYİD?
        grad0 = 1.0/m * hypothesis(theta0=t0, theta_set=t1, =, ): #theta_set, param_set, theta0 #sum([(t0 + t1*x[i] - y[i]) for i in range(m)]) # haa hatırladım coruserada anlatıyordu. theta0 için x'le çarpmak olmadığı için. aynen turevını theto0 a gore alınca cost fucntıonunun
        # asagıdakı x[ı] kısmını nedıcez. ben fonksiyonu ONA GÖRE DÜZELTEYİM ŞU t0, t1, x 'in Structureını anlayabilirsem eğer.. bunlar matrix mi ney bunlar. matrıx kanke
        # MACHINE LEARNINGDE DATASET VERİLİRKEN MATIRX OLARAK MI? cogu zaman evet. ben matrıx dısında bı seyle
        # ugrasmadım sanırım
        # O HALDE X KAÇA KAÇLIK Bİ MATRİX OLACAK? 4 SÜTUNLU MU
        # assumıng that m = number of traınıng data, lınes ın the tıtanıc dataset.
        # ıts gonna be m by 3, plus m by 1 bıas term
        # ın terms of matrıx multıplıacatıon
        # x0 + X*W = Y
        # h_theta(y).
        # W ıs a 3 by 1 vector
        # yıeldıng m by 1 output predıctıon.
 				# Bİ DÜŞÜNİYM TAMAM ANLADIM SANKİ. TAMAM DA x0 İLE MATRIX NASIL TOPLUYORSUN ? HAA DAHA SONRA MULTIPLICATIONDAN ÇIKAN MATRİSİN HER SATIRINI X0 ile toplayıp reel bişey alıyon
        #aynen her satıra ayrı muamele.
        # DİMİ?
        # you are rıght
        #  emın mı o lanetı sılen  HAYIR YUNUS EMRE ABİ GELDİ :) :)
        # ben dumur. tam vaktinde :)

        # yerınde olsam sıgmoıd ı fılan matrıslere uygun bı sekılde yazardımü
        # hah tam olarak onun için uğraştım dün baya. sigmoid i matrix ile nasıl implement edeceğimi bulamadım. stackoverflowda bi soru vardı ama anlayamadım cevabını
        # yanı
        #  numpy kullanmak bence sıkıntı olmamalı ya
        # ya da gerek yok mu kı...
        # bılmıyompyhtonu da matlabda resmen bı satırdı ya bu.
        # sigmoid mi? evet matrısformunda
        # mesela  pythonda sınuse matrıs verırsen sonuc ne oluyor.
        #
        # matlab gibyidi yanlış hatırlamıyorusam
        # DİREK REEL VERMENE GEREK YOK YANİ LIST VERINCE SAN
        # deneyele bı kardaşşşş . A LIST DONUYOR SANIRIM
        # PYTHON DA OLMUYORMUŞ NUMPY LAZIM.


'''However, you are allowed to use linear algebra primitives, which includes operations like matrix addition,
multiplication, element-wise operators, matrix inversion and eigenvalue decomposition. When you are in doubt, please don't hesitate to contact us.
'''
# SİGMOİD 1/(1+exp(-x)) değil mi? BUNA NASIL MATRIX VERİYORUZ Kİ ÇÖZEMEDİM?

#BAK COURSERA ÖDEVİNDE ŞUNU SÖYLÜYOR
'''
For
large positive values of x, the sigmoid should be close to 1, while for large
negative values, the sigmoid should be close to 0. Evaluating sigmoid(0)
should give you exactly 0.5. Your code should also work with vectors and
matrices. For a matrix, your function should perform the sigmoid
function on every element.
'''
        grad1 = 1.0/m * sum([(t0 + t1*x[i] - y[i])*x[i] for i in range(m)]) # tamam anladım ŞİMDİ. OK.

        # update the theta_temp
        # buralarda sıkıntı yok dı mı
        # alpha YOK HACI LEARNING RATE BURASI .:D :D learnıng rate aynen klavyemden aldın :D
        temp0 = t0 - alpha * grad0
        temp1 = t1 - alpha * grad1

        # update theta
        t0 = temp0
        t1 = temp1

        # mean squared error
        e = sum( [ (t0 + t1*x[i] - y[i])**2 for i in range(m)] ) # BU NEYDİ COST FUNCTION MIYDI aynen
        # ıste parametrelerı update ettıkten sonra tekrar hesaplıyor
        # AMA SQUARE COST FUNCTION LINEAR REGRESSION İÇİN GEÇERLİYDİ. BİZ LOGISTIC REG. YAPCAZ LOG'LU Bİ COST FUNC. LAZIM.
				# ya aslında evet bu cost functıon degıl.
        # ama bunun amacı zaten cost functıonunu olcmek degıl
        # sadece bır oncekı ıterastona gore ne kadar degısmıs ona bakıp ıterasyonu bıtırıp bıtırmemeye karar vermeye yarıyor.
        # o loglu cost fonksıyonunun turevı yukardakı grad0 ve grad1 de verılmıs zaten. lazım olan kısım orasu
        # eger her ıterasyonda dogru olan costu yazdırmak fıkrındeysen dogru erroru hesaplamak mantıklı.

        #hatırladğım kadarıyla square cost fcn. logistic'de sıkıntı çıakrıyordu. iteration durup durmamaya karar verirken de etkili olabilir heralde . emin değilim ama..
        # ya bence olmamalı. cunku hanı errorun bu versıyonundakı degısım 0.000001 den azsa gercek costtakı degısım
        # OLABİLİ.R..
        # yanı bence hanı o loglu kısım yukarda gradlarda kullanılmıs zaten. burda kullanılmasının amacı descentı bıtırıp bıtırmemeye karar vermek bence. learnıng le alakası yok.
        # cunku gradlar loga gore alınmıs.
        # EVET DOĞRU SÖYLÜYORSUN. SADAKTE VE BİLHAKKI NATAKTE. BEN SADECE BİTİRP BİTİRMEMEYTE DE NEGATİF ETKİ YAPABİLİR DİYE DÜŞÜNDÜM. ÇOK ÖNEMLİ DEĞİL EN KÖTÜ LOGLU İLE DEĞİŞTİRİRİM ZOR DEĞİL
        # TAMAM BU KISMI DA ANLADIM. BU FONKSİYON
        # sanırım turevler yanlıs alınmıs
        # cunku ıflı bıseydı o ya..
        # sadece gradıent degısecek telasa gerek yok
        # ABİ GRADIENT (H_THETA(X[i])-Y[i])*Xj evet o zaman h_thetanın bu versıyonunda bı sıkıntı var

        # YA BU IMPLEMENTATION BİZİM 3 PARAMETLİYE UYUYOR MU Kİ? KAFAM KARIŞIYOR..  onda sıkıntı yok daç
        # grad1 = 1.0/m * sum([(t0 + t1*x[i] - y[i])*x[i] for i in range(m)]) YA BURADA SON KISIMDA *x[i] demiş ama.. BENİM X'LERDE YANİ EXAMPLELARDA DATA DICTIONARY.
        # BENDE DATA SET ŞÖYLE
        '''
        [{'Survived': 1, 'Age': 22, 'Sex': 2, 'Pclass': 2},
        	{'Survived': 1, 'Age': 22, 'Sex': 2, 'Pclass': 2},
          {'Survived': 1, 'Age': 22, 'Sex': 2, 'Pclass': 2}]
					her bir iterationda x olarak bu dictionarylerden birini veriyor.
        '''
        """
        	kanka bunu kabul etmez su anda.
          ben boyle dusunmemıstım.
          boyle bıldıgın bı matrıs
          ılk kolon yasayıp yasamadıgı
          ıkıncısı...
          gıbı her satırda bı ınstance
          ondan bunda degısmek


        	BEN ML IMPLEMENTATION YAPMADIĞIM İÇİN HİÇ WEB APP YAZAR GİBİ DICTIONARY'E DOLDURDUM NE BİLEYİM :)
          MATRIXLERLE ARAM YOK HİÇ .
          STAJI WEB APP'LERDE YAPTIM HOCAM NAPAYIM :)
          TAMAM HALLEDERİM BEN ONU. MATRIXIN YAPISI NASIL OLACAK PEKİ? KAÇA KAÇLIK MATRIX VS.
        	ahahahahahahahahhaahhahah
          saglam tespıt... gercekten tam bı web developer gıbı dusunmusun :D
          kanka onu ondan ona degısmesı sıkıntı degıl de su anda daha cıddı bı sorunumuz var :D

					gradıent genel anlamda dogru ama ıcındkı h-theta kısmı yanlıs. İŞTE BENİM KAFAM KARIŞAN YER DE ORASIYDI DÜN AKŞAM. H_THETA VERMEK LAZIM SUM'IN İÇİNE AMA NASIL OLACAK O İŞ?
          satır 302 ye bı bakale

        """
        #sum += ( (-y_i * math.log10(h_theta)) - ((1 - y_i) * math.log10(1 - h_theta)) )
				#e = -1.0/m * sum( [y[i] * math.log10()


        if abs(J-e) <= ep:
            print 'Converged, iterations: ', iter, '!!!'
            converged = True

        J = e   # update error
        iter += 1  # update iter

        if iter == max_iter:
            print 'Max interactions exceeded!'
            converged = True

    return t0,t1



def read_csv(filename):
    dataset = []
    splitted_data = {'training': [], 'validation': [], 'test':[]}

    with open(filename) as csvfile:
        read_array = lambda line: line.strip('\r\n').split(',')
        attributes = read_array(csvfile.readline())

        for row in csvfile:
            dataset.append({attr: float(value) if attr == 'Age' else int(value) for attr, value in zip(attributes, read_array(row))})

        normalize_features(dataset)

        splitted_data['training'] = dataset[:401]
        splitted_data['validation'] = dataset[401:701]
        splitted_data['test'] = dataset[701:]

        return splitted_data


def normalize_features(dataset):
    age_set = [data['Age'] for data in dataset]
    min_age, max_age = min(age_set), max(age_set)

    for index in range(len(dataset)):
        dataset[index]['Age'] = (dataset[index]['Age'] - min_age) / (max_age - min_age)



def next_batch(X, y, batchSize):
	# loop over our dataset `X` in mini-batches of size `batchSize`
	for i in np.arange(0, X.shape[0], batchSize):
		# yield a tuple of the current batched data and labels
		yield (X[i:i + batchSize], y[i:i + batchSize])


def sigmoid(z):
    return 1 / (1 + math.exp(-z))


def hypothesis(**kwargs): #theta_set, param_set, theta0
    sum = kwargs['theta0'] # HALLOLDU BİLE
    for theta, x in zip(kwargs['theta_set'], kwargs['param_set']):
        sum += theta * x #### burda theta0 nerede ????? UNUTMUŞUM ONU HACI YAV
                           # sımdı onu eklememız lazım bu bır.
                           # bunu ekleyınce sen yukardakı gradıent descent fonskıyonu ıcınde for kullanmadan bı metodla foru moru halletmıs olacan Allah'ın (C.C.) (S.W.T.) ıznıynen
    return sum
                           #arapcada buyuk harf varmı. DUMURRRRRRRR. ÖFF:) LATİNCE DE VAR HOCU :d:d:d:d
                           # bu metodu duzeltmeyı sana bırakıyorum :D ben yukardayım


def cost_function(theta_set, data_set):
    m = len(data_set)
    sum = 0
    for example in data_set:
        y_i = example['Survived']
        param_set = [ example['Age'], example['Pclass'], example['Sex']]
        h_theta = hypothesis(theta_set=theta_set, param_set=param_set)

        sum += ( (-y_i * math.log10(h_theta)) - ((1 - y_i) * math.log10(1 - h_theta)) )

    return sum / m


'''
theta_j_param_name: derivative of J(theta) is going to be taken according to that parameter
theta_set: current theta values
dataset: example rows
'''
def gradient_function(theta_j_param_name, theta_set, data_set):
    m = len(data_set)
    sum = 0
    for example in data_set:
        y_i = example['Survived']
        param_set = [ example['Age'], example['Pclass'], example['Sex']]
        h_theta = hypothesis(theta_set=theta_set, param_set=param_set) #calculate h_theta of x

        sum += ( h_theta - y_i) * example[theta_j_param_name]

    return sum / m


def converge():
    theta_set =

def test():
    # x = np.arange(-20, 20, 0.1)
    # y = list(map(sigmoid, x))
    # plt.plot(x, y)
    # plt.show()

    normalized_data = read_csv('titanicdata.csv')




test()

