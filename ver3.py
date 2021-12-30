from __future__ import division
import re
import numpy as np
from nltk import word_tokenize, WordNetLemmatizer
from nltk import ngrams
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from gensim import corpora, models
import gensim

import gensim

stopword = ["ada","adalah","adanya","adapun","agak","agaknya","agar","akan","akankah","akhir","akhiri","akhirnya","aku","akulah","amat","amatlah","anda","andalah","antar","antara","antaranya","apa","apaan","apabila","apakah","apalagi","apatah","artinya","asal","asalkan","atas","atau","ataukah","ataupun","awal","awalnya","bagai","bagaikan","bagaimana","bagaimanakah","bagaimanapun","bagi","bagian","bahkan","bahwa","bahwasanya","baik","bakal","bakalan","balik","banyak","bapak","baru","bawah","beberapa","begini","beginian","beginikah","beginilah","begitu","begitukah","begitulah","begitupun","bekerja","belakang","belakangan","belum","belumlah","benar","benarkah","benarlah","berada","berakhir","berakhirlah","berakhirnya","berapa","berapakah","berapalah","berapapun","berarti","berawal","berbagai","berdatangan","beri","berikan","berikut","berikutnya","berjumlah","berkali-kali","berkata","berkehendak","berkeinginan","berkenaan","berlainan","berlalu","berlangsung","berlebihan","bermacam","bermacam-macam","bermaksud","bermula","bersama","bersama-sama","bersiap","bersiap-siap","bertanya","bertanya-tanya","berturut","berturut-turut","bertutur","berujar","berupa","besar","betul","betulkah","biasa","biasanya","bila","bilakah","bisa","bisakah","boleh","bolehkah","bolehlah","buat","bukan","bukankah","bukanlah","bukannya","bulan","bung","cara","caranya","cukup","cukupkah","cukuplah","cuma","dahulu","dalam","dan","dapat","dari","daripada","datang","dekat","demi","demikian","demikianlah","dengan","depan","di","dia","diakhiri","diakhirinya","dialah","diantara","diantaranya","diberi","diberikan","diberikannya","dibuat","dibuatnya","didapat","didatangkan","digunakan","diibaratkan","diibaratkannya","diingat","diingatkan","diinginkan","dijawab","dijelaskan","dijelaskannya","dikarenakan","dikatakan","dikatakannya","dikerjakan","diketahui","diketahuinya","dikira","dilakukan","dilalui","dilihat","dimaksud","dimaksudkan","dimaksudkannya","dimaksudnya","diminta","dimintai","dimisalkan","dimulai","dimulailah","dimulainya","dimungkinkan","dini","dipastikan","diperbuat","diperbuatnya","dipergunakan","diperkirakan","diperlihatkan","diperlukan","diperlukannya","dipersoalkan","dipertanyakan","dipunyai","diri","dirinya","disampaikan","disebut","disebutkan","disebutkannya","disini","disinilah","ditambahkan","ditandaskan","ditanya","ditanyai","ditanyakan","ditegaskan","ditujukan","ditunjuk","ditunjuki","ditunjukkan","ditunjukkannya","ditunjuknya","dituturkan","dituturkannya","diucapkan","diucapkannya","diungkapkan","dong","dua","dulu","empat","enggak","enggaknya","entah","entahlah","guna","gunakan","hal","hampir","hanya","hanyalah","hari","harus","haruslah","harusnya","hendak","hendaklah","hendaknya","hingga","ia","ialah","ibarat","ibaratkan","ibaratnya","ibu","ikut","ingat","ingat-ingat","ingin","inginkah","inginkan","ini","inikah","inilah","itu","itukah","itulah","jadi","jadilah","jadinya","jangan","jangankan","janganlah","jauh","jawab","jawaban","jawabnya","jelas","jelaskan","jelaslah","jelasnya","jika","jikalau","juga","jumlah","jumlahnya","justru","kala","kalau","kalaulah","kalaupun","kalian","kami","kamilah","kamu","kamulah","kan","kapan","kapankah","kapanpun","karena","karenanya","kasus","kata","katakan","katakanlah","katanya","ke","keadaan","kebetulan","kecil","kedua","keduanya","keinginan","kelamaan","kelihatan","kelihatannya","kelima","keluar","kembali","kemudian","kemungkinan","kemungkinannya","kenapa","kepada","kepadanya","kesampaian","keseluruhan","keseluruhannya","keterlaluan","ketika","khususnya","kini","kinilah","kira","kira-kira","kiranya","kita","kitalah","kok","kurang","lagi","lagian","lah","lain","lainnya","lalu","lama","lamanya","lanjut","lanjutnya","lebih","lewat","lima","luar","macam","maka","makanya","makin","malah","malahan","mampu","mampukah","mana","manakala","manalagi","masa","masalah","masalahnya","masih","masihkah","masing","masing-masing","mau","maupun","melainkan","melakukan","melalui","melihat","melihatnya","memang","memastikan","memberi","memberikan","membuat","memerlukan","memihak","meminta","memintakan","memisalkan","memperbuat","mempergunakan","memperkirakan","memperlihatkan","mempersiapkan","mempersoalkan","mempertanyakan","mempunyai","memulai","memungkinkan","menaiki","menambahkan","menandaskan","menanti","menanti-nanti","menantikan","menanya","menanyai","menanyakan","mendapat","mendapatkan","mendatang","mendatangi","mendatangkan","menegaskan","mengakhiri","mengapa","mengatakan","mengatakannya","mengenai","mengerjakan","mengetahui","menggunakan","menghendaki","mengibaratkan","mengibaratkannya","mengingat","mengingatkan","menginginkan","mengira","mengucapkan","mengucapkannya","mengungkapkan","menjadi","menjawab","menjelaskan","menuju","menunjuk","menunjuki","menunjukkan","menunjuknya","menurut","menuturkan","menyampaikan","menyangkut","menyatakan","menyebutkan","menyeluruh","menyiapkan","merasa","mereka","merekalah","merupakan","meski","meskipun","meyakini","meyakinkan","minta","mirip","misal","misalkan","misalnya","mula","mulai","mulailah","mulanya","mungkin","mungkinkah","nah","naik","namun","nanti","nantinya","nyaris","nyatanya","oleh","olehnya","pada","padahal","padanya","pak","paling","panjang","pantas","para","pasti","pastilah","penting","pentingnya","per","percuma","perlu","perlukah","perlunya","pernah","persoalan","pertama","pertama-tama","pertanyaan","pertanyakan","pihak","pihaknya","pukul","pula","pun","punya","rasa","rasanya","rata","rupanya","saat","saatnya","saja","sajalah","saling","sama","sama-sama","sambil","sampai","sampai-sampai","sampaikan","sana","sangat","sangatlah","satu","saya","sayalah","se","sebab","sebabnya","sebagai","sebagaimana","sebagainya","sebagian","sebaik","sebaik-baiknya","sebaiknya","sebaliknya","sebanyak","sebegini","sebegitu","sebelum","sebelumnya","sebenarnya","seberapa","sebesar","sebetulnya","sebisanya","sebuah","sebut","sebutlah","sebutnya","secara","secukupnya","sedang","sedangkan","sedemikian","sedikit","sedikitnya","seenaknya","segala","segalanya","segera","seharusnya","sehingga","seingat","sejak","sejauh","sejenak","sejumlah","sekadar","sekadarnya","sekali","sekali-kali","sekalian","sekaligus","sekalipun","sekarang","sekarang","sekecil","seketika","sekiranya","sekitar","sekitarnya","sekurang-kurangnya","sekurangnya","sela","selain","selaku","selalu","selama","selama-lamanya","selamanya","selanjutnya","seluruh","seluruhnya","semacam","semakin","semampu","semampunya","semasa","semasih","semata","semata-mata","semaunya","sementara","semisal","semisalnya","sempat","semua","semuanya","semula","sendiri","sendirian","sendirinya","seolah","seolah-olah","seorang","sepanjang","sepantasnya","sepantasnyalah","seperlunya","seperti","sepertinya","sepihak","sering","seringnya","serta","serupa","sesaat","sesama","sesampai","sesegera","sesekali","seseorang","sesuatu","sesuatunya","sesudah","sesudahnya","setelah","setempat","setengah","seterusnya","setiap","setiba","setibanya","setidak-tidaknya","setidaknya","setinggi","seusai","sewaktu","siap","siapa","siapakah","siapapun","sini","sinilah","soal","soalnya","suatu","sudah","sudahkah","sudahlah","supaya","tadi","tadinya","tahu","tahun","tak","tambah","tambahnya","tampak","tampaknya","tandas","tandasnya","tanpa","tanya","tanyakan","tanyanya","tapi","tegas","tegasnya","telah","tempat","tengah","tentang","tentu","tentulah","tentunya","tepat","terakhir","terasa","terbanyak","terdahulu","terdapat","terdiri","terhadap","terhadapnya","teringat","teringat-ingat","terjadi","terjadilah","terjadinya","terkira","terlalu","terlebih","terlihat","termasuk","ternyata","tersampaikan","tersebut","tersebutlah","tertentu","tertuju","terus","terutama","tetap","tetapi","tiap","tiba","tiba-tiba","tidak","tidakkah","tidaklah","tiga","tinggi","toh","tunjuk","turut","tutur","tuturnya","ucap","ucapnya","ujar","ujarnya","umum","umumnya","ungkap","ungkapnya","untuk","usah","usai","waduh","wah","wahai","waktu","waktunya","walau","walaupun","wong","yaitu","yakin","yakni","yang"]

def read_data_mena():
	data = "formalized_provokasi_mena.txt"
	file = open(data, "r")

	hs = []
	subj =[]
	topik = []
	kalimat = []
	for line in file:
		hs.append("HS")
		subj.append("SUBJ")
		topik.append(line.split("\t")[0])
		kalimat.append(line.split("\t")[1])

	data = "formalized_non_provokasi_mena.txt"
	file = open(data, "r")

	for line in file:
		hs.append("Non_HS")
		subj.append(line.split()[0])
		topik.append(line.split()[1])
		temp_kalimat = ' '.join(line.split()[2:])
		kalimat.append(temp_kalimat)

	return hs, subj, topik, kalimat

def read_data_novi(folder_hs, folder_non_hs):
    f = open(folder_hs, "r")
    kal = []
    topic = []
    subjectivity = []
    hs = []
    for line in f:
        msg = line.strip().split("\t")
        kal.append(msg[1])
        topic.append(msg[0])
        subjectivity.append("subj")
        hs.append("hs")

    subjectivity1 = []
    j = open(folder_non_hs, "r")
    for line in j:
        msg = line.strip().split("\t")
        kal.append(msg[2])
        topic.append(msg[0])
        subjectivity.append(msg[1])
        hs.append("non_hs")
    return kal, topic, subjectivity,hs

def read_data_sari():
	data = "labelling_provokasi_sari.txt"
	file = open(data, "r")

	hs = []
	subj =[]
	topik = []
	kalimat = []
	for line in file:
		hs.append("HS")
		subj.append("SUBJ")
		topik.append(line.split()[1])
		kalimat.append(' '.join(line.split()[2:]))

	data = "labeling_nonprovokasi_sari.txt"
	file = open(data, "r")

	for line in file:
		hs.append("Non_HS")
		subj.append(line.split()[0])
		topik.append(line.split()[1])
		temp_kalimat = ' '.join(line.split()[2:])
		kalimat.append(temp_kalimat)

	return hs, subj, topik, kalimat

def read_data_laksmi(folder):
    f = open(folder, "r")
    hs = []
    sbj = []
    topik = []
    kalimat =[]

    for line in f:
        msg = line.strip().split("\t")
        hs.append(msg[0])
        sbj.append(msg[1])
        topik.append(msg[2])
        kalimat.append(msg[3])

    return hs,sbj,topik,kalimat

def preprocess(sentence):
	sentence=sentence
	sentence=re.sub(r'\d+', '', sentence)
	sentence=re.sub(r'([^\s\w]|_)+', '', sentence)
	tokens = word_tokenize(sentence)
	temp_sentence = ' '.join(tokens)
	result = [word for word in tokens if word not in stopword]
	n_gram = ngrams(temp_sentence.split(), 2)
	for gram in n_gram:
		result.append(' '.join(gram))
	return result

def train_svm2(feature_train, label_train, feature_test, label_test):
    clf = LinearSVC()
    # clf = RandomForestClassifier(max_depth=2, random_state=0)
    text_clf = clf.fit(feature_train, label_train)
    predicted = clf.predict(feature_test)

    accuracy = accuracy_score(label_test, predicted )
    return predicted

all_hs = []
all_subj = []
all_topic = []
all_sent = []

hs, subj, topik, kalimat = read_data_mena()
[all_hs.append(lab.lower()) for lab in hs]
[all_subj.append(lab.lower()) for lab in subj]
[all_topic.append(lab.lower()) for lab in topik]
[all_sent.append(lab.lower()) for lab in kalimat]
kalimat, topik, subj, hs = read_data_novi("provo-nop.txt","nonprovo-nop.txt")
[all_hs.append(lab.lower()) for lab in hs]
[all_subj.append("subj") if lab == "sub" else all_subj.append("nonsubj") for lab in subj]
[all_topic.append(lab.lower()) for lab in topik]
[all_sent.append(lab.lower()) for lab in kalimat]
hs, subj, topik, kalimat = read_data_sari()
[all_hs.append(lab.lower()) for lab in hs]
[all_subj.append(lab.lower()) for lab in subj]
[all_topic.append(lab.lower()) for lab in topik]
[all_sent.append(lab.lower()) for lab in kalimat]
# hs, subj, topik, kalimat = read_data_laksmi("datasetlaksmi.txt")
# [all_hs.append(lab.lower()) for lab in hs]
# [all_subj.append(lab.lower()) for lab in subj]
# [all_topic.append(lab.lower()) for lab in topik]
# [all_sent.append(lab.lower()) for lab in kalimat]


## ekstraksi fitur
all_feature_train=[]
for i in range(0,len(all_sent)):
    all_feature_train.append(preprocess(all_sent[i]))

modified_doc = [' '.join(i) for i in all_feature_train]
mod_doc = []
for doc in modified_doc:
    mod_doc.append(doc.encode('ascii', 'ignore'))
vectorizer = CountVectorizer()
transformer = TfidfTransformer()
trainVectorizerArray = vectorizer.fit_transform(mod_doc).toarray()
data_train = trainVectorizerArray[:int((len(trainVectorizerArray)*70)/100)]
data_test = trainVectorizerArray[int((len(trainVectorizerArray)*70)/100):]
hs_train = all_subj[:int((len(all_hs)*70)/100)]
hs_tr_label = all_hs[:int((len(all_hs)*70)/100)]
hs_test = all_subj[int((len(all_hs)*70)/100):]
hs_label = all_hs[int((len(all_hs)*70)/100):]

data_trains =all_feature_train[:int((len(all_feature_train)*70)/100)]
data_tests =all_feature_train[int((len(all_feature_train)*70)/100):]

result_subj = train_svm2(data_train,hs_train,data_test,hs_test)
counter = 0
for i in range(len(result_subj)):
	if result_subj[i] == hs_test[i]:
		counter += 1

print(str(float(counter/len(result_subj))))
a = np.unique(np.asarray(hs_train), return_counts=True)
print(a)
label_ts_subj = []
data_subj = []
data_tr_subj = []
label_tr_subj = []
data_tr_top =[]
data_top = []

# for i in range(len(hs_train)):
# 	if hs_train[i] == 'subj':
# 		label_tr_subj.append(hs_tr_label[i])
#         data_tr_top.append(data_trains[i])
#         data_tr_subj.append(data_train[i])
#
#
#
# for i in range(len(hs_test)):
#     if result_subj[i] == 'subj':
#         data_subj.append(data_test[i])
#         label_ts_subj.append(hs_label[i])
#         data_top.append(data_tests[i])


for i in range(len(hs_train)):
	if hs_train[i] == 'subj':
		label_tr_subj.append(hs_tr_label[i])
		data_tr_subj.append(data_train[i])
		data_tr_top.append(data_trains[i])

for i in range(len(hs_test)):
	if result_subj[i] == 'subj':
		data_subj.append(data_test[i])
		label_ts_subj.append(hs_label[i])
		data_top.append(data_tests[i])


# print(len(data_tr_top))
# print(len(data_top))
data_train_topic= data_tr_top+data_top
print(len(data_train_topic))
print("Seluruh Data:")
dictionary = corpora.Dictionary(all_feature_train)
# convert tokenized documents into a document-term matrix
corpus = [dictionary.doc2bow(text) for text in all_feature_train]
print(corpus)
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=5,id2word = dictionary, passes=20)
print(ldamodel.print_topics(num_topics=5, num_words=8))

counter = 0
print("Seluruh Data Subj:")
dictionary1 = corpora.Dictionary(data_train_topic)
# convert tokenized documents into a document-term matrix
corpus1 = [dictionary1.doc2bow(text) for text in data_train_topic]
print(corpus)
ldamodel1 = gensim.models.ldamodel.LdaModel(corpus1, num_topics=5,id2word = dictionary1, passes=20)
print(ldamodel1.print_topics(num_topics=5, num_words=8))
print("HDP:")
hdp =  gensim.models.hdpmodel.HdpModel(corpus, dictionary)
# doc_hdp = hdp[doc_bow]
print(hdp.print_topics(num_topics=-1, num_words=10))

counter = 0
result_hs = train_svm2(data_tr_subj,label_tr_subj,data_subj,label_ts_subj)
for i in range(len(result_hs)):
	if result_hs[i] == label_ts_subj[i]:
		counter += 1

print(str(float(counter/len(result_hs))))

