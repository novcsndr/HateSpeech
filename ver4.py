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
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
import random
import json
from gensim import corpora, models
import gensim


stopword = ["ada","adalah","adanya","adapun","agak","agaknya","agar","akan","akankah","akhir","akhiri","akhirnya","aku","akulah","amat","amatlah","anda","andalah","antar","antara","antaranya","apa","apaan","apabila","apakah","apalagi","apatah","artinya","asal","asalkan","atas","atau","ataukah","ataupun","awal","awalnya","bagai","bagaikan","bagaimana","bagaimanakah","bagaimanapun","bagi","bagian","bahkan","bahwa","bahwasanya","baik","bakal","bakalan","balik","banyak","bapak","baru","bawah","beberapa","begini","beginian","beginikah","beginilah","begitu","begitukah","begitulah","begitupun","bekerja","belakang","belakangan","belum","belumlah","benar","benarkah","benarlah","berada","berakhir","berakhirlah","berakhirnya","berapa","berapakah","berapalah","berapapun","berarti","berawal","berbagai","berdatangan","beri","berikan","berikut","berikutnya","berjumlah","berkali-kali","berkata","berkehendak","berkeinginan","berkenaan","berlainan","berlalu","berlangsung","berlebihan","bermacam","bermacam-macam","bermaksud","bermula","bersama","bersama-sama","bersiap","bersiap-siap","bertanya","bertanya-tanya","berturut","berturut-turut","bertutur","berujar","berupa","besar","betul","betulkah","biasa","biasanya","bila","bilakah","bisa","bisakah","boleh","bolehkah","bolehlah","buat","bukan","bukankah","bukanlah","bukannya","bulan","bung","cara","caranya","cukup","cukupkah","cukuplah","cuma","dahulu","dalam","dan","dapat","dari","daripada","datang","dekat","demi","demikian","demikianlah","dengan","depan","di","dia","diakhiri","diakhirinya","dialah","diantara","diantaranya","diberi","diberikan","diberikannya","dibuat","dibuatnya","didapat","didatangkan","digunakan","diibaratkan","diibaratkannya","diingat","diingatkan","diinginkan","dijawab","dijelaskan","dijelaskannya","dikarenakan","dikatakan","dikatakannya","dikerjakan","diketahui","diketahuinya","dikira","dilakukan","dilalui","dilihat","dimaksud","dimaksudkan","dimaksudkannya","dimaksudnya","diminta","dimintai","dimisalkan","dimulai","dimulailah","dimulainya","dimungkinkan","dini","dipastikan","diperbuat","diperbuatnya","dipergunakan","diperkirakan","diperlihatkan","diperlukan","diperlukannya","dipersoalkan","dipertanyakan","dipunyai","diri","dirinya","disampaikan","disebut","disebutkan","disebutkannya","disini","disinilah","ditambahkan","ditandaskan","ditanya","ditanyai","ditanyakan","ditegaskan","ditujukan","ditunjuk","ditunjuki","ditunjukkan","ditunjukkannya","ditunjuknya","dituturkan","dituturkannya","diucapkan","diucapkannya","diungkapkan","dong","dua","dulu","empat","enggak","enggaknya","entah","entahlah","guna","gunakan","hal","hampir","hanya","hanyalah","hari","harus","haruslah","harusnya","hendak","hendaklah","hendaknya","hingga","ia","ialah","ibarat","ibaratkan","ibaratnya","ibu","ikut","ingat","ingat-ingat","ingin","inginkah","inginkan","ini","inikah","inilah","itu","itukah","itulah","jadi","jadilah","jadinya","jangan","jangankan","janganlah","jauh","jawab","jawaban","jawabnya","jelas","jelaskan","jelaslah","jelasnya","jika","jikalau","juga","jumlah","jumlahnya","justru","kala","kalau","kalaulah","kalaupun","kalian","kami","kamilah","kamu","kamulah","kan","kapan","kapankah","kapanpun","karena","karenanya","kasus","kata","katakan","katakanlah","katanya","ke","keadaan","kebetulan","kecil","kedua","keduanya","keinginan","kelamaan","kelihatan","kelihatannya","kelima","keluar","kembali","kemudian","kemungkinan","kemungkinannya","kenapa","kepada","kepadanya","kesampaian","keseluruhan","keseluruhannya","keterlaluan","ketika","khususnya","kini","kinilah","kira","kira-kira","kiranya","kita","kitalah","kok","kurang","lagi","lagian","lah","lain","lainnya","lalu","lama","lamanya","lanjut","lanjutnya","lebih","lewat","lima","luar","macam","maka","makanya","makin","malah","malahan","mampu","mampukah","mana","manakala","manalagi","masa","masalah","masalahnya","masih","masihkah","masing","masing-masing","mau","maupun","melainkan","melakukan","melalui","melihat","melihatnya","memang","memastikan","memberi","memberikan","membuat","memerlukan","memihak","meminta","memintakan","memisalkan","memperbuat","mempergunakan","memperkirakan","memperlihatkan","mempersiapkan","mempersoalkan","mempertanyakan","mempunyai","memulai","memungkinkan","menaiki","menambahkan","menandaskan","menanti","menanti-nanti","menantikan","menanya","menanyai","menanyakan","mendapat","mendapatkan","mendatang","mendatangi","mendatangkan","menegaskan","mengakhiri","mengapa","mengatakan","mengatakannya","mengenai","mengerjakan","mengetahui","menggunakan","menghendaki","mengibaratkan","mengibaratkannya","mengingat","mengingatkan","menginginkan","mengira","mengucapkan","mengucapkannya","mengungkapkan","menjadi","menjawab","menjelaskan","menuju","menunjuk","menunjuki","menunjukkan","menunjuknya","menurut","menuturkan","menyampaikan","menyangkut","menyatakan","menyebutkan","menyeluruh","menyiapkan","merasa","mereka","merekalah","merupakan","meski","meskipun","meyakini","meyakinkan","minta","mirip","misal","misalkan","misalnya","mula","mulai","mulailah","mulanya","mungkin","mungkinkah","nah","naik","namun","nanti","nantinya","nyaris","nyatanya","oleh","olehnya","pada","padahal","padanya","pak","paling","panjang","pantas","para","pasti","pastilah","penting","pentingnya","per","percuma","perlu","perlukah","perlunya","pernah","persoalan","pertama","pertama-tama","pertanyaan","pertanyakan","pihak","pihaknya","pukul","pula","pun","punya","rasa","rasanya","rata","rupanya","saat","saatnya","saja","sajalah","saling","sama","sama-sama","sambil","sampai","sampai-sampai","sampaikan","sana","sangat","sangatlah","satu","saya","sayalah","se","sebab","sebabnya","sebagai","sebagaimana","sebagainya","sebagian","sebaik","sebaik-baiknya","sebaiknya","sebaliknya","sebanyak","sebegini","sebegitu","sebelum","sebelumnya","sebenarnya","seberapa","sebesar","sebetulnya","sebisanya","sebuah","sebut","sebutlah","sebutnya","secara","secukupnya","sedang","sedangkan","sedemikian","sedikit","sedikitnya","seenaknya","segala","segalanya","segera","seharusnya","sehingga","seingat","sejak","sejauh","sejenak","sejumlah","sekadar","sekadarnya","sekali","sekali-kali","sekalian","sekaligus","sekalipun","sekarang","sekarang","sekecil","seketika","sekiranya","sekitar","sekitarnya","sekurang-kurangnya","sekurangnya","sela","selain","selaku","selalu","selama","selama-lamanya","selamanya","selanjutnya","seluruh","seluruhnya","semacam","semakin","semampu","semampunya","semasa","semasih","semata","semata-mata","semaunya","sementara","semisal","semisalnya","sempat","semua","semuanya","semula","sendiri","sendirian","sendirinya","seolah","seolah-olah","seorang","sepanjang","sepantasnya","sepantasnyalah","seperlunya","seperti","sepertinya","sepihak","sering","seringnya","serta","serupa","sesaat","sesama","sesampai","sesegera","sesekali","seseorang","sesuatu","sesuatunya","sesudah","sesudahnya","setelah","setempat","setengah","seterusnya","setiap","setiba","setibanya","setidak-tidaknya","setidaknya","setinggi","seusai","sewaktu","siap","siapa","siapakah","siapapun","sini","sinilah","soal","soalnya","suatu","sudah","sudahkah","sudahlah","supaya","tadi","tadinya","tahu","tahun","tak","tambah","tambahnya","tampak","tampaknya","tandas","tandasnya","tanpa","tanya","tanyakan","tanyanya","tapi","tegas","tegasnya","telah","tempat","tengah","tentang","tentu","tentulah","tentunya","tepat","terakhir","terasa","terbanyak","terdahulu","terdapat","terdiri","terhadap","terhadapnya","teringat","teringat-ingat","terjadi","terjadilah","terjadinya","terkira","terlalu","terlebih","terlihat","termasuk","ternyata","tersampaikan","tersebut","tersebutlah","tertentu","tertuju","terus","terutama","tetap","tetapi","tiap","tiba","tiba-tiba","tidak","tidakkah","tidaklah","tiga","tinggi","toh","tunjuk","turut","tutur","tuturnya","ucap","ucapnya","ujar","ujarnya","umum","umumnya","ungkap","ungkapnya","untuk","usah","usai","waduh","wah","wahai","waktu","waktunya","walau","walaupun","wong","yaitu","yakin","yakni","yang"]

# ==================================================
# Written in March 2016 by Victoria Anugrah Lestari
# ==================================================

# ==================================================
# Membaca dictionary dari json
# ==================================================
def load(filename):
	with open(filename) as data_file:
		data = json.load(data_file)

	return data

# load dictionary
mydict = load('dict.json')

# ==================================================
# Mencari sinonim suatu kata
# ==================================================
def getSinonim(word):
	if word in mydict.keys():
		return mydict[word]['sinonim']
	else:
		return []

# ==================================================
# Mencari antonim suatu kata
# ==================================================
def getAntonim(word):
	if word in mydict.keys():
		if 'antonim' in mydict[word].keys():
			return mydict[word]['antonim']

	return []

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

    f = open(folder_non_hs, "r")
    for line in f:
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
		subj.append(line.split()[0])
		topik.append(line.split(" ")[1])
		temp_kalimat = ' '.join(line.split()[2:])
		kalimat.append(temp_kalimat)

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
	hs_number=453
	i=0
	hs_list=[]
	nonhs_list=[]
	for line in f:
		if i <= hs_number:
			hs_list.append(line)
		elif i> hs_number:
			nonhs_list.append(line)
		i = i + 1
	train_all= hs_list[:int((len(hs_list)*70)/100)]+nonhs_list[:int((len(nonhs_list)*70)/100)]
	test_all= hs_list[int((len(hs_list)*70)/100):]+nonhs_list[int((len(nonhs_list)*70)/100):]
	for line in train_all:
		msg = line.strip().split("\t")
		hs.append(msg[0].translate(None, ' '))
		sbj.append(msg[1].translate(None, ' '))
		topik.append(msg[2])
		kalimat.append(msg[3])
	for line in test_all:
		msg = line.strip().split("\t")
		hs.append(msg[0].translate(None, ' '))
		sbj.append(msg[1].translate(None, ' '))
		topik.append(msg[2])
		kalimat.append(msg[3])

	return hs,sbj,topik,kalimat


def preprocess(sentence):
	sentence=sentence.decode("utf8")
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
    #clf = RandomForestClassifier(max_depth=2, random_state=0)
    text_clf = clf.fit(feature_train, label_train)
    predicted = clf.predict(feature_test)

    accuracy = accuracy_score(label_test, predicted )
    return predicted
def train_random_forest(feature_train, label_train, feature_test, label_test):
    #clf = LinearSVC()
    clf = RandomForestClassifier(max_depth=2, random_state=0)
    text_clf = clf.fit(feature_train, label_train)
    predicted = clf.predict(feature_test)

    accuracy = accuracy_score(label_test, predicted )
    return predicted

def train_sgd(feature_train, label_train, feature_test, label_test):
    clf = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-10, random_state=42)
    text_clf = clf.fit(feature_train, label_train)
    predicted = clf.predict(feature_test)
    accuracy =accuracy_score(label_test, predicted)
    return predicted
def train_neural_network(feature_train, label_train, feature_test, label_test):
    clf = MLPClassifier()
    text_clf = clf.fit(feature_train, label_train)
    predicted = clf.predict(feature_test)
    accuracy =accuracy_score(label_test, predicted)
    return predicted

all_hs = []
all_subj = []
all_topic = []
all_sent = []

# hs, subj, topik, kalimat = read_data_mena()
# [all_hs.append(lab.lower()) for lab in hs]
# [all_subj.append(lab.lower()) for lab in subj]
# [all_topic.append(lab.lower()) for lab in topik]
# [all_sent.append(lab.lower()) for lab in kalimat]
# kalimat, topik, subj, hs = read_data_novi("provo-nop.txt","nonprovo-nop.txt")
# [all_hs.append(lab.lower()) for lab in hs]
# [all_subj.append("subj") if lab == "sub" else all_subj.append("nonsubj") for lab in subj]
# [all_topic.append(lab.lower()) for lab in topik]
# [all_sent.append(lab.lower()) for lab in kalimat]
# hs, subj, topik, kalimat = read_data_sari()
# [all_hs.append(lab.lower()) for lab in hs]
# [all_subj.append(lab.lower()) for lab in subj]
# [all_topic.append(lab.lower()) for lab in topik]
# [all_sent.append(lab.lower()) for lab in kalimat]
hs, subj, topik, kalimat = read_data_laksmi("datasetlaksmi.txt")
[all_hs.append(lab.lower()) for lab in hs]
[all_subj.append(lab.lower()) for lab in subj]
[all_topic.append(lab.lower()) for lab in topik]
[all_sent.append(lab.lower()) for lab in kalimat]

## ekstraksi fitur
all_feature_train=[]
for i in range(0,len(all_sent)):
    all_feature_train.append(preprocess(all_sent[i]))

modified_doc = [' '.join(i) for i in all_feature_train]
mod_doc = []
for doc in modified_doc:
    mod_doc.append(doc.encode('ascii', 'ignore'))

mod_doc_train = mod_doc[:int((len(mod_doc)*70)/100)]
mod_doc_test = mod_doc[int((len(all_hs)*70)/100):]

vectorizer = CountVectorizer()
transformer = TfidfTransformer()
vectorizer.fit(mod_doc_train)
trainVectorizerArray = vectorizer.transform(mod_doc_train).toarray()
data_train = trainVectorizerArray
transformer.fit(data_train)
tfidf = transformer.transform(data_train).toarray()
data_test = vectorizer.transform(mod_doc_test).toarray()
tfidf_test = transformer.transform(data_test).toarray()
subj_train = all_subj[:int((len(all_hs)*70)/100)]
hs_train = all_hs[:int((len(all_hs)*70)/100)]
subj_test = all_subj[int((len(all_hs)*70)/100):]
hs_test = all_hs[int((len(all_hs)*70)/100):]

data_trains =all_feature_train[:int((len(all_feature_train)*70)/100)]
data_tests =all_feature_train[int((len(all_feature_train)*70)/100):]

sentence_train_subj = []
new_data_train_subj = []
new_label_train_subj = []
counter_subjektif = 0
counter_objektif = 0
data_tr_top = []
data_top = []
for i in range(len(subj_train)):
    if subj_train[i] == "subj":
        counter_subjektif += 1
        data_tr_top.append(data_trains[i])
        sentence_train_subj.append(mod_doc_train[i])
        new_label_train_subj.append(subj_train[i])
        new_data_train_subj.append(tfidf[i])
    else:
        counter_objektif += 1
        sentence_train_subj.append(mod_doc_train[i])
        new_label_train_subj.append(subj_train[i])
        new_data_train_subj.append(tfidf[i])
        data_top.append(data_tests[i])

if counter_objektif < counter_subjektif:
	difference = counter_subjektif - counter_objektif
	data_aug = int(difference/counter_objektif)
	obj_data = []
	vector_obj_data = []
	for i in range(len(new_label_train_subj)):
		if new_label_train_subj[i] == "nonsubj":
			obj_data.append(sentence_train_subj[i])
	for sent in obj_data:
		counter_new = 0
		sent_temp = sent.split()
		for w in sent_temp:
			sentences = sent_temp
			if getSinonim(w) != []:
				sentences[sentences == w] = getSinonim(w)[0]
				if (counter_new < 10):
					sentence_train_subj.append(' '.join(sentences))
					vector_obj_data.append(' '.join(sentences))
					new_label_train_subj.append("nonsubj")
					counter_new += 1
	temp_vector = vectorizer.transform(vector_obj_data).toarray()
	tfidf_new = transformer.transform(temp_vector).toarray()
	for t in tfidf_new:
		new_data_train_subj.append(t)

# print np.unique(np.asarray(new_label_train_subj), return_counts=True)
new_data_train_subj_bal = []
new_label_train_subj_bal = []
counter_subj = 0
counter_non = 0
for i in range(len(new_label_train_subj)):
	if new_label_train_subj[i] == 'subj' and counter_subj < 405:
		new_label_train_subj_bal.append(new_label_train_subj[i])
		new_data_train_subj_bal.append(new_data_train_subj[i])
		counter_subj += 1
	elif new_label_train_subj[i] == 'nonsubj' and counter_non < 405:
		new_label_train_subj_bal.append(new_label_train_subj[i])
		new_data_train_subj_bal.append(new_data_train_subj[i])
		counter_non += 1
	else:
		continue

data_topic = []
result_subj = train_svm2(np.asarray(new_data_train_subj_bal),new_label_train_subj_bal,tfidf_test,subj_test)
counter = 0
for i in range(len(result_subj)):
    if result_subj[i] == subj_test[i]:
        counter += 1
        if result_subj[i] == "subj":
            data_topic.append(data_tests[i])


print (str(float(counter/len(result_subj))))
a=np.unique(np.asarray(new_label_train_subj_bal), return_counts=True)
print(a)
label_ts_subj = []
data_subj = []
data_tr_subj = []
label_tr_subj = []
counter = 0
counter_hs = 0
counter_nonhs = 0
sent_hs_subj_train = []
for i in range(len(hs_train)):
	if hs_train[i] == 'hs':
		label_tr_subj.append(hs_train[i])
		data_tr_subj.append(tfidf[i])
	else:
		if subj_train[i] == 'subj' and counter_hs < 181:
			label_tr_subj.append(hs_train[i])
			data_tr_subj.append(tfidf[i])
			counter_hs += 1
		else:
			continue



b= np.unique(np.asarray(label_tr_subj), return_counts=True)
print(b)
# for i in xrange(len(subj_train)):
# 	if hs_train[i] == 'hs':
# 		label_tr_subj.append(hs_train[i])
# 		data_tr_subj.append(tfidf[i])
# 		sent_hs_subj_train.append(mod_doc_train[i])
# 		counter_hs += 1
# 	else:
# 		if subj_train[i] == 'subj':
# 			label_tr_subj.append(hs_train[i])
# 			data_tr_subj.append(tfidf[i])
# 			sent_hs_subj_train.append(mod_doc_train[i])
# 			counter += 1
# 			counter_nonhs += 1
# 		else:
# 			continue

# if counter_hs < counter_nonhs:
# difference = counter_nonhs - counter_hs
# data_aug = int(difference/counter_hs)
# obj_data = []
# vector_obj_data = []
# for i in xrange(len(label_tr_subj)):
# 	if label_tr_subj[i] == "hs":
# 		obj_data.append(sent_hs_subj_train[i])
# for sent in obj_data:
# 	counter_new = 0
# 	sent_temp = sent.split()
# 	for w in sent_temp:
# 		sentences = sent_temp
# 		if getSinonim(w) != []:
# 			sentences[sentences == w] = getSinonim(w)[0]
# 			if (counter_new < 1):
# 				sent_hs_subj_train.append(' '.join(sentences))
# 				vector_obj_data.append(' '.join(sentences))
# 				label_tr_subj.append("hs")
# 				counter_new += 1
# temp_vector = vectorizer.transform(vector_obj_data).toarray()
# tfidf_new = transformer.transform(temp_vector).toarray()
# for t in tfidf_new:
# 	data_tr_subj.append(t)

# difference = counter_nonhs - counter_hs
# data_aug = int(difference/counter_hs)
# obj_data = []
# vector_obj_data = []
# for i in xrange(len(label_tr_subj)):
# 	if label_tr_subj[i] == "non_hs":
# 		obj_data.append(sent_hs_subj_train[i])
# for sent in obj_data:
# 	counter_new = 0
# 	sent_temp = sent.split()
# 	for w in sent_temp:
# 		sentences = sent_temp
# 		if getSinonim(w) != []:
# 			sentences[sentences == w] = getSinonim(w)[0]
# 			if (counter_new < 1):
# 				sent_hs_subj_train.append(' '.join(sentences))
# 				vector_obj_data.append(' '.join(sentences))
# 				label_tr_subj.append("non_hs")
# 				counter_new += 1
# temp_vector = vectorizer.transform(vector_obj_data).toarray()
# tfidf_new = transformer.transform(temp_vector).toarray()
# for t in tfidf_new:
# 	data_tr_subj.append(t)

# new_data_train_hs_bal = []
# new_label_train_hs_bal = []
# counter_hs = 0
# counter_nonhs = 0
# for i in xrange(len(label_tr_subj)):
# 	if label_tr_subj[i] == 'hs' and counter_hs < 358:
# 		new_label_train_hs_bal.append(label_tr_subj[i])
# 		new_data_train_hs_bal.append(data_tr_subj[i])
# 		counter_hs += 1
# 	elif label_tr_subj[i] == 'non_hs' and counter_nonhs < 358:
# 		new_label_train_hs_bal.append(label_tr_subj[i])
# 		new_data_train_hs_bal.append(data_tr_subj[i])
# 		counter_nonhs += 1
# 	else:
# 		continue

for i in range(len(subj_test)):
	if result_subj[i] == 'subj':
		data_subj.append(tfidf_test[i])
		label_ts_subj.append(hs_test[i])

# result_hs = train_svm2(np.asarray(new_data_train_hs_bal),new_label_train_hs_bal,data_subj,label_ts_subj)

# print np.unique(np.asarray(new_label_train_hs_bal), return_counts=True)

result_hs = train_svm2(np.asarray(data_tr_subj),label_tr_subj,data_subj,label_ts_subj)

counter = 0
training_topic=[]
for i in range(len(result_hs)):
    if result_hs[i] == label_ts_subj[i]:
        counter += 1
        if result_hs[i] == "hs":
            training_topic.append(data_topic[i])

print (str(float(counter/len(result_hs))()))

print("Seluruh Data LDA 1:")
dictionary = corpora.Dictionary(training_topic)
# convert tokenized documents into a document-term matrix
corpus = [dictionary.doc2bow(text) for text in training_topic]
print(corpus)
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=5,id2word = dictionary, passes=20)
print(ldamodel.print_topics(num_topics=5, num_words=8))
print('\nPerplexity: ', ldamodel.log_perplexity(corpus))

print("Seluruh Data LDA 2:")
ldamodel1 = gensim.models.ldamodel.LdaModel(corpus, num_topics=5,id2word = dictionary, passes=20)
print(ldamodel.print_topics(num_topics=5, num_words=8))
print('\nPerplexity: ', ldamodel1.log_perplexity(corpus))

print("Seluruh Data LDA 3:")
ldamodel2 = gensim.models.ldamodel.LdaModel(corpus, num_topics=5,id2word = dictionary, passes=20)
print(ldamodel.print_topics(num_topics=5, num_words=8))
print('\nPerplexity: ', ldamodel2.log_perplexity(corpus))

print("Seluruh Data LDA 4:")
ldamodel3 = gensim.models.ldamodel.LdaModel(corpus, num_topics=5,id2word = dictionary, passes=20)
print(ldamodel.print_topics(num_topics=5, num_words=8))
print('\nPerplexity: ', ldamodel3.log_perplexity(corpus))

print("Seluruh Data LDA 5:")
ldamodel4 = gensim.models.ldamodel.LdaModel(corpus, num_topics=5,id2word = dictionary, passes=20)
print(ldamodel.print_topics(num_topics=5, num_words=8))
print('\nPerplexity: ', ldamodel4.log_perplexity(corpus))


# coherence_model_lda = models.CoherenceModel(model=ldamodel, texts=data_lemmatized, dictionary=dictionary, coherence='c_v')
# coherence_lda = coherence_model_lda.get_coherence()
# print('\nCoherence Score: ', coherence_lda)


# counter = 0
print("Seluruh Data HDP:")
dictionary1 = corpora.Dictionary(training_topic)
# convert tokenized documents into a document-term matrix
corpus1 = [dictionary1.doc2bow(text) for text in training_topic]
print(corpus)
ldamodel1 = gensim.models.ldamodel.LdaModel(corpus1, num_topics=5,id2word = dictionary1, passes=20)
print(ldamodel1.print_topics(num_topics=5, num_words=8))
print("HDP:")
hdp =  gensim.models.hdpmodel.HdpModel(corpus, dictionary, max_chunks=None, max_time=None, chunksize=256, kappa=1.0, tau=64.0, K=15, T=-1, alpha=1, gamma=1, eta=0.01, scale=1.0, var_converge=0.0001, outputdir=None, random_state=None)
# doc_hdp = hdp[doc_bow]
print(hdp.print_topics(num_topics=-1, num_words=10))
print('\nPerplexity: ', hdp.log_perplexity(corpus))