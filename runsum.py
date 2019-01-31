#!/usr/bin/python
# encoding=utf-8
from subprocess import call
from glob import glob
from nltk.corpus import stopwords
import os, struct
from tensorflow.core.example import example_pb2
#import pyrouge
from pythonrouge.pythonrouge import Pythonrouge
import shutil
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.porter import *
stemmer = PorterStemmer()

ratio = 1
duc_num = 6
cmd = 'python run_summarization.py --mode=decode --single_pass=1 --coverage=True --vocab_path=../data/DMQA/finished_files/vocab --log_root=log --exp_name=myexperiment --data_path=test/temp_file --max_enc_steps=4000'
generated_path = 'log/myexperiment/decode_test_4000maxenc_4beam_35mindec_100maxdec_ckpt-238410/'
cmd = cmd.split()
stopwords = set(stopwords.words('english'))

max_len = 250


def pp(string):
    return ' '.join([stemmer.stem(word.decode('utf8')) for word in string.lower().split() if not word in stopwords])
    
def write_to_file(article, abstract, rel, writer):
    abstract = '<s> '+' '.join(abstract)+' </s>'
    #abstract = abstract.encode('utf8', 'ignore')
    #rel = rel.encode('utf8', 'ignore')
    #article = article.encode('utf8', 'ignore')
    tf_example = example_pb2.Example()
    tf_example.features.feature['abstract'].bytes_list.value.extend([bytes(abstract)])
    tf_example.features.feature['relevancy'].bytes_list.value.extend([bytes(rel)])
    tf_example.features.feature['article'].bytes_list.value.extend([bytes(article)])
    tf_example_str = tf_example.SerializeToString()
    str_len = len(tf_example_str)
    writer.write(struct.pack('q', str_len))
    writer.write(struct.pack('%ds' % str_len, tf_example_str))


def duck_iterator(i):
    duc_folder = 'duc0' + str(i) + 'tokenized/'
    print("Loading corpus from  " + duc_folder + "...")
    for topic in os.listdir(duc_folder + 'testdata/docs/'):
        topic_folder = duc_folder + 'testdata/docs/' + topic
        if not os.path.isdir(topic_folder):
            continue
        query = ' '.join(open(duc_folder + 'queries/' + topic).readlines())
        model_files = glob(duc_folder + 'models/' + topic[:-1].upper() + '.*')

        topic_texts = [' '.join(open(topic_folder + '/' + file).readlines()).replace('\n', '') for file in
                       os.listdir(topic_folder)]

        #abstracts = [' '.join(open(f).readlines()) for f in model_files]
        abstracts = [open(f).readlines() for f in model_files]
        yield topic_texts, abstracts, query

def ones(sent, ref): return 1.

def count_score(sent, ref):
    ref = pp(ref).split()
    sent = ' '.join(pp(w) for w in sent.lower().split() if not w in stopwords)
    return sum([1. if w in ref else 0. for w in sent.split()])


def get_w2v_score_func(magic = 10):
    import gensim
    google = gensim.models.KeyedVectors.load_word2vec_format(
        'GoogleNews-vectors-negative300.bin', binary=True)
    def w2v_score(sent, ref):
        ref = ref.lower()
        sent = sent.lower()
        sent = [w for w in sent.split() if w in google]
        ref = [w for w in ref.split() if w in google]
        try:
            score = google.n_similarity(sent, ref)
        except:
            score = 0.
        return score * magic
    return w2v_score

def get_tfidf_score_func_glob(magic = 1):
    corpus = []
    for i in range(5, 8):
        for topic_texts, _, _ in duck_iterator(i):
            corpus += [pp(t) for t in topic_texts]

    vectorizer = TfidfVectorizer()
    vectorizer.fit_transform(corpus)

    def tfidf_score_func(sent, ref):
        #ref = [pp(s) for s in ref.split(' . ')]
        sent = pp(sent)
        v1 = vectorizer.transform([sent])
        #v2s = [vectorizer.transform([r]) for r in ref]
        #return max([cosine_similarity(v1, v2)[0][0] for v2 in v2s])
        v2 = vectorizer.transform([ref])
        return cosine_similarity(v1, v2)[0][0]

    return tfidf_score_func

#tfidf_score = get_tfidf_score_func_glob()


def get_tfidf_score_func(magic = 10):
    corpus = []
    for i in range(5, 8):
        for topic_texts, _, _ in duck_iterator(i):
            corpus += [t.lower() for t in topic_texts]

    vectorizer = TfidfVectorizer()
    vectorizer.fit_transform(corpus)

    def tfidf_score_func(sent, ref):
        ref = ref.lower()
        sent = sent.lower()
        v1 = vectorizer.transform([sent])
        v2 = vectorizer.transform([ref])
        return cosine_similarity(v1, v2)[0][0]*magic
    return tfidf_score_func


def just_relevant(text, query):
    text = text.split(' . ')
    score_per_sent = [count_score(sent, query) for sent in text]
    sents_gold = list(zip(*sorted(zip(score_per_sent, text), reverse=True)))[1]
    sents_gold = sents_gold[:int(len(sents_gold)*ratio)]

    filtered_sents = []
    for s in text:
        if not s: continue
        if s in sents_gold: filtered_sents.append(s)
    return ' . '.join(filtered_sents)

class Summary:
    def __init__(self, texts, abstracts, query):
        #texts = sorted([(tfidf_score(query, text), text) for text in texts], reverse=True)
        #texts = sorted([(tfidf_score(text, ' '.join(abstracts)), text) for text in texts], reverse=True)

        #texts = [text[1] for text in texts]
        self.texts = texts
        self.abstracts = abstracts
        self.query = query
        self.summary = []
        self.words = set()
        self.length = 0

    def add_sum(self, summ):
        for sent in summ:
            self.summary.append(sent)

    def get(self):
        text = max([(len(t.split()), t) for t in  self.texts])[1]
        #text = texts[0]
        if ratio < 1: text = just_relevant(text, self.query)

        sents = text.split(' . ')
        score_per_sent = [(score_func(sent, self.query), sent) for sent in sents]
        #score_per_sent = [(count_score(sent, ' '.join(self.abstracts)), sent) for sent in sents]

        scores = []
        for score, sent in score_per_sent:
            scores += [score] * (len(sent.split()) + 1)
        scores = str(scores[:-1])
        return text, 'a', scores

def get_summaries(path):
    path = path+'decoded/'
    out = {}
    for file_name in os.listdir(path):
        index = int(file_name.split('_')[0])
        out[index] = open(path+file_name).readlines()
    return out



#def evaluate(summaries):
#    for path in ['eval/ref', 'eval/dec']:
#        if os.path.exists(path):
#            shutil.rmtree(path, True)
#        os.mkdir(path)
#    scores = []
#    for i, summ in enumerate(summaries):
#        rouge = Pythonrouge(summary_file_exist=False,
#                summary=summ.summary, reference=summ.abstracts,
#                n_gram=2, ROUGE_SU4=True, ROUGE_L=False,
#                recall_only=True, stemming=True, stopwords=True,
#                word_level=True, length_limit=True, length=50,
#                use_cf=False, cf=95, scoring_formula='average',
#                resampling=True, samples=1000, favor=True, p=0.5)
#        score = rouge.calc_score()
#        scores.append(score)
#    return scores



print("COMPUTE TF-IDF SCORES:")
#count_score # OLD
#score_func = ones#get_w2v_score_func()#get_tfidf_score_func()#count_score # OLD
#tfidf_score = get_tfidf_score_func_glob()
score_func = get_tfidf_score_func()


print("LOAD DOCUMENTS:")
summaries = [Summary(texts, abstracts, query) for texts, abstracts, query in duck_iterator(duc_num)]
print("DONE!")

print("CREATING FILES...")
#with open('test/temp_file', 'wb') as writer:
#    for summ in summaries:
#        article, abstract, scores = summ.get()
#        write_to_file(article, abstracts, scores, writer)
print("DONE!")


print("GENERATE SUMMARIES:")
#call(['rm', '-r', generated_path])
#call(cmd)
print("DONE!")


print("READING SUMMARIES FROM FILE...")
generated_summaries = get_summaries(generated_path)#
for i in range(len(summaries)):
    summaries[i].add_sum(generated_summaries[i])
print("DONE!")


print("EVALUATE SUMMARIES:")
scores = []
avg_r1 = 0.0
avg_r2 = 0.0
avg_l = 0.0
avg_su4 = 0.0
for i, summ in enumerate(summaries):
    rouge = Pythonrouge(summary_file_exist=False,
                summary=summ.summary, reference=summ.abstracts,
                n_gram=2, ROUGE_SU4=True, ROUGE_L=True,
                recall_only=True, stemming=True, stopwords=True,
                word_level=True, length_limit=True, length=50,
                use_cf=False, cf=95, scoring_formula='average',
                resampling=True, samples=1000, favor=True, p=0.5)
    score = rouge.calc_score()
    print score
    avg_r1 += score['ROUGE-1']
    avg_r2 += score['ROUGE-2']
    avg_l += score['ROUGE-L']
    avg_su4 += score['ROUGE-SU4']

    scores.append(score)
#scores = evaluate(summaries)
print("DONE!")

print("RESULTS FOR DUC 200" + str(duc_num) + ":")
print "AVERAGE ROUGE-1:   " + str(avg_r1 / len(summaries))
print "AVERAGE ROUGE-2:   " + str(avg_r2 / len(summaries))
print "AVERAGE ROUGE-L:   " + str(avg_l / len(summaries))
print "AVERAGE ROUGE-SU4: " + str(avg_su4 / len(summaries))

#print score_func 

