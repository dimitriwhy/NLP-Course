from collections import Counter
from math import log
class HMM(object) :

    def __init__(self, training_data = None) :
        if training_data is not None :
            self.fit(training_data)

    def fit(self, training_data) :
        self.wordcount = Counter()
        self.tri_words = Counter()
        self.bi_words = Counter()
        sentences = training_data.split('\n')
        for sentence in sentences :
            words = sentence.split('\t')
            pre_pre_word = '<p>'
            pre_word = '<p>'
            for i in xrange(len(words)) :
                now_word = words[i]
                self.wordcount[now_word] += 1
                self.bi_words[(pre_word, now_word)] += 1
                self.tri_words[(pre_pre_word, pre_word, now_word)] += 1
                pre_pre_word = pre_word
                pre_word = now_word
    
    def trans(self, pre_pre_word, pre_word, now_word) :
        M = len(self.bi_words)
        return log(float(0.01 + self.tri_words[(pre_pre_word, pre_word, now_word)]) / (0.01 * M + self.bi_words[(pre_pre_word, pre_word)]));
    

def viterbi(hmm,line) :
    path = [(-1,-1,0) for i in xrange(len(line))]
    score = [-float("inf") for j in xrange(len(line))]
    #print len(line)
    for i in xrange(len(line)) :
        pre_pre_word = "<p>"
        pre_word = "<p>"
        #j-i   k-j-1    l-k-1
        if i != len(line) - 1 :
            now_word = line[:i + 1]
        else :
            now_word = line
        score[i] = hmm.trans(pre_pre_word,pre_word,now_word)
        for j in xrange(1, i) :
            now_word = line[j : i + 1]
            #print now_word.encode("gbk"),hmm.wordcount[now_word]
            for k in xrange(0,j) :
                pre_word = line[k : j]
                if k == 0 :
                    pre_pre_word = "<p>"
                    prob = hmm.trans(pre_pre_word,pre_word,now_word);
                    if (prob > score[i]) :
                        score[i] = prob
                        path[i] = (-1,0,j)
                else :
                    for l in xrange(0,k) :
                        if l == 0 :
                            pre_pre_word = line[ :k]
                            prob = hmm.trans(pre_pre_word,pre_word,now_word);
                        else :
                            pre_pre_word = line[l : k]
                            prob = score[i - 1] + hmm.trans(pre_pre_word,pre_word,now_word);
                        if (prob > score[i]) :
                            score[i] = prob
                            path[i] = (l,k,j)


    words = []
    nowp = len(line) - 1
    while nowp > 0 :
        l = path[nowp][0]
        k = path[nowp][1]
        j = path[nowp][2]
        if (nowp == len(line) - 1) and (j == 0) :
            words.append(line)
            break
        words.append(line[j : nowp + 1])
        if (j == 0) :
            break
        words.append(line[k : j])
        if (k == 0) :
            break
        words.append(line[l : k])
        if (l == 0) :
            break
        nowp = l - 1
        
        
    words.reverse()
    return words
            
import codecs
file = codecs.open('cip-data.train','r','utf-8')
training_dataset = file.read()
file.close()
hmm = HMM(training_data = training_dataset);
file = codecs.open("judge.data.1" ,"r", "utf-8")
write_file = codecs.open("result.txt", "w", "utf-8")
try :
    for line in file.readlines() :
        line = line.strip("\n")
        words = viterbi(hmm,line)
        result = words[0]
        for i in xrange(1,len(words)) :
            result = result + "\t" + words[i] 
        write_file.write(result + "\n")
        break;
finally :
    file.close()
    write_file.close()

