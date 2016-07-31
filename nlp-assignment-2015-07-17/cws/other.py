from collections import Counter
from math import log

class CRF(object) :
    def __init__(self,training_data = None) :
        if training_data is not None :
            self.fit(training_data)

    def fit(self, training_data) :
        self.alphacount = Counter()
        self.tagcount = Counter()
        self.alpha_tag = Counter()
        self.tag_ntag = Counter()
        self.alpha_tag_nalpha = Counter()
        self.alpha_tag_palpha = Counter()
        sentences = training_data.split('\n')
        #print sentences[0].encode('gbk')
        for sentence in sentences :
            words = sentence.split('\t')
            alpha_tag_list = []
            #Change words to (alpha,tag) list
            for word in words :
                tags = []
                if len(word) == 1 :
                    tags = ['S']
                else :
                    tags.append('B')
                    for i in xrange(len(word) - 2) :
                        tags.append('M')
                    tags.append('E')
                for i in xrange(len(word)) :
                    alpha_tag_list.append((word[i],tags[i]))
            #Count
            for i in xrange(len(alpha_tag_list)) :
                #print alpha_tag_list[i][0].encode('gbk'),alpha_tag_list[i][1],
                self.alphacount[alpha_tag_list[i][0]] += 1
                self.tagcount[alpha_tag_list[i][1]] += 1
                self.alpha_tag[alpha_tag_list[i]] += 1
                if (i != len(alpha_tag_list) - 1) :
                    self.tag_ntag[(alpha_tag_list[i][0],alpha_tag_list[i][1],alpha_tag_list[i + 1][1])] += 1
                    self.alpha_tag_nalpha[(alpha_tag_list[i][0],alpha_tag_list[i][1],alpha_tag_list[i + 1][0])] += 1 ##!!!!
                if (i != 0) :
                    self.alpha_tag_palpha[(alpha_tag_list[i][0],alpha_tag_list[i][1],alpha_tag_list[i - 1][1])] += 1 ##!!!!

    def emit(self,alpha,tag) :
        #P(alpha | tag)
        M = len(self.alphacount)
        prob = log(float(self.alpha_tag[(alpha,tag)] + 0.1)/(self.tagcount[tag] + 0.1 * M))
        return prob
    def trans(self,ptag,tag) :
        #P(tag | ptag)
        M = len(self.tagcount)
        prob = log(float(self.tag_ntag[(ptag,tag)] + 0.2) / (self.tagcount[tag] + 0.2 * M))
        return prob
    def palpha_tag(self,alpha,tag,palpha) :
        M = len(self.tagcount) ##!!!!
        prob = log(float(self.alpha_tag_palpha[(alpha,tag,palpha)] + 0.2) / (self.alpha_tag[(alpha,tag)] + 0.2 * M))
        return prob
    def nalpha_tag(self,alpha,tag,nalpha) :
        M = len(self.tagcount) ##!!!!
        prob = log(float(self.alpha_tag_nalpha[(alpha,tag,nalpha)] + 0.2) / (self.alpha_tag[(alpha,tag)] + 0.2 * M))
        return prob


def viterbi(crf,sentence) :
    postags = ['B','M','E','S']
    N = len(sentence)
    T = 4
    score = [[-float('inf') for j in range(T)] for i in range(N)]
    path = [[-1 for j in range(T)] for i in range(N)]
    
    for i,alpha in enumerate(sentence) :
        if i == 0 :
            for j, tag in enumerate(postags) :
                score[i][j] = crf.emit(alpha,tag)
        else :
            for j, tag in enumerate(postags) :
                best, best_t = -1e20, -1
                for k,ptag in enumerate(postags) :
                    nowp = score[i - 1][k] + crf.emit(alpha,tag) + crf.trans(ptag,tag) + 2.5 * crf.palpha_tag(alpha,tag,ptag) ;
                    if i != len(sentence) - 1:
                        nowp += 0.75 * crf.nalpha_tag(alpha,tag,sentence[i + 1])
                    if nowp > best :
                        best = nowp
                        best_t = k
                score[i][j] = best
                path[i][j] = best_t
    result = [best_t]
    for i in xrange(len(sentence) - 1, 0, - 1) :
        result.append(path[i][best_t])
        best_t = path[i][best_t]

    result = [postags[t] for t in reversed(result)]
    return result


########################
import codecs
file = codecs.open('cip-data.train','r','utf-8');
training_dataset = file.read();
training_dataset = training_dataset
file.close();
crf = CRF(training_data = training_dataset)
file = codecs.open("para.txt","w",'utf-8')
for item in crf.alpha_tag_nalpha :
    file.write(item[0] + item[1] + item[2] + "\n")
file.close()
file = codecs.open("judge.data.1",'r','utf-8')
write_file = codecs.open("result.txt","w","utf-8")
try :

    for line in file.readlines():
        line = line.strip("\n")
        tags = viterbi(crf,line)
        result = ""
        #print line.encode('gbk')
        #print tags
        #break
        for i in xrange(len(line)) :
            result = result + line[i]
            if (((tags[i] == 'E') | (tags[i] == 'S')) & (i != len(line) - 1)) :
                result = result + '\t'
        write_file.write(result + "\n")
            
finally :
    file.close()
    write_file.close()
