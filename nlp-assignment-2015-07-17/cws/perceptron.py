from collections import Counter
from math import log
class PerceptronClassifier(object) :
    def __init__(self, max_iter = 10, training_data = None) :
        self.max_iter = max_iter
        if training_data is not None :
            self.fit(training_data)

    def fit(self, training_data) :
        self.feature_alphabet = {'None': 0}
        self.label_alphabet = {}

        instances = []
        sentences = training_data.split("\n")
        sentences_tags = []
        for sentence in sentences :
            words = sentence.split("\t")
            words_tags = []
            for word in words :
                tags = []
                if len(word) == 1 :
                    tags = ["S"]
                else :
                    tags.append("B")
                    for i in xrange(len(word) - 2) :
                        tags.append("M")
                    tags.append("E")
                for i in xrange(len(word)) :
                    words_tags.append((word[i],tags[i]))
            L = len(words_tags)
            prev = "<s>"
            sentences_tags.append(words_tags)
            for i in range(L) :
                X = self.extract_feature(words_tags, i, prev, True)
                if words_tags[i][1] not in self.label_alphabet :
                    num = len(self.label_alphabet)
                    self.label_alphabet[words_tags[i][1]] = num
                Y = self.label_alphabet[words_tags[i][1]]
                instances.append((X,Y))
                prev = words_tags[i][1]
        
        self.labels = [-1 for k in self.label_alphabet]
        for k in self.label_alphabet :
            self.labels[self.label_alphabet[k]] = k
        
        self.D, self.T = len(self.feature_alphabet), len(self.label_alphabet)
        print('number of features : %d' % self.D)
        print('number of labels : %d' % self.T)

        self.W = [[0 for j in range(self.D)] for i in range(self.T)]
        
        self.best_W = None
        best_acc = None

        for it in range(self.max_iter) :
            n_errors = 0
            print("training iteration #%d" % it)
            for X, Y in instances :
                Z = self._predict(X);
                if Z != Y :
                    n_errors += 1
                    for x in X :
                        self.W[Z][x] -= 1
                        self.W[Y][x] += 1
            print ("training_error %d" % n_errors)



    def extract_feature(self, words, i, prev_tag = None, add=True) :
        L = len(words)
        context = ["<s>" if i - 2 < 0 else words[i - 2][0],
                   "<s>" if i - 1 < 0 else words[i - 1][0],
                   words[i][0],
                   "<e>" if i + 1 >= L else words[i + 1][0],
                   "<e>" if i + 2 >= L else words[i + 1][0]]
        raw_features = ['U1=%s' % context[0],
                    'U2=%s' % context[1],
                    'U3=%s' % context[2],
                    'U4=%s' % context[3],
                    'U5=%s' % context[4],
                    'U1,2=%s/%s' % (context[0],context[1]),
                    'U2,3=%s/%s' % (context[1],context[2]),
                    'U3,4=%s/%s' % (context[2],context[3]),
                    'U4,5=%s/%s' % (context[3],context[4]),
                    ]
        if prev_tag is not None:
            raw_features.append('B=%s' % prev_tag)
        
        mapped_features = []
        for f in raw_features :
            if add and (f not in self.feature_alphabet):
                num = len(self.feature_alphabet)
                self.feature_alphabet[f] = num
            if (f in self.feature_alphabet) :
                mapped_features.append(self.feature_alphabet[f])
            
        return mapped_features
    def _score(self, features, t):
        s = 0
        for m_feature in features :
            s += self.W[t][m_feature]
        return s

    
    def _predict(self, features):
        pred_scores = [self._score(features, y) for y in range(self.T)]
        best_score, best_y = None, None
        for i in xrange(len(pred_scores)) :
            if ((pred_scores[i] > best_score) or (best_score == None)) :
                best_score = pred_scores[i]
                best_y = i
        return best_y
            
    def predict(self, words, i, prev_tag = None) :
        X = self.extract_feature(words, i, prev_tag, False)
        y = self._predict(X)
        return y

def greedy_search(words, classifier) :
    prev = "<s>"
    ret = []
    words_tags = []
    for i in range(len(words)) :
        words_tags.append((words[i],"blabla"));
    for i in range(len(words_tags)) :
        label = classifier.labels[classifier.predict(words, i, prev)]
        ret.append(label)
        prev = ret[-1]
    return ret

#########################
import codecs
file = codecs.open("cip-data.train","r","utf-8")
training_dataset = file.read()
file.close();
perceptron = PerceptronClassifier(10, training_data = training_dataset)

file = codecs.open("judge.data.2","r","utf-8")
write_file = codecs.open("result2.txt","w","utf-8")
try :
    for line in file.readlines() :
        line = line.strip("\n")
        tags = greedy_search(line, perceptron)
        result = ""
        for i in xrange(len(line)) :
            result = result + line[i]
            if ((tags[i] == 'E') or (tags[i] == 'S')) & (i != len(line) - 1) :
                result = result + "\t"
        write_file.write(result + "\n")
finally :
    file.close()
    write_file.close()
