import re
from sklearn.feature_extraction.text import TfidfVectorizer
from pyarabic.normalize import normalize_searchtext,strip_tatweel,strip_tashkeel
import pandas as pd
from camel_tools.disambig.mle import MLEDisambiguator
import nltk
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from autocorrect import Speller
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pickle






class ElementalSentimentAnalysis:
    def __init__(self,train=False):
        self.stopWords=self.readStopWords(fileName='normalizedStopWords.txt')
        self.mle = MLEDisambiguator.pretrained()
        # Init the Wordnet Lemmatizer for english
        self.eLemmatizer = WordNetLemmatizer()
        self.eSpell = Speller(lang='en')
        if train:
            self.vectorizer = CountVectorizer()
        else:
            self.vectorizer=pickle.load(open('CounVectorizer.sav', 'rb'))
        if train:
            self.model = None
        else:
            self.model=pickle.load(open('SvmModel.sav', 'rb'))
        # print(self.preprocessData(["أنا احب ان العب مع أصدقائي i like to play with for freinds"]))
        # # print(self.getWordTypeArabic("أنا احب أن ألعب الكرة و أنا سعيد"))


    def deEmojify(self,text):
        regrex_pattern = re.compile(pattern = "["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                            "]+", flags = re.UNICODE)
        return regrex_pattern.sub(r'',text)

    def preprocessData(self,dataSet):
        processedDataSet = []
        i=0
        for sentence in dataSet:
            i=i+1
            print(i)
            # Remove emojis
            processed_feature = self.deEmojify(sentence)
            arabic = re.sub(r'\s*[A-Za-z0-9]+\b', '', processed_feature)
            english = re.sub(r'\s*[^A-Za-z0-9\s]+\b', "", processed_feature)
            processed_feature=self.prepareArabicSentence(arabic).strip()+" "+self.prepareEnglishSentence(english).strip()
            # Remove all the special characters
            processed_feature = re.sub(r'\W', ' ', processed_feature)
            # remove all single characters
            processed_feature = re.sub(r'\s+[a-zA-Z]\s+', ' ', processed_feature)
            # Remove single characters from the start
            processed_feature = re.sub(r'\^[a-zA-Z]\s+', ' ', processed_feature)
            # Substituting multiple spaces with single space
            processed_feature = re.sub(r'\s+', ' ', processed_feature, flags=re.I)
            # #remove double chars
            # processed_feature= self.checkDoubleChars(sentence)
            #normalize english words
            processed_feature= self.prepareEnglishSentence(processed_feature)
            #normalize arabic words
            processed_feature= self.prepareArabicSentence(processed_feature)
            # Remove stop words
            processed_feature = self.removeStopWords(processed_feature)
            processed_feature = processed_feature.strip()
            processedDataSet.append(processed_feature.split())
        return processedDataSet

    def checkDoubleChars(self,sentence):
        # Iterate over index
        previousChar=None
        firstTime=True
        for char in sentence:
            if firstTime:
                firstTime=False
                previousChar=char
                continue
            else:
                if previousChar==char:
                    sentence.remove(char)
                    continue
                previousChar=char
        return sentence

    def prepareEnglishSentence(self,sentence):
        # Removing prefixed 'b'
        processed_feature = re.sub(r'^b\s+', '', sentence)
        # Converting to Lowercase
        processed_feature = processed_feature.lower()
        #correcting spelling
        processed_feature = ' '.join([self.eSpell(w) for w in processed_feature.split()])
        #lemmatize
        processed_feature = ' '.join([self.eLemmatizer.lemmatize(w) for w in processed_feature.split()])
        return processed_feature

    #lemmtization
    #check type of word
    #spelling checker
    #tokenize

    def prepareArabicSentence(self,sentence):
        #lemmatization
        lemmas=sentence.split()
        disambig = self.mle.disambiguate(lemmas)
        sentence = " ".join(d.analyses[0].analysis['lex'] for d in disambig if len(d.analyses)>0)
        # strip tashkeel strip tatweel
        sentence=strip_tashkeel(sentence)
        sentence=strip_tatweel(sentence)
        # normalize Teh Marbuta
        sentence = sentence.replace('ة', 'ه')
        # normalize ya2
        sentence = sentence.replace('أ','ا')
        # normalize Alef
        sentence = sentence.replace('أ','ا')
        sentence = sentence.replace('إ','ا')
        sentence = sentence.replace('آ', 'ا')
        sentence = sentence.replace('ى','ي')
        return sentence

    def vectorizeListOfWords(self, list,train=False):
        if (train):
            processed_features = self.vectorizer.fit(list)
        processed_features = self.vectorizer.transform(list)
        return processed_features

    def readStopWords(self,fileName):
        file1 = open(fileName, 'r', encoding='utf8')
        Lines = file1.readlines()
        myWordSet = set()
        for line in Lines:
            myWordSet.add(line.strip())
        return myWordSet

    def removeStopWords(self,sentence):
        newSentence = ""
        for i in sentence.split():
            if i not in self.stopWords:
                newSentence = newSentence + " " + i
        return newSentence


    def readDataSet(self,fileName,maxEnteries):
        texts = []
        labels = []
        line_count = 0
        df1 = pd.read_csv(fileName)
        df1.columns = ['index', 'label', 'text']
        for index, row in df1.iterrows():
            line_count = line_count + 1
            texts.append(row['text'])
            labels.append(row['label'])
            if line_count >= maxEnteries:
                break
        print(f' INFO: Processed {line_count} lines.')
        return texts, labels

    def getWordTypeArabic(self,sentence):
        lemmas=sentence.split()
        disambig = self.mle.disambiguate(lemmas)
        new_sentence = [d.analyses[0].analysis['pos'] for d in disambig if len(d.analyses)>0]
        i=0
        my_list={}
        while i<len(lemmas):
            if len(disambig[i].analyses)>0:
                my_list[lemmas[i]]=disambig[i].analyses[0].analysis['pos']
            i=i+1
        return my_list

    def trainModel(self):
        features,labels=self.readDataSet('abdo_shuffled.csv',125000)
        self.preprocessData(features)
        X_train,X_test,y_train,y_test=train_test_split(features,labels,test_size=0.1)
        X_train_vectorized=self.vectorizeListOfWords(X_train,train=True)
        pickle.dump(self.vectorizer, open('CounVectorizer.sav', 'wb'))
        X_test_vectorized=self.vectorizeListOfWords(X_test,train=False)
        svclassifier = SVC(kernel='linear')
        svclassifier.fit(X_train_vectorized, y_train)
        y_predict=svclassifier.predict(X_test_vectorized)
        pickle.dump(svclassifier, open('SvmModel.sav', 'wb'))
        from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
        print("Accuracy for SVM: %s"
              % (accuracy_score(y_test, y_predict)))
        print(confusion_matrix(y_test, y_predict))
        print(classification_report(y_test, y_predict))







if __name__=="__main__":
    sa = ElementalSentimentAnalysis(train=True)
    sa.trainModel()



