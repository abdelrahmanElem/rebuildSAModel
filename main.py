import re
from pyarabic.normalize import strip_tatweel,strip_tashkeel
from camel_tools.disambig.mle import MLEDisambiguator
import nltk
from nltk.stem import WordNetLemmatizer
from autocorrect import Speller
from ar_corrector.corrector import Corrector
from googletrans import Translator
import enchant




class preprocessData():
    def __init__(self):
        self.translator = Translator()
        #english words
        self.words = set(nltk.corpus.words.words())
        nltk.download('wordnet')
        self.stopWords=self.readStopWords(fileName='normalizedStopWords.txt')
        # Init the Wordnet Lemmatizer for english
        self.eLemmatizer = WordNetLemmatizer()
        self.eSpell = Speller(lang='en')
        self.mle = MLEDisambiguator.pretrained()
        self.englishDictionary=enchant.Dict("en_US")
        self.aSpell = Corrector()


    def deEmojify(self,text):
        regrex_pattern = re.compile(pattern = "["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                            "]+", flags = re.UNICODE)
        return regrex_pattern.sub(r'',text)

    def readStopWords(self,fileName):
        file1 = open(fileName, 'r', encoding='utf8')
        Lines = file1.readlines()
        myWordSet = set()
        for line in Lines:
            myWordSet.add(line.strip())
        return myWordSet

    def preprocessData(self,dataSet,Tokinize=False,hasFranco=True):
        processedDataSet = []
        i=0
        for sentence in dataSet:
            i=i+1
            print(i)
            # Remove emojis
            # Remove all the special characters
            processed_feature = re.sub(r'\W', ' ', sentence)
            processed_feature = self.deEmojify(processed_feature)
            arabic = re.sub(r'\s*[A-Za-z0-9]+\b', '', processed_feature)
            english = re.sub(r'\s*[^A-Za-z0-9\s]+\b', "", processed_feature)
            if hasFranco:
                english = self.filterEnglish(english)
            processed_feature=self.prepareArabicSentence(arabic).strip()+" "+self.prepareEnglishSentence(english).strip()
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
            if Tokinize:
                processedDataSet.append(processed_feature.split())
            else:
                processedDataSet.append(processed_feature)
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

    def prepareArabicSentence(self,sentence):
        #lemmatization
        lemmas=sentence.split()
        disambig = self.mle.disambiguate(lemmas)
        sentence = " ".join(d.analyses[0].analysis['lex'] for d in disambig if len(d.analyses)>0)
        #spelling correction
        sentence=self.aSpell.contextual_correct(sentence)
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

    def removeStopWords(self,sentence):
        newSentence = ""
        for i in sentence.split():
            if i not in self.stopWords:
                newSentence = newSentence + " " + i
        return newSentence

    #translate franco (it takes a lot of time for large datasets)
    def translateFrancoSentence(self,word):
        translation=self.translator.translate(str(word),src='ar',dest='en')
        return translation.text

    def filterEnglish(self,sent):
        if sent==None or len(sent)<=1:
            return ''
        sent=sent.lower()
        english = ""
        franco=""
        notTranslated=0
        for w in nltk.wordpunct_tokenize(sent):
            if (self.englishDictionary.check(w.lower())):
                if notTranslated ==1:
                    franco = self.translateFrancoSentence(franco)
                    english = english+" "+franco
                    franco=""
                    notTranslated=0
                english= english+" "+w
            else:
                franco = franco+" "+w
                notTranslated=1
        if notTranslated==1:
            franco = self.translateFrancoSentence(franco)
            english = english + " " + franco
        return english

    def removeNonEnglishWords(self,sentence):
        newSentence=""
        for w in nltk.wordpunct_tokenize(sentence):
            if (self.englishDictionary.check(w.lower())):
                newSentence= newSentence+" "+w
            else:
                pass
        return newSentence.strip()

    def preprocessDataToEnglish(self,dataSet, Tokinize=False):
        processedDataSet = []
        i=0
        for sentence in dataSet:
            i=i+1
            print(i)
            # Remove emojis
            processed_feature = self.deEmojify(sentence)
            arabic = re.sub(r'\s*[A-Za-z0-9]+\b', '', processed_feature)
            english = re.sub(r'\s*[^A-Za-z0-9\s]+\b', "", processed_feature)
            try:
                translatedArabic = self.translateFrancoSentence(arabic)
            except:
                translatedArabic=""
            english=self.filterEnglish(english)
            processed_feature=translatedArabic.strip()+" "+english.strip()
            processed_feature=self.prepareEnglishSentence(processed_feature)
            # Remove all the special characters
            processed_feature = re.sub(r'\W', ' ', processed_feature)
            # remove all single characters
            processed_feature = re.sub(r'\s+[a-zA-Z]\s+', ' ', processed_feature)
            # Remove single characters from the start
            processed_feature = re.sub(r'\^[a-zA-Z]\s+', ' ', processed_feature)
            # Substituting multiple spaces with single space
            processed_feature = re.sub(r'\s+', ' ', processed_feature, flags=re.I)
            # Remove stop words
            processed_feature = self.removeStopWords(processed_feature)
            # Remove non english Words
            processed_feature=self.removeNonEnglishWords(processed_feature)
            if Tokinize:
                processedDataSet.append(processed_feature.split())
            else:
                processedDataSet.append(processed_feature)
        return processedDataSet

if __name__=="__main__":
    dataPreProcessor= preprocessData()
    # print(dataPreProcessor.preprocessDataToEnglish(['I love going to mexico','ana ba7eb aroo7 henak','انا مش مصدق نفسي I am on Television ','انا بحب ألعب كوره']))
    # Program to extract a particular row value
    import pandas as pd
    dataCustom = pd.read_excel("custom data2.xlsx")
    dataset=[]
    customTest=[]
    # data=[row[0] for row in dataset]
    # labels=[row[1] for row in dataset]

    counter=0
    for index, row in dataCustom.iterrows():
        counter=counter+1
        if counter<2500:
            dataset.append([(row['Column1']), int(row['Column2'])])
        elif counter>=2500 and counter<=2750:
            customTest.append([(row['Column1']), int(row['Column2'])])
        elif counter>2750:
            break
    dataCustom = pd.read_csv("abdo_shuffled.csv")
    counter=0
    # for index, row in dataCustom.iterrows():
    #     counter=counter+1
    #     if counter<15000:
    #         dataset.append([(row['text']), int(row['label'])])
    #     else:
    #         break
    data=[row[0] for row in dataset]
    labels=[row[1] for row in dataset]
    X_test=[row[0] for row in customTest]
    y_test=[row[1] for row in customTest]

    #train data
    from sklearn.model_selection import train_test_split

    X_train,X_val,y_train,y_val= train_test_split(data,labels,test_size=0.05)
    # X_train=dataPreProcessor.preprocessDataToEnglish(X_train)
    # X_val=dataPreProcessor.preprocessDataToEnglish(X_val)
    # X_test=dataPreProcessor.preprocessDataToEnglish(X_test)
    X_train=dataPreProcessor.preprocessData(X_train,Tokinize=False,hasFranco=False)
    X_val=dataPreProcessor.preprocessData(X_val,Tokinize=False,hasFranco=False)
    X_test=dataPreProcessor.preprocessData(X_test,Tokinize=False,hasFranco=False)

    from sklearn.feature_extraction.text import CountVectorizer
    vec= CountVectorizer()
    vec.fit(X_train)
    X_train_vec=vec.transform(X_train)
    X_val_vec=vec.transform(X_val)
    X_test_vec= vec.transform(X_test)
    from sklearn.svm import SVC
    svclassifier = SVC()
    y_val_predict= svclassifier.predict(X_val_vec)
    y_test_predict = svclassifier.predict(X_test_vec)
    from sklearn.metrics import accuracy_score
    print("Accuracy for RFC: %s"
          % (accuracy_score(y_test, y_test_predict)))
    print("Accuracy for RFC: %s"
          % (accuracy_score(y_val, y_val_predict)))

    # for index,row in dataCustom.iterrows():
    #     dataset.append([(row['Column1']), int(row['Column2'])])

    # df = pd.read_csv('sentiment_tweets3.csv')
    # i=0
    # for index, row in df.iterrows():
    #     customTest.append([row['message to examine'],int(row['label'])])
    #     i=i+1
    #     if (i==5):
    #         break

    # data=[row[0] for row in dataset]
    # labels=[row[1] for row in dataset]
    # i=0
    # while i < len(labels):
    #     dataPreProcessor.translateFrancoSentence(data[i])
    # print(labels)