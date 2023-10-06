# import necessary packages
import nltk
import math
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class CorpusReader_TFIDF:
    def __init__(self, corpus, tf="raw", idf="base", stopWord="none", toStem=False, stemFirst=False, ignoreCase=True):

        """
        Constructor class for CorpusReader_TFIDF where the class will take a 
        corpus object in NLTK and construct the td-idf vector for each document

            Params:
                corpus:
                    a coorpus object in NLTK
                tf:
                    the method used to calculate term frequency. The following values are supported
                        “raw” (default) = raw term frequency
                        “log” : log normalized (1 + log (frequency) if frequency > 0; 0 otherwise)
                idf:
                    the method used to calculate the inverse document frequency
                        “base” (default) : basic inverse document frequency
                        “smooth”: inverse frequency smoothed
                stopWord:
                    what stopWords to remove
                        “none” : no stopwords need to be removed
                        “standard”: use the standard English stopWord available in NLTK
                        Others: this should treat as a filename where stopwords are to be read. You should assume any word inside the stopwords file is a stopword.
                toStem:
                    if true, use the Snowball stemmer to stem the words beforehand
                stemFirst:
                    if stopwords are used and stemming is set to yes (otherwise this flag is ignored), then true
                    means you stem before you remove stopwords, and false means you remove stopwords before you stem
                ignoreCase:
                    if true, ignore the case of the word (i.e. “Apple”, “apple”, “APPLE” are the same word). 
                    In such case, represent the word as the all lower-case version (this include the words in the stopWord file). 
                    Also, you will change all words into lowercase before you do any subsequent processing 
                    (e.g. remove stopwords and stemming)

            Returns:

        """

        # initialize variables
        self.corpus = corpus
        self.tf = tf
        self.idf = idf
        self.stopWord = stopWord
        self.toStem = toStem
        self.stemFirst = stemFirst
        self.ignoreCase = ignoreCase

        # define stop words
        if self.stopWord == "none":  # if no stop words
            self.stopWords = ""

        elif self.stopWord == "standard":  # if standard stop words
            self.stopWords = stopwords.words("english")

        else:  # if custom stop words
            self.stopWords = []
            with open(stopWord, "r") as stopWords_list:
                for line in stopWords_list:
                    self.stopWords.append(line.lower().strip("\n"))

        # define stemmer
        if self.toStem:
            self.stemmer = SnowballStemmer("english")

    def __preprocess(self, document):
        """
        Helper function that takes in a document and applies ignoreCase and Stemming
        """

        # words = nltk.word_tokenize(document)
        words = document

        # if ignoreCase == True, list comprehend all words to be lower
        if self.ignoreCase:
            words = [word.lower() for word in words]

        # if stopwords are used and stemming is set to True
        if (self.stopWord != "none") and (self.toStem == True):
            
            # if stemFirst == True ...
            if self.stemFirst == True:

                # stem words
                words = [self.stemmer.stem(word) for word in words]

                # remove stopwords
                words = [word for word in words if word not in self.stopWords]

            # if stemFirst == False ...
            if self.stemFirst == False:

                # remove stopwords
                words = [word for word in words if word not in self.stopWords]

                # stem words
                words = [self.stemmer.stem(word) for word in words]

        return words


    def __tf_calc(self, fileid):

        term_frequencies = {}

        # extract document for a single fileid
        if fileid:
            document = self.corpus.words([fileid])

            # preprocess document
            words = self.__preprocess(document=document)

            # calculate tf raw, if tf == log, then calculate log on top of raw
            for word in words:
                term_frequencies[word] = term_frequencies.get(word, 0) + 1

            # adjust if log normalization
            if self.tf == "log":
                for word, frequency in term_frequencies.items():
                    term_frequencies[word] = 1 + math.log(frequency, 2) if frequency > 0 else 0
        
        # extract doocument for all documents
        else:
            for id in self.corpus.fileids():
                document = self.corpus.words([id])

                # preprocess document
                words = self.__preprocess(document=document)

                # calculate tf raw, if tf == log, then calculate log on top of raw
                for word in words:
                    term_frequencies[word] = term_frequencies.get(word, 0) + 1

                # adjust if log normalization
                if self.tf == "log":
                    for word, frequency in term_frequencies.items():
                        term_frequencies[word] = 1 + math.log(frequency, 2) if frequency > 0 else 0

        return term_frequencies
    

    def __idf_calc(self, fileid):

        idf_scores = {}
        number_of_documents = len(self.corpus.fileids())

        # set number of documents to 1 or all
        if fileid:
            document = self.corpus.words([fileid])

            # preprocess document and get a unique set of words
            words = set(self.__preprocess(document=document))

            # calculate word frequency
            for word in words:
                idf_scores[word] = idf_scores.get(word, 0) + 1

            # calculate idf
            for word, frequency in idf_scores.items():
                if self.idf == "base":
                    idf_scores[word] = math.log(number_of_documents / (1 + frequency), 2)
                if self.idf == "smooth":  # https://en.wikipedia.org/wiki/Tf%E2%80%93idf
                    idf_scores[word] = math.log(number_of_documents / (1 + frequency), 2) + 1

        else:
            for id in self.corpus.fileids():
                document = self.corpus.words([id])

                # preprocess document and get a unique set of words
                words = set(self.__preprocess(document=document))

                # calculate word frequency
                for word in words:
                    idf_scores[word] = idf_scores.get(word, 0) + 1

                # calculate idf
                for word, frequency in idf_scores.items():
                    if self.idf == "base":
                        idf_scores[word] = math.log(number_of_documents / (1 + frequency), 2)
                    if self.idf == "smooth":  # https://en.wikipedia.org/wiki/Tf%E2%80%93idf
                        idf_scores[word] = math.log(number_of_documents / (1 + frequency), 2) + 1

        return idf_scores
    

    def __tfidf_calc(self, term_frequencies, idf_scores, returnZero):
        tfidf_scores = {}

        if returnZero:
            for word, frequency in term_frequencies.items():
                tfidf_scores[word] = frequency * idf_scores[word]

        else:
            for word, frequency in term_frequencies.items():
                tfidf_score = frequency * idf_scores[word]
                if tfidf_score != 0:
                    tfidf_scores[word] = tfidf_score

        return tfidf_scores
    

    def tfidf(self, fileid="", returnZero=False):
        """
        return the TF-IDF for the specific document in the corpus (specified by fileid). 
        The vector is represented by a dictionary/hash in python. 
        The keys are the terms, and the values are the tf-idf value of the dimension. 
        If returnZero is true, then the dictionary will contain terms that have 0 value for that vector, 
        otherwise the vector will omit those terms
        """
        
        # calculate tf
        term_frequencies = self.__tf_calc(fileid=fileid)

        # calculate idf
        idf_scores = self.__idf_calc(fileid=fileid)

        # calculate tfidf
        tfidf_scores = self.__tfidf_calc(term_frequencies=term_frequencies, idf_scores=idf_scores, returnZero=returnZero)

        return tfidf_scores
    

    def tfidfAll(self, returnZero=False):
        fileid = ""

        # calculate tf
        term_frequencies = self.__tf_calc(fileid=fileid)

        # calculate idf
        idf_scores = self.__idf_calc(fileid=fileid)

        # calculate tfidf
        tfidf_scores = self.__tfidf_calc(term_frequencies=term_frequencies, idf_scores=idf_scores, returnZero=returnZero)

        return tfidf_scores
    

    def tfidfNew(self, document=[], returnZero=False):
        term_frequencies = {}

        # preprocess document
        words = self.__preprocess(document=document)

        # calculate tf raw, if tf == log, then calculate log on top of raw
        for word in words:
            term_frequencies[word] = term_frequencies.get(word, 0) + 1

        # adjust if log normalization
        if self.tf == "log":
            for word, frequency in term_frequencies.items():
                if frequency > 0:
                    term_frequencies[word] = 1 + math.log(frequency, 2)
                else:
                    term_frequencies[word] = 0

        # set number of documents but do not include the new document as part of the corpus
        number_of_documents = len(self.corpus.fileids()) + 1
        idf_scores = {}

        # preprocess document and get a unique set of words
        words = set(self.__preprocess(document=document))

        # calculate word frequency
        for word in words:
            idf_scores[word] = idf_scores.get(word, 0) + 1

        # calculate idf
        for word, frequency in idf_scores.items():
            if self.idf == "base":
                idf_scores[word] = math.log(number_of_documents / (1 + frequency), 2)
            if self.idf == "smooth":  # https://en.wikipedia.org/wiki/Tf%E2%80%93idf
                idf_scores[word] = math.log(number_of_documents / (1 + frequency), 2) + 1

        # calculate tfidf
        tfidf_scores = self.__tfidf_calc(term_frequencies=term_frequencies, idf_scores=idf_scores, returnZero=returnZero)

        return tfidf_scores
    

    def cosine_sim(self, fileid1, fileid2):
        # initialize tfidf vectorizer
        tfidfvectorizer = TfidfVectorizer()

        # preprocess documents
        document1 = self.__preprocess(self.corpus.words([fileid1]))
        document2 = self.__preprocess(self.corpus.words([fileid2]))

        # join tokens into a single document
        document1 = " ".join(document1)
        document2 = " ".join(document2)

        # fit transform into matrix
        tfidf_matrix = tfidfvectorizer.fit_transform([document1, document2])

        # calculate cosine similarity
        cos_sim = cosine_similarity(tfidf_matrix)

        return cos_sim
    

    def cosine_sim_new(self, document, fileid):
        # initialize tfidf vectorizer
        tfidfvectorizer = TfidfVectorizer()

        # preprocess documents
        document1 = self.__preprocess(document)
        document2 = self.__preprocess(self.corpus.words([fileid]))

        # join tokens into a single document
        document1 = " ".join(document1)
        document2 = " ".join(document2)

        # fit transform into matrix
        tfidf_matrix = tfidfvectorizer.fit_transform([document1, document2])

        # calculate cosine similarity
        cos_sim = cosine_similarity(tfidf_matrix)

        return cos_sim
    

    def query(self, document):

        query_result = []
        
        document1 = self.__preprocess(document)
        document1 = " ".join(document1)

        for id in self.corpus.fileids():
            # initialize tfidf vectorizer
            tfidfvectorizer = TfidfVectorizer()

            # preprocess documents
            document2 = self.__preprocess(self.corpus.words([id]))
            document2 = " ".join(document2)

            # fit transform into matrix
            tfidf_matrix = tfidfvectorizer.fit_transform([document1, document2])

            # calculate cosine similarity
            cos_sim = cosine_similarity(tfidf_matrix)[0][1]

            query_result.append((id, cos_sim))

        # sort list in descending order
        query_result.sort(key=lambda x: x[1], reverse=True)
        return query_result


# ----- Test Area -----

if __name__ == "__main__":

    from nltk.corpus import inaugural

    myCorpus = CorpusReader_TFIDF(inaugural, tf="raw", idf="smooth", stopWord="standard", toStem=True)
    # print(myCorpus.tfidf('1789-Washington.txt', returnZero=False)["fellow"])
    # print(myCorpus.tfidfAll(returnZero=True))
    # print(myCorpus.tfidfNew(document=['citizens', 'economic', 'growth', 'economic'], returnZero=True))
    # print(myCorpus.cosine_sim(fileid1='1789-Washington.txt', fileid2='2021-Biden.txt'))
    # print(myCorpus.cosine_sim_new(document=['citizens', 'economic', 'growth', 'economic'], fileid='2021-Biden.txt'))
    # print(myCorpus.query(document=['citizens', 'economic', 'growth', 'economic']))
