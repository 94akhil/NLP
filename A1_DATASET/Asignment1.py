
import math
import re

def readFile(path):
    content=[]
    with open(path, 'r') as file:
        for line in file:
            #removing trailing space and newline characters 
            content.append(line.strip())
    return content


def dataProcessing(datalist):
    # converting the contractions to complete word
    # removing special characters 
    # changing the casing to lower case
    # added start and end characters for each line

    special_chars = ['.', ',', '!', '?', ';', '`',':', '-', '(', ')', '[', ']', '{', '}', '\'', '"', '&', '*', '%', '$', '#', '@', '^', '~', '|', '\\', '/', '<', '>', '+', '=', '_']

    contractions = ["n't","'ve", "'re", "'ll", "'d","'m","&","'cause"]
    replacement = ["not", "have", "are", "will", "had","am","and","because"]

    processed_data = ""
    
    for data in datalist:

        modified = data.lower()

        for word, alternate in zip(contractions,replacement):
            modified = modified.replace(word,alternate )
        
        for word in special_chars:
            modified = modified.replace(word,"" )
        
        modified= re.sub(r'\s+', ' ', modified).strip()

        modified= '<s> ' + modified + ' </s> '

        processed_data+=modified
    
    return processed_data

def generateWordFrequency(data):
    wordFrequency={}
    split_data=data.split(" ")
    for word in split_data:
        if word not in wordFrequency:
            wordFrequency[word]=1
        else:
            wordFrequency[word]+=1
    return wordFrequency
    
def getTotalWords(wordFrequency):
    return sum(wordFrequency.values())

def generateUnigramModel(wordFrequency):

    unigram_model = {}
    totalWords = getTotalWords(wordFrequency)

    for word in wordFrequency:
        unigram_model[word] = (wordFrequency[word]/totalWords)

    return unigram_model

def smoothedAddKUnigram(wordFrequency, k, N):
    V= len(wordFrequency)
    smoothed_prob= {}

    for key, value in wordFrequency.items():
        smoothed_prob[key]= (value) + k/ (N + k * V)
    
    return smoothed_prob



def testUnigramModel(unigram_model, data):
    split_data=data.split(" ")
    unigram_test={}
    for word in split_data:
        unigram_test[word] = unigram_model.get(word, 0)
    
    return unigram_test

def generateBigramModel(data,wordFrequency):
    bigram_model={}
    bigramFrequency={}

    split_data= data.split(" ")
    n = len(split_data)

    for i in range(n-1):
        substr = split_data[i]+" "+split_data[i+1]
        if substr == "</s> <s>":
            continue

        if substr not in bigramFrequency:
            bigramFrequency[substr]=data.count(substr)
            bigram_model[substr] = bigramFrequency[substr]/wordFrequency[split_data[i]]
    
    # print(bigram_model)
    return bigram_model, bigramFrequency

def smoothedAddKBigram(wordFrequency, bigram_freq, k):
    V= len(wordFrequency)
    smoothed_prob= {}

    for key, value in bigram_freq.items():
        prev_word = key.split(" ")[0]
        smoothed_prob[key]= ((bigram_freq[key] ) + k)/ ((wordFrequency[prev_word]) + k * V)
    
    return smoothed_prob

# Handling unknown words in training data
def handlingUnknownWord(word_freq, data,threshold):

    split_data= data.split(" ")

    training_with_unk=""

    for word in split_data:
        if word_freq[word] <= threshold:
            training_with_unk+='<UNK> '
        else:
            training_with_unk+=word+" "
    
    word_freq_unk= generateWordFrequency(training_with_unk)

    return training_with_unk.strip(), word_freq_unk

# Handling unknown words in training data
def testHandlingUnknownWord(vocab, data):

    split_data= data.split(" ")

    training_with_unk=""

    for word in split_data:
        if word not in vocab:
            training_with_unk+='<UNK> '
        else:
            training_with_unk+=word+" "

    return training_with_unk.strip()


def compute_perplexity(prob_data):

    prob = list(prob_data.values())

    n = len(prob_data)
    log_sum = sum([-math.log2(p) for p in prob])
    l = log_sum / n
    return 2 ** l
    

def compute_linear_interpolation(data,bigram_prob, unigram_prob, l):
    split_data=data.split(" ")
    linear_interpolation={}
    n = len(split_data)

    for i in range(n-1):
        substr = split_data[i]+" "+split_data[i+1]
        if substr == "</s> <s>":
            continue
        
        linear_interpolation[substr] = l * (bigram_prob.get(substr,0)) + (1-l) * (unigram_prob[split_data[i+1]] if unigram_prob[split_data[i+1]] else unigram_prob['<UNK>'])
          
    return linear_interpolation

def testBigramModel(bigram_model, data, word_freq):
    split_data = data.split(" ")
    bigram_test = {}
    n = len(split_data)
    v = 1/len(word_freq)
    
    for i in range(n-1):
        word1, word2 = split_data[i], split_data[i+1]
        if word1 not in word_freq:
            word1 = "<UNK>"
        if word2 not in word_freq:
            word2 = "<UNK>"

        substr = f"{word1} {word2}"

        # Skip the end and start tokens combination
        if substr == "</s> <s>":
            continue

        # Fetch the probability from bigram model or assign a default
        bigram_test[substr] = bigram_model.get(substr, v)
          
    return bigram_test

#-------------------------------------------------------------------------------------------------------------------------------------------------

# --------------------------------------------- Training ----------------------------------------------------------------------------------------

# read the training data file
data =readFile('./A1_DATASET/train.txt')

# removes special characters and replace contractions with respective whole word
processed_data = dataProcessing(data)

# generates word frequency
word_freq = generateWordFrequency(processed_data)

# print(dict(sorted(word_freq.items(), key=lambda x:x[1])))

# ------- Unigram and Bigram models for Unsmoothed data -------

# computes probability for unigram
unigram_model = generateUnigramModel(word_freq)

# computes probability for bigram and bigram frequency
bigram_model, bigram_freq= generateBigramModel(processed_data,word_freq)

# ------- Handling Unknown words and Smoothing---------------
# k_values = [0.01,0.05,0.5,1,1.5,2]
# threshold _values= [1,2,3,4,5]

k = 0.01
threshold = 5
lambda_arr = [0.1,0.2,0.4,0.6,0.8,0.9]

# handles unknown words based on threshold
# return data text containing unknown character and new word frequency
unk_data, unk_unigram_freq = handlingUnknownWord(word_freq,processed_data,threshold)

# recalculate unigram probability
unk_unigram_prob = generateUnigramModel(unk_unigram_freq)

# recalculates the bigram frequency and probability
unk_bigram_model, unk_bigram_frequency = generateBigramModel(unk_data,unk_unigram_freq)

total = getTotalWords(unk_unigram_freq)

# performs add-k smoothing on data after (unknown word handling)
add_k_unigram_prob = smoothedAddKUnigram(unk_unigram_freq,k,total)
add_k_bigram_prob = smoothedAddKBigram(unk_unigram_freq, unk_bigram_frequency,k)

# --------------------------------------------- Testing ----------------------------------------------------------------------------------------
test_data =readFile('./A1_DATASET/val.txt')

# test data pre-processing
test_processed_data = dataProcessing(test_data)
#test data unknown word handling
test_unk_data =testHandlingUnknownWord(unk_unigram_freq, test_processed_data)

#unsmoothed ngram without handling unknown word 
uni_test_prob1= testUnigramModel(unigram_model, test_processed_data)
bi_test_prob1 = testBigramModel(bigram_model, test_processed_data,word_freq)

#unsmoothed ngram with unknown word handling
uni_test_prob2= testUnigramModel(unk_unigram_prob, test_unk_data)
bi_test_prob2 = testBigramModel(unk_bigram_model, test_unk_data,unk_unigram_freq)


#perplexity on smoothed test data for bigram
bi_test_prob = testBigramModel(add_k_bigram_prob, test_unk_data,unk_unigram_freq)
perplexity_bi = compute_perplexity(bi_test_prob)

#perplexity on smoothed test data for uniigram
test_unigram_prob= testUnigramModel(unk_unigram_prob, test_unk_data)
perplexity_uni = compute_perplexity(test_unigram_prob)

# This code runs a forloop on different value of lambda to generate perplexity for linear interpolation

# for l in lambda_arr:
#     interpolation_prob = compute_linear_interpolation(test_unk_data,unk_bigram_model,unk_unigram_prob,l)
#     perplexity = compute_perplexity(interpolation_prob)
#     print('Threshold: '+str(threshold)+" lambda: "+str(l))
#     print(perplexity)

train_lp_prob = compute_linear_interpolation(unk_data,unk_bigram_model,unk_unigram_prob,0.6)

test_lp_unigram_prob = compute_linear_interpolation(test_unk_data,bi_test_prob,test_unigram_prob,0.6)


#-------------------------------------------------------- Results ----------------------------------------------------------------------------

print('---- Unsmoothed :  Training data after handling unknown words ----')
print("Perplexity of unsmoothed training data for unigram "+ str(compute_perplexity(unk_unigram_prob)))
print("Perplexity of unsmoothed training data for bigram "+ str(compute_perplexity(unk_bigram_model)))

print('---- Smoothed :  Training data  ----')
print("Perplexity of smoothed training data for unigram "+ str(compute_perplexity(add_k_unigram_prob)))
print("Perplexity of add-k smoothed training data for bigram "+ str(compute_perplexity(add_k_bigram_prob)))
print("Perplexity of linear interpolation smoothed training data set "+ str(compute_perplexity(train_lp_prob)))

print('---- Unsmoothed :  Testing data after handling unknown words ----')
print("Perplexity of unsmoothed test data for bigram "+ str(compute_perplexity(uni_test_prob2)))
print("Perplexity of unsmoothed test data for bigram "+ str(compute_perplexity(bi_test_prob2)))

print('---- Smoothed :  Test data ----')
print("Perplexity of smoothed test data for bigram "+ str(perplexity_bi))
print("Perplexity of smoothed test data for unigram "+ str(perplexity_uni))
print("Perplexity of smoothed test data set "+ str(compute_perplexity(test_lp_unigram_prob)))