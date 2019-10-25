--cross x    

x is the index of domain, using cross to check dev and test results of specific domain and select differrent hyper-parameter of model

put your vocabulary as ../wordEmb/vocab_t  and your word vectors as ../wordEmb/vector_t

vocab_t is a str(dict)，(e.g. {'word': ind})

vector_t is a 2-d np.array saved by np.save(), vector_t[i] is the vector of the i-th word

put your data in ../data/
