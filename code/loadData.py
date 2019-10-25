import numpy as np
import os
import util
import random

class TextIterator():
    def __init__(self, config):
        self.config = config
        self.dataPrefix = "../data/"
        self.validInd = [0 for i in range(config.task)]
        self.trainInd = [0 for i in range(config.task)]
        self.testInd = [0 for i in range(config.task)]
        self.encodingSet = set(["dvd.task.train", "MR.task.test", "MR.task.train"])
        self.name = []
        self.epoch = 0
        self.train = [[] for i in range(config.task)]
        self.valid = [[] for i in range(config.task)]
        self.test = [[] for i in range(config.task)]
        self.word2id = self.getVocab()
        self.readData()
        self.threshold = [len(self.train[i]) // self.config.batch_size for i in range(self.config.task)]
        for i in range(config.task):
            print(len(self.train[i]), len(self.valid[i]), len(self.test[i]))
        if self.config.wordemb_suffix == "mask_t":
            print("mask")
        else:
            print("unmask")

    def getVocab(self):
        dic = eval(open("../wordEmb/vocab_"+self.config.wordemb_suffix, encoding='utf-8').readline())
        return dic

    def readData(self):
        dirr = list(os.walk(self.dataPrefix))[0][2]
        ind = 0
        for i in range(3):
            if i == 0:
                ls = self.train
                string = "train"
            elif i == 1:
                ls = self.valid
                string = "valid"
            else:
                ls = self.test
                string = "test"
            for fileName in dirr:
                fileNameLS = fileName.split('.')
                if fileNameLS[-1] != string:
                    continue
                file = open(self.dataPrefix+fileName, encoding="utf-8" if fileName not in self.encodingSet else "ISO-8859-1")
                if i == 0:
                    self.name.append(fileNameLS[0])
                    insertedInd = ind
                    ind += 1
                else:
                    insertedInd = self.name.index(fileNameLS[0])
                tmpI = 0
                for line in file:
                    lineLS = line.split('\t')
                    lineLS[0] = int(lineLS[0])
                    lineLS[1] = lineLS[1].split()
                    tmpI += 1
                    ls[insertedInd].append(lineLS)
        for i in range(self.config.task):
            random.shuffle(self.train[i])

    def nextBatch(self):
        if self.config.wordemb_suffix == "mask_t":
            retX = [np.ones(shape=[self.config.batch_size, self.config.task_len[i]], dtype='int64') for i in range(self.config.task)]
        else:
            retX = [np.zeros(shape=[self.config.batch_size, self.config.task_len[i]], dtype='int64') for i in range(self.config.task)]
        retY = [np.zeros(shape=[self.config.batch_size]) for i in range(self.config.task)]
        retDomain = [np.zeros(shape=[self.config.batch_size]) for i in range(self.config.task)]
        retLength = [np.zeros(shape=[self.config.batch_size]) for i in range(self.config.task)]
        p = np.array([self.config.mask_prob, 1-self.config.mask_prob])
        for i in range(self.config.task):
            for j in range(self.config.batch_size):
                textItem = self.train[i][self.trainInd[i]*self.config.batch_size+j]
                minLen = min(self.config.task_len[i], len(textItem[1]))
                mask_arr = np.random.choice([0, 1], size=[minLen], p=p)
                for k in range(minLen):
                    if self.config.wordemb_suffix != "mask_t" or mask_arr[k] == 1:
                        retX[i][j][k] = self.word2id[textItem[1][k]] if textItem[1][k] in self.word2id else self.word2id["<unk>"]
                    else:
                        retX[i][j][k] = 0
                retY[i][j] = textItem[0]
                retDomain[i][j] = i
                retLength[i][j] = minLen
            self.trainInd[i] += 1
            if self.trainInd[i] == self.threshold[i]:
                self.trainInd[i] = 0
                random.shuffle(self.train[i])
        return retX, retY, retDomain, retLength 

    def getValid(self, type=0):
        if type == 0:
            indLS = self.validInd
            dataLS = self.valid
        else:
            indLS = self.testInd
            dataLS = self.test

        retLen = min(self.config.batch_size, len(dataLS[0]) - indLS[0]*self.config.batch_size)
        if retLen <= 0:
            for i in range(self.config.task):
                indLS[i] = 0
            return None, None, None, None, False
        if self.config.wordemb_suffix == "mask_t":
            retX = [np.ones(shape=[retLen, self.config.task_len[i]], dtype='int64') for i in range(self.config.task)]
        else:
            retX = [np.zeros(shape=[retLen, self.config.task_len[i]], dtype='int64') for i in range(self.config.task)]
        retY = [np.zeros(shape=[retLen]) for i in range(self.config.task)]
        retDomain = [np.zeros(shape=[retLen]) for i in range(self.config.task)]
        retLength = [np.zeros(shape=[retLen]) for i in range(self.config.task)]
        for i in range(self.config.task):
            for j in range(retLen):
                textItem = dataLS[i][indLS[i]*self.config.batch_size+j]
                minLen = min(self.config.task_len[i], len(textItem[1]))
                for k in range(minLen):
                    retX[i][j][k] = self.word2id[textItem[1][k]] if textItem[1][k] in self.word2id else self.word2id["<unk>"]
                retY[i][j] = textItem[0]
                retDomain[i][j] = i
                retLength[i][j] = minLen
            indLS[i] += 1
        return retX, retY, retDomain, retLength, True


    def getTest(self):
        return self.getValid(type=1)

if __name__ == "__main__":
    texti = TextIterator(util.get_args())
    for i in range(42):
        texti.nextBatch()
    while True:
        a, b, c, flag = texti.getValid()
        print(texti.validInd)
        if flag == False:
            break
    print(texti.trainInd)
