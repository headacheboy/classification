import numpy as np
import math
import torch
import util
import loadData
import NNManager
import sys
import time

class Main():
    def __init__(self):
        self.maxAcc = 0.0
        self.config = util.get_args()
        self.config.lr_decay = self.config.lr_decay * (1400 // self.config.batch_size)
        self.config.lr_decay_begin = self.config.lr_decay_begin * (1400 // self.config.batch_size)
        self.config.maxSteps = self.config.epochs * 1400 // self.config.batch_size + 1
        print(self.config.maxSteps, " max steps")
        self.texti = loadData.TextIterator(self.config)
        self.config.text_vocab_size = len(self.texti.word2id)
        embed_weight = np.load("../wordEmb/vector_"+self.config.wordemb_suffix+".npy")
        self.model = NNManager.Model(self.config, self.config.model_name)
        self.model.emb.emb.weight.data.copy_(torch.from_numpy(embed_weight))
        if self.config.pretrain == 1:
            self.model.load_state_dict(torch.load(self.config.pretrain_path, map_location='cpu'))
        self.model.cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        self.lossfunc = torch.nn.CrossEntropyLoss().cuda()
        self.start = time.time()

    def valid(self):
        tot = [0 for i in range(self.config.task)]
        err = [0 for i in range(self.config.task)]
        while True:
            validX_c, validY_c, validDomain_c, validLength_c, flag = self.texti.getValid()
            
            if not flag:
                break

            validX = [torch.autograd.Variable(torch.from_numpy(validX_c[i])).cuda() for i in range(self.config.task)]
            validY = [torch.autograd.Variable(torch.from_numpy(validY_c[i]).long()).cuda() for i in range(self.config.task)]
            validDomain = [torch.autograd.Variable(torch.from_numpy(validDomain_c[i]).long()).cuda() for i in range(self.config.task)]
            validLength = [torch.autograd.Variable(torch.from_numpy(validLength_c[i]).long()).cuda() for i in range(self.config.task)]

            taskLogit, advLogit, weightLogit, tmpShareOutput, enableVector = self.model(validX, validY, validLength)

            for i in range(self.config.task):
                taskOutput = np.argmax(taskLogit[i].cpu().data.numpy(), axis=1)
                err[i] += sum(taskOutput == validY_c[i])
                tot[i] += validX_c[i].shape[0]
        if self.config.cross is None:
            print("valid acc: tot rate " + str(sum(err) / sum(tot)))
        else:
            print("valid acc: tot rate " + str(err[self.config.cross] / tot[self.config.cross]))

        tot = [0 for i in range(self.config.task)]
        err = [0 for i in range(self.config.task)]
        while True:
            testX_c, testY_c, testDomain_c, testLength_c, flag = self.texti.getTest()
            if flag == False:
                break
            testX = [torch.autograd.Variable(torch.from_numpy(testX_c[i])).cuda() for i in range(self.config.task)]
            testY = [torch.autograd.Variable(torch.from_numpy(testY_c[i]).long()).cuda() for i in range(self.config.task)]
            testDomain = [torch.autograd.Variable(torch.from_numpy(testDomain_c[i]).long()).cuda() for i in range(self.config.task)]
            testLength = [torch.autograd.Variable(torch.from_numpy(testLength_c[i]).long()).cuda() for i in range(self.config.task)]

            taskLogit, advLogit, weightLogit, tmpShareOutput, enableVector = self.model(testX, testY, testLength)

            for i in range(self.config.task):
                taskOutput = np.argmax(taskLogit[i].cpu().data.numpy(), axis=1)
                err[i] += sum(taskOutput == testY_c[i])
                tot[i] += testX_c[i].shape[0]
        if self.config.cross is None:
            print("test: tot rate" + str(sum(err) / sum(tot)))
        else:
            print("test: tot rate" + str(err[self.config.cross] / tot[self.config.cross]))

    def display(self, loss, lossT):
        timeN = time.time()
        print("loss: {0:.5f}, lossTask: {1:.5f}, time: {2:.5f}".format(loss, lossT, timeN-self.start))
        self.start = timeN

    def trainingProcess(self):
        avgLoss = 0.0
        avgLossTask = 0.0
        avgLossAdv = 0.0
        self.model.train()
        step = 0
        while step < self.config.maxSteps:
            batchX, batchY, batchDomain, batchLength = self.texti.nextBatch()

            self.optimizer.zero_grad()

            batchX = [torch.autograd.Variable(torch.from_numpy(batchX[i])).cuda() for i in range(self.config.task)]
            batchY = [torch.autograd.Variable(torch.from_numpy(batchY[i]).long()).cuda() for i in range(self.config.task)]
            batchDomain = [torch.autograd.Variable(torch.from_numpy(batchDomain[i]).long()).cuda() for i in range(self.config.task)]
            batchLength = [torch.autograd.Variable(torch.from_numpy(batchLength[i]).long()).cuda() for i in range(self.config.task)]

            lossTask = 0.0
            lossAdv = 0.0
            lossWeight = 0.0
            lossDomain = 0.0
            
            taskLogit, advLogit, weightLogit, tmpShareOutput, enableVector = self.model(batchX, batchY, batchLength, training=True)

            batchY = torch.cat(batchY, dim=0)
            batchDomain = torch.cat(batchDomain, dim=0)
            lossTask = self.lossfunc(taskLogit, batchY)
            lossDomain = self.lossfunc(advLogit, batchDomain)

            if step > 10 * (1400 // self.config.batch_size):
                loss = lossTask 
            else:
                loss = lossTask + self.config.lamb*(lossDomain) 

            if step == 10 * (1400 // self.config.batch_size):
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.config.learning_rate
                    
                
            avgLoss += float(loss)
            avgLossTask += float(lossTask)
            loss.backward()

            # clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.maxClip)

            self.optimizer.step()



            if step % self.config.display_step == 0:
                print("step: ", step, end=" ")
                self.display(avgLoss/self.config.display_step, avgLossTask/self.config.display_step)
                avgLoss = 0.0
                avgLossTask = 0.0
                avgLossAdv = 0.0
            
            if step % self.config.valid_step == 0:
                tmpLS = []
                self.model.eval()
                self.valid()
                self.model.train()

            step += 1

            if self.config.decay_method == "exp":
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.config.learning_rate * (self.config.lr_decay_rate ** (step/self.config.lr_decay))
            elif self.config.decay_method == "linear":
                if step > self.config.lr_decay_begin and (step - self.config.lr_decay_begin) % self.config.lr_decay == 0:
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = param_group['lr'] * self.config.lr_decay_rate
            elif self.config.decay_method == "cosine":
                gstep = min(step, self.config.lr_decay)
                cosine_decay = 0.5 * (1 + math.cos(math.pi * gstep / self.config.lr_decay))
                decayed = (1 - self.config.min_lr) * cosine_decay + self.config.min_lr
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.config.learning_rate * decayed

        print("last eval:")
        self.model.eval()
        self.valid()

if __name__ == "__main__":
    m = Main()
    m.trainingProcess()
