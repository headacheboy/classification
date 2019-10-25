import torch
import torch.nn.functional as F
import numpy as np

class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lamb=1.0):
        ctx.lamb = lamb
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lamb, None 

class EmbedLayer(torch.nn.Module):
    def __init__(self, config):
        super(EmbedLayer, self).__init__()
        self.emb = torch.nn.Embedding(config.text_vocab_size, config.text_emb)
        self.drop = torch.nn.Dropout(config.dropout_keep_rate)

    def forward(self, x, single=False):
        if single:
            return self.emb(x)
        word_emb = []
        for ele in x:
            word_emb.append(self.emb(ele).detach())
            #word_emb.append(self.emb(ele))
        return word_emb

class LSTMLayer(torch.nn.Module):
    def __init__(self, config, layer=1):
        super(LSTMLayer, self).__init__()
        self.config = config
        self.drop = torch.nn.Dropout(config.dropout_keep_rate)
        if layer > 1:
            text_emb = config.text_emb*2
        else:
            text_emb = config.text_emb
        self.lstm = torch.nn.LSTM(text_emb, config.hidden_size, config.lstm_layer_size, dropout=config.dropout_keep_rate, batch_first=True, bidirectional=self.config.bidirectional)

    def forward(self, word_emb, initial_state=None):
        word_emb_new = self.drop(word_emb)
        #word_emb_new = word_emb
        if initial_state is None:
            h0 = torch.autograd.Variable(torch.zeros((1+self.config.bidirectional)*self.config.lstm_layer_size, word_emb_new.size(0), self.config.hidden_size)).cuda()
            c0 = torch.autograd.Variable(torch.zeros((1+self.config.bidirectional)*self.config.lstm_layer_size, word_emb_new.size(0), self.config.hidden_size)).cuda()
            initial_state = (h0, c0)
        out, (hn, cn) = self.lstm(word_emb_new, initial_state)
        avg_out = torch.mean(out, dim=1)
        return avg_out, out

class CNNLayer(torch.nn.Module):
    def __init__(self, config):
        super(CNNLayer, self).__init__()
        ind = 0
        self.config = config
        self.drop = torch.nn.Dropout(config.dropout_keep_rate)
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=config.out_channel, kernel_size=(2, config.text_emb), stride=1, padding=0)# out_channel=100 
        self.conv2 = torch.nn.Conv2d(in_channels=1, out_channels=config.out_channel, kernel_size=(4, config.text_emb), stride=1, padding=0) 
        self.conv3 = torch.nn.Conv2d(in_channels=1, out_channels=config.out_channel, kernel_size=(6, config.text_emb), stride=1, padding=0) 
        self.maxPool1 = torch.nn.MaxPool2d(kernel_size=(config.task_len[ind] - 2 + 1, 1))
        self.maxPool2 = torch.nn.MaxPool2d(kernel_size=(config.task_len[ind] - 4 + 1, 1))
        self.maxPool3 = torch.nn.MaxPool2d(kernel_size=(config.task_len[ind] - 6 + 1, 1))

    def forward(self, word_emb):
        word_emb_1 = self.drop(word_emb)
        word_emb_new = word_emb.view(word_emb_1.shape[0], 1, word_emb_1.shape[1], -1)
        x1 = self.maxPool1(F.relu(self.conv1(word_emb_new))).view(word_emb.shape[0], -1)
        x2 = self.maxPool2(F.relu(self.conv2(word_emb_new))).view(word_emb.shape[0], -1)
        x3 = self.maxPool3(F.relu(self.conv3(word_emb_new))).view(word_emb.shape[0], -1)
        output = self.drop(torch.cat((x1, x2, x3), 1))
        return output, None


class LinearLayer(torch.nn.Module):
    def __init__(self, config, name, namePrv):
        super(LinearLayer, self).__init__()
        self.config = config
        self.name = name
        if namePrv == "LSTM":
            if self.config.add_no_rev_grad:
                con = 3
                advCon = 2
            else:
                con = 2
                advCon = 1
            if name == "adv":
                self.fc = torch.nn.Linear(config.hidden_size*advCon*(1+self.config.bidirectional), config.task)
            elif name == "task":
                self.fc = torch.nn.Linear(config.hidden_size*con*(1+self.config.bidirectional), config.label_num)
        elif namePrv == "CNN":
            if self.config.add_no_rev_grad:
                con = 3
                advCon = 2
            else:
                con = 2
                advCon = 1
            if name == "adv":
                self.fc = torch.nn.Linear(config.out_channel*advCon*3, config.task)
            elif name == "task":
                #self.fc = torch.nn.Linear(config.out_channel*con*3, config.label_num)
                self.fc = torch.nn.Linear(config.out_channel*3, config.label_num)

    def forward(self, x):
        if self.name == "adv":
            if self.config.add_no_rev_grad:
                x[0] = GradReverse.apply(x[0])
                logits = self.fc(torch.cat(x, dim=1))
            else:
                x = GradReverse.apply(x)
                logits = self.fc(x)
        elif self.name == "task":
            #logits = self.fc(torch.cat(x, dim=1))
            logits = self.fc(x[0])
        return logits

class Model(torch.nn.Module):
    def __init__(self, config, model_name):
        super(Model, self).__init__()
        self.config = config
        self.emb = EmbedLayer(config)
        self.mem = torch.nn.Parameter(torch.randn(config.task, config.text_emb))
        self.domainLSTM = LSTMLayer(config)
        self.taskLSTM = LSTMLayer(config, 2)
        self.drop = torch.nn.Dropout(config.dropout_keep_rate)
        self.taskLinear = torch.nn.Linear((1+config.bidirectional)*config.hidden_size, config.hidden_size)
        self.taskLinear2 = torch.nn.Linear(config.hidden_size, config.label_num)
        self.mapLinear = torch.nn.ModuleList([
            torch.nn.Linear(config.text_emb, config.hidden_size, bias=False), 
            torch.nn.Linear((1+config.bidirectional)*config.hidden_size * 2  , config.hidden_size, bias=False), 
            torch.nn.Linear((1+config.bidirectional)*config.hidden_size, config.hidden_size, bias=False), 
            torch.nn.Linear((1+config.bidirectional)*config.hidden_size + config.text_emb, config.hidden_size, bias=False), 
            torch.nn.Linear((1+config.bidirectional)*config.hidden_size, config.hidden_size, bias=False), 
            torch.nn.Linear((1+config.bidirectional)*config.hidden_size, config.hidden_size, bias=False), 
            ])
        self.attLinear = torch.nn.ModuleList([torch.nn.Linear(config.hidden_size, 1) for i in range(3)])
        self.domainLinear = torch.nn.Linear(config.text_emb, config.task)
        self.embLinear = torch.nn.ModuleList([torch.nn.Linear(config.text_emb, config.text_emb) for i in range(16)])
        self.embMap = torch.nn.Linear((1+config.bidirectional)*config.hidden_size, config.text_emb)
        self.embMap2 = torch.nn.Linear((1+config.bidirectional)*config.hidden_size, config.text_emb)
        def weights_init(m):
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight.data, mean=0., std=0.1)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias.data, 0.01)
            elif isinstance(m, torch.nn.LSTM):
                torch.nn.init.normal_(m.all_weights[0][0], mean=0., std=0.1)
                torch.nn.init.normal_(m.all_weights[0][1], mean=0., std=0.1)
                torch.nn.init.normal_(m.all_weights[1][0], mean=0., std=0.1)
                torch.nn.init.normal_(m.all_weights[1][1], mean=0., std=0.1)
        self.apply(weights_init)

    def forward(self, x, y, length, training=False, loss_func=torch.nn.MSELoss()):
        taskLogit, advLogit, weightLogit, _, _ = self.fftraining(x, y, length)
        if training:
            return taskLogit, advLogit, weightLogit, None, None
        taskLogit = torch.split(taskLogit, x[0].size(0))
        return taskLogit, None, None, None, None


    def fftraining(self, x, y, length):
        length = torch.cat(length, dim=0)

        word_emb = self.emb(x)
        word_emb = torch.cat(word_emb, dim=0)
        domainOut_ori, domainSeqOut = self.domainLSTM(word_emb)
        domainEmb = self.embMap2(domainSeqOut)
        domainOut = self.embMap(domainOut_ori)
        advLogit = self.domainLinear(F.relu(domainOut))
        word_emb_new = torch.cat([word_emb, domainEmb], dim=2)

        finalOut, taskSeqOut = self.taskLSTM(word_emb_new)
        
        hid = F.tanh(self.mapLinear[1](torch.cat([taskSeqOut, domainOut_ori.expand(taskSeqOut.size(1), domainOut_ori.size(0), domainOut_ori.size(1)).transpose(0, 1)], dim=2)))
        score = self.attLinear[0](hid).squeeze()
        if length is not None:
            max_len = score.size(1)
            idxes = torch.arange(0, max_len, out=torch.LongTensor(max_len)).unsqueeze(0).cuda()
            mask = (idxes<length.unsqueeze(1)).float()
        score = F.softmax(score, dim=1)
        if length is not None:
            score = score * mask
        score = score.view(-1, 1, score.size(1))
        finalOut = torch.matmul(score, taskSeqOut).view(-1,taskSeqOut.size(2))
        
        taskLogit = self.taskLinear2(F.relu(self.taskLinear(finalOut)))
        return taskLogit, advLogit, None, None, None

    def att(self, seq, guidence, ind, domainInd=-1, length=None):
        score = []
        tmp = self.mapLinear[ind*2+1](guidence)
        tmp = tmp.expand(seq.size(1), tmp.size(0), tmp.size(1)).transpose(0, 1)
        hid = F.relu(self.mapLinear[ind*2](seq)+tmp)
        score = self.attLinear[ind](hid).squeeze()
        if length is not None:
            max_len = score.size(1)
            idxes = torch.arange(0, max_len, out=torch.LongTensor(max_len)).unsqueeze(0).cuda()
            if domainInd == -2:
                mask = (idxes<length.unsqueeze(1)).float()
            else:
                mask = (idxes<length[domainInd].unsqueeze(1)).float()
        score = F.softmax(score, dim=1)
        if length is not None:
            score = score * mask
        score = score.view(-1, 1, score.size(1))
        finalOut = torch.matmul(score, seq).view(-1, seq.size(2))
        return finalOut, score
