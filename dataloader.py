import numpy as np
import torch.nn.utils.rnn as rnn_utils
import torch
import random

SOS_token = 0
EOS_token = 1
PAD_token = 29
UNK_token = 30
MAX_LENGTH = 16

def stringToList(S):
    return [ord(c)-95 for c in S]

class dataloader(object):
    def __init__(self, type):
        if type == 'train':
            self.X = loadTrainDataset()
            self.N = self.X.shape[0]
            # gen all pair
            self.pairs = []
            for cond in range(4):
                for i in range(self.N):
                    lis = stringToList(self.X[i][cond])
                    self.pairs.append((lis, lis + [EOS_token], cond))
            self.PairN = len(self.pairs)
        else:
            self.X = loadTestDataset()
            self.Tense = loadTestTenseDataset()
            self.N = self.X.shape[0]


        '''
        for cond in range(4):
            arr = self.X[:,cond]
            arr = arr[np.argsort(-np.char.str_len(arr))]
            self.X[:,cond] = arr
        '''

        '''
        self.input_tensors = np.zeros((4,self.N,MAX_LENGTH),int)
        self.target_tensors = np.zeros((4,self.N,MAX_LENGTH),int)
        self.length_tensors = np.zeros((4,self.N),int)



        for cond in range(4):
            b=0
            e=self.N

            input_tensor = [[SOS_token] + stringToList(self.X[index][cond])  for index in range(b,e)]
            target_tensor = [stringToList(self.X[index][cond]) + [EOS_token] for index in range(b,e)]
            length_tensor = [len(t) for t in input_tensor]
            input_tensor = [t + [PAD_token] * (MAX_LENGTH - len(t)) for t in input_tensor]
            target_tensor = [t + [PAD_token] * (MAX_LENGTH - len(t)) for t in target_tensor]

            input_tensor = np.array(input_tensor)
            target_tensor = np.array(target_tensor)
            length_tensor = np.array(length_tensor)


            self.input_tensors[cond] = input_tensor
            self.target_tensors[cond] = target_tensor
            self.length_tensors[cond] = length_tensor
        '''


    def genShufflePairs(self):
        random.shuffle(self.pairs)
        return self.pairs


    def tensorsFromPair(self,index,train_col=0,EOSToken=1):

        strX = stringToList(self.X[index][train_col])
        strY = stringToList(self.X[index][train_col]) + [EOSToken]
        cond = train_col
        return (strX, strY, cond)

    def genAllBatchOrder(self,batch_size):
        orders = []
        for cond in range(4):
            cond_b = []
            b_counter = 0
            for b in range(0,self.N,batch_size):
                e = b+batch_size
                if e > self.N:
                    e = self.N
                orders.append((cond,range(b,e)))

        random.shuffle(orders)
        return orders
    def getBatch(self, order):
        cond, indx = order
        #print(self.input_tensors[cond].shape)
        #print(self.target_tensors[cond].shape)
        #print(self.length_tensors[cond].shape)
        print(cond,end=' | ')
        print(indx)
        return {'input':self.input_tensors[cond][indx]
            ,'target':self.target_tensors[cond][indx]
            , 'length': self.length_tensors[cond][indx]
            , 'cond': cond}

def loadTrainDataset():
    f = open('data/train.txt', 'r')
    x = f.readlines()
    f.close()
    x = [str.replace('\n','').split(' ') for str in x]
    return np.array(x)

def loadTestDataset():
    f = open('data/test.txt', 'r')
    x = f.readlines()
    f.close()
    x = [str.replace('\n','').split(' ') for str in x]
    return np.array(x)

def loadTestTenseDataset():
    f = open('data/testtense.txt', 'r')
    x = f.readlines()
    f.close()
    x = [str.replace('\n','').replace(' ','').split('->') for str in x]
    trans = {'sp':0,'tp':1,'pg':2,'p':3}
    out = []
    for p in x:
        out.append([trans[p[0]],trans[p[1]]])
    return np.array(out)

def getTenseLabel(i):
    l = ['sp','tp','pg','p']
    return l[i]

def allStringsToList(Sarr):
    return [stringToList(S) for S in Sarr]


