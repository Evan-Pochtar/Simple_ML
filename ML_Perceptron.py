
import numpy as np

def normalize(x, mean=None, std=None):
    if mean is None:
        mean = np.mean(x, axis=0).reshape(1,-1)
        std = np.std(x, axis=0).reshape(1,-1)
    x = (x-mean)/(std+1e-5)
    return x, mean, std

def process_label(label):
    one_hot = np.zeros([len(label),10])
    for i in range(len(label)):
        one_hot[i,label[i]] = 1
    return one_hot

def tanh(x):
    out = np.zeros_like(x)
    x = np.clip(x,a_min=-100,a_max=100)
    out = (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
    return out 

def softmax(x):
    out = np.zeros_like(x)
    out = np.exp(x)/np.exp(x).sum(-1).reshape(-1,1)
    return out


class MLP:
    def __init__(self,num_hid):
        self.num_hid = num_hid
        self.lr = 5e-3
        self.w = np.random.random([64,num_hid])
        self.w0 = np.random.random([1,num_hid])
        self.v= np.random.random([num_hid,10])
        self.v0 = np.random.random([1,10])

    def fit(self,train_x,train_y, valid_x, valid_y):
        count = 0
        best_valid_acc = 0

        # Training stops if there is no improvment over the best validation accuracy for more than 100 iterations
        while count<=50:
            z, y = self.forward(train_x)

            gra_v = self.dEdv(z, y, train_y)
            gra_v0 = self.dEdv0(y, train_y)
            gra_w = self.dEdw(z, y, train_x, train_y)
            gra_w0 = self.dEdw0(z, y, train_y)
            
            self.update(gra_w, gra_w0, gra_v, gra_v0)

            predictions = self.predict(valid_x)
            valid_acc = np.count_nonzero(predictions.reshape(-1)==valid_y.reshape(-1))/len(valid_x)

            if valid_acc > best_valid_acc:
                best_valid_acc = valid_acc
                count = 0
            else:
                count += 1

        return best_valid_acc

    def forward(self, x):
        z = np.zeros([len(x), self.num_hid])
        y = np.zeros([len(x), 10])

        z = tanh(x @ self.w + self.w0)
        y = softmax(z @ self.v + self.v0)

        return z, y

    def dEdv(self, z, y, r):
        out = np.zeros_like(self.v)
        out = np.transpose(z) @ (y - r)
        return out

    def dEdv0(self, y, r):
        out = np.zeros_like(self.v0)
        out = np.sum(y - r, axis=0)
        return out

    def dEdw(self, z, y, x, r):
        out = np.zeros_like(self.w)
        out = np.transpose(x) @ ((y - r) @ np.transpose(self.v) * (1 - z**2))
        return out

    def dEdw0(self, z, y, r):
        out = np.zeros_like(self.w0)
        out = np.sum((y - r) @ np.transpose(self.v) * (1 - z**2), axis=0)
        return out

    def update(self, gra_w, gra_w0, gra_v, gra_v0):
        self.w -= self.lr * gra_w
        self.w0 -= self.lr * gra_w0
        self.v -= self.lr * gra_v
        self.v0 -= self.lr * gra_v0
        return 

    def predict(self,x):
        z = tanh(x.dot(self.w) + self.w0)
        y = softmax(z.dot(self.v) + self.v0)
        y = np.argmax(y,axis=1)

        return y

    def get_hidden(self,x):
        z = tanh(x.dot(self.w) + self.w0)
        return z