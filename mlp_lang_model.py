'''
This code is the implementation of the paper Neural Language Model Bengio et al 2003
Initialization is done using Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification He et al (2015)
BatchNorm layer is made according to Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift Ioffe et al (2015)
'''
import torch
import torch.nn.functional as F
import random
import argparse
import time

g = torch.Generator().manual_seed(2147483647)

class Linear:
    def __init__(self, fan_in, fan_out, bias=True):
        self.weight = torch.randn((fan_in, fan_out),generator=g) / (fan_in ** 0.5)
        self.bias = torch.zeros(fan_out) if bias else None
    
    def __call__(self, x):
        self.out = x @ self.weight
        if self.bias is not None:
            self.out += self.bias
        return self.out
    
    def parameters(self):
        return [self.weight] + ([] if self.bias is None else [self.bias])

class BatchNorm1d:
    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.momentum = momentum
        self.training = True
        # params trained with backprop
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)
        # buffers trained with momentum update
        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)

    def __call__(self, x):
        # calculate forward pass
        if self.training:
            xmean = x.mean(dim=0, keepdim=True)
            xvar = x.var(dim=0, keepdim=True)
        else:
            xmean = self.running_mean
            xvar = self.running_var
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps)
        self.out = self.gamma * xhat + self.beta
        # update buffers
        if self.training:
            with torch.no_grad():
                self.running_mean = (1-self.momentum) * self.running_mean + self.momentum*xmean
                self.running_var = (1-self.momentum) * self.running_var + self.momentum*xvar
        return self.out
    
    def parameters(self):
        return [self.gamma, self.beta]

class Tanh:
  def __call__(self, x):
    self.out = torch.tanh(x)
    return self.out
  def parameters(self):
    return []

def read_file(args):
    path = args.file_path
    words = open(path, 'r').read().splitlines()
    chars = sorted(list(set(''.join(words)))) # getting the list of all unique characters in the dataset in sorted order(a-z)
    stoi = {s:i+1 for i,s in enumerate(chars)}
    stoi['.'] = 0
    itos = {i:s for s,i in stoi.items()}
    vocab_size = len(itos)
    return words, stoi, itos, vocab_size

def build_dataset(words, blocksz, stoi):
    block_size = blocksz
    X,Y = [], []

    for w in words:
        context = [0]*block_size
        for ch in w+'.':
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]

    X = torch.tensor(X)
    Y = torch.tensor(Y)
    # print(X.shape, Y.shape)
    return X,Y

def train_network(Xtr, Ytr, Xdev, Ydev, Xte, Yte, g, epochs, batch_sz, blr, decay, emb_dim, block_sz, n_hidden, samples, stoi, itos, vocab_size):
    C = torch.randn((vocab_size,emb_dim), generator=g)
    layers = [
        Linear(emb_dim * block_sz, n_hidden), BatchNorm1d(n_hidden), Tanh(),
        Linear(n_hidden, n_hidden), BatchNorm1d(n_hidden), Tanh(),
        Linear(n_hidden, n_hidden), BatchNorm1d(n_hidden), Tanh(),
        Linear(n_hidden, n_hidden), BatchNorm1d(n_hidden), Tanh(),
        Linear(n_hidden, n_hidden), BatchNorm1d(n_hidden), Tanh(),
        Linear(n_hidden, vocab_size), BatchNorm1d(vocab_size)
    ]
    
    with torch.no_grad():
        layers[-1].gamma *= 0.1
        for layer in layers[:-1]:
            if isinstance(layer, Linear):
                layer.weight *= 5/3

    parameters = [C] + [p for layer in layers for p in layer.parameters()]
    print(sum(p.nelement() for p in parameters))
    for p in parameters:
        p.requires_grad = True

    lossi = []
    stepi = []

    for i in range(epochs):
        # mini batch construct
        ix = torch.randint(0, Xtr.shape[0], (batch_sz,), generator=g) 
        Xb, Yb = Xtr[ix], Ytr[ix] # batch X,Y

        # forward pass
        emb = C[Xb]
        x = emb.view(emb.shape[0], -1) # concatenate the vectors
        for layer in layers:
            x = layer(x)
        loss = F.cross_entropy(x, Yb) # loss function

        # backwards pass
        for p in parameters:
            p.grad = None
        loss.backward()

        # update
        lr = blr if i < epochs/2 else (blr*decay)
        for p in parameters:
            p.data += -lr * p.grad
        
        lossi.append(loss.log10().item())
        stepi.append(i)

        if i%10000 == 0:
            print(f"Step: {i} || Loss: {loss.item()}")

    @torch.no_grad()
    def split_loss(X,Y,split):
        # final train loss
        emb = C[X]
        x = emb.view(emb.shape[0], -1)
        for layer in layers:
            x = layer(x)
        loss = F.cross_entropy(x, Y)
        print(f"Final {split} Loss: {loss.item()}")
    
    # put layers into eval mode
    for layer in layers:
        layer.training = False
    split_loss(Xtr, Ytr, 'Train')
    split_loss(Xdev, Ydev, 'Val')

    # generating samples from the trained model
    generated_samples = []
    for _ in range(samples):
        out = []
        context = [0] * block_sz
        while True:
            emb = C[torch.tensor([context])]
            x = emb.view(emb.shape[0], -1)
            for layer in layers:
                x = layer(x)
            logits = x
            probs = F.softmax(logits, dim=1)
            # sample from the distribution
            ix = torch.multinomial(probs, num_samples=1, generator=g).item()
            context = context[1:] + [ix]
            out.append(ix)
            # if we sample the special '.' token, break
            if ix == 0:
                break
        generated_samples.append((''.join(itos[i] for i in out)))
    
    return generated_samples

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--file_path', type=str,
                    help='Path of the text file to train and test the model')
    parser.add_argument('--epochs', type=int, default=200000,
                    help='Epochs to train the model')
    parser.add_argument('--batch_sz', type=int, default=32,
                    help='Batch Size')
    parser.add_argument('--blr', type=float, default=0.09,
                    help='Learning rate')
    parser.add_argument('--decay', type=float, default=0.1,
                    help='Decay in learning rate after first stage of training is completed')
    parser.add_argument('--emd_dim', type=int, default=10,
                    help='Embedding Dimension')
    parser.add_argument('--block_sz', type=int, default=3,
                    help='Block Size')
    parser.add_argument('--n_hidden', type=int, default=100,
                    help='Number of neurons in the hidden layer')
    parser.add_argument('--samples', type=int, default=20,
                    help='Number of samples to be generated')
    args = parser.parse_args()

    words, stoi, itos, vocab_size = read_file(args)

    n1 = int(0.8 * len(words))
    n2 = int(0.9 * len(words))
    Xtr, Ytr = build_dataset(words[:n1], args.block_sz, stoi)
    Xdev, Ydev = build_dataset(words[n1:n2], args.block_sz, stoi)
    Xte, Yte = build_dataset(words[n2:], args.block_sz, stoi)

    print('Training Started')
    start_time = time.time()
    generated_samples = train_network(Xtr, Ytr, Xdev, Ydev, Xte, Yte, g, args.epochs, args.batch_sz, args.blr, args.decay, args.emd_dim, args.block_sz, args.n_hidden, args.samples, stoi, itos, vocab_size)
    end_time = time.time()
    print(f"Time taken for training: {end_time - start_time :.4f}")

    for sample in generated_samples:
        print(sample)

if __name__ == '__main__':
    main()