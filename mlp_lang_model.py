'''
This code is the implementation of the paper Neural Language Model Bengio et al 2003
'''
import torch
import torch.nn.functional as F
import random
import argparse
import time

def read_file(args):
    path = args.file_path
    words = open(path, 'r').read().splitlines()
    chars = sorted(list(set(''.join(words)))) # getting the list of all unique characters in the dataset in sorted order(a-z)
    stoi = {s:i+1 for i,s in enumerate(chars)}
    stoi['.'] = 0
    itos = {i:s for s,i in stoi.items()}
    return words, stoi, itos

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

def train_network(Xtr, Ytr, Xdev, Ydev, Xte, Yte, g, epochs, batch_sz, blr, decay, emb_dim, block_sz, neurons_second_layer, samples, stoi, itos):
    C = torch.randn((27,emb_dim), generator=g)
    W1 = torch.randn((emb_dim*block_sz,neurons_second_layer), generator=g)
    b1 = torch.randn((neurons_second_layer), generator=g)
    W2 = torch.randn((neurons_second_layer,27), generator=g)
    b2 = torch.randn((27), generator=g)
    parameters = [C,W1,b1,W2,b2]
    

    print(f"Number of parameters in the model: {sum(p.nelement() for p in parameters)}")

    for p in parameters:
        p.requires_grad = True

    lossi = []
    stepi = []

    for i in range(epochs):

        # mini batch construct
        ix = torch.randint(0, Xtr.shape[0], (batch_sz,))  

        # forward pass
        emd = C[Xtr[ix]]
        h = torch.tanh(emd.view(-1,emb_dim*block_sz) @ W1 + b1)
        logits = h @ W2 + b2
        loss = F.cross_entropy(logits, Ytr[ix])

        # backwards pass
        for p in parameters:
            p.grad = None
        loss.backward()

        # optimize
        lr = blr if i < epochs/2 else (blr*decay)
        for p in parameters:
            p.data += -lr * p.grad
        
        lossi.append(loss.log10().item())
        stepi.append(i)

        if i%10000 == 0:
            print(f"Step: {i} || Loss: {loss.item()}")
    
    # final train loss
    emd = C[Xtr]
    h = torch.tanh(emd.view(-1,emb_dim*block_sz) @ W1 + b1)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Ytr)
    print(f"Final Train Loss: {loss.item()}")

    # final val loss
    emd = C[Xdev]
    h = torch.tanh(emd.view(-1,emb_dim*block_sz) @ W1 + b1)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Ydev)
    print(f"Final Dev Loss: {loss.item()}")

    # generating samples from the trained model
    generated_samples = []
    for _ in range(samples):
        out = []
        context = [0] * block_sz
        while True:
            emd = C[torch.tensor([context])] # (1, block_size, d)
            h = torch.tanh(emd.view(1,-1) @ W1 + b1)
            logits = h @ W2 + b2
            probs = F.softmax(logits, dim=1)
            ix = torch.multinomial(probs, num_samples=1, generator=g).item()
            context = context[1:] + [ix]
            out.append(ix)
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
    parser.add_argument('--blr', type=float, default=0.1,
                    help='Learning rate')
    parser.add_argument('--decay', type=float, default=0.1,
                    help='Decay in learning rate after first stage of training is completed')
    parser.add_argument('--emd_dim', type=int, default=10,
                    help='Embedding Dimension')
    parser.add_argument('--block_sz', type=int, default=3,
                    help='Block Size')
    parser.add_argument('--neurons_second_layer', type=int, default=200,
                    help='Number of neurons in the second layer')
    parser.add_argument('--samples', type=int, default=20,
                    help='Number of samples to be generated')
    args = parser.parse_args()

    words, stoi, itos = read_file(args)

    g = torch.Generator().manual_seed(2147483647+10)
    n1 = int(0.8 * len(words))
    n2 = int(0.9 * len(words))
    Xtr, Ytr = build_dataset(words[:n1], args.block_sz, stoi)
    Xdev, Ydev = build_dataset(words[n1:n2], args.block_sz, stoi)
    Xte, Yte = build_dataset(words[n2:], args.block_sz, stoi)

    print('Training Started')
    start_time = time.time()
    generated_samples = train_network(Xtr, Ytr, Xdev, Ydev, Xte, Yte, g, args.epochs, args.batch_sz, args.blr, args.decay, args.emd_dim, args.block_sz, args.neurons_second_layer, args.samples, stoi, itos)
    end_time = time.time()
    print(f"Time taken for training: {end_time - start_time :.4f}")

    for sample in generated_samples:
        print(sample)

if __name__ == '__main__':
    main()