import argparse
import torch
import random
import torch.nn.functional as F

def read_file(args):
    path = args.file_path
    words = open(path, 'r').read().splitlines()
    return words

def train_neural_network(train, test, dev, g, epochs,lr):

    chars = sorted(list(set(''.join(train)))) # getting the list of all unique characters in the dataset in sorted order(a-z)
    stoi = {s:i+1 for i,s in enumerate(chars)}
    stoi['.'] = 0
    itos = {i:s for s,i in stoi.items()}

    def create_dataset(words):
        # dataset
        xs , ys = [], []
        for w in words:
            chs = ['.'] + list(w) + ['.']
            for ch1,ch2,ch3 in zip(chs, chs[1:], chs[2:]):
                ix1 = stoi[ch1]
                ix2 = stoi[ch2]
                ix3 = stoi[ch3]
                xs.append((ix1,ix2))
                ys.append(ix3)

        xs = torch.tensor(xs)
        ys = torch.tensor(ys)
        num = xs.shape[0]
        return xs, ys, num

    # Model Training
    # train set
    xs_train, ys_train, num_train = create_dataset(train)

    # initializing the network
    W = torch.randn((54,27), generator=g, requires_grad=True)

    # gradient descent
    for k in range(epochs):
        # forward pass
        xenc = torch.cat([F.one_hot(xs_train[:,0], num_classes=27),
                          F.one_hot(xs_train[:,1], num_classes=27)],dim=1).float()
        logits = xenc @ W # log-counts
        counts = logits.exp()
        probs = counts/counts.sum(1, keepdim=True)
        loss = -probs[torch.arange(num_train), ys_train].log().mean() + 0.01*(W**2).mean() # nll + regularisation
        print(f"Loss at iteration {k+1} : {loss}")

        # backwards pass
        W.grad = None # set to zero gradient
        loss.backward()

        # update
        W.data += -lr * W.grad

    def evaluate_model(xs,ys,num):
        xenc = torch.cat([F.one_hot(xs[:,0], num_classes=27),
                          F.one_hot(xs[:,1], num_classes=27)],dim=1).float()
        logits = xenc @ W # log-counts
        counts = logits.exp()
        probs = counts/counts.sum(1, keepdim=True)
        loss = -probs[torch.arange(num), ys].log().mean() + 0.01*(W**2).mean() # nll + regularisation
        return loss

    # performance evaluation
    xs_val, ys_val, num_val = create_dataset(dev)
    xs_test, ys_test, num_test = create_dataset(test)
    train_loss = evaluate_model(xs_train,ys_train,num_train)
    val_loss = evaluate_model(xs_val, ys_val, num_val)
    test_loss = evaluate_model(xs_test, ys_test, num_test)

    def generate_sample_names():
        names = []
        for _ in range(10):
            out = []
            ix1 = 0
            ix2 = random.randint(1,26)
            out.append(itos[ix2])
            while True:
                xenc = torch.cat([F.one_hot(torch.tensor([ix1]), num_classes=27),
                            F.one_hot(torch.tensor([ix2]), num_classes=27)],dim=1).float()
                logits = xenc @ W # predict log-counts
                counts = logits.exp() # counts, equivalent to N
                p = counts / counts.sum(1, keepdims=True) # probabilities for next character
                
                ix1 = ix2
                ix2 = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
                out.append(itos[ix2])
                if ix2 == 0:
                    break
            names.append(''.join(out))
        return names

    names = generate_sample_names() 

    return train_loss, val_loss, test_loss, names

def train_count_trigram_model(train, test, dev, g):
    # creating the count matrix that keeps the count of [char1+char2] followed by [char3], we ensure that there are atleast 3 chars as the starting and ending tokens are `.` and each word is atleast one char long
    N = torch.zeros((27*27,27), dtype=torch.int32)

    chars = sorted(list(set(''.join(train)))) # getting the list of all unique characters in the dataset in sorted order(a-z)
    stoi = {s:i+1 for i,s in enumerate(chars)}
    stoi['.'] = 0
    itos = {i:s for s,i in stoi.items()}

    for w in train:
        chs = ['.'] + list(w) + ['.'] # adding the special start and end characters
        for ch1,ch2,ch3 in zip(chs, chs[1:], chs[2:]):
            ix1 = stoi[ch1]
            ix2 = stoi[ch2]
            ix3 = stoi[ch3]
            N[27*ix1+ix2, ix3] += 1

    # calculating the probability matrix
    P = (N+1).float()
    P /= P.sum(dim=1, keepdim=True) # +1 is added for smoothing, to ensure we don't encounter inf during predictions

    def evaluate_model(words):
        log_likelihood = 0.0
        n = 0
        for w in words:
            chs = ['.'] + list(w) + ['.']
            for ch1,ch2,ch3 in zip(chs, chs[1:], chs[2:]):
                ix1 = stoi[ch1]
                ix2 = stoi[ch2]
                ix3 = stoi[ch3]
                prob = P[27*ix1+ix2, ix3]
                logprob = torch.log(prob)
                log_likelihood += logprob
                n += 1

        return (-log_likelihood/n) # normalized negative log likelihood
    
    train_loss, dev_loss, test_loss = evaluate_model(train), evaluate_model(dev), evaluate_model(test)

    # generate a list sample values
    def generate_sample_names():
        names = []
        for _ in range(10):
            row_idx = random.randint(1,26) # uniformly selects the first character that name starts from
            char = itos[row_idx] 
            name = char
            idx1 = row_idx
            while char != '.':
                p = P[row_idx, :]
                idx2 = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
                char = itos[idx2]
                name += char
                row_idx = 27*idx1+idx2
                idx1 = idx2
            # print(name)
            names.append(name)
        return names
    
    sample_names = generate_sample_names()

    return train_loss, dev_loss, test_loss, sample_names

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--file_path', type=str,
                    help='Path of the text file to train and test the model')
    parser.add_argument('--method', type=str, choices=['count', 'neural'], default='count',
                        help='Either "count" based or "neural network" based training')
    parser.add_argument('--epochs', type=int, default=200,
                    help='Epochs to train the model')
    parser.add_argument('--lr', type=int, default=50,
                    help='Learning rate')
    
    args = parser.parse_args()

    words = read_file(args)

    g = torch.Generator().manual_seed(2147483647)
    train_size = int(0.8 * len(words))
    test_size = int(0.1 * len(words))
    dev_size = len(words) - (train_size+test_size)
    train, dev, test = torch.utils.data.random_split(words, [train_size, dev_size, test_size], generator=g)

    if args.method == 'count':
        ## `training` the count based trigram model
        train_loss, dev_loss, test_loss, sample_names = train_count_trigram_model(train, test, dev, g)
        print('--------------------------------')
        print('Count Based Training')
        print(f'Train Loss: {train_loss :.4f}')
        print(f'Val Loss: {dev_loss :.4f}')
        print(f'Test Loss: {test_loss :.4f}')
        print(f'Sample names generated by the model: ')
        for names in sample_names:
            print(names)
    else:
        print('--------------------------------')
        print('Neural Network Based Training')
        train_loss, dev_loss, test_loss, sample_names = train_neural_network(train, test, dev, g, args.epochs, args.lr)
        print('--------------------------------')
        print('Count Based Training')
        print(f'Train Loss: {train_loss :.4f}')
        print(f'Val Loss: {dev_loss :.4f}')
        print(f'Test Loss: {test_loss :.4f}')
        print(f'Sample names generated by the model: ')
        for names in sample_names:
            print(names)

if __name__ == '__main__':
    main()