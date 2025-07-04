import torch
from torch.utils.data import Dataset

class AdditionDataset(Dataset):
    """
    Creates n-digit addition problems. For example, if n=2, then an example
    addition problem would be to add 85 + 50 = 135. This problem would be
    represented as the following string for the GPT:

    "8550531"
    x: 8  5  5  0  5 3 
    y: -1 -1 -1 5  3 1 

    This is because:
    - we are discarding the + and =, which are not necessary. We just encode the digits
      of the input numbers concatenated together.
    - the result 135 is encoded backwards to make the addition easier to learn for the
      GPT model, because of how the addition algorithm works.

    As one more example, the problem 6 + 39 = 45 would be encoded as:

    "0639540"
    x: 0  6  3  9  5 4
    y: -1 -1 -1 5  4 0

    At test time, we will feed in an addition problem by giving the first 2n digits,
    and hoping that the GPT model completes the sequence with the next (n+1) digits
    correctly.

    if formated is True, '+' and '=' are included in the input sequence as
    
    "85+50=531"
    x: 8  5  +  5  0  = 5 3 
    y: -1 -1 -1 -1 -1 5 3 1 
    """

    def __init__(self, ndigit=2, split='train', format_vocab=None, seed=42):
        self.ndigit = ndigit
        self.split = split          # train/test/val
        self.format_vocab = format_vocab
        if format_vocab is not None:
            assert '+' in format_vocab and '=' in format_vocab
            assert format_vocab['+'] not in list(range(10))
            assert format_vocab['='] not in list(range(10))

        # split up all addition problems into either training data or test data
        ndigit = self.ndigit
        assert ndigit <= 4, "the lines below would be very memory inefficient, in future maybe refactor to support"
        num = (10**ndigit)**2   # total number of possible addition problems with ndigit numbers
        rng = torch.Generator()
        rng.manual_seed(seed)
        perm = torch.randperm(num, generator=rng)

        num_test = int(num*0.15)
        num_val = int(num*0.15)  
        
        if split == 'train':
            self.ixes = perm[num_test+num_val:]
        elif split == 'val':
            self.ixes = perm[num_test:num_test+num_val]
        elif split == 'test':
            self.ixes = perm[:num_test]
        else:
            raise ValueError(f"Unknown split: {split}")

        self.length = len(self.ixes)

    def __len__(self):
        return self.length

    def __getitem__(self, idx, with_raw=False):        
        # given a problem index idx, first recover the associated a + b
        idx = idx % self.length
        idx = self.ixes[idx].item()
        nd = 10**self.ndigit
        a = idx // nd
        b = idx %  nd

        # calculate the "label" of the addition problem a + b
        c = a + b
        
        # encode the digits of a, b, c into strings
        astr = f'%0{self.ndigit}d' % a
        bstr = f'%0{self.ndigit}d' % b
        cstr = (f'%0{self.ndigit+1}d' % c)[::-1] # reverse c to make addition easier
        
        # convert each character to its token index
        aseq = [int(a) for a in astr] 
        bseq = [int(b) for b in bstr]
        cseq = [int(c) for c in cstr]
        if self.format_vocab is not None:
            plus = self.format_vocab['+']
            equal = self.format_vocab['=']
            dix = aseq + [plus] + bseq + [equal] + cseq
        else:
            dix = aseq + bseq + cseq

        # x will be input to GPT and y will be the associated expected outputs
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long) # predict the next token in the sequence
        
        # we will only train in the output locations. -1 will mask loss to zero
        if self.format_vocab is not None:
            y[:self.ndigit*2+1] = -1
        else:
            y[:self.ndigit*2-1] = -1

        # create the input and output sequences
        if with_raw:
            return x, y, idx, a, b, c
        return x, y, idx

class AdditionTokenizer():
    """
    This class is used to convert the input and output sequences into
    the appropriate format for the model. It is a simple wrapper around
    the AdditionDataset class.
    """

    def __init__(self, ndigit=2, format_vocab=None):
        self.ndigit = ndigit
        self.format_vocab = format_vocab
        self.pad_token_id = -1
        if format_vocab is None:
            self.vocab_size = 10                        # digits 0..9
        else:
            self.vocab_size = 10 + len(format_vocab)    # digits 0..9, +, =

    def decode(self, d1d2, d1d2d3):
        factors = torch.tensor([[10**i for i in range(self.ndigit+1)][::-1]]).to(d1d2.device)

        # isolate the last digit of the sampled sequence
        d3 = d1d2d3[:, -(self.ndigit+1):]
        d3 = d3.flip(1) # reverse the digits to their "normal" order
        d3i_pred = (d3 * factors).sum(1)
        
        # decode the integers from individual digits
        if self.format_vocab is None:
            d1i = (d1d2[:,:self.ndigit] * factors[:,1:]).sum(1)
            d2i = (d1d2[:,self.ndigit:self.ndigit*2] * factors[:,1:]).sum(1)
        else:
            d1i = (d1d2[:,:self.ndigit] * factors[:,1:]).sum(1)
            d2i = (d1d2[:,self.ndigit+1:self.ndigit*2+1] * factors[:,1:]).sum(1)
        d3i_gt = d1i + d2i # manually calculate the ground truth
        
        # evaluate the correctness of the results in this batch
        correct = (d3i_pred == d3i_gt)
        return correct
    
if __name__ == "__main__":
    format_vocab={'+': 10, '=': 11}
    dataset = AdditionDataset(ndigit=2, split='train', format_vocab=format_vocab)
    x, y, _, a, b, c = dataset.__getitem__(222, with_raw=True)
    print(f'{a} + {b} = {c}')
    for xi, yi in zip(x, y):
        print(f'{xi} -> {yi}')
    