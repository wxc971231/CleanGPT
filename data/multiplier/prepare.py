import torch
from torch.utils.data import Dataset

class MultiplicationDataset(Dataset):
    """
    Creates n-digit multiplication problems. For example, if n=2:
    85 * 50 = 4250 → encoded as "85500524" (reversed result)
    Or formatted:  "85*50=0524" (symbols added, reversed result)

    Reverse result to make learning easier. Pad result to 2n digits.

    If format_vocab is provided, includes 'x' and '=' symbols.
    """

    def __init__(self, ndigit=2, split='train', format_vocab=None, seed=42):
        self.ndigit = ndigit
        self.split = split
        self.format_vocab = format_vocab
        if format_vocab is not None:
            assert 'x' in format_vocab and '=' in format_vocab
            assert format_vocab['x'] not in range(10)
            assert format_vocab['='] not in range(10)

        num_total = (10 ** ndigit) ** 2
        rng = torch.Generator()
        rng.manual_seed(seed)
        perm = torch.randperm(num_total, generator=rng)

        num_test = int(num_total * 0.15)
        num_val = int(num_total * 0.15)

        if split == 'train':
            self.ixes = perm[num_test + num_val:]
        elif split == 'val':
            self.ixes = perm[num_test:num_test + num_val]
        elif split == 'test':
            self.ixes = perm[:num_test]
        else:
            raise ValueError(f"Unknown split: {split}")

        self.length = len(self.ixes)

    def __len__(self):
        return self.length

    def __getitem__(self, idx, with_raw=False):
        idx = idx % self.length
        nd = 10 ** self.ndigit
        ab_idx = self.ixes[idx].item()
        a = ab_idx // nd
        b = ab_idx % nd
        c = a * b

        # pad inputs
        astr = f'%0{self.ndigit}d' % a
        bstr = f'%0{self.ndigit}d' % b
        cstr = (f'%0{2 * self.ndigit}d' % c)[::-1]  # reverse the output digits

        aseq = [int(ch) for ch in astr]
        bseq = [int(ch) for ch in bstr]
        cseq = [int(ch) for ch in cstr]

        if self.format_vocab is not None:
            star = self.format_vocab['x']
            equal = self.format_vocab['=']
            dix = aseq + [star] + bseq + [equal] + cseq
        else:
            dix = aseq + bseq + cseq

        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)

        # loss masking：不训练输入部分
        if self.format_vocab is not None:
            y[:self.ndigit * 2 + 1] = -1  # includes 'x' and '='
        else:
            y[:self.ndigit * 2 - 1] = -1

        if with_raw:
            return x, y, idx, a, b, c
        return x, y, idx

class MultiplicationTokenizer():
    """
    Converts digit-level sequences to integer results, and evaluates correctness.
    """

    def __init__(self, ndigit=2, format_vocab=None):
        self.ndigit = ndigit
        self.format_vocab = format_vocab
        self.pad_token_id = -1
        if format_vocab is None:
            self.vocab_size = 10
        else:
            self.vocab_size = 10 + len(format_vocab)

    def decode(self, d1d2, d1d2d3):
        n = self.ndigit
        factors_n = torch.tensor([[10 ** i for i in reversed(range(n))]], device=d1d2.device)
        factors_2n = torch.tensor([[10 ** i for i in reversed(range(2*n))]], device=d1d2.device)

        # isolate the last digit of the sampled sequence
        d3 = d1d2d3[:, -2*n:]
        d3 = d3.flip(1)         # reverse the digits to their "normal" order
        d3i_pred = (d3 * factors_2n).sum(1)

        # decode the integers from individual digits
        if self.format_vocab is None:
            d1i = (d1d2[:, :n] * factors_n).sum(1)
            d2i = (d1d2[:, n:2*n] * factors_n).sum(1)
        else:
            d1i = (d1d2[:, :n] * factors_n).sum(1)
            d2i = (d1d2[:, n+1:2*n+1] * factors_n).sum(1)
        d3i_gt = d1i * d2i # manually calculate the ground truth

        # evaluate the correctness of the results in this batch
        correct = (d3i_pred == d3i_gt)
        return correct

if __name__ == "__main__":
    format_vocab = {'x': 10, '=': 11}
    dataset = MultiplicationDataset(ndigit=3, split='train', format_vocab=format_vocab)
    x, y, _, a, b, c = dataset.__getitem__(888, with_raw=True)
    print(f'{a} * {b} = {c}')
    for xi, yi in zip(x, y):
        print(f'{xi:>2} -> {yi:>2}')
