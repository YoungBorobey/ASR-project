from typing import List, NamedTuple

import torch

from .char_text_encoder import CharTextEncoder
from collections import defaultdict
from tqdm import tqdm
class Hypothesis(NamedTuple):
    text: str
    prob: float


class CTCCharTextEncoder(CharTextEncoder):
    EMPTY_TOK = "^"

    def __init__(self, alphabet: List[str] = None):
        super().__init__(alphabet)
        vocab = [self.EMPTY_TOK] + list(self.alphabet)
        self.ind2char = dict(enumerate(vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}

    def ctc_decode(self, inds: List[int]) -> str:
        # TODO: your code here
        merged_symbols = []
        last_symbol = 51
        for i in inds:
            curr_symb = self.ind2char[i]
            
            #check if valid 
            if curr_symb != last_symbol and curr_symb != self.EMPTY_TOK:
                merged_symbols.append(curr_symb)
            last_symbol = curr_symb
        
        return ''.join(merged_symbols)
    

    def ctc_beam_search(self, probs: torch.tensor, probs_length,
                        beam_size: int = 100) -> List[Hypothesis]:
        """
        Performs beam search and returns a list of pairs (hypothesis, hypothesis probability).
        """
        assert len(probs.shape) == 2
        char_length, voc_size = probs.shape
        assert voc_size == len(self.ind2char)
        hypos: List[Hypothesis] = []
        

        # TODO: your code here
        #a bit changed seminar code 

        dp = {('', self.EMPTY_TOK) : 1} #key = (text, last_char), values : prob
        for i in tqdm(probs):
            dp = self.extend_and_merge(dp, i)
            dp = self.cut_beams(dp, beam_size)
        result = [Hypothesis(i[0].strip(), dp[i].item()) for i in dp]
        return result

    def extend_and_merge(self, dp, probs):
        new_dp = defaultdict(float)

        for (text, last_char), text_prob in dp.items():
            for i in range(len(probs)):
                curr_char = self.ind2char[i]
                if curr_char == last_char:
                    new_dp[(text, last_char)] += text_prob * probs[i]
                elif curr_char == self.EMPTY_TOK:
                    new_dp[(text, curr_char)] += text_prob * probs[i]
                else:
                    new_dp[((text + curr_char), curr_char)] += text_prob * probs[i]

        return new_dp

    def cut_beams(self, dp, beam_size):
        return dict(list(sorted(dp.items(), key=lambda x: x[1].item(), reverse=True))[:beam_size])

