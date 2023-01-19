from typing import List, NamedTuple, Union

import torch
import numpy as np
import json

from string import ascii_lowercase
from .char_text_encoder import CharTextEncoder
from collections import defaultdict
from tqdm import tqdm
from hw_asr.utils import ROOT_PATH
from pathlib import Path
import os
import torchaudio
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors
import shutil
from torch import tensor as Tensor
from speechbrain.utils.data_utils import download_file
os.environ["TOKENIZERS_PARALLELISM"] = "True"
URL_LINKS = {
    "dev-clean": "https://www.openslr.org/resources/12/dev-clean.tar.gz",
    "dev-other": "https://www.openslr.org/resources/12/dev-other.tar.gz",
    "test-clean": "https://www.openslr.org/resources/12/test-clean.tar.gz",
    "test-other": "https://www.openslr.org/resources/12/test-other.tar.gz",
    "train-clean-100": "https://www.openslr.org/resources/12/train-clean-100.tar.gz",
    "train-clean-360": "https://www.openslr.org/resources/12/train-clean-360.tar.gz",
    "train-other-500": "https://www.openslr.org/resources/12/train-other-500.tar.gz",
}


class Hypothesis(NamedTuple):
    text: str
    prob: float


class CTC_BPE_TextEncoder(CharTextEncoder):
    EMPTY_TOK = "^"

    def __init__(self, data, alphabet: List[str] = None, vocab_size=50, path=None, *args, **kwargs):
        super().__init__(alphabet)

        self._data_dir = ROOT_PATH / "data" / "datasets" / "librispeech"
        self.alphabet = list(ascii_lowercase)
        if path is not None:
            tokenizer = Tokenizer.from_file(path)
        else:
            text_index = []
            for dataset in data['train']['datasets']:
                part = dataset['args']['part']
                text_index.extend([i["text"] for i in self._get_or_load_index(part)])

            tmp_path = '/tmp/texts_libspeech_bpe.txt'
            with open(tmp_path, 'w') as f:
                for line in text_index:
                    f.write(line)
        
            print('train bpe')
            tokenizer = self.train_bpe(tmp_path, vocab_size)
            tokenizer.save(str(ROOT_PATH) + f'/tokenizer_BPE_{vocab_size}.json', pretty=True)
            os.remove(tmp_path)


        self.tokenizer = tokenizer

    def encode(self, text) -> Tensor:
        text = self.normalize_text(text)
        return Tensor(self.tokenizer.encode(text.lower()).ids).unsqueeze(0)

    def decode(self, vector: Union[Tensor, np.ndarray, List[int]]):
        return self.tokenizer.decode(vector).strip()
    def ctc_decode(self, inds: List[int]) -> str:
        # TODO: your code here
        merged_ids = []
        last = -1
        for i in inds:
            if i != last:
                merged_ids.append(i)
            last = i
        return self.decode(merged_ids)
    def __len__(self):
        return self.tokenizer.get_vocab_size()

    def __getitem__(self, item: int):
        return self.tokenizer.decode([item])
        
    def train_bpe(self, file_path, vocab_size):
        #init tokenizer
        special_tokens = [self.EMPTY_TOK]
        tokenizer = Tokenizer(models.BPE())
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
        tokenizer.decoder = decoders.ByteLevel()

        #init trainer 
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=2,
            initial_alphabet=self.alphabet,
            special_tokens=special_tokens
        )
        tokenizer.train([file_path], trainer=trainer)
        return tokenizer


    def ctc_beam_search(self, probs: torch.tensor, probs_length,
                        beam_size: int = 100) -> List[Hypothesis]:
        """
        Performs beam search and returns a list of pairs (hypothesis, hypothesis probability).
        """
        assert len(probs.shape) == 2
        char_length, voc_size = probs.shape
        assert voc_size == len(self.tokenizer.get_vocab_size())
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
                curr_char = self.__getitem__(i)
                if curr_char == last_char:
                    new_dp[(text, last_char)] += text_prob * probs[i]
                elif curr_char == self.EMPTY_TOK:
                    new_dp[(text, curr_char)] += text_prob * probs[i]
                else:
                    new_dp[((text + curr_char), curr_char)] += text_prob * probs[i]

        return new_dp

    def cut_beams(self, dp, beam_size):
        return dict(list(sorted(dp.items(), key=lambda x: x[1].item(), reverse=True))[:beam_size])

    
    
    # code below is part of a code of librispeech_dataset.py. We want do load all files and get texts from it
    def _get_or_load_index(self, part):
        index_path = self._data_dir / f"{part}_index.json"
        if index_path.exists():
            with index_path.open() as f:
                index = json.load(f)
        else:
            index = self._create_index(part)
            with index_path.open("w") as f:
                json.dump(index, f, indent=2)
        return index

    def _create_index(self, part):
        index = []
        split_dir = self._data_dir / part
        if not split_dir.exists():
            self._load_part(part)

        flac_dirs = set()
        for dirpath, dirnames, filenames in os.walk(str(split_dir)):
            if any([f.endswith(".flac") for f in filenames]):
                flac_dirs.add(dirpath)
        for flac_dir in tqdm(
                list(flac_dirs), desc=f"Preparing librispeech folders: {part}"
        ):
            flac_dir = Path(flac_dir)
            trans_path = list(flac_dir.glob("*.trans.txt"))[0]
            with trans_path.open() as f:
                for line in f:
                    f_id = line.split()[0]
                    f_text = " ".join(line.split()[1:]).strip()
                    flac_path = flac_dir / f"{f_id}.flac"
                    t_info = torchaudio.info(str(flac_path))
                    length = t_info.num_frames / t_info.sample_rate
                    index.append(
                        {
                            "path": str(flac_path.absolute().resolve()),
                            "text": f_text.lower(),
                            "audio_len": length,
                        }
                    )
        return index

    def _load_part(self, part):
        arch_path = self._data_dir / f"{part}.tar.gz"
        print(f"Loading part {part}")
        download_file(URL_LINKS[part], arch_path)
        shutil.unpack_archive(arch_path, self._data_dir)
        for fpath in (self._data_dir / "LibriSpeech").iterdir():
            shutil.move(str(fpath), str(self._data_dir / fpath.name))
        os.remove(str(arch_path))
        shutil.rmtree(str(self._data_dir / "LibriSpeech"))
