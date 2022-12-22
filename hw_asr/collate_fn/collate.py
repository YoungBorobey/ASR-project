import logging
from typing import List
from torch.nn.utils.rnn import pad_sequence
import torch
logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """
    # TODO: your code here
    #fields: audio, spectrogram, duration, text, text_encoded, audio_path
    
    spectrogram = []
    text_encoded = []
    text_encoded_length = []
    text = []
    spectrogram_length = []
    audio_path = []
    audio_records = []

    for i in dataset_items:
        spectrogram.append(i['spectrogram'].squeeze(0).transpose(0, 1))
        text_encoded.append(i['text_encoded'].squeeze(0))
        text_encoded_length.append(i['text_encoded'].shape[1])
        text.append(i['text'])
        spectrogram_length.append(i['spectrogram'].shape[2])
        audio_path.append(i['audio_path'])
        audio_records.append(i['audio'])

    
    spectrogram = pad_sequence(spectrogram, batch_first=True).transpose(1, 2)
    text_encoded = pad_sequence(text_encoded, batch_first=True)
    text_encoded_length = torch.tensor(text_encoded_length)
    spectrogram_length = torch.tensor(spectrogram_length)
    result_batch = {
        'spectrogram' : spectrogram,
        'text_encoded' : text_encoded,
        'text_encoded_length' : text_encoded_length,
        'text' : text,
        'spectrogram_length' : spectrogram_length,
        'audio_path' : audio_path,
        'audio' : audio_records
    }

    return result_batch