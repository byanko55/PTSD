from ._audioloader import *

import torch
import torchaudio
from typing import Tuple

from transformers import BertTokenizer, BertModel

DEFAULT_LABELS = (
    '-', '|', 'E', 'T', 'A', 'O', 'N', 'I', 'H', 'S', 
    'R', 'D', 'L', 'U', 'M', 'W', 'C', 'F', 'G', 'Y', 
    'P', 'B', 'V', 'K', "'", 'X', 'J', 'Q', 'Z'
)


class Wav2Vec(torch.nn.Module):
    def __init__(
        self,
        labels:Tuple[str, ...] = DEFAULT_LABELS,
        blank:int = 0,
    ) -> None:
        """
        Wav2vec 2.0 model ("large" architecture with an extra linear module), 
        pre-trained on 960 hours of unlabeled audio from LibriSpeech dataset.

        Args:
            labels (Tuple[str, ...], optional): The output class labels.
                (only applicable to fine-tuned bundles)
                The first is blank token, and it is customizable.
            blank (int, optional): index of the Blank token. (default: 0 = '-')
        """

        super().__init__()

        self.labels = labels
        self.blank = blank

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        bundle = torchaudio.pipelines.WAV2VEC2_ASR_LARGE_960H
        self.model = bundle.get_model().to(self.device)


    def forward(self, speech_file:str) -> str:
        """
        Given a speech wav file over labels, get the best path string.

        Args:
            speech_file (str): a path-like object or file-like object.

        Returns:
            str: The resulting transcript.
        """

        waveform, _ = torchaudio.load(speech_file, num_frames=819200)
        waveform = waveform.to(self.device)

        emission, _ = self.model(waveform)

        indices = torch.argmax(emission[0], dim=-1)  # [num_seq,]
        indices = torch.unique_consecutive(indices, dim=-1)
        indices = [i for i in indices if i != self.blank]
        return "".join([self.labels[i] for i in indices])
    

class STTLoader():
    def __init__(
        self,
        audio_data:AudioLoader
    ) -> None:
        """
        Dataloader to serve the audio-oriented features and linguistic features,
        obtained by voice recognition model (STT).

        Args:
            audio_data (AudioLoader): collection of acoustic data samples, 
            each of them paired with file path and mel-spectrogram feature.
        """

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_stt = Wav2Vec()

        tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

        model_bert = BertModel.from_pretrained(
            'bert-base-multilingual-cased', 
            output_hidden_states = True
        ).to(self.device)

        self.num_tuples = len(audio_data)
        self.ori_data = audio_data
        self.stt_data = []

        x_txt = []

        for i, wavfile in enumerate(audio_data.audio_paths):
            torch.cuda.empty_cache()

            txt = model_stt(wavfile)
            txt = txt.replace("|", " ")[:-1]

            self.stt_data.append(txt)
            
            toks = tokenizer(txt, return_tensors='pt')
            text_feature = model_bert(toks['input_ids'].to(self.device))
            x_txt.append(text_feature.pooler_output.detach())

            print("  ┕ Extracting linguistic features... : %d/%d \
                  "%(i+1, self.num_tuples), end="\r")

        x_txt = torch.cat(x_txt, dim=0)

        self.data = x_txt
        self.targets = audio_data.targets

        print("  ┕ Final result:", 
            self.data.shape,
            "\n  Done."
        )

    def showSample(self) -> None:
        """
        Print the original text while plotting corresponding wav signal.
        """

        ridx = random.randrange(0, self.num_tuples)

        for s in self.stt_data[ridx].split(' '):
            print(s, '|', end=' ')

        print()
        self.ori_data.waveplot(ridx)

    def __len__(self) -> int:
        """
        Number of samples in a dataset.
        """

        return self.num_tuples

    def __getitem__(self, item:int) -> Tuple[Any, Any]:
        """
        Return a stream of data tuples (containing a sample of each dataset, respectively).

        Args:
            item (int): indices of sampled data tuples.

        Returns:
            (tuples, target): where target is indices of the target class.
        """
        
        texts = self.data[item]
        labels = self.targets[item]
        
        return texts, labels