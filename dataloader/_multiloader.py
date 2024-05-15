from ._audioloader import *
from .stt import Wav2Vec

from transformers import BertTokenizer, BertModel


class MultiLoader():
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

        self.data = {'mod_audio': audio_data.data, 'mod_text': x_txt}
        self.targets = audio_data.targets

        print("  ┕ Final result:", 
            self.data['mod_audio'].shape,
            self.data['mod_text'].shape,
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
        
        audios, texts = self.data['mod_audio'][item], self.data['mod_text'][item]
        labels = self.targets[item]
        
        return audios, texts, labels