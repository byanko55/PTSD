from ops.misc import *

import random
import librosa
import matplotlib.pyplot as plt

from IPython.display import Audio


def extract_mel(filename:str, n_mels:int = 128, compress:bool = False) -> np.ndarray:
    """
    Extract a mel-scaled spectrogram.

    Args:
        filename (str): wav file name.
        n_mels (int) : number of Mel bands to generate.
        compress (bool) : if True, calculate the mean of each channel.
    """

    data, sampling_rate = librosa.load(filename)
    out = librosa.feature.melspectrogram(y=data, sr=sampling_rate, n_mels=n_mels).T

    if compress:
        out = np.mean(out, axis=0)
        return out
    
    return out


def extract_mfcc(filename:str, n_mfcc:int = 20, compress:bool = False) -> np.ndarray:
    """
    Extract a Mel-frequency cepstral coefficients (MFCCs).

    Args:
        filename (str): wav file name.
        n_mfcc (int): number of MFCCs to return.
        compress (bool): if True, calculate the mean of each channel.
    """

    data, sampling_rate = librosa.load(filename)
    out = librosa.feature.mfcc(y=data, sr=sampling_rate, n_mfcc=n_mfcc).T

    if compress:
        out = np.mean(out, axis=0)
        return out
    
    return out


class AudioLoader():
    def __init__(
        self,
        audio_paths:List[str],
        targets:List[Any],
        classes:List[str],
        dataset_name:str,
        mode:str = 'mel',
        trim:int = 128,
        n_channels:int = 64,
        sampling_rate:float = 1.0,
        shuffle:bool = True
    ) -> None:
        """
        Audio data I/O.

        Args:
            audio_paths (list): audio file paths.
            targets (list): audio labels.
            dataset_name (str): explicit name given by users.
            mode (str): define what type of acoustic feature to be extracted.
                1) mel: mean of mel-scaled spectrogram
                2) mel-2d: mel-scaled spectrogram
                3) mfcc: mean of Mel-frequency cepstral coefficients
                4) mfcc-2d: Mel-frequency cepstral coefficients
            trim (int) : maximum number of time window.
            n_channels (int) : number of channels composed of a spectrum.
            sampling_rate (float): define the ratio to draw samples from the dataset.
            shuffle (bool): set to 'True' to have the data reshuffled.
        """

        if mode not in ['mel', 'mfcc', 'mel-2d', 'mfcc-2d']:
            raise ValueError(
                "[AudioLoader: extract] Extraction mode should be one among \
                \{\'mel\', \'mel-2d\', \'mfcc\', \'mfcc-2d\'\}"
            )

        print("---------- Loading %s... ----------\n  ┕ #Original audios = %d \
              "%(dataset_name, len(audio_paths)))
        
        targets = torch.LongTensor(targets)
        audio_paths = np.array(audio_paths)
        
        # Randomly choose sample data
        if sampling_rate < 0 or sampling_rate > 1.0:
            raise ValueError("Samples can't be negative larger than its original population")
        
        if sampling_rate != 1.0:
            num_raws = len(targets)

            num_samples = int(sampling_rate * num_raws)
            indices = torch.tensor(random.sample(range(num_raws), num_samples))

            audio_paths = audio_paths[indices]
            targets = targets[indices]

            print("  ┕ Sampled by (%.1f)%% => #audios = %d \
                  "%(100*sampling_rate, num_samples))

        if shuffle:
            num_samples = len(targets)
            indices = torch.randperm(num_samples)
            audio_paths = audio_paths[indices]
            targets = targets[indices]
        
        # if True, calculate the mean of each channel
        compress_mode = (mode[-2:] != '2d' and mode[:4] != 'lfcc')
        self.data = []
        self.num_audios = len(targets)

        for i, wavfile in enumerate(audio_paths):
            if mode[:3] == 'mel' :
                feature = extract_mel(wavfile, n_mels=n_channels, compress=compress_mode)
            else :
                feature = extract_mfcc(wavfile, n_mfcc=n_channels, compress=compress_mode)

            # fix the input size as [trim, n_mels/n_mfcc]
            if not compress_mode :
                d = feature.shape[0]

                if d < trim: # add padding
                    padding = np.zeros((trim - d, feature.shape[1]))
                    feature = np.concatenate((feature, padding), axis=0)
                elif d > trim: # trimming
                    feature = np.delete(feature, range(trim, d), 0)

            # add new channels (1d -> 2d, 2d -> 3d)
            # ex) [200, 128] -> [1, 200, 128]
            #feature = feature/(np.max(feature) - np.min(feature) + 0.5) # normalize
            feature = np.reshape(feature, (-1, *(feature.shape)))
            self.data.append(feature)

            print("  ┕ Extracting acoustic features... : %d/%d \
                  "%(i+1, self.num_audios), end="\r")

        print(np.array(self.data).shape)

        self.audio_paths = audio_paths
        self.data = torch.FloatTensor(np.array(self.data))
        self.targets = targets
        self.classes = classes
        self.shape = self.data.shape

        print("  ┕ Final result:", 
              self.shape, 
              "\n  Done.\n%s"%('-'*(33+len(dataset_name))))
        
    def waveplot(self, item:int=-1) -> None:
        """
        Plot the loudness of the audio at a given time

        Args:
            item (int): index of sampled audio file
        """

        if item == -1: # Randomly pick up one sample
            item = random.randrange(0, self.num_audios)
        
        data, sampling_rate = librosa.load(self.audio_paths[item])

        plt.figure(figsize=(12, 3))
        librosa.display.waveshow(data, sr=sampling_rate)
        plt.tight_layout()
        plt.show()

    def spectrogram(self) -> None:
        """
        Plot a representation of frequencies changing 
        with respect to time for given audio/music signals
        """

        ridx = random.randrange(0, self.num_audios)
        data, sampling_rate = librosa.load(self.audio_paths[ridx])

        X = librosa.stft(data)
        Xdb = librosa.amplitude_to_db(abs(X))
        plt.figure(figsize=(12, 3))
        librosa.display.specshow(Xdb, sr=sampling_rate, x_axis='time', y_axis='hz')   
        plt.colorbar()

    def sampleplay(self) -> Audio:
        """
        Create an audio object.
        It will result in Audio controls being displayed in the frontend (only works in the notebook).
        
        Retruns:
            (Audio) : audio object
        """

        ridx = random.randrange(0, self.num_audios)
        return Audio(self.audio_paths[ridx])

    def __len__(self) -> int:
        """
        Number of samples in a dataset.
        """

        return self.num_audios

    def __getitem__(self, item:int) -> Tuple[Any, Any]:
        """
        Return a stream of data reading from a dataset.

        Args:
            item (int): indices of sampled data.

        Returns:
            (audios, target): where target is index of the target class.
        """

        audios, labels = self.data[item], self.targets[item]
        
        return audios, labels