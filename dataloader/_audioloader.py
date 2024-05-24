from ops.misc import *
from _dataloader import CustomLoader

import random
import librosa
import warnings
import matplotlib.pyplot as plt

from scipy import fftpack
from IPython.display import Audio

warnings.filterwarnings("ignore")


FEATURE_MEL = 0
FEATURE_MFCC = 1
FEATURE_LFCC = 2
FEATURE_ALL = 3


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


def extract_lfcc(filename:str, n_fft:int=2048, n_lfcc:int=20, compress:bool = False) -> np.ndarray:
    """
    Extract a Linear-frequency cepstral coefficients (LFCCs).

    Args:
        filename (str): wav file name.
        n_fft (int): length of the windowed signal after padding with zeros.
        n_lfcc (int): number of LFCCs to return.
        compress (bool): if True, calculate the mean of each channel.
    Returns:
        out (np.ndarray) : LFCC sequence.
    """
    data, sampling_rate = librosa.load(filename)

    S = np.abs(librosa.stft(y = data, n_fft=n_fft, pad_mode='reflect'))**2

    fmin, fmax = 0.0, float(sampling_rate) / 2
    n_filter = 128
    weights = np.zeros((n_filter, int(1 + n_fft // 2)), dtype=np.float32)

    fftfreqs = librosa.fft_frequencies(sr=sampling_rate, n_fft=n_fft)

    linear_f = np.linspace(fmin, fmax, n_filter + 2)

    fdiff = np.diff(linear_f)
    ramps = np.subtract.outer(linear_f, fftfreqs)

    for i in range(n_filter):
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]

        weights[i] = np.maximum(0, np.minimum(lower, upper))

    linear_spec = np.dot(weights, S)

    S = librosa.power_to_db(linear_spec)

    out = (fftpack.dct(S, axis=0, norm='ortho')[:n_lfcc]).T

    if compress:
        out = np.mean(out, axis=0)
        return out
    
    return out


def extract_full(filename:str, n_channels:int = 20) -> np.ndarray:
    data, sampling_rate = librosa.load(filename)

    mel = extract_mel(filename, n_channels).T
    mfcc = extract_mfcc(filename, n_channels).T
    lfcc = extract_lfcc(filename, n_lfcc=n_channels).T
    zcr = librosa.feature.zero_crossing_rate(y=data)

    ch = librosa.feature.chroma_stft(y=data, sr=sampling_rate)
    qch = librosa.feature.chroma_cqt(y=data, sr=sampling_rate)
    cch = librosa.feature.chroma_cens(y=data, sr=sampling_rate)
    vch = librosa.feature.chroma_vqt(y=data, sr=sampling_rate, intervals='ji5')

    ton = librosa.feature.tonnetz(y=data, sr=sampling_rate)
    pol = librosa.feature.poly_features(y=data, sr=sampling_rate)
    spec_rol = librosa.feature.spectral_rolloff(y=data, sr=sampling_rate)
    spec_fla = librosa.feature.spectral_flatness(y=data)
    spec_con = librosa.feature.spectral_contrast(y=data, sr=sampling_rate)
    spec_ban = librosa.feature.spectral_bandwidth(y=data, sr=sampling_rate)
    spec_cen = librosa.feature.spectral_centroid(y=data, sr=sampling_rate)

    out = np.concatenate((
        mel, 
        mfcc, 
        lfcc, 
        zcr, 
        ch,
        qch, 
        cch, 
        vch, 
        ton, 
        pol, 
        spec_rol, 
        spec_fla, 
        spec_con, 
        spec_ban, 
        spec_cen
    ))
    
    return out.T


class AudioLoader(CustomLoader):
    def __init__(
        self,
        audio_paths:List[str],
        targets:List[Any],
        classes:List[str],
        dataset_name:str,
        mode:int = FEATURE_MEL,
        trim:int = 128,
        n_channels:int = 64,
        sampling_rate:float = 1.0,
        shuffle:bool = True,
        compress:bool = False
    ) -> None:
        """
        Audio data I/O.

        Args:
            audio_paths (list): audio file paths.
            targets (list): audio labels.
            dataset_name (str): explicit name given by users.
            mode (int): define what type of acoustic feature to be extracted.
                1) mode = 0: Mel-scaled spectrogram.
                2) mode = 1: Mel-frequency cepstral coefficients.
                3) mode = 2: Linear-frequency cepstral coefficients.
                4) mode = 3: Full Features obtained from the librosa library.
            trim (int) : maximum number of time window.
            n_channels (int) : number of channels composed of a spectrum.
            sampling_rate (float): define the ratio to draw samples from the dataset.
            shuffle (bool): set to 'True' to have the data reshuffled.
            compress (bool): if True, calculate the mean of each channel.
        """

        if mode not in [FEATURE_MEL, FEATURE_MFCC, FEATURE_LFCC, FEATURE_ALL]:
            raise ValueError(
                "[AudioLoader: extract] Extraction mode should be one among \
                \{\'mel\', \'mel-2d\', \'mfcc\', \'mfcc-2d\', \'lfcc\', \'lfcc-2d\'\}"
            )

        print("---------- Loading %s... ----------\n  ┕ #Original audios = %d \
              "%(dataset_name, len(audio_paths)))
        
        targets = torch.LongTensor(targets)
        audio_paths = np.array(audio_paths)
        
        # Randomly choose sample data
        if sampling_rate < 0 or sampling_rate > 1.0:
            raise ValueError(" \
                Samples can't be negative larger than its original population \
            ")
        
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
        
        self.data = []
        self.num_audios = len(targets)

        for i, wavfile in enumerate(audio_paths):
            if mode == FEATURE_MEL:
                feature = extract_mel(wavfile, n_mels=n_channels, compress=compress)
            elif mode == FEATURE_MFCC:
                feature = extract_mfcc(wavfile, n_mfcc=n_channels, compress=compress)
            elif mode == FEATURE_LFCC:
                feature = extract_lfcc(wavfile, n_lfcc=n_channels, compress=compress)
            else:
                feature = extract_full(wavfile)
                compress = False

            # fix the input size as [trim, n_mels/n_mfcc/n_lfcc]
            if not compress :
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
        Plot the loudness of the audio at a given time.

        Args:
            item (int): index of sampled audio file.
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
        Plot a representation of frequencies changing.
        with respect to time for given audio/music signals.
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
            (Audio) : audio object.
        """

        ridx = random.randrange(0, self.num_audios)
        return Audio(self.audio_paths[ridx])
    
    def noise(self, x:np.ndarray) -> np.ndarray:
        """
        Data Augmentation 1: noise injection.

        Args:
            x (np.ndarray) : audio time series.

        Returns:
            (np.ndarray) : noised audio data.
        """

        noise_amp = 0.035*np.random.uniform()*np.amax(x)
        x = x + noise_amp*np.random.normal(size=x.shape[0])
        return x

    def stretch(self, x:np.ndarray, rate:float = 0.8) -> np.ndarray:
        """
        Data Augmentation 2: strectch the audio file length.

        Args:
            x (np.ndarray) : audio time series.
            rate (float, Optional) : Stretch factor.  
                If ``rate > 1``, then the signal is sped up.
                If ``rate < 1``, then the signal is slowed down.
        
        Returns:
            (np.ndarray) : audio time series stretched by the specified rate.
        """

        return librosa.effects.time_stretch(x, rate)

    def shift(self, x:np.ndarray) -> np.ndarray:
        """
        Data Augmentation 3: Time shift.

        Args:
            x (np.ndarray) : audio time series.

        Returns:
            (np.ndarray) : output audio time-series, with the same shape as `x`.
        """

        shift_range = int(np.random.uniform(low=-5, high = 5)*1000)
        return np.roll(x, shift_range)

    def pitch(self, x:np.ndarray, sampling_rate:float, pitch_factor:float = 0.7) -> np.ndarray:
        """
        Data Augmentation 4: change pitch.

        Args:
            x (np.ndarray) : audio time series.
            sampling_rate (float) : audio sampling rate.
            pitch_factor (float, Optional) : how many (fractional) steps to shift.
        
        Returns:
            (np.ndarray) : the pitch-shifted audio time-series.
        """

        return librosa.effects.pitch_shift(x, sampling_rate, pitch_factor)

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