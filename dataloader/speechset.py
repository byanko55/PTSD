from ._audioloader import *

import os
import pandas as pd
from pydub import AudioSegment


class Crema(AudioLoader):
    def __init__(
        self, 
        data_path:str = "datasets/Crema",
        binary_mode:bool = False,
        **kwargs
    ) -> None:
        """
        Crowd-sourced Emotional Mutimodal Actors Dataset (Crema-D).

        Args:
            data_path (str): directory to be loaded. 
        """

        audio_paths = []; labels = []; label_id = {}

        voice_files = os.listdir(data_path)

        for wavfile in voice_files:
            label = wavfile.split('_')[2]

            if label not in label_id:
                label_id[label] = len(label_id.keys())

            voice_path = data_path + '/' + wavfile
            audio_paths.append(voice_path)

            if binary_mode:
                label_tag = 0 if label == 'NEU' or label == 'HAP' else 1
                labels.append(label_tag)
            else :
                labels.append(label_id[label])

        if binary_mode:
            classes = ['positive', 'negative']
        else :
            classes = sorted(label_id, key=label_id.get)

        super().__init__(
            audio_paths=audio_paths,
            targets=labels,
            classes=classes,
            dataset_name='Crema',
            **kwargs
        )


class Ravdess(AudioLoader):
    def __init__(
        self, 
        data_path:str = "datasets/Ravdess/audio_speech_actors_01-24", 
        **kwargs
    ) -> None:
        """
        Ryerson Audio-Visual Database of Emotional Speech and Song (Ravdess).

        Args:
            data_path (str): directory to be loaded. 
        """

        dir_actors = os.listdir(data_path)
        audio_paths = []; labels = []

        emotions = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fear', 'disgust', 'surprised']

        for dirname in dir_actors:
            actor = os.listdir(data_path + '/' + dirname)
            for wavfile in actor:
                # third part of each filename represents the emotion associated to that file.
                # ex) 03-01-'01'-01-01-01-01.wav
                # label info:
                # 0: neutral, 1: calm, 2: happy, 3: sad, 4:angry, 5:fear, 6:disgust, 7: surprised
                label = int(wavfile.split('-')[2]) - 1
                voice_path = data_path + '/' + dirname + '/' + wavfile
                audio_paths.append(voice_path)
                labels.append(label)

        super().__init__(
            audio_paths=audio_paths,
            targets=labels,
            classes=emotions,
            dataset_name='Ravdess', 
            **kwargs
        )


class Savee(AudioLoader):
    def __init__(
        self, 
        data_path:str = "datasets/Savee", 
        **kwargs
    ) -> None:
        """
        Surrey Audio-Visual Expressed Emotion (Savee).

        Args:
            data_path (str): directory to be loaded. 
        """

        label_id = {'sa':0, 'a':1, 'd':2, 'f':3, 'h':4, 'n':5, 'su':6}

        emotions = ['sad', 'angry', 'disgust', 'fear', 'happy', 'neutral', 'surprised']

        voice_files = os.listdir(data_path)
        audio_paths = []; labels = []

        for wavfile in voice_files:
            label_tag = wavfile.split('_')[1][:-6]
            voice_path = data_path + '/' + wavfile
            audio_paths.append(voice_path)
            labels.append(label_id[label_tag])

        super().__init__(
            audio_paths=audio_paths,
            targets=labels,
            classes=emotions,
            dataset_name='Savee',
            **kwargs
        )


class Tess(AudioLoader):
    def __init__(
        self, 
        data_path:str = "datasets/Tess", 
        **kwargs
    ) -> None:
        """
        Toronto emotional speech set (Tess).

        Args:
            data_path (str): directory to be loaded. 
        """

        label_id = {'sad':0, 'angry':1, 'disgust':2, 'fear':3, 'happy':4, 'neutral':5, 'ps':6}

        emotions = ['sad', 'angry', 'disgust', 'fear', 'happy', 'neutral', 'surprised']

        dir_actors = os.listdir(data_path)
        audio_paths = []; labels = []

        for dirname in dir_actors:
            actor = os.listdir(data_path + '/' + dirname)
            for wavfile in actor:
                # third part of each filename represents the emotion associated to that file.
                # ex) OAF_back_'angry'.wav
                label = wavfile.split('_')[2][:-4]
                voice_path = data_path + '/' + dirname + '/' + wavfile
                audio_paths.append(voice_path)
                labels.append(label_id[label])

        super().__init__(
            audio_paths=audio_paths,
            targets=labels,
            classes=emotions,
            dataset_name='Tess', 
            **kwargs
        )


class Daic(AudioLoader):
    def __init__(
        self,
        root_path:str = "datasets/DAIC", 
        **kwargs
    ) -> None:
        """
        The Extended Distress Analysis Interview Corpus (E-DAIC) (DeVault et al., 2014) is an
        extended version of WOZ-DAIC (Gratch et al.,2014) that contains semi-clinical
        interviews designed to support the diagnosis of psychological distress conditions such
        as anxiety, depression, and post-traumatic stress disorder. These interviews were
        collected as part of a larger effort to create a computer agent that interviews people and
        identifies verbal and nonverbal indicators of mental illnesses.
        
        Args:
            root_path (str): directory to be loaded. 
        """

        # Labeling
        label_path = os.path.join(root_path, 'Detailed_PHQ8_Labels.csv')
        record_label = pd.read_csv(label_path)

        record_label['label'] = pd.Categorical(np.where(record_label['PHQ_8Total'] >= 10, 'ptsd', 'normal'))
        participant_label = {}

        for id, lab in zip(record_label['Participant_ID'], record_label['label']):
            participant_label[id] = lab

        classes = ['normal', 'ptsd']

        # Loading audio data
        data_path = os.path.join(root_path, 'audio')

        if not os.path.exists(data_path): 
            # Trimmed audio files were not ready
            # Require the data build process
            os.mkdir(data_path)
            audio_paths = []; labels = []
            raw_path = os.path.join(root_path, 'records')

            patient_list = [f for f in os.listdir(raw_path) if os.path.isdir(raw_path + '/' + f)]

            for patient in patient_list:
                patient_id = int(patient[:3])

                if patient_id not in participant_label:
                    continue

                dir_path = os.path.join(raw_path, patient, str(patient_id))
                raw_script = pd.read_csv(dir_path + '_Transcript.csv')
                raw_audio = AudioSegment.from_wav(dir_path + "_AUDIO.wav")
                raw_audio = raw_audio.set_channels(1)

                mask = (raw_script['End_Time'] - raw_script['Start_Time'] > 0.5) \
                & (raw_script['End_Time'] - raw_script['Start_Time'] < 25) \
                & (raw_script['Text'].str.count(' ') >= 3)
                
                filtered_script = raw_script[mask]
                label = participant_label[patient_id]
                
                for index, row in filtered_script.iterrows():
                    audio_part = raw_audio[1000*row['Start_Time']: 1000*row['End_Time']]
                    file_path = os.path.join(
                        data_path, 
                        'pID=%d_Lab=%s_rID=%d_St=%d_Et=%d.wav'%(
                            patient_id, 
                            label, 
                            index, 
                            row['Start_Time'],
                            row['End_Time']
                        )
                    )

                    audio_part.export(file_path, format="wav")
                    audio_paths.append(file_path)
                    label_tag = 0 if label == 'normal' else 1
                    labels.append(label_tag)
        else :
            audio_paths = []; labels = []

            file_list = os.listdir(data_path)

            for audio_file in file_list:
                audio_paths.append(data_path + "/" + audio_file)
                label = audio_file.split('_')[1][4:]
                label_tag = 0 if label == 'normal' else 1
                labels.append(label_tag)

        super().__init__(
            audio_paths=audio_paths,
            targets=labels,
            classes=classes,
            dataset_name='DAIC', 
            **kwargs
        )