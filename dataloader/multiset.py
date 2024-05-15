from .speechset import *
from ._multiloader import *

__all__ = [
    "CremaTwoModal",
    "RavdessTwoModal",
    "SaveeTwoModal",
    "TessTwoModal",
    "DaicTwoModal"
]


class CremaTwoModal(MultiLoader):
    def __init__(
        self, 
        **kwargs
    ) -> None:
        """
        Crowd-sourced Emotional Mutimodal Actors Dataset (Crema-D) with bi-modality (audio & text).
        """

        cr = Crema(**kwargs)

        super().__init__(
            audio_data=cr
        )


class RavdessTwoModal(MultiLoader):
    def __init__(
        self, 
        **kwargs
    ) -> None:
        """
        Ryerson Audio-Visual Database of Emotional Speech and Song (Ravdess) with bi-modality (audio & text).
        """

        rv = Ravdess(**kwargs)

        super().__init__(
            audio_data=rv
        )


class SaveeTwoModal(MultiLoader):
    def __init__(
        self, 
        **kwargs
    ) -> None:
        """
        Surrey Audio-Visual Expressed Emotion (Savee) with bi-modality (audio & text).
        """

        sv = Savee(**kwargs)

        super().__init__(
            audio_data=sv
        )


class TessTwoModal(MultiLoader):
    def __init__(
        self, 
        **kwargs
    ) -> None:
        """
        Toronto emotional speech set (Tess) with bi-modality (audio & text).
        """

        te = Tess(**kwargs)

        super().__init__(
            audio_data=te
        )


class DaicTwoModal(MultiLoader):
    def __init__(
        self, 
        **kwargs
    ) -> None:
        """
        Extended Distress Analysis Interview Corpus (E-DAIC) with bi-modality (audio & text).
        """

        da = Daic(**kwargs)

        super().__init__(
            audio_data=da
        )