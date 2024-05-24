from ops.misc import *


class CustomLoader:
    """
    An abstract class representing a :class:`CustomLoader`.

    All dataloaders that represent a map from keys to data samples should subclass it. 
    All subclasses should overwrite `__getitem__`, supporting fetching a data sample for a given key. 
    Subclasses could also optionally overwrite `__len__`, which is expected to return the size of the dataset.
    """

    def __init__(self) -> None:
        self.name = "__Unknown Loader__"
        self.classes = []

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, item:int) -> Tuple[Any, Any]:
        raise NotImplementedError