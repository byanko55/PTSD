from ops.misc import *

from torch import Generator
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn import decomposition


def split_dataset(dataset:Dataset, train_ratio:float = 0.7, test_ratio:float = 0.2) -> Tuple[Any, Any, Any]:
    """
    Split whole dataset into train/eval/test sets.

    Args:
        dataset (Dataset): dataset from which to load the data.
        train_ratio (float, optional): if train_ratio == 1, then it only generate an train batch.
        test_ratio (float, optional): if test_ratio == 0, then it does not generate an test batch.

    Returns:
        train_ds, eval_ds, test_ds (Subset): Randomly split subsets.
    """

    if train_ratio + test_ratio > 1.0:
        raise ValueError(" \
            the sum of train_ratio and test_ratio can't be larger than 1 \
        ")

    num_samples = len(dataset)
    train_samples = int(train_ratio * num_samples)
    test_samples = int(test_ratio * num_samples)
    eval_samples = num_samples - train_samples - test_samples

    # No test
    if test_samples == 0:
        # train only
        if eval_samples == 0:
            return dataset, None, None
        else :
            train_ds, eval_ds = random_split(dataset, 
                [train_samples, eval_samples],
                generator=Generator().manual_seed(42)
            )

            return train_ds, eval_ds, None
    else :
        # No dev set
        if eval_samples == 0 :
            train_ds, test_ds = random_split(dataset, 
                [train_samples, test_samples],
                generator=Generator().manual_seed(42)
            )

            return train_ds, None, test_ds
        else :
            train_ds, eval_ds, test_ds = random_split(dataset, 
                [train_samples, eval_samples, test_samples],
                generator=Generator().manual_seed(42)
            )

            return train_ds, eval_ds, test_ds
        
def build_batches(dataset:Dataset, batch_size:int = 128, train_ratio:float = 0.7, test_ratio:float = 0.2) -> dict:
    """
    Build a training/eval/test batch.

    Args:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load.
        train_ratio (float, optional): if train_ratio == 1, then it only generate an train batch.
        test_ratio (float, optional): if test_ratio == 0, then it does not generate an test batch.
        
    Returns:
        batch_loader (dict): collection of train/eval/test batches.
    """

    train_ds, eval_ds, test_ds = split_dataset(dataset, train_ratio, test_ratio)

    train_batch = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    eval_batch = DataLoader(eval_ds, batch_size=batch_size, shuffle=True) if eval_ds != None else None
    test_batch = DataLoader(test_ds, batch_size=batch_size, shuffle=True) if test_ds != None else None

    batch_loader = {
        'train': train_batch,
        'eval': eval_batch,
        'test': test_batch
    }

    return batch_loader


def make_clusters(dataset:Dataset, sample_size:int = 500, method='t-sne') -> None:
    if method not in ['t-sne', 'pca']:
        raise ValueError(" \
                [make_clusters] decomposition method should be one among \
                \{\'t-sne\', \'pca\'\}"
            )

    pop_size = len(dataset)
    indices = np.random.choice(pop_size, size=min(sample_size, pop_size), replace=False)

    x, y = dataset[indices]
    x_ = np.reshape(x, (len(x), -1))

    print("Found %d objects with %d class types"%(len(x), len(np.unique(y))))

    if method == 't-sne':
        tsne = TSNE(n_components=2, random_state=1)
        x_c = tsne.fit_transform(x_)
    else :
        pca = decomposition.PCA(n_components=2)
        pca.fit(x_)
        x_c = pca.transform(x_)

    # t-SNE Visualization
    _, ax = plt.subplots()
    scatter = ax.scatter(x_c[:, 0], x_c[:, 1], c=y)

    # produce a legend with the unique colors from the scatter
    legend = ax.legend(*scatter.legend_elements(), loc="lower left", title="Classes")
    ax.add_artist(legend)

    if method == 't-sne':
        ax.set_title("T-sne clusters")
    else :
        ax.set_title("PCA clusters")

    plt.show()

