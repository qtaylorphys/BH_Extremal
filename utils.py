import h5py
import numpy as np
from dateutil.relativedelta import relativedelta as rd
from time import process_time

from numbers import Real
from nptyping import NDArray

def binary_size(num: Real, suffix: str = "B") -> str:
    """
    Return human-readable string from file size in bytes

    Parameters:
    num (Real): the size of the file in bytes
    suffix (str): the suffix to use after the units

    Returns:
    str: a human-readable string for the size of the file
    """
    for unit in ("", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"):
        if abs(num) < 1024.0:
            return f"{num:3.1f} {unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f} Yi{suffix}"

def time_fmt(time_s: Real) -> str:
    """
    Return a human-readable string from time interval in seconds

    Parameters:
    time_s (Real): the time interval in seconds

    Returns:
    str: a human-readable string for the time interval
    """
    intervals = ["days", "hours", "minutes", "seconds"]
    x = rd(seconds=time_s)
    vals = [getattr(x, k) for k in intervals]
    
    if not all(v == 0 for v in vals[:-1]):
        vals = [int(v) for v in vals]
    else:
        vals[-1] = np.round(vals[-1], 1)

    return ' '.join(
        f"{val} {interval}"
        for val, interval in zip(vals, intervals) if val != 0
    )

def save_data_h5(data: NDArray, filename: str, dataset_name: str) -> None:
    """
    Save a numpy array in an H5 file

    Parameters:
    data (NDArray): array to be saved to file
    filename (str): path where the H5 file should be saved
    dataset_name (str): string for the dataset name in the H5 file
    """
    with h5py.File(filename, 'w') as f:
        f.create_dataset(
            dataset_name,
            data=data,
            compression="gzip",
            compression_opts=9,
        )

def load_CDF_data(filename: str) -> NDArray:
    """
    Load the tabulated CDF function from an H5 file

    Parameters:
    filename (str): path to the H5 file

    Returns:
    NDArray: the data in an array of shape (N, 2)
    """
    with h5py.File(filename,'r') as f:
        CDF_data = f["CDF_data"][:]
    return CDF_data

def timing(func):
    """
    Decorator for measuring execution time of functions

    Parameters:
    func (Callable): the function to be timed

    Returns:
    Callable: the function wrapped in the decorator
    """
    def wrapper(*args, **kwargs):
        t1 = process_time()
        result = func(*args, **kwargs)
        t2 = process_time()
        time = t2 - t1
        return result, time
    return wrapper

def progress_bar(
    iteration: int,
    total: int,
    prefix: str = '',
    suffix: str = '',
    decimals: int = 1,
    length: int = 100,
    fill: str = '█',
    printEnd: str = "\r",
):
    percent = ("{0:." + str(decimals) + "f}").format(100.0 * iteration / total)
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    if iteration == total: 
        print()
        

if __name__ == "__main__":
    pass
