import h5py

from numbers import Real
from nptyping import NDArray

def binary_size(num: Real, suffix: str = "B") -> str:
    for unit in ("", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"):
        if abs(num) < 1024.0:
            return f"{num:3.1f} {unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f} Yi{suffix}"

def time_fmt(time_s: Real) -> str:
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
