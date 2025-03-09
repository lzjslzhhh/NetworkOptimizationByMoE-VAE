import numpy as np
import pandas as pd

from config import Config


def generate_slice_data(slice_type, num_samples):
    time = np.arange(num_samples)
    if slice_type == 'iot':
        demand = 10 * np.sin(0.1 * time) + np.random.normal(0, 1, num_samples)
    elif slice_type == 'video':
        demand = np.random.poisson(lam=5, size=num_samples) + 20 * (np.sin(0.05 * time) > 0)
    elif slice_type == 'web':
        demand = 15 + np.random.normal(0, 3, num_samples)
    return np.clip(demand, 0, None)


if __name__ == "__main__":
    config = Config()
    slices = ['iot', 'video', 'web']
    data = {slice: generate_slice_data(slice, config.num_samples) for slice in slices}
    df = pd.DataFrame(data)
    df.to_csv(config.data_path, index=False)
    print(f"Generated data saved to {config.data_path}")
