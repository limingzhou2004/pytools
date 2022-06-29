import numpy as np


class WeatherScaler:
    # only need to scale weather data, mean/std or min/max; no need to un-scale

    def __init__(self, data: np.ndarray, mode: str = "minmax"):
        # aggregate by channel
        assert len(data.shape) == 4, "data dimension should be 4!"
        self.w_mean = np.mean(data, axis=(0, 1, 2))
        self.w_std = np.std(data, axis=(0, 1, 2))
        self.w_min = np.min(data, axis=(0, 1, 2))
        self.w_max = np.max(data, axis=(0, 1, 2))
        self.mode = mode

    def transform(self, data: np.ndarray) -> np.ndarray:
        return self.scale(data)

    def scale(self, data: np.ndarray) -> np.ndarray:
        if self.mode == "minmax":
            return self._scale(data, self.w_min, self.w_max - self.w_min)
        else:
            return self._scale(data, self.w_mean, self.w_std)

    def _scale(
        self, weather_array: np.ndarray, mean: np.ndarray, std: np.ndarray
    ) -> np.ndarray:
        para = np.vstack([mean, std]).T
        for i in range(0, para.shape[0]):
            weather_array[:, :, :, i] = (weather_array[:, :, :, i] - para[i, 0]) / para[
                i, 1
            ]
        return weather_array
