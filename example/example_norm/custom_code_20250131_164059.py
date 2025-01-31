from cellmap_flow.norm.input_normalize import InputNormalizer

class ZScoreNormalizer(InputNormalizer):

    def __init__(self, mean=0.0, std=1.0):
        self.mean = mean
        self.std = std

    def normalize(self, data: np.ndarray) -> np.ndarray:
        data = data.astype(np.float32)
        return ((data - self.mean) / self.std).astype(np.float32)