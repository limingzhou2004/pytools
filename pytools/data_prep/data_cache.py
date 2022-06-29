import pickle


class DataCache:
    def save_pickle(self, fn: str):
        with open(fn, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_pickle(self, fn):
        with open(fn, "rb") as f:
            return pickle.load(f)
