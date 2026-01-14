from project_name.data import MyDataset
from project_name.model import Model


def train():
    _dataset = MyDataset("data/raw")
    _model = Model()


if __name__ == "__main__":
    train()
