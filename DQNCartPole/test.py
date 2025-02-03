import torch

from network import QNetwork

def test_QNetwork():
    model = QNetwork(2, (4, ), [120, 84])
    param_count = sum(p.nelement() for p in model.parameters())
    print(f"Number of params is {param_count}")
    print(model)





if __name__ == "__main__":
    test_QNetwork()
    