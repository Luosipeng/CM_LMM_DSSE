import torch
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

# 构造一个简单的图:4个节点,每个节点3维特征
x = torch.randn(4, 3)

# 定义边(节点0-1, 1-2, 2-3 相连,无向图需要双向)
edge_index = torch.tensor([[0, 1, 1, 2, 2, 3],
                            [1, 0, 2, 1, 3, 2]], dtype=torch.long)

data = Data(x=x, edge_index=edge_index).to("cuda")

# 定义一个简单的两层GCN
class SimpleGCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(3, 8)
        self.conv2 = GCNConv(8, 2)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

model = SimpleGCN().to("cuda")
out = model(data.x, data.edge_index)

print("✅ GNN前向传播成功!")
print("输出的节点嵌入向量:\n", out)
