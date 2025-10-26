from options import Options, RegOptions
import models.models_utils as m_utils
import constants
from custom_types import *
import torch.nn.functional as F


# class ProcessEmbSimple(Module):
#     def __init__(self, emb_dim: int, hidden_dim: int):
#         super(ProcessEmbSimple, self).__init__()
#         self.net_up = nn.Sequential(m_utils.MLP((emb_dim, 2 * hidden_dim,
#                                     4 * hidden_dim, 8 * hidden_dim)),
#                                     m_utils.View(-1, 1, 8, hidden_dim))
#
#     def forward(self, *args):
#         return self.net_up(args[0])


class ProcessEmbSimple(nn.Module):
    def __init__(self, emb_dim: int, hidden_dim: int):
        super(ProcessEmbSimple, self).__init__()
        upsample = [
            nn.Linear(emb_dim, 2 * hidden_dim),
            nn.LayerNorm(2 * hidden_dim),
            nn.ReLU(True),
            m_utils.View(-1, 2, hidden_dim),
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.LayerNorm(2 * hidden_dim),
            nn.ReLU(True),
            m_utils.View(-1, 4, hidden_dim),
            nn.Linear(hidden_dim, 2 * hidden_dim),
            m_utils.View(-1, 1, 8, hidden_dim)
        ]
        self.net_up = nn.Sequential(*upsample)

    def forward(self, *args):
        return self.net_up(args[0])


class GMCast(Module):

    def __init__(self, hidden_dim: int):
        super(GMCast, self).__init__()
        projection_dim = constants.DIM ** 2 + 2 * constants.DIM + 1
        self.mlp = m_utils.MLP((hidden_dim, hidden_dim // 2, projection_dim), dropout=0.1)
        self.split_shape = tuple((constants.DIM + 2) * [constants.DIM] + [1])

    @staticmethod
    def dot(x, y, dim=3):
        return torch.sum(x * y, dim=dim)

    def remove_projection(self, v_1, v_2):
        proj = (self.dot(v_1, v_2) / self.dot(v_2, v_2))
        return v_1 - proj[:, :, :, None] * v_2

    def forward(self, x, t: None or T):
        if t is not None:
            t = t.unsqueeze(1).unsqueeze(1).expand(x.shape[0], x.shape[1], x.shape[2], -1)
            x = torch.cat((x, t), 3)
        x = self.mlp(x)
        splitted = torch.split(x, self.split_shape, dim=3)
        # Gram–Schmidt process
        raw_base = []
        for i in range(constants.DIM):
            u = splitted[i]
            for j in range(i):
                u = self.remove_projection(u, raw_base[j])
            raw_base.append(u)
        p = torch.stack(raw_base, dim=3)
        p = p / torch.norm(p, p=2, dim=4)[:, :, :, :, None]  # + self.noise[None, None, :, :]
        # eigenvalues
        eigen = splitted[constants.DIM] ** 2 + constants.EPSILON
        sigma_det = eigen[:, :, :, 0]
        for i in range(1, constants.DIM):
            sigma_det = sigma_det * eigen[:, :, :, i]
        mu = splitted[constants.DIM + 1]
        phi = splitted[constants.DIM + 2].squeeze(3)
        return mu, p, sigma_det, phi, eigen


class MultiGMM(Module):

    def __init__(self, opt: Union[Options, RegOptions]):
        super(MultiGMM, self).__init__()
        self.split_cast = opt.split_cast
        self.process_layer = ProcessEmbSimple(opt.dim_z, opt.dim_h)
        t_dim = opt.dim_t if opt.registration else 0
        self.projector = GMCast(opt.dim_h + t_dim)
        if opt.split_cast:
            self.mid_projector = nn.ModuleList([GMCast(opt.dim_h + t_dim) for _ in range(opt.num_splits)])
        else:
            self.mid_projector = (lambda *x: None)if opt.only_last else self.projector
        if opt.attentive:
            self.attention = nn.ModuleList([m_utils.GMAttend(opt.dim_h) for _ in range(opt.num_splits)])
        else:
            self.attention = [m_utils.Dummy() for _ in range(opt.num_splits)]
        self.mlp_split = nn.ModuleList([m_utils.MLP((opt.dim_h, opt.dim_h * 2, opt.dim_h * 4), dropout=0.1)
                                        for _ in range(opt.num_splits)])

        self.linear_layers = nn.ModuleList([
            nn.Linear(128 * 4 * 3, 128),  # 对 (8, 128, 4, 3) 张量进行降维
            nn.Linear(128 * 4 * 3 * 3, 128),  # 对 (8, 128, 4, 3, 3) 张量进行降维
            nn.Linear(128 * 4, 128),  # 对 (8, 128, 4) 张量进行降维
            nn.Linear(128 * 4, 128),  # 对 (8, 128, 4) 张量进行降维
            nn.Linear(128 * 4 * 3, 128)  # 对 (8, 128, 4, 3) 张量进行降维
        ])

    def forward(self, x: T, t: TN = None) -> List[Tuple[T, ...]]:
        gms = []
        raw_gm = self.process_layer(x)
        for i in range(len(self.attention)):
            if self.split_cast:
                gms.append(self.mid_projector[i](raw_gm, t))
            else:
                gms.append(self.mid_projector(raw_gm, t))
            raw_gm = self.gm_split(raw_gm, self.attention[i], self.mlp_split[i])
        gms.append(self.projector(raw_gm, t))
        return gms

    @staticmethod
    def reshape_and_reduce(gms_tensor: T, linear_layer: nn.Linear) -> T:

        # 展平张量
        gms_tensor_resized = gms_tensor.reshape(gms_tensor.shape[0], -1)
        # 确保展平后的形状与线性层的输入维度匹配
        assert gms_tensor_resized.shape[1] == linear_layer.in_features, \
            f"Input dimension mismatch: expected {linear_layer.in_features}, got {gms_tensor_resized.shape[1]}"
        # 使用线性层进行降维
        gms_tensor_resized = linear_layer(gms_tensor_resized)
        return gms_tensor_resized



    @staticmethod
    def gm_split(x: T, attention: nn.Module, mlp: nn.Module) -> T:
        b_size, grand_parents, parents, dim = x.shape
        x = attention(x)
        out = mlp(x).view(b_size, grand_parents, parents, -1, dim) # + x[:, :, :, None, :]
        return out.view(b_size, grand_parents * parents, -1, dim)


def weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight, gain=np.sqrt(2.0))
    elif classname.find('Conv') != -1:
        nn.init.xavier_normal_(m.weight, gain=np.sqrt(2.0))
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight, gain=np.sqrt(2.0))
    elif classname.find('Embe') != -1:
        # nn.init.xavier_uniform(m.weight, gain=np.sqrt(2.0))
        nn.init.normal_(m.weight, mean=0, std=1)


class Concatenate(Module):
    def __init__(self, dim):
        super(Concatenate, self).__init__()
        self.dim = dim

    def forward(self, x):
        return torch.cat(x, dim=self.dim)


class View(Module):

    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)


class Transpose(Module):

    def __init__(self, dim0, dim1):
        super(Transpose, self).__init__()
        self.dim0, self.dim1 = dim0, dim1

    def forward(self, x):
        return x.transpose(self.dim0, self.dim1)


class Dummy(Module):

    def __init__(self, *args):
        super(Dummy, self).__init__()

    def forward(self, *args):
        return args[0]


class MLP(Module):

    def __init__(self,ch: tuple, norm_class=nn.LayerNorm, dropout=0, skip=False):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(ch) - 1):
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(ch[i], ch[i + 1]))
            if i < len(ch) - 2:
                layers += [
                    norm_class(ch[i + 1]),
                    nn.ReLU(True)
                ]
        self.skip = skip
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        out = self.net(x)
        if self.skip:
            out = x + out
        return out


class GMAttend(Module):

    def __init__(self, hidden_dim: int):
        super(GMAttend, self).__init__()
        self.key_dim = hidden_dim // 8
        self.query_w = nn.Linear(hidden_dim, self.key_dim)
        self.key_w = nn.Linear(hidden_dim, self.key_dim)
        self.value_w = nn.Linear(hidden_dim, hidden_dim)
        self.softmax = nn.Softmax(dim=3)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.scale = 1 / torch.sqrt(torch.tensor(self.key_dim, dtype=torch.float32))

    def forward(self, x):
        queries = self.query_w(x)
        keys = self.key_w(x)
        vals = self.value_w(x)
        attention = self.softmax(torch.einsum('bgqf,bgkf->bgqk', queries, keys))
        out = torch.einsum('bgvf,bgqv->bgqf', vals, attention)
        out = self.gamma * out + x
        return out


def dkl(mu, log_sigma):
    if log_sigma is None:
        return torch.zeros(1).to(mu.device)
    else:
        return 0.5 * torch.sum(torch.exp(log_sigma) - 1 - log_sigma + mu ** 2) / (mu.shape[0] * mu.shape[1])


def recursive_to(item, device):
    if type(item) is T:
        return item.to(device)
    elif type(item) is tuple or type(item) is list:
        return [recursive_to(item[i]) for i in range(len(item))]
    return item



