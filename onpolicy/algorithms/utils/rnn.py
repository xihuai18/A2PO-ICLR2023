import torch
import torch.nn as nn

from .util import get_clones, init
"""RNN modules."""


class RNNLayer(nn.Module):

    def __init__(
        self,
        inputs_dim,
        outputs_dim,
        recurrent_N,
        use_orthogonal,
        layer_after_N: int = 1,
        use_ReLU: bool = True,
    ):
        super(RNNLayer, self).__init__()
        self._recurrent_N = recurrent_N
        self._use_orthogonal = use_orthogonal
        self._layer_after_N = layer_after_N
        self._use_ReLU = use_ReLU

        active_func = [nn.Tanh(), nn.ReLU()][use_ReLU]
        init_method = [nn.init.xavier_uniform_,
                       nn.init.orthogonal_][use_orthogonal]
        gain = nn.init.calculate_gain(["tanh", "relu"][use_ReLU])

        self.rnn = nn.GRU(inputs_dim,
                          outputs_dim,
                          num_layers=self._recurrent_N)
        for name, param in self.rnn.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                if self._use_orthogonal:
                    nn.init.orthogonal_(param)
                else:
                    nn.init.xavier_uniform_(param)

        def init_(m: nn.Module) -> nn.Module:
            return init(m,
                        init_method,
                        lambda x: nn.init.constant_(x, 0),
                        gain=gain)

        self.norm = nn.LayerNorm(outputs_dim)

        self.fc_after = nn.Sequential(
            init_(nn.Linear(outputs_dim, outputs_dim)),
            active_func,
            nn.LayerNorm(outputs_dim),
        )

        self.fc = get_clones(self.fc_after, self._layer_after_N)
        self.fc = nn.Sequential(*self.fc)

    def forward(self, x, hxs, masks):
        if x.size(0) == hxs.size(0):
            x, hxs = self.rnn(
                x.unsqueeze(0),
                (hxs *
                 masks.repeat(1, self._recurrent_N).unsqueeze(-1)).transpose(
                     0, 1).contiguous(),
            )
            x = x.squeeze(0)
            hxs = hxs.transpose(0, 1)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = (masks[1:] == 0.0).any(
                dim=-1).nonzero().squeeze().cpu()

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            hxs = hxs.transpose(0, 1)

            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]
                temp = (hxs * masks[start_idx].view(1, -1, 1).repeat(
                    self._recurrent_N, 1, 1)).contiguous()
                rnn_scores, hxs = self.rnn(x[start_idx:end_idx], temp)
                outputs.append(rnn_scores)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)

            # flatten
            x = x.reshape(T * N, -1)
            hxs = hxs.transpose(0, 1)

        x = self.norm(x)

        x = self.fc(x)

        return x, hxs
