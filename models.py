"""
Model definitions for the JerseyIQ pipeline.

These architectures were reverse-engineered directly from the shapes stored
inside the provided checkpoints (checkpoints/jersey_cnn/best.pt and
checkpoints/ccnn_filter/best.pt), so that `load_state_dict(..., strict=True)`
succeeds without guessing. If your original training code differs slightly
in non-parametric layers (activation choice, dropout rate, etc.) that's fine
those don't show up in the state dict and won't break loading.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------
# Jersey number OCR network (checkpoints/jersey_cnn/best.pt)
#
# Reconstructed layout (names/shapes match the checkpoint exactly):
#   stn.loc   : Conv2d(3,16,7,pad=3) -> MaxPool2d(2) -> ReLU
#               -> Conv2d(16,32,5,pad=2) -> MaxPool2d(2) -> ReLU
#   stn.fc    : Linear(8192,64) -> ReLU -> Linear(64,6)   (affine grid params)
#   backbone  : [Conv2d(3,32,3,pad=1), BN(32), ReLU, MaxPool2d(2)]
#               [Conv2d(32,64,3,pad=1), BN(64), ReLU, MaxPool2d(2)]
#               [Conv2d(64,128,3,pad=1), BN(128), ReLU]
#   trunk     : Dropout -> Linear(128,128) -> ReLU   (applied after global avg pool)
#   heads     : head_visible (2), head_tens (11: 0-9 + none), head_units (11: 0-9 + none)
#
# Input crop size is 64x64 RGB (derived from the 8192 = 32*16*16 STN fc input).
# --------------------------------------------------------------------------

JERSEY_INPUT_SIZE = 64


class JerseyCNN(nn.Module):
    def __init__(self, dropout: float = 0.3):
        super().__init__()

        # --- Spatial Transformer localization network ---
        self.stn = nn.Module()
        self.stn.loc = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=7, padding=3),   # stn.loc.0
            nn.MaxPool2d(2),                               # stn.loc.1
            nn.ReLU(inplace=True),                         # stn.loc.2
            nn.Conv2d(16, 32, kernel_size=5, padding=2),   # stn.loc.3
            nn.MaxPool2d(2),                               # stn.loc.4
            nn.ReLU(inplace=True),                         # stn.loc.5
        )
        self.stn.fc = nn.Sequential(
            nn.Linear(32 * 16 * 16, 64),  # stn.fc.0
            nn.ReLU(inplace=True),        # stn.fc.1
            nn.Linear(64, 6),             # stn.fc.2
        )
        # Initialize as identity transform so an untrained/partial STN never
        # destroys the image before weights are loaded.
        self.stn.fc[2].weight.data.zero_()
        self.stn.fc[2].bias.data.copy_(
            torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float32)
        )

        # --- Backbone ---
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),    # backbone.0
            nn.BatchNorm2d(32),                             # backbone.1
            nn.ReLU(inplace=True),                          # backbone.2
            nn.MaxPool2d(2),                                # backbone.3
            nn.Conv2d(32, 64, kernel_size=3, padding=1),    # backbone.4
            nn.BatchNorm2d(64),                             # backbone.5
            nn.ReLU(inplace=True),                          # backbone.6
            nn.MaxPool2d(2),                                # backbone.7
            nn.Conv2d(64, 128, kernel_size=3, padding=1),   # backbone.8
            nn.BatchNorm2d(128),                            # backbone.9
            nn.ReLU(inplace=True),                          # backbone.10
        )

        # --- Trunk (after global average pooling, done in forward()) ---
        self.trunk = nn.Sequential(
            nn.Dropout(dropout),   # trunk.0 (no params)
            nn.Linear(128, 128),   # trunk.1
            nn.ReLU(inplace=True), # trunk.2
        )

        # --- Heads ---
        self.head_visible = nn.Linear(128, 2)   # 0=not visible, 1=visible
        self.head_tens = nn.Linear(128, 11)     # digits 0-9, 10=none
        self.head_units = nn.Linear(128, 11)    # digits 0-9, 10=none

    def _stn_transform(self, x):
        xs = self.stn.loc(x)
        xs = xs.reshape(xs.size(0), -1)
        theta = self.stn.fc(xs)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        return F.grid_sample(x, grid, align_corners=False)

    def forward(self, x):
        # x: (B, 3, 64, 64), normalized RGB float tensor
        x = self._stn_transform(x)
        feat = self.backbone(x)
        feat = F.adaptive_avg_pool2d(feat, 1).flatten(1)  # (B, 128)
        feat = self.trunk(feat)
        return {
            "visible": self.head_visible(feat),
            "tens": self.head_tens(feat),
            "units": self.head_units(feat),
        }

    @torch.no_grad()
    def predict_number(self, x):
        """Returns list of dicts: {number:int|None, confidence:float, visible:bool}"""
        out = self.forward(x)
        vis_p = F.softmax(out["visible"], dim=1)
        tens_p = F.softmax(out["tens"], dim=1)
        units_p = F.softmax(out["units"], dim=1)

        vis_conf, vis_idx = vis_p.max(dim=1)
        tens_conf, tens_idx = tens_p.max(dim=1)
        units_conf, units_idx = units_p.max(dim=1)

        results = []
        for i in range(x.size(0)):
            visible = bool(vis_idx[i].item() == 1)
            tens_d = tens_idx[i].item()
            units_d = units_idx[i].item()
            if not visible:
                results.append({"number": None, "confidence": float(vis_conf[i].item()), "visible": False})
                continue
            digits = ""
            confs = [vis_conf[i].item()]
            if tens_d != 10:
                digits += str(tens_d)
                confs.append(tens_conf[i].item())
            if units_d != 10:
                digits += str(units_d)
                confs.append(units_conf[i].item())
            number = int(digits) if digits else None
            conf = sum(confs) / len(confs)
            results.append({"number": number, "confidence": float(conf), "visible": True})
        return results


# --------------------------------------------------------------------------
# Temporal filter / possession-touch classifier (checkpoints/ccnn_filter/best.pt)
#
# 4 residual Conv1d blocks (kernel=5, pad=2, channels=32) operating on a
# short window of per-frame features (3 input channels), followed by a
# 1x1 conv projecting to 2 classes (no-touch / touch) per timestep.
# --------------------------------------------------------------------------

CCNN_IN_CHANNELS = 3
CCNN_WINDOW = 16  # frames of context fed to the filter per prediction


class ResBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=5):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad)
        self.norm1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad)
        self.norm2 = nn.BatchNorm1d(out_ch)
        self.skip = None
        if in_ch != out_ch:
            self.skip = nn.Conv1d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        identity = x if self.skip is None else self.skip(x)
        out = F.relu(self.norm1(self.conv1(x)), inplace=True)
        out = self.norm2(self.conv2(out))
        return F.relu(out + identity, inplace=True)


class CCNNFilter(nn.Module):
    def __init__(self, in_channels=CCNN_IN_CHANNELS, channels=32, n_blocks=4):
        super().__init__()
        blocks = []
        cur_in = in_channels
        for _ in range(n_blocks):
            blocks.append(ResBlock1D(cur_in, channels))
            cur_in = channels
        self.blocks = nn.ModuleList(blocks)
        self.out = nn.Conv1d(channels, 2, kernel_size=1)  # per-timestep 2-class logits

    def forward(self, x):
        # x: (B, in_channels, T)
        for blk in self.blocks:
            x = blk(x)
        return self.out(x)  # (B, 2, T)

    @torch.no_grad()
    def predict_touch_prob(self, x):
        """Returns touch probability per timestep: (B, T)"""
        logits = self.forward(x)
        probs = F.softmax(logits, dim=1)
        return probs[:, 1, :]


def load_jersey_cnn(weights_path, device="cpu"):
    model = JerseyCNN()
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model


def load_ccnn_filter(weights_path, device="cpu"):
    model = CCNNFilter()
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model
