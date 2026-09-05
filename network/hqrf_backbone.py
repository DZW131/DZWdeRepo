"""Plain official ResNet38 with frozen HQRF Phase-0 feature taps."""
from __future__ import annotations

import hashlib
from pathlib import Path

from torch.nn import functional as F

from network.resnet38d import Net, convert_mxnet_to_torch


INIT_SHA256 = "f668a2add80e33dfa8f1a0695df91f6d8cfad5ffbb26d1dc7bcd35903a1f6e16"
FEATURE_TAPS = {
    "F3": {"source": "b3_2 output", "channels": 256, "stride": 4},
    "F4": {"source": "relu(bn45(b4_5 output))", "channels": 512, "stride": 8},
    "F5": {"source": "relu(bn52(b5_2 output))", "channels": 1024, "stride": 8},
    "FD": {"source": "relu(bn7(b7 output))", "channels": 4096, "stride": 8},
}


class HQRFBackbone(Net):
    def __init__(self):
        super().__init__()
        self.not_training = [self.conv1a, self.b2, self.b2_1, self.b2_2]

    def forward(self, image):
        value = self.conv1a(image)
        value = self.b2_2(self.b2_1(self.b2(value)))
        value = self.b3_2(self.b3_1(self.b3(value)))
        f3 = value
        value = self.b4_5(self.b4_4(self.b4_3(self.b4_2(self.b4_1(self.b4(value))))))
        f4 = F.relu(self.bn45(value))
        value, _ = self.b5(value, get_x_bn_relu=True)
        value = self.b5_2(self.b5_1(value))
        f5 = F.relu(self.bn52(value))
        value, _ = self.b6(value, get_x_bn_relu=True)
        fd = F.relu(self.bn7(self.b7(value)))
        return {"F3": f3, "F4": f4, "F5": f5, "FD": fd}

    def load_official_initialization(self, path):
        path = Path(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if path.suffix != ".params" or digest != INIT_SHA256:
            raise ValueError("Expected hash-locked official MXNet ResNet38 initialization")
        converted = convert_mxnet_to_torch(str(path))
        missing, unexpected = self.load_state_dict(converted, strict=False)
        if unexpected or any(not name.startswith(("bn45.", "bn52.")) for name in missing):
            raise ValueError(f"Unexpected initialization state: {missing=}, {unexpected=}")
        return {
            "source": str(path.resolve()),
            "source_sha256": digest,
            "missing_parameters": list(missing),
            "feature_taps": FEATURE_TAPS,
        }
