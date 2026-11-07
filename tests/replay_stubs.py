"""CPU stand-ins for the CLIP model and a dataset, for the T01/T02 tests.

`StubCLIP` exposes the surface of `clip.custom_clip.ClipTestTimeTuning` that
the single-view methods and the extractor use: `image_encoder`, `dtype`,
`logit_scale` (log value), `get_text_features`, `forward_features`, `__call__`
and `eval`. Its image features are a fixed function of the pixels, so a run that
feeds it images and a replay that feeds cached features are comparable.
"""

import math

import torch
from torch.nn import functional as F
from torch.utils.data import Dataset


class StubCLIP:
    def __init__(self, n_classes=5, dim=16, pixels=3 * 4 * 4, seed=123):
        g = torch.Generator().manual_seed(seed)
        self.weight = torch.randn(pixels, dim, generator=g)
        self.text = F.normalize(torch.randn(n_classes, dim, generator=g), dim=-1)
        self.logit_scale = torch.tensor(math.log(100.0))

    @property
    def dtype(self):
        return self.weight.dtype

    def eval(self):
        return self

    def image_encoder(self, x):
        return x.reshape(x.shape[0], -1) @ self.weight

    def get_text_features(self):
        return self.text.clone()

    def forward_features(self, input):
        image_features = self.image_encoder(input.type(self.dtype))
        text_features = self.get_text_features()
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features, text_features, self.logit_scale.exp()

    def __call__(self, input):
        f, t, s = self.forward_features(input)
        return s * f @ t.t()


class FakeImageDataset(Dataset):
    """`(image, label)` pairs like upstream's BaseJsonDataset, after the transform.

    Images are clustered around one prototype per class, so TDA's caches see
    realistic agreement, and each image is unique, so stream order matters.
    """

    def __init__(self, n=48, n_classes=5, shape=(3, 4, 4), seed=7):
        g = torch.Generator().manual_seed(seed)
        protos = torch.randn(n_classes, *shape, generator=g)
        self.labels = torch.randint(0, n_classes, (n,), generator=g)
        self.images = protos[self.labels] + 0.8 * torch.randn(n, *shape, generator=g)
        self.image_list = [f"class{int(c)}/img_{i:03d}.jpg" for i, c in enumerate(self.labels)]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.images[index], self.labels[index].clone()


def consume_rng_like_model_init():
    """Stand-in for CLIP's `__init__`: draws from the global RNG, then returns a model."""
    torch.nn.Linear(64, 64)  # kaiming init draws from the global generator
    torch.randn(1000)
    return StubCLIP()
