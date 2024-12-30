import torch
import torch.nn as nn
import torch.nn.functional as F


# Adapted from PyTorch Linear module
# https://pytorch.org/docs/stable/_modules/torch/nn/modules/linear.html#Linear
class NonNegLinear(torch.nn.Module):
    """
    Applies a linear transformation to the incoming data with non-negative
    weights.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super(NonNegLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(
            (out_features, in_features), **factory_kwargs))
        self.normalization_multiplier = torch.nn.Parameter(
            torch.ones((1,), requires_grad=True))
        if bias:
            self.bias = nn.Parameter(torch.empty(
                out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

    def forward(self, model_input: torch.Tensor) -> torch.Tensor:
        return F.linear(model_input, torch.relu(self.weight), self.bias)


# Adapted from get_network() method in PIPNet code
# https://github.com/M-Nauta/PIPNet/blob/main/pipnet/pipnet.py
class PIPHead(torch.nn.Module):
    """
    Head of the PIPNet model. Uses the provided model as backbone.
    Model requires a Conv2d layer.
    """

    def __init__(self,
                 num_classes: int,
                 model: nn.Module,
                 num_features: int = 0,
                 bias: bool = False):
        super(PIPHead, self).__init__()

        # Use the provided model as the feature extractor
        self.features = model

        # Find the number of output channels of the last Conv2d layer
        first_add_on_layer_in_channels = None
        for layer in reversed(list(self.features.modules())):
            if isinstance(layer, nn.Conv2d):
                first_add_on_layer_in_channels = layer.out_channels
                break

        if first_add_on_layer_in_channels is None:
            raise Exception("No Conv2d layer found in the model")

        # Initialize add_on_layers and num_prototypes
        if num_features == 0:
            self.num_prototypes = first_add_on_layer_in_channels
            print("Number of prototypes: ", self.num_prototypes, flush=True)
            self.add_on_layers = nn.Sequential(
                nn.Softmax(dim=1),
            )
        else:
            self.num_prototypes = num_features
            print(
                "Number of prototypes set from",
                first_add_on_layer_in_channels,
                "to",
                self.num_prototypes,
                ". Extra 1x1 conv layer added. Not recommended.",
                flush=True,
            )
            self.add_on_layers = nn.Sequential(
                nn.Conv2d(
                    in_channels=first_add_on_layer_in_channels,
                    out_channels=self.num_prototypes,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=True,
                ),
                nn.Softmax(dim=1),
            )

        # Initialize pool_layer
        self.pool_layer = nn.Sequential(
            nn.AdaptiveMaxPool2d(output_size=(1, 1)),
            nn.Flatten(),
        )

        # Initialize classification_layer
        if bias:
            self.classification_layer = NonNegLinear(
                self.num_prototypes, num_classes, bias=True)
        else:
            self.classification_layer = NonNegLinear(
                self.num_prototypes, num_classes, bias=False)

    def forward(self, x):
        x = self.features(x)
        x = self.add_on_layers(x)
        x = self.pool_layer(x)
        x = self.classification_layer(x)
        return x
