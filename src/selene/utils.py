"""Replaces the original NonStrandSpecific class, as flip now is built into torch

Returns:
    [type] -- [description]
"""
import torch
import selene_sdk


class NonStrandSpecific(selene_sdk.utils.NonStrandSpecific):
    def forward(self, input):
        reverse_input = torch.flip(input, (1, 2))
        output = self.model.forward(input)
        output_from_rev = self.model.forward(
            reverse_input)
        if self.mode == "mean":
            return (output + output_from_rev) / 2
        else:
            return torch.max(output, output_from_rev)
