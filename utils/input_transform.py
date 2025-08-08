import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------
# Simple Input Transformer
# --------------------------

class InputTransformer(nn.Module):
    def __init__(self, output_length=1024):
        """
        Transforms a 5x5 input (flattened 25 floats) into a 1D signal of length output_length.
        """
        super(InputTransformer, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(25, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, output_length)
        )
    
    def forward(self, x):
        # x: expected shape (batch, 1, 5, 5)
        # Flatten to (batch, 25)
        x = x.view(x.size(0), -1)
        x = self.fc(x)  # (batch, output_length)
        # Add a channel dimension to match AttnSleep's expected input: (batch, 1, output_length)
        return x.unsqueeze(1)


from AttnSleep.model.attnsleep_model import AttnSleep

class AttnSleepWrapper(nn.Module):
    def __init__(self, transformer_output_length=3000, freeze_attnsleep=True):
        """
        A wrapper model that first transforms the input to match AttnSleep's input,
        then feeds the transformed signal into AttnSleep for classification.
        
        Parameters:
          transformer_output_length: Length of the 1D signal output by the transformer.
          freeze_attnsleep: If True, the AttnSleep weights are frozen.
        """
        super(AttnSleepWrapper, self).__init__()
        self.input_transformer = InputTransformer(output_length=transformer_output_length)
        self.attnsleep = AttnSleep()  # from your cloned AttnSleep repository
        
        if freeze_attnsleep:
            for param in self.attnsleep.parameters():
                param.requires_grad = False  # freeze AttnSleep
        
    def forward(self, x):
        """
        x: shape (batch, 1, 5, 5) — your chirplet-transformed input.
        """
        # Transform the 5x5 input into a 1D signal of shape (batch, 1, transformer_output_length)
        x_trans = self.input_transformer(x)
        # Feed the transformed input into AttnSleep.
        output = self.attnsleep(x_trans)
        return output

# --------------------------
# Example Usage
# --------------------------
if __name__ == "__main__":
    # Create a dummy input that mimics your 5x5 input.
    dummy_input = torch.randn(8, 1, 5, 5)  # batch of 8 samples
    
    # Instantiate the wrapper model.
    model = AttnSleepWrapper(transformer_output_length=1024, freeze_attnsleep=True)
    print(model)
    
    # Forward pass through the model.
    output = model(dummy_input)
    print("Output shape:", output.shape)
    
    # You can now use this model as a starting point.
    # Optionally, you can later unfreeze AttnSleep and fine-tune the entire network on your dataset.
