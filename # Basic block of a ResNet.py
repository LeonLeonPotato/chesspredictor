# Basic block of a ResNet
class ResBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(filter_size, filter_size, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(filter_size, filter_size, kernel_size=3, padding=1),
            nn.LeakyReLU(),
        ) if not use_batchnorm else nn.Sequential(
            nn.Conv2d(filter_size, filter_size, kernel_size=3, padding=1),
            nn.BatchNorm2d(filter_size),
            nn.LeakyReLU(),
            nn.Conv2d(filter_size, filter_size, kernel_size=3, padding=1),
            nn.BatchNorm2d(filter_size),
        )
    
    def forward(self, x):
        x1 = self.convs(x)
        x1 += x
        return F.leaky_relu(x1)

# Processes a single state from the last 4 states
class MoveProcessor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(6, filter_size, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            *[ResBlock() for _ in range(move_processor_blocks)],
            nn.Conv2d(filter_size, filter_size, kernel_size=3, padding=1),
            nn.LeakyReLU(),
        )
    
    def forward(self, x):
        return self.convs(x)

# The main model
class ChessNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.move_processor = MoveProcessor()
        self.convs = nn.Sequential(
            nn.Conv2d(filter_size * history_count, filter_size, kernel_size=3, padding=1),
            *[ResBlock() for _ in range(main_model_blocks)],
            nn.Flatten(),
            nn.Linear(64*8*8, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 1),
            nn.Tanh()
        )
    
    def forward(self, x):
        # We do this because it leverages the GPU more
        newx = []
        for i in range(4):
            newx.append(self.move_processor(x[:, i*6:(i+1)*6, :, :]))
        x = torch.cat(newx, dim=1)
        return self.convs(x)
    
    def predict(self, x):
        x = x.reshape(1, 6*history_count, 8, 8)
        return self.forward(x).item()