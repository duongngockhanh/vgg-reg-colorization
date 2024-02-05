import wandb
import torch
import matplotlib.pyplot as plt
from colorizers import *
from dataloaders import *
from torchsummary import summary



## --------------- Default ---------------
train_in_path = "small-coco-stuff/train2017/train2017"
val_in_path = "small-coco-stuff/train2017/train2017"

train_batch_size = 16
val_batch_size = 16

train_loader = create_dataloader(train_in_path, batch_size=train_batch_size, shuffle=True)
val_loader = create_dataloader(val_in_path, batch_size=val_batch_size,shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ECCV_Regression().to(device)
# summary(model, (1, 256, 256))

lr = 1e-3
epochs = 50

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)



## --------------- Addition ---------------
def evaluate(model, dataloader, criterion, device):
    model.eval()
    losses = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            losses.append(loss.item())
    loss = sum(losses) / len(losses)
    return loss


def fit(model, train_loader, val_loader, criterion, optimizer, device, epochs, wandb_api):
    wandb.login(key=wandb_api)
    wandb.init(
        project="zhang-train-reg",
        config={
            "dataset": "coco-stuff",
            "architecture": "ECCV - Linear",
            "criterion": "MSE",
            "optimizer": "Adam",
            "epochs": 50,
            "lr": lr
        }
    )
    
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        batch_train_losses = []
        model.train()
        for _, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            batch_train_losses.append(loss.item())
            wandb.log({"batch_train_loss": loss})
            if train_batch_size * len(batch_train_losses) > 10000:
                break

        train_loss = sum(batch_train_losses) / len(batch_train_losses)
        train_losses.append(train_loss)

        val_loss = evaluate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        
        wandb.log({"train_loss": train_loss, "val_loss": val_loss})

        print(f'EPOCH {epoch + 1}:\tTrain loss: {train_loss:.4f}\tVal loss: {val_loss:.4f}')

    return train_losses, val_losses