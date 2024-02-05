import wandb
import torch
import matplotlib.pyplot as plt
from colorizers import *
from dataloaders import *


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


def fit(model, train_loader, val_loader, criterion, optimizer, device, epochs, lr, train_batch_size):
    wandb.init(
        project="zhang-train-reg-2",
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
            if train_batch_size * len(batch_train_losses) > 180000:
                break

        train_loss = sum(batch_train_losses) / len(batch_train_losses)
        train_losses.append(train_loss)

        val_loss = evaluate(model, val_loader, criterion, device)
        val_losses.append(val_loss)

        # # Show image
        # val_iter = iter(val_loader)
        # val_first = next(val_iter)
        # showed_in = val_first[0][:1].to(device)
        # print(showed_in.dtype)
        # showed_pred = model(showed_in)
        # print(showed_pred.dtype)
        # showed_res = postprocess_tens(showed_in, showed_pred)
        # showed_res = Image.fromarray(showed_res)
        # image = wandb.Image(showed_res, caption=f"epoch {epoch}")

        wandb.log({"train_loss": train_loss, "val_loss": val_loss})

        print(f'EPOCH {epoch + 1}:\tTrain loss: {train_loss:.4f}\tVal loss: {val_loss:.4f}')

    wandb.finish()

    return train_losses, val_losses


def main(train_in_path=None, val_in_path=None):
    if train_in_path == None or val_in_path == None:
        train_in_path = "small-coco-stuff/train2017/train2017"
        val_in_path = "small-coco-stuff/train2017/train2017"

    train_batch_size = 32
    val_batch_size = 8

    train_loader = create_dataloader(train_in_path, batch_size=train_batch_size, shuffle=True)
    val_loader = create_dataloader(val_in_path, batch_size=val_batch_size,shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ECCV_Regression().to(device)

    lr = 1e-3
    epochs = 50

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses, val_losses = fit(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device,
        epochs,
        lr,
        train_batch_size
    )