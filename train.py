import wandb
import time
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from colorizers import *
from dataloaders import *


def show_image_wandb(val_loader, model, val_batch_size, device, epoch):
    model.eval()
    with torch.no_grad():
        val_iter = iter(val_loader)
        val_first = next(val_iter)

        images_pred = []
        images_gt = []

        fixed_num_showed_image = 5
        num_showed_image = val_batch_size if val_batch_size < fixed_num_showed_image else fixed_num_showed_image

        for i in range(num_showed_image):
            l_in = val_first[0][i:i+1].to(device)
            ab_pred = model(l_in)

            # temp = ab_pred.detach().cpu().numpy().reshape(-1)
            # plt.hist(temp)
            # plt.show()
            
            rgb_pred = postprocess_tens(l_in, ab_pred)
            rgb_pred = Image.fromarray((rgb_pred * 255).astype(np.uint8))
            image_pred = wandb.Image(rgb_pred, caption=f"epoch {epoch}")
            images_pred.append(image_pred)

            ab_gt = val_first[1][i:i+1].to(device)
            rgb_gt = postprocess_tens(l_in, ab_gt)
            rgb_gt = Image.fromarray((rgb_gt * 255).astype(np.uint8))
            image_gt = wandb.Image(rgb_gt, caption=f"epoch {epoch}")
            images_gt.append(image_gt)

    return images_pred, images_gt


def evaluate(model, dataloader, criterion, device, val_batch_size, val_num_max):
    model.eval()
    losses = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            losses.append(loss.item())
            if val_batch_size * len(losses) > val_num_max:
                break
    loss = sum(losses) / len(losses)
    return loss


def fit(model, train_loader, val_loader, saved_weight_path,
        criterion, optimizer, device, epochs, 
        train_batch_size, val_batch_size,
        train_num_max, val_num_max,
        use_wandb, wandb_proj_name, wandb_config
        ):

    if use_wandb == True:
        wandb.init(
            project=wandb_proj_name,
            config=wandb_config
        )
    
    train_losses = []
    val_losses = []

    best_val_loss = 10e5
    start_time = time.time()

    for epoch in range(epochs):
        start_time_epoch = time.time()
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
            if train_batch_size * len(batch_train_losses) > train_num_max:
                break

        train_loss = sum(batch_train_losses) / len(batch_train_losses)
        train_losses.append(train_loss)

        val_loss = evaluate(model, val_loader, criterion, device, val_batch_size, val_num_max)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            torch.save(model.state_dict(), saved_weight_path)
            best_val_loss = val_loss

        # Show image
        if use_wandb == True:
            images_pred, images_gt = show_image_wandb(val_loader, model, val_batch_size, device, epoch)
            wandb.log({"train_loss": train_loss, "val_loss": val_loss, "images_pred": images_pred, "images_gt": images_gt})

        print(f'EPOCH {epoch + 1}:\tTrain loss: {train_loss:.4f}\tVal loss: {val_loss:.4f}\tTime: {time.time() - start_time_epoch:.2f}s')

    print(f"Complete training in {time.time() - start_time:2f}s")

    if use_wandb == True:
        wandb.finish()

    return train_losses, val_losses


def main():
    save_dir = "exp_eccv"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_batch_size = 32
    val_batch_size = 8
    train_in_path = "/kaggle/input/aio-coco-stuff/train2017/train2017"
    val_in_path = "/kaggle/input/aio-coco-stuff/val2017/val2017"
    train_loader = create_dataloader(train_in_path, batch_size=train_batch_size, shuffle=True)
    val_loader = create_dataloader(val_in_path, batch_size=val_batch_size, shuffle=False)

    # Hyperparameters
    model = ECCV_Regression_Lab().to(device)
    epochs = 100
    lr = 5e-4
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_num_max = 2000
    val_num_max = 200
    pretrained = None

    if pretrained != None:
        print(f"Load model from {pretrained}")
        model.load_state_dict(torch.load(pretrained))

    # Save weight
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    saved_weights = sorted(os.listdir(save_dir))
    if len(saved_weights) == 0:
        saved_weight_file = "exp01.pt"
        saved_weight_path = os.path.join(save_dir, saved_weight_file)
    else:
        saved_weight_file = f"exp{int(saved_weights[-1][3:-3]) + 1:02d}.pt"
        saved_weight_path = os.path.join(save_dir, saved_weight_file)
    print(f"Weights will be saved in {saved_weight_path}")

    # Use WanDB
    use_wandb = True 
    wandb_proj_name = "zhang-reg-norm-lab-13"
    wandb_config = {
        "dataset": "coco-stuff",
        "model": "ECCV_Regression_Lab",
        "epochs": epochs,
        "lr": lr,
        "criterion": "MSE",
        "optimizer": "Adam",
        "train_num_max": train_num_max,
        "val_num_max": val_num_max,
        "pretrained": pretrained,
        "saved_weight_path": saved_weight_path
    }

    train_losses, val_losses = fit(
        model, train_loader, val_loader, saved_weight_path,
        criterion, optimizer, device, epochs,
        train_batch_size, val_batch_size,
        train_num_max, val_num_max,
        use_wandb, wandb_proj_name, wandb_config
    )

if __name__ == "__main__":
    main()