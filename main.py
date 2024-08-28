import torch
import torchvision
import os 
import pyrootutils
import wandb
from tqdm.auto import tqdm
from PIL import Image
from torchvision import transforms
from data_setup import get_data_loaders
from utils import save_checkpoint, load_checkpoint, predict_plot_image
from train import train_model
from model import VisionTransformer

if __name__ == "__main__":
    NUM_WORKERS = os.cpu_count()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    EPOCHS = 2
    BATCH_SIZE = 32
    RANDOM_SEED = 42
    torch.manual_seed(RANDOM_SEED)
    path = pyrootutils.find_root(__file__, indicator=".project-root")
    data_dir = path / "data" / "food-101-tiny"
    train_dir = data_dir / "train"
    test_dir = data_dir / "valid"
    model_dir = path / "models"
    
    
    wandb.init(
        project="VisionTransformer",
        config={
            "learning_rate": 3e-2,
            "epochs": EPOCHS,
            "architecture": "Vision Transformer",
            "dataset": "Tiny-Food101",
        }
    )
    
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    train_dataloader, test_dataloader, classes = get_data_loaders(train_dir, test_dir,
                                                              data_transform, BATCH_SIZE, NUM_WORKERS)

    vit = VisionTransformer(img_size=224, in_channels=3, patch_size=16, emb_dim=768, 
                            n_heads=12, mlp_size=3072, num_transformer_layers=12, 
                            num_classes=10, dropout=0.1).to(device)

    optimizer = torch.optim.Adam(vit.parameters(), lr=3e-2, betas=(0.9, 0.999), weight_decay=0.3)
    loss_fn = torch.nn.CrossEntropyLoss()

    # vit2 = load_checkpoint(vit, model_dir + "/vit.pt")
    # train_model(vit, train_dataloader, test_dataloader, optimizer, loss_fn, EPOCHS, device)
    

    test_image_path = test_dir / "cannoli" / "129808.jpg"
    predict_plot_image(model=vit, 
                       image_path=test_image_path,
                       class_names=classes,
                       transform=data_transform)
    
    # save_checkpoint(vit2, path / "models", "vit2.pt")