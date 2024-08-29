import matplotlib.pyplot as plt
import os 
import torch
import torchvision
from torchvision import transforms
from pathlib import Path
from typing import List, Tuple, Dict
from PIL import Image
from imageio import imread

device = "cuda" if torch.cuda.is_available() else "cpu"

def create_vit_pretrained_model(num_classes: int,
                                seed: int):
    weights = torchvision.models.ViT_B_16_Weights.DEFAULT
    transform = weights.transforms()
    model = torchvision.models.vit_b_16(weights=weights)
    
    for param in model.parameters():
        param.requires_grad = False
    
    torch.manual_seed(seed)    
    model.heads = torch.nn.Sequential(
        torch.nn.Linear(in_features=768, out_features=num_classes),
    )
    return model, transform

def save_checkpoint(model: torch.nn.Module, target_dir: str, model_name: str):
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)
    model_save_path = target_dir_path / model_name
    torch.save(model.state_dict(), model_save_path)
    
    print(f"[INFO] Saving model to: {model_save_path}")
    
def load_checkpoint(model: torch.nn.Module, model_save_path: str):
    model.load_state_dict(torch.load(model_save_path))
    print(f"[INFO] Loading model from: {model_save_path}")
    
def predict_plot_image(model: torch.nn.Module, 
                    image_path: str,
                    class_names: List[str],
                    image_size: Tuple[int, int] = (224, 224),
                    device: torch.device = device,
                    transform: torch.nn.Module = None):
    img = Image.open(image_path).convert("RGB")
    if transform is not None:
        img_transform = transform
    else:
        img_transform = torchvision.transforms.Compose(
            [transforms.Resize(image_size),
             transforms.ToTensor(),]
        )
        
    model.to(device)
    model.eval()
    with torch.inference_mode():
        img = img_transform(img).unsqueeze(0).to(device)
        pred = torch.softmax(model(img), dim=1)
        pred_class = torch.argmax(pred, dim=1)
        
        
        plt.figure()
        plt.imshow(img.clip(0, 255).squeeze().permute(1, 2, 0))
        plt.title(f"Pred: {class_names[int(pred_class.item())]} | Prob: {pred.max():.3f}" 
                  if pred_class.item() < len(class_names) else f"Pred: Unknown | Prob: {pred.max():.3f}")
        plt.axis(False)
    
# def predict(img) -> Tuple[Dict, Float]:
        