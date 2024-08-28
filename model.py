import torch
import torchvision
import torch.nn
from torchinfo import summary

class PatchEmbedding(torch.nn.Module):
    def __init__(self, in_channels, patch_size, emb_dim):
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.emb_dim = emb_dim
        
        self.patch = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, emb_dim, kernel_size=patch_size, stride=patch_size),
            torch.nn.BatchNorm2d(emb_dim),
            torch.nn.ReLU()
        )
        
        self.flatten = torch.nn.Flatten(start_dim=2, end_dim=3)
    def forward(self, x):
        x = self.patch(x)
        x = self.flatten(x)
        return x.permute(0, 2, 1)
    
class MultiHeadSelfAttention(torch.nn.Module):
    def __init__(self, emb_dim, n_heads, att_dropout):
        super().__init__()
        self.emb_size = emb_dim
        self.n_heads = n_heads
        self.att_dropout = att_dropout
        
        self.layer_norm = torch.nn.LayerNorm(emb_dim)
        self.multiheadatt = torch.nn.MultiheadAttention(emb_dim, n_heads, dropout=att_dropout, batch_first=True)

    def forward(self, x):
        x = self.layer_norm(x)
        output_att, _ = self.multiheadatt(key=x, query=x, value=x, need_weights=False)
        return output_att
    
class MLP(torch.nn.Module):
    def __init__(self, emb_dim, mlp_size, dropout):
        super().__init__()
        self.emb_size = emb_dim
        self.mlp_dim = mlp_size
        self.dropout = dropout
        
        self.layer_norm = torch.nn.LayerNorm(emb_dim)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, mlp_size),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(mlp_size, emb_dim),
            torch.nn.Dropout(dropout)
        )
    def forward(self, x):
        x = self.layer_norm(x)
        return self.mlp(x)
    
class TransformerEncoder(torch.nn.Module):
    def __init__(self, emb_dim, n_heads, mlp_size, dropout):
        super().__init__()
        self.emb_dim = emb_dim
        self.n_heads = n_heads
        self.mlp_size = mlp_size
        self.dropout = dropout
        
        self.attention = MultiHeadSelfAttention(emb_dim, n_heads, dropout)
        self.mlp = MLP(emb_dim, mlp_size, dropout)
        
    def forward(self, x):
        x = self.attention(x) + x
        x = self.mlp(x) + x
        return x
    
class VisionTransformer(torch.nn.Module):
    def __init__(self, img_size, in_channels, patch_size, emb_dim, 
                 n_heads, mlp_size, num_transformer_layers,
                 num_classes, dropout):
        super().__init__()
        self.img_size = img_size
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.emb_dim = emb_dim
        self.n_heads = n_heads
        self.mlp_size = mlp_size
        self.num_layers = num_transformer_layers
        self.num_classes = num_classes
        self.dropout = dropout
        
        self.num_patches = (img_size // patch_size) ** 2
        
        self.embedding_dropout = torch.nn.Dropout(dropout)
        self.class_embedding = torch.nn.Parameter(torch.rand(1, 1, emb_dim), requires_grad=True)
        self.position_embedding = torch.nn.Parameter(torch.rand(1, self.num_patches+1, emb_dim), requires_grad=True)
        self.patch = PatchEmbedding(in_channels, patch_size, emb_dim)
        self.transformer_encoder = torch.nn.Sequential(
            *[TransformerEncoder(emb_dim, n_heads, mlp_size, dropout) for _ in range(num_transformer_layers)])
        self.classifier = torch.nn.Linear(emb_dim, num_classes)
        
    def forward(self, x):
        batch_size = x.shape[0]
        class_token = self.class_embedding.expand(batch_size, -1, -1)
        x = self.patch(x)
        # print(x.shape)
        # print(class_token.shape)
        x = torch.cat((class_token, x), dim=1) #(batch_size, num_patches+1, emb_dim)
        x = self.position_embedding + x
        self.embedding_dropout(x)
        x = self.transformer_encoder(x)
        x = self.classifier(x[:, 0])
        return x
        
# if __name__ == "__main__":
#     random_img_tensor = torch.rand((1, 3, 224, 224))  
#     vit = VisionTransformer(img_size=224, in_channels=3, patch_size=16, emb_dim=768, 
#                             n_heads=12, mlp_size=3072, num_transformer_layers=12, 
#                             num_classes=10, dropout=0.1)
#     summary(vit, (32, 3, 224, 224), 
#             col_names=["input_size", "output_size", "num_params", "trainable"],
#             col_width=20,
#             row_settings=["var_names"])
