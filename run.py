import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from sklearn.manifold import TSNE
import random
import os
import math
import time
import argparse
import json

# ==============================================================================
# ENVIRONMENT & HARDWARE CONFIGURATION
# ==============================================================================
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

NUM_WORKERS = 0 if device.type == 'mps' else 2

def set_seed(seed=42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

set_seed(42)
os.makedirs('saved', exist_ok=True)

CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

EVAL_TF = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
])

# ==============================================================================
# SIMCLR COMPONENTS
# ==============================================================================
class SimCLRAugmentation:
    def __init__(self, image_size=32):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=3),
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
        ])
    def __call__(self, x):
        return self.transform(x), self.transform(x)

class CIFAR10SSL(Dataset):
    def __init__(self, root='./data', train=True):
        self.dataset = torchvision.datasets.CIFAR10(root=root, train=train, download=True)
        self.augment = SimCLRAugmentation()
    def __len__(self): return len(self.dataset)
    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        x_i, x_j = self.augment(img)
        return x_i, x_j, label

class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature
    def forward(self, z_i, z_j):
        N = z_i.shape[0]
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)
        z = torch.cat([z_i, z_j], dim=0)
        sim = torch.mm(z, z.T) / self.temperature
        mask = torch.eye(2 * N, dtype=torch.bool, device=z.device)
        sim = sim.masked_fill(mask, float('-inf'))
        labels = torch.cat([torch.arange(N, 2*N), torch.arange(0, N)]).to(z.device)
        return F.cross_entropy(sim, labels)

class SimCLR(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = torchvision.models.resnet18(weights=None)
        resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        resnet.maxpool = nn.Identity()
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])
        self.projector = nn.Sequential(
            nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 128)
        )
    def forward(self, x_i, x_j):
        h_i = torch.flatten(self.encoder(x_i), 1)
        h_j = torch.flatten(self.encoder(x_j), 1)
        return self.projector(h_i), self.projector(h_j), h_i, h_j

# ==============================================================================
# DINO COMPONENTS
# ==============================================================================
class DINOAugmentation:
    def __init__(self, image_size=32, n_local=4):
        normalize = transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
        flip_jitter = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
        ]
        self.global_transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.4, 1.0)),
            *flip_jitter,
            transforms.ToTensor(), normalize
        ])
        self.local_transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.05, 0.4)),
            *flip_jitter,
            transforms.ToTensor(), normalize
        ])
        self.n_local = n_local

    def __call__(self, img):
        global1 = self.global_transform(img)
        global2 = self.global_transform(img)
        locals_ = [self.local_transform(img) for _ in range(self.n_local)]
        return [global1, global2] + locals_

class CIFAR10DINO(Dataset):
    def __init__(self, root='./data', train=True, n_local=4):
        self.dataset = torchvision.datasets.CIFAR10(root=root, train=train, download=True)
        self.augment = DINOAugmentation(n_local=n_local)
    def __len__(self): return len(self.dataset)
    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        return self.augment(img), label

class DINOHead(nn.Module):
    def __init__(self, in_dim=192, hidden_dim=512, out_dim=256, n_layers=3):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden_dim), nn.GELU()]
        for _ in range(n_layers - 2):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.GELU()]
        layers.append(nn.Linear(hidden_dim, out_dim, bias=False))
        self.mlp = nn.Sequential(*layers)
        self.last_layer = nn.utils.weight_norm(nn.Linear(out_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)

    def forward(self, x):
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
        return self.last_layer(x)

import timm
def build_dino_model(out_dim=256):
    vit = timm.create_model('vit_tiny_patch16_224', pretrained=False,
                             img_size=32, patch_size=4, num_classes=0)
    embed_dim = vit.embed_dim
    head = DINOHead(in_dim=embed_dim, out_dim=out_dim)
    return vit, head

class DINOLoss(nn.Module):
    def __init__(self, out_dim=256, teacher_temp=0.04, student_temp=0.1, center_momentum=0.9, no_centering=False):
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.center_momentum = center_momentum
        self.no_centering = no_centering
        self.register_buffer('center', torch.zeros(1, out_dim))

    def forward(self, student_out, teacher_out):
        s_probs = [F.log_softmax(s / self.student_temp, dim=-1) for s in student_out]
        if self.no_centering:
            t_probs = [F.softmax(t / self.teacher_temp, dim=-1).detach() for t in teacher_out]
        else:
            t_probs = [F.softmax((t - self.center) / self.teacher_temp, dim=-1).detach() for t in teacher_out]

        total_loss = 0
        n_loss_terms = 0
        for t_idx, t_prob in enumerate(t_probs):
            for s_idx, s_log_prob in enumerate(s_probs):
                if s_idx == t_idx:
                    continue
                loss = -(t_prob * s_log_prob).sum(dim=-1).mean()
                total_loss += loss
                n_loss_terms += 1

        total_loss /= n_loss_terms
        if not self.no_centering:
            self.update_center(torch.stack(teacher_out).mean(dim=0))
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_mean):
        self.center = self.center * self.center_momentum + teacher_mean * (1 - self.center_momentum)

# ==============================================================================
# MAE COMPONENTS
# ==============================================================================
class PatchEmbed(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_ch=3, embed_dim=192):
        super().__init__()
        self.n_patches = (img_size // patch_size) ** 2
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x

def get_2d_sincos_pos_embed(embed_dim, grid_size):
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid_w, grid_h = np.meshgrid(grid_w, grid_h)

    def sincos_1d(pos, dim):
        omega = 1.0 / (10000 ** (np.arange(0, dim, 2) / dim))
        out = pos.reshape(-1, 1) * omega.reshape(1, -1)
        return np.concatenate([np.sin(out), np.cos(out)], axis=1)

    half = embed_dim // 2
    emb = np.concatenate([sincos_1d(grid_h.flatten(), half),
                           sincos_1d(grid_w.flatten(), half)], axis=1)
    return torch.tensor(emb, dtype=torch.float32)

class MAEEncoder(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_ch=3,
                 embed_dim=192, depth=6, num_heads=3, mlp_ratio=4.0,
                 mask_ratio=0.75):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.patch_embed = PatchEmbed(img_size, patch_size, in_ch, embed_dim)

        pos_embed = get_2d_sincos_pos_embed(embed_dim, img_size // patch_size)
        self.register_buffer('pos_embed', pos_embed.unsqueeze(0))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=0.0, activation='gelu',
            batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.embed_dim = embed_dim

    def random_masking(self, x):
        N, L, D = x.shape
        n_keep = int(L * (1 - self.mask_ratio))

        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = noise.argsort(dim=1)
        ids_restore = ids_shuffle.argsort(dim=1)

        ids_keep = ids_shuffle[:, :n_keep]
        x_visible = torch.gather(x, 1, ids_keep.unsqueeze(-1).expand(-1, -1, D))

        mask = torch.ones(N, L, device=x.device)
        mask[:, :n_keep] = 0
        mask = torch.gather(mask, 1, ids_restore)

        return x_visible, mask, ids_restore

    def forward(self, x):
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x_vis, mask, ids_restore = self.random_masking(x)
        x_vis = self.norm(self.transformer(x_vis))
        return x_vis, mask, ids_restore

class MAEDecoder(nn.Module):
    def __init__(self, n_patches, patch_size=4, in_ch=3,
                 encoder_dim=192, decoder_dim=128,
                 depth=4, num_heads=4, mlp_ratio=4.0):
        super().__init__()
        patch_pixels = patch_size * patch_size * in_ch
        grid_size = int(math.sqrt(n_patches))

        self.embed = nn.Linear(encoder_dim, decoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))

        pos_embed = get_2d_sincos_pos_embed(decoder_dim, grid_size)
        self.register_buffer('pos_embed', pos_embed.unsqueeze(0))

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=decoder_dim, nhead=num_heads,
            dim_feedforward=int(decoder_dim * mlp_ratio),
            dropout=0.0, activation='gelu',
            batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(decoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(decoder_dim)
        self.pred = nn.Linear(decoder_dim, patch_pixels)

        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def forward(self, x_vis, ids_restore):
        N = x_vis.size(0)
        x = self.embed(x_vis)

        n_masked = ids_restore.size(1) - x.size(1)
        mask_tokens = self.mask_token.expand(N, n_masked, -1)
        x_full = torch.cat([x, mask_tokens], dim=1)
        x_full = torch.gather(
            x_full, 1,
            ids_restore.unsqueeze(-1).expand(-1, -1, x_full.size(-1))
        )

        x_full = x_full + self.pos_embed
        x_full = self.norm(self.transformer(x_full))
        return self.pred(x_full)

class MAE(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_ch=3,
                 encoder_dim=192, encoder_depth=6, encoder_heads=3,
                 decoder_dim=128, decoder_depth=4, decoder_heads=4,
                 mask_ratio=0.75, norm_pix_loss=True):
        super().__init__()
        self.patch_size = patch_size
        self.in_ch = in_ch
        self.norm_pix_loss = norm_pix_loss

        self.encoder = MAEEncoder(
            img_size, patch_size, in_ch,
            encoder_dim, encoder_depth, encoder_heads,
            mask_ratio=mask_ratio
        )
        n_patches = self.encoder.patch_embed.n_patches
        self.decoder = MAEDecoder(
            n_patches, patch_size, in_ch,
            encoder_dim, decoder_dim, decoder_depth, decoder_heads
        )

    def patchify(self, imgs):
        p = self.patch_size
        h = w = imgs.shape[2] // p
        x = imgs.reshape(imgs.shape[0], self.in_ch, h, p, w, p)
        x = x.permute(0, 2, 4, 3, 5, 1)
        return x.reshape(imgs.shape[0], h * w, p * p * self.in_ch)

    def forward(self, imgs):
        x_vis, mask, ids_restore = self.encoder(imgs)
        pred = self.decoder(x_vis, ids_restore)

        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var  = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1e-6).sqrt()

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum()
        return loss, pred, mask

# ==============================================================================
# LOGGING UTILITY
# ==============================================================================
def log_result(config_name, metrics):
    filepath = 'saved/results.json'
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception:
            data = {}
    else:
        data = {}
    
    # Merge or initialize config dictionary
    if config_name not in data:
        data[config_name] = {}
    
    for k, v in metrics.items():
        data[config_name][k] = v
        
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    print(f"Logged results for '{config_name}' in {filepath}")

# ==============================================================================
# VISUALIZATION UTILITIES
# ==============================================================================
def plot_loss_curves():
    filepath = 'saved/results.json'
    if not os.path.exists(filepath):
        print(f"No results file at {filepath} to plot loss curves.")
        return
    
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    # We plot side-by-side because scales are very different
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # SimCLR
    if 'simclr' in data and 'train_loss' in data['simclr']:
        axes[0].plot(data['simclr']['train_loss'], marker='o', color='royalblue')
        axes[0].set_title('SimCLR Training Loss')
        axes[0].set_ylabel('NT-Xent Loss')
        axes[0].grid(True)
    else:
        axes[0].text(0.5, 0.5, 'No SimCLR Data', ha='center', va='center')
        
    # DINO
    if 'dino' in data and 'train_loss' in data['dino']:
        axes[1].plot(data['dino']['train_loss'], marker='s', color='darkorange')
        axes[1].set_title('DINO Training Loss')
        axes[1].set_ylabel('Cross-Entropy')
        axes[1].grid(True)
    elif 'dino_default' in data and 'train_loss' in data['dino_default']:
        axes[1].plot(data['dino_default']['train_loss'], marker='s', color='darkorange')
        axes[1].set_title('DINO Training Loss')
        axes[1].set_ylabel('Cross-Entropy')
        axes[1].grid(True)
    else:
        axes[1].text(0.5, 0.5, 'No DINO Data', ha='center', va='center')
        
    # MAE
    if 'mae' in data and 'train_loss' in data['mae']:
        axes[2].plot(data['mae']['train_loss'], marker='^', color='forestgreen')
        axes[2].set_title('MAE Training Loss')
        axes[2].set_ylabel('MSE Loss')
        axes[2].grid(True)
    elif 'mae_default' in data and 'train_loss' in data['mae_default']:
        axes[2].plot(data['mae_default']['train_loss'], marker='^', color='forestgreen')
        axes[2].set_title('MAE Training Loss')
        axes[2].set_ylabel('MSE Loss')
        axes[2].grid(True)
    else:
        axes[2].text(0.5, 0.5, 'No MAE Data', ha='center', va='center')
        
    for ax in axes:
        ax.set_xlabel('Epochs')
        
    plt.suptitle('Self-Supervised Learning Training Loss Comparison')
    plt.tight_layout()
    out_path = 'saved/loss_curves.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved loss curves comparison to {out_path}")

def plot_dino_center_norm():
    filepath = 'saved/results.json'
    if not os.path.exists(filepath):
        return
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    fig, ax = plt.subplots(figsize=(7, 4))
    legend_added = False
    
    for key, color, name in [('dino', 'darkorange', 'DINO (Default)'),
                              ('dino_default', 'darkorange', 'DINO (Default)'),
                              ('dino_no_centering', 'crimson', 'DINO (No Centering)')]:
        if key in data and 'center_norms' in data[key]:
            ax.plot(data[key]['center_norms'], marker='o', color=color, label=name)
            legend_added = True
            
    if legend_added:
        ax.set_title('DINO Center Vector Norm Across Epochs')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Center Norm')
        ax.grid(True)
        ax.legend()
        plt.tight_layout()
        plt.savefig('saved/dino_center_norm.png', dpi=150)
        plt.close()
        print("Saved DINO center norm tracking plot to saved/dino_center_norm.png")

# ==============================================================================
# MAIN COMMAND LINE ROUTINE
# ==============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Assignment 3 SSL CLI Tools")
    parser.add_argument("--model", type=str, required=True, choices=["simclr", "dino", "mae"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--linear", action="store_true")
    parser.add_argument("--weights", type=str, default=None)
    parser.add_argument("--no-centering", action="store_true")
    parser.add_argument("--n-local", type=int, default=4)
    parser.add_argument("--mask-ratio", type=float, default=0.75)
    
    args = parser.parse_args()
    
    # Make sure we have saved folder
    os.makedirs('saved', exist_ok=True)
    
    # --------------------------------------------------------------------------
    # SIMCLR MODEL EXECUTION
    # --------------------------------------------------------------------------
    if args.model == "simclr":
        simclr_model = SimCLR().to(device)
        
        if args.train:
            print("--- Training SimCLR ---")
            train_dataset = CIFAR10SSL()
            train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True,
                                       num_workers=NUM_WORKERS, drop_last=True)
            criterion = NTXentLoss(temperature=0.5)
            optimizer = torch.optim.Adam(simclr_model.parameters(), lr=3e-4, weight_decay=1e-4)
            
            losses = []
            epoch_times = []
            total_start = time.time()
            
            for epoch in range(args.epochs):
                simclr_model.train()
                ep_losses = []
                t0 = time.time()
                for x_i, x_j, _ in tqdm(train_loader, desc=f'SimCLR {epoch+1}/{args.epochs}'):
                    x_i, x_j = x_i.to(device), x_j.to(device)
                    z_i, z_j, _, _ = simclr_model(x_i, x_j)
                    loss = criterion(z_i, z_j)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    ep_losses.append(loss.item())
                elapsed = time.time() - t0
                epoch_times.append(elapsed)
                avg_loss = np.mean(ep_losses)
                losses.append(avg_loss)
                print(f'Epoch {epoch+1:02d} | Loss: {avg_loss:.4f} | Time: {elapsed:.1f}s')
                
            total_time = time.time() - total_start
            torch.save(simclr_model.state_dict(), 'saved/simclr.pt')
            
            log_result('simclr', {
                'train_loss': losses,
                'epoch_times': epoch_times,
                'total_time_min': total_time / 60
            })
            plot_loss_curves()
            
        elif args.evaluate and args.linear:
            print("--- Linear Evaluating SimCLR ---")
            w_path = args.weights if args.weights else 'saved/simclr.pt'
            if not os.path.exists(w_path):
                raise FileNotFoundError(f"Weights path {w_path} not found. Please train model or specify correct path.")
            
            simclr_model.load_state_dict(torch.load(w_path, map_location=device))
            simclr_model.eval()
            for p in simclr_model.encoder.parameters():
                p.requires_grad = False
                
            clf = nn.Linear(512, 10).to(device)
            optimizer_clf = torch.optim.Adam(clf.parameters(), lr=1e-3)
            
            train_lbl = torchvision.datasets.CIFAR10('./data', train=True, download=True, transform=EVAL_TF)
            test_lbl  = torchvision.datasets.CIFAR10('./data', train=False, download=True, transform=EVAL_TF)
            trl = DataLoader(train_lbl, batch_size=256, shuffle=True, num_workers=NUM_WORKERS)
            tel = DataLoader(test_lbl,  batch_size=256, shuffle=False, num_workers=NUM_WORKERS)
            
            for epoch in range(10):
                clf.train()
                correct, total = 0, 0
                for imgs, labels in tqdm(trl, desc=f'SimCLR Linear Eval {epoch+1}/10'):
                    imgs, labels = imgs.to(device), labels.to(device)
                    with torch.no_grad():
                        h = torch.flatten(simclr_model.encoder(imgs), 1)
                    logits = clf(h)
                    loss = F.cross_entropy(logits, labels)
                    optimizer_clf.zero_grad()
                    loss.backward()
                    optimizer_clf.step()
                    correct += (logits.argmax(dim=1) == labels).sum().item()
                    total += labels.size(0)
                print(f'Epoch {epoch+1} | Train Acc: {100 * correct/total:.2f}%')
                
            clf.eval()
            correct, total = 0, 0
            simclr_embeddings, simclr_labels = [], []
            with torch.no_grad():
                for imgs, labels in tel:
                    imgs, labels = imgs.to(device), labels.to(device)
                    h = torch.flatten(simclr_model.encoder(imgs), 1)
                    logits = clf(h)
                    correct += (logits.argmax(dim=1) == labels).sum().item()
                    total += labels.size(0)
                    simclr_embeddings.append(h.cpu())
                    simclr_labels.append(labels.cpu())
            
            test_acc = 100 * correct / total
            print(f'\n✅ SimCLR Linear Eval Test Accuracy: {test_acc:.2f}%')
            
            simclr_embeddings = torch.cat(simclr_embeddings, dim=0)
            simclr_labels = torch.cat(simclr_labels, dim=0)
            torch.save(simclr_embeddings, 'saved/simclr_embeddings.pt')
            torch.save(simclr_labels, 'saved/simclr_labels.pt')
            
            log_result('simclr', {
                'linear_eval_acc': test_acc
            })
            
    # --------------------------------------------------------------------------
    # DINO MODEL EXECUTION
    # --------------------------------------------------------------------------
    elif args.model == "dino":
        student_vit, student_head = build_dino_model()
        teacher_vit, teacher_head = build_dino_model()
        
        student_vit, student_head = student_vit.to(device), student_head.to(device)
        teacher_vit, teacher_head = teacher_vit.to(device), teacher_head.to(device)
        
        if args.train:
            print("--- Training DINO ---")
            teacher_vit.load_state_dict(student_vit.state_dict())
            teacher_head.load_state_dict(student_head.state_dict())
            for p in teacher_vit.parameters(): p.requires_grad = False
            for p in teacher_head.parameters(): p.requires_grad = False
            
            dino_dataset = CIFAR10DINO(n_local=args.n_local)
            
            def dino_collate(batch):
                crops_list, labels = zip(*batch)
                n_views = len(crops_list[0])
                stacked = [torch.stack([crops_list[i][v] for i in range(len(crops_list))]) for v in range(n_views)]
                return stacked, torch.tensor(labels)
            
            dino_loader = DataLoader(dino_dataset, batch_size=64, shuffle=True,
                                      num_workers=NUM_WORKERS, drop_last=True, collate_fn=dino_collate)
            
            dino_loss_fn = DINOLoss(out_dim=256, no_centering=args.no_centering).to(device)
            optimizer_d = torch.optim.AdamW(
                list(student_vit.parameters()) + list(student_head.parameters()),
                lr=5e-4, weight_decay=0.04
            )
            
            losses = []
            center_norms = []
            epoch_times = []
            total_start = time.time()
            EMA_M = 0.996
            
            # Name configurations for ablations
            config_name = "dino"
            if args.no_centering:
                config_name = "dino_no_centering"
            elif args.n_local == 0:
                config_name = "dino_no_local"
                
            for epoch in range(args.epochs):
                student_vit.train(); student_head.train()
                ep_losses = []
                t0 = time.time()
                
                for crops, _ in tqdm(dino_loader, desc=f'{config_name.upper()} {epoch+1}/{args.epochs}'):
                    crops = [c.to(device) for c in crops]
                    student_out = [student_head(student_vit(c)) for c in crops]
                    
                    with torch.no_grad():
                        teacher_out = [teacher_head(teacher_vit(crops[0])),
                                       teacher_head(teacher_vit(crops[1]))]
                                       
                    loss = dino_loss_fn(student_out, teacher_out)
                    optimizer_d.zero_grad()
                    loss.backward()
                    optimizer_d.step()
                    
                    with torch.no_grad():
                        for s_p, t_p in zip(student_vit.parameters(), teacher_vit.parameters()):
                            t_p.data = EMA_M * t_p.data + (1 - EMA_M) * s_p.data
                        for s_p, t_p in zip(student_head.parameters(), teacher_head.parameters()):
                            t_p.data = EMA_M * t_p.data + (1 - EMA_M) * s_p.data
                            
                    ep_losses.append(loss.item())
                    
                elapsed = time.time() - t0
                epoch_times.append(elapsed)
                avg_loss = np.mean(ep_losses)
                losses.append(avg_loss)
                current_norm = dino_loss_fn.center.norm().item()
                center_norms.append(current_norm)
                
                print(f'Epoch {epoch+1:02d} | Loss: {avg_loss:.4f} | Center norm: {current_norm:.4f} | Time: {elapsed:.1f}s')
                
            total_time = time.time() - total_start
            save_path = f'saved/{config_name}.pt'
            torch.save({
                'student_vit': student_vit.state_dict(),
                'student_head': student_head.state_dict()
            }, save_path)
            
            log_result(config_name, {
                'train_loss': losses,
                'center_norms': center_norms,
                'epoch_times': epoch_times,
                'total_time_min': total_time / 60
            })
            
            plot_loss_curves()
            plot_dino_center_norm()
            
            # If default DINO completes training, generate attention maps
            if config_name == "dino":
                print("--- Generating DINO Attention Maps (10 images) ---")
                student_vit.eval()
                attentions = {}
                attn_module = student_vit.blocks[-1].attn
                
                def _patched_attn_forward(x, **kwargs):
                    B, N, C = x.shape
                    qkv = attn_module.qkv(x).reshape(B, N, 3, attn_module.num_heads, C // attn_module.num_heads).permute(2, 0, 3, 1, 4)
                    q, k, v = qkv.unbind(0)
                    attn_w = (q @ k.transpose(-2, -1)) * attn_module.scale
                    attn_w = attn_w.softmax(dim=-1)
                    attentions['last'] = attn_w.detach()
                    attn_w = attn_module.attn_drop(attn_w)
                    x = (attn_w @ v).transpose(1, 2).reshape(B, N, C)
                    x = attn_module.proj(x)
                    x = attn_module.proj_drop(x)
                    return x
                
                attn_module.forward = _patched_attn_forward
                raw_test = torchvision.datasets.CIFAR10('./data', train=False, transform=EVAL_TF)
                img_loader = DataLoader(raw_test, batch_size=1, shuffle=True)
                
                n_heads = attn_module.num_heads
                patch_h = patch_w = 8
                
                fig, axes = plt.subplots(10, n_heads + 1, figsize=(2*(n_heads+1), 20))
                img_mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
                img_std  = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)
                
                sample_iter = iter(img_loader)
                for row in range(10):
                    img_tensor, label = next(sample_iter)
                    img_tensor = img_tensor.to(device)
                    
                    with torch.no_grad():
                        _ = student_vit(img_tensor)
                        
                    attn = attentions['last']
                    cls_attn = attn[0, :, 0, 1:]
                    
                    img_disp = torch.clamp(img_tensor[0].cpu() * img_std + img_mean, 0, 1).permute(1,2,0).numpy()
                    axes[row][0].imshow(img_disp)
                    axes[row][0].set_title(f'{CLASSES[label.item()]}', fontsize=8)
                    axes[row][0].axis('off')
                    
                    for h in range(n_heads):
                        head_map = cls_attn[h].reshape(patch_h, patch_w).cpu().numpy()
                        head_map = (head_map - head_map.min()) / (head_map.max() - head_map.min() + 1e-8)
                        head_up = np.array(Image.fromarray((head_map * 255).astype(np.uint8)).resize((32, 32)))
                        axes[row][h+1].imshow(img_disp, alpha=0.4)
                        axes[row][h+1].imshow(head_up, cmap='hot', alpha=0.7, vmin=0, vmax=255)
                        if row == 0:
                            axes[row][h+1].set_title(f'Head {h+1}', fontsize=8)
                        axes[row][h+1].axis('off')
                
                plt.suptitle('DINO Self-Attention Maps (10 Images)\n[CLS] Token to Patch Attention Maps', fontsize=12, y=1.01)
                plt.tight_layout()
                plt.savefig('saved/dino_attention_grid.png', dpi=150, bbox_inches='tight')
                plt.close()
                print("Saved DINO attention grid to saved/dino_attention_grid.png")
                
        elif args.evaluate and args.linear:
            print("--- Linear Evaluating DINO ---")
            w_path = args.weights if args.weights else 'saved/dino.pt'
            if not os.path.exists(w_path):
                raise FileNotFoundError(f"Weights path {w_path} not found.")
                
            ckpt = torch.load(w_path, map_location=device)
            student_vit.load_state_dict(ckpt['student_vit'])
            student_vit.eval()
            for p in student_vit.parameters(): p.requires_grad = False
            
            clf_dino = nn.Linear(student_vit.embed_dim, 10).to(device)
            optimizer_clf = torch.optim.Adam(clf_dino.parameters(), lr=1e-3)
            
            train_lbl = torchvision.datasets.CIFAR10('./data', train=True, download=True, transform=EVAL_TF)
            test_lbl  = torchvision.datasets.CIFAR10('./data', train=False, download=True, transform=EVAL_TF)
            trl = DataLoader(train_lbl, batch_size=256, shuffle=True, num_workers=NUM_WORKERS)
            tel = DataLoader(test_lbl,  batch_size=256, shuffle=False, num_workers=NUM_WORKERS)
            
            for epoch in range(10):
                clf_dino.train()
                correct, total = 0, 0
                for imgs, labels in tqdm(trl, desc=f'DINO Linear Eval {epoch+1}/10'):
                    imgs, labels = imgs.to(device), labels.to(device)
                    with torch.no_grad():
                        h = student_vit(imgs)
                    logits = clf_dino(h)
                    loss = F.cross_entropy(logits, labels)
                    optimizer_clf.zero_grad()
                    loss.backward()
                    optimizer_clf.step()
                    correct += (logits.argmax(dim=1) == labels).sum().item()
                    total += labels.size(0)
                print(f'Epoch {epoch+1} | Train Acc: {100 * correct/total:.2f}%')
                
            clf_dino.eval()
            correct, total = 0, 0
            dino_embeddings, dino_labels = [], []
            with torch.no_grad():
                for imgs, labels in tel:
                    imgs, labels = imgs.to(device), labels.to(device)
                    h = student_vit(imgs)
                    logits = clf_dino(h)
                    correct += (logits.argmax(dim=1) == labels).sum().item()
                    total += labels.size(0)
                    dino_embeddings.append(h.cpu())
                    dino_labels.append(labels.cpu())
                    
            test_acc = 100 * correct / total
            print(f'\n✅ DINO Linear Eval Test Accuracy: {test_acc:.2f}%')
            
            dino_embeddings = torch.cat(dino_embeddings, dim=0)
            dino_labels = torch.cat(dino_labels, dim=0)
            torch.save(dino_embeddings, 'saved/dino_embeddings.pt')
            torch.save(dino_labels, 'saved/dino_labels.pt')
            
            # Determine configuration key name based on weights filename
            config_name = "dino"
            if "no_centering" in w_path:
                config_name = "dino_no_centering"
            elif "no_local" in w_path:
                config_name = "dino_no_local"
                
            log_result(config_name, {
                'linear_eval_acc': test_acc
            })
            
            # If default DINO, generate attention maps
            if config_name == "dino":
                print("--- Generating DINO Attention Maps (10 images) ---")
                student_vit.eval()
                attentions = {}
                attn_module = student_vit.blocks[-1].attn
                
                def _patched_attn_forward(x, **kwargs):
                    B, N, C = x.shape
                    qkv = attn_module.qkv(x).reshape(B, N, 3, attn_module.num_heads, C // attn_module.num_heads).permute(2, 0, 3, 1, 4)
                    q, k, v = qkv.unbind(0)
                    attn_w = (q @ k.transpose(-2, -1)) * attn_module.scale
                    attn_w = attn_w.softmax(dim=-1)
                    attentions['last'] = attn_w.detach()
                    attn_w = attn_module.attn_drop(attn_w)
                    x = (attn_w @ v).transpose(1, 2).reshape(B, N, C)
                    x = attn_module.proj(x)
                    x = attn_module.proj_drop(x)
                    return x
                
                attn_module.forward = _patched_attn_forward
                raw_test = torchvision.datasets.CIFAR10('./data', train=False, transform=EVAL_TF)
                img_loader = DataLoader(raw_test, batch_size=1, shuffle=True)
                
                n_heads = attn_module.num_heads
                patch_h = patch_w = 8
                
                fig, axes = plt.subplots(10, n_heads + 1, figsize=(2*(n_heads+1), 20))
                img_mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
                img_std  = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)
                
                sample_iter = iter(img_loader)
                for row in range(10):
                    img_tensor, label = next(sample_iter)
                    img_tensor = img_tensor.to(device)
                    
                    with torch.no_grad():
                        _ = student_vit(img_tensor)
                        
                    attn = attentions['last']
                    cls_attn = attn[0, :, 0, 1:]
                    
                    img_disp = torch.clamp(img_tensor[0].cpu() * img_std + img_mean, 0, 1).permute(1,2,0).numpy()
                    axes[row][0].imshow(img_disp)
                    axes[row][0].set_title(f'{CLASSES[label.item()]}', fontsize=8)
                    axes[row][0].axis('off')
                    
                    for h in range(n_heads):
                        head_map = cls_attn[h].reshape(patch_h, patch_w).cpu().numpy()
                        head_map = (head_map - head_map.min()) / (head_map.max() - head_map.min() + 1e-8)
                        head_up = np.array(Image.fromarray((head_map * 255).astype(np.uint8)).resize((32, 32)))
                        axes[row][h+1].imshow(img_disp, alpha=0.4)
                        axes[row][h+1].imshow(head_up, cmap='hot', alpha=0.7, vmin=0, vmax=255)
                        if row == 0:
                            axes[row][h+1].set_title(f'Head {h+1}', fontsize=8)
                        axes[row][h+1].axis('off')
                
                plt.suptitle('DINO Self-Attention Maps (10 Images)\n[CLS] Token to Patch Attention Maps', fontsize=12, y=1.01)
                plt.tight_layout()
                plt.savefig('saved/dino_attention_grid.png', dpi=150, bbox_inches='tight')
                plt.close()
                print("Saved DINO attention grid to saved/dino_attention_grid.png")

    # --------------------------------------------------------------------------
    # MAE MODEL EXECUTION
    # --------------------------------------------------------------------------
    elif args.model == "mae":
        mae_model = MAE(
            img_size=32, patch_size=4, in_ch=3,
            encoder_dim=192, encoder_depth=6, encoder_heads=3,
            decoder_dim=128, decoder_depth=4, decoder_heads=4,
            mask_ratio=args.mask_ratio, norm_pix_loss=True
        ).to(device)
        
        mae_mean = [0.4914, 0.4822, 0.4465]
        mae_std  = [0.247,  0.243,  0.261]
        
        if args.train:
            print(f"--- Training MAE (Mask Ratio: {args.mask_ratio}) ---")
            mae_train_tf = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mae_mean, mae_std)
            ])
            mae_train_ds = torchvision.datasets.CIFAR10('./data', train=True, transform=mae_train_tf, download=True)
            mae_loader = DataLoader(mae_train_ds, batch_size=128, shuffle=True,
                                     num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
            
            optimizer_m = torch.optim.AdamW(mae_model.parameters(), lr=1.5e-4, weight_decay=0.05, betas=(0.9, 0.95))
            scheduler_m = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_m, T_max=args.epochs)
            
            losses = []
            epoch_times = []
            total_start = time.time()
            
            config_name = "mae"
            if args.mask_ratio == 0.25:
                config_name = "mae_mask_0.25"
            elif args.mask_ratio == 0.50:
                config_name = "mae_mask_0.50"
                
            for epoch in range(args.epochs):
                mae_model.train()
                ep_losses = []
                t0 = time.time()
                for imgs, _ in tqdm(mae_loader, desc=f'{config_name.upper()} {epoch+1}/{args.epochs}'):
                    imgs = imgs.to(device)
                    loss, _, _ = mae_model(imgs)
                    optimizer_m.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(mae_model.parameters(), max_norm=1.0)
                    optimizer_m.step()
                    ep_losses.append(loss.item())
                scheduler_m.step()
                
                elapsed = time.time() - t0
                epoch_times.append(elapsed)
                avg_loss = np.mean(ep_losses)
                losses.append(avg_loss)
                print(f'Epoch {epoch+1:02d} | Recon Loss: {avg_loss:.4f} | Time: {elapsed:.1f}s')
                
            total_time = time.time() - total_start
            
            # Save encoder weights specifically as required
            save_path = f'saved/{config_name}_encoder.pt'
            torch.save(mae_model.encoder.state_dict(), save_path)
            
            log_result(config_name, {
                'train_loss': losses,
                'epoch_times': epoch_times,
                'total_time_min': total_time / 60
            })
            
            plot_loss_curves()
            
            # If default MAE, generate reconstruction visualization
            if args.mask_ratio == 0.75:
                print("--- Generating MAE Reconstruction Grid ---")
                mae_model.eval()
                mae_test_tf = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mae_mean, mae_std)
                ])
                viz_loader = DataLoader(
                    torchvision.datasets.CIFAR10('./data', train=False, transform=mae_test_tf),
                    batch_size=8, shuffle=True
                )
                imgs_viz, _ = next(iter(viz_loader))
                imgs_viz = imgs_viz.to(device)
                
                with torch.no_grad():
                    loss_viz, pred, mask = mae_model(imgs_viz)
                    
                p = mae_model.patch_size
                h_g = w_g = 8
                
                def unpatchify(patches, p, h, w, in_ch=3):
                    N = patches.size(0)
                    x = patches.reshape(N, h, w, p, p, in_ch)
                    x = x.permute(0, 5, 1, 3, 2, 4)
                    return x.reshape(N, in_ch, h*p, w*p)
                
                pred_imgs = unpatchify(pred.cpu(), p, h_g, w_g)
                mean_t = torch.tensor(mae_mean).view(3, 1, 1)
                std_t  = torch.tensor(mae_std).view(3, 1, 1)
                orig_np = (imgs_viz.cpu() * std_t + mean_t).clamp(0, 1).permute(0, 2, 3, 1).numpy()
                pred_np = (pred_imgs * std_t + mean_t).clamp(0, 1).permute(0, 2, 3, 1).numpy()
                
                mask_exp = mask.cpu().view(-1, h_g, w_g).unsqueeze(1)
                mask_exp = mask_exp.repeat_interleave(p, dim=2).repeat_interleave(p, dim=3)
                mask_np  = mask_exp.expand(-1, 3, -1, -1).permute(0, 2, 3, 1).numpy()
                
                masked_np = orig_np.copy()
                masked_np[mask_np.astype(bool)] = 0.5
                
                N_show = 8
                fig, axes = plt.subplots(3, N_show, figsize=(2 * N_show, 6))
                for row, (imgs_row, title) in enumerate(zip([orig_np, masked_np, pred_np],
                                                             ['Original', 'Masked (75%)', 'Reconstructed'])):
                    axes[row, 0].set_ylabel(title, fontsize=10)
                    for col in range(N_show):
                        axes[row, col].imshow(imgs_row[col])
                        axes[row, col].axis('off')
                
                plt.suptitle('MAE Reconstruction Comparison Grid', fontsize=12, y=1.02)
                plt.tight_layout()
                plt.savefig('saved/mae_reconstruction.png', dpi=150, bbox_inches='tight')
                plt.close()
                print("Saved MAE reconstruction visualization to saved/mae_reconstruction.png")
                
        elif args.evaluate and args.linear:
            print("--- Linear Evaluating MAE ---")
            w_path = args.weights if args.weights else 'saved/mae_encoder.pt'
            if not os.path.exists(w_path):
                raise FileNotFoundError(f"Weights path {w_path} not found.")
                
            mae_model.encoder.load_state_dict(torch.load(w_path, map_location=device))
            mae_model.encoder.eval()
            for p in mae_model.encoder.parameters(): p.requires_grad = False
            mae_model.encoder.mask_ratio = 0.0 # disable masking for evaluation
            
            clf_mae = nn.Linear(mae_model.encoder.embed_dim, 10).to(device)
            optimizer_clf = torch.optim.Adam(clf_mae.parameters(), lr=1e-3)
            
            mae_test_tf = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mae_mean, mae_std)
            ])
            mae_clf_train_tf = transforms.Compose([
                transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
                transforms.ToTensor(), transforms.Normalize(mae_mean, mae_std)
            ])
            
            train_lbl = torchvision.datasets.CIFAR10('./data', train=True, transform=mae_clf_train_tf, download=True)
            test_lbl  = torchvision.datasets.CIFAR10('./data', train=False, transform=mae_test_tf, download=True)
            trl = DataLoader(train_lbl, batch_size=256, shuffle=True, num_workers=NUM_WORKERS)
            tel = DataLoader(test_lbl,  batch_size=256, shuffle=False, num_workers=NUM_WORKERS)
            
            for epoch in range(10):
                clf_mae.train()
                correct, total = 0, 0
                for imgs, labels in tqdm(trl, desc=f'MAE Linear Eval {epoch+1}/10'):
                    imgs, labels = imgs.to(device), labels.to(device)
                    with torch.no_grad():
                        x_vis, _, _ = mae_model.encoder(imgs)
                        feats = x_vis.mean(dim=1)
                    logits = clf_mae(feats)
                    loss = F.cross_entropy(logits, labels)
                    optimizer_clf.zero_grad()
                    loss.backward()
                    optimizer_clf.step()
                    correct += (logits.argmax(dim=1) == labels).sum().item()
                    total += labels.size(0)
                print(f'Epoch {epoch+1} | Train Acc: {100 * correct/total:.2f}%')
                
            clf_mae.eval()
            correct, total = 0, 0
            mae_embeddings, mae_labels_list = [], []
            with torch.no_grad():
                for imgs, labels in tel:
                    imgs, labels = imgs.to(device), labels.to(device)
                    x_vis, _, _ = mae_model.encoder(imgs)
                    feats = x_vis.mean(dim=1)
                    logits = clf_mae(feats)
                    correct += (logits.argmax(dim=1) == labels).sum().item()
                    total += labels.size(0)
                    mae_embeddings.append(feats.cpu())
                    mae_labels_list.append(labels.cpu())
            
            test_acc = 100 * correct / total
            print(f'\n✅ MAE Linear Eval Test Accuracy: {test_acc:.2f}%')
            
            mae_embeddings = torch.cat(mae_embeddings, dim=0)
            mae_labels_list = torch.cat(mae_labels_list, dim=0)
            torch.save(mae_embeddings, 'saved/mae_embeddings.pt')
            torch.save(mae_labels_list, 'saved/mae_labels.pt')
            
            config_name = "mae"
            if "0.25" in w_path:
                config_name = "mae_mask_0.25"
            elif "0.50" in w_path:
                config_name = "mae_mask_0.50"
                
            log_result(config_name, {
                'linear_eval_acc': test_acc
            })
            
    # --------------------------------------------------------------------------
    # MULTI-MODEL COMPARISON / T-SNE / PLOT GENERATION
    # --------------------------------------------------------------------------
    # Trigger t-SNE plot if all three sets of embeddings exist
    if (os.path.exists('saved/simclr_embeddings.pt') and 
        os.path.exists('saved/dino_embeddings.pt') and 
        os.path.exists('saved/mae_embeddings.pt')):
        print("--- All embeddings found. Running t-SNE projection comparison ---")
        simclr_emb = torch.load('saved/simclr_embeddings.pt')
        simclr_lbl = torch.load('saved/simclr_labels.pt')
        dino_emb = torch.load('saved/dino_embeddings.pt')
        dino_lbl = torch.load('saved/dino_labels.pt')
        mae_emb = torch.load('saved/mae_embeddings.pt')
        mae_lbl = torch.load('saved/mae_labels.pt')
        
        fig, axes = plt.subplots(1, 3, figsize=(21, 6))
        colors = plt.cm.tab10(np.linspace(0, 1, 10))
        
        for ax, (name, emb, lbls) in zip(axes, [
            ('SimCLR (ResNet-18)', simclr_emb, simclr_lbl),
            ('DINO (ViT-Tiny)',    dino_emb,   dino_lbl),
            ('MAE (ViT)',          mae_emb,    mae_lbl),
        ]):
            idx = np.random.choice(len(emb), 2000, replace=False)
            proj = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(emb[idx].numpy())
            for c in range(10):
                mask_c = lbls[idx].numpy() == c
                ax.scatter(proj[mask_c,0], proj[mask_c,1], c=[colors[c]], label=CLASSES[c], alpha=0.6, s=10)
            ax.set_title(name, fontsize=12)
            ax.legend(fontsize=7, markerscale=2)
            ax.axis('off')
            
        plt.suptitle('t-SNE Projection: Representation Space Comparison (no labels in pre-training)', fontsize=13)
        plt.tight_layout()
        plt.savefig('saved/tsne_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("Saved t-SNE comparative visualization to saved/tsne_comparison.png")
