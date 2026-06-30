import argparse
import numpy as np
import os
from PIL import Image
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, roc_curve, average_precision_score
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models
import torchvision.transforms.v2 as transforms


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
scaler = torch.amp.GradScaler('cuda')


class StampNet(nn.Module):
    def __init__(self, embed_dim: int = 256, pretrained: bool = True, dropout: float = 0.15):
        super().__init__()
        # EfficientNet-B0 backbone (ImageNet-pretrained)
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.efficientnet_b0(weights=weights)
        # Get feature dimension before classifier and remove classifier
        feat_dim = backbone.classifier[1].in_features
        backbone.classifier = nn.Identity()
        self.backbone = backbone
        # Metric head: feat_dim → 512 → embed_dim
        self.head = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, embed_dim, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)      # [B, feat_dim]
        emb = self.head(feats)        # [B, embed_dim]
        # L2-normalize embeddings for cosine similarity
        emb = F.normalize(emb, p=2, dim=1)
        return emb
    


transform = transforms.Compose([
    transforms.Resize((192, 192), antialias=True),
    transforms.ToImage(),
    transforms.ToDtype(torch.float32, scale=True),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
class TestingTriplets(Dataset):
    def __init__(self, root_dir, transform=transform):
        self.root_dir = root_dir
        self.transform = transform
        # List all subdirectories that contain the three required files
        self.triplet_dirs = []
        for name in os.listdir(root_dir):
            subdir = os.path.join(root_dir, name)
            if not os.path.isdir(subdir):
                continue
            anchor_path = os.path.join(subdir, 'anchor.png')
            positive_path = os.path.join(subdir, 'positive.png')
            negative_path = os.path.join(subdir, 'negative.png')

            if all(os.path.isfile(p) for p in [anchor_path, positive_path, negative_path]):
                self.triplet_dirs.append(subdir)
        self.triplet_dirs.sort()  # for reproducibility

        if len(self.triplet_dirs) == 0:
            print(f'Warning: No valid triplet folders found in {root_dir}')

    def __len__(self):
        return len(self.triplet_dirs)

    def __getitem__(self, idx):
        triplet_dir = self.triplet_dirs[idx]
        anchor_path = os.path.join(triplet_dir, 'anchor.png')
        positive_path = os.path.join(triplet_dir, 'positive.png')
        negative_path = os.path.join(triplet_dir, 'negative.png')
        anchor = Image.open(anchor_path).convert('RGB')    # 'L' for grayscale (your saved images)
        positive = Image.open(positive_path).convert('RGB')
        negative = Image.open(negative_path).convert('RGB')

        if self.transform is not None:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)

        return anchor, positive, negative
    

def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate Object Re-Identification Model.'
    )
    parser.add_argument(
        '--path',
        type=str,
        required=True,
        help='Model path',
    )
    parser.add_argument(
        '--batch',
        type=int,
        default=64,
        help='BATCHSIZE',
    )
    parser.add_argument(
        '--directory',
        type=str,
        required=True,
        help='Testing Directory',
    )
    return parser.parse_args()


def evaluate_model(model, dataloader, device):
    model.eval()
    all_pos_scores = []
    all_neg_scores = []

    with torch.no_grad():
        for anchor, positive, negative in tqdm(dataloader, desc="Evaluating"):
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)

            a_emb = model(anchor)
            p_emb = model(positive)
            n_emb = model(negative)

            pos_scores = F.cosine_similarity(a_emb, p_emb)
            neg_scores = F.cosine_similarity(a_emb, n_emb)

            all_pos_scores.append(pos_scores.cpu())
            all_neg_scores.append(neg_scores.cpu())

    pos_scores = torch.cat(all_pos_scores).numpy()
    neg_scores = torch.cat(all_neg_scores).numpy()

    scores = np.concatenate([pos_scores, neg_scores])
    labels = np.concatenate([
        np.ones_like(pos_scores),
        np.zeros_like(neg_scores)
    ])
    return scores, labels, pos_scores, neg_scores


def evaluate_metrics(scores, labels, pos_scores, neg_scores):
    roc_auc = roc_auc_score(labels, scores)
    precision, recall, _ = precision_recall_curve(labels, scores)
    pr_auc = auc(recall, precision)
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr
    eer_index = np.nanargmin(np.absolute(fnr - fpr))
    eer = fpr[eer_index]
    mAP = average_precision_score(labels, scores)

    # Recall@1
    K = 1
    correct = 0
    total = len(pos_scores)
    for p, n in zip(pos_scores, neg_scores):
        sims = np.array([p, n])
        topk = sims.argsort()[::-1][:K]
        if 0 in topk:
            correct += 1

    recall_k = correct / total
    print("\n----- Evaluation Metrics -----")
    print(f"ROC-AUC: {roc_auc:.6f}")
    print(f"PR-AUC: {pr_auc:.6f}")
    print(f"EER: {eer:.6f}")
    print(f"mAP: {mAP:.6f}")
    print(f"Recall@{K}: {recall_k:.6f}")


if __name__ == "__main__":
    args = parse_args()

    print("Loading model...")
    model = StampNet(embed_dim=256)
    model.load_state_dict(torch.load(args.path, map_location=device))
    model = model.to(device)
    print("Loading testing dataset...")
    test_dataset = TestingTriplets(args.directory, transform=transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    print(f"Total Test Triplets: {len(test_dataset)}")
    scores, labels, pos_scores, neg_scores = evaluate_model(
        model,
        test_loader,
        device
    )
    evaluate_metrics(scores, labels, pos_scores, neg_scores)