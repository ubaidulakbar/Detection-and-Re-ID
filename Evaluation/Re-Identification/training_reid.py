import argparse
from collections import defaultdict
import math
import numpy as np
import os
from PIL import Image
import random
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, average_precision_score, roc_curve
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as transforms
from torchvision import models
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
scaler = torch.amp.GradScaler('cuda')

DROPOUT = 0.15
MARGIN = 0.3
PHASE1EPOCHS = 10
PHASE2EPOCHS = 50
EMBED_DIM = 256


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train Object Re-Identification Model.'
    )

    parser.add_argument(
        '--triplets',
        type=int,
        required=True,
        help='Number of Triplets to be generated',
    )
    parser.add_argument(
        '--algo',
        type=str,
        default='r',
        help='Algorithm selection (Random or Hierarchical)',
    )
    parser.add_argument(
        '--pre_trained',
        type=bool,
        default=True,
        help='Pretrained Model (default True)',
    )
    parser.add_argument(
        '--batch',
        type=int,
        default=64,
        help='BATCHSIZE',
    )

    return parser.parse_args()


transform = transforms.Compose([
    transforms.Resize((192, 192), antialias=True),
    transforms.ToImage(),
    transforms.ToDtype(torch.float32, scale=True),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

class TripletDataset(Dataset):
    def __init__(
        self,
        main_dir: str,
        num_triplets: int,
        transform=transform,
        cross_triplets_per_pair: int | None = None,
    ):
        self.main_dir = main_dir
        self.num_triplets = num_triplets
        self.transform = transform

        # -----------------------------
        # 1. Scan directory structure
        # -----------------------------
        entries = [
            d for d in os.listdir(main_dir)
            if os.path.isdir(os.path.join(main_dir, d))
        ]

        # Separate base and non-base groups
        self.base_name = 'base' if 'base' in entries else None
        self.group_names = [d for d in entries if d != 'base']
        self.group_names.sort()

        if len(self.group_names) == 0:
            raise ValueError('No group folders found in the dataset directory.')

        # For each group, gather class folders and images
        # group_classes[group_name] = [class_dir_path1, class_dir_path2, ...]
        self.group_classes: Dict[str, List[str]] = {}
        # class_images[class_dir_path] = [img_path1, img_path2, ...]
        self.class_images: Dict[str, List[str]] = {}

        # Helper to load classes and images
        def load_group_classes(group_name: str) -> List[str]:
            group_path = os.path.join(main_dir, group_name)
            class_dirs = [
                os.path.join(group_path, c)
                for c in os.listdir(group_path)
                if os.path.isdir(os.path.join(group_path, c))
            ]
            class_dirs.sort()

            # For each class dir, collect image paths
            good_class_dirs = []
            for cdir in class_dirs:
                imgs = sorted([
                    os.path.join(cdir, f)
                    for f in os.listdir(cdir)
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))
                ])
                if imgs:
                    self.class_images[cdir] = imgs
                    good_class_dirs.append(cdir)

            # Must have at least 2 non-empty classes
            if len(good_class_dirs) < 2:
                raise ValueError(f'Group "{group_name}" has less than 2 classes')

            return good_class_dirs

        # Load non-base groups
        for g in self.group_names:
            self.group_classes[g] = load_group_classes(g)

        # Load base (if present)
        self.base_classes: Optional[List[str]] = None
        if self.base_name is not None:
            self.base_classes = load_group_classes(self.base_name)

        # -----------------------------
        # 2. Plan triplet allocation
        # -----------------------------
        self.triplet_specs = []  # list of dicts describing each triplet

        # Counters for nice printing
        cross_group_counts = defaultdict(int)   # per anchor group/base
        same_group_counts = defaultdict(int)    # per non-base group
        base_group_count = 0                    # base same-group triplets

        # ---- 2.a Cross-Group Triplets ----
        # Build ordered pairs (anchor_group, negative_group)

        pairs: List[Tuple[str, str]] = []

        if self.base_classes is None:
            # No base: all ordered pairs among non-base groups
            if len(self.group_names) < 2:
                raise ValueError('Need at least 2 groups for cross-group triplets (no base present).')

            for g1 in self.group_names:
                for g2 in self.group_names:
                    if g1 != g2:
                        pairs.append((g1, g2))
        else:
            # With base: base->group + all ordered pairs among non-base groups
            if len(self.group_names) < 1:
                raise ValueError('At least 1 non-base group needed when base is present.')

            # base -> each group
            for g in self.group_names:
                pairs.append(('base', g))

            # non-base group -> other non-base groups
            for g1 in self.group_names:
                for g2 in self.group_names:
                    if g1 != g2:
                        pairs.append((g1, g2))
    
        num_pairs = len(pairs)
        if cross_triplets_per_pair is None:
            cross_triplets_per_pair = int((num_triplets * 0.20) // num_pairs)
        self.cross_triplets_per_pair = cross_triplets_per_pair
        cross_capacity = num_pairs * self.cross_triplets_per_pair
        target_cross = min(self.num_triplets, cross_capacity)

        pair_counts = {pair: 0 for pair in pairs}

        # Fill cross-group specs
        while len(self.triplet_specs) < target_cross:
            for anchor_group, neg_group in pairs:
                if len(self.triplet_specs) >= target_cross:
                    break
                if pair_counts[(anchor_group, neg_group)] < self.cross_triplets_per_pair:
                    self.triplet_specs.append({
                        'type': 'cross',
                        'anchor_group': anchor_group,
                        'neg_group': neg_group,
                    })
                    pair_counts[(anchor_group, neg_group)] += 1
                    cross_group_counts[anchor_group] += 1

        # ---- 2.b Same-Group / Base-Group Triplets ----
        leftover = self.num_triplets - len(self.triplet_specs)

        if leftover > 0:
            if self.base_classes is None:
                # No base: leftover is distributed across non-base groups
                G = len(self.group_names)
                base_per_group = leftover // G
                remainder = leftover % G

                for idx, g in enumerate(self.group_names):
                    # Last group gets the remainder
                    quota = base_per_group + (remainder if idx == G - 1 else 0)
                    for _ in range(quota):
                        self.triplet_specs.append({
                            'type': 'same',
                            'group': g,
                        })
                        same_group_counts[g] += 1
            else:
                # With base: use (num_groups + 2) rule described
                G = len(self.group_names)
                base_per_group = leftover // (G + 1)

                # Same-group for each non-base group
                assigned_to_groups = 0
                for g in self.group_names:
                    quota = base_per_group
                    assigned_to_groups += quota
                    for _ in range(quota):
                        self.triplet_specs.append({
                            'type': 'same',
                            'group': g,
                        })
                        same_group_counts[g] += 1

                # Remaining go to base
                base_quota = leftover - assigned_to_groups
                for _ in range(base_quota):
                    self.triplet_specs.append({
                        'type': 'base_same',
                    })
                    base_group_count += 1

        # Safety: may be less than requested if capacity is limited
        if len(self.triplet_specs) < self.num_triplets:
            print(
                f'Warning: Only {len(self.triplet_specs)} triplets could be planned '
                f'(requested {self.num_triplets}).'
            )

        # -----------------------------
        # 3. Summary printing
        # -----------------------------
        # Cross-Group summary
        for g in sorted(cross_group_counts.keys()):
            label = 'Base' if g == 'base' else f'Group "{g}"'
            print(f'{cross_group_counts[g]} Cross-Group triplets generated for {label}')

        # Same-Group summary (non-base)
        for g in sorted(same_group_counts.keys()):
            print(f'{same_group_counts[g]} Same-Group triplets generated for Group "{g}"')

        # Base-Group same summary
        if base_group_count > 0:
            print(f'{base_group_count} Base-Group triplets generated for Base')

    # -----------------------------
    # Dataset protocol
    # -----------------------------
    def __len__(self):
        return len(self.triplet_specs)

    def _sample_anchor_positive_from_class(self, class_dir: str) -> Tuple[str, str]:
        imgs = self.class_images[class_dir]
        anchor_path = random.choice(imgs)
        if len(imgs) > 1:
            positive_path = random.choice([p for p in imgs if p != anchor_path])
        else:
            positive_path = anchor_path
        return anchor_path, positive_path

    def _sample_class(self, group_name: str) -> str:
        if group_name == 'base':
            assert self.base_classes is not None
            return random.choice(self.base_classes)
        return random.choice(self.group_classes[group_name])

    def __getitem__(self, idx):
        spec = self.triplet_specs[idx]
        ttype = spec['type']

        if ttype == 'cross':
            anchor_group = spec['anchor_group']
            neg_group = spec['neg_group']

            # Anchor & positive from one class in anchor_group
            anchor_class = self._sample_class(anchor_group)
            anchor_path, positive_path = self._sample_anchor_positive_from_class(anchor_class)

            # Negative from a class in neg_group
            negative_class = self._sample_class(neg_group)
            negative_imgs = self.class_images[negative_class]
            negative_path = random.choice(negative_imgs)

        elif ttype == 'same':
            group = spec['group']
            class_list = self.group_classes[group]
            # Two different classes within the same group
            class_anchor, class_negative = random.sample(class_list, 2)

            anchor_path, positive_path = self._sample_anchor_positive_from_class(class_anchor)
            negative_imgs = self.class_images[class_negative]
            negative_path = random.choice(negative_imgs)

        elif ttype == 'base_same':
            # Two classes within base
            assert self.base_classes is not None
            class_anchor, class_negative = random.sample(self.base_classes, 2)

            anchor_path, positive_path = self._sample_anchor_positive_from_class(class_anchor)
            negative_imgs = self.class_images[class_negative]
            negative_path = random.choice(negative_imgs)

        else:
            raise ValueError(f'Unknown triplet type: {ttype}')

        # Load and transform images
        anchor_img = Image.open(anchor_path).convert('RGB')
        positive_img = Image.open(positive_path).convert('RGB')
        negative_img = Image.open(negative_path).convert('RGB')

        if self.transform:
            anchor_img = self.transform(anchor_img)
            positive_img = self.transform(positive_img)
            negative_img = self.transform(negative_img)

        return anchor_img, positive_img, negative_img  


class RandomTripletDataset(Dataset):
    def __init__(self, main_dir, num_triplets, transform=transform):
        self.main_dir = main_dir
        self.num_triplets = num_triplets
        self.transform = transform

        # Load classes and images
        self.classes = []
        self.class_images = {}

        for cls in sorted(os.listdir(main_dir)):
            cls_path = os.path.join(main_dir, cls)
            if not os.path.isdir(cls_path):
                continue

            images = [
                os.path.join(cls_path, f)
                for f in os.listdir(cls_path)
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ]

            if len(images) >= 2:
                self.classes.append(cls)
                self.class_images[cls] = images

        if len(self.classes) < 2:
            raise ValueError(
                "Need at least 2 classes with >=2 images each to form triplets."
            )

    def __len__(self):
        return self.num_triplets

    def __getitem__(self, idx):
        # Pick anchor/positive class
        cls_anchor = random.choice(self.classes)
        images_anchor = self.class_images[cls_anchor]
        anchor_path, positive_path = random.sample(images_anchor, 2)
        # Pick negative class (different)
        cls_negative = random.choice([c for c in self.classes if c != cls_anchor])
        negative_path = random.choice(self.class_images[cls_negative])

        # Load images
        anchor = Image.open(anchor_path).convert("RGB")
        positive = Image.open(positive_path).convert("RGB")
        negative = Image.open(negative_path).convert("RGB")

        if self.transform:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)
        return anchor, positive, negative


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
    

class StampNet(nn.Module):
    def __init__(self, embed_dim: int = EMBED_DIM, pretrained: bool = True, dropout: float = DROPOUT):
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


def triplet_loss(anchor, positive, negative, margin: float = MARGIN):
    d_ap = 1.0 - F.cosine_similarity(anchor, positive)
    d_an = 1.0 - F.cosine_similarity(anchor, negative)
    loss = F.relu(d_ap - d_an + margin).mean()
    return loss


def run_epoch(model, dataloader, optimizer=None, scaler=None, train=True):
    if train:
        model.train()
    else:
        model.eval()
    total_loss = 0.0
    num_samples = 0
    for batch in tqdm(dataloader, desc='Train' if train else 'Val', leave=False):
        anchor, positive, negative = batch
        anchor = anchor.to(device, non_blocking=True)
        positive = positive.to(device, non_blocking=True)
        negative = negative.to(device, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            with torch.autocast(device_type="cuda", enabled=train):

                # batch = torch.stack((anchor, positive, negative), dim=1)
                # batch = batch.view(-1, *anchor.shape[1:])
                batch = torch.cat([anchor, positive, negative], dim=0)

                embeddings = model(batch)

                bs = anchor.size(0)

                a_emb = embeddings[:bs]
                p_emb = embeddings[bs:2*bs]
                n_emb = embeddings[2*bs:]

                loss = triplet_loss(a_emb, p_emb, n_emb, margin=MARGIN)

            if train and scaler is not None:
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
        bs = anchor.size(0)
        total_loss += loss.item() * bs
        num_samples += bs
    avg_loss = total_loss / max(1, num_samples)
    return avg_loss

def evaluate_model(model, dataloader, device, margin=MARGIN):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    all_pos_scores = []
    all_neg_scores = []

    with torch.no_grad():
        for anchor, positive, negative in tqdm(dataloader, desc="Evaluating", leave=False):
            anchor = anchor.to(device, non_blocking=True)
            positive = positive.to(device, non_blocking=True)
            negative = negative.to(device, non_blocking=True)

            # Forward pass
            a_emb = model(anchor)
            p_emb = model(positive)
            n_emb = model(negative)

            # Triplet loss
            loss = triplet_loss(a_emb, p_emb, n_emb, margin=margin)

            bs = anchor.size(0)
            total += bs
            total_loss += loss.item() * bs

            # Triplet accuracy
            dist_pos = torch.norm(a_emb - p_emb, p=2, dim=1)
            dist_neg = torch.norm(a_emb - n_emb, p=2, dim=1)
            correct += torch.sum(dist_pos < dist_neg).item()

            # Cosine similarity scores
            pos_scores = F.cosine_similarity(a_emb, p_emb)
            neg_scores = F.cosine_similarity(a_emb, n_emb)

            all_pos_scores.append(pos_scores.cpu())
            all_neg_scores.append(neg_scores.cpu())

    # Aggregate metrics
    avg_loss = total_loss / max(1, total)
    triplet_acc = correct / max(1, total)

    print(f"Validation Loss: {avg_loss:.6f}")
    print(f"Triplet Accuracy: {triplet_acc:.6f}")

    # Convert scores to numpy arrays for further evaluation
    pos_scores = torch.cat(all_pos_scores).numpy()
    neg_scores = torch.cat(all_neg_scores).numpy()

    scores = np.concatenate([pos_scores, neg_scores])
    labels = np.concatenate([
        np.ones_like(pos_scores),
        np.zeros_like(neg_scores)
    ])

    return scores, labels



def evaluate_metrics(scores, labels, pos_scores, neg_scores):
    roc_auc = roc_auc_score(labels, scores)
    precision, recall, _ = precision_recall_curve(labels, scores)
    pr_auc = auc(recall, precision)
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr
    eer_index = np.nanargmin(np.absolute(fnr - fpr))
    eer = fpr[eer_index]
    mAP = average_precision_score(labels, scores)
    K = 1
    correct = 0
    total = len(pos_scores)

    for p, n in zip(pos_scores, neg_scores):
        sims = np.array([p, n])
        labels_local = np.array([1, 0])
        topk = sims.argsort()[::-1][:K]
        if 0 in topk:  # positive is index 0
            correct += 1
    recall_k = correct / total

    print("\n----- Evaluation Metrics -----")
    print(f"ROC-AUC: {roc_auc:.6f}")
    print(f"PR-AUC: {pr_auc:.6f}")
    print(f"EER: {eer:.6f}")
    print(f"mAP: {mAP:.6f}")
    print(f"Recall@{K}: {recall_k:.6f}")
    # return roc_auc, pr_auc, eer, mAP, recall_k


if __name__ == "__main__":
    args = parse_args()
    phase1_best_model = f'Phase 1/{args.triplets}.pth'
    phase2_best_model = f'Phase 2/{args.triplets}.pth'

    if args.algo == 'r':
        training_dataset = RandomTripletDataset(main_dir='Training', num_triplets=args.triplets, transform=transform)
        train_loader = DataLoader(
            training_dataset,
            batch_size=args.batch,
            shuffle=True,
            pin_memory=True,
            num_workers=4,
            persistent_workers=True,
            prefetch_factor=4
        )
        print(f'Total Training Size: Triplets={len(training_dataset)}, Batches={len(train_loader)}.')
    else:
        training_dataset = TripletDataset(main_dir='Training', num_triplets=args.triplets, transform=transform)
        train_loader = DataLoader(
            training_dataset,
            batch_size=args.batch,
            shuffle=True,
            pin_memory=True,
            num_workers=4,
            persistent_workers=True,
            prefetch_factor=4
        )
        print(f'Total Training Size: Triplets={len(training_dataset)}, Batches={len(train_loader)}.')

    testing_dataset = TestingTriplets(root_dir='Testing', transform=transform)
    test_loader = DataLoader(
        testing_dataset,
        batch_size=args.batch,
        shuffle=False,
        pin_memory=True,
        num_workers=4,
        persistent_workers=True,
        prefetch_factor=4
    )
    print(f'Total Testing Size: Triplets={len(testing_dataset)}, Batches={len(test_loader)}.')


    model = StampNet(embed_dim=EMBED_DIM, pretrained=args.pre_trained)
    model = model.to(device)
    for p in model.backbone.parameters():
        p.requires_grad = False
    for m in model.backbone.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
    for p in model.backbone.features[-1].parameters():
        p.requires_grad = True
    for p in model.head.parameters():
        p.requires_grad = True
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=3e-4, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=PHASE1EPOCHS, eta_min=1e-6)
    print('Trainable parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))
    print('Total parameters:', sum(p.numel() for p in model.parameters()))

    best_val_loss = math.inf
    best_state_dict = None
    for epoch in range(1, PHASE1EPOCHS + 1):
        print(f'\nEpoch {epoch}/{PHASE1EPOCHS}')
        train_loss = run_epoch(model, train_loader, optimizer, scaler, train=True)
        val_loss = run_epoch(model, test_loader, optimizer=None, scaler=None, train=False)
        scheduler.step()
        print(f'  Train Loss: {train_loss:.8f} | Val Loss: {val_loss:.8f}')
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state_dict = {
                k: v.cpu().clone()
                for k, v in model.state_dict().items()
            }
            torch.save(best_state_dict, phase1_best_model)
            print(f'  → Best model saved to {phase1_best_model}')
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        model.to(device)
        print(f'\nLoaded best model with val loss = {best_val_loss:.8f}')
    else:
        print('\nWarning: no best_state_dict saved (something went wrong).')

    try:
        model.load_state_dict(best_state_dict)
        print('Model taken from memory')
    except:
        state_dict = torch.load(phase1_best_model, map_location=device)
        model.load_state_dict(state_dict)
        print('Model loaded from directory')
    model.to(device)
    for p in model.backbone.parameters():
        p.requires_grad = True
    for p in model.head.parameters():
        p.requires_grad = True
    print('Trainable parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))
    print('Total parameters:', sum(p.numel() for p in model.parameters()))

        
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=PHASE2EPOCHS, eta_min=1e-6)
    best_val_loss = math.inf
    best_state_dict = None
    for epoch in range(1, PHASE2EPOCHS + 1):
        print(f'\nEpoch {epoch}/{PHASE2EPOCHS}')
        train_loss = run_epoch(model, train_loader, optimizer, scaler, train=True)
        val_loss = run_epoch(model, test_loader, optimizer=None, scaler=None, train=False)
        scheduler.step()
        print(f'  Train Loss: {train_loss:.8f} | Val Loss: {val_loss:.8f}')
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state_dict = {
                k: v.cpu().clone()
                for k, v in model.state_dict().items()
            }
            torch.save(best_state_dict, phase2_best_model)
            print(f'  → Best model saved to {phase2_best_model}')
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        model.to(device)
        print(f'\nLoaded best model with val loss = {best_val_loss:.8f}')
    else:
        print('\nWarning: no best_state_dict saved (something went wrong).')

    scores, labels = evaluate_model(model, test_loader, device)
    half = len(scores) // 2
    pos_scores = scores[:half]
    neg_scores = scores[half:]
    evaluate_metrics(scores, labels, pos_scores, neg_scores)