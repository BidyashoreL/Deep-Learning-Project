"""
RF-AttentionUNet: A Novel Hybrid Architecture
==============================================
Combines:
  1. Random Forest feature importance maps -> spatial gate vector G
  2. U-Net encoder-decoder with skip connections
  3. Attention gate modulated by G at each decoder stage

Key innovation: The RF gate G tells the attention mechanism
*which input features* matter most per-pixel, grounding the
deep network's attention in interpretable, tree-based priors.

Loss = CrossEntropy + (1 - Dice) + lambda * RF_consistency_loss
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import rasterio
import os


# ============================================================
# 1. BUILDING BLOCKS
# ============================================================

class ConvBlock(nn.Module):
    """Double Conv -> BN -> ReLU block (standard U-Net unit)."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class RFAttentionGate(nn.Module):
    """
    RF-modulated attention gate (the core novel contribution).

    Standard additive attention gate (Oktay et al., 2018):
        att = sigmoid(W_psi(ReLU(W_g(g) + W_x(x))))

    Our extension -- multiply by RF gate G:
        att_rf = att * G

    where G is a learned transformation of the per-pixel
    Random Forest feature importance vector, broadcast to
    the spatial resolution of the current decoder stage.

    Args:
        F_g   : channels from gating signal (decoder)
        F_l   : channels from skip connection (encoder)
        F_int : intermediate channels
        rf_dim: dimension of RF importance vector (= n_features)
    """
    def __init__(self, F_g, F_l, F_int, rf_dim=5):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, 1, bias=False),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, 1, bias=False),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        # RF gate: maps importance vector -> spatial gate [0,1]
        self.rf_gate = nn.Sequential(
            nn.Linear(rf_dim, F_int),
            nn.ReLU(inplace=True),
            nn.Linear(F_int, 1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x, rf_importance):
        """
        g             : (B, F_g, H, W)  -- gating signal from decoder
        x             : (B, F_l, H, W)  -- skip from encoder
        rf_importance : (B, rf_dim)     -- per-image RF importance vector
        Returns:
            x_att     : (B, F_l, H, W)  -- attention-gated features
            att_map   : (B, 1,  H, W)   -- attention map (for visualization)
        """
        # Standard additive attention
        g1 = self.W_g(g)
        x1 = self.W_x(x)

        # Upsample g1 if spatial dims differ (happens at deep stages)
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode='bilinear', align_corners=False)

        psi = self.psi(self.relu(g1 + x1))          # (B, 1, H, W)

        # RF gate: scalar per image, broadcast to spatial
        G = self.rf_gate(rf_importance)              # (B, 1)
        G = G.unsqueeze(-1).unsqueeze(-1)            # (B, 1, 1, 1)

        # Combined attention: A = psi * G
        att = psi * G                                # (B, 1, H, W)

        x_att = x * att
        return x_att, att


class DecoderBlock(nn.Module):
    """UpConv -> concat skip -> ConvBlock, with RF-attention gate."""
    def __init__(self, in_ch, skip_ch, out_ch, rf_dim=5):
        super().__init__()
        self.up    = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.gate  = RFAttentionGate(F_g=out_ch, F_l=skip_ch, F_int=out_ch // 2, rf_dim=rf_dim)
        self.conv  = ConvBlock(out_ch + skip_ch, out_ch)

    def forward(self, x, skip, rf_importance):
        x         = self.up(x)
        skip_att, att_map = self.gate(x, skip, rf_importance)
        x         = torch.cat([x, skip_att], dim=1)
        x         = self.conv(x)
        return x, att_map


# ============================================================
# 2. RF BRANCH -- feature importance extractor
# ============================================================

class RFBranch:
    """
    Trains a Random Forest on the flat per-pixel feature vectors
    and extracts two things:
      (a) global feature_importances_ (n_features,)
      (b) per-pixel class probability maps for soft gating
    """
    def __init__(self, n_estimators=200, n_jobs=-1, random_state=42):
        self.rf  = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=12,
            min_samples_leaf=4,
            n_jobs=n_jobs,
            random_state=random_state,
            class_weight='balanced'
        )
        self.scaler = StandardScaler()
        self.fitted = False

    def fit(self, X_flat, y_flat):
        """
        X_flat : (N, n_features)  -- flattened pixels
        y_flat : (N,)             -- class labels {0,1,2}
        """
        X_scaled = self.scaler.fit_transform(X_flat)
        self.rf.fit(X_scaled, y_flat)
        self.fitted = True
        imp = self.rf.feature_importances_
        print(f"RF fitted. Feature importances: {np.round(imp, 4)}")
        return imp

    def get_importance_tensor(self, batch_size):
        """Returns (B, n_features) importance vector, repeated for the batch."""
        assert self.fitted, "RF not fitted yet -- call fit() first"
        imp = torch.tensor(self.rf.feature_importances_, dtype=torch.float32)
        return imp.unsqueeze(0).repeat(batch_size, 1)   # (B, n_features)

    def predict_proba_map(self, X_flat, H, W):
        """
        Returns (H, W, 3) soft probability map from RF.
        Used as an auxiliary supervision signal.
        """
        assert self.fitted
        X_scaled = self.scaler.transform(X_flat)
        proba    = self.rf.predict_proba(X_scaled)       # (N, 3)
        return proba.reshape(H, W, 3)


# ============================================================
# 3. MAIN MODEL: RF-AttentionUNet
# ============================================================

class RFAttentionUNet(nn.Module):
    """
    RF-AttentionUNet

    Architecture:
      Encoder: 4 ConvBlocks with MaxPool downsampling
      Bottleneck: ConvBlock at 512ch
      Decoder: 4 DecoderBlocks with RF-attention gating
      Head: 1x1 Conv -> 3-class output

    The RF branch is trained separately (offline) and its
    feature importance vector is passed at each decoder stage
    to modulate the attention gate.

    Parameters:
      in_channels  : number of input feature channels (5)
      num_classes  : output classes (3)
      base_ch      : base channel count (default 64)
      rf_dim       : length of RF importance vector (= in_channels)
    """
    def __init__(self, in_channels=5, num_classes=3, base_ch=64, rf_dim=5):
        super().__init__()
        b = base_ch

        # Encoder
        self.enc1 = ConvBlock(in_channels, b)       # -> (B, 64,  H,   W)
        self.enc2 = ConvBlock(b,     b*2)            # -> (B, 128, H/2, W/2)
        self.enc3 = ConvBlock(b*2,   b*4)            # -> (B, 256, H/4, W/4)
        self.enc4 = ConvBlock(b*4,   b*8)            # -> (B, 512, H/8, W/8)
        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = ConvBlock(b*8, b*16)       # -> (B, 1024, H/16, W/16)

        # Decoder (with RF-attention gates)
        self.dec4 = DecoderBlock(b*16, b*8,  b*8,  rf_dim)
        self.dec3 = DecoderBlock(b*8,  b*4,  b*4,  rf_dim)
        self.dec2 = DecoderBlock(b*4,  b*2,  b*2,  rf_dim)
        self.dec1 = DecoderBlock(b*2,  b,    b,    rf_dim)

        # Output head
        self.head = nn.Conv2d(b, num_classes, kernel_size=1)

    def forward(self, x, rf_importance):
        """
        x             : (B, in_channels, H, W)
        rf_importance : (B, rf_dim)           -- from RFBranch.get_importance_tensor()

        Returns:
            logits    : (B, num_classes, H, W)
            att_maps  : dict of attention maps per decoder stage (for visualization)
        """
        # Encode
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck
        b  = self.bottleneck(self.pool(e4))

        # Decode (each stage returns attended features + attention map)
        d4, att4 = self.dec4(b,  e4, rf_importance)
        d3, att3 = self.dec3(d4, e3, rf_importance)
        d2, att2 = self.dec2(d3, e2, rf_importance)
        d1, att1 = self.dec1(d2, e1, rf_importance)

        logits = self.head(d1)

        att_maps = {'stage1': att1, 'stage2': att2,
                    'stage3': att3, 'stage4': att4}
        return logits, att_maps


# ============================================================
# 4. LOSS FUNCTION
# ============================================================

class RFAttentionLoss(nn.Module):
    """
    Combined loss:
      L = L_CE + alpha * (1 - L_Dice) + beta * L_RF_consistency

    L_RF_consistency:
      Encourages the model's output to be consistent with the
      RF soft predictions in regions where RF is confident
      (i.e., max probability > threshold). This regularizes
      the deep branch using the tree-based prior.

    Equations:
      L_Dice  = (2 * |Y n P| + eps) / (|Y| + |P| + eps)
      L_RF    = KL(p_rf || p_unet) where p_rf > conf_threshold
    """
    def __init__(self, alpha=1.0, beta=0.3, conf_threshold=0.7, num_classes=3):
        super().__init__()
        self.alpha     = alpha
        self.beta      = beta
        self.threshold = conf_threshold
        self.nc        = num_classes
        self.ce        = nn.CrossEntropyLoss()

    def dice_loss(self, pred, target):
        pred   = F.softmax(pred, dim=1)
        target_oh = F.one_hot(target, self.nc).permute(0,3,1,2).float()
        inter  = (pred * target_oh).sum(dim=(2,3))
        union  = pred.sum(dim=(2,3)) + target_oh.sum(dim=(2,3))
        dice   = (2 * inter + 1e-5) / (union + 1e-5)
        return 1 - dice.mean()

    def rf_consistency_loss(self, pred_logits, rf_proba_map):
        """
        pred_logits  : (B, C, H, W)
        rf_proba_map : (B, C, H, W) -- RF soft predictions
        Applies KL only where RF is confident.
        """
        rf_max_prob = rf_proba_map.max(dim=1, keepdim=True).values
        mask        = (rf_max_prob > self.threshold).float()

        if mask.sum() == 0:
            return torch.tensor(0.0, device=pred_logits.device)

        p_unet = F.log_softmax(pred_logits, dim=1)
        p_rf   = rf_proba_map.clamp(1e-6, 1.0)

        kl = F.kl_div(p_unet, p_rf, reduction='none').sum(dim=1, keepdim=True)
        return (kl * mask).sum() / (mask.sum() + 1e-5)

    def forward(self, pred_logits, target, rf_proba_map=None):
        l_ce   = self.ce(pred_logits, target)
        l_dice = self.dice_loss(pred_logits, target)
        loss   = l_ce + self.alpha * l_dice

        if rf_proba_map is not None and self.beta > 0:
            l_rf = self.rf_consistency_loss(pred_logits, rf_proba_map)
            loss = loss + self.beta * l_rf
        else:
            l_rf = torch.tensor(0.0)

        return loss, {'ce': l_ce.item(), 'dice': l_dice.item(), 'rf': l_rf.item()}


# ============================================================
# 5. TRAINING PIPELINE
# ============================================================

class RFAttentionUNetTrainer:
    """
    Full training pipeline:
      Phase 1 -- Fit RF on pixel features (fast, CPU)
      Phase 2 -- Train RF-AttentionUNet (GPU/MPS)
    """
    def __init__(self, config):
        self.cfg    = config
        self.device = torch.device(
            'mps'  if torch.backends.mps.is_available() else
            'cuda' if torch.cuda.is_available()         else 'cpu'
        )
        print(f"Device: {self.device}")

        self.rf_branch = RFBranch(
            n_estimators=config.get('rf_trees', 200)
        )
        self.model = RFAttentionUNet(
            in_channels=config.get('in_channels', 5),
            num_classes=config.get('num_classes', 3),
            base_ch=config.get('base_ch', 32),       # 32 for memory efficiency on CPU/MPS
            rf_dim=config.get('in_channels', 5)
        ).to(self.device)

        self.criterion = RFAttentionLoss(
            alpha=config.get('alpha', 1.0),
            beta=config.get('beta', 0.3),
            num_classes=config.get('num_classes', 3)
        )
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.get('lr', 1e-3),
            weight_decay=1e-5
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=5, factor=0.5
        )

        self.best_loss = float('inf')
        os.makedirs(config.get('save_dir', 'outputs'), exist_ok=True)

    def phase1_fit_rf(self, image_tensor, label_tensor):
        """
        image_tensor : (C, H, W) numpy array
        label_tensor : (H, W)    numpy array with values {0,1,2}
        """
        print("\n=== Phase 1: Fitting Random Forest ===")
        C, H, W = image_tensor.shape
        X = image_tensor.reshape(C, -1).T    # (H*W, C)
        y = label_tensor.flatten()           # (H*W,)

        # Keep only pixels where: image has no NaN AND label is a valid class (0,1,2)
        valid_img   = ~np.isnan(X).any(axis=1)
        valid_label = np.isin(y, [0, 1, 2])
        valid       = valid_img & valid_label

        imp = self.rf_branch.fit(X[valid], y[valid])
        print(f"Phase 1 complete. Importances: {imp}")
        return imp

    def _get_rf_proba_map_tensor(self, image_tensor):
        """Convert RF prediction to (1, 3, H, W) tensor for loss."""
        C, H, W = image_tensor.shape
        X = image_tensor.reshape(C, -1).T
        X = np.nan_to_num(X)
        proba_map = self.rf_branch.predict_proba_map(X, H, W)   # (H, W, 3)
        t = torch.tensor(proba_map, dtype=torch.float32)
        return t.permute(2, 0, 1).unsqueeze(0).to(self.device)  # (1, 3, H, W)

    def train_epoch(self, patches, labels_patches, rf_importance_tensor, image_np):
        """
        patches         : list of (C, H, W) tensors
        labels_patches  : list of (H, W) tensors
        """
        self.model.train()
        total_loss = 0.0
        rf_proba   = self._get_rf_proba_map_tensor(image_np)

        for patch, label in zip(patches, labels_patches):
            patch  = patch.unsqueeze(0).to(self.device)    # (1, C, H, W)
            label  = label.unsqueeze(0).long().to(self.device)  # (1, H, W)

            # Crop RF proba map to patch if needed (simplified: use full map resized)
            _, _, ph, pw = patch.shape
            rf_p = F.interpolate(rf_proba, size=(ph, pw), mode='bilinear', align_corners=False)

            self.optimizer.zero_grad()
            logits, att_maps = self.model(patch, rf_importance_tensor)
            loss, components = self.criterion(logits, label, rf_p)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / max(len(patches), 1)

    def train(self, image_np, label_np, patches, label_patches):
        """
        Full training run.
        image_np      : (C, H, W) numpy  -- full raster
        label_np      : (H, W)    numpy
        patches       : list of torch tensors (C, ph, pw)
        label_patches : list of torch tensors (ph, pw)
        """
        # Phase 1
        self.phase1_fit_rf(image_np, label_np)

        # RF importance as fixed tensor for the batch
        rf_imp = self.rf_branch.get_importance_tensor(batch_size=1).to(self.device)

        # Phase 2
        print("\n=== Phase 2: Training RF-AttentionUNet ===")
        epochs     = self.cfg.get('epochs', 50)
        save_path  = os.path.join(self.cfg.get('save_dir', 'outputs'),
                                  'rf_attention_unet_best.pth')

        for epoch in range(1, epochs + 1):
            avg_loss = self.train_epoch(patches, label_patches, rf_imp, image_np)
            self.scheduler.step(avg_loss)

            star = ""
            if avg_loss < self.best_loss:
                self.best_loss = avg_loss
                torch.save({
                    'epoch': epoch,
                    'model_state': self.model.state_dict(),
                    'optimizer_state': self.optimizer.state_dict(),
                    'loss': self.best_loss,
                    'rf_importances': self.rf_branch.rf.feature_importances_
                }, save_path)
                star = "  *** best saved"

            print(f"Epoch {epoch:3d}/{epochs} | Loss: {avg_loss:.4f}{star}")

        print(f"\nTraining complete. Best loss: {self.best_loss:.4f}")
        return self.best_loss


# ============================================================
# 6. INFERENCE
# ============================================================

def predict(model, rf_branch, image_np, device, patch_size=128):
    """
    Full-image inference using sliding window.
    Returns:
        pred_map  : (H, W)    -- class labels
        prob_map  : (H, W, 3) -- class probabilities
    """
    model.eval()
    C, H, W = image_np.shape

    rf_imp   = rf_branch.get_importance_tensor(batch_size=1).to(device)
    pred_acc = np.zeros((3, H, W), dtype=np.float32)
    count    = np.zeros((H, W),    dtype=np.float32)

    stride   = patch_size // 2    # 50% overlap for smooth stitching

    with torch.no_grad():
        for y in range(0, H - patch_size + 1, stride):
            for x in range(0, W - patch_size + 1, stride):
                patch = image_np[:, y:y+patch_size, x:x+patch_size]
                patch = np.nan_to_num(patch)
                t = torch.tensor(patch, dtype=torch.float32).unsqueeze(0).to(device)
                logits, _ = model(t, rf_imp)
                probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
                pred_acc[:, y:y+patch_size, x:x+patch_size] += probs
                count[y:y+patch_size, x:x+patch_size]        += 1

    count = np.maximum(count, 1)
    pred_acc /= count
    pred_map  = pred_acc.argmax(axis=0).astype(np.uint8)
    prob_map  = pred_acc.transpose(1, 2, 0)
    return pred_map, prob_map


# ============================================================
# 7. ENTRY POINT
# ============================================================

if __name__ == "__main__":
    import sys
    sys.path.insert(0, "src")

    CONFIG = {
        'in_channels'  : 5,
        'num_classes'  : 3,
        'base_ch'      : 32,       # increase to 64 if you have >8GB RAM/VRAM
        'rf_trees'     : 200,
        'epochs'       : 50,
        'lr'           : 1e-3,
        'alpha'        : 1.0,      # Dice weight
        'beta'         : 0.3,      # RF consistency weight
        'patch_size'   : 128,
        'save_dir'     : 'outputs',
    }

    # Load full raster image
    FLOOD_STACK_PATH = "outputs/input_stack.tif"
    LABEL_PATH       = "data/label.tif"

    print("Loading data...")

    with rasterio.open(FLOOD_STACK_PATH) as src:
        image_np = src.read().astype(np.float32)    # (5, H, W)
        profile  = src.profile

    with rasterio.open(LABEL_PATH) as src:
        label_np = src.read(1).astype(np.int64)     # (H, W)

    # Normalize each channel independently
    for c in range(image_np.shape[0]):
        ch = image_np[c]
        mn, mx = np.nanmin(ch), np.nanmax(ch)
        if mx > mn:
            image_np[c] = (ch - mn) / (mx - mn)

    # Build patch dataset
    PATCH  = CONFIG['patch_size']
    STRIDE = PATCH // 2

    patches = []
    label_patches = []

    H, W = label_np.shape

    for y in range(0, H - PATCH + 1, STRIDE):
        for x in range(0, W - PATCH + 1, STRIDE):
            img_patch = image_np[:, y:y+PATCH, x:x+PATCH]
            lbl_patch = label_np[y:y+PATCH, x:x+PATCH]

            patches.append(torch.tensor(img_patch, dtype=torch.float32))
            label_patches.append(torch.tensor(lbl_patch, dtype=torch.long))

    print(f"Dataset: {len(patches)} patches")

    # Train
    trainer = RFAttentionUNetTrainer(CONFIG)
    trainer.train(image_np, label_np, patches, label_patches)

    # Inference
    print("\nRunning inference...")
    device  = trainer.device
    pred_map, prob_map = predict(
        trainer.model, trainer.rf_branch, image_np, device,
        patch_size=CONFIG['patch_size']
    )

    # Save prediction
    out_profile = profile.copy()
    out_profile.update(count=1, dtype='uint8')
    out_path = "outputs/rf_attention_unet_prediction.tif"
    with rasterio.open(out_path, 'w', **out_profile) as dst:
        dst.write(pred_map, 1)

    print(f"\nPrediction saved -> {out_path}")
    unique, counts = np.unique(pred_map, return_counts=True)
    total = pred_map.size
    for cls, cnt in zip(unique, counts):
        names = {0: 'Low Risk', 1: 'Medium Risk', 2: 'High Risk'}
        print(f"  Class {cls} ({names.get(cls,'?')}): {cnt:,} px ({cnt/total*100:.1f}%)")
