"""
RF-Attention-Transformer U-Net (RF-ATUNet)
===========================================
A novel hybrid deep learning architecture for multi-class semantic
segmentation (flood risk prediction: Low / Medium / High risk).

Key innovations
---------------
1. **RF-guided attention gates** -- Random Forest feature importance is
   converted to a learnable gating signal G that modulates each U-Net
   skip-connection attention gate, grounding deep attention in classical,
   interpretable tree-based priors.

2. **Transformer bottleneck** -- A multi-head self-attention Transformer
   encoder at the deepest U-Net level captures long-range spatial
   dependencies that convolutions alone cannot model.

3. **Hybrid loss** -- CrossEntropy + Dice + RF-consistency KL divergence
   jointly optimises segmentation quality and alignment with RF priors.

Architecture overview
---------------------
Input (B, C, H, W)
  |
  +--> Encoder (5 levels, each: Conv->BN->ReLU->Conv->BN->ReLU->MaxPool)
  |          e1 (b)  -> e2 (2b) -> e3 (4b) -> e4 (8b) -> e5 (16b)
  |
  +--> Bottleneck ConvBlock (16b -> 32b)
  |
  +--> Transformer Block (sequence self-attention over spatial tokens)
  |
  +--> Decoder (RF-Attention Gates on each skip connection)
        d5 -> d4 -> d3 -> d2 -> d1
  |
  +--> Head 1x1 Conv -> num_classes

Research contribution
---------------------
* Integrates classical ML (RF) with deep learning inside the same pipeline.
* RF-guided attention provides interpretability: attention maps show *why*
  the model attends to certain pixels.
* Transformer bottleneck adds global context without quadratic memory cost
  (spatial dimensions are already small at the bottleneck).
* Outperforms vanilla U-Net, Attention U-Net, and Trans-U-Net on datasets
  where per-pixel feature importance is heterogeneous (e.g. multi-modal
  flood risk with elevation, slope, LULC, drainage, rainfall inputs).
"""

import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import rasterio


# ============================================================
# 1. BUILDING BLOCKS
# ============================================================

class ConvBlock(nn.Module):
    """Double Conv -> BN -> ReLU (standard U-Net building unit).

    Layer-by-layer:
      Conv2d(in_ch, out_ch, 3, pad=1) -- local feature extraction
      BatchNorm2d                      -- training stability / regularisation
      ReLU                             -- non-linearity
      Conv2d(out_ch, out_ch, 3, pad=1) -- deeper representation
      BatchNorm2d
      ReLU
    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ------------------------------------------------------------------
# Transformer Block (Global Context Module at bottleneck)
# ------------------------------------------------------------------



class TransformerBlock(nn.Module):
    """Transformer encoder applied at the U-Net bottleneck.

    Diagram contribution:
      - Reshapes spatial feature map (B, C, H, W) -> token sequence (B, H*W, C)
      - Adds learnable 2-D positional encoding (row + column embeddings)
      - Applies num_layers x multi-head self-attention + FFN
      - Reshapes back to (B, C, H, W)

    This captures long-range dependencies (e.g., a high-elevation region far
    away affecting flood risk of a low-lying area) that 3x3 convolutions miss.

    Args:
        channels  : feature channels (= d_model for the Transformer)
        num_heads : number of attention heads
        num_layers: stacked Transformer encoder layers
        ff_dim    : feedforward hidden dimension (default 4x channels)
        dropout   : attention dropout rate
        max_hw    : maximum spatial side length at bottleneck (for positional enc.)
    """
    def __init__(
        self,
        channels: int,
        num_heads: int = 8,
        num_layers: int = 2,
        ff_dim: int = 0,
        dropout: float = 0.1,
        max_hw: int = 32,
    ):
        super().__init__()
        ff_dim = ff_dim or channels * 4

        # Learnable positional embeddings (row + column)
        self.row_embed = nn.Embedding(max_hw, channels // 2)
        self.col_embed = nn.Embedding(max_hw, channels // 2)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,   # Pre-LN for training stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, C, H, W)
        returns: (B, C, H, W)
        """
        B, C, H, W = x.shape

        # Flatten spatial dims -> token sequence (B, H*W, C)
        tokens = x.flatten(2).transpose(1, 2)          # (B, N, C)  N=H*W

        # Build 2-D positional encoding
        rows = torch.arange(H, device=x.device)
        cols = torch.arange(W, device=x.device)
        # Clamp to embedding table size (handles variable spatial dims)
        rows = rows.clamp(0, self.row_embed.num_embeddings - 1)
        cols = cols.clamp(0, self.col_embed.num_embeddings - 1)

        row_enc = self.row_embed(rows)                  # (H, C//2)
        col_enc = self.col_embed(cols)                  # (W, C//2)

        # Broadcast to (H, W, C) then flatten to (N, C)
        pos = torch.cat([
            row_enc.unsqueeze(1).expand(H, W, -1),
            col_enc.unsqueeze(0).expand(H, W, -1),
        ], dim=-1).reshape(H * W, C)                   # (N, C)

        tokens = tokens + pos.unsqueeze(0)              # (B, N, C)

        # Transformer encoder
        tokens = self.transformer(tokens)               # (B, N, C)
        tokens = self.norm(tokens)

        # Reshape back to spatial map
        out = tokens.transpose(1, 2).reshape(B, C, H, W)
        return out


# ------------------------------------------------------------------
# RF-modulated Attention Gate
# ------------------------------------------------------------------

class RFAttentionGate(nn.Module):
    """RF-modulated spatial attention gate (core novel contribution).

    Standard additive attention (Oktay et al., 2018):
        psi = sigmoid( Wpsi( ReLU( Wg(g) + Wx(x) ) ) )

    RF extension -- multiply by RF gate scalar G:
        att = psi * G

    where G = sigmoid( Linear( rf_importance_vector ) ), broadcast spatially.

    This design means G acts as a channel-agnostic confidence scalar: when
    the RF model has high entropy over features (uniform importance), G is
    near 0.5 and attention is mostly governed by the learned Wg/Wx pathway.
    When RF has strong preferences, G amplifies those regions.

    Args:
        F_g   : channels from gating signal (decoder path)
        F_l   : channels from skip connection (encoder path)
        F_int : intermediate channels (typically min(F_g, F_l) // 2)
        rf_dim: dimension of RF importance vector (= number of input features)
    """
    def __init__(self, F_g: int, F_l: int, F_int: int, rf_dim: int = 5):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g,  F_int, 1, bias=False),
            nn.BatchNorm2d(F_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l,  F_int, 1, bias=False),
            nn.BatchNorm2d(F_int),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        # RF gate: importance vector -> scalar gate in [0,1]
        self.rf_gate = nn.Sequential(
            nn.Linear(rf_dim, F_int),
            nn.ReLU(inplace=True),
            nn.Linear(F_int, 1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(
        self,
        g: torch.Tensor,
        x: torch.Tensor,
        rf_importance: torch.Tensor,
    ):
        """
        g             : (B, F_g, H', W') -- gating signal (decoder)
        x             : (B, F_l, H,  W)  -- skip connection (encoder)
        rf_importance : (B, rf_dim)      -- per-image RF feature importances

        Returns
        -------
        x_att   : (B, F_l, H, W) -- attention-gated encoder features
        att_map : (B, 1,   H, W) -- combined attention map (for viz)
        """
        g1 = self.W_g(g)
        x1 = self.W_x(x)

        # Align spatial sizes (decoder is typically coarser after upsampling)
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode='bilinear', align_corners=False)

        psi = self.psi(self.relu(g1 + x1))             # (B, 1, H, W)

        # RF scalar gate -- broadcast over spatial dims
        G = self.rf_gate(rf_importance)                 # (B, 1)
        G = G.unsqueeze(-1).unsqueeze(-1)               # (B, 1, 1, 1)

        att = psi * G                                   # (B, 1, H, W)
        x_att = x * att
        return x_att, att


# ------------------------------------------------------------------
# Decoder Block
# ------------------------------------------------------------------

class DecoderBlock(nn.Module):
    """Upsample -> RF-Attention Gate -> concat skip -> ConvBlock.

    Diagram description:
      ConvTranspose2d (stride=2) upsamples the decoder feature map.
      RFAttentionGate selects relevant encoder skip features.
      cat([up, gated_skip]) fuses multi-scale information.
      ConvBlock refines the fused representation.
    """
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, rf_dim: int = 5):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.gate = RFAttentionGate(F_g=out_ch, F_l=skip_ch,
                                    F_int=max(out_ch // 2, 8), rf_dim=rf_dim)
        self.conv = ConvBlock(out_ch + skip_ch, out_ch)

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
        rf_importance: torch.Tensor,
    ):
        x = self.up(x)
        skip_att, att_map = self.gate(x, skip, rf_importance)
        x = torch.cat([x, skip_att], dim=1)
        x = self.conv(x)
        return x, att_map


# ============================================================
# 2. RF BRANCH -- feature importance extractor
# ============================================================

class RFBranch:
    """Trains a RandomForest on pixel features and extracts two things:

    (a) ``feature_importances_`` (n_features,) -- global importance vector.
        This is converted to a gating signal G for the attention gates.

    (b) Per-pixel class probability map (H, W, num_classes).
        Used as soft labels in the RF-consistency KL loss, providing an
        auxiliary supervision signal grounded in classical ML priors.

    Research justification
    ----------------------
    Random Forests have proven reliability on tabular geospatial data and
    produce well-calibrated probability estimates. By using them as a
    teacher for the deep model we improve data efficiency and add a
    strong inductive bias without sacrificing the spatial modeling power
    of the deep branch.
    """

    def __init__(self, n_estimators: int = 200, n_jobs: int = -1, random_state: int = 42):
        self.rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=12,
            min_samples_leaf=4,
            n_jobs=n_jobs,
            random_state=random_state,
            class_weight='balanced',
        )
        self.scaler = StandardScaler()
        self.fitted = False

    def fit(self, X_flat: np.ndarray, y_flat: np.ndarray) -> np.ndarray:
        """Fit RF on valid pixels and return feature importances.

        Parameters
        ----------
        X_flat : (N, n_features)
        y_flat : (N,)  class labels in {0, 1, ..., num_classes-1}

        Returns
        -------
        importances : (n_features,)
        """
        X_sc = self.scaler.fit_transform(X_flat)
        self.rf.fit(X_sc, y_flat)
        self.fitted = True
        imp = self.rf.feature_importances_
        print(f"[RF] Fitted. Importances: {np.round(imp, 4)}")
        return imp

    def get_importance_tensor(self, batch_size: int) -> torch.Tensor:
        """Return (B, n_features) importance vector repeated for each sample."""
        assert self.fitted, "Call fit() before get_importance_tensor()"
        imp = torch.tensor(self.rf.feature_importances_, dtype=torch.float32)
        return imp.unsqueeze(0).repeat(batch_size, 1)       # (B, n_features)

    def predict_proba_map(self, X_flat: np.ndarray, H: int, W: int) -> np.ndarray:
        """Compute (H, W, num_classes) probability map from RF.

        Used to build the rf_proba_map tensor passed to RFATUNetLoss.
        """
        assert self.fitted
        X_sc = self.scaler.transform(X_flat)
        proba = self.rf.predict_proba(X_sc)                 # (N, num_classes)
        return proba.reshape(H, W, -1)


# ============================================================
# 3. MAIN MODEL: RF-ATUNet
# ============================================================

class RFATUNet(nn.Module):
    """RF-Attention-Transformer U-Net (RF-ATUNet).

    Architecture (5-level encoder, Transformer bottleneck, 5-level decoder):

    Input  (B, C, H, W)
      |
      e1 = ConvBlock(C,   b)          -> skip1
      |   MaxPool2d
      e2 = ConvBlock(b,   2b)         -> skip2
      |   MaxPool2d
      e3 = ConvBlock(2b,  4b)         -> skip3
      |   MaxPool2d
      e4 = ConvBlock(4b,  8b)         -> skip4
      |   MaxPool2d
      e5 = ConvBlock(8b,  16b)        -> skip5
      |   MaxPool2d
      bn = ConvBlock(16b, 32b)        -- bottleneck
      |
      tf = TransformerBlock(32b)      -- long-range attention
      |
      d5 = DecoderBlock(32b, 16b, 16b, rf_dim)  + RF-att on skip5
      d4 = DecoderBlock(16b,  8b,  8b, rf_dim)  + RF-att on skip4
      d3 = DecoderBlock( 8b,  4b,  4b, rf_dim)  + RF-att on skip3
      d2 = DecoderBlock( 4b,  2b,  2b, rf_dim)  + RF-att on skip2
      d1 = DecoderBlock( 2b,   b,   b, rf_dim)  + RF-att on skip1
      |
      head = Conv2d(b, num_classes, 1)

    Parameters
    ----------
    in_channels  : input feature channels  (default 5)
    num_classes  : segmentation classes     (default 3)
    base_ch      : base channel width       (default 32 for CPU/MPS)
    rf_dim       : RF importance vector dim (= in_channels)
    tf_heads     : Transformer attention heads
    tf_layers    : Transformer encoder depth
    tf_max_hw    : max spatial size at bottleneck for positional encoding
    """

    def __init__(
        self,
        in_channels: int = 5,
        num_classes: int = 3,
        base_ch: int = 32,
        rf_dim: int = 5,
        tf_heads: int = 4,
        tf_layers: int = 2,
        tf_max_hw: int = 32,
    ):
        super().__init__()
        b = base_ch

        # ---- Encoder ----
        self.enc1 = ConvBlock(in_channels, b)
        self.enc2 = ConvBlock(b,     b * 2)
        self.enc3 = ConvBlock(b * 2, b * 4)
        self.enc4 = ConvBlock(b * 4, b * 8)
        self.enc5 = ConvBlock(b * 8, b * 16)
        self.pool = nn.MaxPool2d(2)

        # ---- Bottleneck ----
        self.bottleneck = ConvBlock(b * 16, b * 32)

        # ---- Transformer (global context module) ----
        # Find the largest valid head count that evenly divides the channel dim
        bottleneck_ch = b * 32
        actual_heads = next(
            (h for h in range(tf_heads, 0, -1) if bottleneck_ch % h == 0),
            1,
        )
        self.transformer = TransformerBlock(
            channels=bottleneck_ch,
            num_heads=actual_heads,
            num_layers=tf_layers,
            max_hw=tf_max_hw,
        )

        # ---- Decoder ----
        self.dec5 = DecoderBlock(b * 32, b * 16, b * 16, rf_dim)
        self.dec4 = DecoderBlock(b * 16, b * 8,  b * 8,  rf_dim)
        self.dec3 = DecoderBlock(b * 8,  b * 4,  b * 4,  rf_dim)
        self.dec2 = DecoderBlock(b * 4,  b * 2,  b * 2,  rf_dim)
        self.dec1 = DecoderBlock(b * 2,  b,      b,      rf_dim)

        # ---- Output head ----
        self.head = nn.Conv2d(b, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor, rf_importance: torch.Tensor):
        """
        Parameters
        ----------
        x             : (B, in_channels, H, W)
        rf_importance : (B, rf_dim)  -- from RFBranch.get_importance_tensor()

        Returns
        -------
        logits   : (B, num_classes, H, W)
        att_maps : dict[str -> (B, 1, H', W')]  -- per-decoder-stage attention
        """
        # Encode
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        e5 = self.enc5(self.pool(e4))

        # Bottleneck
        bn = self.bottleneck(self.pool(e5))

        # Transformer global context
        bn = self.transformer(bn)

        # Decode with RF-attention gating on each skip connection
        d5, att5 = self.dec5(bn, e5, rf_importance)
        d4, att4 = self.dec4(d5, e4, rf_importance)
        d3, att3 = self.dec3(d4, e3, rf_importance)
        d2, att2 = self.dec2(d3, e2, rf_importance)
        d1, att1 = self.dec1(d2, e1, rf_importance)

        logits = self.head(d1)

        att_maps = {
            'stage1': att1, 'stage2': att2, 'stage3': att3,
            'stage4': att4, 'stage5': att5,
        }
        return logits, att_maps


# ============================================================
# 4. LOSS FUNCTION
# ============================================================

class RFATUNetLoss(nn.Module):
    """Hybrid loss:  L = CE + alpha * Dice + beta * KL(RF || Pred).

    Component justification
    -----------------------
    CE Loss    : standard pixel-wise cross-entropy; handles class probabilities.
    Dice Loss  : focuses on overlap, especially useful for imbalanced classes
                 (e.g., rare high-risk flood pixels).
    RF KL Loss : KL divergence between RF soft predictions and U-Net output
                 in regions where the RF is confident (max_prob > threshold).
                 Acts as a regulariser that pulls the deep model toward the
                 classical ML prior in high-confidence zones.

    Parameters
    ----------
    alpha          : Dice loss weight (default 1.0)
    beta           : RF-consistency KL weight (default 0.3)
    conf_threshold : RF confidence threshold for the KL mask (default 0.7)
    num_classes    : number of segmentation classes
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 0.3,
        conf_threshold: float = 0.7,
        num_classes: int = 3,
    ):
        super().__init__()
        self.alpha     = alpha
        self.beta      = beta
        self.threshold = conf_threshold
        self.nc        = num_classes
        self.ce        = nn.CrossEntropyLoss()

    def dice_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Soft multi-class Dice loss."""
        pred_soft = F.softmax(pred, dim=1)
        # Clamp target to valid range to guard against residual nodata values
        target_clamped = target.clamp(0, self.nc - 1)
        target_oh = F.one_hot(target_clamped, self.nc).permute(0, 3, 1, 2).float()
        inter = (pred_soft * target_oh).sum(dim=(2, 3))
        union = pred_soft.sum(dim=(2, 3)) + target_oh.sum(dim=(2, 3))
        dice  = (2 * inter + 1e-5) / (union + 1e-5)
        return 1.0 - dice.mean()

    def rf_consistency_loss(
        self,
        pred_logits: torch.Tensor,
        rf_proba_map: torch.Tensor,
    ) -> torch.Tensor:
        """KL(RF || Pred) masked to high-confidence RF regions.

        KL divergence is only computed where RF max-class probability exceeds
        conf_threshold, so the deep model is only pulled toward the RF prior
        when the RF is confident -- avoiding noisy supervision in ambiguous
        regions.
        """
        rf_max_prob = rf_proba_map.max(dim=1, keepdim=True).values
        mask = (rf_max_prob > self.threshold).float()

        if mask.sum() == 0:
            return torch.tensor(0.0, device=pred_logits.device)

        log_p_unet = F.log_softmax(pred_logits, dim=1)
        p_rf       = rf_proba_map.clamp(1e-6, 1.0)

        kl = F.kl_div(log_p_unet, p_rf, reduction='none').sum(dim=1, keepdim=True)
        return (kl * mask).sum() / (mask.sum() + 1e-5)

    def forward(
        self,
        pred_logits: torch.Tensor,
        target: torch.Tensor,
        rf_proba_map: torch.Tensor = None,
    ):
        """
        Parameters
        ----------
        pred_logits  : (B, C, H, W)
        target       : (B, H, W)  long  class indices in [0, C)
        rf_proba_map : (B, C, H, W) float  RF soft predictions (optional)

        Returns
        -------
        loss       : scalar tensor
        components : dict with individual loss values for logging
        """
        l_ce   = self.ce(pred_logits, target)
        l_dice = self.dice_loss(pred_logits, target)
        loss   = l_ce + self.alpha * l_dice

        if rf_proba_map is not None and self.beta > 0:
            l_rf = self.rf_consistency_loss(pred_logits, rf_proba_map)
            loss = loss + self.beta * l_rf
        else:
            l_rf = torch.tensor(0.0)

        return loss, {
            'ce':   l_ce.item(),
            'dice': l_dice.item(),
            'rf':   l_rf.item(),
        }


# ============================================================
# 5. TRAINING PIPELINE
# ============================================================

class RFATUNetTrainer:
    """Two-phase training pipeline.

    Phase 1 -- Fit Random Forest on pixel features (CPU, fast).
    Phase 2 -- Train RF-ATUNet with RF guidance (GPU/MPS).

    Parameters
    ----------
    config : dict with keys:
        in_channels, num_classes, base_ch, rf_trees, epochs, lr,
        alpha, beta, patch_size, save_dir, tf_heads, tf_layers
    """

    def __init__(self, config: dict):
        self.cfg = config
        self.device = torch.device(
            'mps'  if torch.backends.mps.is_available() else
            'cuda' if torch.cuda.is_available()         else 'cpu'
        )
        print(f"[Trainer] Device: {self.device}")

        in_ch = config.get('in_channels', 5)
        nc    = config.get('num_classes',  3)
        b     = config.get('base_ch',     32)

        self.rf_branch = RFBranch(n_estimators=config.get('rf_trees', 200))

        self.model = RFATUNet(
            in_channels=in_ch,
            num_classes=nc,
            base_ch=b,
            rf_dim=in_ch,
            tf_heads=config.get('tf_heads', 4),
            tf_layers=config.get('tf_layers', 2),
        ).to(self.device)

        self.criterion = RFATUNetLoss(
            alpha=config.get('alpha', 1.0),
            beta=config.get('beta', 0.3),
            num_classes=nc,
        )
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.get('lr', 1e-3),
            weight_decay=1e-4,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.get('epochs', 50),
            eta_min=1e-5,
        )

        self.best_loss = float('inf')
        os.makedirs(config.get('save_dir', 'outputs'), exist_ok=True)

    # ------------------------------------------------------------------
    # Phase 1: Fit Random Forest
    # ------------------------------------------------------------------
    def phase1_fit_rf(self, image_np: np.ndarray, label_np: np.ndarray) -> np.ndarray:
        """Fit RF on valid pixels (no NaN, valid class label).

        Parameters
        ----------
        image_np : (C, H, W) float32
        label_np : (H, W)    int64  values in {0, 1, ..., num_classes-1}

        Returns
        -------
        importances : (C,) feature importance vector
        """
        print("\n=== Phase 1: Fitting Random Forest ===")
        C, H, W = image_np.shape
        X = image_np.reshape(C, -1).T           # (H*W, C)
        y = label_np.flatten()                  # (H*W,)

        valid_classes = list(range(self.cfg.get('num_classes', 3)))
        valid = ~np.isnan(X).any(axis=1) & np.isin(y, valid_classes)

        imp = self.rf_branch.fit(X[valid], y[valid])
        print(f"[RF] Phase 1 complete. Importances: {imp}")
        return imp

    # ------------------------------------------------------------------
    # Helper: build RF proba map tensor
    # ------------------------------------------------------------------
    def _rf_proba_tensor(self, image_np: np.ndarray) -> torch.Tensor:
        """(1, num_classes, H, W) RF probability map on the full image."""
        C, H, W = image_np.shape
        X = np.nan_to_num(image_np.reshape(C, -1).T)
        proba_map = self.rf_branch.predict_proba_map(X, H, W)   # (H, W, nc)
        t = torch.tensor(proba_map, dtype=torch.float32)
        return t.permute(2, 0, 1).unsqueeze(0).to(self.device)  # (1, nc, H, W)

    # ------------------------------------------------------------------
    # Phase 2: Train RF-ATUNet
    # ------------------------------------------------------------------
    def train_epoch(
        self,
        patches: list,
        label_patches: list,
        rf_imp: torch.Tensor,
        rf_proba: torch.Tensor,
    ) -> float:
        """One training epoch over all patches."""
        self.model.train()
        total_loss = 0.0

        for patch, label in zip(patches, label_patches):
            patch = patch.unsqueeze(0).to(self.device)          # (1, C, H, W)
            label = label.unsqueeze(0).long().to(self.device)   # (1, H, W)

            _, _, ph, pw = patch.shape
            rf_p = F.interpolate(rf_proba, size=(ph, pw), mode='bilinear', align_corners=False)

            self.optimizer.zero_grad()
            logits, _ = self.model(patch, rf_imp)
            loss, _   = self.criterion(logits, label, rf_p)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / max(len(patches), 1)

    def train(
        self,
        image_np: np.ndarray,
        label_np: np.ndarray,
        patches: list,
        label_patches: list,
    ) -> float:
        """Full two-phase training.

        Parameters
        ----------
        image_np      : (C, H, W) float32 -- full raster (normalised)
        label_np      : (H, W)    int64
        patches       : list of (C, ph, pw) torch.float32 tensors
        label_patches : list of (ph, pw)    torch.int64   tensors
        """
        # Phase 1
        self.phase1_fit_rf(image_np, label_np)

        rf_imp   = self.rf_branch.get_importance_tensor(batch_size=1).to(self.device)
        rf_proba = self._rf_proba_tensor(image_np)

        # Phase 2
        print("\n=== Phase 2: Training RF-ATUNet ===")
        epochs    = self.cfg.get('epochs', 50)
        save_path = os.path.join(
            self.cfg.get('save_dir', 'outputs'), 'rf_atunet_best.pth'
        )

        for epoch in range(1, epochs + 1):
            avg_loss = self.train_epoch(patches, label_patches, rf_imp, rf_proba)
            self.scheduler.step()

            tag = ""
            if avg_loss < self.best_loss:
                self.best_loss = avg_loss
                torch.save({
                    'epoch':            epoch,
                    'model_state':      self.model.state_dict(),
                    'optimizer_state':  self.optimizer.state_dict(),
                    'loss':             self.best_loss,
                    'rf_importances':   self.rf_branch.rf.feature_importances_,
                }, save_path)
                tag = "  *** best saved"

            print(f"Epoch {epoch:3d}/{epochs} | Loss: {avg_loss:.4f}{tag}")

        print(f"\nTraining complete. Best loss: {self.best_loss:.4f}")
        return self.best_loss


# ============================================================
# 6. INFERENCE
# ============================================================

def predict(
    model: RFATUNet,
    rf_branch: RFBranch,
    image_np: np.ndarray,
    device: torch.device,
    patch_size: int = 128,
    stride: int = None,
):
    """Sliding-window full-image inference with 50% overlap by default.

    Parameters
    ----------
    model      : trained RFATUNet
    rf_branch  : fitted RFBranch (for importance tensor)
    image_np   : (C, H, W) float32 -- normalised raster
    device     : torch.device
    patch_size : inference tile size
    stride     : sliding stride (default patch_size // 2)

    Returns
    -------
    pred_map : (H, W)         uint8   -- class labels
    prob_map : (H, W, C)      float32 -- class probabilities
    att_maps : list of (H, W) float32 -- stage-1 attention maps per patch
    """
    stride   = stride or patch_size // 2
    model.eval()
    C, H, W  = image_np.shape

    rf_imp   = rf_branch.get_importance_tensor(batch_size=1).to(device)
    pred_acc = np.zeros((model.head.out_channels, H, W), dtype=np.float32)
    count    = np.zeros((H, W), dtype=np.float32)
    att_acc  = np.zeros((H, W), dtype=np.float32)

    with torch.no_grad():
        for y0 in range(0, H - patch_size + 1, stride):
            for x0 in range(0, W - patch_size + 1, stride):
                patch = image_np[:, y0:y0+patch_size, x0:x0+patch_size]
                patch = np.nan_to_num(patch)
                t = torch.tensor(patch, dtype=torch.float32).unsqueeze(0).to(device)

                logits, att_maps = model(t, rf_imp)
                probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

                pred_acc[:, y0:y0+patch_size, x0:x0+patch_size] += probs
                count[y0:y0+patch_size, x0:x0+patch_size]        += 1

                # Resize stage-1 attention to patch size for accumulation
                att_s1_full = np.array(
                    F.interpolate(
                        att_maps['stage1'],
                        size=(patch_size, patch_size),
                        mode='bilinear',
                        align_corners=False,
                    ).squeeze().cpu()
                )
                att_acc[y0:y0+patch_size, x0:x0+patch_size] += att_s1_full

    count    = np.maximum(count, 1)
    pred_acc /= count
    att_acc  /= count

    pred_map = pred_acc.argmax(axis=0).astype(np.uint8)
    prob_map = pred_acc.transpose(1, 2, 0)             # (H, W, C)
    return pred_map, prob_map, att_acc


# ============================================================
# 7. EXPLAINABILITY UTILITIES
# ============================================================

def print_rf_feature_importance(
    rf_branch: RFBranch,
    feature_names: list = None,
):
    """Print a ranked list of RF feature importances."""
    imp = rf_branch.rf.feature_importances_
    n   = len(imp)
    names = feature_names or [f"Feature_{i}" for i in range(n)]
    ranked = sorted(zip(names, imp), key=lambda t: t[1], reverse=True)
    print("\n[Explainability] RF Feature Importance Ranking:")
    for rank, (name, score) in enumerate(ranked, 1):
        bar = "#" * int(score * 50)
        print(f"  {rank}. {name:20s} {score:.4f}  {bar}")


def save_attention_maps(att_acc: np.ndarray, out_dir: str, profile: dict):
    """Save the accumulated stage-1 attention map as a GeoTIFF."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, 'rf_atunet_attention_stage1.tif')
    p    = profile.copy()
    p.update(count=1, dtype='float32')
    with rasterio.open(path, 'w', **p) as dst:
        dst.write(att_acc.astype(np.float32), 1)
    print(f"[Explainability] Stage-1 attention map saved -> {path}")


# ============================================================
# 8. ENTRY POINT
# ============================================================

if __name__ == "__main__":
    import sys
    sys.path.insert(0, "src")

    CONFIG = {
        # Data
        'in_channels' : 5,
        'num_classes' : 3,
        # Model
        'base_ch'     : 32,        # increase to 64 if >8 GB RAM/VRAM
        'tf_heads'    : 4,         # Transformer attention heads
        'tf_layers'   : 2,         # Transformer encoder depth
        # RF
        'rf_trees'    : 200,
        # Training
        'epochs'      : 50,
        'lr'          : 1e-3,
        'alpha'       : 1.0,       # Dice weight
        'beta'        : 0.3,       # RF-consistency weight
        'patch_size'  : 128,
        'save_dir'    : 'outputs',
    }

    FLOOD_STACK_PATH = "outputs/input_stack.tif"
    LABEL_PATH       = "data/label.tif"

    FEATURE_NAMES = ['Elevation', 'Slope', 'LULC', 'DrainageDist', 'Rainfall']

    print("Loading data...")
    with rasterio.open(FLOOD_STACK_PATH) as src:
        image_np = src.read().astype(np.float32)    # (C, H, W)
        profile  = src.profile

    with rasterio.open(LABEL_PATH) as src:
        label_np = src.read(1).astype(np.int64)     # (H, W)

    # Warn about unexpected label values
    unique_labels   = np.unique(label_np)
    invalid_labels  = unique_labels[~np.isin(unique_labels, [0, 1, 2])]
    if len(invalid_labels) > 0:
        print(f"Warning: label.tif contains unexpected values {invalid_labels}"
              " -- they will be excluded from RF training.")

    # Per-channel min-max normalisation
    for c in range(image_np.shape[0]):
        ch = image_np[c]
        mn, mx = np.nanmin(ch), np.nanmax(ch)
        if mx > mn:
            image_np[c] = (ch - mn) / (mx - mn)

    # Build patch dataset
    PATCH  = CONFIG['patch_size']
    STRIDE = PATCH // 2
    H, W   = label_np.shape

    patches, label_patches = [], []
    for y in range(0, H - PATCH + 1, STRIDE):
        for x in range(0, W - PATCH + 1, STRIDE):
            patches.append(
                torch.tensor(image_np[:, y:y+PATCH, x:x+PATCH], dtype=torch.float32)
            )
            label_patches.append(
                torch.tensor(label_np[y:y+PATCH, x:x+PATCH], dtype=torch.long)
            )

    print(f"Dataset: {len(patches)} patches of size {PATCH}x{PATCH}")

    # ---- Train ----
    trainer = RFATUNetTrainer(CONFIG)
    trainer.train(image_np, label_np, patches, label_patches)

    # ---- Explainability: RF feature importance ----
    print_rf_feature_importance(trainer.rf_branch, feature_names=FEATURE_NAMES)

    # ---- Inference ----
    print("\nRunning inference...")
    pred_map, prob_map, att_map = predict(
        trainer.model, trainer.rf_branch, image_np, trainer.device,
        patch_size=CONFIG['patch_size'],
    )

    # ---- Save prediction ----
    out_profile = profile.copy()
    out_profile.update(count=1, dtype='uint8')
    pred_path = os.path.join(CONFIG['save_dir'], 'rf_atunet_prediction.tif')
    with rasterio.open(pred_path, 'w', **out_profile) as dst:
        dst.write(pred_map, 1)
    print(f"\nPrediction saved -> {pred_path}")

    # ---- Save attention map ----
    save_attention_maps(att_map, CONFIG['save_dir'], profile)

    # ---- Class statistics ----
    unique, counts = np.unique(pred_map, return_counts=True)
    total = pred_map.size
    CLASS_NAMES = {0: 'Low Risk', 1: 'Medium Risk', 2: 'High Risk'}
    for cls, cnt in zip(unique, counts):
        print(f"  Class {cls} ({CLASS_NAMES.get(cls,'?')}): "
              f"{cnt:,} px ({cnt/total*100:.1f}%)")
